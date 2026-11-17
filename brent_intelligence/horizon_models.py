"""Horizon-specific complete recurrent models for Brent, with honest temporal validation.

Each h=5/63/252 has an independent network and learns from EVERY h-label
matured by the relevant cutoff. Architectural selection is done only on
outer folds ending in 2023; inner chronology chooses epochs and shrinkage.
Final models refit through the most recent observation and stay isolated
from the existing frozen/production weights.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path
import csv
import json
import math
import os
import random
import statistics

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score

from brent_intelligence.config import ARTIFACTS_DIR, DATA_PATH, REPORTS_DIR
from brent_intelligence.delivery_training import (
    HORIZONS, KINDS, SELECTION_FOLDS, build_examples, FullExamples,
    _cutoff_index,
)
from brent_intelligence.features import load_prices, make_features

SHRINK_GRID = (0., .25, .5, .75, 1.)
HOLDOUT_CUTOFF = "2023-12-29"


@dataclass(frozen=True)
class HorizonConfig:
    window: int = 60
    hidden_size: int = 16
    dropout: float = .15
    weight_decay: float = .01
    prediction_penalty: float = .025
    learning_rate: float = .001
    batch_size: int = 128
    epochs: int = 18
    patience: int = 4
    min_feature_coverage: float = .60
    min_inner_gain: float = .01
    seed: int = 42
    architectures: tuple[str, ...] = ("rnn", "lstm", "gru")
    horizons: tuple[int, ...] = HORIZONS
    folds: int = 3

    def validate(self) -> None:
        if (not self.architectures or len(set(self.architectures)) != len(self.architectures)
            or any(x not in KINDS for x in self.architectures)
            or not self.horizons or len(set(self.horizons)) != len(self.horizons)
            or any(h not in HORIZONS for h in self.horizons)
            or self.window < 2 or self.hidden_size < 1 or self.batch_size < 1
            or self.epochs < 1 or self.patience < 1
            or not (1 <= self.folds <= len(SELECTION_FOLDS))
            or not (0 <= self.dropout < 1)
            or not (0 < self.learning_rate)
            or self.weight_decay < 0 or self.prediction_penalty < 0
            or not (0 < self.min_feature_coverage <= 1)
            or not (0 <= self.min_inner_gain < 1)):
            raise ValueError("Configuracion de horizonte invalida")


@dataclass
class HorizonScaler:
    columns: list[str]
    medians: list[float]
    mean: list[float]
    std: list[float]
    target_rms: float

    @classmethod
    def fit(cls, ex: FullExamples, ids: np.ndarray,
            horizon: int, cutoff: str, coverage: float) -> "HorizonScaler":
        end = _cutoff_index(ex, cutoff)
        j = HORIZONS.index(horizon)
        if (not len(ids) or (ex.target_indices[ids, j] > end).any()
            or (ex.anchors[ids] > end).any()):
            raise ValueError("Fuga temporal o entrenamiento vacio")
        known = ex.features.iloc[:end+1]
        columns = known.columns[known.notna().mean() >= coverage].tolist()
        if not columns:
            raise ValueError("Ninguna caracteristica con cobertura")
        median = known[columns].median().fillna(0.)
        matrix = known[columns].fillna(median).to_numpy(dtype=float)
        mean, std = matrix.mean(axis=0), matrix.std(axis=0)
        std = np.where(std < 1e-6, 1., std)
        target = ex.log_returns[ids, j]
        rms = float(max(np.sqrt(np.mean(target ** 2)), 1e-4))
        if not np.isfinite(rms):
            raise ValueError("Objetivo no finito")
        return cls(columns, median.tolist(), mean.tolist(), std.tolist(), rms)

    def transform(self, features: pd.DataFrame) -> np.ndarray:
        values = features.reindex(columns=self.columns).to_numpy(dtype=float)
        med = np.asarray(self.medians, dtype=float)
        values = np.where(np.isfinite(values), values, med[None, :])
        result = (values - np.asarray(self.mean)) / np.asarray(self.std)
        return np.clip(result, -8, 8).astype(np.float32)


class DirectRecurrent(nn.Module):
    """One horizon and one output: a learnable log-return correction to naive."""

    def __init__(self, architecture: str, n_features: int, cfg: HorizonConfig):
        super().__init__()
        if architecture not in KINDS:
            raise ValueError("Arquitectura recurrente invalida")
        kind = {"rnn": nn.RNN, "lstm": nn.LSTM, "gru": nn.GRU}[architecture]
        self.dropout = nn.Dropout(cfg.dropout)
        self.recurrent = kind(n_features, cfg.hidden_size, batch_first=True)
        self.head = nn.Linear(cfg.hidden_size, 1)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        sequence, _ = self.recurrent(self.dropout(x))
        return self.head(sequence[:, -1, :]).squeeze(-1)


def _seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def ids_for_cut(ex: FullExamples, horizon: int, cutoff: str) -> np.ndarray:
    j = HORIZONS.index(horizon)
    end = _cutoff_index(ex, cutoff)
    return np.flatnonzero(
        (ex.anchors <= end) & (ex.target_indices[:, j] <= end)
    )


def outer_split(ex: FullExamples, horizon: int, cutoff: str, end: str):
    j = HORIZONS.index(horizon)
    left = _cutoff_index(ex, cutoff)
    right = _cutoff_index(ex, end)
    train = ids_for_cut(ex, horizon, cutoff)
    validation = np.flatnonzero(
        (ex.anchors > left) & (ex.anchors <= right)
        & (ex.target_indices[:, j] <= right)
    )
    if len(train) < 100 or len(validation) < 20:
        raise ValueError(f"Muestras maduras insuficientes en h={horizon}")
    return train, validation


def inner_split(ex: FullExamples, horizon: int, cutoff: str):
    inner_cutoff = (pd.Timestamp(cutoff) -
                    pd.DateOffset(years=3)).date().isoformat()
    train, valid = outer_split(ex, horizon, inner_cutoff, cutoff)
    return inner_cutoff, train, valid


def _arrays(ex, ids, scaler):
    return ex.windows(scaler.transform(ex.features), ids)


def _train_epoch(model, loader, optimizer, cfg):
    model.train()
    data_sum, objective_sum, seen = 0., 0., 0
    for x, y in loader:
        optimizer.zero_grad(set_to_none=True)
        pred = model(x)
        data_loss = nn.functional.smooth_l1_loss(pred, y)
        objective = data_loss + cfg.prediction_penalty * pred.square().mean()
        objective.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.)
        optimizer.step()
        data_sum += float(data_loss.item()) * len(x)
        objective_sum += float(objective.item()) * len(x)
        seen += len(x)
    return data_sum / seen, objective_sum / seen


@torch.no_grad()
def _infer(model, x: np.ndarray) -> np.ndarray:
    model.eval()
    return np.concatenate([
        model(torch.from_numpy(x[i:i+512])).cpu().numpy()
        for i in range(0, len(x), 512)
    ])


def _optimizer(model, cfg):
    return torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate,
                             weight_decay=cfg.weight_decay)


def _loader(x, y, cfg):
    return DataLoader(
        TensorDataset(torch.from_numpy(x), torch.from_numpy(y)),
        batch_size=cfg.batch_size, shuffle=True,
        generator=torch.Generator().manual_seed(cfg.seed),
    )


def observed_prices(ex: FullExamples, ids: np.ndarray, horizon: int):
    j = HORIZONS.index(horizon)
    if (not len(ids) or not ex.observed[ids, j].all()
        or (ex.target_indices[ids, j] >= len(ex.prices)).any()):
        raise ValueError("Intento evaluar precios futuros no observados")
    anchors = ex.prices[ex.anchors[ids]]
    actual = ex.prices[ex.target_indices[ids, j]]
    return anchors, actual


def return_to_price(anchor, normalized_prediction, scale, alpha):
    log_return = np.asarray(normalized_prediction) * scale * alpha
    return np.asarray(anchor) * np.exp(np.clip(log_return, -4, 4))


def score_prices(anchor, actual, predicted):
    a = np.asarray(actual, dtype=float)
    p = np.asarray(predicted, dtype=float)
    b = np.asarray(anchor, dtype=float)
    if (not len(a) or a.shape != p.shape or a.shape != b.shape
        or not np.isfinite(a).all() or not np.isfinite(p).all()):
        raise ValueError("Predicciones o reales no validos")
    rmse = math.sqrt(float(np.mean((p-a)**2)))
    naive = math.sqrt(float(np.mean((b-a)**2)))
    if naive <= 1e-10:
        raise ValueError("Basal ingenuo degenerado")
    denom = float(np.sum((a-a.mean())**2))
    metric = {
        "n": len(a),
        "mae": float(np.mean(np.abs(p-a))),
        "mse": float(np.mean((p-a)**2)),
        "rmse": rmse,
        "naive_rmse": naive,
        "rmse_ratio": rmse/naive,
        "improvement_pct": 100*(1-rmse/naive),
        "r2": float(1-np.sum((p-a)**2)/denom) if denom > 1e-12 else None,
        "naive_r2": float(1-np.sum((b-a)**2)/denom) if denom > 1e-12 else None,
    }
    valid = a != b
    positives = a[valid] > b[valid]
    if len(np.unique(positives)) == 2:
        score = np.log(p[valid]/b[valid])
        decision = score > 0
        metric["direction"] = {
            "roc_auc": float(roc_auc_score(positives, score)),
            "precision": float(precision_score(positives, decision, zero_division=0)),
            "recall": float(recall_score(positives, decision, zero_division=0)),
            "f1": float(f1_score(positives, decision, zero_division=0)),
            "accuracy": float(np.mean(decision == positives)),
        }
    else:
        metric["direction"] = None
    return metric


def fit_inner(ex: FullExamples, architecture: str,
              horizon: int, cutoff: str, cfg: HorizonConfig):
    inner_cut, tr, va = inner_split(ex, horizon, cutoff)
    scaler = HorizonScaler.fit(ex, tr, horizon, inner_cut,
                               cfg.min_feature_coverage)
    xt, xv = _arrays(ex, tr, scaler), _arrays(ex, va, scaler)
    j = HORIZONS.index(horizon)
    yt = (ex.log_returns[tr, j]/scaler.target_rms).astype(np.float32)
    yv = (ex.log_returns[va, j]/scaler.target_rms).astype(np.float32)
    anchor, observed = observed_prices(ex, va, horizon)
    baseline = math.sqrt(float(np.mean((anchor-observed)**2)))
    if baseline <= 1e-10:
        raise ValueError("No se puede calcular RMSE relativo interno")
    _seed(cfg.seed)
    model = DirectRecurrent(architecture, len(scaler.columns), cfg)
    optimizer = _optimizer(model, cfg)
    loader = _loader(xt, yt, cfg)
    history = []
    best_score, best_epoch, best_out, wait = math.inf, 0, None, 0
    for epoch in range(1, cfg.epochs + 1):
        train_data, train_objective = _train_epoch(
            model, loader, optimizer, cfg
        )
        preds = _infer(model, xv)
        val_loss = float(nn.functional.smooth_l1_loss(
            torch.from_numpy(preds), torch.from_numpy(yv)
        ).item())
        expected = return_to_price(anchor, preds, scaler.target_rms, 1.)
        rmse = math.sqrt(float(np.mean((expected-observed)**2)))
        val_ratio = rmse/baseline
        if not np.isfinite(val_ratio):
            raise ValueError("Predicciones internas no finitas")
        history.append({
            "epoch": epoch,
            "train_smooth_l1": train_data,
            "train_objective": train_objective,
            "val_smooth_l1": val_loss,
            "val_rmse_relative_unshrunk": val_ratio,
        })
        if val_ratio < best_score-1e-4:
            best_score, best_epoch, best_out, wait = (
                val_ratio, epoch, preds.copy(), 0
            )
        else:
            wait += 1
        if wait >= cfg.patience:
            break
    if best_out is None:
        raise ValueError("No hay checkpoint interno")
    candidate_rmse = {}
    for alpha in SHRINK_GRID:
        predicted = return_to_price(anchor, best_out,
                                    scaler.target_rms, alpha)
        candidate_rmse[str(alpha)] = math.sqrt(float(np.mean(
            (predicted-observed)**2
        )))
    best_alpha = min(SHRINK_GRID, key=lambda a: candidate_rmse[str(a)])
    if candidate_rmse[str(best_alpha)] >= baseline*(1-cfg.min_inner_gain):
        best_alpha = 0.
    return {
        "epoch": best_epoch, "alpha": float(best_alpha),
        "history": history,
        "inner_cutoff": inner_cut,
        "n_inner_train": int(len(tr)),
        "n_inner_valid": int(len(va)),
        "unshrunk_inner_rmse_ratio": best_score,
        "inner_naive_rmse": baseline,
        "inner_alpha_rmse": candidate_rmse,
    }


def fit_full(ex: FullExamples, architecture: str, horizon: int,
             cutoff: str, cfg: HorizonConfig, epochs: int):
    train = ids_for_cut(ex, horizon, cutoff)
    scaler = HorizonScaler.fit(ex, train, horizon, cutoff,
                               cfg.min_feature_coverage)
    x = _arrays(ex, train, scaler)
    j = HORIZONS.index(horizon)
    y = (ex.log_returns[train, j]/scaler.target_rms).astype(np.float32)
    _seed(cfg.seed)
    model = DirectRecurrent(architecture, len(scaler.columns), cfg)
    optimizer = _optimizer(model, cfg)
    loader = _loader(x, y, cfg)
    curve = []
    for epoch in range(1, epochs+1):
        data_loss, objective = _train_epoch(model, loader, optimizer, cfg)
        curve.append({"epoch": epoch, "train_smooth_l1": data_loss,
                      "train_objective": objective})
    return model, scaler, {
        "n_train": len(train),
        "last_matured_training_origin": str(
            ex.dates[ex.anchors[train[-1]]].date()),
        "last_training_label_date": str(ex.dates[
            ex.target_indices[train[-1], j]].date()),
        "epochs": epochs,
        "training_curve": curve,
    }


def evaluate_period(ex: FullExamples, architecture: str, horizon: int,
                    cutoff: str, period_end: str, cfg: HorizonConfig):
    selected = fit_inner(ex, architecture, horizon, cutoff, cfg)
    model, scaler, fit_meta = fit_full(ex, architecture, horizon,
                                       cutoff, cfg, selected["epoch"])
    _, valid = outer_split(ex, horizon, cutoff, period_end)
    anchor, observed = observed_prices(ex, valid, horizon)
    scaled = _infer(model, _arrays(ex, valid, scaler))
    forecast = return_to_price(
        anchor, scaled, scaler.target_rms, selected["alpha"]
    )
    unshrunk = return_to_price(anchor, scaled, scaler.target_rms, 1.)
    metrics = score_prices(anchor, observed, forecast)
    raw_metrics = score_prices(anchor, observed, unshrunk)
    rows = []
    for i, id_ in enumerate(valid):
        j = HORIZONS.index(horizon)
        rows.append({
            "stage": "validation",
            "architecture": architecture,
            "horizon": horizon,
            "origin_date": str(ex.dates[ex.anchors[id_]].date()),
            "target_date": str(ex.dates[ex.target_indices[id_,j]].date()),
            "anchor": float(anchor[i]),
            "predicted": float(forecast[i]),
            "predicted_unshrunk": float(unshrunk[i]),
            "actual": float(observed[i]),
        })
    return {
        "architecture": architecture, "horizon": horizon,
        "cutoff": cutoff, "period_end": period_end,
        "inner": selected, "train": fit_meta,
        "n_features": len(scaler.columns),
        "n_parameters": sum(p.numel() for p in model.parameters()),
        "metrics": metrics, "unshrunk_metrics": raw_metrics,
    }, rows


def _write_csv(path, records):
    if not records:
        raise ValueError("No existen registros")
    with Path(path).open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)


def _charts(folder: Path, scores: list[dict], historical_rows: list[dict]):
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    combos = [(a,h) for a in KINDS for h in HORIZONS
              if any(row["architecture"] == a and row["horizon"] == h
                     for row in scores)]
    fig = make_subplots(rows=len(combos), cols=1,
        subplot_titles=[f"{a.upper()} · {h} sesiones · validacion interna"
                        for a,h in combos])
    for i, (arch,h) in enumerate(combos,1):
        chosen = [row for row in scores if row["architecture"] == arch
                  and row["horizon"] == h]
        for fold_index, row in enumerate(chosen):
            hist = row["inner"]["history"]
            fig.add_trace(go.Scatter(
                x=[x["epoch"] for x in hist],
                y=[x["train_smooth_l1"] for x in hist],
                mode="lines+markers",
                name=f"Train fold {fold_index+1}",
                showlegend=(i == 1)), row=i, col=1)
            fig.add_trace(go.Scatter(
                x=[x["epoch"] for x in hist],
                y=[x["val_smooth_l1"] for x in hist],
                mode="lines+markers", line={"dash":"dot"},
                name=f"Val fold {fold_index+1}",
                showlegend=(i == 1)), row=i, col=1)
    fig.update_layout(height=max(650,240*len(combos)),
                      title="Train/validation SmoothL1 · experimentos separados por horizonte")
    fig.write_html(str(folder/"learning_curves.html"), include_plotlyjs=True)
    if not historical_rows:
        return
    comparison = make_subplots(rows=len(HORIZONS),cols=1,
        subplot_titles=[f"Diagnostico 2024+ · {h} sesiones (NO test virgen)"
                        for h in HORIZONS])
    for j,h in enumerate(HORIZONS,1):
        subset = sorted([r for r in historical_rows if r["horizon"]==h],
                        key=lambda x:x["target_date"])
        if not subset:
            continue
        comparison.add_trace(go.Scatter(
            x=[x["target_date"] for x in subset],
            y=[x["actual"] for x in subset],
            name="Real", legendgroup="real",
            showlegend=j==1),row=j,col=1)
        comparison.add_trace(go.Scatter(
            x=[x["target_date"] for x in subset],
            y=[x["predicted"] for x in subset],
            name="Predicho", legendgroup="pred",
            showlegend=j==1),row=j,col=1)
        comparison.add_trace(go.Scatter(
            x=[x["target_date"] for x in subset],
            y=[x["anchor"] for x in subset],
            name="Ingenuo", legendgroup="naive",
            line={"dash":"dot"}, showlegend=j==1),row=j,col=1)
    comparison.update_layout(height=1050,
        title="Predicho vs real en fecha objetivo: modelos pre-2024, tres horizontes")
    comparison.write_html(str(folder/"real_vs_predicted.html"), include_plotlyjs=True)


def run_horizon_models(*, data_path: Path=DATA_PATH,
                      report_dir: Path=REPORTS_DIR/"horizon_v5",
                      model_dir: Path=ARTIFACTS_DIR/"horizon_candidates",
                      cfg: HorizonConfig|None=None,
                      dry_run: bool=False, quick: bool=False,
                      allow_stale: bool=False) -> dict:
    cfg = cfg or HorizonConfig()
    cfg.validate()
    torch.set_num_threads(min(4,os.cpu_count() or 1))
    source=Path(data_path)
    if not source.exists():
        raise FileNotFoundError(f"CSV no disponible: {source}")
    raw = load_prices(source)
    if raw.index[0] > pd.Timestamp("2000-03-01"):
        raise ValueError("El dataset no cubre enero de 2000")
    stale=(datetime.now(timezone.utc).date()-raw.index[-1].date()).days
    if stale > 10 and not allow_stale:
        raise ValueError("CSV desactualizado; refrescar antes de entrenar")
    if stale < -1:
        raise ValueError("CSV contiene cotizaciones futuras")
    ex = build_examples(raw,cfg.window)
    cutoffs = SELECTION_FOLDS[:cfg.folds]
    plan=[]
    for h in cfg.horizons:
        for cut,end in cutoffs:
            tr,va=outer_split(ex,h,cut,end)
            innercut,it,iv=inner_split(ex,h,cut)
            plan.append({"horizon":h,"train_until":cut,"validation_until":end,
                         "n_train":len(tr),"n_validation":len(va),
                         "inner_until":innercut,"n_inner_train":len(it),
                         "n_inner_validation":len(iv)})
    known={str(h):len(ids_for_cut(ex,h,str(raw.index[-1].date())))
           for h in cfg.horizons}
    provenance={
        "start":str(raw.index[0].date()),"end":str(raw.index[-1].date()),
        "n_price_observations":len(raw),"data_sha256":sha256(source.read_bytes()).hexdigest(),
        "n_features_available":len(ex.features.columns),
        "stale_days":stale,
        "warnings":[
            "Externas no verificadas point-in-time (lag de un dia no garantiza disponibilidad).",
            "El periodo 2024+ ya se ha analizado; su diagnostico no es un test virgen.",
            "Los horizontes solapados implican dependencia temporal.",
            "La red nueva no sustituye automaticamente la produccion."
        ],
    }
    if dry_run:
        return {"mode":"dry-run","config":asdict(cfg),"provenance":provenance,
                "folds":plan,"matured_latest":known,
                "trained":False}
    folds, validation_rows=[],[]
    for h in cfg.horizons:
        for arch in cfg.architectures:
            for cut,end in cutoffs:
                result,rows=evaluate_period(ex,arch,h,cut,end,cfg)
                folds.append(result)
                validation_rows.extend(rows)
                print(f"{arch.upper():4s} h{h:3d} {cut}..{end}: "
                      f"RMSE/naive {result['metrics']['rmse_ratio']:.4f}, "
                      f"sin contraer {result['unshrunk_metrics']['rmse_ratio']:.4f}",
                      flush=True)
    ranking={}
    for h in cfg.horizons:
        candidates={}
        for arch in cfg.architectures:
            records=[x for x in folds if x["horizon"]==h and
                     x["architecture"]==arch]
            ratios=[x["metrics"]["rmse_ratio"] for x in records]
            candidates[arch]={
                "mean_fold_rmse_ratio":float(statistics.mean(ratios)),
                "won_folds":sum(x<1 for x in ratios),
                "folds":len(records),
                "rmse_ratio_by_fold":ratios,
            }
        winner=min(cfg.architectures,
                   key=lambda a:candidates[a]["mean_fold_rmse_ratio"])
        ranking[str(h)]={
            "chosen_architecture":winner,"candidates":candidates,
            "selection_only_before_2024":True
        }
    # Descriptive 2024+ heldout evaluation after architecture selection:
    # the public research has examined these years; never retune on them.
    diagnosis, diagnosis_rows={},[]
    for h in cfg.horizons:
        arch=ranking[str(h)]["chosen_architecture"]
        evaluation,rows=evaluate_period(
            ex,arch,h,HOLDOUT_CUTOFF,str(raw.index[-1].date()),cfg)
        diagnosis[str(h)]={
            "architecture":arch,
            "metric":evaluation["metrics"],
            "unshrunk_metric":evaluation["unshrunk_metrics"],
            "n_train":evaluation["train"]["n_train"],
            "n_eval":evaluation["metrics"]["n"],
            "inner_epochs":evaluation["inner"]["epoch"],
            "inner_alpha":evaluation["inner"]["alpha"],
        }
        for row in rows:
            row["stage"]="historical_diagnostic_not_pristine"
        diagnosis_rows.extend(rows)
    # Refit selected architecture PER HORIZON using ALL h-labels matured today.
    final,models={},{}
    for h in cfg.horizons:
        arch=ranking[str(h)]["chosen_architecture"]
        end=str(raw.index[-1].date())
        inner=fit_inner(ex,arch,h,end,cfg)
        model,scaler,meta=fit_full(ex,arch,h,end,cfg,inner["epoch"])
        lastx=scaler.transform(ex.features)[-cfg.window:][None,:,:]
        pred=_infer(model,lastx)
        forecast=float(return_to_price(np.array([ex.prices[-1]]),
                     pred,scaler.target_rms,inner["alpha"])[0])
        models[str(h)]=model.cpu()
        final[str(h)]={
            "horizon":h,"architecture":arch,
            "train_until":end,"train":meta,
            "n_parameters":sum(p.numel() for p in model.parameters()),
            "scaler":asdict(scaler),
            "selected":inner,
            "base_price":float(ex.prices[-1]),
            "predicted_price":forecast,
        }
    timestamp=datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    run_id=timestamp+"_"+provenance["data_sha256"][:8]
    output=Path(report_dir)/run_id
    destination=Path(model_dir)/run_id
    output.mkdir(parents=True,exist_ok=False)
    destination.mkdir(parents=True,exist_ok=False)
    for h,model in models.items():
        weight=destination/f"h{h}.pt"
        torch.save(model.state_dict(),weight)
        final[h]["weights_file"]=weight.name
        final[h]["weights_sha256"]=sha256(weight.read_bytes()).hexdigest()
    summary={
        "mode":"independent-horizon-complete-data",
        "created_utc":datetime.now(timezone.utc).isoformat(),
        "config":asdict(cfg),"provenance":provenance,
        "selection_cutoffs":plan,
        "outer_validation":folds,"architecture_selection":ranking,
        "historical_diagnostic":diagnosis,
        "final_models":final,
        "historical_diagnostic_is_independent":False,
        "previous_production_model_untouched":True,
        "matured_latest":known,
    }
    (output/"summary.json").write_text(json.dumps(
        summary,indent=2,ensure_ascii=False,allow_nan=False),
        encoding="utf-8")
    metrics=[]
    for r in folds:
        m=r["metrics"]
        metrics.append({
            "architecture":r["architecture"],"horizon":r["horizon"],
            "cutoff":r["cutoff"],"validation_end":r["period_end"],
            "n":m["n"],"rmse":m["rmse"],
            "naive_rmse":m["naive_rmse"],"improvement_pct":m["improvement_pct"],
            "r2":m["r2"],"mae":m["mae"],
            "unshrunk_rmse":r["unshrunk_metrics"]["rmse"],
            "alpha":r["inner"]["alpha"],"epochs":r["inner"]["epoch"]
        })
    _write_csv(output/"fold_metrics.csv",metrics)
    _write_csv(output/"validation_predictions.csv",validation_rows)
    _write_csv(output/"historical_diagnostic_predictions.csv",diagnosis_rows)
    diagnostic_metrics=[]
    for h,d in diagnosis.items():
        diagnostic_metrics.append({
            "horizon":int(h),"architecture":d["architecture"],
            "train_until":HOLDOUT_CUTOFF,
            **{k:d["metric"][k] for k in
               ("n","mae","mse","rmse","r2","naive_rmse",
                "naive_r2","improvement_pct")},
            "roc_auc":(d["metric"]["direction"] or {}).get("roc_auc"),
        })
    _write_csv(output/"historical_diagnostic_metrics.csv",diagnostic_metrics)
    _charts(output,folds,diagnosis_rows)
    report=[
        "# Brent Intelligence V5: modelos independientes por horizonte",
        "",
        "Experimento completo con todos los precios disponibles hasta "
        + provenance["end"] + ".",
        "Los datos 2024+ ya fueron consultados antes: el diagnostico NO es virgen.",
        "",
        "| Horizonte | Arquitectura elegida por validacion | Media fold RMSE/naive | Folds ganados | Ejemplos finales |",
        "|---|---|---:|---:|---:|",
    ]
    for h in cfg.horizons:
        d=ranking[str(h)]
        info=d["candidates"][d["chosen_architecture"]]
        report.append(f"| {h} | {d['chosen_architecture'].upper()} | "
                      f"{info['mean_fold_rmse_ratio']:.4f} | "
                      f"{info['won_folds']}/{info['folds']} | "
                      f"{final[str(h)]['train']['n_train']} |")
    report += [
        "",
        "## Metodologia y advertencias",
        "Cada red es independiente y usa todas sus etiquetas maduras en su "
        "horizonte. El aprendizaje es de retornos logaritmicos, con "
        "cabeza residual inicialmente igual al ingenuo, AdamW y dropout.",
        "La epoca y la contraccion del retorno se determinan "
        "con validacion interna cronologica; la arquitectura se elige "
        "unicamente con folds exteriores terminados en 2023.",
        "Se publican metricas raw/unshrunk y post-contraccion para detectar "
        "si la supuesta mejora es en realidad prediccion ingenua.",
        "Los resultados historicos despues de 2023 pertenecen a modelos "
        "ajustados exclusivamente antes de 2024. Los modelos finales "
        "entrenados hasta hoy NO tienen test independiente propio.",
        "Nunca publicar pesos como modelos de produccion sin revisar.",
        "",
        "## Archivos",
        "summary.json, fold_metrics.csv, validation_predictions.csv, "
        "historical_diagnostic_predictions.csv, "
        "historical_diagnostic_metrics.csv, learning_curves.html, "
        "real_vs_predicted.html.",
    ]
    (output/"REPORT.md").write_text("\n".join(report)+"\n",encoding="utf-8")
    summary["output"]=str(output)
    summary["candidate_weights"]=str(destination)
    return summary
