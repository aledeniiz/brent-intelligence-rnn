"""Research v4: causal, regularized residual LSTM/GRU; NEVER touches production.

This program trains ONLY on examples whose three labels matured by each cutoff.
It physically excludes data after 2021-12-31 before constructing features.
Model and shrinkage are selected in an INNER chronological split. Outer
2012-2021 folds provide the comparison against the last-price baseline.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from hashlib import sha256
import csv
import json
import math
import os
from pathlib import Path
import random
import statistics

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.metrics import roc_auc_score, balanced_accuracy_score
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from brent_intelligence.config import DATA_PATH, REPORTS_DIR
from brent_intelligence.evaluation import prices_from_returns, score
from brent_intelligence.experiment import Preprocessor
from brent_intelligence.features import load_prices, make_features
from brent_intelligence.temporal import FOLD_BOUNDARIES, make_examples


# Predeclared before viewing any v4 results; no test-dependent tuning.
HORIZONS = (5, 63, 252)
SHRINK_GRID = (0.0, 0.25, 0.50, 0.75, 1.0)
RESEARCH_END = "2021-12-31"


@dataclass(frozen=True)
class V4Config:
    architecture: str = "lstm"
    hidden_size: int = 16
    dropout: float = 0.20
    weight_decay: float = 0.01
    prediction_penalty: float = 0.08
    learning_rate: float = 0.001
    batch_size: int = 128
    max_epochs: int = 18
    patience: int = 4
    min_inner_gain: float = 0.01
    window: int = 60
    seeds: tuple[int, ...] = (42, 31415)
    max_folds: int = 5

    def validate(self) -> None:
        if self.architecture not in ("lstm", "gru"):
            raise ValueError("Architecture must be lstm or gru")
        if not (0 <= self.dropout < 1 and self.weight_decay >= 0 and
                self.prediction_penalty >= 0 and 0 <= self.min_inner_gain < 1):
            raise ValueError("Invalid regularization setting")
        if (self.hidden_size < 1 or self.window < 3 or self.batch_size < 1 or
            self.max_epochs < 1 or self.patience < 1 or
            not (1 <= self.max_folds <= 5) or not self.seeds or
            len(set(self.seeds)) != len(self.seeds)):
            raise ValueError("Invalid model/validation configuration")


class ResidualRNN(nn.Module):
    """Zero-initialized output predicts a correction to log price persistence."""

    def __init__(self, n_features: int, cfg: V4Config):
        super().__init__()
        self.dropout = nn.Dropout(cfg.dropout)
        recurrent = nn.LSTM if cfg.architecture == "lstm" else nn.GRU
        self.recurrent = recurrent(n_features, cfg.hidden_size, batch_first=True)
        self.head = nn.Linear(cfg.hidden_size, len(HORIZONS))
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        states, _ = self.recurrent(self.dropout(x))
        return self.head(states[:, -1, :])


def prepare_historical(data_path: Path, *, window: int):
    """Hard isolation: no features, targets or index dates after 2021 exist."""
    source = load_prices(data_path)
    historical = source.loc[:RESEARCH_END].copy()
    if len(historical) < window + 4 * max(HORIZONS):
        raise ValueError("Historical training window is too short")
    features = make_features(historical)
    # A small causal Brent-only feature set; avoids exogenous availability
    # and historical revision problems discovered in v1-v3.
    features = features.filter(regex=r"^brent_")
    ex = make_examples(features, historical["brent"],
                       window=window, horizons=HORIZONS)
    historical_hash = sha256(
        historical[["brent"]].to_csv(index=True, float_format="%.12g").encode("utf-8")
    ).hexdigest()
    return ex, {"source_sha256": sha256(Path(data_path).read_bytes()).hexdigest(),
                "historical_brent_sha256": historical_hash,
                "source_end": str(source.index.max().date()),
                "research_end": str(historical.index.max().date()),
                "n_historical_rows": len(historical),
                "n_features": len(features.columns),
                "feature_columns": features.columns.tolist()}


def outer_folds(ex, max_folds: int = 5):
    origin_dates = ex.dates.take(ex.anchors)
    maturity_dates = ex.dates.take(ex.targets.max(axis=1))
    for k, (train_end, valid_end) in enumerate(FOLD_BOUNDARIES[:max_folds], 1):
        train = np.flatnonzero(maturity_dates <= pd.Timestamp(train_end))
        valid = np.flatnonzero((origin_dates > pd.Timestamp(train_end)) &
                               (maturity_dates <= pd.Timestamp(valid_end)))
        if len(train) < 100 or len(valid) < 20:
            raise ValueError(f"Outer fold {k}: insufficient matured targets")
        yield {"fold": k, "train_end": train_end, "validation_end": valid_end,
               "train": train, "validation": valid}


def inner_split(ex, train_ids: np.ndarray, train_end: str):
    """Purge labels across the inner boundary, respecting horizon 252."""
    train_ids = np.asarray(train_ids, dtype=np.int64)
    origins = ex.dates.take(ex.anchors[train_ids])
    cut_position = max(30, int(0.75 * len(train_ids)))
    inner_cut = pd.Timestamp(origins[cut_position - 1])
    maturity = ex.dates.take(ex.targets[train_ids].max(axis=1))
    train = train_ids[np.flatnonzero(maturity <= inner_cut)]
    val = train_ids[np.flatnonzero((origins > inner_cut) &
                                   (maturity <= pd.Timestamp(train_end)))]
    if len(train) < 50 or len(val) < 20:
        raise ValueError("Inner split has too few embargoed examples")
    return str(inner_cut.date()), train, val


def scaled_arrays(ex, train: np.ndarray, validation: np.ndarray, cutoff: str):
    """Imputation and scaling fit exclusively to the past of this cutoff."""
    # Preprocessor checks that all labels in train have already matured.
    prep = Preprocessor.fit(ex, train, cutoff)
    values = prep.x(ex.features)
    scale = np.asarray(prep.y_std, dtype=np.float32)
    # Unlike v1 do NOT subtract the mean: zero output = naive forecast.
    scale = np.where(scale < 1e-5, 1.0, scale)
    return (ex.x(values, train), (ex.y(train) / scale).astype(np.float32),
            ex.x(values, validation), (ex.y(validation) / scale).astype(np.float32),
            scale, prep)


def _seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _loader(x: np.ndarray, y: np.ndarray, cfg: V4Config, seed: int):
    return DataLoader(
        TensorDataset(torch.from_numpy(x), torch.from_numpy(y)),
        batch_size=cfg.batch_size, shuffle=True,
        generator=torch.Generator().manual_seed(seed),
    )


def _epoch_train(model, optimizer, loader, cfg):
    model.train()
    objective_total, data_total, seen = 0., 0., 0
    for x, y in loader:
        optimizer.zero_grad(set_to_none=True)
        predicted = model(x)
        data_loss = nn.functional.smooth_l1_loss(predicted, y)
        objective = data_loss + cfg.prediction_penalty * predicted.square().mean()
        objective.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        data_total += float(data_loss.item()) * len(x)
        objective_total += float(objective.item()) * len(x)
        seen += len(x)
    return data_total / max(seen, 1), objective_total / max(seen, 1)


@torch.no_grad()
def _infer(model, x: np.ndarray) -> np.ndarray:
    model.eval()
    out = []
    for start in range(0, len(x), 512):
        out.append(model(torch.from_numpy(x[start:start + 512])).cpu().numpy())
    return np.concatenate(out)


def _unscaled_returns(pred_scaled: np.ndarray, scale: np.ndarray):
    value = np.asarray(pred_scaled, dtype=np.float64) * scale[None, :]
    if not np.isfinite(value).all():
        raise ValueError("Nonfinite predicted log returns")
    return value


def _score_log_predictions(ex, ids, log_predictions):
    return score(ex.actual(ids),
                 prices_from_returns(ex.anchors_price(ids), log_predictions),
                 ex.anchors_price(ids), HORIZONS)


def train_inner(ex, inner_train, inner_val, cutoff, cfg, seed):
    xt, yt, xv, yv, scale, prep = scaled_arrays(
        ex, inner_train, inner_val, cutoff
    )
    _seed_all(seed)
    model = ResidualRNN(len(prep.columns), cfg)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay
    )
    loader = _loader(xt, yt, cfg, seed)
    observed = ex.actual(inner_val)
    naive = ex.anchors_price(inner_val)
    naive_rmse = np.sqrt(np.mean((naive[:, None] - observed) ** 2, axis=0))
    if np.any(naive_rmse <= 0):
        raise ValueError("Zero baseline RMSE prevents fair ratio selection")
    best_score = math.inf
    best_epoch = 1
    best_predictions = None
    wait = 0
    history = []
    for epoch in range(1, cfg.max_epochs + 1):
        train_loss, train_objective = _epoch_train(model, optimizer, loader, cfg)
        val_pred_scaled = _infer(model, xv)
        val_loss = float(nn.functional.smooth_l1_loss(
            torch.from_numpy(val_pred_scaled), torch.from_numpy(yv)
        ).item())
        log_preds = _unscaled_returns(val_pred_scaled, scale)
        price_preds = prices_from_returns(naive, log_preds)
        ratios = np.sqrt(np.mean((price_preds - observed) ** 2, axis=0)) / naive_rmse
        criterion = float(np.mean(ratios))
        if not np.isfinite(criterion):
            raise ValueError("Unstable inner validation")
        history.append({"epoch": epoch,
                        "train_loss": train_loss,
                        "train_loss_with_penalty": train_objective,
                        "val_smooth_l1": val_loss,
                        "mean_rmse_relative": criterion,
                        "rmse_relative_by_horizon": ratios.tolist()})
        if criterion < best_score - 1e-4:
            best_score = criterion
            best_epoch = epoch
            best_predictions = log_preds.copy()
            wait = 0
        else:
            wait += 1
        if wait >= cfg.patience:
            break
    return {"seed": seed, "epoch": best_epoch, "history": history,
            "inner_log_predictions": best_predictions,
            "best_inner_score": best_score,
            "n_features": len(prep.columns)}


def refit_outer(ex, train_ids, valid_ids, cutoff, cfg, seed, epochs):
    xt, yt, xv, _, scale, prep = scaled_arrays(
        ex, train_ids, valid_ids, cutoff
    )
    _seed_all(seed)
    model = ResidualRNN(len(prep.columns), cfg)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay
    )
    loader = _loader(xt, yt, cfg, seed)
    for _ in range(epochs):
        _epoch_train(model, optimizer, loader, cfg)
    return _unscaled_returns(_infer(model, xv), scale)


def choose_shrinkage(ex, inner_ids, ensemble_log_predictions, cfg):
    """Independent horizon coefficients selected ONLY inside training past."""
    actual = ex.actual(inner_ids)
    anchor = ex.anchors_price(inner_ids)
    coefficients = []
    comparisons = []
    for j, h in enumerate(HORIZONS):
        baseline = float(np.sqrt(np.mean((anchor - actual[:, j]) ** 2)))
        candidate_scores = {}
        for alpha in SHRINK_GRID:
            predicted = anchor * np.exp(alpha * ensemble_log_predictions[:, j])
            candidate_scores[str(alpha)] = float(np.sqrt(
                np.mean((predicted - actual[:, j]) ** 2)
            ))
        best = min(SHRINK_GRID, key=lambda a: candidate_scores[str(a)])
        if candidate_scores[str(best)] >= baseline * (1.0 - cfg.min_inner_gain):
            best = 0.0
        coefficients.append(float(best))
        comparisons.append({"horizon": h, "inner_naive_rmse": baseline,
                            "inner_rmse_by_alpha": candidate_scores,
                            "selected_alpha": best})
    return np.array(coefficients, dtype=np.float64), comparisons


def directional_auc(actual, anchor, log_predictions) -> list[dict]:
    """Descriptive ranking only; not a separate classifier or its test AUC."""
    result = []
    for j, h in enumerate(HORIZONS):
        class_true = (actual[:, j] > anchor).astype(int)
        scores = log_predictions[:, j]
        valid = actual[:, j] != anchor
        if len(np.unique(class_true[valid])) == 2:
            auc = float(roc_auc_score(class_true[valid], scores[valid]))
            balanced = float(balanced_accuracy_score(
                class_true[valid], (scores[valid] > 0).astype(int)
            ))
        else:
            auc, balanced = None, None
        result.append({"horizon": h, "roc_auc": auc,
                       "balanced_accuracy": balanced,
                       "n_nonzero_direction": int(np.count_nonzero(valid))})
    return result


def evaluate_fold(ex, fold, cfg):
    train_ids, valid_ids = fold["train"], fold["validation"]
    inner_cut, inner_train, inner_val = inner_split(
        ex, train_ids, fold["train_end"]
    )
    models = [train_inner(ex, inner_train, inner_val, inner_cut, cfg, seed)
              for seed in cfg.seeds]
    inner_pred = np.mean(
        np.stack([item["inner_log_predictions"] for item in models]), axis=0
    )
    shrink, inner_selection = choose_shrinkage(ex, inner_val, inner_pred, cfg)
    out_predictions = np.mean(np.stack([
        refit_outer(ex, train_ids, valid_ids, fold["train_end"],
                    cfg, item["seed"], item["epoch"])
        for item in models
    ]), axis=0)
    effective = out_predictions * shrink[None, :]
    metrics = _score_log_predictions(ex, valid_ids, effective)
    raw_metrics = _score_log_predictions(ex, valid_ids, out_predictions)
    direction = directional_auc(ex.actual(valid_ids),
                                ex.anchors_price(valid_ids), effective)
    dates = ex.dates.take(ex.anchors[valid_ids])
    targets_dates = ex.dates.take(ex.targets[valid_ids].ravel()).to_numpy()
    targets_dates = targets_dates.reshape(-1, len(HORIZONS))
    prices = prices_from_returns(ex.anchors_price(valid_ids), effective)
    raw_prices = prices_from_returns(ex.anchors_price(valid_ids), out_predictions)
    records = []
    for i, origin in enumerate(dates):
        for j, h in enumerate(HORIZONS):
            records.append({
                "fold": fold["fold"], "origin": origin.date().isoformat(),
                "horizon": h,
                "target_date": str(pd.Timestamp(targets_dates[i, j]).date()),
                "anchor": float(ex.anchors_price(valid_ids)[i]),
                "observed": float(ex.actual(valid_ids)[i, j]),
                "raw_model": float(raw_prices[i, j]),
                "shrunk_model": float(prices[i, j]),
                "alpha": float(shrink[j]),
            })
    report = {
        "fold": fold["fold"],
        "train_end": fold["train_end"],
        "validation_end": fold["validation_end"],
        "inner_cutoff": inner_cut,
        "n_outer_train": len(train_ids),
        "n_inner_train": len(inner_train),
        "n_inner_validation": len(inner_val),
        "n_outer_validation": len(valid_ids),
        "seeds": [dict(seed=m["seed"], best_epoch=m["epoch"],
                       best_inner_score=m["best_inner_score"],
                       history=m["history"], n_features=m["n_features"])
                  for m in models],
        "shrinkage_selected_inner_only": inner_selection,
        "metrics": metrics,
        "raw_metrics": raw_metrics,
        "directional": direction,
    }
    return report, records


def aggregate_metrics(folds, key="metrics"):
    summary = []
    for h in HORIZONS:
        sub = [fold[key][str(h)] for fold in folds]
        n = sum(item["n"] for item in sub)
        mae = sum(item["mae"] * item["n"] for item in sub) / n
        mse = sum(item["mse"] * item["n"] for item in sub) / n
        naive_mse = sum(item["naive_rmse"] ** 2 * item["n"] for item in sub) / n
        rmse, naive = math.sqrt(mse), math.sqrt(naive_mse)
        summary.append({
            "horizon": h, "n_origins": n, "mae": mae,
            "rmse": rmse, "naive_rmse": naive,
            "improvement_pct": 100 * (1 - rmse / naive),
            "folds_better_than_naive": sum(item["rmse"] < item["naive_rmse"] for item in sub),
            "n_folds": len(sub),
            "mean_fold_relative_rmse": statistics.mean(
                item["relative_rmse"] for item in sub),
        })
    return summary


def render_plots(folds, aggregate, out_dir: Path):
    loss = make_subplots(rows=len(folds), cols=1,
                         subplot_titles=[f"Fold {f['fold']} - train/inner validation"
                                         for f in folds])
    for i, fold in enumerate(folds, 1):
        for seed in fold["seeds"]:
            epochs = [h["epoch"] for h in seed["history"]]
            loss.add_trace(go.Scatter(
                x=epochs, y=[h["train_loss"] for h in seed["history"]],
                mode="lines+markers", showlegend=(i == 1),
                name="Train (SmoothL1)", legendgroup="train"
            ), row=i, col=1)
            loss.add_trace(go.Scatter(
                x=epochs, y=[h["val_smooth_l1"] for h in seed["history"]],
                mode="lines+markers", showlegend=(i == 1),
                name="Validacion (SmoothL1)", legendgroup="val"
            ), row=i, col=1)
    loss.update_layout(height=max(450, 230 * len(folds)),
                       title="Curvas por fold: SmoothL1 train/validation; la penalizacion extra se guarda en JSON")
    loss.write_html(str(out_dir / "learning_curves.html"), include_plotlyjs=True)
    chart = go.Figure()
    for h in HORIZONS:
        chart.add_trace(go.Bar(
            x=[f"Fold {f['fold']}" for f in folds],
            y=[100 * (1 - f["metrics"][str(h)]["relative_rmse"]) for f in folds],
            name=f"{h} sesiones"
        ))
    chart.add_hline(y=0, line_dash="dash", annotation_text="Empate con ingenuo")
    chart.update_layout(title="Mejora de RMSE sobre ingenuo por fold",
                        yaxis_title="Mejora % (positivo es mejor)", barmode="group")
    chart.write_html(str(out_dir / "fold_improvements.html"), include_plotlyjs=True)


def run_experiment(*, data_path: Path = DATA_PATH,
                   output_dir: Path = REPORTS_DIR / "research_v4",
                   cfg: V4Config | None = None,
                   dry_run: bool = False) -> dict:
    cfg = cfg or V4Config()
    cfg.validate()
    torch.set_num_threads(min(4, os.cpu_count() or 1))
    ex, provenance = prepare_historical(data_path, window=cfg.window)
    folds = list(outer_folds(ex, cfg.max_folds))
    if dry_run:
        return {"mode": "dry-run", "config": asdict(cfg),
                "provenance": provenance,
                "folds": [{"fold": f["fold"], "train_end": f["train_end"],
                           "validation_end": f["validation_end"],
                           "n_train": len(f["train"]),
                           "n_validation": len(f["validation"])}
                          for f in folds],
                "used_test_2024_onwards": False}
    reports, predictions = [], []
    for fold in folds:
        result, records = evaluate_fold(ex, fold, cfg)
        reports.append(result)
        predictions.extend(records)
        print(f"Fold {fold['fold']}: " + ", ".join(
            f"{h}d RMSE/naive={result['metrics'][str(h)]['relative_rmse']:.4f}"
            for h in HORIZONS), flush=True)
    aggregate = aggregate_metrics(reports)
    raw_aggregate = aggregate_metrics(reports, key="raw_metrics")
    result = {
        "protocol": "research-v4-2021-only-nested-validation",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "config": asdict(cfg),
        "provenance": provenance,
        "train_data_cutoff": RESEARCH_END,
        "used_calibration_or_test_dates": False,
        "touched_active_or_frozen_model": False,
        "evaluation": "5 outer folds; chronological matured inner validation for epochs and shrinkage; 2 seed ensemble by default",
        "aggregate": aggregate,
        "unshrunk_aggregate": raw_aggregate,
        "folds": reports,
        "warning": [
            "Exploratory validation only, NOT independent test.",
            "Historical 2024-2026 test of v1 has already been seen and is never used here.",
            "Predictive gains are not guaranteed; possible dependence across overlapping targets.",
            "The production model and prospective forecasts are unchanged.",
            "Train and validation plots use SmoothL1; optimization also includes a prediction penalty.",
            "Directional AUC on outer validation is descriptive, not a classifier probability or a selection criterion."
        ],
    }
    target = Path(output_dir)
    target.mkdir(parents=True, exist_ok=True)
    suffix = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = target / (suffix + "_" + provenance["historical_brent_sha256"][:8])
    run_dir.mkdir(exist_ok=False)
    (run_dir / "summary.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8"
    )
    with (run_dir / "predictions.csv").open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(predictions[0]))
        writer.writeheader()
        writer.writerows(predictions)
    with (run_dir / "validation_metrics.csv").open(
        "w", newline="", encoding="utf-8"
    ) as fh:
        writer = csv.DictWriter(fh, fieldnames=list(aggregate[0]))
        writer.writeheader()
        writer.writerows(aggregate)
    render_plots(reports, aggregate, run_dir)
    document = [
        "# Brent Intelligence: investigacion V4",
        "",
        "Evaluacion exclusivamente con datos hasta 2021. No es un nuevo test.",
        "",
        "| Horizonte | MAE USD | RMSE USD | RMSE ingenuo | Mejora RMSE | Folds ganados |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for m in aggregate:
        document.append(
            f"| {m['horizon']} | {m['mae']:.3f} | {m['rmse']:.3f} | "
            f"{m['naive_rmse']:.3f} | {m['improvement_pct']:+.2f}% | "
            f"{m['folds_better_than_naive']}/{m['n_folds']} |"
        )
    document += [
        "", "## Metodologia",
        "Modelo recurrente pequeno, dropout de entrada, AdamW, penalizacion "
        "a retorno cero (ingenuo) y ensemble de semillas.",
        "Epocas y contraccion de retorno elegidas en split interno purgado. "
        "La validacion externa no selecciona epocas ni coefficients.",
        "Comparar con LSTM v1 en las mismas fechas, pero reconocer que el "
        "protocolo anterior seleccionaba epocas en validacion externa.",
        "La contraccion con alpha=0 devuelve exactamente el predictor ingenuo.",
        "Un AUC alto en periodos solapados no demuestra estabilidad o significacion.",
        "", "## Archivos",
        "summary.json, predictions.csv, validation_metrics.csv, "
        "learning_curves.html y fold_improvements.html.",
        "", "No se modifica ni se reemplaza el modelo publico.",
    ]
    (run_dir / "REPORT.md").write_text("\n".join(document) + "\n", encoding="utf-8")
    result["saved_to"] = str(run_dir)
    return result
