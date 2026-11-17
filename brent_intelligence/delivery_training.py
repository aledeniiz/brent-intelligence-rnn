"""End-to-end delivery training on all currently available Brent observations.

Separate per-horizon label masks let 5/63-session heads use recent mature
examples without waiting 252 sessions. All models are trained as complete
models; hyperparameters are fixed before validation, and early stopping is
selected on chronology-purged inner splits. Original production is untouched.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
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
from sklearn.metrics import (accuracy_score, balanced_accuracy_score,
                             f1_score, precision_score, recall_score,
                             roc_auc_score)

from brent_intelligence.config import ARTIFACTS_DIR, DATA_PATH, REPORTS_DIR
from brent_intelligence.features import load_prices, make_features

HORIZONS = (5, 63, 252)
KINDS = ("rnn", "lstm", "gru")
SELECTION_FOLDS = (
    ("2017-12-29", "2019-12-31"),
    ("2019-12-31", "2021-12-31"),
    ("2021-12-31", "2023-12-29"),
)
TEST_START = "2024-01-01"
ALPHAS = (0.0, .25, .5, .75, 1.0)


@dataclass(frozen=True)
class DeliveryConfig:
    window: int = 60
    hidden_size: int = 24
    dropout: float = 0.15
    weight_decay: float = 0.01
    prediction_penalty: float = 0.05
    learning_rate: float = 0.001
    batch_size: int = 128
    max_epochs: int = 20
    patience: int = 4
    seed: int = 42
    min_coverage: float = .60
    required_inner_gain: float = .01
    architectures: tuple[str, ...] = KINDS
    n_folds: int = 3

    def validate(self):
        if (self.window < 2 or self.hidden_size < 1 or self.batch_size < 1
            or self.max_epochs < 1 or self.patience < 1
            or not (1 <= self.n_folds <= len(SELECTION_FOLDS))
            or not self.architectures
            or any(x not in KINDS for x in self.architectures)
            or len(set(self.architectures)) != len(self.architectures)
            or not (0 <= self.dropout < 1)
            or not (0 <= self.min_coverage <= 1)
            or not (0 <= self.required_inner_gain < 1)
            or min(self.weight_decay, self.prediction_penalty,
                   self.learning_rate) < 0
            or self.learning_rate == 0):
            raise ValueError("Invalid complete-training configuration")


@dataclass
class FullExamples:
    features: pd.DataFrame
    prices: np.ndarray
    anchors: np.ndarray
    target_indices: np.ndarray
    log_returns: np.ndarray
    observed: np.ndarray
    window: int

    @property
    def dates(self):
        return self.features.index

    def windows(self, x: np.ndarray, ids: np.ndarray):
        return np.stack([x[i-self.window+1:i+1]
                         for i in self.anchors[ids]]).astype(np.float32)


def build_examples(raw: pd.DataFrame, window: int = 60) -> FullExamples:
    if (not raw.index.is_monotonic_increasing or raw.index.has_duplicates
        or window < 2 or len(raw) < window + 252):
        raise ValueError("Invalid dates or insufficient data")
    features = make_features(raw)
    prices = raw["brent"].to_numpy(dtype=np.float64)
    anchors = np.arange(window-1, len(prices), dtype=np.int64)
    targets = anchors[:, None] + np.array(HORIZONS)[None, :]
    observed = targets < len(prices)
    safe_target = np.minimum(targets, len(prices)-1)
    log_returns = np.log(prices[safe_target] / prices[anchors, None])
    log_returns[~observed] = 0.
    return FullExamples(features, prices, anchors, targets,
                        log_returns, observed, window)


def _cutoff_index(ex: FullExamples, cutoff: str) -> int:
    return int(ex.dates.searchsorted(pd.Timestamp(cutoff), side="right") - 1)


def split_ids(ex: FullExamples, train_end: str, validation_end: str):
    cut = _cutoff_index(ex, train_end)
    end = _cutoff_index(ex, validation_end)
    if cut < ex.window or end <= cut:
        raise ValueError("Temporal boundaries do not contain valid periods")
    train_mask_all = ex.observed & (ex.target_indices <= cut)
    val_mask_all = ex.observed & (ex.target_indices <= end)
    train = np.flatnonzero((ex.anchors <= cut) & train_mask_all.any(axis=1))
    valid = np.flatnonzero((ex.anchors > cut) & (ex.anchors <= end)
                          & val_mask_all.any(axis=1))
    train_mask, val_mask = train_mask_all[train], val_mask_all[valid]
    if len(train) < 100 or len(valid) < 30:
        raise ValueError("Insufficient mature targets")
    if np.any(train_mask.sum(axis=0) < 30) or np.any(val_mask.sum(axis=0) < 20):
        raise ValueError("Each horizon needs enough mature labels")
    return train, train_mask, valid, val_mask


def inner_split(ex: FullExamples, train_end: str):
    """Three calendar years of recent internal validation, purged by maturity."""
    cutoff = (pd.Timestamp(train_end) - pd.DateOffset(years=3)).date().isoformat()
    tr, tm, vl, vm = split_ids(ex, cutoff, train_end)
    return cutoff, tr, tm, vl, vm


@dataclass
class Scaler:
    columns: list[str]
    medians: list[float]
    mean: list[float]
    std: list[float]
    y_std: list[float]

    @classmethod
    def fit(cls, ex: FullExamples, train_ids: np.ndarray,
            train_mask: np.ndarray, cutoff: str, coverage: float):
        end = _cutoff_index(ex, cutoff)
        # Prove that no masked label requires future observations.
        if np.any(ex.target_indices[train_ids][train_mask] > end):
            raise ValueError("Temporal leakage: training uses future labels")
        known = ex.features.iloc[:end+1]
        columns = known.columns[known.notna().mean() >= coverage].tolist()
        if not columns:
            raise ValueError("No input features meet minimum coverage")
        med = known[columns].median().fillna(0.)
        matrix = known[columns].fillna(med).to_numpy(dtype=float)
        mean = matrix.mean(axis=0)
        std = np.maximum(matrix.std(axis=0), 1e-6)
        scales = []
        for j in range(len(HORIZONS)):
            targets = ex.log_returns[train_ids, j][train_mask[:, j]]
            scales.append(float(max(np.sqrt(np.mean(targets**2)), 1e-4)))
        return cls(columns, med.tolist(), mean.tolist(),
                   std.tolist(), scales)

    def x(self, features):
        matrix = features.reindex(columns=self.columns).to_numpy(dtype=float)
        med = np.asarray(self.medians)
        matrix = np.where(np.isfinite(matrix), matrix, med[None, :])
        result = (matrix - self.mean) / self.std
        return np.clip(result, -8., 8.).astype(np.float32)


class DeliveryForecaster(nn.Module):
    """Small regularized recurrent backbone, zero-return residual head."""

    def __init__(self, kind: str, n_features: int, cfg: DeliveryConfig):
        super().__init__()
        if kind not in KINDS:
            raise ValueError(f"Unsupported architecture: {kind}")
        recurrent = {"rnn": nn.RNN, "lstm": nn.LSTM, "gru": nn.GRU}[kind]
        self.dropout = nn.Dropout(cfg.dropout)
        self.recurrent = recurrent(n_features, cfg.hidden_size,
                                   batch_first=True)
        self.head = nn.Linear(cfg.hidden_size, len(HORIZONS))
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward(self, windows):
        seq, _ = self.recurrent(self.dropout(windows))
        return self.head(seq[:, -1, :])


def _set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _masked_huber(yhat: torch.Tensor, actual: torch.Tensor,
                  mask: torch.Tensor):
    element = nn.functional.smooth_l1_loss(yhat, actual, reduction="none")
    return (element * mask).sum() / mask.sum().clamp_min(1.)


def _fit_arrays(ex, train, trmask, valid, valmask, cutoff, cfg):
    scaler = Scaler.fit(ex, train, trmask, cutoff, cfg.min_coverage)
    full_x = scaler.x(ex.features)
    s = np.asarray(scaler.y_std, dtype=np.float32)
    return (scaler, ex.windows(full_x, train),
            (ex.log_returns[train] / s).astype(np.float32),
            trmask.astype(np.float32), ex.windows(full_x, valid),
            (ex.log_returns[valid] / s).astype(np.float32),
            valmask.astype(np.float32))


def _loader(x, y, masks, cfg):
    return DataLoader(TensorDataset(
        torch.from_numpy(x), torch.from_numpy(y), torch.from_numpy(masks)
    ), batch_size=cfg.batch_size, shuffle=True,
       generator=torch.Generator().manual_seed(cfg.seed))


def _fit_epoch(model, optim, data, cfg):
    model.train()
    loss_total, data_total, seen = 0., 0., 0
    for x, y, m in data:
        optim.zero_grad(set_to_none=True)
        pred = model(x)
        loss_data = _masked_huber(pred, y, m)
        reg = cfg.prediction_penalty * (pred.square() * m).sum() / m.sum().clamp_min(1.)
        loss = loss_data + reg
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.)
        optim.step()
        loss_total += float(loss.item()) * len(x)
        data_total += float(loss_data.item()) * len(x)
        seen += len(x)
    return data_total / seen, loss_total / seen


@torch.no_grad()
def _infer(model, xs):
    model.eval()
    return np.concatenate([
        model(torch.from_numpy(xs[k:k+512])).cpu().numpy()
        for k in range(0, len(xs), 512)
    ])


def _metrics(ex, ids, masks, predictions):
    anchors = ex.prices[ex.anchors[ids]]
    results = {}
    for j, horizon in enumerate(HORIZONS):
        m = masks[:, j].astype(bool)
        actual = ex.prices[ex.target_indices[ids, j][m]]
        initial = anchors[m]
        forecast = predictions[m, j]
        err = forecast - actual
        naive = initial - actual
        rmse = math.sqrt(float(np.mean(err**2)))
        naive_rmse = math.sqrt(float(np.mean(naive**2)))
        if naive_rmse <= 1e-8:
            raise ValueError("Degenerate naive RMSE")
        variance = float(np.sum((actual - np.mean(actual)) ** 2))
        r2 = (float(1. - np.sum(err**2) / variance)
              if variance > 1e-12 else None)
        naive_r2 = (float(1. - np.sum(naive**2) / variance)
                    if variance > 1e-12 else None)
        smape = 100. * np.mean(
            2. * np.abs(err) / np.maximum(
                np.abs(forecast) + np.abs(actual), 1e-8
            )
        )
        results[str(horizon)] = {
            "n": len(actual), "mae": float(np.mean(np.abs(err))),
            "mse": float(np.mean(err**2)), "rmse": rmse,
            "r2": r2, "naive_r2": naive_r2, "smape_pct": float(smape),
            "naive_rmse": naive_rmse,
            "relative_rmse": rmse / naive_rmse,
            "improvement_pct": 100 * (1 - rmse / naive_rmse),
        }
        positive = actual > initial
        nonzero = actual != initial
        if np.unique(positive[nonzero]).size == 2:
            scores = np.log(forecast[nonzero] / initial[nonzero])
            predicted = scores > 0
            results[str(horizon)]["direction"] = {
                "roc_auc": float(roc_auc_score(positive[nonzero], scores)),
                "accuracy": float(accuracy_score(positive[nonzero], predicted)),
                "balanced_accuracy": float(balanced_accuracy_score(positive[nonzero], predicted)),
                "precision": float(precision_score(positive[nonzero], predicted, zero_division=0)),
                "recall": float(recall_score(positive[nonzero], predicted, zero_division=0)),
                "f1": float(f1_score(positive[nonzero], predicted, zero_division=0)),
                "n": int(nonzero.sum()),
            }
        else:
            results[str(horizon)]["direction"] = None
    return results


def _price_prediction(ex, ids, scaled_output, scaler, alpha):
    predicted_log = scaled_output * np.asarray(scaler.y_std)[None, :] * np.asarray(alpha)[None, :]
    predicted_log = np.clip(predicted_log, -4., 4.)
    return ex.prices[ex.anchors[ids], None] * np.exp(predicted_log)


def _inner_metric(ex, ids, masks, predicted):
    scored = _metrics(ex, ids, masks, predicted)
    return float(statistics.mean(scored[str(h)]["relative_rmse"] for h in HORIZONS))


def train_inner(ex, kind, outer_cutoff, cfg):
    cut, train, tm, valid, vm = inner_split(ex, outer_cutoff)
    scaler, xt, yt, mt, xv, yv, mv = _fit_arrays(
        ex, train, tm, valid, vm, cut, cfg
    )
    _set_seed(cfg.seed)
    model = DeliveryForecaster(kind, len(scaler.columns), cfg)
    optim = torch.optim.AdamW(model.parameters(),
                              lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
    loader = _loader(xt, yt, mt, cfg)
    history = []
    best, best_epoch, best_pred, wait = math.inf, 1, None, 0
    for epoch in range(1, cfg.max_epochs+1):
        train_loss, objective = _fit_epoch(model, optim, loader, cfg)
        scaled_pred = _infer(model, xv)
        val_loss = float(_masked_huber(
            torch.from_numpy(scaled_pred), torch.from_numpy(yv),
            torch.from_numpy(mv)
        ).item())
        pred = _price_prediction(ex, valid, scaled_pred, scaler, [1, 1, 1])
        criterion = _inner_metric(ex, valid, vm, pred)
        history.append({"epoch": epoch, "train_smooth_l1": train_loss,
                        "train_objective": objective, "validation_smooth_l1": val_loss,
                        "validation_mean_relative_rmse": criterion})
        if criterion < best - 1e-4:
            best, best_epoch, best_pred, wait = criterion, epoch, scaled_pred.copy(), 0
        else:
            wait += 1
        if wait >= cfg.patience:
            break
    if best_pred is None:
        raise RuntimeError("No valid inner checkpoint")
    shrinkage = []
    table = []
    for j, h in enumerate(HORIZONS):
        m = vm[:, j].astype(bool)
        initial = ex.prices[ex.anchors[valid[m]]]
        actual = ex.prices[ex.target_indices[valid[m], j]]
        baseline = math.sqrt(float(np.mean((initial-actual)**2)))
        raw_log = best_pred[m, j] * scaler.y_std[j]
        rmse = {}
        for alpha in ALPHAS:
            forecast = initial * np.exp(np.clip(alpha * raw_log, -4., 4.))
            rmse[str(alpha)] = math.sqrt(float(np.mean((forecast - actual)**2)))
        best_alpha = min(ALPHAS, key=lambda a: rmse[str(a)])
        if rmse[str(best_alpha)] >= baseline * (1-cfg.required_inner_gain):
            best_alpha = 0.0
        shrinkage.append(float(best_alpha))
        table.append({"horizon": h, "baseline_rmse": baseline,
                      "alpha_rmse": rmse, "selected_alpha": best_alpha})
    return {"epoch": best_epoch, "alpha": shrinkage,
            "history": history, "inner_cutoff": cut,
            "inner_n_train": len(train),
            "inner_n_validation": {str(h): int(vm[:, j].sum())
                                   for j, h in enumerate(HORIZONS)},
            "shrinkage_selection": table}


def refit(ex, kind, cutoff, cfg, epochs):
    end = _cutoff_index(ex, cutoff)
    mask_all = ex.observed & (ex.target_indices <= end)
    train = np.flatnonzero((ex.anchors <= end) & mask_all.any(axis=1))
    tm = mask_all[train]
    if np.min(tm.sum(axis=0)) < 50:
        raise ValueError("Insufficient mature labels for full refit")
    scaler = Scaler.fit(ex, train, tm, cutoff, cfg.min_coverage)
    x = ex.windows(scaler.x(ex.features), train)
    y = (ex.log_returns[train] / np.asarray(scaler.y_std)).astype(np.float32)
    _set_seed(cfg.seed)
    model = DeliveryForecaster(kind, len(scaler.columns), cfg)
    optim = torch.optim.AdamW(model.parameters(),
                              lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
    loader = _loader(x, y, tm.astype(np.float32), cfg)
    history = []
    for epoch in range(1, epochs + 1):
        data_loss, objective = _fit_epoch(model, optim, loader, cfg)
        history.append({"epoch": epoch, "train_smooth_l1": data_loss,
                        "train_objective": objective})
    training = {
        "n_unique_origins": len(train),
        "n_mature_by_horizon": {str(h): int(tm[:, j].sum())
                               for j, h in enumerate(HORIZONS)},
        "last_mature_origin_by_horizon": {
            str(h): str(ex.dates[ex.anchors[train[tm[:, j]]][-1]].date())
            for j, h in enumerate(HORIZONS)},
        "last_label_date": str(ex.dates[end].date()),
        "epochs": epochs, "loss_history": history,
    }
    return model, scaler, training


def evaluate_refit(ex, model, scaler, ids, masks, alpha):
    x = ex.windows(scaler.x(ex.features), ids)
    predicted = _price_prediction(ex, ids, _infer(model, x), scaler, alpha)
    return _metrics(ex, ids, masks, predicted)


def run_delivery(*, data_path: Path = DATA_PATH,
                 output_dir: Path = REPORTS_DIR / "delivery_training",
                 model_dir: Path = ARTIFACTS_DIR / "delivery_candidates",
                 cfg: DeliveryConfig | None = None, dry_run=False,
                 allow_stale=False):
    cfg = cfg or DeliveryConfig()
    cfg.validate()
    torch.set_num_threads(min(4, os.cpu_count() or 1))
    data_path = Path(data_path)
    if not data_path.is_file():
        raise FileNotFoundError("Missing CSV; run scripts.download_data")
    raw = load_prices(data_path)
    first, last = raw.index[0], raw.index[-1]
    if first > pd.Timestamp("2000-03-01"):
        raise ValueError("Dataset must start near January 2000")
    lag = (datetime.now(timezone.utc).date() - last.date()).days
    if lag > 10 and not allow_stale:
        raise ValueError(f"Outdated Brent dataset (last observation {last.date()}); refresh first")
    ex = build_examples(raw, window=cfg.window)
    folds = SELECTION_FOLDS[:cfg.n_folds]
    plan = []
    for cut, val_end in folds:
        tr, tm, va, vm = split_ids(ex, cut, val_end)
        plan.append({"train_cutoff": cut, "validation_end": val_end,
                     "n_training_examples": len(tr),
                     "mature_train": {str(h): int(tm[:,j].sum())
                                      for j,h in enumerate(HORIZONS)},
                     "n_validation_by_horizon": {str(h): int(vm[:,j].sum())
                                                  for j,h in enumerate(HORIZONS)}})
    provenance = {
        "data_sha256": sha256(data_path.read_bytes()).hexdigest(),
        "first_date": str(first.date()), "last_date": str(last.date()),
        "n_price_rows": len(raw),
        "input_columns_available": ex.features.columns.tolist(),
        "stale_days": lag,
        "warning": "Exogenous source revisions and publication times not point-in-time certified."
    }
    if dry_run:
        return {"mode": "dry-run", "provenance": provenance,
                "folds": plan, "config": asdict(cfg),
                "no_training_performed": True}
    fold_results = []
    for kind in cfg.architectures:
        for cut, val_end in folds:
            inner = train_inner(ex, kind, cut, cfg)
            model, scaler, trainmeta = refit(ex, kind, cut, cfg, inner["epoch"])
            tr, tm, va, vm = split_ids(ex, cut, val_end)
            metrics = evaluate_refit(ex, model, scaler, va, vm, inner["alpha"])
            fold_results.append({
                "architecture": kind, "train_cutoff": cut,
                "validation_end": val_end, "inner": inner,
                "outer_train": trainmeta,
                "n_inputs": len(scaler.columns),
                "n_parameters": sum(p.numel() for p in model.parameters()),
                "metrics": metrics,
            })
            print(f"{kind.upper()} {cut}->{val_end}: " +
                  " | ".join(f"{h}: {metrics[str(h)]['improvement_pct']:+.2f}%"
                             for h in HORIZONS), flush=True)
    selection = {}
    for kind in cfg.architectures:
        rows = [f for f in fold_results if f["architecture"] == kind]
        ratios = [f["metrics"][str(h)]["relative_rmse"]
                  for f in rows for h in HORIZONS]
        selection[kind] = {
            "validation_mean_relative_rmse": float(statistics.mean(ratios)),
            "folds": len(rows),
            "folds_horizon_better_than_naive": {
                str(h): sum(f["metrics"][str(h)]["relative_rmse"] < 1
                            for f in rows) for h in HORIZONS},
        }
    winner = min(cfg.architectures,
                 key=lambda x: selection[x]["validation_mean_relative_rmse"])
    # Temporal evaluation: fit ONLY through 2023, then evaluate origins
    # from 2024 onward against prices subsequently observed per horizon.
    # ALL architectures are evaluated, and results are never used for
    # architecture selection (the 2024+ period was already examined).
    from brent_intelligence.delivery_validation import compare_after_cutoff
    test_cutoff = "2023-12-29"
    teacher_holdout, holdout_predictions = compare_after_cutoff(
        ex, cfg, cutoff=test_cutoff
    )
    # Full-data candidates: use ALL labels matured as of latest available date.
    # A separate inner validation determines final epochs and shrinkage.
    latest_cutoff = str(last.date())
    final_models = {}
    out_root = Path(output_dir)
    run_id = (datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
              + "_" + provenance["data_sha256"][:8])
    out = out_root / run_id
    out.mkdir(parents=True, exist_ok=False)
    saved_models = Path(model_dir) / run_id
    saved_models.mkdir(parents=True, exist_ok=False)
    for kind in cfg.architectures:
        inner = train_inner(ex, kind, latest_cutoff, cfg)
        model, scaler, meta = refit(ex, kind, latest_cutoff, cfg, inner["epoch"])
        model.eval()
        torch.save(model.state_dict(), saved_models / f"{kind}.pt")
        latest_x = scaler.x(ex.features)[-cfg.window:][None, :, :]
        raw_logits = _infer(model, latest_x)[0] * np.asarray(scaler.y_std)
        predicted = float(ex.prices[-1]) * np.exp(
            np.clip(raw_logits * np.asarray(inner["alpha"]), -4., 4.))
        final_models[kind] = {
            "architecture": kind, "n_parameters": sum(p.numel() for p in model.parameters()),
            "weights": f"{kind}.pt", "trained_until": latest_cutoff,
            "n_input_features": len(scaler.columns),
            "input_columns": scaler.columns, "preprocessing": asdict(scaler),
            "training": meta, "final_inner": inner,
            "last_observed_brent": float(ex.prices[-1]),
            "prices": {str(h): float(predicted[j]) for j, h in enumerate(HORIZONS)},
            "forecast_horizons_in_observed_sessions": list(HORIZONS),
        }
    result = {
        "format_version": 1, "mode": "complete-delivery-candidates",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "data": provenance, "config": asdict(cfg),
        "fold_plan": plan, "selection_folds": fold_results,
        "architecture_selection": selection,
        "selected_architecture": winner,
        "selection_criterion": "Lowest unweighted mean fold-and-horizon RMSE/naive",
        "historical_test_diagnostic": teacher_holdout,
        "full_data_models": final_models,
        "production_changed": False,
        "limitations": [
            "All 3 models refit on all mature labels through the latest observed date.",
            "For 5 and 63 sessions, more recent labels are used than for 252.",
            "Selection periods have been previously researched; not a pristine test.",
            "2024+ diagnostics are not used to select architecture or hyperparameters.",
            "Final training may include 2024+ mature labels, so it has no independent test.",
            "Some exogenous historical columns are revised or not point-in-time certified.",
            "Overlapping future horizons create dependent forecast errors.",
            "The frozen production model, app and active.json are never modified.",
        ]
    }
    from brent_intelligence.delivery_validation import save_teacher_evaluation
    saved_metrics = save_teacher_evaluation(
        teacher_holdout, holdout_predictions, out
    )
    result["teacher_criteria"] = {
        "chronological_cutoff": test_cutoff,
        "out_of_time_start": teacher_holdout["forecast_origins_start"],
        "r2_mae_mse_rmse_reported": True,
        "individual_predictions_recorded": len(holdout_predictions),
        "metric_rows": len(saved_metrics),
        "files": [
            "holdout_metrics.csv", "holdout_predictions.csv",
            "real_vs_predicted.html", "real_vs_predicted_scatter.html",
            "learning_curves.html"
        ],
        "production_and_active_model_untouched": True
    }
    (out / "summary.json").write_text(json.dumps(
        result, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    for kind in cfg.architectures:
        (saved_models / f"{kind}_metadata.json").write_text(
            json.dumps({"data_sha256": provenance["data_sha256"],
                        "config": asdict(cfg), **final_models[kind]},
                       indent=2, ensure_ascii=False, allow_nan=False),
            encoding="utf-8")
    with (out / "fold_metrics.csv").open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=[
            "architecture", "train_cutoff", "validation_end",
            "horizon", "n", "mae", "rmse", "naive_rmse",
            "improvement_pct", "relative_rmse", "alpha", "epochs"
        ])
        writer.writeheader()
        for row in fold_results:
            for j, h in enumerate(HORIZONS):
                metric = row["metrics"][str(h)]
                writer.writerow({
                    "architecture": row["architecture"], "train_cutoff": row["train_cutoff"],
                    "validation_end": row["validation_end"], "horizon": h,
                    **{k: metric[k] for k in ("n", "mae", "rmse", "naive_rmse",
                                              "improvement_pct", "relative_rmse")},
                    "alpha": row["inner"]["alpha"][j],
                    "epochs": row["inner"]["epoch"],
                })
    # Lightweight offline HTML plots via bundled Plotly dependency.
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    fig = make_subplots(rows=len(fold_results), cols=1,
                        subplot_titles=[f"{x['architecture']} through {x['train_cutoff']}"
                                        for x in fold_results])
    for i, item in enumerate(fold_results, 1):
        hist = item["inner"]["history"]
        fig.add_trace(go.Scatter(x=[x["epoch"] for x in hist],
                                 y=[x["train_smooth_l1"] for x in hist],
                                 name="Train SmoothL1", legendgroup="train",
                                 showlegend=i==1), row=i, col=1)
        fig.add_trace(go.Scatter(x=[x["epoch"] for x in hist],
                                 y=[x["validation_smooth_l1"] for x in hist],
                                 name="Validation SmoothL1", legendgroup="val",
                                 showlegend=i==1), row=i, col=1)
    fig.update_layout(height=max(700, 210*len(fold_results)),
                      title="Full-model training: time-purged inner losses")
    fig.write_html(str(out / "learning_curves.html"), include_plotlyjs=True)
    md = [
        "# Brent Intelligence — entrenamiento completo para entrega",
        "",
        f"Ultima fecha observada: **{last.date()}**, desde **{first.date()}**.",
        f"Arquitectura elegida por validacion: **{winner.upper()}**.",
        "",
        "| Arquitectura | 5 sesiones (folds ganados) | 63 sesiones | 252 sesiones | RMSE relativo medio |",
        "|---|---:|---:|---:|---:|",
    ]
    for kind in cfg.architectures:
        row = selection[kind]
        won = row["folds_horizon_better_than_naive"]
        md.append(f"| {kind.upper()} | {won['5']}/{row['folds']} | "
                  f"{won['63']}/{row['folds']} | {won['252']}/{row['folds']} | "
                  f"{row['validation_mean_relative_rmse']:.4f} |")
    md += [
        "",
        "## Entrega",
        "Todos los modelos finales se han entrenado con el dataset completo",
        "y todas las etiquetas maduradas, sin mezclar los objetivos desconocidos.",
        "El modelo congelado y el dashboard publicado siguen sin cambios.",
        "",
        "## Cautelas",
        "El test 2024+ ya se ha visto anteriormente y NO es un test virgen.",
        "Consultar summary.json para errores detallados y métricas direccionales.",
        "Los hiperparametros son fijos; la arquitectura se elige únicamente en folds antiguos.",
        "Nunca desplegar automáticamente un modelo que no supere al ingenuo.",
        "Curvas y folds: learning_curves.html, fold_metrics.csv.",
        "Evaluacion cronologica del profesor (2023 corte, 2024+):",
        "holdout_metrics.csv, holdout_predictions.csv, real_vs_predicted.html.",
        "R2, MAE, MSE y RMSE se calculan contra valores reales posteriores",
        "sin reentrenamiento con etiquetas 2024+ del modelo evaluado.",
    ]
    (out / "REPORT.md").write_text("\n".join(md)+"\n", encoding="utf-8")
    result["saved_to"] = str(out)
    result["candidate_weights_dir"] = str(saved_models)
    return result
