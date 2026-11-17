"""Teacher-facing, strictly chronological evaluation of complete recurrent models.

For each architecture: train using mature labels by 2023-12-29, predict
origins from 2024 onward, and compare with *subsequently observed*
Brent prices (only once the individual horizon has matured). This 2024+
period has already been examined in earlier research: descriptive
out-of-time evaluation, NOT a pristine untouched test.
"""
from __future__ import annotations

import csv
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from brent_intelligence.delivery_training import (
    HORIZONS, _infer, _metrics, _price_prediction,
    refit, split_ids, train_inner,
)

DEFAULT_TRAIN_CUTOFF = "2023-12-29"


def compare_after_cutoff(ex, cfg, *, cutoff=DEFAULT_TRAIN_CUTOFF):
    """Evaluate complete RNN/LSTM/GRU on later prices, never training on them."""
    last = str(ex.dates[-1].date())
    if pd.Timestamp(cutoff) >= pd.Timestamp(last):
        raise ValueError("Evaluation cutoff must precede the observed series end")
    _, _, valid, masks = split_ids(ex, cutoff, last)
    original_prices = ex.prices[ex.anchors[valid]]
    original_dates = ex.dates.take(ex.anchors[valid])
    results, rows = {}, []
    cutoff_index = int(ex.dates.searchsorted(pd.Timestamp(cutoff), side="right") - 1)

    for kind in cfg.architectures:
        # Nested validation cutoff <= 2023; neither checkpoint nor alpha
        # uses any labels or features after 2023.
        selected = train_inner(ex, kind, cutoff, cfg)
        model, scaler, training = refit(
            ex, kind, cutoff, cfg, selected["epoch"]
        )
        model.eval()
        logits = _infer(model, ex.windows(scaler.x(ex.features), valid))
        forecasts = _price_prediction(
            ex, valid, logits, scaler, selected["alpha"]
        )
        if any(pd.Timestamp(x) > pd.Timestamp(cutoff)
               for x in [training["last_label_date"]]):
            raise AssertionError("Future labels were used in holdout fitting")
        # Temporal maturity is enforced in refit + Scaler.fit.
        # No 2024+ target is eligible when cutoff is in 2023.

        metrics = _metrics(ex, valid, masks, forecasts)
        results[kind] = {
            "trained_through": cutoff,
            "latest_mature_train_origin": training["last_mature_origin_by_horizon"],
            "n_training_origins": training["n_unique_origins"],
            "selected_epochs_using_pre_cutoff_data": selected["epoch"],
            "selected_alpha_using_pre_cutoff_data": selected["alpha"],
            "metrics": metrics,
        }
        for i, record_id in enumerate(valid):
            for j, horizon in enumerate(HORIZONS):
                if not masks[i, j]:
                    continue
                origin = original_dates[i]
                target_idx = int(ex.target_indices[record_id, j])
                target = ex.dates[target_idx]
                if origin <= pd.Timestamp(cutoff) or target > ex.dates[-1]:
                    raise AssertionError("Invalid post-cutoff evaluation row")
                actual = float(ex.prices[target_idx])
                predicted = float(forecasts[i, j])
                naive = float(original_prices[i])
                rows.append({
                    "architecture": kind,
                    "origin_date": origin.date().isoformat(),
                    "target_date": target.date().isoformat(),
                    "horizon": horizon, "last_known_price": naive,
                    "predicted": predicted, "actual": actual,
                    "naive_prediction": naive,
                    "error": predicted - actual,
                    "absolute_error": abs(predicted - actual),
                    "baseline_absolute_error": abs(naive - actual),
                })
    return {
        "train_cutoff": cutoff,
        "forecast_origins_start": str(ex.dates[
            ex.anchors[valid[0]]].date()),
        "last_price_observation": last,
        "models": results,
        "already_seen_period": True,
        "qualification": (
            "Every evaluated price is observed after its forecast origin, "
            "with origin after the training cutoff; models are fit before "
            "this period. However, 2024+ has already been inspected in "
            "earlier experiments and cannot be treated as an untouched test."
        ),
    }, rows


def _save_csv(path: Path, records: list[dict]):
    if not records:
        raise ValueError("No mature predictions to report")
    with Path(path).open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)


def save_teacher_evaluation(report: dict, predictions: list[dict], folder: Path):
    """Generate R2/MAE/MSE/RMSE table and measured vs predicted HTML charts."""
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    _save_csv(folder / "holdout_predictions.csv", predictions)
    metrics = []
    for kind, model_report in report["models"].items():
        for horizon in HORIZONS:
            stats = model_report["metrics"][str(horizon)]
            directional = stats.get("direction") or {}
            metrics.append({
                "architecture": kind, "horizon": horizon,
                "n": stats["n"], "mae": stats["mae"], "mse": stats["mse"],
                "rmse": stats["rmse"], "r2": stats["r2"],
                "naive_rmse": stats["naive_rmse"],
                "naive_r2": stats["naive_r2"],
                "improvement_pct": stats["improvement_pct"],
                "directional_roc_auc": directional.get("roc_auc"),
                "directional_accuracy": directional.get("accuracy"),
                "directional_precision": directional.get("precision"),
                "directional_recall": directional.get("recall"),
                "directional_f1": directional.get("f1"),
            })
    _save_csv(folder / "holdout_metrics.csv", metrics)

    fig = make_subplots(
        rows=3, cols=1, shared_xaxes=False,
        subplot_titles=[
            f"Brent real vs predicho a {h} sesiones (precio objetivo observado)"
            for h in HORIZONS
        ]
    )
    for j, horizon in enumerate(HORIZONS, 1):
        subset = [row for row in predictions
                  if row["horizon"] == horizon]
        first_kind = next(iter(report["models"]))
        actual = [p for p in subset if p["architecture"] == first_kind]
        actual.sort(key=lambda p: p["target_date"])
        fig.add_trace(go.Scatter(
            x=[p["target_date"] for p in actual],
            y=[p["actual"] for p in actual], name="Real", mode="lines",
            line={"width": 3, "color": "#111111"},
            legendgroup="actual", showlegend=(j == 1)
        ), row=j, col=1)
        fig.add_trace(go.Scatter(
            x=[p["target_date"] for p in actual],
            y=[p["naive_prediction"] for p in actual],
            name="Ingenuo (ultimo precio)", mode="lines",
            line={"dash": "dot", "color": "#989898"},
            legendgroup="naive", showlegend=(j == 1)
        ), row=j, col=1)
        for kind in report["models"]:
            vals = [p for p in subset if p["architecture"] == kind]
            vals.sort(key=lambda p: p["target_date"])
            fig.add_trace(go.Scatter(
                x=[p["target_date"] for p in vals],
                y=[p["predicted"] for p in vals],
                name=kind.upper(), mode="lines",
                legendgroup=kind, showlegend=(j == 1)
            ), row=j, col=1)
        fig.update_yaxes(title_text="USD / barril", row=j, col=1)
        fig.update_xaxes(title_text="Fecha objetivo observada", row=j, col=1)
    fig.update_layout(
        height=1050, hovermode="x unified",
        title=("Evaluacion temporal: entrenamiento hasta "
               + report["train_cutoff"] +
               ", prediccion de origenes posteriores (SOLO objetivos maduros)"),
    )
    fig.write_html(str(folder / "real_vs_predicted.html"),
                   include_plotlyjs=True)

    scatter = make_subplots(rows=1, cols=3, subplot_titles=[
        f"{h} sesiones" for h in HORIZONS])
    for col, h in enumerate(HORIZONS, 1):
        for kind in report["models"]:
            vals = [row for row in predictions
                    if row["horizon"] == h and row["architecture"] == kind]
            scatter.add_trace(go.Scatter(
                x=[x["actual"] for x in vals],
                y=[x["predicted"] for x in vals],
                mode="markers", name=kind.upper(),
                legendgroup=kind, showlegend=(col == 1),
                marker={"size": 4, "opacity": 0.45},
            ), row=1, col=col)
        scatter.update_xaxes(title_text="Real (USD)", row=1, col=col)
        scatter.update_yaxes(title_text="Predicho (USD)", row=1, col=col)
    scatter.update_layout(height=420,
                          title="Dispersión real vs predicho por horizonte")
    scatter.write_html(str(folder / "real_vs_predicted_scatter.html"),
                       include_plotlyjs=True)
    return metrics
