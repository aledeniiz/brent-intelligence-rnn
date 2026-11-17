"""Inference from a SAVED complete model: no fit, optimizer or retraining.

Supports daily updated FRED CSVs and manually updated CSVs in the same
Date/brent/exogenous source format. Model weights and fitted scalers do
not change when new rows arrive.
"""
from __future__ import annotations

from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from brent_intelligence.config import DATA_PATH, ARTIFACTS_DIR
from brent_intelligence.delivery_training import (
    DeliveryConfig, DeliveryForecaster, HORIZONS, Scaler
)
from brent_intelligence.features import load_prices, make_features


def forecast_candidate(*, summary_path: Path,
                       data_path: Path = DATA_PATH,
                       weights_dir: Path | None = None,
                       architecture: str | None = None) -> dict:
    """Run model.forward() on the latest rows, using published learned weights.

    A previously saved complete training run is required; this function
    cannot train, refit, update scalers or alter any state on disk.
    """
    summary_path = Path(summary_path)
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if summary.get("mode") != "complete-delivery-candidates":
        raise ValueError("The provided report is not a full delivery training run")
    kind = architecture or summary["selected_architecture"]
    candidates = summary["full_data_models"]
    if kind not in candidates:
        raise ValueError(f"Unknown architecture in saved run: {kind}")
    config = DeliveryConfig(**summary["config"])
    config.validate()
    model_info = candidates[kind]
    if model_info["architecture"] != kind:
        raise ValueError("Architecture/weights metadata mismatch")
    if weights_dir is None:
        weights_dir = (ARTIFACTS_DIR / "delivery_candidates"
                       / summary_path.parent.name)
    weights_dir = Path(weights_dir)
    weight_path = weights_dir / model_info["weights"]
    if not weight_path.is_file():
        raise FileNotFoundError(f"Missing frozen candidate weights: {weight_path}")
    expected_sha = model_info.get("weights_sha256")
    actual_sha = sha256(weight_path.read_bytes()).hexdigest()
    if expected_sha and actual_sha != expected_sha:
        raise ValueError("Saved model weights changed since training")

    raw = load_prices(data_path)
    training_end = pd.Timestamp(model_info["trained_until"])
    last = raw.index[-1]
    if last < training_end:
        raise ValueError(
            "Cannot forecast with prices older than the model training cutoff"
        )
    if len(raw) < config.window:
        raise ValueError("Insufficient observed dates for 60-session input window")
    feat = make_features(raw)
    prep = Scaler(**model_info["preprocessing"])
    missing = sorted(set(prep.columns) - set(feat.columns))
    if missing:
        raise ValueError("Required input columns missing: " + ", ".join(missing))
    recent = feat[prep.columns].tail(config.window)
    if (recent.notna().mean() < 0.50).any():
        raise ValueError("One or more input features have insufficient recent coverage")
    if recent.iloc[-1].isna().any():
        raise ValueError("Latest row lacks one or more mandatory input features")
    model = DeliveryForecaster(kind, len(prep.columns), config)
    model.load_state_dict(torch.load(
        weight_path, map_location="cpu", weights_only=True
    ))
    model.eval()
    input_scaled = prep.x(feat)[-config.window:].astype(np.float32)
    with torch.no_grad():
        predicted = model(torch.from_numpy(
            input_scaled[None, :, :]
        )).cpu().numpy()[0]
    alpha = np.asarray(model_info["final_inner"]["alpha"], dtype=float)
    y_std = np.asarray(prep.y_std, dtype=float)
    if alpha.shape != (3,) or y_std.shape != (3,):
        raise ValueError("Saved model horizon metadata incompatible")
    log_returns = np.clip(predicted * y_std * alpha, -4., 4.)
    base = float(raw["brent"].iloc[-1])
    prices = base * np.exp(log_returns)
    if not np.isfinite(prices).all() or np.any(prices <= 0):
        raise ValueError("Invalid price prediction")
    return {
        "format_version": 1,
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "latest_observed_date": last.date().isoformat(),
        "model_trained_through": training_end.date().isoformat(),
        "architecture": kind,
        "weights_sha256": actual_sha,
        "dataset_sha256": sha256(Path(data_path).read_bytes()).hexdigest(),
        "input_features": prep.columns,
        "horizons_in_observed_sessions": list(HORIZONS),
        "last_observed_price": base,
        "predictions": {str(h): float(prices[j])
                        for j, h in enumerate(HORIZONS)},
        "naive_predictions": {str(h): base for h in HORIZONS},
        "retrained": False,
        "description": (
            "Pure forward inference on most recent observed prices using "
            "previously trained and fixed network and scaler. "
            "No actual future prices are available yet."
        ),
    }
