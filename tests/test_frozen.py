from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from brent_intelligence.frozen import (
    export_frozen, file_hash, forecast_frozen, read_json,
    save_json_atomic, validate_bundle,
)
from brent_intelligence.modeling import RecurrentForecaster
from scripts.plot_diagnostics import build_loss_figure


def example_data(path):
    dates = pd.bdate_range(end=datetime.now(timezone.utc).date(), periods=110)
    rng = np.random.default_rng(123)
    price = 75 * np.exp(np.cumsum(rng.normal(0, 0.005, len(dates))))
    pd.DataFrame({"Date": dates, "brent": price}).to_csv(path, index=False)


def bundle(tmp_path):
    folder = tmp_path / "frozen"
    folder.mkdir()
    model = RecurrentForecaster("lstm", 2, 8, 1, 3)
    torch.save(model.state_dict(), folder / "production.pt")
    prep = {"columns": ["brent_log_level", "brent_ret_1"],
            "medians": [4.3, 0.0], "x_mean": [4.3, 0.0], "x_std": [1.0, 1.0],
            "y_mean": [0.0, 0.0, 0.0], "y_std": [1.0, 1.0, 1.0]}
    save_json_atomic(folder / "preprocessing.json", prep)
    manifest = {"format_version": 1, "architecture": "lstm",
                "input_features": prep["columns"], "weights_sha256": file_hash(folder / "production.pt"),
                "config": {"window": 60, "horizons": [5, 63, 252],
                           "hidden_size": 8, "layers": 1},
                "trained_data_end": "2026-11-01",
                "calibration": {"nominal_coverage": 0.8, "log_error_quantiles": [0.1, 0.2, 0.3],
                                "historical_test_coverage": {"5": 0.9, "63": 0.9, "252": 0.9}},
                "evaluation": {"all_models_historical_test": {
                    "lstm": {str(h): {"mae": 1.0, "rmse": 2.0, "naive_rmse": 2.5,
                                      "relative_rmse": 0.8}
                             for h in (5, 63, 252)}}},
                "caveats": ["Modelo experimental"]}
    save_json_atomic(folder / "manifest.json", manifest)
    return folder


def test_frozen_forecast_no_training_and_update_same_day_revision(tmp_path):
    file = tmp_path / "prices.csv"
    example_data(file)
    frozen = bundle(tmp_path)
    out = tmp_path / "latest_forecast.json"
    result = forecast_frozen(data=file, bundle=frozen, output=out)
    assert result["changed"]
    assert result["forecast"]["model"]["architecture"] == "lstm"
    assert len(result["forecast"]["prices"]) == 3
    assert all(lo < p < hi for lo, p, hi in zip(
        result["forecast"]["lower"], result["forecast"]["prices"], result["forecast"]["upper"]))
    again = forecast_frozen(data=file, bundle=frozen, output=out)
    assert not again["changed"]
    original_sha = read_json(out)["data_sha256"]
    frame = pd.read_csv(file)
    frame.loc[len(frame)-1, "brent"] += 1.0
    frame.to_csv(file, index=False)
    updated = forecast_frozen(data=file, bundle=frozen, output=out)
    assert updated["changed"]  # same date, revised price must be republished
    assert read_json(out)["data_sha256"] != original_sha


def test_corrupted_weights_and_missing_features_block_refresh(tmp_path):
    file = tmp_path / "prices.csv"
    example_data(file)
    frozen = bundle(tmp_path)
    (frozen / "production.pt").write_bytes(b"corrupted")
    with pytest.raises(ValueError, match="SHA256"):
        validate_bundle(frozen)
    frozen = bundle(tmp_path)
    manifest = read_json(frozen / "manifest.json")
    manifest["input_features"] = ["brent_log_level", "missing_source"]
    save_json_atomic(frozen / "manifest.json", manifest)
    with pytest.raises(ValueError):
        forecast_frozen(data=file, bundle=frozen, output=tmp_path / "x.json")


def test_loss_curves_use_stored_histories():
    folds = [{"history": [
        {"epoch": 1, "train_loss": 0.4, "val_loss": 0.5},
        {"epoch": 2, "train_loss": 0.3, "val_loss": 0.6}
    ]} for _ in range(5)]
    fig = build_loss_figure("lstm", folds)
    assert len(fig.data) == 10
    assert all(len(trace.x) == 2 for trace in fig.data)


def test_model_export_without_training_checks_data_fingerprint(tmp_path):
    data = tmp_path / "prices.csv"
    example_data(data)
    artifacts = tmp_path / "artifacts"
    run = artifacts / "runs" / "frozen-source"
    run.mkdir(parents=True)
    save_json_atomic(artifacts / "active.json", {"run": "runs/frozen-source"})
    model = RecurrentForecaster("lstm", 2, 8, 1, 3)
    torch.save(model.state_dict(), run / "production.pt")
    scaler = {"columns": ["brent_log_level", "brent_ret_1"],
              "medians": [4.3, 0.0], "x_mean": [4.3, 0.0], "x_std": [1.0, 1.0],
              "y_mean": [0.0]*3, "y_std": [1.0]*3}
    save_json_atomic(run / "production_preprocessing.json", scaler)
    metrics = {str(h): {"rmse": 2.0} for h in (5, 63, 252)}
    summary = {
        "mode": "historical-evaluation", "data_sha256": file_hash(data),
        "config": {"window": 60, "horizons": [5, 63, 252],
                   "hidden_size": 8, "layers": 1},
        "selection": {"architecture": "lstm", "criterion": "val ratio"},
        "folds": [{}, {}, {}, {}, {}],
        "uncertainty": {"nominal": .8, "log_error_quantiles": [.1, .2, .3],
                        "test_coverage": {"5": .9, "63": .9, "252": .9},
                        "n_calibration": 251},
        "models": {"lstm": {"test": metrics}},
        "production": {"last_known_price_date": str(pd.Timestamp.now().date()),
                       "last_matured_training_origin": "2025-01-01",
                       "n_examples": 1000}
    }
    save_json_atomic(run / "summary.json", summary)
    dest = tmp_path / "bundle"
    assert export_frozen(artifacts=artifacts, data=data, destination=dest)["architecture"] == "lstm"
    assert validate_bundle(dest)[0]["weights_sha256"] == file_hash(dest / "production.pt")
    data.write_text(data.read_text()+"\n")
    with pytest.raises(ValueError, match="CSV distinto"):
        export_frozen(artifacts=artifacts, data=data, destination=dest)
