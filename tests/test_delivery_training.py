"""Temporal maturity, lookahead, full refit and end-to-end smoke."""
from dataclasses import replace
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
import torch
from brent_intelligence.delivery_training import (
    DeliveryConfig, DeliveryForecaster, build_examples, split_ids,
    inner_split, Scaler, run_delivery
)

@pytest.fixture
def synthetic(tmp_path):
    dates = pd.bdate_range("2000-01-03", "2026-11-16")
    rng = np.random.default_rng(29)
    prices = 70 * np.exp(np.cumsum(rng.normal(.00007, .011, len(dates))))
    frame = pd.DataFrame({
        "brent": prices, "wti": prices - 3.,
        "vix": 20. + rng.uniform(0., 3., len(dates)),
        "usd_por_eur": 1. + rng.uniform(0., .2, len(dates))
    }, index=dates)
    frame.index.name = "Date"
    path = tmp_path / "brent.csv"
    frame.to_csv(path)
    return path, frame

def test_horizon_maturity_uses_recent_labels(synthetic):
    _, frame = synthetic
    ex = build_examples(frame)
    train, masks, val, valmask = split_ids(ex, "2023-12-29", "2026-11-16")
    last = [ex.anchors[train[masks[:,j]]][-1] for j in range(3)]
    assert last[0] > last[1] > last[2]
    n = masks.sum(axis=0)
    assert n[0] > n[1] > n[2]
    end = ex.dates.searchsorted("2023-12-29", side="right") - 1
    assert (ex.target_indices[train][masks] <= end).all()
    assert (ex.anchors[val] > end).all()
    assert (valmask.sum(axis=0) >= 20).all()
    cut, ti, tm, vi, vm = inner_split(ex, "2023-12-29")
    assert cut < "2023-12-29"
    assert tm.shape == (len(ti), 3)
    assert vm.shape == (len(vi), 3)

def test_no_future_changes_previous_inputs(synthetic):
    _, frame = synthetic
    ex1 = build_examples(frame)
    altered = frame.copy()
    altered.loc["2024-01-01":, "brent"] *= 2.
    altered.loc["2024-01-01":, "vix"] = 900.
    ex2 = build_examples(altered)
    prior = ex1.dates <= "2023-12-29"
    np.testing.assert_allclose(ex1.features.loc[prior].fillna(-999),
                               ex2.features.loc[prior].fillna(-999))
    ti1, m1, _, _ = split_ids(ex1, "2021-12-31", "2023-12-29")
    ti2, m2, _, _ = split_ids(ex2, "2021-12-31", "2023-12-29")
    np.testing.assert_array_equal(ti1, ti2)
    np.testing.assert_array_equal(m1, m2)
    np.testing.assert_allclose(ex1.log_returns[ti1],
                               ex2.log_returns[ti2])

def test_unmatured_targets_mask_and_identity(synthetic):
    _, frame = synthetic
    ex = build_examples(frame)
    assert ex.observed[-1].tolist() == [False, False, False]
    assert (ex.log_returns[~ex.observed] == 0).all()
    cfg = DeliveryConfig()
    for kind in ("rnn", "lstm", "gru"):
        model = DeliveryForecaster(kind, 7, cfg)
        model.eval()
        with torch.no_grad():
            pred = model(torch.randn(2, 60, 7))
        assert pred.shape == (2, 3)
        assert torch.count_nonzero(pred) == 0
    tr, masks, _, _ = split_ids(ex, "2019-12-31", "2021-12-31")
    scaler = Scaler.fit(ex, tr, masks, "2019-12-31", .6)
    assert scaler.columns and min(scaler.y_std) > 0

def test_entire_delivery_no_production_changes(synthetic, tmp_path):
    path, _ = synthetic
    reports = tmp_path / "reports" / "delivery_training"
    weights = tmp_path / "artifacts" / "delivery_candidates"
    cfg = replace(DeliveryConfig(), architectures=("lstm",),
                  n_folds=1, max_epochs=1, patience=1, hidden_size=8)
    plan = run_delivery(data_path=path, output_dir=reports,
                        model_dir=weights, cfg=cfg, dry_run=True,
                        allow_stale=True)
    assert plan["no_training_performed"] and len(plan["folds"]) == 1
    assert not reports.exists() and not weights.exists()
    result = run_delivery(data_path=path, output_dir=reports,
                          model_dir=weights, cfg=cfg, allow_stale=True)
    assert result["selected_architecture"] == "lstm"
    assert result["production_changed"] is False
    final = result["full_data_models"]["lstm"]
    counts = final["training"]["n_mature_by_horizon"]
    assert counts["5"] > counts["63"] > counts["252"]
    assert final["trained_until"] == "2026-11-16"
    assert set(final["prices"]) == {"5", "63", "252"}
    assert all(p > 0 for p in final["prices"].values())
    for name in ("summary.json", "REPORT.md", "fold_metrics.csv",
                 "learning_curves.html"):
        assert (Path(result["saved_to"]) / name).is_file()
    for name in ("lstm.pt", "lstm_metadata.json"):
        assert (Path(result["candidate_weights_dir"]) / name).is_file()
    assert result["teacher_criteria"]["r2_mae_mse_rmse_reported"] is True
    assert result["teacher_criteria"]["metric_rows"] == 3
    holdout = result["historical_test_diagnostic"]
    assert holdout["train_cutoff"] == "2023-12-29"
    assert holdout["forecast_origins_start"] >= "2024-01-01"
    assert holdout["models"]["lstm"]["trained_through"] == "2023-12-29"
    for h in ("5", "63", "252"):
        m = holdout["models"]["lstm"]["metrics"][h]
        assert m["n"] >= 20
        assert np.isfinite(m["mae"]) and np.isfinite(m["mse"])
        assert np.isfinite(m["rmse"]) and np.isfinite(m["r2"])
    for name in ("holdout_metrics.csv", "holdout_predictions.csv",
                 "real_vs_predicted.html", "real_vs_predicted_scatter.html"):
        assert (Path(result["saved_to"]) / name).is_file()
    heldout = pd.read_csv(Path(result["saved_to"]) / "holdout_predictions.csv")
    assert (heldout["origin_date"] > "2023-12-29").all()
    assert (heldout["target_date"] >= heldout["origin_date"]).all()
    assert (heldout["target_date"] <= "2026-11-16").all()
    subset = heldout.loc[heldout["horizon"] == 5]
    actual = subset["actual"].to_numpy(dtype=float)
    predicted = subset["predicted"].to_numpy(dtype=float)
    manual_r2 = 1. - np.sum((actual-predicted)**2)/np.sum((actual-actual.mean())**2)
    assert manual_r2 == pytest.approx(holdout["models"]["lstm"]["metrics"]["5"]["r2"])
    from brent_intelligence.delivery_inference import forecast_candidate
    weights_dir = Path(result["candidate_weights_dir"])
    summary_path = Path(result["saved_to"]) / "summary.json"
    previous_weights = (weights_dir / "lstm.pt").read_bytes()
    old_forecast = forecast_candidate(
        summary_path=summary_path, data_path=path, weights_dir=weights_dir
    )
    assert old_forecast["retrained"] is False
    assert old_forecast["latest_observed_date"] == "2026-11-16"
    assert old_forecast["predictions"] == final["prices"]
    next_frame = synthetic[1].copy()
    future_date = pd.Timestamp("2026-11-17")
    previous = next_frame.iloc[-1].copy()
    previous["brent"] = previous["brent"] * 1.2
    next_frame.loc[future_date] = previous
    next_frame.to_csv(path)
    fresh_forecast = forecast_candidate(
        summary_path=summary_path, data_path=path, weights_dir=weights_dir
    )
    assert fresh_forecast["latest_observed_date"] == "2026-11-17"
    assert fresh_forecast["model_trained_through"] == "2026-11-16"
    assert fresh_forecast["predictions"] != old_forecast["predictions"]
    assert fresh_forecast["weights_sha256"] == old_forecast["weights_sha256"]
    assert (weights_dir / "lstm.pt").read_bytes() == previous_weights
    assert not (tmp_path / "artifacts" / "active.json").exists()
    assert not (tmp_path / "deployment").exists()
    assert not (tmp_path / "public").exists()

def test_invalid_settings_fail():
    with pytest.raises(ValueError):
        DeliveryConfig(architectures=("transformer",)).validate()
    with pytest.raises(ValueError):
        DeliveryConfig(max_epochs=0).validate()
