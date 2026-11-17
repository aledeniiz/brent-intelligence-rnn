from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from brent_intelligence.research_v4 import (
    V4Config, HORIZONS, ResidualRNN, prepare_historical,
    outer_folds, inner_split, choose_shrinkage, run_experiment,
)


@pytest.fixture
def synthetic(tmp_path):
    dates = pd.bdate_range("2000-01-03", "2026-11-12")
    rng = np.random.default_rng(19)
    returns = rng.normal(0.00012, 0.009, len(dates))
    prices = 70 * np.exp(np.cumsum(returns))
    path = tmp_path / "synthetic.csv"
    pd.DataFrame({"Date": dates, "brent": prices}).to_csv(path, index=False)
    return path


def test_future_data_do_not_affect_features_labels_or_folds(synthetic, tmp_path):
    ex1, provenance1 = prepare_historical(synthetic, window=60)
    folds1 = list(outer_folds(ex1))
    assert ex1.dates.max() <= pd.Timestamp("2021-12-31")
    assert len(folds1) == 5
    for fold in folds1:
        dates = ex1.dates
        targets = dates.take(ex1.targets[fold["train"]].max(axis=1))
        validation_targets = dates.take(ex1.targets[fold["validation"]].max(axis=1))
        assert (targets <= pd.Timestamp(fold["train_end"])).all()
        assert (validation_targets <= pd.Timestamp(fold["validation_end"])).all()
        inner_cut, inner_train, inner_val = inner_split(
            ex1, fold["train"], fold["train_end"]
        )
        assert not set(inner_train).intersection(set(inner_val))
        assert (dates.take(ex1.targets[inner_train].max(axis=1))
                <= pd.Timestamp(inner_cut)).all()
        assert (dates.take(ex1.anchors[inner_val])
                > pd.Timestamp(inner_cut)).all()
    other = tmp_path / "altered.csv"
    frame = pd.read_csv(synthetic)
    frame.loc[frame["Date"] > "2021-12-31", "brent"] *= 3
    frame.to_csv(other, index=False)
    ex2, provenance2 = prepare_historical(other, window=60)
    np.testing.assert_array_equal(ex1.returns, ex2.returns)
    np.testing.assert_array_equal(ex1.features.to_numpy(), ex2.features.to_numpy())
    assert provenance1["historical_brent_sha256"] == provenance2["historical_brent_sha256"]
    assert provenance1["source_sha256"] != provenance2["source_sha256"]


def test_zero_initialized_network_and_shrinkage_to_naive(synthetic):
    cfg = V4Config()
    model = ResidualRNN(6, cfg)
    model.eval()
    assert sum(p.numel() for p in model.parameters()) < 6499
    with torch.no_grad():
        out = model(torch.randn(3, 60, 6))
    assert out.shape == (3, len(HORIZONS))
    assert torch.all(out == 0)  # exact random-walk initial state
    ex, _ = prepare_historical(synthetic, window=60)
    fold = next(iter(outer_folds(ex)))
    ids = fold["train"][-120:]
    # Very bad constant prediction should not move away from naive.
    shrink, selection = choose_shrinkage(
        ex, ids, np.full((len(ids), 3), 2.0), cfg
    )
    np.testing.assert_array_equal(shrink, 0.0)
    assert len(selection) == 3


def test_dry_run_no_artifacts_and_tiny_training(synthetic, tmp_path):
    report_dir = tmp_path / "research"
    cfg = replace(V4Config(), max_folds=1, max_epochs=2,
                  patience=1, seeds=(42,), hidden_size=8)
    plan = run_experiment(data_path=synthetic, output_dir=report_dir,
                          cfg=cfg, dry_run=True)
    assert plan["used_test_2024_onwards"] is False
    assert not report_dir.exists()
    finished = run_experiment(data_path=synthetic, output_dir=report_dir,
                              cfg=cfg)
    assert not finished["used_calibration_or_test_dates"]
    assert len(finished["folds"]) == 1
    assert len(finished["aggregate"]) == 3
    assert finished["folds"][0]["n_inner_validation"] >= 20
    assert set(finished["folds"][0]["metrics"]) == {"5", "63", "252"}
    for m in finished["aggregate"]:
        assert m["n_origins"] >= 20
        assert np.isfinite(m["rmse"])
        assert np.isfinite(m["naive_rmse"])
    saved = Path(finished["saved_to"])
    assert (saved / "learning_curves.html").is_file()
    assert (saved / "predictions.csv").is_file()
    assert (saved / "REPORT.md").is_file()
    assert not (tmp_path / "artifacts").exists()
    assert not (tmp_path / "deployment").exists()
    assert not (tmp_path / "public").exists()


def test_invalid_settings_fail_before_training():
    with pytest.raises(ValueError):
        V4Config(dropout=1.2).validate()
    with pytest.raises(ValueError):
        V4Config(seeds=()).validate()
