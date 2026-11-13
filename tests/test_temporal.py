import numpy as np
import pandas as pd
from brent_intelligence.features import make_features
from brent_intelligence.temporal import make_examples, make_plan

def synthetic():
    rng = np.random.default_rng(11)
    idx = pd.bdate_range("2000-01-03", "2026-11-01")
    prices = 75 * np.exp(np.cumsum(rng.normal(0, .01, len(idx))))
    return pd.DataFrame({"brent": prices, "wti": prices - 4,
                         "vix": 18 + rng.uniform(0, 6, len(idx))}, index=idx)

def test_labels_mature_before_cutoff():
    raw = synthetic()
    ex = make_examples(make_features(raw), raw["brent"])
    plan = make_plan(ex)
    assert len(plan["folds"]) == 5
    assert (ex.targets > ex.anchors[:, None]).all()
    for fold in plan["folds"]:
        assert (ex.dates.take(ex.targets[fold["train"]].max(axis=1)) <= pd.Timestamp(fold["train_end"])).all()
        assert (ex.dates.take(ex.anchors[fold["validation"]]) > pd.Timestamp(fold["train_end"])).all()
        assert (ex.dates.take(ex.targets[fold["validation"]].max(axis=1)) <= pd.Timestamp(fold["val_end"])).all()
    assert len(plan["calibration"]) > 20 and len(plan["test"]) > 20

def test_exogenous_same_day_hidden():
    raw = synthetic()
    before = make_features(raw)["vix_log_level"].iloc[-1]
    raw.iloc[-1, raw.columns.get_loc("vix")] = 9999
    after = make_features(raw)["vix_log_level"].iloc[-1]
    assert before == after

def test_invalid_calendar_rejected():
    raw = synthetic()
    f = make_features(raw)
    with np.testing.assert_raises(ValueError):
        make_examples(f.iloc[::-1], raw["brent"])
