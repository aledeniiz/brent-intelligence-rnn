"""Independent-horizon training: maturity, leakage, metrics, inference and smoke."""
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from brent_intelligence.horizon_models import (
    HorizonConfig, HorizonScaler, DirectRecurrent, build_examples,
    ids_for_cut, outer_split, inner_split, return_to_price,
    score_prices, run_horizon_models,
)
from brent_intelligence.horizon_inference import predict_from_saved


@pytest.fixture
def synthetic(tmp_path):
    dates = pd.bdate_range("2000-01-03", "2026-11-16")
    rng = np.random.default_rng(2410)
    returns = rng.normal(0.00014, 0.011, len(dates))
    brent = 65 * np.exp(np.cumsum(returns))
    frame = pd.DataFrame({
        "brent":brent, "wti":brent-4.,
        "vix":20+rng.uniform(0.,3.,len(dates)),
        "usd_por_eur":1.+rng.uniform(0.,.2,len(dates)),
        "gas_natural":3.+rng.uniform(0.,.5,len(dates))
    },index=dates)
    frame.index.name = "Date"
    path=tmp_path/"brent.csv"
    frame.to_csv(path)
    return path, frame


def test_per_horizon_uses_all_mature_labels_and_purges(synthetic):
    _, raw=synthetic
    ex=build_examples(raw,window=60)
    cut="2023-12-29"
    train_by_h=[]
    for h in (5,63,252):
        train=ids_for_cut(ex,h,cut)
        j=(5,63,252).index(h)
        end=ex.dates.searchsorted(cut,side="right")-1
        assert len(train)>100
        assert (ex.target_indices[train,j]<=end).all()
        train_by_h.append(train)
        train2,valid=outer_split(ex,h,cut,"2026-11-16")
        np.testing.assert_array_equal(train,train2)
        assert (ex.anchors[valid]>end).all()
        assert (ex.target_indices[valid,j]<len(ex.dates)).all()
        inner_cut,inner_train,inner_val=inner_split(ex,h,cut)
        assert inner_cut<cut and len(inner_train)>100 and len(inner_val)>20
        assert (ex.target_indices[inner_train,j] <=
                ex.dates.searchsorted(inner_cut,side="right")-1).all()
        scale=HorizonScaler.fit(ex,train,h,cut,.6)
        assert scale.columns and scale.target_rms>0
    assert len(train_by_h[0])>len(train_by_h[1])>len(train_by_h[2])
    assert ex.anchors[train_by_h[0][-1]] > ex.anchors[train_by_h[1][-1]]
    assert ex.anchors[train_by_h[1][-1]] > ex.anchors[train_by_h[2][-1]]


def test_future_changes_do_not_affect_past_features(synthetic):
    _, original=synthetic
    a=build_examples(original)
    other=original.copy()
    other.loc["2024-01-01":,"brent"]*=2
    other.loc["2024-01-01":,"vix"]*=3
    b=build_examples(other)
    past=a.dates<="2023-12-29"
    np.testing.assert_allclose(
        a.features.loc[past].fillna(-999),
        b.features.loc[past].fillna(-999))
    for h in (5,63,252):
        ia=ids_for_cut(a,h,"2023-12-29")
        ib=ids_for_cut(b,h,"2023-12-29")
        np.testing.assert_array_equal(ia,ib)
        j=(5,63,252).index(h)
        np.testing.assert_allclose(
            a.log_returns[ia,j],b.log_returns[ib,j])


def test_naive_initialized_and_metrics_distinguish_r2(synthetic):
    cfg=HorizonConfig()
    for arch in cfg.architectures:
        m=DirectRecurrent(arch,4,cfg)
        m.eval()
        with torch.no_grad():
            pred=m(torch.randn(5,60,4))
        assert pred.shape==(5,)
        assert torch.all(pred==0)
    anchor=np.array([100.,100.,100.,100.])
    actual=np.array([90.,105.,110.,95.])
    pred=return_to_price(anchor,np.zeros(4),.2,1.)
    np.testing.assert_allclose(pred,anchor)
    metric=score_prices(anchor,actual,pred)
    assert metric["rmse_ratio"]==pytest.approx(1.)
    assert metric["improvement_pct"]==pytest.approx(0.)
    assert metric["r2"]==pytest.approx(metric["naive_r2"])
    assert metric["direction"]["roc_auc"]==pytest.approx(.5)
    assert metric["direction"]["recall"]==0.


def test_training_then_forward_only_updated_csv(synthetic,tmp_path):
    path, raw=synthetic
    cfg=replace(HorizonConfig(),architectures=("rnn",),horizons=(5,),
                folds=1,epochs=1,patience=1,hidden_size=8)
    report=tmp_path/"reports"
    weights=tmp_path/"candidates"
    dry=run_horizon_models(data_path=path,report_dir=report,
                           model_dir=weights,cfg=cfg,dry_run=True)
    assert dry["trained"] is False
    assert len(dry["folds"])==1
    assert not report.exists() and not weights.exists()
    run=run_horizon_models(data_path=path,report_dir=report,
                          model_dir=weights,cfg=cfg)
    assert run["previous_production_model_untouched"]
    assert run["final_models"]["5"]["train"]["n_train"]>6000
    assert run["final_models"]["5"]["train"]["last_training_label_date"]=="2026-11-16"
    assert run["historical_diagnostic"]["5"]["metric"]["n"]>100
    output=Path(run["output"])
    for f in ("summary.json","fold_metrics.csv","validation_predictions.csv",
              "historical_diagnostic_metrics.csv",
              "historical_diagnostic_predictions.csv",
              "learning_curves.html","real_vs_predicted.html","REPORT.md"):
        assert (output/f).is_file()
    df=pd.read_csv(output/"historical_diagnostic_predictions.csv")
    assert (df["origin_date"]>"2023-12-29").all()
    assert (df["target_date"]<=str(raw.index[-1].date())).all()
    assert (df["target_date"]>df["origin_date"]).all()
    m=run["historical_diagnostic"]["5"]["metric"]
    assert m["n"]==len(df)
    assert m["rmse"]==pytest.approx(float(np.sqrt(
        np.mean((df["predicted"]-df["actual"])**2))))
    state=Path(run["candidate_weights"])/"h5.pt"
    before=state.read_bytes()
    old=predict_from_saved(summary_path=output/"summary.json",
                           data_path=path,
                           weights_dir=Path(run["candidate_weights"]))
    assert old["retrained"] is False
    assert old["predictions"]["5"]==pytest.approx(
        run["final_models"]["5"]["predicted_price"],rel=1e-5)
    recent=raw.iloc[-1].copy()
    recent["brent"]*=1.2
    raw.loc[pd.Timestamp("2026-11-17")]=recent
    raw.to_csv(path)
    new=predict_from_saved(summary_path=output/"summary.json",
                           data_path=path,
                           weights_dir=Path(run["candidate_weights"]))
    assert new["latest_observed_price_date"]=="2026-11-17"
    assert new["predictions"]!=old["predictions"]
    assert new["architectures"]["5"]["weights_sha256"]==old["architectures"]["5"]["weights_sha256"]
    assert state.read_bytes()==before
    assert not (tmp_path/"artifacts"/"active.json").exists()
    assert not (tmp_path/"deployment").exists()
    assert not (tmp_path/"public").exists()


def test_invalid_config_and_future_labels_rejected(synthetic):
    _,raw=synthetic
    ex=build_examples(raw)
    with pytest.raises(ValueError):
        HorizonConfig(epochs=0).validate()
    with pytest.raises(ValueError):
        HorizonConfig(architectures=("transformer",)).validate()
    with pytest.raises(ValueError):
        HorizonConfig(horizons=(7,)).validate()
    future_ids=ids_for_cut(ex,5,"2026-11-16")
    with pytest.raises(ValueError):
        HorizonScaler.fit(ex,future_ids,5,"2021-12-31",.6)
