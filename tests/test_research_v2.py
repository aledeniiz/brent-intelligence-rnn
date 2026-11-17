import numpy as np
import pandas as pd
from scripts.research_v2 import _folds_for_horizon,_inner_split,run
from brent_intelligence.features import make_features
from brent_intelligence.temporal import make_examples


def fixture(tmp_path):
    date=pd.bdate_range("2000-01-03","2026-11-10")
    rng=np.random.default_rng(15)
    brent=70*np.exp(np.cumsum(rng.normal(0,0.006,len(date))))
    data=pd.DataFrame({"Date":date,"brent":brent})
    path=tmp_path/"prices.csv"
    data.to_csv(path,index=False)
    return path,data.set_index("Date")


def test_horizon_specific_matured_labels_and_same_outer_origin_dates(tmp_path):
    _,raw=fixture(tmp_path)
    features=make_features(raw)
    outer={}
    for h in (5,63,252):
        ex=make_examples(features,raw["brent"],horizons=(h,))
        folds=list(_folds_for_horizon(ex))
        assert len(folds)==5
        outer[h]=[]
        for fold in folds:
            train_end=pd.Timestamp(fold["cutoff"])
            val_end=pd.Timestamp(fold["validation_end"])
            assert (ex.dates.take(ex.targets[fold["train"],0])<=train_end).all()
            assert (ex.dates.take(ex.anchors[fold["validation"]])>train_end).all()
            assert (ex.dates.take(ex.targets[fold["validation"],0])<=val_end).all()
            origins=tuple(ex.dates.take(ex.anchors[fold["validation"]]))
            outer[h].append(origins)
            cut,inner_train,inner_val=_inner_split(ex,fold["train"],fold["cutoff"])
            assert (ex.dates.take(ex.targets[inner_train,0])<=pd.Timestamp(cut)).all()
            assert (ex.dates.take(ex.anchors[inner_val])>pd.Timestamp(cut)).all()
            assert not set(inner_train)&set(inner_val)
    assert outer[5]==outer[63]==outer[252]


def test_validation_only_ridge_and_dry_run_do_not_publish_model(tmp_path):
    path,raw=fixture(tmp_path)
    reports=tmp_path/"reports"
    plan=run(data=path,output=reports,mode="ridge",
             feature_sets=("brent_only",),horizons=(5,),folds=1,
             dry_run=True)
    assert plan["used_test"] is False
    assert plan["used_calibration"] is False
    assert not reports.exists()
    summary=run(data=path,output=reports,mode="ridge",
                feature_sets=("brent_only",),horizons=(5,),folds=1)
    assert summary["used_test"] is False
    assert len(summary["results"])==1
    first=summary["results"][0]
    assert first["n_inner_validation"]>20
    assert first["n_train"]>first["n_inner_train"]
    assert first["metrics"]["rmse"]>=0
    assert (reports.exists() and
            len(list(reports.glob("validation_*.json")))==1)
    assert not (tmp_path/"artifacts").exists()
    assert not (tmp_path/"deployment").exists()
    assert not (tmp_path/"public").exists()
