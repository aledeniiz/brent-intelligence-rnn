from pathlib import Path
import numpy as np
import pandas as pd
from brent_intelligence.config import ExperimentConfig
from brent_intelligence.experiment import Preprocessor,predict_latest,run_experiment
from brent_intelligence.features import make_features
from brent_intelligence.temporal import make_examples,make_plan

def dataset(path:Path):
    idx=pd.bdate_range("2000-01-03","2026-10-30")
    rng=np.random.default_rng(17)
    prices=75*np.exp(np.cumsum(rng.normal(0,0.005,len(idx))))
    pd.DataFrame({"Date":idx,"brent":prices,"wti":prices-2.0,
                  "vix":20+rng.random(len(idx))}).to_csv(path,index=False)

def test_preprocessor_only_uses_mature_history(tmp_path):
    file=tmp_path/"prices.csv"
    dataset(file)
    frame=pd.read_csv(file,parse_dates=["Date"]).set_index("Date")
    ex=make_examples(make_features(frame),frame["brent"])
    plan=make_plan(ex,max_folds=1)
    fold=plan["folds"][0]
    prep=Preprocessor.fit(ex,fold["train"],fold["train_end"])
    assert "brent_ret_1" in prep.columns
    assert np.isfinite(prep.x(ex.features)).all()
    with np.testing.assert_raises(ValueError):
        Preprocessor.fit(ex,plan["test"],fold["train_end"])

def test_smoke_does_not_publish_active_artifacts(tmp_path):
    file=tmp_path/"prices.csv"
    dataset(file)
    root=tmp_path/"artifacts"
    cfg=ExperimentConfig(hidden_size=8,epochs=1,patience=1,batch_size=256)
    result=run_experiment(data_path=file,artifact_dir=root,cfg=cfg,smoke=True)
    assert result["selection"]["architecture"] in ("rnn","lstm","gru")
    assert not (root/"active.json").exists()
    assert len(result["models"])==3
    assert 0<=result["uncertainty"]["test_coverage"]["5"]<=1

def test_full_training_and_production_inference(tmp_path):
    file=tmp_path/"prices.csv"
    dataset(file)
    root=tmp_path/"artifacts"
    cfg=ExperimentConfig(hidden_size=8,epochs=1,patience=1,batch_size=256)
    result=run_experiment(data_path=file,artifact_dir=root,cfg=cfg)
    assert len(result["folds"])==5
    assert (root/"active.json").exists()
    pred=predict_latest(data_path=file,artifact_dir=root)
    assert len(pred["prices"])==3
    assert pred["last_price"]>0
    assert all(a<b<c for a,b,c in zip(pred["lower"],pred["prices"],pred["upper"]))
    assert not pred["data_changed_since_training"]
