"""Investigacion v2: modelos separados por horizonte, sin tocar el test.

Evalua exclusivamente los cinco folds historicos de VALIDACION (2012-2021).
Pone a prueba la hipotesis de que un modelo multisalida perjudica los horizontes
cortos. Incluye Ridge regularizado y LSTM monohorizonte con validacion interna
para seleccionar hiperparametros / numero de epocas.

No escribe en artifacts/active.json, deployment/, public/ ni la app.
"""
from __future__ import annotations

import argparse
from dataclasses import replace
from datetime import datetime,timezone
from hashlib import sha256
import json
from pathlib import Path
import statistics

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

from brent_intelligence.config import DATA_PATH,REPORTS_DIR,ExperimentConfig
from brent_intelligence.evaluation import prices_from_returns,score
from brent_intelligence.experiment import Preprocessor,_predict,_train
from brent_intelligence.features import load_prices,make_features
from brent_intelligence.temporal import FOLD_BOUNDARIES,make_examples

KINDS=("ridge","lstm")
ALPHAS=(0.1,1.0,10.0,100.0)


def _folds_for_horizon(ex, *, min_valid=20):
    """Same outer forecast origins as the original multi-output evaluation.

    A single-target learner may train on more mature examples, but both
    experiments are scored against EXACTLY the same validation origins.
    """
    origins=ex.dates.take(ex.anchors)
    matured=ex.dates.take(ex.targets[:,0])
    # The historical shared-label baseline can only use origins with its
    # longest (252-session) target matured by validation cutoff.
    max_horizon=252
    long_end=ex.anchors + max_horizon
    if long_end.max() >= len(ex.dates):
        # Trim potential short-horizon observations near the global data end.
        valid_global=long_end < len(ex.dates)
    else:
        valid_global=np.ones(len(ex.anchors),dtype=bool)
    for index,(cutoff,val_end) in enumerate(FOLD_BOUNDARIES,1):
        cut=pd.Timestamp(cutoff)
        val=pd.Timestamp(val_end)
        train=np.flatnonzero(matured <= cut)
        compare_mature=np.full(len(ex.anchors),False,dtype=bool)
        compare_mature[valid_global] = (
            ex.dates.take(long_end[valid_global]) <= val
        )
        validation=np.flatnonzero((origins > cut) & (matured <= val) & compare_mature)
        if len(train)<100 or len(validation)<min_valid:
            raise ValueError(f"Fold {index} sin suficientes datos")
        yield {"name":f"fold_{index:02d}","cutoff":cutoff,
               "validation_end":val_end,"train":train,"validation":validation}


def _inner_split(ex,outer_train,cutoff,ratio=0.25):
    """Chronological inner split; all inner labels must have matured."""
    ids=np.asarray(outer_train,dtype=np.int64)
    if len(ids)<120:
        raise ValueError("Pocos ejemplos para validacion interna")
    origins=ex.dates.take(ex.anchors[ids])
    q=max(30,min(len(ids)-30,int(len(ids)*(1-ratio))))
    inner_cutoff=origins[q-1]
    maturity=ex.dates.take(ex.targets[ids,0])
    train=ids[np.flatnonzero(maturity <= inner_cutoff)]
    val=ids[np.flatnonzero((origins>inner_cutoff)&
                           (maturity<=pd.Timestamp(cutoff)))]
    if len(train)<30 or len(val)<20:
        raise ValueError("La validacion interna no tiene etiquetas maduras suficientes")
    return str(inner_cutoff.date()),train,val


def _ridge_features(prep,ex,ids):
    sequence=ex.x(prep.x(ex.features),ids).astype(np.float64)
    return np.concatenate((sequence[:,-1,:],sequence.mean(axis=1),
                           sequence.std(axis=1)),axis=1)


def _outer_metrics(ex,ids,expected_log_returns):
    pred=prices_from_returns(ex.anchors_price(ids),expected_log_returns)
    return score(ex.actual(ids),pred,ex.anchors_price(ids),ex.horizons)[str(ex.horizons[0])]


def evaluate_horizon(ex,fold,*,kind,cfg,feature_set,epochs=None):
    """One outer validation fold; never read test or calibration labels."""
    train,val=fold["train"],fold["validation"]
    inner_date,inner_train,inner_val=_inner_split(ex,train,fold["cutoff"])
    prep_inner=Preprocessor.fit(ex,inner_train,inner_date,cfg.min_feature_coverage)

    if kind=="ridge":
        xtrain=_ridge_features(prep_inner,ex,inner_train)
        xv=_ridge_features(prep_inner,ex,inner_val)
        ytrain=prep_inner.y(ex.y(inner_train)).ravel()
        selected_alpha=None
        best_inner=float("inf")
        for alpha in ALPHAS:
            m=Ridge(alpha=alpha).fit(xtrain,ytrain)
            log_pred=prep_inner.inverse_y(m.predict(xv).reshape(-1,1))
            metric=_outer_metrics(ex,inner_val,log_pred)
            if metric["rmse"]<best_inner:
                best_inner=metric["rmse"]
                selected_alpha=alpha
        prep=Preprocessor.fit(ex,train,fold["cutoff"],cfg.min_feature_coverage)
        model=Ridge(alpha=selected_alpha).fit(
            _ridge_features(prep,ex,train),
            prep.y(ex.y(train)).ravel())
        predicted=prep.inverse_y(model.predict(
            _ridge_features(prep,ex,val)).reshape(-1,1))
        settings={"alpha":selected_alpha,"inner_val_rmse":best_inner}
    elif kind=="lstm":
        inner_x=ex.x(prep_inner.x(ex.features),inner_train)
        inner_y=prep_inner.y(ex.y(inner_train))
        val_x=ex.x(prep_inner.x(ex.features),inner_val)
        val_y=prep_inner.y(ex.y(inner_val))
        _, inner_history=_train("lstm",inner_x,inner_y,cfg,
                                len(prep_inner.columns),xv=val_x,yv=val_y,
                                device="cpu")
        selected_epochs=int(inner_history["best_epoch"])
        prep=Preprocessor.fit(ex,train,fold["cutoff"],cfg.min_feature_coverage)
        full_x=ex.x(prep.x(ex.features),train)
        full_y=prep.y(ex.y(train))
        outer_model,train_history=_train(
            "lstm",full_x,full_y,cfg,len(prep.columns),
            epochs=selected_epochs,device="cpu")
        out_x=ex.x(prep.x(ex.features),val)
        predicted=prep.inverse_y(_predict(outer_model,out_x))
        settings={"selected_epochs":selected_epochs,
                  "inner_early_stop_epochs":inner_history["epochs"],
                  "refit_seconds":train_history["seconds"]}
    else:
        raise ValueError("Solo ridge o lstm")
    metric=_outer_metrics(ex,val,predicted)
    return {
        "fold":fold["name"],"cutoff":fold["cutoff"],
        "validation_end":fold["validation_end"],"kind":kind,
        "horizon":int(ex.horizons[0]),"feature_set":feature_set,
        "n_train":len(train),"n_inner_train":len(inner_train),
        "n_inner_validation":len(inner_val),"n_outer_validation":len(val),
        "metrics":metric,"settings":settings,
    }


def run(*,data:Path=DATA_PATH,output:Path=REPORTS_DIR/"research_v2",
        mode:str="ridge",feature_sets:tuple[str,...]=("brent_only","all"),
        horizons:tuple[int,...]=(5,63,252),folds:int=5,
        epochs:int=12,hidden_size:int=16,window:int=60,
        dry_run:bool=False)->dict:
    if mode not in ("ridge","lstm","both"):
        raise ValueError("Modelo incorrecto")
    if not horizons or any(h not in (5,63,252) for h in horizons):
        raise ValueError("Horizonte invalido")
    if not feature_sets or any(f not in ("brent_only","all") for f in feature_sets):
        raise ValueError("Configuracion de variables invalida")
    if not 1<=folds<=5:
        raise ValueError("Se admiten de uno a cinco folds")
    raw_source=load_prices(data)
    # Exclusión física de 2022-2026: ni etiquetas ni features del test y
    # calibración se construyen en este laboratorio de investigación.
    raw=raw_source.loc[: "2021-12-31"].copy()
    if len(raw)<window+252:
        raise ValueError("Historial insuficiente antes del bloqueo 2022")
    all_features=make_features(raw)
    records=[]
    run_key=sha256(Path(data).read_bytes()).hexdigest()
    classes=("ridge","lstm") if mode=="both" else (mode,)
    for feature_set in feature_sets:
        features=(all_features.filter(regex=r"^brent_")
                  if feature_set=="brent_only" else all_features)
        for horizon in horizons:
            cfg=replace(ExperimentConfig(),window=window,horizons=(horizon,),
                        epochs=epochs,hidden_size=hidden_size)
            ex=make_examples(features,raw["brent"],window=window,horizons=(horizon,))
            plans=list(_folds_for_horizon(ex))[:folds]
            for kind in classes:
                for fold in plans:
                    if dry_run:
                        records.append({"model":kind,"features":feature_set,
                                        "horizon":horizon,"fold":fold["name"],
                                        "n_train":len(fold["train"]),
                                        "n_validation":len(fold["validation"])})
                    else:
                        result=evaluate_horizon(ex,fold,kind=kind,cfg=cfg,
                                                feature_set=feature_set)
                        records.append(result)
    summary={"status":"research-only","created_utc":datetime.now(timezone.utc).isoformat(),
             "data_sha256":run_key,"research_data_end":str(raw.index[-1].date()),
             "source_data_end":str(raw_source.index[-1].date()),
             "used_test":False,"used_calibration":False,
             "dry_run":dry_run,
             "mode":mode,"feature_sets":feature_sets,"horizons":horizons,
             "n_folds":folds,"epochs_limit":epochs,"results":records,
             "warning":"No es un test nuevo. El test original ya fue consultado; este estudio usa solo los folds de validacion historica. El modelo desplegado permanece intacto."}
    if not dry_run:
        grouped={}
        for row in records:
            key=(row["kind"],row["feature_set"],row["horizon"])
            grouped.setdefault(key,[]).append(row["metrics"]["relative_rmse"])
        summary["validation_ranking"]=[
            {"model":k[0],"feature_set":k[1],"horizon":k[2],
             "mean_rmse_over_naive":float(statistics.mean(v)),
             "n_folds":len(v)} for k,v in grouped.items()
        ]
        summary["validation_ranking"].sort(
            key=lambda x:(x["horizon"],x["mean_rmse_over_naive"]))
        out=Path(output)
        out.mkdir(parents=True,exist_ok=True)
        path=out/("validation_"+datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
                  +"_"+run_key[:8]+".json")
        path.write_text(json.dumps(summary,indent=2,ensure_ascii=False,allow_nan=False),
                        encoding="utf-8")
        summary["saved_to"]=str(path)
    return summary


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data",type=Path,default=DATA_PATH)
    parser.add_argument("--output",type=Path,default=REPORTS_DIR/"research_v2")
    parser.add_argument("--model",choices=("ridge","lstm","both"),default="ridge")
    parser.add_argument("--features",choices=("brent_only","all"),
                        nargs="+",default=("brent_only","all"))
    parser.add_argument("--horizons",type=int,nargs="+",default=(5,63,252))
    parser.add_argument("--folds",type=int,default=5)
    parser.add_argument("--epochs",type=int,default=12)
    parser.add_argument("--hidden-size",type=int,default=16)
    parser.add_argument("--window",type=int,default=60)
    parser.add_argument("--dry-run",action="store_true")
    args=parser.parse_args()
    summary=run(data=args.data,output=args.output,mode=args.model,
                feature_sets=tuple(args.features),horizons=tuple(args.horizons),
                folds=args.folds,epochs=args.epochs,hidden_size=args.hidden_size,
                window=args.window,dry_run=args.dry_run)
    if args.dry_run:
        print("Plan de validacion - no se entreno:",len(summary["results"]),"filas")
    else:
        for row in summary["validation_ranking"]:
            print(f"{row['horizon']:3d} {row['model']:5s} {row['feature_set']:10s} "
                  f"RMSE/naive={row['mean_rmse_over_naive']:.4f} "
                  f"folds={row['n_folds']}")
        print("Informe:",summary["saved_to"])
    print("Sin cambio del modelo de produccion. Test reservado NO utilizado.")


if __name__=="__main__":
    main()
