"""Curvas train/validation, reales-predichos y residuos sin reentrenamiento.

Reconstruye la evaluacion SOLO si el CSV conserva el mismo SHA256 del
entrenamiento; impide analizar pesos antiguos contra datos revisados.
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import torch

from brent_intelligence.config import ARTIFACTS_DIR,DATA_PATH,REPORTS_DIR
from brent_intelligence.evaluation import prices_from_returns,score
from brent_intelligence.experiment import Preprocessor,_predict
from brent_intelligence.features import load_prices,make_features
from brent_intelligence.frozen import _active_run,file_hash,read_json
from brent_intelligence.modeling import RecurrentForecaster
from brent_intelligence.temporal import make_examples,make_plan


def build_loss_figure(kind: str, folds: list[dict]):
    fig = make_subplots(rows=3,cols=2,subplot_titles=[
        f"Fold {i+1}: SmoothL1 (retornos escalados)" for i in range(5)])
    for i, fold in enumerate(folds):
        values=fold["history"]
        x=[row["epoch"] for row in values]
        for col,name in (("#1D7F6E","train_loss"),("#DC8D35","val_loss")):
            if not all(name in row for row in values):
                continue
            fig.add_trace(go.Scatter(x=x,y=[row[name] for row in values],mode="lines+markers",
                                     name="Train" if name=="train_loss" else "Validacion",
                                     line=dict(color=col),legendgroup=name,showlegend=i==0),
                          row=(i//2)+1,col=(i%2)+1)
    fig.update_layout(title=f"{kind.upper()} · perdida por epoca y fold",
                      height=900,template="plotly_white")
    fig.update_xaxes(title_text="Epoca")
    fig.update_yaxes(title_text="SmoothL1")
    return fig


def generate(*, data: Path = DATA_PATH, artifacts: Path = ARTIFACTS_DIR,
             reports: Path = REPORTS_DIR) -> dict:
    run=_active_run(artifacts)
    summary=read_json(run/"summary.json")
    if summary.get("mode")!="historical-evaluation":
        raise ValueError("No se publican diagnosticos del modo smoke")
    if file_hash(data)!=summary["data_sha256"]:
        raise ValueError("Dataset distinto del entrenamiento. Conserva el CSV original o ejecuta antes de actualizar.")
    out=Path(reports)/"diagnostics"
    out.mkdir(parents=True,exist_ok=True)
    figures=[]
    for kind,values in summary["models"].items():
        fig=build_loss_figure(kind,values["folds"])
        path=out/f"loss_{kind}.html"
        fig.write_html(str(path),include_plotlyjs=True)
        figures.append(str(path))
    winner=summary["selection"]["architecture"]
    raw=load_prices(data)
    cfg=summary["config"]
    feat=make_features(raw)
    examples=make_examples(feat,raw["brent"],window=int(cfg["window"]),
                           horizons=tuple(cfg["horizons"]))
    ids=make_plan(examples)["test"]
    prep=Preprocessor(**read_json(run/"evaluation_preprocessing.json"))
    model=RecurrentForecaster(winner,len(prep.columns),int(cfg["hidden_size"]),
                             int(cfg["layers"]),len(cfg["horizons"]))
    model.load_state_dict(torch.load(run/f"{winner}_evaluation.pt",
                                     map_location="cpu",weights_only=True))
    x=examples.x(prep.x(examples.features),ids)
    pred=prices_from_returns(examples.anchors_price(ids),
                             prep.inverse_y(_predict(model,x)))
    actual=examples.actual(ids)
    anchor_price=examples.anchors_price(ids)
    observed=score(actual,pred,anchor_price,tuple(cfg["horizons"]))
    for h in cfg["horizons"]:
        measured=observed[str(h)]["rmse"]
        recorded=summary["models"][winner]["test"][str(h)]["rmse"]
        if not np.isclose(measured,recorded,rtol=5e-4,atol=0.01):
            raise ValueError(f"Reconstruccion RMSE incompatible para {h} sesiones")
    dates=examples.dates.take(examples.anchors[ids])
    rows=[]
    for j,h in enumerate(cfg["horizons"]):
        for i,date in enumerate(dates):
            rows.append({"anchor_date":date.date().isoformat(),
                         "horizon":h,"actual":float(actual[i,j]),
                         "predicted":float(pred[i,j]),
                         "naive":float(anchor_price[i]),
                         "residual":float(pred[i,j]-actual[i,j])})
        fig=go.Figure()
        fig.add_trace(go.Scatter(x=dates,y=actual[:,j],name="Real",mode="lines"))
        fig.add_trace(go.Scatter(x=dates,y=pred[:,j],name=winner.upper(),mode="lines"))
        fig.add_trace(go.Scatter(x=dates,y=anchor_price,name="Ingenuo",mode="lines",
                                 line=dict(dash="dot")))
        fig.update_layout(title=f"Precio real y predicho · {h} sesiones",
                          xaxis_title="Fecha de origen",yaxis_title="USD/barril",
                          template="plotly_white")
        path=out/f"real_vs_predicho_{h}.html"
        fig.write_html(str(path),include_plotlyjs=True)
        figures.append(str(path))
        fig=go.Figure()
        fig.add_trace(go.Scatter(x=dates,y=pred[:,j]-actual[:,j],
                                 name="Error predicho - real",mode="lines"))
        fig.add_hline(y=0,line_dash="dash")
        fig.update_layout(title=f"Residuos test · {h} sesiones",
                          xaxis_title="Fecha de origen",yaxis_title="USD/barril",
                          template="plotly_white")
        path=out/f"residuos_{h}.html"
        fig.write_html(str(path),include_plotlyjs=True)
        figures.append(str(path))
    csv_path=out/"predicciones_test.csv"
    with csv_path.open("w",newline="",encoding="utf-8") as dest:
        writer=csv.DictWriter(dest,fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return {"winner":winner,"figures":figures,"test_prediction_csv":str(csv_path),
            "n_test_anchors":len(ids),"training_rerun":False}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data",type=Path,default=DATA_PATH)
    p.add_argument("--artifacts",type=Path,default=ARTIFACTS_DIR)
    p.add_argument("--reports",type=Path,default=REPORTS_DIR)
    args=p.parse_args()
    info=generate(data=args.data,artifacts=args.artifacts,reports=args.reports)
    print(f"Sin reentrenar: {len(info['figures'])} graficos; ganadora {info['winner']}")
    print("Archivos en:",args.reports/"diagnostics")

if __name__=="__main__":
    main()
