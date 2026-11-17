"""Academic Brent app backed by the evaluated v4 full-data candidate.

python -m streamlit run app_delivery.py
Uses saved RNN/LSTM/GRU weights, never calls training.
If 2024+ historical predictions are present, descriptive (not guaranteed)
80% log-error bands are shown with explicit caveats.
"""
from __future__ import annotations

from io import BytesIO
import json
import os
from pathlib import Path
import tempfile

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from brent_intelligence.config import ARTIFACTS_DIR, DATA_PATH, REPORTS_DIR
from brent_intelligence.delivery_inference import forecast_candidate
from brent_intelligence.features import load_prices

EVENTS = (
    ("Crisis financiera 2008", "2008-01-01", "2009-03-31",
     "Pico y desplome del crudo durante la crisis financiera"),
    ("Exceso de oferta", "2014-06-01", "2016-12-31",
     "Caída de 2014–2016 en un contexto de exceso de oferta"),
    ("Pandemia", "2020-02-01", "2020-12-31",
     "Desplome mundial por la pandemia; el WTI llegó a negativo"),
    ("Invasión de Ucrania", "2022-02-24", "2022-12-31",
     "Tensión energética y subida del petróleo tras la invasión"),
    ("Movimientos recientes", "2026-01-01", "2026-12-31",
     "Volatilidad de 2026; no se atribuye causa sin evidencia"),
)
HORIZONS = (5, 63, 252)


def last_run():
    runs = sorted((REPORTS_DIR/"delivery_training").glob("*/summary.json"),
                  reverse=True)
    return runs[0] if runs else None


def intervals_from_evaluation(csv_path, kind, h, point):
    """Historical 80th percentile of abs log-errors; NOT validated coverage."""
    if not csv_path.is_file():
        return None
    df = pd.read_csv(csv_path)
    needed = {"architecture", "horizon", "actual", "predicted"}
    if not needed.issubset(df.columns):
        return None
    sub = df.loc[
        (df["architecture"].str.lower() == kind) &
        (pd.to_numeric(df["horizon"], errors="coerce") == h)
    ]
    actual = pd.to_numeric(sub["actual"], errors="coerce").to_numpy(dtype=float)
    pred = pd.to_numeric(sub["predicted"], errors="coerce").to_numpy(dtype=float)
    valid = np.isfinite(actual) & np.isfinite(pred) & (actual>0) & (pred>0)
    log_errors = np.abs(np.log(actual[valid]/pred[valid]))
    if len(log_errors) < 30:
        return None
    q = float(np.quantile(log_errors, .8))
    return (float(point*np.exp(-q)),float(point*np.exp(q)),len(log_errors))


st.set_page_config(page_title="Brent Intelligence · Entrega", page_icon="🛢️",
                   layout="wide")
st.title("Brent Intelligence | Caso de uso III")
st.caption("Predicción de Brent con RNN/LSTM/GRU, 5/63/252 sesiones y validación temporal.")
st.info("Los pesos actuales se reajustaron con datos maduros hasta noviembre de 2026. "
        "El RMSE histórico mostrado pertenece a modelos entrenados hasta 2023, "
        "NO al reajuste final ni a un test independiente de este.")

candidate = last_run()
summary_text = st.sidebar.text_input(
    "summary.json de la entrega v4", value=str(candidate) if candidate else "")
if not summary_text:
    st.error("No hay ejecución v4 local. Ejecuta primero python -m scripts.train_delivery.")
    st.stop()
summary_path = Path(summary_text).expanduser()
try:
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if summary["mode"] != "complete-delivery-candidates":
        raise ValueError("Este JSON no corresponde al entrenamiento v4 completo.")
    selected = summary["selected_architecture"]
    available = summary["full_data_models"]
except (OSError, ValueError, KeyError) as err:
    st.error(f"Resumen no disponible: {err}")
    st.stop()
architecture = st.sidebar.selectbox(
    "Arquitectura para pronosticar", list(available),
    index=list(available).index(selected),
    format_func=lambda kind: kind.upper() +
    (" (seleccionada en validación)" if kind==selected else "")
)
if architecture != selected:
    st.sidebar.warning("Esta arquitectura no fue la elegida por validación temporal.")

st.sidebar.caption("CSV opcional: histórico completo o nuevas filas Date,brent.")
file = st.sidebar.file_uploader("Actualizar mediante CSV (sin reentrenar)",type="csv")
temporary = None
try:
    if not DATA_PATH.is_file():
        raise FileNotFoundError(f"Falta {DATA_PATH}")
    prices = load_prices(DATA_PATH)
    data_path = DATA_PATH
    if file is not None:
        extra = pd.read_csv(BytesIO(file.getvalue()))
        if "Date" not in extra or "brent" not in extra:
            raise ValueError("El CSV cargado necesita Date y brent.")
        extra["Date"]=pd.to_datetime(extra["Date"],errors="raise")
        extra=extra.set_index("Date")
        if extra.index.has_duplicates:
            raise ValueError("El fichero tiene fechas repetidas.")
        extra=extra.apply(pd.to_numeric,errors="coerce")
        is_full=(len(extra) > len(prices)//2 and
                 extra.index.min() <= prices.index[0])
        if is_full:
            merged=extra.sort_index()
        else:
            if (extra.index <= prices.index[-1]).any():
                raise ValueError("Las filas incrementales deben ser posteriores "
                                 "a la última fecha. Para corregir el pasado "
                                 "carga el histórico completo.")
            merged=pd.concat([prices,extra]).sort_index()
        if merged["brent"].isna().any() or (merged["brent"]<=0).any():
            raise ValueError("Precios Brent inválidos en el CSV.")
        merged.index.name="Date"
        with tempfile.NamedTemporaryFile(
            suffix=".csv",prefix="brent_delivery_",delete=False
        ) as fh:
            temporary=Path(fh.name)
        merged.to_csv(temporary)
        data_path=temporary
        prices=load_prices(data_path)
        st.sidebar.success("Observaciones incorporadas temporalmente, pesos sin cambios.")
    forecast=forecast_candidate(
        summary_path=summary_path,data_path=data_path,
        weights_dir=ARTIFACTS_DIR/"delivery_candidates"/summary_path.parent.name,
        architecture=architecture
    )
except (ValueError, OSError, KeyError, RuntimeError,
        pd.errors.ParserError) as err:
    st.error(f"No se ha podido realizar la inferencia: {err}")
    st.stop()
finally:
    if temporary:
        try:
            os.unlink(temporary)
        except OSError:
            pass

st.sidebar.caption("Descarga API FRED/EIA: python -m scripts.download_data. "
                   "Recarga después esta aplicación; descargar no reentrena.")
last_price=float(forecast["last_observed_price"])
last_date=pd.Timestamp(forecast["latest_observed_date"])
mature_date=pd.Timestamp(forecast["model_trained_through"])
if (pd.Timestamp.now().normalize()-last_date).days>7:
    st.warning(f"Último dato {last_date.date()}: comprueba posibles retrasos EIA/FRED.")
m1,m2,m3=st.columns(3)
m1.metric("Último Brent observado",f"{last_price:.2f} USD")
m2.metric("Modelo que realmente pronostica",architecture.upper())
m3.metric("Último dato",str(last_date.date()))
st.caption(f"Modelo ajustado hasta {mature_date.date()}. "
           "Una observación posterior solo pasa por los pesos ya entrenados.")

h=st.radio("Horizonte",HORIZONS,horizontal=True,
           format_func=lambda x:{5:"5 sesiones · una semana",
                                 63:"63 sesiones · tres meses",
                                 252:"252 sesiones · doce meses"}[x])
point=float(forecast["predictions"][str(h)])
st.metric("Precio previsto",f"{point:.2f} USD/barril",
          f"{100*(point/last_price-1):+.2f}% frente al último Brent")
try:
    bounds=intervals_from_evaluation(
        summary_path.parent/"holdout_predictions.csv",architecture,h,point)
except (ValueError, pd.errors.ParserError, OSError):
    bounds=None
if bounds:
    lo,hi,n=bounds
    st.write(f"**Banda orientativa (80 % de errores absolutos históricos):** "
             f"{lo:.2f}–{hi:.2f} USD/barril · {n} observaciones.")
    st.warning("Banda derivada de residuos de modelos históricos distintos "
               "del reajuste final. NO equivale a intervalo validado con "
               "cobertura futura del 80 %. Los años 2024+ ya se examinaron.")
else:
    st.warning("No hay residuos históricos suficientes para generar "
               "un rango justificable; no se muestra incertidumbre inventada.")

st.subheader("Precio histórico y ciclo económico")
fig=go.Figure()
fig.add_trace(go.Scatter(x=prices.index,y=prices["brent"],
                         mode="lines",name="Brent EIA/FRED",
                         line={"width":1.5,"color":"#298EC2"}))
if st.checkbox("Marcar picos, caídas y episodios internacionales",value=True):
    for name,start,end,description in EVENTS:
        if pd.Timestamp(end)<prices.index[0] or pd.Timestamp(start)>prices.index[-1]:
            continue
        fig.add_vrect(x0=max(pd.Timestamp(start),prices.index[0]),
                      x1=min(pd.Timestamp(end),prices.index[-1]),
                      fillcolor="#D8A13E",opacity=.12,line_width=0,
                      annotation_text=name,
                      annotation_position="top left",
                      annotation_font_size=10)
    with st.expander("Relación histórica con los episodios"):
        for name,start,end,description in EVENTS:
            st.write(f"**{name} ({start[:4]}):** {description}.")
        st.caption("Los periodos son contexto explicativo aproximado. "
                   "La red no infiere causalidad geopolítica.")
fig.update_layout(height=480,xaxis_title="Fecha observada",
                  yaxis_title="USD/barril",hovermode="x unified")
st.plotly_chart(fig,use_container_width=True)

st.subheader("Tres predicciones")
points=[float(forecast["predictions"][str(x)]) for x in HORIZONS]
overview=go.Figure(go.Bar(
    x=[f"{x} sesiones" for x in HORIZONS],y=points,
    marker={"color":"#298EC2"},name="Precio previsto"))
overview.add_hline(y=last_price,line_dash="dash",annotation_text="Último precio")
overview.update_layout(height=300,yaxis_title="USD/barril")
st.plotly_chart(overview,use_container_width=True)
st.caption("No se representa una trayectoria diaria futura: cada valor "
           "es una predicción directa de un horizonte de sesiones observado.")

st.subheader("Evaluación temporal 2024+ · no es test del reajuste actual")
rows=[]
tested=summary.get("historical_test_diagnostic",{}).get("models",{})
for kind,info in tested.items():
    for period in HORIZONS:
        m=info["metrics"][str(period)]
        rows.append({
            "Red":kind.upper(),"Sesiones":period,"N":m["n"],
            "R²":round(m["r2"],3) if m["r2"] is not None else None,
            "MAE USD":round(m["mae"],3),"MSE USD²":round(m["mse"],3),
            "RMSE USD":round(m["rmse"],3),
            "RMSE ingenuo":round(m["naive_rmse"],3),
            "Mejora %":round(m["improvement_pct"],2),
            "Parámetros":available[kind]["n_parameters"],
        })
if rows:
    st.dataframe(pd.DataFrame(rows),hide_index=True,use_container_width=True)
    actual_metric=tested[architecture]["metrics"][str(h)]
    improvement=actual_metric["improvement_pct"]
    desc=(f"Mejora histórica en RMSE de {improvement:+.2f}% respecto al ingenuo. "
          "No implica superioridad futura.")
    st.info(f"**Fiabilidad del horizonte {h}:** {desc}")
else:
    st.warning("El resumen no incluye resultados históricos comparables.")

with st.expander("Limitaciones y datos complementarios"):
    st.write("Los indicadores WTI, gas, tipos, divisas, metales y VIX "
             "son señales auxiliares. Su disponibilidad histórica exacta "
             "y revisiones no están certificadas punto por punto.")
    st.write("El CSV OPEP (Date,orb) solo se integra si se ha "
             "suministrado expresamente; no se atribuye al modelo si falta.")
    st.write("Las gráficas del precio Brent contienen cotizaciones spot; "
             "los OHLCV de futuros son complementarios y no se consumen "
             "actualmente como entradas recurrentes.")
    st.write("El entrenamiento no registró tiempos de CPU comparables "
             "para el experimento v4: no se inventan esos segundos.")
    st.write("El último modelo reajustado no cuenta con un test independiente.")
st.caption("Demostración académica para informar decisiones. "
           "No supone una recomendación de inversión.")
