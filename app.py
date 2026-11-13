"""Panel ejecutivo Streamlit de Brent Intelligence."""
from __future__ import annotations
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from brent_intelligence.config import DATA_PATH, ARTIFACTS_DIR
from brent_intelligence.experiment import predict_latest
from brent_intelligence.features import load_prices

st.set_page_config(page_title="Brent Intelligence",page_icon="🛢️",layout="wide")
st.title("Brent Intelligence")
st.caption("Pronostico experimental del Brent spot (USD/barril) · RNN / LSTM / GRU")
if not DATA_PATH.exists():
    st.warning("Descarga los datos y ejecuta el entrenamiento antes de abrir el panel.")
    st.code("python -m scripts.download_data\npython -m scripts.audit_data\npython -m scripts.train_models\nstreamlit run app.py",language="bash")
    st.stop()

@st.cache_data(show_spinner=False)
def read_prices(path: str, modified: float):
    return load_prices(path)

prices=read_prices(str(DATA_PATH),DATA_PATH.stat().st_mtime)
if not (ARTIFACTS_DIR/"active.json").exists():
    st.info("Datos disponibles. Falta ejecutar el experimento completo.")
    st.line_chart(prices["brent"].tail(800))
    st.code("python -m scripts.train_models",language="bash")
    st.stop()
try:
    forecast=predict_latest()
except (FileNotFoundError,RuntimeError,KeyError,ValueError) as exc:
    st.error(f"No se han podido cargar los modelos: {exc}")
    st.stop()
summary=forecast["summary"]
last_price=forecast["last_price"]
if forecast["stale_data"]:
    st.warning("El ultimo Brent tiene mas de siete dias. Verifica la descarga EIA/FRED.")
if forecast["data_changed_since_training"]:
    st.warning("El CSV ha cambiado desde el entrenamiento; debes reentrenar.")
st.info("Modelo experimental: las bandas son orientativas y los indicadores externos no cuentan con historicos point-in-time certificados.")
a,b,c=st.columns(3)
a.metric("Ultimo Brent observado",f"USD {last_price:,.2f}")
b.metric("Arquitectura seleccionada",forecast["model"].upper())
c.metric("Fecha ultima observacion",forecast["last_date"])
st.subheader("Predicciones")
names={5:"Corto: 5 sesiones",63:"Medio: 63 sesiones",252:"Largo: 252 sesiones"}
selected=st.radio("Horizonte",list(names),format_func=lambda h:names[h],horizontal=True)
j=forecast["horizons"].index(selected)
point,lo,hi=(forecast[k][j] for k in ("prices","lower","upper"))
ca,cb,cc=st.columns(3)
ca.metric("Precio central",f"USD {point:,.2f}",f"{100*(point/last_price-1):+.2f}%")
cb.metric("Limite inferior (80% nominal)",f"USD {lo:,.2f}")
cc.metric("Limite superior (80% nominal)",f"USD {hi:,.2f}")
fig=go.Figure()
fig.add_trace(go.Scatter(x=prices.index,y=prices["brent"],mode="lines",name="Brent observado",line=dict(width=1.7)))
fig.update_layout(height=350,xaxis_title="Fecha",yaxis_title="USD/barril",margin=dict(t=20,b=20,l=20,r=20))
st.plotly_chart(fig,use_container_width=True)
st.caption("Historico observado, no trayectoria diaria pronosticada.")
bar=go.Figure()
bar.add_trace(go.Bar(x=["5 sesiones","63 sesiones","252 sesiones"],
                     y=forecast["prices"],name="Precio previsto",
                     error_y=dict(type="data",symmetric=False,
                         array=[h-p for p,h in zip(forecast["prices"],forecast["upper"])],
                         arrayminus=[p-l for p,l in zip(forecast["prices"],forecast["lower"])])))
bar.add_hline(y=last_price,line_dash="dash",annotation_text="Ultimo Brent")
bar.update_layout(height=320,yaxis_title="USD/barril",margin=dict(t=15,b=25,l=20,r=20))
st.plotly_chart(bar,use_container_width=True)
st.subheader("Comparacion fuera de muestra")
rows=[]
for kind,info in summary["models"].items():
    for h in forecast["horizons"]:
        m=info["test"][str(h)]
        rows.append({"Red":kind.upper(),"Horizonte":h,
                     "MAE ($)":round(m["mae"],2),"RMSE ($)":round(m["rmse"],2),
                     "RMSE ingenuo ($)":round(m["naive_rmse"],2),
                     "Mejora RMSE vs ingenuo (%)":round(100*(1-m["relative_rmse"]),2) if m["relative_rmse"] is not None else None,
                     "Parametros":info["parameter_count"],"Score validacion":round(info["validation_score"],4)})
st.dataframe(pd.DataFrame(rows),hide_index=True,use_container_width=True)
st.caption("La arquitectura se elige ANTES del test. Mejoras negativas implican que el predictor ingenuo es superior.")
with st.expander("Validacion temporal e incertidumbre"):
    st.dataframe(pd.DataFrame(summary["folds"]),hide_index=True,use_container_width=True)
    st.write("Cobertura empirica en test de los intervalos nominales del 80%:")
    st.json(summary["uncertainty"]["test_coverage"])
    st.caption("Dependencia temporal y cambios de regimen: no existen garantias universales de cobertura.")
with st.expander("Contexto y limitaciones economicas"):
    st.markdown("""- **2008:** subida y posterior desplome con la crisis financiera.
- **2014-2016:** caida en un contexto de exceso de oferta.
- **2020:** desplome durante la pandemia; el WTI llego a valores negativos.
- **2022:** aumento tras la invasion de Ucrania.
Los episodios son contexto descriptivo, no una explicacion causal del modelo.""")
with st.expander("Fuente, versiones y actualizacion"):
    st.write("Brent spot: EIA/FRED; indicadores externos cuando estan disponibles.")
    st.write("Dataset de entrenamiento:",summary["last_data_date"])
    st.write("Ultimo origen con etiquetas maduras:",summary["production"]["last_matured_training_origin"])
    st.write("Huella SHA256 del dataset:",summary["data_sha256"])
    st.code("python -m scripts.download_data\npython -m scripts.audit_data\npython -m scripts.train_models\nstreamlit run app.py",language="bash")
st.divider()
st.caption("Trabajo academico. Sin recomendacion de inversion ni capacidad de ejecucion.")
