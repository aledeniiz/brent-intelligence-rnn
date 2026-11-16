"""Public forecast dashboard. It never trains or downloads data."""
from __future__ import annotations
from datetime import datetime, timezone
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from brent_intelligence.frozen import PUBLISHED_FORECAST, read_json

st.set_page_config(page_title="Brent Intelligence · Live",page_icon="🛢️",layout="wide")
st.title("Brent Intelligence")
st.caption("Actualizacion automatica de datos · Modelo RNN congelado")
if not PUBLISHED_FORECAST.is_file():
    st.warning("Aun no se ha publicado el primer pronostico congelado.")
    st.info("Exporta el modelo existente y publica deployment/frozen/ para activar la actualizacion.")
    st.stop()

@st.cache_data(show_spinner=False)
def read_public(path, mtime):
    return read_json(PUBLISHED_FORECAST)

record=read_public(str(PUBLISHED_FORECAST),PUBLISHED_FORECAST.stat().st_mtime)
last_date=pd.Timestamp(record["last_date"]).date()
age=(datetime.now(timezone.utc).date()-last_date).days
if age>7:
    st.error(f"Cotizacion desactualizada: ultima observacion hace {age} dias.")
else:
    st.success(f"Ultima observacion Brent: {last_date}")
st.info("La LSTM de produccion conserva sus pesos. Las metricas historicas corresponden a los modelos de evaluacion, no al actual.")
a,b,c=st.columns(3)
a.metric("Brent observado",f"USD {record['last_price']:.2f}")
b.metric("Arquitectura",record["model"]["architecture"].upper())
c.metric("Modelo congelado desde",record["model"]["trained_data_end"])
hs=record["horizons"]
selected=st.radio("Horizonte",hs,format_func=lambda h:f"{h} sesiones",horizontal=True)
i=hs.index(selected)
p=record["prices"][i]
a,b,c=st.columns(3)
a.metric("Prediccion central",f"USD {p:.2f}",f"{100*(p/record['last_price']-1):+.2f}%")
b.metric("Limite inferior 80% nominal",f"USD {record['lower'][i]:.2f}")
c.metric("Limite superior 80% nominal",f"USD {record['upper'][i]:.2f}")
history=pd.DataFrame(record["history"])
history["date"]=pd.to_datetime(history["date"])
fig=go.Figure()
fig.add_trace(go.Scatter(x=history["date"],y=history["brent"],mode="lines",name="Brent observado"))
fig.update_layout(title="Historico (ultimas 450 sesiones)",height=370,yaxis_title="USD/barril",template="plotly_white")
st.plotly_chart(fig,use_container_width=True)
fig=go.Figure()
fig.add_trace(go.Bar(x=[f"{h} sesiones" for h in hs],y=record["prices"],
                     error_y=dict(type="data",symmetric=False,
                     array=[u-p for p,u in zip(record["prices"],record["upper"])],
                     arrayminus=[p-l for p,l in zip(record["prices"],record["lower"])])))
fig.add_hline(y=record["last_price"],line_dash="dot",annotation_text="Ultimo Brent")
fig.update_layout(title="Tres horizontes e intervalos",height=330,yaxis_title="USD/barril")
st.plotly_chart(fig,use_container_width=True)
st.subheader("Resultados historicos (NO del modelo de produccion)")
rows=[]
for kind,scores in record["historical_test_metrics"].items():
    for h in hs:
        m=scores[str(h)]
        ratio=m["relative_rmse"]
        rows.append({"Modelo":kind.upper(),"Sesiones":h,
                     "MAE USD":round(m["mae"],3),"RMSE USD":round(m["rmse"],3),
                     "RMSE ingenuo":round(m["naive_rmse"],3),
                     "Mejora RMSE %":round(100*(1-ratio),2) if ratio is not None else None})
st.dataframe(pd.DataFrame(rows),hide_index=True,use_container_width=True)
st.caption("Una mejora negativa significa que el ingenuo gana; no hay significacion estadistica ni rentabilidad demostradas.")
st.write("Cobertura empirica del 80% nominal, en test historico:")
st.json(record["historical_test_coverage"])
with st.expander("Version y advertencias"):
    st.write("UTC de generacion:",record["generated_utc"])
    st.write("SHA256 del dataset:",record["data_sha256"])
    st.write("SHA256 pesos congelados:",record["model"]["weights_sha256"])
    for item in record["notes"]:
        st.write("•",item)
st.caption("Caso docente, no asesoramiento financiero.")
