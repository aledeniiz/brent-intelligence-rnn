"""Deterministic invented observations; no provider data and no training."""
from pathlib import Path
import hashlib,json
import numpy as np
import pandas as pd
from brent_intelligence.features import make_features
from brent_intelligence.experiment import Preprocessor
ROOT=Path(__file__).resolve().parents[1]
def build():
    m=json.loads((ROOT/'web/pages/assets/model/manifest.json').read_text(encoding='utf-8'))
    t=np.arange(100,dtype=float)
    raw=pd.DataFrame(index=pd.bdate_range('2021-01-04',periods=100))
    raw['brent']=70*np.exp(.0007*t+.04*np.sin(t/9)+.01*np.sin(t/2))
    raw['wti']=raw.brent-3+.3*np.cos(t/7)
    for key,base,scale in [('gas_natural',3,.03),('usd_por_eur',1.1,.01),('cny_por_usd',6.8,.01),('vix',20,.05),('cobre',4,.02),('oro',1800,.01),('indice_dolar',100,.01)]:
        raw[key]=base*np.exp(scale*np.sin(t/11)+.0001*t)
    raw['tipo_bce']=.25
    prep=Preprocessor(**m['scaler']);x=prep.x(make_features(raw))[-60:]
    assert x.shape==(60,16) and np.isfinite(x).all()
    packet=dict(kind='synthetic',schema_version=1,fixture_version='uax-synthetic-v1',license='CC0-1.0',model_id=m['model_id'],onnx_sha256=m['onnx_sha256'],scaler_sha256=m['scaler_sha256'],feature_names=m['feature_names'],origin=dict(date=str(raw.index[-1].date()),price=float(raw.brent.iloc[-1])),input=dict(shape=[1,60,16],values=x.flatten().tolist(),sha256=hashlib.sha256(x.astype('<f4').tobytes()).hexdigest()),raw=[dict(date=str(i.date()),**{k:float(v) for k,v in r.items()}) for i,r in raw.iterrows()],history=[dict(date=str(i.date()),price=float(v)) for i,v in raw.brent.items()],sources=[],used_sources=['Generador matemático propio CC0; ninguna cotización observada'],events=[],generated_utc=None,warning='Datos totalmente sintéticos. Fechas ficticias fijas; no son cotizaciones actuales, datos históricos de mercado ni pronósticos utilizables. Bandas históricas sin cobertura validada en este escenario.')
    target=ROOT/'web/pages/assets/synthetic';target.mkdir(exist_ok=True)
    body=json.dumps(packet,ensure_ascii=False,separators=(',',':'),allow_nan=False).encode()
    (target/'scenario.json').write_bytes(body)
    policy=dict(schema_version=1,complete_live_inputs_approved=False,live_input_url=None,blocked_features=['vix_log_level','cobre_log_change_5','oro_log_change_5','indice_dolar_log_change_5'],synthetic_sha256=hashlib.sha256(body).hexdigest(),synthetic_url='synthetic/scenario.json',license='CC0-1.0',reason='No están acreditados todos los derechos de redistribución. No se consultan fuentes incompletas ni se mezclan con el escenario sintético.')
    (ROOT/'web/pages/assets/data-policy.json').write_text(json.dumps(policy,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({'fixture_sha256':policy['synthetic_sha256'],'shape':x.shape,'kind':'synthetic'}))
if __name__=='__main__':build()
