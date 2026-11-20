"""Local HTML product: versioned downloads and calibrated historical LSTM inference.

No training, original writes, architecture selection or shared interval swapping.
"""
from concurrent.futures import ThreadPoolExecutor,as_completed
from datetime import datetime,timezone,date
from hashlib import sha256
from io import StringIO
from pathlib import Path
import json,threading
import numpy as np
import pandas as pd
import requests
import torch
from brent_intelligence.features import make_features
from brent_intelligence.experiment import Preprocessor,_predict
from brent_intelligence.modeling import RecurrentForecaster
from scripts.download_data import FRED,download_yahoo,YAHOO_CLOSE

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'reports/academic_final/UAX'
RUN=ROOT/'artifacts/runs/20261116T201630599664Z_5f12b8df'
ORIGINAL_DATA=ROOT/'data/petroleo_dataset.csv'
REQUIRED=['wti','gas_natural','tipo_bce','usd_por_eur','cny_por_usd','vix','cobre','oro','indice_dolar']
EVENTS=[
 {'date':'2008-09-15','title':'2008: crisis y caída de demanda','detail':'Contexto de crisis financiera y desplome posterior del petróleo; no atribución causal de la red.','source':'https://www.eia.gov/todayinenergy/detail.php?id=15251'},
 {'date':'2014-06-30','end_date':'2016-01-20','title':'2014–2016: oferta superior a demanda','detail':'Aumento de oferta e inventarios en un período de caída de precios.','source':'https://www.eia.gov/TodayinEnergy/detail.php?id=19451'},
 {'date':'2020-04-20','title':'2020: pandemia y shock de demanda','detail':'El futuro WTI registró precios negativos; esta anotación no afirma Brent spot negativo.','source':'https://www.eia.gov/tODAyinenergy/detail.php?id=46336'},
 {'date':'2022-02-24','title':'2022: invasión de Ucrania','detail':'Contexto de incertidumbre de oferta; no contribución calculada del modelo.','source':'https://www.eia.gov/outlooks/steo/archives/apr22.pdf'},
 {'date':'2025-09-07','title':'OPEP+: ajuste de oferta para octubre2025','detail':'Decisión de ajuste de137mil barriles diarios. Acontecimiento contextual, no explicación causal de una predicción.','source':'https://www.opec.org/pr-detail/573-07-september-2025.html'}]

def utc():return datetime.now(timezone.utc).isoformat()
def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))
def digest(path):return sha256(Path(path).read_bytes()).hexdigest()
def write(path,value):Path(path).write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')

def strict_prices(path):
    raw=pd.read_csv(path,index_col='Date',parse_dates=['Date'])
    if raw.index.has_duplicates or not raw.index.is_monotonic_increasing:raise ValueError('Fechas duplicadas/desordenadas')
    raw=raw.apply(pd.to_numeric,errors='raise')
    if 'brent' not in raw or raw.brent.isna().any() or (raw.brent<=0).any() or not np.isfinite(raw.brent).all():raise ValueError('Brent inválido')
    if raw.index[-1].date()>date.today():raise ValueError('Observaciones futuras')
    return raw

def validate_inputs(raw,columns,today=None):
    today=today or date.today();origin=raw.index[-1].date();warnings=[]
    if (today-origin).days>7:raise ValueError(f'Brent desactualizado: {origin}; actualice datos')
    if len(raw)<60:raise ValueError('Se necesitan60 sesiones')
    missing=set(REQUIRED)-set(raw.columns)
    if missing:raise ValueError('Faltan fuentes: '+', '.join(sorted(missing)))
    if np.isinf(raw[REQUIRED].to_numpy(dtype=float)).any():raise ValueError('Fuente con valores infinitos')
    lineage=[]
    # Input lag1 is exactly the frozen make_features contract. No unbounded ffill.
    for name in REQUIRED:
        observed=raw[name].iloc[:-1].dropna()
        if observed.empty:raise ValueError('Fuente sin observaciones: '+name)
        latest=observed.index[-1].date();age=(origin-latest).days
        lineage.append(dict(name=name,last_observation=latest.isoformat(),age_at_origin_days=age))
        if age>7:raise ValueError(f'Fuente {name} obsoleta ({latest}) para origen{origin}')
    features=make_features(raw)
    if list(columns)!=list(features.reindex(columns=columns).columns) or set(columns)-set(features):raise ValueError('Variables incompatibles')
    recent=features[columns].tail(60)
    if recent.iloc[-1].isna().any():raise ValueError('Entrada final incompleta: '+', '.join(recent.columns[recent.iloc[-1].isna()]))
    if (recent.notna().mean()<.5).any():raise ValueError('Cobertura insuficiente de variables')
    return features,lineage,warnings

def model_details():
    summary=read(RUN/'summary.json');prep=Preprocessor(**read(RUN/'evaluation_preprocessing.json'))
    if summary['selection']['architecture']!='lstm' or len(prep.columns)!=16:raise ValueError('Modelo de calibración incompatible')
    config=summary['config'];weights=RUN/'lstm_evaluation.pt'
    if digest(weights)!='55878cdd309ea074d5c07211f6a5dcdae0b8a034e71b4f562f64985bf3cbd125':raise ValueError('Checkpoint calibrado cambiado')
    if digest(RUN/'evaluation_preprocessing.json')!='ae1e1171417e6f2f5e37e38c1a31d95ba57ce1ceb7509999a2dbc0441138931d':raise ValueError('Preprocesamiento calibrado cambiado')
    info=dict(architecture='lstm',parameters=6499,window=60,features=16,feature_names=prep.columns,
        checkpoint=str(weights),weights_sha256=digest(weights),preprocessing_sha256=digest(RUN/'evaluation_preprocessing.json'),
        protocol='LSTM_evaluation_2021_cal2022_23',trained_until='2021-12-31',last_training_origin='2021-01-04',
        hidden_size=32,layers=1,training_epochs=3,training_seconds=9.757,
        academic_model='RNN1083: no integrada porque su checkpoint final no dispone de calibración separada compatible',
        selection='menor media fold/horizonte RMSE/ingenuo antes del test',
        validation_score=summary['models']['lstm']['validation_score'])
    return summary,prep,config,weights,info

def infer_state(path,downloaded_at=None,sources=None):
    summary,prep,cfg,weights,model_info=model_details();raw=strict_prices(path)
    history=[dict(date=str(t.date()),price=float(v)) for t,v in raw.brent.items()]
    common=dict(model=model_info,data=dict(last_date=str(raw.index[-1].date()),last_price=float(raw.brent.iloc[-1]),
        downloaded_at=downloaded_at,n_observations=len(raw),stale=(date.today()-raw.index[-1].date()).days>7,
        snapshot=str(path),sha256=digest(path),sources=sources or [],provider_sources=sources or []),history=history,events=list(EVENTS),updated_at=utc())
    maximum=raw.brent.tail(252);peak=maximum.idxmax()
    common['events'].append(dict(date=str(peak.date()),title='Máximo de las últimas252 observaciones',
        detail=f'{float(maximum.max()):.2f} USD/barril; estadística del snapshot, sin atribución causal.',
        source='https://fred.stlouisfed.org/series/DCOILBRENTEU'))
    try:feat,lineage,warnings=validate_inputs(raw,prep.columns)
    except ValueError as exc:return dict(status='blocked',horizons=[],warnings=[str(exc)],error=str(exc),**common)
    common['data']['sources']=lineage
    before=(digest(weights),digest(RUN/'evaluation_preprocessing.json'))
    torch.set_num_threads(4)
    model=RecurrentForecaster('lstm',16,cfg['hidden_size'],cfg['layers'],3)
    model.load_state_dict(torch.load(weights,map_location='cpu',weights_only=True));model.eval()
    if sum(p.numel() for p in model.parameters())!=6499:raise ValueError('Tamaño del modelo incorrecto')
    scaled=prep.x(feat).astype(np.float32)
    logret=prep.inverse_y(_predict(model,scaled[-60:][None,...])[0])
    anchor=float(raw.brent.iloc[-1]);prices=anchor*np.exp(logret)
    q=np.asarray(summary['uncertainty']['log_error_quantiles']);lower=prices*np.exp(-q);upper=prices*np.exp(q)
    if not np.isfinite(np.concatenate([prices,lower,upper])).all():raise ValueError('Salida no finita')
    assert before==(digest(weights),digest(RUN/'evaluation_preprocessing.json'))
    horizons=[]
    explanations={5:'Escenario corto: el ruido puede superar la señal; comparar con conservar el último precio.',
        63:'Escenario trimestral: sensible a cambios de tendencia y condiciones económicas.',
        252:'Escenario anual: gran incertidumbre y cambios de régimen; no es una trayectoria diaria.'}
    for j,h in enumerate((5,63,252)):
        metrics=dict(summary['models']['lstm']['test'][str(h)])
        metrics['improvement_pct']=100*(1-metrics['relative_rmse'])
        metrics.update(origin_start='2024-01-02',origin_end='2025-11-12',
            target_start={5:'2024-01-09',63:'2024-04-02',252:'2024-12-30'}[h],
            target_end={5:'2025-11-19',63:'2026-02-12',252:'2026-11-11'}[h])
        horizons.append(dict(sessions=h,price=float(prices[j]),delta_usd=float(prices[j]-anchor),
            delta_pct=float(100*(prices[j]/anchor-1)),lower=float(lower[j]),upper=float(upper[j]),nominal=.8,
            method='Cuantil80% del error absoluto logarítmico en251 anclajes separados del entrenamiento; bandas empíricas, sin garantía futura',
            explanation=explanations[h],limitations=['Pesos históricos2021; no se reentrenaron con datos actuales',
                'Horizontes cuentan observaciones; no días naturales','Exógenas históricas no certificadas PIT; error futuro puede cambiar'],metrics=metrics))
    warnings+=['LSTM histórica con calibración separada; distinta de RNN académica y del antiguo refit Streamlit2026',
        'Evaluación2024+ ya examinada: no test virgen. Las etiquetas solapadas no son independientes.',
        'Inferencia sin entrenamiento. No garantiza acierto ni rentabilidad.']
    return dict(status='ready',horizons=horizons,warnings=warnings,uncertainty=dict(nominal=.8,calibration_n=251,
        calibration_start='2022-01-03',calibration_end='2022-12-29',calibration_target_end='2023-12-29',
        historical_test_n=474,coverage=summary['uncertainty']['test_coverage'],calibrationindependent_training=True,
        independent_current_model=True,current_future_guarantee=False),**common)

def _fred(code,until):
    url=f'https://fred.stlouisfed.org/graph/fredgraph.csv?id={code}&cosd=2000-01-01&coed={until}'
    response=requests.get(url,timeout=35);response.raise_for_status()
    frame=pd.read_csv(StringIO(response.text),index_col=0,parse_dates=True,na_values='.')
    series=pd.to_numeric(frame.iloc[:,0],errors='raise').rename(FRED[code]);series.index=pd.to_datetime(series.index).normalize()
    if series.index.has_duplicates or series.dropna().empty:raise ValueError('Fuente duplicada/vacía '+code)
    return series,url

def download_version(output_dir=OUT):
    version=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    directory=Path(output_dir)/'versions'/version;directory.mkdir(parents=True,exist_ok=False)
    until=date.today().isoformat();end=(pd.Timestamp(until)+pd.Timedelta(days=1)).date().isoformat()
    frames=[];records=[];failures=[]
    def fetch(item):
        kind,code,name=item
        if kind=='FRED':series,url=_fred(code,until)
        else:series=download_yahoo(code,name,end,False).iloc[:,0];url='https://finance.yahoo.com/quote/'+code
        series=pd.to_numeric(series,errors='raise').rename(name)
        if series.index.has_duplicates or series.dropna().empty:raise ValueError('Fuente inválida '+code)
        if not series.index.is_monotonic_increasing:series=series.sort_index()
        available=series.dropna()
        return series,dict(provider=kind,id=code,name=name,url=url,last_observation=str(available.index[-1].date()),
            first_observation=str(available.index[0].date()),n=int(available.size),status='ok')
    tasks=[('FRED',code,name) for code,name in FRED.items()]+[('Yahoo',code,name) for code,name in YAHOO_CLOSE.items()]
    with ThreadPoolExecutor(max_workers=4) as pool:
        futures={pool.submit(fetch,item):item for item in tasks}
        for future in as_completed(futures):
            item=futures[future]
            try:series,record=future.result();frames.append(series);records.append(record)
            except Exception as exc:failures.append(dict(source=item[1],error=type(exc).__name__+': '+str(exc)))
    if not any(s.name=='brent' for s in frames):
        write(directory/'sources.json',dict(downloaded_at=utc(),failures=failures,sources=records))
        raise ValueError('No se pudo descargar Brent oficial; fuentes registradas en'+str(directory))
    raw=pd.concat(frames,axis=1).sort_index();raw=raw[raw.brent.notna()].copy()
    if (raw.brent<=0).any() or raw.index.has_duplicates:raise ValueError('Brent descargado inválido')
    raw.index.name='Date';path=directory/'dataset.csv';raw.to_csv(path)
    manifest=dict(downloaded_at=utc(),sources=records,failures=failures,sha256=digest(path),last_date=str(raw.index[-1].date()),
        n=len(raw),warning='Mismo contrato de variables; nuevas observaciones revisadas actuales, NO certificación PIT histórica')
    write(directory/'sources.json',manifest)
    state=infer_state(path,manifest['downloaded_at'],records)
    if failures:state['warnings'].append('Fuentes no descargadas: '+', '.join(x['source'] for x in failures))
    state['source_downloads']=records;state['download_failures']=failures
    write(directory/'state.json',state)
    write(Path(output_dir)/'active.json',dict(version=version))
    return state

class Product:
    def __init__(self,output_dir=OUT):
        self.output_dir=Path(output_dir);self.lock=threading.Lock()
    def state(self):
        pointer=self.output_dir/'active.json'
        if pointer.exists():
            version=read(pointer)['version']
            if not version.isalnum():raise ValueError('Versión de datos no válida')
            directory=self.output_dir/'versions'/version
            manifest=read(directory/'sources.json')
            if digest(directory/'dataset.csv')!=manifest['sha256']:raise ValueError('Snapshot modificado')
            return infer_state(directory/'dataset.csv',manifest['downloaded_at'],manifest['sources'])
        return infer_state(ORIGINAL_DATA)
    def refresh(self):
        if not self.lock.acquire(blocking=False):raise ValueError('Ya hay una actualización en curso')
        try:return download_version(self.output_dir)
        finally:self.lock.release()
