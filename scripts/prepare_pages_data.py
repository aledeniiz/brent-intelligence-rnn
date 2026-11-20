"""Prepare validated 60x16 inputs, not forecasts; never load or train weights."""
from pathlib import Path
from datetime import date,datetime,timezone
from concurrent.futures import ThreadPoolExecutor
from io import StringIO
import argparse,json,hashlib,tempfile,os
import numpy as np
import pandas as pd
from brent_intelligence.uax_product import ROOT,strict_prices,validate_inputs,_fred,EVENTS
from brent_intelligence.experiment import Preprocessor
from scripts.download_data import FRED,YAHOO_CLOSE,download_yahoo


def prepare(raw,manifest,source_records):
    prep=Preprocessor(**manifest['scaler']);feat,lineage,_=validate_inputs(raw,prep.columns)
    matrix=prep.x(feat)[-60:]
    if matrix.shape!=(60,16) or not np.isfinite(matrix).all():raise ValueError('Invalid prepared tensor')
    latest=raw.index[-1].date()
    if latest>date.today():raise ValueError('Future origin')
    peak=raw.brent.tail(252).idxmax()
    events=list(EVENTS)+[dict(date=str(peak.date()),title='Máximo de las últimas252 observaciones',
        detail=f'{float(raw.brent.loc[peak]):.2f} USD/barril; dato observado, no atribución causal.',
        source='https://fred.stlouisfed.org/series/DCOILBRENTEU')]
    return dict(schema_version=1,data_version=latest.isoformat()+'_'+hashlib.sha256(matrix.astype('<f4').tobytes()).hexdigest()[:12],model_id=manifest['model_id'],onnx_sha256=manifest['onnx_sha256'],
        scaler_sha256=manifest['scaler_sha256'],feature_names=prep.columns,
        origin=dict(date=latest.isoformat(),price=float(raw.brent.iloc[-1])),
        input=dict(shape=[1,60,16],values=matrix.flatten().tolist(),sha256=hashlib.sha256(matrix.astype('<f4').tobytes()).hexdigest()),
        generated_utc=datetime.now(timezone.utc).isoformat(),sources=source_records,used_sources=lineage,events=events,
        history=[dict(date=str(t.date()),price=float(v)) for t,v in raw.brent.items()],
        warning='Inputs derived from latest revised sources; historical exogenous PIT not certified. Source redistribution rights require approval.')


def main():
    p=argparse.ArgumentParser();p.add_argument('--data',type=Path);p.add_argument('--output',type=Path,default=ROOT/'web/pages/assets/data/latest.json');a=p.parse_args()
    manifest=json.loads((ROOT/'web/pages/assets/model/manifest.json').read_text(encoding='utf-8'))
    records=[]
    if a.data:raw=strict_prices(a.data)
    else:
        end=(pd.Timestamp(date.today())+pd.Timedelta(days=1)).date().isoformat()
        def get(item):
            kind,code,name=item
            if kind=='FRED':series,url=_fred(code,date.today().isoformat())
            else:series=download_yahoo(code,name,end,False).iloc[:,0];url='https://finance.yahoo.com/quote/'+code
            valid=series.dropna()
            if series.index.has_duplicates or valid.empty:raise ValueError('Invalid source '+code)
            return series.rename(name),dict(provider=kind,id=code,url=url,last_observation=str(valid.index.max().date()))
        tasks=[('FRED',k,v) for k,v in FRED.items()]+[('Yahoo',k,v) for k,v in YAHOO_CLOSE.items()]
        with ThreadPoolExecutor(max_workers=4) as pool:results=list(pool.map(get,tasks))
        raw=pd.concat([s for s,_ in results],axis=1).sort_index();raw=raw.loc[raw.brent.notna()];raw.index.name='Date'
        records=[r for _,r in results]
        with tempfile.TemporaryDirectory() as tmp:
            file=Path(tmp)/'data.csv';raw.to_csv(file);raw=strict_prices(file)
    packet=prepare(raw,manifest,records)
    a.output.parent.mkdir(parents=True,exist_ok=True)
    if a.output.exists():
        previous=json.loads(a.output.read_text(encoding='utf-8'))
        if packet['origin']['date']<=previous['origin']['date']:
            print('No newer valid Brent origin: no publication');return
    body=json.dumps(packet,ensure_ascii=False,separators=(',',':'))
    version=a.output.parent/'versions'/(packet['origin']['date']+'_'+hashlib.sha256(body.encode()).hexdigest()[:12]+'.json')
    version.parent.mkdir(exist_ok=True);version.write_text(body,encoding='utf-8')
    temp=a.output.with_suffix('.tmp');temp.write_text(body,encoding='utf-8');os.replace(temp,a.output)
    print(json.dumps({'prepared':True,'origin':packet['origin'],'shape':packet['input']['shape'],'version':str(version)}))


if __name__=='__main__':main()
