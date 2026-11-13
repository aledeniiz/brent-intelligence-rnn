"""FRED/EIA obligatorio; Yahoo opcional; calendario de Brent observado."""
from __future__ import annotations
import argparse
from datetime import date,timedelta
import json
import logging
from pathlib import Path
import pandas as pd
from brent_intelligence.config import DATA_PATH,REPORTS_DIR

LOG=logging.getLogger(__name__)
START="2000-01-01"
FRED={"DCOILBRENTEU":"brent","DCOILWTICO":"wti","DHHNGSP":"gas_natural",
      "ECBDFR":"tipo_bce","DEXUSEU":"usd_por_eur","DEXCHUS":"cny_por_usd","VIXCLS":"vix"}
YAHOO_FUTURES={"BZ=F":"brent_fut","CL=F":"wti_fut"}
YAHOO_CLOSE={"HG=F":"cobre","GC=F":"oro","DX-Y.NYB":"indice_dolar"}

def download_fred(code:str,until:str)->pd.Series:
    url="https://fred.stlouisfed.org/graph/fredgraph.csv"
    url+=f"?id={code}&cosd={START}&coed={until}"
    df=pd.read_csv(url,index_col=0,parse_dates=True,na_values=".")
    if df.empty:
        raise ValueError(f"FRED sin datos: {code}")
    serie=pd.to_numeric(df.iloc[:,0],errors="coerce")
    serie.index=pd.to_datetime(serie.index).normalize()
    return serie.rename(FRED[code])

def download_yahoo(ticker:str,prefix:str,end_exclusive:str,ohlcv:bool)->pd.DataFrame:
    import yfinance as yf
    df=yf.download(ticker,start=START,end=end_exclusive,
                   auto_adjust=False,progress=False,threads=False)
    if df.empty:
        raise ValueError(f"Yahoo sin datos para {ticker}")
    if isinstance(df.columns,pd.MultiIndex):
        df.columns=df.columns.get_level_values(0)
    dates=pd.to_datetime(df.index)
    if dates.tz is not None:
        dates=dates.tz_localize(None)
    df.index=dates.normalize()
    if ohlcv:
        return df[["Open","High","Low","Close","Volume"]].add_prefix(prefix+"_")
    return df[["Close"]].rename(columns={"Close":prefix})

def download_dataset(path:Path=DATA_PATH,until:str|None=None,
                     include_yahoo:bool=True,orb_path:Path|None=None)->pd.DataFrame:
    until=until or date.today().isoformat()
    end_exclusive=(date.fromisoformat(until)+timedelta(days=1)).isoformat()
    statuses={"FRED":{},"Yahoo":{},"ORB":"not supplied"}
    # Solo Brent es obligatorio. Fallos auxiliares se registran.
    frames=[download_fred("DCOILBRENTEU",until)]
    statuses["FRED"]["DCOILBRENTEU"]="ok"
    for code in FRED:
        if code=="DCOILBRENTEU":continue
        try:
            frames.append(download_fred(code,until))
            statuses["FRED"][code]="ok"
        except Exception as exc:
            LOG.warning("FRED %s: %s",code,exc)
            statuses["FRED"][code]=f"unavailable: {type(exc).__name__}"
    if include_yahoo:
        for ticker,prefix in {**YAHOO_FUTURES,**YAHOO_CLOSE}.items():
            try:
                frames.append(download_yahoo(ticker,prefix,end_exclusive,ticker in YAHOO_FUTURES))
                statuses["Yahoo"][ticker]="ok"
            except Exception as exc:
                LOG.warning("Yahoo %s: %s",ticker,exc)
                statuses["Yahoo"][ticker]=f"unavailable: {type(exc).__name__}"
    if orb_path is not None:
        orb=pd.read_csv(orb_path,parse_dates=["Date"]).set_index("Date")
        if "orb" not in orb.columns:
            raise ValueError("El CSV OPEP debe contener Date y orb")
        frames.append(pd.to_numeric(orb["orb"],errors="coerce"))
        statuses["ORB"]=str(orb_path)
    df=pd.concat(frames,axis=1,sort=True).sort_index()
    df.index=pd.to_datetime(df.index).normalize()
    df=df.loc[~df.index.duplicated(keep="last")].sort_index()
    df=df.loc[df["brent"].notna()&(df["brent"]>0)]
    df=df.loc[df.index.day_of_week<5]
    if len(df)<1000:
        raise ValueError("Historial Brent insuficiente; dataset previo sin modificar")
    df.index.name="Date"
    path=Path(path)
    path.parent.mkdir(parents=True,exist_ok=True)
    df.to_csv(path)
    REPORTS_DIR.mkdir(parents=True,exist_ok=True)
    manifest={"start":str(df.index.min().date()),"end":str(df.index.max().date()),
              "rows":len(df),"columns":list(df.columns),
              "downloaded_utc":pd.Timestamp.now(tz="UTC").isoformat(),
              "source_status":statuses,
              "warning":"Las variables externas no estan verificadas point-in-time"}
    (REPORTS_DIR/"download_sources.json").write_text(json.dumps(manifest,indent=2,ensure_ascii=False),encoding="utf-8")
    LOG.info("%s: %s observaciones, %s columnas",path,len(df),len(df.columns))
    return df

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--until",default=None)
    p.add_argument("--output",type=Path,default=DATA_PATH)
    p.add_argument("--fred-only",action="store_true")
    p.add_argument("--orb-csv",type=Path,default=None)
    a=p.parse_args()
    logging.basicConfig(level=logging.INFO,format="%(levelname)s %(message)s")
    download_dataset(a.output,a.until,include_yahoo=not a.fred_only,orb_path=a.orb_csv)

if __name__=="__main__":
    main()
