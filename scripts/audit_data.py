"""Informe de integridad del CSV sin reemplazar precios faltantes."""
from __future__ import annotations
import json
import pandas as pd
from brent_intelligence.config import DATA_PATH,REPORTS_DIR
from brent_intelligence.features import load_prices

def audit(path=DATA_PATH)->dict:
    raw=pd.read_csv(path,parse_dates=["Date"])
    raw=raw.sort_values("Date")
    dup=int(raw["Date"].duplicated().sum())
    invalid=int((pd.to_numeric(raw["brent"],errors="coerce")<=0).sum())
    df=load_prices(path)
    results={
        "rows":len(df),"columns":len(df.columns),
        "first_date":str(df.index.min().date()),
        "last_date":str(df.index.max().date()),
        "duplicate_raw_dates":dup,"nonpositive_raw_brent":invalid,
        "wti_negative_count":int((df["wti"]<0).sum()) if "wti" in df else None,
        "missing_by_column":df.isna().sum().astype(int).to_dict(),
        "missing_pct_by_column":(100*df.isna().mean()).round(2).to_dict(),
        "brent_min":float(df["brent"].min()),
        "brent_max":float(df["brent"].max()),
        "warning":"Algunos datos historicos pueden haberse revisado posteriormente a la fecha de observacion"
    }
    REPORTS_DIR.mkdir(parents=True,exist_ok=True)
    (REPORTS_DIR/"data_audit.json").write_text(json.dumps(results,indent=2,ensure_ascii=False),encoding="utf-8")
    print(json.dumps(results,indent=2,ensure_ascii=False))
    return results

if __name__=="__main__":
    audit()
