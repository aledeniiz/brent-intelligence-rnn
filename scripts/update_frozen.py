"""Actualiza solo la prediccion del modelo congelado, nunca lo reentrena."""
from __future__ import annotations
import argparse
from pathlib import Path
from brent_intelligence.config import DATA_PATH
from brent_intelligence.frozen import FROZEN_DIR,PUBLISHED_FORECAST,forecast_frozen

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data",type=Path,default=DATA_PATH)
    p.add_argument("--bundle",type=Path,default=FROZEN_DIR)
    p.add_argument("--output",type=Path,default=PUBLISHED_FORECAST)
    p.add_argument("--allow-stale",action="store_true",
                   help="SOLO pruebas; no usar para publicar datos caducados")
    args=p.parse_args()
    result=forecast_frozen(data=args.data,bundle=args.bundle,
                           output=args.output,allow_stale=args.allow_stale)
    report=result["forecast"]
    print("changed:",result["changed"])
    print("last_date:",report["last_date"])
    print("architecture:",report["model"]["architecture"])
    print("prediction_prices:",report["prices"])
    print("output:",args.output)

if __name__=="__main__":
    main()
