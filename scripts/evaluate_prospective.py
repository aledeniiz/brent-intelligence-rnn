"""Evaluate matured archived forecasts; optional legacy import, never inference/training."""
import argparse
from pathlib import Path
from brent_intelligence.config import DATA_PATH
from brent_intelligence.frozen import PUBLISHED_FORECAST,read_json
from brent_intelligence.prospective import (
    EVALUATION_PATH,JOURNAL_DIR,archive_forecast,evaluate_prospective,
)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data",type=Path,default=DATA_PATH)
    p.add_argument("--history",type=Path,default=JOURNAL_DIR)
    p.add_argument("--output",type=Path,default=EVALUATION_PATH)
    p.add_argument("--import-latest",action="store_true")
    args=p.parse_args()
    if args.import_latest:
        archive_forecast(read_json(PUBLISHED_FORECAST),directory=args.history,provenance="legacy_import")
    result=evaluate_prospective(data=args.data,directory=args.history,output=args.output)
    print("Archived forecasts:",result["forecast_records"])
    print("Matured prospective points:",result["evaluated_points"])
    print("Output:",args.output)


if __name__=="__main__":
    main()
