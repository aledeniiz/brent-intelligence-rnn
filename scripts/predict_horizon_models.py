"""Generate updated forecasts using saved 5/63/252 horizon weights (NO training)."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
from brent_intelligence.config import DATA_PATH
from brent_intelligence.horizon_inference import predict_from_saved

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary",required=True,type=Path)
    parser.add_argument("--data",type=Path,default=DATA_PATH)
    parser.add_argument("--weights-dir",type=Path,default=None)
    parser.add_argument("--output",type=Path,default=None)
    args=parser.parse_args()
    result=predict_from_saved(summary_path=args.summary,
              data_path=args.data,weights_dir=args.weights_dir)
    if args.output is not None:
        args.output.parent.mkdir(parents=True,exist_ok=True)
        args.output.write_text(json.dumps(
            result,indent=2,ensure_ascii=False,allow_nan=False),
            encoding="utf-8")
    print(json.dumps(result,indent=2,ensure_ascii=False,allow_nan=False))

if __name__ == "__main__":
    main()
