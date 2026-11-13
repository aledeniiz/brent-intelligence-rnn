"""Entrena, compara y versiona RNN/LSTM/GRU."""
import argparse
from dataclasses import replace
import json
from pathlib import Path
from brent_intelligence.config import DATA_PATH, ARTIFACTS_DIR, ExperimentConfig
from brent_intelligence.experiment import run_experiment

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data",type=Path,default=DATA_PATH)
    parser.add_argument("--artifacts",type=Path,default=ARTIFACTS_DIR)
    parser.add_argument("--epochs",type=int,default=20)
    parser.add_argument("--hidden-size",type=int,default=32)
    parser.add_argument("--device",choices=["cpu","cuda"],default="cpu")
    parser.add_argument("--dry-run",action="store_true")
    parser.add_argument("--smoke",action="store_true")
    args=parser.parse_args()
    cfg=replace(ExperimentConfig(),epochs=args.epochs,hidden_size=args.hidden_size)
    result=run_experiment(data_path=args.data,artifact_dir=args.artifacts,
                          cfg=cfg,smoke=args.smoke,device=args.device,
                          dry_run=args.dry_run)
    print(json.dumps({
        "mode":result["mode"],
        "selection":result.get("selection"),
        "validation_scores":{k:v["validation_score"] for k,v in result["models"].items()},
        "test":{k:v["test"] for k,v in result["models"].items()} if not args.dry_run else {}
    },indent=2,ensure_ascii=False))

if __name__=="__main__":
    main()
