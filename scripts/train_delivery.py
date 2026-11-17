"""Train complete RNN/LSTM/GRU models on available Brent data for submission.
Saves only candidate weights; frozen production and apps are not replaced.
"""
from __future__ import annotations
import argparse
import json
from dataclasses import replace
from pathlib import Path
from brent_intelligence.config import DATA_PATH, REPORTS_DIR, ARTIFACTS_DIR
from brent_intelligence.delivery_training import DeliveryConfig, run_delivery

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", type=Path, default=DATA_PATH)
    p.add_argument("--reports", type=Path, default=REPORTS_DIR / "delivery_training")
    p.add_argument("--models", type=Path, default=ARTIFACTS_DIR / "delivery_candidates")
    p.add_argument("--dry-run", action="store_true",
                   help="Inspect splits; no training or files")
    p.add_argument("--quick", action="store_true",
                   help="One LSTM and fold at one epoch for a code smoke test")
    p.add_argument("--allow-stale", action="store_true",
                   help="Only for offline synthetic tests; not final runs")
    args = p.parse_args()
    cfg = DeliveryConfig()
    if args.quick:
        cfg = replace(cfg, architectures=("lstm",), n_folds=1,
                      hidden_size=8, max_epochs=1, patience=1)
    result = run_delivery(
        data_path=args.data, output_dir=args.reports, model_dir=args.models,
        cfg=cfg, dry_run=args.dry_run, allow_stale=args.allow_stale
    )
    if args.dry_run:
        print(json.dumps(result, indent=2, ensure_ascii=False))
        return
    print("Dataset through:", result["data"]["last_date"])
    print("Selected from old folds:", result["selected_architecture"])
    print("Teacher evaluation: train through 2023, compare observed 2024+")
    print("(2024+ has been inspected before: NOT a pristine test)")
    for architecture, values in result["historical_test_diagnostic"]["models"].items():
        print(architecture.upper())
        for h, m in values["metrics"].items():
            print(f"  {h} sessions: R2={m['r2']}, MAE={m['mae']:.4f}, "
                  f"MSE={m['mse']:.4f}, RMSE={m['rmse']:.4f}, "
                  f"naive={m['naive_rmse']:.4f}, "
                  f"improvement={m['improvement_pct']:+.2f}%")
    print("All final model candidates:", result["candidate_weights_dir"])
    print("Report and curves:", result["saved_to"])
    print("Production models, apps and active.json: unchanged.")

if __name__ == "__main__":
    main()
