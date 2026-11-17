"""Run fully trained independent Brent models for 5, 63 and 252 sessions.

Default compares three architectures on historical temporal folds, then
fits independent full-history candidates. Never overwrites production.
"""
from __future__ import annotations
import argparse
from dataclasses import replace
import json
from pathlib import Path

from brent_intelligence.config import DATA_PATH, REPORTS_DIR, ARTIFACTS_DIR
from brent_intelligence.horizon_models import HorizonConfig, run_horizon_models


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DATA_PATH)
    parser.add_argument("--reports", type=Path, default=REPORTS_DIR/"horizon_v5")
    parser.add_argument("--models", type=Path, default=ARTIFACTS_DIR/"horizon_candidates")
    parser.add_argument("--dry-run", action="store_true",
                        help="Inspect folds and all mature labels; no training")
    parser.add_argument("--quick", action="store_true",
                        help="One RNN, horizon 5, one fold, one epoch: smoke only")
    parser.add_argument("--allow-stale", action="store_true",
                        help="For synthetic/offline tests only, not submission")
    args = parser.parse_args()
    cfg = HorizonConfig()
    if args.quick:
        cfg = replace(cfg,architectures=("rnn",),horizons=(5,),
                      folds=1,epochs=1,patience=1,hidden_size=8)
    run = run_horizon_models(
        data_path=args.data, report_dir=args.reports, model_dir=args.models,
        cfg=cfg, dry_run=args.dry_run, allow_stale=args.allow_stale
    )
    if args.dry_run:
        print(json.dumps(run,indent=2,ensure_ascii=False))
        return
    print("\nHORIZONTE / ARQUITECTURA / VALIDACION")
    for h, row in run["architecture_selection"].items():
        chosen=row["chosen_architecture"]
        r=row["candidates"][chosen]
        print(f" h{h}: {chosen.upper()} "
              f"RMSE/ingenuo={r['mean_fold_rmse_ratio']:.4f} "
              f"folds ganados={r['won_folds']}/{r['folds']}")
    print("Historico 2024+ NO es test virgen.")
    for h, row in run["historical_diagnostic"].items():
        m=row["metric"]
        print(f" h{h}: RMSE={m['rmse']:.4f}, ingenuo={m['naive_rmse']:.4f}, "
              f"mejora={m['improvement_pct']:+.2f}%, AUC="
              f"{(m['direction'] or {}).get('roc_auc')}")
    print("Modelos finales hasta hoy:",run["candidate_weights"])
    print("Reporte y curvas:",run["output"])
    print("Produccion congelada: INTACTA.")


if __name__ == "__main__":
    main()
