"""Run controlled Brent v4 experiments; no production model is overwritten."""
from __future__ import annotations
import argparse
from dataclasses import replace
from pathlib import Path

from brent_intelligence.config import DATA_PATH, REPORTS_DIR
from brent_intelligence.research_v4 import V4Config, run_experiment


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DATA_PATH)
    parser.add_argument("--output", type=Path, default=REPORTS_DIR / "research_v4")
    parser.add_argument("--architecture", choices=("lstm", "gru"), default="lstm")
    parser.add_argument("--quick", action="store_true",
                        help="One seed, one fold, four epochs. A code smoke test only.")
    parser.add_argument("--dry-run", action="store_true",
                        help="Show fold splits and dataset provenance, do not train.")
    args = parser.parse_args()
    cfg = V4Config(architecture=args.architecture)
    if args.quick:
        cfg = replace(cfg, seeds=(42,), max_folds=1, max_epochs=4,
                      hidden_size=8)
    result = run_experiment(data_path=args.data,
                            output_dir=args.output / ("quick" if args.quick else "full"),
                            cfg=cfg, dry_run=args.dry_run)
    if args.dry_run:
        print("Research V4 dry-run: no training and no files written")
        for fold in result["folds"]:
            print(fold)
    else:
        print("RESULTADOS V4 · VALIDACION HASTA 2021, NO TEST")
        for row in result["aggregate"]:
            print(f"{row['horizon']:>3} sesiones: "
                  f"RMSE={row['rmse']:.4f}, ingenuo={row['naive_rmse']:.4f}, "
                  f"mejora={row['improvement_pct']:+.2f}%, "
                  f"folds ganados={row['folds_better_than_naive']}/{row['n_folds']}")
        print("Informe y graficos:", result["saved_to"])
        print("Modelo original y actualizacion congelada: SIN CAMBIOS.")


if __name__ == "__main__":
    main()
