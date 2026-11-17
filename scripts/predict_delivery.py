"""Predict with saved RNN/LSTM/GRU weights from complete-delivery training.

Does not fit a scaler, train a model or modify the production application.
Example:
 python -m scripts.predict_delivery --summary reports/delivery_training/RUN/summary.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from brent_intelligence.config import DATA_PATH
from brent_intelligence.delivery_inference import forecast_candidate


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--summary", required=True, type=Path,
                   help="summary.json from a completed scripts.train_delivery run")
    p.add_argument("--data", default=DATA_PATH, type=Path,
                   help="Existing API-downloaded or manually updated Brent CSV")
    p.add_argument("--weights-dir", default=None, type=Path,
                   help="Override candidate model directory if copied")
    p.add_argument("--architecture", choices=("rnn", "lstm", "gru"),
                   default=None, help="Defaults to architecture selected in the saved run")
    p.add_argument("--output", default=None, type=Path,
                   help="Optional output JSON; original weights are never modified")
    args = p.parse_args()
    result = forecast_candidate(
        summary_path=args.summary, data_path=args.data,
        weights_dir=args.weights_dir, architecture=args.architecture
    )
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(
            result, indent=2, ensure_ascii=False, allow_nan=False
        ), encoding="utf-8")
    print(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False))


if __name__ == "__main__":
    main()
