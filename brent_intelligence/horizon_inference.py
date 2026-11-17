"""Forward-only inference with three saved, independently trained horizon models."""
from __future__ import annotations
from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path
import numpy as np
import torch

from brent_intelligence.config import DATA_PATH, ARTIFACTS_DIR
from brent_intelligence.features import load_prices, make_features
from brent_intelligence.horizon_models import (
    DirectRecurrent, HorizonConfig, HorizonScaler, return_to_price
)


def predict_from_saved(*, summary_path: Path,
                       data_path: Path = DATA_PATH,
                       weights_dir: Path | None = None) -> dict:
    summary_path = Path(summary_path)
    run = json.loads(summary_path.read_text(encoding="utf-8"))
    if run.get("mode") != "independent-horizon-complete-data":
        raise ValueError("Expected independent-horizon training summary")
    cfg = HorizonConfig(**run["config"])
    cfg.validate()
    raw = load_prices(data_path)
    latest = raw.index[-1]
    if weights_dir is None:
        weights_dir = (ARTIFACTS_DIR / "horizon_candidates" /
                       summary_path.parent.name)
    weights_dir = Path(weights_dir)
    features = make_features(raw)
    result = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "latest_observed_price_date": str(latest.date()),
        "latest_observed_price": float(raw["brent"].iloc[-1]),
        "data_sha256": sha256(Path(data_path).read_bytes()).hexdigest(),
        "retrained": False,
        "architectures": {},
        "predictions": {},
        "source_model_report": str(summary_path),
    }
    for h, info in run["final_models"].items():
        if latest < __import__("pandas").Timestamp(info["train_until"]):
            raise ValueError(f"New data are older than the model h{h}")
        scaler = HorizonScaler(**info["scaler"])
        missing = sorted(set(scaler.columns) - set(features.columns))
        if missing:
            raise ValueError(f"Missing required features for h{h}: {missing}")
        recent = features[scaler.columns].tail(cfg.window)
        if len(recent) != cfg.window or (
            (recent.notna().mean() < .5).any()
            or recent.iloc[-1].isna().any()
        ):
            raise ValueError(f"Missing latest input values for h{h}")
        weights = weights_dir / info["weights_file"]
        digest = sha256(weights.read_bytes()).hexdigest()
        if digest != info["weights_sha256"]:
            raise ValueError(f"Checkpoint integrity mismatch for h{h}")
        model = DirectRecurrent(info["architecture"], len(scaler.columns), cfg)
        model.load_state_dict(torch.load(
            weights, map_location="cpu", weights_only=True
        ))
        model.eval()
        seq = scaler.transform(features)[-cfg.window:][None,:,:]
        with torch.no_grad():
            scaled = float(model(torch.from_numpy(seq)).item())
        price = float(return_to_price(
            np.array([float(raw["brent"].iloc[-1])]),
            np.array([scaled]), scaler.target_rms, info["selected"]["alpha"]
        )[0])
        result["predictions"][h] = price
        result["architectures"][h] = {
            "architecture": info["architecture"],
            "trained_until": info["train_until"],
            "weights_sha256": digest,
            "shrinkage_alpha": info["selected"]["alpha"],
        }
    return result
