"""Frozen-model packaging and safe, zero-training forecasts for scheduled updates.

A frozen snapshot is intentionally small and public: model weights, input scaler,
model hyperparameters and a limited, audited historical performance summary.
No dataset, training labels, secrets or raw private artifacts are published.
"""
from __future__ import annotations

from datetime import datetime, timezone
from hashlib import sha256
import json
import os
from pathlib import Path
import shutil
import tempfile

import numpy as np
import pandas as pd
import torch

from brent_intelligence.config import ARTIFACTS_DIR, DATA_PATH, ROOT
from brent_intelligence.experiment import Preprocessor, _predict
from brent_intelligence.features import load_prices, make_features
from brent_intelligence.modeling import RecurrentForecaster

FROZEN_DIR = ROOT / "deployment" / "frozen"
PUBLISHED_FORECAST = ROOT / "public" / "latest_forecast.json"


def file_hash(path: Path) -> str:
    return sha256(Path(path).read_bytes()).hexdigest()


def read_json(path: Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def save_json_atomic(path: Path, value: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=path.parent, suffix=".tmp", delete=False
    ) as tmp:
        json.dump(value, tmp, indent=2, ensure_ascii=False, allow_nan=False)
        tmp.write("\n")
        temp_name = tmp.name
    os.replace(temp_name, path)


def _active_run(artifacts: Path) -> Path:
    pointer = read_json(Path(artifacts) / "active.json")["run"]
    if Path(pointer).is_absolute() or ".." in Path(pointer).parts:
        raise ValueError("Ruta de ejecucion invalida")
    run = Path(artifacts) / pointer
    if not run.is_dir():
        raise FileNotFoundError(f"No existe el directorio de modelos {run}")
    return run


def export_frozen(*, artifacts: Path = ARTIFACTS_DIR, data: Path = DATA_PATH,
                  destination: Path = FROZEN_DIR) -> dict:
    """Snapshot the evaluated production model, never run training.

    Must execute before refreshing the CSV, otherwise the saved training-data
    checksum will intentionally differ.
    """
    run = _active_run(artifacts)
    summary = read_json(run / "summary.json")
    if summary.get("mode") != "historical-evaluation":
        raise ValueError("No se puede congelar un modelo smoke")
    if file_hash(data) != summary["data_sha256"]:
        raise ValueError("CSV distinto al utilizado para entrenar: exporta antes de refrescar")
    config = summary["config"]
    architecture = summary["selection"]["architecture"]
    if architecture not in ("rnn", "lstm", "gru"):
        raise ValueError("Arquitectura no soportada")
    if len(summary.get("folds", [])) != 5:
        raise ValueError("Faltan folds de validacion del entrenamiento definitivo")
    weights = run / "production.pt"
    scaler = run / "production_preprocessing.json"
    if not weights.is_file() or not scaler.is_file():
        raise FileNotFoundError("No estan los pesos de produccion y su escalador")
    preprocessing = read_json(scaler)
    prep = Preprocessor(**preprocessing)
    if not prep.columns:
        raise ValueError("Escalador sin columnas")
    if len(prep.columns) != len(prep.x_mean) or len(prep.columns) != len(prep.x_std):
        raise ValueError("Escalador incompatible")
    # Prove state_dict shape matches recorded configuration and scaler.
    model = RecurrentForecaster(architecture, len(prep.columns),
                                int(config["hidden_size"]), int(config["layers"]),
                                len(config["horizons"]))
    model.load_state_dict(torch.load(weights, map_location="cpu", weights_only=True))
    model.eval()
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    dest_weights = destination / "production.pt"
    shutil.copyfile(weights, dest_weights)
    save_json_atomic(destination / "preprocessing.json", preprocessing)
    manifest = {
        "format_version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "architecture": architecture,
        "config": {k: config[k] for k in ("window", "horizons", "hidden_size", "layers")},
        "weights_sha256": file_hash(dest_weights),
        "training_data_sha256": summary["data_sha256"],
        "trained_data_end": summary["production"]["last_known_price_date"],
        "last_matured_training_origin":
            summary["production"]["last_matured_training_origin"],
        "n_training_examples": summary["production"]["n_examples"],
        "input_features": prep.columns,
        "calibration": {
            "nominal_coverage": summary["uncertainty"]["nominal"],
            "log_error_quantiles": summary["uncertainty"]["log_error_quantiles"],
            "historical_test_coverage": summary["uncertainty"]["test_coverage"],
            "n_calibration": summary["uncertainty"]["n_calibration"],
        },
        "evaluation": {
            "fold_count": len(summary["folds"]),
            "selection_criterion": summary["selection"]["criterion"],
            "selected_model_historical_test": summary["models"][architecture]["test"],
            "all_models_historical_test": {
                name: item["test"] for name, item in summary["models"].items()
            },
        },
        "caveats": [
            "Modelo de produccion congelado: NO tiene test independiente propio.",
            "El backtest historico pertenece a los modelos de evaluacion.",
            "Variables externas sin certificacion point-in-time.",
            "Intervalos nominales empiricos, no garantizados.",
        ],
    }
    save_json_atomic(destination / "manifest.json", manifest)
    return manifest


def validate_bundle(directory: Path = FROZEN_DIR) -> tuple[dict, Preprocessor]:
    directory = Path(directory)
    manifest = read_json(directory / "manifest.json")
    if manifest.get("format_version") != 1:
        raise ValueError("Version de bundle desconocida")
    if file_hash(directory / "production.pt") != manifest["weights_sha256"]:
        raise ValueError("Los pesos no coinciden con el SHA256 del manifest")
    prep = Preprocessor(**read_json(directory / "preprocessing.json"))
    config = manifest["config"]
    if (not prep.columns or prep.columns != manifest["input_features"]
        or len(prep.columns) != len(prep.medians)
        or len(prep.columns) != len(prep.x_mean)
        or len(prep.columns) != len(prep.x_std)
        or len(prep.y_mean) != len(config["horizons"])):
        raise ValueError("Escalador incompatible con el modelo publicado")
    return manifest, prep


def forecast_frozen(*, data: Path = DATA_PATH, bundle: Path = FROZEN_DIR,
                    output: Path = PUBLISHED_FORECAST, allow_stale: bool = False,
                    min_feature_coverage: float = 0.50) -> dict:
    """Forecast without fitting, training, optimizers or mutable model weights."""
    manifest, prep = validate_bundle(bundle)
    raw = load_prices(data)
    latest_date = raw.index[-1].date()
    age = (datetime.now(timezone.utc).date() - latest_date).days
    if not allow_stale and age > 7:
        raise ValueError(f"Brent desactualizado ({latest_date}); publicacion cancelada")
    if age < -1:
        raise ValueError("Fecha de Brent en el futuro")
    feat = make_features(raw)
    window = int(manifest["config"]["window"])
    if len(feat) < window:
        raise ValueError("Historial insuficiente para la ventana recurrente")
    missing = sorted(set(prep.columns) - set(feat.columns))
    if missing:
        raise ValueError(f"Faltan fuentes o variables necesarias: {missing}")
    recent = feat[prep.columns].tail(window)
    coverage = recent.notna().mean()
    unusable = coverage[coverage < min_feature_coverage].index.tolist()
    if unusable:
        raise ValueError(f"Fuentes demasiado incompletas para inferencia: {unusable}")
    last_missing = recent.columns[recent.iloc[-1].isna()].tolist()
    if last_missing:
        raise ValueError(f"Faltan variables para la ultima fecha: {last_missing}")
    config = manifest["config"]
    model = RecurrentForecaster(manifest["architecture"], len(prep.columns),
                                int(config["hidden_size"]), int(config["layers"]),
                                len(config["horizons"]))
    model.load_state_dict(torch.load(Path(bundle) / "production.pt",
                                     map_location="cpu", weights_only=True))
    scaled = prep.x(feat).astype(np.float32)
    predicted_scaled = _predict(model, scaled[-window:][None, ...])[0]
    predicted_log = prep.inverse_y(predicted_scaled)
    last = float(raw["brent"].iloc[-1])
    point = last * np.exp(predicted_log)
    quantile = np.asarray(manifest["calibration"]["log_error_quantiles"], dtype=float)
    if not np.isfinite(point).all() or not np.isfinite(quantile).all():
        raise ValueError("Prediccion numericamente invalida")
    if len(point) != len(quantile) or np.any(quantile < 0):
        raise ValueError("Calibracion incompatible")
    older = read_json(output) if Path(output).is_file() else None
    if older:
        previous_date = pd.Timestamp(older["last_date"]).date()
        if latest_date < previous_date:
            raise ValueError("Fecha de datos mas antigua que el ultimo pronostico publicado")
        if (latest_date == previous_date
            and older.get("model", {}).get("weights_sha256") == manifest["weights_sha256"]
            and older.get("data_sha256") == file_hash(data)):
            return {"changed": False, "forecast": older}
    history = raw["brent"].tail(450)
    result = {
        "format_version": 1,
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "last_date": latest_date.isoformat(),
        "last_price": last,
        "data_sha256": file_hash(data),
        "model": {
            "architecture": manifest["architecture"],
            "weights_sha256": manifest["weights_sha256"],
            "trained_data_end": manifest["trained_data_end"],
            "has_independent_test": False,
        },
        "horizons": list(config["horizons"]),
        "prices": point.tolist(),
        "lower": (point * np.exp(-quantile)).tolist(),
        "upper": (point * np.exp(quantile)).tolist(),
        "nominal_coverage": manifest["calibration"]["nominal_coverage"],
        "historical_test_coverage": manifest["calibration"]["historical_test_coverage"],
        "historical_test_metrics": manifest["evaluation"]["all_models_historical_test"],
        "history": [{"date": date.date().isoformat(), "brent": float(price)}
                    for date, price in history.items()],
        "notes": manifest["caveats"],
    }
    save_json_atomic(output, result)
    return {"changed": True, "forecast": result}
