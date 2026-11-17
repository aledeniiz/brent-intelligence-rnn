"""Append-only forecast journal and first-observed, matured-label evaluation.

Legacy imports/reconstructions never count as real prospective forecasts.
No model fitting, price prediction or alteration of issued forecasts occurs here.
"""
from __future__ import annotations

from datetime import datetime, timezone
from hashlib import sha256
import json
import math
import os
from pathlib import Path
import tempfile

import numpy as np
import pandas as pd

from brent_intelligence.config import ROOT, DATA_PATH
from brent_intelligence.features import load_prices
from brent_intelligence.frozen import file_hash, read_json, save_json_atomic

JOURNAL_DIR = ROOT / "public" / "prospective"
EVALUATION_PATH = ROOT / "public" / "prospective_evaluation.json"
HORIZONS = (5, 63, 252)


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, allow_nan=False).encode("utf8")


def utc(value=None):
    stamp = pd.Timestamp(value if value is not None else datetime.now(timezone.utc))
    if stamp.tzinfo is None:
        raise ValueError("Se requiere fecha UTC con zona horaria")
    return stamp.tz_convert("UTC")


def immutable_json(path, value):
    """Publish a complete file atomically, without ever replacing an old record."""
    path = Path(path)
    value = {**value, "record_sha256": sha256(canonical(value)).hexdigest()}
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, suffix=".tmp", delete=False) as f:
        temporary = Path(f.name)
        f.write(canonical(value) + b"\n")
        f.flush()
        os.fsync(f.fileno())
    try:
        try:
            os.link(temporary, path)  # atomic create-if-absent on Windows/Linux
            return value
        except FileExistsError:
            return checked_record(path)
    finally:
        temporary.unlink(missing_ok=True)


def checked_record(path):
    value = read_json(path)
    contents = {k: v for k, v in value.items() if k != "record_sha256"}
    if value.get("record_sha256") != sha256(canonical(contents)).hexdigest():
        raise ValueError(f"Registro alterado: {Path(path).name}")
    return value


def validate_forecast(forecast):
    if forecast.get("format_version") != 1 or forecast.get("horizons") != list(HORIZONS):
        raise ValueError("Version u horizontes de pronostico invalidos")
    utc(forecast["generated_utc"])
    pd.Timestamp(forecast["last_date"])
    if not math.isfinite(forecast["last_price"]) or forecast["last_price"] <= 0:
        raise ValueError("Precio de origen invalido")
    if not all(len(forecast[key]) == 3 for key in ("prices", "lower", "upper")):
        raise ValueError("Predicciones desalineadas")
    for lo, point, hi in zip(forecast["lower"], forecast["prices"], forecast["upper"]):
        if not all(math.isfinite(x) for x in (lo, point, hi)) or not 0 < lo <= point <= hi:
            raise ValueError("Predicciones o bandas invalidas")
    weights = forecast["model"]["weights_sha256"]
    if len(weights) != 64 or any(c not in "0123456789abcdef" for c in weights):
        raise ValueError("Version de pesos invalida")


def archive_forecast(forecast, *, directory=JOURNAL_DIR, provenance="legacy_import",
                     bundle=None, now=None):
    validate_forecast(forecast)
    if provenance not in ("live_issue", "legacy_import", "historical_reconstruction"):
        raise ValueError("Procedencia desconocida")
    recorded = utc(now)
    if utc(forecast["generated_utc"]) > recorded:
        raise ValueError("Pronostico con emision posterior al registro")
    if provenance == "live_issue" and pd.Timestamp(forecast["last_date"]).date() > recorded.date():
        raise ValueError("Origen futuro en una emision en vivo")
    forecast_id = sha256(canonical(forecast)).hexdigest()
    # Same payload is never promoted from legacy to live or overwritten later.
    path = Path(directory) / "forecasts" / str(pd.Timestamp(forecast["last_date"]).date()) / (forecast_id + ".json")
    if path.exists():
        prior = checked_record(path)
        if prior["forecast_id"] != forecast_id or prior["forecast"] != forecast:
            raise ValueError("Colision o alteracion de prediccion archivada")
        return prior
    bundle_hashes = {}
    if bundle is not None:
        bundle = Path(bundle)
        bundle_hashes = {name: file_hash(bundle / name)
                         for name in ("production.pt", "preprocessing.json", "manifest.json")}
        if bundle_hashes["production.pt"] != forecast["model"]["weights_sha256"]:
            raise ValueError("Modelo archivado incompatible con el bundle")
    if provenance == "live_issue" and not bundle_hashes:
        raise ValueError("Una emision en vivo necesita la version completa del bundle")
    version = sha256(canonical(bundle_hashes)).hexdigest() if bundle_hashes else forecast["model"]["weights_sha256"]
    record = {"schema_version": 1, "forecast_id": forecast_id,
              "recorded_utc": recorded.isoformat(), "provenance": provenance,
              "model_version": version, "bundle_sha256": bundle_hashes,
              "eligible_for_prospective_evaluation": provenance == "live_issue",
              "forecast": forecast}
    return immutable_json(path, record)


def evaluate_prospective(*, data=DATA_PATH, directory=JOURNAL_DIR,
                         output=EVALUATION_PATH, as_of=None):
    """Freeze first available labels; retain revisions as warnings, not new scores.

    Session h is the h-th observed Brent quotation after the forecast origin,
    not a calendar-day offset. A target on/before recording date is excluded.
    """
    now = utc(as_of)
    directory = Path(directory)
    data_hash = file_hash(data)
    raw = load_prices(data)
    raw = raw.loc[raw.index.date <= now.date()]
    entries = [checked_record(p) for p in sorted((directory / "forecasts").glob("*/*.json"))]
    entries.sort(key=lambda r: (r["recorded_utc"], r["forecast_id"]))
    details, warnings, used = [], [], set()
    for entry in entries:
        f = entry["forecast"]
        validate_forecast(f)
        if entry["forecast_id"] != sha256(canonical(f)).hexdigest():
            raise ValueError("Identificador de pronostico incompatible")
        for j, h in enumerate(HORIZONS):
            detail = {"forecast_id": entry["forecast_id"], "model_version": entry["model_version"],
                      "origin_date": f["last_date"], "horizon": h}
            details.append(detail)
            if utc(entry["recorded_utc"]) > now:
                detail["status"] = "not_yet_recorded"
                continue
            if not entry["eligible_for_prospective_evaluation"] or entry["provenance"] != "live_issue":
                detail["status"] = "excluded_nonprospective"
                continue
            if pd.Timestamp(f["model"]["trained_data_end"]) > pd.Timestamp(f["last_date"]):
                detail["status"] = "excluded_training_after_origin"
                continue
            # Earliest live issue per origin and model. Revisions remain archived.
            key = (entry["model_version"], f["last_date"], h)
            if key in used:
                detail["status"] = "excluded_later_revision"
                continue
            used.add(key)
            origin = pd.Timestamp(f["last_date"])
            issue_day = max(utc(entry["recorded_utc"]).date(), utc(f["generated_utc"]).date())
            label_path = directory / "labels" / (entry["forecast_id"] + f"_{h}.json")
            following = raw.loc[raw.index > origin, "brent"]
            available = origin in raw.index and len(following) >= h
            if label_path.exists():
                label = checked_record(label_path)
            else:
                if not available:
                    detail["status"] = "pending_maturity" if origin in raw.index else "pending_origin_not_available"
                    continue
                target_date, actual = following.index[h-1], float(following.iloc[h-1])
                if target_date.date() <= issue_day:
                    detail["status"] = "excluded_target_known_before_recording"
                    continue
                label = {"schema_version": 1, "forecast_id": entry["forecast_id"], "horizon": h,
                         "target_date": str(target_date.date()), "actual": actual,
                         "observed_utc": now.isoformat(), "data_sha256": data_hash,
                         "session_dates": [str(d.date()) for d in following.index[:h]]}
                label = immutable_json(label_path, label)
            if label["forecast_id"] != entry["forecast_id"] or label["horizon"] != h:
                raise ValueError("Etiqueta incompatible con el pronostico")
            if utc(label["observed_utc"]) > now:
                detail["status"] = "pending_label_not_observed_as_of"
                continue
            if not issue_day < pd.Timestamp(label["target_date"]).date() <= utc(label["observed_utc"]).date():
                raise ValueError("Etiqueta anterior a la emision o posterior a su observacion")
            if (not available or label["target_date"] != str(following.index[h-1].date())
                    or label["actual"] != float(following.iloc[h-1])):
                warnings.append({"forecast_id": entry["forecast_id"], "horizon": h,
                                 "warning": "Revised data/calendar; first-observed label retained"})
            error = f["prices"][j] - label["actual"]
            naive_error = f["last_price"] - label["actual"]
            detail.update(status="matured", target_date=label["target_date"], actual=label["actual"],
                          prediction=f["prices"][j], naive=f["last_price"], error=error,
                          naive_error=naive_error, label_path=str(label_path.relative_to(directory)))
    groups = []
    for version in sorted({d["model_version"] for d in details}):
        for h in HORIZONS:
            points = [d for d in details if d["model_version"] == version and d["horizon"] == h and d["status"] == "matured"]
            n = len(points)
            rmse = float(np.sqrt(np.mean([d["error"] ** 2 for d in points]))) if n else None
            naive_rmse = float(np.sqrt(np.mean([d["naive_error"] ** 2 for d in points]))) if n else None
            groups.append({"model_version": version, "horizon": h, "n": n,
                           "mae": float(np.mean([abs(d["error"]) for d in points])) if n else None,
                           "rmse": rmse,
                           "naive_mae": float(np.mean([abs(d["naive_error"]) for d in points])) if n else None,
                           "naive_rmse": naive_rmse,
                           "improvement_pct": 100 * (1-rmse/naive_rmse) if naive_rmse else None})
    result = {"schema_version": 1, "as_of_utc": now.isoformat(), "data_sha256": data_hash,
              "forecast_records": len(entries), "evaluated_points": sum(d["status"] == "matured" for d in details),
              "groups": groups, "details": details, "warnings": warnings,
              "notes": ["Only archived live issues with targets strictly after recording day count.",
                        "Legacy imports and reconstructed backtests are not prospective forecasts.",
                        "First live revision per model/origin; first-observed labels retained.",
                        "Overlapping horizons are dependent; no significance or predictive gain is implied."]}
    if output is not None:
        save_json_atomic(output, result)
    return result
