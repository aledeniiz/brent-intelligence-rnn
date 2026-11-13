"""Walk-forward con test reservado y modelo de produccion independiente."""
from __future__ import annotations
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from hashlib import sha256
import json
import os
from pathlib import Path
import statistics
import time

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from brent_intelligence.config import ARTIFACTS_DIR, DATA_PATH, ExperimentConfig
from brent_intelligence.evaluation import prices_from_returns, score
from brent_intelligence.features import load_prices, make_features
from brent_intelligence.modeling import RecurrentForecaster, set_seed
from brent_intelligence.temporal import make_examples, make_plan

KINDS = ("rnn", "lstm", "gru")


@dataclass
class Preprocessor:
    columns: list[str]
    medians: list[float]
    x_mean: list[float]
    x_std: list[float]
    y_mean: list[float]
    y_std: list[float]

    @classmethod
    def fit(cls, examples, train_ids: np.ndarray, cutoff: str,
            min_coverage: float = 0.6) -> "Preprocessor":
        if len(train_ids) == 0:
            raise ValueError("El entrenamiento esta vacio")
        known_targets = examples.dates.take(examples.targets[train_ids].max(axis=1))
        if (known_targets > pd.Timestamp(cutoff)).any():
            raise ValueError("Fuga temporal: etiquetas futuras en entrenamiento")
        known = examples.features.loc[:cutoff]
        cols = known.columns[known.notna().mean() >= min_coverage].tolist()
        if not cols:
            raise ValueError("No hay variables con cobertura suficiente")
        med = known[cols].median().fillna(0.0)
        matrix = known[cols].fillna(med).to_numpy(dtype=float)
        xmean, xstd = matrix.mean(axis=0), matrix.std(axis=0)
        y = examples.y(train_ids).astype(float)
        ymean, ystd = y.mean(axis=0), y.std(axis=0)
        return cls(cols, med.to_numpy().tolist(), xmean.tolist(),
                   np.where(xstd < 1e-8, 1., xstd).tolist(), ymean.tolist(),
                   np.where(ystd < 1e-8, 1., ystd).tolist())

    def x(self, features: pd.DataFrame) -> np.ndarray:
        matrix = features.reindex(columns=self.columns).to_numpy(dtype=float)
        meds = np.array(self.medians)
        matrix = np.where(np.isfinite(matrix), matrix, meds[None, :])
        return ((matrix - np.array(self.x_mean)) / np.array(self.x_std)).astype(np.float32)

    def y(self, returns: np.ndarray) -> np.ndarray:
        return ((returns - np.array(self.y_mean)) / np.array(self.y_std)).astype(np.float32)

    def inverse_y(self, scaled: np.ndarray) -> np.ndarray:
        return scaled * np.array(self.y_std) + np.array(self.y_mean)


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    os.replace(tmp, path)


def _train(kind: str, xtrain: np.ndarray, ytrain: np.ndarray,
           cfg: ExperimentConfig, n_features: int, *,
           xv: np.ndarray | None = None, yv: np.ndarray | None = None,
           epochs: int | None = None, device: str = "cpu"):
    set_seed(cfg.seed)
    model = RecurrentForecaster(kind, n_features, cfg.hidden_size, cfg.layers,
                                len(cfg.horizons)).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.learning_rate)
    criterion = nn.SmoothL1Loss()
    loader = DataLoader(TensorDataset(torch.from_numpy(xtrain), torch.from_numpy(ytrain)),
                        batch_size=cfg.batch_size, shuffle=True,
                        generator=torch.Generator().manual_seed(cfg.seed))
    best_epoch, best_loss, wait = 1, float("inf"), 0
    best_state = None
    history = []
    t0 = time.perf_counter()
    for epoch in range(1, (epochs or cfg.epochs) + 1):
        model.train()
        losses = []
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(x), y)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            losses.append((float(loss.item()), len(x)))
        item = {"epoch": epoch, "train_loss": sum(a*n for a,n in losses)/sum(n for _,n in losses)}
        if xv is not None and yv is not None:
            model.eval()
            with torch.no_grad():
                val = criterion(model(torch.from_numpy(xv).to(device)),
                                torch.from_numpy(yv).to(device))
                item["val_loss"] = float(val.item())
            if item["val_loss"] < best_loss - 1e-5:
                best_loss, best_epoch, wait = item["val_loss"], epoch, 0
                best_state = {k: v.detach().clone() for k,v in model.state_dict().items()}
            else:
                wait += 1
        history.append(item)
        if xv is not None and wait >= cfg.patience:
            break
    if best_state is not None:
        model.load_state_dict(best_state)
    return model, {"history": history, "best_epoch": best_epoch if xv is not None else len(history),
                   "epochs": len(history), "seconds": round(time.perf_counter()-t0, 3)}


def _predict(model: nn.Module, x: np.ndarray, device: str = "cpu") -> np.ndarray:
    model.eval()
    chunks = []
    with torch.no_grad():
        for start in range(0, len(x), 512):
            chunks.append(model(torch.from_numpy(x[start:start+512]).to(device)).cpu().numpy())
    return np.concatenate(chunks)


def _fold_arrays(ex, train_ids, evaluation_ids, cutoff, cfg):
    prep = Preprocessor.fit(ex, train_ids, cutoff, cfg.min_feature_coverage)
    sx = prep.x(ex.features)
    train_x = ex.x(sx, train_ids)
    eval_x = ex.x(sx, evaluation_ids)
    return prep, train_x, prep.y(ex.y(train_ids)), eval_x, prep.y(ex.y(evaluation_ids))


def run_experiment(*, data_path: Path = DATA_PATH,
                   artifact_dir: Path = ARTIFACTS_DIR,
                   cfg: ExperimentConfig | None = None,
                   smoke: bool = False, device: str = "cpu", dry_run: bool = False) -> dict:
    cfg = cfg or ExperimentConfig()
    if cfg.epochs < 1 or cfg.patience < 1 or cfg.batch_size < 1:
        raise ValueError("Configuracion de entrenamiento invalida")
    data_path = Path(data_path)
    if not data_path.is_file():
        raise FileNotFoundError(f"No existe {data_path}. Ejecuta python -m scripts.download_data")
    if device == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA no esta disponible")
    torch.set_num_threads(min(4, os.cpu_count() or 1))
    raw = load_prices(data_path)
    feat = make_features(raw)
    ex = make_examples(feat, raw["brent"], window=cfg.window, horizons=cfg.horizons)
    plan = make_plan(ex, max_folds=1 if smoke else None)
    fingerprint = sha256(data_path.read_bytes()).hexdigest()
    report = {"format_version": "1.0", "mode": "synthetic-smoke-or-debug" if smoke else "historical-evaluation",
              "created_utc": datetime.now(timezone.utc).isoformat(),
              "data_sha256": fingerprint, "last_data_date": str(ex.dates[-1].date()),
              "config": asdict(cfg), "folds": [], "models": {},
              "warnings": ["Variables externas no verificadas point-in-time; posible revision historica.",
                           "Las predicciones de 252 sesiones solo se evaluan cuando maduran.",
                           "Los intervalos calibrados son empiricos, sin garantia de cobertura."]}
    for fold in plan["folds"]:
        report["folds"].append({"name": fold["name"], "train_end": fold["train_end"],
                                "validation_end": fold["val_end"],
                                "n_train": len(fold["train"]), "n_validation": len(fold["validation"])})
    if dry_run:
        return report
    root = Path(artifact_dir) / ("smoke" if smoke else "runs")
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ") + "_" + fingerprint[:8]
    out = root / run_id
    out.mkdir(parents=True, exist_ok=False)
    model_scores = {}
    for kind in KINDS:
        frows = []
        for fold in plan["folds"]:
            train_id, val_id = fold["train"], fold["validation"]
            prep, xtrain, ytrain, xval, yval = _fold_arrays(ex, train_id, val_id,
                                                            fold["train_end"], cfg)
            model, history = _train(kind, xtrain, ytrain, cfg, len(prep.columns),
                                    xv=xval, yv=yval, device=device)
            pred = prices_from_returns(ex.anchors_price(val_id),
                                       prep.inverse_y(_predict(model, xval, device)))
            metrics = score(ex.actual(val_id), pred, ex.anchors_price(val_id), cfg.horizons)
            ratios = [metrics[str(h)]["relative_rmse"] for h in cfg.horizons]
            if any(x is None for x in ratios):
                raise ValueError("Baseline ingenuo con RMSE nulo; no se puede ordenar")
            frows.append({"fold": fold["name"], "metrics": metrics,
                          "score": float(statistics.mean(ratios)), **history})
        val_score = float(statistics.mean(r["score"] for r in frows))
        ep = max(1, round(statistics.median(r["best_epoch"] for r in frows)))
        model_scores[kind] = {"validation_score": val_score,
                              "refit_epochs": ep, "folds": frows}
    winner = min(KINDS, key=lambda k: model_scores[k]["validation_score"])
    report["selection"] = {"architecture": winner,
                           "criterion": "mean fold and horizon RMSE/naive_RMSE; lower is better"}
    report["models"] = model_scores
    # Calibracion y test se consultan despues de elegir arquitectura.
    cal, test = plan["calibration"], plan["test"]
    calibration_prediction = None
    for kind in KINDS:
        train_ids = plan["train_final"]
        prep, xtrain, ytrain, xcal, _ = _fold_arrays(ex, train_ids, cal, "2021-12-31", cfg)
        xtest = ex.x(prep.x(ex.features), test)
        model, train_log = _train(kind, xtrain, ytrain, cfg, len(prep.columns),
                                  epochs=model_scores[kind]["refit_epochs"], device=device)
        model.cpu()
        torch.save(model.state_dict(), out / f"{kind}_evaluation.pt")
        y_cal = prep.inverse_y(_predict(model, xcal))
        y_test = prep.inverse_y(_predict(model, xtest))
        pcal = prices_from_returns(ex.anchors_price(cal), y_cal)
        ptest = prices_from_returns(ex.anchors_price(test), y_test)
        model_scores[kind]["calibration"] = score(ex.actual(cal), pcal,
                                                   ex.anchors_price(cal), cfg.horizons)
        model_scores[kind]["test"] = score(ex.actual(test), ptest,
                                            ex.anchors_price(test), cfg.horizons)
        model_scores[kind]["parameter_count"] = sum(p.numel() for p in model.parameters())
        model_scores[kind]["refit"] = train_log
        if kind == winner:
            calibration_prediction = pcal
            test_prediction = ptest
            _write_json(out / "evaluation_preprocessing.json", asdict(prep))
    residual_log = np.abs(np.log(ex.actual(cal) / calibration_prediction))
    quantiles = np.quantile(residual_log, cfg.interval_coverage, axis=0)
    lower = test_prediction * np.exp(-quantiles)
    upper = test_prediction * np.exp(quantiles)
    empirical = np.mean((ex.actual(test) >= lower) & (ex.actual(test) <= upper), axis=0)
    report["uncertainty"] = {"nominal": cfg.interval_coverage,
                             "log_error_quantiles": quantiles.tolist(),
                             "test_coverage": {str(h): float(empirical[i]) for i,h in enumerate(cfg.horizons)},
                             "n_calibration": len(cal), "n_test": len(test)}
    report["test_period"] = {"start": "2024-01-01", "n_anchors": len(test),
                             "last_matured_origin": str(ex.dates[ex.anchors[test[-1]]].date())}
    # Produccion: se ajusta usando todos los objetivos madurados hasta hoy.
    # No se asigna una puntuacion de test a este nuevo modelo.
    maturity = ex.targets.max(axis=1)
    prod_train = np.flatnonzero(maturity <= len(ex.dates)-1)
    prod = Preprocessor.fit(ex, prod_train, str(ex.dates[-1].date()), cfg.min_feature_coverage)
    prod_x = ex.x(prod.x(ex.features), prod_train)
    prod_y = prod.y(ex.y(prod_train))
    prod_model, prod_log = _train(winner, prod_x, prod_y, cfg, len(prod.columns),
                                  epochs=model_scores[winner]["refit_epochs"], device=device)
    prod_model.cpu()
    torch.save(prod_model.state_dict(), out / "production.pt")
    _write_json(out / "production_preprocessing.json", asdict(prod))
    report["production"] = {"last_known_price_date": str(ex.dates[-1].date()),
                            "last_matured_training_origin": str(ex.dates[ex.anchors[prod_train[-1]]].date()),
                            "n_examples": len(prod_train), "train": prod_log,
                            "has_independent_test": False}
    _write_json(out / "summary.json", report)
    if not smoke:
        _write_json(Path(artifact_dir) / "active.json", {"run": str(out.relative_to(artifact_dir))})
    return report


def predict_latest(*, data_path: Path = DATA_PATH,
                   artifact_dir: Path = ARTIFACTS_DIR) -> dict:
    root = Path(artifact_dir)
    pointer = root / "active.json"
    if not pointer.exists():
        raise FileNotFoundError("Falta entrenamiento final. Ejecuta python -m scripts.train_models")
    selected = json.loads(pointer.read_text(encoding="utf-8"))["run"]
    if Path(selected).is_absolute() or ".." in Path(selected).parts:
        raise ValueError("Ruta de modelo invalida")
    folder = root / selected
    summary = json.loads((folder / "summary.json").read_text(encoding="utf-8"))
    prep = Preprocessor(**json.loads((folder / "production_preprocessing.json").read_text(encoding="utf-8")))
    prices = load_prices(data_path)
    features = make_features(prices)
    cfg = summary["config"]
    window = int(cfg["window"])
    if len(features) < window:
        raise ValueError("No hay suficientes sesiones")
    arch = summary["selection"]["architecture"]
    model = RecurrentForecaster(arch, len(prep.columns), int(cfg["hidden_size"]),
                                int(cfg["layers"]), len(cfg["horizons"]))
    model.load_state_dict(torch.load(folder / "production.pt", map_location="cpu", weights_only=True))
    y = prep.inverse_y(_predict(model, prep.x(features)[-window:][None, :, :]))[0]
    last = float(prices["brent"].iloc[-1])
    central = last * np.exp(y)
    q = np.asarray(summary["uncertainty"]["log_error_quantiles"])
    return {"model": arch, "last_date": str(prices.index[-1].date()),
            "last_price": last, "model_training_end": summary["production"]["last_known_price_date"],
            "horizons": list(cfg["horizons"]), "prices": central.tolist(),
            "lower": (central * np.exp(-q)).tolist(),
            "upper": (central * np.exp(q)).tolist(), "summary": summary,
            "stale_data": prices.index[-1].date() < (pd.Timestamp.now() - pd.Timedelta(days=7)).date(),
            "data_changed_since_training": sha256(Path(data_path).read_bytes()).hexdigest() != summary["data_sha256"]}
