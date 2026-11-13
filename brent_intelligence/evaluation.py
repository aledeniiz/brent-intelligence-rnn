"""Evaluacion en dolares y comparacion con precio constante."""
from __future__ import annotations
import numpy as np

def prices_from_returns(anchor: np.ndarray, logs: np.ndarray) -> np.ndarray:
    return anchor[:, None] * np.exp(logs)

def score(actual: np.ndarray, predicted: np.ndarray, anchor: np.ndarray,
          horizons: tuple[int, ...]) -> dict:
    if actual.shape != predicted.shape or actual.shape[1] != len(horizons):
        raise ValueError("Predicciones y objetivos desalineados")
    out = {}
    for j, h in enumerate(horizons):
        residual = predicted[:, j] - actual[:, j]
        naive = anchor - actual[:, j]
        rmse = float(np.sqrt(np.mean(residual ** 2)))
        naive_rmse = float(np.sqrt(np.mean(naive ** 2)))
        out[str(h)] = {
            "mae": float(np.mean(np.abs(residual))),
            "mse": float(np.mean(residual ** 2)),
            "rmse": rmse,
            "naive_mae": float(np.mean(np.abs(naive))),
            "naive_rmse": naive_rmse,
            "relative_rmse": float(rmse / naive_rmse) if naive_rmse > 0 else None,
            "n": len(anchor),
        }
    return out
