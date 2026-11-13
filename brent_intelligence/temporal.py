"""Construccion de etiquetas futuras y folds sin fuga temporal."""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import pandas as pd

FOLD_BOUNDARIES = (
    ("2011-12-30", "2013-12-31"),
    ("2013-12-31", "2015-12-31"),
    ("2015-12-31", "2017-12-31"),
    ("2017-12-31", "2019-12-31"),
    ("2019-12-31", "2021-12-31"),
)
CAL_END = "2023-12-29"

@dataclass
class Examples:
    features: pd.DataFrame
    prices: np.ndarray
    anchors: np.ndarray
    targets: np.ndarray
    returns: np.ndarray
    window: int
    horizons: tuple[int, ...]

    @property
    def dates(self):
        return self.features.index

    def x(self, scaled: np.ndarray, ids: np.ndarray) -> np.ndarray:
        return np.stack([scaled[a-self.window+1:a+1] for a in self.anchors[ids]]).astype(np.float32)

    def y(self, ids: np.ndarray) -> np.ndarray:
        return self.returns[ids].astype(np.float32)

    def actual(self, ids: np.ndarray) -> np.ndarray:
        return self.prices[self.targets[ids]]

    def anchors_price(self, ids: np.ndarray) -> np.ndarray:
        return self.prices[self.anchors[ids]]


def make_examples(features: pd.DataFrame, brent: pd.Series, *,
                  window: int = 60, horizons: tuple[int, ...] = (5, 63, 252)) -> Examples:
    if not features.index.equals(brent.index) or not features.index.is_monotonic_increasing or features.index.has_duplicates:
        raise ValueError("Caracteristicas y objetivo deben tener las mismas fechas ordenadas")
    if window < 2 or not horizons or len(set(horizons)) != len(horizons) or any(h < 1 for h in horizons):
        raise ValueError("Ventana u horizontes invalidos")
    prices = brent.to_numpy(dtype=np.float64)
    if not np.isfinite(prices).all() or (prices <= 0).any():
        raise ValueError("Brent debe ser estrictamente positivo")
    if len(prices) < window + max(horizons):
        raise ValueError("Historial insuficiente")
    anchors = np.arange(window-1, len(prices)-max(horizons), dtype=np.int64)
    targets = anchors[:, None] + np.array(horizons)[None, :]
    returns = np.log(prices[targets] / prices[anchors, None])
    return Examples(features, prices, anchors, targets, returns, window, horizons)


def make_plan(ex: Examples, *, max_folds: int | None = None) -> dict:
    """Train: solo etiquetas con futuro maximo madurado antes del cutoff."""
    ad = ex.dates.take(ex.anchors)
    mature = ex.dates.take(ex.targets.max(axis=1))
    folds = []
    for i, (train_end, val_end) in enumerate(FOLD_BOUNDARIES, 1):
        train = np.flatnonzero(mature <= pd.Timestamp(train_end))
        val = np.flatnonzero((ad > pd.Timestamp(train_end)) & (mature <= pd.Timestamp(val_end)))
        if not len(train) or len(val) < 20:
            raise ValueError(f"Fold {i} sin datos suficientes")
        folds.append({"name": f"fold_{i:02d}", "train_end": train_end,
                      "val_end": val_end, "train": train, "validation": val})
        if max_folds and len(folds) >= max_folds:
            break
    train_final = np.flatnonzero(mature <= pd.Timestamp("2021-12-31"))
    cal = np.flatnonzero((ad > pd.Timestamp("2021-12-31")) & (mature <= pd.Timestamp(CAL_END)))
    test = np.flatnonzero(ad > pd.Timestamp(CAL_END))
    if min(len(train_final), len(cal), len(test)) < 20:
        raise ValueError("Train/calibracion/test insuficientes")
    return {"folds": folds, "train_final": train_final,
            "calibration": cal, "test": test}
