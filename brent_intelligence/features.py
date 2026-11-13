"""Transformaciones causales sobre Brent y variables economicas."""
from __future__ import annotations
import numpy as np
import pandas as pd

def load_prices(path) -> pd.DataFrame:
    raw = pd.read_csv(path, index_col="Date", parse_dates=["Date"])
    if "brent" not in raw.columns:
        raise ValueError("Falta la columna brent")
    raw = raw.loc[~raw.index.duplicated(keep="last")].sort_index()
    raw = raw.apply(pd.to_numeric, errors="coerce")
    # No inventar el objetivo con ffill ni convertir shocks negativos del WTI.
    raw = raw.loc[raw["brent"].notna() & (raw["brent"] > 0)]
    if raw.empty:
        raise ValueError("Brent no contiene observaciones validas")
    return raw

def make_features(data: pd.DataFrame) -> pd.DataFrame:
    if not data.index.is_monotonic_increasing or data.index.has_duplicates:
        raise ValueError("Indice no ordenado o con fechas duplicadas")
    brent = pd.to_numeric(data["brent"], errors="coerce")
    if brent.isna().any() or (brent <= 0).any():
        raise ValueError("Brent debe ser positivo y observado")
    lp = np.log(brent)
    f = pd.DataFrame(index=data.index)
    f["brent_log_level"] = lp
    f["brent_ret_1"] = lp.diff()
    f["brent_ret_5"] = lp.diff(5)
    f["brent_ret_21"] = lp.diff(21)
    f["brent_vol_21"] = lp.diff().rolling(21).std()
    f["brent_vs_ma_21"] = lp - lp.rolling(21).mean()
    # El desfase de un dia es conservador, NO equivale a datos point-in-time.
    lag = data.drop(columns=["brent"]).shift(1).ffill(limit=5)
    for name in ("wti", "gas_natural", "tipo_bce", "usd_por_eur",
                 "cny_por_usd", "vix", "cobre", "oro", "indice_dolar"):
        if name not in lag:
            continue
        v = pd.to_numeric(lag[name], errors="coerce")
        if name == "wti":
            f["wti_spread_relative"] = (v - brent) / brent
            f["wti_change_scaled"] = v.diff(5) / brent.shift(5)
        elif name == "tipo_bce":
            f[name] = v
        elif name == "vix":
            f["vix_log_level"] = np.log1p(v.where(v >= 0))
        else:
            f[f"{name}_log_change_5"] = np.log(v.where(v > 0)).diff(5)
    if "orb" in lag:
        opec = pd.to_numeric(lag["orb"], errors="coerce")
        f["orb_log_change_5"] = np.log(opec.where(opec > 0)).diff(5)
    return f.replace([np.inf, -np.inf], np.nan)
