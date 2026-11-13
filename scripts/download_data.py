"""Descarga Brent de FRED/EIA y variables adicionales; no rellena Brent."""
from __future__ import annotations
import logging
from datetime import date
from pathlib import Path
import pandas as pd

from brent_intelligence.config import DATA_PATH

SERIES = {
    "DCOILBRENTEU": "brent", "DCOILWTICO": "wti",
    "DHHNGSP": "gas_natural", "ECBDFR": "tipo_bce",
    "DEXUSEU": "usd_por_eur", "DEXCHUS": "cny_por_usd",
    "VIXCLS": "vix",
}
START = "2000-01-01"

def from_fred(series_id: str, until: str) -> pd.Series:
    url = ("https://fred.stlouisfed.org/graph/fredgraph.csv"
           f"?id={series_id}&cosd={START}&coed={until}")
    df = pd.read_csv(url, index_col=0, parse_dates=True, na_values=".")
    if df.empty:
        raise ValueError(f"Serie vacia: {series_id}")
    return pd.to_numeric(df.iloc[:, 0], errors="coerce").rename(SERIES[series_id])

def download(path: Path = DATA_PATH, until: str | None = None) -> pd.DataFrame:
    until = until or date.today().isoformat()
    frames = [from_fred("DCOILBRENTEU", until)]
    for k in SERIES:
        if k == "DCOILBRENTEU":
            continue
        try:
            frames.append(from_fred(k, until))
        except Exception as exc:
            logging.warning("Serie auxiliar %s fallida: %s", k, exc)
    df = pd.concat(frames, axis=1).sort_index()
    df = df.loc[~df.index.duplicated(keep="last")]
    df = df.loc[df["brent"].notna() & (df["brent"] > 0)]
    df = df.loc[df.index.dayofweek < 5]
    df.index.name = "Date"
    if len(df) < 1000:
        raise ValueError("Muy pocas observaciones; no se sobreescribe el dataset")
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path)
    print(f"Guardado {path}: {len(df)} observaciones, {len(df.columns)} variables")
    return df

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    download()
