"""Informe minimo reproducible de calidad de datos."""
from __future__ import annotations
import json
from brent_intelligence.config import DATA_PATH, REPORTS_DIR
from brent_intelligence.features import load_prices

def audit():
    df = load_prices(DATA_PATH)
    result = {
        "rows": len(df),
        "start": str(df.index.min().date()),
        "end": str(df.index.max().date()),
        "columns": list(df.columns),
        "missing_by_column": df.isna().sum().astype(int).to_dict(),
        "brent_min": float(df["brent"].min()),
        "brent_max": float(df["brent"].max()),
        "warning": "Revision historica y disponibilidad de variables auxiliares no certificadas point-in-time",
    }
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    (REPORTS_DIR / "data_audit.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return result

if __name__ == "__main__":
    audit()
