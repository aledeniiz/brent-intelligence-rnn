"""Configuracion compartida: no modificar fechas tras ver el test."""
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "data" / "petroleo_dataset.csv"
ARTIFACTS_DIR = ROOT / "artifacts"
REPORTS_DIR = ROOT / "reports"
HORIZONS = (5, 63, 252)

@dataclass(frozen=True)
class ExperimentConfig:
    window: int = 60
    horizons: tuple[int, ...] = HORIZONS
    hidden_size: int = 32
    layers: int = 1
    epochs: int = 20
    patience: int = 5
    batch_size: int = 128
    learning_rate: float = 0.001
    seed: int = 42
    min_feature_coverage: float = 0.60
    interval_coverage: float = 0.80
