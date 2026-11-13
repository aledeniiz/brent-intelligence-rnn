"""Modelos comparables con cabezas multihorizonte."""
from __future__ import annotations
import random
import numpy as np
import torch
from torch import nn

class RecurrentForecaster(nn.Module):
    def __init__(self, kind: str, n_features: int, hidden_size: int = 32,
                 layers: int = 1, horizons: int = 3):
        super().__init__()
        mapping = {"rnn": nn.RNN, "lstm": nn.LSTM, "gru": nn.GRU}
        if kind not in mapping:
            raise ValueError(f"Modelo desconocido: {kind}")
        self.recurrent = mapping[kind](n_features, hidden_size, num_layers=layers,
                                       batch_first=True)
        self.head = nn.Linear(hidden_size, horizons)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        states, _ = self.recurrent(x)
        return self.head(states[:, -1, :])

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
