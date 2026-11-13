import numpy as np
import pytest
import torch
from brent_intelligence.modeling import RecurrentForecaster
from brent_intelligence.evaluation import prices_from_returns, score

@pytest.mark.parametrize("kind", ["rnn", "lstm", "gru"])
def test_output_shape(kind):
    net = RecurrentForecaster(kind, n_features=6, hidden_size=8, horizons=3)
    assert net(torch.zeros((4, 60, 6))).shape == (4, 3)

def test_baseline_has_zero_error_for_flat_price():
    anchors = np.array([80., 100.])
    actual = np.tile(anchors[:, None], (1, 3))
    pred = prices_from_returns(anchors, np.zeros((2, 3)))
    metrics = score(actual, pred, anchors, (5, 63, 252))
    assert metrics["5"]["rmse"] == 0.0
    assert metrics["252"]["relative_rmse"] is None
