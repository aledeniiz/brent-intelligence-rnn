from pathlib import Path
import hashlib,json
import numpy as np
import pytest
from brent_intelligence.features import make_features
from brent_intelligence.experiment import Preprocessor
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]
def test_synthetic_is_reproducible_and_not_financial():
    assets=ROOT/'web/pages/assets'
    policy=json.loads((assets/'data-policy.json').read_text(encoding='utf-8'))
    data=(assets/'synthetic/scenario.json').read_bytes();p=json.loads(data)
    m=json.loads((assets/'model/manifest.json').read_text(encoding='utf-8'))
    assert hashlib.sha256(data).hexdigest()==policy['synthetic_sha256']
    assert not policy['complete_live_inputs_approved'] and policy['live_input_url'] is None
    assert p['kind']=='synthetic' and p['license']=='CC0-1.0' and not p['sources'] and not p['events']
    raw=pd.DataFrame(p['raw']).set_index('date');raw.index=pd.to_datetime(raw.index)
    x=Preprocessor(**m['scaler']).x(make_features(raw))[-60:]
    np.testing.assert_array_equal(x.flatten(),np.array(p['input']['values'],dtype=np.float32))
    assert hashlib.sha256(x.astype('<f4').tobytes()).hexdigest()==p['input']['sha256']
    assert hashlib.sha256((assets/'model/lstm.onnx').read_bytes()).hexdigest()==m['onnx_sha256']

def test_pytorch_onnx_parity_without_export():
    import torch
    ort=pytest.importorskip('onnxruntime')
    from brent_intelligence.modeling import RecurrentForecaster
    repo=ROOT.parents[1]
    weights=repo/'artifacts/runs/20261116T201630599664Z_5f12b8df/lstm_evaluation.pt'
    if not weights.exists():pytest.skip('Original checkpoint is intentionally not in public source checkout')
    a=ROOT/'web/pages/assets';m=json.loads((a/'model/manifest.json').read_text(encoding='utf-8'))
    before=hashlib.sha256(weights.read_bytes()).hexdigest();assert before==m['weights_sha256']
    torch.set_num_threads(2);model=RecurrentForecaster('lstm',16,32,1,3)
    model.load_state_dict(torch.load(weights,map_location='cpu',weights_only=True));model.eval()
    engine=ort.InferenceSession(str(a/'model/lstm.onnx'),providers=['CPUExecutionProvider'])
    p=json.loads((a/'synthetic/scenario.json').read_text(encoding='utf-8'))
    x=np.array(p['input']['values'],dtype=np.float32).reshape(1,60,16)
    with torch.no_grad():expected=model(torch.from_numpy(x)).numpy()[0]
    actual=engine.run(None,{'features':x})[0][0]
    scale=np.array(m['scaler']['y_std']);mean=np.array(m['scaler']['y_mean']);q=np.array(m['calibration']['log_error_quantiles']);anchor=p['origin']['price']
    prices=anchor*np.exp(expected*scale+mean);observed=anchor*np.exp(actual*scale+mean)
    for mult in [np.ones(3),np.exp(q),np.exp(-q)]:np.testing.assert_allclose(prices*mult,observed*mult,atol=.001,rtol=0)
    assert hashlib.sha256(weights.read_bytes()).hexdigest()==before
