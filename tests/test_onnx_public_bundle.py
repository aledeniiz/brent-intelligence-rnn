from pathlib import Path
import json,hashlib
ROOT=Path(__file__).resolve().parents[1]
def test_public_bundle_has_no_financial_inputs():
 p=ROOT/'web/pages'
 assert not (p/'assets/data').exists()
 assert not (p/'assets/demo-state.json').exists()
 assert not (p/'assets/private-input.js').exists()
 assert 'type="file"' not in (p/'index.html').read_text(encoding='utf-8')
 assert (p/'assets/synthetic/scenario.json').exists()
 assert (p/'assets/ort/LICENSE').exists()
 assert (p/'assets/ort/ThirdPartyNotices.txt').exists()
 m=json.loads((p/'assets/model/manifest.json').read_text(encoding='utf-8'))
 assert m['parameters']==6499 and len(m['feature_names'])==16
 assert hashlib.sha256((p/'assets/model/lstm.onnx').read_bytes()).hexdigest()==m['onnx_sha256']
 assert 'mode: "onnx"' in (p/'config.js').read_text()
