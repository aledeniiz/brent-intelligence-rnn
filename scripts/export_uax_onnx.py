"""Export the exact calibrated UAX LSTM; never train or overwrite source weights."""
from pathlib import Path
import json,hashlib
import numpy as np
import torch
import onnx
import onnxruntime as ort
from brent_intelligence.uax_product import ROOT,model_details,strict_prices,validate_inputs,OUT,read
from brent_intelligence.modeling import RecurrentForecaster


def main():
    summary,prep,cfg,weights,info=model_details();before=hashlib.sha256(weights.read_bytes()).hexdigest()
    directory=ROOT/'web/pages/assets/model';directory.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(4);model=RecurrentForecaster('lstm',16,32,1,3)
    model.load_state_dict(torch.load(weights,map_location='cpu',weights_only=True));model.eval()
    destination=directory/'lstm.onnx'
    torch.onnx.export(model,torch.zeros(1,60,16),str(destination),input_names=['features'],output_names=['normalized_returns'],opset_version=17,dynamo=False)
    onnx.checker.check_model(onnx.load(str(destination)))
    raw=strict_prices(OUT/'versions'/read(OUT/'active.json')['version']/'dataset.csv')
    feat,_,_=validate_inputs(raw,prep.columns);matrix=prep.x(feat)
    session=ort.InferenceSession(str(destination),providers=['CPUExecutionProvider'])
    cases=[];q=np.array(summary['uncertainty']['log_error_quantiles'])
    for end in [len(raw)-1,len(raw)-20,len(raw)-65,len(raw)-253]:
        x=matrix[end-59:end+1][None].astype(np.float32)
        with torch.no_grad():expected=model(torch.from_numpy(x)).numpy()
        actual=session.run(None,{'features':x})[0]
        a=float(raw.brent.iloc[end]);p=a*np.exp(prep.inverse_y(expected)[0]);p2=a*np.exp(prep.inverse_y(actual)[0])
        err=float(np.max(np.abs(p-p2)));assert err<.001,err
        cases.append(dict(origin=str(raw.index[end].date()),input=x.flatten().tolist(),anchor=a,
            normalized_pytorch=expected.flatten().tolist(),prices_pytorch=p.tolist(),lower_pytorch=(p*np.exp(-q)).tolist(),upper_pytorch=(p*np.exp(q)).tolist(),max_price_error=err))
    manifest=dict(format_version=1,model_id='LSTM_evaluation_2021_cal2022_23',architecture='lstm',parameters=6499,
        window=60,feature_names=prep.columns,weights_sha256=before,onnx_sha256=hashlib.sha256(destination.read_bytes()).hexdigest(),
        scaler_sha256=info['preprocessing_sha256'],scaler=prep.__dict__,calibration=summary['uncertainty'],
        metrics=summary['models']['lstm']['test'],model_info={**{k:v for k,v in info.items() if k!='checkpoint'},'checkpoint':'lstm_evaluation.pt / lstm.onnx'})
    (directory/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
    report=ROOT/'reports/academic_final/UAX/ONNX_PARITY.json'
    report.write_text(json.dumps(dict(passed=True,onnx_version=onnx.__version__,runtime_version=ort.__version__,tolerance_usd=.001,cases=cases),indent=2),encoding='utf-8')
    assert before==hashlib.sha256(weights.read_bytes()).hexdigest()
    print(json.dumps({'passed':True,'cases':len(cases),'max_price_error_usd':max(c['max_price_error'] for c in cases),'onnx_bytes':destination.stat().st_size}))


if __name__=='__main__':main()
