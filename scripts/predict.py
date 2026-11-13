"""Prediccion desde modelo de produccion guardado."""
import json
from brent_intelligence.experiment import predict_latest

if __name__=="__main__":
    pred=predict_latest()
    print(json.dumps({k:v for k,v in pred.items() if k!="summary"},indent=2,ensure_ascii=False))
