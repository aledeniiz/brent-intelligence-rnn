"""Preflight del entorno local. No descarga ni entrena."""
from __future__ import annotations
import importlib
import json
import platform
import sys
from brent_intelligence.config import ARTIFACTS_DIR,DATA_PATH

def main():
    versions={}
    for mod in ("numpy","pandas","sklearn","torch","yfinance","streamlit","plotly"):
        try:
            obj=importlib.import_module(mod)
            versions[mod]=getattr(obj,"__version__","installed")
        except Exception as exc:
            versions[mod]=f"ERROR: {type(exc).__name__}: {exc}"
    result={"python":sys.version.split()[0],"platform":platform.platform(),
            "packages":versions,"data_present":DATA_PATH.exists(),
            "model_ready":(ARTIFACTS_DIR/"active.json").exists()}
    try:
        import torch
        result["cuda_available"]=bool(torch.cuda.is_available())
    except ImportError:
        result["cuda_available"]=False
    print(json.dumps(result,indent=2,ensure_ascii=False))
    return result

if __name__=="__main__":
    main()
