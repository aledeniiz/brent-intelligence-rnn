"""Congela el modelo de produccion sin entrenar. Solo escribe deployment/frozen/."""
from __future__ import annotations
import argparse
from pathlib import Path
from brent_intelligence.config import ARTIFACTS_DIR,DATA_PATH
from brent_intelligence.frozen import FROZEN_DIR,export_frozen

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data",type=Path,default=DATA_PATH)
    p.add_argument("--artifacts",type=Path,default=ARTIFACTS_DIR)
    p.add_argument("--destination",type=Path,default=FROZEN_DIR)
    args=p.parse_args()
    manifest=export_frozen(artifacts=args.artifacts,data=args.data,destination=args.destination)
    print("Modelo congelado preparado (aun no publicado en GitHub).")
    print("Arquitectura:",manifest["architecture"])
    print("Fecha de corte:",manifest["trained_data_end"])
    print("SHA256 pesos:",manifest["weights_sha256"])
    print("Directorio:",args.destination)
    print("Para habilitar GitHub Actions se deben publicar los 3 archivos de este directorio.")

if __name__=="__main__":
    main()
