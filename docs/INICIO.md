# Inicio rapido

El repositorio conserva los archivos README y .gitignore creados al abrirlo en GitHub.

Requisitos: Python 3.12, conexion a Internet, Git.

1. Crear entorno: python -m venv .venv
2. Activarlo en PowerShell: .venv\Scripts\Activate.ps1
3. Instalar: python -m pip install -e ".[test]"
4. Descargar Brent/FRED: python -m scripts.download_data
5. Auditar: python -m scripts.audit_data
6. Probar: python -m pytest -q

El descargador integrado en esta primera subida obtiene Brent, WTI y otros indicadores de FRED. Yahoo Finance (metales, futuros e indice dolar) y cesta OPEP se incorporaran como conectores auxiliares con auditoria por fuente.

Estado: codigo de arquitectura y pruebas. La generacion de modelos y la app siguen pendientes de la siguiente fase. Ninguna metrica ha sido entrenada sobre precios reales en este repositorio.

Fuentes docentes: https://github.com/UAX-VCMT-26-27/aa-case-02-oil-price-prediction
Guia PAT docente: https://github.com/UAX-VCMT-26-27/.github/blob/main/authentication/README.md
