# Brent Intelligence — Prediccion de Brent con RNN, LSTM y GRU

Caso 2 de Aprendizaje Automatico (UAX). Aplicacion de pronostico del Brent spot
con comparacion de redes recurrentes, baseline ingenuo e incertidumbre empirica.
Material docente: https://github.com/UAX-VCMT-26-27/aa-case-02-oil-price-prediction

ESTADO: codigo integrado y verificado con tests sinteticos; la descarga y el
entrenamiento con Brent real deben ejecutarse antes de la entrega. No se
presentan metricas de datasets sinteticos como evidencia del mercado.

## Windows 11: ejecucion completa

Instala Python 3.12 (con lanzador py.exe) y Git, descarga este repositorio
y haz doble clic sobre run_windows.cmd. Este lanzador crea .venv, instala
dependencias, ejecuta pruebas, descarga y audita el Brent, entrena RNN/LSTM/GRU,
genera pronosticos y abre Streamlit. Para volver a abrir el panel sin entrenar
usa start_app.cmd.

Tambien se puede ejecutar manualmente desde PowerShell, en el directorio
del proyecto:

    py -3.12 -m venv .venv
    .\.venv\Scripts\python.exe -m pip install -e ".[test]"
    .\.venv\Scripts\python.exe -m pytest -q
    .\.venv\Scripts\python.exe -m scripts.check_environment
    .\.venv\Scripts\python.exe -m scripts.download_data
    .\.venv\Scripts\python.exe -m scripts.audit_data
    .\.venv\Scripts\python.exe -m scripts.train_models --dry-run
    .\.venv\Scripts\python.exe -m scripts.train_models
    .\.venv\Scripts\python.exe -m scripts.predict
    .\.venv\Scripts\python.exe -m streamlit run app.py

Para una prueba rapida aislada:

    .\.venv\Scripts\python.exe -m scripts.train_models --smoke --epochs 2 --hidden-size 8

El modo smoke NO publica el modelo activo. La opcion --device cuda solo es
adecuada cuando PyTorch confirma GPU NVIDIA disponible.

## Estructura

- brent_intelligence/config.py: configuracion inicial y tres horizontes.
- brent_intelligence/features.py: transformaciones de variables sin mirar al futuro.
- brent_intelligence/temporal.py: ventanas y folds de validacion temporal.
- brent_intelligence/modeling.py: RNN, LSTM y GRU multihorizonte.
- brent_intelligence/evaluation.py: RMSE, MAE, MSE y naive.
- brent_intelligence/experiment.py: entrenamiento, seleccion, calibracion,
  evaluacion con test reservado, artefactos y modelo de produccion separado.
- scripts/download_data.py: FRED/EIA, Yahoo opcional, CSV OPEP opcional.
- scripts/audit_data.py: auditoria de cobertura e integridad.
- scripts/train_models.py: CLI de investigacion completa.
- scripts/predict.py: inferencia desde un modelo ya versionado.
- scripts/check_environment.py: comprobaciones del sistema.
- app.py: dashboard empresarial Streamlit.
- docs: metodologia, ejecucion y guion comercial.

Los archivos descargados se guardan en data/, los pesos y transformadores
en artifacts/runs/ y las auditorias en reports/. NO se suben a GitHub.
La ultima ejecucion completa queda identificada por artifacts/active.json
y conserva el hash SHA256 del CSV, configuracion, metricas y pesos.

## Protocolo matematico

Objetivo: Brent spot (USD/barril), desde 2000. Entradas: ventana de 60
sesiones observadas. Salidas: 5, 63 y 252 sesiones; retornos logaritmicos
futuros convertidos despues a precios positivos. Tres redes, mismo
presupuesto y mismas particiones; predictor ingenuo = ultimo Brent.

Cinco folds walk-forward entre 2011 y 2021. En cada corte solo se entrenan
ejemplos cuyas TRES etiquetas ya hayan madurado. El escalador de las
caracteristicas se ajusta al pasado del fold. Entrenamiento final evaluado
con etiquetas hasta 2021; calibracion posterior 2022-2023; test reservado
para anclajes desde 2024 con etiquetas ya observadas.

La arquitectura se escoge usando la media del cociente RMSE/RMSE ingenuo
en los folds y horizontes, antes del test. Las bandas del 80% nominal se
calibran en errores logaritmicos y se mide su cobertura en test.

Despues se entrena un MODELO DE PRODUCCION distinto con las etiquetas que
ya hayan madurado hasta la ultima descarga. Las metricas del test describen
los modelos historicos, no constituyen una evaluacion independiente del
modelo actualizado de produccion.

## Datos complementarios y fuentes

FRED/EIA proporciona Brent y WTI, gas, FX, BCE y VIX. Yahoo agrega
futuros, cobre, oro e indice dolar cuando el proveedor responde. El
script del profesor no descarga la cesta OPEP; para añadir un CSV diario
verificado de dos columnas Date,orb, ejecutar:

    python -m scripts.download_data --orb-csv ruta/a/orb.csv

Puedes omitir Yahoo con --fred-only. El Brent no se rellena con ffill,
y los valores negativos historicos de WTI no se eliminan arbitrariamente.

## Limitaciones y entrega

Las variables externas pueden haberse publicado despues del precio spot
o revisado historicamente. Desfasarlas una sesion no garantiza un conjunto
point-in-time. Las bandas y los errores fuera de muestra no garantizan
ganancias, ni anticipan shocks geopoliticos, ni constituyen backtest de
trading. Un modelo puede perder frente al baseline y debe comunicarse.

El profesor pide el CODIGO EN GITHUB y una presentacion de MAXIMO CINCO
diapositivas (sin contar portada). No inventar valores de precision o
cobertura. Consulta docs/METODOLOGIA.md, docs/ENTREGA.md y
docs/PRESENTACION.md. No compartir ni subir PAT, secretos o credenciales.
