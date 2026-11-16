# Actualizacion automatica del Brent sin reentrenar

## En que consiste

El flujo de GitHub Actions Daily frozen Brent refresh esta programado de
lunes a viernes, 20:25 UTC. Puede arrancarse manualmente con Run workflow.
Se ejecuta en la nube, no depende de encender el portatil.

Cada ejecucion:
1. Comprueba que existe el paquete de pesos congelados.
2. Descarga datos nuevos EIA/FRED y variables externas Yahoo disponibles.
3. Verifica versiones de modelo, checksum de pesos, frescura de Brent y
   caracteristicas que necesita la red.
4. Realiza inferencia sin entrenar y, si llega una nueva observacion, guarda
   public/latest_forecast.json en el repositorio.

Si falta un dato imprescindible, falla sin sobrescribir el pronostico valido.
No inventa valores. Tampoco realiza optimizacion de hiperparametros.

## Una sola vez: preparar modelo congelado desde el portatil

Desde la carpeta con datos y modelos REALES del entrenamiento definitivo:

    .\.venv\Scripts\python.exe -m scripts.plot_diagnostics

Este comando no reentrena; genera reports/diagnostics con:
- Tres HTML de loss train/validation, uno por arquitectura y cinco folds.
- Tres HTML de precios reales frente a predichos e ingenuo en test.
- Tres HTML de residuos por horizonte.
- CSV reproducible de predicciones evaluadas.

Requiere el CSV original exacto del entrenamiento (se coteja SHA256). Si
ya cambiaste ese CSV, recupera la copia original antes de reconstruir test.

Despues ejecuta, igualmente sin entrenar:

    .\.venv\Scripts\python.exe -m scripts.export_frozen

Produce:
    deployment/frozen/production.pt
    deployment/frozen/preprocessing.json
    deployment/frozen/manifest.json

El manifest contiene fecha del modelo, variables, calibracion, benchmark
historico y SHA256 del fichero de pesos. No incluye dataset completo.
Los tres ficheros se pueden consultar publicamente una vez versionados
en GitHub; solo publica tu propio entrenamiento academico.

Prueba primero la inferencia local con un dataset actualizado:

    .\.venv\Scripts\python.exe -m scripts.download_data
    .\.venv\Scripts\python.exe -m scripts.update_frozen

La actualizacion local genera public/latest_forecast.json. El dashboard
publico se abre asi:

    .\.venv\Scripts\python.exe -m streamlit run app_public.py

Si funciona y has revisado el checkpoint, puedes autorizar a Codex
a versionar EXCLUSIVAMENTE los tres archivos de deployment/frozen
y el primer public/latest_forecast.json. No incluyas datasets ni artifacts.
No fuerces pushes ni sobrescribas cambios locales del dashboard.

## Activacion automatica

En GitHub: Actions > Daily frozen Brent refresh > Run workflow.
En Settings > Actions > General verifica Workflow permissions con
acceso de lectura/escritura si el token no permite crear commits.

El YAML esta preparado para actuar solo cuando estan publicados los tres
archivos del modelo. Sin ellos, informa que falta el bundle y omite
descarga/prediccion. La agenda no puede correr si Actions esta deshabilitado
en el repositorio. Las ejecuciones schedule pueden demorarse.

## Interfaz online

Para publicar la app en Streamlit Community Cloud selecciona como archivo
principal app_public.py. Esta app no entrena ni descarga datos; solo lee
public/latest_forecast.json, guardado por GitHub Actions. El original app.py
sigue disponible para ejecutar el backtest en tu portatil.

## Riesgos y limitaciones

- El modelo congelado NO aprende de las nuevas cotizaciones.
- Las metricas historicas no son un test independiente del modelo de
  produccion; se presentan solo como referencia del estudio original.
- Las bandas del 80% nominal son empiricas y su cobertura futura no esta
  garantizada.
- Las exogenas no estan certificadas point-in-time.
- Una respuesta fallida de Yahoo/FRED puede bloquear la actualizacion
  antes que aceptar entradas incompletas sin avisar.
- El panel marca datos de mas de una semana como desactualizados.
- El entrenamiento permanece congelado hasta que se decida realizar
  una nueva evaluacion cientifica, no se reentrena automaticamente.
