# Registro y evaluación prospectiva del modelo congelado

## Alcance y estado

El sistema previo mantenía `public/latest_forecast.json`, que se sobrescribía.
Git conserva algunas versiones publicadas, pero no un registro evaluable por
emisión. Se añade un archivo inmutable por predicción y un evaluador de
etiquetas maduras. Los pesos, escaladores, hiperparámetros, dashboards y
predicciones previamente emitidas permanecen intactos.

El seguimiento se activa únicamente cuando el PR se fusione en `main`.
Publicar la rama no activa la actualización: el workflow de producción
rechaza ejecuciones desde otras ramas. El CI comprueba el código sin realizar
el antiguo entrenamiento smoke con datos reales; mantiene las pruebas sintéticas.

## Archivos y versiones

- `public/prospective/forecasts/AAAA-MM-DD/<forecast_id>.json`: predicción
  original completa, fecha de registro UTC, procedencia, versión del esquema,
  SHA256 del contenido y hashes de pesos, escalador y manifiesto para nuevas
  emisiones. La versión del modelo identifica el bundle completo, no solo pesos.
- `public/prospective/labels/<forecast_id>_<h>.json`: primera etiqueta observada
  cuando madura, fecha objetivo, cotización, calendario observado y hash del
  CSV utilizado. Nunca se sustituye una etiqueta por una revisión posterior.
- `public/prospective_evaluation.json`: resumen regenerable por versión y
  horizonte, MAE/RMSE de modelo e ingenuo, número de pares y exclusiones.
  Sin pares maduros, las métricas son `null`, no cero.
- `public/latest_forecast.json`: sigue atendiendo al dashboard sin cambios
  de formato. Solo se actualiza después de archivar la nueva predicción.

La escritura del registro publica archivos completos con creación exclusiva;
las repeticiones son idempotentes y nunca promueven un registro antiguo a
prospectivo. Las revisiones de datos o de un pronóstico quedan en archivos
distintos. Una alteración de contenido invalida su checksum y detiene la
evaluación antes de reemplazar el informe anterior.

Esto no es un servicio de sellado temporal criptográfico: un propietario con
acceso al repositorio podría manipular archivos y recalcular sus hashes.
Los commits del workflow aportan evidencia externa de cuándo se publicaron,
pero registrar localmente no equivale a haber publicado en GitHub.

## Qué cuenta como prospectivo

Solo una nueva inferencia del modelo congelado capturada en vivo durante
`scripts.update_frozen`, con comprobación de frescura, puede registrarse como
`live_issue`. No se ajusta ni optimiza ningún parámetro.

El JSON ya existente se importa como `legacy_import`, conservando exactamente
sus valores y su fecha original de generación. La fecha de incorporación al
registro se guarda aparte. No se acredita retroactivamente como emisión
prospectiva. Los backtests reconstruidos y las inferencias ejecutadas con
`--allow-stale` se excluyen igualmente como reconstrucciones históricas.

No se recalculan ni se rellenan predicciones para fechas pasadas. Los resultados
de v1/v2/v3 hasta 2021 siguen siendo validación histórica reutilizada, no un
test prospectivo independiente.

## Maduración sin fuga temporal

1. En cada origen se conservan el precio ingenuo y los precios pronosticados
   de 5, 63 y 252 sesiones tal como se emitieron.
2. El objetivo es la h-ésima cotización **observada** del Brent posterior al
   origen, no h días naturales ni un calendario genérico de días laborables.
3. Solo se examinan cotizaciones con fecha hasta el día UTC de evaluación.
   Una predicción debe haber quedado registrada antes del día de su objetivo.
   Se excluye el objetivo si su fecha es igual o anterior al día de registro,
   incluso si la generación original declaraba una fecha más antigua.
   Esta regla conservadora evita atribuir como desconocido un resultado de
   una fecha que ya había pasado; no supone conocer su hora de publicación.
4. Se excluyen modelos cuyo corte de entrenamiento sea posterior al origen,
   registros futuros e importaciones. No se consumen etiquetas registradas
   después del instante de evaluación solicitado.
5. Para cada versión/origen/horizonte se evalúa la primera emisión en vivo.
   Las revisiones posteriores permanecen visibles pero no duplican la muestra
   ni permiten escoger retrospectivamente el pronóstico más acertado.
6. Al madurar se congela la primera cotización objetivo observada. Revisiones
   posteriores de precios o del calendario generan advertencias; no alteran
   etiquetas previas ni los errores ya vinculados a ellas. El precio ingenuo
   permanece igual al precio de origen originalmente emitido.

MAE es la media de errores absolutos y RMSE la raíz de la media de errores
cuadrados sobre los mismos pares maduros para modelo e ingenuo. La mejora es
`100 * (1 - RMSE_modelo / RMSE_ingenuo)`; con RMSE ingenuo cero se deja `null`.
Se informa el número de observaciones por horizonte y versión. Las etiquetas
solapadas son dependientes: no se afirma significación, rentabilidad ni mejora
predictiva sin datos nuevos, y estos errores no se usan para optimizar el modelo.

No se certifican las fechas históricas de publicación de variables externas ni
se dispone de un calendario oficial inmutable de cotizaciones. El calendario
utilizado al resolver cada etiqueta se conserva para auditar revisiones.

## Ejecución

Inicializar desde el JSON existente, sin nueva inferencia ni descarga:

    .\.venv\Scripts\python.exe -m scripts.evaluate_prospective --import-latest

Evaluar el registro con el CSV existente, sin generar pronósticos:

    .\.venv\Scripts\python.exe -m scripts.evaluate_prospective

En futuras actualizaciones normales:

    .\.venv\Scripts\python.exe -m scripts.update_frozen

Ese comando archiva la emisión y evalúa los objetivos que hayan madurado. El
workflow preparado publica exclusivamente el último JSON, el registro
prospectivo y el informe de evaluación. Si no cambia la cotización, se sigue
evaluando el registro; no se crea otra emisión del mismo contenido.
Antes de publicar, el workflow rechaza cualquier modificación o eliminación de
archivos ya archivados y cualquier archivo fuera de la lista permitida.

## Comprobaciones

Las pruebas cubren inmutabilidad, idempotencia, revisiones, ventanas futuras,
maduración por sesiones, coherencia de versiones, importaciones excluidas,
etiquetas primeras observadas y fallo de archivo antes de publicar.
La batería completa incluye dos ajustes neuronales sintéticos preexistentes,
autorizados expresamente para pruebas; nunca modifica los pesos publicados.

Los hashes de investigación se explican en
`reports/research_consolidation/dataset_audit.json`, y los 26 resultados de
validación guardados se reúnen en `research_comparison.json` y `CONSOLIDACION.md`
del mismo directorio. Los reportes no se reconstruyen entrenando modelos.
