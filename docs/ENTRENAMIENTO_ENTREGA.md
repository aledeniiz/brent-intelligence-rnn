# Entrenamiento completo para la entrega del caso Brent

El profesor solicita datos desde enero 2000 hasta la fecha actual,
comparar RNN/LSTM/GRU, predecir 5/63/252 sesiones y usar
validacion walk-forward. El experimento v4 limitado a 2021 **NO es el
modelo final**: este comando entrena TRES MODELOS COMPLETOS con el
historico mas reciente, conservando los ya publicados.

## Problema resuelto: datos recientes descartados

El experimento original excluia cada origen si no habia madurado su
etiqueta a 252 sesiones. Esto tiraba datos de entrenamiento ya validos
para los horizontes de 5 y 63. El nuevo pipeline calcula una mascara de
etiquetas conocidas por cada horizonte: la perdida SmoothL1 y los
escaladores del objetivo emplean solo etiquetas ya observadas antes
del corte temporal. Los objetivos futuros desconocidos nunca
participan en entrenamiento o evaluacion.

## Arquitectura completa

- Tres redes RNN, LSTM y GRU, una capa recurrente, 24 unidades ocultas,
  ventana de 60 sesiones y cabeza residual de 3 salidas de retornos.
- Usa datos Brent y variables economicas WTI, gas, VIX, FX, tipos, metales
  e indice dolar cuando estan disponibles y cumplen cobertura minima;
  seleccion y escalado ajustados solo con datos historicos hasta el corte.
- Cabeza inicial a cero (prediccion inicial ingenua), dropout entrada 15%,
  AdamW weight decay 0,01, penalizacion retorno 0,05, clip gradiente a 1.
- Cada red usa un split interno con tres anos de validacion y purga
  por maduracion para escoger epoca (maximo 20, paciencia 4) y un
  coeficiente de contraccion separado por horizonte, en la cuadricula
  0/0,25/0,5/0,75/1; exige mejora interna minima 1% frente al ingenuo.
- Seleccion de arquitectura por RMSE relativo medio frente al ingenuo
  en folds externos de 2017-19, 2019-21 y 2021-23. La evaluacion
  descriptiva posterior de 2024+ no influye en esta seleccion.
- Los tres candidatos definitivos se reajustan con TODAS las etiquetas
  maduradas hasta la ultima fecha observada (el horizonte de 5 sesiones
  aprovecha datos mas recientes que 63, y este mas que 252).
- Metricas: MAE, MSE, RMSE, mejora frente al ingenuo, ROC-AUC,
  accuracy, precision, recall y F1 direccionales derivados del precio
  previsto (no se ha entrenado un clasificador independiente).
- Curvas de perdida interna train/validation SmoothL1 comparables.

## Ejecucion Windows

    .\.venv\Scripts\python.exe -m scripts.download_data
    .\.venv\Scripts\python.exe -m scripts.train_delivery --dry-run
    .\.venv\Scripts\python.exe -m scripts.train_delivery

Si el CSV ya esta actualizado se puede omitir la descarga. El comando
rechaza por defecto datos con mas de diez dias de retraso; --allow-stale
es solo para pruebas con datos offline/sinteticos.

## Archivos generados

- reports/delivery_training/TIMESTAMP_SHA/{summary.json,REPORT.md,fold_metrics.csv,learning_curves.html}
- artifacts/delivery_candidates/TIMESTAMP_SHA/{rnn,lstm,gru}.pt
- artifacts/delivery_candidates/TIMESTAMP_SHA/{rnn,lstm,gru}_metadata.json

**No se sustituyen automaticamente** los pesos congelados de produccion,
el dashboard, active.json o el seguimiento prospectivo. Primero se revisan
las metricas reales y se decide si tiene sentido reemplazar al existente.

## Limitaciones

Los periodos 2017-23 han sido explorados antes; el diagnostico 2024+
tampoco es un test virgen. Ninguna mejora historica debe presentarse
como un resultado prospectivo independiente. Los movimientos de Brent
en horizontes solapados producen errores dependientes; y el dia de
publicacion de las variables externas no esta certificado point-in-time.
No ajustar repetidamente tras mirar los resultados para favorecer un
periodo. ROC-AUC evalua direcciones, NO una probabilidad de clasificacion.


## Alineación específica con el mensaje del profesor (17/11/2026)

El objetivo central NO es una API ni un dashboard: es entrenar y
comprobar que la red generaliza sobre precios verdaderos no empleados
para ajustarla.

**Protocolo evaluativo reproducible:**

1. Cortar entrenamiento el **29/12/2023**.
2. Para cada arquitectura RNN/LSTM/GRU, elegir época y coeficientes
   usando sólo validación interna anterior al corte.
3. Ajustar la red seleccionada de esa arquitectura exclusivamente con
   etiquetas maduras **hasta dicho corte**.
4. Emitir predicciones para orígenes desde **01/01/2024**, sin incluir
   estos datos en el entrenamiento de los modelos evaluados.
5. Comparar cada pronóstico con el precio real **posteriormente
   observado en su fecha objetivo**. Nunca evaluar una etiqueta que
   todavía no ha madurado (especialmente el horizonte de 252).
6. Mostrar una tabla por red/horizonte con **R², MAE, MSE y RMSE**,
   RMSE ingenuo, y gráficos visuales reales frente a predichos.
7. Tras registrar la evaluación histórica, reajustar **tres modelos de
   entrega independientes** con el histórico íntegro y etiquetas maduras
   hasta la última fecha del CSV; estos modelos de hoy **no tienen
   evaluación independiente propia**. No mezclar resultados.

**Nota de honestidad científica:** los años 2024–2026 ya habían sido
consultados en investigaciones anteriores. La separación temporal es
correcta en este nuevo proceso pero no son datos jamás vistos por el
equipo. No atribuir significación ni presentar este backtest como
validación prospectiva no contaminada.

Los ficheros entregables son:
- holdout_metrics.csv (R², MAE, MSE, RMSE y diagnósticos direccionales)
- holdout_predictions.csv (cada origen, fecha objetivo, precio real,
  estimado e ingenuo)
- real_vs_predicted.html (tres horizontes, valor real alineado a la
  fecha objetivo, RNN/LSTM/GRU e ingenuo)
- real_vs_predicted_scatter.html (dispersión real contra predicho)
- learning_curves.html (train loss y validación interna)
- fold_metrics.csv y summary.json (selección y entrenamiento completo)

**Actualización SIN reentrenar**: el proyecto tiene
scripts.download_data (API FRED/EIA) y GitHub Actions para actualizar
datos e inferir con pesos congelados. Para un nuevo modelo final de este
entrenamiento se puede usar un CSV con las mismas columnas y:

    python -m scripts.predict_delivery --summary reports/delivery_training/RUN/summary.json

O especificar un CSV actualizado manualmente:

    python -m scripts.predict_delivery --summary reports/delivery_training/RUN/summary.json --data ruta/al/csv_actualizado.csv

Eso ejecuta únicamente forward(), sin optimizer, fit, nuevos
escaladores ni reentrenamiento. Comprobar siempre que la fecha del dato
sea posterior a la de entrenamiento. La predicción de hoy es **un
pronóstico**, no una evaluación de su acierto futuro.
