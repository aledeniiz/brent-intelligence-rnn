# Brent Intelligence V5: candidatos independientes 5 / 63 / 252 sesiones

## Objetivo

La nueva evaluación cronológica hasta 2023 mostró que la RNN final
seleccionada no superaba al predictor ingenuo a 5 sesiones,
apenas mejoraba un 0,20 % a 63, y en 252 producía un resultado
histórico positivo pero inestable. Esos resultados pertenecen al
modelo multiobjetivo; **no son resultados de V5**.

Este experimento prueba la hipótesis de interferencia entre
horizontes: una red única de tres salidas debe ajustar objetivos
muy distintos y el horizonte anual dispone de menos etiquetas
maduras. Ahora entrenamos redes independientes RNN/LSTM/GRU de
**una salida por horizonte** usando todas las etiquetas ya
conocidas de ese objetivo, sin esperar al 252 para entrenar
los horizontes 5 y 63.

Anteriormente se investigaron LSTM monohorizonte en V2 sin ventaja
convincente, así que NO afirmamos que separar objetivos garantice
una mejora: el resultado decidirá.

## Modelo

- 60 observaciones de ventana, características Brent y exógenas
  económicas que cumplen cobertura en el corte pasado.
- Una capa recurrente, 16 unidades, dropout de entrada 0,15,
  AdamW weight decay 0,01 y penalización L2 sobre el retorno 0,025.
- Cabeza residual inicializada a 0: la primera predicción es
  exactamente el último Brent (baseline ingenuo).
- Cada objetivo es el retorno logaritmico 5/63/252, normalizado por
  RMS calculada con exclusivamente las etiquetas de entrenamiento
  maduradas antes del corte.
- Early stopping usando RMSE/precio ingenuo en una partición interna
  purgada de 3 años. Se selecciona una contracción interna de
  0/0,25/0,5/0,75/1 con exigencia de mejora del 1 %; se informa
  tanto de la predicción original sin contracción como de la final.
  Una contracción a cero **no es aprendizaje predictivo**.
- Se comparan las tres arquitecturas en tres folds anteriores
  a 2024 (2017-19, 2019-21, 2021-23). Para cada horizonte se elige
  por la media de RMSE/ingenuo en esos folds.
- Después de elegir arquitectura se calcula un diagnóstico de
  2024+ con cada modelo ajustado solo hasta 2023; este periodo ya
  había sido examinado y NO es un test virgen ni criterio de ajuste.
- Refit final de tres modelos, uno por horizonte, con todas las
  etiquetas maduras hasta la última fecha del CSV. NUNCA se
  les atribuyen métricas del modelo histórico como propias.
- Se registran R², MAE, MSE, RMSE, naive, ROC-AUC, precision, recall,
  F1, pérdidas, predicciones por fecha y mejoras de cada fold.

## Ejecutar

Primero inspeccionar fechas:

    python -m scripts.train_horizon_models --dry-run

Ejecutar una única comparación predefinida completa:

    python -m scripts.train_horizon_models

Reproducir pronósticos usando pesos y escaladores guardados **sin entrenar**:

    python -m scripts.predict_horizon_models --summary reports/horizon_v5/RUN/summary.json

Para incorporar datos recientes se puede actualizar el CSV mediante
scripts.download_data, o pasar un CSV manual con --data a inferencia.
Actualizar datos NO significa reentrenar.

El modo --quick es exclusivamente una prueba pequeña del código, no
evidencia de mejora sobre Brent.

## Entregables

- reports/horizon_v5/RUN/REPORT.md
- reports/horizon_v5/RUN/summary.json
- reports/horizon_v5/RUN/fold_metrics.csv
- reports/horizon_v5/RUN/validation_predictions.csv
- reports/horizon_v5/RUN/historical_diagnostic_metrics.csv
- reports/horizon_v5/RUN/historical_diagnostic_predictions.csv
- reports/horizon_v5/RUN/learning_curves.html
- reports/horizon_v5/RUN/real_vs_predicted.html
- artifacts/horizon_candidates/RUN/h5.pt, h63.pt, h252.pt

## Interpretación

No seleccionar según el test 2024+, ni inventar significación con
orígenes solapados. Una mejora es creíble cuando aparece en distintos
folds y no depende solo de un periodo favorable. Si el modelo
selecciona contracción 0 a 5, no superó al ingenuo: documentarlo.

Los datos exógenos históricos no están certificados point-in-time.
El modelo congelado de la aplicación permanece intacto hasta
elegir justificadamente un sustituto, si lo hubiera.
