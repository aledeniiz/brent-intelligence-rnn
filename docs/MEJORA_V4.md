# Brent Intelligence V4: investigacion de regularizacion y persistencia

## Situacion inicial y objetivo

El modelo LSTM v1 (16 inputs, 32 unidades, 6.499 parametros) supero al
ingenuo en el test historico SOLO a 252 sesiones (22,48% RMSE), pero perdio
a 5 y 63. El AUC direccional a 252 fue 0,9277 agregado, aunque cambio
drasticamente entre 2024 y 2025. El entrenamiento mostro senales
compatibles con sobreajuste. Los experimentos v2/v3 no demostraron mejoras
consistentes y el test 2024+ ya fue consultado.

**Objetivo V4: comprobar si una red mas pequena y regularizada generaliza
mejor, SIN optimizar para el test conocido ni declarar mejoras no medidas.**

## Hipotesis y cambios al algoritmo

1. **Menor capacidad:** LSTM recurrente con 16 unidades ocultas (en lugar de
   32), una capa y cabeza de 3 retornos.
2. **Features causales Brent-only:** seis variables derivadas de Brent
   (nivel logaritmico, retornos, volatilidad y desviacion de media movil).
   Evita por defecto exogenas con revision o disponibilidad incierta.
3. **Regularizacion:** dropout de entrada 0,20; AdamW con weight decay
   0,01; penalizacion L2 de salida 0,08 que contrae el retorno hacia cero.
4. **Retorno residual:** no se resta la media del objetivo al normalizar;
   cabeza inicializada a cero, de modo que el pronostico inicial es
   exactamente el ingenuo, ultimo precio observado.
5. **Early stopping cronologico:** la eleccion de epoca se realiza usando
   un split INTERNO con etiquetas maduradas. El criterio es el promedio
   del RMSE/RMSE ingenuo de 5/63/252, no las curvas del test.
6. **Shrinkage sin informacion futura:** para cada horizonte, elegir
   alpha entre 0, 0,25, 0,50, 0,75 y 1 sobre el split interno; exigir
   al menos 1% de mejora interna frente al ingenuo para salir de cero.
   Alpha=0 predice exactamente el ultimo Brent. Por ello, el sistema
   puede honestamente decidir que NO tiene senal predictiva.
7. **Ensemble de dos semillas:** 42 y 31415; promedia retornos
   estimados y reduce dependencia de una sola inicializacion.
8. **Refit por fold:** despues de elegir epocas/shrinkage en el pasado
   interno, se entrena con todos los ejemplos maduros antes del corte
   externo y se evalua SOLO en el periodo externo.

## Periodos y proteccion contra fuga

El codigo recorta fisicamente los datos al 31-12-2021 ANTES de generar
caracteristicas, ventanas o etiquetas. Ningun valor posterior a 2021
interviene. Conserva los cinco cortes externos de v1, usando los mismos
origenes de validacion con etiquetas a 252 sesiones ya maduras.
Para el split interno, el final del aprendizaje se restringe a muestras
cuyas tres etiquetas maduraron antes del corte interno. Despues se elige
la contraccion tambien SOLO en esos datos internos.

No se consulta calibracion (2022-2023), ni test (2024+), ni el archivo
latest_forecast.json de produccion para seleccionar este modelo. No se
modifican los pesos publicados, el workflow programado ni app.py.

**Limite cientifico:** estos folds v1 ya han sido consultados durante
muchas exploraciones. V4 es investigacion y NO un test completamente virgen.
Aunque gane, requerira evaluacion prospectiva nueva antes de afirmar una
mejora generalizable o reemplazar produccion.

## Ejecucion

Preparacion y fechas, sin entrenamiento:

    .\.venv\Scripts\python.exe -m scripts.research_v4 --dry-run

Prueba minima de codigo (un fold, una semilla, 4 epocas y 8 unidades):

    .\.venv\Scripts\python.exe -m scripts.research_v4 --quick

Validacion V4 completa (LSTM, cinco folds, dos semillas y 18 epocas max):

    .\.venv\Scripts\python.exe -m scripts.research_v4

Para probar GRU con EXACTAMENTE el mismo protocolo mas adelante, si se
justifica una segunda arquitectura:

    .\.venv\Scripts\python.exe -m scripts.research_v4 --architecture gru

Genera, sin sobrescribir anteriores, una carpeta versionada bajo:
    reports/research_v4/full/TIMESTAMP_HASH/

Contiene summary.json, REPORT.md, validation_metrics.csv, predictions.csv,
learning_curves.html y fold_improvements.html. Los graficos no muestran datos inventados; tanto train como validacion
muestran SmoothL1. El objetivo total de train (con penalizacion) se
conserva por separado en summary.json.

## Criterios para evaluar resultados

- Calcular RMSE combinado y mejora vs naive por horizonte.
- Contar folds ganados y analizar estabilidad entre anos.
- Comprobar si alpha se anula en muchos folds: indica ausencia de
  evidencia interna suficiente para justificar predicciones activas.
- Examinar curvas y seleccion temprana de epocas para evitar sobreajuste.
- Comparar v4 con LSTM v1 y v2/v3 SOLO en validacion historica compatible.
  No elegir hiperparametros ni afirmar mejoras por el test 2024-2026.
- Evitar declarar significance estadistica con observaciones solapadas.

Una mejora deberia ser persistente, no depender de un fold favorable.
Un resultado igual al ingenuo es mas honesto que una red que empeora
consistentemente el error. No se despliega v4 automaticamente.
