# Auditoría de cumplimiento — Caso de uso III (17/11/2026)

Auditoría del caso académico de predicción del Brent, frente al
código y los artefactos realmente disponibles.

## Implementado y comprobable

- Brent y WTI EIA/FRED diarios desde enero de 2000, con fuentes
  adicionales FRED/Yahoo, además de importación opcional de cesta OPEP.
- Limpieza del Brent observado, tratamiento de ausentes, secuencias
  de 60 sesiones y escaladores entrenados solo con datos del pasado.
- Comparación RNN/LSTM/GRU, baseline ingenuo, 5/63/252 sesiones,
  validación temporal walk-forward y MAE, MSE, RMSE y R².
- La evaluación histórica termina su entrenamiento en 2023 y
  evalúa orígenes posteriores con los objetivos maduros.
- El reajuste final utiliza todas las etiquetas maduras hasta el
  último Brent descargado, sin atribuirse sus métricas históricas.
- app_delivery.py muestra la arquitectura de v4 realmente seleccionada
  (RNN por validación), horizonte, pronósticos, cambios, métricas,
  bandas empíricas orientativas, episodios de 2008, 2014–2016, 2020,
  2022 y 2026, y limitaciones.
- La app permite incorporar un CSV histórico actualizado o nuevas filas
  y generar previsiones con los mismos pesos, sin entrenamiento.
- El generador nuevo de la presentación usa los artefactos del modelo
  v4, no el de una LSTM anterior; crea exactamente cinco diapositivas.

## Pendientes de cumplir o confirmar

1. OPEP ORB: la serie NO estaba en el último CSV. El importador
   opcional ya existe: python -m scripts.download_data --orb-csv
   ruta/serie_oficial.csv . Debe conseguirse y verificarse la
   serie oficial diaria antes de afirmar que se ha incluido.
2. Tiempo de entrenamiento v4: no se registraron segundos por red
   comparables en la nueva ejecución; los tiempos del experimento
   anterior no pueden reutilizarse como si fueran v4. El requisito
   de tabla con tiempo debe declararse pendiente o instrumentarse
   en una nueva ejecución explícitamente autorizada.
3. Incertidumbre RNN v4: sus bandas son DESCRIPTIVAS, calculadas
   con residuos de modelos históricos ya inspeccionados, no
   están calibradas ni validadas para el reajuste final.
   Nunca afirmar una cobertura futura del 80 % como hecho.
4. Las revisiones históricas y la disponibilidad point-in-time de
   datos externos no están certificadas; retraso de una sesión no
   resuelve completamente esa limitación.
5. Los pesos y reportes v4 están en el ordenador local, no en
   GitHub. Un repositorio limpio requiere ejecutar el entrenamiento
   para obtenerlos. Es código reproducible, no un binario publicado.
6. Confirmar que el profesor tiene acceso al repositorio compartido
   definitivo y que su URL es la anunciada.
7. Hay que generar y revisar el nuevo PowerPoint comercial de v4.
   La presentación anterior utiliza métricas de otra LSTM.
8. app_public.py sigue con el modelo LSTM congelado. No presentarlo
   como la RNN v4 seleccionada. La demostración académica v4
   recomendada es app_delivery.py.
9. run_windows.cmd sigue siendo el lanzador antiguo del flujo v1.
   Para la demostración v4 usar start_delivery_app.cmd.

## Comandos (Windows PowerShell)

Para abrir la app (sin entrenamiento):
    .\.venv\Scripts\python.exe -m streamlit run app_delivery.py

O abrir start_delivery_app.cmd.

Para crear el documento comercial con cinco diapositivas:
    .\.venv\Scripts\python.exe -m scripts.build_delivery_presentation --summary reports/delivery_training/RUN/summary.json

Para inferencia directa, sin actualizar pesos:
    .\.venv\Scripts\python.exe -m scripts.predict_delivery --summary reports/delivery_training/RUN/summary.json

Se exige un resumen v4 y sus pesos correspondientes. Si no existen,
no presentar las previsiones de otra red como propias de v4.

## Advertencias científicas

El periodo 2024–2026 ya se había inspeccionado durante el desarrollo
y no es un test completamente virgen. El reajuste final carece de
evaluación independiente propia. Los horizontes solapados inducen
dependencia en los errores y reducen la evidencia estadística.
