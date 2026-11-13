# Arquitectura de Brent Intelligence

Fase inicial del Caso 2 UAX. Estructura modular:

1. scripts/download_data.py adquiere datos spot diarios FRED/EIA y deja ausentes sin interpolar.
2. brent_intelligence/features.py transforma retornos y retrasa exogenas al menos una sesion.
3. brent_intelligence/temporal.py construye ventanas de 60 sesiones, etiquetas 5/63/252 y cinco folds.
4. brent_intelligence/modeling.py define RNN, LSTM y GRU con la misma cabeza multisalida.
5. brent_intelligence/evaluation.py mide precios en USD/barril frente a naive.
6. scripts/audit_data.py deja un manifiesto de calidad.

## Principios

- Ninguna etiqueta puede participar en el entrenamiento antes de su fecha de maduracion.
- Entrenamiento y escalado deben usar solo el pasado de cada fold.
- Los indicadores auxiliares tienen revisiones y distintos calendarios: retrasar una sesion no constituye prueba point-in-time.
- No informar resultados empiricos sin ejecutar los entrenamientos sobre Brent.
- No guardar secretos, grandes modelos ni datos descargados en Git.
- Siguiente fase: entrenamiento walk-forward, calibracion 2022-2023, test reservado desde 2024 y app Streamlit.
