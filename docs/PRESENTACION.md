# Guion comercial: maximo cinco diapositivas, sin portada
1. **Problema de negocio:** volatilidad de precios y necesidad de cuantificar riesgos.
2. **Datos y producto:** Brent EIA/FRED, tres horizontes y datos explicativos.
3. **Evidencia:** tabla RNN/LSTM/GRU/ingenuo y resultados reales de test.
4. **Demo:** captura Streamlit + previsiones + intervalo empirico + advertencias.
5. **Propuesta de valor y limites:** que decisiones puede informar, cobertura,
   shocks, limitacion point-in-time, siguientes validaciones.

No inventar precios, mejoras ni tasas de cobertura. Obtener las cifras del
JSON artifacts/runs/<id>/summary.json generado con datos reales.


## Generacion automatica

Ejecuta python -m scripts.build_presentation una vez finalizado el
entrenamiento real. Se creara un PowerPoint de exactamente cinco diapositivas
con los resultados del CSV/modelos. Revisalo antes de defenderlo.
