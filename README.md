# Brent Intelligence

**Predicción del precio del petróleo Brent mediante redes neuronales recurrentes — Caso de uso III, Aprendizaje Automático (UAX).**

[**Abrir aplicación web**](https://aledeniiz.github.io/brent-intelligence-rnn/) · [**Ver código del modelo**](brent_intelligence/) · [**Metodología**](docs/METODOLOGIA.md) · [**Evaluación y límites**](docs/UAX_AUTOMATIC_WEB.md)

> **Importante:** la web pública ejecuta una **LSTM real en ONNX Runtime Web**, pero arranca con **datos sintéticos CC0** para demostrar la tecnología sin redistribuir cotizaciones de terceros. Sus resultados en ese modo **no son predicciones del mercado actual**. Las métricas académicas de abajo proceden de una evaluación histórica con datos reales y no de la demostración sintética.

## ¿Qué hemos construido?

Un sistema de análisis de series temporales del petróleo Brent que compara **RNN, LSTM y GRU** frente a una referencia sencilla (predicción *naive*: mantener el último precio). Combina investigación reproducible con una aplicación interactiva accesible desde el navegador.

- **Horizontes:** 5, 63 y 252 sesiones.
- **Entradas:** ventana de 60 sesiones y 16 variables financieras y energéticas.
- **Modelo de la web:** LSTM histórica, **6.499 parámetros**, exportada a ONNX sin reentrenar ni cambiar sus pesos.
- **Visualización:** tres horizontes, variaciones estimadas, bandas empíricas de incertidumbre del 80 % nominal, gráficos y selector de fecha aproximada.
- **Validación:** comparación homogénea entre arquitecturas, cinco folds temporales (*walk-forward*), calibración diferenciada y evaluación retrospectiva posterior.
- **Entrega técnica:** GitHub Pages, ONNX Runtime Web, documentación y tests automatizados.

## Resultados académicos: LSTM frente a naive

El protocolo de la LSTM publicada en ONNX utiliza entrenamiento histórico hasta **2021**, calibración en **2022–2023** y una evaluación retrospectiva de orígenes desde **2024**, con **474 ejemplos por horizonte**. Un RMSE más bajo es mejor.

| Horizonte | RMSE LSTM | RMSE naive | Reducción relativa de RMSE |
|:--|--:|--:|--:|
| 5 sesiones | 3,263 | 3,207 | **−1,73 %** |
| 63 sesiones | 8,498 | 6,609 | **−28,58 %** |
| 252 sesiones | 20,049 | 25,863 | **+22,48 %** |

**Interpretación:** la LSTM mejora al naive en el horizonte anual, pero **no** en los horizontes de 5 y 63 sesiones. No se reivindica significación estadística, cobertura futura garantizada ni rentabilidad. La evaluación de 2024 en adelante ha sido examinada durante el desarrollo y **no debe describirse como test virgen**. Las observaciones de horizontes largos se solapan temporalmente.

Estos valores pertenecen al **checkpoint histórico de la web**, no a la LSTM reajustada con datos de producción ni a los experimentos de investigación V4/V5. Fuente técnica auditable: [manifiesto ONNX con parámetros, escalador, métricas, calibración y hashes](web/pages/assets/model/manifest.json).

## Probar la aplicación en 30 segundos

1. Abre **[Brent Intelligence](https://aledeniiz.github.io/brent-intelligence-rnn/)**; no requiere instalación ni cargar archivos.
2. Comprueba el aviso **«escenario sintético / demostración»**. La red ONNX sí ejecuta inferencia real; **las entradas no son cotizaciones reales**.
3. Explora los horizontes 5, 63 y 252, los gráficos y el selector de fecha. El selector aproxima el horizonte disponible más cercano: **no predice exactamente cualquier fecha arbitraria**.

La aplicación puede admitir entradas preparadas y autorizadas, procesadas localmente en el navegador. La **ingesta pública automática de datos financieros reales no está habilitada** porque todavía no se han acreditado todos los derechos de redistribución. El workflow de comprobación periódica valida el escenario sintético, **no descarga precios de mercado**.

## Organización del repositorio

| Ruta | Contenido |
|:--|:--|
| [`web/pages/`](web/pages/) | Aplicación pública, inferencia ONNX, interfaz y escenario sintético |
| [`brent_intelligence/`](brent_intelligence/) | Preparación temporal, modelos RNN/LSTM/GRU y evaluación |
| [`scripts/`](scripts/) | Descarga, entrenamiento, predicción, diagnóstico y exportación |
| [`tests/`](tests/) | Tests de modelos, protocolos temporales y paquete ONNX |
| [`docs/`](docs/) | Metodología, arquitectura, evolución de experimentos y limitaciones |
| [`.github/workflows/`](.github/workflows/) | Tests, publicación del frontend y verificación de la demo |

## Reproducir las verificaciones

Con Python 3.12, desde la raíz del repositorio:

```bash
python -m pip install -e ".[test]"
python -m pytest -q
```

Para servir localmente el mismo frontend estático:

```bash
python -m http.server 8770 --directory web/pages
```

Abre `http://localhost:8770`. Los tests de paridad con PyTorch que requieren los checkpoints históricos originales pueden quedar omitidos si dichos archivos privados no están presentes; el ONNX y su manifiesto público sí se distribuyen.

El entrenamiento sobre datos reales y algunos diagnósticos requieren datasets y artefactos locales no redistribuidos. No se debe interpretar la clonación del repositorio como una reproducción automática de todo el experimento histórico sin esas fuentes.

## Documentación y trazabilidad

- [Metodología científica](docs/METODOLOGIA.md) · [Protocolo de entrenamiento y evaluación](docs/ENTRENAMIENTO_ENTREGA.md)
- [Implementación de la demo automática y auditoría de fuentes](docs/UAX_AUTOMATIC_WEB.md)
- [Exportación ONNX y requisitos de publicación](docs/ONNX_RELEASE.md) · [Licencias del paquete](web/pages/LICENSES_AND_MODEL.md)
- [Asignación original del caso](https://github.com/UAX-VCMT-26-27/aa-case-02-oil-price-prediction)

**Limitaciones:** los datos exógenos no están plenamente certificados *point-in-time*; los intervalos de cobertura nominal del 80 % se calibraron históricamente y no garantizan cobertura futura; el estudio no constituye una estrategia de trading validada. El escenario sintético solo demuestra funcionamiento de software.

---

*Proyecto académico UAX. Las implementaciones exploratorias antiguas (Streamlit, V4/V5 y producción congelada) se conservan en el repositorio como material de investigación, pero no deben confundirse con la LSTM histórica publicada en la web.*
