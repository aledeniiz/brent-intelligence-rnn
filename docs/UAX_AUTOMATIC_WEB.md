# Arranque automático UAX sin datos privados

La implementación NO ofrece pronósticos de mercado actualizados. Ofrece inferencia ONNX real automática sobre un escenario completamente sintético, explícitamente rotulado. Ninguna entrada se descarga del ordenador del usuario. No se publica la instantánea financiera privada.

## Contrato congelado
LSTM histórica: 6.499 parámetros; ventana 60×16; checkpoint entrenado hasta 2021 y calibración histórica 2022–2023. Pesos, escalador y calibración permanecen sin cambios. Las características, en orden, son:
1. brent_log_level
2. brent_ret_1
3. brent_ret_5
4. brent_ret_21
5. brent_vol_21
6. brent_vs_ma_21
7. wti_spread_relative
8. wti_change_scaled
9. gas_natural_log_change_5
10. tipo_bce
11. usd_por_eur_log_change_5
12. cny_por_usd_log_change_5
13. vix_log_level
14. cobre_log_change_5
15. oro_log_change_5
16. indice_dolar_log_change_5

Se necesitan Brent y nueve exógenas; al menos 82 filas completas para obtener 60 filas con calentamiento. Las exógenas llevan lag de una observación, cambios de cinco sesiones, volatilidad muestral ddof=1 y normalización congelada; no se ajusta ningún estadístico. El pipeline Python original genera el tensor de referencia; JavaScript reconstruye características desde las 100 filas sintéticas y comprueba la concordancia antes de inferir. El hash SHA256 del archivo verifica también las entradas brutas.

## Auditoría de fuentes — 20/11/2026
- Brent, WTI y gas natural: series EIA disponibles. La política general permite reutilizar productos gubernamentales con atribución, pero exceptúa materiales licenciados de terceros. Esa política por sí sola no licencia las nueve exógenas ni acredita cada serie: https://www.eia.gov/about/copyrights_reuse.php
- Tipo BCE: ECBDFR en FRED. No se ha documentado aquí una autorización específica de redistribución del producto FRED. No publicado.
- USD/EUR y CNY/USD: DEXUSEU y DEXCHUS en FRED. No se ha cerrado la verificación específica de redistribución del producto. No publicados.
- VIXCLS: FRED identifica expresamente copyright de CBOE; acceso público no concede por sí solo redistribución. No publicado. https://fred.stlouisfed.org/series/VIXCLS
- Cobre, oro e índice dólar: HG=F, GC=F y DX-Y.NYB obtenidos mediante Yahoo; no se acreditó derecho de republicar estas series ni sus entradas derivadas. El acceso a los términos devolvió HTTP 999; no se interpreta como permiso. No publicados.
- FRED distingue derechos propios y de terceros: https://fred.stlouisfed.org/legal/terms/
- No se sustituyen VIX, futuros o índice dólar por proxies: cambiaría el significado de las entradas del modelo congelado. Tampoco se consideran CORS o una descarga directa como permisos de uso.

## Alternativa implementada y derechos
El archivo assets/synthetic/scenario.json contiene 100 filas inventadas por fórmulas propias, fechas fijas ficticias, licencia CC0, sin fuentes de mercado ni eventos históricos asociados. La red y su inferencia son reales; los datos y el escenario NO son reales. Las bandas son ilustrativas: no existe calibración de cobertura en entradas sintéticas. Las métricas mostradas son históricas del checkpoint, no resultados del escenario.

Al abrir se consulta assets/data-policy.json. Como ninguna cesta completa está autorizada, no se consultan fuentes restringidas y se carga automáticamente la demostración sintética verificada mediante SHA256. Una futura cesta autorizada requiere live_input_url y complete_live_inputs_approved=true; si falla o está obsoleta, el estado pasa explícitamente a demostración sintética. Este cambio exige revisión de licencias y aprobación de publicación. Nunca se completa una cesta real con entradas sintéticas silenciosamente.

## Automatización y límites
El workflow verify_uax_public.yml está preparado para lunes–viernes y manual: valida únicamente el escenario propio, sin descargar finanzas, entrenar, exportar o escribir pesos, ni publicar commits. No altera las fechas ficticias para aparentar actualización. refresh_frozen.yml mantiene su guardia remota falsa. No hay actualización automática real de la cesta financiera completa; no se declara resuelto ese requisito.

## Reproducción
python -m scripts.build_uax_fixture
python -m pytest tests/test_uax_automatic.py tests/test_onnx_public_bundle.py -q
python -m http.server 8770 --directory web/pages

La prueba local de paridad carga los checkpoints originales desde el repositorio padre de este worktree, sin modificarlos; en CI se omite expresamente si no existen. La prueba pública de navegador se repetirá después de la aprobación, push y despliegue. La URL actualmente pública todavía corresponde a la versión anterior con carga privada.
