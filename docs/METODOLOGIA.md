# Protocolo experimental de Brent Intelligence

## Contrato predictivo
Objetivo: Brent spot EIA/FRED, en USD por barril. Entrada: las ultimas 60
sesiones realmente observadas del Brent y variables explicativas causales.
Salidas directas: retornos logaritmicos acumulados a 5, 63 y 252 sesiones,
reconstruidos como precios. Los horizontes no son dias naturales.

## Datos y sus limites
FRED proporciona Brent y varias variables externas; Yahoo agrega futuros y
mercados cuando estan disponibles. El Brent no se rellena con ffill.
Las variables externas se retrasan una sesion Brent como precaucion; no prueba
disponibilidad historica real ni controla revisiones de series. Una evaluacion
point-in-time exigiría vintages y registros de publicacion por origen.

## Walk-forward y maduracion
Se ejecutan cinco cortes crecientes con entrenamiento maduro hasta 2011,
2013, 2015, 2017 y 2019, y validacion hasta 2013, 2015, 2017, 2019 y
2021, respectivamente. Si la prediccion de 252 sesiones se conoce despues
del corte, NO se usa en train aunque el anclaje sea anterior.
La seleccion de arquitectura se realiza con el promedio de (RMSE/RMSE ingenuo)
por horizonte y fold. Todas comparten semilla, ventanas e hiperparametros.

## Bloques bloqueados
Los datos desde 2022 con objetivos madurados hasta fin de 2023 calibran
la incertidumbre. Los anclajes desde enero de 2024 hasta aquellos cuyo
horizonte de 252 sesiones ya haya madurado conforman test.
Los criterios de seleccion no leen las metricas de test.
Las redes se reentrenan para evaluacion solo con etiquetas maduras hasta 2021.

## Despliegue posterior a la evaluacion
Finalizada la evaluacion, un modelo de PRODUCCION independiente se entrena
con TODAS las etiquetas que han madurado hasta el ultimo Brent descargado.
Sus pesos y escalador se guardan separadamente, sin atribuirles el RMSE
del modelo historico. La aplicacion muestra el RMSE del estudio historico
como contexto, no como validacion independiente del modelo actualizado.

## Incertidumbre
Para la red seleccionada: cuantiles 80% de |log(precio_real / predicho)|
calculados en calibracion, aplicados de manera simetrica al logprecio.
Se informa de cobertura empirica posterior en test, no se garantiza una
cobertura exacta en todos los regímenes del futuro.
Los errores a horizontes largos se solapan y la muestra no es independiente.

## Integridad y repetibilidad
Cada ejecucion guarda SHA256 de CSV, configuracion, semilla, modelos, escaladores
y resultados en un directorio versionado. El puntero active.json se publica
SOLO al terminar toda la ejecucion. El modo --smoke no modifica ese puntero.
No deben citarse resultados de datos sinteticos como historicos del Brent.
