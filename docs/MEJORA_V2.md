# Investigacion V2 — mejora del Brent sin manipular el test

## Diagnostico de la version original

La LSTM (arquitectura seleccionada por validacion) redujo el RMSE en test
historico un 22,48% a 252 sesiones, pero empeoro a 5 (1,73%) y a 63
(28,58%). La media de validacion de la LSTM fue 1,0422 en RMSE/ingenuo
(mayor que 1). Por tanto, no se ha demostrado superioridad general.

El test 2024-2026 YA SE HA CONSULTADO: no puede tratarse como conjunto
virgen para seleccionar las mejoras v2 ni atribuirle nueva significacion.

## Hipotesis (sin suponer resultados)

La cabeza multisalida de las tres redes actuales utiliza una sola secuencia
de estados para aprender retornos futuros en tres escalas temporales muy
distintas. Esto podria provocar interferencia entre horizontes.
La alternativa es entrenar una LSTM independiente por horizonte, con mas
ejemplos cuyos respectivos objetivos ya han madurado.
Una regresion Ridge regularizada sobre niveles, media y volatilidad de
las caracteristicas temporales sirve como referencia economica sencilla.

La investigacion V2 prueba ambos modelos con dos conjuntos de variables:
- brent_only: precio, retornos y volatilidad del Brent.
- all: Brent + variables macro, financieras y externos disponibles.

## Protocolo matematico

- MISMAS fechas de origen de validacion que la evaluacion multisalida
  antigua, limitando las observaciones a aquellas cuyo horizonte de 252
  sesiones ya habia madurado. Esto permite comparaciones historicas
  con cautela. No se leen etiquetas de 2022-2026.
- Entrenamiento externo por horizonte: todas las etiquetas del horizonte h
  que hayan madurado antes del corte del fold. No se aplican etiquetas
  futuras ni se requiere maduracion de 252 sesiones para entrenar a h=5.
- El 25% mas reciente de la historia de entrenamiento se reserva para
  validacion INTERNA (con embargo de maduracion): alphas de Ridge y epocas
  de LSTM se eligen aqui, nunca en validacion externa.
- Despues se reajusta cada candidato sobre el pasado maduro completo,
  empleando parametros elegidos en el entrenamiento interno.
- Escalado e imputacion se ajustan exclusivamente a los datos anteriores
  a cada corte de entrenamiento. No se lee el test, no se publica un modelo
  de produccion y no se cambia el sistema automatico existente.

Conservamos 5 folds historicos y 3 horizontes, comparando contra ingenuo
en RMSE y MAE en dolares. La seleccion definitiva debe basarse en validacion
y estabilidad: no conviene añadir complejidad si Ridge/ingenuo domina.

## Comandos

Primero prueba el protocolo de fechas, sin entrenar:

    python -m scripts.research_v2 --dry-run

La referencia Ridge es ligera y puede evaluarse en CPU:

    python -m scripts.research_v2 --model ridge

La LSTM por horizonte consume mas tiempo. Una unica prueba pequeña:

    python -m scripts.research_v2 --model lstm --features brent_only --horizons 5 --folds 1 --epochs 8

Experimento completo opcional, despues de revisar el anterior:

    python -m scripts.research_v2 --model both --epochs 12

Los informes se generan en reports/research_v2/*.json, no en artifacts,
deployment ni public. El codigo no contiene reentrenamiento automatico.

## Criterio para adoptar algo nuevo

1. Verificar que los resultados mejoran la validacion del horizonte y
   no dependen de un solo fold.
2. Comparar con naive y Ridge, no solo contra otras redes.
3. Documentar costo computacional, numeros de parametros y cobertura.
4. Estabilizar la configuracion y congelarla ANTES de cualquier evaluacion
   verdaderamente nueva, idealmente observaciones futuras que maduren.
5. No reemplazar la LSTM actualmente publicada sin comprobar primero la
   integridad y la version de sus artefactos.

El estudio es exploratorio y no debe presentarse como test independiente.
