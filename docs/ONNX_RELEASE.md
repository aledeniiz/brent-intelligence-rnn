# Release ONNX preparada, no publicada

Modelo: LSTM histórica6499, evaluación2021 y calibración2022–2023. ONNX/exportación no entrenan. Los checkpointsPyTorch originales no se distribuyen en estecommit; elONNX contiene sus mismos parámetros numéricos.

web/pages contiene únicamente frontend,ONNX,manifiesto de escalador/calibración,métricas propias,licencias yruntime. No contiene lasventanas financieras,elhistórico,JSONdemo oclaves. Para probarinferenciareal,cargar unJSONpreparado privado con el selector: seprocesa localmente sin subirse. El error inicial por ausencia de entradas públicas es deliberado; nunca se presentan números inventados. Las pruebas de paridad local utilizaron4ejemplosrealesyconfirmaron errores<0,001USD; no son datos publicables.

ONNXRuntimeWeb1.31.1: binarios e integridad verificados desdeNPM. LICENSE/ThirdPartyNotices oficiales de ONNXRuntime1.31.1 incluidos; procedencia exacta documentada en LICENSES_AND_MODEL.md. Modelo entrenado localmente en esteproyecto y distribución autorizada por su propietario; no es unmodelo preentrenado de terceros. No se licencia nuevamente el dataset de entrenamiento.

Primer despliegue: sólo código yassets; workflow deploy_onnx_frontend requiere ejecución manual autorizada yPages habilitado. El refresco financiero conserva la guardiaBRENT_DATA_PUBLICATION_APPROVED; mantenerfalse hastaacreditar derechos de todaslasfuentesy datos derivados. Noactivarlo por tener licenciaMIT del software.

Activar/publicar requiere aprobación definitiva: no se ha hecho push ni cambiadoconfiguración remota. Después comprobarHTTPSenunanavegación nueva yparidad con archivo privadoantes de poner laURLenAbout/Website delrepositorio. No usarunaURLprevista comosiyaestuvieraverificada.

Python3.12: pip install -e '.[test,onnx]'. scripts.prepare_pages_data prepara entrada960float32 sincargarpesos; requiere fuentesválidas,modelmanifest yderechos si se pretende publicar. scripts.export_uax_onnx requiere los artefactos privados originales fuera deGit. No cambiar16variables/orden ocalibración.
