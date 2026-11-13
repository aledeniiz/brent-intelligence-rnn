# Ultima fase en el ordenador: instrucciones de entrega

En Windows 11 (PowerShell):
1. Instalar Python 3.12 desde python.org y Git si faltan.
2. git clone https://github.com/aledeniiz/brent-intelligence-rnn.git
3. cd brent-intelligence-rnn
4. py -3.12 -m venv .venv
5. .\.venv\Scripts\python.exe -m pip install --upgrade pip
6. .\.venv\Scripts\python.exe -m pip install -e ".[test]"
7. .\.venv\Scripts\python.exe -m pytest -q
8. .\.venv\Scripts\python.exe -m scripts.download_data
9. .\.venv\Scripts\python.exe -m scripts.audit_data
10. .\.venv\Scripts\python.exe -m scripts.train_models --dry-run
11. .\.venv\Scripts\python.exe -m scripts.train_models
12. .\.venv\Scripts\python.exe -m scripts.predict
13. .\.venv\Scripts\python.exe -m streamlit run app.py

Para pruebas RAPIDAS antes del entrenamiento final:
.\.venv\Scripts\python.exe -m scripts.train_models --smoke --epochs 2 --hidden-size 8

Si se quiere intentar GPU NVIDIA, comprobar primero torch.cuda.is_available()
y ejecutar --device cuda. CPU es la opcion mas compatible.

El informe del test final solo puede elaborarse tras descargar datos reales
y completar el entrenamiento. No se proporcionan cifras de rendimiento ficticias.


14. Generar CSV de comparacion, notas y PPTX de cinco diapositivas:
    .\.venv\Scripts\python.exe -m scripts.export_results
    .\.venv\Scripts\python.exe -m scripts.build_presentation

Estos archivos se generan en reports/ y no se suben automaticamente a GitHub.
