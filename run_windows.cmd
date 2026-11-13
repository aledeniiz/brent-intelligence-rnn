@echo off
setlocal
cd /d "%~dp0"
echo ===================================================
echo Brent Intelligence - instalacion, train y dashboard
echo ===================================================
where py >nul 2>nul
if errorlevel 1 (
  echo [ERROR] Falta Python Launcher con Python 3.12.
  goto fail
)
if not exist ".venv\Scripts\python.exe" (
  py -3.12 -m venv .venv
  if errorlevel 1 goto fail
)
set "PY=.venv\Scripts\python.exe"
"%PY%" -m pip install --upgrade pip
if errorlevel 1 goto fail
"%PY%" -m pip install -e ".[test]"
if errorlevel 1 goto fail
"%PY%" -m pytest -q
if errorlevel 1 goto fail
"%PY%" -m scripts.check_environment
if errorlevel 1 goto fail
"%PY%" -m scripts.download_data
if errorlevel 1 goto fail
"%PY%" -m scripts.audit_data
if errorlevel 1 goto fail
"%PY%" -m scripts.train_models --dry-run
if errorlevel 1 goto fail
"%PY%" -m scripts.train_models
if errorlevel 1 goto fail
"%PY%" -m scripts.predict
if errorlevel 1 goto fail
"%PY%" -m scripts.export_results
if errorlevel 1 goto fail
"%PY%" -m scripts.build_presentation
if errorlevel 1 goto fail
echo [OK] Abriendo Streamlit.
"%PY%" -m streamlit run app.py
if errorlevel 1 goto fail
exit /b 0
:fail
echo [ERROR] Comprueba el mensaje anterior.
pause
exit /b 1
