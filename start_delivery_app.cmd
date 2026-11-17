@echo off
setlocal
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
 echo Falta el entorno Python 3.12.
 pause
 exit /b 1
)
".venv\Scripts\python.exe" -m streamlit run app_delivery.py
if errorlevel 1 pause
