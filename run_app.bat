@echo off
setlocal
cd /d %~dp0

set "PYTHON_EXE=%~dp0.venv\Scripts\python.exe"
if not exist "%PYTHON_EXE%" (
    echo .venv not found. Please create it in this folder.
    pause
    exit /b 1
)

"%PYTHON_EXE%" -m app
if errorlevel 1 (
    echo.
    echo App exited with an error.
    pause
    exit /b 1
)
