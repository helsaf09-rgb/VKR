@echo off
setlocal

chcp 65001 >nul
set "PYTHONIOENCODING=utf-8"
set "PYTHONUTF8=1"

set "SCRIPT_DIR=%~dp0"
for %%I in ("%SCRIPT_DIR%..") do set "ROOT=%%~fI"
set "PY=%ROOT%\.venv\Scripts\python.exe"
set "APP=%ROOT%\src\ui\streamlit_app.py"
set "REQ=%ROOT%\requirements-dev.txt"
set "BOOTSTRAP_MODE="

if not exist "%APP%" (
  echo Streamlit app not found: "%APP%"
  exit /b 1
)

if not exist "%PY%" (
  where python >nul 2>nul
  if %ERRORLEVEL%==0 (
    set "BOOTSTRAP_MODE=python"
  ) else (
    where py >nul 2>nul
    if %ERRORLEVEL%==0 set "BOOTSTRAP_MODE=py"
  )

  if not defined BOOTSTRAP_MODE (
    echo Python was not found in PATH.
    echo Install Python 3.12+ and run this launcher again.
    exit /b 1
  )

  echo Creating local virtual environment...
  if /I "%BOOTSTRAP_MODE%"=="python" (
    python -m venv "%ROOT%\.venv"
  ) else (
    py -3.12 -m venv "%ROOT%\.venv"
  )
  if errorlevel 1 (
    echo Failed to create virtual environment.
    exit /b 1
  )

  echo Installing project dependencies...
  call "%PY%" -m pip install --upgrade pip
  if errorlevel 1 (
    echo Failed to upgrade pip.
    exit /b 1
  )

  call "%PY%" -m pip install -r "%REQ%"
  if errorlevel 1 (
    echo Failed to install requirements from "%REQ%".
    exit /b 1
  )
)

pushd "%ROOT%" || exit /b 1
"%PY%" -m streamlit run "%APP%" --server.address 127.0.0.1 --server.port 8501 %*
set "EXIT_CODE=%ERRORLEVEL%"
popd

exit /b %EXIT_CODE%
