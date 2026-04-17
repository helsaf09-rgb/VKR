@echo off
setlocal

set "ROOT=%~dp0"
set "LAUNCHER=%ROOT%scripts\run_ui.bat"

if not exist "%LAUNCHER%" (
  echo Launcher not found: "%LAUNCHER%"
  exit /b 1
)

call "%LAUNCHER%" %*
exit /b %ERRORLEVEL%
