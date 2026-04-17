$root = Split-Path -Parent $PSScriptRoot
$py = Join-Path $root ".venv\Scripts\python.exe"

if (-not (Test-Path $py)) {
  throw "Python virtual environment was not found at $py"
}

Push-Location $root
& $py -m uvicorn src.service.app:app --host 127.0.0.1 --port 8000 --reload
$code = $LASTEXITCODE
Pop-Location

if ($code -ne 0) { exit $code }
