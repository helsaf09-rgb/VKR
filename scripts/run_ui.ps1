$root = Split-Path -Parent $PSScriptRoot
$py = Join-Path $root ".venv\Scripts\python.exe"
$app = Join-Path $root "src\ui\streamlit_app.py"

if (-not (Test-Path $py)) {
  throw "Python virtual environment was not found at $py"
}

Push-Location $root
& $py -m streamlit run $app --server.address 127.0.0.1 --server.port 8501
$code = $LASTEXITCODE
Pop-Location

if ($code -ne 0) { exit $code }
