$root = Split-Path -Parent $PSScriptRoot
$py = Join-Path $root ".venv\Scripts\python.exe"

if (-not (Test-Path $py)) {
  throw "Python virtual environment was not found at $py"
}

Push-Location $root
& $py -m src.pipelines.run_multiseed_benchmark --n-users 800 --avg-transactions 140 --top-k 5 --seeds 7,13,21,42,77 --n-bootstrap 5000
$code = $LASTEXITCODE
Pop-Location

if ($code -ne 0) { exit $code }
