$root = Split-Path -Parent $PSScriptRoot
$py = Join-Path $root ".venv\Scripts\python.exe"
$bestConfig = Join-Path $root "reports\time_decay_best_config.txt"

if (-not (Test-Path $py)) {
  throw "Python virtual environment was not found at $py"
}

Push-Location $root

& $py -m src.pipelines.run_time_decay_sweep --top-k 5
if ($LASTEXITCODE -ne 0) { $code = $LASTEXITCODE; Pop-Location; exit $code }

Get-Content $bestConfig

Pop-Location

Write-Host '[ok] advanced search completed'
