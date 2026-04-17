param(
  [int]$Users = 500,
  [int]$AvgTransactions = 120,
  [int]$TopK = 5,
  [int]$Seed = 42
)

$root = Split-Path -Parent $PSScriptRoot
$py = Join-Path $root ".venv\Scripts\python.exe"

if (-not (Test-Path $py)) {
  throw "Python virtual environment was not found at $py"
}

Push-Location $root

& $py -m src.pipelines.run_baseline_pipeline `
  --n-users $Users `
  --avg-transactions $AvgTransactions `
  --top-k $TopK `
  --seed $Seed

$code = $LASTEXITCODE
Pop-Location

if ($code -ne 0) { exit $code }
