param(
  [string]$Dataset = 'online_retail',
  [int]$TopK = 10,
  [int]$MinRating = 4,
  [int]$MinUserInteractions = 5,
  [int]$MinItemInteractions = 10,
  [int]$Factors = 32,
  [int]$Neighbors = 50,
  [int]$MaxUsers = 3000,
  [int]$MaxItems = 1500
)

$root = Split-Path -Parent $PSScriptRoot
$py = Join-Path $root ".venv\Scripts\python.exe"

if (-not (Test-Path $py)) {
  throw "Python virtual environment was not found at $py"
}

Push-Location $root

& $py -m src.pipelines.run_real_dataset_validation --dataset $Dataset --top-k $TopK --min-rating $MinRating --min-user-interactions $MinUserInteractions --min-item-interactions $MinItemInteractions --n-factors $Factors --n-neighbors $Neighbors --max-users $MaxUsers --max-items $MaxItems
if ($LASTEXITCODE -ne 0) { $code = $LASTEXITCODE; Pop-Location; exit $code }

Pop-Location

Write-Host '[ok] real-data validation completed'
