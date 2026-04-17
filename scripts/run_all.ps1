param(
  [int]$Users = 800,
  [int]$AvgTransactions = 140,
  [int]$TopK = 5,
  [int]$Seed = 42,
  [int]$Factors = 10
)

$root = Split-Path -Parent $PSScriptRoot
$py = Join-Path $root ".venv\Scripts\python.exe"
$notebook = Join-Path $root "notebooks\01_eda_transactions.ipynb"

if (-not (Test-Path $py)) {
  throw "Python virtual environment was not found at $py"
}

Push-Location $root

& $py -m src.pipelines.run_baseline_pipeline --n-users $Users --avg-transactions $AvgTransactions --top-k $TopK --seed $Seed
if ($LASTEXITCODE -ne 0) { $code = $LASTEXITCODE; Pop-Location; exit $code }

& $py -m src.pipelines.run_eda_report
if ($LASTEXITCODE -ne 0) { $code = $LASTEXITCODE; Pop-Location; exit $code }

& $py -m src.pipelines.run_mf_baseline --n-factors $Factors --top-k $TopK --seed $Seed
if ($LASTEXITCODE -ne 0) { $code = $LASTEXITCODE; Pop-Location; exit $code }

& $py -m src.pipelines.run_ncf_baseline --top-k $TopK --seed $Seed
if ($LASTEXITCODE -ne 0) { $code = $LASTEXITCODE; Pop-Location; exit $code }

& $py -m src.pipelines.run_lightgcn_baseline --top-k $TopK --seed $Seed --embedding-dim 24 --n-layers 2 --learning-rate 0.03 --epochs 25 --samples-per-epoch 30000 --batch-size 4096
if ($LASTEXITCODE -ne 0) { $code = $LASTEXITCODE; Pop-Location; exit $code }

& $py -m src.pipelines.run_item_knn_baseline --n-neighbors 10 --top-k $TopK
if ($LASTEXITCODE -ne 0) { $code = $LASTEXITCODE; Pop-Location; exit $code }

& $py -m src.pipelines.run_hybrid_baseline --top-k $TopK --profile-weight 0.7 --semantic-weight 0.3
if ($LASTEXITCODE -ne 0) { $code = $LASTEXITCODE; Pop-Location; exit $code }

& $py -m src.pipelines.run_time_decay_model --top-k $TopK --decay-rate 0.01 --short-term-days 30 --short-term-weight 0.2 --spend-weight 0.6 --freq-weight 0.4
if ($LASTEXITCODE -ne 0) { $code = $LASTEXITCODE; Pop-Location; exit $code }

& $py -m src.pipelines.generate_service_demo_output --top-k $TopK --n-users 5
if ($LASTEXITCODE -ne 0) { $code = $LASTEXITCODE; Pop-Location; exit $code }

& $py -m src.pipelines.run_analysis_reports --n-bootstrap 3000 --seed $Seed
if ($LASTEXITCODE -ne 0) { $code = $LASTEXITCODE; Pop-Location; exit $code }

$oldPyWarnings = $env:PYTHONWARNINGS
$env:PYTHONWARNINGS = 'ignore:Proactor event loop does not implement add_reader family of methods required for zmq:RuntimeWarning'
& $py -m jupyter nbconvert --to notebook --execute $notebook --output '01_eda_transactions.ipynb' --output-dir (Split-Path $notebook -Parent)
if ($null -eq $oldPyWarnings) {
  Remove-Item Env:PYTHONWARNINGS -ErrorAction SilentlyContinue
} else {
  $env:PYTHONWARNINGS = $oldPyWarnings
}
if ($LASTEXITCODE -ne 0) { $code = $LASTEXITCODE; Pop-Location; exit $code }

Pop-Location

Write-Host '[ok] full refresh completed'
