param(
  [int]$TopK = 10,
  [int]$MinUserInteractions = 5,
  [int]$MinItemInteractions = 10,
  [int]$MaxUsers = 3000,
  [int]$MaxItems = 1500,
  [int]$EmbeddingDim = 32,
  [int]$NumHeads = 4,
  [int]$NumBlocks = 2,
  [int]$MaxSeqLen = 50,
  [int]$WindowStride = 1,
  [double]$Dropout = 0.1,
  [double]$LearningRate = 0.002,
  [double]$WeightDecay = 0.00001,
  [int]$Epochs = 8,
  [int]$BatchSize = 256,
  [int]$SamplesPerEpoch = 50000
)

$root = Split-Path -Parent $PSScriptRoot
$py = Join-Path $root ".venv\Scripts\python.exe"

if (-not (Test-Path $py)) {
  throw "Python virtual environment was not found at $py"
}

Push-Location $root

& $py -m src.pipelines.run_sasrec_real_validation `
  --top-k $TopK `
  --min-user-interactions $MinUserInteractions `
  --min-item-interactions $MinItemInteractions `
  --max-users $MaxUsers `
  --max-items $MaxItems `
  --embedding-dim $EmbeddingDim `
  --num-heads $NumHeads `
  --num-blocks $NumBlocks `
  --max-seq-len $MaxSeqLen `
  --window-stride $WindowStride `
  --dropout $Dropout `
  --learning-rate $LearningRate `
  --weight-decay $WeightDecay `
  --epochs $Epochs `
  --batch-size $BatchSize `
  --samples-per-epoch $SamplesPerEpoch
if ($LASTEXITCODE -ne 0) { $code = $LASTEXITCODE; Pop-Location; exit $code }

Pop-Location

Write-Host '[ok] sasrec real-data validation completed'
