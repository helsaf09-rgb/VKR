$ErrorActionPreference = "Stop"

$root = Split-Path -Parent $PSScriptRoot
$python = Join-Path $root ".venv\Scripts\python.exe"
$script = Join-Path $PSScriptRoot "build_thesis_docx.py"
$figureScript = Join-Path $PSScriptRoot "generate_thesis_figures.py"
$exportScript = Join-Path $PSScriptRoot "export_docx_via_word.ps1"
$outputDocx = Join-Path $root "deliverables\VKR_draft_gost.docx"
$outputPdf = Join-Path $root "deliverables\VKR_draft_gost.pdf"

if (-not (Test-Path $python)) {
    throw "Python virtual environment was not found at $python"
}

& $python $figureScript
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

& $python $script --auto-figure-images --output $outputDocx
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

& $exportScript -InputDocx $outputDocx -OutputPdf $outputPdf
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
