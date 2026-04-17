$ErrorActionPreference = "Stop"

$root = Split-Path -Parent $PSScriptRoot
$python = Join-Path $root ".venv\Scripts\python.exe"
$script = Join-Path $PSScriptRoot "build_thesis_docx.py"
$figureScript = Join-Path $PSScriptRoot "generate_thesis_figures.py"
$exportScript = Join-Path $PSScriptRoot "export_docx_via_word.ps1"
$source = Join-Path $root "docs\24_thesis_final_assembled_en.md"
$output = Join-Path $root "deliverables\VKR_draft_gost_en.docx"
$outputPdf = Join-Path $root "deliverables\VKR_draft_gost_en.pdf"

if (-not (Test-Path $python)) {
    throw "Python virtual environment was not found at $python"
}

& $python $figureScript
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

& $python $script `
    --source $source `
    --output $output `
    --title "Personalization of Customer Offers Based on Banking Services User Transaction Activity" `
    --language en `
    --figure-prefix "Figure" `
    --student-name "Safonova Elena Mikhailovna" `
    --supervisor-name "Ilvovsky Dmitry Alekseevich" `
    --degree-program "01.04.02 Applied Mathematics and Informatics" `
    --auto-figure-images
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

& $exportScript -InputDocx $output -OutputPdf $outputPdf
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
