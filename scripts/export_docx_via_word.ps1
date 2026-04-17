param(
    [Parameter(Mandatory = $true)]
    [string]$InputDocx,

    [Parameter(Mandatory = $false)]
    [string]$OutputPdf
)

$ErrorActionPreference = "Stop"

$inputPath = (Resolve-Path $InputDocx).Path

if (-not $OutputPdf) {
    $OutputPdf = [System.IO.Path]::ChangeExtension($inputPath, ".pdf")
}

$outputPath = [System.IO.Path]::GetFullPath($OutputPdf)
$outputDir = Split-Path -Parent $outputPath

if (-not (Test-Path $outputDir)) {
    New-Item -ItemType Directory -Path $outputDir | Out-Null
}

$wdExportFormatPDF = 17
$wdExportOptimizeForPrint = 0
$wdExportAllDocument = 0
$wdExportDocumentContent = 0
$wdExportCreateHeadingBookmarks = 1

$word = $null
$doc = $null

try {
    $word = New-Object -ComObject Word.Application
    $word.Visible = $false
    $word.DisplayAlerts = 0

    $doc = $word.Documents.Open($inputPath, $false, $false)

    foreach ($toc in @($doc.TablesOfContents)) {
        $toc.Update()
    }

    $null = $doc.Fields.Update()

    foreach ($section in @($doc.Sections)) {
        foreach ($headerFooter in @($section.Headers) + @($section.Footers)) {
            $null = $headerFooter.Range.Fields.Update()
        }
    }

    $doc.Repaginate()

    foreach ($toc in @($doc.TablesOfContents)) {
        $toc.Update()
    }

    $doc.Save()
    $doc.ExportAsFixedFormat(
        $outputPath,
        $wdExportFormatPDF,
        $false,
        $wdExportOptimizeForPrint,
        $wdExportAllDocument,
        1,
        1,
        $wdExportDocumentContent,
        $true,
        $true,
        $wdExportCreateHeadingBookmarks,
        $true,
        $true,
        $false
    )

    Write-Output "[ok] exported via Word: $outputPath"
}
finally {
    if ($doc -ne $null) {
        $doc.Close($false)
    }
    if ($word -ne $null) {
        $word.Quit()
    }
}
