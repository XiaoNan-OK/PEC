$csvPath = 'D:\Lab\Quantum_Project_Exercise\Python Code\PEC\citation_review.csv'
$xlsxPath = 'D:\Lab\Quantum_Project_Exercise\Essay\Master Essay\HsiaoNanLu\citation_review.xlsx'
$csvOutPath = 'D:\Lab\Quantum_Project_Exercise\Essay\Master Essay\HsiaoNanLu\citation_review.csv'

function Get-NormalizedText {
    param([string]$Text)
    if (-not $Text) { return '' }
    return (($Text.ToLower() -replace '\\[A-Za-z]+\{([^}]*)\}', '$1') -replace '[^a-z0-9]+', ' ').Trim()
}

function Get-OverlapScore {
    param(
        [string]$Paragraph,
        [string[]]$Keywords
    )
    $norm = Get-NormalizedText $Paragraph
    $score = 0
    foreach ($word in $Keywords) {
        if ($word.Length -gt 3 -and $norm.Contains($word)) {
            $score += 1
        }
    }
    if ($Paragraph -match 'Abstract|Introduction|noise|error|mitigation|quantum|readout|circuit|qubit') {
        $score += 1
    }
    return $score
}

$rows = Import-Csv -LiteralPath $csvPath

foreach ($row in $rows) {
    if (-not $row.PDFPath -or -not (Test-Path -LiteralPath $row.PDFPath)) {
        continue
    }

    $raw = & pdftotext -f 1 -l 2 -layout $row.PDFPath - 2>&1 | Out-String
    if (-not $raw) {
        continue
    }

    $paragraphs = ($raw -replace "`r", '') -split "`n\s*`n" |
        ForEach-Object { ($_ -replace '\s+', ' ').Trim() } |
        Where-Object { $_.Length -gt 80 } |
        Select-Object -First 18

    if (-not $paragraphs) {
        continue
    }

    $keywords = @(
        (Get-NormalizedText $row.PaperTitle),
        (Get-NormalizedText $row.ThesisCitingParagraph),
        (Get-NormalizedText $row.RelationToCitingParagraph)
    ) -join ' '
    $keywordList = $keywords.Split(' ', [System.StringSplitOptions]::RemoveEmptyEntries) | Select-Object -Unique

    $bestPara = $paragraphs[0]
    $bestScore = -1
    foreach ($para in $paragraphs) {
        $score = Get-OverlapScore -Paragraph $para -Keywords $keywordList
        if ($score -gt $bestScore) {
            $bestScore = $score
            $bestPara = $para
        }
    }

    $row.RelevantPassageInPaper = $bestPara
    if ($bestPara -match 'Abstract') {
        $row.RelevantPassageSection = 'Abstract'
    } elseif ($bestPara -match 'Introduction') {
        $row.RelevantPassageSection = 'Introduction / opening discussion'
    } else {
        $row.RelevantPassageSection = 'Opening section of the paper'
    }
}

$rows | Export-Csv -LiteralPath $csvOutPath -NoTypeInformation -Encoding UTF8

$excel = New-Object -ComObject Excel.Application
$excel.Visible = $false
$excel.DisplayAlerts = $false

$workbook = $excel.Workbooks.Add()
$sheet = $workbook.Worksheets.Item(1)
$sheet.Name = 'Citation Review'

$headers = @(
    'CitationKey',
    'PaperTitle',
    'ThesisCitationLocation',
    'ThesisCitingParagraph',
    'RelationToCitingParagraph',
    'RelevantPassageSection',
    'RelevantPassageInPaper',
    'PDFPath'
)

for ($c = 0; $c -lt $headers.Count; $c++) {
    $sheet.Cells.Item(1, $c + 1) = $headers[$c]
}

$r = 2
foreach ($row in $rows) {
    for ($c = 0; $c -lt $headers.Count; $c++) {
        $sheet.Cells.Item($r, $c + 1) = $row.($headers[$c])
    }
    $r++
}

$used = $sheet.UsedRange
$used.WrapText = $true
$used.VerticalAlignment = -4160
$used.Columns.AutoFit() | Out-Null
$sheet.Columns.Item(2).ColumnWidth = 42
$sheet.Columns.Item(4).ColumnWidth = 70
$sheet.Columns.Item(5).ColumnWidth = 42
$sheet.Columns.Item(6).ColumnWidth = 24
$sheet.Columns.Item(7).ColumnWidth = 95
$sheet.Columns.Item(8).ColumnWidth = 55
$sheet.Rows.Item('1:1').Font.Bold = $true
$sheet.Rows.Item('1:1').Interior.ColorIndex = 15
$sheet.Application.ActiveWindow.SplitRow = 1
$sheet.Application.ActiveWindow.FreezePanes = $true

$workbook.SaveAs($xlsxPath, 51)
$workbook.Close($false)
$excel.Quit()

[System.Runtime.InteropServices.Marshal]::ReleaseComObject($used) | Out-Null
[System.Runtime.InteropServices.Marshal]::ReleaseComObject($sheet) | Out-Null
[System.Runtime.InteropServices.Marshal]::ReleaseComObject($workbook) | Out-Null
[System.Runtime.InteropServices.Marshal]::ReleaseComObject($excel) | Out-Null

[System.GC]::Collect()
[System.GC]::WaitForPendingFinalizers()

Write-Output $csvOutPath
Write-Output $xlsxPath
