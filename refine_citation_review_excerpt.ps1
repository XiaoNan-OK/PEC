$csvPath = 'D:\Lab\Quantum_Project_Exercise\Essay\Master Essay\HsiaoNanLu\citation_review.csv'
$xlsxPath = 'D:\Lab\Quantum_Project_Exercise\Essay\Master Essay\HsiaoNanLu\citation_review_precise.xlsx'
$csvOutPath = 'D:\Lab\Quantum_Project_Exercise\Essay\Master Essay\HsiaoNanLu\citation_review_precise.csv'

function Get-NormalizedText {
    param([string]$Text)
    if (-not $Text) { return '' }
    return (($Text.ToLower() -replace '\\[A-Za-z]+\{([^}]*)\}', '$1') -replace '[^a-z0-9]+', ' ').Trim()
}

function Get-WordList {
    param([string]$Text)
    return (Get-NormalizedText $Text).Split(' ', [System.StringSplitOptions]::RemoveEmptyEntries) |
        Where-Object { $_.Length -gt 3 } |
        Select-Object -Unique
}

function Get-Score {
    param(
        [string]$Text,
        [string[]]$Keywords
    )
    $norm = Get-NormalizedText $Text
    $score = 0
    foreach ($word in $Keywords) {
        if ($norm.Contains($word)) {
            $score += 1
        }
    }
    return $score
}

function Get-SectionLabel {
    param([string]$PageText)
    if ($PageText -match '(?im)^\s*abstract\s*$') { return 'Abstract' }
    if ($PageText -match '(?im)^\s*1\s+introduction\s*$') { return 'Introduction' }
    if ($PageText -match '(?im)^\s*introduction\s*$') { return 'Introduction' }
    return 'Opening pages'
}

function Get-BestSupport {
    param(
        [string]$PdfPath,
        [string[]]$Keywords
    )

    $best = [ordered]@{
        Location = ''
        Section = ''
        Excerpt = ''
        Score = -1
    }

    foreach ($page in 1..3) {
        $pageText = & pdftotext -f $page -l $page -layout $PdfPath - 2>$null | Out-String
        if (-not $pageText) { continue }

        $section = Get-SectionLabel $pageText
        $paragraphs = ($pageText -replace "`r", '') -split "`n\s*`n" |
            ForEach-Object { ($_ -replace '\s+', ' ').Trim() } |
            Where-Object { $_.Length -gt 50 }

        for ($i = 0; $i -lt $paragraphs.Count; $i++) {
            $para = $paragraphs[$i]
            $sentences = $para -split '(?<=[\.\!\?])\s+'
            if (-not $sentences -or $sentences.Count -eq 0) {
                $sentences = @($para)
            }

            $bestSentence = $para
            $bestSentenceScore = -1
            foreach ($sentence in $sentences) {
                $s = Get-Score -Text $sentence -Keywords $Keywords
                if ($sentence -match 'noise|error|mitigation|readout|qubit|circuit|sampling|variance|NISQ|quantum') {
                    $s += 1
                }
                if ($s -gt $bestSentenceScore) {
                    $bestSentenceScore = $s
                    $bestSentence = $sentence.Trim()
                }
            }

            $paraScore = Get-Score -Text $para -Keywords $Keywords
            $totalScore = $paraScore + $bestSentenceScore

            if ($totalScore -gt $best.Score) {
                $best.Location = "Page $page, Paragraph " + ($i + 1)
                $best.Section = $section
                $best.Excerpt = $bestSentence
                $best.Score = $totalScore
            }
        }
    }

    return $best
}

$rows = Import-Csv -LiteralPath $csvPath

foreach ($row in $rows) {
    if (-not $row.PDFPath -or -not (Test-Path -LiteralPath $row.PDFPath)) {
        continue
    }

    $keywords = @()
    $keywords += Get-WordList $row.PaperTitle
    $keywords += Get-WordList $row.ThesisCitingParagraph
    $keywords += Get-WordList $row.RelationToCitingParagraph
    $keywords = $keywords | Select-Object -Unique

    $support = Get-BestSupport -PdfPath $row.PDFPath -Keywords $keywords

    $row | Add-Member -NotePropertyName RelevantPreciseLocation -NotePropertyValue $support.Location -Force
    $row | Add-Member -NotePropertyName RelevantPreciseSection -NotePropertyValue $support.Section -Force
    $row | Add-Member -NotePropertyName RelevantPreciseExcerpt -NotePropertyValue $support.Excerpt -Force
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
    'RelevantPreciseSection',
    'RelevantPreciseLocation',
    'RelevantPreciseExcerpt',
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
$sheet.Columns.Item(6).ColumnWidth = 18
$sheet.Columns.Item(7).ColumnWidth = 18
$sheet.Columns.Item(8).ColumnWidth = 90
$sheet.Columns.Item(9).ColumnWidth = 55
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
