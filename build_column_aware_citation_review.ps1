$sourceCsv = 'D:\Lab\Quantum_Project_Exercise\Essay\Master Essay\HsiaoNanLu\citation_review.csv'
$outCsv = 'D:\Lab\Quantum_Project_Exercise\Essay\Master Essay\HsiaoNanLu\citation_review_column_aware.csv'
$outXlsx = 'D:\Lab\Quantum_Project_Exercise\Essay\Master Essay\HsiaoNanLu\citation_review_column_aware.xlsx'

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

function Get-OverlapScore {
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

function Get-ColumnClusters {
    param(
        [double[]]$Positions,
        [double]$PageWidth
    )

    $sorted = $Positions | Sort-Object
    if (-not $sorted -or $sorted.Count -eq 0) {
        return @()
    }

    $gap = [Math]::Max(45.0, $PageWidth * 0.14)
    $clusters = @()
    $current = New-Object System.Collections.ArrayList
    [void]$current.Add($sorted[0])

    for ($i = 1; $i -lt $sorted.Count; $i++) {
        if (($sorted[$i] - $sorted[$i - 1]) -gt $gap) {
            $clusters += ,@($current.ToArray())
            $current = New-Object System.Collections.ArrayList
        }
        [void]$current.Add($sorted[$i])
    }
    $clusters += ,@($current.ToArray())

    $centers = foreach ($cluster in $clusters) {
        ($cluster | Measure-Object -Average).Average
    }
    return $centers
}

function Parse-BBoxDocument {
    param([string]$PdfPath)

    $xmlText = & pdftotext -f 1 -l 3 -bbox-layout $PdfPath - 2>$null | Out-String
    if (-not $xmlText) { return @() }

    $xmlText = $xmlText -replace '<!DOCTYPE[^>]*>', ''
    try {
        [xml]$xml = $xmlText
    } catch {
        return @()
    }

    $pages = @()
    foreach ($page in $xml.html.body.doc.page) {
        $pageWidth = [double]$page.width
        $blocks = @()
        foreach ($block in $page.flow.block) {
            $lines = @()
            foreach ($line in $block.line) {
                $words = @()
                foreach ($word in $line.word) {
                    if ($word.'#text') {
                        $words += [string]$word.'#text'
                    }
                }
                $lineText = ($words -join ' ').Trim()
                if ($lineText) {
                    $lines += $lineText
                }
            }
            $text = ($lines -join ' ').Trim()
            if (-not $text) { continue }

            $xMin = [double]$block.xMin
            $xMax = [double]$block.xMax
            $yMin = [double]$block.yMin
            $yMax = [double]$block.yMax
            $width = $xMax - $xMin

            $blocks += [pscustomobject]@{
                Text = $text
                XMin = $xMin
                XMax = $xMax
                YMin = $yMin
                YMax = $yMax
                Width = $width
            }
        }

        $bodyCandidates = $blocks | Where-Object {
            $_.Text.Length -gt 50 -and $_.Width -lt ($pageWidth * 0.75)
        }
        if (-not $bodyCandidates) {
            $bodyCandidates = $blocks | Where-Object { $_.Text.Length -gt 50 }
        }

        $columnCenters = Get-ColumnClusters -Positions ($bodyCandidates | ForEach-Object { $_.XMin }) -PageWidth $pageWidth
        $columnCount = [Math]::Min([Math]::Max($columnCenters.Count, 1), 3)

        foreach ($block in $blocks) {
            $columnIndex = 1
            if ($columnCenters.Count -gt 0 -and $block.Width -lt ($pageWidth * 0.80)) {
                $nearest = 0
                $bestDist = [double]::PositiveInfinity
                for ($i = 0; $i -lt $columnCenters.Count; $i++) {
                    $dist = [Math]::Abs($block.XMin - $columnCenters[$i])
                    if ($dist -lt $bestDist) {
                        $bestDist = $dist
                        $nearest = $i
                    }
                }
                $columnIndex = $nearest + 1
            }
            $block | Add-Member -NotePropertyName ColumnIndex -NotePropertyValue $columnIndex -Force
        }

        $orderedBlocks = $blocks | Sort-Object ColumnIndex, YMin
        $pages += [pscustomobject]@{
            Width = $pageWidth
            ColumnCount = $columnCount
            Blocks = $orderedBlocks
        }
    }

    return $pages
}

function Get-BestBlockSupport {
    param(
        [object[]]$Pages,
        [string[]]$Keywords
    )

    $best = [ordered]@{
        ColumnCount = ''
        Section = ''
        Location = ''
        Excerpt = ''
        Score = -1
    }

    for ($p = 0; $p -lt $Pages.Count; $p++) {
        $page = $Pages[$p]
        $blocks = $page.Blocks | Where-Object { $_.Text.Length -gt 50 }
        $blockNum = 0
        foreach ($block in $blocks) {
            $blockNum++
            $score = Get-OverlapScore -Text $block.Text -Keywords $Keywords
            if ($block.Text -match 'noise|error|mitigation|readout|qubit|circuit|sampling|variance|NISQ|quantum') {
                $score += 1
            }
            if ($score -gt $best.Score) {
                $section = 'Opening pages'
                if ($block.Text -match 'Abstract') {
                    $section = 'Abstract'
                } elseif ($block.Text -match 'Introduction') {
                    $section = 'Introduction'
                }
                $excerpt = $block.Text
                $sentences = $block.Text -split '(?<=[\.\!\?])\s+'
                if ($sentences.Count -gt 1) {
                    $bestSentence = $sentences[0]
                    $bestSentenceScore = -1
                    foreach ($sentence in $sentences) {
                        $sScore = Get-OverlapScore -Text $sentence -Keywords $Keywords
                        if ($sentence -match 'noise|error|mitigation|readout|qubit|circuit|sampling|variance|NISQ|quantum') {
                            $sScore += 1
                        }
                        if ($sScore -gt $bestSentenceScore) {
                            $bestSentenceScore = $sScore
                            $bestSentence = $sentence.Trim()
                        }
                    }
                    $excerpt = $bestSentence
                }
                $best.ColumnCount = $page.ColumnCount
                $best.Section = $section
                $best.Location = "Page $($p + 1), Column $($block.ColumnIndex), Block $blockNum"
                $best.Excerpt = $excerpt
                $best.Score = $score
            }
        }
    }

    return $best
}

$rows = Import-Csv -LiteralPath $sourceCsv

foreach ($row in $rows) {
    if (-not $row.PDFPath -or -not (Test-Path -LiteralPath $row.PDFPath)) {
        continue
    }

    $keywords = @()
    $keywords += Get-WordList $row.PaperTitle
    $keywords += Get-WordList $row.ThesisCitingParagraph
    $keywords += Get-WordList $row.RelationToCitingParagraph
    $keywords = $keywords | Select-Object -Unique

    $pages = Parse-BBoxDocument -PdfPath $row.PDFPath
    if (-not $pages -or $pages.Count -eq 0) {
        continue
    }

    $support = Get-BestBlockSupport -Pages $pages -Keywords $keywords

    $row | Add-Member -NotePropertyName DetectedColumnCount -NotePropertyValue $support.ColumnCount -Force
    $row | Add-Member -NotePropertyName RelevantColumnAwareSection -NotePropertyValue $support.Section -Force
    $row | Add-Member -NotePropertyName RelevantColumnAwareLocation -NotePropertyValue $support.Location -Force
    $row | Add-Member -NotePropertyName RelevantColumnAwareExcerpt -NotePropertyValue $support.Excerpt -Force
}

$rows | Export-Csv -LiteralPath $outCsv -NoTypeInformation -Encoding UTF8

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
    'RelationToCitingParagraph',
    'DetectedColumnCount',
    'RelevantColumnAwareSection',
    'RelevantColumnAwareLocation',
    'RelevantColumnAwareExcerpt',
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
$sheet.Columns.Item(4).ColumnWidth = 42
$sheet.Columns.Item(5).ColumnWidth = 12
$sheet.Columns.Item(6).ColumnWidth = 18
$sheet.Columns.Item(7).ColumnWidth = 24
$sheet.Columns.Item(8).ColumnWidth = 90
$sheet.Columns.Item(9).ColumnWidth = 55
$sheet.Rows.Item('1:1').Font.Bold = $true
$sheet.Rows.Item('1:1').Interior.ColorIndex = 15
$sheet.Application.ActiveWindow.SplitRow = 1
$sheet.Application.ActiveWindow.FreezePanes = $true

$workbook.SaveAs($outXlsx, 51)
$workbook.Close($false)
$excel.Quit()

[System.Runtime.InteropServices.Marshal]::ReleaseComObject($used) | Out-Null
[System.Runtime.InteropServices.Marshal]::ReleaseComObject($sheet) | Out-Null
[System.Runtime.InteropServices.Marshal]::ReleaseComObject($workbook) | Out-Null
[System.Runtime.InteropServices.Marshal]::ReleaseComObject($excel) | Out-Null

[System.GC]::Collect()
[System.GC]::WaitForPendingFinalizers()

Write-Output $outCsv
Write-Output $outXlsx
