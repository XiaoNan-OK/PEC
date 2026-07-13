$ProgressPreference = 'SilentlyContinue'
$bibPath = 'D:\Lab\Quantum_Project_Exercise\Essay\Master Essay\HsiaoNanLu\reference.bib'
$refDir = 'D:\Lab\Quantum_Project_Exercise\Reference'
$headers = @{
  'User-Agent' = 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) Codex Reference Fetcher'
  'Accept' = 'application/pdf,text/html;q=0.9,*/*;q=0.8'
}

function Get-FieldValue([string]$body, [string]$name) {
  $pattern = $name + '\s*=\s*\{([\s\S]*?)\}\s*(,|$)'
  $m = [regex]::Match($body, $pattern, [System.Text.RegularExpressions.RegexOptions]::IgnoreCase)
  if ($m.Success) { return ($m.Groups[1].Value -replace '\s+', ' ').Trim() }
  return ''
}

function Get-Entries {
  $text = Get-Content -LiteralPath $bibPath -Raw
  $matches = [regex]::Matches($text, '@([A-Za-z]+)\s*\{\s*([^,]+),([\s\S]*?)\n\}', [System.Text.RegularExpressions.RegexOptions]::Singleline)
  $list = @()
  foreach ($m in $matches) {
    $type = $m.Groups[1].Value.ToLower()
    if ($type -eq 'control') { continue }
    $key = $m.Groups[2].Value.Trim()
    $body = $m.Groups[3].Value
    $title = (Get-FieldValue $body 'title') -replace '\{([^{}]+)\}', '$1'
    $doi = Get-FieldValue $body 'doi'
    if (-not $doi) {
      $note = Get-FieldValue $body 'note'
      $dm = [regex]::Match($note, 'doi:\s*\\href\{[^}]+\}\{([^}]+)\}', [System.Text.RegularExpressions.RegexOptions]::IgnoreCase)
      if ($dm.Success) { $doi = $dm.Groups[1].Value.Trim() }
    }
    $url = Get-FieldValue $body 'url'
    if (-not $url) {
      $note = Get-FieldValue $body 'note'
      $um = [regex]::Match($note, '\\href\{([^}]+)\}\{[^}]+\}', [System.Text.RegularExpressions.RegexOptions]::IgnoreCase)
      if ($um.Success) { $url = $um.Groups[1].Value.Trim() }
    }
    $list += [pscustomobject]@{
      Key = $key
      Type = $type
      Title = $title.Trim()
      Doi = $doi.Trim()
      Url = $url.Trim()
    }
  }
  return $list
}

function Sanitize-Title([string]$title) {
  if (-not $title) { return '' }
  return (($title -replace '[\\/:*?"<>|]', '') -replace '\s+', ' ').Trim()
}

function Get-DoiTail([string]$doi) {
  if (-not $doi) { return '' }
  $clean = $doi -replace '^https?://doi\.org/', ''
  $parts = $clean.Split('/')
  if ($parts.Count -le 1) { return $clean }
  $tail = ($parts[1..($parts.Count - 1)] -join '/')
  return ($tail -replace '[\\/:*?"<>|]', '_').Trim()
}

function Get-OutputFile([object]$entry) {
  $tail = Get-DoiTail $entry.Doi
  $title = Sanitize-Title $entry.Title
  if ($tail) { return Join-Path $refDir ($tail + ' ' + $title + '.pdf') }
  return Join-Path $refDir ($entry.Key + ' ' + $title + '.pdf')
}

function Get-ArxivPdf([string]$text) {
  $m = [regex]::Match($text, 'arxiv(?:\.org/abs/|:)(\d{4}\.\d{5})(v\d+)?', [System.Text.RegularExpressions.RegexOptions]::IgnoreCase)
  if ($m.Success) { return 'https://arxiv.org/pdf/' + $m.Groups[1].Value + '.pdf' }
  return $null
}

function Get-Candidates([object]$entry) {
  $urls = New-Object System.Collections.Generic.List[string]
  $doi = $entry.Doi
  if ($doi) {
    if ($doi -match '^10\.1103/(PhysRevLett|PhysRevA|PhysRevX|PRXQuantum|RevModPhys)\.(.+)$') {
      $map = @{
        PhysRevLett = 'prl'
        PhysRevA = 'pra'
        PhysRevX = 'prx'
        PRXQuantum = 'prxquantum'
        RevModPhys = 'rmp'
      }
      $urls.Add('https://journals.aps.org/' + $map[$matches[1]] + '/pdf/' + $doi)
    }
    if ($doi -match '^10\.22331/(q-[\d-]+-\d+)$') {
      $urls.Add('https://quantum-journal.org/papers/' + $matches[1] + '/pdf/')
    }
    if ($doi -like '10.1126/*') {
      $urls.Add('https://www.science.org/doi/pdf/' + $doi)
    }
    if ($doi -match '^10\.(1038|1007)/(.+)$') {
      $suffix = $matches[2]
      $urls.Add('https://www.nature.com/articles/' + $suffix + '.pdf')
      $urls.Add('https://www.nature.com/articles/' + $suffix)
      $urls.Add('https://link.springer.com/content/pdf/' + $doi + '.pdf')
      $urls.Add('https://link.springer.com/article/' + $doi)
    }
    $urls.Add('https://doi.org/' + $doi)
  }
  $arxiv = Get-ArxivPdf ($entry.Url + ' ' + $entry.Doi)
  if ($arxiv) { $urls.Add($arxiv) }
  if ($entry.Url) { $urls.Add($entry.Url) }
  return ($urls | Where-Object { $_ } | Select-Object -Unique)
}

function Save-FromUrl([string]$url, [string]$outFile, [int]$depth = 0) {
  if ($depth -gt 2) { return $false }
  $tmp = Join-Path $env:TEMP ([guid]::NewGuid().ToString() + '.bin')
  try {
    Invoke-WebRequest -Uri $url -OutFile $tmp -Headers $headers -MaximumRedirection 5 -ErrorAction Stop | Out-Null
    $bytes = [System.IO.File]::ReadAllBytes($tmp)
    if ($bytes.Length -ge 4 -and $bytes[0] -eq 37 -and $bytes[1] -eq 80 -and $bytes[2] -eq 68 -and $bytes[3] -eq 70) {
      Move-Item -LiteralPath $tmp -Destination $outFile -Force
      return $true
    }
    $html = ''
    try { $html = [System.Text.Encoding]::UTF8.GetString($bytes) } catch {}
    $patterns = @(
      'https?://[^"''<>\s]+\.pdf(?:\?[^"''<>\s]*)?',
      'href=["'']([^"'']+\.pdf(?:\?[^"'']*)?)["'']',
      'citation_pdf_url["''][^>]*content=["'']([^"'']+)["'']',
      'name=["'']citation_pdf_url["''][^>]*content=["'']([^"'']+)["'']'
    )
    foreach ($p in $patterns) {
      $m = [regex]::Match($html, $p, [System.Text.RegularExpressions.RegexOptions]::IgnoreCase)
      if ($m.Success) {
        $candidate = if ($m.Groups.Count -gt 1 -and $m.Groups[1].Value) { $m.Groups[1].Value } else { $m.Value }
        $candidate = $candidate -replace '^href=["'']', '' -replace '["'']$', ''
        try {
          $resolved = [System.Uri]::new([System.Uri]$url, $candidate).AbsoluteUri
        } catch {
          $resolved = $candidate
        }
        if (Save-FromUrl $resolved $outFile ($depth + 1)) { return $true }
      }
    }
  } catch {
  } finally {
    if (Test-Path -LiteralPath $tmp) { Remove-Item -LiteralPath $tmp -Force -ErrorAction SilentlyContinue }
  }
  return $false
}

$entries = Get-Entries
$results = @()
$testFile = Join-Path $refDir 'TEST Quantum Computing in the NISQ era and beyond.pdf'
if (Test-Path -LiteralPath $testFile) { Remove-Item -LiteralPath $testFile -Force }

foreach ($entry in $entries) {
  $outFile = Get-OutputFile $entry
  if (Test-Path -LiteralPath $outFile) {
    $results += [pscustomobject]@{
      Key = $entry.Key
      Status = 'exists'
      File = [System.IO.Path]::GetFileName($outFile)
      Source = ''
    }
    continue
  }

  $ok = $false
  $source = ''
  foreach ($candidate in (Get-Candidates $entry)) {
    if (Save-FromUrl $candidate $outFile 0) {
      $ok = $true
      $source = $candidate
      break
    }
  }

  $results += [pscustomobject]@{
    Key = $entry.Key
    Status = $(if ($ok) { 'downloaded' } else { 'failed' })
    File = [System.IO.Path]::GetFileName($outFile)
    Source = $source
  }
}

$summary = [pscustomobject]@{
  Total = $results.Count
  Downloaded = ($results | Where-Object Status -eq 'downloaded').Count
  Exists = ($results | Where-Object Status -eq 'exists').Count
  Failed = ($results | Where-Object Status -eq 'failed').Count
}

$reportPath = Join-Path $refDir '_download_summary.json'
[pscustomobject]@{
  Summary = $summary
  Results = $results
} | ConvertTo-Json -Depth 4 | Set-Content -LiteralPath $reportPath -Encoding UTF8

$summary | ConvertTo-Json -Compress
'FAILED:'
$results | Where-Object Status -eq 'failed' | Select-Object Key, File
