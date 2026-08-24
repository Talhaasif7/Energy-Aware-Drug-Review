<#
=============================================================================
 make_handoff.ps1  —  Build the single zip to send to the professor.

     powershell -ExecutionPolicy Bypass -File .\make_handoff.ps1

 Produces:  handoff_to_professor_<stamp>.zip   (in the repo root)

 WHY AN ALLOW-LIST AND NOT A FOLDER ZIP
 The repo holds ~9,900 files, almost all of them raw CADEC .ann annotations that
 the CPU pipeline never reads (it reads the harmonised CSVs). Zipping the folder
 would produce a package many times larger than needed and would also ship the
 derived results/*.json from earlier Windows runs — see the next note, which is
 the more important of the two reasons.

 WHY THE DERIVED JSONs ARE DELIBERATELY EXCLUDED
 results/frozen_split_reconciled.json and friends already exist here from the
 Windows runs. If they travelled with the package and a step failed on the
 professor's machine, the collector would find the old file sitting in results/
 and return it as this run's output — a Windows software estimate presented as a
 Linux sensor measurement. Leaving them out means any artifact that comes back
 must have been produced there. run_on_linux.sh also timestamp-checks each one,
 so the guarantee holds even if a stale file appears some other way.
=============================================================================
#>

[CmdletBinding()]
param(
    # Skip the SHA-256 verification of the input corpora (not recommended).
    [switch]$SkipHashCheck,
    # Keep the staging folder after zipping, to inspect what was packaged.
    [switch]$KeepStaging
)

$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest

$Root  = $PSScriptRoot
$Stamp = (Get-Date).ToUniversalTime().ToString('yyyyMMddTHHmmssZ')
$StageParent = Join-Path $Root 'handoff_package'
$Stage = Join-Path $StageParent 'energy-aware-drug-review'
$Zip   = Join-Path $Root "handoff_to_professor_$Stamp.zip"

# ---------------------------------------------------------------------------
# The manifest.
#   Required = absent means the run cannot work, so we refuse to build the zip.
#   Optional = nice to have; absence is reported but not fatal.
# Expected SHA-16 values are the first 16 hex of SHA-256, the same convention
# csv_sha() in run_frozen_split_analysis.py and preflight_linux.py both use, so
# what is checked here is exactly what the pre-flight will check there.
# ---------------------------------------------------------------------------
$RequiredFiles = @(
    @{ Path = 'requirements.txt';        Sha = $null; Note = 'dependency list' }
    @{ Path = 'run_on_linux.sh';         Sha = $null; Note = 'the one command they run' }
    @{ Path = 'HANDOFF_FOR_LINUX.md';    Sha = $null; Note = 'instructions, read this first' }
    @{ Path = 'RUN_ORDER.md';            Sha = $null; Note = 'what each stage does' }

    @{ Path = 'data/01_primary_adr_detection/dev_psytar/psytar_harmonised.csv'
       Sha = '859fb25f0fb1854f'; Note = 'PsyTAR dev corpus, 6,003 sentences' }
    @{ Path = 'data/01_primary_adr_detection/dev_psytar/PsyTAR_dataset.xlsx'
       Sha = $null; Note = 'PsyTAR raw workbook, ST7 subgroup audit' }
    @{ Path = 'data/01_primary_adr_detection/external_val_cadec/cadec_harmonised.csv'
       Sha = '87f88871fa9b07c7'; Note = 'CADEC external validation, 7,823 sentences' }
    @{ Path = 'data/02_secondary_sentiment_scaling/dev_uci_drug_review/uci_druglib_harmonised.csv'
       Sha = '960d3f25de715bdb'; Note = 'UCI DrugLib, 4,107 rows' }
    @{ Path = 'data/02_secondary_sentiment_scaling/external_val_webmd/webmd_harmonised.csv'
       Sha = $null; Note = 'WebMD, 320,093 rows (ST6 corpus scaling)' }

    @{ Path = 'results/colab_transformer_gpu_results.json'
       Sha = $null; Note = 'GPU saturated-run energy + metrics (INPUT)' }
    @{ Path = 'results/efficient_transformer_seed42_predictions.npz'
       Sha = $null; Note = 'DistilBERT preds + frozen split texts (INPUT)' }
    @{ Path = 'results/biomedical_transformer_seed42_predictions.npz'
       Sha = $null; Note = 'PubMedBERT preds + frozen split texts (INPUT)' }
)

# Extra seeds. run_frozen_split_analysis.py globs results/*_predictions.npz and
# groups by seed, using 42 for the paired bootstrap and the others for the
# seed-variance report. Shipping them keeps the output identical in shape to what
# we already have; they are small.
$OptionalFiles = @(
    'README.md'
    'results/efficient_transformer_seed123_predictions.npz'
    'results/biomedical_transformer_seed123_predictions.npz'
    'results/efficient_transformer_seed456_predictions.npz'
    'results/biomedical_transformer_seed456_predictions.npz'
)

# Every .py under scripts/ is copied. These specific ones are asserted present,
# because a missing one fails only mid-run, after the professor has waited.
$CriticalScripts = @(
    'run_all_cpu.py'
    'measure_cpu_energy.py'
    'run_frozen_split_analysis.py'
    'eccms_regime_st8.py'
    'eccms_selection.py'
    'budget_and_subgroup_st6_st7.py'
    'minimal_pipeline_st3.py'
    'metrics_utils.py'
    'rapl_utils.py'
    'preflight_linux.py'
)

# Shell scripts must have LF endings and no BOM. A CRLF shebang makes Linux
# report "bad interpreter: /usr/bin/env bash^M"; a UTF-8 BOM makes the first line
# unparseable. Git for Windows checkouts routinely introduce CRLF, so normalise
# rather than assume.
$LfFiles = @('run_on_linux.sh')

# ---------------------------------------------------------------------------
function Write-Head($text) {
    Write-Host ''
    Write-Host ('=' * 78)
    Write-Host "  $text"
    Write-Host ('=' * 78)
}
function Sha256Full($path) { (Get-FileHash -Path $path -Algorithm SHA256).Hash.ToLower() }
function Sha16($path)      { (Sha256Full $path).Substring(0, 16) }
function HumanSize($bytes) {
    if ($bytes -lt 1024)      { return "$bytes B" }
    if ($bytes -lt 1048576)   { return ('{0:N1} KB' -f ($bytes / 1024)) }
    if ($bytes -lt 1073741824){ return ('{0:N1} MB' -f ($bytes / 1048576)) }
    return ('{0:N2} GB' -f ($bytes / 1073741824))
}
function Stage-File($rel) {
    $src = Join-Path $Root ($rel -replace '/', '\')
    $dst = Join-Path $Stage ($rel -replace '/', '\')
    $dir = Split-Path -Parent $dst
    if (-not (Test-Path -LiteralPath $dir)) { New-Item -ItemType Directory -Path $dir -Force | Out-Null }
    Copy-Item -LiteralPath $src -Destination $dst -Force
    return $dst
}

Write-Head 'BUILDING THE LINUX HANDOFF PACKAGE'
Write-Host "  repo root : $Root"
Write-Host "  timestamp : $Stamp (UTC)"

# ---------------- 1. verify everything required is present -----------------
Write-Host ''
Write-Host '--- checking required files ---------------------------------------------'
$missing = @()
foreach ($item in $RequiredFiles) {
    $p = Join-Path $Root ($item.Path -replace '/', '\')
    if (-not (Test-Path -LiteralPath $p -PathType Leaf)) {
        Write-Host "  [MISSING] $($item.Path)" -ForegroundColor Red
        $missing += $item.Path
    } elseif ((Get-Item -LiteralPath $p).Length -eq 0) {
        Write-Host "  [EMPTY  ] $($item.Path)" -ForegroundColor Red
        $missing += "$($item.Path) (zero bytes)"
    } else {
        Write-Host ("  [ ok    ] {0,-72} {1}" -f $item.Path, (HumanSize (Get-Item -LiteralPath $p).Length))
    }
}
foreach ($s in $CriticalScripts) {
    $p = Join-Path $Root "scripts\$s"
    if (-not (Test-Path -LiteralPath $p -PathType Leaf)) {
        Write-Host "  [MISSING] scripts/$s" -ForegroundColor Red
        $missing += "scripts/$s"
    }
}
if ($missing.Count -gt 0) {
    Write-Head 'ABORTED — required files are missing'
    $missing | ForEach-Object { Write-Host "    * $_" -ForegroundColor Red }
    Write-Host ''
    Write-Host '  Nothing was written. Restore these and re-run. Shipping an incomplete'
    Write-Host '  package costs a full round trip with the professor to discover.'
    exit 1
}

# ---------------- 2. verify the corpora are the published ones -------------
if (-not $SkipHashCheck) {
    Write-Host ''
    Write-Host '--- verifying corpus digests (SHA-256, first 16 hex) --------------------'
    $badHash = @()
    foreach ($item in $RequiredFiles | Where-Object { $_.Sha }) {
        $p = Join-Path $Root ($item.Path -replace '/', '\')
        $got = Sha16 $p
        if ($got -eq $item.Sha) {
            Write-Host "  [ ok    ] $($item.Path) = $got"
        } else {
            Write-Host "  [DIFFERS] $($item.Path)" -ForegroundColor Red
            Write-Host "            got $got, expected $($item.Sha)" -ForegroundColor Red
            $badHash += $item.Path
        }
    }
    if ($badHash.Count -gt 0) {
        Write-Head 'ABORTED — a corpus file no longer matches the published digest'
        Write-Host '  The numbers in the paper were computed on the files whose digests are'
        Write-Host '  recorded above. A file that differs means the results would not be'
        Write-Host '  comparable, and the pre-flight on the professor''s machine will refuse'
        Write-Host '  to run for the same reason.'
        Write-Host ''
        Write-Host '  The usual cause is opening a .csv in Excel and saving it, which'
        Write-Host '  rewrites line endings. Restore the file from git:'
        foreach ($b in $badHash) { Write-Host "      git checkout -- `"$b`"" -ForegroundColor Yellow }
        exit 1
    }
}

# ---------------- 3. stage -------------------------------------------------
Write-Host ''
Write-Host '--- staging --------------------------------------------------------------'
if (Test-Path -LiteralPath $StageParent) { Remove-Item -LiteralPath $StageParent -Recurse -Force }
New-Item -ItemType Directory -Path $Stage -Force | Out-Null

foreach ($item in $RequiredFiles) { Stage-File $item.Path | Out-Null }

$optPresent = 0
foreach ($rel in $OptionalFiles) {
    $p = Join-Path $Root ($rel -replace '/', '\')
    if (Test-Path -LiteralPath $p -PathType Leaf) { Stage-File $rel | Out-Null; $optPresent++ }
    else { Write-Host "  [  -    ] $rel (optional, absent)" }
}

$pyCount = 0
foreach ($f in Get-ChildItem -LiteralPath (Join-Path $Root 'scripts') -Filter '*.py' -File) {
    Stage-File "scripts/$($f.Name)" | Out-Null
    $pyCount++
}
Write-Host "  staged: $($RequiredFiles.Count) required, $optPresent optional, $pyCount scripts"

# results/ must contain ONLY the three GPU inputs plus the extra-seed .npz.
# Assert it, rather than trusting the loop above, because this is the property
# that stops a stale Windows JSON from coming back as a Linux measurement.
$stagedDerived = Get-ChildItem -LiteralPath (Join-Path $Stage 'results') -Filter '*.json' -File |
                 Where-Object { $_.Name -ne 'colab_transformer_gpu_results.json' }
if ($stagedDerived) {
    Write-Head 'ABORTED — a derived JSON reached the staging folder'
    $stagedDerived | ForEach-Object { Write-Host "    * results/$($_.Name)" -ForegroundColor Red }
    Write-Host '  These are pipeline OUTPUTS. See the header comment: shipping them lets a'
    Write-Host '  failed step return old Windows numbers as this run''s result.'
    exit 1
}

# ---------------- 4. normalise line endings on shell scripts ---------------
Write-Host ''
Write-Host '--- normalising shell scripts to LF, no BOM -----------------------------'
$utf8NoBom = New-Object System.Text.UTF8Encoding($false)
foreach ($rel in $LfFiles) {
    $dst  = Join-Path $Stage ($rel -replace '/', '\')
    $text = [System.IO.File]::ReadAllText($dst)
    $had  = ([regex]::Matches($text, "`r`n")).Count
    $text = $text -replace "`r`n", "`n"
    $text = $text -replace "`r", "`n"
    [System.IO.File]::WriteAllText($dst, $text, $utf8NoBom)
    $check = [System.IO.File]::ReadAllBytes($dst)
    # @() around the pipeline on purpose: Set-StrictMode -Version Latest makes
    # .Count on a null result (zero matches) throw instead of returning 0.
    $stillCr = @($check | Where-Object { $_ -eq 13 }).Count
    $bom = ($check.Length -ge 3 -and $check[0] -eq 0xEF -and $check[1] -eq 0xBB -and $check[2] -eq 0xBF)
    if ($stillCr -gt 0 -or $bom) {
        Write-Host "  [FAIL] $rel still has CR bytes ($stillCr) or a BOM ($bom)" -ForegroundColor Red
        exit 1
    }
    Write-Host "  [ ok  ] $rel — converted $had CRLF pair(s), no BOM, LF only"
}

# ---------------- 5. manifest with digests --------------------------------
# The mirror image of the SHA256SUMS.txt that comes back: if a file is corrupted
# in transit, comparing the two localises it immediately.
Write-Host ''
Write-Host '--- writing PACKAGE_MANIFEST.txt ---------------------------------------'
$manifestPath = Join-Path $Stage 'PACKAGE_MANIFEST.txt'
$lines = New-Object System.Collections.Generic.List[string]
$lines.Add("# Energy-Aware Drug-Review — Linux handoff package")
$lines.Add("# built (UTC): $Stamp")
$lines.Add("# SHA-256 of every file in this package, computed on the sending machine.")
$lines.Add("# Verify after extracting:   sha256sum -c PACKAGE_MANIFEST.txt --ignore-missing")
$lines.Add("")
$total = 0L
foreach ($f in Get-ChildItem -LiteralPath $Stage -Recurse -File | Sort-Object FullName) {
    if ($f.Name -eq 'PACKAGE_MANIFEST.txt') { continue }
    $rel = $f.FullName.Substring($Stage.Length + 1) -replace '\\', '/'
    $lines.Add("$(Sha256Full $f.FullName)  $rel")
    $total += $f.Length
}
[System.IO.File]::WriteAllLines($manifestPath, $lines, $utf8NoBom)
Write-Host "  $($lines.Count - 5) file(s), $(HumanSize $total) unpacked"

# ---------------- 6. zip --------------------------------------------------
Write-Host ''
Write-Host '--- compressing ---------------------------------------------------------'
if (Test-Path -LiteralPath $Zip) { Remove-Item -LiteralPath $Zip -Force }
# .NET ZipFile first: Windows PowerShell 5.1's Compress-Archive is very slow on a
# corpus this size (WebMD alone is 320k rows) and has a 2 GB ceiling. Fall back to
# Compress-Archive if the assembly is unavailable, so this works on any host.
$zipped = $false
try {
    Add-Type -AssemblyName System.IO.Compression.FileSystem -ErrorAction Stop
    [System.IO.Compression.ZipFile]::CreateFromDirectory(
        $StageParent, $Zip,
        [System.IO.Compression.CompressionLevel]::Optimal, $false)
    $zipped = $true
    Write-Host '  used System.IO.Compression.ZipFile'
} catch {
    Write-Host "  ZipFile unavailable ($($_.Exception.GetType().Name)); using Compress-Archive"
}
if (-not $zipped) {
    Compress-Archive -Path $Stage -DestinationPath $Zip -CompressionLevel Optimal
}
if (-not (Test-Path -LiteralPath $Zip)) {
    Write-Head 'ABORTED — the archive was not created'
    Write-Host '  Both compression paths failed. The staged folder is intact at:'
    Write-Host "      $Stage"
    Write-Host '  You can zip it by hand (right-click > Send to > Compressed folder).'
    exit 1
}
$zipSize = (Get-Item -LiteralPath $Zip).Length

if (-not $KeepStaging) { Remove-Item -LiteralPath $StageParent -Recurse -Force }
else { Write-Host "  staging kept at: $StageParent" }

# ---------------- 7. what to do next --------------------------------------
Write-Head 'PACKAGE READY'
Write-Host "  $Zip"
Write-Host "  $(HumanSize $zipSize) zipped / $(HumanSize $total) unpacked"
Write-Host ''
Write-Host '  Send that one file. Ask them to:'
Write-Host '      1. extract it'
Write-Host '      2. read HANDOFF_FOR_LINUX.md  (four steps, about 20 minutes)'
Write-Host '      3. send back the single handoff_results_<timestamp>.tar.gz it produces'
Write-Host ''
Write-Host '  Two things worth saying explicitly when you send it:'
Write-Host '      * it has to be Linux booted on the hardware — not WSL, not a VM, not'
Write-Host '        Docker. None of those expose the CPU energy counters, which is the'
Write-Host '        only reason the run is being asked for.'
Write-Host '      * send the .tar.gz file itself, not a pasted or summarised log.'
Write-Host ''
