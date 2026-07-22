param(
    [ValidateSet("fast", "full")]
    [string]$Mode = "fast",
    [switch]$Bib
)

$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

$job = if ($Mode -eq "fast") { "main-fast" } else { "main" }
$texInput = if ($Mode -eq "fast") { "\def\FASTCOMPILE{1}\input{main.tex}" } else { "main.tex" }
$xelatexArgs = @(
    "-synctex=0",
    "-interaction=nonstopmode",
    "-halt-on-error",
    "-file-line-error",
    "-jobname=$job"
)

function Invoke-LoggedCommand {
    param(
        [string]$Command,
        [string[]]$Arguments
    )

    Write-Host "==> $Command $($Arguments -join ' ')"
    $elapsed = Measure-Command {
        & $Command @Arguments
    }
    if ($LASTEXITCODE -ne 0) {
        throw "$Command failed with exit code $LASTEXITCODE"
    }
    Write-Host ("    {0:N2}s" -f $elapsed.TotalSeconds)
}

function Test-BibliographyOutdated {
    param([string]$JobName)

    $bbl = "$JobName.bbl"
    $signatureFile = "$JobName.bibsig"
    if ($Bib -or -not (Test-Path $bbl)) {
        return $true
    }

    $signature = Get-BibliographySignature $JobName
    if (-not (Test-Path $signatureFile)) {
        return $true
    }

    $oldSignature = Get-Content -Raw -Encoding UTF8 $signatureFile
    return ($signature.TrimEnd() -ne $oldSignature.TrimEnd())
}

function Get-BibliographySignature {
    param([string]$JobName)

    $parts = [System.Collections.Generic.List[string]]::new()
    if (Test-Path "$JobName.aux") {
        Get-Content "$JobName.aux" |
            Where-Object { $_ -match "\\(citation|bibstyle|bibdata)\{" } |
            ForEach-Object { $parts.Add($_) }
    }

    Get-ChildItem -Path ".\bib" -Recurse -Include *.bib,*.bst -ErrorAction SilentlyContinue |
        Sort-Object FullName |
        ForEach-Object {
            $hash = (Get-FileHash -Algorithm SHA256 -LiteralPath $_.FullName).Hash
            $parts.Add("$($_.FullName)|$hash")
        }

    return ($parts -join "`n")
}

function Save-BibliographySignature {
    param([string]$JobName)

    if (Test-Path "$JobName.aux") {
        Set-Content -Encoding UTF8 -Path "$JobName.bibsig" -Value (Get-BibliographySignature $JobName)
    }
}

Invoke-LoggedCommand "xelatex" ($xelatexArgs + @($texInput))

if (Test-BibliographyOutdated $job) {
    Invoke-LoggedCommand "bibtex" @($job)
    Save-BibliographySignature $job
    Invoke-LoggedCommand "xelatex" ($xelatexArgs + @($texInput))
    Invoke-LoggedCommand "xelatex" ($xelatexArgs + @($texInput))
}
elseif (Test-Path "$job.log") {
    $needsRerun = Select-String -Path "$job.log" -Pattern "Rerun to get cross-references right|Label\(s\) may have changed|Citation.*undefined" -Quiet
    if ($needsRerun) {
        Invoke-LoggedCommand "xelatex" ($xelatexArgs + @($texInput))
    }
}

Write-Host "Output: $job.pdf"
