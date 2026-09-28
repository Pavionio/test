param([Parameter(Mandatory = $true)][string]$DatasetZip)

# End-to-end reproduction on Windows with the existing conda environment mlsec.
# Checkpoints in src.encode, src.index, src.retrieve and src.rerank allow resume.
$ErrorActionPreference = 'Stop'
Set-Location (Split-Path $PSScriptRoot -Parent)
$resolvedZip = (Resolve-Path -LiteralPath $DatasetZip).Path
$env:AVITO_DATASET_ZIP = $resolvedZip
$py = Join-Path $env:USERPROFILE 'anaconda3\envs\mlsec\python.exe'
if (-not (Test-Path $py)) { throw "mlsec Python not found at $py" }

function Run([string[]]$Arguments) {
    & $py -m @Arguments
    if ($LASTEXITCODE -ne 0) { throw "Failed: python -m $($Arguments -join ' ')" }
}

docker compose up -d qdrant
if ($LASTEXITCODE -ne 0) { throw 'Could not start local Qdrant' }
Run @('src.download_data', '--archive', $resolvedZip)
if (-not (Test-Path 'models\bge-m3\pytorch_model.bin') -or
    -not (Test-Path 'models\bge-reranker-v2-m3\model.safetensors')) {
    Run @('src.download_models')
}
Run @('src.prepare')
Run @('src.queries')
Run @('src.start_run')
Run @('src.lexical', '--dataset', 'train')
Run @('src.lexical', '--dataset', 'benchmark')
Run @('src.encode', '--dataset', 'train')
Run @('src.encode', '--dataset', 'benchmark')
Run @('src.index', '--dataset', 'train')
Run @('src.index', '--dataset', 'benchmark')
foreach ($set in @('fit','validation','benchmark')) {
    Run @('src.encode_queries', '--dataset', $set)
}
Run @('src.retrieve', '--dataset', 'fit', '--geo', 'geo_off')
Run @('src.retrieve', '--dataset', 'fit', '--geo', 'geo_exact')
foreach ($geo in @('geo_off','geo_exact')) {
    Run @('src.retrieve', '--dataset', 'validation', '--geo', $geo)
}
Run @('src.evaluate', 'retrieval')
foreach ($geo in @('geo_off','geo_exact')) {
    Run @('src.catboost_stage', 'train', '--geo', $geo)
    Run @('src.catboost_stage', 'score', '--dataset', 'validation', '--geo', $geo)
}
Run @('src.rerank', '--dataset', 'validation', '--geo', 'geo_off', '--branch', 'B', '--pilot-queries', '50')
foreach ($geo in @('geo_off','geo_exact')) {
    foreach ($branch in @('A','B')) {
        Run @('src.rerank', '--dataset', 'validation', '--geo', $geo, '--branch', $branch)
        Run @('src.evaluate', 'branch', '--geo', $geo, '--branch', $branch)
    }
}
foreach ($branch in @('A','B')) { Run @('src.evaluate_fill', '--branch', $branch) }
Run @('src.evaluate_bias')
Run @('src.provenance')
Run @('src.select_best', '--budget-hours', '10')
$selected = (Get-Content 'artifacts\experiments\selection.json' -Raw | ConvertFrom-Json).chosen
if ($selected.geo_bias) { Run @('src.error_analysis', '--branch', 'B', '--method', 'bias') }
if ($selected.branch -eq 'A') { Run @('src.catboost_stage', 'train', '--full', '--geo', $selected.geo) }
Run @('src.retrieve', '--dataset', 'benchmark', '--geo', $selected.geo)
if ($selected.branch -eq 'A') {
    Run @('src.catboost_stage', 'score', '--dataset', 'benchmark', '--geo', $selected.geo, '--final')
}
Run @('src.rerank', '--dataset', 'benchmark', '--geo', $selected.geo, '--branch', $selected.branch)
if ($selected.geo_bias) {
    Run @('src.retrieve', '--dataset', 'benchmark', '--geo', 'geo_off')
    Run @('src.rerank', '--dataset', 'benchmark', '--geo', 'geo_off', '--branch', 'B')
    Run @('src.finalize', '--geo', 'geo_exact', '--branch', 'B', '--geo-bias')
} elseif ($selected.fill_geo_off) {
    Run @('src.retrieve', '--dataset', 'benchmark', '--geo', 'geo_off')
    if ($selected.branch -eq 'A') {
        Run @('src.catboost_stage', 'train', '--full', '--geo', 'geo_off')
        Run @('src.catboost_stage', 'score', '--dataset', 'benchmark', '--geo', 'geo_off', '--final')
    }
    Run @('src.rerank', '--dataset', 'benchmark', '--geo', 'geo_off', '--branch', $selected.branch)
    Run @('src.finalize', '--geo', $selected.geo, '--branch', $selected.branch, '--fill-geo-off')
} else {
    Run @('src.finalize', '--geo', $selected.geo, '--branch', $selected.branch)
}
# An additional 200 query texts audit the selected B variant after submission
# generation; its labels cannot affect the branch or quota selection.
if ($selected.branch -eq 'B') {
    foreach ($geo in @('geo_off','geo_exact')) {
        Run @('src.rerank', '--dataset', 'validation', '--geo', $geo,
             '--branch', 'B', '--audit-queries', '200')
    }
    Run @('src.evaluate_audit', '--branch', 'B', '--count', '200')
}
Run @('src.provenance')
Run @('pytest', '-q', 'tests')
