<#
.SYNOPSIS
    Dispara as réplicas de produção de um sistema, em sequência.

.DESCRIPTION
    As réplicas rodam uma após a outra porque compartilham a mesma GPU:
    executá-las em paralelo divide a banda da placa e não reduz o tempo total.

    Cada réplica usa uma semente distinta (src/config.py::replica_seed), o que
    as torna estatisticamente independentes — condição para o teste de
    convergência da Fase 1.

.EXAMPLE
    .\scripts\run_replicas.ps1 -System A
    .\scripts\run_replicas.ps1 -System B -Replicas 1,2,3
    .\scripts\run_replicas.ps1 -System A -Ns 1 -EquilNs 0.02      # teste rápido
#>
[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)][ValidateSet('A', 'B')][string]$System,
    [int[]]$Replicas = @(1, 2, 3),
    [double]$Ns = 0,
    [double]$EquilNs = -1,
    [string]$Platform = ''
)

$ErrorActionPreference = 'Stop'
$repo = Split-Path -Parent $PSScriptRoot
Set-Location $repo

$logDir = Join-Path $repo 'logs'
if (-not (Test-Path $logDir)) { New-Item -ItemType Directory $logDir | Out-Null }

foreach ($r in $Replicas) {
    $stamp = Get-Date -Format 'yyyyMMdd-HHmmss'
    $log = Join-Path $logDir "sys$System-rep$r-$stamp.log"

    $args = @('-m', 'src.simulate', '--system', $System, '--replica', "$r")
    if ($Ns -gt 0)      { $args += @('--ns', "$Ns") }
    if ($EquilNs -ge 0) { $args += @('--equil-ns', "$EquilNs") }
    if ($Platform)      { $args += @('--platform', $Platform) }

    Write-Host "=== Sistema $System, réplica $r -> $log ===" -ForegroundColor Cyan
    & python $args | Tee-Object -FilePath $log
    if ($LASTEXITCODE -ne 0) {
        Write-Error "réplica $r falhou (exit $LASTEXITCODE); veja $log"
    }
}

Write-Host "Todas as réplicas concluídas." -ForegroundColor Green
