# Ingest the outlets marked `runner: home` in config/outlets.yaml (their CDNs block GitHub's
# datacenter IPs). Run by the "Nazariya home ingest" scheduled task; see install_home_task.ps1.
# Needs DATABASE_URL in the repo's .env file, otherwise it would silently write to local SQLite.

$ErrorActionPreference = "Stop"
$repo = Split-Path -Parent $PSScriptRoot
Set-Location $repo

$logDir = Join-Path $repo "data\logs"
New-Item -ItemType Directory -Force $logDir | Out-Null
$log = Join-Path $logDir ("home_ingest_{0:yyyy-MM}.log" -f (Get-Date))

function Write-Log($msg) { "{0:yyyy-MM-dd HH:mm:ss} $msg" -f (Get-Date) | Add-Content -Path $log -Encoding utf8 }

$envFile = Join-Path $repo ".env"
if (-not ((Test-Path $envFile) -and (Select-String -Path $envFile -Pattern '^\s*DATABASE_URL\s*=' -Quiet))) {
    Write-Log "ERROR: DATABASE_URL missing from $envFile; not running"
    exit 1
}

$env:PYTHONIOENCODING = "utf-8"
Write-Log "start"
& "$repo\.venv\Scripts\python.exe" -m nazariya.ingest --runner home *>> $log
$code = $LASTEXITCODE
Write-Log "end (exit $code)"
exit $code
