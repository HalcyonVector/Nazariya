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
# Redirect through cmd: Windows PowerShell 5.1's *>> re-encodes native output as UTF-16,
# which garbles a UTF-8 log, and 2>&1 would turn stderr lines into terminating errors.
cmd /c "`"$repo\.venv\Scripts\python.exe`" -m nazariya.ingest --runner home >> `"$log`" 2>&1"
$code = $LASTEXITCODE
Write-Log "end (exit $code)"

# Embed the backlog here (GPU when available; much faster than the cloud's CPU). Assignment to
# events stays with the cloud workflow, so there is only ever one writer of events.
cmd /c "`"$repo\.venv\Scripts\python.exe`" -m nazariya.cluster --embed-only --budget-min 20 >> `"$log`" 2>&1"
Write-Log "embed-only done (exit $LASTEXITCODE)"

# Deliberately no cloud-workflow trigger here: every extra run wakes the free-tier database and
# spends compute hours; the workflow's own 4-hourly schedule is enough.
exit $code
