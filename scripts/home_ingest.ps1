# Ingest the outlets marked `runner: home` in config/outlets.yaml (their CDNs block GitHub's
# datacenter IPs). Run by the "Nazariya home ingest" scheduled task; see install_home_task.ps1.
#
# The PC keeps its own SQLite file (data/home.db), embeds the new articles on the GPU, and hands
# the last three days to the cloud pipeline as an encrypted release asset (home_inbox), which the
# next cloud run imports. Needs STATE_KEY in the repo's .env, Git Bash, and an authenticated gh.

$ErrorActionPreference = "Stop"
$repo = Split-Path -Parent $PSScriptRoot
Set-Location $repo

$logDir = Join-Path $repo "data\logs"
New-Item -ItemType Directory -Force $logDir | Out-Null
$log = Join-Path $logDir ("home_ingest_{0:yyyy-MM}.log" -f (Get-Date))

function Write-Log($msg) { "{0:yyyy-MM-dd HH:mm:ss} $msg" -f (Get-Date) | Add-Content -Path $log -Encoding utf8 }

$envFile = Join-Path $repo ".env"
$keyLine = if (Test-Path $envFile) { Select-String -Path $envFile -Pattern '^\s*STATE_KEY\s*=\s*(.+)$' | Select-Object -First 1 }
if (-not $keyLine) {
    Write-Log "ERROR: STATE_KEY missing from $envFile; not running"
    exit 1
}
$env:STATE_KEY = $keyLine.Matches[0].Groups[1].Value.Trim().Trim('"').Trim("'")
$env:DATABASE_URL = "sqlite:///data/home.db"
$env:PYTHONIOENCODING = "utf-8"
$py = "`"$repo\.venv\Scripts\python.exe`""
$bash = "C:\Program Files\Git\bin\bash.exe"

Write-Log "start"
# Redirect through cmd: Windows PowerShell 5.1's *>> re-encodes native output as UTF-16,
# which garbles a UTF-8 log, and 2>&1 would turn stderr lines into terminating errors.
cmd /c "$py -m nazariya.ingest --runner home >> `"$log`" 2>&1"
$code = $LASTEXITCODE
Write-Log "ingest done (exit $code)"
cmd /c "$py -m nazariya.cluster --embed-only --budget-min 20 >> `"$log`" 2>&1"
Write-Log "embed-only done (exit $LASTEXITCODE)"
cmd /c "$py -m nazariya.inbox export data\home_inbox.jsonl.gz --days 3 >> `"$log`" 2>&1"
cmd /c "`"$bash`" scripts/state.sh push data/home_inbox.jsonl.gz home_inbox.jsonl.gz.enc >> `"$log`" 2>&1"
Write-Log "inbox pushed (exit $LASTEXITCODE)"
cmd /c "$py -m nazariya.retain >> `"$log`" 2>&1"
Write-Log "end"
exit $code
