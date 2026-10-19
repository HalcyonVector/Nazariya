# Registers (or re-registers) the "Nazariya home ingest" scheduled task for the current user:
# every 2 hours, catching up after sleep/shutdown, only when a network is available.
# Re-run this after moving or renaming the repo folder, since the task stores the script path.
#
#   powershell -ExecutionPolicy Bypass -File scripts\install_home_task.ps1
#   Unregister-ScheduledTask -TaskName "Nazariya home ingest"   # to remove it

$ErrorActionPreference = "Stop"
$name = "Nazariya home ingest"
$script = Join-Path $PSScriptRoot "home_ingest.ps1"

# conhost --headless (Windows 10 21H2+/11) gives PowerShell a console that is never shown;
# `-WindowStyle Hidden` alone still flashes a window on screen for a moment at every run.
$action = New-ScheduledTaskAction -Execute "conhost.exe" `
    -Argument "--headless powershell.exe -NoProfile -NonInteractive -ExecutionPolicy Bypass -File `"$script`""
# start 10 minutes from now, then every 2 hours, indefinitely
$trigger = New-ScheduledTaskTrigger -Once -At (Get-Date).AddMinutes(10) `
    -RepetitionInterval (New-TimeSpan -Hours 2)
$settings = New-ScheduledTaskSettingsSet -StartWhenAvailable -RunOnlyIfNetworkAvailable `
    -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries `
    -ExecutionTimeLimit (New-TimeSpan -Minutes 45) -MultipleInstances IgnoreNew
$principal = New-ScheduledTaskPrincipal -UserId "$env:USERDOMAIN\$env:USERNAME" -LogonType Interactive -RunLevel Limited

Register-ScheduledTask -TaskName $name -Action $action -Trigger $trigger -Settings $settings `
    -Principal $principal -Description "Ingests Nazariya outlets that block datacenter IPs (runner: home)" -Force | Out-Null
Get-ScheduledTask -TaskName $name | Get-ScheduledTaskInfo | Select-Object TaskName, NextRunTime
