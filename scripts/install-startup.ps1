# install-startup.ps1 — Register a Windows Task Scheduler task to auto-start
# the trading assistant on user login.
#
# Run as: powershell -ExecutionPolicy Bypass -File scripts\install-startup.ps1
# Requires: Administrator privileges (for Task Scheduler)

$ErrorActionPreference = "Stop"

$TaskName = "TradingAssistantAutoStart"
$ProjectRoot = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
$StartScript = Join-Path $ProjectRoot "scripts\start.ps1"

if (-not (Test-Path $StartScript)) {
    Write-Error "start.ps1 not found at $StartScript"
    exit 1
}

# Remove existing task if present
$Existing = Get-ScheduledTask -TaskName $TaskName -ErrorAction SilentlyContinue
if ($Existing) {
    Unregister-ScheduledTask -TaskName $TaskName -Confirm:$false
    Write-Host "Removed existing task '$TaskName'"
}

# Create trigger: at logon of current user
$Trigger = New-ScheduledTaskTrigger -AtLogOn -User $env:USERNAME

# Create action: run start.ps1 via PowerShell
$Action = New-ScheduledTaskAction `
    -Execute "powershell.exe" `
    -Argument "-ExecutionPolicy Bypass -WindowStyle Hidden -File `"$StartScript`"" `
    -WorkingDirectory $ProjectRoot

# Settings: run even on battery, start if missed
$Settings = New-ScheduledTaskSettingsSet `
    -AllowStartIfOnBatteries `
    -DontStopIfGoingOnBatteries `
    -StartWhenAvailable `
    -ExecutionTimeLimit (New-TimeSpan -Hours 0)

Register-ScheduledTask `
    -TaskName $TaskName `
    -Trigger $Trigger `
    -Action $Action `
    -Settings $Settings `
    -Description "Auto-start trading assistant orchestrator on login" `
    -RunLevel Limited

Write-Host "Task '$TaskName' registered successfully."
Write-Host "The trading assistant will auto-start on your next login."
Write-Host ""
Write-Host "To remove: Unregister-ScheduledTask -TaskName '$TaskName'"
