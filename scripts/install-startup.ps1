# Register a current-user Windows Task Scheduler task that starts the
# trading assistant supervisor at logon.
#
# Run as: powershell -ExecutionPolicy Bypass -File scripts\install-startup.ps1

$ErrorActionPreference = "Stop"

$TaskName = "TradingAssistantAutoStart"
$ProjectRoot = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
$StartScript = Join-Path $ProjectRoot "scripts\start.ps1"
$CurrentUser = [System.Security.Principal.WindowsIdentity]::GetCurrent().Name

if (-not (Test-Path $StartScript)) {
    Write-Error "start.ps1 not found at $StartScript"
    exit 1
}

$Existing = Get-ScheduledTask -TaskName $TaskName -ErrorAction SilentlyContinue
if ($Existing) {
    Unregister-ScheduledTask -TaskName $TaskName -Confirm:$false
    Write-Host "Removed existing task '$TaskName'"
}

$Trigger = New-ScheduledTaskTrigger -AtLogOn -User $CurrentUser
$Action = New-ScheduledTaskAction `
    -Execute "powershell.exe" `
    -Argument "-NoProfile -ExecutionPolicy Bypass -WindowStyle Hidden -File `"$StartScript`"" `
    -WorkingDirectory $ProjectRoot

$Settings = New-ScheduledTaskSettingsSet `
    -AllowStartIfOnBatteries `
    -DontStopIfGoingOnBatteries `
    -MultipleInstances IgnoreNew `
    -StartWhenAvailable `
    -ExecutionTimeLimit (New-TimeSpan -Hours 0)

Register-ScheduledTask `
    -TaskName $TaskName `
    -Trigger $Trigger `
    -Action $Action `
    -Settings $Settings `
    -Description "Start the trading assistant supervisor in the background at logon" `
    -RunLevel Limited | Out-Null

Write-Host "Task '$TaskName' registered for $CurrentUser."
Write-Host "The trading assistant supervisor will start in the background at logon."
Write-Host ""
Write-Host "To remove: Unregister-ScheduledTask -TaskName '$TaskName'"
