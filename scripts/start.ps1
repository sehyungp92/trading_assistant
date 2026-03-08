# start.ps1 — Start the trading assistant orchestrator as a hidden background process.
# Features: log rotation, health check, watchdog with auto-restart.
# Usage: powershell -ExecutionPolicy Bypass -File scripts\start.ps1

$ErrorActionPreference = "Stop"

# Resolve project root (parent of scripts/)
$ProjectRoot = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
Set-Location $ProjectRoot

# Activate virtual environment
$VenvActivate = Join-Path $ProjectRoot ".venv\Scripts\Activate.ps1"
if (Test-Path $VenvActivate) {
    & $VenvActivate
} else {
    Write-Warning "No .venv found at $VenvActivate — using system Python"
}

# Check if already running
$Existing = Get-Process -Name "pythonw" -ErrorAction SilentlyContinue |
    Where-Object { $_.CommandLine -like "*uvicorn orchestrator.app*" }
if ($Existing) {
    Write-Host "Trading assistant already running (PID $($Existing.Id))"
    exit 0
}

# --- Log rotation ---
function Rotate-LogFile {
    param([string]$LogPath, [int]$MaxSizeMB = 10, [int]$MaxFiles = 5)
    if (-not (Test-Path $LogPath)) { return }
    $file = Get-Item $LogPath
    if ($file.Length -lt ($MaxSizeMB * 1MB)) { return }

    # Shift existing rotated files: .4 -> .5 (delete), .3 -> .4, etc.
    for ($i = $MaxFiles; $i -ge 1; $i--) {
        $src = "$LogPath.$i"
        if ($i -eq $MaxFiles) {
            if (Test-Path $src) { Remove-Item $src -Force }
        } else {
            $dst = "$LogPath.$($i + 1)"
            if (Test-Path $src) { Rename-Item $src $dst -Force }
        }
    }
    Rename-Item $LogPath "$LogPath.1" -Force
    Write-Host "Rotated $LogPath (was $([math]::Round($file.Length / 1MB, 1)) MB)"
}

$LogFile = Join-Path $ProjectRoot "logs\orchestrator.log"
$ErrFile = "$LogFile.err"
$LogDir = Split-Path $LogFile
if (-not (Test-Path $LogDir)) { New-Item -ItemType Directory -Path $LogDir -Force | Out-Null }

Rotate-LogFile -LogPath $LogFile
Rotate-LogFile -LogPath $ErrFile

# --- Health check function ---
function Wait-ForHealth {
    param([int]$TimeoutSeconds = 30)
    $deadline = (Get-Date).AddSeconds($TimeoutSeconds)
    while ((Get-Date) -lt $deadline) {
        try {
            $response = Invoke-WebRequest -Uri "http://127.0.0.1:8000/health" -TimeoutSec 2 -UseBasicParsing -ErrorAction Stop
            if ($response.StatusCode -eq 200) {
                Write-Host "Health check passed"
                return $true
            }
        } catch {
            # Not ready yet
        }
        Start-Sleep -Seconds 2
    }
    return $false
}

# --- Watchdog loop ---
$MaxRestarts = 5
$RestartCount = 0

while ($RestartCount -lt $MaxRestarts) {
    $timestamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"

    if ($RestartCount -gt 0) {
        Write-Host "[$timestamp] Restart attempt $RestartCount/$MaxRestarts..."
        Start-Sleep -Seconds 10
    }

    # Start as hidden background process
    $proc = Start-Process -WindowStyle Hidden -FilePath "pythonw" -ArgumentList @(
        "-m", "uvicorn", "orchestrator.app:app",
        "--host", "127.0.0.1",
        "--port", "8000"
    ) -WorkingDirectory $ProjectRoot -RedirectStandardOutput $LogFile -RedirectStandardError $ErrFile -PassThru

    Write-Host "[$timestamp] Started trading assistant (PID $($proc.Id)). Log: $LogFile"

    # Wait for health
    $healthy = Wait-ForHealth -TimeoutSeconds 30
    if (-not $healthy) {
        Write-Host "[$timestamp] ERROR: Health check failed after 30s"
        if (-not $proc.HasExited) {
            Stop-Process -Id $proc.Id -Force -ErrorAction SilentlyContinue
        }
        $RestartCount++
        continue
    }

    # Monitor — wait for process to exit
    $proc.WaitForExit()
    $exitCode = $proc.ExitCode
    $timestamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    Write-Host "[$timestamp] Process exited with code $exitCode"
    $RestartCount++
}

if ($RestartCount -ge $MaxRestarts) {
    $timestamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    Write-Host "[$timestamp] ERROR: Max restarts ($MaxRestarts) reached. Giving up."
    exit 1
}
