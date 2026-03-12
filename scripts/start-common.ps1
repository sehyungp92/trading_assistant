# Shared helpers for starting the trading assistant orchestrator on Windows.

$Script:OrchestratorHealthUrl = "http://127.0.0.1:8000/health"
$Script:OrchestratorCommandLinePattern = "orchestrator.app:app"

function Enter-OrchestratorSupervisorLock {
    param(
        [Parameter(Mandatory = $true)]
        [string]$LockFile
    )

    $lockDir = Split-Path -Parent $LockFile
    if ($lockDir) {
        New-Item -ItemType Directory -Path $lockDir -Force | Out-Null
    }

    try {
        return [System.IO.File]::Open(
            $LockFile,
            [System.IO.FileMode]::OpenOrCreate,
            [System.IO.FileAccess]::ReadWrite,
            [System.IO.FileShare]::None
        )
    } catch [System.IO.IOException] {
        return $null
    }
}

function Exit-OrchestratorSupervisorLock {
    param(
        [AllowNull()]
        [object]$LockHandle,
        [Parameter(Mandatory = $true)]
        [string]$LockFile
    )

    if ($LockHandle) {
        try {
            $LockHandle.Close()
            $LockHandle.Dispose()
        } catch {
        }
    }

    Remove-Item $LockFile -Force -ErrorAction SilentlyContinue
}

function Resolve-OrchestratorPythonw {
    param(
        [Parameter(Mandatory = $true)]
        [string]$ProjectRoot
    )

    $candidates = @(
        (Join-Path $ProjectRoot ".venv\Scripts\pythonw.exe"),
        (Join-Path $ProjectRoot "venv\Scripts\pythonw.exe")
    )

    foreach ($candidate in $candidates) {
        if (Test-Path $candidate) {
            return (Resolve-Path $candidate).Path
        }
    }

    $command = Get-Command pythonw.exe -ErrorAction SilentlyContinue
    if (-not $command) {
        $command = Get-Command pythonw -ErrorAction SilentlyContinue
    }
    if ($command) {
        return $command.Source
    }

    return $null
}

function Rotate-OrchestratorLog {
    param(
        [Parameter(Mandatory = $true)]
        [string]$LogPath,
        [int]$MaxSizeMB = 10,
        [int]$MaxFiles = 5
    )

    if (-not (Test-Path $LogPath)) {
        return
    }

    $file = Get-Item $LogPath
    if ($file.Length -lt ($MaxSizeMB * 1MB)) {
        return
    }

    for ($i = $MaxFiles; $i -ge 1; $i--) {
        $src = "$LogPath.$i"
        if ($i -eq $MaxFiles) {
            if (Test-Path $src) {
                Remove-Item $src -Force
            }
        } else {
            $dst = "$LogPath.$($i + 1)"
            if (Test-Path $src) {
                Rename-Item $src $dst -Force
            }
        }
    }

    Rename-Item $LogPath "$LogPath.1" -Force
}

function Test-OrchestratorHealthy {
    param(
        [string]$Url = $Script:OrchestratorHealthUrl,
        [int]$TimeoutSeconds = 2
    )

    try {
        $response = Invoke-WebRequest `
            -Uri $Url `
            -TimeoutSec $TimeoutSeconds `
            -UseBasicParsing `
            -ErrorAction Stop
        return $response.StatusCode -eq 200
    } catch {
        return $false
    }
}

function Wait-OrchestratorHealthy {
    param(
        [int]$TimeoutSeconds = 60,
        [string]$Url = $Script:OrchestratorHealthUrl
    )

    $deadline = (Get-Date).AddSeconds($TimeoutSeconds)
    while ((Get-Date) -lt $deadline) {
        if (Test-OrchestratorHealthy -Url $Url) {
            return $true
        }
        Start-Sleep -Seconds 2
    }
    return $false
}

function Get-OrchestratorProcessFromPidFile {
    param(
        [Parameter(Mandatory = $true)]
        [string]$PidFile
    )

    if (-not (Test-Path $PidFile)) {
        return $null
    }

    try {
        $pidText = (Get-Content $PidFile -Raw -ErrorAction Stop).Trim()
        if (-not $pidText) {
            return $null
        }
        return Get-Process -Id ([int]$pidText) -ErrorAction SilentlyContinue
    } catch {
        return $null
    }
}

function Get-OrchestratorProcessMetadata {
    param(
        [Parameter(Mandatory = $true)]
        [int]$ProcessId
    )

    try {
        return Get-CimInstance Win32_Process -Filter "ProcessId = $ProcessId" -ErrorAction Stop
    } catch {
        return $null
    }
}

function Test-OrchestratorProcessCommandLine {
    param(
        [string]$ProcessName,
        [string]$CommandLine
    )

    if (-not $commandLine) {
        return $false
    }

    $normalizedName = $ProcessName.ToLowerInvariant()
    $isPythonProcess = $normalizedName -eq "python.exe" -or $normalizedName -eq "pythonw.exe"
    if (-not $isPythonProcess) {
        return $false
    }

    $normalizedCommandLine = $commandLine.ToLowerInvariant()
    $hasUvicornCommand = $normalizedCommandLine -match "(^|\\s)-m\\s+uvicorn(\\s|$)"
    $hasOrchestratorApp = $normalizedCommandLine.Contains(
        $Script:OrchestratorCommandLinePattern.ToLowerInvariant()
    )
    return ($hasUvicornCommand -and $hasOrchestratorApp)
}

function Test-OrchestratorProcessRecordMatches {
    param(
        [AllowNull()]
        [object]$ProcessRecord
    )

    if (-not $ProcessRecord) {
        return $false
    }

    return Test-OrchestratorProcessCommandLine `
        -ProcessName ([string]$ProcessRecord.Name) `
        -CommandLine ([string]$ProcessRecord.CommandLine)
}

function Find-OrchestratorProcess {
    try {
        return Get-CimInstance Win32_Process `
            -Filter "Name = 'python.exe' OR Name = 'pythonw.exe'" `
            -ErrorAction Stop |
            Where-Object { Test-OrchestratorProcessRecordMatches $_ } |
            Select-Object -First 1
    } catch {
        return $null
    }
}

function Test-OrchestratorAlreadyRunning {
    param(
        [Parameter(Mandatory = $true)]
        [string]$PidFile,
        [string]$HealthUrl = $Script:OrchestratorHealthUrl,
        [int]$StartupGraceSeconds = 90
    )

    $process = Get-OrchestratorProcessFromPidFile -PidFile $PidFile
    if ($process) {
        if (Test-OrchestratorHealthy -Url $HealthUrl) {
            return $true
        }

        $startedRecently = $false
        try {
            $startedRecently = $process.StartTime -ge (Get-Date).AddSeconds(-1 * $StartupGraceSeconds)
        } catch {
        }

        if ($startedRecently) {
            $processRecord = Get-OrchestratorProcessMetadata -ProcessId $process.Id
            if (Test-OrchestratorProcessRecordMatches -ProcessRecord $processRecord) {
                return $true
            }
        }
    }

    if ((-not $process) -and (Test-Path $PidFile)) {
        Remove-Item $PidFile -Force -ErrorAction SilentlyContinue
    }

    if (-not (Test-OrchestratorHealthy -Url $HealthUrl)) {
        return $false
    }

    $fallbackProcess = Find-OrchestratorProcess
    if ($fallbackProcess) {
        Set-Content -Path $PidFile -Value $fallbackProcess.ProcessId -Encoding ASCII -ErrorAction SilentlyContinue
        return $true
    }

    return $false
}
