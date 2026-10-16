param(
    [Parameter(Mandatory = $true)]
    [ValidateSet('probe', 'handoff', 'run')]
    [string]$Mode
)

$ErrorActionPreference = 'Stop'
$RepositoryRoot = [System.IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..\..'))
$Controller = Join-Path $RepositoryRoot 'scripts\training\run_taiji_n1_transition_factorial_paired_run86.py'
$FrozenRunner = Join-Path $RepositoryRoot 'scripts\training\run_taiji_n1_transition_factorial_paired.py'
$Python = (Get-Command python.exe -ErrorAction Stop).Source
$Pythonw = Join-Path (Split-Path -Parent $Python) 'pythonw.exe'
$ReportRoot = Join-Path $RepositoryRoot 'reports'
$CheckpointRoot = Join-Path $RepositoryRoot 'data\checkpoints\run86b'

if (-not (Test-Path -LiteralPath $Pythonw -PathType Leaf)) {
    throw "pythonw.exe missing beside Python interpreter: $Pythonw"
}

if ($Mode -eq 'probe') {
    $Stem = Join-Path $ReportRoot 'taiji_n1_transition_factorial_run86_runtime_preflight_scheduled_retry2'
    $TaskName = 'Seed-N1-Run86-Probe-20261016'
    $Arguments = "-u `"$Controller`" --synthetic-heartbeat"
    $Artifacts = @(
        "${Stem}.json",
        "${Stem}.jsonl",
        "${Stem}_launcher.json",
        "${Stem}_returned.json",
        "${Stem}_worker.json",
        "${Stem}.stdout.log",
        "${Stem}.stderr.log"
    )
}
elseif ($Mode -eq 'handoff') {
    $Stem = Join-Path $ReportRoot 'taiji_n1_transition_factorial_run86b_return_preflight'
    $TaskName = 'Seed-N1-Run86B-Handoff-20261016'
    $Arguments = '-u ' + [char]34 + $Controller + [char]34 + ' --run-handoff-probe'
    $Artifacts = @(
        "${Stem}.json",
        "${Stem}_launcher.json",
        "${Stem}_launcher_returned.json",
        "${Stem}_worker.json",
        "${Stem}.stdout.log",
        "${Stem}.stderr.log"
    )
}
else {
    $Stem = Join-Path $ReportRoot 'taiji_n1_transition_factorial_run86b'
    $TaskName = 'Seed-N1-Run86B-20261016'
    $Arguments = "-u `"$Controller`" --execute-approved"
    $Artifacts = @(
        "${Stem}_attempt.json",
        "${Stem}.json",
        "${Stem}_completion.json",
        "${Stem}_launcher.json",
        "${Stem}_launcher_returned.json",
        "${Stem}_worker.json",
        "${Stem}.stdout.log",
        "${Stem}.stderr.log"
    )
    if (Test-Path -LiteralPath $CheckpointRoot) {
        throw "run86 checkpoint root already exists: $CheckpointRoot"
    }
    if (Get-ChildItem -LiteralPath $ReportRoot -Filter 'taiji_n1_transition_factorial_run86_s*.json.gz' -ErrorAction SilentlyContinue) {
        throw 'run86 shard family already exists'
    }
}

foreach ($Path in $Artifacts) {
    if (Test-Path -LiteralPath $Path) {
        throw "run86 artifact already exists: $Path"
    }
}
if (Get-ScheduledTask -TaskName $TaskName -ErrorAction SilentlyContinue) {
    throw "run86 task already exists and will not be overwritten: $TaskName"
}

$RegisteredUtc = [DateTime]::UtcNow.ToString('o')
$ControllerHash = (Get-FileHash -LiteralPath $Controller -Algorithm SHA256).Hash.ToLowerInvariant()
$RunnerHash = (Get-FileHash -LiteralPath $FrozenRunner -Algorithm SHA256).Hash.ToLowerInvariant()
$LauncherHash = (Get-FileHash -LiteralPath $PSCommandPath -Algorithm SHA256).Hash.ToLowerInvariant()
$Action = New-ScheduledTaskAction -Execute $Pythonw -Argument $Arguments -WorkingDirectory $RepositoryRoot
$Principal = New-ScheduledTaskPrincipal `
    -UserId ([System.Security.Principal.WindowsIdentity]::GetCurrent().Name) `
    -LogonType Interactive `
    -RunLevel Limited
$Settings = New-ScheduledTaskSettingsSet `
    -ExecutionTimeLimit (New-TimeSpan -Hours 5) `
    -MultipleInstances IgnoreNew

$Registered = $false
$Started = $false
$Manifest = $null
try {
    Register-ScheduledTask `
        -TaskName $TaskName `
        -Action $Action `
        -Principal $Principal `
        -Settings $Settings `
        -Description 'On-demand N1 run86 worker; no triggers; local interactive token.' `
        -ErrorAction Stop | Out-Null
    $Registered = $true
    $Task = Get-ScheduledTask -TaskName $TaskName -ErrorAction Stop
    $TriggerCount = if ($null -eq $Task.Triggers) { 0 } else { @($Task.Triggers).Count }
    if ($TriggerCount -ne 0) {
        throw "run86 task unexpectedly has $TriggerCount triggers"
    }

    $Manifest = "${Stem}_launcher.json"
    if ($Mode -eq 'probe') {
        $Returned = "${Stem}_returned.json"
    }
    else {
        $Returned = "${Stem}_launcher_returned.json"
    }
    $Record = [ordered]@{
        schema = 'n1-transition-factorial-run86-task-launch-v1'
        mode = $Mode
        task_name = $TaskName
        trigger_count = $TriggerCount
        principal_logon_type = 'Interactive'
        run_level = 'Limited'
        python_path = $Pythonw
        arguments = $Arguments
        repository_root = $RepositoryRoot
        registered_utc = $RegisteredUtc
        controller_path = $Controller
        controller_sha256 = $ControllerHash
        runner_path = $FrozenRunner
        runner_sha256 = $RunnerHash
        launcher_path = $PSCommandPath
        launcher_sha256 = $LauncherHash
        stdout_path = "${Stem}.stdout.log"
        stderr_path = "${Stem}.stderr.log"
    }
    $Bytes = [System.Text.UTF8Encoding]::new($false).GetBytes(($Record | ConvertTo-Json -Depth 5))
    $Stream = [System.IO.File]::Open($Manifest, [System.IO.FileMode]::CreateNew, [System.IO.FileAccess]::Write, [System.IO.FileShare]::None)
    try {
        $Stream.Write($Bytes, 0, $Bytes.Length)
        $Stream.Flush($true)
    }
    finally {
        $Stream.Dispose()
    }

    $RequestedUtc = [DateTime]::UtcNow.ToString('o')
    Start-ScheduledTask -TaskName $TaskName -ErrorAction Stop
    $Started = $true
    $ReturningUtc = [DateTime]::UtcNow.ToString('o')
    $ReturnRecord = [ordered]@{
        schema = 'n1-transition-factorial-run86-task-return-v1'
        task_name = $TaskName
        start_succeeded = $true
        start_requested_utc = $RequestedUtc
        launcher_returning_utc = $ReturningUtc
    }
    $ReturnBytes = [System.Text.UTF8Encoding]::new($false).GetBytes(($ReturnRecord | ConvertTo-Json -Depth 5))
    $ReturnStream = [System.IO.File]::Open($Returned, [System.IO.FileMode]::CreateNew, [System.IO.FileAccess]::Write, [System.IO.FileShare]::None)
    try {
        $ReturnStream.Write($ReturnBytes, 0, $ReturnBytes.Length)
        $ReturnStream.Flush($true)
    }
    finally {
        $ReturnStream.Dispose()
    }
    $Record | ConvertTo-Json -Depth 5 -Compress
}
catch {
    if ($Registered -and -not $Started) {
        Unregister-ScheduledTask -TaskName $TaskName -Confirm:$false -ErrorAction SilentlyContinue
    }
    throw
}
