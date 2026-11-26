param(
    [Parameter(Mandatory = $true)]
    [ValidateSet('run')]
    [string]$Mode
)

$ErrorActionPreference = 'Stop'
$RepositoryRoot = [System.IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..\..'))
$Controller = Join-Path $RepositoryRoot 'scripts\training\run_taiji_n1_transition_factorial_lr_decay_run88.py'
$FrozenRunner = Join-Path $RepositoryRoot 'scripts\training\run_taiji_n1_transition_factorial_paired.py'
$Preflight = Join-Path $RepositoryRoot 'reports\taiji_n1_transition_factorial_lr_decay_run88_preflight.json'
$Binding = Join-Path $RepositoryRoot 'reports\taiji_n1_transition_factorial_lr_decay_run88_preflight_binding.json'
$Report = Join-Path $RepositoryRoot 'reports\taiji_n1_transition_factorial_lr_decay_run88.json'
$Attempt = Join-Path $RepositoryRoot 'reports\taiji_n1_transition_factorial_lr_decay_run88_attempt.json'
$Completion = Join-Path $RepositoryRoot 'reports\taiji_n1_transition_factorial_lr_decay_run88_completion.json'
$Manifest = Join-Path $RepositoryRoot 'reports\taiji_n1_transition_factorial_lr_decay_run88_launcher.json'
$Returned = Join-Path $RepositoryRoot 'reports\taiji_n1_transition_factorial_lr_decay_run88_launcher_returned.json'
$Worker = Join-Path $RepositoryRoot 'reports\taiji_n1_transition_factorial_lr_decay_run88_worker.json'
$Stdout = Join-Path $RepositoryRoot 'reports\taiji_n1_transition_factorial_lr_decay_run88.stdout.log'
$Stderr = Join-Path $RepositoryRoot 'reports\taiji_n1_transition_factorial_lr_decay_run88.stderr.log'
$CheckpointRoot = Join-Path $RepositoryRoot 'data\checkpoints\run88\n1_transition_factorial_lr_decay_20261016'
$TaskName = 'Seed-N1-LRDecay-Run88-20261016'
$Python = (Get-Command python.exe -ErrorAction Stop).Source
$Pythonw = Join-Path (Split-Path -Parent $Python) 'pythonw.exe'

if (-not (Test-Path -LiteralPath $Pythonw -PathType Leaf)) {
    throw "pythonw.exe missing beside Python interpreter: $Pythonw"
}
foreach ($Path in @($Preflight, $Binding, $CheckpointRoot)) {
    if (-not (Test-Path -LiteralPath $Path)) {
        throw "run88 checkpoint preflight artifact is missing: $Path"
    }
}
$PreflightRecord = Get-Content -LiteralPath $Preflight -Raw | ConvertFrom-Json
$BindingRecord = Get-Content -LiteralPath $Binding -Raw | ConvertFrom-Json
$PreflightHash = (Get-FileHash -LiteralPath $Preflight -Algorithm SHA256).Hash.ToLowerInvariant()
$BindingHash = (Get-FileHash -LiteralPath $Binding -Algorithm SHA256).Hash.ToLowerInvariant()
if (-not $PreflightRecord.complete -or $BindingRecord.preflight_report_sha256 -ne $PreflightHash) {
    throw 'run88 checkpoint preflight binding does not verify'
}

$Artifacts = @($Attempt, $Report, $Completion, $Manifest, $Returned, $Worker, $Stdout, $Stderr)
foreach ($Path in $Artifacts) {
    if (Test-Path -LiteralPath $Path) {
        throw "run88 formal artifact already exists and will not be overwritten: $Path"
    }
}
if (Get-ChildItem -LiteralPath (Join-Path $RepositoryRoot 'reports') -Filter 'taiji_n1_transition_factorial_lr_decay_run88_s*.json.gz' -ErrorAction SilentlyContinue) {
    throw 'run88 shard family already exists'
}
if (Get-ScheduledTask -TaskName $TaskName -ErrorAction SilentlyContinue) {
    throw "run88 task already exists and will not be overwritten: $TaskName"
}

$ControllerHash = (Get-FileHash -LiteralPath $Controller -Algorithm SHA256).Hash.ToLowerInvariant()
$RunnerHash = (Get-FileHash -LiteralPath $FrozenRunner -Algorithm SHA256).Hash.ToLowerInvariant()
$LauncherHash = (Get-FileHash -LiteralPath $PSCommandPath -Algorithm SHA256).Hash.ToLowerInvariant()
if (
    $PreflightRecord.code_identity.controller_sha256 -ne $ControllerHash -or
    $PreflightRecord.code_identity.frozen_runner_sha256 -ne $RunnerHash -or
    $PreflightRecord.code_identity.launcher_sha256 -ne $LauncherHash
) {
    throw 'run88 source code changed after checkpoint preflight'
}
$Arguments = '-u ' + [char]34 + $Controller + [char]34 + ' --execute-approved'
$Action = New-ScheduledTaskAction -Execute $Pythonw -Argument $Arguments -WorkingDirectory $RepositoryRoot
$Principal = New-ScheduledTaskPrincipal `
    -UserId ([System.Security.Principal.WindowsIdentity]::GetCurrent().Name) `
    -LogonType Interactive `
    -RunLevel Limited
$Settings = New-ScheduledTaskSettingsSet `
    -ExecutionTimeLimit (New-TimeSpan -Hours 3) `
    -MultipleInstances IgnoreNew

$Registered = $false
$Started = $false
try {
    Register-ScheduledTask `
        -TaskName $TaskName `
        -Action $Action `
        -Principal $Principal `
        -Settings $Settings `
        -Description 'On-demand N1 run88 final-cycle learning-rate experiment; no triggers; local interactive token.' `
        -ErrorAction Stop | Out-Null
    $Registered = $true
    $Task = Get-ScheduledTask -TaskName $TaskName -ErrorAction Stop
    $TriggerCount = if ($null -eq $Task.Triggers) { 0 } else { @($Task.Triggers).Count }
    if ($TriggerCount -ne 0) {
        throw "run88 task unexpectedly has $TriggerCount triggers"
    }
    $Record = [ordered]@{
        schema = 'n1-transition-factorial-lr-decay-run88-task-launch-v1'
        mode = 'run'
        task_name = $TaskName
        trigger_count = $TriggerCount
        principal_logon_type = 'Interactive'
        run_level = 'Limited'
        python_path = $Pythonw
        arguments = $Arguments
        repository_root = $RepositoryRoot
        registered_utc = [DateTime]::UtcNow.ToString('o')
        controller_path = $Controller
        controller_sha256 = $ControllerHash
        frozen_runner_path = $FrozenRunner
        frozen_runner_sha256 = $RunnerHash
        launcher_path = $PSCommandPath
        launcher_sha256 = $LauncherHash
        preflight_report_sha256 = $PreflightHash
        preflight_binding_sha256 = $BindingHash
        stdout_path = $Stdout
        stderr_path = $Stderr
    }
    $Bytes = [System.Text.UTF8Encoding]::new($false).GetBytes(($Record | ConvertTo-Json -Depth 6))
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
    $ReturnRecord = [ordered]@{
        schema = 'n1-transition-factorial-lr-decay-run88-task-return-v1'
        task_name = $TaskName
        start_succeeded = $true
        start_requested_utc = $RequestedUtc
        launcher_returning_utc = [DateTime]::UtcNow.ToString('o')
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
    $Record | ConvertTo-Json -Depth 6 -Compress
}
catch {
    if ($Registered -and -not $Started) {
        Unregister-ScheduledTask -TaskName $TaskName -Confirm:$false -ErrorAction SilentlyContinue
    }
    throw
}
