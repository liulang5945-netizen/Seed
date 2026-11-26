# N32.0.90 (confirmation): release-02 substitute. Runs only after N32.0.01 = signal_positive.
# Pre-registered: plans/reference/DESIGN-N32-0-90_20261124.md (D-20261124-495). Card <= 600 CPU.
# Confirmation seeds are recorded by the guard and can be read only once: do not rerun.
$ErrorActionPreference = "Stop"
Set-Location E:\Seed
$sum = Get-Content reports/n32_0_01_summary.json -Raw | ConvertFrom-Json
if ($sum.exit -ne "signal_positive") { throw "N32.0.01 exit is $($sum.exit); N32.0.90 not allowed" }
Write-Host "config: units=$($sum.chosen.units) steps=$($sum.chosen.steps)"
python scripts/training/n31_guard.py N32.0.90 --seeds 20261201,20261202,20261203 --no-record
if ($LASTEXITCODE -ne 0) { throw "preflight failed" }
foreach ($s in 20261201, 20261202, 20261203) {
    python scripts/training/n32_0_rt_admission.py N32.0.90 $s
    if ($LASTEXITCODE -ne 0) { throw "N32.0.90 seed $s failed (see reports/n32_0_90_seed$s.json)" }
}
python scripts/training/n32_0_rt_admission.py N32.0.90 --aggregate
Write-Host "N32.0.90 done: reports/n32_0_90_summary.json"
