# N31.1.02 (exploration): task-agnostic learned routing on GRU+read_gate.
# Pre-registered: plans/reference/DESIGN-N31-1-02_20261124.md (D-20261124-490).
# Card budget <= 1,500 CPU (exploration sub-budget of the N31.1 exception).
$ErrorActionPreference = "Stop"
Set-Location E:\Seed
python scripts/training/n31_guard.py N31.1.02 --seeds 20261120,20261121,20261122 --no-record
if ($LASTEXITCODE -ne 0) { throw "N31.1.02 preflight failed" }
ruff check scripts/training/n31_1_02_routed_gru.py scripts/training/n31_guard.py
if ($LASTEXITCODE -ne 0) { throw "ruff failed" }
python scripts/training/n31_1_02_routed_gru.py 20261120 --smoke
if ($LASTEXITCODE -ne 0) { throw "N31.1.02 smoke failed" }
$smoke = Get-Content reports/n31_1_02_smoke_seed20261120.json -Raw | ConvertFrom-Json
$arms = $smoke.planned_arms
if (-not $arms) { throw "projected CPU exceeds the 1,500 card budget; stop and report" }
Write-Host "smoke ok; projection all=$($smoke.projection.all_arms) no_c3=$($smoke.projection.without_c3); arms=$arms"
foreach ($s in 20261120, 20261121, 20261122) {
    python scripts/training/n31_1_02_routed_gru.py $s --arms $arms
    if ($LASTEXITCODE -ne 0) { throw "N31.1.02 seed $s failed (see reports/n31_1_02_seed$s.json)" }
}
python scripts/training/n31_1_02_routed_gru.py --aggregate
Write-Host "N31.1.02 done: reports/n31_1_02_seed*.json, reports/n31_1_02_summary.json"
