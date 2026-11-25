# N32.2.03 (exploration): task-agnostic shared S vs external-column RT on v2 data.
# Pre-registered: plans/reference/DESIGN-N32-2-03_20261125.md (D-20261125-511). Card <= 1,100 CPU.
$ErrorActionPreference = "Stop"
Set-Location E:\Seed
python scripts/training/n31_guard.py N32.2.03 --seeds 20261120,20261121,20261122,20261123,20261124 --no-record
if ($LASTEXITCODE -ne 0) { throw "preflight failed" }
ruff check scripts/training/n32_2_03_s_vs_rt.py scripts/training/n31_guard.py
if ($LASTEXITCODE -ne 0) { throw "ruff failed" }
python scripts/training/n32_2_03_s_vs_rt.py 20261120 --smoke
if ($LASTEXITCODE -ne 0) { throw "smoke failed" }
$smoke = Get-Content reports/n32_2_03_seed20261120_smoke.json -Raw | ConvertFrom-Json
$proj = [math]::Round($smoke.cost.cpu * 32 * 5, 1)
Write-Host "smoke ok; projected card CPU $proj"
if ($proj -gt 1100) { throw "projected CPU $proj exceeds the 1,100 card budget; stop and report" }
foreach ($s in 20261120, 20261121, 20261122, 20261123, 20261124) {
    python scripts/training/n32_2_03_s_vs_rt.py $s
    if ($LASTEXITCODE -ne 0) { throw "N32.2.03 seed $s failed (see reports/n32_2_03_seed$s.json)" }
}
python scripts/training/n32_2_03_s_vs_rt.py --aggregate
Write-Host "N32.2.03 done: reports/n32_2_03_summary.json"
