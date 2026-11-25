# N32.3.01 (pre-check): single-task learnability on v2 data. Card <= 600 CPU.
# Pre-registered: plans/reference/DESIGN-N32-3-01_20261125.md (D-20261125-515).
$ErrorActionPreference = "Stop"
Set-Location E:\Seed
python scripts/training/n31_guard.py N32.3.01 --seeds 20261120 --no-record
if ($LASTEXITCODE -ne 0) { throw "preflight failed" }
ruff check scripts/training/n32_3_01_learnability_precheck.py scripts/training/n32_2_03_s_vs_rt.py
if ($LASTEXITCODE -ne 0) { throw "ruff failed" }
python scripts/training/n32_3_01_learnability_precheck.py --smoke
if ($LASTEXITCODE -ne 0) { throw "smoke failed" }
$smoke = Get-Content reports/n32_3_01_precheck.json -Raw | ConvertFrom-Json
$proj = [math]::Round($smoke.cost.cpu * (16200 / 200), 1)
Write-Host "smoke ok; projected card CPU $proj"
if ($proj -gt 600) { throw "projected CPU $proj exceeds the 600 card budget; stop and report" }
python scripts/training/n32_3_01_learnability_precheck.py
if ($LASTEXITCODE -ne 0) { throw "pre-check exit is not an allowed pre-registered exit" }
Write-Host "N32.3.01 done: reports/n32_3_01_precheck.json"
