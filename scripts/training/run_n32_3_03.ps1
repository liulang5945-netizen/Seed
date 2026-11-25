# N32.3.03 (exploration): Q2 learnability v3 -- standardised inputs, name relabelling, MLP gate.
# Pre-registered: plans/reference/DESIGN-N32-3-03_20261125.md (D-20261125-521). Card <= 442 CPU.
$ErrorActionPreference = "Stop"
Set-Location E:\Seed
ruff check scripts/training/n32_3_03_learnability_v3.py
if ($LASTEXITCODE -ne 0) { throw "ruff failed" }
python scripts/training/n31_guard.py N32.3.03 --seeds 20261120,20261121,20261122 --no-record
if ($LASTEXITCODE -ne 0) { throw "preflight failed" }
python scripts/training/n32_3_03_learnability_v3.py --smoke
if ($LASTEXITCODE -ne 0) { throw "smoke failed or projection exceeds 442 CPU (see reports/n32_3_03_smoke.json)" }
python scripts/training/n32_3_03_learnability_v3.py
if ($LASTEXITCODE -ne 0) { throw "N32.3.03 engineering_failure (see reports/n32_3_03_learnability.json)" }
Write-Host "N32.3.03 done: reports/n32_3_03_learnability.json"
