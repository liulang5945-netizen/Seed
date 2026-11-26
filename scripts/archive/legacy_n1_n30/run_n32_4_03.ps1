# N32.4.03 (exploration): Q3 learnability v3 -- 48k-step MLP gate, GRU 3 seeds, Q2 retrain. Card <= 800 CPU.
# Pre-registered: plans/reference/DESIGN-N32-4-03_20261125.md (D-20261125-529, option B1).
$ErrorActionPreference = "Stop"
Set-Location E:\Seed
ruff check scripts/training/n32_4_03_q3_learnability.py
if ($LASTEXITCODE -ne 0) { throw "ruff failed" }
python scripts/training/n31_guard.py N32.4.03 --seeds 20261120,20261121,20261122 --no-record
if ($LASTEXITCODE -ne 0) { throw "preflight failed" }
python scripts/training/n32_4_03_q3_learnability.py --smoke
if ($LASTEXITCODE -ne 0) { throw "smoke failed or projection exceeds 800 CPU (see reports/n32_4_03_smoke.json)" }
python scripts/training/n32_4_03_q3_learnability.py
if ($LASTEXITCODE -ne 0) { throw "N32.4.03 engineering_failure (see reports/n32_4_03_learnability.json)" }
Write-Host "N32.4.03 done: reports/n32_4_03_learnability.json"
