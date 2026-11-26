# N32.4.04 (exploration): Q3 learnability v4 -- 64k-step MLP gate (0.90 unchanged), GRU 3 seeds, Q2 retrain. Card <= 700 CPU.
# Pre-registered: plans/reference/DESIGN-N32-4-04_20261125.md (D-20261125-531, option D2).
$ErrorActionPreference = "Stop"
Set-Location E:\Seed
ruff check scripts/training/n32_4_04_q3_learnability.py
if ($LASTEXITCODE -ne 0) { throw "ruff failed" }
python scripts/training/n31_guard.py N32.4.04 --seeds 20261120,20261121,20261122 --no-record
if ($LASTEXITCODE -ne 0) { throw "preflight failed" }
python scripts/training/n32_4_04_q3_learnability.py --smoke
if ($LASTEXITCODE -ne 0) { throw "smoke failed or projection exceeds 700 CPU (see reports/n32_4_04_smoke.json)" }
python scripts/training/n32_4_04_q3_learnability.py
if ($LASTEXITCODE -ne 0) { throw "N32.4.04 engineering_failure (see reports/n32_4_04_learnability.json)" }
Write-Host "N32.4.04 done: reports/n32_4_04_learnability.json"
