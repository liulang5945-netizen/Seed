# N32.4.01 (exploration): Q3 learnability + Q2 retrain control. Card <= 800 CPU.
# Pre-registered: plans/reference/DESIGN-N32-4-01_20261125.md (D-20261125-523).
$ErrorActionPreference = "Stop"
Set-Location E:\Seed
ruff check scripts/training/n32_4_01_q3_learnability.py
if ($LASTEXITCODE -ne 0) { throw "ruff failed" }
python scripts/training/n31_guard.py N32.4.01 --seeds 20261120,20261121,20261122 --no-record
if ($LASTEXITCODE -ne 0) { throw "preflight failed" }
python scripts/training/n32_4_01_q3_learnability.py --smoke
if ($LASTEXITCODE -ne 0) { throw "smoke failed or projection exceeds 800 CPU (see reports/n32_4_01_smoke.json)" }
python scripts/training/n32_4_01_q3_learnability.py
if ($LASTEXITCODE -ne 0) { throw "N32.4.01 engineering_failure (see reports/n32_4_01_learnability.json)" }
Write-Host "N32.4.01 done: reports/n32_4_01_learnability.json"
