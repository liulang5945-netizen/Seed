# N32.5.03 (exploration): Plain GRU u192 / 18k-step Q3 ceiling check (A2-d, <= 850 CPU).
# Pre-registered: plans/reference/DESIGN-N32-5-03_20261125.md (D-20261125-538).
$ErrorActionPreference = "Stop"
Set-Location E:\Seed
ruff check scripts/training/n32_5_03_gru192_q3.py
if ($LASTEXITCODE -ne 0) { throw "ruff failed" }
python scripts/training/n31_guard.py N32.5.03 --seeds 20261120 --no-record
if ($LASTEXITCODE -ne 0) { throw "preflight failed" }
python scripts/training/n32_5_03_gru192_q3.py --smoke
if ($LASTEXITCODE -ne 0) { throw "smoke failed or projection exceeds 850 CPU (see reports/n32_5_03_smoke.json)" }
python scripts/training/n32_5_03_gru192_q3.py
if ($LASTEXITCODE -ne 0) { throw "N32.5.03 failed (see reports/n32_5_03_learnability.json)" }
Write-Host "N32.5.03 done: reports/n32_5_03_learnability.json"
