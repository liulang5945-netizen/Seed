# N32.6.90 (confirmation): joint task-agnostic S_joint on sealed v2 confirm split. Card <= 700 CPU.
# Pre-registered: plans/reference/DESIGN-N32-6-90_20261125.md (D-20261125-550).
$ErrorActionPreference = "Stop"
Set-Location E:\Seed
ruff check scripts/training/n32_6_90_confirm.py
if ($LASTEXITCODE -ne 0) { throw "ruff failed" }
python scripts/training/n32_6_90_confirm.py --smoke
if ($LASTEXITCODE -ne 0) { throw "smoke failed or projection exceeds 700 CPU (see reports/n32_6_90_smoke.json)" }
python scripts/training/n32_6_90_confirm.py
if ($LASTEXITCODE -ne 0) { throw "N32.6.90 engineering_failure (see reports/n32_6_90_confirm.json)" }
Write-Host "N32.6.90 done: reports/n32_6_90_confirm.json"
