# N32.6.02 (exploration): Joint interleaved Q1+Q2+Q3 diagnostic on ModalityEdgeHop (<= 300 CPU).
# Pre-registered: plans/reference/DESIGN-N32-6-02_20261125.md (D-20261125-546).
$ErrorActionPreference = "Stop"
Set-Location E:\Seed
ruff check scripts/training/n32_6_02_r1_joint_diag.py
if ($LASTEXITCODE -ne 0) { throw "ruff failed" }
python scripts/training/n31_guard.py N32.6.02 --seeds 20261120 --no-record
if ($LASTEXITCODE -ne 0) { throw "preflight failed" }
python scripts/training/n32_6_02_r1_joint_diag.py --smoke
if ($LASTEXITCODE -ne 0) { throw "smoke failed or projection exceeds 300 CPU (see reports/n32_6_02_smoke.json)" }
python scripts/training/n32_6_02_r1_joint_diag.py
if ($LASTEXITCODE -ne 0) { throw "N32.6.02 failed (see reports/n32_6_02_joint_diag.json)" }
Write-Host "N32.6.02 done: reports/n32_6_02_joint_diag.json"
