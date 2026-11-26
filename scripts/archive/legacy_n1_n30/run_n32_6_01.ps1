# N32.6.01 (exploration): R1 Pilot -- Shared ModalityEdgeHop (S) vs external RT columns (<= 580 CPU).
# Pre-registered: plans/reference/DESIGN-N32-6-01_20261125.md (D-20261125-544).
$ErrorActionPreference = "Stop"
Set-Location E:\Seed
ruff check scripts/training/n32_6_01_r1_edge_hop.py tests/test_n32_6_01_r1_pipeline_20261125.py
if ($LASTEXITCODE -ne 0) { throw "ruff failed" }
python -m pytest tests/test_n32_6_01_r1_pipeline_20261125.py tests/test_modality_edge_hop_20261125.py -q
if ($LASTEXITCODE -ne 0) { throw "unit tests failed" }
python scripts/training/n31_guard.py N32.6.01 --seeds 20261120 --no-record
if ($LASTEXITCODE -ne 0) { throw "preflight failed" }
python scripts/training/n32_6_01_r1_edge_hop.py --smoke
if ($LASTEXITCODE -ne 0) { throw "smoke failed or projection exceeds 580 CPU (see reports/n32_6_01_smoke.json)" }
python scripts/training/n32_6_01_r1_edge_hop.py
if ($LASTEXITCODE -ne 0) { throw "N32.6.01 failed (see reports/n32_6_01_r1_pilot.json)" }
Write-Host "N32.6.01 done: reports/n32_6_01_r1_pilot.json"
