# N32.5.04 (exploration): ModalityEdgeHop Q3 learnability check (Option P1, <= 320 CPU).
# Pre-registered: plans/reference/DESIGN-N32-5-04_20261125.md (D-20261125-540).
$ErrorActionPreference = "Stop"
Set-Location E:\Seed
ruff check taiji/modality_edge_hop.py scripts/training/n32_5_04_modality_edge_hop_q3.py tests/test_modality_edge_hop_20261125.py
if ($LASTEXITCODE -ne 0) { throw "ruff failed" }
python -m pytest tests/test_modality_edge_hop_20261125.py tests/test_n32_5_01_fast_encode_20261125.py -q
if ($LASTEXITCODE -ne 0) { throw "unit tests failed" }
python scripts/training/n31_guard.py N32.5.04 --seeds 20261120 --no-record
if ($LASTEXITCODE -ne 0) { throw "preflight failed" }
python scripts/training/n32_5_04_modality_edge_hop_q3.py --smoke
if ($LASTEXITCODE -ne 0) { throw "smoke failed or projection exceeds 320 CPU (see reports/n32_5_04_smoke.json)" }
python scripts/training/n32_5_04_modality_edge_hop_q3.py
if ($LASTEXITCODE -ne 0) { throw "N32.5.04 failed (see reports/n32_5_04_learnability.json)" }
Write-Host "N32.5.04 done: reports/n32_5_04_learnability.json"
