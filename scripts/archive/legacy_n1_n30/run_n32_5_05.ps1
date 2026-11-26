# N32.5.05 (exploration): 2-seed replication (20261121, 20261122) of ModalityEdgeHop Q3 (<= 500 CPU).
# Pre-registered: plans/reference/DESIGN-N32-5-05_20261125.md (D-20261125-542).
$ErrorActionPreference = "Stop"
Set-Location E:\Seed
ruff check scripts/training/n32_5_05_modality_edge_hop_rep.py
if ($LASTEXITCODE -ne 0) { throw "ruff failed" }
python scripts/training/n31_guard.py N32.5.05 --seeds 20261121,20261122 --no-record
if ($LASTEXITCODE -ne 0) { throw "preflight failed" }
python scripts/training/n32_5_05_modality_edge_hop_rep.py --smoke
if ($LASTEXITCODE -ne 0) { throw "smoke failed or projection exceeds 500 CPU (see reports/n32_5_05_smoke.json)" }
python scripts/training/n32_5_05_modality_edge_hop_rep.py
if ($LASTEXITCODE -ne 0) { throw "N32.5.05 failed (see reports/n32_5_05_learnability.json)" }
Write-Host "N32.5.05 done: reports/n32_5_05_learnability.json"
