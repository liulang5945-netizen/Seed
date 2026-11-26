# N32.5.02 (exploration): SlotHopGRU Q3 two-hop learnability check (Option 1, <= 350 CPU).
# Pre-registered: plans/reference/DESIGN-N32-5-02_20261125.md (D-20261125-536).
$ErrorActionPreference = "Stop"
Set-Location E:\Seed
ruff check taiji/slot_memory_gru.py scripts/training/n32_5_02_slot_hop_q3.py tests/test_slot_memory_gru_20261125.py
if ($LASTEXITCODE -ne 0) { throw "ruff failed" }
python -m pytest tests/test_slot_memory_gru_20261125.py tests/test_n32_5_01_fast_encode_20261125.py -q
if ($LASTEXITCODE -ne 0) { throw "unit tests failed" }
python scripts/training/n31_guard.py N32.5.02 --seeds 20261120,20261121,20261122 --no-record
if ($LASTEXITCODE -ne 0) { throw "preflight failed" }
python scripts/training/n32_5_02_slot_hop_q3.py --smoke
if ($LASTEXITCODE -ne 0) { throw "smoke failed or projection exceeds 350 CPU (see reports/n32_5_02_smoke.json)" }
python scripts/training/n32_5_02_slot_hop_q3.py
if ($LASTEXITCODE -ne 0) { throw "N32.5.02 failed (see reports/n32_5_02_learnability.json)" }
Write-Host "N32.5.02 done: reports/n32_5_02_learnability.json"
