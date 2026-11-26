# N32.5.01 (exploration): Lightweight Q3 GRU learnability check (Route X, <= 250 CPU).
# Pre-registered: plans/reference/DESIGN-N32-5-01_20261125.md (D-20261125-534).
$ErrorActionPreference = "Stop"
Set-Location E:\Seed
ruff check scripts/training/n32_5_01_q3_gru_learnability.py tests/test_n32_5_01_fast_encode_20261125.py
if ($LASTEXITCODE -ne 0) { throw "ruff failed" }
python -m pytest tests/test_n32_5_01_fast_encode_20261125.py -q
if ($LASTEXITCODE -ne 0) { throw "fast encode unit test failed" }
python scripts/training/n31_guard.py N32.5.01 --seeds 20261120,20261121,20261122 --no-record
if ($LASTEXITCODE -ne 0) { throw "preflight failed" }
python scripts/training/n32_5_01_q3_gru_learnability.py --smoke
if ($LASTEXITCODE -ne 0) { throw "smoke failed or projection exceeds 250 CPU (see reports/n32_5_01_smoke.json)" }
python scripts/training/n32_5_01_q3_gru_learnability.py
if ($LASTEXITCODE -ne 0) { throw "N32.5.01 failed (see reports/n32_5_01_learnability.json)" }
Write-Host "N32.5.01 done: reports/n32_5_01_learnability.json"
