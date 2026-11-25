# N32.2.01 (zero training): 23.1 tick encoding, data pool, rule baselines, gates.
# Pre-registered: plans/reference/DESIGN-N32-2-01_20261125.md (D-20261125-506).
$ErrorActionPreference = "Stop"
Set-Location E:\Seed
ruff check taiji/composition_encoding.py tests/test_composition_encoding_20261125.py scripts/training/n32_2_01_composition_data.py scripts/training/n31_guard.py
if ($LASTEXITCODE -ne 0) { throw "ruff failed" }
python -m mypy taiji/composition_encoding.py
if ($LASTEXITCODE -ne 0) { throw "mypy failed" }
python -m pytest tests/test_composition_encoding_20261125.py -q
if ($LASTEXITCODE -ne 0) { throw "encoding tests failed" }
python scripts/training/n32_2_01_composition_data.py
if ($LASTEXITCODE -ne 0) { throw "N32.2.01 exit is not data_ready (see reports/n32_2_01_data.json)" }
Write-Host "N32.2.01 done: reports/n32_2_01_data.json"
