# N32.2.02 (zero training): data contract v2 for R1 (scenes, shortcut baselines, gates).
# Pre-registered: plans/reference/DESIGN-N32-2-02_20261125.md (D-20261125-509).
$ErrorActionPreference = "Stop"
Set-Location E:\Seed
ruff check taiji/native_composition_v2.py tests/test_composition_v2_20261125.py scripts/training/n32_2_02_composition_v2_data.py
if ($LASTEXITCODE -ne 0) { throw "ruff failed" }
python -m mypy taiji/native_composition_v2.py
if ($LASTEXITCODE -ne 0) { throw "mypy failed" }
python scripts/training/n32_2_02_composition_v2_data.py
$dataExit = $LASTEXITCODE
python -m pytest tests/test_composition_v2_20261125.py -q
if ($LASTEXITCODE -ne 0) { throw "v2 data tests failed: exit is engineering_failure" }
if ($dataExit -ne 0) { throw "N32.2.02 exit is not data_ready (see reports/n32_2_02_data.json)" }
Write-Host "N32.2.02 done: data_ready; reports/n32_2_02_data.json"
