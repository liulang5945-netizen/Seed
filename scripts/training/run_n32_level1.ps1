# Level-1 gate for the N32 backbones (v1 and v2). Must pass before N32 main training.
# D-20261124-495 (v1, N32.0) / D-20261124-496 (v2 lifecycle, release-01).
$ErrorActionPreference = "Stop"
Set-Location E:\Seed
python -m pytest tests/test_gru_rt_level1_20261124.py tests/test_gru_rt_lifecycle_20261124.py -q
if ($LASTEXITCODE -ne 0) { throw "level-1 tests failed" }
ruff check taiji/gru_rt.py taiji/gru_rt_lifecycle.py tests/test_gru_rt_level1_20261124.py tests/test_gru_rt_lifecycle_20261124.py
if ($LASTEXITCODE -ne 0) { throw "ruff failed" }
python -m mypy taiji/gru_rt.py taiji/gru_rt_lifecycle.py
if ($LASTEXITCODE -ne 0) { throw "mypy failed" }
Write-Host "N32 backbone level-1 PASS (v1 + v2)"
