# N32.0.01 (exploration): RT columns on GRU+read_gate; pick the cheapest config.
# Pre-registered: plans/reference/DESIGN-N32-0-01_20261124.md (D-20261124-495). Card <= 600 CPU.
$ErrorActionPreference = "Stop"
Set-Location E:\Seed
python -m pytest tests/test_gru_rt_level1_20261124.py -q
if ($LASTEXITCODE -ne 0) { throw "level-1 tests failed: do not train" }
ruff check taiji/gru_rt.py tests/test_gru_rt_level1_20261124.py scripts/training/n32_0_rt_admission.py
if ($LASTEXITCODE -ne 0) { throw "ruff failed" }
python -m mypy taiji/gru_rt.py
if ($LASTEXITCODE -ne 0) { throw "mypy failed" }
python scripts/training/n31_guard.py N32.0.01 --seeds 20261120,20261121,20261122 --no-record
if ($LASTEXITCODE -ne 0) { throw "preflight failed" }
python scripts/training/n32_0_rt_admission.py N32.0.01 20261120 --smoke
if ($LASTEXITCODE -ne 0) { throw "smoke failed" }
$smoke = Get-Content reports/n32_0_01_smoke_seed20261120.json -Raw | ConvertFrom-Json
if (-not $smoke.projection_ok) { throw "projected CPU $($smoke.projection.card_cpu) exceeds 600; stop" }
Write-Host "smoke ok; projected card CPU $($smoke.projection.card_cpu)"
foreach ($s in 20261120, 20261121, 20261122) {
    python scripts/training/n32_0_rt_admission.py N32.0.01 $s
    if ($LASTEXITCODE -ne 0) { throw "N32.0.01 seed $s failed (see reports/n32_0_01_seed$s.json)" }
}
python scripts/training/n32_0_rt_admission.py N32.0.01 --aggregate
Write-Host "N32.0.01 done: reports/n32_0_01_summary.json"
