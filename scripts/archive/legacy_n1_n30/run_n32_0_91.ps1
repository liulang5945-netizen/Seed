# N32.0.91 (confirmation): release-02 substitute under core requirements §2.1 v4.
# Pre-registered: plans/reference/DESIGN-N32-0-91_20261125.md (D-20261125-503). Card <= 600 CPU.
# Config u48/1200 is fixed by the design. Confirmation seeds are void after one read: no rerun.
$ErrorActionPreference = "Stop"
Set-Location E:\Seed
python -m pytest tests/test_gru_rt_level1_20261124.py tests/test_gru_rt_lifecycle_20261124.py -q
if ($LASTEXITCODE -ne 0) { throw "level-1 tests failed: do not train" }
ruff check scripts/training/n31_guard.py scripts/training/n32_0_rt_admission.py
if ($LASTEXITCODE -ne 0) { throw "ruff failed" }
python scripts/training/n31_guard.py N32.0.91 --seeds 20261301,20261302,20261303 --no-record
if ($LASTEXITCODE -ne 0) { throw "preflight failed" }
foreach ($s in 20261301, 20261302, 20261303) {
    python scripts/training/n32_0_rt_admission.py N32.0.91 $s
    if ($LASTEXITCODE -ne 0) { throw "N32.0.91 seed $s failed (see reports/n32_0_91_seed$s.json)" }
}
python scripts/training/n32_0_rt_admission.py N32.0.91 --aggregate
Write-Host "N32.0.91 done: reports/n32_0_91_summary.json"
