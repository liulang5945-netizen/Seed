# N32.3.02 (diagnostic): pre-encoded learnability v2, MLP reference, G1-G3. Card <= 600 CPU.
# Pre-registered: plans/reference/DESIGN-N32-3-02_20261125.md (D-20261125-518).
$ErrorActionPreference = "Stop"
Set-Location E:\Seed
python scripts/training/n31_guard.py N32.3.02 --seeds 20261120 --no-record
if ($LASTEXITCODE -ne 0) { throw "preflight failed" }
ruff check scripts/training/n32_3_02_learnability_v2.py
if ($LASTEXITCODE -ne 0) { throw "ruff failed" }
python scripts/training/n32_3_02_learnability_v2.py --smoke
if ($LASTEXITCODE -ne 0) { throw "smoke failed" }
$smoke = Get-Content reports/n32_3_02_precheck.json -Raw | ConvertFrom-Json
$enc = [double]$smoke.cost.encode_cpu
$train = [double]$smoke.cost.cpu - $enc
$proj = [math]::Round($enc + $train * ((2*2500 + 2500 + 2*1500) / (4*40)), 1)
Write-Host "smoke ok; projected card CPU $proj (encode $enc, train-of-smoke $train)"
if ($proj -gt 600) { throw "projected CPU $proj exceeds the 600 card budget; stop and report" }
python scripts/training/n32_3_02_learnability_v2.py
if ($LASTEXITCODE -ne 0) { throw "diagnostic exit is not an allowed pre-registered exit" }
Write-Host "N32.3.02 done: reports/n32_3_02_precheck.json"
