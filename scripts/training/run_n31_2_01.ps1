# N31.2.01 exploration runs (leak-free unseen-pair generalisation): aug arm, exploration seeds only.
$ErrorActionPreference = "Stop"
Set-Location E:\Seed
foreach ($seed in 20261120, 20261121, 20261122) {
  python scripts/training/n31_2_01_unseen_leakfree.py $seed aug
  if ($LASTEXITCODE -ne 0) { throw "N31.2.01 failed: seed=$seed" }
}
Write-Host "N31.2.01 done: reports/n31_2_01_aug_seed2026112{0,1,2}.json"
