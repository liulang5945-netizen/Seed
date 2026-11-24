# N31.2.02: failure-mode breakdown, seed 20261121 (weakest), aug arm. Final planned card of N31.2.
$ErrorActionPreference = "Stop"
Set-Location E:\Seed
python scripts/training/n31_2_02_unseen_breakdown.py 20261121 aug
if ($LASTEXITCODE -ne 0) { throw "N31.2.02 failed" }
Write-Host "N31.2.02 done: reports/n31_2_02_aug_seed20261121.json"
