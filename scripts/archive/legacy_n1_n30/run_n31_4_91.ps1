# N31.4.91: the single permitted re-attempt of N31.4.90 (H3 inconclusive). Pre-registered in
# plans/reference/DESIGN-N31-4-91_20261124.md. Backup confirmation seeds, H3 only.
$ErrorActionPreference = "Stop"
Set-Location E:\Seed
python scripts/training/diag_m3_hard_20261123.py `
  --seeds 20261211,20261212,20261213 --levels H3 --lr-sweep --lrs 0.007,0.015,0.03 `
  --fit 1024 --arms selective,gru_gate --cap-cpu 1500 `
  --out reports/n31_4_91_m3_hard_h3_backup_20261124.json
if ($LASTEXITCODE -ne 0) { throw "N31.4.91 failed" }
Write-Host "N31.4.91 done: reports/n31_4_91_m3_hard_h3_backup_20261124.json"
