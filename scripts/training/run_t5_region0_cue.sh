#!/usr/bin/env bash
# T5：读出只吃区 0（掩码式，默认关），跑到 5M。
# 预注册：plans/reference/M5_R2_T5_REGION0_CUE_PREREG_20260924.md
cd /e/Seed || exit 1
exec "C:/Users/23747/AppData/Local/Programs/Python/Python312/python.exe" \
  scripts/training/run_taiji_r2_t1_t2_short_runs.py \
  --arm-name t5-r0-cue \
  --budget 5000000 \
  --measure-every 250000 \
  --corpus data/simple_zh/simple_zh_texts.jsonl \
  --set predictive_context_region0_only=1 \
  --out-dir output/taiji_r2_t5/t5_r0_cue
