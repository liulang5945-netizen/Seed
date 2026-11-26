#!/usr/bin/env bash
# T4 轮 1：四笔 fabric 写入一起 ÷5（配置等价于 learn_scale≈0.2），跑到 5M。
# 预注册：plans/reference/M5_R2_T4_SLOW_ALL_PREREG_20260924.md
cd /e/Seed || exit 1
exec "C:/Users/23747/AppData/Local/Programs/Python/Python312/python.exe" \
  scripts/training/run_taiji_r2_t1_t2_short_runs.py \
  --arm-name t4-slow-all \
  --budget 5000000 \
  --measure-every 250000 \
  --corpus data/simple_zh/simple_zh_texts.jsonl \
  --set predictive_learning_rate=0.005 \
  --set transition_learning_rate=0.0024 \
  --set lateral_learning_rate=0.004 \
  --set synapse_decay=2e-6 \
  --out-dir output/taiji_r2_t4/t4_slow_all
