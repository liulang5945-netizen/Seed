# M5 R2-D8 用户裁决记录：采纳 (b) —— 判 loss 门失败为单 seed 抖动，放行 matched

日期：2026-09-18。状态：**已裁决**。
用户指令：在补证 (c) 完成后，对 (a)/(b) 选项回复「**b**」。

## §1 裁决内容

**采纳 (b)**：`probe` 阶段唯一失败的门 `loss_increases_bounded_15pct`
（仅 seed 20260917 触发，另两 seed 跳幅 +0.39% / +0.71%）**判定为单 seed 偶然抖动**，
**放行 matched dev**。

## §2 裁决依据（补证报告已记录）

1. **预置判据落入 1/3 分支** —— [补证](M5_R2_D8_PROBE_MULTISEED_20260918.md) §2
   在跑之前就写定"3/3 = 门与规模不匹配；1/3 = 偶然抖动"；
2. **该门与机制层无关** —— 机制门三 seed 全过（`copy_M1`、multibyte `576/576`、copy 概率、
   misbind、`bias_end`）；
3. **不按单点抖动去改跨 8 个包沿用的冻结门**（代价与收益不成比例）。

## §3 执行约束（**已落实**）

裁决 (b) 附带两项硬约束，均已在实现中落实：

1. **matched 报告须显式标注该门经特批按抖动处理** ⇒ 已在
   `eval_taiji_r2_d7_matched_dev.py` 的报告字段中加入
   `loss_increases_gate_note`：`"probe-stage gate: treated as single-seed jitter (1/3)
   per user adjudication (c) -> (b); see plans/reference/M5_R2_D8_PROBE_MULTISEED_20260918.md"`；
2. **附补证表作为依据** ⇒ 报告另含 `multibyte_full_rate_per_seed`，
   补证表见上述文档 §1。

## §4 实现调整（**最小侵入，v1/v2 路径不变**）

`scripts/training/eval_taiji_r2_d7_matched_dev.py`：

- 新增 `FIXTURES["v3"]` 与 `TRAIN_CORPUS_DIGESTS`（v3 digest 锚
  `a73703de2759491c923066e24beac56f910d820b365ed6148e062f12565c3554`）；
- 新增 CLI：`--fixture {v1,v2,v3}`、`--out-report`、`--checkpoint-root`（默认值原样
  ⇒ **v1 路径行为逐位不变**，dry check 已验证 v1 digest 锚仍匹配）；
- **G5 按 D8 合同 §5 重新定义**（「≥2 seed 的 `full_rate > 0.10` 且最差 ≥0」），
  **但仅对 `--fixture v3` 生效** —— v1/v2 保持 D7 原定义（该包已结项，不得回改）；
- **dev 仍读 v1**：v3 的 dev 与 v1 **逐字节相同**（dry check 已验证），
  故 dev 评测口径与 D7 完全一致，改动只落在 train 来源。

## §5 下一步

**matched dev（v3 train × 3 seeds）已按合同 §5 启动**，门为
K1（多字值 `full_rate` 均值 ≥0.30，主门）、K2（`M4_flip` ≥0.50）、G1–G4/G6 同 D7、
**G5 按 D8 定义**；结果按合同 §5 的四行路由表裁决，combo/unknown 单列。
