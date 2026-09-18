# M5 R2-D8 probe 结项：机制层 8/9 全过，**唯一失败为「loss 抖动」门** —— 按合同路由回评审

日期：2026-09-18。状态：**probe 未过门（1 条）⇒ 依合同 §4 不进 matched，路由回评审**。
合同：[M5_R2_D8_SCALE_CONTRACT_FROZEN_20260918](M5_R2_D8_SCALE_CONTRACT_FROZEN_20260918.md)；
前置：[实现门报告](M5_R2_D8_IMPLEMENTATION_GATE_20260918.md)。
报告：`reports/r2_d7_probe_T1_d8v3.json`；checkpoint：`reports/r2_d8_checkpoints/T1/`。
（v2 train 的重放报告在 `reports/r2_d8_replay/`，不覆盖 D7 冻结报告。）

## §1 结果（train-only，seed 20260917，30 epochs，lr 0.01，microbatch 8）

| 门 | 值 | 判 |
|---|---|---|
| `copy_supported_m1_ge_0_90` | **1.0** | pass |
| `multibyte_m1_ge_0_90` | **576/576 = 1.0** | pass |
| `singlebyte_m1_ge_0_90` | **576/576 = 1.0** | pass |
| `copy_value_prob_ge_0_90` | **1.0** | pass |
| `misbind_copy_prob_le_0_50` | **0.0426** | pass |
| `no_late_collapse_gt_0_10` | pass | pass |
| `preflight_passed` / `within_wall_cap` | pass / pass（519.5 s）| pass |
| **`loss_increases_bounded_15pct`** | **false** | **FAIL** |

`outcome = failed`（9 门中 8 门通过）。

## §2 失败项的性质：**抖动，不是发散**（关键判读）

| epoch | 0 | 5 | 10 | 15 | 20 | 25 | 30 |
|---|---|---|---|---|---|---|---|
| loss | 4.0729 | 0.0847 | **0.1149** | 0.0865 | 0.0833 | 0.0828 | **0.0826** |

- **唯一超限的是 epoch 5→10 的 +35.7%**；
- 它**立即回落**（epoch 15 = 0.0865），且**最终 0.0826 < epoch 5 的 0.0847** ⇒ **净下降、末值最优**；
- **对照 D7（v2，174 条）**：loss `[3.826, 0.118, 0.120, 0.119, 0.118, 0.119, 0.116]`
  —— 同样存在 `0.118→0.120`（+1.8%）的小幅上升，**恰好未触发 15% 阈值**。

⇒ **同一门在 7.34× 语料下更易触发**：更大语料 ⇒ 每 epoch 内 160 个 micro-batch（D7 为 22 个）
⇒ 段间评估的方差更大。**这是门与规模的交互，不是机制失稳。**

## §3 机制层读数（**v3 全面不劣于 D7，多处更强**）

| 量 | v3（1278 条）| D7（174 条）|
|---|---|---|
| `copy_supported_M1` | **1.0** | 1.0 |
| multibyte 值行 | **576/576** | 72/72 |
| copy 概率 intact / misbound | **1.0 / 0.0426** | 1.0 / 0.0696 |
| **`bias_end`（induction 被使用程度）** | **0.6549** | 0.178 |
| per-shape：fact / negation / same_opening_fact | **1.0 / 1.0 / 1.0** | 1.0 / 1.0 / 1.0 |
| **combination_same** | **1.0（31/31）** | 0.714（5/7）|
| combination_different | 0.0（0/31）| 0.286（2/7）|
| unknown / same_opening_unknown | 0.0 / 0.0 | 0.0 / 0.0 |

⇒ 扩尺度后**复制链条更强**（bias 3.7×、combo_same 满分），**未见任何机制层退化**。

## §4 路由（**按合同，未擅自放行**）

合同 §4 明确：「**probe 败 → 按 D7 路由回实现门/评审，不进 matched**」。
因此本包**停在 probe**，**未启动 matched dev**。可供裁决的选项：

- **(a) 严格按合同**：以 probe 败论，回实现门/评审 —— 需要裁决的是
  「`loss_increases_bounded_15pct` 这门在 7.34× 语料下是否仍适用」；
- **(b) 判抖动可接受并放行 matched**：**需要用户特批**（合同门不得由助手自行放宽）；
  若采纳，建议**在 matched 报告中显式标注门为"经特批放宽"**，并在结项时说明放宽依据；
- **(c) 补证**：用**另两个 seed** 重跑 probe（train-only，各约 9 min），
  看失败是否**跨 seed 复现** —— 若 3/3 失败则是"门与规模不匹配"的稳健证据，
  若 1/3 失败则更像偶然抖动。

**助手建议：(c) → 再裁决 (a)/(b)** —— 它把"抖动 vs 不稳定"这一问题用最小成本钉死，
且不需要动任何冻结门。

## §5 边界

- probe 是 **train-only 可学性**检查，**不是能力主张**；
- `dev_read=false`、`final_read=false`；`growth_admitted=false`、`can_promote=false`；
- 本文件不改变任何冻结门，也不解除合同；matched 未启动。
