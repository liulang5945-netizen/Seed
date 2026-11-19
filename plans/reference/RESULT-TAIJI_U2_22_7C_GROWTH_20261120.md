# 结果卡：22.7c 投影训练下的最小生长收益（`growth_gain_absent`；新生儿是死单元）

> 文档状态：[结果卡] — 2026-11-20。合同：[`taiji_u2_22_7c_growth_joint24_20261120.json`](../manifests/taiji_u2_22_7c_growth_joint24_20261120.json)（`status=frozen`、`training_authority=owner:standing-authorization:2026-11-20#recommendation-default`、`result.closed=true`）。回执：[run](../../reports/taiji_u2_22_7c_growth_run_20261120.json)／smoke。**0 check/final。**

## 1. 协议（读前冻结，修正 22.7b 的投影缺陷后）

joint24 任务；三臂等步数 1,500；**每步 `project_parameters_()`**；出生前置门＝出生前后同条件 logits 逐位一致；门＝净益 ≥0.05／配对 LB>0／3 seed 同向／旧能力 ≤0.02／恢复 exact／引用清理。

## 2. 读数（`97.12 ≤ 帽 250 CPU`）

| 臂 | seed 0 | seed 1 | seed 2 |
|---|---|---|---|
| G0_growth | 0.78125 | 0.625 | 0.625 |
| C1_no_growth | 0.78125 | 0.625 | 0.625 |
| C2_preallocated | 0.78125 | 0.625 | 0.625 |

- 出生前置门 **通过**：3/3 seed `birth_neutral_diff = 0.0`；旧能力损伤 `0.0`；恢复 exact；引用清理通过。
- **生长净益 `[0,0,0]`、均值 `0.0`、配对 LB `0.0` ⇒ `growth_gain_absent`**；三臂逐项**完全相同**。

## 3. 归因：出生单元是「死单元」

三臂结果完全一致 ⇒ 新生儿对 readout 与梯度均无贡献。检查参数化：`with_neutral_birth` 同时置 `r_new=0` 与 `p_new=0`、`c_new=0`：
- `drive_new = r_new·x = 0` ⇒ `z_new = tanh(0) = 0` ⇒ `activity_new = 0`；
- `p_new` 的梯度 ∝ 该组 activity（=0）⇒ `p_new` 永为 0；`r_new` 的梯度 ∝ `p_new`（=0）⇒ `r_new` 永为 0。

**双零初始化把输入/输出通路同时关闭，新生儿永久不可学**——「中性出生」在行为上中性，但机制上没有可塑性入口，增益必然为 0。这不是「生长无用」的证据，是**出生初始化设计缺陷**。

## 4. 修正方向（须新卡，不复活本卡）

- 保持「出生瞬时输出不变」：`p_new=0`（输出通路关闭，中性成立）。
- 同时给**可学入口**：`r_new` 取小随机值（或继承父单位行向量），使 `activity_new≠0`，从而 `p_new` 第一步即获非零梯度、随后 `r_new` 也可学。
- 前置门升级为两条：① 出生前后 logits 逐位一致；② 出生后 N 步内 `r_new`/`p_new` 梯度非零且权重发生可测变化（活单元证据）。
- 通过后才可重测生长收益（22.7d）。

## 5. 边界

本卡不签生长收益或损失结论（死单元混杂）；`growth_gain_absent` 只说明当前出生机制下增益为 0。不修改 `u2_neural_units.py` 源码（pin 保持），修复走新卡与初始化协议。
