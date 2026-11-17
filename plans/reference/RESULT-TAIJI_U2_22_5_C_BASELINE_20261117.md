# Plan C：同成本基线身份、FLOPs 配平与 22.5.0 重冻（待训练授权）

> 卡号：`22.5.0.C`｜日期：2026-11-17｜状态：**`c_identity_frozen_flops_parity_available`**｜性质：**零训练身份/配平/重冻**，尚未开 A 父门、未训练、无能力。
> 依据：[22.5.0 收束卡](CLOSEOUT-TAIJI_U2_22_5_0_GATE_20261117.md)、[22.4.4 价格](RESULT-TAIJI_U2_22_4_4_PRICING_20261117.md)、[计划 §22.5 前置](PLAN-TAIJI-CLOSED_LOOP_DEVELOPMENT_20261110.md)。
> 机器件：[C 身份清单](../manifests/taiji_u2_22_4_4_baseline_identity_c_20261117.json)（sha256 `49431a36…8f6e`）、[C 配平](../manifests/taiji_u2_22_5_c_parity_20261117.json)、[重冻合同](../manifests/taiji_u2_22_5_0_refreeze_c_20261117.json)（sha256 `2a566fb3…3fc2`）。

## 1 新基线身份（不碰 U2）

- `architecture_id`：**`weight_tied_transformer_sequence`**；同一 transformer block **权重共享应用 K=3 次**。
- **参数量 21,977／35 张量**，对锚点 22,742 **+3.3638%**，仍在 ±10% 带 [20,467, 25,016]。
- **前向+反向 FLOPs 2,960,256**；初始化同 seed 两次构造 digest 相同；`state_tensors=0`（无跨步 KV 持久化）。
- 旧 `transformer_sequence` 身份清单保持原字节；U2 臂、22.3／22.4 工程父、分区、价格均不改。

## 2 配平读数

| 量 | 值 | 门 | 结论 |
|---|---:|---:|---|
| FLOPs（U2÷C） | **10.4365×** | 256/24 = **10.6667×** | **FLOPs 配平可用** |
| 参数量偏离 | **+3.3638%** | ±10% | 通过 |
| wall（U2÷C） | **135.99×** | 未作为门 | 披露，不声称时间配平 |

- 冻结的配平量是 **FLOPs**（计划 `:378` 允许二选一）。时间比 135.99× 只作限制披露；A 门必须写成 **FLOPs-matched**，不得写成等 wall、等 CPU 或等参数量。

## 3 剂量包络的结

- 在 **256 更新/架构** 上限内：基线用满 256 次只相当于 **24 次 U2** 的 FLOPs；U2 用满 256 次则需要基线 **2,671 次** > 256。
- 因此当前 256 上限下，FLOPs 匹配的 U2 剂量只有 **24 次**——这正是首次 A 清单的最低匹配剂量，且 P3.10 已记它低于有效课程最低剂量。**配平解决了，但剂量表达不了**；直接开 A 门只会落 `budget_cannot_express`。
- 这不是"把便宜臂放大取胜"：基线 2,671 次与 U2 256 次是**同一 FLOPs 预算**；只是现有 256 上限是按更新数写的，不是按 FLOPs 写的。

## 4 重冻内容（只改一处）

- **基线更新上限：256 → 2,671／seed**；**U2 仍 256**；共同 FLOPs **7,909,060,608／seed**。
- 不改：U2 上限、check 曝光上限 8 与分配、final seal、反馈权利、数据分区、U2 价格。
- A 门预计成本 **约 417.6 CPU**（3 seed ×256 U2 更新×0.453125×1.20；基线 CPU 低于计时粒度，wall 另计），远低于 22.5 的 16,000 计划。
- check① 会在 A 门使用 **1/2**；本卡不读 check、不读 final。
- 合同 `training_authorized=false`、`requires_owner_authorization=true`。

## 5 不签

A 父、训练、B、check②、22.6／22.7、架构优劣、任何能力。**也不写"U2 被证否"**；C 只是把"配平不可达"修成"FLOPs 可配平、但需重冻剂量"。

## 6 唯一后继

请 owner 决定是否**批准这份重冻并授权 A 门训练**。批准后执行顺序：checkpoint 保存/恢复预检 → A 门 3 seed（FLOPs-matched 256/2,671）→ check①（1/2）→ 按 A 门出口结案。未批准前：0 训练、0 能力、0 新父；22.5.1／22.6／22.7 保持锁定。
