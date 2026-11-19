# 设计卡（草稿）：22.7b 最小生长收益卡（多属性联合 24 类任务）

> 文档状态：[设计卡/待授权] — 2026-11-20。新卡，非复活 22.7 旧窗（22.7 仍按 `growth_not_needed_in_scope` 结案，D-20261119-36）。任务族依据：[joint24 可行性回执](../../reports/taiji_u2_22_7_joint24_20261120.json)（`joint24_family_candidate`：16/32 单位中间带、32→64 头空间 `+0.1875`）。合同：[`taiji_u2_22_7b_growth_joint24_20261120.json`](../manifests/taiji_u2_22_7b_growth_joint24_20261120.json)（`designed_pending_owner`、`training_authority=null`）。

## 1. 任务面与依据

- 任务：first-A **多属性联合判定**——标签＝查询实体颜色（6）× 位置词（4）＝ **24 类**；特征＝适配器 v3（48 维原始媒体统计，无标签泄露）。
- 依据：容量扫描 `8/16/32/64` 单位＝`0.1875/0.5469/0.6875/0.8750`，**通道中间带 + 容量头空间**同时成立（噪声族只到 `+0.0156`，已否决用于生长）。
- 本卡是**设计选择**：任务面扩展需在结果卡显式标注，不伪装成既有 first-A 任务；check/final 不参与。

## 2. 三臂最小对照（其余六臂 out-of-scope）

| 臂 | 操作 |
|---|---|
| `G0_growth` | 32 单位热启动 → **step 750 中性出生**（`with_neutral_birth`）→ 33 单位继续训练，总步数不变 |
| `C1_no_growth` | 32 单位、同总步数、无出生 |
| `C2_preallocated` | 33 单位从 step 0 开始、同总步数 |

- 配平：**等总优化步**；FLOPs 比（33/32）须报告且 `≤1.05`；只按参数量配平不合格。
- G1/G2/G3（随机/置换门、GradMax 方向）与 G4/G5/G6 明确列为 **out-of-scope**，需另行裁定后补测；本卡不据此签完整 22.7 七臂结论。

## 3. 判据与出口

- 生长净益 `G0 − max(C1, C2) ≥ 0.05`；fit 32 world 逐项配对单侧 95% 下界 `>0`；3 seed 同向。
- 旧能力保持：基础 4 类任务准确率下降单侧上界 `≤0.02`（与主任务同适配器、同媒体）。
- 恢复：出生后 capsule 保存→独立进程重载 logits 逐位一致；引用清理（无悬空 parent_ids）。
- 出口顺序：`engineering_failure→not_admitted→budget_stop→credit_insufficient→retention_breach→recovery_failure→growth_gain_absent→growth_gain_pass`。
- 预算：**包价 200、帽 200 CPU**；`0 check/final`。

## 4. 边界

- 合同 `training_authority=null`，`--run` fail-closed；训练须 owner 显式放行或按常设授权记录后回溯。
- 本卡不签「生长在一般任务上的收益」，只签「该 24 类任务面、该三臂对照下的生长净益」；负结果按 `growth_gain_absent` 收束，禁同范围调参重开。
