# 结果卡：22.7d 活入口出生（`engineering_failure`；p 通路已活，单步 r 梯度门过严）

> 文档状态：[结果卡] — 2026-11-20。合同：[`taiji_u2_22_7d_growth_joint24_20261120.json`](../manifests/taiji_u2_22_7d_growth_joint24_20261120.json)（`status=frozen`、`training_authority=owner:standing-authorization:2026-11-20#recommendation-default`、`result.closed=true`）。回执：[run](../../reports/taiji_u2_22_7d_growth_run_20261120.json)／smoke。**0 check/final。**

## 1. 协议（读前冻结）

22.7c 的投影训练＋新出生实现 [`taiji/u2_neural_birth.py`](../../taiji/u2_neural_birth.py)：**`p_new=0`（出生瞬时输出不变）＋`r_new` 继承父单位行向量**（活输入通路）；三臂等步数 1,500；出生后 1 步探针记录 `p/r` 梯度、权重变化、新生单元活动；前置门＝出生 logits 逐位一致 **且** 活单元证据（梯度/权重变化/活动均 >0）。

## 2. 读数（`96.28 ≤ 帽 250 CPU`）

- 出生中性：3/3 seed `birth_neutral_diff=0.0`；旧能力损伤 0；恢复 exact；引用清理 ✓。
- **活入口部分成立**：`p_grad_norm 0.0034–0.0172 > 0`、`p_change_norm 0.0028 > 0`、`newborn_activity_max 0.39–0.47 > 0`；
- **`r_grad_norm = 0`、`r_change_norm = 0`**：第一步时 `p_new=0`，故 `r_new` 梯度 ∝ `p_new` 仍为 0（第 2 步起才会活）。
- 判 **`engineering_failure`**（活单元门未全过，按预注册出口先于收益判定）；同期收益 `[−0.0313, 0, −0.0313]`、均值 −0.0208、配对 LB −0.0938（**不可作为生长净益结论**，因门未过）。

## 3. 判读

1. **死单元问题已修好一半**：`p_new` 可学、新生单元有非零活动 ⇒ 输入/输出通路打开，不再永久死单元。
2. **门的设计过严**：单步探针要求 `r_new` 梯度非零不成立是**预期物理**（`r` 梯度依赖 `p_new`，后者第一步后才离开 0）；应改为「N=5 步内 p 梯度非零、r 梯度在第 2 步起非零、p/r 权重变化均 >0」。
3. 本卡不签生长收益/损失；`engineering_failure` 只说明前置门口径需修正。

## 4. 后继（22.7e）

- 新卡：同一活入口出生，探针改 N=5 步（记录 p 第 1 步 grad、r 第 2+ 步 grad、5 步后 p/r 权重变化），仍保持「出生 logits 逐位一致」；
- 若 22.7e 活单元门通过且净益 ≥0.05／配对 LB>0／3 seed 同向 ⇒ `growth_gain_pass`；否则 `growth_gain_absent`（并披露 +1 单位容量是否足够）；
- 21 步口径内的步数配平：探针步计入总步数（1,500），不额外增加预算。
