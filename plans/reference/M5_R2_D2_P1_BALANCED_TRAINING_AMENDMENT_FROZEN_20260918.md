# R2-D2 修订三：P1 形状均衡训练（A2 图，冻结版）

2026-09-18。状态：**已冻结**（用户在 [A 族评审备忘](M5_R2_D2_A_FAMILY_REVIEW_MEMO_20260918.md)裁决 P1 后提交即冻结）。
定位：这是**训练分布合同**，不是图修订；graph 不改、D1 仪器不改、判据不改。

## §1 假设与干预

- 已定位现象：失败答案全部以「不」开头（55/174），首字节训练先验为颜色 96 : 不 55 : 未 16 : 相 7；H-A2 除第一步外 teacher-forced 目标概率 1.0。
- 假设：首字节贪心错误是类别先验压倒内容梯度的**决策层**问题；把每 epoch 的形状边缘分布均衡，内容梯度可在不改变图的前提下把「不」首字节拉过贪心阈值。
- 图：**沿用稳定的 graph v4 A2**（per_position+copy，常量 start，82,658 参数；不用 v5——避免与其晚期失稳混淆，也直接检验「仅训练分布」这一单变量）。

## §2 均衡采样器（冻结算法，只影响训练）

1. 每个 epoch 构造 174 个训练样本（与原 epoch 更新数相同），按 **sorted(shape)** 顺序在 7 个形状间分配配额：174 = 24×7 + 6，前 6 个形状配额 25、最后一个 24。
2. 每个形状的样本从该形状 train 组内用 `random.Random(SEED + epoch)` 独立有放回抽取；顺序为形状外循环、抽取内循环。
3. 评估、probe 门、checkpoint preflight、dev/final 全部仍用**原始未均衡**数据；均衡只改每 epoch 喂给 train_step 的批次序列。
4. 其余超参不变：Adam lr=0.01、seed 20260917、30 epochs、wall 20 min、preflight 先行。

## §3 门（冻结；评估在未改动 train split 上）

1. copy-supported（fact+negation+sof，144 题）train M1 **≥0.90**（与前两次 probe 同门）。
2. **negation M1 必须 >0** 且 fact/sof 不回落（各自 ≥0.90；H-A2 它们为 1.0）。
3. loss 有限、相对初值下降；30 epoch 末不得出现 H-A3 式失稳（末 5 epoch M1 不得回撤超过 0.10）。
4. preflight 通过。
- probe 通过才允许 matched dev：臂为 A2-balanced vs C1-copy-balanced（broadcast+均衡，同采样器），3 seeds；G1–G6 阈值同 H-A2 修订，surface 天花板与 D1 digest 不变。
- 描述项：unknown/combo（含 combo_flip）逐形状分，作为 P2/P3 路由证据。

## §4 预承诺（一次，不叠加技巧）

- 本干预**只做一次**：不过 ⇒ 决策层假设（至少其均衡版本）被否，按评审备忘转 P2（表达容量）或 P3（绑定），不得再调采样比例、加权方式、lr、epoch 或回 v5 重试。
- 不读 dev/final；不改 D1 仪器与门；产物 `growth_admitted=false`、`can_promote=false`；train-only 成功不构成 L2 或内容能力主张。
