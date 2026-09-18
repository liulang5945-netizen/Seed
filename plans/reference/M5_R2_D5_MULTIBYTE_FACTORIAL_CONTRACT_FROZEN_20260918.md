# M5 R2-D5：多字节持续 copy 的 2×2 因子包（数据覆盖 × copy 驻留，冻结版）

2026-09-18。状态：**已冻结**（用户在 [D4 结项评审](M5_R2_D4_MATCHED_DEV_REVIEW_MEMO_20260918.md) 裁决「2×2 对比包」后提交即冻结）。前置：D4 结论——copy 值监督使绑定建立（train 值位置 copy 概率=1.0、错位即崩 0.0417），dev 失败模式为「值读取正确但多字节值仅复制首字节即截断」；候选根因两个且各覆盖失败模式一半：**E 覆盖**（train 值域全单字节，6-byte 值从未进入训练分布）与 **M 偏置**（copy 每步独立寻址，无「留在同一行继续」的归纳偏置）。

## §1 假设（冻结）

> **H-E（覆盖）**：train 值域混入双字色后，持续复制的第二字节获得 CE 监督，v6 图（每步独立寻址）即可学到多字节 copy 并在 dev 多字值上泛化（dev 值域仍全新，泛化只能来自规律）。
> **H-M（偏置）**：copy 驻留先验（上一步命中位置的下一字节获得加性 bonus）即使 train 无多字节值，也能把「逐字节独立 argmax」变成「行内序列延续」，直接改善 dev 双字值完成率。
> **交互**：两者可能都必要（A11 过而单因子不过）或其一充分。本包的主目的=一次读数裁决两因子的相对贡献，而非堆到最复杂臂。

## §2 因子规格（冻结）

### 2.1 数据因子 E：train 值域替换（fixture v2，dev/final 逐字节不变）

- train 颜色池 `("蓝","绿","红","黄","青","紫")` → `("黄","青","紫","琥珀","珊瑚","翡翠")`：3 单字 + 3 双字（6-byte 值）。被弃单字（蓝/绿/红）不再出现于任何 train 行；新六字（琥/珀/珊/瑚/翡/翠）经核对不与 train/dev/final 对象池、模板固定文字、其余值池任何字符重合。
- 行数 174、七形状、模板、翻转对结构、行序（ORDER_SEED=0xD1）全部不变；fact_flip 的 colors[0]/[1] 链接随新池自动为「黄 vs 青」。
- 新文件 `tests/fixtures/r2_d1_measurement_v2.jsonl`：train 174 行=替换版；dev 98 + final 94 行与 v1 **逐字节相同**（合同数据报告断言 dev/final 子集 digest 与 v1 一致）。D1 仪器（scorer、dev/final、表面基线）**不动**；matched dev 评测仍读 v1 路径（与 v2 的 dev 子集相同，二选一以 v1 为准）。
- 等暴露口径：行数/epochs/微批/更新数全不变——E 因子的唯一变化是值长度分布。

### 2.2 机制因子 M：graph v7 copy 驻留（+1 参数）

- 在 v6 基础上新增可训练标量 `copy_persist_bias`（初值 **0.0**，1 参数：96,034→96,035）与配置标志 `copy_persistence: bool = False`。
- **指针规则（冻结）**：每生成步 t，取本步跨头平均 copy 分布的 `j* = argmax p_copy`（`zero_read` 或无 copy 分量时指针=None）。步 t+1 的每个头寻址 logits 在 softmax 前：`logits[(j*+1) mod L] += copy_persist_bias`，仅当 `j*+1 < len(entries)`（行尾不绕回，越界不加）。episode 首步无 bonus。
- teacher-forced 路径指针由**真实字节序列**的逐步 argmax 更新；自由生成路径由贪心 argmax 更新（与 generate 一致）。lesion（entry_rotation）作用于 begin_episode 的 entries 索引空间，指针规则在同一空间自然成立；`bias=0` 或标志关闭时与 v6 **逐位一致**。
- checkpoint 版本 6→**7**，载荷含新参数与标志；旧版本继续可读。

## §3 实现门（零训练）

1. v7 标志关闭或 bias=0：v6/v7 同 seed 前向/损失/生成逐位一致；参数清单 96,035 含 `copy_persist_bias`。
2. 指针与 bonus 结构：首步无 bonus；bias 设大值（如 +10）时下一步 argmax 强制命中 j*+1（单测构造）；行尾越界不加；zero_read 下指针=None。
3. bias 梯度：teacher-forced 损失对 `copy_persist_bias` 数值差分正确、有限非零；其余参数梯度路径与 v6 相同（不含 bonus 位置时 bias 梯度为 0 的分支检查）。
4. fixture v2：行数/形状/模板/顺序与 v1 逐行对齐（仅值替换处不同）；新六字不出现在任何非值位置；dev/final 子集与 v1 逐字节一致；新 digest 登记入数据报告。
5. v7/v6/v5/v4/v3/v2 checkpoint 恢复矩阵扩展；fresh-process preflight。

## §4 probe（train-only，3 次新运行）

- **A10**（原 train + v7 驻留 + λ=1）：D4 全部门槛作**回归门**（copy-supported M1≥0.90、值 copy 概率≥0.90、错位≤0.50、稳定）——驻留不得破坏单字节能力；另报 bias 终值。
- **A01**（v2 train + v6 + λ=1）：**核心新门**——多字节值行（train 中琥珀/珊瑚/翡翠的 fact/sof/negation 行）M1 ≥0.90；单字节行回归 ≥0.90；错位、稳定同口径。
- **A11**（v2 train + v7 + λ=1）：同 A01 门；另报 bias 终值。
- A00（D4 T2 原配置）不重跑，probe 引 D4 报告；其 matched 对照按 §5 复核后引用历史。
- 任一 probe 门失败→对应因子判否，**匹配 dev 只跑存活因子**（A00 引用不消耗运行；失败路由见 §5）。

## §5 matched dev（4 臂判定，预承诺）

- **对照合法性**：A00=D4 T2 历史（同 seed 同配置）；开跑前 1 seed 重训逐位复核（与 D4 checkpoint 一致），不一致即中止（实现漂移）。
- **G1–G6 沿用 D3/D4 口径，δ 基准=A00**。新描述性主读数（不设门、供归因）：**dev 双字值行的 per-byte copy 完成率**（首字节命中率、全对率分别统计）。
- 判定表（预承诺）：
  | 存活模式 | 裁决 |
  |---|---|
  | A01 过（A11 任意） | E 覆盖为主因；若 A10 不过而 A11 过→M 仅在覆盖后生效，登记交互 |
  | A10 过（A11 任意） | M 偏置为主因；覆盖非必要 |
  | 仅 A11 过 | 两因子都必要（交互）；后续包在 v7+v2 上推进 |
  | 全败但双字完成率显著上升（≥2 臂） | 瓶颈后移：值完成率上去而 flip 对仍 0 → 绑定精度/combo 问题，转 B 槽位/课程评审 |
  | 全败且完成率无变化 | 双因子在当前尺度判否；转 §6 评审（任务族外部对照或 R2 阶段层面审视） |
- combo_different train 恒 0 的 C/B 信号继续单列。

## §6 边界与预算

- 不读 final；dev/final/scorer/表面基线口径不变（v2 只动 train 值）；`growth_admitted=false`、`can_promote=false`。
- probe 3 运行 + matched 至多 3 新臂×3 seeds + 1 复核 ≈ 13 次训练运行（每次 ≤2 min CPU，串行 <30 min）；checkpoint 隔离 reports/r2_d5_checkpoints/。
- 本包关闭 H-E/H-M 因子线；两因子在当前尺度均不获支持时不再追加数据/驻留变体（扩词表、换 bonus 形式等均属变体）。
