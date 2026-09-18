# M5 R2-D3 结项评审备忘：首步读取几何（H-G）train 成功、dev 泛化失败与路由

2026-09-18。前置：[D3 合同（冻结）](M5_R2_D3_FIRST_STEP_GEOMETRY_CONTRACT_FROZEN_20260918.md)、[A 族评审备忘 §6（S1 裁决）](M5_R2_D2_A_FAMILY_REVIEW_MEMO_20260918.md)。状态：**probe 通过（train-only）→ matched dev G1–G6 全败 → 按合同 §4/§5 预承诺，H-G 在 dev 上不获支持，转入评审裁决（目标函数/B/规模），不再追加几何变体。**

## §1 结果汇总（报告为准）

- **A-G probe**（H=4+PE，微批8，seed 20260917，[报告](../../../reports/r2_d3_multihead_probe_20260918.json)）：passed。epoch 5 即收敛：loss 4.56→0.0081，train M1=167/174=0.96，copy-supported M1=144/144=1.0，**negation 48/48（D2 六构型中恒 0 的瓶颈首次突破）**，combo_same 7/7；仅 combo_different 0/7（答案同以「不」开头，预注册的 C/B 路由信号）。门含稳定性判据全过（loss 单调下降、e25→30 无回撤）。
- **matched dev**（3 臂 × 3 seeds，[报告](../../../reports/r2_d3_matched_dev_20260918.json)）：**G1–G6 全败，outcome=failed**。dev（98 题，训练中未读）上：
  - H4（treatment）：M1 = 0.000 / 0.000 / 0.092（seeds 20260917/18/19）
  - H1+PE（单头对照）：M1 = 0.000 / 0.010 / 0.112
  - C4（广播对照）：M1 = 0.010 / 0.061 / 0.061
  - 三臂 no_context、zero_read、copy_misbind、value_misbind 各条件 M1/M4 全贴地；fact_flip / combo_flip / invariance 逐对率全 0；boundary_min=0.582（H4 seed 20260918）。surface 天花板未超越（0.092 < 0.1735）。
  - 多头与位置编码的贡献在 dev 维度不可分辨：三臂全部≈0，无从按合同 §5 第三行区分关键变量。

## §2 归因链（排除实现缺陷后定性失败模式）

1. **训练完全可复现**：同 seed（20260917）下 probe 与 matched dev 的 epoch30 checkpoint 参数逐位一致（ALL_IDENTICAL，config 相同，代码修订号不同仅记录来源）。probe 成功不是脚本差异假象；dev 失败不是实现缺陷（排合同 §5 第四行）。
2. **train 拟合真实**：加载该 checkpoint 重放 train 行，fact 输出精确（紫/黄/青逐位命中）；loss 0.0081 非 loss 景观假象。
3. **dev 失败模式刻画**（checkpoint 重放 dev 行）：
   - 首字节策略残存：negation 输出 `不是䪍是`（目标 `不是乳白`）——首两字节「不是」正确，内容字节错误；unknown 首字「缺」部分语义相关。这与 D2 结论一致：首字节策略从来可学。
   - 内容退化：fact/combination_different/same_opening_fact/combination_same 多形状输出同一高频单字 `索` 或乱码 `籢`；边界停止率部分 seed 低至 0.582。
   - **不是记忆答案直抄**：0/98 的 dev 输出等于任何 train response；dev/train prefix 零重叠（0/98）。
   - **copy 未锁定材料行**：dev 目标值（乳白/米色/灰白）就在该行材料中，copy 通道却未复制，而是输出 train 分布碎片。
4. **判定**：模型学到的是 **train prefix→response 映射的记忆化**（含「不/缺」首字节策略与高频单字回退），而非「问题类型→材料内位置→值」的组合绑定规律。**瓶颈从 D2 的「可学习性」（0.667 平台、negation 无法拟合）转移到 D3 的「泛化」（train 全拟合、dev 全崩）**——表达力已足够，缺的是组合绑定的外推机制。

## §3 对既有结论的更新（口径收紧）

- **成立的**：多头+PE 在 train-only 可学习性维度解决 negation 首字节（H-G 假设的可学习性半边获支持）；微批 8 稳定性判据在 H4/H1/C4 三臂 9 运行中复现（无 D2 式晚期失稳，仅 H4 一 seed 边界行为退化）。
- **不成立的**：train-only 可学性≠内容条件化能力。probe 报告预承诺「not a capability claim、train-only 成功不是 L2 主张」被 dev 直接证实为必要边界。**D2 时代「0.667 平台=内容决策不可条件化」的表述需修正为「在无 PE/多头的图族下 train 上不可条件化」**；条件化在 v6 图族 train 上可实现，但以记忆化方式实现。
- **不可推的**：不能推成「所有几何/架构变体均无法泛化」——判否范围限当前尺度、当前目标函数（仅答案 CE）、当前数据量（174 episodes）下的几何族；绑定/目标函数/规模未被排除（恰是评审议题）。

## §4 合同路由执行

合同 §4：「dev 上过门才支持 H-G」→ 未过，H-G **不获支持**（并非「证伪多头是关键变量」——三臂同崩，变量不可分辨）。合同 §5 预承诺的评审方向打开：**目标函数 / B（绑定）/ 规模**；不再追加几何变体（H5/H8/更多头/PE 变体等）。combo_flip 作为 C/B 路由证据在本轮 matched dev 中全 0，与 fact_flip 不可区分，仅保留「combo_different train 仍 0/7」作为 C/B 信号。

## §5 评审议题（三个预承诺方向的具体化）

| 方向 | 核心问题 | 最小判别证据设想（供裁决，未冻结） |
|---|---|---|
| 目标函数 | 仅答案 CE 允许「记忆 prefix→response」捷径；材料内容从未被显式监督读取 | 训练目标加入材料行读取监督（如 masked 材料值预测 / 强制 copy 通道损失），同微批 8 重跑 probe，看 train 拟合是否转为绑定式（no_context 恶化、full 泛化改善） |
| B 绑定 | 值↔位置↔问题类型的绑定无任何归纳偏置或显式表征 | 在 v6 上增加绑定结构（对象槽位/显式指针），单独检验 dev flip 对 |
| 规模/数据 | 174 episodes + lr0.01×30ep 下记忆化是优化自然解 | 等暴露口径下扩 train（组合生成更多 episodes）看 dev M1 是否随数据量上升——直接区分「容量不足」与「目标允许捷径」 |

## §6 边界

`growth_admitted=false`、`can_promote=false`；本包关闭的是 H-G 假设检验，不计为能力阶段进度；dev/final 口径不变；下一包裁决前不追加训练。
