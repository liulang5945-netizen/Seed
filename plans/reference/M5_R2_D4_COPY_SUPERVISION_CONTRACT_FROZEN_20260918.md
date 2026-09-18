# M5 R2-D4：copy 通道值监督假设包（训练目标单变量，冻结版）

2026-09-18。状态：**已冻结**（用户在 [D3 结项评审](M5_R2_D3_MATCHED_DEV_REVIEW_MEMO_20260918.md) 裁决「目标函数：材料读取监督」后提交即冻结）。前置：D3 结论——v6 图（多头+PE）在 train 上完全解决可学习性（M1=0.96、negation 48/48）但 matched dev G1–G6 全败；同 seed checkpoint 逐位一致排除实现缺陷；dev 输出刻画证明模型记忆化 train prefix→response 映射，copy 通道从未锁定材料行。结构性事实：**train/dev/final 颜色值域完全不相交**（train 蓝绿红黄青紫 / dev 白灰白乳白米色 / final 黑墨黑漆黑暗红），dev 答案值在 train 中从未出现，唯一产生途径是 copy 通道从材料行读取。

## §1 假设 H-T（冻结）

> 仅答案 CE 下，答案值 bytes 可经 vocab+copy 混合的记忆化路径拟合（train 上值反复出现，无泛化压力），copy 寻址从未被迫学会「问题→材料行→值」绑定；对答案值 bytes 位置追加 copy 分量辅助 NLL（值 bytes 结构上必然出现于材料行中，copy 是其唯一自然来源）后，寻址被监督信号驱动到锁定问题指向的材料行，dev 上组合绑定泛化出现（fact_flip 随事实正确翻转），同时 no_context 条件恶化（证明答案确实经材料读取）。

## §2 精确规格（冻结）

1. **图不变**：v6 原样（readout_heads=4、positional_keys、copy_mixture、question_conditioned_start，96,034 参数）。**零新参数**——单一变量是训练目标。
2. **辅助损失**：`total = answer_CE + λ · L_copy`，**λ=1.0 冻结**。
   - `answer_CE`：现状混合分布 per-position NLL（`first_byte_weight=1.0`，不变）。
   - `L_copy = −mean_{t∈value_positions} log(p_copy_t[target_t] + 1e-12)`；`p_copy_t` 为该步 copy 分量（跨头平均、prefix 可见位置归一化，即既有 `_copy_distribution`）；值位置为空的 episode 辅助项记 0 且不进均值分母。
   - **值位置按 shape 推导（派生规则，mask 不入 checkpoint）**：fact / same_opening_fact → 全 response；negation → 跳过前 6 bytes（「不是」）后全部；unknown / same_opening_unknown / combination_same / combination_different → 空（答案不含颜色值）。
3. **实现**：
   - 新方法 `teacher_forced_mixture_and_copy(prefix, response)` 返回（混合堆叠、copy 堆叠）；`teacher_forced_distributions` 改为其薄包装（仅取混合分量），保证既有路径逐位不变。
   - `sequence_loss(..., copy_value_weight: float = 0.0, value_mask: Sequence[bool] | None = None)`；`copy_value_weight=0` 或 `value_mask=None` 时行为与现状**逐位一致**。
   - `train_step(..., value_masks: Sequence[Sequence[bool]] | None = None)` keyword-only、与 batch 对齐（先例：`contrastive_prefixes`）；checkpoint/episodes 结构不动。
4. **训练**：微批 8（Q1/R1 验证构型）、**原始固定顺序**、30 epochs、lr 0.01；probe 单 seed 20260917；matched dev 3 seeds（20260917/18/19）；训练前 preflight（零步保存→新进程 logits 一致→续训一致）。

## §3 实现门（零训练）

1. `copy_value_weight=0` 或 `value_mask=None`：`sequence_loss` 与 `train_step` 输出与现状逐位一致（loss/metrics/梯度）。
2. 结构断言：`p_mix = gate·p_vocab + (1−gate)·p_copy` 逐位成立；`p_copy[boundary]≡0`；copy 堆叠与混合堆叠同形。
3. 值 mask 推导合同测试：七 shape 全覆盖（fact/sof 全 True、negation 仅前 6 False、其余全 False），对每 split 语料逐行验证。
4. L_copy 梯度路径：数值差分确认流向 addressing 链（head_query_h、evidence_key、question 起点投影、decoder 扫描参数）；对 vocab 分支参数（decoder 输出投影等）无梯度；copy_gate 无梯度（gate 只在混合中）。
5. v6 checkpoint 恢复矩阵与 fresh-process preflight 不受影响（零新参数）。

## §4 probe 与门（一次，train-only）

- 门（全部满足才进 matched dev）：
  - **P1** copy-supported（fact/negation/sof，144）M1 ≥ 0.90（train 拟合保持，与 D2/D3 同门）；
  - **P2** 值位置 copy 分量概率均值 ≥ 0.90（copy 通道真被使用，不是 vocab 记忆替代）；
  - **P3** train copy_misbind（entry_rotation lesion，评测时施加）：值位置 copy 概率降至 ≤ 0.50（寻址是位置绑定而非全局 byte 频率）；
  - **P4** 稳定性沿用 Q1 判据：观测 loss 上升 ≤15%、e25→30 copy-supported M1 回撤 ≤0.10。
- 描述性（不设门）：negation M1、combo_different M1（C/B 路由信号）、boundary 停止率、L_copy 轨迹。

## §5 matched dev 与失败路由（预承诺）

- **臂**：T2（λ=1.0 copy 值监督）on H4+PE × 3 seeds；**λ=0 对照引用 D3 matched 的 H4 臂历史数据**（同 seed 同图同顺序；λ=0 逐位等价由实现门 1 断言，另以 1 seed 复核 checkpoint 一致性后引用）。
- **门沿用 D3 G1–G6 口径**，δ 基准改为 T2 臂 − D3 H4 历史臂：G1 M3≥0.26 且 δM3≥0.20；G2 M4≥0.50 且 δM4≥0.25 且 fact_flip≥2/6（combo_flip 单列）；G3 copy_misbind ΔM4≥0.50；G4 上下文 ΔM3≥0.30 且 no_context M1≤0.50；G5 δM4 两胜一平；G6 boundary_min≥0.95；surface 天花板对照照旧。
- 失败路由：
  | 结果 | 路由 |
  |---|---|
  | probe 任一门失败 | copy 监督在冻结机制内不足以驱动寻址 → 目标函数假设（copy 监督变体）判否，转评审（B 绑定/规模），**不再追加损失变体**（λ 调参、mask 变体、加权方案等） |
  | probe 过 + matched dev 败 | train 上 copy 绑定成功但 dev 不泛化 → 值域不相交结构下需机制外变量 → 转评审（规模/数据扩 train 或 B 绑定） |
  | 不稳定 | 与 Q1 矛盾（微批已稳）→ 回实现门，不放预算 |
  | combo_different train 仍 0 | 保留为 C/B 路由信号，单列记录 |

## §6 边界

不读 dev（probe 阶段）/final；M1–M5 口径与 D1 仪器不变；一次 probe；`growth_admitted=false`、`can_promote=false`；train-only 成功不是能力主张；不改图、不加参数、不动默认入口。
