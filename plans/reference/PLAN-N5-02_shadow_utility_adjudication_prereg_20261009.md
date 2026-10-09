# PLAN-N5-02 设计预注册：影子通电后的贡献验收（判据先冻，读数未回）

- 主线：N5 四层循环的「影子学习→贡献/保持验收」段，前置为 DEBT-G69 的通电旋钮（`2216e2521`）。
- 本篇性质：**判据冻结**。写于 G 跑（`output/n5_shadow_gate_on/`，冻结命令＋`--n5-shadow-gate 1`）
  读数回来**之前**；读数回来之后本篇§2 的字句只许就地打日期更正或升版，不许追改。
- 为什么必须先冻：本轮已两次踩过同一形状——E 的 `null` 是我 `get()` 取法造的假象（㊵-590/595），
  F 的 `gate=0.0` 含义要靠读码才知道（㊵-596）。没有先冻的尺，任何数字都能被读成想要的结论。

## 1. 现场事实（逐条可复核）

- 影子 `_gate` 初值 `0.0`（`taiji/adaptive_residual_shadow.py:118`），`forward()`（`:443`）与
  `learn()`（`:506`）共用 `if self._gate == 0.0 or self._lesioned: return` ⇒ 不通电＝零步学习。
- 全仓 `set_gate` 的调用点只有 `adaptive_residual_shadow.py:945`（`from_checkpoint` 回灌）、
  `eval_taiji_m4v2_r4_shadow.py:195`、`eval_taiji_m4v2_r5_conditional_canary.py:96`；
  `taiji/model.py:867` 开的是 bridge 闸，不是 shadow 闸 ⇒ 训练链此前无人通电。
- F 跑（通电旋钮落地前）实测：`n5_shadow` 在场、`unit_count=97`、`gate=0.0`、`RC_TRAIN=0`
  ⇒ 「物化并挂上」已证，「学习发生」未证。
- 60k 符号的正式跑墙钟实测 **448.5 s**（`elapsed_seconds` 读自 `progress_exit.json`）。

## 2. 判据（六条，先冻）

- **J-N5b-1（自述在场性）**：治疗臂与对照臂的信封 `n5_shadow` 都必须含
  `gate`／`shadow_gate_requested`／`candidate_gate`／`candidate_utility`／`candidate_counterfactual_utility`
  五枚键。**判对条件**：用 `key in env` 断言在场（不用 `get`），缺任一 ⇒ 该臂记 `ran_not_measured`，
  整条验收不判，且**不许**用补默认值的方式凑齐。
- **J-N5b-2（学习确曾发生）**：治疗臂需 `gate > 0.0` **且** `candidate_utility` 与
  `candidate_counterfactual_utility` **不同时为 0**。**判对条件**：两条都成立才允许进入收益判读；
  若 `gate > 0` 而两值同为 0 ⇒ 判 `shadow_inert`，这是一条**独立的、可发表的否证**（通电了但没学到东西），
  不得写成「影子无收益」，也不得回头改 gate 再试而不记档。
- **J-N5b-3（同跑配对，不跨跑拼接）**：收益只在**同一次批次**产出的两臂之间算——
  治疗臂＝冻结命令＋`--n5-shadow-gate 1`，对照臂＝同命令但**物化后不通电**（旋钮缺省）。
  **判对条件**：两臂的 `corpus_fingerprint` 相同、`parameter_budget`／τ／seed／读出链四元组相同，
  并出版 `wall_clock_ms` 与 `elapsed_seconds`；配对不成立 ⇒ 整条不判（沿用 N2 乙档的形状）。
- **J-N5b-4（母量与上界沿用已冻的尺）**：寻址母量取 `wrong_top1_rate`（尺下界 `1/(n//5)`），
  学习侧取 `online_accuracy` 末两段段均值；过线界＝对照臂 **+0.02**（N3 甲用过的同一量，不新造），
  且必须先出版 `ruler_usable`（尺自身动态范围）**为真**才允许比数。
- **J-N5b-5（保持侧不被吃掉）**：CAP 严格真命中与 `replay_well_formed` 各列相对对照臂**下降为 0 才算保持**；
  任一下降 ⇒ 记 `cost_persists`，收益与代价要同时发表，不许只报收益那一半。
- **J-N5b-6（合取才出「有贡献」）**：只有 J-N5b-2 ∧ J-N5b-3 ∧ J-N5b-4 ∧ J-N5b-5 全成立，
  才允许写「影子对该母量有贡献」。缺任一支 ⇒ 明确写「不成立／未判」并点名缺哪支；
  **不许**在预注册判据读平时改用次要指标顶替（本仓已有先例：`J-N3a′`、`J-S1a` 都是这样判否的）。

## 3. 成本与配额（摊开说人话）

- 两臂各 60k 符号 ≈ **448.5 s／臂**，加一次配对判读＝本格约 15–20 分钟墙钟，零权重共享、不改产品码。
- 若 J-N5b-2 判 `shadow_inert`，**不再追加通电档**（另档 gate 值要新批文，见㊵-596 修法④）。

## 4. 失败出口（预注册）

- 配对不成立 ⇒ 本篇判读作废，回到「只在同批内比」重跑两臂；旧件不删，索引标作废。
- 五枚自述键有一枚缺失 ⇒ 记 `ran_not_measured` 并把它当成仪器缺陷立债，**不**当作影子的阴性结果。
- 若 J-N5b-4 的 `ruler_usable` 为假 ⇒ 说明尺无动态范围，结论只能是「这把尺答不了这个问题」。
