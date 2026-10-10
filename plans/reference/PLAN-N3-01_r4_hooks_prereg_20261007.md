# PLAN-N3-01 · N3 乙线预注册：R4 生长协议接钩子（压强阈先在线面冻结）（2026-10-07 起草）

> **来源与效力**：09 §2 N3 乙选项＋§3.2 第 3 条（owner 2026-10-07 裁"甲乙并行"，乙的压强阈**须先在线面冻结**）。本件冻结"怎么冻"与"接完钩子判什么"，**不构成长跑授权**；§1 的只读记录臂零风险、可先行。成熟度＝§13 V1（机制隔离）→ N5 的 V2。

## 0. 现状核对（本会话实测）

- 协议三件在库且是活的：`taiji/adaptive_residual_growth.py`（617 行）的 `AdaptiveResidualGrowthPressure`（:37-49，**五个信号**：`residual_error`/`fast_slow_conflict`/`activity_saturation`/`utility_gap`/`resource_state`，全部经 `_unit` 校验落在 [0,1]，内容寻址 digest 自证 :66-67）；`pressure` 属性＝加权和（:79-88，首项系数 0.30×`residual_error`）；决策侧有 `pressure_below_threshold` 与 `pressure_persistence_below_threshold` 两条拒绝原因（:461-463），阈值与持续性是决策入参并校验为正（:269）。影子面＝`taiji/adaptive_residual_shadow.py`（1033 行，含 `freeze_parent` 时 parent 阈值/单元的保存与恢复 :519-592）。
- **训练器零钩子（复确认 09 §2 N5 的痛点，方式＝按词 grep）**：`scripts/training/train_seed_corpus.py`（877 行）里 `bridge`/`trigger`/`pressure`/`residual`/`growth` **命中 0 处**；其进度行字段实测只有 `epoch/ticks/window_ticks/online_accuracy/mean_surprise/holdout_surprise/elapsed_seconds/exit_reason/base_ticks/budget_max_symbols/ticks_at_exit/reached_budget`（`reports/seed_corpus_smoke_progress.jsonl` 末行，该件由 pytest 产生、非真跑）。
- **但"信号今天取不到"的原因不是信号不存在（这条是起草时差点写错的地方）**：产品路径里**已有生产者** `Seed._record_adaptive_residual_growth_pressure`（[taiji/model.py:1069-1106](../../taiji/model.py)），其中 `residual_error = 1 − prior_probability`（:1081）**正是 09"残差阈 ≥0.4"所指的残差**，`utility_gap = residual_error × (0.5 + 0.5×activity_saturation)`（:1083），`resource_state` 由 `development_structural_budget` 对 `policy.growth_resource_cost` 判（:1085）；它在 **bridge 与 trigger 任一未挂载时直接返回 None**（:1079-1080）。挂载与放行都是显式产品 API：`enable_adaptive_residual_bridge`（:831，**默认 `gate=0.0`＝挂上但不放行**，且要求 settled state :842-843）、`set_adaptive_residual_bridge_gate`（:860）、`enable_adaptive_residual_growth`（:917）、`lesion/unlesion/disable_adaptive_residual_bridge`（:871-885），读侧有 `adaptive_residual_bridge_enabled`/`..._growth_enabled`/`..._growth_decision`（:801-903）。
- ⇒ 所以乙第一步的正确形态＝**在一张面上显式挂载 bridge＋growth trigger、只读记录 `pressure`**（不改结构：`gate` 保持 0.0 不放行、全程不调 `propose_*_candidate()`；权重仍按普通训练跑更新，见下方"面的硬前提"），**不是在仪器里新写一个生产者**（§"别重抄生成链"纪律），也不是"挑一个阈值"。
- **阈值参数早已存在，只是从未按在线面论证过（本轮关键发现）**：`AdaptiveResidualGrowthPolicy`（[taiji/adaptive_residual_growth.py:185-196](../../taiji/adaptive_residual_growth.py)）的产品默认值＝`minimum_pressure=0.70`、`required_pressure_steps=3`、`growth_resource_cost=1`，另有 `minimum_residual_error`/`minimum_fast_slow_conflict`/`minimum_activity_saturation`/`minimum_utility_gap`/`minimum_resource_state` **五道分项闸**（同族字段，全部参与 `pressure_below_threshold` 那类拒绝判定）。既有 canary [eval_taiji_m4v2_r4_shadow.py](../../scripts/training/eval_taiji_m4v2_r4_shadow.py) 的 `_growth_policy()`（:128-141）把**所有 `minimum_*` 设成 0.0、`required_pressure_steps=1`**（＝刻意放行，为的是跑通 shadow 通路），并已完成一整套挂载序列：`enable_adaptive_residual_bridge(gate=1.0, residual_gain=1.0)`（:158）→ `enable_adaptive_residual_growth(policy=...)`（:163）→ `_observe_pressure(model)`（:164）→ 检查点摘要自证（:169-173），报告恒 `can_promote=false`。⇒ 乙的"接钩子"**不必从零造**：复用这条已跑通的挂载序列即可（但 canary 的放行设置**不能当作产品阈**）；09 要的"压强阈先在线面冻结"的落点＝**`minimum_pressure` 与五道分项闸的取值**，而默认 0.70 从未在任何在线面上被论证过。

## 1. 步骤一（零风险，先跑）：在线面取压强分布并把阈值冻下来

- **面的硬前提（本轮实测核对；不写清就会静默空转）**：压强记录的调用点在 `Taiji.observe` 的**预测读出分支**里（[taiji/model.py:2151-2160](../../taiji/model.py)），三条同时成立才产出：①`_adaptive_residual_growth_trigger` 已挂载；②`predictive_readout is self.predictive_readout` ⇒ **必须 `--readout predictive`**（训练器自己的注释 `train_seed_corpus.py:270-272` 就写着"在 `action` 档跑预测链实验是**静默空转**"）；③`_adaptive_residual_shadow is None`（不能同时挂 shadow 臂）。外层 `if`（:2130-2143）还要求四个学习标志位里**至少一个为真** ⇒ **本面是一次会更新权重的普通训练跑**（先前草稿写"零改权重"不准确，此处更正）；它不改的是**结构**——全程不调 `propose_adaptive_residual_growth_candidate()`、`gate` 留 0.0 不放行、`--smoke` 只落 `output/seed_corpus_smoke.pt` 不碰产品件。另一条分支警告：训练器在 answer/self-answer 档走的是 `model.learn_bytes(chunk)`（:455-487），**不经过 `observe`**⇒ 那条路上一个压强读数都不会有，本面必须走 symbol 流分支（:488-491）。
- **面**：`train_seed_corpus.py` 的 smoke 档（`--smoke` 落 `output/seed_corpus_smoke.pt`，不碰产品件；`--max-symbols` 与 `--seed` 随批文写死），加只读压强记录旗标——旗标只做两件事：调 `enable_adaptive_residual_bridge`（**`gate` 留默认 0.0**）＋`enable_adaptive_residual_growth`，然后把每次 `pressure` 观测原样记进行内披露字段（挂载需在 settled state，实施时按 `model.py:842-843` 的约束安排在回合边界）。**"挂载即行为中性"不预设、也**不是**本面的成立条件**：canary 里 bridge 是以 `gate=1.0, residual_gain=1.0` 挂的（`eval_taiji_m4v2_r4_shadow.py:158`），`gate=0.0` 档在本仓没有既成读数 ⇒ 中性与否由 G-N3-1 的默认位对照**实测判定**，判不过就整件不判。
- **读数（先定义再取）**：每 tick 的五信号与合成 `pressure` 的分布（中位/p75/p90/max）＋`online_accuracy` 同步序列；按 §8.7 报告纪律同时给语料可用量、实际唯一 episode、实际更新数、序列长度、独立测试覆盖。
- **冻结规则（判据先于数）**：生长阈值 τ 取 **实测 `pressure` 的 p90 向上取整到 0.05 格**，持续性参数取"连续 3 个记录窗"。并预注册**否证对照**：若同一分布的 p90 ≤ 现行口径推得的 0.4，则沿用 0.4 并在件里登记"两者相等/更低"的事实；若 p90 无法与 `1 − online_accuracy` 对齐（差 >0.1），判"**这套信号与 accuracy 口径不同源**"，阈值不冻、回到协议侧另立假设——不许用调 τ 来凑出生长事件。**本次要冻的不止 `minimum_pressure` 一个**：`minimum_residual_error`/`minimum_fast_slow_conflict`/`minimum_activity_saturation`/`minimum_utility_gap`/`minimum_resource_state` 五道分项闸各按同一规则从各自的在线分布取（各自 p90），一次冻齐六个值＋`required_pressure_steps`；默认 0.70 与 canary 的 0.0 两者都**不作为起点沿用**，只作为对照登记。
- **0.4 这条现行口径的出处如实登记**：09 §2 N3 写的是"accuracy 0.59 ⇒ 残差阈 ≥0.4"，即 `1 − 0.59` 的推得值；它来自**自答档全程平在 0.5934→0.5941** 的两条 2M run 读数（㊵ 系列登记，非本会话实测）⇒ 本件第一步要用在线面复算它，而不是引用它当结论。

## 1bis. 步骤一实测结果（2026-10-08 本会话；§1 的冻结规则一字未改）

**仪器**：`train_seed_corpus.py --pressure-record`（新旗标，默认关；开旗标时在 `model.substrate` 上调
`enable_adaptive_residual_bridge(gate=0.0)`＋`enable_adaptive_residual_growth()`（**产品默认 policy**，
`minimum_pressure=0.70`/`required_pressure_steps=3`/`growth_resource_cost=1`），再把 `trigger.observe`
套一层记录器——**生产者不重写**，五信号仍由 `taiji/model.py:1069-1106` 自己算）；
判读器＝[count_taiji_n3_pressure_thresholds.py](../../scripts/training/count_taiji_n3_pressure_thresholds.py)。

**两条链各一张面**（同命令形状，只换 `--resume` 底座；`--max-symbols 4000`、`--progress-every 200000`、seed 20260822）：

| 面 | 底座 | 观测数 / 覆盖率 | pressure p50 / p90 / max | residual_error p90 | `fast_slow_conflict`、`activity_saturation` | 默认 policy 下 should_propose 占比 | 口径对照 | 判级 |
|---|---|---|---|---|---|---|---|---|
| `reports/taiji_n3_pressure_face_productdefault_20261008.json` | 产品默认 `DEFAULT_CHECKPOINT`＝`checkpoints/seed_a31self_with_circuit.pt` | 4000 / **1.0** | 0.423101 / 0.425 / **0.425** | **1.0** | 整场恒 **0** | **0.0** | 控制面 online_accuracy 0.172 ⇒ `1−acc` 0.828 对 residual p90 1.0 ⇒ 差 **0.172** | **not_frozen_caliber_mismatch** |
| `reports/taiji_n3_pressure_face_trained4m_20261008.json` | 2M/4M 语料档续训底座 `output/a31_chunked_self/checkpoint.pt` | 4000 / **1.0** | 0.388628 / 0.422494 / 0.424992 | 0.994103 | 整场恒 **0** | **0.0** | 控制面 0.28725 ⇒ `1−acc` 0.71275 对 0.994103 ⇒ 差 **0.281353** | **not_frozen_caliber_mismatch** |

判读件＝`reports/taiji_n3_pressure_thresholds_productdefault_20261008.json` 与 `..._trained4m_20261008.json`（均 rc=0）。

**四条结论**：

1. **按 §1 的冻结规则，阈值不冻**——两条链都触发预注册的否证支（`|residual_error p90 − (1 − online_accuracy)|` = 0.172 与 0.281，均 >0.10）⇒ 判"这套信号与 accuracy 口径不同源"。
   ⇒ **09 §2 N3 里"accuracy 0.59 ⇒ 残差阈 ≥0.4"这条现行口径被否证**：trigger 的 `residual_error` 是 `1 − prior_probability`（逐符号预测概率，`model.py:1081`），不是 `1 − 窗口正确率`；两者在本面上相差 0.17–0.28，**拿 accuracy 推出来的数不能当 τ 用**。
2. **默认阈在这张配置上算术不可达**：`fast_slow_conflict ≡ 0` 且 `activity_saturation ≡ 0` 时 `utility_gap = 0.5·residual`，于是 `pressure = 0.30r + 0.25(0.5r) = 0.425·r ≤ **0.425**`——与两条链的实测 max（0.425 / 0.424992）对上；而默认 `minimum_pressure=0.70` ⇒ **4000 步里 `should_propose` 为真的步数＝0**。这就是 09 §2 N5 记的"出厂件零进化状态"的一条机制解释：**不是没接线，是这条链上够不着阈**。
3. **两维恒零的机制已定位**（不是神秘缺失）：`fast_slow_conflict` 来自 `_developmental_f1_fast_slow_conflict()`，bundle 只在检查点带 `developmental_f1` 键时才挂上（`model.py:1058-1067` 与 :3783-3797），**trainer 没有任何旗标能开它**（`train_seed_corpus.py` 里 `developmental` 命中 0）；`activity_saturation`＝"单元活动 ≥ 目标活动点的比例"（`taiji/adaptive_residual_bridge.py:101-107`），而 `gate=0.0` 的 bridge 从不参与读出 ⇒ 无活动可饱和 ⇒ **本面测的是"两维被装配压住"的配置**。
4. **因此 §1 的冻结规则本身需要一条前置**（登记为下一步的假设，不在本件加码）：τ 只能定义在**四维修正都活着**的装配上；要拿到那个装配，要么 (i) 底座检查点自带 developmental F1 状态、要么 (ii) 把 bridge 以 `gate>0` 放行——**两者都改变模型行为，不是只读面**，须按 09 §3.2 回 owner。在此之前，任何 τ 都是"压住两维的配置"下的数，本件**不给数**。

**守卫与负对照（仪器能为 false 的证据，全部本会话实跑）**：
- 面收尾自述 `kind=tail`＋判读器覆盖率守卫（`观测数 / 面内 tick ≥ 0.95`）——**这条是被一次真实自伤逼出来的**：第一版用默认 `--progress-every 10000`，周期性 `_flush` 里的 holdout 探针之后压强支不再产出观测而进度行照涨，面跑到 5 万 tick 只有 9,727 条记录且**看不出来**；那次的面件已删除、未入库，判读器补了守卫后重跑两支（覆盖率 1.0）。
- 同因还修掉一处会伪装结论的仪器缺陷：两个判读器（本件与 N1-02 的 `count_taiji_n1_trajectory_following.py`）的拒绝路径原本把带"⇒"的中文 `print` 到 GBK 控制台，会在**已经判完之后**抛 `UnicodeEncodeError`，把 rc=2 的响亮拒绝伪装成 rc=1 的崩溃 ⇒ 现改为 stdout 走 ASCII 转义、入库件保留中文。
- 判读器**五条拒绝路径各自实跑 rc=2**（每种都造了专用件）：缺 face 行／缺 tail 行／`tail` 自述 4000 而实际 0 条（不自洽）／`tail` 自述 4005 而实际 4000 条（不自洽）／**覆盖率截断**——把 `ticks_at_close` 改成 40000 复现第一版的形状，判读器答"观测只覆盖面内 tick 的 10.1%（4000/39727）⇒ 分布被截断，不判"。两条挂载守卫也各自实跑响亮失败（`--readout action` 与 `--answer-chunking per-answer` 时拒绝开跑，且不留半成品文件）。
- 行为对照（同参开/不开旗标各跑一支）：进度行的 `ticks`/`window_ticks`/`online_accuracy`/`mean_surprise`/`holdout_surprise`/`base_ticks`/`ticks_at_exit`/`reached_budget` **八个键逐位相同**（如 trained4m 链两支都是 `online_accuracy 0.28725`、`mean_surprise 2.7364605140439897`）⇒ **记录器不改学习行为**；差异只在 `elapsed_seconds`、终件 sha 与参数量（bridge 挂上＝+96 单元/+2304 边，`769,951 → 772,255`）——这正是"挂载即改装配"的披露，不参与任何判据。
- 既有测试面：`tests/seed/test_seed_corpus_pipeline.py`＋`test_seed_corpus_eval.py` **8 passed**（旗标默认关）；`ruff check .` 0 error、`black --check .` 全绿。

## 2. 步骤二（接钩子，需 owner 批长跑）：四层循环走通

- **主判据 J-N3b（生长事件真实且可归因）**＝一次真实长跑里同时取到 09 §2 N5 的五面：出生（新单元/区域数）→ 消费（哪些 batch 走了新单元）→ 贡献（同预算下有/无新单元的读数差）→ 保持（旧任务矩阵不退）→ 资源（参数与峰值存储增量）。缺一面即判"接钩子未成立"，不以"日志里有出生事件"结项。
- **同容量对照（§9 硬要求，随同一批文）**：fixed-large（同最终容量、不生长）与随机成长（随机出生）两条对照臂必须在同一预算、同一 seed 下并跑；**群体数量不证明协作**，故贡献面只按"同预算读数差"计。
- **否证出口（预注册）**：若治疗臂与 fixed-large 无读数差 ⇒ 判"生长无增益"，按 09 §4 第五行处置（保留旧内核、记质量—成本曲线），并把负结果留在台账（§18.9：不在一个局部参数上无限打转）。

## 3. 守卫

- **G-N3-1 无旗标即原样**：记录旗标默认关，**不开旗标时训练器的输出与今日逐位同**（参数摘要＋进度行两条对照，对照件名随实施登记）。**开旗标那一臂不假设行为中性**：挂载 `gate=0.0` 的 bridge 仍可能改变发育计数/RNG 流，因此把"开旗标 vs 同参不开旗标"的参数摘要差与进度行差**量出来并披露**；若确有差，本件把读数定性为"**挂载配置下**的压强分布"（这正是步骤二要跑的那个配置），不写作"生产路径的压强分布"。
- **G-N3-2 影子不写主参数**：shadow 阶段的主参数摘要必须在前后不变（`freeze_parent` 路径已有保存/恢复，须实证一次）。
- **G-N3-3 训练前保存检查**：02 §2.2＋§12 快照合同七项齐备；正式跑不显式给 `--checkpoint` 会被响亮拒绝（trainer 既有纪律，2026-09-28 起）。
- **G-N3-4 经历分源**：重放/想象与实际经历分开记（§1 全局不变量），生长事件的经验单位按 §17.2 第 5 类"巩固与保持经历"标注。

## 4. 预算与 owner 待批

步骤一＝**已完成**（2026-10-08，结果见 §1bis）：两条链各 1 张 4000 步面、旗标默认关、不碰产品件；**步骤二＝3 臂长跑（治疗/fixed-large/随机成长）需 owner 批算力与时长**，批文里写死 `--scale` 或 `--parameter-budget`、`--max-symbols`、`--epochs`、`--seed`、`--device`。若 owner 只批步骤一，本件按"阈值已冻、钩子未接"如实停靠，不声称 N3 乙启动。

## 5. 不变项

M6 收官、M8 挂起；N1（S1 待批）、N2（通电待批）、N3 甲（算力预算未批）独立推进；N4/N5 后置（N5 依赖本件步骤二的钩子面）；台账行序以行首标号为准（㊵-419⑥）。

## 5bis. 判据载体在哪（2026-10-10 ㊵-655 带日期补注，只点名、不新增判据）

- 本件的可判内容住在 §1「步骤一」的 2 枚读数（在线面的压强分布与 `should_propose` 触发计数）与 §3「守卫」的 5 条 fail-closed 形状；
- §2「步骤二」的判据＝接钩子之后的 4 张在线面，其参数住在 §4「预算与 owner 待批」；㊵-484 已当场否证本件原先那句"残差阈 ≥0.4"（trigger 的 `residual_error` 是 `1 − prior_probability`，不是窗口正确率），现行口径见 PLAN-N3-07；
- 本件对措辞门不可见的原因是节名用了「步骤／守卫／预算／不变项」，不在门的 3 个标题词（判据／出口／验收）之内 ⇒ `structural_criterion_lines` 计 0；本条把载体点名给门看，**不**改上面任何一条已冻口径；
- 本条不引入新数值线：上面出现的 2 与 5 与 4 是本件既有小节的条目数，不是新增的判据。
