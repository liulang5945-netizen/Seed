# PLAN-N2-06 写入与巩固合同的仓内承载与缺口登记（V-17 的对账件；判据先冻、零算力、不动产品码，2026-10-11）

## §0 这件结的是哪一条，为什么值得单独有一件

`09_NEXT_MAINLINE_PLAN.md` §1.3 的普查（仪器 `scripts/training/audit_taiji_vision_clause_traceability.py`）
把「确未展开」的条款从 7 条一路压到 **2 条**，四枚读数件序列（id／锚／unmapped）＝
`0/12/7 → 1/13/6 → 3/15/4 → 5/17/2`，分母 `clause_rows_parsed=41` 与 `clauses_required_now=19` 全程未动。
余下两条里的 **V-17（VISION §5.3 写入与巩固）** 归属 **N2**，09 §1.3 那行写的状态是「09 摘要；待 PLAN-N2 展开」。

VISION §5.3 原文很短，但它是**六个子句加一条测试清单**的合同（逐字拆见 §1），而 N2 的出口交付
今天只结到了「通电两次＋速率成列」（㊵-677），**没有一条判据在管「谁可以写什么、写错了怎么撤销」**。
⇒ 本件把 §5.3 的六个子句**逐条对到仓内实名**，并把「合同有承载、缺口在哪一格」写成可判的形式。

**本件零算力、不动 `taiji/`／`seed/`、不新增存储对象、不新增阈值字面量。**

## §1 V-17 六个子句的承载点（逐子句给实名与行锚；找不到就写未承载，不许造名字）

| # | VISION §5.3 原文子句 | 仓内承载（现读） | 今天的真实状态 |
|---|---|---|---|
| a | 真实 outcome 可写 E | `EpisodicMemoryRecord`（`taiji/contracts.py:1942`，带 `outcome`／`action_intent`／`world_transition`／`prediction_error`＋**`provenance` 默认 `experienced`**）；写入 `EpisodicMemoryStore.write`（`taiji/episodic_memory.py:47`）；巩固侧写入 `SeedSleepScheduler._write_episodic_observations`（`seed/sleep.py:173`） | **有承载**：记录合同里就有 outcome 槽，来源字段必填 |
| b | A 写入需**来源／预算与置信**检查 | A 侧写入 `EpisodicField.write(..., provenance, learning_targets, learning_scale)`（`taiji/memory.py:451`）；来源枚举 `EpisodicField.PROVENANCE_KINDS`；置信 `last_confidence`（`taiji/memory.py:434`）与衰减 `memory_confidence_decay`（:400）；预算＝`capacity`（`taiji/episodic_memory.py:30`）＋语料档 `--max-symbols` | **部分承载**：来源／置信都在 **A 侧**；`EpisodicMemoryStore.write`（E 侧）**只校验类型与 `cue` 维度**，既无来源枚举校验也无置信闸 ⇒ 见 J-N2f-2 |
| c | φ 只接受**批准的**适应目标 | 白名单 `EPISODIC_LEARNING_TARGETS`＝`all／association／readout／action_readout／outcome_readout`（`taiji/config.py:22`）＋`validate_episodic_learning_target`（:31，不在表内即 `ValueError`）；结构侧准入 `AdaptiveResidualGrowthPolicy` 的六道 `minimum_*`、`propose_structural_candidate_from_pressure`（`taiji/adapter.py:1851`）与 `can_promote` | **有承载**：批准的适应目标今天是一份**可机检的白名单**，不是口头承诺 |
| d | 过期、错误和重复经历有**撤销／去重**规则 | 撤销＝`RecoveryPortfolio.mark_expired`（`taiji/planning.py:387`）、`RecoveryStrategyLedger.revoke`（:851）与 `revoked_memory_ids`（:1090）、`Taiji.revoke_recovery_strategy`（`taiji/adapter.py:9365`）；去重＝`deduplicated_count`（`taiji/internalization.py:609`）与 E 侧**同 `memory_id` 覆盖**（`taiji/episodic_memory.py:54-55`） | **分三处、互不作用于 E**：`mark_expired`/`revoke` 的对象是**恢复策略与计划账本**，`deduplicated_count` 属内化重放分区；E 侧的「重复」只有同 id 覆盖，**没有按内容去重、也没有按来源或时效撤销** |
| e | **E 删除 ≠ θ 中已巩固知识删除**，不能承诺自动遗忘 | E 侧唯一删除＝容量淘汰 `while len > capacity: del oldest`（`taiji/episodic_memory.py:56-58`，FIFO）；θ 侧最接近的只有 `propose_region_prune_from_underuse`、`lesion_region`（`taiji/neuron_network.py:1110`）、`prune_schema`（`taiji/world_learning.py:1290`），三者都按**结构证据或欠使用**，不是按「某条事实」 | **未承载（这条要写成否证）**：今天既没有「删一条 E」的公开操作，也没有「从 θ 里忘掉某条已巩固知识」的操作；两者都不是自动遗忘 ⇒ 任何 `automatic_forgetting` 类表述一律禁止出版 |
| f | 仅增加**存储量**不算结构成长 | 现成仪器 `scripts/training/summarize_taiji_n5_resource_face.py` 的 `unit_count` 与 `shared_numel_delta`（㊵-678 已把这两枚键出版） | **已可判**：本件**引用而不复制**这两把尺；存储量与参数量分列出版，不许相加 |

**§5.3 的测试清单（七类）今天的在场情况**：覆盖更新／重复输入／容量饱和／跨会话隔离／移除后能力变化这五类
**都有命名夹具**——例如 `tests/seed/test_memory_store.py::test_cross_process_duplicates_collapse_on_read`（重复坍缩）、
`tests/seed/test_sleep_pass.py::test_state_rolls_forward_across_passes`（跨趟状态）、
`tests/taiji_native/test_adaptive_neuron_region.py::test_new_unit_can_learn_a_holdout_pattern_and_lesion_is_causal`
与 `::test_neuron_region_checkpoint_roundtrip_preserves_growth_and_lesion`（移除的因果性与恢复归属）。
**矛盾事实**与**噪声**两类的夹具今天**未逐枚点名**，由 J-N2f-5 判并登记（不许用「测试很多」代答）。

## §2 三条缺口点名（这是本件真正要交出去的东西）

1. **E 写入侧没有来源枚举校验**：`EpisodicMemoryStore.write` 接受任何 `EpisodicMemoryRecord`，
   而来源白名单长在 `EpisodicField`（A 侧）⇒ 「真实 outcome 可写 E」这句今天**只由记录合同的字段存在性保证**，
   不由写入闸保证（`provenance` 默认值 `experienced` 甚至意味着「不声明也是 experienced」）。
2. **撤销规则不覆盖 E**：`mark_expired`／`revoke` 的作用对象里没有情景记录 ⇒ 「过期经历可撤销」
   今天**只在恢复策略层成立**。
3. **来源枚举里没有 `generated`**：`EpisodicField.PROVENANCE_KINDS` ＝
   `("experienced", "imagined", "replayed", "external")`，**没有** V-21 要求的 `generated`
   ⇒ 自生成内容今天只能借用 `external` 或 `experienced` 写入，「不混进事实/奖励」缺一个可判的取值。
   （这条同时是 PLAN-N5-11 §2「防自证学习」那一行的前置，两件共用这枚事实，各自判。）

## §3 判据（冻；五条，全部零算力，不新增阈值字面量）

* **J-N2f-1 写入来源必须显式声明**：凡被采纳为「真实经历」的 E 记录，其 `provenance` 必须由调用方**显式给出**；
  若一条记录靠默认值取得 `experienced` ⇒ 记 `implicit_experienced_provenance`，并收回「这条是真实经历」这句。
  执行体＝构造性复算（造一枚不带 `provenance` 的记录，断言今天会被接受 ⇒ 该缺口为真且可检）。
* **J-N2f-2 检查的作用位置必须点名到哪一侧**：来源／置信／预算三项检查，必须逐条写明它住在 A 侧（`EpisodicField.write`）
  还是 E 侧（`EpisodicMemoryStore.write`）；**不许合并成一句「写入有检查」**⇒ 本件现读＝来源与置信在 A 侧、
  E 侧只有类型与维度校验，故 J-N2f-2 判 `checks_asymmetric`（登记为缺口，不判成通过）。
* **J-N2f-3 撤销与去重的对象集合必须分类登记**：每个撤销/去重机制必须写出它作用的集合
  （恢复组合／计划账本／内化重放分区／E 情景记录）。四个集合里若 E 情景记录无人作用 ⇒
  记 `episodic_revoke_absent`；**不许**用「有 revoke 符号」代答「有撤销规则」。
* **J-N2f-4 删除语义分离（否证式判据）**：本件要求任何读数件／判读文里**不得出现**把 FIFO 淘汰或结构剪枝
  说成「遗忘某条知识」的表述。复算＝**只扫出版面**（`reports/` 与 `plans/reference/PLAN-*.md`）现扫
  `automatic_forgetting`／「自动遗忘」字样，命中即判 `forbidden_claim`。**扫描面必须排除 VISION 本体**：
  ㊵-680 实测现扫全仓得到 1 枚命中，而那枚就是 VISION §5.3 那句禁令自身（把违禁词抄进禁令会自指命中），
  把它算成违规就是门的取法瞎，不是文档没钉。
* **J-N2f-5 七类夹具逐类有实名或写缺失**：§1 末段的七类必须逐类给出一枚 `tests/` 里的实名函数，
  或明确写 `fixture_absent`。缺 ⇒ 转 §4 批次 C，不许静默跳过。

## §4 实施批次（哪些格子需要谁的批准）

| 批次 | 内容 | 触面 | 前置 |
|---|---|---|---|
| A | E 写入侧加来源校验与显式声明闸（对齐 `PROVENANCE_KINDS`） | `taiji/episodic_memory.py`／`taiji/contracts.py` | **owner 批文**（产品码） |
| A′ | 来源枚举增设 `generated`（与 §2 缺口 3 同一格） | `taiji/memory.py` 的 `PROVENANCE_KINDS` | **owner 批文**；加取值会改信封语义，须与 PLAN-N5-11 一起裁 |
| B | 七类夹具里缺的两类补测（矛盾事实／噪声） | 仅 `tests/` | 我可自办；判据已在本件冻住 |
| C | E 侧撤销／失效（按来源与时效，不改 θ） | `taiji/episodic_memory.py` | **owner 批文**，且须与 §3 J-N2f-4 的删除语义一起裁 |
| D | 出值与判读 | 零算力（读现成件＋夹具） | 批次 B 绿之后 |

## §5 判据载体在哪（措辞门要求的一节）

J-N2f-1／2／3 的执行体＝本件 §1 那张表加一次构造性复算（造不带来源的记录、读两侧 `write` 的校验分支），
不需要开跑；J-N2f-4 的执行体＝对 `reports/` 与 `plans/` 的字样现扫（读面按版本控制取）；
J-N2f-5 的执行体＝`tests/` 里的命名夹具（按函数名扫，不按文件数）。
「存储量≠成长」不另造尺——沿用 `summarize_taiji_n5_resource_face.py` 的 `unit_count`／`shared_numel_delta`，
「带内即尺不可用」沿用 `adjudicate_taiji_n5_shadow_gate.py:295` 的 `ruler_usable` 式（**引用而非复制**）。
⇒ 本件没有一条判据是手算的；同时**没有一条已经出值**（五条待批次 B／D 之后，J-N2f-4 除外，它今天即可扫）。

## §6 出口（三条互斥）

* 五判据出值且 §3 J-N2f-4 为 0 命中 ⇒ V-17 转为「已展开且已判」，DEBT-G97 的最后两条之一结清，
  N2 的写入合同第一次有机械形状；
* 任一判据出值但判**不成立**（今天预期就是 J-N2f-2／3 判不成立）⇒ **仍算 V-17 已消化**（§1.3 的基准要的是
  「展开并判」，不是「判成」），结清 DEBT-G97 的同一条，并把缺口按批次 A／A′／C 归到 owner；
* 若只能停在「表填了但复算读不到」⇒ 本件只算设计件，**不得**声明 V-17 已消化，DEBT-G97 保持未清，
  并收回「N2 有写入合同」这句。

## §7 落笔即钉的三行

* **预期为绿的成员**：J-N2f-4（按上面的出版面口径现扫＝`automatic_forgetting` 0 命中；全仓那 1 枚命中住在
  `plans/reference/VISION_FUTURE_TECHNOLOGY.md` 的禁令原文里，不计违规）、J-N2f-5 里已点名的五类夹具。
  **预期为红的成员**：J-N2f-1（默认值就是 `experienced`）、J-N2f-2（检查不对称）、J-N2f-3（E 无人撤销）。
  这三条的红是本件的目的，不是失败。
* **仍红时收回的句子并点名**：一旦 J-N2f-1／2／3 判红 ⇒ 收回「写入与巩固合同已生效」这句，点名是
  E 侧 `EpisodicMemoryStore.write` 缺来源与置信校验、缺按来源/时效的撤销；不得改判据迁就现状。
* **发表资格前置**：任何「Taiji 记得住也能撤销」的对外表述必须同时给出 ①§1 表逐子句的实名与行锚，
  ②三条红判据的当前值与复算命令，③批次 A／A′／C 的批文状态。三条缺一即视为未交付。

## §8 不变项

M6 收官、M8 挂起；N2 出口两样已齐（㊵-677），N5 五面已齐（㊵-678）；DEBT-G78 按 owner #24① 继续挂起，
本件不代为接线。本件**不新增存储对象、不新增阈值字面量、不开跑**；
`generated` 取值（§2 缺口 3、批次 A′）今天**只是登记**，本件一个枚举值都没改。债表现至 **DEBT-G97**。
