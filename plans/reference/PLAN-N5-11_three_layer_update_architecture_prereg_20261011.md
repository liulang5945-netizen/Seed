# PLAN-N5-11 三层更新架构与一次更新的七步流程登记（V-21 的对账件；判据先冻、零算力、不动产品码，2026-10-11）

## §0 这件结的是哪一条，以及它和 PLAN-N2-06 的分工

`09_NEXT_MAINLINE_PLAN.md` §1.3 普查里**最后两条**「确未展开」是 V-17 与 V-21（读数件
`reports/taiji_vision_clause_traceability_20261011_after_n103.json`：id 命中 5／锚命中 17／unmapped 2）。
V-17（§5.3）是**写入合同**——谁可以写什么、写错了怎么撤销，已由
[PLAN-N2-06](PLAN-N2-06_write_consolidation_contract_prereg_20261011.md) 展开。
**V-21（VISION §19.6 原生持续适应）是另一件事**：它规定运行期适应要分成
**状态/记忆、快参数 φ、慢参数 θ** 三个**可独立声明**的更新层，并给出「一次更新的逻辑流程」七步、
一条「防止自证学习」禁令和一套 support/query 评价规范。09 §1.3 那行标它归属 **N2/N5**（跨两条线），
故本件取 N5 编号、在 §2 把与 N2 共用的那枚事实**同写两遍但各自判**（两件不互相代答）。

**本件零算力、不动 `taiji/`／`seed/`、不新增层、不新增阈值字面量。**

## §1 三层的仓内承载（逐层给「允许更新什么／实际做法／失败后的边界」）

| 层 | VISION §19.6 允许更新什么 | 仓内实际做法（现读实名） | 失败后的边界（今天能否兑现） |
|---|---|---|---|
| 状态／记忆 | 会话状态、临时对象绑定、合法情景与关联记忆 | `EpisodicMemoryStore`（`taiji/episodic_memory.py:23`，容量 `capacity`、`write`/`retrieve`）＋`MemoryRecall`（`taiji/state.py:369`）＋信封 `episodic_memory` 块；会话边界＝`reset_dynamics(episode_id=...)`（`taiji/model.py:1841`、`seed/model.py:71`） | **可兑现**：该层不动权重，恢复由 `checkpoint()`／`from_checkpoint()` 承担；「存储增多≠新技能」已由资源面两枚键机检（`unit_count`、`shared_numel_delta`，㊵-678） |
| 快参数 φ | **预先列出的**低秩残差、读取投影或适应头，不默认改全核心 | 目标白名单 `EPISODIC_LEARNING_TARGETS`＝`all／association／readout／action_readout／outcome_readout`（`taiji/config.py:22`）＋`validate_episodic_learning_target`（:31）；写入 `EpisodicField.write(..., learning_targets, learning_scale)`（`taiji/memory.py:451`）；快固化通道 `consolidate_fast`（`taiji/developmental_synapse.py:358`）与 `k_fast_slow.consolidate`（`taiji/k_fast_slow.py:243`） | **可兑现且已被证过能为 0**：bridge／shadow 的 `lesion()`／`lesioned()`（`taiji/adaptive_residual_bridge.py:125`、`taiji/adaptive_residual_shadow.py:320/329`）就是「关闭 φ」的现成出口；PLAN-N5-07 的守卫臂已证 `fast_delta` 可 0 可非 0（有动态范围） |
| 慢参数 θ | 经过批准的共享核心／语言等参数 | `Seed.consolidate(cycles, learn)`（`seed/model.py:129`）→ `TaijiConsolidation`（出版 `accepted`／`cycles`／`mean_priority`／`mean_confidence`／`mean_error_norm`）；夜间调度 `SeedSleepScheduler.night`（`seed/sleep.py:37`）与 `seed_platform/sleep_pass.py:run`；结构候选 `propose_structural_candidate_from_pressure`（`taiji/adapter.py:1851`）、`propose_region_split_from_error`、`propose_unit_add` | **一半可兑现**：回退父权重有真演示（`restore_native`／`SeedRuntime.load`，PLAN-N2-02 结清 DEBT-G47 那次）；但**未准入 child 不替换 parent** 这条今天只有 `can_promote` 在场，**没有**「从产物里抹去已学知识」的操作（VISION 明写不许承诺，故这里不是缺陷而是边界） |

## §2 一次更新的七步流程逐步登记（每步要么给实名，要么写未承载）

| 步 | VISION §19.6 的流程原文 | 仓内承载 | 状态 |
|---|---|---|---|
| 1 | 反馈身份核验 | `experience_id` 唯一记账：`taiji/evolution_credit.py:198` 拒绝对同一 `experience_id` 重复记账 | **有承载** |
| 2 | 权限／来源／时效／重复检查 | 来源＝`EpisodicField.PROVENANCE_KINDS`；时效＝`RecoveryPortfolio.mark_expired`（`taiji/planning.py:387`）；重复＝`deduplicated_count`（`taiji/internalization.py:609`） | **有承载但分散**，且三项都不作用于 E（与 PLAN-N2-06 §1 的 d 行同物，**两件各自判**） |
| 3 | 生成**允许的**学习目标 | `EPISODIC_LEARNING_TARGETS` 白名单（`taiji/config.py:22`）＋任务边界对象 `WorkbenchTaskBoundary`（`score_bytes(..., boundary=...)`） | **有承载** |
| 4 | 候选更新 | `propose_structural_candidate_from_pressure`、`propose_region_split_from_error`、`propose_unit_add`（均只 propose） | **有承载** |
| 5 | 行为与保持检查 | 保持侧配对件 `taiji-n2-04-retention-pair-v1`＋判读器 `scripts/training/adjudicate_taiji_n2_04_retention_pair.py`；能力侧＝`SeedJudge` 的成句与严格真命中两把尺（口径分列，不许互换） | **有承载** |
| 6 | 接受或回退 | `TaijiConsolidation.accepted`（计数）、`can_promote`、回退＝`restore_native`／`SeedRuntime.load` | **有承载** |
| 7 | **记录实际更新对象** | 唯一候选是那枚 `last_decision` 键，而它㊵-570 起现读至今**从未发生**（写侧存在、消费侧为零） | **未承载** ⇒ 由 J-N5k-3 判成 `update_object_ledger_absent`，这是本件预期最红的一格 |

**防自证学习一行（VISION §19.6 的禁令）**：`EpisodicField.PROVENANCE_KINDS` 的四枚取值是
`("experienced", "imagined", "replayed", "external")`，**没有 `generated`** ⇒ 今天自生成答案写回时
只能借用 `external` 或 `experienced`，「未经核验的生成内容保留 generated 来源，不混进事实/奖励」这句
**没有可判的形式**（同 §2 第 2 步与 PLAN-N2-06 §2 缺口 3；本件判它造成的后果，那件判写入合同）。

**support/query 隔离一行**：现成两把尺＝`taiji/foundation_tasks.py:965` 的 `whole_segment_duplicate`
报告与训练侧 `_disjointness`（`scripts/training/adjudicate_taiji_n2_04_retention_pair.py` 等 6 处在用）；
本件**不重造第三把尺**，只要求评价件的每次比较都自带这两枚读数。

## §3 判据（冻；五条，全部零算力，不新增阈值字面量）

* **J-N5k-1 三层必须可独立声明**：任何一次「适应」读数，必须点名它动的是状态／φ／θ 中的**哪一层**；
  同一份读数若同时被记到两层头上 ⇒ 判 `layer_attribution_ambiguous`。执行体＝信封与退出件里的层自述键，
  缺自述记 `layer_self_report_absent`（「没读到」不许读成「没动」）。
* **J-N5k-2 φ 的允许范围必须是白名单而不是散文**：`learning_targets` 的取值必须落在
  `EPISODIC_LEARNING_TARGETS` 五值内，越界必须 `ValueError`（这条今天**应为真**，其反向复算＝
  用表外取值打一次 `write`，断言响亮拒绝）。
* **J-N5k-3 第 7 步必须真的记了对象**：每次被接受的更新，必须能读出「实际更新了哪些张量／槽位」，
  本件对此的现成尺是 `compare_taiji_n5_arm_tensors.py` 的逐槽差（㊵-672 那 20 槽＝10 枚权重就是这种形状）；
  若只有 `last_decision` 这类从不消费的键 ⇒ 判 `update_object_ledger_absent`。
* **J-N5k-4 生成来源不得混入事实**：任何写回经历若产自模型自身，其来源必须与真实反馈可分；
  今天因缺 `generated` 取值而**判不了** ⇒ 记 `generated_provenance_absent`（这条不是失败，是把禁令变成读数）。
* **J-N5k-5 评价必须带两把隔离尺**：每个 support/query 结论必须同时出版 `whole_segment_duplicate`
  与 `_disjointness` 两枚读数（或点名为什么该面不适用）；缺任一枚 ⇒ 该结论记 `isolation_unverified`。

## §4 守卫（三条，各自能为假）

* **G-N5k-1 不新增层、不新增张量参数面**：本件不得为凑三层而造新对象（新增参数面要走「参数面加新张量的三件配套事」流程）。
* **G-N5k-2 不重造尺**：`ruler_usable`（`adjudicate_taiji_n5_shadow_gate.py:295 〔2026-10-11 ㊵-685 行号更正：现行行号＝299。㊵-673 那次提交 `240b20210` 给该文件 +28/−10，把 `ruler_usable` 那行从 295 推到 299；口径未变，引用前按现行行号回查。〕`）、`unit_count`／`shared_numel_delta`
  （`summarize_taiji_n5_resource_face.py`）、逐槽差（`compare_taiji_n5_arm_tensors.py`）三处一律**引用**，
  本件不写第二实现。
* **G-N5k-3 不开跑、不动产品码**：任何「要跑面才能答」的问题记 `unverified`，不许为凑判据起档。

## §5 判据载体在哪（措辞门要求的一节）

J-N5k-1／3 的执行体＝信封／退出件的自述键与 `compare_taiji_n5_arm_tensors.py` 的逐槽差（现成件即可复算）；
J-N5k-2 的执行体＝`taiji/config.py:31` 的校验函数（进程内一次表外取值即可证）；
J-N5k-4 的执行体＝`EpisodicField.PROVENANCE_KINDS` 的取值枚举（现扫）；
J-N5k-5 的执行体＝`taiji/foundation_tasks.py:965` 与 `_disjointness` 的出版位置。
⇒ 本件没有一条判据是手算的；其中 **J-N5k-2 今天即可判真**，J-N5k-1／3／4／5 预期判红或记 `absent`（这正是登记缺口的形状）。

## §6 出口（三条互斥）

* 五判据出值 ⇒ V-21 转为「已展开且已判」，DEBT-G97 的最后一条结清，09 §1.3 的 unmapped 归零；
* 任一判据出值但判不成立（今天预期是 J-N5k-1／3／4／5）⇒ **仍算 V-21 已消化**，
  但必须把每一格点名成批次（§7），不许合并成「架构还没做完」一句；
* 若只能停在「表填了但复算读不到」⇒ 本件只算设计件，收回「三层更新架构已登记」这句。

## §7 落笔即钉的三行

* **预期为绿的成员**：J-N5k-2（白名单校验今天就在，且拒绝路径可构造性验证）、§1 三层各自的承载行（实名已给）。
  **预期为红的成员**：J-N5k-3（第 7 步无账本）、J-N5k-4（缺 `generated` 来源）。
* **仍红时收回的句子并点名**：J-N5k-3 判红 ⇒ 收回「系统知道自己刚改了哪些参数」这句，点名住在
  `last_decision`（写侧有、消费侧无，DEBT-G68 同族）；J-N5k-4 判红 ⇒ 收回「自生成内容不会污染事实」，
  点名住在 `EpisodicField.PROVENANCE_KINDS` 的四枚取值。两处都**不许**靠改判据迁就。
* **发表资格前置**：任何「原生持续适应已分层」的对外表述必须同时给出 ①§1 表逐格实名，
  ②七步流程每步的承载或未承载状态，③J-N5k-2 的表外取值拒绝复算证据。三条缺一即视为未交付。

## §8 不变项

M6 收官、M8 挂起；N5 出口已判＝不成立（㊵-673），资源面已齐（㊵-678）；DEBT-G78 按 owner #24① 继续挂起，
本件不代为接线；`generated` 取值与第 7 步账本今天**只是登记**，本件一个枚举值、一个键都没加。
两件共用的一枚事实（缺 `generated`）在 PLAN-N2-06 §2 与本件 §2 **各写一遍、各判一次**，不相加、不互证。
债表现至 **DEBT-G97**。
