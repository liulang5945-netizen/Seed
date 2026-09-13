# M5 · P5.1e 同预算因果收益 Gate 预注册（sourced vs placebo · 冻结 2026-09-12）

前置事实：P5.1d 语义 encoder 注入 Gate 已通过（disc `0.870033`，九门全过，wall `62.9s`）。本文件在执行前冻结判据；冻结后不训练、不接新数据源、不读取 sealed。

## 1. 假设

在产品边界（`ArtifactInternalizationTrainer` + `SemanticArtifactKnowledgeEncoder`，384 维 MiniLM 锚定）与 governed 真实语料下：**同预算内化中，训练内容与未见任务同族（sourced）所带来的未见任务收益，显著高于训练内容与任务无关（placebo）的收益**。收益操作化为共享 holdout 上的 procedural accuracy delta；单变量 = 训练内容族，trainer 全部超参两臂逐位相同。

## 2. 两臂设计（家族排他）

| 要素 | sourced 臂 | placebo 臂 |
|---|---|---|
| 训练内容族 | A 族（capability 词表 `editor.open/read/inspect`） | B 族（capability 词表 `network.search/index.scan/cache.fetch`） |
| outcome 结构 | 8 active + 4 failed | 8 active + 4 failed（同构同数） |
| partitions | 同 12 skill_id 跨 train(v1)/holdout(v2)/retention(v3)，source digest 不同、unit content 逐字相同（P5.1d 范式） | 同左（B 族） |
| 构造通道 | `DeclarativeSourceRegistry(SkillArtifactAdapter())` + `EvolutionExperienceLedger`；`_register_skill` 按 outcome 走 staged→shadow→active / →failed | 同左 |

- **共享评估面**：4 条**全新 A 族 gate skills**（新 subject/verb 文本、同 capability steps；非 re-version、不属于任何 partition）——在两臂 admit 后的 trainer 本体上做 evaluation-only 复评（`trainer._examples` / `_sequence_accuracy` / `_affordance_mse` / `semantic_value_from_feature`），依据 `consolidate` 的 commit-on-admit 语义（admitted 时三个 trial 提交回 trainer 本体）。
- **扩展 workflow 集**：runner 本地定义每族 16 条（A 族 12 进 partitions + 4 作 gate holdout；B 族 12 进 partitions + 4 作对称评估记录面），steps 冻结沿用族词表；文案风格沿 P5.1c `A_WORKFLOWS` 范式。
- **trainer 配置（两臂逐位相同）**：`feature_dim=384`、`encoder=SemanticArtifactKnowledgeEncoder(...)`、`semantic_pairwise_margin=0.5`、`semantic_passes=12`、`procedural_epochs=250`、`affordance_epochs=200`、`seed=17`，其余默认（P5.1d 冻结配置原样）。
- **ranking_pairs**：`_ranking_pairs` 同构公式（preferred = `target_reward > 0.5` 且族匹配；other = `target_reward <= 0.5`；全笛卡尔积），冻结两臂对数逐项相同（结构等式断言），绝对值入报告不设门。

## 3. 判据门表（九门）

| # | 门 | 判据 |
|---|---|---|
| 1 | static_four_checks | ruff + `black --check` + mypy（`--follow-imports=silent seed taiji`，61 错 HEAD 基线；runner 追加直检）+ pytest，全绿 |
| 2 | same_budget_enforced | 运行时断言：两臂 artifact 数、experience 数、unit_kind 组成、ranking pairs 数逐项相同；绝对计数入报告 |
| 3 | capability_vocabulary_disjoint | sourced 词表 ⊇ A-gate 词表 且 placebo 词表 ∩ A-gate 词表 = ∅（P5.1 断言范式） |
| 4 | arm_sanity_both_arms | 每臂 `consolidate` admission passed=true（semantic + procedural holdout > lesion + retention ≥ 0.5 + affordance native < frozen，四条件全过） |
| 5 | **content_transfer_margin（核心）** | A-gate procedural accuracy：sourced − placebo ≥ **0.15**（P5.1 冻结常数原样继承：探针实测 delta 0.5 的 30%） |
| 6 | sourced_beats_lesion | sourced 臂自身族 holdout accuracy > lesion holdout accuracy（收益非器官噪声；admission 内含、显式断言） |
| 7 | affordance_content_specificity | sourced native affordance holdout MSE < frozen 且 < placebo native（关系门，免设尺度相关 MSE margin） |
| 8 | checkpoint_roundtrip_both_arms | 两臂 checkpoint 往返 digest 一致 |
| 9 | deterministic_and_budget | 同 seed replica 报告逐位一致；wall ≤ 1200s（继承 P5.1d 预算，非 P5.1 的 300s） |

## 4. 三态映射

- `same_budget_content_benefit_supported`：九门全过。
- `content_benefit_insufficient`：门 1–4、6–9 全过而核心门 5 delta < 0.15 → 如实记录实测 delta，回 S 轴归因；不回改判据。
- `failed`：任一机械门（1/2/3/8/9）或 arm-sanity 门（4/6/7）失败 → 构造或基建问题，不构成内容收益判据。

## 5. 诚实边界

- 语料为 governed 构造 fixtures，非生产语料；unseen = 同族新内容（新 subject/verb 文本、同 capability steps），非全新词表。
- **语义器官收益不以本 gate 度量**：家族排他臂下，holdout 价值预测正确性对常数/中偏预测器不构成区分（ill-posed）；语义辨别力已由 P5.1d disc `0.870033` 建立，本 gate 语义面降为臂内 admission sanity（门 4）。
- quarantine 拒绝行为继承 E4/P5.1 认证，本 gate 不重复设门；`growth_admitted=false`、`can_promote=false` 贯穿，不授执行权限。
- 绝对计数（examples/experience 换算率）首次执行前不可前导冻结：仅结构等式设门，绝对值记录入报告。
- `fit_called` 语义沿 P5.1 口径：encoder 拟合断言不作为本 gate 判据面。

## 6. 产物顺序

1. `scripts/training/eval_taiji_p5_1e_same_budget_content_benefit_gate.py`（静态四项先行）；
2. 执行落盘 `reports/taiji_p5_1e_same_budget_content_benefit_20260912.json` + 与本文件对账；
3. 路线图同步 + 独立提交。

## 7. 对账条款

报告以实测为准；若实测与本文件判据冲突，如实报告，禁止反向修改判据或报告凑数。

---

## 修订附录（2026-09-12，gate 执行前）

**范围**：§2 ranking_pairs 构造公式一处修订；其余章节（九门判据、三态映射、§5 诚实边界、§7 对账条款）不变。

**修订内容**：`_ranking_pairs` 在全笛卡尔积之上追加**可区分性剪枝**——跳过 `p.grounding.equal(o.grounding)`（张量逐位相等）的配对。preferred/other 集合定义不变（preferred = `target_reward > 0.5` 且族匹配；other = `target_reward <= 0.5`）；两臂对数逐项相同的结构等式断言不变；绝对值仍入报告不设门。

**修订理由（实现事实，执行前发现）**：

1. learner `_apply_pairwise_update`（`taiji/internalization_learner.py`）在配对特征 squared_norm ≤ 1e-12 时硬性 `raise ValueError("pairwise preference requires distinguishable grounded features")`——zero-norm 配对使 `consolidate` 无法完成。
2. P5.1d 的跨族过滤使 preferred/other 特征结构性不重合（其 runner docstring 明示该保证）；本 gate 同族臂下该保证失效：
   - 同一 failed skill 的 discovered（status=success，`target_reward=+1.0`）与 failed（`target_reward=-1.0`）两次 lifecycle 事件共享同一 `source_digest` → 匹配同一批 artifact unit → 同 grounding 冲突对；
   - 同族 skills 共享族 capability 词表（`editor.open/read/inspect`）→ affordance unit content 均为 `{"value": <capability 字符串>}` → 跨 skill 同 capability affordance examples 的 grounding 全同。
3. 同 artifact 冲突对源于 lifecycle 事件语义本身，不可通过构造消除；capability 词表为 §2 冻结原文（`steps 冻结沿用族词表`），不宜为绕开碰撞而改动；故 grounding 相等剪枝是使冻结公式可执行的最小修订。

**与判据的关系**：核心门 5（content_transfer_margin）度量 procedural organ 在 A-gate 上的 accuracy delta，与 pairs 构造无关；门 2 结构等式在两臂同构语料 + 确定性 embedder 下剪枝后仍逐项成立；门 3 词表判据按原文直接操作化（capability 字符串集合），不受剪枝影响。§7 条款不受影响：本修订执行前落盘、依据为冻结代码事实，非按结果改判据。

---

## 修订附录二（2026-09-12，gate 执行前）

**范围**：§2 gate 评估记录构造与 gate workflow 构造规则两处操作化修订，并显式冻结 gate skill id；九门判据数值、三态映射、§5 诚实边界、§7 对账条款均不变。

**修订内容**：

1. **designated-label 记录构造（门 5 评估面）**。partitions 冻结为 `skill.p51e.a.01..12`（train v1 / holdout v2 / retention v3 各 4 条），gate skills 冻结为全新 id `skill.p51e.a.13..16`（与 partitions 无碰撞）。每条 gate skill 的 records：actual kind = designated in-vocabulary label——按 index 对齐复制对应训练 workflow 0..3 的 capability steps 逐字文本，actual kind 冻结为 `skill.p51e.a.01..04`（非 gate id、非 capability 字符串）；每 skill 4 条 records 共享同一 procedure cue（与 lifecycle 语料 4 事件共享 procedure cue 的结构同构），`episode_id=f"p51e-gate:<gate_id>"`、tick 1–4、`outcome=None`（不伪造 outcome）。gate 投影走 `SkillArtifactAdapter.project(artifact, partition="gate")`、version `"1"`，不进 registry/admission 流。
2. **gate workflow 构造规则**。gate workflow = 复制对应训练 workflow 的 capability steps（逐字）+ 全新 name/description/target 族内文本。procedure cue 由 steps + target 决定：新 target 使 gate cue 偏离训练 cues，门 5 因此度量真实 cue→kind 内容泛化，而非 cue 恒等记忆。
3. **共享评估面适配**。gate 语义面直接调用该臂 trainer 本体公开接口：`trainer.encoder.encode(artifact)` + `semantic_value_from_feature`（不经过 `_examples`——后者按 source_digest 匹配 experiences，gate 无 experiences）。gate-skill affordance MSE **不计算**（需伪造 reward，违反 §5 诚实边界）；门 7 native affordance 口径不变（仍由 consolidate 内部 holdout 通路产出）。两臂各自使用该臂 trainer 本体的 encoder/organ 评估。gate records 计数结构等式纳入门 2。

**修订理由（实现事实，执行前发现）**：

1. P5.1e 语料为 lifecycle 范式（registry→ledger，事件 capability_id = skill_id）；P5.1d runner `_ranking_pairs` 的 `_family(item.action_kind) = str(...).split(".")[2]` 确证两臂 procedural 输出空间 = 各自 12 个训练 skill_id。
2. `ProceduralSequenceLearner.readout`（`taiji/procedural_memory.py`）在首次 consolidate 后冻结（nn.Linear，kinds 变更 raise）；`predict_episode` 对 readout argmax 后只能返回 `self.action_kinds`（训练词表）。
3. 若 gate records 的 actual kind 取 gate id 或 capability 字符串，`_sequence_accuracy` 中 actual ∉ predicted 词表 → accuracy 恒为 0/0，门 5 结构性必败。
4. 0.15 常数的来源口径（P5.1 probe）即为 designated-label 同构：probe 用 capability-kind 语料走 consolidate holdout 通路，placebo accuracy 由词表排他结构性为 0，delta = sourced accuracy。本修订与该口径对齐；chain-mate 混淆如实披露——12 训练 skills 覆盖 3 条族链旋转，mate 级区分依赖 target 文本（训练任务本身如此）。

**与判据的关系**：门 5 阈值 0.15、`_sequence_accuracy` 口径、三态映射均不变；placebo accuracy ≡ 0 为词表排他的结构性结果，如实披露于报告；gate skill 身份不变（全新 id、非 re-version、不属于任何 partition、无 experiences 不伪造 outcome）；§7 条款不受影响：本修订执行前落盘、依据为冻结代码事实，非按结果改判据。

---

## 修订附录三（2026-09-12，gate 执行前）

**范围**：附录二两处执行面修正（partitions 读法勘误 + gate 投影 partition 标签）；九门判据数值、三态映射、§5 诚实边界、§7 对账条款均不变。

**修正内容**：

1. **partitions 读法勘误**。附录二第 1 条括注「train v1 / holdout v2 / retention v3 各 4 条」为起草笔误；按 §2 表冻结原文（「同 12 skill_id 跨 train(v1)/holdout(v2)/retention(v3)」）执行：12 个 skill_id 全部跨三分区（每分区 12 skills @ v1/v2/v3），每分区 8 active×4 lifecycle 事件 + 4 failed×2 事件 = 40 experiences。依据（冻结代码事实）：`procedural_trial.consolidate(procedural_train, ...)` 仅以 train 记录发现并冻结 readout 词表（`taiji/artifact_internalization.py` 635–640 行；`ProceduralSequenceLearner._ensure_readout` 首次 consolidate 冻结词表，词表变更即 raise "sequential procedural action kinds changed after consolidation"）；若按 4/4/4 拆分，holdout/retention 的 actual kinds ∉ train 词表 → accuracy ≡ 0 = lesion 水平 → 门 4 结构性必败。
2. **gate 投影 partition 标签**。附录二第 1 条 `SkillArtifactAdapter.project(artifact, partition="gate")` 不可执行：`EVOLUTION_PARTITIONS = ("train", "holdout", "retention", "security")`（`taiji/evolution_experience.py` 32 行）且 `EvolutionCorpusArtifact.__post_init__` 强制校验 partition（249 行 `_partition()`），`"gate"` 构造即 raise。改为 `partition="holdout"`（evaluation-only 标签）：gate 投影单元不进 registry / ledger / admission / consolidate 任何流，标签仅用于 runner 本地定位 procedure unit 并编码 cue，语义惰性，不触碰任何训练数据。

**与判据的关系**：九门判据、三态映射、§5、§7 全部不变；本修正仅使冻结构造可执行，执行前落盘、依据为冻结代码事实，非按结果改判据。
