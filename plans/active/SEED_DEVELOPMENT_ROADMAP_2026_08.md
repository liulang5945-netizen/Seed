# Seed / Taiji Native v1 开发路线（2026-08）

状态：**当前唯一执行路线**

更新时间：2026-08-26（Taiji 架构重新定基线）

## 1. 目标与纠正后的边界

Taiji 是完整原生认知架构；Seed 是项目、产品和运行时。路线不再以“把 Transformer 各功能换成更原始的神经元算子”为目标，而以项目原始需求中的异质协作、`1+1>2`、自适应激活、身体—生命闭环、睡眠/玩耍、持续学习、自我成长，以及可学习表征、世界模型、记忆、目标、推理、规划和生成的完整闭环为目标。

```text
Seed runtime hosts Taiji
Taiji owns perception → cognition → learning → action
TSK-v8 is a reusable kernel, not the completed architecture
Legacy NeuroPlex is a frozen offline comparison
```

长期目的见 [TAIJI_CORE_REQUIREMENTS.md](TAIJI_CORE_REQUIREMENTS.md)，权威设计见 [TAIJI_NATIVE_ARCHITECTURE_V1.md](TAIJI_NATIVE_ARCHITECTURE_V1.md)。旧 raw-byte substrate 路线完整保存在 [旧路线归档](../archive/history/SEED_DEVELOPMENT_ROADMAP_PRE_NATIVE_V1_20260825.md)，其中 R1–R5 不再是当前研究执行顺序。

## 2. 当前事实

### 2.1 可以保留的成果

- `taiji/` 已与 `neuroplex`、`transformers` 解耦，具备独立命名空间、状态和 checkpoint。
- TSK-v8 已有 raw-byte codec、持续 predictive fabric、局部学习、情景原型、行动闭环和 K 系列机制证据。
- `CapacityPolicy`、参数预算、CPU/CUDA device 语义和恢复测试可以成为新架构的资源治理基础。
- Seed 原生 API、训练入口、桌面产品和 Legacy 开关已有可复用工程基线。
- S1 产品体验验收已完成；阶段报告和归档体系有效。

### 2.2 不再成立的声明

- raw byte one-hot 不是 Taiji 的完整输入表示，只是文本器官边界。
- `sensor → fabric → episodic field → byte motor` 不是完整认知架构。
- 固定 fan-in、固定区域比例和 257 动作不是 Taiji 身份。
- “没有 embedding/attention/optimizer”不是智能或原生性的充分条件。
- N0–N11/M5–M7 只证明 kernel 机制，不证明概念、推理、语言或 AGI。
- Seed 不再拥有认知器官、目标和自我模型；这些属于 Taiji。

### 2.3 P1 当前实现状态（2026-08-25）

P1 已在当前分支落地为兼容纵切片：

- `taiji/contracts.py` 定义版本化的 `Observation`、`PerceptEvent`、`WorkspaceState`、
  `WorldState`、`MemoryState`、`GoalState`、`PlanState`、`ActionIntent`、`Outcome`，
  并提供可恢复的 `CognitiveState` 与 `NativeCheckpoint`；
- `taiji/adapter.py` 的 `TSKV8Adapter` 保留 TSK-v8 公开 API，同时把观察→感知事件→
  工作空间/世界/记忆摘要→行动意图→环境结果接入 Taiji-owned v1 状态；
- `Seed.architecture` 是正式入口，`Seed.substrate` 仅保留为旧调用方的兼容别名；
- Seed 的 `seed-native-v1` 旧 checkpoint 仍可读取，新 checkpoint 增加 Taiji v1 原子信封，
  认知状态由 Taiji 序列化和恢复；
- 旧 TSK-v8 行为、参数和 kernel checkpoint 仍保持原路径，不把 adapter 的桥接摘要宣称为
  完整 Taiji 智能。

P1 的剩余工作只限于回归门禁与边界维护；下一阶段进入 P2 学习型感知与时间抽象。

### 2.4 P2 当前进度（首个纵切片）

已完成的 P2 基础能力：

- `PerceptionConfig` 把特征维度、局部窗口、assembly 时长、边界阈值和局部学习率纳入配置，
  不再把这些能力参数散落在 adapter 代码中；
- `LearnedPerception` 使用可训练 embedding、局部窗口投影和递归预测投影形成连续特征；
- assembly 结束由变化度、预测误差、最短时长和最长时长共同决定，byte 只保留无损边界编码；
- 感知动态与参数进入 Taiji v1 checkpoint，Seed 不保存第二份感知状态；
- 已有单元测试证明连续特征、可变时长、冻结学习和恢复确定性。

A1 评测合同已建立在 `taiji/evaluation.py`：训练集、未见组合、边界扰动和随机 chunk
对照分开输入；同一个 ridge probe 同时报告 learned assembly 与 byte-only；多个 seed
输出均值、方差和明确的 `gate_passed`，评测器不内置词表或答案映射。

历史阶段记录（当时 P2 尚未退出）：首份无预测训练的 smoke 基线保存在
`reports/taiji_a1_perception_smoke_20260825.json`；tick-level next-byte 训练记录在
`reports/taiji_a1_perception_predictive_20260825.json`，completed-assembly 与边界
校准记录在 `reports/taiji_a1_perception_boundary_20260825.json`。加入 future-window
assembly 目标后的最新报告为 `reports/taiji_a1_perception_assembly_20260825.json`：
primary 未见组合 gain 为 `-0.0088`，marker score delta 为 `+0.0222`（门槛 `+0.05`），
marker rate delta 为 `+0.0098`，random-chunk drop 为 `+0.0077`，跨 seed std 为
`+0.0068`。随后加入 assembly consistency/contrastive 目标，最新报告为
`reports/taiji_a1_perception_contrastive_20260825.json`：primary 未见组合 gain 为
`-0.0225`，marker score delta 为 `+0.0184`，marker rate delta 为 `+0.0401`，
random-chunk drop 为 `+0.0048`，跨 seed std 为 `+0.0101`。A1 Gate 仍为 `false`；
连续的 next-byte、future-window 和 consistency/contrastive 目标都没有让 completed
assembly 稳定超过 byte-only，这已经是 P2 目标定义需要重审的架构信号。

P2/A1 合同已完成一次结构性重定：`taiji/assembly_relations.py` 定义 ordered atom
composition，训练与未见组合共享 atom 但 pair 完全不重叠；
`scripts/training/build_taiji_a1_relation_manifest.py` 已从真实语料生成
`reports/taiji_a1_assembly_relation_manifest_20260825.json`（8 atoms、40 train pairs、
16 unseen pairs）。pair provenance 只存在于评测 metadata，不进入模型输入；旧的
next-byte A1 报告保留为历史对照，不再作为组合关系的唯一合同。

`AssemblyRelationEvaluator` 已对该 manifest 生成
`reports/taiji_a1_assembly_relation_baseline_20260825.json`。加强后的 relation
subgate 在三个 seed 上通过：slot generalization gain 最小 `+0.75`，boundary
consistency 最小 `0.9825`，random binding drop 最小 `+0.1875`，slot cross-seed
std `0.1062`。这只证明当前结构性组合合同在小规模 manifest 上成立；旧 next-byte
A1 仍未通过，完整 P2 不能因此退出（均为历史结论）。

扩展验证已完成：`dialogue16` 独立 manifest 的 slot gain / boundary consistency /
random binding drop 最小值为 `+0.9375 / 0.9841 / +0.6875`；`shared16` 独立
manifest 为 `+0.9219 / 0.9811 / +0.6406`。因此 relation subgate 已在两个语料
分区、16 atoms、240 个 ordered-pair 组合规模上稳定通过；旧 next-byte A1 仍保留为
失败历史对照，不阻塞结构性 P2 relation subgate 的收口。

截至 2026-08-27，旧 next-byte 评测已升级为动态 assembly、marker-specific boundary
evidence、multi-step credit 和跨 assembly 后段负样本合同；smoke `32/16` 与独立
`shared_core 128/64` 两级正式报告均已通过 A1 Gate。上面的旧报告和失败数字只作为
演进轨迹保留，不再代表当前 P2 状态。

## 3. 执行原则

1. 能力合同先于模块命名和代码目录。
2. 每阶段交付一个可执行纵切片，禁止只创建空框架。
3. raw input、内部表征和最终输出必须分层，不能共享一个 byte alphabet 冒充认知。
4. 每个新增能力都要有 holdout、损伤实验和失败标准。
5. development training 与 lifetime learning 分开报告；辅助优化不能冒充自主在线学习。
6. CUDA 优化只针对已证明必要的 v1 算子，不绑定旧 kernel 拓扑。
7. Legacy 继续冻结，不进入 Taiji forward，也不因本轮重构立即删除。
8. 优先复用成熟的 embedding、路由、状态空间、图计算、优化器、强化学习、检索和 CUDA 方法；只有它们无法满足持续状态、因果闭环、终身学习或资源约束时才自研替代。

## 4. 阶段顺序

| 阶段 | 目标 | 退出结果 |
|---|---|---|
| P0 | 架构重新定基线 | 权威身份、目标架构、旧 kernel 和旧路线边界清晰 |
| P1 | v1 合同与兼容骨架 | 新状态/事件/所有权合同可执行，TSK-v8 行为和 checkpoint 不回退 |
| P2 | 学习型感知与时间抽象 | 从 byte 流形成可变长度 assembly，并在未见组合上迁移 |
| P3 | 世界状态与工作空间 | 对象/事件/关系持续存在，选择性路由支持多步任务 |
| P4 | 情景、语义与程序性记忆 | one-shot 经历可巩固成可迁移概念和技能 |
| P5 | 目标、推理、想象与规划 | model-based rollout 在未见目标上优于 reactive baseline |
| P6 | 原生语言与工具行动 | 内容计划经表达器官稳定生成可读语言和结构化工具调用 |
| P7 | 持续进化、多模态与规模化 | 内生调节/结构成长、保持旧能力、跨模态迁移、真实 CUDA 与资源治理闭环 |
| P8 | 产品原生化与公开测试 | Seed 默认发行只承载真实 Taiji v1 能力，达到发布门槛 |

阶段严格按 P0 → P8 推进。产品安全修复可以并行，但不得用 UI、Agent 壳或 Legacy fallback 伪造尚未完成的 Taiji 能力。

## 5. P0：架构重新定基线

### 工作项

- 把 Taiji 定义为完整认知架构，把 Seed 定义为产品/运行时。
- 新建 Taiji Native Architecture v1 权威合同。
- 将当前精确方程规范和旧路线移入 archive，定位为 TSK-v8 历史/兼容证据。
- 更新 README、计划入口、边界测试和能力声明。
- 停止旧 R4 语言长训、旧 R5 机制叠加和针对固定 fan-in 的 CUDA kernel 工作。

### 退出门槛

- active 文档不存在“Seed 是认知模型主体”“TSK-v8 是完整 Taiji”的冲突。
- 旧文档链接可追溯，归档中的旧“下一步”全部失效。
- 架构/命名边界测试通过，`main` 干净并有单一提交。

## 6. P1：v1 合同与兼容骨架

### 工作项

1. 定义 `Observation`、`PerceptEvent`、`WorkspaceState`、`WorldState`、`MemoryState`、`GoalState`、`PlanState`、`ActionIntent` 和 `Outcome` 的版本化协议。
2. 建立 Taiji 顶层认知状态和原子 checkpoint envelope；Seed 只保存产品元数据并委托 Taiji checkpoint。
3. 将现有 `Taiji` 类明确包装为 `TSK-v8` compatibility adapter，不立刻移动全部源码。
4. 增加所有权测试：`seed/` 不得新增概念记忆、规划、语言模型或 teacher policy；`taiji/` 仍不得导入 Legacy/Transformer。
5. 建立一个最小纵切片：观察进入、状态推进、产生意图、执行、接收 outcome；初期可调用 kernel，但所有接口使用 v1 语义。

### 退出门槛

- 现有 kernel 测试、Seed API 和 checkpoint 恢复无回退。
- v1 状态能保存、恢复和确定性续跑。
- 认知所有权由 AST/contract tests 强制，而不是只写在计划中。

## 7. P2：学习型感知与时间抽象

### 工作项

- 保留 byte codec，新增可学习局部特征和可变时长 assembly。
- 用预测稳定性、边界惊讶、重复和上下文区分学习 chunk，不提供人工 token 答案。
- 支持低层 byte 纠错通路与高层事件通路并存。
- 建立 A1 数据：未见词形、未见组合、边界扰动、随机 chunk 和 byte-only 对照。

### 退出门槛

- 高层表示相对 byte-only 基线提升 holdout 预测/压缩和未见组合迁移。
- chunk lesion 显著降低跨边界泛化，但不破坏无损 byte 回退。
- assembly 数量和长度由数据形成，不写数据集词表或答案映射。

## 8. P3：世界状态与工作空间

### 8.1 当前实现状态（2026-08-25）

P3 的第一段合同与恢复边界已落地：

- `taiji/contracts.py` 增加 `WorldObject`、`WorldEvent`、`WorldAffordance`、`WorldAction`、
  `WorldTransition` 和 train/holdout 分离的 `WorldInterventionCase/Corpus`；属性和参数不使用
  固定领域字段，而是以可序列化的键值对承载；
- `taiji/world.py` 的 `TaijiWorldState` 持有当前结构化世界状态和可选 transition history，
  并把它们放进 Taiji-owned checkpoint；外部环境只能提交与当前状态严格衔接的 transition；
- 合同、因果 tick、动作—结果绑定、恢复连续性和干预 split 泄漏均有测试覆盖；
- `taiji/world_learning.py` 已提供 data-derived schema、target/parameter 组合特征、状态/结果预测器，
  并以 frequency、action-only、target-binding lesion 做对照；
- `scripts/training/eval_taiji_a2_world.py` 已扩展到对象持续、关系槽位变化和时间打乱 control；3 seed
  的扩展 A2 Gate 通过：`state_error_max=0.0615`、`outcome_error_max=0.2579`、最小 state gain
  `+0.2885`、最小 target-binding lesion drop `+0.2429`，time-shuffled split 的 success accuracy
  为 1.0；报告和 manifest 保存在 `reports/taiji_a2_world_*_20260825.json`；
- 该 Gate 仍只证明固定小型 benchmark 的结构化一步干预，不等于已经学会一般世界动力学；更大对象/关系组合和
  更长跨 episode 持续性仍未通过。
- 多步扩展已通过独立 episode-ID holdout：3 个训练 episode、2 个新 episode 的 3-seed
  `rollout_state_mse_max=0.00536`、`final_state_mse_max=0.00625`、success accuracy=1.0，
  checkpoint continuation=true；该结果仍是小型 move benchmark，尚未接入 adapter 的真实认知循环。
- `TSKV8Adapter` 已接入结构化 transition lineage：`settle_action(world_state=...)` 生成并保存
  `CognitiveState.world_transition`，后续 observe 保留对象/关系/event，native checkpoint 可恢复；
  旧 scalar reward API 保持兼容。该 adapter contract 只证明状态所有权和恢复，不证明 runtime 已能预测世界。
- `TSKV8Adapter` 可注入 `WorldDynamicsLearner`：动作前记录 `world_prediction`，真实 transition 到达后
  回写 state/reward error，并把 learner/schema 与 prediction record 放入 native checkpoint；当前只完成
  误差观测与 lineage 固化，尚未宣称在线权重校正已经有效。
- `WorldDynamicsLearner.online_update(transition)` 已接入 adapter 的 `learn_world` 路径；重复干预测试显示
  error-driven update 优于 no-update lesion，`online_update_count` 和 learner 参数可随 native checkpoint 恢复。
- Workspace 路由已从单一 broadcast 提升为可审计的候选选择合同：`WorkspaceCandidate` 描述候选及其特征，
  `WorkspaceSelection` 保存全体分数、选中 ID、容量和 broadcast；`WorkspaceState` 保存候选与选择结果并校验二者对齐。
- `taiji/workspace.py` 的 `WorkspaceRouter` 用独立候选 scorer 学习相关性，执行 capacity-limited top-k；
  `learned`、`none`、`random` 三种模式分别作为主路径与 workspace lesion，且路由参数、训练计数和 native checkpoint
  可恢复。`TSKV8Adapter.observe/observe_event` 已接入同一 runtime tick，旧调用仍保留 predictive-context 回退。
- Workspace 合同、监督训练、容量约束、none/random lesion、adapter runtime 接入和 checkpoint round-trip 已通过 4 项新测试；
  这只证明路由机制存在，不等于 A3 的异质协作或 `1+1>2` Gate 已通过。
- `scripts/training/eval_taiji_a3_workspace.py` 已注册静态组合窄 Gate：两个异质源各提供目标的一半，干扰源提供独立噪声，
  holdout 为新采样组合；3 seeds 的 learned exact route rate 最小值为 `1.0`，learned MSE 为 `0.0`，相对 strongest-single
  平均 gain `+0.1922`，相对 dense mean 平均 gain `+0.7016`，报告与 manifest 为 `reports/taiji_a3_workspace_*_20260825.json`。
  该结果只关闭 A3 的静态组合子门，不代表多步世界行动、目标规划或通用异质协作已通过。
- `scripts/training/eval_taiji_a3_world_workspace.py` 已把 learned workspace 接入真实 `TaijiWorldState` 两步 transition：
  `assemble` 依据 workspace 组合是否正确产生 outcome，`commit` 只在第一步成功后完成；3 seeds 的 learned final success
  均为 `1.0`、mean total reward 均为 `2.0`，strongest-single/dense final success 均为 `0`，random 平均为 `0.2292`，
  none 为 `0`；报告和 manifest 为 `reports/taiji_a3_world_workspace_*_20260825.json`。这关闭 A3 的小型 world-outcome
  子门，但仍不是长程规划、动态容量或一般异质群体证明。

### 8.2 P4 记忆入口（2026-08-25）

P4 的最小真实经历边界已落地：

- `taiji/contracts.py` 增加 `WorkingMemoryItem`、`EpisodicMemoryRecord`；`MemoryState` 同时保存当前 working items、
  working capacity、检索到的 episodic IDs 和现有语义/程序上下文，旧 payload 仍可按默认值恢复；
- `taiji/episodic_memory.py` 提供容量可配置的 `EpisodicMemoryStore`，以 cue cosine similarity 做内容寻址，记录真实
  `ActionIntent`、`Outcome` 和可选 `WorldTransition`，不依赖领域事实表或固定事件槽；
- `TSKV8Adapter` 在真实 `settle_action` 后写入一条 `EpisodicMemoryRecord`，后续 `observe` 检索相关记录，store 与认知
  state 一起进入 legacy/native checkpoint；
- working item、写入/检索、容量淘汰、真实 outcome 绑定和 checkpoint round-trip 已通过定向测试；P4 阶段原生回归为 `90 passed,
  1 skipped`（另 2 个旧 manifest 测试仍受本机 pytest 系统临时目录权限影响）。该入口只完成经历保持与检索，不宣称
  已形成语义/程序记忆或跨 episode 迁移。
- `scripts/training/eval_taiji_p4_episodic_recall.py` 已完成 cue-conditioned one-shot recall 窄 Gate：8 条训练经历、8 条
  新 episode 查询中，full/episode-ID lesion/checkpoint continuation action recall 均为 `1.0`，retrieval/write lesion 均为
  `0.0`；报告与 manifest 为 `reports/taiji_p4_episodic_recall_*_20260825.json`。这只关闭 fast episodic retrieval 子门，
  不代表多次经历已巩固为可迁移语义。
- `taiji/semantic_memory.py` 的 `SemanticMemoryLearner` 已完成最小 replay/consolidation：从 3 条 `[0,0]→0`、`[1,0]→1`、
  `[0,1]→1` 经历预测未见 `[1,1]→2`，episodic nearest error=`1.0`，semantic error≈`0.0045`；replay lesion error=`2.0`，
  episode-ID lesion 和 checkpoint continuation error≈`0.0045`。报告与 manifest 为 `reports/taiji_p4_semantic_consolidation_*_20260825.json`。
  这是 additive numeric relation 子门，不是一般语义巩固；当前 learner 还未接入 adapter runtime。
- `TSKV8Adapter.attach_semantic_memory()` 和 `consolidate_semantic_memory()` 已把 learner 接入真实 episodic outcome 写入链；
  semantic state 与 episodic store 一起进入 legacy/native checkpoint，相关 ownership/checkpoint 回归通过。该接入只关闭
  runtime ownership 子门，仍需多关系、多噪声和更大 episode holdout 才能评估语义巩固稳定性。
- `scripts/training/eval_taiji_p4_semantic_scale.py` 已完成多因子/噪声扩展：60 条经历覆盖 15 个已见组合，留出全激活组合；
  semantic error≈`0.0082`、episodic nearest error=`1.0`、replay lesion error=`4.0`、episode-ID/checkpoint error≈`0.0082`。
  报告和 manifest 为 `reports/taiji_p4_semantic_scale_*_20260825.json`。该 Gate 仍只覆盖 additive relation，不等价于一般
  语义、程序技能或长程记忆能力。
- `taiji/episodic_memory.py` 已把记录容器改为 insertion-ordered dictionary，重复 memory_id 替换和容量淘汰不再每次
  全表重建；`scripts/training/eval_taiji_p4_capacity_procedural.py` 已通过 `100/1000/10000` 容量/干扰曲线：三档均
  保留准确容量、淘汰最旧目标并召回最新记录。
- `taiji/procedural_memory.py` 已加入数据驱动的 `ProceduralMemoryLearner`：动作类别从 `action_intent.kind` 发现，四类
  cue→action holdout 准确率=`1.0`，skill lesion=`0.25`，episode-ID lesion 与 checkpoint continuation=`1.0`；报告和
  manifest 为 `reports/taiji_p4_capacity_procedural_*_20260825.json`。当前仍是独立 consolidation 原型，尚未接入
  adapter action selection。
- `TSKV8Adapter` 已拥有 procedural runtime ownership：显式 action-kind affordance 合同接入 `act()`，由 learner 预测类别并
  改写真实 motor action；`settle_action()` 将类别写回 `ActionIntent.kind`，adapter consolidation 与 native checkpoint
  均已接通。runtime Gate 为 procedural=`1.0`、route lesion=`0.0`、episode-ID/checkpoint=`1.0`，报告和 manifest 为
  `reports/taiji_p4_procedural_runtime_*_20260825.json`。
- `taiji/procedural_memory.py` 已增加 `ProceduralSequenceLearner`：按 episode/tick 聚合多步轨迹，使用 recurrent procedural
  context；未见 transition transfer=`1.0`、相似 cue 干扰后=`1.0`、checkpoint=`1.0`，容量等于原训练集并加入干扰后准确率=`0.5`。
  报告和 manifest 为 `reports/taiji_p4_procedural_robustness_*_20260825.json`。这证明序列技能与资源失忆边界，不代表规划。
- `taiji/homeostasis.py` 已提供事件驱动 `HomeostaticController`，adapter 在 observation/outcome 更新内部状态，并把 controller
  配置纳入 native checkpoint；高误差/负 reward 自动选择 sleep，sleep/play/fixed/random/no-modulator lesions 均通过。报告和
  manifest 为 `reports/taiji_p4_homeostasis_*_20260825.json`。
- `taiji/planning.py` 已提供 `GoalPlanner`：对真实 `WorldAction` 候选综合 predicted reward、success、progress、uncertainty、
  resource cost 和 conflict，adapter 通过 goal→plan→act→outcome 更新 progress 并恢复 plan；safe/risky 单步 Gate 与
  reward-only lesion 已通过，报告和 manifest 为 `reports/taiji_p5_goal_planning_*_20260825.json`。当前仍需多步 imagined rollout
  和 prediction-error-driven replanning。
- `ImaginedRollout` 与 `GoalPlanner.plan_rollouts()` 已接入 adapter：planner 选择 safe 2-step rollout，记录 imagined provenance/
  confidence；真实首步预测误差=`0.6` 超过 threshold 后设置并 checkpoint `replan_required`。报告和 manifest 为
  `reports/taiji_p5_imagined_rollout_*_20260825.json`。当前仍需实际替代 rollout 执行与 confidence calibration。
- `scripts/training/eval_taiji_p5_replan_calibration.py` 已验证实际替代执行：safe rollout 失败后 replan 选择 risky rollout，
  首次 confidence=`0.0`、替代成功后 confidence=`1.0`，两条 success calibration 均在 native checkpoint 中恢复；报告和 manifest
  为 `reports/taiji_p5_replan_calibration_*_20260825.json`。当前仍需 delayed reward、环境干预和 reactive/value/world-model
  lesion 扩展。
- `scripts/training/eval_taiji_p5_intervention_latency.py` 已通过 delayed reward/intervention 窄 Gate：完整 planner 选择
  delayed-safe，reactive 与 discount=0 world-model lesion 选择 immediate-risky，成功概率 gain=`0.4`；真实干预触发 replan
  并执行 recovery，最终 goal progress=`0.16`。报告和 manifest 为 `reports/taiji_p5_intervention_latency_*_20260825.json`。
- `taiji/generation.py` 已通过结构化 generation 窄 Gate：`ActionIntent → ContentPlan → ExpressionPlan → ToolCall → UTF-8 codec`
  保持 intent kind、semantic slots、tool name 和 goal provenance，并可由 `ToolCall.to_world_action()` 回到同一 intent 的因果行动合同；
  `TSKV8Adapter` 的 generation trace/controller 已纳入 native checkpoint。报告和 manifest 为
  `reports/taiji_p6_generation_*_20260825.json`。该结果不等于语言流畅性、自主内容创造或真实外部工具成功。
- `TaijiToolEnvironment` 与 `TSKV8Adapter.execute_tool_call()` 已通过 tool execution/outcome 窄 Gate：模拟环境执行结构化调用后，
  `Outcome` 保持 intent ID、success、reward、terminal 并进入 episodic memory；关闭 generation organ 的 direct-byte lesion 无法执行
  同一工具合同。报告和 manifest 为 `reports/taiji_p6_tool_execution_*_20260825.json`。该结果不代表外部服务可靠性或失败恢复。
- `scripts/training/eval_taiji_p6_tool_failure_replan.py` 已通过 tool failure/replan 窄 Gate：首次工具失败 prediction error=`2.0`、触发
  `replan_required`，recovery tool 成功后清除重规划，两个 Outcome 均保留在 episodic memory。报告和 manifest 为
  `reports/taiji_p6_tool_failure_replan_*_20260825.json`。该结果不代表外部服务可靠性或通用长程规划。
- `scripts/training/eval_taiji_p6_unseen_tool_transfer.py` 已通过 unseen-tool/parameter transfer 窄 Gate：未见工具名、嵌套参数与重排
  key 顺序均保持并成功执行；同时 `act(world_action=...)` 已保留通用结构化参数与兼容 action metadata，报告和 manifest 为
  `reports/taiji_p6_unseen_tool_transfer_*_20260825.json`。该结果关闭固定工具表/扁平参数假设，不代表广泛工具生态泛化。
- `scripts/training/eval_taiji_p6_cross_organ_expression.py` 已通过 cross-organ expression consistency 窄 Gate：同一 `ContentPlan` 同时
  生成 tool/text 结构化表达，content ID、semantic slots、confidence 和 goal provenance 一致，只改变 modality/channel；报告和
  manifest 为 `reports/taiji_p6_cross_organ_expression_*_20260825.json`。该结果不等于语言流畅性。
- `taiji/content_selection.py` 已通过 learned content selection 窄 Gate：utility 在相同候选下按 world uncertainty 选择不同 semantic
  content，并从 checkpoint 恢复选择；报告和 manifest 为 `reports/taiji_p6_learned_content_selection_*_20260825.json`。这是独立
  selector 证据，尚未宣称 adapter runtime 已拥有内容选择或开放域语义生成。
- `scripts/training/eval_taiji_p6_content_runtime_ownership.py` 已通过 runtime content-selection ownership 窄 Gate：adapter 读取
  current goal/world state 选择 content、生成 `ExpressionPlan`，native checkpoint 恢复 selector/decision/expression；报告和 manifest
  为 `reports/taiji_p6_content_runtime_ownership_*_20260825.json`。selector 仍需真实 Outcome 在线 credit assignment。
- `scripts/training/eval_taiji_p6_online_content_credit.py` 已通过 online content credit assignment 窄 Gate：真实 adapter reward 对已选
  semantic content 执行一次 utility 更新，失败候选降权、成功候选提升并迁移，prediction error/training step/applied 标记进入 checkpoint；
  报告和 manifest 为 `reports/taiji_p6_online_content_credit_*_20260825.json`。该结果不代表开放域语义学习。
- `scripts/training/eval_taiji_p6_holdout_content_transfer.py` 已通过 holdout content transfer 窄 Gate：训练未见的 intent kind、候选 ID
  与嵌套 slot 结构仍按 learned context utility 被选中并由 checkpoint 恢复；报告和 manifest 为
  `reports/taiji_p6_holdout_content_transfer_*_20260825.json`。该结果不代表开放域语义发明。
- `scripts/training/eval_taiji_p6_text_organ_codec.py` 已通过 text organ codec 窄 Gate：holdout `ContentPlan` 的 text expression 经
  UTF-8 codec 后 semantic slots、confidence、`source_goal_id` 无损恢复；报告和 manifest 为
  `reports/taiji_p6_text_organ_codec_*_20260825.json`。该结果不等于自然语言流畅性、句法或语言智能。
- `scripts/training/eval_taiji_p6_language_organ_boundary.py` 已通过 terminal language-organ boundary 窄 Gate：可替换的
  `LanguageOrgan` 只接收 Taiji-owned `ExpressionPlan`，默认 `structured-stub` 输出可回解码文本；detached-organ lesion、native
  checkpoint 和参数/认知不变性均通过。该结果只证明末端器官所有权与替换边界，不等于自然语言流畅性、句法或 decoder 智能。
- `LanguageBackendRegistry` 与 `LanguageTrainingExample` 窄 Gate 已通过：registry 可登记未来成熟 decoder，但强制 text modality 与
  `owns_cognition=False`；训练样本固定为 `ExpressionPlan → target_text`，可独立 checkpoint/holdout，不把目标、记忆或
  `ActionIntent` 注入 decoder。该结果只证明接入/训练数据边界，不等于 decoder 能力。
- `ExternalTextDecoderLanguageOrgan` external decoder realization/lesion 窄 Gate 已通过：通过注入的 prompt builder 调用外部
  `generate()`，输入仍只有 Taiji-owned `ExpressionPlan`；detached-organ lesion 通过，且 Taiji 核心未导入 Legacy/Transformer。
  该结果只证明外部适配器边界，不等于具体模型已加载、训练质量或自然语言流畅性。
- P6 decoder provider inventory 与真实 provider smoke Gate 已完成：当前项目有 `0` 个 `data/neurons` Legacy 权重、`4` 个
  `seed-native-v1` 原生 checkpoint 和 `11` 个 Legacy tokenizer 文件；但本机 Hugging Face 缓存提供 Qwen2.5-0.5B-Instruct 权重与
  tokenizer。该 provider 已通过真实 `generate()`、非空文本、detached-organ lesion、认知不变、registry checkpoint 和训练合同
  Gate；这只是外部 provider smoke/ownership 结果，不等于语言质量或通用智能。
- P6 Qwen 多样化 holdout realization 质量 Gate 未通过：3 个 holdout 的非空率=`1.0`、结构化字段泄漏率=`0.0`，但必需语义词
  覆盖率仅=`0.5`；decoder 会生成文本，却丢失或改写关键 slot。Qwen 因此暂不能作为“语义保真”的已验收语言器官，只能作为
  外部候选 provider。
- P6 Taiji-owned realization validator/fallback Gate 已通过真实 Qwen：3 个 holdout 中 1 个文本通过语义检查，2 个丢失 slot 的
  输出被拒绝并回退为无损结构化表达；`safe_realization_rate=1.0`、`fallback_count=2`，且 organ lesion/认知不变通过。该 Gate
  证明安全边界，不等于 Qwen 语义质量已达标。
- P6 runtime semantic constraint/feedback 窄 Gate 已通过：`ContentPlan.required_terms` 是语义保真约束的唯一运行时来源，
  自动传播到 `ExpressionPlan`；评估脚本不再维护第二份 content-ID 映射。语言回退会更新已选 content 的在线信用、标记
  `replan_required`，并在 legacy/native checkpoint 中恢复；真实 Qwen guard 复跑仍为 `safe_realization_rate=1.0`、`fallback_count=2`。
- P6 language fallback/replan 窄 Gate 已通过：首个缺失必需语义词的 `status` 表达被安全回退并产生 `prediction_error=1.0`，
  Taiji 排除失败候选后选择 `recovery`，生成的新 `ExpressionPlan` 通过验证；最终 `replan_required=false`，且 checkpoint 恢复
  替代 content 与 fallback 计数。该 Gate 证明回退信号已被 planner 消费，不代表开放域语言质量。
- P6 language train/holdout boundary 与 provider baseline Gate 已通过：`LanguageTrainingCorpus` 强制 train/holdout 非空、样本 ID
  与 expression ID 跨 split 不重复，并可 checkpoint round-trip；真实 Qwen provider 在未更新权重的前提下完成 2/2 train、2/2 holdout
  测量，holdout 非空率=`1.0`、必需语义词覆盖率=`0.75`、结构化泄漏率=`0.0`。该 Gate 证明数据边界和基线测量，不宣称已训练 Qwen。
- P6 rollbackable provider trainer Gate 已通过：真实 Qwen 上以 `peft-LoRA` 更新 `270336` 个外部 adapter 参数，4 epochs/16 steps，
  共享词汇与未见组合 holdout 的必需语义词覆盖率从 raw=`0.75` 提升到 adapted=`1.0`，结构化泄漏率=`0.0`；关闭 adapter 后输出与
  raw 完全一致，base checkpoint 未修改，Taiji cognition 仍可 lesion。该 Gate 证明外部器官训练和回滚边界，不等于开放域语言智能。
- P6 trained-provider safety integration Gate 已通过：加载训练后的 LoRA 到原始三类多样化 holdout，raw 必需语义词覆盖率=`0.5`，
  guarded adapted 的 `safe_realization_rate=1.0`、`fallback_count=1`；fallback case 触发 `replan_required`，后续新 episode 不继承
  stale signal，关闭 adapter 后输出与 raw 一致，cognition lesion 通过。该 Gate 允许“可验证的外部器官候选”，不自动把它设为产品默认。
- P6 provider artifact/loader Gate 已通过：`LanguageProviderArtifact` 统一记录 base model、adapter、train/safety report、rollback strategy
  与 mode；integration-edge loader 成功加载 guarded LoRA，artifact checkpoint round-trip 与 cognition unchanged 通过，且
  `default_enabled=false` 强制保持 opt-in。raw/LoRA/guarded 不再依赖散落路径或隐式分支。
- P6 client input-boundary Gate 已通过：`InputFrame` 版本化承载客户端原始 bytes 与来源元数据；`TSKV8Adapter.ingest_input()` 将
  当前支持的 text/text-utf8/text-byte 输入逐字节转换为 Taiji-owned `Observation/PerceptEvent`，`InputTrace` 提供可检查的
  感知轨迹并支持合同 round-trip。`ActionIntent` 在该边界保持为空，禁止固定 intent 映射；`SeedRuntime.chat` 已通过
  `generate_input()` 走同一输入合同，仍保留 raw-byte 兼容输出。该 Gate 只证明输入所有权与感知可观测性，不证明 executive、
  语义对话或语言智能。
- P7 executive contract Gate 已通过：`ExecutiveController` 使用 percept/world/memory/goal/homeostasis 派生 context 学习候选
  utility，输出保持同一候选携带的结构化 `ActionIntent + ContentPlan`；`TSKV8Adapter` 已接入选择、Outcome 反馈、native checkpoint
  和 parameter surface。该 Gate 只证明学习型候选选择与所有权，不证明真实环境 action/outcome 闭环或语言智能。
- P7 executive environment-loop Gate 已通过：`ExecutiveDecision` 通过显式 `WorldAction` 元数据和 motor `action_symbol` 接入
  `TaijiEnvironment.step()`，真实 `EnvironmentOutcome` 回写 executive utility、下一感知并触发失败重规划；selected/alternative、
  checkpoint continuation、utility update 与 executive lesion 均有测试。该 Gate 不伪造环境 after-state，不证明长程规划或通用智能。
- P7 candidate synthesis contract Gate 已通过：adapter 从当前 `PerceptEvent`、`WorldState.affordances` 和 active `GoalState` 自动产生
  带 provenance 的 `ExecutiveCandidate`，不需要客户端候选表；当前 affordance 特征仍是保守 scaffold，不宣称已学会通用 affordance
  表征。
- P7 affordance feature transfer Gate 已通过：`WorldAffordance` 携带带 provenance 的 numeric grounding，`LearnedAffordanceFeatures`
  由 Taiji-owned outcome objective 学习连续投影；candidate synthesis 只消费该投影，不读取 `affordance_id/action_kind` 查表，未见
  affordance/action holdout 已通过，且 native checkpoint 可恢复该 source。
- P7 affordance online-credit Gate 已通过：真实 `EnvironmentOutcome` 的 reward 会回写当前 selected affordance 的 feature source；source
  lesion 会阻断候选合成，online update 计数、预测误差和权重可经 native checkpoint continuation 恢复。
- P7 contextual grounding Gate 已通过：adapter 强制 source 的 `context_dim` 对齐 Taiji perception，producer 读取
  `Percept.features + WorldState.latent + uncertainty`；world latent 缺失时使用显式 percept fallback，context 改变会改变连续表示，
  组合/扰动 holdout 已通过。
- P7 world-grounding lineage Gate 已通过：adapter 在 `observe_event` 与 `settle_action` 进入认知状态前统一由
  `WorldAffordanceGroundingProducer` 从 actor/target numeric object summary、relation binding、world latent 和 confidence 生成 raw
  grounding，并记录 `grounding_lineage`；`action_kind/affordance_id` 不参与特征查表。
- P7 end-to-end grounding transfer Gate 已通过：`WorldAffordanceGroundingProducer → LearnedAffordanceFeatures → ExecutiveController`
  在新对象、新关系谓词和新 action kind 的 holdout 上保持正确选择；producer lesion 会使选择退化，证明 executive 消费的是 grounding
  表征而非符号表。
- P7 grounded multi-step environment Gate 已通过：`EnvironmentOutcome.world_state` 进入真实 `WorldTransition` 后，adapter 在行动前后
  都保留 `grounding_lineage`；失败 action 触发 alternative replan，原决策的 delayed credit 可跨 replan 与 native checkpoint 恢复，
  并继续更新对应 affordance source。
- P7 grounded multi-step train/holdout Gate 已通过：4 条 train affordance、未见 actor/target/relation/action kind 的 holdout 和 3 个 seed
  均达到 holdout selection、四步链路中前三步连续 failure replan、全程 before/after lineage、checkpoint pending credit 与跨步 delayed credit `1.0`；
  manifest/report 为 `reports/taiji_p7_grounded_multistep_*_20260825.json`。该结果仍是小型数值世界 transfer，不代表通用关系推理。
- P7 grounded multi-step causal-lesion Gate 已通过：3 个 seed 的 producer lesion 均使 holdout 选择退化，feature-source lesion 均阻断候选
  合成，跳过 delayed credit 均少一次 source/executive online update；结果与主 Gate 一起写入同一 report。该结果证明当前控制变量有因果
  效应，不代表长程规划。
- P7 variable-horizon episode Gate 已通过：同一 train/holdout 学习结果在 3/4/5 步 episode、不同失败位置和多个 after-state relation
  变化下，3 个 seed 均完成预期 replan、全程 lineage 与每个非终止步的 delayed credit。该结果扩大了 horizon 边界，但仍不是长程规划证明。
- P7 executive-to-world prediction/calibration Gate 已通过：executive bridge 现在把带 actor/target 的 `WorldAction` 送入 `WorldDynamicsLearner`，真实
  after-state settle 回写 state/reward error；data-derived schema 的 train/holdout 为 `2/2`，3 个 seed 均在逐条真实转移的 online correction 后降低状态预测误差，
  no-online-update clone 保持原误差。reward error 继续独立记录，不与状态校准混成一个指标；该 Gate 只证明窄数值世界上的预测误差可回写并校准，不证明开放世界预测精度。
- P7 runtime calibration trace contract 已通过：每次带 `EnvironmentOutcome.world_state` 的结算都会把真实 `WorldTransition`、预测 state/reward error、是否执行 online update 及更新前后计数写入 `CognitiveState.world_calibration_trace`；历史容量由 `TaijiConfig.world_calibration_history_limit` 管理，并随 native checkpoint 恢复。该 Gate 证明运行时 ownership 和可恢复性，不代表多步 runtime calibration 已完成。
- P7 runtime calibration trace multi-step Gate 已通过：3 个 seed 的四步链在首步 checkpoint continuation 后均恢复 trace，并保持 update count=`1,2,3,4`；变量 3/4/5 步 episode 也均保持 trace 长度、连续计数、lineage 和 credit 完整。report/manifest 为 `reports/taiji_p7_grounded_multistep_*_20260825.json`。该 Gate 证明 runtime trace 连续性，不代表世界模型已经接入高级规划。
- P7 world-model planner projection/replan lesion Gate 已通过：adapter 的 `predict_world_candidates → plan_world_actions` 将 world learner 的结构化 reward/success 和近期 prediction error uncertainty 交给 `GoalPlanner`；真实 state error 超过规划阈值会触发 replan，即使 reward/success 为正；无 world learner 的 lesion 明确阻断该路径。该 Gate 为单步窄边界，不代表多步 imagined rollout 已由 world dynamics 自动生成。
- P7 world-dynamics imagined rollout narrow Gate 已通过：adapter 按预测 state/tick 滚动两步结构化 `WorldAction` 序列，逐步填充 reward/success/uncertainty，并写入 `prediction_provenance=world-dynamics` 后交给 `GoalPlanner.plan_rollouts`；既有 P5/P6 rollout/replan 回归仍通过。该 Gate 只证明两步生成和 provenance 边界，不代表跨 seed 或长 horizon 稳定性。
- P7 world-dynamics imagined rollout cross-seed Gate 已通过：3 个 seed 在 3/4/5 步 horizon 均生成并选中 data-derived rollout，逐步 tick chain 与 `world-dynamics` provenance 完整，native checkpoint 可恢复选中 rollout，world-model lesion fail closed；report/manifest 为 `reports/taiji_p7_world_model_rollout_*_20260825.json`。该 Gate 仍是数值世界 imagined execution，不代表真实环境执行已自动消费整条 rollout。
- P7 imagined-to-real execution Gate 已通过：3 个 seed 的 3/4/5 步 rollout 均经显式 motor routing 进入真实 environment，逐步写入 prediction/error trace，剩余计划被消费，learner update 与 trace 可经 native checkpoint 恢复；report/manifest 为 `reports/taiji_p7_imagined_execution_*_20260825.json`。错误 action-symbol 路由会 fail closed。该 Gate 不代表中途失败后的 rollout recovery 已完成。
- P7 rollout recovery Gate 已通过：3 个 seed 首步注入高 world-state error（reward/success 仍为正）会停止剩余 rollout、保留 prediction trace，并在 `CognitiveState.planning_recovery` 中记录 mode、trigger、error、threshold、source rollout 与被清空的剩余步数；native checkpoint 中断后无需重装 planner 即可继续恢复，终局成功会退出 recovery mode。report/manifest 为 `reports/taiji_p7_rollout_recovery_*_20260825.json`，`checkpoint_recovery_preserved=true`。
- P7 rollout recovery transfer Gate 已通过：3 个 seed、3/4/5 horizon、全部非终止失败位置共 27 个 case 均在中断后经 native checkpoint continuation 完成恢复，trace 长度与 learner updates 跟随实际 horizon；阈值校验已改为有限非负数，避免把 world-state MSE 当成概率。report/manifest 为 `reports/taiji_p7_rollout_recovery_transfer_*_20260826.json`。当前 transfer 仍由评估侧按该数值世界显式配置 `4.0`，不宣称 threshold calibration 已内生完成。
- P7 world-error calibration policy Gate 已通过：`GoalPlanner` 可接收真实 calibration error samples，按 quantile/std/margin 计算 world-error policy，recovery 期间再以触发误差自适应提高容忍度；samples、policy config 与 threshold 可经 planner/native checkpoint 恢复，移除 calibration source 的 ownership lesion 可检测。当前小型数据的 `0.25` config floor 仍主导 threshold，不宣称 raw MSE scale 已完全归一化。
- P7 normalized world-error contract Gate 已通过：`WorldSchema` 从训练语料生成并 checkpoint `state_scales`；`WorldPredictionRecord` 同时保存 raw MSE 与 schema-normalized `state_error`，runtime recovery/planner 使用 normalized error，scale 变换测试确认 raw error 不变而 normalized error 随 schema scale 改变。该 Gate 关闭了把 raw MSE 直接当跨 schema 阈值的边界，不等于跨任务 scale transfer 已通过。
- P7 schema-scale transfer contract Gate 已通过：同一 world state 差异整体放大 10 倍时 raw MSE 放大，而 schema-normalized error、calibrated planner threshold 与 checkpoint payload 保持；该行为已纳入 v1 contract tests，仍不替代多 seed runtime scale transfer。
- 本轮 native 回归为 `131 passed, 1 skipped`；命令显式排除两个受本机 Windows pytest 临时目录权限影响的旧 manifest 测试，
  环境状态不作为代码能力结论。
- 多信号 concept formation Gate 已通过（2026-08-26 由 `TAIJI_CONCEPT_FORMATION_GATE_2026_08.md` 归并）：概念形成同时受感知 latent、世界对象/关系结构与 outcome 三类证据约束，组合权重由 `TaijiConfig.concept_signal_weights` 配置（默认 `latent=0.45 / world=0.35 / outcome=0.20`）；语义巩固只接受有事件血缘且跨至少两个 episode 的经历，删除任一类证据都 fail-closed。该 Gate 是数据驱动的临时不变量形成，不等同于开放域语义或符号知识。
- `ConceptFormationOrgan` 已从 `TSKV8Adapter` 提取为独立器官，自持多信号匹配、概念 identity、支持集更新、可配置容量、塑性率、强度剪枝、显式 `lesion` 与独立 checkpoint；adapter 只保留接线与兼容 API，不再承载概念形成规则。容量 1/2/4 槽位分别保留 1/2/4 个概念。
- 概念到规划的窄消费路径已通过：`ConceptMatch` 把匹配 concept IDs 写入 `MemoryState`，并以匹配度 × 置信度 × outcome 质量映射为 `PlanningCandidate.concept_affinity`，由 planner 可配置 `concept_weight` 消费；lesion 后该 prior 归零并改变对照。schema 数量 1/2/4/8 的未见对象与关系查询保持 100% 规划迁移，容量 1/2 显示可测的概念干扰。
- 多步 sequence 与状态条件 suffix Gate 已通过：Concept 由 episode 时间顺序形成 `action_sequences`，rollout 按 `concept_sequence_weight` 消费，正确顺序击败高即时收益的反转序列；`ConceptSequenceTrace` 从真实 `WorldTransition` 保存每步 before/after latent、prediction error 与折扣后 step credit，部分执行后可按 after-state 重新检索剩余 suffix，错位动作与错误状态 fail-closed。
- 分支塑性、trace 容量与在线分支出生 Gate 已通过，报告为 `reports/taiji_concept_branch_20260826.json`、`reports/taiji_concept_trace_capacity_20260826.json`、`reports/taiji_concept_online_birth_20260826.json`：`suffix_sequence_affinity` 在 horizon=1/2/3 下区分正确分支与反转动作，真实 `confirm` 转移只对对应 trace 做 EMA 更新；`trace_capacity=1/2/4` 分别保留 1/2/2 条分支且按 trace strength 取舍；`TSKV8Adapter.grow_online_concept_branch` 可从不命中已有 trace 的真实转移链形成稳定 `trace_id` 新分支，重复链不产生副本，`settle_action` 的 episode buffer 可在 terminal 自动触发并经 checkpoint 续接。
- branch attribution Gate 已通过，报告为 `reports/taiji_concept_branch_attribution_20260826.json`：多个同时激活的 Concept 不再共享写入同一在线链，器官按 match confidence、已学习 before/after-state 证据和 prediction-error fit 选出唯一 owner；低置信度、近似平分的跨 concept 干扰与 owner trace lesion 均 fail-closed，权重与最小胜出间隔由 `TaijiConfig` 管理。
- 结构生长与拓扑 ledger Gate 已通过，报告为 `reports/taiji_structural_growth_20260826.json`、`reports/taiji_topology_proposal_20260826.json`、`reports/taiji_topology_runtime_ledger_20260826.json`：`StructuralGrowthRequest` 把结构变更记录为版本化 proposal，必须经 trial checkpoint roundtrip、trace lesion 与 replayability 验证才扣减 `DevelopmentState.structural_budget`；`StructuralTopologyProposal` 只描述现有合法固定 fan-in bank 的 rewire，不依赖 action/intent；runtime ledger 负责资源成本、预算耗尽 fail-closed 与按最新接受顺序的 rollback。
- neuron growth 与 cross-region wiring Gate 已通过，报告为 `reports/taiji_neuron_growth_20260826.json`、`reports/taiji_cross_region_20260826.json`：`AdaptiveNeuronRegion` 以稳定 `unit_id` 与显式活动/阈值/膜电位/trace 状态承载稀疏突触，新增单元只追加状态和突触行、不改动旧单元身份与权重；`AdaptiveNeuronNetwork` 以稳定 `connection_id` 建立稀疏跨区突触，上游生长会迁移连接输入维度并保留旧支持/权重，连接 lesion 后下游活动归零。
- 学习型跨区域协作 Gate 已通过：`CrossRegionCooperationLearner` 为显式连接维护可 checkpoint 的 prediction-error、holdout-transfer、resource-state EMA 与探索状态，学习路径在 holdout 证据上优于固定全连接/随机基线，并通过 connection/region lesion 与 checkpoint continuation；在线 credit loop 已接入真实 network tick，由 expected target activity 自动计算 prediction error 与 holdout transfer。
- 区域生命周期 Gate 已通过，报告为 `reports/taiji_region_growth_20260826.json`、`reports/taiji_region_pruning_20260826.json`、`reports/taiji_connection_pruning_20260826.json`、`reports/taiji_region_split_20260826.json`、`reports/taiji_region_merge_20260826.json`：持续区域瓶颈可生成带非语义 child identity 的 region proposal，post-growth validation 的相对 holdout gain 为 `0.8735` 且未通过验证的区域会阻断跨区连接；低使用 + 高资源压力 + learning stagnation 且移除后 holdout 不退化才允许 region/connection pruning；split 保持父区域与单位身份可追溯并迁移 route learner lineage，merge 需冗余证据且内部连接 fail-closed。以上全部覆盖预算、checkpoint continuation 与 reverse rollback。

### 工作项

- 建立实体、属性、关系、事件和 affordance 的分布式动态绑定。
- ~~引入容量受限、可学习的选择性路由与广播。~~ 已完成最小 runtime 机制及静态/world-outcome 窄 Gate；P4 记忆合同最小纵切片已进入。
- 把预测目标从 next byte 扩展到下一事件、状态变化和行动后果。
- 在对象持续性、关系交换、时间打乱和干预任务上预注册 A2/A3。

### 退出门槛

- 未见组合与干预预测显著优于频率/反应式基线。
- workspace 路由 lesion 和关系绑定 lesion 产生可解释损失。
- 世界状态可从 checkpoint 恢复，不依赖外部 Python 事实表。
- 异质群体的学习型路由在至少一类组合任务上显著优于最强单体、稠密平均和随机/固定路由，形成 A3 的 `1+1>2` 证据。

## 9. P4：多系统记忆

### 工作项

- 分离 working、episodic、semantic、procedural 四类记忆职责。
- 把当前 `EpisodicField` 接入 fast episodic 角色，而非复制成所有记忆。
- replay 从情景重现升级为结构抽取、关系重组和技能压缩。
- 建立新组合、相似事件干扰、来源追踪和长期遗忘评测。

### 退出门槛

- one-shot 情景保持与跨 episode 语义迁移同时成立。
- 删除 episode 身份后仍保留可迁移规律，删除语义巩固后迁移显著下降。
- 10²→10⁴ episode 扩展给出容量、干扰和资源曲线。
- fatigue/curiosity/stress 等 homeostatic state 能在无外部硬编码日程下正确触发探索、专注、睡眠或休息；sleep/play/dream lesion 分别产生预注册的能力损失。

## 10. P5：执行认知

### 工作项

- 建立目标层级、价值、不确定性、冲突和进度状态。
- 世界模型支持带 provenance 的 imagined rollout。
- 规划器比较真实可执行候选，不直接产生漂亮文本作为替代。
- 自我监控预测成功率并触发搜集信息、重规划或请求外部决策。

### 退出门槛

- 在未见目标、延迟奖励和环境干预中优于 reactive policy。
- world-model、rollout、goal-memory 和 value lesion 均有独立效应。
- 内部置信度对外部成功率校准，并能改善资源分配。

## 11. P6：语言与工具生成

### 工作项

- 区分内容计划、表达计划和最终 byte 编码。
- 从 `ActionIntent` 生成语言或结构化工具调用；byte motor 只做末端 codec/回退。
- 训练和评测同时覆盖语义保持、可读性、上下文、工具参数正确性和执行结果。
- 建立 direct-next-byte、无内容规划、无 workspace 和随机表达对照。

### 退出门槛

- 语言输出在人类盲测、语义一致性和多轮目标保持上达到预注册门槛。
- 工具调用不仅格式正确，而且真实执行成功并把 outcome 回写认知状态。
- 生成不依赖 Legacy 路由或外部 teacher 的运行时决策。

## 12. P7：持续进化、多模态与规模化

- 新任务学习后测量旧能力保持、适应速度和结构增长。
- 把异质专门化、结构重连/剪枝、assembly 分化和能力缺口驱动的资源申请变成 Taiji 内生发展过程；结构变化必须可回滚。
- 将好奇、疲劳、压力、安全和资源需求纳入持久 homeostatic state，真实控制探索、睡眠、玩耍和学习预算。
- 增加图像/音频/身体器官，共享世界状态而非晚期答案拼接。
- `CapacityPolicy` 转为资源治理器，支持可塑拓扑和按层预算。
- 对 v1 实际热点做 CPU/CUDA profiler，再决定 fused/sparse kernel。
- 报告吞吐、显存、能耗、数值一致性和 checkpoint 跨设备恢复。

## 13. P8：产品与发布

- Seed UI/API 只展示已通过 Gate 的 Taiji 能力。
- S2 安全、覆盖率和门禁继续完成；S3 打包、版本、更新和回滚在 v1 API 稳定后收口。
- 默认发行不安装 Legacy 重依赖；Legacy 只保留离线对照和显式兼容构建。
- 发布物包含模型卡、数据卡、能力 Gate、失败边界和恢复方式。

## 14. 持续门禁

- Taiji/Seed/Legacy 所有权 AST 测试；
- v1 state/checkpoint schema 和确定性恢复；
- TSK-v8 K 系列回归；
- 当前阶段 A Gate 的 holdout、lesion 和跨 seed 结果；
- 数据 manifest、实验注册、代码 commit 和训练 lineage；
- planned/actual learned state 与资源预算；
- 后端、前端、桌面、Legacy-off 启动和安全门禁。

辅助训练结果必须标记 `native-assisted`；只有不依赖辅助 teacher 决策且能继续终身学习的路径才能标记 `native-local`。A0–A9 的目的追溯和 Gate 定义以 Taiji v1 架构文档为准。

### 14.1 门禁自身的可信度纪律（2026-08-26 事故后新增）

一次 CI 事故暴露出「门禁写下来」不等于「门禁跑过」：提交 `470f2af` 同时引入了 `black==24.12.0` 这个 **PyPI 上不存在的版本**（`24.10.0` 之后直接是 `25.1.0`）和多道新 blocking 门禁及「存量已清零」注释。依赖安装步骤因此在 30 秒内失败，其后 **全部门禁被跳过**，CI 连续 8 天红灯，期间累积的 84 个提交没有被任何门禁检验过。

因此以下规则生效：

- 任何 pin 的版本号必须先确认上游真实存在。PyPI pin 查 PyPI，pre-commit `rev:` 查上游 **git tag**（两者是不同的命名空间，`24.12.0` 在两边都不存在）。
- `.github/workflows/ci.yml` 的 pip pin 与 `.pre-commit-config.yaml` 的 `rev:` 必须同步改动，保持本地钩子与 CI 同版本。当前统一为 `ruff==0.16.4` / `black==26.5.1`。
- 门禁注释里的数字必须是**实测值**，不是期望值。声明「存量已清零」之前必须有一次真正跑绿的 run 作为证据。
- 依赖安装步骤失败会让后续门禁静默跳过（显示 `-` 而非 ✗）。判断 CI 是否真的验证过代码，要看步骤是否执行，而不只看 job 的红绿。

### 14.2 mypy 类型债（核心层已转棘轮 blocking，2026-08-26 收口）

2026-08-26 修好 pin 后，上述 blocking 门禁首次真正执行，实测与注释不符：

| 门禁 | 原注释声称 | 本机首测（py3.12） | CI 实测（run 32986602722，3.10/3.12） | 现状 |
|---|---|---|---|---|
| `ruff check .` | 存量已清零 | 0 | 0 | blocking |
| `ruff check . --select B,SIM` | 存量 32 | 4 | 0 | blocking |
| `black --check .` | — | 68 个文件待重排 | 0 | blocking |
| `mypy --follow-imports=silent seed taiji` | 0 错误 | 47 → **63** | **63**（两腿一致） | **棘轮 blocking，基线已降至 0；待新提交 CI 双矩阵复核** |
| 全仓 mypy | 基线 212 | 259 → **275** | **281**（两腿一致） | advisory 观测 |

**47→63 / 259→281 的漂移根因不是代码退化，而是 `mypy` 与 `pip-audit` 在 CI 里从未钉版本。** `ruff`/`black` 早已按 `.pre-commit-config.yaml` 钉死（0.16.4 / 26.5.1），唯独这两个漏了。检查器静默升版会带来新检查项，于是**没人改代码，门禁数字自己会变**。由此确立通用规则：**凡把工具输出数字当阈值的门禁，工具本身必须钉版本**，否则棘轮基线随时失效。现已钉 `mypy==2.3.1`、`pip-audit==2.10.1`。

原先「不能设阈值」的顾虑（mypy 报错数随 Python 版本变化，本机单版本数字不足为凭）已被双矩阵实测**否证**：3.10 与 3.12 的核心数（63）与全仓数（281）完全相同，双矩阵取较大值即等于单值，可直接钉。

核心 63 错经与最后一次绿色提交 `42d268e` 对比确认**不是新增退化**（那时 mypy 仍是 `continue-on-error: true`），属存量类型债。分布：`taiji/adapter.py` 12、`world_learning.py` 9、`workspace.py` 8、`local_learning.py` 7、`contracts.py` 5、`procedural_memory.py` 4、`seed/language_provider.py` 4，其余 14 文件各 1–3。主因是 checkpoint / `state_dict` 反序列化后为 `object | Any` 缺少类型收窄——这类缺陷与 14.3 记录的 checkpoint 静默失败同源，须实修而非长期忽略。

**为什么选棘轮而不是「等实修到 0 再转 blocking」**：advisory 对退化零约束，63 涨到 100 也照样绿灯，门禁形同不存在；而等清完 63 项再上门禁，这期间新增退化无人拦。棘轮（`errors > MYPY_CORE_BASELINE` 即失败）把「不许变差」立刻变成硬约束，又不阻塞开发。步骤同时对解析失败显式 `exit 1`——门禁绝不允许在读不到数字时静默放行，这是 14.1 的直接应用。

收紧路径：每次实修使核心数下降后，把 `ci.yml` 中的 `MYPY_CORE_BASELINE` 同步下调（步骤会打 `::notice::` 提示当前实际值），单向收紧至 0；全仓层待核心归零后按同一棘轮形式转正。2026-08-27 已完成核心层归零：本机 `mypy==2.3.1` 对 `seed taiji` 的 44 个源文件报告 `Success: no issues found`，因此门禁基线已从 63 下调为 0；这只证明当前 checkout，不能替代 CI 的 3.10/3.12 双矩阵实跑。

### 14.3 checkpoint 往返对称不变量（2026-08-26 回归后新增）

停写后复跑全量测试暴露一处真实回归：`TSKV8Adapter.checkpoint()` 从不写出 `cognitive_state`、`restore()` 也从不恢复它，而 `reset_dynamics()` **会**覆写 `_cognitive_state`。于是 `TaijiModel.score_bytes()` 的 `checkpoint → reset_dynamics(episode_id="evaluation") → finally restore` 三段式只回滚了内核状态，认知状态被永久留在 `evaluation` episode 上并带着漂移的 tick；`native_checkpoint()` 随后把这对不一致的状态写盘，`restore_native()` 的一致性校验抛出 `native cognitive state is out of sync with kernel state`。触发路径是 `scripts/training/train_seed_corpus.py` 的 `_flush()` 调用 `score_bytes()`，而 `_flush(final=True)` 紧接 `_persist()`——**每一次最终 checkpoint 都写在被污染之后**。

因此以下规则生效：

- 任何被 `reset_dynamics()`（或其他 in-place 状态重置）改写的字段，必须同时出现在 `checkpoint()` 的 payload 和 `restore()` 的恢复路径里。三者缺一即为缺陷，不是风格问题。
- `checkpoint()`/`restore()` 是成对契约，新增可变运行时状态时必须同步改这两处，并补一条往返断言，而不是等 `restore_native()` 的不变量在训练末期才爆。
- 新增 payload 键必须带向后兼容分支：旧信封缺键时按内核状态重建，而不是抛错或静默留下不一致值。
- 该缺陷类会让长训在最后一步失败并丢弃 checkpoint，直接违反「训练之前检查是否能够正确保存 checkpoints」这条前置要求；因此 checkpoint 往返测试属于阻塞级，不接受 advisory。

### 14.4 `plans/active` 编制与单一下一步纪律（2026-08-26 收敛后新增）

同一轮排查发现 `plans/active/` 出现第 6 份文档 `TAIJI_CONCEPT_FORMATION_GATE_2026_08.md`，全仓无任何引用（README、测试、脚本均未提及），但自带一节「下一步唯一入口」，其后半段与本文件第 16 节的 Gate 链几乎逐字重复——即**存在第二个「唯一下一步」权威源**，这才是 `tests/seed/test_project_identity.py` 失败的实质，而不是文件数量超限。

处置方式是归并而非改名或放宽白名单：其独有的运行时事实已并入上文 P7 事实清单，重复的下一步整节删除，原文件移入 `plans/archive/implementation/` 并在归档索引登记，`plans/README.md` 与身份测试均不改动。

因此以下规则生效：

- 「当前唯一下一步」只允许出现在本文件第 16 节。任何其他文档若要记录进展，只能写已完成事实，不得设立自己的下一步入口。
- 新增 `plans/active/` 文档前必须先确认它不是既有文档某节的复制品；能并入现有章节的一律并入。白名单是编制约束，冲突时收敛内容，不是放宽白名单。
- 归档文档必须显式声明其历史「下一步」不得恢复执行，避免残留方向在后续调用中与总路线竞争。

### 14.5 双学习栈收口与「假绿检查」纪律（2026-08-26 收口后新增）

`verify_taiji_native_v7.py` 的原生性 AST 契约把 `backward` 列为禁用属性，但 `taiji/` 内 8 个模块共 13 处仍在用 `SGD/Adam + loss.backward()`，契约长期失效。同一次排查还发现 `no_autograd_parameters` 是**假绿**：`PerceptionModule.parameter_tensors()` 返回 `parameter.detach()` 视图，而 detach 视图的 `requires_grad` 恒为 `False`，于是这条检查无论参数真实状态如何都通过。

处置方式取上限最高的一条：不是放宽契约，而是把 autograd 学习平面整体替换为与内核一致的原生局部信用分配。新增 `taiji/local_learning.py` 作为唯一来源，每条规则都对 autograd 做逐位等价验证（最差偏差 5.96e-08）；`LocalAdam` 在 detached 张量上复现含偏差校正的 Adam 更新式，因此迁移没有连带改变优化器、上层已调好的学习率与收敛阈值继续有效。`parameter_tensors()` 改为返回活体参数，该检查转为真检查并通过。收口后 15/15 检查为 true，8 道阻塞 verify 全 pass，`tests/` 437 passed / 5 skipped。

因此以下规则生效：

- 契约与实现冲突时，先问「哪一侧代表想要的架构」。契约代表目标架构时，改实现，不改契约。
- 布尔检查必须能失败。`detach()`、`copy()`、`float()` 之类的转换会顺手清掉 `requires_grad`、梯度和设备信息，把断言变成恒真式；断言 `requires_grad`/`grad`/`device` 之类元数据时必须作用于活体对象。
- 新增一条检查后必须构造一次**故意的失败**来证明它会红，否则它只是装饰。
- 手写梯度替换 autograd 时，必须逐处对 autograd 做数值等价验证再删除对照代码；GRU 的 reset gate 会给 `_hh` 侧 n 段多乘一个 `reset` 因子，这类不对称写错时**不会报错、只会静默学不动**。
- **假绿的第二种形态：被测函数自带早退门（platform / feature flag / 环境变量），测试没钉住该门，断言就在部分平台上根本没进被测分支。** 实例：`api/routes_terminal.py::_normalize_terminal_input` 首行是 `if sys.platform != "win32": return text`，而 `tests/test_terminal_input_normalization.py` 里断言「已转成 CRLF」的 3 条用例未钉 `sys.platform`，于是它们在 Windows 本机绿、在 Linux CI 红；同文件的 `test_existing_crlf_not_double_converted` 更隐蔽——它在 Linux 上也绿，但绿的原因是压根没进转换分支，验证不了「不重复转换」。修法是用 fixture 显式 `monkeypatch.setattr(mod.sys, "platform", "win32")` 钉住转换分支，另设一条独立用例验非 win32 的原样返回。**凡断言「发生了某个转换」的用例，都要能说清它在哪个分支里被执行。**

### 14.6 本机工具链环境事实（沙箱）

- `black` 的默认缓存目录 `%LOCALAPPDATA%\black\black\Cache\<ver>\` 在本沙箱下不可写。black 不会报错退出，而是反复重试建临时文件，表现为**永不返回并满载 CPU**（实测单文件 30 秒墙钟烧掉 651 秒 CPU），且其多进程池被中断后会**泄漏 worker 进程**（一次排查中发现 ~90 个残留 `python.exe` + 3 个 `black.exe`）。修复方式是把缓存指进工作区：`$env:BLACK_CACHE_DIR="<repo>\.black_cache"`，之后 `black --check .` 4 秒完成。`.black_cache/` 已入 `.gitignore`。
- 诊断这类「无输出挂死」不能靠 stdout——输出重定向同样为空。有效手段是在新终端观察副作用（`git diff --stat` 看文件是否已被改写、`Get-Process` 看进程与 CPU、`Get-CimInstance Win32_Process` 取准确命令行），必要时用 `faulthandler.dump_traceback_later(N, exit=True)` 强制打栈。
- `reports/ci_verify/` 是 `ci.yml` 中 8 道 verify 门禁的 `--output` 产物目录，属运行产物，不入库。

### 14.7 「本地绿 / CI 红」的排查纪律（2026-08-26 新增）

本机是 Windows，CI 的 `test (3.10)` / `test (3.12)` 是 Linux，`test-windows` 才是 Windows。因此本地全绿从不等于 CI 会绿，**本地跑过不构成推送前的充分证据**。出现分歧时按此顺序排查，不许先归因为「环境差异」：

1. **先确认两边跑的是不是同一份代码。** 本次分歧的真实原因就在这一步：CI 红的那次跑的是 `ffe1da2`（迁移前，`taiji/` 内 14 处 `.backward()`，AST 契约理应 false），而本地 pass 时工作区已是迁移后的未提交代码。用 `git grep -c "<pattern>" <commit> -- <path>` 直接查历史提交的内容，而不是看当前工作区。**CI 报红时默认它是对的。**
2. **再看失败项是否平台相关。** 断言里出现 `\r\n`、路径分隔符、大小写敏感、文件锁、`sys.platform` 分支时，优先怀疑测试自身缺平台钉定（见 14.5 最后一条）。
3. **在本机复现 CI 的平台条件，而不是等下一次 CI。** 对纯逻辑的平台门，直接改 `mod.sys.platform` 后跑该测试文件即可复现；复现后必须验证「旧断言在此条件下确实红、新写法确实绿」，两侧都验才算修好。
4. **读日志要认准步骤归属。** `gh run view --job=<id> --log-failed` 的输出里混有 advisory 步骤（mypy 的 47/259 条错误）的大量噪声，它们不是失败原因；用 `Select-String "FAILED|VERIFY_RESULT|Process completed"` 定位，并核对行首的步骤名。

### 14.8 `needs:` 会把整条下游门禁隐藏为 skipped（2026-08-26 收口后新增）

这是「假绿」的第三种形态，且比前两种更隐蔽：前两种是**断言本身**不成立，这一种是**门禁根本没跑**。

- **事实**：`ci.yml` 中 `build-frontend`（:162）与 `docker-build`（:215）都声明 `needs: test`。`test` 连续红 7 次期间，这两个 job 一直是 `skipped` 而非 `failure`，从未真正执行。`test` 转绿的那一刻它们首次运行、立刻双红——这不是新引入的回归，而是**被上游红长期遮蔽的既存缺陷**。
- **纪律**：判断「CI 是否绿」不能只看有没有红色条目，必须核对**期望执行的 job 集合是否都真的执行了**。`gh run view <id>` 里 job 数量少于预期时，要顺着 `needs:` 链回溯是谁被 skip 了。修完一个长期红的上游门禁后，**默认下游还有未曾运行过的门禁在等着红**，不要在上游转绿时就宣布收口完成。
- **两处被暴露的既存缺陷及其修法**：
  - `docker-build` 在 `Dockerfile:57 COPY data/ ./data/` 失败（`"/data": not found`）。根因是 `data/` 命中 `.gitignore:132:data/`、`git ls-files data` 为 0，只存在于本机（约 1.9GB），CI 全新 checkout 下必然不存在。修法是收敛到项目**既有**的「大体积/本地资源走挂载不进镜像」约定（`.dockerignore` 里该约定已列 `checkpoints`/`logs`/`reports` 等）：Dockerfile 改为 `RUN mkdir -p ./data`，`docker-compose.yml` 增 `./data:/app/data`，`.dockerignore` 增 `data`。空目录是安全的——`routes_model_switch.load_runtime_pref()` 异常即返回 `{}`，`training/resume._resolve_datasets()` 返回 `missing` 列表而不抛错，且 CI 后续的 metadata/healthcheck 步骤均不触及 `data/`。
  - `build-frontend` 在 `npm audit --production --audit-level=high` 失败：`nanoid@3.3.12`（GHSA-28wg-ghj8-5hjv、GHSA-2v37-7h3g-55p8）与 `postcss@8.5.14`（GHSA-fxqj-rqcc-2cmp、GHSA-r28c-9q8g-f849）两个 high。两者都是 vite 的**传递**依赖，项目源码零引用，因此不写 `devDependencies`（那是语义造假、且只保证顶层 hoist），改用 `package.json` 的 `overrides` 对全树强制 `nanoid ^3.3.18` / `postcss ^8.5.26`，未来 vite 自升后可整块删除、不留孤儿声明。余下 2 个 moderate 是 dompurify/monaco-editor，其 `npm audit fix --force` 会把 monaco 升到 0.56.0 的 breaking change，在 `--audit-level=high` 下不阻塞，故不动。
- **对不可本机验证项的诚实处理**：本机无 Docker（`docker` 命令不存在），无法本地构建验证。此时不假称已验证，改做等价的**静态审计**：逐个核对 Dockerfile 全部 10 个 `COPY` 源在版本控制中的跟踪文件数（`git ls-files -- <path>`），确认 `data`/`checkpoints`/`logs` 均为 0 且已无任何 `COPY` 引用它们，同类根因一次排净。
- **静态审计只覆盖它所提的那个问题，不等于该门禁会绿（2026-08-26 二次收口补记）**：上述审计问的是「每个 `COPY` **源**是否存在」，因此它确实预测对了 build 层转绿；但它问不到「每个被 import 的**包**是否都在 `COPY` 清单里」，于是漏过了下一层缺陷——`docker-build` 的 build 步骤通过后，`Startup smoke and healthcheck` 以 `api/app.py:26 ModuleNotFoundError: No module named 'seed_platform'` 失败。根因是 Dockerfile 手工枚举的 `COPY` 清单与 `pyproject.toml:64` 的 `[tool.setuptools.packages.find].include = ["seed*", "taiji*", "seed_platform*", "neuroplex*", "api*"]` 是**两份互不校验的重复清单**，而 `packages.find` 对缺失目录是**静默跳过**的：漏拷 `seed_platform/` 后 `pip install -e ".[legacy]"` 依然退出 0，缺失只在容器启动时才炸。`seed_platform` 是 `api`/`neuroplex` 的运行时核心（全仓 60 处 import、10 个跟踪文件）。
- **修法要消除重复清单本身，而非补一个包**：除补 `COPY seed_platform/ ./seed_platform/` 外，在 `pip install` 之后加一道**构建期导入断言** `RUN python -c "import api.app"`，把「镜像内缺包」从运行时 smoke 前移到 build 层，此后任何漏拷贝立即在构建时失败。断言位置须在前端产物与 `data/` 之前才安全，这一点经核实：`api/app.py` 的 `StaticFiles`/`dist` 使用全在第 264 行之后的 app 工厂函数体内，模块级只做路径常量计算，`seed_platform/paths.py` 的 `makedirs` 均带 `exist_ok=True`，故 `import api.app` 不依赖 dist 或 `data/`（本机同句实测退出 0，证明断言不会误红）。清单一致性亦已复核：pyproject include 的 5 个包与 Dockerfile `COPY` 完全对齐，`MISSING: none`；`desktop/` 不在 include 内，仅由 `[project.scripts]` 的 PyQt 桌面入口使用，容器不需要。
- **由此得到的通用纪律**：任何"手工枚举 + 上游有权威清单"的结构都是复发源，收口时要么让枚举可校验、要么加一道断言让偏差立刻失败；而**多步 job 只有全部步骤都绿才叫绿**——`docker-build` 的 `Build image` 打勾极易被误读成该 job 已通过。
- **闭环已实证（2026-08-26）**：`gh run view 32984530278 --json status,conclusion` 返回 `status=completed` / `conclusion=success`，7 个 job（`test 3.10`/`test 3.12`/`test-windows`/`Startup smoke (legacy)`/`Startup smoke (no-legacy)`/`build-frontend`/`docker-build`）全绿，其中 `docker-build` 的 `Startup smoke and healthcheck` 通过，确认 `seed_platform` 漏拷已修且构建期导入断言不误红。查询时注意：run 未结束时 `status=queued` 且 `conclusion=""`，此刻 `--log-failed` 会拒绝执行，须等 `completed` 再判定，不要把中途快照当结论。

### 14.9 npm 侧的沙箱事实与安全升级手法

- `npm audit fix` 会重解析整棵依赖树，在本沙箱下**无输出挂死**（实测 15 分钟、node 进程仅耗 2.9 秒 CPU、`package-lock.json` mtime 未变，属网络/解析阻塞而非计算）。判定手法同 14.6：看进程 CPU 与文件 mtime 的副作用，不看 stdout。
- registry 本身可用：`npm view <pkg> version --json` 秒回。可用它确认目标版本后走窄范围路径——`npm install <pkg>@<exact> --no-audit --no-fund` 或改 `overrides` 后 `npm install --package-lock-only`（实测 14 秒）。
- 改 `overrides` 后必须验证 lock 与 package.json 是否同步，因为 CI 跑的是 `npm ci`（不同步会直接失败）。npm 11 在解析结果已满足 override 时不会往 lock 根条目写 `overrides` 字段，所以**不能以"lock 里搜不到 overrides"判定失败**，权威判据是 `npm ci --dry-run` 退出 0。
- Windows PowerShell 5.1 的 `ConvertFrom-Json` 无法处理空字符串键，而 lockfileVersion 3 的根条目正是 `packages[""]`；检查 lock 结构要用 `node -e`。
- 前端门禁必须四道齐验，只跑 audit 不够：`npm audit --production --audit-level=high`、`npx eslint src --ext .js,.vue`（0 errors，warnings 容忍）、`npx vitest run`、`npm run build` + `dist/index.html` 存在性。

### 14.10 仓库可发现性：元数据是必要条件，且受令牌能力边界限制（2026-08-26 新增）

- 实测 `gh repo view --json` 确认：仓库自 2026-07-15 公开，但 `description=""`、`repositoryTopics=null`、`homepageUrl=""`、`usesCustomOpenGraphImage=false`，1 star / 0 fork / 0 watcher。GitHub 用于分发流量的字段全空，这不是"设计不佳"而是该层未填写。
- `repositoryTopics=null` 的后果是**缺席全部 topic 浏览页**（不是排名靠后，是不在列表里）；`description=""` 则把搜索匹配面全部推给 README 全文，而仓库名 `Seed` 是高冲突通用词，几乎不可能靠名字被检索到。
- topic 选择必须按**真实仓库规模**取中间区间，不能取最热。实测 `gh api search/repositories?q=topic:<t>` 计数：`local-learning` 25、`sparse-neural-networks` 43、`predictive-coding` 116、`episodic-memory` 140、`hebbian-learning` 148、`neuromorphic-computing` 328、`world-models` 551、`computational-neuroscience` 728、`cognitive-architecture` 784、`online-learning` 888、`pytorch` 59222、`artificial-intelligence` 47251、`deep-learning` 104237、`machine-learning` 235783。在 116 个仓库的页面里会被看到，在 235783 个里等于不存在，故 `machine-learning` 不贴；`spiking-neural-networks`（597）虽在好区间也不贴，因为内核不是脉冲网络，贴上是误导。禁贴 `agi`——README 自身声明 `not an AGI claim`，贴上即自相矛盾。
- **令牌能力边界（重要）**：本机 `GH_TOKEN` 是 App/细粒度令牌。`gh api repos/... --jq .permissions` 返回 `admin:true`，但 `gh repo edit --description/--add-topic` 与 `PUT /repos/{owner}/{repo}/topics` 均返回 `HTTP 403 Resource not accessible by integration`。措辞中的 "by integration" 是判据：App 令牌能力由 App 声明的 permission set 决定，与账号是否 admin 无关；该令牌有 `contents:write`（故 `git push` 一路成功）但无 `administration:write`，而 description/topics/homepage 属 Administration 档。**结论：这三个字段无法由 agent 用当前令牌写入（换会话亦无效，见下），不要反复换写法撞同一面墙。** social preview 图 GitHub 从未提供 REST 接口，本来也只能网页上传。
- **绕行路线穷尽结果（三条全否）**：（1）GitHub MCP 暴露的 40 个工具只覆盖 issue / PR / 文件 / 分支 / release / 搜索，无任何 repo settings 写入能力，排除；（2）`agent-browser` 本机未安装（`CommandNotFoundException`），浏览器自动化需先 `npm i -g agent-browser`，为改两个字段引入全局依赖不值当；（3）`RequestAuthorization(administration:write)` 回执为 success，但**授权成功 ≠ 能力到账**，见下条。

- **「换新会话即可写入」这一推断已被实测否定（2026-08-26 新会话验证）**：在全新会话中按原 §16 逐条重跑，三条 API 路线全部失败：`gh repo edit --description` → `HTTP 403 Resource not accessible by integration`；`PUT /repos/{o}/{r}/topics` → 403，响应头 `X-Accepted-Github-Permissions: administration=write`；GraphQL `updateTopics` → `{"type":"FORBIDDEN","path":["updateTopics"]}`。同会话内二次 `RequestAuthorization` 仍返回 success，但进程内 `GH_TOKEN` 前缀与长度不变（`ghu_`/40），写入依旧 403。
- **判据与根因**：`ghu_` 前缀说明这是 GitHub App 的 user-to-server 令牌，其能力上限由 **App installation 声明的 permission set** 决定，而非由本地 `RequestAuthorization` 的回执决定。`gh api repos/... --jq .permissions` 返回 `admin:true`（那是**账号对仓库的角色**）而 `X-Oauth-Scopes` 为空、`X-Accepted-Github-Permissions: administration=write`（那是**接口要求的 App 权限档**）——两者是不同维度，前者为 true 完全不蕴含后者放行。REST 与 GraphQL 走同一权限档，故 GraphQL 不是绕过 403 的后门。
- **通用纪律**：授权类回执（"authorization granted"、"start a new conversation"）属于**未验证的能力承诺**，必须以一次真实写入调用作为唯一验收判据；不能把它写成计划里的"已解决"。同理，凡出现 `by integration` 措辞，不要再在同一令牌上换 REST/GraphQL/参数写法反复尝试——那是同一面墙的不同侧面，正确动作是换执行主体（本人网页操作或换用具备 `administration:write` 的 PAT）。
- **闭环结果（2026-08-27，换执行主体后一次成功）**：用户在 GitHub 网页端完成三项写入，agent 侧用只需 `metadata=read` 的 `gh repo view --json description,repositoryTopics,homepageUrl,usesCustomOpenGraphImage,openGraphImageUrl` 复核并通过。这印证了上条纪律：受阻的是**执行主体的权限档**，不是方案本身，换主体后零重试即成功。
- **复核纪律（易踩）**：不要肉眼比对 `gh repo view` 输出。description 须做**逐字符相等**判定（250+ 字符里一个折行或全角标点差异肉眼不可见），topics 须做**集合相等**判定（同时报 missing 与 unexpected，因为 GitHub 返回时按字母重排，顺序不同不等于内容不同，而漏一个/多一个才是真错）。social preview 的唯一可信判据是 `usesCustomOpenGraphImage: true` 加 `openGraphImageUrl` 非空——打开仓库页面看图会被浏览器缓存与 CDN 边缘缓存欺骗，看到旧图或看到新图都不足以定论。实测踩坑：PowerShell `>` 重定向会给 JSON 写入 UTF-8 BOM，Python `json.load` 直接抛 `Unexpected UTF-8 BOM`，须用 `encoding='utf-8-sig'` 读取。
- **social preview 图的设计判据（改数字时复用）**：卡片在时间线里通常只被扫视约 1 秒，能留下的只有一个数字加一句机制主张，故只印 `0% → 94.12%`（全仓最强实测数字，来自双区 `[64, 48]` benchmark、seed 7 的 byte-cycle accuracy，见 README L203-L210）与 `no backprop / no attention`（区分于任何 Transformer 仓库的最短差异化陈述），并附 `two-region [64, 48] benchmark · seed 7` 使数字可追溯；不印 logo 或抽象插画。用 PIL 程序化渲染而非文生图，因为文生图会把数字糊掉，而这张图的全部价值就在数字的可读性上。配色取自现有品牌资产 `frontend/public/logo-taiji-ink.jpg` 的宣纸白 `#FAFBF6` 与焦墨黑 `#060604`，2× 超采样 + LANCZOS 缩放保证字缘锐利。
- **首版两个缺陷及修正（记录以免重犯）**：（1）surprise 衰减曲线横穿底部文字，视觉上把 `surprise 5.4041 → 0.1069` 划成删除线——把自家指标划掉，语义完全反了；结论是**造成语义反转的装饰应删除而非挪位**，已移除该曲线。（2）太极水印用了 `INK_FAINT` 且坐标写死，压住 `94.12%`；改为浅色背景层，位置由实测文字宽度算出，空间不足时**自动不画**——宁可留白也不撞字。另外脚本自检本身也抓到过一次真实问题（`bbox=(100,66,1180,614)`，页脚距下边缘仅 26px，有被各平台按不同比例裁切的风险），压缩纵向节奏后收敛到 `(100,66) → (1180,596)`。
- README 首屏顺序是唯一不依赖令牌权限的杠杆，且转化价值高于 topics（topics 带人进来，首屏决定是否留下）。原首屏被"命名分工 + 免责声明"占据，而最有传播力的两个资产（Transformer 责任对照表、`0%→94.12%` / `98.02%` 数字）分别埋在 L52 与 L184。已重排为：一句话机制主张 → badge → 对照表 → 实测数字 → 明示 status。**诚实声明一条未删**，只是移出首屏主位，并新增 `## Project scope` 承接原命名段。
- 改 README 首屏必须回原文核对每个被前移的数字有出处（实测首屏 `94.12/5.4041/0.1069/98.02/83,841` 全部对应 L203-L210 原表），并确认锚点标题真实存在（`#reproducible-tsk-v8-kernel-results` → L197）以及旧免责声明残留计数为 0——Markdown 锚点失效与声明重复都不会报错，只会静默劣化。

### 14.11 平台停机产生的「假红」：run 级结论不可信，须看 job 级是否真的跑过（2026-08-26 新增）

这是「假绿」的镜像形态，同样会误导收口判断：CI 显示红，但代码毫无问题。

- **事实**：`2026-08-26T15:11:58Z` 起 GitHub Actions 发生 `impact: critical` 停机（incident `y1t7p9fzrlj2`，15:48 的官方更新为「throttled inbound traffic … upstream Vitess issues」）。期间三次推送的 run 呈现三种异常：`560525c` → `startup_failure`；`c8acff5`、`4e6a827` → run 级 `failure`；以及推送后一段时间内根本不产生 run。
- **判别手法（关键）**：不要看 `gh run view <id>` 的 run 级 `conclusion`，要看 `gh api repos/{o}/{r}/actions/runs/<id>/jobs`。实测 `32986449122` 的 5 个测试 job 全部 `status=queued` / `conclusion=null`，`32985649140` 的 5 个 job 全部 `cancelled`，两者的 `build-frontend`/`docker-build` 均 `skipped`。**job 从未被分配 runner 却出现 run 级 failure，这种形状不可能由测试失败产生**——测试失败必然伴随 job 已 `completed` 且有真实耗时。据此可判定为平台产物而非本仓缺陷。
- **同时排除本仓嫌疑的四项旁证**：`ci.yml` 在该时段无改动；`git diff dae6464..HEAD` 的非文档代码差异为空；`gh workflow list` 显示工作流仍 `active`（未被禁用）；`HEAD == origin/main`。另有一条**反向证据陷阱**：此时 `gh workflow run` 返回 403，容易被误读为「工作流被停用」，实为 `ghu_` 令牌缺 `actions:write`（见 14.10），与停机无关。
- **停机期间的正确动作**：把 CI 的门禁在本机跑一遍，作为唯一还能推进「代码是否真绿」的手段；平台恢复后立即以新 run 取代本机结论，并停掉本机长任务（本机只能覆盖 lint/版本/类型，覆盖不了 docker/前端/多矩阵）。
- **恢复后的权威结论**：`3e6e5b0` 的 run `32986602722` 为 `status=completed` / `conclusion=success`，5 个 job（`test 3.10`、`test 3.12`、`test-windows`、`Startup smoke (legacy)`、`Startup smoke (no-legacy)`）全绿。停机期那三次红全部作废，不需要任何代码修复。
- **通用纪律**：判定 CI 结论前先确认**job 真的执行过**（有 `started_at`、有耗时、`conclusion` 非 null）。这与 14.1「依赖安装失败会让后续门禁静默跳过」是同一条原则的两面——红与绿都不可只看颜色，要看执行事实。
- **停机尚未结束，`98e36db` 的棘轮门禁线上验证仍未完成（待办）**：`98e36db`（钉版本 + mypy 棘轮）推送成功（`3e6e5b0..98e36db`）后 5 分钟内 `gh run list` 未出现对应 run，`githubstatus` 的 Actions 组件仍为 `major_outage`。故**新棘轮门禁在 CI 中的首次实跑尚无证据**，不得记为已验证。停机期间改做本机等价校验：四个钉版本经 PyPI 逐个确认真实存在（`mypy==2.3.1` 2026-08-15、`pip-audit==2.10.1` 2026-06-10、`ruff==0.16.4` 2026-08-20、`black==26.5.1` 2026-05-18），排除了「幻觉版本号导致安装步骤失败、其后门禁全部静默跳过」这一已发生过的复发路径（见 14.1 与 ci.yml 的 `black==24.12.0` 注释）；棘轮的解析与比较逻辑亦已在本机用真实 mypy 输出复现（`parsed=63`、`baseline=63`、`PASS(equal)`）。平台恢复后须补验：核心步骤在 3.10/3.12 两腿都打印 `mypy core errors: 63 (baseline 63)` 且不失败。

### 14.12 项目改名的环境变量残留：`E:\taiji-neuron` 反复自动重建（2026-08-27 收口）

现象：删掉 `E:\taiji-neuron` 后它总会再出现。这不是本仓代码所为，而是改名（`taiji-neuron` → `Seed`）时只搬了目录、没清用户级环境变量。

- **先排除本仓嫌疑**：全仓 grep `taiji-neuron` 只命中历史痕迹——`neuroplex/loader.py:23` 的注释（说明历史 ckpt 用 `taiji.*` 命名空间序列化）与 `plans/archive/**` 里的旧绝对路径链接。无任何活代码创建该目录。
- **真实成因**：三个 **用户级（注册表）** 变量仍指向旧路径：`XDG_CONFIG_HOME=E:\taiji-neuron\.local\config`、`XDG_STATE_HOME=E:\taiji-neuron\.local\state`，以及 `Path` 中的 `E:\taiji-neuron\.npm-global`（机器级作用域干净）。遵循 XDG 规范的工具（`opencode`、`gh`）启动时若发现目标路径不存在会**整条重建目录链**，所以删除永远不生效。证据：目录 `CreationTime` 为 2026-08-25，而 `.local\config\opencode` 的 `LastWriteTime` 是 08-27 11:04（当天），且内容清一色是工具配置/状态（3672 文件、278 目录，绝大多数是 `opencode\node_modules`），零项目代码。
- **动手前先证明无损**：`opencode.jsonc` 只有一行 `$schema`（无个人配置）；`gh auth status` 显示登录来自 `GH_TOKEN` 环境变量而非该目录（`.local\state\gh\device-id` 只是 36 字节匿名遥测 ID）；`npm config get prefix` 本就是 `C:\Users\23747\AppData\Roaming\npm`，而 `E:\taiji-neuron\.npm-global` **根本不存在**（死 Path 项）。因此「改回系统默认位置」无需迁移任何数据。
- **已执行**：备份用户 `Path` 至 `C:\Users\23747\user_path_backup_20260827.txt` → 用户级删除 `XDG_CONFIG_HOME`/`XDG_STATE_HOME` → 从用户 `Path` 过滤掉含 `taiji-neuron` 的段。复核：用户级与机器级全变量扫描已无任何 `taiji` 命中，`npm prefix` 为默认值。
- **验证纪律（易踩）**：注册表改动**不回灌已运行的进程**，而子进程继承父进程的环境副本，所以必须用 `[Environment]::GetEnvironmentVariable(..., "User")` 直读注册表来判定，不能看 `$env:`。实测新开子进程里 `$env:XDG_CONFIG_HOME` 仍是旧值——说明当前 IDE 进程仍持有旧变量，**由它拉起的工具还会重建该目录**，须重启 IDE/终端后再删。
- **未完成的一步**：`Remove-Item E:\taiji-neuron -Recurse -Force` 被沙箱拒绝（仅允许写 `E:\Seed`），目录仍存在，需用户手动删除。环境变量已清，故删除一次即永久生效。

### 14.13 `taiji/` 对 torch 的依赖面实测（2026-08-27，为公网 demo 可行性做的前置核查）

起因：讨论"能否把成果部署到公网"。结论先行——现有 `frontend/` **不能**直接静态托管，三条代码级证据：`src/composables/apiClient.js` 的 `resolveApiBase()` 在生产构建下推导后端为 `${hostname}:8000`（部到 Pages 会去请求 `https://xxx.github.io:8000`）；`vite.config.js` 的 `server.proxy` 只在 `vite dev` 生效、`build` 后消失；同文件的 `strip-crossorigin` 插件注释明写"QWebEngineView 兼容"，说明它本就是桌面壳内嵌页。且 `api/` 有 36 个路由模块（terminal、训练控制、模型切换、workspace 文件读写），整站公开托管是**安全问题**而非难度问题，已排除。故候选收敛为「静态成果页」与「WASM 内核 demo」，后者需要一个不依赖 PyTorch 的最小推理内核，遂做本核查。

- **命名事实（先纠错）**：`taiji_native` 不是包名，是 `e:\Seed\taiji\` 在测试目录、脚本名与报告名里的对外称呼。`Glob **/taiji_native/**/*.py` 只会命中 `tests/taiji_native/`，据此判断"实现不在这里"是错的。
- **截断陷阱（易踩）**：全仓 `import torch` 的 grep 输出被截断到 100 个文件，可见结果里没有 `taiji/*.py`，一度误判为"零 torch 依赖"。**输出被截断时"没出现"不等于"不存在"**，必须把 path 收紧到目标目录重跑才算证据。实测 `taiji/` 有 28 个模块共 33 处 `import torch`。
- **决定性事实：torch 只承担张量库角色，不承担自动微分**。全包 `.backward()` **0 处**（仅 `model.py:31`、`world_learning.py:335` 的注释提到"不用它"）、`torch.optim` **0 处**、`torch.autograd` **0 处**；`@torch.no_grad()` / `with torch.no_grad()` 共 **77 处**覆盖全部学习与推理函数；`nn.Parameter` 仅 1 处且 `local_learning.freeze_parameters` 会 `requires_grad_(False)`。这是 14.5 那次学习平面迁移的直接后果：`taiji/local_learning.py` 手写了 `backproject_linear`、`gru_forward_trace`、`cosine_similarity_delta` 与复刻 Adam 更新式的 `LocalAdam`。**移植 torch 到 WASM 的真正难点是 autograd tape / 动态图 / dispatcher，这三样已被整体绕开。**
- **算子面（43 个 distinct torch 函数，无硬骨头）**：`zeros`54 `tensor`45 `stack`32 `mean`28 `zeros_like`28 `cat`26 `abs`18 `empty`17 `linalg.vector_norm`16 `softmax`11 `Generator`10 `tanh`10 为主体，其余为 `arange`/`relu`/`sigmoid`/`clamp`/`eye`/`dot`/`outer`/`bincount`/`argmax`/`multinomial` 等初等算子；`nn.functional` 侧只有 `cosine_similarity`/`cross_entropy`/`one_hot`/`normalize`/`mse_loss`/`softplus`/`softmax`/`binary_cross_entropy_with_logits`。nn 层仅 8 种：`Linear`25 `Module`8 `GRU`4 `Embedding`1 `Parameter`1 `Sequential`1 `Tanh`1。张量方法以 `.detach()`178 `.clone()`174 `.to()`140 `.unsqueeze()`33 为主，难替代的索引/稀疏算子极少（`scatter_`1、`scatter_add_`2、`index_select`2、`masked_fill`2），**无 einsum、无 conv、无 sparse_coo_tensor、无 as_strided**。仅 `torch.linalg.solve`（2 处，稠密小矩阵）与 `nn.GRU` 需要专门实现，而 GRU 门控数学已在 `local_learning.py:246-303` 被逐步展开。
- **最小推理内核边界：8 / 35 模块、6623 行（占 `taiji/` 25404 行的 26%）**。从 `taiji.model.Taiji` 出发的传递闭包为 `contracts`2583 + `memory`907 + `model`810 + `config`603 + `fabric`591 + `sparse`548 + `state`337 + `organs`244。**闭包外 27 个模块与推理无关**，含最大的 `adapter.py`(6803) 以及 `concept_formation`/`perception`/`workspace`/`world_learning`/`planning`/`executive`。`contracts.py` 虽占闭包 39%，但是纯 dataclass + `_check_version`/`_check_text`/`_check_unit` 校验 + hashlib，且 `fabric`/`sparse` 只从它取一个 `StructuralTopologyProposal`，实际可再砍大半。
- **checkpoint 序列化耦合：低**。全包 `torch.save` / `torch.load` **0 处调用**；`checkpoint()` 统一返回 `{name: tensor.detach().cpu().clone()}` 的纯 dict（见 `perception.py:910`、`workspace.py:342`、`affordance.py:529` 等），`CHECKPOINT_FORMAT = "taiji-native-v8"` 为自定义格式。导出为 JSON / safetensors / 裸 Float32Array 不需改动内核逻辑，只需在边界写 dump 脚本。
- **尚未核查、不得先行决定的一项**：上述 6623 行是 Python，WASM 化有两条路且各有代价——(a) Pyodide 装载 Python + 一层纯 JS/WASM mini-tensor 替换 torch：快，但需下载约 6MB 运行时并写 torch shim；(b) 用 Rust/C++ 重写内核再编译：产物干净，但等于重写数值代码，且必须与 Python 版保持逐位一致，风险最高——`plans/manifests/taiji_native_runtime_profile_v1.json` 的 controls 里本就有 `cross_device_numerical_consistency_when_available`，说明项目自身对数值一致性有硬要求。此二选一需要独立尽调后再定，**不能因为依赖面结论乐观就顺势拍板**。

## 15. 停止项

在 P2 通过前：

- 不续跑旧 16M→100M raw-byte 长训；
- 不为 TSK-v8 继续增加认知补丁；
- 不写绑定固定 fan-in 的自定义 CUDA kernel；
- 不用增加神经元数量替代学习型抽象；
- 不删除 Legacy 对照；
- 不把旧 N/M 通过记录宣传为完整智能进展。

## 16. 当前唯一下一步

**已完成（2026-08-27）：仓库可发现性元数据三项全部落地并通过程序化复核。** 由用户在 GitHub 网页端写入（agent 令牌缺 `administration:write`，详见 14.10），agent 侧用 `gh repo view --json description,repositoryTopics,homepageUrl,usesCustomOpenGraphImage,openGraphImageUrl` 复核：

- description：与定稿文本**逐字符相等**，252 字符（350 上限内）。
- topics：**集合相等**，13/13，missing 0、unexpected 0。GitHub 返回时按字母重排，故只判集合不判顺序。清单为 `predictive-coding` `cognitive-architecture` `episodic-memory` `hebbian-learning` `local-learning` `online-learning` `computational-neuroscience` `neuromorphic-computing` `sparse-neural-networks` `world-models` `pytorch` `deep-learning` `artificial-intelligence`（选取依据与实测仓库计数见 14.10）。
- social preview：`usesCustomOpenGraphImage: true`，`openGraphImageUrl` 指向 `repository-images.githubusercontent.com/1301491809/69a87c39-…`，即 GitHub 已完成 CDN 转存。图源为 `frontend/public/social-preview.png`（1280×640，139.9 KB），由 `scripts/make_social_preview.py` 生成，改文案后重跑即可重出图；脚本内置两道非零退出自检（墨迹 bbox 须落在四边 100px 安全边距内、产物须 <1 MB），首版两个已修正的缺陷记录见 14.10 上游条目。图不走仓库文件系统，仓库内保留 PNG 与脚本仅为可复现。
- homepageUrl：**刻意留空**。项目暂无独立站点，填 README 锚点等于制造一个自指链接，对访客无增量信息。

后续若要让 agent 自行改这类元数据，唯一有效做法是换执行主体：提供带 `Administration: Read and write` 的 fine-grained PAT 作为 `GH_TOKEN`（或给该 App installation 补 Administration 权限），届时下面两条命令即可放行。

```bash
gh repo edit liulang5945-netizen/Seed --description "Byte-level predictive-coding kernel that learns online from local prediction errors: no backpropagation, no attention matrix, no optimizer. Sparse fixed-fan-in synapses, slot-free distributed episodic memory, lesion-controlled reproducible experiments."

gh api -X PUT repos/liulang5945-netizen/Seed/topics \
  -f "names[]=predictive-coding" -f "names[]=cognitive-architecture" \
  -f "names[]=episodic-memory" -f "names[]=hebbian-learning" \
  -f "names[]=local-learning" -f "names[]=online-learning" \
  -f "names[]=computational-neuroscience" -f "names[]=neuromorphic-computing" \
  -f "names[]=sparse-neural-networks" -f "names[]=world-models" \
  -f "names[]=pytorch" -f "names[]=deep-learning" \
  -f "names[]=artificial-intelligence"
```

注意 social preview 例外：GitHub 从未提供该字段的 REST/GraphQL 接口，**任何令牌都写不了**，只能人工在 Settings → Social preview 上传，换 PAT 也不能自动化这一项。

**已完成：`TSKV8Adapter.step_cross_region_network()` 已把 growth/pruning/split/merge 所需的 activity、route evidence、prediction error、learning gain、holdout transfer 和资源压力接入可 checkpoint 的 runtime observation；Gate 为 `reports/taiji_runtime_structure_20260826.json`。无 expected activity 时不伪造 growth supervision，route credit 来自实际 target activity，runtime tick 不直接改变 topology。**

**已完成：真实 runtime evidence 已汇聚为有边界、可 checkpoint、按 substrate 去重的 `StructuralProposalCandidate` 队列；候选覆盖 split、region/connection prune 和兼容 region merge，保存证据、source tick、priority、参数与 resource cost，且不会绕过 ledger 直接改变 topology。**

**已完成：candidate 可幂等 materialize 为 pending `StructuralTopologyProposal`，candidate→proposal lineage 随 native checkpoint 恢复；materialization 不改变 live topology。**

**已完成：split、region-prune、connection-prune、merge candidate 已接入统一 holdout validator dispatch；验证只更新 pending proposal 的 validation score/status，未验证 candidate 仍被 commit gate 阻断。**

**已完成：统一 commit/rollback dispatcher 已按 topology role 路由 candidate，依次执行 holdout score、budget、trial checkpoint、live topology mutation 和 latest-change reverse rollback；runtime Gate 覆盖 commit 后拓扑变化、父结构恢复和 checkpoint continuation。**

**已完成：candidate queue 已与真实 holdout 数据绑定为逐项 fail-closed maintenance cycle；两个 runtime candidates 完成 materialize、holdout、commit、rollback，缺数据/异常/预算不足不会绕过 ledger，`StructuralMaintenanceResult` 随 native checkpoint 恢复。**

**已完成：直接 neuron birth (`add`) 已纳入同一 candidate contract；`TSKV8Adapter.step_adaptive_neuron_region()` 从真实 standalone region tick 生成带 substrate/evidence/source tick/priority/resource cost 的候选，统一 materialize、holdout validator、commit/rollback dispatcher 和 native checkpoint 均已通过 `reports/taiji_runtime_structure_20260826.json` 的 direct-add 子门禁。**

**已完成：maintenance cycle 已具备显式 candidate dependency/conflict 判定；反向输入仍按依赖拓扑顺序执行，依赖失败会阻断下游，同一 substrate 的竞争变更全部 `failed_closed`；不同 neuron identity 的 `add` 可并存并按依赖连续出生，队列只对同一目标 unit 去重。**

**当前唯一下一步：建立三层以上自适应区域的规模化结构维护 Gate，覆盖跨区域 route、混合 add/split/prune、资源竞争、checkpoint continuation 和拓扑不变量。**

**已完成：三层自适应区域规模化结构维护 Gate 已通过；`source→relay→target` 显式 route 在 connected split 后保留并按受影响边展开，standalone neuron `add` 可与 network split 混合进入同一 maintenance cycle，checkpoint continuation、资源预算和双向 rollback 均通过。**

**当前唯一下一步：对已落地的 native sparse neuron/network runtime 做 CPU/CUDA 实际热点剖析，建立跨设备 checkpoint 恢复与数值一致性基线，再决定是否需要 fused/sparse kernel。**

**已完成：native sparse neuron/network runtime profile 已执行并通过；本机为 `torch 2.13.0+cpu` 且无 CUDA，报告明确记录 CUDA 未执行。CPU region/network 热点、CPU checkpoint roundtrip 和 continuation 均通过，CPU 实测分别约为 `18,735 ticks/s` 与 `5,291 ticks/s`，主要热点为 `aten::_to_copy`、`aten::to`、`aten::index`。**

**当前唯一下一步：消除 native tick 中可避免的设备/标量转换与临时分配，复跑同一 profile 并比较热点/吞吐；在获得 CUDA-capable 主机前不写 fused/sparse kernel。**

**已完成：native tick hardening 已落地并通过；稀疏输入仅在设备不一致时转换，norm 常量按 device/dtype/limit 缓存，network 的 zero scratch vector 按区域复用且不进入 checkpoint。重跑 profile 仍通过 CPU profile、checkpoint roundtrip 与 continuation，热点收敛到 `aten::index`、`aten::sum` 及少量 copy；本机仍无 CUDA，故没有伪造 CUDA 结论。**

**当前唯一下一步：将当前 profile 固化为稳定的性能回归基线，并在 CUDA-capable 主机上复跑同一 workload；在此之前不引入自定义 fused/sparse kernel。**

**已完成：profile 固定 workload 与 manifest 已提交，CPU profile 报告、checkpoint roundtrip/continuation 和 network scratch 复用回归测试均已固化；吞吐仅作为本机观测，不钉死为跨设备硬阈值。**

**当前唯一下一步：在 CUDA-capable 主机上复跑同一 workload，完成跨设备输出/checkpoint continuation 验证，再依据真实热点决定是否引入 fused/sparse kernel。当前 CPU-only 环境不宣称 CUDA 已验证。**

**已完成：`taiji/` 的 autograd 学习平面已整体替换为原生局部信用分配（详见 14.5）。8 个模块 13 处 `loss.backward()` 全部迁移至 `taiji/local_learning.py`，`no_autograd_parameters` 从假绿转为真检查并通过；原生性契约 15/15、8 道阻塞 verify 全 pass、`tests/` 437 passed / 5 skipped、lint 三件套与版本一致性全绿。`LocalAdam` 保留 Adam 更新式，因此未连带改变任何已调优的学习率。**

**已完成：迁移已获 CI 实证。`02f6602` 的 Linux job 上 8 道阻塞 verify 全部转绿，其中 `no_legacy_or_transformer_dependency` 从 `ffe1da2` 的 false 变为 true——该项此前红了 7 次连续构建，根因确为 autograd 学习平面，不是环境差异（详见 14.7）。同一构建暴露出唯一剩余红点 `tests/test_terminal_input_normalization.py` 的 3 条平台未钉定用例，与本次迁移无关，已按 14.5 修复并在本机复现 Linux 条件验证。**

**已完成：`test` 转绿后首次真正执行的 `build-frontend` / `docker-build` 双红已定位并修复（详见 14.8）。两者因 `needs: test` 在此前 7 次连续红期间一直是 skipped、从未运行，故属被遮蔽的既存缺陷而非本次回归。`docker-build` 的 `COPY data/` 已收敛到项目既有的挂载约定；`build-frontend` 的 2 个 high CVE 已用 `overrides` 全树强制到安全版。前端四道门禁本机全绿：audit 退出 0（余 2 moderate 不阻塞）、eslint 0 errors / 17 warnings、vitest 19 files 160 passed、build 成功且 `dist/index.html` 存在，`npm ci --dry-run` 退出 0 证明 lock 与 package.json 同步。Docker 侧因本机无 Docker 未做构建验证，改以 `COPY` 源跟踪文件数静态审计替代，并已如实记录。**

**已完成：`build-frontend` 已在 CI 实证转绿（`32982579047`，1m37s），5 个上游 job 亦全绿（test 3.10 13m25s、test 3.12 13m8s、test-windows 6m27s、两个 startup smoke）。`docker-build` 的 `COPY data/` 根因确认修复——`Build image via docker compose` 与 `Verify Docker image metadata` 均已打勾；但其后从未运行过的 `Startup smoke and healthcheck` 暴露出下一层缺陷 `ModuleNotFoundError: No module named 'seed_platform'`。根因是 Dockerfile 手工 `COPY` 清单漏了 pyproject 已声明的 `seed_platform*`，而 `packages.find` 静默跳过缺失目录使 `pip install` 仍退出 0（详见 14.8）。修法为补齐 `COPY` 并加构建期导入断言 `RUN python -c "import api.app"`，使漏拷贝此后在 build 层即刻失败；清单与 pyproject 已复核对齐（`MISSING: none`），断言语句本机实测退出 0 证明不会误红。**
**已完成：CI 的「基线不可复现」根因已修。`ci.yml` 原只钉 `ruff`/`black`，`mypy`/`pip-audit` 浮动，导致门禁数字在无人改代码时自己漂移（核心 47→63、全仓 259→281）；现已钉 `mypy==2.3.1`、`pip-audit==2.10.1`，并确立通用规则「凡把工具输出数字当阈值的门禁，工具本身必须钉版本」。同时 mypy 核心门禁由 advisory 升为**棘轮 blocking**（初始 `MYPY_CORE_BASELINE=63`，超基线即 `exit 1`，解析不到数字亦 `exit 1` 绝不静默放行，低于基线打 `::notice::` 提示下调；当前基线已收紧为 0）。双矩阵实测否证了「报错数随 Python 版本变化故不能设阈值」——3.10 与 3.12 的核心数 63、全仓数 281 完全相同（详见 14.2）。停机期三次假红已判定为平台产物并记入 14.11。**

**已完成：核心 mypy 类型债已归零。2026-08-27 在当前固定工具链 `mypy==2.3.1` 下，`python -m mypy --follow-imports=silent seed taiji` 对 44 个源文件报告 0 错误；修复覆盖 checkpoint/state_dict 类型收窄、可选值边界、结构化参数契约、局部 GRU 学习张量和可替换语言器官协议。定向 ruff 通过，相关回归 46 passed。`.github/workflows/ci.yml` 的 `MYPY_CORE_BASELINE` 已同步从 63 收紧至 0。**

**已完成：核心类型债提交 `11ca75c` 已在当前 `main` 与 `origin/main` 同步；本地 `mypy==2.3.1` 对 `seed taiji` 的 44 个源文件保持 0 错误。GitHub Actions 实跑仍未因 CLI 未认证而声称完成。**

**已完成：P2/A1 感知训练—运行时边界合同已修正。`LearnedPerception.fit_predictive()` 现在复用与 `observe()` 相同的 prediction-error、surprise baseline、hysteresis 和 maximum-duration 边界时钟；训练使用动态 assembly 的每个活动前缀监督，运行时与训练不再分别使用固定滑窗/可变切段。新增训练 rollout 与 runtime boundary 的回归测试；定向 P2 回归 8 passed，完整 `tests/taiji_native` 为 192 passed、1 skipped，另有 2 个 Windows pytest 临时锁 setup error。旧 next-byte A1 在真实 manifest 上仍未通过，说明评测任务本身还需继续提高语义层级，不能把这次合同修复冒充 Gate 通过。**

**已决定：CUDA 相关 profile、跨设备 checkpoint 和 fused/sparse kernel 暂缓，直到具备 CUDA-capable 主机；本轮继续推进 CPU 可验证的 Taiji 能力，不修改 CUDA 结论。**

**已完成：动态 assembly pooled state 已从无序均值提升为可学习的顺序敏感读出。`assembly_recency_logit` 通过正值 softplus 增益学习当前活动的 recency 权重；训练、运行时和 checkpoint 共用同一池化公式，不新增固定词表或 Transformer 组件。训练暴露统计又形成连续 novelty 信号，参与 boundary competition 并随 checkpoint 保存、在线更新；checkpoint 往返、mypy 0、ruff 通过，P2 定向回归 9 passed。**

**已完成：A1 边界合同已改为以 marker 位置的 boundary rate 作为因果指标；整段 aggregate boundary rate 仍保留为诊断，因为插入 marker 会改变序列长度并重排邻近 assembly。**

**已完成：A1 Gate 已收紧为所有 seed 的最差 generalization、marker score/rate 和 random-chunk drop，而不是只看 primary seed；正式报告 `reports/taiji_a1_perception_20260827.json` 如实为 `gate_passed=false`：primary gain=`+0.0089`，但最差 seed gain=`-0.0398`，最差 random-chunk drop=`+0.0030`，marker score/rate 最差仍为 `+0.1299/+0.1095`。这关闭了“单个幸运 seed 收口”的评测漏洞，同时证明 novelty 已修复边界响应但未完成稳健组合迁移。**

**已完成：A1 predictive temperature 从 `0.15` 调整为 `0.5`，同一 32/16 smoke manifest 在严格最差 seed 口径下通过：三 seed 的 generalization gain=`+0.0022/+0.0035/+0.0231`，random-chunk drop=`+0.0135/+0.0092/+0.0116`，marker score/rate 最小=`+0.2025/+0.4262`，报告为 `reports/taiji_a1_perception_20260827.json`。**

**已完成：独立规模化验证已执行于 `shared_core` 的 128 train / 64 holdout manifest，报告为 `reports/taiji_a1_perception_shared128_20260827.json`；跨 seed std=`0.0048`、marker score/rate 最小=`+0.2872/+0.5606`，但最差 generalization gain=`-0.0058`、最差 random-chunk drop=`+0.0023`，Gate 仍为 `false`。这说明 temperature 修复和边界 novelty 已有效，但 assembly 的组合迁移和 lesion 抗性尚未规模化稳定。**

**已完成：顺序敏感 assembly 已加入可 checkpoint 的多步预测信用分配；`multi_step_prediction_weight=0.05`、`horizon=4` 纳入 A1 默认，误差沿连续 transition 展开并回写 assembly/transition/embedding 的原生局部梯度。smoke 32/16 在三 seed 下 Gate 通过；128/64 独立 manifest 仍为 false（最差 gain=`-0.0010`、最差 random drop=`-0.0001`），故没有用多步模块掩盖规模化失败。**

**已完成：跨 assembly 边界结构已落地。训练先按与 runtime 相同的闭合 boundary 映射出“边界后的下一段”，该段不跨越下一条 runtime boundary；可选的 boundary-after 多步 CE 已实现但默认权重为 `0.0`，默认使用跨后续 assembly 的多步对比负样本（`cross_assembly_negative_weight=0.01`），对不同 boundary 后段的上下文进行显式正/负匹配。该目标通过 native local-credit 路径回写 embedding/transition，未引入 token 表、固定段表或 Transformer。checkpoint 往返、定向回归 `10 passed`、核心 mypy 0、Ruff 和 Black API 检查均通过；relation subgate 复核为 true。**

**已完成：A1 感知 Gate 已在两级规模正式通过。smoke `32/16` 报告 `reports/taiji_a1_perception_20260827.json` 的最差泛化=`+0.00310`、最差 random-chunk drop=`+0.00578`、marker score/rate 最小=`+0.1734/+0.3567`、cross-seed std=`0.00608`；`shared_core` `128/64` 报告 `reports/taiji_a1_perception_shared128_20260827.json` 的最差泛化=`0.0`、最差 random-chunk drop=`+0.00527`、marker score/rate 最小=`+0.2161/+0.4483`、cross-seed std=`0.00834`。两份报告均为 `gate_passed=true`；完整 `tests/taiji_native` 为 `193 passed, 1 skipped, 2 errors`，两个 error 仍是 Windows pytest 临时目录锁 setup 权限问题，未进入测试体。**

**已完成：P2→P3 lineage contract 已接入 runtime。`WorkspaceState` 与 `WorldState` 新增可选的 `percept_event_id`、`percept_assembly_id`、`percept_boundary_closed`；`TSKV8Adapter.observe()` 在生成 lineage 后同步写入两者，`observe_event(world_state=...)` 与 `settle_action(world_state=...)` 的外部状态替换都会保留当前来源，native checkpoint/restore 可恢复。closed boundary 若缺 event/assembly ID 会 fail closed；定向 world/concept/v1 回归 `21 passed`，Ruff、核心 mypy 0 通过。该改动只建立 provenance contract，不把 lineage 存在冒充为跨 episode 能力。**

**已完成：P2→P3 perception-to-world closure Gate 已通过。`scripts/training/eval_taiji_p2_p3_closure.py` 用 64 train / 32 新对象与新候选组合 holdout、3 seeds 驱动真实 `TSKV8Adapter`；每个样本先经历两次 raw observation 和 boundary-closed percept，再把 learned/none workspace 选择绑定到 `TaijiWorldState` 的对象—关系 transition。报告 `reports/taiji_p2_p3_closure_20260827.json` 的 learned route/world transition 最差均为 `1.0`，none lesion 最高为 `0.0`，lineage 最差为 `1.0`，192 次 boundary-closed assembly、三 seed checkpoint continuation 和 world checkpoint roundtrip 全部通过；shared16 relation subgate 复核为 true。该 Gate 关闭 provenance-to-world 的窄闭环缺口，不宣称长程世界模型或开放域语义智能。**

**已完成：P2→P3 variable-horizon continuation Gate 已通过。`scripts/training/eval_taiji_p2_p3_variable_horizon.py` 在 64 train / 32 holdout、3 seeds、3/4/5 个 closed assembly 上驱动真实 `TSKV8Adapter`；learned route/world success、lineage、两步 history、checkpoint continuation、`TaijiWorldState` roundtrip 和 runtime `WorldDynamicsLearner` online calibration 均为 `1.0`，workspace lesion route 为 `0.0`。第一步后通过真实 bridge observation 消费 pending experience，再从 native checkpoint 续接第二步；第二步使用训练 schema 未见的 `secured` relation，`assembled → secured` progression Gate 为 `1.0`。新增同 tick `TaijiWorldState.synchronize_observation()` 只同步观察快照、不伪造 action transition，并保持历史 checkpoint 连续。该 Gate 证明变量时长与跨 checkpoint 的两步因果续接已成立，但未知 relation 当前只保证被保存和传递，不宣称 world learner 已完成开放集关系预测。报告为 `reports/taiji_p2_p3_variable_horizon_20260827.json`，manifest 为 `reports/taiji_p2_p3_variable_horizon_manifest_20260827.json`。**

**已完成（旁支，不改主线）：公网 demo 的前置核查已出结论，记入 14.13。`taiji/` 对 torch 的依赖是"张量库"而非"自动微分引擎"（0 处 `.backward()` / 0 处 `torch.optim` / 0 处 `torch.autograd`，77 处 `no_grad`），算子面为 43 个初等 torch 函数 + 8 种 nn 层，`taiji.model.Taiji` 的推理闭包仅 8/35 模块 6623 行，checkpoint 为自定义纯 dict 格式且零 `torch.save`。故 WASM 内核 demo 技术上可行；但 Pyodide+shim 与 Rust/C++ 重写的二选一尚未尽调，未做决定。本项不占用主线唯一下一步。**

**已完成：P2→P3 open-set world schema evolution Gate 已通过。** `WorldSchema` 现在以语义 feature key 扩展 object、numeric attribute、relation、action kind、actor、target 和 parameter；`WorldDynamicsLearner` 可在运行时注册新边界、按语义迁移旧 input/output 权重、保留 checkpoint 并通过真实 `WorldTransition` 的 outcome feedback 在线校准。`TSKV8Adapter` 的 runtime 适配路径不会把 `action_symbol` 等控制元数据误注册为 world 参数；`begin_episode()` 保留 world/events/concepts/calibration 和已学网络，只清理 episode transient。`TaijiWorldState.advance_observation()` 补齐了无动作感知跨 tick 的 owned snapshot 推进，action history 仍只记录真实 transition。正式报告 `reports/taiji_p3_open_set_20260827.json` 在 64 train / 32 holdout、3 seeds 下通过：learned route/world、relation progression、open object/relation/action、lineage、跨 episode、checkpoint、四 transition ownership、history/roundtrip、calibration 均为 `1.0`，workspace lesion route 为 `0.0`；完整 native 回归为 `203 passed, 1 skipped`，另有 2 个 Windows pytest 临时目录锁 setup error，未进入测试体。该 Gate 证明的是可扩展 schema 与窄因果闭环，不是开放域语义理解或通用智能。

**当前唯一下一步：建立跨 episode 的 Taiji world schema registry 与可回滚生命周期 Gate。** 将当前单 learner 的即时扩展提升为可审计 registry：统一 canonical identity/alias、关系 predicate 的新增与冲突、slot confidence、checkpoint lineage、版本回滚，以及在资源上限下的 merge/prune/tombstone；用混合旧/新 schema 的 holdout 和矛盾 outcome 验证不会静默覆盖旧知识，CUDA 继续暂缓。

**已完成：P3 world schema registry lifecycle Gate 已通过。** `WorldSchemaRegistry` 为 schema 提供 revision proposal/commit/rollback、canonical object alias、slot confidence、矛盾 feedback 记录、资源预算、prune+tombstone 和 checkpoint lineage；`WorldDynamicsLearner` 保存每个 revision 的网络快照，可在不重置旧权重的前提下回滚 schema 与网络，`TSKV8Adapter` 原生 checkpoint 同时恢复 registry 和多版本快照。正式报告 `reports/taiji_p3_schema_registry_20260827.json` 在 3 seeds 下全部通过：alias 稳定与冲突拒绝、旧/新 schema 混合预测、旧权重保留、矛盾反馈 fail-closed、预算阻断、prune/tombstone、rollback 和 checkpoint rollback 均为 `1.0`。生命周期单测 3 passed；核心 mypy 0、Ruff/Black 通过；native 全量为 `208 passed, 1 skipped`，另有 2 个既有 Windows pytest 临时目录锁 setup error，未进入测试体。该 Gate 验收 schema 生命周期安全，不宣称开放域关系语义已解决。

**已完成：P3 registry adjudication 已接入真实 adapter `WorldTransition` outcome 闭环。** `WorldSchemaRegistry` 对不含 tick/event 噪声的 semantic before/action 生成稳定 evidence key，并保存 after-state/reward/success outcome signature；跨 episode 的一致结果会增加主假设置信度，矛盾 after-state 会写入 conflict ledger 且不覆盖既有证据。`WorldDynamicsLearner.online_update()` 在本地信用分配前执行 adjudication，矛盾样本 fail-closed，不增加 `online_updates`；`TSKV8Adapter` 的 `WorldCalibrationTrace.calibration_applied` 与实际接受的 update 对齐，并把 accept/reject 计数和 registry evidence 一同纳入 native checkpoint。真实 adapter Gate `scripts/training/eval_taiji_p3_transition_adjudication.py` 在 seeds `11/29/47` 全部通过：跨 episode confidence=`1.0`、relation-specific contradiction、prediction calibration、no-update-on-reject、registry/network checkpoint 和 continuation 均为 `1.0`；报告/manifest 为 `reports/taiji_p3_transition_adjudication_20260827.json` 与 `reports/taiji_p3_transition_adjudication_manifest_20260827.json`。定向闭环为 `23 passed`；native 全量为 `211 passed, 1 skipped, 2 errors`，2 个 error 仍是 Windows pytest 临时目录锁 setup 权限问题，未进入测试体；核心 mypy 0、Ruff/Black 通过。该 Gate 证明真实反馈的证据一致性与 fail-closed ownership，不宣称随机世界建模或开放域关系语义已解决。

**已完成：P3 多假设 outcome ledger Gate 已通过。** transition evidence 不再只保留单一结果签名，而是按 semantic context 记录多个 after-state/reward/success 假设、evidence count 和 outcome probability；一过性矛盾进入 `conflicted`，两个及以上候选都达到重复支持后进入 `stochastic`。`WorldDynamicsLearner` 只在当前样本属于拥有明确 lead 的主假设时更新，平票或少数结果 fail-closed；`TSKV8Adapter` 的 calibration trace 反映真实接受/拒绝结果。3 seeds 的 deterministic/stochastic lesion、relation-specific holdout、跨 episode、registry/network checkpoint continuation 全部通过，stochastic 主假设 confidence=`0.6`；报告/manifest 仍为 `reports/taiji_p3_transition_adjudication_20260827.json` 与 `reports/taiji_p3_transition_adjudication_manifest_20260827.json`。新增 ledger 单测与真实 adapter Gate 通过；CUDA 继续暂缓。该 Gate 证明有限证据下的结果不确定性边界，不宣称已完成概率世界预测。

**已完成：P3 ledger outcome 分布已接入 world prediction 与 uncertainty。** `WorldSchemaRegistry` 保存每个 outcome hypothesis 的 reward/success 统计，并统一返回 `unseen`、`deterministic`、`conflicted`、`stochastic` 四种不确定性语义；`WorldDynamicsLearner.predict()` 对已观测 context 使用 ledger 的经验 outcome estimate，对未见 context 保留网络预测但标记最高 uncertainty。`WorldPredictionRecord`、native checkpoint 和规划器候选都传递同一 uncertainty；真实 adapter Gate 验证 known=`0.0`、conflicted=`1.0`、stochastic=`0.5`，3 seeds 全部通过，核心 mypy 0、Ruff/Black 通过。该 Gate 关闭了“模型没见过”和“环境本身多结果”混为一谈的接口漏洞，但不等价于概率校准质量已达标。

**已完成：P3 ledger-driven probability calibration Gate 已通过。** `scripts/training/eval_taiji_p3_probability_calibration.py` 使用独立 relation-specific holdout，只测量不写回 transition ledger；3 seeds 的 success Brier=`0.24`、binary NLL=`0.6730`、组级 confidence coverage=`1.0`、reward MAE=`0`，holdout evidence count 未变化。未知 target/`tracks` relation 保持 `unseen/1.0`，已观测随机 context 保持 `stochastic/0.4`，native checkpoint continuation 与无 world learner 的 planner lesion 全部通过；报告/manifest 为 `reports/taiji_p3_probability_calibration_20260827.json` 与 `reports/taiji_p3_probability_calibration_manifest_20260827.json`。该 Gate 证明测量边界和 ledger 不泄漏，不宣称开放世界概率预测质量。

**已完成：P3 outcome hypothesis 分布已接入多步 imagined rollout 的风险敏感规划。** `PlanningCandidate` 与 `ImaginedRollout` 现在携带 `uncertainty_mode`；`TSKV8Adapter.predict_world_candidates()` 和 `imagine_world_rollout()` 将 world learner 的 `unseen/deterministic/conflicted/stochastic` 语义传入规划层，`GoalPlanner` 按可配置 multiplier 对不同风险类型施加惩罚，并在单步与多步 rollout 使用同一公式。概率校准 Gate 已扩展为 3 seeds 的风险敏感单步/多步选择、relation holdout、ledger 隔离、checkpoint continuation 与 world-model lesion，全部通过；报告/manifest 为 `reports/taiji_p3_probability_calibration_20260827.json` 与 `reports/taiji_p3_probability_calibration_manifest_20260827.json`。该 Gate 证明风险语义已贯穿 imagined scoring，不宣称真实环境已自动执行整条随机 rollout。

**当前唯一下一步：建立 stochastic/conflicted rollout 的真实逐步执行闭环。** 让真实 environment 每一步的 after-state 回写 ledger 与 prediction trace，成功/失败或歧义时按 risk mode 中止或重规划，并验证从中断 checkpoint 继续不会重复消费或污染 outcome evidence；CUDA 继续暂缓。

**已完成：P3 stochastic/conflicted risk-sensitive execution Gate 已通过。** `TSKV8Adapter.execute_imagined_rollout_step()` 现在把真实 environment 的 after-state、outcome adjudication、ledger uncertainty 和证据数量写入 `WorldCalibrationTrace`；ledger 拒绝不明确结果时 fail-closed，不增加 world learner 的 `online_updates`，并触发当前 rollout 中止/重规划。非终止失败或负奖励也会清空剩余 imagined plan 并要求重规划。`scripts/training/eval_taiji_p3_risk_sensitive_execution.py` 在 seeds `11/23/37` 下验证了 stochastic (`uncertainty=0.4`) 与 conflicted (`uncertainty=1.0`) 两类风险、真实失败、恢复 rollout、trace checkpoint 恢复和 checkpoint 前后不重复消费证据，三 seed cross-seed gate rate=`1.0`；报告/manifest 为 `reports/taiji_p3_risk_sensitive_execution_20260827.json` 与 `reports/taiji_p3_risk_sensitive_execution_manifest_20260827.json`。定向风险/ledger/probability/P7 回归 `13 passed`，核心 mypy 0、Ruff/Black 通过。该 Gate 证明真实执行边界已能识别并阻断不可靠 outcome，不宣称已完成自动分支搜索或开放世界智能。

**当前唯一下一步：把布尔 `replan_required` 提升为可审计的 outcome-aware recovery branch contract。** 记录触发重规划的 transition evidence key、风险模式、被拒绝分支与剩余 rollout lineage；由 planner 生成并持久化排除失败分支后的候选 recovery rollout，checkpoint 续跑后自动恢复同一 branch context，并用冲突/随机/真实失败混合 episode 验证不会重复选择已拒绝分支；CUDA 继续暂缓。

**已完成：P3 outcome-aware recovery branch contract Gate 已通过。** 新增可 checkpoint 的 `RecoveryBranchState`，保存原 rollout/goal lineage、被拒绝 `WorldAction`、semantic evidence key、风险模式、剩余步数、触发原因和替代 rollout；`TSKV8Adapter.plan_rollouts()` 在恢复上下文中按 action semantic key fail-closed 过滤被拒绝首步，若没有替代分支则拒绝规划。真实执行中的 stochastic/conflicted adjudication rejection 与非终止失败都会建立 branch context，terminal recovery 完成后清理；native checkpoint 往返保留 branch，且不会重复写入 ledger evidence。三 seed 风险执行 Gate 与相关回归全部通过，核心 mypy 0、Ruff/Black 通过。该 Gate 证明 recovery branch 的 ownership、过滤与持久化边界，不宣称已经完成大规模分支搜索或开放世界智能。

**当前唯一下一步：建立多分支 recovery rollout 的真实选择与反事实评估 Gate。** 在同一失败上下文生成至少两个语义不同的替代 rollout，使用 ledger 风险、预测误差和真实后果对候选逐一评估，验证 planner 选择可执行且风险最低的分支；同时保持被拒绝首步不可重入、checkpoint continuation 不重复消费证据，CUDA 继续暂缓。

**已完成：P3 multi-branch recovery rollout Gate 已通过。** 恢复规划现在可同时接收被阻断分支、低风险 deterministic 分支和高风险 unseen 分支；planner 先按 recovery branch 的 action semantic key 排除已拒绝首步，再用 ledger uncertainty、预测 reward/success、预测误差和资源项统一评分，选出可执行的最低风险替代分支。三 seed 真实 environment Gate 均通过：被拒绝分支未重入、deterministic recovery 优于 unseen counterfactual、真实恢复结果成功终止、trace/checkpoint 与 evidence accounting 保持一致。该 Gate 验证分支选择和反事实风险排序，不宣称 recovery candidate 已能从开放世界自动生成。

**当前唯一下一步：把 recovery candidate synthesis 收回 Taiji-owned runtime。** 从当前 world affordance/schema/ledger 状态自动产生语义不同的替代 action 与 imagined rollout，统一做 schema 可执行性、预算和被拒绝分支过滤，再交给现有 risk planner；保留外部候选注入作为受控测试接口，CUDA 继续暂缓。

**已完成：P3 Taiji-owned recovery candidate synthesis Gate 已通过。** `TSKV8Adapter.synthesize_recovery_rollouts()` 从当前 `WorldAffordance`、可用 action-kind/motor capability、horizon 与 resource budget 生成结构化 recovery rollouts，并在生成阶段过滤被拒绝 action semantic key；生成结果继续经过 world learner 投影、risk planner 和 branch lineage checkpoint。三 seed 真实 environment Gate 验证了 `assemble` 被过滤、`idle` deterministic 低风险分支优于 `secure` unseen 反事实分支，最终执行成功且相关 trace/evidence/checkpoint 保持一致。该 Gate 关闭了 recovery 候选由调用方手工拼接的边界，不宣称 affordance 生成已经覆盖开放世界。

**当前唯一下一步：建立 recovery affordance 的真实可执行性与预算闭环。** 将候选生成的 resource cost、motor capability、world schema 可编码性和环境实际拒绝结果统一为可审计 Gate；对不可执行/超预算候选 fail-closed，并验证 checkpoint 续跑不会把被拒绝候选重新注入，CUDA 继续暂缓。

**已完成：P3 recovery affordance executable/budget Gate 已通过。** `synthesize_recovery_rollouts()` 现在校验 action/motor 对齐、唯一 action-kind、motor alphabet 范围和 resource budget；从当前 affordance 生成的 action 先经过 world learner 编码/投影，再进入 recovery branch 过滤与 risk planner。三 seed 真实 Gate 均通过：被拒绝 `assemble` 不重新注入，超预算 `archive` 不进入候选，不在当前 motor capability 中的 affordance 不进入候选，`idle` deterministic 分支仍被实际环境执行并完成 terminal recovery；checkpoint、ledger evidence 与 adjudication trace 保持一致。核心 mypy 0、Ruff/Black 和相关 P7/P3 回归全绿。该 Gate 证明 runtime 可执行性与资源边界，不宣称 affordance 生成已达到开放世界完整性。

**当前唯一下一步：建立真实 environment capability discovery 与 recovery affordance freshness Gate。** 让环境在每个 step 返回的可用 action/motor capability 参与下一轮 recovery synthesis，并用 capability 变化、过期 affordance、schema 新增和 checkpoint continuation 验证旧候选不会越权或复用；CUDA 继续暂缓。

**已完成：P3 environment capability discovery 与 recovery synthesis freshness Gate 已通过。** `EnvironmentOutcome` 现在可携带当前 step 的 `available_actions/action_kinds`，adapter 将其收敛为可 checkpoint 的 `EnvironmentCapability`，并在 `begin_episode()` 清除旧 episode 能力；恢复候选在未显式传入能力时只消费当前快照，显式能力与快照不一致时 fail-closed，同时要求 capability tick 与当前 world tick 对齐。三 seed 真实 environment Gate 验证了：能力边界被发现并持久化、checkpoint continuation 保留能力、step 后能力刷新会约束下一轮候选，`archive` 超预算和不在能力边界的候选均不会生成；报告 `reports/taiji_p3_risk_sensitive_execution_20260827.json` 的 cross-seed gate rate=`1.0`，相关定向回归 `27 passed`，核心 mypy=`0`、Ruff/Black 全绿。该 Gate 证明当前能力快照的发现、恢复与下一轮 synthesis 约束，不宣称已完成对已持有 pending rollout 的执行时重验证。

**当前唯一下一步：建立 pending recovery rollout 的执行时 capability/affordance freshness Gate。** 为每个 synthesized rollout 保存 capability tick、affordance identity 与 world-schema revision；在 `plan_rollouts()` 和真正执行前重新校验这些 lineage，环境能力、affordance 或 schema 变化时自动失效旧候选并要求重新 synthesis，CUDA 继续暂缓。

**已完成：P3 pending recovery rollout capability/schema freshness Gate 已通过。** 新增 `RecoveryRolloutLineage`，为 Taiji-owned recovery rollout 保存 capability tick、完整 action/action-kind 快照、affordance identity 和 world-schema revision；`plan_rollouts()` 会过滤或拒绝过期候选，真正执行前再次 fail-closed，过期候选不会触发环境动作。候选 synthesis 先统一完成本批 schema 注册，再为整批 rollout 记录同一最终 revision，避免生成未知 action kind 时把同批候选误判为过期；planned rollout、lineage 和 capability 均通过 native checkpoint 恢复。三 seed Gate 验证 planning-time rejection、execution-time rejection、lineage checkpoint continuation 与 terminal recovery 全部为真，报告 `reports/taiji_p3_risk_sensitive_execution_20260827.json` 的 cross-seed gate rate=`1.0`；定向回归 `27 passed`，核心 mypy=`0`、Ruff/Black 全绿。该 Gate 证明 pending rollout 的边界版本控制，不宣称同一 affordance ID 下参数内容变化已具备内容指纹校验。

**当前唯一下一步：建立 affordance content identity 与 schema-bound action validation Gate。** 为 affordance 生成稳定的内容指纹（action kind、actor/target、规范化参数和 grounding lineage），并将其写入 recovery lineage；在 planning/执行前校验当前 affordance 内容及 action semantic key，任何同 ID 内容替换、action 参数漂移或 schema 编码不一致都必须失效旧候选并重新 synthesis，CUDA 继续暂缓。

**已完成：P3 affordance content identity 与 schema-bound action validation Gate 已通过。** `WorldAffordance.content_identity` 基于 action kind、actor/target、规范化参数和 grounding lineage 生成稳定指纹；`RecoveryRolloutLineage` 同时保存该指纹与第一步 action semantic key。`plan_rollouts()` 和真实执行前都会校验 capability 快照、affordance 内容、schema revision 及 action symbol 映射，同 ID 内容替换、action 参数漂移、schema/action semantic key 不一致均 fail-closed，且不会触发环境动作。三 seed 风险执行 Gate 的 lineage 记录、checkpoint continuation、stale planning、stale execution 全部为真，cross-seed gate rate=`1.0`；相关回归 `27 passed`，核心 mypy=`0`、Ruff/Black 全绿。该 Gate 证明恢复候选的内容身份和执行边界可审计，不宣称已对所有非 recovery planner 候选建立同等 lineage 约束。

**当前唯一下一步：将 freshness contract 扩展到多步 recovery rollout 的逐步 affordance revalidation。** 每次真实 step 成功后重新绑定下一步的 affordance/content identity、capability snapshot 与 schema revision；若下一步候选不再存在或环境边界变化，自动截断旧 suffix、保留 recovery branch 并要求重新 synthesis，而不是只在下一次执行入口才发现失效，CUDA 继续暂缓。

**已完成：P3 multi-step recovery suffix revalidation Gate 已通过。** 非终止成功 step 后，adapter 不再原样保留旧 suffix，而是以 post-action world tick 重新预测 suffix，并重绑下一步的 capability snapshot、affordance content identity、action semantic key 与 schema revision；若任一边界已失效，则当场截断 suffix、保留 recovery branch 并要求重新 synthesis。三 seed 真实两步 environment Gate 验证了第一步成功后的 suffix rebind、第二步 terminal recovery、两次真实 evidence 写入及 checkpoint lineage continuation，cross-seed gate rate=`1.0`；相关回归 `27 passed`，核心 mypy=`0`、Ruff/Black 全绿。该 Gate 证明多步恢复不会盲目复用旧预测 suffix，不宣称未来任意 action 序列都能在当前 capability 未知时安全预绑定。

**当前唯一下一步：建立 recovery branch 的动态再规划与预算累积 Gate。** 将已消费 step 的 resource cost、剩余预算、失败/拒绝次数和 branch lineage 作为持久状态；每次 suffix 失效后由 Taiji-owned synthesis 在剩余预算内重新生成候选，禁止通过 checkpoint 或重复 rebind 绕过累计预算，CUDA 继续暂缓。

**已完成：P3 recovery branch dynamic replanning 与 cumulative budget Gate 已通过。** `RecoveryBranchState` 现在持久化总预算、已消费资源、failure/rejection counters，并提供剩余预算；恢复 synthesis 首次绑定 branch budget，之后只按剩余预算过滤候选。真实执行的 recovery step 会累加实际 candidate resource cost，环境失败与 ledger rejection 分开记账，checkpoint continuation 保持这些累计状态，不能通过重绑定或重新 synthesis 复原预算。三 seed Gate 验证了 checkpoint budget preservation、成功 step 消费 `0.2`、剩余 `0.8` 时 `0.9` 的 secure 候选被阻断、rejection/failure accounting 与 terminal recovery 全部通过，cross-seed gate rate=`1.0`；相关回归 `27 passed`，核心 mypy=`0`、Ruff/Black 全绿。该 Gate 证明 recovery branch 的资源与风险记账闭环，不宣称已实现跨多个并行 branch 的全局资源仲裁。

**当前唯一下一步：建立并行 recovery branch 的全局预算与公平选择 Gate。** 当同一失败上下文产生多个候选 branch 时，维护 branch-owned 与 episode-global 两级预算，按风险/进度/资源做可审计仲裁；验证多个 checkpoint continuation、branch 淘汰和失败重试不会重复消费全局资源，CUDA 继续暂缓。

**已完成：P3 branch-owned + episode-global recovery budget Gate 已通过。** 新增可 checkpoint 的 `RecoveryBudgetState` 作为 episode-global ledger，与 `RecoveryBranchState` 的 branch budget/consumption 分层；每个 recovery step 使用唯一 action identity 幂等扣费，checkpoint 恢复和重复 replay 不会重复消费；失败和 ledger rejection counters 分开累计。Taiji-owned synthesis 同时受 branch 剩余预算和 global 剩余预算约束，三 seed 真实 Gate 验证了 parallel candidate 中高成本分支在累计消耗后被阻断、global ledger checkpoint preservation、duplicate consumption blocking、failure/rejection accounting 和 terminal recovery 全部通过，cross-seed gate rate=`1.0`；相关回归 `27 passed`，核心 mypy=`0`、Ruff/Black 全绿。该 Gate 证明两级资源账本与幂等扣费，不宣称多个并行 branch 已作为独立持久对象同时运行。

**当前唯一下一步：建立持久化 recovery branch portfolio 与公平仲裁 Gate。** 将同一失败上下文的多个候选从一次性 tuple 提升为带 branch identity、状态（active/selected/pruned/expired）、风险/进度/资源审计的 portfolio；checkpoint 续跑后恢复全部候选状态，按全局预算公平选择并可淘汰 branch，禁止被淘汰候选重新注入，CUDA 继续暂缓。

**已完成：P3 持久化 recovery branch portfolio 与公平仲裁 Gate 已通过。** 新增 `RecoveryPortfolio`，为同一失败上下文的候选保存唯一 generation/branch identity、完整 imagined rollout、`active/selected/pruned/expired` 状态、revision 和 retired branch 集合；Taiji-owned synthesis 每轮生成唯一候选 ID，重新合成会退休上一轮 ID，防止旧候选伪装成新候选回流。adapter 在 planning 前只向 risk planner 暴露当前 active/selected 分支，选择后持久化唯一 selected 状态，显式淘汰分支不会重新注入；多步 suffix rebind 会更新 portfolio 中对应候选。普通 checkpoint 与 native checkpoint 都恢复 portfolio 全量候选、状态和 lineage。三 seed 真实 Gate 的 `portfolio_selection_audited`、`portfolio_pruned_not_reintroduced`、`checkpoint_portfolio_preserved` 全部为真，cross-seed gate rate=`1.0`；报告/manifest 为 `reports/taiji_p3_risk_sensitive_execution_20260827.json` 与 `reports/taiji_p3_risk_sensitive_execution_manifest_20260827.json`，相关回归 `4 passed`，核心 mypy=`0`、Black 全绿。该 Gate 证明并行候选的状态 ownership、选择、淘汰和 checkpoint 恢复，不宣称跨 episode 的长期 portfolio archive 或大规模 branch scheduler 已完成。

**当前唯一下一步：建立跨 episode recovery portfolio archive 与 branch liveness Gate。** 在不污染新 episode transient 的前提下，持久保留可复用的候选 lineage/结果摘要，定义 branch 的 completed/abandoned/expired 生命周期与容量淘汰；验证 episode 切换、checkpoint continuation 和长期预算边界不会复活已终止或已淘汰 branch，CUDA 继续暂缓。

**已完成：P3 跨 episode recovery portfolio archive 与 branch liveness Gate 已通过。** 新增有容量上限的 `RecoveryPortfolioArchive` 和不可执行的 `RecoveryArchiveEntry`：archive 只保存 source episode、portfolio/rollout identity、action lineage、capability/affordance 摘要、resource cost、outcome 与 `completed/abandoned/expired` 生命周期，不把候选重新暴露为可执行对象。terminal 成功 branch 记录为 `completed`，episode 切换清除当前 portfolio/capability 但保留 archive；archive 采用配置化容量并淘汰最旧条目，planner 对 archive 中的 rollout ID fail-closed，不能跨 episode 复活旧 branch。普通/native checkpoint 同时恢复 archive 和生成序号，避免同 episode 重用 branch ID；当前仍执行的 selected suffix 在新一轮 synthesis 时保留到 portfolio，避免归档时丢失最终 outcome。三 seed 真实 Gate 的 archive lifecycle、checkpoint、capacity eviction、archived-branch rejection 和 episode transient isolation 全部为真，cross-seed gate rate=`1.0`；报告/manifest 为 `reports/taiji_p3_risk_sensitive_execution_20260827.json` 与 `reports/taiji_p3_risk_sensitive_execution_manifest_20260827.json`，相关回归 `4 passed`，核心 mypy=`0`、Black 全绿。该 Gate 证明跨 episode 的 recovery 生命周期和不可复活边界，不宣称 archive 已形成可泛化的长期策略学习器。

**当前唯一下一步：建立 recovery outcome 到长期策略记忆的受控写入 Gate。** 仅允许 completed/有足够证据的 branch 摘要进入 Taiji 的长期 procedural/semantic memory，禁止 abandoned/expired 或冲突 outcome 污染策略记忆；验证跨 episode replay、证据阈值、checkpoint continuation 与错误策略撤销，CUDA 继续暂缓。
