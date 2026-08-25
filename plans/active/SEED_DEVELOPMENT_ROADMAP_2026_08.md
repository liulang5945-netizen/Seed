# Seed / Taiji Native v1 开发路线（2026-08）

状态：**当前唯一执行路线**

更新时间：2026-08-25（Taiji 架构重新定基线）

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

P2 尚未退出。首份无预测训练的 smoke 基线保存在
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
A1 仍未通过，完整 P2 不能因此退出。

扩展验证已完成：`dialogue16` 独立 manifest 的 slot gain / boundary consistency /
random binding drop 最小值为 `+0.9375 / 0.9841 / +0.6875`；`shared16` 独立
manifest 为 `+0.9219 / 0.9811 / +0.6406`。因此 relation subgate 已在两个语料
分区、16 atoms、240 个 ordered-pair 组合规模上稳定通过；旧 next-byte A1 仍保留为
失败历史对照，不阻塞结构性 P2 relation subgate 的收口。

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
- 本轮 native 回归为 `122 passed, 1 skipped`；命令显式排除两个受本机 Windows pytest 临时目录权限影响的旧 manifest 测试，
  环境状态不作为代码能力结论。

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

## 15. 停止项

在 P2 通过前：

- 不续跑旧 16M→100M raw-byte 长训；
- 不为 TSK-v8 继续增加认知补丁；
- 不写绑定固定 fan-in 的自定义 CUDA kernel；
- 不用增加神经元数量替代学习型抽象；
- 不删除 Legacy 对照；
- 不把旧 N/M 通过记录宣传为完整智能进展。

## 16. 当前唯一下一步

**下一步：把 `ExecutiveDecision` 接入真实 `WorldAction → TaijiEnvironment → Outcome` 闭环，让每次实际结果更新 executive
utility，并验证 selected/alternative、失败重规划、checkpoint continuation 和 executive lesion；禁止让聊天 decoder 代替环境反馈。**
