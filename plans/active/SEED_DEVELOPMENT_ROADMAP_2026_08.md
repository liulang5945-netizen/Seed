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
  `LanguageOrgan` 只接收 Taiji-owned `ExpressionPlan`，产品默认的 `native-readable` 表层保留有效候选或生成诚实的可读状态文本；
  `structured-stub` 降为显式无损调试 codec。detached-organ lesion、native checkpoint 和参数/认知不变性均通过。该结果修复产品
  乱码/RAW 冒充语言的边界，但只证明可读表层，不等于自然语言流畅性、开放域语义回答或 decoder 智能。
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
  `generate_input()` 走同一输入合同，并在产品出口经本地 `native-readable` 表层形成可读文本；raw-byte 只保留为底层兼容/调试信息。
  该 Gate 证明输入所有权与可读输出边界，不证明 executive、开放域语义对话或语言智能。
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

### 13.1 桌面客户端 UX 修复轮（2026-08-27）

实测澄清的运行形态：桌面端（`desktop/main.py`，PyQt6 无边框窗）= 子进程 uvicorn `api.app:app`(8000，同时服务 REST 与 `frontend/dist` 静态前端) + 子进程 WS 服务器(8765)；聊天走 Seed 原生运行时（`checkpoints/seed_corpus.pt`，**0.51 M 可学习权重 / 960 神经元式单元**，详见 §13.3.1 规模勘误；底层仍为 byte predictor，本轮当时的语言器官是 `structured-stub`，产品表层已在后续 P6 Gate 改为 `native-readable`）。本轮十项修复：

| # | 问题 | 根因 | 修复 |
|---|---|---|---|
| 1 | 外框边框不跟主题 | 标题栏 QSS 只在加载后同步一次 | `desktop/main.py` 1s 轮询 `data-theme`，变化才重设 QSS |
| 2 | 进入页面弹「已刷新」 | `AgentConfigView.onActivated` 调带 toast 的刷新 | 自动刷新静默化，仅手动点击提示 |
| 3 | 页面切换生硬 | router-view 无过渡 | `App.vue` 增加 `route` 过渡（out-in，220ms，reduced-motion 降级）。**⚠ 本项引入白屏回归，已在 §13.3 推翻重做** |
| 4 | IDE 无法唤起系统文件管理器 | Web 沙箱无原生对话框 | 后端 `POST /api/workspace/pick_folder`（PowerShell STA BrowseForFolder）+ 前端「浏览系统目录」。**⚠ 仅解决"选得到"，对话框仍弹在主窗后面，见 §13.3** |
| 5 | IDE 简陋 / 终端不可用 | 终端 WS 在 auth 关闭时默认拒绝 | 终端默认放行（与全局 JWT 中间件一致，可配置收紧）；新增 Ctrl+\`、Ctrl+P 快速打开、新建文件夹、刷新树、「在资源管理器中显示」(`/api/workspace/reveal`)、`/api/workspace/mkdir`。**⚠ "默认放行"是局域网免鉴权 shell 漏洞，已在 §13.3 收紧为对端地址感知** |
| 6 | 侧边栏搜索右侧不明符号 | macOS 专用 `⌘K` 硬编码 | 平台感知提示（Win/Linux: `Ctrl K`），并真正绑定 Ctrl+K 聚焦 |
| 7 | 「你好」回复乱码 | **模型真实输出**：0.51 M 权重的 byte 级基底 + raw prediction 未经过语言器官；旧 `structured-stub` 只会做无损结构序列化，不会形成可读语言 | 本轮先做诚实呈现：后端 final 事件标注 `readable`（U+FFFD/控制符占比启发式），前端以「RAW 原始字节输出」卡片呈现而非伪装成正常回复，历史消息同启发式。**根治已在 P6 语言表层 Gate 落地**（见 §16）：聊天路径构造 Taiji-owned `ExpressionPlan`，先经本地 `native-readable` 表层，有效候选保留、不可读 prediction 转为诚实可读状态文本，final event 暴露 `language_backend`，前端只在真正不可读时显示 RAW 调试卡片 |
| 8 | 输入栏按钮「没用」 | 按钮实际可用（Chromium 实测全通过）；体感来自发送按钮 disabled 且无反馈 | 发送门控保留但移除 disabled，点击未就绪时 toast 明确原因（连接中/模型未装载/生成中） |
| 9 | 生命状态数据来源存疑 | needs 数据源是 Cortex legacy `life_scheduler`；Seed 运行时下后端返回空（无假数据） | `LifeStatusView` 增显式 DATA SOURCE 说明卡；`is_seed` 透传至前端；Seed 下生命活动按钮给真实提示 |
| 10 | 对话页面无法上下滑动 | `.chat-stage` 为 `flex:1; min-height:0`，在 flex 列滚动容器中被压缩到小于内容高度；内容以 `overflow:visible` 溢出绘制，但父级 `scrollHeight` 仍按 stage 盒子计算 ⇒ 滚动条永不出现，内容被 sticky 输入栏遮挡 | `.chat-stage` 改为 `flex:1 0 auto`（可涨不可缩，去掉 `min-height:0`）；`.composer-wrap` 加 `flex:none; z-index:2`；`.msg` 的 `contain-intrinsic-size` 由 80px 提到 140px 以减少 `scrollHeight` 失真 |

滚动修复的实测证据（Chromium，注入内容后量测）：修复前 h=610 时 `stageScroll 433 > stageBox 397` 而 `saScroll === saClient(558)`、滚轮无效；修复后 `stageBox === stageScroll(449)`、`saScroll 610 > saClient 558`、滚轮生效；12 条真实消息场景 `saScroll 1565`、可滚到底且末条消息 bottom(522) < 输入栏 top(538) 不被遮挡。

配套：OpenAPI 基线快照已更新（新增 3 个 workspace 端点）；vitest 160/160、e2e 冒烟 22/22 通过；`frontend/dist` 已重建。

遗留（下一轮候选）：native-readable 已解决产品乱码与 structured-stub 误用，但它不是开放域语言模型；下一轮需为 Taiji-owned `ExpressionPlan` 建立真实语言表达训练/holdout Gate。终端默认 shell 仍是 cmd.exe；侧边栏搜索框尚未接线为会话过滤。

### 13.2 打包链收敛与客户端重打包（2026-08-27）

上一轮十项修复提交后，用户实测反馈「没有重新构建前端和打包客户端」。核查结论一分为二：`frontend/dist` 确已重建（构建产物含全部修复），但 `dist/Seed/Seed.exe` 仍是 08-24 17:40 的旧包（45.43 MB），内置 `index.html` 哈希与源码构建不一致 ⇒ 客户端里跑的是三天前、不含任何修复的前端。

**机制层根因**：存在两条重叠的桌面打包入口，而被文档推荐的那条恰好缺少防漂移断言。

| 入口 | 独有能力 | 缺陷 |
|---|---|---|
| `scripts/release.py`（CONTRIBUTING 推荐） | 前端 + PyInstaller + NSIS 编排、产物验证 | **缺少**「源码 dist == 客户端内置 dist」字节断言；无旧产物清理 |
| `desktop/build.py` | 字节级前端一致性断言、dist/build 清理、运行时可写目录后处理 | 未被文档与 CI 之外的任何流程调用 |

按「机制演化时收敛、清理旧的」收敛为**唯一入口** `scripts/release.py`：

- 并入 `_verify_packaged_frontend()`——对比 `frontend/dist/index.html` 与 `dist/Seed/_internal/frontend/dist/index.html` 字节，把「改了前端却打出旧包」从静默漂移变成显式构建失败；在 PyInstaller 之后作硬门禁，并复用于 `_verify_artifacts()`。
- 并入 `clean_outputs()`（dist/build 清理，新增 `--no-clean` 供增量调试）与 `postprocess()`（随包复制 `knowledge_store/`、`user_data/`、`security/`，创建 `agent_workspace/`、`taiji_data/{feed,sleep,life,evolution}_data/`）。
- 步骤重编号为 [1/4]…[4/4]；`_verify_artifacts()` 修正为按 `seed.spec` 的 `COLLECT name="Seed"` 检查 `dist/Seed/{Seed,SeedBackend}.exe`（非 Windows 跳过 SeedBackend）。
- 删除 `desktop/build.py`；同步更新 `ci.yml` F05 步骤（不再 py_compile 已删文件）与 `seed.spec` 文档字符串。

顺带修掉三个会让完整发布必然失败的 NSIS 缺陷（`makensis` 的 `OutFile`/`File` 相对**工作目录** `desktop/` 解析，而非 .nsi 所在目录）：

| 缺陷 | 现象 | 修复 |
|---|---|---|
| `OutFile "SeedSetup.exe"` | 装机包落在 `desktop/`，而验证步骤查 `dist/SeedSetup.exe` ⇒ 永远失败 | `OutFile "..\dist\SeedSetup.exe"` |
| `File /r "dist\Seed\*.*"` | 去找不存在的 `desktop/dist/Seed`，与同文件 `..\icon.ico` 自相矛盾 | `File /r "..\dist\Seed\*.*"` |
| `APP_VERSION "1.6.0\"` 多余反斜杠 | 版本串被污染 | 源头在 `scripts/sync_version.py` 的 raw f-string `rf"...\""`（raw 串里 `\"` 会把反斜杠写进文件），改为单引号 f-string `rf'\g<1>{ver}"'`，杜绝再生 |

**重打包实测证据**（`python scripts/release.py --skip-nsis`，本机无 NSIS）：

| 指标 | 结果 |
|---|---|
| 流水线 | 清理 dist/build → 前端构建 → 一致性校验通过 → 后处理 → 产物验证通过，`Seed v1.6.0 构建完成`，1273.7 MB / 9111 文件 |
| `dist/Seed/Seed.exe` | 69.14 MB @ 2026-08-27 23:46:56（旧包 45.43 MB @ 08-24 17:40:58）|
| `dist/Seed/SeedBackend.exe` | 69.07 MB，同时间戳 |
| index.html 哈希 | 源码 == 打包 == `76E4B2B8…17BA`，**MATCH** |
| 打包内资产抽查 | CSS `flex:1 0 auto`、`contain-intrinsic-size:auto 140px`、JS `raw-output` 均命中；`seed_corpus.pt`、`tokenizer_contract.json`、`agent_workspace/`、`taiji_data/life_data/` 就位 |
| 冷启动冒烟 | `Seed` + `SeedBackend` 双进程存活，`GET /api/health` 200，`model_loaded:true`、`seed_active:true`、`security_middleware:true` |

冒烟返回的 `language_provider.backend_id = "structured-stub"` 再次确认 §13.1 第 7 项（乱码）的根因仍在语言器官，**下一步应做 P6 真实语言器官接入**，而非继续在 UI 侧修补。

附注：`seed.spec` 的 `_datas` 用 `if src.exists()` 软条件，`version.json` / `app_settings.json` 在仓库中本就不存在且无任何代码读取，被静默跳过属预期，不是本次打包缺陷。

### 13.3 白屏回归根治、原生对话框前台化、终端鉴权收紧与规模勘误（2026-08-28）

用户实测反馈四件事：① 各页面点着突然全变空白（最严重）；② IDE 能选文件了但仍拉不起系统文件管理器；③ 终端和文件各有两个重复按钮；④ 追问模型真实规模。前三项均是 §13.1 修复本身的回归或未彻底，按「机制演化时收敛、清理旧的」逐项推翻重做。

#### 13.3.1 模型规模勘误（口径统一）

`checkpoints/seed_corpus.pt` 是自研 `seed-native-v1` 格式，**不是** PyTorch `state_dict`；稀疏突触以 `pre_index`（拓扑，整型索引）+ `edge_weight`（权重）成对存储，直接 `sum(numel())` 会把拓扑当参数一并计入。

| 口径 | 数值 |
|---|---|
| **可学习权重** | **509,521 元素（≈0.51 M），40 个张量** |
| 拓扑索引 `pre_index` | 506,768（不是参数） |
| 其他状态量 | 13,539 |
| 张量元素合计 | 1,029,828 |
| 文件体积 | 3.95 MB |
| **神经元式单元** | **960** = 皮层 `[256,192,128]`=576 + `memory_units` 384 |
| 字符表 / 训练 tick | 257 / 4,800,000 |
| 语料 / 训练器 | `simple_zh_texts.jsonl`（1,394,775,610 B）/ `train_seed_corpus`，存档 2026-08-23T09:28:23Z |

**结论**：此前记载的「43.7 万参数」是把 `pre_index` 与 `edge_weight` 混算所致，准确数字是 **0.51 M 可学习权重**。这是微型类脑基质，不是 Transformer 量级 LLM——§13.1 第 7 项乱码属该规模下的预期行为。§13.1 相关表述已同步勘误。

#### 13.3.2 全页白屏：`out-in` + `keep-alive` + `:key` 三者互斥

> **诊断范围勘误（2026-08-28，§13.8）**：本节修掉的是**真实存在的过渡竞态**（`:key` 与 keep-alive 语义冲突、`delayedLeave` 持旧 vnode），这部分结论与收敛依然有效。但当时把它当作用户所报白屏的**唯一**根因，是**推理而非观测**——没有打开真实浏览器控制台看有无异常。用户随后二次上报同一现象，§13.8 用远程调试实测到真正的致命项是 `FileUploadQueue.vue` 把 emoji 字符串喂给 `<component :is>`，在 Blink 下抛 `InvalidCharacterError` 并摧毁整个 router-view 子树。两者是**不同层的两个缺陷**，本节不构成对用户所报白屏的完整解释。

根因在 §13.1 第 3 项引入的 `App.vue` 过渡结构 `<transition mode="out-in"> → <keep-alive> → <component :is :key="$route.path">`，三个因素叠加致命：

1. `:key="$route.path"` 强制每次导航销毁重建，**使基于组件 name 的 keep-alive 缓存永不命中**，且同一次更新里 `KeepAlive` 返回全新 vnode；
2. `mode="out-in"` 把 enter 阶段推迟到 leave 完成后，经由绑定**旧 vnode** 的 `delayedLeave` 回调触发；
3. 用户行为恰是快速连点切页 ⇒ 下一次导航在上一次过渡未结束时抵达。

结果：`delayedLeave` 持有旧 vnode，而 `:key` 已变的新元素拿不到 enter 钩子，**停留在 `.route-enter-from` 的 `opacity: 0`**——DOM 完整存在、只是全透明。这解释了为何白屏无任何报错、也不触发 `RouteErrorView`（后者渲染可见 UI）。时间线亦吻合：`b656ff5` 只改了 `App.vue`，两个 CSS 文件未动。

修法（不是加补丁，而是拆掉互斥前提）：

- 删除 `:key="$route.path"`——它与 keep-alive 语义冲突，是纯冗余；
- **彻底删除离场过渡 CSS**（`.route-leave-active` / `.route-leave-to`），只保留淡入。`out-in` 下 leave 因检测不到 CSS 过渡而同步结束，`delayedLeave` 竞态窗口归零；enter 即使被打断，元素也只是丢掉 class 回落到自然的 `opacity: 1`，**物理上不可能卡在透明态**。

排除过程（逐一实证否定）：`.router-wrapper` 重复声明（`index.css` 导入序 shell→app，且级联按属性生效，后者不含 `flex` 无法取消前者）、`views/ChatView.vue` 缺失（自查误报，实际在 `components/`，路由引用正确）、多根模板、chunk 加载失败（`npm run build` exit 0，7 个 chunk 齐全）、`animations.css`/`overrides.css` 冲突关键帧（零匹配）、`appStore.applyBgImage()`（只改背景图）、`product.css`（只改背景色）——全部排除后嫌疑完全收敛到过渡组合。

顺带收敛：`.router-wrapper` 从 `app.css` + `styles/shell.css` 两处重复声明合并到 `shell.css` 单一定义（合并前先把 `app.css` 独有的 `background: var(--bg)`、`min-height: 0` 迁入，避免静默丢样式），`app.css` 处留指向注释。

#### 13.3.3 原生目录对话框：不是没创建，是没有宿主窗口

§13.1 第 4 项的 `Shell.Application.BrowseForFolder(0, ...)` 传 **hwnd = 0（无归属窗口）**。实证探针显示子进程阻塞 6.1 秒并生成 Explorer iconcache 临时文件 ⇒ **对话框确实被创建了**，只是拿不到前台激活，弹在无边框 Qt 主窗**后面**，用户完全看不见。诊断由此从「未创建」反转为「创建了但没前台化」。

修法：改用 WinForms `FolderBrowserDialog`（BIF_NEWDIALOGSTYLE 可缩放树），并先创建一个 `TopMost=$true`、`Opacity=0`、1×1、`ShowInTaskbar=$false` 的宿主窗体，`Show()` + `Activate()` 后以它为 owner 调 `ShowDialog($owner)`，用完 `Close()`/`Dispose()`。同时移除 `-NonInteractive`（本调用的全部目的就是展示交互式 UI），保留 `-STA`（COM 对话框需单线程套间）。

可行性交叉验证：`desktop/main.py:601` 仅设 `Qt.WindowType.Window | FramelessWindowHint`，**无 `WindowStaysOnTopHint`** ⇒ TopMost 宿主窗必然压在主窗之上。新脚本探针复测：脚本长度 693、8 秒超时未返回（对话框正常等待输入）+ iconcache 副作用，语法与行为均成立。

#### 13.3.4 重复按钮收敛（各留视觉层级最高的那个）

| 功能 | 保留 | 移除 | 理由 |
|---|---|---|---|
| 终端 | 顶栏 `终端` 按钮 | 右栏「快捷操作」分组内的 `quick-btn` | 顶栏项有 `active` 状态、与 运行/保存 同组、图标+文字，视觉层级最高；右栏那个是分组里唯一的孤立填充 |
| 目录 | 顶栏 `打开文件夹` | 空树状态里的 `切换目录` 按钮 | 顶栏常驻可见；空态按钮只在空态出现，改为指向顶栏的文案，避免死路 |

配套清理：右栏「快捷操作」分组整体删除（其唯一子项已移除）、`.quick-btn` 相关死 CSS 删除并留注释；`Terminal` 图标 import 保留（顶栏与文件图标映射 `sh: Terminal` 仍在用）。

#### 13.3.5 终端免鉴权漏洞：判定依据从配置项改为对端地址

§13.1 第 5 项把 `terminal_allow_unauthenticated` 默认为 `True`，前提写的是「默认 127.0.0.1」；但 README 推荐用 `SEED_HOST=0.0.0.0` 让手机连电脑（`901a8c5`），两者叠加**在局域网上暴露一个免鉴权 shell**。且 `_verify_ws_token` 的 docstring 写「默认不允许」，与代码相反。

修法上取上限更高的方案：**不读 `SEED_HOST` 之类的服务端声明，而是判定这条连接的真实对端地址**——绑定 `0.0.0.0` 时回环与局域网请求走同一个监听套接字，只看绑定值根本无法区分风险来源。新增 `_is_loopback_peer(ws)`，兼容 IPv6 回环 `::1` 与 IPv4-mapped `::ffff:127.0.0.1`，地址缺失（反向代理剥离）按不可信处理。策略变为：认证启用→必须有效 token；认证未启用→仅放行回环对端，非回环一律拒绝并给出「请先启用 JWT 认证」的日志。`terminal_allow_unauthenticated` 语义收窄为**只能收紧不能放宽**（置 false 时连本机也要求鉴权），无法再用来给局域网开后门。模块 docstring 与函数 docstring 同步勘误。

边界实测（9/9 正确）：`127.0.0.1`/`::1`/`::ffff:127.0.0.1`/`localhost` → 放行；`192.168.1.7`/`10.0.0.5`/`0.0.0.0`/`None`/`""` → 拒绝。

#### 13.3.6 验证与产物

| 项目 | 结果 |
|---|---|
| vitest | **19 文件 / 160 用例全通过** |
| `npm run build` | exit 0，7 个 view chunk 齐全（ChatView 1,001.84 kB），945 ms |
| 构建产物断言 | `route-leave` **0 次**（竞态窗口消失）、`route-enter` 3 次、`quick-btn` **0 次**（死码清除）、`.router-wrapper` **1 次**（两处收敛为一处）|
| 重新打包 | `Seed.exe` 69.14 MB、`SeedBackend.exe` 69.07 MB，均为 2026-08-28 01:06:11 |
| 内置前端一致性 | `frontend/dist/index.html` 与 `dist/Seed/_internal/frontend/dist/index.html` SHA256 同为 `DF4069E4…790D`，**MATCH** |
| 打包内 CSS 断言 | `route-leave=0`、`quick-btn=0`、`.router-wrapper=1`，与源码构建一致 |

`python scripts/release.py` 在本机以 exit 1 结束，但**打包主体成功**：前端一致性字节门禁通过两次、PyInstaller 报告 `Build complete!`、后处理已复制 `user_data/` 与 `security/`。失败只在最后 `_verify_artifacts()` 检查 `dist/SeedSetup.exe`——本机没有 `makensis`，NSIS 步骤被跳过而验证仍要求安装包。

> **本条已过时（2026-08-28，§13.8）**：当时的处置是「本机执行必须加 `--skip-nsis`」，即用人的记忆绕过脚本缺陷；该缺陷已在 §13.8 修掉——`build_nsis()` 改为回传「本机是否真的编译出安装包」这一事实供验证消费，因此现在**不需要任何标志**，`python scripts/release.py` 在无 makensis 的机器上也会如实以 exit 0 结束。

改动文件（6 个）：`frontend/src/App.vue`、`frontend/src/assets/app.css`、`frontend/src/assets/styles/shell.css`、`frontend/src/views/WorkspaceView.vue`、`api/routes_agent_workspace.py`、`api/routes_terminal.py`。

**方法论沉淀**：本轮三个问题全部是「上一轮修复引入的新缺陷」，且两个的初诊都是错的（白屏一度归因 CSS 重复、对话框一度归因未创建）。有效手段是**实证否定**而非推理：CSS 用导入序+级联语义排除、对话框用子进程阻塞时长与文件系统副作用反证「已创建」、鉴权用穷举边界地址验证。凡涉及「看不见」的失败（透明元素、隐藏窗口），必须找到能观测的侧信道。

遗留：终端默认 shell 仍是 cmd.exe；侧边栏搜索框尚未接线为会话过滤；P6 真实语言器官接入仍是消除乱码的唯一根治路径。

### 13.4 `ChatView` chunk 拆分与「假测试」收敛（2026-08-28）

起因是 §13.3 的遗留项「`ChatView` chunk 已达 1 MB 需拆分」。拆分本身顺利，但过程中**顺带查出三个一直存在、且被测试全绿掩盖的生产 bug**。这一轮的真实价值在后者。

#### 13.4.1 体积构成：唯一大头是 highlight.js 全量语法

`frontend/src/` 中只有 `composables/useMarkdown.js` 一处 `import hljs from 'highlight.js'`，该默认入口静态注册全部语法。实测 `highlight.js` 目录构成（此前笔记里的「384 种语言」是勘误）：

| 事实 | 数值 |
|---|---|
| 真实语法数 | **192** |
| 名称总数（含别名） | 371（其中纯别名 179） |
| `es/languages/` 文件数 | 384 = 192 真实语法 + 192 个 `<name>.js.js` 兼容 shim |
| 单个语法极端体积 | `mathematica` 109,852 B（约 107 KB，此前白吃） |

`highlight.js` 的 `exports` 条件映射在 `import` 条件下会把 `./lib/core` 解析到 `es/core.js`、`./lib/languages/*` 解析到 `es/languages/*.js`，所以 bare specifier 可直接用于 ESM 按需加载。

#### 13.4.2 方案：core + 192 语法按需加载，而非静态挑选子集

选择上限更高的方案：只静态引入 `highlight.js/lib/core`，**192 种语法一个不减**，全部改为按需动态加载。被否决的方案是「静态注册十几种常用语法」——那是能力降级。

三个必须解决的技术前提，均以探针实测确认（探针用完即删）：

1. **别名要在加载前就能解析。** hljs 的别名（`py`→`python`）只在语法注册后才生效，而 fence 标记恰恰在注册前到达；未注册语言调 `hljs.highlight` 会**抛异常**。因此在构建期生成 `frontend/src/composables/hljsAliases.js`（179 条纯别名，3,318 B）做前置映射。
2. **`renderMarkdown` 不能变成 async。** 模板里是 `v-html="renderMarkdown(msg.content)"` 同步调用。解法：模块级 `grammarVersion = ref(0)`，`renderMarkdown` 内 `void grammarVersion.value` 读一次让 Vue 记为渲染依赖，语法到位后自增即触发重渲染 —— `ChatView.vue` **零改动**。
3. **shim 不能进产物。** 曾断言「用精确路径 `import(\`…/${name}.js\`)` 就不会展开 shim」，**实测被证伪**：Rollup 把 `${name}` 当 `[^/]*` 匹配，`python-repl.js` 同样命中，产出 192 个永不加载的死 chunk（54,201 B）。改用 `import.meta.glob` 的否定模式 `'!…/*.js.js'` 后归零；该 record 同时充当白名单，未知语言名可同步拒绝而不发起失败请求。

#### 13.4.3 三个被「假测试」掩盖的生产 bug

`src/__tests__/useMarkdown.test.js` 原本**复制了一份自己的 `parseMessageContent`**（注释写明 "Simplified … for testing core logic"），从不 import 真模块。11 个断言长期全绿，而真实模块同时坏着三处：

| bug | 现象 | 根因 |
|---|---|---|
| 代码块内容全丢 | 每个 fence 渲染成 `[object Object]`，语言标签恒为 `text` | marked v13+ 把 `renderer.code` 改为接收 **token 对象**，代码仍用 v12 的位置参数 `code(code, lang)`，模板插值把对象字符串化 |
| 答案标签残留 | 「思考过程：…\n最终答案：…」的正文开头留着「最终答案：」 | 清理正则 `/^(?:最终)?(?:回答\|答案)[：:]/` 缺少前导 `\s*`，而 lookahead 未消费的 `\n` 就在开头；且只认中文前缀，`Answer:`/`Final:` 完全没清 |
| 复制按钮点不动 | 代码块「📋 复制」按钮全程无响应 | 钩子用 `data-action="copy-code"`，但 `purifyConfig` 里 `ALLOW_DATA_ATTR: false`，DOMPurify 每次都把它剥掉，而事件委托正以该属性为选择器 |

三处均已修复。第三处按收敛原则**不放宽 `ALLOW_DATA_ATTR`**（那是有意的安全姿态），而是删掉多余标记、统一以已存活的 `.code-copy-btn` 类为锚点，取文本改走 `.code-block-wrapper > pre`。

测试文件重写为 import 真模块，用例 11 → 27，新增覆盖：token 对象渲染器（正文非 `[object Object]`、语言标签正确、无语言回落 `text`）、未加载语法路径的 HTML 转义、未知 fence 标记的安全降级、`<推理>` 中文标签、中文标签不残留、markdown 标题分隔、`formatDuration`。**复制按钮的断言不测字符串而测契约**：把 HTML 塞进真实 DOM，验证委托选择器 `.code-copy-btn` 能选中、且 `.code-block-wrapper > pre` 可达。

#### 13.4.4 验证与产物

| 项目 | 结果 |
|---|---|
| `ChatView` chunk | **1,001.84 kB → 132.55 kB（−86.8%）**，gzip 44.03 kB；500 kB 警告消失 |
| chunk 总数 / 死 shim | 207 个 / **0 个**（修正前为 399 / 192） |
| 语法体外置校验 | ChatView 内 `LiveScript`/`Mathematica`/`PostgreSQL` 命中 **0** 次；对应语法各自成独立 chunk（`python` 3,258 B、`typescript` 7,590 B、`rust` 2,667 B、`x86asm` 19,007 B、`mathematica` 109,852 B）|
| vitest | **19 文件 / 175 用例全通过**（原 160，useMarkdown 由 11 假断言 → 27 真断言）|
| 重新打包 | `Seed.exe` 69.14 MB、`SeedBackend.exe` 69.07 MB（位于 `dist/Seed/`，**不在** `_internal/`），均为 2026-08-28 02:25:01 |
| 内置前端一致性 | `frontend/dist/index.html` 与 `dist/Seed/_internal/frontend/dist/index.html` SHA256 同为 `5BE51F49FE10…`，**MATCH**（release.py 内部亦自校两次）|
| 打包内产物断言 | 包内 `ChatView-C5zeGOa0.js` 129.45 KB、207 个 js chunk、`data-action` **0** 次 |

`scripts/release.py --skip-nsis` 以 exit 1 结束，但**构建完整成功**（输出 "Seed v1.6.0 构建完成"、总大小 1273.7 MB、前端一致性两次通过）。失败来自沙箱拦截 PyInstaller 分析阶段对 `Python312/Lib/**/__pycache__/*.pyc.<pid>` 临时文件的写入，与构建结果无关 —— 这是本机第二类已知的「假红」（第一类见 §13.3.6 的 NSIS 缺失）。**判定打包成功必须看产物本身，不能看 release.py 的退出码。**

改动文件（3 个）：`frontend/src/composables/useMarkdown.js`、`frontend/src/composables/hljsAliases.js`（新增，构建期生成）、`frontend/src/__tests__/useMarkdown.test.js`。提交 `e564029`（含本节 plans，4 文件 +445/−92）。构建产物 `frontend/dist/`、`dist/` 均在 `.gitignore` 内，不入库。

**方法论沉淀**：本轮最大教训不是体积，而是**「测试复制被测逻辑」等于零覆盖且伪装成满覆盖**——`[object Object]` 这种毁灭级 bug 与 160 全绿共存了很久。凡是 `__tests__` 里出现被测函数的本地副本（尤其带 "Simplified"/"for testing" 字样），一律视为门禁缺口。其次，本轮我两次把推断写进代码注释（shim 是否展开、`${name + '.js'}` 是否改变模式），两次都靠实测产物计数才被纠正：**注释里不能出现未实测的构建行为断言**。

遗留：终端默认 shell 仍是 cmd.exe；侧边栏搜索框尚未接线为会话过滤；P6 真实语言器官接入仍是消除乱码的唯一根治路径。

### 13.5 `hljsAliases.js` 再生成门禁（2026-08-28）

清偿 §13.4 的遗留项。别名表是构建期固化的静态产物，其生成器随探针一起删掉了：highlight.js 升级新增语言（如 `zig`）后，用户写 ```` ```zig ```` 会**静默退化成无高亮纯文本，而全部测试依然全绿**——正是 §13.4 刚付过学费的失效模式。

#### 13.5.1 四个必须先实测的前提

动手前用两个一次性探针把设计前提全部测出来，因为 §13.4 证明了「把推断写进代码」会被产物打脸：

| 实测结论 | 数据 | 对设计的约束 |
| --- | --- | --- |
| `highlight.js` 与 `highlight.js/lib/core` **是同一个单例** | `before=0 → afterFullImport=192`，`sameObj=true`（`lib/index.js` 对 `require('./core')` 注册 192 个语法；`lib/core.js:2589` 导出单例） | 门禁不能与 `useMarkdown.test.js` 同文件，且生成时必须用 `newInstance()` 隔离，否则会把 192 个语法注册进被测单例，让按需加载测试失去意义 |
| 别名存在**真实冲突** | `ls`: lasso vs livescript；`ml`: ocaml vs sml。上游 `registerAliases` 直接覆盖（后注册者胜），实测 `lasso@93 < livescript@100` → `ls`=LiveScript，`ocaml@124 < sml@162` → `ml`=SML | 必须解析 `lib/index.js` 复刻注册顺序 |
| 注册顺序**不是文件名字典序** | `REG_ORDER isSorted=false` | 现有表与字典序恰好吻合是**巧合**；自行排序在未来某次升级会静默产生错误归属 |
| `spec.name` 是**展示名而非键** | 几乎每个语法都不同（`1c` → `1C:Enterprise`），且 `python-repl` **根本没有 `name` 字段** | 只能以文件名为唯一标识，反查校验不可用展示名 |

行尾另需注意：无 `.gitattributes`、`core.autocrlf=true`、源文件磁盘上 100% CRLF，故生成器写 LF、`--check` 比较前归一化 CRLF→LF，避免制造全文件 diff。

#### 13.5.2 结构：生成逻辑只存在一份

`frontend/scripts/gen-hljs-aliases.mjs` 既是 CLI 也是模块，导出 `buildAliasMap`/`renderModule`/`listGrammarFiles`/`resolveHljsRoot`，由 `src/__tests__/hljsAliases.test.js` 直接 import。**测试不重算任何别名**——若在测试里重写一份「简化版」推导，就是 §13.4「假测试」的原样重犯。CLI 入口用 `import.meta.url === pathToFileURL(process.argv[1]).href` 守卫，保证被 vitest import 时不触发写盘。

新增 `npm run gen:aliases`（重生成）与 `npm run check:aliases`（`--check`，过期即 exit 1）。

#### 13.5.3 验证：门禁必须被证明能变红

| 验证项 | 结果 |
| --- | --- |
| 生成结果与既有表一致 | 重新生成后 `git diff` 仅 **2 insertions**（新增的「生成勿改」头注释），179 条别名逐字节复现 |
| 幂等 | 连续两次生成后 `--check` 均 exit 0 |
| **注入缺失别名** | 删掉 `yml` → 精确报 `missing: ['yml']`，红 |
| **注入错误归属** | `py: 'ruby'` → 报 `py: 文件=ruby 实际=python`，红 |
| 全量回归 | **20 文件 / 181 用例**全绿（原 19/175，新增 1 文件 6 用例）|
| 脚本未泄漏进前端产物 | `dist/assets/*.js` 中 `gen-hljs-aliases`/`node:fs`/`node:module` 均 **none**；唯一命中的 `registerLanguage(` 是 `useMarkdown.js` 的按需注册。ChatView 仍 132.55 kB |

**方法论沉淀**：一次红色验证比十次绿色更有信息量。首轮篡改时键集断言没红，我一度以为是漏判，实测发现是 PowerShell `Set-Content -NoNewline` 注入了 BOM 且使 CRLF 正则失效、`yml` 实际未被删除——**验证手段本身也会假**，改用 node 精确改写后立刻变红。因此「新增门禁」的完成标准不是它通过，而是它在人为破坏下必定失败。

改动文件（3 个）：`frontend/scripts/gen-hljs-aliases.mjs`（新增）、`frontend/src/__tests__/hljsAliases.test.js`（新增）、`frontend/package.json`（两个 script）；`frontend/src/composables/hljsAliases.js` 补头注释 2 行。临时探针 `probe_alias.mjs`/`probe_alias2.mjs` 已删除，不留残留。提交 `a2a4488`（含本节 plans，5 文件 +258/−2）。

### 13.6 把别名门禁接到发版必经路径（2026-08-28）

13.5 建成的门禁只在 `npm test` 时生效，而真正会出事的场景恰好绕过它：升级 highlight.js 后直接打包发版。门禁存在但不在关键路径上，等于不存在。§13.2 已把打包收敛到 `scripts/release.py` 这唯一入口，因此把检查插在它的构建之前。

`scripts/release.py` 新增 `check_generated_sources()`，执行 `npm run check:aliases`，失败即 `sys.exit(1)` 并打印修复命令。三个不显然的设计决策：

| 决策 | 理由 |
| --- | --- |
| **不受 `--skip-frontend` 影响** | 跳过的是构建，不是校验。`frontend/dist` 正是由这份可能已过期的源码产出的，跳过构建时打进安装包的 dist 同样过期。 |
| **置于 Step 0 清理之前** | 脱同步是必然中止的错误。若先清理再报错，会把上一版可用产物白删一遍。 |
| **`--check-only` 不跑该检查** | `--check-only` 的语义是「验证已有产物」，别名表属于源码而非产物。混进去会模糊两者职责。 |

顺带收敛：Windows 上 `npm` 实为 `npm.cmd`（直接调 `"npm"` 会 WinError 2）这一特例原本要在两处重复，抽成 `_npm()` helper。构建标号随之从 `[1/4]~[4/4]` 统一为 `[1/5]~[5/5]`。

实测（两条路径都验过，不止验绿）：

| 场景 | 结果 |
| --- | --- |
| 正常状态 | `[1/5] 生成式源码同步门禁` → `179 条别名一致`，放行进入后续步骤 |
| 删掉 `yml:` 一行后 `python scripts/release.py --skip-nsis` | `EXIT=1`，`hljsAliases.js 已过期` → `生成式源码已与依赖脱同步，构建中止`；**「清理旧产物」一行从未打印**（本次特意不加 `--no-clean`），证明产物未被删 |
| 恢复后 | `git status` 仅 `M scripts/release.py`，`check:aliases` 转绿，`--check-only` exit 0 |

改动文件（1 个）：`scripts/release.py`（+41/−11）。提交 `878316f`（含本节 plans 与 §14 修订，2 文件 +66/−12）。

### 13.7 解除 CI 下游 job 的 `needs: test` 挟持（2026-08-28）

本轮起点是一个**被推翻的假设**。上一轮收尾时我建议"把 `check:aliases` 与 `npm test` 接进 CI 前端 job"，动手前通读 `ci.yml`（373 行，不做局部读——`needs` 链局部读极易误判）才发现：`build-frontend` 第 204-206 行**早已有 `npx vitest run`**，而 `hljsAliases.test.js` 的断言里就含"磁盘文件与生成器输出逐字节一致（等价于 `--check`）"。即别名门禁自 `a2a4488` 起就已在 CI 生效。实跑 `npx vitest run` 确认收集到该文件（20 文件 / 181 测试全绿，含"别名键集合与当前 highlight.js 完全一致"）。**按收敛原则，不新增任何重复步骤。**

同时排除了第二个疑似风险：`highlight.js` 声明为 `^11.11.1`，但 CI 用 `npm ci` 按 lockfile 装（钉死 11.11.1，与本地 `node_modules` 一致），caret 并非静默漂移口子——真正刷新 lockfile 的时刻（`npm update` / dependabot）会让 vitest 那条断言当场变红，该路径已被封住。

真问题在依赖图上：`build-frontend` 与 `docker-build` 都挂着 `needs: test`，而 §14.8 已实测过其后果——`test` 连红 7 次期间这两个 job 一直是 `skipped` 而非 `failure`，**从未执行**。这与 §13.6 治的是同一个病（门禁不在必经路径上），病灶换到了 CI 的依赖图里：一个只改前端的 PR，若 Python 矩阵因无关原因（含 flaky）变红，eslint / vitest 别名门禁 / npm audit / E2E 全部静默失效一次。

解除依赖前逐条排除了耦合：

| 核查项 | 结论 |
| --- | --- |
| `build-frontend` 是否消费 `test` 的产物 | 否。10 个步骤全部自给（`npm ci` 起链） |
| `e2e/smoke.cjs` 是否需要后端 | 否。只依赖 `vite preview`（`BASE_URL` 默认 5173，CI 传 4173），不访问 `/api/*` |
| `docker-build` 是否消费 `test` 的产物 | 否。镜像构建自带完整依赖安装 |
| 其余 job 的写法 | `startup-smoke`、`test-windows` 本就无 `needs`——只有这两个挂着，不一致本身即线索 |

`docker-build` 一并解除的理由更强：§14.8 记载的两个缺陷（Dockerfile 缺 `data/` 目录、`seed_platform` 未随包安装导致启动 `ModuleNotFoundError`）都是它**独家**发现的，Python 测试矩阵抓不到。把一个具备独立发现能力的 job 挂在另一个 job 之后，等于让这份能力随上游一起失效。

验证（本地解析真实依赖图，而非只验 YAML 能否 parse）：

| 手段 | 结果 |
| --- | --- |
| `yaml.safe_load` 后枚举 `jobs` 的 `needs` | 5 个 job 全部 `needs = None`，依赖图扁平化，无悬空引用 |
| 同时输出各 job 步骤数 | 26 / 10 / 5 / 7 / 8，与改前一致——只删了 `needs` 行，未误伤步骤 |
| 枚举 `build-frontend` 十步的 `run`/`uses` | `npx vitest run` 在位，别名门禁执行点未动 |
| `npx vitest run` 实跑 | 20 文件 / 181 测试通过，`hljsAliases.test.js` 6 测试全绿 |

并发面变化：原先 `test`（2 矩阵）跑完才轮到 2 个下游，现在 5 个 job 立即并发，峰值 7（2+1+1+2+1），远低于公开仓库 20 的并发上限；副作用是反馈更快。

**本轮刻意未做**：`ci.yml` 既无 `concurrency` 也无 `timeout-minutes`（见 §14.14）。二者是真实欠账，但本轮意图是"解除门禁挟持"，混入并发治理会让这次提交不可审计。

改动文件（1 个）：`.github/workflows/ci.yml`（删 2 行 `needs: test`，加 8 行理由注释）。提交 `9dab2e5`（含本节 plans，2 文件 +53/−3）。

### 13.8 知识库白屏根治、jsdom-Blink 门禁与子进程内核级回收（2026-08-28）

本轮由四条客户端反馈驱动，其中一条是**同一现象的第二次上报**——"点击知识库后标签页内容全白屏，这个问题还是没解决"，并附带一条流程质问："为什么不启动开发者模式调试好了再打包"。后者是本轮最有价值的输入：§13.3.2 那次"白屏已修"是在没有真实浏览器控制台的前提下宣布的，用的是推理而非观测，所以修错了层。

**白屏真因（与 §13.3.2 的过渡动画完全无关）**：`FileUploadQueue.vue` 把 `icon` / `uploadIcon` 两个 prop 声明为 `String` 且默认值是 emoji（`📤`），又交给 `<component :is="uploadIcon">`。Vue 对字符串型 `is` 的处理是"解析不到组件就当原生标签",于是执行 `document.createElement('📤')`。Blink 对标签名做严格校验，emoji 不是合法标签名，**同步抛出 `InvalidCharacterError`**；该异常发生在 `keep-alive` / `router-view` 的渲染过程中，导致整棵子树被销毁——表现为点进知识库后**所有**路由都白屏，且不可恢复。修法是把 prop 类型改为 `[Object, Function]`、默认值换成 lucide 组件（`FileText` / `Upload`），并加 `asComponent()` 归一化 computed 兜住历史调用方。

**为什么 181 个前端测试全绿却放过了它**：`jsdom` 的 `createElement` 不校验标签名，`createElement('📤')` 在 jsdom 里合法，在 Blink 里抛异常。这是**环境差异造成的门禁盲区**，不是用例写少了。故新增 `frontend/src/__tests__/setup/blinkDom.js`，在 vitest `setupFiles` 里给 `Document.prototype.createElement` 打上 Blink 同级的标签名正则校验（`/^[A-Za-z][^\0\t\n\f\r >/]*$/`），不合法即抛 `InvalidCharacterError`。配套 4 个用例。**门禁必须能变红**：临时把修复回退，新用例当场以客户端里那条一模一样的错误失败；恢复后 185/185 全绿（181 → 185）。

**另三条反馈**：`WorkspaceView.vue` 去掉「项目文件」文字；托盘通知图标改为 `self.tray.icon()`（原先传 `MessageIcon.Information`，那是系统蓝色 i 图标，与 taiji logo 无关）；`MonacoEditor.vue` 的纯图标保存按钮确认与顶栏「运行/保存」功能重复，按 §13.3.4 同一原则删除视觉层级低的那个。

**验证方式改为真实观测**：QtWebEngine 不支持 Playwright 的 `connectOverCDP`，故用 `QTWEBENGINE_REMOTE_DEBUGGING=9222` + 裸 CDP over WebSocket 驱动。11 次路由跳转 + 3 个知识库标签页逐一断言 `routeError` 与容器内容长度：修复前 `nav-kb` 的 `len: 0`，修复后 `len: 205`，全程零异常零 console error。**这才是"调试好了再打包"该有的证据形态。**

**顺带暴露并根治的隐患：子进程在主进程被强杀后独活占用端口。** 排查白屏时两次遇到"代码改了但行为像旧的"，实测是上一轮的后端 worker（PID 20488、4944）仍在监听 8000，而就绪探测只看 `/api/health` 是否响应、不校验持有者是不是自己的子进程，于是静默接管了陈旧后端。WebSocket 服务（8765）也有同一现象（PID 11636）。清理路径 `_quit() → backend.stop()` 只在优雅退出时执行，强杀/崩溃时根本不跑。**不在 Python 层再加 `try/finally` 或 `atexit`（强杀时同样不执行）**，改用内核级 Windows Job Object：`JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE` 使 Job 句柄随主进程消亡时内核自动终止 Job 内全部子进程，与主进程如何死亡无关；三处 `Popen` 之后一律 `adopt_child()`。另加 `_reap_orphan_listener()` 处理"上一轮遗留的孤儿"，回收判定要求两个条件同时成立——映像名恰为 `SeedBackend.exe`（该名字只存在于本产品包里）且其父进程已不在系统中——避免误杀用户自己的服务或第二个客户端实例；5 个契约测试锁住这条边界（`tests/test_desktop_orphan_reap.py`，本仓库第一个 desktop 层 python 测试）。

**一次假警报及其教训**：强杀主进程后子进程存活，我一度判定 Job Object 未生效。升级排查（读日志 → `IsProcessInJob` 直查成员归属 → `QueryInformationJobObject` 回读 `LimitFlags` 确认 `0x00002000` 排除 ctypes 结构体错位 → 无 shell 中间层的隔离父子实验），逐项证明机制正确。真因是**我的验证方法有缺陷**：按命令行文本匹配挑进程，选中的是 shell 包装层，真正持有 Job 句柄的父进程（由子进程 `ParentProcessId` 反查得到）还活着。改按子进程反查父进程后复验通过。**记入方法论：进程身份不能靠命令行文本匹配确定，必须由子进程的 `ParentProcessId` 反向确认。**

**排查过程中发现并消除的潜伏缺陷**：`desktop/__init__.py` 里有 `from desktop.main import main`。`python -m desktop.main` 时 runpy 先导入 `desktop` 包 → `__init__` 把 `desktop.main` 载入 `sys.modules` → runpy 再把同一份源码作为 `__main__` 执行一遍。症状不只是日志 handler 装两遍导致每行重复（已观测），更严重的是**模块级全局出现两份副本**（包括 `_CHILD_JOB` 这个 Job 句柄本身，日志显示两次 armed），以及 `BackendManager` 类对象重复使 `isinstance` 失效。这正是它让上面那次假警报更难排查的原因。已删除该导入并在 docstring 里记录此陷阱；grep 确认主线无 `from desktop import main` 依赖，打包 spec 用脚本路径而非包导入；重启验证 runpy warning 消失、日志不再重复、Job 只 armed 一次。

**`scripts/release.py` 的自相矛盾**：`build_nsis()` 在 makensis 缺失时打印警告并 `return True`（判为非致命），而 `_verify_artifacts()` 仍按 `--skip-nsis` 标志硬性要求 `dist/SeedSetup.exe` 存在。后果是无 NSIS 环境下一次完全健康的构建必然以「产物验证失败」收尾——这就是 §13.3.6 记的"第一类假红"，当时的处置是"记住要加 `--skip-nsis`"，属于用人的记忆绕过缺陷。本轮按收敛原则改为**事实回传**：`build_nsis()` 返回 `(是否可继续, 是否应产出安装包)`，验证消费后者而非猜标志；`--check-only` 走同一套判定以免两条路径对同一产物给出不同结论；并补 `_find_makensis()` 兼查 NSIS 默认安装位置（NSIS 安装器不写 PATH，只查 PATH 会把"已装"误判成"未装"）。

**仓库卫生**：`.gitignore` 只有 `.codex_tmp/`，匹配不到 `.codex/`（gitignore 无前缀通配语义），而后者含 29 张 QA 截图、若干 CDP 探针，以及**两份活跃 git worktree 副本**（`git worktree list` 确认），副本里有同名的 `plans/ tests/ scripts/ frontend/`，会被仓库级 Grep / ruff / vitest 一并扫到而使统计基数失真。已补 `.codex/` 为**独立一行**（不删 `.codex_tmp/`——两者无覆盖关系，删了会让它重新被跟踪）。worktree 属活跃工作树，须走 `git worktree remove` 而非直接删目录，本轮不动。

验证与产物：

| 手段 | 结果 |
| --- | --- |
| vitest 全量 | 185/185（原 181，新增 4 个 Blink DOM 用例） |
| 门禁变红验证 | 回退 `FileUploadQueue.vue` 修复 → 新用例以 `InvalidCharacterError` 失败 |
| 裸 CDP 实测 | 11 次路由跳转 + 3 个知识库标签页，`routeError` 全为 `null`，零 console error；`nav-kb` len 0 → 205 |
| `tests/test_desktop_orphan_reap.py` | 5 passed |
| Job Object 机制隔离验证 | `LimitFlags = 0x00002000`、`IsProcessInJob(mine) = True`、隔离父进程强杀后子进程 alive = False |
| **打包模式端到端强杀** | 强杀 `Seed.exe`(25308) → `SeedBackend.exe`(25044) alive = False，8000/8765 全部释放 |
| `python scripts/release.py --check-only` | 全绿（修复前必然报 `✗ dist/SeedSetup.exe 不存在`） |
| 打包产物 | `Seed.exe` 72,507,172 B / `SeedBackend.exe` 72,422,700 B，前端一致性校验通过 |

**本轮方法论沉淀**（三条，均已在本轮内被实测检验过）：

1. 宣布"UI 缺陷已修"之前必须有真实浏览器控制台的观测证据；推理修出的是另一个 bug。
2. 单元测试环境（jsdom）与生产环境（Blink）的能力差异本身是门禁盲区，发现一例就要把校验补进 setup 层，而不是只补一个用例。
3. 机制看起来"没生效"时，先怀疑验证手段。本轮"Job Object 失效"与 §13.5 "PowerShell 注入 BOM"是同一类错误的两次发作。

## 13.9 外壳边框收敛为"整体圆角卡片内嵌"、标签页"常驻 + 零动画 + URL 同步"（2026-08-28）

用户对照主流客户端（TRAE/Doubao）截图提出四个互相关联的质疑：(1) 主流客户端是一条外围边框整体包裹、标签页嵌入其中，而本应用是"顶部边框与下方边框分割、两段对不齐"；(2) 标签页切换不如主流客户端丝滑，像"刷新显示"；(3) 这是否也是白屏的原因；(4) 商用前端是不是不用 Vue 这类平台、自己直接写的。四个问题逐一回答并落地实现（用户已确认目标形态：整体圆角卡片内嵌 + 常驻/零动画/URL 同步）。

**(1) 边框分割与 Vue 无关，是"边框所有权"颠倒。** 主流客户端由 shell（外壳）持有唯一边框，内容视图只是填充；本应用反了过来——每个视图自己画 `border-bottom`，且 `view-header` 的 `max-width: 800px` 使"线"的宽度永远取决于各视图内容宽度，全局无法对齐。React/Svelte/原生 DOM 会同样出错，框架无关。**修法**：收敛到 `styles/shell.css` 单一真源——`.app-wrapper` 降级为窗口/背景宿主（保留 `appStore.applyBgImage()` 的挂载点，用 `--bg-base` 暗部营造"卡片内嵌"的亮度差），`.router-wrapper` 成为**全应用唯一外围边框**（`border + border-radius: var(--radius-lg) + box-shadow + margin`），`.sidebar` 变为无边框透明面板；全部 5 处 `.topbar`、3 处 `.tabs`、`.view-header` 的 `border-bottom` 全部移除，并附注释禁止回潮。同时消除三份重复定义的级联债：`.app-wrapper` 原先在 shell.css（flex）/ app.css（grid）/ product.css（background，且最后加载会盖掉亮度差）各一份，`prefers-reduced-motion` 有两份全局块，响应式断点 768/880 冲突——统一为 880 + 560 两级。

**(2) "刷新显示"的根源是动画过多而非少了动画。** 面板用 `v-if` 切换 = 卸载 → 重建 → 重跑 setup → 重取数 → 从 `opacity: 0` 淡入，滚动位置、展开状态、输入内容全丢。主流客户端标签切换是 0ms（VS Code/Chrome 皆如此）——"丝滑"指的就是瞬时，动画只会让它看起来像刷新。

**(3) 白屏的直接成因仍是一行确凿的渲染异常（§13.8 已根治）；但 `v-if` + 淡入确实制造了"白屏易感体质"。** 判别法：量 `container.innerHTML.length`，0 = DOM 被清空（真白屏），非 0 + 透明 = 动画卡住。两者结论完全相反，先量再猜。把面板改为常驻后，任何真实渲染错误都会立即以可见形态暴露，降低再误诊概率。

**(4) 商用客户端没有"不用 Vue"。** TRAE/Doubao、VS Code、Slack、Notion、Linear 全是 Electron + Web 技术栈；VS Code 工作台是手写 TS + 直接 DOM，但它的两个要点（单一自顶向下的布局真源、视图永不销毁）在本项目用 Vue 完整可达——本轮同时落地了这两点。

**实现**：新增 `composables/useTabs.js`（唯一实现，三视图复用）：`activeTab` 写入 `?tab=`（`router.replace` 防污染后退栈），URL → 状态（前进后退/深链/刷新保持），`onActivated` 应对 keep-alive 下 `onMounted` 只触发一次的问题，方向键/Home/End + roving tabindex + `aria-selected/aria-controls/role=tabpanel`（WAI-ARIA tablist 手动激活模式），无 vue-router 环境（单元测试）自动降级为纯状态模式。`KBView`（白屏案发地，三个 `v-if` 面板）`TrainingView`（四个面板）`AgentConfigView`（三个 `v-if` 面板，且原本连 `role="tab"` 都没有）全部改为 `display` 切换；AgentConfig 原先内联在 `@click` 的 `loadInstalled()/loadMarketplace()` 收敛为对 `activeTab` 的 watch，深链直达也能触发加载（比内联点击更高上限）。`AppSidebar.vue` 响应式升级：880px 以下压缩为 56px 图标轨道（`!important` 压内联 `width`，仅此能赢），560px 以下才隐藏。

验证与产物：

| 手段 | 结果 |
| --- | --- |
| vitest 全量 | 185/185 全绿 |
| eslint（改动的 4 个 vue/js 文件） | 0 error 0 warning（`--fix` 属性换行） |
| ruff check（根目录） | All checks passed |
| `npm run build` | ✓ built，无编译级遗漏 |
| 上线请求 | 5 个 `.topbar` + 3 个 `.tabs` + `.view-header` 边框全部收敛到 `.router-wrapper` 唯一外围边框 |

**实机观测（QtWebEngine 裸 CDP @9222，source 模式 `python -m desktop.main`）**：

| 测量项 | 结果 |
| --- | --- |
| 三个视图 9 次标签切换耗时 | 3–22ms，全部同帧内完成（DNS 语义上的 0ms；旧 v-if+fade 需等整帧动画） |
| 面板显隐 | 恒为「1 显示 + N 隐藏」，DOM 常驻（切换无白屏帧、无重建） |
| URL 同步 | 每次切换 `#/kb/train/agent` 后附 `?tab=`，前进后退/刷新可还原 |
| 深链直达 | `location.hash='#/train?tab=dataset'` → 面板直接是高亮「数据集」 |
| keep-alive 折返保持 | KB 选「检索配置」→ 切 agent → 折返 KB，标签仍为「检索配置」 |
| 边框形态 | `.router-wrapper` 恒为 `1px solid` + `19.2px` 圆角；`.topbar/.tabs/.view-header` 边框全为 0 |
| 异常监控 | 全程 `Runtime.exceptionThrown` 与 console error 零触发 |
| 截图（card-kb.png） | 外壳灰底 + 大圆角卡片内嵌 + sidebar 独立间隙，内外部无任何分割边框错位 |

## 13.10 标题栏所有权移交前端、系统通知署名收敛为 AppUserModelID（2026-08-28）

用户附四张截图提出两条诉求：(1) 系统通知左上角不是应用 logo，而是紫色占位方块加字面量 `Seed.exe`——"用正确的 logo，或者直接不给这个弹窗提示"；(2) 顶部栏与下方"还不是一体的"，参照图 3（TRAE/Doubao）与图 4（Codex）——顶部一条与主体是同一个连续平面。

**(1) 通知署名不由 QIcon 决定，改 icon 永远无效。** Windows 通知左上角的归属槽取的是**进程的 AppUserModelID**；未声明时系统回退到 exe 身份，于是渲染占位方块 + `Seed.exe`。§13.8 那次"改为 `self.tray.icon()`"只换了通知体内的图标，署名槽根本不在该 API 的作用域内，所以用户看到的问题原封不动。**修法采纳"两条都做"的高上限组合**：一是声明 `APP_USER_MODEL_ID = "Seed.Desktop.Shell"`，经 `ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID` 在**任何窗口创建之前**设置（`desktop/main.py` 与 `api/run_app.py` 两个入口各一处，就在 `QApplication` 构造前）；二是**彻底删除 `closeEvent` 里的 `tray_icon.showMessage(...)` 气泡**——"已最小化到托盘"这类信息价值极低，而托盘图标本身就是可见反馈，改由 tooltip「Seed — 双击图标恢复窗口」承载。两者叠加后，即便未来别处需要弹通知，署名也已经是正确的应用身份。

**(2) "不一体"的物理根因：两个渲染平面各自画自己的背景，永远拼不成一个面。** 旧实现的顶部条是 Qt 控件（`QWidget` + `QHBoxLayout` + `QLabel` + 三个 `QPushButton`，36px，QSS 上色），下方才是 `QWebEngineView`；两者是不同的绘制宿主，无论把颜色调得多接近，接缝处的抗锯齿、DPI 缩放取整和主题时序差都会显形——这也是 §13.1 那条"1s 轮询 `data-theme` 重设 QSS"补丁的存在理由，它本身就是这个错误架构的并发症。**修法是把标题栏所有权整体移交前端**：中央区域改为 `QWebEngineView` 独占，标题栏成为 DOM 的一部分，与 sidebar 共享同一个 `.app-wrapper` 背景宿主，且中间**不存在任何 border**——由于 `.sidebar` 本就透明地坐在 `.app-wrapper` 上，一个同样透明的 `.app-titlebar` 自动就是同一个平面，"一体化"从此是结构保证而非调色结果。

**实现**：

- **窗口控制桥**：新增 `_WindowBridge(QObject)`，以 `pyqtSlot` 暴露 `minimize` / `toggleMaximize` / `close` / `startDrag` / `isMaximized`，经 `QWebChannel` 注册为 `seedWindow`。拖拽走 `windowHandle().startSystemMove()`（系统级移动，比手算 `globalPos` 增量更跟手且不丢焦点）。
- **客户端库注入**：页面由 http 提供，无法引用 `qrc:`，故把 `:/qtwebchannel/qwebchannel.js` 读出后注册为 `QWebEngineScript`（`DocumentCreation` + `MainWorld`，`seed_qwebchannel` 幂等去重），前端拿到的是原生 `QWebChannel` 全局。
- **最大化状态通道**：`resizeEvent` / `changeEvent` → `_sync_window_state()` 把 `data-maximized` 写到 `document.documentElement`；CSS 用 `:root[data-maximized='true']` 抹掉圆角与边框，Vue 侧用 `MutationObserver` 切换按钮图标。单向数据流，无轮询。
- **外框所有权也一并下移**：`.app-wrapper` 接管 `border: 1px solid var(--border)` + `border-radius: 18px`，Qt 只保留 `QRegion` 圆角裁切（`WINDOW_RADIUS = 18`，与 CSS 同值）防白直角露出。
- **前端**：新增 `components/AppTitlebar.vue`（无 `<style>` 块，样式全部落在 `shell.css` 单一真源），`App.vue` 增 `.app-body` 与 `sidebarCollapsed`（持久化到 `taiji_sidebar_collapsed`）。标题栏刻意**不放搜索框**——`AppSidebar.vue` 已持有 `.search-field` 与全局 Ctrl/⌘K，再加一个就是第二套入口。

**旧机制清理（收敛而非叠加）**：`desktop/main.py` 与 `api/run_app.py` 两处各删除 `_titlebar_qss`、`_window_frame_qss`、`_apply_titlebar_theme`、`_sync_titlebar_theme`、`_build_titlebar`、`_titlebar_mouse_press`、`_titlebar_double_click` 共 7 个方法（约 120 行/处）、§13.1 的主题轮询定时器、以及随之失效的 `QVBoxLayout/QHBoxLayout/QLabel/QPushButton/QWidget` 导入。净机制数下降。另确认 `desktop/seed.spec` 打包的是 `desktop/main.py`（`run_app.py` 当时的文件头"打包环境"注释是错的，已在 §13.10.2 改掉），但后者仍是可运行入口，按"清理旧的以免残留干扰"原则同步收敛，否则就留下第二套标题栏；`seed.spec` 的 `hiddenimports` 补 `PyQt6.QtWebChannel`。

**实机观测（QtWebEngine 裸 CDP @9222，先调试后打包）**：

| 测量项 | 结果 |
| --- | --- |
| 桥与客户端库 | `hasQt/hasTransport/hasQWebChannel` 全 true；`objects: ["seedWindow"]`，五个 slot 全部可达 |
| 一体化度量 | `.app-titlebar` 背景 `rgba(0,0,0,0)`、`border-bottom: 0px none`、高 40；`gap_bar_to_body = 0` |
| 接缝取色 | 跨越标题栏下沿的 4 个采样点（y=35/40/42/47）**实际背景全为 `rgb(241,243,245)`**——同一个平面，无缝 |
| 侧边栏收起 | false→true→false 双向可用，`taiji_sidebar_collapsed` 正确持久化 |
| 最大化联动 | `data-maximized` true 时圆角 `0px`/边框 `0px`/2560×1392；还原为 false/`18px`/`1px`/1280×800 |
| 截图（topleft/topright 2× 裁切） | 左上「收起按钮 → 太极 logo → Seed/在线」同一背景连续过渡、零分隔线；右上三个窗口按钮直接坐在窗口背景上，下方才是内容卡片圆角——与参考图 3/4 结构一致 |
| ruff / vitest 边界 | `ruff check` All passed；`tests/seed/test_platform_boundary.py` 9 passed |

**顺带收敛的悬空引用**：`frontend/public/` 下 `logo.svg` / `favicon.svg` / `icons.svg` 三个文件在前几轮已被删除，但 `frontend/index.html:5` 仍在 `<link rel="icon" type="image/svg+xml" href="/logo.svg?v=ink-20260624">` 引用 `logo.svg`，构成一个每次加载都 404 的悬空引用。已删除该行——`favicon.ico`（同文件第 6 行，文件实际存在）单独就足以承担 favicon 职责，且 taiji logo 在应用内由 `logo-taiji-ink.jpg` 提供。

### 13.10.1 冻结产物复验（`python scripts/release.py`，2026-08-28）

上面那张表是**源码模式**的观测，不足以结案：本轮新增的 `PyQt6.QtWebChannel` 是运行时依赖，而 `qwebchannel.js` 是**编译进 Qt 的 qrc 资源、不是磁盘文件**，`Get-ChildItem` 永远找不到它；更要紧的是 `_inject_webchannel_client()` 读取失败只打一条 `logger.warning`（"前端标题栏将退化为无窗口控制"），`_set_windows_app_identity()` 同样只 warning——**两条都是静默降级**，应用照样启动、外观几乎正常。所以必须在真实 exe 里把资源读出来才算证明。

打包产物：`dist/Seed/Seed.exe` 69.2 MB + `dist/Seed/SeedBackend.exe` 69.1 MB。经 `QTWEBENGINE_REMOTE_DEBUGGING=9333` 启动后裸 CDP 复验：

| 复验项 | 结果 |
| --- | --- |
| 依赖收集 | `_internal/PyQt6/QtWebChannel.pyd`、`Qt6/bin/Qt6WebChannel.dll`、`Qt6WebChannelQuick.dll`、`Qt6/qml/QtWebChannel/webchannelquickplugin.dll` 四件齐备 |
| qrc 资源实读（充分条件） | 冻结进程内 `typeof window.QWebChannel === 'function'` 为 true，`window.qt.webChannelTransport` 存在，`Object.keys(channel.objects) === ["seedWindow"]` |
| 一体化度量 | 与源码模式**逐项相同**：标题栏 `rgba(0,0,0,0)` / `border-bottom: 0px none` / 高 40 / `gap_bar_to_body = 0`；wrapper `1px solid rgb(231,234,239)` + `18px` |
| 接缝取色 | y=30/38/40 命中 `.titlebar-drag`、y=42/50 命中 `.sidebar-header`，五点背景**全为 `rgba(0,0,0,0)`**，统一落在 `.app-wrapper` 的 `rgb(241,243,245)` 上——纵向连续，无缝 |
| 最大化往返 | `false/18px/1px/1280×800` → `true/0px/0px/2560×1392` → 还原完全一致 |
| 冻结日志负向证据 | `dist/Seed/logs/desktop_main.log` 无 `qwebchannel.js 资源读取失败`、无 `AppUserModelID 设置失败`，两条降级分支均未走到 |
| 子进程生命周期 | `Stop-Process Seed` 后 `SeedBackend` 同步消失，job object 的 kill-on-close 在打包产物中同样生效 |
| 截图（full/topleft/topright） | 整窗一张外框，标题栏与 sidebar 同底、零分隔线，白色对话区圆角内嵌——与参考图 3/4 形态一致 |

**AUMID 的验证边界（记录方法论，避免下次误判）**：Win32 **没有**读取其他进程 explicit AUMID 的 API。`SHGetPropertyStoreForWindow` + `PKEY_AppUserModel_ID` 读的是**窗口级**属性存储，而 `SetCurrentProcessExplicitAppUserModelID` 设的是**进程级**值——实测目标窗口（`hwnd=2950018`，标题 `Seed - AI 生命体`）返回 `VT_EMPTY`，这是**预期结果**，不代表修复失效，通知系统会回退到进程级值。因此改用三条合证：(1) 同段代码在进程内 set/read-back，`E_FAIL` → `S_OK` → `'Seed.Desktop.Shell'`，机制有效；(2) 调用点在 `QApplication(sys.argv)` 构造**之前**（`desktop/main.py` L773 紧邻 L774），满足"任何窗口创建前"的时序要求；(3) 冻结日志无失败 warning。另：`HKCU:\...\Notifications\Settings` 下**不存在** `*Seed*` 项，这恰好侧面印证"直接不给弹窗提示"那一半生效了——`showMessage` 已删，进程从未发通知，系统自然不会建项。

**一个会反复踩的构建陷阱**：`python scripts/release.py 2>&1 | Tee-Object -FilePath build_release.log` 返回 exit 1，但日志尾部是 `Seed v1.6.0 构建完成`。原因是 PyInstaller 把全部 INFO 写 stderr，PowerShell 在管道中遇到原生命令写 stderr 会抛 `NativeCommandError`，**掩盖 python 的真实退出码**；日志里的 `ModuleNotFoundError: No module named 'tensorboard'` 也只是 PyInstaller 的可选导入探测，无害。构建是否成功的权威判据是脚本自己的 `python scripts/release.py --check-only`（本轮返回 0，含"前端一致性校验通过（源码 dist = 客户端内置 dist）"），而不是 shell 的 `$LASTEXITCODE`。

### 13.10.2 入口所有权收敛：消除"`run_app.py` 是打包入口"的错误共识（2026-08-28）

§13.10.1 里记了一句「`api/run_app.py` 文件头『打包环境』注释已过时」，本轮把它真正改掉。这不是措辞问题：两个文件头**互相印证**了一个反的事实——`api/run_app.py` 自称 `[打包入口] PyInstaller 桌面客户端`，`desktop/main.py` 自称 `[产品入口] … 开发环境版本` 并写着「api/run_app.py：打包环境」「未来计划：合并为一个入口，**以 api/run_app.py 为基础**」。任何人（包括我自己）照此改桌面行为，都会把改动落在一个**既不被打包、也不被版本同步覆盖**的文件上，然后打出旧行为的包。

判定入口身份用的是证据而不是注释：

| 证据 | 结论 |
| --- | --- |
| `desktop/seed.spec` L60 `a_main = Analysis([str(ROOT / "desktop" / "main.py")], …)` | 打包入口是 `desktop/main.py`，产物 `dist/Seed/Seed.exe`（spec 自己的文件头 L4 早就写对了） |
| `scripts/sync_version.py` 的同步清单 | 只覆盖 `frontend/package.json` / `desktop/installer.nsi` / `desktop/main.py` / `desktop/loading.html`，**没有 `api/run_app.py`** |
| grep `setApplicationVersion|SeedDesktop/|1\.\d+\.\d+` on `api/run_app.py` | 零命中。它连版本号都报不出来——真产品入口不可能如此 |
| glob `docs/ENTRYPOINTS.md` | **文件不存在**。两处文件头都在把读者指向一份不存在的文档 |

最后一条把问题性质升级了：这与上一轮删掉的 `logo.svg` 是同一类**悬空引用**，只是指向文档而非资源。已一并清除，仓库内（plans 外）`ENTRYPOINTS` 命中归零。

改法上没有写"以后再合并"这种会再次腐烂的承诺，而是各自写死当前事实：`desktop/main.py` 改为 `[唯一产品入口] … 开发与打包共用`，直接点名 `seed.spec` 的 `a_main` 与产物路径，顺手把功能描述校正到现状（标题栏由前端 DOM 承载、不发气泡通知、进程内 WebSocket、job object）；`api/run_app.py` 改为 `[历史入口·非打包]`，开头即**否定式断言**「**本文件不是打包入口。**」，并说明**保留理由**（依赖自检自动安装、`HotUpdateImporter` 热更新——这两项 main.py 没有），避免下次有人把"过时"误读为"可删"。

门禁与一处判断边界：`tests/seed/test_platform_boundary.py` 9 passed（该测试只断言 `run_app.py` 的 import 边界与 `CORE_DEPENDENCIES`/`transformers` 两个字面量不出现，文件头改写安全）、`ruff check` All passed、`py_compile` 0，并用 AST `get_docstring()` 反读确认两个 docstring 仍是模块首语句、内容正确。`black --check` 报这两个文件 `would reformat`——`git stash` 后复跑**基线同样 exit 1、同样这两个文件**，且 git 提示 `LF will be replaced by CRLF`，属既有换行符交互，非本轮引入，不扩大范围。

**遗留观察（本轮不动，记录以免丢失）**：(1) `test_desktop_entrypoint_keeps_transformer_dependencies_opt_in` 函数名断言的却是 `api/run_app.py`，是同一错误共识的命名残留；(2) 上述 black/CRLF 基线。（原第 (3) 条「本地与 `origin/main` 分叉」已由 §13.10.3 解决，故删除。）

### 13.10.3 分叉归零：桌面壳层三提交 rebase 到 P6 provider 四提交之上（2026-08-28）

`git stash` 时暴露出本地与 `origin/main` 分叉（本地 3 / 远端 4）。这是当时唯一的阻塞项，理由不是"分叉本身难看"，而是**远端那 4 个提交内容未知，一旦触碰 `desktop/` 或 `frontend/`，§13.10.1 的冻结产物验证就不再代表合并后的代码**——而那份验证是前两轮的全部结论依据。

**先按文件求交集再决定策略，不靠提交信息猜。** 提交标题（`provider registry rotation` 等）看起来与桌面无关，但"看起来无关"不是判据。用 `merge-base` 分别取两侧 `diff --name-only` 后做 `Compare-Object -IncludeEqual -ExcludeDifferent`：

| | 文件 |
| --- | --- |
| 本地 11 个 | `desktop/main.py`、`desktop/seed.spec`、`api/run_app.py`、`frontend/` 6 个、路线图 |
| 远端 11 个 | `seed/config.py`、`seed/language_provider.py`、`taiji/{__init__,adapter,language_organ}.py`、`api/seed_runtime.py`、2 个测试、`plans/README.md`、`scripts/training/…`、路线图 |
| **交集** | **只有路线图 1 个** |

且两侧在路线图内的 hunk 也不重叠：本地 `@@ -889,0 +890,74 @@`（§13.10 区），远端 `@@ -1346,3 +1346,22 @@`（§16 P6 区）。**远端 4 个提交完全不触碰 `desktop/` 与 `frontend/`**，因此冻结验证结论无需重新打包复验——这是本轮最重要的判定，它把"必须重跑 69 MB 打包"降为"跑静态门禁即可"。

**选 rebase 而非 merge**：本地 3 个提交是尚未共享的线性叙事（标题栏移交 → 冻结复验 → 入口收敛），三者有明确因果顺序；merge 会插入一个无信息量的 merge commit 并把这条因果链打散在图里。rebase 前先建 `backup/pre-rebase-20260828` 分支作为可回退锚点。结果：`Rebasing (1/3)(2/3)(3/3)` **零冲突**，`9fc7ecf/e7680c2/ccf0167` → `f82b169/0b59da2/ad47075`。

**合并后复验（不是"应该没问题"）**：

| 项 | 结果 |
| --- | --- |
| 路线图两侧内容并存 | §13.10.1/§13.10.2 在 922/943 行，远端 P6 provider 段在 1431-1439 行，共 1441 行，无一方被吞 |
| 远端文件完整落地 | `git diff --name-only origin/main HEAD` 在 push 前为 11（即本地三提交的改动），无远端文件丢失 |
| 联合测试 | `test_platform_boundary` + `test_language_provider_runtime` + `test_p6_language_organ_boundary` = **34 passed**（我方 9 + 远端 25，与远端提交声明的"25 passed"吻合） |
| 静态 | `ruff check desktop/ api/ seed/ taiji/` All passed；`py_compile` 0 |
| 跨层风险点 | 远端改了 `api/seed_runtime.py`（后端运行时，桌面壳层唯一可能被跨越隔离影响的地方），`importlib` 实导 `api.seed_runtime`/`seed.language_provider`/`taiji.language_organ` 三模块 → `backend_import_ok` |
| 产物一致性 | `python scripts/release.py --check-only` → 0，含"前端一致性校验通过（源码 dist = 客户端内置 dist）"，证明既有打包产物在合并态仍有效 |

`git push origin main` → `634c15a..ad47075`，`git status -sb` 回到无 ahead/behind 的 `## main...origin/main`。另：`git rebase` 与 `git push` 都再次触发 PowerShell 的 `NativeCommandError`（git 把进度写 stderr），真实 `$LASTEXITCODE` 均为 0——与 §13.10.1 记录的 PyInstaller 陷阱同一根因，**凡在 PowerShell 里判断原生命令成败，都必须看退出码而不是有无 stderr 输出**。

## 14. 持续门禁

- Taiji/Seed/Legacy 所有权 AST 测试；
- v1 state/checkpoint schema 和确定性恢复；
- TSK-v8 K 系列回归；
- 当前阶段 A Gate 的 holdout、lesion 和跨 seed 结果；
- 数据 manifest、实验注册、代码 commit 和训练 lineage；
- planned/actual learned state 与资源预算；
- 后端、前端、桌面、Legacy-off 启动和安全门禁；
- 构建期静态产物与上游依赖同步：`npm run check:aliases`（`hljsAliases.js` vs 当前 highlight.js，过期 exit 1）。该断言有两个执行点，且**两条都在必经路径上**——(1) CI `build-frontend` job 的 `npx vitest run`，经由 `src/__tests__/hljsAliases.test.js` 的逐字节一致断言（该 job 已于 §13.7 解除 `needs: test`，不再会被上游红隐藏为 skipped）；(2) `scripts/release.py` 的 `[1/5]` 前置步骤（不受 `--skip-frontend` 影响，失败即中止且不清理旧产物）。凡引入「由依赖推导、写死进源码」的产物，都必须同时引入这类门禁，**并且确认它在每条必经路径上都真的会执行**——只挂在测试里会被「升级依赖后直接打包」绕过；挂在测试里但那个 job 被 `needs:` 挟持，则连测试都不会跑。

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
- **结构性收口：`needs` 已删除，本条从「纪律」降级为「历史成因」（2026-08-28，详见 §13.7）**。上面那条纪律（"核对 job 集合是否都真的执行了"）是**依赖人记得去查**的补偿手段，属于下位对策。经核实 `build-frontend` / `docker-build` 均不消费 `test` 的任何产物（前者 10 步自 `npm ci` 起自给、`e2e/smoke.cjs` 只依赖 `vite preview` 不碰后端；后者镜像构建自带依赖安装），两处 `needs: test` 已删除，5 个 job 全部 `needs = None`。此后该失效模式**不可能再发生**，而非"要记得检查"。`skipped` 现象本身的描述仍然成立，保留作为成因记录。

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

### 14.14 CI 未设 `concurrency` 与 `timeout-minutes`（2026-08-28 记账，随后独立一轮已收敛）

**.github/workflows/ci.yml** 全文既无 `concurrency` 也无 `timeout-minutes`（grep 三个关键词零命中）。§13.7 删掉两处 `needs: test` 后 5 个 job 立即全并发，峰值 7 个（`test`×2 + `build-frontend` + `docker-build` + `startup-smoke`×2 + `test-windows`），低于公开仓库 20 的并发上限，故不构成资源争用。两项欠账引入后各自独立：

- 无 `concurrency` + `cancel-in-progress`：同一分支连续推送时旧 run 不取消，白烧额度；PR 迭代频繁时尤甚。
- 无 `timeout-minutes`：任一步骤挂死（`vite preview` 端口未就绪、compose 健康检查轮询、pip 解析）会跑满 runner 默认 6 小时上限才被杀。

**处理（commit `b11cb7f`，2026-08-28）**：
- 顶层加 `concurrency: group: ${{ github.ref }} + cancel-in-progress: true`——同一 ref 只保留最新 run，旧 run 被取消；不同分支按 ref 隔离不串扰。
- 按 run 33158773941 实测时长留 2–2.5 倍余量设逐 job timeout：`test`（两腿 495–697s）30m、`test-windows`（694s）30m、`docker-build`（139s）20m、`build-frontend`（108s）15m、`startup-smoke`（41–92s）15m。
- 行为验证：PyYAML 解析通过；合并远端 `2b81c0d` 后 run 33164390727 全部 7 job success。concurrency 是消除旧 run 浪费而非解决资源争用（峰值 7 < 上限 20），`cancel-in-progress` 对 main 连续推送语义正确。

未在记账轮一并处理是刻意的：记账轮的可审计意图是"解除门禁挟持"，把并发治理混进同一次提交会让改动动机不再单一，日后回溯无法判断某行是为哪个目标而改。此项已按此原则以独立提交收敛。

### 14.15 测试环境与生产环境的能力差异是门禁盲区（2026-08-28 收口后新增）

`jsdom` 的 `document.createElement` 不校验标签名，Blink 严格校验。同一行代码在 vitest 里通行、在客户端里抛 `InvalidCharacterError` 并摧毁整棵 `router-view` 子树——**181 个用例全绿而线上白屏**，根因不是用例写少了，是运行环境比生产宽松（详见 13.8）。

纪律：**发现一处环境宽松，就把校验补进 vitest `setupFiles` 层，而不是只补一个用例。** 当前 `frontend/src/__tests__/setup/blinkDom.js` 已把 `createElement` 收紧到 Blink 同级（`/^[A-Za-z][^\0\t\n\f\r >/]*$/`），在 `vite.config.js` 的 `test.setupFiles` 注册，对全部用例生效。后续若再遇同类差异（如 `URL` 解析、`ResizeObserver`、CSS 解析宽严不一），一律加到同一个 setup 模块内收敛，不要另起并列机制。

配套的可信度要求沿用 14.1：**新门禁必须被证明能变红**。本例的做法是临时回退业务修复，确认新用例以客户端里那条一模一样的错误失败，再恢复。

### 14.16 进程身份不能靠命令行文本匹配确定（2026-08-28 假警报后新增）

排查子进程回收时，我按命令行文本匹配挑"主进程"，选中的却是 shell 包装层，于是把一个**完全正确**的内核级机制误判为失效，白烧一轮排查（详见 13.8）。

纪律：**需要确认某进程是否为另一进程的父/子时，唯一可信来源是 `ParentProcessId` 反查，不是命令行文本、不是窗口标题、不是启动顺序。** 涉及端口占用时同理——`/api/health` 有响应只证明"有人在监听"，不证明"监听者是我起的"，必须用 `GetExtendedTcpTable` 拿到 owner PID 再比对进程树。

这与 14.7「本地绿 / CI 红」、13.5「PowerShell 注入 BOM」属同一类：**机制看起来没生效时，先怀疑验证手段本身。** 已连续三次发作，故单列成条。

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

**已完成：P3 recovery outcome 到长期策略记忆的受控写入 Gate 已通过。** 新增 `RecoveryStrategyLedger` 与 `RecoveryStrategyApproval`：terminal completed branch 只有在 `evidence_count` 达到配置阈值后，才按对应 episodic memory record 准入 recovery consolidation；abandoned/expired、低证据和非 terminal entry 都 fail-closed。adapter 新增 `consolidate_recovery_memory()`，只向已挂载的 semantic/procedural learner 提供 active approvals；普通/native checkpoint 恢复 archive、准入 ledger 与两个长期 learner 的 consolidation 状态，`revoke_recovery_strategy()` 使该 branch 后续 replay/consolidation 失效。P3 三 seed 真实 Gate 的 low-evidence blocking、strategy admission、两类 memory consolidation、checkpoint preservation 和 revocation 全部为真，cross-seed gate rate=`1.0`；相关回归 `23 passed`，核心 mypy=`0`、Black 全绿。必须保留的边界是：当前撤销阻止未来写入/回放，但尚未对已经写入的历史权重执行反向擦除。

**当前唯一下一步：建立可回滚的 recovery strategy consolidation Gate。** 为 procedural/semantic consolidation 保存可重建的 approved-record provenance 与版本快照；撤销策略后从未撤销记录重建长期读出并验证 checkpoint/重放不再携带已撤销 branch 的权重影响，CUDA 继续暂缓。

**已完成：P3 可回滚 recovery strategy consolidation Gate 已通过。** `RecoveryStrategyLedger` 保存 approved memory record provenance、证据计数、结果和 revoked branch 集合；撤销策略后，adapter 会从 episodic records 中排除对应 memory IDs，重建 procedural/semantic readers，而不是只切换一个布尔开关，并保存 consolidation 参数、rebuild 次数和 ledger 状态到普通/native checkpoint。三 seed 真实 Gate 验证了 low-evidence 阻断、approved-only consolidation、revocation 后 reader rebuild 以及 rebuild checkpoint continuation，cross-seed gate rate=`1.0`；相关回归 `23 passed`，核心 mypy=`0`、Black 全绿。当前边界是：重建保证撤销 record 不再参与后续读出生成，但尚未做大规模策略冲突下的精确影响分解或多版本并行 memory merge。

**当前唯一下一步：建立多策略冲突下的 recovery memory 竞争与撤销传播 Gate。** 在多个 completed strategy 同时准入时，按证据、结果一致性和资源预算进行 memory competition；撤销一个策略后验证相关 alias/sequence/semantic 影响传播到所有下游读出，不误伤仍有效策略，CUDA 继续暂缓。

**已完成：P3 多策略 recovery memory 竞争与撤销传播 Gate 已通过。** `RecoveryArchiveEntry`/`RecoveryStrategyApproval` 现在携带结果一致性与资源代价；`RecoveryStrategyLedger` 通过配置化的 evidence/consistency/resource 权重进行确定性排序，并按 `recovery_strategy_memory_budget` 选择不超预算且不重复绑定同一 episodic record 的策略。adapter 的 recovery consolidation 只读取 selected strategy records；撤销后重建会排除 revoked 与未选中的 recovery records，同时保留仍被选中的 survivor，semantic/procedural 两类长期读者和 native checkpoint 均验证了传播边界。新增 ledger 单测 `2 passed`，P3 风险执行 Gate 在 seeds `11/23/37` 全部通过，cross-seed gate rate=`1.0`；新增指标 `strategy_competition_selected`、`strategy_competition_checkpoint_preserved`、`strategy_revocation_preserves_survivor` 全部为真。为避免验证脚本对每个场景重复拟合，按 seed 缓存只读 baseline 并对每个场景 deep-copy，生产运行时逻辑未改变；核心 mypy=`0`，报告/manifest 已更新为 `reports/taiji_p3_risk_sensitive_execution_20260827.json` 与 `reports/taiji_p3_risk_sensitive_execution_manifest_20260827.json`，CUDA 继续暂缓。该 Gate 证明策略级竞争、预算裁剪和撤销后的幸存者保护，不宣称 sequence/concept/alias 等所有下游读者已经具备同等 provenance 反向擦除能力。

**当前唯一下一步：建立跨 reader 的 recovery provenance 与精确影响分解 Gate。** 将 selected strategy 的 provenance 继续贯穿 procedural sequence、concept/semantic alias 和 replay/readout 依赖，记录每个 reader 对策略的实际贡献；撤销单个策略时只重建受影响依赖，保留未受影响的 reader 状态，并验证多版本 checkpoint continuation，CUDA 继续暂缓。

**已完成：P3 跨 reader recovery provenance 与精确影响分解 Gate 已通过。** 新增 `RecoveryReaderDependency`/`RecoveryReaderDependencyGraph`，为 semantic/procedural/sequence/concept 四类 reader 记录 selected strategy rollout 与 episodic memory provenance，并持久化到普通/native checkpoint；adapter 新增可选 `ProceduralSequenceLearner` 挂载，recovery consolidation 统一向四类读者写入 selected records。撤销时按 dependency graph 只重建真正受影响的 reader，排除 revoked/未选 recovery records，保留未受影响依赖，并将 revoked/未选记录从 adapter episodic readout 隐藏。三 seed `11/23/37` 真实 Gate 的 `recovery_reader_dependencies_recorded`、`recovery_reader_checkpoint_preserved`、`recovery_reader_revocation_propagates` 全部为真，cross-seed gate rate=`1.0`；相关定向回归 `4 passed`，核心 mypy=`0`，py_compile 与 Ruff format 通过，报告/manifest 已更新。schema alias 仍是 world schema 身份而非 memory reader，因此不在策略撤销时删除；CUDA 继续暂缓。该 Gate 证明 recovery provenance 已贯穿当前真实挂载的 reader 与 adapter episodic readout，不宣称所有外部 reader/alias 已接入同一依赖图。

**当前唯一下一步：建立 recovery provenance 的 contribution attribution Gate。** 将每个 reader 对多个 selected strategy 的实际增量贡献从“输入依赖”细化为可重放的 per-strategy credit/weight delta，验证撤销任一策略只回滚对应增量，不重建无关 reader；继续覆盖普通/native checkpoint，CUDA 继续暂缓。

**已完成：P3 recovery provenance contribution attribution Gate 已通过。** 新增 `RecoveryReaderContribution`，以 deterministic leave-one-out replay 记录每个 selected strategy 对 semantic/procedural/sequence/concept reader 的 `effect_delta_l2`、归一化 `credit`、replay epochs/learning rate；每类 reader 同时保存 consolidation 前的 baseline checkpoint 及内容 digest。procedural/sequence reader 支持固定 action vocabulary 的 ablation replay，撤销时从保存的 baseline 只重放幸存策略，并重新计算 survivor attribution，不再重训普通历史记录。普通/native checkpoint 均恢复 baseline、贡献账本和 digest。三 seed `11/23/37` 的 contribution recording、checkpoint、selective revocation 全部为真，cross-seed gate rate=`1.0`；定向回归 `5 passed`，核心 mypy=`0`，py_compile、Ruff format 与 diff 检查通过，报告/manifest 已更新。该 Gate 的 credit 是 leave-one-out 边际影响，不宣称多个策略之间的交互影响已经被分解为可加和的 Shapley/线性权重；概念 reader 的 `effect_delta_l2` 是符号状态位移而非神经参数权重，CUDA 继续暂缓。

**当前唯一下一步：建立 interaction-aware recovery attribution Gate。** 针对多个 selected strategy 的非线性交互，增加可重放的 pairwise interaction residual 与顺序无关性校验，明确哪些影响可以安全相加、哪些必须保留为组合贡献；继续覆盖撤销、普通/native checkpoint，CUDA 继续暂缓。
**已完成：CI Python 门禁修复。** GitHub 失败运行 `33037813507`、`33037154061`、`33036706021` 的共同失败点均为 Black，而非测试、Ruff、启动冒烟或 Windows 任务；远端日志明确指出 `scripts/make_social_preview.py` 未格式化。本轮按 CI 固定版本 Black `26.5.1` 的 API 对全仓 463 个 Python 文件复核并修复，同时清除 Ruff 暴露的导入排序、`cache` 规则和嵌套条件问题；不降低 CI 规则、不触碰 CUDA。Ubuntu 等价门禁已通过：Ruff 两道检查、Black 0 个未格式化、mypy `44` 个源文件无错误、版本一致性通过；native `221 passed, 1 skipped`、Seed `72 passed`、全量 `465 passed, 5 skipped`，覆盖率 `40.83%`。本地 Black CLI 在 Windows 会挂起，因此采用同版本格式化 API 完成确定性校验；这属于本机工具异常，不改变仓库 CI。提交后唯一下一步仍是 interaction-aware recovery attribution Gate，CUDA 继续暂缓。

**已复核：CI 修复已在远端全绿。** `f8d54cc` 将 checkpoint digest 改为只依赖 PyTorch 的 byte view，消除 CI 未安装 NumPy 时 Python 3.10 的失败；GitHub Actions run `33067181142` 的 Python 3.10/3.12、Windows、启动冒烟、前端、Docker 全部通过，未放宽任何门禁。

**已复核：interaction-aware recovery attribution 提交已通过完整远端 CI。** 提交 `313d4cf` 的 GitHub Actions run `33069906564` 共 7 个 job 全部成功：启动冒烟（legacy/no-legacy）、Python 3.10、Python 3.12、Windows、前端构建和 Docker 构建均实际执行并通过；没有因上游失败而静默跳过下游门禁。

**已完成：P3 interaction-aware recovery attribution Gate 已通过。** 新增 `RecoveryReaderInteraction` 与 checkpoint 格式，为 semantic/procedural/sequence/concept reader 对每一对 selected strategy 记录同一 baseline 下的单体效果、pair effect、可加和基线、带符号交互 delta、非负 residual，以及 A→B/B→A 的 order delta 和 order-invariant 结果；交互账本随 reader dependency 进入普通/native checkpoint，撤销时仅保留幸存策略仍然成立的 pair。adapter 使用真实 reader checkpoint 重放，不引入 Transformer 或 CUDA 依赖；报告/manifest 新增 pairwise interaction 与 order-invariance controls。三 seed 风险执行 Gate 的 interaction recording、checkpoint continuation、撤销裁剪全部为真；定向回归 `6 passed`，完整 `tests/taiji_native` 为 `222 passed, 1 skipped`，核心 mypy `0`、Ruff/Black 全绿。这里的 residual 是 reader 状态的确定性 L2-like 组合效应审计，不冒充 Shapley，也不把交互测量自动当作新的权重；CUDA 继续暂缓。

**已完成：P3 interaction-aware recovery selection Gate 已通过。** `RecoveryStrategyLedger` 的 canonical selection 现在读取 checkpoint 化的 reader interaction audit：在 residual/order delta 均位于配置容差内且已完成 pair audit 时，策略保持独立竞争；明显非加和、顺序敏感或缺少 pair evidence 的关系 fail-closed 地并入 connected atomic selection unit，组合按成员最小 competition score/evidence、成员 resource cost 之和参与同一 memory budget，整组只能一起准入或一起被预算拒绝。选择结果、audit、容差与 reader audit-complete 标记均进入普通/native checkpoint；撤销后重新选择和 reader rebuild 使用同一 canonical policy，不会把已撤销策略或未审计关系静默当作可加和独立项。新增 atomic pair、未知 pair fail-closed、checkpoint/revocation selection assertions；P3 evaluator 三 seed cross-seed gate rate=`1.0`，selection/checkpoint/revoke 三项新增指标均为 `true`；定向回归 `8 passed`，native 回归（排除两个已确认的 Windows pytest 临时目录权限错误）`221 passed, 1 skipped`，Ruff、Black、mypy=`0`、compileall 全部通过。提交 `6a8327e` 的 GitHub Actions run `33079659762` 已完成，7 个 job（Python 3.10/3.12、Windows、双 startup smoke、frontend、Docker）全部成功。该 Gate 只证明当前 recovery memory 的交互约束与可回滚选择边界，不宣称已完成三层以上交互的精确联合 replay 或 Shapley 分解；CUDA 继续暂缓。

**已完成：P3 高阶 interaction-group replay Gate 已通过。** 对 pairwise audit graph 中的每个 connected 三策略以上组件，adapter 在同一 baseline 上真实重放完整 group、每个 singleton，并以 canonical/reverse 两种顺序复演；ledger 同时保存 group effect、additive effect、signed pairwise interaction sum、pairwise-predicted effect、高阶 delta/residual 和 order-invariance。高阶 residual、顺序敏感或缺少完整 pair evidence 的 group 会作为 atomic selection unit，整组共享 competition score/resource cost 并在 budget 下原子准入；普通/native checkpoint 恢复 group audit，撤销时整组删除并保留幸存者贡献 attribution。三 seed `11/23/37` 的 higher-order group replay、checkpoint preservation、atomic revocation 全部为真，cross-seed gate rate=`1.0`；定向回归 `10 passed`，native 回归（排除两个已确认的 Windows pytest 临时目录权限错误）`223 passed, 1 skipped`，核心 mypy=`0`、Ruff/Black、compileall 全部通过。该 Gate 证明三阶以上 group 的非线性审计和选择边界，不宣称已完成可组合 group 的增量 replay 或 Shapley 分解；CUDA 继续暂缓。

**已复核：高阶 interaction-group replay 已通过远端 CI。** 提交 `ed1aae1` 的 GitHub Actions run `33089028142` 共 7 个 job 全部成功：Python 3.10/3.12、Windows、双 startup smoke、frontend、Docker 均实际执行并通过；Node.js 20 弃用与既有 frontend lint 提示仍为非阻断注释，不影响本次 Taiji 门禁结论。

**已完成：CI 下游门禁的 `needs: test` 挟持已结构性解除（详见 13.7）。** 上面记载的"`build-frontend`/`docker-build` 在 `test` 连红 7 次期间一直是 skipped、从未运行"是**遮蔽机制本身**，当时只作为纪律（"核对 job 集合是否都真的执行了"）记账，没动依赖图；本轮删除两处 `needs: test`，`yaml.safe_load` 复核 5 个 job 全部 `needs = None`，步骤数 26/10/5/7/8 与改动前一致，证明未误伤任何步骤，该失效模式此后不可能再发生而非"要记得检查"。同轮否证了上一轮自己提出的建议——CI 并不缺别名门禁：`build-frontend` 的 `npx vitest run` 已收集 `hljsAliases.test.js`（本机实测 20 files / 181 passed），其断言与 `check:aliases` 逐字节等价，按收敛原则不新增重复步骤。`concurrency`/`timeout-minutes` 两项欠账刻意留到独立一轮（见 14.14），当前峰值 7 个并发 job 低于公开仓库 20 上限，不构成阻塞。提交 `9dab2e5`。

**当前唯一下一步：建立可组合 interaction-group 的增量 replay Gate。** 在两个已审计 group 合并、拆分或新增策略时，只重放受影响的 group 与 pairwise 边，验证高阶 residual、顺序不变性、预算原子性、checkpoint continuation 和局部撤销与全量重放一致；未受影响 group 必须保持 digest/attribution 不变，CUDA 继续暂缓。

**已完成：客户端白屏真因已根治，并补上了让它逃过门禁的那层盲区（详见 13.8）。** 上面 13.3.2 记的"白屏已修"是推理结论、修的是另一层缺陷；用户二次上报同一现象后改用真实观测（`QTWEBENGINE_REMOTE_DEBUGGING=9222` + 裸 CDP），实测真因是 `FileUploadQueue.vue` 把 emoji 字符串喂给 `<component :is>`，Blink 校验标签名时抛 `InvalidCharacterError` 摧毁整棵 `router-view` 子树，故点进知识库后所有路由都白屏。修法为 prop 改 `[Object, Function]` + lucide 默认组件 + `asComponent()` 归一化。181 个用例全绿却放过它是因为 jsdom 不校验标签名，已把 Blink 同级校验补进 `setupFiles`（`blinkDom.js`），回退修复可当场变红，套件 181 → 185。同轮另修三项客户端反馈（去「项目文件」文字、托盘通知改用 `self.tray.icon()`、删除与顶栏重复的图标保存按钮），并根治了两条基础设施缺陷：子进程在主进程被强杀后独活占用 8000/8765，改用内核级 Windows Job Object（`JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE` + 三处 `Popen` 后 `adopt_child()`）而非再加一层 Python `atexit`，**打包模式实测强杀 `Seed.exe` 后 `SeedBackend.exe` 同步消亡、两个端口全部释放**；`scripts/release.py` 的 NSIS 判定自相矛盾（非致命跳过却硬性要求安装包）改为事实回传，`--check-only` 全绿，13.3.6 那条"必须加 `--skip-nsis`"的记忆式绕过随之作废。另清除 `desktop/__init__.py` 的双重导入陷阱（曾使 Job 句柄出现两份副本），并把 `.codex/` 补进 `.gitignore`（含两份活跃 worktree 副本，会污染仓库级统计）。两条持久纪律已登记为 14.15 / 14.16。

**遗留欠账（前端路由级冒烟门禁，尚未落地）。** 本轮白屏能连着两轮逃过 185 个用例，是因为"整棵 `router-view` 被销毁"这一失败态没有任何自动化断言——`blinkDom.js` 只堵住了已知的 `createElement` 这一种触发方式，换个渠道（异步组件解析失败、`defineAsyncComponent` 无 `errorComponent`、子组件 setup 抛异常）同样会白屏而门禁全绿。待做：把本轮那套裸 CDP 脚本从一次性探针固化为受版本管理的门禁——headless 起前端 preview，逐个路由断言"容器内容长度 > 0 且 `window.onerror` / `console.error` 零命中"，任一路由为空即 `exit 1`；先在本机跑通并证明能变红（回退 `FileUploadQueue.vue` 应立即失败），再接入 `build-frontend` job 与 `scripts/release.py` 的必经路径。不新增第二套 E2E 框架，复用现有 vitest/preview 与已验证的 CDP 通道。此项与下面 Taiji 内核线并行排队，当前唯一下一步以本节末尾为准。

**已完成：P3 可组合 interaction-group 增量 replay Gate 已通过。** recovery consolidation 现在优先从 reader dependency 保存的稳定 baseline 重建，新增策略不会再次叠加已训练记录；pairwise audit 保存 replay action-kind fingerprint，group audit 保存 singleton effect、replay digest 和 attribution digest。增量路径只复用 baseline、成员、动作集合、参数和内部 pair audit 全部一致的 pair/group；新增策略、group 合并、group 拆分、局部撤销或审计变化只重放受影响边/组件，未受影响 group 的 digest 与 attribution 保持原值。三 seed 风险执行 Gate 的增量 replay 统计均为 pairwise `8 replay / 4 reuse`、group `4 replay / 0 reuse`，重复 consolidation 不发生 double replay；组合回归同时证明未受影响 group 稳定、变化 group 与全量重放相等，以及 merge/split replay 数量正确。相关回归 `10 passed`，native 回归（排除两个已确认的 Windows pytest 临时目录锁权限错误）`223 passed, 1 skipped`，核心 mypy `0`、Ruff/Black 全绿，CUDA 继续暂缓。该 Gate 证明 group 组合变化下的局部重放与 provenance 稳定性，不宣称 group 内成员 credit 已完成守恒分解。**

**当前唯一下一步：建立 interaction-group 的可验证 credit decomposition Gate。** 在不把高阶 residual 粗略平均给成员的前提下，为 group 建立基于可重放子集的成员增量 credit、交互 credit 和 residual 归属，验证 credit 守恒、顺序敏感时 fail-closed、局部撤销只移除相关归属，以及普通/native checkpoint continuation 与全量重算一致；CUDA 继续暂缓。

**已完成：P3 interaction-group credit decomposition Gate 已通过。** `RecoveryReaderInteractionGroup` 现在持久化成员单体子集增量、按稳定 pair 顺序排列的带符号交互 credit、独立归属的高阶 residual 和守恒误差；group effect 必须满足 `member increments + pair interaction credits + explicit residual` 的确定性守恒，不再把高阶 residual 平均摊给成员。完整归因随普通/native checkpoint 保存，增量 group 复用要求 credit decomposition 完整，旧格式或缺失子集证据自动重放；顺序敏感、守恒不安全或未完成归因的 group 在 selection 中 fail-closed 为原子单元。三 seed 真实 Gate 的 credit decomposition、非平均 residual、普通/native checkpoint continuation、局部撤销和变化 group 与全量重放一致性全部通过，cross-seed gate rate=`1.0`；相关回归 `12 passed`，核心 mypy=`0`、Ruff/Black 全绿。该 Gate 证明当前 interaction-group 的可审计 credit 守恒和安全边界，不宣称已实现 Shapley 或任意规模 group 的指数级全子集分解；CUDA 继续暂缓。

**当前唯一下一步：建立 interaction-group credit 的跨 reader 一致性与漂移 Gate。** 对 semantic/procedural/sequence/concept 四类 reader 比较同一策略组的 credit 结构、reader 状态漂移和 checkpoint 版本变化；当某一 reader 的 credit 结构变化时只失效该 reader 的 group attribution，保留其他 reader 与未受影响 group，CUDA 继续暂缓。

**已完成：P3 interaction-group credit 跨 reader 一致性与漂移 Gate 已通过。** 新增 `RecoveryReaderCreditConsistency` 与 `RecoveryReaderDependencyGraph.credit_consistency`，对同一策略组在 semantic/procedural/sequence/concept 四类 reader 的成员/交互/residual 分解建立 reader-independent 结构 digest，并将不同 reader 的状态尺度归一为 signed credit profile；原始 reader group replay digest 与 baseline checkpoint digest 作为状态漂移和 checkpoint 版本证据保存，不要求不同 reader 的原始数值相等。adapter 在每次 audit 后比较 coverage、结构、归一化 credit L1 漂移和 checkpoint/state digest 完整性；若单个 reader 的 group 结构或 credit profile 变化，仅将该 reader 的 group attribution fail-closed，未变化 reader 与其他 group 保持原对象和 replay 边界。普通/native checkpoint 均恢复一致性记录；真实三 seed 风险执行 Gate 的跨 reader consistency、checkpoint preservation、semantic-only drift isolation 全部为真，定向回归 `13 passed`，native 全量 `228 passed, 1 skipped`，另有 2 个既有 Windows pytest 临时目录锁权限 setup error 未进入测试体；Ruff、Black、mypy 全绿，CUDA 继续暂缓。该 Gate 证明的是当前四类 reader 的 group attribution 可比性、版本证据与局部失效边界，不宣称 reader 输出已经共享同一语义空间，也不宣称 credit 已达到 Shapley 或跨模态因果真值。

**当前唯一下一步：建立跨 reader credit consistency 的多组、跨 checkpoint 增量回滚 Gate。** 在多个 interaction group 同时存在时，为每个 group 保存独立 audit revision；验证 group 新增/合并/拆分、单 reader 漂移、checkpoint continuation 与局部回滚只更新受影响 audit，未受影响 group 的 structure/profile/state digest 保持稳定，CUDA 继续暂缓。

**已完成：P3 跨 reader credit consistency 多组、跨 checkpoint 增量回滚 Gate 已通过。** `RecoveryReaderCreditConsistency` 为每个 group 增加独立正整数 `audit_revision`；revision 只由该 group 的 reader 集合、结构 digest、normalized signed credit profile、base checkpoint digest 或 replay state digest 变化推进，不跟随 dependency graph 的全局 revision。真实 evaluator 覆盖两组基线、group 新增、A+B 合并、拆分恢复、semantic reader 单独漂移、native payload continuation 和局部回滚：未受影响 group revision/profile/state digest 保持稳定，受影响 group 依次从 `1 -> 2 -> 3`，单 reader 仍只被局部 fail-closed。定向回归 `15 passed`，三 seed cross-seed gate rate=`1.0`；native 回归 `229 passed, 1 skipped`，另有两个既有 Windows pytest 临时目录锁权限 setup error 未进入测试体；Ruff、Black、核心 mypy=`0`，CUDA 继续暂缓。

**当前唯一下一步：建立 cross-reader audit revision 的有限历史、回滚目标校验与容量淘汰 Gate。** 在不让旧 audit 重新成为可执行 attribution 的前提下，为 group 保存可验证的前序 revision 摘要；checkpoint 恢复后只允许回滚到存在且结构兼容的目标 revision，超过容量的历史不可复活，CUDA 继续暂缓。

**已完成：P3 cross-reader audit revision 有限历史、回滚目标校验与容量淘汰 Gate 已通过。** 新增 digest-only 的 `RecoveryReaderCreditAuditRevision`，每个 group 只保存 rollout 集合、reader/structure/profile/base/state digest 与完整性摘要，不保存 raw credit profile 或 `reader_attribution_safe`，因此历史记录不能重新变成可执行 attribution。`RecoveryReaderDependencyGraph` 现在按 group 保存有限 revision history，checkpoint/native payload 可往返恢复；回滚校验要求目标 revision 仍在容量窗口内，且 reader 集合、结构 digest、profile digest 和 base checkpoint digest 与当前 audit 兼容，缺失、篡改或结构漂移目标均 fail-closed。容量由 `recovery_strategy_cross_reader_credit_revision_history_limit` 配置，adapter 初始化、reset、restore 和每次 audit 持久化均使用同一上限；撤销策略时同步裁剪历史，合并/拆分留下的旧摘要仍不可执行。P3 evaluator 新增历史完整性、checkpoint continuation、目标 revision 校验和容量淘汰三项 Gate，三 seed 均为 `true`、cross-seed gate rate=`1.0`；定向回归 `15 passed`，native 回归 `229 passed, 1 skipped`，另有两个既有 Windows pytest 临时目录锁权限 setup error 未进入测试体；Ruff、Black、Taiji Mypy=`0`。本 Gate 只证明“可验证的有限审计历史与回滚 allowlist”，不宣称摘要本身能恢复神经状态或执行真实 rollback；CUDA 继续暂缓。

**已完成：P6 native-readable 产品语言表层 Gate。** 根因已确认：Seed 聊天虽然调用 Taiji 的 byte prediction，但直接把 raw bytes 当作答案，且默认 `structured-stub` 只能做无损结构序列化，不能形成可读语言。现在 `SeedRuntime.chat` 将 prediction 和本地会话上下文封装为 Taiji-owned `ExpressionPlan`，经过无外部依赖的 `native-readable` 语言表层；有效的 `surface_text/answer/native_prediction` 候选会被保留，不可读字节会转成诚实的可读状态文本。`structured-stub` 保留为显式 debug codec；Seed 配置升级为 v2，旧的未版本化 structured 默认会迁移到 native，显式 v2 structured 仍保持可用；native organ 已纳入 registry、checkpoint restore 和 `/api` final event 的 `language_backend` 可观测性。产品聊天默认不把用户历史静默转发给外部 decoder；Qwen/LoRA 仍是显式 provider 的表达器候选，不因此宣称 Taiji 已具备开放域语言智能。定向语言/provider 回归 `17 passed`，产品聊天冒烟 `4 passed`。

**已完成：P6 Taiji-owned `ExpressionPlan` 到真实语言表达的训练/holdout admission Gate。** 新增 `LanguageRealizationGate`，以
`LanguageTrainingCorpus` 为唯一监督边界，逐例验证 train/holdout 不串集、UTF-8/可读文本、必需语义词完整覆盖、无结构化泄漏、无
fallback，并要求 rollback reference 与保存后 checkpoint loader 的输出逐例一致。Qwen LoRA trainer 在写出 adapter/tokenizer 后重新加载
保存目录，再把 checkpoint continuation 纳入最终 Gate；真实本机 CPU 复核为 4 epochs/16 steps、270336 个外部参数，train/holdout
质量均为 `1.0`、rollback 与 continuation 均为 `true`。Seed 新增 `chat_enabled` 显式开关，且强制 `guarded` 模式；只有训练 realization
Gate 与 safety Gate 都通过才允许外部 decoder 进入产品聊天，旧报告或缺失证据 fail-closed，默认仍为本地 `native-readable`。相关定向回归
`20 passed`，核心 mypy=`0`、Ruff/Black 全绿；CUDA 继续暂缓。该 Gate 证明可审计的表达准入，不宣称开放域语言智能。

**已完成：P6 语言 provider artifact 内容寻址与首轮 chat canary Gate。** `LanguageProviderArtifact` 现在为文件或目录内容生成路径无关的
SHA-256 digest，并以 role 列表和稳定 manifest digest 绑定 base model、LoRA、训练语料、训练报告和安全报告；目录摘要只依赖相对 POSIX
路径、文件大小和字节，不依赖绝对路径、mtime 或遍历顺序。guarded product chat 在加载前严格要求五类内容摘要、manifest digest、固定
canary 合同和未过期 `expires_at`，逐项重新计算并拒绝缺失、替换、manifest 漂移和过期 artifact；旧 artifact/checkpoint 仍可读取，但没有
内容寻址证据时不能进入 product chat。provider 挂载后，`LanguageProviderCanaryGate` 对实际 language organ 执行两条固定语义表达，要求 UTF-8
可读、`数据库/正常` 与 `接口/恢复` 完整覆盖、无结构化泄漏且不触发 validated fallback；失败统一回退到 `native-readable`，并区分
`chat_artifact_missing`、`chat_artifact_drift`、`chat_artifact_expired`、`chat_canary_failed`。训练侧 artifact loader smoke 也已输出内容摘要和
canary 结果。相关定向回归 `23 passed`，Ruff、Black、核心 Mypy=`0`；本机 Seed/Taiji 全量测试的测试体未见本次回归，但仍受既有 Windows
worktree/pytest 临时目录 ACL setup/cleanup error 影响，未计为全量 Gate 通过；CUDA 继续暂缓。该 Gate 证明的是 provider 资产完整性和首次真实
表达准入，不宣称开放域语言智能或消除外部 decoder。

**已完成：P6 provider artifact 多版本 registry 与原子轮换 Gate。** `LanguageProviderArtifactRegistry` 只保存经过内容寻址的 immutable
manifest，按 artifact ID 去重，显式维护版本 allowlist、active/previous 指针和 monotonic revision，并通过 native checkpoint 保存与恢复；
未 allowlist 的版本、manifest 冲突、未知回退目标和 registry 指针漂移均 fail-closed。Seed 新增 `rotate_language_provider` 与
`SeedRuntime.rotate_language_provider`：候选 provider 在脱离线上 language organ 的 staging adapter 中加载，依次通过内容摘要、训练/安全报告和
首轮 chat canary 后，才以一个 `commit_language_provider_state` 操作同时发布 organ、backend registry、artifact 和新 registry snapshot；候选
失败时旧 provider、旧 runtime 和 active/previous 关系保持不变。定向语言/provider 回归 `25 passed`，Ruff、Black、核心 Mypy=`0`；提交后 CI
已复核全绿（Python 3.10/3.12、Windows、前端、Docker、启动冒烟），CUDA 继续暂缓。

**已完成：P6 provider runtime health watchdog 与自动回退 Gate。** 在 active artifact 已通过首轮 canary 的发布后运行时，增加请求级健康探针
`LanguageProviderHealthProbe`、连续失败阈值、有限冷却窗口和 previous-version 自动回退：`observe_language_provider` 把每次真实发射折叠进
`LanguageProviderHealthState`（可读表层、结构化泄漏、validated fallback 三项判据，异常/不可解码记失败），达阈值后 `auto_rollback_language_provider`
随 `now` 判定 nominal/冷却/回退；有 distinct previous 时回退到 previous 版本（`provider_health_rollback_previous`），隔离劣化版本并移出 allowlist，
冷却期内保持现状，无 previous 则落到 `native-readable` 且 `chat_enabled=False`（`provider_health_rollback_native`）。健康状态随 native
checkpoint 保存与恢复，重启后续接；探针与回退共用原子轮换路径（收敛而非叠加），且任何误报只能保持现状或回到 `native-readable`，不静默加载未
allowlist 的 artifact。Taiji/adapter/config/seed 四层实现，seed 层活体验证 9 项全绿，api 层 `SeedRuntime.chat` 请求级回退实测通过；定向回归
`18+96 passed`，Ruff 全绿、核心 Mypy=`0`；CUDA 继续暂缓。该 Gate 证明发布后劣化可被请求级吊销，不宣称开放域语言智能。

**2026-08-28 watchdog 收尾漏洞回扫（暂停推进期间）：** 按用户指示不向后推进、只回头核对前序 watchdog 推进里的真实漏洞并修复。共修 6 处——
A. **状态键集不一致**：`SeedRuntime._native_status()` 之前手写 9 键 dict，与 `LanguageProviderStatus.to_dict()` 的 14 键形状不符，原生/受管模式 status API 键集漂移；改为复用状态对象自身投影，单一事实来源。
B. **回退后配置残留**：回退到 native/structured 后 `_provider_config` 仍指向被降级版本；新增 `_sync_provider_config()` 在回退提交后清空/重锚配置，杜绝残留降级配置被后续观察误用。
C. **名义探针健康位不实时**：未触发回退的名义探针只更新 adapter 健康记录、不叠加进 status API；新增 `_overlay_health()` 让 status 实时反映健康负载而不翻转角色语义。
D. **重启复活隔离版本**：watchdog 隔离的版本会随重启被 config 重建而跳过 `require_allowed` 复活，违反复苏→劣化→回退死循环承诺；新增 `_registry_revokes_candidate` + `activate_language_provider` 拒活隔离版本，镜像/保留被隔离的持久 registry，覆盖 `provider_health_quarantined`。
E. **核心 mypy 漏报**：`taiji/language_organ.py` 对混合值类型 `metrics` dict 做 `>= 1` 排序（对 int|float|bool|str 联合不合法）遗漏 2 错；改为用局部标量比较。
F. **`_chat_organ` 类型**：`api/seed_runtime.py` 三处把 `LanguageOrgan` 赋给推断为 `NativeReadableTextLanguageOrgan` 的变量；显式标注为 `LanguageOrgan` 协议。
全量门禁复验：核心 mypy（`seed taiji`，`--follow-imports=silent`）`0`、Ruff 全绿、Black 无改动、全量 `498 passed / 5 skipped`。定位用诊断脚本见 `scripts/archive/diagnostics/diag_provider_health_audit.py`。

**当前唯一下一步（已收敛回主线）：** 回到 §16.1 的 Workbench Closure W0–W7。Taiji 已构造世界/计划/`ActionIntent`/`ToolCall`/`Outcome` 认知与
效应器合同，Seed 产品却仍缺一个 Taiji-native 的执行平面把这些合同接到 IDE、文件、终端、诊断和 MCP；watchdog、CUDA/fused kernel、新视觉打磨
等末端优化一律冻结，直到真实工作台纵切片（W0 起步：选定最小真实工具并打通认知→效应器→结果闭环）通过。

### 16.1 全盘审计后的路线校准：从研究 Gate 转向产品执行闭环（2026-08-28）

#### 16.1.1 审计结论

本轮按用户要求暂停功能开发，只核对 `main@6e2204b` 的真实代码、计划、API、前端和客户端链路。结论不是 Taiji 缺少一个 IDE
按钮，而是项目存在一条系统级断层：**Taiji 内已经构造出世界、计划、`ActionIntent`、`ToolCall` 和 `Outcome` 等认知/效应器合同，
Seed 产品却没有一个 Taiji-native 的执行平面把这些合同接到 IDE、文件、终端、LSP、诊断和 MCP。**

这解释了为什么研究 Gate 数量持续增加，客户端仍像若干互不相连的面板：当前 Taiji 能在模拟环境中证明工具闭环，语言 provider 能形成
可读文本，IDE 也能被人手操作，但三者没有共享同一 capability、权限、执行、结果和状态合同。继续沿 interaction-group attribution、provider
watchdog 或更多小型数值 Gate 纵深推进，会扩大内部证明数量，却不会关闭产品最关键的因果闭环，属于路径偏移。

因此主线立即重排为 **Workbench Closure W0–W7**。P1–P7 的既有成果保留为认知基础，不回滚；抽象 recovery attribution、provider
watchdog、CUDA/fused kernel 和新视觉打磨全部冻结，直到真实工作台纵切片通过。

#### 16.1.2 当前进度的分层事实

| 层 | 已完成事实 | 尚未完成/不能宣称 |
|---|---|---|
| Taiji cognition | P1–P7 已覆盖版本化状态、感知/世界/工作空间、记忆、规划、结构生长、生成、工具合同与大量 checkpoint/lesion Gate | 未证明开放域智能；大量 Gate 仍是小型数值/模拟环境 |
| 语言器官 | `native-readable`、外部 Qwen provider、训练/安全准入、内容寻址、registry 和原子轮换已落地 | provider 只负责表达，不拥有 IDE/工具权限；watchdog 尚未做且不再是当前瓶颈 |
| 产品运行时 | `SeedRuntime.chat()` 可走 Taiji 输入边界并返回可读文本；桌面壳、标题栏、托盘、构建和 CI 已收束 | 原生聊天没有 tool event、执行循环或 IDE after-state；不能自主完成代码任务 |
| 工作台 | Monaco、文件树、人工保存、Python run 和交互式终端 UI 已存在 | API 被归为 Legacy 可选路由；没有 Taiji-native capability registry、事务、审批、outcome 或自主语言选择 |
| 工具/MCP | NeuroPlex 路线有 ReAct、工具表、MCP 和插件历史实现；Taiji 有通用 `TaijiToolEnvironment` 协议 | 原生模式上报空工具列表；现有 MCP/Agent 路由没有接入 Taiji，部分前后端路径/参数还不一致 |
| 训练/发布 | Taiji native checkpoint、训练恢复、provider artifact 已存在 | 产品页仍把 GGUF、LoRA 合并和 Legacy 模型发布当作 Taiji 正式能力 |
| 前端产品口径 | provider 回退、运行时和错误中心已有一定可观测性 | 多处界面仍展示 TSK-v8 旧叙事、Cortex 热切换、Legacy life/Agent 配置和未实现能力 |
| 工程门禁 | 主分支与远端同步，最近 CI 已全绿；前端 185 tests，跨平台/容器/启动门禁已建立 | API/前端 capability 契约没有生成或一致性门禁；“界面有入口但后端不存在/原生模式不注册”仍可全绿 |

规模事实也支持“先收执行平面”的判断：当前仓库约 202 个 API route decorators，17 个 Seed/API/desktop 文件仍直接导入 NeuroPlex；
`taiji/adapter.py` 已达约 9300 行，Taiji native 有 85 个测试文件、68 个 eval 脚本和 253 个跟踪报告。项目不缺继续增加局部 Gate 的能力，
缺的是把这些能力变成一个真实、可观测、可撤销的产品纵切片。

#### 16.1.3 已确认的问题清单与根因

| 编号 | 代码证据 | 实际问题 | 根因分类 |
|---|---|---|---|
| G1 | `api/routes_chat.py::_seed_event_generator()` 只调用 `seed_runtime.chat()` 并返回 `final` 文本 | Seed 原生聊天不会生成/执行工作台工具事件 | 产品执行链缺失 |
| G2 | `taiji/adapter.py::generate_tool_call/execute_tool_call` 只被 Taiji 测试消费，`api/`/`seed/` 无调用者 | P6 工具合同停留在模拟环境，没有产品适配器 | research→product 断层 |
| G3 | `api/app.py::_register_routers()` 通过 `_load_optional_router()` 挂载 `routes_agent_workspace` | 关闭 Legacy 时，内置 IDE 的文件 API 一起消失 | 所有权分类错误 |
| G4 | `frontend/src/composables/useWorkspaceBridge.js` 全仓无调用者 | 文件打开、命令、错误回流只是注释承诺 | 死桥/假接线 |
| G5 | `MonacoEditor.vue` 用硬编码列表、扩展名表和组件内 `ref` 切换语言 | Taiji、后端和 checkpoint 不知道当前编程语言，也无法自主选择 | 状态只在 UI |
| G6 | 原生模式的 `runtime_service._tools_section()` 返回空列表，`runtimeStore.modelLifecycle` 却宣称可工具调用/自主探索 | 产品状态与真实 capability 相互矛盾 | 双重真相源 |
| G7 | `routes_agent_workspace.py` 的 run/create/analyze 仍导入 `neuroplex.agent_ext` | 即使界面可用，也不是 Taiji-native 工作台执行器 | Legacy 反向占位 |
| G8 | `AgentConfigView.vue` 使用 `/api/mcp/start/{id}`、`install/{id}` 等路径，后端要求 `/api/mcp/start` + JSON body；搜索参数也不一致 | MCP 面板存在可稳定复现的前后端合同漂移 | 无契约 Gate |
| G9 | TrainingView/useTraining/locales 展示 GGUF 导出和“合并 LoRA 权重”，后端只返回“Seed 不支持” | 旧 HF/GGUF 模型格式仍被呈现为正式产品操作 | 迁移残留 |
| G10 | `routes_settings.py`、`routes_models.py`、`seed_platform.config` 仍保存 GGUF/HF/model_type API 与字段 | 前端残留背后还有设置、OpenAPI、测试快照和兼容数据残留 | 只隐藏 UI 不够 |
| G11 | Settings 仍允许 Seed↔Cortex 热切换，Agent/ReAct/MCP/RAG 只在 Legacy router 下出现 | “Legacy 仅离线对照”没有落实到产品边界 | 架构决策未产品化 |
| G12 | Chat/Life/Settings 仍出现“不经过学习式 embedding”“ByteSensor→ByteMotor 即 Taiji”等旧文案 | 产品继续传播已被 2026-08-25 架构纠正否定的方向 | 文案/心智模型漂移 |
| G13 | `ChatRequest`、Agent 设置仍暴露 `engine/temperature/max_iterations`，Seed 原生分支实际忽略这些字段 | 用户配置看似有效，实际不进入原生运行时 | 幽灵配置 |
| G14 | `taiji/adapter.py`、主要 Vue view 和路线图持续膨胀 | 新能力容易继续堆进巨型文件并产生隐藏耦合 | 模块边界债 |

#### 16.1.4 术语和所有权重新钉定

后续接口禁止继续使用含义模糊的 `language` 或 `model_type`：

| 概念 | 规范名 | 决策权 | 状态/证据 |
|---|---|---|---|
| 人类自然语言表达器 | `natural_language_backend` | Taiji 生成合同 + Seed provider loader | provider artifact、Gate、health |
| IDE 编程语言 | `programming_language_id` | Taiji 可提出/选择；Seed capability 执行 | 文件内容、扩展名、manifest、LSP、confidence、provenance |
| 文件语法高亮 | `editor_language_id` | Workbench projection | 可与 programming language 相同，但不是认知主体 |
| 运行器/工具链 | `runner_id` / `toolchain_id` | Seed capability registry + policy | 可用性、版本、平台、资源、权限 |
| Taiji 保存格式 | `taiji_checkpoint_format` | Taiji/Seed checkpoint contract | `seed-native-v1` 兼容信封与 native payload |
| 外部嘴巴资产 | `language_provider_artifact` | Seed 集成边界 | 可使用 HF/Transformers/LoRA，但仅是末端器官 |
| 导入/导出适配格式 | `artifact_adapter_format` | Seed 发布工具 | 不得成为认知架构或全局 runtime 开关 |

HF 本身不是禁词：Hugging Face 数据集、缓存、Qwen/Transformers provider 和 adapter 可以继续存在于数据/语言器官集成边界。
必须清除的是把 HF/GGUF/LoRA 当作 Taiji 核心 checkpoint、全局模型类型或正式产品认知切换的 UI/API。合法的外部 provider
能力移动到“语言器官资产”语境，不再出现在“Taiji 模型格式”语境。

#### 16.1.5 目标架构

```text
User / environment observation
        |
        v
Taiji perception -> world/self/memory/goal -> plan -> ActionIntent
                                                    |
                                                    v
                                           structured ToolCall
                                                    |
                                                    v
Seed Workbench Capability Plane
  registry -> snapshot/freshness -> policy/approval -> transaction/executor -> audit
       |             |                    |                    |
       |             |                    |                    +-> file/terminal/LSP/MCP result
       |             |                    +-> deny / ask / allow / budget
       |             +-> current files, languages, tools, permissions, versions
       +-> typed schemas, risk, reversibility, resource cost
                                                    |
                                                    v
typed WorkbenchOutcome + after-state + diagnostics + provenance
                                                    |
                                                    v
Taiji Outcome / Observation / episodic+procedural memory / online credit / replan

Frontend IDE: observes the same snapshot, transaction and audit stream; it never becomes the hidden executor.
Language provider: realizes ExpressionPlan only; it never receives workbench authority.
```

工作台 capability 至少分为：

- `workspace.list/read/stat/search`：只读、可默认自动执行；
- `editor.open/reveal/set_language/diagnostics`：可撤销 UI/分析状态；
- `workspace.apply_patch/create/rename/delete`：文件事务，必须有 before digest、patch、after digest 和撤销记录；
- `terminal.run/test/build/debug`：命令 schema、cwd、timeout、环境变量白名单、资源预算和完整结果；
- `toolchain.detect/select`：识别项目语言、LSP、解释器/编译器，不静默安装依赖；
- `mcp.list/invoke`：通过统一 schema 注册，不能继续直接复用 NeuroPlex registry；
- `dependency.install/network/destructive`：高风险能力，除非用户预先建立窄 allowlist，否则必须显式审批。

#### 16.1.6 不可破坏的不变量

1. Taiji 决定做什么；Seed 决定能力是否存在、是否获准以及如何安全执行；前端只观察和承载人机控制。
2. 不允许语言 provider、ReAct、RAG、工作流或 UI 先决定动作，再让 Taiji 只做文案/打分。
3. 每个 action 必须绑定 `intent_id/call_id/capability_revision/world_tick`，每个 outcome 必须绑定真实执行和 after-state。
4. capability snapshot、审批、预算、执行和 outcome 必须可 checkpoint/重启续接；过期 snapshot fail-closed。
5. 文件修改使用事务/patch，不把任意自然语言直接写盘；执行前后可 diff、可撤销、可审计。
6. IDE 编程语言允许 Taiji 自主切换，但自动执行只限高置信、可逆的 `editor.set_language`；若会改变 runner、安装依赖、
   执行命令或覆盖未保存状态，必须进入对应风险 Gate。
7. UI 不得展示后端/原生模式未注册的能力；API 不得保留永远返回“不支持”的正式操作来制造假能力。
8. Legacy 只保留离线 benchmark/兼容启动，不再作为正式客户端的隐藏能力供应商。
9. 每个阶段先证明门禁能变红，再验收绿；CI 必须同时跑 native、legacy-off、frontend contract、Windows 和 packaged smoke。
10. CUDA 继续暂缓；工作台闭环不依赖本机硬件升级，不能以 CUDA 为阻塞理由。

#### 16.1.7 唯一顺序：Workbench Closure W0–W7

以下是严格顺序，不是可并行菜单。前一阶段退出 Gate 未通过，不进入后一阶段。

##### W0：Workbench Capability Contract + 只读真实纵切片

目标是先打通最小但真实的 `Taiji → Seed → IDE/workspace → Taiji` 回路，而不是先做万能 Agent。

工作项：

1. 在 Seed 产品边界定义版本化 `CapabilityDescriptor`、`CapabilitySnapshot`、`WorkbenchActionRequest`、
   `ExecutionPolicyDecision`、`WorkbenchTransaction` 和 `WorkbenchOutcome`；Taiji 继续只使用自身 `ToolCall/Outcome`。
2. 新建 Taiji-native `WorkbenchEnvironment(TaijiToolEnvironment)`，首批只注册 `workspace.list/read/stat/search`、
   `editor.open` 和 `editor.diagnostics.read`；不得导入 NeuroPlex。
3. 把工作区基础 API 从 Legacy optional router 中拆出为 core router；Legacy create-project/analyze/install 单独隔离或返回明确
   `legacy_only`，不再让 IDE 是否存在取决于 `SEED_ENABLE_LEGACY`。
4. SeedRuntime 新增 action event stream：`planned → policy → executing → outcome`；前端聊天与 IDE 订阅同一事件，
   `editor.open` 由状态投影驱动，不再通过无人消费的 window event bridge。
5. `/api/runtime/status` 只从 capability registry 上报工具；删除“空工具列表但宣称可自主探索”的推断文案。
6. 建立第一个真实 canary：在临时工作区放入未见文件，Taiji 形成读取意图，Seed 执行读取，文件 digest/内容摘要作为
   `WorkbenchOutcome` 回写，checkpoint 后可继续，关闭 environment 时 fail-closed。

退出 Gate：

- legacy-off 启动仍能打开 IDE、列目录和读取文件；
- 一条真实 read-only action 从 `ActionIntent` 到 UI 可见 outcome 全链路保留同一 lineage；
- 断开 WorkbenchEnvironment、篡改 capability revision、路径越界和过期 snapshot 均确定性失败；
- 任何前端显示的 capability 均存在于 OpenAPI/runtime snapshot，契约测试可通过故意删端点变红；
- 未实现 write/terminal/MCP 时 UI 明确显示“未授权/未接入”，不得伪装可用。

##### W1：编程语言识别、选择与 IDE 自主切换

1. 用 `ProgrammingLanguageEvidence` 统一扩展名、shebang、文件内容、项目 manifest、邻近文件、LSP 与 toolchain 可用性；
   现有 `extToLang` 只降为一个低权重证据源。
2. `programming_language_id`、`editor_language_id`、confidence、provenance、capability revision 和用户 override 进入
   Workbench state；Monaco 不再维护第二份隐藏真相。
3. 注册可逆 `editor.set_language` action。高置信且不改变运行器/文件内容时可由 Taiji 自动执行；低置信、语言冲突或会改变
   toolchain 时产生 `ask_user` policy outcome。
4. 语言列表由 backend capability 动态提供；未知语言保持 `plaintext`，不得因不在硬编码数组而丢失状态。
5. 用 `.h`、无扩展 shebang、多语言 monorepo、Vue/TS、notebook/markdown code block 和错误扩展名建立 holdout；
   filename-only lesion 必须显著退化，证明不是扩展名查表。

退出 Gate：Taiji 能解释“为何选择该语言”、自主切换后 Monaco/LSP/runner snapshot 一致，用户 override 可保持并撤销，
checkpoint/重启不会把旧语言状态错误应用到新文件。

##### W2：受控写入、终端与测试执行

1. 文件修改只接受结构化 patch/transaction，包含 before digest、目标路径、预期 after digest、冲突处理和 undo token；
   create/rename/delete 使用同一事务模型。
2. 终端从交互式 WebSocket UI 中抽出非交互 `terminal.run` executor，参数包含 argv、cwd、timeout、env allowlist、
   output limit 和 expected artifacts；不把 shell 字符串直接拼接执行。
3. capability 风险分级采用渐进自治：只读默认自动；可逆编辑按用户 autonomy policy；写入需预览/撤销；安装、网络、删除和
   破坏性命令默认显式审批。
4. 真实 diagnostics/test/build 结果回写 Taiji；成功不以 exit code 单独判断，还要记录 diagnostics、产物和 after-state。
5. 故意覆盖未保存文件、cwd 漂移、超时、输出洪泛、部分 patch 冲突和进程中断，验证原子失败与恢复。

退出 Gate：Taiji 可在临时项目中读文件、选择语言、生成 patch、运行测试、观察失败、重规划并修复；全过程可审计、可撤销、
checkpoint 续跑不重复执行已提交事务。

##### W3：原生工具/MCP registry 与自主循环

1. 将 workspace/terminal/LSP 与 MCP 都适配到同一 Seed capability registry；复用协议思想，不复用 NeuroPlex 认知/工具选择器。
2. MCP 管理 API 与前端按 OpenAPI 生成/校验，修复 path/body/query 漂移；安装/启动服务与调用工具分开授权。
3. Taiji `SelfState` 保存可用工具、权限、成功率、延迟、资源成本和最近失败，不把 UI localStorage 当自我模型。
4. 以真实 outcome 更新 affordance、procedural memory、world model 和 replan；语言 provider 只解释结果，不做隐藏 tool selection。
5. 建立有限 horizon autonomous task loop，并有 step/time/resource budget、取消、暂停、人工接管和恢复。

退出 Gate：在全新临时项目完成一个跨文件、诊断、测试的代码任务；去掉 Taiji planner 或 WorkbenchEnvironment 任一侧均失败，
证明不是 Legacy ReAct/外部 decoder 偷做；多次 checkpoint 不重复工具副作用。

##### W4：HF/GGUF/Transformer/Legacy 产品残留迁移

1. 前端删除 GGUF 导出按钮、LoRA 合并发布文案、无效的 `engine/temperature/max_iterations` 和正式产品 Cortex 热切换；
   外部 Qwen/LoRA 只在“语言器官资产”页面/高级配置中出现。
2. 后端将 artifact 分类收敛为 `taiji_checkpoint`、`language_provider_artifact`、`legacy_benchmark_artifact`；删除全局
   `model_type=gguf/self/cortex` 语义。
3. 对已保存 `gguf_path/model_type/model_name` 做一次显式设置迁移：能识别则转到 legacy/provider 配置，不能识别则隔离并提示，
   不静默猜测；旧端点先返回版本化 deprecation/410，再在一个兼容窗口后删除。
4. OpenAPI snapshot、Pydantic model、settings schema、frontend locales/composables/tests 一次清完；`download_hf` 等永远“不支持”
   的正式路由不能继续留在产品 API。
5. NeuroPlex 保留离线 benchmark CLI、固定数据/报告和 opt-in compatibility profile；默认客户端、主导航和 runtime status 不再暴露。

退出 Gate：frontend/source/OpenAPI/core settings 中不再存在 GGUF 或认知主体热切换；`taiji/` 仍零 Transformer import；
Qwen provider canary 仍通过，证明清理的是错误产品语义而不是合法语言器官。

##### W5：客户端全部内容与 Taiji 实际能力对齐

1. Chat 首屏、Life、Agent、Training、Settings、KB 的每一项状态标注真实 source、owner、freshness 和可用性；删除
   “ByteSensor→ByteMotor 即完整 Taiji”“不经过学习式 embedding”等已失效文案。
2. Life 面板改读 Taiji homeostasis/self-state；尚无原生数据的卡片隐藏或标为 roadmap，不再用 Legacy scheduler 代填。
3. Agent 配置改为 autonomy policy、capability scope、预算和审批偏好；不再展示 Seed 原生不消费的 ReAct 温度/迭代配置。
4. KB/RAG 只有在检索结果能作为带 provenance 的 Observation 进入 Taiji 时才称“知识能力”；否则仅称资料库管理。
5. 建立 route-level packaged smoke 和 capability screenshot/state contract，防止“页面可见但功能未接”再次全绿。

退出 Gate：默认客户端只展示 Taiji-native 实际能力；断开任一后端 capability 时 UI 自动降级且不保留假按钮/假状态；
文案、health、runtime snapshot 和可执行行为一致。

##### W6：模块化、契约生成与发布可靠性

1. 按 perception/world/memory/planning/execution/language/checkpoint facade 拆分 9300 行 adapter；保持公开 facade 和 checkpoint
   兼容，不做无验证的大重写。
2. 大型 Vue view 拆为 view model + typed API client + 可复用 panels；所有 mutation 经统一 client，不散落 URL 字符串。
3. 从 OpenAPI/capability schema 生成或校验前端 client，CI 检查每个前端调用有端点、method/body/query 对齐，native/legacy-off
   注册表与界面 capability 一致。
4. 增加真实任务 trace、action latency、policy deny、rollback、checkpoint resume 和 outcome learning 指标；研究 report 与产品 SLO 分离。
5. 发版门禁覆盖源码、dist、打包客户端、legacy-off、首次工作区、升级设置迁移和进程回收。

退出 Gate：模块拆分前后 checkpoint digest/行为等价；故意制造一个 URL、schema、能力状态或 checkpoint 漂移时 CI 必红；
打包客户端完成 W2/W3 canary。

##### W7：后续可靠性、研究、性能与产品体验工作包

W7 不是把 provider watchdog、interaction-group、小型模拟 Gate、CUDA 或视觉体验永久搁置，而是把它们从“现在就继续加功能”
改为**有前置条件、有升级路径、有真实验收的后续工作包**。其中小型模拟 Gate 是 W0–W7 全程使用的验证层，不需要等到 W7
才恢复；其余工作包只有在所依赖的产品合同稳定后才进入实施，避免继续用内部模拟替代尚未闭合的工作台能力。
W7 的实施阶段仍在 W0–W6 全部通过后开始；下表的进入条件是各工作包除总顺序外还必须满足的证据条件，不是允许提前插队。

排程定位如下：

| 工作包 | 是否保留 | 进入条件 | 在总路线中的位置 |
|---|---|---|---|
| 小型模拟 Gate | 保留且立即作为验证手段使用 | 对应阶段已有可故意打红的因果假设 | W0–W7 横切，不单独宣称产品完成 |
| provider runtime watchdog | 完整保留 | W0 的事件、审计、checkpoint lineage 与 W3 registry 稳定 | W7-R1 |
| interaction-group / recovery attribution | 完整保留 | W2/W3 已产生真实多步失败、恢复和工具 outcome trace | W7-R2 |
| 视觉与桌面体验收口 | 完整保留 | W5 已完成能力、状态、文案与路由真实性对齐 | W7-R3 |
| CUDA 与性能优化 | 完整保留，当前仅硬件验证受阻 | W6 固定 CPU 基线、checkpoint 合同，并取得可用 CUDA 主机 | W7-R4 |
| 开放域学习与结构自进化 | 完整保留 | 上述真实任务 trace、资源指标和 causal lesion 可共同支撑增长决策 | W7-R5 |

###### W7-G0：三层 Gate 梯度——小型模拟不取消，但必须向真实环境毕业

小型模拟 Gate 的正确定位是低成本证明机制是否存在，而不是能力终点。此梯度从 W0 开始适用于每一个工作包：

1. **S0 确定性小型模拟**：最小数值世界、固定 seed、边界输入和单一因果变量；必须先通过 lesion、错误输入或断开关键组件证明
   Gate 会红，再验证实现后变绿。S0 可以阻止错误实现进入下一层，但不得单独形成“已具备通用能力”的产品声明。
2. **S1 replay / sandbox Gate**：使用脱敏的真实 action/outcome trace、临时仓库、失败重放和 checkpoint 中断续接；验证机制能处理
   非理想顺序、工具错误、状态漂移和资源限制，而不是只适配手写 toy schema。
3. **S2 packaged-client / real-workbench canary**：在打包客户端、legacy-off 和真实工作台 capability 下完成用户任务；以真实文件、
   diagnostics、命令结果、UI 状态和 audit lineage 作为最终证据。

每个新 Gate 必须在 manifest 中声明 `claim`、`owner`、`S0/S1/S2 level`、`red proof`、`graduation target`、输入摘要、
checkpoint revision 和替代了哪些旧报告。S1/S2 已覆盖的 S0 执行日志进入 archive，只保留可复现脚本、合同和最终报告，避免
模拟报告无限堆积。任何能力若只有 S0 证据，路线图必须显式写“模拟机制成立，产品能力未验收”。

退出 Gate：每项长期能力都能从当前 claim 追溯到对应 S0/S1/S2 证据；故意移除关键组件会在最低适用层变红；不存在用 S0
通过结果替代 S2 产品完成声明的情况。

###### W7-R1：provider runtime watchdog、稳定回退与恢复

目标不是让外部 Qwen/provider 变成 Taiji 的认知主体，而是保证作为“语言器官”的 provider 在运行时退化时可检测、可隔离、
可回退，且失败不会污染 Taiji 的认知状态、checkpoint 或下一次请求。

工作项：

1. 为每次语言 realization 建立版本化健康记录，至少区分 `accepted`、语义校验失败、validated fallback、timeout、加载异常、
   artifact 漂移和 canary 失败；健康状态按 artifact digest 隔离，禁止跨版本继承计数。
2. 使用连续失败阈值、滚动接受率、冷却期和恢复迟滞共同决定 `healthy/degraded/quarantined/probing`，避免单次抖动触发回退，
   也避免 provider 在 active/previous 间频繁振荡。
3. 自动回退只允许落到 registry 中 allowlisted、内容寻址仍有效且 canary 通过的 previous version；previous 漂移、过期或不存在时，
   必须 fail closed 到 `native-readable`，不得选择任意本地模型。
4. watchdog 状态、计数、冷却期限、active/previous revision 和最后失败原因进入 checkpoint；重启后继续原状态，但不得重放已经完成
   的语言请求或泄漏 prompt/history。
5. runtime status、聊天 final event 和异常中心显示 active/fallback/quarantine/probe、artifact revision 和可操作原因；前端只观察，
   不自行决定轮换或清空错误。
6. canary 覆盖“active 连续退化 → previous 原子回退 → 冷却 → 隔离 probe → 恢复 active”，以及 previous 漂移、进程中断、
   并发请求和 checkpoint continuation；任何失败不能形成半提交 registry。

退出 Gate：provider 退化可在请求级 trace 中重现；回退目标经过内容寻址和 canary 重新确认；重启前后 watchdog 决策一致；移除
health source、篡改 previous digest 或关闭语义 validator 时 Gate 确定性变红；Taiji 在 provider 全部不可用时仍通过
`native-readable` 给出可读、来源清楚的降级输出。

###### W7-R2：interaction-group 与 recovery attribution 的真实任务化

interaction-group 不再为了增加“神经元群”概念而扩展，而是用于解释和改善真实多策略、多工具、多记忆源共同参与时的成功、失败与恢复。

工作项：

1. 从 W2/W3 的真实 task trace 定义 interaction observation：参与的 workspace route、memory source、planner branch、tool call、
   recovery action、资源成本和最终 outcome；不得按名称或手工角色表直接指定贡献。
2. 在预算内实现可计算的边际归因：先使用 leave-one-group-out、成对交互和局部反事实；只有证据显示高阶交互必要时才提高阶数，
   不默认做指数级全子集搜索。
3. group 的形成、合并、拆分、休眠和剪枝由持续贡献、互补性、冲突率、恢复价值和资源预算驱动；结构变化写入 provenance，
   可单独回滚，不得破坏无关 group 的 digest 与 checkpoint 状态。
4. 把 recovery attribution 回写到 workspace routing、procedural/episodic/semantic memory 和 planning policy，但保留各 owner 的更新边界，
   不建立一个重新包办全部学习的中心控制器。
5. 使用未见工具组合、跨文件故障、错误诊断、部分 patch 冲突和多步恢复建立 holdout；与 single-strategy、no-group、random-group
   和 no-attribution 做同预算对照。

退出 Gate：interaction-group 在至少一类真实工作台任务上稳定优于最强单策略和随机分组；lesion 能定位退化来源；错误归因可局部回滚；
计算与内存开销随 group 数量保持有界；不存在只有 group 数量增加、任务成功率和恢复效率不改善的“规模即进化”声明。

###### W7-R3：视觉、桌面外壳与交互体验收口

视觉工作不取消，但必须建立在 W5 的真实 capability 和状态模型上；否则只会把错误的 Legacy/HF/伪 Agent 内容包装得更漂亮。

工作项：

1. 统一客户端信息架构：侧边栏在目标窗口高度内完整显示核心导航，低频项进入显式二级入口；工作台、Taiji 状态、训练、语言器官、
   设置和异常中心的层级与真实 owner 对齐，不再使用滚动隐藏关键入口。
2. 建立设计 token 和可复用组件，统一字体、间距、圆角、阴影、色彩、分隔、focus ring、loading/empty/disabled/error/fallback/
   approval/executing/rollback 状态，清除各页面独立硬编码样式。
3. 收口 Windows 桌面品牌资产：窗口/任务栏、系统托盘、托盘通知和打包产物使用同一 Taiji logo 来源与多尺寸资源；应用内允许
   低成本流转动画，任务栏和系统通知使用平台兼容的静态帧；圆润外壳在缩放、最大化和系统阴影下不裁切内容。
4. UI 只展示 registry 中真实存在的 capability，并完整呈现 action lineage、审批、执行、回退和 provider 降级；视觉状态不能掩盖
   capability 不可用、Legacy-only 或实验性边界。
5. 覆盖键盘导航、焦点顺序、对比度、reduced-motion、100/125/150/200% DPI、多显示器、浅深色和小窗口；禁止为追求动效
   牺牲可访问性、启动时间或托盘稳定性。
6. 建立 route screenshot/state contract 与 packaged-client smoke，重点检查侧边栏溢出、窗口圆角、任务栏/托盘/通知图标、真实状态源、
   首屏任务完成路径和异常降级。

退出 Gate：默认打包客户端在各 DPI 下无关键导航滚动、裁切和空白壳；桌面所有品牌入口一致；关闭 capability/provider 时 UI 能准确降级；
视觉回归、可访问性和打包 smoke 均可通过故意破坏 token、图标或状态绑定而变红。

###### W7-R4：CUDA、跨设备一致性与测量驱动的性能优化

CUDA 不是取消，而是**当前缺少可验证硬件**。在获得 CUDA 主机前允许整理 device abstraction、benchmark schema 和 CPU 基线，
但不得提交“已适配 CUDA”或“已加速”的能力结论；CUDA 主机不可用也不阻塞 W0–W7-R3。

工作项：

1. 在 W6 固定代表性 CPU workload、数据 manifest、seed、checkpoint revision、精度指标、峰值内存、吞吐与延迟，先 profile 出真实热点；
   没有测量证据的模块不进入 CUDA 优化。
2. 建立显式 device/dtype/capability 合同，禁止模块内部私自选择设备；CPU-only、CUDA unavailable、显存不足和设备切换均有确定性回退。
3. 验证 CPU → CUDA → CPU checkpoint continuation，覆盖 optimizer/local-learning 状态、随机状态、稀疏结构、provider artifact 引用和
   长序列中断；旧 CPU checkpoint 必须继续可读。
4. 先做算子迁移和批处理/向量化，再按 profile 证据评估 mixed precision、稀疏布局与 fused kernel；自定义 kernel 必须保留参考实现、
   数值对照和硬件 capability fallback。
5. 跨设备验证 deterministic/tolerance 边界、NaN/Inf、OOM 恢复、吞吐、p50/p95 latency、峰值显存和能耗代理；性能提升不能以
   破坏 Gate、checkpoint 或学习质量为代价。

退出 Gate：在真实 CUDA 主机上通过 CPU/CUDA 数值与 checkpoint 一致性；目标 workload 达到预先登记的加速和显存阈值；移除 CUDA、
降低 capability 或触发 OOM 时自动回落且结果可审计；只有实测热点才允许保留 fused/sparse kernel。

###### W7-R5：开放域学习与结构自进化

自进化不等于持续增加神经元数量。它必须由真实任务上的长期误差、容量拥塞、恢复失败和新分布证据触发，并同时允许生长、重组、
巩固、休眠、剪枝和回退。

工作项：

1. 聚合 W2/W3/W7-R2 的真实失败簇，区分表示容量不足、记忆干扰、路由冲突、世界模型误差、工具缺失和语言 realization 失败，
   防止所有问题都被误判为“需要扩大神经元规模”。
2. 为 perception/workspace/memory/world/planning 分别定义可观测的 capacity pressure 与 growth proposal；结构增长由局部 owner 提议，
   由全局资源治理器按收益、预算和可回滚性批准，不使用固定任务名触发器。
3. 新增单元/连接/group 先在隔离 shadow 状态学习，通过 holdout、lesion 和资源收益 Gate 后原子并入；未获益或产生漂移时恢复旧 topology，
   checkpoint 同时保存结构 revision 和参数状态。
4. 开放域 world/semantic/skill 学习从真实 provenance Observation 与 outcome 中形成，语言 provider 只负责表达；新知识必须能追溯、
   冲突检测、遗忘控制并被任务成功率验证。
5. 长期评测同时记录能力收益、遗忘、恢复时间、参数/连接规模、内存、延迟和能耗代理；禁止只报告规模扩大或训练 loss 下降。

退出 Gate：至少一类未见真实任务触发结构增长后，在同预算 holdout 上优于冻结结构，并且无关能力遗忘受控；growth lesion、错误增长和
rollback Gate 均成立；当容量压力消失时系统不会继续无界扩张。只有满足这些条件，才能把“自然生长式迭代”作为 Taiji 已实现能力。

W7 的实际实施顺序固定为 **G0 贯穿全程，R1 → R2 → R3 → R4 → R5**。若到达 R4 时仍没有 CUDA 主机，R4 标记为
`hardware-blocked`，可先整理不声称完成的基线和测试资产，但不得绕过真实 CUDA Gate 把它记为完成；R5 的结构增长仍必须等待
真实资源数据，不能因为 CUDA 暂缺而改用更多 toy Gate 代替。

#### 16.1.8 立即冻结和归档边界

- 不删除 P1–P7 核心架构讨论、requirements、native architecture 和本路线；它们仍是后续开发的实时依据。
- 测试过程日志、一次性调试探针、已被后续 Gate 覆盖的执行记录可继续进入 archive；核心决策和未关闭缺口不归档。
- W0 已闭合（2026-08-29）；在 W1 之前不增加写文件自治、终端自治、MCP 或新的研究 Gate。interaction-group、provider watchdog、
  CUDA、视觉美化、Legacy Agent 和新格式支持仍完整保留在 W7/后续边界，不是删除；小型模拟 Gate 继续作为各阶段的 S0 验证工具。
- W0 首批实现已落地（2026-08-29）：Seed 已拥有版本化 workbench 合同、内容寻址 capability snapshot、只读
  `WorkbenchEnvironment`、core router、runtime capability projection、`planned → policy → executing → outcome` 审计链，且
  `ActionIntent → ToolCall` 的结构化工具路径已与 motor-symbol `settle_action` 解耦，工作台摘要感知值遵守 Taiji byte sensor 值域。
- W0 前端投影已接线（2026-08-29）：`WorkspaceView` 按 native capability 懒加载目录、`MonacoEditor` 通过 native read 打开文件，
  页面显示 snapshot/最近 outcome，`editor.open` outcome 可由统一 audit projection 驱动 IDE 打开；旧 Legacy 写入、终端、重命名仍未被
  冒充为 native。
- W0 checkpoint continuation 已通过：checkpoint round-trip 恢复 capability snapshot、workbench audit 和 tick，审计阶段保持
  `planned → policy → executing → outcome`；失效 snapshot、越界路径、错误 sensor 值域和断开环境均 fail-closed。
- W0 packaged-client/real-workbench S2 canary 已通过：`SEED_ENABLE_LEGACY=0` 下真实 `dist/Seed/Seed.exe` 成功启动后端，
  `/api/workbench/capabilities` 与 `/api/workbench/files?path=.` 均返回 200，native workspace bytes 与 capability snapshot 可读；
  打包期间发现的 Qt6/ICU DLL 冲突已在 `desktop/seed.spec` 过滤，并纳入 release 检查。该证据确认打包客户端启动和 native route
  可用，不等同于 W5 的 GUI 视觉、DPI、托盘或人工点击验收。

**W1 语言识别、选择与 IDE 自主切换退出 Gate 已通过（2026-08-29）。** `ProgrammingLanguageRegistry` 以内容寻址规则统一
扩展名、shebang、内容、manifest、邻近文件、可选 LSP 与 toolchain 证据；`programming_language_id/editor_language_id`、
confidence、provenance、registry revision、explanation 与 user override 已进入 Workbench state。`editor.set_language` 已成为
Taiji-native 可逆 action：高置信且与证据一致时允许 Taiji 自动切换，低置信、语言冲突或 `.h` 等歧义场景返回 `ask_user`；
runner/LSP 上下文和可用工具链快照与同一语言选择绑定，显式用户覆盖可撤销并按文件 digest 失效，checkpoint 不会把旧覆盖
错误应用到新内容，Monaco 已提供“自动检测”入口。holdout 覆盖 `.h`、无扩展 shebang、多语言 monorepo、Vue/TS、notebook、
markdown code block、错误扩展名和 filename-only lesion；API/OpenAPI、runtime/checkpoint 与 Monaco 动态 projection 已接通，
后端 Workbench 合同 `8 passed`、Monaco 回归 `10 passed`，前端 lint `0 errors`、构建通过。该 Gate 是语言/IDE 合同闭环，
不代表 W2 runner 已可执行。

**已完成（2026-08-29）：W2 首批受控执行合同与 executor。** native capability snapshot 升至 revision 3，新增
`workspace.apply_patch/create/rename/delete/undo` 与 `terminal.run`。文件修改已收敛为 UTF-8 结构化 text-replace 和统一
transaction：before/after SHA-256、原子写入、冲突 fail-closed、唯一单次 undo token；create/rename/delete 共享同一撤销模型。
终端执行已收敛为 argv-only、明确 `shell=False`、workspace 内 cwd、bounded timeout/output、env allowlist 与 expected artifacts，
非零退出和超时都会产生失败 outcome；runtime 会保留 executor 返回的真实 transaction payload，旧只读能力保持兼容投影。
文件事务、终端边界、审批策略与失败结果回归共 `11 passed`，ruff、Black、py_compile 和 diff check 通过。该 slice 只完成
contract/executor，不把直接 executor 调用等同于产品自治：写入和终端仍由 policy 默认返回 `ask_user`，未完成 IDE 预览/审批、
真实 diagnostics/test/build outcome 回写以及 checkpoint 续跑 Gate。

**已完成（2026-08-29）：W2 第二 slice 的审批、预览与真实 outcome 闭环。** `/api/workbench/preview` 对精确 action request
做不落盘验证并生成短期一次性 approval token；`/api/workbench/execute` 只有携带同一请求绑定的 token 才能执行高风险写入/终端，
重放、过期、参数或 snapshot 漂移均 fail-closed，审计请求只记录 approval presence。`terminal.run` 已增加 command/diagnostics/test/build
execution kind、结构化 diagnostics、expected artifacts、after-state，并按 timeout、exit code、诊断错误和缺失产物综合计算 success；
runtime 和通用前端 projection 已接入 preview/execute client。后端 Workbench 合同 `12 passed`、前端完整回归 `187 passed`、构建通过，
ruff/Black/py_compile/diff check 通过。该 slice 完成的是审批/结果合同，不等于 checkpoint 后 undo/approval 状态和真实临时项目续跑已通过。

**已完成（2026-08-29）：W2 退出 Gate 的 checkpoint 续跑与真实临时项目闭环。** transaction state 随 SeedRuntime checkpoint
保存并恢复 undo lineage，approval token 明确为 session-scoped、重启后失效；恢复后重新预览/审批可完成撤销。临时多文件项目已完成
语言识别→patch 预览/执行→test 产物→diagnostics 失败回写链路，预览无副作用，冲突/输出洪泛/cwd 漂移/超时/进程中断均有
fail-closed 证据；旧 `runtime_service` 边界测试同步到 native capability 事实。Seed/native 回归 `320 passed, 1 skipped`，W2
退出 Gate 通过，具备进入 W3 的证据。

**已完成（2026-08-29）：W3 第一纵切片的 native MCP registry 与 canary Gate。** 新增 Seed-owned `McpToolRegistry`，以内容寻址
registry snapshot、版本化 input schema、source/risk/timeout/output budget 和 registry revision 统一 MCP-shaped 工具合同；native
Workbench 新增 `mcp.list/invoke`，仅接入无安装、无网络副作用的本地 `workspace-summary` canary。参数 schema、registry revision、
未知/禁用工具、动态风险审批和输出超限均经过 Workbench policy/executor，失败时 fail-closed 并保留 outcome；API 的 capabilities 与
`/api/workbench/mcp`、前端 `mcpRegistry` projection 已接通。该 slice 没有接回 Legacy `mcp_manager`，不等于外部 MCP 生命周期管理、
真实远端服务连接或多步有限自治循环已完成。Workbench 定向回归 `18 passed`，前端 `187 passed`、构建通过、ESLint `0 errors/17 warnings`。

**已完成（2026-08-29）：W3 第二 slice 的 MCP identity、checkpoint 与 loop preflight Gate。** MCP registry 内容身份已随 runtime
checkpoint 保存/恢复；单次 Workbench request、Outcome 和返回 ToolCall 共享 capability snapshot/registry snapshot binding，审批
摘要也纳入 registry identity。`/api/workbench/loop/preflight` 与前端 `preflightLoop` 已接通，loop 只做不执行 admission，强制最多
8 步、总预算不超过 32 units、拒绝重复调用、首错终止和 `after_each_step` checkpoint 边界。Workbench 定向回归 `19 passed`，
Seed/native 全量回归 `320 passed, 1 skipped`，前端 `187 passed`、构建通过、ESLint `0 errors/17 warnings`。该 slice 不等于真正的
多步执行、逐步 checkpoint 提交或外部 MCP 生命周期管理。

**已完成（2026-08-29）：W3 第三 slice 的受预检有限多步执行 Gate。** 新增 `/api/workbench/loop/execute` 与前端 `executeLoop`，
只接受 preflight identity 未漂移的 native Workbench request；每个已尝试步骤都真实执行、写入 ToolCall/Outcome audit 并保存 checkpoint，
遇到失败立即停止并保留已完成前缀，恢复后重放已提交 request 会 fail-closed。真实成功两步、失败停机和 checkpoint 恢复重放定向回归
`21 passed`，Seed/native 全量回归 `320 passed, 1 skipped`，前端 `187 passed`、构建通过、ESLint `0 errors/17 warnings`。该 slice 尚未
扩展到跨文件 patch/test/diagnostics 任务，也未接入外部 MCP 生命周期。

**已完成（2026-08-29）：W3 退出 Gate 的真实跨文件代码任务 loop。** 在现有 preflight/execute/checkpoint 约束内，真实临时项目已完成
语言识别→跨文件 patch→test/build 产物→diagnostics 失败→checkpoint 恢复→创建修复标记→diagnostics 重试；失败后只从未提交步骤继续，
已提交 request 的旧审批令牌即使失效也会先被 checkpoint 提交历史拒绝，避免误报为普通审批失败或重复副作用。去掉 Taiji planner 或
WorkbenchEnvironment 任一侧均 fail-closed，仍不接外部安装、网络服务或开放式自治。Workbench 定向回归 `22 passed`，Seed/native 全量
回归 `320 passed, 1 skipped`，前端 `187 passed`、生产构建通过、ESLint `0 errors/17 warnings`。

**当前唯一下一步：开始 W4 第一 slice 的产品语义残留清理。** 先对前端、API/OpenAPI、settings schema 和发布入口建立 GGUF/LoRA/
Transformer/Cortex/Legacy Agent/HF 认知主体残留清单，删除或迁移第一批仍暴露为正式 Taiji 能力的入口；保留 Qwen 等语言 provider
作为语言器官资产边界，并为每个迁移项补 native/legacy-off 回归，暂不触碰合法 provider artifact 和离线 benchmark。
