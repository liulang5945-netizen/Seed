# Seed / Taiji 计划与架构入口

## 当前唯一口径

**Taiji 是完整原生认知架构和模型；Seed 是项目、产品和运行时。**

当前顶层 `taiji/` 实现是 Taiji Substrate Kernel v8（TSK-v8）：它验证了 raw-byte codec、持续预测状态、局部学习、情景原型、行动闭环和 checkpoint，但不是完整 Taiji。Taiji v1 将在此基础上吸收成熟的表示学习、选择性路由、世界模型、记忆、强化学习、规划、生成和 CUDA 方法，按照项目需求重新组织，而不是从原始 one-hot 神经元重新发明全部能力。

Legacy NeuroPlex 是冻结的 Transformer 离线对照；它不进入 Taiji cognition。

## 当前权威文档

| 文档 | 权威范围 |
|---|---|
| [TAIJI_CORE_REQUIREMENTS.md](active/TAIJI_CORE_REQUIREMENTS.md) | 项目长期使命、CR-1–CR-10、旧 Transformer 壳失败教训与不可归档的核心依据 |
| [TAIJI_NATIVE_ARCHITECTURE_V1.md](active/TAIJI_NATIVE_ARCHITECTURE_V1.md) | Taiji 的完整目标：感知、表征、世界状态、工作空间、记忆、推理、规划、生成、学习和硬编码治理 |
| [SEED_DEVELOPMENT_ROADMAP_2026_08.md](active/SEED_DEVELOPMENT_ROADMAP_2026_08.md) | 唯一执行顺序：P0–P8、A0–A9 Gate 和当前下一步 |
| [ARCHITECTURE_DIRECTION_2026_08.md](active/ARCHITECTURE_DIRECTION_2026_08.md) | 规范词表、不可回退边界、成熟技术采纳规则和 Legacy 边界 |
| [SEED_ARCHITECTURE.md](active/SEED_ARCHITECTURE.md) | Seed 产品/runtime 所有权、允许/禁止职责与 checkpoint/API 迁移边界 |

`plans/active/` 只保留以上五份文档。发生冲突时，项目使命以核心需求为准，目标能力以 Taiji v1 架构为准，执行顺序以总路线为准，身份和依赖以方向决策为准。

## 旧实现与旧路线

- [TAIJI_SUBSTRATE_KERNEL_V8_SPEC.md](archive/implementation/TAIJI_SUBSTRATE_KERNEL_V8_SPEC.md)：TSK-v8 的精确方程、tick、局部学习、checkpoint 和旧 N/M 门槛。
- [SEED_DEVELOPMENT_ROADMAP_PRE_NATIVE_V1_20260825.md](archive/history/SEED_DEVELOPMENT_ROADMAP_PRE_NATIVE_V1_20260825.md)：架构纠正前的 R0–R7/S1–S3 路线。
- [SEED_STAGE_CLOSEOUT_20260825.md](archive/history/SEED_STAGE_CLOSEOUT_20260825.md)：上一阶段工程、产品和 kernel 成果收束。
- [archive/README.md](archive/README.md)：完整归档索引。

归档中的“当前状态”“下一步”和“完整 Taiji”声明全部按历史语境解释，不再指导开发。

## 当前代码事实

- `taiji/` 不导入 `seed`、`neuroplex` 或 `transformers`；该独立性继续保留。
- `seed/` 当前包装 `Taiji` kernel；P1 compatibility adapter 已迁移首个 v1 纵切片，不破坏产品 API 和旧 checkpoint。
- P6 client input-boundary Gate 已通过：`InputFrame` 版本化承载客户端原始 bytes 与来源元数据，`TSKV8Adapter.ingest_input()` 将其逐字节转换为 Taiji-owned `Observation/PerceptEvent`，`InputTrace` 可检查并 round-trip；`ActionIntent` 保持为空，未引入固定意图映射。`SeedRuntime.chat` 已通过 `generate_input()` 走同一合同，仍保留 raw-byte 兼容输出。
- P7 executive contract Gate 已通过：`ExecutiveController` 从 percept/world/memory/goal/homeostasis context 学习候选 utility，选择结果保持结构化 `ActionIntent + ContentPlan` 配对；adapter 提供选择、Outcome 反馈、lesion-safe checkpoint 与 round-trip。该 Gate 证明学习型候选选择，不证明已完成真实环境 action/outcome 闭环。
- P7 executive environment-loop Gate 已通过：`ExecutiveDecision` 通过显式 `WorldAction` 元数据和 motor `action_symbol` 接入 `TaijiEnvironment.step()`，真实 `EnvironmentOutcome` 回写 utility、感知和失败重规划；selected/alternative、checkpoint continuation、utility update 与 executive lesion 均有测试。环境可显式返回行动后 `WorldState`，但不会由 adapter 伪造。
- P7 candidate synthesis contract Gate 已通过：adapter 从当前 `PerceptEvent`、`WorldState.affordances` 和 active `GoalState` 自动生成带 provenance 的 `ExecutiveCandidate`，不需要客户端候选表；当前 affordance 特征仍是保守 scaffold，不宣称已学会通用 affordance 表征。
- P7 affordance feature transfer Gate 已通过：`WorldAffordance` 携带带 provenance 的 numeric grounding，`LearnedAffordanceFeatures` 由 Taiji-owned outcome objective 学习连续投影；candidate synthesis 只消费该投影，不读取 `affordance_id/action_kind` 查表，未见 affordance/action holdout 已通过，且 native checkpoint 可恢复该 source。
- P7 affordance online-credit Gate 已通过：真实 `EnvironmentOutcome` 的 reward 会回写当前 selected affordance 的 feature source；source lesion 会阻断候选合成，online update 计数、预测误差和权重可经 native checkpoint continuation 恢复。
- P7 contextual grounding Gate 已通过：adapter 强制 source 的 `context_dim` 对齐 Taiji perception，producer 读取 `Percept.features + WorldState.latent + uncertainty`；world latent 缺失时使用显式 percept fallback，context 改变会改变连续表示，组合/扰动 holdout 已通过。
- P7 world-grounding lineage Gate 已通过：adapter 在 `observe_event` 与 `settle_action` 进入认知状态前统一由 `WorldAffordanceGroundingProducer` 从 actor/target numeric object summary、relation binding、world latent 和 confidence 生成 raw grounding，并记录 `grounding_lineage`；`action_kind/affordance_id` 不参与特征查表。
- P7 end-to-end grounding transfer Gate 已通过：`WorldAffordanceGroundingProducer → LearnedAffordanceFeatures → ExecutiveController` 在新对象、新关系谓词和新 action kind 的 holdout 上保持正确选择；producer lesion 会使选择退化，证明 executive 消费的是 grounding 表征而非符号表。
- P7 grounded multi-step environment Gate 已通过：`EnvironmentOutcome.world_state` 进入真实 `WorldTransition` 后，adapter 在行动前后都保留 `grounding_lineage`；失败 action 触发 alternative replan，原决策的 delayed credit 可跨 replan 与 native checkpoint 恢复，并继续更新对应 affordance source。
- P7 grounded multi-step train/holdout Gate 已通过：4 条 train affordance、未见 actor/target/relation/action kind 的 holdout 和 3 个 seed 均达到 holdout selection、四步链路中前三步连续 failure replan、全程 before/after lineage、checkpoint pending credit 与跨步 delayed credit `1.0`；manifest/report 为 `reports/taiji_p7_grounded_multistep_*_20260825.json`。该结果仍是小型数值世界 transfer，不代表通用关系推理。
- P7 grounded multi-step causal-lesion Gate 已通过：3 个 seed 的 producer lesion 均使 holdout 选择退化，feature-source lesion 均阻断候选合成，跳过 delayed credit 均少一次 source/executive online update；结果与主 Gate 一起写入同一 report。该结果证明当前控制变量有因果效应，不代表长程规划。
- P7 variable-horizon episode Gate 已通过：同一 train/holdout 学习结果在 3/4/5 步 episode、不同失败位置和多个 after-state relation 变化下，3 个 seed 均完成预期 replan、全程 lineage 与每个非终止步的 delayed credit。该结果扩大了 horizon 边界，但仍不是长程规划证明。
- P7 executive-to-world prediction/calibration Gate 已通过：executive bridge 现在把带 actor/target 的 `WorldAction` 送入 `WorldDynamicsLearner`，真实 after-state settle 回写 state/reward error；data-derived schema 的 train/holdout 为 `2/2`，3 个 seed 均在逐条真实转移的 online correction 后降低状态预测误差，no-online-update clone 保持原误差。reward error 继续独立记录，不与状态校准混成一个指标；该 Gate 只证明窄数值世界上的预测误差可回写并校准，不证明开放世界预测精度。
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
- `neuroplex/` 保持冻结，只用于离线对照和显式兼容。
- `CapacityPolicy` 当前规划固定区域/fan-in/memory 资源；v1 中将降为资源治理器，不再规定认知结构。
- N0–N11/M5–M7 保留为 TSK-v8 kernel 回归，不再作为概念、推理、语言或智能进展证明。
- 旧 16M raw-byte 长训暂停；P2 学习型时间抽象通过前不恢复 100M 路线。
- P3 已完成世界 transition lineage、预测误差在线校正，以及 WorkspaceCandidate/WorkspaceSelection 合同；
  `WorkspaceRouter` 已提供容量受限 learned/none/random 路由和 native checkpoint。
- A3 静态组合窄 Gate 已通过：3 seeds 的 learned 路由精确选中率均为 1.0，组合重建 MSE 为 0；相对最强单体
  平均改善 `+0.1922`，相对 dense mean 平均改善 `+0.7016`。这不是多步世界任务或通用协作能力证明。
- A3 world-outcome 窄 Gate 也已通过：`assemble → commit` 两步 `TaijiWorldState` episode 中，learned 路由 3 seeds
  的 final success 均为 1.0、mean reward 为 2.0；strongest-single/dense 均为 0，random 平均为 0.2292，none 为 0。
  这只证明当前小型组合任务的 workspace→action→outcome 因果链，不代表一般规划或通用智能。
- P4 最小记忆纵切片已落地：`WorkingMemoryItem`、`EpisodicMemoryRecord` 和可容量治理的 `EpisodicMemoryStore` 属于
  Taiji；adapter 在真实 action outcome 后写入经历、下一观察按 cue 检索，native checkpoint 可恢复记录与 working state。
  store 已改为 insertion-ordered dictionary，重复 memory_id 替换与容量淘汰不再每次全表重建；当前原生回归为
  94 passed、1 skipped。
- P4 cue-conditioned one-shot recall 窄 Gate 已通过：full、episode-ID lesion、checkpoint continuation 的 action recall
  均为 1.0，retrieval/write lesion 均为 0；报告和 manifest 为 `reports/taiji_p4_episodic_recall_*_20260825.json`。
  该结果证明经历检索和来源独立性，不证明从多次经历抽取新组合语义。
- P4 additive semantic consolidation 窄 Gate 已通过：3 条 episodic records 对未见 `[1,1]` 组合的最近情景误差为 `1.0`，
  consolidation 误差约 `0.0045`；replay lesion 误差 `2.0`，episode-ID lesion 与 checkpoint continuation 误差约 `0.0045`。
  这只证明一类数值关系可从经历中压缩，不证明一般概念、语言或程序技能。
- `SemanticMemoryLearner` 已接入 `TSKV8Adapter`：真实 settle outcome 进入 episodic store 后可由
  `consolidate_semantic_memory()` replay，semantic learner 与 episodic store 一起进入 legacy/native checkpoint；相关
  adapter/checkpoint 回归已通过。这关闭的是 runtime ownership 子门，不扩大 additive benchmark 的能力声明。
- P4 multi-factor/noisy semantic Gate 已通过：60 条经历覆盖 15 个已见组合，留出全激活组合；semantic error≈`0.0082`，
  episodic nearest error=`1.0`，replay lesion error=`4.0`，episode-ID/checkpoint error≈`0.0082`。这仍是 additive relation
  子门，不代表一般语义、程序技能或长期容量已通过。
- P4 capacity/procedural Gate 已通过：`100/1000/10000` 三档均严格保留容量上限，最旧目标在相似经历干扰下被淘汰，最新记录可
  召回；`ProceduralMemoryLearner` 从 `action_intent.kind` 数据发现动作类别，在四类 cue→action holdout 上准确率=`1.0`，
  skill lesion 基线=`0.25`，episode-ID lesion/checkpoint continuation 均=`1.0`。这证明资源边界与独立的程序性巩固原型，
  尚未证明 adapter runtime 已用该技能作出真实决策。
- P4 procedural runtime ownership Gate 已通过：adapter 通过显式 `available_actions ↔ action_kinds` 合同调用自身
  `consolidate_procedural_memory()`，真实 action selection 为 `1.0`，关闭 procedural route 后为 `0.0`，episode-ID lesion
  与 checkpoint continuation 均为 `1.0`；动作类别仍来自 replay 数据，adapter 不内置动作表。
- P4 procedural robustness Gate 已通过：GRU 按 `episode_id/tick` 学习多步 `prepare→transition`，未见 transition transfer 与
  checkpoint continuation 均为 `1.0`，相似 cue 干扰后为 `1.0`；当 episodic capacity 等于原训练集并加入干扰时，迁移准确率降为
  `0.5`，形成可测的资源受限遗忘边界。该结果证明序列技能 replay 原型，不等于规划或长期自我调节。
- P4 homeostatic regulation Gate 已通过：高 prediction error/负 reward/资源成本驱动 curiosity=`0.585`、fatigue=`0.3`、stress=`0.95`
  并自动选择 sleep；sleep、play、fixed schedule、random drive 和 no-modulator lesion 均产生预期差异；adapter outcome 更新与
  native checkpoint round-trip 已通过。这是内部调节子门，不等于完整生命系统。
- P5 goal-planning 窄 Gate 已通过：planner 综合 reward/success/progress/uncertainty/resource/conflict 选择 safe-route，
  reward-only lesion 选择 risky-route；adapter 真实执行 selected plan 后 goal progress=`0.4`，native checkpoint 保持 plan 和
  progress。该结果是单步可执行规划子门，不等于长程 rollout 或通用目标推理。
- P5 imagined rollout Gate 已通过：planner 选择 2-step safe rollout（provenance=`imagined`、confidence=`1.0`），真实首步
  reward 与预测差异 `0.6` 后设置 `replan_required`，该信号在 native checkpoint 中保持。当前只证明误差触发，不证明已执行替代计划。
- P5 replan/calibration Gate 已通过：首个 safe rollout 失败后 confidence 降至 `0.0` 并触发 replan；第二次实际执行 risky
  alternative，成功后 replan 清除、confidence 恢复至 `1.0`，safe/risky success calibration 均进入 native checkpoint。这证明
  了替代计划闭环，不等于 delayed-reward 或环境干预泛化。
- P5 intervention/latency 窄 Gate 已通过：完整 planner 选择 delayed-safe，reactive 与 discount=0 world-model lesion 均选择
  immediate-risky；planner 成功概率优势=`0.4`，真实干预触发 replan 并执行 recovery，最终 goal progress=`0.16`。这关闭 P5
  的首个 delayed reward/intervention 子门，不等于长程规划或通用目标推理。
- P6 structured generation 窄 Gate 已通过：`ActionIntent → ContentPlan → ExpressionPlan → ToolCall → UTF-8 codec` 保持
  `intent_kind`、semantic slots、tool name 和 goal provenance；codec round-trip 后可还原为同一 intent 绑定的 `WorldAction`。
  `TSKV8Adapter` 已拥有 generation controller 与 native checkpoint 恢复。该 Gate 只证明结构化工具效应器边界，不证明语言流畅性、
  自主内容创造或真实外部工具成功。
- P6 tool execution/outcome 窄 Gate 已通过：`TaijiToolEnvironment` 执行结构化 tool call 后，真实 `Outcome` 保持 intent ID、success、
  reward、terminal 并写入 episodic memory；无 generation organ 的 direct-byte lesion 不能执行同一工具合同。该 Gate 使用模拟环境，
  不代表外部服务可靠性或失败后的自动恢复。
- P6 tool failure/replan 窄 Gate 已通过：首次工具失败产生 prediction error=`2.0` 并触发 `replan_required`，随后 planner 选择 recovery
  tool，成功后清除重规划，两个工具 Outcome 均保留在 episodic memory。该 Gate 证明既有因果重规划可承接工具失败，不证明外部服务
  可靠性或通用长程规划。
- P6 unseen-tool/parameter transfer 窄 Gate 已通过：未见工具名 `maps.search.v42`、嵌套参数、重排 key 顺序均保持并成功执行；同时修复
  `act(world_action=...)` 丢失结构化参数的问题，保留通用参数与兼容 action metadata。该结果关闭固定工具表与扁平参数假设，不证明广泛
  工具生态或语言泛化。
- P6 cross-organ expression consistency 窄 Gate 已通过：同一 `ContentPlan` 同时生成 tool 与 text 结构化表达，`content_id`、semantic slots、
  confidence 和 goal provenance 保持一致，只改变 modality/channel。该结果证明表达器不夺取目标/计划所有权，不证明语言流畅性。
- P6 learned content selection 窄 Gate 已通过：可学习 utility 在相同候选下按 world uncertainty 在 `answer`/`ask` 之间切换，选择结果与
  semantic slots 可转成 `ContentPlan`，checkpoint 后选择保持一致。该 Gate 证明内容选择不必原样复制 `ActionIntent`，但尚未接入 adapter
  runtime，也不证明开放域语义生成。
- P6 runtime content-selection ownership 窄 Gate 已通过：adapter 从当前 goal/world state 选择 content，生成 `ExpressionPlan`，并在
  native checkpoint 恢复 selector、decision 与表达结果。该 Gate 关闭独立模块漂移，但 selector 仍需真实 Outcome 在线 credit assignment。
- P6 online content credit assignment 窄 Gate 已通过：真实 adapter reward 对已选 semantic content 执行一次 utility 更新；失败候选被降权、
  成功候选被提升并迁移，prediction error、training step 和 applied 标记均可 checkpoint。该 Gate 证明反馈回路存在，不证明开放域
  语义学习或长期概念形成。
- P6 holdout content transfer 窄 Gate 已通过：训练未见的 `forecast_digest`、新候选 ID 与嵌套 slot 结构仍按 learned context utility 被选中，
  checkpoint 后保持。该结果关闭候选名/intent kind/slot shape 固定表假设，不证明开放域语义发明。
- P6 text organ codec 窄 Gate 已通过：holdout `ContentPlan` 经 text expression UTF-8 codec 后，semantic slots、confidence 和
  `source_goal_id` 无损恢复；这只证明结构化文字器官边界，不证明自然语言流畅性、句法或语言智能。
- P6 terminal language-organ boundary 窄 Gate 已通过：可替换的 `LanguageOrgan` 只接收 Taiji-owned `ExpressionPlan`，默认
  `structured-stub` 输出可回解码文本；detached-organ lesion、native checkpoint 和参数/认知不变性均通过。该结果只证明末端
  器官所有权与替换边界，不证明自然语言流畅性、句法或 decoder 智能。
- P6 language backend registry/training contract 窄 Gate 已通过：registry 可登记未来成熟 decoder，但强制 text modality 与
  `owns_cognition=False`；训练样本固定为 `ExpressionPlan → target_text`，可独立 checkpoint/holdout，不把目标、记忆或 ActionIntent
  注入 decoder。该结果只证明接入/训练数据边界，不证明 decoder 能力。
- P6 external decoder realization/lesion 窄 Gate 已通过：`ExternalTextDecoderLanguageOrgan` 通过注入的 prompt builder 调用外部
  `generate()`，输入仍只有 Taiji-owned `ExpressionPlan`；detached-organ lesion 通过，且 Taiji 核心未导入 Legacy/Transformer。
  该结果只证明外部适配器边界，不证明具体模型已加载、训练质量或自然语言流畅性。
- P6 decoder provider inventory 与真实 provider smoke Gate 已完成：当前项目有 `0` 个 `data/neurons` Legacy 权重、`4` 个
  `seed-native-v1` 原生 checkpoint 和 `11` 个 Legacy tokenizer 文件；但本机 Hugging Face 缓存提供 Qwen2.5-0.5B-Instruct 权重与
  tokenizer。该 provider 已通过真实 `generate()`、非空文本、detached-organ lesion、认知不变、registry checkpoint 和训练合同
  Gate；这只是外部 provider smoke/ownership 结果，不证明语言质量或通用智能。
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
- P6 Seed client provider startup Gate 已通过：`SeedConfig` 提供 structured/raw/LoRA/guarded 的产品侧选择，默认只装配
  `structured-stub`；显式 provider 由 Seed runtime 启动链路调用 artifact loader，缺失、可选依赖缺失、manifest mismatch 和其他
  加载异常都会回退到 structured-stub，并通过 `/api/health` 与 `/api/runtime/status` 暴露 `language_provider` 状态。Seed 静态边界
  不绑定 Transformer，guarded 仍强制显式 opt-in。
- P6 client observability Gate 已通过：frontend runtime store 保存 `language_provider`，聊天页和异常中心可显示 active/fallback、
  回退原因与 structured-stub 恢复状态；前端只观察 runtime，不参与 provider 选择、认知决策或 decoder 装载。前端构建通过，Vitest
  `160 passed`。
- 原生 `tests/taiji_native` 最近一次完整执行为 `192 passed, 1 skipped, 2 errors`；两个 error 均发生在
  Windows pytest 临时目录锁创建阶段，未进入测试体，不作为代码断言失败或能力结论。
- P2 感知训练已改为复用运行时的动态边界时钟：训练按同一 adaptive assembly 起点监督每个活动前缀，
  不再使用与运行时不一致的固定滑窗；CUDA 实际 profile 暂缓到具备 CUDA 主机后再做，不阻塞当前 CPU 开发。
- A1 评测已使用 marker-specific boundary evidence，并要求所有 seed 的最差指标共同满足 Gate；最新
  `reports/taiji_a1_perception_20260827.json` 的 smoke Gate 在 32/16 manifest 上通过，独立的
  `reports/taiji_a1_perception_shared128_20260827.json` 也在 128/64 manifest 上通过，P2 感知纵切片的
  组合迁移、边界响应、random-chunk lesion 和变量时长合同已满足当前门槛。
- P2 默认训练包含低权重多步 predictive credit（weight=`0.05`, horizon=`4`），并新增只针对真实闭合
  boundary 后续 assembly 的跨段负样本对比目标（`cross_assembly_negative_weight=0.01`）；边界后逐符号
  CE 保留为显式可选实验项，默认关闭以避免重复监督。shared128 最差泛化=`0.0`、最差 random-chunk
  drop=`+0.00527`、marker score/rate 最小=`+0.2161/+0.4483`、cross-seed std=`0.00834`，Gate 为 true。
- P2→P3 lineage contract 已落地：`PerceptEvent` 的 event/assembly 来源与 `boundary_closed` 状态
  同时进入 `WorkspaceState` 和 `WorldState`；外部环境替换 world state 时不丢失当前感知 lineage，
  native checkpoint 往返保持一致，相关定向回归 `21 passed`。这只收口来源可审计性，不等于 perception-to-world
  的 holdout 能力已经通过。
- P2→P3 perception-to-world closure Gate 已通过：`reports/taiji_p2_p3_closure_20260827.json`
  在 64 train / 32 新对象与新候选组合 holdout、3 seeds 上，learned route/world transition 最差均为
  `1.0`，none workspace lesion 最高为 `0.0`，lineage 最差为 `1.0`，192 次 boundary-closed assembly
  与 3/3 checkpoint continuation 全部成立；`shared16` relation subgate 复核仍为 true。该 Gate 证明
  runtime provenance 与窄 world transition 已闭环，不等于长程世界建模或开放域语义理解。
- 2026-08-26 门禁与 checkpoint 收口：CI 因 pin 了不存在的 `black==24.12.0` 连续 8 天红灯且期间所有门禁被静默跳过，已改钉
  `ruff==0.16.4` / `black==26.5.1`；`TSKV8Adapter.checkpoint()`/`restore()` 补齐 `cognitive_state` 往返后，全量测试为
  `437 passed, 5 skipped`。门禁可信度、mypy 类型债、checkpoint 往返不变量和本目录编制纪律见总路线第 14.1–14.4 节。

## 当前唯一下一步

当前唯一下一步只看 [SEED_DEVELOPMENT_ROADMAP_2026_08.md](active/SEED_DEVELOPMENT_ROADMAP_2026_08.md) 第 16 节，本文件不再复制该结论。
按总路线第 14.4 节，「当前唯一下一步」在全仓只允许有一个权威源；此处保留指针是为了避免再次出现相互竞争的下一步表述。
