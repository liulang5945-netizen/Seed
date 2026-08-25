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
- 原生套件当前 94 passed、1 skipped；另 2 个旧 manifest 测试在本机 pytest 系统临时目录创建阶段受 Windows 权限影响，
  尚未把该环境问题误记为代码通过。

## 当前唯一下一步

继续 P6 learned content selection Gate：让 content planner 从 goal/world/context 选择可解释 semantic slots，而不是直接复制
`ActionIntent`；保留当前 P3/P4/P5 Gate，不让 byte motor 或 Legacy Transformer 代替目标规划与内容决策。
