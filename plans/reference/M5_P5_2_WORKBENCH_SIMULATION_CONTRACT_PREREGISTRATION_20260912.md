# M5 P5.2 — Workbench 小型模拟合同 Gate 预注册（冻结版）

日期：2026-09-12。状态：**已冻结**（本文提交即冻结；冻结后不接预注册之外数据源、不读取 sealed）。
上游：P5.2 只读摸底（roadmap 已记，recon 提交 `970e04c9`）；P5.1 证据线已闭合（构造+真实双成立）。

## §1 问题与动机

roadmap P5 外围顺序第 2 项：「先支持模型产出类型化动作、识别文件语言、解释语言切换、preview/执行/undo；模拟和真实 Workbench 使用一致的动作合同」。摸底确认：真实 Workbench 合同（`seed-workbench-contract-v1`，18 capability）与语言识别面已完整、interaction-group 学习面已有，但 **P5.1 procedural readout 与合同/学习面之间的桥缺失**。

P5.2 回答两个问题：(Q1) **合同一致性**——模型产出的动作能否 100% 走与真实使用完全相同的 Workbench 合同路径（preview→approval→execute→undo），且语言识别/切换解释/undo 语义逐位复用真合同？(Q2) **能力迁移**——在 scripted IDE 场景上训练的类型化动作生成器，能否在未见场景上超越频率先验？

**核心设计决策：小型模拟 = 真实 Workbench 合同的受限实例**——不造平行合同。模拟环境就是 `WorkbenchEnvironment` 本身（临时 workspace 根目录）；模型侧动作生成器产出的 `ActionIntent` 经 `WorkbenchActionRequest.from_action_intent`（kind→capability_id 直通）绑定当前 `CapabilitySnapshot`，走同一条 preview→approval→execute→(undo) 代码路径。

## §2 设计

### 2.1 场景色料（runner 内确定性构造，不接外部数据源）

- **scripted IDE 场景**：每场景 = 初始 workspace 状态（确定性文件集 + 扩展名/内容语言信号）+ 目标描述文本（cue 来源）+ 真实动作序列（ground truth，kind = capability_id 全名：`workspace.read`、`workspace.programming_language.resolve`、`editor.set_language`、`workspace.apply_patch`、`workspace.undo`、`workspace.create`、`workspace.search` 等）。
- **分区**：train 40 / holdout 12 / a-gate 8 场景。a-gate 场景的动作词表 ⊆ train 词表（P5.1 范式），场景文件状态不同。
- **参数确定性生成**：动作参数从场景文件状态推导（patch 内容、目标语言名、路径等），保证 replica 逐位可复现。

### 2.2 动作生成器（P5.1 trainer 范式 × workbench 词表）

- procedural readout：`ProceduralSequenceLearner`（**hidden 64、epochs 250**——P5.1g 冻结构造常数沿用；cue = `SemanticArtifactKnowledgeEncoder` 编码的场景目标文本，384 维 MiniLM 锚定同 P5.1d/g）在 train 场景的 `EpisodicMemoryRecord`（episode=场景、tick=动作序、`action_intent.kind` = capability_id）上训练。
- 生成循环：readout 预测 next capability_id → `from_action_intent` 绑定 snapshot → `preview_tool` → approval（模拟审批者确定性签发，与真实 human approval 的差异如实披露）→ `execute_tool` → 需要撤销的场景走 `workspace.undo`（真实 token 事务）。
- 交互记忆沉淀双路：(a) taiji 侧 `EpisodicMemoryRecord`（训练 readout）；(b) interaction-group 侧投影 `InteractionTraceEvent`（owner_id=场景动作生成器标识，event_id/outcome_id/resource_cost 从真实 outcome 派生）→ `InteractionTraceEpisode`。

### 2.3 诚实边界

- 模拟审批者为确定性策略（非 human approval）——披露为「同一路径、不同审批主体」。
- `terminal.run` 副作用不可 undo（真实合同既有边界），场景避免依赖其撤销语义。
- scripted demo 语料为 L3 教师轨迹（构造场景），unseen = 同词表新场景（文件状态与目标不同）。
- `growth_admitted=false`、`can_promote=false` 贯穿。

## §3 九门

| # | 门 | 判据 |
|---|---|---|
| 1 | static_four_checks | py_compile / ruff / black / mypy 61 错基线持平零新增 / pytest `4 failed/1199 passed/6 skipped` 零新增 |
| 2 | contract_path_identity | 全部模拟动作 100% 经 `WorkbenchEnvironment` 真实代码路径执行（runner 内零平行实现）；outcome payload 携带 contract-v1 审计字段（request_id/intent_id/snapshot_id/approval 状态），形态与 `tests/test_workbench_contract.py` 基线一致 |
| 3 | action_typing_validity | 生成动作 100% 经 snapshot 绑定：capability 不在 snapshot 或参数非法时被合同拒绝并如实记录（区分「合同拒绝」与「绕过合同」）；被接受动作的 capability_id ⊆ snapshot 18 项 |
| 4 | language_resolve_reuse | 语言识别行为与 `ProgrammingLanguageRegistry` 直接调用逐位一致（resolved/ambiguous/override 三态各至少一场景验证） |
| 5 | language_switch_explanation | 每次 `editor.set_language` 产生 before/after + 证据原因的结构化叙事并沉淀为记忆记录（字段级断言） |
| 6 | undo_semantics | `workspace.undo` 后 workspace 状态逐位等于事务前快照（每场景文件内容+语言 checkpoint 断言） |
| 7 | readout_sanity | workbench 词表 procedural readout：train accuracy ≥ **0.9** 且 holdout > lesion（P5.1 consolidate 同构四分区） |
| 8 | interaction_trace_projection | episode→`InteractionTraceEpisode` 投影字段级无损（event_id/owner_id/episode_id/outcome_id/resource_cost 逐项断言）+ trace corpus 通过 `InteractionGroupEvaluator` 消费（不 raise） |
| 9 | transfer_and_budget | a-gate（8 未见场景）next-action accuracy − per-tick 多数动作基线 ≥ **0.15**（跨代 margin 常数）；replica 逐位一致；wall ≤ **600s** |

## §4 三态

- `workbench_simulation_contract_supported`：九门全过。
- `simulation_capability_insufficient`：机械与合同门（1–6、8）全过而门 7 或门 9 的迁移部分失败——能力不足的实质判据。
- `failed`：任一机械/合同门失败（构造/基建问题）。

## §5 冻结常数

margin `0.15`（跨代，独立于任何摸底观测）；wall cap `600s`；hidden `64`、epochs `250`（P5.1g 构造常数沿用）；场景分区 40/12/8；a-gate 词表 ⊆ train 词表。

## §6 产物顺序

1. `scripts/training/eval_taiji_p5_2_workbench_simulation_contract_gate.py`（静态四项先行）。
2. 执行落盘 `reports/taiji_p5_2_workbench_simulation_contract_20260912.json` + 九门对账。
3. 路线图同步 + 独立提交。

## §7 对账纪律

门值以报告为准逐项对账；禁止反向改判据、剪枝凑数、重跑挑结果；报告不覆写不改绿；不接预注册之外数据源、不读取 sealed；场景构造确定性（replica 逐位复现）。
