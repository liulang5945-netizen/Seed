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
| [SEED_DEVELOPMENT_ROADMAP_2026_08.md](active/SEED_DEVELOPMENT_ROADMAP_2026_08.md) | 唯一执行顺序：P0–P8 既有成果、Workbench Closure W0–W7、后续可靠性/研究/性能/体验工作包和当前下一步 |
| [ARCHITECTURE_DIRECTION_2026_08.md](active/ARCHITECTURE_DIRECTION_2026_08.md) | 规范词表、不可回退边界、成熟技术采纳规则和 Legacy 边界 |
| [SEED_ARCHITECTURE.md](active/SEED_ARCHITECTURE.md) | Seed 产品/runtime 所有权、允许/禁止职责与 checkpoint/API 迁移边界 |

`plans/active/` 只保留以上五份文档。发生冲突时，项目使命以核心需求为准，目标能力以 Taiji v1 架构为准，执行顺序以总路线为准，身份和依赖以方向决策为准。

总路线 W7 已完整保留 provider watchdog、interaction-group、视觉体验、CUDA、开放域学习和结构自进化；它们不是取消或归档，
而是按真实工作台 trace、稳定合同和硬件可用性设置进入条件。小型模拟 Gate 从 W0 起继续使用，但只作为 S0 机制证据，必须逐步升级到
replay/sandbox 的 S1 和 packaged-client/real-workbench 的 S2，不能单独替代产品完成声明。

## 旧实现与旧路线

- [TAIJI_SUBSTRATE_KERNEL_V8_SPEC.md](archive/implementation/TAIJI_SUBSTRATE_KERNEL_V8_SPEC.md)：TSK-v8 的精确方程、tick、局部学习、checkpoint 和旧 N/M 门槛。
- [SEED_DEVELOPMENT_ROADMAP_PRE_NATIVE_V1_20260825.md](archive/history/SEED_DEVELOPMENT_ROADMAP_PRE_NATIVE_V1_20260825.md)：架构纠正前的 R0–R7/S1–S3 路线。
- [SEED_STAGE_CLOSEOUT_20260825.md](archive/history/SEED_STAGE_CLOSEOUT_20260825.md)：上一阶段工程、产品和 kernel 成果收束。
- [archive/README.md](archive/README.md)：完整归档索引。

归档中的“当前状态”“下一步”和“完整 Taiji”声明全部按历史语境解释，不再指导开发。

## 当前代码事实

- `taiji/` 不导入 `seed`、`neuroplex` 或 `transformers`；该独立性继续保留。
- `seed/` 当前包装 `Taiji` kernel；P1 compatibility adapter 已迁移首个 v1 纵切片，不破坏产品 API 和旧 checkpoint。
- P6 client input-boundary Gate 已通过：`InputFrame` 版本化承载客户端原始 bytes 与来源元数据，`TSKV8Adapter.ingest_input()` 将其逐字节转换为 Taiji-owned `Observation/PerceptEvent`，`InputTrace` 可检查并 round-trip；`ActionIntent` 保持为空，未引入固定意图映射。`SeedRuntime.chat` 已通过 `generate_input()` 走同一合同，并在产品出口经本地 `native-readable` 语言表层形成可读文本；raw-byte 仍只保留为底层兼容/调试信息。
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
- 2026-08-28 产品执行平面审计已确认：以上 `ActionIntent/ToolCall/Outcome` 研究合同尚未接入 Seed 自带 IDE。`SeedRuntime.chat` 只返回文本，工作台/API 被错误挂在 Legacy optional router 下，原生 runtime 上报空工具列表，Monaco 编程语言只存在于前端局部状态；因此当前不能宣称模型可自主操作工作台或自主选择编程语言。
- 同轮确认产品面仍有 GGUF/LoRA 发布、Cortex 热切换、Legacy ReAct/Agent 设置、旧 raw-byte 完整架构文案和 MCP 前后端合同漂移。合法的 HF/Qwen/Transformers 使用只保留在数据与语言 provider 集成边界，不再作为 Taiji 核心格式或认知主体切换。
- 主线已从 provider watchdog/interaction-group 深挖重排为 Workbench Closure W0–W7；W0 的 native 合同、只读环境、core router、runtime capability projection、审计链、前端 projection、checkpoint continuation 和 packaged-client canary 已闭合，下一步进入 W1 IDE 编程语言识别与自主切换；写入/终端、MCP、自主循环和产品残留迁移仍按 W2–W4 顺序执行。
- 2026-08-29 W0 首批纵切片已闭合：`ActionIntent → ToolCall → WorkbenchEnvironment → Outcome` 能在真实仓库文件上执行，过期 snapshot、路径越界、错误值域和断开环境均 fail-closed；checkpoint continuation、OpenAPI、编译、lint 和前端回归通过。
- 2026-08-29 W0 前端与打包投影已回归：`WorkspaceView` 使用 native capability 懒加载目录，`MonacoEditor` 使用 native read 打开文件，`editor.open` outcome 可通过统一 audit projection 驱动 IDE；`SEED_ENABLE_LEGACY=0` 的真实 `Seed.exe` packaged canary 对 native capabilities/files 均返回 200。写入、终端、重命名仍明确停留在 Legacy 未接入边界；GUI 视觉/DPI/托盘验收保留到 W5/W7-R3。
- 2026-08-29 打包根因已修复：PyQt6 的 Qt6Core 被机器学习依赖带入的 ICU 78 同名 DLL 覆盖，导致 frozen `QtCore` WinError 127；`desktop/seed.spec` 过滤冲突 ICU、桌面启动不再预加载错误 ICU，release 产物重新通过客户端 canary。
- 2026-08-29 W1 语言识别、选择与 IDE 自主切换退出 Gate 已通过：`ProgrammingLanguageRegistry` 以内容寻址规则统一扩展名、shebang、内容、manifest、邻近文件、可选 LSP 和 toolchain 证据；`programming_language_id/editor_language_id`、confidence、provenance、registry revision、explanation 与用户覆盖进入 Workbench state。`editor.set_language` 在高置信且与证据一致时允许 Taiji 自动切换，低置信、语言冲突或 `.h` 等歧义场景返回 `ask_user`；runner/LSP 上下文和可用工具链快照与同一语言选择绑定，显式用户覆盖可撤销且按文件 digest 失效，checkpoint 不会把旧覆盖套到新内容。Monaco 已提供“自动检测”入口。holdout 覆盖 `.h`、无扩展 shebang、多语言 monorepo、Vue/TS、notebook、markdown code block、错误扩展名和 filename-only lesion；后端 Workbench 合同 `8 passed`、Monaco 回归 `10 passed`，前端 lint `0 errors`、构建通过。该 Gate 是语言/IDE 合同闭环，不代表 W2 runner 已可执行。
- 2026-08-29 W2 首批受控执行合同已落地：native capability snapshot 升至 revision 3，新增 `workspace.apply_patch/create/rename/delete/undo` 与 `terminal.run`；文件写入统一采用 UTF-8 结构化 text-replace、before/after SHA-256、原子替换、冲突 fail-closed 和唯一单次 undo token，创建/重命名/删除复用同一事务返回；终端只接收 argv，明确 `shell=False`，并绑定 workspace cwd、超时、输出上限、环境变量 allowlist 和 expected artifacts，非零退出/超时会进入失败 outcome。runtime 已保留真实 transaction payload，旧只读能力继续使用兼容投影。文件事务、终端边界、审批策略与失败结果回归共 `11 passed`，ruff、Black、py_compile 与 diff check 通过。该 slice 只闭合 executor/contract，写入和终端在 runtime policy 中仍默认 `ask_user`，尚未宣称 IDE 预览/审批 UI 或 diagnostics/test/build 回写完成。
- 2026-08-29 W2 第二 slice 的审批与结果闭环已落地：`/api/workbench/preview` 对精确的 action request 做不落盘验证并生成短期一次性 approval token，`/api/workbench/execute` 只有携带同一请求绑定的 token 才能执行高风险写入/终端，重放、过期、参数或 snapshot 漂移均 fail-closed；审计请求只记录 approval presence，不泄漏 token。`terminal.run` 增加 command/diagnostics/test/build execution kind、结构化 diagnostics、expected artifacts、after-state 和基于超时/退出码/诊断错误/缺失产物的综合 success。runtime 与通用前端 projection 已接入 preview/execute client；后端 Workbench 合同 `12 passed`、前端完整回归 `187 passed`、构建通过、ruff/Black/py_compile/diff check 通过。checkpoint 后的 undo/approval 状态、真实临时项目端到端续跑和重复副作用 Gate 仍未完成。
- 2026-08-29 W2 退出 Gate 已通过：transaction state 随 SeedRuntime checkpoint 保存并恢复 undo lineage，approval token 明确为 session-scoped、重启后失效；恢复后重新预览/审批可完成撤销，临时多文件项目完成语言识别→patch 预览/执行→test 产物→diagnostics 失败回写链路，预览无副作用、冲突/输出洪泛/cwd 漂移/超时/进程中断均有 fail-closed 证据。旧 `runtime_service` 边界测试同步到 native capability 事实，Seed/native 回归 `320 passed, 1 skipped`；W2 具备进入 W3 的证据，不代表 MCP 或有限自主循环已完成。
- 2026-08-29 W3 第一纵切片已通过：新增 Seed-owned `McpToolRegistry`，以内容寻址 registry snapshot、版本化 input schema、source/risk/timeout/output budget 和 registry revision 统一 MCP-shaped 工具合同；native Workbench 新增 `mcp.list/invoke`，仅接入无安装、无网络副作用的本地 workspace-summary canary，参数漂移、未知/禁用工具、schema 错误和输出超限均 fail-closed，动态高风险工具仍必须走审批。`/api/workbench/capabilities`、`/api/workbench/mcp` 和前端 projection 已接通；没有接回 Legacy `mcp_manager`，也没有宣称外部 MCP 管理或多步自治循环。Workbench 定向回归 `18 passed`，前端 `187 passed`、构建通过、ESLint `0 errors/17 warnings`。
- 2026-08-29 W3 第二 slice 已通过：MCP registry 内容身份随 runtime checkpoint 保存/恢复，单次 Workbench request、Outcome 和返回 ToolCall 共享 capability snapshot/registry snapshot binding；`/api/workbench/loop/preflight` 与前端 `preflightLoop` 已接通，loop 只做不执行 admission，强制最多 8 步、总预算不超过 32 units、拒绝重复调用、首错终止和 `after_each_step` checkpoint 边界。Workbench 定向回归 `19 passed`，Seed/native 全量回归 `320 passed, 1 skipped`，前端 `187 passed`、构建通过、ESLint `0 errors/17 warnings`；真正的多步执行、逐步 checkpoint 提交和外部 MCP 生命周期仍未宣称完成。
- 2026-08-29 W3 第三 slice 已通过：新增 `/api/workbench/loop/execute` 与前端 `executeLoop`，只接受 preflight identity 未漂移的 native Workbench request；每个已尝试步骤都真实执行、写入 ToolCall/Outcome audit 并保存 checkpoint，遇到失败立即停止并保留已完成前缀，恢复后重放已提交 request 会 fail-closed。真实成功两步、失败停机和 checkpoint 恢复重放定向回归 `21 passed`，Seed/native 全量回归 `320 passed, 1 skipped`，前端 `187 passed`、构建通过、ESLint `0 errors/17 warnings`；尚未扩展到跨文件 patch/test/diagnostics 任务，也未接入外部 MCP 生命周期。
- 2026-08-29 W3 退出 Gate 已通过：真实临时项目完成语言识别→跨文件 patch→test/build 产物→diagnostics 失败→checkpoint 恢复→创建修复标记→diagnostics 重试；失败后只从未提交步骤继续，已提交 request 的旧审批令牌即使失效也会先被 checkpoint 提交历史拒绝，避免误报为普通审批失败或重复副作用。去掉 Taiji planner 或 WorkbenchEnvironment 任一侧的路径均 fail-closed；该 Gate 不接外部安装、网络服务或开放式自治。Workbench 定向回归 `22 passed`，Seed/native 全量 `320 passed, 1 skipped`，前端 `187 passed`、生产构建通过、ESLint `0 errors/17 warnings`。
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
- P6 terminal language-organ boundary 窄 Gate 已通过：可替换的 `LanguageOrgan` 只接收 Taiji-owned `ExpressionPlan`，产品默认的
  `native-readable` 表层会保留有效候选或生成诚实的可读状态文本；`structured-stub` 降为显式无损调试 codec。detached-organ
  lesion、native checkpoint 和参数/认知不变性均通过。该结果修复产品乱码/RAW 冒充语言的边界，但只证明可读表层，不证明自然
  语言流畅性、开放域语义回答或 decoder 智能。
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
- P6 Seed client provider startup Gate 已通过：`SeedConfig` 提供 native/structured/raw/LoRA/guarded 的产品侧选择，默认装配
  `native-readable`；`structured-stub` 仅在显式 debug mode 使用。显式 provider 由 Seed runtime 启动链路调用 artifact loader，缺失、
  可选依赖缺失、manifest mismatch 和其他加载异常都会回退到 `native-readable`，并通过 `/api/health` 与 `/api/runtime/status`
  暴露 `language_provider` 状态。无论 provider 状态如何，产品聊天默认使用本地语言表层，不会静默把用户历史转发给外部 decoder；Seed
  静态边界不绑定 Transformer，guarded 仍强制显式 opt-in。
- P6 client observability Gate 已通过：frontend runtime store 保存 `language_provider`，聊天页和异常中心可显示 active/fallback、
  回退原因与 `native-readable` 状态；聊天 final event 额外暴露实际 `language_backend`，前端只观察 runtime，不参与 provider 选择、
  认知决策或 decoder 装载。前端构建通过，Vitest `160 passed`。
- P6 `ExpressionPlan → target_text` realization admission Gate 已实现并通过真实本机 Qwen CPU 复核：新的
  `LanguageRealizationGate` 同时检查 train/holdout 隔离、可读性、必需语义词覆盖、结构化泄漏、fallback、精确回滚与
  adapter checkpoint continuation；真实 4 epochs/16 steps 的 270336 个 LoRA 参数在保存后重新加载，全部条件通过。
  产品聊天只有在 `mode=guarded`、`chat_enabled=true` 且训练/安全报告均通过时才接入外部 decoder；旧报告缺少新 Gate
  证据会 fail-closed，默认仍使用 `native-readable`。该结果证明准入边界，不宣称开放域语言智能。
- P6 provider artifact 内容寻址与首轮 chat canary Gate 已通过：artifact 为 base model、LoRA、训练语料、训练报告和安全报告记录路径无关的
  SHA-256 内容摘要与稳定 manifest digest，guarded product chat 加载前拒绝内容漂移、缺失、manifest 不一致和过期 artifact；加载后固定两条
  canary 要求 `数据库/正常`、`接口/恢复` 完整语义覆盖、可读、无结构化泄漏且无 validated fallback。任一条件失败均回退到
  `native-readable`，并暴露区分的 artifact/canary failure code；旧 artifact/checkpoint 仍可读取，但缺少内容寻址证据时不能进入 product chat。
  训练侧 artifact loader smoke 同步输出摘要和 canary 结果，定向回归 `23 passed`，Ruff、Black、核心 Mypy=`0`。本机全量 Seed/Taiji 测试的
  测试体未见本次回归，但仍受既有 Windows worktree/pytest 临时目录 ACL setup/cleanup error 影响，未计为全量 Gate 通过；CUDA 继续暂缓。
- P6 provider artifact 多版本 registry 与原子轮换 Gate 已通过：Taiji 新增只保存 manifest 的版本 registry，要求 artifact ID 唯一、显式
  allowlist、active/previous 版本关系和单调 revision，并纳入 native checkpoint continuation。Seed 轮换在隔离 staging adapter 中加载候选，
  先执行内容寻址、训练/安全 Gate 和首轮 chat canary，全部通过后才一次性提交 language organ、backend registry、artifact 和 registry snapshot；
  版本冲突、未授权版本、候选加载失败或 canary 失败均保持线上旧版本不变，不会半写入。新增 `SeedRuntime.rotate_language_provider`，让产品层更新
  provider runtime 时同步保持锁和旧 runtime；定向语言/provider 回归 `25 passed`，Ruff、Black、核心 Mypy=`0`，CUDA 继续暂缓。
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
- P2→P3 variable-horizon continuation Gate 已通过：`reports/taiji_p2_p3_variable_horizon_20260827.json`
  在 64 train / 32 holdout、3 seeds、3/4/5 个 closed assembly 上，learned route/world success、
  lineage、两步 history、checkpoint continuation、TaijiWorldState roundtrip 和 runtime world
  learner calibration 均为 `1.0`，workspace lesion route 为 `0.0`；第二步使用训练 schema 未见的
  `secured` relation，并由 `assembled → secured` progression Gate 明确验收。该 Gate 证明变量时长
  与跨 checkpoint 的两步因果续接已成立，但未知关系目前只保证被保存和传递，不宣称 world learner
  已完成开放集关系预测。
- 2026-08-26 门禁与 checkpoint 收口：CI 因 pin 了不存在的 `black==24.12.0` 连续 8 天红灯且期间所有门禁被静默跳过，已改钉
  `ruff==0.16.4` / `black==26.5.1`；`TSKV8Adapter.checkpoint()`/`restore()` 补齐 `cognitive_state` 往返后，全量测试为
  `437 passed, 5 skipped`。门禁可信度、mypy 类型债、checkpoint 往返不变量和本目录编制纪律见总路线第 14.1–14.4 节。

**已完成（2026-08-29）：W4 第一 slice 的正式产品语义残留清理。** 前端已移除 GGUF 导出、旧模型发布、Agent 参数配置、Cortex
认知主体切换和旧 Agent 日志筛选；能力页只展示 Seed-owned native capability registry，不再提供 Legacy MCP 安装/市场管理入口；聊天请求
不再携带旧 engine/迭代/温度参数，生命状态页不再调用历史生命动作接口，训练 composable 的旧生命动作、发布和 GGUF 死路径已删除。
合法的 native MCP projection、语言 provider artifact 和离线 benchmark 未被误删。前端完整 Vitest 为 `22 files / 188 passed`，ESLint 为
`0 errors / 15 warnings`，生产构建通过。

**已完成（2026-08-29）：W4 第二 slice 与退出 Gate。** 后端已建立统一 artifact 词表和 `/api/artifacts`，以 `/api/runtime/activate`、
`/api/settings/runtime` 替代全局 model switch；settings schema v2 会安全迁移明确的 native checkpoint，旧 model/GGUF/HF/LoRA/量化
语义进入可审计 quarantine，旧路由返回版本化 410 并从默认 OpenAPI 隐藏，Legacy 依赖仅在 `SEED_ENABLE_LEGACY=1` opt-in。`ChatRequest`
已删除 `engine/agent_*`；完整 Python CI `526 passed, 6 skipped`，provider artifact/canary、Taiji Transformer 隔离和 Legacy-off
冒烟 Gate `34 passed`。

**已完成（2026-08-29）：W5 第一 slice 的客户端真实性审计与旧调用清理。** Workspace、KB、Chat、Life、Settings 已完成 native 边界对齐：旧 workspace/RAG/生命状态调用和失效文案已清理，副作用统一通过 Workbench preview→approval→execute，Seed active 时 `is_taiji` 正确上报。
KB 默认不再调用 `/api/rag/*`，仅依据 runtime snapshot 判断 `knowledge.*` capability；Monaco 文件写入携带 digest/编码/完整性检查，未登记的目录创建和资源管理器入口已移除；`useWorkspaceBridge.js` 已删除。证据为 frontend Vitest `22 files / 185 passed`、ESLint `0 errors / 13 warnings`、生产构建通过，后端原生 Workbench/系统路由回归 `72 passed, 1 skipped`，ruff/compileall/diff check 通过。尚未宣称 knowledge capability 已实现，也尚未进行 packaged route-level smoke。

**已完成（2026-08-29）：W5 第二 slice 的真实状态接入。** `/api/runtime/status.tools` 已补齐 Workbench snapshot 的身份、版本、来源、归属和观测时间；前端统一投影 runtime、provider、Workbench、homeostasis/self-state、training、knowledge 的 `source/owner/freshness/availability`，并接入 Chat、Life、Agent/能力、Training、Settings、KB 六个入口。homeostasis/self-state 缺失时明确显示“未上报”。前端 Vitest `23 files / 187 passed`、ESLint `0 errors`、生产构建通过；后端定向回归 `65 passed, 1 skipped`，OpenAPI、ruff、compileall、diff check 通过。

**已完成（2026-08-29）：W5 第三 slice 的 packaged route-level smoke 与 frontend/source capability contract。** 前端源码门禁已固定六个产品入口的状态证据投影，并禁止 Legacy RAG/model/HF/GGUF/engine 旧路径回流；生产预览逐路由 smoke 精确巡检 7 路由、错误页、6 个证据页面、导航、聊天/训练交互与移动端，共 `35 项 / 0 失败`。后端 Legacy-off 启动合同、frontend `23 files / 187 passed`、ESLint、生产构建、native boundary、ruff、compileall、diff check 均通过。

**已完成（2026-08-29）：W6 第一 slice 的 OpenAPI→frontend endpoint contract。** OpenAPI 快照测试已覆盖 method、query
parameters 和 request body 漂移；前端 `check-api-contract` 已校验 83 个 API 字面量及直接调用的 method、查询参数和 JSON 顶层字段，
并接入 CI。未使用的 Legacy 上传 composable 已删除，`select_folder` 标题参数已正式进入后端合同与快照；后端 `67 passed, 1 skipped`、
前端 `23 files / 187 passed`，native boundary/API contract、ESLint、构建、Ruff、compileall、diff check 全部通过。

**已完成（2026-08-29）：W6 第二 slice 的 typed/native API facade 第一批迁移。** 新增 `nativeApi` facade，集中维护 runtime、
Workbench 和 system 三组 native 端点、查询参数构造、JSON 序列化和错误解包；Workbench projection、runtime health/status、Workspace
快速路径、PathSelector 已切换到命名操作。facade 路径已纳入 API contract 门禁，并新增 4 项单测；前端 Vitest `24 files / 191 passed`、
API contract/native boundary、ESLint、生产构建全部通过。

**已完成（2026-08-29）：W6 第三 slice 的 typed/native API facade 第二批迁移。** settings、auth、chat、training 和 App 健康/版本入口
已切换到 `nativeApi`，普通 JSON 的 URL、method、请求体序列化和错误解包不再散落在页面/Store；流式聊天、附件上传、训练原生训练和检查点
恢复保留为显式 raw-response 方法。前端 Vitest `24 files / 191 passed`、API contract/native boundary、ESLint、生产构建全部通过。

**已完成（2026-08-29）：W6 第四 slice 的特殊传输边界收口。** 训练产品页改用 `nativeDatasetUpload` 语义开关，native dataset 上传通过
命名 facade 操作，旧 endpoint prop 仅保留在隔离兼容路径；chat/训练 SSE 与 FormData raw-response 契约测试已覆盖取消、非 2xx、空 body
和 multipart body。前端 Vitest `24 files / 194 passed`、API contract/native boundary、ESLint、生产构建全部通过。

**已完成（2026-08-29）：W6 第五 slice 的 WorkspaceView 第一阶段结构拆分与最小观测点。** 路径选择对话框已拆为
`WorkspacePathDialog`；父视图继续负责 native 工作台状态、路径切换和 preview→approval→execute 协调。native API facade 已新增请求数、
成功/失败、最后状态和延迟 snapshot；Vitest `25 files / 196 passed`、API/native boundary、ESLint、生产构建全部通过。

**已完成（2026-08-29）：W6 第六 slice 的 WorkspaceView 文件树展示拆分。** 文件树渲染、展开/折叠图标、工具栏和树节点事件转发已抽为
`WorkspaceFileTree`；父视图继续唯一拥有目录加载、文件读写、快捷打开、编辑器联动和 native mutation 状态。Vitest `26 files / 197 passed`、
API/native boundary、ESLint、生产构建全部通过。

**已完成（2026-08-29）：W6 第七 slice 的 WorkspaceView 编辑器/终端协调拆分。** 编辑器区域、终端显示、终端尺寸事件和 Monaco 保存
事件转发已抽为 `WorkspaceEditorPane`；父视图通过显式 expose 引用继续读取当前文件、保存状态和编辑器动作，仍保留 native approval handler
与 mutation 流程。前端 Vitest `27 files / 198 passed`、API/native boundary、ESLint、生产构建全部通过。

**已完成（2026-08-29）：W6 第八 slice 的 SettingsView 运行环境区拆分。** 运行环境状态与未认证终端开关已抽为
`SettingsRuntimePanel`；父视图继续唯一持有 `nativeApi` 设置持久化、GET/POST 竞态防护、失败回滚和运行时健康探测。新增组件级开关事件回归，
前端 Vitest `28 files / 199 passed`、API contract/native boundary、ESLint、生产构建全部通过（ESLint `0 errors / 13 warnings`）。

**已完成（2026-08-29）：W6 第九 slice 的 nativeApi trace/SLO 观测面板。** 新增 `RuntimeApiMetricsPanel`，由 facade snapshot 只展示请求路径、
请求/成功/失败计数、平均延迟和最后状态码，不采集或展示请求正文；`RuntimeEvidenceStrip` 仅在 Settings 显式开启该面板，并在组件卸载时清理刷新计时器。
前端 Vitest `29 files / 201 passed`、API contract/native boundary、ESLint、生产构建全部通过（ESLint `0 errors / 13 warnings`）。

**已完成（2026-08-29）：W6 第十片的 TrainingView 训练概览拆分。** Loss 画布、真实进度、吞吐/ETA 指标和检查点列表已抽为
`TrainingOverviewPanel`；训练启动、暂停、停止、SSE 状态和检查点恢复仍由父视图与 `useTraining` 负责，组件仅通过 props/events 连接。
父视图同步移除已迁移的概览样式与绘图 watcher。前端 Vitest `30 files / 203 passed`、API contract/native boundary、ESLint、生产构建全部通过
（ESLint `0 errors / 13 warnings`）；中途发现并修复了组件 props 与模块状态同名造成的 `vue/no-dupe-keys` CI 错误。

**已完成（2026-08-29）：W6 第十一片的 TrainingView 数据集面板拆分。** 上传入口、文件表格、选择/删除/预览事件和样本预览已抽为
`TrainingDatasetPanel`；父视图继续唯一持有数据集加载、删除、预览和 native 上传协调，组件通过显式 props/events 连接。旧数据集专属样式已从父视图移除。
前端 Vitest `31 files / 205 passed`、API contract/native boundary、ESLint、生产构建全部通过（ESLint `0 errors / 13 warnings`）。

**当前唯一下一步：开始 W6 第十二片的 TrainingView 日志与训练控制展示区拆分。** 将日志展示、清空事件和暂停/恢复/停止控制隔离，
父视图继续唯一持有 SSE、训练状态和 native 操作；每片保持 `nativeApi` 单一入口、补齐组件回归并在提交前跑完整 CI，
不得先做视觉包装或 CUDA kernel。

## 当前唯一下一步

当前唯一下一步只看 [SEED_DEVELOPMENT_ROADMAP_2026_08.md](active/SEED_DEVELOPMENT_ROADMAP_2026_08.md) 第 16 节，本文件不再复制该结论。
按总路线第 14.4 节，「当前唯一下一步」在全仓只允许有一个权威源；此处保留指针是为了避免再次出现相互竞争的下一步表述。
