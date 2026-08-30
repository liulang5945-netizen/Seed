# Seed / Taiji 当前执行状态

> 快照日期：2026-08-30。本文件是“现在做什么”的唯一事实源；历史过程位于 `plans/archive/`，详细阶段定义见 [04_EXECUTION_PLAN.md](04_EXECUTION_PLAN.md)。

## 1. 当前能力状态

| 范围 | 状态 | 可以声明 | 仍不能声明 |
|---|---|---|---|
| W0–W3 Workbench 闭环 | 已完成基线 | 原生 workspace 证据、语言识别/IDE 切换、受控工具执行、Outcome、recovery、checkpoint continuation | 默认写入自治、无限循环、开放域自然语言工具选择 |
| W4–W6 产品边界 | 已完成基线 | Legacy/HF/GGUF/Transformer 退出 cognition 和前端主语义；native facade/客户端真实性 | Legacy 已物理删除、provider 是认知主体 |
| W7-R1 provider watchdog | S0/S1/S2 已完成 | provider artifact digest、健康隔离、checkpoint replay、native-readable packaged 观测 | 外部 provider artifact 的真实客户端轮换已完成 |
| W7-R2 interaction-group | S0/S1/S2 已完成 | 从真实 trace 归因互补/冲突/恢复，不硬编码神经元角色 | 已自动改写 executive、memory 或结构 |
| W7-R3 visual/desktop | S0/S1 + 页面证据完成 | 生命雷达、窄布局、前端/包字节一致、客户端真实状态投影 | Windows 任务栏、托盘、通知、高 DPI 已现场通过 |
| W7-R4 CUDA | `hardware-blocked` | CPU 基线与设备/checkpoint 合同仍有效 | CUDA 性能、数值一致性或自定义 kernel 已验证 |
| W7-R5-S0 学习通道 | 已完成 | 真实 Workbench Outcome 可进入 `record_executive_outcome()`；`learn=False` 冻结，`learn=True` 在线更新；checkpoint 保留计数与选择 | 知识已内化、外挂可删、效应器可注册、开放域自进化 |
| W7-R5A/R5B | R5A-S2-C 已通过真实双 seed/task slice；R5B-L0/S1 registry-backed Workbench dispatch、L1 candidate package、L2 shadow Gate、L3 resource/rollback Gate 已实现；L4 纯计算能力架构评审已完成并判定当前无候选可实施 | R5A 已具备 DTO、内容寻址 replay、原生学习器、真实 Workbench 纵向 holdout/lesion/recovery、跨 seed/task slice 聚合与独立可删评审；R5B 已具备 bundle/candidate 内容寻址、snapshot、审批前 shadow、disposer 约束、stale fail-closed、lifecycle checkpoint、request/approval registry snapshot 绑定、全量 enabled capability 覆盖、原子 replacement/rollback、candidate proposed→validated/rejected 分离、digest-only shadow observation、after-state/resource/policy/approval Gate、原子资源 reservation 与父账本恢复；三个 L0–L3 evaluator 均报告 `gate.passed=true`，L4 已形成独立审查边界 | 正常 CI 主机上尚未跑完整 Workbench 文件回归；没有任何纯计算执行体获准进入 registry；R5C 开放域成长尚未开始 |
| W7-R5C 开放域成长 | R5C-S0–S17 已通过定向 Gate：证据窗口、跨任务 pressure、candidate bridge、shadow validation、准入策略、atomic admission、稳定性/rollback、多步 continuation、真实 Workbench evidence 调度、按 stream 隔离 cooldown、候选延续、多候选仲裁、跨区域容量压力、可逆回滚、多区域真实批次生成与完整 batch 生命周期、单候选与多候选 replay validation artifact、独立 replay measurement owner、多轮 measured evidence continuation、artifact/measurement integrity | 已证明多区域真实 Workbench evidence 能在预算内形成可恢复、可 checkpoint 的多轮 measured candidate batch；区域之间不会因全局 scheduler cooldown 互相饿死，measurement 与 artifact 的内容寻址、篡改拒绝、旧格式兼容和 checkpoint provenance 已闭合；仍不能声明无限扩张、自动增加预算、全面自进化或真实开放域收益 |

最新证据数字与报告只在 [IMPLEMENTATION_STATUS_2026_08.md](../../reference/IMPLEMENTATION_STATUS_2026_08.md) 维护。

## 2. 当前阻塞与非阻塞边界

- **R3 Windows shell 为 `tool-blocked`。** Chrome 页面和窄布局证据已通过，但 Computer Use 无法激活 Seed 窗口，不能把桌面背景截图当任务栏/托盘/通知/DPI 证据。工具恢复后补证，不返工已通过页面层。
- **R3 页面层的两处渲染缺陷已由用户点名修复并收敛，不改变任何运行时语义，也不构成 R3 Gate 证据。** 训练页 `.tk-card h3` 内联 `<svg class="ic">` 此前在组件与全局样式里都无尺寸规则，按 SVG 默认 300×150 渲染并在 flex 标题内把「检查点列表」挤成逐字换行，现由 `TrainingOverviewPanel.vue` / `TrainingView.vue` 统一的 `.tk-card h3 .ic` 规则约束，并删除 `TrainingView.vue` 内同目的的 `style` 硬补丁（`97a3dac`）。生命状态页 `NeedsPentagram.vue` 的配色经六轮取舍定为「全中性主体」：轮廓、填充、引导环、轴线全部由 `--muted-foreground` 经 `color-mix` 派生（描边 52%/1.8px、填充 8%、引导环 12% 与最外环 26%、轴线 10%），整图唯一的非中性语义色是 `--danger`，且只出现在告警轴向（46%/1.6px）与告警数值上；顶点圆点整体移除，需求等级改由「轴线染色 + 数值三档字重/色」双通道编码（`> 70` alert / `40–70` watch / `< 40` calm），满足不依赖单一颜色传达语义，并规避了 warm 主题下 `--chart-1` 与琥珀色系撞色、以及三色相超出无图例配色预算的既有问题。同轮清除的失效或冲突写法：`<defs>` / `radialGradient` / `useId()` / `areaGradientId` / `.area-core` / `.area-edge` 渐变残留、CSS `r:` 几何覆写、`@keyframes critical-pulse`；数据面早前已由 `<polygon>` 改 `<path>` 使 `transition: d` 真正生效。视觉契约变更同步落到测试：`circle.pentagram-dot`×5 断言退役，替换为 `line.pentagram-axis`×5、`circle` 数为 0、告警轴 `critical` 类、以及 `tspan.pentagram-value` 的三档分级（含 70 与 40 两个阈值边界），`polygon.pentagram-guide`×5 与 `text.pentagram-label`×5 保持不变；`LifeStatusView.test.js` 与 `LifeNeedsDashboard.test.js` 经检索确认不依赖顶点选择器。前端回归 43 文件 247 例全绿（基线 245 例，新增 2 例为上述分级与轴向断言），ESLint 无告警，`npm run build` 成功。
- **R4 CUDA 为 `hardware-blocked`。** 当前主机没有可用 CUDA；不写自定义 fused/sparse kernel，不用 CPU 结果代替 GPU 结论。
- 两条阻塞线仍影响对应发布声明，但不再冻结无依赖的 R5 CPU/native 合同与实现。R5 的任何进展也不能反向把 R3/R4 标为通过。
- 训练或结构试验开始前必须先通过 checkpoint 保存→关闭→恢复→继续的阻塞 Gate；只在内存中成立的结果不进入路线。

## 3. 仓库收敛状态

- 当前 checkout 为 `main`；`output/` 是未跟踪的现场证据目录，本轮不暂存、不删除。
- `backup-local-20260828` 与干净的 `codex/interaction-group-credit` 已收束并删除；`codex/interaction-group-incremental` 仍附着含 5 个未提交文件的 worktree，未强行删除或混入主线。
- CI 基线由 `b6d1bf2` / 远端运行 `33295880356` 完成过全量验收；本轮 `9f30eb4` 接入 S1 canary 后，运行 `33298105636` 暴露了 Linux/Windows 最小 torch 环境缺 NumPy 时 `tensor.numpy()` 的跨平台失败。`513cb1f` 已改为纯 PyTorch 字节视图并补充无 NumPy contract；远端运行 `33298754868` 的 7 个 job 已全部成功。
- `plans/` 没有空目录或 0 字节文件。核心架构讨论留在 active/reference；已完成 Gate 过程和旧执行蓝图已移到 archive。

## 4. 当前唯一下一步

执行 **W7-R5C-S16：把 measured artifact batch 推进到多轮真实 evidence continuation 与可逆 rollback**。

R5-G1 与 R5A-S0 已完成并交付：

1. `plans/manifests/taiji_w7_r5_internalization_v1.json` 与 `plans/manifests/taiji_w7_r5_effector_registry_v1.json`；
2. `taiji/internalization.py`：不依赖 `seed_platform` 的 Outcome/evidence DTO、内容 digest、train-only replay、生命周期和五项因果门控；
3. `tests/taiji_native/test_internalization_contract.py`：未 grounding、缺 reward、越界、provider/capability 文本、holdout 写穿、重复 evidence、未通过 causal gate、checkpoint resurrection 的 red proof；
4. `tests/test_w7_gate_manifests.py`：合同边界与 R5A/R5B 分离关系；
5. R5A S0/S1 证据为 S0 定向测试 `14 passed`、S1 定向测试 `5 passed`、原生 canary `gate.passed=true`，以及本地 checkpoint/holdout/lesion 检查；R5B L0–L4 与 R5C S0–S12 的当前事实见下方对应条目。

R5A-S2-A 已完成：`api/seed_runtime.py` 只会在当前、校验过的 `workbench.evidence`、同一 capability snapshot 和由该 evidence 重投影出的 grounded successor affordance 同时成立时，创建 `GroundedOutcomeEvidence`。运行时不拥有 replay、learner 或 lifecycle 的写权；缺失/陈旧 snapshot 或非当前 affordance 全部 fail-closed。定向用例在真实只读 workspace 上通过，并已回归完整 Workbench 合同 `44 passed`。

R5A-S2-B 已完成：`taiji/internalization_longitudinal.py` 将真实 Workbench Outcome 转换成 train/holdout task slice，使用 train-only pairwise preference 更新，验证外部规则移除、内化特征/grounding lesion、旧任务 retention、checkpoint restore，并且只生成 `candidate_only_no_physical_deletion` 候选；真实 external artifact 未被删除，holdout 未进入 replay。

R5A-S2-C 已完成并通过真实集成：`InternalizationStabilityTrial` 按 seed 和 task slice 封装 S2-B 报告，`InternalizationStabilityGate` 检查跨试验收益、lesion、retention、指标离散度和 bounded resource counters，`IndependentDeletionReview` 单独检查 artifact/checkpoint/lifecycle/manifest 绑定，并拒绝任何 path、disposer、executor 或删除字段。真实 Workbench 集成用例以 seed 11 与 seed 29 执行通过；由于当前 pytest 临时目录受限，使用预创建的主机系统临时工作区运行了同一测试函数，结果为 `direct S2-C Workbench integration passed`。

R5B-L0/S1 已推进：`seed_platform/capability_registry.py` 将 capability bundle、executor/disposer 版本、policy revision、snapshot revision、生命周期和 checkpoint 绑定成独立内容寻址合同；Workbench `execute_tool()` 已先解析 active bundle，再按 registry 的 executor identity 进入原生执行表，旧的 `elif tool_name` 分派已移除。请求、approval digest、policy、runtime 和 API 均绑定 registry snapshot；side-effecting bundle 缺 disposer、旧 snapshot、源文件路径/自动激活字段和 tombstoned resurrection 均 fail-closed。全量 enabled capability 覆盖、registry dispatch、stale replacement、checkpoint roundtrip 和 rollback 已通过 `scripts/training/eval_capability_registry.py`（报告 `gate.passed=true`）及定向/直接集成 Gate；完整 Workbench pytest 回归仍受本机 pytest 临时目录权限影响，不能把直接 harness 结果伪装成 CI 全量结果。

R5B-L1 已推进：`CapabilityCandidate` 将候选包、证据 digest、有限资源预算、评估门和审计元数据独立于可执行 registry；`propose()` 只记录 `proposed`，`validate_candidate()` 才进入 bundle `validated`，之后仍必须 `shadow()` + approval 才可激活；拒绝、checkpoint 恢复和嵌套 executable-source 字段均有 fail-closed 证据。`scripts/training/eval_capability_candidate.py` 报告 `gate.passed=true`。

R5B-L2 已推进：`seed_platform/capability_shadow.py` 只保存输入/输出/after-state/resource 的 digest 与差异，不导入或执行 executor source；read-only 候选要求输出等价且 after-state 不变，side-effecting 候选必须满足 policy/approval，真实副作用、旧 registry snapshot 和 policy deny 均 fail-closed。`scripts/training/eval_capability_shadow.py` 报告 `gate.passed=true`。

R5B-L3 已推进：registry 在激活/替换前计算完整资源 reservation，超限时不改变 active set、snapshot 或 ledger；checkpoint 保存 resource limits、candidate budgets、active/prior reservations，恢复后可继续 rollback；带 disposer 的回滚记录 `disposer_release_recorded`，但不在 registry 内调用任意 disposer source。`scripts/training/eval_capability_resource.py` 报告 `gate.passed=true`。

R5B-L4 已完成架构评审：`plans/active/roadmap/05_R5B_L4_PURE_COMPUTATION_REVIEW_20260830.md` 对当前 workspace、IDE、terminal、MCP 与编辑能力逐项检查，结论为 `architecture_review_required` 且 **No-Go for implementation now**。当前能力都涉及可变工作区、UI/进程状态或外部副作用，尚无同时满足显式值输入、确定性无副作用、独立 oracle 的候选；因此没有把“伪纯计算”塞入 registry。下一步转入 R5C-S0，先建立真实长期 evidence 的内容寻址观察窗口，再由既有 structural growth controller 提案。

R5C-S0 已完成：`taiji/structural_evidence.py` 为 standalone/cross-region runtime observation 提供有界窗口、内容寻址、重复 evidence 幂等/冲突拒绝、单调 tick、容量失败原子性和 checkpoint roundtrip；`TSKV8Adapter` 将两条原生 runtime 路径写入同一 ledger。`tests/taiji_native/test_structural_evidence_window.py` 为 `3 passed`，`scripts/training/eval_taiji_long_horizon_evidence.py` 报告 `gate.passed=true`。本 slice 没有把窗口摘要喂回 growth controller，也没有改变 topology。

R5C-S1 已完成：`StructuralRuntimeObservation` 增加 task slice/partition 归因，ledger 按 train/holdout/retention 隔离窗口；`project_structural_growth_pressure()` 只读取封存窗口，要求至少两个独立 train task slices，并把 holdout/retention 保持为验证计量。pressure projection 有独立 digest、checkpoint roundtrip，且不写 ledger、不调用 controller。相关定向回归共 `33 passed`，`scripts/training/eval_taiji_structural_pressure.py` 报告 `gate.passed=true`。

R5C-S2 已完成：`TSKV8Adapter.propose_structural_candidate_from_pressure()` 将带 holdout 的 pressure projection 单向桥接到既有 growth controller，只生成 lineage-bound、projection-digest 去重的 candidate；外部 projection 的 evidence tick 会被纳入 runtime clock 以保证 checkpoint continuation。candidate materialization 仍是 pending proposal，不提交 topology、不消耗 structural budget。相关定向回归为 `4 passed`，`scripts/training/eval_taiji_structural_bridge.py` 报告 `gate.passed=true`。

R5C-S3A 已完成：新增 `StructuralCandidateValidation` 与 `validate_structural_candidate_shadow()`，复用 operation-specific holdout shadow，记录 parent/validation checkpoint、topology/预算前后 digest，并在 malformed holdout 或 shadow 异常时 fail-closed；合法 proposal 保持 pending，rejected proposal 恢复后不再进入 candidate 队列。相关定向回归为 `6 passed`，`scripts/training/eval_taiji_structural_validation.py` 报告 `gate.passed=true`。

R5C-S3B 已完成：新增 `StructuralValidationGateDecision` 与 `evaluate_structural_candidate_validation()`，把 holdout gain、retention regression、lesion effect、resource state 和 structural budget 的阈值集中为可配置、内容寻址、无副作用 policy；任一维度失败都会给出明确 reason。相关定向回归为 `3 passed`，`scripts/training/eval_taiji_structural_validation_gate.py` 报告 `gate.passed=true`。

R5C-S3C 已完成：`TSKV8Adapter.evaluate_structural_candidate_gate()` 将当前 adapter 的 validated shadow record 与 pending proposal 绑定到 S3B policy；通过时保持 pending，失败时原子标为 rejected，并把 decision 纳入 checkpoint。accepted/rejected 两条路径的 metric integration canary 均通过，`scripts/training/eval_taiji_structural_metric_integration.py` 报告 `gate.passed=true`。

R5C-S3D 已完成：`TSKV8Adapter.admit_structural_candidate()` 只允许通过 policy 的 pending candidate 复用既有 topology commit；结果记录 parent/child checkpoint、topology/budget 前后 digest，重复调用幂等，恢复后保留 admitted 状态。`scripts/training/eval_taiji_structural_admission.py` 报告 `gate.passed=true`。

R5C-S4 已完成：`eval_taiji_structural_stability.py` 在 seed 11/29 和独立 task slice 上重复 pressure→admission，实际测量 retention regression 与 lesion effect，并从 admitted checkpoint rollback 恢复 parent/budget；两组 Gate 全通过。

R5C-S5 已完成：`eval_taiji_structural_continuation.py` 从同一 parent 预算 2 出发，跨 checkpoint 连续完成两次受限 neuron admission，恢复后保留两步 topology、lineage 和精确预算扣减；预算归零后第三个候选即使 holdout/lesion 通过，也因 resource state 与 structural budget 不足被拒绝，拓扑不变且拒绝状态可恢复。报告 `gate.passed=true`。

R5C-S6 已完成：`WorkbenchStructuralEvidence` 将真实 Workbench Outcome 的内容摘要、请求/调用/能力快照与显式 evaluator metrics 绑定为结构观测；结构运行时使用独立可恢复单调时钟，不混用动作起始 tick。`StructuralGrowthScheduleState` 只消费新的 sealed window，经过 train task-slice、holdout 分区隔离后调用既有 pressure→candidate bridge，candidate 保持 pending、拓扑和预算不变；checkpoint restore 后重复调度返回 `no_new_sealed_window`。`scripts/training/eval_taiji_workbench_structural_scheduler.py` 报告 `gate.passed=true`。

R5C-S7 已完成：`continue_structural_candidate()` 将调度出来的 candidate 接入既有 shadow→policy→atomic admission 生命周期；真实 Workbench evidence 只提供可追溯事实，holdout/retention/lesion/resource 指标仍由独立 continuation 输入。通过 Gate 后才改变 topology 并扣减预算，checkpoint restore 后重复 continuation 不重复 admission；`scripts/training/eval_taiji_workbench_growth_continuation.py` 报告 `gate.passed=true`。

R5C-S8 已完成：`StructuralCandidateBatch` 建立内容寻址的多候选 arbitration ledger，按显式 priority/source tick/resource cost/candidate id 确定性排序，在冲突和预算不足时记录 deferred/rejected 原因，并通过独立 reservation 防止批次超售；batch 与 reservation 可 checkpoint 恢复，首个 candidate admission 后剩余候选可继续，重复 arbitration/continuation 幂等。`scripts/training/eval_taiji_structural_arbitration.py` 报告 `gate.passed=true`。

R5C-S9 已完成：`StructuralRegionCapacityPressure` 以只读快照汇总区域占用、待处理候选、reservation 和 structural budget，跨区域容量压力不写 topology；`rollback_structural_candidate_batch()` 将 admitted candidate 从 child checkpoint 可逆恢复到 parent，重开预算并保留 rollback audit，旧 deferred candidate 不被静默删除，新 evidence 可重新仲裁并胜出。`scripts/training/eval_taiji_structural_continuation_recovery.py` 报告 `gate.passed=true`，覆盖跨 checkpoint 第二轮 admission、容量压力变化、回滚、checkpoint restore、重新证据和幂等回滚。

R5C-S10 已完成：`schedule_structural_candidate_batch_from_workbench_evidence()` 将多个真实 Workbench 区域的成功 Outcome 先绑定为内容寻址 evidence，再分别经过 train task-slice/holdout 窗口、pressure projection 和 candidate-only bridge，最后一次性进入既有 `StructuralCandidateBatch` 仲裁；批次请求、源窗口、candidate ids、batch id 和 scheduler revision 均 checkpoint。`scripts/training/eval_taiji_workbench_multi_region_batch.py` 报告 `gate.passed=true`，覆盖 6 次真实 `workspace.read`、两个区域各 2 train+1 holdout、两个不同 candidate、restore 后同 batch/digest/reservation 幂等，以及仲裁不改变 topology/budget。

R5C-S11 已完成：复用既有 batch continuation 将 S10 的真实多区域 candidate 逐个送入 shadow→policy→atomic admission；单候选 malformed holdout 在独立恢复分支中 fail-closed，不能污染另一区域已 admitted 的 candidate，正常恢复分支可完成第二个 admission，随后 rollback 恢复对应区域 parent topology、重开预算，checkpoint restore 后重复 rollback 幂等。`scripts/training/eval_taiji_workbench_multi_region_lifecycle.py` 报告 `gate.passed=true`。

R5C-S12 已完成：`WorkbenchStructuralValidationArtifact` 将真实 Workbench replay 的 holdout 输入/输出、retention/lesion 对照、resource measurement、Outcome/evidence digest 与 parent/trial checkpoint digest 绑定为不可变内容寻址 artifact；`TSKV8Adapter.continue_structural_candidate_from_validation_artifact()` 只在候选、区域、资源、父 checkpoint、holdout replay 和 trial checkpoint 全部匹配时调用既有 shadow→policy→atomic admission。回放不匹配 fail-closed，artifact 支持 checkpoint restore、幂等重复消费与篡改检测；`api/seed_runtime.py` 已提供同一公共入口。`scripts/training/eval_taiji_workbench_validation_artifact.py` 报告 `gate.passed=true`。本 slice 仍不把 provider 成功直接转成准入指标。

R5C-S13 已完成：新增 `StructuralValidationArtifactBatch` 与 `TSKV8Adapter.continue_structural_candidate_batch_from_validation_artifacts()`，多区域 batch 只接收各 candidate 的 replay-bound artifact 和 holdout replay，不再接收批次级手工指标集合；artifact batch digest 按 candidate/artifact 对建立并 checkpoint。第一 candidate admission 后，第二 candidate 可基于新的 parent checkpoint 继续；失败 replay、跨 candidate 错配只影响对应 reservation，恢复后 artifact 集合与重复消费保持幂等。`scripts/training/eval_taiji_workbench_validation_artifact_batch.py` 报告 `gate.passed=true`。

R5C-S14 已完成：新增 `StructuralValidationMeasurements`，从 baseline/candidate/lesion 的实际 replay 张量和原始容量 pressure 计算 holdout gain、retention regression、lesion effect 与 resource state，并把各输入 digest、measurement digest 和计算结果绑定；artifact continuation 显式将 measured `holdout_gain` 传入 policy，避免 shadow score 偷换验证事实。`scripts/training/eval_taiji_workbench_validation_measurements.py` 报告 `gate.passed=true`，实测 holdout gain/lesion effect 约为 `0.1214`、retention regression 为 `0`、resource state 为 `0.5`。

R5C-S15 已完成：将 S14 measurement owner 接入 S13 的多区域 artifact batch，两个真实 region candidate 分别使用自己的 measured replay metrics、原始 resource digest 和 parent checkpoint；policy 两侧均消费对应实测值，增量 batch admission、restore 与重复消费幂等。`scripts/training/eval_taiji_workbench_measured_artifact_batch.py` 报告 `gate.passed=true`，当前不再保留 batch 级手工 metrics 注入路径。

## 5. 本 slice 明确不做

- 不执行物理删除或外部 artifact tombstone 提交；结构成长仍只产生可恢复 candidate，validation artifact 不能绕过 shadow/holdout/retention/lesion 与原子拒绝；S16 之前也不把单轮 canary replay 指标扩大成长期开放域质量结论；
- 不把直接 harness 或 evaluator 证据扩大成正常 CI 全量通过；R5B-S1 的核心接线/替换/回滚/checkpoint 合同已通过，剩余是完整 Workbench 文件回归和 CI 环境验收；
- 不删除 skill/MCP、Legacy、`codex/interaction-group-incremental` 或 `output/`；
- 不启动训练、不改模型权重、不做 CUDA；
- 不把用户点名的渲染缺陷修复扩大成主动视觉美化，不用页面层改动或模拟截图关闭 R3 的任务栏/托盘/通知/DPI 取证；
- 不把 capability registry、candidate package 或 shadow observation 当成认知主体，不让 Taiji/provider/frontend 拥有注册、激活、替换或删除 executor 的权限；
- 不把单个 tick、scale target、单次演示或 holdout 标签直接当作结构成长依据；R5C-S0 只接收可追溯、去重、可 checkpoint 的长期 evidence 聚合。

按用户决定，CI 全量验收暂缓；当前唯一后继为 [04_EXECUTION_PLAN.md §6](04_EXECUTION_PLAN.md) 的 **R5C-S18 多轮 ledger compactness 与跨轮 evidence 消费审计**。L4 已明确暂不实施纯计算执行体；R5C-S0/S1 完成证据保全与非突变 projection，R5C-S2 完成 candidate-only controller bridge，R5C-S3A/B/C 完成 validation 与 policy 接线，R5C-S3D 完成单次受限 admission，R5C-S4 完成双 seed/task slice 稳定性与 rollback，R5C-S5 完成跨 checkpoint 两步连续成长与预算耗尽拒绝，R5C-S6 完成真实 Workbench evidence 接线与可恢复调度，R5C-S7 完成调度候选的验证闭环与 Workbench continuation，R5C-S8 完成多候选确定性仲裁与 reservation continuation，R5C-S9 完成跨区域容量压力、多轮 continuation、回滚与新证据再仲裁，R5C-S10 完成多个真实 Workbench 区域直接生成同一候选批次，R5C-S11 完成真实多区域 batch 的逐候选生命周期，R5C-S12 完成单候选 replay-bound validation artifact，R5C-S13 完成多区域 artifact batch continuation，R5C-S14 完成独立 replay measurement owner，R5C-S15 完成 measured artifact batch，R5C-S16 完成多轮 measured evidence continuation、跨 stream cooldown 修复与 rollback，R5C-S17 完成 measurement/artifact digest 完整性与 provenance 绑定；完整 Workbench CI 回归仍作为后置统一门禁。

## 6. 更新规则

- 每个 slice 完成后先更新 manifest、实现事实和本文件，再提交；不得并列维护第二个“当前唯一下一步”。
- 若实现与架构合同冲突、checkpoint 不能恢复、red proof 不会红或 CI 新增失败，立即停止功能推进，先修该错误。
- 已完成过程进入 archive；仍影响设计和后续开发的核心需求、所有权、接口合同与未关闭缺口必须留在 active/reference。
