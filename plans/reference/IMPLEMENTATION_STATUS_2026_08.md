# Seed / Taiji 实现事实参考

> 事实快照：2026-08-30。本文件只描述当前代码和最近可追溯证据，不决定执行顺序；当前下一步见 [03_CURRENT_EXECUTION.md](../active/roadmap/03_CURRENT_EXECUTION.md)。历史测试轮次和事故链已归档。

## 1. 所有权与依赖

- `taiji/` 是 Taiji-owned substrate 与认知纵切片，不导入 `seed`、`seed_platform`、`neuroplex` 或 Transformers。
- `seed/`、`api/` 和 `seed_platform/` 负责产品/runtime、Workbench、policy、provider 装配和外部副作用边界。
- `neuroplex/` 是冻结 Legacy Transformer 对照；Legacy-off 时不得注册其 router 或认知路径。
- Taiji 形成 `ActionIntent`、`ContentPlan`、`WorldAffordance`、选择和状态 lineage；语言 provider 只实现可读表达。
- capability 必须来自内容寻址 snapshot；前端、prompt、provider 不得维护第二份工具或能力真相。

## 2. 当前已经闭合的主链

### 输入、世界与认知状态

- `InputFrame → Observation/PerceptEvent → WorldState/WorldEvent` 的来源、时间和 checkpoint lineage 已存在。
- 感知、世界预测、workspace、working/episodic/semantic/procedural memory、homeostasis、goal/planning、generation 和结构治理均有 native owner 与窄 Gate。
- interaction-group 由真实 trace 的 contribution/interaction/recovery/lesion 推导，不依赖预设“规划神经元/记忆神经元”角色表。

### Workbench 与 IDE

- Executive 从认知状态和真实 grounding 选择候选，经 capability freshness、policy、executor 和真实 after-state/Outcome 回写。
- IDE 语言识别综合扩展名、shebang、内容、manifest、邻近文件和 toolchain/LSP 证据；高置信可逆自动切换，歧义返回 `ask_user`。
- 文件 patch/create/rename/delete/undo、terminal/diagnostics/test/build 具备结构化预览、审批、原子执行、冲突检测、输出预算和 checkpoint continuation 合同。
- MCP-shaped 本地 registry、有限多步 loop、successor graph、recovery handoff/portfolio 与客户端只读审计投影已接通。

### 学习、provider 与客户端

- 生产 Workbench 路径在真实执行后绑定 source affordance、感知/世界上下文和 Outcome，并在 `learn=True` 时调用 `record_executive_outcome()`；`learn=False` 不产生在线更新。
- checkpoint 往返保留 `fit_updates`、`online_updates` 和最后选择，并能恢复后继续。
- provider artifact 具备内容寻址、registry、loader、startup、watchdog、回滚和客户端观测；当前产品默认 `native-readable`，`structured-stub` 仅保留为显式调试 codec。
- 客户端从 native facade 读取 runtime/provider/homeostasis/training/knowledge/Workbench 状态；生命雷达、窄布局和打包前端字节一致已验证。

## 3. 最近可追溯证据

| 范围 | 最近证据 | 结论边界 |
|---|---|---|
| checkpoint 等价性 | `tests/seed/test_checkpoint_roundtrip_contract.py` 3 例 | provider 脱挂/重绑、状态与 continuation 受门禁保护 |
| R1 provider watchdog | [S0](../../reports/taiji_w7_r1_provider_watchdog_20260829.json)、[S1](../../reports/taiji_w7_r1_provider_watchdog_s1_20260829.json)、[S2](../../reports/taiji_w7_r1_provider_watchdog_s2_20260829.json) | native-readable packaged 观测通过；外部 artifact 轮换未宣称 |
| R2 interaction-group | [S0](../../reports/taiji_w7_r2_interaction_groups_20260829.json)、[S1](../../reports/taiji_w7_r2_interaction_groups_s1_20260829.json)、[S2](../../reports/taiji_w7_r2_interaction_groups_s2_20260829.json) | native replay 与真实只读 Workbench 通过；不写回结构/provider |
| R3 visual/desktop | [S1](../../reports/taiji_w7_r3_visual_desktop_s1_20260829.json)、[S2 部分证据](../../reports/taiji_w7_r3_visual_desktop_s2_20260829.json) | 页面/窄布局通过；Windows shell 仍 `tool-blocked` |
| R5-S0 学习通道 | [报告](../../reports/taiji_w7_r5_s0_learning_channel_20260829.json) | 真实 Outcome 在线更新和 checkpoint continuation 通过 |
| R5A-S0/S1/S2-C implementation | [S0 报告](../../reports/taiji_w7_r5a_s0_internalization_20260830.json)、[S1 报告](../../reports/taiji_w7_r5a_s1_internalization_20260830.json) | S0 纯转换/内容寻址/train-only 去重/生命周期通过；S1 原生学习器、checkpoint lineage、holdout、retention 与 feature/grounding lesion canary 通过；S2-A 已在真实只读 Workbench 上验证 current evidence + snapshot + reprojected affordance 的受限 Outcome 投影；S2-B 单 slice 纵向 holdout/lesion/recovery 与 candidate-only deletion Gate 已通过；S2-C 聚合 Gate、跨 seed/task slice 约束和独立删除评审已实现，并以 seed 11/29 的真实 Workbench task slice 通过 |
| R5B-L0/S1 capability registry + Workbench dispatch | [合同](../../manifests/taiji_w7_r5_effector_registry_v1.json)、[定向 Gate](../../tests/seed_platform/test_capability_registry_contract.py)、[evaluator 报告](../../reports/taiji_w7_r5b_s1_capability_registry_20260830.json) | capability bundle 内容寻址、active snapshot/revision、validated→shadow→active 生命周期、显式 approval、side-effect disposer 约束、stale fail-closed、retire/tombstone、checkpoint roundtrip、request/approval registry snapshot binding、全量 enabled capability 覆盖、原子 replacement/rollback 和 registry-backed Workbench executor dispatch 已通过定向/直接集成验证；evaluator `gate.passed=true` 且 checkpoint roundtrip=true；完整 Workbench CI 回归待正常临时目录环境验收 |
| R5B-L1 capability candidate package | [合同](../../manifests/taiji_w7_r5_effector_registry_v1.json)、[定向 Gate](../../tests/seed_platform/test_capability_registry_contract.py)、[evaluator 报告](../../reports/taiji_w7_r5b_l1_capability_candidate_20260830.json) | candidate artifact 内容寻址、evidence digest、有限 resource budget、evaluation gates、proposed→validated/rejected 分离、checkpoint continuation、嵌套 executable-source fail-closed 已通过；验证后仍不激活，必须经过 shadow 与 approval；L2 shadow 差异 Gate 尚未实现 |
| R5B-L2 shadow and approval Gate | [合同](../../manifests/taiji_w7_r5_effector_registry_v1.json)、[定向 Gate](../../tests/seed_platform/test_capability_shadow_contract.py)、[evaluator 报告](../../reports/taiji_w7_r5b_l2_capability_shadow_20260830.json) | digest-only shadow observation、输出/after-state/资源差异、policy deny、stale snapshot、side-effect detection 和 side-effect approval 已通过；shadow 不执行 executor source、不产生真实副作用；L3 原子激活与资源回滚尚未实现 |
| R5B-L3 resource and rollback Gate | [合同](../../manifests/taiji_w7_r5_effector_registry_v1.json)、[定向 Gate](../../tests/seed_platform/test_capability_resource_contract.py)、[evaluator 报告](../../reports/taiji_w7_r5b_l3_capability_resource_20260830.json) | active-set 原子提交、bounded resource reservation、resource exhaustion isolation、checkpoint ledger continuation、replacement rollback 和 disposer release audit 已通过；未知 disposer source 不在 registry 内执行；L4 纯计算能力架构评审尚未开始 |
| R5C-S0 long-horizon evidence window | [定向 Gate](../../tests/taiji_native/test_structural_evidence_window.py)、[evaluator 报告](../../reports/taiji_w7_r5c_s0_long_horizon_evidence_20260830.json) | 内容寻址窗口、重复/冲突证据、单调 tick、容量原子性、adapter capture 和 checkpoint roundtrip 已通过；不调用 growth controller、不改变 topology |
| R5C-S1 structural pressure projection | [定向 Gate](../../tests/taiji_native/test_structural_pressure.py)、[evaluator 报告](../../reports/taiji_w7_r5c_s1_structural_pressure_20260830.json) | train task-slice 与 holdout/retention 隔离、跨任务 projection、projection digest 和 ledger immutability 已通过；不产生 candidate、不触发 topology growth |
| R5C-S2 candidate-only structural bridge | [路线](../active/roadmap/08_R5C_S2_STRUCTURAL_BRIDGE_20260830.md)、[定向 Gate](../../tests/taiji_native/test_structural_pressure.py)、[evaluator 报告](../../reports/taiji_w7_r5c_s2_structural_bridge_20260830.json) | projection digest 去重、holdout-gated controller bridge、parent checkpoint lineage、外部 evidence clock continuation 和 checkpoint restore 已通过；candidate materialization 仍 pending，不改变 topology、不消耗 structural budget；S3 validation Gate 尚未实现 |
| R5C-S3A candidate shadow validation | [路线](../active/roadmap/09_R5C_S3A_CANDIDATE_VALIDATION_20260830.md)、[定向 Gate](../../tests/taiji_native/test_structural_pressure.py)、[evaluator 报告](../../reports/taiji_w7_r5c_s3a_structural_validation_20260830.json) | checkpointed validation record、operation-specific holdout shadow、topology/budget invariants、malformed holdout fail-closed 和 rejected candidate non-resurrection 已通过；合法 proposal 保持 pending，后续 policy/admission 由 S3B–S3D 负责 |
| R5C-S3B independent validation policy | [路线](../active/roadmap/10_R5C_S3B_VALIDATION_POLICY_20260830.md)、[定向 Gate](../../tests/taiji_native/test_structural_validation.py)、[evaluator 报告](../../reports/taiji_w7_r5c_s3b_structural_validation_gate_20260830.json) | holdout gain、retention regression、lesion effect、resource state 和 budget 的可配置内容寻址 policy 已通过；任一失败维度显式拒绝且无模型突变，S3C/S3D 已完成后续 metric integration 与受限 admission |
| R5C-S3C metric-to-policy integration | [路线](../active/roadmap/11_R5C_S3C_METRIC_INTEGRATION_20260830.md)、[定向 Gate](../../tests/taiji_native/test_structural_pressure.py)、[evaluator 报告](../../reports/taiji_w7_r5c_s3c_structural_metric_integration_20260830.json) | 当前 adapter validation record 与 pending proposal 已接入独立 policy；accepted 保持 pending，failed 原子 rejected，decision checkpoint restore 和 metric integration 均通过 |
| R5C-S3D atomic structural admission | [路线](../active/roadmap/12_R5C_S3D_ATOMIC_ADMISSION_20260830.md)、[定向 Gate](../../tests/taiji_native/test_structural_pressure.py)、[evaluator 报告](../../reports/taiji_w7_r5c_s3d_structural_admission_20260830.json) | 通过 policy 的 pending candidate 可按既有 topology transaction 单次 admission；topology/budget lineage、幂等和 checkpoint restore 已通过 |
| R5C-S4 cross-seed/task-slice stability and rollback | [路线](../active/roadmap/13_R5C_S4_STABILITY_ROLLBACK_20260830.md)、[evaluator 报告](../../reports/taiji_w7_r5c_s4_structural_stability_20260830.json) | seed 11/29 独立重复 admission，实际 retention/lesion 测量和 admitted checkpoint rollback 均通过 |
| R5C-S5 multi-step bounded continuation | [路线](../active/roadmap/14_R5C_S5_STRUCTURAL_CONTINUATION_20260830.md)、[evaluator 报告](../../reports/taiji_w7_r5c_s5_structural_continuation_20260830.json) | 预算 2 下两次 admission 可跨 checkpoint 连续恢复，预算精确扣减到 0；第三候选在预算耗尽时 fail-closed，rejection restore 保持 topology 不变 |
| R5C-S6 Workbench structural-growth scheduler | [路线](../active/roadmap/15_R5C_S6_WORKBENCH_STRUCTURAL_SCHEDULER_20260830.md)、[evaluator 报告](../../reports/taiji_w7_r5c_s6_workbench_structural_scheduler_20260830.json) | `WorkbenchStructuralEvidence` 将真实成功 Outcome 绑定 outcome digest、请求/调用/能力快照和显式 evaluator metrics；独立 structural runtime tick 接入 adapter；sealed windows 按 task slice/partition 聚合，scheduler 只消费新窗口并创建 candidate-only proposal；checkpoint restore 与重复调度幂等均通过，报告 `gate.passed=true` |
| R5C-S7 Workbench candidate continuation | [evaluator 报告](../../reports/taiji_w7_r5c_s7_workbench_growth_continuation_20260830.json) | 调度 candidate 接入 shadow→policy→atomic admission；真实 Workbench evidence 不绕过独立 holdout/retention/lesion/resource 指标；通过后才改变 topology/预算，checkpoint restore 与重复 continuation 幂等通过，报告 `gate.passed=true` |
| R5C-S8 multi-candidate arbitration | [路线](../active/roadmap/17_R5C_S8_MULTI_CANDIDATE_ARBITRATION_20260830.md)、[evaluator 报告](../../reports/taiji_w7_r5c_s8_structural_arbitration_20260830.json) | `StructuralCandidateBatch` 对候选按显式排序规则进行内容寻址 arbitration；冲突与预算不足转为可恢复 deferred/rejected，reservation 不修改真实预算；首个 admission 后剩余候选可跨 checkpoint continuation，重复 arbitration/continuation 幂等，报告 `gate.passed=true` |
| R5C-S9 continuation recovery | [路线](../active/roadmap/18_R5C_S9_CONTINUATION_RECOVERY_20260830.md)、[evaluator 报告](../../reports/taiji_w7_r5c_s9_structural_continuation_recovery_20260830.json) | `StructuralRegionCapacityPressure` 只读记录跨区域占用、队列与 reservation 压力；多轮 batch 可跨 checkpoint continuation；admitted candidate 可回滚到 parent、重开预算并保留 audit；新 evidence 可重新仲裁 deferred candidate，重复 rollback 幂等，报告 `gate.passed=true` |
| R5C-S10 Workbench multi-region batch | [路线](../active/roadmap/19_R5C_S10_WORKBENCH_MULTI_REGION_BATCH_20260830.md)、[evaluator 报告](../../reports/taiji_w7_r5c_s10_workbench_multi_region_batch_20260830.json) | 多个真实 Workbench 区域分别形成 2 train+1 holdout evidence windows，经 pressure projection 生成两个不同 candidate，并一次性进入同一 `StructuralCandidateBatch`；请求、源窗口、batch、scheduler revision checkpoint 可恢复，重复调度幂等，仲裁不改 topology/budget，报告 `gate.passed=true` |
| R5C-S11 Workbench multi-region lifecycle | [路线](../active/roadmap/20_R5C_S11_WORKBENCH_MULTI_REGION_LIFECYCLE_20260830.md)、[evaluator 报告](../../reports/taiji_w7_r5c_s11_workbench_multi_region_lifecycle_20260830.json) | S10 真实多区域 batch 接入既有 shadow→policy→atomic admission→rollback；独立恢复分支验证单候选 fail-closed 不污染另一候选，正常分支完成第二候选 admission，回滚恢复区域 topology/预算且 checkpoint/replay 幂等，报告 `gate.passed=true` |
| R5C-S12 Workbench replay validation artifact | [路线](../active/roadmap/21_R5C_S12_WORKBENCH_VALIDATION_ARTIFACT_20260830.md)、[evaluator 报告](../../reports/taiji_w7_r5c_s12_workbench_validation_artifact_20260830.json) | `WorkbenchStructuralValidationArtifact` 将真实 replay 的 holdout 输入/输出、retention/lesion 对照、resource measurement、Outcome/evidence digest 与 parent/trial checkpoint digest 内容寻址绑定；回放不匹配、篡改和错配 fail-closed，artifact 进入既有 shadow→policy→atomic admission，公共 SeedRuntime 入口、checkpoint restore 与重复消费幂等均通过，报告 `gate.passed=true` |
| R5C-S13 Workbench validation artifact batch continuation | [evaluator 报告](../../reports/taiji_w7_r5c_s13_workbench_validation_artifact_batch_20260830.json) | `StructuralValidationArtifactBatch` 按 candidate/artifact digest 汇总多区域 batch 的已消费 artifact；batch continuation 只接收 replay-bound artifact 与 holdout replay，不接收手工 metrics；第一候选准入后第二候选绑定新 parent checkpoint，失败/错配隔离到对应 candidate，restore 后 batch digest 与重复消费幂等，报告 `gate.passed=true` |
| R5C-S14 replay measurement owner | [evaluator 报告](../../reports/taiji_w7_r5c_s14_workbench_validation_measurements_20260830.json) | `StructuralValidationMeasurements` 从真实 baseline/candidate/lesion replay 张量和原始容量 pressure 计算 holdout gain、retention regression、lesion effect、resource state，并绑定输入/测量 digest；artifact continuation 显式将 measured holdout gain 交给 policy，实测约为 0.1214/0/0.1214/0.5，报告 `gate.passed=true` |
| R5C-S15 measured multi-region artifact batch | [evaluator 报告](../../reports/taiji_w7_r5c_s15_workbench_measured_artifact_batch_20260830.json) | S14 measurement owner 接入 S13 多区域 batch；两个 region candidate 各自使用 measured metrics、原始 resource digest 和当前 parent checkpoint，policy 两侧消费对应实测值，增量 admission、artifact batch restore 与重复消费幂等，报告 `gate.passed=true` |
| R5C-S16 multi-round measured evidence continuation | [路线](../active/roadmap/25_R5C_S16_WORKBENCH_MEASURED_MULTI_ROUND_20260830.md)、[evaluator 报告](../../reports/taiji_w7_r5c_s16_workbench_measured_multi_round_20260830.json) | 两轮各 6 次真实 Workbench 读取形成新窗口/候选 batch；第二轮旧 parent artifact fail-closed，新 measured artifact 完成逐候选 admission，rollback 恢复到当轮第一候选后的 topology/budget，两个 artifact batch、四个 admission 与 rollback checkpoint 可恢复且重复 rollback 幂等；同时修复按 stream 隔离 scheduler cooldown，报告 `gate.passed=true` |
| R5C-S17 artifact/measurement integrity | 待整理路线、[evaluator 报告](../../reports/taiji_w7_r5c_s17_workbench_integrity_20260830.json) | `StructuralValidationMeasurements` 反序列化重算 measurement digest；`WorkbenchStructuralValidationArtifact` 显式绑定 measurement digest，同时保留无该字段旧 artifact 的读取兼容；metric/raw digest/format 和 artifact binding 篡改均 fail-closed，checkpoint ledger 保持绑定，报告 `gate.passed=true` |

R3 最终包为 `dist/Seed/Seed.exe`，已记录 SHA-256 `76b432b43922d5d70c64fca36b8e7045f2f5d03d4492f09b68b47eb31756368b`、大小 72,752,598 字节；源码与包内前端 211 个文件集合/字节一致，前端回归 `43 files / 245 passed`，Vite build 与 ESLint 通过。Chrome 已验证生命页和 900px/760px IDE 布局；Windows Computer Use 无法激活窗口，因此任务栏、托盘、通知和高 DPI 未通过。

R5-S0 定向 native/executive/desktop/project identity 回归记录为 `24 passed`，Ruff、compileall、checkpoint 往返和 diff 检查通过。2026-08-30 CI 修复链新增 `b6d1bf2`：只读 Workbench 在 admission 后不再要求可变的当前 executive decision，前端 E2E 仅要求生命状态页展示 `RuntimeEvidenceStrip`，其余页面显式验证不展示；本地对应 Workbench 回归为 `2 passed`，mypy 为 `45 source files` 无问题。远端运行 `33295880356` 已完成最终验收，Docker、前端含 E2E、Legacy/no-Legacy smoke、Python 3.10/3.12 与 Windows 全量回归 7 个 job 全部成功。

R5-G1 合同 Gate 已新增两份独立 manifest，并覆盖合法合同与缺 owner、混合 owner、缺 checkpoint、认知越权、错误删除边界的 red contract。R5A-S0 已实现 `taiji/internalization.py`：纯 grounded Outcome DTO、内容寻址、train-only replay 去重、生命周期/五项因果门控和 checkpoint roundtrip；R5A-S1 新增 `taiji/internalization_learner.py`：原生局部更新、父/子 checkpoint lineage、留出/保持集只读评估和 feature/grounding lesion；定向测试 `19 passed`，S1 canary `gate.passed=true`。R5A-S2-C 真实双 seed/task slice 集成已通过。R5B-L0/S1 新增 `seed_platform/capability_registry.py` 与 registry-backed Workbench dispatch，完成 6 个 registry unit tests、5 个直接 Workbench integration tests 和 evaluator canary；完整 Workbench 文件回归仍待正常 CI 临时目录环境验收。

## 4. 明确未完成

- R5A-S1 已有原生学习器和 synthetic holdout/lesion 收益证据；R5A-S2-B 已补齐单 slice 的真实 Workbench 纵向收益、跨任务保持和五类 candidate-only Gate；S2-C 的跨 seed/task slice 聚合、独立删除评审和真实 seed 11/29 Workbench 集成已通过。
- `seed_platform/workbench.py` 的真实能力执行已通过 registry resolve 后进入 executor identity 表；R5B-L0/S1 合同、全量 enabled 覆盖、stale binding、replacement 和 rollback 已通过定向/直接集成验证，但完整 Workbench 文件回归尚未在正常 CI 临时目录环境验收。
- `taiji_w7_r5_open_domain_growth_v1.json` 只冻结结构成长合同，不能覆盖知识内化或效应器成长；R5A/R5B 的独立 manifest 已创建，R5B-L4 已完成 No-Go 架构评审，R5C-S0–S11 已完成证据窗口、pressure projection、candidate bridge、validation、admission、稳定性/rollback、多步 continuation、Workbench 调度、候选延续、多候选仲裁、跨区域容量压力、可逆回滚、多区域真实批次生成和完整 batch 生命周期闭环。
- R5C-S0 已落地：`taiji/structural_evidence.py` 为 standalone/cross-region runtime observation 提供有界窗口、内容寻址、重复证据幂等/冲突拒绝、容量失败原子性和 checkpoint roundtrip；`eval_taiji_long_horizon_evidence.py` 的 `gate.passed=true`。它不直接改变 topology。
- R5C-S1 已落地：`StructuralRuntimeObservation` 增加 task slice/partition 归因，`project_structural_growth_pressure()` 要求至少两个 train task slices，隔离 holdout/retention 并保持 ledger 不变；`eval_taiji_structural_pressure.py` 的 `gate.passed=true`。
- R5C-S2 已落地：pressure projection 经过 digest 去重后可观察既有 growth controller，并只生成绑定 parent checkpoint 的 candidate；重复 projection、缺 holdout 和 checkpoint continuation 均有边界。`eval_taiji_structural_bridge.py` 的 `gate.passed=true`。
- R5C-S3A 已落地：candidate-only holdout shadow 生成可恢复的 `StructuralCandidateValidation`，合法 proposal 保持 pending，malformed holdout 不改变 parent/budget，rejected candidate 在 restore 后不复活；`eval_taiji_structural_validation.py` 的 `gate.passed=true`。
- R5C-S3B 已落地：`StructuralValidationGateDecision` 把五类候选准入指标统一为可配置、内容寻址、无副作用 policy；`eval_taiji_structural_validation_gate.py` 的 `gate.passed=true`。
- R5C-S3C 已落地：真实 candidate shadow validation record 可绑定 retention/lesion/resource 指标进入 policy，accepted/rejected 两条路径均保持 topology/budget 边界并支持 restore；`eval_taiji_structural_metric_integration.py` 的 `gate.passed=true`。
- R5C-S3D 已落地：通过的 candidate decision 可进入一次受限 topology admission，结果绑定 parent/child checkpoint、topology/budget after-state，并且重复调用幂等；`eval_taiji_structural_admission.py` 的 `gate.passed=true`。
- R5C-S4 已落地：seed 11/29 在独立 task slice 上重复同一受限 growth contract，retention regression、lesion effect、single admission 和 parent rollback 均通过；`eval_taiji_structural_stability.py` 的 `gate.passed=true`。
- R5C-S5 已落地：预算为 2 时两次 admission 可跨 checkpoint 连续恢复，预算精确变为 0；第三候选因 resource state/budget 不足拒绝，重启后拒绝状态和 exhausted topology 均保持；`eval_taiji_structural_continuation.py` 的 `gate.passed=true`。
- R5C-S6 已落地：`WorkbenchStructuralEvidence` 将真实 Workbench 成功 Outcome 的内容摘要、请求/调用/能力快照与显式 evaluator metrics 绑定到 `StructuralRuntimeObservation`；adapter 使用独立可恢复 structural runtime tick，sealed windows 按 task slice/partition 隔离，scheduler 只消费新窗口并创建 candidate-only proposal。`eval_taiji_workbench_structural_scheduler.py` 报告 `gate.passed=true`，restore 后 cursor/candidate 保持且重复调度幂等。
- R5C-S7 已落地：`continue_structural_candidate()` 将 scheduler candidate 接入已有 shadow、policy 与 atomic admission；`eval_taiji_workbench_growth_continuation.py` 报告 `gate.passed=true`，验证真实 Workbench evidence、独立 holdout/retention/lesion/resource 指标、预算精确扣减、checkpoint restore 和重复 continuation 幂等。
- R5C-S8 已落地：`StructuralCandidateBatch` 持久化候选集、显式排序、区域/连接冲突、budget reservation、deferred/rejected 原因和 candidate 状态；`eval_taiji_structural_arbitration.py` 报告 `gate.passed=true`，验证 arbitration 不改 topology/budget、两个无冲突候选跨 checkpoint 继续 admission、reservation 精确释放和重复执行幂等。
- R5C-S9 已落地：`StructuralRegionCapacityPressure` 提供跨区域只读容量快照，`rollback_structural_candidate_batch()` 绑定 admitted parent/child checkpoint 并恢复 topology/预算；`eval_taiji_structural_continuation_recovery.py` 报告 `gate.passed=true`，验证第二轮 continuation、容量压力跟踪、回滚重开预算、新 evidence 再仲裁、checkpoint restore 与重复回滚幂等。
- R5C-S10 已落地：`schedule_structural_candidate_batch_from_workbench_evidence()` 将多个真实 Workbench 区域的 Outcome 先绑定为 evidence，再分别进入窗口、pressure 与 candidate bridge，最后一次性建立确定性 batch；`eval_taiji_workbench_multi_region_batch.py` 报告 `gate.passed=true`，验证 6 次真实读取、两个区域的独立 train/holdout、两个候选、同 batch restore/replay 和 topology/budget 不变。
 - R5C-S11 已落地：真实多区域 batch 复用 `continue_structural_candidate_batch()` 完成逐候选 shadow→policy→atomic admission；失败候选在独立恢复分支中 fail-closed 且不污染已 admission 候选，正常分支完成第二候选后可通过 `rollback_structural_candidate_batch()` 恢复对应 region、重开预算，`eval_taiji_workbench_multi_region_lifecycle.py` 报告 `gate.passed=true`。
 - R5C-S12 已落地：`WorkbenchStructuralValidationArtifact` 把真实 Workbench replay 的 holdout、retention、lesion、resource 事实与 Outcome/evidence、parent/trial checkpoint digest 统一内容寻址；`continue_structural_candidate_from_validation_artifact()` 在候选/region/resource/checkpoint/replay 全部匹配时才进入既有 policy lifecycle，错配 fail-closed，checkpoint restore、重复消费和 tamper rejection 均通过，`eval_taiji_workbench_validation_artifact.py` 报告 `gate.passed=true`。当前唯一后继为 R5C-S13 多区域 batch replay validation artifact continuation。
 - R5C-S13 已落地：`StructuralValidationArtifactBatch` 与 `continue_structural_candidate_batch_from_validation_artifacts()` 将 S12 artifact 扩展到多区域 batch；batch API 只消费 candidate-bound artifact 和 replay payload，第一 candidate admission 后第二 candidate 重新绑定 parent checkpoint，坏 replay/跨 candidate 错配只影响对应 reservation，artifact batch checkpoint/replay 与重复消费幂等，`eval_taiji_workbench_validation_artifact_batch.py` 报告 `gate.passed=true`。当前唯一后继为 R5C-S14 独立 replay measurement owner。
 - R5C-S14 已落地：`StructuralValidationMeasurements` 从 raw baseline/candidate/lesion probes 和原始容量快照计算四类 validation metrics，所有输入与计算结果都有 digest；artifact continuation 不再让 shadow score 覆盖 measured holdout gain，`eval_taiji_workbench_validation_measurements.py` 报告 `gate.passed=true`。当前唯一后继为 R5C-S15 实测多区域 artifact batch。
 - R5C-S15 已落地：S14 measurement owner 已接入多区域 artifact batch，两个 candidate 的 policy 分别消费独立 measured metrics 和 resource digest，第一轮/第二轮增量 admission、checkpoint restore 和重复 artifact batch consumption 均通过，`eval_taiji_workbench_measured_artifact_batch.py` 报告 `gate.passed=true`。当前唯一后继为 R5C-S16 多轮 measured evidence continuation 与 rollback。
 - R5C-S16 已落地：`StructuralGrowthScheduleState` 新增按 `network_id:region_id` 隔离的 cooldown cursor，解决多区域先处理高 tick stream 后饿死低 tick stream 的核心调度错误；两轮真实 Workbench evidence 通过 measured artifact batch continuation，旧 parent artifact 在状态变化后 fail-closed，第二轮成功 admission、rollback、checkpoint restore 与重复 rollback 幂等，`eval_taiji_workbench_measured_multi_round.py` 报告 `gate.passed=true`。当前唯一后继为 R5C-S17 多轮 artifact/measurement integrity 与 provenance closure。
 - R5C-S17 已落地：measurement payload 在构造和反序列化时重算 digest，artifact 新增可选 `measurement_digest` 绑定并兼容旧 payload；篡改 metric/raw digest/measurement digest/artifact binding 均 fail-closed，checkpoint restore 保持绑定，`eval_taiji_workbench_integrity.py` 报告 `gate.passed=true`。当前唯一后继为 R5C-S18 多轮 ledger compactness 与跨轮 evidence 消费审计。
- 默认自治仍是 freshness-valid、Taiji-owned 的只读 Workbench 路径；写入自治、外部 MCP 生命周期、长程开放域任务未完成。
- 外部语言 provider artifact 的真实 packaged-client 轮换/重启重绑尚未形成 S2；native-readable 可用不等于语言质量已经成熟。
- 当前无 CUDA 实测；CPU profile 不构成 CUDA 支持。
- Windows shell 的任务栏、托盘、通知和高 DPI 缺真实现场证据。

## 5. 参数、训练与资源边界

- Taiji 当前是多 owner 的 native 参数/状态系统，不存在一个可诚实汇总为“完整 Taiji 大模型参数量”的单一发布数字；发布参数规模前必须按器官、稳定参数、可增长容量、非参数状态和 provider artifact 分账。
- 训练优势不能由架构口号推断。当前已证明的是局部/在线更新、结构预算和 checkpoint continuation；吞吐、显存、能耗和质量必须在固定 workload 上实测。
- 任何训练前先运行 checkpoint 保存→关闭→恢复→继续 Gate；检查失败时禁止启动长训。
- CUDA 到位前只允许维护设备抽象、CPU profiler 基线和禁止路径测试，不提交 GPU 性能结论。

## 6. 当前合同入口

- [Taiji 核心需求](../active/TAIJI_CORE_REQUIREMENTS.md)
- [Taiji Native Architecture v1](../active/TAIJI_NATIVE_ARCHITECTURE_V1.md)
- [架构方向与 Transformer 边界](../active/ARCHITECTURE_DIRECTION_2026_08.md)
- [Seed 产品与运行时架构](../active/SEED_ARCHITECTURE.md)
- [当前门禁与 CI](../active/roadmap/02_GATES_AND_CI.md)
- [后续详细计划](../active/roadmap/04_EXECUTION_PLAN.md)
