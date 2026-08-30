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
| W7-R5C 开放域成长 | 合同已冻结；进入 R5C-S0 真实长期证据接入 | 结构成长的输入/证据/回滚边界已版本化，既有 structural growth/topology ledger 可复用 | 尚未把长期真实 Workbench evidence 聚合成可审计的成长触发输入 |

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

执行 **W7-R5C-S0：把真实长期 Workbench evidence 接入结构成长的可审计观察窗口**。

R5-G1 与 R5A-S0 已完成并交付：

1. `plans/manifests/taiji_w7_r5_internalization_v1.json` 与 `plans/manifests/taiji_w7_r5_effector_registry_v1.json`；
2. `taiji/internalization.py`：不依赖 `seed_platform` 的 Outcome/evidence DTO、内容 digest、train-only replay、生命周期和五项因果门控；
3. `tests/taiji_native/test_internalization_contract.py`：未 grounding、缺 reward、越界、provider/capability 文本、holdout 写穿、重复 evidence、未通过 causal gate、checkpoint resurrection 的 red proof；
4. `tests/test_w7_gate_manifests.py`：合同边界与 R5A/R5B 分离关系；
5. R5A S0/S1 证据为 S0 定向测试 `14 passed`、S1 定向测试 `5 passed`、原生 canary `gate.passed=true`，以及本地 checkpoint/holdout/lesion 检查；R5B 与 R5C 仍未实现。

R5A-S2-A 已完成：`api/seed_runtime.py` 只会在当前、校验过的 `workbench.evidence`、同一 capability snapshot 和由该 evidence 重投影出的 grounded successor affordance 同时成立时，创建 `GroundedOutcomeEvidence`。运行时不拥有 replay、learner 或 lifecycle 的写权；缺失/陈旧 snapshot 或非当前 affordance 全部 fail-closed。定向用例在真实只读 workspace 上通过，并已回归完整 Workbench 合同 `44 passed`。

R5A-S2-B 已完成：`taiji/internalization_longitudinal.py` 将真实 Workbench Outcome 转换成 train/holdout task slice，使用 train-only pairwise preference 更新，验证外部规则移除、内化特征/grounding lesion、旧任务 retention、checkpoint restore，并且只生成 `candidate_only_no_physical_deletion` 候选；真实 external artifact 未被删除，holdout 未进入 replay。

R5A-S2-C 已完成并通过真实集成：`InternalizationStabilityTrial` 按 seed 和 task slice 封装 S2-B 报告，`InternalizationStabilityGate` 检查跨试验收益、lesion、retention、指标离散度和 bounded resource counters，`IndependentDeletionReview` 单独检查 artifact/checkpoint/lifecycle/manifest 绑定，并拒绝任何 path、disposer、executor 或删除字段。真实 Workbench 集成用例以 seed 11 与 seed 29 执行通过；由于当前 pytest 临时目录受限，使用预创建的主机系统临时工作区运行了同一测试函数，结果为 `direct S2-C Workbench integration passed`。

R5B-L0/S1 已推进：`seed_platform/capability_registry.py` 将 capability bundle、executor/disposer 版本、policy revision、snapshot revision、生命周期和 checkpoint 绑定成独立内容寻址合同；Workbench `execute_tool()` 已先解析 active bundle，再按 registry 的 executor identity 进入原生执行表，旧的 `elif tool_name` 分派已移除。请求、approval digest、policy、runtime 和 API 均绑定 registry snapshot；side-effecting bundle 缺 disposer、旧 snapshot、源文件路径/自动激活字段和 tombstoned resurrection 均 fail-closed。全量 enabled capability 覆盖、registry dispatch、stale replacement、checkpoint roundtrip 和 rollback 已通过 `scripts/training/eval_capability_registry.py`（报告 `gate.passed=true`）及定向/直接集成 Gate；完整 Workbench pytest 回归仍受本机 pytest 临时目录权限影响，不能把直接 harness 结果伪装成 CI 全量结果。

R5B-L1 已推进：`CapabilityCandidate` 将候选包、证据 digest、有限资源预算、评估门和审计元数据独立于可执行 registry；`propose()` 只记录 `proposed`，`validate_candidate()` 才进入 bundle `validated`，之后仍必须 `shadow()` + approval 才可激活；拒绝、checkpoint 恢复和嵌套 executable-source 字段均有 fail-closed 证据。`scripts/training/eval_capability_candidate.py` 报告 `gate.passed=true`。

R5B-L2 已推进：`seed_platform/capability_shadow.py` 只保存输入/输出/after-state/resource 的 digest 与差异，不导入或执行 executor source；read-only 候选要求输出等价且 after-state 不变，side-effecting 候选必须满足 policy/approval，真实副作用、旧 registry snapshot 和 policy deny 均 fail-closed。`scripts/training/eval_capability_shadow.py` 报告 `gate.passed=true`。

R5B-L3 已推进：registry 在激活/替换前计算完整资源 reservation，超限时不改变 active set、snapshot 或 ledger；checkpoint 保存 resource limits、candidate budgets、active/prior reservations，恢复后可继续 rollback；带 disposer 的回滚记录 `disposer_release_recorded`，但不在 registry 内调用任意 disposer source。`scripts/training/eval_capability_resource.py` 报告 `gate.passed=true`。

R5B-L4 已完成架构评审：`plans/active/roadmap/05_R5B_L4_PURE_COMPUTATION_REVIEW_20260830.md` 对当前 workspace、IDE、terminal、MCP 与编辑能力逐项检查，结论为 `architecture_review_required` 且 **No-Go for implementation now**。当前能力都涉及可变工作区、UI/进程状态或外部副作用，尚无同时满足显式值输入、确定性无副作用、独立 oracle 的候选；因此没有把“伪纯计算”塞入 registry。下一步转入 R5C-S0，先建立真实长期 evidence 的内容寻址观察窗口，再由既有 structural growth controller 提案。

## 5. 本 slice 明确不做

- 不执行物理删除或外部 artifact tombstone 提交；S2-B 只产生可恢复的候选，S2-C 只增加稳定性与独立评审证据；
- 不把直接 harness 或 evaluator 证据扩大成正常 CI 全量通过；R5B-S1 的核心接线/替换/回滚/checkpoint 合同已通过，剩余是完整 Workbench 文件回归和 CI 环境验收；
- 不删除 skill/MCP、Legacy、`codex/interaction-group-incremental` 或 `output/`；
- 不启动训练、不改模型权重、不做 CUDA；
- 不把用户点名的渲染缺陷修复扩大成主动视觉美化，不用页面层改动或模拟截图关闭 R3 的任务栏/托盘/通知/DPI 取证；
- 不把 capability registry、candidate package 或 shadow observation 当成认知主体，不让 Taiji/provider/frontend 拥有注册、激活、替换或删除 executor 的权限；
- 不把单个 tick、scale target、单次演示或 holdout 标签直接当作结构成长依据；R5C-S0 只接收可追溯、去重、可 checkpoint 的长期 evidence 聚合。

按用户决定，CI 全量验收暂缓；当前唯一后继为 [04_EXECUTION_PLAN.md §6](04_EXECUTION_PLAN.md) 的 **R5C-S0 真实长期 evidence 观察窗口**。L4 已明确暂不实施纯计算执行体；完整 Workbench CI 回归仍作为后置统一门禁，确认结果/策略/错误等价、stale snapshot、replacement/rollback 和 checkpoint continuation 没有新增失败。

## 6. 更新规则

- 每个 slice 完成后先更新 manifest、实现事实和本文件，再提交；不得并列维护第二个“当前唯一下一步”。
- 若实现与架构合同冲突、checkpoint 不能恢复、red proof 不会红或 CI 新增失败，立即停止功能推进，先修该错误。
- 已完成过程进入 archive；仍影响设计和后续开发的核心需求、所有权、接口合同与未关闭缺口必须留在 active/reference。
