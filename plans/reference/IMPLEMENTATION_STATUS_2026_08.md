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
| R5A-S0/S1/S2-A 内化 | [S0 报告](../../reports/taiji_w7_r5a_s0_internalization_20260830.json)、[S1 报告](../../reports/taiji_w7_r5a_s1_internalization_20260830.json) | S0 纯转换/内容寻址/train-only 去重/生命周期通过；S1 原生学习器、checkpoint lineage、holdout、retention 与 feature/grounding lesion canary 通过；S2-A 已在真实只读 Workbench 上验证 current evidence + snapshot + reprojected affordance 的受限 Outcome 投影，S2-B 纵向 holdout/lesion/recovery 与可删性仍未开始 |

R3 最终包为 `dist/Seed/Seed.exe`，已记录 SHA-256 `76b432b43922d5d70c64fca36b8e7045f2f5d03d4492f09b68b47eb31756368b`、大小 72,752,598 字节；源码与包内前端 211 个文件集合/字节一致，前端回归 `43 files / 245 passed`，Vite build 与 ESLint 通过。Chrome 已验证生命页和 900px/760px IDE 布局；Windows Computer Use 无法激活窗口，因此任务栏、托盘、通知和高 DPI 未通过。

R5-S0 定向 native/executive/desktop/project identity 回归记录为 `24 passed`，Ruff、compileall、checkpoint 往返和 diff 检查通过。2026-08-30 CI 修复链新增 `b6d1bf2`：只读 Workbench 在 admission 后不再要求可变的当前 executive decision，前端 E2E 仅要求生命状态页展示 `RuntimeEvidenceStrip`，其余页面显式验证不展示；本地对应 Workbench 回归为 `2 passed`，mypy 为 `45 source files` 无问题。远端运行 `33295880356` 已完成最终验收，Docker、前端含 E2E、Legacy/no-Legacy smoke、Python 3.10/3.12 与 Windows 全量回归 7 个 job 全部成功。

R5-G1 合同 Gate 已新增两份独立 manifest，并覆盖合法合同与缺 owner、混合 owner、缺 checkpoint、认知越权、错误删除边界的 red contract。R5A-S0 已实现 `taiji/internalization.py`：纯 grounded Outcome DTO、内容寻址、train-only replay 去重、生命周期/五项因果门控和 checkpoint roundtrip；R5A-S1 新增 `taiji/internalization_learner.py`：原生局部更新、父/子 checkpoint lineage、留出/保持集只读评估和 feature/grounding lesion；定向测试 `19 passed`，S1 canary `gate.passed=true`。本轮 CI `33298105636` 发现最小 torch 环境缺 NumPy 时 digest 路径的跨平台失败，`513cb1f` 已改为纯 PyTorch 字节视图并加入无 NumPy contract；后续全矩阵运行 `33298754868` 的 7 个 job 全部成功。R5B 生产实现仍未开始。

## 4. 明确未完成

- R5A-S1 已有原生学习器和 synthetic holdout/lesion 收益证据，但没有真实 Workbench 纵向收益、跨任务保持和通过五类 Gate 的可删性判据；R5A-S2 仍未完成。
- `seed_platform/workbench.py` 的能力执行仍依赖硬编码分派；没有统一 capability bundle 注册、disposer、候选/影子/激活生命周期。
- `taiji_w7_r5_open_domain_growth_v1.json` 只冻结结构成长合同，不能覆盖知识内化或效应器成长；R5A/R5B 的独立 manifest 已创建，但生产转换器/注册表尚未实现。
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
