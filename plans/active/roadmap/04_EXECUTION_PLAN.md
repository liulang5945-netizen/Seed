# Seed / Taiji 后续详细开发计划

> 计划基线：2026-08-30。本文件给出后续阶段、依赖和验收，不创建第二个即时入口；当前只执行 [03_CURRENT_EXECUTION.md](03_CURRENT_EXECUTION.md) 指定的一项。已完成的 W0–W7 历史蓝图见 [SEED_W7_EXECUTION_PLAN_SNAPSHOT_20260829.md](../../archive/history/SEED_W7_EXECUTION_PLAN_SNAPSHOT_20260829.md)。

## 1. 收敛后的架构主线

Taiji 不从“原始神经元模拟”重新发明全部技术，也不让 Transformer/provider 接管 cognition。后续开发沿四条有所有权的链路闭合：

```text
真实任务 Outcome ─→ R5A 知识内化 ─→ Taiji-owned 选择/记忆参数
能力包与执行器 ───→ R5B 效应器注册 ─→ Workbench 身体能力
长期误差/容量压力 ─→ R5C 结构提案 ─→ shadow/holdout/lesion/rollback
已形成的 ContentPlan ─→ 语言 provider ─→ 可读输出（不参与前三条决策）
```

这四条链共享 lineage、资源账本和 checkpoint，但不能共享 owner。特别是：

- **描述性知识可以在验证内化后删除；真实执行器不能因“模型学会选择”而删除。**
- **能力注册不等于能力自治；只有 Taiji 的 structured affordance 选择、policy 准入和真实 Outcome 回写闭合后才算身体成长。**
- **结构增加不等于进化；只有旧能力保持、资源可接受且可回滚的增益才进入稳定模型。**

## 2. 阶段总览

| 顺位 | 工作包 | 目标 | 退出产物 |
|---|---|---|---|
| C0 | 计划与事实源收敛 | 去除多重下一步、拆出历史、固定阻塞边界 | 本轮计划提交 |
| C1 | W7-R5-G1 合同分离 | 分别冻结知识内化与效应器成长合同 | 两份 manifest + contract red/green |
| C2 | W7-R5A 知识内化 | 让真实 Outcome 形成可恢复、可 lesion 的 Taiji-owned 学习 | S0/S1/S2 + 可删性账本 |
| C3 | W7-R5B 效应器成长 | 从硬编码分派演进为内容寻址、可撤销注册生命周期 | L0–L3；L4 独立评审 |
| C4 | W7-R5C 自进化闭环 | 让长期证据触发局部结构变化，而非全量重训 | shadow→admit/rollback 纵向 Gate |
| C5 | 语言与 provider 成熟 | 提高可读表达并验证外部 artifact 轮换 | packaged canary + 质量/安全回归 |
| C6 | 工作台自治扩展 | 在已有只读闭环上逐级开放受控写入和长程任务 | 审批/撤销/恢复/长期评测 |
| C7 | 阻塞线补证 | 恢复 R3 Windows shell 与 R4 CUDA | 各自 S2/硬件报告 |
| C8 | 发布收口 | 全门禁、文档、安装包和远端发布一致 | release candidate |

R3/R4 未通过时不得声明相应能力，但它们不再作为 R5 的伪串行依赖。C7 一旦具备工具或硬件即可插入执行；插入后仍须单独提交，不与 R5 改动混合。

## 3. C1：W7-R5-G1 合同分离（已完成基线）

### 目标

创建两份而不是一份混合 manifest：

- `taiji_w7_r5_internalization_v1.json`：知识/规则内化、replay、可删性和遗忘边界；
- `taiji_w7_r5_effector_registry_v1.json`：能力包、执行器注册、snapshot、隔离、卸载和回滚。

选择分离的原因是失败后果不同：错误删除描述知识可以回滚 artifact；错误删除执行器等同截肢。把两者塞进一个 Gate 会允许某一侧通过掩盖另一侧未验证。

### 必须冻结的合同

- owner 与禁止依赖；真实 input/output DTO；内容 digest 与 revision；
- checkpoint 必存字段、旧版本兼容和 continuation；
- S0/S1/S2、red proof、holdout、lesion、资源、失败隔离、rollback；
- “可以物理删除什么、永远不能自动删除什么”；
- 与既有 `taiji_w7_r5_open_domain_growth_v1.json` 的依赖关系，三者不得互相冒充完成。

### Gate

已扩展 `tests/test_w7_gate_manifests.py`：缺失/混合 owner、缺 checkpoint、认知越权和错误删除边界会红，合法合同通过；R5B manifest 已推进到 `s1_dispatch_integrated`，R5A 的实现状态由其 S0/S1/S2 分阶段记录。效应器注册表仍由 R5B 独立拥有，避免 R5A 与 R5B 的 owner 边界漂移。

## 4. C2：W7-R5A 知识内化（下一阶段）

### S0：纯 DTO 转换与确定性 replay

- 在 `taiji/internalization.py` 定义不依赖 `seed_platform` 的 Outcome/evidence DTO、内容 digest 和训练样本转换器。
- 真实 affordance 必须来自 grounding；失败记录不得凭 `capability_id` 造出 affordance。
- `reward_terms` 缺测即缺失，越界直接 fail-closed；样本 ID 由 evidence + affordance 内容寻址。
- 有界 replay buffer 按内容键去重，训练/holdout 分区不可写穿。

退出：相同 checkpoint、manifest、evidence digest 和 seed 产生相同样本/账本；污染 holdout、伪造 grounding、重复 evidence 或越界奖励均红。

### S1：native checkpoint 与离线巩固（已完成 synthetic canary）

- Seed runtime 只负责把真实 Workbench Outcome 投影为 DTO；Taiji owner 批量巩固，不逐条重置优化器状态。
- checkpoint 保存 replay digest、训练计数、外部 artifact 绑定、`external → shadow → internalized/tombstone` 生命周期。
- 恢复后继续一步，选择结果、计数、lineage 与预算一致。

当前实现：`taiji/internalization_learner.py` 使用无优化器重置的归一化局部更新；父 checkpoint 在 trial mutation 前保存，holdout/retention 只读，feature/grounding lesion 可观测，恢复后 online counter 可继续。S1 synthetic native canary 已通过；S2-B 已在真实只读 Workbench 任务上补齐纵向 selection、train-only preference、外挂移除、feature/grounding lesion、旧任务保持、checkpoint recovery 与 candidate-only deletion boundary；S2-C 已用独立 seed 11/29 与 task slice 通过稳定性、资源和独立删除评审。

### S2：真实 Workbench 纵向证据

- S2-A 已完成：`SeedRuntime` 只把当前、签名验证过的只读 `workbench.evidence` 与其同 snapshot 的重投影 grounded successor affordance 投影成 `GroundedOutcomeEvidence`。运行时不可写 replay、训练 learner 或推进 lifecycle；陈旧 snapshot、旧 affordance 和失败 evidence 全部 fail-closed。
- S2-B 已完成：使用未参与训练的新任务组合执行真实只读 Workbench；比较外部规则存在与移除后的选择质量，并完成外部/内化/grounding/retention/checkpoint 五类 Gate。通过后只推进 Taiji replay 生命周期并生成可恢复的 deletion candidate，不删除外部 artifact。
- S2-C 已完成：`InternalizationStabilityTrial`/`InternalizationStabilityGate` 在独立 seed 与任务切片上汇总收益、保持、lesion、指标离散度、bounded resource counters 和 checkpoint digest；`IndependentDeletionReview` 独立检查 artifact 内容寻址、候选理由、lifecycle/manifest/checkpoint 绑定和物理删除边界。真实 Workbench 用例以两个实际 seed/task slice 在主机系统临时工作区执行通过；未通过稳定性或评审时，artifact 保持 active，不能进入 R5B/R5C。
- 只有 S2-C 稳定性与独立删除评审也通过，才允许讨论外部描述的物理删除提交；真实执行器、MCP 通道和 capability bundle 永远不在此删除范围内。R5B-L0 已先建立 capability bundle 的独立生命周期合同，物理删除仍不在范围内。

## 5. C3：W7-R5B 效应器成长

### L0：注册表重构，能力集合不变

- 已新建 `seed_platform/capability_registry.py`，完成 bundle 内容寻址、snapshot/revision、生命周期记录、disposer 约束、stale fail-closed 和 checkpoint roundtrip 合同；`WorkbenchEnvironment.execute_tool()` 已改为 registry resolve → executor identity → 原生执行表，旧 `elif tool_name` 分派已移除。全量 enabled capability 覆盖、request/approval snapshot binding、原子 replacement/rollback 和 checkpoint continuation 已有定向/直接集成证据，`scripts/training/eval_capability_registry.py` 的 R5B-S1 evaluator 也已报告 `gate.passed=true`。
- 注册返回 disposer；卸载、替换和失败回滚不直接修改全局散列表。
- `CapabilitySnapshot` 由已装配 bundle 内容生成 digest + revision；装配变化后旧 evidence 自动 stale。
- 15 个既有能力、错误码、policy 和 Legacy-off 行为逐项等价。

当前退出：registry 自身与 Workbench request/approval 的未知/禁用/陈旧能力全部 fail-closed，注册不自动激活，side-effecting bundle 缺 disposer 不能注册；核心 replacement/rollback Gate 已通过，最终退出还要在正常 CI 环境跑完整 Workbench 文件回归。

### L1：候选能力包

- 包含 schema、effect/risk/reversibility、权限、资源、版本、内容 digest、执行入口和卸载器。
- 校验、预编译和保存候选与激活分离；候选默认 `proposed`，不因落盘自动可执行。
- 文本描述只供审计，不供 provider/LLM 选择工具。

当前实现：`CapabilityCandidate` 将 bundle 与 rationale、evidence digests、有限数值 resource budget、evaluation gates 和 metadata 组成独立内容寻址 artifact；`CapabilityRegistry.propose()` 不注册 bundle，`validate_candidate()` 只进入 `validated`，`reject_candidate()` 保留拒绝审计；candidate 与 lifecycle 均进入 checkpoint，嵌套 executable-source 字段 fail-closed。L1 evaluator 已通过，下一步进入 L2 shadow 差异 Gate。

### L2：shadow 与审批

- 在相同输入上做影子执行或无副作用模拟；记录结果、after-state、资源和风险差异。
- 需要真实副作用的能力必须经过产品 policy/用户审批；Taiji 不可绕过。

当前实现：`CapabilityShadowObservation` 以 digest 绑定输入、baseline/candidate 输出、after-state 和资源指标；`evaluate_shadow()` 只解析 registry 中的 shadow bundle，不调用 executor，read-only 等价、policy deny、stale snapshot、side-effect detection 和 side-effect approval 均有独立结果。L2 evaluator 已通过，下一步进入 L3 原子激活和资源回滚。

### L3：可撤销激活

- 原子更新 snapshot、registry、资源账本和 checkpoint；失败恢复上一装配。
- 真实 Outcome 回写 R5A/R5C，但注册表本身不学习认知内容。

当前实现：registry 激活与 replacement 先计算 active bundle 的完整 resource reservation，超限会在任何状态提交前 fail-closed；`resource_ledger` 与 prior reservation 进入 checkpoint，rollback 恢复父 active set 与资源使用，并对 disposer 释放留下审计事件而不执行未知 source。L3 evaluator 已通过，L4 仅允许进入独立架构评审。

### L4：纯计算执行体替代

只有无外部副作用、可用独立 oracle 完全验证的纯计算能力可进入 L4。它需要单独架构评审，不随 L0–L3 自动推进。

2026-08-30 评审结论见 [05_R5B_L4_PURE_COMPUTATION_REVIEW_20260830.md](05_R5B_L4_PURE_COMPUTATION_REVIEW_20260830.md)：当前 workspace、IDE、terminal、MCP 和编辑能力都不能证明同时满足显式值输入、确定性无副作用、独立 oracle 三项条件，因此暂不实施任何 L4 executor。后续只有新候选满足该准入合同，才重新打开评审。

## 6. C4：W7-R5C 结构成长与自进化

复用已有 structural growth、topology ledger、neuron/region growth 与 rollback 基础，不另起“原始神经元”架构。R5C-S0 先把 R5A/R5B 的长期真实 evidence 做成内容寻址、去重、可 checkpoint 的观察窗口；触发输入来自窗口聚合后的持续错误簇、恢复不足、容量饱和、遗忘和资源压力，而不是单个 tick、规模目标或人工标签。

固定生命周期：

1. 保存 parent checkpoint；
2. owner 提出局部连接、神经元、区域、记忆容量或 pruning/merge 候选；
3. 在预算内 shadow learn；
4. 独立 holdout 与 lesion；
5. 比较收益、遗忘、恢复时间、参数/连接、内存、延迟和能耗近似；
6. 原子 admit，或恢复 parent 并写 tombstone；
7. 跨 seed/任务片稳定后才扩大长期容量。

禁止全量从零训练作为默认迭代方式；允许基础模型版本升级时进行受控迁移，但必须保留父 lineage、旧能力回归和可逆转换。

## 7. C5–C6：语言成熟与工作台自治

### 语言/provider

- `native-readable` 保持默认产品表层；`structured-stub` 只作为显式无损调试 codec。
- 完成真实外部 provider artifact 的 packaged-client 轮换、失败回退、重启重绑和安全 canary；R1 的 native-only S2 不冒充该能力。
- 质量评估绑定相同 ContentPlan，比较可读性、约束保持、事实遗漏和 fallback；provider 不改 intent、tool 或 memory。

### 工作台自治

- 继续沿已有 IDE 语言识别/高置信自动切换、policy、预览、审批、undo 和 Outcome 链扩展。
- 默认自治先从可逆、小影响写入开始，再到跨文件任务；每层设置预算、人工接管点和 checkpoint continuation。
- HF/GGUF/Transformer 继续只存在于 provider/离线对照边界，前端不得恢复模型格式切换残留。

## 8. C7–C8：阻塞线与发布

### R3 Windows shell

工具可用后只补真实窗口、任务栏、托盘通知、高 DPI、键盘导航和 reduced-motion 证据；已通过页面证据不重做。失败不修改 Taiji checkpoint。

### R4 CUDA

真实 CUDA 主机到位后，先运行同一 CPU workload 的 profiler，再验证 CPU→CUDA→CPU checkpoint、结构/lineage/预算一致和数值容差；只有热点证据支持时才评审 fused/sparse kernel。

### 发布收口

- 后端、前端、桌面、Legacy-off、checkpoint、manifest、OpenAPI、安全和安装包 Gate 全部执行且无 skipped；
- `dist/Seed/Seed.exe`、前端字节、报告 digest、文档状态和 Git commit 对齐；
- 当前计划只留未完成阶段，已完成执行日志归档；
- 发布前实时检查本地 `main`、其他 worktree refs、`origin/main` 和远端同步，不把“已提交”写成“已推送”。

## 9. 每个 slice 的固定交付格式

1. 先写 red contract 与失败证据；
2. 做最小 owner 内实现，不顺手扩展相邻能力；
3. 运行与改动范围匹配的阻塞 Gate；
4. 生成内容寻址报告，记录未验证边界；
5. 更新 manifest、实现事实和唯一下一步；
6. 单一主题提交；若 CI 红，下一提交只修 CI，不能继续堆功能。
