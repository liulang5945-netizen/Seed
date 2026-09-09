# M5.S6 预注册：真实执行 outcome 驱动的内化（reward 真实化）

> 起草日期：2026-09-09。本文把 M5.S5 诊断锁定的根因（常量 target 使 grounding lesion 结构性退化）落成下一个实验的可证伪设计。outcome 来源已由用户确认为「M3 隔离工作区真实执行」。确认本文 §4 的 reward 派生决策前不写实现、不跑 canary。执行顺序以 [03_CURRENT_EXECUTION.md](../active/roadmap/03_CURRENT_EXECUTION.md) 为准。

## 1. 假设

> 用真实 Workbench 执行产生的、随任务结果变化的 reward 替换常量占位后，grounding lesion 与 internalized lesion 探针在真实 384 维 embedding / world-state grounding 上恢复可观测的因果边际（S5 已证明 target 携带信息时 margin 从负恢复到 0.003~0.46）。

若真实变化的 reward 下 lesion 仍不恢复，则内化管线的 lesion 语义存在比常量 target 更深的缺陷，M5 内化证据线冻结重审。

## 2. 已确认前提（S4/S5 证据）

- S5：`score(grounding_enabled=False)` 只返回 bias；常量 target 下 bias 完美预测 → weights 无需学习 → grounding lesion 坍缩（margin 为负）。**变化的 target 是让探针有意义的必要条件。**
- 真实执行同时提供两个变化源：(a) world-state grounding 的 17 维特征随 after-state（路径/result/transaction digest）变化；(b) reward 随任务成功质量变化。

## 3. 复用与新建

| 组件 | 状态 |
|---|---|
| 隔离执行回路 | 复用 M3.R5 `WorkbenchEnvironment(root=临时目录)` + approval/consume + `execute_tool` + undo，进程私有临时目录，不触碰仓库路径 |
| 证据→内化输入 | 复用 `SeedRuntime.project_workbench_outcome_for_internalization`（已实现 snapshot 绑定、affordance grounding 校验、reward 外部注入）|
| grounding 特征 | 复用 `WorldAffordanceGroundingProducer.ground`（17 维，从 WorldState 数值投影）|
| 内化 learner/ledger | 复用 `InternalizedFeatureLearner` / `InternalizationLedger` / `InternalizationCausalGate`（S1 语义）|
| 新建 | 一个「隔离执行一批真实任务 → 逐条投影 → 喂内化 → 测 lesion」的胶水 canary 脚本 |

## 4. 核心决策：变化的 reward 如何派生（**待用户确认**）

S5 要求 target 变化，但 `project_workbench_outcome_for_internalization` 硬性拒绝 `success=False` 的证据。两条出路：

**选项 A（推荐，低风险首版）：只在成功执行上派生分级 reward，不改证据边界。**
- 一批真实只读/写任务全部成功（通过 rejection gate），reward 由**真实后置条件评估器**给出分级值：例如只读任务的 reward = 读取命中目标内容 digest 的程度（全中 +1.0、部分命中 0<x<1、结构成功但内容非目标 −0.x），全部是 `success=True` 的证据但 reward 连续变化。
- 符合投影函数契约（reward 由 evaluator 注入，不从 capability 名臆造目标）；不触碰执行安全边界。
- 局限：失败态不进入学习，reward 变化幅度受"成功任务的分级质量"约束。

**选项 B（上限更高，需显式改边界）：允许失败证据以负 reward 进入内化。**
- 需要修改 `project_workbench_outcome_for_internalization` 的 `if not evidence.success: raise`，改为"失败证据带负 reward 可入、但仍需 snapshot 绑定与 grounding 校验"。
- 真实 success/failure 是最自然的变化 target，上限最高。
- 代价：改动一个刻意拒绝失败的学习准入边界，属执行安全语义变更，必须单独评审、单独回归、默认路径保护。

**建议顺序**：先 A 出 canary 证明"真实变化 reward 恢复 lesion"这条因果链闭合；若 A 的变化幅度不足以让 grounding lesion 稳定过判据，再按 B 提一个独立的证据边界变更预注册。不在 A 未跑前就改边界。

## 5. Gate 与判据（沿用 S1 + 新增）

- 沿用 S1 五控制（外部充分性、内化必要性、来源必要性、checkpoint、retention）与 S2/S3 技术检查；
- **新增 grounding lesion 判据**：`margin = holdout_grounding_lesion_loss − holdout_loss_after` 必须 ≥ 预注册下限（A 方案先设 `≥ 0.05`，与 S5 非归一化信息 target 的 0.46 同量级下界，避免重蹈"名义通过"）；
- 语义/来源探针继续 report-only；
- `can_promote=false` 固定。

## 6. 停止线

- canary 先行（单批任务、小 N），lesion 恢复才进多任务/多 seed formal；
- 任一 S1 控制失败或 lesion 边际低于下限 → 不晋级，回到 §4 决策重审（先判是 reward 变化幅度不足还是管线更深缺陷）；
- 不借真实执行之名引入 provider/联网/真实客户端写入；隔离临时目录 + undo + digest 校验全部保留；
- 不通过放宽 lesion 下限来"修"探针（S5 的教训）。

## 7. 确认点

1. §4 选 A（成功上分级 reward，不改边界）还是直接 A→B 两步都排；
2. §5 grounding lesion 下限 `0.05` 是否接受；
3. canary 任务规模（建议：约 200~400 条真实只读任务，覆盖命中/部分命中/非目标三档）。
