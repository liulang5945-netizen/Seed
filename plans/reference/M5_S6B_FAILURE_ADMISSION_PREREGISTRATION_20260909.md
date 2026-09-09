# M5.S6B 预注册：失败证据进入内化（执行安全边界变更）

> 起草日期：2026-09-09。用户已确认选择选项 B。本文是边界变更的预注册，实现与 canary 按本文执行。执行顺序以 [03_CURRENT_EXECUTION.md](../active/roadmap/03_CURRENT_EXECUTION.md) 为准。

## 1. 边界变更的精确范围

现状（两处）：

1. `seed_runtime.py` `project_workbench_outcome_for_internalization`：`if not evidence.success: raise ValueError("failed Workbench evidence cannot enter internalization")` —— 运行时投影层硬拒绝失败证据；
2. `taiji/internalization.py` `InternalizationConverter.convert`：对 `Outcome.success` **无任何检查**——失败进入是隐式的、不可审计的。

变更（同一政策，两层对齐）：

- **失败证据准入为显式一等政策**：`Outcome.success=False` 的真实执行证据可以进入内化，条件是 `reward <= 0`（失败携带正 reward 一律拒绝，fail-closed）；
- converter 层：新增显式校验 `failure ⇒ reward ≤ 0`（违反即 `_reject("failure_with_positive_reward")`），lifecycle 事件新增 `failure_admitted`；
- runtime 投影层：`raise` 替换为同一校验（失败 + 正 reward → raise；失败 + 非正 reward → 准入），snapshot 绑定、affordance grounding 校验、reward 外部注入**全部不变**；
- 成功证据的政策不变（reward 由 evaluator 注入，可正可负）。

**不变的部分**：执行安全边界（approval token、digest 校验、undo、隔离工作区、路径越界拒绝）、`can_promote=false`、内化 Gate 五控制。

## 2. 依据

- S5：常量 target 使 lesion 结构性坍缩；target 携带信息时 margin 恢复（0.003~0.46）；
- S6 选项 A formal 9/9：真实分级 reward（仅成功执行）已恢复 lesion，但成功执行不包含真实失败——"有来源经历"的失败半边缺失；
- 环境层事实：`execute_tool` 对失败已返回真实 `EnvironmentOutcome(success=False, reward=-1.0)` 与结构化 `error_code`（not_found/not_a_file/unsafe_path 等）——失败在执行层本就是一等结果，环境的 reward 惯例（失败=−1）与选项 B 语义一致；
- 学习论依据：只有正例的学习不能形成"什么导致失败"的表征；真实失败是最强的有来源经历之一。

## 3. Gate（沿用 + 强化）

- S1 五控制不变；grounding lesion margin floor `0.05` 不变；
- 新增检查：`failure_evidence_admitted`（失败证据经 converter 显式准入且 lifecycle 带 `failure_admitted`）、`no_failure_with_positive_reward`（政策反向测试）；
- 3×3 formal（task_seed × learner_seed）全 cell 过 floor + 过完整 gate 才判稳健（S6 教训）。

## 4. 停止线

- 单元测试先行钉住新政策（失败+负 reward 准入、失败+正 reward 拒绝），现有成功路径测试不得退化；
- 不放宽 approval/undo/隔离工作区的任何执行安全语义；
- canary 失败 → 回到 reward 派生设计，不调 floor。
