# R5C-S14：独立 replay measurement owner

> 状态：已完成（2026-08-30）
>
> 本 slice 把 validation artifact 中最后一组由 evaluator 显式填写的 metrics 收回 Taiji-owned measurement owner，由 baseline/candidate/lesion 的原始 replay 张量和容量观测计算，并把计算输入与结果一并内容寻址。

## 1. 目标

S12/S13 已经完成 replay-bound artifact 和多区域 batch continuation，但 artifact 构建端仍可能把 `holdout_gain`、`retention_regression`、`lesion_effect`、`resource_state` 当作外部常量写入。S14 要求这些值由可复用、无 admission 权限的 measurement owner 计算，消除“调用方给分”的硬编码入口。

## 2. 实现合同

| 对象 | 合同 |
|---|---|
| measurement owner | `StructuralValidationMeasurements` 只读取 baseline/candidate/lesion replay 与原始容量 pressure，不改变 topology、预算或 lifecycle |
| metric derivation | holdout gain 来自 baseline/candidate 误差差值；retention regression 来自旧任务 candidate 相对 baseline 的误差变化；lesion effect 来自 lesioned 与完整 candidate 的误差差值；resource state 来自原始 capacity pressure 的有界测量 |
| provenance | 每个原始输入、容量测量、计算结果和 measurement payload 都有 digest；artifact 绑定 measurement digest，不能只绑定最终浮点数 |
| policy boundary | measurement 只提供验证事实；`continue_structural_candidate_from_validation_artifact()` 显式把 measured holdout gain 交给既有 policy，policy 仍拥有 admission 决策 |
| failure isolation | 缺失 baseline/candidate/lesion、错配候选、非法范围或错误 resource binding 必须 fail-closed，不得改变 topology 或预算 |
| recovery | measurement/artifact 与 parent/trial checkpoint lineage 可 roundtrip；恢复后重复生成相同 digest，重复消费幂等 |

## 3. Gate

canary：`scripts/training/eval_taiji_workbench_validation_measurements.py`。

结果：`gate.passed=true`。

覆盖：

- 使用真实 Workbench Outcome 产生 replay-bound evidence，并由 measurement owner 生成 baseline/candidate/lesion/resource 原始 digest；
- 计算并验证 measured holdout gain、retention regression、lesion effect 和 resource state，而不是从 evaluator 的 metrics 参数读取；
- 验证 artifact metrics 与 measurement owner 一致，policy 消费 measured holdout gain；
- 缺失或错误 resource measurement、错配 replay 和 payload tamper 均 fail-closed；
- checkpoint restore 后 measurement/artifact digest 保持，合法 artifact admission 与重复消费保持原有 lifecycle 语义。

## 4. 未关闭边界

- S14 仍是单候选 measurement/artifact 闭环；多区域同时消费独立测量 artifact 由 S15 负责；
- 当前 probe 证明的是结构验证链和 canary 指标来源，不等于开放域语言质量、无限结构扩张或全面自进化；
- 不扩展 CUDA、CI、provider 自治或全量重新训练。

## 5. 下一步

R5C-S15：把 measurement owner 接入多区域 artifact batch，确保每个 candidate 使用独立 measured metrics、resource digest 和 parent checkpoint。
