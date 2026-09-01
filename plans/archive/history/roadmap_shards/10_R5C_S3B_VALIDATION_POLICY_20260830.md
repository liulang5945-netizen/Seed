# W7-R5C-S3B：independent validation policy

## 目标

把 candidate 是否值得进入后续 admission 的判断从执行器中抽离，形成可配置、内容寻址、无副作用
的独立 policy。policy 同时检查 holdout gain、old-task retention、lesion causal effect、
resource state 和 structural budget，避免任何单一指标或硬编码角色直接触发成长。

## 已实现

- `taiji/structural_validation.py` 新增 `StructuralValidationGateDecision` 与
  `evaluate_structural_candidate_validation()`。
- 阈值由调用方显式提供并写入 decision digest：最低 holdout gain、最大 retention regression、
  最低 lesion effect、最低 resource state；resource cost 还必须不超过当前 structural budget。
- 任意一项失败都会生成可审计 reason，并使 `passed=false`；decision 不修改 adapter、topology、
  controller、budget 或 evidence ledger。
- decision 支持 payload roundtrip，并拒绝 digest 篡改、重复 evidence 和不一致的 passed/reasons。

## Gate 证据

- `tests/taiji_native/test_structural_validation.py`：3 个 policy contract 测试通过。
- `scripts/training/eval_taiji_structural_validation_gate.py`：合法/失败候选、五类失败维度、
  digest roundtrip 和 non-mutating boundary 均通过，报告 `gate.passed=true`。

## 当前边界

S3B 只定义并验证 metric policy，不负责采集真实 retention/lesion 数据，也不调用 topology
admission。真实模型指标接线必须由下一 slice 完成，并继续绑定 S3A validation record。

## 下一阶段准入

R5C-S3C 将把真实 candidate shadow、holdout、retention 与 lesion 观测接到该 policy；只有 policy
通过且 parent/trial checkpoint、resource reservation 和 rollback 都成立，才允许进入 admission
评审，仍不自动扩大长期结构。
