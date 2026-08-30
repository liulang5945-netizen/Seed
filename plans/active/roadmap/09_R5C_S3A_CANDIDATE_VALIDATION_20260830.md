# W7-R5C-S3A：candidate shadow validation

## 目标

在 R5C-S2 candidate-only bridge 之后，建立一个不执行 admission 的候选验证层。验证结果必须
可 checkpoint、可恢复、可审计，并明确证明 candidate shadow 没有改动 parent topology 或
structural budget。

## 已实现

- 新增 `StructuralCandidateValidation`，记录 candidate/proposal、parent checkpoint digest、
  validation checkpoint digest、topology 前后 digest、预算前后值、evidence 和结果状态。
- `TSKV8Adapter.validate_structural_candidate_shadow()` 复用已有 operation-specific holdout
  validator，只 materialize pending proposal 和 shadow trial，不调用 `commit_*`。
- 合法 holdout 验证结果为 `validated`，proposal 仍保持 `pending`，等待后续独立 Gate；缺失或
  数量不一致的 holdout 返回 `failed_closed`，候选保持可重试且 parent/budget 不变。
- shadow validator 运行期异常会把已 materialize 且仍 pending 的 proposal 原子标为
  `rejected`；checkpoint restore 后不会把它重新放回 pending candidate 队列。
- validation records 已进入 `structural_runtime` checkpoint，恢复后保持内容一致。

## Gate 证据

- `tests/taiji_native/test_structural_pressure.py`：6 个结构 pressure/bridge/validation
  定向测试通过。
- `scripts/training/eval_taiji_structural_validation.py`：合法 candidate shadow、拓扑/预算
  不变、checkpoint roundtrip、malformed holdout fail-closed 与 rejected candidate 不复活
  均通过，报告 `gate.passed=true`。

## 当前边界

S3A 只验证单候选 holdout 与原子失败边界；它还没有把 retention 回归、lesion 因果贡献、跨
seed 稳定性和资源/延迟差异纳入 admission Gate，也没有调用真实 topology commit。

## 下一阶段准入

R5C-S3B 才能建立 retention/lesion 的独立 metric contract，并把这些 metric 与 S3A validation
record 绑定；在 S3B 通过前，任何 candidate 仍不能进入 admitted topology。
