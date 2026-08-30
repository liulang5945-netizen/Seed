# W7-R5C-S4：cross-seed/task-slice stability and rollback

## 目标

验证一条受限 structural growth contract 是否能在独立 seed 和 task slice 上重复，并确认
admission 后旧任务保持、grown unit 的 lesion 因果效应、预算记账和 parent rollback 都稳定。

## 已实现

- `scripts/training/eval_taiji_structural_stability.py` 以 seed 11 和 seed 29 重复
  `pressure → candidate → shadow → validation policy → admission`。
- 每个 seed 独立计算 old-task retention regression 和 grown-unit lesion effect；不是只复用
  固定布尔值或 scale target。
- 每个 seed 都从 admitted checkpoint restore，再执行 rollback，检查原始 unit identity 和
  structural budget 恢复。
- 只有两组都满足 policy、单次 topology growth、retention、lesion、rollback 才通过总 Gate。

## Gate 证据

- 报告：`reports/taiji_w7_r5c_s4_structural_stability_20260830.json`，`gate.passed=true`。
- seed 11：retention regression `0.0`，lesion effect `0.155467...`，rollback 恢复 parent/budget。
- seed 29：retention regression `0.0`，lesion effect `0.157090...`，rollback 恢复 parent/budget。
- 两个独立 trial 的 policy、admission、rollback 检查全部通过；未启动训练、CUDA 或无限扩张。

## 当前边界

这是两个 seed 的受限 neuron birth canary，不是开放域智能证明，也不能证明多步连续扩张、
长期遗忘控制或更大预算下的稳定性。现有 admission 仍需由明确 evidence 和 policy 驱动。

## 下一阶段准入

R5C-S5 才能评估多步 bounded growth：预算分层、candidate concurrency、跨步骤 retention、
中途 checkpoint/恢复和失败后不复活；在 S5 通过前不能扩大默认 structural budget。
