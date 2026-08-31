# R5C-S30：重启恢复后的 structural maintenance continuation

状态：已完成（2026-08-31）

## 目标

证明磁盘恢复后的 runtime 能继续接收新的 Workbench evidence，并且把“历史状态恢复”和“新一轮动作”分开：只有新的 task slice/window 可以推进 cursor、形成 candidate batch 和产生新的显式 maintenance audit。

## 实现边界

- 恢复 S29 的 v2 retention policy、migration、retention result 和已删除 terminal lineage。
- 追加 6 条新的真实 Workbench evidence，使用新的 task slices 和新的 candidate unit ids。
- 新 evidence 只推进 structural runtime tick/scheduler revision，并创建一个新的 batch。
- 显式 maintenance 产生新的 retention result；默认 maintenance 不重放恢复前的 retention audit。
- 第二次 checkpoint restore 后 continuation cursor、policy、audit、batch 与 no-replay 语义保持一致。

## Gate

真实 Workbench evidence + native/CPU restart canary：

`scripts/training/eval_taiji_structural_lineage_restart_continuation.py`

必须证明：

1. 新 evidence 能推进 structural tick 和 scheduler cursor；
2. 新 task slices 只创建新的 candidate batch，不重放旧窗口；
3. 显式 maintenance 生成新的 retention audit；
4. 已删除 lineage 不复活，默认 maintenance 不产生历史 retention 动作；
5. 第二次 checkpoint restore 保留 continuation state。

## 证据

- 定向用例：[test_structural_lineage_restart_continuation.py](../../../tests/taiji_native/test_structural_lineage_restart_continuation.py) 为 `1 passed`。
- Canary：[taiji_w7_r5c_s30_structural_lineage_restart_continuation_20260831.json](../../../reports/taiji_w7_r5c_s30_structural_lineage_restart_continuation_20260831.json)，`gate.passed=true`。

## 明确未覆盖

- 不在本 slice 执行 candidate admission、权重训练或结构无限扩张。
- 不把 restart continuation 视为开放域收益、全面自进化、CUDA、前端或 CI 证据。

## 唯一后继

R5C-S31：在重启后的新 batch 上继续 candidate-only replay、atomic admission、rollback 和 checkpoint continuation。
