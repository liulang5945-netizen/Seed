# R5C-S29：SeedRuntime 磁盘 checkpoint 继续与回滚

状态：已完成（2026-08-31）

## 目标

把 S28 的 retention-policy migration 从内存对象推进到真实 `SeedRuntime.save()` / `SeedRuntime.load()` 磁盘边界，证明重启后仍可观察、继续和回滚，而不是只在 evaluator 进程内成立。

## 实现边界

- 真实 SeedRuntime checkpoint 保存并加载 retention policy、migration、retention result、status projection 和已压缩 lineage。
- 加载后显式 rollback migration，再次保存和加载，确认旧 policy、rollback 状态和 retention result 继续可用。
- 已删除的 terminal lineage 在两次磁盘恢复后都不会复活；topology 与 structural budget 不被 rollback 改写。
- 篡改 migration digest、缺失配置字段和原文件读取均 fail-closed；临时调试过程不改变生产 checkpoint 语义。
- 测试使用仓库现有的 `output/manual-r5-canary` 证据目录，避免本机 `%TEMP%` 中遗留 pytest 进程造成的 Windows 访问拒绝；每次测试结束清理自身唯一命名的 checkpoint 文件。

## Gate

真实 Workbench evidence + native/CPU 磁盘 canary：

`scripts/training/eval_taiji_structural_lineage_disk_checkpoint.py`

必须同时证明：

1. runtime 磁盘 checkpoint 可保存、加载并保留 retention audit、policy、migration、status 与 result；
2. 已删除 terminal lineage 不会在加载或第二次保存/加载后复活；
3. 加载后的显式 rollback 恢复 v1 policy，且 rollback 结果可以再次落盘恢复；
4. topology、structural budget 和原运行时状态不因 checkpoint/rollback 改变；
5. 篡改 migration、缺失关键配置字段拒绝加载，原 checkpoint 字节与运行中状态保持不变。

## 证据

- 定向用例：[test_structural_lineage_disk_checkpoint.py](../../../tests/taiji_native/test_structural_lineage_disk_checkpoint.py) 为 `1 passed`。
- Canary：[taiji_w7_r5c_s29_structural_lineage_disk_checkpoint_20260831.json](../../../reports/taiji_w7_r5c_s29_structural_lineage_disk_checkpoint_20260831.json)，`gate.passed=true`。
- 两个真实磁盘 artifact 均成功生成并加载，约 3.1 MB；迁移后与回滚后 checkpoint 均通过，tampered/incomplete checkpoint 均 fail-closed。
- Ruff 定向检查通过；本轮不运行 CI，不把本地 Gate 扩大为远端 CI 结论。

## 明确未覆盖

- 不提供任意字段的通用 checkpoint 签名；本 Gate 覆盖 migration/result/status 现有内容寻址和 SeedRuntime 版本/字段恢复边界。
- 不把磁盘恢复等同于无限增长、自动增加预算、开放域质量、全面自进化、CUDA、前端或 CI 通过。
- 不启用后台 retention、不隐式迁移 policy、不启动训练。

## 唯一后继

R5C-S30：在重启恢复后的 runtime 上执行新的显式 structural maintenance continuation，验证 policy、cursor、audit 和 lineage 继续消费新 evidence 且不重放旧动作。
