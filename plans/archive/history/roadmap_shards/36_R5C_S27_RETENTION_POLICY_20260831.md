# R5C-S27：版本化、内容寻址的 lineage retention policy

状态：已完成（2026-08-31）

## 目标

把 retention 上限从裸整数提升为 Taiji-owned 的版本化 policy snapshot，消除调用方、全局常量和内部 ledger 之间可能出现的多套语义，同时保留现有整数入口的受控兼容性。

## 实现边界

- 新增 `StructuralLineageRetentionPolicy`，以 revision、max_batches、固定安全 protection rules 和 policy digest 定义完整 policy。
- `TSKV8Adapter` 接受 policy object/payload；旧 `max_batches` 只在边界转换为同一 v1 policy。
- policy 与 retention result 一起进入 structural checkpoint，并在 restore 时检查 policy/result 上限一致性。
- `StructuralMaintenanceAudit` 和 runtime status 同步投影本次/最近使用的 policy。
- policy 不能关闭 active reservation、pending candidate、pending topology proposal 或 rollbackable admission 保护规则。
- policy 切换只影响后续显式 maintenance，不回写旧 retention result，不启动后台维护。

## Gate

真实 Workbench evidence 的 native/CPU canary：

`scripts/training/eval_taiji_structural_lineage_policy.py`

必须同时证明：

1. 相同 policy payload 跨实例 canonical roundtrip，digest 稳定；
2. 旧整数入口与 policy 入口产生相同 retention 语义；
3. policy、retention result、status projection checkpoint restore 一致；
4. policy 切换只影响后续显式 maintenance；
5. 双重输入、未知 revision、非法 protection rules、digest 篡改和 policy/result 不一致 checkpoint fail-closed；
6. 不改变 topology、budget 或保护 lineage 规则。

## 证据

- 报告：[taiji_w7_r5c_s27_structural_lineage_policy_20260831.json](../../../../reports/taiji_w7_r5c_s27_structural_lineage_policy_20260831.json)
- Gate：`gate.passed=true`
- 定向用例：`tests/taiji_native/test_structural_lineage_policy.py` 为 `4 passed`
- S18–S27 相关回归：`34 passed`
- 语法/lint：Ruff、compileall、`git diff --check` 通过

## 明确未覆盖

- 不支持未知 policy revision 的隐式迁移。
- 不把 policy/status 变成 candidate 准入或结构成长信号。
- 不声明无限增长、自动增加预算、开放域质量或全面自进化。
- 不声明 CUDA、Windows shell、前端视觉或完整 CI 通过。

## 唯一后继

R5C-S28：建立 retention policy 的显式迁移、版本兼容和回滚生命周期，验证 schema 演进不会破坏既有 lineage。
