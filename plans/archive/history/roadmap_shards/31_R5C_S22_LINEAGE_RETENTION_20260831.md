# R5C-S22：受保护 candidate/batch lineage 保留

## 目标

把 structural candidate retention 从“按最后 N 条保留”的隐含实现，收敛为以 live lineage 为中心的安全策略：保留仍能继续、占用资源或 rollback 的记录，只淘汰真正终结的记录。

## 已发现的风险

旧逻辑在 `_record_structural_candidate_batch()` 中按 batch id 排序直接删除旧项，在 `_queue_structural_proposal_candidate()` 中按插入顺序直接删除旧 candidate。batch id 是内容 digest，不代表时间；这种写法可能淘汰仍有 reservation、deferred continuation 或 admitted rollback 的结构链。

## 实现

- `TSKV8Adapter._structural_candidate_batch_has_live_lineage()` 保护 active reservation、reserved/deferred candidate 和尚未完成 rollback 的 admitted candidate。
- candidate batch retention 只从终结记录中按插入顺序淘汰，不再按 digest 排序盲删；若受保护项超过目标上限，保留超限状态而不破坏链。
- pending candidate queue 只淘汰未被活动 batch 引用的候选。
- validation、gate decision、validation artifact、admission、rollback 与 artifact-batch records 在 live continuation/rollback 依赖存在时使用同一保护判断；终结记录才可进入淘汰范围。

## Gate

报告：`reports/taiji_w7_r5c_s22_structural_lineage_retention_20260831.json`

必须全部满足：

- `active_batch_survives_terminal_eviction`
- `pending_candidates_survive_batch_retention`
- `rollbackable_admitted_batch_survives_terminal_eviction`
- `checkpoint_preserves_protected_lineage`
- `no_unsafe_eviction_under_pressure`

当前结果：`gate.passed=true`；lineage limit=1 的真实 Workbench canary 通过，R5C 定向回归 `31 passed`。

## 取舍

此阶段优先保证正确性和可逆性：受保护记录允许暂时超过目标上限。这样短期可能增加内存/ checkpoint 体积，但不会为了满足计数而静默丢失活动 lineage；真正的协同淘汰和 retention pressure 可观测性由 S23 统一处理。

## 边界

不执行物理删除，不扩大 structural budget，不处理 CI、CUDA、前端或开放域质量；不把终结记录淘汰等同于能力遗忘或模型权重重训。

## 后继

唯一后继为 R5C-S23：跨 candidate/artifact/admission/rollback 账本的协同终结保留 Gate。
