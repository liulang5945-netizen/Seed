# R5C-S17：多轮 artifact/measurement integrity 与 provenance closure

> 状态：已完成（2026-08-30）
>
> 本 slice 补齐 S14–S16 之后的内容寻址边界：measurement payload 不再只检查 digest 非空，新的 validation artifact 显式保存 measurement provenance，同时兼容历史 artifact payload。

## 1. 目标

S14 的 measurement owner 已能从 raw replay/resource observation 计算 metrics，但其 payload 反序列化此前没有重新计算 `measurement_digest`。S15/S16 的 artifact 能验证自身 digest，却没有一个显式字段把 artifact 与 measurement owner 的 digest 连接起来。S17 将这两层绑定收敛为可验证的 provenance 链。

## 2. 实现合同

| 对象 | 合同 |
|---|---|
| measurement integrity | `StructuralValidationMeasurements` 在构造和 `from_payload()` 时依据全部 metric、raw probe digest、resource digest 和 format 重算 measurement digest |
| artifact provenance | `WorkbenchStructuralValidationArtifact` 保存可选 `measurement_digest`；新 measurement-produced artifact 必须携带该绑定，artifact digest 覆盖该字段 |
| tamper boundary | metric、raw digest、resource digest、measurement digest、format 或 artifact binding 的修改必须 fail-closed |
| legacy compatibility | 旧 artifact payload 没有 `measurement_digest` 时仍按旧 payload 形状校验并可恢复；新 artifact 不得静默丢失 measurement binding |
| checkpoint | artifact ledger restore 后保留 measurement/artifact digest 关系；provider、frontend、Workbench executor 不拥有修改权限 |

## 3. Gate

canary：`scripts/training/eval_taiji_workbench_integrity.py`。

结果：`gate.passed=true`。

覆盖：

- 真实 Workbench evidence 生成 candidate-bound measured artifact；
- tamper measurement metric、measurement digest、artifact measurement binding 均 fail-closed；
- measurement raw probe/resource digests 保持非空且内容寻址；
- 新 artifact 显式绑定 measurement digest，旧无字段 artifact 仍可 roundtrip；
- artifact ledger checkpoint restore 后 digest/provenance 保持。

相关回归：S14 measurement canary 与 S15 measured multi-region artifact batch 均重新通过，且 S15 的“无手工 metrics”判定改为显式布尔值。

## 4. 未关闭边界

- S17 只闭合 payload integrity 和 provenance 绑定，不改变 pressure projection 的长期历史聚合策略；
- 历史 sealed windows 与当前轮消费集合的压缩/审计由 S18 负责；
- 不扩展 CUDA、CI、provider 自治、无限预算或开放域语言质量。

## 5. 下一步

R5C-S18：建立多轮 ledger compactness 和跨轮 evidence 消费审计，确保长期运行既能保留 lineage 又不会重复消费历史证据。
