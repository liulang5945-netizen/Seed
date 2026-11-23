# SYN-3 预注册说明：双块协同（位组合标签；1 seed、≤300 CPU、零 check）

> [预注册说明] — 2026-11-23（D-20261123-331）。判据清单 [`taiji_u2_syn3_criteria_20261123.json`](../manifests/taiji_u2_syn3_criteria_20261123.json) 先行冻结（`criteria_sha256=085e77d3…a78d`，运行前校验）；本文件为设计说明。上游 SYN-1 系列（D-321…325）与 SYN-2（D-326）在组合标签 ((CN//2)+P27)%4 上联合 test 仅 0.766。本卡改**位组合标签**：每块只贡献 1 bit，检验是否为标签结构导致集成困难。

## 1. 协议

- 主体/块训练/标准化/联合读出与 SYN-1e 完全相同（CN 600、P24 500、P27 600；联合读出 96 维、8192 样本 1600 步）；
- **标签**：`2*(CN 类 % 2) + (P27 类 % 2)`（cn 最低位与 p27 最低位拼成 4 类；单块只知 1 bit）；
- 评估：联合 test 256；块级休眠、单位级休眠（各 12）、单块头对照、capsule。

## 2. 判据（读前冻结）

| 门 | 判据 |
|---|---|
| G0 联合任务 acc | ≥0.85 |
| G1 CN 块休眠 drop | ≥0.30 |
| G2 P27 块休眠 drop | ≥0.30 |
| G3 单块头 acc | ≤0.60（单 bit 最多 ~0.5） |
| G4 单位级休眠（各 12）drop | ≥0.10 |
| G5 capsule | ≤1e-6 |
| G6 预算/纪律 | ≤300 CPU、`criteria_sha256` 读前校验、零 check/final |

**出口**：全过⇒`syn3_verified`；G0 不过⇒`syn3_low_joint`；G1/G2 不过⇒`syn3_not_synergistic`；G3 不过⇒`syn3_single_block_sufficient`；G4 不过⇒`syn3_no_unit_causality`；其余照旧。**不调阈、不改门。**

## 3. 披露

- 人工组合标签（两位拼接）；只训读出（块冻结）；单 seed；不签 24-S/29/final/C/D/J；零 check/final。
