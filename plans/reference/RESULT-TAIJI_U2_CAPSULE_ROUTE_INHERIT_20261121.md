# 结果卡：任务路由＋参数保护的 capsule 继承（`capsule_route_inheritance_verified`；204.08 CPU；零 check）

> [结果卡] — 2026-11-21（D-20261121-251）。上游 [参数保护 3 seed](RESULT-TAIJI_U2_S1RT_PROTECTION_20261121.md)、[task_routes 工程卡](ENG-TAIJI_U2_CAPSULE_TASK_ROUTES_20261121.md)。**零 check/final；常设授权下执行，待 owner 回溯。**

## 1. 协议

- 1 seed：同一 72 单元主体，顺序 P-A→P-B→P-D，参数级保护（行/列/read_gate 行掩码、保护阶段 wd=0）；
- 训练后 `build_capsule_v2`（含 `task_routes`）→ 保存/载入 → 新图 `apply_capsule_v2` → 恢复三个读出头 → 三任务复评；
- ≤200 CPU、零 check/final。

## 2. 读数（[回执](../../reports/taiji_u2_capsule_route_inherit_20261121.json)）

| 任务 | 训练后 | 恢复后 | 差 |
|---|---|---|---|
| CN | 1.0000 | 1.0000 | 0.0 |
| P-B | 1.0000 | 1.0000 | 0.0 |
| P-D | 0.8516 | 0.8516 | 0.0 |

- `max_abs_diff = 0.0`；`routes` 三任务各 24 单元完整恢复；204.08 CPU。

## 3. 判读

- **任务路由＋参数保护可随 capsule_v2 继承并精确恢复**（同一主体、三个读出头、恢复后读数逐位一致）；
- 限制：固定分区为实验层机制；1 seed；读出头单独序列化（capsule 不含 head 权重）；不构成协同/生长收益；不签 check/final/C/D/J。

## 4. 后继

- 阶段 2 学习式路由；或以本机制重开 S1/23-S 联合卡（3 seed、≤800 CPU）。
