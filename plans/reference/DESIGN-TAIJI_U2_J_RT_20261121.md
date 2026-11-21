# 联合卡预注册：路由＋保护下的三任务共存与胶囊继承（J-RT；≤800 CPU；零 check）

> [预注册] — 2026-11-21（D-20261121-252）。上游 D-249（保护 3 seed）、D-251（继承）。

## 1. 范围
72 单元单主体，24 单元/任务固定子群；顺序 P-A→P-B→P-D，参数级保护（行/列/read_gate 行掩码、wd=0）；训练后 capsule_v2（含 task_routes）→ 新图恢复→ 三任务复评。3 seed、≤800 CPU、零 check/final。

## 2. 门（冻结）
| 门 | 判据 |
|---|---|
| G1 共存 | 每 seed：CN ≥0.95、P-B ≥0.85、P-D ≥0.80 |
| G2 继承 | 恢复前后三任务准确率差 ≤1e-6；routes 完整 |
| G3 预算 | 3 seed、≤800 CPU、零 check/final |

出口：全过⇒`j_rt_verified`；G1 不过⇒`j_rt_partial`；G2 不过⇒`capsule_restore_partial`；超帽⇒`budget_stop`；工程失败⇒`engineering_failure`。

## 3. 披露
固定分区为实验层机制；heads 单独序列化；只验共存＋继承，不签协同/生长收益/check/final/C/D/J。
