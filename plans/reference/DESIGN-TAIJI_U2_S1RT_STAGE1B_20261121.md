# S1-RT 阶段 1b 预注册：修正门实现的任务分区复检（≤800 CPU；零 check）

> 文档状态：[预注册] — 2026-11-21（D-20261121-246）。上游 [阶段 1 结果](RESULT-TAIJI_U2_S1RT_STAGE1_20261121.md)。**零 check/final；常设授权下执行，待回溯。**

## 1. 修正点（针对阶段 1 的门实现错误）

1. **统一评估集**：CN 普通双向与对抗集分列；P-B/P-D 各用自己的 fit；同一指标同一分母，不再跨集比较；
2. **3×3 lesion 矩阵**：每个任务 mask 休眠后，记录本任务与其余任务的准确率变化（特异性判据）；
3. **容量**：总单元 72（12 群×6），每任务 24 单元（0–23/24–47/48–71），避免 16 单元对 P-B 容量不足；
4. **P-B 在线数据** 900 步；CN 600 步（step_mask）；P-D 500 步；均单主体顺序训练。

## 2. 臂

- `shared`：72 单元全激活（对照，预期复现覆盖）；
- `partitioned`：任务 gate mask 固定分区 24/24/24。

## 3. 门（读前冻结）

| 门 | 判据 |
|---|---|
| G0 控制 | shared：末次 P-B 或 CN 相对各自峰值下降 >0.05，**或** P-D <0.80（证明干扰真实） |
| G1 共存 | partitioned：CN 普通 ≥0.95、CN 对抗 ≥0.90、P-B ≥0.85、P-D ≥0.80 |
| G2 旧技能 | partitioned：CN/P-B 相对各自峰值损伤 ≤0.02 |
| G3 特异性 | partitioned：3×3 矩阵中，自任务 lesion 降 ≥0.20，其余任务降 ≤0.05 |
| G4 恢复 | partitioned：capsule_v2 前向差 ≤1e-6 |
| G5 预算 | ≤800 CPU、3 seed、零 check/final |

**出口**：全过⇒`s1rt_stage1b_verified`；G1–G3 任一不过⇒`s1rt_stage1b_partial`；G0 不过⇒`control_interference_absent`；超帽⇒`budget_stop`；工程失败⇒`engineering_failure`。

## 4. 披露

- 分区仍为实验层机制（不改内核）；只验「共存＋特异性」，不签协同/生长收益/check/final/C/D/J；
- 预注册后不再改阈值；结果为负同样作为归因证据。
