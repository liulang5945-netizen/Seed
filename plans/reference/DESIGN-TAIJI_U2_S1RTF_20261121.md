# 预注册：J-RT 机制回灌正式联合判据（S1-RTF；3 seed；≤800 CPU；零 check）

> [预注册] — 2026-11-21（D-20261121-256）。上游 J-RT（D-253）、路由继承（D-251）。

## 1. 协议
72 单元单主体、24/24/24 固定子群、顺序 P-A→P-B→P-D、参数级保护（行/列/read_gate 行、wd=0）；训练后 capsule_v2（routes）→新图恢复。3 seed、≤800 CPU、零 check/final。

## 2. 门（冻结）
| 门 | 判据 |
|---|---|
| G1 共存 | CN ≥0.95、P-B ≥0.85、P-D ≥0.80（每 seed） |
| G2 纠错 | P-B conflict ≥0.95（每 seed） |
| G3 消融 | P-D 无探针下降 ≥0.20（每 seed） |
| G4 旧技能 | CN/P-B 相对各自峰值损伤 ≤0.02 |
| G5 胶囊 | 内部 readout 前向差 ≤1e-6；恢复后三任务差 ≤1e-6 |
| G6 预算 | 3 seed、≤800 CPU、零 check/final |

出口：全过⇒`s1rtf_verified`；任一 G1–G5 不过⇒`s1rtf_partial`；超帽⇒`budget_stop`；工程失败⇒`engineering_failure`。披露：固定分区实验层机制、heads 单独序列化、不签协同/生长收益/check/final/C/D/J。
