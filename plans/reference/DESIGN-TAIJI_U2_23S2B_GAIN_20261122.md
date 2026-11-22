# 23-S.2b 预注册：等算力 GRU/TF 基线增益（3 seed；≤1,500 CPU；零 check）

> [预注册] — 2026-11-22（D-20261122-278）。上游 23-S.2a（D-277：shared/frozen 增益 0.398，但非合同等算力基线）。

## 1. 设计
- R 路由臂：23-S.1b 已冻结读数（3 seed；Q=min(CN unseen, P-B, P-D) = 0.8359/0.8828/0.8594）；
- B 基线臂：**单模型**（GRU 或 TF），同一三任务顺序课程（CN→P-B→P-D）、同 batch 48、同步数 1000/900/1300、同数据生成器与同 fit 集；三个任务专用读出头；无任务路由/无保护；
- Q_B := min(CN unseen, P-B, P-D)；增益 := Q_R − max(Q_GRU, Q_TF)（逐 seed）；主判据：3 seed 均值增益 ≥0.05 且最小增益 ≥0.0。

## 2. 门
| 门 | 判据 |
|---|---|
| G1 增益 | mean(3 seed gain) ≥0.05 且 min gain ≥0.0 |
| G2 预算 | 3 seeds × 2 baselines、≤1,500 CPU、零 check/final |
| G3 纪律 | criteria_sha256 读前冻结；不调阈 |

出口：全过⇒`23s2b_gain_verified`；G1 不过⇒`23s2b_gain_absent`；超帽⇒`budget_stop`；工程失败⇒`engineering_failure`。

## 3. 披露
- 等算力按**同步数/同 batch/同数据**对齐，非精确 FLOPs 对齐；模型参数与形态不同；
- 基线为单模型顺序课程（不重置），若基线因顺序遗忘失败属正常现象，需如实说明；
- 合成面；不签 23-S 整包（还需 Q 定义认可与 §3A.2 映射）；零 check/final/C/D/J。
