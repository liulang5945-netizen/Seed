# 22.5.V.2 选择机制复核与设计卡（零训练）

> 卡号：`22.5.V.2`｜日期：2026-11-17｜状态：**复核完成＝`neither_reaches_gate`**（指针/共享 W_q 在 3/3 seed 都达不到 fit 0.95；见[结果卡](RESULT-TAIJI_U2_V2_SELECTION_REVIEW_20261117.md)），**停在 owner 收束/内容查询裁决**｜性质：P3.9 后继新范围，**不是 V.1 修订**（V.1 已按 D-20261117-09 收束）。
> 上游：[V.1 收束卡](CLOSEOUT-TAIJI_U2_V1_HARD_SELECTION_20261117.md)（`budget_stop`＋部分负；完整 seed 选择 0.84／0.52）、[V 值侧复核](RESULT-TAIJI_U2_VALUE_SIDE_REVIEW_20261117.md)（答案列 3/3 可读、attended 0/3、top-1 选择 0/3）。
> 父：U.1r2 三枚训练后胶囊；值侧冻结；只读 train 64／fit 32；**0 优化器步、0 权重写入、封存 0**。
> 结果：[RESULT-TAIJI_U2_V2_SELECTION_REVIEW_20261117.md](RESULT-TAIJI_U2_V2_SELECTION_REVIEW_20261117.md)（`neither_reaches_gate`；报告 [reports/taiji_u2_v2_selection_review_20261117.json](../../reports/taiji_u2_v2_selection_review_20261117.json)）。

## 1 单问题

V.1 把缺口收窄到"选择"：值侧冻结且答案列 3/3 可读，但当前 `W_q`＋聚焦监督只把完整 seed 的 top-1-on-span 抬到 **0.84／0.52**（门 0.95）。本卡问：

> **在值侧冻结、零训练的条件下，是否存在一种选择机制能把未见 fit 场景的 top-1-on-span 推到 0.95？**

候选机制（全部只用 train 拟合、fit 评分，不训练模型）：

- **A 基线**：V.1 训练后的 `W_q` 注意力 top-1（0.84／0.52／0.64）。
- **B 指针头（answer-span supervised pointer）**：对每列打分 `s_t`，softmax 后取 top-1；两种参数类：
  - B1 线性：`s_t = w^T [projected[t], question]`（16 参数）；
  - B2 双线性：`s_t = question^T M projected[t]`（64 参数）；
  - 目标：答案跨度上的均匀分布（主）/答案列 one-hot（次）；凸/非凸 CE，LBFGS 到收敛；只在 train 64 上拟合，fit 32 评分。
- **C 答案跨度监督的共享 W_q（最小二乘）**：沿用 r2 可行性探针的线性系统 `A[t,(d,h)] = scale·projected[t,h]·question[d]`＋每场景一个自由 logit 偏移，目标＝答案跨度均匀分布；`lstsq` 求解，fit 评分 top-1-on-span。
- **D 上界**：oracle 选择（答案列 argmax 查询色质量）＝1.0；答案列位置头 3/3 达双门（V 复核）。

## 2 读数与判据（读数前冻结）

**主测量**：fit top-1-on-span（有答案跨度的 24 条 fit 场景；分母固定）。

**门**：
- 选择门：**≥0.95**（3 seed 全过才算过）。
- 仪器：几何正对照 ≥0.80；`capture` 的 `attended == attention @ projected` max_abs ≤1e-12；run 读数复算逐值；三 seed 参数不变；oracle 选择 1.0；deranged span 控制显著下降。
- 预算：自设帽 **120 CPU／240 wall**；到达即停。

**出口（互斥按序）**：
1. `budget_stop`；
2. `instrument_invalid`；
3. `pointer_reaches_gate`——B1 或 B2 在 3/3 seed 上 fit selection ≥0.95 ⇒ 指针机制可行，下一卡 `22.5.V.2.1` 训练指针头（零新值侧、参数 ≤ 实测）；
4. `wq_lsq_reaches_gate`——C 在 3/3 seed ≥0.95 ⇒ 共享 W_q 参数化可行，下一卡改成答案跨度监督（只训 W_q）；
5. `neither_reaches_gate`——B/C 都不到 0.95 ⇒ 当前列 key／查询表示不支持 0.95 选择；**关读路径**或另立查询表示复核，不用训练硬顶；
6. `mixed`——逐 seed 分歧，写明分歧与最接近的机制。

## 3 条件设计（只在出口 3/4 时立项）

**出口 3（pointer）→ `22.5.V.2.1`**：值侧冻结，只训指针头（B1 16 参数或 B2 64 参数，按实际达门者）；训练目标＝答案跨度 CE；门：fit selection ≥0.95、读门 fit≥0.80/train≥0.90、双控制<0.5、四干预、颜色诊断不降、独立恢复、实价先行。参数上限按 smoke 实测，不预先加宽。

**出口 4（W_q 监督）→ `22.5.V.2.2`**：值侧冻结，只训 `W_q`；监督＝答案跨度 CE（不是颜色质量）；门同上；不加 `log_tau`、不加新参数。

**出口 5（都不过）**：不训练；写"当前列 key／查询表示不支持 0.95 选择"的具名负结果，回到查询表示/编码合同复核或 N 主线下一轴。

## 4 能签与不能签

**能签**（复核后）：选择机制的零训练可行性与上限；仪器与成本；下一卡是否立项。**不能签**：任何能力/父/check②/22.6；零训练指针头不是模型能力证据；`neither` 不得写成"选择原理不可能"。

## 5 当前状态与唯一后继

**当前**：设计/预注册执行中；**0 训练**。**唯一后继**：跑零训练复核 → 按 §2 出口分类 → 若出口 3/4，写下一卡冻结契约＋定价，请 owner 授权；若出口 5，关读路径或另立查询表示复核。复核前 0 训练新增、0 能力、0 A 父；check② 不读、22.6 不解锁。
