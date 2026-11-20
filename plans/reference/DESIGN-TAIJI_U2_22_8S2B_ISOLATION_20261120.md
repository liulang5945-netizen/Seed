# 22.8S2b 预注册：纠错参数隔离（新 pin，零 check）

> 文档状态：[预注册草案] — 2026-11-20（D-20261119-155）。上游 [S2 结果卡](RESULT-TAIJI_U2_22_8_S2_20261120.md)、[S2 归因卡](ATTR-TAIJI_U2_22_8_S2_FAILURES_20261120.md)、[D-76/77 参数隔离有界证据](DECISION-LEDGER_20261109.md)。**未冻结、未运行、不读 check/final。**

## 1. 唯一问题

在 22.8 S2 的同一主体/会话结构下，把**纠正子集**与**旧子集**分到**不同任务头**并冻结共享单位，能否同时满足：

- 纠错增益 ≥0.05（3 seed 同向），且
- 旧子集保持损伤 ≤0.02？

本卡只回答 J3；不重测 J1/J5，不改 S2 判词。

## 2. 与 S2 的差集（唯一变量）

| 项 | S2 | S2b |
|---|---|---|
| 纠错写入 | 共享 30 单位＋同一 head3（旧标签与翻转标签共用） | 共享单位**冻结**；纠正子集用 `head_correct`，旧子集用 `head_old`（两新头，互不共享） |
| 旧子集读出 | 同一 head3 | `head_old`（未受纠正训练影响） |
| 其余 | 3 seed×3 session×32 轮、五类片段、预算/门 | **完全一致** |

## 3. 协议

- 主体：30 单位 RouteB（S2 同规格，seed 20261120–22）；
- 先按 S2 协议训练 appearance（8 轮/session×3 session×8 步）得到基座；
- 会话 3 起：冻结共享单位（`freeze_module(units)`），新增 `head_correct`、`head_old`（`nn.Linear(query_slots,24)`，随机初始化），只训练两头：
  - 纠正：corrected 子集（`y_app % 4 == 0`）标签翻转 `(y+12) % 24`，`head_correct` 训练 200 步；
  - 旧标签：uncorrected 子集，`head_old` 训练 200 步（保底学习，确保旧头可用）；
- 评估：`head_correct` 在 corrected 子集翻转标签上的准确率（前/后），`head_old` 在 uncorrected 子集原标签上的准确率（前/后）。

## 4. 门（读前冻结）

| 编号 | 条件 | 出口 |
|---|---|---|
| K1 纠错增益 | corrected 子集准确率提升 **≥0.05**，3 seed 同向 | `correction_isolation_verified` |
| K2 旧保持 | uncorrected 子集损伤 **≤0.02**（`head_old` 前后） | 失败 ⇒ `correction_isolation_partial/absent` |
| K3 工程 | 冻结单位后 `units.*` 无梯度、仅两头有梯度；恢复 exact | 失败 ⇒ `engineering_failure` |
| 总判 | K1+K2+K3 全过 ⇒ verified；部分过 ⇒ partial；全负 ⇒ absent | —— |

## 5. 预算与边界

- 3 seed、**价 ≤300 CPU**、零 check/final、阈值不调；新 pin：`run_s2_22_8b_isolation_20261120.py`；
- 不重测 J1/J5；不改 S2 判词；不签 final/C/D/J/模型成型；
- 若 K 门通过：J3 从 S2 的失败项改为「参数隔离下有界 verified」，22.8 分账更新为 J2/J3/J4/J6 有界、J1/J5 range-limited negative；若不过：J3 维持失败并停止同范围。
