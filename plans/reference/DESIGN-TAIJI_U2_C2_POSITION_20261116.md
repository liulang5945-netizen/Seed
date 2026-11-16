# 22.5.C.2 设计卡：位置定向检索（C 线最后修订，预承诺收束）

> 卡号：`22.5.C.2`｜日期：2026-11-16｜状态：**已设计、已定价、smoke 全过，训练未授权**。
> 契约 [`taiji_u2_c2_position_20261116.json`](../manifests/taiji_u2_c2_position_20261116.json)（`criteria_sha256=6841fd1a…`）；[smoke 回执](../../reports/taiji_u2_c2_position_smoke_20261116.json)、[定价回执](../../reports/taiji_u2_c2_position_price_20261116.json)。
> 前置：[C.1 结果卡](RESULT-TAIJI_U2_C1_RETRIEVAL_20261116.md)（D-20261116-14，`both_failed`，缺口＝位置注意力不聚焦）、owner 裁决（D-20261116-15）。
> **预承诺（owner 裁决时确立）**：C.2 是 C 线第 2 次、**最后一次**架构修订；若表示门仍未过，按 MAINLINE §5 **全局收束**字节世界联合父门 effort，不再局部追加。

## 1 这张卡问什么

C.1 定位：值读出 read 记录不变（std 0.026/范数 5.73），因为位置注意力近均匀（mass 0.157/熵 3.66）把整条记录平均成常量，标签信号被抹掉。C.2 单问题：

**给位置注意力加监督（把 mass 训到标签字节）之后，read 能不能携带标签、表示门能不能过？**

相对 C.1 的**唯一变量**：新增位置聚焦损失。其余（可训集、数据、解码/选择性目标、门、emission 冻结）逐字继承 C.1。

## 2 结构（最小干预）

- 主体 v4，起点＝A.5 反馈前 bundle；可训＝record 通道 18/762＋decode_head 脚手架；emission 冻结＋纯度。
- **新增目标（位置聚焦）**：`position_focus_loss = −log( mass(label_span) / mass(valid) + eps )`，其中 attention weights = exp(log_alpha+log_beta) 由 record_position_query/key 控制，label_span 从记录 raw 导出（标签字节位置，可观测）。迫使位置注意力集中到标签字节，使 read 携带标签而非平均常量。
- 总损失 = 解码 CE（C.1）＋ 选择性 margin（C.1）＋ **位置聚焦**（C.2 新增）。三项权重冻结（1:1:1，先验等权，不调）。

## 3 门（读数前冻结，继承 C.1＋位置诊断）

| 判据 | 阈值 |
|---|---|
| 内容解码 fit | 96 条 ≥ **0.95**（3 seed） |
| 查询选择性 fit | 32 对 ≥ **0.95**（3 seed） |
| 位置注意力 mass（诊断） | fit 记录答案 span mass 均值（报告，不设硬门——它是机制读数，表示门由解码/选择性承担） |
| read 记录可分性（诊断） | fit read 跨记录 std/范数（报告，对照 C.1 的 0.005） |
| 纯度／应答不变／恢复 | 同 C.1（emission 冻结、decode_head 不进 output、独立进程逐位） |

## 4 出口（含预承诺）

- `retrieval_representation_earned`：解码＋选择性两门 3 seed 全过 ⇒ 表示前提成立，后继 C.3（接 emission 做联合门，另行设计授权）；
- `content_decode_failed`／`selectivity_failed`／`both_failed`：表示仍未建立 ⇒ **触发预承诺，全局收束**（出收束卡，合成 A/B/C 三线负结果与定位，判字节世界联合父门在当前主体的架构性限制）；
- `budget_stop`、`engineering_failure`、`pricing_over_budget`。

## 5 能签与不能签

**能签**：位置定向是否使表示建立（及 mass/read 可分性机制读数）。**不能签**：父/能力（联合门未测）/check②/22.6；解码准确率不得引为"主体会读记录作答"（脚手架未接 emission）。负结果触发全局收束，不自动开 C.3 或新修订。

## 6 预算（实测单位）

定价实测组合更新单价（解码＋选择性＋位置聚焦）**0.179688 CPU**。`position_3seed`：3 seed × **384 更新**＝1152＋评测，估计 **325.126**，帽 **373.895 CPU／747.79 wall**（设计帽 600/1200 内，15% 余量）。wiring 另账：smoke **31.19**（3 seed×出生/梯度路[含 record_position_key 梯度]/应答不变/纯度，0 优化器步）、定价 **2.88**（8 步于丢弃模型）。出生诊断：答案 mass 0.199、read std/范数 0.0（record_output 起点零）。

## 7 唯一的请求（是非题）

**是否授权按 `position_3seed` 实价执行 `22.5.C.2`（帽 373.895 CPU／747.79 wall，1152 更新，3 seed）？** 当前 0 正式优化器调用、0 训练步、0 能力声称、0 A 父。**负结果触发预承诺全局收束。**