# 22.5.C.1 设计卡：把记录通道训成可解码、查询可区分的值（检索表示基础）

> 卡号：`22.5.C.1`｜日期：2026-11-16｜状态：**已设计、已定价、smoke 全过，训练未授权**。
> 契约 [`taiji_u2_c1_retrieval_20261116.json`](../manifests/taiji_u2_c1_retrieval_20261116.json)（`criteria_sha256=cdff0e19…`）；[smoke 回执](../../reports/taiji_u2_c1_retrieval_smoke_20261116.json)、[定价回执](../../reports/taiji_u2_c1_retrieval_price_20261116.json)。
> 前置：[B 线收束卡](CLOSEOUT-TAIJI_U2_B_LINE_20261116.md)（D-20261116-11）、[C 线立项](DECISION-LEDGER_20261109.md)（D-20261116-12）、[接地探针](../../reports/taiji_u2_c1_feasibility_20261116.json)。

## 1 这张卡问什么

B 线两次修订证明：更新路径修补不能使联合父门可达，卡点在**记录表示/读出**——值读出通路起点未训练、字节 CE 从不迫使内容使用。C 线第一问只有一个：

**现有 record 通道能不能被训练到「值读出可解码记录所述标签」且「路由可区分匹配实体与无关实体」？**

这是检索的表示前提。C.1 只建立表示（内容可解码＋查询可区分），**不接 emission、不做纠正协议**；C.2 才把训好的读出接进输出做联合门。单变量、单问题。

## 2 接地测量（零训练探针，1.98 CPU，0 优化器步，start bundle）

| 量 | 读数 | 含义 |
|---|---:|---|
| route match−distractor 均值 | **−0.0009** | 路由查询盲 |
| match>distractor 对数 | **8/24**（机会 12） | 路由不区分实体匹配 |
| 答案词注意力 mass | 0.157（熵 3.66） | 位置注意力近均匀，不指向答案 |
| 值读出 read 范数 | **0.0** | record_output 起点为零，读出携带 nothing |
| 值向量标签间余弦 | 0.834（类内 0.994） | 值通路标签弱可分，须训练 |

**设计后果**：①relevance 门不能复用查询盲的 route（C.2 须训练路由选择性，本卡先训）；②值通路必须内容训练（本卡核心）；③位置注意力须被内容目标牵引。

## 3 结构（最小干预，不新增主体结构）

- **主体**：v4（metadata+request），起点＝A.5 反馈前 bundle 逐字节。**不新增主体身份/张量**——record 通道 18 张量是主体既有参数，本卡训练它们。
- **解码头（runner 级脚手架，非主体参数）**：`decode_head = MLP(d→2d→4)`，消费值读出 `read = record_output(context)`，输出 4 类标签 logits（左/中/右/未知）。零初始化末层⇒出生解码在机会水平。它是测量/训练脚手架（同 A.1b 颜色头先例），不进 `output()`，故主体应答路径逐位不变。
- **可训集**：record 通道 18 张量/762 ＋ decode_head（脚手架）。**emission 8 张量/3,919 与其余基张量全程冻结**（纯度比对）。
- **两个目标**：
  1. **内容解码**：`CE(decode_head(read), 记录所述标签)`，记录在场回合。迫使 record_value/record_output 把标签编进 read。
  2. **查询选择性**：margin 损失 `max(0, margin − (route_match − route_distractor))`，配对回合（同查询，匹配实体记录 vs 不同实体记录）。迫使 record_query/record_key/record_goal_query 区分实体匹配。

## 4 数据（冻结构造，不读封存）

- 实体来自现有 train64/fit32 的 request（去后缀）。记录用 4 模板构造，所述标签 ∈ {左,中,右}。
- **解码集**：每实体×3 标签×模板 0 ⇒ train 192 / fit 96 条；查询＝该实体 request；目标＝所述标签。train 训、fit 评（held-out 实体）。
- **选择性对**：每 fit 实体，匹配记录（述其真值）vs distractor（不同实体述一标签）⇒ 32 对；route_match>route_distractor 计选择性。
- 数据 digest 冻结；不碰项目语料/Task1/封存。

## 5 门（读数前冻结）

| 判据 | 阈值 |
|---|---|
| 内容解码 fit | 96 条标签 argmax 精确 ≥ **0.95** |
| 查询选择性 fit | 32 对 route_match>route_distractor ≥ **0.95** |
| 出生解码 | 零初始化⇒零 logits⇒fit 解码在定类 argmax 机会（3 标签 0.333），证明训练前不携带；监视项 |
| 纯度 | emission＋非可训基张量逐位 == start |
| 主体应答不变 | decode_head 不进 output⇒A fit 32 摘要逐位 == start（结构监视） |
| 恢复 | 独立进程重载 record 通道＋decode_head，fit 解码/选择性逐位复现 |

阈值 0.95 是表示门（高于父门 0.80，因为这是 C.2 的前提，须留余量）；3 seed（20261110/11/12）同向。

## 6 出口

- `retrieval_representation_earned`：解码＋选择性两门 3 seed 全过 ⇒ 表示前提成立，后继 C.2（接 emission 做联合门）；
- `content_decode_failed`（解码 <0.95）／`selectivity_failed`（选择性 <0.95）／`both_failed`：表示未建立，按读数定位（值通路 vs 路由），C.2 不自动启动；
- `budget_stop`、`engineering_failure`、`pricing_over_budget`。

## 7 能签与不能签

**能签**：record 通道可被训到内容可解码＋查询可区分（表示能力），及纯度/恢复/出生中性。**不能签**：父/能力（联合门未测）/check②/22.6；解码准确率不得引为"主体会读记录作答"（decode_head 是脚手架，未接 emission）；C.2 须另行设计授权。

## 8 预算（实测单位）

定价实测组合更新单价（解码＋选择性，含 3 episode 重建）**0.234375 CPU**。`representation_3seed`：3 seed × **384 更新**（2 epoch×192 train）＝1152 更新＋评测，估计 **401.25**，帽 **461.437 CPU／922.874 wall**（设计帽 600/1200 内，15% 余量）。wiring 另账：smoke **30.16**（3 seed×出生解码/选择性/梯度路/应答不变/纯度，0 优化器步）、定价 **3.48**（8 步于丢弃模型）。出生读数：解码 0.333（定类机会）、选择性 0.719（路由出生已有部分实体信号，须训到 ≥0.95）。

## 9 唯一的请求（是非题）

**是否授权按 `representation_3seed` 实价执行 `22.5.C.1`（帽 461.437 CPU／922.874 wall，1152 更新，3 seed）？** 当前 0 正式优化器调用、0 训练步、0 能力声称、0 A 父。