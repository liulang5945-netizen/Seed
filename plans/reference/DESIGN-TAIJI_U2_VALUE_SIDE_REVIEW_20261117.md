# 22.5.V 值侧表示复核与设计卡（零训练，P3.9 后继范围）

> 卡号：`22.5.V`｜日期：2026-11-17｜状态：**复核完成＝`mixed`（见[结果卡](RESULT-TAIJI_U2_VALUE_SIDE_REVIEW_20261117.md)）；`22.5.V.1` 设计已收敛，待冻结契约／精确定价／训练授权**｜性质：值侧表示复核＋下一卡设计（不是训练卡）。
> 上游：[U.1r2 结果](RESULT-TAIJI_U2_U1R2_SCALE_SUPERVISION_20261117.md)（`focus_not_met`）、[事后定位](../../reports/taiji_u2_u1r2_posthoc_scale_value_20261117.json)（`scale_and_value_both_short`）、[读路径收束卡](CLOSEOUT-TAIJI_U2_UNIT_CORE_B_READ_PATH_20261117.md)（D-20261117-04）、[编号对账](DECISION-LEDGER_20261109.md)（D-20261117-05）。
> 结果：[RESULT-TAIJI_U2_VALUE_SIDE_REVIEW_20261117.md](RESULT-TAIJI_U2_VALUE_SIDE_REVIEW_20261117.md)（正式 `mixed`；描述性模式＝答案列可读、平均读出不可读）；报告 [reports/taiji_u2_value_side_review_20261117.json](../../reports/taiji_u2_value_side_review_20261117.json)。
> 登记：包22计划 §P3.9 后继范围 `22.5.V`（2026-11-17）；**不是 P3.9 修订**，不重开已收束的 P3.9。

## 1 单问题

U.1r2 的事后定位已经把墙从"尺度"移到"方向＋值侧"：把 `log_tau` 加到设计所需，fit 增益 0/0/0；`W_q` 置零对照显示方向只贡献 +0.1875/+0.0625/0；oracle 强制聚焦下，与 `position_head` 同类、拟合到收敛的线性头仍 **0/3 达 fit 门**，且相对 r1 权重在 seed01/seed03 退化。现在要回答的是：

> **信息在值链的哪一段丢失？** 是（a）每个列的 value 向量根本不带颜色；（b）带颜色但不带"查询实体 → 位置"的映射；（c）列 value 带信息，但注意力平均／背景主导把它稀释掉；（d）信息在，缺的是读出变换或读出信用。

这个问题不回答，任何"再训一版"都会重复 U.1r2 的混淆：不知道是表示没长出来、还是读出没学出来。

## 2 范围、数据与边界

- **模型**：U.1r2 的三枚训练后胶囊（`data/experiments/taiji_u2/u1r2_scale_supervision_20261116/seed*_final.bin`），权重冻结；编码合同 `unit-input-column-color-v1` 冻结；不新增参数、不改产品代码。
- **数据**：只用 train 64／fit 32（本卡自己的开发面）；**check／final／confirm 读取 0**，封存分区 0。
- **训练**：**0 优化器步、0 权重写入**；所有头都是诊断用的线性/二次读出，拟合到收敛即停，不进模型、不存档。
- **预算**：本复核自设帽 **120 CPU／240 wall**；到达即停并记 `budget_stop`。下一卡定价另立。

## 3 测量（全部零训练，先声明再读数）

### 3.1 列级 value 是否带颜色

用一次前向捕获模块自己的 `projected [T,H]`（patch `torch.einsum("hn,tn->th")`，不在模块外重算）。对每个 (scene, column) 训练一个线性头 `H→7`（6 色调 + 背景），train 64 拟合、fit 32 评分；报告总体与 painted-only（非背景列）准确率。

- **输入侧参照**：同一标签、同一头类，特征换成冻结编码 `rgb[:, :6]`（6 色质量），应接近 1.0（尺子有量程）。
- **判别**：painted-only ≥0.80 ⇒ 列 value 保留颜色；<0.80 ⇒ 颜色在 value 向量里已丢。

### 3.2 答案列的 value 是否带位置

对每个 scene 取答案列 `argmax` 查询色质量（无质量时 present=0），特征 `[projected[answer_col], present]`（9 维），线性头 `→4` 位置类；train 拟合／fit 评分。

- 对照：训练后自身注意力的 `attended`（8 维）；oracle soft 加权 `oracle_soft @ projected`（8 维）；答案跨度均值 `mean(projected[answer_span])`（8 维）；`[attended, projected[answer_col], present]`（17 维）；二次特征 `[attended, attended²]`（16 维）。
- **判别**：答案列头或 concat 头达 fit≥0.80 且 train≥0.90，而 `attended` 头不达 ⇒ **信息在列里，读出/平均是缺口**；只有二次头达 ⇒ 信息在但需要非线性；所有模型侧头都不达而输入侧达 ⇒ 值向量本身不承载位置。

### 3.3 背景稀释与注意力质量

- 背景列（总质量 < `_MASS_FLOOR`）的注意力质量均值、painted 列的注意力质量均值；
- 背景列 `projected` 范数／painted 列范数比值；
- 答案跨度上的注意力质量（沿用 `focus_stats`）与"答案列 value 头"的对照。
- **判别**：背景注意力质量 >0.5 且答案列头高而 attended 头低 ⇒ 平均被背景稀释；若背景质量不高但 attended 头仍低 ⇒ 问题在列 value 或读出。

### 3.4 方向与注意力选择

- 训练后注意力在答案跨度上的质量、有效列数、top-1 列是否落在答案跨度；
- `W_q` 置零后的列 value 头读数（同权重配对）——若置零不改变列 value 头，说明列 value 与查询方向无关（符合事实），但 attended 头会退化。
- 这一节只描述"选没选对"，不把选择读数当值侧证据。

### 3.5 仪器控制（fail-closed）

- 几何正对照 `[x, x², present, 1]` fit≥0.80（D-24 实测 1.0）；
- 置换标签头 fit≤0.50；
- 输入侧列颜色头 ≥0.80；
- 三 seed `params_unchanged`、run 的 fit/train/focus 复算逐值相同、`attended == attention @ projected` 逐位（max_abs 0）；
- 任何一项不过 ⇒ `instrument_invalid`，不得解释后续读数。

## 4 判据与出口（读数前冻结，互斥按序）

1. `budget_stop`——超 120 CPU／240 wall。
2. `instrument_invalid`——任一仪器控制不过。
3. `value_absent_in_columns`——列颜色头 painted-only <0.80，且答案列头不达双门；而输入侧列颜色头 ≥0.80 ⇒ 颜色在输入里有、在单位 value 里没有。
4. `value_present_position_absent`——列颜色头 ≥0.80，但答案列头、concat 头、oracle 头、masked 头全部不达双门 ⇒ value 有颜色、缺位置/答案映射。
5. `value_present_needs_extra_features`——列颜色头 ≥0.80，且只有二次头或 concat 头达双门，`attended` 头不达 ⇒ 信息在，缺的是非线性/逐列变换。
6. `value_present_readout_gap`——列颜色头 ≥0.80，且答案列头或 masked 头达双门，`attended` 头不达 ⇒ 信息在列里，平均/读出是缺口。
7. `value_already_readable`——`attended` 头本身达双门（与 U.1r2 事后读数矛盾；若出现，说明本复核的某个头/数据面与事后探针不同，必须先解释差异，不得直接采信）。

**聚合**：3 seed 逐 seed 分类；若 ≥2/3 同判，写该判；否则写 `mixed` 并逐 seed 列明。**任一出口都不签能力/父/check②/22.6。**

## 5 条件设计：下一卡 `22.5.V.1`（只在复核支持时立项）

> 本节是**设计预案**，不是训练授权；参数、数据、门与价格须在复核结果出来后按实际分类冻结，届时另写 manifest 并请 owner 授权。

**分支 A（`value_present_readout_gap`）——查询索引的列 value 读出头**：在现有 `projected` 上，用查询选列（argmax 或 soft top-k）直接读该列的 value，替代/补充注意力平均；只训读出侧（`W_q` + 新读出投影），值侧冻结；门：列 value 头 ≥0.80/0.90、attended 门回退不得超过 0.05、读门与 U.1r2 同口径、旧能力保持、独立恢复、成本 ≤ 实测价。参数尽量少（≤2k），零初始化注入。

**分支 B（`value_present_needs_extra_features`）——逐列 value + 查询选择的非线性/掩码读出**：读所有 16 列的 value 与查询的匹配分数，输出"答案列的 value + 位置轴"；可加一个小投影（≤8k 参数），但必须声明它是否替代 `position_head`，并给出中性出生与旧能力保持。门同上。

**分支 C（`value_absent_in_columns`）——值编码卡或收束**：先做零训练设计探针，回答"颜色质量能否在 `p`/state 投影中保留"；若能，设计一个颜色保持的 value 投影（参数/初始化/门/价另立）；若不能，按"值编码在当前合同下不可达"收束，不再用读出侧训练掩盖。**分支 C 不得直接进入训练。**

**分支 D（`value_present_position_absent`）——位置轴读出**：列 value 有颜色但位置不可读，优先检查 `x` 轴是否在 `projected` 里保留；设计一个位置轴显式读出（不新增值侧参数），门同上。

**统一约束**：新卡必须用同一候选前后版本、同出生配对、3 seed、未见 fit、旧能力保持、独立恢复、实价先行；不得复用 `unit-core-b-v1` 的修订额度（已 2/2 用满），必须作为**新范围**登记；在 owner 批准设计与实价前 **0 训练**。

## 6 能签与不能签

**能签**（复核完成后）：值链缺口的具体段落（列 value／位置／平均／读出之一）＋仪器控制＋零训练纪律＋下一卡设计是否成立。**不能签**：单位核心 B 架构整体不可用；任何能力/父/check②/22.6；"复核分类"不是训练结果，不能当能力证据。

## 7 当前状态与唯一后继

**当前**：本卡设计已登记；复核执行中（0 训练、0 封存读取）。**唯一后继**：跑零训练复核 → 按 §4 分类 → 若分类支持，写 `22.5.V.1` 的冻结契约与实价，请 owner 授权；若分类是 `value_absent_in_columns`，先做零训练值编码设计探针，不直接训练。复核结果出来前 **0 训练新增**、0 能力、0 A 父；check② 不读、22.6 不解锁。
