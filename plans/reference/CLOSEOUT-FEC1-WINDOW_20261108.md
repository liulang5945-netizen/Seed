# FEC-1 窗口全局收束卡（2026-11-08）

> 状态 [窗口收束·强制停点]。W1 修复-复测循环第 2 轮（最后轮）判 `query_key_misaligned`（2/2 一致），按 [W1 分支计划](PLAN-FEC1_ADJUDICATION_WINDOW_20261107.md) §3.4 第 4 条强制收束。本卡六件：产物清单、能力/判据账、成本总账、负结果表、架构需求输入、下一轮唯一核心问题。owner 裁定前不开新包、不训练、21.2／21.4／U1 冻结维持。

## 1. 产物清单（全部入库）

| 产物 | 身份 |
|---|---|
| v1 预注册＋执行（旧图 4 checkpoint） | manifest `1e0dde22…`／[报告](../../reports/taiji_f_expressive_capacity_20261108.json)（runner sha 双列一致）／322.69 CPU |
| W1(a) round-1 实现＋复测 | [报告](../../reports/taiji_fec1_w1a_retest_20261108.json)＋[card](../manifests/taiji_fec1_w1a_retest_feature_card_20261108.json)／150.92 CPU |
| v2 预注册（仅修对照统计量） | [manifest](../manifests/taiji_f_expressive_capacity_prereg_v2_20261108.json) `aa3a8aef…`（v1 原块保留审计） |
| W1 round-2 实现＋复测 | [报告](../../reports/taiji_fec1_w1r2_retest_20261108.json)＋[card](../manifests/taiji_fec1_w1r2_retest_feature_card_20261108.json) `4855e43e…`／126.31 CPU |
| 图层加法式改动（default-off） | `living_core.py`（config 三旗＋content_feature＋read/write 通路）、`growth_credit.py`（GatedGrowthCore.step 同步）、`rule_learning_20_1_net.py`（enable_content_keys）；disabled parity 逐位过、R021 oracle 12/12 |

**能力账**：唯一已对外能力仍为 C12；本窗口 0 训练、0 confirm、不产任何能力声称。**判据账**：v1 判据 PASS 分支算术不可达（D16-42）已由 v2 修复（仅对照容量封顶，门值未动）；v2 下 PASS 恢复原则可达（需 primary ≥ 0.9273）。

## 2. 成本总账

窗口累计 **599.92 CPU**（322.69＋150.92＋126.31），wall 同量级；工程重跑 5 次（钩子 reshape、空缓冲、dtype、step 覆盖漏改、goal 重复传参）均在 preflight/早期崩溃、progress jsonl 留痕、费用可忽略。0 训练／0 优化器步／0 新增已训练参数（W1 内容通路 +4,096 可训练参数**未训练**，属于图身份变更并已按 card 登记）。

## 3. 负结果表（按范围结案，不升格"不可能"）

| # | 判负 | 范围 |
|---|---|---|
| N-F1 | 旧图 F 写键函数类结构性塌缩（pairwise 0.0、秩 2，random_init 同塌缩） | 当前图＋checkpoint 谱系 |
| N-F2 | W1(a) 原始 byte-count 内容特征判别力不足（pairwise 0.104） | 该特征＋该夹具 |
| N-F3 | W1 round-2：写键空间已可分（pairwise 0.974、秩 7 全过门）但 **query_alignment 0.110 < 0.5** 且闭式读出 event-id 0.0 | round-2 图＋init 状态＋本夹具 |

## 4. 架构需求输入（本窗口最有价值的产出）

三轮读数把"F 要能按问句/按反馈事件可分可寻址"分解为**三个已定位、可施工的部件**：

1. **内容驱动的写键**：已由 round-2 证实可修——居中内容特征＋加性内容通路把 pairwise 从 0.0 推到 0.974、秩从 2 到 7，全部结构门通过。设计已实现并入库（default-off）。
2. **读键对齐**：内容 boost 的 tying 不足以把 query_alignment 拉过 0.5（实测 0.110）——两个读/写 MLP 的固有输出方向与内容方向近反平行。修复选项：读写 MLP **完全共享权重**（全 tied），或依赖训练把对齐学出来（当前 init 不可达）。这是 `query_key_misaligned` verdict 指名的最小充分修复的**完全版**。
3. **值通路判别力**：22 个写入值只张成 5 维（值面秩门 ≥3 过，但检索需要逐对不同）——值段需要与键段同样的内容条件化，否则闭式读出无东西可取（event-id 0.0 的直接原因）。

三者齐备后，函数类"按构造成立"（键可分＋对齐＋值可分 ⇒ 闭式读出精确恢复，v2 判据的 PASS 分支可达），届时才有资格为有界训练窗定价。

## 5. 下一轮唯一核心问题（owner 裁定）

**是否按上述三部件修复清单立"F 写入通路重设计 v2"新窗口**（设计→定价→实现→FEC-1 v2 复测→（条件）有界训练窗），还是**接受本窗口负结果为最终裁决**、把 F 线降级为长期研究方向并回到其他线。两轮循环已用尽，本卡发布后不自动开跑任何修复。
