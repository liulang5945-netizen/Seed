# 共同内容核心：重构-5b 复制轨候选合同

> 审批状态勘误：用户已于2026-10-30批准本合同，原始记录见§10；下文页首、§1/§9等“待批准/本件不授权”措辞仅保留起草时点，不覆盖该批准。当前顺序为[总计划步骤08.1–08.5](../MAINLINE.md#3-唯一执行队列)；本注只统一状态，不改候选、数据、门值、预算或既有授权范围。

日期：2026-10-30，Asia/Shanghai。状态：**合同草案，待用户批准；未授权实现或训练。** 依据 [R4 §9.3](SPEC-NATIVE-CONTENT-CORE_R4_PRICING_AND_GATES_20261030.md)预定义的唯一后继起草；方向见[唯一主线](../MAINLINE.md)，当前状态见[03](../active/roadmap/03_CURRENT_EXECUTION.md)，候选定义见[R2](SPEC-NATIVE-CONTENT-CORE_R2_CANDIDATE_COMPARISON_20261030.md)，重构框架见[重构方案](PLAN-NATIVE-CONTENT-CORE_REDESIGN_20261030.md)。

## 1. 目的与边界

R5 以 `FAIL_BEHAVIOR` 结案，归因清晰：比较门真实通过（holdout 比最好基线高 0.169，CI [0.150,0.188]；Q2 信用差 +0.117，CI [0.095,0.137]），但行为门全败（answer 0.323/门0.80），失败集中两处架构级缺口：

1. **答案未稳定消费选中内容**：收据 0.429（门 0.95）、cut_drop 0.166（门 0.50），过半正确回答与选中值不对应，任务收益走 query/输出头旁路；
2. **自由生成与教师强制差距**：train loss 0.08 nats 而同分布自由生成 exact 仅 0.329（门 0.85），加更新数不解决。

本件只立一个有界候选合同：候选在 `new_core_v1` 基础上做且仅做针对上述两缺口的改动，同一数据、同一基线、同一预算口径、同一门重新比较。本件不授权该候选的实现或训练；实现与训练在用户批准后按本件§5/§7 执行。

## 2. 候选定义与身份

候选 id：`new_core_v2`；前身 `new_core_v1`（R3/R5），身份可追溯，参数 revision +1。

### 2.1 改动清单（仅两处）

| 改动 | 内容 | 针对缺口 |
|---|---|---|
| A 复制轨 | 每生成步，读取路径从 store 打分选出记录；被选记录 value span 的**原始字节**经复制/指针通道进入该步与后续步的字节分布（选中值作为原子复制片段供给解码器，最终下一字节分布=词表 softmax 与复制分布的合成）。通道输入只允许来自 store 内容字节与模型自身状态；**禁止接触 gold 答案、评分参照或宿主注入的选择**；通道参数按前缀归入 `reader`/`decoder` 信用组，不新增信用组 | 缺口1 |
| B 自前缀日程 | 训练侧：更新 1–256 纯教师强制；更新 257–512 每 episode 以 0.5 概率把目标后半段的前缀替换为模型自身贪心已生成前缀（教师强制目标字节不变，仅换前缀来源；抽样由 run seed 决定，日程预声明、不因结果调整）。仅作用于本候选 Q1/Q2 run，不作用于任何基线 | 缺口2 |

### 2.2 不变清单

store 结构与记录格式、读取打分、condition、评分器（固定分母、cut/rescue/swap、收据定义）、manifests（train/query/holdout sha）、seed 调度、512 更新、batch 4、Q2 信用组语义（stop 仍恰为 reader+decoder）、[R4 §4.1–4.5](SPEC-NATIVE-CONTENT-CORE_R4_PRICING_AND_GATES_20261030.md)全部门值——全部逐字沿用，本件不改任何门。bundle 格式 revision +1；`new_core_v1` 端点保持可加载，不做迁移。

## 3. 比较问题

- **Q1 架构包比较（同 R5）**：`new_core_v2` 对照两条基线的 holdout/query 任务净收益与机制/保持/成本合取；两处改动一起变化时只称架构包收益。
- **Q2 信用单因素（同 R5）**：`new_core_v2` 内 credit_on/credit_stop 只切 reader+decoder（含复制轨参数），同初态前向、同结构、同数据。
- **归因读数（描述性，不新增门）**：收据按 current/previous/unknown 分层；cut_drop 直接度量复制轨因果贡献；内插−自由生成差距度量日程 B 效果；自前缀阶段（257–512）与纯教师阶段（1–256）的 train loss 对照。

## 4. 公平性、数据与基线复用

| 口径 | 规则 |
|---|---|
| 数据 | 同 [R5](../../reports/taiji_native_core_r5_training_20261030.json)：train `native_core_r1_train_v1.json`（sha 5d151ba9…）；query/holdout 仅评价时读；sealed 不读 |
| 曝光 | 每更新 4 episode、512 更新、batch 4；三组同 |
| seed | 0/1/2，同 R5 数据顺序 |
| 基线 | **复用 R5 端点与评价，不重训**：`corrected_old_v1`、`ordinary_v1` 的 R5 训练/评价报告按 sha 引用为基线证据。理由：数据/协议/seed 与 R5 逐字相同，重训零信息增量、多花约 1.5 万 CPU 秒。替代方案（全量重跑）被否决；若用户要求重跑则修订本节后再执行 |
| 计算 | 以实测 CPU 秒/更新为主口径；候选定价后冻结 |

## 5. 实施与工程门（批准后执行）

1. 实现 `new_core_v2`（复制轨+日程 B）＋测试；公开 fixture 必须包含：复制通道不接触 gold 的反例（宿主伪造选择必须被拒）、切断选中 span 后复制分布随之为空的退化路径、日程 B 的确定性抽样可复现；
2. 结构探针：零步前向 digest、复制轨参数梯度归组正确（reader/decoder，content_encoder 仍无）、信用 stop 后复制轨无梯度；
3. 恢复门：零步新进程恢复 digest 一致＋512 端点恢复（逐 run）；
4. 任一工程门失败 → `ENGINEERING_FAILURE`，不进入训练。

## 6. 数值门（沿用 R4 §4.1–4.5 原值，不改）

行为门（holdout 固定分母）：`answer_exact ≥0.80`、`current ≥0.85`、`previous ≥0.75`、`unknown ≥0.85`、`state_exact ≥0.90`；机制门：`cut_drop ≥0.50`、`rescue_gain ≥0.90`、相关/无关交换改变差 CI 下界 >0；保持门：内插 `≥0.85`、holdout−内插 `≤0.10`、收据 `≥0.95`；比较门：holdout 差 `≥0.05` 且 CI 下界 >0、query 不低于 −0.03、Q2 差 `≥0.05` 且 CI 下界 >0；成本门：新核心/最高基线 CPU `≤2.0`；恢复门：全过。判定与分类完全按 [R4 §4.6](SPEC-NATIVE-CONTENT-CORE_R4_PRICING_AND_GATES_20261030.md)：`PASS_Q1` / `FAIL_BEHAVIOR` / `FAIL_MECHANISM` / `FAIL_RETENTION` / `FAIL_RECOVERY` / `FAIL_COST` / `INCONCLUSIVE` / `ENGINEERING_FAILURE` / `BUDGET_STOP`。

## 7. 预算与停止线

1. 先跑 50 更新 train-only pilot，实测 `new_core_v2` 单价（复制轨+自前缀增耗）；
2. 冻结预算：预计训练 9 run（Q1 候选×3 seed ＋ Q2 两臂×3 seed）×512 更新 ≈4,608 更新；按 R5 实测单价上界估计训练 ≈35,000 CPU 秒、评价 ≈10,000、pilot ≈2,000；
3. **总停止线 120,000 CPU 秒 / 48 墙钟小时**（含 pilot、训练、评价；约 2.6 倍余量）；unit_cap_factor 1.5 沿用；
4. 停止线触发任一（恢复失败、NaN、超预算、query/holdout 被提前读、checkpoint 不能保存）即 `BUDGET_STOP`/`ENGINEERING_FAILURE`，不续跑。

## 8. 执行顺序

用户批准本件 → §5 实现与工程门 → §7 pilot 定价并冻结 → Q1 3 run ＋ Q2 6 run → 固定分母评价 → 按 R4 §4.6 结案并落账 03。

## 9. 本件明确不做

- 不读 sealed；不加数据、不加更新、不加 seed；不重训基线；
- 不改 R4 §4.1–4.5 任何门值，不新增门；
- 不因一次失败自动加剂量/换题/重跑；负结果回唯一主要缺口（架构归因照实记录，不洗成成功）；
- `PASS_Q1` 不自动晋级 R6：R6 能力接力另按[重构方案](PLAN-NATIVE-CONTENT-CORE_REDESIGN_20261030.md)§2 准入；
- 本件自身不授权实现或训练。

## 10. 审批方式

与 R5 同法：用户显式批准（`--approve-r5b`）后，§5 起可执行；未批准前任何实现/训练/评价都违反本件。

**执行记录**：2026-10-30 用户经弹窗选择「批准，按合同执行」，本件自 §5 起生效；实现与训练授权成立，门值不得再改。
