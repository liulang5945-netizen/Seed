# N177 F1 冻结提案（待 owner 签署，未启动）

日期：2026-10-29，Asia/Shanghai。状态：**owner 已签（2026-10-29）**：候选范围取「main 臂只训 encoder/bias，V4 冻结」；预算上限批准 ≈121,500 CPU 秒。本件不改[F 合同](SPEC-N1-177_F_value_encoder_20261029.md)、N177.5 v2 或 N177.6 冻结 SPEC；签署后应把本件的数字写回 F 合同或另发 v2。

前置现状（2026-10-29 实测）：

| 前置 | 状态 | 证据 |
|---|---|---|
| E0 | 闭合 | `reports/taiji_n1_plan177_6_t1_endpoint_receipts_20261029.json`、`..._historical_replay_20261029.json`、`..._state_isolation_20261029.json` |
| E2c 信用 on/stop | 闭合 | `reports/taiji_n1_plan177_e2c_credit_control_20261029.json` |
| E3 零步恢复/禁写组 | 闭合 | `reports/taiji_n1_plan177_e3_state_credit_20261029.json` |
| F0b 单价/恢复 | 闭合，**wall 超 96 秒** | `reports/taiji_n1_plan177_f_f0b_20261029.json` |
| 训练面覆盖 | 收窄到 4 实体（16 家族） | `reports/taiji_n1_plan177_e1_parse_coverage_20261029.json` |

## 1. 候选与四臂（待签：main 臂是否训 V4）

| 臂 | 特征来源 | 反向范围 | 其余 |
|---|---|---|---|
| `main_credit_on` | value encoder 输出（带图） | encoder + bias(+V4?) | 同一父路径/顺序/曝光 |
| `main_credit_stop` | 同一输出，`detach()` | 只有 bias(+V4?) | 同上 |
| `copy_only` | 值的一热地址 | 只有 bias(+V4?) | 不回传 encoder |
| `no_state` | 零向量 | 只有 bias(+V4?) | 无状态 |

**已裁 A（owner 2026-10-29）**：main 臂**只训 encoder/bias，V4 冻结**。理由同建议：F1 测的是"状态特征是否被消费、信用是否到特征"；V4 训练另立合同、另测单价，不混进本轮归因。

## 2.5 答案生成口径（2026-10-29 补记，实现面发现）

F 合同 §5 的主读数 `answer_exact` 要求完整中文回答，而本轮候选只训 encoder/bias、V4 冻结——必须明确"冻结 V4 如何用上这个 feature"。本件冻结为：**回答仍由冻结 V4 贪心逐步生成，生成时把同一 `bias(feature)` 加到每一步的下一字节 logits 上**。实现入口是 `ConditionalLanguageCandidateV4._generate_from_joint_cache(..., logit_bias=None)`：默认 `None` 时生成路径与旧版逐字节相同（有测：`tests/taiji_native/test_n177_f1_generation_bias.py` 两支——不给 bias 与原贪心一致；给了 bias 当场改选中字节）。这只加了一个默认为空的入口，不改旧生成语义。

未采纳的另一条：把评分降级为"答案首字节/值 token 精确"（会让 `answer_exact` 变成不适用，等于改冻结门）。

## 2. 更新、曝光与批量

- 1,280 updates/run × batch 4 = 5,120 episode-exposures（owner 2026-10-29 裁 A 口径；"5 epoch"仍是已登记的勘误）。
- 12 run = 4 臂 × 3 seed；每 run 独立初始化/顺序，父权重与课程相同。
- 课程：E1 train 的 16 个已覆盖家族（4 实体 × 4 值）；顺序按 `update_order` 固定，不按中途表现调整。

## 3. 预算（由 F0b 实测推出，待签）

- 单价：4.68 CPU 秒/更新（273 次计量更新的均值；中位 4.58，最大 7.69）。
- 15,360 updates（1,280×12）× 4.68 = **71,885 CPU 秒**。
- ×1.3 余量 = 93,450；再 +30% 保存/评价/失败尾账 = **≈121,500 CPU 秒（≈33.8 小时）**。
- 建议 stop line：单 run 上限 10,100 CPU 秒（1,280×4.68×1.3×1.3）；整包 121,500 CPU 秒；任一触顶即写 partial 并停，不补跑、不加 seed。
- 若待裁 A 改成"也训 V4"，本节数字作废，先补一次 V4 训练成本的短测。

## 4. 数值门（读 holdout 前冻结，待签）

沿用 F 合同 §5 的指标定义，建议初值（按 N177.6 F0 与 176 的实绩定，不用 holdout 调）：

| 门 | 建议阈值 | 理由 |
|---|---|---|
| `answer_exact`（main） | ≥ 0.85 | 与 N177.6 F0 的门同量级；低于此"会用状态答"不成立 |
| `value_exact`（main） | ≥ 0.90 | 值本身是主信号，应高于整句 |
| `state_exact` | ≥ 0.90 | 状态写入正确性 |
| `relevant_swap_delta` | ≥ +0.30 | 换值后答案必须跟着变，且幅度可观 |
| `irrelevant_swap_keep` | ≥ 0.95 | 无关交换不许改答案 |
| `cut_drop` | ≥ 0.30 | 切断特征应显著掉 |
| `rescue_gain` | ≥ 0.20 | 恢复后至少回来一部分 |
| `main − copy_only` | ≥ +0.10 | 超出复制下界 |
| `main − no_state` | ≥ +0.10 | 特征确实被用 |
| `credit_gap`（on − stop） | ≥ +0.10，且 ≥2/3 seed 为正、min ≥ −0.05 | 反馈作用到负责模块 |
| `old_retention` | 下降 ≤ 3pp/seed | 旧能力保持 |

任一方向门被满足与否，按 F 合同 §5.1 的出口判决（PASS/FAIL_*/INCONCLUSIVE/ENGINEERING_FAILURE/BUDGET_STOP）原样记录。

## 5. 停止线

- 读 holdout 前：12 run 全完成且恢复全真；任一 run 恢复失败 ⇒ ENGINEERING_FAILURE，不读 holdout。
- 读 holdout：一次，只读（与 N177.6 同口径），不因中途数字改阈值。
- 资源：触顶写 partial；不补跑、不加轮次、不换 seed。

## 6. owner 裁定（2026-10-29）

1. **候选范围**：main 臂只训 encoder/bias（V4 冻结）。
2. **预算**：批准 ≈121,500 CPU 秒作为 F1 上限（单线程 ≈33.8 小时；用多线程压缩 wall 不改 CPU 账）。

后续：本件数字写回 F 合同（或发 v2）后 F1 启动；停止线按 §5 执行，触顶写 partial 并停。
