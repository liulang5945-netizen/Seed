# 共同内容核心：重构-4 定价与 R5 数值门合同

日期：2026-10-30，Asia/Shanghai。状态：**设计合同，未执行。执行前需要用户评审 R2 并批准 R4；本件不授权训练、评价或读取 query/holdout。** 方向见[唯一主线](../MAINLINE.md)，当前状态见[03](../active/roadmap/03_CURRENT_EXECUTION.md)，候选定义见[R2](SPEC-NATIVE-CONTENT-CORE_R2_CANDIDATE_COMPARISON_20261030.md)。

## 1. 目的与边界

R3 已实现三组对象并通过结构门，R4-prep 已物化 216/72/72 课程。本件只解决两件事：

1. **定价**：用 train 分区测出三组的真实单位成本，冻结 R5 的更新数、seed、预算与停止线；
2. **数值门**：在开训前把 R5 的行为、机制、保持、恢复和成本判据写死，避免看到结果再改门。

R4 不训练能力候选、不评价 query/holdout、不读 sealed。定价中的短程 pilot 只用于测成本/检查学习信号，不产生能力主张。

## 2. 公平性口径

三组对象必须使用同一 train 分区、同一 8 轮协议、同一目标字节、同一固定分母、同一评分器、同一 seed 调度。禁止给新核心额外数据、额外预训练、更大 batch 或更长更新。

| 口径 | 规则 |
|---|---|
| 数据曝光 | 每更新 4 个 episode（8 轮/ episode）；每 run 512 更新；三组相同 |
| 更新数 | 512；R4 只能在 [256, 1024] 内调整一次，并记录原因 |
| seed | 0/1/2；三组与 Q2 两臂共用同一 seed 的数据顺序 |
| 参数 | 记录 total/trainable 参数；不以“参数更多”给新核心额外预算，也不强行裁剪基线 |
| 计算 | 以实测 CPU 秒/更新与 wall 为主口径；所有 run 受同一总预算 |
| 权限 | 只读 `native_core_r1_train_v1.json`；query/holdout 与旧 sealed 不读 |
| 评分 | R1 固定分母；解析失败/拒答/错误都计入；首字节/第 t 步对齐 |

参数现状（R3 结构门实测）：新核心 144,355，全部参数 `requires_grad`；但 store 路径只有 reader/decoder/core/query/attention 实际接收任务梯度，content_encoder 需 `reencode` 才回传。修正旧图 123,438（仅 value bias 3,341 可训练）；普通端到端 120,097（全部可训练）。R4 不按参数量直接比较，报告参数量并检查计算预算。

## 3. R4 定价流程

### 3.1 训练前恢复门（逐对象）

1. 零步保存完整 bundle；
2. 新进程加载；
3. 同一 probe 输入的 selection/read/logits/answer digest 必须完全一致；
4. 任一对象失败即停止 R4，记工程失败，不进入定价。

2026-10-30 工程预检已跑（不训练）：用 4 个真实 train episode、每对象 16 个应答轮/460 目标字节，三组前向+反向均有限，新核心 reader/decoder/core/query/attention 组有梯度、content_encoder 无（符合持久 latent 边界），修正旧图仅 bias 可训练，普通基线全参数可训练；零步保存在新进程恢复 digest 一致；`optimizer_updates=0`。这只是数据适配与恢复预检，forward+backward 的 17.27 CPU/4.69 wall 是下界，不是 R4 单价。

### 3.2 单位成本

在 train 分区固定 batch=4 episodes、20 次更新（前 5 次预热丢弃），记录：

- `cpu_seconds_per_update` 中位数；
- `wall_seconds_per_update`；
- `peak_rss_mb`；
- `checkpoint_mb`；
- `restore_seconds`；
- `target_bytes_per_update`、`parameter_total`、`parameter_trainable`。

### 3.3 短程 pilot（每对象 50 更新，train-only）

检查三件事，不做能力评价：

- loss 是否下降（首 10 与末 10 的中位数差 > 0）；
- 每 25 更新保存/新进程恢复是否通过；
- 是否出现 NaN/非有限梯度。

pilot 失败记工程/优化失败，不自动加更新或换 seed。

### 3.4 预算投影

```text
projected_cpu = cpu_seconds_per_update × 512 × 15 × 1.25
```

15 个 run = Q1 三对象×3 seed（9） + Q2 新核心 credit_on/stop×3 seed（6）。

上限：**200,000 CPU 秒**、**48 小时 wall**。若投影超限，只允许一次把更新数降到 256；仍超限则预算停止，不自动提额。用户可在看到定价后另行批准更高额度。

### 3.5 定价出口

R4 报告必须给出三组单位成本、15-run 投影、峰值内存、恢复时间、pilot 学习信号与停止判断。没有这份报告，不启动 R5。

## 4. R5 数值门（开训前冻结）

### 4.1 行为门（holdout 分区，固定分母）

| 指标 | 门 |
|---|---|
| 全轮 `answer_exact` | ≥ 0.80 |
| `current_query` exact | ≥ 0.85 |
| `previous_query` exact | ≥ 0.75 |
| `unknown_query` exact | ≥ 0.85 |
| `state_exact`（读取值正确） | ≥ 0.90 |
| 解析失败计入分母 | 必须为 true；不得筛出 |

### 4.2 机制门

- 相关内容交换必须改变答案；无关内容交换的改变幅度必须显著小于相关交换（bootstrap 95% 下界 > 0）。
- cut 必须降低正确率：在 baseline 正确的 case 上，cut 后 exact 下降 ≥ 0.50。
- 合法救援必须恢复正确率：rescue 后 exact 恢复到 cut 前的 ≥ 0.90。
- 以上只用正确答案，不用“输出是否变化”。

评分实现：[native_content_evaluator.py](../../taiji/native_content_evaluator.py)。分母由声明的 answer-turn 列表决定，缺失/解析失败/拒答都计错；cut/rescue 只在 baseline 正确且 cut 错误的 case 上计算；swap 按期望答案判正确。3 项测试覆盖固定分母、状态值、cut/rescue 与 swap。

### 4.3 保持门

- 训练分布内插（train-like probe）exact ≥ 0.85；
- holdout 与内插的差距 ≤ 0.10；
- 复制/收据探针（逐字复制已陈述事实，不引入新内容）在训练后 exact ≥ 0.95；
- 当前三组没有可继承的父能力，R4 必须记录 `retention_parent = none`；R5 的“旧保持”指上述内插/复制稳定性，不冒充跨会话旧能力。

### 4.4 恢复与成本门

- 15/15 run 的零步与端点新进程恢复全部通过；
- 新核心总 CPU ≤ 最高基线的 2.0 倍；
- 任一 run 超过单价投影 1.5 倍或出现 NaN，记工程失败并停止该 run，不重跑加轮。

### 4.5 比较门（Q1）

- 新核心在 holdout `answer_exact` 上比 `corrected_old_v1` 与 `ordinary_v1` 的最好者高 ≥ 0.05（绝对），且 bootstrap 95% CI 下界 > 0；
- 新核心在 query 分区不得低于最好基线 0.03 以上；
- Q2 只看 `credit_on` 与 `credit_stop` 的同初态/同结构差异：任务净收益差 ≥ 0.05 且 CI 下界 > 0 才称信用贡献；梯度存在或 loss 下降不算。

### 4.6 结论分类

- `PASS_Q1`：4.1–4.5 全部满足；
- `FAIL_BEHAVIOR` / `FAIL_MECHANISM` / `FAIL_RETENTION` / `FAIL_RECOVERY` / `FAIL_COST`：对应门失败；
- `INCONCLUSIVE`：判别力、seed 一致性或预算不足；
- `ENGINEERING_FAILURE` / `BUDGET_STOP`：不是科学反证。

正负均按 R2/R3 出口结案；不因一次失败自动加更新、加 seed 或换题。

## 5. 执行顺序与停止线

1. 用户评审 R2，批准 R4；
2. 三组零步恢复门；
3. 单位成本 + 50 更新 pilot；
4. 写 R4 定价报告，冻结 R5 预算/更新/seed；
5. 用户批准 R5；
6. Q1 9 run + Q2 6 run；
7. 按 4.6 结案。

停止线：恢复失败、NaN、投影超 200k CPU/48h、query/holdout 被读、checkpoint 不能保存——任一出现即停，不进入 R5。

## 6. 本件明确不做

- 不训练、不评价、不读 query/holdout/sealed；
- 不把 pilot loss 下降当能力；
- 不因为新核心参数更多或更少而改公平口径；
- 不在看到 R5 结果后修改 4.1–4.5 的门。
