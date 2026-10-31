# R025：09.3 连续动力学与内容活动合同

> 文档状态：[合同] — 冻结09.3在[R021参考图](SPEC-TAIJI-CORE_REFERENCE_GRAPH_R021_20261031.md)（含R023门控出生/R024主体恢复）上的活动循环、活动选择与触发量范围。不授予任务训练、定价或自主性结论。调度只见[MAINLINE](../MAINLINE.md)。

日期2026-10-31。`taiji/living_core.py`字节不变；活动机构落在新模块`taiji/native_activity.py`。真实世界损失、世界EMA目标与可学习活动信用属09.4；本阶段只交付工程闭环。

## 1. 活动集合与语义（K=6）

| 活动 | 声明操作 | 状态结果 | 不许发生 |
|---|---|---|---|
| interact（交互） | 消费环境消息队列下一条raw为事件步 | H/F/nu经编码更新 | 队列为空时选择 |
| recall/consolidate（回忆/巩固） | 取store中最近record，按I3语义以**当前编码器**重编码原文为事件步 | H经重编码更新 | 生成新store条目；消费序列化latent |
| imagine（想象） | 以声明方式构造内部事件（最后事件编码＋固定种子固定范数扰动），mode=2前向 | 内部状态演化 | **写入store/源记录**；结果记为real |
| explore（探索） | 消费声明novel探针字节（不在store中）为事件步 | H经新上下文更新 | 写store |
| maintain（维护） | F范数限位复查（超限按edit_fast同式裁剪）、nu_ema一致性核查 | 超限时裁剪；否则显式no-op并落账 | 静默无记录 |
| rest/wait（休息/等待） | 不做前向步；时钟推进wait_ticks | H逐位不变；journal/时钟推进 | 任何状态漂移 |

## 2. 活动选择：模型所有、宿主供边界

- **策略头**（声明仪器，未训练、参数保持seeded初值）：输入`[mean_i N(h_i)(d); goal向量(d); nu_features(16); 预测误差标量; 未结结果计数; 队列计数]`→Linear→tanh(16)→Linear→6 logits；softmax记录概率，**确定性argmax**选择（tie按声明活动序）。
- **触发量**：外部消息（队列非空）、内在目标（声明goal字节）、预测误差（`mean_i‖world_pred_i−N(h_next)‖₂`，取上一轮最后一步）、未结结果（journal pending计数）、合法重放（store非空）全部进入特征，逐轮落账。
- **宿主边界**：只提供时钟、预算、合法mask（interact需队列非空、recall需store非空、wait恒合法）；全mask不可达（wait恒合法兜底）。宿主不替模型预选策略。

## 3. 对照臂与判别

固定阈值调度（声明阶梯：队列非空→interact；误差>0.10→recall；未结>0→维护；否则wait）在同一环境脚本上运行，两臂决策序列逐轮落账。本阶段只做**工程分岔核验**（两臂在相同轨迹上至少一轮选择不同，逐字记录），不签优劣。

## 4. journal与可追溯

每轮追加：`{seq, kind: activity_decision, activity, logits[6], probs[6], chosen_by: model|control, 触发量快照, h_norm_after, step_delta, imagined标志, store_size_after}`；journal沿用R024单调账本（seq只增）。**无新用户消息阶段**的判据：队列持续为空的≥3轮内，journal仍有逐轮decision条目，且至少一轮非wait活动产生状态变化（h范数或step数变化）。

## 5. 恢复

活动循环状态（模型+状态+队列+pending+policy头state_dict+journal）经R024 bundle的`extra`携带，同进程逐位恢复后下一轮决策与状态结果与父端逐位一致；独立进程续推沿用R024机制（本轮不加新进程协议）。

## 6. 预算、出口与交付

- 预算：全验证≤900 CPU秒/20墙钟分钟。策略头参数在本阶段**不变**（哈希前=后落账）；无optimizer、无数据分区、无任务评价。
- 出口：`passed_r025_activity_engineering_only`／`failed_loop`／`failed_traceability`／`failed_differentiation`／`failed_restore`／`budget_stop`／`engineering_failure`。通过仅表示活动循环、选择可追溯性、想象不写store、等待/维护语义与对照臂分岔在合成夹具上成立；**不签模型学会选择、活动策略优劣或自主性**。
- 交付：本合同、`taiji/native_activity.py`、`tests/taiji_native/test_native_activity_r025.py`、`scripts/training/verify_taiji_native_activity_r025.py`、`reports/taiji_core_r025_activity_dynamics_20261031.json`（逐轮journal、两臂序列、触发量、源码SHA与R021–R024绑定核对）。
