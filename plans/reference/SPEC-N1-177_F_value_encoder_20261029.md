# N177 F 字节级值编码器候选合同

日期：2026-10-29，Asia/Shanghai。状态：**合同冻结；F0已完成；完整E3内容层通过、严格lineage gap；F1未启动。** 前置 [E决策](PLAN-N1-177_E_state_consumption_and_credit_decision_20261029.md)、[E1合同](SPEC-N1-177_E1_state_consumption_contract_20261029.md)、[03当前执行](../active/roadmap/03_CURRENT_EXECUTION.md)。本件不修改 N177.5 v2、N177.6 冻结 SPEC 或已发布评价。

## 1. 要解决的问题与范围

E2b-2 已确认：producer 的 mean、value-span mean、value-span 逐字节三种连续特征都只有 rank 2，8 色不可分；value-token 一热特征能通过控制，但 `copy_only` 与主路径一致，说明它只是离散值地址复制。F 因此不再测“能否复制值”，而是测：

**同一模型能否从值 span 的原始字节学出一个连续特征，并用它驱动回答；当主路径被禁止读原始值字节时，这个特征仍能工作，而 copy-only 路径不能替代它。**

范围仍是 E1 的封闭小世界：8 实体、单属性“颜色”、8 色、单对象事实/更正/旧值/无关干扰/歧义。不表示开放中文、跨经历记忆、自然目标、规划、行动、自主性或 L2/L3/L4 通过。

## 2. 数据与隔离

- 使用 [E1 课程](SPEC-N1-177_E1_state_consumption_contract_20261029.md)：64 family，32 train / 16 query / 16 holdout；每 family 4 episodes、每 episode 8 turns。
- F1 训练只读 train（128 episodes）；query 只作描述；holdout 是主判定。训练进程不得读 query/holdout 正文或标签，评分器在训练退出后独立运行。
- 8 实体完整词表进入 F1 训练；已知实体“钥匙”只用于 F0/工程 fixture。producer 的新实体实体抽取缺口由 F1 的 slot/act 监督负责，不冒充状态消费失败。
- 不重读 N175/N176 未消费面、N177.6 holdout 或任何旧 sealed 面。旧能力保持只用 [N176 F0 训练探针](../../reports/taiji_n1_plan176_pilot_selection_20261026.json) 16 题。

## 3. 候选与信息路径

```text
事实原始字节 → producer span parse（entity/attribute/value + 来源）
             → value byte encoder（值 span 原始字节 → 连续特征）
             → persistent content store → query selector
             → value readout → byte decoder（查询字节 + 选中特征）
             → 中文回答
```

- **主路径**：byte decoder 只接收当前查询字节、选中特征和必要的状态元数据；**禁止**接收事实原文、值 span 原始字节、Gold WorldObject、Gold span 或答案模板。
- **value byte encoder**：小型的可训练字节编码器（建议 1 层 GRU/卷积，输入为 producer 给出的 value span 字节，输出固定 64 维）；参数必须进入 checkpoint，回答 loss 可反传到这里。
- **byte decoder**：推荐复用 V4 query-only 状态 + feature-conditioned byte bias；F0 标定若证明 V4 路径不可行，可改用独立小字节解码器，但必须在 F0 报告中显式登记，不能静默换架构。两臂必须共用同一 decoder。
- **copy_only 下界**：同一 decoder，但特征输入改为离散值的原始字节/一热地址，不经过 value encoder，也不向 encoder 反传。它只作下界，不作为学习证据。
- **no_state 下界**：只有查询字节、没有选中特征；用于证明特征不是可有可无。
- producer、selector、store、readout、decoder 的身份、来源和 revision 必须同 bundle；离散值与连续特征指向同一 `(entity, attribute, value, source)`。

## 4. 训练臂与信用控制

正式 F1 为 4 臂 × 3 seeds：

| 臂 | 特征输入 | 回答 loss 对 value encoder | 其余 |
|---|---|---|---|
| `main_credit_on` | encoder 输出 | 允许反传 | 同一父/数据顺序/曝光/slot-act 辅助目标 |
| `main_credit_stop` | 同一 encoder 输出 | stop-gradient | 解码器仍更新；其余与 on 相同 |
| `copy_only` | 离散值原始字节/一热地址 | 无 encoder | 同一 decoder/数据/曝光 |
| `no_state` | 无特征 | 无 | 同一 decoder/数据/曝光 |

- 三 seeds 是独立初始化/数据顺序；父权重、producer、课程、更新数、batch、允许参数组、资源口径相同。
- 回答 loss 是唯一被切换的任务信用；slot/act CE 在四臂中保持相同。离散 span/act 的信用另行声明，不能由连续特征梯度推断它们已受任务纠正。
- `main_credit_on` 必须证明回答 loss 到 value encoder 的真实梯度、参数差和 finite；仅梯度非零不算学习收益。
- F1 固定更新数在 F0 标定后冻结；初拟 **1,280 updates（5 个 128-episode epoch，batch 4）**，不允许按中途表现加轮或换 seed。
- 每个 run 先保存 zero 并新进程恢复；每 256 步、课程边界、终点保存耐久 checkpoint；终点再次新进程恢复，并重现同一输入的 parse/feature/answer。

## 5. 一次固定评价

训练结束后，评分器独占打开 holdout：

| 指标 | 定义 | 用途 |
|---|---|---|
| `answer_exact` | 规范化后完整中文回答（含 `<|end|>` 前正文）完全一致 | 主行为读数 |
| `value_exact` | 颜色值 token 精确 | 与完整回答分账 |
| `state_exact` | 当前 `(entity, attribute, value)` 与事实一致 | 状态正确性 |
| `relevant_swap_delta` | 同 query 换合法当前值后答案改为交换值 | 状态是否被消费 |
| `irrelevant_swap_keep` | 换无关实体值后答案保持 | 选择是否聚焦 |
| `cut_drop` / `rescue_gain` | 移除/恢复选中特征后的下降/恢复 | 特征是否必要 |
| `copy_only_match` | copy_only 与 main 的 holdout 一致率/差值 | 是否仍是复制 |
| `no_state_gap` | main 与 no_state 的差值 | 特征是否可用 |
| `credit_gap` | `main_credit_on` 与 `main_credit_stop` 的差值 | 反馈是否作用到负责模块 |
| `old_retention` | N176 F0 16 探针的训练前后下降 | 旧能力保持 |
| `old_value_exact` / `ambiguity_abstain` | 旧值查询/歧义轮 | 历史与不猜 |

所有指标按 episode 内轮次评分，再 family macro，三 seed 分别报告。holdout 只在训练退出后读取；query 不参与阈值/checkpoint/超参选择。

### 5.1 预注册出口

数值门在 F0 标定后、读 holdout 前冻结；以下方向先冻：

- `PASS_LEARNED_VALUE`：main 的 holdout `answer_exact`/`value_exact` 达到冻结门，且相对 `copy_only` 与 `no_state` 均有冻结的最小差值；`relevant_swap_delta` 与 `cut_drop`/`rescue_gain` 达到冻结门；`credit_gap` 达到冻结门；旧保持下降 ≤3 个百分点。
- `FAIL_COPY_ONLY`：`copy_only` 与 main 同等或更好；主路径没有超出复制下界。
- `FAIL_NO_LEARNING`：`main_credit_on` 与 `main_credit_stop` 无冻结门差异；回答 loss 未作用于 encoder。
- `FAIL_NO_STATE_EFFECT`：main 与 no_state 无差异，或相关交换不改变答案。
- `FAIL_NO_GENERALIZATION`：holdout 上主门未达，但训练侧拟合/控制有效。
- `INCONCLUSIVE`：工程/来源/恢复有效，但效应量或三 seed 一致性不足。
- `ENGINEERING_FAILURE` / `BUDGET_STOP`：保存、恢复、来源、隔离或资源停止；原样留证。

## 6. 工程前置与恢复

F1 启动前必须闭合：

- **E0**：15 臂终点 fresh-process 恢复/parse-answer 复现回执；B 真实输入/有效 mask/optimizer/cursor 隔离补证。
- **完整 E3**：producer、V4/decoder、value encoder、selector、store、readout 同 bundle 的零步保存/独立恢复；同输入 parse/feature/answer 一致；禁止更新组；无越权写入；checkpoint 不保存 autograd 图。
- **copy_only 实现**：同一 decoder 的 copy-only 路径已可运行，且不向 encoder 反传；F0 报告登记其来源和参数组。
- **F0 train-only 标定**：12 个 run 的单位成本、checkpoint 保存/恢复、更新数、阈值和预算；不读 query/holdout，不执行正式训练。

任一前置缺失，F1 不启动；不以“先跑一版看结果”绕过。

## 7. 资源与停止线

- F0 标定上限 **2,000 process-CPU 秒 / 1,500 wall 秒**；只做 train-only 成本与恢复，不读 sealed。
- F1 预算在 F0 后用实测单位成本 × 1,280 updates × 12 runs × 1.3 余量计算，另加 30% 保存/评价/失败尾账；按用户 2026-10-26 预算政策评估合理性，不预设死上限，也不把“机时还够”当作继续训练的自动理由。
- 每个 run 设 runaway guard：超过 F0 实测单位成本 × 计划更新 × 1.5 即停止并留 partial；触顶不是科学负结果。
- 跨包停止线：连续 2 次 F1 级实验未取得机制净收益（不是单次负结果），暂停扩面，回到消费/信用诊断；不在同一配方上重复加轮/换 seed。

## 8. 架构归因（必交）

- `architecture_verdict`：F1 结案必须写“结构缺陷已确认/架构瓶颈有因果支持/嫌疑未证实/其他原因”；不得用复制路径通过冒充学习。
- `task_scope`：封闭小世界单实体单属性颜色事实、更正、旧值、无关干扰、歧义。
- `suspected_mechanism`：producer 连续编码丢值；一热 value-token 是地址复制；回答 loss 可能未到 encoder。
- `information_path`：事实字节 → producer span → value byte encoder → store/selector → readout → decoder。
- `credit_path`：answer CE → decoder + value encoder/readout（credit_on）或 stop-gradient（credit_stop）；slot/act CE 单列。
- `competing_causes`：数据覆盖、encoder 容量、decoder 容量、优化剂量、初始化/数据顺序。
- `next_action`：按出口转 T2/T3、改 encoder/decoder，或登记消费/信用缺口；不以扩大数据或参数默认续跑。

## 9. 明确不做

- 不把 copy-only、一热地址、Gold 值字节或宿主模板当作学习型特征。
- 不在 E0/完整 E3 未闭合时启动 F1；不读 holdout 调参/选 checkpoint。
- 不在同一配方上因未过门自动加 epoch、换 seed、换题或扩数据。
- 不把 F1 结果外推为开放中文、自主性或 L2/L3/L4 通过。
- 不修改 N176/N177.6 已冻结运行语义；共享源码改动先在独立候选验证，旧运行图保留。
