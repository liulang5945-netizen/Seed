# N177 E1 状态依赖查询与内容消费合同

日期：2026-10-29，Asia/Shanghai。状态：**合同冻结，未启动实现或训练。** 上级方向见[E决策](PLAN-N1-177_E_state_consumption_and_credit_decision_20261029.md)；当前队首与进度只看[03](../active/roadmap/03_CURRENT_EXECUTION.md)。本件不修改N177.5 v2、N177.6冻结SPEC、已发布评价或N176独立证据。

## 1. 要解决的问题与范围

旧N177.6端点的查询回答已接近饱和，但规范world state在Parser-permuted臂为0而回答仍约98.35%。这说明回答可从原始证据、hidden或其他路径得到，旧文本端点无法区分“模型读了状态”与“模型重放了题面”。E1因此不再以旧问答分涨跌为主证，而是建立一个**同一查询、不同合法事实必须给出不同答案**的新端点，并让回答只能经“状态内容特征”进入解码器。

本包只覆盖封闭小世界：单一实体、单一属性（颜色）、固定取值表、事实陈述/延迟查询/更正/旧值查询/无关干扰/歧义。它不表示开放中文理解、跨经历记忆、自然目标、规划、行动或L2/L3/L4通过。

## 2. 与旧资产的关系

- 复用：Taiji `CognitiveState`、`JointTaskStateV1`、`JointInputProducerV1`、V4语言候选、现有checkpoint/恢复合同、来源span与SHA绑定。
- 不复用：N177.6的query/holdout题面、评分器阈值、已消费sealed面；N175/N176未消费面继续封存。
- 新数据、新family、新manifest、新seed、新版本号。旧15臂与旧评价保持独立身份，不因本包改写。
- E0（终点独立恢复/B真实链补证）可与E1并行；E0缺项未闭合前不得进入F正式训练。

## 3. 新任务端点

### 3.1 封闭世界

| 维度 | 固定集合 |
|---|---|
| 实体 | 杯子、盒子、钥匙、书、球、笔、手机、眼镜（8） |
| 属性 | 颜色（单一属性，避免组合爆炸） |
| 取值 | 红、蓝、绿、黄、黑、白、紫、橙（8） |
| 模板 | 每类事实陈述/查询/更正/旧值查询各2种等义改写 |

核心family＝`(entity, value)`，共64个；按固定seed分成32 train / 16 query / 16 holdout。每个family生成4个episode，每个episode固定8轮：

1. 事实陈述：`<entity>的颜色是<value>。`
2. 无关短对话（延迟）。
3. 更正前查询：`<entity>的颜色是什么？`
4. 更正：`<entity>的颜色改成<other_value>。`
5. 更正后查询：同一句查询文本。
6. 无关实体干扰：`<other_entity>的颜色是<other_value>。`
7. 旧值查询：`<entity>之前是什么颜色？`
8. 歧义/否定：`<entity>的颜色不确定。`

关键约束：**同一实体、同一查询文本，在不同family中对应不同当前值**。模型若忽略状态，只能按固定先验猜8个颜色之一；查询文本本身不能确定答案。

### 3.2 分区与泄漏检查

- family整体属于一个split；实体、属性、颜色原子词可跨split复用，`(entity, value)`完整组合不得跨split。
- 分层约束：8个实体都出现在train/query/holdout三个split；每个实体在holdout中的当前值不得与该实体在train中的任一当前值相同，用来排除`entity→value`死记。
- holdout包含“训练中见过的实体、但当前值不同”的组合；query与holdout的取值分布按实体和颜色平衡。
- query仅作描述性评分，不参与checkpoint/阈值/超参选择；holdout是主判定。
- 生成manifest时校验：family唯一、episode脚本摘要唯一、完整input/target对无跨split重复、每个family的更正前后值合法且不同、干扰实体不等于查询实体、歧义轮不携带可写事实。
- 训练进程不得读取query/holdout正文或标签；评分器在训练退出后独立运行。

## 4. 允许的证据面与信息路径

主端点只允许：

```text
原始输入 → Taiji producer → 持久内容特征（离散值+连续向量+来源span/版本）
        → 查询条件选择（学习型打分） → 选中特征
        → 解码器（当前查询字节 + 选中特征） → 中文回答
```

- **主端点禁止**解码器直接读取事实陈述的原始字节、完整历史、Gold WorldObject、Gold span或答案模板。
- 原始字节仍保存在来源账本，用于审计、交换和救援；但主端点对解码器的访问必须显式受限。
- 选中特征必须带`entity/attribute/value/valid_from/revision/source_span/input_sha256`；离散值与连续向量指向同一内容身份。
- 查询选择由模型自己的可学习打分器完成；宿主只提供合法候选、权限和来源校验，不能按Gold字段套模板输出。
- 另设`raw_history_only`描述性对照：同一查询，解码器只读原始事实字节、不读选中特征。若它能答对，主端点的“状态消费”主张不成立。

### 4.1 持久内容特征

- 离散：`entity_id`、`attribute`、`value_id`、`valid_from`、`superseded_by`、`revision`、`source_span`、`input_sha256`。
- 连续：producer从原始输入生成的固定维度向量（建议64维），与离散值同一来源、同一revision；不允许用Python字典查表冒充学习表征。
- 更正：旧值不删除，标记`superseded_by`并保留`valid_from`；当前查询选最新有效值，旧值查询选被替换值。
- 歧义/否定：不产生可写事实；主端点候选为空时输出“不确定/无法判断”，不得猜值。

### 4.2 查询条件选择

- 训练期：对匹配`entity+attribute`的合法候选做softmax注意力，得到加权特征；注意力权重与选中候选ID写入receipt。
- 推理期：hard argmax选一个候选，但报告注意力分布；不得在推理时注入Gold候选顺序。
- 对照：相关交换（换当前值）、无关交换（换别的实体值）、切断（移除选中特征）、救援（恢复选中特征）必须走同一接口，不能靠破坏SHA/revision伪造失败。

## 5. 评分与因果判据

答案正确性与state正确性分账：

| 指标 | 定义 | 用途 |
|---|---|---|
| `answer_exact` | 规范化后生成正文与合法目标完全一致 | 主行为读数 |
| `state_exact` | 当前`(entity, attribute, value)`与事实一致 | 状态正确性，不替代答案 |
| `relevant_swap_delta` | 同query换合法当前值后答案改为交换值的比例 | 状态是否被实际消费 |
| `irrelevant_swap_keep` | 换无关实体值后答案保持不变的比例 | 选择是否聚焦 |
| `cut_drop` | 移除选中特征后答案正确率下降 | 特征是否必要 |
| `rescue_gain` | 恢复特征后答案恢复比例 | 是否可合法救援 |
| `raw_only_match` | raw-history-only对照与主端点的一致率 | 是否仍靠原始历史旁路 |
| `old_value_exact` | 旧值查询答对被替换值 | 历史/版本是否被消费 |
| `ambiguity_abstain` | 歧义轮输出“不确定/无法判断”且不写入事实 | 不猜、不误写 |

F阶段主判决使用holdout family-macro，三seed分别报告；query只作描述。相关交换/无关交换/切断/救援均用固定参数、固定query、固定初始状态，成对报告，不把报错计作成功。

### 5.1 预注册方向性出口

E1只冻结方向，具体数值在F训练前由train-only标定冻结，且不得用holdout调参：

- `PASS_STATE_CONSUMPTION`：`relevant_swap_delta`方向为正且达到F冻结的数值门、`irrelevant_swap_keep`达到F冻结的保持门、`cut_drop`达到F冻结的下降门、`rescue_gain`达到F冻结的恢复门，且`raw_only_match`不能解释主端点收益。
- `FAIL_RAW_BYPASS`：raw-history-only与主端点同等或更好；状态特征不是必要路径。
- `FAIL_NO_STATE_EFFECT`：相关交换不改变答案，或切断不降低答案；特征存在但未被消费。
- `INCONCLUSIVE`：工程/来源/恢复有效，但效应量或三seed一致性未达到F冻结的数值门。
- `ENGINEERING_FAILURE` / `BUDGET_STOP`：保存、恢复、来源、隔离或资源触顶；原样留证，不写成能力负结果。

## 6. 任务信用控制

在同一新接口上比较两臂：

| 臂 | 回答loss对producer/选择器 | 其余 |
|---|---|---|
| `credit_on` | 允许反传到连续内容特征与查询选择器 | 与`credit_stop`共享父参数、数据顺序、曝光、辅助slot/act目标、允许参数组和资源 |
| `credit_stop` | 对同一特征stop-gradient | 解码器仍更新；辅助目标仍按原规则 |

- 训练后轨迹分化是被测效果，不要求继续前向相同。
- 回答loss到producer的真实梯度、参数差和ﬁnite检查单列；离散span/act选择的信用另行声明，不能由连续特征梯度推断它们已受任务纠正。
- 辅助slot/act CE可以帮助producer定位实体/属性，但不能替代回答信用；两臂必须相同。
- 旧能力保持：固定[N176 F0训练探针](../../reports/taiji_n1_plan176_pilot_selection_20261026.json)16题的自由生成UTF-8/结束/答案，训练前后对比；下降>3个百分点记局部遗忘。不得读N177.6 holdout或N176 sealed面。

## 7. 工程与恢复门

E2实现、E3零步验收必须先于F：

- 新checkpoint包含producer、内容特征、查询选择器、解码器、优化器、RNG、update cursor、manifest/order/cache SHA、下一episode cursor；不得把autograd图存入checkpoint。
- zero保存→新进程恢复→同一输入parse/answer/注意力分布一致→下一步输入一致。
- 禁止更新组：冻结producer/选择器/解码器分别验证；无越权写入；optimizer参数组与cursor不变。
- 来源隔离：参数、optimizer、cursor和需保持状态不变；不加载正式checkpoint/cache、不读query/holdout/sealed、不执行optimizer step。
- 相关/无关交换必须使用合法成对输入或受控克隆，保留有效来源；不能靠破坏hash/revision让校验报错。
- 真实输入链（非合成logit替身）必须至少跑通一个train episode的producer→feature→selector→decoder；合成fixture只证明接口。

## 8. 实施分解与资源

| 子步 | 交付 | 出口 |
|---|---|---|
| E2a | 新课程/生成器、manifest、来源与泄漏检查 | 静态检查通过，不读sealed |
| E2b | 持久内容特征+查询选择器+解码器消费接口；raw-only对照 | 相关/无关/切断/救援接口可运行 |
| E2c | 信用on/stop同图控制；连续特征梯度与参数责任 | 合成+真实train episode各一次 |
| E3 | 零步checkpoint/optimizer/RNG/cursor恢复、禁写组、前向等价 | 全部通过后交F |
| F | 冻结样本/种子/数值门/保持/资源，一次有界训练与评价 | 按F合同结案 |

E2/E3为零步工程包，上限8,000 process-CPU秒/6,000 wall秒；触顶留partial。F预算在E3后用实测单位成本×固定更新数×1.3余量计算并写入F合同，不在此预设；按用户2026-10-26预算政策评估合理性，不硬性封顶机时。

## 9. 架构归因（必交）

- `architecture_verdict`：E1合同冻结的是可检验假设，尚非判决；F结案必须写“结构缺陷已确认/架构瓶颈有因果支持/嫌疑未证实/其他原因”。
- `task_scope`：封闭小世界单实体单属性颜色事实、延迟查询、更正、旧值、无关干扰、歧义。
- `suspected_mechanism`：旧端点回答可从原始证据或hidden旁路得到；状态特征没有独立消费者；回答loss未到producer。
- `information_path`：raw→producer→持久内容特征→查询选择→解码器。
- `credit_path`：answer CE→解码器+内容特征/选择器（credit_on）或stop-gradient（credit_stop）；slot/act CE单列。
- `competing_causes`：数据平衡/模板泄漏、特征维度、解码器容量、优化剂量、初始化/数据顺序。
- `next_action`：E2a→E2b→E2c→E3；E0缺项与E2并行补齐，未闭合不得进F。

## 10. 明确不做

- 不在旧N177.6近饱和端点上再加训练或换seed求过门。
- 不用宿主播模板、Gold WorldObject、Gold span或答案缓冲区冒充模型消费。
- 不把精确值字节复制当作学习型选择的证据；若使用值字节，必须单列`copy_only`对照并披露寻址责任。
- 不把相关交换的报错、SHA失配或校验拒绝计作因果成功。
- 不在E2/E3未通过前启动F正式训练，不读新sealed面，不追加epoch/换题/重抽seed。
