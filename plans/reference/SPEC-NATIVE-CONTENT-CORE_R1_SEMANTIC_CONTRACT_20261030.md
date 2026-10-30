# 共同内容核心：重构-1 统一语义合同

日期：2026-10-30，Asia/Shanghai。状态：**设计合同草案，待用户评审；不授权实现、训练、评价或读取新 sealed 面。** 上级方向见[共同内容核心重构方案](PLAN-NATIVE-CONTENT-CORE_REDESIGN_20261030.md)、[20261030核心设计重审](TAIJI_CORE_DESIGN_REVIEW_20261030.md)；当前队首与进度只看[03](../active/roadmap/03_CURRENT_EXECUTION.md)。本件不修改 N177/F/N178 已发布结果、冻结阈值、费用或旧 sealed 面。

## 1. 目的与范围

本件只解决一件事：在开训前把**任务语义、时间读取、byte 目标、生成协议、评分分母和来源身份**统一成一份可检查的合同，消除 F1/F1b 中已确认的目标/接口/评分错位。它不定义新模型参数、数据量、预算或能力门；这些在重构-2/4 另行冻结。

首个范围仍是有限合成世界，但必须能区分：当前值与历史值、同 query 不同当前值、竞争候选、无关干扰、未知与歧义。旧 N177/F/N178 面只能作开发诊断，不能作为新合同的主判定面。

## 2. 必须解除的旧矛盾

| 旧事实 | 本合同的约束 |
|---|---|
| F 的 answer loss 先取当前 logits、后 observe 当前 symbol，却对齐 `target[1:]` 并跳过首字节 | §5 固定第 t 步可见前缀与第 t 个目标 byte，首字节必须被监督；训练/生成/评分同一调用顺序 |
| F1b 只训可解析 query，`answer_exact` 却计整个 8 轮；部分指标按预测 act/实体筛选 | §6 固定分母：所有应答轮按类别计入，解析失败/拒答/错误不得被筛出分母 |
| 旧值查询目标是 old_value，但训练/评分统一取 `store.current`；selector 只按 entity/attribute 过滤 | §3/§4 显式时间意图与版本；selector 必须消费时间条件；当前/旧值有独立合法候选 |
| store 主要持久离散色名，连续 latent 由 Python 重建 | §4 同一内容身份同时有 latent、精确引用、来源、版本和 encoder revision |
| 生成是冻结 V4 每步加同一 value bias | §4.4 每步消费 query/time + 选中内容 + 生成前缀；固定 bias 只作修正基线 |
| query-only 只过滤 evidence sources，hidden 可保留事实 | §4.5 明确 hidden/workspace/source cache 可见性；只过滤引用表不算隔离 |
| cut 计“颜色与 baseline 不同”，rescue 计“恢复后又等于 baseline” | §6.3 cut 必须降低正确率、rescue 必须恢复正确率；复现一致只算复现 |
| 硬解析失败时内容不能入库，回答 loss 也到不了解析门 | §4.1 解析是辅助视图，不是唯一准入门；未解析内容可带来源/歧义进入，任务信用可到达声明模块 |

## 3. 任务与时间语义

### 3.1 最小世界（暂定形状）

- 实体：12 个；属性：3 个（颜色、位置、归属）；每属性 4 个合法值；版本：至少 3 个时间点。
- 内容身份：`(entity, attribute, value, valid_from, revision)`；实体/值/措辞/时间不得固定共现。
- 精确数量、中文模板和分词/byte 口径在重构-2 的数据审计后冻结；本件只约束结构，不把暂定数字当作已具备数据。

### 3.2 family 与必须存在的对照

每个 family 至少包含：

1. 同一 `(entity, attribute)` 的当前值与历史值；
2. 同 query 文本、不同当前值的两个 family；
3. 同当前值、不同历史值的两个 family；
4. 竞争候选：同属性、不同实体/值；
5. 无关干扰：另一个实体的合法事实；
6. 未知：实体或属性不在 store；
7. 歧义/否定：不得写入可执行事实。

### 3.3 固定回合协议（每个 episode 至少 8 轮）

| 轮 | 输入 | 目标/类别 |
|---|---|---|
| 1 | 事实陈述 A=v0 | 非应答，记录来源 |
| 2 | 无关短对话 | 非应答，延迟 |
| 3 | 当前查询 A=？ | `current_query` → v0 |
| 4 | 更正 A=v1 | 非应答，写新版本 |
| 5 | 同一句当前查询 | `current_query` → v1 |
| 6 | 无关实体 B=w | 非应答，干扰 |
| 7 | 旧值查询 A=之前？ | `previous_query` → v0 |
| 8 | 歧义/否定 A=不确定 | `unknown_query` 或 `abstain` |

第 3/5 轮 query 文本相同，答案必须随当前值变化；第 5/7 轮使用同一 store 的当前/历史版本，答案必须不同。若旧值不存在，目标为明确的“不知道/无法判断”，不得猜值。

### 3.4 时间意图

- 查询携带 `time_intent`：`current`、`previous`、`unknown`；`at_revision` 作为后续扩展，不在首版强制。
- `time_intent` 必须对读取器可见；它可以出现在 query 文本（“现在/之前”）或结构化条件中，但不能由宿主直接给出答案。
- 同一 query 文本配不同 `time_intent` 必须产生不同答案；这是时间读取的主对照。
- `previous` 指当前值之前最近一个被 `superseded_by` 的合法版本；无历史时返回明确未知。
- 更正保留旧版本，不删除；当前值由 `valid_from/valid_to` 和 revision 决定。
- 禁止用 `store.current` 统一回答所有查询；旧值路径必须实际读取历史。

### 3.5 未知、歧义与否定

- 未知实体/属性：合法候选为空；目标为“不知道/无法判断”，计为 `abstain_correct` 或 `answer_exact` 的一类，不按模型 act 筛除。
- 歧义/否定：不产生可写事实；若已有当前值，查询仍按既有版本回答；不得把否定句误写成新事实。
- 任何“猜一个合法值”的未知回答计为错误，不能因值在词表内就算合理。

## 4. 内容状态、读取与生成

### 4.1 准入与辅助解析

- 原始输入 + 来源先进入可学习编码；结构化 span/act 是**辅助视图**，用于校验、审计和辅助目标，不是唯一准入。
- 解析失败/低置信内容可以带 `status=unresolved/ambiguous` 进入 store；它保留原始 span、来源、置信和 `encoder_revision`，但不得被当作已确认事实写入当前值。
- 任务回答 loss 必须能到达至少一个声明的内容编码/读取/表达模块；硬解析头的辅助 CE 单列，不能替代任务信用。
- 来源、权限、owner/scope/session 是硬边界，不由软学习绕过。

### 4.2 持久内容身份

每条内容至少包含：

- `content_id`（稳定身份）；
- `entity/attribute/value`（辅助标签，不是唯一入口）；
- `latent`（连续内容向量）与 `exact_ref`（原始 span/字节引用），两者指向同一 `content_id`；
- `source_input_sha256`、`source_byte_span`、`owner/scope/session`；
- `valid_from`、`valid_to`、`superseded_by`、`revision`；
- `encoder_revision`、`confidence`、`status`（resolved/ambiguous/unresolved）。

更新规则：

- 更正写入新 revision；旧 revision 保留并标记 `superseded_by`；
- 当前读取只取有效版本；历史读取只取被替换版本；
- encoder 参数更新后，旧 latent 必须失效、重编码或显式迁移；checkpoint 记录 encoder revision；
- 精确引用/复制是独立可控通道；模型必须学“何时/读哪条”，宿主不得用 gold 地址或模板代选。

### 4.3 查询/时间条件读取

- 读取器输入：query 文本/目标、`time_intent`、合法候选 mask；不得输入 gold 答案、gold 候选顺序或宿主预选结果。
- 读取器必须对 `entity/attribute/time` 联合条件打分；同实体同属性不同版本必须有可区分的分数。
- 训练期可用 softmax 注意力；推理期 hard 选择但报告注意力分布；选择 ID 与来源写入 receipt。
- 候选 mask 只由权限/来源/合法版本决定；模型错误选择不能靠宿主回退纠正。
- 若声称“学习型时间选择”，必须在有竞争的合法候选上验证；单候选精确索引只能作机制对照。

### 4.4 表达与生成

- 每个生成步消费：query/time 条件、选中内容的读取向量、已生成前缀；内容读取随前缀逐步重算或经递归状态更新。
- 不允许把整句固定为“冻结语言头 + 同一个 value bias”；该路径保留为修正后的基线，不是新主候选。
- 生成头可以是 byte 级，也可以使用成熟的编码/解码组件；原生性约束是认知所有权，不是禁止成熟算法。
- 首字节、值字节、结束标记都按 §5 对齐；不得只监督模板而跳过值字节。

### 4.5 信息隔离

对 query-only 回答，必须重建/切断以下通道并分别报告：

- 原始事实字节不得出现在 query 的 observations/source cache；
- 先前事实不得只藏在 language hidden；需要重建 hidden 或显式切断；
- workspace 行、source cache、exact ref 的可见性逐项声明；
- 只过滤 evidence source 表不算隔离；
- 隔离测试必须用同一 query 的相关/无关交换和隐藏通道探针，而不是只看文本前缀。

## 5. Byte 时间对齐与训练/生成/评分协议

### 5.1 第 t 步定义

对目标答案字节序列 `y_0…y_{L-1}`：

- 第 t 步可见：输入、时间条件、选中内容、以及 `y_0…y_{t-1}`（教师强制用目标前缀，自由生成用模型已生成前缀）；
- 第 t 步计算 `logits_t`，只预测 `y_t`；
- 第 t 步 loss 为 `CE(logits_t, y_t)`；
- t=0 首字节必须被监督；不得先 observe `y_t` 再算 `logits_t`，不得用 `target[1:]` 跳过首字节。

### 5.2 训练/生成/评分同一协议

- 训练、验证、最终评分共用同一个 `next_logits` 调用顺序和同一个目标切片函数；
- 评分不得用教师强制的前缀；必须用模型自己的生成前缀，按同一停止规则结束；
- 目标字节包含完整中文回答和结束标记；不得在评分时重新分词或补模板；
- 任何 off-by-one/首字节跳过/目标重排都先修合同与代码，再谈训练结果。

### 5.3 固定分母

- `task_units` = 固定评价集的全部应答轮，按 `current_query/previous_query/distractor_query/unknown_query/non_query` 分类；
- 每个 unit 只计一次；解析失败、拒答、错误、空输出都计入对应分母；
- 不得按预测 act、实体是否存在、解析是否成功缩小分母；
- 主报告同时给 query-only 与全轮、state 与 answer、正确性与敏感性；
- 解析成功率/状态成功率作为诊断率单列，不替代 answer_exact 分母。

## 6. 评分与因果判据

### 6.1 指标

| 指标 | 定义 | 分母 |
|---|---|---|
| `answer_exact` | 完整回答字节（含结束标记）与目标完全一致 | 固定全部应答轮；另报 query-only |
| `value_exact` | 值 token 精确 | 同上，单独分账 |
| `state_exact` | 读取的 `(entity, attribute, value, revision)` 正确 | 同上 |
| `abstain_correct` | 未知/歧义轮输出明确不知道/无法判断 | 对应类别全部轮 |
| `parse_success` / `state_success` | 诊断率 | 固定分母，不用来筛主指标 |

### 6.2 相关/无关交换

- `relevant_swap`：在同一 query 下换合法当前值或合法历史版本；正确答案必须随之改变；报告“改变率”和“正确率”。
- `irrelevant_swap`：换无关实体/属性；正确答案必须保持；报告“保持率”和“正确率”。
- 只报告输出变化率、不报告正确率，不算因果证据。

### 6.3 切断与救援

- `cut`：移除选中内容读取/关键通道；必须使正确率下降；报告下降量与分母。
- `rescue`：恢复同一内容/通道；必须使正确率恢复到 baseline；报告恢复量。
- 若 baseline 本身错误，该 case 不计入因果分子，但必须单列数量；不得用错误 baseline 的“恢复一致”冒充救援。
- 重复同输入得到同输出只证明复现，不证明正确性因果。

### 6.4 必须分账

- query-only vs 全轮；
- state 正确 vs answer 正确；
- 正确性 vs 敏感性/保持；
- 解析/状态工程上界 vs 生产路径；
- 训练面拟合 vs 未见面；
- 架构包整体 vs 单一 credit 因素。

## 7. 信用与可训练模块

| 模块组 | 首版允许 | 必须声明 |
|---|---|---|
| 内容编码/写入 | 任务 loss + 辅助 loss | 哪些参数接收 answer loss；离散写入用何种信用机制 |
| 读取/时间选择 | 任务 loss | 候选 mask 来源；是否 hard/soft；是否接收回答信用 |
| 语言/行动头 | 任务 loss | 每步内容读取如何进入；是否冻结父模型 |
| 辅助解析头 | 辅助 CE | 不自动获得回答信用；与任务路径分账 |
| 来源/权限/owner | 禁止写入 | 任何 loss 不得更新 |

- `credit_on/stop` 必须同初态前向、同结构、同数据/曝光，只切指定信用；比较任务净收益，不把梯度非零当收益。
- 禁写组逐项测试；离散分支允许合适的信用机制，不要求所有接口 autograd。
- 回答 loss 必须能到达至少一个声明的内容读取/表达模块；否则合同不通过。

## 8. 比较设计

| 对象 | 作用 | 不能得出的结论 |
|---|---|---|
| 修正语义的旧图基线 | 公平评价硬解析/current-only/固定 bias 的实际代价 | 不能只修新图而让旧基线背负已知错目标 |
| 普通端到端学习基线 | 同数据/权限/资源下区分机制是否值得复杂度 | 不借不同语料/参数/外部模型偷换研究对象 |
| 新共同内容核心 | 主架构包的行为/保持/资源比较 | 多部分一起变化时只称架构包收益 |
| 新图内 credit on/stop | 只切指定信用 | 不把架构包差异混成信用单因素 |

- 三者共用同一数据分区、父能力保持集、权限、评分器和资源口径；不得用旧已消费 sealed 面做最终门。
- 先比较架构包整体，再在胜者内做 credit 单因素；不要同时改任务、表示、读出和信用。
- 普通基线可以使用成熟序列模型，但必须同参数/计算预算、同输入/目标、同评分协议；它是对照，不是采用对象。

## 9. 来源、恢复与工程门

- 每条内容绑定 source input SHA、byte span、owner/scope/session、revision、encoder revision；checkpoint 保存这些身份。
- 训练前必须验证：完整 checkpoint（模型/优化器/RNG/cursor/内容 store/encoder revision）可保存、独立进程可恢复、继续行为一致；checkpoint 不保存 autograd 图。
- 公开开发 fixture 与 sealed 面分离；重构-1 只用公开/合成案例，不读新 sealed、不加载正式评价面。
- 切 cut/rescue、相关/无关交换必须走同一生产接口，不能靠破坏 SHA/revision 或宿主回填伪造。
- 工程上界（oracle 解析/状态）只能作隔离参照，不能进入生产路径或能力主张。

## 10. 重构-1 出口清单

以下全部通过才进入重构-2 的候选/基线与成本设计；任一矛盾则先改合同，不开训：

- [ ] 每个应答轮有类别、固定分母和目标；解析失败/拒答/错误不被筛出。
- [ ] 每个目标 byte 有同索引监督；首字节包含；训练/生成/评分同一协议。
- [ ] 存在同 query 不同当前值、同当前值不同历史值、同 query 文本不同 time_intent 的合法对照。
- [ ] 当前/旧值查询的目标与 store 版本语义一致；旧值不靠 `current` 回退。
- [ ] selector 输入含 entity/attribute/time；有竞争候选和无关干扰；无 gold 选择。
- [ ] store 同时有 latent、exact_ref、来源、valid time、revision、encoder revision。
- [ ] query-only 的 fact bytes/hidden/workspace/source cache 可见性逐项声明并有探针。
- [ ] cut 计正确率下降、rescue 计正确率恢复；baseline 错误 case 单列。
- [ ] 任务 loss/辅助 loss/禁写组矩阵完整；至少一条任务信用到达声明的内容读取/表达模块。
- [ ] 修正旧图、普通基线、新核心同资源比较方案已写；credit on/stop 单独定义。
- [ ] 公开 fixture 与 sealed 分离；未读新 sealed；训练前 checkpoint/恢复门已定义。
- [ ] 用户已确认本合同的语义与范围；未确认前不进入重构-2/4 的实现或训练。

## 11. 明确不做

- 不把本件当作训练批准、能力通过或预算承诺。
- 不用 parser 覆盖率、train loss、恢复一致或输出变化冒充回答/状态因果收益。
- 不重读 N175/N176 未消费面、N177.6/F1/F1b/N178 旧 sealed 面做最终门。
- 不在同一配方上自动加 epoch/seed/剂量；不把旧 F/N178 费用清零。
- 不一次推倒 TSK、语言、记忆、世界和全部训练机制；按重构-2 的候选/基线比较选择。
