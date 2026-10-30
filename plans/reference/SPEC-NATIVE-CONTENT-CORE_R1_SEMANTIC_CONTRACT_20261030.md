# 共同内容核心：重构-1 统一语义合同

日期：2026-10-30，Asia/Shanghai。状态：**五项首版设计边界已确认；世界只读审计已完成（名单/模板/样本计划）；精确split/覆盖/成本仍待R2/R4；其余语义一致性审计待执行，不授权训练、评价或读取新sealed面。** 上级方向见[共同内容核心重构方案](PLAN-NATIVE-CONTENT-CORE_REDESIGN_20261030.md)、[20261030核心设计重审](TAIJI_CORE_DESIGN_REVIEW_20261030.md)；当前队首与进度只看[03](../active/roadmap/03_CURRENT_EXECUTION.md)。本件不修改 N177/F/N178 已发布结果、冻结阈值、费用或旧 sealed 面。

## 1. 目的与范围

本件只解决一件事：在开训前把**任务语义、时间读取、byte 目标、生成协议、评分分母和来源身份**统一成一份可检查的合同，消除 F1/F1b 中已确认的目标/接口/评分错位。它确认首个世界形状与语义边界，不定义模型参数、episode/样本量、预算或能力数值门；这些在重构-2/4另行冻结。

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

### 2.1 当前代码触点（只读基线 22cd81b25）

下表只用于把 R1 条款落到具体文件，不代表已实现或已批准修改；行号对应当前 HEAD，代码未在本轮改动。

| R1 条款 | 当前触点 | 已确认事实 | 实现顺序 |
|---|---|---|---|
| 第 t 步 byte 目标 | [run_taiji_n1_plan177_f_f1.py](../../scripts/training/run_taiji_n1_plan177_f_f1.py) L87–L101；[conditional_language_v4.py](../../taiji/conditional_language_v4.py) `next_logits` | `for symbol in target[:-1]` 先取当前 logits、后 observe 当前 symbol，loss 用 `target[1:]`；首字节被跳过，目标与调用顺序错位 | 1. 先修为 `logits_t -> target_t`，首字节监督；训练/评分共用同一 target 切片 |
| 固定分母 | [evaluate_taiji_n1_plan177_f_f1_holdout.py](../../scripts/training/evaluate_taiji_n1_plan177_f_f1_holdout.py) L258–L343 | `answer_exact` 分母是 `len(rows)`，但 `value_exact/state_exact` 只用解析成功且 act=query 的 `query_turns`；`query_unparsed` 在 L279–L281 直接 `continue`，被筛出分母 | 2. 建 `task_units` 固定分母；解析失败/拒答/错误计对应类别；query-only 与全轮分开 |
| 当前/旧值 | [joint_state_consumption_course.py](../../taiji/joint_state_consumption_course.py) L275–L361；[joint_state_consumption.py](../../taiji/joint_state_consumption.py) L232–L252、L355–L441；评分器 L262 | 课程有 old_value 和“之前”查询；store 有 current/history；selector 只按 entity/attribute 打分；评分器对两轮都取 `store.current` | 3. 查询带 `time_intent`；selector 输入 time；current/previous 走不同合法版本；不用 `store.current` 统一回答 |
| cut/rescue 正确性 | 评分器 L321–L324 | cut 计 `cut != baseline`，rescue 计 `rescue == baseline`；错误 baseline 也能 rescue=1 | 4. 改为正确率下降/恢复；baseline 错误 case 单列 |
| 固定 bias/逐步读取 | [run_taiji_n1_plan177_f_f1.py](../../scripts/training/run_taiji_n1_plan177_f_f1.py) L92–L101；`_replay`/`_answer` | 整句每步加同一个 `bias(feature(value))`；V4 冻结 | 5. 新核心每步读 query/time + 选中内容 + 前缀；固定 bias 只作修正基线 |
| hidden/query-only | 评分器 `_replay`/`_answer` 路径；[conditional_language_v4.py](../../taiji/conditional_language_v4.py) | `_answer` 只过滤 evidence sources，仍用同一 `language_state`；hidden 可保留事实 | 6. 重建 query-only hidden/workspace/source cache，并做隐藏通道探针 |
| 解析准入与信用 | [joint_input_producer.py](../../taiji/joint_input_producer.py)；[joint_t1_trainer.py](../../taiji/joint_t1_trainer.py) | slot/act 离散化/detach；回答 loss 未到 producer；解析失败内容不能入库 | 7. 未解析内容带来源/状态入库；任务信用到达声明模块；辅助 CE 单列 |
| 来源/版本/恢复 | [joint_state_consumption.py](../../taiji/joint_state_consumption.py) 的 feature/store 结构；F0/F1 checkpoint | 现有 feature 有 source/revision，但缺统一 encoder revision/valid_to/status 迁移规则 | 8. 补 schema 与 checkpoint 迁移；训练前零步恢复 |

**最小修改顺序**：先 1（byte 对齐）与 2（固定分母），得到可解释的修正旧基线；再 3（time/selector/history），再做 6（hidden 隔离）；最后 5（动态读取）与 7（解析准入/信用）。第 1/2 条不修就开训，任何结果都不能作为新架构证据。

## 3. 任务与时间语义

### 3.1 首版世界形状（本轮确认）

**术语与实现边界：本节定义“任务域清单”，不是模型语义词表。** 12实体、属性及取值集合用于生成合法世界、数据分层与独立评分，不得成为自然输入的宿主查表解析器或唯一内容准入门。主模型不能因名称不在清单或辅助解析失败就丢弃原始内容；它可按来源/权限保留为未知或未解析候选，是否采信/回答另判。固定域内通过不表示已获得开放实体能力。

实体/值ID可用于存储寻址、训练标签或评分身份，但不能由宿主从正确答案反推并注入读取器。语言边界的256/257字节符号表与任务域清单分开，均不等于模型唯一内部语义；连续表示、来源引用及时间关系仍由新核心承担。是否采用其他可学习编码属于后续比较设计，不由“词表”一词本身决定。

| 维度 | 决策 | 计数/语义 |
|---|---|---|
| 实体 | 12个 | 首个有限世界的实体集合，不据此主张开放实体泛化 |
| 属性 | 3个：颜色、位置、归属 | 每属性4个类型合法值，值域按属性声明 |
| 版本 | 每个(entity,attribute)最多3个合法版本 | 包含初始v0及两次更正v1/v2；允许不足3版，用于无历史/未知案例 |
| 评分任务 | 全部应答轮 | 类别/分母由任务声明，不由预测act/解析成功决定 |

12×3有36个实体属性槽；全部槽都出现3版时有108条槽版本记录。432是实体×属性×值选项×版本位置的组合格数，**不是样本/episode数量或同时活跃事实数**，也不要求穷举所有历史轨迹。

具体12个名称、每属性的4值、模板、样本数与split由后续审计登记；形状已确认，不再以这些细节未定重复搁置本轮五项方向。原子实体/属性/值可在train中覆盖，独立确认重点是新组合、历史与措辞；更强未见实体主张须另声明。family和成对反事实必须按事实历史整体划分，不能把同一历史改写分散到train/确认面。

首版采用顺序发生的合法版本链，不引入追溯生效/并发冲突解析；仍保留valid time与获知时间字段以便扩展。相邻版本允许同值，previous是版本关系，不做“不同值去重”；需要观察答案翻转的对照另选不同值，并以revision/source检查同值不同版本。

2026-10-30 只读世界审计已登记：[世界审计报告](../../reports/taiji_native_core_r1_world_audit_20261030.json)。12 个实体、3 属性、每属性 4 值、48 个属性模板 + 2 个延迟模板；36 个实体属性槽、108 条槽版本记录、432 个值位置组合格。审计提出 **360 条不同值双版本历史的样本计划**（216 train / 72 query / 72 holdout，4 episodes/history，1,440 episodes / 11,520 turns，字节估算 0.80–1.11 MB）；这只是可行性提案，精确覆盖/split/成本留重构-2/4，未生成数据、未调用模型。

2026-10-30 公开语义 fixture 已建并通过审计：[fixture](../../plans/manifests/native_core_r1_public_fixture_v1.json)、[审计报告](../../reports/taiji_native_core_r1_public_fixture_audit_20261030.json)。17 案覆盖同 query 不同当前值、同当前值不同历史、同 query 文本不同 time_intent、更正、三版本、同值版本、干扰、未知、未知历史、歧义、否定和非应答；三属性全覆盖，新实体与旧 E1 无交集，全部 `sealed=false`，逐案 target 首字节/目标切片/固定分母检查通过。fixture 是公开合成面，不是训练数据，未调用模型。

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

第 3/5 轮 query 文本相同，答案必须随当前值变化；第5/7轮使用同一store的当前/历史版本；该翻转fixture选择不同值，因此答案必须不同。同值更正另设版本身份检查，不强求文本不同。若旧值不存在，目标为明确的“不知道/无法判断”，不得猜值。

### 3.4 时间意图

- 首版time_intent只允许current/previous/unknown；at_revision不在首版。unknown表示输入时间要求不可确定，不能由宿主根据正确答案是否存在来赋值。缺少合法事实/历史也可输出未知，此时time_intent仍可为current/previous，输入意图与目标可回答性必须分开。
- `time_intent` 必须对读取器可见；它可以出现在 query 文本（“现在/之前”）或结构化条件中，但不能由宿主直接给出答案。
- 同一query文本可配current/previous显式条件，所选版本不同时须分别读对应版本；值不同时要求答案不同。若有结构化time_intent输入，应对所有比较模型同样可见并披露其为条件化任务，不能据此宣称模型已从自然语言学会时间意图。
- `previous`指同一实体/属性当前合法版本的直接前驱；不是前一个不同值，也不跳过同值更正。无前驱则返回明确未知，不退回current。
- 更正保留旧版本，不删除；当前值由 `valid_from/valid_to` 和 revision 决定。
- 禁止用 `store.current` 统一回答所有查询；旧值路径必须实际读取历史。

### 3.5 未知、歧义与否定

- 未知实体/属性：合法候选为空；目标为“不知道/无法判断”，计为 `abstain_correct` 或 `answer_exact` 的一类，不按模型 act 筛除。
- 歧义/否定：不产生可写事实；若已有当前值，查询仍按既有版本回答；不得把否定句误写成新事实。
- 任何“猜一个合法值”的未知回答计为错误，不能因值属于该属性的合法取值集合就算合理。

## 4. 内容状态、读取与生成

### 4.1 准入与辅助解析

- 原始输入 + 来源先进入可学习编码；结构化 span/act 是**辅助视图**，用于校验、审计和辅助目标，不是唯一准入。
- 解析失败/低置信内容可以带 `status=unresolved/ambiguous` 进入 store；它保留原始 span、来源、置信和 `encoder_revision`，但不得被当作已确认事实写入当前值。
- 任务回答 loss 必须能到达至少一个声明的内容编码/读取/表达模块；硬解析头的辅助 CE 单列，不能替代任务信用。
- 来源、权限、owner/scope/session 是硬边界，不由软学习绕过。

未解析候选不得因`unresolved`状态而一律失去内容读取/任务信用，否则会恢复成原硬门；读取仍受owner/scope/来源权限约束，采信为可信事实或外部动作依据须另过相应验证。否定/歧义可作为带状态的原始证据留存，不自动写成肯定事实。

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
- 每个unit只计一次；解析失败、拒答、错误、空输出均不剔除。解析失败造成漏答/错答/无效输出，主回答计错；辅助硬解析失败但新核心正确回答时，主回答按正确计，解析诊断计错，不能用辅助头成败替代答案正确性；
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
- 新核心必须允许合法回答任务信用同时作用于声明的可学习内容读取组与内容条件表达组。只让分类/输出bias头有梯度不满足读取组要求；首版编码/写入若冻结须单列，不能声称其获得任务塑形。信用机制不限定autograd，但需实际接收者/参数变化、禁写组及对照验证；机制净收益归后续正式评价。

## 8. 比较设计

| 对象 | 作用 | 不能得出的结论 |
|---|---|---|
| 修正语义的旧图基线 | 公平评价硬解析/current-only/固定 bias 的实际代价 | 不能只修新图而让旧基线背负已知错目标 |
| 普通端到端学习基线 | 同数据/权限/资源下区分机制是否值得复杂度 | 不借不同语料/参数/外部模型偷换研究对象 |
| 新共同内容核心 | 主架构包的行为/保持/资源比较 | 多部分一起变化时只称架构包收益 |
| 新图内 credit on/stop | 只切指定信用 | 不把架构包差异混成信用单因素 |

- 本轮确认三类架构比较组：修正旧图、普通端到端基线、新共同核心。前两是基线，第三是候选；credit on/stop为新图内部机制控制，不是第四类架构比较。三组共用数据分区、父保持集、输入/状态权限、评分器和资源口径；不得用旧已消费sealed面做最终门。
- 同资源以预先冻结的训练/前处理/恢复/评价总成本边界和参数上限口径比较，实际曝光/更新数分别报告；不能只同更新数却忽略每步成本，也不能给新图额外预训练或削弱基线。具体数值在重构-4定稿。
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
- [ ] 任务 loss/辅助 loss/禁写组矩阵完整；任务信用能作用于可学习读取及内容条件表达两组。
- [ ] 修正旧图、普通基线、新核心同资源比较方案已写；credit on/stop 单独定义。
- [x] 公开 fixture 与 sealed 分离；未读新 sealed；训练前 checkpoint/恢复门已定义（17 案审计通过，见报告）。
- [x] 首版五项设计边界已由本轮用户授权拍板确认；不代表其余语义检查已通过。
- [ ] 其余合同矛盾、公开fixture和数据/接口审计闭合后才转重构-2；实现/训练仍依对应工作包与预算准入。

## 11. 明确不做

- 不把本件当作训练批准、能力通过或预算承诺。
- 不用 parser 覆盖率、train loss、恢复一致或输出变化冒充回答/状态因果收益。
- 不重读 N175/N176 未消费面、N177.6/F1/F1b/N178 旧 sealed 面做最终门。
- 不在同一配方上自动加 epoch/seed/剂量；不把旧 F/N178 费用清零。
- 不一次推倒 TSK、语言、记忆、世界和全部训练机制；按重构-2 的候选/基线比较选择。

## 12. 本轮决策边界

本轮仅确认世界形状、时间语义、固定评分、候选内容准入与三组比较/任务信用要求，并同步计划状态。2026-10-30 已运行只读世界审计（名单/模板/360-history 样本计划）和公开 fixture 审计（17 案通过），未生成数据、未调用模型；精确分区/覆盖、其余合同矛盾与接口审计、模型与预算仍待各包实施/审计。
