# PLAN-R2-DIALOGUE-06：全轮自然对话预训练与助手回复校准

**状态：已冻结并按用户授予的执行计划批准权准入。** 本批直接修复“从微小任务模型续训”的错误起点；不使用外部预训练权重，不重用已消费final family。

## 目标与证据边界

现有H3.7B父档实际由Taiji配置新建，只有278,010个active scalar参数，并在12个任务episode上训练两轮；它不是37亿参数或广泛预训练语言模型。8-epoch assistant-only OASST2续训使end-marker率提高，但预定样例0/12可读。下一批从新的Taiji随机初始化开始，让模型先从真实中文对话中学习双方轮次与语言形态，再对assistant回复做短校准。该批是“容量＋全轮语料＋目标配方”比较包，结果不单独归因任何一个变量，也不自动晋升。

## 固定来源和划分

- 唯一来源仍为本地固定OASST2 ready export：source revision `179dd21fc55192153d94adb0e0ce8f69e222bf75`，SHA-256 `a9f240c4c77aa1378364f70d37e753c07ba284e247b019d700e1947a0e5da751`。
- 只使用既定tree-hash train split中通过现行审核、安全和PII规则的所有消息：3,762条（assistant 2,244、prompter 1,518）、508个tree family、原始正文1,780,881个UTF-8字节。每条消息按其角色作为target；prefix带正确的user/assistant角色标记、最多3个既往turn pair、最多4,096字节。target在UTF-8边界完整保留，并追加原有`\n<|end|>\n`分隔符。完整预期target面为1,814,739字节/epoch、8个epoch共14,517,912次pretraining字节更新。
- 全轮pretraining排除全部dev和final family。assistant-only校准复用原525条高质量train episode、固定1个epoch、预期113,209次target字节更新。两阶段预计共14,631,121次target字节更新。
- 39个未查询dev family（62条现有episode）只保留作一次性自由生成审阅面：每family固定选最小episode id，共39条输入；final的82个family已全部消费，永不重用。生成时只传prompt/history，不使用reference response、exact match或teacher-forced评分。

## 模型与训练合同

- Fresh Taiji初始化：`TaijiConfig.capacity_profile(1_000_000, seed=20261022, additional_predictive_readouts=1, response_plan_width=48)`，打开现有factorized response-plan readout；不加载H3.7B父档或外部LLM权重。预检时实际参数计数须≤1,000,000并固定具体config/hash。
- 新增独立的role-aware全轮byte预训练器，复用Taiji本地逐字节学习；对每个train消息从它所在family的安全祖先链构造prefix，在该消息的角色标记后逐字节预测。目标面是两种speaker的自然消息，不改动现有`LanguageEpisode`序列化或历史checkpoint合同。
- 全轮pretraining固定8个完整epoch；随后仅在原525条assistant目标上做1个完整校准epoch。无早停挑点、无追加epoch、无多配置扫参。校准阶段复用现有byte-aligned response-plan目标几何。
- 训练前fresh-process零bundle保存/恢复必须通过；两阶段之间full bundle保存/恢复通过；assistant校准zero/final bundle均独立恢复。任何checkpoint门失败则0正式更新并停止。final保存需包含模型结构、RNG、训练游标和完整history。
- 总CPU≤240,000秒，含数据构建、checkpoint预检/保存恢复、训练和78次自由生成。到上限即停止并保留partial；不以CPU/损失改写配方、不重试、不晋升。

## 一次性自由生成检查

- 同一39条未查dev-family输入上比较两个端点：既有R2-DIALOGUE-03R1的8-epoch多轮candidate与本批新candidate，共78次确定性自由生成。history输入保持原episode形式；使用UTF-8合法byte mask和既有最大生成长度512。
- 机器摘要只记录UTF-8、replacement、end-marker、输出唯一数和重复4-gram等格式统计。人工并排审阅12个预选family，按可读中文、直接回应当前用户、历史连贯、重复情况记录定性判断；不看reference response，不作逐字对错判断。
- `can_promote=false`。任何改善只支持此全轮预训练＋校准包在未消费dev-family上的局部表现，不作外推或因果分账。

## 产物

- 新role-aware预训练器、OASST2 train-family builder及契约测试，分别放在`taiji/`、`scripts/data_prep/`、`tests/taiji_native/`。
- checkpoint只放`data/checkpoints/r2_daily_dialogue_oasst2_full_turn_20261021/`。
- 预检、训练progress、generation与结果放`reports/`；中文人工审阅和结案放`plans/reference/`。

### 停止边界

如果最终候选仍不能在12个样例中形成至少可读、贴题的中文回复，按负结果结案；不再对这批模型叠加epoch，下一步必须重新审视原生语言架构与目标。不因end-marker或格式改善单独宣称日常对话能力成立。
