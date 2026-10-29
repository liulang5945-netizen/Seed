# N177 E决策：先验证状态因果消费，再验证任务信用收益

日期：2026-10-29，Asia/Shanghai。状态：**E1合同已冻结，E2a课程/manifest已生成；E2b/E3与F训练尚未执行。** 当前调度由[03](../active/roadmap/03_CURRENT_EXECUTION.md)维护，阶段依赖见[10](../active/roadmap/10_CONTINUOUS_CAPABILITY_EXECUTION_PLAN.md)。本件是后继候选设计，不修改SPEC-N1-177.5 v2或旧结果。

## 1. A–D复核及继承范围

| 阶段 | 已确认的交付 | 仍须保留的边界 |
|---|---|---|
| A | 15臂各2560日程步完成，合计38400；run_state为evaluation_complete | No-update步数不等于optimizer调用次数 |
| B | 评价准备及梯度边界可用；逐loss合成测试覆盖五模式分支/语言mask | 输入追加/初始化和高置信解析为替身；真实cache/Adapter链、producer更新mask、optimizer/cursor隔离未全部交付 |
| C | 一次正式评价，15臂齐全；十项绝对门、两条三seed学习门与原F0保持线报告通过 | 当前只读verifier不是完整科学验收；终点独立进程恢复/同输入parse-answer复现尚未找到耐久回执 |
| D | 已发布联合诊断及分项读数，可进入E方向决策 | 以下归因勘误适用于后继决策，不回写原机器报告 |

本轮显式指定评价文件执行只读verifier，返回publishable、15臂、attempts=1；独立检查[结案清单](../../reports/taiji_n1_plan177_6_t1_closeout_manifest_20261029.json)所钉pilot/evaluation的字节与SHA一致。未重跑评分、未加载训练checkpoint、未消费新封存题面。

依据：[训练报告](../../reports/taiji_n1_plan177_6_t1_pilot_20261027.json)、[终评报告](../../reports/taiji_n1_plan177_6_t1_evaluation_20261029.json)、[原D诊断](../../reports/taiji_n1_plan177_6_t1_joint_diagnosis_20261029.json)、[SPEC](SPEC-N1-177_5_T1_F0_F1_pilot_20261027.md)。原包结论限定固定合成F0/F1 T1，不补发开放语言、T2/T3或L3/L4通过。

## 2. 改变E优先级的读数与归因勘误

| 聚合读数 | 对E的意义 |
|---|---|
| Correct查询exact约98.958%，更正后规范state exact=100% | 当前固定切片已有正向结果；不能继续预判联合训练必败 |
| Parser-permuted查询exact约98.351%，规范state及无关事实保持均为0 | 回答仍可通过原始证据、hidden或其他合法路径获得事实；现文本端点很难区分规范world是否正确，不能断言模型完全不使用事实 |
| Correct禁commit后查询均值约75.781%，启用约98.958% | commit路径确有约23.18个百分点查询收益；不是答案完全无贡献 |
| 复合state-query差约60.8–62.1个百分点 | 原分数一半直接来自state exact；禁写后该项归零。差值不能全部解释为语言依赖world的收益 |

勘误与限制：

- `state_writes=False`跳过整个`commit_to_joint_state`，同时改变world、workspace及证据选择。现消融测的是commit路径总效应，不是纯world值读出的单因素效果。
- Parser-permuted约0.492、owner-frozen约0.387是复合分，不是“parser贡献一半”或“其他模块贡献三分之一”的比例分解。不同臂也改变更新/写入条件，不能相减后直接命名模块贡献率。
- 原SPEC固定三个producer seeds，所有臂共享同一V4父。没有“必须再做独立初始化确认”的条款；新父/新seed复验可以另立研究，但不是补原门。D中的250k CPU估算也不作为可用预算：按其列出的三臂单位成本乘2560已约292,352 CPU秒，且还未含额外开销。
- 原F0保持是zero到final的原规则对照，不是与一个已掌握F0的强父模型比较后证明抗遗忘。新候选须另定义其适用的父能力保持。

## 3. 选项与唯一决策

| 选项 | 能回答什么 | 代价及局限 | 本轮决定 |
|---|---|---|---|
| 旧切片新初始化确认 | 旧切片结果对初始化是否稳健 | 需新增训练；近饱和回答端点仍不能区分状态消费或信用收益 | 延后为稳健性支路 |
| 旧切片直接接通producer信用 | 可证明一条新梯度路径存在 | 旧回答近满分，零/小差值难解释；也不能证明规范状态成为回答依据 | 不作为下一正式实验 |
| 状态依赖新端点＋显式内容消费，再比较信用开/关 | 能识别回答是否受必要状态差异影响，再隔离任务信用的收益 | 需要新数据/评分边界及版本化接口，工程量更大 | **采用此方向** |

目标是发展同一模型中可被任务反馈塑形的持久内容表征与查询消费，而非只追求旧问答分再涨一点。任务判别力是新信用实验的前提；两个信用臂必须共用同一新任务和消费接口，避免把任务、读出与梯度三种变化混成一次因果结论。

## 4. E候选的任务、信息路径与信用设计

### 4.1 新任务端点

使用新独立episode/manifest：同一个查询文本可对应不同随机事实，包含延迟查询、相关更正、无关对象干扰与旧值竞争。平衡实体、属性、值、模板和顺序，避免由查询文本或固定共现直接猜答案。新训练/开发/确认按episode或事实图划分，不用旧holdout挑题、训练或定阈值。

先用公开合成fixture验证接口和成对评分，不生成或消费新sealed确认面。答案由独立任务规则从合法输入事实确定，运行时不注入gold WorldObject、gold span或答案。

### 4.2 必须实际消费的内容接口

`raw input → 模型producer → 带来源的对象/属性/值与持久内容特征 → query条件选择 → 同一内容驱动语言输出`。

新接口须明确decoder能读取的证据面。主端点不能让当前查询文本或可直接重放的完整历史答案代替被声称的状态消费；历史raw bytes可保留作来源审计，但在主端点对decoder的访问必须明确限制。另设raw-history-only路径作描述性对照，不把它的能力冒充状态消费。

持久状态的离散值、连续特征、input/source/span及版本必须一致。相关交换使用合法成对输入产生的候选状态或受控克隆，并保留有效来源；不能靠破坏hash或revision、让校验报错来冒充因果响应。独立保留规范状态正确性，不用“state项加进总分”替代语言消费证据。

query内容选择及语言解码须由Taiji自有的可学习消费者完成；宿主只负责输入、权限和来源校验，不能把WorldObject字段套模板输出或注入金标选择。若采用可复制的值字节，必须披露模型的寻址/选择责任及相应对照，精确复制本身不代证学习型选择。

### 4.3 公平信用控制

在同一新消费接口中，task-credit-on与stop-gradient共享父参数、初始前向结构/数值、数据曝光、辅助目标、允许参数组及资源口径，只切换指定回答任务信用。训练后轨迹分化是被测效果，不要求继续前向相同。

训练专用连续feature可保留图；推理/持久化仍使用source-bound状态合同。离散span/act选择的信用另行声明，不能由连续feature获得梯度推断它们已受任务纠正。无需强求推理接口处处可微；但若声称回答误差可塑形producer，就必须交付真实参数责任及行为收益。

不保存autograd图到checkpoint；新进程从合法状态/来源重建训练所需计算。先确定episode/更新边界和截断语义，再验证optimizer/RNG/cursor与跨边界恢复。旧15臂、父checkpoint及旧正式评价均保持独立身份。

## 5. E前置补证与实施分解

| 子步 | 交付 | 边界 |
|---|---|---|
| E0 原证据补齐 | 终点fresh-process恢复/parse-answer复现回执；B真实输入/有效mask及状态隔离矩阵 | 用train或新合成固定输入只读核验，不重训、不重评、不再次读取holdout；缺项如实记未知 |
| E1 任务与消费合同 | 已冻结：[E1状态依赖查询与内容消费合同](SPEC-N1-177_E1_state_consumption_contract_20261029.md)；新端点、允许证据面、来源/生命周期、独立评分及控制矩阵 | 只做公开合成/静态检查，不用旧sealed迭代方案 |
| E2 opt-in候选实现 | 持久内容/查询消费及连续任务信用接口，on/off同图控制 | 不覆盖旧默认；先证明相关干预改变答案分布、无关干预保持与切断/救援 |
| E3 零步工程验收 | 逐loss责任、前向等价、禁止更新组、fresh-process checkpoint/optimizer/RNG/cursor恢复 | 零optimizer正式训练；工程敏感性不称学会任务 |

E0可与E2实现并行，但旧资产恢复及新候选的必要恢复门未闭合前，不能带缺项进入F正式训练。若完成补证需超出已有资源/读取权限，先登记具体输入、一次成本及有限出口，不自动重开旧实验。

2026-10-29 E2b-1 起点探针：N177.6 correct/seed0 终点producer在已知实体“钥匙”上正确抽出entity/attribute/value并写入world object/event；在新实体“笔”上只抽到attribute/value、entity缺失，world objects=0。该事实只限定零步fixture与F训练起点，不改E1课程或判据：E2b-2先用已知实体/已知颜色fixture，完整8实体词表留F训练；新实体实体抽取缺口单列为producer责任，不冒充状态消费失败。证据见[已知实体](../../reports/taiji_n1_plan177_e1_state_path_known_entity_20261029.json)与[新实体](../../reports/taiji_n1_plan177_e1_state_path_20261029.json)。

2026-10-29 E2b-2 读出结果：producer的mean特征、value-span mean特征与value-span逐字节特征在8色上均只有rank 2，任意两色最低余弦约0.93/0.93/0.798；闭式读出仅25%原型准确，8色不可分。改用可训练value-token嵌入fixture（one-hot初始化）后，原型8/8、相关交换蓝→绿、无关交换保持、切断无候选、救援恢复；query-only路径只有1个binding/1个observation且无事实字节。判决：**解码器读出与选择接口已通；producer连续编码本身不保留值身份，不能直接进F**。下一步E2b-3做value-preserving可学习特征（value-token嵌入或producer值身份辅助目标）并重跑producer特征读数；fixture通过不冒充producer特征通过。证据见[mean特征](../../reports/taiji_n1_plan177_e1_value_readout_20261029.json)、[value-span mean](../../reports/taiji_n1_plan177_e1_value_readout_value_feature_20261029.json)、[value-span逐字节](../../reports/taiji_n1_plan177_e1_value_readout_value_bytes_20261029.json)与[value-token fixture](../../reports/taiji_n1_plan177_e1_value_readout_value_token_20261029.json)。

2026-10-29 E2b-3 组件恢复预检：把value-token encoder、query selector、closed-form readout、active store与替换事实打成bundle；独立进程恢复后baseline/相关/无关/切断/救援的选中值与logits完全一致（max logit diff 0.0），bundle/source SHA可核验，optimizer更新0、sealed读取0。但copy-only与主路径一致，说明当前value-token一热特征仍是**离散值复制的下界**；组件级恢复通过不等于完整E3，也不证明学习型选择。下一步F候选改为**字节级value encoder + answer loss + copy-only消融**：主路径不得读原始值字节，只读encoder输出；若copy-only仍能替代，则特征未被学习。E0与完整E3（producer/V4/decoder）闭合前不启动F。证据见[组件恢复](../../reports/taiji_n1_plan177_e1_value_feature_restore_20261029.json)。

2026-10-29 F合同冻结：F1改为4臂×3 seeds（main_credit_on/main_credit_stop/copy_only/no_state），主路径只读value byte encoder输出，copy-only/no_state为下界，answer loss是唯一切换的任务信用；先F0 train-only标定（≤2,000 CPU）冻结更新数/阈值/预算，再过E0/完整E3，一次有界F1。合同见[字节级值编码器F合同](SPEC-N1-177_F_value_encoder_20261029.md)；本轮未启动F1、未读holdout。

2026-10-29 F0标定结果：byte-level value encoder（值span UTF-8字节→GRU 64维）+线性值读出，在64条E1 train事实上训练200步；train accuracy **1.0**、no-state **0.078**、copy-only **1.0**；zero与final均新进程恢复通过，CPU 118.25/墙钟50.95，未读query/holdout。该结果只证明训练侧值身份可分，copy-only仍是下界，不能证明holdout、完整中文回答或学习收益；F1未启动，仍受E0与完整E3前置约束。证据见[F0报告](../../reports/taiji_n1_plan177_f_value_encoder_f0_20261029.json)。

2026-10-29 完整E3 v6：producer/V4、F0 byte encoder、selector/store、readout、feature→byte bias 同bundle，父/子进程各跑同一更正事实与查询。parse 的 act/status/entity/value、span 角色与字节范围、confidence/ambiguity、semantic features、selected feature、store digest、base 与 conditioned 下一字节 logits 全部一致（最大差0.0），query-only 1 binding/1 observation 且无事实字节；`optimizer_updates=0`、`sealed_text_read=0`。唯一不一致是 span 的 percept/assembly lineage ID（`parse_sha256`/`span_lineage_digest` 不同），因此状态为 **`full_e3_content_passed_lineage_gap`**，严格E3尚未通过。下一步先修 adapter/感知 lineage ID 的跨进程确定性并重跑严格E3；E0并行，F1不启动。证据见[完整E3 v6](../../reports/taiji_n1_plan177_f_full_e3_20261029_v6.json)。

2026-10-29 严格E3修复与通过：逐字段定位发现 percept/assembly ID 本身相同，差异来自 span 在 parse payload 中的顺序；`JointSemanticSpanV1.ROLES` 是 frozenset，`decoded` 字典迭代顺序跨进程不稳定。已把 spans 按 `(start_byte, end_byte, role)` 排序。v11/v12 连续两次严格通过：`parse_digest_equal=true`、`span_lineage_equal=true`、base adapter digest 相等、base/conditioned logits 最大差0.0；选中蓝色、相关交换 delta 1.9756、query-only 无事实字节；`optimizer_updates=0`、`sealed_text_read=0`。F1仅剩E0前置；N177.6历史报告/checkpoint不重算，旧源码保留在git历史，E0需用历史源码或登记来源漂移。证据见[完整E3 v11](../../reports/taiji_n1_plan177_f_full_e3_20261029_v11.json)与[v12](../../reports/taiji_n1_plan177_f_full_e3_20261029_v12.json)。

## 6. 验收与有限出口

| 判据 | E阶段要求 | F阶段才裁决的内容 |
|---|---|---|
| 信息判别 | 同query的合法相关事实差异能进入实际内容消费；无关差异不覆盖所需内容 | 预注册未见实例上的答案正确/相关翻转/无关保持 |
| 消费因果 | 固定参数的切断、相关/无关交换及合法救援分别报告；不把报错计作成功 | 学习后稳定依赖所需状态，而非旁路重构 |
| 信用责任 | 回答loss到指定producer连续参数的实际梯度；辅助/离散目标单列；对照仅切信用 | 新任务净收益及旧能力保持，而非仅梯度非零 |
| 工程恢复 | 同父前向等价、无越权写入、完整保存/独立恢复及下一个输入一致 | 有更新训练后的保持/恢复及真实资源 |

E到合同、公开合成因果/信用检查及零步恢复齐全即可有限结案；不足就回具体消费/测量/恢复缺口，不靠增加训练解决接线。若新端点仍让无状态路径近饱和，先结束该端点设计，禁止用它烧正式信用比较预算。

F另冻样本/种子、数值主门、保持、恢复、硬资源上限及跨包累计停止线。主行为分与state正确性分开，必须有能判定失败的相关翻转/保持及对照，不以旧复合分为新主证。独立初始化的稳健性可在F设计中明确，但不能声称已经获批；D的250k报价不沿用。本件不授权或启动训练，也不默认更换产品模型。

本轮验证：原报告只读验收与清单哈希核对通过；逐loss基准、项目身份、卫生及预注册措辞14项本机检查通过，Ruff/Black及八份计划文件的导航检查通过。该验证不包含终点恢复回执补做、新任务实现或远端CI结果。
