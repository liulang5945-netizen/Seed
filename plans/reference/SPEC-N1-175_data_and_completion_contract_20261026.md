# N1-175 数据与有限结案合同 v1

日期：2026-10-26，持续更新。seed169/170/171正式训练均完成epoch 1–15/40，epoch01–15检查点均通过独立进程恢复与digest往返；当前续训继续运行。承接N1-174家族结案及[N175元数据审计](PLAN-N1-175_metadata_feasibility_audit_20261025.md)。冻结训练包为1148题/517family、3 seeds×40 epochs；64-family留出正文读取为0。训练损失只作teacher-forced拟合信号，不代表能力提升。

## 方向与本轮范围

保留共享条件语言主链V2，继续检验可靠数据覆盖能否改善自由生成和按题回答。N174仅1/3 seed训练滚动达门、未见绑定未证明，尚不足以推翻整套架构，也不足以认定只需更多数据。驱动力、自主性和生命页不进入本轮训练变量。

本轮采用一问一个优选监督目标：只选来源rank=0；不以低排名同题回复填充数量，不在rank=0不合格时自动降级换答案。排名不是事实正确性证明，结构筛选不是语义审核。按合格实际规模报告，不以2000/5000为硬门。正文规范化去重不宣称语义去重。

## 已冻结的数据选择

- 来源沿用既有冻结OASST2与旧episode文件；SHA在每次运行前核验。
- 旧非train的121个root-family整体排除；旧train family可进入训练，不能进入新留出。
- 从rank=0候选涉及、且旧全部episode均未涉及的family中，以`sha256("n175-root-holdout-v1-20261026:" + root_id)`升序选64个family；盐值在正文读取前固定。
- 同一family全部后代、上下文、兄弟答案不跨split。每个prompt只能有一个rank=0目标，多目标冲突直接报错；未知旧ID、断链、循环均不静默忽略。
- [划分回执](../../reports/taiji_n1_plan175_metadata_partition_20261026.json)：1258条train候选，64个留出family内135条候选回复。后者不是135个独立验收单位。划分只读元数据，没有读留出正文。
- 正式评价每family仅选1个题，题ID按上述盐值加prompt_id的SHA排序预选第一条；不看生成质量换题。若质量检查不适合评价，登记不可判，不补抽。64个原定family保留完整分母与缺失说明。

## 正文与长度检查结果

[训练侧检查](../../reports/taiji_n1_plan175_train_quality_20261026.json)仅解析train候选及其祖先2965个文本节点；新留出和旧保护family正文解析为0。要求完整交替角色、中文来源标签、未删除、非空、无异常控制字符；保留全部历史上下文。这里只是结构检查，不证明正文确为中文、事实正确或答案贴题。

完整target≤512字节的早期可行性场景有537条/354family；这只是长度探针，不是最终数据合同。V3实际完整上下文编码并按带结束标记的target≤2048、序列化prompt≤4096筛选，得到1148题、517个family；其target总量822311字节/epoch、prompt总量788550字节/epoch，target最长2043、prompt最长4078字节。分布为target≤128/512/1024/2048：204/532/768/1148题。详细清单与冻结SHA见[train corpus回执](../../reports/taiji_n1_plan175_v3_train_corpus_v2_20261026.json)。正文仅解析训练family；留出正文仍为0。

完整回答的变长训练准备已完成并进入正式V3训练：按target字节分桶、保留完整回答边界，训练合同固定到40 epochs/seed。原128字节前缀只保留为历史可比机制指标，不代替完整目标训练和评价；不因中间损失或训练内读数追加epoch、改样本或扩大长度。

较早的训练侧候选target统计为1,078,103字节；≤128/512/1024/2048/4096字节分别248/572/814/1189/1245条、最长15822字节。该统计是质量去重前的历史工程探针，不是本次最终样本量；执行清单以1148题的冻结corpus回执为准。

## 发现的主链边界与下一工程包

V2不支持靠配置直接完成长回答：`taiji/task_state_v2.py`的`MAX_EVIDENCE_BYTES=512`，`task_core_v2.py`按此裁保最后512个逐字节状态，`conditional_language_v2.py::attention_weights`又用同一position表索引输出步，step≥512直接报错。只提高`max_generation_bytes`仍会失败；4096字节上下文也不能等同全部拥有逐字节寻址能力。块workspace仍提供摘要路径，因此不把逐字节裁剪表述成所有较早信息都不可达。

采用独立候选V3工程包，分离输入证据窗口与输出位置，保留旧V2与checkpoint原语义；不直接修改V2全局上限来伪装兼容。候选设计：输入逐字节窗口4096、输出位置2048分别配置；显式记录较早上下文进入摘要/窗口的覆盖；独立版本化状态与checkpoint，拒绝隐式加载V2。位置参数初始化和父模型迁移必须登记；相同预算对照不得把数据、窗口、初始化三者收益混称为数据规模收益。

候选代码已落在`taiji/conditional_language_v3.py`：新TaskStateV3上限4096、独立input/output position表、V3格式checkpoint、train-only teacher forcing逐样本反传。V2模块与checkpoint格式未改。V3定向5项、V2回归9项通过；覆盖边界、V2/V3互拒、新进程续训一致性、分样本梯度等价。

工程验收使用合成字节，不读取新留出：验证输出第511/512步边界、2048上限拒绝、4096输入窗口裁剪与引用一致，且旧V2 checkpoint拒绝进入V3、V3 checkpoint拒绝进入V2；变长batch按总目标字节归一化，逐样本反传累计梯度、整batch只裁剪/更新一次，检查其梯度与原整batch实现一致。状态、位置、优化器、RNG与数据cursor写入版本化checkpoint，先过新进程恢复和同批续训一致性，再进行train-only标定。若4096证据成本不可接受，以实测资源和机制覆盖作裁决，不悄悄退回短前缀而仍称完整任务训练。

V2→V3迁移函数按参数名映射了全部共享权重与AdamW状态；V2位置表前512行复制到V3两表，扩展行按run seed确定性初始化、扩展Adam矩清零。三个N174C TF seed的迁移父checkpoint已保存且新进程恢复通过，见[迁移回执](../../reports/taiji_n1_plan175_v3_parent_migration_20261026.json)。N174父checkpoint不变。

train-only真实batch更新测得3700目标字节/8.421875 process-CPU秒=439.33目标字节/CPU秒，第二个与首批题目及family均不重合；该数字是小样本量级标定，不保证长程线性。首份标定回执把累计历史曝光当作本批分子，吞吐率错误；其持久结果保留，勿用于成本估算，正确分子/计时见[独立更新成本回执](../../reports/taiji_n1_plan175_v3_update_cost_20261026.json)。据实际corpus计算40 epochs为32892440目标字节/seed，初估74800 CPU秒/seed、3 seed约225000；2倍余量约450000 process-CPU秒。输入context和长序列成本可能使实耗偏离，训练期间以进度账本为准。

这属于条件语言候选的局部主链升级，当前证据不要求推翻整个认知架构。N175从“纯数据规模对照”修订为“V3完整回答训练”，结构/恢复验收与train-only成本标定已完成，正式三seed TF训练当前到15/40 epochs；按冻结40 epochs、batch4及真实父模型迁移继续。每seed CPU保护上限300000、全队列上限900000 process-CPU秒，包含预计费用和大余量，均低于既有每臂授权；若触顶则按合同保留partial并停止，不自动提额。训练完成后再读取64个新留出family，不在训练中查询。

## 工程与运行顺序

1. 候选划分和train结构审计：本轮已完成，代码测试覆盖确定性、旧家族排除、同题多目标拒绝、断链与长度完整性。
2. V3状态/模型/逐样本梯度与checkpoint：已完成并经V3/V2定向回归及新进程恢复验证；三个N174 TF seed真实父checkpoint迁移完成。
3. Train-only完整答复标定：已完成；corpus1148题/517family，3700 byte update为439.33 target byte/process-CPU秒。第一份统计回执率值错误，按独立第二批实测为准。
4. 现冻结训练包：corpus 1148条、517 families、manifest SHA见回执；V3输入4096/输出2048；父模型seed169/170/171、V2→V3按名迁移；tf单臂、batch4、按完整target字节分桶，40 epochs/seed、每epoch耐久checkpoint。CPU guard 300000/seed、合计900000 process-CPU秒。旧3–8万CPU撤销。当前正式训练已完成15/40 epochs/seed，短标定仅2个单步batch，不计入正式训练结果。
5. 新留出仅用于冻结后的评价，不参与梯度、超参、答案筛选或训练长度选择。正式输出后不因负结果追加epoch、重抽seed、换题或重跑。

## 收官条件与结论边界

本轮以固定工作包结束，不以“直到模型学会”为运行条件。

- 工程完成：所有计划run达到固定终点，checkpoint可独立恢复，费用/失败/partial全计入；工程失败可在题面消费前按同一合同修复，实际读题后不得通过换题或重跑美化结果。
- 滚动门R：训练侧冻结样本的自由前缀匹配中位数≥0.50，至少2/3 seed达成，保留N174机制口径；另报告完整回答可读性与终止行为。R未达即`INCONCLUSIVE_ROLLOUT`，不能据此宣布架构泛化失败。
- 语义门S：对64个预选family的回答盲化判读。0=不可读/无关/只套话；1=部分相关但核心问题未解决或有关键事实错误；2=直接处理问题且核心回答正确；无法核实的事实记不可判。每seed正确（2分）至少16/64，且相对确定性错配prompt对照至少多8/64；至少2/3 seed同时达到R和S才记本轮候选获得有限收益。错配只在不同family间固定轮转，输出匿名，判读分歧保留给用户裁决；不可判单列，不算成功。
- 逐字前缀、own/错配TF和A/B历史对照均保留，语义2分不要求逐字复现参考答案。不得悄悄用新语义门改判N174。
- 终点全部达成但R未达：有限不可判结案；R达成但S未达：该固定数据/剂量下未见可靠贴题收益，进入条件消费/学习信用诊断，而非继续无限扩数据；有效题不足以判断：数据/评价不可判结案，不补抽重新求过门。
- 即使通过，仅支持本数据分布与本候选的有限收益，不等于完整N主线收官、默认入口替换、成长或自主性能力通过。推广仍按N整体验收。

语义门为本次预注册的新候选标准；数据质量、变长工程与成本已冻结并通过前置检查。正式读留出仍必须等三seed全部到40 epochs、最终checkpoint lineage验证通过。当前留出正文读取0，不制造已获能力的表述。


正式训练入口为[`run_taiji_n1_plan175_v3_scale.py`](../../scripts/training/run_taiji_n1_plan175_v3_scale.py)，按seed单独进程运行，续训从最近一个epoch checkpoint恢复，progress账本记录当前batch。三seed均完成40 epochs、最终checkpoint与数据/源码来源核验通过后，再按本合同固定R/S门生成64个留出family回答；训练期间不解析其正文。


盲化评价生成器[`evaluate_taiji_n1_plan175_v3_holdout.py`](../../scripts/training/evaluate_taiji_n1_plan175_v3_holdout.py)已准备但未运行。读题前逐seed核验固定run/40轮报告、train corpus与partition SHA、当前V3模型和runner源码SHA、迁移回执中的父checkpoint lineage、固定终点checkpoint路径/文件SHA/内部digest、V3恢复游标与累计更新、以及训练期留出/开发读取为0。随后复验原始OASST、既有split SHA和64个预选family的metadata/祖先链。所有检查通过后，在第一次提取留出正文前用独占创建和fsync耐久登记attempt；登记后的成功、失败或零输出均不重试，登记前的工程故障可修复。每题用各seed分别生成matched与确定性跨family错配prompt输出；案例和响应使用随机匿名ID并随机排列，不从seed、family或condition派生，映射只保存在独立揭盲key中。盲评文件与揭盲key在同一原子目录发布，失效family保留在64分母且不补抽。人工评阅按0/1/2标准记录，只有盲评后才揭示seed/条件。


N175首个正式epoch已全部完成，seed169/170/171均完成288更新、各曝光822311 target bytes，平均teacher-forced loss分别2.6297/2.5999/2.6203 nats。epoch01检查点已在独立新进程恢复并验证step/exposure/cursor及digest往返，详见[epoch01恢复回执](../../reports/taiji_n1_plan175_v3_checkpoint_restore_epoch01_20261026.json)。三个seed均已从epoch01 checkpoint进入后续训练，精确cursor见progress账本；单epoch训练损失只证明训练运转，不作能力结论。


N175 epoch02已完成：三个seed各288更新、822311 target bytes，mean loss分别2.1047/2.0941/2.0965 nats，CPU分别1406.30/1421.52/1413.27秒。三个epoch02检查点均已独立进程恢复、cursor与optimizer计数校验、重序列化digest一致；见[epoch02恢复回执](../../reports/taiji_n1_plan175_v3_checkpoint_restore_epoch02_20261026.json)。loss下降只说明teacher-forced拟合在推进，当前无自由生成或留出能力证据。


N175 epoch03已完成：三个seed各288更新、822311 target bytes，mean loss分别1.9887/1.9816/1.9823 nats，CPU分别1406.67/1414.56/1416.31秒。epoch03 durable checkpoints均通过独立进程恢复、cursor/optimizer计数和digest往返校验，见[epoch03恢复回执](../../reports/taiji_n1_plan175_v3_checkpoint_restore_epoch03_20261026.json)。三个seed现继续epoch04；此处仍无自由生成、未见任务或人工语义判读证据。


N175 epoch04已完成：三个seed各288更新、822311 target bytes，mean loss分别1.9197/1.9140/1.9138 nats，CPU分别1415.38/1425.02/1413.06秒。epoch04三个checkpoint均通过独立新进程恢复、cursor/optimizer计数和digest往返验证；报告为[epoch04恢复回执](../../reports/taiji_n1_plan175_v3_checkpoint_restore_epoch04_20261026.json)。三个seed继续epoch05。loss仍只是teacher-forced训练读数，不作能力结论。


N175 epoch05已完成：三个seed各288更新、822311 target bytes，mean loss分别1.8708/1.8659/1.8645 nats，CPU分别1450.98/1447.73/1447.31秒。epoch05检查点均通过独立进程恢复、计数和digest往返校验；见[epoch05恢复回执](../../reports/taiji_n1_plan175_v3_checkpoint_restore_epoch05_20261026.json)。三个seed继续epoch06；损失下降尚不能替代自由生成与盲测。


N175 epoch06已完成：三个seed各288更新、822311 target bytes，mean loss分别1.8322/1.8220/1.8257 nats，CPU分别1548.34/1551.45/1547.08秒。epoch06检查点均通过独立进程恢复和digest往返验证，见[epoch06恢复回执](../../reports/taiji_n1_plan175_v3_checkpoint_restore_epoch06_20261026.json)。seed169已进入epoch07；其余seed按epoch06 durable cursor继续。loss仅为teacher-forced训练拟合读数。


N175 epoch07已完成：三个seed各288更新、822311 target bytes，mean loss分别1.7966/1.7896/1.7902 nats，CPU分别1549.36/1555.14/1556.73秒。三个epoch07检查点通过独立进程恢复与digest往返校验，见[epoch07恢复回执](../../reports/taiji_n1_plan175_v3_checkpoint_restore_epoch07_20261026.json)。三seed继续epoch08，loss仅为训练teacher-forced读数。


N175 epoch08已完成：三个seed各288更新、822311 target bytes，mean loss分别1.7654/1.7580/1.7595 nats，CPU分别1456.33/1446.25/1449.00秒。epoch08 checkpoint均通过新进程恢复、cursor/optimizer计数和digest往返验证，见[epoch08恢复回执](../../reports/taiji_n1_plan175_v3_checkpoint_restore_epoch08_20261026.json)。三个seed继续下一epoch；损失只说明teacher-forced训练拟合，不作泛化结论。


N175 epoch09已完成：三个seed各288更新、822311 target bytes，mean loss分别1.7381/1.7299/1.7307 nats，CPU分别1412.05/1411.27/1411.67秒。epoch09检查点通过独立新进程恢复与digest往返校验，见[epoch09恢复回执](../../reports/taiji_n1_plan175_v3_checkpoint_restore_epoch09_20261026.json)。seed169/171继续epoch10，seed170刚完成epoch09；loss仅为teacher-forced训练读数。


N175 epoch10已完成：三个seed各288更新、822311 target bytes，mean loss分别1.7081/1.7002/1.7027 nats，CPU分别1412.13/1405.41/1412.36秒。epoch10检查点通过独立恢复及digest往返验证，见[epoch10恢复回执](../../reports/taiji_n1_plan175_v3_checkpoint_restore_epoch10_20261026.json)。seed169进入epoch11，另外两个seed从epoch10 durable点续跑；loss仍不代表自由生成或泛化。


N175 epoch11已完成：三个seed各288更新、822311 target bytes，mean loss分别1.6829/1.6744/1.6813 nats，CPU分别1410.47/1406.36/1417.20秒。epoch11三个检查点独立恢复及digest往返通过，见[epoch11恢复回执](../../reports/taiji_n1_plan175_v3_checkpoint_restore_epoch11_20261026.json)。seed171已进入epoch12，其余两seed正在续跑；teacher-forced损失不代表泛化。


N175 epoch12已完成：seed169/170/171各288更新、822311 target bytes，mean loss分别1.6569/1.6479/1.6568 nats，CPU分别1408.86/1409.00/1416.33秒。三个epoch12检查点均通过独立进程恢复、step/exposure/cursor核对及digest往返；见[epoch12恢复回执](../../reports/taiji_n1_plan175_v3_checkpoint_restore_epoch12_20261026.json)。三seed续训进入epoch13，64-family留出正文仍未读；teacher-forced损失不代表泛化。


N175 epoch13已完成：seed169/170/171各288更新、822311 target bytes，mean loss分别1.6352/1.6258/1.6347 nats，CPU分别1414.48/1406.56/1422.81秒。三个epoch13检查点均通过独立进程恢复、step/exposure/cursor核对及digest往返；见[epoch13恢复回执](../../reports/taiji_n1_plan175_v3_checkpoint_restore_epoch13_20261026.json)。三seed继续运行后续固定训练，64-family留出正文仍未读；teacher-forced损失不代表泛化。


N175 epoch14已完成：seed169/170/171各288更新、822311 target bytes，mean loss分别1.6140/1.6049/1.6150 nats，CPU分别1409.30/1412.86/1408.64秒。三个epoch14检查点均通过独立进程恢复、step/exposure/cursor核对及digest往返；见[epoch14恢复回执](../../reports/taiji_n1_plan175_v3_checkpoint_restore_epoch14_20261026.json)。三seed继续运行后续固定训练，64-family留出正文仍未读；teacher-forced损失不代表泛化。


N175 epoch15已完成：seed169/170/171各288更新、822311 target bytes，mean loss分别1.5895/1.5820/1.5989 nats，CPU分别1404.55/1407.55/1412.94秒。三个epoch15检查点均通过独立进程恢复、step/exposure/cursor核对及digest往返；见[epoch15恢复回执](../../reports/taiji_n1_plan175_v3_checkpoint_restore_epoch15_20261026.json)。三seed继续运行后续固定训练，64-family留出正文仍未读；teacher-forced损失不代表泛化。
