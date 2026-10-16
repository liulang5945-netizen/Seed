# PLAN-N1-87 run86c 终点波动离线诊断（提案待批准，2026-10-16）

## 目的

run86c 的九模型训练工程已完整，但三种子 position 只有 seed11 与 seed13 通过原六门。seed12/position 在 18,432 步四项主评分均满分，固定终点变为原题176/192（unknown 38/48）、改名181/192，原始与改名四条件联合各24/48。现有逐步记录还显示末 6,144 步 QA loss 从 cycle 3 均值0.000250升至cycle 4的0.004688，observation loss同期约0.000248降至0.000029；cycle 4内的QA波峰集中在20,992–22,016与24,064–24,576。此相关性尚不能说明参数更新与终点失败的因果关系。

本提案只分析已经封存的数据，判断QA损失回升、样本种类、逐组更新读数和终点错误之间有哪些稳定对应，为后续唯一训练修订提供依据。它不改变训练协议、预注册门或评估测面。

## 冻结输入与前置

- 完整run86c报告：`reports/taiji_n1_transition_factorial_run86c.json`，SHA256 `0a8fc62e9aa00577d67001c70008773e6e5cee54cebad4115956e0360446d785`。
- 九个`run86c_s{11,12,13}_{position,terminal,cut}.json.gz`封存分片；必须逐一核对清单摘要、双层seal、模型身份和固定样本数量。
- completion回执与任务退出证据作为来源链，确认报告SHA、0退出码和工程完成。
- 仅读取已封存的loss、五组update measurement、五个训练评分点及其原始43200评分行；不得打开任何checkpoint或重跑模型。

前置缺失或摘要不匹配时立即停止并保留失败证据，不做部分解释。

## 单次分析

1. 按冻结6144步周期和原训练样本顺序重建每个已记录更新对应的train item、原名/改名轮次及QA序列数。
2. 以seed12/position为焦点，对照seed11/13 position及全部terminal/cut臂，报告cycle 3至4的每512步QA/observation loss窗口；另按四种样本类型、原名/改名子批及最终正确/错误样本分组。
3. 对相同窗口汇总reader/writer/local/context/relation五组的实际位移与上限比、裁剪次数和有限值状态；不把位移相关性解释成因果。
4. 把最终原题、改名、配对与48组四条件错误按样本类型和条件列出，并与18,432步终点评分对照；不得构造新模型输出或重新查询任何样本。
5. 明确区分直接观察、相关性和不能由现有轨迹确定的因果问题，最后提出单一下一步实验建议。任何新训练配方、判据或评估测面需另行冻结与批准。

## 资源与边界

- 独立本机CPU上限60秒；串行、单线程；一次执行，不自动重试。
- 0模型/checkpoint加载，0模型推理与查询，0权重/optimizer/signature更新。
- 0训练集重评分、0dev、v3 holdout、v4、v5或其他未见集查询。
- 输入只读，85继续`TRAIN_SHA=None`，v5仍未消费，`can_promote=false`。
- 生成唯一机器报告：`reports/taiji_n1_transition_factorial_run86c_offline_trajectory_audit.json`，绑定报告及全部九分片SHA；随后提交报告和本计划的结论追加段。

## 当前批准状态

已获所有者批准执行本次新增离线诊断；首次启动在导入阶段失败且未读取输入，零副作用回执已追加。原方案的单次限制已消耗，修复后再运行需另按PLAN-N1-87R批准。原批准不授权任何新训练、留出查询、改门或候选晋升。


## 首次执行处置（2026-10-16）

已获所有者批准后尝试一次。执行器在顶层导入阶段因未加入仓库根路径而失败；在读取任何批准输入前终止，0报告/分片/训练样本读取、0模型与checkpoint加载、0评分/查询/更新，CPU时间未在本次预检中单独计量。机器回执为`reports/taiji_n1_transition_factorial_run86c_offline_trajectory_audit_attempt.json`。本计划的单次/不重试限制已消耗，因此未自动重跑。唯一下一步为审议[PLAN-N1-87R](PLAN-N1-87R_run86c_offline_audit_retry_review_20261016.md)，申请修复导入路径后额外执行一次；该提案保持60秒CPU上限及全部分析/数据边界。
