# PLAN-N1-164：N1-163配对逐题归因与结果完整性核对

**状态：已按用户授予的执行计划批准权冻结并执行**
**日期：2026-10-20**
**前置批次：** [N1-163转向问句模板配对续训](PLAN-N1-163_transition_template_rebalanced_matched_continuation_20261020.md)

## 目的

N1-163完整结果显示六个endpoint中四个达到原N1-132训练门，失败集中于seed11的control与candidate。为判断模板替换是否对同一训练题产生稳定收益，只读配对复核已经生成的训练评分，按seed、任务、原问句/重命名/因子条件和上下文长度统计正确转错、错误转对及未变化项，并把复算摘要逐项与N1-163封存result对齐。

本分析只使用已消费的train评分，不加载模型或checkpoint、不推理、不训练、不产生新查询、不访问dev/holdout/confirmation，不改变原门和晋级状态。

## 冻结输入与执行界限

- N1-163 result：`reports/taiji_n1_plan163_transition_template_rebalanced_result_20261020.json`，SHA-256 `942883debd24b21559b8c27f7b2a54945fd22bffeaf7a39e3fef1da1c7669861`。
- N1-163 progress：`reports/taiji_n1_plan163_transition_template_rebalanced_progress_20261020.jsonl`，SHA-256 `32456df6730e8e5591dd5d94fa80040cf3a130a659b9c06d1c2bba017a6d5862`。
- 训练runner历史SHA-256 `e31b2c4f1a00f4554c6256cff433f4f6cc67400285f252393288f8a1eb5bfd96`；它必须与N1-163 result绑定值相同。
- CPU上限120秒；模型/检查点加载、模型查询、参数更新、checkpoint写入均为0。
- 每个endpoint固定按progress顺序读取960条原门行和288条长度评分；分组顺序必须与runner的seed外层、arm内层循环一致。逐题gate行的`arm`字段受旧runner字典展开顺序影响，显示为父档标签`position`；分析器不得据此猜测，必须用固定顺序分块，并将重算后的原门摘要与六个result记录全部对齐，否则拒绝出版归因。

## 输出与停止条件

- 只读审计器：`scripts/training/audit_taiji_n1_plan163_itemwise_attribution.py`
- 初版结果：`reports/taiji_n1_plan163_itemwise_attribution_20261020.json`，SHA-256 `83ce0037a44a040d8a9f241baf2ae319c1cdf6eadb2c9a4d507e623905f864b6`。该版的六个原门摘要复算均匹配，但长度评分在候选/对照移动统计中未按8/16/32行拆分，保留作历史并由v2取代。
- 修正版结果：`reports/taiji_n1_plan163_itemwise_attribution_v2_20261020.json`，增加逐长度分组、绑定审计器SHA和v1 SHA；原训练数据及输入hash不变。
- 若result/progress摘要、行数、seed顺序、endpoint分块或配对键不一致，审计器应失败且不得写出完整结果。
- 本计划只解释N1-163训练面内的变化。它不证明新题泛化、语言能力、架构因果或RNN固有上限。

## 后续判读

先报告同seed候选相对control的逐题净变化，再与各自N1-162B父档读数并列。若改进仅在训练面或单seed出现，下一步继续修订稳定训练配方并沿用冻结门；不据此查询新确认面或晋升。

## 执行结果

审计在2.0625 CPU秒完成；model/checkpoint加载、参数更新、checkpoint写入和所有新查询均为0。六个原门块复算均匹配封存result。v2报告SHA-256为`1f4da75db8b5b0dc4c5fd583d8c60c789f842ea0a64163bd9c67f4b6220471b9`；审计器SHA-256为`5f3dce6b0704964632d80d19efbf31a80093c74128d721128fe820a8956be279`。v2按8/16/32行分别核算长度分布，并绑定初版SHA；初版保留，不再用于长度判读。

逐题配对的主要读数：seed11 candidate在transition各评分类合计改对16项、改错59项；seed12两臂transition无变化；seed13 candidate有4项factorial transition回退、没有对应恢复。长度层面seed11在8/16/32行分别净−1/−4/−4，seed12 +2/−1/0，seed13 +2/+1/+4。原门通过状态为candidate 2/3、control 2/3；两臂均因seed11未过门而不能满足全seed充分性。

逐题原门行的`arm`字段均显示父档`position`，所以v2按封存result顺序和runner的seed/arm循环分成六个960行块；六块复算全部匹配。factorial行没有具名条件字段，报告仅按同一父题的重复出现顺序配对四条评分，不对单一因子作解释。详见[修正版报告](../../reports/taiji_n1_plan163_itemwise_attribution_v2_20261020.json)；它取代将长度合并统计的初版。
