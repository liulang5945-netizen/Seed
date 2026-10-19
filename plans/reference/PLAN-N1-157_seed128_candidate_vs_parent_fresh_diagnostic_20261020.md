# PLAN-N1-157：seed128 fresh face candidate与父模型配对诊断

**状态：已依据用户授予的执行计划批准权批准，待实现与执行**  
**日期：2026-10-20**  
**依据：** [N1-155A执行结案](PLAN-N1-155A_execution_closeout_20261020.md)、[N1-156逐题归因](PLAN-N1-156_execution_closeout_20261020.md)

## 目的与比较对象

N1-155A的三个`unknown8_rebalanced` candidate均通过N1-132原训练门；seed12的`uniform_length_replay` control transition为30/48，导致上位N1-155六端点合取门失败，seed128 face当时按协议未生成、未查询。N1-156进一步显示seed12 control的transition错误集中在control：原始和重命名48题中各有18题仅candidate正确、无题仅control正确；但unknown长度评分在8/16/32行没有稳定的candidate优势。

为继续检验再平衡candidate在未见合成题上的表现，本方案使用一张此前未生成、未消费的seed128 face，配对比较：

- 三个N1-155A `unknown8_rebalanced` final checkpoint；
- 对应三个N1-150 `length_mixed` final parent checkpoint。

这是一项新的candidate对父模型fresh诊断测面。N1-155失败的uniform control仍按失败保留；本方案不会把它替换成通过结果，也不会借此晋级。

## 冻结样本与查询

- 复用N1-155已冻结的seed128 face规格：40题/类、四类各40题；每题有8/16/32行嵌套上下文。
- 每endpoint 480次评分，六endpoint合计最多2,880次；先以只读source readiness绑定全部题面来源与六个checkpoint，再独立恢复六个full bundle后一次性评分。
- 来源排除所有已消费面、N1-153A seed127 face、N1-150/155训练实体及材料值；逐项验证目标/干扰实体唯一、材料唯一且与排除集不相交。
- face一经生成即write-once；查询前核对manifest、source audit、checkpoint哈希与diagnostic split；完整保存逐题target、teacher-forced、free-generation及原始输出字节。

## 资源与判读边界

- N1-157阶段CPU≤3,600秒，涵盖readiness、六bundle独立恢复、建面与评分；累计计入N1-155A已用9,325.21875秒后，总CPU不得超过上位18,000秒。
- 不训练、不更新参数，不访问dev、holdout或confirmation；不改变任何训练门或N1-155失败结论。
- 仅报告候选与原始length-mixed父模型在这张fresh synthetic face上的描述性差异；不将其称为confirmation，不作因果或能力晋级主张，`can_promote=false`。

## 实现与输出

- 建面/来源审计：`scripts/training/build_taiji_n1_plan157_seed128_candidate_vs_parent_face.py`
- 诊断执行器：`scripts/training/run_taiji_n1_plan157_seed128_candidate_vs_parent_diagnostic.py`
- 契约测试：`tests/taiji_native/test_n1_plan157_seed128_candidate_vs_parent_diagnostic_contract.py`
- seed128题面：`plans/manifests/n1_plan157_v30_candidate_vs_parent_seed128_20261020.json`
- 检查点仍使用既有N1-150/N1-155A文件，不新建权重；审计、readiness、attempt、progress、result均写入`reports/`。

代码、测试及本计划提交后，先完成只读来源ready；再完成六bundle独立恢复前置并提交；最后才生成题面、验证排除与合同并执行一次诊断。任一write-once阶段失败即封存回执，不覆盖、不自动重试。
