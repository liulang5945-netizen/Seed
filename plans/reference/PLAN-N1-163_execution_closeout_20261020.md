# PLAN-N1-163执行结案：转向问句模板重配对续训

**状态：训练、评分和checkpoint核验完整；原训练充分性门仅4/6 endpoint通过；不晋级。**
**日期：2026-10-20**
**冻结方案：** [PLAN-N1-163](PLAN-N1-163_transition_template_rebalanced_matched_continuation_20261020.md)
**来源重绑定：** [PLAN-N1-163A](PLAN-N1-163A_readiness_source_rebinding_20261020.md)

## 执行完整性

- 完整结果：[result](../../reports/taiji_n1_plan163_transition_template_rebalanced_result_20261020.json)，SHA-256 `942883debd24b21559b8c27f7b2a54945fd22bffeaf7a39e3fef1da1c7669861`；`status=complete`、`complete=true`、`errors=[]`。
- 固定更新 **110,592/110,592**，train-only评分 **7,488/7,488**（每endpoint原门960条、长度分层288条）。进度包含六个endpoint各18,432个更新步、5,760条原门行和1,728条长度评分行；独立复算与结果汇总逐项一致。
- 六枚zero bundle与六枚final bundle都独立完整恢复（6/6、6/6），父checkpoint和来源守卫通过。阶段CPU **17,006.796875/25,000秒**；关联累计CPU **38,251.734375/50,000秒**。
- 新面、dev、holdout及confirmation查询均为0；`can_promote=false`。

## 原训练门与候选/对照变化

对照和candidate都只在seed12、seed13通过原N1-132门，共4/6；seed11两臂都未通过。因而candidate与control的全seed充分性均为false，未触发后续能力确认。

- seed11的candidate没有优于匹配control：原题transition为41/48对47/48，重命名transition为42/48对44/48；原题/重命名factorial joint-four为28/48与26/48，对照为45/48与40/48。逐题配对在全部transition评分条件上统计到candidate改对16项、改错59项，净−43。
- seed12两臂transition均为48/48，factorial joint-four也都为48/48；candidate没有带来可辨认的transition增益。
- seed13两臂原题与重命名transition均为48/48；candidate在factorial joint-four为45/48与46/48，低于对照47/48与48/48，但两臂仍过原门。
- 288条长度评分的candidate-control净变化按8/16/32行分别为：seed11 −1/−4/−4，seed12 +2/−1/0，seed13 +2/+1/+4。方向随seed和长度变化，不能宣称稳定的长上下文增益。

逐题读数来自[PLAN-N1-164 v2归因报告](../../reports/taiji_n1_plan163_itemwise_attribution_v2_20261020.json)。factorial原门行未记录具名条件，v2只按progress里的重复出现顺序配对四行，不将变化归到特定条件。初版报告将长度混在一起，已保留并由v2取代，见[PLAN-N1-164](PLAN-N1-164_plan163_itemwise_pair_attribution_20261020.md)。

## 终端偏差与来源边界

runner在完整result成功写出后，终端摘要索引了不存在的`training_updates`键（报告字段名为`total_updates`），因此进程最终退出码为1；原子结果本身完整且`errors=[]`。write-once attempt仍记作`started`。progress中的原门行`arm`也被`score_row`展开覆盖为父档标签`position`；PLAN-N1-164依固定seed/arm顺序重分六个960行块，并逐项复算六个result摘要后才报告配对变化。原始result、attempt与progress均保留，runner源码没有在本次执行后改写。

## 判读与唯一下一步

将`direction_after_turning`槽位全部替换为`direction_after_turn`，未实现跨seed稳定改善：seed11显著落后于同预算control，seed12持平，seed13略有回退。现有结果支持模板比例存在seed敏感性，不能证明模板名称本身具有普遍因果收益，也不能证明RNN固有局限。

唯一下一步是[PLAN-N1-165中等剂量续训](PLAN-N1-165_transition_template_mid_dose_calibration_20261020.md)：从同一N1-162B父checkpoint测试50% original、37.5% preferred、12.5%保留变体；每个parent/route/长度八槽块为4/3/1，三seed各训练一个candidate，与N1-163同seed的25%和50% preferred剂量锚点比较；保留原门，不查询新面。
