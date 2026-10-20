# PLAN-N1-157A2 执行结案

**日期：2026-10-20**  
**状态：完整结束；描述性fresh synthetic诊断**  
**更正方案：** [PLAN-N1-157A2](PLAN-N1-157A2_readiness_binding_metadata_correction_20261020.md)  
**上位诊断：** [PLAN-N1-157](PLAN-N1-157_seed128_candidate_vs_parent_fresh_diagnostic_20261020.md)

## 执行完整性

N1-157的原尝试保留seed11 candidate 480条记录及停止错误；N1-157A source binding因两个遗漏字段在checkpoint恢复前停止，恢复、查询和更新均为0。N1-157A2补齐绑定元数据后，六个candidate/parent full bundle独立恢复6/6，复用原seed128面及原480条，未重测该endpoint；只对其余五个endpoint新增2,400条查询。合并后2,880条逐题记录齐全。更正轮progress有2,400行、无坏行，SHA-256 `95992dc16b58ba32d5ea1de37837601b5eb1fef8d555e9bb24079d1193b2f441`；source与checkpoint哈希稳定。

N1-157A2阶段累计CPU264.500/3,600秒，关联项目累计CPU9,771.531/18,000秒。0训练、0参数更新、0 dev/holdout/confirmation查询，`can_promote=false`。首轮错误与binding失败保留在各自回执中；同一face没有重生成，seed11 candidate未重复查询。

## 描述性读数

candidate的每seed总命中为seed11 457/480、seed12 478/480、seed13 466/480；对应N1-150 `length_mixed`父模型为415/480、472/480、454/480。三seed在该面上candidate均高于父模型。按类别汇总，candidate的lookup/revision/transition为356/360、360/360、360/360；父模型为358/360、360/360、360/360。

unknown正确弃答在8/16/32行分别为candidate 96/120、110/120、119/120，父模型65/120、93/120、105/120。题内配对的candidate-only/parent-only为：8行35/4，16行20/3，32行14/0。差值主要来自unknown题；这只是seed128合成测面上的描述性结果，不是confirmation、因果隔离或能力晋级。

完整逐题字节、分层结果、来源及CPU证据见[结果报告](../../reports/taiji_n1_plan157a2_candidate_parent_result_20261020.json)。

## 后续唯一队首

依据用户授予的执行计划批准权，已批准[PLAN-N1-158](PLAN-N1-158_plan157a2_fresh_face_itemwise_attribution_20261020.md)：对本次2,880条已消费逐题行做完全离线paired归因，按seed、kind、长度分解candidate/parent独有正误及8→16→32行逐题变化；CPU≤30秒，不加载模型/检查点，不做推理或新查询，不训练、不访问dev/holdout/confirmation，不改门或晋级状态。
