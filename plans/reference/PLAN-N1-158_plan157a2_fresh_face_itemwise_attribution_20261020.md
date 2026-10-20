# PLAN-N1-158：N1-157A2 fresh face逐题配对归因

**状态：已依据用户授予的执行计划批准权批准，待实现与执行**  
**日期：2026-10-20**  
**来源执行：** [N1-157A2结案](PLAN-N1-157A2_execution_closeout_20261020.md)

## 目的与冻结输入

N1-157A2已消费seed128 face的2,880条逐题结果：三seed、两个分支（N1-155A candidate与N1-150 `length_mixed`父模型）、四类题、8/16/32行上下文。上位结果SHA-256为`c502ade59d20bcf05c7d84b898dcbc7b192187c42224c0e20ed342215a9cd1b5`，face SHA-256为`cf34e7de874b316581c4c6520786e2bf6406d328d4989b869901eb667e9c9a2f`，N1-157A2 progress SHA-256为`95992dc16b58ba32d5ea1de37837601b5eb1fef8d555e9bb24079d1193b2f441`。

本方案只分析N1-157A2已保存结果中的逐题输出。按`seed / parent_id / context_length / kind`配对candidate与parent，报告两边都正确、candidate-only正确、parent-only正确、两边都错；同时按8→16、16→32、8→32分别统计逐题错转对、对转错及保持情况。报告重点检查unknown任务各长度的弃答准确度，并保留lookup/revision/transition作为已知类对照。

## 输入合同与停止线

- 上位结果必须`complete=true`，逐题行2,880条齐全，原480行与新2,400行的计数正确，`can_promote=false`，source/checkpoint哈希稳定。
- 每个seed×kind×length必须有40个candidate/parent配对；缺失、重复、endpoint不一致或face/source SHA漂移即停止，不写部分归因结论。
- CPU≤30秒；0模型/检查点加载，0推理、0新查询、0训练/更新，0 dev/holdout/confirmation。
- 结论仅描述seed128 synthetic face，不作未见现实能力、机制因果或晋级主张，`can_promote=false`。

## 输出

- 分析器：`scripts/training/analyze_taiji_n1_plan158_plan157a2_fresh_face_itemwise_attribution.py`
- 契约测试：`tests/taiji_native/test_n1_plan158_plan157a2_fresh_face_itemwise_attribution_contract.py`
- 报告：`reports/taiji_n1_plan158_plan157a2_fresh_face_itemwise_attribution_20261020.json`

计划、实现与契约测试提交后，单次读取已消费结果并生成配对报告；不修改N1-157A2结果、原seed128 face、任一progress文件或checkpoint。
