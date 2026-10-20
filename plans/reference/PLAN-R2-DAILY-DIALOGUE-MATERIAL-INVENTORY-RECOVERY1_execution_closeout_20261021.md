# PLAN-R2-DIALOGUE-05R1执行结案：未消费自然对话材料清点

**状态：完成。** 有效读数为recovery1；初稿报告保留但作废，不参与结论。

## 清点结果

- OASST2固定源共135,174条消息，其中8,615条中文；严格语言、审核、隐私和安全筛选后接受5,250条中文消息。当前builder只生成710个回复episode（525 train、62 dev、123 final），覆盖340个train family、39个dev family和82个final family。
- train split内有509棵含中文消息的树；508棵至少有一条消息通过现行安全过滤。安全过滤后train有3,762条消息、共1,780,881个UTF-8正文字节：assistant 2,244条/1,623,306字节，prompter 1,518条/157,575字节。相对当前525条assistant回复样本，这是更大的全轮材料池；这些计数本身不代表模型能力。
- 按当前长度和质量约束，每个prompter选一条最佳assistant回复得到525条；若保留所有rank0/1/未排序候选，可得到704条（多179条），仍不能覆盖全部已接受自然对话文本。safe train中有168个family不在当前selected episode family集中，可进入全轮预训练候选。
- 39个dev family（62条episode）从未做模型查询，恢复清点只读取family metadata。82个final family已全部消费并排除。新的未见生成面可保留这39个dev family，做一次性自由生成审阅；不读reference response、不做匹配评分。

## 初稿报告的失效与恢复

初稿[报告](../../reports/taiji_r2_daily_dialogue_material_inventory_20261021.json) SHA-256=`19ec1cad3d98b1802066f275c8a96224ce0fd2acb4091e284ae50094362d0280`绑定的runner SHA与最终脚本不一致；旧解析路径还会把dev/final行完整JSON解码进内存。它没有打印或保存原文、没有模型加载/查询/评分，但不满足本计划的盲区边界，因此保留追溯、不作有效结论。

recovery1[报告](../../reports/taiji_r2_daily_dialogue_material_inventory_recovery1_20261021.json) SHA-256=`6bf34320c2963236f916fa920eb19911bf369ea8dc5ccdb120a81b1a2d3a5dc8`绑定最终runner及恢复计划SHA。选择字段解析器只跳过dev/final正文值并读取family metadata；train侧继续应用原审核、安全与隐私规则。初稿加恢复合计CPU24.765625/300秒；0模型加载、0训练/更新、0生成/dev/final查询、0checkpoint写入。

## 唯一下一步

依据审计证实的初始化和材料规模差距，执行[PLAN-R2-DIALOGUE-06](PLAN-R2-DAILY-DIALOGUE-FULL-TURN-PRETRAINING_20261021.md)：从更高容量的Taiji原生模型fresh initialization开始，以安全过滤后的train-family完整user/assistant消息做角色感知全轮byte预训练，再在精选assistant回复上校准；把未触碰的39个dev family保留为一次性自由生成面。旧final82 family不再用于训练或评估。
