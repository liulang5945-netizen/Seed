# EXP-N1-48 局部字节匹配问题残差工程裁决

[完整工程报告](../../reports/taiji_n1_local_question_binding_preflight_20261015.json)：26bd3baa7冻结，CPU7.546875秒，0参数更新/课程/dev/holdout查询。47480参数（新增3395），迁入41所有共享参数/Adam moments step6144，六个新参数张量无moment；新分支seed11初始化/RNG明确，不冒充旧RNG续接。

合成byte输入初始多步teacher logits/地址及free trace逐位等同41；初始新fusion信用非零、投影零信用符合零残差链式关系；合成非零fusion后投影/温度/池化/question/writer信用可达，cut时writer信用阻断。探针后参数/Adam/RNG/会话恢复原样；zero新进程恢复全bundle及多步状态签名一致，所有源码/源报告/档守卫过，26相关合同通过。工程通过不证明名称稳定、泛化或产品可采用。

唯一下一步49固定6144更新训练集校准，单独冻结配方和300秒CPU；从48同一迁移初态，保留原QA＋0.25观察保真（不使用46地址/保持损失），43固定原/改名stream和四门不变。未见/旧能力保持/C4/三器官及根CI/14历史格式债保留。
