# EXP-N1-37 残差问题对齐工程裁决

用户弹窗批准37，独立候选冻结fbf06839c。参数44,085（新增3616）不变，fused=state+tanh(fusion([state,context]))，融合权重/bias零初始化。保留35旧实现及格式，残差拥有新format/机制标识，不静默恢复旧语义。

34项相关合同通过（3.32秒），Ruff/B/SIM通过。初始共有参数与旧relative同seed逐位一致，position/terminal/cut多步logits/attention及teacher相等；残差融合初始梯度非零，question key/value因零融合初始梯度为零是预期，合成非零融合测试中两者梯度非零；无训练的人工路径测试不作模型能力证据。cut对QA与辅助无writer旁路，teacher/free同轨迹、来源/session/tape隔离及独立格式恢复有效。

[正式预检](../../reports/taiji_n1_residual_question_preflight_20261015.json)完整，CPU6.265625<60，0更新/课程/dev/holdout；共有参数/原生多步相等、三臂finite及信用、来源/参数不变、新进程恢复、checkpoint和代码守卫通过。zero Adam为空moment，只验证零更新恢复，不声称训练后完整moment已验证。

尚无新模型训练或泛化成绩。33未见门失败、36训练门失败均保留；全项目C1工程过，原C2严格收益零，旧能力保持/C4/三器官/其他N和根CI债未闭合。唯一建议38固定训练集校准：保持36原9216/seed11/300秒配方，仅候选换成获批残差结构，正式训练须另批；不复用v3留出。
