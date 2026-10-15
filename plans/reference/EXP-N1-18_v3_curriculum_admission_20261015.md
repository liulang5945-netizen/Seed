# EXP-N1-18 · 新内容课程准入（2026-10-15）

按用户对同类低成本工作的长期授权执行[18设计](PLAN-N1-18_fresh_content_curriculum_20261015.md)，未改14/15冻结协议或旧留出结果。

[v3 manifest](../manifests/n1_native_evidence_curriculum_v3_20261015.json) SHA256为`ac56fb0aba2ef602779f58749e6f8c35324c40c433fd16ce55b89689235b0604`。train/dev/holdout为192/32/32，四类各48/8/8；实体和观察材料词值跨split分离，每题两条观察。训练提取/修订有96个不同六字母目标词；正目标覆盖全部26字母，每个至少15次，超过预定至少8次。不以模型分数挑数据，不删除困难字符。

[预检报告](../../reports/taiji_n1_native_evidence_curriculum_v3_preflight_20261015.json)：CPU1.578125秒，12个train前向/反向检查通过，参数不变、optimizer更新0、dev与holdout模型查询均0。报告的model_identity记录复用的模型及旧运行器实现身份，内含旧配方；它不是新课程训练协议，新数据以顶层manifest SHA为准。此次没有正式训练或能力得分。

全项目仍为C1工程通过、C2严格收益零；17只证明已见90/96可拟合，C3未见行为、保持、C4净增益及同bundle三器官仍欠。其余N面和CI债保留；本轮不推送、不替换默认模型。

唯一下一步：为v3冻结独立train-only学习充分性校准，先验证新词值任务的学习剂量及成本，再确定全新三种子三臂配对协议。沿用模型/优化器，先独立恢复检查；不得用dev/holdout选参，不挪用旧预算，不把无更新预检推算成训练能力。

验证：相关模型/课程/训练合同与项目身份、台账唯一性、目录和secret守卫合计43 passed（4.14秒）；常规Ruff、B/SIM与Black通过。未运行全仓CI。

