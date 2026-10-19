# PLAN-N1-145：v23确认结果逐题错误归因

**状态：依据用户授予的执行计划批准权已批准并冻结。**本计划只分析N1-144已经消费并封存的确认行，不生成新题、不再查询模型，也不修改任何冻结门。

## 依据与目标

N1-144范围门8/12通过并完整查询960条confirm，但确认合取门未通过。N1-141 candidate在同面为lookup 2/60、revision 60/60、transition 45/60、unknown 60/60；逐seed lookup为0/20、1/20、1/20。N1-145把失败分布精确连接到题目目标位置、答案长度及同seed的冻结比较端点，帮助判断后续应继续做机制诊断还是另立训练/确认方案。

所有读数仅作post-hoc描述，不作因果解释、不作未见能力主张，也不能用于在同一v23面上调参后重新确认。

## 冻结输入

- N1-144冻结计划：`plans/reference/PLAN-N1-144_v23_long_context_confirmation_after_n143_draft_20261019.md`，SHA-256 `b5527b5f8a56aa12604a7dbc40dad1fa74c3f4ea8e953a6b2539f1fa3c5fcf42`。
- v23面：`plans/manifests/n1_v23_confirmation_seed123_20261019.json`，SHA-256 `e0ddf062e3c5f081dd010db3a8951d9aa426d9717de39872527a4eeabb76f6ae`。
- N1-144完整结果：`reports/taiji_n1_v23_confirmation_result_20261019.json`，SHA-256 `67038a9848b3c7ac088fea1983d9ba40ea153e91c8a4c436db830e8894e80388`；attempt/progress SHA须分别与结果中的绑定相同。
- 只用完整结果中的960条confirm逐题行。必须确认12端点各有相同80个唯一题ID、每端点每类20题，结果状态`confirmation_failed`、来源哈希未变、0更新、0 dev/旧v3 holdout查询。任一前置不符即停止且不写分析报告。

## 固定分析

1. 按endpoint与lookup/revision/transition/unknown报告精确命中数、分母和命中率；不要重新解码或重评分输出。
2. 对N1-141 candidate每个seed，按目标事实起始位置分层报告lookup/revision/transition命中；lookup/revision另按6/7字符答案长度分层。逐层同时列出题数和正确数，不做显著性或因果主张。
3. 对N1-141 candidate分别与同seed的N1-132 parent、N1-140 block candidate、N1-141 control逐题配对，按任务类别报告正确→正确、正确→错、错→正确、错→错四格计数。比较仅用于定位结果差异，不更改N1-144门。
4. 复算汇总必须与冻结N1-144结果中的`confirmation_summary`逐值相符；输出包含源SHA、完整逐题行数、CPU、0模型/checkpoint加载、0模型查询、0参数更新、0 dev/旧v3 holdout查询，以及`can_promote=false`。

## 资源、边界与产物

- CPU≤30秒；仅解析绑定的JSON报告/题面，禁止读取或加载checkpoint、训练状态或原始训练/验证数据。
- 不生成新测面、不运行推理、不改训练配方、不放宽范围/确认门、不重查dev或holdout。
- 脚本：`scripts/training/analyze_taiji_n1_v23_itemwise_errors_after_n144.py`。
- 契约测试：`tests/taiji_native/test_n1_plan145_v23_itemwise_error_attribution_contract.py`。
- write-once报告：`reports/taiji_n1_v23_itemwise_error_attribution_20261019.json`。
- 即使分析揭示稳定的分层差异，也只可据此起草新方案；不得回头重用v23作新确认或自动晋升，`can_promote=false`。
