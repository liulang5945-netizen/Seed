# PLAN-N1-161A：seed131未见问句面逐题输出归因

**状态：已依用户授予的执行计划批准权批准并冻结**
**日期：2026-10-20**
**上位结果：** [PLAN-N1-161结案](PLAN-N1-161_execution_closeout_20261020.md)

## 目的

N1-161完成2,880条未见问句句式迁移诊断后，candidate总分低于parent，但unknown略高；已知差异集中在lookup/revision主动式模板。对封存输出做一次可复算、零查询逐题错误类型归因，判断错误是否表现为literal unknown、旧revision值、观测材料值、固定方向/turn词或其他字节输出。该归因只描述这张synthetic face，不据此作自然语言能力、机制因果或晋级主张。

## 冻结输入与计算

- 输入N1-161逐题result SHA-256 `2808a969299143c692643bf4bd5ccabdd67a63021f5ad9ba0cab65bcb51a385a`、seed131 manifest SHA-256 `39ce277831a0461a45bf7fb0f793d57f89ade84e988d1ba830703b07346393ef`、face audit SHA-256 `cc08a8ec7af299207c8f474db967b8d0eda76aca2fc8be2480b06fe41e2b3b46`、progress SHA-256 `dd80604cd66519f04bfe48d5929756e0f2f5436a68435d3a5df7a31068f66cce`。
- 2,880行必须完整且唯一地按`seed/branch/parent_id/context_length/question_template_id`关联到manifest；question/kind必须一致。离线重算`free_exact = raw_bytes == answer_bytes AND stop_reason == boundary`并与已存字段逐行一致，报告teacher/free exact不一致数。
- 对错误free输出按固定优先顺序分类：literal unknown；revision旧材料值；其他观测材料值；固定direction/turn词；其他输出（另计空输出）。保留每类的endpoint、类别、template、长度分层以及candidate/parent逐题转移。
- 报告post-hoc分类，明确字符串类别不等于模型内部检索机制证明。

## 资源与产物

- CPU≤30秒；0模型加载、0 checkpoint加载、0模型查询、0参数更新、0 dev/holdout/confirmation访问；不生成新face。
- 写入`reports/taiji_n1_plan161_question_surface_itemwise_attribution_20261020.json`。
- 计算器：`scripts/training/analyze_taiji_n1_plan161_question_surface_itemwise_attribution.py`；契约测试：`tests/taiji_native/test_n1_plan161_question_surface_itemwise_attribution_contract.py`。
- 任一SHA、逐题关联、exact复算或预算守卫失败即写停止回执，不更改原始数据、不重试。
