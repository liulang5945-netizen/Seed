# PLAN-N1-150：匹配长上下文训练续接与8行control

**状态：依据用户授予的执行计划批准权已批准并冻结。**N1-149全体endpoint的known答案均随8→16→32行下降，480组相邻长度配对无错转对；N1-150据此训练一支长度混合candidate，并与同seed短上下文control匹配更新量。

## 目标与冻结父模型

- 每seed使用N1-132已过原始train gate的position full bundle（seed11/12/13），完整复制模型、Adam、RNG、history后，建立length-mixed candidate与8-row control；同seed两支zero bundle逐位相等。
- 三seed×两分支共6个endpoint。每模型固定9,216次更新，总计55,296次。
- 只用N1-132原始192条train父任务及其raw/hard训练视图，保留lookup/revision/transition/unknown四类和原问答目标。生成上下文仅加入Hamming d=1干扰事实；target事实保持原顺序并放在序列前部：lookup 1行、revision/transition 2行、unknown无target行。

## 唯一处理变量：上下文长度分布

- candidate对每个train父项48次更新：8/16/32行各16次，固定轮转8→16→32；每种长度下原始raw与hard路径各8次。不同长度是嵌套上下文，共享target、answer和distractor顺序，只增加尾部干扰行。
- control对同一父项执行48次8行上下文更新，raw/hard各24次。candidate/control的父项、类别、目标、优化器、学习率、损失函数、更新数及保存/恢复策略完全匹配。
- 训练所有数据必须来自原v3 train sources；v23/v24/v25/v26整面和其他留出来源继续排除。新生成的distractor实体/材料值须在face builder层做唯一性及排除核验，不能由任何holdout题面构造。

## 保存恢复、评分与门

- 训练前对六个zero bundle做独立完整保存/恢复，核对模型参数、Adam、RNG、history及指纹；任何一项失败即0更新停止。训练后六个final bundle再次独立恢复并做同样核对。
- CPU总上限7,200秒，包含来源/slot readiness、六个zero bundle前置、55,296次更新、final恢复、train-only评分与末端SHA复核；checkpoint目录write-once，不覆盖或重试。
- 每endpoint保留N1-132原192项训练门和每类门，判据/分母/阈值逐字不变；另按8/16/32行输出train-only描述性准确率与弃答率，不把长度分层增列为通过门。
- 只访问v3 train；0 dev、0 holdout、0 confirmation查询，不生成新评估面、不自动晋升。无论训练门结果如何，`can_promote=false`。
- 任一来源、更新数、保存、恢复、数据配平或CPU守卫失败即停止；不追加训练、不改变计划或门。

## 实施前顺序与产物路径

1. 实现schedule builder、matched continuation runner和合同测试，核验8/16/32及raw/hard slot计数。
2. 提交代码后运行zero-update readiness，确认来源SHA、三个N1-132父bundle完整恢复及六个zero bundle保存/恢复均通过；readiness通过并提交后才启动训练。
3. 独立恢复final checkpoints，运行冻结N1-132原门与length分层train-only评分，生成一份完整报告和中文结案。

- Builder：`scripts/training/build_taiji_n1_v27_length_robust_training_after_n149.py`
- Runner：`scripts/training/run_taiji_n1_v27_length_robust_training_after_n149.py`
- 合同测试：`tests/taiji_native/test_n1_plan150_v27_length_robust_training_contract.py`
- write-once checkpoints：`data/checkpoints/n1_plan150_v27_length_robust_training_seed126_20261020/`
- 结果报告：`reports/taiji_n1_plan150_v27_length_robust_training_result_20261020.json`
