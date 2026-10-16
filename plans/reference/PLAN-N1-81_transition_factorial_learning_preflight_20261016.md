# 81 同实体四条件学习工程预注册（2026-10-16）

80报告SHA `842d34edee083ffaa82f7d4e83349d97a688804f4acd679d425636ab7fe13290`完整；seed13单因素翻转输出变化均0/48、联合0/48，seed12同架构联合48/48。唯一修订是学习目标，先验证工程，不在81正式训练或查询课程/dev/holdout/v4/v5。

## 固定学习合同

结构62056参数、两次读取、相对地址、问题/局部/上下文/关系机制、lr0.001及五组实际Adam位移上限0.001不变。原训练调度和改名课程继续适用；对每次transition更新，用80同一纯训练生成器生成original/turn_flip/direction_opposite/both，四个自然观察与同一问题分别前向，各自QA均值＋0.25观察保真，四份总损失等权平均，一次反传累积后仅一次有界Adam，四个独立会话保留原始观察历史。非transition调用原69更新器：lookup/unknown双问题、revision单问题。参数梯度/Adam矩不额外改写；正常推理不接入语法规则、条件标签或gold地址。生成规则仅供离线训练造题和评分。

## 工程范围与判据

fresh seed13 position，严格等于74相同seed全新初始化，空Adam/历史。独立CPU≤120秒，主轨迹0更新、0真实课程查询。零态checkpoint在开始任何正式训练前实际保存并由独立进程恢复；父进程和子进程各在副本执行相同两次合成transition四条件更新，共4副本更新/16合成QA，完整初态/末态参数、Adam、RNG、全部观察历史与每步损失/五组读数逐位相同；原零态不变。全部41参数Adam步数2，四组非writer与writer信用有限，第二步关系GRU梯度须正。失败保留证据，禁止自动重试/覆盖。

## 预声明产物

`scripts/training/_taiji_n1_transition_factorial_learning.py`；`scripts/training/audit_taiji_n1_transition_factorial_learning.py`；`tests/taiji_native/test_native_transition_factorial_learning_contract.py`；`reports/taiji_n1_transition_factorial_preflight_20261016.json`及`_attempt.json`；忽略目录`data/checkpoints/n1_transition_factorial_preflight_20261016/s13_position_zero.pt`。旧77/79/80代码和结果均不改。先artifact-free测试、静态检查、提交，再独占运行。所有实际合成更新单列，不算模型能力。

通过后另冻82：fresh seed13固定24576更新训练集校准，沿77父题/改名调度；转向题四条件平均目标，共55296QA及43008训练会话，独立CPU≤30分钟含零/终点恢复与评分。原四训练门不变，新增原名及已知改名48项各四条件联合≥44/48的前置，仅训练充分性，非未见能力门。不选中点/补训/自动追加，不查dev/v3holdout/v4/v5；正式多种子及首次未见准入另冻。此处只说明下一配方方向，82正式运行必须先有自己的冻结实现与提交。
