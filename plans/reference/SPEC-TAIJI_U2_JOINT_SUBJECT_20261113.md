# 联合表征正常主体集成与恢复合同

> P3.7.3c，2026-11-13。工程360 CPU／720 wall，4次工程Adam更新，0影子步。预留后未分配4950 CPU。正式学习剂量在后续实价卡冻结。

## 1. 父与正常入口

[机器合同](../manifests/taiji_u2_joint_subject_contract_20261113.json)固定P3.6预算内D父和TF参考父。新候选JointQuerySubject在有效request期间使用问题局部单位，其余使用persistent goal；JointGoalSubject始终在新融合路径使用persistent goal。两者共用原D权重、真实Adam及相同新增初始化，旧214之前的202张量顺序保留，新增12张量1696参数，合计25086参数／214张量／44状态字段。

同一joint_fusion在native事件入口和branch_context中调用，状态改变后按当前H重算。两路径参数共享、可微，原actor资格范围沿policy_named_parameters保留。新参数归Θ。关闭joint_tokens切片时，两路径增量为0；joint_native或joint_expression切片分别隔离另一条任务信用路径。

原同化correct含两次绑定纠正中的Native调用和一次快记忆后整合，共三次。新融合沿这些原有转换接入，读取已登记证据前缀，当前packet提交后才进入目录。本轮补充测试曾将该段误写为一次调用，已通过父／候选对照及源码核对更正。推理、动作、字节接收和EOS继续沿原事件顺序。

## 2. 方程与实现覆盖

| 方程／责任 | 正常源码 | 参数／状态 | 核验 |
|---|---|---|---|
| c=Fusion(Q,E,H) | u2_joint_subject.joint_feature | joint_fusion的12张量；request／goal、canonical目录、H | 有效／缺失query来源，已提交前缀、逐新参数梯度 |
| Native(e＋c) | u2_joint_subject.native | 原Native与同一Fusion参数；H | 表达新支路关闭时answer_loss仍到达12张量 |
| z_old＋c | u2_joint_subject.branch_context | 原读取、同一Fusion参数及原字节头 | 递推新支路关闭时answer_loss仍到达12张量 |
| 关闭态 | joint_tokens验证切片 | 旧202张量、logp、H | 双精度逐字输出／H／旧梯度／HVP及None模式 |
| 参数与moment继承 | joint_token_learner.fork_with_optimizer | 旧状态179项、新12项最初不存在 | 旧值逐项保留，参数索引扩展，真实step后新龄1 |
| typed恢复 | JointQuery/GoalRuntime，专门learner factory | mode、父摘要、214张量及44字段 | 8个独立恢复，下一问完整214梯度与状态／moment／历史摘要 |
| 参数升级 | 原Runtime.migrate_parameters | 完整原始历史、固定过去字节、当前参数状态 | 实际更新后重编码，一次事务提交，随后独立接续 |

旧u2_recovery源码确切归档后新增两个认知模块pin。父加载逐条检查原pin，只有登记的运行时源码差异与新增模块允许迁移。新subject和runtime采用专门schema／factory，默认旧类型继续承担各自的源身份。

## 3. 工程执行与资源

两mode×两精度各做一次真实Adam，共4步。每次先保存并读回完整pre learner及optimizer，再建立含公开图像、错误首答和次问的genesis运行时窗口。错误首答字节为脚本协议证据。随后在独立候选上用原actual-self两问CE更新一次，保存post learner，重编码原始历史后按原参数迁移事务提交。工程acceptance仅核验事务可提交，任务父资格沿后继行为门。

每组保存更新前／后运行时快照，共8个，由独立worker恢复。CPU总账包括worker，超360／720则budget_stop。全部新张量须有有限非零任务梯度、float32/64真实位移及真实Adam龄1。保留更新前float32研究父供后续统一实价，post只作工程证据。

新增状态0，派生融合特征重算。原请求、来源、模式、状态／moment错配守卫和关闭态回归均检查。工程产物归data/experiments，报告归reports，合同与回执归plans。失败按已发生的真实step保留计数与checkpoint。

## 4. 工程及实价结案

[主体／价格回执](VALIDATION-TAIJI_U2_JOINT_SUBJECT_20261113.md)记录全部4步及8独立恢复／1712梯度等值。首次默认执行器夹具在0更新时失败，旧报告／合同／源码归档，PhysicalExecutor按同一genesis重建并逐值核验公开输入；总CPU27.75包含失败5.953125。当前四臂实价与完整检查点准入已冻结，下一P3.7.4按实际CPU终点及原任务门裁决。
