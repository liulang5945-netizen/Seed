# 联合表征主体与四臂学习准入

> P3.7.3c／d，2026-11-13。正常主体、真实更新与恢复工程已通过，当前P3.7.4四臂有界学习。正式A父仍未成立。

## 1. 正常主体与信用

[主体报告](../../reports/taiji_u2_joint_subject_20261113.json)覆盖query／goal-control两模式、float32／64。25086参数／214张量／44字段，新增12张量共享于Native事件和表达上下文。两个路径分别关闭时，剩余路径的回答损失仍到达全部新张量，原Ψ资格范围保持其原定义。

关闭新融合后，逐字logp、H、旧202张量梯度／HVP及None模式精确一致。旧参数、真实Adam逐项继承，新12张量在工程step前无moment。4次真实工程Adam后全部新张量获得有限梯度和真实位移，新moment龄均1。新张量梯度范数最低约1.16e-4，最大位移最低约0.00099986。

8个更新前／后genesis运行时快照由独立进程恢复。真实公开图像、脚本错误首答及次问进入完整历史，1712条下一问逐参数梯度比较最大差0，state／optimizer／history摘要相同。更新后沿既有参数迁移事务重编码原始历史并原子提交，过去字节保留事实身份。后续学习使用更新前float32研究父，4个post只作工程证据。

## 2. 夹具更正与来源

原同化含两次绑定纠正及一次快记忆后整合，共三次Native调用。测试曾假设一次，源码与父／候选对照后改为保持三次；当前packet在这些调用中留在待提交侧，注册后才进入新读取。

首次运行使用默认ReferenceExecutor，物理观测守卫在0更新时拒绝，5.953125 CPU的[原始失败](../../reports/taiji_u2_joint_subject_20261113_v1.json)保留。确切验证脚本及v1合同归档，修订使用原PhysicalExecutor与同一genesis，生成的公开观测／元数据／outcome逐值等于任务fixture。4步预算及任务门保持原身份。成功报告CPU27.75已含该次失败，未越360帽。

旧u2_recovery源码归档后新增两个认知模块pin，旧D与TF父按登记的源差异显式迁移。正常保存采用新typed factory／schema，参数和源绑定。关闭态及错误mode／owner、真实moment、完整物理历史回归已补充。

## 3. 四臂实价与检查点

[实价报告](../../reports/taiji_u2_joint_learning_price_20261113.json)采用原D、query Fusion、goal-only Fusion和TF-actual父。各组8次独立影子Adam，共32步，真实learner更新0。四份完整初始learner保存后读回，下一任务梯度一致，原参数与moment摘要未被价格操作修改。

训练CPU中位数依次为0.2578125、0.40625、0.4140625、0.0859375秒；程序16 CPU，低于120帽。两个Fusion组参数及moment初值逐位相同，模式导致的函数差异保留，因此每组单独评估初始行为。旧D／TF因源版本重基线递增revision，权重与真实moment继承。

[执行合同](../manifests/taiji_u2_joint_learning_execution_20261113.json)冻结各组300实际训练CPU，最多8192更新，runner 3600／7200帽。复用已有预算终点函数，显式传入新typed loader／saver及source reader，原接口默认行为保留，完整旧源码归档。默认／显式两种API的越界回滚回归均通过。首步另记录全部joint_fusion参数信用和位移。

每步完整checkpoint，首个越界步隔离并恢复预算内参数、Adam、RNG和游标。CPU端点容差5%，磁盘最坏估计约26.2 GB。主组query仍须train≥0.90、dev及两轮各≥0.80、冻结父增益及RGB／问题切断各≥0.15。三个组间差只报开发点估计，正式多实体／三seed和独立check按原资格。

预留主体360、价格120和runner3600后，未分配1230 CPU，审计120在后续预留。当前唯一执行P3.7.4，完成后全局收束。

后续P3.7.4/5已按[四臂全局收束](CLOSEOUT-TAIJI_U2_JOINT_LEARNING_20261113.md)负结果结案。工程接受范围保留，正式任务资格为空；当前唯一后继见[P3.8学习校准](PLAN-TAIJI_U2_LEARNING_CALIBRATION_20261113.md)。
