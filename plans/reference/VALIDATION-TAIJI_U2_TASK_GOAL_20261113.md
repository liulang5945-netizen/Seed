# 活跃任务条件化与实际历史工程回执

> P3.6.1–3，2026-11-13。passed_neutral_goal_and_conditional_history_engineering。真实learner更新0，影子Adam8次。当前队首P3.6.4实价及四组准入。

## 1. 接口定稿

[接口合同](../manifests/taiji_u2_p3_6_interface_contract_20261113.json)冻结两问损失各0.5权重，每问CE按7个监督符号均值。actual-self第二问只使用模型greedy首答状态，首问teacher CE位于独立分支。未结束首答保留first CE半权重，second CE不计算，记录停止梯度的blocked outcome半权重，评测第二轮计错。64字节/EOS边界、argmax不导选择分布梯度、确定性RNG及状态来源均明确。

活跃任务投影g_eff=g+Aq仅在request pending/answering时使用，answered/cancelled/不存在时关闭。仍有效但goal_version较早的pending request沿原接口持续到显式结案，persistent goal原始值和版本保留。active-query以Enc(request)为输入，persistent-control以原goal参数为输入；A均为8×8无bias、零初始化。

A归共享表征Θ，沿原task/预测/开发责任表登记，原Ψ只含actor头的资格迹边界继续签署。新A没有伪造历史actor资格。旧201参数顺序和moment原值继承，新第202张量moment尚未建立；typed bundle登记mode、祖先参数摘要、参数名和Θ归属。

## 2. 正常实现及来源

`taiji/u2_task_goal.py`提供两种显式subject/runtime。所有Native消费者接收同一effective goal，模型state仍44字段、23,390参数/202张量。旧model默认类型保有其行为，新增类型通过专门factory映射加载。

原运行时及learner serializer的确切源码按接口合同归档。加载研究父先校验P3.5选定文件SHA及全部旧pin，仅登记的两处源变化和新模块可通过。旧参数、Adam、RNG与数据游标按其原身份继承；新研究父parameter_revision加1。serializer默认调用保持原接口，候选loader显式传入类型表及额外源码身份。

## 3. 数值与恢复证据

[机器核验](../../reports/taiji_u2_task_goal_engineering_20261113.json)覆盖2模式×2历史×2精度共8组。在A=0时，逐字logp及H精确一致，旧201参数梯度和旧参数方向HVP最大差均0，旧None模式保留。策略的数值输出、value、critic输入及发展预测在中性起点等值，版本身份递增。

新A梯度范数最低2.652e-7；影子Adam位移最低8.3346e-4，真实模型参数与optimizer摘要不变。单episode的active第二奇异值约2e-11，control约1e-14，按既有1e-8绝对秩阈值观察均仅一个主方向。本回执签署可优化路径，条件语义信用增益尚待行为验证。

4个typed独立进程恢复覆盖两模式/双精度，脚本首答后设置第二问，808条梯度比较最大差0。脚本字节仅为协议证据。状态切换、goal版本、错误类型拒绝、moment继承、实际历史不被teacher替换及blocked固定分母均有回归。

本轮相关回归107项通过，追加策略数值中性检查单列复核。初版回归夹具误用owner方法名并将候选状态传给父版本，版本守卫正确拒绝，夹具已修正。程序CPU17.765625低于240/480帽；工具与回归资源另列。影子Adam8，真实learner更新0，check/final读取0。预留240后当前未分配9750 CPU。

## 4. 唯一后继

P3.6.4按同一研究父构造teacher/actual-self×persistent-control/active-query四组，校准真实Adam与实际CPU终点，逐组验证checkpoint。参考组如纳入，须显式固定其父及来源。新学习剂量在读取候选新行为前冻结。

[工程接受清单](../manifests/taiji_u2_task_goal_engineering_20261113.json)绑定父文件和当前源，状态为工程接受，learning_admitted=false。
