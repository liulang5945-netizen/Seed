# 输入—答案配对信用干预

> 22.5.P2已执行并按intervention_not_supported_no_A_parent结案。本卡保留冻结设计，P1研究checkpoint与全部原始失败证据保持其身份。

## 1. 事实与命题

P1三模型在训练内及未见三对象上的候选序列排名均为0.25，Native/GRU自由生成仍0，Transformer为常量答案。RGB切断对NLL影响很小。下一项命题限定为输入条件化的学习效率，当前证据不独立证明架构不可学。

同一场景替换请求实体，公开RGB oracle确认其正确答案不同，将该输入作为负配对。两个输入对同一个真实完整答案y评分。定义平均逐字NLL为ℓ，采用

\[
L_{token}=\ell(y|x,q),\qquad
L_{conditional}=L_{token}+10\,softplus(0.1+\ell(y|x,q)-\ell(y|x,q^-)).
\]

正确/错误输入共用y，差分对公共语言项形成抵消，目标是让任务信用区分真正的输入配对。该预测需实测，不能由梯度非零代签行为。

## 2. 四臂

Native和P1最强行为基线Transformer各分token与conditional两臂，同架构从各自P1末参数fork，重置Adam且两臂相同。无新参数/输出头，推理保持257字节、原UTF-8/EOS掩码和64字节上限。负配对只用于训练，推理不查答案表。

train沿P1的64场景，开发fit使用原recipe的128–159号新场景32例；原fit32的P1结论保留。新fit仅开发范围，check/final读取0。原GRU结果保留，本卡只做目标干预，后续完整架构比较仍需三架构。

每臂按冻结价格获得约32训练CPU秒、最多256更新，总帽600 CPU/1200 wall，从22.5剩余预算预留。8次前反向定价不更新权重，计数在新fit模型结果读取前冻结。每步checkpoint、Adam和数据游标读回。

## 3. 终点与出口

主行为门仍为fit完整自由回答≥0.50，且conditional相对同架构token增益≥0.15。阶段只签开发干预；正式A父0.80/3seed/check保留。

另记录正确/错误请求下正确答案的NLL差、RGB切断差以及负责参数梯度与实际位移。只增加条件依赖但行为未过门为conditional_credit_only。未见增益为intervention_not_supported，不加步。异常、预算停止、无法配对均独立结案。

若两臂都学不到输入关系，下一次重审应检查目标条件读取/感知表征路径。当前不以辅助分类头的成功替代原生表达能力。

## 4. 实际结案

四臂完成282次更新，Native-token/conditional均0/32，Transformer-token/conditional均8/32。准备及运行316.359375 CPU，符合600帽。原始completed标记只签运行完成，科学出口按[结案回执](CLOSEOUT-TAIJI_FIRST_A_CONDITIONAL_20261112.md)签为intervention_not_supported_no_A_parent。0.50及增益0.15均未成立，未追加本轮更新。

后继[读取重审](PLAN-TAIJI_FIRST_A_READING_REVIEW_20261112.md)已完成只读归因，选定请求条件证据读取的架构候选。新模块局部核验与主体集成各自签署，新训练待新源与实价合同。
