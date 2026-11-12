# 归因段工程失败与续接修订

> 2026-11-12。v1原始结案为engineering_failure，保留原报告；v2按已接受1000步的规范检查点续接。学习合同数值、模型、数据及总帽保持同一身份。

## 1. 故障与证据

运行器在query训练490步全部完成后进入归因评估，直接访问可选`_verification_cuts`属性，实例尚未初始化该诊断属性，产生AttributeError。U2正式路径本来使用getattr默认空集合，运行器漏用了相同处理。该错误发生在诊断关闭新残差之前，已完成的模型更新和保存读回均已提交。

v1[原始报告](../../reports/taiji_u2_evidence_A_learning_20261112.json)记录1000真实optimizer调用及1000 accepted checkpoint，账本opened/accepted均1000。Native-old510、Native-query490步完成，Native-old完整fit为0/32；query终点检查点可用，诊断段记录未完成。累计CPU1227.3125。

## 2. 修订与恢复准入

诊断改用getattr默认空集合，并增加未初始化诊断属性的回归检查。控制/learner模块9项通过。v1运行器确切源码保存在`scripts/archive/run_evidence_a_22_5_20261112_v1.py.txt`，其规范UTF-8摘要与原执行合同匹配。

恢复准备逐一核对1000份accepted文件摘要、claim、最终参数/Adam/游标，完整前缀连续且无未定提交。query的首步信用从原initial、原第一批和已接受step1重建，字段明确标注重建来源。已完成Native-old结果沿原记录保留；query从490终点补全归因评估。后续只执行Native-capacity490及Transformer3891，共4381个剩余更新。

[v2执行清单](../manifests/taiji_u2_evidence_A_execution_20261112_v2.json)绑定恢复源、原失败报告字节摘要、账本摘要、前缀检查点及累计资源。恢复准备CPU6.625计入同一3240帽，续接前CPU1233.9375。wall按原报告时间锚包含故障后的修复间隔。模型源码、所有初值、Adam配方、数据顺序、门值、原更新总数5381及预算边界逐项保持原合同身份。

后继按v2完成余下训练及完整结案，保留v1工程失败证据，最终比较须有四臂完整读数。恢复期间追加预算或修改门值均不属于本次修订。
