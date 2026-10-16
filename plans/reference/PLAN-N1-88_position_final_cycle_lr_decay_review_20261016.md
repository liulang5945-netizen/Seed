# PLAN-N1-88 position末周期学习率衰减复验（2026-10-16）

## 依据与目标

run86c工程完整但seed12/position在末点未过六门。PLAN-N1-87在不加载模型的前提下分析已封存逐步loss和更新读数：seed12/position cycle4平均QA loss约为cycle3的18.8倍，增量主要来自transition；observation loss反而下降。五组Adam位移均未越上限，末点评分也显示transition/unknown错误与高loss部分关联。seed11/13 position保持六门，但cycle4 loss和组位移下降。该诊断不能确立因果，却把可检验的首要机制指向特定seed的末周期过度更新。

本轮只改变position臂最后6144次更新的实际Adam学习率：前三周期0.001，末周期0.00025。其余架构、Adam矩估计连续性、逐步样本/命名/QA目标、五组relative update上限、评分视图和六门全部沿run86c原协议。该方案用较小末周期步幅保留已形成能力，同时完成固定更新数；它不选中间checkpoint。

## 固定实验

- fresh seeds 11、12、13，全部position；每个24576更新、55296 QA序列、43008训练会话。
- 所有模型从各自新建零态开始。不得从run86c训练权重继承；可只比较记录的初始bundle签名以确认可比性。
- 五阶段0/6144/12288/18432/24576；每模型4800评分行，视图仍为original/pair/rename/rename_pair及原/改名四条件。
- 总计73728更新、165888 QA序列、129024训练会话、14400评分行、6次独立恢复、24副本签名步、96合成QA。
- 独立本机CPU上限4800秒，包含zero/final完整保存及独立恢复；串行单线程。训练前先执行三fresh零态checkpoint格式保存/恢复预检，预检不通过则不得训练。
- 正式输出使用独立run88报告/checkpoint族；隔离执行并校验sources/code/hash，禁止自动重试或覆盖。

## 冻结验收

沿用run86c/85的原六门，不降低、不换测面：position每seed原题≥173/192、四类各≥40/48、pair≥87/96、改名≥173/192且各类≥40/48、rename_pair≥87/96、原与改名四条件joint各≥44/48。三seed必须全部通过才算训练充分。要求完整SHA/原始行复算、零态与终点独立恢复、样本/更新/session/QA准确、sources/code unchanged及CPU accounting完整。

v5、dev、v3 holdout、v4及其他未见集查询一律为0；`TRAIN_SHA=None`、`can_promote=false`。run86c的失败与控制证据保持原状。即使本轮三position通过，也须另行完成准入裁决与确认面授权，不自动绑定85或晋升候选。

## 边界与后续

这是学习率配方变更，使用新的attempt/report/checkpoint族；不修改冻结runner或run86c输入。仅做一次正式三seed处理，不追加或挑选checkpoint。若任何seed六门失败，如实失败，先从全程逐步轨迹检视是否末周期loss/位移仍异常，再提出唯一下一版。CPU资源已由所有者全权授权，不需逐次确认；任何远端推送或删除数据仍视为不可逆操作。

## 开始条件

正式训练前完成独立fresh zero checkpoint保存/恢复，核对参数、完整Adam空态、RNG和历史一致，独立恢复CPU纳入4800秒总限。完成后的hash与preflight回执必须提交后才可启动三seed训练。
