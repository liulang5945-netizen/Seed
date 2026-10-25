# N1-168隔离小集学习充分性结案

日期：2026-10-25。状态：**CLOSED_COMPLETE，固定8轮未通过小集拟合门**。[PLAN](PLAN-N1-168_isolated_train_fit_20261025.md)/[SPEC](SPEC-N1-168_isolated_train_fit_20261025.md)与实现先在`6aad91fee`冻结；本包不因未达到能力线延长。

## 实际完成与恢复

- 原full_turn_1m只读，新副本4题×8轮=32额外episode、9320目标字节；累计525→557 episode、113209→122529更新。原学习规则、配置、目标encoder与corpus身份保持，没有超参搜索。
- zero先保存并在新进程weights-only恢复，完整payload/model/RNG/计数及首步概率签名一致后才训练；final同样独立恢复通过。zero/final分别保存在`data/checkpoints/n1_plan168_train_fit_20261025/`，原parent SHA不变；未采用child为默认模型。
- 两次恢复及训练全部进程由同一Windows Job计量，实测118.890625 CPU秒、122.141 wall秒，树退出已确认、watchdog未强制触发。
- 0/1/2/4/8轮共20个固定评估全部完成，0 dev/holdout/reference/N1-166访问，`can_promote=false`。

原始学习/评估/恢复记录见[结果](../../reports/taiji_n1_plan168_isolated_train_fit_20261025.json)，曲线见[汇总](../../reports/taiji_n1_plan168_learning_curve_summary_20261025.json)，全进程费用见[Job回执](../../reports/taiji_n1_plan168_job_exit_20261025.json)。原始数据未覆写；child是隔离实验资产，不是能力晋级证据。

## 学习曲线

TF字节和前缀匹配是4题宏平均；完整汉字TF按148个目标汉字汇总，沿N1-167的事后描述定义，不是新通过门。

| 额外轮数 | TF字节准确率 | 原始NLL bit/byte | 完整汉字TF | 自由生成目标前缀匹配 |
|---:|---:|---:|---:|---:|
| 0 | 24.85% | 4.396 | 2/148 | 0% |
| 1 | 28.87% | 4.164 | 3/148 | 0.20% |
| 2 | 29.62% | 4.080 | 2/148 | 0.20% |
| 4 | 30.60% | 3.952 | 3/148 | 0.20% |
| 8 | 32.10% | 3.775 | 4/148 | 0.39% |

8轮四题TF分别31.53%、32.81%、27.34%、36.72%，没有一题达到≥80%的拟合门；自由前缀最多只匹配一个UTF-8前导byte，不能误称完整首字学会。输出仍为“我我我…”、控制符或不成句拼接。这里的文本观察由助手进行，不冒称用户对新20项作了人工确认。

## 任务错误是否改变了负责模块

参数审计只统计已登记group的weight/bias/values/bridge张量，排除动态plan state、source及RNG。

| 参数组 | 变化tensor/已测tensor | L2变化 |
|---|---:|---:|
| fabric | 9/12 | 0.3845 |
| predictive_context | 1/1 | 1.9146 |
| response_plan_readout（renderer/planner/bridge） | 5/5 | 7.8073 |
| protected predictive_readout | 0/2 | 0 |

本训练配置的实际字节消费者是response-plan readout，因此protected readout不变是路径归属事实，不是“所有模块都不学习”。真实参数改变及NLL/字节准确率提升证明学习不是零；但完整字符和自由生成几乎没有同步改善，不能把变化直接等同于任务能力获得。

## 联合CC-2结论与边界

结合[N1-167](PLAN-N1-167_execution_closeout_20261025.md)：prompt改变实际影响字节分布，训练误差确实改变部分负责参数；当前缺口在“这种表示/目标/局部更新能否变成稳定的条件序列能力”。本次没有支持单纯替换decoder或注入正确plan即可救援，也没有证明所有更大剂量、所有局部学习或整个Taiji架构都不可能有效。

两包模型进程合计253.359375 CPU秒、259.266 wall秒，低于本轮1800/2400；工具/CI/汇总未逐进程精确计量部分留在原预留，不把未用额度当追加训练授权。旧三臂训练、评价和失败成本保留，不归零。

## 唯一后继

进入[N1-169可学条件语言主链候选设计](PLAN-N1-169_shared_language_chain_design_20261025.md)：把现有局部学习链保留为同资源对照，按VISION VX1/VX3设计共享任务状态与序列误差责任明确的新候选。需要比较的是“当前目标/学习规则继续加量”和“主链表示/信用改造”，不是以本次阴性宣布必须推翻整个模型。

N1-168固定8轮已结束，不自动增加到16/64轮；下一步涉及主计算/学习链选择，属于真正的模型路线讨论节点。原parent与zero/final均可恢复，讨论期间不重新训练或接触已消费dev面。
