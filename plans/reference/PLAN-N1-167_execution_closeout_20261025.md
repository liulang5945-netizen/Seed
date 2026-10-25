# N1-167训练侧decoder归因结案

日期：2026-10-25。状态：**CLOSED_COMPLETE**，完成固定矩阵与阴性/有界结论。原[PLAN](PLAN-N1-167_train_decoder_attribution_20261025.md)/[SPEC](SPEC-N1-167_train_decoder_attribution_20261025.md)在`5616a8a3c`先冻结；本包未训练、未读dev、未改原checkpoint。

## 实际完成

4个固定train样本×4端点×3路径=48/48，均适用；另有16项原训练路径只读重放、16项循环错配prompt TF及16项仪器/实际configured生成核对。全部configured输出逐字节一致，四端原文件SHA/模型digest不变。完整进程实测134.46875 CPU秒、137.125 wall秒，未触发硬停止。

原始矩阵、逐位置TF和生成见[结果](../../reports/taiji_n1_plan167_train_decoder_result_20261025.json)，可复算汇总见[统计](../../reports/taiji_n1_plan167_train_decoder_analysis_20261025.json)，完整退出费用见[回执](../../reports/taiji_n1_plan167_process_exit_20261025.json)。`oracle_plan`显式使用train答案信息，仅是动态状态上界诊断，不是自主能力。

## 固定读数

TF字节准确率是4题宏平均；汉字列是完整汉字全部UTF-8字节同时正确、在148个完整目标汉字上的合计。汉字指标为事后描述统计，不新增预注册门。

| 端点 | configured TF字节 | protected TF字节 | oracle TF字节 | configured完整汉字TF | 自由生成目标前缀匹配 |
|---|---:|---:|---:|---:|---:|
| parent | 24.80% | 0.81% | 26.36% | 5/148（3.38%） | 0.20% |
| full_turn_1m | 24.85% | 25.30% | 23.49% | 2/148（1.35%） | 0% |
| warm_start | 21.86% | 20.14% | 22.22% | 1/148（0.68%） | 0.20% |
| fresh_same_capacity | 21.93% | 21.46% | 23.06% | 3/148（2.03%） | 0% |

- configured TF原始NLL约4.30–4.58 bit/byte，UTF-8掩码后约3.98–4.18；字节统计已明显区别于均匀分布，但不能据此说句子学会了。
- 16项都不满足预注册“高TF/低rollout”，0项oracle救援、0项protected救援；所以本次没有支持“仅更换读出/给正确plan即可救援”的证据，也不能只归咎于滚动误差。
- 16项prompt替换都实际改变分布，平均逐位置JS约0.0076–0.0558 bit、top1一致率约0.680–0.867；0项符合近乎不敏感线。不能判“prompt完全没有进入输出路径”，也不证明内容已经正确使用。
- 各相位都低，未观察到只在64字节之后单独崩溃的简单切点。助手检查train输出仍为空答、控制串或无词法结构的拼接；这里不是新的用户人工判读或独立泛化成绩。

## 静态学习责任与剩余不确定性

代码显示prompt经`_prime(learn=False)`进入动态motor context；`begin_response_plan`由该context生成plan，训练时`learn_response_plan_target`及逐byte误差可写planner/bridge，配置还允许fabric/predictive-context/readout学习。故不能仅凭`grad=None`或本包0更新推断没有学习通路。是否确实学得动，仍需受控更新验证。

源配方的全轮预训练是8轮，而assistant条件SFT仅1轮、525 episode/113209目标字节；两者不是同一种剂量。此次4题拟合差，既不证明“完全没有学习”，也不足以决定核心必须推倒、容量无效或旧配方排序。

## 唯一后继与自动接力

自动接[隔离小集学习充分性N1-168](PLAN-N1-168_isolated_train_fit_20261025.md)：选最大full_turn_1m保留上限，仅新副本重复原4题固定8轮，在0/1/2/4/8轮观察；先完整zero保存/新进程恢复。32额外episode/9320目标字节更新，不动原档、不碰dev、不追加本轮总资源。它回答现有学习链是否能学会少量已知序列，再据结果决定剂量/信用/主链改造；不直接给整个架构判死刑。

N1-167的48单元、0更新及阴性结果至此冻结；N1-168是独立预注册子包，不能将其学习收益回填N1-167。新核心/新学习算法、继续增量或额外评价范围如成为必要决策，才停下来讨论。
