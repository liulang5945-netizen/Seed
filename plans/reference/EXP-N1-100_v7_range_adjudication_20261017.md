# EXP-N1-100 · v7 sacrifice range 判读 · 2026-10-17

## 结论

v7 sacrifice 范围执行完整，但未过冻结门：只有4/9端点在15道可答题中出现至少6个错误，低于6/9。按PLAN-N1-100，不查询confirm。本面一次范围执行已记录，confirm attempt和confirm report均不存在。

范围报告 `reports/taiji_n1_hard_neighbor_v7_range_check_20261017.json` SHA-256 `c0bf9e588b3785240c61a0a90907f4849de64b1027546a749d00a8e2768c30eb`；attempt `reports/taiji_n1_hard_neighbor_v7_range_check_20261017_attempt.json` SHA-256 `f45062b2abb8cb43ee026b379f64b7caca610c03923b927a933d7cc1e5f4529c`。180条sacrifice模型查询，CPU38.171875/180秒，零更新。

## 端点错误分布

每类分母为5；`addressable_errors`只合计lookup、revision、transition，分母15，unknown不计入范围门。

| Seed | Arm | Lookup | Revision | Transition | Unknown | 可答题错误/15 | 范围可用 |
|---|---|---:|---:|---:|---:|---:|---|
| 11 | position | 1 | 0 | 0 | 0 | 1 | 否 |
| 11 | terminal | 5 | 5 | 3 | 4 | 13 | 是 |
| 11 | cut | 1 | 0 | 1 | 2 | 2 | 否 |
| 12 | position | 0 | 0 | 2 | 1 | 2 | 否 |
| 12 | terminal | 5 | 5 | 4 | 3 | 14 | 是 |
| 12 | cut | 1 | 0 | 0 | 1 | 1 | 否 |
| 13 | position | 5 | 5 | 0 | 0 | 10 | 是 |
| 13 | terminal | 5 | 5 | 3 | 2 | 13 | 是 |
| 13 | cut | 2 | 0 | 1 | 0 | 3 | 否 |

范围门失败由冻结工具按总错误数计算；每类平均错误只披露、不参与门。可用端点为三个terminal和seed13/position。seed11/12 position和三个cut端点均未达到6/15；不能把控制臂的三枚范围贡献解释为position能力。

## 执行边界

- v7 face SHA-256 `c5a7ef0bc18bf9376e5038348204e838bbc11d88ad6784605c858b12f6d992dc`；只选择`subset=sacrifice`的20项，每个新训练端点20条，共180条。
- 端点、来源和代码守卫未变；完成报告为`range_not_met`，`confirm_subset_queried=false`，`v7_confirm_model_queries=0`，`parameter_updates=0`，`can_promote=false`。
- v7 confirm保留为未查询数据，不能基于本结果宣称C3，不得重抽或把阈值从6/9降为4/9。

## 下一项方案提议（未批准、未执行）

建议另立[PLAN-N1-101 draft](PLAN-N1-101_v8_hard_neighbor_range_recovery_draft_20261017.md)及全新v8 face，保持6/9范围线和原v7四项确认门不变，只将每题观测数由4提高到8并保留d=1、40题/四类平衡及一次性sacrifice→confirm次序。理由是此前PLAN-N1-95/㊵-1035的仓外d=1预飞中，8条观测曾给旧九端点9/9的范围动态；该结果不能预测本轮新模型，v8仍须从零验证，失败则停下，不循环调参。草案只排除v7已查询的sacrifice项，v7 confirm payload继续不读、不查。新face及范围设计属测面变更，等待owner批准；此草案未冻结、未执行。
