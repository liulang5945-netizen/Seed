# 83 四条件目标三种子三臂工程准入（2026-10-16）

82报告SHA `a0dd6e74c669275fbfd7e629e17a8ff7a6122e232bb6797574f7bfaf0a73160e`完整且六训练门通过；终点192/96/192/96、原/改名四条件联合各48/48。仍是fresh seed13已知训练校准，不是未见确认。

## 冻结范围与判据

保持81学习目标/62056参数/五组实际Adam位移规则；seed11/12/13×position/terminal/cut九个全新零态，来自74已封存九zero，仅加载零态，不继承82训练结果。完整参数/RNG、空Adam/历史逐位等于各seed的全新初始化；同seed三臂零态完全相同、三seed不同。

独立CPU≤120秒，单线程，主轨迹0更新、真实课程/dev/v3holdout/v4/v5查询均0。每个零态实际保存后独立进程恢复，父子各在副本执行同臂两次合成transition四条件更新（共36副本步/144合成QA）；完整初末参数/Adam/RNG/history、损失/五组读数逐位一致，原零态不变。position/terminal41Adam，cut37且writer四参数完全不变/无Adam，其他参数步数2，历史8会话；writer梯度在原生臂正、cut为0，关系GRU第二步有限且正。信用读数只算工程可测，不算行为优势。

复核82完整4800行/24576步/55296QA/43008历史/两恢复和六门、原源链及九zero字节；不执行任何已知或未见题。attempt独占，预算/恢复/守卫失败保留部分记录，不覆盖/重试。

## 预声明路径与顺序

`scripts/training/audit_taiji_n1_transition_factorial_paired.py`；`tests/taiji_native/test_native_transition_factorial_paired_preflight_contract.py`；`reports/taiji_n1_transition_factorial_paired_preflight_20261016.json`及`_attempt.json`；忽略目录`data/checkpoints/n1_transition_factorial_paired_preflight_20261016/s{11,12,13}_{position,terminal,cut}_zero.pt`。先纯合成合同/静态检查和提交，再执行。既有74/77/79/80/81/82代码、报告和checkpoint不改。82终点证据不删除。

## 后续唯一方向

通过后另冻84九fresh模型，每模型固定24576更新/55296QA/43008训练会话，全部共221184更新/497664QA；三臂共享同seed完整零态、同父题改名流与四条件目标，终点及所有控制都保留。独立整批CPU上限240分钟（14400秒，依据82单臂1301.89秒×9并留恢复/评分余量），不补训/选点/改门。每position必须通过82六训练门、九模型工程全部完整并冻结，才考虑既有未消费v5首次评估；v5沿14原门，不改样本/阈值，不在83查询。84运行需先有独立预注册/执行器/无损分片及合同并提交。全项目保持/C4/三器官/N系列/完整CI仍欠。

## 工程裁决

完整，CPU37.3125秒；九次独立恢复、36副本步144合成QA、同seed三臂全零态相同/三seed不同、67来源守卫终后字节核验，九新checkpoint字节核验通过；0主轨迹更新/真实题查询。报告SHA 4b157a1eaf2f0ec6cae20eef705a441a408f24a5cc7816dd1c84fa59acb9dd06。进入另冻84/85，尚未正式训练或v5查询。
