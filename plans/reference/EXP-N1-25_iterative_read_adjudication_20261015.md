# EXP-N1-25 · 两次内部重读校准裁决（2026-10-15）

用户通过选择弹窗批准24后，按[SPEC25](SPEC-N1-25_iterative_read_execution_contract_20261015.md)完成。代码冻结`55749188e`；参数40,337，与22初始参数逐位相同，保留21辅助目标0.25。仅train，0 dev/holdout。

[预检](../../reports/taiji_n1_iterative_read_preflight_20261015.json) CPU7.28125秒，四类×三臂12行零更新，信用/参数不变/独立zero恢复全部通过。SHA `c8a66d24fc19690392c80add147529c34b8176d91db76f5e0e85599519f8bafc`。

[正式报告](../../reports/taiji_n1_iterative_read_train_20261015.json) SHA `874d06d0452b7e09305232eed3c7d7d95a62f9de82b1971ee35bc1f49f1dff4b`，6144更新、960组评估完整，CPU90.953125秒<300，zero/final新进程恢复和全部源码/数据/预算守卫通过。

| 更新 | 0 | 768 | 1536 | 3072 | 6144 |
| --- | --- | --- | --- | --- | --- |
| train自由exact /192 | 0 | 51 | 66 | 61 | 115 |

终点lookup11/48、revision15/48、transition47/48、unknown42/48；词值26/96，teacher字节90.6818%，CE0.365455，观察保真99.9109%。充分性门≥173/192且各类≥40/48未过。相同初始/数据/更新的22为121/192、词值29/96；本次未观察到总命中提升，不能由单seed判定所有迭代读取无效，也不能宣称机制或能力成功。计算增加，不作同CPU比较。

final checkpoint SHA `40038475198ec67ec6908ef2a3d37d0303b02d710c841ed35acba96c64fafc7f`。保存新格式两hop候选，不替换默认模型、不改旧档语义、不删除历史。

全项目C1工程已过、C2严格收益仍零；C3精确绑定/未见/保持、C4净增益、同bundle三器官及其余N面/远端根CI债未齐。相关62项测试通过（5.58秒），不代表全库或远端CI通过。

唯一下一步按原失败出口执行[SPEC26](SPEC-N1-26_iterative_content_read_diagnostic_20261015.md)同材料替换诊断，先保留负结果，不追加训练或叠新机制。
