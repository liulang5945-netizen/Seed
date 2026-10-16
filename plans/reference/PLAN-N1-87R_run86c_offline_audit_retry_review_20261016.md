# PLAN-N1-87R run86c 离线诊断单次修复重跑（已授权，2026-10-16）

## 变更理由

已批准的PLAN-N1-87首次启动在Python顶层导入阶段失败：直接执行`scripts/training/`内脚本时未将仓库根加入`sys.path`，触发`ModuleNotFoundError: No module named scripts`。失败发生在读取任何报告、分片或训练样本之前；0模型/checkpoint加载、0评估/查询/更新。失败回执保留于`reports/taiji_n1_transition_factorial_run86c_offline_trajectory_audit_attempt.json`。原PLAN-N1-87规定单次尝试且不重试，因此本提案单独申请一次修复后重跑。

## 仅请求的范围修订

- 将PLAN-N1-87总尝试数由1次改为最多2次：第一次失败在输入读取前；批准后只允许再执行1次。
- 执行器修复仅在导入前把仓库根加入`sys.path`，其余输入、输出、指标、窗口、解释限制和分析代码保持不变。 执行器SHA256 `47012d046433a3ffc901cd35379b5884c86016c030a2d3204f87f56a59e9d544`；不一致即停止。
- 本机CPU≤60秒；所有者已授权自行承担此类CPU资源成本，无需另行询问。
- 第二次无论成功或失败即终止；不得第三次运行、不得更换报告、分片、CPU上限或诊断问题。

## 固定边界

仍只读取PLAN-N1-87列明的run86c报告、九封存分片和训练样本顺序；不加载模型/checkpoint，不做模型评估、推理、训练更新或dev/holdout/v3/v4/v5查询。原报告输出路径若已存在则停止，不覆盖。训练配方、预注册判据、测面、85准入状态均不改变，`TRAIN_SHA=None`、v5未消费、`can_promote=false`。

## 状态

已由所有者于2026-10-16明确授权执行：可撤销的计划调整无需逐项等待；因此本计划允许额外一次离线审计重试。授权不含远端推送或数据删除。执行后即按结果结束本计划，不再自动尝试第三次。

## 执行结果

唯一获批重跑完成：CPU3.015625秒，9封存模型、216窗口完整；模型/checkpoint加载、模型查询、评估、optimizer/参数/签名更新及所有未见面查询均为0。详见`reports/taiji_n1_transition_factorial_run86c_offline_trajectory_audit.json`，SHA256 `c5b21641de584d6e871ad4747ce8e181205e790a886c7b58bbb1305c9e46b2a0`。不再执行第三次审计。结果及唯一下一步见PLAN-N1-87与run86最终裁决追加段。
