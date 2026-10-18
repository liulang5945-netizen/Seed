# PLAN-N1-110 执行结案与下一步（2026-10-18）

**结论：固定训练批次执行完整，但原训练充分性门未通过；本配方关闭，不生成 v11 面、不查询确认集，也不晋升模型。** 原冻结方案保持不变；本页只绑定其执行结果。

## 冻结身份与证据

- 冻结方案：`PLAN-N1-110_length_balanced_training_and_v11_confirmation_draft_20261017.md`，SHA-256 `16d0cecb325af6463c107973fc0e741478140134a7642cd64c86e045c6069783`。
- 训练 schedule：`plans/manifests/n1_hard_neighbor_length_balanced_schedule_v1_20261018.json`，SHA-256 `b2d637fcc1e096d40bba35bbc91bc66b0c27257cd74a4bcac5f5f488f55b6d4c`。
- 静态 readiness：`reports/taiji_n1_length_balanced_training_20261018_runner_readiness.json`，SHA-256 `fea9a11bdce6e712de4f9ea1862cb8d9016ea6118eda68269165d879bfd8a07b`。
- 零 checkpoint 预检：`reports/taiji_n1_length_balanced_training_20261018_checkpoint_preflight.json`，SHA-256 `eab188fb2b1999e5573566326d081b38d38f542b876bfe75aa3975f7981d7ae0`；9/9 独立恢复一致，CPU 55.375 秒。
- 完整训练报告：`reports/taiji_n1_length_balanced_training_20261018.json`，SHA-256 `e848433de597d378e9e1dc52bf488f296f90c5ce67f3855e19bcfe49268aaa0c`。九个 lossless shard、attempt 和 progress 文件与报告一并保留。

## 执行结果

- 九个 seed/arm 端点均完成固定 24,576 更新，共 221,184 更新；全部 18 次零点/终点独立恢复通过，checkpoint、shard 与 source guard 摘要一致。
- 原始/配对/重命名/重命名配对长度账本全部有效；完整报告 `complete=true`、`sources_unchanged=true`、`code_unchanged=true`、`can_promote=false`。
- CPU 总计 9,499.359375 秒（含零 checkpoint 预检），低于冻结上限 18,000 秒。dev、holdout、confirmation 查询均为 0。
- 原训练充分性只对三个 `position` seed 做批次合取：seed11 与 seed12 通过；seed13 未通过。seed13/position 的 `core.calibration.sufficient` 为真，但 **original/rename `joint_four_exact` 仅 5/48、4/48，低于各自 44 门；rename `transition` free-exact 为 38/48，低于 40 门**。因此批次 `training_sufficient=false`。
- 依原方案第 19–20 条，训练门失败后不得生成 v11 范围/确认面；`reports/` 与 `plans/manifests/` 中均未发现 PLAN-N1-110 的 v11 面或确认报告。没有留出查询或能力晋升。

这项结果关闭的是本次固定长度平衡配方，不是否定 N 主线。不能追加训练、重跑或放松门槛来补救该批。

## 训练分片归因

只读归因报告：`reports/taiji_n1_length_balance_failure_attribution_20261018.json`，SHA-256 `08e80702f0b217e117094c28879f4a62e6f99d6a2315ee38556383afa7d31f4a`；CPU 1.96875 秒，复用 1,152 条终点 factorial 评分行，0 checkpoint/model load、0 model query、0 update、0 dev/holdout/confirmation。

- 匹配基线 `reports/taiji_n1_hard_neighbor_training_20261017.json`（SHA `dad10ec72c605367f99da035bab07d522ce1793154915ebcc3c7f1022f411909`）的 seed13/position 在原始与改名 factorial 上均为 **48/48 joint-four**，position 训练充分门通过。N1-110 候选 seed13/position 对应为 **5/48 与 4/48**；free exact=184/192、改名=182/192，改名 transition=38/48。seed11/12 在 N1-110 仍是 48/48。
- seed13 的 48 个父组中，原始视图 43 个恰好四条件中对三项、5 个四项全对；改名视图 44 个三项对、4 个四项全对。原始与改名四项全对的父组交集为 **0**。
- 四条件每一行的 `free_exact` 与 `teacher_exact` 完全相同；错误输出均为合法方向词，非格式/循环文本。错误集中于 `turn_flip`（原始32/48、改名36/48），且 west/left 的 turn_flip 为 0/9、east/left 的 both 为 0/6（按各自实际分母）。

这些是已消费训练面上的 matched 对照，不是未见能力或因果证明。N1-110 只改变 lookup/revision/unknown 的材料长度曝光，transition 训练输入和方向目标保持不变；读数提示新的长度训练可能干扰共享关系表征，但需单独诊断确认。

## 唯一建议的下一步

先审查 [PLAN-N1-111](PLAN-N1-111_position_weight_history_crossover_diagnostic_draft_20261018.md)：在 baseline 与 N1-110 seed13/position 两枚冻结终点间，做零训练的 network-state × observation-history 2×2 交叉归因。它只查询已消费的 48 个 train transition 父组及其四条件原始/改名视图，用以区分错误主要跟随学习权重还是存储的训练观察历史。该项会新增模型查询测面；草案尚未批准，未实现或运行前必须等 owner 选择。
