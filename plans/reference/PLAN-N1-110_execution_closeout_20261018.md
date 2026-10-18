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

## 唯一建议的下一步

对九枚已封存训练 shard 做一次只读逐题归因，重点比较 seed13/position 与 seed11/12 的 48 个四条件父组，定位 joint-four 与 rename-transition 的错误集中在哪些题型/条件；仅分析既有训练评分行，不加载 checkpoint、不生成新答案、不训练、不读 dev/holdout。用报告摘要 SHA 和三个 shard SHA 绑定结果，再据证据提出下一版训练方案。
