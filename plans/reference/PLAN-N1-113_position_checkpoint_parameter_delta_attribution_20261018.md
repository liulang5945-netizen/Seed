# PLAN-N1-113：seed13/position checkpoint 参数组差异归因

**状态：只读分析已完成（2026-10-18，can_promote=false）。** PLAN-N1-112 显示退化跟随 candidate 权重来源；本件比较两枚既有 final checkpoint 的终态参数差异，为训练修订定位变化最大的共享参数组。只读 checkpoint 参数，不改变模型、不生成答案、不触碰新题面或门槛。

## 输入与边界

- baseline：`data/checkpoints/n1_hard_neighbor_training_v1_20261017/s13_position_final.pt`，SHA-256 `09df1d47eb490ef9afc5d563d08b40eca6b51951096875d016a67f133f7ed653`。
- candidate：`data/checkpoints/n1_hard_neighbor_length_balanced_training_20261018/s13_position_final.pt`，SHA-256 `d215a51d571976f216af43d0b7372f3cb702b6258237a7144252df4609f3b881`。
- 绑定训练报告：baseline `dad10ec72c605367f99da035bab07d522ce1793154915ebcc3c7f1022f411909`；candidate `e848433de597d378e9e1dc52bf488f296f90c5ce67f3855e19bcfe49268aaa0c`。
- 以 frozen N1 五组 updater 映射 reader/writer/local/context/relation；要求41个参数 tensor、62,056个标量、键/形状/dtype逐项相同。
- 每组报告 baseline/candidate L2、差值 L2、相对差值、余弦相似度、非零变化标量数及差值最大的 tensor。整体验证五组平方和与全部参数差值范数闭合。
- CPU≤60秒；最多读取2个 checkpoint payload，不恢复 optimizer、session history 或模型，不执行 forward、measure、answer、训练/optimizer step或 checkpoint save。0 dev/holdout/confirmation 查询，`can_promote=false`。
- 输出 write-once：`reports/taiji_n1_position_weight_checkpoint_parameter_delta_s13_20261018.json`；失败即停止，不重试。此终态差异是描述性定位，不能单独作因果或能力晋升主张。

## 来自 PLAN-N1-112 的边界

PLAN-N1-112 主报告 `e38b08a857ffe69acb8fbf57e040ed672df538b88988dc333e1182e58ffb16bf` 的原始/改名 joint-four：baseline weights 两种 history 都48/48；candidate weights 两种 history 都5/4。两种 weights 下的全部384行预测均随 history 互换保持逐行相同。另有静态代码路径和离线记录表明本轮 evaluator 以新 session key 读取输入，本 checkpoint 的其他 session observation 不可达；这仅限定该 evaluator，不能泛化为所有运行模式的记忆结论。方向×转向 cell 读数以 PLAN-N1-112 [更正报告](../../reports/taiji_n1_position_weight_history_crossover_s13_r1_cell_summary_correction_20261018.json)为准。

## 执行结果

- 报告 `reports/taiji_n1_position_weight_checkpoint_parameter_delta_s13_20261018.json`，SHA-256 `292647be402a0fadecdb63e6d2886743d08fb2a1afd1ddffe7cbbc85175c8281`；CPU 3.09375 秒。读取2份 checkpoint payload，0 model load/construction/query、0 update/step/save、0 dev/holdout/confirmation；checkpoint 前后哈希与11项 source guards 全相同。attempt 文件是不可变启动记录，保留 `status=running`；最终完成状态以此报告的 `complete=true` 为准。
- 41个 tensor／62,056个参数中58,424个标量不同；baseline 与 candidate 的全参数 delta L2 为58.8288、cosine similarity 0.8419。reader 组 delta L2=54.2394，占总 delta 平方范数85.0%；`readout.weight` 单 tensor delta L2=47.288，占总平方范数64.6%。writer/context/relation/local 组差值依次更小。
- 该差值定位不是因果证明。它与 PLAN-N1-112 的读数一致：candidate 训练后的权重状态伴随 transition 性能回退；需要更新轨迹读数再设计训练修复。

唯一建议的下一步：[PLAN-N1-114](PLAN-N1-114_seed13_parameter_update_trajectory_attribution_20261018.md) 已完成，只读分解 baseline 与 length-balanced candidate 的 24,576 步、五组参数更新日志；cell/权重审计显示 reader 组（尤其 readout.weight）差异最大。下一步转为裁定 [PLAN-N1-115](PLAN-N1-115_five_group_parameter_factorial_diagnostic_draft_20261018.md)，测试五组参数组合对原训练面输出的贡献与交互，不训练。
