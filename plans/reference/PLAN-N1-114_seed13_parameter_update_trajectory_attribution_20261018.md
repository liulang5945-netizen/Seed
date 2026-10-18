# PLAN-N1-114：seed13/position 长度平衡更新轨迹归因

**状态：只读分析已完成（2026-10-18，can_promote=false）。** PLAN-N1-112 显示差异跟随 candidate 权重；PLAN-N1-113 显示最大终态参数差值位于 reader 组，尤其 `readout.weight`。本件把两份已封存 training shard 的 24,576 次更新记录与各自冻结 schedule 对齐，定位长度平衡配方下不同任务/长度/路径的 loss 与五组 Adam proposal 读数。

## 固定输入

- baseline training shard：`reports/taiji_n1_hard_neighbor_training_20261017_s13_position.json.gz`，由 baseline 报告 shard seal 验证。
- candidate training shard：`reports/taiji_n1_length_balanced_training_20261018_s13_position.json.gz`，由 N1-110 报告 shard seal 验证。
- baseline schedule：`plans/manifests/n1_hard_neighbor_training_schedule_v1_20261017.json`，SHA-256 `dc3a96ad17d43bd898e403ecd119c12c781317a71ab06ba39cdab6db24c40d50`。
- candidate schedule：`plans/manifests/n1_hard_neighbor_length_balanced_schedule_v1_20261018.json`，SHA-256 `b2d637fcc1e096d40bba35bbc91bc66b0c27257cd74a4bcac5f5f488f55b6d4c`。
- 只看 seed13/position；按 step 映射 raw/hard path、task kind、材料长度、cycle。核验24,576 loss/measurement、step/cycle顺序与 schedule 全覆盖；transition 项目在两版 schedule 的输入/目标需逐项一致。

## 输出与安全界限

- 对 task/path/material-length/cycle bucket 汇总 QA/observation/total loss，以及 reader/writer/local/context/relation 的 gradient、proposed/actual L2、cap、scale、clipping 读数；并计算相同 task/path/length bucket 的 baseline-candidate 差。
- CPU 上限120秒。报告 write-once：`reports/taiji_n1_seed13_parameter_update_trajectory_attribution_20261018.json`；失败即停，不重试。
- 0 checkpoint payload/model load、0 model construction、0 inference/model query、0 parameter update/optimizer step/checkpoint save、0 dev/holdout/confirmation访问。只读取封存 training shards、schedule/manifest 与 reports。`can_promote=false`。
- 该离线关联不作因果或能力晋升主张；用途是选择下一项训练修订需要守护/复现的参数组和任务路径，不改既有 gate。

## 执行结果

- 报告：`reports/taiji_n1_seed13_parameter_update_trajectory_attribution_20261018.json`，SHA-256 `be5fddc5d62195d9ad706444d25d714801e54baa9cd34b2a3e0031db949bd8ab`；CPU 4.453125秒，读取两份 sealed shard 的 98,304 个 loss/measurement 记录，两个 24,576-step schedule 全覆盖，transition 输入一致6,144步。
- 对相同 transition 输入，candidate 的 QA loss 比 baseline 高：raw +0.02148、hard/frozen-rename +0.02389；observation loss分别约低0.0118/0.0126。reader 组平均 actual-L2变化很小（raw −0.000076、hard −0.000183），但 relation/writer/context 的 clipped-update次数在 transition 步增加383/382/264与420/370/291；显示共享更新轨迹改变，不能单凭相关性判定因果。
- candidate 新增的七字符 raw lookup/revision/unknown 更新平均 QA loss 分别为1.232/1.057/0.710；匹配六字符桶中 lookup/revision QA loss多为上升，unknown则下降或接近。该读数指向长度平衡任务产生了不同梯度/裁剪压力，但不能证明具体哪个任务造成最终 reader 权重偏移。
- 0 checkpoint/model load、0 model query、0参数更新、optimizer step、checkpoint save、dev/holdout/confirmation；source guards全真，`can_promote=false`。attempt 文件保留启动身份；完整结果以报告 `status=complete` 为准。

## 唯一建议的下一步

审议[PLAN-N1-115 五组参数全因子诊断](PLAN-N1-115_five_group_parameter_factorial_diagnostic_draft_20261018.md)。终态参数差值和更新轨迹都指向共享 reader/读出发生大幅变化，但尚无组级输出干预证据；该提案在原训练面做一次参数组交叉，不训练、不改变gate，获准前不发起新查询。
