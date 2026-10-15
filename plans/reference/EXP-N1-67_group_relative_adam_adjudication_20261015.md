# N1-67 · 分组实际Adam位移工程裁决

工程通过，CPU16.453125秒；主合成轨迹2次机械更新、签名临时副本4次，两个独立进程恢复通过。0正式课程评分/训练、0dev/holdout/v4查询。

五组互斥覆盖全部41参数：reader20、writer4、local6、context5、relation6，62056参数不变。相对上限固定0.001、norm floor1.0；合成两步均实际触发writer/local/context/relation缩放，reader未触发；原梯度与moment在投影时保持，关系零头能更新，writer梯度保持非零。zero/final完整状态、严格规则marker和下一步受约束更新在独立worker逐位恢复，来源/源码/CPU计账守卫通过。

三项专项合同另外证明：未触发上限的完整两步bundle与原Adam逐位相同；同起点的触发步原梯度和完整Adam state与原Adam逐位相同；错误/缺失规则marker拒绝。两套仓库Ruff及项目身份检查通过。此为工程证据，不证明训练稳定性或新能力。

报告`reports/taiji_n1_group_relative_adam_preflight_20261015.json` SHA256 `88313caa6a27dc9a7dfcb2340b15f1e009526fd0afa82db1e60a9bf6362d07ff`；checkpoint为`data/checkpoints/n1_group_relative_adam_preflight_20261015/zero.pt`及`final.pt`，保留，不作正式训练起点。

唯一下一步[68固定校准](PLAN-N1-68_group_relative_adam_calibration_20261015.md)，回到63原seed13零档重新执行6144新配方更新，保持原四门。63失败、64不准入和全项目出口债不变。
