# PLAN-N1-67 · 五组实际Adam相对位移约束工程

## 证据与唯一修订

66精确复现了63 seed13后程退化，并记录到一次local扰动后多步持续的Adam位移；尚未证明位移约束必能解决问题。唯一候选修订为实际Adam步的分组信赖域，保留62056参数、全部原生观察信用、原Adam moment/lr和全部训练样本；不冻结writer、不使用gold地址、不选中点。

五个互不重叠组为reader（排除writer）、writer、local、context、relation；分组沿原parameter_group，仅把writer四参数从reader拆出。每步先计算原loss/原梯度并执行原Adam提议；对每组保存更新前参数θ及提议Δ，若`||Δ||₂ > 0.001 * max(||θ||₂, 1.0)`，仅把该组实际参数位移整体缩放到该上限。未触发时保持Adam提议逐位不动。原一阶/二阶moment和step仍由未裁梯度完整更新，不伪造moment；不裁loss或梯度，不增加学习参数。

固定relative上限0.001与norm floor1.0，不搜索其他阈值。组整体norm避免把新零初始化关系投影永远锁为零；全部组和全部参数仍可学习。该候选是可证伪的优化器修订，不代表已确立的因果修复或更强能力。

## 本67只做工程

独立CPU≤60秒，包含前置、最多两次合成train机械更新、zero/final保存及两个新进程独立恢复。源为63 s13_position_zero.pt（通过63冻结报告SHA `d982909f6722d37040fb21f7d869aed6817f3a42a3c4b08300b6746ef14e2922`绑定）。先zero保存与独立恢复再机械更新；0正式课程评分/训练，0dev/holdout/v4。

必验：五组互斥且覆盖全部41参数；足够大上限时完整bundle两步逐位等同原Adam；触发时实际位移满足固定边界，原梯度和Adam moment在同起点同一步与原Adam逐位相同；新关系零头可更新、早期writer信用仍通；RNG无额外消耗；自身zero/final参数/Adam/RNG/history、更新规则marker及下一步机械行为独立恢复一致，缺失/错误规则marker拒绝。不修改旧checkpoint或63/66源码，不改默认模型。

实现helper放`scripts/training/_taiji_n1_group_relative_adam.py`，工程审计器放`scripts/training/audit_taiji_n1_group_relative_adam.py`，合同放`tests/taiji_native/test_native_group_relative_adam_contract.py`，报告放`reports/taiji_n1_group_relative_adam_preflight_20261015.json`，checkpoint目录为`data/checkpoints/n1_group_relative_adam_preflight_20261015/`。exclusive、不覆盖旧档；失败保存证据，不循环追加。

## 后续与全项目边界

工程通过后另冻68：从同63 seed13零档进行新配方固定6144更新、CPU≤300秒含zero/final独立恢复，仍按原四门判断，不追加；中点不作可选档，v4始终零查询。此失败seed的训练校准不是三seed稳健性，正式九模型与首次确认仍须重新冻结前置。67不授权自身执行正式6144更新。

本优化器候选保留全部表示容量与信用，目的是允许长期训练同时限制一次实际步长。C4真实收益、旧能力保持、同bundle三器官、端到端独立seed和N系列其他出口及历史CI债仍欠；不得以这次工程或已见校准替代完整目标。

## 执行前机械计账与数值容差澄清

“最多两次”指67主工程轨迹；为验证zero/final恢复后的下一步实际行为，每档主进程签名和独立worker各在临时深复制副本走一步，另外固定4次。因此本工程总计6次合成机械optimizer更新（主轨迹2＋签名副本4），全部计入同一60秒CPU，不是课程训练，不触及192项或新确认面；报告分别明确两类计数，不隐去副本成本。单元合同中的参考副本另属工程测试，不冒称本正式预检轨迹。

实际位移的数学上限不变，浮点copy后的机械验证仅容许`1e-6 + 1e-4*cap`舍入误差。规则marker绑定固定0.001/floor1.0/互斥五组与未裁moment语义；载荷缺失或阈值不符拒绝恢复，不能恢复为普通Adam而丢失更新规则。

## 完整裁决

CPU16.453125秒，主轨迹2步/副本4步、两独立恢复及全部守卫通过。五组覆盖41参数，位移边界、writer信用/关系头学习、完整状态与下一步受约束行为均过。见[67裁决](EXP-N1-67_group_relative_adam_adjudication_20261015.md)；唯一下一步[68固定校准](PLAN-N1-68_group_relative_adam_calibration_20261015.md)，不从67合成档续接，仍0新确认查询。
