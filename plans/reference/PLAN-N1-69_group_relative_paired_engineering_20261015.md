# PLAN-N1-69 · 分组位移规则三seed三臂共同起点与控制工程

## 前置与共同来源

68固定校准通过原四门，报告SHA `9484a7abd47bde2d61fcc084af7f64a51e39a95392f006222345fbfec6e30248`。下一步仍先工程：不能把失败seed的定向校准当成三seed正式结果，也不选68终点作共同母档。

按62/63已冻结迁移，从52完整55480参数/35项Adam/history母档新增6576关系参数，共62056；seed11/12/13初始化关系参数，同seed position/terminal/cut三臂完整初态指纹与62原配对指纹一致。52来源与62/63/65/66/67/68及课程/v4字节守卫，68终点只作为校准证据，不加载为训练起点。

## 固定更新规则与控制

67分组实际位移规则完全不变：五互斥组reader排除writer、writer、local、context、relation，relative0.001、floor1.0、完整原Adam moment。三臂都用同一规则，不削弱cut或给position另一个预算。

新增小包装器显式接受arm并调用63原同臂train_step；复用冻结67的bounded_step实现，不改67/68源码以破坏旧证据身份。terminal沿每source末态重复；cut沿63观察writer/局部输入信用截断，writer参数/moment/Adam step保持母档，不能因规则包装重新接通信用。position/terminal保持writer梯度。

## 本69只做合成工程

独立CPU≤180秒，线程1。九fresh零档、九独立新进程恢复，正常保存前不对主模型更新或引入合成session。每个零档主进程及worker各在临时深复制副本上做一次合成下一步：共18次机械副本更新、36条合成QA序列，0主轨迹/正式课程更新；所有副本成本列入同预算。

必验：九全bundle指纹按seed配对；恢复后完整状态/规则marker及同arm下一步行为逐位一致；所有五组位移边界成立；position/terminal合成步writer信用非零，cut writer梯度为零且参数/moment逐位不变、step18432。旧参数和新关系参数的Adam step分别按各臂+1，全部41状态齐全；新增关系零头能更新。未触发边界时同arm两步原Adam与包装器完整bundle逐位相同，由专项合同覆盖。

0正式train/probe teacher/free评分，0v3dev/holdout/v4模型查询；v4只核已有严格LF/CRLF双seal，不生成面或题目变体。错误/预算停止保存报告，不重启、不覆盖旧档。

## 产物

包装器`scripts/training/_taiji_n1_group_relative_paired_adam.py`，审计器`scripts/training/audit_taiji_n1_group_relative_paired.py`，合同`tests/taiji_native/test_native_group_relative_paired_contract.py`；exclusive报告`reports/taiji_n1_group_relative_paired_preflight_20261015.json`，checkpoint目录`data/checkpoints/n1_group_relative_paired_preflight_20261015/`。

## 后续准入与边界

工程通过后另冻70：九fresh模型各6144正式更新，总55296/CPU≤2400秒含18次独立恢复及36次合成签名副本更新；三臂同初始化/数据/目标/上限/次数，唯一差异仍是原读取/信用干预。所有position原四门通过、九模型完整并冻结后，才另冻71首次v4确认。

首次确认沿14原门：每position seed≥16/32、每类≥2/8，三个seed合计各领先terminal/cut≥12/96。任何position训练不足则v4保持零查询，不追加。共享母档的新关系初始化三seed不是端到端独立三seed；C4、旧能力保持、同bundle三器官及N系列其他出口债不变。工程不是未见能力，未推送/删数据/默认推广。

## 执行器提交前验证

新增包装器显式传arm给原63训练函数，原67/68文件不变。10项专项合同通过：九种seed/arm的未触发两步完整bundle逐位中性，以及cut梯度/参数/moment保持和严格规则/阶段恢复。两套仓库Ruff通过，提交后执行九零档工程，尚无正式工程或多seed训练结果。
