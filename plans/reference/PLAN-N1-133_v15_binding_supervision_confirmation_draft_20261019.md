# PLAN-N1-133：N1-132续训包的v15泛化确认

**状态：2026-10-19已获owner授权并冻结。** 用户选择了“执行包级泛化验证”。本计划只检验N1-132续训包能否在新v15题面通过冻结的泛化门，不单独归因绑定损失的因果作用；N1-132候选比N1-120母档多9,216次续训，且没有匹配的无`L_bind`续训对照。

## N1-126至N1-132前序证据收束

- **N1-126–128：** 在已消费的v13/v14同一批题和六个N1-120 position/cut终点上做了金标准目标行干预。仅保留目标实体行时，position达到90/90、cut达到86/90；保留全16行并遮蔽解码器/地址后为position 82/90、cut 78/90；再限制问题绑定后回到90/90与86/90。目标行来自金标准，且行裁剪改变上下文，因此这些是oracle上限，不证明模型能自主找行。详见[126](PLAN-N1-126_execution_closeout_20261019.md)、[127](PLAN-N1-127_execution_closeout_20261019.md)、[128](PLAN-N1-128_execution_closeout_20261019.md)。
- **N1-129/130：** 原生绑定确实给目标事实行分配了高于字节占比的注意质量，但提升有限，也不能区分正确与错误输出；剩余cut错误仍同时关注两个目标关系。因此现有读数不支持“只因漏检事实行而答错”的解释。详见[129](PLAN-N1-129_execution_closeout_20261019.md)和[130](PLAN-N1-130_execution_closeout_20261019.md)。
- **N1-131：** 四个N1-128 cut残错均为seed13的反向转向输出；同一oracle绑定下position为30/30正确。这定位的是oracle干预下的转向组合错误，不能当作原生检索或泛化证据。详见[131](PLAN-N1-131_execution_closeout_20261019.md)。
- **N1-132：** 在三枚N1-120 position母档上追加`0.25 * L_bind`及每seed 9,216次更新；三seed均通过原训练门（original 192/192、paired 96/96、renamed 192/192、renamed-paired 96/96），共27,648次更新。没有dev、v13、v14或新确认查询，`can_promote=false`。这证明训练集拟合通过，不证明新题泛化，也不能单独证明`L_bind`造成收益。详见[N1-132结案](PLAN-N1-132_execution_closeout_20261019.md)。

所以，本计划的主问题是“N1-132续训包能否在新题面泛化”，不是“`L_bind`单独是否产生因果增益”。若后续需要后一个结论，须另立匹配无`L_bind`、相同9,216次更新的训练对照方案。

## 冻结题面与终点

- 只生成一次seed115的v15题面：40题，lookup、revision、transition、unknown各10题；每题16条观测，实体邻近距离`d=1`；20题sacrifice、20题confirm，每部分每类各5题。
- 完整排除v3训练、已消费的v5–v14题面、N1-95 scratch题面及其派生排除清单中的实体名和材料/答案值；将来源并集、生成器版本和随机种子写入manifest。发现路径已占用、来源重叠或完整性检查失败时立即停止，不另抽seed、不重试。
- 为保持与已消费v14长上下文面的构造可比，lookup/revision答案长度使用6/7字符；每类sacrifice按6字符3题、7字符2题，confirm按6字符2题、7字符3题。目标证据起始索引沿用v14的早/中/晚覆盖：lookup `[0,4,8,12,15]`；revision/transition `[0,4,7,11,14]`；unknown无目标行。此项固定题面构造，不改变40题规模或任何评分门。
- 来源并集静态核验已完成：[最新预授权来源并集报告](../../reports/taiji_n1_plan133_source_union_precheck_v2_20261019.json)在冻结v14继承的答案长度与目标位置分布后重新绑定当前计划SHA。N1-124前排除6,678个实体名与5,328个词值，88个来源哈希均匹配；v14面新增630个实体名、480个观测材料值和25个答案值，其中5个通用答案值已在旧排除集。故v15生成前应绑定的完整并集为7,308个实体名、5,808个词值；最终builder仍须重新计算并逐项校验，任何计数/哈希不符即停止。初次预检留作历史回执，不作为当前计划SHA绑定证据。
- 冻结12个终点：N1-132 position候选seed11–13三枚；同seed N1-120 position母档三枚；N1-120 terminal与cut控制各三枚。face生成前再次独立恢复完整bundle并核对checkpoint SHA与N1-132来源对账。
- 已完成的授权前只读核验见[预授权终点完整性报告](../../reports/taiji_n1_plan133_pre_authorization_endpoint_integrity_20261019.json)：12/12当前checkpoint哈希匹配既有恢复回执。N1-120主报告因通用回执校验器要求有限Adam提案证据而保持`stopped`；独立绑定的revalidation为complete、18/18恢复相等，position三seed训练门通过。terminal/cut在此只作比较控制，不要求通过position训练门。

## 一次性查询与冻结判据

1. face与来源readiness通过、12个终点恢复完成后，只在9个主比较终点（N1-132 position三枚、N1-120 terminal/cut六枚）查询20题sacrifice，共180条。15道lookup/revision/transition题构成range分母，5道unknown不计入该分母。至少6/9终点各有至少6个错误才继续；否则停在range门，不查询confirm，不重抽。
2. 若range门通过，在全部12个终点对20题confirm各查询一次，共240条；含sacrifice最多420条。不得重训、重抽、重试或查询dev。
3. 保留N1-121的数值确认门用于N1-132候选：每seed exact至少10/20；每个类别至少2/5；三seed合计position正确数分别比terminal、cut高至少12/60。由于本计划的终点队列不同于N1-121，这里沿用其数值门，不宣称队列完全复刻。
4. 对同seed N1-120 position母档报告逐题配对的正确转错/错误转正确，按seed和类别分层，仅作描述性读数，不增加通过门。
5. readiness、题面构造、恢复、range/confirm与最终哈希核验累计CPU≤1,200秒。无论结果如何`can_promote=false`；本地门通过不等于全项目能力晋升。

## 结果解释与停止规则

- range门失败：记录180条sacrifice后停止，confirm为0，关闭本计划，不重抽。
- range门通过但确认门任一项失败：记录全部240条confirm并关闭为未通过，不追加训练、不重测。
- 所有冻结门通过：只说明N1-132续训包通过本次v15泛化确认；不证明`L_bind`单独的因果效应，不证明原生转向组合已解决，也不自动晋升模型。

## 授权与输出

用户于2026-10-19明确批准包级泛化验证及这张新v15测面。此次授权不包括训练或修改门槛。

- 新建source-exclusion与face builder、face manifest、readiness/恢复报告和range报告，分别放入`plans/manifests/`、`scripts/training/`与`reports/`。
- 只有range门通过才生成confirmation报告；未通过时该报告必须明确为未执行。
- 新增执行结案写入`plans/reference/PLAN-N1-133_execution_closeout_20261019.md`；计划与结果说明使用中文，机器字段保留仓库schema。
