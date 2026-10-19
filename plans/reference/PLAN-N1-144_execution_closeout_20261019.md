# PLAN-N1-144 执行结案：v23长上下文新题确认

**状态：执行完整，确认门未通过；v23题面已消费，`can_promote=false`。**范围门按原门槛通过，故按冻结协议查询了确认集；最终候选只通过5项合取门中的1项。本结果不支持能力晋升，也不替代项目级出口门。

## 执行完整性

- 按冻结seed123生成160题v23面，四类各40题，每题32条观察；source exclusion并集含31,722个实体名和10,128个词值。v22新增2,520个实体名、1,920个词值与此前并集零交集；380项来源哈希未变。
- face合同通过，生成阶段CPU 6.547秒，未加载模型、恢复checkpoint、训练或查询。12/12个N1-132/140/141完整bundle随后独立恢复并通过哈希核对；readiness CPU 149.266秒，累计155.813/7,200秒。
- 12端点各查询80题sacrifice和80题confirm，共1,920条查询。执行CPU 2,280.375秒，累计2,436.188/7,200秒。范围和确认汇总均由完整逐题行独立重算一致；attempt/progress SHA与结果报告匹配，来源/checkpoint哈希复核不变。
- 0参数更新、0 optimizer step、0 dev查询、0旧v3 holdout查询；查询期间无runner错误。详细逐题数据见[完整结果](../../reports/taiji_n1_v23_confirmation_result_20261019.json)，进度明细由结果报告SHA绑定。

## 冻结范围门

每端点以60道lookup/revision/transition为分母，须至少24题错误；至少8/12端点过线。实际8/12，恰好达到继续门槛。

| endpoint | 错误数/60 | 是否过线 |
| --- | ---: | --- |
| N1-132 seed11 parent | 30 | 是 |
| N1-132 seed12 parent | 32 | 是 |
| N1-132 seed13 parent | 34 | 是 |
| N1-140 seed11 block candidate | 28 | 是 |
| N1-140 seed12 block candidate | 11 | 否 |
| N1-140 seed13 block candidate | 30 | 是 |
| N1-141 seed11 candidate | 31 | 是 |
| N1-141 seed12 candidate | 19 | 否 |
| N1-141 seed13 candidate | 15 | 否 |
| N1-141 seed11 control | 34 | 是 |
| N1-141 seed12 control | 23 | 否 |
| N1-141 seed13 control | 30 | 是 |

## 冻结确认门

N1-141 candidate三seed各须≥40/80、每类各≥8/20；合计还须分别领先N1-141 control和N1-132 parent至少48/240、领先N1-140 block candidate至少24/240。

| N1-141 candidate | lookup | revision | transition | unknown | 总正确 |
| --- | ---: | ---: | ---: | ---: | ---: |
| seed11 | 0/20 | 20/20 | 9/20 | 20/20 | 49/80 |
| seed12 | 1/20 | 20/20 | 18/20 | 20/20 | 59/80 |
| seed13 | 1/20 | 20/20 | 18/20 | 20/20 | 59/80 |

三seed candidate合计167/240；N1-132 parent为144/240，N1-140 block candidate为168/240，N1-141 control为157/240。候选相对三者分别为+23、−1、+10，均未达到冻结优势线。候选每seed总正确门通过，但每seed每类门因lookup为0/1/1而失败；最终五项合取门仅一项通过，`confirmation_pass=false`。

因此，本轮支持的结论仅是：在这张32行v23确认面上，候选包未达到预注册比较门；lookup表现尤其低。不能据此断言所有长上下文、所有读出机制或整个模型能力均失败，也不能据此自动改变训练目标或确认门。

## 后续

依据用户授予的执行计划批准权，已批准并冻结[PLAN-N1-145](PLAN-N1-145_v23_itemwise_error_attribution_draft_20261019.md)：只对本报告已消费的960条confirm行做逐题、任务类型和目标位置的描述性错因归因，不加载模型、不新增查询、不训练、不改门。该分析属于post-hoc探索，不能充当新的泛化确认。

## 绑定证据

- 冻结计划SHA-256：`b5527b5f8a56aa12604a7dbc40dad1fa74c3f4ea8e953a6b2539f1fa3c5fcf42`
- v23 face SHA-256：`e0ddf062e3c5f081dd010db3a8951d9aa426d9717de39872527a4eeabb76f6ae`
- attempt SHA-256：`a9fa6a8526506114ebd1ae8956c69c4373434b639621c8f96592d45d2fb1d360`
- progress SHA-256：`587664ccefac4b1021b09fe388a924af9795c60bbd16d5fda96b1c519a075b9e`
- 结果SHA-256：`67038a9848b3c7ac088fea1983d9ba40ea153e91c8a4c436db830e8894e80388`
