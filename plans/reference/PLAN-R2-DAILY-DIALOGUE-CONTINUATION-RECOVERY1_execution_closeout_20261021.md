# PLAN-R2-DIALOGUE-03R1执行结案：匹配续训自由生成日志恢复

**状态：完成；`can_promote=false`。** 原训练批次与恢复生成分开留档，原 partial 报告未覆盖。

## 结果

- 多轮candidate与单轮control均从R2-DIALOGUE-01的2-epoch终点完成6个额外epoch；每臂3,150个episode、679,254次响应字节更新，合计1,358,508次。两臂各累计8个epoch，最终`global_step=906746`、`episode_count=4224`。
- 训练CPU 5,187.078125秒；日志恢复、六端点检查点恢复与420次已保存自由生成后，累计CPU为5,536.859375/30,000秒。
- 两枚新final full bundle、两枚旧final和两枚父档zero共6/6通过独立进程恢复。生成前后checkpoint digest一致；输入来源哈希守卫通过。
- 固定70棵未消费final对话树、6个端点，共保存420条生成记录（每端点70条；55条无历史、15条有历史）。首次生成在写日志前触发字段键错误，首棵树6个输出未落盘；恢复重放该树并完成其余69棵，所以物理生成调用426次，报告只将恢复后持久化的420条作为证据。未读取参考回复、不做exact-match或teacher-forced评分、不查询dev。

## 自由生成读数

| 端点 | end-marker率 | 输出唯一数 | 重复4-gram率 | 平均字节 |
|---|---:|---:|---:|---:|
| 父档含历史 | 5.7% | 69/70 | 24.44% | 485.0 |
| 父档单轮 | 5.7% | 69/70 | 24.64% | 485.0 |
| 2-epoch多轮 | 50.0% | 59/70 | 27.31% | 344.7 |
| 2-epoch单轮 | 50.0% | 50/70 | 24.92% | 302.6 |
| 8-epoch多轮 | 90.0% | 69/70 | 8.51% | 315.9 |
| 8-epoch单轮 | 100.0% | 69/70 | 10.65% | 219.5 |

以上是可见格式与重复统计，不表示语言理解或对话能力。人工检查预先取出的12个样例，8-epoch多轮候选12/12均未形成连贯、贴题的自然回复；常见输出为字节碎片、特殊标记与无序汉字。虽然停止标记和重复率改善，但对话语义没有达标，不能把这次训练描述为日常交流能力提升。

## 故障与修复记录

原续训完成两臂训练和checkpoint保存后，在第一棵新树的6个输出全部生成后、进度写入前，logger以内部键`parent_dialogue_context`索引名为`parent_with_history`的样例字段，触发`KeyError`。原attempt/result/progress保留，partial result SHA-256为`2a2e2e7a9c7d4993d4fdf56a1b6d5ad6b9e0f3a94b75f846f2b0c0f1f11086ea`。

修复在[续训runner](../../scripts/training/run_taiji_r2_daily_dialogue_continuation.py)统一端点到样例字段的映射，并新增[回归测试](../../tests/taiji_native/test_r2_daily_dialogue_continuation_contract.py)。恢复由[独立runner](../../scripts/training/run_taiji_r2_daily_dialogue_continuation_recovery.py)执行；它在每个端点输出后立即写progress，避免整行缓冲导致再次丢失。

## 判读与下一步

当前H3.7B原生语言通路在这组自然中文材料上完成8个epoch后，能更稳定地产生停止标记，却仍不能生成可读回复。继续对同一批材料机械叠加epoch不符合[原冻结方案](PLAN-R2-DAILY-DIALOGUE-CONTINUATION_20261021.md)的停止边界。唯一下一步是只读梳理实际加载的base checkpoint、参数/字节表征、训练目标与生成解码路径，确认模型容量、初始化来源及数据到输出的转换是否匹配自然中文对话；审计完成前不再启动同语料训练。此举不作模型能力晋升，也不消费新测面。

## 绑定产物

- 执行恢复方案：[PLAN-R2-DIALOGUE-03R1](PLAN-R2-DAILY-DIALOGUE-CONTINUATION-RECOVERY1_20261021.md)
- 原失败partial：[报告](../../reports/taiji_r2_daily_dialogue_20261021_continuation_result.json)
- 训练恢复生成：[结果](../../reports/taiji_r2_daily_dialogue_20261021_continuation_recovery1_result.json)
- 逐题自由生成：[生成记录](../../reports/taiji_r2_daily_dialogue_20261021_continuation_recovery1_generations.json)
- 人工检查样例：[review](../../reports/taiji_r2_daily_dialogue_20261021_continuation_recovery1_generation_review.md)
