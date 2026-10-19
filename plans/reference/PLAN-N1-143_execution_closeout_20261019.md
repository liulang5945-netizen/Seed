# PLAN-N1-143 执行结案：v22新题范围门

**状态：范围阶段完整执行，范围门未通过；confirm未查询；`can_promote=false`。**本批不是确认门失败，而是冻结的先测尺阶段未达到继续条件；v22 seed122面已消费且不重测。

## 执行与完整性

- v22面160题（四类各40），12个endpoint均通过独立完整bundle恢复；source union为29,202个实体名、8,208个词值，来源与checkpoint哈希未变化。
- 12端点各查询80道sacrifice，共960条；逐题结果完整保存在[范围结果报告](../../reports/taiji_n1_v22_confirmation_result_20261019.json)，本地progress由报告SHA绑定。
- 范围阶段CPU 751.891秒，含face准备与12端点readiness后累计901.609/3,600秒；0训练/参数更新、0 dev、0旧v3 holdout查询、0 confirm。

## 冻结范围门

范围分母为每endpoint的60道lookup/revision/transition；20道unknown不计错误底线。需要至少8/12个endpoint各有≥24/60个错误，实际仅6/12通过：

| endpoint | answerable错误 | 是否达到24错误底线 |
|---|---:|---|
| N1-132 seed11 parent | 21/60 | 否 |
| N1-132 seed12 parent | 25/60 | 是 |
| N1-132 seed13 parent | 28/60 | 是 |
| N1-140 seed11 block candidate | 21/60 | 否 |
| N1-140 seed12 block candidate | 6/60 | 否 |
| N1-140 seed13 block candidate | 25/60 | 是 |
| N1-141 seed11 candidate | 24/60 | 是 |
| N1-141 seed12 candidate | 11/60 | 否 |
| N1-141 seed13 candidate | 5/60 | 否 |
| N1-141 seed11 control | 28/60 | 是 |
| N1-141 seed12 control | 9/60 | 否 |
| N1-141 seed13 control | 28/60 | 是 |

因此本计划没有确认结果，也不能对v22表现作泛化或能力主张。N1-141 candidate在seed12/13的v22 answerable错误较少，说明此面未能达到冻结的跨端点范围线；不能用这一现象替代确认门。

## 停止规则与下一步

按冻结的一次性规则，v22面已消费，不重抽、不补查、不放宽8/12或24/60判据。runner无错误、12端点哈希保持一致，停止原因为范围门本身未通过。

依据用户授予的执行计划批准权，已批准[N1-144 v23长上下文确认计划](PLAN-N1-144_v23_long_context_confirmation_after_n143_draft_20261019.md)：新seed123、32行上下文，其他比较端点与判据不变；source union加入整张v22面。该新面独立构造，不续用v22的题或读数。
