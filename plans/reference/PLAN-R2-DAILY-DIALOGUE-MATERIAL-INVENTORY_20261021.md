# PLAN-R2-DIALOGUE-05：未消费自然对话材料容量清点

**状态：已冻结并执行。** 本步骤依据R2-DIALOGUE-04审计结论授权；仅做本地数据清点，不训练或生成。

## 目标

确认固定OASST2来源中，在不重用已消费final family、不过滤安全与隐私标签的前提下，能否把当前525个assistant回复episode扩成更完整的全轮对话预训练语料；同时确认仍未使用的生成评估family数量。

## 方法

- 绑定已固定的OASST2压缩源SHA、source revision、当前episode manifest与JSONL SHA，以及R2-DIALOGUE-01/R2-DIALOGUE-03R1生成报告SHA。
- 按`message_tree_id`沿原固定hash规则重建train/dev/final family。train侧复用当前接受规则统计所有有效角色消息、回复rank、字节长度、可恢复父链、可训练新增目标数；不输出原文、用户/消息ID或个人信息。
- 对未查询dev仅读取family标识与计数，不读出消息文本；对已查询final只记录排除计数，不再次解析其内容。将已消耗final82 family排除在任何后续训练候选外。
- 用只读摘要回答：训练family中有多少未用assistant目标与完整会话路径；扩展后估计的字节量相对现行525 episode增长多少；尚未消费的dev family能否作为下一阶段一次性自由生成检查面。若现有许可数据不足，则把“新增安全数据源检索”列为之后的独立范围决策，不自动导入。

## 边界与验收

- 仅本地读取已固定的OASST2归档及其现有派生清单；CPU≤300秒。
- 0模型/checkpoint加载、0生成、0参数更新、0dev模型查询、0final内容读取、0新语料写入、0checkpoint写入。
- 仅输出计数、长度分布、哈希与家族层摘要；禁止把原始对话文本、消息ID或个人信息写进报告。
- 明确给出可用的训练候选增量、可保留的新生成family数量及所有排除边界；该结果只用于下一份训练合同，不是训练授权或能力证据。

## 产物

- 机器可读清点：`reports/taiji_r2_daily_dialogue_material_inventory_20261021.json`
- 中文结案：`plans/reference/PLAN-R2-DAILY-DIALOGUE-MATERIAL-INVENTORY_20261021_execution_closeout.md`
