# PLAN-N1-152：N1-150长度响应逐题配对归因

**状态：已依据用户授予的执行计划批准权批准并冻结。**N1-151A已恢复六个endpoint完整逐题train-only评分台账，N1-150长度结果可逐题对齐。本计划仅读已封存的N1-151A result，不加载模型/checkpoint、不追加查询。

## 固定分析面

- 只读取N1-151A结果中的1,728条`length_score_row`（6 endpoint×8/16/32行各96项）。每个endpoint、长度、task kind必须各有24项；匹配主键固定为seed、branch、context length、item ID。
- 按lookup/revision/transition/unknown分别给出8/16/32行正确数和literal `unknown` 数；unknown正确只按答案为`unknown`且输出逐字匹配计数。
- 对同seed、同题、同长度配对`length_mixed`与`eight_row_control`，统计双方都对/都错及仅candidate/仅control正确数。
- 对每个分支做逐题8→16、16→32、8→32转移计数（对→错、错→对、保持），另按task kind分层；计数必须能回算至1,728条源行。
- 除JSON内SHA绑定和对分析输出SHA外，不读取N150 progress、其他原始数据或模型；source result文件必须与N1-151A报告SHA一致。

## 边界与产物

- 0模型加载、0 checkpoint加载、0新查询、0训练/更新、0 dev/holdout/confirmation；不修改既有门、N1-150/N1-151A结果或晋级状态，`can_promote=false`。
- 这是对同一已见训练任务上的长度分层读数做配对描述性归因；不当作独立样本、泛化验证或长度机制因果证明。
- CPU≤30秒；SHA、分母、主键唯一性、按类配平任一失败即停止，不重抽、不重试。
- Analyzer：`scripts/training/analyze_taiji_n1_plan152_length_itemwise_attribution.py`
- 合同测试：`tests/taiji_native/test_n1_plan152_length_itemwise_attribution_contract.py`
- 报告：`reports/taiji_n1_plan152_length_itemwise_attribution_20261020.json`
