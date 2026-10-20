# PLAN-R2-DIALOGUE-02：自然对话生成输出离线归因

**状态：已冻结，零训练/零模型查询；承接R2-DIALOGUE-01生成失败读数。**  
**上游结案：** [R2-DIALOGUE-01](PLAN-R2-DAILY-DIALOGUE-01_execution_closeout_20261021.md)

## 要回答的问题

R2-DIALOGUE-01的UTF-8和结束标记改善，是否伴随可观察的上下文敏感、输出去重或可读性改善？candidate仍有重复片段；需要把“停止行为改善”和“自然中文对话生成”分开描述。

## 输入与预算

- 只读[retry2结果](../../reports/taiji_r2_daily_dialogue_20261021_retry2_result.json)，SHA-256 `e939e4c91219b709e0a3799db13610fc9cb1407916d3a9240b4af451f00411f0`，以及[逐条生成](../../reports/taiji_r2_daily_dialogue_20261021_retry2_generations.json)，SHA-256 `32b97f4a66d562a607967eb17ac0fbb452307cd429806f551ff3c42acbc0d02f`。
- 不读训练或最终参考回复；不加载模型或checkpoint；0新生成、0参数更新、0dev查询。
- 只计算并核对48条已消费输出的UTF-8/边界/长度/重复指标；按12条自然上下文逐题计算parent含历史/单轮对照、candidate/control在无历史六题上的输出相同率；人工可读性只作明确标为主观的样例观察，不伪装成自动语义分数。
- CPU≤30秒。source哈希、行数或端点不符即停止；不重试查询已消费面。

## 产物

- 归因器：`scripts/training/audit_taiji_r2_daily_dialogue_generations.py`
- 机器报告：`reports/taiji_r2_daily_dialogue_20261021_attribution.json`
- 中文结案：`plans/reference/PLAN-R2-DAILY-DIALOGUE-ATTRIBUTION_20261021_execution_closeout.md`

结果只用于选择后续数据/模型方案，`can_promote=false`。
