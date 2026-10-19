# PLAN-N1-142 执行结案：v21范围查询在首端点后停止

**状态：按一次性停止规则结案；行为门未判定；`can_promote=false`。**v21 seed121面已消费，不重跑、不补齐，也不查询confirm。

## 执行结果

- v21面160题、四类各40题、16行、d=1；80题sacrifice与80题confirm按计划冻结。来源排除并集26,682个实体名、6,288个词值；12/12完整endpoint在查询前均独立恢复成功。
- 只对首个endpoint `N1-132/seed11/parent` 查询80道sacrifice，逐题进度与结果报告均保存；其余11个endpoint未查询。runner在完成该endpoint后生成进度汇总时因字段名错误（读取不存在的`receipt["rows"]`）停止，未开始confirm。
- attempt、progress及partial result哈希互相吻合；342项来源哈希未变化。零参数更新、零dev/旧v3 holdout查询；runner报告累计CPU223.625秒，另有一次N120 scorer初始化0.234秒，合计约223.859秒。
- 范围门需要8/12端点，现有1/12，无法判定范围门；80条读数不支持任何模型能力结论。

## 停止处置

本计划明确规定面/查询一旦中断或失败即消费且不得重试，因此保留v21 seed121面及80条部分结果原样，不在同面续跑。问题定位为endpoint完成后的progress receipt记录字段错误，与模型输出无关。

完整失败记录见[attempt](../../reports/taiji_n1_v21_confirmation_attempt_20261019.json)及[partial result](../../reports/taiji_n1_v21_confirmation_result_20261019.json)；本机ignored progress JSONL由`progress_sha256`绑定在结果报告中。

## 后续唯一建议

依据用户授予的执行计划批准权，已批准[N1-143 v22新面确认计划](PLAN-N1-143_v22_fresh_confirmation_after_n142_closeout_20261019.md)。它使用全新seed122面，排除并集扩展为29,202个实体名与8,208个词值；12端点、范围线、确认门、CPU上限及训练/晋升边界保持不变。progress receipt字段修复与合同测试通过前，不生成v22面；通过后仅一次构造和查询。
