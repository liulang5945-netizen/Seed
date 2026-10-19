# PLAN-N1-147A：readiness哈希校验器绑定修正

**状态：依据用户授予的执行计划批准权已批准并冻结。**此项只修复N1-147 readiness入口的工程调用；v24题面、source exclusion、测量条件、960查询量、CPU上限、读数及解释边界全部保持不变。

## 首次前置失败与根因

N1-147首次`--validate-only`在face-prepared来源核验时停止：`eval_taiji_n1_v24_abstention_after_n146.py`调用`face_builder._verify_hashes`，但N1-147建面器没有该属性。该纯函数在复用的N1-144建面器中存在，且v24准备期的403项source hash已通过该共享函数只读复核。

失败发生在写readiness attempt和加载checkpoint之前：readiness/attempt/source-binding及query attempt/progress/result六个输出均不存在；完整bundle恢复0次、checkpoint/model加载0次、模型查询0、参数更新0。见[首次前置失败回执](../../reports/taiji_n1_v24_abstention_preflight_failure_20261019.json)。

## 修正范围

- 保留冻结的N1-147计划、建面器、原评估器、v24面、source manifest及其SHA；不得修改或重建。
- 增加独立入口`run_taiji_n1_v24_abstention_after_n146_repair.py`，在进程内把缺失的校验器别名到已有N1-144共享校验函数，然后调用原N1-147 `--validate-only`/`--execute-approved`。修正不得修改N1-147 scorer、面或查询条件。
- readiness/source-binding额外绑定本修正计划、入口和契约测试SHA。N1-147原始N1-147 runner SHA继续作为prepared-face来源核验的一部分。
- 修复后先运行契约测试并提交，再只运行一次`--validate-only`。必须12/12完整bundle恢复相等、来源/checkpoint哈希未变、累计CPU≤3,600秒；任何失败均停止且不查询。
- readiness通过后，按原PLAN-N1-147只查询新v24面一次，共960条；不训练、不访问dev或旧v3 holdout，结果继续`can_promote=false`。任何查询中断都不重试。

## 不变的边界

本修正不改变新面、配对事实槽位、endpoint、CPU预算、逐题保存、指标、判读规则或能力主张。N1-147首次前置失败保留为历史证据。
