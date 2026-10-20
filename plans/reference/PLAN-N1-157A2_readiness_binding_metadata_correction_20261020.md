# PLAN-N1-157A2：readiness binding元数据更正

**状态：已依据用户授予的执行计划批准权批准，待实现**  
**日期：2026-10-20**  
**续接方案：** [PLAN-N1-157A](PLAN-N1-157A_remaining_endpoint_continuation_20261020.md)

## 触发与审计结果

N1-157A代码与首轮结果绑定通过；首次checkpoint-preflight守卫在加载checkpoint前停止，因为N1-157A source binding写入了readiness SHA及source map，但漏写两个可复核字段`prior_result_sha256`和`prior_consumed_queries`。readiness自身包含正确值，binding所引用的readiness SHA和完整source map也一致。只读复现该守卫CPU为0.0秒，checkpoint restore、模型查询和参数更新均为0。失败回执写于`reports/taiji_n1_plan157a_preflight_binding_failure_20261020.json`并随本更正提交。

## 批准的更正

只修复source/readiness/preflight binding字段传递和对应验证；`prior_result_sha256`绑定N1-157首次480条停止报告，`prior_consumed_queries=480`，`remaining_queries=2400`。所有N1-157与N1-157A旧readiness、source binding、首轮失败回执、480条输出、progress、face和审计均保留；N1-157A2使用新的readiness、preflight、attempt、progress和result路径。seed128 face仍为唯一既有面，继续跳过seed11 candidate，不重新生成或查询它，仅评估剩余五个endpoint。

科学范围不变：复用N1-157A endpoint、指标、2,400条未消费查询和停止规则，不训练、不更新、不访问dev/holdout/confirmation，`can_promote=false`。CPU总上限仍为N1-157A规定的新增阶段≤3,600秒、关联累计≤18,000秒。只读复现的binding拒绝CPU实测0.0秒；原preflight命令未单独输出CPU，故保守追加5秒预算计入关联基线。

若N1-157A2独立readiness及六bundle恢复前置通过，则仅继续seed11父模型与seed12/13 candidate/parent五个未消费endpoint；已保存的seed11 candidate逐题结果只从原失败报告并入最终汇总。
