# PLAN-N1-149A：readiness来源校验模块绑定修正

**状态：依据用户授予的执行计划批准权已批准并冻结。**只修正N1-149 zero-query readiness入口的模块引用；v26 face、source exclusion、测量题目、query scorer、长度因素、CPU预算和判读边界全部不变。

## 前置失败与根因

N1-149首次`--validate-only`在读取prepared source hashes时停止：评估器调用`n148_builder.n144_builder._verify_hashes`，但实际导入的N1-148建面器没有直接暴露`n144_builder`属性；共享校验器位于`n148_builder.n147_builder.n144_builder`。

失败发生在readiness attempt写入和checkpoint restore之前。v26题面已生成但未查询；六个readiness/query输出路径均不存在。准备期source hashes已用嵌套路径的共享校验器只读复核通过。见[失败回执](../../reports/taiji_n1_v26_abstention_length_preflight_failure_20261020.json)。

## 修正范围

- 保持已提交N1-149计划、builder、evaluator、题面及所有source SHA不变。
- 新增独立入口`run_taiji_n1_v26_abstention_length_after_n148_repair.py`，在当前进程内为N1-148 builder绑定已经存在的共享SHA verifier，再调用原N1-149 evaluator；不重标diagnostic split、不改scorer。
- readiness source binding额外纳入本修正计划、入口、测试和失败回执SHA。
- 修复入口和契约测试先通过并提交，再只运行一次`--validate-only`。须12/12完整bundle恢复、source/checkpoint SHA不变、累计CPU≤7,200秒，否则停止且0查询。
- readiness通过后，才按原N1-149冻结协议查询v26面2,880条；不训练、不重试、不改门、不晋升。
