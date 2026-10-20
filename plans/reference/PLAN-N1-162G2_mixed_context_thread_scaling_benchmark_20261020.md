# PLAN-N1-162G2：完整上下文长度配比的 CPU 线程吞吐复核

**状态：已依用户授予的执行计划批准权批准并冻结**<br>
**日期：2026-10-20**<br>
**依据：** [G1结案](PLAN-N1-162G1_execution_closeout_20261020.md)<br>
**前置基准：** [G方案](PLAN-N1-162G_training_thread_scaling_benchmark_20261020.md)与[修复方案](PLAN-N1-162G1_training_thread_scaling_preflight_repair_20261020.md)

## 为什么补充

G1按父方案冻结的schedule第0–287步完成了真实训练步对照。该片段全为8行上下文，因此只能回答短上下文是否受益，不能代表N1-162B完整训练中的8/16/32行混合输入。G1结果显示2/4线程都慢于1线程，但本方案只补足较长上下文的执行吞吐，不重做、不覆盖G1，不计学习质量或能力。

## 设计

复用同一seed11/surface_control final checkpoint、原训练实现、1/2/4线程六次交错顺序`1,2,4,4,2,1`、每次32步预热加256步计时。新样本按原9216行N1-162B schedule的kind×context_length比例分层选择：总计288槽，其中8/16/32行分别108/90/90；测量部分256槽分别96/80/80；12个kind×长度单元均覆盖，槽位在原schedule各单元中均匀分布。六次×288步合计1,728次仅内存更新。

每个repeat单独恢复相同的seed11/surface_control full bundle，核验optimizer step、history count、fingerprint及checkpoint SHA。计时范围和G一致；report同时保存warmup与测量样本的kind×长度分母、原schedule槽位索引摘要和逐次更新速率。固定10%重复收益规则不变。

## 预算与边界

- CPU上限1,200秒；G2为独立的执行性能复核，不改变G/G1的900秒封存预算。
- checkpoint、N1-162B结果与schedule沿用原SHA身份；G、G1计划/runner/attempt/progress/result全部绑定且保持不变。
- 不写checkpoint、不访问dev/holdout/confirmation、不做能力评分、不改变训练配方或正式runner；所有更新仅作用于各次独立恢复的内存副本。`can_promote=false`。
- 若分层计数、来源/恢复核验或CPU守卫不通过，停止并保留write-once失败材料；不在原输出上重试或覆盖。

## 产物

- runner：`scripts/training/run_taiji_n1_plan162g2_mixed_context_thread_scaling_benchmark.py`
- 契约测试：`tests/taiji_native/test_n1_plan162g2_mixed_context_thread_scaling_contract.py`
- 报告：`reports/taiji_n1_plan162g2_mixed_context_thread_scaling_20261020.json`
