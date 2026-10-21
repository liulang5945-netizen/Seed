# EXP-R2-DIALOGUE-08：CPU线程扩展诊断结案

**日期：2026-10-21**  
**状态：部分测量后按CPU停止线结案；不重跑。**  
**计划：** [R2-DIALOGUE-08](PLAN-R2-THREAD-SCALING-DIAGNOSTIC_20261021.md)  
**机器读数：** [部分裁定JSON](../../reports/taiji_r2_daily_dialogue_cpu_thread_scaling_adjudication_20261021.json)

## 结论

用户感觉“并行后单项训练更慢、CPU占用没拉高”有合理原因：R2-06/07的runner都设置`OMP_NUM_THREADS=1`、`MKL_NUM_THREADS=1`、`OPENBLAS_NUM_THREADS=1`，并调用`torch.set_num_threads(1)`。24逻辑核机器上，每条训练主要使用一个intra-op线程；并发增加总任务吞吐，不会自动让单个模型占用更多核。逐字节训练的外层也按episode和目标字节串行推进。

R2-08在同一warm-start zero checkpoint的内存副本、相同16条train episode上完成7次/计划8次trial：

| intra-op线程 | 完成重复数 | wall中位数 | process-CPU中位数 | 更新/秒（wall） | 相对单线程wall变化 | CPU成本倍率 |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 1 | 33.355秒 | 33.141秒 | 199.158 | 基线 | 1.00× |
| 2 | 2 | 32.762秒 | 65.156秒 | 202.779 | +1.82% | 1.97× |
| 4 | 2 | 31.760秒 | 126.508秒 | 209.168 | +5.03% | 3.82× |
| 8 | 2 | 32.724秒 | 260.547秒 | 203.012 | +1.94% | 7.86× |

多线程确实会抬高CPU使用，但样本内wall吞吐只小幅上升；4线程约快5%，process-CPU成本约3.8倍，8线程没有超过4线程。单线程仅测一次，且基准与三条正式训练并发运行，因此这只是方向性结果，不足以宣称多线程确定更快或确定最优。

## 执行偏差

attempt记录7/8 trial后累计CPU **948.4375/900秒**，超过48.4375秒。原因是runner只在trial边界检查预算，没有给下一trial预留最坏耗时；末个已启动trial运行中越线。attempt和progress保留，不重跑。父checkpoint SHA前后相同；0 dev/reference查询、0 checkpoint写入、0正式模型更新。七个trial只更新内存副本，共46,501次train-only target-byte更新，`can_promote=false`。

原attempt还遗漏了`cpu_limit_seconds`字段；裁定按冻结计划中的900秒上限计算，后续attempt已补记预算字段。随后给runner增加每trial 300秒预留守卫，并通过预算合同测试、Ruff与py_compile。attempt绑定的原runner SHA保存在源清单；后续代码修复不改写历史读数。

## 对当前训练的处理

R2-06/07已冻结为单线程，因此不中途改线程数。继续完成这批训练、独立恢复终点checkpoint，并按原计划做同面自由生成审阅。以后若要升线程数，需先做隔离且重复充分的R2吞吐校准，再决定新的正式训练配置。CPU使用率本身不作为速度或能力结论。
