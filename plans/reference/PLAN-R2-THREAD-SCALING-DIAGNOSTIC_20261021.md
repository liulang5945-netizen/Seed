# PLAN-R2-DIALOGUE-08：R2单臂CPU线程扩展诊断

**状态：已结案（7/8 trial；触发CPU停止线并有48.4375秒预算偏差；不重跑本attempt）。**  
**日期：2026-10-21**  
**依据：** [R2-DIALOGUE-06](PLAN-R2-DAILY-DIALOGUE-FULL-TURN-PRETRAINING_20261021.md) · [R2-DIALOGUE-07](PLAN-R2-DAILY-DIALOGUE-WARMSTART-MATCHED_20261021.md)

## 目标与问题边界

R2-06/07 runner把OMP、MKL、OpenBLAS和PyTorch intra-op线程固定为1。在24逻辑核机器上，三条训练不应预期占满整机CPU；但线程数高于1是否能提高R2单臂wall-time吞吐尚无R2专属实测。本诊断用完全相同的checkpoint副本和train-only片段比较1、2、4、8线程，不改变正式训练的冻结设置，也不把高CPU百分比当作能力提升。

## 冻结测量

- 唯一父状态是R2-07 `warm_start/full_turn_zero.pt`，每次trial用weights-only完整恢复至独立内存trainer；校验checkpoint digest、corpus digest、参数与history恢复一致。父checkpoint和来源文件SHA前后必须不变，不写任何模型或checkpoint。
- 唯一数据来源是冻结的3,762条全轮OASST2 train corpus；每次trial从相同zero状态处理同一前16个train episode。线程顺序固定为`1, 8, 2, 4, 4, 2, 8, 1`，共8个trial、128次副本episode。
- 记录实际线程数、target-byte updates、训练wall秒、process-CPU秒、updates/秒。恢复时间单独计量，不混入训练吞吐。
- 不读dev/holdout/reference、不做模型generation或答案评分；副本训练更新不进入任何正式checkpoint，`formal_model_updates=0`、`can_promote=false`。
- 启动时全机CPU采样须≤85%、可用内存≥8 GiB；总process CPU≤900秒。超过限制、线程数无法设置、恢复不等或父SHA/source变化即停止，保留失败attempt。

## 判读

只比较相同模型结构、相同权重起点、相同16条训练episode的吞吐。若2/4/8线程wall吞吐未稳定超过单线程，则正式训练继续使用1线程；若出现稳定提升，只把它作为后续训练配方候选，另经同预算、保存恢复与训练门验证后再应用，不中途改写R2-06/07。

## 执行结案与偏差

实际完成7/8个trial：1线程1次、2线程2次、4线程2次、8线程2次。7个trial各处理相同的前16条train episode与6,643 target-byte updates；零 dev/reference 查询、零 checkpoint 写入，warm-start父zero SHA前后相同。wall吞吐中位数分别为1线程199.158、2线程202.779、4线程209.168、8线程203.012 updates/秒；process-CPU成本中位数分别33.141、65.156、126.508、260.547秒。相对单线程，wall变化约+1.8%、+5.0%、+1.9%，CPU成本约1.97×、3.82×、7.86×。仅1线程有一次重复，且诊断与R2-06/07正式训练并行，故读数只作方向性证据，不足以更改当前或后续训练线程合同。

**预算偏差：** attempt记录948.4375/900 CPU秒，超出48.4375秒；因runner只在trial边界检查，最后一个trial期间超过上限。该偏差如实留档于[adjudication报告](../../reports/taiji_r2_daily_dialogue_cpu_thread_scaling_adjudication_20261021.json)，不重跑。之后已给runner增加每trial 300秒CPU预留守卫，修复未来预算越界风险；此代码修复不改变本次attempt的已绑定runner SHA。`can_promote=false`。

## 产物

- runner：`scripts/training/benchmark_taiji_r2_cpu_thread_scaling.py`
- attempt/progress/result：`reports/taiji_r2_daily_dialogue_cpu_thread_scaling_20261021_*`
