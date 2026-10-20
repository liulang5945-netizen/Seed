# PLAN-N1-163：转向问句模板重配对续训

**状态：已依用户授予的执行计划批准权批准并冻结**<br>
**日期：2026-10-20**<br>
**前置训练：** [N1-162B结案](PLAN-N1-162B_execution_closeout_20261020.md)<br>
**设计证据：** [N1-162F1逐条复核](PLAN-N1-162F1_execution_closeout_20261020.md)、[N1-162G2线程吞吐结案](PLAN-N1-162G2_execution_closeout_20261020.md)

## 目的与假设

N1-162B在相同父checkpoint上做了`surface_control`与`surface_augmented`配对续训。N1-132原门仅2/6 endpoint通过，失败集中在seed11/12 augmented transition与seed12 factorial；seed13 augmented通过。N1-162D把seed12错误主要分类为反向turn，N1-162F1复核的6,912条模板交叉输出显示，`direction_after_turn`明显优于`direction_after_turning`，但F旧runner源码SHA漂移，因此这只能作为有边界的训练模板选择证据。G2混合长度吞吐实测2/4线程均慢于1线程，后续训练固定单线程。

本批检验：在已训练的N1-162B augmented模型上继续同等更新，若仅将transition监督中的`direction_after_turning`替换为已出现过的`direction_after_turn`，相对于保留原配比的匹配对照，能否改善并保持原N1-132训练门。它是`surface_augmented`包的定向后续续训比较，不单独证明问句措辞的普遍因果作用或模型泛化能力。

## 成对训练设计

- **父模型：** seed11/12/13各自N1-162B `surface_augmented_final.pt` 完整bundle；candidate与control在每seed内从同一checkpoint、同一Adam/RNG/history/fingerprint分叉。这是从N1-162B已训练模型累计续训，不重置到初始基座。
- **两臂：** `surface_replay_control`保留N1-162B augmented现有配比；`preferred_turn_replacement`仅把transition题的`direction_after_turning`槽位换成`direction_after_turn`。每个48-update transition父项中，对照保持24 original/12 after_turning/12 after_turn；候选为24 original/24 after_turn。逐长度槽位为对照8/4/4，候选8/8/0。lookup/revision/unknown模板、任务内容、context长度、样本顺序、目标函数及其余schedule逐槽一致。
- **训练量：** 每endpoint固定18,432更新，即冻结N1-162B 9,216槽完整顺序重复两遍；3 seeds×2 arms×18,432＝**110,592次更新**。候选与对照每个seed的step数、slot集合及slot顺序完全匹配。
- **优化目标：** 原N1-162B `n132._training_step`及0.25·L_bind等所有loss权重不变；PyTorch/OpenMP/MKL/OpenBLAS均固定1线程。

## 保存恢复、计分与停止

- 六个zero checkpoint分别从三个相同父模型fork；每个包含完整model/Adam/RNG/history及N1-163 arm元数据。训练开始前必须write-once保存并在fresh process独立恢复6/6；验证父SHA、bundle fingerprint、optimizer step=61,440、history=79,872、pair内零态bundle完全相等。任一失败即0更新停止。
- 每个endpoint完成18,432更新后，保存final full bundle并fresh process独立恢复6/6；预期optimizer step=79,872、history=98,304。每一步计入progress，预算或异常中断时另存write-once partial bundle，保留已完成记录，不自动重跑。
- 每endpoint只复用N1-132原192项train gate（960条评分）及N1-162B同一288条train-only长度分层评分；共7,488条train-only评分。逐endpoint报告原门和candidate/control配对差异，判据及分母原样不变。
- **不生成fresh face；0 dev、holdout、confirmation查询；不改变N1-132门；不自动晋升。** 即使六endpoint均过门，报告仍为`can_promote=false`，下一步能力验证另行立项。
- Stage CPU≤25,000秒，关联N1-162 CPU≤50,000秒（已用21,244.9375秒）。训练前预检、restore workers、gate及length scoring均计入总额；守卫不通过即停止。固定N1-162B结果、schedule、三枚父checkpoint及全部代码/计划SHA。

## 产物

- schedule builder：`scripts/training/build_taiji_n1_plan163_transition_template_schedule.py`
- runner：`scripts/training/run_taiji_n1_plan163_transition_template_rebalanced_continuation.py`
- 契约测试：`tests/taiji_native/test_n1_plan163_transition_template_rebalanced_contract.py`
- manifest：`plans/manifests/n1_plan163_transition_template_rebalanced_schedule_20261020.json`
- readiness/source audit：`reports/taiji_n1_plan163_transition_template_rebalanced_readiness_20261020.json`
- checkpoint：`data/checkpoints/n1_plan163_transition_template_rebalanced_20261020/`
- attempt/progress/result：`reports/taiji_n1_plan163_transition_template_rebalanced_{attempt,progress,result}_20261020.json`（progress为JSONL）
