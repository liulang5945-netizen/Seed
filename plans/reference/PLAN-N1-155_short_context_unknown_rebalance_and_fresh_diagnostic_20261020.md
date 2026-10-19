# PLAN-N1-155：短上下文unknown再平衡训练与fresh诊断

**状态：已依据用户授予的执行计划批准并冻结。**N1-153A fresh face上，8行candidate/control的known命中为360/360与353/360，但unknown正确弃答为76/120与109/120；32行known命中为356/360与302/360，unknown弃答为103/120与120/120。N1-154逐题配对分析显示，candidate的8→32行有34项错转对、11项对转错，control有11项错转对、51项对转错。计划检验是否可通过增加短上下文unknown训练保留长上下文known收益，同时改善unknown弃答；结果只作训练与fresh合成诊断，不晋级。

## 父模型与固定训练变量

- 每seed以N1-150的`length_mixed` final full bundle（seed11/12/13）为共同父模型；每seed分叉`unknown8_rebalanced` candidate与`uniform_length_replay` control，zero bundle逐位相等。恢复完整模型、41个Adam状态、RNG和history。
- 六endpoint×9,216追加更新=55,296总更新。继续使用原192条v3 train父任务、N1-132原objective（包括原binding辅助项）、优化器、学习率与训练门；不把fresh face题目用于训练。
- 两支每父项固定48更新（raw/hard各24）。control延续原8/16/32各16次的均匀schedule。candidate的lookup/revision/transition每种也各16次/长度（共48次）；只有unknown父项改为8行24次、16行12次、32行12次。raw/hard拆分分别为candidate 12/6/6，control 8/8/8。
- N1-132的unknown父项仍包含原配对known lookup第二问题；当unknown父项slot分配到8/16/32行时，这两个既有问题继续共享同一观察带。此短上下文再平衡按原有每slot query结构计数，不增添问题或训练目标。
- 单模型候选分布：每种known类8/16/32各768步；unknown分别1,152/576/576步。总schedule为8行3,456步、16行2,880步、32行2,880步；类别、父项、raw/hard路径、上下文行内容、目标和答案均保持配平。

## 保存恢复、计分与新fresh面

- 训练前6个zero bundle及训练后6个final full bundle均独立恢复6/6，核对模型/Adam/RNG/history/fingerprint；任一失败即0训练停止。
- 每endpoint运行原N1-132 train gate 960条和N1-150A长度分层288条，六endpoint共7,488条train-only评分；门、分母与阈值不变。训练后六个endpoint必须全部通过原训练充分性门，才生成新face。
- 只有训练与final恢复成功且六个原门全通过时，才一次生成seed128 fresh face：40题/类、160 base任务，8/16/32行嵌套，四类平衡；来源排除覆盖全部已消费面、N1-150生成的94,848个实体/94,848个材料值、v3 train源以及N1-153 seed127 face。新face只用于评分，绝不进入训练。
- 在六endpoint上评估fresh face，每endpoint480条、合计2,880条。固定输出known命中、unknown literal abstention和逐题配对转移；这是新fresh synthetic diagnostic，不改任何历史gate或晋级判据。
- CPU总上限18,000秒，覆盖来源/配方/面构造、完整zero/final保存恢复、55,296次更新、7,488条train-only评分、2,880条fresh诊断和末端SHA复核。失败立即封存，不减量、不追加、不重试。
- 0 dev、0 holdout、0 confirmation查询；不自动晋级，最终`can_promote=false`。

## 实施顺序与产物

1. 实现schedule builder、完整checkpoint runner、fresh-face builder、scorer和合同测试；提交后运行无查询readiness并封存完整配方计数。
2. 保存/独立恢复六个zero bundle并提交；通过后才开始固定训练。
3. 六endpoint训练、final restore与原train gate完成且六门全通过后，才一次生成并查询seed128 fresh face；最后逐项核对source exclusion、面SHA、分母和candidate/control配对。

- Schedule builder：`scripts/training/build_taiji_n1_plan155_short_unknown_context_length.py`
- Training runner：`scripts/training/run_taiji_n1_plan155_short_unknown_context_length_training.py`
- Fresh-face generation and diagnostic scoring are implemented as explicit phases in the same plan builder/runner above, avoiding duplicate wrapper modules.
- 合同测试：`tests/taiji_native/test_n1_plan155_short_unknown_context_length_contract.py`
- seed128 face：`plans/manifests/n1_plan155_v29_short_unknown_length_seed128_20261020.json`
- write-once checkpoints：`data/checkpoints/n1_plan155_v29_short_unknown_rebalance_20261020/`
- attempt/progress/preflight/result：`reports/taiji_n1_plan155_short_unknown_..._20261020.json`及`reports/taiji_n1_plan155_short_unknown_progress_20261020.jsonl`
