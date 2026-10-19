# PLAN-N1-139 执行结案：匹配对照的上下文鲁棒续训校准

**状态：完整执行；原训练充分性门5/6 endpoint通过，未晋升。**

## 执行边界

本批次按已批准的7,488条train-only计分口径完成：每个endpoint运行N1-132冻结的960条原门计分，六个endpoint共5,760条；再对六个endpoint各评估48个transition父项×6种增强上下文，共1,728条。全部模型来自seed11/12/13的N1-132 position终点；每个候选/控制分支各9,216次更新，总计55,296次。没有查询dev、holdout或confirmation面。

N1-132原门未改。候选与控制的目标函数、Adam、学习率及每分支更新数一致；候选仅改变transition上下文分布。控制始终使用schedule原两行输入。最终状态不得晋升，can_promote=false。

## 工程与预算读数

- 270项绑定来源哈希保持不变；三个N1-132父checkpoint哈希与冻结计划一致。
- N1-138 v20输入排除清单与已消费v20面的344个实体并为9,354个排除实体；候选另生成12,096个全局唯一训练干扰实体，并在schedule中逐slot验证Hamming距离、目标事实位置与8/16行上下文长度。
- 2,304个transition更新slot中，候选每父项24次raw和24次hard暴露；288个两行baseline slot及六种增强条件各336个slot。8行与16行增强slot各1,008个。
- 六个zero bundle与六个final bundle共12份checkpoint。六份zero均逐位等同对应N1-132父bundle；六份final独立恢复成功，Adam均为41组状态，history从52,224增至61,440。
- 完成55,296次参数更新和7,488条计分。N1-139执行阶段CPU为3,378.515625秒；加来源readiness 7.234375秒与zero预检108.140625秒，累计3,493.890625/3,600秒。

完整机器读数、7,488条逐题输出、checkpoint哈希和来源绑定见[最终结果报告](../../reports/taiji_n1_plan139_context_robust_matched_continuation_calibration_result_20261019.json)。逐更新日志保留在工作区的progress JSONL；结果报告记录了它的SHA-256。该日志受仓库全局JSONL忽略规则约束，没有加入Git提交。

## 训练门与上下文读数

六个endpoint中五个通过原N1-132训练充分性门：候选seed11/12及三个控制endpoint。候选seed13未通过，原始192项为172/192，transition为28/48；其他三类仍各48/48。N1-132父终点及N139其他五个endpoint的原始transition均为48/48。seed13候选的20个回退transition ID及配对上下文读数见[离线逐题归因](../../reports/taiji_n1_plan139_itemwise_retention_context_attribution_20261019.json)。

在新的train-only增强上下文上，三seed候选共840/864正确，对照411/864。各条件读数如下；每格分母为144。

| 上下文 | 候选 | 两行控制 | 候选净差 |
|---|---:|---:|---:|
| 8行，d=1 | 139/144 | 58/144 | +81 |
| 8行，d=2 | 142/144 | 79/144 | +63 |
| 8行，d=4 | 143/144 | 93/144 | +50 |
| 16行，d=1 | 129/144 | 46/144 | +83 |
| 16行，d=2 | 144/144 | 61/144 | +83 |
| 16行，d=4 | 143/144 | 74/144 | +69 |

逐题配对显示，seed13候选在原两行transition上有20项从全对变错；同一批目标加上训练时的增强上下文后，每项条件恢复17至20项，同时至多回退1项。这说明N139候选学会了已测增强上下文中的transition，但该seed未保住原两行输入门。该结果只限于已消费的train样本，不能当作未见泛化证据。

## 后续唯一建议

起草并审批PLAN-N1-140：保持目标函数、每模型9,216更新、六个endpoint、原N1-132门及7,488条train-only读数不变；只提高候选两行baseline暴露，从每父项6/48改为24/48，六种增强上下文各4/48，raw/hard按路径平衡。该训练配方改变需单独批准；不生成新测面、不访问dev/holdout、不晋升。
