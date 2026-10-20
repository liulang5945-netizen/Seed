# PLAN-N1-162B 执行结案

**日期：2026-10-20**  
**状态：六endpoint训练、训练集计分与checkpoint恢复完整；充分性门未全过，未生成新面**  
**执行修订：** [PLAN-N1-162B](PLAN-N1-162B_首步解析器修正与隔离续执行_20261020.md)  
**原方案：** [PLAN-N1-162](PLAN-N1-162_question_surface_matched_continuation_20261020.md)  
**预算修订：** [PLAN-N1-162A](PLAN-N1-162A_执行预算可行性修订_20261020.md)

## 执行完整性

N1-162首轮在第一个更新前因问句解析器递归停止，0更新、0评分、0新面查询。该attempt、空progress及0更新partial原样保留。N1-162B捕获原始v3解析器后从同一组三枚N1-155A候选父bundle的干净副本重新起跑；修正版契约测试覆盖原问句回退和四种增强问句解析。

N1-162B完成55,296/55,296次更新及7,488/7,488条train-only评分。六个zero bundle保存和独立恢复6/6，六个final full bundle独立恢复6/6；progress 62,784行完整、无坏行，包含55,296更新、5,760原门逐题行及1,728长度行。来源哈希一致。B阶段累计CPU 10,186.594秒（含首轮270.4375秒），关联累计21,244.938/24,000秒。0 dev、0 holdout、0 confirmation，`can_promote=false`。

首轮递归不代表训练结果；正式训练来自N1-162B，累计参数更新仍为55,296。最终结果SHA-256为`19dfaeb2086abaa503ee5150dba3bbfc9756d0892e8b6b54222fbfc4ecba2c31`；progress SHA-256为`48212bc24ff9267390f7568142a0afd99dfeb39ab787e36c420d6d5d28ed4614`。

## 原N1-132训练门

原门要求原题与重命名各自`free_exact≥173/192`、四类各至少40/48、pair至少87/96；此外原始与重命名两种factorial视图的`joint_four_exact`都至少44/48。六endpoint中只有seed11/surface_control与seed13/surface_augmented通过，因此没有生成seed132面。

| seed | arm | 原题命中 | 重命名命中 | transition 原题/重命名 | unknown 原题/重命名 | factorial joint 原题/重命名 | 结论 |
|---:|---|---:|---:|---:|---:|---:|---|
| 11 | surface_control | 189/192 | 189/192 | 48/48、48/48 | 45/48、45/48 | 48/48、48/48 | 通过 |
| 11 | surface_augmented | 185/192 | 177/192 | 42/48、38/48 | 47/48、43/48 | 17/48、13/48 | 未通过 |
| 12 | surface_control | 183/192 | 182/192 | 48/48、48/48 | 39/48、38/48 | 48/48、48/48 | 未通过 |
| 12 | surface_augmented | 168/192 | 167/192 | 29/48、27/48 | 43/48、44/48 | 0/48、0/48 | 未通过 |
| 13 | surface_control | 191/192 | 190/192 | 48/48、48/48 | 47/48、46/48 | 42/48、46/48 | 未通过 |
| 13 | surface_augmented | 191/192 | 191/192 | 47/48、47/48 | 48/48、48/48 | 45/48、46/48 | 通过 |

配对结果呈明显seed异质性：surface_augmented在seed11/12使transition与四条件factorial读数大幅退化；seed13的candidate通过原门，但不能抵消另外两seed的失败。seed12 control单独未过门的原因是unknown原题39/48、重命名38/48。按冻结协议，seed132 heldout面没有生成，0新面查询；不得把训练集读数解释成句式泛化收益。

## 证据限制与唯一下一步

progress中的N1-132 gate行继承了上游`arm=position`字段，覆盖了外层实验臂字段；因此逐行分析必须按冻结执行顺序将每个seed/arm的连续960行绑定到对应endpoint，并核对seed与行数，不能直接按该`arm`字段分组。长度评分行保留外层arm。此映射由runner endpoint顺序与逐endpoint固定分母定义，归因报告需显式记录该字段碰撞。

依据用户授予的执行计划批准权，唯一下一步是[PLAN-N1-162C训练门逐题配对归因](PLAN-N1-162C_train_only_itemwise_gate_attribution_20261020.md)：离线读取封存的5,760条原门行、1,728条长度行及schedule，恢复每个paired endpoint的外部arm标签，定位surface_augmented造成的transition/factorial和unknown门差异，并把错误父项映射到其训练模板曝光。CPU≤120秒，0模型/checkpoint加载、0查询、0更新、0新面，不改判据或晋级状态；据归因再排下一项训练决策。
