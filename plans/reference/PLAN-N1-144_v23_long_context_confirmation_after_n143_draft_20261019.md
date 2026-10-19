# PLAN-N1-144：v23长上下文新题确认

**状态：已根据用户授予的执行计划批准权批准并冻结。**N1-143的v22范围阶段完整执行，但只有6/12端点达到24/60错误底线；面已消费且不重测。用户已授权助手批准后续执行计划以保持推进连续性，据此批准新的v23 seed123面。此计划不改动N1-143已消费题面、范围线或确认判据。

## 依据与目标

[N1-143结案](PLAN-N1-143_execution_closeout_20261019.md)显示N1-141 seed12/13 candidate在16行v22面上的answerable错误分别为11/60和5/60，导致先测尺只有6/12通过。N1-144将新面上下文加倍至32行，检验交错候选在训练最大16行长度之外是否仍保有问答表现。该难度调整发生在一张新面上，不据v22题目重抽或追加。

## 冻结题面与来源排除

- 固定v23 seed123，160题：lookup、revision、transition、unknown各40题；每题32条观察；实体长10、目标近邻Hamming距离d=1；答案材料词6/7字符平衡。
- sacrifice和confirm各80题，每部分每类20题。lookup目标行起始位置固定轮转`[0, 8, 16, 24, 31]`；revision/transition双事实起始位置固定轮转`[0, 8, 15, 23, 30]`，相邻行分别编码revision证据或转向序列。四种transition方向与左右转向在两个subset内平衡。
- source exclusion继承N1-143的29,202个实体名与8,208个词值，并加入已消费v22面新增2,520个实体名、1,920个材料/答案值；预期并集31,722个唯一实体名、10,128个词值。生成前必须重算并绑定每条来源SHA、实体/词值集合SHA和交集为0；任一不符即停止。
- v23面、manifest及checks使用独立write-once路径；seed123一次，失败不换seed、不重试。

## endpoint、查询顺序与冻结门

比较终点与N1-143相同，共12个完整bundle：N1-132 position父端点3枚、N1-140 block candidate3枚、N1-141 candidate3枚、N1-141 control3枚。查询前独立恢复12枚bundle并核对checkpoint、Adam、RNG/history和源哈希；任一失败即0查询停止。

1. 先对12端点各查询80道sacrifice，共960条；错误分母仍为60道answerable题。至少8/12端点各有≥24/60个错误才继续。
2. 范围门通过后，对12端点各查询80道confirm，共960条；两阶段最多1,920条。未过范围线时confirm查询严格为0。
3. confirm门与N1-143保持完全相同：N1-141 candidate每seed≥40/80且每类≥8/20；三seed合计至少比N1-141 control及N1-132 parent各高48/240，并至少比N1-140 block candidate高24/240。各门合取，不临场改线。
4. 任一部分查询、runner中断或源校验失败都消费本计划；每条query行先落progress，后续不重跑、不补抽、不替换面。

## 资源与边界

- CPU上限7,200秒，覆盖来源并集构造、32行face合同、12枚checkpoint恢复、range/confirm及末端哈希复核。
- 0训练、0参数更新；不查询dev、旧v3 holdout、v21或v22面。v23是新的独立确认面。
- 无论结果如何`can_promote=false`。即使v23所有局部门通过，也不能替代项目级三器官/出口门或自动晋升。
