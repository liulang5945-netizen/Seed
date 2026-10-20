# PLAN-N1-162C 执行结案

**日期：2026-10-20**  
**状态：完成；零模型读取、零查询的训练门逐题配对归因**  
**分析计划：** [PLAN-N1-162C](PLAN-N1-162C_train_only_itemwise_gate_attribution_20261020.md)  
**上位执行：** [PLAN-N1-162B结案](PLAN-N1-162B_execution_closeout_20261020.md)

## 来源与完整性

分析器绑定N1-162B结果SHA-256 `19dfaeb2086abaa503ee5150dba3bbfc9756d0892e8b6b54222fbfc4ecba2c31`、progress SHA-256 `48212bc24ff9267390f7568142a0afd99dfeb39ab787e36c420d6d5d28ed4614`与冻结schedule。62,784条progress事件完整（55,296更新、5,760训练门行、1,728长度行），六endpoint来源/checkpoint哈希重新核验不变。进度记录的5,760条训练门行中，外层arm标签被上游`arm=position`覆盖；分析器按冻结执行顺序每endpoint连续960行重建六个实验臂，并验证每块seed与分母，5,760/5,760行映射成功；长度行1,728条自身保留外层臂标签，零不一致。

对六endpoint的N1-132门逐项重算，与封存结果6/6一致。完整逐题配对报告见[分析结果](../../reports/taiji_n1_plan162b_train_only_itemwise_attribution_20261020.json)，SHA-256 `f98fa027a55e83a03eb18a1288b61168f619b6ace0e31662d596ce82bf584e63`；CPU6.09375/240秒，0模型/checkpoint加载、0推理/查询/更新/dev/holdout/confirmation，`can_promote=false`。

## 主要读数

- transition原题由control→augmented：seed11 `48/48→42/48`，seed12 `48/48→29/48`，seed13 `48/48→47/48`；重命名transition分别`48→38`、`48→27`、`48→47`。seed11/12所有原题和重命名transition不一致项都是control正确、augmented错误，没有一题反向恢复。
- transition factorial四条件joint：seed11原始/重命名`48/48→17/13`，seed12`48/48→0/0`，seed13`42/46→45/46`。seed13的`turn_flip`有少量修复，但两个seed上的整体退化很大。
- unknown类方向相反：seed12原题`39/48→43/48`、重命名`38/48→44/48`；seed13分别`47→48`、`46→48`；seed11原题`45→47`、重命名`45→43`。配对长度读数中，seed12 unknown的8/16/32行是`24/24/24→21/23/23`。
- 所有transition父项在每种长度均平衡接触8个原问句、4个direction-after-turning、4个direction-after-turn模板；unknown父项8行按12/6/6，16/32行按6/3/3平衡接触原句及两种材料问法。因而当前汇总支持“问句表面增强伴随seed异质的transition保持失败和unknown改善”，不支持把问题归给单个模板，也不构成模板因果证明。

## 唯一下一步

依据用户授予的执行计划批准权，已批准[PLAN-N1-162D方向输出类型归因](PLAN-N1-162D_transition_output_type_attribution_20261020.md)：仅对同一批封存transition gate输出和已哈希的训练题定义，按每题的初始方向、turn、条件及期望方向分类错答为初始方向复述、反向turn、对向/其他方向、literal unknown、空输出或其他字节；先要求重算后的期望答案逐行复现封存free_exact。CPU≤120秒，0模型/checkpoint加载、0查询、0更新、0新面；归因只解释已消费训练集门，不作新泛化主张。
