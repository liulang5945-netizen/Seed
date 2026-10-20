# PLAN-N1-161A 执行结案

**日期：2026-10-20**
**状态：完成；2,880条封存输出已做零查询逐题归因**
**执行计划：** [PLAN-N1-161A](PLAN-N1-161A_question_surface_itemwise_attribution_20261020.md)
**来源诊断：** [N1-161执行结案](PLAN-N1-161_execution_closeout_20261020.md)

## 完整性与资源

输入SHA逐项通过：N1-161 result `2808a969299143c692643bf4bd5ccabdd67a63021f5ad9ba0cab65bcb51a385a`、seed131 face `39ce277831a0461a45bf7fb0f793d57f89ade84e988d1ba830703b07346393ef`、face audit `cc08a8ec7af299207c8f474db967b8d0eda76aca2fc8be2480b06fe41e2b3b46`、progress `dd80604cd66519f04bfe48d5929756e0f2f5436a68435d3a5df7a31068f66cce`。2,880行全部关联到唯一manifest题目，1,440组candidate/parent配对完整；free exact逐行复算一致，teacher/free exact不一致0行。

归因报告SHA-256为`0d3dac6ffe778a0b4b421a8c14faffc6113bf121bb2501c34afcc7bdcf307ee5`，包含2,880行逐题类别和144个endpoint×seed×任务×模板×长度汇总。CPU报告0.03125秒；独立复核重算另耗0.046875秒，合计0.078125/30秒。0模型/checkpoint加载、0模型查询、0更新、0 dev/holdout/confirmation，`can_promote=false`。

运行CLI时发现同名完整报告已存在，write-once守卫拒绝覆盖。随后将现存报告与输入重新计算结果逐字段比对；除运行CPU读数和中文summary中的计时数值外，其余数据、分层及逐题分类完全相同，因此保留原报告，不生成第二份副本。

## 错误来源读数

N1-161的整体模板迁移读数是candidate 441/1,440、parent 460/1,440。对照输出类别后，最明显的是主动式材料问法下的revision：candidate仅7/180正确，parent32/180；在candidate的173个错答中，有47个与观测第0行的旧revision材料值完全相同、59个输出为固定direction/turn词、64个是其他输出、3个是字面`unknown`。parent在同一层有32个旧值、45个direction/turn词、68个其他输出及3个字面`unknown`。

被动式revision为candidate51/180、parent49/180；lookup的主动/被动模板分别为candidate57/45、parent69/56；transition两个模板为candidate89/83、parent89/80。unknown主动/被动模板为candidate60/49、parent48/37，但合计净增24只由seed13贡献，seed11持平、seed12少3。输出类别是基于字节精确匹配的post-hoc归因，只说明错误输出落在哪类，不能证明模型内部机制。

这些读数把失败定位到问句形式的条件迁移，尤其是主动式revision的旧值复现；unknown再平衡带来的合计增益仍未跨seed复现。所有结论限于这张程序生成面，`can_promote=false`。

## 后续唯一队首

依据用户授予的执行计划批准权，已批准[N1-162配对问句表面训练续接](PLAN-N1-162_question_surface_matched_continuation_20261020.md)：从同一N1-155A candidate完整checkpoint分叉原问句控制与N1-161问句增强臂，保持任务、答案、观测、更新数和原N1-132门一致；最终只在另一组训练未见问法上评估，并与N1-150 parent及N1-155A原候选配对。该方案旨在修复并检验已观察到的主动问句/旧值混淆，不改变既有能力晋级门。
