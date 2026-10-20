# PLAN-N1-162D 执行结案

**日期：2026-10-20**  
**状态：2,880条transition输出全部重建并归类，复算一致**  
**分析计划：** [PLAN-N1-162D](PLAN-N1-162D_transition_output_type_attribution_20261020.md)

## 完整性

对封存progress中的2,880条transition门行（每endpoint 480条）重建原始/重命名任务的train答案及factorial条件答案，2,880/2,880条输出分类复算与封存`free_exact`一致。六endpoint、原题/重命名及四个factorial条件的分母都通过核验；来源/checkpoint哈希稳定。CPU8.890625/120秒；读取384个原题/重命名transition输入视图并重算384个factorial期望值，0模型/checkpoint加载、0推理/查询、0参数更新、0新面，`can_promote=false`。

结果报告SHA-256为`27fc0b88ea53fb3d9f211cbaee9997b2372e2bf56dc9f04c93190e98380ef6a2`，完整类别与逐题映射见[归因报告](../../reports/taiji_n1_plan162d_transition_output_type_attribution_20261020.json)。

## 输出类型读数

- seed12/surface_augmented是主要异常：原题transition 19/48、重命名20/48输出反向turn结果；factorial原始/重命名分别86/192、85/192也是反向turn。控制臂在这两个view的transition门均48/48。
- seed11/surface_augmented的原题/重命名transition分别6/48、10/48失败，其中错误分散在其他字节、反向turn及少量方向复述；factorial视图也有较多非方向字节输出。控制臂两种view均48/48。
- seed13/surface_augmented原题与重命名均47/48；surface_control自身在factorial中有少数反向turn输出，增强臂总体相近，并通过N1-132门。

这支持“seed12增强臂出现系统性turn符号翻转，seed11增强臂则有异质输出失败”的训练集归因；它不能证明模型内部机制或未见问句表现。

## 唯一下一步

依据用户授予的执行计划批准权，已批准[PLAN-N1-162E续训漂移逐题分解](PLAN-N1-162E_parent_control_surface_drift_attribution_20261020.md)：对N1-155A候选、N1-162B surface_control和surface_augmented的同一训练门逐题行作三方匹配，分别量化“继续同一原问句训练”与“再加入transition问句改写”两段的增错/修复；并配对unknown长度评分。CPU≤180秒，0模型/checkpoint加载、0查询/更新/dev/holdout/confirmation，不改训练门或晋级结论。分清续训漂移与transition模板增量后，再冻结下一训练配方。
