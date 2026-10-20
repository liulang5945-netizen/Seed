# PLAN-N1-162E 执行结案

**日期：2026-10-20**  
**状态：N1-155A父档、N1-162B control与augmented三方逐题归因完成**  
**计划：** [PLAN-N1-162E](PLAN-N1-162E_parent_control_surface_drift_attribution_20261020.md)

## 完整性

N1-155A候选与N1-162B六端点的封存progress全部复核；N1-155A与N1-162B各自事件分母、seed/arm分块、gate pair键及length键唯一性通过。共配对8,640条训练门行与2,592条长度评分行；0模型/checkpoint加载、0查询、0更新、0 dev/holdout/confirmation。CPU6.3125/180秒，737项来源/checkpoint哈希并集不变，`can_promote=false`。完整报告SHA-256为`7a2351050e55c98af4160ff0f5b23cf9feabcd49f10abeba2a7b9db86044f39c`，见[三方逐题报告](../../reports/taiji_n1_plan162e_parent_control_surface_drift_20261020.json)。

## 对配方漂移的分解

- N1-155A candidate到N1-162B surface_control的transition原题与rename在三个seed均保持48/48；说明在这两个常规问法上，延长原问句训练没有造成下降。factorial续训在seed13有少数退化：original-view的direction_opposite 48→47、turn_flip 48→43，导致factorial joint低于门槛。
- 从surface_control到surface_augmented，transition regular分数下降为：seed11 original/rename从48/48降至42/38，seed12降至29/27，seed13降至47/47。逐题轨迹中327项是parent与control均正确、augmented错误；26项是parent正确、control暂错、augmented恢复。
- seed12 unknown原题/rename从N1-155A的47/48、48/48降至control的39/48、38/48；surface_augmented恢复到43/48、44/48。材料问句表面增强能部分修复续训后的unknown下降，但本批transition模板增强又引入更大的transition损失。

当前最强解释是两种现象叠加：相同问法续训会带来少量unknown/factorial漂移；加入transition问句改写后，seed11/12出现显著transition回归。训练集结果不能证明未见泛化，也不足以把现象归因到某一个特定问句模板。

## 唯一下一步

依据用户授予的执行计划批准权，已批准[PLAN-N1-162F transition问句模板交叉探针](PLAN-N1-162F_transition_template_crossover_train_diagnostic_20261020.md)：在N1-155A candidate与N1-162B control/augmented九个冻结checkpoint上，对同一批训练transition原题、rename与factorial上下文，分别查询两种已使用过的after-turn问句，共6,912次train-only生成；原v3问句基线从封存progress复用。独立恢复9/9后才查询；CPU≤1,200秒，0训练/更新、不访问dev/holdout/confirmation，不晋级。这能判断方向错误由哪种问句形式触发，结果仍不作未见能力证据。
