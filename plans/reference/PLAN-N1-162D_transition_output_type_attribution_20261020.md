# PLAN-N1-162D：transition训练门错误输出类型归因

**状态：已依用户授予的执行计划批准权批准并冻结**  
**日期：2026-10-20**  
**上位执行：** [PLAN-N1-162B结案](PLAN-N1-162B_execution_closeout_20261020.md)  
**逐题归因：** [PLAN-N1-162C结案](PLAN-N1-162C_execution_closeout_20261020.md)

## 目的

N1-162C显示surface_augmented在seed11/12造成大规模transition错误，seed13近乎保持，并伴随seed12 unknown改善。根据封存的transition gate原始输出、每题训练观察和四条件变体的既定定义，离线判别错误输出是否为：初始方向复述、反向turn结果、错误对向、其他合法方向、literal `unknown`、空输出或其他字节。确认问句增强是否破坏了方向运算、事实消费或仅改变自由输出表面；此归因只用于定位训练集错误形态，不证明模型内部机制。

## 冻结输入与计算

- 读取N1-162B result/progress、PLAN-N1-162C逐题归因报告、v3 train任务与factorial变体定义；这些输入及六个endpoint checkpoint SHA均已由N1-162B绑定。
- original/rename transition期望方向从对应train父项的初始方向与turn重建；factorial_original/factorial_rename按progress逐行记录的`initial_direction`、`turn`及`condition`重算。rename只重命名实体，不改方向标签。
- 每条输出按上述确定顺序进行单类分类，并重新计算`free_exact`；全部expected-answer与free-exact必须逐行复现封存progress。surface_control/augmented由N1-162C已验证的endpoint顺序映射，不使用被覆盖的gate-row arm字段。
- 按seed×实验臂×view×condition统计错误类型、control/augmented配对转移、原始与重命名差异，并链接该transition父项的训练模板曝光。错误字符串类别为post-hoc描述，不作为因果或自然语言能力结论。

## 边界与停止

- 仅处理封存的transition gate rows，不扩大到其他kind，不加载模型或checkpoint。
- CPU≤120秒；0模型加载、0查询/推理、0更新、0 dev/holdout/confirmation、新面为0；不改训练门、不晋级。
- 任一source SHA、期望方向重建、progress分母、endpoint映射或`free_exact`复算不一致即停止并报告不一致，不重试。

## 产物

- 分析器：`scripts/training/analyze_taiji_n1_plan162d_transition_output_type_attribution.py`
- 契约测试：`tests/taiji_native/test_n1_plan162d_transition_output_type_attribution_contract.py`
- 报告：`reports/taiji_n1_plan162d_transition_output_type_attribution_20261020.json`
