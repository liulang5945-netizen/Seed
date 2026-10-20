# PLAN-N1-162E：N1-155A父档与N1-162B配对续训漂移归因

**状态：已依用户授予的执行计划批准权批准并冻结**  
**日期：2026-10-20**  
**上位执行：** [PLAN-N1-162B结案](PLAN-N1-162B_execution_closeout_20261020.md)  
**逐题门归因：** [PLAN-N1-162C结案](PLAN-N1-162C_execution_closeout_20261020.md)  
**transition输出归因：** [PLAN-N1-162D结案](PLAN-N1-162D_execution_closeout_20261020.md)

## 目的

N1-162D显示seed12 augmented transition错答主要是反向turn，seed11 augmented则错误类型更分散。N1-162B每个新endpoint又从N1-155A候选完整bundle续训9,216步；surface_control与surface_augmented分支分别提供“原问句续训”和“新增问句表面”的匹配路径。为区分N1-155A→surface_control的共同续训漂移与surface_control→surface_augmented的问句表面增量，只对已有train-only gate/length评分做N1-155A candidate、N1-162B control、N1-162B augmented三方逐题匹配。

## 冻结输入与分析

- 读取N1-155A封存progress中candidate `unknown8_rebalanced`的2,880条gate及864条length行；从N1-162B progress读取六endpoint完整门/长度行，并恢复其gate行被`arm=position`覆盖的实验臂，沿用N1-162C已验证的固定行序和960行endpoint chunk合同。
- 三方gate以`seed/evaluation_set/kind/id/condition/case_id`配对；对每题统计N1-155A→control、control→augmented、N1-155A→augmented的correct/incorrect和raw输出变化，重点列transition factorial四条件、unknown原/重命名及出现反向turn类别的父项。
- length评分按`seed/context_length/kind/id/split`三方匹配，量化unknown 8/16/32行的逐题轨迹。
- 每个endpoint/chunk的实际行数、seed、key唯一性必须通过；三方gate复算原N1-132充分性相关分子，并与三份封存summary相符。错误字符串类别与配方变化仅描述共同现象，不单独声称因果机制。

## 预算和停止

- 冻结N1-155A结果SHA-256 `e577d4dda753d016a46e11ab2e0202b2845206af107c54f87aff7f9fdf1ebb11`、progress SHA-256 `83e951930272d30ab3b70a0164dba0ef628d8876a8138708be08a97d3a7acf73`；N1-162B结果SHA-256 `19dfaeb2086abaa503ee5150dba3bbfc9756d0892e8b6b54222fbfc4ecba2c31`、progress SHA-256 `48212bc24ff9267390f7568142a0afd99dfeb39ab787e36c420d6d5d28ed4614`。
- CPU≤180秒；0模型/checkpoint加载、0推理/查询、0更新、0 dev/holdout/confirmation、新面为0；不改判据、不晋级。
- 任一来源SHA、progress分母、三方pair键或充分性复算不一致即停止并报告，不覆盖原始产物、不重试。

## 产物

- 分析器：`scripts/training/analyze_taiji_n1_plan162e_parent_control_surface_drift.py`
- 契约测试：`tests/taiji_native/test_n1_plan162e_parent_control_surface_drift_contract.py`
- 报告：`reports/taiji_n1_plan162e_parent_control_surface_drift_20261020.json`
