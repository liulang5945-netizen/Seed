# PLAN-N1-165：transition问句模板中间剂量配对续训

**状态：已按用户授予的执行计划批准权冻结；准备实施。**
**日期：2026-10-20**
**前置结案：** [N1-163](PLAN-N1-163_execution_closeout_20261020.md)
**逐题证据：** [N1-164 v2归因](PLAN-N1-164_plan163_itemwise_pair_attribution_20261020.md)

## 假设

N1-163从共同N1-162B父checkpoint做了两个匹配剂量：control的transition监督比例为50% original、25% `direction_after_turn`、25% `direction_after_turning`；全量preferred candidate为50% original、50% `direction_after_turn`。逐题归因显示全量preferred在seed11较control多出transition回退，seed12持平，seed13少量回退。下一步只测试两者之间的中间剂量，确认preferred比例从25%升至37.5%时能否保留跨seed稳定性。schedule每个`parent_id/route/context_length`八槽块固定4 original、3 preferred、1保留变体。

本实验是训练配方剂量校准，结果仍限于现有train题。若三seed原门全部过线，也只代表训练充分；新题泛化另行准入，不自动生成或查询确认面。

## 冻结训练设计

- **父checkpoint：** seed11/12/13各自N1-162B `surface_augmented_final.pt` 完整bundle；每个新candidate从对应父checkpoint的model、Adam、RNG和history独立分叉。与N1-163 control/full-preferred两剂量使用相同逐seed父SHA、课程和评分器。
- **唯一新增剂量：** transition监督总体比例固定为50% original、37.5% `direction_after_turn`、12.5% `direction_after_turning`。每个`parent_id/route/context_length`八槽块由N1-163 control的4 original/2 preferred/2原变体改为4/3/1；同一对原变体槽按冻结的step顺序与平衡奇偶规则只替换其中一个。每个上下文长度合计384 original、288 preferred、96保留变体。其他任务、问句、slot顺序、loss与系数、优化器和1线程设置不变。
- **训练量：** 每seed固定18,432更新（同一9,216槽N1-162B schedule顺序重复两遍）；3 seeds共55,296更新。只新建三个mid-dose endpoint；N1-163的control和full-preferred结果作为同seed、同父checkpoint、同更新量锚点，不重跑旧臂。
- **评分：** 每endpoint沿用N1-132原192项训练门（960条评分）和N1-163同一288项长度评分，共3,744条train-only评分。原门及分母不变；逐seed同时报告三剂量的正确转错/错误转对、旧能力保持和长度分层结果。
- **边界：** 训练开始前完成三个zero full bundle的write-once保存和fresh process独立恢复3/3；final bundle同样独立恢复3/3。任何来源、schedule、保存恢复、step/history、评分状态或预算守卫失败即停止，保留失败回执，不自动重跑。
- **禁止项：** 不生成新题面，0 dev/holdout/confirmation/fresh-face查询，不改变N1-132训练门，不自动晋级；`can_promote=false`。
- **预算：** 本阶段CPU≤10,500秒；关联N1-162/163/164 CPU累计≤50,000秒。N1-163已用38,251.734375秒，N1-164归因另用2.0625秒，故本批按上限仍保留至少1,246.203125秒总余量。评分与独立恢复CPU均计入本批。

## 必须修正的执行记录问题

N1-163完整result落盘后，终端摘要因把`total_updates`误写为`training_updates`而返回退出码1；新runner须按result实际schema打印摘要。N1-163逐题gate行的arm字段也被`score_row`覆盖为父档`position`；新runner写progress时必须让当前seed和续训arm覆盖score row字段。此修正只作用于新runner，不改写已封存的N1-163源文件和结果。

## 产物

- schedule builder：`scripts/training/build_taiji_n1_plan165_transition_template_schedule.py`
- continuation runner：`scripts/training/run_taiji_n1_plan165_transition_template_mid_dose.py`
- schedule/readiness：`plans/manifests/n1_plan165_transition_template_mid_dose_schedule_20261020.json`、`reports/taiji_n1_plan165_transition_template_mid_dose_readiness_20261020.json`
- zero preflight：`reports/taiji_n1_plan165_transition_template_mid_dose_zero_preflight_20261020.json`
- checkpoints：`data/checkpoints/n1_plan165_transition_template_mid_dose_20261020/`
- attempt/progress/result：`reports/taiji_n1_plan165_transition_template_mid_dose_{attempt,progress,result}_20261020.json`（progress为JSONL）

## 停止与后续

若mid-dose三seed全部通过原训练门，且相较25%剂量没有广泛旧题回退，则下一步根据三剂量训练曲线冻结一次新的未见确认方案；否则先按seed和长度定位失效，再重新修订训练分布。无论何种结果，本批不查询新面，也不晋升。
