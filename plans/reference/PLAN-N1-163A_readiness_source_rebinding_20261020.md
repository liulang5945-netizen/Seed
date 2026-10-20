# PLAN-N1-163A：N1-163 readiness来源绑定修复

**状态：已依用户授予的执行计划批准权批准并冻结**<br>
**日期：2026-10-20**<br>
**父方案：** [PLAN-N1-163](PLAN-N1-163_transition_template_rebalanced_matched_continuation_20261020.md)<br>
**依据：** [旧readiness](../../../reports/taiji_n1_plan163_transition_template_rebalanced_readiness_20261020.json)、[旧manifest](../../manifests/n1_plan163_transition_template_rebalanced_schedule_20261020.json)

## 触发及旧材料状态

首份N1-163 schedule readiness完成了来源核验和模板分母复算，0模型/检查点加载、0更新、0查询；但随后为加强来源绑定，schedule builder才补入了正式N1-163 runner的SHA。旧readiness绑定的builder SHA为`39e7944651fe71c8b9056677f85455e628ad35fd650218e21f36c287379b350d`，当前builder已变化，且旧source map未包含N1-163 runner。旧manifest SHA为`231d98e8a40457a30933d930940b868f4ce6615f316520065562bcd843b6371f`，旧readiness SHA为`e43db7afd228a16f5eaf00f9046627816fb536324f36a3b462ec8eb128932813`。两份旧输出均保留，不覆盖；N1-163 checkpoint目录不存在，0训练/更新。

## 修复范围

仅在新路径重建schedule manifest/readiness，显式绑定本addendum、base N1-163 plan、最终schedule builder、最终训练runner、契约测试、N1-162B父结果、F1/G2设计证据及六枚surface_augmented父checkpoint。重建后必须逐项确认新的base schedule SHA、9,216行derived-row SHA及全部schedule行与旧manifest相同；任何题目、模板分配、更新量、验收门或checkpoint来源差异都停止。N1-163训练runner随后只读取A版manifest/readiness。

本修复CPU≤900秒，0模型/检查点加载、0训练更新、0查询；不写checkpoint、不改N1-163训练设计或原N1-132门、不触碰dev/holdout/confirmation。旧版与A版使用独立write-once路径；A版通过后才允许进入N1-163六个zero checkpoint保存/恢复前置。

## A版产物路径

- manifest：`plans/manifests/n1_plan163a_transition_template_rebalanced_schedule_20261020.json`
- readiness/source map：`reports/taiji_n1_plan163a_transition_template_rebalanced_readiness_20261020.json`
- builder仍使用`build_taiji_n1_plan163_transition_template_schedule.py`，运行期间固定其A版路径与本addendum SHA。
