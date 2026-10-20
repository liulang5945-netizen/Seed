# PLAN-N1-160：seed130 candidate-versus-parent fresh replication

**状态：已依据用户授予的执行计划批准权批准，待实现与执行**  
**日期：2026-10-20**  
**依据：** [N1-157A2结案](PLAN-N1-157A2_execution_closeout_20261020.md)、[N1-159A结案](PLAN-N1-159A_execution_closeout_20261020.md)

## 目的

seed128与seed129两张source-excluded fresh face均显示N1-155A candidate相对N1-150同seed `length_mixed`父模型的unknown正确弃答较高，已知类总分近乎持平。批准第三张独立fresh face seed130复制该paired结果，以检验趋势是否跨face保持；本次仍是描述性合成诊断，不是promotion gate或现实能力确认。

## 冻结测面与比较

- 生成唯一新seed130 face：lookup/revision/transition/unknown各40个base题，每题嵌套8/16/32行上下文；6个endpoint各480条，共2,880条。
- endpoint固定为seed11/12/13的N1-155A `unknown8_rebalanced` candidate final与各自N1-150 `length_mixed`父模型。
- source exclusion包含全部历史训练与已消费诊断/确认面，特别绑定并完整排除seed128和seed129两张face的实体名、材料值及观测词值；同时保留N1-153A seed127和N1-150训练生成值排除。建面合同验证新旧目标/干扰词唯一、d=1、跨面无交集及来源SHA。
- readiness通过后，先独立恢复六个完整bundle并逐项比fingerprint；只有前置全部通过才生成seed130 face并评分。逐题保存teacher/free输出、停止原因和原始字节，保留branch×seed×类别×长度汇总。

## 资源与边界

- PLAN-N1-160阶段CPU≤3,600秒，覆盖source readiness、六bundle恢复、建面和2,880条查询。
- 已用关联CPU为10,229.453125秒；N1-160后关联CPU不得超过18,000秒。
- 不训练、不更新参数，不访问dev、holdout或confirmation；不调整N1-132/N1-155判据，`can_promote=false`。结果只用于第三面paired复制。
- 任一write-once阶段失败即封存回执，不覆盖、不自动重试。

## 实现与输出

- 建面/来源审计：`scripts/training/build_taiji_n1_plan160_seed130_candidate_parent_face.py`
- 诊断执行器：`scripts/training/run_taiji_n1_plan160_seed130_candidate_parent_diagnostic.py`
- 契约测试：`tests/taiji_native/test_n1_plan160_seed130_candidate_parent_diagnostic_contract.py`
- seed130题面：`plans/manifests/n1_plan160_v32_candidate_parent_seed130_20261020.json`
- 报告及progress写入`reports/`，checkpoint沿用现存N1-150/N1-155A文件，不新建模型权重。
