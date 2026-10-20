# PLAN-N1-159：seed129 candidate-versus-parent fresh replication

**状态：已依据用户授予的执行计划批准权批准，待实现与执行**  
**日期：2026-10-20**  
**依据：** [N1-157A2结案](PLAN-N1-157A2_execution_closeout_20261020.md)、[N1-158逐题归因](PLAN-N1-158_execution_closeout_20261020.md)

## 目的

seed128 fresh face上，N1-155A candidate相对N1-150父模型的overall exact高60/1,440；已知三类合计低2/1,080，unknown在8/16/32行分别多31、17、14项。需要在第二张独立fresh face上复制这项unknown差异，避免仅凭单面判断再平衡的稳定性。

## 冻结比较与新测面

- 新seed129 face，题型、类别比例、答案长度配比、8/16/32嵌套上下文和seed128相同：每类40题、每endpoint 480次、六endpoint最多2,880条。
- 六endpoint固定为seed11/12/13的N1-155A `unknown8_rebalanced` candidate final，与各自N1-150 `length_mixed`父模型。
- 题面实体名、材料值与观测词值完整排除N1-157 seed128 face、N1-153A seed127 face、N1-150训练生成值、全部已消费训练/诊断/确认面。建面时逐项验证目标/干扰唯一、Hamming d=1合同、跨面无重叠及来源SHA。
- 六个checkpoint先做独立full-bundle恢复与指纹复核；均通过后才生成seed129 face并一次性查询。
- 保存2,880条逐题teacher/free结果和原始字节，按endpoint/长度/kind汇总，并输出candidate-parent逐题配对转移。比较只作独立复制读数，不改变N1-155或N1-132门。

## 资源与边界

- PLAN-N1-159阶段CPU≤3,600秒，涵盖readiness、六bundle恢复、建面和诊断查询。
- 上位已用关联CPU为9,771.53125秒；N1-159完成后的关联CPU不得超过18,000秒。
- 不训练、不更新参数，不访问dev、holdout或confirmation；`can_promote=false`。结果仅为第二张fresh synthetic face上的描述性复制，不能单独晋级能力或声称因果机制。

## 实现与输出

- 建面/来源审计：`scripts/training/build_taiji_n1_plan159_seed129_candidate_parent_face.py`
- 执行器：`scripts/training/run_taiji_n1_plan159_seed129_candidate_parent_diagnostic.py`
- 契约测试：`tests/taiji_native/test_n1_plan159_seed129_candidate_parent_diagnostic_contract.py`
- seed129题面：`plans/manifests/n1_plan159_v31_candidate_parent_seed129_20261020.json`
- 审计、readiness、preflight、attempt、progress和result均写入`reports/`；复用既有checkpoint，不新建模型权重。

提交代码、测试和本计划后，先做只读来源ready，再做六bundle独立恢复前置；通过后生成fresh face并消费一次。任何write-once阶段失败均封存回执，不覆盖或自动重试。
