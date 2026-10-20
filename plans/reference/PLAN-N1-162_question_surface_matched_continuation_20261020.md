# PLAN-N1-162：问句表面配对续训与未见句式复测

**状态：已依用户授予的执行计划批准权批准并冻结**
**日期：2026-10-20**
**依据：** [N1-161结案](PLAN-N1-161_execution_closeout_20261020.md)、[N1-161A逐题归因](PLAN-N1-161A_question_surface_itemwise_attribution_20261020.md)

## 目的与边界

N1-161在独立问句面上得到candidate 441/1,440、parent 460/1,440；unknown提升只出现在seed13，已知类下降。逐题归因显示主动式revision candidate仅7/180、parent32/180，candidate有47条输出与旧revision材料完全相同；被动式revision则为51/180对49/180。下一步以**问句表面训练增强**为唯一新变量，检验它能否在保留旧问法训练表现的同时改善另一组训练未见问法迁移。

本方案不复用N1-161题目、实体或材料值，不把N1-161测试问法样本用于训练；只把其两个问句表面作为训练source形式。新seed132 face使用另外四种holdout问法。结果仍为程序生成的英文诊断，不代表广泛自然语言泛化或真实用户能力；`can_promote=false`。

## 训练比较

- 对seed11/12/13的N1-155A `unknown8_rebalanced` final完整bundle分别分叉两个配对续训臂：`surface_control`与`surface_augmented`。每对从逐位相同的模型/Adam/RNG/history起点恢复；control始终使用原v3问句，treatment只改变训练问句表面。zero/final bundle内保留原N1-155A `branch`元数据；两个新arm标签作为外部实验字段，不写入模型payload或改变fingerprint算法。
- 两臂使用完全相同的192个v3 train父任务、答案、观察内容、target evidence、N1-155A objective/optimizer/学习率/seed、raw/hard来源槽及每父项48更新。每endpoint固定9,216更新，六endpoint共55,296更新。
- control的48/48 slot均用原v3问句。treatment每父项24/48保留原问句，另外24/48均分为N1-161使用过的active/passive材料问法或两种direction-after-turn问法，各12/48；仅改question bytes，不换任务或答案。每个上下文长度内固定配比分配：known每长度16 slot按8/4/4分配原句/模板A/模板B；unknown的8/16/32行分别24/12/12 slot，对应按12/6/6、6/3/3、6/3/3分配。模板标签按现存slot顺序确定性平衡分配，不重排raw/hard来源槽；paired unknown父项的两条语义配对问题使用相同slot模板选择，观察带与目标关联不变。
- N1-132原始192项gate及N1-150长度评分在最终训练bundle上原样计分：每endpoint 960+288条，六endpoint共7,488条train-only评分。六endpoint必须全部通过原N1-132训练gate，才允许建face；不改变任何历史分母、门槛或promotion状态。

## checkpoint、题面与评分

- 训练前为六个zero full bundle写入独立write-once路径，并独立保存/恢复6/6，核对模型、Adam、RNG、history及fingerprint；任何失败均为0更新并停止。zero checkpoint另行确认control/treatment两个外部arm在同seed下逐位同起点。
- 训练后六个final full bundle独立恢复6/6，并复算7,488条train-only分数。任一N1-132 gate失败即封存，不生成测试face。
- 六门全过后一次生成seed132 fresh face：四类各40题，每类两个holdout问句模板各20题；8/16/32行嵌套上下文；每题facts/answer和目标证据位置保持N1-161结构。材料模板为`what material is associated with {entity}?`及`which material belongs to {entity}?`；transition模板为`after turning, what direction does {entity} face?`及`what is the final direction for {entity}?`。readiness必须验证这些模板未出现在v3训练题及N1-162 treatment问句中。
- seed132 face source exclusion绑定seed128/129/130/131全部fresh face及所有既有训练/诊断来源，实体名和随机材料/观测值无交集；保留transition固定direction/turn词表。建面合同与逐题语义审计通过后，对12 endpoint计分：N1-150 parent×3、N1-155A candidate×3、surface_control×3、surface_augmented×3。共**5,760条新查询**，四组均使用同一新题逐题配对；不把N1-161旧面重测。评分前对12个endpoint完整bundle作独立只读恢复预检12/12，复核来源/checkpoint SHA不变。
- 面结果按task×template×length×seed报告free/teacher exact、unknown弃答及candidate-only/parent-only转移；主要观察surface_augmented相对等量surface_control的heldout-template差异。此项没有能力晋级阈值，不能替代现实语言能力门。

## 资源、守卫与停止

- 新阶段CPU≤6,000秒；N1-161关联CPU基线11,058.34375秒，按stage上限计算关联累计≤17,058.34375/18,000秒，预留约941.656秒余量；任一stage或linked硬限先到即停止。
- 所有zero/final checkpoint与face/result采用新write-once路径。面生成前核对12个计分endpoint的完整bundle和来源SHA；zero/final训练bundle分别独立恢复6/6。
- 任一来源、保存恢复、训练门、语义合同、分母、progress或预算守卫失败，立即封存attempt并停止；不覆盖、不补训、不重抽、不重测。
- 0 dev、0 holdout、0 confirmation；所有评分限于v3 train-only和本方案一次性diagnostic face；不晋级，`can_promote=false`。

## 实现与输出位置

- schedule/source/face builder：`scripts/training/build_taiji_n1_plan162_question_surface_schedule_and_face.py`
- matched training/scoring runner：`scripts/training/run_taiji_n1_plan162_question_surface_matched_continuation.py`
- 契约测试：`tests/taiji_native/test_n1_plan162_question_surface_matched_continuation_contract.py`
- seed132 manifest：`plans/manifests/n1_plan162_v34_question_surface_seed132_20261020.json`
- write-once checkpoint根：`data/checkpoints/n1_plan162_question_surface_20261020/`
- readiness、保存恢复、进度及结果均写入`reports/`。
