# PLAN-N1-161：独立问句句式迁移诊断

**状态：已依用户授予的执行计划批准权批准并冻结；实现验证中**
**日期：2026-10-20**
**依据：** [N1-160A执行结案](PLAN-N1-160A_execution_closeout_20261020.md)

## 目的与证据边界

N1-157A2、N1-159A、N1-160A在三张互斥的程序生成face上复现了同一结论：N1-155A `unknown8_rebalanced` candidate的unknown精确弃答高于同seed N1-150 `length_mixed` parent，已知类总分接近，但这些face复用了训练期问句表面`material of {entity}?`与`direction of {entity}?`。下一步只检验冻结candidate/parent能否在新实体、新材料答案值及相同观测结构下，迁移到结构不同的问句句式。

本方案是同类程序生成任务上的**问句模板迁移诊断**。通过结果只能支持或反驳这组模型对本方案问法的迁移，不得表述为自然语言泛化、真实用户能力、训练配方因果效应或N系列出口证据；`can_promote=false`。

## 冻结endpoint与题面

- 使用N1-155A三个`unknown8_rebalanced` candidate final及对应N1-150 `length_mixed` parent，共6个冻结endpoint；不训练、不写checkpoint。
- 新face seed131，lookup/revision/transition/unknown各40题；每类20题分配到两个固定模板，每模板20题。题目按确定性slot分配，不抽样、不重试。
- 每题保留8/16/32行嵌套观测。lookup答案位于第0行；revision的旧/新事实位于第0/1行；transition的朝向/转向事实位于第0/1行；unknown目标在观测中无事实。与N1-160使用相同的任务语义、证据行形态、实体近邻构造及free/teacher计分函数，题面实体与材料答案值重新生成。
- 材料类（lookup、revision、unknown）两种问句：`what material does {entity} use?`、`which material is used by {entity}?`。
- transition两种问句：`which direction does {entity} face after turning?`、`what direction is {entity} facing after its turn?`。
- 四种完整模板在N1-150/N1-155A训练source中不得出现。题面仍为程序生成英文句子；只将查询句式作为本轮变化维度。保留逐题question、template id、answer、context、输出字节及评分字段。
- 总量160题×3长度×6 endpoint＝**2,880条查询**；每endpoint480条。汇总固定为6 endpoint×4类别×3长度×2模板，共144个分层。

## 来源、前置与一次性执行

- 复用N1-160已核验的来源并集与六个endpoint，额外排除seed130 face的实体及材料值；并绑定N1-160A完整结果与seed130 face SHA。新题实体、材料值及随机观测材料值必须与既有来源无交集；transition沿用原合同中的固定direction/turn词表。readiness还需从v3训练题清单确认本方案四种问句模板均未出现。
- source readiness只做来源/代码/题面模板静态核验，0模型加载、0模型查询、0参数更新。
- readiness提交并绑定后，对6个完整bundle各做一次独立保存恢复和fingerprint比对。六项全通过后才一次生成seed131 face；face合同与逐题语义审计通过后才启动评分。
- 评分仅对6 endpoint的2,880条固定新题各执行一次；候选和父档用同一题目配对。不设中途范围筛选，不因读数好坏追加或重抽。
- 任一前置失败即停止，不消费face；face生成或评分开始后如遇失败，封存attempt/progress/result，禁止覆盖、补测或自动重试。

## 指标、预算与停止边界

- 主读数：按endpoint、类别、上下文长度、模板报告`free_exact`及candidate/parent逐题配对的candidate-only/parent-only；`teacher_exact`与逐token输出作辅助诊断。
- 先报告两种模板各自结果，再报告合并结果；不得把一个模板的提升掩盖另一个模板的失败。没有晋级阈值，本面仅给描述性迁移读数。
- 本阶段CPU≤3,600秒；N1-160A后关联CPU基线10,700.171875秒，整个关联上限18,000秒。超预算立即停止并封存。
- 0训练、0参数更新、0 dev、0 holdout、0 confirmation；不读取或复用保留留出内容，不改变既有N1-132/N1-155门。

## 实现与输出位置

- face builder：`scripts/training/build_taiji_n1_plan161_seed131_question_surface_face.py`
- 诊断runner：`scripts/training/run_taiji_n1_plan161_seed131_question_surface_diagnostic.py`
- 契约测试：`tests/taiji_native/test_n1_plan161_seed131_question_surface_diagnostic_contract.py`
- seed131 manifest：`plans/manifests/n1_plan161_v33_question_surface_seed131_20261020.json`
- 所有机器审计、attempt、progress及result均写入`reports/`；不在根目录或任意临时文件夹新建产物。
