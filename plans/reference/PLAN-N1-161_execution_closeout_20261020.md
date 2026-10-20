# PLAN-N1-161 执行结案

**日期：2026-10-20**
**状态：完成；seed131问句句式迁移诊断已一次性消费**
**执行计划：** [PLAN-N1-161](PLAN-N1-161_unseen_question_surface_diagnostic_20261020.md)
**seed131题面：** [face manifest](../manifests/n1_plan161_v33_question_surface_seed131_20261020.json)

## 完整性、来源与资源

source readiness绑定692项来源，包含N1-160A完整结果及seed130 face；352条v3训练题的两个绑定manifest里，四种新问句模式均为0匹配。排除并集含169,202个实体名和131,618个材料值。六个完整bundle独立恢复6/6，score时六个checkpoint再次加载6/6；fingerprint/source哈希均稳定。

seed131 face包含160题，lookup/revision/transition/unknown各40题；每类两个模板各20题，8/16/32行均为嵌套上下文。面合同、问题目标、证据位置、source exclusion与模板配比均通过。之后对6个冻结endpoint完成2,880条查询，144个endpoint×类别×长度×模板分层齐全。progress有2,880行、无坏行，SHA-256为`dd80604cd66519f04bfe48d5929756e0f2f5436a68435d3a5df7a31068f66cce`；合并逐题结果与progress末端行数和内容一致。

source readiness、六bundle恢复、face生成审计、评分分别耗用20.578、103.984、43.484、190.125 CPU秒，本阶段累计358.172/3,600秒；关联累计11,058.344/18,000秒。0训练、0参数更新、0 dev/holdout/confirmation，执行无错误，`can_promote=false`。逐题result SHA-256为`2808a969299143c692643bf4bd5ccabdd67a63021f5ad9ba0cab65bcb51a385a`。

face生成已成功一次。随后为确认先前长命令状态而重复调用`--prepare-face`，write-once守卫拒绝该重复请求；没有覆盖face/audit/checks，也没有发出查询。原成功产物和审计SHA保持有效。

## 冻结模型的问句迁移读数

下表是候选与父档在同一seed131题目的free exact，分母均为180。

| 任务类别 | 新问句模板 | candidate | parent | 差值 |
|---|---|---:|---:|---:|
| lookup | `what material does ... use?` | 57 | 69 | −12 |
| lookup | `which material is used by ...?` | 45 | 56 | −11 |
| revision | `what material does ... use?` | 7 | 32 | −25 |
| revision | `which material is used by ...?` | 51 | 49 | +2 |
| transition | `which direction does ... face after turning?` | 89 | 89 | 0 |
| transition | `what direction is ... facing after its turn?` | 83 | 80 | +3 |
| unknown | `what material does ... use?` | 60 | 48 | +12 |
| unknown | `which material is used by ...?` | 49 | 37 | +12 |

合并后candidate总命中441/1,440，parent 460/1,440（−19）。unknown为109/360对85/360（+24），已知类为332/1,080对375/1,080（−43）。seed11/12/13总命中candidate分别120/480、79/480、242/480，对应parent 125/480、113/480、222/480；unknown分别为0/120、17/120、92/120，对应0/120、20/120、65/120。因此unknown净增全部来自seed13，seed11持平、seed12下降；不能声称模板迁移在三个seed上重复成立。teacher exact与free exact在2,880行完全一致。

结果指出，N1-155A的unknown增益在未见问句面上只出现于seed13；候选已知类明显退化，其中主动式材料问法下revision为7/180，而被动式为51/180。三张同模板fresh face上观察到的优势不能直接外推到问句句式迁移。seed131仍是程序生成英文诊断，不能代表广泛自然语言泛化或真实用户能力；`can_promote=false`。

逐题错误类型尚未归因；[完整结果](../../reports/taiji_n1_plan161_question_surface_result_20261020.json)逐行保留question、template id及输出字节/评分，答案与观测上下文可由`parent_id`关联到[seed131 face manifest](../manifests/n1_plan161_v33_question_surface_seed131_20261020.json)，可进行不加载模型的离线分类。

## 后续唯一队首

依据用户授予的执行计划批准权，已批准只读的[N1-161A逐题错误来源归因](PLAN-N1-161A_question_surface_itemwise_attribution_20261020.md)：仅用本结果与seed131 face分类各模板错误输出及旧revision值复现情况；CPU≤30秒，0模型/checkpoint加载、0新查询、0训练、不访问dev/holdout。Ruff、py_compile及3项契约测试通过，接下来提交分析器后执行一次；本诊断不晋级任何模型。
