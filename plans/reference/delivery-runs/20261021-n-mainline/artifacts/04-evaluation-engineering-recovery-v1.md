# CC-1 之后的评价工程恢复合同 v1

日期：2026-10-25，Asia/Shanghai。状态：**恢复入口已实现、真实无模型握手与合成通过；正式恢复协议已获用户一次批准，待预检后执行**。本合同承接[03](../../../../active/roadmap/03_CURRENT_EXECUTION.md)的唯一队首，只为取得已训端点的真实配对表现；不训练、不换模型、不补发能力通过。原CC-1的CLOSED_FAILURE及已消费资格不改。

## 1. 本次已完成的工程证据

- 缺陷与修复继承`d5b0aba51`：worker命令由库模块改为有`--worker`入口的版本化模块。
- 新工具`python -m scripts.training.check_taiji_r2_evaluation_recovery --probe-bootstrap --output reports/<新文件>.json`先只读核对原失败，再在`.seed_test_tmp/`中建立一次性合成attempt/空ledger，通过真实Windows Job Object与现行入口验证身份握手，随后主动终止并确认整个子进程树退出。
- 不发送restore/generate/shutdown任务；本次shutdown不伪装成“四端点已恢复后的正常退出”。探针退出码1来自主动Job终止，成功依据是认证ready与已确认退出，而不是退出码0。
- 修复后原入口的[准备回执](../../../../../reports/taiji_r2_evaluation_recovery_preparation_20261025_v1_1.json)：ready通过、树退出通过、0模型任务、0恢复、0题查询、0参数更新。新恢复入口也已完成同样的[真实bootstrap](../../../../../reports/taiji_r2_evaluation_recovery_v1_bootstrap_20261025.json)，仍为0模型任务。格式化前的[首次回执](../../../../../reports/taiji_r2_evaluation_recovery_preparation_20261025.json)也保留，费用不丢弃。
- 本次不是metadata预检的另一种命名，实际执行了修复后的子进程/Job/IPC装配；仍不证明模型恢复或生成通过。

## 2. 原失败、题面与来源审计

旧attempt ID为`abffec0c23f5274b382a06ce45f47197688d061440e0080e`。工具核对attempt/failure/result身份，保留四份原文件SHA，前后再次校验原字节未变。

原ledger只有attempt、worker startup、evaluation stopped和追加startup tail audit；没有restore、selector、face或query事件，face/generations不存在，query与output均0，`fresh_process_restores={}`。原冻结执行链先握手，再恢复四端，之后才调用selector；本次证据支持**这一次受监督尝试在选择题面之前停止**，不是整台机器或全部历史访问的审计证明。

新旧source不能混用：当前端点准备器、adapter及其测试与旧attempt绑定SHA不同；它们是已有后继修复，不能用旧readiness冒称新启动来源。新恢复必须重新冻结修复后的完整source map；冻结evaluator/selector与原失败记录保持原字节，不靠修改旧guard或全局monkeypatch放行。

## 3. 固定候选与评价范围

四端点、顺序和checkpoint身份继承原attempt，实际执行前重新核验文件SHA；不得换成中间checkpoint或换decoder。

| 端点 | checkpoint SHA256 |
|---|---|
| parent_baseline | `41655eb8b5fcd7e446acb924e94e458ce9b1e966a96a95517198a5ac38df2086` |
| full_turn_1m | `5654b9eac004e98525e55661e6606be8de1f40027bebfa547ae02ee378efa4ca` |
| warm_start | `c8a1068f3e0886b46694cabd2b0c03e07c89d8dbd182e654e4599073a528cd50` |
| fresh_same_capacity | `86b9167507f53daec2a57956a01bb054dfe2f4c041ba5852414b04e3f78989e5` |

拟恢复范围仍为39 family×4端点=156条原始输出，前12 family×4=48回答固定人工判读；固定中文可读性、贴题、历史连贯和重复四维。逐端点39，reuse=0，参数/优化器更新0，不进行reference/exact-answer评分，`can_promote=false`。

拟采用原开发面及原selector，理由是本次故障在selector前；它仍只提供开发面上的描述性配对证据，不冒称未见独立确认，不触碰N1-166或其他封存面。**这项复用须在正式恢复版本中明确承接权限**；若新审计发现已暴露/已消费的不同事实，应回到合同裁决，不能偷偷换面、换分母后继续。

## 4. 费用：保留未知，不因一ULP自动清零

- 原失败发布时测得CPU60.78125秒、wall63.953秒；追加audit仍为unknown，charged upper为空。原始两次boot identity相差`2.384185791015625e-7`秒，恰为该数量级一ULP；现场当前boot又与原before值相同。这支持“表示精度差异”的解释，但本合同不修改冻结比较器，也不把未知项改成已测0。
- 旧host CPU读数差414.453125秒可作同boot计量的候选保守上界，不能称它是该worker的精确CPU；正式恢复前需给该计量方法及boot证据一个明确处置。
- 另一种保守预留提案：启动至追加审计138.542240秒，若核验该窗口逻辑CPU上界为现场24，则窗口上界3325.013760 CPU秒；再完整加上60.78125测量以允许重叠收费，总3385.795010，取**3400 CPU秒历史预留**。已核对2026-10-22训练合同三臂`accounting_start.logical_cpu_upper=24`与现场`os.cpu_count()=24`；正式恢复前仍核验同机/拓扑适用性，不把该条件上界称精确实耗。
- 三次真实握手探针（修复原入口两次、新恢复入口一次）父CPU合计5.312500秒、子CPU合计5.125000秒，总10.437500 CPU秒，wall合计12.313秒；准确未知的旧尾账继续与已测值分列，不改训练三臂4.56M累计账。

为避免再逐次小额提额，拟在原评价**CPU/wall各14,400秒的同一总账**内一次分配：

| 用途 | CPU上限 | wall上限 | 包含内容 |
|---|---:|---:|---|
| 历史失败保守预留 | 3400 | 180 | 上述条件上界；不是最终实耗声明 |
| 恢复准备 | 240 | 300 | 两次探针、来源/尾账/资源检查与合同准备 |
| 正式恢复一次 | 10500 | 13000 | 自身metadata预检、握手、四端恢复、156生成、IPC/停止与原300秒发布预留 |
| 总计 | 14140 | 13480 | 低于14,400；未用余额不当作追加查询授权 |

继承的完整执行保守reserve为10200秒，加cancel 5秒；正式入口须实测确认10500秒足以覆盖预检、startup及该reserve，不能为放行而压低固定单次上界。历史CPU上界/准备成本任一无法证明被分配额度覆盖，则暂不形成正式attempt，只报告具体缺项；不改原已消耗账，也不转为训练预算。

## 5. 正式恢复入口已实现的边界

已实现`taiji_r2_evaluation_recovery_v1.py` provider及`evaluate_taiji_r2_engineering_recovery_v1.py`入口。现行extension worker新增显式可选provider/path注入，默认行为保持；新parent/worker都接同一恢复source/path。以下接线已交付，正式恢复消费仍关闭：

1. 用明确的恢复协议ID绑定本合同、旧attempt/failure/ledger SHA与新的source快照；授权字段区分“旧失败保留”“一次新的恢复资格”，不能仅把旧输出prefix改名。
2. 新parent/worker显式共用恢复provider与path，已修正旧worker回查旧path的缺口；未修改冻结evaluator或全局monkeypatch。
3. 用新budget_factory约束正式阶段10500 CPU/13000 wall，保留冻结的单任务上界和完整reserve，历史/准备分配单列计入原14400总账；合成检查包括旧attempt已消费但恢复授权缺失、source漂移、超预算、重复恢复、parent/worker snapshot不同、无四端receipt读题等合成拒绝检查。
4. `python -m scripts.training.evaluate_taiji_r2_engineering_recovery_v1 --validate-only`及`--execute`均先检查一份机器合同的明确恢复授权；用户已明确批准本合同，授权字段随本次批准启用；先validate-only冻结完整快照，再execute一次；执行仍先四端点恢复和receipt，再读selector/生成，最后固定判读。

机器合同为`reports/taiji_r2_evaluation_recovery_v1_contract_20261025.json`，固定旧失败SHA、四端点SHA、新source closure、分母、历史/准备预留及正式阶段额度；`authorizes_new_attempt/same_dev_face_reuse_approved/historical_tail_reserve_accepted`是一项协议裁决的三个记录，不是三轮审批。源码/端点/账本有漂移就拒绝，不静默重算snapshot。

## 6. 停止与结案

正式恢复只允许一个attempt；中断/工程失败保留原始错误、消费范围和所有费用后停止，不补题、不重试、不继续最后一条。完整恢复以156耐久输出、48回答四维判读、身份/保持覆盖/实际费用及未知项落盘后结束；改善、无收益、退化、混合、不可判均可结束，不为取得好结果追加训练。

若仍在worker或恢复阶段失败，只有工程结论，不能填模型0分或直接判J1无效。只有真实行为证据具备后，才进入CC-2/VX0选择必要机制实验；恢复包不要求完成生命页、整个VISION或全库CI维护债。

## 给下游

用户已回复“批准合同v1，启动一次恢复评价（推荐）”，一次接受恢复资格、原开发面复用及同账方案。唯一下一步：预检通过后按本版本执行一次，固定156输出/48回答；工程入口和合成已完成，不再重复确认。需要此裁决是因为旧03合同明确attempt即消费、失败不重试，新的正式查询会改变该一次性协议；当前“下一步”的可逆准备不被当作静默重开旧资格。本次批准只开启本版本一次，按§6有限结案；不代表能力通过。
