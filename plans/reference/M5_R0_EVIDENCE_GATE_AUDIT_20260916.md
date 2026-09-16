# M5 R0 证据封存与门禁一致性审计（2026-09-16）

状态：R0 只读审计完成；P5.2d 修正仪器尚未执行。

本审计只读取当前提交 ec6b9680、已提交报告和工作区未提交产物，不接管或改写用户正在修改的脚本、预注册和报告。它的目的，是把实验真实结果与门禁仪器结果分开，防止把一个门禁布尔值当成模型能力或在线学习结论。

## 1. 结论摘要

- P3b 已停止，不应继续写成“实验臂仍在跑”：对照臂在 17M 因 material 停止；实验臂在 18M 因同一维度连续低于 P3a 被判 persistent 停止。只有 17M 是共同检查点，因此数据分布主效应仍为 not_resolved，不能写成“数据分布无效”或“架构无效”。
- P3b 已产生真实模型输出，但能力明显未达标：18M 快照在同一 CAP-0 链路生成了原始字符序列；C=0、D=0.0625、E=0.10，B/G仍待人工复核，A/F/H未执行。它证明该评测链加载并生成了训练态输出，但不构成可对话模型，更不能触发 L2/L3。
- P5.2d v2 的预算校准信号有效但范围很窄：预算从 10 调到 64 后，第4轮 (member-a, member-c) 的真实终态反馈以 realized_interaction=+2.0 被写入。这只证明“该次真实反馈满足准入并进入记录”，不证明选择排序改善、旧能力保持或在线闭环可晋级。
- v2 的 A1=true 是门禁假阳性：更新选择仍是 member-b+member-c，实际收益与父代均为 0；六个预测值并列 1.291667。代码却用 updated_gain >= parent_gain、并列最高 == max(...)、以及“存在 a+c admitted record”组成通过条件，故得到 true。这个布尔值不能进入能力证据。
- v2 的 A2/A3/A5/G1 失败不能直接归因于模型或库机制：当前工作区 runner 仍使用修订前的检查顺序/计数；预注册 §8 写了修订意图，但 runner 没有同步实现。
- A6 只有部分证据可用：回滚恢复父状态、墓碑存在、环境 undo 的直接观测成立；但 replay 实际得到的是 stale，而当前脚本把任意 ValueError 都当作“不可重放”通过，未证明冻结协议要求的 tombstone/rollback 专属拒绝。
- P5.2d 仍是 failed，在线轴不能晋级。R0 的正确出口是“预算校准发现真实写回信号，但验收仪器未闭合”，不是“再把阈值调大一次”。

## 2. P3b 封存判断

证据：

- [treatment campaign](../../reports/taiji_p3b_campaign_treatment_20260915.json)：17M E=0.10，18M仍为 E=0.10，相对 P3a -0.05，第二个连续低点为 persistent。
- [control campaign](../../reports/taiji_p3b_campaign_control_20260915.json)：17M E=0.05，相对 P3a -0.10，按 material 停止。
- [18M阶段报告](../../reports/p3b_stages/treatment/cap0_tick_18000000.json)：评测链与 P3a 一致，trained_during_eval=false，原始回答已保存。
- [阶段核验记录](M5_P3B_STAGE_VERIFICATION_LOG_20260915.md)：seed_corpus.pt 的保护标志变化已追溯为测试对 tick=2 默认基座的同尺寸重存；真正的 16M 起点 seed_beta.pt 未变。该事实排除了“P3b训练产物被覆盖”的解释，但仍保留 DEBT-I7，因为测试可以写产品检查点。

因此 P3b 分成三条账：

| 账目 | R0判定 | 允许主张 |
|---|---|---|
| campaign生命周期 | 已结束 | 两臂按各自停止规则结束 |
| 数据分布效应 | 不可判 | 只有一个共同tick；heldout surprise同样是not_resolved |
| 语言/行为能力 | 未达 | 18M的C/D/E远低最低线，B/G未完成盲审，A/F/H未测 |
| 保护底座 | 有债 | 默认checkpoint被测试重存，未造成16M训练起点损失，但隔离守卫未闭合 |

该结果不授权第二次 P3b campaign。若未来要比较数据分布，必须另立与题量分辨率、material停止门和共同检查点规则同源的新预注册。

## 3. P5.2d v2 逐门审计

直接证据：[v2报告](../../reports/taiji_p5_2d_online_writeback_v2_20260916.json)；执行记录中的 record.commit 为 56a4c3e4。当前工作区脚本相对该提交只看到预算、输出名、回滚异常捕获和诊断输出等改动，没有实现预注册 §8 所列 A2/A3/A5/G1 修订。

| 门 | 报告值 | R0可用性 | 原因 |
|---|---:|---|---|
| G2 数据隔离 | true | 可暂时采用 | 报告给出不相交的 fit/trial/eval 集合；仍只限本次 run |
| G6 预算 | true | 可采用 | 64预算下本次运行在 wall cap 内 |
| 第4轮反馈 | applied | 可采用为局部事实 | (a,c)真实终态、+2.0、native event进入 learner |
| A1 新任务收益 | true | 无效 | 通过条件允许零收益和并列最高，且不要求更新选择等于 (a,c) |
| A2 旧任务保持 | false | 不可解释为机制失败 | 当前 runner仍要求(a,b)预测逐位不变；库会全局重拟合，§8新判据尚未执行 |
| A3 幂等 | false | 未测 | replay发生在所有轮次之后；apply_feedback先检查stale再检查duplicate，故得到stale |
| A4 stale parent | true | 可采用为局部契约事实 | 当前顺序下确实拒绝过期parent |
| A5 中断恢复 | false | 未测 | applied_ids比较使用全量列表切片；pending还混入被拒绝反馈，仪器计数未按实际pending applied集合定义 |
| A6 rollback/undo | true | 部分可用 | 父状态恢复、墓碑、环境撤销可见；replay原因是stale，脚本只要求任意ValueError |
| G1 干预真实性 | false | 不可解释为真实干预失败 | 计数在 A6 rollback 后才读取，rollback会把applied改为rolled_back |
| outcome | failed | 保留 | 六类验收未闭合；不得改成partial或supported |

## 4. 三个关键矛盾的根因

### 4.1 为什么“反馈已写入”但选择没有改善

本次反馈写入与排序改善是两件事。

1. 第4轮 (a,c) 的反馈满足成功、资源和事件条件，observe_records确实把一条 admitted record加入learner。
2. 更新后报告显示六个pair预测全部为 1.291667，不是 (a,c)严格高于其它pair。
3. updated_pair仍为 member-b+member-c；选择器在预测并列时按后续稳定排序键选中该pair。
4. A1只检查：
   - updated_gain >= parent_gain，本例是 0 >= 0；
   - (a,c)预测等于最大值，并列也成立；
   - 存在 (a,c) admitted record。

所以 A1 的 true 是“反馈存在＋并列不差”，不是“在线更新改善了选择”。目前能确定的根因是门禁判据缺少严格的排序/收益条件；至于为何一次正反馈不能产生区分性排序，需在修正仪器后单独观察，不能由本次假阳性结果直接归罪架构。

### 4.2 为什么 A2/A3/A5/G1 不能按报告解释

- A2：InteractionGroupTransferLearner._fit每次 observe_records都会对全部 records重拟合，所以 (a,b)预测从 0.583333... 变为 1.291666...。这证明旧的“预测逐位不变”形态与当前全局拟合语义冲突；它不等于旧任务执行一定退化。工作区脚本虽计算了 ab_records_before，却没有完成 §8声明的前后record计数判据。
- A3：库的 apply_feedback 在 online.py 403–415 先判 parent stale，再判 duplicate。所有轮次完成后重放旧 feedback，必然优先得到 stale；这不是有效的 duplicate 测试。
- A5：工作区脚本 303–304 把第4轮之后的当前 feedback追加到 pending，不区分本轮是否 applied；429–432又用 online.applied_feedback_ids[RECOVERY_AFTER_ROUND:] 作为期望列表。fresh child 的 digest相等只能说明某条重放路径与当时父进程一致，不能覆盖这个应用ID比较错误。
- G1：533–541读取 len(online.applied_feedback_ids)的位置在447–484的 A6 rollback之后；回滚合法地把最新 admission标为 rolled_back，因此计数不再等于此前收集的 applied_feedbacks。
- A6：工作区修改把原先要求的具体拒绝原因替换成“捕获任意 ValueError”。报告中的 replay_note 是 online feedback parent checkpoint is stale，不是 candidate_blocked_after_prior_rejection_or_rollback，因此只能证明被拒绝，不能证明墓碑语义被测试到。

### 4.3 预注册与执行器的关系

工作区预注册 §8 是执行后的修订说明，包含五项仪器修正；但当前 runner diff没有对应实现。因此它应被标记为“待实现的修订设计”，不能被当作本次 v2已经完成的实验条件。历史 v1报告和v2报告均保留，不覆盖、不回写。

## 5. R0出口与范围判定

| 工作包 | R0后状态 | 是否晋级 |
|---|---|---|
| P3b | campaign已结束；分布效应不可判；语言输出能力未达 | 否 |
| P5.2d预算校准 | 发现真实反馈可写入；验收仪器有假阳性/不可判门 | 否 |
| 协作/选择轴 | B2族内研究资产保留，在线轴证据未闭合 | 否 |
| M5 | 知识、在线/身体共同门未闭合 | 否 |
| L2/L3模型验收 | P3b原始输出已可观察但远未达标；不得交用户验收 | 否 |

R0可以关闭“状态不清”问题，但不能关闭在线回写工作包。R0没有授权修改 learner 核心机制，也没有授权再跑预算64或第二次语言campaign。

## 6. 决策节点：R1的唯一推荐动作

推荐新建一个版本化的 P5.2d 修正仪器批次，不改写 v1/v2历史：

1. 将 A1 改为严格的“更新选择确为 (a,c)、新任务实际收益严格高于父代且超过零/预注册最小效应”的可证伪条件；若项目原意允许“持平”，必须另设“无退化”辅助门，不能叫新任务收益通过。
2. 实现预注册 §8 已写出的 A2 record-count＋held-out execution 口径，并把全局预测漂移作为单独披露字段。
3. 在首个 applied admission后、任何后续 admission前立即做 A3 duplicate；A5按实际pending队列逐条记录 applied/rejected；G1在 rollback前采样。
4. A6要求具体拒绝原因与墓碑状态，不能以任意异常代替。
5. 先跑库级/门级小测试与静态检查，再由独立冻结件授权一次新门运行；新报告使用新版本号和新文件名，v2只作问题发现证据。

这是仪器/验收合同修正，不是自动批准新的研究结果。修正完成后再决定是否值得追加在线预算；在此之前不应继续扩大训练或把R2架构问题与P5.2d仪器问题混在一起。

