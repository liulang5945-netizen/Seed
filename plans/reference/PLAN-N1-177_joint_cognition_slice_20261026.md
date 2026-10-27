# N177 最小共同认知纵切片：实施依赖与接口合同

日期：2026-10-27。状态：N177.1及N177.2-T1/T2/T3 opt-in共同状态、记忆/世界/选择/语言消费链已通过有限工程/因果接口检查；N177.3A恢复/只读路由与N177.3B六类更新权限规格已通过有限静态出口。实际参数更新、更新后恢复、行为学习对照和模型能力仍未验收；没有参数更新、外部动作或新封存读取。

## 1. 本次改变的依赖

原先容易被理解为“独立语言训练→语言通过→再加世界/记忆/规划”的排期退出。新的次序是：**共同状态与生命周期→任务所需记忆/预测/选择→同状态的表达与反馈责任→在这同一候选上形成基础语言和语义→连续任务、跨会话学习及保持**。

N176三臂已经完成，仍按原一次评价有限收束，用于解释课程顺序、V4读取及有限保持；成功、负结果、不可判或工程失败均有出口，不补训求通过。其评价结案与本包接口/静态工程可以并行；N176正收益、N175剩余人工S、独立语言L2通过都不是修复已知共同链欠账的前置。N177正式训练前须纳入N176实际结案/未完成范围及费用，不能假装前轮没有发生。

本包整体要覆盖共同链，但F0复制等子任务只激活所声明的必需路径；不要求先实现完整认知器官、驱动力或全自主性，再允许基础训练。跨阶段能力等级与验收数值继续由07管理，本合同不降低或改写它们。

## 2. 唯一推荐的实现底座

| 可选方向 | 已有基础及代价 | 裁决 |
| --- | --- | --- |
| 继续在V4语言头上加更多教材 | 数据/读取候选已可恢复，但仍不能证明世界、记忆选择和规划驱动语言 | 保留为对照，不作为共同认知实现 |
| 只把旧模块引用拼到一个结构体 | 已有CognitiveState等合同，但同一ref/维度不等于语义一致；标量摘要可能丢掉决定性内容 | 不以装配成功结案 |
| **共同内容状态＋连续表示＋精确来源引用，所有任务消费者和反馈登记在同一候选** | 复用现有合同、持久化和可兼容权重；必要时改表示与主计算/学习链，工程量较大 | **唯一推荐** |

以`contracts.CognitiveState`的认知所有权和`WorldObject/WorldEvent/WorldState/ActionIntent/Outcome`身份合同为根，演进同一bundle的可学习内容组件。V3/V4的TaskState只是已有神经证据载体；它与旧器官的共享关系要显式映射、迁移、校验，不能并列维护两套互相覆盖的世界真值。需要新增表示时版本化，不强行把所有语义塞入现有256维或多个汇总标量。

既有V4接口可用于装配基线；其已训练权重是否继承由N176结案与兼容性/保持决定。没有收益或保持证据时，保留已恢复的N175→V4零更新迁移作父候选，不挑新seed反复求好结果；新模块仅明确初始化不兼容部分。所有旧实耗继续记录。

## 3. 第一条真实联合任务

限制为两个对象、一项属性、两个位置、最多两个候选动作及一步后果，不引入专业常识。用三个相互衔接的任务族逐步激活共同链：

| 任务族 | 输入及模型必须完成的事 | 为什么相应模块不可省略 |
| --- | --- | --- |
| T1 事实/更正问答 | 原始中文事实→对象属性/位置绑定；新事实更正后回答当前值 | 输出读取共同内容绑定；改相关事实应改答案，改无关事实应保持 |
| T2 经历/情景选择 | 早前经历写入后，当前prompt不重复答案；跨turn/reset或恢复后回答所需过去/当前事实 | 必须实际选择正确来源/时间的经历；给错/切断记忆会定向损害，不能靠重放完整答案上下文绕过 |
| T3 目标/行动/后果 | 根据此前获准经验预测两个动作各自后果，按当前目标选择，并用中文说明实际意图/结果 | 动作ID/顺序随机化且不编码正确性；预测后果、目标或相关记忆交换应改变选择，语言须与同一ActionIntent一致 |

执行后由环境产生真实Outcome；预测值、想象结果与真实结果分源。至少包含一次失败反馈和一次事实纠正，后继隔离query检验是否学会，不只重述被教的答案。T3从一步开始，短规划扩到两步须另有明确任务需求及固定出口。

环境/宿主可枚举合法动作、计量资源、执行及核验结果；不能给候选预排正确顺序、注入金标answer或代做选择。中文任务运行时的对象/关系抽取由Taiji负责，oracle/world_state标签只用于获准监督和独立评分。若工程测试使用结构化观察或oracle注入，单列为接口测试，不称自然中文理解。

## 4. 数据流与学习责任

| 必需内容 | 所有者及生产者 | 实际消费者 | 反馈/允许改变处 |
| --- | --- | --- | --- |
| 观察、对象/事件/属性绑定及来源 | Taiji可学习感知/TaskCore；原始证据有稳定来源span | 共同内容读取、记忆写入、世界预测、表达 | 合法监督/任务误差→对应表示与读取参数；格式字段不携带答案 |
| 当前/过去事实、有效时间及纠正 | 同一状态中的版本化绑定与经历 | query条件记忆选择、世界预测与内容选择 | 来源选择监督/后继任务误差→获准query/选择/写入参数；错误检索和错误生成分开 |
| 目标与动作参数 | 用户目标合法继承，Taiji生成/选择ActionIntent | 行动条件预测、条件价值/执行选择、中文表达 | 核验Outcome→世界/价值/执行选择及允许共享表示；不从RPC执行反传 |
| 预测后果及不确定性 | Taiji-owned预测头 | 候选比较/一步规划及解释 | 真实transition误差→后果头与相应表示，未知结果不补零 |
| 被选择的内容/意图 | 同一revision的选择结果和引用 | 语言/行动器官 | 回答/任务误差→该读取、表达与对应内容生产者；禁止从另一套答案缓冲区取值 |
| 真实结果及新能力 | 独立环境核验、获准反馈集 | 统一信用路由/学习器 | 各参数组和state/fast/slow更新分账；支持集收益须带隔离query与旧保持 |

必须交付每个任务的`producer → state field → consumer → output → loss/update receiver`矩阵。学习路径可按实际模块采用autograd、局部或离散信用，但需给可执行更新及冻结/置换对照；不能将grad=None一概判坏，也不能将任意非零梯度判为学会。

## 5. 状态、版本与生命周期

最小共享组件包含bundle/schema、user/session/task、state与parameter revision、对象/事件/关系绑定、来源span/证据ID、observed_at/valid_at、目标、被检索经历、pending intent、预测及反馈结算状态。连续向量与精确引用对应同一内容身份。

| 边界 | 必须保留/重建 | 必须清理/隔离 |
| --- | --- | --- |
| turn reset | 任务所需事实、目标、经历和纠正链 | 临时生成游标、UTF8/decoder临时态及失效缓存 |
| 新session | 仅获准user/task记忆及模型能力 | 会话私有临时态；不同user/scope不可串扰 |
| 参数更新 | 已有经历及来源；表示版本映射或原证据重编码 | 旧parameter revision的缓存不能继续伪装可用 |
| 进程恢复 | 全部声明状态、模型/优化器/局部迹、RNG、环境/课程游标、未结反馈与动作receipt | 不能重复执行已提交动作；unknown保持unknown |

同一checkpoint须包含全部实际消费者及更新者，不拼接各自最佳runner结果。零步保存→新进程恢复→下一步选择/输出一致→同批续训一致性先通过；之后才允许研究训练。V4已修复的Adam输入快照别名问题作为回归保留。

## 6. 实际文件与实施顺序

本节按真实代码触点维护：已实现文件和有限出口明确标为完成；`joint_learning.py`等尚未实现仍是后续合同，不伪称存在。接口与现有模块不兼容时再版本化，不为保留旧类名牺牲内容能力。

| 顺序 | 文件/触点 | 交付与有限出口 | 失败回流 |
| --- | --- | --- | --- |
| 177.0 本合同与责任盘点 | 本文件；`contracts.py`、`task_state*.py`、`conditional_language_v4.py`、`adapter.py` | 当前版本、继承/新建/退出、任务所需路径及信用表可交实现；此规划阶段结束 | 明确不可表达的差异并选择一种表示，不以“研究一下”挂起 |
| **177.1 共同状态和读写接口（有限完成）** | `taiji/joint_task_state.py` opt-in bridge；以既有`CognitiveState`为唯一根，映射V4 `TaskState`及`InputTrace`；未创建`joint_task_core.py`包装层 | 20组scope/goal/lifecycle矩阵＋10项来源、状态机、checkpoint专项回归，共30项；pytest、mypy、Ruff、Black通过。workspace视图不复制未验证内容。T1已消费Adapter InputTrace；默认Adapter生成路径未替换，语义收益未测 | T1/T2/T3的opt-in来源和条件消费者已接通；保持默认入口及能力声明边界，进入N177.3信用/恢复 |
| **177.2 必要记忆/预测/选择与语言装配（有限实现完成）** | T1：`joint_task_state.py`来源映射＋V4源帧过滤；T2：`EpisodicMemoryStore`经当前CognitiveState授权ID检索并按事件/assembly/percept tick唯一映射；T3：`joint_action.py`调用`WorldDynamicsLearner.predict`、`ExecutiveController.select`并写回ActionIntent/WorldPredictionRecord，V4读取同一决策投影 | T1的2scope×2goal×4干预16例通过；T2同query memory与query-only logits不同，歧义来源拒绝，V1→V2检查点迁移；T3显式Goal state_slots/relation_targets、目标翻转/预测翻转会改变选择，affordance重排不改变结果，动作投影改变V4 logits。53项联合模块/65项组合回归通过。T3目标与模型/控制器权重由结构化测试提供，不代表自然中文目标理解、训练或行动能力 | 未选目标state slot不得隐式按零比较；目标来源只证明来自当前输入账本、不证明语义解释；owner/scope身份由当前memory allowlist提供而非record自证；跨session来源缺失回保存/恢复；预测或执行未达回具体producer/consumer，不把输出敏感性称为学习 |
| **177.3A 身份、checkpoint与只读信用路由（有限工程完成）** | `joint_task_state.py` v3动作receipt/迁移；`joint_action.py`保存完整WDL/Executive快照与选中预测摘要；`joint_learning.py`只读credit route | v1/v2严格迁移、版本混装/重复组件拒绝；proposed/dispatch/awaiting/unknown/settled/cancelled恢复；Outcome/Transition/Intent/Prediction绑定；CognitiveState状态owner与参数接收者分列。68项模块专项与80项组合回归、独立进程unknown恢复通过。Outcome provenance仍是未认证caller assertion；decision可重建/摘要核验但未重放候选选择；`apply_updates=false` | 该出口只证明身份、恢复与资格路由，不证明世界模型、Executive或语言模块已学习；executor结果仍缺签名来源 |
| **177.3B 状态/参数责任及六类对照规格（静态合同完成）** | `joint_learning.py::JointCreditComparisonSpecV1`；无生产入口写入 | correct/permuted/no-update/readout-only/responsible-owner-frozen/state-only-change六种模式明确状态owner、可更新参数组、冻结参数组和语言目标摘要；所有模式要求同一NativeCheckpoint保存更新前后状态并新进程恢复，`apply_updates=false` | 该规格只可校验，不代表已执行更新或后验恢复；实际更新与六类行为对照随177.6授权训练验收 |
| **177.4A T1原生事实/当前值query生产（有限工程完成）** | opt-in `taiji/joint_input_producer.py`、producer checkpoint与source parse ledger，同joint native envelope；不改N176绑定adapter/runner源 | raw bytes＋逐bytePercept→model-owned span/act候选；entity/attribute/value有UTF-8 source span/percept IDs/hash/confidence；assertion/correction提交WorldObject/WorldEvent，query选择当前已知事实及query source；span pooled semantic feature进入V4 workspace | 91项组合回归含多turn Adapter续接、V4 semantic feature cut改变logits、新进程producer/parse ledger恢复。解析输出使用固定logit test double，监督loss只证明gradient到producer；未训练。query不检索episodic memory，Goal仅candidate |
| **177.4B T2情景检索与T3自然Goal输入（后继扩展）** | 同producer/retriever扩展到跨turn EpisodicMemoryStore query和自然Goal→target slots/action candidate；只用opt-in bundle | inference无provider/gold injection；查询source/时序正确，Goal由raw span映射target slots；歧义不创建ActionIntent | T1结果可判后按范围启用；此前T2/T3结构化fixture只算接口证据 |
| **177.5A 联合Trainer/manifest/零状态preflight（有限完成）** | `taiji/joint_t1_course.py`、`taiji/joint_t1_trace_cache.py`、`taiji/joint_t1_trainer.py`和`preflight_taiji_n1_plan177_5a_checkpoint.py`；[预检报告](../../../reports/taiji_n1_plan177_5a_preflight_20261027_v2.json) | 640 train episodes/2560顺序固定；train-only 590,512,321-byte trace cache逐源SHA绑定，F0/F1共13轮fresh Adapter重放逐字段一致；backward-only五模式梯度归属符合合同且0 optimizer updates；cache-bound zero NativeCheckpoint 19,550,523 bytes fresh-process参数/optimizer/cursor恢复一致 | 结构化接口可进入训练；不代表学到能力 |
| **177.5B 固定T1课程/资源合同（固定）** | [SPEC-N1-177.5 v2](SPEC-N1-177_5_T1_F0_F1_pilot_20261027.md) | 5 modes×3 seeds、2560 steps/arm、write-enabled/disabled配对诊断及family/seed评分已冻结；投影含30%余量2,269,268 CPU s/199,519 wall s/6.29GB | 新硬上限2.4M CPU s/220k wall s/8GiB；由用户AGENTS.md的可逆工作授权运行，旧N174账不继承；触限即保存并停止 |
| **177.6 T1同候选有限训练与验收（接口修复后运行）** | 同一producer+JointTaskState+V4 bundle及冻结train trace cache；独立NativeCheckpoint与只读scorer | update923暴露48→256适配遗漏语言workspace；两条路径统一无损注入，实际维度和模块7项回归通过。Correct/seed0 durable update2304、metrics update2383（17:40快照），已越过旧故障；失败未测尾段保守预留入账。answer-loss不向producer反传是本候选架构信用边界 | 继续固定15臂至各自2560终点或2.4M CPU/220k wall/8GiB硬限，再有限归因结案 |

实施按T1→T2→T3逐任务纵向贯通：对当前子任务落实共同状态、原生输入producer、必要消费者、信用/恢复、固定课程与有限训练。177.4A已接入T1事实/更正和当前值query；177.5A trainer/cache/preflight及177.5B课程/资源合同均有限完成；当前进入177.6一次固定T1训练与只读评价。T2情景检索与T3自然Goal producer在T1结果可判后扩展；不能把T3结构化target替代原生目标解析。

177.1–177.3可在N176未消费面保持不变的同时实施；不修改N176绑定的V3/V4、runner、SPEC、选择集或checkpoint，也不改变其共享传递依赖的运行语义。需要这类共享修改时，先有限收束N176或保留可独立复现的原运行图；这不是要求N176出现正收益。新候选有独立版本和数据/预算/结果；N176的现有固定评价仍是独立证据收束，不重训。

## 7. 基础课程、保持与后继位置

F0只形成短句/结束与输入输出合同；F1将对象属性及更正绑定到共同内容；F2按任务引入情景选择、一步预测/规划和反馈后继query。每一级在同一候选上做未见组合、相关翻转、无关保持和前级保持；不因独立语言未达到开放聊天水平而推迟T2/T3必要模块开发。

F3再增加自然表达、多步连续任务和真实跨会话经验；CC-5/6的最小历史/纠正/更新责任已在本包前置，完整持续适应和L4仍后续验收。状态写入收益、参数学习收益、检索收益分别报告；无更新、置换反馈与模块冻结必须在可比较父状态上执行。

更大多时间尺度/latent组块、收缩/恢复目标、PC/向量信用或SSM/块稀疏由实际长程区分、rollout或计算热点触发。快慢状态与reset责任本包先定义，完整算法研究不作为第一次联合链的预设前置；若必要表示/时间信用或算力瓶颈在前阶段已确认，可将对应研究切片提前，不必等F3；任何方案都须同时测任务收益、保持、成本及可恢复性。

## 8. 本阶段结束与唯一下一步

本次计划重排以03/09/10/07、本合同、N176承接说明及历史索引一致并提交为完成，不新增模型训练或封存消费。冻结实验结果/预算不改、历史失败可精确恢复，是重排的硬边界。

N177.1–N177.5B有限工程/恢复/资源合同已完成；整体Goal仍未结束。唯一队首是**177.6：继续固定T1联合训练并在终点做独立评分与架构归因**。Correct/seed0已越过update923故障并达到durable update2304，metrics到2383；旧账与失败未测尾段的保守预留均计入。parser输出detach，answer-loss不向producer反传，该架构信用边界需单列；scorer未读query/holdout，父checkpoint只读，硬限CPU≤2.4M秒、wall≤220k秒、制品≤8GiB。
