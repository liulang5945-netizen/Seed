# 增量需求

## 结论
R-01 验证06/07完整结果、源绑定、checkpoint恢复和更新量后才能访问dev prompt。
R-02 复用06已保存39条prompt及两端点输出，仅为07两臂各生成一次；不加载reference response。
R-03 输出逐条先落盘，write-once保护；结果记录全部查询、来源hash、恢复回执和CPU开销，can_promote=false。
R-04 人工样例固定同一12 families；格式统计不得冒充语义评价。

## 验收标准
AC-R-01 缺失或partial训练结果在任何prompt访问前拒绝。
AC-R-02 prompt身份不一致、重复family、hash漂移拒绝。
AC-R-03 模拟运行验证78次查询、156条组合输出及失败留证，测试不读取真实dev。
AC-R-04 无训练更新；原checkpoint文件hash保持不变；执行预算明确且fail closed。

## 遗留风险
所有目标依旧是局部研究证据，不自动晋升模型。

## 给下游
审查现有runner实际字段再定义接口，禁止猜测checkpoint和结果格式。

## R2 recovery v2 需求修订（2026-10-21）

本修订为客户端更新后从zero恢复的v2三臂新增独立评测需求。旧R-01..R-04及AC-R-01..AC-R-04保留为原R2-07历史；对于v2评测，以本节和 `artifacts/04-v2-evaluation-architecture.md` 为准，不把旧78次新生成/复用两端输出规则套用到v2。

### 用户故事与验收标准

|编号|需求 / 用户故事|优先级 / 依赖|对应验收标准|
|---|---|---|---|
|R2V2-01|作为研究负责人，我希望在读取任何保留prompt前核验只读parent基线及三臂v2最终结果、readiness、来源绑定、固定更新分母、checkpoint SHA与fresh-process恢复回执，以便只评测完整且来源可证的端点。|P0；三臂训练终点|AC-R2V2-01：给定任一result/readiness/checkpoint/绑定缺失、partial、字段不符、路径越出冻结root或SHA/digest漂移，当执行预检，则在prompt selector调用数为0时拒绝；全部核验通过时只接受parent baseline、`full_turn_1m`、`warm_start`、`fresh_same_capacity`四个指定端点。|
|R2V2-02|作为评测负责人，我希望评测面在四个端点间完全一致，并如实记录这次恢复v2的新调用，以便配对结果可追溯。|P0；R2V2-01|AC-R2V2-02：给定保留面尚未消费，当固定prompt selector执行，则产出39个唯一family的规范化输入face；每个端点在同一face上各生成39条，共156次新调用和156条持久化输出。旧06输出当前不存在；不得将其标作复用，不得通过输出数、attempt名或endpoint名称猜测已消费。若发现既有调用的证据，则在selector前拒绝并要求架构复审，不得静默减少本次156分母。|
|R2V2-03|作为评测负责人，我希望所有恢复成本在首个恢复worker前进入不可重试的耐久账本，并在恢复成功前屏蔽留出prompt，以便失败不能绕过总预算或误消费评测面。|P0；R2V2-01|AC-R2V2-03：给定元数据、source、预算和资源门通过，当执行真实评测入口，则先write-once创建attempt与初始CPU账本并flush/fsync，再启动第一个fresh-process恢复；每个restore前先fsync `restore_started`，结束后记录结果及parent/child CPU。四端点fresh-process恢复全部成功、receipt完整且无未闭合/unknown事件后，selector才可被调用。账本已存在、写入/目录fsync失败、预算不明或任何restore失败/强杀时，attempt被消费、selector调用数为0且禁止重跑或续跑。|
|R2V2-04|作为研究负责人，我希望每次生成都先记录调用意图、返回后立即持久化，并以单次合同封顶，以便强杀或I/O失败不会造成隐式重试或错误计数。|P0；R2V2-01、R2V2-03|AC-R2V2-04：给定四个端点和39条face，当开始一条generation，则先写入并fsync `query_started(index, endpoint, prompt_sha256)` 后才调用模型；返回后立即持久化输出并fsync。每个端点最多39次，成功条件为attempts=outputs=persisted=156。异常、强杀、部分写盘或返回后持久化失败均保留已知计数及partial证据，消费attempt并停止；不补跑、不跳family、不从末条续跑。|
|R2V2-05|作为预算负责人，我希望评测同时受process-CPU与wall上限约束，且资源或预算不足时在prompt访问前停止，以便成本不会隐性增长。|P0；R2V2-03|AC-R2V2-05：给定真实执行，当累计process-CPU达到14,400秒或wall达到14,400秒、任一资源门不满足、或剩余余量不足以完成下一恢复/调用并写出失败报告，则不得启动下一子进程或generation；报告记录parent/child/total CPU、wall、资源样本和停止原因，且不得把失败调用、restore或预检支出归零。|
|R2V2-06|作为研究负责人，我希望配对评测只观察模型生成、不训练、不使用答案评分，并保持晋级门关闭，以便此结果只支持有界的对话可读性判断。|P0；R2V2-02至05|AC-R2V2-06：给定完整评测，当结果生成，则`parameter_updates=0`、`optimizer_updates=0`、`reference_answer_scoring_queries=0`、`exact_match_scoring_queries=0`且`can_promote=false`；输入与输出记录不得包含reference answer或assistant target。|
|R2V2-07|作为评审者，我希望四个端点在固定样例上并列审阅，以便记录中文可读性、贴题、历史连贯及重复现象。|P1；R2V2-02|AC-R2V2-07：给定完整156条生成，当人工审阅启动，则固定使用同一39-family face中的前12个family，对四端点逐题并列记录可读性、贴题、历史连贯与重复；格式或停止标记只作描述性统计，不替代人工语义判断或独立确认。|

### 非目标、依赖与变更影响

- 非目标：本次不改训练/恢复runner、训练合同、checkpoint、产品默认模型或数据划分；不查询reference、不做exact-match评分、不自动晋级；N1-166继续因其既有前置门未满足而冻结。
- 依赖：真实评测必须等待三个v2训练result、readiness、assistant_sft终点checkpoint和独立fresh-process恢复全部齐备；评测实现、synthetic测试、代码审查可在训练期间并行。
- 变更原因：客户端更新后旧06/07端点未完成，旧06未保存合法39面generation；v2 runner/result schema与旧R2-07不同。影响：替代v2评测的端点、调用分母、预算账本顺序与验收；仅影响恢复v2评测，不改旧R2-07历史记录或训练合同。
- 遗留风险：小规模同面描述性生成不能单独证明自然对话能力、泛化或晋级；预算、恢复、face身份任一无法证明时应fail closed。

## 回流判定

- 结论：通过；现有v2计划和架构已把端点、分母、预算和恢复顺序定义到可验收程度。
- 回流目标：无。
- 返工原因：无。
- 给下游：按R2V2-01至R2V2-07实现与synthetic验证；真实prompt访问须等所有门通过。

## R2 recovery v2运行时预算澄清（2026-10-22）

本补充将 `artifacts/04-v2-evaluation-architecture.md` 中“父进程监督的长驻worker、逐条授权与有限期限/进程树终止”细化为可验收要求。它补充R2V2-01..07的运行时边界，保留39×4=156调用分母、14,400 process-CPU秒与14,400 wall秒双预算及此前其余门；不改变训练合同、N1-166冻结状态，也不形成新的能力主张。

### 新增运行时要求

|编号|需求|优先级 / 依赖|验收标准|
|---|---|---|---|
|R2V2-08|评测由一个新启动且受父进程监督的长驻worker承载四端点恢复与模型驻留；恢复全部完成前不得选择prompt。之后worker只能处理父进程逐条授权的单个query，不具备selector、批量预取、重试或自行取prompt能力。|P0；R2V2-01、R2V2-03|AC-R2V2-08：给定synthetic运行，当四个endpoint都在该worker中fresh-process恢复并把回执交给父进程，则父进程先逐项durable记录四个成功回执，之后才可调用selector。每条query须由父进程在`query_started`耐久落盘后单独IPC授权；worker未收到该授权不得生成，处理完单条后等待下一授权。测试证明worker无法自行调用selector、读取prompt源、预取下一题或重试。|
|R2V2-09|每个restore与generation均有有限deadline，且整棵受监督进程树的CPU/wall、取消、终止和失败落账成本计入原14,400秒双预算。父进程必须能在deadline或预算门触发时有界终止worker及所有后代。|P0；R2V2-03、R2V2-05|AC-R2V2-09：给定restore/generation任务，当准备启动worker时，必须存在有限的单次deadline、有限采样间隔、有限终止宽限、强制终止/wait确认路径及足够的failure-finalization reserve；父进程按全局剩余CPU/wall预算扣除这些reserve后逐任务授权。给定永久阻塞、CPU忙循环或创建后代的synthetic worker，当deadline/CPU门触发，则父进程停止派发并在有界时间内终止整棵树、记入实际进程树CPU/wall和超时失败事件、消费attempt且不重试。|
|R2V2-10|预算、deadline或整树取消能力有一项无法在prompt访问前证明时，评测必须fail closed。|P0；R2V2-08、R2V2-09|AC-R2V2-10：给定缺失/非有限的budget、deadline或终止证明，或平台无法确认worker树及其后代会被终止，当真实执行预检，则拒绝启动评测worker/selector，记录拒绝原因，`selector_calls=0`且`generation_calls=0`。任务中途超时、CPU/wall触限或整树终止失败时，保留已耐久attempt/query账本，消费attempt并禁止续跑或第二次执行；不得超过14,400秒任一预算门。|

### 边界

- 本修订仅澄清v2评测实现和合成验收。parent只读基线、三条v2 `assistant_sft` final、source/checkpoint核验、旧输出不可冒充复用、156分母、reference/exact scoring为0、training/optimizer updates为0、`can_promote=false`及固定12-family人工审阅全部保持原要求。
- 若单worker内存/资源门或进程树取消能力不成立，不允许退回父进程同步生成、逐query重载模型或任何未审查拓扑；应在selector前拒绝并保留证据。任何替代拓扑须先由架构岗审查，且重新证明同样的预算、deadline、耐久账本和156分母。
- 真实selector和生成仍等待Phase8三臂final结果及全部恢复门通过；此运行时澄清不授权读取真实prompt、运行模型或训练。

## R2 continuation extension v1 需求修订（2026-10-22）

本补充响应用户批准的三臂累计续训上限，并为各臂从其原 durable checkpoint 延续固定 R2 recovery v2 训练新增可验收要求。它不改变自然中文多轮对话的N主线方向，也不把训练完成视为能力提升。旧R2/R2V2需求与失败证据均保留；实现合同以 `artifacts/04-continuation-extension-architecture.md` 为准。

### 用户故事与验收标准

|编号|需求 / 用户故事|优先级 / 依赖|对应验收标准|
|---|---|---|---|
|R2EXT-01|作为实验负责人，我希望获得明确的版本化续训预算，以便能在不清除先前成本的情况下继续既有三臂实验。|P0；用户批准预算|AC-R2EXT-01：给定任一臂的已记账CPU与冻结规则尾账，当扩展runner启动或恢复时，则本臂累计上限为1,200,000 process-CPU秒、三臂合计上限为3,600,000秒；启动前重算当前尾账，任一上界触限或剩余额度不足以覆盖下一episode与失败收敛reserve时，必须在新训练更新前拒绝并耐久记录原因。额度不是旧账之外的新额度，任何中断都不能重置。|
|R2EXT-02|作为研究负责人，我希望每臂从自己的最后一个验证checkpoint及其游标续训，以便保留此前的有效权重和更新。|P0；父bundle与receipt校验通过|AC-R2EXT-02：给定父attempt、parent latest receipt/bundle、bindings与cursor，当恢复时，则fresh process核验receipt SHA、bundle SHA、checkpoint digest、arm/stage/corpus/source binding和cursor后，训练必须从该checkpoint下一条继续；不得从zero重建、合并不同arm权重或按日志推高cursor。|
|R2EXT-03|作为审计者，我希望新续训证据与旧证据分离且可回溯，以便能判断每次训练属于哪个预算窗口。|P0；R2EXT-02|AC-R2EXT-03：给定已存在的旧attempt/progress/result/checkpoint，当创建和执行extension时，则旧文件内容与SHA保持不变；新attempt/progress/result使用独立schema与sidecar路径；新checkpoint写入版本化目录，并通过parent attempt SHA与parent receipt SHA形成可验证谱系。旧result缺失时，不得把新result伪写回旧attempt。|
|R2EXT-04|作为预算负责人，我希望CPU实际值与未知尾账单调累计，以便宿主更新或强杀后仍可阻止超额训练。|P0；R2EXT-01|AC-R2EXT-04：给定父checkpoint ledger和任何后续extension ledger，当boot identity不变且主机CPU计数可用时按冻结规则计入其增量；当boot identity改变或计数倒退时，按`elapsed_wall × max(saved logical CPU, current logical CPU)`形成未知尾账。每次恢复的累计charge不得低于所有已知事件值、前次累计值和当前尾账上界的最大值；restore、准备、训练、子进程、保存、失败收敛及回滚重算均计入。|
|R2EXT-05|作为实验负责人，我希望来源差异被明确核验并绑定，以便checkpoint续接不会掩盖源码身份变化。|P0；R2EXT-02|AC-R2EXT-05：给定父attempt source hash与当前源码，当extension预检时，则逐文件记录parent SHA、实际运行SHA、对应Git blob/commit和兼容性审查报告SHA；extension runner/tests另记录自身SHA。不得修改父hash、将当前SHA冒充父SHA、忽略差异或静默overlay；任何无法证明与parent checkpoint兼容的文件都应拒绝该臂。|
|R2EXT-06|作为研究负责人，我希望初始预算意图和每个checkpoint事务在恢复前已可靠落盘，以便中断时不丢失成本或采用未验证权重。|P0；R2EXT-01..05|AC-R2EXT-06：给定extension已通过静态准入，当首次restore/训练更新启动前，则write-once manifest、parent lineage、accounting start与aggregate reservation必须在已验证本地文件系统上耐久发布；任何durability失败时restore/训练调用数为0。每个checkpoint需经fresh-process验证与receipt后方可成为latest；写入中断、receipt缺失、latest损坏或writer身份不明时fail closed且保留既有数据。|
|R2EXT-07|作为实验负责人，我希望单臂写入与三臂共享预算/资源受到同一守卫，以便在并发时仍能维持上限和速率。|P0；R2EXT-01、R2EXT-06|AC-R2EXT-07：给定三臂并行准入，当持久化aggregate ledger分配额度时，则必须用原子串行化检查每臂累计不超过1.2M且三臂之和不超过3.6M；同一臂最多一个writer。CPU双采样需均≤85%、可用内存≥8 GiB，并监测实际吞吐；若资源或速率不足以在剩余额度内完成下一有界工作单元，则停止派发并记账，不能静默降速后超预算。|
|R2EXT-08|作为N主线负责人，我希望训练结果通过全部来源、readiness和独立恢复门后再进行对话评测，以便能力结论保持证据约束。|P0；R2EXT-02..07|AC-R2EXT-08：给定extension三臂均达到冻结训练分母（每臂30,096 pretraining episodes / 14,517,912 byte updates，525 assistant-SFT episodes / 113,209 byte updates），且新result、readiness、source bindings及fresh-process恢复全部通过，则可交W5b评测门审查；任一缺失或unknown时真实prompt selector调用数必须为0。全过程`dev_queries=0`、`reference_queries=0`、`generation_queries=0`、`can_promote=false`，N1-166继续冻结；不得据工程或保存恢复结果宣称模型能力晋级。|

### 变更边界与待确认事项

- 变更原因：boot identity改变后冻结尾账超过原480,000秒预算；用户先批准累计每臂1,000,000、聚合3,000,000，随后于2026-10-22 15:19批准将同一版本化续训窗口上限修订为每臂1,200,000、聚合3,600,000 process-CPU秒。额度包含旧账，不重置，也不修改父checkpoint或来源身份。
- 非目标：不从zero重训、不覆盖旧attempt/progress/result/checkpoint，不读取dev/reference，不运行selector或`--execute`，不改训练目标、分母、数据顺序、超参与N1-166晋级门。
- 假设：当前父latest bundle/receipt与原训练数据仍完整；所有source compatibility与fresh-process parent恢复均须在执行前重新验证。审查快照随墙钟增长而过期。
- 遗留风险：1.2M/臂仅适用于当前续训窗口；若尾账在执行前或中断后吃尽余量，必须停止并另行讨论，不得隐式增加预算。

## 回流判定

- 结论：通过；R2EXT-01至R2EXT-08可逐项验收，且与已审查extension架构合同一致。
- 回流目标：无。
- 返工原因：无；来源/账本/恢复差异按下游实现门处理。
- 给下游：按 `artifacts/04-continuation-extension-architecture.md` 实现并测试；独立代码审查与三臂执行前门全部通过后才可训练。
