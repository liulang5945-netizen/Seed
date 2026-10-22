# Phase 2 补充：R2 三臂续训扩展 v1 计划

日期：2026-10-22（Asia/Shanghai）。状态：预算已获用户批准；续训合同与实现门仍未通过，当前不启动训练。

## 目标与边界

N 主线当前能力目标仍是自然中文多轮对话。此工作只把 R2 recovery v2 的三臂实验从各自已验证的 durable checkpoint 继续到原定训练终点，不代表对话能力已提升，也不解锁 W5b、dev/reference、selector 或 `--execute`。`can_promote=false` 与 N1-166 冻结保持不变。

三臂分别从自己的父 checkpoint 继续；不合并不同候选权重、不从零重训、不把已完成 episode 或 CPU 记账清零。

## 用户决策与预算语义

用户先批准1M/臂、3M总额，15:19将同一窗口修订为1.2M/臂、3.6M总额；17:50又批准当时累计上限为每臂 **1,300,000 process-CPU 秒**、三臂合计 **3,900,000 process-CPU 秒**；最新用户批准每臂 **1,450,000**、三臂 **4,200,000 process-CPU秒**，覆盖中间1.38M/4.05M审批。当前上限包含旧attempt已记录CPU、冻结规则未知尾账和本次续训消耗，不是新增额度；恢复或中断不能重置。超过当前上限需再次决策。

旧 attempt、progress、已有 receipt、checkpoint bundle 与原冻结 source hash 均保留原样。扩展使用新的版本化 attempt/progress/result 记录；旧 checkpoint 树保持只读，新 checkpoint 写入独立版本化目录并在新 receipt/manifest 中明确链接父 receipt SHA。新全局账本须用原子串行化方式同时检查各臂1.45M上限和4.2M聚合上限。

## 历史审批快照与当前只读账本

以下各表依次记录13:30原1M/3M窗口、15:20的1.2M/3.6M早期快照、17:12和17:50快照，以及18:02当前父checkpoint上界；均不是实际新增CPU。执行前仍必须重算，不能直接当作启动账值。boot identity已改变，冻结`tail_charge`使用检查点ledger已记账值加`elapsed wall × max(saved logical CPU, current logical CPU)`；本次取24核上界。

| 实验臂 | 父 attempt SHA256 | 最后 durable episodes / cursor | 最后 durable CPU 秒 | 父 latest receipt SHA256 | 保守尾账上界 | 1M 下剩余额度 |
|---|---|---:|---:|---|---:|---:|
| `full_turn_1m` | `c3557687753a0ae6b27baaa365175f73157e442db0ab3a6a4d725b8f056c9692` | 9,472 / epoch 2, index 1,948 | 34,465.78125 | `8ec115196c39f9936ea3704ac5aaa8934f8efeaf8794f58e709f285ceb3a23bb` | 780,628.774 | 219,371.226 |
| `warm_start` | `fff7ef55d72aef39778b8a781b13e7d18c1161253ceee21d7d3eb78aaf38f7e0` | 14,528 / epoch 3, index 3,242 | 34,039.859375 | `f8a2a72eec4d67ca362351158f4d11d19aee5bd044cec5b2ed7a399396b07583` | 780,081.702 | 219,918.298 |
| `fresh_same_capacity` | `db2b3e5f311fbcb35669da54107351033c7c527bea7e9e69a98fd49d744ed2d4` | 14,528 / epoch 3, index 3,242 | 34,066.109375 | `cdf91736fd6ea1c837ed075886c478b6df430f2494ecaf1163f81eefa6ad067e` | 779,196.622 | 220,803.378 |
| **聚合** |  |  | **102,571.750** |  | **2,339,907.098** | **660,092.902** |

`full_turn_1m` 曾记录 `recovery_stopped`，CPU 值为 748,972.986107；当前按父 checkpoint 重新计算的尾账更高，启动账本必须取二者及其他已记录费用的最大值。warm/fresh 仍无终点 result，三臂 attempt 都不能单凭 `status=training` 被视为仍在运行或完成。上一轮独立预算评审按训练吞吐估算剩余模型计算约 148,000 CPU 秒（full-turn 约 75,000、另两臂各约 36,500）；这是预测而非上界。冻结尾账规则每增加一小时会给每臂增加最多 86,400 CPU 秒，因此本快照不保证可以无限期等待后再启动。

15:20附近只读复核沿用同一父attempt、receipt与durable cursor；只更新按冻结尾账公式随时间增长的累计上界（历史）：

| 实验臂 | 保守累计上界 CPU 秒 | 新1.2M额度余额 CPU 秒 |
|---|---:|---:|
| `full_turn_1m` | 937,951.469 | 262,048.531 |
| `warm_start` | 937,416.609 | 262,583.391 |
| `fresh_same_capacity` | 936,544.407 | 263,455.593 |
| **聚合** | **2,811,912.485** | **788,087.515** |

这是动态保守上界，不是已发生训练CPU；准备、恢复与续训开始前必须重新采样。

### 17:12历史只读父checkpoint快照

独立只读审计已核验三个parent attempt、latest receipt/bundle、cursor和绑定来源；来源表分别26/29/29项，含每臂6个未跟踪的父绑定数据工件，SHA均与parent记录精确相符。

| arm | 当前保守累计上界 CPU 秒 | 到1.2M的当前余额 CPU 秒 |
|---|---:|---:|
| `full_turn_1m` | 1,098,577.619 | 101,422.381 |
| `warm_start` | 1,098,127.719 | 101,872.281 |
| `fresh_same_capacity` | 1,097,340.395 | 102,659.605 |
| **aggregate** | **3,294,045.733** | **305,954.267** |

该17:12快照当时尚未计入实际恢复、checkpoint写入与最终化开销；执行前必须刷新并证明吞吐投影、每臂和aggregate headroom均覆盖剩余工作。

### 17:50当前只读父checkpoint快照与预算修订

父attempt、latest receipt/bundle、cursor与source lineage只读审计通过；每臂6个未进入Git的父绑定data工件SHA均与parent记录精确匹配。用户批准将累计上限提高至每臂1.3M、aggregate3.9M：

| arm | 保守累计上界 CPU 秒 | 到1.3M的当前余额 CPU 秒 |
|---|---:|---:|
| `full_turn_1m` | 1,155,726.983 | 144,273.017 |
| `warm_start` | 1,155,263.650 | 144,736.350 |
| `fresh_same_capacity` | 1,154,476.359 | 145,523.641 |
| **aggregate** | **3,465,466.992** | **434,533.008** |

该上界随wall×24继续增加，不是新增训练CPU；执行前仍须重算。当前约148,000秒预计模型计算可在该headroom内覆盖，余量用于prepare/restore、保存及失败结算。实测速率投影仍为执行前门禁。

### 18:02当前checkpoint/预算快照

18:02历史窗口累计ceiling为1.3M/arm、3.9M aggregate。parent source、attempt、latest receipt/bundle与cursor只读审计再次通过；完整来源计数26/29/29，包含每臂6个未跟踪数据工件，均以parent SHA和parent lineage绑定。当前保守上界与余额为：

| arm | 当前保守累计上界 CPU 秒 | 到1.3M的当前余额 CPU 秒 |
|---|---:|---:|
| `full_turn_1m` | 1,170,919.559 | 129,080.441 |
| `warm_start` | 1,170,469.115 | 129,530.885 |
| `fresh_same_capacity` | 1,169,679.911 | 130,320.089 |
| **aggregate** | **3,511,068.585** | **388,931.415** |

当前续训runner/账本联合测试58项通过；Ruff、格式/Black、py_compile、CLI和diff检查通过。独立code review仍待完成；真实`prepare-only`与训练尚未启动。按冻结wall×24规则，执行前重新核算。

## 源码绑定与准入

旧 checkpoint 的 parent bindings 不得改写。当前源码对旧 attempt 绑定存在差异：full-turn 有 5 个不同源码文件；warm/fresh 各有 7 项。已检视的训练和数据构建 diff 主要为格式重排及未使用行号变量改名；这不能等同于 hash 相同，也不能替代逐文件独立兼容性审查。新 extension runner 必须单独绑定自己的源码 SHA，并记录每项旧 SHA、当前 SHA、差异性质和审查结论；任何未解释差异、父 bundle/receipt校验失败或新进程恢复失败都应 fail closed。不得用当前 SHA 回写旧 attempt，也不得以进度日志替代 checkpoint。

## 实施与验收顺序

1. 实现版本化 extension runner、manifest、全局 CPU 账本及版本化 checkpoint 分支。初始化时先耐久写入父 attempt SHA、父 checkpoint/receipt SHA、原累计 CPU、当前未知尾账和 boot/CPU/wall 快照，再恢复或训练；每次后续 checkpoint 将累计 charge 与新尾账取最大，严格单调。
2. 用合成状态验证父 receipt 分叉、各臂1.45M与聚合4.2M守卫、并发额度竞争、旧文件不变、首 checkpoint 前中断、checkpoint 事务半途失败、恢复后不重算已提交 episodes。测试不得访问真实 dev/reference 或启动正式训练。
3. 对 extension runner 做独立代码审查；逐臂确认原 source binding/兼容性证据、parent bundle 与新进程恢复、stale lock owner、启动 CPU/内存和剩余预算。真实训练仅在所有门均通过后开始；临启动重新计算尾账，并且只在资源与实测速率允许时并发三臂。
4. 从各臂独立父 checkpoint 与游标继续，累计到原定 30,096 预训练 episodes 和 525 assistant-SFT episodes；完成后 fresh-process 恢复并生成版本化终点证据，再由评测门审查是否允许 W5b。

### 完成判据

- 旧 attempt/progress/result/checkpoint 的 SHA 与内容保持不变，新 extension 记录能回溯父 lineage。
- CPU账本从旧成本与尾账连续累计；当前批准窗口每臂不超过1,450,000秒、三臂合计不超过4,200,000秒；预算不足时停止并保留账目，不自动扩容或重试。
- 三臂都达到冻结 episode/byte 分母，新终点 receipt、readiness 与 fresh-process restore 通过；没有未闭合写入或未知尾账。
- 训练、恢复与静态门禁不被写成语言能力结论；W5b 仍需单独评测和人工审阅，能力晋级门保持关闭。

## 唯一下一步

当前唯一下一步是冻结本轮runner/backend/tests与架构SHA并完成独立代码复审；复审无阻断后，重新核验预算/资源并运行本地NTFS `prepare-only`耐久门。只有`prepare-only`、父checkpoint恢复和开跑时CPU/内存/每臂及aggregate投影全部通过，才可并发续训三臂。W5b、真实dev/reference、selector、`--execute`和能力晋级继续冻结。

## 最新预算修订与续训准入快照（2026-10-22）

用户最新明确批准本窗口累计上限为每臂 **1,450,000 process-CPU秒**、三臂合计 **4,200,000 process-CPU秒**；该上限覆盖并取代此前1.38M/4.05M及所有更低窗口。旧实耗、冻结未知尾账、准备、恢复、训练、写盘、失败收敛均继续累计，不重置、不作为旧成本之外的新增额度。两个上限同时生效；聚合4.2M低于三臂各1.45M之和4.35M，因此任一单臂或总额先触即停。

19:19历史账本快照的aggregate保守累计上界为3,805,047.806345秒，剩余模型训练预测186,719.457141秒；该预测不是上界或实耗，执行前须用最新ledger重算。19:31:42刷新如下：

| arm | 保守累计上界 CPU 秒 | 到1.45M的账本 headroom CPU 秒 |
|---|---:|---:|
| `full_turn_1m` | 1,300,114.326747 | 149,885.673253 |
| `warm_start` | 1,299,649.842946 | 150,350.157054 |
| `fresh_same_capacity` | 1,298,841.390844 | 151,158.609156 |
| **aggregate** | **3,898,605.560537** | **301,394.439463** |

本次剩余训练预测仍为186,719.457141秒；19:31:42滚动投影显示每臂与aggregate可行（`feasible=true`），但不得替代开跑时复核，所有active reservation仍须计入预算守卫。CPU字段语义统一如下：result/readiness的累计CPU是各自发布时的下界；终点必须以已验证append-only ledger中唯一的`continuation_completed`事件作为完成凭据，并匹配`extension_id`、`arm`、`parent_attempt_sha256`及`details.result_sha256`/`details.readiness_sha256`。该事件的`charged_cpu_seconds`是更晚的累计下界，仍不含事件自身写入与进程退出成本；`reserved_cpu_seconds`只是预留上界，不是已消耗CPU，也不得与charged混称。

续训执行状态：前一次正式资源门仅采到7.50GiB可用内存，低于8GiB而拒绝，保留为历史失败；19:31:42最新`resource_snapshot`已通过，CPU双样本15.9%/18.5%、可用内存27.669GiB。该快照不等于训练已启动，当前仍未记录真实续训训练启动；开跑前必须再次刷新resource/budget snapshot。联合backend+runner合成测试72/72 passed（33.09秒），Ruff通过。架构评审最新无P1/P2阻断；这不取代执行前NTFS `prepare-only`、来源、父恢复与实时预算门。

## Lock-first writer与死锁归档恢复补充（2026-10-22）

runner必须先在同一arm根目录以固定本地NTFS write-through原子发布`writer.lock`，之后才可读取并调和可变progress/ledger、预留CPU或启动恢复/训练工作；锁文件含PID、process create time与token，路径为symlink、owner存活或身份不可核验时fail closed。`writer.lock`、`reclaim.guard`及archive记录都使用同目录temporary、flush/write-through发布，不允许以覆盖或删除历史锁替代恢复。

显式resume遇到dead writer时，先write-through发布`reclaim.guard`，核验owner身份（仅死reclaim.guard恢复由OS byte-range recovery lock串行化）；dead writer锁必须原字节耐久归档到`stale-locks/writer-lock-<token>.json`并确认源锁未变化后才能清理当前锁，再原子发布替代writer。若reclaim期间中断或替代writer发布失败，必须保留guard/archive证据；后续空resume只有在guard owner确认为dead、archive schema/SHA/owner身份相互吻合、没有活writer时才可恢复。任何锁发布、归档、证据核验失败都不得进入账本调和、恢复或训练。

已有组合回归覆盖“dead writer归档成功→replacement writer原子发布失败→guard清理→仅依赖archive evidence的空resume成功”；活owner拒绝不改ledger/progress。历史资源拒绝和最新资源通过均保留，不据此写成训练已启动。

最新19:31:42预算、资源与恢复合同见04架构末节；真实训练尚未启动。


## 19:44真实执行门失败与修复方向

NTFS prepare-only成功，三臂prepared；累计账本分别1,317,124.782691/1,316,636.232466/1,315,811.157573秒。准备后资源门CPU15.3/14.0%、RAM27.655GiB通过，aggregate余量250,427.827270秒，剩余训练预测186,719.457141秒。

三臂真实--run随后均exit1：base._validate_sources拒绝`full-turn dialogue source changed: scripts/data_prep/build_taiji_r2_daily_dialogue_full_turn_pretraining_corpus.py`。拒绝发生在root writer lock、start intent与权重更新之前；无真实续训进展。扩展source compatibility已通过，但旧base validator仍按原raw SHA检查已审查版本，合成测试未覆盖此实际衔接，保留本失败作为回归证据。

唯一下一步：实现严格来源证明下的extension专用训练数据校验，并建立显式版本化来源修订绑定；v1已prepared的合同、sidecar、父checkpoint和CPU账本不得改写或清零。新修订须红绿回归和独立审查，通过实际来源门后再续训。04架构报告已被v1绑定，保持原字节；新修订另记版本化补充。W5b、dev/reference、selector与晋级冻结。


## v1.1 来源修订实现进展（20:00）

采用独立架构复核推荐的append-only amendment：原v1合同、sidecar、progress、04报告及checkpoint副本保持原字节，同CPU账本连续累计；新增来源修订绑定原合同/attempt/readiness/receipt与新runner/validator/tests/补充报告。仅prepared、无执行事件、无pending reservation/写锁/terminal时允许发布。

extension专用validator保留12项来源与manifest/readiness/分母校验，仅builder允许精确审查兼容证明；新validator已对三臂真实只读载入与probe face通过，每臂3762预训练/525 assistant episodes。原72合同与新增9来源测试合计81通过（36.25s）；source-revision独立测试与Phase7复审进行中。没有发布amendment、没有恢复真实训练。

唯一下一步：修订合同回归及独立复审通过后，write-once发布v1.1 amendment，刷新同账本预算/资源门，再执行计费父恢复与续训；不重新prepare或重置旧费用。


## 20:38 真实开跑结果与预算决策

v1.1 amendment成功write-once发布；三臂`--run`都进入真实runtime，CPU/RAM资源门通过并写入start与`training_resource_preflight_passed`。随后统一被冻结的全任务吞吐门拒绝，原因`throughput_projection_exceeds_remaining_budget`。fresh restore次数0、durable training commit次数0，权重没有更新，reservation已全部settle。

20:38只读开跑尾账为full_turn_1m=1,328,948.048316、warm_start=1,328,462.169966、fresh_same_capacity=1,394,373.538816 CPU秒，合4,051,783.757098；剩余预算148,216.242902秒。滚动训练投影186,719.457141秒，超过现总上限28,503.214239秒。单臂最低ceiling投影为1,420,653.296396/1,375,970.695677/1,441,879.222166，fresh臂在1.45M下仅余8,120.778秒缓冲；预算保守尾账还随wall时长增长。

当前唯一决策点：如要在保留训练终点、15%吞吐安全系数和故障尾账的前提下继续三臂，需要用户将同一累计窗口修订至每臂1,500,000、aggregate4,300,000 process-CPU秒。按20:38快照，总投影后保留约61,496.786秒聚合缓冲，fresh臂保留约58,120.778秒臂内缓冲；所有旧charge和未知尾账继续计入、不重置。维持1.45M/4.2M则不应再次启动；将保留v1.1合同及全部失败/准备证据等待预算决策。


## 20:40 用户预算修订已批准

用户批准将同一累计续训窗口从每臂1.45M/aggregate4.2M提高至每臂1.5M/aggregate4.3M process-CPU秒。旧账、失败启动与保守未知尾账全部继续累计，不重置。唯一下一步是完成v1.2 append-only source/budget amendment并在4.3M窗口实时门通过后立即三臂并发续训；当前账本仍4,028,211.891916秒，未有checkpoint restore或training commit。

## v1.2真实恢复与boot identity缺陷（22:00附近）

用户批准的1.5M/臂、4.3M修订已成功发布，实际`--resume`已运行；三臂均无episode_completed或durable_training_commit。full/warm被吞吐投影拒绝，fresh在开跑前被累计臂上限拒绝。进程均返回exit1；两个遗留22秒reservation已按其保守上界全额结算并写入continuation_stopped，当前pending为空。最新durable charge为1,343,557.907691/1,455,709.397772/1,376,064.517384，总计4,175,331.822847秒。

已确认近期extension快照boot_identity有1792002779.5610702与1792002779.56107两种值，仅差约0.0000002秒。严格float相等把它们当成跨boot，触发wall×24未知尾账，使实际闲置等待抬高预算。parent boot值为1792002780.622641，与当前相差1.06157秒，不能把原parent未知尾账一并当作该微小抖动退款。

唯一下一步：修正extension boot身份的浮点稳定性，并对v1.2新窗口内已证实的误分类尾账形成独立、追加的审计校正方案。原历史event/contract保留；不得删除或清零原parent未知费用。已批准cap保持1.5M/4.3M，修复后先核验同boot host-CPU上界，再恢复训练。当前不能声称权重已有更新。


## extension尾账校正核算完成

已新增boot-accounting helper与4项定向回归。原prepared基线3,949,572.172730完整保留，读取当前全部host CPU得到新的可证明上界4,034,133.407105；现账4,175,331.822847。proposal已write-once保存在reports的boot_accounting_audit_v1，尚未应用credit。原parent boot差1.06秒，不纳入微秒级修正。下一步按04-continuation-boot-accounting-v1-3.md追加来源/会计修订与可复算校正event，然后在已批准预算内续训。


## 三臂真实训练与首个新增checkpoint已确认

boot审计校正已按v1.3追加到原账本，旧event及prepared基线完整保留。三臂实际恢复后均出现episode_completed与durable_training_commit。首个耐久点：full_turn_1m=9,504、warm_start=14,560、fresh_same_capacity=14,560；bundle SHA、receipt及独立恢复digest核对通过，previous_receipt仍链接各臂原parent。

存活exec句柄：full=22331、warm=80306、fresh=96103（本次已轮询确认仍运行）。下一步仅轮询这些现有句柄和新增durable progress，持续训练到30,096预训练+525 SFT终点；不重复启动。预算仍1.5M/臂、4.3M总额，W5b和晋级冻结。


### 训练运行状态修正

exec 80306（warm-start）已确认exit1，错误为剩余吞吐投影超过预算；最后耐久checkpoint为14,624 episodes（已新增96）。full-turn exec22331与fresh exec96103已再次轮询确认仍活跃，继续运行。warm无pending reservation；不据旧锁或最后progress推断其仍活跃，不重复启动另两臂。唯一下一步是在监测两条存活句柄的同时，核查warm本次投影拒绝及14,624 checkpoint恢复条件；预算仍1.5M/臂、4.3M。
