# Phase 4 补充架构合同：R2 recovery v2 四端点配对评测

日期：2026-10-21。状态：架构审查通过；本补充只为W5评测实现定界，不阻塞Phase 8三臂训练，不授权现在读取dev、运行模型或生成。

## 结论

原 `04-architecture.md` 的配对逻辑保留为旧R2-07评测历史，但不适用于当前v2结果接口：原合同依赖旧06/07 result schema、既存06两端 generation，并只为两条07臂新增78次生成。客户端更新后采用的新恢复runner输出 `taiji-r2-dialogue-recovery-result-v1`，且训练过程明确 `generation_queries=0`；恢复计划亦记录旧06未保存39面generation。因此当前必须使用独立版本化v2评测合同，默认在相同39条prompt上为4个端点各生成一次，共156次新调用，不能把78条旧输出当作已存在。

## 模块边界与端点

- 基线：旧R2-03R1只读parent checkpoint `data/checkpoints/r2_daily_dialogue_oasst2_20261021_continuation/dialogue_context_final.pt`。绑定冻结文件SHA `41655eb8b5fcd7e446acb924e94e458ce9b1e966a96a95517198a5ac38df2086` 与payload `checkpoint_digest=1a77364a42633686dc9e1e9ee6a13a7e09b4f07770e3841d873bacd9eadaeaeb`；按原baseline配置构造推理trainer，不进行训练。
- 新端点一：`full_turn_1m` 的 `assistant_sft` 最终checkpoint。
- 新端点二：`warm_start` 的 `assistant_sft` 最终checkpoint。
- 新端点三：`fresh_same_capacity` 的 `assistant_sft` 最终checkpoint。
- 三个v2 checkpoint必须来自各自结果 `reports/taiji_r2_daily_dialogue_recovery_v2_20261021_{arm}_result.json` 的 `stages.assistant_sft.path`；中间 `pretraining_final.pt` 仅用于血缘及更新量核验，不作为对话端点。恢复根在 `data/checkpoints/r2_daily_dialogue_recovery_v2_20261021/{arm}/`。
- 当前固定面每个family只选一个prompt/history；四端点共享完全相同的39行输入，所以完整面为39 families、156 endpoint generations。旧06未完成合法39面generation；只有发现带完整source与checkpoint证明、且family/prompt SHA及端点checkpoint digest完全相同的已存输出时才可逐端点复用。任何身份无法证明的历史输出均令preflight拒绝并要求新的架构裁决，不得靠endpoint名称或相同prompt猜测等价。

## 元数据合同与拒绝条件

每臂result必须满足：schema=`taiji-r2-dialogue-recovery-result-v1`、`status=complete`、`complete=true`、arm与路径相符、`restart_from_zero=true`；`bindings.run`固定为 `r2_daily_dialogue_recovery_v2_20261021`，其arm、zero SHA、source hash map与同臂readiness逐字一致。每条绑定源路径均须存在且当前SHA匹配，结果文件SHA及stage checkpoint路径/字节SHA/restore digest也须逐项核验。不能只信attempt里的 `status=training` 或progress文本；v2 runner的最终result和两阶段receipt才是终点事实。

三个结果都须同时满足：

| 阶段 | 固定分母 | 固定字节更新 | final检查点 |
|---|---:|---:|---|
| `pretraining` | 30,096 episodes | 14,517,912 | 存在且独立恢复回执与文件SHA一致 |
| `assistant_sft` | 525 episodes | 113,209 | 存在且独立恢复回执与文件SHA一致 |

`result.dev_queries=0`、`generation_queries=0`、`can_promote=false`必须成立。新评测只接受三个result、三个readiness、基线checkpoint和所有source binding均齐全的状态；任一缺失、partial、计数/全局步不同、路径越出固定root、SHA/digest漂移、预算超限、已有同名评测attempt/result/progress或资源门失败，都须在第一次读取保留prompt前停止。不能修改训练runner或恢复合同来弥补评测输入不合格。

评测自身要有freeze manifest，绑定新架构补充、评测runner、契约测试、prompt选择/生成依赖及固定训练计划。其结果还应记录三个v2结果/readiness SHA、基线checkpoint SHA、四个终点checkpoint SHA、生成前后checkpoint文件SHA和模型摘要；任一生成期间变化即拒绝。

## Prompt、生成与持久化顺序

执行顺序固定为：只读metadata/source preflight → 核验三臂final receipts与基线 → 预算余量和CPU/内存资源门 → 创建并fsync write-once attempt及初始预算账本 → 按端点逐个fresh-process restore并逐笔记账 → **全部四端点恢复通过后**才用冻结的 `_select_dev_prompts()` 规则选39个family并写入规范化prompt face/SHA → 对每个未被精确证明已消费的endpoint逐条生成并持久化 → 汇总及人工审阅。该selector只将 `episode_id/family_id/split/system/context/history/user_input/unknown_policy` 映射为 `PromptOnlyEpisode`；评测结果不得包含reference answer、assistant target或exact-response评分。

### 执行账本的关键顺序门

- metadata/source preflight与CPU/资源门只读固定结果、readiness、源文件及checkpoint字节；不启动恢复子进程、不选择或读取prompt。门通过后才用独占写入创建write-once attempt及初始预算账本，记录输入绑定、固定CPU/wall上限、执行起点和已耗预检CPU；两个文件均flush/fsync后才能启动第一个恢复worker。
- 每个fresh-process restore前，先向追加式账本写入并fsync `restore_started(endpoint, checkpoint_sha256, cpu_before, boot_identity, timestamp)`；worker返回或失败后写入其restore receipt、child CPU、parent/host CPU差额与新的累计charge。restore CPU必须进入14,400秒上限，不能把预算从prompt selector时才开始计。
- attempt/账本若已存在、独占创建失败、source/resource/budget门失败、事件写盘/目录fsync失败，必须在启动任何restore或读取prompt前拒绝。恢复worker返回失败、预算耗尽、进程崩溃或强杀均消费该attempt；不得resume、重试、删除账本或用新的attempt名再次运行同一评测合同。
- 强杀发生在 `restore_started` 之后时，保留未闭合事件并把此次CPU尾账标为unknown；若能用同boot主机CPU计数核实就计实际保守delta，若不能则记录wall×logical-core条件上界与 `unknown_tail=true`。预算守卫须先为当前恢复及后续失败留出合同余量；无法证明剩余预算充足则不启动worker。禁止以“没有final result”把已运行的restore成本归零。
- prompt selector严格位于四个fresh-process恢复成功和完整receipt持久化之后。任何模型无法恢复、receipt/hash不匹配、账本出现unknown未闭合restore或累计CPU达到上限，均在读取39-family prompt之前停止；仅元数据 `--validate-only` 可重复运行，其不得加载模型、启动worker、选prompt或生成，且不构成正式执行attempt。

每条generation先持久化 `query_started(index, arm, prompt_sha256)` 并flush/fsync，再开始该次调用；生成结果返回后立即持久化输出并fsync。每端点最多39次；当前无历史合法输出分支下 `query_attempts=generation_outputs=persisted_generation_outputs=156`。发生异常、强杀、部分写盘或恢复失败即保留attempt和已落盘行，消费本次attempt并停止；任何情况下不自动重试、不跳过family、不从最后一条继续。使用与既有全轮脚本一致的 `_generate_readonly` 语义：`constrained_decode=true`、sampling关闭、`max_generation_bytes=512`，生成前后恢复/比对模型摘要且正式/优化器更新均为0。

建议新增入口 `scripts/training/evaluate_taiji_r2_daily_dialogue_recovery_v2.py` 和独立synthetic契约 `tests/taiji_native/test_r2_daily_dialogue_recovery_v2_evaluation_contract.py`；只写 `reports/taiji_r2_daily_dialogue_recovery_v2_20261021_paired_eval_*`。`--validate-only`只能审元数据与固定文件SHA，不选择prompt、不加载模型、不生成。真实dev读取只能由独立 `--execute` 在所有preflight和fresh-process restores完成后发生。

## 预算与结果

- v2恢复计划的评测增量上限是 **14,400 process-CPU秒**，覆盖source/face校验、全部四端点恢复、生成、落盘、失败调用与汇总；不得沿用旧04文档的7,200 CPU秒或旧“78新/78复用”假设，也不得把评测CPU二次计入训练三臂硬预算。该增量计入恢复计划总新增上限503,400秒。
- 保留既有单线程、启动前两次CPU采样均≤85%、可用内存≥8 GiB的资源门；任何一项不满足就在读取prompt前拒绝。旧架构的14,400秒wall上限继续作为更严格的墙钟停止门；先到任一限额都停止并留证。
- 输出schema必须逐项记录四endpoint identity与checkpoint SHA、输入face及逐prompt SHA、result/readiness hashes、复用与新调用计数、156组合输出、child/parent/total CPU、资源采样、错误和持久化计数；显式写 `parameter_updates=0`、`optimizer_updates=0`、`reference_answer_scoring_queries=0`、`exact_match_scoring_queries=0`、`can_promote=false`。
- 人工审阅固定同一39面前12个family，四端并列判断中文可读性、贴题、历史连贯和重复。表面格式/停止指标只作描述统计；本小样本对照不等于独立确认、自然语言能力晋级或N1-166准入。

## 风险与下游

当前v2训练结果尚未齐备，评测实现与synthetic测试可并行开发；不得读真实dev正文、运行模型或产生generation，直到所有三臂终点与恢复门完整。训练中不得修改任何source-bound plan、runner、数据、checkpoint或结果。测试使用合成39行prompt和虚拟result/receipt，必须证明缺失终点会在prompt selector前拒绝、四端点映射和156分母正确、失败查询不重试、source drift/重复family/错prompt SHA/磁盘不足/CPU与内存超限均fail closed，且checkpoint SHA不变。

## 回流判定

- 结论：通过；不需要上游返工，也不阻塞Phase 8训练。
- 回流目标：无。
- 给下游：按此v2接口实现独立评测runner和synthetic契约测试；先代码审查和定向验证，所有v2训练终点齐备后才可开启一次真实39面评测。

## P1-06 平台耐久裁决：Windows本地NTFS write-through发布

### 裁决

**接受同目录exclusive-temp → 写完并`FlushFileBuffers(temp_handle)` → `MoveFileExW(temp, final, MOVEFILE_WRITE_THROUGH)`作为Windows本地NTFS上的耐久目录项发布等价；Windows不再因缺少POSIX式目录`fsync`而整体unsupported。** 该裁决限定为同目录、同卷、目标不存在、成功返回的Win32发布。不要再尝试以`CreateFile(FILE_FLAG_BACKUP_SEMANTICS)`打开目录后调用`FlushFileBuffers`来补门：Microsoft文档把该标志定义为取得目录handle的方式，但列出的可接收directory handle函数不含`FlushFileBuffers`；后者文档描述的是有`GENERIC_WRITE`权限的打开文件handle。

依据是：`MoveFileExW`的`MOVEFILE_WRITE_THROUGH`合同是函数等待文件实际移到磁盘；同卷、同目录时不需要也不得启用copy/delete fallback。Windows NTFS的`FILE_FLAG_WRITE_THROUGH`说明也明确将rename列为需刷到磁盘的metadata变更。对本任务，temp文件先刷净内容，随后write-through rename成功返回后才允许启动restore worker，因此durable marker在昂贵执行开始前已发布。目录handle打开成功本身不算durability证据。

此保证表达Windows API/本地NTFS的成功返回语义，不声称超越OS、文件系统和设备flush契约；不能承诺故障硬件/控制器虚报flush、介质损坏或未验证文件系统上的物理持久性。

### 支持边界

- 当前支持：Windows上的本地固定卷NTFS；目标目录必须已存在，attempt、initial ledger和临时文件位于同一目录/卷。启动前核验目标路径所属卷为本地固定卷、文件系统名为`NTFS`，并核验temp与final解析到同一卷。
- 本次审查环境的只读资格证据：根任务通过`GetVolumeInformationW`核验`E:\Seed`卷的filesystem name为`NTFS`，`GetDriveTypeW`返回`3`（`DRIVE_FIXED`）。这只确认当前工作区满足本次本地NTFS裁决；实现必须对每次实际attempt所在报告根动态探测，不能硬编码当前盘符或探测结果。
- 当前拒绝：UNC/mapped/network/SMB路径、CSVFS/集群共享、可移动介质、ReFS、FAT/exFAT、第三方/加密/overlay/cloud-sync或无法识别的文件系统。`MoveFileExW`的支持列表表示API可用，不足以证明此应用的server、failover或过滤层具有相同durability语义；这些平台需单独架构裁决与故障验证后才能加入。
- 不创建尚未耐久确认的新父目录来放attempt/ledger；当前报告根`reports/`须预先存在。若未来要创建目录，须另为目录树建立已审查的发布协议，否则拒绝。

### Phase 6实现与测试验收

1. 通过Windows原生API（例如`ctypes`封装）在预先存在的目录使用独占创建随机temp；写完整payload后对文件handle调用`FlushFileBuffers`并检查返回值，再关闭handle。
2. 使用`MoveFileExW`只传`MOVEFILE_WRITE_THROUGH`；不得传`MOVEFILE_REPLACE_EXISTING`、`MOVEFILE_COPY_ALLOWED`或`MOVEFILE_DELAY_UNTIL_REBOOT`。目标已存在、跨卷、API返回失败或flags不符均fail closed。
3. attempt和initial ledger分别执行上述publish；两者都成功返回且最终文件SHA/内容核验通过之前，不得启动任何restore。若attempt已发布而ledger发布失败，则attempt已消费，严禁删除marker、换attempt名或重跑。
4. Windows集成测试在本地NTFS临时目录核验：调用flags/同目录同卷、目标不覆盖、成功后新进程可读取相同SHA；注入flush失败、rename失败、目录类型拒绝和ledger第二步失败时restore启动数均为0；模拟在write、flush后、rename前和rename返回后的进程终止，验证只有两个durable marker都发布后才准许worker。测试严禁访问dev/reference或加载模型。
5. CI/运行时必须在attempt之前执行文件系统资格探测；任何探测API失败、不是本地NTFS或无法证明同卷时，preflight返回明确`DURABILITY_UNSUPPORTED`并不创建attempt、不启动worker。Windows本地NTFS契约通过后，append现有ledger记录仍须`FlushFileBuffers`；若该刷新失败，停止并消费已创建attempt。
6. 故障注入验证API和进程崩溃边界，不冒充断电/硬件断电测试；文档只报告NTFS与WriteFile/FlushFileBuffers/MoveFileExW公开合同覆盖的范围。

### Microsoft依据

- [Directory Handles](https://learn.microsoft.com/en-us/windows/win32/fileio/obtaining-a-handle-to-a-directory)：`FILE_FLAG_BACKUP_SEMANTICS`可取得目录handle，列出可接受该handle的API。
- [FlushFileBuffers](https://learn.microsoft.com/en-us/windows/win32/api/fileapi/nf-fileapi-flushfilebuffers)：刷新打开文件handle；要求`GENERIC_WRITE`。
- [MoveFileExW](https://learn.microsoft.com/en-us/windows/win32/api/winbase/nf-winbase-movefileexw)：`MOVEFILE_WRITE_THROUGH`等待文件实际移到磁盘；不设置`MOVEFILE_REPLACE_EXISTING`时保持write-once拒绝覆盖。
- [CreateFileW](https://learn.microsoft.com/en-us/windows/win32/api/fileapi/nf-fileapi-createfilew)：`FILE_FLAG_WRITE_THROUGH`部分说明NTFS会刷写rename等metadata变化；文档也提示硬件不一定支持write-through。

### 回流判定

- 结论：通过；Windows本地NTFS采用MoveFileExW write-through，不需要未记录的目录handle flush，也不再整体fail closed。
- 条件：Phase 6必须落实上述路径/卷门、写入与MoveFileExW flags、marker发布顺序和故障注入；网络及未资格化文件系统仍拒绝。
- 不阻塞：Phase 8训练及已冻结source合同。

## Phase 4补充：P7生成调用的deadline与父进程取消边界

### 裁决

**通过架构补充，P1不接受父进程内无界同步调用。** 所有恢复及生成模型工作必须运行在父进程可观察、可取消、可强制终止的受监督worker进程边界内。父进程必须在生成前持久化`query_started`，用单次IPC请求授权恰好一个(endpoint, prompt)工作项；生成调用中的`_generate_readonly`即使阻塞，也不能阻塞预算监督器。超时或CPU预算将尽时，父进程终止worker（包含其子进程树），收齐已消耗CPU/wall账目，持久化该query的失败结果并消费attempt；不重试、不继续下一个query。

为避免按每个endpoint×prompt反复启动Python并重新加载模型损害吞吐，默认拓扑是一支父进程监督的长驻worker：在prompt selector之前于worker中完成四endpoint fresh restore并把各restore回执逐项交回父进程，父进程耐久写入全部四个restore成功事件后才选择prompt；worker保持已恢复模型驻留，之后仅按父进程逐条授权生成。工作进程没有自主selector、重试或批量预取prompt的权限。若单worker内存门不满足，不得退化成父进程同步生成或156次模型重载；该运行在prompt访问前fail closed，任何替代拓扑必须先由架构门审查且仍满足相同吞吐预算/边界。

worker不能通过公开CLI脱离attempt/ledger自行恢复或生成。父进程以私有IPC/一次性run授权绑定attempt id、source manifest、endpoint、query id与prompt SHA；worker仅接受当前已耐久记账的单个任务。父进程逐个发送任务，不得预发完整39×4 query队列。Windows上worker及任何后代必须属于同一可终止进程树；若平台无法证明worker结束后不存在仍在运行的后代，评测preflight拒绝。

### CPU/wall硬门

- 执行总CPU和总wall上限分别仍为14,400秒，worker启动/restore/模型驻留期间计算、所有生成、IPC、失败调用、终止、ledger写入和最终report均计入原预算，不新增长预算。父进程使用单调wall时钟并轮询worker进程CPU；CPU累计为父进程与worker树的CPU总和，不能只依赖worker成功回执。
- 每个restore/generation等待均有有限deadline。generation deadline取“冻结的单次上限”与“剩余全局wall预算扣除取消/终止及failure-finalization reserve后、按尚未开始的query数分摊额度”二者较小值；CPU门按相同方式从剩余CPU预算扣除保留额度并分摊。剩余额度、冻结上限或终止/失败落账reserve任一非有限、缺失或不足时，不得发送任务。由此单次调用不可能获得整段未封顶等待时间。
- 监督器按有界间隔采样；其采样延迟、软终止宽限、强制终止与wait确认的最坏耗时必须作为固定reserve计入14,400秒，而非事后估算。先到单次deadline、CPU分享额度或全局停止阈值时，父进程停止发新任务、发起取消并在有界宽限后强制终止worker树；随后记录实际可读的进程树CPU及wall尾账。此规则适用于checkpoint restore、模型/语料加载和生成，不得把任何模型工作留在父进程同步等待。
- 只有worker树已确认退出、父进程写入`query_timeout`/`restore_timeout`失败事件并fsync，且剩余finalization reserve仍足够时，才写出失败结果。若外部强杀父进程导致该失败行未能落盘，已有`query_started`会被判作状态unknown；attempt仍已消费，下一次执行必须拒绝。

### 必须具备的synthetic验证

1. 假generator永久阻塞：父进程在单次deadline内取消并强制结束worker，worker树无遗留；ledger含同一query的`query_started`与唯一`query_timeout`，attempt/result为失败，调用数和CPU/wall均计入。
2. 假worker执行CPU忙循环：父进程监测到本次分享CPU额度或全局停止阈值前停止新工作并终止worker；14,400 CPU/wall上限中保留取消和最终落账余量，不允许`budget.guard`等到同步函数返回后才执行。
3. 假worker忽略软取消或创建后代：强制终止覆盖整个进程树；测试确认无残留PID、无后续任务处理。无法终止整棵树时验证execute fail closed。
4. 在budget只剩finalization reserve、单次预算为零或期限配置缺失时，确认父进程不向worker发送新query；prompt selector仅在四个restore成功回执已fsync后调用。
5. 在query完成与deadline同时到达的竞争场景，状态机只允许一个终态（已持久化output或timeout），不得重复计数；timeout后的二次`--execute`必须在selector和worker启动前拒绝。
6. 测试断言restore/生成worker均由父进程创建、所有查询先有durable intent、timeout失败写入能在预留预算内完成，且1支长驻模型worker不会因逐query重载改变156分母或训练/评测source binding。

### 回流判定

- 结论：通过；所有restore/生成调用须在父进程可取消的有界worker中运行。
- 条件：以单个长驻worker保持四端点模型已加载以维持吞吐；一次只授权一个已落账query；deadline、轮询/终止/落账reserve须在冻结manifest中有限且受14,400秒总门约束。
- 给下游：Phase 6实现父监督IPC、进程树取消、CPU/wall实时记账和上述synthetic timeout测试；在Phase 7复审通过前，真实selector和生成继续禁止。
