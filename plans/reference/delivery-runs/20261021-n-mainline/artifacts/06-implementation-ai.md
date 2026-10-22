# Phase 6 实现：R2 recovery v2 独立配对评测器

## 结论

P7首轮审查打回后，ai-integration已完成六项P1与三项P2回流，并以synthetic夹具复核。当前评测器使用单个父进程监督的长驻worker，逐条授权四端点restore与156次generation；worker在Job Object关联、账本授权和单次预算确认前不会读取训练语料或checkpoint。`--validate-only`仍只读报告、来源与checkpoint字节元数据，不选prompt或加载模型。所有验证仅运行合成夹具和虚拟checkpoint元数据；未访问真实dev/reference、启动模型、执行`--execute`或生成结果。Phase 7独立复审与三臂训练终点审计仍是未来真实评测的前置门。

## 模型选型与调用链

- 全部使用本地 Taiji 原生模型，没有远端模型API或外部token成本。
- 固定端点顺序为只读parent基线、`full_turn_1m`、`warm_start`、`fresh_same_capacity`。三个候选只取各自 `assistant_sft_final.pt`；`pretraining_final.pt`仅参与更新量与血缘核验。
- `metadata_preflight()`逐schema核验三个readiness/result、zero与source绑定、两个阶段episode/byte denominator、final路径、文件SHA、独立restore digest、CPU预算和`can_promote=false`；未知字段或来源漂移均fail closed。
- readiness freeze manifest绑定当前v2架构补充、评测器、测试、prompt/generation实现、恢复runner、训练计划、模型/生成依赖及其来源SHA。模型payload只会在父进程监督、已验证Job Object成员身份并消费唯一durable任务授权后由worker读取；生成不在父进程同步执行。

## 提示词与工具设计

- `--validate-only`可重复校验并复用相同readiness；它不启动restore worker，也不调用prompt selector。
- `--execute`先确认readiness和所有输出路径，再执行CPU/wall、系统CPU、内存、磁盘及本地固定NTFS门。随后以独占临时文件、`FlushFileBuffers`、同目录`MoveFileExW(MOVEFILE_WRITE_THROUGH)`无覆盖发布attempt与初始账本；两者都成功才可创建worker。没有目录自动创建、rename fallback或非NTFS降级。
- worker通过`python -m scripts.training.evaluate_taiji_r2_daily_dialogue_recovery_v2`从仓库root启动；父进程建立命名Windows Job Object、设置`KILL_ON_JOB_CLOSE`并确认worker已关联，worker收到任务后还会按该Job名称验证成员身份。任务IPC每次只发送一个restore或query；缺attempt、无唯一开放intent、token不符/重放、claim已存在或Job身份不可证，都在读取语料/checkpoint前拒绝。
- 每个restore/query前先将带随机task id和一次性token SHA的`restore_started`/`query_started`耐久记账，再通过父进程监督的worker IPC授权。Windows Job Object累计整棵树CPU；父进程同时以单调时钟监控wall，超时用Job终止worker树并核验`ActiveProcesses=0`。无法确认成员、采样、终止或退出即fail closed；不以psutil单独证明整树终止。恢复/生成失败、强杀或落盘失败消费attempt且不重试、不续跑。
- 四端点restore回执全部耐久落账后，父进程重跑source、result/readiness与四个checkpoint绑定核验；随后先fsync `prompt_selection_started`，再调用冻结的`_select_dev_prompts()`并持久化39条规范化prompt-only face及SHA。selector开放但face未写成的重启状态会审计为unknown，不会再次选择prompt。
- generation worker没有selector、重试或预取能力；每条`query_started(index, endpoint_id, prompt_sha256)`耐久写入后才发送单条IPC。worker持有四端模型，使用`_generate_readonly`相同语义（constrained decode、sampling关闭、512 byte上限），输出返回后立即fsync。生成前后核对模型digest与checkpoint SHA；训练/优化器更新和reference/exact-match评分始终为0，`can_promote=false`。

## 评测方案与基线

当前实现包括固定39-family唯一性与face hash、四端点顺序/156分母、三组恢复result终点schema、各阶段分母、source/checkpoint漂移、路径越界、预算/资源门、restore前耐久记账、调用前记录、错误不重试、输入/输出SHA，以及人工审阅文件生成。人工审阅固定39面前12个family，对四端点并列记录中文可读性、贴题、历史连贯和重复；样例是本地脚本生成的格式草稿，尚未实际人工评审。

合成验证命令：

```powershell
python -m pytest -q tests/taiji_native/test_r2_daily_dialogue_recovery_v2_evaluation_contract.py --basetemp=.seed_test_tmp/r2_recovery_v2_eval_final
```

Phase 6首轮历史结果为12 passed（5.76秒）。Phase 7回流后的最终结果见下方“P7回流修复与复核”。下列两个失败重试测试当时已独立最小复测：

```powershell
python -m pytest -q tests/taiji_native/test_r2_daily_dialogue_recovery_v2_evaluation_contract.py -k failed_restore_consumes_attempt --basetemp=.seed_test_tmp/r2_eval_restore_retry
python -m pytest -q tests/taiji_native/test_r2_daily_dialogue_recovery_v2_evaluation_contract.py -k failed_generation_is_persisted --basetemp=.seed_test_tmp/r2_eval_generation_retry
```

每项均1 passed。`python -m ruff check`对两个新增文件通过；`python -m py_compile`通过；`python -m scripts.training.evaluate_taiji_r2_daily_dialogue_recovery_v2 --help`通过。没有跑全仓CI，以免给运行中的训练增加无关负载。

## 成本估算

- 真实评测严格受14,400 process-CPU秒与14,400 wall秒双预算约束，使用先到即停的守卫；独立训练预算不重复计入评测账本。
- 端点共4个、每端点39次、总调用156。每次restore预留120 CPU/墙钟秒，每次generation预留60秒，face准备60秒、最终化300秒；整轮初始保守reserve为`4×120 + 156×60 + 60 + 300 = 10,200` CPU秒和10,200 wall秒，另留5秒监督取消门。reserve不是实测成本，超出剩余门槛即不启动下一项工作。
- 合成测试只用虚拟结果与轻量伪生成器；全模块运行墙钟5.76秒，无真实模型恢复、训练、dev查询或远端API调用。

## P7回流修复与复核（2026-10-22）

P7首轮复审为P0=0、P1=6、P2=3并打回。此次实现已关闭全部列出的工程门，等待Phase 7独立复审；不构成真实评测授权。

- P1-01：移除公开`--restore-worker --endpoint`入口，改成隐藏的私有`--worker` stdin/stdout单任务IPC。worker要求固定attempt/ledger路径、唯一未闭合intent、匹配attempt/task/endpoint/token SHA和一次性durable claim；token跨task复用、重复claim、无父Job身份均在语料或checkpoint读取前拒绝。synthetic测试覆盖伪造token、claim重放和无Job direct bypass。
- P1-02：父进程从仓库root以`python -m scripts.training.evaluate_taiji_r2_daily_dialogue_recovery_v2`启动worker。模块入口拒绝缺attempt/ledger的私有启动；`--help`与子进程模块导入烟测通过。
- P1-03：四端点恢复receipt全部fsync后，selector前重新比对source manifest、result/readiness绑定和checkpoint SHA；漂移故障注入确认selector调用数为0。
- P1-04：单个父监督长驻worker持有四端模型，恢复与生成均只接受逐条授权任务，不运行selector、不预取、不重试。Windows命名Job Object设`KILL_ON_JOB_CLOSE`，父进程确认worker关联后才发任务，worker用Job名称再次核验成员身份。Job累计整个进程树CPU，父进程按有限wall deadline轮询；超时终止Job并等待`ActiveProcesses=0`。拒绝任何采样/终止/退出证明失败，绝不以psutil单独宣称整树终止。startup CPU也进入账本；恢复、生成永久挂起、CPU忙循环、Job关闭和后代进程均有synthetic测试。
- P1-05/-06：只支持动态探测为本地固定NTFS的预建报告目录。attempt与initial ledger分别使用同目录独占temp、`FlushFileBuffers`、只带`MOVEFILE_WRITE_THROUGH`的`MoveFileExW`无覆盖发布；任一资格、flush或rename失败均不启动worker。第二marker发布失败时保留已发布attempt并消费本次执行。测试包含实际本机NTFS发布与no-overwrite，以及ReFS/FAT/可移动/网络盘、flush、rename及第二marker故障注入。
- P2-01：result将CPU/wall明确标为finalization-start计数；write-through cost receipt另记preflight、父进程、Job进程树、逐端CPU、总CPU/wall、预算上限/余量、restore/query计数和accounting boundary。receipt明确排除其自身和result最终发布的写入成本，并由result的SHA字段绑定；所有promote与reference/exact-match字段为false/0。
- P2-02：整轮reserve统一为`4×120 + 156×60 + 60 + 300 = 10,200` CPU秒与10,200 wall秒；监督取消另保留5秒。余量是门控上界，不是实测成本。
- P2-03：selector执行前先耐久写入`prompt_selection_started`；face写入后再耐久记录prompt数与SHA。重启时未闭合restore、query或selector都会标成unknown尾账并拒绝续跑。

最终定向验证：

```powershell
python -m pytest -q tests/taiji_native/test_r2_daily_dialogue_recovery_v2_evaluation_contract.py --basetemp=.seed_test_tmp/r2_p7_final_full2
python -m ruff check scripts/training/evaluate_taiji_r2_daily_dialogue_recovery_v2.py tests/taiji_native/test_r2_daily_dialogue_recovery_v2_evaluation_contract.py
python -m py_compile scripts/training/evaluate_taiji_r2_daily_dialogue_recovery_v2.py tests/taiji_native/test_r2_daily_dialogue_recovery_v2_evaluation_contract.py
python -m scripts.training.evaluate_taiji_r2_daily_dialogue_recovery_v2 --help
git diff --check
```

结果为37 passed（8.49秒）、Ruff通过、py_compile通过、模块帮助通过，diff-check无错误且三份交付文件无尾随空格。另独立复核的Windows Job/NTFS故障集12 passed（3.32秒）；restore/query失败一次性重试、selector前漂移、attempt/ledger发布失败与真实本地NTFS写穿发布单项集4 passed（4.50秒）。未跑全仓CI。

调试记录：预算分享测试首次调用`_task_limits`时使用了`future_cpu_reserve_seconds`，而函数接口名是`future_cpu_reserve`，精确测试`test_generation_deadline_shares_remaining_global_budget_and_refuses_zero`以`TypeError: unexpected keyword argument`失败；只修正测试关键字后，同名隔离复测1 passed（2.57秒），随后模块复测通过。一次Windows Job smoke最初在0.15秒任务deadline内导入整个评测模块，未及创建child pid文件；根因为测试子进程冷启动耗时超过测试deadline。改为原生Win32 named-Job membership探针后，hang/close单项集2 passed（2.87秒）。这些测试修正没有放宽CPU/wall门或执行器。

根任务会话读取确认三臂训练仍在增长。本回流只改评测器、合成测试和本岗位产物，没有改训练runner、数据、checkpoints、恢复计划或board；未访问真实dev/reference、执行模型恢复、selector、generation或`--execute`。

## 降级兜底

缺失final、partial result、错误schema、来源SHA变化、checkpoint越界/哈希不符、CPU或内存不够、磁盘少于512 MiB、attempt/账本已存在、恢复receipt缺失、unknown tail或预算余量不足都会拒绝。可核实同boot host CPU时用其差值作为保守尾账；boot改变时记录wall×逻辑核数的条件上界和`unknown_tail=true`。attempt已创建后任何恢复或生成失败都不重试。人工只读审阅不使用reference评分或晋级阈值。

## 调试记录与根因

首次完整synthetic运行命令为上面的定向`pytest`命令，结果8 passed、2 failed（约7.5秒）：

执行顺序说明：这两项初始修正已在父任务要求我系统记录之前分别打入工作区，所以我不声称最初就是“每修一项立刻单测复测”。随后按父任务要求核对原始异常、追踪依赖流，并对两项分别进行隔离复测；精确输出和最终核验如下。

1. `test_resource_and_cpu_wall_budget_guards_fail_closed`，测试行286：`pytest.raises`要求消息匹配`8 GiB`，实际为`evaluation requires at least 8.0 GiB available memory`。根因是断言格式过窄，资源守卫行为正确。将测试匹配改为稳定的`GiB`标记；单项复测1 passed（2.94秒）。
2. `test_metadata_preflight_is_model_and_prompt_free`，调用点测试行345：`metadata_preflight()`读到真实固定路径`E:\Seed\reports\taiji_r2_daily_dialogue_recovery_v2_20261021_full_turn_1m_result.json`并报`FileNotFoundError`，随后由`read_json()`第95行包装为`Refuse`。根因是函数签名将`arm_paths=recovery.paths`作为定义时默认值，测试运行时monkeypatch被旧函数对象捕获，合成输入没有进入预检。改成`arm_paths=None`并在调用时绑定当前`recovery.paths`；隔离复测1 passed（3.01秒）。
3. 后续加入“attempt已消费”明确拒绝语义后，两个二次执行测试仍期待旧`write-once`字样：restore重试测试与generation重试测试各1次`AssertionError`，实际拒绝为`paired evaluation attempt already exists and is consumed; resume/retry is forbidden`。根因仍是测试期望落后于更严格的一次性语义；只更新两条断言，不放宽runner。逐项`-k failed_restore_consumes_attempt`及`-k failed_generation_is_persisted`复测各1 passed（分别3.01、3.04秒）。

另一次验证加入中断restore尾账审计后，第一次运行12项中有上述两条过期字符串断言失败、10项通过；修正后最终12项全部通过。修复过程未调用真实selector或checkpoint loader，也未读取dev/reference。

## 遗留风险

- W5a通过不代表Phase 8训练完成、评测通过或模型自然中文对话能力晋级。
- 三臂final result/readiness/checkpoint及末端fresh-process恢复尚未齐备；真实`--validate-only`、`--execute`、prompt face和generation均未运行/生成。
- Phase 7代码审查和训练终点审计必须先完成；随后才可执行唯一一次真实39面评测。N1-166既有冻结前置不因本评测改变。
- 全项目CI远端重跑仍待发布；本次仅验证新增模块定向Ruff、编译与synthetic测试。

## 给下游的注意事项

先由`code-reviewer`独立审查新runner/契约测试和账本中断语义。待三臂结果齐备后再单独运行`--validate-only`并人工核对readiness；其后才进入一次性`--execute`。本实现没有访问或保存任何真实39-family提示词/输出。

## 回流判定

- 结论：P7回流实现与synthetic复核已完成，待独立代码复审；在复审通过前不启动真实评测。
- 回流目标：Phase 7 `code-reviewer`。
- 返工原因：关闭首轮发现的六项P1、三项P2并新增对应failure injection；完整模块37/37通过，是否满足冻结架构由独立复审确认。

### Phase 6 追加：P1-07/P2-04 IPC与worker启动硬化（2026-10-22）

- 修改：IPC帧上限2 KiB；生成任务IPC只含持久prompt face的path/index/SHA，由worker按已发布face重读并验SHA。父进程写IPC使用独立线程，写入与响应均由deadline循环监督，超时会终止Job整树并在有限宽限内等待writer退出。
- 启动隔离：Windows worker以`CREATE_SUSPENDED`创建，Job Object分配及成员核验成功后才恢复；CPU启动值记入`worker_startup_cpu_seconds_charged_total`。分配/验证失败在任何IPC与selector前消费attempt并fail closed。
- 根因修复留痕：第一次完整模块复跑39 passed、1 failed。唯一失败是现有合同测试仍读取旧ledger键`worker_startup_cpu_seconds_before_job_assignment`；实现已将字段改为`worker_startup_cpu_seconds_charged_total`以表达启动CPU总记账，测试断言未随实现同步。更新断言后精确单测1 passed，再跑模块全套40 passed（9.05秒）。
- 验证：IPC阻塞、oversized frame、Job分配失败与先分配后resume定向测试4/4通过；Ruff、py_compile、`python -m scripts.training.evaluate_taiji_r2_daily_dialogue_recovery_v2 --help`及`git diff --check`通过。
- 边界：此前P7第二轮报告保留为不放行；复审预算原为2/2，本次硬化不构成审查通过。根据用户持续推进授权，提交补丁后安排新的独立代码复审。在新复审前不运行真实restore、dev/reference、selector或`--execute`；N1-166冻结状态不变。

### Phase 6 追加：writer取消与失败尾账（2026-10-22）

- 根因：第三轮审查发现writer超时仍存活时`failed_task()`会抛普通`Refuse`；外层失去`ipc_write_thread_stopped`，且generation/restore catch先清除`active_task`，因此最终事件可能遗漏writer状态与task id，并把`unknown_tail`误记为false。
- 红测：新增`test_unstopped_ipc_writer_state_is_preserved_in_failure_tail`后首次复跑失败，精确异常为`KeyError: 'ipc_writer_exit_unverified'`，证实终态字段缺失。
- 修复：专用writer使用Win32 `CancelSynchronousIo`请求取消当前同步I/O，随后有界join并检查真实线程状态；成功响应前也要求writer线程已退出。失败仍未退出时保留带状态的`WorkerTaskFailure`，不清空`active_task`；终态明确记录`ipc_writer_exit_unverified`、`unknown_tail`、task id/operation及取消错误，并消费attempt。
- 验证：原生Win32匿名管道阻塞读取取消集成测、deadline取消并join、writer未确认尾账测试均通过；定向4/4通过；完整模块43/43通过（8.97秒）；Ruff、py_compile、模块CLI `--help`、diff检查通过。微软文档说明该API向指定线程的待处理同步I/O发出取消请求，因此实现仍等待并核验线程终态，不把API返回当作退出证明。
- 门禁：第三轮复审P0=0/P1=1/P2=0仍是有效历史结论；本补丁尚未独立审查，Phase 7/W5a未通过，W5b及真实数据读取保持冻结。

### Phase 6 追加：task terminal缺失显式记账（2026-10-22）

- 根因：第四轮审查针对提交`1dabdc963`发现false→true时序：`failed_task()`第一次取消/join仍未确认writer，未写query terminal；outer cleanup第二次才确认writer退出后，会把writer状态刷新为true，但既有`query_started`仍未闭合，且旧`unknown_tail`不检查活动task。
- 红测：`test_writer_stopping_on_outer_cleanup_keeps_unwritten_query_terminal_unknown`先按旧实现运行，失败于`KeyError: task_terminal_missing`，证明终态遗漏。
- 修复：新增`active_task_terminal_written`，只有durable ledger restore/query terminal写入成功后才置true。异常收敛时若当前task仍无terminal，则写`task_terminal_missing=true`、`unknown_tail=true`，并保留task_id/operation；后续writer退出不能抹去未闭合task事实。
- 验证：false→true二次清理竞态测试1/1通过；模块完整合同44/44通过（9.14秒）；Ruff、Black、py_compile、模块CLI帮助与diff检查通过。未运行模型、restore、selector、`--execute`或真实dev/reference访问。
- 门禁：第四轮报告`07-evaluation-review-1dabdc963.md`对旧提交仍判P0=0/P1=1/P2=0；此处修改未提交、未复核，W5a继续不放行。

### Phase 6 追加：成功terminal状态同步（2026-10-22）

- 根因：第五轮复审针对提交`aed03128a`发现，`active_task_terminal_written`只在`failed_task()`写失败terminal后置true；正常`restore_completed`/`query_output_persisted`落账成功后仍false。若restore ledger成功但progress append失败，catch会再调用`failed_task()`写`restore_failed`，造成同一task两个terminal并误报missing。
- 红测：`test_durable_restore_terminal_is_not_duplicated_when_progress_append_fails`让restore terminal ledger append成功、后续progress append抛`OSError`；旧代码断言发现两个terminal事件（`restore_completed`与`restore_failed`）。
- 修复：两个正常terminal事件完成durable ledger append后立即置terminal状态；后续progress/清理故障保留已存在的唯一terminal，由外层写evaluation-stopped记录，不再伪造第二条query/restore失败terminal。
- 验证：专项单测1/1通过；评估器合同45/45通过（9.42秒）；Ruff、Black、py_compile、CLI `--help`与diff检查通过。未访问真实dev/reference或运行模型、restore、selector、`--execute`。
- 门禁：第五轮复审对旧提交报告P0=0/P1=0/P2=1；当前补丁尚未提交/复审，W5a/P7仍未放行。
