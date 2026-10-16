# 86 Codex 重启隔离与四条件三臂 fresh 训练（2026-10-16）

## 依据与目标

84在客户端更新重启时中断。最后可验证证据是seed11/position/zero；attempt没有运行进度，实际停止前更新数未知。84已由[中断裁决](EXP-N1-84_transition_factorial_paired_training_adjudication_20261016.md)封存为不完整，不恢复或覆盖它。

86解决同一训练主线的运行生命周期问题：将完全相同的固定实验放入独立隐藏进程，使Codex客户端退出或更新不再终止训练；将新报告、attempt、日志和检查点写入独立的86族。九模型一律从fresh零态开始，绝不继承84的未知状态。84既有模型协议、样本、目标、门槛、查询禁区和CPU上限原样沿用；86是新鲜执行尝试，不是84的续训。

## 固定实验范围

seed11/12/13 × position/terminal/cut，共九个模型，每个24576更新、55296 QA序列和43008训练会话；总计221184更新、497664 QA序列、387072训练会话。固定五个评分点、43200行、18次独立恢复及72副本步/288合成QA。使用84相同的六项position训练门与原始四条件目标；单线程、串行，整批CPU≤14400秒。完整保留控制臂和失败，不挑中点、不续训、不更改目标。

不访问dev、v3 holdout、v4或v5。85保持未准入，`TRAIN_SHA=None`。只有86完整、来源/代码/九分片/18次恢复全部通过，且三个position模型都通过六项训练门，才按原85冻结判据考虑首次v5查询；本计划本身不授权查询。

## 独立进程与文件完整性

先实现专用控制器和PowerShell启动器：控制器只导入冻结84运行器并为`REPORT`、attempt及checkpoint目录重定向到86唯一文件族；把控制器和启动器哈希加入运行时来源守卫。不得编辑84运行器、学习器、数据或优化器；控制器需核对84 protocol与代码身份逐项相同。启动器注册一个无触发器、仅当前交互登录用户可运行、有限权限的Task Scheduler任务，直接以隐藏`pythonw.exe`运行控制器；启动前写注册清单，任务启动后由控制器写worker清单，记录任务名、PID、UTC时间、完整参数、Python路径、控制器/运行器SHA和日志位置。运行器已有flush里程碑写入独立日志。进程自然结束后写UTC结束时间、退出码、累计CPU及报告SHA的completion回执；核验任务状态/LastTaskResult后移除精确的临时任务。启动或运行失败时保留所有产物，不自动重启。

训练前以合成心跳短任务验证：启动命令返回后父shell结束，后台子进程仍按时追加心跳并以预期退出码结束；它不载入候选模型、不加载训练材料、不更新权重、不查询课程/dev/holdout。实测日志与独立进程状态必须对应清单中的PID，之后才启动正式86。超过60秒的工程探针CPU即停止并裁决，不碰正式训练。

前两次`Start-Process`探针分别返回PID 12032和11556，但观察时未找到对应进程、心跳或报告，四个重定向日志均为空；它们作为启动器失败证据保留。第三次显式`CREATE_BREAKAWAY_FROM_JOB`探针收到Windows `WinError 5: Access denied`，因此该Codex执行宿主禁止子进程脱离其job。当前转为本机Task Scheduler无触发器交互任务；验证一项合成心跳并核对OS任务结果后立即移除精确探针任务。

补充：当前默认shell读取Task Scheduler服务/COM也返回`Access denied (0x80070005)`。用户已授予本地推进的持续权限；仅对唯一、无触发器、Interactive/Limited的合成探针任务请求一次策略审核的提权执行。若自动审核拒绝，则不再尝试系统级操作，不启动正式86，保留工作区内工程和失败证据。

本次受控注册创建了精确探针任务，但启动脚本把空`Triggers`误计为1并在调用`Start-ScheduledTask`前终止。独立查询显示任务`State=Ready`、`LastRunTime=1999-11-30`、`LastTaskResult=267011`，没有启动记录；未生成manifest/log/heartbeat/report。已移除该精确任务。启动器现把null触发器按0计数，并在启动前初始化回滚状态；用同一未消费的scheduled探针文件族完成一次有效短探针。

## 预检结论（㊵-1007）

合成Task Scheduler任务通过：worker PID 24408与OS运行记录一致，启动器返回后完成6/6心跳，CPU 0.0秒，报告`complete=true`；仅合成代码，未载入候选或训练材料，0更新/课程/dev/holdout查询。`LastTaskResult=0`已独立核实，精确探针任务已注销。正式86检查点路径另以fresh seed11/position zero执行保存和独立Python进程恢复：从83预检加载器`engineering.load`载入来源，77来源守卫通过，检查点269989字节、SHA256 `7cac570a0575c0edde56ebb7baf96b679edf19ff7989a6232ce826564ad70e4f`，恢复bundle相同、空Adam、0主模型更新，CPU含子进程7.46875秒；临时测试目录已清理。哈希回执在`reports/taiji_n1_transition_factorial_run86_checkpoint_preflight.json`。首次临时检查误将83来源传给仅接受84保存格式的`train.load`，在任何测试写入前被拒；对照真实训练路径修正为`engineering.load`后完整通过，冻结训练器未修改。回执helper`validate(receipt)`按传入报告对象和分片seal路径工作，不依赖84固定主报告路径，最终完整运行后调用。

㊵-1007当时计划在提交86实现、计划和预检证据后启动正式任务；随后㊵-1008记录了首轮启动阶段失败，未进入训练。正式批次的fresh九模型、221184更新、497664 QA、CPU≤14400秒、原六position训练门与零dev/v3 holdout/v4/v5边界保持冻结；85仍未准入，v5未消费。

## 首轮正式任务启动失败裁决（㊵-1008）

首轮Seed-N1-Run86-20261016的Task Scheduler任务已启动，但启动器写reports/taiji_n1_transition_factorial_run86_returned.json，控制器等待reports/taiji_n1_transition_factorial_run86_launcher_returned.json。60秒后worker在task_receipts阶段抛FileNotFoundError，OS LastTaskResult=1。失败发生在冻结attempt读取与readiness()来源/材料加载前；没有worker manifest、attempt、主报告或checkpoint，候选更新/dev/holdout查询均为0。该启动任务已精确注销；manifest、实际return、completion、错误日志及reports/taiji_n1_transition_factorial_run86_startup_failure.json保留，不覆盖。

修正版在launch_taiji_n1_transition_factorial_run86.ps1中令formal return文件名与控制器一致，并使用新输出族run86b、新任务Seed-N1-Run86B-20261016和独立checkpoint根data/checkpoints/run86b。再次正式运行前必须先完成静态检查、Task Scheduler独立心跳、run86b正式return文件名的receipt-only交接探针，以及run86b路径checkpoint保存/独立恢复预检；提交修正版及回执后才启动训练。心跳与交接探针均不得载入候选或训练材料，必须核对manifest/worker PID、任务LastTaskResult=0、启动器返回先于worker继续运行及更新/查询计数为0。训练协议与原六门、样本、更新数、预算、测面禁区均不变。

## 修正版预检结论（㊵-1009）

最终控制器/启动器源码静态检查通过，`tests/seed/test_project_identity.py` 4项通过。scheduled heartbeat复验：worker PID 15656与清单和全部6条心跳一致，6条均晚于启动器返回，`LastTaskResult=0`，CPU 0.015625秒，`model_loaded=false`、`training_material_loaded=false`，更新/课程/dev/holdout查询均为0。正式回执路径receipt-only交接：worker PID 29512一致、正式`*_launcher_returned.json`存在且SHA经核对、无触发器、`LastTaskResult=0`，CPU 0.015625秒；候选/训练材料未加载、更新和查询均为0。两项任务均已注销。

run86b正式检查点路径的保存/独立进程恢复预检完整：fresh seed11/position zero、77来源守卫、checkpoint 269989字节、SHA256 `7cac570a0575c0edde56ebb7baf96b679edf19ff7989a6232ce826564ad70e4f`、bundle独立恢复相同、空Adam/0主模型更新、CPU 11.46875秒，临时测试目录已清理；回执为`reports/taiji_n1_transition_factorial_run86b_checkpoint_preflight.json`。随后run86b正式启动在训练材料加载前被独占批次守卫拒绝。根因是控制器修改了`runpy.run_path()`返回映射，而runner函数仍读取原`__globals__`；因此实际看到84的attempt与检查点目录。修正版证据见[run86b启动失败裁决](EXP-N1-86B_runpy_namespace_startup_adjudication_20261016.md)。run86b失败产物全部保留，空检查点目录不清理。唯一下一步修复runner globals注入和checkpoint路由，以run86c全新文件族复做隔离/保存恢复预检，再提交并启动完全相同的冻结九fresh模型任务。

## 预声明产物

- 控制器：`scripts/training/run_taiji_n1_transition_factorial_paired_run86.py`
- 当前用户交互任务启动器：`scripts/training/launch_taiji_n1_transition_factorial_run86.ps1`
- 两次Start-Process探针（启动失败并保留）：`reports/taiji_n1_transition_factorial_run86_runtime_preflight_launcher.json`、`_runtime_preflight.stdout.log`、`_runtime_preflight.stderr.log`、`_runtime_preflight_retry_launcher.json`、`_runtime_preflight_retry.stdout.log`、`_runtime_preflight_retry.stderr.log`
- checkpoint路径保存/独立恢复预检回执：`reports/taiji_n1_transition_factorial_run86_checkpoint_preflight.json`
- 修正版Task Scheduler心跳预检：reports/taiji_n1_transition_factorial_run86_runtime_preflight_scheduled_retry.json、_scheduled_retry.jsonl、_scheduled_retry_launcher.json、_scheduled_retry_returned.json、_scheduled_retry_worker.json、_scheduled_retry.stdout.log、_scheduled_retry.stderr.log
- 最终修正版Task Scheduler心跳复验（在本计划预声明后执行）：reports/taiji_n1_transition_factorial_run86_runtime_preflight_scheduled_retry2.json、_scheduled_retry2.jsonl、_scheduled_retry2_launcher.json、_scheduled_retry2_returned.json、_scheduled_retry2_worker.json、_scheduled_retry2.stdout.log、_scheduled_retry2.stderr.log
- run86b正式return路径的receipt-only交接预检（仅验证启动器交接，不加载模型或训练材料）：reports/taiji_n1_transition_factorial_run86b_return_preflight.json、_launcher.json、_launcher_returned.json、_worker.json、.stdout.log、.stderr.log
- run86b checkpoint路径保存/独立恢复预检：reports/taiji_n1_transition_factorial_run86b_checkpoint_preflight.json（由冻结84训练器独立保存/恢复，0主模型更新；随后新增的receipt-only handoff分支未更改checkpoint路径或读写逻辑）
- run86b启动失败保留：reports/taiji_n1_transition_factorial_run86b_launcher.json、_launcher_returned.json、_worker.json、_completion.json、.stdout.log、.stderr.log、taiji_n1_transition_factorial_run86b_startup_failure.json；对应裁决：plans/reference/EXP-N1-86B_runpy_namespace_startup_adjudication_20261016.md
- breakaway失败记录留在命令输出；有限权限Task Scheduler探针：`reports/taiji_n1_transition_factorial_run86_runtime_preflight_scheduled.json`、`_runtime_preflight_scheduled.jsonl`、`_runtime_preflight_scheduled_launcher.json`、`_runtime_preflight_scheduled_returned.json`、`_runtime_preflight_scheduled_worker.json`、`_runtime_preflight_scheduled.stdout.log`、`_runtime_preflight_scheduled.stderr.log`
- 首轮正式启动失败产物（保留）：reports/taiji_n1_transition_factorial_run86_launcher.json、run86_returned.json、run86_completion.json、run86.stderr.log、run86_startup_failure.json
- 正式run86c attempt、主报告和九无损分片：reports/taiji_n1_transition_factorial_run86c_attempt.json、run86c.json、run86c_s{11,12,13}_{position,terminal,cut}.json.gz
- 正式run86c任务清单/日志/完成回执：reports/taiji_n1_transition_factorial_run86c_launcher.json、run86c_launcher_returned.json、run86c_worker.json、run86c.stdout.log、run86c.stderr.log、run86c_completion.json
- 正式run86c独立检查点：忽略目录data/checkpoints/run86c/n1_transition_factorial_paired_train_20261016/
- 结果裁决：`plans/reference/EXP-N1-86_detached_factorial_execution_adjudication_20261016.md`

探针使用独立任务名Seed-N1-Run86-Probe-20261016，修正版正式训练任务名Seed-N1-Run86B-20261016；二者均无触发器，由当前用户显式启动。

## 全项目边界

86只修复N1已验证的长任务受客户端生命周期影响这一工程缺口，不改变能力声明或其他主线队列。Task Scheduler注册只创建无触发器、有限权限、唯一命名的本地用户任务；探针验收后、正式任务报告核验后分别清理对应精确任务项。只有本地默认策略拒绝注册时才对合成探针提交一次自动策略审核；审核拒绝则停止，不启动正式86。C1载体、C2消费、C3信用、C4净增益、C5联合能力、保持、三器官出口与其他N系列/整族CI仍逐项独立核验。完整HEAD CI债未清；本地工程结果不能宣称远端CI通过。不得推远端、删档或更改既有门槛。
