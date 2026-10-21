# Phase 7 第二轮独立复审：R2 recovery v2 paired evaluator

日期：2026-10-22。审查对象为首轮打回后返工的v2评测器、synthetic测试与最新Phase 2/3/4/6文件。没有运行真实模型、worker、selector、reference/dev读取或`--execute`。根任务报告新增synthetic模块37/37通过及Ruff/py_compile通过；本审查引用该独立验证，不重复增加CPU负载。

## 结论

**仍不放行Phase 7。** 首轮六项P1、三项P2中的五项P1及全部P2已关闭；剩余一项P1：IPC管道写入位于deadline监督循环之外，可能在worker没有读取时阻塞父进程并越过14,400秒墙钟门。另有一项P2启动窗口问题：Job assignment前的CPU只采样一次，且失败时只能kill根worker并将整树标记为未确认。

当前计数：P0=0，P1=1，P2=1。修复剩余P1并对oversized/no-reader管道故障注入复测后，再由Phase 7复审；真实39面仍冻结。

## 首轮问题逐项关闭情况

|首轮问题|复审状态|依据|
|---|---|---|
|P1-01：公开restore worker绕过attempt/预算|关闭|`--worker`要求固定attempt/ledger路径与Job身份；每个task还需唯一durable intent、attempt id、endpoint、一次性token hash和exclusive claim。无Job、无授权或重放在语料/checkpoint读取前拒绝。synthetic测试覆盖直接/伪造/重放请求。|
|P1-02：子进程脚本路径启动缺repo import root|关闭|`_worker_command()`以repo root为cwd并用`python -m scripts.training.evaluate_taiji_r2_daily_dialogue_recovery_v2 --worker`启动；模块导入烟测在返工验证中通过。|
|P1-03：restore无deadline/整树终止|关闭|每项restore有有限CPU/wall share；父进程轮询Job Object累计树CPU与wall，deadline后终止Job并确认`ActiveProcesses=0`；无法确认即fail closed。测试覆盖hang、忙循环和后代。|
|P1-04：generation无deadline|部分关闭，仍有本报告P1|生成已移入受监督长驻worker，有有限CPU/wall deadline及Job tree终止；但父进程同步写完整IPC行发生在deadline poll loop之前，见P1-07。|
|P1-05：selector前未复核sources|关闭|四端恢复成功后，在selector前重新比较readiness payload、source manifest、result/readiness hashes；source drift测试确认selector为0。最终生成后还会再次核验。|
|P1-06：attempt/ledger目录项durability不足|关闭（限定平台）|仅支持动态探测到的本地固定NTFS与预建目录；exclusive temp写完后`FlushFileBuffers`，再同目录`MoveFileExW(MOVEFILE_WRITE_THROUGH)`无覆盖发布。attempt与initial ledger任一发布失败均不启动worker；不支持路径fail closed，真实NTFS集成测试已覆盖。|
|P2-01：final CPU/wall total低报最终写盘|关闭为显式边界|结果字段改为`*_at_finalization_start`；单独write-through cost receipt标明pre-publication测量、publication excluded及剩余预算/finalization reserve，不再冒充最终实测total。reserve仍纳入执行门。|
|P2-02：reserve算术错误|关闭|实现与岗位产物均为`4×120+156×60+60+300=10,200`；新增直接算术测试。|
|P2-03：selector阶段意图/unknown尾账缺失|关闭|selector前先durable append `prompt_selection_started`；face发布后durable append `prompt_face_persisted`。重启扫描把未闭合selector标为unknown，消费attempt且拒绝续跑。synthetic测试覆盖。|

## 本轮问题清单

|级别|位置|问题与影响|修改方向|
|---|---|---|---|
|P1-07|`scripts/training/evaluate_taiji_r2_daily_dialogue_recovery_v2.py:1696-1706,1716-1742`；payload构造`2414-2428`|`SupervisedWorker.request()`在deadline采样循环启动前同步执行`stdin.write()`和`flush()`。generation IPC携带完整`prompt_row`及history，当前没有序列化行大小上限；worker尚未进入read loop或异常停止读取时，管道写满会阻塞父进程，task CPU/wall deadline、Job终止及failure-finalization reserve都无法运行。worker没有ready握手，且输出最多512字节的限制不约束输入prompt行大小。|把IPC发送本身纳入可取消deadline：使用有界/可取消写入机制，或在父进程监督线程中发送并确保超时能关闭写端、终止整树；可增加worker-ready握手但不能只靠握手。仍只发送单个query、不预取。增加stalled/no-reader worker与超过pipe buffer的synthetic prompt测试，断言deadline内树退出、unknown/query_timeout落账、attempt消费且不越过wall门。|
|P2-04|`scripts/training/evaluate_taiji_r2_daily_dialogue_recovery_v2.py:95-98,1586-1616,1617-1632`|startup CPU从Popen后`psutil.Process.cpu_times()`单次取样至Job assignment前记入`initial_child_cpu_seconds`，并在2298计费；因此导入/启动截至该采样时的CPU已覆盖，但采样与1612-1616 Job assignment之间仍有进程可执行的短窗口，Job统计未涵盖该窗口。若`job_factory`在Job对象可用前失败，代码只kill并wait根worker；若Job已建立，则尝试`terminate_and_confirm()`后关闭，但异常被吞并且最终仍报告`worker_tree_stopped=False`，因此assignment故障路径无法证明整树确已退出。外层消费attempt、记`worker_tree_exit_unverified`且不进入IPC/selector；当前assignment故障测试使用无后代FakeProcess。|优先以suspended process创建worker，加入Job并验证成功后再resume，使导入CPU都在Job内；若保留当前拓扑，应补一个真实本地synthetic startup/assignment故障测试，确认assignment前没有可执行任务、所有CPU窗口有保守账本且无残留PID。未完成前报告保持unknown-tail并fail closed，不得默认root.kill等于整树终止。|

## 合并门槛核对

- P0：0。
- P1：1，`P1-07`未修复，不能进入Phase 8真实评测。
- P2：1，启动CPU采样/assignment失败树退出仍有精度与验证缺口；当前代码对此fail closed，但应在下次补丁中关闭。
- P2-01已通过显式pre-publication cost receipt与保留finalization预算关闭口径问题；报告准确说明finalization reserve而不把开始时读数标成完整的实测CPU/wall。
- 根任务独立验证记录为37/37 synthetic tests通过、定向Ruff/py_compile通过；此次复审不访问真实评测输入或模型。
- 06/07训练仍可继续；Phase 8主指针保持不变；W5b不能启动。

## 给下游的注意事项

先关闭P1-07，并补stalled/no-reader/oversized IPC故障注入；同时建议处理P2-04的suspended startup及树失败证据。复审通过前，不调用selector、不运行restore或generation，也不访问真实dev/reference。

## 回流判定

- 结论：打回；尚不满足Phase 7合入门。
- 回流目标：`ai-integration`。
- 返工原因：父进程可能卡在同步IPC write/flush，deadline和整树取消无法执行，违反AC-R2V2-09/10及14,400秒双预算。
- 复检范围：deadline覆盖IPC发送、stalled/no-reader任务在预算内树终止、unknown-tail/attempt消费、无重试；确认P2-04 startup_cpu和assignment故障仍无未计CPU或未确认残树后再复审。
