# Phase 7 独立复审：提交 1dabdc963

日期：2026-10-22。审查对象严格限定为提交 `1dabdc963` 的代码快照。复核重点为P1-07写线程取消/失败尾账、P2-04 suspended worker顺序，以及旧P1/P2是否回归。根任务已验证完整43项合同测试、Win32同步管道取消集成测试、Ruff/Black/py_compile和CLI；本轮不重跑测试、不读取真实dev/reference、不restore模型、不运行selector或`--execute`。当前worktree后续未提交修改不计入本报告结论。

## 结论

**Phase 7仍不放行。** P2-04已关闭，writer取消/退出核验以及task身份保留的主要修复有效；此前P1/P2关闭项未见回归。残留一项P1：如果writer第一次取消/有界join后仍存活，`failed_task()`会拒绝写该query的terminal事件；外层随后再次清理若恰好使writer退出，会把`ipc_writer_exit_unverified`刷新为false，但仍未补query终态，且`evaluation_stopped.unknown_tail`只取决于当前writer/worker/selector状态，没有检查未闭合task。于是durable `query_started`可以仍未闭合，却被当前失败事件报告成`unknown_tail=false`。恢复审计后来虽会识别未闭合query，当前attempt的尾账仍错误。

计数：P0=0，P1=1，P2=0。历史2/2预算和先前不放行结论保持原样；本次按用户持续推进授权审查新补丁，不代表预先通过。

## 本轮问题

|级别|位置|发现与影响|整改及复检要求|
|---|---|---|---|
|P1-07（残留）|`scripts/training/evaluate_taiji_r2_daily_dialogue_recovery_v2.py:2911-2926,3158-3171,3364-3377,3553-3634`；尾账审计`2430-2449`|`failed_task()`在发现树或writer仍未确认时再次`terminate_tree()`；若依旧未停，则保留`WorkerTaskFailure`并退出，尚未写`query_timeout/query_failed`（2911-2926）。内层restore/generate catch把`active_task=None`放在`failed_task()`返回之后，所以这次抛出时task id/operation保留（3158-3171、3364-3377）。最外层异常处理再次调用`terminate_tree()`，并用此时的worker属性刷新`ipc_writer_exit_verified`（3557-3572）。若writer到这次清理才停止，`evaluation_stopped`会记录writer已确认，但没有补query terminal；`unknown_tail`没有检查仍未terminal的`active_task`（3598-3603）。`_audit_unclosed_restore()`的terminal集合不包含`evaluation_stopped`，因此该`query_started`仍是开放task（2430-2449）。当前测试只覆盖writer持续未停状态（1758-1789），没有覆盖writer状态从false转为true的二次清理竞态。|保留query是否已写terminal的显式状态，并让失败终态在缺terminal时无条件标`task_terminal_missing=true`及`unknown_tail=true`，或在确认writer/worker退出后补写唯一的query terminal。新增false→true synthetic测试：第一次`failed_task()`确认失败，外层二次清理确认writer停止；断言attempt仍消费、query有唯一终态或明确unknown尾账，禁止错误地写`unknown_tail=false`，不继续派发。|

## 请求写入与writer退出证明

- `_cancel_synchronous_thread_io()`以writer的native thread id打开线程并调用Win32 `CancelSynchronousIo`；`ERROR_NOT_FOUND`只作为操作可能已完成处理（1679-1702）。
- `_cancel_and_join_ipc_writer()`调用取消后最多join一个`SUPERVISOR_TERMINATE_GRACE_SECONDS`，随后直接检查`Thread.is_alive()`；只有实际退出才清掉active writer引用，取消异常留在`_ipc_writer_cancel_error`（1952-1972）。Job正常终止、等待超时和异常清理分支调用该路径（2286-2327）。若post-termination CPU/Job/PID核验抛错，2321-2322会直接返回未确认；上层仍保守记录writer/worker未确认和unknown，不放行。
- request deadline超时将`worker_tree_stopped`及writer真实退出属性放入`WorkerTaskFailure`（2183-2193）；成功响应路径在解码/返回前join并验证writer，线程仍活则终止worker并返回失败（2221-2231）。
- 当writer持续存活时，`failed_task()`重新尝试终止；仍不确定则保留带状态的`WorkerTaskFailure`，并且内层catch在failed_task之后才清除active task。外层终态包含task id/operation、`ipc_writer_exit_unverified`及unknown标记（2911-2926、3158-3171、3553-3634）。这关闭了第三轮发现的“writer状态丢失、task身份清空”路径。
- 尚未关闭的仅是writer在最外层二次清理才变为已退出的瞬态：这时线程状态正确反映为已退出，但原query没有terminal事件；当前`unknown_tail`据此错误变为false。

## P2-04与旧门禁回归

|检查项|状态|证据|
|---|---|---|
|P2-04：启动CPU与Job赋值|关闭|`Popen`使用`CREATE_SUSPENDED`；先采样初始CPU，再分配并核验Job及记CPU基线，最后才resume并核验活动成员（1873-1904）。resume后的启动CPU进入Job采样/ready账本（1974-2067、3030-3048）。assignment/核验失败不resume并kill/wait，不能发IPC授权（1905-1927）。新增顺序与失败测试由根任务完整43项验证。|
|P1-01 worker直调绕attempt/预算|未见回归|固定attempt/ledger、Job成员、唯一intent、随机token和exclusive claim仍在（308-400、1285-1324）。|
|P1-02 子进程module启动|未见回归|`_worker_command()`仍以repo root的`python -m`启动（1705-1717）。|
|P1-03/P1-04 deadline/整树终止|未见回归（本报告P1-07尾账问题除外）|request监督deadline、整树终止并核验（2183-2327）；worker仍要求授权且一项任务一次IPC。|
|P1-05 selector前source/checkpoint重新绑定|未见回归|四端restore后重跑preflight并比较readiness、source、report及checkpoint SHA，然后写selector intent（3195-3222）。|
|P1-06 Windows NTFS write-through durability|未见回归|NTFS资格探测、FlushFileBuffers与MoveFileExW write-through发布路径不在本次硬化中被替换（188-280、2688-2750）。|
|P2-01/P2-02 final receipt及10,200 reserve|未见回归|显式finalization-start/cost receipt字段及固定reserve算式仍存在（1041-1053、3480-3538）。|
|P2-03 selector intent与中断尾账|未见回归|selector intent/face事件及unknown selector tail审计保持（2430-2524、3205-3222）。|
|2 KiB/face绑定、156分母、validate-only|未见回归|2 KiB校验和固定face query binding继续有效；156分母和validate-only只读metadata门保持（82、1216-1254、2130-2149、1013-1021、3379-3380）。|

## 回流判定

- 根任务验证：完整43项合同测试和原生Win32管道取消集成通过；Ruff、Black、py_compile、CLI检查通过。本轮未重跑这些验证。
- 代码审查结论：P0=0、P1=1、P2=0。P1-07仍未关闭；需要false→true writer竞态测试及明确的task terminal/unknown机制后再审。真实restore、selector、dev/reference读取和`--execute`继续冻结。
- 不修改历史2/2预算记录；本轮结论保持Phase 7不放行，Phase 8顶部主指针不变。