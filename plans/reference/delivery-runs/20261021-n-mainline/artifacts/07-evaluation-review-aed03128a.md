# Phase 7 独立复审：提交 aed03128a

日期：2026-10-22。复审严格基于提交 `aed03128a` 的代码快照。重点核对`task_terminal_missing`是否依赖durable terminal、上一轮false→true writer竞态、terminal已durable时是否误报，以及P2-04和旧P1/P2是否回归。根任务已验证完整44项合同测试、Ruff/Black/py_compile/CLI和提交门禁；本轮不重跑测试，不读取真实dev/reference，不restore模型，不运行selector或`--execute`。未提交的后续worktree修改不计入本结论。

## 结论

**Phase 7仍不放行。** 上一轮P1-07的false→true writer竞态已正确关闭：是否缺失task terminal会在外层异常收敛时反映到`unknown_tail`，并且`active_task`直到durable terminal成功写入才清理。P2-04及此前P1/P2均未见回归。

当前发现一项P2：`active_task_terminal_written`只在`failed_task()`将失败terminal耐久追加到ledger后置true；正常`restore_completed`和`query_output_persisted`的ledger terminal成功追加后没有同步置true。若成功restore的ledger terminal已刷新，而紧接着progress追加失败，外层仍会把已有durable terminal误报为`task_terminal_missing=true`和`unknown_tail=true`。这不会继续派发任务，但导致一次性失败证据与ledger事实不一致；计划门槛要求P2全部关闭。

计数：P0=0，P1=0，P2=1。按当前门槛仍不放行Phase 7/W5a；应在每个正常terminal ledger append成功后同步更新task状态，并补充“terminal已durable、后续progress写入失败”的故障注入测试。

## 本轮问题

|级别|位置|发现与影响|整改及复检要求|
|---|---|---|---|
|P2-05|`scripts/training/evaluate_taiji_r2_daily_dialogue_recovery_v2.py:2851,2967-2969,3113-3114,3136-3160,3287-3288,3342-3367,3558-3609`|`active_task_terminal_written`在新task开始时重置；失败路径仅在`failed_task()`把terminal事件成功durable追加到ledger后设true（2967-2969）。正常restore同样将`restore_completed`耐久写入ledger（3159），但未设true；随后`progress_path`追加可以失败（3160），catch保留active task，outer便因局部flag仍false报告`task_terminal_missing=true`，虽然ledger已有terminal。generation的`query_output_persisted`也在3342-3366写ledger，但没有同步更新该flag；随后persist计数/清理task前若发生异常，同样误报。现有false→true测试只覆盖尚未写task terminal的query（test 1801-1834），没有覆盖terminal已耐久写入的反例。|每个正常成功terminal事件写入ledger并fsync成功后立即置`active_task_terminal_written=true`；将状态点绑定到ledger terminal确认，不要等progress追加。补synthetic测试：让`restore_completed` ledger append成功、progress append失败，断言`evaluation_stopped.task_terminal_missing=false`、terminal事件仅一条且attempt仍按失败消费；再覆盖`query_output_persisted`成功terminal后异常收敛的路径。|

## 重点核查结果

- false→true writer竞态已关闭：task开始后初始terminal标志为false；writer第一次取消/join仍未确认时，`failed_task()`不会写terminal并保留带状态的异常；外层第二次清理即使使writer退出，`active_task_terminal_written`仍为false，故`task_terminal_missing`和`unknown_tail`保持true，task id/operation仍从`active_task`读取。测试`test_writer_stopping_on_outer_cleanup_keeps_unwritten_query_terminal_unknown`已覆盖此路径（测试1801-1834）。
- 失败terminal只有在ledger durable append成功后才将标志设true（2967-2969），这一步满足“ledger是terminal事实源”。缺口在正常成功terminal路径漏设同一状态。
- P2-04关闭：worker以`CREATE_SUSPENDED`启动，采样CPU、建立并核验Job及CPU基线之后才resume（1873-1904）；resume后的bootstrap预算受Job采样覆盖；assignment/核验失败不resume且不发IPC授权（1905-1927）。

## 旧问题回归核对

|检查项|状态|依据|
|---|---|---|
|P1-01 私有worker授权|未见回归|固定attempt/ledger路径、Job成员、durable intent、一次性token与exclusive claim保持（308-400、1285-1324）。|
|P1-02 module启动路径|未见回归|`_worker_command()`仍以repo root下的`python -m`入口启动（1705-1717）。|
|P1-03/P1-04 deadline与整树终止|未见回归|恢复/生成deadline、Job终止确认、writer取消与有界join路径保持（1957-1972、2183-2327）。|
|P1-05 selector前source/checkpoint重验|未见回归|四端restore后重新比较readiness/source/report/checkpoint，再写selector intent（3195-3222）。|
|P1-06 NTFS耐久发布|未见回归|本地固定NTFS探测、FlushFileBuffers及write-through无覆盖发布保持（188-280、2688-2750）。|
|P2-01/P2-02 final成本与reserve|未见回归|finalization-start/cost receipt边界及10,200 reserve公式保持（1041-1053、3480-3538）。|
|P2-03 selector intent与中断尾账|未见回归|selector intent、持久face与unknown selector tail审计保持（2430-2524、3205-3222）。|
|2 KiB/face绑定、156分母、validate-only|未见回归|IPC界限、持久face query SHA绑定、156固定分母和metadata-only readiness保持（82、1216-1254、2130-2149、1013-1021、3379-3380）。|

## 回流判定

- 根任务提供的验证为完整44项合同测试、Ruff/Black/py_compile/CLI及提交门禁全部通过；本轮未重跑。
- P1=0，P2-05未关闭。按计划P2全闭门槛，当前不放行；正常terminal durable写入后同步更新标志并补故障注入，再安排下一次聚焦复审。
- 本轮不修改实现代码；Phase 8顶部主指针、历史2/2预算与先前复审结论均保持原样。