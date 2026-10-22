# Phase 7 独立复审：提交 8e4c736e2

日期：2026-10-22。审查严格基于提交 `8e4c736e2` 的代码快照；重点是耐久task terminal与writer false→true尾账、P2-04和旧门禁回归。根任务已验证完整45项合同、Ruff/Black/py_compile/CLI及提交门禁；本轮不重跑测试，不访问真实dev/reference、不restore模型、不运行selector或`--execute`。worktree后续修改不计入本结论。

## 结论

**Phase 7仍不放行。** 成功`restore_completed`和`query_output_persisted`在ledger durable append返回成功后立即设置`active_task_terminal_written=true`。随后progress append失败时，异常分支重抛而不伪造第二个failed terminal；若ledger已有terminal，外层不会误标task terminal缺失。上一轮false→true writer竞态也仍保留unknown尾账；P2-04及此前P1/P2未见回归。

仍有一项P2起始边界：durable `restore_started/query_started`先写入ledger，再写progress；`active_task`和terminal-missing跟踪状态直到progress写入成功才建立。若ledger intent已durable、progress append失败，异常收敛看见`active_task=None`，可能把仍开放的ledger task漏出当前`evaluation_stopped` task identity/unknown尾账。该情况fail closed且attempt不可重试，但审计事件不完整。

计数：P0=0，P1=0，P2=1。Phase 7仍不放行，因为交付计划的门禁要求P2全部关闭；修正task-start边界并补测试后再判。

## 复核项

|检查项|结论|依据|
|---|---|---|
|成功restore terminal状态|通过|`restore_completed`耐久追加ledger后立即置`active_task_terminal_written=true`（3136-3161）。进度文件随后写入若失败，catch检查该状态并直接重抛，不再调用`failed_task()`制造第二个terminal（3162-3174）。|
|成功generation terminal状态|通过|`query_output_persisted`耐久追加ledger后立即置terminal状态，随后才更新persist计数（3342-3371）；后续异常分支同样在标志为true时重抛，不会追加`query_failed`/`query_timeout`（3372-3384）。|
|失败terminal durable标志|通过|`failed_task()`只在ledger terminal durable append成功后置true（2967-2969）；ledger append失败则保持missing。|
|terminal已耐久时不误报missing/重复terminal|通过|progress append发生在成功ledger terminal与置位之后；caught failure不会重复写失败terminal。专项测试注入restore terminal ledger成功后progress append失败，断言只有一个`restore_completed`且不选prompt（测试1836-1884）。|
|writer false→true尾账|通过|false→true二次清理测试仍断言task id/operation保留、writer退出最终确认、`task_terminal_missing=true`和`unknown_tail=true`，且没有伪造`query_timeout`（测试1801-1834）。外层状态以未闭合task布尔为依据，不因writer后续退出抹除。|
|P2-04及旧关闭项|未见回归|suspended worker、Job分配/核验后resume和启动CPU记账未改变；其余历史P1/P2实现路径本次只读核对，见下表。|

## 本轮问题

|级别|位置|问题与影响|整改及复检要求|
|---|---|---|---|
|P2-06|`scripts/training/evaluate_taiji_r2_daily_dialogue_recovery_v2.py:3111-3114,3287-3296,3563-3612`|在restore/query任务启动时，代码顺序是ledger `*_started` durable append、progress append、然后才设置`active_task_terminal_written=false`并赋值`active_task`。若ledger追加成功而progress追加失败，外层捕获异常时task身份仍为None；`task_terminal_missing = active_task is not None and not active_task_terminal_written`因此为false，评测停止事件可能不带task id/operation，也不把这个已经durable开始、尚无terminal的任务记为unknown。后续attempt拒绝/中断审计可发现开放ledger task，但本次失败尾账不完整。|在ledger `restore_started/query_started`成功durable后立即建立active task身份并清空terminal状态，再写progress；若第二份intent发布失败，则消费attempt并保留对应task id/operation、`task_terminal_missing=true`及`unknown_tail=true`，或写唯一失败terminal。增加progress intent append故障测试，分别覆盖restore和query，断言ledger开口不被漏记、attempt不重试。|

## 旧门禁回归核对

|旧问题|状态|证据|
|---|---|---|
|P1-01私有worker授权|未见回归|固定路径、Job membership、unique intent/token/claim（308-400、1285-1324）。|
|P1-02 module启动路径|未见回归|repo root下的`python -m` worker入口（1705-1717）。|
|P1-03/P1-04 deadline及进程树|未见回归|单次deadline、Job整树终止和writer取消/退出核验（1957-1972、2183-2327）。|
|P1-05 selector前来源/结果/checkpoint重新绑定|未见回归|四端restore后再校验source/readiness/checkpoint，之后持久selector intent（3195-3222）。|
|P1-06 Windows NTFS durability|未见回归|本地NTFS资格探测、FlushFileBuffers和MoveFileExW write-through发布（188-280、2688-2750）。|
|P2-01/P2-02成本receipt与10,200 reserve|未见回归|finalization-start receipt和reserve计算仍在（1041-1053、3480-3538）。|
|P2-03 selector intent/unknown tail|未见回归|selector事件和中断审计路径保持（2430-2524、3205-3222）。|
|P2-04 suspended bootstrap|关闭|`CREATE_SUSPENDED`、Job assignment核验后resume、CPU记账顺序未变（1873-1927、1974-2067）。|
|2KiB face绑定、156分母、validate-only|未见回归|固定IPC尺寸/face SHA/分母及只读readiness入口保持（82、1216-1254、2130-2149、1013-1021、3379-3380）。|

## 回流判定

- 根任务提供的完整45项及静态检查通过；本审查未重跑验证。
- P1=0，P2-06未关闭。计划门槛要求P2全闭，因此当前不放行Phase 7/W5a；修正ledger-start后progress失败边界并补两类intent故障测试，再安排复审。
- 历史2/2预算及此前不放行结论保持不变；Phase 8顶部主指针不变。