# Phase 7 独立复审：提交 c7411a688

日期：2026-10-22。复审严格基于提交 `c7411a688` 的代码快照，检查task-start intent身份、terminal状态、失败尾账、P2-04和此前门禁。根任务已验证完整47项合同、Ruff、Black、py_compile、CLI和提交门禁；本轮未重跑整套测试。另对ledger restore-intent append失败执行一个隔离synthetic复核。未读取真实dev/reference、restore模型、运行selector或`--execute`；未修改实现代码。

## 结论

**Phase 7评测器代码门通过。** P0=0、P1=0、P2=0。task intent状态现在与ledger durable边界对齐：intent append前先保存active task身份；ledger append成功后才标intent已确认并增加query_attempts；progress append失败仍保留task id/operation并报告unknown。terminal ledger append成功后立即置终态标志，后续progress或收敛错误不会生成重复failed terminal。false→true writer取消路径仍不会抹掉未写terminal的unknown尾账。P2-04与旧门禁未见回归。

此放行只针对代码审查门；真实W5b评测仍受Phase 8训练终点、checkpoint/readiness及独立恢复等既有前置门约束。

## 重点核查

|检查项|结果|证据|
|---|---|---|
|task身份在intent append前建立|通过|restore先设`active_task_intent_written=false`和active task，再durable append ledger；query同序。progress failure不会失去task id/operation（3112-3117、3287-3301）。|
|ledger intent append成功后的状态|通过|`active_task_intent_written=true`只在ledger append返回成功后设置；`query_attempts`也在该确认点后增加，progress append在其后（3116-3118、3296-3302）。|
|ledger intent append本身失败|通过|active task已记录但intent确认位仍false；外层分别计算`task_intent_unverified=true`与`task_terminal_missing=false`，unknown_tail为true（3563-3578、3609-3617）。本审查隔离synthetic注入restore_started ledger append失败，观察到task identity保留、intent未验证且unknown_tail=true。|
|progress intent append失败|通过|ledger已durable时intent确认位为true、terminal缺失位为true，`evaluation_stopped`保留task身份并标unknown；根任务已有restore/query两类progress append故障注入（测试1890-1938）。|
|成功restore/generation terminal append|通过|ledger `restore_completed` append成功后立即置terminal flag，之后才写progress；catch检测flag后重抛，不会伪造`restore_failed`（3159-3174）。ledger `query_output_persisted` append成功后同样立即置flag（3342-3384）。restore-progress失败测试断言仅有唯一durable restore terminal（测试1836-1884）。|
|没有terminal的false→true writer竞态|通过|任务起始时terminal flag为false，活动task在intent durable后保持；outer cleanup期间writer退出仍不会置terminal flag，因而`task_terminal_missing`和`unknown_tail`持续为true（3563-3617）；参数测试覆盖writer在外层清理时才停止（测试1801-1834）。|

## 旧门禁回归核对

|检查项|状态|依据|
|---|---|---|
|P2-04 suspended worker启动与CPU账本|未见回归|CREATE_SUSPENDED、Job分配/核验后resume、启动CPU计入bootstrap账本仍保持（1873-1927、1974-2067）。|
|P1-01/P1-02 私有worker授权及module启动|未见回归|固定attempt/ledger、Job membership、唯一intent/token/claim与repo-root `python -m`入口保持（308-400、1285-1324、1705-1717）。|
|P1-03/P1-04 deadline及整树终止|未见回归|任务deadline、Job Object整树终止、CancelSynchronousIo与有界writer join保持（1957-1972、2183-2327）。|
|P1-05/P1-06 source重验与NTFS durability|未见回归|selector前readiness/source/checkpoint复核和本地NTFS write-through marker发布保持（188-280、2688-2750、3195-3222）。|
|P2-01/P2-02/P2-03成本、reserve及selector尾账|未见回归|finalization-start/cost receipt、10,200 reserve与durable selector intent/unknown audit保持（1041-1053、2430-2524、3493-3542）。|
|2 KiB IPC、持久face、156分母、validate-only|未见回归|受限IPC frame、face路径/SHA绑定、156输出门与metadata-only readiness保持（82、1216-1254、2130-2149、1013-1021、3394）。|

## 放行边界

- 结论：P0=0、P1=0、P2=0；本轮Phase 7代码审查通过。
- 根任务完整47项与静态工具验证已通过；本轮只做提交blob静态复审和一个synthetic ledger-intent append失败复核。
- 真实模型、restore、selector、dev/reference与`--execute`仍需等待Phase 8终点readiness及恢复门；本报告不授权这些操作。
- Phase 8顶部主指针与历史2/2审查记录保持不变。