# Phase 6 恢复基础设施实现

## 结论
完成独立恢复基础层；未改旧taiji、runner与冻结来源，未正式训练、未读取dev/reference。用户已确认旧中断由客户端更新触发。微型纯合成语料验证性更新仅用于工程等价验证。

## 接口实现清单
- `Cursor(epoch=0,index=0)`：stage局部下一episode游标。
- `train_next(trainer,cursor,epochs) -> (record,Cursor)`：复制旧train运算顺序，仅static/train-only；完整episode后在epoch边界先finish再返回；终点StopIteration。
- `RecoveryStore(directory,bindings,corpus,*,reclaim_token=None)`：上下文单writer锁。
- `store.commit(trainer,cursor,epochs,ledger) -> receipt`：完整trainer/cursor、来源bindings、corpus与顺序、runtime、Python/torch RNG、CPU ledger、前序receipt摘要写不可变bundle；fsync、原子发布、新进程完整恢复，再写receipt与latest。
- `store.load_latest(*,restore_rng=True) -> (trainer,cursor,envelope)`：核验manifest/receipt/bundle hash、绑定、history/update、runtime及corpus；不扫描孤立bundle推断提交。
- `reclaim_writer_lock(directory)`：上下文返回token，供内层RecoveryStore使用；原子reclaim.guard排斥普通writer和其他回收者。只在pid确实不存在或同一进程已退出后将原lock原字节rename入stale-locks；活owner、pid复用、非法内容、无法确认均fail closed。
- receipt包含bundle相对名、绝对path、sha256、checkpoint_digest及verification.cpu_seconds；后者为验证子进程从启动累计process_time，包含torch导入（退出尾部仍由上层监督账本保守覆盖）。

## 数据模型与迁移
新schema `taiji-r2-recovery-v1`，新目录使用，不迁移、不覆盖旧产物；全部代保留。latest只发布已独立恢复的receipt。Python完整checkpoint含RNG tuple，恢复仅接受本地可信绑定产物；非通用不可信上传格式。正式runner负责每32episode与阶段边界调commit、source hash重验、预算与stage转换。

## 鉴权与错误码
源bindings非空且逐值匹配，stage trainer必须从0累计并对应cursor/history；只接收train corpus。失配抛ValueError；锁/恢复worker失败抛RuntimeError；文件存在和无文件保持标准异常。锁记录pid/create_time/token，获取前后双检reclaim guard，绝不依mtime删锁。陈旧reclaim guard也不自动删除。

## 自测结果
`python -m pytest tests/taiji_native/test_r2_dialogue_recovery_contract.py -q --basetemp=.seed_test_tmp/r2-recovery-contract-v4`：10 passed in 40.97s。
`ruff check`两新增Python文件：All checks passed；ruff format已执行。
覆盖旧train连续2 epoch与新loop在0/1/2/3/4条提交恢复的最终checkpoint_digest完全相等、epoch finish、孤立提交忽略、bundle损坏拒绝、来源绑定拒绝、活writer拒绝、连续commit链与RNG不变、latest损坏拒绝，以及死owner安全回收/原字节归档、活owner/PID复用/非法内容拒绝、并发reclaim与普通writer互斥。
首次测试暴露Windows沙箱拒绝hard-link，定位后改为单writer+UUID+同目录rename；非覆盖发布前检查存在。未改冻结源。

## 遗留风险
未宣称覆盖断电；文件flush/fsync已做，Windows目录元数据持久性未实测。尚未做真实kill注入的全矩阵；当前验证已覆盖序列化恢复边界与损坏拒绝。损坏latest不自动fallback，fail closed留待显式审查。CPU账本内容与上界计算由正式runner负责，基础层只保存和绑定，不代为批准预算。lock遗留回收需显式API，不能直接重跑覆盖。完整history验证每step成本随history增长，正式运行须计入CPU预算。

## 给下游的注意事项
示例：`with reclaim_writer_lock(directory) as token: with RecoveryStore(directory,bindings,corpus,reclaim_token=token) as store: ...`（实际Python需换行缩进）。没有旧锁时普通RecoveryStore进入。每stage使用独立目录与stage绑定，assistant新trainer/history从0开始。训练runner对commit返回验证CPU计账；杀进程未返回的验证CPU需保守计费。基础层通过不代表三臂正式重启准入、训练完成或能力晋升。

## 回流判定
- 结论：通过
- 回流目标：无
- 返工原因：无

## Phase 6 追加：prepare-only 中断预算保护（2026-10-21）

结论：已修复准备阶段无持久预算锚点导致重试归零的P1风险，仅修改新恢复runner及其契约测试，没有执行真实prepare、正式训练或dev读取。

接口：`prepare(arm, *, resume=False)`；命令行新增互斥的`--prepare-resume`。初次`--prepare-only`在语料载入、探针与checkpoint操作前写write-once独立prepare_attempt，包含runner源码SHA、旧zero父SHA和完整CPU ledger；已有attempt必须显式prepare-resume。readiness/result已存在一律拒绝重跑，execute的独立attempt保持原接口。

恢复：`preparation_resume_charge(output,bindings,pretrain,assistant)`读取并核验初始attempt与所有可用零步pretraining/assistant_canary envelope，另核验不可变准备事件的来源绑定；从最新有效ledger计中断尾部，完全无envelope时保留初始attempt锚点。同boot使用全机非idle CPU计数差，boot变化使用wall×逻辑CPU上界。source验证失败也写失败CPU回执，resume验证失败仍先补回旧attempt尾账，不允许失败新事件将旧开销归零。

持久产物：独立prepare_attempt.json、每次准备session开始/失败/ready的唯一UUID JSON事件；ledger包括host_boot_time、host_cpu_seconds、snapshot_utc、logical_cpu_upper、charged_cpu_seconds。准备owner使用既有单writer/store与显式stale lock回收防双写。准备commit通过统一commit守卫，验证子进程失败/超时仍用host CPU delta入账。

零步复用：`prepared_zero`发现已验证latest时核对完整trainer checkpoint digest与Cursor(0,0)，复用原receipt，不重新commit、不覆盖父zero。来源绑定含恢复runner、基础层及两套契约测试SHA；正式更新数保持0。成功保留原readiness status=ready接口，并追加prepare_ready事件。

验证：联合两契约集`python -m pytest tests/taiji_native/test_r2_dialogue_recovery_contract.py tests/taiji_native/test_r2_daily_dialogue_recovery_contract.py -q --basetemp=.seed_test_tmp/r2-recovery-prepare-contract`：24 passed in 40.73s。四新增Python文件ruff check全部通过；runner与其契约测试ruff format完成。
新增纯模拟覆盖无checkpoint尾账、已有验证envelope优先、ready及隐式重试拒绝、receipt复用与旧父SHA不变、完整CPU ledger事件。未调用真实prepare。

回流判定：通过。正式启动仍由主理人完成独立审查与预算准入。

## Phase 6 复核修正：重复来源验证失败的CPU单调账本

复核定位到上一版prepare-resume先验证source/corpus、随后才加载准备事件账本，来源验证异常路径仅回退初始attempt；后写的新失败事件可能遮蔽较早但收费更高的事件。现已移除这条回退路径。

新增`preparation_journal(output,arm)`，在任何source_bindings、语料验证或checkpoint恢复之前核验初始attempt及全部准备事件的runner SHA与arm，选择最新ledger计算尾部，且不低于全部历史charged_cpu_seconds最大值；先将其入Budget。后续验证通过后的checkpoint envelope只允许通过max补充已有预算，不得减账。日志损坏或runner/arm身份失配直接拒绝，不写一条未经闭合的更低费用事件覆盖证据。

所有session/failed/ready/source-validation-failed事件均明确runner_sha256与arm。新增回归测试预置500秒失败账本和10秒尾部，模拟随后连续三次resume来源验证失败，确认每次新增事件费用单调不减，且不回到初始10秒attempt。

联合验证：两契约集25 passed in 41.01s；四文件ruff check通过。仅合成与mock校验，未执行真实prepare/正式训练/dev。回流判定：通过。
