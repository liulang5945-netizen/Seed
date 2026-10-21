# PLAN-R2-DIALOGUE-RECOVERY v1：三臂事务式checkpoint恢复与从zero重做

日期：2026-10-21。状态：按用户持续推进和灵活修订预算授权确立；工程实现准入，正式训练须通过本文门禁。此为新执行合同，不修改旧06/07冻结计划。

## 事实与目标
用户已确认客户端更新导致中断。旧06进度14880/30096，07 warm9984/30096、fresh9856/30096；无训练final/partial，只有zero/canary。日志中的更新不具有可恢复权重，不能从日志续训。唯一方向是保留原配方与初始化，从三枚原zero只读恢复，在新目录重做，同时加入经证明等价的周期checkpoint。N1-166仍不准入，can_promote=false。

## 路径与身份
- 新runner：scripts/training/run_taiji_r2_daily_dialogue_recovery.py；基础层契约测试：tests/taiji_native/test_r2_dialogue_recovery_contract.py；正式runner契约测试：tests/taiji_native/test_r2_daily_dialogue_recovery_contract.py。
- 新checkpoint根：data/checkpoints/r2_daily_dialogue_recovery_20261021/{full_turn_1m,warm_start,fresh_same_capacity}/。
- 新报告：reports/taiji_r2_daily_dialogue_recovery_20261021_*，各臂独立readiness/attempt/progress/result与提交回执。
- 原zero：data/checkpoints/r2_daily_dialogue_full_turn_20261021/full_turn_zero.pt；data/checkpoints/r2_daily_dialogue_warmstart_matched_20261021/{warm_start,fresh_same_capacity}/full_turn_zero.pt。只读、SHA绑定；禁止重建新seed假称旧zero。
- 新计划、runner、基础层测试、runner测试、旧冻结计划与旧绑定源分别hash；旧zero原code_revision保持原样，新代码身份独立记录。

## 旧损失账本：已知数、极宽上界、未知部分分开
2026-10-21 UTC09:16本地只读核验；Environment.ProcessorCount=24。CIM查询沙箱拒绝，因此24是本次可见逻辑CPU数，尚非整个旧生命周期硬件容量不变的独立证明。用户确认中断、主理人已在沙箱外确认无python；以下区间终点统一保守取UTC09:20（北京时间17:20）。

|臂|progress创建UTC|最后写入UTC|最后已落盘CPU秒|其他已知CPU秒|
|---|---|---|---|---|
|06|10-20 22:03:21.831|10-21 08:57:10.122|未记录主训练CPU|data build12.65625；zero/canary restore0.6875+0.59375|
|07 warm|10-21 03:16:46.491|10-21 08:56:08.957|19922.141|preflight50.90625|
|07 fresh|10-21 03:18:33.165|10-21 08:52:35.051|19690.328|preflight32.84375|

两07 attempt.training_started_utc均03:22:03Z；progress CPU为preflight_cpu+budget.used()，已包含preflight，不得再次相加。明确可识别下界分别为warm19922.141、fresh19690.328秒；06至少已知data build及两restore合计13.9375秒，不能把该小数当成实际训练损耗。

容量法采用区间wall秒×24，覆盖该区间整机全部工作进程和子进程，比单线程wall估算保守得多：06 progress创建向下取整22:03:21至终点为974376 CPU秒；07每臂从正式训练03:22:03至终点为515448秒，再加各自preflight为515498.90625与515480.84375秒。这是相应时间区间的极宽容量上界，不是实际消耗；三者求和会重复覆盖同机并发时间，只能视为更松上界。

关键限制：06在progress创建前已做来源校验，07启动preflight前亦有导入开销；文件mtime不是进程启动证据。故上述三个数不能声称完整旧生命周期的无条件上界。完整旧消耗保留 `charged_upper=null, unknown_tail=true`。若后续有可验证证据证明全部相关进程均始于UTC10-20 00:00之后、结束于上述终点且全期至多24核，可用整机120000秒×24=2880000 CPU秒作为三臂联合极宽上界，再另计区间外data build12.65625；当前该起点仅是假设，禁止作为已证实值入账。账本必须同时存区间、容量来源、假设、下界、条件上界与null的完整上界。

未知损失不会被归零，也不以旧480000限额冒充数学上界。旧守卫可能在episode后才发现超限，未知退出CPU仍然未知。本计划直接单列新授权重做预算，不再声称“旧+新仍在480000以内”；日后取到可靠退出CPU或生命周期证据再收紧账本。

## 新增预算与授权
用户持续推进指示允许项目据实际修订，本计划明确增加一次从zero重做的额度：full_turn_1m≤240000、warm_start≤120000、fresh_same_capacity≤120000 process-CPU秒，三臂合计≤480000新增秒；此额度包含各臂启动/来源校验、保存恢复子进程、周期checkpoint、失败尝试与回滚重算，不重复收取既有data build。

另列共享恢复等价性与故障验证≤7200 CPU秒、真实zero保存恢复readiness≤1800 CPU秒、最终同面自由生成评测≤14400 CPU秒（最多4端点×39=156调用，见下节），辅助合计≤23400。新执行总增量上限503400 CPU秒。生命周期总账表达为旧实际未知损失+新增实测≤503400；若采用有证据支撑的条件极宽旧上界则另列该上界+503400，绝不写成已花费或实际预测。

不得以旧损耗未知阻止工程实现；正式训练按新增限额计费并持续保存完整进程树CPU。新增额度耗尽即留partial停止；不得自动开启第二轮重做。资源守卫：CPU/torch及BLAS均单线程；可用内存≥8GiB、全机CPU连续两次1秒采样≤85%才启动新臂；允许资源足够时三臂并行。同臂单writer锁，PID与create_time双验。每次episode预留实测最坏episode+checkpoint+restore CPU，父监督器记全部子树CPU，避免只记成功退出子进程。

## 恢复正确性及训练前保存门
严格遵循artifacts/04-recovery-architecture.md：显式episode游标，不调用旧train从头循环冒充resume；static模式、原corpus/config/order、原8轮全轮+1轮525助手配方不变。每32个完成episode、epoch末、stage转换及终点强制提交；账本记录每核累计非idle CPU与boot_time，只有boot identity相同且计数单调时才取主机CPU差作为checkpoint后的保守损耗；系统重启或计数回退时使用UTC墙钟×逻辑核上界，相关证据不足即fail closed。失败/超时的恢复子进程也用执行期间主机active CPU差保守计入。最后一条先且仅一次finish epoch，再提交下一epoch首游标。envelope保存完整trainer、RNG、history、developmental_history、游标、阶段encoder/targets、源hash与CPU前序账本hash。resume先选最新阶段账本只计一次尾账；已完成stage只恢复与核验血缘，不重复扩大损失。死owner lock原字节归档后安全回收，活PID、PID复用与身份不清均拒绝。

每次唯一编号bundle同目录tmp写入、flush/fsync、原子发布，fresh-process完整恢复核验后写commit receipt，再更新latest；保留最近两个已验证代。无receipt的孤立bundle不可自动采用。不得从progress提高cursor。强制中断最多回滚32条已执行episode，重算CPU入账。

正式训练前必须完成：合成数据旧train连续结果与新loop连续/中断恢复逐位相等；涵盖epoch finish和stage边界、损坏/tmp/receipt/manifest/双写/预算失败；三枚旧zero分别只读恢复、新envelope保存并fresh-process完整恢复通过，证明checkpoint确实可保存和恢复。source/zero SHA不变、0 dev/reference/query/正式训练更新（信用探针只在独立副本更新并独立计数）。门禁不通过只修工程，不开始正式训练。

每臂终点须核对全轮30096 episode/14517912目标字节更新、校准525/113209；两阶段global_step、history与恢复回执一致；三臂总量分别14631121。full_turn_1m保持旧06 zero配置，07两臂保持278010 active parameters，不能在恢复里扩架构或切任务。

## 终点评测及证据边界
三臂训练全终点、独立恢复通过后才允许生成。当前旧06尚无已保存39条dev generation报告；因此本任务修订07原评测合同的“复用06输出”前提，因此原“复用旧06两端输出”条件未成立：新恢复评测需一次选择与旧06同冻结规则的39个保留family，每family唯一prompt/history，绝不读reference；同次write-once报告对旧8epoch只读baseline与新三臂各生成一次，共156调用。若发现旧06完整合法generation已存在则复用其旧baseline及同面输入、严格按已存输出身份决定新增调用，不重复生成任何已消费端点。

评测入口需独立版本化架构补充及契约测试后才可运行；不得把旧07评测器默默改成恢复评测。预算14400包含所有评测恢复、来源/face校验、失败调用与持久化。attempt在首次prompt访问前独占创建，query_started先持久化、输出逐条fsync、失败留证不静默重试；合成测试先验证计数。人工样例固定同一39面前12 families，四端并列中文可读性/贴题/历史连贯/重复；表面指标只作描述，can_promote=false，不作为新独立确认或自动晋升。

## 执行顺序与收官
1. 新runner、CPU账本与合成等价/故障测试实现，审查并通过定向验证后提交。
2. 三臂zero envelope保存及fresh-process恢复前置，通过并记录资源/预算后提交readiness。
3. 资源允许并发从zero重做，监测事务checkpoint；终点核对与独立恢复。
4. 评测补充合同通过后一次39面比较、固定12样例人工复核，保留失败历史。
5. 同步active当前执行与N主线计划、交付黑板、报告及结案，提交；依据真实结果决定下一研究节点，不预先宣布能力收官。

唯一下一步：实现并验证带事务式周期checkpoint的独立恢复runner及CPU账本；当前不启动正式训练、不读取真实dev。
