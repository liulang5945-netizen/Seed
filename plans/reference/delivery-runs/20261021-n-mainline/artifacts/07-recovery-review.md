# Phase 7 恢复实现代码审查

## 结论
返工；当前不得执行真实prepare/训练。静态审查期间文件持续由主理人修改，以下按最新读取版本记录，已修复的早期问题不继续列为现存finding。未读取dev/reference，未执行prepare/execute。

## P0
无已确认P0。

## P1
1. **resume不能经过正常释放锁的阶段**。主runner execute在每个stage的resume分支无条件调用reclaim_writer_lock；基础层该函数立即读取writer.lock。正常完成pretraining或捕获异常后锁已由__exit__删除，故在assistant阶段中断后恢复，先进入已完成pretraining便FileNotFoundError；普通可捕获异常后也无法恢复。需要在同一reclaim guard内允许无锁正常获取；活PID、PID复用、真stale三种身份须保持拒绝/留档策略。增加跨stage恢复与无锁恢复测试。
2. **旧stage账本重复扩大当前stage的恢复损耗**。execute对pretraining及assistant已有manifest依次budget.resume_from；若中断发生在assistant，先对更早pretraining终点取整机CPU差，budget随后max不能被assistant较新账本调低，已经覆盖的整段assistant工作再次按整机容量计入，可能立即耗尽120000秒导致有效checkpoint无法继续。应先选定最近已提交的执行stage，以该stage账本一次性计算lost tail；正常加载旧stage只核验训练血缘，不再次计费。测试比较assistant中断时是否只采用assistant最新ledger。
3. **失败恢复子进程CPU仍漏账**。主commit只在store.commit成功返回后加verification.cpu_seconds；基础层verify进程非零退出/timeout/返回损坏JSON会先抛错，主except仅写父budget。正常终点base._independent_restore也仅在成功receipt后加子CPU。再次resume可能以区间上界弥补，但本次失败报告cpu_seconds仍低报，而且没有完整进程树/失败上界账本。必须在finally计成功/失败/timeout的已证CPU或保守上界，父监督器捕获退出前CPU；fail closed并保留attempt事件。不能只保证成功路径。
4. **同boot判定不能靠累计CPU差正负**。新tail_charge用host_cpu_delta>=0视作同次启动，但系统重启后累计CPU可能重新超过旧值，差值依旧非负，导致使用不覆盖重启前丢失区间的差值。ledger须记录psutil.boot_time或等价boot identity；仅相等且计数单调时采用host差，否则回退UTC区间×容量并明确假设/拒绝时钟倒退。缺boot字段不得默认为已验证同boot。

## P2
- source_bindings绑定基础层测试但尚未绑定正式runner测试test_r2_daily_dialogue_recovery_contract.py，与计划“新测试分别hash”不一致；补上以免正式入口验收测试可漂移。
- Budget.guard目前只查内存与CPU额度，无全机CPU连续采样启动守卫；固定120秒reserve亦未来源于实测最坏episode+保存恢复，尚不能声称硬限有充分预留。正式启动前补资源readiness/最坏成本证据，或把硬上限守卫风险明确升级处理。
- 当前末端final使用旧atomic_save且不fsync；已验证事务envelope可作为权威恢复载体，但final普通文件不能另行声称断电持久性。建议明确“最终权威=已验证commit，final为只读导出”，导出损坏时从权威commit重建并写修复回执。

## 已核实修正与正向检查
- receipt CPU已改读verification.cpu_seconds；Budget使用child_seconds兼容旧独立恢复接口，早期接口KeyError/AttributeError已消除。
- 首次execute创建progress后错误判partial的分支已在最新读取中移除；attempt与progress要求显式resume，结果存在拒绝复跑。
- final_payload显式初始化并有assistant stage前置；阶段final分母按30096/14517912与525/113209核验，独立恢复后source再验；路径符合三arm新根与旧zero只读边界。
- Budget构造用此前进程CPU补入导入/校验成本，成功恢复child CPU不在parent process_time内重复出现；但仍须修正上述跨stage及失败路径。
- 计划事实已修正：07 progress含preflight，下界19922.141/19690.328；信用探针为副本更新、0正式训练更新；基础层与runner测试分别命名。

## 验证记录
python -m pytest tests/taiji_native/test_r2_daily_dialogue_recovery_contract.py -q 首次因Windows默认pytest临时目录创建失败，4项setup错误，非功能结论。改用既有规范.seed_test_tmp/recovery_review_20261021/pytest后，动态文件当时为5项测试，4通过1失败：test_tail_accounting_refuses_missing_cpu_capacity_evidence预期KeyError但未抛出。host_cpu新协议已替代旧容量直读，测试合同尚未同步，不得宣布正式runner全通过。此运行只做离线契约测试；未真实加载训练checkpoint或执行dev查询。

## 给下游
修复P1后新增“首次execute→正常训练、assistant中断→resume、捕获异常无锁→resume、失败/超时verify CPU计费、重启后CPU差为正”的合成集成测试；基础层完成自身等价/故障门后再复审。P1修复完成前不启动真实三臂。

## 回流判定
- 结论：返工
- 回流目标：python实现岗（主runner）与恢复基础层实现岗
- 返工原因：上述P1阻止可靠恢复或使预算账本不符合合同。

## 最新增量复核（替代以上已修复项状态）
主理人随后修复并已静态确认：首次execute误拒绝已移除；resume在循环前仅选assistant/pretraining最新已提交stage计一次tail，旧stage不再重复计费；无writer.lock直接正常获取，有锁才进入受guard保护回收；commit验证失败已用整机active CPU差保守补记。故原P1-1、P1-2撤销，P1-3仅剩stage-final旧恢复入口失败路径。

**当前仍需修复P1：**
- P1-3剩余：stage终点直接base._independent_restore，失败子进程仍未收到保守CPU补记；对其包裹与commit相同失败账本，并验证非零/损坏stdout/超时。
- P1-4保留：新active计数排除idle/iowait正确，但delta>=0仍不是同boot证据；记录并比较boot_time，覆盖重启后计数反超旧值的合成测试。

最新门禁仍为返工，以上最新状态优先于前文初版finding。runner契约测试需同步host计数新字段，不能把此前4通过1失败说成全通过。

## 最终增量复审：执行路径旧P1清零；prepare重试预算新增P1
本轮读取确认boot_time identity已加入ledger及tail判定，final restore失败已有host active CPU补记，正式runner测试SHA已绑定。root级lease先于progress打开，同臂progress单writer；stage锁只在存在时显式回收；resume最新阶段只计一次tail。此前P1全部修复。离线runner测试本轮9/9通过：python -m pytest tests/taiji_native/test_r2_daily_dialogue_recovery_contract.py -q --basetemp=.seed_test_tmp/recovery_review_20261021/pytest_latest。没有执行真实prepare/训练/dev。

**新增P1：prepare失败后隐式重做清零CPU。** prepare仅以readiness/result存在拒绝重试；在readiness尚未落盘时，第一次或第二次commit之后进程被中断、write_new_json失败、verify失败，均无prepare attempt/失败CPU回执。再次--prepare-only创建Budget(arm)从零开始，并通过open_store(...resume=True)在旧目录再提交zero，却没有加载旧ledger或计tail。这样旧预检/信用探针/恢复消耗不进readiness.cpu_seconds，后续execute预算不含该失败费用。该路径违反“所有失败仍入账、不隐式重试”合同。

修复建议：prepare开始前也持久化独占attempt和CPU ledger；已有prepare产物时必须明确恢复，读最近有效prepare提交与失败记录计CPU/tail，或者fail closed停止并记录未知费用而非重置。增加“pretraining zero已提交、canary/readiness前失败，再次prepare不得清零”的合成测试。不要求真实重做来验证。

**当前门禁结论：旧执行P1已清零，但此prepare预算P1未清零，仍返工。** 若新增prepare失败保护后复审通过，可解除此代码审查阻断；真实训练仍依赖基础层等价/故障测试与三arm真实保存恢复前置。

## prepare恢复补丁复审：Phase 7通过，可进入真实zero准备验证
最新prepare实现已覆盖此前新增P1：来源/语料解析前write-once发布prepare_attempt和CPU anchor；普通重复prepare拒绝，必须--prepare-resume；preparation_resume_charge验证runner SHA与parent zero SHA，校验既存zero envelope完整bindings/游标与0正式更新，合并持久化失败事件，选择最新ledger并计tail。无commit时保留初始attempt尾账。prepare失败catch写独立事件与完整ledger，来源绑定尚未可用时亦保留runner SHA和CPU anchor。新源码及两套测试均纳入source_bindings；旧zero只读。

零步回执通过prepared_zero验证完整checkpoint_digest及Cursor后复用，避免重复写parent；prepare_owner lease保护模型准备与checkpoint提交。事件采用UUID唯一名、原子发布的不可变文件，不会覆盖既有失败证据；绑定错误拒绝。正常源验证失败的catch亦写不可变事件，不能冒充成功readiness。execute以readiness.cpu_seconds计入已准备成本；root lease与显式resume、最新stage一次tail、boot identity、失败child保守计费等前次修复仍在。

本轮独立离线运行：python -m pytest tests/taiji_native/test_r2_daily_dialogue_recovery_contract.py -q --basetemp=.seed_test_tmp/recovery_review_20261021/pytest_prepare_patch，14/14通过。基础层联合24通过及Ruff由主理人另有证据，本审查不把其冒充本轮独立重跑。未执行真实prepare/execute、未读dev。

**P0=0；P1=0。Phase 7恢复专项代码审查通过，允许下一步真实三arm zero checkpoint准备与独立恢复验证。** 此放行仅针对准备验证，不代表训练终点或语言能力通过；正式训练仍以三arm真实readiness、来源/磁盘/内存/预算门全部通过为前置。

保留P2：固定120秒reserve尚需真实准备成本形成充足预留证据；全机CPU启动采样由主理人执行前门禁落实；final普通导出无fsync的断电承诺不得超出已验证事务commit范围。prepare事件在源验证失败/锁争用时可由多个调用并行追加，但UUID原子发布避免覆盖；不要把事件数量当成功准备次数。以上不阻断真实zero准备验证。

### 当前有效回流判定（覆盖前次返工状态）
- 结论：通过
- 回流目标：无
- 返工原因：原P1及prepare重试预算P1均已修复并离线验证

## source-validation失败尾账专项最终复审
主理人发现前版prepare-resume在source_bindings/语料校验失败前，catch可能回退初始attempt而丢弃既有事件已计上界；此前放行对该旧版本不再适用。本次新代码已修：preparation_journal在任何source/corpus工作前读取不可变attempt与所有事件，验证runner SHA和arm，取最新tail且不低于任何历史charged_cpu_seconds；budget先继承该值，再开始校验。后续无bindings的失败事件带runner SHA/arm及继承后的完整ledger，因此连续校验失败不能把总账归零或倒退。来源校验成功后verified-envelope的计费通过resume_from的max保持journal地板，不能被较低envelope账本覆盖。

boot identity与active CPU差仍存在，异boot回退wall×core；prepare_owner及runner root lease、显式stale回收、UUID原子不可变事件均保持前次边界。校验前journal本身损坏或runner/arm不符直接拒绝，不写低值新事件掩盖历史。来源SHA与两套测试绑定保持。

本轮独立离线runner合同15/15通过，命令：python -m pytest tests/taiji_native/test_r2_daily_dialogue_recovery_contract.py -q --basetemp=.seed_test_tmp/recovery_review_20261021/pytest_journal_final；包括新增连续3次来源验证失败的账本单调测试。联合25项与Ruff通过为主理人提供证据，本轮未重复基础层测试。未执行真实prepare/训练/dev。

**当前有效结论：P0=0/P1=0，Phase 7专项复审通过，可进入真实三arm zero checkpoint准备与独立恢复验证。** 正式训练仍需真实readiness与资源/预算前置通过；P2资源reserve与断电承诺边界仍按前文保留。

## 最终回流判定
- 结论：通过
- 回流目标：无
- 返工原因：无；source-validation连续失败的尾账单调缺口已修并合成验证

## 最后P2增量审查：启动资源与final导出
已静态确认resource_snapshot在真实zero准备工作前及训练stage循环前各取两次1秒全机CPU样本，任一样本>85%或可用内存<8GiB即拒绝；准备失败进入已有不可变失败事件/CPU账本，训练失败进入progress失败账本。准备与训练成功记录各自资源快照，禁止把一次prepare通过当成未来训练资源永久通过。

atomic_save_final使用目标同目录唯一tmp、torch.save后flush/fsync、目标存在拒绝、os.rename发布、finally清理tmp；训练stage final已改用此入口。Windows下rename不覆盖既有目标，且root lease避免本实现双写；随后独立恢复仍是有效性门。此变更解决原普通导出未fsync的P2；不额外承诺磁盘硬件/文件系统在任意断电下的目录元数据持久性。

本轮独立离线runner契约18/18通过（--basetemp=.seed_test_tmp/recovery_review_20261021/pytest_resource_final）。主理人提供联合28测试及四Python文件Ruff通过证据。本轮未真实prepare/execute/dev，P0/P1继续为0，可放行三arm真实zero准备与独立恢复验证。

原固定120秒reserve的充分性仍需真实准备成本/最坏episode证据后再作为正式训练前门禁核对，此项不阻断当前只做zero准备。当前有效Phase7结论：通过。

## v2动态预算reserve增量复审
已确认checkpoint与报告切换独立recovery_v2_20261021根，旧产物保留。每臂source_bindings绑定该臂全轮及助手最长episode两份probe报告SHA（合计六份），并验证probe zero_sha256_before等于本地旧zero实际SHA；reserve验证complete、arm、0正式更新/0 dev/reference、父zero前后SHA不变、copy更新量等于target_bytes×response_repeats。prepare与execute均在实际工作前对照冻结train corpus中最大target的episode_id和字节数，避免probe来自错误题目。

只读检查六份probe的元数据：full_turn_1m全轮597.796875秒、助手4.8125秒；warm32.9375/3.0625；fresh32.8125/2.984375，状态均complete、formal_updates=0、父zero不变。ceil((2×maxCPU+120)/60)×60对应reserve1320/240/240秒。Budget默认guard改用此值，episode前与commit前后守卫保留；ledger/readiness/结果记录reserve和报告hash。正式恢复source binding包含probe，避免在已准备/已训练期间静默改测量。

本轮独立离线runner合同19/19通过（--basetemp=.seed_test_tmp/recovery_review_20261021/pytest_v2_reserve）；联合29、Ruff和py_compile为主理人已有证据。未加载模型或运行真实prepare/训练/dev。P0/P1仍为0，允许进入v2三arm真实zero checkpoint准备与独立恢复验证。

边界：最长target测量加2倍余量是有来源的工程预留，不是所有未来状态运行时间的数学上界。120秒保存恢复余量应随真实zero提交/恢复成本核对；若实际接近或超过余量，停止正式启动并版本化修订，不宣称硬实时保证。该限制不阻断当前zero准备验证。

当前有效Phase7判定：通过；下一步仅v2真实zero准备/恢复验证，正式训练仍待其readiness通过。

## v2三arm真实zero准备结果（主理人执行，2026-10-21）
Phase7放行的准备操作已完成。三臂zero与assistant canary独立fresh-process恢复均通过，准备CPU分别125.90625/45.828125/44.859375秒，episode reserve为1320/240/240秒；双CPU样本与内存门均通过。三臂正式更新和dev/reference查询均为0，can_promote=false。结果见v2 readiness报告及`PLAN-R2-DAILY-DIALOGUE-RECOVERY_V2_READINESS_20261021.md`。这仅关闭checkpoint保存/恢复门；正式训练尚未启动。

## v2最终计划与测量绑定确认
最新静态核对与上次动态reserve复审一致：source_bindings逐臂核验两份probe的本地zero父SHA并绑定报告SHA，六份在三arm合计完整；prepare/execute均exact-match冻结train最大target的episode_id与target_bytes；run身份及checkpoint/reports根均为v2。计划已把v1成功zero准备标为保留历史、不被v2接受，并同步1320/240/240动态reserve与“先三arm v2准备、全ready再训练”的顺序。

无新增finding，P0=0/P1=0。沿用本轮前次独立19/19与主理人联合29/Ruff/py_compile证据，不无故重复测试。Phase7通过，放行v2三arm真实zero准备及fresh-process恢复验证；未执行真实prepare/训练/dev。
