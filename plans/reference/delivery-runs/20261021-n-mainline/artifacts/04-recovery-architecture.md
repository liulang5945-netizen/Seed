# Phase 4 补充：R2-06/07 中断恢复架构合同

日期：2026-10-21。角色：architecture-reviewer。范围：只读源码审查；未训练、未读 dev/reference、未改冻结源。进程消失与无 final/partial 是上游已双次沙箱外核验的输入事实，本审查不把 attempt 的 started/training 当成存活证据。

## 根因与证据

1. `taiji/language_alignment.py:2081-2131` 每次 train 从 epoch=0 和 train 首条开始；episode 完成后才增加 global_step/episode_count、追加 history，随后 callback，最后每 epoch 调 `_finish_developmental_epoch`。checkpoint 保存完整模型/计数/history，但没有“下一条”“epoch finish 是否已执行”的恢复控制状态。`from_checkpoint` 只还原计数与 history，并不会让下一次 train 自动跳到正确位置。重复调用 train 或对 corpus 切片都不能直接宣称等价恢复。
2. 06 runner 的 progress 每条 flush，07 每128条记 CPU/进度；均只在全轮结束保存 final。异常分支能保存 partial，但宿主直接终止、掉电或进程被杀不会进入 Python 异常分支。因此现有 zero/canary 只能支持从零重做；日志中的数万次更新没有权重载体，不能恢复为已训练成果。
3. `_finish_developmental_epoch` 即使 static 模式也追加 developmental_history。恰在最后一条 callback 存 checkpoint 后中断，如果恢复直接进入下一 epoch，会漏一次 finish；重复调用则会多记。这是必须显式处理的边界。
4. `taiji/persistence.py::atomic_save` 用同目录临时文件与 os.replace，能防止普通进程中断产生半文件；没有 fsync，不能声称已证明断电持久性。旧来源无需改动。

## 唯一推荐：新增独立、显式游标、事务式恢复 runner

新增 runner 放 scripts/training，测试放 tests/taiji_native，恢复计划放 plans/reference，产物沿 reports 与 data/checkpoints 既有规则，使用独立 recovery1 attempt 命名；旧06/07目录、attempt、progress、zero及源文件只读保留。原训练配置、数据顺序、epoch数、目标更新量和三臂初始化来源不变。使用各臂独立验证过的 zero bundle，从0重做，标记 restart_from_zero，不标记 resume_from_progress。

独立实现最小显式 episode loop，按现有 train 的 `_target_pass`、repeats、指标累加/平均、计数、record/history和epoch finish完全同序执行；不 monkeypatch 类、不修改旧 train、不用新建切片 corpus 绕过 digest。仅接受 developmental_mode=static、原冻结单线程配置，未知模式 fail closed。私有API耦合以绑定源 hash 和等价性测试封住；源 hash 改变必须重新审查。

恢复 envelope 必含 schema、run/arm/stage、所有冻结源hash、corpus/config/order digest、原zero sha256、trainer完整checkpoint及其digest、next_epoch、next_episode_index、pending_epoch_finish、stage_transition、全局/阶段episode与byte累计、history长度与digest、developmental_history长度、Python/torch版本和线程配置、Python/torch RNG（有其他RNG则同时保存）、CPU账本前序hash。trainer 的 code_revision 维持原 zero 来源，新 runner 的 revision/hash 单独记录，避免改变训练checkpoint的身份字段。

提交点只能在完整 episode 后；每32 episode保存，epoch末、stage切换和终点强制保存。epoch最后一条后先执行finish，再把下一游标写为下一epoch首条；或者持久化pending finish并在恢复后严格补一次，两种协议不能混用，推荐前者。中断发生在episode内或save前只回滚到上一个已提交episode，最多重算32条；这部分CPU仍入账。不能从progress推高cursor。

每次提交写唯一编号不可变bundle，先写同目录tmp、flush+fsync，原子发布；计算文件hash，新进程恢复并校验完整digest/游标/history/前序hash后写独立commit receipt，最后原子更新latest manifest。manifest只指向已验证bundle；无receipt的孤立bundle不自动采用。保存至少最近两个已验证代，损坏最新代时显式记录fallback并累计重算CPU。目录单writer锁，恢复先确认旧PID及create_time不再存活，拒绝双写与并发同arm。fsync后也仅承诺已测试的故障模型，断电语义依平台验证记录。

阶段转换把完整pretrain final和assistant zero分别提交并独立恢复后，才进入525条校准；持久化encoder/targets，不在resume时重新fit。最终每臂全轮30,096 episode、14,517,912 target-byte更新，校准525 episode、113,209更新；全量计数从恢复trainer与commit receipt核对，不用本次进程局部计数冒充总量。新runner训练部分0 dev/reference/generation，评测必须另行在三臂终点完成后执行。

## 必须先通过的恢复等价性测试

使用纯合成train corpus与小模型，不读取真实dev，不启动正式训练。A按旧train连续完成2 epoch；B按新loop连续完成；C在第1条、中段、epoch最后一条前后、finish前后、stage转换前后强制进程退出，再由全新进程恢复到同一终点。逐项要求最终trainer checkpoint_digest完全相同、tensor逐位相同、每条history/epoch/episode_id顺序相同、global_step/episode_count/developmental_history相同；RNG后续抽样相同。不得放宽成“模型差不多”或只比损失。

覆盖tmp写一半、bundle已发布但receipt未写、receipt已写但manifest未换、manifest损坏、bundle hash损坏、错误arm/corpus/config/zero/hash、重复进程、预算不足、save失败、restore子进程失败与超时。证据报告标记每种故障实际执行结果。恢复等价性尚未实测，本合同不声称恢复功能通过。

## CPU损失上界与三臂重启预算

原冻结限额：06≤240,000 CPU秒（含既有data build 12.65625），07 warm_start≤120,000、fresh_same_capacity≤120,000；关联总额≤480,000。重启不能偷偷重置原总账。每臂区分 actual_lower、charged_upper、unknown_tail，后续准入用charged_upper扣减，而非用最后日志值。

优先取父进程/监督器已持久化的退出process-CPU，加所有restore子进程CPU与预检CPU；不能取得时用可证明覆盖全部旧进程生命周期的UTC区间乘同期机器可用逻辑CPU数量上界，覆盖父进程和子进程，另加区间外的已知prepare/data build。单线程设置不足以证明所有内部线程和子进程严格≤1核；不能直接以wall秒作为CPU上界。07有preflight/execute两个阶段，区间必须同时覆盖，或已知preflight CPU另加。没有可信起点/终点/CPU容量上界则标记unknown，禁止把原限额当成已消耗CPU的数学上界（守卫只在episode后运行，子进程还可在退出前漏账）。

令 U06/Uw/Uf 为经审核的旧消耗上界，B06=240000-U06，Bw=120000-Uw，Bf=120000-Uf；共享辅助验证CPU另列并从关联总预算预留，不重复计入data build。每臂重做30,096+525条的成本预测需取原progress分段每字节CPU最大值、剩余字节量及已测save/restore最坏成本，加明确余量；分别与B比较。三臂均须满足，合计旧charged_upper+新各臂预算+独立验证/最终评测预留≤480000。阶段内保存/恢复、canary、失败尝试、回滚重算与子进程退出前CPU全部计入。

在当前审查输入下没有足够时间戳和退出CPU证据，无法给出诚实的U与可执行重启预算；不得自动批准三臂启动。若保守上界使任一B不足，先在新恢复计划明确原预算不足及拟增总预算，再经项目预算决策节点准入，不能静默扩充或用旧日志折算成可恢复更新。每次episode前预留最坏单episode+保存+恢复CPU，监督器监测完整进程树并在硬限前终止；累计账本同时保存wall/CPU和旧损失上界，所有失败仍留证。

## 交付门

本架构审查完成；工程准入条件是恢复runner等价性与故障测试通过、旧CPU上界账本闭合、三臂预算分别可行、checkpoint独立恢复可用。唯一下一步：先实现并验证独立恢复runner及CPU账本，不启动正式训练。通过后再依据账本准入三臂从zero重做，保持N主线方向不变与can_promote=false。
