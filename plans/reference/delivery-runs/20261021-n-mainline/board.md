# N 主线交付黑板

- 运行目录: E:\Seed\plans\reference\delivery-runs\20261021-n-mainline
- 当前阶段: Phase 8 测试/执行监控（续训扩展实现与执行前门禁待完成；真实训练未启动，W5b仍阻断）
- 任务一句话: 延续当前R2-06/07冻结训练，补齐终点评测与证据闭环，依据结果推进N主线。

## 阶段记录

### Phase 1 立项核对（delivery-team-lead）完成
- 结论: 当前代码HEAD b6d7292da，06/07训练progress仍在追加；沙箱Get-Process只显示沙箱内进程，不能据此判训练退出。
- 产物: artifacts/01-intake.md
- 关键要点: 不重复启动训练；不改冻结来源；07配对评测器尚未存在；N1-166仍不准入。
- 风险: 尚无训练中途checkpoint；attempt状态不能单独证明存活或完成。
- 给下游: 读取03当前执行、06/07冻结计划及两个runner。

## 决策记录
- 2026-10-21: 用户授权持续推进；本次新建交付编排，不伪造旧交付进度。遵循仓库目录规范，将技能运行目录放在plans/reference下，不新增根级目录。
- 训练合同及源码绑定保持原样；评测器为独立新增交付，先静态验证，三候选训练结束且独立恢复后才查询保留prompt。
- 2026-10-21 23:33：根据Phase 8中途审计及三臂最新durable receipts，原Phase 5指针已过时，现对齐到Phase 8执行监控；此次对齐不代表训练/评测通过或项目晋级，继续遵守终点恢复前不查询留出面。
- 2026-10-21：Phase 3需求岗新增R2 recovery v2评测需求补充，因旧06未保存39面generation且v2 result schema不同；只替代v2评测调用与验收口径，不改训练合同、N1-166冻结门或旧R2-07历史。下游实现与测试按R2V2-01..07执行。
- 2026-10-22：为关闭P7的P1-06，architecture-reviewer裁决仅支持Windows本地固定NTFS，以exclusive-temp写入/刷新后`MoveFileExW(MOVEFILE_WRITE_THROUGH)`无覆盖发布attempt和initial ledger；网络、ReFS、可移动及未验证文件系统拒绝。架构补充已提交`75cba6ec5`；Phase6可按该边界返工，Phase8主训练指针不变。
- 2026-10-22：Phase 3需求岗按最新Phase 4长驻worker/deadline合同新增R2V2-08..10运行时预算验收，不改R2V2-01..07、156分母、训练合同或能力边界；若预算/有限deadline/进程树取消能力无法证明，selector与generation均须为0。
- 2026-10-22：用户批准R2 continuation extension v1累计预算每臂1,000,000、三臂聚合3,000,000 process-CPU秒；额度包含旧实耗与未知尾账，仅授权一个版本化续训窗口，不重置、不覆写旧证据。预算决策草案已完成；须先实现并独立审查extension runner、checkpoint谱系、source compatibility与CPU账本，所有门通过后再从各自父checkpoint续训。

## 回流记录

### 回流 1 · Phase 7 评测审查 → Phase 6 AI实现
- 门禁：code-reviewer / 目标：ai-integration / 预算：2（已用2）/ 状态：未解决（预算用尽；剩余P1待主理人安排定向处理）。
- 原因：P1共6项：restore-worker可绕attempt/预算；子进程直接脚本启动缺仓库import root；restore与generation无硬deadline；prompt selector前未复核source/result/readiness；attempt/ledger仅fsync文件，Windows目录项durability未证明。P2共3项：最终CPU/wall成本低报final写盘；岗位reserve计算错误；selector前缺耐久意图事件。
- 复检范围：全部P1/P2，新增负向/中断测试；不得访问dev/reference或运行模型/真实selector/`--execute`。
- 当前先决：P1-06已由architecture-reviewer补充合同并提交`75cba6ec5`；Phase 6 AI实现可开始六项P1/三项P2返工，Phase 8三臂训练指针维持不变。

## 风险与待办
- 执行门通过前不写续训checkpoint；通过后只写版本化extension checkpoint根，不改旧checkpoint。训练中不读取dev/reference，不把格式统计或工程门通过当成语言能力。
- 当前任务不能承诺“完美”或能力晋升；收官需真实实验、可读性审阅和有效门槛证据。

### Phase 4 架构审查（architecture-reviewer）完成
- 结论: 通过；07独立评测合同v1明确，真实评测因缺三run终点严格阻断。
- 产物: artifacts/04-architecture.md
- 关键要点: 逐schema映射06/07字段；只复用06保存prompt；78新查询/156组合输出；消费attempt后不重试；CPU≤7200且不得突破07原臂预算及关联480000秒。
- 风险: 主理人沙箱外核实无python进程，三progress停滞且无partial/final，旧attempt不证明存活；CI日志403无法归因。
- 给下游: 只新增评测入口及合成契约测试，不访问真实dev、不启动模型；恢复训练须另立版本化合同。
- 回流判定: 通过；无上游返工。

### Phase 4 独立补充：训练中断恢复架构审查完成
- 角色：architecture-reviewer；产物：artifacts/04-recovery-architecture.md。
- 根因：旧训练器无恢复游标，06/07无周期checkpoint；progress不能恢复已丢失权重，异常partial不覆盖宿主终止。
- 合同：新路径独立runner，显式episode/epoch/stage游标、32条周期事务保存与新进程恢复、合成数据逐位等价及故障注入验证；冻结源和旧失败证据保持原样。
- 预算门：旧CPU上界先入账，06/07各臂余额及480000关联总额均需可证明可行；当前旧上界尚不闭合，禁止直接启动重做。
- 唯一下一步：实现并验证独立恢复runner与CPU账本。本补充不修改其他角色阶段指针；0 dev/reference/正式训练。

### Phase 2 恢复补充（implementation-planner）完成
- 结论: 用户确认更新中断；新版本化计划从三只读zero重做，旧冻结合同原样保留。
- 产物: artifacts/02-recovery-plan.md；../../PLAN-R2-DAILY-DIALOGUE-RECOVERY_20261021.md
- 关键要点: 新增训练480000 CPU秒，辅助23400，共503400；旧完整损失未知且不归零；24核区间容量法只给条件极宽界；先合成等价/故障测试与真实zero保存恢复门。
- 风险: 原06未保存39面generation，恢复评测须新合同允许一次四端点156调用，不能假称复用不存在输出。
- 给下游: 实现独立恢复runner与账本；主理人记录新预算取代旧总额内扣约束，不改旧冻结计划；当前不启动训练/dev查询。
- 主阶段指针: 保持不变（专项补充）。

### Phase 6 独立补充：恢复基础层实现完成（backend-architect）
- 产物：artifacts/06-implementation-backend.md；新增taiji_r2_dialogue_recovery.py及合成契约测试。
- 要点：static显式cursor与epoch finish精确时序；不可变bundle/receipt/latest事务和新进程恢复；单writer与显式安全stale锁回收。
- 验证：定向10 passed（40.97秒），ruff通过；合成验证更新，0正式训练、0 dev/reference。
- 风险：不声称断电保证；真实kill故障矩阵及CPU尾部由后续集成验证；损坏latest fail closed。
- 给下游：正式runner组装32episode周期保存、阶段与来源绑定、CPU监督账本；旧进度不伪作权重恢复。
- 回流判定：通过；不修改其他岗位阶段指针。

### Phase 7 恢复专项代码审查（code-reviewer）返工
- 结论: 当前主runner存在P1恢复/CPU账本问题，禁止真实prepare/训练。
- 产物: artifacts/07-recovery-review.md
- 关键要点: 无锁stage被无条件stale回收阻断；resume重复采用旧stage账本；失败子进程CPU漏计；同boot判定缺boot identity。
- 验证: 离线runner契约测试4通过1失败（host CPU协议与缺容量测试未同步）；Windows临时目录问题通过指定.seed_test_tmp绕过。
- 给下游: 修复后补跨stage/无锁恢复及失败子进程/重启账本合成集成测试，再复审；不读dev。
- 主阶段指针: 保持不变（专项审查）。
- Phase 7增量复核: 首次execute、无锁恢复、最新stage一次计费、commit失败CPU已修；剩余P1为stage-final失败子进程CPU及boot identity，仍需复审。
- Phase 7最新复审: 旧执行路径P1全部修复，runner离线9/9通过；新增P1为prepare在readiness前失败后隐式重做将旧CPU清零，需prepare attempt/ledger或明确拒绝重试保护。仍返工，不执行真实prepare/训练。

### Phase 6 专项追加：prepare-only 中断预算保护完成
- 产物：06-implementation-backend.md追加；新恢复runner与其契约测试。
- 修复：write-once prepare attempt、session/失败/ready CPU事件、显式prepare-resume，已验证zero envelope/receipt复用；ready后拒绝重跑。
- 账本：同boot全机非idle CPU delta，boot变化wall×core；无checkpoint仍有初始attempt尾账；验证child失败开销保留。
- 验证：两契约集联合24 passed（40.73秒），四文件ruff通过；未真实prepare/训练/dev。
- 回流判定：通过；等待主理人独立审查，不改变其他阶段指针。
- Phase 7 prepare补丁复审通过: P0=0/P1=0；本轮独立离线runner合同14/14通过。prepare持久化attempt/失败ledger、显式prepare-resume、最新zero/事件tail计账和回执复用已覆盖原缺口。允许进入真实三arm zero checkpoint准备与独立恢复验证；正式训练仍须真实readiness及资源/预算门。未执行真实prepare/训练/dev。

### Phase 6 复核追加：来源验证重复失败CPU账本单调性修正
- 恢复前先核验attempt+全部同runner/arm事件并计尾部，后续source/corpus/restore失败不再回退初始attempt；envelope仅max补充。
- 每个prepare事件显式runner_sha256与arm；连续3次来源失败回归证明收费不归零、不下降。
- 验证：联合25 passed（41.01秒），四文件ruff通过；0真实prepare/正式训练/dev。
- 产物：artifacts/06-implementation-backend.md追加；回流判定通过。
- Phase 7 source失败尾账最终复审通过: preparation_journal在校验前继承attempt及全部事件并保持历史charged地板，连续3次失败不归零；独立runner15/15通过，P0/P1均0。可进入真实zero准备与恢复验证；未执行真实prepare/训练/dev。

### Phase 8 v2真实zero准备验证通过（2026-10-21）
- 三arm均通过`--prepare-only`，zero full-turn与assistant canary分别fresh-process恢复2/2；父zero SHA不变。
- full_turn_1m readiness CPU125.90625秒，reserve1320；warm_start 45.828125秒，reserve240；fresh_same_capacity 44.859375秒，reserve240。
- 三arm正式更新/dev/reference查询均0，can_promote=false；可用内存约28GiB，CPU启动双采样均通过。
- 六份最长消息probe绑定source SHA：1M最大全轮14,536 bytes/597.796875CPU秒、assistant489 bytes/4.8125秒；warm/fresh全轮32.9375/32.8125秒、assistant3.0625/2.984375秒。零点checkpoint未改变。
- v1 readiness已由max-episode校准合同取代，不用于训练；v2 readiness/零包是当前唯一训练入口。
- 唯一下一步：资源允许时并发执行三臂v2 fixed training，监测32-episode durable commits，先不查询reserved dev。
- 产物：plans/reference/PLAN-R2-DAILY-DIALOGUE-RECOVERY_V2_READINESS_20261021.md；对应v2 readiness及六份probe报告。

### Phase 8 v2三臂训练执行中（2026-10-21 19:06启动）
- 三臂在资源样本通过后并发启动，使用v2新恢复根；每32 episode写不可变bundle、fresh-process恢复、receipt后推进latest。
- 最新durable receipts：full_turn_1m 4,160/30,096、2,003,782字节更新、CPU14,892.171875秒；warm/fresh各6,336/30,096、3,038,901字节更新、CPU14,537.078125/14,540.359375秒。fresh已追平warm，latest恢复通过、0dev/reference。
- full_turn_1m首个32-episode checkpoint与两个300K臂的zero checkpoint均有恢复receipt；仍在训练，0 dev/reference查询。
- 唯一下一步：继续监测并发训练及durable checkpoints，按固定预算完成后验证最终checkpoint与分母。
- can_promote=false；不消费39-family留出面。端点齐备且独立恢复通过后再进入一次性自由生成评测。
- Phase 7最终P2审查通过: 启动双CPU采样/8GiB门及失败账本、final同目录fsync/write-once rename已落实；独立runner18/18通过，P0/P1仍0。可放行三arm真实zero准备验证；正式训练前仍核对预算reserve充分性。未真实prepare/execute/dev。
- Phase7 v2动态reserve复审通过: probe零父档/最长train episode及SHA绑定有效，reserve1320/240/240；独立runner19/19，P0/P1仍0。放行v2三arm真实zero准备验证；120秒恢复余量随真实成本核对，未真实prepare/训练/dev。
- Phase7 v2最终同步确认: 计划已标v1准备被v2取代、动态reserve及新路径；本地zero/probe SHA和最大train题目exact-match保持。P0/P1=0，允许v2真实zero准备验证，无新增测试/真实训练/dev操作。

### CI审查更新（远端run 39537675979）
- Ubuntu Ruff I001已在本地修复，当前全仓ruff通过；Windows全回归141项collection error均因测试job漏装psutil，workflow已补依赖。
- Startup smoke与docker-build通过；Windows/Ubuntu详细日志已从GitHub connector取证，CLI 403不再阻断归因。
- 当前修复尚未推送，远端重跑仍待代码发布；不声称CI整体全绿。

### Phase 8 v2 三臂训练中途证据审计（2026-10-21 23:32 Asia/Shanghai）
- 结论：三臂仍在固定全轮预训练；latest receipt / bundle 哈希和记录的恢复digest关系自洽，但没有终点报告，Phase 8 不可结案。
- 产物：artifacts/08-execution-status-20261021.md。
- 当前durable进度：full_turn_1m 4,320/30,096、2,069,142 target-byte updates、CPU15,516.546875秒；warm_start 6,528/30,096、3,168,217 updates、CPU15,127.40625秒；fresh_same_capacity 6,528/30,096、3,168,217 updates、CPU15,131.671875秒。
- 验证范围：三条latest指针所列receipt文件SHA均匹配，receipt所指bundle文件SHA均匹配，receipt的verification.digest均等于digest；attempt均为training、dev_queries=0、can_promote=false。此审计未重新运行fresh-process worker；恢复字段引用runner写入的receipt。
- 缺失门：每臂全轮30,096与assistant 525 episode固定分母、final report/checkpoint、final fresh-process恢复及后续独立一次性留出面评测均未完成。
- 唯一下一步：继续监测同三臂v2训练与每32 episode durable commits；不得查询dev/reference，直至三个固定终点及最终恢复门全部完成。
- 状态：中途更新；不表示Phase 8通过或项目可晋级。

### Phase 8 v2 进度刷新（2026-10-21 23:33 Asia/Shanghai）
- 结论：三臂仍在固定预训练，耐久提交继续增长；1M臂4,320/30,096，对照两臂各6,560/30,096。
- 最新durable状态：full_turn_1m 2,069,142 updates、CPU15,516.546875秒；warm_start 3,187,955 updates、CPU15,220.0625秒；fresh_same_capacity 3,187,955 updates、CPU15,224.421875秒。
- 最新receipt核验：三臂progress、latest与receipt一致；receipt SHA及bundle SHA均匹配，恢复digest与receipt digest相同。此前23:32的审计产物保留其当时快照；本次按23:33实际写盘状态更新路线图。
- 唯一下一步：继续监测同三臂至固定分母和final fresh-process恢复完成；期间不查询dev/reference，不判定能力晋级。
- 状态：执行中，Phase 8未结案。

### Phase 4 补充：R2 recovery v2 四端点评测架构（2026-10-21）
- 结论：通过；旧04架构仅适用于旧06/07接口及78新查询/156组合输出，不能直接用于恢复v2结果。
- 产物：artifacts/04-v2-evaluation-architecture.md。
- 关键要点：当前默认比较旧parent只读baseline + `full_turn_1m`、`warm_start`、`fresh_same_capacity`三条assistant_sft final，共39 family×4 endpoint=156新调用；v2最终result schema为`taiji-r2-dialogue-recovery-result-v1`，校验`bindings`/阶段计数/恢复receipt/所有源和checkpoint SHA。
- 风险：source或endpoint身份漂移、缺少final report、旧generation存在但无法逐端点严格匹配时必须在读prompt前拒绝；评测预算使用恢复计划单列14,400 CPU秒，不沿用旧7,200上限。
- 给下游：评测runner与synthetic契约测试可和Phase 8训练并行开发；真实39面读取/生成仍须等待三臂final与独立恢复全部完成。
- 回流判定：通过，无返工；本补充不阻塞Phase 8训练。

### Phase 2 补充：R2 recovery v2 W5计划修订（2026-10-21）
- 结论：旧计划中的78次新增/复用旧06输出仅作历史；W5按v2四端点和新终点接口执行。
- 产物：artifacts/02-plan.md（追加“R2 recovery v2 W5修订计划”）。
- 关键要点：parent只读baseline+三条v2 assistant_sft final；39×4=156新调用；14,400 process-CPU秒与14,400 wall秒双门；合成实现/测试可并行训练。
- 风险：真实数据面必须等四端点恢复和账本完整；attempt/CPU ledger须在首个restore前耐久落盘，失败消费attempt且禁止重跑。
- 给下游：AI集成岗按04-v2架构实现合成验证；Phase 8主训练指针不变，不访问dev/reference，不改训练合同。
- 状态：计划补充完成；不推进主阶段指针，不表示评测或训练通过。

### Phase 3 补充：R2 recovery v2评测需求（2026-10-21）
- 结论：通过；旧R-01..04保留历史，R2V2-01..07定义本次v2评测的可验收行为。
- 产物：artifacts/03-requirements.md（追加“R2 recovery v2需求修订”）。
- 关键要点：parent+三v2终点；39×4=156新生成；restore前write-once预算账本；CPU/wall各14,400秒；零reference/答案评分/训练更新；can_promote=false。
- 风险：任何source、checkpoint、prompt face、账本或budget证据不完整均须在留出prompt读取前拒绝；用户能力目标不能由此小样本自动判定。
- 给下游：按R2V2需求实现synthetic测试；待Phase8终点独立恢复通过后才运行真实评测。
- 回流判定：通过；无需求缺口需要退回Phase2。本补充不改变Phase8主指针。

### Phase 3补充：R2 recovery v2运行时预算澄清（2026-10-22）
- 结论：通过；将最新Phase 4长驻worker、逐条授权、单次deadline与进程树终止约束转为R2V2-08..10及AC-R2V2-08..10。
- 产物：artifacts/03-requirements.md（追加“R2 recovery v2运行时预算澄清”）。
- 关键要点：单个父进程监督长驻worker；四endpoint durable restore前selector为0；query逐条fsync授权；所有restore/generation有有限期限；进程树CPU/wall和终止/落账计入14,400双门。
- 风险：取消树、预算或期限任一不可证明时，执行在selector前fail closed；不改变156次分母和已冻结训练来源。
- 给下游：Phase6实现与Phase7审查按R2V2-08..10补synthetic timeout/进程树/预算测试，主Phase8指针保持不变。
- 回流判定：通过；架构已明确可实现合同，无上游返工。

### Phase 2 补充：W5a Phase 7返工设计澄清（2026-10-22）
- 结论：计划已补齐单父进程监督的长驻worker、durable restore门与逐query IPC授权顺序；旧P7打回结论仍有效，尚未返工通过。
- 产物：artifacts/02-plan.md（追加“Phase 2 W5a返工设计澄清”）。
- 关键要点：四endpoint回执durable完成后才可selector；worker没有selector/预取/重试权限；每条query父进程先fsync授权。
- 风险：restore/generation需有限deadline和整棵worker tree有界终止；CPU/wall、取消与finalization共用原14,400秒双门；attempt/ledger只允许已验证的Windows本地固定NTFS write-through发布。
- 给下游：关闭全部6项P1、3项P2并通过对应synthetic故障测试后再回P7；W5b仍等待三臂终点及独立恢复，Phase 8主指针保持不变。
- 状态：Phase 2计划澄清完成；未修改代码或读取真实数据。

### Phase 6 补充：R2 recovery v2独立评测实现（2026-10-21）
- 结论：W5a实现完成；端点固定parent基线加三条v2 `assistant_sft` final，39×4=156次新生成。Phase8主阶段指针保持不变。
- 产物：artifacts/06-implementation-ai.md；新增`evaluate_taiji_r2_daily_dialogue_recovery_v2.py`与对应synthetic契约测试。
- 关键要点：metadata-only readiness与source/checkpoint SHA核验；restore前write-once attempt和fsync预算账本；每个restore前写durable事件；四个restore全部通过后才选prompt；generation每条先记`query_started`再落盘输出；失败消费attempt且禁止重试/续跑。
- 验证：定向契约12/12通过（5.76秒），新增文件Ruff、py_compile与CLI `--help`通过；另有restore失败与generation失败二次调用的独立拒绝复测各1通过。
- 风险：真实三臂final/readiness/checkpoint与独立恢复仍未齐；本轮未运行真实validate、restore、selector或generation。Phase7代码审查是W5b之前门禁。
- 给下游：先独立代码审查；只有所有三臂终点restore齐备后才可真实`--validate-only`，再执行一次性156调用。结果保持`can_promote=false`，不打分reference答案。
- 调试留痕：初次synthetic 8/10中的两项根因和逐项复测顺序见artifacts/06-implementation-ai.md；后续二次执行断言改为明确的consumed/refuse语义，runner保持fail closed。
- 回流判定：通过；架构与需求充分，无上游返工。本岗位完成不代表Phase8通过或模型晋级。

### Phase 7 补充：R2 recovery v2评测代码审查（2026-10-22）
- 结论：打回；P0=0、P1=6、P2=3，W5a暂不放行Phase 7合入门。
- 产物：artifacts/07-evaluation-review.md。
- 关键阻断：公开restore-worker可绕attempt/预算；子进程路径启动缺repo import root；restore与generation均无硬deadline；prompt selector前未重验source；attempt/ledger仅fsync文件、缺架构要求的父目录durability（Windows本地目录fsync实测PermissionError 13）。
- 其他问题：最终CPU/wall total低报final result写盘成本；initial reserve说明算术不符；selector访问前缺耐久阶段事件。
- 验证范围：静态审查；引用Phase6报告的synthetic 12/12但本次未重跑；0真实模型/selector/dev/reference/--execute。
- 给下游：P1逐项返工并补失败注入覆盖后复审；三臂Phase8训练继续独立推进，真实39面仍冻结。当前主阶段指针保持Phase8。
- 回流判定：打回ai-integration；若Windows目录durability无可验证实现，先交architecture-reviewer修订平台合同。

### Phase 7 第二轮独立复审：R2 recovery v2评测器（2026-10-22）
- 结论：不放行；P0=0、P1=1、P2=1。首轮问题中P1-01..06与P2-01..03均已关闭；残留P1-07为`SupervisedWorker.request()`在deadline loop前同步写入完整generation IPC，worker未读取且payload超过管道缓冲时可阻塞父进程并越过14,400秒wall门。P2-04为Job assignment前CPU采样窗口及Job对象未建立或树终止无法确认时的整树退出证据缺口；当前路径fail closed、不进入IPC/selector。
- 产物：`artifacts/07-evaluation-review-rework.md`。
- 验证：根任务独立报告新增synthetic测试37/37通过、Ruff与py_compile通过；本审查未读dev/reference、加载模型、运行worker、selector或`--execute`。
- 风险与待办：Phase 7复审预算已用尽（2/2）；selector与真实156调用继续冻结，待主理人安排剩余P1定向修复及后续门禁。
- 回流判定：P1-07未关闭；本次不放行，也不将返工预算外的新复审视为已授权。
- 阶段指针：保持Phase 8三臂训练/执行监控；不代表训练通过、模型晋级或W5b解锁。

### Phase 6 追加返工：IPC deadline与worker启动隔离完成（2026-10-22）
- 修改：2 KiB IPC上限、generation仅传持久prompt face path/index/SHA；独立writer线程受父进程deadline监督；worker suspended启动，Job Object分配及成员验证后才resume。
- 验证：针对性IPC/Job合同4/4通过；评估器模块全套40/40通过；Ruff、py_compile、模块`--help`与diff检查通过。一次全套运行出现的1条失败是测试仍读取实现已替换的旧ledger键，更新断言后精准复测1/1及最终全套40/40均通过。
- 复审状态：第二轮P7原结论和预算2/2保留为历史；用户已明确要求持续推进至收官，故对这份新补丁安排额外独立复审。当前P7仍未通过，W5b仍锁定。
- 训练证据：最后durable进度04:52:46-04:53:28；full_turn_1m 9,472/30,096，warm_start与fresh_same_capacity各14,528/30,096；三attempt仍标`training`但无新progress/final。沙箱进程清单不足以确认外部会话，未重启、未覆盖、未读取dev/reference。
- 主线边界：N主线仍以自然中文多轮对话为当前能力目标；本补丁只修评估门禁，不构成能力提升或晋级证据，`can_promote=false`及N1-166冻结保持不变。

### Phase 7 第三轮独立复审：提交 4ce91e141（2026-10-22）
- 结论：不放行；P0=0、P1=1、P2=0。P2-04已关闭：worker暂停创建、Job分配与核验完成后才resume，启动CPU受Job/bootstrap账本覆盖。P1-07主要IPC写入阻塞已受线程deadline监督、2 KiB上限与持久face引用保护；仍有writer线程超时未退出时，失败收敛会丢失`ipc_write_thread_stopped`，可能遗漏当前query的unknown尾账。
- 产物：`artifacts/07-evaluation-review-4ce91e141.md`。
- 验证：根任务40项synthetic测试通过；本次独立定向复跑8项通过（32项deselected，6.07秒）；未读取dev/reference、加载模型、运行selector或`--execute`。
- 回流判定：P1-07未关闭；需保留writer退出状态、把未闭合query明确记为unknown，并补Job已终止但writer仍存活的synthetic故障测试。历史回流预算2/2及第二轮打回原样保留；本轮新提交复审不等于预先放行。
- 阶段指针：仍为Phase 8；真实156次评测/W5b继续冻结。

### Phase 6 追加返工：writer取消与失败尾账修复（2026-10-22）
- 第三轮P7结论由`2b65b7e30`记录：P0=0/P1=1/P2=0，P2-04关闭，P1-07仍因writer退出状态可能丢失而不放行。
- 根因复现：合成writer未停止时，原失败终态缺`ipc_writer_exit_unverified`，且generation task会被清空。修复在Job终止后调用`CancelSynchronousIo`，有界join并核验线程；仍不确定则保留task/operation，写`unknown_tail=true`和writer未确认字段，不再抛弃异常属性。
- 验证：原生Windows同步管道取消、deadline writer退出及unknown尾账故障测试通过；完整合同43/43，Ruff、py_compile、CLI帮助、diff检查通过。
- 状态：本修复待新一轮独立Phase 7审查。历史预算2/2与第三轮不放行仍保留；复审前不运行真实restore/dev/reference/selector/`--execute`。
- 训练：三臂最新durable进度仍为4:52-4:53快照、attempt记`training`且无final；未从沙箱确认外部训练会话状态，不重启、不覆盖。
- 主线边界：目标仍为自然中文多轮对话能力。本修复只改善监督与证据链，不作为模型能力证据；`can_promote=false`和N1-166冻结不变。

### Phase 7 第四轮独立复审：提交 1dabdc963（2026-10-22）
- 结论：不放行；P0=0、P1=1、P2=0。P2-04关闭；取消同步writer、join核验、失败task身份保留和成功响应前writer检查均已落实。残留P1为writer第一次join仍存活、外层二次清理才退出时，`failed_task()`此前未写query terminal，而`evaluation_stopped`可能按当前writer已停记`unknown_tail=false`。
- 产物：`artifacts/07-evaluation-review-1dabdc963.md`。
- 验证：提交对应记录为完整合同43/43与Win32匿名管道取消集成测试通过；本轮按要求未重跑。未读取真实dev/reference、restore模型、运行selector或`--execute`。
- 回流判定：需显式保存task terminal缺失状态，并补false→true writer竞态测试；历史预算2/2和第三轮不放行结论保持原样。
- 阶段指针：保持Phase 8；真实评测仍冻结。

### Phase 6 追加返工：task terminal缺失显式记账（2026-10-22）
- 第四轮审查`7859f7b1e`针对`1dabdc963`仍判P0=0/P1=1/P2=0：writer第二次清理才停止时，真实线程状态虽已确认，query terminal仍未写，旧`unknown_tail`可能误记false。
- 红测复现`KeyError: task_terminal_missing`。修复新增durable terminal状态位；仅在ledger terminal落盘后置true；异常收敛时未闭合task总是记`task_terminal_missing=true`与`unknown_tail=true`，保留task id/operation。
- 验证：false→true writer退出竞态测1/1、模块合同44/44；Ruff、Black、py_compile、CLI帮助与diff检查通过。
- 状态：新修复尚未提交、待独立Phase 7复审；第四轮不放行结论仍是对旧提交的有效历史证据。真实评测及W5b继续冻结。
- 训练快照未变：三臂last durable分别为9,472、14,528、14,528 episodes，attempt仍标training且无final；沙箱不能证明外部会话存活，不重启、不覆盖。
- 方向与能力边界：继续以自然中文多轮对话为N主线当前目标；门禁修补不代表模型能力提升，`can_promote=false`及N1-166冻结不变。

### Phase 7 第五轮独立复审：提交 aed03128a（2026-10-22）
- 结论：不放行；P0=0、P1=0、P2=1。false→true writer二次清理后的`task_terminal_missing/unknown_tail`已正确保持unknown；P2-04与旧P1/P2未见回归。
- 新发现：成功restore的`restore_completed`或generation的`query_output_persisted`写入ledger后，正常成功路径未同步置`active_task_terminal_written=true`。若restore terminal已durable而随后progress append失败，outer仍可能误报task terminal缺失，违反P2全闭门槛。
- 产物：`artifacts/07-evaluation-review-aed03128a.md`。
- 验证：根任务已验证44项合同、Ruff/Black/py_compile/CLI和提交门禁；本轮未重跑。未读dev/reference、restore模型、运行selector或`--execute`。
- 回流判定：P2-05需在正常成功terminal ledger append后更新状态，并补terminal已durable但后续progress失败的故障注入；此前2/2历史预算与复审结论保持原样。
- 阶段指针：保持Phase 8；P7/W5a当前不放行，真实评测继续冻结。

### Phase 6 追加返工：task-start intent身份与尾账（2026-10-22）
- 第六轮复审`1b8eceb5c`针对`8e4c736e2`判P0=0/P1=0/P2=1：restore/query started intent ledger已durable但progress追加失败时，旧顺序还未建立active task身份。
- 红测复现restore与generation两条路径的停止事件task_id为空。修复在intent事件前记录active task；ledger durable返回后置intent确认并按query递增query_attempts。progress失败时task id/operation保留、`task_terminal_missing=true`、`unknown_tail=true`。
- 验证：restore/query progress故障注入2/2通过；模块合同47/47，Ruff、Black、py_compile、CLI帮助和diff检查通过。
- 状态：当前修复未提交/未独立复核；第六轮P2-06不放行结论对提交`8e4c736e2`仍有效。
- 训练进度无更新：last durable 9,472/14,528/14,528，attempt=`training`无final；外部会话不可由沙箱确认，不重启、不覆盖。
- N主线方向仍是自然中文多轮对话能力；门禁改进不代表模型能力，`can_promote=false`、N1-166冻结与W5b冻结保留。

### Phase 6 追加返工：正常terminal状态同步（2026-10-22）
- 第五轮复审`2bd90c39d`针对`aed03128a`判P0=0/P1=0/P2=1：成功`restore_completed`/`query_output_persisted`已durable写入后，状态位没有同步更新；后续progress故障可误报task terminal missing。
- 红测在restore ledger terminal成功、progress append失败的场景下观察到重复terminal（`restore_completed`+`restore_failed`）。修复在成功terminal ledger append后立即置位；后续故障保留已有terminal并进入外层失败收敛。
- 验证：该失败注入1/1通过；完整评估器合同45/45、Ruff、Black、py_compile、CLI帮助和diff检查通过。
- 状态：最新修复尚未提交、待独立Phase 7复审；第五轮报告的P2未关闭结论只针对提交`aed03128a`，不预判新修复。
- 训练三臂last durable仍为9,472/14,528/14,528 episodes、attempt=`training`且无final；未确认外部会话，不重启、不覆盖。
- 方向仍为先提升自然中文多轮对话；评测门禁代码不等同模型能力证据，`can_promote=false`、N1-166冻结及W5b冻结保持。

### Phase 7 第六轮独立复审：提交 8e4c736e2（2026-10-22）
- 结论：不放行；P0=0、P1=0、P2=1。正常restore/query terminal耐久写入后立即置位；progress后续失败不会重复failed terminal。上一轮false→true writer竞态仍保持task身份和unknown尾账；P2-04及旧门禁无回归。
- 新发现：写入restore/query started ledger intent后，progress intent追加成功前，active_task才赋值；若progress追加失败，outer可能遗漏已打开ledger task的task id与unknown尾账。
- 产物：`artifacts/07-evaluation-review-8e4c736e2.md`。
- 验证：根任务完整45项合同、Ruff/Black/py_compile/CLI及提交门禁通过；本轮未重跑，不读取真实dev/reference、不restore模型、不运行selector/`--execute`。
- 回流判定：P2-06需在started intent ledger durable后立即建立active task身份，或为progress失败路径写唯一terminal；补restore/query intent progress失败注入测试。此前2/2历史预算和复审记录保持原样。
- 阶段指针：保持Phase 8；P7/W5a当前不放行，真实评测继续冻结。

### Phase 7 第七轮独立复审：提交 c7411a688（2026-10-22）
- 结论：通过代码审查门；P0=0、P1=0、P2=0。task-start身份在ledger intent写入前建立，intent耐久后再更新intent状态/query_attempts；progress失败时task identity与unknown尾账完整。成功terminal ledger append后同步置位且不重复写失败terminal；false→true writer竞态及P2-04/旧门禁无回归。
- 产物：`artifacts/07-evaluation-review-c7411a688.md`。
- 验证：根任务完整47项合同、Ruff/Black/py_compile/CLI/提交门禁通过；本审查另做synthetic restore_started ledger append失败复核，task identity/intent_unverified/unknown均符合预期；未读取真实dev/reference或运行真实评测。
- 放行边界：仅Phase 7评测器代码审查门通过；真实W5b仍等待Phase 8训练终点、readiness与独立恢复，未授权selector或`--execute`。
- 阶段指针：Phase 8顶部主指针不变；历史2/2预算及此前结论保留。

### Phase 8 中断后恢复核验（2026-10-22）
- Phase 7评测器代码门已通过（P0/P1/P2=0），依据报告`07-evaluation-review-c7411a688.md`；真实W5b依赖仍未满足。
- 三臂attempt保持`training`且无final；progress末次分别为9,472/14,528/14,528 episodes。六个`writer.lock`记录的root与pretraining PID均已核验不存在：full_turn_1m 35200、warm_start 26248、fresh_same_capacity 35196。
- 当前资源预检CPU双采样10.9%/9.6%、可用内存26.298 GiB；均满足冻结runner的启动门。
- 决策：按原attempt执行显式`--resume`，从latest已提交并带verification receipt的bundle继续，不新开attempt、不清除或覆盖checkpoint；stale writer lock由runner在死PID确认后安全归档。
- 约束：只恢复训练并监控durable commits；不读取dev/reference，不selector，不运行`--execute`。三臂final与fresh-process独立恢复前W5b仍冻结，能力状态未知、`can_promote=false`、N1-166冻结不变。

### Phase 8 source-bound resume拒绝（2026-10-22）
- Phase7代码门已通过；三臂PID从lock身份核验不存在、CPU与内存资源门通过。
- 三臂各一次`--resume`均在source hash校验处拒绝，原因是5个工作区源码版本偏离attempt冻结哈希。拒绝发生在writer lock reclaim/open_store之前，未改attempt、checkpoint、progress或锁；最后durable进度仍为9,472/14,528/14,528，三个final均缺。
- 精确冻结源码的Git对象已找到：`02e82d921`、`85bfc59c8`、`0669123d`。当前文件均有index中暂存的用户改动，保留不覆盖。
- 调整：以字节级备份+Git历史原文短时source overlay启动原attempt；全部arm通过hash/readiness/latest receipt门后恢复当前工作树字节。失败即停，不改attempt source hash，不用当前偏移源码恢复。
- W5b、dev/reference、selector/`--execute`仍冻结；N主线对话方向、`can_promote=false`和N1-166冻结不变。

### Phase 8 冻结CPU预算阻断（2026-10-22）
- 在exact-source overlay下，full_turn_1m通过source、latest bundle/fresh-process与资源门，随后按旧CPU限写`recovery_stopped`并因budget exhausted停止；无新checkpoint。warm_start/fresh_same_capacity仍被额外source hash差异拒绝，未进入open_store/训练。
- boot identity改变后tail_charge上界为full_turn_1m 764,220.402、warm_start 763,673.331、fresh_same_capacity 762,788.250 CPU秒；旧限分别240k/120k/120k，均超限。数值是安全上界，非实际新增训练CPU。
- 决策点：任何继续训练均需新版本continuation合同、budget与独立审查。推荐提案上限每臂1M、总3M，原partial/unknown tail/hash保持只读。未经用户明确选择，不改预算、不再启动；保留旧attempt/source及最新checkpoint。
- 方向仍是对话能力优先，但本节点先等预算决策；W5b、真实dev/reference、selector/`--execute`继续冻结。

### Phase 4 补充：R2 continuation extension v1 架构审查（2026-10-22）
- 决策：用户已批准每臂累计1,000,000、三臂聚合3,000,000 process-CPU秒；架构报告`artifacts/04-continuation-extension-architecture.md`通过进入实现，不代表训练放行。
- 合同：每臂从自己的last durable checkpoint续训；旧attempt/progress/result/checkpoint及parent source hash只读。新attempt/progress/result用sidecar；新checkpoint写到`continuation_v1`版本化根，以parent receipt SHA建立谱系。
- CPU：1M/臂含旧实耗与未知尾账，非额外额度；13:30审查快照tail为780,628.774/780,081.702/779,196.622秒，aggregate 2,339,907.098秒。该快照仅审查基线，执行前重算；boot mismatch期间每小时每臂最多再增长86,400秒。
- 来源：当前仍有7个不同源码文件的parent hash mismatch；已查看Git差异但不改写旧hash。extension须交source compatibility receipt，逐项记录parent/current SHA和审查报告；无法证明则fail closed。
- 给下游：实现versioned runner、fork checkpoint lineage、单调CPU与aggregate ledger、源码兼容证据和故障测试；独立code review通过及所有来源/恢复/资源/预算门满足后方可训练。W5b/真实dev/reference/selector/`--execute`继续冻结，`can_promote=false`、N1-166冻结。
- 主阶段指针：保持Phase 8；本补充不覆盖历史合同或变更原attempt状态。

### Phase 3 补充：R2 continuation extension v1 需求（2026-10-22）
- 结论：新增R2EXT-01..08与AC-R2EXT-01..08，覆盖累计预算、父checkpoint续接、旧证据只读、单调尾账、source兼容性、耐久账本、并发/资源与能力边界；与`artifacts/04-continuation-extension-architecture.md`一致。
- 产物：`artifacts/03-requirements.md`（追加R2 continuation extension v1小节）。
- 关键要点：每臂累计≤1,000,000、聚合≤3,000,000 process-CPU秒；各臂只从自身last durable checkpoint续训；sidecar和版本化checkpoint分支由parent attempt/receipt SHA锚定；未知尾账不得清零；来源差异不改写父hash。
- 风险：13:30预算快照随wall×24继续增长；任何启动须重算，且source、durability、fresh-process restore及资源门未通过前不恢复训练。
- 给下游：依架构合同实现extension runner/账本并由独立code reviewer复核；W5b/真实dev/reference/selector/`--execute`保持冻结，`can_promote=false`、N1-166冻结。
- 主阶段指针：保持Phase 8；本补充仅为需求门记录，不代表实现、训练或评测通过。

### Phase 6 补充：R2 continuation extension backend账本实现（2026-10-22）
- 结论：预算账本后端子项完成；不放行真实训练。
- 产物：`scripts/training/taiji_r2_dialogue_continuation_extension.py`、`tests/taiji_native/test_r2_dialogue_continuation_extension_contract.py`、`artifacts/06-implementation-backend.md`。
- 关键要点：全局合同只写一次，父三臂charge与tail状态必须完整；每个work unit先在跨进程锁内write-through预留CPU并返回token；settlement必须用同token原子入账实际CPU并释放reserve。`latest_charges()`把未结reserve按上界计入，resume可用`continuation_resumed`显式结算。
- 预算门：每臂1M与聚合3M同时检查`charges+reservations`；同臂只允许一个active reservation；重复、错token与未结reserve直接拒绝下一工作单元。编号JSON事件write-through追加，summary仅是可重建派生索引。
- 验证：合同测试8/8通过，含两个独立进程并发争用同一aggregate余量（仅一个reservation获批、序号唯一）；Ruff、format、py_compile通过。Durability adapter为合成注入，真实运行仍需现场确认本地固定NTFS。
- 风险/给下游：AI runner在任何parent restore前初始化全局合同；每个恢复/新进程启动先full-chain `latest_charges()`/`latest_reservations()`，resume时将未结reserve按上界结算；每个restore/episode用`guard`并以token调用`append_charge`结算。runner集成和独立代码审查通过前不启动训练。
- 主阶段指针：保持Phase 8；该后端子项不代表runner、fresh-process checkpoint恢复或训练放行。

### Phase 6 补充：R2 continuation extension AI runner实现（2026-10-22）
- 结论：versioned continuation runner与合成验证完成；真实恢复和训练仍未放行。
- 产物：`scripts/training/run_taiji_r2_daily_dialogue_continuation_extension.py`、`tests/taiji_native/test_r2_dialogue_continuation_extension_runner_contract.py`、`artifacts/06-implementation-ai-continuation-extension.md`。
- 关键要点：全局`--prepare-only`一次准备三臂；`--run/--resume`逐臂显式执行；旧证据保持只读，新checkpoint位于版本化root。每个bounded work unit使用后端token reservation；fresh-process restore按120秒deadline+2秒ledger settlement reserve保护。失败/超时计入host CPU delta，并在停止或resume时按实际charge与未结reservation上界较大值结算。
- 验证：runner与backend synthetic合同测试20/20通过；`py_compile`、Ruff check/format、Black check及CLI帮助验证通过。未加载/恢复真实checkpoint、未训练、未访问dev/reference/selector、未运行`--execute`。
- 风险/给下游：runner与ledger仍需独立代码审查；后续准入必须重算wall×24未知尾账并复核固定NTFS durability、源码兼容性、资源与预算。review与执行门通过前不启动真实续训；工程结果不代表能力提升，`can_promote=false`与N1-166冻结保持。
- 主阶段指针：保持Phase 8；该AI runner子项不代表训练或W5b放行。

### Phase 8 续训额度修订与实现审查回流（2026-10-22）
- 决策：用户批准将当前版本化续训窗口由此前每臂1,000,000/聚合3,000,000 CPU秒提高至累计每臂1,200,000/聚合3,600,000秒。旧实耗、未知尾账、准备/恢复/训练/失败开销全计入；不清零、不重置。各臂仍从自身父checkpoint续接。
- 只读账本：15:20附近父checkpoint保守累计上界为937,951.469/937,416.609/936,544.407秒，聚合2,811,912.485秒；新窗口剩余额度为262,048.531/262,583.391/263,455.593秒，聚合788,087.515秒。该值按wall×24动态增加，任何执行前重算。
- 复审回流：独立runner审查发现source SHA/Git provenance、启动与最终化崩溃恢复、部分prepare恢复、NTFS write-through覆盖、cap拒绝耐久记录、final restore deadline、端到端恢复与吞吐门等P1未清零。后端durable publisher正在接入runner；当前不运行prepare-only、parent restore或训练。
- 唯一下一步：关闭实现P1并由独立审查复核；全部通过后做NTFS写入prepare-only门，再按实时预算、资源和速率准入三臂并发续训。W5b/真实dev/reference/selector/`--execute`继续冻结，`can_promote=false`、N1-166冻结。
- 主阶段指针：Phase 8；本额度修订不改变自然中文多轮对话主线，不代表模型能力提升。

### Phase 6 补充：R2 extension来源/准备/最终恢复实现（2026-10-22 17:12）
- 预算：当前窗口每臂累计1,200,000、三臂合计3,600,000 process-CPU秒，旧实耗与冻结未知尾账全计入，不重置。
- 父checkpoint只读审计通过：full_turn_1m/warm_start/fresh_same_capacity保守累计上界为1,098,577.619/1,098,127.719/1,097,340.395秒，合计3,294,045.733秒；剩余额度101,422.381/101,872.281/102,659.605秒，聚合305,954.267秒。parent attempt、latest receipt/bundle、cursor与source lineage均通过核验。
- 来源证明：tracked源文件含Git commit/blob；每臂6个未跟踪`data/`工件要求当前SHA等于parent SHA并绑定parent attempt/receipt/bundle；Windows行尾只在Git clean-filter OID精确匹配时认定为规范化。来源/准备/父副本门定向11/11通过，final restore成功与timeout测试2/2通过；后两者为合成验证，未加载真实checkpoint。
- 剩余门：结果已发布但completion账本未闭合时的幂等恢复、预算触顶时耐久停止事件、吞吐/剩余工作量投影、端到端cursor/RNG/32-episode提交与中断续跑测试，以及独立代码审查仍未完成。当前尚未运行真实`prepare-only`、父checkpoint恢复或训练。
- 唯一下一步：关闭上述实现P1并交独立复审；通过后立即重采样预算/吞吐，执行固定NTFS `prepare-only` 耐久门，再按headroom并发续训。W5b、真实dev/reference、selector与`--execute`继续冻结，`can_promote=false`、N1-166冻结。

### Phase 8 预算上限修订与代码门复核准备（2026-10-22 18:02）
- 用户批准当前累计额度由1.2M/臂、3.6M总额提高至1.3M/臂、3.9M总额；旧实耗和未知尾账继续计入、不重置。
- 18:02只读尾账为1,170,919.559/1,170,469.115/1,169,679.911秒，聚合3,511,068.585秒；当前余额129,080.441/129,530.885/130,320.089秒，聚合388,931.415秒。
- source、durable publisher、prepare恢复、final restore deadline、终点幂等结算、预算stop reserve、滚动吞吐投影和32-episode/cursor/RNG synthetic续跑均已覆盖。账本+runner合成测试58项通过；Ruff、format/Black、py_compile、CLI和diff检查通过。
- 唯一下一步：以冻结SHA交独立code review；通过后立即运行真实目录的NTFS `prepare-only`，校验checkpoint父副本与durability，再按刷新预算和吞吐门并发恢复。训练/W5b未启动，N1-166冻结，`can_promote=false`。

### Phase 8 续训窗口预算修订（2026-10-22 17:50）
- 用户批准将当前累计上限从1.2M/臂、3.6M总额提高至1.3M/臂、3.9M总额；旧实耗与未知尾账继续计入、不重置。三臂仍各自从原durable checkpoint续训。
- 17:50只读父账本上界为1,155,726.983/1,155,263.650/1,154,476.359秒，聚合3,465,466.992秒；新窗口余额144,273.017/144,736.350/145,523.641秒，aggregate434,533.008秒。准备和训练启动前按冻结公式重算。
- 实现状态：来源、durable publisher、prepare恢复与final restore deadline已实现；结果完成事件恢复、预算拒绝停机、滚动吞吐投影、32 episode/cursor/RNG续跑等最终测试与独立复审仍在收敛。未运行真实prepare/restore/训练。
- 唯一下一步：完成剩余合成验证及独立审查，随后立即运行durable `prepare-only`并在预算内三臂并发续训；W5b与能力晋级仍冻结。

### Phase 7 补充：R2 continuation extension v1 最终独立代码审查（2026-10-22）
- 冻结runner/backend/两份合同测试SHA分别为58A96D63186681E7CDAAA1034F4E0A68EE14B20C22050BAD214CD12EEC3EF18F、5BC1787499BEE803A8E5310349672E40B038205CED35F55D28B0C7A7893C72D9、3B25E646E1F823C025FB51FF89DE0FB3DE930E396B8AB9FA8834176E025CD39A和09E67815752B23D9DB1797B1EE640056D931F80168A851954F78E4DA68488469。
- 结论：P0=0、P1=0、P2=0；独立合同测试72/72通过（33.60秒），py_compile、Ruff、Black、CLI帮助和目标文件diff-check通过。
- 复核覆盖锁优先/死owner恢复、reservation账本与1.45M/臂及4.2M aggregate、parent/source/cursor与copy-only lineage、bounded restore、terminal stop reserve、吞吐投影和finalization reconciliation。
- 主任务随后报告真实固定NTFS prepare-only exit 0，三臂status=prepared；准备后账本累计1,317,124.782691/1,316,636.232466/1,315,811.157573秒，aggregate 3,949,572.172730秒。未执行真实父checkpoint恢复或训练。
- 剩余训练投影186,719.457141秒，aggregate headroom约250,427.827270秒。19:41:41 CPU/内存样本早于prepare；各run仍须实时复核资源、预算与吞吐。
- 唯一下一步：各臂实时门通过后启动run入口，由根锁、reservation、bounded父checkpoint恢复与逐episode预算守卫把关；任一门失败即停止。主阶段指针保持Phase 8，W5b/能力晋级仍冻结。


## 19:44真实执行门失败与修复方向

NTFS prepare-only成功，三臂prepared；累计账本分别1,317,124.782691/1,316,636.232466/1,315,811.157573秒。准备后资源门CPU15.3/14.0%、RAM27.655GiB通过，aggregate余量250,427.827270秒，剩余训练预测186,719.457141秒。

三臂真实--run随后均exit1：base._validate_sources拒绝`full-turn dialogue source changed: scripts/data_prep/build_taiji_r2_daily_dialogue_full_turn_pretraining_corpus.py`。拒绝发生在root writer lock、start intent与权重更新之前；无真实续训进展。扩展source compatibility已通过，但旧base validator仍按原raw SHA检查已审查版本，合成测试未覆盖此实际衔接，保留本失败作为回归证据。

唯一下一步：实现严格来源证明下的extension专用训练数据校验，并建立显式版本化来源修订绑定；v1已prepared的合同、sidecar、父checkpoint和CPU账本不得改写或清零。新修订须红绿回归和独立审查，通过实际来源门后再续训。04架构报告已被v1绑定，保持原字节；新修订另记版本化补充。W5b、dev/reference、selector与晋级冻结。


## v1.1 来源修订实现进展（20:00）

采用独立架构复核推荐的append-only amendment：原v1合同、sidecar、progress、04报告及checkpoint副本保持原字节，同CPU账本连续累计；新增来源修订绑定原合同/attempt/readiness/receipt与新runner/validator/tests/补充报告。仅prepared、无执行事件、无pending reservation/写锁/terminal时允许发布。

extension专用validator保留12项来源与manifest/readiness/分母校验，仅builder允许精确审查兼容证明；新validator已对三臂真实只读载入与probe face通过，每臂3762预训练/525 assistant episodes。原72合同与新增9来源测试合计81通过（36.25s）；source-revision独立测试与Phase7复审进行中。没有发布amendment、没有恢复真实训练。

唯一下一步：修订合同回归及独立复审通过后，write-once发布v1.1 amendment，刷新同账本预算/资源门，再执行计费父恢复与续训；不重新prepare或重置旧费用。


## 20:38 真实开跑结果与预算决策

v1.1 amendment成功write-once发布；三臂`--run`都进入真实runtime，CPU/RAM资源门通过并写入start与`training_resource_preflight_passed`。随后统一被冻结的全任务吞吐门拒绝，原因`throughput_projection_exceeds_remaining_budget`。fresh restore次数0、durable training commit次数0，权重没有更新，reservation已全部settle。

20:38只读开跑尾账为full_turn_1m=1,328,948.048316、warm_start=1,328,462.169966、fresh_same_capacity=1,394,373.538816 CPU秒，合4,051,783.757098；剩余预算148,216.242902秒。滚动训练投影186,719.457141秒，超过现总上限28,503.214239秒。单臂最低ceiling投影为1,420,653.296396/1,375,970.695677/1,441,879.222166，fresh臂在1.45M下仅余8,120.778秒缓冲；预算保守尾账还随wall时长增长。

当前唯一决策点：如要在保留训练终点、15%吞吐安全系数和故障尾账的前提下继续三臂，需要用户将同一累计窗口修订至每臂1,500,000、aggregate4,300,000 process-CPU秒。按20:38快照，总投影后保留约61,496.786秒聚合缓冲，fresh臂保留约58,120.778秒臂内缓冲；所有旧charge和未知尾账继续计入、不重置。维持1.45M/4.2M则不应再次启动；将保留v1.1合同及全部失败/准备证据等待预算决策。


## 20:40 用户预算修订已批准

用户批准将同一累计续训窗口从每臂1.45M/aggregate4.2M提高至每臂1.5M/aggregate4.3M process-CPU秒。旧账、失败启动与保守未知尾账全部继续累计，不重置。唯一下一步是完成v1.2 append-only source/budget amendment并在4.3M窗口实时门通过后立即三臂并发续训；当前账本仍4,028,211.891916秒，未有checkpoint restore或training commit。

## v1.2真实恢复与boot identity缺陷（22:00附近）

用户批准的1.5M/臂、4.3M修订已成功发布，实际`--resume`已运行；三臂均无episode_completed或durable_training_commit。full/warm被吞吐投影拒绝，fresh在开跑前被累计臂上限拒绝。进程均返回exit1；两个遗留22秒reservation已按其保守上界全额结算并写入continuation_stopped，当前pending为空。最新durable charge为1,343,557.907691/1,455,709.397772/1,376,064.517384，总计4,175,331.822847秒。

已确认近期extension快照boot_identity有1792002779.5610702与1792002779.56107两种值，仅差约0.0000002秒。严格float相等把它们当成跨boot，触发wall×24未知尾账，使实际闲置等待抬高预算。parent boot值为1792002780.622641，与当前相差1.06157秒，不能把原parent未知尾账一并当作该微小抖动退款。

唯一下一步：修正extension boot身份的浮点稳定性，并对v1.2新窗口内已证实的误分类尾账形成独立、追加的审计校正方案。原历史event/contract保留；不得删除或清零原parent未知费用。已批准cap保持1.5M/4.3M，修复后先核验同boot host-CPU上界，再恢复训练。当前不能声称权重已有更新。


## extension尾账校正核算完成

已新增boot-accounting helper与4项定向回归。原prepared基线3,949,572.172730完整保留，读取当前全部host CPU得到新的可证明上界4,034,133.407105；现账4,175,331.822847。proposal已write-once保存在reports的boot_accounting_audit_v1，尚未应用credit。原parent boot差1.06秒，不纳入微秒级修正。下一步按04-continuation-boot-accounting-v1-3.md追加来源/会计修订与可复算校正event，然后在已批准预算内续训。


## 三臂真实训练与首个新增checkpoint已确认

boot审计校正已按v1.3追加到原账本，旧event及prepared基线完整保留。三臂实际恢复后均出现episode_completed与durable_training_commit。首个耐久点：full_turn_1m=9,504、warm_start=14,560、fresh_same_capacity=14,560；bundle SHA、receipt及独立恢复digest核对通过，previous_receipt仍链接各臂原parent。

存活exec句柄：full=22331、warm=80306、fresh=96103（本次已轮询确认仍运行）。下一步仅轮询这些现有句柄和新增durable progress，持续训练到30,096预训练+525 SFT终点；不重复启动。预算仍1.5M/臂、4.3M总额，W5b和晋级冻结。


### 训练运行状态修正

exec 80306（warm-start）已确认exit1，错误为剩余吞吐投影超过预算；最后耐久checkpoint为14,624 episodes（已新增96）。full-turn exec22331与fresh exec96103已再次轮询确认仍活跃，继续运行。warm无pending reservation；不据旧锁或最后progress推断其仍活跃，不重复启动另两臂。唯一下一步是在监测两条存活句柄的同时，核查warm本次投影拒绝及14,624 checkpoint恢复条件；预算仍1.5M/臂、4.3M。
