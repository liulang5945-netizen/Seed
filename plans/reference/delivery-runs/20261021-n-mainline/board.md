# N 主线交付黑板

- 运行目录: E:\Seed\plans\reference\delivery-runs\20261021-n-mainline
- 当前阶段: Phase 8 测试/执行监控（三臂v2固定训练进行中；终点恢复前的留出评测仍阻断）
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

## 回流记录

### 回流 1 · Phase 7 评测审查 → Phase 6 AI实现
- 门禁：code-reviewer / 目标：ai-integration / 预算：2（已用1）/ 状态：进行中。
- 原因：P1共6项：restore-worker可绕attempt/预算；子进程直接脚本启动缺仓库import root；restore与generation无硬deadline；prompt selector前未复核source/result/readiness；attempt/ledger仅fsync文件，Windows目录项durability未证明。P2共3项：最终CPU/wall成本低报final写盘；岗位reserve计算错误；selector前缺耐久意图事件。
- 复检范围：全部P1/P2，新增负向/中断测试；不得访问dev/reference或运行模型/真实selector/`--execute`。
- 当前先决：P1-06已由architecture-reviewer补充合同并提交`75cba6ec5`；Phase 6 AI实现可开始六项P1/三项P2返工，Phase 8三臂训练指针维持不变。

## 风险与待办
- 训练中不读取dev/reference，不写训练checkpoint，不把格式统计当成语言能力。
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
