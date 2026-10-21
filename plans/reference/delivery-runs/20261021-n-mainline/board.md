# N 主线交付黑板

- 运行目录: E:\Seed\plans\reference\delivery-runs\20261021-n-mainline
- 当前阶段: Phase 5 实现（07独立评测器；真实执行阻断）
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

## 回流记录

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
- Phase 7最终P2审查通过: 启动双CPU采样/8GiB门及失败账本、final同目录fsync/write-once rename已落实；独立runner18/18通过，P0/P1仍0。可放行三arm真实zero准备验证；正式训练前仍核对预算reserve充分性。未真实prepare/execute/dev。
