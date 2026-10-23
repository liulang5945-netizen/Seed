# N主线当前执行

更新：2026-10-23，Asia/Shanghai。本页是唯一活动队首；[10](10_CONTINUOUS_CAPABILITY_EXECUTION_PLAN.md)负责详细WBS，[09](09_NEXT_MAINLINE_PLAN.md)负责机制继承，[07](07_MINI_MODEL_DELIVERY.md)负责能力/用户验收。

## 目标与阶段

用户已将持续Goal正式设为：沿N主线，把态极发展成由自身认知机制驱动、能通过自然中文可靠完成连续任务、跨会话积累和修正经验、从反馈中学会新能力并保持旧能力的原生认知模型；每轮以同一可恢复版本的未见表现、保持和实际成本验收。

当前CC-1执行中。整体衔接为CC-1训练/评价出口→CC-2归因→CC-3语言/L2→CC-4同bundle联合能力/L3→CC-5连续经验→CC-6学习与保持/L4→按证据选择CC-7扩面。L3所需有限状态可从CC-5.1提前抽取，隔离加载可在CC-3期间开发。

## 当前运行与事实

- 2026-10-23 17:48:48 Asia/Shanghai：warm因剩余单臂额度不足下一工作单元及预留而停止（事件66,699）；执行会话62394已返回exit=1。最近耐久27,584 episodes / 13,304,596 updates（事件66,653），最后未耐久进度27,606 / 13,311,741；warm累计1,549,763.982 CPU秒，无活动reservation，保留partial。
- 清理前已核验62394存活，最终复核确认其预算出口停止；历史活跃快照不再代表当前状态。checkpoint、账本与运行源码不在本轮清理范围。
- full_turn_1m与fresh_same_capacity暂缓，原checkpoint和parent链保留；目前无完整三臂final、无新语言能力通过结论。
- 当前v1.4版本化累计上限1,550,000 CPU秒/臂、4,400,000总计，包含旧实耗、reserve和未知尾账。此数值是上轮按推进单臂要求落盘的修订，不追写为用户逐字批准该数额；本轮不改额度。
- 原终点每臂30,096预训练episodes / 14,517,912 updates，再525 SFT episodes / 113,209 updates。剩余费用与完整终点尚有缺口风险；运行中估计作告警，实际硬预算守卫继续，触顶不重启循环。
- endpoint准备器的v1.4激活（68ed3495c）与extension适配/worker/launch血缘（612c2dfe4／b1817cbcf／1fc8bac40）均已提交；真实预检仍因三臂未齐被拒绝，不等于新旧manifest、provider与worker已在真实终点贯通。

## 当前工作包与接续

| 工作 | 状态 | 出口 |
|---|---|---|
| CC-1.1 来源/launch/endpoint provider | v1.4 预算激活（68ed3495c）；完整血缘/provider 接线（1fc8bac40）；真实预检待三臂齐备 | 新简化入口及历史V3、v1.4与旧修订可验证，四端点snapshot；合成拒绝漂移/重复completion/活writer |
| CC-1.2 独立extension evaluator/worker | 适配层＋versioned worker 已提交（612c2dfe4／b1817cbcf）；真实预检待三臂齐备 | 父进程和受监督worker使用同provider、snapshot与授权路径；无伪zero或旧schema回退 |
| CC-1.3 合成与独立审查 | 独立审查完成，修正三项（d39f6c8af）＋launch/ledger/budget接线（1fc8bac40）；81支合成契约通过；extension 评测合同冻结 | one-shot跨入口、39×4、固定12、双预算与失败树终止；对应提交CI与独立审查 |
| CC-1.4 当前训练 | warm已按预算保护停止；另两臂暂缓 | 耐久partial与停止事件保留，无完整final；D1准备到原完整实验出口的费用/范围裁决 |
| CC-1.5 预算/三臂完整性 | D1 材料已备（18-d1-budget-decision.md）；推荐 v1.5 补额 1,590,000／4,560,000；等 owner 裁决 | D1一次性提交到原完整出口的缺口与推荐；不补额则封存partial，W5b继续阻断 |
| CC-1.6 原W5b | 前置未齐 | parent+三臂、39×4=156、固定前12-family人工、CPU/wall各14,400秒，一次调用并保存原始失败 |

## 当前唯一下一步：D1 裁决——批准 v1.5 补额并授权三臂逐臂续训

D1 材料见[18-d1-budget-decision.md](../../reference/delivery-runs/20261021-n-mainline/artifacts/18-d1-budget-decision.md)。按耐久游标：full 16,992 episodes／8,176,295 updates、warm 27,584／13,304,596、fresh 25,088／12,091,199；三臂合计已计 4,369,799 CPU 秒，v1.4 总计余量仅 30,201。按最近边际速率与 1.15 安全系数，补完剩余预训练＋SFT 预计需 118,303 CPU 秒：full 81,023、warm 11,396、fresh 25,884。单臂只有 warm 缺 11,160；总计缺口 88,102。

**唯一推荐**：批准 v1.5 把每臂上限提到 1,590,000、总计提到 4,560,000（+40,000／+160,000），留总计约 7.2 万、warm 单臂约 2.9 万缓冲；按原终点完成三臂 final 后再进 W5b。备选是封存 partial 转 CC-2，但只能得到“训练未完成”混淆的归因，不能产出四端配对证据。裁决前不启动训练、不改冻结协议、不把 partial 称 final；fresh 恢复前必须确认可用内存 ≥8 GiB。

CC-1.3 已收口：独立审查发现并修正三项——评测源码未进 `source_hashes`、worker 未固定 torch 线程、restore 收据未核对阶段/digest；审查中又发现 CC-1.1 的 launch 血缘未接，已把 `launch_provenance`、账本 tip/charges 与批准额度绑定进 extension 快照，并让端点清单与 worker 恢复收据携带端点类型/血缘快照。旧 recovery-v2 评测 48 支＋extension 19 支＋生产→适配衔接 1 支＋既有预检/launch 套件＝ **81 支通过**；extension 评测合同冻结。残余：旧冻结入口无法检查 extension 前缀（不改冻结源），运行纪律为 extension 已跑则不再跑旧入口，extension 侧守卫已挡旧入口消费；真实预检仍被三臂未齐挡住。

N1-166及原确认面、dev/reference/selector、能力晋级继续按原门冻结。普通实现与受影响验证自动推进；只有实质改变预算/题面/冻结协议/路线时到10的对应D节点裁决，其余独立工作持续。

## 使用与中断规则

唯一现行入口为`python scripts/training/run_taiji_r2_resume.py`，`--check-only`是独立预检，不代表启动；本轮不重复启动同臂；当前因预算出口停止，不能按旧运行快照重启。句柄超时不等于退出，先复核同句柄；确认终止后才按有效预算、来源和checkpoint恢复。新训练前检查完整保存/独立恢复；不修改运行中已绑定源码、语料或谱系。

## 历史与计划去向

[17过渡矩阵](../../reference/delivery-runs/20261021-n-mainline/artifacts/17-plan-transition-and-reconciliation.md)登记旧计划的继承、合并、退役与后续准入；[历史全文恢复](../../reference/PROJECT_HISTORY_CONSOLIDATION_20261023.md)、[更早N历史](../../archive/history/N_MAINLINE_EXECUTION_BEFORE_CONSOLIDATION_20261022.md)保留旧判断/失败。历史0/12不套到未测新候选，训练步数不代替语言能力；整项目未完成，M阶段限定收束不补发L2/L3/L4。
