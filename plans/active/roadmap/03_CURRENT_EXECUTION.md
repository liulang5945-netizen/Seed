# N主线当前执行

更新：2026-10-23，Asia/Shanghai。本页是唯一活动队首；[10](10_CONTINUOUS_CAPABILITY_EXECUTION_PLAN.md)负责详细WBS，[09](09_NEXT_MAINLINE_PLAN.md)负责机制继承，[07](07_MINI_MODEL_DELIVERY.md)负责能力/用户验收。

## 目标与阶段

用户已将持续Goal正式设为：沿N主线，把态极发展成由自身认知机制驱动、能通过自然中文可靠完成连续任务、跨会话积累和修正经验、从反馈中学会新能力并保持旧能力的原生认知模型；每轮以同一可恢复版本的未见表现、保持和实际成本验收。

当前CC-1执行中。整体衔接为CC-1训练/评价出口→CC-2归因→CC-3语言/L2→CC-4同bundle联合能力/L3→CC-5连续经验→CC-6学习与保持/L4→按证据选择CC-7扩面。L3所需有限状态可从CC-5.1提前抽取，隔离加载可在CC-3期间开发。

## 当前运行与事实

- 截至2026-10-23 16:06:55 Asia/Shanghai：warm进度26,111 episodes / 12,625,793 updates（事件63,515）；最近耐久26,080 / 12,612,602（事件63,453）。账本序号63,516，warm累计charges 1,544,058.061秒，含活动reserve后的单臂余量约5,689.9秒。
- 本轮已通过执行会话62394核验warm进程存活；之后以实时句柄/receipt/账本为准，旧锁或本页快照不能证明存活。
- full_turn_1m与fresh_same_capacity暂缓，原checkpoint和parent链保留；目前无完整三臂final、无新语言能力通过结论。
- 当前v1.4版本化累计上限1,550,000 CPU秒/臂、4,400,000总计，包含旧实耗、reserve和未知尾账。此数值是上轮按推进单臂要求落盘的修订，不追写为用户逐字批准该数额；本轮不改额度。
- 原终点每臂30,096预训练episodes / 14,517,912 updates，再525 SFT episodes / 113,209 updates。剩余费用与完整终点尚有缺口风险；运行中估计作告警，实际硬预算守卫继续，触顶不重启循环。
- endpoint准备器的v1.4激活是局部未提交改动，已有5项preflight本地检查证据；不等于新旧manifest、provider与worker已贯通。相关实现未纳入本次文档提交。

## 当前工作包与接续

| 工作 | 状态 | 出口 |
|---|---|---|
| CC-1.1 来源/launch/endpoint provider | 部分实现 | 新简化入口及历史V3、v1.4与旧修订可验证，四端点snapshot；合成拒绝漂移/重复completion/活writer |
| CC-1.2 独立extension evaluator/worker | 架构已明确，完整实现待做 | 父进程和受监督worker使用同provider、snapshot与授权路径；无伪zero或旧schema回退 |
| CC-1.3 合成与独立审查 | 旧局部门可继承，全链待做 | one-shot跨入口、39×4、固定12、双预算与失败树终止；对应提交CI与独立审查 |
| CC-1.4 当前训练 | warm运行；另两臂暂缓 | 完整final/readiness/唯一completion及独立恢复，或如实partial停止回执 |
| CC-1.5 预算/三臂完整性 | 待实际费用闭合 | D1一次性提交到原完整出口的缺口与推荐；不补额则封存partial，W5b继续阻断 |
| CC-1.6 原W5b | 前置未齐 | parent+三臂、39×4=156、固定前12-family人工、CPU/wall各14,400秒，一次调用并保存原始失败 |

## 当前唯一下一步：完成CC-1.1至CC-1.3的extension评测适配链

保留已确认的warm续训，先做新旧launch→endpoint/provider→受监督worker接线与合成独立审查；不消费保留题面。若训练先触顶，保留完整停止证据和耐久partial，适配工作可继续；按D1讨论完整预算与原配对问题，不降分母、不把partial称final。

N1-166及原确认面、dev/reference/selector、能力晋级继续按原门冻结。普通实现与受影响验证自动推进；只有实质改变预算/题面/冻结协议/路线时到10的对应D节点裁决，其余独立工作持续。

## 使用与中断规则

唯一现行入口为`python scripts/training/run_taiji_r2_resume.py`，`--check-only`是独立预检，不代表启动；本轮不重复启动同臂。句柄超时不等于退出，先复核同句柄；确认终止后才按有效预算、来源和checkpoint恢复。新训练前检查完整保存/独立恢复；不修改运行中已绑定源码、语料或谱系。

## 历史与计划去向

[17过渡矩阵](../../reference/delivery-runs/20261021-n-mainline/artifacts/17-plan-transition-and-reconciliation.md)登记旧计划的继承、合并、退役与后续准入；[本页过渡前全文](../../archive/history/CONTINUOUS_TRANSITION_CURRENT_EXECUTION_BEFORE_20261023.md)、[更早N历史](../../archive/history/N_MAINLINE_EXECUTION_BEFORE_CONSOLIDATION_20261022.md)保留旧判断/失败。历史0/12不套到未测新候选，训练步数不代替语言能力；整项目未完成，M阶段限定收束不补发L2/L3/L4。
