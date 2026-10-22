# N 主线当前执行

更新：2026-10-22 23:27 Asia/Shanghai 实测快照；本页是唯一活动队首。运行状态优先于本文快照，以已确认的进程句柄、progress、receipt 和账本为准。

## 目标与方向

持续提高**态极（Taiji）原生认知模型**的能力与上限；Seed 提供产品与运行环境。当前先补自然中文多轮对话的训练与真实输出证据，再依据结果修订 N 主线。N1 结构实验已有成果保留，N1-166 仍不准入；不以局部实验完成代替整模型验收。

本阶段的高上限路径是：建立训练充分、来源明确、可恢复的原生模型基线，通过配对输出定位数据、学习目标、容量或结构的限制，再选择能解决根因的后继方案。当前三臂尚无完整终点，不先指定赢家或追加模型改造。

## 当前事实

| 臂 | 作用 | 最新耐久 episodes | 耐久 target-byte updates | 已确认存活句柄 |
|---|---|---:|---:|---:|
| full_turn_1m | fresh 1M-capacity 全轮预训练 | 10,432 | 5,066,333 | 45943 |
| warm_start | 旧 R2 权重热启动 | 16,000 | 7,699,000 | 45618 |
| fresh_same_capacity | 与 warm 相同结构/容量的随机初始化对照 | 16,192 | 7,805,988 | 39673 |

- 三臂均在真实训练，已有新增 checkpoint 与独立恢复 digest；均无完整 final result。句柄已轮询确认存活，不能仅凭旧 status/锁判断。
- 每臂终点：**30,096 pretraining episodes / 14,517,912 target-byte updates**，然后 **525 assistant-SFT episodes / 113,209 updates**。
- 用户已批准同一累计窗口：**1,500,000 CPU 秒/臂，4,300,000 CPU 秒总额**。23:27账本约4,055,990秒，余量244,010秒，剩余预测176,472秒；全部臂投影可行。费用与余量须实时重算。
- CPU双采样53.3%/52.2%，可用内存23.585GiB；后续资源仍由运行守卫检查。这些是执行健康证据，尚非对话能力结论。

## 已收束的决策与故障

| 问题 | 当前处理与证据边界 |
|---|---|
| 客户端更新中断、旧训练无 partial | 保留失败；当前从各臂已验证 durable parent 接续，不重置旧账 |
| 冻结 raw SHA 拒绝已审来源、HEAD前移使证明漂移 | extension 专用来源校验与固定 prepare commit 证明；旧源码/合同保留，版本化修订绑定新来源 |
| boot浮点抖动误判重启 | v1.3追加审计校正，仅处理 extension prepared 之后可复算的误判；prepared 基线3,949,572.172730秒完整保留，原parent不核销 |
| 新checkpoint被当成parent覆盖而拒绝恢复 | 以receipt祖先链验证最新checkpoint；bundle SHA与独立恢复仍必过 |
| 最近长episode拉高全程预测、反复停止 | 改按固定target-byte updates预测；保留1.15安全系数、worst probe、完成reserve与硬预算 |

对应[技术计划与证据](../../reference/delivery-runs/20261021-n-mainline/artifacts/02-continuation-extension-plan.md)；逐次失败和预算审批见[整理前快照](../../archive/history/N_MAINLINE_EXECUTION_BEFORE_CONSOLIDATION_20261022.md)。

## 修订后的推进顺序

| 顺序 | 工作与当前状态 | 完成证据 / 下一阶段条件 |
|---|---|---|
| 1（当前） | 监测三条现有训练句柄，完成原定三臂训练 | 全部冻结episode/update分母；final文件、receipt与fresh-process restore一致；来源、账本与恢复链闭合 |
| 2 | 收束extension终点元数据；校验器和入口已实现，尚无真实终点通过 | result/readiness/唯一completion事件互相绑定；parent与来源不漂移；无writer或pending reservation；纳入实际启动适配器manifest来源复核 |
| 3 | 完成extension配对评测入口与最终源码链复核 | 旧评测器不能直接接extension schema；需接好新终点和worker合同，并复核v1.3会计校正、恢复/预测适配器；当前只有各项合同与运行证据，不能补写为全链独立审查通过 |
| 4 | 前置全过后，按冻结W5b合同做一次配对生成与人工判读 | 固定parent+三臂，39 family×4端点=156次generation；原CPU/wall双预算14,400秒、固定12-family人工样例、参考回答隔离；不追加题面或看结果降线 |
| 5 | 依据完整结果给出唯一后继方向 | 区分格式/停止、贴题连贯、历史消费、初始化/容量效应和保持；说明失败根因与高上限方案。需要改变预算、题面或研究路线时再讨论 |
| 6 | 有能力证据后进入同bundle整模型验收 | 按[07整模型交付](07_MINI_MODEL_DELIVERY.md)和[02晋级规则](02_GATES_AND_CI.md)逐项证明；本次R2实验收官不等于项目完美收官 |

**当前唯一下一步：继续监测45943 / 45618 / 39673至原定训练终点。** 等待期间可准备不接触保留面的元数据/评测适配；不另开训练方向。

## 中断与决策规则

- 正常运行：保留当前进程与来源哈希，按checkpoint/资源/预测检查，不因例行状态读取重复启动。
- 确认退出：核对真实句柄、最后耐久receipt与祖先链；在实时预算、资源和来源门通过后，使用[当前target-update恢复入口](../../../scripts/training/run_taiji_r2_target_update_resume.py)接续同一arm。保留失败证据。
- 预算不足、来源/题面权限变化或结果需要改变研究路线：先给出实测缺口和可审阅方案，再进行必要决策；已批准额度不重复询问。
- W5b、dev/reference、selector和晋级当前仍冻结。测试、恢复、loss和训练步数都不能代替语言能力。

## 项目收官边界与入口

N1-165A已结案但训练充分性门仅2/3；N1-166确认面未消费。R2-DIALOGUE-03R1虽然停止标记改善，人工自然回复仍0/12连贯贴题，该负结果保留。此次真实训练/评测用来判断后继方向，不能提前宣称已得到强通用模型。

总体能力和退出账本仍见[01阶段地图](01_SCOPE_AND_PHASES.md)、[09 N主线设计](09_NEXT_MAINLINE_PLAN.md)、[根需求](../TAIJI_CORE_REQUIREMENTS.md)。知识、上下文、记忆消费、推理/规划、稳定运行等实际能力须在同一真实入口/bundle上验证；长期VISION不插队，也不因本阶段训练完成被整体标为已完成。

| 入口 | 用途 |
|---|---|
| [续训活动摘要](../../reference/delivery-runs/20261021-n-mainline/artifacts/02-continuation-extension-plan.md) | 运行方式、预算、来源与checkpoint规则 |
| [交付黑板](../../reference/delivery-runs/20261021-n-mainline/board.md) | 阶段状态、待办与历史证据索引 |
| [终点元数据入口](../../../scripts/training/prepare_taiji_r2_continuation_endpoints.py) | 当前只返回training；真实就绪前不生成完成证据 |
| [完整历史快照](../../archive/history/N_MAINLINE_EXECUTION_BEFORE_CONSOLIDATION_20261022.md) | 保留旧判断、失败、审批和原执行日志，历史“下一步”已失效 |
