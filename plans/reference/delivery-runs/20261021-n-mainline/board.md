# N 主线交付黑板

- 主阶段：Phase 8 训练执行与监测。
- 唯一活动状态：[03当前执行](../../../active/roadmap/03_CURRENT_EXECUTION.md)。
- 活动入口：`python scripts/training/run_taiji_r2_resume.py`，只续接 warm_start。
- 当前累计额度：1.55M CPU秒/臂、4.4M 总额，旧账不重置。
- 2026-10-23 精简决策：删除活动启动链的多版本 manifest 嵌套、重复历史扫描、反复三臂预测否决和 CPU 采样否决，统一 warm 7GiB；保留 checkpoint 与硬预算。真实预检与20项测试通过。
- 最近训练事件 24,998，耐久点24,992；旧句柄失效，不作为存活证据。
- [精简产物与剩余事项](artifacts/15-training-system-simplification.md)。下方旧阶段表是历史记录，其运行快照以03为准。

## 阶段与交付状态

| 工作 | 状态 | 证据 / 未完成项 |
|---|---|---|
| 来源、保存/恢复、累计账与中断恢复 | 已实现并用于实际训练 | [活动合同](artifacts/02-continuation-extension-plan.md)及其技术证据；保留全部旧失败 |
| 当前三臂训练 | warm_start运行中，其余两臂停止 | 最后durable progress及receipt保留；无完整final，不判能力通过；账本锁、内存与full预算停机均已登记 |
| extension终点校验 | 实现及synthetic检查已过，真实终点待齐 | endpoint校验器8项、preflight入口5项；6份旧manifest/26项来源现场核验通过；warm fast-resume manifest已生成，终态来源复核待训练完成 |
| 完整评测接口与最终来源链复核 | 待完成 | 旧driver不直接支持extension schema；v1.3/恢复/预测适配器须在真实评价前复核 |
| W5b / 人工判读 | 冻结，等待前置 | parent+三臂、39×4=156 generation；固定人工样例，不增加题面 |
| 能力晋级 / 整模型验收 / 发布 | 未通过 | 按02/07，不以训练或工程检查替代 |
| 本轮文档收束 | 已整理并核验 | [收束记录](artifacts/12-plan-consolidation-20261022.md) |
| 实际续训launch来源验证与fast-ledger恢复 | 已实现、37项定向测试通过；warm实际恢复 | [内存守卫与快速恢复更新](artifacts/14-memory-gated-fast-resume.md)；fast-ledger只恢复获选单臂，终态来源复核待做 |

## 当前已采用的修订

1. v1.1固定来源证明与已审代码兼容，保存原合同和父谱系。
2. v1.2追加已批准1.5M/4.3M预算，不重置累计费用。
3. v1.3仅校正prepared之后可复算的boot浮点误判，原parent尾账与全部旧event保留。
4. receipt祖先链适配器允许从真实新checkpoint恢复；target-update投影保留1.15安全系数和全部硬预算门。

[续训活动摘要](artifacts/02-continuation-extension-plan.md)是技术导航；[增量工程计划](artifacts/02-plan.md)给出训练后交接顺序。当前真实通过证据与历史独立审查分开记账，不补写未发生的复核。

## 历史与回流

原447行阶段、审批、回流及监测记录保存在[黑板历史快照](../../../archive/history/N_MAINLINE_BOARD_BEFORE_CONSOLIDATION_20261022.md)。各07审查报告、04架构/修订、原始JSON和checkpoint都在原位置。历史额度、队首和“未启动”等陈述只解释当时时点。

### Phase4 extension评测适配增量架构完成
- 产物: artifacts/04-continuation-evaluation-adapter.md。架构通过，仅实现准入；旧worker硬编码metadata/路径必须与新provider贯通，禁止extension伪装restart_from_zero。保留parent+三arm、39×4、固定12与CPU/wall各14400、跨入口one-shot。当前训练主阶段不变，未读dev/题面或执行模型。
