# N 主线交付黑板

- 主阶段：**Phase 8 训练执行与监测**。
- 当前队首：[03当前执行](../../../active/roadmap/03_CURRENT_EXECUTION.md)。
- 当前唯一动作：待内存稳定通过守卫后单臂恢复，资源稳定后续完三臂至终点。
- 预算：同一累计窗口1.5M CPU秒/臂、4.3M总额；历史费用按账本与v1.3审计校正解释。
- 10:13耐久进度：full17,019 / warm24,915 / fresh25,104 episodes；原句柄均失效。charges 4,150,675.782秒、余量149,324.218秒；target-update预测136,656.835秒，无活动reservation。
- 恢复情况：并发resume曾触发共享ledger锁超时；warm/fresh恢复后因内存低于8GiB停止。11:07三次内存采样7.663/7.859/7.839GiB，fast-ledger入口已实现和测试，尚未实跑。

## 阶段与交付状态

| 工作 | 状态 | 证据 / 未完成项 |
|---|---|---|
| 来源、保存/恢复、累计账与中断恢复 | 已实现并用于实际训练 | [活动合同](artifacts/02-continuation-extension-plan.md)及其技术证据；保留全部旧失败 |
| 当前三臂训练 | 资源门待恢复 | 最后durable progress及receipt保留；无完整final，不判能力通过；账本锁超时与内存停机均已登记 |
| extension终点校验 | 实现及synthetic检查已过，真实终点待齐 | endpoint校验器8项、preflight入口5项；launch provenance及fast-ledger新增覆盖；三臂6份既有manifest与26项来源现场核验通过 |
| 完整评测接口与最终来源链复核 | 待完成 | 旧driver不直接支持extension schema；v1.3/恢复/预测适配器须在真实评价前复核 |
| W5b / 人工判读 | 冻结，等待前置 | parent+三臂、39×4=156 generation；固定人工样例，不增加题面 |
| 能力晋级 / 整模型验收 / 发布 | 未通过 | 按02/07，不以训练或工程检查替代 |
| 本轮文档收束 | 已整理并核验 | [收束记录](artifacts/12-plan-consolidation-20261022.md) |
| 实际续训launch来源验证与fast-ledger恢复 | 已实现、35项定向测试通过 | [内存守卫与快速恢复更新](artifacts/14-memory-gated-fast-resume.md)；fast-ledger入口待资源门通过后实跑 |

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
