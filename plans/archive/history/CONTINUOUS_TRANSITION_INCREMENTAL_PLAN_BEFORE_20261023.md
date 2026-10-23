# 当前增量工程计划

## 2026-10-23 活动流程修订

以[03当前执行](../../active/roadmap/03_CURRENT_EXECUTION.md)为唯一活动状态。日常只使用 `python scripts/training/run_taiji_r2_resume.py`；先用 `--check-only` 验证。累计额度 1.55M CPU秒/臂、4.4M 总额，旧账不清零。warm 最新耐久点 24,992 episodes，最后训练事件 24,998。启动对账、资源和预算采用顺序已精简，见[精简记录](../../reference/delivery-runs/20261021-n-mainline/artifacts/15-training-system-simplification.md)。以下为修订前历史快照，其中旧额度、旧入口和“运行中”不再代表当前状态。


当前主阶段为三臂实际训练；唯一活动队首以[03](../../active/roadmap/03_CURRENT_EXECUTION.md)为准。已批准1.5M CPU秒/臂、4.3M总额，旧费用与版本化校正均在同一账本内解释。

## 推进与完成标准

| 顺序 | 交付 | 状态 / 出口 |
|---|---|---|
| 1 | 完成获选单臂训练，再决定是否续跑其他臂 | warm_start已从latest durable receipt恢复并运行；selected-arm预测保留硬预算/资源守卫。终点须达30,096预训练+525SFT及fresh restore |
| 2 | extension终点manifest | metadata validator、preflight与launch来源校验均已实现；三臂6份既有launch manifest与26项来源现场核验通过，真实结果和唯一completion事件尚缺 |
| 3 | extension配对评测接线与最终代码/来源复核 | 待完成；旧recovery-v1 schema/CPU上限不能冒充extension；全链最终复核不能由历史局部审查代替 |
| 4 | W5b一次配对生成与人工判读 | 前置齐备后按原39×4=156、14,400 CPU/wall秒和固定人工样例执行；当前仍冻结 |
| 5 | 实验结案与唯一后继方向 | 完整成本、来源、checkpoint、输出及负结果；按数据/目标/容量/结构根因选择高上限方案 |

## 风险与处置

- 运行跨会话：用真实句柄判活；确认退出后从验证过的latest接续，不把status/旧锁当存活证明。
- 预算/来源/题面变化：保存证据并按实测缺口决策；已批准内容不重复审批。
- 现有synthetic和工程通过只证明局部门，真实终点与语言能力尚待证据。

当前技术合同见[续训活动摘要](../../reference/delivery-runs/20261021-n-mainline/artifacts/02-continuation-extension-plan.md)，评测原合同见[04-v2架构](../../reference/delivery-runs/20261021-n-mainline/artifacts/04-v2-evaluation-architecture.md)与[03需求](../../reference/delivery-runs/20261021-n-mainline/artifacts/03-requirements.md)。早期06/07复用旧输出方案和后继v2方案都保留原文，但当前以v2四端点及后继extension适配为准。

原逐次追加计划见[增量计划历史](N_MAINLINE_INCREMENTAL_PLAN_BEFORE_CONSOLIDATION_20261022.md)。
