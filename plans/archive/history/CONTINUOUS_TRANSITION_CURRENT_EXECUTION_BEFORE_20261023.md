# N 主线当前执行

更新：2026-10-23 Asia/Shanghai。实时状态以 progress、耐久 receipt、账本和有效进程为准。

## 目标与当前阶段

先提高态极（Taiji）原生认知模型的自然中文多轮对话能力，完成真实训练与输出证据，再按结果修订 N 主线。当前只推进 warm_start；full_turn_1m 与 fresh_same_capacity 暂缓。局部训练完成不代表整模型验收或能力晋级。

2026-10-23项目复核提出一个持续目标建议：**让同一态极模型版本可靠完成自然中文连续任务，跨会话积累并修正经验，学习新能力且保持旧能力。** 沿用07的L2可对话→L3同bundle代表能力→L4持续适应，不把全部长期机制前置到当前训练。完整依据与改进优先级见[项目评审](../../reference/delivery-runs/20261021-n-mainline/artifacts/16-project-review-and-continuous-objective.md)；该表述为建议，尚不代替根需求或已冻结合同。

## 当前事实

- **2026-10-23 15:24 Asia/Shanghai 实测：精简入口训练事件持续推进。** episode事件62,009已到25,402 / 12,273,366 updates；最近耐久点25,376 / 12,263,243（事件61,957）。账本62,010记录warm charges约1,541,629.857秒、reserve252秒，距单臂上限的有效余量约8.1k秒；与完成全终点所需费用仍有缺口风险，详见项目评审。以下24,992耐久点和61,133账本是启动前历史快照。

- warm_start 上次进程确实新增了训练事件，最后记录为 24,998 episodes / 12,038,798 updates；最新耐久 checkpoint 为 24,992 episodes / 12,035,537 updates。更早的 25,010 是旧进程未耐久进度，恢复时回退到耐久点，旧 CPU 仍计费。
- full_turn_1m 与 fresh_same_capacity 保留原 checkpoint；无完整三臂 final。
- 当前版本化累计上限为 1,550,000 CPU 秒/臂、4,400,000 总计；v1.4 sidecar 已存在，旧实耗和未知尾账不重置。它是上轮根据用户要求推进单臂所作的修订，不追写为用户逐字批准该数额。
- 整理时账本序号 61,133：full 1,436,869.033、warm 1,514,385.201、fresh 1,383,166.236 CPU 秒，无活动 reservation。启动时仍须计入更新后的尾账。

## 唯一活动入口

```powershell
python scripts/training/run_taiji_r2_resume.py --check-only
python scripts/training/run_taiji_r2_resume.py
```

真实预检通过后可直接运行第二条。固定 warm_start，禁止重复启动同臂。旧 target-update、fast-ledger V1/V2/V3 退出日常操作路径，仅保留已有 launch/checkpoint 的来源证据。

精简内容、实测故障与保护边界见[训练系统精简](../../reference/delivery-runs/20261021-n-mainline/artifacts/15-training-system-simplification.md)。checkpoint、来源、NTFS 耐久写、同臂写锁、真实 CPU 硬上限与停止 reserve 继续生效。

## 推进顺序

1. 使用单入口从最新耐久 warm checkpoint 续接至 30,096 pretraining episodes / 14,517,912 updates，再完成 525 SFT episodes / 113,209 updates。
2. 训练期间并行补齐extension endpoint/provider、受监督评测worker与新旧launch manifest核验；正式评价仍等待完整final、receipt与独立恢复。新简化manifest尚须接入最终endpoint审计；不得绕过该项执行评测。
3. 根据实际费用与 warm 结果修订其他两臂计划；原四端点配对合同没有因此变成单臂能力验收。
4. 完成 extension 评测接线后执行冻结 W5b，再按真实输出给出唯一后继方案。

当前唯一下一步：在保留warm_start续训的同时补齐extension评测适配链，让训练产物能够形成可判定的能力证据；到顶未完成则按partial如实结案。N1-166、dev/reference、selector、晋级仍冻结。

## 历史与整体边界

历史失败、预算与决策见[历史快照](N_MAINLINE_EXECUTION_BEFORE_CONSOLIDATION_20261022.md)和[交付黑板](../../reference/delivery-runs/20261021-n-mainline/board.md)。R2 旧自然回复 0/12 的负结果保留；训练步数和预检通过不能替代语言能力。

整体要求继续遵循[根需求](../../active/TAIJI_CORE_REQUIREMENTS.md)、[阶段地图](../../active/roadmap/01_SCOPE_AND_PHASES.md)、[晋级规则](../../active/roadmap/02_GATES_AND_CI.md)、[整模型交付](../../active/roadmap/07_MINI_MODEL_DELIVERY.md)与[N 主线设计](../../active/roadmap/09_NEXT_MAINLINE_PLAN.md)。
