# N 主线当前执行

更新：2026-10-23 Asia/Shanghai。实时状态以 progress、耐久 receipt、账本和有效进程为准。

## 目标与当前阶段

先提高态极（Taiji）原生认知模型的自然中文多轮对话能力，完成真实训练与输出证据，再按结果修订 N 主线。当前只推进 warm_start；full_turn_1m 与 fresh_same_capacity 暂缓。局部训练完成不代表整模型验收或能力晋级。

## 当前事实

- **2026-10-23 15:00 Asia/Shanghai 实测：精简入口已实际恢复训练。** resumed 账本事件61,135；新耐久 checkpoint `commit-00025024-da08a9ffc8fb450ca993a20e8002ac5e.pt`（durable事件61,209）；随后episode事件61,221推进到25,030 / 12,052,192 updates。运行会话62394仍活跃。以下上一轮24,992耐久点和61,133账本是启动前快照，不再代表最新进度。

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
2. 验证 final、receipt 与独立恢复，收束终点元数据及新旧 launch 来源核验。新简化 manifest 尚须接入最终 endpoint 审计；不得绕过该项执行评测。
3. 根据实际费用与 warm 结果修订其他两臂计划；原四端点配对合同没有因此变成单臂能力验收。
4. 完成 extension 评测接线后执行冻结 W5b，再按真实输出给出唯一后继方案。

当前唯一下一步：通过单入口续训 warm_start，并观察新的耐久 checkpoint。N1-166、dev/reference、selector、晋级仍冻结。

## 历史与整体边界

历史失败、预算与决策见[历史快照](../../archive/history/N_MAINLINE_EXECUTION_BEFORE_CONSOLIDATION_20261022.md)和[交付黑板](../../reference/delivery-runs/20261021-n-mainline/board.md)。R2 旧自然回复 0/12 的负结果保留；训练步数和预检通过不能替代语言能力。

整体要求继续遵循[根需求](../TAIJI_CORE_REQUIREMENTS.md)、[阶段地图](01_SCOPE_AND_PHASES.md)、[晋级规则](02_GATES_AND_CI.md)、[整模型交付](07_MINI_MODEL_DELIVERY.md)与[N 主线设计](09_NEXT_MAINLINE_PLAN.md)。
