# R2 三臂续训活动摘要

## 2026-10-23 活动流程修订

以[03当前执行](../../active/roadmap/03_CURRENT_EXECUTION.md)为唯一活动状态。日常只使用 `python scripts/training/run_taiji_r2_resume.py`；先用 `--check-only` 验证。累计额度 1.55M CPU秒/臂、4.4M 总额，旧账不清零。warm 最新耐久点 24,992 episodes，最后训练事件 24,998。启动对账、资源和预算采用顺序已精简，见[精简记录](../../reference/delivery-runs/20261021-n-mainline/artifacts/15-training-system-simplification.md)。以下为修订前历史快照，其中旧额度、旧入口和“运行中”不再代表当前状态。


更新：2026-10-22。状态：**三臂实际训练中**。本页是活动执行摘要，原冻结contract、source报告和历史账本保持原路径；当前队首见[03](../../active/roadmap/03_CURRENT_EXECUTION.md)。

## 当前执行合同

| 项 | 当前有效内容 |
|---|---|
| 目标 | 原生中文多轮对话训练充分性与初始化/容量对照；尚无能力通过结论 |
| 臂 | full_turn_1m / warm_start / fresh_same_capacity，各用自己的父权重、游标与RNG |
| 原parent episodes | 9,472 / 14,528 / 14,528；旧partial与父谱系保留 |
| 当前恢复事实 | full17,019仍停，fresh25,104因内存守卫停；warm从24,915恢复至24,975并运行。所有charge/reservation保留，不重置尾账 |
| 单臂执行边界 | 当前只跑warm；预测只需该臂及当前总余量足够，step-level账本仍强制执行原1.5M/臂、4.3M总上限 |
| full后续判断 | 恢复后新增的保守系统CPU尾账使full预计还需80.8k、单臂余量约63.1k；本次暂缓，warm完成后据实复核 |

| 每臂终点 | 30,096 pretraining episodes、14,517,912 updates；525 SFT episodes、113,209 updates |
| 批准预算 | 累计每臂1,500,000、总计4,300,000 process-CPU秒 |
| 费用语义 | 旧费用保留；v1.3仅以同boot host-CPU证据追加校正误判，prepared基线3,949,572.172730完整保留，原parent尾账不核销 |
| 保存与恢复 | 本地固定NTFS耐久发布；每32episode/边界保存，独立恢复与receipt/bundle SHA验证 |
| 预测与硬门 | target-byte updates预测、1.15安全系数、worst probe和完成reserve；单臂/aggregate reserve及CPU双采样≤85%、RAM≥8GiB仍必过 |

## 当前使用的入口与证据

| 功能 | 实现 / 证据 |
|---|---|
| 基础extension训练、来源与账本 | [runner](../../../scripts/training/run_taiji_r2_daily_dialogue_continuation_extension.py)、[账本](../../../scripts/training/taiji_r2_dialogue_continuation_extension.py)、[原架构](../../reference/delivery-runs/20261021-n-mainline/artifacts/04-continuation-extension-architecture.md) |
| 来源兼容与版本化预算 | [v1.1](../../reference/delivery-runs/20261021-n-mainline/artifacts/04-continuation-source-revision-v1-1.md)、[v1.2](../../reference/delivery-runs/20261021-n-mainline/artifacts/04-continuation-source-revision-v1-2.md) |
| boot误判校正 | [v1.3](../../reference/delivery-runs/20261021-n-mainline/artifacts/04-continuation-boot-accounting-v1-3.md)、[审计JSON](../../../reports/taiji_r2_daily_dialogue_recovery_v2_20261021_boot_accounting_audit_v1.json) |
| 新checkpoint恢复 | [receipt祖先链适配器](../../../scripts/training/run_taiji_r2_checkpoint_resume_adapter.py)；4项检查及真实恢复 |
| 当前恢复/预测入口 | [target-update入口](../../../scripts/training/run_taiji_r2_target_update_resume.py)、[投影说明](../../reference/delivery-runs/20261021-n-mainline/artifacts/04-target-update-throughput-projection.md)；6项检查及真实训练 |
| 终点元数据与实际启动来源 | [endpoint validator](../../../scripts/training/validate_taiji_r2_continuation_endpoint.py)、[preflight](../../../scripts/training/prepare_taiji_r2_continuation_endpoints.py)、[launch validator](../../../scripts/training/validate_taiji_r2_continuation_launches.py)、[fast-ledger resume](../../../scripts/training/run_taiji_r2_fast_ledger_resume.py)、[入口说明](../../reference/delivery-runs/20261021-n-mainline/artifacts/04-extension-endpoint-metadata-entry.md)；6份既有launch manifest与26项来源现场核验通过，fast-ledger已用于warm单臂，真实终点待齐 |

恢复入口的launch manifest保存于reports/taiji_r2_target_update_resume_<arm>_*.json，绑定额外适配器源码、测试、v1.3 source revision与起始receipt。当前三臂使用该入口；代码提交6484dfbfa，当前运行期间不改其绑定源。

## 退出与后继

1. 三臂完整终点的episode/update、最终文件、fresh-process restore和来源一致。
2. result/readiness与唯一completion event按文件SHA互相绑定；无writer和pending reservation；CPU字段按发布前/完成事件下界解释，不冒称退出后精确实耗。
3. 按[实际launch来源核验记录](../../reference/delivery-runs/20261021-n-mainline/artifacts/13-continuation-launch-provenance.md)验证每份resume manifest和启动checkpoint；fast-ledger适配按已全审计锚点验证新增尾部链，终点manifest仍须完整回放账本。训练完成后收束extension评测合同与最终source链复核，再按冻结W5b流程生成与人工判读。现有旧driver不能直接拿来执行extension结果。
4. 只有真实能力证据满足对应门槛后讨论晋级与整模型交付；当前can_promote=false，N1-166仍冻结。

原预算审批、source拒绝、浮点误判、receipt与预测失败全过程见[整理前活动计划](R2_CONTINUATION_PLAN_BEFORE_CONSOLIDATION_20261022.md)，不删除旧事件或改写原合同。
