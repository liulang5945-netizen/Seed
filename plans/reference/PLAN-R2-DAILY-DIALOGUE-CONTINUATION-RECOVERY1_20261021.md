# PLAN-R2-DIALOGUE-03R1：续训后自由生成的日志恢复

**状态：已按用户授予的计划批准权冻结并执行。** 这是 [PLAN-R2-DIALOGUE-03](PLAN-R2-DAILY-DIALOGUE-CONTINUATION_20261021.md) 的执行恢复，不改训练、数据或生成测面。

## 故障与保留证据

原续训两臂均完整完成：多轮与单轮各3,150条 episode、各679,254次响应字节观测；两枚 final checkpoint 均保存成功，并在独立进程恢复为 `global_step=906746`、`episode_count=4224`。原 partial 报告 SHA-256 为 `2a2e2e7a9c7d4993d4fdf56a1b6d5ad6b9e0f3a94b75f846f2b0c0f1f11086ea`，记录两臂训练完整及6次生成调用。

错误发生在第一棵未消费 final 对话树的6个端点生成之后、进度行写入之前。logger以模型键 `parent_dialogue_context` 索引样例行，但样例行使用 `parent_with_history`；二者名称不一致，触发 `KeyError`。因此这6个输出没有落盘。第一棵树固定为 episode `oasst2:25d50180-7062-4e4a-b1b6-69ea5b23668d`、family `oasst2-tree:18c35ba0-0db4-4bf9-b129-f8988cd888fd`。

## 最小修复与恢复约束

- 续训 runner 以显式端点到样例字段映射统一进度日志与汇总读数；新增回归测试覆盖6个端点，验证字段映射完整。
- 不再训练；保留并校验原始6枚父档、2-epoch终点和8-epoch终点。恢复前每枚都需通过独立进程完整 bundle 加载，确认参数、优化器、RNG/历史随检查点恢复。
- 在同一固定70棵树、相同6个端点与确定性解码配置下重新生成并持久化420条审阅记录。第一棵树的6个调用属于日志故障后的恢复重放；故物理调用总数为426，其中首次6条未保存、恢复后保存420条。报告不把丢失输出当作证据，也不声称两次输出逐字一致。
- 每次生成后立即写入 progress JSONL，再进入下一个端点；若后续发生错误，已完成输出仍可恢复。reference answer 不进入模型输入或计分；不查询 dev，不计算 exact match。
- 原批 CPU 上限仍为30,000秒，包含已失败执行消耗的5,187.078125秒；恢复预算不得超过剩余上限。不得再训练、改动数据划分、重抽样、增删测面或晋升。

## 根因回归验证

在修复前，新增测试因缺少 `_generation_progress_outputs` helper 而失败；修复后测试通过。runner 另通过 ruff 检查、格式检查和 `py_compile`。恢复执行前再次独立恢复两臂及父档端点，检查6枚 bundle 与输入哈希。

## 恢复产物

- 修复后的原续训 runner：`scripts/training/run_taiji_r2_daily_dialogue_continuation.py`
- 回归测试：`tests/taiji_native/test_r2_daily_dialogue_continuation_contract.py`
- 恢复 runner：`scripts/training/run_taiji_r2_daily_dialogue_continuation_recovery.py`
- readiness、attempt、progress、generations、review、result：`reports/taiji_r2_daily_dialogue_20261021_continuation_recovery1_*`
- 本计划只补全原计划的自由生成报告；能力晋升仍为 `can_promote=false`。
