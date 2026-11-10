# R027：09.5 主体工程与恢复门合同

> 文档状态：[合同] — 把[R024](SPEC-TAIJI-CORE_RESTORE_R024_20261031.md)主体bundle与[R025](SPEC-TAIJI-CORE_ACTIVITY_R025_20261031.md)活动循环按09.5清单合拢：统一主体生命序列上六类边界的保存与独立恢复。不授予任务训练、定价或自主性结论。调度只见[MAINLINE](../MAINLINE.md)。

日期2026-10-31。`taiji/living_core.py`字节不变；合拢落`native_activity.py`（env序列化与来源合法性过滤）与`state_migration.run_worker_job`（`activity_round`模式）。

## 1. 统一主体生命序列与八次边界保存

同一subject（"r027-subject"）沿一条生命序列推进，每类边界各保存一次bundle并由**spawn独立解释器进程**恢复续推，父端同续推逐位比较：

1. `after_interaction`：宿主注入一条消息并完成一轮interact后。
2. `no_prompt_activity`：空队列阶段两轮模型自选活动后。
3. `sleep_wait`：强制wait轮后（时钟推进、H不变）。
4. `first_learning`：首次声明学习更新（R024 update模式）后。
5. `before_birth`／6. `after_birth`：R023门控出生前（父）与出生后（子），出生后由worker带门值表续推活动轮。
7. `after_migration`：R023编码变更事务提交（revision+1）后。
8. `pending_outcome`：未结Outcome挂起（resolution幂等闸沿用R024）。

## 2. 活动环境入bundle（清单合拢）

bundle `extra`携带：`activity_env`（message_queue、store记录（含owner/scope/real/observed_at/version）、pending_outcomes、goal、now、last_event、prediction_error）、`policy_state`、`fee_total`、`probe_exposures`。worker `activity_round`模式：重建env与policy→执行一轮模型自选活动→journal追加→bundle回写。清单逐项核验：参数/拓扑（R/P形状与增长结构）、optimizer/eligibility、RNG、游标（journal_cursor＋活动seq）、记忆（store记录逐字节）、编码revision、预算（fee/probe不清零）、receipt（resolution幂等）。

## 3. 越权来源与query-only隔离

- **来源合法性**：活动env在每轮开始按owner/scope/observed_at≤now/real过滤store（非法记录剔除并journal落账），R021图内合法性过滤保持最后一道；注入foreign-owner记录的轮次必须被剔除且不得进入记忆读数。
- **query-only隔离**：只读探针（评估/查询）在**状态副本**上运行；subject的h/fast/coverage/prior/keys在探针前后逐位不变（hidden与cache不含于探针写入路径）。
- **分账**：journal每轮携带`chosen_by`（model/control）与触发量快照；宿主注入消息的轮次记为external_trigger=true，内生轮（空队列）为endogenous——外控触发与内生策略分账，不据接口签自主性。

## 4. 出口与交付

- 出口：`passed_r027_gates_engineering_only`／`failed_boundary`／`failed_legality`／`failed_isolation`／`budget_stop`／`engineering_failure`。通过仅表示：八次边界的保存/独立恢复逐位一致、清单逐项核验、来源过滤与query-only隔离、外控/内生根因分账在该合成序列上成立；**不签语义保持、自主性、任务能力**。
- 预算：全验证≤900 CPU秒/20墙钟分钟。
- 交付：本合同、`native_activity.py`扩展、`state_migration.py`扩展、`tests/taiji_native/test_subject_gates_r027.py`、`scripts/training/verify_taiji_subject_gates_r027.py`、`reports/taiji_core_r027_subject_gates_20261101.json`（逐边界结果、清单核验表、分账账、费用、源码SHA与R021–R026绑定核对；本行原写 `_20261031`，实际落盘日期为 2026-11-01，已按真实路径更正）。
