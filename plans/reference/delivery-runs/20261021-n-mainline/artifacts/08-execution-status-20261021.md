# Phase 8 v2 三臂训练中途证据审计

## 结论

三臂仍处于固定全轮预训练阶段，当前记录继续增长，不能判定训练或端点评测完成。最近一次只读核验中，三臂均有新的 durable receipt；receipt 的 SHA 与 `latest.json` 一致，receipt 所指 bundle 的文件 SHA 也与 receipt 一致，且记录的恢复 digest 与 checkpoint digest 一致。三臂 attempt 均为 `training`、`can_promote=false`，尚无 final report。

本次仅审计本地状态文件与 receipt 元数据，没有读取 dev/reference 正文、查询留出 prompt、运行模型或干预训练。

## 当前证据

| Arm | Durable episodes / 30,096 | Target-byte updates | 累计 CPU 秒 | 最新 receipt | receipt / bundle 核验 |
|---|---:|---:|---:|---|---|
| `full_turn_1m` | 4,320 | 2,069,142 | 15,516.546875 | `commit-00004320-d45bdaab1114442dbe99509b65bf7b90.json` | latest SHA、receipt 文件 SHA、bundle SHA、恢复 digest 均匹配 |
| `warm_start` | 6,528 | 3,168,217 | 15,127.40625 | `commit-00006528-4e482399a0404380b66cb5dfd5c03c14.json` | latest SHA、receipt 文件 SHA、bundle SHA、恢复 digest 均匹配 |
| `fresh_same_capacity` | 6,528 | 3,168,217 | 15,131.671875 | `commit-00006528-c612957d7a694b649a67c94dd4aad202.json` | latest SHA、receipt 文件 SHA、bundle SHA、恢复 digest 均匹配 |

进度来源为 `reports/taiji_r2_daily_dialogue_recovery_v2_20261021_{arm}_progress.jsonl` 最后一行；当前各文件最后写入时间为 2026-10-21 23:32:07、23:31:42、23:31:50（Asia/Shanghai）。三个 v2 attempt 报告均记 `status=training`、`dev_queries=0`、`can_promote=false`。各臂 1320/240/240 秒单 episode reserve 与固定训练预算仍受原 v2 合同约束。

Receipt 指向的训练 bundle 位于 `data/checkpoints/r2_daily_dialogue_recovery_v2_20261021/{full_turn_1m,warm_start,fresh_same_capacity}/pretraining/`。上述文件哈希审计证明当前 `latest` 指针与写入文件一致；本审计没有重新启动独立恢复 worker，独立恢复结论仅按 receipt 已记录的验证 digest/CPU 字段引用。

## 验收门状态

- 已满足：v2 zero 和 assistant canary 的独立准备恢复；训练过程周期 durable commits；本次 latest pointer、receipt 与 bundle 元数据/哈希关系自洽。
- 未满足：三臂均到达 30,096 个全轮 episode、各完成 525 个 assistant 校准 episode 和约定更新分母；三臂 final report / final checkpoint 落盘；三个最终端点各自在新进程完成最终恢复。
- 后续门禁：仅在所有最终端点与恢复凭证齐备后，按单独版本化合同执行一次 39-family 同面生成与既定人工可读性审阅。此阶段不做 reference 答案评分，不以训练损失、格式统计或 checkpoint 存在推断对话能力或晋级。

## 阻塞与下一步

Phase 8 的中途执行状态为“继续中、端到端验收阻塞”，不是训练失败，也不是已完成。唯一下一步是继续监测同三臂 v2 固定训练与每 32 episode 的 durable commit；保留 dev/reference 面，直至固定分母、final 文件和最终独立恢复门全部通过。

## 遗留风险

- 中途 durable receipt 只保证当前可恢复检查点证据；训练仍可能在后续阶段失败，不能据此声称终点保存或语言能力改善。
- `reports/taiji_r2_daily_dialogue_recovery_v2_20261021_{full_turn_1m,warm_start,fresh_same_capacity}_final.json` 当前均不存在。
- 旧 06/07 attempt 与 readiness 为历史运行产物，不替代本次 v2 三臂的固定训练量或终点验证。

## 给下游

不要改写已绑定的训练源码、恢复合同、冻结计划或 checkpoint。继续只监测 v2 进度与 receipts；所有三臂 final 与最终恢复齐备后，再由独立评测合同门控一次保留面评测。
