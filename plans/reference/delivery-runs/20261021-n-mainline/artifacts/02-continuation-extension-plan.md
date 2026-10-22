# Phase 2 补充：R2 三臂续训扩展 v1 计划

日期：2026-10-22（Asia/Shanghai）。状态：预算已获用户批准；续训合同与实现门仍未通过，当前不启动训练。

## 目标与边界

N 主线当前能力目标仍是自然中文多轮对话。此工作只把 R2 recovery v2 的三臂实验从各自已验证的 durable checkpoint 继续到原定训练终点，不代表对话能力已提升，也不解锁 W5b、dev/reference、selector 或 `--execute`。`can_promote=false` 与 N1-166 冻结保持不变。

三臂分别从自己的父 checkpoint 继续；不合并不同候选权重、不从零重训、不把已完成 episode 或 CPU 记账清零。

## 用户决策与预算语义

用户已批准版本化续训上限：每臂最多 **1,000,000 process-CPU 秒**，三臂累计最多 **3,000,000 process-CPU 秒**。这是一轮当前续训窗口的累计总上限，包含旧 attempt 已记录 CPU、冻结规则下的未知尾账和本次续训 CPU；不是在旧额度之上额外增加 1M/臂。恢复或再次中断不能重置上限。超过该上限的后续续训须重新决策。

旧 attempt、progress、已有 receipt、checkpoint bundle 与原冻结 source hash 均保留原样。扩展使用新的版本化 attempt/progress/result 记录；旧 checkpoint 树保持只读，新 checkpoint 写入独立版本化目录并在新 receipt/manifest 中明确链接父 receipt SHA。新全局账本须用原子串行化方式同时检查各臂 1M 上限和 3M 聚合上限。

## 当前只读账本快照

以下是 2026-10-22 13:30:50 Asia/Shanghai 的审查快照，执行前必须重算，不能直接当作启动账值。boot identity 已改变，冻结 `tail_charge` 使用检查点 ledger 的已记账值加 `elapsed wall × max(saved logical CPU, current logical CPU)`；本次取 24 核上界。

| 实验臂 | 父 attempt SHA256 | 最后 durable episodes / cursor | 最后 durable CPU 秒 | 父 latest receipt SHA256 | 保守尾账上界 | 1M 下剩余额度 |
|---|---|---:|---:|---|---:|---:|
| `full_turn_1m` | `c3557687753a0ae6b27baaa365175f73157e442db0ab3a6a4d725b8f056c9692` | 9,472 / epoch 2, index 1,948 | 34,465.78125 | `8ec115196c39f9936ea3704ac5aaa8934f8efeaf8794f58e709f285ceb3a23bb` | 780,628.774 | 219,371.226 |
| `warm_start` | `fff7ef55d72aef39778b8a781b13e7d18c1161253ceee21d7d3eb78aaf38f7e0` | 14,528 / epoch 3, index 3,242 | 34,039.859375 | `f8a2a72eec4d67ca362351158f4d11d19aee5bd044cec5b2ed7a399396b07583` | 780,081.702 | 219,918.298 |
| `fresh_same_capacity` | `db2b3e5f311fbcb35669da54107351033c7c527bea7e9e69a98fd49d744ed2d4` | 14,528 / epoch 3, index 3,242 | 34,066.109375 | `cdf91736fd6ea1c837ed075886c478b6df430f2494ecaf1163f81eefa6ad067e` | 779,196.622 | 220,803.378 |
| **聚合** |  |  | **102,571.750** |  | **2,339,907.098** | **660,092.902** |

`full_turn_1m` 曾记录 `recovery_stopped`，CPU 值为 748,972.986107；当前按父 checkpoint 重新计算的尾账更高，启动账本必须取二者及其他已记录费用的最大值。warm/fresh 仍无终点 result，三臂 attempt 都不能单凭 `status=training` 被视为仍在运行或完成。上一轮独立预算评审按训练吞吐估算剩余模型计算约 148,000 CPU 秒（full-turn 约 75,000、另两臂各约 36,500）；这是预测而非上界。冻结尾账规则每增加一小时会给每臂增加最多 86,400 CPU 秒，因此本快照不保证可以无限期等待后再启动。

## 源码绑定与准入

旧 checkpoint 的 parent bindings 不得改写。当前源码对旧 attempt 绑定存在差异：full-turn 有 5 个不同源码文件；warm/fresh 各有 7 项。已检视的训练和数据构建 diff 主要为格式重排及未使用行号变量改名；这不能等同于 hash 相同，也不能替代逐文件独立兼容性审查。新 extension runner 必须单独绑定自己的源码 SHA，并记录每项旧 SHA、当前 SHA、差异性质和审查结论；任何未解释差异、父 bundle/receipt校验失败或新进程恢复失败都应 fail closed。不得用当前 SHA 回写旧 attempt，也不得以进度日志替代 checkpoint。

## 实施与验收顺序

1. 实现版本化 extension runner、manifest、全局 CPU 账本及版本化 checkpoint 分支。初始化时先耐久写入父 attempt SHA、父 checkpoint/receipt SHA、原累计 CPU、当前未知尾账和 boot/CPU/wall 快照，再恢复或训练；每次后续 checkpoint 将累计 charge 与新尾账取最大，严格单调。
2. 用合成状态验证父 receipt 分叉、各臂 1M 与聚合 3M 守卫、并发额度竞争、旧文件不变、首 checkpoint 前中断、checkpoint 事务半途失败、恢复后不重算已提交 episodes。测试不得访问真实 dev/reference 或启动正式训练。
3. 对 extension runner 做独立代码审查；逐臂确认原 source binding/兼容性证据、parent bundle 与新进程恢复、stale lock owner、启动 CPU/内存和剩余预算。真实训练仅在所有门均通过后开始；临启动重新计算尾账，并且只在资源与实测速率允许时并发三臂。
4. 从各臂独立父 checkpoint 与游标继续，累计到原定 30,096 预训练 episodes 和 525 assistant-SFT episodes；完成后 fresh-process 恢复并生成版本化终点证据，再由评测门审查是否允许 W5b。

### 完成判据

- 旧 attempt/progress/result/checkpoint 的 SHA 与内容保持不变，新 extension 记录能回溯父 lineage。
- CPU 账本从旧成本与尾账连续累计；每臂不超过 1,000,000 秒，三臂合计不超过 3,000,000 秒；预算不足时停止且保留账目，不自动扩容或重试。
- 三臂都达到冻结 episode/byte 分母，新终点 receipt、readiness 与 fresh-process restore 通过；没有未闭合写入或未知尾账。
- 训练、恢复与静态门禁不被写成语言能力结论；W5b 仍需单独评测和人工审阅，能力晋级门保持关闭。

## 唯一下一步

按上述合同实现版本化 extension runner 和 CPU 账本，并通过合成故障测试；随后交独立 code-reviewer 复核。审查与逐臂来源/恢复门未通过前不恢复真实训练。
