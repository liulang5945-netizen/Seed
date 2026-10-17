# EXP-N1-97 · 2026-10-17 · 已消费 v5／v6 面上的 unknown 输出普查

本件执行 [HANDOVER-N1-AFTER-V6 §2 甲](HANDOVER_N1_AFTER_V6_20261017.md) 的首步：只读两枚已出版报告的逐题行，检查 `kind=unknown` 时模型是否逐字输出 `unknown`。这是描述性普查，不是新能力测试，也不是新的预注册判据；本轮新增模型查询、checkpoint 加载、参数更新、题面生成均为 0。

## 1. 输入与绑定

- v6：`reports/taiji_n1_transition_factorial_confirmation_v6.json`，`status=published`、180 行、既有 `confirmation_queries=180`、`retry_allowed=false`；本轮读取的文件 SHA-256：`3d924e19d97b59ea12505111603b7d562ada8e8d9bdac71351a51c24c5918849`。
- run89／v5：`reports/taiji_n1_transition_factorial_confirmation_run89.json`，`status=complete`、288 行、既有 `confirmation_queries=288`、`confirmation_face_consumed=true`；本轮读取的文件 SHA-256：`3b4fc7f3fb387de8a615f23c6eeb2657e205508614de9629e972865ae5e61015`。
- 两报告的 9 枚正式终端 checkpoint 路径和逐文件原始 SHA-256 相同；逐项核对依据为 v6 的 `guards[*].sha_raw` 与 run89 的 `source_guards[*]`。九枚端点的跨报告哈希不一致数为 0。run89 的绑定协议见 [PLAN-N1-90 §1](PLAN-N1-90_three_family_binding_v5_confirmation_20261017.md)。
- 过滤条件：`row.kind == "unknown"`。逐行把 `output_hex` 解码为字节，只有字节串恰为 `b"unknown"` 才计入分子；该分子也须与 `free_exact=true` 一致。没有把不同题面或不同臂合并成能力分数。

## 2. 逐臂结果

| 已消费题面 | position | terminal | cut |
| --- | ---: | ---: | ---: |
| v6（每臂 15 行） | **0/15** | 4/15 | 2/15 |
| v5（每臂 24 行） | **23/24** | 9/24 | 24/24 |

按 seed 展开：

| 题面 | seed | position | terminal | cut |
| --- | ---: | ---: | ---: | ---: |
| v6 | 11（每臂 5） | 0/5 | 1/5 | 1/5 |
| v6 | 12（每臂 5） | 0/5 | 2/5 | 1/5 |
| v6 | 13（每臂 5） | 0/5 | 1/5 | 0/5 |
| v5 | 11（每臂 8） | 7/8 | 4/8 | 8/8 |
| v5 | 12（每臂 8） | 8/8 | 3/8 | 8/8 |
| v5 | 13（每臂 8） | 8/8 | 2/8 | 8/8 |

## 3. 判读边界

“position 在所有题面上都不会弃权”被这次交叉普查否证：同一组 position 终端在 v5 上为 23/24，在 v6 上为 0/15。这个变化与具体题面相关，但本件没有识别导致变化的题面特征；两个确认面不同、样本量也不同，不能据此判定哪一面更难、宣称新机制或挑选训练配方。v6 上的 0/15 仍是该面上的实在失败，v5 上的 23/24 也不能抹去它。

这组读数不支持直接把“position 缺少 unknown 行为”固化成产品规则或立刻改权重。建议的唯一下一步是对两份**已经消费**的 v5/v6 清单做静态交叉审计，只分析 unknown 题的提问、实体名、观测候选与顺序等输入结构，并与现有报告行连接；不加载模型、不生成或查询新题、不训练、不改判据。该步从交接页限定的“只读两份报告”扩到“读取其原始清单”，因此单独提交范围供 owner 裁定后再执行。

N1 的 C3 仍未达成，v5 与 v6 均不可重测；本件不改变 `can_promote=false`，也不提供默认模型替换、保持集、C4 净增益或三器官证据。
