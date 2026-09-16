# M5 R2-H3.5-A 分层回答计划候选预注册

> 状态：实现前冻结。本文只授权隔离 candidate 与 plumbing smoke，不授权正式训练、默认入口、L2/Mini 或 shared fabric 修改。
>
> 上游合同：[H3.5 分层回答计划与渲染合同](M5_R2_H3_5_HIERARCHICAL_RESPONSE_PLAN_CONTRACT_20260916.md)

## 1. 唯一假设

在相同父 checkpoint、300k 总 active 参数、数据顺序和更新预算下，从 assistant boundary 的 native prefix state 预测一个回答期间持续存在的 `response_plan_state`，并让 byte renderer 显式读取该状态，会比现有 response-phase owner 更好地迁移未见条件的内容 continuation；收益应在 plan 消融后消失。

## 2. 冻结实现合同

- serialization：`r2-h3-5a-hierarchical-response-v1`；旧 v2 checkpoint 不得静默解释为 v3 candidate。
- plan width：32；首版固定，不搜索。
- plan target：把 reference response 去掉 `\n<|end|>\n` 后按 UTF-8 codepoint 切分为最多 8 个连续 span；每个 span 经带版本盐的确定性 signed-hash 投影到 32 维，取均值后 L2 normalize。它只是训练监督坐标，运行时不存在 target encoder。
- planner：从 assistant boundary 的 `motor_context` 产生 32 维有界状态；首版仅 candidate 参数可写。
- renderer：读取 `[motor_context, response_plan_state]` 的隔离 byte readout；UTF-8 约束与 end-marker 停止器沿用现有实现。
- lifecycle：每个 episode 在 assistant boundary 创建一次 plan；reset/new episode 清空；teacher forcing 与 free generation 使用同一 owner。
- ownership：planner、plan-conditioned renderer 与其配置进入 child checkpoint；parent、protected readout、response-start 和 response-phase 不可写且与本 candidate 互斥。

具体字段、预算和门见 `plans/reference/contracts/r2_h3_5a_response_plan_v1.json`。实现若无法满足该合同，先修合同或停止，不得边跑边改口径。

## 3. 数据冻结规则

新 fixture 必须同时覆盖：共享首 codepoint/不同内容、不同表面/同一意图、四种 policy、history 改写、context 置换。train/dev/final 的 family 与完整 response 互斥；已曝光 H3.3/H3.4 fixture 只作 regression。final 在 smoke 阶段只做 schema 与恢复检查，不读取能力分；正式能力读取需另行授权。

所有样本保存 `episode_id/family_id/split`、运行时输入字段和只读评分字段。`task_family`、split、required/forbidden terms、unknown markers 和 reference response 不得进入运行时 planner/renderer。

## 4. 两臂与保存恢复前置

| 项 | Control | Treatment |
|---|---|---|
| parent | 同一 digest | 同一 digest |
| owner | response-phase | response-plan + plan-conditioned renderer |
| 总 active 参数 | ≤300,000 | ≤300,000 |
| 数据/顺序/seed/更新数 | 相同 | 相同 |
| UTF-8/end-marker | 现有路径 | 同一路径 |

任何训练前必须自动通过：zero save、fresh-process restore、一次 target update、child atomic save、child restore、parent digest 不变、恢复后只读重复、candidate clear 后 protected 路径恢复。checkpoint 不完整时禁止训练。

## 5. Smoke 出口

Smoke 只验证因果链，不声称能力：

1. v3 schema/digest 对错版本 fail closed；
2. plan state 由两个不同 prefix 产生可重复的不同 digest；
3. teacher forcing 只更新 candidate，parent digest 不变；
4. 保存恢复前后 plan、renderer 与生成结果一致；
5. plan 消融能切回预注册 control/ablated 路径；
6. 参数报告在 tensor 分配前给出 core、planner、renderer、effective 和余量。

全部通过后，唯一下一步才是冻结小规模 control/treatment 训练命令并执行；任一失败则停在 plumbing 修复，不增加 epoch。

## 6. 训练门与停止规则

训练阶段仍需另行记录逐 split 的 plan target rank/margin、continuation rank/probability、原始生成、collision、exact、sequence、policy、retention 和资源。Treatment 必须同时满足 dev 方向一致改善、final 无 collision/UTF-8/boundary 退化、plan 消融撤销收益，才保留该机制。

只改善 train、只改善 end-marker、超预算、parent 漂移、恢复不一致、运行时读取标签/答案或消融无效，均立即关闭本次 candidate。不得用更多轮数、不同 final 题或手工模板挽救失败结果。
