# 10.2：自然 Goal 绑定·契约、反例与基线（不授予训练）

> 文档状态：[合同] — owner 已于 2026-11-01 批准立项。本合同冻结绑定的输入/输出契约、五类公开反例、失败计数口径与一个**规则基线**；**模型侧绑定不在本步**，也不授予训练。

日期：2026-11-01。`taiji/living_core.py` 字节保持不变；10.1 的环境与评分不改。

## 1. 绑定的输入与输出（契约）

- **输入**：`(goal_text: str, observable_state)` —— 只有中文原文与可观察状态。**宿主不得传入 Goal、gold 终态、正确性标签或评分**。
- **输出**：`Binding`：
  - `conditions`：`tuple[(object, place, polarity)]`，`polarity ∈ {+1, −1}`（−1 表示"不要在该处"）；
  - `abstain`：是否拒答；
  - `reason`：拒答原因（`missing_referent`／`ambiguous`／`unsupported` 之一）；
  - `unknowns`：绑定过程中显式登记的不确定性（如"未指明对象"）。
- 绑定结果交给 10.1 的评分器判定终态，**绑定器自身不评分、不看终态**。

## 2. 五类公开反例（每条一个夹具，全部入分母）

| 类别 | 例子 | 期望 |
|---|---|---|
| 改写 | 「把蓝色杯子放到窗台」vs「蓝色杯子应该出现在窗台上」 | 两条绑定出**同一**条件集 |
| 否定 | 「红色印章不要留在左盒」 | 条件极性为 −1 |
| 约束变化 | 「把蓝色杯子放到窗台，并且红色印章不要留在左盒」 | 合取，两条条件都在 |
| 缺条件 | 「把它放到窗台」 | `abstain` + `missing_referent`，**不得猜对象** |
| 歧义 | 「把蓝色杯子放到窗台或木桌」 | `abstain` + `ambiguous`，**不得任选一个** |

## 3. 判据

1. **分母不跳**：每条夹具都必须产出 `binding` 或 `abstain`，逐条计入分母；有任何一条被静默丢弃 ⇒ `fixture_dropped`。
2. **改写等价**：改写对绑定出同一条件集 ⇒ 否则 `paraphrase_mismatch`。
3. **否定与合取**：否定极性为 −1、合取含两条条件 ⇒ 否则 `negation_or_conjunction_wrong`。
4. **缺条件与歧义必须拒答**：`abstain` 为真且 `reason` 正确；**猜了就算错**（即使猜中还判错，因为猜中不可被判据区分）⇒ 否则 `abstain_not_enforced`。
5. **宿主不喂 gold**：绑定器签名只含 `(goal_text, state)`；AST 扫描 `taiji/goal_binding.py` 内不得出现 `Goal`／`gold`／`score` 标识符 ⇒ 否则 `gold_leaked_into_binder`。
6. **基线声明**：本步的绑定器是**规则基线**，卡片必须写明"模型侧绑定未交付"⇒ 否则 `model_binding_claimed`。

## 4. 预算、出口与交付

- 预算：≤120 CPU 秒 / 240 墙钟秒；超限 `budget_stop`。
- 出口：`passed_10_2_binding_contract_only`／`fixture_dropped`／`paraphrase_mismatch`／`negation_or_conjunction_wrong`／`abstain_not_enforced`／`gold_leaked_into_binder`／`model_binding_claimed`／`budget_stop`／`engineering_failure`。
- 交付：本合同、`taiji/goal_binding.py`（规则基线）、`tests/taiji_native/test_goal_binding.py`、`scripts/training/verify_taiji_goal_binding_10_2.py`、`reports/taiji_core_goal_binding_10_2_20261101.json`、MAINLINE/03 同步。
- **通过只表示**绑定契约、五类反例与失败口径成立，且规则基线在该夹具集上按声明行为工作；**不表示**模型能绑定自然中文目标，也不表示 10.3 准入。
