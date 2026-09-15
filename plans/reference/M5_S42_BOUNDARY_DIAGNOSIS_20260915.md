# M5 §4.2 诊断：`select` 失败是「**恰好压线**」，不是数值噪声

日期：2026-09-15。状态：**已诊断（只读：未改任何源码、未训练、未重跑 CI）**。
依据：技术债册 [§4.2](../active/roadmap/05_TECH_DEBT_REGISTER.md)。
产物：[诊断报告](../../reports/taiji_p3b_s42_boundary_diagnosis_20260915.json) /
[探针](../../scripts/training/diagnose_p3b_s42_boundary.py)。

## §1 结论

CI 的 `test (3.10)` 腿报 `selector did not choose a group for held-out complementary-*`，
根因**不是**"随机数值噪声"，而是 group 的 `utility` 或 `resource_cost`
**恰好落在阈值上**（`closest_margin = 0.0`）⇒ **任何末位浮点差异都会翻转 `select` 的结论**。

## §2 机制（读码）

`taiji/interaction_group_learning.py::select`：

```python
candidates = [
    item for item in self.groups
    if item.utility >= self.minimum_utility                                   # 0.0
    and (resource_budget is None or item.resource_cost <= float(resource_budget))  # 2.0
]
if not candidates:
    return None
```

而 `taiji/interaction_groups.py::_estimate_pair` 里两个量都是浮点：

```python
interaction = pair - first - second                        # 浮点减法
"pair_resource_cost": sum(...) / len(pair_episodes)        # 浮点均值
```

⇒ **两个判据都是浮点比较，且没有容差**。

## §3 实测（9 个 case 全覆盖，Python 3.12）

探针复现 3 个 family × 3 个 seed 的 leave-one-out 候选与 learner 状态：

| family | seed | groups | **closest_margin** | ±ε 翻转 |
|---|---|---|---|---|
| complementary-alpha | 11 / 29 / 47 | 2 | **0.0** | **1e-12 / 1e-9 / 1e-6 全部翻转** |
| complementary-beta | 11 / 29 / 47 | 2 | **0.0** | 同上 |
| complementary-gamma | 11 / 29 / 47 | 2 | **0.0** | 同上 |

**⇒ 9/9 都精确压在阈值上**；给 utility 与 cost 同时加 **±1e-12** 就足以让 `select`
从"选中"变成"返回 `None`"（或反之）。

**⇒ CPython 3.10 与 3.12 的浮点求值/累积差异足以触发该翻转。** 这与先前排除的两个假设
（顺序不确定、torch 版本不同）**不矛盾**：那两个确实不是原因，**真正原因是判据没有容差**。

## §4 修法建议（**未实施，需决策**）

`select` 是**被多个冻结报告依赖的机制**，改动须先定性、后预注册。

- **建议 A（推荐）**：给两个比较加**显式容差** `eps = 1e-9` ——
  `utility >= minimum_utility - eps`、`resource_cost <= budget + eps`。
  语义变化极小（只影响"恰好压线"这一情形），且**正是它要表达的意思**：
  gate 想判的是"候选可用"，而不是"浮点恰好"。
- **建议 B**：**不动比较**，改由调用方（multifamily gate）在阈值/预算上留余量。
  ⇒ 不动冻结机制，但要改 gate 的判据常量。
- **不建议**：改 tie-break（`min(key=(-utility, cost, group_id))` **已经是确定性的**）。

## §5 界限

- 本诊断**未在 3.10 上复现**（本机无该解释器）⇒ 结论是**边界敏感度的实测** +
  对 3.10 差异的**推断**；若要铁证，应在 3.10 腿补一次带该探针的 job。
- **未**改动 `select` 或任何 gate 常量；**未**重跑 CI。
