# M5 B0 交接可行性探针：结果与最小机制修法提案

> 2026-09-13；基线 `f9825943`。
> 本文是 [机制与预检续篇](M5_B0_MECHANISM_AND_TASK_PRECHECK_20260913.md) 的实证收口，
> 回答 §5.3 **条件 2**（冻结机制能否真的产生交错轨迹）——该条件**无法由 oracle 推出**。
> **只做只读探针，不训练新模型、不改任何 gate、不注册任何任务、不放宽阈值。**
> 证据：[探针脚本](../../scripts/training/probe_taiji_b0_handoff_feasibility.py) /
> [探针报告](../../reports/taiji_b0_handoff_feasibility_probe_20260913.json) /
> [契约测试](../../tests/taiji_native/test_b0_handoff_probe_contract.py)（10 passed，不跑训练）。

## §1 结论摘要

1. **探针可信**：先用**冻结执行器逐字**重跑冻结验证面，**11/11 cell 成功率与冻结报告一致**（264 episodes，`mismatches=[]`）。不能复现历史历史的探针无权谈未来。
2. **三层已被完全分离**——这是本轮最重要的成果：

   | 层 | 状态 | 证据 |
   |---|---|---|
   | **任务层** | **可修，且已找到修法** | `create_and_override` 候选：四个单体**全部失败**、四个 context **全部**为组合专属可解（`k=4/4`）、**三种候选参照全部 feasible**、可支撑参照上限 **1.85** |
   | **机制层** | **仍是唯一阻塞** | 同一候选上六个 pair **全部失败**，`interleaved = 0`，`order_headroom = 0` |
   | **表征层** | 与本轮无关 | 任务层达标后机制仍做不到，故此刻改表征无意义 |

3. **机制根因已精确到触发条件**：交接的触发条件是"**第一个成员绑定失败**"。但成员的未训练位置仍会绑定无害动作（`workspace.read` / `workspace.list`），于是第一个成员**独占整个 episode**（直到 `STEP_CAP=8`），其他成员永远轮不到。
4. **同根因在冻结矩阵上也有实测代价**：`dual_requirement` 面上 `member-d` 单独成功率 **1.0**，但 pair `(a,d)` 正向 **−1.0**、反向 **+1.0** —— 顺序一换就从失败变成功，`mean_order_headroom = 1.0`。这正是冻结矩阵里 `a+b`/`a+c`/`c+d` 各丢 0.5 的同一现象。
5. **两个候选被实测否决（并给出了原因）**：
   - `create_then_patch`：`member-c` 单独成功率 **1.0** —— `workspace.create` 直接绑定 `goal_files[main_path]`，创建即达标 ⇒ **T1 形态不可表达**，除非改 `_bind`。
   - `dual_requirement`（覆盖+补丁）：`member-d` 单独成功率 **1.0** —— `header_override` 家族的政策本身就是 `read→resolve→set_language→apply_patch`，**既有任务集里已经存在"双要求"形态，只是分配给了一个成员**。拆成两个成员不改变"有成员能独做"这一事实。
6. **提出最小机制修法 M1（待决策 D5，未实施）**：把交接触发从"绑定失败"改为"**无进展**"。M1 不违反仲裁上界——上界约束的是"结果取自某个单体政策"的机制，而 M1 的目的恰恰是让 `P ∉ {S_i}` 成为可能。
7. **注意区分**：`create_and_override` 只是**探针候选**，不是已选定的新任务；选定仍需 D2 决策，M1 落地仍需 D5 决策。

## §2 探针可信度（先证明，再使用）

| 项 | 值 |
|---|---|
| 复现对象 | 冻结 P5.2c″ 报告的 `success_matrix`（12 个验证 context × 11 个 cell） |
| 执行器 | `p5_2b._execute_matrix`（**逐字复用，非重写**） |
| episodes | 264 |
| 比对 cell 数 | 11 |
| `reproduced` | **true** |
| `mismatches` | `[]` |

契约测试另行钉住"探针不得重写冻结机制"：源码中不得出现 `def _member_episode` / `def _train_members`，
且必须实际调用 `frozen._member_episode(` / `frozen._execute_matrix(` / `frozen._train_members(`。

**探针复用的冻结产物**：组合规则 `_member_episode`、四个成员 `_train_members`（只用四个既有 train 模板族训练）、
cell 枚举 `CELL_MEMBER_SETS`、重复数 `REPEATS`。⇒ 候选任务面对的是**从未见过它的成员**，且**不为候选任务训练任何模型**。

## §3 三个候选面的实测

每个面 4 个 context × 2 重复；结果为 `±1` 均值（成功 +1 / 失败 −1）。

| 候选 | 单体成功率（a/b/c/d） | pair 结果 | 组合专属可解 context | `interleaved` | 顺序敏感 |
|---|---|---|---|---|---|
| `create_then_patch` | 0 / 0 / **1.0** / 0 | 部分 1.0（含 c 的对） | **0** | 0 | 是（`a+c` −1→+1） |
| `dual_requirement` | 0 / 0 / 0 / **1.0** | 全 −1.0 | **0** | 0 | 是（`a+d`/`b+d`/`c+d` 均 −1→+1） |
| **`create_and_override`** | **0 / 0 / 0 / 0** | **全 −1.0** | **4 / 4** | **0** | 否 |

`create_and_override` 的参照可行性（n=4，margin=0.15）：

| 参照 | 参照值 | 需要 k | 现有 k | feasible | 可支撑参照上限 |
|---|---|---|---|---|---|
| 全体单体 oracle | 1.5 | 4 | **4** | **是** | **1.85** |
| 最佳已观测固定 pair | 1.0 | 3 | **4** | **是** | **1.85** |
| 最佳固定单体 | 0.5 | 2 | **4** | **是** | **1.85** |

⇒ **任务层不再是阻塞**：这是 B0 两轮以来第一次出现"参照无论怎么选都可达"的任务形状。

### 3.1 候选构造（探针内，未注册）

| 候选 | `initial_files` | `goal_files` | `goal_language` | `requires_explicit_language_override` |
|---|---|---|---|---|
| `create_then_patch` | `{}` | `{main: content}` | `{}` | 否 |
| `dual_requirement` | `{main: base}` | `{main: patched}` | `{main: python}` | **是** |
| **`create_and_override`** | `{}` | `{main: content}` | `{main: python}` | **是** |

**结构约束（探针实测得出，非推断）**：`p52a._bind` 对**所有**动作种类都只作用于 `task.main_path`，
且 `workspace.create` 绑定 `content = task.goal_files[main_path]`。
⇒ 单任务只能有一个 `main_path`；`goal_files` 里任何非 `main_path` 的文件**永远无法被满足**；
"创建后打补丁"因创建即达标而**不可表达**。契约测试从冻结 binder 重新推导这两条，防止文档与代码脱节。

## §4 机制根因（精确到触发条件）

`p5_2b._member_episode` 的交接条件是 **`chosen = bindable[0]`，即"第一个绑定成功的成员"**。

```
每 tick：
  for member in active_members:
      kind = learner.predict_episode(cues)[-1]      # 未训练位置仍会输出一个 kind
      if _bind(kind, task, state) 成功: bindable.append(member)
  chosen = bindable[0]                              # ← 只有绑定失败才会让位
```

**为什么永不交错**：`workspace.read` / `workspace.list` / `workspace.resolve` 这类动作**几乎总能绑定成功**。
第一个成员即使在自己的专长用完后，仍会持续绑定无害动作，直到 `STEP_CAP=8` 耗尽，**第二个成员一次也没机会执行**。

**三条独立证据**：

1. `create_and_override`：没有任何单体能做（全 0），六个 pair 也全失败，且**顺序无关**（换顺序也一样失败）⇒ 不是"选错成员"，而是**根本没有交接**。
2. `dual_requirement`：`member-d` 单独 **1.0**，而 `(a,d)` 正向 **−1.0**、反向 **+1.0** ⇒ 有能力的成员被排在前面的成员**压制**，`mean_order_headroom = 1.0`。
3. 冻结矩阵：`a+b`/`a+c`/`c+d` 各丢 0.5（`arbitration_headroom`），**同根因**。

## §5 最小机制修法提案 M1（**待决策 D5，未实施**）

### 5.1 提案

> 把交接触发条件从"**当前成员绑定失败**"改为"**当前成员相对上次执行未产生进展**"。

具体判据（三选一，须在预注册中冻结其一）：

| 编号 | 判据 | 说明 |
|---|---|---|
| **M1-a（推荐）** | 若所选动作与**上一次已执行动作**（kind + 参数摘要）相同，或该 kind 已在本 episode 执行过且目标未推进 ⇒ 视为无进展，轮到下一个可绑定成员 | 最小改动，保留优先级语义，只去掉"独占" |
| M1-b | 每 tick 按固定轮转让成员依次执行，直到达标或预算耗尽 | 改动更大，丢弃优先级语义 |
| M1-c | 由 learner 按预测给成员排序后再执行 | 最大改动，需 B1 的表示工作 |

### 5.2 为什么 M1 不与仲裁上界矛盾

续篇 §3 的上界针对"**结果取自某个单体政策**"的机制（`P ∈ {S_i}`）。
M1 的目的正是让 `P ∉ {S_i}` 成为可能——即**真正的交错轨迹**。上界仍然成立，M1 只是在它的适用范围之外。

### 5.3 M1 的验收（必须全部满足，且必须先预注册）

1. **不回归**：在冻结验证面上，11 个 cell 的成功率**不得低于**旧规则（旧规则结果作为对照保留）。
2. **真能交错**：在 `create_and_override` 面上 `interleaved > 0`，且该 pair 的 `mean(P − max_i S_i) > 0`。
3. **旧结果可比性**：M1 改变冻结规则 ⇒ 既有 P5.2b/c 报告**只对旧规则有效**；M1 下的任何结论必须重跑并单独报告，**不得改写旧报告**。
4. **不允许放宽**：不得为了让 M1 通过而放宽安全/合同门；`STEP_CAP`、预算、审批路径不变。
5. **可撤回**：M1 是单点改动，须可回滚到 `chosen = bindable[0]`。

### 5.4 M1 不解决的事

M1 让交接**成为可能**，但**不保证**在 `create_and_override` 上成功——必须重跑探针实测（这正是 §5.3 第 2 条）。
M1 也**不**自动产生 H3 排序能力，也不改变成本与安全口径。

## §6 对 D1–D5 的影响

| # | 决策点 | 本轮实测后的状态 |
|---|---|---|
| D1 | 估计目标与参照 | **被 `create_and_override` 解除阻塞**：该任务形状下三种参照全部 feasible，可支撑参照上限 1.85。推荐仍是"主判据用可部署对照、oracle 仅作诊断上界" |
| D2 | 复合任务候选 | **`create_then_patch`（T1）实测否决**（创建即达标）；**`dual_requirement` 实测否决**（`member-d` 可独做）；**`create_and_override` 实测通过任务层**（k=4/4）。候选清单据此收敛为一项，但仍需用户确认 |
| D3 | 旧 1.65 判据 | 不变：保留为历史记录，新判据须声明参照与所需 k |
| D4 | 旧载体定位 | 不变：144/88 降级为开发回归 |
| **D5（新增）** | **是否预注册并落地机制修法** | 三选项 M1-a / M1-b / M1-c；**不落地 M1 则 H2 协作主张在当前机制下不可达** |

> **D5 已实测（2026-09-13 续）**：[机制修法反事实测量](M5_B0_M1_COUNTERFACTUAL_RESULT_20260913.md)。
> 六个变体**五个失败、一个成功**：`m1a` 安全但完全无效、`m1b` 破坏基线、`m2`/`m2a`/`m3` 均回归且无效；
> **`m4_failure_handoff`（失败即让位 + 每成员自身进度）冻结面 0 回归 / 2 改善，候选面 `interleaved=4/4`、
> 同参照增益 `+2.000 > 1.65`** ⇒ **H2 协作主张首次可达**。M4 仍未实施。

## §7 纪律

- 探针**只读**：不训练候选任务的模型、不写 checkpoint、不注册任务、不改任何 gate/预注册/阈值。
- 候选任务定义**只存在于探针内**，`partition="probe"`，id 前缀 `b0probe-`，与 100–111 冻结区间刻意分离。
- 先复现历史（11/11）再谈未来；不能复现即拒绝出报告。
- 旧规则下的既有报告**原样保留**；M1 若落地，必须重跑并单独报告。
- `growth_admitted=false`、`can_promote=false` 贯穿；历史负结果未改绿。
- 契约测试不跑训练（<2s），探针执行（≈25s）作为脚本化仪器单独运行。
