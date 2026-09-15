# M5 B0 / WP-6 步骤 1：binder 放宽的**只读可行性论证**（binder v1 → v2）

> 日期：2026-09-15。性质：**只读取证 + 设计推演**，未改任何代码、未改任何已冻结预注册。
> 依据：[路线 B 冻结版](M5_B0_ROUTE_B_PREREGISTRATION_FROZEN_20260915.md)（适用范围"仅 binder v1"，
> 并规定 WP-6 必须另立带版本标注的预注册）、[结构空间结果](M5_B0_STRUCTURE_SPACE_RESULT_20260913.md) §7/§11、
> [落地结果](M5_B0_M4_LANDING_RESULT_20260915.md)（指纹仍 3/11 ⇒ **L1 未闭合**，落地没有改变它）。
> 所有行号均来自本轮实测读取（HEAD `c0f7b4c6` 之后）。

## 1. 结论先说（三条，各自可核对）

1. **"改 binder"不是一个动作，是两个。**
   - **T3（两个目标文件）只缺 binder**：目标谓词 `_goal_reached` 已经遍历**全部** `goal_files` 并逐一比摘要
     （`eval_taiji_p5_2a_predictive_execution_gate.py:620-643`），缺的是"没有任何动作能绑定到第二个路径"。
   - **T1（先创建后补丁）还缺三样**：`create` 绑定的是**目标内容本身**（`_bind:573-581`
     `content = task.goal_files.get(main)`）⇒ 创建即达标、随后 `apply_patch` 在
     `_bind:582-607` 返回 `"bind_failure: file already at goal state"`；此外还有
     非平凡性门 `_assert_nontrivial_goals`（`:299-322`）与主循环"tick 顶端先判达标、直接 `goal_reached` 退出"
     （`:710`、`p52b:243`）。**只放宽 `_bind` 不足以让 T1 可表达**。
2. **binder 不在产品里。** `main_path` / `goal_files` 在 `taiji/`、`seed_platform/`、`api/` 中不存在；
   它是测量 gate 的一个函数 `_bind(kind, task, state)`（`:533`，`main = task.main_path` 在 `:545`，
   10 个可绑定 kind 里 **9 个**从 `main` 解析路径，只有 `workspace.list` 用 `{"path": "."}`）。
   产品侧 `seed_platform/workbench.py` 的 `_apply_patch:2956` / `_create_file:3030` 本来就接受任意 `path` 参数。
   ⇒ **WP-6 改的是仪器口径，不是产品能力**；它与 01 号账本"机制正式版本"的缺口是两件事，不得互相顶替。
3. **一个必须先处理的不对称**：参照臂 `_oracle_arm`（`:807-849`）**绕过 `_bind`**，直接执行
   `p52._resolve_params(step.params, state)`（`:816`）⇒ 参照步骤**早就**可以指向第二个文件，
   而被测模型路径被钉在 `main_path`。若只放宽模型路径而不复核参照臂，v2 下的"参照 vs 被测"
   会**系统性偏向参照**，那正是本工作包要防的"分数被口径制造"。**这条必须在 v2 预注册里显式处置**
   （要么两臂同走放宽后的 binder，要么两臂都保持 v1 并在报告里并列双口径）。

## 2. 最小改动面（实测计数，不是估计）

| 改动 | 位置 | 触达面 |
|---|---|---|
| 给 `_bind` 一个路径选择参数（目标索引 / 显式 path） | `p52a:533`（体内 `main` 引用 15 次、9 个 kind） | **7 个非测试调用点**（`p52a:714`、`p52b:261`、`probe_taiji_b0_m1_counterfactual:650`、`probe_taiji_b0_structure_space:366,517,545,546`）+ **5 个测试调用点** |
| `Task.main_path`（`:102`，frozen dataclass）→ 多目标路径 | 构造处 | **19 个 `main_path=` 构造点** + 3 处 `Scene` 重投影（`p52a:909`、`p52b:161`、`audit_taiji_b0_m4_artifact:116`）+ 身份键 `(task.main_path, tuple(sorted(goal_files.items())), template)`（`p52a:941`） |

## 3. 会被打破的冻结断言（改名或放宽都算改口径，须逐条处置）

| 断言 | 位置 |
|---|---|
| `binder_targets_main_path_only is True`、`bindable_kind_count == 10` | `test_b0_structure_space_contract.py:127,133` |
| T1/T3 必须 `inexpressible_under_frozen_binder`（含 `create_bound_content`、两路径目标实际只绑到 `t3_main.py`） | 同上 `:617,632` |
| 每格只有一个目标路径、`not_covered` 必须提 T2 | 同上 `:655-656,692` |
| `set(task.goal_files) == {task.main_path}`、`params["content"] == goal_files[main_path]`、`override_params["path"] == main_path` | `test_b0_handoff_probe_contract.py:103,124,132,141,149` |
| 校验任务的每一步都必须能绑定成功 | `test_intervention_reality_gate.py:180,201,228-236` |
| CAP-0 边界"`modifies_binder` 为假" | `test_cap0_eval_set_contract.py:161`（对应 `M5_CAP0_EVAL_SET_FROZEN_20260915.md:64`） |
| 词汇表冻结：`action_kinds` 解析后**只增不减**（发现集必须是已解析集的子集） | `taiji/procedural_memory.py:42-81`（子集校验在 `:68`） |

注意 `binder_targets_main_path_only` 当前是由 `binder_expression_surface` 从**源码文本**算出的形状判据
（`probe_taiji_b0_structure_space.py:704-739`，键在 `:739`）——即"表达式宽不严"这件事今天仍是被文本形状
度量的。v2 若引入路径参数，这个度量必须改成**行为度量**（能否真的绑出第二个路径并达标），
否则它会一边报 `false` 一边什么都没验证。

## 4. v2 应当继承的出口判据（预先写死，避免看到结果后调门）

1. **v1 基线仍逐字节复现**：`binder_version=v1` 路径下所有已封存报告哈希不变（沿用 WP-3 的封条做法）。
2. 新格（T1/T3 形态）必须 `measurable=true` 且 `contract_blocked_steps = 0`。
3. **新格结果指纹必须与 create 行三格不同**——否则如实记为"L1 界限未突破"，而不是"多测了一格"。
   （v1 的事实：`3 distinct fingerprints over 11 cells` ⇒ 独立结构因素 = 1。）
4. 参照臂与被测臂的口径按 §1.3 的处置写进预注册，并在报告里同时给出两口径数字。
5. 全量测试失败集合对基线无新增；任何"放宽断言让它变绿"必须换成**双版本断言**（v1/v2 各钉一次）。

## 5. 成本与排期（诚实估计）

- 改动本身中等（上表 26 个触达点），但**成本主要在口径**：`binder_targets_main_path_only` 从文本度量改行为度量、
  6 组冻结断言要做成双版本、五件探针要按 `binder_version` 重跑一遍（结构空间满档一轮约 12 分钟，两臂口径 ×2）。
- **本轮不做**：P3b 双臂正在跑，任何 `_bind`/`Task` 改动都会改变评测链路 ⇒ 会破坏在跑 campaign 的 J1 可比性。
  等 P3b 结论落定后再开 v2 预注册，并把它作为**独立工作包**推进。
- 决策请求（默认按上限档）：v2 采用"**两臂同走放宽后 binder**"作为主口径 + v1 双列披露，而不是只放宽被测路径。

## 6. 本步的边界

本文**不证明**"放宽 binder 后一定出现第二个独立结构因素"——它只证明：(a) T3 的瓶颈确实在 binder 且**仅在** binder；
(b) T1 的瓶颈不止在 binder；(c) 需要处置的冻结面与一处会偏向参照的口径不对称；(d) 可表达性判据必须从文本形状改成行为形状。
在这些之前，任何"跨内容结构已稳健"的表述都不成立。
