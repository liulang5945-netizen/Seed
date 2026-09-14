# M5 B0 续：组合机制根因、逃生通道与复合任务规格

> **规则版本**：本文对组合机制的根因分析（`chosen = bindable[0]` 优先级回退、`cues` 取 episode 级步数）
> 与全部数值只对 **`rule_revision = 0`** 有效。HANDOFF-M4 已于 2026-09-15 落地为 `rule_revision = 1`；
> 本文作为 revision-0 的成因记录原样保留，不回改、不重跑覆盖。

> 2026-09-13；基线 `f9825943`。
> 本文是 [B0 设计包](M5_B0_MEASUREMENT_AND_REACHABILITY_AUDIT_20260913.md) 的续篇，
> 回答 B0 只回答了"不可达"、未回答的"**为什么不可达、最少要改什么**"。
> **只做只读测量与设计，不训练、不改冻结产物、不放宽阈值。**
> 证据：[预检脚本](../../scripts/training/audit_taiji_b0_task_reachability_precheck.py) /
> [预检报告](../../reports/taiji_b0_task_reachability_precheck_20260913.json) /
> [回归测试](../../tests/taiji_native/test_b0_task_reachability_precheck.py)（15 passed）。

## §1 结论摘要

1. **组合机制是"优先级回退链"，不是自由组合。** `_member_episode` 每 tick 让所有成员预测，但**只执行 `active_members` 顺序中第一个绑定成功的成员**（`chosen = bindable[0]`）。所以 pair cell 的实际轨迹 = "第一个成员的政策，在它绑定失败的 tick 上由第二个成员接管"。
2. **仲裁上界可证**：任何"结果取自某个单体政策"的机制，其每 context 结果 ≤ `max(S_i,S_j) ≤ max_i S_i`。⇒ **无论仲裁多好，`mean(P − max_i S_i)` 都不可能为正。** 冻结矩阵实测六对上限 −0.5（五对）与 −1.0（一对），全部 ≤ 0。
3. **路由不是瓶颈**：三对（`a+d`、`b+c`、`b+d`）在**全部 4 个 context 上已达到 pair 内仲裁最优**。另三对各丢 1 个 context，损失恰为 +0.5 —— 这是**可修但不足以过关**的部分。
4. **24 个 pair×context 单元中，"交错轨迹"数为 0**：每个 pair 在每个 context 上的结果都等于它某个成员的单独结果。⇒ **从未产生过任何单体做不到的轨迹。**
5. **唯一逃生通道 = "组合专属可解 context"**：该 context 上所有单体都失败（oracle 等于空白），组合成功即贡献 `成功−B`。闭式要求：`k > n·(参照+margin)/(成功−B)`。
6. **D1 被数值化**（n=4，margin=0.15）：

   | 参照 | 参照值 | 要求增益 | **需要 k** | 现有 k | 可行 |
   |---|---|---|---|---|---|
   | 全体单体 oracle | 1.5 | 1.65 | **4 / 4** | 1 | 否 |
   | 最佳已观测固定 pair | 1.0 | 1.15 | **3 / 4** | 1 | 否 |
   | 最佳固定单体 | 0.5 | 0.65 | **2 / 4** | 1 | 否 |

7. **现行任务能支撑的参照上限只有 0.35**（`2×1/4 − 0.15`），**低于最佳可部署单体 0.5**。⇒ **换参照救不了这个 gate，任务必须改。** 要让最佳可部署单体（0.5）当参照，至少要有 **2/4** 个组合专属可解 context。
8. **给了三个复合任务的具体规格**（§5），其中 T3 由既有原语直接构造、风险最低，且天然产生 `interaction = +2`；T1 依赖"先创建后打补丁"的 digest 绑定，与冻结回退顺序天然契合。**候选已排序，未选定。**

## §2 冻结的组合规则（代码级）

位置：`scripts/training/eval_taiji_p5_2b_group_causal_corpora_gate.py::_member_episode`（P5.2c 系列逐字继承）。

```
每 tick：
  for member in active_members:            # 顺序 = 干预配置顺序
      kind = learner.predict_episode(cues)[-1]
      params, provenance, failure = _bind(kind, task, state)
      if failure is None: bindable.append(member)
  if not bindable: return "all_members_exhausted"
  chosen = bindable[0]                     # ← 冻结的字典序优先回退
  执行 chosen 的动作
```

**推论（决定后续所有设计）**：

| 推论 | 含义 |
|---|---|
| 每 tick 只有一个成员真正执行 | pair cell 不是"两个政策并联"，而是"优先级回退" |
| 第二个成员只在第一个**绑定失败**时接管 | 交错轨迹是否出现，取决于**绑定失败模式**，而这是任务设计的属性 |
| 绑定成功但动作错误时，第二个成员**没有机会** | 这解释了 `a+c` 在 context-110 上拿到 −1（`a` 绑定成功但走错），而 `c` 单独能拿 +1 |
| 学习者的作用仅限 (a) 每个成员的动作类型 (b) 选择哪一对 | 学习者**从不仲裁** cell 内部谁上 |

## §3 仲裁上界（可证，不依赖具体数据）

> **命题**：设机制 M 在 context `ctx` 上的结果为 `P_M(ctx)`，且 `P_M(ctx)` 取自某个单体政策在该 context 上的结果。则 `P_M(ctx) ≤ max_i S_i(ctx)`，从而 `mean(P_M − max_i S_i) ≤ 0`。

证明：`max(S_i,S_j) ≤ max_i S_i`（子集最大值不超过全集最大值），对任意子集成立。

⇒ **任何"成员仲裁"类机制（固定选择、条件化选择、完美 oracle 选择）都无法产生正的 `mean(P − max_i S_i)`。** 表征改进、排序改进、加 per-block 列，都不可能越过这条线。

⇒ 冻结矩阵实测：六对仲裁上限 = −0.5（五对）/ −1.0（`a+d`）。**这不是"效果不好"，是数学上界为负。**

**推论（对 B1 的直接约束）**：B1 的表示改进最多能回收 §4 的"冻结规则 headroom"（≤ +0.5），**永远不能单独支撑 H2 协作主张**。H2 必须先有 §5 的任务。

## §4 冻结矩阵实测：轨迹分类

逐 context 分类 pair 结果与两成员结果的关系（24 个单元）：

| pair | 达到仲裁最优的 context 数 | 交错 context 数 | 冻结规则损失（headroom） |
|---|---|---|---|
| `member-a+member-b` | 3 / 4 | **0** | +0.5 |
| `member-a+member-c` | 3 / 4 | **0** | +0.5 |
| `member-a+member-d` | 4 / 4 | **0** | 0.0 |
| `member-b+member-c` | 4 / 4 | **0** | 0.0 |
| `member-b+member-d` | 4 / 4 | **0** | 0.0 |
| `member-c+member-d` | 3 / 4 | **0** | +0.5 |

两条独立读数：

1. **路由已到位**：三对在全部 context 上达到 pair 内最优，说明回退机制在"选谁上"这件事上已经做对了。
2. **从未交错**：0 个交错单元 ⇒ 所有 pair 结果都是某个单体的结果 ⇒ §3 的上界是**紧的**，不是保守估计。

⇒ **失败归因**：不是表征（A 的 rank 修复真实但作用在负上界上），不是路由（已达最优），而是**任务里不存在只有组合才能解的 context**，加上**机制只能产出单体轨迹**。

## §5 唯一逃生通道与复合任务规格

### 5.1 组合专属可解 context（闭式要求）

> `k > n · (参照增益 + margin) / (成功 − 空白)`

n=4、margin=0.15、成功−空白=2 时：要求 k ≥ 2（参照 0.5）/ 3（参照 1.0）/ 4（参照 1.5）。

**现行任务只有 1 个这样的 context**（`p52a-validation-111`，block-3），而且它是 `contract_intercepted: language_evidence_ambiguous`——**组合在那个 context 上也没成功**。所以有效 k = 0，可支撑参照上限 0.35 < 0.5。

### 5.2 三个候选规格（**供审阅，未选定**）

既有原语：`workspace.read / list / create / apply_patch / programming_language.resolve`、`editor.set_language`；四个模板族 = block0 `lang_confirm`、block1 `patch_persist`、block2 `create_persist`、block3 `header_override`（合同拦截）。

| 候选 | 具体构造 | 为什么"组合专属可解" | 回退机制能否产生交错 | lesion 判别 | 风险 |
|---|---|---|---|---|---|
| **T3 联合必需（双目标）** | 一个任务含两个独立子目标：文件 1 需显式语言覆盖（block0 能力）+ 文件 2 需补丁落盘（block1 能力）。任一单体只具备其中一个能力 | 两个单体各失败，只有组合能同时满足 ⇒ `k` 可**按构造**做到 ≥2 | 第一个成员绑定失败/走完自己的部分后由第二个接管；顺序无关，最稳 | 去掉任一成员 ⇒ 增益归零；去掉任一子目标 ⇒ 退化为单成员任务 | **低**（零新机制、零新原语） |
| **T1 串联交接（先建后补）** | 初始无文件；目标 = 文件存在且内容为补丁后内容；补丁的 `before_digest` 必须是**刚创建内容**的 digest | 创建者不会打补丁，打补丁者无文件可打 ⇒ 两个单体都失败 | **天然契合**：`member-b` 先（字典序），它绑定 `apply_patch` 时文件不存在 ⇒ 绑定失败 ⇒ `member-c` 接管创建 ⇒ 下一 tick `b` 的补丁绑定成功 | 去掉 digest 绑定（补丁不带 `before_digest`）⇒ 退化为单成员可解 ⇒ **任务无效**；去掉任一成员 ⇒ 归零 | 中（依赖绑定失败模式，须探针验证） |
| **T2 纠错交接** | 成员 A 产出被合同拦截的候选；成员 B 必须读取 A 的失败证据才能修复 | 单体的失败证据不构成能力 ⇒ 只有组合有解 | 需要显式证据通道；若通道由 runner 提供即等于脚本补答案 | 断开证据通道 ⇒ 增益消失 | **高**（需新证据通道与权限论证） |

**T3 与 B0 手算用例「联合必需」一一对应**（`interaction = +2`，`mean(P − max_i S_i) = +2`）；T1 对应「串联交接」并额外验证回退顺序；T2 对应「纠错交接」，工程量与合规风险最高。

### 5.3 任何候选的预检通过条件（训练前强制，六条全过）

| # | 条件 | 判据 |
|---|---|---|
| 1 | 组合专属可解 context 数足够 | `available_k ≥ required_k`（§5.1 闭式） |
| 2 | 冻结机制真能产生交错 | **脚本化探针**（未训练）下至少一个候选 pair 出现 `contexts_interleaved ≥ required_k` |
| 3 | lesion 使收益消失 | 拆除任一成员后组合增益归零 |
| 4 | 非平凡 | 无 tick-0 满足（沿用 `_assert_nontrivial_goals`） |
| 5 | 合同合法 | 新模板无 `contract_intercepted` |
| 6 | 不可脚本补答案 | runner 不构造目标终态；digest 绑定等约束由 contract 层强制 |

**条件 2 无法由 oracle 推出**（oracle 不知道绑定失败模式），必须用脚本化参考步骤跑一次真实执行——这正是 T1 与 T3 需要先做探针的原因。

> **条件 2 已实测（2026-09-13）**：[交接可行性探针结果](M5_B0_HANDOFF_PROBE_RESULT_20260913.md)。
> 探针先复现冻结矩阵 **11/11**，随后实测：`create_then_patch` 与 `dual_requirement` **被否决**
> （分别有 `member-c` / `member-d` 可独做），`create_and_override` **通过任务层**（`k=4/4`、三种参照全 feasible）
> 但**六个 pair 仍全部失败、`interleaved=0`** ⇒ **机制层是唯一阻塞**，据此提出最小修法 **M1（待决策 D5）**。

## §6 预检工具

`scripts/training/audit_taiji_b0_task_reachability_precheck.py`（只读，duck-typed，零 torch 依赖）：

| 函数 | 回答 |
|---|---|
| `arbitration_ceiling(table, pair)` | 该对的仲裁上界（≤0，实测 −0.5 / −1.0） |
| `arbitration_headroom(table, pair)` | 冻结规则损失了多少（可回收部分，≤0.5） |
| `classify_pair_trajectory(table, pair)` | 逐 context 分类：等于某成员 / 交错 / 达到仲裁最优 |
| `combination_only_solvable_contexts(table)` | 逃生通道在哪几个 context |
| `combination_only_gain_potential(table)` | 完美利用逃生通道时的增益上限 |
| `required_combination_only_contexts(...)` | **需要几个**（D1 的数值化） |
| `max_clearable_reference(...)` | 该任务形状最多能支撑多高的参照 |
| `precheck(...)` | 汇总判定 + 阻塞层归因 |

用法：`python scripts/training/audit_taiji_b0_task_reachability_precheck.py`。
新任务候选在**训练前**必须跑该工具并归档报告；未通过 §5.3 六条的不进入 B1。

## §7 全量测试基线（本轮采集，闭合 B0 §10 缺口）

命令：`python -m pytest tests/taiji_native/ -q --junitxml=<tmp>/taiji_b0_full.xml`（后台，墙钟 938s ≈ 15.6 分钟）。

| 指标 | `aa124f52`（登记册基线） | 本轮 `f9825943`+B0 | 变化 |
|---|---|---|---|
| 用例总数 | 788 | **851** | +63（C/A/B0 新增测试） |
| 失败 | 30 | **27** | −3 |
| 错误 | 0 | 0 | — |
| 跳过 | 1 | 1 | — |
| **类别 A（架构边界违反）** | **2** | **0** | **已结项** |
| 类别 B（`SystemExit: 1`） | 28 | **27** | −1（顺序依赖，集合每次不同） |

**结论**：

1. **类别 A 已结项且可复现**：`test_architecture_contract`、`test_naming_boundary_contract` 在**全量上下文**下也转绿（此前只有目标集 64 passed 证据）。
2. **类别 B 是严格子集**：本轮 27 项**全部**落在旧 28 项之内，**没有新增失败**。唯一消失的是
   `test_natural_language_workbench::test_natural_language_workbench_gate_passes`——
   与"失败集合每次不同"的既有观测一致，也进一步支持"顺序/状态污染"而非逻辑缺陷的定性。
3. 27 项仍**全部**是 `SystemExit: 1`，仍无可读栈（登记册 §4 的采集障碍未解决）；处置阶段仍需先建立 `SystemExit` 可观测性。

## §8 待决策（D1–D4 现在带数字）

| # | 决策点 | 带数字的选项 |
|---|---|---|
| D1 | 估计目标与参照 | 参照 0.5（最佳可部署单体）⇒ 需 **k ≥ 2/4**；参照 1.0 ⇒ 需 **3/4**；参照 1.5（oracle，不可部署）⇒ 需 **4/4**。**现行任务支撑上限仅 0.35，故无论选哪个都必须先改任务。** |
| D2 | 复合任务候选 | **T3 双目标联合必需**（风险最低、天然 `interaction=+2`）/ **T1 先建后补**（与回退顺序天然契合，需探针验证绑定失败模式）/ T2 纠错交接（需新证据通道）。均须过 §5.3 六条 |
| D3 | 旧 1.65 判据 | 保留为历史记录；新判据按统一字典推导，且**必须声明参照与所需 k** |
| D4 | 旧载体定位 | 144/88 降级为开发回归（已有证据：24 个单元零交错、k=1）；新测试面独立冻结 |
| **D5（新增）** | **是否预注册并落地机制修法 M1** | 由[交接探针](M5_B0_HANDOFF_PROBE_RESULT_20260913.md) §5 提出：M1-a（无进展即让位，推荐）/ M1-b（固定轮转）/ M1-c（learner 排序）。**不落地 M1 ⇒ H2 协作主张在当前机制下不可达** |

**D1–D5 明确前不启动 B1 训练。** 本轮未代替用户批准任何候选。

## §9 纪律

- 只读：预检工具无训练、无 checkpoint 写入、无 `shutil.rmtree`（测试断言禁止出现这些符号）。
- 未改任何冻结报告/预注册/阈值；`growth_admitted=false`、`can_promote=false` 贯穿。
- 历史负结果（`transfer_no_gain` / `transfer_signal_constant`）原样保留。
- 复算一致性：预检建立在 B0 §3.1 的 32/32 字段对照之上；该对照失败则拒绝出报告。
- 唯一代码改动是新增两个只读审计脚本与两个测试文件，不触及 `taiji/`、`seed_platform/` 与既有 runner。
