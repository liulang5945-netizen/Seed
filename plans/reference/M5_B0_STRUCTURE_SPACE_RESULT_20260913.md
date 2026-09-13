# B0 / N1：M4 的结构稳健性探针结果（2026-09-13）

> 状态：**只读测量轮，结论草稿待审阅**。未实施 M4，未注册任务，未训练候选专用模型，
> 未更改任何冻结预注册、门禁或阈值。
> 脚本：`scripts/training/probe_taiji_b0_structure_space.py`
> 报告：`reports/taiji_b0_structure_space_probe_20260913.json`（两次重跑**字节相同**，80273 B）
> 测试：`tests/taiji_native/test_b0_structure_space_contract.py`（**30 passed**）
> 前置：[加固轮结果](M5_B0_M4_HARDENING_RESULT_20260913.md) §8 登记的开放项 **N1**
> 回答的问题（用户决策）：**M4 的收益是否只在 `create + override` 这一种结构下成立？**

## 1. 一句话结论

**不只在一种结构下成立，但也没有超出一种结构因素。** M4 的正增益覆盖 `create` 行的**全部三条语言路由**
（observation / override / mismatch，各自 `+2.000`、`interleaved 0→2`、其余 8 个 cell 增益 `0.000`），
且修复的是**两种不同的冻结失败形态**（合同拦截型与步数耗尽型）；
网格中**除 `create` 行外没有任何 cell** 出现正增益 ⇒ 承重的结构因素是**存在性前提**，不是 override。

## 2. 为什么不能用 §5.2 的三个候选直接做

加固轮 §10 给的下一步是"用 T1/T2/T3 构造结构不同的候选面"。先做可表达性探针，**结论是两个建不出来**：

| 候选 | 可表达性（实测见证，非断言） | 证据 |
|---|---|---|
| **T1 串联交接**（create→patch） | **不可表达** | `workspace.create` 绑定 `content = task.goal_files[main_path]`，创建即达标；随后 `workspace.apply_patch` 以 `file already at goal state` 拒绝。`goal_reached_by_create_alone = true` |
| **T3 双文件联合目标** | **不可表达** | `_bind` 把**每个**动作都解析到 `task.main_path`：两个目标路径的绑定 `path` 全等于 `t3_main.py`；第二文件的语言子句永远无法被任何动作满足 ⇒ `scripted_goal_reached = false` |
| **T2 纠错交接** | **不可表达（成员无证据通道）** | `predict_episode(self, cues)` **只有一个参数**，且元组内每个元素是**同一张量对象的重复** ⇒ 成员每次调用看到的唯一变化是自身历史长度。"B 读 A 的失败证据"不是任务设计能给的，是**能力改动** |

⇒ 硬造 T1/T3 只会产出与机制无关的负结果。**本轮把"手工设计三个候选"替换为
"枚举冻结目标谓词能表达的全部结构并逐格测量"**，同时把三个候选作为**界限**报告。

## 3. 网格：冻结目标谓词的全部可表达结构

目标谓词是两个独立子句的合取（文件内容、语言选择，外加可选 override 标志），
每个子句从给定初始世界只有一条路线 ⇒ **3 × 4 路线网格**，去掉平凡的 `none × none` = **11 个 cell**，
每 cell 2 个 context（共 22 个 context），每 context 跑 11 个因子单元 × 2 次重复。

扩展名、语言证据与内容形状全程恒定（加固轮已证它们是**表面**而非结构）：
内容用 `def run_{i}(): return {i}` —— **真正含 Python 证据**，见 §4.2。

## 4. 有效性门（先于任何测量）

### 4.1 三条准则

`satisfiable`（自身参考步序在**合同之下**达到目标）、`nontrivial`（不是 tick-0 即达标，P5.2b 伪成功形状）、
`contract_admissible`（参考步序不被合同拦截）。**11/11 全部通过，无 cell 被丢弃。**

### 4.2 本轮自查出的缺陷与修正（影响结论，如实落账）

第一版把 `contract_admissible` 算在 `counterfactual._run_scripted` 上，而该函数**直接调 `execute_tool`、
跳过 `policy_for`**——但 `language_evidence_ambiguous` / `language_assessment_unavailable` 这类拦截**只由策略层发出**。
后果：**四个 cell 被错误认证为合法**（三条 `*_mismatch` 与 `patch__observation`），
实测零成功被当成"结构结论"，而实际是"没人能合法执行"。

修法两步：① `validity()` 改为走 `_member_episode` 的同一条路径（bind → intent → `policy_for` → 审批 → `execute_tool`）；
② 从根因消掉歧义：内容改为真含 Python 证据，并让 `mismatch` 路线也带 `user_override=True`
（选择文件本身无证据支持的语言，正是合同无 override 时拒绝的事——那样一个 cell 是"做不到的任务"，不是"机制会失败的结构"）。

**该修正改变了结果**（不是只改了措辞）：

| | 第一版 | 修正后 |
|---|---|---|
| `create__mismatch` 增益 | `0.000`（空洞 cell） | **`+2.000`** |
| 预测↔观测矛盾 cell | 3 个（`none__mismatch`、`patch__observation`、`patch__mismatch`） | **0 个** |
| 被报告为"无法解释"的变化 | `create__none` | 无（重分类为回退效应，见 §6.2） |

**关键完整性说明**：预测函数 `cell_is_combination_only` 在两版之间**一字未改**，只有测量侧被修好 ⇒
第二版"预测与观测在 11 个 cell 上全部一致"是真的通过了一次预注册式对照，而不是事后改预测。

### 4.3 顺序强制性

只有 `create × 语言` 三格 `order_is_mandatory = true`（文件必须先存在才能设定语言）；其余 8 格反转参考步序同样达标。

## 5. 结果表（冻结规则 vs M4，同参照增益）

| cell | 联合必需 | 冻结 | M4 | Δ | 交错 |
|---|---|---|---|---|---|
| `none__observation` / `none__override` / `none__mismatch` | False | 0.000 | 0.000 | 0.000 | 0→0 |
| `create__none` | False | 0.000 | 0.000 | 0.000 | 0→0 |
| **`create__observation`** | **True** | 0.000 | **+2.000** | **+2.000** | **0→2** |
| **`create__override`** | **True** | 0.000 | **+2.000** | **+2.000** | **0→2** |
| **`create__mismatch`** | **True** | 0.000 | **+2.000** | **+2.000** | **0→2** |
| `patch__none` | False | 0.000 | 0.000 | 0.000 | 0→0 |
| `patch__observation` / `patch__override` / `patch__mismatch` | False | **−2.000** | **−2.000** | 0.000 | 0→0 |

- **无 cell 回归**（`m4_regresses_cells = []`）；`patch` 行在两规则下同为 `−2.000`，
  与"仲裁类机制上界恒 ≤ 0"及"`member-d` 单体已覆盖两子句"一致 ⇒ **M4 没有把不可达变可达**。
- **种子**：3 个偏移（0/101/202），三格在**每个偏移**都为正，`reality_ok_everywhere = true`。
- **干预真实性**：11 格在两规则下均无惰性 cell。
- **结果指纹**：11 格收敛到 **3 种**（正 3 格 / 零 5 格 / −2.0 3 格）⇒ 网格确实把结构分开了，
  但也暴露 §7.1 的界限。
- **D5 数字**：三格各自 `required_combination_only_contexts = 2 = available`，`ceiling_gain = 2.0`，
  `max_clearable_reference = 1.85`，**三种候选参照在该格全部 feasible**。

## 6. 交接归因（不靠标签，靠逐步轨迹）

### 6.1 正增益全部由真实交接解释

三格唯一变化的 pair 都是 `member-a + member-c`：`baseline 0.0 → audited 1.0`（Δ +1.0），
其**两个** jointly-required context 的轨迹里**两个成员都执行过**。
`handoff_explained_pair_cells = ["member-a+member-c"]`，`unexplained_changes` 全空。

### 6.2 回退效应与"伪造翻转"被分开记账

pair 成功率变化有两个完全不同的来源。落在**某单体已能解决**的 context 上时，
all-singleton oracle 已经在 `success`，该变化**不可能**移动承重的增益主张——这是阻塞死路成员后的普通回退效应。
本轮把它单列 `fallback_only`（`member-a+member-b`、`member-a+member-c` 各一次），
`create__none` 即属此类（`changed_cells > 0` 而 `gain_delta = 0.0`）。
只有**联合必需 context 上缺少第二执行者**的翻转才算 `unexplained`：**本轮为 0**。

### 6.3 两种冻结失败形态（"复制"的真实宽度）

| cell | 冻结规则下的停止原因 | M4 下的停止原因 |
|---|---|---|
| `create__observation` | 24 × `contract_intercepted:language_assessment_unavailable`、12 × `preview_ValueError`、8 × exhausted，**0 goal** | 24 × `all_members_blocked`、**4 × goal_reached**、12 × `preview_ValueError`、4 × exhausted |
| `create__override` | 16 × `step_cap`、16 × exhausted、12 × `preview_ValueError`，**0 goal** | 24 × `all_members_blocked`、**4 × goal_reached**、12 × `preview_ValueError`、4 × exhausted |
| `create__mismatch` | 同上（`step_cap` 型） | 同上 |

⇒ 冻结规则的失败形状**不同**（在尚不存在的文件上评估语言 ⇒ 被合同拦截 vs 首成员独占直到步数耗尽），
M4 用**同一个修法**（自身步失败即让位）把两者都转成"先创建、后设定语言"。
这是"结构稳健"在本轮能被支持的**最强**形式。

## 7. 界限（D5 必须按此读）

### 7.1 三条正增益格共享同一结果指纹

`create × 语言` 三格的 `(增益, 冻结增益, 联合必需 context 数, 交错数)` **完全相同** ⇒
它们是**一个结构因素经由三条路线**，不是三次独立确认。按加固轮对六变体使用的同一把尺子，
本轮可主张的独立度是 **1 个结构因素（存在性前提）× 3 条路线 × 2 种失败形态**，
而不是"3 个结构"。该界限已由 `outcome_distinctness` 字段与本节同时登记。

### 7.2 残余风险

| # | 风险 | 本轮后 |
|---|---|---|
| N1 | 收益是否只在 `create + override` 一种结构 | **已回答（否）**，但按 §7.1 收窄为"存在性前提 + 语言子句"这一族；**内容轴上只有 1 个正发行** |
| N1a（新） | 内容侧的第二种联合必需结构 | **在冻结 binder 下不可表达**（T1/T3）⇒ 若要主张跨内容结构稳健，须先改 binder，而那是另一项决策，不属 D5 的"改让位规则" |
| N1b（新） | 规模：每格 2 context（本轮共 6 个联合必需 context） | 低于加固轮的 12；落地时应按 create 行扩 context，成本已实测（整轮 ≈ 3.5 分钟） |
| N1c（新） | 种子：3 个（5 个的证据只存在于 `create__override`） | 三格 × 3 偏移全正；若要 5 偏移须重跑（≈ 6 分钟） |
| N2 | `all_members_blocked` 门禁语义 | **仍开放**，且本轮让它更必要：M4 在三格产生 24 次该停止原因，落地时门禁必须知道它意味着"合法让位"而非"失败" |
| 1 | 反序成功 pair 不同 ⇒ 冻结优先级定义 | **仍开放**（本轮 `order_is_mandatory` 只在 create 行为真，与此一致） |
| 5 | 成员族变化须重跑 | **仍开放**（本轮 4 成员族未变） |

## 8. 纪律

- **未实施 M4**：gate / runner / 规则 / 冻结产物一律未改；未注册任务；未训练候选专用模型。
- `frozen_attribute_intact = true`：冻结 `_member_episode` 未被重绑，M4 仍按文档化反事实运行。
- **阈值未放宽**：`+2.000` 对照的仍是未改动的 1.65；新增的两道 fail-closed 门
  （可测联合必需格 < 2 ⇒ 拒绝发布；任何惰性 cell ⇒ 拒绝发布）本轮**均未触发**，它们是为后续编辑准备的。
- 对自己不利的发现如实落账：§4.2 记出**自己写坏的门**及其**改变了结果**的事实，
  并给出修正前后对照；§7.1 主动把"3 个结构"降级为"1 个结构因素 × 3 条路线"。
- **顺序偏离**：加固轮 §10 写的是"先 N2 → 再 N1"。本轮先做 N1，理由是 N1 为只读测量、
  而 N2 是要写进预注册的语义，其内容最好在**看到 `all_members_blocked` 在多大范围内使用之后**再定；
  该偏离是判断而非疏忽，故在此写明，D5 落地顺序仍须由用户确认。
- 临时脚本（可表达性探针、有效性半程检查）全部放在仓库外（`%TEMP%`），`scripts/` 下不留抛弃件。
- 历史冻结文档未被改写：加固轮 §8/§10 只**追加前向指针**。

## 9. 下一步

1. **D5（用户）**：现在证据面为
   **测量可信（伪影审计）+ 规模与种子（加固轮）+ 结构宽度（本轮 N1，按 §7.1 界限读）**；
   落地前仍须 **N2 预注册**（含 §7.2 的 N1b/N1c 扩面）与**风险 1 的优先级定义冻结**。
2. **D1–D4** 仍待确认；**B1 训练在 D1–D5 明确前不启动**。
3. 若决定探索"跨内容结构"：那需要一个 binder 决策（N1a），不在 M4 的范围内，应先立独立议题。
