# M5.K3 outcome→world 与任务依赖预注册

> 注册日期：2026-09-09。本文只冻结 K3 的问题、边界、指标和停止线；在 canary 运行前不得改判据、扩大执行权限或把 K2 shadow 接入默认 runtime。

## 1. 为什么 K3 必须单独测

当前代码已经存在真实 outcome 的第一段链路：

```text
Workbench 执行
  → WorkbenchOutcome
  → WorkbenchTaijiEvidence.to_taiji_event()
  → TaijiAdapter.record_world_event()
  → Taiji world.events
```

这证明了“外部器官的结果可以进入 Taiji 的世界事件账本”，但还没有证明结果会成为下一步认知的输入。K2 的主循环明确把真实 outcome 用于 S6B 准入和审计，下一步 `current_world` 仍取转移头预测的 `after_world`；因此 K2 不能回答以下问题：

1. 真实成功/失败是否会改写可供下一步使用的世界状态；
2. 下一任务是否必须依赖上一步的 outcome，而不是只依赖预先给出的当前 observation；
3. 删除 outcome 反馈后，后续动作是否按因果预期失效；
4. 事件血缘、checkpoint 恢复和 stale/duplicate outcome 是否仍然 fail-closed。

K3 只测这段缺口，不把“已有事件记录”扩大解释为“已经完成自进化”。

## 2. 可证伪假设

### H1：outcome→world 反馈对依赖链是必要的

在同一个真实、隔离、只读 Workbench 上，A 臂的 outcome-aware full chain 在未见的
`任务分支 × 语言 × 文件` 组合上，应同时满足：

- 真实依赖链 episode success ≥ `0.75`；
- 相对 B（不把 outcome 反馈给模型世界）提升 ≥ `0.25`；
- 相对 C（事件已记录但 outcome→dependency projection 被 lesion）提升 ≥ `0.25`。

### H2：依赖必须携带可验证的事件血缘

每个需要前置结果的后续动作，都必须携带并验证：

- `parent_event_id`；
- 规范化的 `outcome_signature`；
- `dependency_digest`；
- 与当前 Workbench capability snapshot 和 tick 的匹配关系。

缺任一项、重复消费、跨 episode 使用或使用旧 tick 事件，都必须在执行前拒绝，不能靠执行器返回失败来“补测安全性”。

## 3. 真实项目资产与 owner 边界

### 3.1 已有资产

- `SeedRuntime.execute_workbench_intent` 已接收真实 `WorkbenchOutcome`，对 workspace read-only 结果生成 `WorkbenchTaijiEvidence`；
- `TaijiAdapter.record_world_event` 已拥有当前 tick 的事件写入和 affordance 失效责任；
- M5.S6B 已证明成功和失败的真实只读 outcome 可以通过统一 17 维 grounding producer 进入内化准入；
- M5.K2 已证明 transition learner 可以驱动多步预测世界，但明确没有把真实 outcome 回写为下一步世界。

### 3.2 K3 新增 owner，不新增旁路

K3 只允许新增一个 Taiji-owned 的 typed feedback projection（实现时应放在 `taiji/`，而不是 `api/` 或 `seed_platform/`）：

```text
Outcome evidence
  → canonical outcome signature
  → typed outcome/dependency projection
  → next-step world context + dependency witness
  → existing semantic/transition/planner/Workbench boundary
```

- `SeedRuntime` 只负责转发已经产生的 typed evidence，不解释任务依赖，不直接改模型权重；
- `Workbench` 只提供执行结果，不创建 Taiji world transition；
- `record_world_event` 继续是事件账本唯一入口；不得再造第二套 world event store；
- transition learner 仍是预测 owner；真实 outcome projection 是 observed-world/credit owner，不能把观察结果伪装成 prediction；
- K3 canary 阶段 projection 只能作为 shadow owner 使用，`can_promote=false` 固定。

## 4. 输入、输出与任务依赖定义

### 4.1 允许进入 feedback owner 的输入

输入只来自真实隔离 Workbench outcome 的规范化字段：

- capability id、request/intent/call id；
- success、terminal、error code；
- read-only 结果中的稳定 digest、byte length 和确定性结果统计；
- Workbench capability snapshot id/revision；
- Taiji tick 与已存在的 `WorkbenchTaijiEvidence` event id。

原始自然语言、provider 响应、网络内容、MCP 未审计 payload、客户端写入结果和 CUDA 状态不进入本 K3。

### 4.2 输出

feedback owner 必须输出可 checkpoint 的 typed projection，至少包含：

- 观察事件的唯一 `event_id`；
- `outcome_signature`（canonical、content-addressed，不含临时路径）；
- `dependency_digest`；
- `parent_tick` 与 `next_tick`；
- 可供 transition/semantic owner 消费的结构化 dependency facts；
- `accepted/rejected`、拒绝码和 lineage。

它可以丰富同 tick world，也可以产生下一 tick 的 observed-world snapshot；不得直接修改 transition learner 的权重。若 projection 不能保存并 fresh restore 后保持 digest/owner 一致，canary 立即无效。

### 4.3 任务依赖不是静态三步脚本

每个 episode 至少包含三次真实动作：

1. **probe**：对一个训练/holdout target 执行真实 `workspace.read`，产生成功或失败 outcome；
2. **dependent follow-up**：由第 1 步的 outcome class 和 event lineage 决定下一任务分支（例如成功后进入同 target 的语言核验，失败后进入预注册 recovery target）；该分支在第 1 步结束前不得作为模型可见的静态任务计划注入；
3. **verification**：验证依赖分支的结果，要求 parent event 和 dependency digest 仍连续。

分支生成器只定义受控 fixture 的因果规则，不充当模型。模型必须在第 2、3 步消费 feedback owner 产出的 world/dependency context；若把后续 path、capability 或 outcome label 提前写进 observation、goal、content 或外部 `parameter_bindings`，该 cell 判数据泄漏并作废。

为避免把“当前 observation 自带答案”误当反馈能力，K3 的 success 判据额外要求：后续 `ActionIntent` 的审计 lineage 能回指第 1 步真实 event，且 projection lesion 后该 lineage gate 必须拒绝，不允许仅凭当前文件 observation 通过。

## 5. Canary 设计

### 5.1 训练与 holdout

- 训练：至少 6 个三动作 episode；
- dev/test：各 1 个；
- holdout：至少 4 个未见的 `分支 × 语言 × 文件` 组合；
- 训练和 holdout 使用不同 task seed，文件内容 digest 去重；
- 成功和失败 outcome 均须出现，失败使用 M5.S6B 已通过的负 reward 准入规则；
- 所有 workspace 位于进程私有、repo-writable 的临时目录，不能写用户 workspace。

### 5.2 三个臂

| 臂 | world 输入 | dependency projection | 目的 |
|---|---|---|---|
| A full-feedback | 真实 outcome event + typed projection | 开启 | 测完整 outcome→world→依赖链 |
| B no-feedback | 只保留 probe 前 world；真实 outcome 仅审计，不进入模型 world | 关闭 | 测没有结果反馈时的因果差异 |
| C outcome-lesion | event 可见，但 projection 的 dependency facts/lineage gate 被清零 | lesion | 区分“事件被记录”与“事件被认知使用” |

三臂共享同一个 parent、同一训练数据、同一预算、同一 capability snapshot、同一执行工作区和同一 holdout episode；只改变上表中的反馈变量。不得为 B/C 重新训练一个更容易失败的模型。

### 5.3 主指标和技术 Gate

主指标：

- `dependency_chain_success_rate`：三次动作全部 accepted、真实执行成功/按预注册失败分支完成、lineage 连续；
- `A−B` 与 `A−C` 的绝对差；
- `feedback_lineage_admission_rate`；
- 成功/失败 outcome 的 reward variance。

技术 Gate：

- train episodes 上 A 的 chain success = `1.0`；
- feedback projection checkpoint → fresh restore 后 digest、owner 和 lineage 相同；
- stale event、duplicate event、跨 episode event、缺 dependency digest 全部 fail-closed；
- S6B successful/failed outcome admission 全部通过；
- zero/lesion projection 不得仍能通过 dependency lineage gate；
- 只读、无写入、无网络、无 provider/MCP/client/CUDA 变量。

K3 canary 的 `can_promote=false` 固定。只有 canary 通过后才允许另行预注册 3×3 formal；不能因为 canary 通过就接入默认 runtime 或宣称连续成长。

## 6. 停止线与归因顺序

1. A < `0.50`：先检查 projection/fixture/observation leakage、tick 对齐和 planner fail-closed，不调阈值；
2. A 通过但 A−B 或 A−C 不足：判定 outcome feedback 尚未证明必要，保留诊断产物，不扩展任务或接入默认路径；
3. lineage/restore/安全 Gate 任一失败：停止，不把失败改成“仅审计问题”；
4. B/C 仍高成功：优先检查后续分支是否静态泄漏、当前 observation 是否携带答案、projection lesion 是否真正阻断；
5. 失败 outcome 无法准入：回到 S6B policy，不修改 K3 指标来绕过；
6. Windows 临时目录 ACL 或 pytest 基础设施失败：与模型 Gate 分开记录，使用 repo-writable temp 模式复核，不改模型判据。

## 7. 实施顺序

1. 先实现并单测 typed outcome/dependency projection 与 checkpoint/restore/lesion；
2. 再写单 cell canary，复用 K2 的 workspace、semantic、transition、planner 和 S6B helper，不复制既有 owner 逻辑；
3. 先跑静态检查和定向测试，再跑 1 个 canary；
4. canary 通过后才写 formal 预注册和 runner；
5. formal 通过前，K3 保持 shadow，K1/K2 不重算、不改阈值，默认 Taiji runtime 不接入。

## 8. 明确不回答的问题

K3 不证明开放域语言能力、通用规划、Skill/MCP 内化后的智能、自主写入客户端、CUDA 性能或“大模型规模增长”。它只验证一个更基础但不可省略的闭环：**真实行动结果能否以可验证的 typed world/dependency 形式被下一步 Taiji 认知消费，并在移除该反馈时出现可重复的因果退化。**

## 9. 实施记录（2026-09-09）

第一步 projection contract 已落地：

- `taiji/outcome_dependency.py` 新增 `OutcomeDependencySpec`、`OutcomeDependencyProjection` 和 `OutcomeDependencyProjector`；owner 只接受 typed `WorldEvent`，生成 outcome signature、dependency digest、facts、lineage，并在同 tick enrich `WorldState`；
- 复核后明确：动态 `dependency_digest` 只属于 projection/lineage，不进入可泛化 semantic facts；world facts 仅保留稳定的 outcome class、capability、dependency id/task，避免把事件 ID 学成知识；
- checkpoint 带 content digest，projection payload 带 content digest；restore、stale tick、wrong outcome、event identity conflict、duplicate dependency、cross-scope 和 lesion 均 fail-closed；
- `tests/taiji_native/test_m5_k3_outcome_dependency.py`：7/7 通过；K2 transition binding + executive 回归：13/13 通过；ruff、py_compile、mypy 通过；
- `tests/test_workbench_contract.py` 的复核仍被本机 pytest 临时目录 ACL 阻断（5 个测试实际进入执行，43 个在 `tmp_path` setup 阶段失败），没有出现由本次 projection 断言引起的失败；
- projection 尚未接入 `SeedRuntime`、planner 或默认 runtime；`can_promote=false` 不变。

## 10. K3 单 cell canary 执行记录（2026-09-09）

首轮运行没有被当作模型证据保留，按停止线定位为 canary fixture/训练编排错误：

- 原 runner 将 TypeScript 放入 holdout，却没有让 transition corpus 产生对应的类型行；同时训练和 holdout 复用了同一工作区内容；
- 修复后仍发现 runtime 的 workspace root selector 没有随 train/holdout root 切换，导致真实 read 被路由到空工作区；
- 以上两项均不改变 Gate、阈值或 projection 语义，修复后才重新运行正式单 cell。

正式单 cell 已通过：

- runner：`scripts/training/eval_taiji_m5_k3_outcome_dependency.py`；报告：`reports/taiji_m5_k3_outcome_dependency_20260909.json`；
- 训练/holdout 使用两个进程私有 repo-writable workspace，holdout 使用不同 task seed；训练 vocabulary 覆盖 Python、Rust、TypeScript，训练序列覆盖跨语言转移组合；
- K3 仅使用 fixture 内的可用工具链模拟来隔离 outcome dependency，不把开放集语言识别混入本 canary；
- transition 仍使用既有 typed mask 和本地学习规则，只把固定训练预算从 320 增至 1280，以补足新增 outcome rows 后稀疏语言行的收敛预算；未修改 materialization threshold、planner safety gate 或 A/B/C 判据；
- `A-full-feedback` holdout chain `1.0`（4/4），`B-no-feedback` `0.0`，`C-outcome-lesion` `0.0`；`A-B=1.0`、`A-C=1.0`；A 训练 chain `1.0`（6/6）；
- A 的 probe outcome admission `1.0`，feedback lineage admission `1.0`（8/8），reward variance `0.03390739073439137 > 0`；prefit/postfit checkpoint gate、model feedback fact consumption 全部通过；
- 定向回归 `20 passed`；K3 projection 的 mypy、ruff、py_compile 全过。工作台全量回归的 Windows pytest temp ACL 阻断仍是环境问题，未被本 canary 改变；
- 这是 shadow canary 的通过，不是默认 runtime 晋级：projection、transition、planner 均未接入默认路径，`can_promote=false` 继续固定。

formal 的跨 seed 合同已另行冻结：[M5_K3_FORMAL_PREREGISTRATION_20260909.md](M5_K3_FORMAL_PREREGISTRATION_20260909.md)。单 cell 结果不能直接外推为 formal 稳健性，formal runner 必须复用本 canary 的 `run_cell`。
