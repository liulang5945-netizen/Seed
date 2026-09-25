# PLAN-M6-01 · C6 提案：**经验投影的运行时生产者**（含消费者与新适配器）（2026-09-26）

> 归属：M6（Taiji Harness / 产品交付线）· 类别：PLAN（提案，非判决，可修订）
> 来历：2026-09-23「Taiji 原生改造三件事」审计登记的 **C6（P2）**——native 经验投影在运行时无生产者 ⇒ E1 ledger 恒空。
> 状态：**待批**。本件不授权实施；属产品运行时改动，按 A2.4 先例先提案。
> 复核：2026-09-26 对活运行时（PID 24412，`/api/health` 200）只读取证 ＋ 代码面全仓 grep。**前提被部分否证**（见 §2、§3）。

## 1 · 一句话

C6 按原样「把现有三个适配器接上运行时事件」今天会投影出 **0 条**语料单元——真正有料的面（workbench 能力快照）**没有对应适配器**，而三个现成适配器里唯一活着的面（MCP）**声明为空**；且 ledger 在产品侧**也没有消费方**。故本件提议的是「一个新适配器 ＋ 生产者 ＋ 消费者」的最小两端闭环，并以一次零训练投影作为前置否证门。

## 2 · 前提读数（2026-09-26，只读 GET，未改任何状态）

| 面 | 读数 | 含义 |
|---|---|---|
| `GET /api/workbench/capabilities` | 200，`format=seed-workbench-contract-v1`、`revision=6`、`snapshot_id=93bd90db…`、**16 条 capability**（每条含 `capability_id`／`description`／`risk`／`reversible`／`source`／`category`／`parameters`／`enabled`／`semantic_contract`） | **有料**：已是声明式、**内容寻址**、带修订号——正是 `ArtifactCorpusProjection` 想要的形状 |
| `GET /api/mcp-client-capabilities` | 200，`records=[]`、`shadow_validated=[]`、`activation_proposals=[]`、`connection_authorizations=[]`、`connection_targets=[]`；`mcp_registry_snapshot_id=3a9678d2…`，且 `client_capability_snapshot_id` 与 workbench 的 `snapshot_id` **逐字相同** | 面**存在但零声明**；影子注册表是从 workbench 快照派生的 ⇒ 接它今天投影 0 条 |
| 挂载方式 | `api/app.py:226`／`:231` 直接 import（**常开**），不经 `_load_optional_router` | 与 RAG 不同（那面整面 legacy 门控、真机 404，已按清单②降级销账）⇒ 这两面**不是「不存在」** |

代码面（全仓 grep，排除 `taiji-harness/`）：

* `DeclarativeSourceRegistry.project_to_ledger`（`seed_platform/source_registry.py:182`）的调用者**只有** `scripts/training/eval_taiji_p5_1{d,e,f,g}*.py`、`scripts/training/verify_taiji_e2b_source_registry.py`、`tests/taiji_native/test_source_registry_lifecycle.py`；**`api/` 侧零引用**（C6 原判定成立）。
* 三个 registry 子类（`SkillRegistry`／`McpArtifactRegistry`／`ClientPluginRegistry`）在运行时**无人构造**。
* **E1 ledger＝`EvolutionExperienceLedger`**（`seed_platform/evolution_ledger.py:73`，公开面含 `add_corpus`／`admit_corpus`／`append`／`records`／`training_view`／`checkpoint`／`from_checkpoint`）——产品侧**同样无人构造、无人读**，引用面只有离线 eval 脚本。**这是审计没写的一条**：C6 不只是"缺生产者"。

## 3 · 缺口是三条，不是一条

1. **生产者缺**：无人构造 registry、无人记生命周期事件、无人调 `project_to_ledger`。
2. **消费者也缺**：ledger 的 `training_view` 无人读 ⇒ 即便生产出来也到不了训练。这正是审计对 C3/C4 的警告「只补一端无效」的同型；**只补生产者会得到一个只写不读的日志**。
3. **物料错配**：现成适配器是 skill／mcp／client-plugin 三类；当前运行时**只有 MCP 面活着且为空**，而真正有料的 **workbench 能力快照没有适配器**。skill 面在 native 侧无运行时来源（技能来自 harness 侧的 loader），client-plugin 面同理属架构重叠。

⇒ 因此「照原样接线」是**给空集铺管道**：可投影物料为 0，落地后 ledger 仍恒空，只是多了一条无人读的写路径。

## 4 · 提案（最小两端 ＋ 一个新适配器）

* **P1 新适配器 `WorkbenchCapabilityAdapter`**（与 `SkillArtifactAdapter` 同族，`source_kind="workbench_capability"`）：输入＝`/api/workbench/capabilities` 的快照；输出＝`ArtifactCorpusProjection`，其中 `source_id=snapshot_id`、`source_version=str(revision)`、`source_digest=content_digest(快照)`、`corpus`＝每条 capability 一个语料单元（id／描述／risk／reversible／category／parameters）。**不新造标识**：沿用快照自带的 `snapshot_id` 与 `revision`。
* **P2 生产者挂在哪条缝**——候选：**(a) `seed_platform/sleep_pass.py` 的 project 段（推荐）**：B 已在那里读四个环、已有跨 pass 的 digest 增量状态、已写 `data/consolidated/corpus-*.jsonl`，且它天然是「离线一趟」，不给任何请求路径加开销；(b) 在 `/api/workbench/capabilities` 的读取路径上就地记——**不推荐**（给只读面加写副作用）；(c) 独立轮询器——**不推荐**（与 life-controller 的 5s 轮询重复）。事件语义＝`snapshot_id`/`revision` 变化即一次 `transition()`，首次见到即 `register()`（两者 registry 已有）。
* **P3 消费者（必须与 P2 同批，否则又是只补一端）**：`sleep_pass.project` 把 `ledger.training_view(partition="train")` 的语料单元折进当趟 `data/consolidated/corpus-*.jsonl`（与约束种子、journal interaction **同一份产物、同一套 digest 去重**），并在 `next_training_data_spec.json` 的 `datasets` 里点名；ledger 自身按 `checkpoint()`／`from_checkpoint()` 落 `data/evolution/ledger.json`（append-only 事件 ＋ 可重放）。
* **P4 面板读数（可选、最小、只读）**：Life 面板「记忆与巩固」分区加一行「经验投影：N 条语料单元／M 条生命周期事件／快照 revision R」，**不加 verb**。

## 5 · 判据与止损（零训练先行，先测再设计）

* **Z1（前置否证，最便宜，先做）**：只写 P1 ＋ 一个离线脚本，对**当前活运行时**的快照跑一次投影。判据＝`corpus` 单元数 **> 0** 且产物经 `seed.datasets` 判定 `native_trainable=true`（B 已有同款断言可复用）。**不过即停报**：C6 记为「物料不足，与 MCP 面同族降级」，**不铺 P2/P3**。
* **Z2**：P2＋P3 落地后，一趟 `POST /api/consolidate` 的产物里**确实含**来自 workbench 能力的语料行，且**第二趟不重复**（跨 pass digest 增量生效），`spec.datasets` 点名该文件。
* **Z3（守卫，双向）**：① `/api/workbench/capabilities` 的响应体**逐字节不变**（只读面不得被写副作用污染）；② ledger 文件缺失时 `sleep_pass` 如实报 `skipped` ＋原因，**不造默认态**；③ 未启用本件时，既有 `sleep_pass` 报告与产物**逐位不变**。
* **止损**：Z1 不过即停；Z2 若与 B 既有投影出现同 digest 双写（重复率 > 0）即停下重设计去重键，不靠"事后去重"糊过去。

## 6 · 明确不声称

* **不声称能力提升**：把能力描述投影进语料只是让「运行时自己声明能做什么」成为可训练素材；native 表层成句率底子仍是 0.03–0.08，本件不解释也不改善它。
* **不声称 MCP／skill 面已通**：MCP 面当前 `records=[]`，skill 与 client-plugin 在 native 侧无运行时来源 ⇒ **本件不接**，等真有声明再接。
* **不声称递归闭环完成**：`next_training_data_spec.json` 仍只是「建议」（C5 已并入 B），是否真拿它训练属所有者动作。
* **不声称经验与回合可相联**：ledger 记的是能力声明的变化，不是「哪一回合用了哪个能力」——后者需要 workbench 的真实调用记录（C1 的 `turn_records` 已记工具名，但两者尚无可连接的键），本件不造这个键。

## 7 · 预算与影响面

| 段 | 内容 | 估时 | 影响面 |
|---|---|---|---|
| P1＋Z1 | 新适配器 ＋ 离线投影脚本 ＋ 合同测试 | ≈半天 | **纯新增文件**，零产品行为变化 |
| P2＋P3＋Z2/Z3 | `sleep_pass` 接线 ＋ `data/evolution/ledger.json` ＋ 守卫 | ≈一天 | **产品运行时改动**（需批）；不动权重、不动 checkpoint |
| P4（可选） | Life 面板一行只读读数 | ≈2 小时 | harness 侧只读面 |

零训练、零权重改动、零 checkpoint 写入；不与 R2 窗口争 CPU（无长任务）。

## 8 · 决策点（需所有者裁定，未批前不动代码）

1. **生产者落点**：`sleep_pass` 的 project 段（推荐）／读取路径就地记／独立轮询器？
2. **消费者是否同批**：P3 与 P2 同批（推荐，否则只补一端）／先只做生产者＋落盘、消费者另批？
3. **P4 面板行**要不要（只读、最小）？
4. 若 Z1 不过（投影 0 条或不可训练）：按「物料不足降级销账」处理，还是保留 P1 等 workbench 面扩容后再测？
