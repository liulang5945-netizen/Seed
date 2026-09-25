# PLAN-M6-01 · C6 提案：**经验投影的运行时生产者**（含消费者与新适配器）（2026-09-26）

> 归属：M6（Taiji Harness / 产品交付线）· 类别：PLAN（提案，非判决，可修订）
> 来历：2026-09-23「Taiji 原生改造三件事」审计登记的 **C6（P2）**——native 经验投影在运行时无生产者 ⇒ E1 ledger 恒空。
> 状态：**待批**。本件不授权实施；属产品运行时改动，按 A2.4 先例先提案。
> 复核：2026-09-26 对活运行时（PID 24412，`/api/health` 200）只读取证 ＋ 代码面全仓 grep。**前提被部分否证**（见 §2、§3）。

## 1 · 一句话

C6 按原样「把现有三个适配器接上运行时事件」今天会投影出 **0 条**语料单元——真正有料的面（workbench 能力快照）**没有对应适配器**，而三个现成适配器里唯一活着的面（MCP）**声明为空**；且 ledger 在产品侧**也没有消费方**。故本件提议的是「一个新适配器 ＋ 生产者 ＋ 消费者」的最小两端闭环，并以一次零训练投影作为前置否证门。

> **2026-09-26 更新（Z1 已跑）**：否证门的物料判据**通过**（16 条能力全部渲染成功、`invalid_records=0`、`native_trainable=true`），但跑出一个**不利的语言读数**（CJK 占比中位 0.243）与**两块初稿没料到的硬约束**——P1 撞 `EvolutionCorpusArtifact` 的闭合 `source_kind` 白名单，以及**第四块缺口：渲染器**（ledger 给结构化记录，训练器只认 `{"text": …}`）。其后又跑了 **Z1b**（同一支脚本内比三种渲染模板：0.243／0.503／0.630，且三者皆"可训练" ⇒ `native_trainable` 无区分度，模板选择只能按分布贴合判）。见 §3 第 4 条、§4 的更正注、§5 的 Z1/Z1b 读数、§8 重排后的六个决策点。
>
> **2026-09-26 二次更新（Z1c 已跑，决策点 1 有了价格）**：扩 kind 的真实代价被量小、复用路被量堵——① **事件侧 kind 早已含 `workbench`**，关着的只有语料侧那 1 处**行内字面集**；② 「复用 `verified_domain_material`」**在注册表路径上实测不可行**（`register()` 能过、`from_checkpoint()` 抛 `artifact kind mismatch` ⇒ 重启就读不回来），只在绕开 `DeclarativeSourceRegistry` 时成立；③ 未知 kind 的 durable 保护本来就由白名单提供（**整档响亮拒绝、无逐条跳过**）⇒ **`EVOLUTION_CONTRACT_VERSION` 建议维持 1**（升版会把旧档也变成读不回来）。另更正初稿两处：「两处闭合白名单」的说法、以及 P1 的 `source_kind` 取值（命名规则要求 `workbench`/`workbench_artifact`）。见 §5 的 Z1c 与 §4 更正③。

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

## 3 · 缺口是四条，不是一条

1. **生产者缺**：无人构造 registry、无人记生命周期事件、无人调 `project_to_ledger`。
2. **消费者也缺**：ledger 的 `training_view` 无人读 ⇒ 即便生产出来也到不了训练。这正是审计对 C3/C4 的警告「只补一端无效」的同型；**只补生产者会得到一个只写不读的日志**。
3. **物料错配**：现成适配器是 skill／mcp／client-plugin 三类；当前运行时**只有 MCP 面活着且为空**，而真正有料的 **workbench 能力快照没有适配器**。skill 面在 native 侧无运行时来源（技能来自 harness 侧的 loader），client-plugin 面同理属架构重叠。
4. **渲染器缺（2026-09-26 执行 Z1 时才发现，本提案初稿未列）**：`training_view()` 返回的是 `EvolutionCorpusArtifact` **结构化记录**，而原生训练器只认 `{"text": …}` 的 JSONL（`seed/datasets.py:17` 的 `NATIVE_TEXT_FIELD`、`:238` 的 `native_trainable = documents > 0 and invalid_records == 0`）⇒ 「把 ledger 折进语料」这句话**不成立**，中间必须有一个把结构化记录渲染成问/答文本行的模板；而模板本身是设计决定（Z1 实测：照直译渲染出的行 **CJK 占比中位仅 0.243**，英文描述与 `workspace.list` 这类 id 占大头，与中文对话语料分布不合）。

⇒ 因此「照原样接线」是**给空集铺管道**：现成三适配器面对的物料为 0，落地后 ledger 仍恒空，只是多了一条无人读的写路径；而**即便补上 workbench 适配器，也还差一个渲染器**才谈得上「进训练」。

## 4 · 提案（最小两端 ＋ 一个新适配器 ＋ 一个渲染器）

> **Z1／Z1c 期间对本节的三处更正**（2026-09-26，实测所得，初稿写错了）：
> ① **P1 不是「纯新增文件」**——`EvolutionCorpusArtifact.__post_init__`（`taiji/evolution_experience.py:245-251`）对 `source_kind` 是**闭合白名单** `{skill_artifact, mcp_artifact, client_plugin_artifact, verified_domain_material}`，且该 dataclass 受 `EVOLUTION_CONTRACT_VERSION` 守护；`runtime_event_to_experience` 另有一道 `_EVENT_SOURCE_KINDS = {"skill", "mcp", "client_plugin"}`（`seed_platform/evolution_adapters.py:36`）。⇒ 要么**复用 `verified_domain_material`**（语义妥协：它是「已验证领域素材」而非能力声明），要么**扩这两处白名单**——而新增 kind 会随 `EvolutionCorpusArtifact` 进入 ledger 的 `checkpoint()` 载荷，即** durable 数据的新形态**，须同时回答「旧读取遇到未知 kind 怎么办」（§8 决策点⑤）。
> ② **P3 的「折进语料」不成立**——见 §3 第 4 条与 P5。
> ③ **「两处闭合白名单」这个说法是我写错的**（Z1c 实测）：真正的第二处不是与语料侧并列的白名单——`_EVENT_SOURCE_KINDS` 只管**事件**，而事件侧的 kind 集合**本来就有 `workbench`**；语料侧那 4 种是 `__post_init__` 里的**行内字面集**（没有具名常量可扩）。并且**P1 草稿里写的 `source_kind="workbench_capability"` 与 `seed_platform/source_registry.py:239` 的命名规则冲突**：注册表要求语料条目的 kind ＝ `f"{registry.source_kind}_artifact"`，所以适配器若声明 `workbench`，语料 kind 只能是 **`workbench_artifact`**（`workbench_capability` 会在注册表重放时被判 `artifact kind mismatch`）。详见 §5 的 Z1c（C1–C6）。

* **P1 新适配器 `WorkbenchCapabilityAdapter`**（与 `SkillArtifactAdapter` 同族，**`source_kind="workbench"`**，其语料单元 kind 因而必须是 `workbench_artifact`——见 §4 更正③与 §5 Z1c 的 C4）：输入＝`/api/workbench/capabilities` 的快照；输出＝`ArtifactCorpusProjection`，其中 `source_id=snapshot_id`、`source_version=str(revision)`、`source_digest=content_digest(快照)`、`corpus`＝每条 capability 一个语料单元（id／描述／risk／reversible／category／parameters）。**不新造标识**：沿用快照自带的 `snapshot_id` 与 `revision`。
* **P2 生产者挂在哪条缝**——候选：**(a) `seed_platform/sleep_pass.py` 的 project 段（推荐）**：B 已在那里读四个环、已有跨 pass 的 digest 增量状态、已写 `data/consolidated/corpus-*.jsonl`，且它天然是「离线一趟」，不给任何请求路径加开销；(b) 在 `/api/workbench/capabilities` 的读取路径上就地记——**不推荐**（给只读面加写副作用）；(c) 独立轮询器——**不推荐**（与 life-controller 的 5s 轮询重复）。事件语义＝`snapshot_id`/`revision` 变化即一次 `transition()`，首次见到即 `register()`（两者 registry 已有）。
* **P3 消费者（必须与 P2 同批，否则又是只补一端）**：`sleep_pass.project` 把**经 P5 渲染后的** workbench 能力语料行折进当趟 `data/consolidated/corpus-*.jsonl`（与约束种子、journal interaction **同一份产物、同一套 digest 去重**），并在 `next_training_data_spec.json` 的 `datasets` 里点名；ledger 自身按 `checkpoint()`／`from_checkpoint()` 落 `data/evolution/ledger.json`（append-only 事件 ＋ 可重放）。
* **P5 渲染器（Z1 后新增的必要件）**：把 `training_view()` 的结构化记录变成训练器认的 `{"text": "问：…\n答：…"}` 行。三件事必须一起定：**问句/答句模板**、**语言**（Z1 直译读数的 CJK 占比中位仅 0.243 ⇒ 若沿用英文描述就是分布外文本；Z1b 已量三种形态 0.243／0.503／0.630，见 §5，结论是**不含作者中文表就到不了中文主导且保住信息量**）、**上限**（快照 16 条，渲染后中位 197 字／最大 473 字；B 的 `_MAX_TEXT_CHARS` 是 4000，不构成约束，但 16 条对训练的实际权重接近零，需如实说明）。现成先例：B 的 `CONSTRAINT_QUESTION`（约束只能以「答某问」的形态进语料）。
* **P4 面板读数（可选、最小、只读）**：Life 面板「记忆与巩固」分区加一行「经验投影：N 条语料单元／M 条生命周期事件／快照 revision R」，**不加 verb**。

## 5 · 判据与止损（零训练先行，先测再设计）

* **Z1（已执行 2026-09-26：判据通过，但带一条不利读数）**——**执行形式与初稿不同**：原计划「先写 P1 适配器再投影」在 P1 上撞到 §4 的闭合白名单，故 Z1 改用**不碰契约、不碰产品码**的独立诊断 `scripts/training/diag_taiji_c6_workbench_capability_projection.py`（只读 `GET /api/workbench/capabilities`，或从捕获件读；渲染问/答行；写到训练器扫描根**之外**的 `output/c6_probe/`；用 `seed.datasets.inspect_native_dataset` 判定；报告落 `reports/taiji_c6_workbench_capability_projection_20260926.json`）。**读数**：快照 `seed-workbench-contract-v1`／`version=1`／`revision=6`／`snapshot_id=93bd90db…`／16 条能力 ⇒ 渲染 **documents 16、invalid_records 0、`native_trainable=true`**；行长 min 132／中位 197／max 473；**CJK 占比 min 0.101／中位 0.243／max 0.344**。**零污染复核**：`GET /api/train/files` 仍 68 条，不含 `c6_probe`（探针语料没进训练器名册）。⇒ **物料侧成立**（不是 MCP 那种空集），但 **P5 的语言问题是真的**：直译会得到约四分之三为非中文字符的行，按 D 批「分布外输入」的理由不能就这么进语料。原判据（「有没有、可不可训练」）两项皆过，止损不触发。
* **Z1b（已执行 2026-09-26：同一支诊断脚本内加三种渲染形态，用来把决策点 2 变成可判的）**——同一个快照（`revision=6`／16 条），只换模板：`literal` 照直译带英文描述、`no_prose` 去掉英文 `description` 只留结构化字段、`id_only` 只留 `capability_id`。

| 形态 | documents | invalid_records | native_trainable | 行长中位 | CJK 占比中位 |
|---|---|---|---|---|---|
| `literal`（直译） | 16 | 0 | true | 197 | **0.243** |
| `no_prose`（去英文描述） | 16 | 0 | true | 86.5 | **0.503** |
| `id_only`（只投 id） | 16 | 0 | true | 55 | **0.630** |

  ⇒ 三条结论：**英文描述是主要负载**（去掉它 CJK 占比翻倍、行长减半）；**不作者中文释义就到不了中文主导行**（上限 0.63，且行已短到 55 字，信息量随之趋零——只剩 id 与风险等级，模型学不到"这个能力做什么"）；三种形态**都「可训练」**（`invalid_records=0`）——**`native_trainable` 在这个决定上不构成区分度，模板选择只能按分布贴合与信息量来判**。因此决策点 2 不是「调调措辞」，而是要在**作者一张中文能力表**与**承认这 16 条不值得进语料**之间选。
* **Z2**：P2＋P3 落地后，一趟 `POST /api/consolidate` 的产物里**确实含**来自 workbench 能力的语料行，且**第二趟不重复**（跨 pass digest 增量生效），`spec.datasets` 点名该文件。
* **Z3（守卫，双向）**：① `/api/workbench/capabilities` 的响应体**逐字节不变**（只读面不得被写副作用污染）；② ledger 文件缺失时 `sleep_pass` 如实报 `skipped` ＋原因，**不造默认态**；③ 未启用本件时，既有 `sleep_pass` 报告与产物**逐位不变**。
* **止损**：Z1 不过即停；Z2 若与 B 既有投影出现同 digest 双写（重复率 > 0）即停下重设计去重键，不靠"事后去重"糊过去。
* **Z1c（已执行 2026-09-26：把决策点 1 从"感觉贵"变成有价格；零改产品码、零训练、只写自己的报告件）**——新诊断件 `scripts/training/diag_taiji_c6_kind_widening_impact.py`＋报告 `reports/taiji_c6_kind_widening_impact_20260926.json`。六个实测读数：
  * **C1 白名单不对称**：**事件侧已经有 `workbench`**（`taiji/evolution_experience.py:33` 的 `EVOLUTION_EXPERIENCE_SOURCE_KINDS`＝workbench/skill/mcp/client_plugin/user_correction/provider，且 `seed_platform/evolution_ledger.py:249` 的 `workbench_outcome_to_experience` 已在用）；**只有语料侧关着**（4 种：`skill_artifact`/`mcp_artifact`/`client_plugin_artifact`/`verified_domain_material`）。⇒ C6 的"kind 缺口"只在一处，不是契约的两处。
  * **C2 语料侧没有具名常量**：那 4 种是 `__post_init__` 里的**行内字面集**（`:245-251`），事件侧却是导出的 tuple ⇒ 扩一种要改字面量、且无法被测试/别的模块引用同一份事实（这也是"两处白名单"这个说法被我写错的地方：真正的第二处不是并列白名单，见 C4）。
  * **C3 门不止两处**：按模式扫 5 个模块得 **22 处 kind 门**（corpus 侧 5／experience 侧 7／`_EVENT_SOURCE_KINDS` 2／`artifact_internalization._ADMITTED_SOURCE_KINDS` 3／registry 命名规则 2／适配器内 `source_kind ==` 分派 3）。但**在 C6 路径上的只有 4 处**：语料白名单、registry 命名规则、`_EVENT_SOURCE_KINDS`（仅当走通用事件路径；现成的 `workbench_outcome_to_experience` 不经它）、`kind_dispatch`（只影响 skill/mcp/client_plugin 三个 digest 字段，workbench 不需要）。`artifact_internalization` 那 3 处属 P5 语义编码器研究线，**不在 `sleep_pass`→语料的路上**（先前把它算进代价是错的）。
  * **C4 命名规则把"复用 `verified_domain_material`"这条路堵在注册表上**：`seed_platform/source_registry.py:239` 要求注册表内每条语料的 `source_kind == f"{registry.source_kind}_artifact"`。用一个 `source_kind="workbench"` 的桩适配器实跑：**注册成功、重放失败**——`ValueError: source registry artifact kind mismatch`。⇒ 复用方案不是"零契约改动"，它是**跑得起来、进程一重启就读不回来**的那种失败（最坏的一类）。它只在**完全绕开 `DeclarativeSourceRegistry`**（直接进 ledger，放弃 `snapshot_id`/`revision` 生命周期与增量去重）时才成立。
  * **C5 新 kind 的失败是响亮的**：同一支桩把语料 kind 换成 `workbench_artifact` ⇒ **在 `register()` 当场**抛 `unsupported evolution corpus source_kind`，落不到 durable 数据里。
  * **C6 契约版本该不该动——读数说"不必，而且动了更糟"**：把已落盘的 checkpoint 里一条语料的 kind 换成未知值、**重算 checkpoint_digest**，`from_checkpoint` 仍**整档拒绝**（`unsupported evolution corpus source_kind`，无逐条跳过）⇒ **旧构建读新档的保护本来就由 kind 白名单提供，不靠版本号**。反之把 `EVOLUTION_CONTRACT_VERSION` 从 1 升到 2：新构造即抛 `unsupported evolution corpus version`，且**旧档也读不回来了**（`from_payload` 透传 `version`，`:390`）⇒ 升版是把"单向兼容"也一起砍掉，换来的只是一条更清楚的报错。
  * ⇒ **给决策点 1 的结论**：三选一里，"复用 `verified_domain_material`"在注册表路径上**已被量出不可行**（C4），要么改语料白名单（C2 那 1 处行内字面集，且命名规则本来就期待 `workbench_artifact`，见 C4 的正向读法）、要么绕开注册表（放弃生命周期）。**版本号维持 1**。
* **Z1d（已执行 2026-09-26：给决策点 3 一个分母）**——"训练权重≈0"此前是措辞，现在是数：向**在跑的**默认运行时取语料名册 `GET /api/train/files`（68 个文件全部在盘上），逐行累加 `text` 字段得 **2,074,007 行／928,766,482 字符**（最大三档：`simple_zh/simple_zh_texts.jsonl` 478.7M 字符、`p3b_dialogue_subset.jsonl` 195.1M、`simple_zh/shared_core.jsonl` 143.5M）。⇒ 16 条投影行（`literal` ≈3,152 字符）＝名册字符的 **0.00034%**、行数的 **0.00077%**。**但分母要如实双向读**：`sleep_pass` 的消费者目标目录 `data/consolidated/` **此前并不存在**（本次解析 `_consolidation_dir()` 时才被 `get_external_path` 建出，读完即删；建出后 `corpus-*.jsonl` 计数为 **0**）⇒ **四环在默认工作区从未产出过语料档**，而若本件落地后它是那一趟的**唯一产出**，这 16 条在"当趟"里就是 **100%**。所以决策点 3 的真问题不是"占比够不够大"，而是**"这 16 条要不要成为巩固产物的一部分"**——想清楚这一点，(a) 训练权重与 (b) 环的完整性就是两个不同的理由，不能混着说。（复现：`python - <<'PY'` 取 `/api/train/files` 后对 68 个文件逐行 `len(json.loads(line)["text"])` 累加；本报告不新增仪器件。）

## 6 · 明确不声称

* **不声称能力提升**：把能力描述投影进语料只是让「运行时自己声明能做什么」成为可训练素材；native 表层成句率底子仍是 0.03–0.08，本件不解释也不改善它。
* **不声称 MCP／skill 面已通**：MCP 面当前 `records=[]`，skill 与 client-plugin 在 native 侧无运行时来源 ⇒ **本件不接**，等真有声明再接。
* **不声称递归闭环完成**：`next_training_data_spec.json` 仍只是「建议」（C5 已并入 B），是否真拿它训练属所有者动作。
* **不声称经验与回合可相联**：ledger 记的是能力声明的变化，不是「哪一回合用了哪个能力」——后者需要 workbench 的真实调用记录（C1 的 `turn_records` 已记工具名，但两者尚无可连接的键），本件不造这个键。

## 7 · 预算与影响面

| 段 | 内容 | 估时 | 影响面 |
|---|---|---|---|
| Z1＋Z1b＋Z1c（**已做**） | 独立诊断脚本（不碰产品码、不碰契约），Z1b 比三种渲染模板、Z1c 量 kind 白名单的价格 | ≈已完成 | 零；产物在 `output/c6_probe/`（训练器扫描根之外）与 `reports/…20260926.json`（两支报告件） |
| P1 | 新适配器 ＋ `taiji/` 白名单处理 ＋ 合同测试 | ≈半天 | **Z1c 已把这里重定价**：要动的只有**语料侧那 1 处行内字面集**（`taiji/evolution_experience.py:245-251`，无具名常量），**版本号建议维持 1**（未知 kind 的整档拒绝已由白名单本身给出，升版反而读不回旧档）；「复用 `verified_domain_material`」在注册表路径上**实测不可行**（注册能过、重启读不回来），只有绕开 `DeclarativeSourceRegistry` 才成立 |
| P2＋P3＋P5＋Z2/Z3 | `sleep_pass` 接线 ＋ `data/evolution/ledger.json` ＋ 渲染模板 ＋ 守卫 | ≈一天 | **产品运行时改动**（需批）；不动权重、不动 checkpoint。注意量级：当前物料只有 **16 条**（中位 197 字），对 16M 符号级训练的实际权重接近零 |
| P4（可选） | Life 面板一行只读读数 | ≈2 小时 | harness 侧只读面 |

零训练、零权重改动、零 checkpoint 写入；不与 R2 窗口争 CPU（无长任务）。

## 8 · 决策点（需所有者裁定，未批前不动代码）

Z1 已跑（§5）且两条判据皆过 ⇒ 初稿第 4 条「Z1 不过怎么记账」作废。Z1 之后真正的决策是这六条：

1. **kind 白名单**（Z1c 已定价，见 §5 的 C1–C6）：**(a) 扩语料侧那 1 处行内字面集**加 `workbench_artifact`（`EVOLUTION_CONTRACT_VERSION` 建议**维持 1**：未知 kind 的重放保护已由白名单本身提供，升版反而让旧档读不回来）／**(b) 复用 `verified_domain_material`——实测只在「绕开 `DeclarativeSourceRegistry`」时才成立**，走注册表会「注册成功、重启后读不回来」⇒ 选它就得同时放弃 `snapshot_id`/`revision` 生命周期与增量去重／**(c) 不接**（C6 只留诊断，不开工）。**「旧读取遇到未知 kind」不用再另定策略**：现状已是整档响亮拒绝（无逐条跳过），(a) 不改变这一点。
2. **P5 的渲染模板与语言**（Z1b 已把三个候选都量过，见 §5）：**(a) 作者一张中文能力表**（唯一能把 CJK 占比推到中文主导、同时保住信息量的做法，但它是**新写的语料作者工作**，不是投影）／**(b) `no_prose`：只投结构化字段**（CJK 0.503，零作者成本，行内仍是中英混排）／**(c) `id_only`：只投 id**（CJK 0.63 最高但信息量趋零）／**(d) 接受 `literal` 的 0.243 分布外行**（初稿默认，Z1b 后不建议）。注意 (b)(c) 都是"少投一点"换来的占比提升，**不是中文化**。
3. **值不值得现在做**（Z1d 已给分母，见 §5）：16 条 ≈ 名册 207 万行的 **0.00077%**／9.29 亿字符的 **0.00034%** ⇒ 作为**训练权重**可忽略；但 `data/consolidated/` 当前为空 ⇒ 作为**那一趟巩固产物的内容**它是 100%。所以选项实质是：**(a) 为了"数据环第四块补齐＋留接口"而做**（本提案的现实定位，权重不是理由）／**(b) 为了训练效果而做**（Z1d 读数不支持，别按这个理由批）／**(c) 等 workbench 面扩容或真要接进训练管线时再做**。
4. **生产者落点**：`sleep_pass` 的 project 段（推荐）／读取路径就地记／独立轮询器？
5. **消费者是否同批**：P3 与 P2 同批（推荐，否则只补一端）／先只做生产者＋落盘、消费者另批？
6. **P4 面板行**要不要（只读、最小）？
