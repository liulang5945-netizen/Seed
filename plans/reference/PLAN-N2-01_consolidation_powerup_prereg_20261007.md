# PLAN-N2-01 · N2 巩固通电预注册（机器加速点①；2026-10-07 起草，**判据先冻、不开跑**）

> **来源与效力**：[09 规划 §2 N2 推进步骤 3-4](../active/roadmap/09_NEXT_MAINLINE_PLAN.md)（"真实长跑启用 `sleep_pass` 的 organs/learn 路径"→"离线重放→巩固循环取读数"）。C6 六决策点已由 owner 2026-10-07 弹窗按推荐组合批（09 §3.2 第 2 条）。**效力**：冻结假设、读数、判据、守卫与预算。**本件不构成开跑授权**：通电会**改权重**（`learn=True`），半径＝产品默认基座，按既有纪律须 owner 批（09 §3.2 将新增第 6 条）。成熟度＝§13 V1（单机制隔离验证）→ 出口后 V2。

## 0. 现状核对（本会话实测，含一条对规划文本的状态更正）

| C6 决策点 | 落地状态 | 证据（代码锚点） |
|---|---|---|
| ①a kind 白名单 | **已在库**（2026-09-27，M6 时代 C6 P1） | [evolution_experience.py:245-255](../../taiji/evolution_experience.py) 行内字面集含 `workbench_artifact`，`EVOLUTION_CONTRACT_VERSION` 维持 1（加性 kind，旧构建响亮拒绝） |
| ②b no_prose 渲染 | **已在库** | [evolution_adapters.py:666-703](../../seed_platform/evolution_adapters.py) `render_workbench_no_prose`＋`workbench_training_record`（标识符＋中文风险/可逆/类别，英文描述不进训练文本）；`WorkbenchCapabilityAdapter` 每条声明能力一个 affordance 单元（:592-659） |
| ③ 现在做 | 已做（16 条能力已折进语料） | ㊵-458 时代的 C6 P1 收官记录（`reports/taiji_c6_p1_wiring_20260927.json`，Z1 门 pass） |
| ④ 生产者挂 project 段 | **已在库** | [sleep_pass.py:323-363](../../seed_platform/sleep_pass.py) snapshot_id×单元 digest 双闸去重、`by_source.workbench_capabilities` 计数；`run()` 从 `runtime.workbench_environment.capability_snapshot` 注入（:572-586） |
| ⑤ 消费者同批 | **已在库** | `_spec_payload` 把投影语料折进 `spec.datasets`（:435-451），就绪门 `spec_written = ready and bool(spec["datasets"])`（:591） |
| ⑥ P4 只读面板行 | **已核＝缺这一行**（本轮核对完成，不再是"未证"） | python 侧**已经带数据**：`by_source` 进 pass 报告与 manifest（[sleep_pass.py:368/388/403/428](../../seed_platform/sleep_pass.py)）。断在**读取侧的三段**：①`_add_recommendations`（:453-475）只对 `constraints`/`interactions` 两支生成建议，**无 `workbench_capabilities` 支**；②harness 的 `LifePassReportView`（`taiji-harness/packages/api/life-controller/src/types.ts:252-263`）只有 `reason`/`specReason`/`durationMs`/`weaknesses`/`notes`，`LifeConsolidationView`（:266-283）亦无对应字段；③面板 `taiji-harness/packages/client/ui-life/src/client/LifePanel.tsx` 因而无行（`consolidation.status` 已被 `api/routes_consolidation.py` → `runtime-client.ts:50 CONSOLIDATION_PATH` 送到前端）。⇒ ⑥ 的落点＝上面三处＋`ui-life/src/client/locales.ts`（面板文案按 locale-owned 纪律走字典）＋host/client spec 两处 |
| **N2 真正未做的一步** | `organs=True`/`learn=True` **从未在真实长跑启用** | `run()` 默认 `organs=False, learn=False`（:549-551），`sleep_organs` 只在 organs 为真时被调（:598-607）；09 §2 N2 痛点证据同此 |

**状态更正（对 09 与 2026-10-07 交接文）**：两处把"按批准的材料方案扩容（①②④⑤）"列为 N2 待做步骤 2——实测该步骤**已在 2026-09-27 随 C6 P1 落地**。N2 的队首因此直接是步骤 3（通电）＋步骤 4（读数），不重做材料线。

## 1. 假设与读数面（全部用现成仪器，不新抄生成链）

**H-N2（加速点①成立形态）**：离线重放真改权重后，**新收益与旧能力同时过门**（09 §2 N2 判据方向＝§10 判据原型；只测损失下降不算过门）⇒ "经验→能力"通道在窗口外也在涨，且不以牺牲旧能力为代价。

读数面（巩固前／巩固后各取一次，**同一 checkpoint 血缘、同面、同题集**）：

| 读数 | 仪器 | 第 0 点对照 |
|---|---|---|
| CAP-0 整模型能力基线（逐维，含 `not_executed`/`needs_human_review` 分账） | [eval_taiji_cap0_baseline.py](../../scripts/training/eval_taiji_cap0_baseline.py)（冻结集 v2 的 reference；每项独立会话、评测期 `learn=False`、不改冻结产物） | 装机 13/72（X24 面） |
| 复述严格命中 | 既有 A 线严格命中口径（分子分母同集合） | 7/24 |
| 停摆面主列 | 既有 A-2 面（第五台配对读数仪） | 41→59 |
| 资格门控 X | 既有 gated 面 | 0/9/27 |
| **新材料侧**（本 pass 投影的语料） | 由 `sleep_pass` 报告的 `by_source` 行集现取，不搬其它链的阈值 | 无（第 0 点＝不学同材料对照，见 §2） |

**速率本体读数**：Δ能力 ÷（经验单位 × 算力单位）。经验单位＝该 pass 投影行数与符号数（`project` 段已记）；算力单位＝`night` 的 `cycles_per_text × max_symbols × texts`（**与墙钟分开记**，09 §2 N3 报告纪律 §11 同条：训练吞吐与项目推进速度不混）。

## 2. 判据（先于任何读数冻结；合取式）

- **J-N2a（旧能力保持）**＝巩固后 CAP-0 的**严格命中项数 ≥ 巩固前 −1**，且四个对照面（13/72、7/24、41→59、X=0/9/27）无一项**跌破**巩固前同面读数。跌破 ≥2 面 ⇒ 判"通电有代价"。
- **J-N2b（新收益）**＝巩固后对**本 pass 新材料**的读数 ≥ **对照档冻结值**：对照＝同一材料、`learn=False` 的同链预读数（通电前先取一次）；预注册要求写明"对照落在什么值算否定"⇒ **若巩固后 ≤ 对照值 ⇒ J-N2b 不成立**。
- **合取**：J-N2a ∧ J-N2b 才判"加速点①通电成立"。任一不过 ⇒ 按 09 §4 分流第四行归因（**适应后遗忘旧能力 ⇒ 查巩固写入范围与保持集 §8.3 候选版本隔离；必要时退回 parent**），并如实登记负结果，不换次要指标。

## 3. 守卫（四条，任一不符即整件不判）

- **G-N2-1 候选版本不覆盖 parent（§8.3 硬要求，回退路径必须真演示一次）**：通电前取 `Seed.snapshot()`（[seed/model.py:68](../../seed/model.py)；substrate 侧 [taiji/model.py:1784](../../taiji/model.py)），验收不过 ⇒ `restore()`（seed/model.py:222、taiji/model.py:3612）并**复算一次 CAP-0**，要求恢复后的读数与巩固前**逐位同**——这条是"回退路径存在且可用"的实证，不是声明。
- **G-N2-2 默认位不动**：`organs`/`learn` 保持默认 False，通电只经显式入口/参数；跑一支"默认参数 pass"守卫臂，要求 `organs_report.ran=False` 且权重摘要不变。
- **G-N2-3 训练前保存检查**：按 02 §2.2＋§12 快照合同（θ/h/w/A/E/φ/优化器/RNG/课程游标/schema/预算齐备、零步完整保存、独立恢复、原子写入、磁盘预检），任一缺项即不启动改权重。
- **G-N2-4 惊讶度调制须披露**：选择器的特征里 `mean_surprise` 确为第一项且方向为"惊讶 hurts"（[seed/judge.py:25-34,113](../../seed/judge.py)），故必须报**调制前/后被选文本数与选择分数分布**；若仪器给不出调制前的量，本项记 `unverified` 而非默认满足（"惊讶≠应强化记忆"是 §8.3 的约束，不是本件的假设）。

## 4. 面、预算与前置

- **⑥ 面板行＝零风险的已批实施项，可与通电并行做**（owner 2026-10-07 已批"面板行加"，不改权重、不花算力）：按 §0 的三段落点补齐，判据＝面板上出现"本 pass 折算了几条工作台能力"这一行且数字与 `by_source.workbench_capabilities` 一致（`api` 侧 `_add_recommendations` 那支若加，需同时补 `tests/seed/test_sleep_pass.py` 的断言）。
- **面**：真实长跑＝活 runtime（默认加载源＝`DEFAULT_CHECKPOINT`＝`checkpoints/seed_a31self_with_circuit.pt`，[seed_runtime.py:47-49](../../api/seed_runtime.py)；**它不等于厂档 `FACTORY_CHECKPOINT`＝`checkpoints/seed_beta.pt`（:52）**，通电面必须点名是哪一份，不许互相代答）＋已入库的 consolidated 语料与 workbench 快照；通电参数（`max_texts`/`max_symbols`/`cycles_per_text`/`max_records`）**随批文写死**，不在跑后调。
- **预算（owner 批后才计）**：1 支通电 pass＋1 支默认参数守卫臂＋巩固前后各 1 张 CAP-0 面＋1 张新旧材料对照面＋1 次回退演示复算；判据级不重跑，机械修复 1 次仅限仪器。
- **前置否证（先跑，不花改权重配额）**：本件 §0 的 ⑥ 面板行核对＋"默认参数 pass 不改权重"守卫臂——两支都是零风险，可在 owner 批通电之前先做。
- **出口交付**（09 §2 N2）：一次真实长跑＋离线巩固的完整读数；巩固后能力增量/单位经验（速率本体读数，附第 0 点对照）；巩固产物按 §8.3 形成候选版本验收后才替换，不覆盖 parent。

## 4bis. 批文参数（owner 2026-10-08 裁"批：按预注册通电一次，参数写死在批文里"后的落地文本）

- **一次通电的调用形态**＝`sleep_pass.run(runtime=<活 runtime>, reason="n2-powerup-1", organs=True, learn=True)`，
  其余四参数**一律沿用 `run()` 的现行默认**（`cycles_per_text=1`、`max_symbols=64`、`max_texts=8`、`max_records=200`；
  锚点 `seed_platform/sleep_pass.py:545-555`），**不在这一次里放大任何一档**：本轮要答的是"改权重之后新旧能力是否同时过门"，
  先把最小真实档跑通并取数；任何放大都要在看过第一次读数后**另立附件**（同 §16.1 的锚点纪律），不在同一判据里换剂量。
- **执行顺序（前置件先于改权重）**：①`Seed.snapshot()` 取母状态并落盘（G-N2-1 的回退锚点）→
  ②零风险前置否证两支先跑（⑥ 面板行的落点核对与补齐；默认参数 pass 的守卫臂，要求 `organs_report.ran=false` 且权重摘要不变）→
  ③CAP-0 与四个对照面的**巩固前**读一次（同一 checkpoint 血缘、点名 `DEFAULT_CHECKPOINT`）→
  ④通电一次 → ⑤巩固后同面再读一次（J-N2a/J-N2b 的分子分母）→ ⑥**演示一次回退**：`restore()` 母状态后复算 CAP-0，
  要求与巩固前逐位同；这一支不通过 ⇒ 整件不结项（回退路径不可用＝§8.3 候选版本隔离不成立）。
- **不进本件的事**：新旧任务矩阵的放大档、organs/learn 进产品默认位、以及 N5 的唤醒调度——都要另立预注册。

## 4ter. 前置否证的实测结果与对本件的三处更正（2026-10-08，**改权重之前**先跑，零算力风险）

**实测一（回退锚点能否逐位比）＝能**：`checkpoints/seed_a31self_with_circuit.pt`（＝`DEFAULT_CHECKPOINT`，[api/seed_runtime.py:47-49](../../api/seed_runtime.py)）
现场 load→save→load→save→load 三轮，`content_digest(model.checkpoint())` 三次同为 `3e00bafc6298ac84…`，`tick=273`，落盘 12,664,391 字节，单趟 load 2.46 秒。
⇒ 磁盘往返对本链是摘要恒等的，G-N2-1 的"回退后复算与巩固前逐位同"在算术上可达，不是许愿。

**实测二（ring 里还有没有可投影的新料）＝有**：现取 `state.projected=79`、`passes=2`、`workbench_snapshot_id` 已在档；
候选材料＝新约束 **7** 条＋新交互 **1,534** 条（aborted 0）⇒ 本 pass 在 `max_records=200` 下投影得满，`sleep_organs` 不会走 `"no texts to sleep on"` 那支（[sleep_pass.py:513-515](../../seed_platform/sleep_pass.py)）。
同时记一条：`workbench_snapshot_id` 与状态里那枚相同 ⇒ 本 pass 的 `by_source.workbench_capabilities` **预期为 0**，⑥ 面板行的判据"数字与 by_source 一致"要按"0 也对得上"来验，不得为了面板好看去动快照 id。

**更正一（G-N2-1 的取还形状，原文不配对）**：§3 写"`Seed.snapshot()` 取、`restore()` 还"，但 `Seed.snapshot()` 返回 `TaijiState`（[seed/model.py:68-69](../../seed/model.py)），其读者是 `restore_dynamics()`（[taiji/model.py:1800](../../taiji/model.py)）；
`Seed.restore()` 要的是 `Seed.checkpoint()` 那枚 mapping（`format`/`config`/`taiji`/`substrate`，[seed/model.py:211-227](../../seed/model.py)，校验见 :223-227）。
⇒ 本件回退锚点冻结为：`mother = model.checkpoint()` 取、`model.restore(mother)` 还，另有 `SeedRuntime.save(母档)` 落盘为独立文件；两支都要测（还原后摘要相同 **且** 母档 load 后摘要相同）。

**更正二（J-N2b 的对照取法，原文在现行 ring 算术下取不到）**：§2 把对照写成"同一材料、`learn=False` 的同链**预**读数"，
但 `project()` 每 pass 把本轮投影/跳过的 digest 写回 `state["projected"]`（[sleep_pass.py:417-420](../../seed_platform/sleep_pass.py)，去重在 :292-317），
故第二支 pass 必然投影**另外** 200 条；先跑一支 `learn=False` 的 pass 会把材料消耗掉，两支的材料天然不配对。
⇒ 对照冻结为**回退面**：通电后 `restore(mother)` 的权重已由更正一证明与母逐位同，所以在回退档上对**同一枚本 pass 落盘语料**取读数＝"通电前那份状态"的读数；
材料同一性由"同一个语料文件、同一行序"保证，强于"两支 pass 各自的语料"。**判据文本一字不改**（巩固后 > 对照 ⇒ J-N2b 成立；≤ ⇒ 不成立），改的只是对照的取时点，且在此点名。
夜文本身份＝现场用现成仪器重算（`SeedSleepScheduler.select_for_sleep`＋`SeedJudge.score`，[seed/sleep.py:44-56](../../seed/sleep.py)），件内 `organs.texts` 与 `stats.texts` 自述 8 条作交叉核对；这条是**重算不是件内自述**，如实标注。

**更正三（四对照面的动态范围与真实落点）**：第 0 点的四个数字全部出自 **A 线两台仪器**，且都绑另一条链——
`probe_taiji_a30_stop_failure.py` 的冻结命令在库里写的是 `--checkpoint output/a31_chunked_self/checkpoint.pt --circuit output/taiji_r2_copy_circuit_chat/judge/circuit-final.pt`
（[PLAN-N1-01_ADJUDICATION_20261008.md:54](PLAN-N1-01_ADJUDICATION_20261008.md)），13/72、41→59、X=0/9/27 同出这台（资格门控三臂＝`--copy-evidence-injection-mode gated` 配 `--copy-evidence-gate-min-overlap` 阶梯 {0.0,0.2,0.4}，:1136-1150，v40/v41），7/24 出自 `measure_taiji_a30_repetition_penalty.py`（`verdict` 按惩罚档 0.0/0.5/1.0/2.0 分列，件 `taiji_a30_repetition_penalty_20260928.json`）。
⇒ 两条后果：① 第 0 点数字**不得**当 J-N2a 的对照基线（链不同：那条量的是 A 线 a31 件＋外接回路），只作历史参考；J-N2a 的对照＝**同一条命令**把 `--checkpoint` 换成母档 / 候选档的**前后配对**。
② 四张面前后各一次（8 张）×停摆面 ×96 一档 ≈ 本件最大算力项，故本件**分档结项**：先跑满"通电＋回退＋CAP-0 三面＋×24 两面前后"，×96 主列与 gated 三臂留作附件；未取到的面在判据里记 `unverified`，不许拿已取到的面代答（§2 的合取式因此**不结项**，除非全部取到）。

**冻结命令（通电面只认这几行，`--checkpoint` 的 `MOTHER`＝`checkpoints/seed_n2_mother_20261008.pt`、`CAND`＝`checkpoints/seed_n2_candidate_20261008.pt`）**：

| 面 | 命令 | 分子取法 |
|---|---|---|
| CAP-0（巩固前） | `python scripts/training/eval_taiji_cap0_baseline.py --checkpoint checkpoints/seed_a31self_with_circuit.pt --report reports/taiji_n2_cap0_before_20261008.json` | `Σ dimensions[driven].tally.machine_scored_correct`，并逐维抄 `machine_scored_items`/`pending_human_review_items`（人审未结的项**不进**分子，如实分账） |
| CAP-0（巩固后） | 同上，`--checkpoint` 换 `CAND`，`--report` 换 `taiji_n2_cap0_after_20261008.json` | 同上 |
| CAP-0（回退复算） | 同上，`--checkpoint checkpoints/seed_n2_rollback_20261008.pt`，`--report taiji_n2_cap0_rollback_20261008.json` | G-N2-1：与"巩固前"件 `--subtree per_item` 逐位同 ⇒ `identical=true` |
| 装机自停 ×24 | `python scripts/training/probe_taiji_a30_stop_failure.py --checkpoint <档> --limit 24 --out-report reports/taiji_n2_stop24_{before,after}_20261008.json` | 判读器从件里现取，取不到键即 rc=2 fail-closed |
| 复述严格命中 ×24 | `python scripts/training/measure_taiji_a30_repetition_penalty.py --checkpoint <档> --limit 24 --out-report reports/taiji_n2_replay24_{before,after}_20261008.json` | 同上（`verdict` 各惩罚档的严格命中数，档名从件里读） |
| 通电本体 | `python scripts/training/run_taiji_n2_powerup.py --phase guard` 再 `--phase powerup` | `reports/taiji_n2_guard_default_20261008.json`／`reports/taiji_n2_powerup_20261008.json` 自述 |

**G-N2-3 的覆盖半径（不假装满足）**：运行时信封不是训练器的 §12 全字段合同；本件实测的是"母档落盘→独立 load→摘要逐位同"这一条，
`优化器/RNG/课程游标/预算` 四件套在 `SeedRuntime.save` 的信封里没有对应字段 ⇒ 记 `unverified`，不得写成"快照合同已满足"。

**成本预算（本轮实测外推，非猜）**：CAP-0 一张面＝100 项 / 147 轮 / 模型内 83.4–163.9 秒（三份入库件 `taiji_cap0_baseline_{v1_20260917,repro_20260918,postmigration_20260918}.json` 自述）
＋每项一次子进程 load（2.46 秒）⇒ **约 6–10 分钟/张**，三张 ≈ 20–30 分钟；×24 两面前后四张，停摆面单张历史耗时约 10 分钟（×96 档，见 PLAN-N1-01 判读件），×24 档按其预算线性取下界；通电本体＝判定 200 条投影＋8 条夜文本，分钟级。

## 5. owner 决策点

**批通电长跑**（含改权重半径与算力/时长）＝本件唯一待批项；不批则 N2 停在"材料环已通、加速点①未通电"。§4 前置否证两支不需新批，可在等待期先跑。

## 6. 不变项

M6 收官、M8 挂起；M7-CI 绿只在推送后实测；N1（S1 待批）、N3（甲乙并行已批）、N4/N5（后置）排序不变；§21 信息治理在 N5 真实采料开跑时生效（不可信输入不进巩固）；台账行序以行首标号为准（㊵-419⑥）。
