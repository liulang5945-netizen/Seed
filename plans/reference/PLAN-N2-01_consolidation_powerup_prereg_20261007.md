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
| ⑥ P4 只读面板行 | **部分**：API 面与 harness 侧 workbench section 在场，**面板是否单列 `by_source.workbench_capabilities` 一行未证** | `api/routes_consolidation.py`（status/spec/consolidate）＋`taiji-harness/packages/api/life-controller/src/runtime-client.ts:633 workbenchView()`；**本件实施时核这条，未核前不声称 ⑥ 完成** |
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

- **面**：真实长跑＝活 runtime（默认加载源＝`DEFAULT_CHECKPOINT`＝`checkpoints/seed_a31self_with_circuit.pt`，[seed_runtime.py:47-49](../../api/seed_runtime.py)；**它不等于厂档 `FACTORY_CHECKPOINT`＝`checkpoints/seed_beta.pt`（:52）**，通电面必须点名是哪一份，不许互相代答）＋已入库的 consolidated 语料与 workbench 快照；通电参数（`max_texts`/`max_symbols`/`cycles_per_text`/`max_records`）**随批文写死**，不在跑后调。
- **预算（owner 批后才计）**：1 支通电 pass＋1 支默认参数守卫臂＋巩固前后各 1 张 CAP-0 面＋1 张新旧材料对照面＋1 次回退演示复算；判据级不重跑，机械修复 1 次仅限仪器。
- **前置否证（先跑，不花改权重配额）**：本件 §0 的 ⑥ 面板行核对＋"默认参数 pass 不改权重"守卫臂——两支都是零风险，可在 owner 批通电之前先做。
- **出口交付**（09 §2 N2）：一次真实长跑＋离线巩固的完整读数；巩固后能力增量/单位经验（速率本体读数，附第 0 点对照）；巩固产物按 §8.3 形成候选版本验收后才替换，不覆盖 parent。

## 5. owner 决策点

**批通电长跑**（含改权重半径与算力/时长）＝本件唯一待批项；不批则 N2 停在"材料环已通、加速点①未通电"。§4 前置否证两支不需新批，可在等待期先跑。

## 6. 不变项

M6 收官、M8 挂起；M7-CI 绿只在推送后实测；N1（S1 待批）、N3（甲乙并行已批）、N4/N5（后置）排序不变；§21 信息治理在 N5 真实采料开跑时生效（不可信输入不进巩固）；台账行序以行首标号为准（㊵-419⑥）。
