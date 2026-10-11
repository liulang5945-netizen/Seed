# PLAN-N1-03 回答计算链五环与信息通路五角色的承载设计（V-19／V-11 的对账件；判据先冻、零算力、不动产品码，2026-10-11）

## §0 这件结的是哪两条，为什么值得单独有一件

`09_NEXT_MAINLINE_PLAN.md` §1.3 的普查（㊵-674，仪器 `scripts/training/audit_taiji_vision_clause_traceability.py`）
量出归属 N1 的两条条款**既无条款号引用也无章节锚引用**，逐句核对确认是**确未展开**：

* **V-19（VISION §7.1 回答计算链）**：原文五环＝
  `问题编码 → 当前会话和可见记忆 → 内容选择/必要检索或计算 → 表达解码 → 回答与来源记录`，
  并带一条硬约束"格式检查可拒绝错误输出，**不能把参考答案重新写成模型答案**"；
* **V-11（VISION §18.2 信息通路 Q/E/M/W/R 五角色）**：原文先写"这五个是**逻辑角色**，
  可在同一张量中实现，也可分开；**不要求先建立五个新模块**。若共享存储，仍须能解释来源、更新和消费者"。

N1 的出口交付（09 §2 第 121 行）第三项要的正是"**生产者→实际消费者→更新来源→恢复归属**"那张映射表——
㊵-676 的完成度对照扫全仓，"实际消费者"只命中三处**规则文本**（09／07／VISION），任何 PLAN-N1-* 件里
都没有这张表。本件把两件事一次做掉：**给出五环与五角色在仓内的实际承载点**，并把"这张表存在且逐角色非空"
写成**可判的判据**（J-N1c-2），而不是再抄一遍散文。

**本件零算力、不动 `taiji/`／`seed/`、不新增任何模块或张量。**

## §1 V-19 五环的承载点（逐环给仓内实名；找不到就写未承载，不许造名字）

| 环 | VISION §7.1 原文 | 仓内承载（现读） | 出版它的读数件／键 |
|---|---|---|---|
| ① 问题编码 | 当前请求、合法任务约束 | `train_seed_corpus.py` 的问答接缝编码（`answer_chunking="per-answer"` 走 `问：…\n答：`）与 `--readout` 档位（`predictive`／`action`） | 进度件的 `epoch`／`ticks`；`config` 自述 |
| ② 当前会话与可见记忆 | 实际输入片段＋可回读 | `EpisodicMemoryStore`（`seed`，经 `--episodic-mount` 挂载）＋ `substrate.state.*` | 信封 `episodic_memory` 块（㊵-593 之后先填后落盘）；`unique_documents` |
| ③ 内容选择／必要检索或计算 | 检索与组合 | `taiji/state.py:369 MemoryRecall` ＋ N4 寻址面 | `measure_taiji_n4_addressing_surface.py` 的 `wrong_top1_rate` |
| ④ 表达解码 | 原生输出、停止与行动意图 | `taiji.kernel.state.motor_context`／`motor_probabilities`（臂件里现读存在）＋ 停止信号族（A 线） | 压强面尾行；`developmental_f1_replay.events[*].readout_trace` |
| ⑤ 回答与来源记录 | 格式检查可拒错，不许把参考答成模型答案 | **`answer_source` 键**（`train_seed_corpus.py` 写进信封） | 信封 `answer_source`；`reports/*` 的 verdict 件 |

⇒ 五环**都有承载点**，第⑤环那条硬约束今天**有可判的形式**（J-N1c-3），不是只能靠人说。

## §2 V-11 五角色的映射表（这就是 §18.4 要的那张表；逐格填，空着就是缺）

| 角色 | 生产者（谁写） | 实际消费者（谁读） | 更新来源 | 恢复归属 |
|---|---|---|---|---|
| Q 问题／目标条件 | 语料编码与 `--readout` 档（`train_seed_corpus.py`） | 前向条件与读出分支（`Taiji.observe` 的 `readout=predictive`） | 训练步的预测误差 | 信封 `config`＋`command_surface`（㊵-667 的 `n5_shadow_request` 同族自述） |
| E 当前可见证据 | 逐符号喂入的输入片段 | `substrate.state.*`、K/V 侧读出 | 每 tick 的观察 | 检查点张量（`substrate.*` 与 `taiji.kernel.*` 双份镜像，㊵-662 已定单位） |
| M 跨时经历 | `EpisodicMemoryStore.checkpoint()`（信封 `episodic_memory`） | 检索面 `MemoryRecall` | 巩固 pass（后端 `sleep_pass`；**CLI 跑道不产它**，见 DEBT-G86／夜间件那条） | 信封块；`*.pt.history` 保号存档 |
| W 当前任务信念 | 预测读出与（挂起中的）`WorldDynamicsLearner` 提案 | N4 寻址面、N5 影子支 | 误差与资格迹（`developmental_f1/*eligibility`） | 训练态检查点＋**四维修正装配不入档**（PLAN-N3-03 §4：每次 load 后重施加） |
| R 生成／行动计算状态 | `motor_context`／`motor_probabilities` | 停止与输出侧（A 线 stop 信号族） | 同一误差的反传 | 张量槽同名比较由 `compare_taiji_n5_arm_tensors.py` 出 |

**这张表的性质**：它是 §18.4 要求的登记本身，不是一次测量。它的**存在与逐格非空**由 J-N1c-2 判，
它的**内容**若与代码不符，由 J-N1c-4（下方守卫）在复算时抓住。

## §3 判据（冻；四条，都不新增阈值字面量、都不需要开跑）

* **J-N1c-1 五环都有承载点**：§1 表里每一环必须至少给一个**仓内实名**（文件或键），且该名字能被
  `grep` 现读到；任一环给不出 ⇒ 该条判 `chain_incomplete` 并点名是哪一环，不许用"隐含在模型里"顶过。〔2026-10-11 ㊵-707 执行体已落地＝仪器 `scripts/training/audit_taiji_n1_03_role_table_face.py`＋契约册 `tests/taiji_native/test_n1_06_role_table_face_contract.py`，读数件 `reports/taiji_n1_03_role_table_face_20261011.json` 判 **`chain_complete`**：五环 25 枚实名逐枚要求仓内解析＝强 22／弱 2／解析不到 1（`substrate.state.*`）。**口径限制**：本条把"能被 `grep` 现读到"细化成按强度分级——弱解析（点号路径的模块部分不存在、只有末段命中）单列清单，**既不算承载也不算否证**；判级用的仍是原文那句"至少一个"，故第②环那枚死锚不推翻 `chain_complete`，但它是表里必须看得见的一格缺口。〕
* **J-N1c-2 五角色映射表存在且逐角色非空**：§2 表四个必填列（生产者／实际消费者／更新来源／恢复归属）
  对 Q/E/M/W/R 五格都不为空 ⇒ N1 出口交付第三项由"未兑现"转为**有验收式且已满足登记形式**。〔2026-10-11 ㊵-707 执行体已落地＝同一台仪器，读数判 **`table_registered`**：20/20 格非空（占位符 `—`／`-`／`TBD`／`N/A`／`待填` 一律算空，表解析不出来则响亮失败而非读成"没有缺格"）。**这条判的是登记形式**：填得满不等于说得对——内容与代码不符仍由 J-N1c-4 在复算时抓（§2 末句那句"由 J-N1c-4 抓住"是本条的下游，不是并列的另一件事）。〕
* **J-N1c-3 来源记录不许反写**（VISION §7.1 那句硬约束的可判形式）：任何被采纳的回答，其
  `answer_source` 必须与生成路径的自述一致；若一条回答的 `answer_source` 指向参考侧而产出路径是模型侧
  ⇒ 判 `answer_source_mismatch`，并收回"这条是模型答出来的"这句。**零算力**：只读已存在的件。〔2026-10-11 ㊵-708 执行体已落地＝新仪器 `scripts/training/audit_taiji_answer_source_provenance.py`＋契约册 `tests/taiji_native/test_n1_07_answer_source_provenance_contract.py`（13 passed），读数件 `reports/taiji_answer_source_provenance_20261011.json`。**今天既没判绿也没判红**：JSON 面 0 枚载体 ⇒ `no_carrier_outside_weight_artifacts`——"只读已存在的件"在这张面上**没有可读物**，因为跑道器把 `answer_source` 只写进 `.pt` 信封（`train_seed_corpus.py:547` 的 `attach_metadata(extra=...)`），退出件不带它（已登 DEBT-G112）。同批出版两条独立读数：①强制面＝AST 现取 3 支以 `answer_source` 为条件的 `parser.error` 守卫（1344／1349／1351 行）＋两支**真跑**的负对照各自 rc=2 且错误原文命中 ⇒ "非法组合写不出来"是有行为的；②产品面写入者 0 枚（不复用 ㊵-701 的结论，现取一遍，结果与它的 `read_only_outside_product` 一致）。**取法上最要紧的一条**：仓里有两枚审计件把 `answer_source` 当**数据**登着（`"answer_source": []` 是按键名归的桶），裸按字段名扫会把它们读成"一趟跑的自述"、于是本条白得一个绿 ⇒ 载体谓词要求跑道器的兄弟键（`trainer`／`answer_chunking`／`self_answers_path`…）同在一个对象里，那 2 枚现记为 `name_mental_only_files`。**口径限制**：三枚配方键由同一份 CLI 参数同时写出，故"内部一致"只证记录没被手改／被 patch，不证标签所指那条路真被走过；本件**不开** `.pt`（真载体在那上面，现读 230 枚／3124.4 MB，且 `torch.load` 本轮被权限拦下）。〕
  〔2026-10-11 ㊵-709 就地限定，原文不删：上面那句"退出件不带它"从今天起是**历史性**的——跑道器已把 `answer_source`／`self_answers_path`／`answer_chunking` 写进独立退出件（DEBT-G112 修法①），所以 `J-N1c-3` 的载体面**从下一档真跑起就有读物**；本件不预测那一枚的值，复算仍走同一台仪器。测试面已先接上：`test_g14_trainer_exit_accounting.py` 里 producer 产出的记录会被交给本仪器的 `is_carrier`／`mismatches` 逐值核对（11 passed）。〕
* **J-N1c-4 恢复归属可复核**：§2 "恢复归属"列里点名的每个键，必须能在**信封或退出件**里现读到；
  读不到 ⇒ 该格记 `unverifiable_recovery_claim`（"没读到"不许读成"不需要恢复"）。〔2026-10-11 ㊵-707 执行体已落地＝同一台仪器，读数判 **`recovery_claims_verifiable`**："恢复归属"列点名 3 枚键（`command_surface`／`config`／`n5_shadow_request`），三枚都能走通两条道之一：在 tracked `reports/*.json` 里现读到，或在源码里被写成字典键／构造关键字参数。**取法要点**：信封是**跑道器**落盘的，故写侧扫描面＝`taiji`／`seed`／`scripts/training` 三面，且件里自述为 `write_side_scan_faces`——只扫产品那两目录会把 `command_surface` 与 `n5_shadow_request` 误判成"未取到"（这是我第一版在这两枚键上的 scope 错，已写成契约测 `test_envelope_key_written_only_by_runner_still_verifies`）。**口径限制**：本条证的是"这个键今天在某个件或某段写代码里出现"，**不证**"恢复流程真能把它读回来"；后者要真恢复一次，按 G-N1c-3 记 `unverified`。〕

## §4 守卫（三条，各自能为假）

* **G-N1c-1 不新增模块**：本件不得为凑五角色而造新张量／新对象（VISION 明写"不要求先建立五个新模块"）。
  复算＝本件里所有出现的 `class`／`nn.Parameter` 新增数必须为 0（本件为 0，故满足）。
* **G-N1c-2 每个被点名的键都要有读者**：§1／§2 表里点名的每个键，`grep` 全仓必须同时命中"写"与"读" 〔2026-10-11 ㊵-701 执行体已落地：本条从「人工 grep」变成长驻测＝仪器 `scripts/training/audit_taiji_write_only_key_face.py`＋契约册 `tests/taiji_native/test_n1_04_write_only_key_guard_contract.py`（11 passed），读数件 `reports/taiji_write_only_key_face_20261011.json`（`files_parsed` 1963）。今天判**通过**（`write_only_in_product` 0 枚），但两件事要按读数的形状改写本条：①「有读者」不再是二元判定——仪器分第 2 层薄访问器与第 3 层带动作，并按面分产品／训练／脚本／测试，`read_only_outside_product`（`answer_source`、`unique_documents`）与 `produced_and_used_locally_only`（`expected_reward`，登 DEBT-G111）是两个独立出口，都不许被并进「已消费」或「写-only」；②键名出处已按 ㊵-699 的规矩逐枚钉到**声明文件**（仪器里的 `KEY_ORIGINS`），**不再**声称「本件 §1／§2 点名 13 枚」——现读那件只有 5 枚，其余 8 枚在整棵 `plans/` 树零命中，这句话我自己写过并据它派过工（㊵-701④）。〕 〔2026-10-11 ㊵-706 本条执行体补齐：可达性那一列不再要求人工登记——新仪器 `scripts/training/audit_taiji_default_reachability.py`＋读数件 `reports/taiji_default_reachability_20261011.json`（67 个产品面第三层读取点：`reachable_at_default` 66、`flag_gated` 1，唯一那枚是 `taiji/model.py:968@propose_adaptive_residual_growth_candidate`，门控旗标 `n5_shadow`＝㊵-699 的机器复算）。**对 §2 表的直接影响**：凡在"实际消费者"列写"无"的句子今天都不成立（13 枚键全都有产品面读取点，其中 66 处静态默认位可达）；`write_only` 一栏判通过（0 枚）。**口径限制**：`reachable_at_default`＝静态面上存在非门控调用者，**不等于**运行时必然走到；动态派发与改名调用看不见，所以本条判据只能否证"没人读"，不能证明"每条默认链都在读"。〕
  两处；只写不读 ⇒ 记 `write_only`（这正是 ㊵-570 抓 `last_decision` 那次的形状，也是 DEBT-G68 的形状）。〔2026-10-11 ㊵-699 就地更正：这里的 `last_decision` **不能当 `write_only` 的样板**——它有第三层消费者（生产者 property `taiji/adaptive_residual_growth.py:385` ⇒ 同值别名 property `taiji/model.py:903-908` ⇒ `taiji/model.py:968`，后者住在 `propose_adaptive_residual_growth_candidate()` 里，按该 decision 做四道 `RuntimeError` 前置并取 `evidence_ids`／`resource_cost`）。㊵-570 在它自己那张面上当时是对的（`git show a9d5a2ce9:scripts/training/train_seed_corpus.py` 里 `adaptive_residual_growth_decision` 与 `last_decision` 两串计数都是 0），它说的是「默认训练链上无人消费」，不是「全仓只写不读」——把后者读成本条的样板，就是 ㊵-698 那条越界死讯的成因。本条真正要抓的形状＝**名义上有读者、按可达性等于没有**（DEBT-G68：消费点住在符号流轮询，而 `should_propose` 只活到下一个观测槽之前，两频不对齐时永远错过；㊵-588 实测 D 跑满 60k 出 1 次提议而收尾信封 `n5_shadow` 仍是 null）⇒ 执行体除「写与读都命中」外必须另出**两列**：①读者所在层（别名读者／读了以后带动作的消费者），②该消费者在默认位是否可达。两列缺一，`write_only` 这个词就会既误杀真有读者的键、又放过永不可达的假读者。〕
* **G-N1c-3 本件不开跑、不动产品码**：任何"要跑面才能答"的问题都记 `unverified`，不许为凑判据去起档。

## §5 判据载体在哪（措辞门要求的一节）

J-N1c-1／2／4 的执行体＝本件的两张表＋一条 `grep` 复算（表是判据的**对象**，复算是它的**判定**）；
〔2026-10-11 ㊵-707：这句里的"一条 `grep` 复算"已长成入库仪器 `audit_taiji_n1_03_role_table_face.py`＋契约册 `test_n1_06_role_table_face_contract.py`（8 passed），三条各自的判级词与口径限制见 §3 对应条的带日期括注。`J-N1c-3` 的执行体**仍未仪器化**——本件四条判据到今天只有三条有读数，§6 第一条出口（要四条全过）因此**不成立**，"V-11／V-19 转为已展开且有承载表"这句在补上 J-N1c-3 之前不许发表。〕
J-N1c-3 的执行体＝`train_seed_corpus.py` 已出版的 `answer_source` 键与 `reports/*` 的 verdict 件；
G-N1c-2 的执行体＝全仓 `grep`（读面按版本控制取，见 [[guard-scan-face-git-not-disk]]）。
⇒ 本件没有一条判据是手算的；同时**没有一条需要新算力**。〔2026-10-11 ㊵-708 补：`J-N1c-3` 的执行体也已成仪器，但它今天出的是**"无可读物"**（`no_carrier_outside_weight_artifacts`）而不是判据成立 ⇒ 上面那句"只有三条有读数"现在要改读成"四条里三条判绿、一条无载体"；§6 第一条出口仍不成立，而且**不是因为缺仪器**——是因为来源记录只住在 `.pt` 信封里（DEBT-G112）。〕

## §6 出口（三条互斥）

* 四条判据全过 ⇒ **V-11 与 V-19 转为"已展开且有承载表"**，DEBT-G97 的 unmapped 再降两格，
  且 N1 出口交付第三项（§18.4 映射）第一次有在盘物；
* 任一判据判不成立（如某环无名、某角色 write-only）⇒ **也算"已消化"**（基准要的是展开并判），
  但必须把不成立那一格点名登债，不许静默通过；
* 若只能停在"表填了但复算读不到" ⇒ 本件只算设计件，收回"N1 出口第三项已交付"，
  并登记缺的是哪一格（J-N1c-4 的 `unverifiable_recovery_claim`）。

## §7 落笔即钉的三行

* **预期为绿的成员**：`J-N1c-1`（五环都有实名）、`J-N1c-2`（表已逐格填）、`G-N1c-1`（本件零新增对象）。
  〔2026-10-11 ㊵-707 现读：前两条确为绿（`chain_complete`／`table_registered`），且 `J-N1c-4` 也判绿
  （`recovery_claims_verifiable`）——但 `J-N1c-3` 无读数，所以§6 第一条出口按预注册仍不可宣布。三条绿是**预期兑现**，不是出口兑现。〕
* **仍红时收回的句子并点名**：若 `G-N1c-2` 抓到 write-only ⇒ 收回"这条通路已接线"这句并点名是哪一个键，
  按 DEBT-G68 那族的形状（提议／决策从不发生）继续查消费者；那是**接线债**，不是本件的判据失败。
* **发表资格前置**：任何"N1 已把回答链与角色通路做完"的对外表述必须同时给出 ①两张表逐格非空，
  ②每个被点名键的读写两处 `grep` 命中数，③`answer_source` 的一致性复算结果；三条缺一即视为未交付。〔2026-10-11 ㊵-708：第③条今天**给不出"一致"这个结果**——不是没算，而是算出来是"无可读物"（`no_carrier_outside_weight_artifacts`，见 `reports/taiji_answer_source_provenance_20261011.json`）⇒ 这一条按**未兑现**记，本件的发表资格前置今天不成立。〕

## §8 不变项

M6 收官、M8 挂起；N2 出口两样已齐（㊵-677）、N5 五面已齐（㊵-678）；本件**不新增模块、不新增阈值、
不开跑**；DEBT-G78（世界学习器接进语料训练链）按 owner #24① 继续挂起，本件只在 §2 的 W 行**登记其为挂起态**，
不代为接线。债表现至 **DEBT-G97**。
