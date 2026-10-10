# Seed / Taiji 技术债登记册

## 收束时仍未闭合（2026-09-17）

**后继修订（2026-09-18）**：下列为收束时快照。36a7b861已提交P3b-v2完整报告及停止结案，不再以“报告不存在”为阻塞；报告实际target geometry仍与结案byte-aligned叙述不符，训练唯一episode覆盖也有限，当前归因边界见[03 §4](03_CURRENT_EXECUTION.md)。后继D1仪器已完成，D2有限A切片已实施并触发路线复审，研究状态以03为准。本轮继续补全开发指导，不重训旧候选、不立项联合任务族、不先清完全部技术债。技术债只有实际阻塞选定开发包时才进入其交付范围。

- P3b-v2：现有checkpoint与当前run身份不同、目标格式偏离文字合同、完整报告缺失；收束中断不等于实验失败裁决。先只读核验，不扩大训练。
- 本轮较大范围pytest在`outputs/pytest_h37b_20260917`遇WinError 5及清理异常；局部18项通过不能替代该套件或当前HEAD全量CI。未删除目录、未批量修改权限。
- 权威快照见[项目收束记录](../../reference/PROJECT_CONSOLIDATION_20260917.md)；其余条目保留历史范围，不补发绿灯。

**DEBT-I7 的第二实例（2026-09-17 定位；根因已于 09-19 结清，见本条末尾 `8a7966b4`）**：除
`checkpoints/seed_corpus.pt` 外，测试曾往**共享非临时目录** `output/manual-r5-canary/`
写中间产物且**从不清理**。

- **实测**：该目录现存 **1054 个残留**（`s29-*` 到 `s51-*`，按 PID 分组；仅 `s45-*` 就有 215 个）。
- **涉及测试至少 7 个**：`test_runtime_artifact_store_audit_projection` / `..._bridge` /
  `..._preflight` / `..._runtime_reconciliation` / `test_runtime_retention_store_audit` /
  `test_structural_artifact_measurement_sidecar` / `test_structural_artifact_store`，
  统一以 `Path(__file__).resolve().parents[2] / "output" / "manual-r5-canary" / f"sNN-store-{os.getpid()}"`
  作为 store 根，并把中间 `.pt` 放在**同一父目录**下。
- **后果（已实测）**：全量套件里 5 项 `runtime/structural artifact store` 测试**单独跑全过、
  全量跑失败**（`unexpected file: invalid-name.json`、orphan 集合多出 `external_orphan`、
  `assert 2 == 1`、`DID NOT RAISE ValueError`）⇒ **是环境残留而非代码缺陷**，
  但**使"全量测试"这一最基本的验收手段不可靠**（有掩盖真失败的风险）。
- **已处置（2026-09-17，归档而非删除）**：1052 个残留条目（31.4 GiB）已**移动**到
  `output/_archived_manual-r5-canary_20260917/` —— **可随时恢复**；原目录只保留
  `README.md`（**git 跟踪，必须保留**）与 `native-canary.pt`。
  处置后复跑 8 个相关测试**全过**（含此前"单独过、全量失败"的 5 项）⇒ 假失败消除。
- **⚠️ 根因仍未修**：那 7 个测试依然把 `store_root` 指向该共享目录 ⇒ **会再次累积**。
  建议的根治（改 pytest `tmp_path`，或 teardown 清理本次 PID 的中间文件）**待实施**。
  **下次清空前必须重新核查**该目录（通过标准：只剩 `README.md` 与 `native-canary.pt`）。
- **DEBT-I7 第二例：根因已修（2026-09-19，`8a7966b4`）**，上面"待实施"就此关闭。做法**不是** tmp_path 也不是
  teardown 清理，而是**让测试根本不拼那个路径**：18 个工件测试的根改为系统临时目录
  （`tests/_scratch.py:artifact_scratch_root()`，保持 `<root>/<sNN-kind-<pid>>` 的形状，所以测试里
  `store_root.parent/...` 的兄弟路径与清理循环原样可用），并**撤掉**会话末 sweep 兜底
  （`tests/_canary_sweep.py` 与它的契约测试已删）—— 那套兜底只覆盖"跑到 teardown"的进程，
  而残留恰恰来自被终止的进程（实测 12 文件 / 496 MiB，6 个已死 pid；今天手工清掉过一次，
  **不清就会按这个速率继续长**）。
  替代清理的是**预防**：`tests/taiji_native/test_artifact_store_scratch_contract.py` 四条守卫 ——
  ① 静态扫 `tests/` 不许再拼该路径（**先跑出红：19 处含 conftest，改完转绿**；唯一具名豁免是
  只读方 `_scratch.py`，第 ④ 条把"豁免面只有一个文件"也钉住）；② scratch 根必须在仓库外；
  ③ 产品目录只应剩 `README.md` + `native-canary.pt`。验证：改写后的 24 个测试文件合跑
  **62 passed / 0 failed（107.5 s）**，ruff/black 干净；跑完后产品目录确实只剩那 2 项、临时目录为空。
  顺带更正本条上方的两处旧数字：涨速不是"335→500 MB / 四次全量"的线性外推，而是**每次被终止的运行
  各留一对 ~43 MB**（12 个 = 6 pid × 2 文件），sweep 只能清活到 teardown 的那些。
- **✅ 根因的第二半已处置（2026-09-18 第五批，采用上面建议的第二种：teardown 清理本次 PID）**：
  累积确实还在发生 —— 今天四次全量之间该目录从 **10 项 / 335 MB 涨到 14 项 / 500 MB**
  （新增的都是 `s45-active-<pid>.pt` / `s45-terminal-scheduled-<pid>.pt`，每个 PID 一对）。
  落地方式不是改那 7 个测试，而是利用它们**已经把 pid 写进文件名**这一点：
  `tests/_canary_sweep.py:sweep(dir, pid)` + `tests/conftest.py` 的会话级 autouse teardown，
  只删"名字末尾是本进程 pid（可选 `.pt`）"的条目 ⇒ 别的会话正在用的文件、`README.md`、
  `native-canary.pt`、以及 09-17 归档的 1052 项都不在删除面内；形状不认识的后缀（`.bak`）宁可留下。
  正反两向由 `tests/taiji_native/test_canary_residue_sweep_contract.py` 钉住
  （在 tmp 目录上跑，不碰真目录）：`2 passed`。
  **未做**：把那 7 个测试改到 `tmp_path`（更彻底，但要动 7 个文件的公共夹具；现在这条已足以止涨）。

## 最新状态补充（2026-09-15，WP-3 落地后：仪器语义债）

- **DEBT-I1（本轮已修）A/B 仪器的基线臂会在被测改动落地后静默变成被测臂**。结构空间探针以
  `frozen._member_episode` 作基线、以反事实副本作对照臂；M4 落地后两者是**同一份源码**，于是
  `gain_delta ≡ 0`、"`regresses: none`" 成为同义反复，而**当时它被记成了"出口②已过"**。
  修法（同批）：`build_reverted()` 反向还原 revision-0 基线臂（部分落地即 `SystemExit`）、按 gate 自报
  `RULE_REVISION` 选臂、**两臂同一函数即 fail-closed**、报告新增 `arm_provenance`、与封存报告逐字段对照入测试。
  **通用形态**：任何"改完再重跑一次对照"的计划条目，都必须先写明**基线从哪里来**（当前源码 / 封存报告 / 显式还原）。
- **DEBT-I2（本轮已修）默认输出指向被 sha256 封存的报告**。gate 早前已版本化，但**另外六支** B0 仪器的
  `DEFAULT_OUTPUT` 仍等于封存文件名，而写入是原地 replace ⇒ 忘记 `--output` 一次即毁掉本轮基线。
  已全部改为 `REVISION_0_OUTPUT`（只读）+ 新默认名，并由
  `test_no_instrument_defaults_its_output_onto_sealed_evidence` 全仓扫描 `DEFAULT_(OUTPUT|REPORT)` 赋值把关。
- **DEBT-I3（只登记，不处置）revision-1 下没有"多修法对照"仪器**。加固扫描逐一构造六变体，其中四个的锚点
  已随规则落地而消失 ⇒ 它现在拒绝运行。后果：**"其他修法（M1a/M1b/M2a/M3）在规则 1 下仍不如 M4"这一命题
  目前只有 revision-0 归档证据，没有规则 1 下的机器证据**。WP-6（改 binder）若需要该主张，须先用
  `build_reverted()` 的能力造出"以 revision-1 为基线"的变体扫描；**本轮不做**（主线轮内登记优先）。
- **DEBT-I4（只登记，不处置）A/F/H 三个维度没有任何仪器覆盖**。`eval_taiji_cap0_baseline.py` 把这三整维
  记成 `status = not_executed`（分别定义 6/4/6 项，按 07 §5 不记 0 也不记通过——这个"不冒充"是对的）。
  后果：**P3b 判据 J4 原文的"A / H 不退化"分支在本链路上无法判定**，任何"A 维/H 维"结论都没有仪器支撑；
  结项文档该分支只能记 `untested`，**禁止**写成"未见退化故通过"。修法：另立 runner 覆盖 A（真实性/来源/
  拒绝类）、F（项目代表能力）、H（性能与稳定性）三类流程性检查；**本轮不做**（改 runner 会打断在跑双臂
  campaign 的评测链路）。
- **DEBT-I4 ◐部分处置（2026-09-18 第六批）—— 缺的不是仪器，是接线**：`--health` 那条 runner **早就存在**
  （`run_health()` 产 `taiji_cap0_health_v1_20260915.json`，含 A01/A01-tick/A02/A03/A04/A06 + H05 布尔判定、
  `stability_runs=30 / crashes=0`、以及 `gate_status: to_be_calibrated`），缺的是"没人把它接进判据"。
  现在接上了：`check_p3b_criteria` 新增 `judge_health()` + `--baseline-health/--candidate-health`，
  J4 的 A/H 支按 07 §4.2 的**布尔**要求逐项链判，三态 `pass / fail / not_supplied`，
  外加一条反错配守卫（两份健康报告的 `checkpoint` 不同 ⇒ `source_mismatch` 拒判——
  与双臂配对那次是同一类失效）。**"没给报告"永远不算通过**：它进 `untested_clauses`，
  所以 `verdict: pass` 不会被读成"A/H 也过了"。`A05_isolated_ablation`（报告里是 `null`）
  刻意排除在必过项之外，并有测试钉住"清单 == 仪器真产出的非空字段集合"，防止生产端悄悄少判一项。
  **（同日第七批更正：A05 已执行并转入必过项，上面那句只描述当时的状态，见下方 A05 条。）**
  实测：`43 passed in 1.25 s`；`--health` 单次墙钟 **17.2 s**（本轮计时），
  ⇒ 未来 campaign 若每阶段都跑健康支，代价是 +17 s/阶段（对照 CAP-0 阶段本身的 345.7 s 是 +5%）。
  **仍开的两半**：① H 的响应/内存**阈值**门按 §4.2 要求"须按目标设备预检标定后冻结"，
  本 runner 不设阈值 ⇒ 那半支永久 `untested` 直到有人标定（是用户/CI 的设备决策，不是我能代做的）；
  ② F 维仍只有"合同引用"，没有任何执行 ⇒ J4/07 §5 里 F 那一支依旧不能声称判过。
- **DEBT-I4 的第二半：配对守卫写反了，已改；并接进战役驱动（同日第六批续）**：
  第一版 `judge_health` 比较的是"两份健康报告彼此的 checkpoint 是否相同"。这句**方向双重错误**：
  真实战役本来就是同一模型的两个 tick ⇒ 会把每一次合法比较都拒掉；
  而它**没检查**真正该检查的东西 —— 健康报告与它旁边那份评价报告是否同一检查点。
  今天盘上就是这种错配的活例：唯一一份 09-15 健康报告来自 `seed_corpus.pt`，
  而 CAP-0 评价报告来自 `seed_beta.pt`（旧守卫会放它过去）。
  改成逐侧配对（`baseline` 与 `candidate` 各比一次），并补一支测试专门钉"用今天那两份真实文件去配，
  必须报 mismatch"。另加 `test_the_p3a_health_sample_pairs_with_the_p3a_baseline` 钉住
  `P3A_HEALTH` 与 `P3A_BASELINE` 同检查点 —— 换掉任一份都会红，而不是让每场战役的 A/H 支
  静默退化成 `source_mismatch`（mismatch 不算通过，那等于整支 J4 白缺着）。
  驱动侧：每阶段多跑一次 `--health`（实测 17.2 s，对照 CAP-0 阶段 345.7 s），
  阶段行记 `health_report / health_checks / health_seconds`，criteria 调用带两份健康路径；
  子进程非零退出 ⇒ `SystemExit("stage health probe failed…")`，不允许"没证据也继续"。
  实测：本批 5 个合同文件合跑 `114 passed in 2.89 s`；ruff/black 干净。
- **DEBT-I4 的 A05 半支：已执行（2026-09-18 第七批）—— 结论是"原始输出参数驱动，表层回答不是"**：
  `A05_isolated_ablation` 从 `null` 变成实测布尔
  （`reports/taiji_cap0_health_v4_seedbeta_20260918.json`）。**改载荷再 restore 这条路走不通**：
  `Taiji.restore` 会重算核心摘要并核对身份器官的 lineage，任何一位权重被改都触发
  `ValueError: identity organ checkpoint lineage does not match Taiji core` ⇒ 消融只能在
  **进程内已加载的副本**上做（检查点文件从不写回；实测前后 sha `ad2a06465e0e` 未变）。
  五个靶点（属性路径逐条来自实测的活对象图，不拼名字）在 16M-tick `seed_beta.pt` 上：
  **F1 读出** `predictive_readout.synapses.edge_weight`（abs_sum 5076.24）与 `.bias`（72.851）、
  **fabric** `decoders[0].edge_weight`（493.35）⇒ 原始字节流改变；**F4 运动**
  `motor.synapses.edge_weight`、**记忆** `memory.cue_encoder.edge_weight`（1099.26）⇒ 不改变。
  （按 07 §4.3："输出不变不自动断言整个模型无效，应核查该题是否触发对应功能" —— 后两个靶点只说明
  **这条题面未触发它们**，要断言"哪个面承载什么"须换题面做正对照，本轮未做，不得写成"运动面/记忆无效"。）
  控制项三条：基线自洽（同题连调两次摘要相同）、全部靶点跑完后复测基线一致
  （`restoration_verified: true`）、`SeedRuntime.load` 之后 `restore(checkpoint())` 是输出恒等操作。
  ⇒ A 支按 07 §2 L1（只要求**原始**输出参数驱动）通过。`A05b_answer_follows_parameters` 五个靶点
  全 False：可读性闸门 `_readable_surface` 因字节流含 U+FFFD 而拒收 `native_prediction`，退回固定模板
  ⇒ 按 07 §3 A 行"不能把纯规则输出归因模型"，**聊天回答不得记为参数驱动**，F04 仍是缺口（gate 文本已改）。
- **DEBT-I4 的链路错配（同日第八批，A05 落地后立刻测出来的）**：上面那句"聊天回答不得记为参数驱动"
  **只对裸链路成立**。同一条题面、同一批靶点，装上 `constrained_decode` 之后重测：原始字节可解码
  （不再有 U+FFFD）⇒ 闸门放行 ⇒ **回答随消融改变（A05b = True）**，回答内容变成模型自己吐的伪汉字
  （"怀怀怀…"/"刈專刈…"，仍非成句汉语）。而 P3b 的**分数**全部取自
  `relax_legacy_guard + constrained_decode` 这条链路的报告，`--health` 却跑在裸链路上
  —— ⇒ J4 的 A/H 支一直在判**另一条链路**的性质，和 DEBT-I5/I6 是同一类（读数与它要描述的分数不同链路/不同面）。
  修法三条，都已落地：① `run_health` 接链路开关、子进程内装与评价支相同的补丁，并把
  `chain` + `identity{git_head, checkpoint_sha256}` 写进报告，**格式号 v1→v2**（不同链路会给出相反的 A05b，
  并排读两份 v1 会得出错误结论，所以必须留痕）；② 战役驱动的 `--health` 调用带上同两个开关；
  ③ `judge_health` 在读任何字段**之前**先要 `chain == REQUIRED_CHAIN`，否则
  `chain_mismatch`（不算通过、进 `untested_clauses`）。缺 `chain` 字段的旧报告（v1..v4）一律拒 ——
  失败封闭是故意的：因此战役基线那份健康报告换新文件
  `reports/taiji_cap0_health_v5_seedbeta_constrained_20260918.json`（`P3A_HEALTH` 已改指）。
  测试两个方向都钉：错链路拒、无 chain 字段拒、**对链路必须不拦事**（否则整支 A/H 被永久判死）。
  另按 07 §4.3 补做正对照：**换五条题面**（含两条必须用上前文历史的）重跑同一批靶点，
  移动的仍是同样三个靶点（读出权重/偏置、fabric 解码器），`motor` 与 `memory.cue_encoder`
  在五条题面下都不动 ⇒ "未触发"不再是单题面的偶然，但仍不得写成"该面无效"（§4.3）。
- **A05b 的取舍随实测翻转（同日第九批）**：上一批把 `A05b_answer_follows_parameters` 排除在必过项
  之外的理由是"盘上每个检查点它都是 False ⇒ 判它等于让每条 campaign 必红"。**这个前提被 v5 推翻**：
  同一检查点、同一批靶点，在 required 链路上 `A05b = True`
  （`taiji_cap0_health_v5_seedbeta_constrained_20260918.json`，identity `git_head 63c38846` /
  `checkpoint_sha256 ad2a06465e0e`）。它可满足，而且它是**唯一真正在判** 07 §3 A 行
  "不能把纯规则输出归因模型"的那一道 ⇒ 现已放进 `A_HEALTH_CHECKS`，并删掉原先那个
  "只披露不判"的 `answer_surface` 回声块（判了就不需要旁路披露）。
  教训与前面那条同型：一个"看起来永远不满足"的检查项，先问是不是在**错误的链路/错误的面**上测的，
  再决定豁免还是修接线；豁免一条恰好管事的断言，等于把该断言要防的失效放行。
  ⚠️ v5 的 H 计时（H03 27.2 s、A05 11.1 s）是在全量套并发时取的，**不是**标定级读数
  ⇒ 任务 #34（H 阈值门）必须在空闲机器上重测，不得引用 v5 的这些数。
  证据：`37f665fc` 提交后，受影响的四个合同文件族（cap0 基线 / 战役驱动 / 清单 / 旧链路加载）
  合跑 **`113 passed / 0 failed + 1 xfailed`（50.9 s）**，ruff/black 干净；
  其后两轮更大规模的复采都已跑完并核对：目录级 `tests/taiji_native`
  **`1333 / 0 失败 / 1 跳过 + 1 xfailed`（1321.03 s，rc=0）**，全量套
  **`1763 / 0 失败 / 6 跳过 + 1 xfailed`（1417.43 s，rc=0）**，且全量套跑前跑后
  `checkpoints/seed_corpus.pt` 仍是 `c8025db44c65` / 43,223,183 B。
  ⚠️ 同一轮里 `output/manual-r5-canary/` 条目数 **16 → 14**：方向是"不涨"（DEBT-I7 的隔离结论不受影响），
  但**降幅不由这次运行解释**（会话末 sweep 只删本 PID 写的那一对），**未归因** ⇒ 不得据此认为
  "累积问题已缓解"，那 7 个测试仍把 `store_root` 指向共享目录、根因未修。
  取舍：A05 进 `A_HEALTH_CHECKS`（必过项），A05b 刻意**不**进 —— 盘上每个检查点它都是 False，
  判它等于让每条 campaign 必红、判据失去判别力；改为随 verdict 读出
  （`health.answer_surface.counts_toward_status: false`）并追加一条 `untested_clauses`。
  **（同日第九批更正：这个前提在 required 链路上不成立，A05b 已转入必过项、`answer_surface` 块删除，
  见上一条 A05b 条。）**
  同批把 J4 的首个布尔支（G 硬安全失败数须为 0）也补进 `untested_clauses` —— 此前只在
  `does_not_cover` 里，`verdict: pass` 的读者看不到。
  代价实测：A05 五靶点 **8.35 s** ⇒ `--health` 单次从 17.2 s 涨到约 25.6 s（对照阶段评价 345.7 s 是 +7%）。
  **顺带两条定位（新事实，本轮不展开）**：① `motor.synapses.edge_weight` 与
  `predictive_readout.synapses.edge_weight` 在这份 16M-tick 检查点里**逐字节相同**
  （`bitwise_equal=True`，各自独立 storage）⇒ 两者至多其一在收梯度，谁在收未查；
  ② 16M tick 后 `predictive_context.recurrent`、`memory.association`、全部 `memory.*_readout`、
  `identity_organ`、`fabric.consolidation_decoders[*]`、`executive` 的权重 abs_sum **仍为 0**
  ⇒ 这些面从未被写入。
- **DEBT-I4 的 F 半支：已处置（2026-09-20，第十批）—— F 维此前是一道空洞门**。
  `run_health()` 的 F 支只产出 `report_present`（被引报告在不在盘上的布尔），**一个门字段都不读**；
  runner 内还另抄了一份 `F_CONTRACTS` 门文本，且已与冻结评价集漂移（F04 抄件写"表层回答不随任何
  消融改变"，而第九批已实测该读数依链路而定）。现在：项名/门文本/被引报告一律取自
  `cap0_eval_set_v1.json`，逐项从被引报告的**原始数字**复算，另逐项机检 `must_show`
  （07 §4.2"展示输入到实际结果，不得只报局部 probe"），**无机检实现的 must_show 会把已过门的项
  压回 partial**。健康报告格式号 v2→v3（v2 的 F 行不可当判据读）。实测：F01 **pass**（六门+outcome+
  自身链）、F02 **fail**（G4/G5 未过，负结果保持）、F03 **partial**（门文本里 "+2.000" 与
  "interleaved 6/6" 在封存件无可定位字段 ⇒ 记 unverified，不猜）、F04 **fail**（默认入口输出
  `templated=true / distinct_signatures=1`，且 `wiring_defect` / `default_tick=36` vs 16,000,000）。
  统一入口五臂**现场重跑**逐位复现封存读数（`matches_sealed_report=true`，13.2 s，产物只写临时目录）。
  教训与 DEBT-I5/普查 §1e 同型：**"证据文件在场"被当成"证据成立"**，是这类只读封存件检查的通病。
- **DEBT-I4 的 H 半支（任务 #34）：标定采样已完成，阈值待冻结（2026-09-20）**。新脚本
  `scripts/training/calibrate_taiji_cap0_h_gates.py`，空闲机 5 次重复、每次新建进程、显式 required 链，
  产物 [reports/taiji_cap0_h_calibration_20260920.json](../../../reports/taiji_cap0_h_calibration_20260920.json)：
  H01 0.2402–0.2801 s、H02 0.4760–0.5042 s、H03 折算 0.5570–0.5678 s/次、H04 207,791–210,942 B、
  H05 150 次零崩溃。建议上限 = max×2，`threshold_status=proposed_not_frozen` ⇒ **冻结归所有者/CI**。
  两条随带事实：① H 读数同样依链路而定（裸链单点 H02 0.3223 / H04 163,511 vs required 链
  H02≈0.49 / H04≈211,000）⇒ **标定与正式评价必须同链取数**，否则门限一上线就永久误判；
  ② `--health` 在 required 链实测 **28.1–28.6 s**（本册先后记过 17.2 s 与 25.6 s，都是裸链路口径）
  ⇒ 战役每阶段预算加数按 ~28 s 记，换链成本约 +3 s。
- **DEBT-I4 的链路纠正后果：A05b 在默认基座上仍是必过项之红（2026-09-20，待用户裁决）**。
  第九批把 A05b 转入 `A_HEALTH_CHECKS` 的依据是 required 链上 `seed_beta.pt`（16M-tick）实测 True。
  本轮把同一批消融在 required 链上对**默认基座 `seed_corpus.pt`** 重测（新件
  `reports/taiji_cap0_exit_health_v3_required_chain_20260920.json`）：**A05=true、A05b=false** ⇒
  那条"换链即翻 true"不成立到默认基座，而默认基座正是产品入口所服务的对象（DEBT-I9 的来源问题未结）。
  后果：J4 的 A 支在默认基座上**永久判红**，除非 ① 换用可加载的 16M-tick 基座作默认，或 ② 把 A05b
  降回"披露不判"，或 ③ 修好可读性闸门让回答真随参数变。本轮只登记，不改判据。
  **另：C/D/E 换链后仍是 0.0（0/14、0/16、0/20，与裸链退出件逐项相同）** ⇒ 语言维度的失败**不是**
  链路伪影，不能用"跑错链了"来解释它——这条要挡住日后拿链路问题给 CAP 语言门开脱的走法。
- **登记：一份自称冻结的合同从未进版本库**。`plans/reference/M5_P5_2A_PREDICTED_EXECUTION_PREREGISTRATION_20260913.md`
  正文写"冻结日期 2026-09-13"，但 `git log -- <file>` 无记录、至今未跟踪。冻结合同不入版本库 ⇒ 无法确定
  它在哪个 revision 上冻结、也不能按摘要复算，而 [M5 就绪度评审表](../../reference/M5_EXIT_READINESS_REVIEW_20260920.md) §8 的依赖链"P5.2a/b→P5.2d"正引用它。处置待定。
- **更正本册上一条"P5.2a 未入库冻结件"的处置（2026-09-20 所有者裁决 + 核对）**。核对结论：该文件是
  已入库的 `plans/reference/M5_P5_2A_PREDICTIVE_EXECUTION_PREREGISTRATION_20260913.md` 的**早期变体草稿**——
  它独有的数值（train/val/final = 24/8/12、相对差 +0.2、wall ≤600 s、max_steps 3/4/3）、
  `PredictedExecutionTask` 与其五个门名在整个仓内**没有任何实现或下游引用**，§11"执行记录"永久空白，
  且文件时间戳是 09-19，比它自称治理的那次运行晚六天。实际管过那次运行的合同、runner 与两份报告
  全部指向已入库那份；P5.2b 也按"predictive_execution_insufficient"承接，依赖链不经过本草稿。
  ⇒ 所有者裁决：**移入 `plans/archive/history/` 作历史草稿，不入版本库当冻结合同**（入库会制造两份
  互相冲突的"09-13 冻结"，且会把证据链日期标错）。就绪度表 §8 的"P5.2a/b→P5.2d"经已入库那份可复核。
  本条覆盖上一条末尾的"处置待定"，那句话说的是当时未核的状态；核完即按本条办。
  教训与本会话反复出现的那类同型：**"自称冻结"不等于"冻结"**——冻结的凭据是版本历史 + 被实现的
  常量 + 执行记录三者齐备，缺一就只能是草稿。
- **DEBT-I9 结项（2026-09-20 第十一批）+ 换底带来的三条读数更正**。产品默认入口由
  `seed_corpus.pt`（套件重初始化产物，`trainer=api_seed_runtime`）换到 16M-tick 训练态
  `seed_beta.pt`，来源清单升 v2：四要素自述 + 进度流 + 当年长训命令行互证，且 `provenance_limits`
  写死"训练日志已失 ⇒ 只是一致性证据不是生成性凭据"。**来源问题的可判定半由此闭合**；
  仍不宣称"官方可复算"。三处硬编码同一默认基座的地方一并收口：来源清单（跟随产品常量）、
  隔离合同（`test_default_checkpoint_isolation_contract`）、以及**启动 smoke 脚本里那句
  `assert DEFAULT_CHECKPOINT.name == "seed_corpus.pt"`** —— 它改为读来源清单，否则"产品事实"
  同时写在三处、少改一处就红一片（本轮实测就是先红在 smoke、后红在隔离合同）。
  换底后的实测更正，全部重测不外推：① **A05b false → true**（旧底的"必过项永久判红"是底造成的）；
  ② **D 0/16 → 1/16、E 0/20 → 3/20**，C 仍 0/14 ⇒ 上一批"C/D/E 全 0 与链路无关"的表述只对链路成立、
  没测换底，已在就绪度表 §6 与队首改写；③ H 新底单次 chat 慢约三成（H03 折算 0.557–0.568 →
  0.736–0.745 s），阈值已在 `…_h_calibration_beta_20260920.json` 上重取并按 max×2 冻结生效。
  随带实测更正：`--health` 在 required 链、新底上是 **36.4–36.7 s**（旧底 28.1–28.6 s），
  战役每阶段预算加数按 ~37 s 记。
- **换底后仍开的三项（登记，未处置）**：① **F04 复判锚在 09-15 的 `taiji_cap0_inventory` 封存件上**，
  它的"默认入口服务训练态"子句在新底已实际成立却读不到 ⇒ 须重出 inventory 报告才能翻；这正是
  "基线取自封存报告"的锚点问题，落地会删掉反事实锚点的同型。② **战役基线对未重出**：
  `P3A_BASELINE`/`P3A_HEALTH` 核实后本就同指 `seed_beta`（换底消掉的是"产品底 ≠ 判据底"这处既有错配），
  但 `P3A_BASELINE` 是 v1 格式、无 identity 块；`judge_health` 不读 identity ⇒ 重出属证据质量项。
  ③ **CAP runner 的两处收尾未做**：`eval_taiji_cap0_baseline.py` 的 `DEFAULT_CHECKPOINT` 仍写死字面串
  （应引用产品常量）、健康支摘要打印仍以 `"contracts"` 兜底（F 块结构已改，那行会谎报形状）。
  原因：这两处所在文件被逐项起子进程的 runner 使用，中途改码会把两版代码混进同一份报告，
  而当时重跑链正在落盘 —— 宁可留下登记，不在跑动的件上动刀。
- **收口套计数更正（2026-09-20）**：换底后全量套 **`1857 passed / 10 failed / 6 skipped + 1 xfailed`**
  （1172.00 s），跑前跑后 `seed_beta.pt` 仍 `ad2a06465e0e…`、`seed_corpus.pt` 仍 `c8025db44c65…`
  ⇒ 换底与重跑都没有改写任何 `.pt`。10 红拆分：6 支既有（5 支结构性增长门 + stop_reason 审计面守卫）、
  1 支是我自己那条 preflight 断言写错（清单里 `contract` 值带了 " §1" 尾巴，已修）、
  3 支是换底必须随之更新的守卫（隔离合同、启动 smoke、以及换底后 D/E 非零带来的读数面）。
  **修完这三类后的复跑见下一批**；旧对照 `1857/7` 已被本行取代。
- **复跑结果（2026-09-20 05:25，换底批收口）：`1861 passed / 7 failed / 6 skipped + 1 xfailed`（1171.15 s）**。
  10 红降到 7 红，减的正是换底必须随之更新的三支（启动 smoke、隔离合同、我自己那条 preflight 断言），
  剩下的 7 支就是常态基线（5 支结构性增长门 + stop_reason 审计面守卫 + 1 支产品工件目录残留）。
  跑前跑后 `seed_beta.pt` 仍 `ad2a06465e0e…`、`seed_corpus.pt` 仍 `c8025db44c65…` ⇒ 全程无 `.pt` 被改写。
  **对照口径改为 `1861/7`。**
- **新底 inventory 重出后翻案：F04 换引用并不能让它转绿，且发现第四处硬编码默认基座**。
  `eval_taiji_cap0_inventory.py` 重跑（rc=0）自报 `default_checkpoint=seed_corpus.pt / default_tick=2 /
  wiring_defect=true` —— 换底都已经落地了它却还这么说，只有一个解释：**该脚本自带一份默认路径，
  没有引用 `api.seed_runtime.DEFAULT_CHECKPOINT`**（与本轮刚收口的三处同型，是第四处）。
  所以 §5 那条"F04 改读新底 inventory"的裁决**不足以闭合 F04**：先得把 inventory 脚本的默认路径
  改为引用产品常量并重出，否则 v2 换的是一份仍描述旧产品事实的件。
  更要紧的是同一次输出里的 **`trained_templated: true`** —— 16M-tick 训练态经入口作答**同样是模板回显**
  （7 题、`templated=true`）。⇒ F04 那条"输出非固定模板回显"在新底上**仍不成立**，F04 预期停在
  fail/partial 而不是 pass；"表层回答随参数改变"（A05b 已转真）与"回答成句/非模板"是两件事，
  不得互相顶替。这也再次说明 §6 那条边界：换底修的是底色与来源，不是语言能力。
- **inventory 的字段面板随底而定（2026-09-20 实测红；修法已定，本批未完）**。把第四处硬编码默认路径
  改为引用 `api.seed_runtime.DEFAULT_CHECKPOINT` 后重跑：`wiring_defect` 由 `true` 转 `false`
  （默认已是训练态，符合预期），`default_templated` 仍 `true`；但 `trained_probed /
  trained_turns_answered / trained_templated` **整块消失** —— 因为"最训练的那份 ≠ 默认"这一分支
  不再进入。字段面板复现守卫 `test_a_fresh_inventory_sample_reproduces_the_sealed_one` 随即红：
  "仪器少产/多产了字段"。**这支护卫是真在判事的，红得对**（它不是只读封存件那一类）。
  修法：`most_trained` 与默认同源时仍**恒发**这三字段（不探针则置 null），使 schema 与底无关；
  本轮未做该修改，故把源改动回退、补丁留在会话外，与评价集 v2 同批做 —— **不把一处未收口的改动
  连同它的红一起提交**。
- **接上条：该修改已在同会话续跑段落地，且"恒发字段"只解决了一半**。补回
  `template_signature` 的四个空值叶子之后，守卫不再报"少产"，改报**"多产字段"**——根因不在仪器，
  而在**比较基线**：`RESAMPLE` 指的是换底前的 09-18 样本，那份描述的是另一个产品事实
  （默认是未训练底、且训练态被单独探过针）。继续拿它比，就是把一次有意的换底读成仪器漂移。
  处置三件，都按本仓既有纪律：
  ① **重基比较基线**到新底样本 `reports/taiji_cap0_inventory_beta4_20260920.json`
  （"改行为须同批再生报告"），09-18 旧件原样留档不覆写，并以
  `RESAMPLE_BEFORE_SUBSTRATE_SWITCH` 具名保留；
  ② 给缺行开一条**有界豁免**：只允许 `raw_output_inventory.most_trained_turns[*]` 在
  `most_trained_entry.probed is False` 时缺失，**多产字段一律红**，缺别的路径也红；
  ③ 为防"重基"被当成"把对不上的一次抹平"，另加一支测试**钉住换底前后两份样本的差异本身**
  （旧：`default_checkpoint=seed_corpus.pt` + `wiring_defect=true`；新：`seed_beta.pt` +
  `default_tick=16000000` + `wiring_defect=false`），若哪天默认被悄悄挪回测试产物，这一支即红。
  **同批再次确认**：新底的 `default_entry.template_signature.templated=true` —— 换底**没有**把
  模板回显改掉 ⇒ F04 的"非模板"子句在新底仍不成立，它与"A05b 已转真"是两件事，不得互相顶替。
  上一条里"补丁留在会话外"的说法就此作废，以本条为准。
- **评价集升 v2 落地（2026-09-20 续跑段，所有者裁决"上限最高项"）**。`cap0_eval_set_v2.json` 由脚本
  从 v1 生成（不手抄），**唯一实质差别是 F04 的 `reference`** 挪向新底 inventory；合同测试
  `test_v2_is_the_current_set_and_differs_from_v1_in_exactly_one_field` 用扁平化路径集把这件事钉死：
  v1 与 v2 去掉版本元字段后必须同形状、且差异集合恰好等于那一条路径，F 项的 `gate`/`must_show`
  文本逐条断言与 v1 相同 ⇒ **升版不是降线**。v1 不覆写，`test_cap0_eval_set_contract.py` 里对 v1 的
  历史钉法原样保留。
  同时给 F 维加了**引用时效守卫**（原第三方案的守卫部分）：F04 复算前先比对被引 inventory 里的
  `default_checkpoint` 与来源清单登记的现行默认；不一致判第三态 **`stale_reference`**（既不是 pass
  也不是 fail——拿旧产品事实出今天的结论时，判哪一边都是冒称），并配两支测试：造一份描述旧底的载荷
  必须落 `stale_reference`；读不到现行登记时该子句记 `unverified` 而不是默认"未过期"。
  换底后 F04 实测：**非 stale、仍 fail**（新底 `templated=true`）⇒ F 维 pass 的依据仍只是 F01 一项。
  随带收掉两处本册记过的收尾：`eval_taiji_cap0_inventory.py` 的第四处硬编码默认路径（见上条）、
  以及健康支摘要打印里 `"contracts"` 的谎报兜底（F 块结构已不是合同清单，现改印维度门判定）。
  **已补的洞**：`run_inventory()` 此前裸跑就会把 `DEFAULT_REPORT`（=09-15 那份历史件）原地覆盖，
  而报告正是判据锚点、盖掉不可复原；现改为"目标已存在即拒跑"，并配一支测试钉"拒跑且旧字节一字未动"
  （50 passed + 1 xfailed，ruff 全仓干净）。**仍开**：战役基线对绑 identity 重出。
- **战役基线对：核完影响面后**建议不做重指向**，本条同时更正本会话上一批写下的"重取剩绑 identity
  的质量项"那句（那是没数引用面就写的）。实测三件事：
  ① 现行两份 `P3A_BASELINE`(09-15) 与 `P3A_HEALTH`(09-18 v5) 的 `checkpoint` **本来就同为
  `seed_beta.pt`**，链路也同为 required 对 ⇒ 换底之后它们与产品默认**同底**，配对守卫要的
  "健康报告与它旁边的评价报告同检查点"已满足；
  ② 我新重出的两份件确实更强（带 `identity{git_head, checkpoint_sha256}`，`trained_during_eval=false`），
  但 **`judge_health` 只读 `chain` 与 `checkpoint`，一个字都不读 `identity`** ⇒ 绑身份不改变任何判定；
  ③ 把常量挪到新件要动 **8 处引用**，其中 4 处是在钉**那份历史件的事实**（评审工作表用
  09-15 件生成、campaign 合同测试读它的特定读数、审计脚本的 `p3a_base`、`check_p3b_criteria` 的
  `DEFAULT_BASELINE`）。挪动它们等于为了一点判定收益去改写"第一次双臂 campaign 当时比的是什么"。
  ⇒ **结论：留旧件为 P3a 历史基线，不动**；新重出的两件（`taiji_cap0_beta_default_20260920.json`
  与 `taiji_cap0_health_v3_beta_default_20260920b.json`）作为**换底后现行产品事实的配对件**入库，
  下一次真要跑新 campaign 时按当时的 git head 现取现用，而不是把一个钉历史的常量搬来搬去。
- **随带发现（登记，未处置）**：`eval_taiji_cap0_inventory.py` 的 `DEFAULT_REPORT` 仍指向
  `reports/taiji_cap0_inventory_20260915.json` ⇒ **裸跑该脚本会覆写一份封存报告**，违反"报告不覆写"。
  本轮两次重跑都显式传了 `--report`，纯属侥幸。改法取后者更合本仓纪律：目标已存在即拒跑并报错，
  而不是把默认名挪走。
- **换底前的收口套状态（2026-09-20 第十批实测；那 7 支在本批之前既有，非本批引入）**。全量套
  `1857 passed / 7 failed / 6 skipped + 1 xfailed`（1164.62 s）。本册最后记录的对照基线是
  `1763 / 0 / 6 + 1 xfailed`，测试数的增长由 D1–D8 与统一入口/协作各批带入，但**这 7 支不是本批造成的**，
  逐支核对如下：
  - **5 支结构性增长门**（`test_continuous_structural_growth`、`test_cross_domain_structural_gain`、
    `test_online_interaction_structural_bridge`、`test_structural_workspace_net_gain`、
    `test_terminal_three_domain_governance`）：把 5 支单独跑（77.19 s）仍然全红，报的是各自
    `evaluate()` 里的门断言（例：`online structural bridge requires two applied feedbacks`）；
    这五支的导入面与本批改动的文件**零交集** ⇒ 属 W7-p4-10b 增长线自身的红，与 CAP/R2 主线不同链路。
    **本轮未处置**，只把"它红在 HEAD"这件事钉下来。
  - **1 支计划一致性门** `test_active_plans_have_one_execution_owner_and_resolvable_links`：
    `plans/active` 里指向 `reports/taiji_p5_2d_online_writeback_v2_20260916.json` 的相对链接少了一级
    `../`，共两处（01 的 P5.2d 行、03 的同一行）。该测试**一次只报第一支**，所以第一轮只暴露 01 ——
    修法是先做一遍全量链接解析，别跟着测试一次改一处。两处已在本批改掉。
  - **1 支审计面守卫** `test_current_review_surface_is_complete`：`live_consumers()` 比冻结清单多出
    4 个文件（`taiji/collab_handoff.py`、`train_taiji_r2_content_binding.py`、
    `run_taiji_collab_handoff_entry_evidence.py`、`run_taiji_unified_entry_evidence.py`）。
    这正是该门的设计意图（fail-closed：新读 stop_reason 的文件必须先复核再入账）。逐条看：
    协作交接的两支在**判定**里比较 stop_reason（相等判断与集合去重后驱动断言），
    另两支只写入/聚合 ⇒ 分类应为 judgement 与 record_only 各二。**登记待复核入账，本批不动冻结清单**。
  - `ruff check .` 同轮 4 红（D5/D6 matched_dev、D4 probe、collab runner 各一处，均行为不变的死变量/
    死导入/等价写法），而 CI 的 "Lint with ruff" 是 blocking ⇒ 与上面同属"HEAD 不绿"。
  - **口径**：后续任何批次的收口数以 **`1857 / 7 / 6 + 1 xfailed`** 为新对照，不得沿用 `1763 / 0`，
    否则会把既有红读成新失败、或把新失败读成既有红。本批跑前跑后 `checkpoints/seed_corpus.pt`
    仍是 `c8025db44c65f9c1…` / 43,223,183 B，`output/manual-r5-canary/` 条目 4（方向仍"不涨"）。
- **DEBT-I5（只登记，不处置）契约测试用字面行号锚定源码位置 ⇒ 源码一漂移，断言就失真而测试仍绿**。
  `scripts/training/eval_taiji_cap0_inventory.py:350-356` 把 `"line 2726-2732"`（以及 `"line 2611"`）
  作为**硬编码字符串**写进诊断文本，`tests/taiji_native/test_cap0_inventory_contract.py:102`
  再断言这个字面串存在。行号没有任何东西重算：只要在 `taiji/model.py` 第 2726 行之上增删代码，
  报告里的行号就指错位置，而**套件不会失败**——这是最坏的一类假绿。
  修法：断言**错误文本**（`"enabled identity organ checkpoint payload is missing"`）或从源码实际
  解析该行区间；**须与 CAP-0 加载器改动同批做**（见 [决策简报 §5 第 3 步](../../reference/CAP0_LEGACY_LOADER_DECISION_BRIEF_20260915.md)）；
  本轮不做（campaign 在跑，不能动 `taiji/` 与其评测面）。
- **DEBT-I6 ✅已处置（2026-09-18 第二批，含故障注入测试）阶段报告非原子写 + 驱动"文件存在即复用"的续跑缺口**。
  实测：`eval_taiji_cap0_baseline.py:792` 用 `report_path.write_text(...)` **普通覆写**写阶段报告；
  `run_p3b_campaign.py:149-150` 的 `_score_stage` 只看 `if stage_path.exists(): return _load(stage_path)`。
  ⇒ 若在写报告途中断电/被杀，续跑会**直接消费一份被截断的报告**（当前表现是 `json.loads` 抛错、
  驱动退出、训练器成孤儿——由等待器的 `driver_stalled` 兜住，故是"响亮的错"而非静默失真，
  这正是本轮不处置的原因）。
  修法（须与 §5 加载器批次同做）：**复用前先校验**（能解析 + 四个评测面字段齐 + C/D/E 各 20 题 +
  `trained_during_eval is false`）；不合格时**只有** `checkpoints/p3b/snapshots/` 里该 tick 的快照仍在
  才可重评（快照才是"该 tick 可复现"的唯一凭据——实时检查点早已前进），快照缺失时**拒绝续跑并报错**，
  不得静默重评一个更晚的状态。**必须配一支故障注入测试**（写一半的 JSON → 断言被识别、且无快照时拒绝）。
- **DEBT-I7（只登记，不处置）测试套件可以写产品默认检查点 `checkpoints/seed_corpus.pt`**。
  实证：本次双臂 campaign 期间（本地 01:41，全量套件在跑）该文件被 `trainer = "api_seed_runtime"`
  重新保存 —— 路径是 `api/seed_runtime.py` 的多处 `self.save()`（默认落在 `DEFAULT_CHECKPOINT`），
  而 `tests/` 下有 26 个文件同时出现 `SeedRuntime` 与 save/train/reset 类动作，没有任何隔离。
  **本次没有丢训练产物**：封存盘点报告显示该文件本来就不是训练态（`tick = 2` 的默认基座，
  重存前后同为 43,223,183 B），损失限于"基座权重被重新初始化了一遍"。
  但危害类别是真的：**一支测试可以改动产品入口指向的模型文件**，而它在共享工作区里还会
  连带触发别的守卫（本次就惊动了 P3b 的受保护检查点核对）。
  修法（二选一，都要先让红测试存在）：① 那些测试必须把 runtime 的 checkpoint 路径指到 `tmp_path`
  （fixture 级隔离，且加一支"任何测试运行后 `seed_corpus.pt` 的 sha256 不变"的守卫测试）；
  ② 或让 `SeedRuntime.save()` 写默认路径需要显式参数，默认拒写。
  本轮不做：涉及 `api/` 与 26 个测试文件的公共夹具，且改 `taiji/`/评测面会打断在跑 campaign。
- **DEBT-I7 ✅已处置（2026-09-18 第三批：写/读默认值拆分 + 会话级重定向）**：根因是结构性的——
  `api/seed_runtime.py` 用**同一个常量**同时充当"产品默认加载的来源"与"不指定路径时 save 的落点"，
  于是一支不传 `checkpoint_path` 的测试**没有办法安全地**调用 `save()`。修法：
  ① 拆名——新增 `DEFAULT_SAVE_TARGET`（默认等于 `DEFAULT_CHECKPOINT`，产品行为一字不变）与
  `resolve_save_target(path, checkpoint_path)`，`SeedRuntime.save()` 改走它；
  ② `tests/conftest.py` 增加会话级 autouse fixture，**只**把 `DEFAULT_SAVE_TARGET` 指到 tmp 目录，
  读侧 `DEFAULT_CHECKPOINT` 保持指向真文件（否则"默认入口服务哪个模型"就没有判断对象）；
  ③ 新增 `tests/taiji_native/test_default_checkpoint_isolation_contract.py` 四支测试，正反都钉：
  无路径的写靶落在 `checkpoints/` 之外、显式路径与"来源路径"仍然优先（重定向不许吞掉正常写盘）、
  读侧未被搬走、以及**真跑一次 `save()`** 后产品默认文件的 sha256 不变。
  实测 `15 passed in 3.11 s`（含既有 SeedRuntime 重用户 `test_native_life_status` / `test_semantic_provider`）。
  本条登记过的"26 个测试文件同时出现 SeedRuntime 与 save/train/reset"这一面**不需要逐个改**：
  写路径被一次性收口在同一个落点上。**第二实例也已处置（同日第五批）**：见本文件开头
  "DEBT-I7 的第二实例"那一条末尾的处置记录 —— `output/manual-r5-canary/` 的残留改成"会话结束时只删本进程 pid 命名的那批"，
  实测该目录在今天四次全量之间从 10 项 / 335 MB 涨到 14 项 / 500 MB，正是这条通路还活着的证据。
  **第一版留了一个代码级缺口（同日第三批续，⚠ 这里的因果已重推过一次，第一版叙述是错的）**：
  `SeedRuntime.load()` 无参时把 `DEFAULT_CHECKPOINT` **记进** `self.checkpoint_path`，而 `save()` 的
  取值顺序是"显式参数 → 来源路径 → 默认"——只重定向"默认"这一档，走 `load()` 的运行时照样能写回产品文件。
  修法是把"来源 == 产品默认"也映射到被重定向的写靶（`resolve_save_target` 内两行；产品侧两名同值
  ⇒ 行为不变），并补一支端到端守卫 `test_a_runtime_that_loaded_from_the_product_default_does_not_save_to_it`
  （真 `load()` + 真 `save()`，断言产品文件 sha 不变）。实测 `5 passed in 2.06 s`。
  **这条缺口是"代码上成立、今天的套件里未被触发"，不是"实测被写了一次"**：
  最初我以为它被实测抓到（新采样本的 `saved_at_utc` 与封存样本不一致），重推时间线后不成立——
  那次改写发生在 **13:58:40 起跑的那套**（我的 conftest 修正在 14:07:58 才落地 ⇒ 那一套没有隔离），
  它在 14:18 结束时把 sha 从 `3fec3e47` 变成 `c8025db4`；
  而 **14:27:34 起跑、带第一版隔离的那套**跑前跑后同为 `c8025db4`、`tick=2`、43,223,183 B
  ⇒ 那才是"套件不再改写产品默认基座"的第一次套件级验证。
  **教训**：一条"某修法不够"的断言，要有"触发它的那次运行"的时间线才算证据；只看到时间戳不一致就归因，
  会把一次早于修法的写入算成修法的失效（[[feedback-recompute-dont-hand-carry]] 的第 6 种模式：
  把相关当因果，且没有核对先后）。
- **DEBT-I8 ✅已处置（2026-09-18 第二批）已提交的 P3b 报告里嵌着机器绝对路径 ⇒ 同一份快照的两次打分无法逐字节比对**。
  实证：驱动写的 `reports/p3b_stages/control/cap0_tick_17000000.json` 顶层
  `checkpoint = "E:\\Seed\\checkpoints\\..."`，而我手工复评同一份快照得到的却是
  `"checkpoints\\..."`（因为我传的是相对路径）——两份报告除这一处拼写外**逐字段全等**。
  `reports/taiji_p3b_campaign_control_20260915.json` 的 `criteria.baseline` / `criteria.candidate`
  同样是 `E:\\Seed\\...`。后果：产物不可跨机比对、把本机目录结构写进了仓库、
  且"复评是否等价"这类判断只能靠字段级 diff 而不能靠哈希。
  修法：报告写出处统一用 `train_p3b_aligned._relative()` 那套相对化，并在合同测试里断言
  报告内不含盘符（`re.search(r"[A-Za-z]:\\\\", text) is None`）。
  本轮不做：需要同时改评测面脚本与驱动收尾，而实验臂正在跑（驱动每阶段起子进程、
  收尾时重写campaign 记录）；等双臂结束后与 DEBT-I5/I6 同批处理。
- **DEBT-I8 ✅处置记录（2026-09-18 第二批）**：写出侧两处都相对化——`eval_taiji_cap0_baseline._relative()`
  用于报告顶层 `checkpoint` 字段（run_baseline / run_health 两处；**子进程 payload 里的路径保持绝对**，
  那是 IPC 不是证据），`check_p3b_criteria._relative()` 用于 `baseline` / `candidate` 两个字段。
  合同测试 `test_criteria_report_paths_are_repo_relative` 钉住三件事：仓内路径 ⇒ 相对且**能解析回同一文件**、
  仓外路径 ⇒ 原样保留（不是崩也不是静默改写）、以及 `check()` 真实产出的记录里两个字段无盘符
  （`re.search(r"[A-Za-z]:[\\/]", ...) is None`）。
  **已封存的历史报告不追改**：`reports/p3b_stages/**`、`taiji_p3b_campaign_*_20260915.json` 里的
  `E:\\Seed\\...` 保持原样——它们是"当时确实是那条路径"的证据；此后同一快照的两次打分才可用哈希比对。
  另注：R2 一侧已有 `_resolve_repo_path()` 同时吃两种拼写（`audit_taiji_r2_h3_7_attribution.py:150`），
  ⇒ 读侧本就能兼容，本次只动写侧。
- **DEBT-I5 ✅已处置（2026-09-18，随 M2-2i 落地同批）**：`eval_taiji_cap0_inventory.py` 的诊断文字
  已改为按**错误文本**描述（不再出现 `line 2726-2732` 之类位置断言），报告已再生；
  `test_cap0_inventory_contract.py` 新增 `assert "line " not in diagnosis and "2726" not in diagnosis`
  ⇒ 位置化石一回来就红。实证价值：R2 的工作把该行推到 3339（漂移 613 行），而旧断言一直绿。
- **DEBT-I6 ◐部分处置（2026-09-18）**：原子写与"复用前校验"仍未做；但**已消除更危险的一种自我覆盖**——
  探针/清单在带 `error` 时改写同名 `.error.json`，**不再覆盖封存报告**。
  触发事实：本轮我重跑 `probe_taiji_cap0_legacy_load.py` 时它崩溃，把 115 行封存证据换成 4 行错误存根
  （已 `git checkout` 还原并留档 `.git/stub_*.json`）。剩余部分仍须配故障注入测试。
- **DEBT-I6 ✅剩余部分处置（2026-09-18 第二批）**：三件都做完。
  ① **写端原子化**——`eval_taiji_cap0_baseline.py` 新增 `_write_report()`（`tmp`+`replace`，临时名带
  `getpid()`，防双臂同刻写同名报告互踩），main() 的三处报告写出（基线 / health / adjudication）全改走它；
  驱动自己的 `_write()` 本来就原子写，未动。
  ② **复用前校验**——`run_p3b_campaign._reuse_defects(report, baseline)`：评测面四字段 + C/D/E 逐题
  **id 序列**（复用 `_surface_drift`，题数由基线导出而非钉死"20"，免化石）+ `chain` 必须等于 P3a 链路 +
  `trained_during_eval is False`。`_evaluate()` 只在该表为空时才复用；不合格时**快照仍在**才重评，并把坏报告
  留成 `*.unusable` 物证（本轮已丢过一次 115 行证据，不再静默覆盖）；快照已失 ⇒ `SystemExit` 拒绝续跑，
  绝不"顺手重评一个更晚的状态"。
  ③ **故障注入测试 4 支**（`test_p3b_campaign_contract.py`，一律 monkeypatch 掉评测子进程，成本 0）：
  截断 JSON / 能解析但语义错（三种字段各自成一条）/ 快照缺失必须拒绝且**不得**起子进程 /
  **正向**一支"合格报告照旧复用且不重评"——缺了正向，"永远重评"的实现也能骗过前三支。
  实测：B0 批 `45 passed in 1.26 s`，cap0+p3b 批 `77 passed in 1.44 s`。
  **顺带更正本节两处文字**：被审函数实名是 `_evaluate`（原文写 `_score_stage`），`:149-150` 已漂到
  `:176-177`。
- **DEBT-I9（新，未处置）产品默认检查点被套件重写且不入 git ⇒ "默认入口=未训练基座"已无法从盘上取证**。
  实证：`checkpoints/seed_corpus.pt` 现为 `tick = 36`、`trainer = api_seed_runtime`、
  `saved_at_utc = 2026-09-18T04:35:33Z`（DEBT-I7 的那条通路），而 `*.pt` 在 `.gitignore` 里 ⇒
  **原 tick=2 基座无法还原**。处置：`test_default_entry_serves_an_untrained_state` 保留全部断言但标
  `xfail(strict=False, reason=DEBT-I9)`——基座恢复后会自动变 XPASS 提醒收严；
  **把期望值改成 36 等于把污染正当化**。根因在 DEBT-I7，须先做隔离。
  **更正（同日，2026-09-18 第二批）**：上面"基座恢复后会自动变 XPASS 提醒收严"是**错的**——
  该测试读的是**已封存报告**的 `model_reality.default_tick`，不是盘上的 `.pt`。所以要它翻红/翻绿
  必须**再生成一次盘点报告**，光恢复基座不会动它（这正是普查 §3 说的"读封存型测试永远不会红"）。
  已给 `eval_taiji_cap0_inventory.py` 补 `--report`，复采一律写新日期文件，不覆盖 09-15 那份证据。
  **DEBT-I9 的可检测半已落地（同日第六批）**：`plans/manifests/product_default_checkpoint_provenance.json`
  记下当前默认基座的 sha256 / 字节数 / envelope tick / `trainer` / `saved_at_utc`，
  `tests/taiji_native/test_product_default_checkpoint_provenance_contract.py` 三支把它变成守卫：
  sha 变了就红，失败信息直接列两种可能（① 又有测试或**子进程**写了默认路径 ⇒ 先修写者；
  ② 有意换基座 ⇒ 更新清单 + 记来源），并禁止"把期望值改成当前值了事"。
  第三条测试钉住清单**必须继续自称 provenance=unknown**——记下来不等于变成出厂基座。
  顺带确认了两件事：envelope 的 `saved_at_utc = 2026-09-18T14:18:32Z` 正是第四次套件结束那一刻
  ⇒ 我先前那条"来源=套件写的"由文件自身证实；而带隔离的两次套件跑完 sha 都不变 ⇒ 守卫当前为绿。
  **仍不结项**：来源问题要一份非测试产生的基座（官方重训或分发文件）。
  已知边界也写进测试 docstring 了：conftest 的重定向只在 pytest 进程内生效，
  **测试起的子进程仍会落到真实默认路径**——那正是本守卫存在的理由。
  **DEBT-I7 处置后的状态（同日第三批）**：写侧已被隔离，产品默认文件从此**不会**再被套件改写；
  盘上现存的是"某次套件重初始化"的 tick=2 基座（09-18 复采实测 `tick=2`、43,223,183 B，
  与 09-15 封存报告里同一 tick 的 43,290,771 B 不同 ⇒ 连"同一 tick"都不保证同一份权重）。
  **本条仍不结项**，因为剩下的问题是**来源**而非通路：需要一个非测试产生的默认基座
  （官方重训或分发一份基座文件），并把它的 sha256 记进受审清单，让"默认入口服务的是哪个模型"
  重新成为可判定命题。在那之前 `test_default_entry_serves_an_untrained_state` 保持
  `xfail(strict=False)`，09-15 报告不追改。
- **DEBT-I7 / I9 第二次复现（2026-09-18，全量套件，当时未处置）**：套件跑前 `seed_corpus.pt` 为
  `tick = 36` / sha `105d621e5e7d`，跑后为 **`tick = 2` / sha `3fec3e477ef4`**（mtime 同步更新到
  套件结束那一刻；**跑前尺寸未记录，故本条不作尺寸断言**）。⇒ 两点新增事实：① **这条通路不是幂等的**，每次跑套件都把默认基座
  重写成一份新的随机初始化态，所以"默认入口=未训练基座"这句话在盘上永远取不到证；
  ② 重写方向是 36→2，也就是说**测试会把一个已训练态退回未训练态**——比"重新初始化一遍基座"更糟，
  若哪天默认入口指向的是真训练态，套件会把它抹掉。
  本轮不处置（要动 `api/seed_runtime.py` 与 26 个测试文件的公共夹具），但**优先级应高于 I6/I8 那批**，
  因为它动摇的是"默认入口"这一产品事实，而不只是仪器。
  **后效（同日第三批）**：已按上文 DEBT-I7 条目处置，写侧隔离落地；上面"26 个测试文件要逐个改"的
  预估不成立——收口在一个落点即可。
  **两处尺寸记载互相冲突，留此不作裁决**：本条复现前的 09-18 04:35 事件写"重存前后同为
  43,223,183 B"，而 09-15 封存报告记同一 `tick=2` 的文件为 43,290,771 B——两句不可能同时为真。
  可复验的只有：当前文件 43,223,183 B、两份报告各自如上。历史条目都不删，按 §4 约定只加指针。
- **DEBT-I10（新，本轮已修，但教训未消化）约束解码是"源码副本补丁"，锚点钉在实现细节上 ⇒ 整条语言测量静默罢工**。
  `probe_taiji_cap0_byte_output.install_constrained_decode()` 曾要求 `Taiji.generate` 源码里存在
  `next_symbol = step.predicted_symbol`；R2 改写 `generate`（新增 `response_start`/`response_phase`）
  后锚点消失 ⇒ **CAP-0 全部维度 child 退出码 1、报告里 0 题**，而合同测试全绿——
  因为唯一的"检查"是 `assert "install_constrained_decode" in <源码文本>`（纯子串 grep，普查判据里的空断言）。
  本轮修法三件：① 锚点改钉**包装真正消费的接口**（`reset_dynamics` / `observe` / `probabilities`）；
  ② 子机管道一律 ASCII（`ensure_ascii=True`：中文 Windows 子进程是 cp936，父进程按 utf-8 解码会
  让 `stdout` 变 None 并触发上面那条 fail-closed 假象）；③ 新增**活体测试**当场安装补丁、
  断言它拒绝不支持的 `response_start/response_phase/boundary` 调用，并在 finally 里还原 `Taiji.generate`。
  **未消化部分的更正（同日第五批，逐支查过）**：原先写"B0/R2 那批 copy-patch 仍用同类写法，应统一改为钉接口"，
  这句把两类不同的东西混成一类了。全仓 `inspect.getsource`/`exec` 型脚本共 6 支：
  `probe_taiji_cap0_byte_output`（**唯一一支静默罢工过的**，已改成钉接口）、
  `probe_taiji_b0_m1_counterfactual` 与 `probe_taiji_b0_structure_space`（锚点失配时
  **直接 SystemExit**"anchor appears N times... must be re-derived"——响亮的错，不是隐患）、
  以及三支 W7 modularization 门（它们**以读源码为判据本身**，锚点失配 ⇒ 指标 False ⇒ 门红）。
  ⇒ 不存在"统一改为钉接口"这件事；剩下的只是可观测性：那三支门的失败信息此前不说是哪条指标，
  已在 `353d0e3d` 补上"报出未通过的指标名"。
- 出口④的对照基线：上一节所述 **1408 / 0 失败 / 6 跳过** 已被后续轮次超出。2026-09-18 本地三次全量：
  `1727 / 0 / 6 + 1 xfailed`（批次二前的同码复采，20:12）→ `1732 / 2 / 6 + 1 xfailed`（批次二后，
  两道 `natural_language_workbench_*_modularization` 门红；单跑绿、与 cap0 同跑也绿 ⇒ 套件内跨测试污染，
  且当时的门测试只报 `False is True` 无法归因，已先加诊断）→ **`1740 / 0 / 6 + 1 xfailed`**（20:19，
  带 DEBT-I7 第一版写靶隔离；同一套跑前跑后 `seed_corpus.pt` sha 不变）。
  ⇒ 判"有无新增失败"自本条起以 **1740 / 0 / 6 + 1 xfailed** 为对照，1408 那条只作历史锚点。
  同日又跑两套：**`1742 / 0 / 6 + 1 xfailed`**（21:08，带 DEBT-I7 第二版；跑前跑后 sha 同为 `c8025db4`）
  与 **`1743 / 0 / 6 + 1 xfailed`**（21:18，另含"放宽守卫已成空操作"那条逐叶断言）。
  ⇒ 对照基线现为 **1743 / 0 / 6 + 1 xfailed**，且"默认基座字节不变"已随套件一同成立。
  **收口那一套（含本日全部改动）：`1746 / 0 / 6 + 1 xfailed`，21:30**。三条同时成立，都是测出来的：
  ① 0 新增失败（用例数从本日早段 1727 涨到 1746，+19 全是本轮新测试）；
  ② `checkpoints/seed_corpus.pt` 跑前跑后同为 sha `c8025db44c65` / 43,223,183 B
  ⇒ **DEBT-I7 在套件规模上成立**（此前四次都把它改写）；
  ③ `output/manual-r5-canary/` 条目数由 16 保持 16，且目录里最新文件的 mtime 早于本套起跑时刻
  ⇒ 会话末 sweep 确实删掉了本会话写的那一对（不是"这套没写"）。
  **远端仍未查询**（`gh` 未认证）⇒ 继续禁止"CI 已绿"表述。
  其后再两套：**`1749 / 0 / 6 + 1 xfailed`**（sha 同为 `c8025db4`，残留 16→16）与
  **`1754 / 0 / 6 + 1 xfailed`，用时 1316.26 s** —— 后者是含 DEBT-I4 接线与
  I5/I6/I7/I8 全部已提交改动的收口套，三条同时成立且都是测出来的：0 新增失败、
  `checkpoints/seed_corpus.pt` 跑前跑后同为 `c8025db44c65` / 43,223,183 B、
  `output/manual-r5-canary/` 条目数 16 保持 16。⇒ **对照基线现为 1754 / 0 / 6 + 1 xfailed**。
  远端同样仍未查询 ⇒ 依旧不得写"CI 已绿"。
  **A05 那一批（`63c38846`）的收口套：`1760 / 0 / 6 + 1 xfailed`，用时 1279.38 s**，退出码 0；
  同一套跑前跑后 `seed_corpus.pt` 仍是 `c8025db44c65` / 43,223,183 B，canary 条目 16→16
  ⇒ 对照基线改为 **1760 / 0 / 6 + 1 xfailed**（+6 是 A05 的 6 支新测试；采集发生在链路批改动落地之前，
  所以它验的是 `63c38846` 那个提交本身）。
  **链路批（`37f665fc` + 台账 `f023ca2e`）的收口套：`1763 / 0 / 6 + 1 xfailed`，用时 1417.43 s**，
  退出码 0，跑前跑后 `seed_corpus.pt` 同为 `c8025db44c65` / 43,223,183 B
  ⇒ **对照基线现为 1763 / 0 / 6 + 1 xfailed**（+3 是链路批新增/改写的测试净数；
  canary 条目那一轮是 16→14，未归因，见上面 A05b 条的 ⚠️）。远端仍未查询 ⇒ 不写"CI 已绿"。


## 最新状态补充（2026-09-15，WP-3 落地前的全量复采）

- **pre-WP-3 基线（HEAD `97aff6ed` + 计划修订，`--junitxml` 后台跑，1063 s）**：
  **1408 用例 / 0 失败 / 0 错误 / 6 跳过**（退出码 0）。原始 XML：`%TEMP%/pre_wp3_baseline.xml`（临时文件，不入库）。
- **类别 B 的 27 项 `SystemExit: 1` 本次未复现**。这**不是本轮的修复成果**，而是 §4 归因的必然结果：
  该级联源于本地 WorkBuddy 沙箱的批量删除守卫（单 tool call 删除 ≥50 路径即 `SystemExit(1)`），
  本轮没有触发批量删除 ⇒ 守卫不介入；而 2026-09-14 晚分离出的 **5 项真实回归已在 CI 修复轮**
  以 19 脚本 + 2 测试共 68 处补丁修掉。⇒ 本轮**未修改任何测试、阈值或产品代码**来"让它变绿"。
- **用例总数 949 → 1408（+459）** 来自 CI 修复轮新增的测试与跳过项（6 跳过 = 产物缺失时跳过本地产物依赖测试）。
- **⇒ WP-3 出口判据④的对照基线自本条起改为 1408 / 0 / 6**（旧表述"对 27 项基线无新增"作废，已在
  [当前推进方案](03_CURRENT_EXECUTION.md) 与[推进计划修订](../../reference/M5_B0_CLOSEOUT_AND_PLAN_REVISION_20260914.md)同步）。
- **类别 B 条目的处置**：保持登记但**降级为环境观察项**——它会在任何触发批量删除的本地会话里重现，
  不是项目缺陷，也不得被记为"已修复"。**远端仍未查询**（`gh` 未认证）⇒ 继续禁止"CI 已绿"表述。

## 最新状态补充（2026-09-14，WP-1 决策窗口前）

- **CI 命令级基线（2026-09-14，当前 HEAD `fba517cf` 复采）**：
  - `ruff check .` → **All checks passed**（B0 修的 `I001` 未回归）；
  - 全量 `tests/taiji_native/`（801s，`--junitxml` 后台跑）：**949 用例 / 27 失败 / 0 错误 / 1 跳过**
    （921 passed + 27 failed + 1 skipped = 949，与 `--collect-only` 一致）；
  - **失败集合与 2026-09-13 基线逐位相同：0 新增、0 消失**（27 项全为 `SystemExit: 1`）。
  - ⇒ B0 十一轮新增的 **+98 个测试全部通过**（851→949），且**在全量顺序上下文下也无新增失败**
    ——这一点重要，因为本仓库的既有失败形态正是顺序/状态污染，局部通过不足以说明问题。
  - 该结果**预验证了 WP-3 出口判据④**（"全量失败集合对 27 项基线无新增"）。
  - 原始 XML：`%TEMP%/taiji_wp1_full.xml`（临时文件，不入库）；基线 XML：`%TEMP%/taiji_b0_full.xml`。
  - **远端仍未查询**（`gh` 未认证）⇒ 继续禁止"CI 已绿"表述；本文只声明**命令级**基线。

- **【2026-09-14 晚｜归因更正 + 真实回归修复（本轮）】** 此条**推翻**上文对类别 B 的"顺序/状态污染"定性：
  - **类别 B 的 `SystemExit` 级联 = 本地 WorkBuddy 沙箱 safe-delete 守卫伪影，不是项目债务**。
    以 `--tb=long -o junit_logging=all` 复采，45 项失败的完整栈**全部**终止于
    `...\cli\vendor\shim\sitecustomize.py:826 (_exit_bulk_guard_control)`
    （帧链 `Path.unlink → _safe_path_unlink → _try_trash → _check_bulk_delete_guard`）；
    同一套件加 `CODEBUDDY_SAFE_DELETE_ENABLED=0` 后 **45 → 5**。CI（GitHub Actions）无此守卫。
  - **剩余 5 项是修复轮引入的真实回归**：`SeedRuntime.load(...)` 不保留构造时注入的
    `workspace_root`，`workspace.read` 因此回退到 `agent_workspace`；旧代码被 gate 脚本的
    **模块级 `get_setting` patch** 掩盖。已对 19 个脚本 + 2 个测试文件共 **68 处**补
    `workspace_root=PROJECT_ROOT`。
  - **本轮验证**：定向 6/6 通过；`tests/` 全量（守卫关闭）**1386 passed / 6 skipped / 0 failed**（1059s）；
    `ruff check .`、`ruff check . --select B,SIM --ignore B008`、
    `mypy --follow-imports=silent seed taiji`（114 源文件）三项**全部干净**。
  - **`black --check .` 已处置（`75e9c97b`）**：实测它是**远端 CI `test` job 唯一的失败步骤**
    （`test (3.10)` 与 `test (3.12)` 的失败步骤均只有 `Format check with black`；其余
    ruff / ruff B,SIM / mypy / pytest / coverage / startup-smoke / build-frontend / docker-build 全绿）。
    按项目既有配置（`[tool.black]` line-length=100 + `.pre-commit-config.yaml` hook == CI pin 26.5.1）
    执行一次全仓格式化 **460 文件**（纯格式、AST 保持）：`black --check .` 460 → 0（两次复检幂等）；
    格式化后四段验证仍全绿 —— ruff ×2、`mypy --follow-imports=silent seed taiji`（114 文件）、
    `tests/` 全量 **1386 passed / 6 skipped / 0 failed**。
    ⇒ 该项**不是可选债务而是 CI 绿的必要条件**；教训：判定"既有格式债"的优先级应先看 CI 口径。
  - ⇒ **本地复跑 `tests/` 必须设 `CODEBUDDY_SAFE_DELETE_ENABLED=0`**，否则读到的是环境伪影而非真实结果。

- DEBT-A1/A2 已结项，见下方修复记录；原“只登记不修复”是建册时范围，不应把已完成修复写回未解决。
- 30 失败/788 用例是 aa124f52 的历史基线；28 个 SystemExit 仍待定位，**当前 HEAD 重测计数为 27**（集合见 §6，为旧 28 项的严格子集）。
- 下文引用图论证仅能缩小直接依赖范围，不能证明间接状态、动态导入、文件和环境污染不存在；失败归属须结合可复现顺序、父提交对照与栈证据。
- DEBT-G1/G2/G3 的数量和路径是旧快照；后续 Git 修复见[路线 A 报告](../../../reports/M5_P5_2C_TRIPLE_PRIME_REPRESENTATION_REPAIR_RESULT_20260913.md)。本轮未做 fsck 或清理；继续禁止未经确认 gc/prune、删除备份。
- 新研究阻塞（已由 B0 审查并给出结论）：路线 B 的收益参照不一致、任务协作上界与门禁语义差异。B0 复算 32/32 一致、三种候选参照全部不可达 ⇒ 暂停正式训练，先改任务与估计目标。见[B0 设计包](../../reference/M5_DISTILLATION_TOMBSTONE.md)。
- **CI 命令级基线（2026-09-13，B0 建立）**：本地 `ruff check .` 原有 1 项 `I001`
  （`tests/taiji_native/test_p5_2c_triple_prime_representation_repair_gate.py` 的导入顺序），
  即两条 Linux CI 的 Ruff 失败原因；B0 已修复，现为 **All checks passed**。
  B0 新增测试 14 passed、目标集八个文件 95 passed；当次全量基线见上（851 / 27）。
  远端 workflow 本轮未查询（`gh` 未认证）。

以下保留建册时的观察、命令和修复记录；现行顺序由[当前推进方案](03_CURRENT_EXECUTION.md)决定。

> 建立：2026-09-13。登记基线：`aa124f52`。
> **本文只登记与量化，不修复。** 修复顺序由 [03_CURRENT_EXECUTION.md](03_CURRENT_EXECUTION.md) 决定；
> 主线（P5.2c 未见组合迁移及其后续）收尾后才进入本文的处置阶段。
> 纪律：登记项不得在未修复的情况下被当作「已解决」；每项必须保留可复现命令与观测基线。

## 0. 为什么单独建册

2026-09-13 修复 P5.2b 零步缺陷时发现，`tests/taiji_native/` 全量运行有大量失败，但这些失败
**无法归因于当次改动**（论证见 §1）。这类既有债务与主线实验证据混在一起时，会污染
「某次改动是否引入回归」的判断。因此把它们与主线解耦，单独建册、单独排期。

**登记册的用途**：任何一次改动后，若全量套件出现失败，可对照本册判断「是否属既有债务」，
而不必重新做一次归属论证。

## 1. 归属论证方法（可复用，重要）

判定「全量失败是否由本次改动引入」，**不要靠反复重跑或手感**。用引用图检查：

```bash
# 列出被改动的模块名，然后全仓扫描谁引用它
python -c "
import os
targets=['<changed_module_a>','<changed_module_b>']
for root,dirs,files in os.walk('.'):
    if any(s in root for s in ['.git','dist','node_modules','__pycache__']): continue
    for f in files:
        if not f.endswith('.py'): continue
        p=os.path.join(root,f)
        src=open(p,encoding='utf-8',errors='ignore').read()
        for t in targets:
            if t in src and t not in os.path.basename(p):
                print(t,'<-',p)
"
```

**2026-09-13 实例**：提交 `aa124f52` 改动了
`scripts/training/eval_taiji_p5_2a_predictive_execution_gate.py` 与
`scripts/training/eval_taiji_p5_2b_group_causal_corpora_gate.py`。
全仓扫描结果：**这两个模块只被 `tests/taiji_native/test_intervention_reality_gate.py`
（本次新增）引用**。其余测试在 import 层无法受影响 → 全部失败可证为既有。
这比「跑两次对比」更快、更硬。

配套观察（同一事件的旁证）：
- 全量套件的失败集合**每次运行都不同**（两次运行分别在不同进度点多出失败）。
- 失败用例单独运行（或小批组合）**全部通过**。

## 2. 量化基线（三次复采，可对比）

采集命令（**必须用 JUnit XML**，`-rf`/`--tb` 的文本输出会被工具截断丢失）：

```bash
python -m pytest tests/taiji_native/ -q --no-header --tb=no -p no:cacheprovider \
  --junitxml=<tmp>/taiji.xml
# 全量约 13–16 分钟，必须 run_in_background
```

| 指标 | `aa124f52`（建册基线） | `f9825943`+B0（2026-09-13） | **`fba517cf`（2026-09-14，当前 HEAD）** |
|---|---|---|---|
| 用例总数 | 788 | 851（+63） | **949（+98）** |
| 通过 | — | 823 | **921** |
| 失败 | **30** | **27** | **27** |
| 错误（error） | 0 | 0 | 0 |
| 跳过（skipped） | 1 | 1 | 1 |
| 全量墙钟 | ~15 分钟 | 938s | **801s** |
| **类别 A（架构边界违反）** | **2** | **0 —— 已结项** | **0** |
| 类别 B（`SystemExit: 1`） | 28 | 27 | **27** |
| 失败集合 vs 上一基线 | — | 旧 28 项的严格子集 | **逐位相同：0 新增、0 消失** |

**2026-09-14 复采结论**：B0 十一轮新增的 **+98 个测试全部通过**，且**失败集合与 09-13 基线逐位相同**
（`new = ∅`、`gone = ∅`，用 JUnit XML 集合差集判定，不靠计数）。
这**预验证了 WP-3 出口判据④**。注意：本仓库既有失败形态正是**顺序/状态污染**，
所以"局部目标集通过"不足以说明问题，必须在**全量顺序上下文**下比对集合。

**复采结论（2026-09-13，B0 本轮）**：

1. **类别 A 已结项且可复现**：`test_architecture_contract` 与 `test_naming_boundary_contract`
   在**全量上下文**下也转绿（此前只有目标集 64 passed 的局部证据）。
2. **类别 B 是严格子集**：本轮 27 项全部落在旧 28 项之内，**无新增失败**；唯一消失的是
   `test_natural_language_workbench::test_natural_language_workbench_gate_passes`。
   这既符合"失败集合每次不同"的既有观测，也进一步支持"顺序/状态污染"而非逻辑缺陷的定性。
3. 27 项仍全部是 `SystemExit: 1`，仍无可读栈 ⇒ 登记册 §4 的采集障碍**未解决**，
   §8 处置入口条件第 2 条仍未满足。
4. 采集命令与原始 XML 路径：`C:/Users/23747/AppData/Local/Temp/taiji_b0_full.xml`（临时文件，不入库）。
   解析脚本模式见[B0 机制文档](../../reference/M5_DISTILLATION_TOMBSTONE.md) §7。
5. **（2026-09-14 追加）集合比对优于计数比对**：当前 HEAD 与 09-13 基线的失败集合
   **逐位相同**（0 新增、0 消失），说明 09-13 那次"消失 1 项"确实是运行间抖动，
   而本轮 +98 个新测试**没有引入任何新失败**。判定脚本见本节末。

```python
# 失败集合差集判定（比对比计数更硬）
import xml.etree.ElementTree as ET
def load(p):
    r = ET.parse(p).getroot()
    return {tc.get("classname") + "::" + tc.get("name")
            for tc in r.iter("testcase")
            if tc.find("failure") is not None or tc.find("error") is not None}
new = load("now.xml") - load("baseline.xml")   # 必须为空
gone = load("baseline.xml") - load("now.xml")  # 记录，用于识别抖动
```

失败分为三类（**分类很重要**：类别决定了严重性和修法）：

| 类别 | 数量 | 特征 | 严重性 |
|---|---|---|---|
| A. 架构边界违反 | **0**（建册时 2，已结项） | 断言失败，非 `SystemExit` | ~~高~~ 已闭环 |
| B. `SystemExit: 1` 级联 | 27（建册时 28） | 调用 gate `evaluate()` 前即退出 | 中（环境/状态污染，非逻辑错误） |

> 类别 A 的结项记录见 §3 两项的【已解决】标注与 [B0 机制文档](../../reference/M5_DISTILLATION_TOMBSTONE.md) §7。
> §6 的 28 项清单保留为 `aa124f52` 的历史记录，**当前计数为 27**（严格子集）。

## 3. 类别 A：架构边界违反（2 项，高）

这两项**不是**环境污染，是真实的边界破坏，需要单独定性。

### DEBT-A1 · 原生核心引入 legacy/序列模型依赖

- 用例：`tests.taiji_native.test_architecture_contract::test_native_core_has_no_legacy_or_sequence_model_dependency`
- 失败形态：`AssertionError: {'__future__','adapter','adaptive_residual_bridge','adaptive_residual_candidate','adaptive_residual_growth',...}`
- 含义：原生核心的依赖集合里出现了不该出现的符号/模块，测试白名单未覆盖。
- 待查：是**新增了真实违规依赖**，还是**白名单未随重构更新**。两者修法完全相反，必须先定性再动手。
- **【已解决 2026-09-13】** 定性结论：**真违规，且与 DEBT-A2 同根因**。A1 的失败信息是整个
  `imported` 集合的噪声转储，实际触发项是 `taiji/document_embedding.py` 贡献的 `transformers`
  （命名边界测试递归证明 taiji/ 内仅此一个文件含违禁 top-level 导入）。修复见 DEBT-A2：
  文件迁出 taiji/ 后本用例恢复通过。

### DEBT-A2 · `taiji/document_embedding.py` 引入 transformers

- 用例：`tests.taiji_native.test_naming_boundary_contract::test_taiji_substrate_never_imports_legacy_or_transformers`
- 失败形态：`AssertionError: Taiji 是自足认知架构，不得依赖 Seed 运行时、seed_platform 治理层、Legacy NeuroPlex 或 HuggingFace transformers：{'taiji/document_embedding.py': {'transformers'...}}`
- 含义：**`taiji/document_embedding.py` 直接依赖 HuggingFace transformers**。
- 影响面较大：`DocumentEmbedder` 是 P5.2b/P5.2c 群体语料的 cue 来源（`embedder.embed([goal_text])[0]`，384 维）。
  这是「原生基底自足性」的核心边界，不是可以随手放宽的白名单。
- 待查：该依赖是历史遗留还是为 P5.1d 语义 encoder 注入所必需；若是后者，需在设计层决定
  「锚定 encoder 是否允许外部依赖」，而不是改测试。
- **【已解决 2026-09-13】** 定性结论（回应原「待查」）：该依赖是 P5.1d 语义 encoder 的
  **功能性必需**（锚定 embedder），不是历史遗留；因此按设计层决策处理——**taiji/ 不再持有
  该依赖**，而不是改测试。修复内容（随本提交落地）：
  1. `taiji/document_embedding.py` → 顶层新包 `instruments/document_embedding.py`；
     checkpoint payload 格式 `taiji-document-embedder-v1` 不变，digest 锚与既有预注册兼容；
     依赖方向 instruments → taiji 单向（仅 `content_digest`），taiji/ 对 instruments 零引用。
  2. `taiji/artifact_internalization.py` 依赖倒置：删除 `from .document_embedding import ...`；
     `SemanticArtifactKnowledgeEncoder(embedder=...)` 改必选注入（None 即 ValueError）；
     `from_checkpoint(..., *, embedder)` 锚校验语义保留；
     `ArtifactInternalizationTrainer.from_checkpoint(..., *, embedder=None)` 对语义 payload
     无注入即 ValueError（fail closed）。
  3. scripts/training 17 个导入行批量更新；8 处 `from_checkpoint` 语义调用点注入 embedder。
  4. pyproject packages.find 增加 `instruments*`；新增 3 个回归测试钉住 fail-closed 与锚漂移。
- 验证：两个契约测试转绿；目标集（architecture/naming/artifact_internalization/
  intervention/P5.2c'''/project_identity）**64 passed**；21 文件 py_compile + ruff 0 错误；
  冒烟确认 `import taiji` 后 `sys.modules` 无 instruments/transformers。
- 残余：p5_1b/1d/1e/1f/1g 等 gate 的完整 `evaluate()`（15 分钟级）未重跑，由 §8(4)
  处置阶段统一重采基线覆盖。

## 4. 类别 B：`SystemExit: 1` 级联（28 项，中）

> **【已定性 2026-09-14】真因不是项目代码，而是本地 WorkBuddy 沙箱的 safe-delete 批量删除守卫。**
> 该守卫经 `sitecustomize.py`（PYTHONPATH 注入）劫持 `pathlib.Path.unlink` / `os.remove`，
> 按「单个 tool call 内删除 ≥50 个路径」计数，超限即 `raise SystemExit(1)`；
> 于是**此后所有做清理的用例都被中断**。这解释了本节的每一项观测：
> 失败集合每次不同（取决于计数器何时跨限）、单独跑全过、集中在后段、拿不到栈。
>
> **实测证据**：以 `--tb=long -o junit_logging=all` 复采，45 项失败的完整栈**全部**终止于
> `D:\WorkBuddy\...\cli\vendor\shim\sitecustomize.py:826 (_exit_bulk_guard_control)`，
> 中间帧为 `_safe_path_unlink → _try_trash → _check_bulk_delete_guard`；
> 同一套件加 `CODEBUDDY_SAFE_DELETE_ENABLED=0` 后失败数 **45 → 5**（详见 §4.1）。
>
> **对 CI 的含义**：GitHub Actions 无此守卫，本类别**不构成项目债务**；
> 本地复跑 `tests/` 必须设 `CODEBUDDY_SAFE_DELETE_ENABLED=0`，否则读到的是环境伪影而非真实结果。

### 现象

28 个用例在**全量套件上下文**中以 `SystemExit: 1` 失败，单独运行或小批组合**全部通过**。

代表用例（完整清单见 §6）：

- `test_semantic_grounding::test_semantic_grounding_gate_passes`
- `test_terminal_three_domain_governance::test_terminal_three_domain_governance_gate`
- `test_runtime_artifact_store_*`（4 项）
- `test_runtime_structural_artifact_*`（5 项）
- `test_structural_artifact_store` / `test_structural_artifact_measurement_*`（3 项）
- `test_structural_lineage_artifact_*` / `test_structural_lineage_restart_*`（6 项）

### 已验证的事实

- 这些测试**在进程内**调用 gate 的 `evaluate()`（如 `from scripts.training.eval_taiji_semantic_grounding import evaluate`），
  不是 subprocess。
- 单独调用 `evaluate()` → 正常返回，`gate.passed = True`。
- 失败时**没有可用的 traceback**：pytest 的 JUnit writer 在此配置下对 `SystemExit` 不写栈，
  `failure.text` 仅有 `E SystemExit: 1`。**这是采集障碍，处置时需先解决可观测性。**

### 已排除

- 不是 `eval_taiji_semantic_grounding.py` 自身调用 `sys.exit`（模块内无该调用）。
- 不是两文件组合可复现（已试 `test_sequence_learning + test_semantic_grounding`、
  `test_semantic_provider* + test_semantic_grounding`，均通过）。
- 不属于本次改动（§1 引用图已证）。

### 已推翻的原假设（保留记录，2026-09-14）

原首要嫌疑是 `tests/conftest.py` 的会话级 fixture `_reset_global_app_state`
**只在 session teardown 重置** `seed_platform.app_state` 单例，于是 788 个用例共享同一单例。
**该假设已被实测推翻**：`AppState` 只有 22 个 api 层字段（trainer/model/tokenizer/locks），
构造开销 0.003 ms；且 `test_runtime_*` / `test_structural_*` 这些失败 gate **不读写 app_state**。
真正的停点在有完整栈时一目了然（见本节开头的实测证据）。

**方法教训（与 §1 并列）**：`SystemExit` 级联在**拿到栈之前**不要猜根因——本项目已因此
把归因写错一轮。可观测性（`--tb=long -o junit_logging=all`）应先于假设建立。

### 4.1 真实回归（关闭守卫后剩余的 5 项，已修）

关闭守卫后剩余的 5 项**不是**污染，而是**修复轮引入的真实回归**：

- 失败点统一在 `eval_taiji_workbench_multi_region_batch.py:97`，形如
  `workspace.read` → `error_code: not_found`（`README.md` 不存在）。
- 根因：`SeedRuntime.load(...)` **不保留**构造时注入的 `workspace_root`
  （override 只在 `__init__` 生效），于是恢复后的 runtime 回退到产品默认工作区
  `default_workspace_root()` → `agent_workspace`；`README.md` 自然找不到。
- 旧代码被 **gate 脚本模块级 patch** `workbench_module.get_setting`（全局副作用，
  本身就是污染源）掩盖；修复轮把 patch 换成显式 `workspace_root=` 注入时，
  **只改了构造、漏改了 `load`**，于是暴露。

处置：对 19 个使用「仓库读取能力」的脚本（import multi_region 的 `_build_runtime` /
`_execute_observation` / `_record_round`）+ 2 个直接调用该能力的测试文件，
统一给 `SeedRuntime.load(...)` 显式传 `workspace_root=PROJECT_ROOT`。
验证：6/6 定向用例通过；全量套件在守卫关闭下 **0 失败**。

### 4.2 跨解释器数值一致性（Python 3.10 腿，2 项，**未修——待决策**）

**发现（2026-09-14 深夜，run `34860023522` @`292b66a5`）**：CI 的 `test (3.12)` 与
`test-windows` **首次全绿**（28 步 / 19 步全 success，含 `complete regression suite`），
仅 `test (3.10)` 在步骤 24 有 2 项失败（该 job 的 JUnit 显示 960 用例中仅此 2 项）：

| 用例 | 失败形态 |
|---|---|
| `test_continuous_structural_growth::test_continuous_structural_growth_gate` | `AssertionError: second-cycle online feedback was not admitted: online-de-next` |
| `test_cap0_inventory_contract::test_a_fresh_inventory_sample_reproduces_the_sealed_one` | **2026-09-30 归属完成（与上面 F04 那条不是同一个头，别并成一条）**：失败是"封存清单外新增字段 `checkpoint_inventory[6].*`"。直接调 `_checkpoint_inventory()` 拿到现行 9 项，逐项点名后有两件事：①**多出的第 6 项之后是 `seed_corpus.pt`／`seed_corpus_prev_20260823.pt`／`seed_native.pt` 这类正常档案**，但清单里还enumerated进两枚**隐藏临时件** `checkpoints/.p2-12-conflict.pt`、`.p2-12-natural-language-write.pt`（各 43,309,755／43,316,555 B，mtime **2026-09-13 20:24**，`git status --ignored` 显示被忽略 ⇒ 不是别会话在飞，是 09-13 那轮留下的残件）——**枚举器把 `.` 开头的临时档也算成清单条目，这是仪器自身的缺陷**；②`seed_beta_with_circuit.pt` 现在 `is_default=True`（与 F04 那条同一个产品变更），所以即便去掉残件，封存期望里"默认是哪枚"那一格也已过期。**处置边界**：一行改法（枚举时跳过 `.` 前缀）能消掉①，但那会**改动 CAP 评价仪器的面并可能把红直接翻绿**，且重封 inventory 要动封存金样 ⇒ 属主线 CAP/R2 那条线的动作，我不在此单方面改；这里只把"红在哪个字段、由哪两半造成、残件不是别人在飞"钉下来。 |
| `test_cap0_f_dimension_contract::test_f04_on_the_current_eval_set_is_not_flagged_stale` | **2026-09-30 归属完成（守卫是对的，红是产品变更的记账后果）**：F04 的时效子句 `sealed inventory 描述的是当前产品默认基座` held=False，原文是"封存 inventory 里 `default_checkpoint='seed_beta.pt'`，而现行产品默认＝`seed_beta_with_circuit.pt`"⇒ **rev51 那次"电路随出厂基座装"把 `DEFAULT_CHECKPOINT` 换到 with_circuit 件（`api/seed_runtime.py:40-45`）之后，CAP0 的封存清单就描述错了基座**，门据此判 `stale_reference`（同一条目另有 `templated=True` 那条 held=False 是 F04 本来就是 `fail`，与此次红无关）。**处置＝重封一份指向新默认件的 inventory**——那会动到封存金样，属**产品默认位那条线**的动作，不由 A 支线单方面刷绿；A 支线这边要留住的是它的**副作用**：§2ab 判据第 3 条（F0 不回退 ≥0.5）读的正是这套 F 维判定，**默认位一换，F0 的读数面就跟着换**，引用旧 F0 数必须点名是哪枚默认件。 |
| `test_interaction_group_multifamily::test_interaction_group_multifamily_leave_one_out_gate` | `AssertionError: selector did not choose a group for held-out complementary-alpha`（`eval_taiji_interaction_group_multifamily.py:159`，即 `InteractionGroupUtilityLearner.select(resource_budget=2.0)` 返回 `None`） |

**性质**：**既有**——3.10 腿此前一直被 black / verify 网关 / timeout 挡在步骤 24 之前，从未执行到。

**已排除的假设**：
- ❌ **顺序 / 哈希不确定**：`taiji/interaction_groups.py::train_only_candidates` 已是确定性的
  （`tuple(sorted({...}))` + `itertools.combinations`），且该脚本用 `seed % 2` 显式对两种顺序都测；
- ❌ **torch 版本差异**：从 CI 日志核对，两腿均装 `2.14.0+cpu`。

**剩余怀疑**：CPython 3.10 与 3.12 的浮点 / 容器迭代边界差异，使候选在 `_estimate_pair`
的资源 / 效用阈值处被判到不同侧。**本机无法复现**（本机只有 3.12.10 / 3.13.12）。

**🔍 已诊断（2026-09-15，只读）：根因是「恰好压线」，不是数值噪声**

只读探针 [`diagnose_p3b_s42_boundary.py`](../../../scripts/training/diagnose_p3b_s42_boundary.py)
在 3.12 上复现 3 family × 3 seed 的 leave-one-out 候选与 learner 状态，结果：

**9 / 9 case 的 `closest_boundary_margin = 0.0`** —— 候选的 `utility` 或 `resource_cost`
**精确压在阈值上**（`utility >= minimum_utility(0.0)`、`resource_cost <= budget(2.0)`），
**±1e-12 即可翻转 `select` 的结论**；而 `select` 本身**没有任何容差**
（`interaction = pair - first - second` 与 `pair_resource_cost` 均值都是浮点量）。
⇒ **CPython 3.10 与 3.12 的浮点差异足以触发该翻转**，这解释了 3.10 腿的失败。

**修法建议（**未实施**，需决策）**：**A（推荐）** 给 `select` 的两个比较加显式容差
`eps = 1e-9`（语义只影响"恰好压线"，正是 gate 想表达的意思）；
**B** 不动比较、由 multifamily gate 在阈值/预算上留余量。
详见[诊断文档](../../reference/M5_DISTILLATION_TOMBSTONE.md) §4。
**铁证仍需在 3.10 腿补一次带该探针的 job**（本机无 3.10）。

**✅ 边界语义已加契约测试（2026-09-15）**：
`tests/taiji_native/test_interaction_group_learner_boundary_contract.py`（**12 passed**）钉住 ——
`utility == minimum_utility` 与 `resource_cost == budget`（**margin = 0**）**必须被选中**；
略负 / 略超界即不选；`-0.0` 被选中；`budget=None` 忽略 cost；负 budget 报错；
tie-break 顺序确定（utility → cost → group_id）；并**复述诊断报告**（所有 case
`closest_boundary_margin == 0.0`、±1e-12 即翻转）。

顺带在测试里显式记录一条浮点事实：**`2.0 + 1e-18 == 2.0`**（增量被舍入吞掉）⇒
"构造刚过界"必须用**可表示**的 ε（如 `1e-12`）；边界敏感的真实尺度由双精度相对精度
（≈2.2e-16 × 量级）决定。
⇒ 现在**任何对阈值 / 比较符 / 排序的改动都会显式失败**，而不是静默改变被测机制的含义。

**✅ 决策已下（2026-09-30，H19c）：删 3.10 腿，不修 A/B** —— CI `test` job 的矩阵由
`["3.10","3.12"]` 收窄为 `["3.12"]`。选它而不是给 `select` 加 `eps`，理由是 3.10 腿自
2026-09-14 起就是一条**修不了、也无法本地证伪**的红腿（本机只有 3.12.10 / 3.13.12），
继续挂着只会训练所有人忽略 CI；Dockerfile 本就发 `python:3.12-slim`，双矩阵是在测一个
谁也不发布的版本。

本决策**明确不解决**的、以及它使哪些取证变难，逐条留名：

1. **`select` 零容差的脆弱性原样留在 3.12 上**。诊断已在 3.12 上复现：9/9 case 的
   `closest_boundary_margin == 0.0`，±1e-12 即翻转 `select` 结论，而 `select` 无任何容差。
   删腿移走的是**症状**，不是这个**病因**。要真正了结，仍需走上面的 A（`eps = 1e-9`）或 B。
2. **上面第 868 行「铁证仍需在 3.10 腿补一次带该探针的 job」在本矩阵下已取不到**。
   跨解释器翻转的实锤现在只能靠：(a) 手动/disp 触发的临时 3.10 job，或 (b) 任何一台装了
   3.10 的机器本地跑 `diagnose_p3b_s42_boundary.py`。谁要补这份铁证，请勿假设 CI 会提供。
3. **跨版本数值一致性不再被持续测量**：`test_continuous_structural_growth` 与
   `test_interaction_group_multifamily` 在 3.12 上可能绿、在 3.10 上会红而无人知晓。
4. **3.10 的"支持"并未撤销**（`pyproject` 仍 `requires-python >=3.10`，两处 tomllib/tomli
   shim 保留），但 3.10 路径从此**不被任何 CI 覆盖**——绿色运行不能当作 3.10 可用的证据。

仍然有效的既有保障：上面那条边界语义契约测试（12 passed）钉住了 `margin == 0` 必须被选中，
所以风险形态是「浮点噪声下静默改变 tie-break」，而不是「行为未定义」。

**✅ §4.2 的修法已实施（H19d，2026-09-30）——但不是本节原先提议的那个**

本节上面「修法建议 A」写的是「给 `select` 的两个比较加显式容差 `eps = 1e-9`」。**该方案被
否决**，理由是它会打破当时已存在的两条边界契约 pin（`utility = -1e-18` 必须不选、
`cost = 2.0 + 1e-12` 必须不选）——即 tripwire 按设计发挥了作用，而不是被绕过。

改为先读诊断产物再定方案。`reports/taiji_p3b_s42_boundary_diagnosis_20260915.json`
的 9 个 case 实际是：

| group | utility | utility_margin | resource_cost | cost_margin |
|---|---|---|---|---|
| `group:3960…` | -1.0 | **-1.0** | 2.0 | **0.0** |
| `group:cd7b…` | 0.5 | **0.5** | 2.0 | **0.0** |

即**压线的是 cost 侧**（`resource_cost == budget` 恰好相等），utility 侧最近的候选距 0.0
阈值还有 0.5 —— 两处比较并不对称，`1e-9` 是按「对称地加在两处」这个错误前提选的。

实施（`taiji/interaction_group_learning.py`）：

- **cost 侧**吸收 `_BUDGET_ULP_TOLERANCE = 8.0` 个 ULP，用 `math.ulp(budget)` 缩放，
  故对任意预算量级都成立（固定 `1e-9` 在 budget=1e6 时毫无意义）。
- **utility 侧刻意保持精确比较**：`interaction` 是 `_estimate_pair` 里
  `m_TT − m_TF − m_FT + m_FF` 的四项交替求和，抵消误差约 1e-15，而契约测试钉的是
  1e-18 —— 那是一个比被测量本身分辨率还细的区分，放宽它等于用一个有文档的语义换零收益。
- `select` 是只读查询，不写 checkpoint，故此改动**不触及任何 checkpoint digest / 封存件**。

新增两条 pin（`test_interaction_group_learner_boundary_contract.py`，现 14 passed）：

1. `test_cost_one_ulp_over_budget_is_still_selected` —— 直接钉住那条红的形态：
   `cost == budget + math.ulp(budget)` 必须被选中。
2. `test_cost_tolerance_does_not_swallow_a_real_overage` —— 防回潮：`budget + 1e-9`
   仍必须被拒，即明确拒绝本节原先提议的那个尺度。

**本节第 2 条「铁证仍需在 3.10 腿补一次带该探针的 job」就此作废**：该修复现在由上面第 1 条
pin 直接在进程内钉住（`budget + 1 ULP` 即失败形态），不再需要跨解释器复跑来取证。

**✅ utility 侧的上游归位也已实施（H19e）**：`_estimate_pair` 新增 `_snap_cancel_residual`，
对 `interaction` 与 `recovery_interaction` 两个四项交替求和的结果做规范化——**当残差落在
操作数量级的几个 ULP 之内时归为精确 0.0**，否则原样返回。

- 容差由**操作数**量级推出（`_CANCEL_RESIDUAL_ULPS * math.ulp(scale)`），不是由结果推出：
  近零结果自身的 ULP 是非规格数级的，据此算容差等于没有容差。
- 只动噪声：~1e-15（单位量级）的容差无法搬动任何有意义的取值，只有恰好压在 0.0 阈值上
  的判定会变——而那正是要消除的模糊性。
- 分层职责由此清晰：**计算层把自己的抵消噪声归位，选择层保持精确比较**。契约测试里
  `utility = -1e-18` 必须被拒那条依然成立且未被放宽（它走直接构造 record 的路径，
  不经过 `_estimate_pair`）。
- 新增 3 条 pin（同一文件，现 17 passed）：残差 ±1e-16 都必须变成精确 0.0；有意义的取值
  （0.5 / -0.5 / ±1e-12）必须原样通过；容差随操作数量级缩放（量级 1e6 时 8 ULP ≈ 9.3e-10，
  故 1e-9 在那才是噪声，而在量级 1.0 时 1e-9 远大于容差必须保留）。

**注意这是会改变记录值的改动**：`interaction` / `recovery_interaction` 会原样进入
`InteractionGroupRecord`（`interaction`、`recovery_effect`、`holdout_*`）并进入 learner 的
checkpoint 载荷。全量回归 2335 passed / 0 failed 作为该改动的验收证据；若将来出现依赖旧
数值的历史封存件，那是预期的口径变化，不是回归。

**处置约束（重要）**：这 2 项触及 `InteractionGroupUtilityLearner`——**被多个既有报告依赖的
冻结机制**；任何阈值或 tie-break 改动都会影响与既有报告的可比性，须先定性再动手，
并遵守 §8「不得放宽断言凑绿」的纪律。

**候选处置**：① 在 3.10 腿复跑该 job，确认是稳定复现还是 flaky（区分两类根因）；
② 对 `_estimate_pair` / `select` 的阈值比较引入显式容差或确定性 tie-break（需预注册式审慎）；
③ 给这 2 项 `xfail(strict=False)` 并注明「3.10 已知数值差异」（会弱化该腿门禁，需明确认可）。

### 4.3 两项交接债（2026-09-15 发现，**未处理**）

#### （一）7 个文件的 black 格式债与锚点契约冲突

夜间 WP-2 / WP-3 提交的 7 个文件不符合 `black --check .`：
`test_b0_n2_stop_reason_disposition_contract.py`、`test_b0_rule_revision_seal_contract.py`、
`test_b0_n2_stop_reason_semantics_contract.py`、`test_b0_m1_counterfactual_contract.py`、
`test_b0_structure_space_contract.py`、`probe_taiji_b0_structure_space.py`、
`eval_taiji_p5_2b_group_causal_corpora_gate.py`。

**已实测的冲突**：直接 `black` 这 7 个文件后出现 **8 项契约测试失败**
（`test_b0_m1_counterfactual_contract` 5 项、`test_b0_rule_revision_seal_contract` 2 项、
`test_b0_n2_stop_reason_disposition_contract` 1 项）——black 重排了**锚点所在的代码行**，
而 WP-3 的锚点/规则文本要求**逐字节匹配**。⇒ 已回滚，**未提交任何格式化**。

**处置选项**：① 对锚点区域加 `# fmt: off` / `# fmt: on` 包裹、其余部分照常格式化（推荐）；
② 把锚点判定改为不依赖格式（AST / 正则）；③ 在 `[tool.black] extend-exclude` 中豁免
（等于放宽门禁，需明确认可）。

**✅ 已处置（2026-09-15，采用选项 ①）**：

- 在 `eval_taiji_p5_2b_group_causal_corpora_gate.py` 里，对被反事实锚点/替换文本**逐字节匹配**
  的两个区域加 `# fmt: off` / `# fmt: on`：`M2_CUE` 覆盖区（原 L256-260）与 `M4_SELECTION`
  覆盖区（原 L282-301）；其余部分照常格式化。**未改任何测试、未改测量文本**。
- 结果：7 个文件全部格式化，`black --check .` ⇒ **1169 files unchanged（0 待格式化）**；
  `M2_CUE` / `M4_SELECTION` 在格式化后**仍逐字节存在于 gate 源码**（用 AST 取值后 `in` 校验）；
  b0 契约组 **172 passed / 1 failed**（该 1 项即下方 (二) 的 N2 清单漂移，与格式化无关）。
- **关键经验（实测 black 26.5.1）**：`# fmt: off` **必须顶格（行首无缩进）才生效** ——
  带块内缩进的 `# fmt: off` 会被照常格式化；顶格的会被 black 规范化为块内注释且**保护依然有效**。**在处置前，CI 的 `Format check with black` 步骤会红。**

#### （二）N2 消费面清单漂移（1 项既有失败）

`tests/taiji_native/test_b0_n2_stop_reason_disposition_contract.py` 的
`test_current_review_surface_is_complete` 失败：live 扫描多出
`scripts/training/audit_taiji_b0_checkpoint_preflight.py`（WP-4 预检脚本，11:44 新增）。

该文件对 `stop_reason` 的用法只是 **payload 字段名**（`"terminal_stop_reason_marker"`），
按 N2 语义**不是判断点** ⇒ **属扫描器误报**，正确处置应是**排除该文件**，
而不是把它登记进 `EXPECTED_CONSUMERS`：已实测「登记 + 重新生成归档报告」会同时引出
`test_historical_inventory_is_not_rewritten` 与
`test_the_frozen_preregistration_states_the_measured_counts` 两项新失败
（**归档报告不得改写**是既定纪律），故已回滚。
⇒ 修复点在 `audit_taiji_b0_m4_hardening.py` 的消费者扫描规则，须与 N2 作者对齐后再改。

**✅ 已处置（2026-09-15，采用"改扫描器排除规则"）**：

根因是**扫描器用子串匹配**（`if "stop_reason" not in source: continue`），而
`audit_taiji_b0_checkpoint_preflight.py` 只输出 **payload 字段名**
`"terminal_stop_reason_marker"` ⇒ 含该子串 ⇒ 被误算作消费者（17 → 18）。

修法：在 `audit_taiji_b0_m4_hardening.py` 增加**审查排除名单** `SCAN_EXCLUSIONS`
（每条必须写明"为什么不构成消费"），在扫描循环里按**归一化相对路径**（`\` → `/`）跳过。
**未登记进 `EXPECTED_CONSUMERS`、未重写归档报告** —— 那条路已实测会牵出
`test_historical_inventory_is_not_rewritten` 等 2 项新失败（见上）。

结果：`live_consumers()` 回到 **17**（与 `EXPECTED_CONSUMERS` 及归档报告的
`consumer_count_now = 17` 一致），N2 契约测试 **10 passed**。

### 处置时需先建立的观测能力（已建立）

1. ✅ 让 `SystemExit` 带栈：CI 已加 `--tb=short --junitxml=... -o junit_logging=all`
   （`.github/workflows/ci.yml`），本次据此一次性定位真因。
2. 二分定位污染源（`--deselect`）——本次未需要，直接由栈定案。
3. 记录触发顺序——守卫计数是会话级，顺序仍应记入本地复跑说明。

## 5. 类别 C：既有 CI 口径差异（旁证，未计入 30）

- `.workbuddy`、`output/manual-r5-canary` 等路径此前已加 `.gitignore` 规则；
  `output/manual-r5-canary` 仍在盘上但被忽略，属预期。
- 全量测试运行中出现沙箱批量删除守卫提示
  （`[safe-delete][SAFE_DELETE_BULK_CONFIRM_REQUIRED] {"count":85,...pytest-of-...garbage-...}`），
  为 pytest tmp 目录清理触发，**未观察到影响测试结果**，仅记录以备后续排查。
- **新增（2026-09-13，B0 全量运行后观察）**：全量跑完后 `reports/` 下出现两个**未跟踪**的隐藏残留
  `reports/.p2-12-edit-fixture.txt`（`Seed editor source\nexternal change\n`，37B）与
  `reports/.p2-13-api-fixture.txt`（`Seed API source\n`，17B），mtime 落在本次运行期间。
  **未定位到写入者**（已在 `tests/`、`seed_platform/`、`api/` 范围内检索文件名与内容，均无匹配）。
  属测试把 fixture 写进仓库目录且未被 `.gitignore` 覆盖的卫生问题，**低严重性**。
  **本轮未删除**（避免误删未知来源产物）；入库时未纳入提交。处置阶段需定位写入者并改为 `tmp_path`。

## 6. 类别 B 完整失败清单（28 项，基线 `aa124f52`）

```
tests.taiji_native.test_natural_language_workbench::test_natural_language_workbench_gate_passes
tests.taiji_native.test_natural_language_workbench_api::test_natural_language_workbench_api_gate_passes
tests.taiji_native.test_natural_language_write::test_natural_language_write_gate_passes
tests.taiji_native.test_runtime_artifact_store_audit_projection::test_runtime_projects_external_store_audit_without_mutation
tests.taiji_native.test_runtime_artifact_store_bridge::test_runtime_artifact_store_bridge_validates_before_native_mutation
tests.taiji_native.test_runtime_artifact_store_preflight::test_runtime_artifact_store_preflights_all_candidates_before_mutation
tests.taiji_native.test_runtime_artifact_store_runtime_reconciliation::test_runtime_store_reconciliation_distinguishes_missing_and_orphan
tests.taiji_native.test_runtime_multi_artifact_store_batch::test_runtime_multi_artifact_store_batch_preserves_parent_order
tests.taiji_native.test_runtime_retention_store_audit::test_store_audit_is_read_only_and_reports_runtime_orphans
tests.taiji_native.test_runtime_retention_store_separation::test_runtime_retention_does_not_delete_or_resurrect_external_artifacts
tests.taiji_native.test_runtime_structural_artifact_batch::test_seed_runtime_restarts_and_consumes_measured_artifact_batch
tests.taiji_native.test_runtime_structural_artifact_failure_concurrency::test_runtime_artifact_failures_are_isolated_and_concurrent_submit_is_idempotent
tests.taiji_native.test_runtime_structural_artifact_multi_round::test_runtime_artifact_multi_round_lifecycle_and_retention
tests.taiji_native.test_runtime_structural_artifact_post_retention::test_runtime_artifact_continues_after_retention_and_restart
tests.taiji_native.test_runtime_structural_artifact_repeated_retention::test_runtime_artifact_repeated_retention_stays_bounded
tests.taiji_native.test_runtime_verified_measurement_bridge::test_verified_measurement_bridge_is_opt_in_and_all_or_nothing
tests.taiji_native.test_semantic_grounding::test_semantic_grounding_gate_passes
tests.taiji_native.test_structural_artifact_measurement_bundle_recovery::test_partial_measurement_bundle_fails_closed_then_recovers_explicitly
tests.taiji_native.test_structural_artifact_measurement_sidecar::test_measured_artifact_sidecar_is_verified_and_legacy_is_explicit
tests.taiji_native.test_structural_artifact_store::test_external_artifact_store_is_immutable_and_runtime_consumable
tests.taiji_native.test_structural_lineage_artifact_batch_isolation::test_artifact_batch_rejects_unknown_keys_and_isolates_partial_failure
tests.taiji_native.test_structural_lineage_artifact_rollback::test_artifact_provenance_survives_rollback_and_terminal_compaction
tests.taiji_native.test_structural_lineage_disk_checkpoint::test_seed_runtime_disk_checkpoint_preserves_migration_and_rollback
tests.taiji_native.test_structural_lineage_multi_batch_artifact::test_multi_batch_artifact_retention_preserves_active_lineage
tests.taiji_native.test_structural_lineage_restart_admission::test_restart_candidate_admission_and_rollback_continue_from_checkpoint
tests.taiji_native.test_structural_lineage_restart_artifact::test_restart_replay_bound_artifact_continues_and_rejects_tampering
tests.taiji_native.test_structural_lineage_restart_continuation::test_restart_continuation_consumes_only_new_evidence
tests.taiji_native.test_terminal_three_domain_governance::test_terminal_three_domain_governance_gate
```

## 7. 其他已知遗留（非测试）

| 编号 | 内容 | 严重性 | 处置约束 |
|---|---|---|---|
| DEBT-G1 | `.git` 内 114 个 dangling 对象 | 低 | **确认无需回溯历史前不要跑 `git gc` / `git prune`** |
| DEBT-G2 | reflog 文本坏行（218 条 `error: invalid reflog entry`），`git fsck` exit 2 | 低 | 仅文本坏，不影响对象与 refs；`HEAD` 可达缺失对象 = 0 |
| DEBT-G3 | `.git/index.corrupt-backup`、`.git/logs/*.corrupt-backup` 备份文件 | 低 | 内含仅存的历史哈希线索，**不要删** |
| DEBT-G4 | 全量测试约 15 分钟且会 SIGTERM，需后台运行或分批 | 低 | 采集一律用 `--junitxml` + `run_in_background` |
| DEBT-G5 | **流程债**（红本身已于 2026-09-28 结清）：裁定 (d) 翻 `lock_selection_rule` 默认后，主干上留着 2 条红守卫（`test_copy_circuit_contract.py` 的两条**旧面**不变量），而那次提交报的是"定向回归 166 条全绿"——**不含它所改模块的既有契约面** | 中 | 任何翻 `TaijiConfig` 默认键的改动，必须连该模块的契约文件一起跑（`-k "copy or circuit or selection"`）；归因与修法见 [PLAN-A-28 §3.1](../../archive/history/a_line_20261007/README.md) |
| DEBT-G6 | v1 信封**重存成 v10 产品信封不是尺寸中性的**：`seed_beta.pt` 4.14 MB → 87.9 MB，其中 `.taiji.kernel`＋`.substrate` 双镜像各 43.9 MB，最大单块是**身份器官的空路由键仓 38.93 MB**（零面普查那张），复制回路本身只 0.18 MB。**进展（2026-09-28，PLAN-A-29 乙档）**：路由仓改按档里已有的 `value_counts` 截断存盘 ⇒ 带回路信封 **87.9 → 12.3 MB**（回装 260 张量逐位无损、器官摘要消费面 40 条未重新钉仍全绿）；**未闭合**：剩 4.7 MB 是散在多个小结构上的全零（`bank.prototypes`、`cortical_readout.edge_weight`…），没有第二块"结构性空仓"可按计数截，再降只能走通用零省略（语义风险已登记）；**运行时那 75.5 MB 的零表分配也未动**（乙只管存档侧） | 中→低 | 换出厂基座前由 owner 在"接受 +8.2 MB 净增"与"授权通用零省略（约 7.4 MB）"之间选；若要连运行时一起解，走 PLAN-A-29 §2 的甲档 |
| DEBT-G7 | 门禁**面值**与账不符：本机同版本工具（ruff 0.16.4／black 26.5.1／mypy 2.3.1）复跑，`black --check` 有 2 档脏（含 `taiji/model.py`）、`ruff check` 有 1 处 I001、`mypy --follow-imports=silent taiji` = **13 错／4 文件**，而 CI 的棘轮写的是 `MYPY_CORE_BASELINE=0`；**这些在 HEAD 的副本上同样红**（非任一在飞提交引入） | 中 | 判"是我弄红的"之前先 `git show HEAD:<f>` 复跑比对；修它属门禁面值问题，不属 A 支线范围 |
| DEBT-G8 | 仪器收尾时才做路径展示 ⇒ 题集／`--out-report` 指到仓库外会**跑完全部计算之后**才在 `relative_to` 上崩（`score_taiji_r2_copy_surface_extension.py` 两处，实测白跑一趟） | 低 | 已改 `is_relative_to` 回退；其余 `probe_/score_/eval_` 仪器按同一形状排查（收尾的展示代码不许持有失败路径） |
| DEBT-G9 | **生成预算口径错配（由 PLAN-A-30 §2h 的定位跑带出）**：评测链的生成预算一直是 **24／32／64 字节**（`score_taiji_r2_copy_circuit_chat_cap.py`＝64、`probe_taiji_context_retention_curve.py`＝24、`probe_taiji_memory_wiring_audit.py`＝32），而产品出口 `SeedRuntime.chat()` 是 **256**。两条后果：①§2d/§2f 的**跨链**对照（`raw`/`raw_masked` vs `surface`）同时动了"链"与"预算"两个变量，"表层比原始链差"的归属未定（同预算对照臂见 PLAN-A-30 §2i）；②同一次跑实测 **72/72 次生成都不因边界符退出环**（18432 步逐位重放自检 0 错），但这**不能**读成"模型没学会停"：默认语料的答复长度**中位 417 字节**，256 预算只覆盖 **39.0%**、64 只覆盖 **18.0%** ⇒ 预算点普遍落在语料答复的中段，"不出边界符"与训练分布一致。真正被证的是 **(A) 掩码饿死否证**（0/18432 步合法候选 ≤1）与**拖写＝内容退化** | 中→高（属能力面，不只是口径） | 已在仪器上加 `--max-bytes`（`_answer_raw` 新增同名参数，默认 64 ⇒ 冻结面逐位不变）；跨链比较必须同预算。停止信号机理**已定**（§2j/§2s/§2w：结束位名次 11／180（masked 面；full 面 21／257，归一 6.1% 对 8.2%）、概率 154× 于其它位，却 0/400 成为 argmax（两枚面同 ⇒ 掩码不是那只手）；而轮结束以换行＋`问：`的字节形态表达时真下一符号胜出 40/81＝49.4%，但普通位置基线本有 21.1% ⇒ **只是微弱倾向，不足以支撑更换停止判据**；硬结论仍是边界符 0/400 从不胜出（§2u 已把 §2s 那条线索打折），判据见 PLAN-A-30 §2h 末"语料长度修正"与 §3 丁-1（`audit_taiji_a30_stop_signal_presence.py`＋同件 `--max-length 1024` 档，2026-09-28 夜已在跑）；**命中随预算的方向与成句相反**（seed-A 同链同装配 64→256：命中 3→9、成句 13→6）⇒ 但预算位移是**面相关**的（X 面 raw_masked 3→9 超线；D/E 面同子集 7→8 未超线 — 见 PLAN-A-30 §2l）⇒ 引用命中必须点名"哪个面 × 哪个预算"，不许一概按**下界**读；另 cap 记分器的成句量的是 `answer[:60]` **字符前缀**，与 a30 两探针的全文口径不可互换（两处已加披露字段） |
| DEBT-G10 | **产品把自己写坏的答复喂回训练面，后果已实测到"会话间自我污染"**：`SeedRuntime.chat()` 缺省 `learn=True` ⇒ 收尾 `learn_bytes(text + answer)`（`api/seed_runtime.py:446-452`），同一答复又会作为历史被拼进后续轮的 prompt 并被逐字节喂进动力学取 cue（**注意**：剪贴板 `store` 只记 user 轮，答复不以事件形态入库——先前写成"又入证据库"不准确）。**32 题 × 3 轮三臂实测**（挂回路装配，零线＝两条同装配控制臂逐位相同）：新会话首轮成句 **31/32 → 8/32（−23）**、`run≥20` 0→1；第 2 轮均值连写 **17.4→35.2（+17.8）**、最长 122→256、`run≥20` 7→10；第 3 轮命中 10→9 ⇒ **三项同向、远超 ≥3 线，"放大退化"成立**。8 题档当时各指标都在线下（`not_resolved`）——已作为"小样本给反向读数"的实例保留在 PLAN-A-30 §2m | **中，且只在挂回路装配上成立（出厂面 32 题同测：成句 31→32、连写均值降到 1.0 ⇒ 不退化反略有好处，见 PLAN-A-30 §2t）** | **并入"回路要不要随出厂基座装"那一笔裁定**（不再单独构成产品行为改动）；挂回路时的退化来自**权重回写**还是**后续 prompt 里的上一条答复**——**已细分（§2v 五臂 2×2，32 题档）：两条通道同向、各自都过 ≥3 线 ⇒ 落点是"两处都要门槛"，不是二选一**。分解：回写＝首轮成句 −23（回路开）／−11（回路断）、第 2 轮连写 +17.8／+3.1；回路＝首轮成句 −12、第 2 轮连写 +13.9（都在回写开时），而**不回写**时回路到第 3 轮才过线（+10.4）；断回路还独立把严格命中从 7→9、10→13 保住。仪器侧已建齐四格（新增 `learn_true_scrubbed`／`learn_false_scrubbed`；四臂版那格是对角线不是效应量），判读线与裁定表先于数写在 [PLAN-A-30 §2v](../../archive/history/a_line_20261007/README.md)，读数件 `reports/taiji_a30_self_contamination_32item_factorial_20260929.json`（五条守卫全真、`circuit_sha256` 已钉 seed-A）；已知弱点＝该档没存逐条 `answer_sha`，与 §2q 的复现只做到**聚合级**（v2 起件里已有 `answer_shas`），`learn=True` 的回写加门槛（只回写通过 `well_formed ∧ 长度上限`的答复）**并且**历史拼装也加门槛（坏答复不进 prompt）。读数见 [PLAN-A-30 §2q](../../archive/history/a_line_20261007/README.md) |
| DEBT-G11 | **广面门里有一支红是别会话的沙箱落点造成的，不是代码缺陷**：`test_source_face_is_the_git_face_not_the_whole_disk` 断言的多出项全部是 `taiji-harness/.dsh-sbx2/lock-audit/{readme_desktop_trim,audit_lock_importers,cast_sites}.py`；`.gitignore` 的 `/.dsh-sbx*/` 是**根锚定**的，管不到嵌套一层的那个目录 ⇒ 它进了"未忽略的未跟踪文件"面，而守卫的已知噪音名单里没有它 ⇒ **每个会话跑广面都会多这一条红** （本会话实测：1804 passed／4 failed 里它就是其一） | 中（它会把"基线红"越滚越大，掩盖真红） | 两条候选修法，都属于**建这些沙箱的人**而不是 A 支线：①把忽略规则改成能覆盖嵌套的形式（`**/.dsh-sbx*/`），或②给守卫的已知噪音名单加这一条并写明归属；本会话**不动 `.gitignore`**（改了会影响别人的检出面），只登记。**2026-09-29 复测把机制钉准（并否证我自己先前的一个猜测）**：失败的断言是 `whole_git <= whole_walked` 那一行（不是"未跟踪噪音"那侧），多出项已涨到 **4 个**（新增 `taiji-harness/.dsh-sbx2/classify-face.py`）；遍历面按 `SKIP_PREFIXES = (".venv", ".dsh-sbx")` 跳掉点目录、而 git 检出面不跳（`tests/seed/test_platform_boundary.py:11-19`）⇒ 两边口径不同。`git ls-files taiji-harness/.dsh-sbx2` ＝ **0** ⇒ 这些沙箱文件**并未入库**，别按"别人把沙箱提交了"去修。同一次复测还确认：本会话新增的未跟踪仪器**不在**这条红的名单里（`scripts/training` 那一层的逐目录断言是过的） **2026-10-03 04:06 复发、但触发面换了（用守卫自己的 helper 复算过，没沿用旧结论）**：这次不符合项 **7,680** 个 `.py`，**100% 落在 `taiji-harness/apps/desktop/.desktop-build/targets/win-x64/…`**（他线桌面打包把整个后端源码复制进 build 目录）；`.dsh-sbx2` 那族已被 `SKIP_PREFIXES=(".venv", ".dsh-sbx")` 挡掉、且 `git ls-files taiji-harness/.dsh-sbx2` 为 0 ⇒ **同一条"遍历面跳点目录、git 检出面不跳"的病，新的入口**：`_is_skipped_dir` 只检查 `parts[0]`，挡不住 `taiji-harness/` 里嵌套三层的 `.desktop-build`。一行修法（跳过判断按任一路径分段命中，或 walk 侧也跳 `.desktop-build`）仍归**建那棵 build 树的人**，本会话不动。 |
| DEBT-G12 | **命中与成句各量一头，缺一把"既要求词在场、又要求成句"的联合尺子**：PLAN-A-30 §2p 逐条读已入库件的 `rows[].answer` 后确认，回路在 D 维买到的严格命中形状是"被检索词反复出现"（`岩阿岩阿…`、`杭州杭州…`），而同一装配在同尺成句上从 71/72 掉到 26/72⇒ 任何单看命中数的排序都会得出错误结论。现有两把尺子（`expected_contains` 严格命中、`well_formed` 结构成句）各自只测一头 | 中（它决定"要不要装回路"这笔账怎么算） | 补一件联合判据：命中 ∧ `well_formed` 同时成立才计入（属仪器改动，不需 owner 裁定；落点＝`score_taiji_r2_copy_circuit_chat_cap` 与 `probe_taiji_a30_*` 共用一支判定）；**联合数已出（PLAN-A-30 §2r）：全量 36 题上 `joint_hits` 只 1～2 题／36（两枚预算），而严格命中 +6/+7 ⇒ "买命中"在联合口径下 `not_resolved`；本债转为"已实现、结论入决策件"，剩下的动作是把联合判据写进给 owner 的账（已做）与让产品要的形状可测（待做）** |
| DEBT-G13 | **✅ 已修（2026-09-29，主线侧落地）——曾经：`Taiji.learn_bytes` 的"边沿分裂"开关在 `Seed` 门面上没有出口＝能力实现了但产品面摸不到**：`taiji/model.py:2750-2764` 提供 `include_start_boundary`／`include_end_boundary`／`reset`（本意是给"续喂相邻块、且不在两块之间人造边界"），而 `seed/model.py:124-137` 的 `Seed.learn_bytes` 只转发 `epochs`／`include_boundary`／`use_memory` ⇒ 凡走产品门面的训练方都**表达不出**"一段会话里每答完一答收一次尾"这个形状，只能整篇一次喂。受影响的正是 PLAN-A-30 §3 丁-3 那条配方（§2u 已量到默认语料 123,090 行里只有 39 行含文档内接缝 ⇒ 结束目标每篇一个、且永远落在 episode 最后一格），也影响 `SeedRuntime.chat()` 的自学习回写。取证方式＝**按签名机检**（`probe_taiji_a30_ding3_stop_target_pilot.py` 的 `facade_gap.edge_split_reachable_from_facade`），不是我读码的口供 | 中（不是崩溃缺陷，是"已实现的架构能力没有交付面"；丁-3 若排机时，第一笔改动就是这一行转发） | 修法＝`Seed.learn_bytes` 加这三个关键字参数并原样转发（默认值与 `Taiji` 一致 ⇒ 现有全部调用点逐位不变），配一条"默认面不变"的守卫——**现状：`seed/model.py:124-140` 已转发，丁-3 的落点可从产品门面表达**；机检口径留在仪器里（`probe_taiji_a30_ding3_stop_target_pilot.py` 的 `facade_gap.edge_split_reachable_from_facade`，本会话冒烟时为 false，现已 true）。**这一行留下的教训比它修掉的缺陷更值钱**：判"某形状产品能不能表达"要**按签名机检门面**，"我在外层调用过同名方法"不是证据 |
| DEBT-G14 | **长跑训练没有一条可核对的「退出原因」落盘 ⇒ 下游判读会拿错样本**：`train_seed_corpus.py` 的预算退出写的是 `ticks >= base_ticks + max_symbols`（`:228` 取载入时的 tick 读数、`:295` 比较），而 `_summary` 只回给调用方、`run.log` 是 **0 字节**，`progress.jsonl` 每 10,000 ticks 一行且**没有 final/中止标记**（预算支与周期支都调同一个 `_persist()`）。实测后果：`output/a31_ding3_boundary/` 首条 `ticks=2010000`、末条 `2750000`，同时停在 22:29:05 后无新行、面上无训练进程 ⇒ 按上式本应在 `2000000+2000000=4000000` 退出，**实际只喂到 +750,000 ticks＝预注册预算的 37.5%**，而 PLAN-A-30 rev52 与记忆当时都记成「在飞／约 2.4 小时」——那三条 §2ab 判据若照此读，量的就不是预注册的那枚件 | 中（它不改变模型，但决定**任何**训练侧判据的口径是否成立；A 支线每一次排机时都会再过一遍） | 修法三件，都不动训练语义：①退出时把「reason／base_ticks／budget／实际 ticks／是否达预算」写进 `progress.jsonl` 末行**并**落一个 `exit.json`（现在这两条信息只能靠我反向推）；②`run.log` 要么接住 stdout 要么删掉这个空文件（0 字节比没有更误导）；③任何引用检查点做判据的读数件，必须自带 `checkpoint_sha256_before`——本会话起的 `probe_taiji_a30_ding3_transfer.py` 与 `probe_taiji_a30_ding3_trajectory_threshold.py` 已按这条做（§2ai/§2ak 两档读数里 a31 的 sha 钉在 `79b1a99cedf3e65b…`）。**归因**：这是仪器/记账缺口，不是训练缺陷；续训本身＝重动作，归 owner 排期，我不自行发射。**2026-09-30 把这条的边界量准了（一好一坏两半，别混着说）**：按 zip 条目取 `data.pkl` 逐字节搜键——**喂入形状确实写进档**：a31 的 `data.pkl` 尾部含键 `end_boundary_after_newline`（偏移 442003／全长 442038），而 a26_p1（配方之前训的）**整檔无此键** ⇒ 主线那句"形状随档写进 metadata"**实测成立**，将来任何一档都能据此判断"这枚件是不是按配方训的"；**缺的是另一半**：档里没有 `max_symbols`／起始 ticks／退出原因这一类预算达成信息（我搜到的 `provenance_encoder`、`recovery_strategy_memory_budget`、`maturity_ticks` 都是**配置项**不是训练记账），所以"这枚件吃满预算没有"仍然只能靠 `progress.jsonl` 反向推——本债要补的就是这一小块。**这一半已有测试锁着，别当成裸约定**：形状那半由主线写的 `tests/taiji_native/test_a30_ding3_boundary_recipe.py::test_envelope_metadata_records_the_feed_shape` 断言 `metadata["end_boundary_after_newline"] is True` 且 `metadata["trainer"] == "train_seed_corpus"`（另有 recipe-on/recipe-off 逐位与默认值三支同册）⇒ 本债**只剩"预算达成/退出原因"这一小块没有守卫**。**2026-09-30 追加一条可推代理判据（读已有产物得到，未改 trainer）**：收尾记录的形状是可辨的——`output/a26_p1/progress.jsonl` 共 81 条，**只有最后一条**是 `window_ticks=0／online_accuracy=0.0／mean_surprise=0.0` 而 `holdout_surprise` 沿用上一条、`ticks` 停在整 18,000,000（全流 ticks 单调不减）；对照 `output/a31_ding3_boundary/progress.jsonl`（75 条）**末条仍是正常记录**（`window_ticks=10000／acc=0.3271`）⇒ **没有收尾 flush ⇒ 该 run 是提前结束、未跑完循环**，这正是"一枚 37.5% 预算的件被当成在跑"那次误报的产物学成因。⇒ 可推代理：**末条 `window_ticks==0` ⇒ 收尾完成；否则视为提前结束**。⚠ 三条边界：①**意图预算仍无处可查**（`run.log` 实测 0 行、档内无 `max_symbols`），"18,000,000 就是目标"这句只是整数巧合，不许当证据；②代理未经守卫，trainer 若改收尾写法它就静默失效；③正式修法仍是 trainer 落 `exit_reason`＋`intended_max_symbols`（两行），但 `scripts/training/train_seed_corpus.py` 今天被他线改过（`d4b470fa`）⇒ **我不在他人在飞时动它**，等其收口后随一次带守卫的小改一起做 **2026-10-03 已修①②（带守卫）**：`run_training` 的三个退出点现在都把 `exit_reason`／`base_ticks`／`budget_max_symbols`（＝**意图预算**，正是此前"无处可查"那一格）／`ticks_at_exit`／`reached_budget` 写进进度日志**收尾那一行**，并另落 `<progress 词干>_exit.json`（同一份字典写两处 ⇒ 两者不可能互相矛盾；路径规则住在 `exit_record_path()` 一处、按臂分文件），`main` 收尾把这条记账**打到 stdout**（0 字节 `run.log` 不再是唯一出口）。守卫 `tests/taiji_native/test_g14_trainer_exit_accounting.py` **6 passed**：两支预算支各自点名、耗尽支必须写 `corpus_exhausted` 且 `budget_max_symbols=None`、周期性行必须仍是那七个旧键（加性）、`_summary()` 必须只含浮点——最后一条钉的是 `train_p3b_aligned.py:240` 那条 `{key: float(value)}` 消费契约（读码时发现的，把字符串塞进返回值会当场炸别线在飞的工具）。连同既有两册训练器测试 **13 passed**、ruff 0 条。**能为假的方式不是临时改源码**：两条分支互斥且都被跑到 ⇒ 任一退出点填错原因就有一支红。**仍存的边界**：历史件（含 a31 那枚 37.5% 预算件）**没有**这条记录，末条 `window_ticks==0` 的反向代理对旧档仍有效、且未经守卫；新档不必再用它。 |
| DEBT-G15 | **产品侧的工件枚举会把隐藏临时档当检查点暴露**（`pathlib.glob` 不过滤点前缀，与 `glob.glob` 的 POSIX 规则不同）：`checkpoints/` 现存 9 个 `*.pt`，其中两枚是 **2026-09-13 20:24** 留下的隐藏残件 `.p2-12-conflict.pt`（43,309,755 B）与 `.p2-12-natural-language-write.pt`（43,316,555 B）；实测 `Path("checkpoints").glob("*.pt")` **把它们一起返回**（9 项里点前缀 2 项）。同一模式在**产品入口**上有两处——`api/training/checkpoints.py:59`（`list_checkpoints`）与 `api/routes_artifacts.py:32`（`list_artifacts`）；仪器侧另有 `scripts/training/eval_taiji_cap0_inventory.py:105`、`inventory_taiji_checkpoints.py:175` | 中偏高（不是崩溃，是暴露面：①装机后"选检查点／看工件"的列表里会出现两枚 43MB 隐藏档，用户可选中→载入到别人测试的状态或载入失败；②它正是上表 `test_cap0_inventory_contract` 那条红的**一半根因**） | 改法＝一行同族过滤（枚举处统一跳过 `p.name.startswith(".")`），**四处一起改**：只改仪器会让②消失而①留在产品里。配套两条＝守卫断言"枚举结果不含点前缀名"，以及按 `product-default-change-pin-all-call-sites` 那条钉全量调用点与录制/回放夹具（`list_artifacts` 走 HTTP 面，可能有金样）。**两枚残件我不删**——不是我造的，删了会毁别人的取证面，且它们已被 git 忽略、不入库；清理归造它们的那条线或 owner 一句授权。本债**只登记不动手**（改产品入口要 owner 认这条能力面）。**2026-10-01 自我更正一次**：我先把补丁写进了两处产品入口（`api/training/checkpoints.py` 与 `api/routes_artifacts.py` 各加 `not p.name.startswith(".")`）并配了 3 条守卫（含一条"未过滤确实会匹上"的前提守卫，本地 3 passed），**随后发现这违反本行"只登记不动手"的自定约束，已 `git checkout` 全量收回、工作树不含该改动**；补丁与守卫测试留在仓外 `C:/Users/23747/AppData/Local/Temp/a30/DEBT-G15_fix.patch`（32 行）与 `DEBT-G15_guard_test.py`，**等 owner 一句即落**（含 `list_artifacts` 的 HTTP 金样涟漪面需一并跑）。另：本行"成因"部分已被实测加固——`pathlib.glob("*.pt")` **确实**匹配点前缀（同目录下 `glob.glob` 不匹配），。**2026-10-02 已落地**（owner 弹窗授权『打，并补守卫提交』＋『重基那枚盘点面板』）：四处枚举一起加同族过滤（`api/training/checkpoints.py`、`api/routes_artifacts.py`、`scripts/training/eval_taiji_cap0_inventory.py`、`scripts/training/inventory_taiji_checkpoints.py`——最后一处只过滤**目录枚举**，显式点名的路径仍按原样接受）；守卫 `tests/seed/test_a30_checkpoint_enumeration_hides_temp_files.py` 三条（含『未过滤确实会匹上』那条前提条），**并当场验过能为 false**：把补丁 `git apply -R` 撤掉 ⇒ 两条产品入口断言红、前提条绿；恢复后 3 passed。涟漪面实测：`test_cap0_inventory_contract` 15 passed／1 xfailed、`test_cap0_eval_set_contract`＋`test_cap0_f_dimension_contract`＋`test_cap0_baseline_contract` 92 passed、`test_api_routes` 与 `test_openapi_snapshot` 绿。盘点面板按同一纪律**第四次重基**：`RESAMPLE`→`reports/taiji_cap0_inventory_a31self_g15_20261002.json`（8 行、零枚点前缀），10-01 那枚转 `RESAMPLE_BEFORE_G15_FILTER` 且其『曾经把两枚隐藏件列为可用基座』由新断言`test_the_g15_enumeration_filter_drops_hidden_checkpoints` 原位钉住；评价集 `F04.reference` **只换引用**、门文本与 `must_show` 一字未动，`history` 追加一条并写明『不是换底』。**两枚 43MB 残件仍未删**（不是我造的，且已被 git 忽略；清理归造它们的那条线或 owner 单独授权） |
| DEBT-G16 | **预注册判据没点名仪器与方向 ⇒ 第 3 条"不回退"现在无法诚实判**（PLAN-A-30 §2ab 那条线自己带的缺陷，不是新跑的读数问题）：判据原文写"F0 无掩码**切尾感知**整句可解码 ≥ 0.5（a26_p1 同底实测 1.000 ⇒ 实质是不回退）"，而这六个字**混指两台机器**——①`probe_taiji_f0_language_floor.py`（pass_line `decodable_whole ≥ 0.5`，a26_p1 的 verdict 实测 **`floor_fail`**，件 `reports/taiji_f0_a26_p1_20260928.json` 尚未入库）；②`score_taiji_r2_copy_surface_extension.py` 的"切尾感知"口径，且它**在两份文档里方向相反**：PLAN-A-25:113"一条非法字节都没有（切尾感知 1.0000）"＝越高越好，M5_R2_A_BRANCH_PLAN_REV2:630 表头"切尾感知（真非法率，**越低越好**）"却把 1.0000 放在基座列 | 高（它决定 §2ab 三条判据里第 3 条能不能算判过；按①判则参照件本身就没过线，"不回退"失去对照；按②判则方向没钉死，同一批字节可以两种说法都成立——这正是"判据不可执行"的形状） | 修法两小步，都不需要新机器时间：①在 §2ab 里把第 3 条改写成**一台仪器＋一个方向**（例如"以 `score_taiji_r2_copy_surface_extension.py` 的 `decodable_*` 字段为准，越高越好，参照值取同尺重跑的 a26_p1"），并注明"改名不改阈值"以免被当作事后放宽；②把 ①/② 两个指标**分名**（`floor_decodable_whole` vs `trim_aware_illegal_rate`），M5 那张表的表头按实际方向重写。**属主线判据文字与 A-25/M5 台账，我不单方面改**；本会话只补做同尺两档读数供改写时引用（`reports/taiji_f0_a31retrain_surface_20260930.json`、`reports/taiji_f0_a26p1_surface_recheck_20260930.json`）。**2026-09-30 追加**：§8① 那条晋升线已单独落成冻结件 `plans/reference/SPEC-A-24_a30_recipe_default_slot_promotion_prereg_20260930.md`（一台仪器一个方向一个装配，并明令 `utf8_decodable_trimmed_rate` 不算证据）⇒ **本债剩余范围缩到"§2ab 原文那条没点名的口径怎么改写＋两个指标分名"**，那两件事仍属主线台账文字，我不单方面改 |
| DEBT-G17 | **产品表层门槛里"UTF-8 合法"这一道在 `str` 上恒真 ⇒ 被预算切在半字上的答复照样过门**：`seed/surface_gate.py:52-55` 写的是 `text.encode("utf-8").decode("utf-8")` 配 `except UnicodeDecodeError`，而 `text` 是 **`str`**——`str→bytes→str` 永远成功（lone surrogate 抛的是 `UnicodeEncodeError`，那个 `except` 不接）⇒ 这条**不可能返回 False**。产品两处调它：`api/seed_runtime.py:444`（门槛②：坏答复不进下一轮 prompt 历史）与 `:506`（门槛①：只回写过门的答复），生效条件都是 `gate_model is not None`＝**挂回路且门槛武装**，也就是"默认载入件带回路自动挂载"那一面。实测件 `reports/taiji_a30_surface_gate_fffd_bypass_20260930.json`（仪器 `scripts/training/probe_taiji_a30_surface_gate_fffd_bypass.py`，四条守卫全真、`decide()` 四支都在进程里走过）：已入库两张表层件各 104 行里，a26_p1 有 **88 行 `well_formed=true` 且预览含 U+FFFD**（`well_formed_true` 共 93），a31 有 9 中 9；用产品随包工件 `checkpoints/seed_surface_ngram.lzma` 对预览重算得到 88 与 9＝**与件里存的位逐位相同** | 中（**不改相对高低**：两臂同尺同判据，§8① 那两个成句数仍可比；但它使"L1 已挡住带断字答复"这个读法为假，且门槛①/② 少了一条声称在执行的检查——`SPEC-A-24` §2 第 2 条"trimmed rate 无判别力"现在有了代码级理由）。**代价已算清（PLAN-A-30 §2at，不需机器时间）**：补上这道检查后四个臂的成句计数上限是 **12／2／15／9**（对现值 241／40／100／24），方向不变但量级塌十倍 ⇒ SPEC-A-24 的 L1 参考值必须重录；另同档取出**过门答复的原文样例**（`well_formed` 只保证"像中文的形状"，不保证通顺），故引用 `well_formed_rate` 时不许写"通顺" | 修法是一行（`if "\ufffd" in text: return False`，或改按 raw 字节判），**但那属改判据语义** ⇒ 一按下去所有已入库 `well_formed_rate` 都不可比、SPEC-A-24 的 L1 参考值要重录 ⇒ **不自行改，等 owner 裁"要不要让这道门真的跑起来"**；若批，先升件 `format` 再重录两枚参考件 |
| DEBT-G18 | **§2at 的"上限反推式"对那两枚旧件不成立；且 L1 缺一份覆盖当前默认件的参考值**（**2026-10-02 勘误重述**：本行最初写的是"实测 39 高于自称上限 12 ⇒ 两套数必有一套不是同一谓词"，那是**我把 a26_p1 的上限套到 a31_chunked_self 的实测上**造成的假矛盾——逐件重算 `cap=utf8_decodable_rate×260` 后，a31_self 39≤43、6≤20，chunked_short 14≤121，onpolicy 60≤73、24≤39，**全部满足**；真正不成立的是 §2at 引用的那两对：a26_p1 实测 241 而公式给 12、a31@37.5% 实测 40 而公式给 2 ⇒ 那两个字段取自不同口径，一个 `well_formed` 为真的文本理应可解码，故上限必须是 `well_formed` 的上界） | 中（原评"高"是建立在假矛盾上的，已降）：缺陷真实但性质不同——**不是**两套数互相打脸，而是其中一套的算法对旧件无效；后果是 SPEC-A-24 §5 的 L1 参考值（241/100，G17 前判据）与 §7b 重录值（12/15，绑两枚旧件）**都不覆盖当前默认件**，所以 L1 现在**无法**作为晋升依据——这条判断不变，理由换掉 | 修法两步（不主张"修公式去迁就 241"）：①对**每一枚要引用的件**现取 `utf8_decodable_rate`、按件算上限并登记（默认件与 (c) 各一份），并把 §2at 那两个旧数字就地标注为"口径不明、不可引用"；②配一条守卫断言**同一件同一装配**下 `well_formed_texts ≤ utf8_decodable_rate × texts`（它现在为真是弱检验，但对旧件那两对必须为 false 才说明公式与口径被分开登记了）。**改判据语义要 owner 认；先登记不动手**。**2026-10-02 修法②已落地（它不改判据语义，只是一条读已入库件的守卫，所以不需要授权）**：`tests/taiji_native/test_g18_well_formed_decodability_bound.py` 两条——G17 后每件逐臂`well_formed ≤ dec_rate×texts`（现核到 8 条臂以上）、G17 前的件必须违反它（≥3 条，否则正向那条就是恒真式），并另在进程内用一枚合成违规件验过"能为 false"；扫描范围锁在 `*surface*.json` 一族（`reports/_m7_check.json` 连 UTF-8 都不是，扩面会假失败）。剩下①的一半（把 §2at 那两个旧数在原节里标"口径不明、不可引用"）已在 PLAN-A-30 §2at 就地标注完成 |
| DEBT-G19 | **复制回路的证据发射没有时序/强度门控（每步全量、无系数地加进 logits）**——本行标题原写"发射没有相关性下限"，该方向连同上限／冻结恒定／整根置换／oracle 寻址／候选集／库恒空共**七族已于 2026-10-02 全部否证**（判读线见本行第三格与 PLAN-A-30 第四十一/四十三次停靠），留旧标题会把下一格引回已否证的方向：`ToldContentStore.best_match`（`taiji/copy_circuit.py:129-134`）只要 store 非空就返回 argmax 余弦那一条事件，而 `taiji/model.py:2217-2228` 把它的 `gate * distribution` **无任何强度系数**地加进每一格预测的 logits（另两条通道各有 `memory_read_gain`／`consolidation_read_gain`）。实测（档 (c)＋seed-A，12 篇审计、`evidence_calls=9,759`）：证据非零率 **100%**（`empty_evidence_share=0.0`）、注入 logit **中位 +14.7／p90 +36.1／max +124.2**、L1 中位 33.4、非零维数中位 9；装载信封里残留的 **2 条陈旧告知**（15B／20B，容量 4）把注入非零率顶到 100%——但**它们不是复述与拖写的来源**（2026-10-02 检索侧档实测：现默认底上把陈旧事件全部请出候选集，真自停仍 13/72、D 仍 7/16，两列 Δ 都是 0 ⇒ 供给与伤害都来自本题对话内的事件）；且 `gate_bias` 顶在参数钳位 ±2.5 的上限（`copy_circuit.py:627` 的 `clamp_`）——学出来的解想要更开的闸 | 高（这就是"模型自己会停"与"逐字复述"互斥的机制解释：接缝上 `p_boundary` 中位 0.258243→**3.1e-05**（约 8,300×）而名次中位只 1→3 ⇒ **不是被谁赢走，是被稀释**；也是 L2 拖写者 0→6 与表层成句 17→6 的同一条来源） | 三条候选修法（都要 owner 认这条能力面，且各自必须配"能被走到"的守卫）：①给 `best_match` 加**相关性下限**（cosine 低于阈 ⇒ 返回 `None` ⇒ 与"未开闸时为精确零向量"同口径）**——此方向已否证**（τ=0.3 把复述命中 6→**0** 而接缝 0/300 一动不动；有害步实测 0.4332、有用步落在 0.1~0.3 ⇒ **全局下限是反的**）；②**剂量**：新增 `copy_evidence_read_gain`（默认 1.00 ⇒ 逐位不变，与 `memory_read_gain` 同族）——剂量档已回（α=0.25：判据面 +27、能力面 +3 临界，见 PLAN-A-30 §剂量）；③**形状**：SPEC-A-22 那条 query-conditioned selector（只在被提问引用时进入读出）；④**上限／自信度衰减**（2026-10-02 新增，实测最强的那一形）：相似度**高于** c 的步不发——`c=0.3` 把接缝判决面从 0/300 **逐位放回** 186/300（`p_boundary` 中位 3.1e-05→0.258243，与"完全不挂回路"两两相同），同 c 的复述命中 6→5（差在分辨率以下 ⇒ 读"没丢"）⇒ 本支线第一枚"知道在哪儿该停"与"逐字复述"同时成立的配置（`reports/taiji_a30_copy_evidence_ceiling_c_300doc_c030_20261002.json` ＋ `reports/taiji_a30_cap_dual_arm_chunked_short_ceiling030_20261002.json`，**必须成对引用**）。**但它的作用域只到判决面**：自身轨迹那一列按环内归因只静音 88/14,276（0.62%）、真自停 23→25（Δ2 < 分辨率）⇒ 判"两列同轴"的下一步尚未成立。补一档 **c=0.2**（夹住那条 1% 界）：环内静音 **1,171/14,194＝8.25%**、真自停仍 **25/72**（Δ2 < 分辨率）、同 c 的 D 命中 4/16（对 6/16，未触发 ≤1/16 的否决线）⇒ **判决面与自身轨迹面不是同一根相似度轴在管**⇒ 2026-10-02 再进一步：**轨迹面听的是「发不发／发多重」，不是「发的是哪一条」**——把证据向量 257 维整根置换（硬度逐位守恒、守卫 true）后真自停 **23→23 一动没动**；而「通道恒零」那一格给出 66/72＝与不挂回路逐项同值 ⇒ 挂载动作本身（prompt／store／记录门）在这条面零作用，全部作用都走那条加性证据；于是本行剩下的活形状只有②强度系数（owner 明确暂不立项）。`reports/taiji_a30_stop_failure_c_contentperm_v13_20261002.json` ＋ `reports/taiji_a30_stop_failure_c_contentfrozen_v13_20261002.json`（后者作设计档作废、作「通道恒零」对照可用）成对引用。⇒ 同日再进一层（v14 冻结档）：把证据**冻结成一条固定的非零向量**后真自停 **23→55/72**，而把同一根向量**整根置换**（硬度逐位守恒）后 **23→23** ⇒ 差 32 落在**「证据跟着 cue 走」这件事本身**上，于是形状候选从三条扩成四条：**加性证据的时间恒定性**。`reports/taiji_a30_stop_failure_c_contentfrozen_v14_20261002.json` ＋ `reports/taiji_a30_stop_failure_c_contentperm_v13_20261002.json` 成对引用；⇒ **同日第二十九次停靠把这一族问到底**：复述面用**同一副**内容档跑两半——整根置换 **D 6→0/16**（硬度逐位守恒，`max_rel_l1_diff=2.4e-07`）、冻结成固定非零向量 **D 6→0/16**；并上轨迹面（置换 23→23、冻结 23→55）得出一句结构结论：**「跟着 cue 换内容」这一个行为同时供给复述与伤害自停 ⇒ 两枚指标不可分头优化**。由此否证三族修法（下限／按余弦切发射／冻结恒定），**只剩 SPEC-A-22 那一族成形**（改「谁有资格被 cue 取到」而不是改发射强度）。⇒ **同日第五种形状出数且是第一个两全的：生命周期门控（时间窗）**。K=64（由 §第三十次停靠 定价档按先写死的规则得出，非挑出）把 (c)＋回路的真自停 **23→63/72**、吃满预算 49→9，同 K 下同底同面的 D 命中 **7/16**（全剂量 6）⇒ 两半同刻度成对满足冻好的第一支；机制与上一句同向：早段的 cue-following 是供给、晚段是伤害。**仍是仪器侧读数，产品源码零改动**；⇒ 同日的**机制定位**（比所有修法更上游）：四行同底同题面读数——全剂量 D 6/16｜**库恒空（回路仍挂着）0/16，与不挂回路逐列同值**｜oracle 每次选对 6/16（命中题一字不换）｜『正确事件在不在库』与命中 2×2=3/5/3/5 独立 ⇒ **注入的『存在』是复述的必要条件、注入的『内容对不对』不决定复述**；同一条通道的发射本身又是拖写来源。⇒ 本行所有修法（下限/剂量/上限/恒定/时间窗/候选集/寻址）**都在错误的那一层**：要改的是读出侧『注入内容如何进入输出分布』，不是发射强度或寻址准确率（`reports/taiji_a30_cap_dual_arm_chunked_short_emptystore_20261002.json`）。仍**不立项**（owner 2026-10-02 裁）。（`reports/taiji_a30_stop_failure_c_ceiling020_v12_20261002.json` ＋ `reports/taiji_a30_cap_dual_arm_chunked_short_ceiling020_20261002.json`，同 c 成对引用；B 半跑在 36 题面上，只有 D 列与 24 题面那两件可比）。⇒ **本行的修法表今天收敛为一族**：下限、上限、冻结恒定、寻址/oracle 四条**对停止问题已三重否证**（oracle 选对 23→23；相关性门拦 97.5% 而命中不动；上限静音 8.25% 仍 23→25），只剩②发射的**时序/强度门控**有效，且**跨两枚回路复现**：seed-A 23→63/72（Δ+40）、seed-B 29→64/72（Δ+35），上界是不挂回路的 66/72（`reports/taiji_a30_stop_failure_c_seedB_anchor_20261002.json` ＋ `reports/taiji_a30_stop_failure_c_seedB_window64_20261002.json`，同 `checkpoint_sha256 ae51700e881f757a`、不同 `circuit_sha256`，成对引用）。**仍只登记不动手**；都不动 §1 三线与 SPEC-A-24 §8 那条 ≥6 线 |
| DEBT-G20 | **（已测：否证，非缺陷）复制回路不会把「本轮没被告知的整条内容」说进答案**：合成探针句『我今天早上喝了豆奶。』（标签在 40 条题面里 0 次、无题以它为自身标签）整根替换被取到的事件内容，检索照做（swaps 23,825／calls 30,968＝77%）、寻址面不动，外来标签在 24 题答复里出现 **0 次**；**基线件**（只测不装）同为 0 次 ⇒ 不是漏检。同件本题 D 命中仍 6/16（同一批题） | 已闭（三探针一致否证：豆奶／仓鼠／B7392 各一次，基线与 decoy 均 0/24，D 恒 6/16；原报「中高·源监控缺陷」先因检测器假阳性撤回，再带基线重做否证）。`reports/taiji_a30_cap_dual_arm_chunked_short_decoybaseline_20261002.json` 三件成对引用 | 留下的真实结论：注入的**存在**是复述必要条件（库恒空 ⇒ D 0/16），注入**是哪条**既不必要也不充分 ⇒ 这条通道是「有内容可用即可」的通用增益，不是按条取回；A2 线此前把复述记在回路账上是**高估**。**残余边界**：只测了一个合成探针、单底 24 题；换第二第三个探针才算穷尽 |
| DEBT-G21 | **（已修·2026-10-02 20:55）cap 仪器的报告信封曾缺 `checkpoint_sha256`**：`score_taiji_r2_copy_circuit_chat_cap.py` 的 `taiji-r2-copy-circuit-chat-cap-v1` 顶层只有 `checkpoint` 路径与 `circuit_sha256`，没有底座的 sha；而 L2 探针（`probe_taiji_a30_stop_failure.py`，v13 起）已经自述 `checkpoint_sha256`。⇒ 同一族跨工件配对时，停止面能按 sha 钉底、复述面只能按**路径**钉底（路径相同不等于内容相同——底座被续训写过就查不出来） | 中（今天 §第五十一次停靠 的发表前置①因此只能降级：路径等值可机检、sha 那半记 unverified；不影响本轮读数，因为 `a31_chunked_short/checkpoint.pt` 在整个 2026-10-02 没有被写过（只读纪律见 roadmap 第四节），但这是**假设**而不是**实测**） | 一行修法：cap 与表层两台仪器各补 `checkpoint_sha256` 自述（与 L2 同一取法、同一字段名），补完把 §51 前置① 从 unverified 升回机检；零机器时间，可与任何在飞取数并行。**已做，验证分两档如实报**：cap 走了一次 3 题冒烟，件里 `checkpoint_sha256="ae51700e881f757a"`，与独立重算相同、也与 §51 开档时写死的前缀相同（`output/tmp_a30_smoke/g21_cap_smoke3.json`，未入库）；表层那台**没有重跑 104 题**，按同源证据证明——新键与已出现在落盘件里的 `base_sha256_unchanged` 属同一个 dict 字面量（源码第 366 行），`sha_before` 在第 331 行赋值 ⇒ 静态可判它必被写出。两台契约测试 14 passed／rc=0。**§51 那两件的 unverified 半条不追溯**（件早于修复生成），从下一档起前置① 可全条机检 |
| DEBT-G22 | **（已修·2026-10-02 21:58，v21）产品侧生命周期门的计步基曾含 prompt 段**：`taiji/model.py` 的门在证据注入点数`_copy_evidence_step`，而这条链的 prompt feeding 同样经过该点 ⇒ K=64 里有一部分被 prompt 步吃掉，答复相反而整段静音（实测 2026-10-02 第五十二次停靠：开档 `steps_seen=340 ≈ 84 prompt＋256 答复`，`emitted=64／silenced=276`，读数 66/72 与"不挂回路"逐列同值） | 中高（它让"仪器侧已证的增益"在产品路径上**读不出来**；更坏的情形是被误当成果——66/72 恰好等于上界，不看"被走到"计数就会当成门把能力买满了） | 门控只数**答复相的步**（与 v17 替身档同一口径：`loop_steps` 只在生成环内自增）。配可机检守卫：产品档件的 `steps_seen` 必须等于该件 `per_item` 的答复步之和，不含 prompt 段；不等即红。修完须重跑开/关双趟才谈"能力进产品"。默认位仍为 `None` ⇒ 未开启时逐位不变（甲已实测 rc=0） | **已修与实测**：复位点挪到 prompt 喂完处（静态守卫 `feed < first_reset < answer_loop` 钉次序）；v21 重跑开/关双趟＝关 23/72、开 **60/72**（替身 63/72，Δ=−3 在 \|Δ\|≤3 内），末趟自证 `steps_seen=130=emitted 64+silenced 66`。件：`reports/taiji_a30_stop_failure_c_v21_product_off_r3_20261002.json`、`reports/taiji_a30_stop_failure_c_v21_product_on64_r3_20261002.json` 成对引用。**默认位仍 None**；表层／复述两列未在产品档下重测 ⇒ 本行状态从"缺陷"转为"已修、待其余两列"。
| DEBT-G23 | **表层仪器的信封不自述产品档旗标**：`score_taiji_r2_copy_surface_extension.py` 的件里没有 `product_window_steps`（cap 有），而它的 `window_arm` 计数在替身档未开时恒为 0 ⇒ 一件表层读数**来自哪条路径只能靠文件名与命令**（实测 2026-10-02 第五十五次停靠那两件） | 中（正是今夜反复在修的那类：披露缺位让归档无法机检"这是产品门还是替身档"；两份件的 `chain/items/ckpt_sha` 都相同，光看件分不开） | 一行修法：把 `--product-window-steps` 写进表层信封，并把 cap 那台的 `product_window_stats`（emitted／silenced／steps_seen）同样带一份；补完重跑一次表层开档做自证 | **状态更正（不吹完）**：旗标与计数已进信封，但**取计数的时机**先是错的（放在生成之前 ⇒ 全零快照），改到返回时取之后 cap 自证通过（臂内 `emitted/silenced` 非零、D 仍 6）；表层那台同样的修法已改，但**尚未重跑自证**（一件 12 分钟），全零的自证件已删除未入库。守卫 `test_the_window_counters_are_read_after_the_run_not_before_it` 防退回。 **状态再更正（我上一条把话说重了）**：cap 自证件臂内 `product_window_stats` 实测 **null**，"emitted/silenced 非零"那句已收回 ⇒ 旗标进信封＝成立、排序守卫＝成立、**件级自证＝两台都还欠**（cap 的 stats 插错了 dict，表层未重跑）。 **最终按路径核实**：cap 自证件的计数其实**非零**（`emitted_steps=64／silenced_steps=58／steps_seen=122`（我先前写的 251/315 是凭印象，已收回）），先前"null"是我查错路径；键已从 `content_arm` 提为 `window_arm` 的 sibling。表层重跑件 `reports/taiji_a30_surface_tradeoff_v2_product_on64_selfprove2_20261002.json` 臂内自述旗标与非零计数（`emitted_steps=64／silenced_steps=20／steps_seen=84`），成句 16／strict 35 与前件同值 ⇒ 两台都自证完，本行关闭。
| DEBT-G24 | **（已修·2026-10-03 00:19，v25）L2 件里 `boundary_rank_in_legal` 与 `legal_candidates` 不是同一集合**：实测出现 rank=**65** 而 legal=**64**（名次比分母大 1 ⇒ 边界符被同时计入"候选数"与"名次"两套口径，或 legal 集合算的是可解码符号数） | 中（它让"名次／候选数"这类比值不可解释；§64 因此只按 `p_boundary` 判，名次列仅作描述） | 修法：统一集合，或另报 `rank_among_legal_excluding_boundary`；配一条**能为假**的守卫（断言 rank <= legal_candidates，且缺该断言时红）——先跑旧件确认它今天确实会红，再定基线 | **已修**：旧列原样保留（与 v6–v24 各件同格可比），逐步行与 `at_steps` 各补 `legal_candidates_including_boundary`；守卫 `test_boundary_rank_is_bounded_by_the_denominator_it_belongs_to` 钉住两条不等式（16 passed），冒烟 18 个点 violations=0（`output/tmp_a30_smoke/v25b_g24_smoke2.json`，scratch 未入库）。写守卫时它先红了一次——合成行缺新键（我的测试的错，非代码），补齐后绿。
| DEBT-G25 | **产品门的"被走到"计数只报最后一趟**：`Taiji.copy_evidence_window_stats()` 的三个计数器每趟复位，所以件里 `emitted/silenced/steps_seen` 描述的是**最后一次生成**（实测 seed-B 装机底 K=128：`43／0／43`，该趟只有 43 步 < K ⇒ 结构上不可能静音过） | 中高（**会把成功的档自我否证**：我据此写过的"两侧非零才算开过枪"自检在这类件上会误判为空档；同类错今夜已犯过一次——把取值时机放早得到全零快照） | 把 `emitted/silenced/steps_seen` 改为**全程累计**（每趟只复位"本趟步序"），件里同时报累计与末趟两组；配能为假的守卫（累计 silenced>0 与末趟 silenced 可区分），改完跑一次冒烟自证。**已修·已自证（2026-10-03 00:58，v26）**：`Taiji` 新增 `emitted_steps_total／silenced_steps_total／steps_seen_total` 三键（末趟三键语义原样不动，与 v21–v25 各件同格可比），`reset_copy_evidence_window()` 只清本趟步序、绝不清累计量（守卫 `test_window_stats_report_cumulative_and_last_turn_separately` 钉住）；冒烟同一件里末趟＝128/128/256 而累计＝emitted 2807／silenced 5048／steps 7855（scratch 件 `output/tmp_a30_smoke/v26_g25_smoke3.json`，未入库）⇒ 门确实开过枪，并首次得到覆盖率 **36% 发／64% 静音**——末趟口径结构性看不见这个 |
| DEBT-G26 | **L2 仪器不记录「终止决策行」⇒ §66 用它反推出一条不成立的机制结论（已收回）**：产品环体 `taiji/model.py:3238-3240` 在 `argmax()==boundary` 时**先 break 再 observe**，而逐帧记录装在 `observe` 包装里（`probe_taiji_a30_stop_failure.py:442-457`）⇒ 边界胜出那一步没有任何 record，`endstep_probe.last_step` 只是"最后一个**在案**步"。我据此写的 `peak_is_last_step` 0/144 对任何 `generation_steps≥2` 的代**结构上不可能为真**（本轮现算：四件 144 代里 `generation_steps==1` 者为 **0**） | 中高（它把"看到边界冒尖与真正停下脱钩"这条**不成立的**结论写进了台账；更要紧的是它让整个"转化失败"问题在现有件内**不可答**——唯一知道边界会赢的那一步恰好是唯一的盲区，而这条盲区本文件 `format_note_v17` 早就写明过） | 修法＝按生成补 `terminal_decision`（用该代最后一个在案 record 携带的**下一步** logits 走同一个 `replay_step`，字节＝边界符、UTF-8 状态＝答复喂完后的状态）**只加字段、不进 `item_rows`**（免得既有聚合列语义漂移）＋每代补 `last_recorded_row`＋件级 `terminal_decision_summary_v27`；配能为假的守卫（`terminal_rank_not_one_count` 必须能 ≠0、`terminals` 与分组长度不一致必须抛错）。改完按 §第七十次停靠 的已冻判据跑两趟不挂回路 | **已修并取数（2026-10-03 01:24，v27）**：两底各 72 代补出终止决策行，**118/118 次自停的终止名次全为 1**（⇒ 重放补行与产品环是同一条决策，不是假设）、`pairing_ok` 两趟 True；读数在 PLAN-A-30 §第七十一次停靠（停下必是该代新高：(c) 66/66＝1.0000、出货底 48/52＝0.9231；终止步相对前一步的中位倍率 ~1200×），§第六十六次停靠第②条的收回在 §第六十九次停靠 |
| DEBT-G27 | **预注册的一条判据里塞了两个 selector ⇒ 读平后两个口径各能说一句话（我自己写的，§第七十六次停靠 末段）**：甲写作"≥6 枚的胜出字节属于**同一类（同一个字／同一位置类）**"。v31 实测装机底 top1 **字节** 231＝5/9（<6 ⇒ 甲不成立）、top1 **位置类** `lead3`＝6/9（≥6 ⇒ 甲成立）⇒ 同一份件、同一个阈值，两个方向。乙那支的"散成 ≥5 类"同样没绑定 selector（类 distinct 2、字节 distinct 3，两支都不成立） | 中高（不是读数错，是**判据错**：它把"结论由口径决定"这件事留在了事后。若不登记，下一轮很容易顺手挑对自己有利的那个口径——那正是本线反复在修的那类错） | 修法＝规矩，不是代码：①写预注册时逐条问"这个词会不会给出两个数"，一条判据只绑定**一个**可机检 selector（要两个就把它们写成两条独立判据并各自预登记分支）；②读数时若已发生歧义，**两口径都报＋标"依口径"**，不挑有利的写；③§七十七 已按此执行（甲在类口径成立、在字节口径不成立 ⇒ 记"没有单一答案"），并把"要拆口径只能扩样"的成本如实写在那一格末段 | **已登记、并已按扩样收窄（2026-10-03 02:44，§第八十次停靠）**：96 题面两趟落地后，两个 selector 不再互相"各说一句话"而是给出**一致的层级结论**——生成级字节 top1 **0.4800**（甲线 0.50 差 0.02 ⇒ 落丙）、步级 top1 **0.4118 且第一名由 231 换成 232**（同样落丙），而类层面 `lead3` **0.6800**（步级 231+232 合计 0.7500）＝**类别集中、字节分散**；⇒ §七十七 的"依口径"降级为一句有方向的读数："**把关的是一族（再写一个字），不是某一个字节**"。**判据缺陷本身保留为规矩**（一条判据只绑一个 selector；扩样档已按 L1/L2/L3 拆开并预先声明类口径为弱证据）。新裁定仍没动、产品默认位仍 `None` |
| DEBT-G28 | **预注册时只检查了"selector 唯一"，没检查"这把尺在该面上有没有动态范围"⇒ v32 造出来才看见它按定义量不到目标量**：§八十五 把"LF 发在正文中间还是收口位"绑到 `in_run` 的占比上，而 `probe_taiji_a30_stop_failure.py:199-217` 里 `in_run` 标的是"**该字节所属汉字处在一段同字连写里**"（相邻字符相等）⇒ 孤立换行按定义永远不在重复段内。实测两面每一代的占比都是 **0.0**（门开启 23 自停＋11 拖写、门关闭 13 自停＋2 拖写，全部二元值），即该列**零动态范围**；字面上乙（`q_eat ≤ 0.20`）成立，但它的推断前提被件内读数否证 | 中高（后果不是"数据白跑"，是**差点把一把无效尺的 0.0 读成"位置不是原因"**——装机底 −19.0pp 的解释至今仍是空的，而错误的"已排除位置"会让人把机器时间支到错的方向上） | 修法＝规矩＋一条可机检前置：预注册里加"**开档前先量动态范围**"——同一批件先报该列两组中位与二元值占比，**两组同值或全部落在二元端点 ⇒ 该判据不建立**（不是等读数回来再找理由）。本轮已按 §五十一 的先例记 **NOT_COVERED**，没有事后补一支把 0.0 说成结论；有动态范围的候选尺（结构位／句末标记距离，或教师强制接缝面那条有真结束位标签的链）已登记在 §八十六 末段，**未跑** | **已登记并已机械化（2026-10-03 03:37，v33／§八十七）**：`_dynamic_range()`＋`ruler_usable` 开关已进仪器，比较型判据的资格前置从此必须含"该列 usable"；守卫两向都测过（§85 那种退化形状 ⇒ False 且 `endpoint_only=True`，而分母开关仍 True——分母够、尺子不够是可分离的两种失败；有真实散布 ⇒ True，防它退化成"永远拒绝"）。v32 那两枚件跑在规矩之前、不含新字段，**不追溯重跑刷绿**；27→28 passed 与两枚件均已入库 |
| DEBT-G29 | **一把"边际／比值"尺在自停组上按定义恒等于 1.0 ⇒ 它只有单侧动态范围，两组并列报中位会把恒等值当成证据**：`ratio_best_over_boundary`＝该步胜出者概率 ÷ 边界概率。若那一步**就是**边界符胜出的终止步，分子分母同物 ⇒ 恒 1.0。实测（2026-10-03，件 `reports/taiji_a30_stop_failure_self_v34_margins_circuitseedA_96_20261003.json`，96 题面 × 3 轮＝288 代、装机底 `ca2628077b21bc4c`、挂 seed-A、门 OFF）：自停组 **41 代全部 min_ratio＝1.0**（`_dynamic_range` 报 `distinct=1／endpoint_only=true／usable=false`），拖写组只有 **9/247 代**有可观测的 LF+1 读数（该面上 LF 共 75 次／71,937 生成步＝**0.00104 LF/步**，与 §83"挂回路压 LF 六到十倍"同向） ⇒ §89 的前置②（两趟 `eaters_with_lf_next ≥ 20`）**未过**，那一问在挂回路面上按冻线记为**不判**。 | 中（它决定下一格能不能拿这把尺下结论：不登记的后果是把"自停组中位 1.0"读成"停的时候只差一点点"这种恒等式伪证据） | 修法两条都在仪器侧、不动产品：①任何"两组中位并列"的列必须自带 `_dynamic_range`，且**恒等侧要显式标 `endpoint_only`**（v33 已实现、本轮第一次真拦住东西）；②要在**有 LF 的那一面**问 LF+1 的竞争，就得取**不挂回路**面的逐代 `per_step`（v31/v30 的旧 96 件只有"该代最好的一次"、没有逐步表 ⇒ 需一次 v34 不挂回路档，约 10 分钟、零训练）——已按这条排在 §94，判读线先于数写。**不许把 §89 的"不判"改写成"耦合没变"**：那条是乙分支，需要分母才配说。 |
| DEBT-G30 | **证据门的"有效状态"在产品里没有公开出口 ⇒ 四个地方各自从私有字段重推同一条式子**：`taiji/model.py:2011-2013` 在每步判定里内联写 `config.copy_evidence_utf8_gate if _copy_evidence_utf8_gate_override is None else bool(override)`，而产品外部的三个读者各自**再抄一遍**——`scripts/training/probe_taiji_a30_stop_failure.py:1500-1511`（件里那三列 `gate_effective`/`gate_config`/`gate_override`）、同文件 `:1610-1611`、以及 `tests/taiji_native/test_a30_copy_evidence_gate_flag.py:30-32` 与 `tests/taiji_native/test_a25_gate_on_the_load_path.py:44-45`；对照物是**窗口那一支已经有公开读数**（`copy_evidence_window_stats()`，`taiji/model.py:1136-1143`，同样把 config-or-override 推一遍） ⇒ 同一个开关族，一个有出口、一个只能扒私有字段。**为什么这条值得修而不是留着**：`默认关闭 ⇒ 逐位不变` 这条验收要能从外部证，而现在外部只能复制推导式——复制的那一份**会随产品改动过期**（本仓已经因为"两处各写一份同一判定"错过一次静默错判）。 | 中（观测面缺口，不改行为；但它是那条验收式能否机检的前提） | **修法（设计已定，代码未落，因为守卫需要 torch 而此刻机器无可用提交内存——见 PLAN-A-30 §99）**：①在 substrate 上加私有 `_copy_evidence_utf8_gate_effective()`（一条式子只住一处）＋公开 `copy_evidence_utf8_gate_state()` 返回 `{config, override, effective}` 三键（与 `copy_evidence_window_stats()` 同族命名）；②把 `:2011-2013` 的内联式改成调用①；③守卫三条：`override=None/True/False × config=True/False` 六种组合下 `state()[effective]` 与手推逐值相等、`set_copy_evidence_utf8_gate(None)` 后回到 config、**调用状态读数不改变任何状态**（前后 `snapshot()` 逐键相等）；④仪器侧那四处复制推导在**各自下次升版**时改读新出口（不为它单独升 v35，避免 gratuitous 版本漂移），并在 `test_a30_instrument_face_disclosure.py` 那条"必须三列一起自述"的守卫上加一条"取自出口而非私有字段"。**未做的原因如实记**：本轮 `import torch` 直接失败 ⇒ 无法跑③，所以不动产品码（未验假设不进索引层）。 **2026-10-03 05:19 已落地（同一条式子收进产品，读数有公开出口）**：`Taiji._copy_evidence_utf8_gate_effective()`（式子唯一住处）＋公开 `copy_evidence_utf8_gate_state()` 返回 `{config, override, effective}`（与 `copy_evidence_window_stats()` 同族命名），生成路径原内联三元式已改成调用它。守卫 `tests/taiji_native/test_a30_gate_state_readout.py` **4 passed**：六种 `override×config` 组合下公开读数与"被超越的旧推导式"逐值相等、`set(None)` 必须回到跟随 config（不是"关"）、**读状态不许改状态**（同种子同字节两条链，一条每步前读三次 `state()` 与 `window_stats()`，逐步 `prior_prediction` 必须逐位相同）、以及源码级断言"产品内不留第二份同一条式子"。回归面：既有 `test_a25_gate_on_the_load_path`＋`test_a30_copy_evidence_gate_flag`＋`test_a30_product_entry_window_steps_forwarding`＋`test_a30_instrument_face_disclosure` 共 **15 passed**；`taiji/model.py` 的 ruff 违规数**改前 0／改后 0**（基线没涨）。**清单按实底更正（我 §99 里写"四处"是按初次 grep 数的，少算两处）**：产品外**重推有效值**的共 6 处——`probe_taiji_a30_stop_failure.py:1500-1511` 与 `:1610-1611`、`audit_taiji_a30_stop_signal_presence.py:403-408`、`score_taiji_r2_copy_surface_extension.py:135-138`、`tests/taiji_native/test_a30_copy_evidence_gate_flag.py:30-32`、`tests/taiji_native/test_a25_gate_on_the_load_path.py:44-45`；另有两处只读覆写位、不推有效值（`measure_taiji_a30_repetition_penalty.py:68`、`test_a30_shipped_base_and_surface_gate.py:90`，属白盒断言，保留）。**迁移策略（也更正一次）**：这 6 处**不在本轮一起改**——仪器改动要各自升版并动钉版测试，属于 gratuitous 版本漂移；按"下次升版顺手改读出口"推进，而 `test_a30_gate_state_readout.py` 里**故意保留**那份旧推导式当独立交叉验证（两边不一致就会红，这正是它该留的理由）。台账状态：观测面缺口已消，迁移为后续顺手项。 **2026-10-03 08:01 迁移面按实底重数（`v38` 已落地 ⇒ 探针那两处是下一个，但不在这一轮动）**：重推**有效值**的推导式仍共 **6 处**——`probe_taiji_a30_stop_failure.py:1659-1660`（件里那三列 `gate_effective`/`gate_config`/`gate_override`）与 `:1783-1784`（同一件的另一处）、`audit_taiji_a30_stop_signal_presence.py:403-408`、`score_taiji_r2_copy_surface_extension.py:135-138`、`tests/taiji_native/test_a30_copy_evidence_gate_flag.py:30`、`tests/taiji_native/test_a25_gate_on_the_load_path.py:44`；另有 **4 处只读覆写位、不推有效值**（`measure_taiji_a30_repetition_penalty.py:68`、`test_a25_gate_on_the_load_path.py:121/131`、`test_a30_shipped_base_and_surface_gate.py:90`，属白盒断言，**保留**）。**次序与约束**：①探针那两处**下一版（v39）**改读 `copy_evidence_utf8_gate_state()`——不在本轮做，因为 §114 的 α=0 档此刻正在跑，而本仓已两次记过「取数在飞时改仪器源码 ⇒ 落件与版本锚点错配、事后无法区分」；②`audit`／`score`／两支测试各自下次升版顺手改，不为迁移单独升版（避免 gratuitous 版本漂移）；③`test_a30_gate_state_readout.py` 里那份旧推导式**故意保留**当交叉验证（两边不一致就会红，这正是它该留的理由）；④迁移的验收式与 v38 同款：同命令跑 1 题面冒烟，与已入库同命令的上一版件比 `--subtree per_item` 必须 `identical=true`、`rc=0`。 **2026-10-03 08:41 第二步已落（探针升 v39）**：`probe_taiji_a30_stop_failure.py` 那两处（信封三列与守卫 `evidence_gate_flag_honored`）改读同一份 `gate_state = runtime.model.substrate.copy_evidence_utf8_gate_state()` ⇒ 该文件里私有字段 `_copy_evidence_utf8_gate_override` 命中数＝**0**。逐位不变有比较器 rc：同命令 1 题面冒烟与已入库 v38 冒烟件比 `--subtree per_item` ⇒ `identical=true`、behavior 0、schema 0、rc=0，三列取值 `effective=true／config=false／override=true` 逐值同，冒烟件入库 `reports/taiji_a30_stop_failure_v39_gate_state_smoke1_envelope_20261003.json`（68,275 B）。新守卫两条都能为假（私有字段回退⇒红；`gate_state[` 计数必须恰好 4，两处各读一遍会跳到 8⇒红）；钉版范围扩到 `range(6, 40)`。**剩余 4 处**（audit:403-408／score:135-138／test_a30_copy_evidence_gate_flag:30／test_a25_gate_on_the_load_path:44）按同一规矩在各自下次升版时改读出口，不为迁移单独升版；`test_a30_gate_state_readout.py` 里那份旧推导式故意保留当交叉验证。门禁 `a30 or g14` **126 passed**、ruff 0 条；本轮零产品改动（动的是仪器读法）。 **2026-10-04 第三步（两支测试迁完；测试不设版，不属 gratuitous 漂移）**：`test_a30_copy_evidence_gate_flag.py` 与 `test_a25_gate_on_the_load_path.py` 的本地 `_effective_gate` 推导副本改读公开出口三键（audit:403-408 与 score:135-138 两台仪器仍按"各自下次升版顺手改"等待）；`test_a25_gate_on_the_load_path.py:121/131` 的白盒读覆写位断言按本登记保留。钉子扩进 `test_a30_gate_state_readout.py`：两文件必须含 `copy_evidence_utf8_gate_state()`、且不得再出现两种推导式形状（getattr 兜底取字段／`if override is None` 三元式）——钉子第一版把白盒断言也拦了、当场红过一次（负向已证），收窄后三文件 **13 passed**、ruff 0 条；负向证据＝HEAD 版本里两种旧形状各命中 1 次。 **同日深夜第四笔（score:135-138 随旗标 occasion 迁掉）**：`score_taiji_r2_copy_surface_extension.py` 接 SPEC-A-26 `--copy-evidence-injection-mode` 旗标即本仪器"下次升版"场合，按本行"各自下次升版顺手改"的既定处方把该处推导一并迁到公开出口；剩余迁移面 **4 → 1 处**（只 audit:403-408）。 |
| DEBT-G31 | **配对档的"同题面"前提此前不能从件内自证**：`probe_taiji_a30_stop_failure.py` 的件里只有 `items: <条数>` 与写死的 manifest 路径，而 §103／§104 的逐格配对表把结论建立在"两枚件读的是同一批题"上——条数相同不等于内容相同（本线已在"同一批文档并非自动成立"上栽过一次，那次补的是 `docs_sha256`/`prompts_sha256`） | 中（它不改变任何已发表结论——本轮已用 manifest 的 git 史外部核过：题集最后变更 `712dd97fa`＝09-26 19:06，三枚被配对的件都跑在 10-03；但它决定**下一轮**还能不能这样引用） | **已修（v35）**：件里加 `items_sha256`＝所选条目 `(id, turns, expected_contains)` 的摘要（**故意不含**元数据列，守卫逐值钉"改元数据不改指纹、改一题必变值"）；`pair_taiji_a30_stop_cells.py` 两侧都有该键时必须相等、不等即拒绝配对，缺任一侧则披露 `unknown_pre_v35` 而不是默认通过。同格另修两条自己的错：升版时造出一个**重复字典键**（Python 静默取最后一份，py_compile/ruff 都不报）⇒ 已删并把"AST 扫重复键"写成守卫；加返回列时旧等式断言先红 ⇒ 改成把新列与正反两支一起断言（`equal`／`不同即拒`），不放松等式。四文件 ruff 4→0、守卫 37 passed |
| DEBT-G32 | **"同字崩塌"这类量必须在字级测，字节相邻不是它的代理**：§106 的 selector 我写成"峰值那一步的字节 ＝ 上一步的字节"，而中文一个字三字节（`哥哥哥` → `E5 93 A5 E5 93 A5`）⇒ **相邻字节永远不相等**，装机装配实测占比只有 **0.0135（222 代里 3 代）**——那是多字节文本的**结构下界**，不是"环与近停态无关"。属本仓反复记过的"计数名与计数物不符"（`guard-must-be-able-to-fail` 第 1 条）在这里再犯一次 | 中高（它会把一整族结论指错方向：按字面那条落"乙＝环不是主犯"，而真相是这一格根本没测到那件事） | **已修**：§106 就地记为"不判（selector 无效）"、不重跑也不改写；v37 把仪器本来就在算的**字级** `in_run` 接到峰值上（每代 `repeat_run_at_peak_step_v37` ＋ 件级 `peak_run_summary_v37`，分母＝该群全部代数），并在件里自带一条`incoherent_zero_run_but_peak_in_run` 自检（某代 `steps_in_repeat_run==0` 却报峰值在段内 ⇒ 违例数入件、非零即整格不可判）；新守卫把这次错**编码成测试**（三字节重复词 ⇒ 字级 True 而字节相邻 False，两列必须能取相反真值）⇒ 改名回坏写法会当场红。v36 那列**保留不删**（字节相邻本身仍是合法描述），但它**不许再被当 selector 引用**——这句写进 §110 与守卫文档串。判据仍按 §110 冻的甲/乙/丙（`r≥0.50`／`r<0.20`／其余不判），档已发出、落地后只读数不改线。
| DEBT-G33 | **默认装配上"证据通道被调了几次"这个读数是恒零的假读数**：`probe_taiji_a30_stop_failure.py:1188` 只在 `--copy-evidence-alpha ≠ 1.0` 或 `--relevance-ceiling-c` 非空时才用 `_observe_silencing` 包住 `copy_circuit.evidence`，而 `loop_silenced` 在装配前就初始化成 `[0, 0]`（`:1126`）⇒ 默认档件里的 `instrument_guard.evidence_calls_in_generation_loop`（`:1709`）与 `relevance_ceiling_silenced_calls` 报的是"没装观察者"，与"通道从未被调用"**在件里无法区分**（同族的 `relevance_ceiling_fired`／`ceiling_fire_count_reported` 已经用 `is None or` 做了条件化，只有这两列没有） | 中高（§110 判为乙之后，下一问就是"近停态那一步的分从哪条通道来"，而那一问的第一只尺子正是这一列；现在它只能在带旗标的档里用） | **登记未修（修法与守卫已设计，改动需升 v38＋跑逐位守卫，故单开一格）**：①观察者与旗标解耦——只要 `copy_circuit` 在场就装一个**只计数并原样转发**的包裹器（`--copy-evidence-alpha`／`--relevance-ceiling-c` 的语义不变，仍按现在两层叠装）；②件里加 `evidence_observer_installed: bool`，未装时那两列报 `None` 而**不是 0**（与 §95 那台上界仪器的 `lower_bound_available` 同一条纪律）；③三条守卫：**(a) 读数必须能为真**——挂回路做一次 n=1 生成，`evidence_calls_in_generation_loop > 0` 才算尺子活着（本仓"恒为零的读数不是观察"那条）；(b) **包裹器不改数**——同种子同字节两条链，一条被包裹，逐步 `prior_prediction` 必须逐位相同（照 `test_a30_gate_state_readout.py` 的写法）；(c) `test_a30_instrument_face_disclosure.py` 那条"三列一起自述"扩成"连同观察者旗标一起"；④**先核后改**：已入库的 40+ 枚件里该列的零分"旗标开/关"两类，改完只影响新件，旧件按 §十九 规矩保留不重跑。 **本轮只登记**（v38 需要机时外的一条守卫跑通，而 (c) 档此刻正在占用机器；未验假设不进索引层） **2026-10-03 07:53 已落地并结清（探针升 v38）**：三条改动照本行设计落码——观察者与旗标解耦（`copy_circuit` 在场即装、只转发不改造）＋新列 `evidence_observer_installed`＋未装观察者或未开上限时那三列报 `null`；钉版范围扩到 `range(6, 39)`，旧写法 `"evidence_calls_in_generation_loop": loop_silenced[0],` 已被反向钉死（写回去就红）。**行为中性这次是比较器给的 rc 而不是口供**：同装配 v37↔v38 两档 `compare_taiji_a30_report_identity.py --subtree per_item` ⇒ `identical=true`、`behavior_diff_count=0`、rc=0，八个件级摘要块全部 equal、`total_steps=71937`／自停 41／吃满 247 逐值同；整档比的**唯一一条 behavior** 就是被修的那一列（0→71649），三条 schema 是新旗标＋两列上限改 `null` ⇒ 引用“v38 行为中性”必须把这些一起报出，别拿豁免当门柱。**“读数必须能为真”也在真档上验过**：默认装配环内调用 **71649 ＝ total_steps − generations**（预测先于数写死、逐值命中），不挂回路那枚冒烟报 `null`，开上限那枚仍报数字（静音 1／占比 0.001328／全链 52）。另补 `tests/taiji_native/test_a30_evidence_alpha_zero_semantics.py`（**5 passed**，含“α=0.99 必须让恒等断言抛错”的负对照）——α 的语义此前**没有任何测试覆盖过**，而 §114（α=0 对照档）整格建立在它上面。 |
| DEBT-G34 | **SPEC-A-17 的一条读数引用指向库里不存在的件**（本轮做"件名机检 `git ls-files`"顺带查出，属 A2 线不是 A-30）：`plans/reference/SPEC-A-17_r2_a2_3b_format_align_prereg_20260925.md:185` 引 `reports/taiji_r2_copy_circuit_chat_training_20260925.json`，但盘上只有**嵌套在另一目录里**的同名件 `reports/taiji_a2_base_pos_off/taiji_r2_copy_circuit_chat_training_20260925.json`（27,376 B），而**对照臂那枚已不在盘上**（`reports/taiji_a2_base_pos_on/` 是空目录）；`git ls-files reports` 里 `taiji_a2_base_pos` 命中 **0 条** ⇒ 两个臂的产物**一个没入库、一个不在被引路径上**，这条 S2 判读（"chat 形态严格真命中 7/16"，也写进了队首 03_CURRENT_EXECUTION.md:1335）目前**无法从库里复算** | 低（不影响任何当前裁定：A-30 的读数不依赖它；但它是"引用即证据"这条规矩的破口，且队首那句还在读） | **只登记、不动手**（改的是 A2 线的判读件与产物，且补库需要重跑那档＝别线的机器时间，我不代做）。三条备选交给 owner／A2 下一次触及时定：①把 off 臂那枚按**引用路径**补入库（`git add` 到 `reports/` 根，并把件名从嵌套目录改指），②若对照臂已不可重跑，就在 SPEC-A-17 与队首两处**就地标注"原件不在库、本读数不可从库复算"**（照 §26 那格"复制冒烟件以免引用悬空"的先例，但这里缺的是对照臂、复制不出来），③重跑两臂再入库（成本最高）。**通用教训（值得进方法层）**：文档里出现 `reports/<件>.json` 这种**扁平路径**时，机检要按**完整路径**核 `git ls-files` 与盘上位置是否一致——我这次只按**文件名**匹配，差点把"嵌套目录里有一枚同名件"读成"引用成立" |
| DEBT-G35 | **（状态 2026-10-03 11:4x：②"覆写需换名或显式旗标"已落地成机器侧防口——`train_seed_corpus.py` 拒绝把 `--checkpoint` 落在"已存在且不是本次 `--resume` 源"的件上，新守卫 `tests/taiji_native/test_g35_checkpoint_overwrite_guard.py` 五支含两支正例，见 PLAN-A-30 §128；①历史标注已随 §126 落；③`artifact_sha_drift` 可见性清单**已落地**——`scripts/training/audit_taiji_artifact_sha_drift.py` 把"路径在场而字节已换"与"路径不在盘上"重算成一份可复跑清单（当代读数 1547 份件／336 条声明：ok 220／missing_file 112／sha_drift 4，另有 163 条 sha 配不出路径单独计数），守卫 `tests/taiji_native/test_a30_artifact_sha_drift_audit.py` 六支含两条硬钉锚点，详见 PLAN-A-30 §129 与 `reports/taiji_a30_artifact_sha_drift_20261003.json`；三半至此全部结清）**一条已发表读数的源件字节已不在盘上，且引用面无标注**（2026-10-03 做模型件收束时顺带查出，属 A-30 线）：§2ai–§2an 那 10 份件（``reports/taiji_a30_ding3_transfer_12pos_20260929.json`、reports/taiji_a30_ding3_transfer_30pos_20260929.json`、reports/taiji_a30_ding3_transfer_120pos_20260930.json`、reports/taiji_a30_ding3_transfer_300pos_b_20260930.json`、reports/taiji_a30_ding3_trajectory_threshold_2pos_smoke_20260929.json`、reports/taiji_a30_ding3_trajectory_threshold_24pos_20260929.json`、reports/taiji_a30_ding3_trajectory_threshold_24pos_v2_20260930.json`、reports/taiji_a30_ding3_trajectory_threshold_24pos_v3_20260930.json`、reports/taiji_a30_ding3_trajectory_threshold_24pos_v8_20260930.json`、reports/taiji_a30_recipe_surface_tradeoff_104item_20260930.json`）把治疗臂记成 `checkpoint_sha256_before = 79b1a99cedf3…`，而①盘上无任何 `.pt` 等于该值（扫 `output/a31_*`／`a26_p1` 五棵树共 176 枚 `.pt` 无命中），②件里自述的路径 `output/a31_ding3_boundary/checkpoint.pt` 现 sha 是 `a3f63b47…` ⇒ 该路径**在两次跑档之间被后续写入覆盖**（`train_seed_corpus.py` 按 `--out` 覆写同一路径是常规行为），这条"重训臂 vs base 臂"的配对**无法从库里复算**，而它的结论（教师强制成立／自身轨迹为零／带条件可价点为空集）已被 §2an–§2aq 与更新十二反复引用 | 中（不推翻任何裁定——所有引用都带件号，件里的数仍可读；但它是"引用即证据"的第二个破口，且与 `DEBT-G34` 不同族：G34 是**路径写错**，本条是**路径正确而字节已换**，按 `git ls-files` 或链接检查都查不出来） | **登记未修（修法不改历史读数，只改"可复算性"的声明面）**：①在 PLAN-A-30 §126 已就地标注本条（引用那 10 份件时须带"源件字节已丢，不可复跑"）；②训练仪器的 `--out` 写靶必须**按 sha 后缀命名或写前把旧件改名**，让"覆盖"变成"并存"（同 `probe_taiji_a30_writeback_gate_shipping_face.py` 那条 `--out-report` 必给＋已存在即 `rc=2` 的修法，只是这里管的是权重不是报告）；③守卫侧只能做到"点名"：新增一条机检，遍历 `reports/*.json` 里 `checkpoint_sha256_before`／`base_sha256` 字段，若件里同时写了路径且路径在场而 sha 不符 ⇒ 输出 `artifact_sha_drift` 清单（**不 skip、不 raise**，因为覆盖是合法行为，但必须在报告面上可见）。改动需 owner 认"要不要为历史件补这条可见性"，本轮只登记 |
| DEBT-G36 | **一条"复现封存"守卫的参照基座被本次收束删除 ⇒ 该臂的字节锚点永久不可复现**（2026-10-03，A-30 线，本轮自造）：`scripts/training/probe_taiji_cap0_legacy_load.py:49` 把控制臂基座硬编码成 `CONTROL_CHECKPOINT = checkpoints/seed_corpus.pt`，而 `tests/taiji_native/test_cap0_legacy_load_contract.py::test_a_fresh_probe_sample_reproduces_the_sealed_one` 会**现跑 `run_probe()`**（同文件第 191 行）⇒ 件被删之后控制臂只能报"载入失败"。已把这处红改成正向断言（控制臂必须 `load_ok is False` 且 `load_error` 点名 `seed_corpus.pt`；两枚在场的 trained 臂仍逐叶＋逐字节复现），**不是扩屏蔽集、不是 skip** | 中（丢的是"当前格式的未训练控制基座经入口仍塌成一条模板"这条**当下事实**的活复现——它现在只剩 `reports/taiji_cap0_legacy_load_probe_a31self_20261001.json` 一份封存件为证；旧格式能否放宽载入那一半不受影响，因为 `TARGET_CHECKPOINT`（`checkpoints/seed_beta.pt`）仍在场且逐字节复现。根因与 `DEBT-G35` 不同：**不是被覆盖，是被判定为"没人引用"而删**——判据用的是"按整路径 grep `.py`"，而硬编码常量与活枚举面都会让这条判据失明） | **登记未修（重建便宜，但必须按"重基"纪律走）**：①重建只需 `Seed()` 新建一枚未训练基座存到 `checkpoints/seed_corpus.pt`（秒级、零机器时间），但**不许**指望它复现旧字节锚点（权重不同）⇒ 若重建，必须同批再生这一支的封存件并写明"控制基座是 2026-10-XX 新造、旧件已随收束删除"；②不重建则本条保持未修，且**任何新增的"复现封存"守卫都不许把参照钉在 gitignored 的 `checkpoints/*.pt` 上**——要么用仓内版本化夹具，要么在守卫里显式声明它只在件在场时成立（同 `test_a30_copy_evidence_gate_flag.py:45` 那种 `not (CIRCUIT.is_file() and CHECKPOINT.is_file())` 的条件式，但那支是 skip，本条要求的是**必须可见**的缺件断言）；③收束类动作的前置检查从此加一条：`grep` 目标文件名于**全部仓内 `.py`**（含 `--base` 默认值与模块常量），命中即归"不能动" |
| DEBT-G37 | **本次模型件收束多丢了一枚 seed-B 回路锚点，而 §126 当时把它说成"两枚回路都保住了"**（由新仪器 `audit_taiji_artifact_sha_drift.py` 第一次运行查出，见 PLAN-A-30 §129）：`reports/` 里有 **11 条**声明指向 `output/taiji_r2_copy_circuit_chat_seedB/judge/circuit-final.pt`，该目录确在 §126 的删除清单（`C:/Users/23747/seed-cull-20261003/final.txt` 内原文可查）；而收束时保留的 seed-B 是**另一条路径**上的 `output/taiji_r2_a23_ding2_seedB/judge/circuit-final.pt`（`3cfb02c2…`）。⇒ "换一枚回路还成不成立"这条跨回路复现在**装机底**这一支上失去锚点（`taiji_a30_stop_failure_self_seedB_v25_*` 那批读数只能引件里的数） | 中（不推翻任何裁定——a23 那枚 seed-B 仍在、`1cfe5961` seed-A 仍在，跨回路结论仍有两枚独立装配中的另一枚撑着；但 §126 那句"两枚回路"按今天的口径要改写，且引用这批 chat-seed-B 读数（含 `taiji_a30_stop_failure_self_seedB_v25_*`）时必须加"源件已删、不可复跑"） | **登记，不回填**：①§129 已把这句更正写回同一份计划文档（不许在 §126 原文上悄悄改字）；②今后凡"按 `.py` 引用面判可删"的收束，必须**再跑一遍 `audit_taiji_artifact_sha_drift.py`** 看 `missing_file` 的增量——引用面只覆盖代码，不覆盖**已入库读数的锚点**，这是本条与 `G35`/`G36` 合起来教的同一课；③要重建只能重跑 seed-B 电路训练（`build_taiji_r2_copy_circuit*.py` 一类），归 owner 排期，不在本轮 |
| DEBT-G38 | 【本条撤回：登记的是一条不存在的仪器缺陷】2026-10-03 13:28 复测作废，全文见 PLAN-A-30 §135。两句都不成立：其一「比较器不认表层件族」实为我传了它不接受的取值——帮助文字写明 --subtree 只比顶层键、示例即 control_no_circuit，用该取值一次成功并输出 identical true、行为差 0、披露豁免 0；其二「subtree_missing 以 rc=0 出场」实为 rc 被管道吞了——我当时把 tail 之后的 echo 结果当成比较器的退出码，不经管道复测是 rc=1，而源码第 157 行 return 0 if identical else 1 本就 fail-closed | 无（作废，不留待修项）；副产品是 §134 那两次独立表层取数的一致改由正式比较器背书，比 inline diff 强一档 | 不修，只留两条防复发：一、凡以退出码为证据必须不经管道取 rc，或在管道后取 PIPESTATUS 首元素（本会话第二次因管道接 tail 误读结论）；二、判「工具不支持某取值」前先读它的 argparse 说明并试文档给的取值——「我用错了」不许升格成「它不行」再进台账。原文保留在本行位置、不删件 |
| DEBT-G39 | **训练器的退出记账自述"哪枚档、吃没吃满"，却不自述"写下去的是哪些字节"**（2026-10-03，A-30 线，为 §125 训后读数造等待器时查出）：`scripts/training/train_seed_corpus.py:209/356` 在收尾时另写一份独立件 `<progress stem>_exit.json`（DEBT-G14 立的），实测样本 `reports/seed_corpus_smoke_progress_exit.json` 键为 `exit_reason / reached_budget / base_ticks / budget_max_symbols / ticks_at_exit / checkpoint_path / corpus_fingerprint` ＋进度行同源的五个计量键——**里面没有 `checkpoint_sha256`**。⇒ 下游"读训后件"只能间接绑字节：跨 90 秒两次取样相同才算认冻住，再拿探针件里的 `checkpoint_sha256` 回头对（今天正是这条路把中途件 `a98d2af8…` 与终件 `3993c323…` 分开的，也正是它查出中途那枚 `instrument_guard.base_sha256_unchanged=false` ⇒ 该读数作废、不许进判定） | 低-中（不推翻任何已裁结论；代价是每次训后取数都要多一段"为什么我信这两次取样相同"的解释，且有两个盲区：①若退出后立刻有人续训，取样窗口会被骗过；②这条理由不在场时，新读者会重犯我今天犯的错——`--keep-checkpoints on` 下 `checkpoint.pt` 每几千 tick 就"出现"一次，**产物存在性只证明开始写了，不证明跑完了**） | **登记未修（修法很小，但必须带守卫一起落）**：①在收尾那次 `atomic_save` **之后**把落盘目标的 sha256 记进同一份字典，键名 `checkpoint_sha256`——顺序要先读 `train_seed_corpus.py:300-325` 确认保存点与 `_flush(final=True)` 的先后，若在 flush 之后则须改到保存之后再写退出件，否则记的是上一轮的字节；②与既有守卫 `test_g14_trainer_exit_accounting` 同册加一条"退出件必须自述字节、且该值等于文件当前哈希"，并配一个**负对照**（把该键从写出字典里拿掉必须红），否则又是一条恒真守卫；③周期行一字不动（G14 那条钉的就是"这些键只出现在收尾那一行"）；④落码前先确认没有门禁在跑、且不在训练在飞时改被 import 的 trainer | **【已修，2026-10-03 15:49，同轮】**读源码定下了本条 ① 里那句"顺序待确认"：三处退出点（`train_seed_corpus.py:424/445/448`）原本全是 **`_flush(final=True)` 在前、`_persist()` 在后** ⇒ 若在 `_flush` 里哈希落盘件，记的确实是**上一次**保存的字节，所以修法是"先落盘再写记账"而不是"加个键"。已按此把三处顺序倒过来、独立件多写 `checkpoint_sha256`（`_file_sha256()`，件不在则 `None`），并**只写进独立件**——进度收尾那一行的键集不变（`test_periodic_lines_keep_their_old_shape` 钉着，新增键会当场红）。守卫 `test_exit_record_names_the_bytes_it_wrote` 断言该值等于落盘件当前哈希、且这一行不进进度件；它同时是顺序守卫：把顺序换回去时新建档场景下 `_file_sha256` 会拿到 `None` 而红（不是恒真式）。G14 那册 7 passed、ruff 0 条。另记一条本轮自查到的副作用：改这个 trainer 时 Edit 把整档翻成 CRLF（工作区 822 个 `\\r\\n`，而 `git diff --numstat` 因 autocrlf 归一**完全看不出来**，只报 18/5），按字节比 HEAD 才发现并归一回 LF——本仓已有"行尾只按字节比"的规矩，这次是它救了一次；训练在飞期间我没动过任何被门禁 import 的文件，跑的门也只有这一支。
| DEBT-G40 | **`--keep-checkpoints on` 的保号存档没有保留策略，也没有自述**（2026-10-04 凌晨，A-30 §125 那次 +2M-tick 训练当场观测）：`train_seed_corpus.py:318-319` 每次落盘**额外**写一份 `checkpoint_{ticks:012d}.pt` 进 `<件>.history/`，这条"保号"是 2026-09-23 为"中途状态一度不可复算"加的，但**没人给它上限**。实测本轮：到 `ticks=1,783,099` 时目录里 **35 枚、428.0 MB**（相邻快照间隔 ≈5 万 tick、每枚 ≈12.2 MB ⇒ 到 2M 约 40 枚／≈490 MB）。⇒ 三个问题都不是"大不大"：①落盘**数量与总字节不进退出记账**（`progress_exit.json` 只有 `checkpoint_path`），所以"这轮留了多少档"只能人事后 `du`；②这些快照既不在任何 sha 审计的引用面内，也没有"哪一枚值得留"的规则 ⇒ 下一次收束时它们既可能被随手删（正是 §126 那次丢掉 106 条声明的形状），也可能被人当成"反正有用"越积越多；③**我自己在 15:50 就把它估错过一次**（口头报"终局约 2.4 GB"，实算是 ≈490 MB，差 5 倍）——一个没有自述的量一定会被估错 | 低-中（本轮不撑盘：收束后剩 40 GB 级空余；风险是**下一次**长跑叠多次训练时无人知道会写到多大，以及保号件的可删性没有被任何仪器登记）| **登记未修（修法小，但要落码与守卫成对）**：①`progress_exit.json` 多两条自述 `history_files`／`history_bytes`（在 `_flush(final=True)` 里目录已存在，可直接数）；②加一条**上限策略**而不是无限保号：首枚＋末枚＋每 N 枚留一（或 `--keep-history-max`），超出即按最旧删除，且删除行为本身进退出记账；③守卫两支：短跑冒烟后 `history_files` 必须等于实际目录计数（不许是"配置值"），以及"超过上限时最旧那枚必须不在场"——**后者是负对照**，写不出红就等于没策略；④在 ①②③ 落地前，**不要**把这些快照当作可删件处理：按 `DEBT-G35/G36` 那条教训，删之前必须按裸文件名扫全部 `.py`（`_persist` 的命名模式 `checkpoint_{ticks:012d}.pt` 就是被引用面，虽然目前没有读数钉它） | **【已修，2026-10-04 00:32，owner 当日裁"限量＋自述"】**三处都落了：①独立退出记账多三条**现数出来**的自述 `history_files`／`history_bytes`／`history_pruned`（不给目录时全部为 `None`，绝不回显配置值）；②新 CLI `--keep-history-max`（默认 `None` ⇒ "每次落盘都留一份"的现行语义逐位不变），超限薄中间、**首尾各一枚必留**，`cap<2` 与"给了 cap 却没给目录"两种误用在 `run_training` 里 `ValueError` 响亮停；③守卫落在既有那册 `tests/taiji_native/test_train_seed_corpus_checkpoint_history.py`，6 passed ＋ G14 册 7 passed ＝ 13 passed，ruff 0 条。负对照的做法值得记一笔：**不手工抄"该留哪三枚"**，而是同一条链先按 `cap=None` 跑一遍拿实际落盘全集（6 枚：250…1500），再拿它当基准断言设限那遍"只剩 3 枚 ⊆ 全集、首＝全集首、尾＝全集尾、`history_pruned == 6-3`"——这样被守卫的东西改了会红，而不是跟着一起绿。本轮无门禁并发（我自己在跑的那趟中宽门禁已于 16:23 为腾出提交内存而撤销，未跑完的那趟不发表任何结论，读数结束后重跑）。 |
| DEBT-G41 | **`packages/client/ui-settings-general/tests/shell.client.spec.ts:112` 那条用例在 win32 真挂起，5 s 与 30 s 都不返回**（2026-10-04 14:57Z 判读结案；出处＝退出件 §13 判据④ 行 14:58Z 戳与 §14 第 11 格）：用例「shows Account first in Desktop while signed in and removes it on sign-out」先 `vi.stubGlobal('dshDesktop', {})`，随后所 await 的东西**永不 resolve**。单变量实验＝`--testTimeout=30000` 仍报 `Error: Test timed out in 30000ms`（`Test Files 1 failed (1)`，日志 `C:/Users/23747/AppData/Local/Temp/m6r13/shell_timeout_test.log`），整张 `vitest run packages/client`（`486 passed`，日志 `client_face_145146.log`，`START_UTC=2026-10-04T14:51:46Z HEAD=a74aa2c6`，`Duration 69.87s`）里同形红 ⇒ **一次实验同时排除三支解释**：宿主负载、跟队争用、"抬超时就好" | 中：这是 `packages/client` 全 scope unit 面上**唯一一枚真缺陷**——同批另一枚 `document-preview-license-bundle.client.spec.ts` 经正文核对属**入口不可测**（`Error: npm_execpath is required to run pnpm on Windows` 抛在 `runPnpm:38`，早于任何 tar 调用，故不是 `7929663ed` 那笔修复的回归），两者不得混记。后果＝判据④ **不得**挂"客户端面绿"；用户可见面是设置页在桌面态下 Account 项的显示与登出移除 | **登记未修（先定位再改，不许用抬超时冒充修复）**：①定位＝给该用例补一条**带哨兵的探针**（每个 await 前打点）单跑，读"停在第几步"，并先证探针自己被执行到（候选：桌面态分支所读的注入 Promise／store 首次订阅通知）；②修＝把那个等待变成用例内可 resolve 的**显式前提**，按本仓既有形状走，不要拦截或 mock 掉被测步骤；③配套＝本仓覆盖率门按每文件 100% 判，新增分支要带测试成员，且**守卫须能为假**（若最终是靠超时参数，就得有一条"挪小一格必须红"的负对照，否则零判别力） | 【登记未修，2026-10-04 15:01Z；本轮预算用尽未动码，两条读数与日志路径见本行第二列，引用时请连 `HEAD=a74aa2c6` 一起取】 | 【15:07Z 静态阅读把挂起点**收窄**到一条 await，但这是收窄、不是定位——没跑探针】读 `:112-127`，该用例的 await 有三类：`await start()`（fixture 装配）、`await c.mock.streams.opened('account/watch', 1)`、以及三段 `vi.waitFor(...)`。两条排除依据：① **`vi.waitFor` 默认 1 s 就以断言失败抛出**，不可能是 30 s 挂起 ⇒ 挂起点在它们**之前**；② 同文件其余用例不调 `dshDesktop` stub 也能通过（配对跑那次 `Tests 2 failed | 8 passed (10)`）⇒ `start()` 本身能 resolve。**剩下唯一嫌疑＝`await c.mock.streams.opened('account/watch', 1)` 永不 resolve**，即把 `dshDesktop` stub 成桌面态之后，Client 一侧根本没有去开 `account/watch` 这条流。⇒ 两种解释**尚未分开**：(a) 产品缺陷（桌面态应当订阅 account/watch 而没有）；(b) 用例前提缺失（需要先有凭据／账号态才会订阅）。分开它只要一次带哨兵的探针单跑（各 await 前打点、并先证探针自己被执行到），成本数秒，本轮预算用尽未做。**注意别用抬超时"修"它**——上一列已写明那支解释被 30 s 实验排除。 【16:00Z 定位完成，**结论是第三种**：既不是产品未订阅，也不是用例前提缺凭据，而是删除整包时漏改的悬空验收】证据链（全部 git 取法，无推测）：① 现 `src` 面里没有任何 client 代码订阅 `account/watch`（`grep -rn "account/watch" --include=*.ts* packages apps` 只命中生成的 `lib/*.d.ts`、本 spec、test-support 默认响应）；② `ui-settings-general/src/client/index.ts` 只注册 `id: 'general'` 一段，而 `SettingsRoot.tsx:32` 仍留有 `id === 'account'` 的图标分支；③ `git log -S "account/watch" -- taiji-harness/packages` 指到 `d292c3f6f`（品牌裁定"彻底删除 DeepSeek 登录那套"），该笔 `git show --name-only` 显示它**整包删除 `packages/client/ui-settings-account`**（含 `AccountSection.tsx` 等）并连带删掉它自己包内的 `tests/apply.client.spec.ts`（diff 里可见被删的 `await c.mock.streams.opened('account/watch', …)` 行）。⇒ 姊妹包 `ui-settings-general` 的这条用例断言的正是**已被裁定移除的整条链路**，它 await 的流再也没人开 ⇒ 30 s 仍超时是**必然**，不是竞态、不是负载、也不是产品缺陷。**修法（删测试属改变覆盖面，需 owner 一句确认，我不擅自）**：甲 摘掉该用例（同批删除的收尾）；乙 改写成"无 account 贡献者时设置段恰为 `PRODUCT_SECTIONS`"的负向断言（保留这枚用例的价值：万一 account 段回归，它会红）；顺带同一批还有 `SettingsRoot.tsx:32` 的孤儿图标分支与 `types.ts:30` 的注释残留可一并登记。**结案后 ④ 的客户端格才可能挂绿**（另一枚 `document-preview…` 属入口不可测，不是缺陷）。 【16:20Z 已修（owner 裁乙：保留覆盖面、不删用例）】`tests/shell.client.spec.ts` 那条改为**负向哨兵**——桌面形态下断言设置轨恰为 `PRODUCT_SECTIONS` 且过一微任务仍不变，不再 `await` 任何开流；account 段若回归会红。同批孤儿一并清三处：`src/client/SettingsRoot.tsx` 的 `id === 'account'` 图标分支（永不命中，`IconUserOutlineMedium` 随 import 摘掉）、`src/types.ts` 仍指 account 行的注释、`packages/test-support/client-runtime/src/assembly/remote-default-responses.ts` 两处把默认响应归给已删包的注释（条目本身保留，API／凭据面仍在）。**读数**：单跑该 spec `Test Files 1 passed (1)`、11.85 s、rc=0；客户端整面复跑 `Test Files 2 failed | 486 passed (488)` → **`1 failed | 487 passed (488)`**、`Tests 1 failed | 7109 passed | 1 skipped (7111)`，余下唯一一枚＝已判"入口不可测"的 `document-preview-license-bundle`（`npm_execpath` 抛在 `runPnpm:38`），**不是新缺陷**；lint 叶子四文件 `Found 0 warnings and 0 errors`；门 `doc-sync 43 passed`＋`hygiene 18 passed`，rc=0。基线 HEAD `a8f0a9dd`，提交 `e5eb7731`。**遗留（不在本条范围）**：那枚入口不可测的用例若要在这张面上可测，需给 `runPnpm` 一条不依赖 `npm_execpath` 的起法——属测试仪器改进，另一条债。
| DEBT-G42 | **`document-preview-license-bundle.client.spec.ts` 的 `runPnpm` 只在存在 `npm_execpath` 时才起得来，于是在 `pnpm exec vitest run` 这类入口上必然早退，成为面级读数里长存的假红**（2026-10-04 16:20Z 定位，与 DEBT-G41 同批：正文 `Error: npm_execpath is required to run pnpm on Windows` 抛在该文件 `:38`，**早于任何 tar 调用**；同一枚用例在 `pnpm run` 脚本入口上是 `Tests 1 passed (1)`） | 低-中：不是产品缺陷，但它让**任何**用 `pnpm exec` 起的客户端面恒挂 1 枚红，掩盖真实面状态；owner 已选"等覆盖率档跑完再签"，这枚红会在每轮判读里被重新剥一次 | **登记未修（仪器改进，两档，都要配能为假的守卫）**：甲＝`runPnpm` 在无 `npm_execpath` 时改用 `process.execPath` 直起 pnpm 的 CLI 入口（仓内已有 `scripts/pnpm-invocation.ts` 同族先例可参照，**别重抄生成链**）；守卫＝甲 落地后 `vitest run packages/client` 的失败文件数必须从 1 降到 0（不降即甲 无效）。乙＝该用例加面级前置守卫：检测不到 `npm_execpath` 时显式 skip 并在读数里按"未测"披露；守卫＝汇总行必须出现对应的 skipped 计数（静默消失等于没做）。两档都不许把"入口不可测"改报成"已测过" | 【登记未修，2026-10-04 16:22Z；出处：`client_face_after.log`（`Test Files 1 failed | 487 passed (488)`、`Tests 1 failed | 7109 passed | 1 skipped (7111)`）与结案同步 `422466eb`】 | 【16:45Z 已修（owner 裁甲：真修起法，排在跑覆盖率档之前）】只改本 spec 的 `runPnpm` 且只动 win32 一支：无 `npm_execpath` 时 `createRequire(import.meta.url).resolve('pnpm')` 读 `bin` 字段（本机 `bin/pnpm.mjs`），用 `process.execPath` 直起——不经 PATH、不经 `.CMD`（`EINVAL`）、不经 corepack 无扩展名垫片（`ENOENT`）。前置证明：`node <解析出的 bin> --version` ⇒ STATUS 0／`11.7.0`。**共享件 `scripts/pnpm-invocation.ts` 故意没动**：它被 `build.ts`／`coverage-partitions.ts` 用着，放宽其抛错契约＝改门禁语义，超出授权。本行写死的守卫已兑现：同一入口 `vitest run packages/client` 失败文件数 1 → **0**（`Test Files 488 passed (488)`、`Tests 7110 passed | 1 skipped (7111)`、FAIL 行 0、53.73 s），单跑该 spec `Tests 1 passed (1)`，lint 叶子 0 条；新防御臂按 AGENTS.md 用 `/* v8 ignore next -- 真实理由 */` 标一行，待 `check:ci:coverage` 复算。提交 `db0624dc`。⇒ 副作用值得记：④ 的客户端面从此有一张**入口无关、命令与面定义都在库、可复算的零红读数**；㊵-178 那句"四包面首次零红"仍是不可复算的历史读数，别再引它。
| DEBT-G43 | **`packages/lsp/lsp-stdio/tests/typescript-server.e2e.ts` 的临时工程目录在 win32 上被 LSP 进程占住，套件级 `rmdir` 抛 `EBUSY` ⇒ 四条用例全 ✓ 而该文件记为红**（2026-10-04 17:00Z 实测：`Test Files 3 failed | 42 passed | 40 skipped (85)` 但 `Tests 2 failed | 161 passed | 117 skipped (280)`，多出的那枚文件级红正文是 `Error: EBUSY: resource busy or locked, rmdir 'C:\Users\23747\AppData\Local\Temp\lsp-ts-e2e-XbQiFF\proj'`，同段四条均为 `✓`） | 中低：用例本身可信（4/4 绿），但它让 win32 整面读数**恒多一枚文件级红**，且会随进程释放时机抖动 ⇒ 面级计数不可比（同一 HEAD 我用错入口跑出 4 枚、用对入口 3 枚、旧档 2 枚）。**它是我把这条 lane 从"win32 根本不执行"修成真跑之后才暴露的**（见 ㊵-265），属修复的正常暴露，不是产品缺陷 | **登记未修（修法＝让清理与进程生命周期对齐，而不是吞异常）**：①用例结束先**关掉 LSP 子进程并 await 其 exit**，再做 `rmSync(dir,{recursive:true})`；②若仍偶发，退一步用 `it.afterAll` 内带**有限重试**的删除（重试次数与间隔要写死并披露，不许静默 catch `EBUSY` 冒充绿）；守卫两条：跑完后 `%TEMP%` 下 `lsp-ts-e2e-*` 目录计数必须为 0（**残留即为红**），且该文件的 `Test Files` 行必须为 passed——**只写"能过"的守卫等于没写**，第一条就是能为假的那条 | 【登记未修，2026-10-04 17:01Z；同面同入口读数见 `e2e_keyless_pnpmrun.log`（`START_UTC=2026-10-04T16:59:0xZ HEAD=b3bc77c9 TREE_DIRTY_TOTAL=…`），另存一次**入口用错**的对照：`pnpm exec vitest run` 起面会让两枚 `built-lib.e2e.ts` 因共享件 `scripts/pnpm-invocation.ts:15` 抛 `npm_execpath is unavailable` 而多红 2 枚 ⇒ 面级计数必须先钉入口（与 DEBT-G42 同族，实例已从 1 枚扩到 3 枚）】 | 【17:10Z 已修】`afterAll` 里 `rm` 之前没有等锁释放，而 `force: true` 不重试 `EBUSY` ⇒ 改成**有界重试**：`REMOVE_RETRY_INTERVAL_MS = 50` × `REMOVE_RETRY_LIMIT = 40`（≈2 s 预算），**耗尽即原样抛出真实错误**（不静默 catch，保持"能为假"）。按本行写死的双守卫复验：用**正确入口**（`vitest run --config vitest.e2e.config.ts <该文件>`）连跑两遍 ⇒ `rc=0`、`EBUSY` 命中 **0**、跑完 `%TEMP%` 下 `lsp-ts-e2e-*` 目录计数 **0**；lint 叶子 `0 errors`。**自陈一条取法错**：第一次"验证"我漏了 `--config`，默认面只收 `*.spec.ts` ⇒ 报 `No test files found, exiting with code 1`（rc=1），而那趟同时给出"EBUSY=0、残留=0"，**全是空跑出来的假干净**；若我没去读日志原文，就会把一次没执行代码的跑当成修复生效。⇒ 通用形状：**验证式必须先看它有没有真的跑到那条 lane**（汇总行或用例数为证据，rc 与"0 命中"都不算）。另记本行的入口教训已扩到 3 枚实例（两枚 `built-lib` 加本条），见 DEBT-G42 与退出件 §13/§14 的 17:04Z 戳。
| DEBT-G44 | **跨包按说明符 import 的源文件在覆盖率测量面上恒 0%**（实测：`packages/util/deque/src/index.ts` 44 条语句命中 0、`packages/core/scope/src/index.ts` 47 命中 0、同包 `store.ts` 78 命中 0、同包 `invariant.ts` 14 命中 0，而同一趟里 `packages/api/gateway/src/index.ts` 627 条语句 627 命中。根因＝linked workspace 依赖在 vitest 下被外部化，运行时命中 `exports` 指向的 `lib/`，`src/**` 从未被加载。别名指向文件还是目录**不是**决定因素——同一趟数据里两种别名形态都是 0，我先前提的"目录别名不生效"已被否证） | 中（它使判据④ 的"每文件 100%"含 5 枚**结构上不可达**的文件：补成员无效、给符号链接特权也无效，故不能记成成员债） | **处置需 owner 排，因为它改的是测量面定义**：出路甲＝让 workspace 依赖不被外部化（如 `server.deps.inline` 一类），代价是这 5 个包的真实缺口会**首次暴露**，涨债还是降债未测；出路乙＝按包显式豁免，但**本仓既有规则不允许**（`scripts/coverage-exempt.ts` 的成文条件要求该文件已被别的 suite 全覆盖，而这里恰恰"从没被加载"）—— 乙 会把真的测量盲区钉成不可见 | 未处置。出处 08 ㊵-313（2026-10-04 22:48Z）；复算命令 `corepack pnpm exec vitest run --coverage.enabled --coverage.reporter=json packages/api/gateway`（在 `taiji-harness/` 内，取 `coverage/coverage-final.json` 里上述四枚文件的 `s` 命中数） 【23:25Z 本行严重性按实测分级，"结构上不可达"改读为"部分可达、须按模块图耦合判定"】五枚零覆盖文件不再是齐质的：`packages/util/deque/src/index.ts` 与 `packages/test-support/loader-smoke/src/index.ts` **已结**——各改一处 spec 的自家根说明符为相对 `../src/index.ts`，该包 src 阈值红即清零、用例数不变（08 ㊵-326／㊵-328）；`packages/core/scope/src/index.ts` 与 `store.ts` **证否改法**——同形改动会把按包名键控的不变量注册表劈成两份（`InvariantError … "agent/created" … without a scope carrier`，2 红），改动已撤销，须先解决身份一致性（08 ㊵-327）；`packages/client/ui-life/src/index.ts` **未试**。⇒ 判据：只有"该测试的模块图里没有按包名建立的注册表／单例"时，改导入才是可行修法（deque 纯结构可、loader-smoke 可、scope 不可）。出路甲／乙的选择只对"改导入不可行"那部分成立，且甲的代价（首次暴露真实缺口）不变。 【02:40Z 本行的**成员名单按当前 HEAD 复测收缩**，且"五枚"这个数不再成立】`core/scope/src/index.ts`＝**80 条缺格**、`src/store.ts`＝**135 条**，两枚在自家全绿定向面（`Test Files 3 passed (3)`、`Tests 24 passed (24)`、`% Stmts` 表头存在）里四指标一律 **0%** ⇒ "从未被加载"在这两枚上**当前仍成立**，且改导入已被 `InvariantError` 否证 ⇒ 它们是本债剩下的**唯一实心成员**。而 `packages/client/ui-life/src/index.ts` **要移出本债**：同一取法下面绿（`1 passed`／`25 passed`）、表头存在，该文件只有 **1 条**缺格（`:4:17 uncovered function apply`），其余语句全部命中 ⇒ 它的模块**确实被加载并计量**，属"空入口里那个 `apply` 从未被调用"，**不是**"恒 0%"族（我当时把它和 scope 两枚并列是错的，错因＝沿用旧 gateway 趟的"0% 五枚"名单而没按当前 HEAD 复测）。⇒ 判据不变（"模块图里无按包名键控的注册表／单例"才可改导入），但**引用本债时请说"实心 2 枚（core/scope）＋1 枚待裁的空入口（ui-life/src/index.ts，已另计入待裁 (b)）"**；另 `packages/util/deque/src/index.ts`、`packages/test-support/loader-smoke/src/index.ts` 维持已结（㊵-326／328）。出处：08 ㊵-336／㊵-339／㊵-340。 【06:05Z 本行的**机制句被实测否证，病因要换】**：原文说"根因＝linked workspace 依赖在 vitest 下被外部化、运行时命中 `exports` 指向的 `lib/`"。可复现证据（08 ㊵-366）：把 `packages/core/scope/src/{index,store}.{js,js.map,d.ts,d.ts.map}` 这 8 个**被 `taiji-harness/.gitignore:12,14` 忽略的编译残留**移开，`packages/core/scope` 的自家面就变成 `3 passed (3)`／`24 passed (24)` 且 `src/index.ts`＋`src/store.ts` **各自 ERROR 0 条＋uncovered 0 条**（做完即移回，`git status` 复核干净）。⇒ 真机制＝`tsconfig.base.json:438` 把根说明符映射到**目录** `…/src`，目录解析优先命中已存在的 `index.js`，而覆盖率 `include`（`vitest.config.ts:209`）只收 `.ts/.tsx` ⇒ **产物被加载、源码不被计量**。同一件事解释了本行早先"改相对导入会炸 `InvariantError`"：只换一处时 spec 走 `src/index.ts`、其余包按裸名走目录命中 `src/index.js`，符号身份裂成两份；也解释了 deque／loader-smoke 为何能靠一行改导入达标（改完直接指 `.ts`，绕开目录解析）。⇒ **处置选项换**：不再是"外部化 vs 豁免"，而是甲＝根映射改指文件 `…/src/index.ts`（一处，覆盖全仓同类包）／乙＝覆盖率档前只清 `packages/**/src` 内的编译残留／丙＝让 typecheck/build 不再吐进 `src/`（根治）。三者都不动业务码，但都属测量/构建面定义，仍归 owner 裁；**下一轮请勿再去查 `server.deps.inline`**，那条路的前提已被本条否证。附带影响：本债名下的 `core/scope` 2 枚**不欠任何用例**，属"树状态相关"档，不是成员债。
| DEBT-G45 | **判据④ 的逐文件阈值结论依赖"树里有没有被忽略的编译残留"，因此在不同树状态下不可复现**（实测：仅移开 `packages/core/scope/src/{index,store}.{js,js.map,d.ts,d.ts.map}` 这 8 个被 `taiji-harness/.gitignore:12,14` 忽略的产物，`vitest run --coverage packages/core/scope` 就从 `src/index.ts`＋`src/store.ts` 四指标 0% 变成两文件 **ERROR 0 条／uncovered 0 条**，做完即移回。机制＝`tsconfig.base.json` 有 **286 条**把裸说明符映射到 `…/src` **目录**，目录解析优先命中已存在的 `index.js`，而 `vitest.config.ts:209` 的 `include` 只收 `.ts/.tsx` ⇒ 产物被加载、源码不计量） | 中（不改产品行为，但决定 ④ 能不能被声称：同一份代码在"刚跑过 typecheck/build"与"清过 src 之后"给出**不同的阈下集合**，故任何"还差 N 枚"都必须绑定树状态才有意义） | **处置需 owner 排（三者都不动业务码，但都改测量/构建面）**：甲＝包根映射指到文件（`…/src/index.ts`，一处消歧覆盖全仓同类包）；乙＝覆盖率/CI 档前窄版清掉 `packages/**/src/*.{js,d.ts,js.map,d.ts.map}`；丙＝停止把 typecheck 产物吐进 `src/`（根治）。并建议 ④ 的合并面复跑记录里**自证产物状态**（把 `ls packages/*/*/src/*.js | wc -l` 与 `git status --short | wc -l` 连同日志入档），否则两次读数不可比 | 未处置。出处 08 ㊵-366／㊵-367；反证已做：`credentials/src/index.ts` 22.85／30.76／22.22／11.76（该目录同样有 src 内产物却仍被正确计量，因其 spec 用 `../src/*.ts` 绕开目录解析）、`ui-workspace/src/client/navigation.ts` 94.55／92.68／94.71／91.42、`llm-taiji/src/index.ts` 91.66／98.03／91.66 ⇒ **本机制只解释 0% 那一族＝`core/scope` 的 2 枚**，未抬高余下 28 枚 ⇒ **㊵-382 后余 22 枚（该总数未经逐枚复核：㊵-386 证我用提交信息做减项的仪器两个方向都不可信，而 `fs-sandbox/src/containment.ts` 按三件齐读数实为已结清——我此前"该包 0 枚在名单"是假零。现值待 23 个包逐一跑定向面后以有名清单发表；只给数不给名一律视为未取到）**（`navigation.ts` 已在自家面四指标 100／100／100／100，`uw_cov3.log:1079`；行内那组 94.55／92.68／94.71／91.42 是 ㊵-366 当时的读数，按历史保留不再作为现状引用）【2026-10-05 ㊵-393：23 包定向面复跑全部完成，48 名单终分类已发表（现证三件齐 25／有表阈下 12／产物遮挡 2／红面无表 9；逐枚名单与定价在 08 ㊵-393③⑤）；本项两枚（scope index/store）本轮再测仍四指标 0%（uncovered 明细 80／135），src 内产物仍在位，遮挡机制再次印证，仍等甲/乙/丙 裁决】【2026-10-05 ㊵-394…396 夹具轮：`credentials/src/index.ts` 52 条⇒0（第 26 枚）、`client/ui-life/src/client/index.ts` 18 条⇒0（第 27 枚）、`LifePanel.tsx` 125 条⇒5 条（余 3 处防御支＝basenameOf 的空值右支＋两个 uploadFile 空 picked 守卫，经 UI 不可达，照 ㊵-345 先例改记等裁 c）；life-controller 六枚以点名 7 支已提交 spec 的 HEAD 面重定价（缺 78／97／14／0%×3），㊵-393 污染树读数作废；终分类现证 27／有表阈下 10／产物遮挡 2／红面无表 9＝48 ✓，阈下钥匙全数归位到裁 a/b/c、seam、特权、他线】 【2026-10-05 ㊵-400：合并面（`check:ci:coverage`）的 ④ 阈下文件由 **9 枚降到 6 枚**——`deliverables/workspace-changes` 的 `git.ts`／`index.ts`／`recorder.ts` 三枚在包内定向面三件齐（表头 1／ERROR 0／uncovered 0，四指标 100），余 6 枚全在 `api/life-controller`（per-file 9／57／52／14／78／97＝307 条，钥匙＝补跨包覆盖测试）。㊵-399 给那三枚的定价"4／1／23＝28 条真实缺格"在 166a8039＋一行测试夹具修正后的干净定向面里**不复现**，其逐枚计数按 ㊵-400⑤ 作废；判定权归属写为"逐枚结清以包内定向面三件齐为准，合并面普查只作上界与交叉核对"。本项两枚（`core/scope` 的 index/store）本轮未测，仍等甲／乙／丙，不受本条影响。】 【2026-10-05 ㊵-402：**更正上一括注**——其中"本项两枚（scope index/store）本轮未测，仍等甲／乙／丙"过期：甲已在 `6607744c` 落地（生成器 `entryFile()` 把包主别名指到 `src/index.ts` 文件、重生成 188 行覆盖全 200 包、`verify-tsconfig-paths` 门同步）。本项第一次复验（`scope_face400.log`：`3 passed (3)`／`24 passed (24)`、表头 1、ERROR 0、明细 0、rc=1 为别包 0% 所致）：`index.ts` 与 `store.ts` 在定向面都是 **100／100／100／100**，且 `src/` 残留未清（表格仍列 `store.d.ts` 0%，不计 ERROR／不计明细）⇒ 遮蔽来源已消、结论不再随"有没有清残留"翻转。**本项改判＝已处置（甲）＋两枚复验通过**；不宣布全清的理由：只消掉"包主别名指目录"这一条，`tsconfig.base.json` 其余目录型映射（原计数 286 条）仍在。④ 阈下文件仍 6 枚（`api/life-controller`），与 ㊵-399 合格合并面吻合（那次剩余 9 枚本就不含 scope×2）。】 |
| DEBT-G46 | **预注册里写了不能被机械套用的判据支**（2026-10-08，PLAN-N1-01 阶段0 判读当场暴露）：§3.1 的三出口写成"J-S1a 过 ⇒ …；不过 ⇒ 收回读取假说；**不过但 F1/F3 大幅改善** ⇒ 登记'读取可修但不充分'"，而**"大幅改善"没有冻结数值**（F1/F3 在 §3.1 里只有"目标线 ≥0.90／≥0.20"，那是辅读数的目标、不是出口条件）。⇒ 本轮实测落在这个含糊区：主判据不过（never-LF 行 225＞118），但 F1 0.549→0.747、F3 0.002066→0.042969（×20.8），**出口 2 与出口 3 都说得通**，判据本身无法裁决；而两把尺还给出**相反排序**（S5 终点档在行数上缓解更多＝209，但对 F1/F3 零改动；S1a 档在 F1/F3 上移动更多，行数只降 11）。我本轮的处理＝**不拿含糊支改判**（判级仍按唯一主判据＝不成立），把"部分中介"作为辅读数叙述，并把这条缺陷登记在此 | 中（**形状问题不是数值问题**：含糊出口会让下一轮把"我倾向哪个结论"包装成"判据说了什么"，正是本册反复登记的那族自伤；且它会在报告里被读成"判据三选一已机械判定"） | **修在下一次预注册（本册只登记不改旧件，判据文本一经冻结不追改）**：①出口必须写成**数值合取式**，例：`不过 且 (F1 − F1_baseline ≥ Δ1) 且 (F3／F3_baseline ≥ R3)` ⇒ 走"部分中介"支，否则走"收回"支，Δ1／R3 与主判据同时冻；②凡"行数"与"逐步率"两类量并存的判据，必须**预先指定谁做判据、谁只做定位**（本轮两者方向相反，事后无解）；③预注册落盘前跑一次**口语词扫描**——出口句里出现"明显／大幅／显著／足够"而无数值即视为未冻结（可用 `grep -E "大幅\|明显\|显著"` 在件上先自查，机器可检、且能为 false）；④守卫面：新预注册的判据段应被 `tests/` 里的契约测读到（同 `test_copy_circuit_*` 那种"判据文本被机检"的做法），否则第 ③ 条只是习惯不是门 | **登记未修**（属预注册写作纪律，非产品码；PLAN-N1-01 已按原判据判读入库，本报告 §4.2 点名本条；下一件 N 系列预注册（S1b 或解码/推导层归因）落地时必须带 ①②③ 的形态，并把本条状态改为已修）。**2026-10-08 就地更新（状态列同轮换）**：修法③ 已落地成机器门＝`scripts/training/audit_taiji_prereg_exit_wording.py`（判据句里出现 明显/大幅/显著/足够 且同行无数值钉 ⇒ 命中，rc=1；无判据段或缺件 ⇒ rc=2 响亮拒绝），契约测 `tests/taiji_native/test_prereg_exit_wording_audit_contract.py` **10 passed**，**两支都能红**：已知含糊件钉成"必须命中且行号条数不变"（N1-00 的 L7/L24、N1-01 的 L35——冻结件不追改，所以钉的是"仍被抓到"而非"已被修掉"），五件干净件钉成"必须零命中"（N1-02／N2-01／N3-01／N3-02／N3-03）；读数件 `reports/taiji_prereg_exit_wording_audit_20261008.json`。**两条如实边界写进仪器与测的 docstring**：rc=1 的语义是"要人看一眼"而不是判据判负（N1-00 那两处实为原文在否定该说法，形状上是假阳性），且豁免按**行**匹配（同行任一数值即豁免，粗粒度）。① ② 仍未落地 ⇒ 本条状态＝**部分已修（③已落地，①②待下一件预注册时核）** |
| DEBT-G47 | **离线巩固跑完之后，产品自己存的档产品自己装不回来**（2026-10-08，N2 通电当场实测）：`sleep_pass.run(organs=True, learn=True)` 结束时 runtime 停在它自己的睡眠回合里，此后 `SeedRuntime.save()` 写出的信封**两半自相矛盾**——`taiji.cognitive_state`＝{tick 66, episode_id `sleep-experience`}，`taiji.kernel.state`＝{tick 92}，而 `TaijiKernel.restore_native` 末尾那条守卫（[taiji/adapter.py:12631-12632](../../../taiji/adapter.py) `if state.tick != self.tick or state.episode_id != self._state.episode_id: raise`）比的**正是这两半** ⇒ `SeedRuntime.load(候选档)` 响亮拒绝 `ValueError: native cognitive state is out of sync with kernel state` | 高（**这不是仪器毛病，是 §8.3 候选版本隔离在"离线巩固"这条路上结构性不成立**：巩固产生的权重改动只能活在内存里，一落盘就废；N5"经验→能力"要能持久化必经此门；且它会静默把评测读成"能力＝0"——本轮 CAP-0 巩固后那张面 100 项**全部** `load_ok=false`、`machine_scored_correct` 恒 0，判读器若不先数装载失败就会把仪器缺陷报成结论）：实测件 `reports/taiji_n2_postcheck_20261008.json`（candidate `disk_load_ok=false`、68/202 张量与母档不同、回退档与母档 **0**/202 有差且摘要同为 `3e00bafc6298ac84`）；另用对齐件反证差的只是计数器不是权重（`reports/taiji_n2_align_20261008.json`：`tensors_differing=0`、`tensors_only_in_one=0`、未对齐候选档仍拒收） | **两条修法都由 owner 定，本册只登记不动产品码**：A＝睡眠回合结束时**收束回醒来态**（`seed_platform/sleep_pass.py:sleep_organs` 或 `seed/sleep.py:SeedSleepScheduler.night` 末尾把 episode/tick 交还给醒来侧——语义上是"睡完要醒"，改动落在巩固路径内部）；B＝`SeedRuntime.save()` 序列化前以 kernel 半边为权威覆写 `cognitive_state.tick/episode_id`（等于公开承认两半里 kernel 说了算，改动落在持久化边界上）。两条都要配**能为 false 的守卫**：`通电后立即 save→load 必须成功`——今天这条形状下它是红的，正是我们要的证据；并补一条反向：对**已一致**的档，修法不得改变 load 后的摘要（防止修法顺手覆写有效状态） | **2026-10-08 owner 弹窗裁＝修法甲**（睡眠回合结束时收束回醒来态，改在巩固路径内部；乙〔save 前以 kernel 半边为权威〕**未选**）**、**2026-10-08 已按甲实施**（`seed_platform/sleep_pass.py:sleep_organs` 在 `scheduler.night(...)` 之后收束 `reset_dynamics(episode_id="wake-after-sleep")`，模块常量 `WAKE_EPISODE_ID`；醒不过来只报 `wake_error` 不改报 `ran=false`）：**三支契约测**入库 `tests/seed/test_sleep_pass_wake_reentry.py`——反支钉缺陷本体（直接调 `night` 不收束 ⇒ `Seed.from_checkpoint` 必须 `ValueError: out of sync`，实测两半 34 对 60）、正支走产品路径（`sleep_pass.run(organs=True, learn=True)` 之后信封两半一致且能装回来）、反向守卫钉"收束只准动回合态"（选择器先证明非空：`edge_weight`/`pre_index`/`post_index` ≥10 枚逐位不变；同时要求回合态确实被动了，防"这步什么也没做"冒充通过）。**仍未做的**：本轮那一枚真实候选档 `seed_n2_candidate_20261008.pt` 仍是坏件（修不好，只能重跑），所以 **N2 的巩固后面仍必须在修法甲之下的**未对齐原生候选档**上重跑一次才算"通道打通"——而那是第二次改权重，不在 ㊵-485 的批文范围内，须回 owner** （本轮 N2 的处置＝用对齐件把候选权重读进来量能力，件内点名"这读的是**信封两半对齐之后**的候选权重，不是装机可用态"；J-N2a 的"巩固后"面全部建立在对齐件上，结项语里必须带这条限定。修法选定后本条状态改已修，且 N2 的巩固后面要在**未对齐的原生候选档**上重跑一次才算"通道打通"）。**⇒ 2026-10-08 本条结清（真实产品链实测，不再只有单测证据）**：owner 批的第二次通电（[PLAN-N2-02 判读](../../reference/PLAN-N2-02_ADJUDICATION_20261008.md)）产出的**未对齐原生候选档** `checkpoints/seed_n2b_candidate_20261008.pt`（12,340,387 B）被产品自己的 `SeedRuntime.load` 读回，摘要逐位等于通电后的内存态（`6f29e87dbccd8c6e9f651001ef7798bed3c52ec5f3db2b8b2ab8a9a1750a4cd7`，件 `reports/taiji_n2b_powerup_20261008.json` 的 `j_persistence=true`），且**反向那半**同绿（没睡过的母档 save→load 摘要不变＝`g_n2_5_mother_roundtrip_identical=true` ⇒ 修法不是靠覆写有效状态换可装载）；巩固前后回退三张 CAP-0 面 `load_not_ok` 全 0、严格命中总分 `[9, 10, 9]`。上面那句"本轮那一枚真实候选档仍是坏件、只能重跑"随之成为**被更正的原话**（第一次那枚 `seed_n2_candidate_20261008.pt` 仍不可救，也不去救——它已被第二次的原生候选档取代）。**本条关闭**；N2 的能力侧负结果另计（加速点① 在该剂量档判不成立，走 09 §4 第四行），不在本债务条目里 |

| DEBT-G48 | **`plans/` 引用面的唯一契约测长期红，而提交前四道门都盖不到它**（2026-10-08 本轮实施 DEBT-G47 修法甲时顺带抓到）：`tests/seed/test_project_identity.py::test_active_plans_have_one_execution_owner_and_resolvable_links` 判红，原因是 active 件里 **30 处**形如 `../../archive/history/a_line_20261007/README.mdPLAN-A-30_surface_repetition_localization_20260928.md` 的链接把**两个路径拼在了一起**（文件系统上不存在该串）；分布实测＝`plans/active/PLAN_INDEX.md` **27 处**＋本册 **3 处**（其余 active 件 0）。**不是本轮造成**：同一正则在 `726a0a75`（㊵-485，本轮进场前）与 `HEAD` 上给出**完全相同的 27／3**，`grep -c "README.mdSPEC"` 在 `726a0a75 / 530b9a37 / 84441370 / 09febc34 / HEAD` 五个 ref 上也恒为 2 ⇒ 这条红比我进场更早，且我这几笔提交一处未增未减 | 中（**它是"计划文档的引用必须可解析"这条身份合同的唯一执行者**：它长期红 ⇒ 以后任何人往 `plans/` 里写坏链接都没人拦，而本仓的 N 系列件靠交叉引用活着；另一层影响是**门的覆盖面**——本轮提交前实跑的仓根 `ruff check .`／`black --check .`＋harness `doc-sync`／`hygiene` 四道门**没有一个会跑这个 python 契约测**，红是 `pytest tests/seed -q`（180 passed / 2 failed）抓出来的，缝就在这儿） | ①先把坏链接的真实意图核清楚再动笔：归档 `plans/archive/history/a_line_20261007/` 下**只有 README.md 一枚**，⇒ 这 27 处八成是"想指 README 里以原文件名做标题的那一段"，正确形状是 `README.md#plan-a-30-…` 锚点而不是拼接串；核法＝在 README 里按名逐条 grep 标题是否存在（机器可检）。②改完必须**把这一测接进提交前门**（加进 doc-sync 的 leaf 清单，或至少在改动 `plans/` 引用面时强制 `pytest tests/seed/test_project_identity.py`），否则下次还是靠 `pytest tests/seed` 顺带撞到。③守卫形状：这条测本身已是"能为 false"的形状（它就是 false），修的时候**不许反过来放宽它**（放宽等于把合同删掉） | **登记未修**（本轮不追修：它不属于 N2/N3 任何一条泳道，且修法需要逐条核对 27 处的意图，属"先修仓库再谈主线"那一档；本轮 N2 通电单元的卫星同步只动 `plans/reference/*` 与 08/09/03/PLAN_INDEX 的**新增**行，未新增任何拼接式链接——两枚计数在我改动前后同为 27／3） | 
| DEBT-G48 状态更新（2026-10-08 ㊵-502：**①已修并实证该测第一次变绿；②仍未做**） | 坏链 30 处已清零；同一测的**第二条断言**当时也红，且红因与本册登记的说法不符（见下） | 同上 | **实际修的是什么**：(a) 30 处拼接串＝删除提交 `9c2ebff1e` 把引用"改指 README"时把旧文件名**粘在了路径尾巴上**（`…/README.mdPLAN-A-30_….md`）⇒ 修法＝去掉粘死的尾巴，**并顺带修掉原本就错的相对深度**（`PLAN_INDEX.md` 在 `plans/active/` 下 ⇒ `../archive/…`；`05` 在 `plans/active/roadmap/` 下 ⇒ `../../archive/…`；两处原先各少一层，被粘住的文件名恰好掩盖了这一点）。(b) 本册 ① 的猜测**是错的**：它猜"八成是想指 README 里以原文件名做标题的那一段"——实测归档 README 只有 **927 B／一条标题**，且其正文自己写明"引用请去 VISION §19.13／09／08 台账"⇒ 锚点形状不存在，正确修法只能是指向 README 本体。**教训＝登记里写的"成因"也是待核假设**，动笔前按台账编号回查一次原文（本条已把这条写进 ㊵-502）。 | **仍未做＝②把这一测接进提交前门**（现在它依旧只被 `pytest tests/seed` 顺带撞到；接法＝给 harness `doc-sync` 加一条调 python 的叶子，或在改动 `plans/` 引用面时强制跑该文件）——**这条属"门覆盖面"改动，不在本轮自行加**，且它与 DEBT-G51 的措辞门扩展可以并成一件。另记同测第二条断言：`expected exactly one execution owner, got []`——03 在 2026-10-06/07 改写成"N 系列执行日志"时把合同要求的 `## 当前唯一下一步：` 整条删掉了（一条都没有），本轮就地补回并写明"重写本文件时只能改内容、不许删声明"；修完 `tests/seed/test_project_identity.py` **4 passed**（此前长期 1 failed） |

| DEBT-G49 | **`HOLDOUT_PROBE` 不是独立集：14.3% 的探针窗口原样出现在训练语料里**（2026-10-08，建 §8.7 五项取数面时现算发现）：新仪 `scripts/training/audit_taiji_n3a_data_face.py` 把探针切成 35 个 24 字节窗口去真实语料里找，**5 个命中 ⇒ hit_rate=0.142857**（读数件 `reports/taiji_n3a_data_face_scale10calib_20261008.json`，语料＝`data/simple_zh/dialogue_extended_clean.jsonl` 108,327,171 B／123,090 篇） | 高（**它改的是历史读数的定性，不只是新跑的**：一切引用 `holdout_surprise` 的结论都含"背过一段"的成分——包括 ㊵-482 那条"平台＋泛化侧缓慢退（+0.041／+0.066）"，以及 ㊵-484 压强面里以 accuracy 侧为对照的部分推论；`holdout` 一词在此链上名不副实，而 §8.7 恰好把"独立测试覆盖"列为缺项，这条就是它缺的证据） | ①把"独立"变成事实：构造一个**按窗口命中率验收**的探针集（现成仪器即可——`audit_taiji_n3a_data_face._disjointness` 就是那台验真机，命中率必须 ≤ 阈值才可当 holdout 用；阈值先冻 0.0 ⇒ 要求零命中，或明写允许的污染率并说明影响）；②在此之前，所有引用 `holdout_surprise` 的判据/台账句都要带"该探针与训练语料的窗口命中率为 0.143"这条限定；③守卫形状＝把 `_disjointness` 的 `hit_rate` 做成面内自述键（已在新读数件的 `holdout_disjointness` 下），并在其 > 阈值时让取数面 rc≠0——"独立"由读数保证，不由命名保证 | **登记未修**（本轮只发现并入库证据；重建探针集会改动 `train_seed_corpus.py` 的 `HOLDOUT_PROBE`，属"改判据所在链"，须 owner 定阈值与换集时机；台账 08 ㊵-494 记录发现） |
| DEBT-G49 状态更新（2026-10-08 同日：**修法① 已实施＝部分结清，遗留三条如实列出**） | 缺陷本体未变（旧探针 5/35 窗口原样命中训练语料、hit_rate=0.142857） | 同上 | 已完成＝owner 第四次弹窗裁"零命中重建"后，按 [PLAN-N3-05 §1](../../reference/PLAN-N3-05_holdout_probe_rebuild_prereg_20261008.md) 的验收式执行：新仪 `scripts/training/verify_taiji_n3_05_probe.py` **复用**现成 `_probe_windows`/`_disjointness`（只加可选入参、默认不变 ⇒ §8.7 的在库读数件仍可逐位复算）；三候选同机两趟实测 `c1_tide` 271 B **0/0**、`c3_seed` 454 B **0/0**、`c2_bridge` 361 B **1/1 ⇒ 被同一把尺拒掉**，旧探针仍报 5 命中（判别力对照 G-N5-3）；选择规则机械（过验者里取字节数最接近 163 的那枚）⇒ 选中 `c1_tide`；落地＝**加 `HOLDOUT_PROBE_V2` 与进度新列 `holdout_surprise_v2`，旧常量与旧列一字不动**（保住历史读数的复算入口）。实证：改前/改后同参两支除 `elapsed_seconds` 外**六键逐位同**；契约测 4 passed 含**打分只读**（权重摘要＋`torch` RNG 状态都不动）与"码里的新常量必须等于验证件选中那枚"（防"件选 A、码放 B"的静默漂移）；放宽进度行键集之前按约定**先让它真红一次**（`1 failed, 6 passed`，点名 `holdout_surprise_v2`）。遗留＝**①**本册 §修法③ 要求的"把 `hit_rate` 做成进度件内自述键"**没做**，改由独立验证件承担（原因：进度件自带就得让训练器 import 审计仪、而审计仪 import 训练器常量⇒循环依赖；且每次收尾扫 108 MB 与"换语料/换探针时扫一次"代价不对称）⇒ 约束改为"**换语料或换探针必须重跑验证件**"，抓手是进度件里已有的 `corpus_fingerprint`；**②**新探针 271 B 对旧 163 B⇒**两列水平不可比**，只能各看同列方向（§4 设计时我把字数当字节数估错，如实记在 §7bis；补一枚更接近 163 B 的候选属下一件）；**③**历史所有引用旧列的结论**继续带 0.143 限定**，含 ㊵-482 那两条"泛化侧缓慢退" |
| DEBT-G50 | **一次 `--smoke` 会覆写一枚被文档锚定的在库读数件**（2026-10-08 本轮落地 N3-04 仪器时**自己踩实**）：`default_output_paths(smoke=True)`（`train_seed_corpus.py:664-678`）把冒烟的 **checkpoint** 改走 `output/`（2026-09-28 那次覆盖产品件事故的修法），但 **progress 仍留在 `reports/`** ⇒ 收尾那枚由 `exit_record_path()` 派生的 `reports/seed_corpus_smoke_progress_exit.json` 是**受版本控制**的文件，任何一次不带 `--progress` 的冒烟都会覆写它。实测代价：本轮两支烟测把该件的 `online_accuracy` 从 0.2436487297459492 改成 0.0034006801360272052、`checkpoint_path` 指向我临时造的 `output/n3_04_smoke_flag/checkpoint.pt`、`history_bytes` 6,038,989→13,327,955，**而且 `--checkpoint` 是我显式给的**——那条修法只堵了写靶的一半 | 中（**不是脏文件问题，是证据可追溯性问题**：这枚件被 05 本册（DEBT-G14 的键形状声称）、08 台账与归档 `plans/archive/history/m_series_execution_20261006/03_CURRENT_EXECUTION_M.md` 引用；值一旦被后一次冒烟换掉，那些句子读的就不是它们当初引用的那份字节。同族先例＝DEBT-G35（`--checkpoint` 覆写非 resume 源，导致 10 份读数永远无法复算）。本轮已 `git checkout HEAD --` 还原该件；键形状未变故未追溯改判任何结论——但"能还原"靠的是我恰好发现，不是机制保证） | ①把另一半堵上：`smoke=True` 时 progress 一并走 `output/`，并配一条**能为 false 的守卫测**"冒烟不得写 `reports/`"（现有 `test_g14_trainer_exit_accounting` 只钉键形状、钉不住落点）。②或按 DEBT-G35 同形加拒绝：派生的 progress／exit 落在**已存在且不是本次 `--resume` 源**的受跟踪文件上 ⇒ 响亮失败，要写就显式换名。③守卫形状：造一次"冒烟指向 reports/ 既有件"的跑，要求它必须红。④顺带更正 `default_output_paths` 的 docstring 过强声明——"冒烟绝不落到产品件上"在 progress 这一半上当时并不成立（只读到机制存在、就把修法规格估低，这是同一类） | **登记未修**（本轮只还原产物＋入库发现；修法动训练器的输出落点并要新守卫，与 DEBT-G35 的拒绝面需一起设计以免互相覆盖；台账 08 ㊵-496 记录发现与这一次自伤） |
| DEBT-G50 状态更新（2026-10-08 ㊵-503：**修法① 已实施并带能为假的守卫；②未做**） | 缺陷本体未变（冒烟缺省的 progress 派生出受跟踪的 `reports/seed_corpus_smoke_progress_exit.json`） | 同上 | **已做＝修法①**：`default_output_paths(smoke=True)` 的两个缺省现在都在 `output/`（checkpoint 早在 2026-09-28 改道，这次把 progress 也改道），`--progress` 的帮助文本同步改；**守卫能为假已证**——改代码之前那条测正向钉着这个错形状（`== root/"reports"/…`），先跑一次得 `1 failed, 13 passed`、失败行正是它；改完再钉四条 `reports not in …parts`（含 `exit_record_path()` 派生那枚），谁把缺省改回 `reports/` 谁就红。`test_a4_mainline_flags.py` **14 passed**、同域三组共 **33 passed**，`--help` rc=0（136 行；argparse 的 `%` 插值那一类雷未复发）。**未做＝修法②**（DEBT-G35 同形的"落到已存在且非本次 `--resume` 源的受跟踪文件 ⇒ 响亮拒绝"）：它要同时覆盖 `--checkpoint`／`--progress`／派生 exit 件三个写靶，得与 G35 的既有拒绝面一起设计（否则两道门互相盖，且 `--i-accept-default-product-checkpoint` 那个显式逃生口会被误伤）⇒ 留作下一件 |
| DEBT-G51 | **一份预注册里的两条守卫对同一个臂互斥，判读时只能替作者挑一种读法**（2026-10-08，跑 PLAN-N3-04 乙步骤二当场撞上）：G-N3b-1（㊵-484 的"默认关＝同参两支进度行八键逐位相同"形状）与 G-N3b-2（㊵-483③ 收紧后的"放行臂**不假设中性**，要把开/不开的参数摘要差与进度行差**量出来并披露**"）**写的是同一个臂**——一个要求它逐位同、一个明令不许假设它同。实测差确实存在且不小：同链开/不开旗标两支里 `online_accuracy` 0.21280320080020004 对 0.1690422605651413、`mean_surprise` 3.3646340823358925 对 3.729552795078888、`holdout_surprise` 2.9334548671532925 对 3.079834115409947（另外五键 `ticks`/`window_ticks`/`base_ticks`/`ticks_at_exit`/`reached_budget` 逐位同） | 高（**缺陷类别＝同一件里两条都"正确"，判读权落到我手上**：按 §4 字面"任一守卫不符即整件不判"⇒ 那 4 张在线面的读数一概不该出版；按 ㊵-483③ 的精神⇒ 这个差正是该报的数，而判据（§2 两条）不受影响。两种读法都能自证，**而我先前登记的 DEBT-G46 修的就是这一族**（含糊出口＋分支不互斥）——说明修法③ 那道措辞门盖不到"守卫段之间互斥"这种形状。本轮的处置＝判读件里把两种读法**并列写出**、按后一种报全数，同时声明"不替自己挑轻的那条"） | ①**守卫要按旗标分档写**：只读/记录类旗标 ⇒ 要求逐位同（沿用 ㊵-484⑥ 的形状）；改行为类旗标（放行、写入模式）⇒ 写成"差必须被量出来并披露，且本件不得据该差主张中性"，两条不许落在同一个臂上。②落盘前加一道**互斥自检**（可扩 `audit_taiji_prereg_exit_wording.py`：同一件里若同时对"开旗标那一臂"出现"逐位相同"与"不假设中性/须量出差"两种要求 ⇒ rc≠0 并点名两行）。③守卫段的措辞要能被机检区分"这是中性声明"还是"这是披露要求"——现形状下两者都写成"必须/不许"，人读也分不出。④已入库的那件（PLAN-N3-04）**不追改判据与守卫文本**（本仓纪律：冻结件不追改），修法只作用到下一件预注册 | **登记未修**（本轮把发现与两种读法并列写进 [PLAN-N3-04 判读件](../../reference/PLAN-N3-04_ADJUDICATION_20261008.md) §5；②那道互斥自检是仪器改动，属"先修仓库再谈主线"那一档，且要与 DEBT-G46 的措辞门合并设计以免两道门互相盖；台账 08 ㊵-499 记录撞出过程） |
| DEBT-G52 | **“数据不扩”这一臂只冻了意图、没冻实现，于是在等符号数下不可执行**（2026-10-08，N3 甲起跑前最后一格核出来；缺口是我自己留在 [PLAN-N3-02](../../reference/PLAN-N3-02_scaling_probe_prereg_20261007.md) §2/§4quater 里的）：§2 只写乙臂＝“同 seed、同 ×10 参数量档、语料量封顶”，没写**怎么在符号数不变的前提下让数据不扩**。两条天然实现都会引入第二个自变量——①把语料截成前 K 篇 ⇒ 默认 `--epochs 1` 下流到第 K 篇就吃完 ⇒ **两臂符号数不再相等**；②为保住 250,000 符号而抬 `--epochs` ⇒ **重复次数**成了新自变量，对照变成“数据不扩”与“同一批数据多看几遍”的混合。本轮处置＝**乙臂记 `unverified`、不起跑**，且**不当场发明封顶规则**（§4quater 明令那个附件不引入新自由度） | 中（**它砍掉本件判据的一半**：§2 的否证对照写着“两臂都算数，不许只报赢的那条”⇒ 甲臂单跑只能回答“×16.8 容量在 250k 暴露上是否把斜率抬出噪声带”，**答不了“封顶在容量还是数据”**——而后者才是 N3 要的容量决定；只报甲臂就是把半判据当结论） | ①给训练器加一个**不动 epochs 的“唯一篇数帽”入口**（例如 `--max-unique-documents K`：流到第 K 篇之后**循环重用同一批文档的字节流**），面内必须分别自述 `unique_documents`（唯一篇数）与 `document_visits`（到达篇次）两列；**两臂要 `unique_documents` 不同而 `ticks` 相同才算作对**——这条能为 false（不帽时两臂会同分，那是仪器坏不是结论）。②或反过来重设计对照＝**等数据量、变参数量**（`--scale 2` 与 `--scale 10` 吃同一段前 K 篇），但那不是 §2 冻的这条否证支 ⇒ 要升版重冻判据（旧件不删、索引标作废）。③先做哪条由 owner 定：①便宜且保住已冻判据，②更干净但代价是整件重冻 | **形状已定（2026-10-08 ㊵-512），债未结清**＝[PLAN-N3-10](../../reference/PLAN-N3-10_arm_b_shape_prereg_20261008.md) 落盘时先读出一条比"没写实现"更硬的事实：**`S ≈ U·L·R` 三轴里只有两轴能自由取值** ⇒ "符号数同 ＋ 唯一篇数不同"必然让**每篇被访问次数**跟着变，所以 ①②两条"天然实现"不是两个可选坑，而是同一约束的两种违约。冻下来的是**可执行形状 B**（`--max-unique-documents K`＝篇池循环重用＋面内自述 `unique_documents`/`document_visits`/`mean_revisits` 三轴＋"两臂 `mean_revisits` 若相等即仪器坏"的响亮失败），并加一条**读数边界 J-N3a-读法**：B 只允许回答"×16.8 容量在等符号暴露下是否把斜率抬出噪声带"，**不许**回答"封顶在容量还是数据"（那需要 C＝等数据量变参数量，而 C 不是 §2 冻的否证支 ⇒ 走 C 须整件升版重冻，owner 裁，我不当场替 owner 选）。仍未做＝①的**实施已于同日完成（㊵-513）**：旗标 `--max-unique-documents`＋三轴自述（`unique_documents`/`document_visits`/`mean_revisits`，只挂收尾行与 exit 件）＋三条响亮拒绝＋5 条契约测，两支 `--smoke` 实跑（K=3 ⇒ 3/6/2.0，默认关 ⇒ 4/4/1.0）；仍待＝②两臂跑量（会改权重 ⇒ 需 owner）、③C 的取舍·**〔2026-10-08 ㊵-529：②的剂量已裁并已开跑〕** owner 第七次弹窗点 **K=32**，已追加进 [PLAN-N3-10 §9](../../reference/PLAN-N3-10_arm_b_shape_prereg_20261008.md)（§2 形状／§3 判据与读数边界／§5 四条守卫一字未改）。两臂已按 PLAN-N3-02 §4quater 的甲臂命令顺序开跑（`--scale 10 --max-symbols 250000 --seed 20260822 --keep-checkpoints off`，乙臂另加 `--max-unique-documents 32`，落 `output/n3a_armA`／`output/n3a_armB`）⇒ **②从"待 owner"转为"跑面中"，判读要等面跑完**；跑完必须核 §2 的判对条件（`ticks` 相同、`unique_documents` 不同、两臂 `mean_revisits` 同时出版，若 `mean_revisits` 相等即判仪器坏、响亮失败）。③（C 形状取舍）owner 已在同一次弹窗里选 K=32 而未走 C ⇒ C 那格保持"需整件升版才能走"的登记状态，本轮不动 §2。 |
| DEBT-G53 | **生长阈的口径与闸的口径不同名：τ 冻在原始分布上，而六项合取比的是 EMA**（2026-10-08，PLAN-N3-06 重放在库面当场量出来；缺口留在 [PLAN-N3-01](../../reference/PLAN-N3-01_r4_hooks_prereg_20261007.md) 的冻结规则与 [PLAN-N3-03](../../reference/PLAN-N3-03_tau_definition_20261008.md) 的 τ 定义里）：`AdaptiveResidualGrowthTrigger.observe` 的合取闸（`taiji/adaptive_residual_growth.py:437-444`）六项比的全是 `*_ema >= minimum_*`，其中 `pressure_ema`（:388-395）＝**五个 EMA 的加权和**；而 `--pressure-record` 面件每行记的 `pressure`＝**五个原始信号的加权和**（`AdaptiveResidualGrowthPressure.pressure`），既有的 p90／"向上取整到 0.05 格"那条规则套的是后者。实测两者不是一回事：同一批 4 张面上原始上界 0.701569／0.705043／0.709197／0.709126（正是 ㊵-499 发表的那四个数），EMA 上界只有 0.67813／0.679165／0.685877／0.680911 ⇒ 默认阈 0.70 在 EMA 口径上**整场一步都够不着**（`meets_pressure` 真步数 0/3,999）；把同一条规则搬到 EMA 口径，beta 链出 **0.65**、circuit 链出 **0.70** | 高（**它使既有 τ 候选全部不可执行**：㊵-499 那句"按规则算出的 τ＝0.70 恰等默认值"说的是原始口径，而这个数放到闸上等于"永远不触发"；`minimum_*` 另外五道阈**面头根本没自述**（`train_seed_corpus.py:535-539` 只写三项），所以今天连"闸实际拿什么数和什么数"都要从产品默认值反推——这正是 ㊵-499 那条"不指认是哪道闸"的根因。它同时**不推翻**任何已发表判级：在场性成立／阈值不成立读的是原始分布，仍对） | ①τ 定义件升版：把 τ 显式定义到 **EMA 侧**，并把 `ema_rate`（:188）与五支 EMA 的初值（:359-363，四支 0.0、`resource_state_ema` 1.0）写进定义件——EMA 从 0 起步意味着**前若干步天然不可能过阈**，τ 若不做 warmup 限定就永远偏严。②面头补齐六道 `minimum_*`＋`ema_rate`，面行补记五支 EMA 与 `reasons`（decision 里全都有，`_record_pressure` 里就在手边 ⇒ 纯加披露、零重算）。③重冻后的 τ 必须**按链分别给**（beta 0.65 与 circuit 0.70 是同规则不同链的两个出口，跨链搬＝换尺）。④动 `AdaptiveResidualGrowthPolicy` 任何常量＝改产品默认行为，**须 owner 签字**，本登记不代改 | **修法① 已落（2026-10-08 ㊵-506）**＝[PLAN-N3-07](../../reference/PLAN-N3-07_tau_ema_caliber_definition_20261008.md) 把 τ 定义升到 `pressure_ema` 口径（六元组加"口径"、`Σwᵢ·EMAᵢ ≡ EMA(Σwᵢxᵢ)` 那条恒等式、warmup 显式处置、以及一条**规则粒度缺陷**：两链 p90 集合不重叠但间距 0.007085–0.009236 而格宽 0.05 ⇒ 候选值落在哪一格由取整决定）。**修法② 已落（2026-10-08 ㊵-508）**＝[PLAN-N3-08](../../reference/PLAN-N3-08_face_self_report_prereg_20261008.md) 实施完：face 头自述九条 policy 值＋五支 EMA 初值、每行观测补记产品 decision 自带的披露键，读侧同时认 v1/v2 并拒绝表外版本；实测新面 `assumed_from_product_defaults` 长度 **0**、自述为真 **6/6**，四张在库 v1 面判读件**逐位不变**（19,154 B／sha256 前缀 `2ee018515e01d9c0`）。**修法③④未做**（③ 已按实测改写＝"格宽 0.05 大于链间间距 0.007–0.009 ⇒ 先升规则，再谈按链给"；④仍未动任何常量）。发现与四条定价读数在 [PLAN-N3-06 判读件](../../reference/PLAN-N3-06_ADJUDICATION_20261008.md) §3/§5；台账 08 ㊵-505/506·**〔2026-10-08 ㊵-527：本行④ 已按 owner 裁定落成旗标，产品默认常量 0.70 一字未动〕** owner 第六次弹窗裁"τ＝**按链分别给值**"，落地形状查明为零产品码改动——`taiji/model.py:917-943` 的 `enable_adaptive_residual_growth` **本来就收 `policy` 参数**。训练器因此新增 `--growth-min-pressure`（`dest=growth_minimum_pressure`）：缺省 None＝**连 `policy` 参数都不给**（逐位不变），给值时 `dataclasses.replace(AdaptiveResidualGrowthPolicy(), minimum_pressure=v)` **只替一枚**、其余五道阈与 `ema_rate` 仍取产品默认，挂载后从 `trigger.policy` **读回复核**，不等即 `RuntimeError("…面作废…")`；面头 `developmental` 段新增成对自述 `minimum_pressure_requested`／`minimum_pressure_actual`（只加键，旧列一字未动＝G-N3c-2 同源）；取值域 0.0..1.0 越界 `parser.error`，且必须与 `--pressure-record` 同给否则响亮拒绝（防"以为接上了"的静默空转）。**测的两面都实走**：契约测 **6 passed（4.64 s）**——默认关那支在**调用边界**上钉"没给 policy"、开旗标那支把收到的 policy 六个字段逐一与产品默认比（"只动一枚"不许靠读侧事后声称）、负对照把挂载点换成吞掉请求值的假挂载 ⇒ 当场红、且面里不留一行把请求值当生效值出版、CLI 两支拒绝各 rc=2；同域回归 **47 passed（11.44 s）**（`test_n3_04`／`test_n3_09`／`test_n3_10`／`test_a4_mainline_flags`，`--help` 形状未破）。**兼容性另有一条实跑**：两枚 `--smoke` 面各被现行判读器 `freeze_taiji_n3_09_tau_rule.py` 读通（rc=0、`faces_refused=0`、`gate_table_rows_checked=4999`／`gate_table_mismatches=0` 两支各自为零）。**但这两支不构成单变量对照**——开档除了 τ 还带 `--developmental-fast-slow --developmental-bridge-gate 0.25`，压强分布差一大截（关档 `pressure_ema` max 0.423982／候选 τ 0.42／`six_gate_true_steps=0` ⇒ `not_freezable_at_this_grid`；开档 max 0.683186、p90 0.664335／候选 τ 0.67／true steps 38、longest run 4 ⇒ `freezable`），所以这里**只报"旗标的值确实进了面并被判读器读到"，不报"τ 改了判定结果"**。两链的 0.65／0.66 要按链跑面才能验"冻出的阈在这条链上够得着"，而跑面改权重 ⇒ **仍回 owner**；①②③ 三条修法不受本旗标影响（各自已落）。·**〔2026-10-08 ㊵-529：④ 已从"候选值"变成实测兑现〕** owner 裁"按链分别给值"后，旗标落地（㊵-527）并**两条链各真跑一支**：β 链装 0.65 ⇒ 面头 `requested=actual=0.65`、`six_gate_true_steps=36`、`longest_run=6`（≥ `required_pressure_steps=3`）、`rows_checked=3999`／`mismatches=0`、`verdict=freezable`；circuit 链装 0.66 ⇒ 同形 `steps=33`／`longest=7`、max 0.685877。同底默认关那支（τ=0.70）此前真步数为 **0** ⇒ **本行缺陷本体（0.70 在 EMA 口径上整场够不着）第一次有了同链的正面对照证据**，而产品默认常量仍未动。读数件：`output/n3_tau_beta/pressure.jsonl`／`output/n3_tau_circuit/pressure.jsonl`＋判读 `C:/…/tau_arm_beta.json`／`tau_arm_circuit.json`（落盘入库另在判读件格）。·**〔2026-10-08 ㊵-533：④ 兑现的不只是"候选值够得着"，产品判决第一次真的提议了生长〕** 只读四张面逐行数产品自己的 `decision_should_propose`（零新算法、零重算）：β 链装 0.65 ⇒ **8 次为真**（`persistent_native_pressure` 出现 8 次）、circuit 链装 0.66 ⇒ **6 次为真**；而**同一批面**在默认 τ=0.70 下两张都是 **0 次**（`pressure_below_threshold` 3,999 行全中）。⇒ **"R4 生长永不触发"不是架构必然，是阈不可达的后果**，这正是 09 §2 N5 那条痛点证据（出厂件 842 条目零进化）的反面。**边界必须说清**：两枚新面都带 `--developmental-fast-slow --developmental-bridge-gate 0.25 --no-readout-position` ⇒ 本条只主张"**提议这一层被打开了**"，不主张影子／晋升那两层已被走过；**产品默认位一字未动**（τ 仍 0.70、默认链仍零挂载），所以这不构成"产品已在进化"。另一条如实观察：`decision_consecutive_pressure_steps` 的最大值只有 **2** 而 `required_pressure_steps=3` ⇒ 计数列与"该列在提议前还是提议后记录"需要对表（已列为下一格的核查项，不在本条下结论）。读数件 `reports/taiji_n5_growth_proposal_evidence_20261008.json`（逐面 sha256、提议 tick 列表、reasons 分账、`structural_budget` 取值集合都在件里）。 |
| DEBT-G54 | **措辞门按"行内是否出现 出口/判据/J- 三个标记"挑判据句，一份不含这些词的文档可以整份对门隐形**（2026-10-08，写 PLAN-N3-07 时撞上）：`audit_taiji_prereg_exit_wording.py:31` 的 `CRITERION_MARKERS = ("出口", "判据", "J-", "J－")`，:38-42 只扫含标记的行，`criterion_lines == 0` 时给 rc=2 `no_criterion_section`（:82-84）。我的新定义件通篇是"规则/恒等式/处置"这类标题，**一条判据句都没被扫到**；rc=2 那一步是对的（它知道自己扫不到），但**"这份件没有判据段"与"这份件的判据段用词合规"是两件事**，而现行实现把前者当拒判、把后者的**缺席**留成一条可通过路径：把"判据"改叫"约定/口径/规则"的文档永远扫不出命中，而人只看 rc | 中（**它削弱 DEBT-G46 那套修法的效力**：措辞门是 G46 的机器侧执行者，而它的覆盖面由**被检者自己起的标题词**决定＝"由被检者定义被检范围"。本轮实证三种失败形态：①自指命中（把四个模糊量词抄进 A-4 那句禁令 ⇒ rc=1，㊵-505⑦）；②整份隐形（新件 rc=2，我改标题而不是改扫描器 ⇒ 见本行状态列）；③**同一处自指连撞两次**（㊵-508⑧：台账引用一句含词表词的原话被判命中，而我**描述这处命中的那句话又把原话抄了一遍**，第二次仍被判命中 ⇒ 修它的人成了触发者，这正是"门只看标记词挑行"这套取法会稳定产生的形状）） | ①扫描面**不靠词表挑行**：默认全文扫描，把"哪些行是判据句"交给结构（`## ` 小节下的**编号条目**）而非作者用词。②或反向加一条守卫：每份 `PLAN-*` 必须至少有一节标题含 `判据` 或 `出口`，否则 rc=2 ⇒ 把"隐形"从静默变成响亮失败。③`no_criterion_section` 与"扫到且零命中"要在 rc 上**分档**（现在 2/0 各表一事，但"扫到 2 行"与"本该扫到 200 行"同码），并披露 `criterion_lines` 的**期望下限**。④扩这道门时与 DEBT-G51 的互斥自检合并设计，以免两道门互相盖；属"先修仓库再谈主线"那一档，本轮只登记不自行加门 | **修法② 已落（2026-10-08 ㊵-514）**＝`audit_taiji_prereg_exit_wording.py` 加"存在性"守卫：`PLAN-*` 若二三级标题里没有 判据/出口/验收 ⇒ `invisible_criterion_surface`、rc=2（老通道一字未动，`criterion_headings` 同报）；新测 4 条＋旧 12 条全跑。**当场扫出 27 份 `PLAN-*` 里 5 份隐形**（`PLAN-B-03_cortex-split_20260925`、`PLAN-N2-02_ADJUDICATION_20261008`、`PLAN-N3-01_r4_hooks_prereg_20261007`、`PLAN-N3-03_tau_definition_20261008`、`PLAN-N3-06_ADJUDICATION_20261008`；后两份是本轮我自己写的），按"冻结件不追改"**只登记不代改**。**修法①③仍未做**（①不靠词表挑行的全文/结构化扫描；③rc 分档＋`criterion_lines` 期望下限）；写这条时又被夹具教了两次：大标题里的"判据"能糊过 `#{1,6}` 版扫描（⇒ 改成只认二三级），含糊句若不自身带 `J-`/判据/出口标记则根本进不了老通道 ⇒ 两条测各红一次才改对·**〔2026-10-08 ㊵-542：修法③ 已落（rc 分档＋期望下限），但默认关〕** `audit_taiji_prereg_exit_wording.py` 新增 `--min-criterion-lines`：扫到的判据行数低于下限 ⇒ 单独一档 `criterion_surface_too_thin`、**rc=3**（与 1/2 互不遮蔽），每件出版 `criterion_lines_floor`；多件时 rc 取**严重度秩最大**（秩 2>1>3>0）——这条以前是"最后一份赢"，`--doc A --doc B` 会把 A 的 rc=2 洗成 B 的 0，是真实缺陷，顺手一起修了。新测 `test_prereg_wording_rc_tier_contract.py` 五支（薄面⇒3／关档⇒0／含糊优先不被薄面洗掉／过下限⇒0／多件取最重），连同旧可见性测 **9 passed（2.10 s）**；两份真件在下限 3 下为 N4-02 **11 行**、N3-12 **14 行**，都 ok。·**剩余两项如实登记**：①默认仍是 0＝不分档（向后兼容），把默认改成 3 需要同步更新既有可见性夹具里的判据行数，不做"偷偷让旧测变红"的事；②写这批测时我自己错过一次——夹具 `## 3. 判据` 那行标题本身含"判据"二字，门的挑句规则按行内标记算，所以 `criterion_lines` 实测是 2 而不是我断言的 1（**行为对、断言错**，第一次跑出 rc=3 但 assert 失败，我先复核的是"到底是门错了还是我数错了"）。 |
| DEBT-G55 | **一份已入库报告件里的 `contract` 字段指向被按留存政策删掉的过程文档，测试的文件存在性断言因此永久不可满足**（2026-10-08 发现于 PLAN-N3-08 实施那一格的同域测，**不是本次改动造成**）：`tests/taiji_native/test_product_default_checkpoint_provenance_contract.py:147` 断言 `(REPO / preflight["contract"]).is_file()`，而 `preflight["contract"]` 取自报告件＝`plans/reference/PLAN-A-30_surface_repetition_localization_20260928.md`；该文档随 `ec5821a1`（2026-10-07"第一批蒸馏收束"）移入 `plans/archive/history/a_line_20261007/`，之后又按 owner 的留存政策（过程文档蒸馏后删除）被清掉——归档目录现存内容只有 `README.md` 一枚，无正文 ⇒ **断言永远无法再真**。另有 6 处仓内引用同一枚已删路径：五支仪器把它当 `prereg` **标签**写进报告（`measure_taiji_a30_repetition_penalty.py:167`、`probe_taiji_a30_stop_failure.py:1919`、`probe_taiji_a30_position_cycling.py:319`、`probe_taiji_a30_self_contamination.py:290`、`audit_taiji_a30_stop_signal_presence.py:471`）＋另一支测 `test_a30_stop_failure_v6_replay.py:236` 直接构造该路径 | 中（**常驻红会把"这条红是不是我造成的"变成每次都要重查的成本**——本格就是靠 `git log --diff-filter=D` ＋读断言源码才确认与本次改动无关；DEBT-G48 那一族"引用面测长期红、提交前四道门盖不到"的**同型复发**，说明 G48②（把该测接进门）没做导致同类问题二次出现）。标签类引用不会红（不读文件），但**同一枚已删文档在仓内有 6 处指认**，其中 2 处会读文件＝2 条常驻红源 | ①把那两条"存在性"断言改成**形态＋留存位置**两级：路径字符串形态正确 ⇒ 通过；同时要求该文档的结论留存位置可指（归档 README／墓碑件），别要求正文还在。②给 6 处引用一次统一标注"过程文档已蒸馏，结论在 VISION §19.13／09 S7 基线"，并把"引用 `plans/**` 的 .py 字面量"纳入 `verify-repository-references` 的扫描面（现在它只查 markdown，所以脚本里的死路径全程无人管）。③修时与 **DEBT-G48②**（把 `test_project_identity` 这类引用面测接进提交前门）合并设计——否则"接进门"会把这条红变成每次提交都挡路的常驻红 | **登记未修**（本轮只登记不修：它属"先修仓库再谈主线"那一档，且修法②要动门、③要与既有待裁项并件；发现经过与证据链记在台账 08 ㊵-508；同域测现状实测＝**1 failed／84 passed**（127.81s），失败那一支与本格外所有改动无关）·**〔2026-10-08 ㊵-515 之后补三条更正与量值，本行原先的数字偏小〕**：① **爆炸半径是 3 条测不是 2 条**——本行"严重"列写的"2 处会读文件＝2 条常驻红源"**少报了一条**；基线实测 `3 failed／35 passed`（两文件合跑），三条分别是 `test_product_default_checkpoint_provenance_contract.py::test_the_switch_was_preflighted_on_the_bare_loader`、`test_a30_stop_failure_v6_replay.py::test_instrument_carries_v6_and_the_correction_note`、`...::test_recall_numbers_after_docking50_name_the_circuit`（后两条同文件，第一条测也要读那份已删件）。② **修法① 已被试过一并不合格回退**（不留半成品）：按"两级守卫"改了两处测后仍不合格，整批 `git checkout` 还原 ⇒ 想直接照抄某个实现的人注意：**没有已验证的实现可抄**。③ 两个必踩的坑先记下省下一步：墓碑 `plans/archive/history/a_line_20261007/README.md`（8 行）**不逐件列文件名**（只写"原存 13 件：PLAN-A-25…30…"）⇒ "按 `PLAN-A-30_...md` 全名查墓碑"那种断言必错；改成查 `PLAN-A-25` 那一段＋对墓碑里反引号 `.md` **逐条**核存在性（我第一版写 `all(...)` 直接判红一次）。③bis **那次判红的"存在性"解释已否证**：墓碑里反引号点到的三枚 `.md` 目标实测**全部存在**（`plans/reference/VISION_FUTURE_TECHNOLOGY.md`／`plans/active/roadmap/09_NEXT_MAINLINE_PLAN.md`／`plans/active/roadmap/08_UPSTREAM_SYNC_PLAYBOOK.md`，逐条 `exists=True`）⇒ 不是存在性失败，我原先"把它算作一条"那句作废。④ **三条红是两种成因，本行原先把三条都记成"读已删文档"＝错**：a) `test_recall_numbers_after_docking50_name_the_circuit` 真因 `FileNotFoundError: ...PLAN-A-30_surface_repetition_localization_20260928.md`（读正文）；b) `test_the_switch_was_preflighted_on_the_bare_loader` 真因 `assert (REPO / preflight["contract"]).is_file()` 为 `False`（同一枚已删路径）；c) `test_instrument_carries_v6_and_the_correction_note` **与已删文档无关**——它红在 `test_a30_stop_failure_v6_replay.py:87` 钉的字符串 `"format": "taiji-a30-stop-failure-v41"`，而探针 `probe_taiji_a30_stop_failure.py:1707` 现写 `...-v44`；该测 :88-90 那条 `all(f"format_note_v{v}" … range(6, 42))` **不红**（`format_note_v` 在探针内出现 43 次，`format_note_v40`…`v44` 五枚注记逐枚在案）。⑤ **第三条红的引入点查到在我自己这一串里**：把探针升到 v44 的那笔是 `726a0a75`（㊵-485，已推送），`git show --stat` 里 `test_a30_stop_failure_v6_replay.py` 命中数 **0** ⇒ 升版那笔没同批改版本钉，而提交前那四道门盖不到这种"钉源码字符串"的红 ⇒ 与本行"严重"列的 G48② 同族**第二次复发**（第一次红的是引用路径，这次红的是版本钉）。⑥ **当前基线复测＝`3 failed, 35 passed in 1.44s`**（两文件合跑），失败名与 ④ 三条逐一对上；本行先前那句"127.81s／1 failed／84 passed"是更宽的同域集合，两个数不同源、不许互相代答。 |
| DEBT-G56 | **预注册把"臂的形状"与"要逐位对标的基线"写成两回事 ⇒ 冻出一条按字面不可能通过的守卫**（2026-10-08，跑 PLAN-N3-09 守卫臂当场撞上；缺口是我自己留在 [PLAN-N3-09](../../reference/PLAN-N3-09_tau_rule_upgrade_prereg_20261008.md) §3/§5 里的）：§3 把守卫臂写成"只给 `--pressure-record`（两个装配旗标都不给）"，§5 的 G-N3e-2 却要求这支臂与 **㊵-498 那支默认关臂**八键逐位相同——而后者是带 `--no-readout-position` 跑的（其 `online_accuracy` ＝ 0.21280320080020004）。⇒ 两臂形状不同源，**这条守卫按冻好的文本永远为假**，与仪器好坏无关 | 中（**同一件里两条文本互相矛盾而两条都能自证**＝DEBT-G51 那一族的新一例，且这次是**臂形状**而不是守卫语义：G51 是"逐位同 vs 不假设中性"打架，G56 是"这臂怎么跑"与"跟谁对表"打架。实测差很大且整齐：八键里 3 键有差（`online_accuracy` 0.21280320080020004 → 0.2503125781445361、`mean_surprise`、`holdout_surprise`），逐行原始列 **3 个字段每行都不同**（11,985 ＝ 3,999×3）⇒ 不是抖动是形状。另买到一条通用事实：**位置输入的开与关会产生两张不同的面**（㊵-497 只说了"不许半路关"） | ①落盘前加一条**臂形状对表自检**：凡守卫句里出现"与 ㊵-NNN 那支…逐位相同"，必须同时把**那支的完整命令形状**（含 `--no-readout-position` 这类"不改变判据但改变分布"的旗标）抄进本件，两者缺一即拒判——可扩 `audit_taiji_prereg_exit_wording.py` 那一族做成机器侧检查（键用"对表／逐位相同／那支"这类**动作词**，不用编号，㊵-502 那条教训）。②本件的即时 remedy 已在判读件 §4 写明：**补跑一支**默认关臂（带上那枚旗标，≈46 秒、≈14.7 MB，仍在原批文数量内），补跑前 G-N3e-2 记 `not_established` 不记"过"。③既有四张治疗面**不受此债影响**（它们的对表用的是"原始六列逐行同值"，实测 3,999×6 **0 处差异**） | **②的 remedy 已于同日执行**（2026-10-08 ㊵-511：按 ㊵-498 形状补跑一支默认关臂，带 `--no-readout-position`，面在 `output/n3_09_v2/guard_off_v2_fix/`）⇒ 与基线**八键逐位相同**（`online_accuracy` 两侧同为 0.21280320080020004）、原始六列 3,999 行 **0 处差异** ⇒ **G-N3e-2 从 `not_established` 转为「过」**，且红因确证是那枚旗标而非仪器或产品。**但本条债仍未修**：修的是**落盘前的自检**（①那条机器侧检查），本轮撞出来的形状错只能靠补跑兜回来；①要与 DEBT-G54 的措辞门扩展并件设计；判读过程与补跑证据在 [PLAN-N3-09 判读件](../../reference/PLAN-N3-09_ADJUDICATION_20261008.md) §2 表末＋台账 08 ㊵-510/511 |
| DEBT-G57 | **仪器自述的格式版本已升到 v44，而钉它的那条测仍钉 v41 ⇒ 升版动作与版本钉分居两文件，四道提交前的门盖不到，红因此常驻**（2026-10-08，查 DEBT-G55 第三条红的真因时量出来）：`tests/taiji_native/test_a30_stop_failure_v6_replay.py:87` 断言 `'"format": "taiji-a30-stop-failure-v41"' in source`，`:88-90` 另要求 `format_note_v6..v41` 逐版在案；而 `scripts/training/probe_taiji_a30_stop_failure.py:1707` 现写 `v44`。把探针升到 v44 的那笔是 `726a0a75`（㊵-485，已推送），同一笔里这条测的改动数＝**0**（`git show --stat` 计数），⇒ 该笔同时留下"探针自述升版"与"版本钉过期"两个事实，而**没有任何一道门把这两者放在一起看** | 中（**它让一条测长期为红，且"这条红是不是我这次改动造成的"每次都要重查一遍**——DEBT-G48②／G55 那一族的第三个实例：G48② 红在引用路径、G55 红在已删文档正文、本条红在**源码字符串钉**。也实证"改到被源码守卫钉住的行要同笔重钉"这条纪律只有口头版，没有机器侧执行者） | ①最低成本修法＝把钉法从"精确等于某版"改成**同源取现值**：测里现读探针里的 `"format": "taiji-a30-stop-failure-v<NN>"`，再要求 `format_note_v6..v<NN>` 逐枚在案（历史注记仍是硬要求，版本号不再需要人肉同步）。②或加一条"成对改动"门：某笔改动若碰到 `scripts/training/probe_taiji_*.py` 的 `"format":` 行，必须同时碰到钉它的那条测，否则 rc≠0（可并入 `verify-repository-references` 那一族，但那一族现在只看 markdown，见 G55 修法②）。③①②任一条都要配**能为假的负对照**：只挪探针或只挪测的版本号 ⇒ 必须红；两边一起挪 ⇒ 必须绿。④与 DEBT-G55 修法②、DEBT-G48② **并件设计**（三者同属"引用面／钉面测不在提交前门里"） | **登记未修**（本轮只更正归因＋立新债，不改那条测也不改探针：①要改的是钉判据面的仪器，属"先修仓库再谈主线"那一档、由 owner 排期；证据与全部数字在台账 08 ㊵-516）·**〔2026-10-08 ㊵-537：修法① 已落，且本条与 G57 的边界写清了〕** `test_a30_stop_failure_v6_replay.py::test_instrument_carries_v6_and_the_correction_note` 里那两枚钉死写法（钉字符串 `"…-v41"` 与 `range(6, 42)`）已换成**同源取**：`re.findall(r'"format": "taiji-a30-stop-failure-v(\d+)"', source)` 读出现在自述的版本，并加三条硬约束——①**自述必须恰好一枚**（出现两枚即 `AssertionError`），②历史注记仍要求 `format_note_v6..v<现值>` 逐枚在案，③现值不得低于本轮认定的 41。**守卫能为假的两面都实走过**：合成"只把自述版本 +1、不同步注记"⇒ 必报 `missing_notes == [44+1]`；合成"自述出现两枚"⇒ 必抛"恰好一枚"，两条都写在测里而不是靠看代码。⇒ **基线实测由 `3 failed, 35 passed` 变为 `2 failed, 36 passed`**，剩下两条正是本行 a/b 那对读已删文档的（`FileNotFoundError`／`is_file()` False），**属 G55 不属本条**。·**顺手记一条自己的错**：第一版正则我写成 `-failure-(\d+)` 漏了字面量 `v`（真实值形如 `-failure-v44`）⇒ `findall` 返回空、测照旧红，是我的 `diag` 打印 `matches=[]` 才定位到的；"匹配不到"当时看起来完全像"文件里没有版本自述"，实际是我自己的模式错了——再次印证 `[缺席类结论要先枚举同类全集]`。·本条剩余的②③（"成对改动"门、与 G48②/G55② 并件）仍未做。 |
| DEBT-G58 | **主线规划拿一个无出处的读数当 S4 的定价依据，而该读数在今天的代码形状上根本不可复现——"34.6% 空库/寻址失败"全仓只有两处互相转述、零取数件，且"空库"与"寻址失败"在代码里塌成同一个 `()`**（2026-10-08，owner 裁"开 N4"后的第一格核查）：`09_NEXT_MAINLINE_PLAN.md:105` 与 `VISION_FUTURE_TECHNOLOGY.md:1127` 各写一次该数，**两处都不给指针**；按值扫 `reports/` 全部 **1,649 枚 json**、在 0.3455–0.3465 与 34.55–34.65 两区间得 **808 条**命中，逐条看键名（`p_boundary`／`min_ratio`／`oracle_position_winner_fraction`…）**无一条语义是"空库率或寻址失败率"**；最接近的仪器 `scripts/training/eval_taiji_p4_episodic_recall.py` 用的是合成 `torch.eye` 线索与 `"action-{index}"` 标签（`:60-71`），产出的是病变对照门（`gate.criterion` 在 `:125`），**不是真经验流量上的寻址失败读数**。更要紧的是代码形状：`taiji/adapter.py:10950-10952` 在 `self._episodic_memory is None` 时把 `episodic_ids` 直接写成 `()` ⇒ **"根本没挂载库"／"库是空的"／"有库但寻址没命中"三种状态塌成同一个空元组**，无计数也无回退原因键；而 `attach_episodic_memory` 的**非测试调用点全仓只有 5 处、全在 `scripts/training/eval_taiji_p{3,4,6}*.py` 内**（定义在 `taiji/adapter.py:7113`），`api/**` 与产品默认链**零挂载点** ⇒ 今天的默认产品形态下 `episodic_ids` 恒为 `()` | 高（**N4 的 S4 切片正是按这个数定价的**（09:105 写"对应仓内 34.6% 空库/寻址失败读数"）⇒ 规划侧"痛点证据"不可复核，与 ㊵-499 那条"不指认是哪道闸"同族：**先在仪器里把状态分开，才谈得上判据**。三态不可区分还使第二重后果成立：即使补了稀疏编码，今天也没有一把尺能读出它起没起作用。另含一条 owner 级前置——要在真产品链上测寻址失败，得先定"记忆库由谁挂载"（改产品默认位＝签字项，本登记不代改）） | ①把三态拆开并计数：在 `episodic_ids` 的用处区分 `store_absent`（`_episodic_memory is None`）／`store_empty`（有库零记录）／`address_miss`（有记录但 `retrieve()` 返回空）／`id_gated`（命中但被 `_recovery_memory_is_readable` 挡下），各自进面自述键，并配"挪一格必须能红"的守卫。②先做一次零训练的挂载态复算：照 `eval_taiji_p4_procedural_runtime.py` 那类挂载脚本，在点名 checkpoint 上走真经验流量，产出 `addressing_failure_rate` 的**分子/分母取法**；然后把 09:105 那句改成"取数件＋口径"，或按 §19.13 的就地更正形状**作废该数**（已发表文不追改结论、只加更正行）。③在②完成之前，N4 的预注册**只能冻测量定义与合取形状**，不许把 34.6% 当基线数值写进判据 | **登记未修**（本轮只核查与入库：①要动 `taiji/adapter.py` 的取用处＝产品码改动，②要跑挂载面（至少挂载库）⇒ 两条都回 owner；扫描规模、代码出处与结论全数记台账 08 ㊵-524；PLAN-N4-01 按本行③的形状起草）·**〔2026-10-08 ㊵-525 追加：本行③的形状已落成仪器，①②仍未做〕** PLAN-N4-01 已落盘（母量＝`wrong_top1_rate`），其 §2/§5 的仪器同日实施完＝`scripts/training/measure_taiji_n4_addressing_surface.py`（format `taiji-n4-addressing-surface-v1`；可数三态＋第四态自述 `not_observable_here`；`ruler_usable` 要求率严格落在开区间；四枚 rc=2 拒绝支＝缺件／`capacity<=0`／`limit<=0`／零配对查询），契约测 **10 passed**（1.47s），CLI 冒烟 rc=0、读数件 1,502 B、`materials_sha256` 前缀 `7964d7fb17f535a4`。夹具还当场抓到实现里一处真错：真值落在前 `limit` 位之外时 `.index()` 会抛 `ValueError` ⇒ 现已单列 `true_rank_outside_top_limit`，不许折进中位数、也不许当成零命中。**冒烟另外买到一条本行③之外的新缺陷＝DEBT-G59**（"五等分相邻段最大跳幅"套在逐查询 0/1 序列上时，带的下界是 `1/(n//5)`，8 条查询处带恒为 1.0 ⇒ J-N4-2 在那个 n 上算术不可满足），故仪器已把 `block_size` 与 `noise_band_floor` 一起出版。·**〔2026-10-08 ㊵-528 追加：本行③ 的另一面已被实测坐实——仓里没有可复用的真值记录〕** `torch.load` 四枚在库档（`output/a31_chunked_self/checkpoint.pt`／`checkpoints/seed_a31self_with_circuit.pt`／`checkpoints/seed_n2_candidate_aligned_20261008.pt`／`output/tau_smoke_off/checkpoint.pt`）逐一查 `episodic_memory` 键 ⇒ **四枚全部为 `False`**（写入点在 `taiji/adapter.py:11771`，只有挂载过库的跑才会写）。⇒ 因为默认链零挂载，**PLAN-N4-02 要的 `paired ≥ 100` 带真值配对查询不可能从在库产物里凑出来**，唯一来源是 §4 那次挂载链跑面 ⇒ **N4 现在卡在跑面批文，不是卡在造题集**（我在 03 队首写过"先造题集再谈批文"，那句已按本条否证并在 ㊵-528 更正）。另更正一条会过期的计数：09:112 的"consolidated 三枚小文件"现为 **7 枚**（三枚 2026-09-27：60,018／60,018／3,230 B；四枚 2026-10-08 当日新增：178,905／166,897／147,731／140,213 B），且它们是语料不是情景记录。 |
| DEBT-G59 | **判据里的"噪声带"是从别族的仪器抄来的形状，套到新母量（逐查询 0/1 序列）上时它的下界由样本数决定，小 n 处恒为 1.0 ⇒ 那条收益判据算术上不可能成立**（2026-10-08，N4 仪器的 CLI 冒烟当场量出来）：[PLAN-N4-01](../../reference/PLAN-N4-01_addressing_measurement_prereg_20261008.md) §2 写"噪声带沿用 [PLAN-N3-02 §0](../../reference/PLAN-N3-02_scaling_probe_prereg_20261007.md) 的形状＝按查询块五等分取段均值、带＝相邻段最大跳幅、不引外部阈值"。但那套形状是为 **tick 时间序列上的连续段均值**设计的；母量换成 0/1 之后，段宽 `size = max(1, n//5)` 直接决定带的下界 `1/(n//5)`。实测：`n=8` ⇒ `block_size=1`、`noise_band_floor=1.0`、该面 `noise_band_adjacent_block_max` 同为 **1.0**，而 `wrong_top1_rate=0.5` ⇒ J-N4-2 要求"降幅 > 1.0"，**算术上不可满足**；`n=10`⇒0.5、`n=20`⇒0.25（`block_size=4`，同夹具实测），要在 0.05 量级的差异上判收益需 `size ≥ 20` 即 **n ≥ 100 条配对查询** | 高（**它让 N4 的第一份判读要等面跑完才发现"判据无法成立"**＝与 DEBT-G53（τ 冻在整场够不到的阈）、DEBT-G56（守卫臂形状与基线不同源）同一族，缺陷类别都是**冻进判据的量没有先算它的可达范围**。本条把代价从"跑完面"提前到"落盘前"，靠的是先写仪器再跑一次秒级冒烟——这条次序本身值得留成规矩） | ①PLAN-N4-02 升版：§2 的噪声带条款换成"带下界 `1/(n//5)` 由面自述"＋一条**配对查询下限**（n ≥ 100 才允许判 J-N4-2；不足即判"尺太粗、整件不判"，不许当场换统计量），旧件 N4-01 不删、索引标 §2 该条作废。②仪器的 `--out-report` 必须同时出版 `block_size`／`noise_band_floor`／`paired_queries`，使"带太宽"成为件内可见事实（**已做**，本轮随本条登记一起落地并配 2 条测：8 条⇒下界 1.0、20 条⇒下界 0.25）。③通用规矩：凡沿用"相邻段最大跳幅"这类自取噪声带的新判据，落盘前必须先算该母量**在自己的取值形态**（连续值 or 0/1 or 计数）下的带下界，并把那个下界写进预注册 | **登记未修**（②已做；①的升版件与③写进预注册模板在下一格；证据＝CLI 冒烟 rc=0、`report.json` 1,502 B、`queries_sha256` 前缀 `810037472c4ccc71`、契约测 10 passed／1.47s，台账 08 ㊵-525）·**〔2026-10-08 ㊵-526：修法① 已落〕** [PLAN-N4-02](../../reference/PLAN-N4-02_band_floor_prereg_20261008.md) 同日落盘（零跑）——§2 冻"判读只用面内实测带、下界只作自检"＋`paired_min=100`、§3 加**出口④ `ruler_too_coarse`**（样本不足时既不判收益也不判否证）、§4 加 **G-N4-5**（四键缺一即 rc=2）。升版过程中多出一条本行原先没有算到的事实：**实测带不随 n 收窄**（`paired=10/20/100` 三档同测 0.5），原因是冒烟夹具"前半全对、后半全错"的构造序 ⇒ **带是被查询顺序造出来的**，故 N4-02 冻死"顺序由 `queries_sha256` 钉死，比带必须同 sha"。四档实跑 rc 全 0：`block_size` 1/2/4/20、下界 1.0/0.5/0.25/0.05。③写进预注册模板仍未做；**由此 N4 的队首前移**：`paired>=100` 的带真值题集不存在（09:112 的材料只有三枚 consolidated 小文件＋16 条 workbench 记录），造题集＋G-N4-2 机检排在跑面批文之前。 |
| DEBT-G60 | **我刚落的预注册把"强制真前驱"的落点指到了错误的行：探针不是在 `:1333` 的回路证据 spy 里决定前驱，而是从 v20 起自己驱动生成环（`probe:1473` 注释原话"这台仪器自己驱动生成环（不走 `Taiji.generate()`）"）——能力主张成立，但指认的锚点错了，实施量也从"加一枚旗标"变成"在自驱动环里换喂入字节的来源"**（2026-10-08，PLAN-N1-03 落盘后下一格去实施 A 档时当场发现）：`grep -n "\.observe(" probe_taiji_a30_stop_failure.py` 在正文里**零命中**（只有 304/1594/1825 三处注释提到环体的 break-then-observe），`prev_byte=kwargs.get("prev_byte")` 在 `:1333` 是探针给 `copy_circuit.evidence` 挂的 spy 转发点；而回放环 `:1568-1593` 的 `byte = int(record["symbol"])` 取的是**已生成序列**的字节，`basis` 取上一条 record 的概率 ⇒ 强制档必须在自驱动环里把喂入换成真值字节，不是在 spy 里覆盖参数 | 中（**它会让实施者照预注册去改错的那一行**：`[写进交接的断言必须真跑过]` 同族——本条错在我自己落盘 5 分钟后的同一格才发现，代价是一格返工而不是一个坏结论；连带 §4 的 A 档估量"零算力、加旗标即可"偏低：要在探针里新增一条驱动路径并自证"除喂入来源外逐字不变"） | ①PLAN-N1-03 的 §0 已加日期更正（更正的是**锚点指认**，§2 测量定义、§3 判据三条、§5 守卫四条**一字未改**，沿用 PLAN-N3-04 §4bis"作废的是面的形状、判据不动"那一形状）。②实施 A 档时的真落点＝探针自驱动生成环那一段（`probe:1473` 注释所指），并必须新增一条守卫：两臂除喂入来源外**逐字相同**（含 `basis` 的来源也要一致——若强制档仍拿自链的概率帧当 basis，那就不叫 teacher forcing 而是混合臂）。③把"锚点要指到决定喂入的那一行，不能指到转发 spy"写进预注册模板的锚点检查（与 DEBT-G57/G59 同族：冻进判据/交接的东西要先算它可达、指它要先跑到） | **登记未修**（本轮只更正锚点＋立新债：A 档实施在下一格，落点按本条②；判读未受影响——N1-03 一条判据都还没判） |
| DEBT-G61 | **面里那一列 `decision_consecutive_pressure_steps` 与 `required_pressure_steps` 的关系从没对表过：实测最大值只有 2，而 `should_propose` 却有 8/6 次为真（`required_pressure_steps=3`）⇒ 这一列到底记的是判定前还是判定后的计数，无人知道**（2026-10-08，N5 取证那一格只读数买出来）：读数件 `reports/taiji_n5_growth_proposal_evidence_20261008.json` 逐面给了 `max_consecutive_pressure_steps=2`、`should_propose_true=8`（β 链 τ=0.65）与 `=6`（circuit 链 τ=0.66），而 `policy.required_pressure_steps=3`；同件里 τ=0.70 的两张面是 `0/0`。产品侧的合取闸在 `taiji/adaptive_residual_growth.py:437-444`，计数更新在 `:422-435`（**先更新后比较**是当时的读码结论，但没人拿这一列逐行验过） | 中（**它挡的是下一类结论**：谁想用"连续段长度"去解释一次提议（例如"提议发生在第 N 步之后"）就会拿一列语义不明的数当证据——正是 `[取值层＝证据所在层]` 与 `[现算的数才配进结论]` 那两条纪律防的形状。本条不影响已发表的判级：N3-06/G59 那套锚点验的是**逐行六道合取**与 `decision_reasons` 的一致性（两支都 0 失配），不是这一列） | ①只读对表一次：用现成的 `decision_digest`/`pressure_digest` 与 `:422-435` 的更新顺序，写一条逐行断言——对每个 `should_propose=true` 的行，检查该行的 `decision_consecutive_pressure_steps` 与"该行之前连续满足 `pressure_ema ≥ minimum_pressure` 的行数"是否满足一个**固定且可解释**的偏移（0 或 1），把结论写成"这一列记的是判定前/后"的一句判读，并钉进测；②把这一列的语义写进面头自述（`consecutive_steps_column_semantics: "pre_increment"` 之类），别让人从数反推；③在语义写明之前，**任何判读不许引用这一列做"连续 N 步"的主张** | **登记未修**（本轮只取证与立债；①②是零算力的只读格，排在两臂判读之后）·**〔2026-10-08 ㊵-534：① 已被实试并判为"只读对表不足以定语义"，但买到一条决定性子事实〕** 三种代理全部逐行对表（都不重抄产品算式，只用面里已披露的列）：**A**＝压强项过阈（取法＝`pressure_below_threshold` 不在 `decision_reasons` 里）在 β 面给出 pre 匹配 **3,947**／post **40**／不合 **12**，circuit 面 **3,953/33/13**；**B**＝A 再加持久项、**C**＝整条合取（`decision_should_propose`），同为 β **3,969/5/25**、circuit **3,976/1/22**；默认 τ 那张面三种代理都 **3,999/0/0**——因全场不过阈而**退化、不具判别力**（这条要先说清，否则那张面会被当成"语义已验证"）。⇒ **没有任何一种代理能逐行复现这一列**，所以①设想的那条"只读对表就能定语义"的路径被否证：语义要么去读产品计数处的更新/清零顺序，要么按②补一枚自述键。·**决定性子事实**：全部 **8（β）／6（circuit）个提议行上 `decision_consecutive_pressure_steps` 恒为 0** ⇒ 我先前"max=2 却有 8 次提议"那个表面矛盾并不矛盾，而是**这一列在提议行被清零**；**用这一列去推"连续 3 步所以提议"会直接推错**——③那条禁令由此从预防升级为实测依据。·A 最接近（不合 12/13 ≈ 提议次数 8/6 加上清零边界）提示候选语义＝"pre 计数 ＋ 提议后清零"，但这**只算候选解释，不作为结论出版**。·**〔2026-10-09 ㊵-582：② 已落、③ 改判为"可解释但不许用来主张连续 N 步"〕** 先读码定序（`taiji/adaptive_residual_growth.py` `:445-447` 自增／`:448-450` 比持久项／`:456-458` 提议即清零（`if should_propose:` 在 :456、清零在 :458）／`:468-490` 才构造 decision 载荷、列写入在 `:481`）⇒ **面里那一列出版的是"清零之后"的读数**；再用真 trigger 复现（新件 `tests/taiji_native/test_n5_02_consecutive_column_semantics_contract.py`，均匀 0.9 热跑 24 步）：`max(published) == required-1`、提议行出版值全为 0、提议前一段恰为 `range(1, required)`、`required` 整段不出现、提议后回到 1（断言与 EMA 爬坡长度无关＝不重抄生成链）。负对照两面实走：断链（一步 0.0）同样出版 0 而 `should_propose=False` ⇒ "0"不可反推"刚提议"；源码钉的键在 `HEAD` 面上不存在（`head_has_key False`）⇒ 加键前必红。②的形状＝面头新增 `consecutive_steps_column_semantics: "post_increment_reset_on_proposal"`（纯增列不动旧列、v2 号未升，逐行锚点仍是 `decision_digest`；代价披露＝新旧 v2 面的**头一行**不再逐字节相同）。③据此改判：**允许解释这一列**，但"连续 N 步达成"类主张只能用 `should_propose`／`reasons`／`required_pressure_steps`，不能用这一列（它结构上取不到 `required`）。 |
| DEBT-G62 | **PLAN-N1-03 §4 把 A 档报成"零算力、探针加一枚旗标即可自办"——本轮核实这条报价不成立：产品的生成入口没有任何强制/喂真值的参数，而探针的发射面是由 `runtime.chat()` 产生的**（2026-10-08，动手实施 A 档前的读码当场查明，紧接 DEBT-G60 之后同一格）：`taiji/model.py:3190` 的 `generate(prompt, length, *, stop_at_boundary, sample, reset, use_memory, response_start, response_phase, boundary, authorization, utf8_strict, repetition_penalty, repetition_window)`——**13 个参数里没有一个是"喂入指定字节序列"**；`probe:1477` 是 `answer = runtime.chat(turn, history=…, learn=False, max_length=…, repetition_penalty=…)` ⇒ 探针并不拥有发射环（`:1473` 那条 v20 注释讲的是**回路证据窗口的复位**归调用方管，不是"探针自己驱动发射环"，我在 G60 里把它读宽了）；产品里唯一的 teacher-forced 机器在 `taiji/language_alignment.py:1710`／`:2189`（`_target_pass`），那是**对齐/巩固那条链**，不能代答表层读出链 | 高（**它决定 N1-03 的 B 档到底能不能跑**：三条路各有代价——①给 `Taiji.generate()` 加一枚强制参数＝**产品码改动 ⇒ 回 owner**；②在探针里重写发射环＝本仓明令禁止（`别在新仪器里重抄生成链`：重抄＝换条链，三次废跑的先例）；③改用它链上的 `_target_pass`＝换链读数，会把"同一张面只差喂入"这个单变量前提破掉（G56/DEBT-G53 那一族）。⇒ 若不做①，本件唯一诚实的出路是**整件作废并另立形状**，而不是硬跑出一张不可比的面） | ①向 owner 申请**最小产品码改动**：给 `generate()` 加一个"逐字节喂入指定序列"的可选参数（默认 None＝行为逐位不变，并照本仓惯例配"挪一格必须能红"的守卫＋"默认关逐位不变"的八键对照）。②或在预注册侧改形状：把"强制真前驱"换成**不需要生成权**的读数（例如直接用现成面的 `decision_pressure`/`reasons` 做轨迹反事实，但这属 `would_have_triggered` 那一类、只在首次触发前有效）。③无论走哪条，PLAN-N1-03 §4 的 A 档报价已在原件里就地标注为错（判据 §3、守卫 §5、测量定义 §2 一字未改） | **登记未修**（本轮只报价与更正；①属产品码改动需 owner 批，②要重开一件。证据：`model.py:3190` 的参数清单、`probe:1477` 的 `runtime.chat(` 调用、`language_alignment.py:1710`／`:2189` 的 teacher-forced target pass，台账 08 ㊵-536） |
| DEBT-G63 | **训练器的收尾件把"窗口已归零"出版成 `online_accuracy=0.0`／`mean_surprise=0.0`，看起来像能力读数**（2026-10-08 ㊵-538，甲臂跑完当场量到）：`output/n3a_armA/progress_exit.json`（770 B）里 `exit_reason=max_symbols_reached`、`reached_budget=True`、`budget_max_symbols=250000`、`ticks_at_exit=250000`、`base_ticks=0` 都正常，`holdout_surprise=2.321688330698063` 与 `holdout_surprise_v2=2.6456143938810643` 是现算的真数，**但同一行的 `online_accuracy=0.0`、`mean_surprise=0.0`**；成因在 `scripts/training/train_seed_corpus.py` 的收尾 `_flush`：周期性记录写出后窗口计数器归零，收尾行仍按 `window_correct / max(1, window_ticks)`、`window_surprise / max(1, window_ticks)` 计算 ⇒ **0/0 被 `max(1, …)` 护栏抹成 0.0** | 高（**零读数的害处是它会被当结论引用**：`0.0` 读起来是"这一臂精度为零"，真相是"收尾那一行没有窗口样本"；本仓已为此付过一次代价（`显示为 0.0 不等于恒为零` 那条我差点据它编出假机制）。同一件里 `history_files=None` 也让"没留档（`--keep-checkpoints off` 的正常语义）"与"没读数"不可区分） | ①收尾行的两个窗口量在 `window_ticks == 0` 时出版 `None`，并同件自述 `window_ticks_at_exit`；`max(1, …)` 护栏只保留给**周期行**（周期行的键集与取值一字不动，沿用 DEBT-G14 那套"只改收尾那一行"的形状）。②判读侧凡引用收尾行的 `online_accuracy`／`mean_surprise`，必须先验 `window_ticks > 0`，为 0 记 `not_defined`，**不许**把 0.0 当读数入任何结论。③加一条能为假的测：构造"窗口为零的收尾"⇒ 必须出版 `None`；同一条测要钉"周期行逐位不变"（防止把修法定成改全体）。④`history_files`／`history_bytes` 的 `None` 与 `0` 要语义分离。 | **登记未修**（本轮只取证入册。次序纪律：**两臂判读必须在改这台仪器之前完成**——①③动 `train_seed_corpus.py` 的取数侧、②动判读器，若在判读前改，就会改到正在被引用的那两张面上；故本条排在主线判读之后，与 G57 剩余②③、G55 两级守卫、G54①③、G48② 同档。证据：台账 08 ㊵-538、件 `output/n3a_armA/progress_exit.json`）·**〔2026-10-08 ㊵-544：修法①③ 本轮落地；② 复核确认已落（㊵-539 那次就在判读器里）；④ 未落〕** ①取数侧＝模块级 `_window_readouts(window_correct, window_surprise, window_ticks)`（`scripts/training/train_seed_corpus.py:125-137`）：`window_ticks<=0` ⇒ `(None, None)`，`_flush` 只经这一条路（`:473`），而 `window_ticks>0` 时除法式子与旧值逐位同 ⇒ 周期行形状与取值一字不动（沿用 G14 那套"新键只进收尾那一行"的纪律）。③新册 `tests/taiji_native/test_g63_window_readout_contract.py` 四支，**先证旧件上会红**：甲臂真件 `output/n3a_armA/progress_exit.json`（770 B）现读 `window_ticks=0 / online_accuracy=0.0 / mean_surprise=0.0 / holdout_surprise=2.321688330698063`——"这一行没有窗口样本"在旧代码里就是印成"精度为零"，故本册第二支（`online_accuracy is None` 与 `window_ticks==0` 同真同假）在那份字节上必红，不是"加了个断言它自然过"；修后连同 g14 那册 **28 passed／rc=0（31.67s）**，`black --check` 两文件 unchanged、`ruff check` 0 条、B/SIM 棘轮 0 条。②判读侧那条**不在本轮**：`adjudicate_taiji_n3a_scaling_probe.py:97-99` 的 `usable=[row for row in rows if window_ticks>0]`＋`lines_dropped_window_zero` 披露列自 ㊵-539 就在，本轮按"引用之前先复核"回读两臂真件确认它真的在挡——`reports/taiji_n3a_scaling_probe_20261008.json` 里 `arm_A`／`arm_B` 各 `lines_dropped_window_zero=1`（挡掉的正是上面那行），且 §8.7 五项里唯一为 `False` 的是 `sequence_length`。④（`history_files`／`history_bytes` 的 `None` 与 `0` 语义分离）**未落**，是本条里最低的一档。同轮撞出一条**躺在仓里的红**并登记为 DEBT-G65（收尾行键集守卫没跟着㊵-513 的三轴自述放宽）。 |
| DEBT-G64 | **一条已冻判据把对照锚钉在了另一条链/另一份语料上，使它的合取项在甲臂这档上结构不可越**（2026-10-08，新判读器 `adjudicate_taiji_n3a_scaling_probe.py` 第一次对甲臂真件出数时量出来）：[PLAN-N3-02](../../reference/PLAN-N3-02_scaling_probe_prereg_20261007.md) §2 的 J-N3a 第二合取项写"acc 末段段均值 > 本件 §0 两条基线末段均值中较大者 +0.02"，而 §0 那两条基线是 **`output/a31_chunked_self`（语料档）与 `output/a31_onpolicy_20261003`（自答档）**，取的是 **2M 档在各自语料上**的读数；甲臂按 §4quater 冻的命令吃 `data/simple_zh/dialogue_extended_clean.jsonl`（另一份语料），实测五段 acc 均值 `0.280905 / 0.31394 / 0.3183 / 0.33464 / 0.352` ⇒ **斜率 +0.071095 > 自取噪声带 0.033035（第一项达线）**，但末段 0.352 对锚 `0.594120 + 0.02` 差 **−0.262**，第二项**不是"没做到"而是"在这档剂量上做不到"**（0.594120 是自答链的数，两份语料的可达区间本就不同）。同件 `holdout_surprise_v2` 末段−首段 = **−0.483725**（在改善）⇒ J-N3a-保达线 | 高（**它让"×10 平台解除与否"这件事只能由第一项判，而冻好的判据写着两条合取**——若照字面执行，任何人跑到甲臂都会读到"不成立"，并把一个**口径缺陷**误报成**容量结论**。这与 DEBT-G53（τ 冻在够不到的阈）／DEBT-G56（守卫臂与基线不同源）同族：**对照锚必须在臂要用的那条链上现取**） | ①判据升版（PLAN-N3-02 → 新件）：把第二合取项的锚换成**同语料同喂法的自取基线**——即"同一份 `dialogue_extended_clean.jsonl`、`--scale 2`、同 seed、同 250,000 预算"的一支作对照（这才是"×10 相对 ×2 有没有抬出平台"），旧的 0.594120 保留为**历史背景数**而不是门槛。②在升版前，本判读器已把两项**分开出版**（`slope_gt_own_band`／`last_mean_gt_baseline_plus_margin`），任何判读**不许**把第二项的 false 说成"容量不解除"。③补一条测：锚与臂不同语料时判读器必须出版 `anchor_face_mismatch` 之类的显式披露（现在是靠人读 §0 与命令比对）。④与 DEBT-G53 修法①、G56① 并件：落盘前核"判据的对照锚是否在同一链/同一份材料上现取"。 | **登记未修**（本条是判据侧缺陷，修法①要升版重冻 ⇒ 需 owner；②③是仪器侧可自办，已随本轮判读器落地②的"分开出版"，③未做。证据：`C:/Users/…/n3a_live.json`（本轮预跑，未入库，等乙臂完成一起入 `reports/`）与判读器三项布尔；台账 08 ㊵-539）·**〔2026-10-08 ㊵-540：修法① 已做成可签字的升版件〕** [PLAN-N3-12](../../reference/PLAN-N3-12_j_n3a_anchor_upgrade_prereg_20261008.md) 落盘（判据先冻、不开跑、**签字前不生效**，也不追改 N3-02 任何已冻文本）：新第二合取项的锚＝`--scale 2`、**同语料**、同 seed、同 250,000 预算、同喂法的一支对照臂末段均值＋0.02；旧锚 0.594120 降级为历史背景数（照旧出版、不参与判决）；门槛数值与合取形状逐字保留。守卫里 `G-N3g-1`（两臂 `corpus_fingerprint` 不等即 rc=2）就是本行③要的那条显式披露，`G-N3g-3` 落实 G63 的窗口为零筛除，`G-N3g-4` 禁止"新锚下不达再回退旧锚重判"。§4 的对照臂墙钟**未测不引外推**（×2 与 ×10 的每 tick 成本不成比例）。⇒ 本行状态从"登记未修"推进到**"①已成形待签字、③已随判读器与 G-N3g-1 落地一半、②④未做"**。·**〔2026-10-08 ㊵-541：③ 已按『两臂同源』的形状落地（随提交 `1d4e625c5`；台账行补在本笔，因为 `1d4e625c5` 的提交信息已引用 ㊵-541 而当时拼接脚本被一处内嵌双引号打断、台账行没写进去——那是内容与提交信息不符，按本仓纪律追一条补正提交而不是 amend）〕** `adjudicate_taiji_n3a_scaling_probe.py` 新增 `single_variable_check.corpus_fingerprints` 与 `corpus_fingerprints_equal`：两侧 `corpus_fingerprint` 不等 ⇒ 两臂 `J_N3a` 一律降为 `not_judged`、顶层出版 `verdict=arms_not_same_source` 且 rc=2，并由一条新夹具（乙臂指向另一份语料）钉住；契约测 8 passed（2.12s），甲臂真件重跑确认无乙臂时该键为 `None`（不假装相等）。本行③的另一半（锚 vs 臂不同语料的显式披露）随 PLAN-N3-12 把锚换成同语料对照臂而消解。 |
| DEBT-G65 | **trainer 收尾行的键集守卫没跟着"三轴自述"一起放宽 ⇒ `test_periodic_lines_keep_their_old_shape` 自 `25bb7fccb` 起就是红的，一直躺在仓里**（2026-10-08，跑 DEBT-G63 那册新测时同册撞出）：㊵-513 给 `scripts/training/train_seed_corpus.py` 收尾那一行加了 `stream_counters.as_dict()` 三键（`unique_documents`／`document_visits`／`mean_revisits`），当时只钉了这三键的**名字与顺序**（`tests/taiji_native/test_n3_10_unique_document_cap_contract.py:100-106`），**没跑**那册按集合相等钉收尾行键集的守卫（`tests/taiji_native/test_g14_trainer_exit_accounting.py:169`）。取证两条并立：`git show HEAD:scripts/training/train_seed_corpus.py` 第 481 行确有 `entry.update(stream_counters.as_dict())`，而 HEAD 那册的期望集里没有这三键 ⇒ 本轮 HEAD 实测 `1 failed, 26 passed`，失败项点名 `Extra items in the left set: document_visits / mean_revisits / unique_documents` | 中（**不推翻任何结论**：红的是"期望集没跟上"，不是行为坏了——周期行那一侧 `set(entries[0])==LEGACY_KEYS` 一直为真；代价是这条守卫是收尾行形状的**唯一**钉法，它长红之后下一个动收尾行的人会把"放宽"误当成自己弄坏的，或者干脆再放宽一次把野键放进去） | **已修（2026-10-08 ㊵-544，同轮）**：按 PLAN-N3-05 §2 立下的同一约定办——**先让它真红一次并留下点名**（上面那条读数就是红的那一次），再把三键加进**收尾行**的期望集，断言形状仍是**集合相等**（"冒出野键必须红"那一侧没被削弱），并补三条正面断言 `for axis in ("unique_documents","document_visits","mean_revisits"): assert axis not in entries[0]`（周期行混进三轴也当场红）。修后五册同跑 **28 passed／RC_PYTEST=0（31.67s）**。**同族纪律（写在这里给下一个动 trainer 的人）**：改收尾行形状的那一轮，收口对照必须**同跑** `test_g14_trainer_exit_accounting.py` 与 `test_n3_10_unique_document_cap_contract.py` 两册——㊵-513 那次报的"组 37 passed"是真跑过的，但跑的组不含前者，所以绿与那次改动无关。 |
| DEBT-G66 | **台账里一条"记录踩坑"的历史行把死链守卫自己点着了 ⇒ `test_project_identity.py` 的链接可解析检查长红**（2026-10-09，改写 03 队首那行后跑这册守卫时撞出）：`plans/active/roadmap/08_UPSTREAM_SYNC_PLAYBOOK.md:2607` 那枚行里写的是"初版把代码块里的 Python 当链接（\`…[i] (field_state)\` 报成死链）"，**这个被引作反面教材的示例串本身**是 `[..](..)` 形状，被同一册守卫当成真链接去解析 ⇒ `AssertionError: active plan link is missing: …08_UPSTREAM_SYNC_PLAYBOOK.md -> field_state`。取证：`git show 0892262ab` 那份 08 里 `field_state` 计数＝1（本轮开工前就在），而守卫件 `tests/seed/test_project_identity.py` 在 `0892262ab..HEAD` 一字未动 ⇒ **红早于本次改动，不是我这次改写 03 造成的**；它长红的原因是这册只在动 03 时才跑，而 08 每天在写 | 中（不推翻任何结论，但**03 的队首改写从此没有可用的守卫**——那册同时钉"活跃计划树里只允许一条执行项声明"，我要改队首就必须先让这册绿；同族先例见 08 里"把违禁词抄进禁令会自指命中"与措辞门那次） | **已修（2026-10-09 ㊵-547，同轮）**：改法只动示例串的形状、不动语义——在方括号与圆括号之间插**一个半角空格**（`…[i] (field_state)`），"方括号紧跟圆括号"那种链接形状不再被匹配；历史行的其余文字一字未改。守卫实测：改前 `1 failed, 7 passed`（失败项点名 `field_state`），改后重跑 `tests/seed/test_project_identity.py` ＋ `tests/taiji_native/test_prereg_wording_surface_visibility_contract.py` 应全绿。**留一条同族纪律（写在这里给下一个写台账的人）**：把死链形状抄进反面教材时要预先加空格，或直接改写成文字描述"方括号紧跟圆括号"，否则守卫会被自己的例子绊倒；反面教材不许靠**删掉**来修 |
| DEBT-G67 | **N2 保持仪器测出的那"1/151 字节级重合"经点名后是一条**通用短句撞车**，不是题目泄露：命中的是 CAP-0 D 维 `D08`，条目文本只有 7 字（`今天天气不错。`），它当然会原样出现在 108 MB 的对话语料里**（2026-09-26…2026-10-09，㊵-556 跑出 `windows_found_in_corpus=1` 后我手工点名复算定位）：`adjudicate_taiji_n2_04_retention_pair.py` 判 `retention_not_disjoint`／rc=2 的行为**是对的**（有交集就不许判），但它现在**只能报个数、不能报是哪一条**——于是"N2 乙档被一条 7 字通用句卡死"这件事必须由人手工复算才能看出来（我刚才就是用一遍 `data/simple_zh/dialogue_extended_clean.jsonl` 全文读入的探针点的名） | 中-高（**直接挡主线**：PLAN-N2-04 §2 的分离机检是 B 档的硬前置，这条不解决就整档不判；而它的真实内容不是"保持集被泄露"，是**尺子对短通用条目过敏感**。登记另一条口径风险：D08 属于 2026-09-15 就冻的清单，动它＝改判据面，要走升版而不是我顺手删） | **已修〔2026-10-09 ㊵-632 更正；原判「两条不等价、我不选」已不成立〕**（证据：修法①与 (乙-1) 口径都在码里（:291/:310/:324/:325），两份入库裁定件带值，守卫测 ㊵-631 补齐（该册 14 passed））（修法两条不等价，交 owner 认口径前我不选）：① **仪器补点名**（我可自办）：让 `G_N2c_4` 出版命中的 `dimension/id/chars` 而不是只报计数——落点是复用 `_row_texts()` 时带上行身份，把 `windows` 变成 `(label, bytes)` 序列并让 `_disjointness` 的逐窗命中能回映到行（**不许**重抄一份扫描链，`dont-reimplement-the-chain` 那条）；② **口径**（要 owner）：给"重合"设最小可判长度（例如条目短于 `NGRAM_BYTES=24` 就不参与包含判定，单列 `skipped_short_rows` 披露），或把 D08 换成一条足够独特的文本（＝动冻结清单，走 PLAN 升版＋索引标作废）。取证：探针现读 `windows=151 / exact-hits=1 / D08 chars=7`；仪器侧现读 `retention_windows=151 / windows_found_in_corpus=1 / hit_rate=0.006623 / window_bytes=24`（件 `C:/Users/23747/AppData/Local/Temp/n556b_pair.json`，落盘件已弃，读数抄自终端）。台账 08 ㊵-557。 · **〔2026-10-09 ㊵-631：状态更正——修法①与 (乙-1) 口径其实**早已落地**，本行原来谎报了开合〕** 取证两条并立：① 仪器侧现读 `adjudicate_taiji_n2_04_retention_pair.py:310/318/326` 有 `matched_rows`（聚合报出交集时**逐窗复算并点名行号**，走的是同一个 `N3A._disjointness`，没重抄扫描链），`:291/:324/:325` 有 (乙-1) 的短条目豁免与 `skipped_short_rows`／`skipped_short_detail` 披露；② 已入库的两份裁定件里这些字段**带值**：`taiji_n2_04_retention_pair_20261009.json` 与 `..._n2c1_...json` 都是 `status=measured／retention_windows=116／windows_found_in_corpus=0／skipped_short_rows=35／matched_rows=[]／manifest_shape=dimensions_items`。**当时唯一缺的是测**：`grep tests` 对 `matched_rows`／`skipped_short` 命中 0 ⇒ 能力在码里但没有守卫。本轮补两支（`tests/taiji_native/test_n2_04_retention_pair_contract.py`，14 passed）：一支种一条真泄露行 ⇒ `matched_rows == ['D99']` 且只点名它；一支放一条 21 字节的通用短句、并把它**原样写进巩固语料** ⇒ 整件仍 `retention_holds`／rc=0，但该句必须出现在 `skipped_short_detail`（＝不许静默剔除，把「我没检」说成「检了没重合」）。余下未结的只有**历史那一次**：D08 事件本身（㊵-556/557）已按 (乙-1) 口径结案，无待修项 ⇒ 本行状态改判为已修。 |
| DEBT-G68 | **N5 的四层循环在这条链上结构不可启动：`--developmental-bridge-gate` 缺省为 `None` ⇒ 落 `0.0` 且连 setter 都不调（`scripts/training/train_seed_corpus.py:677-683`），bridge 不放活动 ⇒ `activity_saturation` 恒 0（定义在 `taiji/adaptive_residual_bridge.py:101-106`＝"活动绝对值 ≥ `dynamics.target_activity` 的单元占比"），而 `should_propose` 是**六道 EMA 合取**（`taiji/adaptive_residual_growth.py:437-444`，含 `minimum_activity_saturation=0.40`）⇒ 提议永不为真、影子永不物化、Seed 层转发与消费全链**在这条链上不可测**（2026-10-09，跑 ㊵-586 的 C 之后按面头自取的六项阈值对表实测：10,000 行里 `decision_pressure` max=0.5494 对 τ=0.65（0/9999 过阈）、`decision_activity_saturation_ema` max=**0.0000** 对 0.40（0/9999），其余四项 9992–9999/9999 过阈） | 高（**直接挡 N5**：PLAN-N5-01 要的"候选生长→影子学习→贡献/保持验收→准入"整条循环，前置是"确有提议发生"；这债不结就只能拿一张永不提议的面去说"影子无收益"＝把仪器默认位读成能力结论。**开门即证**：同一命令加 `--developmental-bridge-gate 1`（面头自述 `bridge_gate_requested=1.0`／`bridge_gate_actual=1.0`），跑到第 **264** 行就出**第一次提议**，同面 `decision_activity_saturation_ema` max=**0.4163**、`decision_pressure` max=**0.6837** ⇒ 合取并非不可满足，此前"10001 行 0 提议"是**默认位的结构性后果**而不是"没训到"。与 owner 当年点破的"bridge 在 gate=0 提前退出"同源） | ①先定默认位（**要 owner**）：N5 正式跑的冻结命令里是否写死 `--developmental-bridge-gate 1`，还是把产品默认改成开门（后者＝动产品默认位，半径更大、要钉全量调用点）。②我可自办：把这条依赖写成响亮前置——给了 `--n5-shadow` 而 gate 为 `None` 时直接拒绝起跑（同 `--growth-min-pressure` 须配 `--pressure-record` 那条已落地的例子，`train_seed_corpus.py` 的 CLI 校验段），否则下一个拿到的是一张永不提议的面并把它读成"无效应"。③判读口径：任何"影子无收益/无提议"的结论必须点名 `bridge_gate_actual`，gate=0 的面上该结论不成立；面头已自述这两枚键（requested/actual 成对），不需要新仪器 | **部分已修〔2026-10-09 ㊵-632〕**（证据：②CLI 双守卫（test_n5_04）、④当场消费（c9a974fbc 形状）与③判读口径（判读器按 gate 分 not_powered/shadow_inert）都在；只剩①默认位，且 owner 第十二次弹窗已裁「只进命令、不改产品默认」⇒ ①按裁定是**不做**而不是未修）（本轮立债并留下 C/D 两侧对照读数；②属"先修仓库"那一档、零算力，排在 owner 裁 ① 之后） · **〔2026-10-09 ㊵-588：修法② 已落〕** CLI 双守卫（`--n5-shadow` 缺 `--pressure-record`、或缺 gate／gate `<=0` 各自 `parser.error`）＋新测 `tests/taiji_native/test_n5_04_shadow_cli_guards_contract.py` 4 passed，含 gate=1.0 时本守卫必须放过的正例、与显式给 0 同样拒绝的判别式。**再补一条决定性实测**：gate=1.0 的 D 跑满 60k 符号、`RC_TRAIN=0`、`reached_budget=True`、面 265 行含 **1 次提议（下标 263）**，但收尾信封的 `n5_shadow` 键**为 `null`** ⇒ 影子从未物化——消费点住在符号流上轮询 `last_decision`，而提议只在下一个观测槽之前可见，两频不对齐⇒概率相遇。**修法④（新增，零算力）**＝把消费点改为在压强观测当场消费（`_record_pressure` 已持有本次 decision），让提议发生⇒同槽物化成为结构保证。①（默认位）仍归 owner。 |
| DEBT-G69 | **训练链里的影子从不通电：`AdaptiveResidualShadow._gate` 初值 0.0（`taiji/adaptive_residual_shadow.py:118`），只有 `set_gate()`（`:261-265`）会改它；而 `forward()`（`:443`）与 `learn()`（`:506`）都带同一道早退 `if self._gate == 0.0 or self._lesioned: return` ⇒ gate 为 0 时影子**既不贡献也不学习**。全仓 `set_gate` 的调用点只有 `adaptive_residual_shadow.py:945`（`from_checkpoint` 回灌）、`eval_taiji_m4v2_r4_shadow.py:195`、`eval_taiji_m4v2_r5_conditional_canary.py:96` 三处，`taiji/model.py:867` 开的是 **bridge** 的闸不是 shadow 的，而 `train_seed_corpus.py` 里一个 `shadow.set_gate` 都没有 ⇒ 训练链即使把影子挂上并前馈（㊵-585／㊵-589／㊵-593 已打通），每步仍走直通分支**（2026-10-09 F 跑实测：信封 `n5_shadow` 在场、`unit_count=97`、`gate=0.0`、`candidate_id=r4-candidate:34f3c679…`、`RC_TRAIN=0`、60k 符号跑满）** | 高（**直接决定 N5 下一格的可解释性**：PLAN-N5-01 要的「影子学习→贡献/保持验收」在 gate=0 时是**零步学习**，任何「影子无收益」的读数都会把「没通电」误读成「没效果」；这也解释 M4v2 那批影子证据为何全在评测脚本里——那里显式 `set_gate(1.0)`） | ①训练侧补一枚显式通电旋钮（`--n5-shadow-gate`，缺省关，物化后调 `shadow.set_gate(value)`）＝纯 `scripts/training/` 一格，不动产品码；②通电与否必须进面/信封自述（`shadow_gate`、`_last_candidate_gate`、`candidate_utility`、`counterfactual_utility`），缺任一按 fail-closed 记 `ran_not_measured`；③判读口径：`gate=0` 的跑只许写「物化成功、零步学习」，不许写「影子无效应」；④通电值的判据要先冻（取 1.0 与评测脚本同源，另档需新批文） | **已修〔2026-10-09 ㊵-632 更正〕**（证据：①--n5-shadow-gate 旋钮、②信封/面自述（含 ㊵-618 三枚在场计数）、③gate=0 只许写「零步学习」的口径、④通电值判据先冻（PLAN-N5-02/03 的 +0.02 线与 ruler_usable 公式）四条全落；配对直证：治疗臂 11735/11735 对对照臂 0/0（㊵-618、㊵-628））（本行由 F 跑读数与代码行号立据；①②是我可自办的零算力仪器格，排在 N4 反例探针之后或并行） |
| DEBT-G70 | **台账里的「冻结命令」是散文，不是可执行物——照抄就撞 argparse（2026-10-09 ㊵-616 实测）**：㊵-593 那份 N5 冻结命令把 `--pressure-record` 写成**裸旗标**，而它的真实签名是`type=Path`（`train_seed_corpus.py:1126-1132`，要一个 JSONL 路径）⇒ 两臂都在起跑前被`parser.error: argument --pressure-record: expected one argument` 拒绝（`RC_ARM_ON=2`／`RC_ARM_OFF=2`，**零训练发生、零权重改动**）。同日第二形：`--smoke` 复用旧靶被覆写守卫响亮拒绝（`it is not this run's --resume source …`），⇒ 冒烟必须给**新** `--checkpoint` 路径。根因不是笔误而是**形状**：命令面住在叙述里，而叙述没有任何东西钉它与代码签名一致 | 中（不推翻任何结论，但**复现成本被抬到「必须重读源码」级**：这次是我重读了参数面才发现，下一个人可能直接把 rc=2 读成「这条链跑不起来」并写下假否证；同族已记 「交接清单里过期的是处方／命令，比过期结论更贵」） | ①**我可自办（零算力，首选）**：让训练器把**完整 argv 与解析后的关键取值**写进面头／信封（与 `bridge_gate_requested`／`bridge_gate_actual` 成对自述同一形状），这样「冻结命令」＝某张面件的自述字段，永不与代码脱钩；不许新造仪器重抄一遍参数表（`dont-reimplement-the-chain`）；②**口径**：台账里的命令段一律注明「取自哪张面件的自述」或「最后一次逐旗标实测的日期」，两者都没有就标「未验」；③已就地做的最小止血：PLAN-N5-04 件尾更正节与 `HANDOFF_N_MAINLINE_20261009.md` 已写明带参事实与 `PYTHONUTF8=1` 要求 | **部分已修〔2026-10-09 ㊵-632〕**（证据：修法①已落（argv 自述进面头与信封，test_n5_10 含真跑回读 4 passed）；②③是**写作口径**（引用命令时须注明取自哪张件/哪次实测），留下一格在动 08/09 时执行）（①排下一格；本轮已把两处实测事实写进预注册更正节，不改产品默认位） · **〔2026-10-09 ㊵-626：修法① 已落〕** 训练器把**本次调用的 argv 原样登记两处**——压强面头（`kind=face` 那行，紧跟 `format` 之前）与落盘信封（`envelope["command_surface"]`，填充仍在 `atomic_save` **之前**）。刻意**只登记 argv**：不重建参数表（重建＝再造一条会腐化的生成链），不放时钟（面头要能逐字对比）。新册 `tests/taiji_native/test_n5_10_command_surface_self_report_contract.py` 4 passed，其中一支是**真跑到落盘再回读**（200 符号、全新写靶）：信封与面头的 argv 与本次调用逐字相同（这里我也先错过一次：子进程 `sys.argv[0]` 是脚本路径而非我传入的解释器，比较要取 `argv[1:]`）。回归面实测：`-k “n5 or n3 or pressure or trainer or face”` 286 passed、`test_g14_trainer_exit_accounting.py` 7 passed（收尾行键集守卫没被新键撞红）、`tests/seed` 182 passed。余下：②③（台账命令段一律注明取自哪张面件的自述）属写作口径，随下一次动 08/09 时执行。 |
| DEBT-G71 | **N5 保持侧（`J-N5b-5`）在产品层被两道响拒绝夹死：影子链拒绝位置输入，而全部可 resume 的基件都带位置输入（2026-10-09 全集实测）**：`--developmental-fast-slow` 走 `Taiji.migrate_f1_to_developmental_synapses`（`taiji/model.py:1284`），它第一件事就是调 `_reject_position_input_without_learning_path`（`:1131`）⇒ `readout_utf8_position_input=True` 时**响亮拒绝**；反过来加 `--no-readout-position` 去 resume，`Seed.restore`（`seed/model.py:240`）以 `checkpoint configuration does not match Seed` 拒。取证是**全集**而不是样本：`checkpoints/*.pt` 实测 13 枚，**每一枚** `readout_utf8_position_input=true`（含 `seed_a31self_with_circuit.pt`、`seed_beta_with_circuit.pt`、`seed_n2_{mother,candidate,rollback}_20261008.pt` 与 `seed_n2b_*{,_n2c1}.pt`，自述 tick 含 0／92／273）⇒ 仓里**不存在**「无位置输入且可 resume」的基件 | 高（**直接挡 N5 收口**：PLAN-N5-04 换底的动机就是让保持判据 `after < before` 能为 false，而这条不结就没有任何一张「底子非零」的 before 面 ⇒ `J-N5b-5`／`J-N5b-6` 继续 `not_adjudicable`；注意它挡的是**取数形状**，不是影子有效性——学习侧母量已判完） | 三条例路**不等价，交 owner 择一前我不选**：①**甲＝改产品**，把位置输入接进发育 F1 学习通路（`:1131` 那道拒绝就是它的缺位声明；半径最大——所有带位置输入的旧读数都要重标，且要钉全量调用点）；②**乙＝先造基件**，训练一枚「无位置输入＋七列底子非零」的检查点再续训双臂（花训练机时，且要先定价「底子非零」需要多少 tick）；③**丙＝换形状**，用同臂自身时间轴的保号存档 `checkpoint_<tick>.pt`（住 `*.pt.history/`）作 before——但它测的是「同一次通电内的漂移」而非「通电前后的代价」，与 09 §2 N5 原问不同 ⇒ 用丙必须**另立判据**，不许套 `J-N5d-1`；④无论选哪条，先把「七列的 before 面如何生成」写进预注册（判据先冻） | **甲已落地（2026-10-09 ㊵-636，PLAN-N5-05 J-N5e-1／2／3 已判；J-N5e-4 续训双臂另批）**（三条例路已于弹窗 #15 呈报，owner 当时裁「先办计数器」，甲／乙／丙仍待择一。台账 08 ㊵-616／㊵-618，预注册见 PLAN-N5-04 §0 F-6 与件尾两节带日期更正） · **〔2026-10-09 ㊵-636：甲已落地，本行的"取数形状不可得"这一半结清〕** 拒绝点已从 `taiji/model.py` 撤除（旧闸名与旧闸话在三个产品文件里都不许出现，钉子＝`test_n5_11_position_wiring_call_sites_contract.py` 第 1 支），换成"重放事件必须带位置类"的新闸；`J-N5e-1` 三档 HEAD 对表逐位相同、`J-N5e-2` 出 `position_path_delta=1.9232144355773926 ∧ n_changed_units=1028`（反支同为 0）、`J-N5e-3` 默认位与参数量未动。**仍未结的是 `J-N5e-4`**：续训双臂要 owner 再批（改权重），批下来才有 `j_n5b_5` 的 before／after 对；`f-2` 里那道 `Seed.restore` 的配置校验**不是**位置输入挡的，它只比 config digest，开启位置输入的臂必须两枚 config 一致。另摊出一条新债 **DEBT-G76**（位置列不进发育叠加层的可逆层）。 |
| DEBT-G72 | **我把一枚恒真读数写进了冻结判据：`shadow_materialized` 只能是真（2026-10-09 实施同轮自查）**：PLAN-N5-04 G-N5d-4 原要求 `n5_shadow` 块带 `shadow_materialized`，而训练器里那段填充整块住在 `if adaptive_shadow is not None:` 分支内 ⇒ 键被写出时该值**恒为 true**，一条不能为 false 的读数不构成证据（`guard-must-be-able-to-fail` 的判据版）。同轮另一形：第一版把两枚 `@property` 插在 `@torch.no_grad()` 与 `set_gate` 之间，装饰器被劈到属性上 ⇒ `lint` 三关全放行、直到真跑存盘时 `int(method)` 才 `TypeError`（两臂各崩一次，件未落盘、零假读数） | 中（**挡的是「证据能不能被引用」**：恒真键若随判读器入库，下一格会把它读成「物化已自证」；装饰器那一形更贵——它能活着过 lint，只在最贵的时刻（真训练收尾）炸） | ①恒真键已**撤换**为三枚可分且能为 0 的在场读数：`shadow_forward_hits`／`shadow_learn_hits`／`shadow_branch_hits`，自增只写在通电闸**之后**（闸关或被损时不涨）；配对冒烟实测治疗臂 `11735／11735／23470` 对对照臂同块内**三枚全 0** ⇒ 读数确实双向；②装饰器形状已**升级成反向守卫**（`_decorator_names(new_props)==['property']` 与邻位 `set_gate` 保留 `torch.no_grad()` 各一条，AST 里 `@torch.no_grad()` 是 Call 节点、期望值要按整串写）；③**长期修法**：新判据入库前先问「它能为 false 吗、以及造它的那条路径能不能为 false」，并要求任何「属性／成员插入」在 lint 之前先做一次**导入面探针**（`getattr_static` 的类型） | **已修（2026-10-09 ㊵-618，同轮）**：键名更正写进 PLAN-N5-04 件尾（带日期，判据语义未动）＋新测 `tests/taiji_native/test_n5_08_shadow_presence_counter_contract.py` 7 passed 在册 |
| DEBT-G73 | **N4 产品档寻址尺的 θ_small 落进几何盲区：扰动强度没按「这把尺的动态范围」定价（2026-10-09，已用升版解）**：PLAN-N4-06 冻的第一版角度对是 `(0.35, 0.70)`，实跑出 `ruler_unusable`——小角那一档的 `wrong_top1` 与大角档**同带不动**（相邻块噪声带吞掉了差异），判读器按已冻公式正确拒判；根因是「扰动角度」这个自变量是**凭直觉挑的**，而查询/候选向量在该基底上的夹角本身就在 0.35 rad 量级附近，小扰动落在分辨盲区内 ⇒ 尺没有动态范围（`frozen-criterion-stays-when-secondary-moves` 里那条「还要问这把尺有没有动态范围」的具体命中） | 中（**属测量设计债，不属产品缺陷**：它差点让「寻址不随角度变」被当成能力否证发表；登记另一条口径风险——升版只许换几何输入，判据公式一字不动，否则「尺可用」与「结论」会被一起重造） | ①已做：**升版 PLAN-N4-07 只替换角度对** `(0.35,0.70)`→`(0.70,1.05)`，公式与失败出口未动，旧篇与旧 `ruler_unusable` 裁定**保留并标取代关系**（不删旧件）；未改动的判读器复跑出 `angle_sensitive`（`wrong_top1 0.2916666666666667 → 0.975`，`delta=0.6833333333333333` 对 `line_required=0.20833333333333334`、两档噪声带 0.2083／0.0833 均 >0）；②**可复用的纪律（写在这里给下一个定扰动强度的人）**：冻判据前先做一次**秒级动态范围探针**——在最小档与最大档各取一次读数，若两档差 ≤ 噪声带就当场不冻，先把自变量拉开再说；③不许用「放宽 `ruler_usable`」来救一张没有动态范围的档 | **已定性并已有解（2026-10-09 ㊵-612）**：登记此行的目的是把「扰动强度要按尺的动态范围定价」变成可检索条目，而不是留一条待修代码债；预注册见 PLAN-N4-06／PLAN-N4-07，判读件 `reports/taiji_n4_product_tier_ruler_verdict_v2_20261009.json` 在册 |
| DEBT-G74 | **措辞门从没被按「全集」跑过 ⇒ 42 份预注册／判读件里有 10 份一直不过门（2026-10-09 首次全集审计实测）**：本轮把 `scripts/training/audit_taiji_prereg_exit_wording.py` 对 `plans/reference/PLAN-*.md` 逐份跑一遍，非零 rc 共 10 份，分两族：〔乙族＝`invisible_criterion_surface`，rc=2，6 份〕`PLAN-B-03_cortex-split`（`criterion_lines=1`、`structural=0`）、`PLAN-N2-02_ADJUDICATION`（`criterion_lines=15` 却 `structural=0`）、`PLAN-N3-01_r4_hooks`（`crit=4`）、`PLAN-N3-03_tau_definition`（`crit=2`）、`PLAN-N3-06_ADJUDICATION`（`crit=1`）、**我自己这轮的 `PLAN-N5-03_ruler_usable_formula`（`crit=8` 而 `structural=0`）**；〔甲族＝`ambiguous_exit_wording`，rc=1，4 份〕`PLAN-N1-00_s5`（`unpinned=2`）、`PLAN-N1-01_ADJUDICATION`（`unpinned=2`）、`PLAN-N1-01_s1`（`unpinned=1`）、`PLAN-N3-08_face_self_report`（`unpinned=1`）⇒ 未钉住的含混措辞合计 **6 行**。**为什么这既是文档债也是门债**：门只在「本轮动过的那份」上跑过，所以这批件从入库起就没被检过；而乙族里两份 `criterion_lines` 明明过线（15／8）却仍 `structural=0` ⇒ 那一部分更像**门的取词面太窄**（同一族的先例：门靠标记词挑判据句，换个标题能整份隐形），不能全算文档缺陷 | 中-高（**挡的是判据可信度**：措辞门是 DEBT-G46 立下来的「出口写成数值合取式」那道防线，10/42 不过门意味着一部分预注册的失败出口仍可能写得既能说得通又不可判；同时它会误导下一个动门的人——他会以为自己改的那份是唯一的红） | 分三步，顺序不可倒（先分清是谁的错再改）：①**先给门补判别力**（我可自办、零算力）：对那两份「`crit` 过线却 `structural=0`」的件做一次反例定位，把结构性判据句的取词面放宽到能看见它们，并配一条「放宽前必须为 0、放宽后必须 >0」的双向测——**不许**为了让计数好看而把 floor 调低；②**再按门修文档**（逐份，禁批量）：甲族 6 行含混措辞各补数值合取或点名判据行；乙族里 `crit=1/2/4` 那三份（`PLAN-B-03`／`PLAN-N3-01`／`PLAN-N3-03`）缺的是判据面本身，动它们＝改已冻预注册 ⇒ 只能**带日期就地补注或升版**，不许把判据补成事后解释；③**把全集跑变成门**（我可自办）：让 CI／`doc-sync` 侧有一条对 `plans/reference/PLAN-*.md` 全量跑的叶子，否则下一次还是只检动过的那份 | **①③已修（2026-10-09 ㊵-625，同轮）**：① `invisible_criterion_surface` 已改成**合取**（无判据标题 **且** 标记通道 `< floor` 才判隐形；标题缺位但已过线的出 `headingless_visible` 披露继续走正常分档），两侧测都实跑过（已过线那份转 rc=0、既无标题又薄的仍 rc=2，防合取被写成恒真）；③ 新增门叶子 `tests/taiji_native/test_prereg_wording_corpus_ratchet_contract.py`（3 passed）：把 42 份全集跑做成**棘轮**（子集判定⇒新增违例红、修掉违例不红）并钉「扫到的份数＝盘上份数」防 glob 变窄。**全集改前／改后实测**：42 份 ⇒ 改前 32 ok／6 invisible／4 ambiguous，改后 34 ok／4 invisible／4 ambiguous；转可见的正是 `criterion_lines=15`（`PLAN-N2-02_ADJUDICATION`）与 `criterion_lines=8`（`PLAN-N5-03`，本会话自己写的那份）。**②仍未修且不许凑数**：剩下 4 份 invisible（`crit=1/2/4/1`）与 4 份 ambiguous 是真内容问题，动它们＝改已冻预注册 ⇒ 只能带日期补注或升版，逐份判、不批量改词。（原立据文字保留：本行由 42 份逐份 rc 与明细读数立据；取证件目录 `C:/Users/23747/AppData/Local/Temp/wa621/`，仓外未入库。）台账 08 ㊵-622。取证件目录：`C:/Users/23747/AppData/Local/Temp/wa621/`（仓外，未入库） · **〔2026-10-09 ㊵-629：② 的一部分实为门侧盲区，已修门〕** 逐行回读证实`PLAN-N1-00` 第 24 行 `R1 ≤ **118**（基线 236 的 50%）` 同行已钉数值，旧 `NUMERIC_PIN` 因粗体标记卡在算符与数字之间认不出 ⇒ 把已钉数的判据句报成含糊。修法＝在原式里只插入 emphasis 允许集、不动原有 `\s*` 量词（第一版我把 `\s*` 写成 `\s`，反而把 `2.482967倍` 那类老写法判瞎，全集 ambiguous 由 4 涨到 5、两支契约测当场红——改判别式必须两侧都能为假才是修好）。契约测的 `FLAGGED` 期望值按新实测由 `[7, 24]` 重钉为 `[7]`（那是在钉门的瞎处，不是在钉文档）。**② 未完的 5 行分账**：`PLAN-N1-00:7`／`PLAN-N1-01_s1:35`／`PLAN-N3-08:75` 是真未钉（要带日期补数值合取）；`PLAN-N1-01_ADJUDICATION:47`（引 DEBT-G46 的四词清单）与 `:40`（引原句并登记其未冻数值）是**把违禁词抄进禁令**的自指命中 ⇒ 正解是给引用格式一个豁免口径或改写引用方式，不是删别人的判据词。四份 invisible（`crit=1/2/4/1`）一字未动。 · **〔2026-10-09 ㊵-630：② 做到自办边界〕** `PLAN-N1-01_ADJUDICATION` 的两行**自指命中**已修（一行把 DEBT-G46 的四词逐字抄进禁令、一行改为就地补同轮实测读数作定位，**不改判据**；所引数字在被改件里本来就有，`grep -c` 命中 5 处）。全集 42 份由 34 ok／4 invisible／4 ambiguous 变成 **35 ok／4 invisible／3 ambiguous**；剩余 3 行（`PLAN-N1-00:7`／`PLAN-N1-01_s1:35`／`PLAN-N3-08:75`）都在**冻结预注册的出口句**里 ⇒ 补数值＝改判据文本，只能升版或由 owner 认口径，我不自行为别人的判据补数。棘轮按子集判定，本轮 `-k` 措辞道 33 passed ⇒ 「修掉一份不红、新增一份会红」两侧都还活着。 · **〔2026-10-09 ㊵-640：② 的甲族三行结清，乙族四份仍未完且不许凑数〕** 三份出口句各改各的、都带日期且**不新增阈值**：`PLAN-N1-00:7` 补成 §2 唯一判据 J1 的 `R1 ≤ 118` 的指针；`PLAN-N1-01_s1:35` 钉成 `F1 前后差值 ≥ 0.02`（沿用 PLAN-N5-02/03 已冻改进线）`∧ F3_after ≥ 2 × F3_before`，并写明不重判 2026-10-08 那轮（当时实测 0.198 与 ×20.8 两边都满足）；`PLAN-N3-08:75` 把"足够"钉回 6 枚阈＋5 支 EMA 的在场计数并把数值与"足够"写在同一行（门的取法是**按行**）。全集复测 43 份 ⇒ **39 ok／4 invisible／0 ambiguous**；四份 `invisible`（`PLAN-B-03`／`PLAN-N3-01`／`PLAN-N3-03`／`PLAN-N3-06`，`crit=1/2/4/1`）缺的是判据面本身，owner 裁的④未授权动它们 ⇒ 继续登记未修。分母现取：`plans/reference/PLAN-*.md`＝43 枚（`git ls-files` 一致、未跟踪 0 枚），比 42 多的那枚是本线自己的 `PLAN-N5-05`。 |
| DEBT-G75 | **M5 蒸馏删档时把活动文档的引用改指墓碑，但三册契约测里硬编码的 M5 预注册路径没跟着改 ⇒ 长红（2026-10-09 归因实测）**：跑 `tests/taiji_native -k “prereg or wording”` 时 3 枚红，隔离复跑`tests/taiji_native/test_cap0_baseline_contract.py` 得 **6 failed／49 passed**；六枚红的指针全部指向两件已不存在的件（`M5_CAP0_P1_LANGUAGE_SUPERVISION_DIAGNOSIS_20260915.md` 与 `M5_P3B_ALIGNED_LANGUAGE_TRAINING_PREREGISTRATION_20260915.md`），另两册指向`M5_WP5A_B2_SELECTION_PREREGISTRATION_FROZEN_20260915.md`／`M5_B0_N2_STOP_REASON_PREREGISTRATION_FROZEN_20260915.md`。取证三条并立：这四件 `git ls-tree HEAD` 均为 **0**（HEAD 里就没有）、`git status` 对这三册**干净**（我没改过测）、删除发生在 `678e35fa2`（owner 2026-10-07 批准的 M5 族蒸馏：删 158 件、活动文档 15 处引用改指墓碑、归档历史文件不改写）⇒ **红早于本轮任何改动，不是我把门改坏的** | 中（**挡的是「收口对照能不能当门用」**：这几册是 CAP-0／P3B 契约面的守卫，长红之后任何一次整面跑都会把 6～8 枚历史红混进读数里，下一个动训练契约的人要么把红当自己弄坏的、要么干脆把这册从对照组里摘掉——两条都是把守卫拆了；另记一条口径风险：测里断言的是「那份件的 §8/§9/§10 记了某两条缺口」，这层内容**不在墓碑里**，改指墓碑等于把断言做水） | 三条不等价，**归 owner 认口径前我不选**：①**按已裁过的同族形状拆用例＋skip＋打出处**（owner 2026-10-03 对「平台挡住的断言」裁的就是这个形状：拆成平台无关半段照跑＋对被删对象的半段 `skip` 并写明删除提交与墓碑位置）——好处是零假绿且红数可解释，代价是那几枚断言从此不进门；②改指墓碑：便宜但**会把内容断言降级成路径存在断言**，我不推荐；③从 git 历史恢复那四件：与 owner 的蒸馏政策直接冲突；④无论选哪条，先把这 6＋2 枚红的名单冻进本行，别让下一次整面跑把它们当新增红 | **已修〔2026-10-09 ㊵-634，按 owner 弹窗 #16 裁的形状〕**（证据：三册合跑复算 75 passed／8 skipped／0 failed，skip 理由逐条含删除提交与墓碑路径）（本轮只归因与立册，未动任何一册测；实测计数：过滤面 3 failed／33 passed，隔离面 6 failed／49 passed，`tests/seed` 182 passed 不受影响）。台账 08 ㊵-625 |
| DEBT-G76 | **甲采器官侧写入：位置列不进发育叠加层的 fast/slow 可逆层 ⇒ 回滚与巩固不覆盖它（2026-10-09 实施时选定，已钉成测）**：`PLAN-N5-05` 的 J-N5e-3③ 给了两条支——「不新增张量」与「新增张量就走参数面三件配套事」。实施选了前者：位置列仍住在 `BytePredictiveReadout.position_weight`（随读出器 payload 存档），发育通路通过 `learn_position_column()` 写它，而不是在 `DevelopmentalSynapseBank` 里再造一对 slow/fast。⇒ 后果三条：① `consolidate_developmental_f1()` 的 fast→slow 搬运**不含**位置列；② 清 `fast_delta`（回滚发育叠加层）**不撤销**位置学习，撤销只能退到检查点级；③ bundle 的 `bank_digest`／`bundle_digest` 看不见位置学习发生（`replay` 报告里的 `changed` 因此可能为 `false` 而模型确实变了）。之所以不是静默：`replay_developmental_f1()` 同轮出版 `position_learn_steps_before/after`（G-N5e-4 那条"从抛错改成不抛必须自带证据"），且 `test_n5_11_position_wiring_call_sites_contract.py` 钉住"bundle payload 里出现不了 `position` 字样"这条边界 | 中（**挡的是"发育叠加层可逆"这句声明的适用范围**：位置输入开启时，拿着 `fast_is_zero=true` 就说"这轮学习全可撤销"是假的；另记一条选型风险——若日后要把位置列搬进 bundle，那是**新增张量**，必须连 J-N5e-3③ 的三件配套事一起做，不能只加字段） | 三条不等价，实施者按主线需要择一：①把位置列搬进 bundle（`position_slow`＋`position_fast`，前向由 bundle 出、读出器自己的那一列要显式让位以免双计），并补参数量钉子＋旧 payload 按零补＋W=0 回退连平手一起钉；②保持器官侧写入，但让 `consolidate_developmental_f1()` 与回滚路径把位置列一并记账（至少出版"本次巩固未覆盖位置列"的自述键）；③只在口径上收口——规定"发育叠加层的可逆性不含位置列"，任何引用 `fast_is_zero` 的结论必须同句点名位置列归属。**不许的写法**：拿 `bundle_digest` 未变断言"模型未变" | **登记未修（本轮已定性并钉成测，2026-10-09 ㊵-636）**：本轮选的是"承认缺口＋让它在读数里可见"，不是消除缺口；预注册见 PLAN-N5-05 §2 J-N5e-3③ 与件尾 §7 的实施读数，守卫见 `tests/taiji_native/test_n5_11_position_wiring_call_sites_contract.py`（12 passed） |
| DEBT-G77 | **CAP-0 普查的"复现封存"测把盘态变化变成常态红（2026-10-09 全道回归时归位）**：`tests/taiji_native/test_cap0_inventory_contract.py::test_a_fresh_inventory_sample_reproduces_the_sealed_one` 当场重跑 `run_inventory` 并与 `reports/taiji_cap0_inventory_a31self_convergence_20261003.json` 逐叶比较。取证三条并立：封存样本的 `checkpoint_inventory` 长度实测 **3**（`seed_a31self_with_circuit.pt`／`seed_beta.pt`／`seed_beta_with_circuit.pt`）、今天 `ls checkpoints/*.pt | wc -l` 实测 **13**（N2 家族 2026-10-08 落盘、`seed_n2b_*_n2c1` 三枚 2026-10-09 落盘）、失败断言原文是 `assert not {'checkpoint_inventory[10].bytes', ...}`（**新增字段**方向，不是缺字段）。⇒ 样本按**排序后的索引**逐叶比较，盘上多一枚档就整体位移一格，于是 `assert not added` 必红。该测自己的 docstring 已预告"改 `run_inventory` 的任何一条判据、或默认基座发生变化，这里就红"，所以这不是假红，而是**对照面过期** | 中（挡的是"每趟全道跑都带一枚可预期的红"：长红会诱使下一个动普查的人扩大屏蔽集，而"被屏蔽的叶子远小于比较面"正是这条测要守的东西；另记一条口径风险——按索引比较＝把分组变量条件在结果上，档数一变历史读数就不可配对） | 三条不等价：①按现行盘态**重封一份新样本**，旧样本与旧测**不删**、只标过期，并另立一支"新样本可复现"（与本仓"升版重跑＋索引标作废、旧件不删"的既有纪律同形）；②把仪器的比较键从索引换成**文件名**（`checkpoint_inventory[<name>].bytes` 这类路径），并配一条"档数变化不得移动键"的守卫；③把 N2 家族档写进显式屏蔽名单——**不推荐**，等于用扩大屏蔽集换绿 | **已修〔2026-10-09 ㊵-638，按 owner 弹窗 #17 裁的第②条：比较键改成文件名〕**：按裁的第②条落地（同一格下方带日期追加段记形状与实测），旧封存样本一字未动、不需要第六次重基。台账 08 ㊵-636 第 ⑦ 段 · **〔2026-10-09 ㊵-638：按裁的②落地，旧样本一字未动〕** 行级叶子改成**以 `filename` 取键**（`_checkpoint_names`／`_flatten_by_checkpoint_name`／`_row_of`），并把"消失要红、新增只披露"拆成两条断言；另配一支**取键法自身**的测（换序 ⇒ 叶子图逐键不变，而旧的按索引取法在同一对夹具上必须给出不同键 ⇒ 这条等式不是恒真）。实测 `tests/taiji_native/test_cap0_inventory_contract.py` **17 passed, 1 xfailed in 55.72s**（改动前该面带一枚红＝本条登记的那支）。历史五次重基的做法本次**不需要第六次**：封存样本仍指 10-03 那份，盘上 13 枚不再移动任何既有键。台账 08 ㊵-638 第 ②③ 段。 |
| DEBT-G78 | **09 §6 的世界模型四法合同今天缺两项：`propose(goal,state,budget)` 与公开成对的 `snapshot`/`restore`（2026-10-09 由普查仪现读，不是印象）**：`scripts/training/audit_taiji_n5_world_model_contract.py` 对 `taiji/world_learning.py:1147` 的 `WorldDynamicsLearner`（公开方法实测 **8** 枚）出版 `contract_verdict=contract_incomplete`、`missing_verbs=["propose","snapshot_restore"]`；快照只存在于私有 `_snapshot_state_dict`（:1183）；另两个动词只由别名命中（`observe`←`online_update`、`feedback`←`record_schema_feedback`）⇒ "齐"只到**名字级**，语义需读签名（取法声明随件出版 `reading_limit`）。同族先例已在库：`taiji/world_evolution.py:235`/`:270` 是公开成对的 `checkpoint()`/`from_checkpoint()`，`taiji/world_learning.py:1748` 的 `rollout_episode` 已做多步展开 | 高（**挡的是 09 §2 N5 步骤③"世界学习器首次真实调用"**：没有带预算的候选生成与公开成对存取，就谈不上 §6 那份合同，更谈不上"恢复后完整继续"；同批还挡住一句过宽的旧话——"WorldDynamicsLearner 零真实调用"只在**语料训练链**成立（`train_seed_corpus.py` 引用实测 0），生产/训练面合计 **73 处、19 枚文件**，09 已就地打日期戳加范围限定） | 见 **PLAN-N5-06**（判据与守卫先冻）：J-N6a-3 把"完整继续"钉成"连续 8 步"对"4 步＋`snapshot`/`restore`＋4 步"的**逐位**摘要相等 ∧ `predict` 最大绝对差 `== 0.0`；G-N6a-1 明令**不许改普查器别名集换绿**；G-N6a-2 要求复用 `NativeWorldPredictionTrainer` 的信封形状且不重抄 `rollout_episode`；G-N6a-3 要求预算取 0/1/4 的条数差能为假 | **登记未修（2026-10-09 ㊵-642 立；实施属产品码 ⇒ 需 owner 批，本债不授权动 `taiji/`）**：读数件 `reports/taiji_n5_world_model_contract_census_20261009.json`，钉子 `tests/taiji_native/test_n5_13_world_model_contract_census_contract.py`（5 passed），设计件 `plans/reference/PLAN-N5-06_world_model_interface_prereg_20261009.md`（措辞门现读 `status=ok`、`criterion_lines=20`、`unpinned=0`）  · **〔2026-10-10 ㊵-651 落地：接口缺席这一半已结清〕** 弹窗 #18 批实施格 ⇒ 产品码 `taiji/world_learning.py` 公开三法：`propose(goal, state, budget, *, bind_target=True)` 与成对的 `snapshot()`／`restore(payload)`（`_snapshot_state_dict` 仍私有，不靠改名换在场）。普查器 `CONTRACT_VERBS` 一字未改（G-N6a-1 已钉成字面量测），现读 `public_method_count` 由 8→**11**、`missing_verbs` 由两枚变空、`contract_verdict=contract_complete`。读数分层（缺层不许合并成一句）：**名字在场**＝普查件；**语义等价**＝J-N6a-1 的位序断言（排除 `self` 后 `budget` 下标 2）＋`restore` 五道拒绝（含一枚"重算过摘要的伪造计数键"信封）＋J-N6a-3 的逐位继续（我在冻结形状之外加了一次污染步，否则 no-op 也能"相同"）；**默认位不变**＝J-N6a-4 三档 HEAD 对表逐位相同，且第一档复现仓内已钉的 `BASELINE_DIGEST`；**训练链真走到**＝**仍未证**（见下）。新增钉子 `tests/taiji_native/test_n5_19_world_model_interface_contract.py` 与 `test_n5_20_head_face_parity_contract.py`；`test_n5_13` 的真树钉按 ㊵-645 的约定重钉为 complete（缺席面仍由夹具两支钉着）。**本债剩下的三段**：① 接进语料训练链（那是第二格，要调用点全清单测＋面件在场计数 `J-N6a-6`）；② `WorldEpisodeEvaluator` 那批三对照读数（G-N6a-4：能力结论一律要走那里）；③ 09 §2 步骤③ 的"真跑"另批。在 ① ② 之前，"世界模型在训练里起作用"这句不许发表。 |
| DEBT-G79 | **保持侧配对的"同源核对"在 replay 半退化成两枚弱键：committed 的 before 面不带 `items_sha256` 也不带 `limit`（2026-10-10 预检现读）**：`reports/taiji_n2_replay24_before_20261008.json` 的键集实测含 `item_offset`／`first_item`，**不含** `items_sha256`／`limit`；而 `adjudicate_taiji_n2_04_retention_pair.py:195-209` 的 `_replay_items_disclosure()` 只对"至少有一边在场"的键做相等比较 ⇒ 它不会 rc=2，但实际只比偏移与首件，**换掉整套题集内容也不会被拦**。cap0 半今天没问题：声明摘要 `538835f6627a87c3faba84dbfd1138b5d8296a5dff37870bd258fdf2a3979c27`（出版在 `identity` 子字典）与重算的 `plans/manifests/cap0_eval_set_v2.json`（39,524 字节）逐位相等 ⇒ `cap0_identity=verified` | 中-高（**挡的是 J-N5d-1/G-N5d-1 这句话能声称到什么程度**：双臂 60k 是 334.63762799999676～473.23540269979276 秒／臂的沉没成本，跑完才发现配对依据只是"首件相同"，等于用一次长跑换一句不能发表的话；另记一条：两张 before 面把同一个 checkpoint 写成 `checkpoints\x` 与 `checkpoints/x` 两种拼写（DEBT-I8 那族），比较前不归一会造出假不同源） | 三条不等价，按成本与强度排：①**零算力重产 before 的 replay24 面**并让出件方 `measure_taiji_a30_repetition_penalty.py` 出版 `items_sha256`／`limit`（基件 `checkpoints/seed_a31self_with_circuit.pt` 仍在盘上，顺带做一次"同一基件重跑是否逐位可复算"的核对；旧 before 件不删，另立"新 before＋新旧对照"两支撑）——**推荐，且必须排在双臂起跑前**；②给判读器加硬门：`items_sha256` 不在场即 rc=2（改的是已冻仪器，要配"改前有判别力、改后弱键仍被拒"的双向测，且会让历史件不可再判）；③接受退化，但在 09/05/判读件里永久限定措辞——只写"偏移与首件相同"，禁止写"题集逐字相同"。**不许的写法**：拿 `first_item` 与 `item_offset` 相同去代答 G-N5d-1 的"题集同源" | **出路①已落地（2026-10-10 ㊵-645）：新 before 面已重产并带内容哈希；旧面与旧预检读数原样保留**：预检 `scripts/training/check_taiji_n5_pairing_preflight.py`＋读数件 `reports/taiji_n5_pairing_preflight_20261010.json`（现读 `verified`／`degraded_to_offset_and_first_item`），钉子 `tests/taiji_native/test_n5_15_pairing_preflight_contract.py`（6 passed，真树那支钉的是今天的读数——出件方开始出版那两枚键时它会红，届时须连本行与 08 一起改口） · **〔2026-10-10 ㊵-645 落地读数〕** 出件方 `measure_taiji_a30_repetition_penalty.py` 新增三枚出版键（`limit`／`items_sha256`／`manifest_path`，摘要走选中子集的规范化 JSON），重产出 `reports/taiji_n2_replay24_before_20261010.json`：同尺同基件同参数（24 题、offset 0、`--no-circuit`、`chain=base_raw_bytes_with_product_mask`），`items_sha256=d6e3f3f1f530888d845ab9c4a1c721e6d2b607acf65c3e095a36f425ae8830dd`，`base_sha256_unchanged=true`。预检指名新面 ⇒ `replay_identity=content_hash_available`、四枚键全在场（`reports/taiji_n5_pairing_preflight_20261010_hashed.json`），指名旧面仍读 `degraded_to_offset_and_first_item` ⇒ 退化这件事没有被参数"洗掉"。**顺带买到一条更强的事实**：新面与 10-08 旧面逐臂逐格对照 **差异 0 处**（含 `per_item` 每一行）⇒ 这张面在同一枚基座上可逐位复算，配对的分母不再依赖"那次跑恰好怎样"。基座这半也核过：`identity.checkpoint_sha256=d6169a358eaee6d194d4795e3167a7bcbb42dfde57b92aa1199bcebbed89699b` 与盘上字节重算值一致、`git_head=726a0a7517688d68cae37205c7277bac37e66853` 是可解析的 commit。⇒ **J-N5d-1 的 replay 半从"只能写偏移与首件"升级为"可写内容逐字相同"**；旧件不删，两条读数并存。台账 08 ㊵-645。 |
| DEBT-G80 | **保持侧跑道器的判读步原来不带巩固语料 ⇒ 分离机检要到四张面付完之后才红（2026-10-10 真跑 `--dry-run` 现读步骤位次）**：`adjudicate_taiji_n2_04_retention_pair.py:268` 需要 `--consolidation-corpus`，缺件时 `:300-306` 记 `G_N2c_4_disjointness=unverified`、`verdict=not_judged`、rc=2（判读器本身 fail-closed，没有争议）；而 `run_taiji_n5_retention_lane.py` 的判读步是计划里第 **3** 与第 **6** 步 ⇒ 前四步（两臂 × cap0 五维＋replay24）已经付费 | 中（**挡的是 `J-N5d-1` 里"保持不是泄露"这一半**：分离读数缺席时，`retention_holds` 即使七列数值成立也不能发表；另一面是成本——一次双臂评测的机器时间会白付） | 三条不等价，按强度与诚实度排：①跑道器**逐臂**点名本次通电的夜间件（㊵-565③ 的口径）并在任何子进程／mkdir 之前拒绝——夜间材料恰恰会被影子开关改变，所以两臂不许共用一枚默认值；②把 `--consolidation-corpus` 做成单枚必填、两臂同件（省事，但会把控制臂的分离检查建立到治疗臂的材料上，属"更弱的结论当更强的形状卖"）；③维持现状（判读器自己 rc=2），只把措辞永久限定成"分离性未核"。**不许的写法**：拿"保持 manifest 与语料大概不重合"代答 `G_N2c_4_disjointness`；也不许在无夜间件时把 `status=separation_unverified` 读成"无交集" | **已落地（2026-10-10 ㊵-648）＝出路①**：两枚 `action="append"` 旗标（`type=Path`、无默认值）＋ `corpus_refusals()` 花钱前拒绝，`--only cap0`／`--only replay` 不把它当假前置（也钉了测，防我造假门）。钉子 `tests/taiji_native/test_n5_16_retention_lane_runner_contract.py`（**16 passed**，含"逐臂语料不同名且各自线进各自的判读步"与"缺语料时零子进程 rc=2"）；真进程冒烟现读第 3 步带 `corpus-20261009T035904Z-8e1c02165ec4.jsonl`、第 6 步带 `corpus-20261009T035900Z-d320f24def65.jsonl` |
| DEBT-G81 | **打印含 `⇒` 的拒绝消息在 win32 GBK stdout 上抛 `UnicodeEncodeError`，把设计好的 rc=2 降级成 rc=1＋traceback（2026-10-10 跑道器真进程实测；契约测在 capsys 下恒绿，看不见这一支）**：机制单独证过一次——三条含 `⇒` 的消息在 `gbk` 下全部抛错，纯中文消息可编码 ⇒ 元凶是这一枚字形而不是中文（`×`／`≠`／`≥` 同族，本轮未逐一验）。仓里**已有**正确写法但用者寥寥：`eval_taiji_artifact_consumption_policy.py:336-338` 的 `hasattr` 守卫版 `sys.stdout.reconfigure(encoding="utf-8")` | 中（**挡的是"fail-closed 会响亮拒绝"这句话的兑现方式**：owner 在终端看到的不是"缺哪一枚键"，而是一段 traceback，容易被读成"仪器坏了"而放弃这一步；影响面＝任何把判据结论或拒绝清单写到 stdout/stderr 的判读器） | 三条不等价：①逐件在 `main()` 里按既有写法加 `reconfigure`，并**用真进程**（`PYTHONIOENCODING=gbk` 的子进程）测一次，capsys 不算；②把出口消息里的字形换成 ASCII（`=>`）——一次性，但与预注册正文里引用同一字形的文本不再一致，且字形在本仓是可读性资产；③做成普查门：扫"打印路径上出现的字符串常量含非 GBK 可编码字形"并出版命中数。③ 有一条必须先解决的取法教训：`grep 'print(.*⇒'` 只抓到 4 处／3 枚文件，而今天崩的那条写作 `print("  - " + row)`，箭头住在被调用函数的常量里 ⇒ **该式是下界**，做门要按"消息字符串定义处"取而不是按 print 行取。**不许的写法**：把未修那两枚的可达性写成"确认会崩"（本轮没起真进程验它们） | **登记未修（2026-10-10 ㊵-648 立；本跑道器一枚已按①修好）**：已修＝`run_taiji_n5_retention_lane.py`（rc 由实测 1 回到设计值 2，钉子 `test_the_refusal_list_survives_a_gbk_console`）；未修的命中＝`audit_taiji_zero_face_census.py:296`、`probe_taiji_a30_copy_evidence_dose.py:373`（两处都写 stderr，可达性本轮未测）；修法③的普查面尚未建，建之前不许发表"全仓只有两处"  · **〔2026-10-10 ㊵-649 就地更正本行的归因与半径，普查仪已建成（修法③落地）〕** 本行③点名的两枚"未修命中"经普查现读**一枚不成立、一枚性质写错**：`probe_taiji_a30_copy_evidence_dose.py:373` **本来就带** UTF-8 守卫（`console_guard=true`），而 `audit_taiji_zero_face_census.py:296` 落 **stderr** ⇒ 实测 `sys.stderr.errors=backslashreplace`，它把字形印成字面量 `⇒` 而**不崩**（难看面）。真正的崩溃面是**未守卫且向 stdout 打印非可编码字形**＝**46 枚文件**（`scripts/archive` 42 枚，多为 ✅❌⚠ 类装饰字形；`scripts/training` 4 枚＝`train_cross_domain_collab.py`／`train_tinystories.py`／`train_tinystories_field.py`／`utils.py`），本行原先那句"两处"是 `grep 'print(.*⇒'` 的**下界**，不是半径。⇒ 半径更正必须连着说：**主线本轮要跑的六枚入口件全部不在崩溃面里**（`train_seed_corpus.py` 的 8 处 `⇒` 只进 `raise RuntimeError`，走 stderr），这条已钉成 `test_n5_17` 的真树测。修法队列按普查面走而不是本行旧名单：先 `training` 那 4 枚（`utils.py` 被 7 枚件 import，要单独一轮验证），`archive` 那 42 枚属历史脚本、可批量但优先级低。仪器与读数＝`scripts/training/audit_taiji_console_glyph_encodability.py`＋`reports/taiji_console_glyph_census_20261010.json`（现读 扫面 1199／有输出调用 1003／已守卫 79／崩溃面 46／转义面 1／pool 上界 92／BOM 解析不了 3 枚已披露）。  · **〔2026-10-10 ㊵-651 第二次更正：普查仪升 v2，加的是"用法屏"这一档，而且当场结清一枚长红〕** 上一格我写"docstring 永远不流向控制台"——**这句被一支既有守卫否证了**：`tests/taiji_native/test_a30_probe_help_runs.py` 里的 `probe_taiji_a30_stop_failure.py --help` 在 GBK 上崩，根因是 `ArgumentParser(description=__doc__)`（:1056）把带 `⇒` 的**模块 docstring** 送上用法屏。⇒ docstring 算不算消息，取决于它有没有被喂进 `description=`／`epilog=`／`help=`（含 `__doc__` 那条线），不能一律排除。仪器据此加第四档并升 **v2**：`constant_count_argparse_surface`＝**88** 条／`help_face_file_count`＝**63** 枚（扫面 1200、构造 argparse 的 685 枚、已守卫 80 枚）。**并档的诱惑要写明**：v2 第一版我把用法屏并进 `crash_face`，数立刻从 46 跳到 **108**——但 `parser.error` 的 usage 落 **stderr**（转义不崩），只有 `--help` 这条路写 stdout，所以并档是把"只在人查用法时才崩"说成"正常跑会崩"，是**更吓人也更错**的一张面 ⇒ 拆回两档、各给判定键（`no_unguarded_stdout_glyph_print` 与 `no_unguarded_argparse_help_glyph`）。本债同时结掉一枚**真实长红**：给那枚探针加了守卫 ⇒ `test_a30_probe_help_runs.py` **2 passed**，并把 `test_n5_17` 的"未修文件名单"里那条 `probe_taiji_a30_copy_evidence_dose.py` 之外的第三枚（本枚）纳入同一形状。**剩余队列不变**：`crash_face` 仍是 46 枚（`scripts/archive` 42／`scripts/training` 4），`help_face` 那 63 枚优先级低但同族；读数件 `reports/taiji_console_glyph_census_20261010_v2.json`（v1 那份原样保留，两条读数并存）。  · **〔2026-10-10 ㊵-652 活队列推进一格：崩溃面 46→43，`scripts/training` 只剩一枚库】`train_cross_domain_collab.py`／`train_tinystories.py`／`train_tinystories_field.py` 三枚入口各加仓内既有那道 `sys.stdout.reconfigure(encoding="utf-8")` ⇒ 读数件 `reports/taiji_console_glyph_census_20261010_v3.json`（v2 原样保留，46→43 是可复算的两件）；已守卫 80→83、`help_face` 63→62、`constant_count_stdout_print` 仍 298（守卫改变的是"是否计入缺陷面"，不是常量枚数，两个量不许混报）。**剩下的那一枚是 `scripts/training/utils.py`，本行明确不在此修**：它被 7 枚入口 import，那 4 处告警 print 住在库函数里；在库里改全局 stdout 编码＝给所有 import 方（含 pytest 的捕获流）下副作用，正解是逐枚入口加守卫或把告警改成返回值由调用方打印——横跨 7 枚调用方的改动不在本线半径内。**不许的写法**：为凑面绿而在库里吞掉编码错误。钉子：`test_n5_17_console_glyph_census_contract.py::test_the_live_training_face_crash_list_is_down_to_the_library_only`（该册 14 passed，含"数不为零"那一半）。 |
| DEBT-G82 | **harness 侧两族门在 HEAD 上就是红的（2026-10-10 ㊵-648 现读：`doc-sync` 40 passed／3 failed，`hygiene` 16 passed／2 failed）**：失败帧＝`verify-translation-pairing`（`docs/config-catalog.md` 与 `docs/config-catalog.zh.md` 与 `.i18n.yaml` 记录不再一致；`packages/web/web-search-scraper/README.zh.md:49/69/71` 三处链接用了错语档）、`verify-package-readme-model-experience`（`web-search-scraper/README.md` 的 `## Model Experience` 与 `## Known Limitations and Deferred Work` 不是最后两节）、`doc-standard.spec.ts`（同件 `missing 概述`／`missing 目录` 等 3 项，1 failed 对 21 passed）、`verify-default-product-isolation`（`@taiji/dsh-desktop` 的 optionalDependencies 指向 `@taiji/dsh-experimental-schedule-bundle`＝默认产品含实验包）、`verify-package-invariants`（同件缺 omitted companion 的理由句） | 中（**挡的是"提交前门全绿"这句话的用法**：这两族红着的时候，任何 PASS 计数只能按叶子点名读，不许发表成整体绿；对 N 主线是间接影响——跑道器与判读器不读 harness 文件，但下一个用总数当验收的人会误判） | 归属已做完且**不在本线半径**（三条并列证据见 08 ㊵-648⑧：失败帧全在 harness 件、那些件 `git status` 干净⇒HEAD 既有红、三枚失败叶子的脚本里 `roadmap` 与 `plans/` 两个词零命中⇒与本笔输入面不相交）。修法属另一工作面（`web-search-scraper` 入库＋实验包挂载），本线不代改：孪生件必须**两侧同改再按对重录 pairing**（[[taiji-harness-bilingual-pair-derivative-coupling]]），optionalDependencies 那条按"登记＋manifest 移项＋同提交改锁，顺序不可拆、必须原子"处理。**不许的写法**：把这 5 枚并成一句"门禁红了"，或为了凑绿去 `--write` 重录 pairing | **登记未修（2026-10-10 ㊵-648 立，首次登记）**：`grep -c translation-pairing` 对本册现读 **0** ⇒ 此前无人登记过这一族；读数面如上，命令面＝`cd taiji-harness` 后 `corepack pnpm run doc-sync`／`corepack pnpm run hygiene`（这台机 `pnpm` 不在 PATH） |

## 8. 处置阶段入口条件

进入本册处置阶段**必须**满足：

1. 主线 P5.2c 及其直接下游（P5.2d 在线结果回写）已收尾或明确暂停；
2. 已按 §4 建立 `SystemExit` 可观测性（能拿到栈）；
3. 类别 A 两项已完成定性（真违规 vs 白名单过期），并各自给出「修代码」或「改契约」的结论；
4. 处置前后各采一次 §2 基线，形成可对比的量化。

**禁止**：在未定性前直接放宽 `test_architecture_contract` / `test_naming_boundary_contract` 的断言
来「让套件变绿」——这两项保护的是原生基底自足性契约。

## B 支线（架构债线）2026-09-26 登记：已量，本日内除"死模块去留"外全部结清

**DEBT-B4-1（本条前半段的旧结论**作废**，2026-09-26 同日推翻）：B-4 读数并不"天生带噪"。**
旧结论：两次全量跑之间 `resonance/ensemble.py` 相差 −203 行（869 ⇒ 666），丢失的是散点单行
⇒ 判成"随机初值路径无全局种子"，并据此把 CI 阈值压到 45.0。
**推翻证据**：同一棵树连跑两次（含本轮 12 个测试件）得 65.03% / 65.25%，面内聚合只差 0.22 个点、
ensemble 只差 8 行 ⇒ 不是抖动。真正的 −203 那次是**有 lane 没跑完**：
`checkpoints/.p*-*.pt` 被并发进程占用 ⇒ `PermissionError [WinError 32]`，
`test_m3_workbench_readonly` / `test_semantic_grounding` / `test_terminal_three_domain_governance`
这些会真跑前向的门当场中断（见 DEBT-B4-5），该面的读数因此低到 46%，被误读成"覆盖率抖动"。
教训（与既有"别从聚合入口继承跑不了"同源）：**读数异常先问"哪条 lane 没跑完"，再谈随机性**；
CI 的门因此改成"只在套件步成功时判"，套件红时用跳过而不是叠一条语义错位的覆盖率红。

**DEBT-B4-5（已修，2026-09-26，`622d6712`）：gate 把临时 checkpoint 以固定名写进仓库共享的 `checkpoints/`，并发必撞。**
登记时写的是"四条 gate"，**实际同族六条**：`eval_taiji_semantic_grounding`（p2-9）、
`eval_taiji_multistep_grounding_recovery`（p2-10）、`eval_taiji_natural_language_workbench`（p2-8）、
`eval_taiji_ide_language_chain`（p2-11）、`eval_taiji_m3_workbench_readonly`（m3）、
`eval_taiji_terminal_three_domain_governance`（p4-12）。它们用 `checkpoints/.<gate>-<seed>.pt` 当 scratch，
收尾 `unlink(missing_ok=True)`；两条套件同时跑时该文件被另一方持有 ⇒ `PermissionError [WinError 32]`
把门打断，表现为"随机红"，并顺带压低覆盖率读数（DEBT-B4-1 那次误判即源于此）。

**修法取 `checkpoints/.scratch/<pid>/<原名>.pt`——按 PID 分目录，而不是给文件名加 PID 后缀。**
两条理由（后一条是本仓特有的坑）：① p2-8/p2-9/p2-10 的判据里钉的是**字面文件名**
`restored_checkpoint_name == "seed:.p2-9-semantic-grounding-11.pt"`，把 PID 塞进文件名就等于改动判据
（`api/seed_runtime.py` 的 `name` 只取 `path.name` ⇒ 换目录不改读数）；②
`eval_taiji_cap0_inventory.py::_checkpoint_inventory()` 用**非递归** `glob("*.pt")` 盘点该目录，而
`test_cap0_inventory_contract.py::test_a_fresh_inventory_sample_reproduces_the_sealed_one` 拿现场重采
与封存样本逐叶比较 ⇒ scratch 平铺在 `checkpoints/` 那一层时，一次并发跑就能把普查的字段面挪红。
不用 `tempfile`／不用 `tests/_scratch.py` 那条仓库外的路：托管 Windows runner 能在 TemporaryDirectory
里建目录却拒绝 Python 独占建文件（`eval_taiji_terminal_three_domain_governance.py` 里同一段说明）。

**判别实验（同一台机器、同一批 lane）**：把放置改回旧的"共享目录＋固定名"后 4 进程并发 ⇒ **4/4 红**
（两条 `WinError 32`，两条整组 5 failed——那是新加的 `fresh_scratch()` 起点删不净就拒跑，不静默跑在
别人的陈旧状态上）；改回新放置后 6 进程并发（3 对）⇒ **30 次 lane 执行全绿**，收尾 `checkpoints/`
零 scratch 残留。守卫 `tests/test_scratch_checkpoint_isolation.py` 6 条，四条变异各自会红。
其中"空 PID 目录要一并收掉"是实测逼出来的两条路：Windows 在删掉目录里最后一个文件时会把该目录 mtime
顶成当下 ⇒ "空且已久"对本轮刚清空者**永不成立**，所以本轮清空者直接 `rmdir`、别轮弃用者才走时间戳。

**顺带否证本条登记时的两句话**（改台账的理由是重推，不是手抄）：
① "同一目录里还留过 6 个测试残留（`s45-active-<pid>.pt`）⇒ DEBT-I7 第二实例"这一句**不成立**：
所有 artifact-store 测试早已改走 `tests/_scratch.py::artifact_scratch_root()`（仓库外），
`test_artifact_store_scratch_contract.py` 4 条绿，`output/manual-r5-canary/` 盘上只剩它自己的
`README.md` + `native-canary.pt`（8-29 的原有件），外部 scratch 根当前 0 文件 0 目录。
② p2-12/p2-13 两条同族 gate 已在 09-14（`f6d9c0cf3`）改走 TemporaryDirectory，不需再动。

**新登记的纠缠（未处置，属所有者裁定）：`checkpoints/.p2-12-conflict.pt` 与
`.p2-12-natural-language-write.pt` 是 09-13 一次跑的残留（该 lane 09-14 已改道），文件如今仍在盘上，
而 CAP-0 封存样本 `reports/taiji_cap0_inventory_beta4_20260920.json` 的 `checkpoint_inventory`
**把这两份残留当成了基线的一部分**（逐叶比较含 filename/bytes/modified_utc）。
所以"顺手清掉仓根残留"会直接把 `test_a_fresh_inventory_sample_reproduces_the_sealed_one` 弄红；
要清必须先按本仓"改行为须同批再生报告"的做法重封一份 CAP-0 样本，那是 CAP 面的动作，不在 B 线里自作主张。
另：`scripts/training/smoke_taiji_r2_h3_6_target_encoder.py` 是同型写法（固定名 + `finally: unlink`），
但它没接进任何 gate／套件（仓内仅一处归档提及），因此只在"人手动跑"时才会撞——记录不动它。


**DEBT-B4-2：`brain/working_memory.py` 是"仅注册未接入"的死模块；其索引错位缺陷已修（2026-09-26，`140f42a6`）。**
`cortex.py:219-223` 自己写明它未接入生成路径（真正的上下文记忆走 `agent/working_memory` 经
ContextManager），但它仍在 `neuroplex/brain/` 里且被算进过覆盖率分母（本已从 B-4 度量面剔除）。
实测 `append_round` 在 deque 触发 FIFO 丢弃后 `round_marks` 整体错位：`max_tokens=8`，
依次追加 (1,2,3|4)、(6,7|8)、(9,10,11|12) 后 buffer=[5..12]，标记却是 `(0,5)/(5,8)/(8,8)`
⇒ **刚写入的一轮记成空区间**，旧轮指向别人的 token；`_first_domain` 式的"簿记漂移"在这里
不会报错、只会让依赖 importance/轮次范围的逻辑静默读错。
**已做**：按实际丢弃数平移**全部**标记，另加 `tests/test_working_memory_window.py` 9 条窗口不变量
（期望值全部手推，变异 `dropped = 0` 时 9 条里红 6 条）。
**仍未裁定**：这个模块是删还是接——修它等于给死代码定行为，只是把"已经存在的簿记"修对，
不构成"它该留在生产路径里"的论据。

**DEBT-B4-3（已修，2026-09-26，`8bb0cb4c`）：`Cortex.device` 实际是字符串，签名却标 `torch.device`。**
`cortex.py:114` 直接存传入值（fallback 装配下为 `'cpu'`），而 `_cortex_quality.rolling_nll_quality(device: torch.device, …)`
等签名标注为 `torch.device`。torch 接受字符串所以能跑，但 `tensor.device == cortex.device` **恒 False**
（本会话写测试时踩到，断言被迫先 `torch.device(str(...))` 归一）。属类型谎报，不改行为。
修法：`Cortex.__init__` 里 `self.device = torch.device(device)`；写测试时那条归一化临时断言已撤。

**DEBT-B4-4：根目录台账外的 `consolidation/`（空目录）——**源头已修，红由一个未重启的旧码 runtime 维持。**
创建者是 `seed_platform/sleep_pass.py` 的 `_DIR_NAME`，旧值是裸 `"consolidation"`，经
`get_external_path()` 落在**外部数据根**，而该根默认＝项目根 ⇒ 产品运行时在仓根建了个未登记空目录，
`test_folder_structure_guard` 自此恒红。00:44 的 `8b52de4f` 已把它并进 `data/consolidation` 族
（与兄弟 `_CORPUS_DIR = data/consolidated` 一致）并 rmdir 了当时那份。
本轮（20:5x）复现并核实：`rmdir` 之后**秒级复建**，持有者是 PID 24412 = `python api/main.py`，
启动于 09-24 23:08 ⇒ 载的是修复前的模块，正是 `8b52de4f` 提交说明里预告的那一支。
**处置**：不与其抢删（删了只会再多一次红），也不放宽守卫。**重启该 runtime 即彻底消失**，
重启属所有者动作、本线不代做。命令：停 PID 24412 后重跑 `python api/main.py`。

`8b52de4f` 那条 rmdir＋本次这条"秒级复建"合起来把守卫的红**完整归因**到一个未重启的进程上；
在它重启前，全量套件的既有红集合里会稳定含这一条，读数时须按"环境态"处理，不要当成新代码的回归。

### 本轮（2026-09-26）B 线收口后的全量读数：既有红从 5 条降到 2 条，补登记后只剩 1 条环境态

**第二次全量复跑（补登记之后，同一面）：1 failed / 2186 passed / 6 skipped / 1 xfailed，1909.05s。**
唯一剩下的红就是 DEBT-B4-4 那条环境态（`consolidation/` 仍在盘上）——b0_n2 那条**实测消失**，
不是靠"只改了那四个文件"推出来的。同一趟 B-4 门 **65.25%（2435/3732）PASS**，
与第一次全量**逐位相同** ⇒ 两次独立的 30 分钟全量给出同一个面内读数，这本身是对
DEBT-B4-1 那条"天生带噪"误判的又一次否证。

`python -m pytest tests/ -q --cov`（与 CI 同面）：**2 failed / 2185 passed / 6 skipped / 1 xfailed，1800.61s**。
同一趟的 `output/coverage_full.json` 过 B-4 门：**65.25%（2435/3732）PASS**（阈值 60.0，
与本会话早前两次同树读数 65.03/65.25 一致 ⇒ scratch 改道没有挪动度量面）。

剩下的 2 条，逐条对账：

1. `test_folder_structure_guard::test_every_root_directory_is_listed_in_the_ledger`
   ＝ DEBT-B4-4，**环境态**，唯一动作是重启载着旧码的 `python api/main.py`（PID 24412）。
2. `test_b0_n2_stop_reason_disposition_contract::test_current_review_surface_is_complete`
   ＝ **A 支线（B0/N2 复核面），非本轮引入，且本会话未改动过它的任何一个源文件**。
   实测漂移：`EXPECTED_CONSUMERS` 冻结在 19，活扫描 25，多出 6 处——
   `run_taiji_collab_handoff_entry_evidence.py`／`run_taiji_unified_entry_evidence.py`／
   `taiji/collab_handoff.py`（09-19 HANDOFF-M4 与统一入口结案）、
   `train_taiji_r2_content_binding.py`（09-19）、`train_taiji_r2_readout_retrain.py`／
   `test_readout_retrain_runner_contract.py`（09-22）。
   即三条特性提交把被审计的面撑大了却没做 disposition，**红了四天**。
   **已于同日补登记**（见冻结预注册 §9）：3 处真消费者进 `EXPECTED_CONSUMERS`
   （其中两条 judgement 各配 J11/J12 源内标记），3 处同名字段按 §8 先例进 `SCAN_EXCLUSIONS`
   ——排除依据是**扫出来的否证**（那三个文件里 `all_members_*`／`goal_reached`／
   `contract_intercepted`／`rule_revision` 出现 0 次），不是"看着不像消费者"。
   补登记后 live 22 / 期望 22 / drift 空 / 12 处标记全在，`review_checks_passed=true`，
   B0 家族 173 条全绿。**没有放宽任何断言**：`test_live_drift_is_a_failing_audit`
   三种变异（added／removed／missing_marker）仍然各自会红。
   留给所有者的一条实质观察：`run_taiji_unified_entry_evidence.py` 是规则的**手抄副本**——
   它自己写死 `all_members_exhausted` 等终值字面量（`success` 独立算、`trace_valid` 只查事件 kind，
   故新 reason 不会被当成成功）。今天的两支条件互不重叠所以标签正确，
   **但若哪天真把"无可绑定成员"与"可绑定成员全失败"两支合并，这个 runner 的标签必须重审**（J12 的
   safe_because 里已写死这句话）。

被消掉的那 3 条（相对本轮开头登记的 5 条红）：

* `naming_boundary`：本会话按守卫规定流程补登记结清（理由入 `ARCHITECTURE_DIRECTION_2026_08.md` §6）。
* `artifact_store_scratch`：早于本轮就已结清（09-19 `8a7966b4`），本会话实跑其契约守卫 4 条全绿——
  此前把它列进"当前红"是我手抄旧判断，已在 DEBT-B4-5 里否证。
* `cap0_inventory 金样复现`：本趟绿。**机制已双向证到**（这是本会话当场跑的探针，不是追认）：
  往 `checkpoints/` 平铺**一个** `.p2-9-*.pt` ⇒ `test_a_fresh_inventory_sample_reproduces_the_sealed_one`
  当场红（16.9s），撤掉⇒绿（18.2s）。所以"scratch 收进 `.scratch/` 子目录"这条设计选择不是洁癖。
  **但历史归因不作断言**：那条红当时为何出现，我手上没有留下当场的字段级证据，只能说机制成立、
  且现在这一族红在结构上不可能再由 scratch 平铺造成。


## 9. 「仓库按策略不入库的产物」与契约不可跑（2026-09-30 登记，H19j–H19o）

登记原因：这条策略原先只存在于 `tests/conftest.py` 的 docstring 里，没有进台账。下次看到成片
skip 的人会怀疑其合理性，或干脆去「修」它们——两种反应都是错的。**决定 CI 能验证什么的策略，
应当进台账。**

**策略本身**：`.gitignore` 携带 `*.pt`（第 50 行）与 `*.jsonl`（第 140 行），第 165 行写明理由
——「Checkpoint directories (large model files)」。因此 `checkpoints/*.pt`、`tests/fixtures/*.jsonl`、
`data/simple_zh/*.jsonl`、`reports/**/epoch30.pt` 是**按策略排除**的，不是遗漏入库的产物。
H19h 曾按「应当入库」的思路提议过，已在 H19j 更正。

**后果**：fresh checkout（也就是 CI 跑的那种环境）拿不到这些产物，依赖它们的契约测试会以
`FileNotFoundError` 或「仪器走降级分支产出额外字段」的形式失败——两者都说不清缺的是什么。

**处置**：`tests/conftest.py` 的 `pytest_collection_modifyitems` 钩子。模块声明
`LOCAL_ONLY_ARTIFACTS = (...)`，缺产物时跳过，跳过原因里点名每个缺失路径并说明它是被 ignore 的。
个别测试可用 `@pytest.mark.no_local_artifacts` 豁免——用于「多数测试需产物、少数不需」的模块。

**使整件事可验证的关键事实**：`git worktree add` 产出的 checkout 只有受跟踪文件。H19j 之前
以「守卫本地验证不了」为由拒绝做守卫，那个理由是可以拆掉的。

> **H19w 更正（2026-09-30）：上面原文写的是「**就是 CI 的环境**」，这个推辑**过了头**。
> worktree 只等价于 CI 的**「产物缺失」这一维**，不等价于其他维度，已被两个反例打脱：
>
> - **git 历史深度**：`actions/checkout@v4` 默认 `depth: 1`，CI 是浅克隆，worktree 是本地完整历史。
>   `test_the_guard_rejects_the_pre_v4_instrument` 跑 `git show 72d81faf^:...`，本地通过、CI 报
>   `CalledProcessError`。已给 `test` / `test-windows` 的 checkout 加 `fetch-depth: 0`。
> - **依赖版本**：CI 装 `torch` **不锁版本**，本机是 2.13.0+cpu、CI 是 2.14.0+cpu。
>   `test_copy_circuit_contract` 断言的是**bitwise identical**（哈希 `json.dumps(tensor.tolist())`），浮点核不同就是不同摘要。
>
> 对续作的启示：**验证保卫可行性时必须声明变量是哪些**；本轮我说过「worktree 就是 CI」，而它在
> 依赖版本这一维上从来不成立。

**测量轨迹**（`tests/taiji_native`，fresh checkout）：

| 阶段 | failed | passed | skipped |
|---|---|---|---|
| 首次复现 | 43 | 1625 | 35 |
| H19j（模块级守卫） | 22 | 1625 | 56 |
| H19n（豁免标记） | 0（10 个模块内） | 131 | 28 |
| H19o（收窄跳过） | 0（10 个模块内） | 133 | 26 |

H19o 的 133 与加守卫前的 133 完全一致，即**没有任何原本能跑的测试停止运行**。

**未完成 / 已知宽于必要**：H19o 仍有约 4 个测试比尚可能多跳（不需要产物但没有豁免）。此外，台账
未登记还剩下的**真实断言不符**（不属于产物缺失）：它们本轮被判定为不应被 skip 掩盖，但尚未定性——
需要单独追根因。
