# M5 退出材料补齐：CAP F 维改为真复判＋现场重跑，H 维完成设备标定采样（2026-09-20）

定位：[M5 退出就绪度评审表](M5_EXIT_READINESS_REVIEW_20260920.md) §3 里 F 维记 `partial`（"合同引用……证据在库待映射"）、H 维记 `partial`（"响应/内存门未标定"）。本件记录这两半的处置结果，并登记处置过程中实测出的三条新问题。**本件不改变**：C/D/E 语言维度仍 0.0、B/G 仍待人工复核、R2 语言能力仍按用户 2026-09-20 裁决列为退出显式排除项。

## §1 F 维：旧判据是"证据文件在不在盘上"，现已换成逐项复算

### §1.1 缺陷

`eval_taiji_cap0_baseline.py` 的 F 支此前只产出 `report_present`——一个文件存在性布尔。被引报告里的门到底过没过，它一个字段都不读。这正是本仓反复登记的失效形态（DEBT-I4 第八批、普查 §1e：**只读封存件的检查永远不会红**）。同时 runner 内自己抄了一份 `F_CONTRACTS`（门文本手抄件），它已经和冻结评价集漂移：F04 那份写的是"表层回答不随任何消融改变（固定模板）"，而 09-18 链路批已实测该读数依链路而定。

### §1.2 现在的判据

新模块 `scripts/training/eval_taiji_cap0_f_dimension.py`：

1. 四项 F 的项名、门文本、被引报告、`must_show` **一律取自冻结评价集** `plans/manifests/cap0_eval_set_v1.json`（不再抄第二份）；
2. 每项按门文本拆成子句，逐句从**被引报告的原始数字**复算；
3. 拆不出可机检形态的子句记 `held=null` ⇒ 该项至多 `partial`；
4. 每项另算 `must_show`（07 §4.2 后半句"展示输入到实际结果，不得只报局部 probe"）；**没有机检实现的 must_show 会把一个已过门的项压回 partial**；
5. 现场重跑统一入口证据包（`--f-live-evidence`）：同一 runner、同一默认参数、产物只写临时目录，**封存报告不覆写**（预注册 §4），并逐臂比对是否复现封存读数。

### §1.3 复判结果（2026-09-20，git head `983c0001`）

| 项 | 冻结门 | 复算结果 | 判定 |
|---|---|---|---|
| F01 B1 表示门 | 六门全过 + `outcome=representation_discriminative` | 六门逐真；`outcome` 相符；`must_show`：候选格 `episodes.candidate=396` 条实际执行、逐对**实际**增益 6 条在册 | **pass** |
| F02 B2 选择门 | G1–G6 全过 | `G4_task_gate_H1=false`、`G5_collaboration_gate_H2=false` ⇒ 与自带 `experiment_passed=false` 一致 | **fail**（负结果保持，不改绿） |
| F03 结构空间 | create 行三格 +2.000、interleaved 6/6、零回归 | 三格归属、零回归、无未解释变化三条可验且成立；**"+2.000" 与 "interleaved 6/6" 在封存件里没有可定位的对应字段** | **partial**（两条子句不可机检） |
| F04 整模型加载链 | 默认入口可加载并产出原始输出 | 加载 `load_ok=true`、产出 1015 字节成立；但 `template_signature.templated=true / distinct_signatures=1`（输出是固定模板回显），且 `wiring_defect=true / default_tick=36` 对 `most_trained_tick=16000000` | **fail**（两条 must_show 直接命中） |

维度门（评价集 `gates.F`：至少一项满足其独立冻结门，并展示输入到实际结果）：**F01 同时过自己的门与自己那条链 ⇒ F 维判 pass**。现场重跑的 bundle 级演示单独记在 `end_to_end_demonstration`，`counts_as_item_chain=false`——它证明的是统一执行入口，拿它点亮 F 维就是代理信号冒充验收。

### §1.4 统一入口现场重跑

`full/simple_strategy/disable_memory/disable_selection/disable_writeback` 五臂逐臂读数（成功率、含注入总步数、执行动作数、未见实例结果）与封存件 `taiji_unified_entry_evidence_20260919.json` **逐位一致**（`matches_sealed_report=true`），`returncode=0`，墙钟 21.5 s（runner 自报 11.2 s，余为进程启动与父子进程开销），落在冻结 sweep 上限 600 s 内。复算 L1–L4 与报告记录一致且全过。

两条**不可机检**子句登记在案（不因此改判，但不得当作已证）：
- **L3**：预注册文本要求"注入 m0 首选失败后 full 臂让位仍 `goal_reached`"，runner 实现的判据是 `main_success_rate ≥ 2/3`——两者不同一，本件按 runner 口径复算并登记该代理；
- **L4**：冻结文本还要求 bundle digest 逐臂一致与越权检查，逐臂 payload 未携带 digest。

### §1.5 F 维读法裁决（所有者 2026-09-20）

07 §4.2 那句按 **"F01 一项即算满足"** 读：不要求代表能力必须出现在 `SeedRuntime.chat` 的整模型出口上。
代价随判词一同入账——**F 维 pass 说的是 B1 表示门这台仪器的判别力经过了实际执行链验证（396 条候选格
episode + 逐对实际增益），不是整模型入口的语言能力**。若改按严读法，F 维只能记 partial，而那要等 R2
表达层（按同批裁决挂起中）才有对应的可判定出口。bundle 级现场重跑仍只记在 `end_to_end_demonstration`
且 `counts_as_item_chain=false`，不作为点亮维度门的依据。

## §2 H 维：完成设备标定采样，阈值只提草案不冻结

新脚本 `scripts/training/calibrate_taiji_cap0_h_gates.py`，产物 [reports/taiji_cap0_h_calibration_20260920.json](../../reports/taiji_cap0_h_calibration_20260920.json)。机器空闲、5 次重复、每次**新建进程**、链路显式取 `relax_legacy_guard+constrained_decode`（与正式评价同链路）。

| 读数 | min | median | max | 建议上限（max×2） |
|---|---|---|---|---|
| H01 冷启动 s | 0.2402 | 0.2611 | 0.2801 | **0.5602** |
| H02 首响应 s | 0.4760 | 0.4943 | 0.5042 | **1.0084** |
| H03 单次 chat s（30 次总时长折算） | 0.5570 | 0.5591 | 0.5678 | **1.1356** |
| H04 峰值 traced 内存 B | 207,791 | 210,642 | 210,942 | **421,884** |
| H05 无崩溃 | 5×30 次全过（150 次零崩溃） | | | 布尔门，无需阈值 |

`threshold_status = proposed_not_frozen`：**冻结归项目所有者/CI 的设备决策**，本件不冻结。口径披露：该上限只约束"相对本机基线是否劣化"，不构成对用户侧设备的能力承诺；换设备或换链路须重标。H06（中断恢复）仍 `not_executed`，不在此标定范围内。

### §2.1 采样顺带测出两条既有记载要更正

1. **`--health` 单次墙钟在 required 链是 ~28 s**。债册 DEBT-I4 先后记过 17.2 s（A05 消融落地前）与 25.6 s（A05 之后、裸链路口径）；本轮 required 链实测 28.07–28.60 s（五次全落 28.1±0.3）。⇒ 战役驱动的每阶段预算加数应改记 **~28 s/阶段**，其中换链本身约值 +3 s。
2. **H 的读数同样依链路而定**。同一 checkpoint 裸链路单点读数 H02=0.3223 s / H03=13.19 s / H04=163,511 B，required 链分布 H02≈0.49 s / H03≈16.8 s / H04≈210,000 B。这与 A05b 的链路依赖是同一类：**标定与正式评价必须同链路取数**，否则门限一上线就永久误判。

## §3 处置 F/H 时实测出的新发现：M5 退出 CAP 跑在与判据不同的链路上

- 本次呈报的退出材料 `reports/taiji_cap0_exit_health_20260920.json` 与 `taiji_cap0_exit_baseline_20260920.json` 的 `chain` 均为 `{relax_legacy_guard: false, constrained_decode: false}`（`checkpoint=seed_corpus.pt`，`git_head=db29dbfd`）。
- 而 `check_p3b_criteria.py:34` 的 `REQUIRED_CHAIN = {true, true}`，且 `judge_health` 在读任何字段**之前**就要求 `chain == REQUIRED_CHAIN`，否则 `chain_mismatch` 拒判（第八批修的正是这个）。
- ⇒ 就绪度表 §3 的 A/H 支此前取自一条 **J4 判据本身会拒收**的链路。已在 required 链上对默认基座重测（新件 `reports/taiji_cap0_exit_health_v3_required_chain_20260920.json`，`git_head=983c0001`）：

| 读数 | 裸链（旧退出件） | required 链（本次重测） |
|---|---|---|
| A05 原始输出随消融改变 | true | **true** |
| A05b 表层回答随参数改变 | false | **false** |
| H01 / H02 / H04 | 0.2299 / 0.3223 / 163,511 | 0.2437 / 0.4949 / 210,812 |

**结论修正（这是重测换来的，不是外推来的）**：09-18 那次"A05b 在 required 链翻 true"的实测是在 16M-tick `seed_beta.pt` 上做的，**不成立到默认基座 `seed_corpus.pt`**——换链后 A05b 仍 false。就绪度表把 A05b 记为必过欠账因此**保持原判定**，且现在它与 J4 要求的链路一致了。链路换过来还带来一处必须记下的代价：H 读数整体上移（H02 0.32→0.49 s、H04 163 KB→211 KB），所以 §2 的标定只能在 required 链上用。

**仍未闭合的一半**：本次退出材料的 **B–G 分数**取自裸链评价报告，与新的 required 链健康报告不同链——按"判据必须与分数同链路取数"，整份 CAP 需在 required 链上重出一次（评价侧，非 F/H 侧）。

## §3.2 换默认基座的前置事实（读源码得到，待实测确认）

要按裁决把默认基座换成 16M-tick 的 `seed_beta.pt`（`taiji-native-v8`），先要回答"产品默认入口到底能不能加载它"。本轮读源码的结果与既有记载并不一致，必须实测裁决：

- `taiji/model.py:3333-3345`（M2-2i）**已经把旧格式缺 organ 载荷的迁移放进产品源码**：v8/v9 缺 `identity_organ` 载荷 ⇒ 保留刚初始化的器官；**只放宽"缺失"**——载荷只要在场就仍走 lineage 校验（错父本仍然响亮地失败），非旧格式缺载荷仍然 raise。
- 而 `REQUIRED_CHAIN` 里的 `relax_legacy_guard` 指向的是 `probe_taiji_cap0_legacy_load.py::_install_legacy_guard`，它的 docstring 自述是 "**Process-local simulation of the proposed legacy guard. Source untouched**"——即它模拟的是**提案**，而该提案后来已入源码。
- ⇒ 推论（待证）：对 `seed_beta.pt` 而言这个补丁现在可能是**冗余**的。若实测确认裸 `SeedRuntime.load(seed_beta.pt)` 就能加载，则"把 16M-tick 设为产品默认"不需要把任何测试期补丁搬进产品路径；若仍加载失败，则换基座前必须先把失败的那一环作为**迁移（非放宽）**补进产品源码，并配一支按设计会失败的测试。
- 另一处待实测：`constrained_decode` 是评价链路的第二个开关，它决定"原始字节能否解码成文本surface"。它属于表达层，与 R2 欠账同侧——换默认基座不能顺带把"产品默认启用约束解码"当成既成事实，需要单独呈报。

## §3.1 顺带登记：仓库根目录有一份从未提交的冻结件

`plans/reference/M5_P5_2A_PREDICTED_EXECUTION_PREREGISTRATION_20260913.md`（P5.2a 预测驱动执行预注册，自述"冻结日期 2026-09-13"）在 `git log` 中**无任何提交记录**，一直是未跟踪文件，而就绪度表 §8 的依赖行引用"P5.2a/b→P5.2d"。冻结合同不入版本库 ⇒ 无从判定它在哪个 revision 上冻结、也不能按摘要复算。本轮只登记，未代做提交。

## §5 F04 锚点裁决与评价集 v2（所有者 2026-09-20 弹窗裁决：上限最高项）

**问题**：F04 的复算器读的是 `reports/taiji_cap0_inventory_20260915.json`。换底之后，"默认入口服务的
是训练态"在产品上已成立（新底实测 `tick_after_load=16000000`），但那份 09-15 封存件里读到的仍是
`default_tick=36 / wiring_defect=true` ⇒ F04 继续判 fail。这不是判据错，是**判据锚在一份描述旧产品事实的
封存件上**——与本仓记过的"基线取自封存报告"同型。

**裁决**：评价集升 **v2**，F04 的 `reference` 改指新底 inventory 件
`reports/taiji_cap0_inventory_beta_20260920.json`（已在收口套之后自动重出）。随带纪律：

1. **v1 不覆写**，原样保留为历史判据；v2 只新增一份 manifest，`EVAL_SET_PATH` 指向它。
   07 §4.2 要求判据修订"换版本、换独立测试" ⇒ `test_cap0_eval_set_contract.py` 里钉 v1 的断言
   要么按 v2 更新、要么成对钉（v1 历史 + v2 现行），**不许**把 v1 的 digest/count 断言直接改成
   迁就新文件而不留旧钉。
2. **只有 F04 的 reference 变**；F01–F03 的门文本、`must_show`、`count=4`、`scoring=per_frozen_gate`
   与 B/C/D/E/G 全部题面零改动 ⇒ v2 是"换引用"不是"放宽门"，diff 应当只有那一处路径。
3. 复算器要新增一条**引用时效守卫**（原第三方案的守卫部分随 v2 一并做）：被引 inventory 里的
   `model_reality` 若与来源清单登记的当前默认不是同一份（sha 或 path 不符），F04 判
   `stale_reference` 而不是 fail/pass —— 这样下次再换底会响亮提醒，而不是静默沿用旧底结论。
   配套必须有一支会红的合同测试（把清单指向别的 sha，断言 F04 变 `stale_reference`）。
4. v2 落地后 F04 预期转 **partial 而非 pass**：新底上"输出非固定模板回显"这一子句是否成立仍未测
   （新 inventory 是否仍记 `templated=true` 要读实件），不得因为换底就把 F04 整体点亮。

**主线顺序（同轮裁决）**：R2 保持挂起，先结清收口与剩余项 —— 收口套复跑、F04 锚点 v2、
CAP runner 两处收尾（默认常量引用产品常量、健康支摘要打印的 `contracts` 兜底谎报）、
战役基线对绑 identity 的重出。B/G 40 题人工复核与所有者第 9 项批准仍是 M5 退出的最后两站。

## §6 交接：在飞的两件与恢复点（2026-09-20 05:10 停手时的状态）

**在飞（自动接棒，无需人看）**：
1. 全量收口套复跑 —— 日志 `%TEMP%\closeout2.log`，起跑于换底批提交之后，预期约 19.5 分钟；
   前后各夹一次 `checkpoints/*.pt` 哈希（`stamp=pre2` / `stamp=post2`）。
   **读法**：与 `1857/7` 旧对照比，预期落在 `1857/≤8` 一侧 —— 换底必须随之更新的 3 支
   （启动 smoke、隔离合同、preflight 断言）本轮已修，6 支既有红（5 支结构性增长门 +
   stop_reason 审计面守卫）不在本批范围。**若复跑出现这 6 支之外的新红，先当成本批引入的
   回归处理**，不要去解释成既有红。
2. 新底 inventory 重出 —— 等 (1) 的 `stamp=post2` 出现后自动跑，日志 `%TEMP%\beta_inventory.log`，
   产物 `reports/taiji_cap0_inventory_beta_20260920.json`。它同时是 §5 评价集 v2 的 F04 新引用件。
   注意其 CLI 参数 `--report` 未经核实用过（沿用 cap0 系列惯例）；若 rc≠0 且报未知参数，
   先看 `--help` 再重跑，不要改脚本。

**已按裁决排队、尚未开工（下一轮的第一件）**：§5 的评价集 v2 全套（含 `stale_reference` 守卫与
其反向测试）—— 它依赖 (2) 落盘，别在 (2) 之前动 `cap0_eval_set_v1.json`。

**不要顺手做的事**：不覆写 v1 评价集与任何封存报告；不把 F04 在 v2 之前改判；不在 (1) 跑动时
改 `scripts/training/eval_taiji_cap0_baseline.py`（该 runner 逐项起子进程重新从盘上导模块，
中途改码会把两版代码混进同一份报告）；不动那份未跟踪的 P5.2a 冻结件（不属本批，来历待所有者定）。

**M5 退出的两站没变**：B/G 40 题人工复核（清单已就绪，机器不能替代）+ 十项清单第 9 项所有者批准。
在此之前本退出保持 `blocked`，不宣称阶段完成。

## §4 这一轮没有推进的事

R2 主线仍按 09-20 裁决挂起（排除登记保留在账）；P3b 二次战役、40 道 B/G 人工评分、链路 (a)/(b) 独立待决不变；Mini 继续后置。F 维判 pass **不**表示模型具备语言能力——F01 证的是表示门这台仪器的判别力，CAP 的语言维度仍由 C/D/E 的 0.0 说话。
