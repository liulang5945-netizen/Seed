# N 主线完成度对照（2026-10-11，台账 ㊵-673…675 之后）

**这份件回答一个问题**：目标"按计划推进到内容与目标完成"离达成还差什么。判据不另立，
一律取 09 自己写的两处：**§2 各节的「出口交付」**与 **§1.3 的 VISION 条款对账表**（后者由
`scripts/training/audit_taiji_vision_clause_traceability.py` 量，读数件
`reports/taiji_vision_clause_traceability_20261011_after_v41.json`）。

**取法（先说，免得被当成"我看过了"）**：每条要么给出**在盘件名**，要么给出一条**可复算命令**；
拿不到就写"未兑现"，不拿"落了很多件"顶替"消化了很多条"。状态分三档：
`已判定`（出口判据有值且方向明确）／`未兑现-可自办`（我这侧不动产品码、不花新算力就能做）／
`未兑现-需批文`（动 `taiji/`、跑新权重、开新算力，或改判据形状）。

## 1. §2 各节「出口交付」逐条

| N | 出口交付原文拆项 | 在盘证据 | 状态 |
|---|---|---|---|
| **N1** | ①S5 判读报告入库 | `plans/reference/PLAN-N1-00_S5_ADJUDICATION_20261007.md`（J1＝209＞118 不成立，按预注册路由 P1 降级为部分成立） | **已判定** |
| | ②S1 后的再激活读取通路＋三器官读数 | 只有阶段0（H-S1a，零训练）跑完并判 `J-S1a 不成立`（`PLAN-N1-01_ADJUDICATION_20261008.md`，225＞118，但 F1 0.549→0.747、F3 ×20.8 ⇒ 登记为"读取可修、不充分"）；**阶段1（再激活重放＋训练＋三器官）未跑** | **未兑现-需批文**（阶段1 要训练机时＋产品码） |
| | ③实施规格登记 §18.4「生产者→实际消费者→更新来源→恢复归属」映射 | ㊵-679 起这张表**有在盘物**：`plans/reference/PLAN-N1-03_chain_and_roles_prereg_20261011.md` 的 §18.4 四列映射表（五角色 Q/E/M/W/R 逐格点到实名，14 枚实名已 `git grep` 点验 rc=0、命中文件数 1–47）；但该件是**先冻判据、不开跑**的设计件，② 的实施规格（真接线）仍未开工 | **登记侧已兑现／实施侧未开工-需批文**（此前那句「任何 PLAN-N1-* 件里都没有这张映射表」按本件作废） |
| **N2** | ①一次真实长跑＋离线巩固完整读数 | 通电跑过两次（`PLAN-N2-01`／`PLAN-N2-02` 各带判读件），合取三条＝持久化成立／`J-N2a′` 不成立／`J-N2b′` 不成立 | **已判定**（加速点①在该剂量档判不成立） |
| | ②能力增量／单位经验（速率本体读数） | **已出版（㊵-677）**：`summarize_taiji_n2_experience_rate.py`＋判据件 [PLAN-N2-05](PLAN-N2-05_experience_rate_prereg_20261011.md)，读数件 `reports/taiji_n2_05_experience_rate_20261011.json`＝每 cycle **−0.138577625**、每 symbol −0.017322203、每 text −0.138577625（`degenerate_denominators=["cycles==texts"]`，因批文里 `cycles_per_text=1`）；`J-N2e_3 = cost_not_gain` | **已判定＝每单位经验是代价，不是增益**（分子所在那条 `J-N2b′` 已判不成立，速率不改变该判定，也不得被用来救方向） |
| | ③§8.3 候选版本验收后替换、不覆盖 parent、回退路径真演示 | DEBT-G47 已在真实产品链结清（修法甲落地，第二次通电的原生候选档可被 `SeedRuntime.load` 读回且摘要逐位等于内存态；回退向 0/202 有差） | **已判定**（通道侧到位；代价侧负结果） |
| **N3** | ①容量决定＋对应读数 | 甲（×10 单点）＝`J_N3a′ = not_holds`（`reports/taiji_n3a_j_n3a_prime_20261009.json`）；乙步骤二四张面在场、`J-N3b-在场` 成立／`J-N3b-阈` 不成立 | **已判定（两半都是负结果）**；"封顶在容量还是数据"未答（要 PLAN-N3-10 §4 的 C 形状＝新判据＋新算力） |
| | ②R4 首轮真实生长读数 | 生长**触发侧**打通：τ 现冻 β 0.64／circuit 0.66、150k 三臂 `shadow_forward_hits=149,762`、逐张量比 20 槽＝10 枚权重；但**"准入"这一步从未执行**（`AdaptiveStructuralGrowthController` 在训练链里无消费点，㊵-570 的现读至今未被推翻） | **未兑现-需批文**（准入接线是产品码） |
| **N4** | S3 轨迹态／S4 模式分离的补全与验收（§3.3 槽与路由、重命名／属性交换／干扰／占满四类验收） | 有基线寻址读数（`wrong_top1_rate` 0.06／S4 稀疏 0.33 判负）与产品档寻址尺 `angle_sensitive`；**§3.3 四类验收未做**，S3 的轨迹态未实现 | **未兑现-需批文**（S3 落点在 §5.2 关联记忆候选＝产品码） **〔2026-10-13 ㊵-761 就地换态，原文不删〕** §3.3 四类验收今天**有执行体、有读数件、有判据门**：判读器 `adjudicate_taiji_n4_four_class_acceptance.py`（㊵-757 升版 v2）＋三枚读数件（20261012 基线／`…_after_binding_channel`／`taiji_n4_33_four_classes_20261013_after_attribute_channel.json`）＋契约册 n4_09／n4_10／n4_11 共 **29 支**。四类现值＝`J-N4g-4(a)=holds`（最大档 5000 配对、差 0.0656 对自取带 0.038）／`4(b)=holds_recorded_eviction`／`4-2=not_applicable_no_independent_object_identity`（绑定通道的 `J-N4h-4` 已 holds，但那一支量的是寻址尺上的对象身份，通道存在不代答验收）／**`4-3=holds`（属性位 ㊵-757 落地，两枚探针同场：换 `attribute_ids` 顶位互换、按 cue 那张面分数逐位不动）**。⇒ 本行上方那句「§3.3 四类验收未做」自本注起过期；**仍未兑现的是 S3 轨迹态**，那条仍落在产品码＝owner。另记一句口径：四类的绿只授权**可判性**，不授权任何能力涨点（冻件 `PLAN-N4-12` §6 第三条）。 |
| **N5** | 首轮真实生长全链**五面**（出生／消费／贡献／保持／资源）＋§10 唤醒调度约束 | 出生＝`candidate_id`＋`birth_anchor`；消费＝`shadow_forward/learn_hits`；贡献＝`J-N5b-4 = not_holds`；保持＝`J-N5b-5 = cost_persists`；**资源＝`PLAN-N5-01` §1 行④ 写了"资源面＝参数量/激活"，但这一列从未取数**（两臂参数量相同只是逐张量比较的同名槽副产物，不是设计内读数）；合取 `j_n5b_6 = not_established`、`unmeasured=[]` | 资源面＝参数量/激活**已出版（㊵-678）**：`summarize_taiji_n5_resource_face.py`＋
`reports/taiji_n5_10_resource_face_20261011.json`——三臂 `shared_numel_delta = 0`（同名集合上参数量**没长**，
即准入门从未真的加过参数）、影子容量 `unit_count = 97` 三臂相同、激活快照 8,734 枚状态张量／
非零 857,662 个元素，**最大绝对值治疗臂 10.920937 对控制臂与第三臂 11.059344**（⇒ 激活这一列第一次有三臂对照）。
取法上堵掉一个坑：`predictive_context.recurrent.usage/age` 这类**计数器**（实测 max 149,998）原先被名字前缀
误并进"激活"，现已分成 `counters` 桶单独出版，仪器测里钉住"计数器不许顶成激活幅度" |
**五面齐（出生／消费／贡献／保持／资源都有在盘读数）；出口判据＝不成立**（四枚成员全有值，㊵-673） **〔2026-10-11 ㊵-688 补上第二半，本行此前只写了五面〕** 09 第 184 行的后半「唤醒行为遵守 §10 约束」今天第一次有读数：`PLAN-N5-12` 四条判据全部出值——J-N5l-1 `scheduler_not_in_loop`、J-N5l-2 `noise_changes_scheduling`（13→15）、J-N5l-3 `cap_absent`、J-N5l-4 `arms_separable`（三条判不成立）。两句都不许写：不许读成「§10 唤醒约束已过」（过的是『展开并判』），也不许读成「唤醒机制无效」（VISION §10 的次序是「先证外部明确触发的学习有效，再学习调度」，那条序还没走完，接线属 owner＝G108）。另登记 G109：J-N5l-2 的幅度口径可被挑选，改正件＝`PLAN-N5-13`（冻 J-N5l-6＝内生梯最坏情形）。 **〔2026-10-13 ㊵-761 追加：VISION §19.6 生产者这条线第一次有真长跑读数，但出口的合取仍未成立〕** ㊵-740～743 把「自生成内容进不进经验场」做成产品入口（`Taiji.record_generated_prediction`，**签名里没有 reward 参数**、非零奖励在构造期 `ValueError`），㊵-753／755 起批次 C 长跑器挂上旗标 `--generated-feed-every`（默认 0＝关），**治疗臂已跑完 8,900,000 符号**：`generated_write_count=178000`、`generated_row_weight_max_abs=1.379441381` ⇒ `J-N5r-2` 两支第一次同真；`measured_symbols_per_second=241.952872`、`eta_hours=10.218` ⇒ `J-N5r-3` 只认本次实测。**仍不成立的两件**：①`J-N5r-4`／`J-N5r-5` 缺并排载体（＝新登记的 DEBT-G127，仪器 ㊵-759 已入库、正式读数等对照臂）；②那枚**配对读数无法归因**——同点 2,000,000 处治疗臂 `cumulative_mean_surprise` 7.1682 对对照臂 6.1814（第一窗 250k 就已是 7.2657 对 6.2757），而 `cumulative_online_accuracy` 只差 0.006963（0.272093 对 0.279056），单变量被证明之前这句不许写成机制（㊵-758 ⑦）。另：`generated_reward_sum=0.0` 在批次 C 两臂件里**没有依据自述**（那两枚键 ㊵-756 才加进跑器），真累加仍欠 `taiji/memory.py` 第四处产品码＝DEBT-G121（owner）。 **〔2026-10-13 ㊵-775 就地换态：上面"仍不成立的两件"自本注起过期〕** 对照臂终盘（`status=completed`、8,900,000 符号、`feeds=0`、`generated_write_count=0`、行权重 `0.0`）之后：①`J-N5r-4`／`J-N5r-5` **都有并排载体并出值**＝并排仪 `verdict=holds`（`differing_config_keys` 为**空**、`differing_flag_keys` 恰是旗标派生那四枚、两臂跑器码版同为 `9299352d4…`）＋磁盘普查 `verdict=full_face_two_arm_pair`（两臂各 2/2 可载入）；②那枚 surprise 差的**单变量前提被证明了**（`J-N5t-1` `holds`：两臂各 36 行、边界键逐行相同、每臂单一 session），但**归因仍未成立**——`J-N5t-2` 的臂↔臂那一半被名集不齐挡住（新登 **DEBT-G133**），定位句被自家资格门拒判（实测 `verdict=rejected`、`failing_verdicts=["both_reports_measured"]` ⇒ DEBT-G131 的前置换成 G133 修法①）；③**第一次有能力面配对读数**＝`PLAN-N5-22` 的 `J-N5u-1` 七条全等（同 `git_head`、同评测集 sha、同链路、同判分纪律、双未训练、逐维无错）⇒ 可配对，入场维 {C, D, E}，逐题翻转只有 D 一动（治疗臂独赢 `D04`、对照臂独赢 `D02`／`D07`，`delta=-1`）⇒ **`direction=not_resolved`**。⇒ 本行的出口判据从"缺载体"换成"三条句子各缺前置"：能力幅度句缺同配置重复臂（owner 机时）、机制句缺集合差（G133→G131）、方向句只到"读平"。 |
| **N6** | 工程债随实施项清（不单独立项） | 债表现至 **DEBT-G103**；㊵-680/681 结清 G96（现行面 `black --check .` 已跟踪残差＝0）与 G97（修法①②③全部做完：普查仪入库＋四条先冻设计件＋存在性门），新挂起 G98–G103 | **按定义持续中** | **〔2026-10-11 ㊵-688 就地换态，原文不删〕** 债表现至 **DEBT-G109**；㊵-681…687 又结清／推进了 G96–G100（三笔产品码落地，G98 停在 `partially_closed`、G100 停在记录层）、G107（仪器：26 格里 10 枚点错名，本轮修完）、G106③／G105③（复算已出值：前者把「没人调用它」否证，后者只做 half 并登记缺口）；新挂起 G101–G109。本行上方那句「债表现至 DEBT-G103」自本注起过期。〔2026-10-11 ㊵-708 就地换态，原文不删〕债表现至 **DEBT-G112**——现取＝`python scripts/training/audit_taiji_debt_table_column_shape.py`（`max_debt_id` 112／`rows` 112／`distinct_ids` 112）；㊵-705 挂 G111、㊵-708 挂 G112（`J-N1c-3` 的来源记录只住在 `.pt` 信封里那条）。本行上方那句「现至 DEBT-G109」自本注起过期。**并且这条漂移又发生了一次**：㊵-705 加 G111 时没换这行，所以 DEBT-G110 的第②支（要不要把"状态类文档的 Gxx 必须等于现取 max"做成常设门）今天多一条支持证据——我按该债 ①的裁定**不做批量换态**，只在这一处状态面就地换。 **〔2026-10-13 ㊵-761 就地换态，原文不删〕** 现取＝`python scripts/training/audit_taiji_debt_table_column_shape.py`（**`rows` 127／`distinct_ids` 127／`max_debt_id` 127**、`cells_per_row_escape_aware` 只有 `{"5": 127}`、`rows_not_closed` 0）。㊵-744…760 这段结清／推进了：**G101／G119／G123**（债表并形真正落盘＋普查器缺省开守门）、**G120**（台账圈号唯一性普查仪入库并消掉一枚已入库的重号）、**G122**（八裁⑴ 甲：对象绑定通道＋属性位两半全落地，DEBT-G122 结清）、**G124**（主线两道整树 lint 面，此前我跑的面≠CI 跑的面）、**G125**（并形器与圈号普查仪各补一册，字节守恒复核器从仓外一次性脚本变成入库仪器）、**G126**（长跑器 `--help` 的 GBK 崩溃面）、**G115 的修法②**（入库 `reports/` 件的行尾棘轮门，㊵-760）；**新挂起 G124…G127**，其中 **G127** 是这一轮最重要的那条：批次 C 的 `J-N5r-4`／`J-N5r-5` 一直没有并排出版的载体，㊵-759 补了仪器，正式读数排在对照臂跑完之后。**这条漂移又发生了一次**（本行上一次写 112、现取 127）⇒ 给 DEBT-G110 第②支再添一条证据，但仍按 ①的裁定不批量换其它已冻件。 |

## 2. §1.3 对账表这一维（计划自称的"完整性基准"）

现读（`reports/taiji_vision_clause_traceability_20261011_after_n206_n511.json`，㊵-680 第五次取数）：**41 行条款，归属某个 N 项的 19 条属「现在必须可核验」**；这 19 条里
**按条款号被归属件引用的＝7 条**（V-11、V-15、V-16、V-17、V-19、V-21、V-41），按章节锚命中 **19** 条（＝**全部 19 条都有锚级命中**，仍是上界口径，锚跨文档同形），
两种都没有的 **0 条**。五枚读数件序列（id／锚／unmapped）＝**0/12/7 → 1/13/6 → 3/15/4 → 5/17/2 → 7/19/0**，**〔㊵-685 追加第六枚〕** `reports/taiji_vision_clause_traceability_20261011_after_n512.json`＝**8/19/0**（V-30 由 `section_only` 转为有 id 级引用，故 id 级 7→8；分母 41／required_now 19 仍未动，`every_clause_has_id_level_citation` 仍为假）。⇒ 本段那句「id 级只有 7 条」自本注起过期，应为 8 条。
而分母 `clause_rows_parsed=41` 与 `clauses_required_now=19` 全程未动。verdict 三条：`table_parsed_every_row_has_clause_id` 真、
**`every_required_clause_has_any_citation` 第一次为真**、`every_clause_has_id_level_citation` **仍为假**。
⇒ **§1.3「缺条即不完整」这一条今天按「any citation」口径已满足**；但**不许**据此说「19 条已消化」：id 级只有 7 条，
且命中数上升的直接原因是我写了带条款号的登记节，它只证存在性、不证语义承载（DEBT-G97 的两处假载体取证仍然有效）。〔㊵-681〕这条存在性基准现在由门守着——`tests/taiji_native/test_n6_05_clause_existence_gate_contract.py`；门**故意**两样都不守：不守命中数、也不守 id 级，各配一支负对照。

## 3. 结论：目标未达成，剩余清单分两栏

**未达成项里，需要 owner 的**（我不会自己动）：N1 阶段1（训练机时＋产品码）与其 §18.4 实施规格、
N3 的 C 形状（新判据＋新算力）、N4 的 S3/§3.3 验收（产品码）、N5 的准入门（产品码）、
DEBT-G78、G94③ 与 G90、G101（债表列形漂移，登记不改）、**G102（`.fork-sync/` 既未登记进目录台账也未进 ignore——同一根因让 folder-structure 门红，且是 `black --check .` 的唯一残差；它的归属不是我自证得了的）**、**G103（OpenAPI 金样缺新路由 `/api/tools/web_search`，刷公共金样属他人半径，我不代刷）**。〔㊵-681 更新〕G97 修法③ 已按 owner #25 乙裁落地成门，不再是待裁项；G98／G99／G100 三笔的实施形状已批（丙＝三笔一起批），改产品码由我这侧执行，见 03 队首。〔2026-10-13 ㊵-768 更新〕**G101**（㊵-751 债表并形真正落盘）与 **G103**（㊵-753 按 owner 八裁⑶ 甲重录 OpenAPI 金样）已结清，不再属于本栏；本栏现在加两枚有名有姓的签字件＝**DEBT-G128 余下两件**：①`packages/web/web-search-scraper` 的 Model Experience 主张要由包主写进 `taiji-harness/scripts/verify-package-readme-model-experience.ts` 的 `SENTENCE_MODEL_EXPERIENCE`／`NO_MODEL_EXPERIENCE_SECTION` 审计登记表（或按完整 model-context 条目形重写那一节），②`docs/config-catalog.md`／`.zh.md` 那对内容漂移要么把中英正文一起带齐、要么由造成漂移那次改动的负责人重记——我不靠加白名单让门变绿，也不代签自己没核过的语义一致性。

**我这侧仍可自办的**（按对计划的实际推进度排序，2026-10-11 ㊵-679 之后）：
1. ~~G97 修法② 的余下 2 条~~ **已做完（㊵-680）**＝[PLAN-N2-06](PLAN-N2-06_write_consolidation_contract_prereg_20261011.md)（V-17）
   ＋[PLAN-N5-11](PLAN-N5-11_three_layer_update_architecture_prereg_20261011.md)（V-21），同一台普查仪复算 unmapped 6→4→2→**0**（此前两件＝
   [PLAN-N4-08](PLAN-N4-08_memory_kinds_and_delta_gate_prereg_20261011.md)（V-15／V-16，㊵-676）与 [PLAN-N1-03](PLAN-N1-03_chain_and_roles_prereg_20261011.md)（V-11／V-19，㊵-679））；
   两件交出的不是「登记完成」而是**四条可判缺口**（E 侧无来源校验、E 侧无撤销、来源枚举缺 `generated`、第 7 步无更新对象账本），
   已分别开成 **DEBT-G98／G99／G100** 与新登记的 **J-N5k-3**；
2. ~~N5 的**资源面**取数~~ **已做（㊵-678）**＝`summarize_taiji_n5_resource_face.py`＋
   `reports/taiji_n5_10_resource_face_20261011.json`（参数量未长、影子容量 97、激活快照三臂对照）；
3. ~~N2 的**每单位经验增益**成列~~ **已做**＝[PLAN-N2-05](PLAN-N2-05_experience_rate_prereg_20261011.md)
   ＋`reports/taiji_n2_05_experience_rate_20261011.json`，判成"每 cycle 代价 −0.138577625"；
4. 台账与基线面的持续同步（每批必做）。

**明确不做的**（避免"看起来完成了"式的收缩）：不把"仪器完善"报成"能力涨点"；不拿
`section_citations` 当覆盖率；不在 owner 未批前动 `taiji/`、跑新权重或开新算力；
不为让门变绿而改已冻判据。

**〔2026-10-13 ㊵-761 就地更新剩余清单——上面三段原文不删，只作过期标注〕**

- **owner 侧，两句已过期**：**G101 已于 ㊵-751 结清**（债表 123 行并成一型真正落盘、普查器 `--enforce-shape` 缺省为真、
  执行点是 `test_n5_26_…::test_real_register_is_one_shape_and_greens_under_default_enforcement`），
  **G103 已于 ㊵-753 结清**（OpenAPI 金样按八裁⑶ 授权重录并归绿）——上面把它们列在"需 owner"里自本注起过期。
  仍待 owner：**N1 阶段 1**（训练机时＋产品码）与其 §18.4 实施规格、**N3 的 C 形状**（新判据＋新算力）、
  **N4 的 S3 轨迹态**（§3.3 四类验收已有执行体与读数，见上表 N4 行，S3 那一半仍是产品码）、**N5 的准入门接线**、
  G94③／G90、G98 余 8 处、**G102**（`.fork-sync/` 归属不是我自证得了的）、
  **G104**（张量层接 `generated` 需要受体通道重映射——230 枚 `.pt` 拒载那笔）、
  **G121**（真累加需要 `taiji/memory.py` 第四处产品码）。
- **我这侧剩下的（按对计划的实际推进度排序，㊵-761 现取）**：
  ①**批次 C 正式读数**＝对照臂跑完后跑 `python scripts/training/audit_taiji_batch_c_arm_parity.py --out-report reports/taiji_batch_c_arm_parity_20261013.json`，
  配置面必须只余旗标一项，之后才谈那枚 surprise 差（对照臂现取 04:40:00Z 处 `2,250,000/8,900,000`、325.2 符号/秒；〔㊵-768 现取〕05:44:59Z 处 `3,500,000/8,900,000`、`elapsed_seconds 10818.8`、按它自取速率 323.5 符号/秒 ⇒ 剩约 4.6 小时、约 10:25Z 收尾；进程面原文＝PID 25584 起于 02:44:39Z、命令行 `train_taiji_r2_readout_retrain.py --arms A --symbols 8900000 --generated-feed-every 0 --fresh --out-dir output/n5_20_batchC_control`）；
  ②**J-N5r-5**＝两臂新落 `.pt` 过在册双臂普查器 `measure_taiji_disk_load_face_two_arms.py`；〔2026-10-13 ㊵-768 · **前置取证已做，发表读数仍等对照臂**〕`--only` 点名治疗臂那枚 13,090,253 字节的 `checkpoint.pt` 单跑一趟（写靶在仓外，不入库）：治疗档 `loaded=true`／`encoder_width=5`／`readout_rows=5`／`vocabulary=5`／`receptors_inherited=true`，对照档（退回一枚来源枚举）refused `ValueError|synapse payload shape does not match architecture`——这一对分岔正是双臂普查器该有的形状，所以「新件拒载＝批次 B 的迁移支没覆盖长跑器写出的形状」那一支在**文件层**已被否证。仪器自给 `verdict=partial_selected_lt_face`（selected 1 < 面上 13 枚、`batch_mode=only_named`）⇒ 正式那一趟要把**两枚**都点名进来才谈覆盖，本格不是发表读数；对照臂那枚 `.pt` 现在不读，因为它每约 250k 符号重写一次，读到半写信封只会造出一枚假 refusal。
  ③**整族回归**＋`test_openapi_snapshot` 面复跑补基线（对照臂在飞期间不跑——争用会把计时类断言污染成不可比）；
  ④**DEBT-G115 修法①**＝其余仪器写靶逐枚现读确认再补 `newline=`（粗筛上界 82 枚；索引面今天 1783 枚里只剩 2 枚豁免件，
  修法② 那道门已建＝㊵-760）〔2026-10-13 ㊵-767／768 · **这一条已分两半，粗筛那句「82 枚」不再是工作面**：㊵-762 把粗筛换成按调用点分桶的普查仪 → ㊵-765 收窄出 D 桶（仪器默认落点指向 `reports/` 的那一撮）＝20 处／18 枚 → ㊵-767 把这 20 处逐枚补上 `newline="\n"`，D 桶现读 **0 处／0 枚**。**残余**＝B 桶 1040 处／766 枚静态不可判的写靶（要逐枚现读才知道有几枚真写入库面）＋修法③ 的统一落盘助手未做；索引面棘轮现读 1784 枚、豁免仍是那两枚同名件〕；⑤**DEBT-G127 结清**（①②③做完才结，仪器本身已入库）；
  ⑥**DEBT-G131 出数**＝仪器与契约册已于 ㊵-771 入库（`audit_taiji_batch_c_moved_face_sets.py`，两趟 base↔臂读数的动名集合差），
  只差两臂终盘件；**DEBT-G130 出披露键**要改跑器文件，正被在飞那臂持有 ⇒ 同排在臂后。〔2026-10-13 ㊵-775 · **①②③⑥ 就地换态**〕①批次 C 正式读数**已出并入库**（并排仪 `holds`、配置面差为空）⇒ **DEBT-G127 结清**；②`J-N5r-5` **已出值**＝`full_face_two_arm_pair`；③整族**已跑**——唯一一枚红是 `test_n5_31` 的定价夹具（`eta_hours` 被 3 位小数舍成 0.0 ⇒ 那条"必须为正"成了不可假的断言），修法是把预算改成**由本次实测速率推**，判据 `J-N5r-3` 一字未动，改后该册 9 passed；⑥**DEBT-G131 未结清**，它的"等两臂终盘"换成"**等 DEBT-G133 修法①**"（两趟 base↔臂补齐后集合差仪实测 `rejected`，起因是治疗臂信封多两枚 `memory.last_event`）；⑦DEBT-G130 的披露键**不再被在飞臂挡着**，与 ④⑤（G115 修法③）并排。新增自办项 **⑧＝DEBT-G133 修法①**（比较器的名集差窄判据面＋两支负对照；它同时解锁 ⑥ 与批次 C 的机制句）。
  〔2026-10-13 ㊵-769／770／771 追加的判据面〕批次 C 的**归因**判据已先冻成
  [PLAN-N5-21](PLAN-N5-21_batch_c_attribution_prereg_20261013.md)（`J-N5t-1` 两臂进度流同序同界、
  `J-N5t-2` 冻结面自证、`J-N5t-3` 可比性三件并立、`J-N5t-4` 幅度需噪声地板），
  且 §2 的原文在取到任何臂↔臂读数**之前**被我自己更正两次（§6 把计量面摘出判决、
  §6.1 把 `J-N5t-2` 从"机制定位"**降级**为"冻结面自证"）——起因是拿真面分档跑了一遍：
  基座↔治疗臂 62 枚差异槽＝权重 54／拓扑 4／计量 4，而"权重"里点名的
  `memory.association.edge_weight`、`identity_organ.bank.prototypes`、`cognitive_state.world.latent`
  都不在冻的三类面上 ⇒ 照原文跑会在正常读数上误报"出现第四类"。降级之后**方向句也要等
  `J-N5t-1`＋`J-N5t-3` 齐绿才许说，定位句要等 ⑥，幅度句被 `J-N5t-4` 挡着。**
  同轮还清掉两枚自办债：DEBT-G129（逐张量比较器形状不齐时 rc=1 崩且不落件，已改成点名出版＋划分，
  件名升 v2，册 7→10 支）、DEBT-G115 修法① 的 D 面（18 枚仪器逐枚补 `newline`，D 20 处→0）。
- **本件自身的现态**（㊵-771）：结论**未变**——目标未达成，且未达成的原因不是工没做，
  而是能力项（N1 阶段 1、N3 的 C 形状、N4 的 S3 轨迹态、N5 的准入门、批次 C 的判读）
  要么等 owner 裁定／机时，要么等对照臂终盘（约 10:25Z）。本件最近一次换态是 ㊵-761，
  本轮只把"我这侧剩下的"这一栏补到 ⑦ 条并就地标注降级与结清，不动 §1／§2 的判定行。
- **本件自身的现态**（㊵-775）：结论**仍未变——目标未达成**，但"未达成"的形状换了。批次 C 的**载体类缺口已清空**（`J-N5r-1…5` 五支有值、`J-N5t-1` 有值、能力面第一次可配对），剩下的三条句子各缺**前置**而不是缺仪器：能力幅度句缺同配置重复臂（owner 机时，`J-N5t-4`／`J-N5u-4` 双重挡）、机制定位句缺 DEBT-G133 修法①（我自办，下一格）、方向句只到 `not_resolved`（D 维 `delta=-1`，阈值 3）。owner 侧照旧：N1 阶段 1、N3 的 C 形状、N4 的 S3 轨迹态、N5 的准入门、G121、G104 批次 B、G98 余 8 处、G102、G94③／G90、`m7-ci` 余下两件签字，外加 CAP 的 B／G 人工复核。本轮把 §1 的 N5 行与本节剩余清单就地换态，§1／§2 的判定行原文一字未动。

- **本件自身的现态**（㊵-780）：结论**仍未变——目标未达成**，但形状有三处移动，都不改结论。
  ①载体类缺口清空后，三条句子各缺**前置**而不是缺仪器：机制定位句已**有据**（㊵-777 的 `taiji_batch_c_moved_face_sets_v2_20261013.json`
  ＝只在治疗臂相对基座动过的名字 9 组，归类只有 `identity_organ` 与 `memory` 两面），但按冻件 §6.1 第 2 条它只说名字、不说幅度与因果；
  幅度句等同配置重复臂的地板（**在飞**：13:08:08Z 起跑，13:53:19Z 处 750,000／8,900,000，本次实测约 277 符号每秒）；
  B／G 两维等人工裁定（两份一页纸已于 ㊵-778 交出，勾选权在 owner）。
  ②十裁批了四件产品码与一笔机时，四件的判据**已全部冻完**＝`PLAN-N5-24`（回读形状，且"计数为 0 不许出版 0.0"）、
  `PLAN-N5-25`（受体通道重映射，旧件必须响亮拒载＋自带负对照）、`PLAN-N2-08`（八处来源**成批**取消默认值＋开放标签不许被拒）、
  `PLAN-N4-13`（S3 轨迹态，并明写它够不到的那半句要等 N1／S1）——**但没有一件开工**：`G-N5v-1` 要求重复臂两臂与批次 C 同跑器码版。
  ③因此下面"**仍守的三条**"里第三句（"不在 owner 未批前动 `taiji/`、跑新权重或开新算力"）今天仍然成立，只是成立的原因换了：
  批文**已到**，不动的原因是**窗口约束**与轮次预算，不是缺批准。离"目标达成"最远的仍是能力主张那一栏：
  N1 阶段 1（它还挡着 `PLAN-N4-13` 的后半句）、N3 的 C 形状、N5 的准入门、G102、G94③／G90、`m7-ci` 两件签字、CAP 的 40 题。

**仍守的三条**：不批量改已冻预注册里的落笔时点陈述（G110 ①的裁定，本件自己也照此办理——旧句只标过期、不删）；
  不把 §1.3 的锚级命中说成 id 级覆盖；不在 owner 未批前动 `taiji/`、跑新权重或开新算力。

- **本件自身的现态**（㊵-787）：结论**仍未变——目标未达成**，本轮有四处移动，其中两处是我自己的错被纠正。
  ①**`m7-ci` 的 doc-sync 半面第一次在 CI 上确认转绿**（run **38431031552**，HEAD `435d7877e`，step 表 `✓ doc-sync (43 leaves)`）＝十一裁 ①② 那两件签字落地；
  但同一 job 的 `hygiene (18 leaves)` 第一次被执行就露出**两枚旧红**（上一趟 38420482505 的 step 表里它显示 `-`＝被顺序遮挡，所以是 newly visible 不是 newly caused）。
  一枚（`package invariants`）已按十二裁 ① 修掉并做第三次 `--write`；另一枚（`default product isolation`）**没修**，
  取证后登记 **DEBT-G136** 并出第二份一页纸＝`plans/reference/HARNESS_DEFAULT_PRODUCT_ISOLATION_OPTION_SHEET_20261014.md`——
  门声明的唯一豁免路 `OPTIONAL_BUNDLES`（`packages/boot/app-boot/src/profile.ts:188-193`）其实已经点名那枚 bundle，
  而遍历腿不查名单、`spec:133` 又用 `it.each` 把三个 section 全算"默认产品包含"，两条主张互相矛盾，所以这是排序与语义裁定，不是我能自办的缺陷。
  ②**CAP 的 40 题载体原本发错了我自己写的量表**：我给 owner 的表写"每题 0／1"，而已冻约定是 B 维 0／1／2 三档、G 维五类判定
  且"硬安全失败必须为 0"（`scripts/training/eval_taiji_cap0_baseline.py:769-776`）。旧件就地打作废横幅（原句不删），
  现行载体改由评测器自己的 `--worksheet` 生成（`plans/reference/CAP_B_G_HUMAN_REVIEW_V3_{TREATED,CONTROL}_20261014.md`）。
  同时按 `--adjudicate` 现读两臂都是 `B scored 0/20`、G 三个计数全 0——**这三个 0 只读作"规则层对这些输出判不了"，不许读作"没有服从越权"**。
  ③**十一裁 ⑷ 的件已出**＝`plans/reference/N1_S1_PHASE1_AUTHORIZATION_OPTION_SHEET_20261014.md`：三条可批形状里要特别记住
  丙（补 N1 判定面）**不能报成零算力**——`DEBT-G62` 已否证那句（`taiji/model.py:3190` 的 `generate(...)` 13 个参数里没有任何"喂入指定字节序列"的口子）。
  ④**本件的姊妹件 03 头部今天被我自己更正一处**：上一版写过"这两臂的速率就是地板的量，负载会把它压平"，
  现读 `audit_taiji_n5_23_repeat_arm_noise_floor.py:31` 表明地板两把尺是 `cumulative_mean_surprise`／`cumulative_online_accuracy`，都不叫速率；
  不在臂在飞时跑整族的真实理由是**争用会造超时红**。这条更正不改任何判据，只是把"我为了解释而编的因果"撤回。

  **下一格（照抄 03 头部，两臂齐之后）**：①地板 `audit_taiji_n5_23_repeat_arm_noise_floor.py`（四枚 `run_report.json` 齐备才出版，资格不齐 ⇒ `rejected`）；
  ②两臂各一趟 CAP（同 `git_head`＝中间不许提交，`J-N5u-1`）＋配对器＋地板仪 `--cap-pairing-*`；③地板成形后才动幅度句（阈值 3 与 `J-N5t-4` 一字不动）；
  ④整族回归＋`test_openapi_snapshot` 面复跑。**我这侧剩余可自办**：DEBT-G115 修法③（统一落盘助手——本轮 `_write_report` 又新写出两枚带 CRLF 的入库件，正是它的活证据）、
  DEBT-G130 跑器披露键（动跑器 ⇒ 排臂后）、DEBT-G132 的 22 处暴露面（半径在别行，只登记）。
  **owner 侧**：CAP 40 题在新载体上逐题回填、DEBT-G136 的甲／乙／丙、N1 阶段 1 的裁（十一裁 ⑷ 件已出）、G104 批次 B（十一裁 ③＝排在重复臂结清之后）、
  `m7-ci` 的 hygiene 面余枚、G121／G98 余 8 处／G102／G94③／G90 照旧。
  **不许从本节读出的三句**：不许读成"`m7-ci` 全绿"（hygiene 仍 16/2，且本轮那枚修复未经 CI 确认）；不许读成"重复臂出了地板"
  （治疗臂现取末行 3,000,000／8,900,000、`written_at_utc=2026-10-13T16:04:25Z`，对照臂目录尚未出现）；不许读成"CAP 的 B／G 已裁定"（现在是"量表已改对、载体已重发、等人回填"）。

## 4. 会话末交接（2026-10-14，㊵-792／793／794 三格，HEAD 2db8f9c10）

1. **两臂在飞时不许再下臂命令**。串接链 `run_repeat_arms_776.sh` 会在治疗臂 `rc=0` 之后自动起对照臂，
   失败即停的守卫是 `ARM1_RC≠0 ⇒ 写 ABORT_ARM2_RC1 并 exit`。判"跑完没有"只认三样（㊵-792 ④）：
   该臂目录下的 `A/run_report.json`、`arms776/state.txt` 里的 `ARM1_RC=`／`ARM2_RC=`／`CHAIN_DONE_UTC=`、
   以及件里的 `status`／`reached_budget`。**目录名逐字核对**：本次重复臂是 `output/n5_20_batchC2_treated`
   与 `output/n5_20_batchC2_control`，与**已完成的批次 C**（`output/n5_20_batchC_treated`／`_control`）
   只差一个 `2`——读错目录会拿"已完成"的旧件当新臂的收口证据（㊵-792 ⑤ 已踩过一次）。
2. 取地板照 03 头部①那条逐字命令（四个旗标名与两枚 batch C 路径的可跑性已在 ㊵-791 用真实件干跑验过）。
   预注册那一支不许绕开：**地板为 0 ⇒ 两支逐指标都落 `no_spread_unjudged`、不发表幅度句**，也不许换指标救它。
3. CAP 两趟必须**同 `git_head`、两趟之间不许提交**（`J-N5u-1`），命令**不传 `--dimensions`**
   （`eval_taiji_cap0_baseline.py:59` 的 `DRIVEN_DIMENSIONS` 才是与批次 C 同面；`:60` 的
   `NOT_EXECUTED_DIMENSIONS` 同件出版 ⇒ 读数件里 8 个维度键不等于 8 个跑过的维度）。
   两枚输出件（`taiji_n5_23_cap0_repeat_treated_20261014.json` 与
   `taiji_n5_23_cap0_repeat_control_20261014.json`，落点由 03 头部②那条 `--report` 指定）
   **今天仍不存在** ⇒ 引用在册普查把它们记成
   `planned_output`（㊵-793 那格就是被这台面的自指咬到的：在台账里逐字写出缺席件的完整路径，
   普查仪会把"报告过一次缺席"读成"引用了一份读数"）。
4. 窗口内可自办三件（03 头部⑤）：DEBT-G137 的**普查面扩到 `tests/`**（会把 37／36／22／21 那组**下界**
   往上推 ⇒ 属改判据，要先冻再动）、DEBT-G115 修法③ 的迁移队列（进度只认 E 桶的"真收口"计数，
   要现值请跑 `audit_taiji_newline_write_face.py`，别抄台账里的历史数）、DEBT-G130 跑器披露键
   （**动跑器 ⇒ 必须排在两臂结清之后**）。
5. 待 owner 的四件：DEBT-G136 的甲／乙／丙、CAP 40 题在 V3 载体（
   `plans/reference/CAP_B_G_HUMAN_REVIEW_V3_TREATED_20261014.md` 与同目录 `_CONTROL_` 那份）上逐题回填、
   N1 阶段 1 的裁（一页纸 `plans/reference/N1_S1_PHASE1_AUTHORIZATION_OPTION_SHEET_20261014.md` 已出）、
   G104 批次 B（十一裁 ③ 已排在重复臂结清之后）。
6. 门的形状本批改过一条，复述以免下个会话误判：`test_n6_06` 的两支测现在都只写 `tmp_path`，
   **跑套件不再改写仓内那枚 episodic-forgetting 封存读数**（㊵-794），所以跑完测
   `git status -- reports/` 应当只剩那 6 枚 autocrlf／stat 幻影件；若又多出一枚**内容真变**的件，
   归因顺序是"谁写过它"（查本轮跑过的每支测／门的落点），不是"谁改过它"。
7. 本批入库三笔且都已推送：`cb5295bf0`（㊵-792／793：臂链存活实证＋自指命中改回件名形，
   点名更正 `ee931faf3` 那次"带着红测入库"）、`8e5ee71ad`（docs(reports)：把扫描面读数 2151→2162 刷进库）、
   `2db8f9c10`（㊵-794：跑测不再刷写在库件＋DEBT-G137 补记普查面盲区）。

## 5. 会话末交接刷新（2026-10-14 晚，㊵-808／809／810）

§4 那七条**仍然有效**，本节只补它落笔后变的两件事，不重写 §4（过期的是处方而不是结论，故就地打日期戳）。

1. **臂后链已交给一个脱离会话的取证驱动**：`seed_scratch/run_post_arm_chain_812.sh`（仓外），日志
   `seed_scratch/chain812/log.txt`。它只在 `arms776/state.txt` 出现 `CHAIN_DONE_UTC=` 且 `ARM1_RC`／`ARM2_RC`
   均为 0 之后动手，再用谓词 `status==completed 且 symbols_consumed==symbols_budget` 复核两臂终盘；
   之后按 §4 第 2／3 条的顺序跑：并排仪 → 地板 → 两趟 CAP（每趟前后各取一次 `git rev-parse HEAD`，
   四个值不全等就跳过配对并记 `HEAD_MOVED`）→ 配对 → 带 CAP 的地板复算。
   **下一格先读日志再决定要不要自己起命令**：日志里已有 `STEP=<名> rc=` 的那步不要起第二遍；
   驱动只取证、不判读，所以它产出的新落点件在判读之前不许引用，也别在 CAP 两趟进行中提交。
2. **地板仪自己已经改过两处**（㊵-809），§4 第 2 条的命令一字未动，但件里多了两句新东西：落盘收口到
   统一助手，以及 `reading_limit` 现在写明「拒收不合格配对件 ⇒ rc=2、不落半件」。所以若某趟地板报
   `refuse_input`，那是 CAP 配对件资格不齐的**正确**出口，不是仪器坏了。
3. 现取（2026-10-13T20:28Z，逐字来自 `output/n5_20_batchC2_treated/A/progress.jsonl` 末行）＝
   `symbols_consumed=7,250,000／8,900,000`、`written_at_utc=2026-10-13T20:25:10Z`、进度 29 行 ⇒ 平均
   276.5 符号每秒（7,250,000／26222.481），治疗臂约 22:05Z 收口，对照臂其后约 8.9 小时（≈07:00Z，10-14）。
   §4 第 1 条的「不许再下臂命令」依旧成立：串接链活着（`ARM1_START_UTC=2026-10-13T13:08:06Z`）。
4. 本批入库并已推送：`bf1721794`（㊵-808＝第 10 枚仪器收口＋把臂后链交给驱动）、`d181d4266`
   （㊵-809＝地板仪收口＋自述句）。§4 第 7 条那三笔在它自己的时刻仍为真，不与本节互斥。

5. ㊵-811 起臂后链还有**第二段**：`seed_scratch/run_post_chain_stage2_811.sh`（日志 `chain811/log.txt`）。
   它等第一段写下 `CHAIN812_DONE` 之后，按 DEBT-G85 口径跑队首 ④ 的整族基线面
   （`python -m pytest tests/ -q`，不带 `-X utf8`，rc 由同一条命令写进日志末行），并在跑前后各快照一次
   `git status --porcelain -- reports/`。第一段若以任一路径中止（臂 rc 非 0／地板拒收／CAP 失败／
   HEAD 移动／配对拒收／等待超时），第二段直接跳过整族并留 `exit 3` 痕迹。同样**只取证不判读**：
   日志里的 rc 与汇总行要由读到它的那一格自己判，"跑完了"不等于"绿了"。

6. ㊵-824～826 三轮的落点（2026-10-14T02:43Z 现取，HEAD 本地 `dfda464e6`）：**地板这条线已经收口**——`PLAN-N5-26` §2 探针实测换 seed 连基座都装不上（`ValueError: receptor channel map does not match architecture`，机制在 `taiji/organs.py:99-104`／`:162-165`），owner 十四裁 ⑴ 因此择丙＝承认可测不到⇒ `PLAN-N5-23` §5 已把「等第二个档」这句假设关掉，批次 C 的幅度句是「无地板可依」不是「差被地板挡住」，两种说法不许混写；那 ≈15.2 小时**不再花**。DEBT-G136 择乙并已落盘（豁免＝两条具名边，本机门 rc=0、该册 76 passed、负对照 `1 failed | 75 passed`），**但结清要的是 CI 面 `hygiene (18 leaves)`＝`18 passed, 0 failed` 加 run 号与时刻**。
7. **本轮 `git push` 两次都失败**（02:42:51Z 与 02:43:02Z，`Could not resolve host: github.com`）⇒ 现状＝`main...origin/main [ahead 1]`，`dfda464e6` 只在本地。下一格的第一件就是：先 `git push origin main`并确认 rc=0，再去读那一条 CI（推不动就先登记「未推」，不要改成「已推」）。臂侧读数仍由两个仓外驱动取证（`chain812/log.txt` 现读不存在＝它还没等到 `CHAIN_DONE_UTC`；对照臂 `state.txt` 末行现读到 `ARM2_START_UTC=2026-10-13T22:03:29Z`）⇒ 先读日志再决定要不要自己起命令。

8. **本节第 6、7 条有两处已过时，按 04:16Z 现读取代**：网络已恢复——02:49Z／02:55Z／03:48Z／04:08Z／04:16Z 五笔 push 全部 rc=0，`main...origin/main` 同步，第 7 条那句「ahead 2、未推」作废。DEBT-G136（修法乙）与 DEBT-G128 的 hygiene 半枚**已在 CI 面结清**＝run 38503603812（head `6b3427f88`，step 表 `doc-sync (43 leaves)`＝success、`hygiene (18 leaves)`＝success，03:06:05Z 完成）；判读时要逐 step 看——更早那趟 38503009999 是**被并发组取消**（doc-sync cancelled／hygiene skipped），它既不是红也不是绿。台账已到 ㊵-831；本轮线上新增在册件＝`scripts/training/audit_taiji_test_write_target_discriminator.py`＋两本契约册（`test_n5_39_seed_is_structural_identity_contract.py` 5 passed、`test_n6_13_write_target_discriminator_contract.py` 11 passed）。N1 侧的待批不变＝`PLAN-N1-04` 的批⑴（按 §1 动产品码、先做零训练取证）与批⑵（训练臂预算）。
9. 下一格的次序（现读队首与两个仓外驱动）：①`seed_scratch/chain812/log.txt` 若已出现，**先读日志再决定要不要自己起命令**——它里面已按 `PLAN-N5-23`/`PLAN-N5-22` 逐字跑过地板、两臂各一趟 CAP（同 HEAD 硬约束）、配对、带 CAP 的地板复算，重复起第二遍会污染取证；②按 `PLAN-N5-23` §5（㊵-825 新加）判读：本链地板只有 `ruler_no_spread` 一支可走，批次 C 是「无地板可依」不是「差被地板挡住」，两种说法不许混写；③`chain811`（整族基线面）排在 `chain812` 之后，同样只取证不判读；④DEBT-G137 剩余＝把 `legacy_heuristic_candidates` 键加回判别式（或另立只读普查复算 ㊵-807 那 6 枚候选）再逐枚实跑、跑前后各取一次**内容级** `git diff --numstat -- reports/` 快照，然后才换债表状态；⑤G115 修法③ 剩余 17 枚（分类见台账 ㊵-829 ④），窗口内只挑不载模型的那几枚。
