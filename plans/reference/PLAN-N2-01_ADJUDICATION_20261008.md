# PLAN-N2-01 判读 · N2 巩固通电第一次（2026-10-08）

> **效力**：本件是 [PLAN-N2-01](PLAN-N2-01_consolidation_powerup_prereg_20261007.md) §4bis/§4ter 冻结判据的**判读报告**，不改判据、不改剂量、不换指标。
> 通电半径＝产品默认基座，owner 2026-10-08 弹窗批"按预注册通电一次"（09 §3.2 第 6 条）。

## 0. 一句话结论

**加速点① 在最小剂量档上不成立**：一次真实离线巩固确实改了权重（候选档对母档 **68/202** 张量有差），旧能力侧的 CAP-0 严格命中 **9→12** 且回退路径**实证可用**（G-N2-1 逐位同），但 J-N2b 的新材料收益**没有**——同材料、同链、配对读数上治疗臂**低于**对照臂；另外当场撞出一条更值钱的产品缺陷 **DEBT-G47：巩固产物装不回来**。合取式因此判 `not_closed`。

## 1. 那一支 pass 的自述（不转述，抄件）

`data/consolidation/last_report.json`：`pass_id=27a34233e581`、`reason=n2-powerup-1`、`duration_ms=643934.3`（10.7 分钟，其中大头是 `select_for_sleep` 要在 200 条全池上逐条 `judge.score`——这是产品自己的选择路径，不是仪器冗余）、累计 `passes=4`。

- **投影**：`records=200`、`by_source={constraints 0, interactions 200, workbench_capabilities 0}`、语料 `consolidated/corpus-20261008T025129Z-27a34233e581.jsonl`。
  `workbench_capabilities=0` 与 §4ter 实测二的预测对上（`snapshot 93bd90db… already projected`）⇒ ⑥ 面板行今天显示的就是 **0**，且 0 也要照实显示。
- **器官**：`{ran: true, learn: true, cycles_per_text: 1, max_symbols: 64, texts: 8, stats: {texts 8, cycles 8, accepted 8, mean_priority 0.27041, mean_confidence 0.66686, mean_error_norm 1.91966}}`。
  ⇒ **这一支的剂量就是 8 条 × 1 回合 × 64 符号**，与批文一致（四参数一律沿用 `run()` 默认，未放大任何一档）。
- **就绪门**：`spec.written=true`，理由自述 `1002 unprojected interactions are waiting for rehearsal`（守卫臂在 10:50 先吃掉 200 条，故通电这一支吃的是接下来的 200 条）。

## 2. 判据逐条（冻结值 → 实测 → 判定）

| 判据（§2/§4ter 原文） | 巩固前 | 巩固后 | 判定 |
|---|---|---|---|
| **J-N2a-1** CAP-0 严格命中项数 ≥ 巩固前 − 1 | **9**（C 2/14、D 6/16、E 1/20；B/G 各 20 项待人审，不进分子） | **12**（C 2/14、D 9/16、E 1/20） | **成立**（+3，且这把尺有动态范围：`cap0_ruler_usable=true`） |
| **J-N2a-2** 复述面 ×24 各惩罚档不跌破 | 严格命中 5/5/6/6（惩罚 0.0/0.5/1.0/2.0）、成句 24/24/24/25 | 6/5/6/6、成句 24/24/24/**23** | **不成立**（命中零档下跌、一档 +1，但 2.0 档成句数 −2） |
| **J-N2a-3** 停摆面 ×24／×96 主列 | ×24 面件在库（rc=0）；×96 面 `never_lf_eaters_counted` **234**／分母 `generation_rows_seen=288` | ×24 面件在库（rc=0）；×96 面主列 **190**／同分母 288（`items_sha256=0541a3f4568a9c5b` 前后一致；巩固前件 `checkpoint_sha256=d6169a358eaee6d1`＝产品默认基座，巩固后件 `1aa97ef920872b37`＝对齐候选档） | **主列已取到、判据对方向沉默**：差 **−44**。历史 236／225 那条是 **a31 链**，与本轮不同链 ⇒ 只作本链前后配对的"前"。§2 那句"无一项跌破"**从未给"拖写行数"这列指定哪个方向算跌破**（越少越好是历史解释，不是冻结判据）⇒ **只报数与差，不冒充判据判定**（DEBT-G46 第三类实例，见 §7） |
| **J-N2b** 新材料读数 > 同材料不学对照 | 对照＝回退件（＝同一份母权重）对**同一批夜文本**的 `SeedJudge` 均值 **−14.2791**（accuracy 0.05960，surprise 14.2105） | 治疗件 **−14.4628**（accuracy 0.05277，surprise 14.3890） | **不成立**（治疗 ≤ 对照，按冻结规则即否；差 −0.1837） |
| **合取** J-N2a ∧ J-N2b | — | — | **`not_closed`**（`reports/taiji_n2_face_verdict_20261008.json`，blocking＝`j_n2a_replay`／`j_n2b`／`stop_face_unverified`） |

## 3. 四条守卫逐条

- **G-N2-1（回退必须真演示一次）＝通过，且是双重实证**：内存还原后 `content_digest` 同为 `3e00bafc6298ac84`；回退档落盘→独立 load→**张量 0/202 有差**；回退面 CAP-0 与巩固前**读数逐位同**（判读器只剥出处字段，剥了哪些与三处原值都列在件里：档路径／档 sha256／git_head——两趟之间我提交了 530b9a37）。
- **G-N2-2（默认位不动）＝通过**（㊵-486，`reports/taiji_n2_guard_default_20261008.json`，五项全真、权重摘要不变）。
- **G-N2-3（训练前保存检查）＝部分**：本件实测的是"母档落盘→独立 load→摘要逐位同"这一条；`优化器/RNG/课程游标/预算` 四件套在运行时信封里没有对应字段 ⇒ 记 **unverified**（§4ter 已点名，不冒充合同满足）。
- **G-N2-4（惊讶度调制须披露）＝通过**：夜文本 8 条由**调制后**排序选出（`scheduler_agrees_with_weighted_sort=true`，重算与 `select_for_sleep` 逐条一致），去掉 `mean_surprise` 权重后前 8 条里有 **3 条换成别的**（件 `taiji_n2_material_20261008.json` 的 `surprise_modulation` 列出两批 sha16）；池 200 条的 quality 均值 −13.0056、surprise 均值 12.9460。

## 4. 本轮最值钱的发现：DEBT-G47（巩固产物装不回来）

睡眠 pass 结束时 runtime 停在它自己的睡眠回合里，`SeedRuntime.save()` 写出的信封**两半自相矛盾**——`taiji.cognitive_state={tick 66, episode_id "sleep-experience"}` 而 `taiji.kernel.state={tick 92}`，而 `TaijiKernel.restore_native` 末尾的守卫比的正是这两半（[taiji/adapter.py:12631-12632](../../taiji/adapter.py)）⇒ `SeedRuntime.load(候选档)` 响亮拒绝 `ValueError: native cognitive state is out of sync with kernel state`。

- **半径**：§8.3 的候选版本隔离在"离线巩固"这条路上**结构性不成立**；巩固产生的权重改动只能活在内存里，一落盘就废；N5"经验→能力"要持久化必经此门。
- **差点被读成结论的那一步**：第一轮 CAP-0「巩固后」100 项**全部** `load_ok=false`、`machine_scored_correct` 恒 0——判读器若不先数装载失败，就会把仪器缺陷报成"能力归零"。现已硬编码：任一档有 load 失败 ⇒ 整面作废、rc=2（`not_judged_face_invalid`）。
- **本轮的处置（不是修法）**：产出一份**只补计数器、张量一字不动**的对齐件 `checkpoints/seed_n2_candidate_aligned_20261008.pt` 来读治疗面；正反对照都在件里（未对齐候选仍拒收；对齐件 202 张量零差、`align_pass=true`）。⇒ **§2 表里所有"巩固后"的读数定性为"信封两半对齐之后的候选权重"，不是装机可用态**。
- **修法已由 owner 2026-10-08 弹窗裁＝甲**（睡眠回合结束时收束回醒来态，改在 `sleep_pass.sleep_organs`／`SeedSleepScheduler.night` 内部；备选乙＝`SeedRuntime.save()` 前以 kernel 半边为权威，未选）。守卫形状随批文一起定：**通电后立即 save→load 必须成功**（今天这条是红的 ⇒ 它能为 false）＋一条反向（对**已一致**的档，修法不得改变 load 后的摘要，防顺手覆写有效状态）。**裁完之后本件的巩固后面必须在未对齐的原生候选档上重跑，才算通道打通**——本轮所有"巩固后"读数都还带着"读的是对齐件"这条限定。**⇒ 2026-10-08 修法甲已实施**：`seed_platform/sleep_pass.py:sleep_organs` 在 `scheduler.night(...)` 之后收束 `reset_dynamics(episode_id="wake-after-sleep")`（只清回合活动、保留已学突触；醒不过来只记 `wake_error`，不改报 `ran=false`），三支契约测入库 `tests/seed/test_sleep_pass_wake_reentry.py`（反支钉缺陷本体、正支走产品路径、反向守卫先证明选择器非空再要求 `edge_weight`/`pre_index`/`post_index` 逐位不变）。**实施为什么落在 `sleep_organs` 而不是 `night` 内部**：`night` 的活调用点还包括三支 `verify_seed_a{2,3,4,5}*.py` 与两支既有测，其中 A3 那条专门量跨夜漂移 ⇒ 按批文给的两个位置里取半径更窄的那个，并如实点名"仪器侧直接调 `night` 仍不收束"这条不对称。**本轮那枚真实候选档救不回**（信封已写坏，只能重跑），而重跑＝第二次改权重、不在 ㊵-485 批文内 ⇒ "经验→能力可持久化"这句话到今天仍记**未验证**。

## 5. 仪器侧的两处自我更正（都是我自己造的）

1. **`--phase powerup` v1 把候选档的磁盘 load 放在取数之前** ⇒ 守卫一拒收，整支除了磁盘三档之外**一个读数都没留下**（`RC_POWERUP=1`，traceback 在旧 :305）。v2 的形状＝治疗臂读数在**还原之前**用内存里那份改过权重的 runtime 取、对照臂用母锚重算夜文本、两档磁盘 load 只记录不中止、整支 `try/finally` 保证失败也落件。**这条修复没被重新走过**：批文只批通电一次，第二次改权重不在授权内 ⇒ v2 的形状靠 `postcheck/align/material/dose` 四支只读相补齐，`powerup` 相本身留待下一次经 owner 批的通电验证。
2. **判读器把"出处字段"当读数差** ⇒ 第一次跑判 `g_n2_1_identical=false`，实际只差 3 个出处叶子（档路径／档 sha／git_head），100 项回答逐位同。修＝读数比对剥 `checkpoint`/`identity`，并把剥掉的三处原值列进件里（`provenance_before/rollback/after`）；同一段代码对 before/after 走**另一支**（`ruler_usable=true`）⇒ 这条比较器仍然能为 false，不是恒真式。

## 6. 一条我自己仪器的口径缺陷（点名，不改判）

J-N2b 的读数用 `SeedJudge.score()` 打**整篇**，而这一支 pass 的经验预算只有 `max_symbols=64`——一条几千字节的交互里模型这轮只活过前 64 字节。⇒ **事后**加了一支诊断臂（`--phase dose`，件 `taiji_n2_dose_diagnostic_20261008.json`，件内自述 `kind=diagnostic_post_hoc`、明写不参与判据）：把同一批夜文本收在 64 符号窗内重读，治疗 **−13.6078** 对 对照 **−12.3376**（accuracy 0.078125 对 0.09375），`window_holds=false`。
⇒ 结论不因口径而翻：**窗内和整篇两个口径都说治疗更差**。但下一件预注册必须把"剂量窗"与"读数面"对齐，否则 J-N2b 天然只能读出稀释。

## 7. 一条判据含糊（DEBT-G46 的第二个实例）

§2 写"四个对照面无一项**跌破**；跌破 **≥2 面** ⇒ 判'通电有代价'"。本轮实测＝**恰好跌破 1 项**（2.0 惩罚档成句数 −2），落在两条线之间、判据本身没说算什么。我本轮的处理＝**按最严读法判 J-N2a 不成立**（不拿含糊处给自己减分），并把这条缺口登记为 [DEBT-G46](../active/roadmap/05_TECH_DEBT_REGISTER.md) 的同类实例：下一件 N 系列预注册的保持侧判据必须写成**单一数值条件**（例如"任一档任一保持列的跌幅 > D 即判有代价"，D 先冻），不许再留"一项/两项"的中间地带。

## 8. 未做与下一步

- **×96 停摆面主列已跑并接入**（234→190，见 §2 表 J-N2a-3；判读器 `--stop-main` 读现成计数仪的读数件 `reports/taiji_n2_stop96_main_20261008.json`），但**方向在冻结判据里没定义** ⇒ 该项进 `blocking_terms` 的名义是 `stop_face_direction_defined`，不是"没测到"。
- **gated 三臂（资格门控 X=0/9/27 那一族）仍未跑** ⇒ J-N2a 保持侧在"资格门控"这一面上还是 `unverified`；它是 copy-circuit 注入形状档，与本轮权重改动的相关性本身也待核（同一台仪器的另一条链）。
- **⑥ P4 只读面板行＝本轮已落地**（零风险、owner 已批）：python 侧 `_add_recommendations` 加 `internalise_workbench_capabilities` 一支（零折算不生成，两条都有单测：`tests/seed/test_sleep_pass.py` **14 passed**）；harness 侧 `LifePassReportView.workbenchCapabilities`（`types.ts`）＋`runtime-client.ts` 从 `projection.by_source.workbench_capabilities` 取值（缺投影读 0，不猜）＋面板一行 `passCapabilitiesLabel`＋zh/en 两个字典键＋派生目录 `tool-cordis/src/api-catalog.ts` 同步；host＋client 两支 spec 一并更新（`consolidation.host.spec.ts` 加"缺投影读 0"一条、`panel.client.spec.tsx` 加行断言），`vitest run` 两文件 **63 passed**。
- **下一次通电的前置**（不在本件）：DEBT-G47 修法裁定＋剂量/读数同窗＋×96 附件。

## 9. 复算命令（读数不连命令不许入库）

```
python scripts/training/run_taiji_n2_powerup.py --phase guard      # G-N2-2
python scripts/training/run_taiji_n2_powerup.py --phase powerup    # ①④⑤⑥（本轮只跑过一次）
python scripts/training/run_taiji_n2_powerup.py --phase postcheck  # 三档可载性＋张量级幅度
python scripts/training/run_taiji_n2_powerup.py --phase align      # 对齐件＋正反对照
python scripts/training/run_taiji_n2_powerup.py --phase material   # J-N2b 配对读数
python scripts/training/run_taiji_n2_powerup.py --phase dose       # 事后诊断臂（不参与判据）
python scripts/training/eval_taiji_cap0_baseline.py --checkpoint <档> --report <新件>     # 三面
python scripts/training/probe_taiji_a30_stop_failure.py --checkpoint <档> --limit 24 --out-report <新件>
python scripts/training/measure_taiji_a30_repetition_penalty.py --checkpoint <档> --limit 24 --out-report <新件>
python scripts/training/count_taiji_n2_faces.py                    # 合取判读（缺件 rc=2）
```

## 10. 不变项

M6 收官、M8 挂起；M7-CI 绿只在推送后实测（本件不声称）；`DEFAULT_CHECKPOINT` 未被覆写（候选／回退／对齐三枚都是新路径）；产品默认位 `organs/learn` 仍为 False；台账行序以行首标号为准（㊵-419⑥）。
