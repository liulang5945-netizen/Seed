# Seed 开发计划索引（2026-09-25 v3，编号体系规范化）

> 用途：**唯一的计划入口**。v3 新增**编号体系**（一个前缀一个意义，§0）；
> v2 修正层级：**主线是 M 系列（里程碑）**，A/B 是**支线**（欠账修复 / 架构债），与主线**并行、互不阻塞**。
> 历史文档**冻结、不重命名**（改名会断日志与提交的链接），用 §4 的**新编号 ↔ 旧文件名**映射检索。
> 冻结判据/预注册/冻结证据只追加；本索引随状态**就地修订**。

## 0. 编号体系（**一个前缀一个意义**；新增文档强制，存量不动只映射）

**格式**：`<类别>-<归属>-<序号>[_<主题>][.<日期>]`
—— **类别**说"这是什么性质的文件"，**归属**说"属于哪条线/哪个里程碑"，**序号**是该归属内的顺序。

| 类别前缀 | 意义（只放这一类东西） | 旧名对应 |
|---|---|---|
| `PLAN` | 计划/提案/路线（**非判决**，可修订） | *_PROPOSAL、*_BRIEF、*_MECHANISM_DESIGN |
| `SPEC` | 判据/预注册/合同（**冻结件，只追加**） | *_PREREG、*_CONTRACT、*_BUDGET |
| `EXP` | 实验执行与判决（训练臂/评测/计分的**落盘件**） | eval_taiji_*、train_taiji_*、reports/*.json 判决件 |
| `DIAG` | 诊断与归因（探针/审计，**非判决**） | probe_taiji_*、audit_taiji_*、diag_* |
| `FIX` | 缺陷修复与守卫 | *_fixes、tests/*_contract |
| `MS` | 里程碑判定/批准 | M5_EXIT_APPROVAL |
| `CONV` | 口径/约定 | metrics_conventions、FOLDER_STRUCTURE_RULES、REPO_HYGIENE_RULES |

**里程碑**：`M1..M6` 不变（大阶段主线，见 §1）。
**归属**：`R2`（M5 排除项调查，已收束）/ `A`（修复线，见 §2）/ `B`（架构债线，见 §3）/ `M6`（产品交付，见 §1）。
**脚本前缀**（代码件，另有约定）：`train_/eval_/verify_/build_/probe_/diag_/audit_/score_` 说"脚本干什么"，
与文档类别前缀**互不混用**。

**存量处置**：历史文件**冻结、不重命名**（改名断日志与提交链接）；§4 给**旧名 → 新编号**的映射；
**新增文档一律按本表**（例：T8 若今天新建应为 `PLAN-A-08_mechanism-design`）。

## 1. 主线：M 系列（里程碑）

| M | 名称 | 状态 |
|---|---|---|
| M1–M4 | 早期里程碑 | ✅ 已完成（详见 01_SCOPE_AND_PHASES） |
| **M5** | 限定退出 | ✅ **2026-09-20 批准落盘**（[批准书](../reference/M5_EXIT_APPROVAL_20260920.md)） |
| **M6** | **产品交付（当前大阶段）** | 🔄 **界面载体＝`taiji-harness/`**（dsh 整仓 fork）；原 `desktop-electron/`＋`frontend/`＋`desktop/` 与发布链 **2026-09-23 随裁决 6 退役**（G3 通过后删旧线）。里程碑：**G1 ✓ G2 ✓（含 G2b–G2h 修复轮）G3 ✓**（`dsh-llm-taiji` 硬接 8000，真机回合由 Taiji 应答）→ **G4 进行中**（六步全落地；判据①②④达成、③挂账待裁定）→ **G5 交付**（web 先行，客户端版后做） |

### M6 开放项（2026-09-25 就地修订；细节与读数在 03 §5.7 的 Taiji Harness 活动卡）

1. **G4 判据③**（面板点真训练→进度流→停止→检查点新增复验；再真续训、真激活各一回合）——**按所有者裁定等 R2 窗口收束**再做，恢复条件不由「继续推进」字样自动触发（03 §5.7 两项裁定条）。
2. ~~**未闭合 I**：模型面板缺 Taiji 组 ⇒ 默认模型不可用、composer 被挡、真机 UI 回合取不到~~ ⇒ **2026-09-25 已定位、修复并产品级复验**：根因是就绪只在 load 与 `loader/volatile-update` 采样一次，运行时冷启动晚于 harness 启动即**永久**不注册（`listProviders()` 只列有 adapter 的 provider，而 `buildModelCatalog` 只读它）；改为按 `readinessPollMs`（默认 5s）在插件存活期内再探测，单元与真组合红绿各跑。**真机红绿（同一 web 实例、未重启）**：路由指向空端口启动 ⇒ 模型面无 Taiji 组、composer 停在「当前模型不可用」；假运行时就绪 ⇒ `group Taiji（本地运行时）` 自行出现且 radio `checked`。判据⑤ 的另两半（启动即就绪时 `taijiAtBoot=true`、**发一个普通对话回合成功**「已完成工作 用时 11 秒／1 轮 1 步」）同批复验 ⇒ C2 批「真机 UI 回合未取得」补上（详见 03 §5.7）。
3. ~~**i18n pairing 全仓欠账**（约 252 对 out-of-sync，G2 rename 改了两侧 `.md` 未重录 `.i18n.yaml`）~~ ⇒ **2026-09-25 全部销账**：按裁定分两批「先核对后重录」（第一批 `docs/subsystems` 19 对＝`f6f9eb6a`；第二批余 233 对＝`5d7ed3e4`）。核对判据＝逐对取「该对 `.i18n.yaml` 最后一次被确认的提交」到 HEAD 的**两侧** diff，断言每处改动只是 `@deepseek-ai/`→`@taiji/` 更名、且两侧更名次数逐对相等；**251 对纯机械、1 对（`CONTRIBUTING` 的 slogan 改写）逐字人工读过** ⇒ 零文档正文改动，全仓门转绿（**1099 对全一致**）。
4. ~~**6 个在飞文件**~~ **已查清为行尾幻影（2026-09-25 实测）**：`life-context` ×4、`session-memory-taiji` ×2 在 `git status` 显示 `M`，而 `git diff HEAD --numstat` 为空 ⇒ 与 HEAD 零内容差（`core.autocrlf=true` 所致）；**无待裁定的在飞改动，升级重放不被它阻塞**（08 §6）。
5. **既存红门**：~~`verify-concrete-terms` 3 处 `provenance` 命中~~ **2026-09-25 已清、该门转绿**（改为点名实际字段 `count`/`revision`/`source`/`owner`，提交 `9a4be7ab`）；余 `verify-plugin-packages` 1 处（产物在他人包，未处置）。
6. ~~**J（2026-09-25 新发现）**：native 链路收到的 `prompt` 不是用户提问~~ ⇒ **同日已钉死并修复**：agent-loop 的 runtime-context 快照是 durable **user** 消息（`source.kind='runtime-context'`，`agent-loop/src/runtime-context.ts:14/20`）且按设计追加在本轮**之后**，而 `llm-taiji/src/chat.ts` 取「最后一条 user-role 消息」当运行时唯一的 `prompt` ⇒ 用户提问被降级进 `history`。修法＝取最后一条**属用户本人输入**的 user 消息（不带 source 或 `source.kind==='user'`，口径同 `memory-context/src/index.ts:95`），全为 harness 自有时**兜底**取最后一条 user-role（辅助调用 `session-title` 依赖该兜底，否则发空 prompt）。取证＝8098 转发代理抓真实请求体（零改仓、零插桩）：修前 `prompt` 496 字符快照／history 3 对，修后 `prompt` 17 字符提问／history 2 对；红绿＋真机各一次，提交 `9155fca0`。**连带提醒 A 线自查**：此前凡经 web 回合取过的读数，其 prompt 都不是题面（直接 curl runtime 的读数不受影响）。
7. **doc-sync 批余 5 条既存红（2026-09-25 逐条定性，均非本轮引入）**：①`verify-dependency-catalog`＝**跨平台排序假红**（本机重生成后与已提交版**字节数相同、561 包集合与逐字段全同、按名排序后整份逐字相同**；差异只是 `localeCompare` 在 darwin/node 24.19 与 win32/node 24.15 下的条目顺序）⇒ 重生成产物已回滚，要真绿须把比较器改为与 locale 无关（上游脚本改动＋一次性重排，**需裁定**）；②`verify-persistence-catalog` 是**生成物陈旧**，但与③纠缠（重生成会把尚未承认的破坏性变更固化进受控文件）⇒ 与③一并处置；③`verify-persistence-changes`＝H3c 给 `SessionPersistence` 加 `delete` 契约后**从未按 `docs/cookbook/reviewing-persistence-type-changes.md` 承认**，处置可能牵动 Session 格式版本 ⇒ **需所有者裁定**；④~~`verify-archived-agent-notes`＝fork 布局假红~~ **2026-09-25 已修、门转绿**（`ls-tree` 的 pathspec 相对 cwd 而 `git show <ref>:<path>` 相对仓库根，改为按 `git rev-parse --show-prefix` 分别拼；`1917 frozen artifact(s)／6 kind(s)`，无真实违规；上游布局下前缀为空 ⇒ 零行为变化；提交 `7b6687eb`，登记为 08 的 **H3k**）；⑤~~`verify-plugin-packages` 1 处（产物在他人包）~~ **2026-09-26 已清、且原归属判断是错的**：落后的是 agent-preset 的组合参考包表，重生成后差异**恰为我方 G3/G4 新增六包各一行** ⇒ 我方漏跑 `gen-plugin-packages`（提交 `fe1dc376`）；该生成器已补进 08 的 M3 清单与 §5 判据②（此前不在其中，与 H3j 同型的清单盲区）。**doc-sync 批的既存红至此只剩两条**（`dependency-catalog` 跨平台排序假红、`doc-typecheck`＋`docs:build` 需 pnpm）。另：聚合入口 `run-gates.ts` 要求经 pnpm 调用而本机 pnpm 不在 PATH ⇒ 只能逐门直跑，`doc-typecheck` 与 `docs:build` 两个 pnpm 叶子未跑。
8. **C6（审计登记的 P2）⇒ 提案已交付、待批**：[PLAN-M6-01](../reference/PLAN-M6-01_experience-projection-producer_20260926.md)。**2026-09-26 只读取证把前提部分否证**——缺口不是一条而是三条：①**生产者缺**（`project_to_ledger` 仍只被训练脚本与测试调用、`api/` 零引用）；②**消费者也缺**（E1 ledger＝`seed_platform/evolution_ledger.py:73` 的 `EvolutionExperienceLedger`，其 `training_view` 在产品侧无人读 ⇒ 只补生产者会得到一个只写不读的日志，正是审计对 C3/C4 的「只补一端无效」同型）；③**物料错配**（现成三适配器是 skill/mcp/client-plugin，而真机 `GET /api/mcp-client-capabilities` 的 `records=[]`，skill 与 client-plugin 在 native 侧无运行时来源；真正有料的 `GET /api/workbench/capabilities`＝16 条能力、内容寻址 `snapshot_id`、`revision=6`，**却没有对应适配器**）⇒ **照原样接线今天投影 0 条**（给空集铺管道）。提案＝新适配器 `WorkbenchCapabilityAdapter` ＋ 生产者挂 `sleep_pass` 的 project 段 ＋ **消费者同批**（折进 `data/consolidated/corpus-*.jsonl` 并进 `spec.datasets`）；前置否证门 **Z1**＝先只写适配器＋离线脚本对活运行时投影一次，判据「单元数 > 0 且 `native_trainable`」，**不过即停报**（记「物料不足降级」，不铺运行时接线）。四个决策点在提案 §8。

### 已作废：M6 desktop 线未闭合五项（2026-09-21 立卡）

①logo 判定、②托盘退不出、⑤单实例锁**均已闭合**（所有者裁决＋根因修复＋实测）；③UI 交互其余项、④跨壳孤儿回收**随该线退役作废**（载体已删）。原活动卡见 03 §5.7「M6 desktop 交付线」（线状态＝已退役，卡为历史记录）。

## 2. 支线 A：R2 欠账修复线（语言读出/答对）

> 来历：M5 退出时 R2 语言能力列为**显式排除项**（欠账保留在账）。调查已收束（归因定案），
> 修复由**并行会话**以 A2 复制回路推进。**与 M6 并行、互不阻塞**（负载边界：R2 训练等待期
> 不并行 ≈30 min 级重构建/重评测）。

| 步 | 名称 | 状态 | 判据 |
|---|---|---|---|
| **A-1** | R2 归因 | ✅ **定案** | CAP D+E=0/36 是「先告知→后提问」与"无归纳机制"的结构性错配 |
| **A-2** | 复制回路（A2.1 写入门 + A2.2 F1 内容直读，gate 零初始化位级不变） | ✅ **落地** | §4.1 结构存在性过（greedy 首字节=0xE9）；合同 6 绿 |
| **A-3** | M-3：问答结构语料 + 召回条件发射训练 | ✅ **并入 A2.3** | 立项 §5：A0 证伪"甲单独走"（无通道则无物可学） |
| **A2.3** | 召回条件发射训练（裸格式） | ✅ 判据过 | 寻址 0.86；CAP 抽查 5/36 ⇒ **严格口径 4/16**（见下） |
| **A2.4** | 协议开闸（chat 链入库＋显式挂载，默认关） | ✅ 落地 | chat 形态 3/36 ⇒ **严格 2/16**；真增量 D09"42"/D12"23" |
| **A2.3b** | **协议格式对齐重训**（问：/答：壳＋多事件＋自历史上文） | ✅ **判据②过** | chat 形态严格真命中 **6–7/16**（两个独立初始化电路：seed-A 7、seed-B 6；线 >2/16；对照 0/16、格式错配臂 2/16、裸格式抽查 4/16）；S2 寻址 0.848、基座 sha256 未动 |
| **GATE** | 立项 §4.3 回归门（装上有没有让 A/B/C 变差） | ✅ **收口：无代价、方向为正**（表层在扩展分母上判 `improved`） | 能力分没劣化：A 维 9 项全不掉、C 维 0.0→**0.0714**（升）、B 维 0 项可机检记 `unverified`。表层成句率 0.0612→0.0408（49 句里 3 句变 2 句）判"未通过"，而同一电路在 D/E 原始字节链上成句率**升**（0.0278→0.0833）⇒ 原条文没钉链路/题集，且 36 题、49 句两个分母上"≥3"阈值都要 8% 摆动＝**结构性不可判**。修订见立项件 §9＋[`SPEC-A-21`](../reference/SPEC-A-21_r2_surface_extension_prereg_20260925.md)（104 题≈260 条文本、两独立电路同向才判） |
| **A2.5?** | ~~事件选择~~（"挑哪条记忆"变可学） | ❌ **被天花板读数否证，不开案** | 把选择替它做对（只留正确告知入历史）只得 **9/16**，`as-is` 是 7/16 ⇒ 增益 +2 < 预登记的 ≥3 分辨率线；**即使检索完美也只有 9/16**，瓶颈在告知之后那一侧（发射/生成时点）。§7.2 那句"第一限制因素是挑哪条记忆"是把相关当因果，已更正（读数件 `taiji_r2_a25_selection_ceiling_20260925.json`） |
| **A2.6?** | 失败题的**逐答案步轨迹**（内容在库里、选择对了，为什么发不出来） | ⏳ **待开（仍是零训练诊断）** | 对 oracle 臂那 7 题取轨迹：目标字节是否进过 copy_dist 前列、当时 gate 开度、生成流从哪一步偏离 |
| **A-4** | 可分离读法推广到主训练线 | ⏳ 前件已满足，排在门补完之后 | `predictive_context_region0_only`（默认关、掩码式、守卫 4 绿） |

**计分口径已收紧（2026-09-25，读旧结论时必带）**：`expected_contains` 里混着通用词，
旧口径把"没装电路也能命中"算成了能力。严格口径＝**期望词必须在更早的用户轮原文里逐字出现**：
裸格式 5/36 ⇒ **4/16**，chat 对照 2/36 ⇒ **0/16**，chat 电路 3/36 ⇒ **2/16**；
且 **E 维严格可命中数恒 0** ⇒ 旧判据"CAP D+E>0"里真正承载复制的只有 D 维。
复算件 `reports/taiji_r2_copy_strict_cap_recompute_20260925.json`。

**支线 A 唯一下一步**：**把 A2.6 探针跑在扩展集的失败题上**（seed-A 那批 81/104 未命中），
拿残余失败的**成因占比**——现在只有 6 题、分三类（`address_miss` 3／`continuation_slips` 1／`emission_loses` 2），
谁占多数无法判定，所以先别选"修哪一类"。仍是零训练。落完再轮到 **A-4**（可分离读法推广）。
**已完成的两次独立取数**：seed-A 严格 **7/16**、seed-B 严格 **6/16**，对照均 0/16、格式错配基线 2/16
⇒ 能力侧增益可重复（此后按 **6–7/16 这一档**引用，不写单点）；两电路命中集不同（seed-B 拿下带干扰项的 D06）。
落完之后才轮到 **A-4**（可分离读法推广）与 A2.6（发射侧轨迹诊断，`probe_taiji_r2_a26_emission_trace.py` 已写好待跑）。
判法固定：两电路同向且成句差 ≥3 条才判劣化/改善，否则 `not_resolved`——**"没判出劣化"不等于"无代价"**。
落完之后才轮到 **A-4**（可分离读法推广）与 A2.6（发射侧轨迹诊断）。
**已作废的一条队列项**：A2.5 事件选择——天花板实测显示把选择替它做对只得 9/16（净 +2），
且 7 题在两臂下都失败；我曾据 38.3% 挑错率判它"第一限制因素"，那是把相关当因果，
更正见 [`SPEC-A-17`](../reference/SPEC-A-17_r2_a2_3b_format_align_prereg_20260925.md) §10。

## 3. 支线 B：架构债线（审计修复）

| 步 | 名称 | 状态 | 判据 |
|---|---|---|---|
| **B-0** | P0+P1+P2 三轮修复 | ✅ 完成（116 测试） | `project_audit_fixes{,_p2}_2026-08-23.md` |
| **B-1** | `detect_modality` 抽为纯函数（无 self） | ✅ 完成 | 等价守卫 3 绿（`a6895b6a`） |
| **B-2** | `_infer_domain` ✅（黄金 330 格等价）；`_reencode`/`nll_quality` 挂起（需 neuroplex 夹具） | ⏳ 留 B-3 | 见 §6 |
| **B-3** | Cortex 神对象完整拆分（路由/域推断/生成三簇） | ⏳ **需负责人签字** | 每迁一簇配等价测试；冻结基线，禁静默漂移 |
| **B-4** | 核心推理路径覆盖率 ≥60% | ⏳ 待 B-3 后 | — |

## 4. 存量映射（**新编号 ↔ 旧文件名**；旧文件冻结不改名）

| 新编号 | 旧文件（`plans/reference/` 与 `scripts/training/`） |
|---|---|
| `MS-M5-01` | `M5_EXIT_APPROVAL_20260920.md` |
| `EXP-A-01` | `M5_R2_READOUT_RETRAIN_*`（A/B/C 合同、判决与 runner） |
| `PLAN-A-02` | `M5_R2_COMPOSITION_BINDING_BUDGET_PROPOSAL_20260923.md` |
| `SPEC-A-03` / `EXP-A-04` | `M5_R2_T1_T2_PREREG_20260923.md`（预注册）与其轨迹/判决件 |
| `SPEC-A-05` / `EXP-A-06` | `M5_R2_T3_FABRIC_WRITE_PREREG_20260923.md` |
| `SPEC-A-07` / `EXP-A-08` | `M5_R2_T4_SLOW_ALL_PREREG_20260924.md` |
| `SPEC-A-09` / `EXP-A-10` | `M5_R2_T5_REGION0_CUE_PREREG_20260924.md` |
| `SPEC-A-11` / `EXP-A-12` | `M5_R2_T6_CAP_PREREG_20260924.md`（判决件 `reports/taiji_r2_cap_checkpoint_scores_20260924.json`） |
| `SPEC-A-13` / `EXP-A-14` | `M5_R2_T7_MASKED_A_PREREG_20260924.md`（判决件 `reports/taiji_r2_masked_arm_a_20260924.json`） |
| `PLAN-A-15` | `M5_R2_T8_MECHANISM_DESIGN_20260925.md` |
| `PLAN-A-16` | `M5_R2_ARCH_LEVEL_PROPOSAL_20260925.md` |
| `SPEC-A-17` | `SPEC-A-17_r2_a2_3b_format_align_prereg_20260925.md`（**新建即按本表编号**） |
| `PLAN-A-18` | `M5_R2_ARCH_COPY_CIRCUIT_PROJECT_20260925.md`（A2 立项） |
| `SPEC-A-19` | `M5_R2_A2_3_PREREG_20260925.md`（含 §6 rev2–rev4 修订与 §7 读数） |
| `PLAN-A-20` | `M5_R2_A2_4_PROPOSAL_20260925.md`（协议开闸提案＋chat 形态读数） |
| `SPEC-A-21` | `SPEC-A-21_r2_surface_extension_prereg_20260925.md`（**新建即按本表编号**；§4.3a 表层子判据的扩展分母） |
| `PLAN-M6-01` | `PLAN-M6-01_experience-projection-producer_20260926.md`（**新建即按本表编号**；C6 经验投影的生产者／消费者／物料三缺口、`WorkbenchCapabilityAdapter` 与 Z1 前置否证门） |
| `FIX-B-01` | `project_audit_fixes_20260823.md` / `project_audit_fixes_p2_20260823.md` |
| `CONV-B-02` | `docs/metrics_conventions.md` / `docs/REPO_HYGIENE_RULES.md` / `docs/FOLDER_STRUCTURE_RULES.md` |

## 5. A-3（M-3）详细设计：**问答结构语料 + 召回条件发射训练**

> **本节已被 A2.3 取代（2026-09-25）**：A0 重判证明"无内容通道则无物可学"，M-3 的语料与制度
> 设计已并入 A2 立项 §3-A2.3 与 `SPEC-A-19`（学习规则、实体表防泄漏、预算止损全部照彼件冻结）。
> 下文保留作设计依据与判读线出处，**不再作为待办条目**。

**目标**：让 F1 读出在"问题条件"下学会**发射答案字节**（T8 已判 (c)：读出没学过"由召回内容发射答案"）。

* **语料来源**：`data/simple_zh/dialogue_extended_clean.jsonl`（已有 问：/答： 结构）。
* **语料格式（设计）**：每行 JSON
  `{"text": "<告知>。<提问>？<答案>。"}`
  例：`{"text": "我叫阿岩。我的名字是什么？阿岩。"}`。
  由 dialogue 的 问：/答： 对改写成此形，**再注入合成的 given-then-ask 对**（不同名字/地点/物品），
  避免只学表面问答形状。
* **⚠️ 防泄漏（沿用指标白皮书）**：合成对的"告知内容"**必须与 CAP-43 评价集的答案不相交**
  （否则 CAP>0 只是测试集记忆）。评价集 `cap0_eval_set_v2.json` 仍作**持有外**基准，不进训练。
* **训练制度**：F1 读出（`readout="predictive"`）全程 `learn=True`；
  `use_memory=True, use_identity=True`（召回条件存在）；写入面开放（Fabric 写入允许）。
* **判读线（冻结）**：CAP D+E 机器计分 **> 0**（对照 `t1@5M=0`、`base=0`）；成句率只报不判；
  0 ⇒ 记"读出对问答条件也无通路"（升级 B/架构侧）。
* **预算**：5M 判读 ≈ 5.5 h → 16M 定论 ≈ 22 h（视判读线定）。

## 6. B-2 / B-3 详细设计：Cortex 神对象拆分

**模式**（已建立样板 `B-1`）：纯函数抽离 + staticmethod 委托 + 黄金向量等价测试。

* **成员**：`detect_modality`（已迁，无 self）✓；`_infer_domain`（依赖 `self.neurons`）；
  `_reencode_domain_generation_context`（依赖 `self.neurons`）。
* **方法**：把 `self.neurons` 提升为参数 ⇒ **签名变 ⇒ 真重构 ⇒ 需负责人签字**。
  每迁一簇，先在 `tests/` 立**黄金向量等价测试**：同一输入下 `Cortex.方法(x)` == `helpers.函数(neurons, x)`
  **逐位相同**，绿后才迁；迁后立刻跑全量测试。
* **顺序**：B-1 `detect_modality` ✓ → B-2 `_infer_domain` → B-2 `_reencode_domain_generation_context`
  → B-3 完整拆分（路由集群 / 域推断集群 / 生成集群）。
* **风险**：冻结基线的静默数值漂移 ⇒ **等价测试先立、不绿不迁**；不签字不动。

## 7. 里程碑索引

| M | 名称 | 状态 |
|---|---|---|
| M5 | 限定退出 | ✅ 2026-09-20 批准落盘 |
| M6 | 产品交付（界面＝Taiji Harness） | 🔄 G1–G3 ✓／G4 判据①②④达成、③待裁定／G5 待做 |
