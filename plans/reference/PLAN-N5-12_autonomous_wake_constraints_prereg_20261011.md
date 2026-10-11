# PLAN-N5-12 自主唤醒行的三条 §10 约束的承载与验收（V-30 的唤醒半边；判据先冻、零算力、不动产品码，2026-10-11）

## §0 为什么是这一件，以及它结的是哪一条

`plans/active/roadmap/09_NEXT_MAINLINE_PLAN.md` §2 第 184 行的 N5 出口交付写着两半：
第一半（首轮真实生长五面读数）已在 ㊵-673 判过（结论＝不成立）；**第二半——"唤醒行为遵守 §10 约束
（噪声不得反复唤醒、次数/预算上限、固定调度对照）"——到今天一个读数都没有**。
条款普查（`scripts/training/audit_taiji_vision_clause_traceability.py`，
`reports/taiji_vision_clause_traceability_20261011_after_n206_n511.json`）对 **V-30（VISION §10 四种自主行为表）**
的判定是 `section_only`：`id_citations=0`、`section_citations=2`，而那两枚锚命中来自
`PLAN-N5-01:40` 等处引用的 **§10「睡眠巩固」判据原型**（"不能只测损失下降"），不是本件的唤醒行。
这正是仪器 `reading_limit` 自己声明的盲区：一节可载多条条款，锚命中不证展开。
⇒ 本件把 V-30 的**唤醒行**展开为"承载实名表＋可判断据"，并登记三条今天实测到的硬事实。
**本件不动 `taiji/`／`seed/`／`api/`，零新算力，不实现任何调度**（理由见 §2 末段：VISION 明写
"先证外部明确触发的学习有效，再学习调度"，那条序还没走完，本件不越序）。

## §1 唤醒行三列在仓内的实际对象（逐格给代码锚；给不出就写未承载）

VISION §10 唤醒行原文三列：输入依据＝"校准的新颖性/失败、可学性和资源"；实际更新＝"选择观察、记忆、学习或求助"；
风险与验收＝"噪声不得反复唤醒；次数/预算上限；固定调度对照"。

| VISION 列 | 仓内实际承载（现读存在） | 今天状态 |
|---|---|---|
| 输入依据（新颖性／失败／资源） | `HomeostaticController.update` 的四个入参 `prediction_error/novelty/reward/resource_cost`（`taiji/homeostasis.py:93-111`）；增益与阈值在 `HomeostaticConfig`（`taiji/homeostasis.py:18-33`，13 枚字段） | 在码；但"校准的"这半没有：入参直接是原始误差／ novelty，无校准件 |
| 实际更新（选择观察／记忆／学习／求助） | 选择器 `HomeostaticController.select_mode`（`taiji/homeostasis.py:83-91`）；挂载口 `TSKV8Adapter.homeostatic_mode`（`taiji/adapter.py:7264-7269`）、`homeostatic_transition(mode="auto")`（`taiji/adapter.py:7271-7294`） | **选择发生但不改行为**：唯一读者是展示面 `api/seed_runtime.py:1667`（`homeostasis_status()` 的 `mode` 键）；`homeostatic_transition` 在全仓 tracked Python 里**零调用者**（`git grep -n homeostatic_transition -- '*.py'` 只命中定义行本身） |
| 风险与验收：噪声不得反复唤醒 | 无承载：`select_mode` 是纯阈值比较、无滞回／死区／最短驻留（`taiji/homeostasis.py:87-91`） | 未承载 ⇒ 本件 J-N5l-2 把它写成可判形式（不新增阈值字面量） |
| 风险与验收：次数／预算上限 | 只有计数器，没有上限：`FastSlowMemoryBank.wake_steps`（`taiji/k_fast_slow.py:112`，`:204` 无条件 `+= 1`，`:281` 入检查点）；`HomeostaticConfig` 的 13 枚字段里没有任何上限字段（现取：`dataclasses.fields` 枚举，无 `cap/limit/max/budget` 命名的字段） | 未承载 ⇒ J-N5l-3 |
| 风险与验收：固定调度对照 | 现成形状：`scripts/training/eval_taiji_p4_homeostasis.py:37-39` 的 `adaptive_mode`／`fixed_mode="wake"`／`random_mode`，以及 `:94-100` 里 `fixed_mode != adaptive_mode` 那一支 | 只有**单 tick、手工挑一个信号点**的对照，不是一条真实轨迹上的两臂 ⇒ J-N5l-4 |

## §2 三条今天实测到的硬事实（本件的登记物，不是推测）

1. **真实回合里 `mode` 是硬编码的**。`taiji/adapter.py:10921`（`observe` 路径）与 `taiji/adapter.py:11660`
   （`settle_action` 路径）两处调 `self._homeostatic_controller.update(...)` 都传字面量 `mode="wake"` ⇒
   产品默认链里驱力**永远按"醒着"演化**，`sleep_recovery`／`sleep_stress_recovery`／`play_fatigue_cost`
   ／`play_stress_relief` 四枚增益在真实回合中从不被走到（`api/seed_runtime.py:195` 确实挂载了控制器，
   所以这不是"没装配"，是"装配了但选择不参与")。
2. **现有仪器测不到这件事，因为它点错了方法名**。`scripts/training/probe_taiji_a27_walked_ledger.py:77`（批次 A 落地后这两格现在 `:81-82`）
   的 homeostatic 格子写的是 `("homeostatic_controller", ("_homeostatic_controller",), "step")`，
   而 `HomeostaticController` 只有 `drive/select_mode/update/checkpoint`（`taiji/homeostasis.py:70-164`），**没有 `step`**
   ⇒ `_resolve_method` 返回 `None`，`_install`（`:108-127`；落地后为 `:114-144`）把它塞进 `absent`，而 `absent` 这个桶
   **同时装着"零件未装配"与"方法名不存在"两种完全不同的事实**——该仪器的 `notes`（`:234-235`；落地后为 `:284-287`）明明写着
   "未装配＝该零件在该 runtime 下根本没装上"／"计数为 0 且未装 ⇒ 两件不同的事：本表把二者分开列"，
   但对"方法名点错"这一支它并没有分开（这条自述比实现宽）。
   根对象没错：`seed/model.py:48` 的 `self.substrate = TSKV8Adapter(...)`、`:55-58` 的 `architecture` 属性
   返回的就是同一个对象，所以 `_homeostatic_controller` 这一跳是够得着的，缺的只是方法名。
   这台仪器是 N5 用来出版"某个零件在五种入口下各被走到几次"的现成件，它的这一格今天给出的是假缺席，
   **而且那份假缺席已经封存在件里**：`reports/taiji_a27_walked_ledger_20260928.json`（tracked）的
   `absent_by_entry` 在 `generate`／`chat`／`observe_action`／`observe_predictive`／`consolidate` 五入口下
   **都含 `homeostatic_controller`**，且五入口的 `absent` 长度一律是 **19**（零件表 26 枚）——
   一个与入口无关的常数，正是"方法名点错"而非"该入口没装"的指纹；批次 A 之后这一列会告诉我们
   19 枚里有几枚属于点错名字、几枚真是默认关。
   **挂载链已按静态读定死**：`SeedRuntime.load`（`api/seed_runtime.py:296`）在 `:335` 构造 `cls(...)`、
   `:348` 返回，而 `SeedRuntime.__init__`（`:167`）第 `:195` 行就调 `ensure_homeostatic_controller()`
   ⇒ 探针五枚入口各自用的**每一枚** runtime 都挂着控制器，所以那条 `absent` 不是"默认关"而是**确证的假缺席**。
   批次 A 之后它应落进 `method_not_found`，且 `update` 那格计数会大于 0（`taiji/adapter.py:10916/11651`
   在 observe／settle 两支里都调它）——**这两句是预期，值由仪器现判**；`select_mode` 那格预期为 0，
   那就是 J-N5l-1 的 `scheduler_not_in_loop`。
3. **"重复唤醒／预算上限"两列在 `reports/` 里没有任何读数**。全集枚举：`grep -rl "wake_steps" reports/`
   只命中 `reports/taiji_m4v2_b3_k_pilot_v2_20260910.json` 与 `reports/taiji_m5_k_p0_equal_replay_diagnostic_20260910.json`
   两枚，且那里它是**步数计数**不是上限；`grep -rln "反复唤醒\|预算上限\|固定调度" reports/` **零命中**
   （命中的全在 `plans/`，是规则文本不是读数）。
4. **同一条缺陷在 `PARTS` 表里至少还有 7 枚**（本件把它从"一格"升成"一族的取法瞎"）。按 AST 静态点验
   （类体直接定义的方法名，不看继承），下列 8 枚格子点的方法名**在该零件的类上不存在**：

   | 零件标签 | 点的名字 | 该类真实的相关方法 |
   |---|---|---|
   | `homeostatic_controller` | `step` | `update`／`select_mode`（`taiji/homeostasis.py:70-164`） |
   | `episodic_memory` | `recall` | `retrieve`（`taiji/episodic_memory.py`） |
   | `semantic_memory` | `recall` | `predict`／`consolidate`（`taiji/semantic_memory.py:60/90`） |
   | `procedural_memory` | `recall` | `predict`／`consolidate`（`taiji/procedural_memory.py:28`） |
   | `procedural_sequence_memory` | `recall` | `predict_episode`（`taiji/procedural_memory.py:198`） |
   | `executive` | `decide` | `select`／`update`（`taiji/executive.py:440`） |
   | `affordance_features` | `observe` | `encode`／`features_for`（`taiji/affordance.py:258`） |
   | `generation_controller` | `step` | `plan_content`／`plan_expression`（`taiji/generation.py:344`） |

   全仓 `git grep "def recall"` 的命中只有 `taiji/identity_organ.py:466`、`taiji/memory.py:321`
   （那是 **A 侧** `EpisodicField.recall`）与 `neuroplex/`／`seed_platform/` 的三处同名不同物。
   另两枚是**另一种形状**，更要紧，因为它不报 absent、而是**恒报 0**：
   `world_dynamics` 点 `forward`——`WorldDynamicsLearner` 是 `nn.Module`
   （`taiji/world_learning.py:1176`）但类体里没有 `forward`，真名是 `predict`
   （被调处＝`taiji/adapter.py:7645/8535/8595`），而 `getattr` 经 `nn.Module` 那枚抛
   `NotImplementedError` 的占位**是可取的** ⇒ 这格会算作"装上了、跑了 0 次"，
   伪装成"零件挂着但没人用"；`language_organ` 点 `render`——`LanguageOrgan` 协议只有
   `backend_id/checkpoint/emit`（`taiji/language_organ.py:1135`），真名是 `emit`
   （被调处＝`taiji/adapter.py:8109`）。
   ⇒ 合计 **26 格里 10 枚名字不对**（8 枚硬缺＋2 枚形状不同）。
   **这条要紧在它的方向**：名字点错不是"历史读数一定错"，而是**这 8 枚格子无论装没装都只会出版 `absent`**
   ⇒ 这台仪器对这 8 枚**没有判别力**（永远测不出"被走到"那一支）。
   已回查 2026-09-28 那棵树（`1052da050`）：那 7 个文件里同样**没有**这些方法名，所以不是后来改名造成的漂移，
   是落笔时点错。至于封存件里那 19 枚 `absent` 有几枚是真的默认关——例如 `episodic_memory` 那枚，
   默认配置 `config.episodic_memory_default_mount` 是 `False`（`taiji/config.py:345`，
   `taiji/adapter.py:299` 只在真值时挂），所以它当时报 absent **可能同时对**——
   这一句由批次 A 的两列现判，本件不预先断言。

**为什么不在此处顺手把调度接进回合**：VISION §10 收尾一句是次序规定——"先证明外部明确触发的学习有效，
再学习调度"。N5 第一半（外部明确触发的生长）已判**不成立**（㊵-673），所以今天把 `mode="auto"` 接进
`observe` 是**越序**，而且它会改产品默认行为、属 owner 半径。本件因此只做两件事：把三条约束写成可判形式，
和修好那台测不到的仪器（仅 `scripts/`）。

## §3 判据（冻；全部沿用已冻公式与在场/计数口径，**不新增任何阈值字面量**——这正是 DEBT-G90 被点名的形状）

* **J-N5l-1 选择是否参与真实回合（在场计数）**：`probe_taiji_a27_walked_ledger.py` 修好后，
  必须同时出版 `homeostatic_controller.update` 与 `homeostatic_controller.select_mode` 在五种入口下的
  **被走到次数＋该入口分母**；`select_mode` 在五入口全为 0 ⇒ 判 `scheduler_not_in_loop`（如实负结果，
  不许读成"控制器没装"——`attached` 是另一格）。分母为 0 沿用该仪器已冻的那支：退出码 2、不出结论。
* **J-N5l-2 噪声不得反复唤醒（翻转计数对照，非阈值）**：同一条确定性干净驱动流与其噪声副本（种子固定、
  幅值取该流自身的最大值，不新造常数），统计两臂在 `select_mode` 上的**模式翻转次数**；
  判据＝两臂翻转数**相等**才许声称噪声不引起反复唤醒；若两臂恒为 0 次翻转 ⇒ **尺不可用**，
  整条记 `ruler_unusable`（沿用 `adjudicate_taiji_n5_shadow_gate.py:299` 的 `ruler_usable` 式，**引用而非复制**；
  行号现取于 2026-10-11：`ruler_usable = bands[...] and line >= band_max` 这行现在在 299，
  ㊵-673 那次提交（`240b20210`，该文件 +28/−10）把它从 295 推到了 299，而四份既有预注册仍写着 295——
  本件不抄那个旧数，见 §8 末段）。
* **J-N5l-3 次数／预算上限存在且可触发**：上限必须是一个**能改变行为的状态**——枚举 `HomeostaticConfig`
  全字段与 `wake_steps` 的全部读者（`git grep`），必须存在一条"越限后 `update`/`select_mode`/`wake_experience`
  输出与未越限不同"的分支；今天不存在 ⇒ 本条按现值直接判 **`cap_absent`（不成立）**。
  负结果同样是 V-30 的消化（基准要的是"展开并判"），但它同时是这条约束**未实现**的登记。
* **J-N5l-4 固定调度对照真的在对照（双臂可分性）**：对照必须跑在**同一条真实轨迹**上（同种子、同信号序列、
  只换臂），两臂＝adaptive 与 fixed-`wake`；判据＝两臂的模式序列必须**可分**（逐位相同 ⇒ 判
  `arm_not_actually_two`，这是"双臂要先证真的是两臂"那条既有教训的形状）；可分性取法复用
  PLAN-N5-07 的 `mode_readings`／幅度现读形状。并且这条必须附带一句自述：**当前产品默认已经就是固定调度臂**
  （§2 事实 1），所以"过了固定调度对照"今天只能读成"与默认无差别"，读不成"自适应调度有效"。

## §4 实施批次（哪些格子需要谁的批准）

| 批次 | 内容 | 触面 | 前置／批文 |
|---|---|---|---|
| A | 修 `probe_taiji_a27_walked_ledger.py`：**新增** `not_attached` 与 `method_not_found` 两列并保留 `absent`（＝两列之并，历史读数不追改、必需键不扩），homeostatic 格的 method 由 `step` 改 `update`，并新增 `select_mode` 一枚零件；**外加一条落盘护栏**：该仪器 `--out-report` 的默认值就指着已封存的 tracked 件 `reports/taiji_a27_walked_ledger_20260928.json`（`:169`；落地后默认落点在 `:198-200`，已是新件 `taiji_a27_walked_ledger_20261011.json`），而 `:257` 是无条件 `write_text`（落地后写入点在 `:311`，其前是护栏 `_guard_overwrite`：定义 `:181`、调用 `:309`） ⇒ 今天直接重跑会**覆写历史读数**；本条要把默认落点改成带日期的新件，并对"目标已存在且已入库"的情形响亮拒绝（要覆写需显式旗标）；范围不只 homeostatic 一格——**26 枚零件名逐枚点验**方法是否存在（§2 事实 4 已静态抓到 4 枚点错名），错名的按各自类的真实读方向方法改（§2 事实 4 已静态抓到 **10** 枚：8 枚硬缺＋`world_dynamics`／`language_organ` 两枚形状不同），历史标签名保留以便与新件逐列对账；按该仪器已冻的两条守卫自证（插桩不改行为＝同入口输出逐字节相同、基座 sha256 跑前后相同） | 仅 `scripts/`、`tests/` | 我可自办，无需批文 |
| B | §3 四条判据的取数件（J-N5l-2／4 的两臂跑面＝同进程内喂确定性流，不训练、不更新权重） | 仅 `scripts/` | 批次 A 绿之后；我可自办 |
| C | 把调度接进真实回合（`mode="auto"` 进 `observe`／`settle_action`）＋滞回＋cap 字段 | `taiji/adapter.py`、`taiji/homeostasis.py` | **owner 批文**，且受 §2 末段的次序约束——须先有"外部触发学习有效"的正读数 |
| D | 出 §10 唤醒行的对外表述 | 不改码 | 批次 B 出值之后，且要满足 §7 第三条前置 |

## §5 判据载体在哪（措辞门要求的一节）

J-N5l-1 的执行体＝`scripts/training/probe_taiji_a27_walked_ledger.py`（批次 A 修好后）；
J-N5l-2／4 的执行体＝批次 B 新落的双臂取数件，落之前本件只能声明"判据已冻、尚不可判"；
J-N5l-3 的执行体＝本件 §1/§2 的字段枚举（它是**在场性**判定，今天就已出值＝`cap_absent`）；
"尺不可用"那一支的执行体＝`adjudicate_taiji_n5_shadow_gate.py:295`（引用不复制）。
⇒ 本件没有一条判据是手算的；J-N5l-3 已出值（负），其余三条待批次 A／B。

## §6 出口（三条互斥）

* 四条都出值且 J-N5l-1/2/4 全过 ⇒ V-30 的唤醒行转为"已展开且已验收"，09 第 184 行第二半转为已交付；
* 任一判据出值但判**不成立**（今天极可能走这一支：J-N5l-1 会是 `scheduler_not_in_loop`、J-N5l-3 已是
  `cap_absent`）⇒ 仍算 V-30 唤醒行**已消化**，并把负结果按 09 §4 分流归到 N5 的路线图上，
  同时结清"§1.3 里 V-30 只有锚级命中"这一格（普查的 `id_citations` 从 0 变 1）；
* 若停在 `unverified_*`／`ruler_unusable` ⇒ 本件只算设计件，**不得**声明 V-30 唤醒行已消化，
  且收回"09 第 184 行第二半有读数"这句。

## §7 落笔即钉的三行

* **预期为绿的成员**：J-N5l-3（它判的是"上限不存在"这一在场性事实，取法今天就能复算）；
  批次 A 的两条自证守卫（该仪器已有同形先例：插桩不改行为的逐字节比较）。
* **仍红时收回的句子并点名**：若 J-N5l-2 落 `ruler_unusable` ⇒ 收回"噪声反复唤醒已被测过"并点名是本件的
  尺出口生效，不是产品变差；若批次 A 之后 `select_mode` 计数仍为 0 而 `update` 计数也为 0 ⇒ 先怀疑我自己
  又点错方法名（§2 事实 2 就是这个形状），不得直接写"控制器未被走到"。
* **发表资格前置**：任何"N5 的 §10 唤醒约束已过"的对外表述必须同时给出 ①四条判据各自的值，
  ②双臂确为两臂的证据（模式序列可分＋同轨迹同种子），③§2 事实 1 那句限定（当前默认＝固定调度臂）。
  三条缺一即视为未判。

## §8 不变项

M6 收官、M8 挂起；N1／N2／N3／N4 收口状态不变；N5 第一半出口已判不成立（㊵-673），本件不翻那笔账、
只补第二半的读数面；本件**不新增阈值字面量、不新增张量参数面、不动产品码、不开训练**；
批次 C 归 owner 且受 VISION §10 次序约束。债表至 **DEBT-G106**，本件按批次 A／B 结果另开
（批次 A 的方法名混桶＝一条仪器债；§2 事实 1 的"装配了但选择不参与"＝一条产品债，均归 owner 或我自办，
按落地时实判登记，不在起草时预填编号）。

**顺带结清的一条我自己造成的锚漂移**：`ruler_usable` 那行被 ㊵-673 的提交 `240b20210` 从
`adjudicate_taiji_n5_shadow_gate.py:295` 推到 `:299`，而 `PLAN-N2-06`／`PLAN-N4-08`／`PLAN-N5-10`／
`PLAN-N5-11` 四份都写着 295（它们是抄前一手的引用、没有回查现行行号）。四份都是已冻判据件，
按"原文不删、带日期就地补注"处理：各在引用句后补一句现行行号，不改判据口径。
教训同源于既有那条"改到被钉住的行要重钉新形状"——**改过被 PLAN 引用的脚本行之后，必须按文件名回扫引用面**；
今天这种回扫是手做的（`grep -rn "adjudicate_taiji_n5_shadow_gate.py:2" plans/`），
是否做成常设门由 owner 裁（它会像存在性门那样需要一条"只守存在性还是守行号一致"的口径决定）。


## §9 批次 A 落地与 J-N5l-1 的首次出值（2026-10-11，同轮）

仪器＝`scripts/training/probe_taiji_a27_walked_ledger.py`（已按 §4 批次 A 改完），
读数件＝`reports/taiji_a27_walked_ledger_20261011.json`，`PROBE_RC=0`，三条自证守卫全 True
（`base_sha256_unchanged`／`instrumentation_is_behavior_neutral`／`every_entry_has_denominator`）。
契约册＝`tests/taiji_native/test_n5_22_walked_ledger_columns_contract.py`，**9 passed**。

* **点名修正后的格子数**：`parts_probed` 26 → **27**（新增 `homeostatic_select_mode`）；
  10 枚错名全部改成"真实被调/真实存在"的名字（`retrieve`／`predict`×3／`select`／`encode`／
  `plan_content`／`emit`／`update`），`methods_by_part` 里可逐枚回读。
* **`method_not_found` 五入口一律为空**，`not_attached` 每入口 **15** 枚（旧 `absent` 是 19）。
  两列的并仍出版成 `absent`，与旧列直接对账：19 − 15 = **4**，而 `moved_out_of_absent` 现读正好是
  `affordance_features`／`executive`／`homeostatic_controller`／`language_organ` 这四枚
  ——即"装着但名字点错"的那四枚；余下 15 枚是真默认关。`newly_listed` 为空（没有格子从计数掉进 absent）。
* **J-N5l-1 出值＝`scheduler_not_in_loop`（判不成立）**：`homeostatic_controller`（＝`update`）
  在五入口被走到 `generate 42／chat 279／observe_action 4／observe_predictive 4／consolidate 0` 次，
  而 `homeostatic_select_mode` **五入口全 0**。这就是 §2 事实 1（`mode="wake"` 硬编码）的仪器侧对照读数。
  按 §6 第二条出口：V-30 的唤醒行至此**算已消化**（展开并判），负结果按 09 §4 分流归 N5。
* **本件仍未出值的**：J-N5l-2（噪声翻转对照）与 J-N5l-4（同轨迹双臂可分）要批次 B 的取数件，
  J-N5l-3 已按在场性判 `cap_absent`。**不得**据本节宣称"§10 三条约束已过"——过的是"已判"，
  不是"判成"。

* **普查的第六次读数**（`reports/taiji_vision_clause_traceability_20261011_after_n512.json`）：V-30 从 `section_only` 转为**有 id 级引用**，`clauses_with_id_citation` 7 → **8**，锚命中 19、unmapped 0、分母 41 行与 required_now 19 未动；`every_clause_has_id_level_citation` 仍为假（8/19）。09 §1.3 的 V-30 承载列已同步换成这一行。**这不等于 §10 三条约束已过**——过的是「展开并判」，J-N5l-2／4 仍待批次 B。


## §10 批次 B 落地：J-N5l-2 与 J-N5l-4 的首次出值（2026-10-11，同轮）

仪器＝`scripts/training/measure_taiji_n5_12_wake_schedule_arms.py`（零训练、同进程喂确定性流），
读数件＝`reports/taiji_n5_12_wake_schedule_arms_first_values_20261011.json`，契约册＝
`tests/taiji_native/test_n5_23_wake_schedule_arms_contract.py`（**8 passed**，两支出口都单独打到），
`ARMS_RC=0`。夹具＝24 步同向爬升流（后半段叠负奖励），种子 7；判据幅值**按本节 §3 已冻原文**
取「该流自身的最大值」＝1.0。

* **J-N5l-4 = `arms_separable`**：同一条流上自适应臂 24 步里翻转 **13** 次，模式只出 `wake`／`sleep`
  两枚（`play` 在这条流里从未被选中）；固定臂 24 步全 `wake` ⇒ 两臂序列逐位可分、确为两臂。
  **但这句的含金量和 §2 末段一致**：固定臂就是现行产品默认（`mode` 写死），
  所以「可分」只证『自适应与默认不同』；有信息量的是两臂的驱力终态差——自适应臂
  `fatigue 0.26726／stress 0.36`，固定臂 `fatigue 1.0／stress 1.0`（双双饱和），
  即睡眠那条恢复支（以及整条 `play` 支）在默认链里从不生效。
* **J-N5l-2 = `noise_changes_scheduling`（判不成立）**：干净流翻转 **13** 次，噪声副本 **15** 次。
  `ruler_usable=true`（两档翻转都 > 0；可用性只按自适应臂要求——固定臂按定义恒 0，
  拿它当动态范围会把「对照臂本应恒定」误读成「尺没有动态范围」）。
  补充梯（逐路步尺，不参与判定）：幅值 0.0435 → 14 次、幅值 1.0 → 15 次，**两档都不等于 13**
  ⇒ 负结论在整条梯上同向，不是某一档的产物。
* **本轮我自己抓到的两处取法错，都在出值前修掉并留痕**：
  ①噪声幅值第一版把四路展平取相邻差，跨路跳变（`reward` 0.0→−1.0）把幅值抬到 **2.0**（超出信号
  可表示范围，那是饱和不是噪声），当时读数 13→14；
  ②`perturb` 的回落范围硬写 [0, 1]，于是 `reward` 的 −1.0 被**系统性**抹平成 0.0——这条与噪声无关，
  是副本与干净流之间多了一处真实差异。它是被契约册的恒等支（`amplitude=0` 必须逐位相同）抓出来的；
  修成「回落进每一路自己的取值范围」后重算＝**13→15**（前一个 13→15 读数在修 [0,1] 那版里同值，
  但归因不干净，故按修后重算为准）。
* **登记一条判据自身的缺陷（DEBT-G109）**：J-N5l-2 冻的是「幅值取该流自身的最大值」，
  而本轮实测**结论对幅度选择敏感**——按各路自己的步尺同时扰动时读到过 13→13（＝判成立）。
  我**没有**采纳那一档：已冻原文 governs，混合逐路版本只作为被否证的对照留在台账里。
  修法（要 owner 或下一件预注册）＝把判据写成「对整条内生尺度梯的最坏情形都要求翻转数不变」，
  而不是单点幅值。

⇒ §6 的三条出口现在有了归属：J-N5l-1（`scheduler_not_in_loop`）、J-N5l-2（`noise_changes_scheduling`）、
J-N5l-3（`cap_absent`）、J-N5l-4（`arms_separable`）**四条全部出值**，其中三条判不成立 ⇒
走 §6 第二条出口：**V-30 的唤醒行算已消化**（基准要的是「展开并判」），负结果按 09 §4 分流归 N5，
09 第 184 行第二半从「零读数」转为「有读数且为负」。
