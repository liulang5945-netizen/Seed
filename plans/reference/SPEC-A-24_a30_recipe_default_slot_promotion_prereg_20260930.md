# A SPEC-A-24 预注册：**"带配方的训练件能不能换掉产品默认基座"这条决定的判据**（2026-09-30，先于任何新重训档冻结）

前件：`PLAN-A-30_surface_repetition_localization_20260928.md` §8①（那条写了"这条线是本单新提的，
不在 §2ab 的预注册内"）。本件就是补那句所缺的预注册：**现在还没有新件**，所以这三条线是在看到任何
新数之前写下的。本件**只追加、不改写**；要动口径另开 SPEC-A-25，并在本件顶部登记"§X 已被 SPEC-A-25 取代"。

## §0 本件定价的是哪一句话

允许写的最高结论（若三条全过）：
**"带配方的重训件在装机形态上不比现默认件差，且在自己的输出轨迹上第一次能停"** ⇒ 可提产品默认位变更。
**不允许**由本件直接写出的：任何关于"模型学会了停止"的一般能力结论（§2n 的教训：离线好解在自己轨迹上蒸发）。

## §1 三条线（一台仪器一个方向；缺键一律 fail-closed 记 `not_resolved`）

| 编号 | 仪器（命令面固定） | 指标键 | 方向 | 线 | 参考值来源（已入库件） |
|---|---|---|---|---|---|
| L1 表层成句 | `scripts/training/score_taiji_r2_copy_surface_extension.py --checkpoint <新件> [--circuit output/taiji_r2_copy_circuit_chat/judge/circuit-final.pt] --out-report <新名>` | `well_formed_texts` | 越高越好 | **≥ 0.9 × 同装配参考值** | 不挂回路＝241/260（0.9269）；挂回路＝100/260（0.3846）——`reports/taiji_f0_a26p1_surface_recheck_20260930.json`、`reports/taiji_f0_a26p1_surface_circuit_20260930.json` |
| L2 自身轨迹真终止 | `scripts/training/probe_taiji_a30_stop_failure.py`（挂 seed-A 回路、预算 256、惩罚 2.0、24 题 × 3 轮＝72 次生成、`all_surfaces_are_replayed_raw=true`） | `generations_cut_by_turn_marker` | 越高越好 | **≥ 6 次／72** | 现状两枚装配都是 0（§2ah） |
| L3 决策面胜出 | `scripts/training/probe_taiji_a30_ding3_transfer.py`（`format` v2 互斥四支，300 个自身轨迹接缝） | `boundary_argmax_positions` | 越高越好 | **新件 − 对照 ≥ 3** | 现重训件 1/300 对 base 0/300（`reports/taiji_a30_ding3_transfer_300pos_b_20260930.json`） |

**L1 的装配必须点名**（这是本件存在的第一个理由）：同一枚 base 件在两种装配下参考值差 2.4 倍，
拿错一侧等于把判据定在另一个数上。**两条侧各自独立成线**，不合并、不取平均：
不挂回路侧 ≥217 条（241×0.9=216.9），挂回路侧 ≥90 条（100×0.9）。

**L2 的单位澄清**（原文 §2af 写作"≥6/24"，分母是题数；本件把它落到**计数单位**上＝72 次生成里至少 6 次
因边界符退出环）。两者在"至少 6 次"这一点上取同一个数，但**引用时必须写清是哪一种**，否则会出现
"6/24"被读成 25% 的口径事故。

## §2 明令**不算证据**的四项（我这一轮已经量到它们会诱人误用）

1. **教师强制面上的胜出数**（§2z/§2aa 的 96/120、239/300）：那是"强度"证据，不是"停止"证据——
   §2ah/§2aq 已经量过它在自身轨迹上不掉过来。可以引用，但**不许**当 L2/L3 的替代。
2. **`utf8_decodable_trimmed_rate`**：它先削掉末尾再打分，对本件要防的退化**无判别力**
   （两枚件都报 1.000，而 `well_formed_texts` 是 241 对 40）。
3. **`answers_stopped_early`**：这是"答复短且无接缝"的混装类别，里面既有边界符停、也有 raw 里有接缝被截——
   判"停"必须在 raw 层数（§2ae 的更正）。
4. **差 <3 的同字拖写 offender 变化**（§2ah 的 8→10）：不过分辨率线就只能写 `not_resolved`，
   既不写"变坏"，也不写"没变坏"。

## §3 独立性与配对（不许在这一格省事）

* L1 若要走 SPEC-A-21 那条"表层劣化／改善"的**正式判定**，仍需**两次独立取数**（两条独立初始化的电路）同向；
  本件的 L1 是**晋升线**（对参考值的比值），不取代那条判定，也不得把"比值过线"写成"劣化不成立"。
* 每条读数必须带**五个绑定量**：设备、链路／装配（挂不挂回路）、checkpoint（`checkpoint_sha256_before`）、
  生成预算、分布（题集 `manifest_sha256`）。缺任一个 ⇒ 该读数不进本件的判读。
* 与对照**按字节指纹配对**，不按"参数看起来一样"。

## §4 三条不过会怎样（预先写明，避免事后换尺子）

* 只有 L2/L3 过、L1 不过 ⇒ **不换默认位**，配方留给后续训练（§8① 的丙），并把退化单独登记成债务。
* 只有 L1 过、L2/L3 不过 ⇒ 这是"没代价但也没得到"，**不许**据此说配方有效。
* 三条都不过 ⇒ 记 `not_resolved`，本支线在此收口等裁定，**不许**改线再跑（换次要指标是本项目明令禁止的那类动作）。

## §5 命令面补全（2026-09-30 同日追加；**不改任何线，只把"照抄就能跑"写全**）

冻结件里最容易失效的一类是"命令面只写了仪器名"。逐条核实过参数名后补全（三个仪器的键名与默认值都是从源码里读的，
不是凭记忆）：

* **L1**（两种装配各跑一次，只换 `--checkpoint`）：
  `python scripts/training/score_taiji_r2_copy_surface_extension.py --checkpoint <新件> --out-report <新名>`（不挂回路侧）
  与 `... --checkpoint <新件> --circuit output/taiji_r2_copy_circuit_chat/judge/circuit-final.pt --out-report <新名>`（挂回路侧）。
  键：`well_formed_texts`（分母 260）。参考值：241／100（见 §1 那两张件）。
* **L2**：`python scripts/training/probe_taiji_a30_stop_failure.py --checkpoint <新件> --circuit <回路> --limit 24 --max-length 256 --penalty 2.0 --out-report <新名>`
  ⇒ 分母是 `--limit 24` 题 × 3 轮＝**72 次生成**；线＝`generations_cut_by_turn_marker ≥ 6`；
  守卫键 `all_surfaces_are_replayed_raw` 必须为 `true`，否则这张面读的是截断层、不算。
  **`--circuit` 不给＝出厂那面（无回路）**，那是另一张面：**不许**拿无回路面去比 §2ah 的 0/72（那是挂回路面）。
  ⇒ **本件 2026-09-30 补一条实测更正**：`--circuit` 不给 **也不等于**"无回路面"——
  `checkpoints/seed_beta_with_circuit.pt` 这类**带回路的信封在 `SeedRuntime.load` 里就自动挂载**
  （冒烟读数：`mount_route=envelope_auto_mount`、`copy_circuit_present_after_load=true`）。
  所以 L2 的取数面**按件里的 `mount_route` 判，不按命令行猜**；仪器自 v4 起加性存这三条
  （`mount_route`／`copy_circuit_present_after_load`／`copy_evidence_utf8_gate_effective`）。
  **另记一条我自己差点写错的口径**：证据门的**有效值不等于 `config` 那一位**——`taiji/config.py:236` 默认 `False`，
  而 restore 的自动挂载分支会 `set_copy_evidence_utf8_gate(True)`（owner 裁定 (b)，`taiji/model.py:3497`）；
  我第一版把 `config` 当成了有效值报出来（会读成"门是关的"），已就地改成报有效值并同带 `config`/`override` 两个成分。
  **归属要点**：这条事实本身**产品侧已有守卫**——`tests/taiji_native/test_a25_gate_on_the_load_path.py`
  六支用例（含 `test_restoring_a_saved_circuit_opens_the_evidence_gate`、
  `test_the_gate_is_live_after_restore_not_just_flagged`）钉着"restore 自动挂载 ⇒ 门是开的不只是标了旗"，
  实测这六支 2026-09-30 全绿 ⇒ **缺口在仪器侧（报错了列），不在产品侧**，不要为这条去动产品码。
  **L2 在产品默认件（自动挂回路那面）上的基线读数尚未取**：在飞那次用的是修字段前的仪器，其 gate 列不作证据。
  ⇒ **2026-09-30 已用 v4 取到**（件 `reports/taiji_a30_stop_failure_defaultload_20260930.json`，`rc_stop_default_v4=0`）：
  `mount_route=envelope_auto_mount`、`copy_circuit_present_after_load=true`、
  `copy_evidence_utf8_gate_effective=true`（`config=false`／`override=true`）、24 题 × 3 轮＝72 次生成
  ——**`generations_cut_by_turn_marker = 0`、`generations_eating_full_budget = 72`**，四条守卫全真。
  ⇒ **L2 的对照基线定在这里：现状装机件 0/72 ⇒ 晋升判定要求新件 ≥6/72**（这条线不是本件新加的，是 §2af 冻结下来的）。
  **L3 的对照也已钉到装机件（2026-09-30 续跑落地）**：件
  `reports/taiji_a30_ding3_transfer_300pos_defaultload_vs_a26_20260930.json`——
  装机件 `seed_beta_with_circuit` **0/300** 对 血缘基座 `a26_p1` **0/300**，两臂 `prompts_sha256` 相同（配对成立），
  仪器自己的互斥分支落在"两臂都零 ⇒ 不成立"那一支。
  ⇒ 按 §8① 事先写死的规矩：**此后 L3 的对照引装机件 0/300，不再引 a26**（"用户今天加载的那一枚"才是晋升判定的对手）。
  **发表条件**：同参数的一次重复任务当时也在写同一件 ⇒ 读数按"两份逐位相同"才发表（同参数幂等，若不同要按取数面重查）。
  **机械锚点（供下一次核）**：当轮件的 `sha256` 前 16 位 `7a4bcc1ba4f6cc35`、`started_utc=2026-09-30T04:26:46Z`、
  两臂 `boundary_argmax_positions` 都是 **0**（分母 300）。重复任务落盘后先比这两个字段，再谈读数。
  ⇒ **两次写入都到齐了，核下来是"读数字段全同、字节不同"**：
  第二写 `started_utc=2026-09-30T04:47:27Z`、`sha16=f30ba3929ff97e60`，两臂仍 `0/300`、verdict 同文。
  **本件据此把发表条件改成可执行的形式**（这是**修我自己的工具口径**，不是放宽判据）：
  比较的是**测量字段**（`runs.*.boundary_argmax_positions`／分母／verdict／`prompts_sha256`），
  **不比整文件字节**——件里带墙钟 `started_utc`，字节相等是永远达不到的标准，
  拿它当门槛只会逼人发表"一次"读数而丢掉这次这种"两次独立跑同值"的加强证据。
  同件另有 `offender_count = 5`（同字拖写），**只作描述**：§2ah 那两个数（8／10）是另两枚件在同一仪器同参数下的读数，
  比较线 ≥3 只对本件预注册的那一对（新件 对 对照件）生效，**不许**拿 5 去和 8/10 讲"变好了"。
  ⇒ **L2 的"指向哪一列"就地改正（数值线一字未动，且改在任何新件读数之前）**：
  读产品码见 `api/seed_runtime.py:455-459` 生成环是 **`stop_at_boundary=True`**
  ——模型在自己写的答案里让**边界符**成为 argmax 时，`chat()` 是**直接退出**，答复里并**不会**出现 `\n问：` 这个接缝。
  所以"自然终止"在件里对应的列是 `generations_eating_full_budget`（应当**下降**）与 `boundary_argmax_steps`（应当 **>0**），
  而我原先写的 `generations_cut_by_turn_marker` 只量 **marker 截断**那一类退出。
  **改正后的 L2 读法**：新件在同一仪器、同参数（24 项 × 3 轮＝72 次生成、预算 256、惩罚 2.0、`all_surfaces_are_replayed_raw=true`）下
  **`72 − generations_eating_full_budget ≥ 6`**（＝至少 6 次因边界符退出环，对照现状 0），
  `generations_cut_by_turn_marker` 保留为**旁列**（它仍在 §2ah 那两张件上可读，但配方效应不走这条路）。
  原句留在上面不删——这条改正本身就是要留下的教训：**判据冻结时要点名"哪一列、朝哪个方向"，
  只写现象名（"自然终止"）会让我自己指着错的列读平一条本来能动的线。**

* **L3**：`python scripts/training/probe_taiji_a30_ding3_transfer.py --retrain <新件> --base <对照件> --positions 300 --mask --out-report <新名>`
  ⇒ 键 `boundary_argmax_positions`，线＝两臂之差 ≥3；配对守卫 `prompts_sha256` 两边必须相等，
  件里 `format` 必须是 **v2**（互斥四支）；v1 件缺胜者列，属"字段不存在"而不是"算出了 0"。

## §8 「戊 可行性探针」的判读线（2026-09-30 追加；**写在读数之前**，不是本件的晋升线）

本件 §1 那三条线定价的是"能不能换默认位"。现在多一格待答的问题：**密度这一手能不能把真自停买过 ≥6 那条线**。
探针档（件 `reports/taiji_a30_onpolicy_quarter_probe_8g3x6e_20260930.json`，臂 `corpus/self/sized/quarter`，
`--gen-max 64`、`--groups 8 --exchanges 3 --epochs 6`）落盘后，用**同一台** L2 仪器复测 `quarter` 臂
（`probe_taiji_a30_stop_failure.py --checkpoint <存下的 quarter 臂> --circuit <seed-A 回路>`，72 次生成）。
**线（三分支，互斥，先于数）**：

* `72 − generations_eating_full_budget ≥ 6` ⇒ **密度可买到真自停** ⇒ 才有资格提请正式训练档（规模、时长、验收同带）。
* `3–5` ⇒ 方向继续存在但**不过线** ⇒ 只能写"随密度上升的单调趋势"，**不许**写"会停"。
* `≤ 2` ⇒ 密度对**真自停**不转移（即便接缝面 17/24 成立）⇒ 那是一堵新墙，
  正确结论是"**接缝级胜出可买、环内自停买不到**"，并据此把 §2aw 的正向读数降级为代理读数。

**三条硬边界**：①探针的训练量极小（8 组 × 3 答 × 6 epoch），它只回答"这一手有没有方向"，
**不构成任何能力主张**；②`quarter` 臂是把语料答案截到 1/4 的**人造短答**，真实对话里的答案长度不由我们指定 ⇒
若这一手成立，落地前要先回答"多短的收尾目标才算合法训练分布"；③本探针**不**改动 §1 三条线，也不替换它们。


## §7 在飞两档的**发表前置**（2026-09-30 追加，同样先于数）

读数之前先核这几条，任何一条不成立 ⇒ **不发表结论**，只发表"这档没做成对照"：

1. **配对三件**（`probe_taiji_a30_onpolicy_shape_pilot.py`）：`instrument_guard` 里
   `arms_share_the_same_questions`／`arms_differ_only_in_answer_author`／`base_untouched_after_run` 必须全为 `true`；
   `heldout_disjoint_from_trained` 必须为 `true`（held-out 与训练集相交则任何"学会"都是背题）。
2. **密度阶梯的单调体积**：`bytes_by_arm` 必须满足 `corpus > half > quarter`。
   若不满足（短答案本来就短、截了也没变），三臂就**不是**只差密度 ⇒ `verdict_length_ladder` 不发表。
3. **作者对照的体积配平质量**：`self` 与 `sized` 的字节差**不得超过 15%**（v1 实测 7%，通过）。
   超了就要写明"体积未配平"，`verdict_matched_volume` 只能作描述。
4. **位置数与件内 `positions` 一致**（v1 主档 18、v2 36、阶梯档 24），且引用时必须带分母。
5. **两臂/多臂之间除了名义变量，不许有第二处不同**：引用前先复看 `facade_reachable_both_arms`
   与三臂 `base_sha` 是否同源（这条就是 v1 那处 `expected_calls` 界错的教训的正面形态——**界要看它是否真的能拦住**）。

**判读线（不动）**：作者线 `self − sized ≥ 3 ⇒ author_holds`／`≤ −3 ⇒ author_negative`；
密度线 `quarter − corpus ≥ 3` 且单调 `⇒ length_holds`；主判据 `self − corpus ≥ 3 ⇒ shape_holds`；其余 `not_resolved`。
**三条线互不替换**：哪条读平就写哪条读平，不许拿另一条的动静冒充进展。


## §6 L1 的判别范围已被就地收窄（2026-09-30 追加；**线本身不动**）

`well_formed` 的第一道条件（UTF-8 合法）**在 `str` 上恒真**，且实测有带 U+FFFD 的答复被放行
（`reports/taiji_a30_surface_gate_fffd_bypass_20260930.json`：a26_p1 件 104 行里 88 行
`well_formed=true` 且预览含 U+FFFD，a31 件 9 行里 9 行；正文 §2as／`DEBT-G17`）⇒

* **L1 实际量的是**：够长 ∧ 单字占比不超限 ∧ 语料 n 元平均分过线。**不量**字节合法性、不量有没有说真话。
* **线不改**（改线＝事后放宽）：`well_formed_texts` 的两侧参考值 241／100 与"≥0.9×"照旧，
  因为两臂用同一条尺子，**相对高低不受这条缺陷影响**。
* **引用 L1 时必须同带一句**："过门≠字节合法"；若 owner 批准把 `DEBT-G17` 那一行补进判据，
  则本件 §1 的**两个参考值作废、须重录**（那是新档，不是本件就地更新）。
* **那条线的价格已先算出来**（PLAN-A-30 §2at，用已入库四张件反推，不需机器时间）：补上"整条可解码"后
  `well_formed_texts` 的**上限**是 不挂回路 12（a26）／2（a31）、挂回路 15（a26）／9（a31），对现值 241／40／100／24
  ⇒ 方向不变、量级塌十倍。**这两组数不许互换**：本件的 L1 只认现定义下的 241／100。
* **L1 也不量"通顺"**：`well_formed` ＝ 够长 ∧ 单字占比不超限 ∧ 语料 n 元平均分过线。
  过门样例（`reports/taiji_a30_surface_gate_fffd_bypass_20260930.json` 的 `samples`）长这样：
  `同，何）：吂何，大家夐：这首诗有吂有，一老…`。⇒ **L1 过线只能说"像中文的形状"，不许写成"答复通顺"**。

## §7 G17 落地后的 L1 参考值重录（2026-09-30，owner 批 PLAN-A-30 §7b-4；按本件 §6 预告以新节重录，§1 不就地改）

**判据更正**：`well_formed` 的"UTF-8 合法"检查已修到能真拒——lone surrogate（encode 抛
`UnicodeEncodeError`）与含 U+FFFD 的文本一律不过门；产品副本（`seed/surface_gate.py`）与
仪器副本（`diag_taiji_r2_surface_decode.py`）同改，漂移守卫保持两副本逐位同。
本更正改变判定集合 ⇒ **2026-09-30 更正前入库的 `well_formed_texts`（241/40/100/24）是旧定义读数**，
与新读数不可直比。

**L1 参考值重录（owner 裁定口径＝按 §2at 上限）**：

| 装配 | 旧参考值（不查字节） | 新参考值（上限口径） | ≥0.9× 线 |
|---|---|---|---|
| a26_p1 不挂回路 | 241/260 | **≤12**/260 | ≥11 条 |
| a26_p1 挂回路 | 100/260 | **≤15**/260 | ≥14 条 |

* 12/15 曾是**上限**（§2at 反推）；**2026-09-30 晚已重跑落定实值**（PLAN-A-30 §2bd 同表）：
  不挂回路 **12/260**、挂回路 **5/260**（a26_p1，G17 修补后判据，`reports/taiji_f0_a26p1_surface_g17_20260930.json`）。
  L1 线随之读作：新件不挂回路 ≥11、挂回路 ≥5（0.9× 后取整）。
* a31 侧对应上限 ≤2（不挂回路）／≤9（挂回路），供差价对照。
* **L1 线本身不动**（仍 ≥0.9× 同装配参考值），只是尺子补齐了那道缺失的条件；两臂自此同尺。
* 门槛①②（回写门槛／坏答复不进历史）随本更正**实际拦截变更严**——U+FFFD 类答复自此不过门，
  行为变化已由 owner 批准（§7b-4），入账。

## §8 晋升状态（2026-10-01，按本件三线判读后执行）

**候选 `a31_chunked_self`（自写档）三线全过**：L1 39/260 不挂、6/260 挂（线 ≥11/≥5）；
L2 13/72（线 ≥6）；L3 18/300（线 ≥3）。回退检查（本件未列但同批记为纪律）：F0 `floor_pass`。
⇒ 按本件 §5 的冻结分支（三线齐 ⇒ 可换）**已执行换底**（PLAN-A-30 §7c；回滚点已留）。
本件自此转为**已用**：后续候选按同一三线判；L1 参考值随新默认重定（新默认自身读数 39/6，
引用旧 12/5 时必须点名"旧默认件"）。
