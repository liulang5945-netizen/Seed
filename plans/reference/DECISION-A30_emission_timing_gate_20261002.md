# DECISION-A30·发射时序门控要不要进产品（2026-10-02）

**决策问题**：给复制回路的证据发射加一条**生命周期门控**（只在答复前 K 步发射，之后静音），
做成产品改动、默认关闭（⇒ 未开启时逐位不变）。**这一格需要 owner 裁；本件不替 owner 下结论，只把可裁的部分摊平。**

## 一、已测事实（全部仪器侧、零训练、产品源码零改动）

三面各有**两枚回路**的证据，件名全文（引用时必须点名枚数，见 §50/§51 与守卫 60fbf91c）：

| 面（判据） | seed-A | seed-B | 读法 |
| --- | --- | --- | --- |
| 自身轨迹真自停 L2（/72） | 不挂 66｜全剂量 23｜窗口 **63**（Δ+40） | 全剂量 29｜窗口 **64**（Δ+35）；不挂回路上界 **66/72**——**已复核**：这台 L2 仪器一趟一配置（`copy_circuit_present_after_load=true`），未挂回路臂住在**单独件**里，两件独立运行同值 66/72（`reports/taiji_a30_stop_failure_c_nocircuit_positions_v16_20261002.json`、`reports/taiji_a30_stop_failure_chunked_short_nocircuit_20261002.json`，同 `checkpoint=output/a31_chunked_short/checkpoint.pt`、同 24 题、同 `max_length=256`） | 窗口把停止买回，跨两枚复现 ⇒ 通道性质 |
| 表层成句 `well_formed_texts`（104 题，`product_surface_chat`） | 不挂 13｜全剂量 3｜窗口 **15**（Δ+12） | 不挂 13｜全剂量 6｜窗口 **13**（Δ+7） | 同上；两枚各读作"窗口后与不挂回路相当"，**不是**"胜过不挂回路" |
| 复述 D 命中（24 题，cap 面） | 不挂 0｜全剂量 6｜窗口 **7**（Δ+1） | 不挂 0｜全剂量 3｜窗口 **3**（Δ0） | 方向不否证，但**水平依枚减半** ⇒ "两全"必须带枚数说 |

件名：
`reports/taiji_a30_stop_failure_c_window64_v17_20261002.json`、
`reports/taiji_a30_stop_failure_c_seedB_anchor_20261002.json`、
`reports/taiji_a30_stop_failure_c_seedB_window64_20261002.json`、
`reports/taiji_a30_surface_tradeoff_window_anchor_20261002.json`、
`reports/taiji_a30_surface_tradeoff_window64_20261002.json`、
`reports/taiji_a30_surface_tradeoff_seedB_anchor_20261002.json`、
`reports/taiji_a30_surface_tradeoff_seedB_window64_20261002.json`、
`reports/taiji_a30_cap_dual_arm_chunked_short_budget256_20261002.json`、
`reports/taiji_a30_cap_dual_arm_chunked_short_window64_20261002.json`、
`reports/taiji_a30_cap_dual_arm_chunked_short_seedB_anchor_20261002.json`、
`reports/taiji_a30_cap_dual_arm_chunked_short_seedB_window64_20261002.json`、
`reports/taiji_a30_stop_failure_c_nocircuit_positions_v16_20261002.json`、
`reports/taiji_a30_stop_failure_chunked_short_nocircuit_20261002.json`。

K 的来历：由定价档按**开档前写死的规则**得出（命中偏移 max=60、停止决定 0 次落在前 16 字节 ⇒ K=64），
不是从网格里挑的；**K 不因工件重标**（它由 (c) 面命中分布定价）。

## 二、反对证据（必须同页呈现，不许只报增益）

1. **现产品默认基底上没到线**：`a31_chunked_self`＋回路，K=64（跨底 transplant）Δ+18、K=128（本底定价）Δ+8，
   **都低于预注册的 +20 线** ⇒ 换到现默认底，这条修法今天不构成"能力提升"。
2. **交回一小截逐字命中**：表层链 `strict_hits` seed-A 40→35、seed-B 15→11；而逐字命中**只来自挂回路**
   （不挂回路两枚都是 0）⇒ "窗口买回表层形态"不等于"窗口免费"。
3. **产品侧今天零生效**：产品默认链的真自停仍是 **13/72**；以上全是仪器侧读数。
   **（本条写作时尚真，现已被 §五 取代）**：门已落在产品代码与产品装配入口里，出货底上开启后读到 **23/72**；
   但**默认位仍是 `None`** ⇒ 装机那份确实仍是 13/72，"生效"仍系于 owner 开不开默认位（§五 末格①）。

## 三、若批准，验收式（两条都可机检，且今天已证明"能为假"）

* **甲·默认关闭 ⇒ 逐位不变**：同一仪器版本、同一参数，开/关两趟必须逐字段相同。
  今天已用同一段比较器验过它的两面：两次独立运行的**未挂回路对照 15 个字段逐位相同**（⇒ 能为真），
  而全剂量臂与窗口臂同一比较器立刻报出差异（⇒ 能为假）。
  **已知误报源**：跨**仪器版本**整块比会把 schema 差异当成行为差异
  （实测第一处差异是 `treated_with_circuit.answer_bytes_quantiles` 一边 `None` 一边 `dict`）。
  ⇒ 验收必须**同版本**比，或只比行为子集（逐 item 的字节序列与 argmax 序列）。
  **比较器已落仓内**（不再依赖仓外临时脚本）：`scripts/training/compare_taiji_a30_report_identity.py`
  ＋守卫 `tests/taiji_native/test_a30_report_identity_comparator.py`（4 passed）。用法
  `python scripts/training/compare_taiji_a30_report_identity.py --left <件A> --right <件B> [--subtree <键>] [--strict]`，
  **rc 就是结论**（0 ⇒ 逐位相同，1 ⇒ 不同），可直接挂门。它把差异分成两堆：
  `behavior`（两边都有值而不同）与 `schema`（一侧缺键，或一侧 `None` 一侧有值——即"旧件按零补"的指纹），
  默认只按 `behavior` 判，`--strict` 时两堆都算 ⇒ 守卫同时钉住了"能为真""能为假""strict 一开会翻"。
* **乙·开启后的能力线**：L2 相对同底同题面的"不挂回路"上界的差距按 §1 三线判（≥3 才算位移），
  复述 D 与表层 `well_formed_texts` 各按其链现取参考值，**禁止跨链搬阈值**（τ=0.3 跨链那一枪已付过学费）。
  开关字段与既有两通道同族（默认 1.0/关闭 ⇒ 逐位不变），并自带"被走到"计数（`emitted`／`silenced` 两侧非零），
  否则档是空的（v12 那次"包装器被走到 ≠ 过滤器开过枪"已踩过）。

## 四、owner 的三选一 → **已裁并已执行到"默认关闭"**（2026-10-02 21:35 裁，23:36 状态）

1. ~~**不立项**~~：未被选。
2. ~~**只继续仪器侧**~~：未被选。
3. **立项进产品（默认关闭）＝已选并执行**。已做完的：产品侧参数 `TaijiConfig.copy_evidence_window_steps=None`、
   注入点前的生命周期门、**每趟答复的复位**、`emitted/silenced/steps_seen` 自证；
   产品装配入口 `SeedRuntime.enable_copy_circuit(..., window_steps=None)` 也已能把它传进去（此前"进代码没进入口"）。
   三条验收式的实测状态见 §五。

**边界**：表层链与原始字节链是两把尺；任何"窗口后仍有 7/16 逐字命中"不点名回路的句子都是错的（守卫 60fbf91c 已钉）。

## 五、执行后的读数（这一节取代 §一/§二/§三 里"仪器侧"口径的部分；旧文不删，按件可追）

| 面（同底同题面） | 关（默认） | **产品门开启** | 仪器替身档 | 判读 |
| --- | --- | --- | --- | --- |
| 停止·(c) 底 seed-A | 23/72 | **60/72** | 63/72 | Δ=−3 ⇒ 产品路径复现 |
| 停止·**出货底** `a31_chunked_self`（本底定价 K=128） | **13/72** | **23/72** | 21/72 | Δ=+2 复现；**净增益 +10** |
| 停止·出货底（跨底搬 K=64） | 13/72 | 28/72 | 31/72 | Δ=−3 复现；+15 **不能当本底成绩** |
| 复述·cap 面 K=64 | D 6 | **D 6 保持** | D 7 | 差 1 枚低于分辨率 ⇒ 没丢 |
| 表层·`product_surface_chat` K=64 | 3 | **16** | 15 | Δ=+1 复现；对不挂回路 13 差 3＝"相当" |

* 不挂回路参照：(c) 底 **66/72**、出货底 **52/72**、表层 13、复述 0。
* **甲已升级为逐 item 机检**：出货底与 (c) 底的"关档"分别与改源码之前的旧件比 `per_item`
  ⇒ `identical=true`／behavior 0／schema 0／rc=0（`scripts/training/compare_taiji_a30_report_identity.py`）。
* **乙已在两面通过**（`|Δ| ≤ 3`）⇒ 今夜所有替身档结论在产品代码路径上成立。
* **丙（门开过枪）已在三台仪器件内自证**：cap `emitted_steps=64／silenced_steps=58／steps_seen=122`、
  表层 `64／20／84`、L2 出货底 `64／192／256`。

**因此现在能说的与不能说的**：
* 能说：机制在产品代码里、从产品装配入口可达、默认关闭经逐 item 实证不改任何行为、三列都复现替身档。
* **不能说**："能力提升已达当初判据"——出货底按本底定价只有 **+10**（当初写死的复现线是 **+20**）；
  也不能说"产品变好了"——**默认位仍是 `None`，装机那份仍是 13/72**。

**还待 owner 的两格**（都不是我能替你裁的）：
① **开不开默认位**（`copy_evidence_window_steps` 从 `None` 改成该底定价 K；开＝装机读到 23/72，代价是表层逐字命中交回 4~5 枚）；
② **+10 这个量级**是接受，还是承认停止问题在出货底上需要另一种机制
（**不是换 K**——K 网格、阈值、剂量、上限、寻址、候选集六族已定价，见队首"别再花的钱"）。
