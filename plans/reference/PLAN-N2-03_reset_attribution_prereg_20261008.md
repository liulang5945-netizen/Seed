# PLAN-N2-03 · "代价来自收束还是巩固"分离实验预注册（2026-10-08 起草，**判据先冻、跑前不解释**）

> **来源与效力**：09 §4 分流第四行（"适应后遗忘旧能力 ⇒ 查巩固写入范围与保持集 §8.3 候选版本隔离"），
> 输入＝[PLAN-N2-02 判读](PLAN-N2-02_ADJUDICATION_20261008.md) 里那 7 列跌破的读数。
> **本件不第三次改权重**：待验的那一臂按代码事实是**权重中性**的（下面 §1 给锚点），所以它不需要新的改权重批文；
> 若跑出来发现该臂权重变了 ⇒ 整件不判并回 owner（§4 G-N6-2）。
> 成熟度：这是**归因实验**，不声称任何能力结论；它的产出只有一句"代价住在哪一层"。
> **2026-10-08 已跑完并判读**：判 **`cost_from_weight_update`**（`n_reproduced = 0/7`，七列 `attrib_R` 全 `−0.0`），
> 读数件 `reports/taiji_n2_03_attribution_20261008.json`，判读在 [PLAN-N2-03_ADJUDICATION_20261008](PLAN-N2-03_ADJUDICATION_20261008.md)。
> **本件的 §2/§3 判据与守卫此后不得再改**；要改口径只能另立一件（§3ter 那次是冻结当天的自我更正，不属于此列）。

## 0. 为什么现在要做这一件（本轮量出来的一条事实）

对同一趟通电的两枚档做只读比对（`checkpoints/seed_n2b_mother_20261008.pt` 对 `seed_n2b_candidate_20261008.pt`，
本机现算、两枚都由同一支 `SeedRuntime.save()` 写出 ⇒ 条目名可比）：

* **权重/结构族**：300 个共有张量叶子里 **138 个有差**，集中在 `substrate.fabric.*`
  （`decoders`／`consolidation_decoders`／`laterals`／`transitions`／`trace_baselines`，含稀疏索引 `pre_index` 最大差 224.0）、
  `substrate.memory.*_readout`（九支全动）、`substrate.motor.*`、`substrate.identity_organ.bank`、`substrate.state.regions[0..2]`。
* **情节/工作记忆族**：**7,148 个叶子在候选档里消失**，且**只**集中在
  `substrate.cognitive_state.assemblies[i].*`、`…events[i].*`、`…memory.working_ids/working_items`、`…observation`、`…percept`
  （候选档里这些容器变成 `[]`／`None`）。

第二族不是巩固学出来的，而是 **[DEBT-G47 修法甲](../active/roadmap/05_TECH_DEBT_REGISTER.md) 那一步 `reset_dynamics` 清的**：
`Taiji.reset_dynamics` 的 docstring 自己写"Clear activity while preserving all learned synapses"，函数体
（[taiji/model.py:1825-1842](../../taiji/model.py)）做四件事＋**整块换掉 `_state`**：
`fabric.clear_cue_snapshot()`、`copy_circuit.drop_selection_lock()`、`adaptive_residual_bridge.reset_dynamics()`、
`response_plan_readout.clear_plan()`、`self._state = self._initial_state(...)`。
⇒ **"落盘可载"与"情节态被清空"是同一行代码的两面**，而 run-2 掉的恰好是
CAP-0 **E 简单指令与推理** 1→0 与复述**成句数四档全跌**——这两类正是可能被情节/工作记忆与选锁喂着的读数。
（**这是待验假设，不是结论**；下面冻的就是让它能被判假的形状。）

## 1. 臂与面（零改权重；同一台仪器、同一题集，只换 `--checkpoint`）

* **Arm R（收束臂）**＝装载母档 → **只调一次** `Seed.reset_dynamics(episode_id="wake-after-sleep")`（与 `sleep_pass` 里那一步**同一式子、同一常量**）
  → `SeedRuntime.save()` 落 **`output/n2b_resetarm/checkpoint.pt`** 新档（产品件不动，G-N2-4 同条）；
  全程不 `observe`、不 `learn`、不调 `sleep_pass.run`。
* 面（与 run-2 完全同名同参，只换档）：`eval_taiji_cap0_baseline.py`（100 项）、
  `measure_taiji_a30_repetition_penalty.py --limit 24`（四惩罚档）、
  `probe_taiji_a30_stop_failure.py` ＋ `count_taiji_a30_eater_p_boundary_floor.py`（×24 主列）。
* 基线两侧都用已入库件：**巩固前**＝`reports/taiji_n2_cap0_before_20261008.json` 等三张（其与回退面逐位同已两次实证）；
  **治疗后**＝`reports/taiji_n2b_*_after_20261008.json` 三张。

## 2. 判据（先冻；每列一个数，中间带也预先给名字）

对 run-2 判为跌破的那 7 列（`CAP-0:E`、`复述命中 1.0/2.0`、`复述成句 0.0/0.5/1.0/2.0`）逐列定义**归因分数**：

```
attrib_R = (R − before) / (after − before)          # 两母都取严格命中/成句的绝对计数
```

* 分母恒不为 0：这 7 列都是 run-2 实测**跌了 ≥1** 的列（`after − before ≤ −1`）⇒ 分数可解释；
  若仪器读到分母为 0 ⇒ 该列记 `unverified` 并从分母列数里剔除（同时把列数写进件里）。
* **J-N6-归因（主判据，三档互斥、阈值先冻）**：
  * `attrib_R ≥ 0.5` 的列数 **≥ 5** ⇒ 判 **"代价主要来自收束"**（⇒ 回 owner 重裁 DEBT-G47：修法乙〔save 前以 kernel 半边为权威、不清情节态〕重新上桌）；
  * 该列数 **2 ≤ n ≤ 4** ⇒ 判 **`partially_resolved`**（两族都在贡献 ⇒ 保持集设计必须同时覆盖"回合收束时保留哪些态"与"权重写入范围"）；
  * 该列数 **≤ 1** ⇒ 判 **"代价主要来自权重更新"**（⇒ 保持集＝§8.3 候选版本隔离那一支，收束无罪）。
* **主列（`never_lf_eaters_counted`）不入归因分数**：它 run-2 是**改善**（56→47）不是跌破，方向又已定为越小越好 ⇒ 只作旁证列报出，不参与判级。
* 合取侧不需要：本件不判"加速点①"，只出上面那一格。

## 3. 守卫（四条，任一不符即整件不判）

* **G-N6-1 该臂必须真的没学**：Arm R 存盘前取一次 `content_digest`、`reset_dynamics` 之后再取一次，
  并与母档摘要比 ⇒ **三者必须逐位同**；任一不同 ⇒ 这一臂就不是"只收束"，整件不判并回 owner。
* **G-N6-2 装载中性**：Arm R 的档 `SeedRuntime.load` 必须成功（DEBT-G47 已修，这条应绿），
  且 load 后摘要等于母档摘要 ⇒ 否则读数面读的不是被检的那份状态。
* **G-N6-3 面同源**：三张面的 `eval_set_sha256` 与主列 `items_sha256` 必须与基线两侧一致，
  任一不同 ⇒ 拒判（跨题集作差＝换尺）。
* **G-N6-4 不新增仪器**：读数一律走 run-2 那三台现成仪器与 `count_taiji_n2b_faces.py` 的取数函数，
  **不重算生成链**；驱动只负责"装载→收束→存盘"。

## 3ter. 冻结当天的一处**口径更正**（我自己的守卫先为假，而且假在我写错了定义）

初稿的 G-N6-1 写的是"`reset_dynamics` 前后 `content_digest(model.checkpoint())` 逐位同"。第一次实跑就**红**：
`3e00bafc6298ac84…` → `7e3e640a49e00fca…`。**红因不是产品，是我把"信封摘要"当成了"权重摘要"**——
`content_digest(checkpoint())` 覆盖整份信封，而 `reset_dynamics` 的操作对象正是信封里的情节/活动那几族。
⇒ 同一次装载下现算的家族差（母档→只收束，本机实测）：

* **掉的叶子 1,612 枚**，全部在 `substrate.cognitive_state.events[i].*`／`…assemblies[i].*` 等情节族；
* **68 处变化**，家族直方图＝`taiji.kernel.state` 27、`substrate.state.regions[0..2]` 各 7、
  `substrate.perception.dynamic` 3、`substrate.state.memory` 3、`taiji.components.perception` 3、`substrate.cognitive_state.memory` 2——
  **全是活动/情节态**；
* **参数族零差**：`substrate.fabric.*`（decoders／consolidation_decoders／laterals／transitions／trace_baselines 与稀疏索引）、
  `substrate.memory.*_readout`、`substrate.motor.*`、`substrate.identity_organ.*` 这些 run-2 候选档里有差的族，
  在"只收束"这一臂里**一项都没动**（与 `Taiji.reset_dynamics` 的 docstring "preserving all learned synapses" 一致）。

⇒ **更正后的 G-N6-1**：臂档与源档按**参数族**比，要求**零差**；情节/活动族的变化**不是缺陷而是这一臂的处理本身**，
必须作为自述报出（`dropped_episodic_leaves`／`changed_state_entries`／族直方图），不许被当成"这臂不干净"而拒判。
这条更正让实验反而更干净：**Arm R 恰好只复现 run-2 两族变化里的那一族（情节清空），参数族严格为零** ⇒
若它复现 7 列跌破，代价就归到收束；不复现，就归到权重更新。**单变量分离成立，且成立是被量出来的，不是被假设的。**

## 4. 预算（读码＋三支面，全部零改权重）

| 项 | 值 | 依据 |
|---|---|---|
| Arm R 驱动 | 一次装载＋一次 `reset_dynamics`＋一次存盘 | 秒级 |
| CAP-0 面 | ≈6–10 分钟 | run-2 实测（同参数、同 100 项） |
| 复述 ×24 | ≈2.5–3 分钟 | 同上 |
| 停摆 ×24 ＋主列 | ≈2.5–3 分钟 | 同上 |
| 磁盘 | 一枚 ≈12.7 MB 档（`output/`）＋三张读数件 | 在库先例 |
| 判据级 | 不重跑；机械修复 1 次仅限仪器 | 既有纪律 |

## 5. 出口与去向

* 出口＝**一格归因**＋七列的 `attrib_R` 数值＋四条守卫的实测；负结果同样入库。
* 若判"主要来自收束" ⇒ 本件**不自行改产品码**：DEBT-G47 的修法甲/乙之争带证据回 owner（甲＝能落盘但清情节态，乙＝不清但把权威交给 kernel 半边），
  并且要重新说明一句：**修法甲之下读到的"代价"里有一部分不是巩固的**。
* 若判"主要来自权重更新" ⇒ 保持集设计照 §8.3 走（候选版本不覆盖 parent、L_keep 与调参集分离），那才是第三件改权重实验的输入。
* 若是中间带 ⇒ 两件都要做，顺序按本件读数大小定（不引入新判据）。

## 6. 本件不做什么

不改权重、不动 `sleep_pass` 默认位、不放大剂量、不给"能力恢复了多少"的承诺；
不把 7 列的跌破解释成机制（只给归因分数）；不借这件重裁 CAP-0 的判据粒度（那是 ㊵-499⑦ 的 DEBT-G51 那一族）。

## 7. 不变项

M6 收官、M8 挂起；M7-CI 绿只在推送后实测；`DEFAULT_CHECKPOINT` 与厂档不动（Arm R 写 `output/`）；
甲臂起跑与 `git push origin main` 两格仍等 owner；N3 乙 τ 那格空着；N4/N5 后置；
台账行序以行首标号为准（㊵-419⑥）。
