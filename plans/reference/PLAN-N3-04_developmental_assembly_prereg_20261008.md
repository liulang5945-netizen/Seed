# PLAN-N3-04 · N3 乙步骤二预注册：把快慢写入打开，看压强是否真有动态范围（2026-10-08 起草；**判据先冻、不开跑**）

> **来源与效力**：owner 2026-10-08 第二次弹窗裁 N3 乙＝"**丙：开 developmental 学习模式**"（09 §3.2 第 3 条、㊵-490①），前置定义件＝[PLAN-N3-03](PLAN-N3-03_tau_definition_20261008.md)。
> 本件冻结假设、面、判据、守卫与预算，**不构成开跑授权**：两臂都会**改权重**（symbol 流训练），须 owner 批（㊵-485 那条纪律）。成熟度＝§13 V1→V2 之间（单机制在线面，尚无能力增量声称）。

## 0. 先更正我自己给的选项形状（这条决定本件为什么要写成两臂）

弹窗里我把丙描述成"四维才齐"，**这是错的**，实测与代码都反对它：

- `AdaptiveResidualBridge.activity_saturation` 量的是 `self.region.activity ≥ target_activity` 的单元比例（[taiji/adaptive_residual_bridge.py:101-107](../../taiji/adaptive_residual_bridge.py)）；
- 而 bridge 的前向在 `if self._gate == 0.0 or self._lesioned: return ...` 处**提前退出**（同文件 :145），
  即 `gate=0.0` 时 region 从不 `step()` ⇒ 活动恒零 ⇒ `activity_saturation` 恒零——与 ㊵-484③ 的实测（两条链整场恒 0）对上。

⇒ **丙单独只解 `fast_slow_conflict` 一维；要 `activity_saturation` 活着必须同时把 bridge 放行（`gate>0`），而放行＝改读出行为**（模型输出会变）。
所以本件把**两臂都冻下来**，由 owner 点其一：**丙（只开快慢写入）** 与 **丙＋乙（快慢写入＋bridge 放行）**。
点错或不敢点都不算缺陷，**如实写出"这条弹窗选项我给错了"才算**——这条更正同时进 09 §3.2 第 3 条与台账 ㊵-492。

## 1. 假设与面

**H-N3b（在场性而非大小）**：今天 `pressure = 0.425·residual ≤ 0.425` 而默认 `minimum_pressure=0.70` ⇒ `should_propose` 恒假（㊵-484③，4000 步 0 次）。
若把快慢写入（必要时加 bridge 放行）打开后**五信号里原本恒零的那几维开始产出非零值**，则 `pressure` 的上界与分布都会改变 ⇒ 才有资格谈 τ；
若打开后恒零维仍然恒零 ⇒ "接线"这条被否证，问题在**生产者本身不在这条链上运行**，按 09 §4 归因第五行处理（保留旧内核、把生长侧降级）。

**本件不声称**能力增量、不声称容量解除——它只回答一件事：**四维（或三维）在场的压强分布长什么样，τ 能不能冻**。

面（全部沿用现成仪器，不重抄生成链）：

| 项 | 冻结形状 |
|---|---|
| 链 | `train_seed_corpus.py` 的 **symbol 流分支**＋`--readout predictive`（answer／self-answer 档走 `learn_bytes`，那条路上**一个压强读数都不会有**，㊵-483② 已钉） |
| 记录 | 既有默认关旗标 `--pressure-record`（bridge `gate=0.0` 挂载＋默认 policy＋包住 `trigger.observe`，㊵-484 已入库） |
| 新臂 | 新**默认关**仪器旗标 `--developmental-fast-slow`：`migrate_f1_to_developmental_synapses()` → `set_developmental_f1_learning_mode("fast_slow")`，**且每次 load/resume 之后重新施加**（模式不入档：`taiji/model.py:1331-1332`、:3785） |
| 放行档 | 丙＋乙那一臂再加 `--developmental-bridge-gate <值>`（**值随批文冻**，不在跑后调；候选 0.25 与 1.0，取自 canary `eval_taiji_m4v2_r4_shadow.py` 的 `gate=1.0` 先例） |
| 读法 | `count_taiji_n3_pressure_thresholds.py`（覆盖率守卫 ≥0.95、`pressure` 与五信号加权和的**身份校验**、口径否证支）＋本件新增的**在场性列**（见 §3） |

## 2. 判据（先于任何读数冻结；合取式，且**每支都能为 false**）

- **J-N3b-在场（主判据）**＝打开的那一维**整场非零**：`fast_slow_conflict` 的 `zero_count / records < 0.05`，
  且其 p90 > 0；（丙＋乙臂额外要求 `activity_saturation` 同判）。
  **不过 ⇒ 判"该维在这条链上仍不产出"**，本件不产出 τ，也不许改用 `pressure` 的均值当替代结论。
- **J-N3b-阈（次判据，只在在场成立后才谈）**＝按 [PLAN-N3-03 §4](PLAN-N3-03_tau_definition_20261008.md) 的规则现取：`τ = ceil(p90(pressure) → 0.05 档)`，
  并要求**动态范围证明**＝面内 `should_propose` 为真的连续段 ≥ `required_pressure_steps`（默认 3）至少 1 次；
  无动态范围 ⇒ 该 τ 不得发表（N3-03 §4 第 5 条）。
- **口径否证支先跑**＝`|residual_error p90 − (1 − online_accuracy)| > 0.10` ⇒ 判"与 accuracy 不同源"，τ 只能来自本面（㊵-484 已两次触发，预期它会再触发；这条**不是失败**，是限制 τ 的来源）。
- **合取**：J-N3b-在场 ∧ J-N3b-阈（含动态范围）才判"乙步骤二给出了可冻的 τ"；否则**整件不结项**，τ 一格继续空着。

## 3. 面内自述（本件硬要求：在场性不许由命令行反推）

判读器输出必须**自带**这五列，缺一列 ⇒ `unverified` 不判：

| 列 | 含义 |
|---|---|
| `dims_present[<五维>]` | 每一维的 `nonzero_count / records`、p50、p90、max |
| `assembly_self_report` | 面件自述：`developmental_bundle_mounted`、`learning_mode`（每次 load 后重报一次）、`bridge_gate` 实际值 |
| `mode_reapplied_after_load` | 若面内发生过 load/resume：每次 load 之后的 `learning_mode` 读数列表；任一为 `read_only` ⇒ 记 violation、整件不判（这条就是把 §1 那条硬事实变成机检） |
| `pressure_identity` | `pressure` 与五信号加权和的逐条重算差（现仪器已有，沿用） |
| `coverage` | 观测数 ÷ 面内 tick ≥ 0.95（现仪器已有，沿用） |

## 4. 守卫（四条，任一不符即整件不判）

- **G-N3b-1 默认关＝逐位不变**：不开新旗标时，同参两支的进度行八键（`ticks`/`window_ticks`/`online_accuracy`/`mean_surprise`/`holdout_surprise`/`base_ticks`/`ticks_at_exit`/`reached_budget`）必须逐位相同——沿用 ㊵-484⑥ 已跑通的形状，**不许继承**它对本旗标的结论（不同旗标≠不同仪器）。
- **G-N3b-2 放行档不假设中性**：丙＋乙那一臂**明确不是**行为中性改动（bridge 参与读出＝输出变了），故该臂所有 accuracy／泛化数一律定性为"**放行装配下的数**"，不得与今日出厂面同表比较（㊵-483 G-N3-1 的收紧同条）。
- **G-N3b-3 只 propose 不 promote**：全程不调 `propose_adaptive_residual_growth_candidate()`，不改产品默认 policy 常量（`minimum_pressure=0.70` 不动）；本件答的是"分布长什么样"，不是"要不要放行生长"。
- **G-N3b-4 不碰产品件**：一律 `--checkpoint output/` 新档；`DEFAULT_CHECKPOINT` 与厂档不动（02 §2.2 既有纪律）。

## 5. 预算（用 ㊵-491 的标定实测外推，不估）

| 项 | 值 | 出处 |
|---|---|---|
| 每臂符号上限 | **4000**（与 ㊵-484 两张面同量，判据是分布形状而非趋势） | 本件冻结 |
| 吞吐（fresh 小档） | `--scale 10`＝61.04 tick/s；budget 档＝97.94 tick/s | `reports/taiji_n3a_calibration_cost_20261008.json` |
| 本臂实际档位 | 与 ㊵-484 同底（`--resume checkpoints/seed_a31self_with_circuit.pt` 与 `output/a31_chunked_self/checkpoint.pt`），非 ×16.8 大档 | 本件冻结 |
| 预计墙钟 | 两支链 × 两臂 ＝ 4 张面，每张历史耗时 ≈ 数分钟（㊵-484 同形状已跑通） | 在库先例 |
| 磁盘 | 面件 ×4 ＋ 每臂一档（≈12.7 MB 量级） | 在库先例 |
| 止损 | 面内 `fast_slow_conflict` 前 500 步全零 ⇒ 早停并记"该维不产出"，不改判据 | 本件冻结 |

**待 owner＝批哪一臂开跑（两臂都改权重）**，以及丙＋乙臂的 `bridge gate` 取值（候选 0.25／1.0）。

## 4bis. 批文参数（owner 2026-10-08 第三次弹窗后的落地文本；判据与守卫一字未改）

- **开跑臂＝丙＋乙**（快慢写入打开 **且** bridge 放行），owner 明确接受"这一臂改了读出"的定性；
- **gate 取两个值都跑**＝`0.25` 与 `1.0`（`1.0` 与既有 canary `eval_taiji_m4v2_r4_shadow.py` 同档，`0.25` 是小剂量探边）；
- 面数因此冻结为＝**2 条链 × 2 个 gate 值＝4 张在线面**（链＝`--resume checkpoints/seed_a31self_with_circuit.pt` 与 `--resume output/a31_chunked_self/checkpoint.pt`，
  与 ㊵-484 同底同 seed 20260822），另加**一支默认关守卫臂**（G-N3b-1：同参开/不开旗标的八键逐位对照，不许继承㊵-484⑥ 对 `--pressure-record` 的证明）；
- 每臂上限仍 `--max-symbols 4000`、`--progress-every 200000`；预算＝5 支面，按 ㊵-491 标定的量级为**分钟级/张**（与 ㊵-484 两张面同形状，其墙钟有件可查），磁盘＝面件 5 枚＋每支训练臂一档（≈12.7 MB 量级，落 `output/`）；
- 止损照 §5；**跑完不追加改 gate 的第二次判据级重跑**（判据级不重跑），gate 的两档差只作机制定位叙述。

## 6. 本件不做什么

不冻产品阈、不改 policy 常量、不 promote 任何候选、不动出厂面；不做 ×16.8 大档（那是 N3 甲的另一条泳道）；
不声称"经验→能力通道打通"——那句话现在仍卡在 **DEBT-G47**（修法甲已实施，但本轮那枚候选档救不回，巩固后面必须在**修法甲之下新跑的一次通电**上重取，而那是第二次改权重）。

## 7. 不变项

M6 收官、M8 挂起；M7-CI 绿只在推送后实测；N1（H-S1b 待批）、N2（巩固后重跑待批）、N3 甲（×16.8 档取哪一档待批）三格 owner 待批不变；N4/N5 后置；台账行序以行首标号为准（㊵-419⑥）。
