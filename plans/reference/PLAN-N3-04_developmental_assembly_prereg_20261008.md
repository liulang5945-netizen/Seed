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
  **⇒ 这一格里"链"那一半已于同日作废**（两条在训链带着 `position_weight`，进不了这档装配；实证与替代链见 §5ter）——**判据、守卫、gate 两档与"4 张在线面＋1 支守卫臂"的面数都没变**，变的只是从哪枚档热启动，所以它要 owner 点、不自行换；
- 每臂上限仍 `--max-symbols 4000`、`--progress-every 200000`；预算＝5 支面，按 ㊵-491 标定的量级为**分钟级/张**（与 ㊵-484 两张面同形状，其墙钟有件可查），磁盘＝面件 5 枚＋每支训练臂一档（≈12.7 MB 量级，落 `output/`）；
- 止损照 §5；**跑完不追加改 gate 的第二次判据级重跑**（判据级不重跑），gate 的两档差只作机制定位叙述。

## 5bis. 仪器形状（本件的实施规格，写在跑之前；默认关，落地不需新批）

挂载点＝`scripts/training/train_seed_corpus.py` 里 `if pressure_record is not None:` 那一段（现锚 :456-…，
㊵-484 的 `--pressure-record` 就是在这里 `enable_adaptive_residual_bridge(gate=0.0)` ＋ `enable_adaptive_residual_growth()`）。
本件在同一段加**两个默认关旗标**，互不隐含：

- `--developmental-fast-slow`（store_true）⇒ 依次 `substrate.migrate_f1_to_developmental_synapses()` →
  `substrate.set_developmental_f1_learning_mode("fast_slow")`；
  **必须挂在 bridge 挂载之后、任何 load/resume 之后重挂一次**（模式不入档，`taiji/model.py:1331-1332`、:3785）。
- `--developmental-bridge-gate <float>`（0.0–1.0，默认 None＝不调）⇒ 把挂载时的 `gate=0.0` 换成该值，
  并在挂载后 `set_adaptive_residual_bridge_gate(...)` 复核一次实际生效值。
  owner 已裁两个值都跑：`0.25` 与 `1.0`（`1.0` 与 canary `eval_taiji_m4v2_r4_shadow.py` 的 `gate=1.0` 同档）。

面件自述（`kind=face` 那行必须新增，缺键 ⇒ 读数器 rc=2）：

| 键 | 取法 |
|---|---|
| `developmental_bundle_mounted` | 迁移返回件的 `fast_is_zero`／`effective_parameter_count`（**注意：挂载即 `fast_is_zero=true`**，见 §0） |
| `learning_mode` | `substrate.developmental_f1_learning_mode`（每次 load 之后重报一次，进下面那个列表） |
| `mode_reapplied_after_load` | 一个列表：每次 load/resume 之后读到的 `learning_mode`；**任一元素是 `read_only` ⇒ 整面判 violation**（G-N3b 的机检形状） |
| `bridge_gate_requested` / `bridge_gate_actual` | 请求值与挂载后读回值，两者不等即响亮失败 |
| `activity_saturation_nonzero_count` | 从压强记录里现算（不新写生产者）；丙臂预期为 0，这一格是"我这臂到底解了几维"的自证 |

两支契约测（形状照 `tests/taiji_native/test_a4_mainline_flags.py` 的"默认关逐键等于入参"路子）：
①**默认关逐位不变**：不开两旗标 vs 开任一旗标之外的同参两支，进度行八键逐位相同（G-N3b-1，**不许继承** ㊵-484⑥ 对 `--pressure-record` 的证明）；
②**开旗标必须五自述键齐 + `mode_reapplied_after_load` 里出现 `read_only` 即 rc=2**（造一份缺键件与一份含 `read_only` 的件各跑一次，两条拒绝分支都要真走到——㊵-484⑤ 的教训）。

跑的形状：`--max-symbols 4000 --progress-every 200000 --seed 20260822`，两条链各一次（`--resume checkpoints/seed_a31self_with_circuit.pt`、`--resume output/a31_chunked_self/checkpoint.pt`），
gate 两档各一遍 ⇒ 4 张在线面＋1 支默认关守卫臂；一律 `--checkpoint output/` 新档（G-N3b-4）。

## 5ter. 实施当天（2026-10-08）落仪器时撞到的前置——判据与守卫一字未改，改的是**命令形状**

- **装配与位置输入互斥（产品守卫，不是我的选择）**。实测：按训练器现行默认（`--readout-position` 自 2026-09-28
  起默认开）跑第一支烟测，产品在 `Taiji.migrate_f1_to_developmental_synapses()` 的起手就响亮拒绝——
  `ValueError: readout_utf8_position_input is not wired to the developmental F1 learning path`
  （守卫本体 `taiji/model.py:1124-1135`，理由写在它自己的注释里：**两者同开会得到一个静默空转**，
  前向喂零列、后向不更新，“看起来装了、其实等于没装”）。
- **本件 §4bis 冻结的链不可执行（实施当天两支 canary 都是 rc 级实证；这条更正覆盖我先前那句"五支面带上 `--no-readout-position` 就行"——它不成立）**：
  ① `--resume checkpoints/seed_a31self_with_circuit.pt` **带** `--no-readout-position` ⇒
  `ValueError: checkpoint carries UTF-8 position columns but readout_utf8_position_input is disabled`
  （守卫 [taiji/organs.py:925](../../taiji/organs.py) 的 `_load_position_weight`）；
  ② 同链**不带**该旗标 ⇒ 位置输入开着 ⇒ 撞上条上面那条互斥拒绝。
  ⇒ **两条在训链都进不了这档装配**，原因是它们的权重里带着位置列（只读取证：两枚档的
  `kernel.predictive_readout` 键集都是 `['bias','format','position_weight','synapses']`，两处 config 都写
  `readout_utf8_position_input=true`），而位置输入**不许半路关掉**（关掉＝拿带位置的权重去配不吃位置的读出，产品按设计拒绝）。
  ⇒ §4bis 里"链＝这两枚、与 ㊵-484 同底"这一格**作废**（作废的是**面的形状**；§2 判据与 §4 守卫一字未改），
  **乙步骤二在替代链被点之前不起跑**。
- **可执行的替代链已经 priced（零额外探测成本）**：出厂链 `checkpoints/seed_beta.pt` **不带位置列**
  （同一取法下它的 `predictive_readout` 键集里没有 `position_weight`，config 也没这个键）⇒
  `--resume checkpoints/seed_beta.pt --readout predictive --no-readout-position --developmental-fast-slow
  --developmental-bridge-gate 0.25` 实测 **rc=0**（200 符号 canary、2.316 s、727,923 参数、`base_ticks=16,000,000`，
  件在 `output/n3_04_canary_factory/`）。canary 面内 `fast_slow_conflict` 非零 198/199（max 0.500016）、
  `activity_saturation` 非零 199/199（max 0.416667）、`pressure` max 0.679682（mean 0.604986）、
  `learning_mode` 挂载后与收尾都读到 `fast_slow`。**只有 199 条观测**（预算 200）⇒ 判读器按 §5 的样本下限与覆盖率会拒判，
  所以这一支**只算"链可用"的实证，不进任何判据**。外推：这条链 4,000 符号每支 ≈ **46 秒**，
  5 支面**分钟级**、每枚档 ≈14.7 MB（canary 实测 14,651,769 B）。
- **换链的科学代价（说不算更正）**：出厂底是**没吃过这条语料的冷基座**，它的压强分布回答的是
  “**出厂件为什么不进化**”（09 §2 N5 那 842 条目零进化的机制那一问），**不**回答“在训链上 τ 该定在哪”。
  §2 的判据形状（在场性＋动态范围）在两档上都适用，但 §1 那句“H-N3b **这条链**”的限定语要跟着换成点名出厂链——
  **换的是分布而不是判据 ⇒ 需要 owner 点，我不自行改**。
- **由此新增一条读数限定（并入 G-N3b-2 那句“不假设中性”，不新建判据）**：本件所有 accuracy／泛化／压强数
  一律定性为“**放行装配＋位置输入关闭**下的数”。它们与步骤一那两张面（位置输入默认开）**不同源**，
  跨面搬 τ 已被 [PLAN-N3-03 §4](PLAN-N3-03_tau_definition_20261008.md) 的五元组禁令挡住，这里是同一条禁令的另一例。
- **烟测只验仪器、不出版结论**（`--smoke` 档：`SeedConfig()` 默认、5000 符号、339,803 参数、随机底，
  `--developmental-fast-slow --developmental-bridge-gate 0.25`，件在盘上但**不进任何判据**）：
  面内自述齐（`bridge_gate_requested=0.25`＝`bridge_gate_actual=0.25`，`learning_mode` 挂载后与收尾**两次都读到 `fast_slow`**）；
  判读器 `--prereg n3-04` rc=0、覆盖率 1.0、`pressure` 与加权和最大偏差 1.1102230246251565e-16；
  **两维都活了**——`fast_slow_conflict` 非零 4,998/4,999（max 0.502029、mean 0.500883）、
  `activity_saturation` 非零 4,999/4,999（max 0.5、mean 0.301206），`pressure` max **0.711851**（mean 0.646321）。
  ⇒ ㊵-484③ 那条算术上限（`pressure = 0.425·r ≤ 0.425`）在这档装配上**不再成立**，
  这是“乙步骤二值不值得花这五支面”的**唯一**先验证据；但它是烟测底、不是 §2 的那张面，故不据它说 τ。
- **同一支烟测里 `decision_should_propose` 全程 0 次**（`should_propose_total=0`、最长连续段 0），
  而 `pressure` 已越过默认 `minimum_pressure=0.70` ⇒ 说明触发**不只由 pressure 决定**：
  `AdaptiveResidualGrowthPolicy` 的六道 `minimum_*` 分项闸里还有别的在拦。本件**不指认是哪一道**
  （面件只自述了 `minimum_pressure`/`required_pressure_steps`/`growth_resource_cost` 三项，其余三项没进件）——
  这条留给步骤二真实面去定位，且定位它需要的是**面内加记分项闸读数**，那是另一件预注册的事。
- **契约测 15 件入库** `tests/taiji_native/test_n3_04_developmental_flags_contract.py`（默认关时三个装配 API
  调用次数**逐键为 0**、开旗标时 bundle/gate/mode 三样都进件、位置输入与 bundle 互斥、`--help` 里两个旗标名都在、
  判读器四条拒绝支各自 rc=2、以及“判据为假而仪器不炸”那一支 rc=0＋`not_present`）。

## 6. 本件不做什么

不冻产品阈、不改 policy 常量、不 promote 任何候选、不动出厂面；不做 ×16.8 大档（那是 N3 甲的另一条泳道）；
不声称"经验→能力通道打通"——**本件起草时**那句话卡在 **DEBT-G47**（修法甲已实施，但第一次通电那枚候选档救不回，巩固后面必须在修法甲之下新跑一次通电上重取）。
**2026-10-08 同日状态更新（写在跑之前，不改任何判据）**：第二次通电（[PLAN-N2-02 判读](PLAN-N2-02_ADJUDICATION_20261008.md)）已把**原生候选档**被 `SeedRuntime.load` 读回这一格实测为绿（摘要逐位同 `6f29e87dbccd8c6e…`、未睡母档反向同绿）⇒ G47 已结清，本件 §6 上面那句"现在仍卡在"随之过期；
**但本件的读数定性不因此改变**：N2 那一轮同时判 J-N2a'／J-N2b' 不成立（通道通、这一剂量档买来的是代价），所以乙步骤二仍然只回答"四维在场的压强分布长什么样、τ 能不能冻"这一件事，不借用 N2 的任何能力声称。

## 7. 不变项

M6 收官、M8 挂起；M7-CI 绿只在推送后实测；**本件的臂与 gate 已由 owner 2026-10-08 第三次弹窗裁死（丙＋乙、gate 0.25 与 1.0 都跑，见 §4bis），待做＝按 §5bis 落两个默认关旗标再跑 5 支面**；N1（H-S1b 训练机时待批）、N2（第二次通电已跑完并判读入库＝通道通、加速点① 在该剂量档不成立；下一件＝保持集与巩固写入范围）、N3 甲（250k 两臂仍不起跑，前置＝§8.7 五项面〔已落成仪器〕＋DEBT-G49 的探针命中率阈值〔owner 定〕）；N4/N5 后置；台账行序以行首标号为准（㊵-419⑥）。
