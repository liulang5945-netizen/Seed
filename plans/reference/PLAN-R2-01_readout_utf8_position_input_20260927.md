# PLAN-R2-01：把 UTF-8 位置状态做成读出的显式输入（架构案，决策就绪提案）

日期：2026-09-27。归属：R2 主线语言能力（M5 退出排除项）。
状态：**已批准并落地**——owner 2026-09-27 弹窗**全批**三件（①架构改动进主干（默认关）
②两臂训练预算（各 2M 符号）③若走分支 2 的一次性追加权）。实现已提交 `c436380a`
（默认关 ⇒ 产品行为逐位不变）；两臂训练在跑，配方见 §8。**判据先于代码冻结**（§3）。
本件是 owner 裁定"2 之后 3"里"3"的落地路径；"2"（解码掩码产品化）已完成，见 `SPEC-R2-02`。

---

## 0. 已量到的、指向这一刀的那件事（全部带出处）

1. **地板判 fail**：无掩码通道整句可解码率全 0（`taiji_f0_language_floor_20260927.json`）。
   M1 成句率 0.87 的真相＝判读仪器带 UTF-8 解码掩码扶手；arm A 无掩码直测 0/32。
2. **训练目标方向已试并否证**（`SPEC-R2-01` 分支 3）：续字节加权 2M 符号，
   训练分布续字节在线准确率 0.3513（对 D0 0.2985），但无掩码 F0 仍全 0。
3. **A3 判决探针（本件前置，`taiji_a3_position_probe_full_20260927.json`）**：沿真实对话语料
   字节流逐步 `observe`，问"`motor_context`（读出赖以预测下一字节的直接输入）能否线性读出
   下一字节的 UTF-8 位置类"：
   * **motor 75.7%／多数类基线 64.4%／DFA-state oracle 96.6%**；
   * 判读走预注册第三支"分层报数不裁定"：位置信息**部分在场**（比瞎猜 +11.3 点），
     但**离 oracle 差 20.9 点**——递归痕迹对"处在字符内第几字节"只有弱的线性可及性。
4. **容量账**（`PLAN-A-24` §2e）：快通路＋motor 只占 18.8% 被训练，读出是承载语言主干的那张面。

## 1. 一句话假设

读出的输入里**没有一个干净的"字符内位置"信号**，它只能从被污染的递归痕迹里弱弱地猜；
把 DFA 位置状态（`remaining ∈ {0,1,2,3}`，4 维 one-hot，由上一字节**确定性地**算出，零学习成本）
做成读出的显式输入，就把它从"要自己悟"变成"直接告它"，oracle 的 96.6% 是这一刀的**特征价值上界**。

## 2. 设计（架构改动，默认关 ⇒ 产品行为逐位不变）

* `BytePredictiveReadout` 输入侧增 4 维（当前字节的位置状态 one-hot），走**已有**的学习规则
  （读出是线性面，加输入维＝加对应权重列，初始化 0 ⇒ 挂载位级不变，`gate` 式零初始化纪律）。
* 由 `TaijiConfig` 一个 frozen 新字段 `readout_utf8_position_input: bool = False` 控制；
  False ⇒ 输入维度与权重形状不变 ⇒ 旧 payload 逐位可载、digest 不外扩。
* `generate`/`observe` 在算读出输入时，用**共享状态机** `taiji/utf8_state.py`（SPEC-R2-02 已落地）
  取当前 `remaining`，喂那 4 维。**训练与生成同一取法**（否则又是"训练学的不是发射用的"）。

## 3. 判据（先于代码冻结；这条是本案与既往翻车的分水岭）

**主判据必须打"自由生成 + 无掩码"的 F0——不许再用任何 teacher-forced 或带掩码数当能力证据。**
（教训：A3 的 96.6% oracle 本身是 teacher-forced 读数，SPEC-R2-01/B1/A2.5 连续四次证明
"训练分布增益不转移到自由生成"。本件若重蹈，就是第五次，判据设计上要先拦住它。）

| # | 判据 | 线 |
|---|---|---|
| **主** | 无掩码 F0：T1a 或 T1b 整句可解码率 | **≥0.5**（与 §5d 地板线同一条，不是新调） |
| 归因 | 同流对照：位置输入开(默认学习) vs 关（W=0 冻结新列） | 两态差 ≥0.10 且同向，否则新维白加 |
| 转移自证 | 报告"训练分布 teacher-forced 续字节在线准确率"与"自由生成无掩码可解码率"两个数，**并排** | 若前者涨、后者不涨 ⇒ 如实记"又一次不转移"，判分支 3 同 SPEC-R2-01 |

## 4. 三分支读法（对齐 SPEC-R2-01）

1. F0 过 0.5 且开/关同向差 ≥0.10 ⇒ 位置输入是语言地板的一个真部件；再谈默认是否开（owner）。
2. F0 涨但未过线（0.1–0.5）⇒ 记"方向为正、单这一维不足以撑地板"，允许一次性叠加下一手
   （成句级目标），需再确认预算。
3. F0 不过线 ⇒ **架构输入侧也不是限制**；旁路＋读出输入两条加部件的路都否证后，
   问题彻底回到 M5 主线的数据规模与目标函数（非本件范围，转 owner 大裁）。

## 5. 实现清单与守卫

| 文件 | 改动 | 守卫 |
|---|---|---|
| `taiji/organs.py` `BytePredictiveReadout` | 加 4 列可选输入（默认关⇒形状不变） | ①关闭时 payload 逐位可载旧档；②新列零初始化⇒开启但未训时逐位等于关闭 |
| `taiji/config.py` | `readout_utf8_position_input: bool = False`（frozen） | 不改现有字段默认 |
| `taiji/model.py` `observe/generate` | 喂位置输入，训练/生成同一取法（调 `utf8_state`） | "被走到"计数：位置输入非零贡献的步数 |
| `train_taiji_langfloor.py` | 加 `--readout-position` 旗标（复用现有臂/预算纪律） | 写面守卫沿用（只动 predictive_readout） |
| `tests/taiji_native/test_readout_utf8_position.py` | 上面三条 | — |

**落地时的两处设计更正（照实记，§8 详述）**：
* §2 原文写"读出是线性面，加输入维＝加对应权重列"——**照字面做会坏**：`SparseSynapses` 是
  **固定扇入**（每行只保留 `fan_in` 条边，建库时按 `in_features` 抽），把 4 维并进
  `in_features` 会重抽每一行已有连接 ⇒ 旧 payload 载不回、`gate` 律失效。
  实际做法＝**独立的 4 列稠密权重**（零初始化、默认不建），前向加 `W_pos @ one_hot`，
  完全不碰 `SparseSynapses` 的形状与随机流。
* §2 写"由上一字节确定性地算出"——位置类需要 DFA 余量（3 字节字的续字节之后，余量仍可能
  是 1 或 2，**不是**上一字节的单值函数）；实际做法＝把余量作为 `TaijiState` 的**可选字段**
  `motor_position_class` 存下来，按 `(上一余量, 本字节)` 经 `utf8_state.remaining_after`
  推进（该函数委托 `advance_utf8`，不另写第二份判定）。

## 6. 预算与签字申请

* **要 owner 批的三件事**：①架构改动进主干（默认关，但毕竟动了 `BytePredictiveReadout` 形状）；
  ②训练预算：两臂（开/关对照）× `SPEC-R2-01` 同款 2M 符号 ≈ 2×1.5h CPU；
  ③若走分支 2 的一次性追加，需再确认。
  ⇒ **2026-09-27 弹窗全批**（三项一次批完）。
* **不批也能做的**（已在做/已做）：A3 探针（零训练，已完成）；实现 + 守卫测试（不动默认，不训）。
* **明确不做**：不开产品默认（`readout_utf8_position_input` 保持 False 直到主判据过线且 owner 批）；
  不碰睡眠巩固／后果语义／G8 挂载（各系其 owner 条）。

## 7. 与"3 是不是该做"的诚实一问

B1、SPEC-R2-01、A2.5 已四次否证"在现有旁路/读出上再加部件"。A3 只是把"下一块最便宜的部件"
（4 维位置输入）指出来，并给了它一个 teacher-forced 特征价值上界。
**合理预期应设低**：它可能第五次不转移。若本件判分支 3，那结论就不再是"再找一块部件"，
而是"这个量级（~10⁶ 参数、逐字节局部学习）的架构，装不下守住 UTF-8 又成句的自由生成"——
那是 M5 主线数据/目标/规模级别的裁决，不是 A 支线能收的尾。本件值得跑，正因为**它是这条岔路的最后一块便宜砖**。

---

## 8. 执行记录（2026-09-27；实现已提交，两臂在跑）

### 8a. 落地刀（提交 `c436380a`，8 文件 +588/−63）

| 面 | 实际改动 |
|---|---|
| `taiji/utf8_state.py` | 新增 `UTF8_POSITION_DIM = 4` 与 `remaining_after(remaining, symbol)`（**委托** `advance_utf8`，不另写判定） |
| `taiji/config.py` | 新 frozen 字段 `readout_utf8_position_input: bool = False`；开启时 `planned_active_parameter_count` 计入 `alphabet_size × 4` |
| `taiji/organs.py` | `BytePredictiveReadout` 新增独立 4 列位置权重 + `position_input_enabled/position_probability_steps/position_learn_steps` + `adopt_position_input`；`probabilities/learn` 收 `position_state`；payload 仅在开启时写 `position_weight`；`ResponsePlanReadout.ablated_probabilities` 同步带上 |
| `taiji/state.py` | 新增可选 `motor_position_class`（`None` 不进 payload，仿 `predictive_context_slow_trace` 先例） |
| `taiji/model.py` | `observe` 计算并喂位置类（前向用本步、后向用上一步）；`response_plan/start/phase` 四条重算路径带上；发育 F1 学习链未接该列 ⇒ 组合开启时**响亮拒绝**；参数账三处计入位置列；读出器 fork 时搬位置列 |
| `taiji/language_alignment.py` | response-plan 分支改走模型包装 `response_plan_probabilities()`（不再直接调 readout，否则会漏位置列） |
| `scripts/training/train_taiji_langfloor.py` | `--readout-position` 旗标；信封**三处** config 副本同步改写；三位"被走到"守卫（前向步/后向步/权重范数）；续训旗标一致性守卫 |
| `tests/taiji_native/test_readout_utf8_position.py` | 11 条守卫（新建） |

### 8b. 守卫读数（照 §5 三条 + 端到端）

* **守卫①（关闭＝逐位不变）**：默认 config 下 payload 不含位置键、载回张量逐位相同；
  `motor_position_class` 不进状态 payload。**守卫②（零初始化惰性）**：开启后位置列全零、
  `SparseSynapses` 初值逐位等于关闭，`learn=False` 推理逐步逐位相同。
* **守卫③（被走到）**：`--readout-position` 冒烟 5000 符号 ⇒ `probability=5000 / learn=4999
  / weight_norm=13.34`；不传旗标 ⇒ 三步全 0。续训（再跑 7000 符号）⇒
  `probability=7000 / learn=6999 / weight_norm=15.73`；旗标不一致 ⇒ 当场拒绝。
* **定向回归 58 绿**（utf8 掩码 / response-plan / receptor 分解 / region0 掩码 / 训练档 / golden 路由）。
* **全量 `tests/taiji_native` 1577 过 / 2 红**：两条红（`test_cap0_inventory_contract`、
  `test_cap0_legacy_load_contract` 的"当场重采 reproduces sealed"支）**经撤掉本改动后复跑同样红** ⇒
  既存红，与本件无关。根因：①`checkpoints/` 比封存样本多 `seed_native.pt` / `resumed_seed_native.pt`
  （2026-09-27 G4 判据③真机回合产出）；②生效日 09-20 的产品默认换底使 legacy-load 封存基线
  （09-18 那份）过期（同一现象在 inventory 支已按"重基到 09-20 样本"处理过，legacy 支未重基）。

### 8c. 两臂配方（owner 2026-09-27 定：W=4.0）

* **唯一变量＝`--readout-position`**；其余两边逐字相同：`--arm D --weight 4.0`（`SPEC-R2-01` 的
  处理配方：续字节位置读出更新 ×4）、语料 `dialogue_extended_clean.jsonl`、基底 `seed_beta.pt`、
  各 2M 符号、`--fresh`。
* 臂目录：`output/taiji_r2_plan01_pos_on`（位置输入**开**，回读）与
  `output/taiji_r2_plan01_pos_off`（**关**，同流对照）。两者 `--arm` 同为 `D`，
  靠 `readout_position` 元数据区分（续训守卫会拒绝不一致的旗标）。
* 两臂都带 W=4.0 ⇒ 归因判据仍干净（同流同配方，只差位置输入）；代价是结论只能写成
  "加权＋位置输入这一组"，**位置单独的价值**若要单列须另跑 W=1.0 的两臂（不在本次预算内）。
* 判读一律事后用 F0 探针（**主判据＝无掩码 F0 整句可解码率 ≥0.5**，§3），本件不做评价。
