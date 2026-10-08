# PLAN-N3-03 · R4 生长阈 τ 的定义与"四维修正活着"的装配代价（2026-10-08，零算力；owner 裁"先定阈值定义，再跑改装配那一档"的落地件）

> **来源与效力**：owner 2026-10-08 弹窗裁 N3 乙＝"**先定阈值定义，再跑改装配那一档**"（09 §3.2 第 3 条、㊵-485①）。
> 本件**不是**预注册（不冻判据、不开跑），它是 [PLAN-N3-01](PLAN-N3-01_r4_hooks_prereg_20261007.md) 的前置定义件：
> 把"τ 是什么的阈值、能在哪张面上取、换装配能不能搬"写死，并给出装配三选一让 owner 有的可裁。
> 触发本件的实测＝㊵-484（两条链上 `pressure ≤ 0.425 < minimum_pressure 0.70`，4000 步 `should_propose` 零次）。

## 1. τ 是什么的阈值（定义，逐字）

`AdaptiveResidualGrowthTrigger.observe(pressure, *, structural_budget)` 放行一次生长提议，需要**同时**满足：

1. `pressure ≥ τ`，其中 `pressure` ＝ 五信号的**加权和**（`AdaptiveResidualGrowthPressure`，权固定在产品常量里），五信号＝`residual_error`／`fast_slow_conflict`／`activity_saturation`／`utility_gap`／`resource_state`；
2. 连续 `required_pressure_steps` 步都满足 1（默认 3）；
3. 五个分项闸 `minimum_residual_error`／`minimum_fast_slow_conflict`／`minimum_activity_saturation`／`minimum_utility_gap`／`minimum_resource_state` 各自不被击穿（默认 policy 里 `minimum_pressure=0.70`，其余五项另有默认值，见 `AdaptiveResidualGrowthPolicy`）。

⇒ **τ 只能是"这张面上、这个装配下、这条加权和"的阈值**。三个限定少写任何一个，冻出来的数就不是同一个量：
换链（不同 checkpoint）τ 不同、换装配（哪几维在算）τ 不同、换权重（加权式）τ 不同。
这也正是 ㊵-484 判"阈值不冻"的依据：09 §2 N3 现行口径拿 `accuracy` 推"残差阈 ≥0.4"，而 trigger 的
`residual_error = 1 − prior_probability` 是**逐符号预测概率**（[taiji/model.py:1081](../../taiji/model.py)），
两条实测口径差 0.172／0.281，超 0.10 的否证线 ⇒ **拿 accuracy 推出来的数不能当 τ**。

## 2. 今天这条训练链上，五维里有两维结构性恒零（㊵-484 已测，本件补机理到位）

| 信号 | 今天是否有值 | 为什么 | 锚点 |
|---|---|---|---|
| `residual_error` | 有（p90≈0.994–1.0） | 逐符号 `1 − prior_probability`，读出分支每步都算 | [taiji/model.py:1081](../../taiji/model.py) |
| `utility_gap` | 有但**退化**（＝0.5·residual） | 定义里含快慢冲突与活动饱和两项，两项为 0 时它塌成残差的倍数 ⇒ `pressure = 0.425·r` | ㊵-484③ 实测与 max 0.425 对上 |
| `resource_state` | 有 | 预算侧自述 | `AdaptiveResidualGrowthPressure` |
| **`fast_slow_conflict`** | **整场恒 0** | 只有 `self._developmental_f1_bundle` 在场才算；且比值取的是 `fast_delta.norm()/slow_weight.norm()`，**挂载刚完成时 `fast` 就是零** ⇒ 挂载≠有值 | [taiji/model.py:1058-1070](../../taiji/model.py)、:1271-1300（`migrate_f1_to_developmental_synapses` 落 `read_only` 且 `fast_is_zero`） |
| **`activity_saturation`** | **整场恒 0** | ＝"单元活动 ≥ 目标活动点的比例"（`adaptive_residual_bridge.py:101-107`）；`gate=0.0` 的 bridge 从不参与读出 ⇒ 无活动可饱和 | ㊵-484③ |

⇒ 今天冻 τ 等于给一个**二维塌成的加权和**冻阈值，且上界 0.425 **算术上够不到默认 0.70**。这不是"接线没接"，是**装配没让那两维活着**——09 §2 N5 记的"出厂件 842 条目零进化"至此有了机制解释。

## 3. 让两维活着的三条装配（三选一，都要改行为 ⇒ 回 owner）

- **甲｜就在退化二维上冻一个低 τ**（把 `minimum_pressure` 从 0.70 降到实测可达区）。
  代价＝**不答本件要答的问题**：τ 是照着退化复合量定的，一旦那两维活起来，同一个数立刻失效；且 `minimum_pressure` 是产品常量 ⇒ 改它＝改产品默认放行面。**本件不推荐，列出来是为了让"为什么不选它"有出处。**
- **乙｜`bridge gate > 0`（放行读出）**——`enable_adaptive_residual_bridge(gate=…)`／`set_adaptive_residual_bridge_gate`（[taiji/model.py:831,860](../../taiji/model.py)）。
  只解 `activity_saturation` 一维；**它本身就是行为改变**（bridge 参与读出＝模型输出变了），所以这一档跑出来的任何 accuracy/泛化数都只能定性为"放行装配下的数"，不得与今日出厂面同表比较（㊵-483 G-N3-1 的收紧正是为这条）。
- **丙｜developmental F1 的学习/重放模式**——`migrate_f1_to_developmental_synapses()` 挂载后，还必须 `set_developmental_f1_learning_mode("fast_slow" 等)` 让 `fast_delta` 真的被写，才有 `fast_slow_conflict > 0`（模式集合与语义见 [taiji/model.py:1326-1347](../../taiji/model.py)；今天只有语言对齐器官的 epoch 循环在驱动它，`language_alignment.py:668-696`，**语料训练器 `train_seed_corpus.py` 里没有任何旗标能开它**——按词 grep 命中 0）。
  ⇒ 光挂载不够（`fast_is_zero` ⇒ 比值 0），这是本轮新读出来的一条：**"两维恒零"不能靠调用一次 migrate 解决**。

**还有一条必须一起进决策的硬事实**：`set_developmental_f1_learning_mode` 的模式**故意不作为权威持久化**——件内注释写明"a fresh restore always returns to `read_only`"（[taiji/model.py:1331-1332](../../taiji/model.py)，恢复路径 :3785 也复位）。
⇒ **四维修正活着的装配无法从任何 checkpoint 里恢复，只能在每次 load 之后重新施加**。任何"长跑中途崩溃恢复后仍带装配"的隐含假设都不成立（与既有那条"治疗臂要在每次 load 之后重新挂载"同族）。

## 4. τ 的冻结规则（定义件冻这条，不冻数）

1. **面**：改装配后的**训练流符号链**（`--readout predictive`、symbol 流分支；answer/self-answer 档走 `learn_bytes` 一个压强读数都不会有，㊵-483② 已钉）。
2. **取法**：在该面上现取 `pressure` 分布，**τ ＝ ceil(p90 到 0.05 档)**（沿用 N3-01 §1 已冻规则；p90 而非 max，是为了让"连续 `required_pressure_steps` 步都过"在有限预算内有解而非永不可能）。
3. **口径否证支先跑**：`|residual_error p90 − (1 − online_accuracy)| > 0.10` ⇒ 判"这套信号与 accuracy 口径不同源"，**本轮两条链都触发了这一支**（㊵-484），故 τ 不得由 accuracy 侧任何数推得。
4. **搬运禁令**：τ 绑（设备, 链路, checkpoint, **装配**, 生成预算）五元组；**换装配不许搬 τ**（既有"结论依链路也依分布"那条纪律）。乙／丙两档各自要各冻一份，且都要点名"这 τ 是在哪几维活着时取的"。
5. **动态范围先测**：冻 τ 之前先报"这一档里 `should_propose` 能否为真"——两维恒零的退化档上界 0.425 < 0.70 就是**这把尺没有动态范围**的现例；无动态范围的档一律不得拿来定 τ。

## 5. 本件不做什么

不立项代码改动、不选装配、不冻任何数值、不开跑。乙步骤二（改装配那一档）在 owner 于 §3 三选一**并确认 §3 末那条"每次 load 后重施加"的运维后果**之后另立预注册；预注册的判据面必须写 §4 的五元组，且把"四维修正活着"做成**面内自述读数**（每维在场与否当场可查），不许由命令行反推。

## 6. 不变项

M6 收官、M8 挂起；M7-CI 绿只在推送后实测；N3 甲的预算上限待 owner（标定跑需在无负载机器上做，当前本机有面在跑 ⇒ 未标定）；N1/N2/N4/N5/N6 排序不变；台账行序以行首标号为准（㊵-419⑥）。
