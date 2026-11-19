# 设计卡：22.5.1 / S0 后续 D 面定位包（路由操作点单变量；已设计＋已定价＋已授权执行＋已收官）

> 文档状态：[设计卡→已执行] — 2026-11-20。对象＝D-20261119-09 §7 决策点① 的推荐选项（`<=10%` 窗口预算的局部定位包）。**本卡已按 owner 弹窗授权执行并收官：`operating_point_insufficient`（负面），实耗 88.19 CPU ≤ 帽 154.7，0 check/final。** 结果卡见 [RESULT-TAIJI_U2_S0_D_LOCALIZATION_20261119.md](RESULT-TAIJI_U2_S0_D_LOCALIZATION_20261119.md)，判决与后继见 D-20261119-11。定价回执见 [`taiji_u2_s0_decision_options_price_20261119.json`](../../reports/taiji_u2_s0_decision_options_price_20261119.json)，事实底座见 [`taiji_u2_s0_accepted_parent_load_20261119.json`](../../reports/taiji_u2_s0_accepted_parent_load_20261119.json)，合同 [`taiji_u2_s0_d_localization_20261119.json`](../manifests/taiji_u2_s0_d_localization_20261119.json)（`result.closed=true`）。

## 1. 立项依据（为什么需要定位，而不是再加一次修订）

S0 三 seed 复核（[结果卡](RESULT-TAIJI_U2_S0_ARCH_REPAIR_20261119.md)）判 `s0_diagnostic_failed`：seed `20261110` 父代 A **32/32** 仍 D `15/24`（已见）/`8/24`（未见），seed `20261111` 全过，seed `20261112` 未见模板 B/多记录 **`1/24`**。失败不是「父代太弱」一种解释能覆盖，至少有两种互斥归因：

- **H1 操作点假设**：匹配滤波的表示/训练已经可分，是 `route_gate` 的判定阈值（`match_prob` 操作点）在不同 seed 落到不同位置，导致 B 与 D 不能同时成立；
- **H2 表示/训练限制假设**：训练后的器官在无关记录上的 `match_prob` 与相关记录不可分，阈值再调也无法同时满足 B 与 D。

**本包只做定位，不改架构、不加参数、不改损失、不改判据。** 结果只决定「下一步该开标定卡（H1）还是按纪律收束 D 面（H2）」，本身不签任何能力。

## 2. 输入、身份与数据边界

| 项 | 冻结内容 |
|---|---|
| 父 | 已接受 v1 A 父 `pixel-salience`、`d=32`：seed `20261110`（`output/a_pixel_parallel_20261119T063705Z/segment.bin`，记录 fit `30/32`）与 seed `20261112`（`...T070447Z`，记录 fit `31/32`）。经**只读历史加载器**：schema/architecture/readout 断言＋strict state load＋记录 fit 逐值复现（3/3 通过）；唯一源码 pin 失配 `taiji/u2_subject.py`（S0 变更，已披露），不训练父、不写回父。 |
| seed `20261111` | **不纳入 D 门**：其接受父 fit 仅 `14/32`，D 门 \(\ge20/24\) 结构上不可达（D 命中 ≤ 父代该子集正确数）；可作诊断行报告，禁止与其他 seed 拼凑。 |
| 训练数据 | `first_a_support_22_5_v2.generate_examples("train", 768)`，器官训练模板沿用 S0 已见 2 模板。 |
| 评测数据 | `generate_examples("fit", 32)`（24 个非未知 B 项；已见 2 模板／未见第 3 模板两面）。 |
| 分区 | 仅 `train/fit`；**0 check、0 final 读取**；不新立曝光账。 |
| 身份 | 器官＝S0 matched-filter `RecordOrgan`（`organ_mix` 出生 0，训练后激活）；父＝v1 冻结。 |

## 3. 唯一变量与允许/禁止动作

- **唯一变量**：推理期路由操作点偏移 \(\delta\)，以闭包补丁实现（**模块字节零改动**）：对 `organ.plan(request, records)` 的原返回值重算
  \[
  \text{route\_prob}(\delta)=\sigma\big(12(\text{match\_prob}-0.5)+\delta\big)\cdot\text{active\_strength}.
  \]
  \(\delta\) 不参与反向、不写回、不进 state_dict；扫描 \(\delta\in\{-4,-3,-2,-1,0,1,2,3,4\}\)。
- **允许**：每 seed 冻结父上的器官训练 300 步（与 S0 诊断同配方：`BOrganTrainConfig(seed, steps=300, batch_size=4)`，`record_organ` 参数唯一可训）；`plan()` 读数采集；对最优 \(\delta^*\) 做一次全 decode 复核（已含在定价内）。
- **禁止**：改架构/新参数/改损失或采样/改父/改数据/改判据；对同一 \(\delta\) 多次扫描后挑最优写回；训练后追加步数；读 check/final；以本包结果直接签能力或联合门。

## 4. 预注册读数、判据与出口（读数前冻结，本卡即冻结件）

**决策级读数（不 decode，按项）**：
- B 项（相关记录，1 条）：`B_hit(δ) = 1[route_prob(δ)≥0.5 ∧ argmax 记录窗口文字 == 目标标签]`；
- D 项（无关记录，1 条）：`D_hit(δ) = 1[route_prob(δ)<0.5]`；
- 多记录（1 相关＋2 干扰，相关居任意位置）：同 B 的窗口判定；
- 两面分别统计（已见／未见），分母 24／24。

**判据（全预注册，读前冻结）**：
1. `operating_point_separable`：存在**同一** \(\delta^*\) 使两个 seed、两个面同时满足 `B_hit ≥ 23/24` 且 `D_hit ≥ 21/24`（决策级）；
2. 若满足 1，再以 \(\delta^*\) 做一次全 decode 复核：`B=24/24`、`D≥20/24`（两面）→ 判 **`operating_point_only`**（证据：失败可由单一阈值解释，后继应开**预注册阈值标定卡**，不是新架构修订）；
3. 若不存在：判 **`operating_point_insufficient`**（H2 表示/训练限制成立）→ D 侧按范围上限收束，联合门「无关记录」判据转入 D-20261119-09 选项③的相对父代修正案（另请 owner 裁）；
4. `engineering_failure`（加载/断言/恢复或非有限值）与 `budget_stop`（超帽）按统一出口。
5. 额外报告（不设门）：`match_prob` 在 B/D 两类上的分布重叠度、seed `20261111` 诊断行、\(\delta\) 曲线。

## 5. 实测定价与硬帽

| 项 | CPU | 来源 |
|---|---|---|
| 器官训练（300 步，冻结父，父 logp 缓存） | 14.78／seed | S0 诊断实测（取三 seed 最大） |
| 全 decode 复核套件（两面） | 37.09／seed | S0 诊断实测（取最大） |
| 决策级读数（`plan` 调用） | 0.001563／次 × 6 组／seed | 本卡实测（丢弃模型） |
| 工程与加载/序列化余量 | 20.0 | 定价脚本 |
| **合计（2 seed）** | **123.76** | 回执 `options.1_d_localization.price_cpu` |
| **硬帽（×1.25）** | **154.7 CPU／20 min wall** | 回执 `cap_cpu_proposed` |

优化器步上限 **600**（2 seed × 300）；check/final 读取 **0**。

## 6. 授权状态与唯一后继

- **授权与执行（2026-11-20 更新）**：owner 弹窗授权按帽执行（`154.7 CPU／20 min`），合同 `training_authority=owner:chat:2026-11-20#popup-d-localization-authorized-cap-154.7`；smoke 12 项全绿（0.38 CPU、0 优化器步）后执行，实耗 **88.19 CPU／88.88s**，判 **`operating_point_insufficient`**（δ 九点平台；seed 10 未见模板 B/多记录 `5/24`，D 决策 `22/24` 两 seed 两面一致）。结果与归因见 [结果卡](RESULT-TAIJI_U2_S0_D_LOCALIZATION_20261119.md)。
- **唯一后继（按预注册与 owner 裁定）**：D 面按范围上限收束（不再开第 4 次架构修订）；联合门「无关记录」判据转**相对父代**修正案；主候选固定为已接受 v1 父 seed `20261112`；下一步＝owner 对 S1（`R-REBUILD-1`，约 `1,400 CPU`）授权或豁免的裁定，S2 零训练设计＋定价随后。
- **边界**：本包不签联合门、A 父、任何能力、22.6/22.7；不改 S0 已出读数；不消耗 check②。
