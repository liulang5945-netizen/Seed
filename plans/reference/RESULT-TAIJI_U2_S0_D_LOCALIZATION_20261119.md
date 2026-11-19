# 结果卡：22.5.1/S0 D 面定位包（`operating_point_insufficient`，负面，已授权执行）

> 文档状态：[结果卡] — 2026-11-20。绑定合同：[`taiji_u2_s0_d_localization_20261119.json`](../manifests/taiji_u2_s0_d_localization_20261119.json)（`status=frozen`、`training_authority=owner:chat:2026-11-20#popup-d-localization-authorized-cap-154.7`、`result.closed=true`）。机器回执：[smoke](../../reports/taiji_u2_s0_d_localization_smoke_20261119.json) / [run](../../reports/taiji_u2_s0_d_localization_run_20261119.json)。本卡不签任何能力/父资格，不读 check/final。

## 1. 范围与授权

- **对象**：D-20261119-09 §7 决策点①（D 面定位），唯一变量＝推理期路由 gate 偏移 δ（闭包补丁、模块字节零改动、不训练不写回）。
- **授权**：owner 2026-11-20 弹窗裁定「授权按帽执行定位包（推荐）」；合同价 `123.76 CPU`、硬帽 `154.7 CPU／20 min`、优化器步上限 `600`（2 seed × 300）、`0 check`。
- **输入**：已接受 v1 A 父 seed `20261110`（记录 fit `30/32`）与 `20261112`（`31/32`），只读历史加载器（strict load＋pin 失配 1 文件已披露）；seed `20261111` 父 fit `14/32` 结构上不可达 D 门，按预注册排除。数据仅 `train/fit`。

## 2. 实现与 smoke（零优化器步）

- **实现**：[runner](../../scripts/training/run_s0_d_localization_20261119.py) 五模式（freeze/authorize/smoke/run/closeout）；`--run` 在缺 `training_authority` 时 fail-closed；δ 由闭包 `plan` 包装实现（`route_prob(δ)=σ(12(match_prob−0.5)+δ)·active_strength`），不写回、不进 state_dict。
- **smoke 12 项全绿**（0.38 CPU、0 优化器步）：父身份/严格加载/pin 披露、出生逐位继承、出生 decode 与父一致、δ 公式与单调性、plan/emit 形状、补丁生效。回执见 [smoke](../../reports/taiji_u2_s0_d_localization_smoke_20261119.json)。

## 3. 执行读数（实耗 `88.19 CPU / 88.88s wall`，帽内）

器官训练 300 步/seed（父 logp 缓存）：`parent_bit_identical=true`，`organ_mix` `1.2572`（seed 10）／`1.3041`（seed 12）。

决策级读数（B/D/多记录，分母 24；δ 扫描 `{-4…+4}`）：

| seed | 面 | B | D | 多记录 |
|---|---|---|---|---|
| 20261110 | 已见 | 24/24 | 22/24 | 21/24 |
| 20261110 | **未见** | **5/24** | 22/24 | **5/24** |
| 20261112 | 已见 | 24/24 | 22/24 | 21/24 |
| 20261112 | 未见 | 24/24 | 22/24 | 21/24 |

**9 个 δ 取值下所有读数逐项完全相同**（δ 曲线为平台）：没有 δ* 满足预注册门（两 seed × 两面 `B_hit≥23` 且 `D_hit≥21`）⇒ 预注册出口 **`operating_point_insufficient`**。

## 4. 判读（H2 成立；阈值不是主因）

1. **δ 平台＝路由 logit 饱和**：±4 的 logit 偏移不改变任何决策，说明相关 route logit 距 0.5 阈值 ≥4（`|match_prob−0.5| > 1/3`）。在预注册 δ 族内，**D 面失败不能由单一阈值解释**；预注册的 H1（操作点假设）被证伪。
2. **失败点是未见模板上的窗口选择，不是 D 门**：两 seed 的 D 决策均 `22/24 ≥ 21`（无关记录正确退避），而 seed 10 未见模板 B/多记录仅 `5/24`——器官在已见两模板上训练后，对倒装未见模板的路由/窗口选择不泛化。这是表示/训练限制（H2），不是 gate 阈值。
3. **与 S0 三 seed 负面读数的关系**：S0 负面的 decode 级 D 低（seed 10 `15/24`、`8/24`）在决策级并不对应 D 误触发；说明 decode 级 D 失败另有来源（route 关闭时的混合/记录通道扰动、或父代 A 差异），本包按预注册范围不追此变量，只作为后继假设登记。
4. **边界**：δ 平台只否定「预注册 ±4 阈值族」；不外推为「任何阈值都不可能」。本卡不签机制结论，只按预注册出口收束 D 面。

## 5. 标准出口、签署与后继

- **出口**：`operating_point_insufficient`（预注册负面）。按 D-20261119-09 预承诺：**D 面按范围上限收束，不再开第 4 次同范围修订**；联合门「无关记录」判据转 owner 已批准的**相对父代**修正案。
- **签**：定位包实现与 smoke（12 项）、执行读数、δ 平台与 H1 证伪、预注册出口。**不签**：联合门、A 父、任何能力、22.6/22.7、decode 级 D 归因。
- **成本**：执行 `88.19 ≤ 154.7 CPU`（帽内，余量 `66.51`）；smoke `0.38`；`0 check/final`。
- **唯一后继**：按 D-20261119-11 的 owner 裁定推进——主候选固定为已接受 v1 父 seed `20261112`（记录 check `174/192`、fit `31/32`，按「记录 check 最高、fit 次优打破平局」预声明规则选出），联合门 D 判据＝相对父代单侧 95% 上界 `≤0.02`；下一步＝owner 对 S1（`R-REBUILD-1`，约 `1,400 CPU`）授权或豁免的裁定，以及 S2 零训练设计＋实测定价（无需授权）。
