# 生长机制设计规格（2026-11-08）：何时该长、长什么、长多少、何时放弃

> 状态 [设计规格·未实现进训练图]。0 训练 / 0 优化器步 / 0 Adam / 0 confirm。
> 代码事实基线：`growth_gates` 只是**已生长**主体的分支贡献门（[growth_credit.py](../../taiji/growth_credit.py#L42-L55)），真生长算子是 `birth()`（[growth_credit.py](../../taiji/growth_credit.py#L459-L525)），只被测试与离线验证脚本调用（`tests/taiji_native/*`、`scripts/training/verify_taiji_birth_*`、`scripts/training/diagnose_*`），当前训练窗只**读**门（[runner](../../scripts/training/run_taiji_training_window_v2_20261108.py#L212-L213)）。本文补的是缺失的那一层：**触发判据与分配规则**。
> 新模块 [taiji/growth_design.py](../../taiji/growth_design.py) 不被 runner 及其 import 闭包引用（见 [禁改清单](FROZEN-READSET_20261108.md)），纯 numpy，无 torch。

---

## 0. 一句话

生长不是一个"开关"，而是**一次带证书的结构变更**：只有当「容量证书」证明瓶颈是容量、且「键谱证书」证明键方向本身可分开时，才允许长；否则长出来的只是**更大的同构不可分空间**。

---

## 1. 观测层：三个可计算信号

三个信号全部可在**只读**路径上算出，不需要反传、不需要 Adam。

### S-A 键谱（键空间是否塌缩、塌缩属于哪一类）

输入：已占用槽的读键 `store_keys[valid]`（形状 `(n, d)`，`d = config.width = 128`；见 [living_core.py](../../taiji/living_core.py#L518-L526)）。先逐行单位化，记 Gram `G = K̂K̂ᵀ`（对角置 0 记为 `G_off`）：

| 量 | 定义 | 读法 |
|---|---|---|
| `mean_abs_cos` | `mean(|G_off|)` | 键平均有多"平行" |
| `top_share` | `λ_max(G) / Σλ` | 共享偏移强度 |
| `eff_rank` | `(Σλ)² / Σλ²` | 有效方向数 |
| `top_share_c` / `eff_rank_c` | 对**去均值**后的键重算 | 去掉共享偏移后的真实可分性 |

### S-B 读余量（打分是否已经饱和）

输入：`addressed_read` 已经记录的 `top_cosine` / `second_cosine` / `margin`（[living_core.py](../../taiji/living_core.py#L568-L578)）。

- `p95_top_cos`：所有查询的 top-1 余弦 95 分位；
- `median_margin`：所有查询 `cos₁ − cos₂` 的中位数；
- `norm_entropy`：`T = ADDRESSED_TEMPERATURE = 0.1` 下 softmax 权重的归一化熵（`−Σp log p / log n`）。

塌缩判据：`p95_top_cos ≥ 0.95` 且 `median_margin ≤ 0.02`（两者都是已记录量，可直接复算）；`norm_entropy` 只作辅助诊断，不参与判定。20.3a 实测 `top_cosine = 0.99766`、`margin = 0.00835`（[报告](../../reports/taiji_package20_3a_addressed_store_20261105.json#L869-L871)）⇒ **两个条件同时满足**。

### S-C 容量证书（容量不够 vs 结构分不开）

对每个 FP1 域（对象域、记录域），由写入事件构造 `K ∈ R^{d×m}`、`V ∈ R^{r×m}`，取球半径 `C_F`（[U1 §5](SPEC-TAIJI-UNIFIED_CORE_U1_20261107.md)）。定义（推导见 [勘误 E1 §1.4](U1-ERRATA_20261108.md)）：

- `ρ = ‖V K⁺‖_F`：精确关联所需的最小范数；
- `δ_cap = max(0, ρ − C_F)`：**系统相容，只是球太小**；
- `δ_comp = ‖V (I − K⁺K)‖_F`：**目标本身不相容**，任何可行解都留下 ≥ δ_comp 的残差；
- `σ₊(K)`：K 的最小非零奇异值；读出误差下界 `≥ σ₊(K)·δ_cap`。

这两条**互不替代**：`δ_comp > 0` 时把 `C_F` 加到无穷也只能把残差压到 `δ_comp`，压不到 0。

---

## 2. 决策树（两个机制分开，禁止混判）

```
键空间（20.3a 内容寻址 store）:
  占用 < 2 个槽                          -> no_growth
  S-B 塌缩 且 S-A 为共享偏移
    (top_share ≥ 0.90 且 top_share_c ≤ 0.50) -> representation_repair   # 扩槽/扩维都禁止
  否则若 eff_rank ≥ 0.90·key_width        -> key_dimension_growth        # 投影是瓶颈
  否则                                    -> no_growth

FP1 写入域（对象域 / 记录域）:
  δ_comp > ε_struct (ε_struct = 1e-6·‖V‖_F) -> structural_repair        # 生长禁止
  否则若 δ_cap > 0                          -> capacity_growth           # 球/秩
  否则                                      -> no_growth
```

三条硬规则：

1. **只有 `capacity_growth` 允许增加容量**（槽位、键维、通道、球半径）。
2. `representation_repair` 的方向是**改键的构造**（去均值/白化/停用共享模板），不是加容量——加容量在这里必然只是把同一个不可分锥体放大。
3. 两个机制的判据**不得互相代言**：store 上没有 `C_F`，FP1 上没有余弦打分。

阈值形式冻结、数值待校准：上表的 `0.90 / 0.50 / 0.95 / 0.02 / 0.35 / 1e-6` 是**先验名义值**，在出现第一份校准报告前，决策函数对任何未校准域一律返回 `no_growth`（默认不长）。

---

## 3. 生长对象与代价

| 对象 | 代码位置 | 参数量变化 | 状态字节变化 | 身份摘要 `candidate_sha_v2` | 预期收益（仅当判据命中） |
|---|---|---|---|---|---|
| 球半径 `C_F` | FP1 解算器 | 0 | 0 | **不变**（参数表不变） | 直接消掉 `δ_cap`，收益可证 |
| 槽位 `ADDRESSED_SLOTS` 32→64 | [living_core.py](../../taiji/living_core.py#L518-L526) | 0 | 每槽 `128 + 4·2·32·32 = 8,320` 个 float（33,280 B）；翻倍 ≈ **+1.02 MiB/状态** | 不变（store 是状态不是参数） | 仅当 `δ_cap>0` 且键可分 |
| 键维 `key_width` 32→64 | [living_core.py](../../taiji/living_core.py#L30) | tied-read 输出与内容投影变宽 | fast 状态翻倍（`4·2·32·32`→`4·2·64·32`，+32 KB） | **改变**（形状进摘要） | 仅当 `eff_rank` 顶到 `key_width` 上限 |
| 通道 `fast_heads`/`groups` | `LivingCoreConfig` | 线性层按头/组倍增 | 线性增长 | **改变** | 仅当证书指向通道而非球 |
| 新增子模块（`birth`） | [growth_credit.py](../../taiji/growth_credit.py#L459-L525) | 每单元每分支 `2d = 256` float（R 行 + P 列）；`R_MAX=8` ⇒ 每分支 ≤ 8 单元 ≈ 8 KB | `growth_parent_units` + 分支门表 | **改变**（新参数行） | 仅当方向由信用排序给出且出生中性 |

身份口径来自 [rule_learning_20_1_identity.py](../../taiji/rule_learning_20_1_identity.py#L92-L115)：摘要覆盖**契约 + 参数名/形状/dtype**，不覆盖源码字节；`source_files()` 另行单列三个冻结模块的 sha。

---

## 4. 分配、继承与确定性

**分配（不平均撒）**：沿用 R023 的整数 max-min 语义（[native_development.py](../../taiji/native_development.py#L118-L126)）：对每个候选域给出"多给一单元带来的边际增益"曲线 `ρ_c(·)`，在 `Σ ≤ R_MAX` 与各域上限内选择使 `min_c ρ_c` 最大、再使总和最大、再使单元最少、最后字典序最小的组合（[native_development.py](../../taiji/native_development.py#L121-L123)）。没有边际曲线的域，分配为 0。

**继承**：
- 新增 R 行**零初始化**、新增 P 列**零初始化** ⇒ 出生步新单元贡献精确为 0（`birth()` 已如此，[growth_credit.py](../../taiji/growth_credit.py#L496-L505)）；
- 父行/父列**逐位拷贝**，不重新初始化；
- 新槽位以 `age = −1`（无效）出生，只由真实 turn 写入转正。

**确定性**：
- 决策函数是**纯函数**（无随机、无时间、无全局状态）；
- 唯一的随机来源是方向抽样，必须显式传 seed 并写进计划；
- 不生长时返回 `no_growth`，**不触碰任何张量**——因此现有 19.x/20.x 身份与逐位复现不受影响。

---

## 5. 失败出口

生长后必须在**同一 fit 窗、同一证书**上复算，并按下列判定退出：

| 复算结果 | 判定 | 动作 |
|---|---|---|
| `δ_cap → 0` 且读出误差降到 `≤ ε_gain` | 容量假设成立 | 记录收益，生长保留 |
| `δ_cap → 0` 但读出误差**没降** | **不是容量问题** | 回退（`lesion`），改结构 |
| `δ_comp > 0` 且几乎不变 | **结构性不可分** | 回退，禁止再加容量 |
| 连续 2 次生长收益 `< ε_gain` | 该域停止生长 | 写死 `growth_retired[domain]=true`，不再触发 |

"改结构"的具体含义：改键的构造（去均值/白化/换写入时机）、改读出投影、或改任务分解——**不是**继续加槽位/维度/通道。

---

## 6. 不变量（生长必须满足）

| 编号 | 条款 | 现有代码依据 |
|---|---|---|
| I-G1 出生中性 | 出生步新单元输出精确为 0，父主体前向逐位不变 | R/P 新行列零初始化（[growth_credit.py](../../taiji/growth_credit.py#L495-L505)） |
| I-G2 父行不变 | 父行/父列逐位拷贝，不重初始化 | `load_state_dict` + 拼接（[native_development.py](../../taiji/native_development.py#L53-L66)） |
| I-G3 身份可复现 | 生长块（parent sha、directions sha、seed、birth step、contract_version）进入摘要；**未生长路径摘要不变** | `graph_identity` 现覆盖参数形状（[identity](../../taiji/rule_learning_20_1_identity.py#L104-L106)） |
| I-G4 会话内冻结 | 只允许在**窗口之间**生长；会话内参数冻结不被破坏 | 20.1 契约 `session_freeze`（[identity](../../taiji/rule_learning_20_1_identity.py#L39)） |
| I-G5 恢复一致 | checkpoint 必须携带 R/P 形状、`growth_parent_units`、分支门表；`lesion` 精确复原父主体 | [growth_credit.py](../../taiji/growth_credit.py#L528-L545) |
| I-G6 默认零开销 | 不调用触发即与今日逐位一致；门在 `growth_parent_units is None` 时完全不生效 | [growth_credit.py](../../taiji/growth_credit.py#L133-L141) |

---

## 7. 与现有的关系（不推翻，只补齐）

- **保留** `growth_gates` 语义：它是生长**之后**的每步分支门，不是触发器；本设计不改变它的含义。
- **保留** `birth()`、`lesion()`、`R_MAX`、max-min 分配：本设计只规定"何时调用、分配多少、失败怎么退"。
- **新增** 证书层（S-A/S-B/S-C）与决策树；证书函数放在新模块，不改冻结文件。
- **不做**：不在训练窗内自动生长；不引入新的可学参数；不改 FP1 算子本身。

## 8. 不宣称 / 开放项

- 本文的阈值未校准；在没有校准报告前，决策默认 `no_growth`。
- 20.3a 的塌缩归因（共享偏移 vs 内在低秩）需要**重放键**才能判定；本次只用其记录的角度读数验证"塌缩会被触发"，不冒充已判。
- 生长后的收益没有训练证据；本设计只给出可证伪的判据与失败出口。

---

## 9. 预审读数（2026-11-08，numpy 构造性，0 训练）

脚本 [verify_taiji_growth_design_20261108.py](../../scripts/training/verify_taiji_growth_design_20261108.py) → [报告](../../reports/taiji_growth_design_verify_20261108.json)，**8/8 通过**，CPU 0.047 s。

| 案例 | 构造 | 关键读数 | 判定 | 期望 |
|---|---|---|---|---|
| A 共享偏移塌缩 d=128 | 1 个共享方向 + 0.03 噪声 | `p95_cos 0.99931`、`median_margin 3.4e-5`、`top_share 0.99913`、`top_share_c 0.0623`、`eff_rank 1.00` | `representation_repair`（塌缩已触发） | ✓ |
| B 可分 d=128 | 32 个独立高斯方向 | `p95_cos 0.24966`、`median_margin 0.0322`、`top_share 0.0681`、`eff_rank 25.88` | `no_growth`（不误触发） | ✓ |
| C 容量不足·相容 | `ρ=6, C_F=4, δ_comp=0` | `δ_cap=2.0`、`σ₊=5.667`、下界 `11.33` | `capacity_growth` | ✓ |
| D 结构不可分 | 目标含行空间外分量 `δ_comp=0.5` | `δ_cap=2.0` 同时成立仍判结构 | `structural_repair` | ✓ |
| E 容量充足 | 同 C，`C_F=10` | `δ_cap=0, δ_comp=0` | `no_growth` | ✓ |
| F 同塌缩 d=512 | A 的生成式，维度 ×4 | `p95_cos 0.99925`、`top_share 0.99913`、`eff_rank 1.00` | 与 A 同判 | ✓ |
| G 20.3a 实测回放 | 记录值 `cos 0.99766 / margin 0.00835` | 触发塌缩 | `no_growth`（归因未定，扣住动作） | ✓ |
| H 分配 | 两条边际曲线 | `object=1, record=3` | 非平均分配 | ✓ |

三条可直接引用的结论：

1. **塌缩会触发，可分不会误触发**（A vs B）：判据只用了已记录量（top-1 余弦、margin）。
2. **扩容不改变角度塌缩**（A vs F）：维度从 128 加到 512，`p95_cos` 从 0.99931 到 0.99925、`eff_rank` 恒为 1.00——这正是"更大的同构不可分空间"的数值形态。
3. **容量不足与结构不可分可分**（C vs D）：C 只需把 `C_F` 提到 `ρ`；D 把 `C_F` 提到无穷也只能停在 `δ_comp=0.5`。

登记一个反直觉读数：A 的 `median_entropy = 1.00`（接近均匀）。当所有键彼此同样相似时，softmax 反而接近均匀——**熵不是塌缩信号**，所以判据不采用它，只把它当诊断量。

边界：全部为合成夹具；G 只用 20.3a 的记录聚合值回放，未重放键张量，故其归因（共享偏移 vs 内在低秩）仍记为未定。
