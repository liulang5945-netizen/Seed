# 生长机制设计规格（2026-11-08）：何时该长、长什么、长多少、何时放弃

> 状态 [设计规格·未实现进训练图]。0 训练 / 0 优化器步 / 0 Adam / 0 confirm。
> 代码事实基线：`growth_gates` 只是**已生长**主体的分支贡献门（[growth_credit.py](../../taiji/growth_credit.py#L42-L55)），真生长算子是 `birth()`（[growth_credit.py](../../taiji/growth_credit.py#L459-L525)），只被测试与离线验证脚本调用（`tests/taiji_native/*`、`scripts/training/verify_taiji_birth_*`、`scripts/training/diagnose_*`），当前训练窗只**读**门（[runner](../../scripts/training/run_taiji_training_window_v2_20261108.py#L212-L213)）。本文补的是缺失的那一层：**触发判据与分配规则**。
> 新模块 [taiji/growth_design.py](../../taiji/growth_design.py) 不被 runner 及其 import 闭包引用（见 [禁改清单](FROZEN-READSET_20261108.md)），纯 numpy，无 torch。

## 0.1 2026-11-08 owner 更正：撤回三条设计约束

> 更正的理由是**模型没有进化替它选好容量上限**：脑在发育期铺好容量、之后主要修剪，是因为进化已经选定了上限；模型不能预设上限，**生长必须是可持续从小到大的过程，扩容是必须的**。以下三条原文约束撤回，替换件写进正文；撤回不是措辞修饰，旧约束在旧版正文中的位置一并标出。

| ID | 撤回的约束（旧版位置） | 撤回理由 | 替换件（本版位置） |
|---|---|---|---|
| G-R1 | 「新单元必须与已有子空间**正交**」（旧版 §4 继承条、§6 I-G1 注释） | 完全正交＝互不相干，新单元无法复用旧方向，**迁移被切断**；真实表示是部分共享＋部分新意 | **部分重叠但可分**：`o_min ≤ o(u) ≤ o_max` 且 `s(u) ≥ s_min`（§4A） |
| G-R2 | 「**装满了才允许扩容**」（旧版 §2 决策树：只有 `δ_cap>0` 才 `capacity_growth`；槽满才换） | 预设上限；写入前不保证余量，会在关键反馈到来时因无槽而丢写或驱逐 | **容量先行**：写入前先算可写余量证书，不足则先扩容再写，失败可回滚（§5A） |
| G-R3 | 「读键＝整句 context 的**均值**」（20.3a 实现，本设计旧版默认接受） | 均值把不同问句压到同一个模板点上，是共享偏移塌缩的直接来源（[20.3a 键归因](../../reports/taiji_package20_3a_key_attribution_20261108.json)：去均值后读余量 0.00855→0.07947，9.3×） | **保留差异的局部化编码**：逐维方差加权池化＋top-k 位置残差，再按 store 均值中心化（§3A） |

> 撤回后的总原则：**生长是主路径，不是例外**；但「长出来的东西必须带来新方向」，否则只是复制。判据见 §4A 与 §5A。

---

## 0. 一句话

生长不是"开关"，而是**可持续从小到大的容量过程**：写入前先保证可写余量（容量先行），余量不足就按证书扩容；扩容必须带来**新方向**（部分重叠但可分），否则只是复制。编码层必须先保留差异（局部化＋中心化），不能把整句平均成单点——否则扩容只是把同一个不可分锥体放大。

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

### S-D 编码差异（第四信号：键是不是被"平均"掉了）

输入：同一批 query 的 context 张量 `H ∈ R^{T×d}`（逐字节位置）与写出的键。

| 量 | 定义 | 读法 |
|---|---|---|
| `pos_residual_ratio` | `mean_t ‖h_t − mean_t(h)‖ / ‖mean_t(h)‖` | 句子内部还有多少可用差异；接近 0 说明各位置几乎一样 |
| `pooled_share` | `‖mean_t(h)‖² / mean_t‖h_t‖²` | 均值池化保留了多少能量；高＝均值主导 |
| `topk_mass` | top-k 位置残差能量 / 总残差能量 | 差异是否集中在少数位置（局部化是否有效） |
| `raw_cos` / `centered_cos` | 键的两两平均余弦（原空间／去均值后） | `raw_cos` 高而 `centered_cos` 低＝共享偏移主导 |

判据：`pos_residual_ratio < 0.05` 或 `pooled_share > 0.90` ⇒ **编码把差异平均掉了**，先改编码（§3A），不得用扩容或后处理缩放补救。

---

## 2. 决策树（两个机制分开，禁止混判；容量先行）

```
每次写入前（两个机制都先跑容量先行）:
  S-D 编码差异不足 (pos_residual_ratio < 0.05 或 pooled_share > 0.90)
      -> encoding_repair                      # 先改 §3A，任何扩容暂缓

键空间（20.3a 内容寻址 store）:
  可写余量 h_slots < h_min_slots            -> capacity_growth(slots)    # 先长后写
  S-B 塌缩 且 S-A 为共享偏移
      -> representation_repair(centering)     # 去均值，不是扩容
  否则若 eff_rank ≥ effort_ratio·key_width   -> key_dimension_growth
  否则                                       -> no_growth

FP1 写入域（对象域 / 记录域）:
  δ_comp > ε_struct (ε_struct = 1e-6·‖V‖_F) -> structural_repair
  否则若 h_cap < h_min_cap 或 δ_cap > 0       -> capacity_growth(ball)     # 先长后写
  否则                                        -> no_growth
```

四条硬规则：

1. **容量先行**：任何写入批次在提交前必须满足 `h ≥ h_min`（§5A）；不满足就先扩容，扩容失败则拒绝写入并回滚，**不允许"写满了再说"**。
2. **编码先于扩容**：S-D 命中时，扩容不会改善可分性（20.3a 的 128→512 同判塌缩就是证据），必须先改 §3A 的编码。
3. **生长有效 ≠ 生长即复制**：新增单元/维度只有在 §4A 的相关带内才计入容量证书；`o(u)=1`（复制）或 `o(u)=0`（孤立）都判 `growth_is_copy`，**不得记为容量收益**。
4. 两个机制的判据**不得互相代言**：store 上没有 `C_F`，FP1 上没有余弦打分。

阈值形式冻结、数值待校准：`0.90 / 0.50 / 0.95 / 0.02 / 0.35 / 1e-6` 与新增的 `h_min`、`o_min/o_max/s_min` 都是**预审建议值**（§9），在出现第一份校准报告前，决策函数对任何未校准域一律返回 `no_growth`（默认不长）。

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

## 3A 编码层：保留差异的读键（替换整句均值）

**问题**：20.3a 的 `store_query_key = context.mean(0)` 把所有位置压成一个点；不同问句的键只差一个共享模板方向，余弦 0.9977、margin 0.0084。归因回执判 `shared_offset`——去均值即可把 margin 提到 0.0795（9.3×），说明差异还在 context 里，是**池化把它平均掉了**。

**新编码（`mode = "localized_centered"`，默认关闭，旧路径逐位不变）**：给定位置序列 `H = [h_1…h_T] ∈ R^{T×d}`，

1. 逐维均值 `μ_j = mean_t h_{t,j}`，逐维离差 `δ_{t,j} = |h_{t,j} − μ_j|`；
2. **逐维方差加权池化**：`w_{t,j} = softmax_t(δ_{t,j} / τ)`，`m_j = Σ_t w_{t,j} h_{t,j}`——差异大的位置在**该维**上获得更大权重，而不是所有维共用一个句级权重；
3. **top-k 位置残差**：`r_t = h_t − μ`，取 `‖r_t‖` 最大的 K 个位置，`z = (1/√K) Σ_{t∈TopK} r_t`；
4. **合并**：`q = m + z`；
5. **中心化**：`q ← q − c`，`c` 为已占用键的均值（store 级统计量；空 store 时 `c=0`）；
6. `L2` 归一化后写入/检索。

**为什么它自然分开、而不是靠后期缩放**：
- 不同问句的差异集中在**不同位置的不同维度**上；逐维 softmax 把"哪一维在哪个位置突出"编码进键，而不是把所有位置平均掉；
- top-k 残差保留"这句话最不寻常的几个位置"，两句只差属性值时，残差方向就不同；
- 中心化去掉模板方向（共享偏移），这是归因回执直接支持的修复；不改变键宽、不引入新参数、无随机性。

**主因与增量（预审案例 J 修正了本节的表述重点）**：中心化是**决定性**修复（margin `3.9e-5→0.02388`）；局部化在其上再加约 30%（`0.03109`），但**不能替代中心化**——只局部化不中心化时仍 `4.8e-4`，依旧塌缩。因此顺序固定为：先中心化，再局部化；两者都不做时不准进入扩容分支。

**接口**（新增 `taiji/growth_runtime.py`，纯 numpy，不被冻结图 import）：

```python
@dataclass(frozen=True)
class EncoderConfig:
    mode: Literal["mean", "localized_centered"] = "mean"   # 默认旧路径
    tau: float = 0.5
    top_k: int = 4
    center: bool = False

def encode_read_key(context: np.ndarray, cfg: EncoderConfig, centroid: np.ndarray | None) -> np.ndarray
def store_centroid(keys: np.ndarray, valid: np.ndarray) -> np.ndarray
```

- 形状：`context (T,d) -> key (d,)`，**键宽不变**（`d = config.width`）；
- 参数增量：0；状态增量：`centroid (d,)` ＋ `centroid_age (1,)`（共 `d+1` 个 float，128→516 B）；
- 身份影响：`mode="mean"` 时 `encoder_schema` 保持旧值，摘要不变；切到 `localized_centered` 时把 `encoder_schema` 写进契约并 bump `contract_version`（摘要改变，属预期）；
- 恢复：checkpoint 必须带 `encoder_schema`、`centroid`、`centroid_age`；缺失时按 `mode="mean"` 恢复（向后兼容）；
- 与 20.3a 的关系：这是**替换** `store_query_key` 的构造，不是改 `addressed_read` 的余弦打分；旧 store 仍可只读重放。

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

## 4A 新方向的相关带：部分重叠但可分（替换"正交"）

设已有单元方向 `u_1…u_n`（单位化）张成子空间 `B = span{u_i}`，候选新方向 `u`（单位化）：

| 量 | 定义 | 含义 |
|---|---|---|
| `o(u)` | `‖Bᵀu‖₂`（= 到子空间的投影范数） | **重叠度**：0＝完全正交，1＝完全落在旧子空间 |
| `r(u)` | `‖u − BBᵀu‖₂ = √(1−o²)` | **新意**：新方向里旧子空间解释不了的部分 |
| `s(u)` | `1 − max_i |cos(u,u_i)|` | **可分辨性**：与最近旧单元的角度间隔 |
| `t(u)` | `max_i |cos(u,u_i)|` | **可迁移性**：能复用多少旧方向 |

**相关带（判据）**：
\[
o_{min}\le o(u)\le o_{max},\qquad s(u)\ge s_{min},\qquad r(u)\ge r_{min}
\]
其中 `r_min = √(1−o_max²)` 由 `o_max` 导出，不独立校准。预审建议值（§9 案例 I）：`o_min=0.15`、`o_max=0.85`、`s_min=0.30` ⇒ `r_min≈0.527`。**数值不拍死**：它们是预审给出的起点，第一份校准报告后按域替换。

**判定**：
- `o > o_max` 或 `s < s_min` ⇒ **`growth_is_copy`**：新单元与旧方向太像，**不计入容量证书**，也不写 `growth_retired`（它只是没长成新方向，不是容量假设失败）；
- `o < o_min` ⇒ **`growth_isolated`**：与旧子空间互不相干，迁移被切断，同样不计入容量收益；
- 带内 ⇒ `growth_new_direction`：允许计入 `δ_cap` 的容量收益，并在失败出口按 §5 复算。

**对键维生长**：新增的键维（列）同样按上式对旧列空间算 `o/s`；整列复制（`o=1`）判复制，整列随机且与旧列近乎正交（`o<o_min`）判孤立。**对槽位生长**：槽位是状态不是方向，不适用相关带；槽位只按 §5A 的余量证书判定。

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

## 5A 容量先行：写入前必须先有的余量（替换"装满才扩容"）

旧规则"装满才扩容"把上限预设成了"当前容量"，关键反馈到来时只能丢写或驱逐。改为：**每次写入前先算可写余量证书；不足则先扩容、再写；扩容失败则整批拒绝并回滚**。

### 5A.1 两个余量证书（都可只读算出）

| 对象 | 余量 | 定义 | 不足时 |
|---|---|---|---|
| 槽位 | `h_slots` | `(slots − occupied − pending) / slots` | 按 2 倍扩容到 `h_slots ≥ h_min_slots`，到 `slot_cap` 仍不足则拒绝 |
| 球半径 | `h_cap` | `(C_F − ρ_after) / C_F`，`ρ_after` 用"已写 + 本批"的 `‖VK⁺‖_F` 试算 | 取 `C_F' = ρ_after / (1 − h_min_cap)` 向上取整后扩容 |

- `h_min_slots = 0.125`（预审建议值）：**写完之后**仍保留至少 1/8 的空槽，用于下一批与纠正写；`pending` 是本批要写的槽数，不是 1。
- `h_min_cap = 0.20`（预审建议值）：球半径至少留 20% 余量，避免下一批立刻再触发扩容。
- 两个证书都只依赖已记录的 `occupied/slots/pending`、`ρ`、`C_F`，不需要反传。
- 余量为负（`spills`）不只是"不够"，而是**本批会溢出**：必须扩容或拒绝，禁止静默驱逐。

### 5A.2 顺序协议（谁先谁后）

1. **干跑编码**：把本批 `(T,d)` context 编成键，只在内存里，不碰 store；
2. **算证书**：`write_headroom()` 与球证书，记录 `before`；
3. **先扩容**：不足时 `plan_growth()` 给出目标容量 → 扩容落盘（新张量 + checkpoint）→ 校验 I-G1/I-G2 → 重算证书；
4. **扩容失败或仍不足**：`reject_write`，恢复快照，记 `write_rejected{reason}`；**不写任何槽、不驱逐任何旧键**；
5. **原子写入**：快照 → 在候选副本上写完整批 → 校验 `size ≤ capacity` → 一次性替换；
6. **写后校验**：父行逐位不变、新槽中性、身份摘要可复算；任一不符 → 回滚到快照；
7. **记账**：写 `turn_id、batch_sha、h_slots before/after、growth_id、counted/rejected directions`，恢复时按账本重放或丢弃。

### 5A.3 回滚

- 槽位扩容的回滚单位是**整批**，不是单槽：快照是 `store_keys/store_values/store_age/store_count` 的一份拷贝，64 槽时约 `1.02 MiB`，只在 **turn 边界**做一次，不在读路径上；
- 参数生长（单元/键宽/通道）用 `lesion()` 精确复原父主体（[growth_credit.py](../../taiji/growth_credit.py#L528-L545)）；
- 同一域连续 2 次"扩容后仍不足"⇒ 写死 `growth_retired[domain]=true`（§5 失败出口），此后该域超限写入一律 `reject_write`，不驱逐。

---

## 6. 不变量（生长必须满足）

| 编号 | 条款 | 现有代码依据 |
|---|---|---|
| I-G1 出生中性 | 出生步新单元输出精确为 0，父主体前向逐位不变 | R/P 新行列零初始化（[growth_credit.py](../../taiji/growth_credit.py#L495-L505)） |
| I-G2 父行不变 | 父行/父列逐位拷贝，不重初始化 | `load_state_dict` + 拼接（[native_development.py](../../taiji/native_development.py#L53-L66)） |
| I-G3 身份可复现 | 生长块（parent sha、directions sha、seed、birth step、contract_version、`encoder_schema`、`generation`）进入摘要；**未生长路径摘要不变** | `graph_identity` 现覆盖参数形状（[identity](../../taiji/rule_learning_20_1_identity.py#L104-L106)） |
| I-G4a 参数冻结 | 单元/键宽/通道的**参数生长只允许在会话之间**；会话内参数仍冻结 | 20.1 契约 `session_freeze`（[identity](../../taiji/rule_learning_20_1_identity.py#L39)） |
| I-G4b 状态容量可长 | 槽位与球半径的容量生长只允许在 **turn 边界**；读路径与 turn 中途不得扩容 | 本设计 §5A.2（新增；不改冻结文件） |
| I-G4c 编码切换边界 | 编码模式切换只允许在**会话边界**，并须整库重键或双 schema 并存；不得在 turn 中途切换 | 本设计 §3A（新增） |
| I-G5 恢复一致 | checkpoint 必须携带 R/P 形状、`growth_parent_units`、分支门表、`encoder_schema`、`centroid/centroid_age`；`lesion` 精确复原父主体 | [growth_credit.py](../../taiji/growth_credit.py#L528-L545) |
| I-G6 默认零开销 | 不调用触发即与今日逐位一致；门在 `growth_parent_units is None` 时完全不生效；`mode="mean"` 编码逐位等于 `context.mean(0)` | [growth_credit.py](../../taiji/growth_credit.py#L133-L141)＋[growth_runtime.py](../../taiji/growth_runtime.py) |
| I-G7 容量先行 | 没有 `h ≥ h_min` 就不提交写入；被拒绝的写入让 store **逐位不变**，且不驱逐任何旧键 | 本设计 §5A（预审案例 K） |
| I-G8 方向带 | `growth_is_copy` / `growth_isolated` 永不进入容量证书；证书里分列 `counted_directions` 与 `rejected_directions` | 本设计 §4A（预审案例 I） |

---

## 7. 与现有的关系（不推翻，只补齐）

- **保留** `growth_gates` 语义：它是生长**之后**的每步分支门，不是触发器；本设计不改变它的含义。
- **保留** `birth()`、`lesion()`、`R_MAX`、max-min 分配：本设计只规定"何时调用、分配多少、失败怎么退"。
- **新增** 证书层（S-A/S-B/S-C/S-D）＋容量先行余量证书（§5A）＋方向相关带（§4A）；证书函数放在新模块，不改冻结文件。
- **不做**：不在训练窗内自动生长；不引入新的可学参数；不改 FP1 算子本身。

## 8. 不宣称 / 开放项

- 本文的阈值未校准；在没有校准报告前，决策默认 `no_growth`。
- 20.3a 的塌缩归因**已判定为共享偏移**（只读重放，[报告](../../reports/taiji_package20_3a_key_attribution_20261108.json)：读余量 `0.00855→0.07947`，9.3×；去均值后 top_share `0.956→0.498`）。它**不是**"容量不足"：该 world 高水位 28/32、0 驱逐，槽位从未成为瓶颈，故"真·容量不足"不可判定（缺负载饱和重放）。结论：20.3a 的塌缩应由编码层修复（§3A），**不是**扩容；生长在该 world 上仍记为"待验"，不宣称有效。
- 该归因只说明**读键几何**，不是任务效用；20.3a 的任务级失败不因此翻案。
- 生长后的收益没有训练证据；本设计只给出可证伪的判据与失败出口。
- §11 的功能清单是**待验证口径**，不是已具备能力；任何一条在拿到离线夹具读数前都不得对外宣称。

---

## 9. 预审读数（2026-11-08，numpy 构造性，0 训练）

脚本 [verify_taiji_growth_design_20261108.py](../../scripts/training/verify_taiji_growth_design_20261108.py) → [报告](../../reports/taiji_growth_design_verify_20261108.json)，**12/12 通过**，CPU 0.062 s。

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
| I 方向相关带 | 8 个旧方向；复制 / 正交 / 重叠 0.55 | 复制 `o=1.00, s=0.00`；正交 `o=0.00, s=1.00`；带内 `o=0.55, r=0.835, s=0.729` | 复制→`growth_is_copy`；正交→`growth_isolated`；带内→`growth_new_direction`（仅此计容量） | ✓ |
| J 编码：均值 vs 局部化 | 32 句、T=8、共享模板 + 单个值 token | 旧均值 `margin 3.9e-5`；均值+中心化 `0.02388`；局部化不中心化 `4.8e-4`；局部化+中心化 `0.03109` | 中心化是决定性修复；局部化在其上再 +30%，**但不能替代中心化** | ✓ |
| K 容量先行协议 | `30/32` 槽、待写 3 | `h_slots=−0.031` → 先扩到 64 再写；余量足则直接写；`slot_cap=32` 拒绝；模拟扩容失败整批回滚、store 不变 | 顺序与回滚符合 §5A.2/§5A.3 | ✓ |
| L 隔离性 | 扫 25 个冻结文件与 runner | 冻结文件 25，提及 `growth_design/growth_runtime` 的 0 个；runner 0 处 | 新模块在冻结图之外 | ✓ |

三条可直接引用的结论：

1. **塌缩会触发，可分不会误触发**（A vs B）：判据只用了已记录量（top-1 余弦、margin）。
2. **扩容不改变角度塌缩**（A vs F）：维度从 128 加到 512，`p95_cos` 从 0.99931 到 0.99925、`eff_rank` 恒为 1.00——这正是"更大的同构不可分空间"的数值形态。
3. **容量不足与结构不可分可分**（C vs D）：C 只需把 `C_F` 提到 `ρ`；D 把 `C_F` 提到无穷也只能停在 `δ_comp=0.5`。

补充两条（本轮新增）：

4. **编码修复的主因是中心化，局部化是增量**（J）：只做中心化已把 margin 从 `3.9e-5` 提到 `0.02388`；再加局部化到 `0.03109`（+30%）。**局部化不能替代中心化**（不中心化时仍 `4.8e-4`，依然塌缩）。这修正了 §3A 的表述重点：先中心化，再局部化。
5. **"正交"与"复制"同被判无效**（I）：两者都不计容量；只有落在 `o∈[0.15,0.85]` 且 `s≥0.30` 的带内方向才算新方向。

登记一个反直觉读数：A 的 `median_entropy = 1.00`（接近均匀）。当所有键彼此同样相似时，softmax 反而接近均匀——**熵不是塌缩信号**，所以判据不采用它，只把它当诊断量。

边界：全部为合成夹具。G 只用 20.3a 的记录聚合值回放，未重放键张量；其归因已由独立的只读重放报告判定为 `shared_offset`（[报告](../../reports/taiji_package20_3a_key_attribution_20261108.json)），本节不复用该结论，只把它当外部输入。

---

## 10. 可实现件：数据结构、接口、状态与身份

### 10.1 模块边界（全部在冻结图之外）

| 模块 | 内容 | 依赖 | 是否被 runner import |
|---|---|---|---|
| [taiji/growth_design.py](../../taiji/growth_design.py) | 键谱/读余量/容量证书/方向带/决策树/分配 | numpy | 否（预审案例 L：0 个冻结文件提及） |
| [taiji/growth_runtime.py](../../taiji/growth_runtime.py) | 编码器、余量证书、扩容计划、原子写入与回滚 | numpy | 否（同上） |
| 未来适配器（未建） | 在 turn 边界把 store 交给 `write_batch`，读侧调用 `encode_read_key` | 上述两模块 | 只有显式打开 `growth.enabled` 才 import；默认关闭 |

**保证不碰在跑训练窗**：新文件是唯一改动面；冻结 25 文件字节不变（清单 [FROZEN-READSET](FROZEN-READSET_20261108.md)）；预审案例 L 每次运行都会重扫冻结集与 runner，出现提及即失败。

### 10.2 数据结构

```python
@dataclass(frozen=True)
class EncoderConfig:          # 已实现
    mode: str = "mean"        # "mean" | "localized_centered"
    tau: float = 0.5
    top_k: int = 4
    center: bool = False
    # schema: "mean-v1" | "localized_centered-v1"

@dataclass
class GrowthState:            # 待实现（纯状态，不含参数）
    generation: int = 0
    slots: int = 32
    slot_cap: int | None = None
    c_f: float = 0.0
    encoder_schema: str = "mean-v1"
    centroid: np.ndarray | None = None
    centroid_age: int = 0
    counted_directions: int = 0
    rejected_directions: int = 0
    retired: set[str] = field(default_factory=set)
```

### 10.3 接口签名

| 函数 | 签名 | 状态 |
|---|---|---|
| 编码 | `encode_read_key(context (T,d), cfg, centroid) -> (d,)` | 已实现 |
| store 统计 | `store_centroid(keys, valid) -> (d,)` | 已实现 |
| 余量 | `write_headroom(occupied, slots, pending, reserve_ratio) -> dict` | 已实现 |
| 扩容计划 | `plan_growth(headroom, slot_cap, growth_factor) -> dict` | 已实现 |
| 原子写 | `write_batch(store, batch, headroom, plan) -> {committed, store, rolled_back}` | 已实现 |
| 方向带 | `direction_band(existing (n,d), candidate (d,)) -> {o,r,s,t,verdict}` | 已实现 |
| 球扩容 | `apply_ball_growth(c_f, rho_after, h_min_cap) -> c_f'` | 待实现（纯算术，见 §5A.1） |
| 参数生长 | `apply_unit_growth(state, directions, seed) -> (state, lesion_token)` | 复用 `birth()`/`lesion()`，不新写 |
| 快照 | `snapshot(store) -> token` / `restore(store, token)` | 待实现（整批拷贝，§5A.3） |

### 10.4 状态与参数增量

| 项 | 增量 | 说明 |
|---|---|---|
| 编码参数 | **0** | 中心化用统计量，不是可学参数 |
| 编码状态 | `centroid(d)` + `centroid_age(1)` = 129 float ≈ **516 B** | 空 store 时 `centroid=0` |
| 槽位 32→64 | 每槽 `128 + 4·2·32·32 + 2 = 8,322` float ≈ **33 KB**；翻倍 **+1.02 MiB** | 状态，不进身份摘要的参数表 |
| 球半径 | 1 个标量 | 参数表不变 |
| 方向计数/代际 | 常数字节 | 进摘要 |

### 10.5 身份摘要影响

- `mode="mean"` 且不生长：`encoder_schema` 保持 `mean-v1`，`candidate_sha_v2` **逐位不变**；
- 切到 `localized_centered`：`encoder_schema` 进契约并 bump `contract_version`，摘要改变（预期）；
- 生长：摘要追加 `parent_sha`、`directions_sha`、`seed`、`birth_step`、`generation`；父行不变（I-G2），未生长路径不受影响（I-G6）。

---

## 11. 功能清单：这套算法要交付什么（每条给可检验口径）

> 全部为**待验证口径**，不是已具备能力。每条都能在"0 训练"的离线夹具上量；本设计只保证判据可算、失败可判。

| # | 功能 | 可检验口径 | 冲突与处理 | 当前状态 |
|---|---|---|---|---|
| F1 | 反馈之后能改对下一次（纠正） | 对每次纠正写 `(q, v')`，下一次同义问句读出的 top-1 为 `v'`：`correction@1 ≥ 0.90`（≥50 次纠正）；未触及的配对不回退（`Δaccuracy ≥ 0`） | 纠正**写新槽**、不覆盖旧槽；容量先行保证纠正不因槽满被丢 | 未验证（需离线夹具，0 训练） |
| F2 | 跨会话/长时程保持 | A 会话写入 → 冻结 → B 会话恢复后不重写直接读：`retention@1 ≥ 0.90`（≥50 对），且恢复逐位一致（I-G5） | 保持靠"旧键不被驱逐"：容量先行＋纠正写新槽共同保证 | 未验证 |
| F3 | 容量按需从零长到足够 | 连续多代 `8→16→32→64→128`，每批满足 `h ≥ h_min`，`dropped_writes = 0`、`evictions = 0` | 生长与冻结：参数只在会话间长（I-G4a），槽位只在 turn 边界长（I-G4b） | 协议已预审（K），代际连续未验证 |
| F4 | 旧的结构能力（Task1 式绑定复制）不被挤掉 | 关闭生长时与今日**逐位相同**（I-G6）；开启生长后 Task1 夹具准确率 `≥ baseline`，且 `δ_comp` 不增大 | 生长不得改写父行（I-G2）；新单元出生中性（I-G1） | 未验证（需跑既有 Task1 夹具，0 训练） |
| F5 | 冲突处理（生长 vs 冻结、纠正 vs 保持） | ①会话内参数 sha 不变；②纠正后旧配对 `retention@1` 不降；③编码切换只在会话边界且整库重键，`encoder_repair` 不计容量 | 见左栏三条边界；任一违反即判该次生长无效并回滚 | 规则已定，数值未验证 |
| F6 | 生长有效性（不是复制） | 新增方向必须 `o∈[o_min,o_max]`、`s≥s_min`；`growth_is_copy`/`growth_isolated` 不计入容量证书 | 复制式扩容不产生容量收益，只能算"更大但同样塌缩"（案例 A vs F） | 判据已预审（I），真实主体未验证 |

**验收顺序（唯一建议的下一步）**：先在离线夹具上跑 F1/F4 的"不生长"分支，确认与今日逐位一致（I-G6）＋ Task1 不下降；只有这两条过了，再验证 F3 的多代扩容。理由：F1/F4 是"不破坏"的门，F3 是"能长"的门；先证不破坏，再证能长，避免用扩容掩盖回归。
