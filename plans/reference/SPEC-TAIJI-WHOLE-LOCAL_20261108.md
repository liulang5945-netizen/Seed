# 整体与局部的矛盾：四层职责、共识门判据、生长对象重定义

> 状态 [设计规格·未实现进训练图]。**0 训练 / 0 优化器步 / 0 Adam / 0 confirm**；未改任何冻结文件；未打断在跑训练窗；CPU 0.00 s。
> 本文回答 owner 的一条判断：**整体与局部的矛盾是生长机制的根本，当前算法没解决它**。本文不停在原则层——每一节都要落到"哪个张量、哪行代码、哪个可计算量、失败时退回哪里"。
> 配套：[生长机制设计规格](SPEC-TAIJI-GROWTH_DESIGN_20261108.md)（管"何时长、长多少"）／[不变量账本](INVARIANT-LEDGER_20261103.md)／[F3 四代读数](../../reports/taiji_growth_f3_multigen_20261108.json)／[F1 纠正读数](../../reports/taiji_growth_f1_correction_20261108.json)。

---

## 一、现状：用代码与配置事实回答三个问题

### 1.1 当前训练是**整体**，不是局部

| 事实 | 代码位置 | 含义 |
|---|---|---|
| 冻结的只有 `world` 一个组 | [probe_taiji_package20_4c_r2_formal_comparison.py](../../scripts/training/probe_taiji_package20_4c_r2_formal_comparison.py#L308-L310) | `parameter.requires_grad_(_group_of(name) != "world")`——除 `world.*` 外**全部可训** |
| 参数分组共 17 组 | [rule_learning_20_1_net.py](../../taiji/rule_learning_20_1_net.py#L43-L62) | `encoder / nu / message / fast_read / episodic / W / U / V / P / R / tau / xi / readout / expression / write_heads / key_content_proj / world`，只有最后一组冻结 |
| Adam 覆盖全部可训参数 | [runner](../../scripts/training/run_taiji_training_window_v2_20261108.py#L426-L435) → [trainer_params](../../scripts/training/probe_taiji_package20_4c_r2_formal_comparison.py#L313-L314) | `[p for p in graph.subject.parameters() if p.requires_grad]`，3,323,547 参数里除 `world.*` 外全在优化器里 |
| 每次更新只有一个标量 loss | [probe](../../scripts/training/probe_taiji_package20_4c_r2_formal_comparison.py#L495-L505)、[L539](../../scripts/training/probe_taiji_package20_4c_r2_formal_comparison.py#L539) | 只有 `correction_surface` 的 turn 才算：`DECISION_POSITIONS` 上交叉熵求和，`loss = losses.sum()` |
| 反传一次即改写全部共享张量 | [runner L216-L224](../../scripts/training/run_taiji_training_window_v2_20261108.py#L216-L224) | `zero_grad → run_pair(train=True) → backward → clip → optimizer.step()` |

**结论（事实，不是立场）**：一条具体纠正 `(q, v')` 进来，经过的是**一次标量梯度、一次 Adam、落在 3.3M 个共享参数上**。这条纠正本身没有"只属于它自己"的存放位置——训练里唯一带位置感的存储（store）在训练窗里根本没开：

| 事实 | 代码位置 | 含义 |
|---|---|---|
| store 四件套是 **session 状态**，不是参数 | [rule_learning_20_1_net.py L236-L278](../../taiji/rule_learning_20_1_net.py#L236-L278) | `kind: "session"`、`plastic: False`、`restore: 值；新 session 首轮清空`——不进梯度、不进身份 |
| 训练窗里 `ADDRESSED_STORE = False` | [probe](../../scripts/training/probe_taiji_package20_4c_r2_formal_comparison.py#L118) | `store_enabled = addressed_store_enabled and fast_enabled`（[L126-L127](../../taiji/rule_learning_20_1_net.py#L126-L127)）⇒ **两臂都关**，读/写分支不执行（[L501-L502](../../taiji/rule_learning_20_1_net.py#L501-L502)） |

所以"整体"有两个层次：**优化器是全局的**（一条纠正推动全部参数），**局部存储在训练时是关着的**（想局部也没有承载物）。

### 1.2 生长实际长的是什么：三种东西，只有一种在跑

| 生长物 | 实现 | 训练时是否发生 | 是否进身份 |
|---|---|---|---|
| **参数整体漂移**（不是"生长"，是"学习"） | Adam 全量更新 | **是**，每步 3.3M 参数 | 是（权重进 `state_dict`） |
| **槽位**（存储） | [addressed_write](../../taiji/living_core.py#L588-L615) | **否**（`ADDRESSED_STORE=False`） | 否（session 状态，[L236-L278](../../taiji/rule_learning_20_1_net.py#L236-L278)） |
| **方向**（R 行 / P 列） | [birth()](../../taiji/growth_credit.py#L459-L525) | **否**（只被测试与离线验证脚本调用） | 是（`R`/`P` 是 `Parameter`） |

`birth()` 长的东西是**方向**：每组每单元新增一行 `R`（宽 `width=128`）＋一列 `P`（高 `parent_p.shape[1]`），合计 `2d = 256` 个 float；新行新列**零初始化**，所以出生步贡献精确为 0（[L480-L500](../../taiji/growth_credit.py#L480-L500)），只能靠分支门逐级放开（`pre += Σ_c g_c · B_c · tanh(R_c x)`，[L39-L56](../../taiji/growth_credit.py#L39-L56)）。`R_MAX = 8`。

### 1.3 "局部扩张"的自主权现在在哪：**没有**

三处证据：

1. **长不长，不由局部决定**。训练循环里没有任何调用点会触发 `birth()`（[runner L205-L224](../../scripts/training/run_taiji_training_window_v2_20261108.py#L205-L224) 全循环没有剪枝/出生）。生长的唯一入口在测试里（`tests/taiji_native/*`、`scripts/training/verify_taiji_birth_*`）。
2. **长了算不算，不由局部决定**。`growth_rows` 只把门值当成一个乘进前向的常数（[living_core_batch.py L92-L105](../../taiji/living_core_batch.py#L92-L105)）：门给 0.3 就是"贡献打三折"，没有任何"准不准入"的判定。
3. **局部写进去，能不能被读出来，也不由局部决定**。见 §1.4 的 α 稀释。

### 1.4 共识门当前的实际取值与语义：**常开 + 未接线，且不是门**

三件事实叠加：

| 事实 | 代码位置 | 后果 |
|---|---|---|
| 配置值是常数 `{"consensus": 1.0}` | 来自父包 `growth_gates`，[runner L419-L421](../../scripts/training/run_taiji_training_window_v2_20261108.py#L419-L421) | 1.0 ＝"新生单元贡献按原值全额给"，是一个**权重**，不是判据 |
| 键名 `consensus` 是**出生方向规则名** | [birth_direction.py L17](../../taiji/birth_direction.py#L17) `RULE_NAME = "consensus"` | 它是 `consensus_directions()` 产出方向的命名空间，不是"共识校验"的意思 |
| **当前主体没出生过，整条路直接短路** | [rule_learning_20_1_net.py L128-L131](../../taiji/rule_learning_20_1_net.py#L128-L131)：`grown = getattr(subject,"growth_parent_units",None) is not None`，不成立则 `step_kwargs = {}`；[living_core_batch.py L93-L95](../../taiji/living_core_batch.py#L93-L95)：`parent >= core.R.shape[1] → return (parent, None)` | 门字典**根本没被送进前向**；既不是"常开"，也不是"校验"——是**死配置** |

**所以"把 consensus 从 1.0 调小"不解决问题**：调小只是让（不存在的）新生单元贡献变小，仍然没有准入判定。真正缺的是一个**提交前校验 + 失败原子回滚**的协议（§3）。

### 1.5 还有一个被忽略的稀释源：α = 0.5

即使把 store 打开、把门接上，读路径本身还会再混一次：

```
retrieved = Σ_i w_i · v_i                     # 温度 0.1 的余弦 softmax
fast_new  = (1-α)·fast_old + α·retrieved       # α = ADDRESSED_ALPHA = 0.5
```

见 [living_core.py L565-L567](../../taiji/living_core.py#L565-L567)、[L18-L20](../../taiji/living_core.py#L18-L20)。代入 F1 实测的 softmax 权重 0.51：

```
fast_new ≈ 0.5·fast_old + 0.5·(0.51·v' + 0.49·v_old) ≈ 0.255·v' + 0.745·v_old
```

即：**一条局部纠正对 `fast` 的理论贡献上限是 `α × w_top = 0.5 × 0.51 ≈ 25.5%`**（上式假设 `fast_old ≈ v_old`，即没有其它写入干扰；这是上界，不是实测）。F1 在脚本里绕开了 α，只测了 softmax 那一层就已经只剩 51%。两层稀释叠起来，局部写入在结构上就不可能"取代"。

**这一节的全部结论可以压成一句**：现在既没有"局部的承载物"（store 在训练时关闭），也没有"局部的裁决者"（门是死配置），也没有"局部的通行权"（α 把贡献压到 1/4）。整体与局部的矛盾不是调度问题，是**三层结构性缺失**。

---

## 二、四层职责划界：接口与不变量

分层原则只有一条：**决定"能不能分开"的东西必须全局一致；决定"这条怎么答"的东西必须只属于这条**。混在一起就会出现"改一条纠正、动全部参数、没人复核"。

### 2.1 L1 共享度量层（shared metric）——决定可分性，全局可训

**管什么**：读键怎么编（`store_query_key`）、打分怎么算（余弦 + 温度）、读出怎么落到 `fast`（α）。这一层决定"两条不同问句能不能被分开"，所以它**必须全局唯一、全局可训**——局部各改一套就没有共同坐标系。

**接口**（新增 `taiji/growth_gate.py`，纯 numpy，不被冻结图 import）：

```python
@dataclass(frozen=True)
class MetricSnapshot:
    version: int                 # 单调递增
    encoder_schema: str          # "mean" | "localized_centered"
    temperature: float
    alpha: float
    sha256: str                  # 上面全部字段 + 参数指纹的摘要

class SharedMetric:
    def snapshot(self) -> bytes                 # 逐位可恢复的完整快照
    def propose(self, delta: MetricDelta) -> CommitId
    def commit(self, commit_id: CommitId) -> CommitReceipt   # 唯一可写入口
    def restore(self, snapshot: bytes) -> None  # 逐位恢复，sha256 复核
```

**能被改的入口与改法（不许有第二个入口）**：
- 改法只有一种：`propose → 门校验（§3）→ commit`；
- 训练期的 Adam 更新**不直接改 L1**，而是产出一个 `MetricDelta`（本次更新的参数增量），由门决定是否 commit；
- 改完必须 `version += 1`，并把 `encoder_schema` 写进契约、bump `contract_version`（见 [生长设计 §3A](SPEC-TAIJI-GROWTH_DESIGN_20261108.md)）。

**不变量**：

| ID | 不变量 | 可执行检查 |
|---|---|---|
| I-W1 | 度量版本单调：`version' = version + 1`，不可回退、不可跳号 | `assert version_prime == version + 1` |
| I-W2 | 度量只能经 `commit` 改变；其余路径只读 | 全模块 grep：除 `commit` 外无 `store_query_key` / `temperature` / `alpha` 赋值 |
| I-W3 | 每次 `commit` 必须携带通过的门证书（§3 的四条件全绿） | `assert receipt.cert.passed` |
| I-W4 | 快照必须逐位可恢复：`restore(snapshot(S)) == S` | `sha256` 相等 + 张量 `array_equal` |
| I-W5 | 度量改变不得降低保留集正确数（G1） | 见 §3.3 |

### 2.2 L2 局部值层（local value）——只存该条修正，会话内可写

**管什么**：一条 `(q, v')` 的**值**。它只回答"这个问句要什么"，不回答"两个问句像不像"。

**现状的硬缺陷**（F1 已证）：[addressed_write](../../taiji/living_core.py#L588-L615) 只会写空闲槽（`slot = occupied`）或满了驱逐最老（`slot = argmin(store_age)`），**同键永不覆盖**。于是同一个问句的第二条纠正变成"又开一个槽"，读的时候按余弦对半分（0.51 / 0.49）。这不是容量问题，是**语义问题**：值层的语义应该是"按地址取代"，实现成了"按时间追加"。

**接口**：

```python
@dataclass(frozen=True)
class ValueCell:
    key: np.ndarray          # (d,)  单位化
    value: np.ndarray        # value_shape
    age: int                 # 写入序号，-1 = 空
    hits: int                # 被 top-1 选中的次数（修剪层用）
    metric_version: int      # 写入时的度量版本

def write_value(state, cell, metric) -> WriteReceipt   # 同键 -> 取代；空槽 -> 占用；满 -> 拒绝（不是驱逐）
```

**权限边界（这是"局部无权改度量"的落地方式）**：

| 允许 | 禁止 |
|---|---|
| 写 `store_values[slot]` | 改 `encoder*` 任何权重 |
| 写 `store_keys[slot]`（用**当前**度量编码出来的键） | 改 `temperature` / `alpha` |
| 写 `store_age` / `hits` | 改任何 `Parameter`（`grad` 必须为 `None`） |
| 读 `metric_version` 做兼容判断 | 触发 `birth()` |

**不变量**：

| ID | 不变量 | 可执行检查 |
|---|---|---|
| I-W6 | 值层写入不产生梯度：写入前后所有 `Parameter.grad is None` | `assert all(p.grad is None for p in params)` |
| I-W7 | 值层写入不改 `metric_version` | `assert v_after == v_before` |
| I-W8 | 值层缓冲的 shape 中不得出现任何 L1 权重 | 白名单校验：只允许 `store_keys/values/age/hits/count` |
| I-W9 | 同键必须**取代**，不得追加（F1 的教训） | 写入前后 `occupied` 增量为 0，且 `store_values[slot]` 逐位等于 v' |
| I-W10 | 满槽时**拒绝写入**，不得驱逐 | `evictions == 0`、`dropped_write == 1`（进隔离队列，§3.5） |

### 2.3 L3 热点反馈层（hotspot）——定位到子空间，只改该子空间

**管什么**：当"某一类问句反复读错"时，判断这是**局部可修的**还是**度量不可分的**。若是前者，只改那个子空间；若是后者，上交 L1。

**定位判据（不是全量重训，全部可离线算）**：

把每次读错记成一条 `(bucket, position, residual)`，其中
- `bucket` = 问句类（同一规则的不同措辞归一类；用现有的规则表归一，不引入新分类器）；
- `position` = 读错的决策位；
- `residual = read_value − target`，`r ∈ R^value_dim`。

对一个桶 `c`，定义：

| 量 | 公式 | 判读 |
|---|---|---|
| `n_c` | 桶内错读条数 | 样本够不够 |
| `err_c` | `n_c / N_c`（桶内错读率） | 是不是系统性问题 |
| `conc_c` | `‖Σ_{i∈c} r_i‖ / Σ_{i∈c}‖r_i‖`（残差方向一致度，`[0,1]`） | 是不是"同一个方向"错了 |
| `U_c` | 对 `{r_i}_{i∈c}` 做 SVD，取累计能量 ≥ 0.9 的前 `r` 个左奇异向量 | 该错读子空间 |
| `share_c` | `Σ_{j≤r} σ_j² / Σ_j σ_j²` | 子空间解释了多少 |

**准入（三条同时成立才允许局部改）**：

```
n_c >= n_min (= 4)            # 太少是噪声，不是热点
err_c >= e_min (= 0.50)       # 半数以上读错才算"反复"
conc_c >= 0.70                # 错法一致，才有一个"子空间"可改
```

**局部改法**（只改该子空间）：

```
Δ_local = U_c · (U_cᵀ · Δ_full)        # 只保留落在子空间内的分量
```

其中 `Δ_full` 是"若全局重训这一步会产生的参数增量"。**局部性判据**：

```
local_fraction = ‖Δ_local‖_F / ‖Δ_full‖_F >= 1 - ε_local,   ε_local = 0.10
```

即：至少 90% 的修正落在该子空间内，才允许按局部提交。不满足 ⇒ `hotspot_not_localizable`，**不改**，上交 L1（因为这意味着错读不是"一个方向错了"，而是整个坐标系不够用）。

**不变量**：

| ID | 不变量 |
|---|---|
| I-W11 | L3 不许改 `metric_version`；它只产出 `MetricDelta` 候选，提交权在 L1 |
| I-W12 | L3 的写入必须带 `(bucket, U_c, local_fraction)` 三项证据，缺一不得提交 |
| I-W13 | `local_fraction < 0.90` 时**必须**停止局部修，禁止"改一点点试试" |

### 2.4 L4 修剪层（prune）——合并共线方向、回收不用的槽

**管什么**：长期不用和长期重复的东西。修剪是生长可持续的必要条件（不修剪就只能单调膨胀）。

**触发条件**：

| 动作 | 触发 | 判据 |
|---|---|---|
| 合并共线方向 | 两方向 `|cos(u_a,u_b)| >= 0.98` **且**连续 `k = 3` 个会话中二者从未同时被 top-1 选中 | 合并为 `u_m = normalize(u_a + sign·u_b)` |
| 回收槽 | `hits == 0` 且 `age` 距当前 ≥ `T_idle = 3` 个会话 | 回收（置 `age = -1`） |

**"合并后不丢可分性"的判据**（这是修剪最容易出错的地方，必须可算）：

设合并前状态 `S`、合并后 `S'`，在**同一批参考问句** `B` 上：

```
G-P1  正确数不降:      A(S') >= A(S)                       # 整数，不设 epsilon
G-P2  余量不降:        median_margin(S') >= median_margin(S) - ε_m,   ε_m = 0.002
G-P3  有效秩不降:      eff_rank(S') >= eff_rank(S) - ε_r,            ε_r = 0
G-P4  单槽贡献可忽略:  c_s = Σ_{q∈B} (p_s(q) - p_s^{-}(q)) <= ε_c,   ε_c = 1e-3
```

其中 `p_s(q)` 是查询 `q` 在槽 `s` 上的 softmax 权重，`p_s^{-}` 是移除该槽后的重归一化权重。`c_s` 就是"这个槽对读出到底贡献了多少"——**它小到可以忽略才允许回收**，否则保留。

四条任一不满足 ⇒ 该次修剪**拒绝并回滚**。修剪的失败出口不写 `growth_retired`（修剪失败不等于容量假设失败）。

### 2.5 层间权限总表（谁能写谁）

|  | L1 度量 | L2 值 | L3 热点 | L4 修剪 |
|---|---|---|---|---|
| 改 L1 度量 | **只能经 `commit`** | ✗ 禁止 | ✗（只能提案） | ✗ |
| 改 L2 值 | 可以（度量变了要重编码） | **可写（本层本职）** | ✗ | 可回收 |
| 改参数 `R`/`P` | `birth` 由 L1 审批 | ✗ | ✗ | 只能合并已有方向 |
| 改 `metric_version` | ✗（不变） | ✗（不变） | ✗（不变） | ✗（不变） |
| 产生梯度 | 是（全局可训） | **否** | 是（在子空间内） | 否 |

---

## 三、把共识门做成真的校验门（本任务核心）

### 3.1 现在为什么等于常开

`growth_gates = {"consensus": 1.0}` 从头到尾没有任何比较动作：它是一个乘法常数（[living_core_batch.py L102](../../taiji/living_core_batch.py#L102) `values = [float(gates[branch]) ...]`），在 `parent >= R.shape[1]` 时连这个乘法都不执行（[L93-L95](../../taiji/living_core_batch.py#L93-L95)）。**它不读任何前后状态，因此不可能拒绝任何东西。**

### 3.2 提交协议：propose → shadow → verify → commit 或 revert

任何一层想改状态（写入一个值、提交一个度量增量、合并两个方向），都必须走同一个协议：

```
1.  S    = snapshot(state)                     # bytes，记 sha_old
2.  S'   = shadow_apply(S, write)              # copy-on-write；禁止驱逐、禁止原地改写
3.  cert = evaluate(S, S', battery)            # 纯函数，numpy，见 §3.3
4.  if cert.passed:
        publish(S');  metric_version += (仅 L1 commit)
    else:
        quarantine(write, cert);  restore(S)   # 逐位；sha 复核
```

三条硬约束：
- **shadow 阶段禁止驱逐**。满槽 ⇒ 直接 G3 失败（`rejected_no_capacity`），而不是"驱逐最老"；
- **verify 必须跑在 shadow 上**，不许先发布再检查（否则"回滚"就不是原子，而是补偿）；
- **判定函数是纯函数**：无随机、无时间、无全局状态，同样的 `(S, S', battery)` 必得同样 `cert`。

### 3.3 判据公式：四个条件，全部可算

记号：`A(S)` = 保留集正确数；`m(S)` = 参考问句的读余量中位数（`cos₁ − cos₂`）；`r(S; q, v')` = 值残差；`B(S)` = 状态字节数；`h(S)` = 可写余量比例。

```
G1  旧能力不降      A(S') >= A(S)                         # 整数比较，ε 严格为 0
G2  可分性不降      m(S') >= m(S) - ε_m      ε_m = 0.002
                    （原第三条 rank 子句已被 P1 证伪，见下方"P1 后更正"）
G3  状态不膨胀      B(S') - B(S) <= Δ_max    Δ_max = 33,280 B（一个槽）
                    h(S') >= h_min           h_min = 0.125
G4  定向收益成立    r(S'; q, v') <= 0.5
                    top1(S'; q) == slot_written
                    read(S'; q) 对旧值的残差 >= r_old - ε_r_old   ε_r_old = 0
```

四条必须**全绿**才 `passed`。

**P1 后更正（2026-11-08，读数已落盘，代码未改）**——两条原判据的推导有错，错误性质不同：

| 原写法 | 实测 | 错误性质 | 更正 |
|---|---|---|---|
| G2 第三条 `eff_rank(S') >= eff_rank(S) - ε_r`，`ε_r = 0`，理由"有效秩是整数" | `7.29079 → 6.75998`（写一条 `o = 0.963` 的复制键后必降） | **归类与理由错，行为恰好正确**：`eff_rank = (Σλ)²/Σλ²` 是实数参与率，**不是整数**；追加一条与既有键近乎共线的键必然让谱更集中、该值必然下降。它编码的其实是"每次写入都必须长一个新方向"（§4.2 生长带），而余量条款（`0.85714 → 0.85714`，没动）才是"可分性不降"。所以拒绝的行为是想要的，但**出口名与理由都错了** | 出口应从 `commit_rejected_separability` 改为独立的 `commit_rejected_no_direction`；理由改写为"有效秩是实数、追加近共线键必降，因此该子句等价于 §4.2 的方向带，不是可分性"。**未验证项**：它是否也会把"真新方向"拒掉——P2 已答（不会，真新条目被准入）。**P1b 已实施（2026-11-08）**：出口独立、只由方向带 `growth_is_copy` 触发、且只对声明方向的策略（`append`）生效；`eff_rank` 降为读数（`detail.rank_is_reading_only`），不再产生任何失败码 |
| G4 第三条 `read(S'; q) 对旧值的残差 >= r_old - ε_r_old`，`ε_r_old = 0`，本意"取代必须发生" | P1 的 `q` 是写入用的 self 查询，`w_new = 0.59731`，残差 `0.5742` | **公式缺失**：当只能追加、不能取代时，"取代是否发生"无法用一个与基线比较的式子表达（基线自己就是混合值）。可算的替代量是 `√2·(1 − w_new)`，`w_new` 是 softmax 在新槽上的权重：`w = 1` ⇒ `0`，`w = 0.5` ⇒ `0.707` | 用 `r(S';q,v') <= 0.5` **单条**即可（该式在 `w_new < 0.646` 时必然失败），第三条删除；`0.5` 不动 |

**阈值来源（不是拍脑袋）**：
- `Δ_max = 33,280 B` = 一槽的 `128 + 4·2·32·32 = 8,320` 个 float × 4 B（[生长设计 §3 代价表](SPEC-TAIJI-GROWTH_DESIGN_20261108.md)）；一次提交只允许长一个槽。
- `h_min = 0.125` 沿用生长设计 §5A 的容量先行阈值；
- `ε_m = 0.002`：F3 两代之间中位余量的最小变化量级是 0.04（0.0686→0.0584→0.0303→0.0200），采样噪声量级是 0.0043（F1 同义问句 margin 全为 0.0043）；取噪声的一半作为"不许变得更差"的公差；
- ~~`ε_r = 0`：有效秩是整数~~ **已撤回**：有效秩是实数，且在任何"追加近共线键"下必降，不能当门禁用。`ε_r` 仅保留在 §2.4 修剪层。

**G4 的 `0.5` 是怎么来的**：F1 实测"一半新一半旧"的残差是 `0.49·√2 = 0.693`；若纠正被完整取代，残差是 0；取 0.5 是为了把"混了不到一半旧值"与"混了一半以上"分开，同时它不是 0——因为实测存在浮点/温度带来的极小混合。**这个阈值随 F1 读数固定，不许事后改。**

### 3.4 谁来算，算多贵

- **谁算**：新增 `taiji/growth_gate.py` 的 `evaluate()`，纯 numpy、无 torch、不被任何冻结模块 import（与 `growth_design.py` 同一隔离口径，见[禁改清单](FROZEN-READSET_20261108.md)）。
- **算多贵**：`G1/G2/G4` 需要在 `battery` 上重跑 `addressed_read`，复杂度 `O(|B| · slots · d)`。以 `|B| = 65`、`slots = 128`、`d = 128` 计约 `1.06e6` 次乘加，纯 numpy 亚毫秒；`G3` 是 `O(1)`。**不需要任何前向、不需要梯度**。
- **一次提交的总代价**：一次快照拷贝（状态 2.13 MB 上限）＋一次矩阵乘 ＋ 四次整数/浮点比较。按 F1 的 0.14 s 量级，门本身不会成为瓶颈。

### 3.5 原子回滚：逐位恢复、不驱逐、不丢弃

| 要求 | 实现 |
|---|---|
| **逐位恢复** | 快照是 `store_keys/values/age/count` 四个数组的 `tobytes()`；回滚 = 反序列化回原 dtype/shape；判定 `np.array_equal` **且** `sha256` 相等。任一不等即 `rollback_corrupt`（这是工程错误，不是策略失败） |
| **不驱逐** | shadow 阶段不写任何已有槽；满槽就拒绝（G3） |
| **不丢弃** | 被拒写入进 `quarantine`（append-only，含 `commit_id`、四项判据的实测值、失败条件名）；容量或度量更新后可重新发起（**新 commit_id**，不是"重试同一个"） |
| **可审计** | 每次提交/拒绝都往 `PROGRESS` 追加一行：`{stage, commit_id, passed, g1..g4 实测值, sha_old, sha_new, evictions:0, dropped:0}` |

### 3.6 失败出口

| 失败条件 | verdict | 动作 |
|---|---|---|
| `G1` | `commit_rejected_retention` | 回滚；该写入不得进 L2；若来自 L3，整条热点提案作废 |
| `G2` | `commit_rejected_separability` | 回滚；写入进隔离队列；若来自 L1，度量**不 bump 版本** |
| `G3`（字节） | `commit_rejected_inflation` | 回滚；说明这次改动不是"一条修正"，须拆小 |
| `G3`（余量） | `commit_rejected_no_capacity` | 回滚；先扩容（生长设计 §5A 容量先行），扩容后再走一遍完整门 |
| `G4` | `commit_rejected_no_gain` | 回滚；**上交 L1**：要么改成"同键取代"，要么改编码 |
| 快照 sha 不等 | `rollback_corrupt` | 停机，不做任何继续写入（工程错误） |

### 3.7 同一案例：门常开 vs 门生效（用 F1 的实测数）

夹具：[F1 纠正分支](../../reports/taiji_growth_f1_correction_20261108.json)，低占用档（8/16 槽），同一纠正 `(q, v')`。

| 项 | 门常开（现状 `consensus=1.0` + 未接线） | 门生效（§3.3 四条件） |
|---|---|---|
| 写入槽 | 新槽 8（不覆盖旧槽） | shadow 里同键取代（I-W9） |
| top-1 命中 `corr@1` | **1.00** | 1.00（取代后仍命中） |
| 读回值 | `0.51·v' + 0.49·v_old` | 取代后应为 `v'`（这是门要求的，不是门测出来的） |
| 值残差 `r` | **0.698** | 门**先拒绝**：0.698 > 0.5（若不取代）；取代后 `r ≈ 0`，准入 |
| G1 旧能力 | 不计算 | 对照 `old@1 = 1.00`，写入后仍 1.00 → 通过 |
| G2 余量 | 不计算 | `0.06863 → ?`，预注册门槛 `≥ 0.06663`，见 §5 P3 |
| G3 字节 | 不计算 | `+33,280 B`（恰好一槽，通过） |
| G4 | 不计算 | **失败**（0.698 > 0.5） |
| 结果 | 静默混合；`growth_is_copy` 只是**标签**，没有任何动作 | `commit_rejected_no_gain` + 逐位回滚；store sha 不变；写入进隔离队列 |
| 后续 | 读错被记为读对（假阳性） | 上交 L1：改为"同键取代"或加宽编码 |

**两种模式的差别不在"能不能读到"，而在"读错了会不会被记成读对"**：常开门下，`corr@1 = 1.00` 会被当成成功；生效门下，同一个数被 G4 拆穿——命中是真的，值不对也是真的。这正是 owner 说的"局部扩张没有裁决者"。

---

## 四、重定义生长对象：长"方向"，不长"格子"

### 4.1 定义

| 名词 | 定义 | 存在形式 |
|---|---|---|
| **格子**（cell） | 状态里的一个存放位置：`store_keys` 一行 + `store_values` 一槽，33,280 B | 状态（session，不进身份） |
| **方向**（direction） | 共享度量层里的一根**可复用**基向量：新的键维（`key_width` 的一列）、`birth()` 的 `R` 行/`P` 列、或键空间中新出现的一个主方向 | 参数（进身份）或键谱的正交分量 |

判据的核心是**复用**：方向之所以叫方向，是因为它能同时服务多条问句。只能服务自己那条的，不是方向，只是"这条记录自己占了一个格子"。

### 4.2 判据：区分"长了方向"与"只长了格子"

对已有方向张成的子空间 `U`（单位化），新方向 `u`：

```
o(u) = ‖Uᵀu‖₂                          # 重叠度
s(u) = 1 - max_i |cos(u, u_i)|         # 可分辨性
reuse(u) = 该方向被 top-1 选中的**不同问句类**数
```

| 判定 | 条件 | 含义 |
|---|---|---|
| `growth_new_direction` | `0.15 ≤ o ≤ 0.85` 且 `s ≥ 0.30` 且 `reuse(u) ≥ 2`（`Δeff_rank` 只作读数，见下） | 真的长了方向 |
| `growth_is_copy` | `o > 0.85` 或 `s < 0.30` | 只长了格子（复制） |
| `growth_isolated` | `o < 0.15` | 与旧子空间无关，迁移被切断，也不算方向 |
| `growth_cell_only` | `o/s` 在带内但 `Δeff_rank = 0` 或 `reuse(u) < 2` | 位置多了，方向没多 |

`reuse ≥ 2` 是**新加的**，也是本节与旧版生长设计（§4A 只看 `o/s`）的唯一差别：旧的带内判据只保证"不复制不孤立"，不保证"能被复用"。

**`Δeff_rank ≥ 1` 的量级有误（P2 实测；P1b 已裁定）**：这个 `1` 是从 F3 的**四次翻倍**整体读数 `7.29 → 40.18` 反推出来的，不是单次写入的量级。P2 实测单次追加一条真新条目只涨 **`+0.862`**（`7.29079 → 8.15326`），方向带与四条件全绿。**P1b 裁定：单次写入不设 `Δeff_rank` 门槛，只作读数**（`detail.rank_is_reading_only`；报告中另记一条"`Δeff_rank > 0`"的观察，不作门槛）。理由：单臂 n=1 无分布可标定下限，且该量与方向带同向（近共线必降），作为门槛只能给同一件事加第二个拒绝理由，不增加信息。`≥ 1` 留给"一次扩容"（F3 的四次翻倍），两者不得混用。

### 4.3 实证：F3 四代扩容与 F1 单条纠正

**同一扩容序列（`8→16→32→64→128`），换编码才是分水岭**（[F3 报告](../../reports/taiji_growth_f3_multigen_20261108.json)、账本 D16-57）：

| 口径 | 四代读数 | 判定 |
|---|---|---|
| 旧编码（`mean`） | 每代 `o ≈ 0.999`、`s ≈ 0.001` | 全部 `growth_is_copy`——**格子翻四倍，键谱不动** |
| 新编码（`localized_centered`） | 新增键 `o = 0.192→0.622`、`s_min = 0.710`；`eff_rank` **7.29→40.18**、`top_share` **0.1931→0.0568** | 方向真的长出来了 |

**这三行是整个 §4 的实证依据**：同样的格子数、同样的写入条数，旧编码下"只长格子"，新编码下"长了方向"。所以生长对象的判据不能只看"槽有没有变多"，必须看**键谱有没有变宽**。

**F1 是反面的单点证据**：只写一条纠正，槽数 +1，但 `o = 0.963/0.978 ≥ 0.85` → `growth_is_copy`；读余量掉 3.4× 而混合权重只动 0.002 → 加格子对可分性毫无贡献。

### 4.4 与四层的关系

- **方向属于 L1**（度量层）：它必须全局唯一、可被所有人复用；
- **格子属于 L2/L4**（值层/修剪层）：它只是承载，随方向扩展而扩展；
- **生长顺序**：先有方向（L1 准入），再扩格子（L2 写入）。反过来做（先扩格子）就是 F3 旧编码那四代——四倍格子、零方向。

---

## 五、三条可检验预测（P1/P2/P1b/P3 已跑）

三条全部可以在**现有离线夹具**上跑（F1 的合成夹具 + F3 的四代序列），0 训练、0 前向、0 Adam。判据先落盘，跑完不改。

### P1 旧编码 + 门生效 ⇒ 这条纠正必须被判"复制、不准入"

- **夹具**：F1 合成夹具，低占用档（8/16），写一条 `(q, v')`。
- **预测**：`growth_band == "growth_is_copy"`（`o = 0.963 ≥ 0.85`，`s = 0.0395 < 0.30`）；`G4` 失败（`r = 0.698 > 0.5`）；`cert.passed == false`；`verdict == "commit_rejected_no_gain"`。
- **预注册漂移（P2 时发现，如实记录，不改 P1 判决）**：本节原写 `mode = "mean"`，但 P1 runner 实际用的是 `localized_centered` + 中心化（P1 报告 `fixture.encoder = localized_centered-v1`）。每一格读数在它自己那行都成立，所以漂移不改变 P1 的任何判据：C0/C3 与编码无关；C1 的 `o ≥ 0.85` 两种编码都过（0.9631 / 1.0000）；C2/C4 是"混合读"的性质，与编码无关。**真正被这条漂移削弱的是 P2**：P1 早就跑过新编码，"旧编码拒 / 新编码准"的对照在 P1↔P2 之间根本不存在，只能在本轮四臂内部重建（见下）。
- **附带必须成立**：`sha_after == sha_before`（逐位回滚）、`evictions == 0`、`dropped == 0`、`quarantine_rows == 1`。
- **判负条件**：若门判 `passed == true`，或回滚后 sha 不等，或发生任何驱逐，P1 判负——说明门没接线。

**已跑（2026-11-08，[读数](../../reports/taiji_growth_gate_p1_20261108.json)）**：

| 项 | 预注册 | 实测 | 判 |
|---|---|---|---|
| C0 保留集可答 | ≥ 6/7 | 7/7 | 过 |
| C1 `growth_is_copy` | `o ≥ 0.85` | `o = 0.9631`、`s = 0.0395` | 过 |
| C2 出口＝`no_gain` | 恰好 `commit_rejected_no_gain` | `commit_rejected_separability` | **负**（见下） |
| C3 拒绝彻底 | 未发布／sha 相等／`evictions 0`／`quarantine 1` | 全成立，`sha 44b25129…` 前后同值 | 过 |
| C4 残差≈0.6930 | ±0.05 | `0.5742` | **负**（见下） |

**结论（一句话）**：门**第一次真的拒绝了**——写入没有发布、store 逐位不变、0 驱逐、被拒写入进隔离队列 `q-0001`；两条判负都不是"门没接线"：

- **C2**：`eff_rank 7.29079 → 6.75998` 先于 G4 触发，`failed = [separability, no_gain]` 两条都真失败。原因是**归类与理由错、行为恰好正确**（见 §3.3 "P1 后更正"）：这条子句实质是"必须长新方向"，不是"可分性不降"（余量 `0.85714 → 0.85714` 完全没动）。
- **C4**：`0.6930` 是 F1 四个**同义**问句的中位预测；P1 的 G4 读的是**写它时用的 self 查询**，`w_new = 0.59731`，该查询的半新半旧预测是 `√2·(1−0.59731) = 0.5695`，实测 `0.5742`，差 `0.0047`。**G4 拦的是对的东西。**

**纪律记录**：未改任何阈值、未改 `taiji/growth_gate.py`、未重跑冒充通过；两项更正以增量块写进报告 `post_reading_corrections`，原判据与 `decision` 原样保留。

### P2 新编码 + 门生效 ⇒ 同一纠正应被准入，且有效秩上升

- **夹具**：同上，`mode = "localized_centered"` + 中心化。
- **预测**：`o ∈ [0.15, 0.85]`、`s ≥ 0.30`；`r ≤ 0.5`；四条件全绿；`eff_rank(S') − eff_rank(S) ≥ 1`（量级参照 F3 的 7.29→40.18）；`reuse(u) ≥ 2`（同一规则的两种措辞都被该方向 top-1 命中）。
- **判负条件**：若准入但 `Δeff_rank = 0`，判"只长了格子"（`growth_cell_only`）——这说明编码修复没到这个夹具上，不许用"接近"措辞。

**已跑（2026-11-08，[读数](../../reports/taiji_growth_gate_p2_20261108.json)，四臂并排、同夹具、同一条 `(q, v')`、0 训练）**：

| 臂 | 编码 | 策略 | 写入的是什么 | `o` | 方向带 | `published` | 出口 |
|---|---|---|---|---|---|---|---|
| primary | `localized_centered-v1` | append | 同一条纠正 | 0.963058 | `growth_is_copy` | **false** | `commit_rejected_separability` |
| legacy | `mean-v1` | append | 同一条纠正 | 1.000000 | `growth_is_copy` | **false** | `commit_rejected_separability` |
| replace | `localized_centered-v1` | replace | 同一条纠正（同键取代） | 0.963058 | `growth_is_copy` | **false** | `commit_rejected_separability` |
| new_direction | `localized_centered-v1` | append | **一条真正的新条目** | 0.283081 | `growth_new_direction` | **true** | `commit_accepted` |

四臂的四条件全项（不只报第一个失败）：

| 臂 | G1 保留数 | G2 中位余量 | G2 有效秩 | G3 字节 | G3 余量 | G4 残差 / top1 |
|---|---|---|---|---|---|---|
| primary | 7→7 ✅ | 0.85714→0.85714 ✅ | 7.29079→6.75998 ❌ | +33,280 ✅ | 0.4375 ✅ | 0.5742 / 8 ❌（>0.5） |
| legacy | 7→7 ✅ | 0.03632→0.03632 ✅ | 1.00332→1.00330 ❌ | +33,280 ✅ | 0.4375 ✅ | 0.9434 / 0 ❌ |
| replace | 7→7 ✅ | 0.85714→0.85714 ✅ | 7.29079→7.25828 ❌ | **+0** ✅ | 0.5 ✅ | **0.000207** / 0 ❌（`top1≠slot_written`） |
| new_direction | 7→7 ✅ | 全绿 ✅ | 7.29079→**8.15326** ✅ | +33,280 ✅ | 0.4375 ✅ | 0.000633 / 8 ✅ |

**结论（一句话）**：P2 预注册的**主臂预测被证伪**，但证伪的是**前提**，不是门——这条写在本编码下 `o = 0.963` 本身就是复制（自检 S2），门拒它是对的；**门确实会准**：同编码下写一条真新条目 `o = 0.2831`、`commit_accepted`、`Δeff_rank = +0.862`、G4 残差 0.000633、写入槽权重 0.99943、8 个旧槽全保留（残差 < 0.001）。

四条必须单独说的读数：

1. **"旧编码拒 / 新编码准"的并排对照不存在**（自检 S3）：P1 报告 `fixture.encoder` 已经是 `localized_centered-v1`，两个臂都拒。本轮重建的对照是"同一条纠正 vs 真新条目"，不是"旧编码 vs 新编码"。
2. **编码质量确实不同，但差别不在"准不准入"上**：`mean` 臂读不干净（G4 残差 0.9434、`top1 = 0`），`localized_centered` 臂读到了 `top1 = 8`、残差 0.5742——同为拒绝，读的行为完全不同。
3. **最关键的负结果在 `replace` 臂**：同键取代把残差从 0.5742 压到 **0.000207**、字节增量 **0**、G1 与余量全不变，唯一挡它的是 **G2 的 rank 子句**（7.29079→7.25828）。也就是说 F1 指出的"缺取代"已经在门内可用，卡它的是那条 **已被 P1 证伪** 的判据。**唯一未解的是 `top1≠slot_written`**（取代后旧槽与新槽并存于同一位置，`top1` 落在哪个槽需在 P1b 里一并定义）。
4. **单次追加的方向增益实测 +0.862**，§4.2 的 `Δeff_rank ≥ 1` 对单次写入过严：该阈值源自 F3 四次翻倍的整体 7.29→40.18，不是单次写入的量级。P1b 须按单次追加重推（候选：`Δeff_rank > 0` 且 `o` 落带）。

**纪律记录**：未改任何阈值、未改 `taiji/growth_gate.py` 一行、未改 §5 判据、未重跑冒充通过；判负即返回退出码 1，`decision = p2_falsified_premise_C1_C2_C3_C5` 原样落盘。

### P1b 拆开闸口：复制被拒、取代被准、新条目被准（2026-11-08，[读数](../../reports/taiji_growth_gate_p1b_20261108.json)，`decision = p1b_passed`）

预注册三件（跑前写入报告 `criteria_written_before_reading`，跑后一字未改）：**R1** 出口与触发解耦——`commit_rejected_no_direction` 独立成码，只由方向带 `growth_is_copy` 触发；G2 只留"读余量不降"（`0.002` 不变）。**R2** 单次写入不设 `Δeff_rank` 门槛，仅作读数。**R3** 两种命中口径同时列出、各自判定——`H-slot`（`top1 == written_slot`）与 `H-value`（`residual ≤ 0.5`）；门的通过与否按 **H-value**，`H-slot` 作读数。**范围裁定（也写在跑前）**：方向是 `append` 的主张，`replace` 的主张是"这个地址现在返回 `v'`"，故方向子句对 `replace` 豁免；另一种读法（对所有策略生效）作为"替代读数"一并报出，不事后挑选。

| 臂 | 策略 | `o` | 带 | 准入 | 出口 | H-slot | H-value | 值残差 |
|---|---|---|---|---|---|---|---|---|
| 复制 | append 同键 | 0.9631 | `growth_is_copy` | **false** | `commit_rejected_no_direction` | true | false | 0.5742 |
| 取代 | replace 同键 | 0.9631 | `growth_is_copy` | **true** | `commit_accepted` | true | true | **0.000207** |
| 新条目 | append 新键 | 0.2831 | `growth_new_direction` | **true** | `commit_accepted` | true | true | 0.000633 |

取代臂另报：字节增量 **0**、G1 `7→7`、中位余量不变、旧 7 槽 top-1 与残差全保持、被取代槽读回新值（残差 0.000207）。新条目臂：rank `7.29079 → 8.15326`（仅读数）、写入槽权重 0.99943。

**同时修掉一处旧缺陷**：`evaluate` 曾把 `written_slot` 取成 `top1`，使 G4 的"`top1 != written_slot`"永远为假——命中判定是恒真的。现在 `run_gate` 从 `apply_receipt["slot"]` 传入真实落点，G4 只按值残差判，槽编号作读数。

**如实留档的实现偏差**：首跑两处实现与预注册不一致（方向子句排在 G4 之后，导致复制臂出口成了 `no_gain`；H-value 测在回滚态而非被判决的 shadow 态），两处都已写进报告 `implementation_corrections`（`criteria_changed=false`、`thresholds_changed=false`），首跑读数完整保留在 `previous_run`。

### P3 门生效后，读余量不应跌破阈值；且高占用档应主动拒绝

- **夹具**：F1 两档（8/16 与 64/128）。
- **预测 A（低档）**：`m(S') ≥ 0.06863 − 0.002 = 0.06663`；写入被 G4 拒绝时，`m` 完全不变（因为没发布）。
- **预测 B（第 4 代）**：该档基线 `m = 0.01999`，**已经贴在塌缩线 0.02 上**（生长设计 §1 S-B 的判据是 `median_margin ≤ 0.02`）。因此任何进一步降低余量的写入都必须被 `commit_rejected_separability` 拒绝；若某次写入使 `m` 下降 ≥ 0.002，预测该次必被拒。
- **判负条件**：若第 4 代档下 `m` 下降 ≥ 0.002 而门仍 `passed`，P3 判负；若第 4 代档下写入完全不降 `m`（例如被 G4 先拒），则 P3-B 记为"未触发"而不是"通过"——**不许把未触发算成通过**。

三条预测共同检验一件事：**门是不是真的会拒绝**。如果三条都"通过"，也可能只是门从没拒绝过——所以每条都带"必须被拒一次"的正面要求（P1 必拒、P3-B 应拒）。

### P3 读数（2026-11-08，[报告](../../reports/taiji_growth_gate_p3_20261108.json)，`decision = p3_falsified_A1`）

**先分清两个"余量"**：本文 §5 的 `0.06863` 是 `m_spec`（占用键 Gram 的最近−次近中位，即 `growth_design.read_margin_stats`）；门里 G2 实际比较的是 `m_gate`（保留集查询的 `1−最近` 中位）。同一个 8 键状态，前者 0.0686、后者 0.857——**拿 0.857 去比 0.06663 会"因错误理由通过"**，所以 A 判在 `m_spec` 上。

| 档 | 基线 `m_spec` | 六/一次写入后 | `m_gate` | 结论 |
|---|---|---|---|---|
| 8/16 | 0.068631 | 0.052401 → 0.047397 → 0.042394 → 0.038957 → 0.033074 → 0.034297 | 恒 0.857136 | **A 判负**：第一次就掉 −0.01623（8×`ε_m`），远低于 0.06663 |
| 64/128 | 0.019988 | 被拒（`commit_rejected_no_direction`，`o=0.983151`） | 0.774064 | **B 未触发**：强制写只让 `m_spec` +0.000757 < 0.002，无预测可验 |
| 112/128 | — | 被拒（`commit_rejected_no_capacity`，`h=0.117188<0.125`） | 不变 | 容量子句只在算术真不允许时才生效 |

纪律项全绿：`evictions=0`、`dropped_writes=0`、被拒各 1 行隔离、逐位不变、8 条"若强制写会怎样"只算不写。**判负原因是架构级而非实现 bug**：G2 用的相对阈 `0.002` 落在 0.857 的量级上，几乎不可能被触发，而 §5 关心的拥挤度体现在 `m_spec` 上（一次写入就掉 8 倍容差）。指令要求的"64/128 必须 `no_capacity`"与该档 `h=0.5` 的算术冲突，已如实记为"未证实"，并由 112/128 臂单独验证容量子句。


---

## 六、与现有规格的衔接

| 现有件 | 关系 | 冲突处理 |
|---|---|---|
| [生长设计 §4A](SPEC-TAIJI-GROWTH_DESIGN_20261108.md) 相关带 `o∈[0.15,0.85]`、`s≥0.30` | 本文 §4.2 沿用，并**增加** `reuse ≥ 2` 与 `Δeff_rank ≥ 1` | 不冲突，是加强；旧判据单独使用会把"带内但不可复用"误判为方向 |
| 生长设计 §5A 容量先行 `h ≥ 0.125` | 本文 §3.3 的 G3 直接复用 | 不冲突；本文把"先扩容再写"收进门的失败出口 `rejected_no_capacity` |
| 不变量 I-G1…I-G8 | 管"生长本身" | 本文新增 I-W1…I-W13，管"谁能改谁"；两套不互相代言 |
| [F1 两处规格更正](SPEC-TAIJI-GROWTH_DESIGN_20261108.md) | §11 F1"写新槽、不覆盖旧槽"已证伪；`correction@1 ≥ 0.90` 不足以判对 | 本文 §2.2 的 I-W9（同键取代）与 §3.3 的 G4（值残差 ≤ 0.5）就是这两处更正的落地 |
| 冻结清单 25 个文件 | `living_core.py::addressed_write` 在冻结内 | 本文不要求改它：门做成 wrapper（§3.2），冻结文件逐位不变 |

---

## 七、未决点与唯一建议

**唯一需要 owner 裁定的选择**（其它都已经写成判据，不需要裁定）：

> **要不要为了"同键取代"去解锁冻结文件 `taiji/living_core.py`？**

两个选项的代价与上限：

| 选项 | 做法 | 上限 | 代价 |
|---|---|---|---|
| A（推荐） | 先在新模块 `taiji/growth_gate.py` 里做 wrapper：shadow 中执行"同键取代"，门通过后才回写。冻结文件 0 改动 | 能立刻把门跑起来、拿到 P1/P2/P3 的真实读数；但**训练图仍走旧写入**，所以门的结论暂时只对离线夹具成立 | 需要维护两条写入路径（wrapper / 冻结原版），并且回写时必须保证逐位一致 |
| B | 解锁 `living_core.py`，直接让 `addressed_write` 支持同键取代 | 上限更高：训练图与门走同一条路径，结论可以直接外推到训练 | 动冻结文件，需要重跑 25 文件 readset 与身份校验，正在跑的训练窗读到的代码会变 |

**建议：选 A**。理由只有一条但足够：门现在的核心问题是"从没拒绝过任何东西"，A 能用最小代价让门第一次真的拒绝（P1 已证明它能拒），而 B 在门还没被验证过的时候就先改被检验对象——那是把两个变量一起动，读数会失去归因。

**P1 之后新增的未决点（P1b 已裁定）**：G2 的 rank 子句改出口名——**已改**为 `commit_rejected_no_direction`，且只由方向带触发，不再由 rank 数值触发；P1b 预注册与读数见 §5。

**P2 之后新增的未决点（P1b 已裁定，并含一处更正）**：P2 报告里写的"`replace` 臂 `top1` 落在旧槽而非 `slot_written`"**不准确**——真实情况是 `evaluate` 把 `written_slot` 取成了 `top1`，命中判定恒真；实际读数 `top1 = 0` 与落点槽 0 一致。P1b 已改为由 `apply_receipt["slot"]` 传入落点，并裁定命中按 **H-value（值残差 ≤0.5）** 判、槽编号只作读数（§5 P1b）。

**P3 之后新增的未决点（G2b 已试）**：G2 的判据量选错了——`m_gate`（`1−最近`，≈0.857）在 8/16 上六次写入纹丝不动，而同一批写入把 `m_spec` 从 0.0686 打到 0.0331（首次即 −0.01623，是 `ε_m` 的 8 倍）。G2b 探针把判据量换成 `m_spec` 的**绝对下限**试了一遍（见下），结论是：**绝对下限能拒、但和占用强耦合**——`m_spec` 随占用自然衰减（8 键 0.0686 → 64 键 0.0200），所以同一个 0.02 在低占用等于"不设防"、在高占用等于"全拒"。

**G2b 读数（2026-11-08，[报告](../../reports/taiji_gate_g2b_20261108.json)，`decision = g2b_usable:F1_collapse_line,F2_half_baseline`）**：只改 G2 的判据量，其余条款与 `commit` 原样。P1b 三条遗判**全不翻转**：replace 准入（残差 0.000171）、真新条目准入（0.000639、`o=0.283`）、复制被拒。三个候选下限：**0.06663（把 §5 的 0.06863−0.002 当绝对下限）被自己的读数否掉**——它把 0.052401 的真新条目拒了；**0.02 可用但 8/16 六次写入全程不触发**（最低 0.033074）、64/128 出口不变（仍 `no_direction`）；**0.03432（基线一半）可用**，8/16 第 5 次写入以 `separability` 拒写、64/128 出口变 `separability`；112/128 在三个下限下都仍 `no_capacity`。**不做二选一**：两者的差别正是"低占用要不要设防"，而这取决于占用，不取决于拥挤。

**下一步（唯一）**：**先测"参考曲线"，再定下限形状（G2c 预注册，0 训练）**——同编码器、同一夹具，在 8/16/32/64/112 五个占用档上**只读不写**测 `m_spec` 随占用的衰减曲线，用这条曲线把下限写成"占用归一化"的形式（例如不得低于同占用参考值的一半），替代现在的固定值。理由：G2b 已经证明"门换量之后仍不误杀、也仍会拒"，唯一没解决的是**下限与占用耦合**——固定值必然在某些档位等于"不设防"、在另一些档位等于"全拒"，这不是阈值松紧问题，是判据形状问题；曲线测出来之前挑 0.02 还是 0.03432 都只是选一个档位特例。G2c 之后再谈把门接上真实 `addressed_write`。
