# FEC-1 快权重 F 可表达性 · 零训练判据冻结规格（2026-11-08）

> 状态 [预注册已冻结·待 owner 裁定执行]。owner 2026-11-08 批准把[包 21 全局收束卡 §5](CLOSEOUT-PACKAGE21_GLOBAL_20261107.md) 的下一轮唯一核心问题立为零训练判据专项。本件只冻结判据，**0 训练 / 0 前向 / 0 checkpoint 读取 / 0 confirm**。判据的唯一机读源是 [FEC-1 预注册 manifest](../manifests/taiji_f_expressive_capacity_prereg_20261108.json) 的 `criteria` 对象，`criteria_sha256 = 1e0dde222d557b4ea97eee528b9f73a2aec3068a2c462e927465e415fa3b6142`（对 canonical JSON 的 SHA-256，自洽可重算）；本件**不复述另一套可抄的门值**，只写口径理由、已知先验、执行契约与出口语义。判据先落盘、读任何数之后不得改。
>
> 上位：[包 21 全局收束卡 §5](CLOSEOUT-PACKAGE21_GLOBAL_20261107.md)、[详细开发计划 P1–P2](PLAN-DEVELOPMENT_DETAIL_20261107.md)、[ROADMAP-NEXT §0](ROADMAP-NEXT_20261104.md)、账本 D16-26 / I22.2 / D16-12；同级先例：[21.3 预注册](SPEC-PACKAGE21_3_REPRESENTATION_SEPARABILITY_PREREG_20261107.md)。

## 1. 准入与边界

- **准入依据**：包 21 收束卡 §5 要求「必须先以零训练的可表达性判据裁决（是／否），未裁决前不得再以训练剂量立项」。owner 已裁定**立此专项**，本件即该裁定的判据落盘。
- **本专项回答的唯一问题**：在保持 Task1 结构父身份（C12，candidate `1d69e945…`）与当前图身份（20.4c_r2 `build_graph(candidate, gate, arm)`，`addressed_store_enabled=False`）**都不变**的前提下，快权重 `F ∈ R^{4×2×32×32}`（8,192 元素）是否具备「按问句／按反馈事件**可分**且**可寻址**」的**表达能力**。
- **"表达"的定义**：存在一个**无需训练**的写入-读出算子对，能把 N 个不同事件映射到 N 个可区分、可分别取回的值。这是**函数类**问题，不是"某个 checkpoint 训出来没有"的问题。
- **硬边界**：0 训练 / 0 优化器步 / 0 Adam / 0 新参数 / 0 写 checkpoint / 0 改图 / 0 定价 / 0 confirm；confirm 索引 240–319 继续封存、0 读取；不新开包号；不重开包 20；不加剂量分支。唯一允许的干预是**写入许可掩码** `allowed`（运行参数，非参数、非结构、非超参）。
- **不是 21.3 的重跑**（四点实质差异）：① 主结论在**强制每轮写**下取，不在实际稀疏写下取；② 判**写入键自身的可分性**（`write_heads` 的 key 段），21.3 未测；③ 判**写键与读键的子空间对齐**（两套独立 MLP，无 weight tying），21.3 未测；④ 增加**随机初始化对照**以分离"架构固有"与"训练压塌"。

## 2. 已知先验（不因本专项改写）

| 来源 | 读数 | 对本专项的含义 |
|---|---|---|
| 20.2s | `distinct_f_vectors=2`；**`session1=1`、`session2=1`**；`f_elements=8192`；`f_norm 0.9499…1.2915`；`f_written_lines=10`（共 51 轮） | 每个 session 内部 F **逐位恒定**；写入覆盖率 **10/51＝0.196**；F 范数远低于 `fast_limit=16`，范数裁剪从未生效 |
| 20.1 夹具（本轮只读构造，不跑图） | 22 纠正面／world，session 1 有 **5** 面、session 2 有 **17** 面；**22 个不同问句文本**；**仅 11 个不同 bits 组合** | 「按问句」的可判上界是 11 个语义独立内容，不是 22；文本级可分 ≠ 内容级可分 |
| 20.3a | 问句键 `median_margin=0.003174`、`fraction_below_0.05=1.0`；`option2_failed` | 退化的**是通路 B 的 encoder 均值键**（128 维），不是通路 A 的 `write_heads` 键 |
| 20.3b | oracle F 替换 **0/22、0 字节变化** | F 的权威是次级缺陷 |
| 21.3 | 主探针 0.6364、窗口身份 **0.6818**、绑定 margin **−0.0455**；秩 2、`distinct=2`、4 checkpoint 逐位相同 | 在**实际写入序列＋通路 B 键**口径下不可分；范围限定成立 |

### 2.1 一条被忽略的算术必然性（本件的核心观察）

若 F 在某 session 内逐位恒定，则**任何只以 F 为输入**的解码器在该 session 内输出恒定，它对"本 session 内不同纠正面"的判别力**恒等于**"只按 session 分"的基线。因此 20.2s 的 `primary=0.6818` 与 `window_identity=0.6818` **完全相等不是巧合，而是该前提的算术后果**；21.3 的 `0.6364` 对 `0.6818` 亦同源。

**推论（必须写进结论模板）**：20.2s 的 `inconclusive` 与 21.3 的 `representation_axis_closed_for_this_graph` **都不是「F 里没有可分信息」的独立证据**，它们证明的是「**在这个写入序列下，F 的可见状态没有 session 内变化**」。前者仍是有效读数（判据问的就是当前序列），但**不能被引用为函数类判负**。

**未被检验过的问题因此是**：若让 F 每轮都写（只改 `allowed`，不改任何参数或结构），写键集合 `{k_i}` 是否张成足够的维数、读键 `{q_i}` 是否落在同一子空间、闭式读出是否可分离？本专项就是补这一格。

### 2.2 判据的定理依据（为什么 tied-key 读出是上界探针）

忽略衰减与擦除，写入为 `M = Σ_i k_i v_iᵀ = V Kᵀ`。则 `M K = V Kᵀ K = V G`，其中 `G = KᵀK`。故

> `V = M K G⁻¹` 精确恢复 **⟺** `G` 可逆 **⟺** `rank(K) = N`。

`rank(K) < N` 时，`M K` 的列空间维数 ≤ `rank(K) < N`，**不存在任何线性解码器**（含训练得到的）能精确恢复全部 N 个值。这是线性代数，与训练剂量、与读出方法优劣均无关。因此「用写入键本身当查询键」不是一种弱探针，而是**该函数类的最优可达上界**；它都读不出来时，任何独立学出的查询只会更差。

反向不成立：`rank(K) = N` 只保证**存在**解码器，不保证该图能学出它。故本判据的 PASS 只产出「架构需求输入」，不产出能力声称（与 21.3 的成功语义同构）。

## 3. 必须在读数前钉死的口径（本件核心）

### 3.1 双通路不可互认

图里有两条寻址通路，**键的函数不同、维度不同、默认状态不同**：

| 通路 | 写入键 | 读出键 | 维度 | 默认 |
|---|---|---|---|---|
| **A（fast）** | `write_heads[g]` 的 key 段，输入 `[normalize(h_g), event, goal, ν, past_error, missing, mode]` | `fast_queries[g*H+h]`，输入 `[normalize(h_g), event, goal, ν]` | 32 | **开** |
| **B（store）** | `encoder.context(query_bytes).mean(0)` | 同一函数（自洽） | 128 | **关** |

**裁定**：本专项主判据只判**通路 A**（因为「当前图」的定义就是 `addressed_store_enabled=False`）。通路 B 的 20.3a 读数只作历史锚，**不得**用来给通路 A 判负，也**不得**用通路 A 的通过去追认通路 B。21.3 的闭式读出用通路 B 的键去读通路 A 的产物，是**跨通路配对**，其结论只在那个口径内有效——本件不据此翻案，只登记口径差异。

### 3.2 强制写与实际写分层判定（防两个方向的偷换）

- **结构性 verdict 只由强制写读数产生**。实际稀疏写（10/51）的读数单列为具名 verdict。
- **禁止方向一**：用"实际没写"去替函数类开脱（"只是没写而已"）——除非强制写读数真的通过。
- **禁止方向二**：用"强制写通过"去追认当前图已具备能力——时机未修时它仍然不产出能力。
- 覆盖率低于 `gates.timing_coverage_min` 时先标 `timing_insufficient`，**并强制进入强制写回放**；不得据实际写入读数直接判结构。

### 3.3 问句轴与内容轴分离

22 个纠正面有 **22 个不同问句文本**，但只有 **11 个不同 bits 组合**（同一 name／order／style 模板下的值替换）。因此：

- 「按问句可分」在**文本级**（N=22）与**内容级**（N=11）分别算、分别报。
- **文本级通过而内容级不通过，判 `text_level_only`，不满足问句轴**。理由：文本差异可由模板槽位字符驱动，而任务要求的是按内容取回值。
- 若某 checkpoint 的 22 面中有任一面不可评估（缺输入、写入异常），该 checkpoint 作废并报 `inconclusive`，不得降 N 重算。

### 3.4 键范数下界是硬失败，不是退化

`unit_ball(x) = x/√(1+‖x‖²)` 是收缩映射而非球面投影。若 MLP 输出接近 0，则 `k_i ≈ 0`，外积 `k vᵀ` 消失、读出恒 0，而 `G` 的奇异值全部接近 0 会让"归一化后"的秩看起来健康。因此必须同时报 `key_norm_median`，低于 `gates.key_norm_median_min` 直接判失败，**不允许**用归一化后的余弦矩阵掩盖。

### 3.5 有效自由度闸门与聚合口径

沿用 20.3a / 21.3 的结构闸：强制写下 `distinct_f_vectors` 低于 `gates.effective_dof_min` 时任何 PASS 无效、自动落 `inconclusive`，配对 bootstrap 以**不同 F 向量**为重采样单位。4 个 checkpoint 各自独立出全套读数，主结论取 **margin 最小者**，不一致即 `inconclusive`；**不得挑表现最好的 checkpoint 报通过**。

### 3.6 随机初始化对照的用途边界

同一图、同一 seed（20261031）、零训练，重放同一序列。用途只有一个：区分

- 随机初始化也塌缩 ⇒ **架构固有**（π 的输入里 h 变化不足，或 `unit_ball` 收缩过强）；
- 随机初始化可分而训练后塌缩 ⇒ **训练压塌**（指向训练目标/正则，而非架构）。

该对照**不构成**通过/失败门槛，只作归因分账，登记为 `attribution` 字段。**不得**因随机初始化通过而宣称"架构没问题"。

## 4. 输入与执行契约

- **图与 checkpoint**：复用 `scripts/training/probe_taiji_package20_4c_r2_formal_comparison.py` 的 `build_graph(candidate, gate, arm)`（已入库 sha `112c76ae…`），`addressed_store_enabled=False`；4 个 checkpoint 路径与 sha256 见 21.1 报告 `cells[*].checkpoint_sha256`，执行前逐一重算，不符即 `engineering_stop`。
- **特征卡先行**：写键截取方式、`allowed` 掩码构造、强制写回放的轮次划分、有效秩相对阈值、Tikhonov `eps`、留一解码距离、seed 列表、bootstrap 次数，必须在**读任何读数之前**落盘为 `feature_card`（独立 JSON，含 `feature_card_sha256` 写进报告）。未先落盘或读数后改动 ⇒ `engineering_stop`。
- **双列 runner sha**：`runner_sha256`（产生读数）与 `submitted_runner_sha256`（提交字节）分列；runner 未写之前两列均为 `null`，**不得预填**。
- **三种写入条件**：① `actual`（原样，记录每轮 written 标志）；② `forced`（`allowed` 全 True）；③ `random_init`（同图同 seed 零训练，`actual` 掩码）。三者共用同一序列、同一评估面。
- **产物**：`reports/taiji_f_expressive_capacity_20261108.json`（write-then-rename）、`plans/manifests/taiji_f_expressive_capacity_20261108.json`、`…_progress.jsonl`（逐行 append＋fsync）、逐面读数（22 面 × 4 checkpoint × 3 条件）、四条对照原值、双统计量敏感性、配对 bootstrap 区间、归因分账。
- **成功语义**：`expressive_capacity_present` 只产出**架构需求输入**（"若要修，必须满足 rank/对齐/容量三条"），不是定价、不是训练准入、不是能力成立。

## 5. 出口（八类，具名 `verdict`）

全部按 manifest `criteria.decision_rule` 与 `verdict_precedence` 判定，本件不复述门值。八类：`engineering_stop` / `capacity_below_event_count` / `key_space_collapsed_structurally` / `query_key_misaligned` / `value_space_collapsed` / `key_space_collapsed_under_actual_timing` / `expressive_capacity_present` / `inconclusive`。

**三类判负各自对应一个不同的下一步**（这是本专项对 owner 决策的直接供给）：

| verdict | 下一步的性质 |
|---|---|
| `key_space_collapsed_structurally` | 架构级改造（改 `π` 的输入或结构、引入 tied key、键白化） |
| `query_key_misaligned` | 最小充分修复＝写键与读键共享投影／weight tying，**不必另立主体** |
| `key_space_collapsed_under_actual_timing` | 工程改动（写入时机），**不是架构问题**，也不构成训练理由 |

CPU 到 1,200（80%）即预警、到 1,500 即停并如实报，不缩数据后重跑；wall 上限按包 21 家族 CPU:wall=1:1.5 比例记为 2,250。

## 6. 执行前硬门（全为真才准开跑）

1. 本预注册 manifest 已入库，且 runner 内 `CRITERIA_SHA256` 常量与 manifest 的 `criteria_sha256` 逐字一致；判据自 2026-11-08 冻结后未改。
2. `feature_card` 已落盘并登记 sha（§4）。
3. 4 个 checkpoint sha 与 21.1 报告一致；图身份与 `candidate_sha_v2` 未漂移。
4. 双列 runner sha 齐备且 runner 已入库；0 训练 / 0 confirm 声明落盘。
5. owner 已按本件单独裁定**执行**（本轮只冻结预注册，**准入 ≠ 执行**）。

## 7. 判据、出口与验收（索引）

冻结判据、门值与控制统计量**只以** [FEC-1 预注册 manifest](../manifests/taiji_f_expressive_capacity_prereg_20261108.json) 的 `criteria` 为准（附 `criteria_sha256`）；出口语义见本件 §5，执行前硬门见 §6。**本节只作索引，不改上述任何判据文字、门值或预算**；改动须另出预注册版本并保留原块审计（D16-38）。

判据自身的可核验性由[判据自检器](../../scripts/training/verify_taiji_f_expressive_capacity_prereg_20261108.py)守住，回执见 [guard 2026-11-08](../../reports/taiji_f_expressive_capacity_prereg_guard_20261108.json)：六项守卫（`criteria_sha256` 重算一致／篡改一个门值后不一致；三处队首逐字一致／篡改一字后不一致；本件不携带第二套门值／植入门值字面量后能被检出）全过。其中队首一致性与门值字面量两项另做**真实文件篡改**演练：把一处队首加两字后守卫确实变红（`headline_green=false`、退出码 1），恢复后字节与演练前逐字节相同、守卫复绿——守卫不是恒真式。

## 8. 停止线

本件冻结后停在 owner 裁定节点：**是否按本预注册合同执行 FEC-1 读数（0 训练 / CPU ≤1,500）**。FEC-1 已立、判据已冻结，但**尚未执行、尚未产出任何读数**；21.2／21.4／U1 与各训练包保持冻结。范围口径：立专项 ≠ 执行 ≠ 里程碑关闭 ≠ 范围排除批准。
