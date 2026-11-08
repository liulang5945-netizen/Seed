# 太极存储地址系统：边界、闭环与推导（2026-11-08）

> 状态：**第一步产物（系统边界）已完成；第二步代码落位与第三步推导见 §8 起。**
> 对象：`taiji/living_core.py` 的 store 读写入口、`taiji/sc_sfk.py` 的地址表、`taiji/growth_wire.py` 的会话/容量入口，以及只在离线存在的 `taiji/growth_gate.py`。
> 口径：本文只描述**一套**系统——SC-SFK 地址系统（`sc_sfk_enabled=True`）。默认关闭时走的 dense slot store 是**冻结兼容壳**，逐字保留用于关闭态零扰动，不属于本文的算法本体；它的出口单独列在 §4.4，且不参与任何结论。
> 纪律：0 训练 / 0 优化器步 / 0 Adam / 0 confirm；不新增训练窗；不接训练 runner；禁止 `python -m black`。

---

## 0. 一句话边界

这套系统的全部能力是：**把一段字节记录按“2 字节词集合”放进一个表，用一个在表内只出现一次的詞做锚点，读的时候只在这个锚点唯一时回答；回答的内容是写入时的 `fast` 快照；写的时候先扩容量再生效，整批要么全进要么全退。**

它现在**不具备**的能力，本文逐条标成“空接”或“未接”，不藏在代码里：
1. 读到的值还没有进入 `fast`（旧版 `_sc_sfk_read` 只返回判定，不改状态）；
2. 读没有成为下一笔写的依据（写入是调用方显式发起的，不读表）；
3. 没有修剪（prune）动作，死地址只增不减；
4. 地址层是闭式统计、0 参数，**不能被训练塑形**。

---

## 1. 状态清单

“生命周期”指谁创建、谁改变、谁销毁；“关闭态”指 `sc_sfk_enabled=False` 时该状态是否存在。

### 1.1 图内张量（参数）

| 状态 | 形状/量 | 生命周期 | 谁能改 | 关闭态 |
|---|---|---|---|---|
| encoder / read / write / world / readout 等全部 `nn.Parameter` | 135 个张量、3,311,259 个元素 | 构造时创建，随 subject 存活 | **只有优化器**；地址层任何动作都不碰 | 存在，字节不变 |
| `W, U, V, P, R, tau_logit, nu_projection` 等 | 同上 | 同上 | 同上 | 存在 |
| `candidate_sha_v2` | 摘要 | 由 `graph_identity` 计算 | 只读 | 不变 |

地址系统对参数的态度是**只读**：写入不产生梯度，读取不产生梯度，容量增长只改 Python 状态。这是“地址是本体缺件而不是阈值问题”的前提，也是“它还不能被训练塑形”的原因。

### 1.2 `LivingState` 字段

| 字段 | 形状/类型 | 生命周期 | 谁能改 | 关闭态 |
|---|---|---|---|---|
| `subject_id` | str | 会话创建 | 只读 | 有 |
| `revision` | int | 调用方 | 调用方 | 有 |
| `step` | int | 每 `step()` +1 | `step` | 有 |
| `h` | `(groups, width)` | 每步更新 | `step` | 有 |
| `fast` | `(groups, fast_heads, key_width, value_width)` | 每步 `write_fast`；**开启后读命中也会混合** | `step` / `addressed_read`(仅开启) | 有 |
| `nu_ema` | `(8,)` | 每步 | `step` | 有 |
| `keys` | `tuple[(source_id, position)]` | 每步 | `step`，`align_positions` 对齐 | 有 |
| `coverage`, `prior` | `(T,)` | 每步 / `consume_output` | `step` / `consume_output` | 有 |
| `store_keys/values/age/count` | 32 槽张量 | `attach_store` 创建，`clear_store` 重置 | legacy `addressed_write` | **None** |
| `sc_sfk` | `AddressTable` 或 None | `attach_store` 创建，`clear_store` 换新表 | 只有 `AddressTable` 自己的方法 | **None** |

关键事实：`sc_sfk` 是**纯 Python 对象**，不在 `state_dict()` 里，也不在 `candidate_sha_v2` 里；关闭态它必须是 `None`，这是关闭态零扰动的一项硬检查。

### 1.3 `AddressTable` 内部状态

| 状态 | 类型 | 生命周期 | 谁能改 | 说明 |
|---|---|---|---|---|
| `capacity` | int | 初始 `START_CAPACITY=2` | `_write_one` 写前倍增 | **只是声明上界，不是分配**；列表 append 不复制 |
| `peak_capacity` | int | 单调 | 倍增时 | 读数 |
| `addresses[i].features` | `tuple[str,...]` | 追加后不再改 | `_write_one` 追加 | 2 字节词的**去重集合**（与位置无关） |
| `addresses[i].values` | `list[Any]`，长度 1 | 每次写更新 | merge/replace | `values[-1]` 是**当前值**（fast 快照字节） |
| `addresses[i].history` | `list[{"value": ...}]` | 每次 replace 追加 | replace | 被取代的旧值，**只进不出**，与当前值不重复存储 |
| `addresses[i].count` | int | 每次写 +1 | merge/replace | 观测次数；参与 DF。重复观测即使值变了也计数 |
| `expansions` | int | 单调 | 倍增 | 本文用作**代际 generation** |
| `merged / replaced / appended / rejected_batches / rollbacks` | int | 单调 | 对应动作 | 事件计数 |
| `global_checks / global_failures` | int | 单调 | 全局自检 | 事件计数 |
| `evictions / silent_overwrites` | int | **恒为 0** | 无写入路径 | 结构零，不是计数器 |
| `history`（表级，`self.history`） | list | **从未被读写** | 无 | **死状态，第二步删除** |

### 1.4 词频统计

`document_frequency` / `idf_table` / `unique_features` 都是**派生量**，每次读或自检时从 `addresses` 重算，不单独存。这是有意的：没有增量维护就没有增量漂移。代价是自检 O(n²)。

DF 的口径：一个地址的 `count` 是“这条记录被观测了几次”，DF 按次数累加。所以**重复写同一条记录会把它的词从“唯一”抬成“共享载体”**。这不是 bug，是夹具里 8 条基础记录合并成 1 个地址、其词变成载体的原因。

### 1.5 会话态 / 容量

- 会话边界由调用方调用 `attach_store` / `clear_store` 表达；`clear_store` **换一张新表**，旧表的值与历史整体丢弃（没有跨会话墓碑）。
- `growth_wire` 的 `generation` 目前**不在 state 里**，是函数参数传进传出；这是“接不上”的一项，第二步把它统一到 `expansions`。
- 容量没有硬上限（`slot_cap=None`），只有倍增。

---

## 2. 动作清单

每个动作给出：输入 → 输出 / 副作用 / 前置条件。**所有动作都在纯 Python 里完成，不碰参数。**

| # | 动作 | 输入 | 输出 | 副作用 | 前置/拒绝 |
|---|---|---|---|---|---|
| A1 | `attach_store` | state | state' | 开启：建空表；关闭：建 32 槽张量 | 幂等（已有则不重建） |
| A2 | `clear_store` | state | state' | 开启：换新空表；关闭：张量清零 | 丢弃全部值与历史 |
| A3 | 写入 `addressed_write` | state, raw | state', info | 见 A5–A9 | 见 §4.2 |
| A4 | 读出 `addressed_read` | state, raw | state', info | **开启且唯一锚定**：`fast ← (1−α)fast + α·value`；否则不动 state | 见 §4.1 |
| A5 | 追加 append | features, value | 新地址 | `capacity` 先倍增，再 append；`appended+1` | 表内无同特征集 |
| A6 | 合并 merge | 同址已有同值 | 原地址 | `values.append(v)`；`count+1`；`merged+1` | 值逐字节相等 |
| A7 | 取代 replace | 同址不同值 | 原地址 | 旧 `values` 快照进 `history`；`values.append(v')`；`replaced+1` | 同特征集 |
| A8 | 整批拒绝 | 两条同址不同值 | 原表 | 先 `restore(snapshot)`；`rejected_batches+1` | 批内冲突 |
| A9 | 回滚 | 写后全局自检失败 | 原表 | `restore(snapshot)`；`rollbacks+1` | 见 A11 |
| A10 | 扩容量 | 表满 | — | `capacity *= 2`；`expansions+1`；“先扩后写” | 无上限 |
| A11 | 全局自检 | 表 + 写前可锚定集合 + 本批目标 | ok/fail | `global_checks+1` | 见 §3.3 |
| A12 | 会话边界切换 | state, pending | state' | 关闭：恒等；开启：**回报表容量/代际，不分配张量** | 见 §6.3 的双开关问题 |
| A13 | 修剪 prune | — | — | **不存在** | **显式未接**；删值会违反旧值守恒，当前没有安全的删除判据 |
| A14 | 生长（神经图） | — | — | 不在这套系统里；地址层只扩容量，不长单元 | — |
| A15 | 神经前向 `step` | state, event | LivingStep | 更新 h/fast/nu/coverage/prior | 与地址层解耦 |

**“读入 fast”是 A4 的副作用，不是 A3 的一部分**：写只产生存储；读决定是否把存储送进下一步。第二步把这两段接上（旧版 A4 关闭时是空接）。

---

## 3. 出口清单（必须由结构条件触发）

### 3.1 SC-SFK 读出口

| 出口 | 触发条件（结构） | 状态变化 | 旧版是否满足 |
|---|---|---|---|
| `no_store` | `state.sc_sfk is None` | 不动 | ✓ |
| `no_match`（未知词） | `∃w∈Q: DF(w)=0` | 不动 | ✓ |
| `no_match`（无唯一词） | `Q∩U(T)=∅` | 不动 | ✓ |
| `matched` | `|Anchors(Q)| = 1` | `fast ← (1−α)fast + α·value`（第二步） | 旧版是 `|anchors|=1 ∨ margin≥0.02`，**第二项是阈值** ✗ |
| `ambiguous_multiple_anchors` | `|Anchors(Q)| ≥ 2` | 不动 | 旧版是 else 分支，带阈值 ✗ |

其中 `U(T)={w:DF(w)=1}`，`Anchors(Q)={i : A_i ∩ (Q∩U(T)) ≠ ∅}`。修法：删掉 `ANCHOR_MARGIN`，`matched` 只由 `|Anchors|=1` 决定。

### 3.2 SC-SFK 写出口

| 出口 | 触发条件（结构） | 状态变化 |
|---|---|---|
| `appended_after_growth` | 表内无**完全相同**的特征集 | 先倍增容量，再追加 |
| `merged_duplicate` | 同特征集且值逐字节相等 | `count+1`，`values` 追加 |
| `replaced_at_same_address` | 同特征集且值不同 | 旧值进 `history`，当前值换新 |
| `rejected_batch_address_collision` | 批内两条同特征集、不同值 | 整批回滚，表不动 |
| `rolled_back_global_check_failed` | 写后自检失败 | 整批回滚，表不动 |

旧版写入口用 `COPY_OVERLAP=0.85` 的加权 Jaccard 判定“同址”，这是**为夹具调出来的阈值**；第二步改成**特征集逐项相等**（结构条件），`COPY_OVERLAP` 退役。代价：不同措辞（不同 2 字节集合）不再合并成同一地址；收益：写入身份不再依赖任何数值阈值。

### 3.3 全局自检（A11）的结构判据

旧版 `check_global` 的两条：
1. `self_read = ∀i: similarity(A_i,A_i) ≥ 1−1e-12` —— **恒真**：加权 Jaccard 的自相似度按定义等于 1（分子分母是同一个求和）。它不检验“能不能被锚定”，只检验“集合等于自己”。这是闭环里的一处**假接线**。
2. 两两 `similarity(A_i,A_j) ≥ 0.85` —— 阈值判据；而且两个地址能在写入时合并（同阈值），只在 IDF 漂移后才可能重新越过阈值，所以它抓到的是“漂移”，不是“身份冲突”。

第二步替换为两条结构判据：
- **G-A 无错指**：`∀i: read(A_i).address ∈ {None, i}`。任何地址的自身特征集，要么读回自己，要么说“不知道”，绝不能读到别的地址。
- **G-B 无附带损失**：记写前可锚定集合 `U_before = {i: A_i ∩ U(T_before) ≠ ∅}`，本批触及的目标集合 `Touched`。要求 `U_before \ Touched ⊆ U_after`。即：**一次写可以把自己的地址降级为载体（重复观测），但不能把别的地址的唯一词抢走。** 违反则整批回滚。
- **G-C 身份唯一**：所有 `A_i` 两两不相等（由“逐项相等才合并”在构造上保证，仍显式断言）。

这三条都是集合/计数条件，不含调出来的常数。

### 3.4 legacy（冻结兼容壳）出口——只登记，不进入本文结论

| 出口 | 触发 | 性质 |
|---|---|---|
| `enabled=False` / `empty_store` | 无 store / 空 | 结构 |
| `blended` | 有槽即 softmax 混合 | **总会回答**，没有“不知道”出口 |
| `below_gate` | `top_cosine < read_gate_cosine` | 阈值；默认 `None` 即关闭，**关掉才安全** |
| `evicted` | 槽满 `argmin(age)` | 结构，但**破坏性**：静默丢掉一条旧值 |
| `growth reject_write` | `(target−needed)/target < reserve_ratio` | 阈值 |
| `rejected_copy` | 方向带 overlap/separation | 阈值 |
| `growth_gate` 的 7 个 exit | 离线评估 G1–G4 | **未接入在线路径**；`growth_gate` 不被冻结图导入 |

---

## 4. 闭环图

### 4.1 旧版真实回路（含三处空接）

```
写：raw ──features──► 相似度≥0.85 找同址 ──是──► merge / replace(value=raw.hex())
                              │否
                              ▼
                       先倍增容量 ─► append
                              │
                              ▼
                     DF/IDF 全表重算
                              │
                              ▼
              全局自检：自相似度=1（恒真） ∧ 两两<0.85
                     │通过                 │失败
                     ▼                     ▼
                  提交                  snapshot 回滚
                     │
                     │ value = 记录字节（不是 fast）
                     ▼
读：raw ──features──► 唯一锚点 ──0──► no_match ──► fast 不动 ──┐
                          ├──1──► matched ──► 只返回判定，不返回值 ◄── [空接1]
                          └─≥2──► ambiguous ──► fast 不动 ──────┘
                                                                  │
                                                    ┌─────────────┘
                                                    ▼
                                       下一笔写不读表 ── [空接2]
                                       纠正只能靠调用方显式再写 ── [空接3]
```

- **空接 1**：读命中了，但 `_sc_sfk_read` 不改 `fast`；即使改了，`value=raw.hex()` 也不是可混合的向量。接不上的原因：写入时没有把“当时的 fast 快照”存进去。
- **空接 2**：写入完全由调用方发起，读的结果不进入下一笔写的输入；没有“读→写”的回授。
- **空接 3**：没有“反馈→写→再读→再纠”的自动环；纠正只有同址不同值的 replace，且必须由外部显式触发。
- **假接线**：全局自检的 `self_read` 恒真，回滚分支实际上只能被 0.85 阈值触发。
- **未接**：修剪不存在；跨会话历史不存在；`growth_gate` 离线存在但未接入。

### 4.2 第二步落位后的目标回路

```
写：raw ──features──► 特征集逐项相等? ──是──► 同值 merge / 异值 replace(旧值进 history)
                              │否
                              ▼
                       先倍增容量 ─► append(value = 当前 fast 快照)
                              │
                              ▼
                     DF/IDF 全表重算（派生，不增量维护）
                              │
                              ▼
        全局自检 G-A 无错指 ∧ G-B 无附带损失 ∧ G-C 身份唯一
                     │通过                         │失败
                     ▼                             ▼
                  提交                       snapshot 整批回滚
                     │
                     ▼
读：raw ──features──► 唯一锚点 ──0──► no_match（fast 不动，显式“不知道”）
                          ├──1──► matched ──► value 快照 ──► fast ← (1−α)fast + α·value
                          └─≥2──► ambiguous（fast 不动，显式“说不准”）
                                              │
                                              ▼
                              新的 fast 成为下一笔写的 value ──► 反馈环闭合
                                              │
                                              ▼
                              再读同一地址 ──► 值覆盖/幂等 ──► 纠正环闭合
```

闭合后的三段回路：
1. **局部写 → 全局统计 → 自检 → 通过/回滚**：`write_batch_checked` 内完成，判据是 §3.3 的 G-A/G-B/G-C。
2. **读 → 是否进 fast**：`matched` 才混合，`ambiguous/no_match` 明确不动。
3. **反馈 → 写 → 再读 → 再纠**：读后的 `fast` 是下一笔写的值；同址不同值走 replace，旧值进 history。

仍然接不上、但已显式占位的两段：
- **修剪**：没有安全的删除判据（删值会破坏旧值守恒），所以不接；见 §6.4。
- **跨会话**：`clear_store` 丢弃整表，没有墓碑/回放；跨会话连续性是另一个系统，不在这套里假装有。

---

## 5. 自检：只存在于探针侧 / 半开 / 双判据 / 夹具常数 / 关掉才安全

| # | 问题 | 位置 | 处置 |
|---|---|---|---|
| S-1 | runner 侧自带一份 `AddressTable`（`features`/`idf`/`class AddressTable` 共约 260 行），与正式实现逐字重复 | `scripts/training/probe_taiji_sc_address_20261108.py` | **退役**：删除本地实现，改为 `from taiji.sc_sfk import ...`；该 runner 标 `SUPERSEDED` 并加启动闸，历史 JSON 不改 |
| S-2 | 表级 `self.history` 从未读写 | `taiji/sc_sfk.py` | **删除**（死状态） |
| S-3 | `evictions/silent_overwrites` 从无写入路径 | `taiji/sc_sfk.py` | **改成断言恒为 0 的不变量**，不再伪装成计数器 |
| S-4 | `self_read` 恒真（自相似度=1） | `AddressTable.check_global` | **替换**为 G-A 无错指 |
| S-5 | `ANCHOR_MARGIN=0.02` 在出口路径上 | `AddressTable.read` | **删除**；`matched ⟺ |Anchors|=1` |
| S-6 | `COPY_OVERLAP=0.85` 同时决定“同址”和“冲突” | `target_of` / `check_global` | **退役**；写身份改逐项相等，冲突改 G-C |
| S-7 | 读返回值不接 `fast` | `_sc_sfk_read` | **接上**：写存 fast 快照，读唯一锚定时混合 |
| S-8 | `value=raw.hex()` 不是状态向量 | `_sc_sfk_write` | **改为 fast 快照字节** |
| S-9 | `growth_wire` 与 `sc_sfk` 两套容量权威并存；开启 SC-SFK 时 `session_boundary` 用 `slots=0` 进 `plan_growth`，`(target-needed)/target` **除零** | `taiji/growth_wire.py` | 统一：`state.sc_sfk is not None` 时走表容量；`GrowthWireConfig.enabled=True` 同时出现即**结构性拒绝**（`ValueError`），不静默混用 |
| S-10 | `state_bytes()` 不计 `sc_sfk` | `taiji/growth_wire.py` | 表存在时加 `table.nbytes()`；关闭态返回 0，零扰动 |
| S-11 | `read_gate_cosine` 与 SC-SFK 判定是两套判据 | `LivingCoreConfig` | 声明为 legacy-only；开启 SC-SFK 时该字段不参与（整路径委托，不是半开） |
| S-12 | `growth_gate` 离线存在，`shadow_apply` 只镜像 legacy append | `taiji/growth_gate.py` | 声明为**离线核验器、未接入**；SC-SFK 的自检在表内，不重复接线 |
| S-13 | 夹具常数（8/22/66/154、token 池、位置）只出现在 runner | `scripts/training/*` | 算法不读这些数；在 §7 写成显式假设，不写进模块默认值 |
| S-14 | `START_CAPACITY=2` 是策略常数，但不决定任何出口 | `taiji/sc_sfk.py` | 保留；声明为增长策略，不是判据 |
| S-15 | legacy 读“总会回答”，没有“不知道” | `addressed_read` legacy 分支 | 声明为兼容壳的已知性质；地址系统只在 SC-SFK 路径上承诺“不知道” |

---

## 6. 显式假设（不成立时哪条能力失效）

| 假设 | 内容 | 不成立时 |
|---|---|---|
| **A1 稀有词** | 每条被存记录在读取时至少含一个 `DF=1` 且属于该记录的特征 | 读退化为 `no_match`/`ambiguous`；“22/22 换措辞”不再是通用召回证据 |
| **A2 集合身份** | 同一内容的 2 字节词集合稳定（位置无关）；不同措辞=不同集合 | 不同措辞会追加为新地址；旧地址可能失去锚点，该批被 G-B 回滚 |
| **A3 单写者** | 同一时刻只有一个线程改表 | 快照/回滚不再原子；本实现不加锁 |
| **A4 值形状** | 存的 fast 快照与当前 config 的 `(groups, fast_heads, key_width, value_width)` 同形同 dtype | 读时解包失败，显式报错，不静默截断 |
| **A5 内存** | 表能装下所有被接受的记录；容量无硬上限 | 触发 OOM；没有 prune 可以退 |
| **A6 记录长度** | 记录至少 2 字节；奇数尾字节被丢弃 | 只差尾字节的两条记录同址（会 merge/replace） |
| **A7 夹具** | 每个 face 带自己的 token 词；8 条基础记录是同一句重复 8 次 | §7 的读数只说明该夹具，不是自然文本结论 |

---

## 7. 第二步改动范围（预登记）

1. `taiji/sc_sfk.py`：特征集逐项相等定址；`matched ⟺ |Anchors|=1`；删除 `ANCHOR_MARGIN`/`COPY_OVERLAP`/表级 `history`/自相似度假检查；新增 G-A/G-B/G-C；`read` 返回 `value`；`nbytes`；`generation=expansions`。
2. `taiji/living_core.py`：`_sc_sfk_write` 存 `fast` 快照；`_sc_sfk_read` 在 `matched` 时混合进 `fast`，abstain 时逐字不动。
3. `taiji/growth_wire.py`：`state.sc_sfk is not None` 时容量权威归表；双开关结构性拒绝；`state_bytes` 计入表。
4. `scripts/training/probe_taiji_sc_address_20261108.py`：删除重复实现，改导入；标 `SUPERSEDED` + 启动闸。
5. `scripts/training/probe_taiji_store_wire_20261108.py`：open 阶段改测值路径（fast 只在 matched 动、值逐字节回读、冲突批改为“同集合不同值”）；先写 v2 manifest 再跑。
6. 关闭态硬门槛不变：同一批 176 条读数 `max|Δ|=0`、参数 135/3,311,259、`state_bytes`、`candidate_sha_v2`、store 签名全部不变，且 `sc_sfk is None`。

---

## 8. 第三步：形式化（定义、定理、证明、反例边界）

> 本节只覆盖第二步落位后的 SC-SFK 系统。符号与 §1–§3 一致：`A_i` 是地址 `i` 的特征集合，`c_i` 是观测数，`v_i` 是当前值，`H_i` 是旧值历史，`T` 是表，`U(T)={w:DF(w)=1}`。
> 不引用 U1 勘误中标为 INVALID EVIDENCE 的结论；不用实测数据代替证明。

### 8.1 定义（全部可计算）

**D1 地址.** 记录 `x∈{0,1}^{8*}` 的特征集 `φ(x) = { hex(x[2k:2k+2]) : 0 ≤ k < ⌊len(x)/2⌋ }`（奇数尾字节丢弃），去重。表 `T` 的地址是序列 `(A_1,…,A_n)`，`A_i=φ(x_i)` 非空；地址 `i` 的身份就是 `A_i` 本身（不是向量，不是哈希）。

**D2 文档频率.** `DF_T(w) = Σ_{i: w∈A_i} c_i`；`N_T = Σ_i c_i`；未出现记 0。

**D3 唯一特征集.** `U(T) = { w : DF_T(w) = 1 }`。`U` 只依赖计数，不依赖 idf。

**D4 锚点.** 对查询 `q`，`Q=φ(q)`，
`Anchors_T(q) = { i : A_i ∩ (Q ∩ U(T)) ≠ ∅ }`。

**D5 读判定.**
- `no_match` 若 `∃w∈Q: DF_T(w)=0` 或 `Q∩U(T)=∅`；
- `matched(i)` 若 `Anchors_T(q)={i}`；
- `ambiguous` 若 `|Anchors_T(q)|≥2`。

**D6 写目标.** `target_T(A) = i` 若 `A_i = A`（逐项相等），否则 `⊥`。写动作：`⊥→append`；`i` 且 `v∈values_i→merge`；`i` 且 `v∉values_i→replace`。

**D7 可锚定.** `Addr(T) = { i : A_i ∩ U(T) ≠ ∅ }`。

**D8 容量.** `C_0=2`；写前若 `n ≥ C` 则 `C←2C`，`expansions+1`。`C` 是声明上界，append 不复制。

**D9 全局自检.** 对写前表 `T`、写后表 `T'`、本批目标集合 `Touched`：
- G-A：`∀i: read_{T'}(A_i).address ∈ {None, i}`；
- G-B：`Addr(T) \ Touched ⊆ Addr(T')`；
- G-C：`i≠j ⇒ A_i ≠ A_j`。
三条全过才提交，否则 `restore` 到写前快照（内容逐项复原；事件计数不回滚）。

**D10 读混合.** `matched(i)` 时 `F' = (1−α)F + α v_i`，`α=ADDRESSED_ALPHA=0.5`；`no_match/ambiguous` 时 `F'=F`（逐字相等）。

**D11 相似度（只作读数，不参与判定）.** 加权 Jaccard
`sim(A,B) = Σ_{w∈A∩B} idf(w) / Σ_{w∈A∪B} idf(w)`，
`idf(w)=log(1+N_T/(1+DF_T(w)))`，`w∉T` 时取 `log(1+N_T)`。它的唯一用途是报告 `top1/top2/margin`；**没有任何出口读它**。

### 8.2 定理

**定理 1（唯一锚定的充分条件 / 不会错指）**
*假设*：`q` 的每个特征都在 `T` 中出现（`∀w∈Q: DF_T(w)≥1`）。
*结论*：`Anchors_T(q) ⊆ { i : A_i ⊇ Q∩U(T) }`；特别地，若 `q` 是某条已存记录的特征集 `A_i`，则 `Anchors_T(A_i) ⊆ {i}`，于是读 `A_i` 只能得到 `matched(i)` 或 `no_match`，**不可能读到别的地址**。
*证明*：取 `i∈Anchors`，存在 `w∈Q∩U` 且 `w∈A_i`。`DF(w)=1` 意味着全表恰有一个地址含 `w`（按次数计，`c_j≥1` 的任何地址都算），所以含 `w` 的地址唯一，只能是 `i`；若 `j∈Anchors` 也由同一个 `w` 锚定，则 `j=i`。对 `q=A_i`：任何 `w∈A_i∩U` 都满足 `DF(w)=1` 且 `w∈A_i`，故其唯一含者就是 `i`；若 `Q∩U=∅` 则 `no_match`；若 `A_i` 有特征 `DF=0` 则不可能（`A_i` 是已存集合）。∎
*退化*：A1 不成立（`Q∩U=∅`）时结论退化为“只能说不知道”，不会给出错误地址。

**推论 1（充分条件的可检验形式）**
若存在 `w∈Q` 使 `DF(w)=1`，且所有这样的 `w` 都属于同一个地址 `i`，则 `matched(i)`。这是构造地址时的充要判据；代码用 `|Anchors|=1` 直接检验，不需要先求“所有唯一词的主”。

**定理 2（词频漂移的界）**
*假设*：一批写入 `B={x_1,…,x_m}`，写前表 `T`，写后 `T'`；`Δ = Σ_j |φ(x_j) ∩ U(T)|`（本批带进来的、写前唯一的特征出现次数，按记录计）。
*结论*：
1. 只有含 `w∈U(T)` 的地址可能失去可锚定性；具体地，`i∈Addr(T)\Addr(T')` 仅当 `A_i ∩ U(T) ≠ ∅` 且 `A_i` 的某个唯一特征在 `B` 中出现。
2. 失去可锚定性的地址数 `|Addr(T)\Addr(T')| ≤ Δ`。
3. 若 `B` 的特征与 `U(T)` 不交（`Δ=0`），则 `Addr(T') ⊇ Addr(T)`，一个旧地址也不会掉。
*证明*：`DF` 只在 `B` 的特征上增加。对 `w∉⋃φ(x_j)`，`DF_{T'}(w)=DF_T(w)`，故 `U(T)` 中未被 `B` 触及的特征仍是唯一，其所属地址保持可锚定。若 `i` 掉出 `Addr`，它写前必有一个唯一特征 `w`（否则写前就不可锚定），而写后 `w` 不再唯一，只能是 `DF` 在 `w` 上增加，即 `w∈⋃φ(x_j)`；每个这样的 `(j,w)` 贡献一次计数，故失去的地址数不超过 `Δ`。`Δ=0` 时没有 `w` 的 `DF` 变化，`U(T')=U(T)`，结论 3 成立。∎
*说明*：`Δ` 是本批“抢锚”的代价上界；G-B 在 `Δ` 落到别的地址时整批拒绝，所以系统实际保证的是 `Addr(T)\Touched ⊆ Addr(T')`，比定理 2 的计数界更强。

**定理 3（容量倍增的摊销代价）**
*假设*：从 `C_0=2` 开始，只追加、不删除，第 `n` 次追加前按 D8 倍增。
*结论*：
1. `expansions(n) = max(0, ⌈log_2(n/C_0)⌉)`（`n` 为追加后的地址数上界意义下），且 `C(n) ≤ 2·max(C_0,n)`。
2. 若 append 需要复制整表（定长数组实现），总复制代价 `Σ_k C_k ≤ 2C(n) = O(n)`，单次摊销 `O(1)`。
3. 本实现 append 是 Python list 追加、容量只是标记，复制代价为 0；单次写的实际主项是全局自检的 `O(n²·f)`（`f` 为平均特征数），容量倍增不改变这一项。
*证明*：倍增序列 `2,4,8,…` 的几何和 `< 2·末项`；每两个容量级之间至少发生一次追加，故级数与 `log n` 同阶。列表追加的均摊复制由 CPython 的 over-allocation 承担，与本模块的 `capacity` 无关；自检对每对地址做集合交并，`n(n−1)/2` 对，每对 `O(f)`。∎
*边界*：`n` 大时写变慢的是自检，不是扩容量；若要 `O(n)` 写，应把 G-C 改成哈希集合（当前不做，因为 `n` 很小且要保留逐项可审计性）。

**定理 4（合并/取代的幂等性与旧值守恒）**
*假设*：同一个地址 `A_i`，写入序列 `(A_i,v_1),(A_i,v_2),…`；值比较是逐字节相等。
*结论*：
1. **地址幂等**：任何次数的同址写都不改变 `A_i`，也不改变 `i`。
2. **值幂等**：重复写同一个 `v` 是 merge，`values_i` 的当前值仍是 `v`，读回的值不变（只有 `c_i` 增加）；因此“读到的值”对重复观测幂等。
3. **旧值守恒**：任何 replace 都把被取代的当前值追加到 `H_i`，再把 `values_i` 的当前值换成新值；`values_i` 与 `H_i` 不重叠，二者的多重集合并等于全部观测值的日志，旧值永不丢失。
4. **取代的收敛**：连续两次 replace 到同一个 `v'`，第二次是 merge（因为 `values_i[-1]=v'`），值固定；replace 序列的当前值等于最后一个写入的值。
*证明*：(1) 由 D6，`target` 只由 `A_i` 决定；(2) merge 只 `count+1`，`values_i` 不变；(3) replace 在改 `values_i` 之前执行 `H_i.append(values_i[-1])`，之后执行 `values_i[-1]=v'`，故被取代的值只进不出且不与当前值重复；(4) 第二次写时 `values_i[-1]=v'`，走 merge。∎
*注意*：`count` 不是幂等的（每次观测 +1，merge 和 replace 都算一次观测），这是有意的“观测次数”语义；它影响 `DF`，进而影响可锚定性（定理 2），所以重复写同一条记录会把它的词降级为载体。`values` 只保留当前值，`history` 只保留被取代的值，两者不重叠，内存 O(观测数)。

**定理 5（整批拒绝的原子性）**
*假设*：`write_batch_checked` 在一次调用内完成；表内容 `Content(T) = (addresses 的 features/values/history/count, capacity)`；`snapshot` 深拷贝这些字段。
*结论*：
1. 批内冲突（同址不同值）时，`Content` 与调用前逐项相等；`rejected_batches` 单调 +1。
2. 写后自检失败时，`Content` 与调用前逐项相等；`rollbacks` 单调 +1。
3. 任何“部分写入后失败”的中间态不会暴露给调用方（函数返回前已 `restore`）。
*证明*：冲突路径在返回前执行 `restore(snapshot)`；自检失败路径在同一函数内 `restore(snapshot)` 后才返回。`snapshot` 含 `features/values/history/count/capacity/peak_capacity`，`restore` 逐字段写回。事件计数（`rejected_batches/rollbacks/global_checks`）不在 `Content` 内，是审计量，不回滚。∎
*边界*：`peak_capacity` 不回滚（它记录曾经达到的最大值），所以回滚后 `capacity` 可以变小而 `peak_capacity` 不变；这是有意的，容量是声明上界，不是内容。

**定理 6（读不到锚点时必须能说“不知道”）**
*假设*：读入口只有 D5 的三个分支；没有其他路径能返回值。
*结论*：若 `Q∩U(T)=∅` 或存在 `DF=0` 的特征，则 `read` 返回 `no_match`，`fast` 逐字不变；系统**不可被强迫作答**。
*证明*：D5 的第一个条件先于锚点计算返回；`Q∩U=∅` 时 `Anchors=∅`，走 `no_match`；D10 规定只有 `matched` 才混合，其余分支 `F'=F`。值只通过 `matched` 分支的 `v_i` 进入状态，空锚点集没有 `i` 可取。∎
*边界*：legacy dense store 不满足这条（softmax 总给权重）；这是本文不把 legacy 计入结论的原因之一。

**定理 7（读入 fast 的可逆性与二次稀释的界）**
*假设*：`F∈R^d`，`V=v_i` 是唯一锚定地址的当前值，`α∈(0,1)`，一次读 `R_V(F)=(1−α)F+αV`。
*结论*：
1. **幂等点**：`R_V(V)=V`；若 `F=V`，读不动。
2. **收缩**：`‖R_V(F)−V‖=(1−α)‖F−V‖`；连续 `k` 次读同一 `V`（中间无其他更新）后，`‖F_k−V‖=(1−α)^k‖F_0−V‖→0`。
3. **二次稀释**：`R_V(R_V(F))=(1−α)²F+α(2−α)V`；原始分量权重 `(1−α)²`，`α=1/2` 时为 `1/4`。
4. **可逆性**：`R_V` 是仿射双射（`α<1`），逆为 `F=(F'−αV)/(1−α)`；在唯一锚定下 `V` 可从表里取回，所以**单次读可逆**。但连续读不同的 `V_1,…,V_k` 后，`F_k` 是 `k+1` 个向量的凸组合，仅凭 `F_k` 无法唯一分解；可逆性只对“已知 `V` 的单次读”成立。
5. **不回答时零扰动**：`no_match/ambiguous` 下 `F'=F`，逐字相等（不是近似）。
*证明*：(1)(2)(3) 直接代入；(4) `α<1` 时 `1−α>0`，仿射映射可逆，逆式代入即得；不同 `V_j` 的复合是 `(1−α)^k F_0 + αΣ_{j=1}^k (1−α)^{k−j} V_j`，`k+1` 个未知量一个方程，不唯一；(5) 由 D10。∎
*注*：读的“可逆”不意味着“无损”：单次读丢掉的是 `F` 在 `V` 方向之外的分量的一部分，逆式需要已知 `V`；系统不保存读前的 `F`，所以工程上不提供回滚读。定理只给代数可逆性与稀释界。

### 8.3 反例边界（让每条定理失效的具体构造）

| 反例 | 构造 | 失效的定理 | 系统表现 |
|---|---|---|---|
| X1 同集合不同值 | 两条记录字节顺序不同、2 字节集合相同，值不同 | 定理 4 的“两条不同记录”直觉 | 第二条 replace 第一条；旧值进 `H_i`，不会两条并存 |
| X2 无唯一词 | 记录的所有词都在别的记录里出现（夹具的 8 条基础记录重复 8 次） | 定理 1 的“能读回” | 该地址不可锚定；读它只能 `no_match`；G-B 允许它自我降级（Touched 含自己） |
| X3 抢锚 | 新记录 `A={a,b,w}`，旧地址 `A_i={w}`，`w` 是 `i` 的唯一词 | 定理 2 结论 3 | `w` 的 `DF` 1→2，`i` 掉出 `Addr`；G-B 失败，整批回滚，新记录不存 |
| X4 复合查询 | `Q` 含两个地址的唯一词 | 定理 1 的“唯一”前提 | `|Anchors|=2`，`ambiguous`，`fast` 不动 |
| X5 错一个词 | 把记录的一个 2 字节词换成未出现过的词 | 定理 6 的“其余锚点仍可用” | 早期 `no_match`（任一 `DF=0` 即拒答），保守但不误答 |
| X6 奇数尾字节 | 两条记录只差最后一个字节 | A6 | 尾字节被丢，两条同址；第二条 replace 第一条 |
| X7 自检缺失 | 关闭 G-B，再执行 X3 | 定理 2 的工程保证 | 旧地址静默失去锚点；系统仍不误答，但召回丢失且无告警 |
| X8 α 边界 | `α=1` / `α→0` | 定理 7 | `α=1` 读即覆盖 `F`，单次读不可逆（`1−α=0`）；`α→0` 读几乎不影响 `F`，环断开 |
| X9 大表 | `n` 很大且每写都自检 | 定理 3 结论 3 | 写代价 `O(n²f)`；容量倍增不是瓶颈，自检才是 |
| X10 并发写 | 两个线程同时 `write_batch_checked` | 定理 5 的原子性前提 A3 | `snapshot/restore` 可能交错，原子性失效；本实现不加锁 |

---

### 8.4 落位后的读数（`reports/taiji_store_wire_20261108_{closed,open}.json`）

**关闭态（默认 `sc_sfk_enabled=False`）**

| 项 | 读数 |
|---|---|
| 判定 | `store_addressing_wire_neutral` |
| 读数条数 | 176（154 问 + 22 已写面） |
| `fast` 逐字节 `max｜Δ｜` | **0.0** |
| 概率 `max｜Δ｜` | **0.0** |
| fast / 概率 / 字段摘要相等 | 176/176、176/176、176/176 |
| 参数 | 135 个张量 / 3,311,259 元素，sha 与基线一致 |
| `state_bytes` | 1,100,064（与基线一致） |
| `candidate_sha_v2` | `ff60ac25…`（与基线一致） |
| store 签名 | 不变（occupied 30） |
| `table_not_allocated` | true |
| 源码 sha | 变化（预期；身份摘要不含源码哈希） |
| 成本 | 34.109 CPU s / 3.308 wall s |

**开启态（`sc_sfk_enabled=True`）**

| 项 | 读数 |
|---|---|
| 判定 | `store_addressing_wired` |
| 按存储键自读 | 22/22 |
| 换措辞（同词不同位置） | 22/22 |
| 未写措辞 | 66/66 |
| 全新词 → `no_match` | 22/22 |
| 双稀有词复合 → `ambiguous` | 22/22 |
| 同词新配对 | 22/22 |
| 地址/容量 | 23 个地址（23 个不同集合）；容量 2→32；4 次倍增；`generation=4` |
| 写侧 | append 23、merge 0、replace 7、整批拒绝 1、回滚 0、驱逐 0、静默覆盖 0 |
| 全局自检 | 31 次，失败 0 |
| 表字节 | 983,776（23 条记录 × 32 KiB 快照 + 历史） |
| 可锚定地址 | 22（基础地址被 8 次观测降级为载体，符合 §8.3 X2） |
| 值路径 | matched 132 行，`fast` 移动 126 行；abstain 44 行，移动 **0** 行；值回读 22/22 |
| 冲突批 | `rejected_batch_address_collision`，表前后逐项不变 |
| 幂等探针 | `merged_duplicate`，容量不变，随后快照恢复 |
| 成本 | 4.312 CPU s / 0.813 wall s；30 次写入 `mean_write_wall_ms` 1.0773；最终表全局复检 2.1622 ms |

两处需要解释、不许含糊的读数：

1. **126/132 而不是 132/132**：最后写入的那个 face 的值就是当前 `fast`，对它的 6 条读（1 自读 + 1 换措辞 + 3 未写 + 1 新配对）满足 `F=V`，混合是恒等映射，所以“没动”。这正是定理 7(1)，不是漏接。
2. **merge 0 / replace 7**：8 条基础记录是同一句的 8 次观测，地址相同；值是 `fast` 快照，观测之间 `fast` 已经变化，所以第 2–8 次是 replace（被取代的快照进 history），不是 merge。`count` 仍是 8，载体频率不变，22 个 face 仍各自可锚定——这是“观测次数”和“值相等”两件事分开的直接证据。

---

## 9. 这条线现在能证明什么 / 不能证明什么（人话）

- **能证明**：地址确实是算法本体的缺件，不是阈值问题。默认关闭时同一批 176 条读数逐字节不变（`max|Δ|=0`，参数/状态/身份/签名四项全同）；打开后同一批 154 问读到换措辞 22/22、未写措辞 66/66，两类该拒的 22/22，而且 22 个写入值全部按写入时的 `fast` 快照回读。旧 runner 侧地址表（0.85 相似度、0.02 锚点边际、恒真的自相似检查）已退役，只剩一套实现。
- **没有证明**：这套地址能被训练塑形。它是闭式统计、0 参数，没有梯度能到达；也没有证据说它在自然文本上同样成立。读入 `fast` 的环是接上了，但环上的地址判定仍是手工规则，不是学出来的。
- **仍依赖**：每条记录带一个表内唯一的词（A1）。这是夹具性质；A1 不成立时系统退化为“不知道”（不会给出错误地址），但召回会掉。22/22 不是通用召回的证明。
- **已退役**：runner 侧重复地址表、`COPY_OVERLAP`、`ANCHOR_MARGIN`、表级 `self.history`、恒真的 `self_read`、`growth_wire` 的除零旁路。
- **显式待接**：修剪（没有安全的删除判据）、跨会话历史（`clear_store` 整表丢弃）、训练塑形（0 参数）、离线 `growth_gate`（未接入在线路径，SC-SFK 的自检在表内完成）。

