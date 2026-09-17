# R2-D2 逐位置证据记忆假设（H-A1）隔离验证预注册（冻结版）

2026-09-18。状态：**已冻结**（本文提交即冻结；修订必须新版本，不就地改判据）。
依据：[假设选择备忘](M5_R2_D2_HYPOTHESIS_SELECTION_MEMO_20260918.md)（用户 `/goal` 持续推进指令；选择问题未否决，按备忘唯一收敛结论执行）、[03 §5.3 第 2 段](../active/roadmap/03_CURRENT_EXECUTION.md)、[VISION §15.3/§16.5](VISION_FUTURE_TECHNOLOGY.md)、[R2-D1 测量合同](M5_R2_D1_DEV_MEASUREMENT_CONTRACT_FROZEN_20260917.md)。
训练授权沿 R2 隔离原型序列（H3.8 同口径：隔离目录、CPU、不触默认入口）；**数值预算在 §7 单独冻结，实现门全绿且 checkpoint 前置检查通过前不启动任何训练**。

## §0 目的与边界

1. **唯一目的**：在 R2-D1 可辨识 dev 上，以同族最小对照检验一个学习假设——证据条目**逐位置写入**（而非由 prefix 末状态一次派生）能否形成内容条件自由回答。
2. **明确不做**：不读封存 final；不改默认 `chat()`/产品入口；不引入 copy 门、快参数、辅助损失、跨 episode 记忆、世界模型或在线适应；不重算任何旧分数；不做无界超参搜索。
3. **身份规则**：graph v3 = 新 checkpoint 版本 = 新参数清单；v2 图与其 checkpoint 读路径原样保留。任何报告不得称本包为「引入 attention」——读出沿用 v2 已批准的同族内容寻址，本包只改变证据条目的来源与写入时机（备忘 §4 的 M4 闸门边界）。

## §1 假设 H-A1（冻结陈述）

在相同数据、解码器与真实下一 byte CE 下，把证据条目从「prefix 末状态 h0 一次性线性派生 4 个固定槽」改为「prefix 因果扫描期间逐位置写入 K_i/V_i、生成期由渲染状态重新内容寻址读取」，将使 M3（内容依赖 exact）与 M4（平衡翻转对）超过等机制的末状态对照；收益必须表现为反事实对上答案随单一事实正确翻转，不能由固定「未知」命中、参数容量或键值捷径解释。

## §2 graph v3 精确规格

实现位置：[taiji/sequence_workspace.py](../../taiji/sequence_workspace.py)，配置新增 `evidence_source: "final_state_slots" | "per_position" | "broadcast_final"`（默认 `"final_state_slots"` ⇒ v2 行为逐位不变）。

1. **prefix 扫描（保留全部位置状态）**：h_0=zeros(rw)；逐 prefix byte 递推 `h_i = tanh(emb[x_i]·prefix_input + h_(i-1)·prefix_recur + prefix_bias)`，i=1..L，保留 (h_1,…,h_L)。L ≤ 84（train）/ 99（dev、final），不截断。
2. **证据条目（v3 per_position，A 臂）**：共享投影 `evidence_key/evidence_value ∈ R^{rw×sw}`，`K_i = h_i·evidence_key`，`V_i = h_i·evidence_value`，形状 (L, sw)。
3. **C1 broadcast_final**：与 A **同参数、同前向代码形状**，但每个条目 `K_i = h_L·evidence_key`、`V_i = h_L·evidence_value`（L 个条目全部相同 ⇒ softmax 权重均匀 ⇒ 读出等于 h_L 的投影）。实现门必须逐张量断言 L 个条目完全相等。
4. **v2 final_state_slots（C0）**：维持现有 `workspace_key/workspace_value`，4 槽全部由 h_L 一次派生。
5. **渲染（三臂相同）**：renderer 从学习常量 `start_vector` 起步（v2 单通道合同不变）；每 byte `r_j = tanh(...)` 与 v2 相同；查询 `q_j = r_j·address_query`；v3 读出 `read_j = softmax(K q_j /√sw)·V`（softmax 维 = L 或 4）；logits = `[r_j, read_j]·decoder + decoder_bias`。
6. **目标**：真实下一 byte CE（含 boundary 256），全序列反传沿 decoder → 读出 → 逐位置条目 → prefix 扫描参数；不新增任何损失项；H-GEN/H-OBJ/H-FBW 训练器开关保持默认关闭。
7. **参数清单（冻结，实现门逐一核对活张量）**：

| 臂 | evidence 来源 | 参数 | 清单（在 v2 清单基础上的差异） |
|---|---|---:|---|
| A | per_position | **82,593** | 以 `evidence_key/evidence_value`（各 3,072）替换 `workspace_key/workspace_value`（各 12,288）；其余同名张量与 v2 共享初始化规则 |
| C1 | broadcast_final | **82,593** | 与 A 完全相同（同代码、同参数、同 seed 初始化） |
| C0 | final_state_slots（v2） | **101,025** | v2 清单原样 |

A/C1 比 C0 少 18,432 参数：若 A 胜出，容量方向被反向排除。同名参数按现有「name-seeded 初始化」在同 seed 下共享初值。

8. **checkpoint**：`SEQUENCE_WORKSPACE_VERSION` 2→3，载荷必须含 `evidence_source`；v3 读 v3、v2 读 v2 双路径保留，旧版拒读规则不变；往返对称（checkpoint 写的键都能被 restore 消费）。

## §3 臂与运行身份（冻结）

- **A**（treatment，per_position）、**C1**（主对照，broadcast_final，等机制无逐位置信息）、**C0**（血缘锚点，v2）。
- seeds 固定 20260917 / 20260918 / 20260919；每臂每 seed 从零构建；三臂同 seed 同名参数初值一致。
- 语料：`tests/fixtures/r2_d1_measurement_v1.jsonl`（train 174 / dev 98），corpus digest `53ac9f88695f135d0bb04b4d25d98ab686c82175c45ca8d4d53e64639efbeecb`；运行前报告必须复核 digest 一致，否则拒绝评分。顺序用 fixture 文件内确定性交错序，不额外 shuffle。
- 几何：prefix_width=48、renderer_width=64、slot_width=48、max_sequence_bytes=512、alphabet=257、boundary=256（沿用 v2）。
- 优化：Adam lr=0.05（trainer 现默认值），无新超参。

## §4 评测口径（全部以完整自由输出为准）

1. **主指标**：D1 合同 §3 的 M1/M2/M3/M4/M5，分母逐项明示，逐题记录（id/shape/pair/gold/generation/正确与否），按 source_group 分组；复用 [eval_taiji_r2_d1_surface_policies.py](../../scripts/training/eval_taiji_r2_d1_surface_policies.py) 的指标实现与参照键，不另造定义。固定策略天花板已冻结：M1=0.1735、M3=0.0581、M4=0.0。
2. **无上下文模型基线（D1 合同 §4.2 的执行冻结）**：同一训练后 checkpoint **不重训**，评测时把材料从句整体删除——删除区间为材料标记（`背景：/线索：/已知：`）之后、回答尾标记（train `答：`、dev `回答：`、final `输出：`；匹配时按 split 取完整尾标记，禁止用 `答：` 误伤 `回答：`）之前的全部字符，保留问题干与尾标记；删除后逐题 prefix 落日志供审计。完整题与删除题的逐题差 = 上下文边际。三臂均做。
3. **A 臂 lesion（评测时确定性施加，不重训）**：
   - `misbind`：V 行相对 K 行按循环移位 L//2 重排（内容全保留、绑定破坏；规则冻结）；
   - `zero_read`：读出向量置零（decoder 输入补零），等价关闭证据访问。
   - 两 lesion 在三臂上同跑（C0 的 4 槽同规则移位），使 lesion 落差可跨臂比较。
4. 生成：现有贪心 `generate()`，boundary 停止，不引入采样。

## §5 门（三段；任一段不过不进入下一段）

### 5.1 实现门（零训练；脚本 `scripts/training/validate_taiji_r2_d2_implementation_gates.py` + 测试 `tests/taiji_native/test_sequence_workspace_contract.py` 扩展）

1. **清单/计数**：A=82,593、C1=82,593、C0=101,025 精确命中；同名参数同 seed 初值逐位相等；`evidence_source` 默认值保持 v2 行为。
2. **单通道结构**：A/C1 renderer 起点仅 `start_vector`；prefix 信息只能经证据读出进入回答流（与 v2 同口径的旁路扫描断言）。
3. **逐位置身份**：合成输入下不同位置 h_i 产生不同 K_i/V_i；把一个标记事实放在指定材料位置时，其对应条目的 key 可由同维查询区分于其余条目；C1 全部条目逐张量相等。
4. **梯度与因果**：CE 对 `evidence_key/evidence_value/prefix_embedding/prefix_recur` 的数值差分有限非零；`zero_read` 时证据投影梯度为零而 renderer 梯度仍非零；因果 mask——未来 response byte 不改变此前 logits；teacher-forced 与逐步前向在同历史下逐位一致。
5. **lesion 接线**：`misbind` 确定性复现（同输入两次移位结果逐位相同）、确实改变读出且不改变 K。
6. **恢复**：零步原子保存→新进程加载→同输入 logits 逐位一致→续训一步与不中断运行一致；v3 载荷含 optimizer/RNG/data_order/cursor/config/evidence_source；v2 旧 checkpoint 在 v2 路径仍可读，版本不匹配显式拒绝。
7. **无上下文构造**：删除规则的合成用例（三 split 尾标记、含干扰句、combo 题各一），断言问题干与尾标记保留、材料标记不残留。

### 5.2 train-only 可学习性 probe（1 运行：A 臂、seed 20260917；脚本 `scripts/training/probe_taiji_r2_d2_learnability.py`）

- 训练前先跑 §5.1-6 的 checkpoint 前置检查（用户硬约束）。
- 通过线：30 epochs 后 train 平均 sequence loss **≤ 0.50** 且 train M1 **≥ 0.90**（train 174 题）。
- 描述性记录 train M3/M4（train 有 8 fact_flip+7 combo_flip 对），不作为门；loss/M1 曲线落报告。
- 不过 ⇒ 回实现/计算图排查，**不加 epoch、不扩数据、不调 lr**；修复后重跑 probe 视为同一门的复验。

### 5.3 matched dev（3 seeds × 3 臂；脚本 `scripts/training/eval_taiji_r2_d2_matched_dev.py`；看 dev 前冻结，不按结果移动）

| 门 | 冻结判据 |
|---|---|
| G1 表面超越 | A 臂 M3 均值 **≥ 0.26** 且 M3_A − M3_C1 **≥ 0.20**（0.26 已含对表面天花板 0.0581 的超越） |
| G2 翻转结构 | A 臂 M4 均值 **≥ 0.50** 且 M4_A − M4_C1 **≥ 0.25**；其中 **fact_flip 子集 ≥ 2/6 对**（金形状标签在该子集为 0/6） |
| G3 内容因果 | A 臂 `misbind` 使 M4 绝对下降 **≥ 0.50**（M4_full − M4_misbind ≥ 0.50） |
| G4 上下文边际 | A 臂 M3_full − M3_no-context **≥ 0.30**，且 A 无上下文 M1 **≤ 0.50** |
| G5 seed 一致性 | M4 上 A>C1 至少 **2/3 seeds**；最差 seed 的 M3_A − M3_C1 **≥ 0** |
| G6 边界与恢复 | 各臂 boundary_stop_rate ≥ 0.95 且 A 不低于 C0 0.05 以上；全部 checkpoint 恢复一致 |

支持假设要求 G1–G6 **全部**满足；任一不过按 §6 路由结案，不补跑同质训练。

## §6 停止线与失败路由（预承诺）

1. 非有限值、参数计数不符、预算超限 ⇒ 立即停，分类为实现/合同缺陷。
2. probe 不过 ⇒ §5.2 的复验路径；两次复验仍不过 ⇒ H-A1 在实现层面不可判定，回实现门并升级人工代码复审，不放预算。
3. matched 任一 G 门不过 ⇒ H-A1 未获支持，按路由结案：
   - M3/M4 仅 M1 升（未知命中）⇒ 内容依赖未形成，假设否决；
   - A≈C1（δ 不达标）⇒ 逐位置信息在该规模非瓶颈，下一假设讨论转 **B（对象绑定）**；
   - fact 对过、combo 对不过且 misbind 落差成立 ⇒ 读取成立、关系计算不成立，转 **C**；
   - lesion 不掉（G3 失败）⇒ 收益不经证据读出，假设否决并检查捷径。
4. 退化（train 崩、恢复不一致、boundary 退化）⇒ 回滚代码，报告保留为负证据。
5. 不读 final；不因负结果重写旧报告或移动判据。

## §7 数值预算（单独冻结；CPU 单线程）

| 阶段 | 运行 | epochs | 数据 | wall cap/运行 |
|---|---|---:|---|---:|
| probe | A × seed 20260917，1 运行 | 30 | D1 train 174 | 20 min |
| matched | A/C1/C0 × 3 seeds = 9 运行 | 30 | D1 train 174 | 30 min/运行 |
| 评测 | 每运行 98 dev × {full, no-context, misbind, zero_read} | — | D1 dev 98 | 含在 30 min 内或另计 10 min/臂 |

合计最坏墙钟上限约 5 小时（实测预计远低；H3.8 同规模每臂为数分钟量级）。checkpoint 写 `reports/r2_d2_checkpoints/` 隔离目录，不触产品 checkpoint。超 cap 即停，按 §6 处理，不自动续时。

## §8 产物（含 format/version/git 身份/corpus digest；`growth_admitted=false`、`can_promote=false`）

1. `reports/r2_d2_implementation_gates_20260918.json`（七项实现门逐项证据）
2. `reports/r2_d2_learnability_probe_20260918.json`（曲线、train M1/M3/M4、恢复核验）
3. `reports/r2_d2_matched_dev_20260918.json`（9 运行逐题记录、M1–M5、lesion、无上下文、G1–G6 判定、参数量与 digest）
4. 全部原始 checkpoint（隔离目录）、无上下文 prefix 审计日志；报告绑定 code revision 与 corpus digest。

## §9 不得主张

全门通过也只支持「D1 合成 dev 上，逐位置证据信息路径相对等机制末状态对照有内容条件收益，且收益经反事实对与 lesion 因果确认」；不构成 L2、不证明真实语料问答、不授权默认核心迁移、不读 final。负结果只否定本假设在冻结预算下的形式，按 §6 路由，不判 VISION 方案 A 整类死刑。
