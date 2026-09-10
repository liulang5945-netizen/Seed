# M5.K 信号空间扩展预注册（visible-signal-space expansion, v4）

> 冻结日期：2026-09-10。前置：[parity v3 预注册 §6](M4V2_B3_K_C_PARITY_V3_PREREGISTRATION_20260910.md)（widened 路线关闭 + 学习问题收敛点：可见类空间只有 3 类）。本文冻结 K 课程 harness 的可见信号空间扩展设计；扩展改变 worker 词汇与父代 artifacts，属**父代重建**级变更，冻结后按 §6 顺序执行。执行顺序以 [03_CURRENT_EXECUTION.md](../active/roadmap/03_CURRENT_EXECUTION.md) 为准。

## 1. 可证伪假设

> v3 formal 的两种合成（顺序分化权重平均 vs replica 概率平均）在 3 类可见空间内只能产生二阶差异（±0.0002）。**若可见类空间从 3 类扩展到 5 类（新增 clarify-language 与 recover 两类），同等预算下两种合成的 sealed 泛化差异将超出二阶带**——即学习规则差异此前被信号空间不足掩盖。若扩展后两合成仍在 ±0.0002 带内，则「合成方式在该 harness 内不可分」被确证，学习规则比较让位给信号工程。

## 2. 词汇扩展规格（唯一变更集）

| 词汇 | 现状 | 扩展后 | 使能来源 |
|---|---|---|---|
| K1/K2 fact keys | 13（含 `language_state::{resolved,unknown}`） | **14**（+ `workbench::language_state::ambiguous`） | `.h` 文件在 c/cpp 扩展平票下解析为 ambiguous（探针已验证） |
| K1 goal/content ids | 3（clarify-toolchain / inspect-language / recover-target） | **4**（+ `clarify-language`） | `.h` 的 `_semantic_kind` = clarify-language |
| K2 goal/content ids | 2（clarify-toolchain / inspect-language） | **4**（+ `clarify-language` + `recover-target`） | D/R 类 experience 进入 K2 训练 |

- 类型化掩码**自动覆盖**新事实：`_typed_fact_feature_masks` 对 `language_state::ambiguous` 映射到 schema 的 `selection:ambiguous` 列（已在 schema.selection_states）；K2.1 转移掩码同源派生。无手写掩码。
- **不变**：K3 deterministic projector、类型化掩码机制、K1.1 readout 排除（身份 facts 剔除）、局部 delta 学习率（semantic 2.0 / transition 0.2）、无 optimizer state。

## 3. 扩展后的可见类空间（3 → 5 类）

| 类 | 首文件 profile | goal/content | 使能 |
|---|---|---|---|
| A | python / resolved / toolchain **available** / read success | inspect-language | 现有 |
| B | rust / resolved / toolchain **missing** / read success | clarify-toolchain | 现有 |
| C | typescript / resolved / **missing** / read success | clarify-toolchain | 现有 |
| **D** | **unknown / ambiguous** / missing / read success（`.h`） | **clarify-language** | ambiguous fact + clarify-language 词汇 |
| **R** | **unknown / unknown / read failure / target missing**（missing 首 file） | **recover-target** | K2 goal 词汇扩展 |

诚实边界：`.h` 类在 planner 侧会被 `language_evidence_ambiguous` 拒绝（resolve 能力要求 resolved），但 parity 比较的量尺是 holdout **MSE**（非执行成功率），D 类可训练可评分；该 planner 边界如实记录，不构成本预注册的阻碍。

## 4. 父代重建合同（词汇变更 ⇒ 父代重建）

- **重建输入**：冻结的 M5.K2 训练 episodes + 新增 D 类（`.h`）与 R 类（missing 首 file）episodes——词汇只能来自训练语料，父代必须见过全部 5 类；
- **重建产物**：新 K1/K2 worker artifacts，落盘 `checkpoints/taiji_k_workers_v4/model_{17,23,31}/`（不覆盖现有 `taiji_k_workers`）；训练 recipe 与 R6 worker build 相同（local delta、每 experience 1 epoch、记录 training_steps）；K3 deterministic projector **不变**（复用现有 checkpoint）；
- **model 维度说明**：现有证据已证三个 model seed 的 parent K workers 权重逐位相同（state_dict diff=0）——重建后预期仍相同，model 维度继续作为 replicate 而非独立样本，统计单元 = course；
- **量尺前置**：重建后 fresh restore、owner digest、参数清单全部落盘；三 model 的重建 worker 权重一致性作为观察项报告（不作为门——若不再逐位相同反而是新信息）。

## 5. 容量核算（parity 由构造保证）

词汇 13→14 使 K1 每通道 405→约 472 参数、K2 每通道 4428→约 5176 参数（builder 落盘精确核算表为准）。双臂（widened 双通道 / fixed-large 双 replica）共享同一 worker 形状 → 参数字节自动 parity（预计 bundle 2×5648 = 11296 参数 ≈ 45,184 字节）；±1% 门保留作核验。

## 6. 重跑合同（扩展空间内的合成比较）

1. **equal-new-budget 课程**：每 cell 150 experiences，5 类平衡（各 30），类块内排列随 course_seed 变化（沿用 v2 的课程独立性设计与硬门）；双臂同流，new_update_steps = 150×2×2 = 600/臂/cell；
2. **双臂**：candidate = widened 双通道 + **权重平均** readout（v3 冻结语义，权重相加已被否决不复用）；control = fixed-large 同构 replica **概率平均**；均从重建父代出发；
3. **fresh sealed v3**：`materialize` 新 sealed（task_seed 未用过、与 train/validation/sealed v1/v2 全不交，materializer 强制校验）——sealed v2 已被 v3 formal 读取，不可复用；
4. **判据**：G1（validation 质量门）/ G2（灾难界，validation 派生先行冻结）/ G3（sealed 课程级 ≥2/3 胜出且均值更优）——结构与 v1/v3 formal 相同，两阶段纪律（pre-sealed 落盘后才读 sealed）不变；
5. **可证伪点**：G3 通过 = 信号空间扩展使合成差异可测，学习规则比较重新开放；G3 失败 = 「合成方式在该 harness 内不可分」确证，学习规则比较让位给信号工程（D 阶段前的信号工程收束）。

## 7. 停止线

- 父代重建后任何 fresh restore/owner/词汇校验失败 → 停；
- 5 类课程下通道差分仍出现精确 0（等价类守卫已内建，若仍发生 → 归因生成多样性）→ 停；
- sealed v3 读取后禁止任何判据/词汇/合成调整；
- 不引入 provider/联网/真实客户端写入/default runtime；`can_promote=false` 固定。

## 8. 产物顺序

1. 父代 worker 重建脚本 + 词汇扩展（K1/K2 corpus 词汇随 experience 自然生长，无硬编码清单）+ 参数核算表 + fresh restore Gate；
2. 扩展空间 equal-new-budget 双臂 build（5 类 × 150、课程独立性门、通道差分门）；
3. sealed v3 materialize + formal v4（G1–G3）；
4. 全部通过/失败均如实落盘；路线图执行记录 + 独立提交。

## 9. 明确不做

- 不改 K3、不改类型化掩码机制、不改学习率/优化语义；
- 不在 3 类空间内继续调课程/合成/阈值；
- 不把 planner 的 `language_evidence_ambiguous` 拒绝边界放开（执行安全性边界不动）；
- 不做 diagnostics-connected 变体（harness 无真实 LSP 来源，fabricated 信号禁止）；
- 不做多信号空间 formal 之外的散点实验。
