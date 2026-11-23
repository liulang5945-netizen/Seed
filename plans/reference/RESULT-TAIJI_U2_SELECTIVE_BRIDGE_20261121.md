# 结果卡：太极 U2 选择性持久状态四轴能力桥接验证（D-20261121-227，`16/17` 门通过，`485.27 CPU`，0 check / 0 final）

> 文档状态：[结果卡] — 2026-11-21（D-20261121-227）。在**零 check、零 final 读取**与完全不改动 [u2_neural_units.py](../../taiji/u2_neural_units.py) 的前提下，基于 [u2_selective_units.py](../../taiji/u2_selective_units.py) 运行四轴桥接验证脚本 [verify_taiji_selective_bridge_20261121.py](../../scripts/training/verify_taiji_selective_bridge_20261121.py)，报告位于 [taiji_u2_selective_bridge_20261121.json](../../reports/taiji_u2_selective_bridge_20261121.json)。

## 1. 四轴总览与 17 项预注册门禁（3 种子均值，`cpu = 485.27 <= 1200.0`）

运行条件：`threads = 1`、`OMP_NUM_THREADS = null`、`MKL_NUM_THREADS = null`、`seeds = [20261120, 20261121, 20261122]`、`fit_offset = 10000000`、`m3_steps = 1000`、`growth_steps = 400+600+900=1900`。

| 轴向 | 指标 / 对照臂 | 3 种子均值 | 三种子明细（`20261120 / 20261121 / 20261122`） | 预注册门禁（不松弛） |
|---|---|---|---|---|
| **Bridge 1: 等 FLOPs 架构对比** | `SelectivePersistentGraph` (`N=32`, `216,832` FLOPs) | **`1.0000`** | `1.0000 / 1.0000 / 1.0000` | `b1_flops_parity = true`、`b1_selective_ge_090 = true` |
| （`M3`, `T=12`, 在线 `1,000` 步） | 等 FLOPs `GRU` (`H=35`, `218,352` FLOPs, 比 `1.0070`) | `0.5299` | `0.5313 / 0.5352 / 0.5234` | `b1_non_inferior_to_gru = true`（`+0.4701`） |
| | 等 FLOPs 因果 `Transformer` (`d=24,h=4,ff=100`, `215,552`, 比 `0.9941`) | `1.0000` | `1.0000 / 1.0000 / 1.0000` | 比值在 `[0.90, 1.10]` 内 |
| **Bridge 2: 两代中性生长与 H2 损伤** | 曲线 `g0 -> g1 -> gen2`（功能单位 `4 -> 12 -> 28`，总孪生 `52`） | `0.5508 -> 0.8607 ->` **`1.0000`** | `g1: 0.9414 / 0.9258 / 0.7148`；`gen2: 1.0 / 1.0 / 1.0` | `b2_growth_beats_fixed_n4 = true`、`b2_gen2_ge_090 = true` |
| （冻结成熟父代 + 冻结出生锚点残差） | 等步对照 `G5_fixed_n4` / `G6_prealloc_n32` (`1,900` 步) | `0.5273` / `1.0000` | `G5: 0.5352 / 0.5547 / 0.4922`；`G6: 1.0 / 1.0 / 1.0` | 相对 `G5` `+0.4727`（门 `>= +0.20`） |
| | 出生漂移 `birth_diff_max` / 首步新生梯度 `min` | **`0.0`** / `0.000331` | 梯度 `0.000331 / 0.001780 / 0.001145` | `b2_birth_neutral = true`、`b2_newborn_alive = true` |
| | H2 可逆休眠损伤 `lesioned` / `drop` / `restored_diff` | `0.5508` / **`+0.4492`** / **`0.0`** | `drop: 0.4648 / 0.4141 / 0.4688` | `b2_lesion_causal = true` |
| | 旧三技能损伤 `max` / 多代 `capsule_v2` 往返差 `max` | **`0.0`** / **`0.0`** | 全种子 `0.0` / 全种子 `0.0` | `b2_old_skill_damage_zero = true`、`b2_capsule_v2_exact = true` |
| **Bridge 3: 移位不变因果中文** | Exact `A3c` (`5` 训 / `5` 留出): `pos_attn_raw` vs 因果 `Selective` | `0.5552` vs **`0.8031`** | `0.8813 / 0.8250 / 0.7031`（`pos_attn: 0.6531 / 0.5156 / 0.4969`） | `b3_a3c_exact_ge_080 = true`、`b3_beats_pos_attn = true`（`+0.2479`） |
| （无任何位置嵌入） | 去混淆双向规则（`12` 训 / `8` 留出，未见字 `[]`） | `0.4187` vs `0.7094` | `0.6656 / 0.7125 / 0.7500`（`pos_attn: 0.4688 / 0.3562 / 0.4313`） | **`b3_bidirectional_ge_080 = false`**（`0.7094 < 0.80`） |
| | 对抗范围探针 `a3c_bi_adversarial_scope`（新词冲突诊断） | `0.6115` | `0.6281 / 0.5969 / 0.6094` | 诚实有界作用域记录 |
| **Bridge 4: 二值正误动作闭环** | Turn 0 初始（颜色故意含糊）→ Turn 1 带反馈 | `0.4648` → **`0.8984`** | `t1: 0.9297 / 0.9023 / 0.8633` | `b4_turn1_ge_085 = true` |
| （仅 `1/0` 对错信号，零答案泄漏） | 无反馈 `no_fb` / 置换反馈 `permuted_fb` | `0.4440` / `0.4596` | `no_fb: 0.4570 / 0.4609 / 0.4141`；`perm: 0.4453 / 0.4883 / 0.4453` | `b4_gain_over_no_fb = true`、`b4_gain_over_permuted_fb = true` |
| | 错后纠错率 `error_correction_rate` | **`0.8138`** | `0.8636 / 0.8188 / 0.7589` | `b4_correction_rate_ge_080 = true` |

总判词：**`16/17` 项门禁通过**（Bridge 1 `3/3`、Bridge 2 `7/7`、Bridge 3 Exact `A3c` `2/2`、Bridge 4 `4/4`）；仅 **`b3_bidirectional_ge_080 = false`**（按预注册规则**未放宽** `0.80` 门禁），故脚本严格输出 `verdict = "selective_bridge_partial"`。

## 2. 过程机制发现与工程修复实录

1. **双线性鞍点阻滞与对偶/冻结出生锚点**：早先用可训练「零出边 + 零读出权重」扩展单元且 `base_logits` 已先收敛时，梯度在两层零初始化处陷入双线性鞍点（`0.527–0.552` 停滞；而单独「零出边」与单独「已收敛父类」各自均可达 `1.000`）。改用反对称子群 `pos - neg` 后消除了零出边鞍点并保持 `birth_diff = 0.0` 与非零首步梯度（[test_u2_selective_units.py](../../tests/taiji_native/test_u2_selective_units.py)）。
2. **「解释消去（explaining-away）」写门饥饿与不成熟期局部监督**：跨三颗种子验证发现，仅让双子均开放梯度且受全局 RNG 漂移影响时，2/3 种子会在 `g0` 已先输出高置信 logits（`|lo0| ≈ 3.7`、精度 `~0.55`）的压制下发生新生写入门梯度饥饿；而一旦跳过 `g0` 预训练，同样的新生子群在 `20261121 / 20261122` 上均可直接达到 `1.000`。最终在 `_AntitheticSelectiveCohort` 中固定两项机制后三颗种子全数达 `1.000`：
   - **确定性种子化与冻结出生锚点（`freeze_anchor=True`）**：线性层以 `torch.random.fork_rng` + `cohort_spec.seed` 确定性初始化，复制后的负支 `neg_*` 永久冻结为静止出生参照锚（`delta = pos_out - no_grad(neg_out)`）。
   - **仅限 `immature` 期的发育局部辅助损失（`lambda = 0.5`）**：训练不成熟子群时叠加 `0.5 * CE(pos_out, y)`；一旦调用 `promote_cohorts_to_mature_()` 转入 `mature` 期以及在全部推理阶段，辅助损失自动归零（不影响休眠损伤、成熟期前向与最终预测）。
3. **双向中文边界诚实落账**：在剔除 `"后说"` 词面混淆后，`SelectivePersistentGraph`（零位置嵌入）在双向留出规则上较原始 `PosAttentionRead` 提升 `+0.2906`（`0.7094 vs 0.4187`），但未跨过 `0.80` 预注册门禁（含明确新词冲突对 `a3c_bi_adversarial_scope = 0.6115`），明确标定其目前仍是有界语言范围能力。

## 3. 边界、成本与非声明

- **零额外曝光与契约不变**：`check_reads = 0`、`final_reads = 0`；实际 CPU 耗时 **`485.27 <= 1200.0`**；`git diff -- taiji/u2_neural_units.py` 保持为空（所有 `22.5.1`、`22.7`、`v1`、`w11`、`b_parent` 契约逐位不变）。
- **明确非声明**：本卡**不声称**模型整体成型、**不声称**开放自然中文双向理解达标（`b3_bidirectional_ge_080 = false` 维持未过）、**不声称**真机多步控制、**不重开** 25/26/27（若重开结构线仍须先独立通过真实结构压力任务卡），且不触碰 `check` / `final` / `C` / `D` / `J`。

## 4. 补注（D-20261123-329，不改上文读数）

Bridge 1 的 `+0.4701` 差距经[公平基线诊断](RESULT-TAIJI_U2_FAIR_BASELINE_M3_20261123.md)确认来自末步 `read_gate`：GRU 加同一门后 3/3 seed `1.0000`，Selective 去门后 `0.5156` ≈ GRU `0.5299`。本卡 Bridge 1 的「非劣于/优于 GRU」只在「GRU 无读门」条件下成立；「单元动力学优于 GRU」未证实。Bridge 2–4 读数不受影响。
