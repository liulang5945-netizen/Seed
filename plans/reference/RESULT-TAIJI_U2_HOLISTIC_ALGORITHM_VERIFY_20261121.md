# 结果卡：太极 U2 选择性持久状态与组合绑定 5 套件整体验证（`holistic_algorithm_verified`，279.55 CPU，0 check / 0 final）

> 文档状态：[结果卡] — 2026-11-21（D-20261121-225）。在零 check、零 final 读取与完全不改动 [u2_neural_units.py](../../taiji/u2_neural_units.py) 的前提下，先以 5 套件整体验证新模块 [u2_selective_units.py](../../taiji/u2_selective_units.py) 与动作绑定口径，再推进下游分支。

## 1. 审计根因与加法修复面

1. **批-当-时串扰（`batch_as_time_bleed`）**：原 `NeuralUnits.forward` 按 dim-0 步进共享一维 `z/a:[N]`，`RouteBGraph` 批大小直接当时间步互混，导致静态单步训练把保留率 `c` 压向 `0`（3 种子均值 `0.0319`、最大 `0.2769`）。新模块 [SelectiveNeuralUnits](../../taiji/u2_selective_units.py) 强制要求四维 `[B, T, G, D]` 输入与独立每样本 `[B, N]` 状态，`per_sample_state=False` 默认严格逐位委派旧图（`max_abs_diff = 0.0`）。
2. **零跨群复现边与单通道编码孤立**：原 `make_identities` 的跨群边数恒为 `0`。`SelectiveNeuralUnits(cross_group_edges=True)` 恢复 `750` 条跨群连通，并加入 `cross_channel_mix` 让 6 类合法输入跨群协同。
3. **正偏置写入门梯度坍缩**：负写入偏置 `init_write_bias = -2.0`（保留优先门控 `z = (1 - g_w) z + g_w tanh(v)`）在 `T=20` 的 `t=0` 输入梯度比 `0.0` 偏置实测放大 **20× 以上**。
4. **M3 中间发现与末步读出选择门（`read_gate`）**：首轮未加 `read_gate` 时（[初始诊断记录](#3-m3-两阶段消融与架构补充发现)），`MLP(state_final)` 在多键查询 `M3` 上卡在 `0.490 / 0.573 / 0.551`（在线流 `0.594 / 0.582 / 0.582`）。合入仅看 `t = T - 1` 全部 48 维合法输入（零历史 KV 缓存、无任务硬编码通道）的通用乘性读出门 `read_gate`（bias=+1.0）后，`M3` 容量按 `N=2/8/32` 单调升至 `1.000`。

## 2. 整体验证结果（3 种子均值与门禁，`cpu = 279.55`，`CAP_CPU = 1200.0`）

报告：[reports/taiji_u2_holistic_algorithm_verify_20261121.json](../../reports/taiji_u2_holistic_algorithm_verify_20261121.json)，运行条件 `threads=1`、`OMP_NUM_THREADS=null`、`MKL_NUM_THREADS=null`、`seeds=[20261120, 20261121, 20261122]`。

| 套件 | 指标 / 臂 | 3 种子结果 | 各门禁状态 |
|---|---|---|---|
| **Suite 1: 架构缺陷审计** | 旧图初值批串扰 max | `0.0300`（`0.0300 / 0.0238 / 0.0138`） | 复现 ✓ |
| | 训练后三技能父批串扰 max | `4.2854`（`1.5238 / 4.2854 / 3.1630`，`mean_c=0.0319`） | 复现 ✓ |
| | 跨群边数（旧 → 新） | `0` → `750` | `arch_cross_group_restored = true` |
| | 默认 `per_sample_state=False` 逐位差 | `0.0` | `arch_default_bit_identical = true` |
| | 开启后样本间串扰 max | `0.0` | `arch_batch_isolation_fixed = true` |
| **Suite 2: M1 选择性保持** | `T=5` (`static_c` vs `selective`) | `0.9831` vs **`1.0000`** | `m1_hold_t5 = true` |
| （随机 `t_target`, 仅 `t=T-1` 读出） | `T=10` (`static_c` vs `selective`) | `0.8320` vs **`1.0000`** | `m1_hold_t10 = true` |
| | `T=20` (`static_c` vs `selective`) | `0.4219` vs **`1.0000`**（`+0.5781`） | `m1_hold_t20` / `m1_beats_static_c = true` |
| | 旧三技能损伤 / `capsule_v2` 往返差 | `0.0` / `0.0` | `old_skill_damage_zero` / `capsule_v2_roundtrip_exact = true` |
| **Suite 3: M2 中途覆盖 (`T=12`)** | `static_c_state_final`（旧错率） | `0.8906`（`old_err=0.0195`） | — |
| | `selective_state_final`（旧错率） | **`1.0000`**（`old_err=0.0`） | `m2_overwrite_pass = true` |
| **Suite 4: M3 多项延迟查询** | `N=2` / `N=8` / `N=32`（`T=12`, 在线流） | `0.4557` / `0.7057` / **`1.0000`** | `m3_capacity_scales = true`（`+0.5443`） |
| **Suite 5: B1 组合动作绑定** | `flat_mlp`（5 训 / 2 留出已见字） | `0.9349`（`0.9805 / 0.9648 / 0.8594`） | 零未见字 `[]` |
| | `compositional_slot_binding` | `0.9323`（`0.9492 / 0.9453 / 0.9023`） | `b1_compositional_ge_080 = true` |
| | `selective_unit_binding` | `0.9128`（`0.8672 / 0.8984 / 0.9727`） | `b1_unit_binding_ge_080 = true` |

总判词：**`verdict = "holistic_algorithm_verified"`**（13/13 预注册门全过）。

## 3. M3 两阶段消融与架构补充发现

1. **首轮静态 `M3`（无 `read_gate`，225.75 CPU）**：`2: 0.4896 / 8: 0.5729 / 32: 0.5508`（12/13 通过，唯 `m3_capacity_scales = false`）。消融证实两点：① 1,536 样本静态集上 `N=32` 训练精度达 `1.0000` 而泛化停在 `0.5313`（音频噪声过拟合）；② 换用在线不相交流消去过拟合后，纯加性 `MLP(state_final)` 仍停在 `2: 0.5938 / 8: 0.5820 / 32: 0.5820`。
2. **引入通用 `read_gate` 后单调缩放恢复**：仅在 `t = T - 1` 增加全通道乘性选读门（诊断 `2: 0.4609 -> 8: 0.6172 -> 32: 1.0000`；正式 3 种子 `2: 0.4557 -> 8: 0.7057 -> 32: 1.0000`），证明多键同时保存需要**“写入保护 + 末步乘性选读”**双门机制，而静态单一衰减率 `c` 或仅改 `[N]→[B,N]` 形状均不足以解决。

## 4. 边界与明确非声明

- 本卡与 [u2_selective_units.py](../../taiji/u2_selective_units.py) 验证的是受控合成基准上的单位级选择性持久状态与组合绑定机制；**不声称**模型整体成型、开放自然语言能力、真机动作闭环、结构生长收益（25/26/27 维持暂停）或通过 check / final / C / D / J。
- [u2_neural_units.py](../../taiji/u2_neural_units.py) 字节零改动；历史 22.5.1 / 22.7 / v1 / w11 契约测试全数未受影响。
