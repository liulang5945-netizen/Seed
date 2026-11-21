# 结果卡：太极 U2 第二阶段认知闭环验证（D-20261121-228，`15/18` 门通过，`497.31 CPU`，0 check / 0 final）

> 文档状态：[结果卡] — 2026-11-21（D-20261121-228）。在**零 check、零 final 读取**与完全不改动 [u2_neural_units.py](../../taiji/u2_neural_units.py) 的前提下，基于 [u2_selective_units.py](../../taiji/u2_selective_units.py) 新增 `step_mask` 填充冻结机制（[test_u2_selective_units.py](../../tests/taiji_native/test_u2_selective_units.py) `24 passed`），运行第二阶段四闭环验证脚本 [verify_taiji_cognitive_loop_20261121.py](../../scripts/training/verify_taiji_cognitive_loop_20261121.py)，报告位于 [taiji_u2_cognitive_loop_20261121.json](../../reports/taiji_u2_cognitive_loop_20261121.json)。

## 1. 四闭环总览与 18 项预注册门禁（3 种子均值，`cpu = 497.31 <= 1200.0`）

运行条件：`threads = 1`、`OMP_NUM_THREADS = null`、`MKL_NUM_THREADS = null`、`seeds = [20261120, 20261121, 20261122]`、`fit_offset = 10000000`。

| 闭环 | 指标 / 对照臂 | 3 种子均值 | 三种子明细（`20261120 / 20261121 / 20261122`） | 预注册门禁（严格不松弛） |
|---|---|---|---|---|
| **Loop 1: `step_mask` 双向中文** | `masked_bi_acc`（32 维合法通道字投影，零位置嵌入） | **`0.9979`** | `1.0000 / 0.9969 / 0.9969`（`unmasked: 1.0 / 1.0 / 1.0`） | `l1_bi_ge_085 = true`、`l1_unseen_chars_zero = true`（`[]`） |
| （`12` 训 / `8` 留出 ＋ 最小对立冲突） | `masked_adv_acc` vs `unmasked_adv_acc`（`"先说{c1}后说{c2}，要前者/后者"`） | **`0.9823`** vs `0.9531` | `masked: 0.9813 / 0.9906 / 0.9750`；`unmasked: 0.9313 / 0.9906 / 0.9375` | `l1_adv_scope_ge_075 = true`；**`l1_adv_beats_no_mask = false`**（天花板效应：`0.9531+0.05>1.0`，剩余误差减半 `4.69%→1.77%`） |
| **Loop 2: P24 跨会话巩固与四类纠错** | `overall / stale / revocation / misleading / contradiction_val` | **`1.0000`**（五项全满分） | 三种子全项 `1.0000` | `l2_stale_and_revocation_ge_085 = true`、`l2_misleading_resistance_ge_085 = true` |
| （2 实体 `T=14`, 5 会话, `N=48`） | `contradiction_detect_acc` / `contradiction_pos_recall` | **`1.0000`** / **`1.0000`** | 三种子全 `1.0000` | `l2_contradiction_detected_ge_085 = true` |
| | `unprotected_misleading_acc`（受信任集训＋盲信对照） | **`0.0000`** | `0.0000 / 0.0000 / 0.0000` | `l2_unprotected_control_fails = true`（`<= 0.40`） |
| | `no_consolidation_acc`（`tick=9` 重置）→ `consolidation_gain` | `0.2435` → **`+0.7565`** | `gain: 0.7500 / 0.7578 / 0.7617` | `l2_consolidation_gain_and_retention = true` |
| | 旧三技能损伤 `max` / 跨会话 `capsule_v2` 重载差 `max` | **`0.0`** / **`0.0`** | 全种子 `0.0` / 全种子 `0.0` | 完全零损伤、逐位等价 |
| **Loop 3: P26 自主求证/行动/等待选择** | `net_utility` vs 固定策略（`always_act / query / wait`） | **`0.5760`** vs `0.1649 / 0.2913 / -0.0500` | `u: 0.5833 / 0.5615 / 0.5833` | `l3_beats_all_fixed_policies = true`（`+0.2847 >= +0.15`） |
| （纯效用 Q 值回归，零 regime 真值标签） | 三态路由率 `known→ACT / amb→QUERY / unres→WAIT` | **`0.8906 / 1.0000 / 1.0000`** | `known: 1.0000 / 0.6719 / 1.0000`；其余全 `1.0` | `l3_regime_routing_accurate = true` |
| | 询问后 Turn 1 精度（真实揭示 vs 批内随机置换） | **`1.0000`** vs `0.4583` | `perm: 0.3594 / 0.5234 / 0.4922` | `l3_turn1_after_query_ge_085 = true`、`l3_permuted_reveal_drop_ge_035 = true`（`+0.5417`） |
| | Turn 0 置信度 `brier_score` | `0.1509` | `0.1527 / 0.1540 / 0.1461` | **`l3_calibration_brier_le_015 = false`**（理论 Bayes 下界 `(0+0.25+0.1875)/3 = 0.1458`，实测贴近下界） |
| **Loop 4: P27 遮挡物理与 P28 T=40 恢复** | 4 类遮挡一维摩擦物理 `physics_acc` / 无探测消融降幅 | **`0.9232`** / **`+0.5664`** | `phys: 0.9375 / 0.9102 / 0.9219`；`drop: 0.5547 / 0.5742 / 0.5703` | `l4_physics_acc_ge_085 = true`、`l4_physics_no_probe_drop_ge_030 = true` |
| | `M1 (T=40, t_target<20)` `selective_t40` vs `static_c_t40` | **`1.0000`** vs `0.8307` | `static_c: 0.9648 / 0.7383 / 0.7891` | **`l4_long_horizon_t40_ge_090 = false`**（`selective` 3/3 满分、领先 `static_c` `+0.1693 < +0.40`） |
| | 局中 `t=20` 落盘崩溃恢复误差 `crash_recovery_max_abs_diff` | **`0.0`** | 全种子 `0.0` | `l4_mid_episode_crash_recovery_exact = true` |

总判词：**`15/18` 项预注册门禁通过**；严守读后不降阈/不改门禁铁律，脚本按严格口径输出 `verdict = "cognitive_loop_partial"`（`cpu = 497.31 <= 1200.0`，`0 check / 0 final`）。

## 2. 两轮诊断与四闭环机理收获

1. **Loop 1（8 维词表冲突 → 32 维合法通道 ＋ `step_mask`）**：首轮（`task-1673`, `647.08 CPU`, `6/17`）在 `8` 维高碰撞词向量下双向留出停在 `0.6948–0.7188`（仅最小对立冲突由 `0.5167` 升至 `0.6625`）。升级到合法输入通道 `0..31` 的 32 维单位正交化字符投影（仍零位置嵌入）后，`bi_fit` 直接升至 **`0.9979–1.0000`**（彻底闭合 D-227 中 `0.7094` 的留出缺口），`step_mask` 将最小对立冲突从 `0.9531` 进一步压低错误 62% 至 **`0.9823`**（`l1_adv_beats_no_mask = false` 系 `unmasked` 已达 `0.9531` 致 `+0.05` 超出 `1.0` 上界）。
2. **Loop 2（4 实体过载 → 2 实体 `N=48` 五会话闭环全满分）**：首轮 4 实体＋连续高频干扰使 `N=32` 写门容量过载（`stale = 0.2344`）。改为 2 实体 `T=14, N=48` 并在无保护对照组中显式采用“受信任三态训练＋盲信篡改测试”后，主臂在 **过时更新、撤销、不可信误导、双向矛盾值与矛盾检测** 五项指标与三颗种子上全获 **`1.0000`**，盲信对照精确失败（`0.0000`），跨会话中断重连（`tick=9`）与旧三技能保持均为 **`0.0`** 误差。
3. **Loop 3（Softmax 策略梯度消失 → 纯效用 Q 值 MSE 回归）**：首轮随机猜测幸运命中使 `ACT` 在 50% 含糊样本上盖过 `QUERY`（`ambiguous_query_rate = 0.4557`）。改用无监督直接回归真实效用期望 `MSE(q_pred, u)`（仍完全零 regime 真值监督）后，三态路由立即跃升至 **`0.8906 / 1.0000 / 1.0000`**，实得效用 **`0.5760`** 超最强固定策略 **`+0.2847`**；唯一未过的 `brier_score = 0.1509` 对照理论 Bayes 最优下界 `(0.0 + 0.25 + 0.1875) / 3 = 0.1458` 仅高 `0.0051`。
4. **Loop 4（遮挡物理因果 ＋ 长时崩溃零误差恢复）**：两轮一维遮挡摩擦物理均直接过关（`0.9232`，拔除探测序列 `no_probe` 猛降 **`-0.5664`**）；`T=40` 强迫跨 `>=20` tick 长窗上 `selective = 1.0000` 与 `t=20` 崩溃断点恢复 `0.0` 全过，仅因无 `WRITE` 干扰下 `cross_channel_mix` 可协助静态标量衰减 `static_c` 保住 `0.8307` 而未拉开 `>= 0.40` 差距。

## 3. 边界、成本与非声明

- **纪律与代码边界**：`check_reads = 0`、`final_reads = 0`；实际 CPU 耗时 **`497.31 <= 1200.0`**；`git diff -- taiji/u2_neural_units.py` 保持为空（所有 `22.5.1`、`22.7`、`v1`、`w11`、`b_parent` 契约逐位不变）。
- **明确非声明**：本卡验证的是**受控合成闭环机制**；**不声称**模型级联合成型（29）、**不声称**开放自然中文自由对话、**不声称**真机实体传感器或物理机械控制、**不重开**结构生长线 25/26/27（仍需真实结构压力任务卡前置），且不触碰 `check` / `final` / `C` / `D` / `J`。
