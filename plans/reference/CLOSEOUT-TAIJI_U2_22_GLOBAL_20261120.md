# 包 22 全局收束卡（22.8.4）：九轴资格表与 23 立项依据

> 文档状态：[收束卡] — 2026-11-20（D-20261119-160）。上游：[22.8 S0](DESIGN-TAIJI_U2_22_8_JOINT_LIFE_20261120.md)、[S1](RESULT-TAIJI_U2_22_8_S1_20261120.md)、[S2](RESULT-TAIJI_U2_22_8_S2_20261120.md)、[S2b](RESULT-TAIJI_U2_22_8S2B_ISOLATION_20261120.md)、[S2b2](RESULT-TAIJI_U2_22_8S2B2_RETENTION_20261120.md)、[J1](RESULT-TAIJI_U2_22_8J1_DOSE_20261120.md)、[J3c](RESULT-TAIJI_U2_22_8J3C_MLP_20261120.md)。**零训练、零 check/final。**

## 1. 一句话结论

**包 22 按 `package22_closed_partial` 全局收束**：22.5.1 已正式取得 `joint_gate_check2_pass / b_parent_earned`；22.5.2 架构优越性 `arch_negative`（范围限定）；22.6 `autonomous_choice_negative / recursive_negative`；22.7 机制/因果 verified ＋ 收益 `growth_benefit_absent`（等 FLOPs）；22.8 联合生命片段 **partial**（J1/J2/J4/J6 有界，J3 partial-positive，J5 negative，J7 退化）。**`final` 未获授权、未运行（`final_not_authorized`）**；不新增 22.9；本卡输出九轴资格表与 23 立项依据。

## 2. 22.5–22.8 判词总账

| 包 | 判词 | 证据 |
|---|---|---|
| 22.5.0 | A 父（pixel-salience, seed 20261112）`a_parent_earned` | D-04 / `taiji_u2_22_5_0_a_parent_earned_20261119.json` |
| 22.5.1 | **`joint_gate_check2_pass` / `b_parent_earned`** | [正式 check② 结果卡](RESULT-TAIJI_U2_22_5_1_FORMAL_CHECK2_20261120.md)、D-144 |
| 22.5.2 | **`arch_negative`**（等 FLOPs，范围限定） | [结果卡](RESULT-TAIJI_U2_22_5_2_ARCH_COMPARISON_20261120.md)、D-147 |
| 22.6 | **`autonomous_choice_negative / recursive_negative`**（范围限定） | [结果卡](RESULT-TAIJI_U2_22_6_CLOSEOUT_20261120.md)、D-148 |
| 22.7 | **机制/因果 verified ＋收益 `growth_benefit_absent`（等 FLOPs）** | [结果卡](RESULT-TAIJI_U2_22_7_WINDOW_20261120.md)、D-151 |
| 22.8 | **`joint_life_segment_partial`** | 本卡 §3、D-148/156/157/158/159 |

## 3. 22.8 联合生命片段分账（终值）

| 门 | 结论 | 读数 |
|---|---|---|
| J1 appearance | **`appearance_dose_verified`（有界）** | 1,152 步 3 seed fit 0.9063/0.8438/0.8125（多数类 0.0703） |
| J2 dynamics/action | **有界 verified** | 0.69–0.80，动作边际 0.36–0.44（mask 后崩回 0.33–0.41） |
| J3 correction | **partial-positive** | 参数隔离下纠正可学（+0.125–0.500）、不写冻结基座；旧保持因 **4 维 `direct_query`** 不可检验（线性 0.042–0.167、MLP 0.208–0.250） |
| J4 recovery | **有界 verified** | 3/3 session capsule 恢复摘要 diff 0.0 |
| J5 no-prompt | **range-limited negative** | 效用恰等最强固定 0.9175、空转 0（与 D-60/123/124 同型） |
| J6 breakdown | **齐全** | 五项分账均有读数 |
| J7 occlusion | 退化任务 | 全遮挡→未知，仅记录，不签能力 |

## 4. 九轴资格表（同一可追溯主体谱系；严禁跨 seed 拼装）

| 轴 | 状态 | 范围与证据 |
|---|---|---|
| `engineering` | **verified** | 22.3/22.4 工程父；22.8 S2/S2b/S2b2/J3b/J3c capsule 恢复 diff 0.0、冻结单位梯度 0 |
| `task` | **verified** | 22.5.1 正式 check②（A 174/192 drop 0.0、B 144/144、D ub95 0.0）；22.8 J1 fit 面 0.81–0.91（check 未读） |
| `learning` | **routing verified ＋ 通路新增有界 verified；内容积累 uncovered** | 22.5.1；D-78/79/83/99（3→9 任务）；22.8 J1 剂量有界 |
| `retention` | **verified（check② 范围）＋ 22.8 J3 partial-positive** | 22.5.1 A drop 0.0、D median ub 0.0（披露 20261110 ub95 0.0208、overwrite drop 0.875）；J3 旧保持受 4 维 direct 限制不可检验 |
| `autonomous_choice` | **negative（C 任务/C 池）＋平稳合成有界 verified** | 22.6.2/2b、D-60/123/124；D-126/127 平稳合成；D-128/129 非平稳/延迟 negative |
| `mature_growth` | **机制/因果 verified ＋ 收益 `growth_benefit_absent`（等 FLOPs）** | D-151：G0 0.6425 vs 预分配强控制 0.6750（Δ=−0.0498、下界 −0.1625）；删除降 0.41–0.69、保持 0.0、生命周期 5/5 过 |
| `recursive_development` | **negative（两形态）＋显式循环 partial ＋记忆槽 partial** | D-102/108；D-132/134；D-136/137/140；22.6.3 negative |
| `physical_transfer` | **verified（有界：仿真动作片段）＋碰撞/真机 uncovered** | 22.4.2b `action_consequence_pass`（full 0.781 vs mask 0.427）；22.8 J2 联合轨迹 0.69–0.80 |
| `lifetime` | **verified（有界：≥3 跨进程会话）＋长跨度 uncovered** | 22.5.1 3/3 recover_exact；22.8 3/3 session diff 0.0；100 会话/24h 未做 |

## 5. 曝光与 final

- check②：22.5.1 正式使用 **1 次**（`formal_2251: used=1, remaining=0`）；22.8 分配的 check **1 次未使用**；
- `final 128 world`：**未生成、未读取、未获当次 R-SEAL-1 授权** ⇒ 明确记 **`final_not_authorized`**，不以其它分区替代；
- 不签 final 通过、C/D/J 全任务、模型级成型、可独立接管交付。

## 6. 23 立项依据（不自动准入）

23 阶段目标（长期路线）：组合关系＋ ≥3 旧技能（≥0.80）。进入 23 的前置条件：

1. **主体**：以 22.8 的 30 单位 RouteB 新候选为 23 起点，登记**身份断点**（B 父原生核心线保留、不声称同一主体）；不得跨 seed 拼装；
2. **能力继承**：J1 appearance（有界）、J2 action（有界）、J4 recovery（有界）可作为旧技能候选；**J3 partial-positive 与 J5 negative 必须随 23 计划显式继承**，不得当作已验证能力；
3. **结构前置**：J3 的旧保持需要**加宽 direct/query**（结构变更、新 pin）；23 若要求纠错保持，须先立该结构卡；
4. **预算**：23 为**新阶段**，须重新定价、单卡授权；包 22 帽不复用；
5. **暂停项**：J5 固定策略（三次负）、unit-level 分化/生长收益（D-120/151）在本阶段不重开。

## 7. 包 22 财务账（如实）

- 旧账：`≈36,804 CPU`（含超支事故与增补轮，见 MAINLINE §0.1 / D-04/144）；
- 本阶段新增可核验回执（22.5.2/22.6 预研/22.7 S2/22.8 全系）**≈408.6 CPU**（14 份回执合计，逐卡为准）；
- 合计 **≈37,213 CPU**，对重冻阶段帽 **24,000** 超 **≈13,213（55.1%）**；**不追溯改帽**，按 D-04/144 先例永久记事故；后续每卡单列报价＋硬帽。

## 8. 不新增 22.9

本卡为包 22 无条件收束；不设 22.9，不续加局部探针；23 及以后须新阶段立项与授权。
