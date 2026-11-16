# 22.5.U.1 设计卡：单位原生读基础（单位图核心 B，首卡）

> 卡号：`22.5.U.1`｜日期：2026-11-16｜状态：**设计＋定价冻结完成，smoke 全过，停在训练授权节点**。
> 上级：[22.5.U 资格认定程序总卡](DESIGN-TAIJI_U2_UNIT_CORE_B_QUALIFICATION_20261116.md)｜战略依据 D-20261116-17。
> 契约：`plans/manifests/taiji_u2_u1_unit_read_20261116.json`（criteria_sha256 `b58f40d9…`，status frozen，training_authority null）。
> 接地：[全局收束](CLOSEOUT-TAIJI_U2_BYTE_WORLD_JOINT_GATE_20261116.md)、[单位生命周期规格](SPEC-TAIJI_U2_NEURAL_UNIT_LIFECYCLE_20261113.md)、[P3.9 收束](CLOSEOUT-TAIJI_U2_P3_9_GLOBAL_20261114.md)。

## 1 单问题

**单位直读能否做到 U2Subject 自身做不到的读？** 在单位图核心 B 上实现字节世界单查询读（first_a 场景＋中文位置请求→左/中/右/未知），检验联合父门的**读前提**。这是 A 线全局收束后转向单位核心 B 的第一层。正结果只签读前提；纠正＋保持归 U.2，正式 A/B 比较归 U.3。

## 2 主体与身份

- **主体**：`RouteBGraph`（`taiji/u2_neural_units.py`），12 单位/6 群，任务条件 query 直读 gated z/a，**不经固定 H**（H 仅 scratch）。身份 `taiji-u2-unit-core-b-v1`，核心版本 `unit-core-b-v1`，修订上限 2。
- **不继承 U2Subject 参数**（无逐值映射）⇒走路径备案 §4"同信息/同资源学习效果"公平性问。
- B 身份性质（smoke/verify 实测）：无 `h_readout`/`w_g`/`ell` 参数；`cut_h`（置零/扰动 H scratch）读出**逐位不变**。

## 3 输入编码合同（关键设计元素，已冻结）

模块 `taiji/u2_unit_input.py`，身份 `unit-input-column-color-v1`，digest `00319fff…`。first_a 公开场景→6 合法通道各 [16,8]（T=16 像素列）：

| 通道 | 布局 | 信息权利与降维损失 |
|---|---|---|
| rgb | [6 调色板色质量, 列中心 x, 总质量] | 768 原始值→128；**丢失**列内 16 行细节与精确像素；**保留**逐列颜色身份（调色板分离良好）与位置轴。用 `locate_public` 同一公开颜色先验 |
| question | [6 实体 one-hot, 问题类型标记, bias] | 丢失模板措辞（所有请求同模板，无任务相关损失）；保留查询实体身份 |
| goal | 固定标记（slot 0） | 常量，无判别信息（声明，非隐藏） |
| audio/proprio/feedback | 精确零 | 公开场景对位置读无任务相关音频/本体；U.1 只读无反馈。不主张信息权利 |

**接地验证**：编码导出的标签（`derive_label`，公开 oracle 在编码色质量上的重算，**仅用于验证、绝不作训练目标或模型输出**）与权威 oracle **96/96 train+fit 全一致**。

**设计裁定（上限更高）**：A 线死于池化 8 维 rgb token 承载不了"查询颜色在哪"＋字节路径未接。本合同**刻意保留任务相关感官证据**（逐列色质量＋位置轴），把"单位能否做 查询→颜色→位置 绑定"与"编码器损失"两个问题干净分离——不给单位图喂损失编码器，而是给忠实感官证据，检验直读机制本身。**单位不预贴实体/颜色/答案标签**（规格 §7）：编码仅供感官证据，绑定是图的计算。

## 4 读与训练

- **读出聚合**：`readout.mean(dim=0)`（T=16 上均值，冻结）→4 类 logits；argmax 预测。
- **训练**：Adam（lr 0.001）训图全部参数（units.r/p/c/η/b/w、input_projection、input_encoders、query、state_mix、readout）；CE 对 4 类位置标签；每步后 `project_parameters_()`。编码是固定外部感官证据。
- **门控**：gates=ones[T,12]；生命周期切通过 `active_gate`（mature=1）。
- **剂量**：384 更新/seed × 3 seed（2026111601/02/03），float64。

## 5 因果干预与控制（规格 §4＋B 身份＋任务控制）

- **四干预**：完整图 / 阻断跨单位消息（w 置零）/ 置换源单位（边源端 derangement 重接）/ 去历史状态（保留参数，目标从 fresh vs 先前 episode 末态）。
- **B 身份切**：`cut_h` 与 H 扰动读出逐位不变。
- **任务控制**：RGB 置零（不能定位）/ 固定问题（实体 0 常量，不能选色）→与完整图 agreement <0.5。
- **中性出生**（规格 §5）：`with_neutral_birth` 输出逐位不变（实测 max_abs 0.0）；出生精确中性点 p[:,new]=0 且 immature 门控关闭⇒r 梯度按构造为零（规格允许出生首刻 ∇p/c/b 为零）；**成熟化＋p 置非零后**任务梯度到达新单位 r（信用路接通，实测非零）。
- **机制探针**：未训练图 c=0/w=0（无记忆/边），smoke 用 `mechanism_probe`（c=0.5/w=0.5·mask，0 优化器步）证明四干预机制因果接通；run/verify 在训练后图上量实际因果贡献。

## 6 门（读前提，3 seed 同向）

fit 读 ≥0.80（32）＋train 读 ≥0.90（64）＋distinct ≥3＋控制 agreement <0.5＋干预（without_edges/permute_source/history 改变读出比例 ≥0.5，cut_h 逐位=1.0）＋中性出生精确＋独立进程恢复逐位＋篡改拒绝。

## 7 验证与定价（已完成，零训练）

- **smoke**：34/34 全过（3 seed×B 身份/四干预/cut_h/中性出生/信用路/梯度覆盖＋编码 oracle），oracle 96/96，0 优化器步，0.25 CPU。
- **契约测试**：9 项（编码 digest/形状/oracle、源钉、B 身份 cut_h、中性出生、干预机制、梯度覆盖、判据冻结）。
- **回归**：376 项全过（u2/architecture/bind2/b1/b2/c1/c2/u1）；mypy 核心 **0 错/261 文件**、runner 0；ruff B/SIM＋black 门禁绿。
- **定价**：per_update 0.01562 CPU、per_eval 0.0（**低于本机 process_time 刻度 0.0156s，分辨率不足，已登记限制**，沿 P3.9）；`read_3seed`（1152 更新＋每 seed 13 评估）帽 **18.0 CPU／36.0 wall**（设计帽 600/1200 内，单位图 B 极小）。smoke 0.25＋price 1.55 CPU。

## 8 能签与不能签

**能签**：单位核心 B 在 U.1 范围内的读行为读数、因果干预、中性出生、工程恢复、价格。**不能签**：父/能力（联合父门须 U.1–U.3 全过）/check②/22.6；编码 oracle 一致性是输入信息忠实性，**不得引为"主体会读"**；构造性 0 损伤不得引为保持；P3.9 合成窗读数不得引为字节世界能力。

## 9 当前状态与下一步

设计＋定价冻结、smoke 全过、契约测试绿。**停在训练授权是非题**。授权后由我执行 `--run`（帽内）→`--recover`→`--verify`，按冻结出口结案（read_prerequisite_passed/failed/budget_stop），更新 plans 与台账。正结果→U.2 单位纠正另行设计授权；负结果→按程序修订上限裁决。