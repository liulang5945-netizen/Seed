# 22.5.U.1r1 设计卡：位置感知读出（单位核心 B 修订 1）

> 卡号：`22.5.U.1r1`｜日期：2026-11-16｜状态：**设计＋定价冻结完成，smoke 全过，停在训练授权节点**。
> 上级：[22.5.U 总卡](DESIGN-TAIJI_U2_UNIT_CORE_B_QUALIFICATION_20261116.md)｜立项依据 D-20261116-21（owner 裁决）｜父卡 [U.1 结果](RESULT-TAIJI_U2_U1_UNIT_READ_20261116.md)（D-20261116-20）。
> 契约：`plans/manifests/taiji_u2_u1r1_position_20261116.json`（criteria_sha256 `f11879d3…`，status frozen，training_authority null，revision 1/2）。

## 1 单问题与修订依据

U.1（mean-pool 读出）判 `read_prerequisite_failed`，归因（D-20261116-20）：**编码信息忠实（96/96 oracle），缺口在读出聚合＋RGB 信用路**——mean(T=16) 把"查询颜色在哪一列"的空间绑定平均掉；rgb 编码器起点梯度 0.0025 vs query 0.073（~30× 饥饿）；非剂量（8× 剂量 train 仅 0.5/fit 塌）。

**修订 1 唯一结构变量**：用**问题条件列注意力**替代 mean-pool 读出。**编码合同 `unit-input-column-color-v1`（digest `00319fff…`）冻结不变**。直接打定位缺口：保留列绑定＋把梯度路由回 RGB 内容。

## 2 机制（`taiji/u2_unit_position_readout.py::PositionRouteBGraph`）

继承 `RouteBGraph`（units/encoding/state_mix/p 不变），删除 mean-pool 读出头 `readout` 与固定 `query` 槽，新增：
- **列 key**：`projected = p @ state_view`（[T,H]，来自单位活动，RGB 驱动）⇒梯度经 units 路由回 RGB 输入。
- **问题条件 attention query**：`attn_query = question_vec @ W_q`（question_vec=question 通道首行 [8]，W_q 学习投影 [8,H]）⇒注意力条件于"问哪个实体"。
- **列注意力**：`attn_logits = (projected @ attn_query)/√H`；`attention = softmax(attn_logits)`（[T]）；`attended = attention @ projected`（[H]）⇒保留被选列的颜色＋位置轴 x。
- **位置头**：`position_head(attended)`（Linear H→4）→4 类位置 logits（**单向量，不再 mean**）。

身份性质（smoke/契约实测）：H 仅 scratch（cut_h 逐位不变）；中性出生精确（新单位 p=0⇒列 key 不变⇒注意力/读出不变，max_abs 0.0）；成熟化＋p 非零后新单位 r 信用路非零；route=`B_position`，无 h_readout/w_g/ell。注意力问题条件化实测（换问题实体⇒注意力分布改变）。

## 3 训练与门（沿用 U.1）

Adam lr 0.001 训全图参数（units.r/p/c/η/b/w、input_projection、input_encoders、state_mix、W_q、position_head），CE 对 4 类位置标签，每步 project_parameters_()。384 更新/seed×3（2026111601/02/03）float64，gates=ones[T,12]。门：fit≥0.80（32）/train≥0.90（64）/distinct≥3/控制（RGB 置零/固定问题）agreement<0.5/四干预（阻断消息/置换源/去历史改变读出≥0.5，cut_h 逐位=1.0）/中性出生精确/独立恢复逐位＋篡改拒绝，3 seed 同向。run 另记注意力熵诊断。

## 4 验证与定价（已完成，零训练）

- **smoke**：34/34（3 seed×B_position 身份/四干预/cut_h/中性出生/信用路/梯度覆盖＋编码 oracle），oracle 96/96，0 优化器步。起点 rgb 信用 0.0024/0.0013/0.0006 vs question 0.0062/0.0047/0.0067（绝对 question 信用较 U.1 大幅下降，注意力机制改变梯度流；能否学会绑定须 run 验证）。
- **契约测试**：8 项（编码冻结/源钉/B_position 身份 cut_h/注意力问题条件化/中性出生/干预机制/梯度覆盖/判据冻结）。
- **回归**：384 项全过；mypy 核心 **0/262**、runner/模块 0；ruff B/SIM＋black 绿。
- **定价**：per_update 0.01562/per_eval 0.0（低刻度限制声明），`read_3seed`（1152 更新＋每 seed 13 评估）帽 **18.0 CPU／36.0 wall**（设计帽 600/1200 内）。

## 5 能签与不能签

**能签**：单位核心 B 修订 1 在范围内的读行为读数、因果干预、中性出生、工程恢复、价格。**不能签**：父/能力/check②/22.6；编码 oracle 一致/机制接通/注意力问题条件化不得引为"主体会读"。正结果只签读前提（U.2/U.3 另案）。

## 5a 必须随读数引用的限制（2026-11-16 独立复核追加，D-20261116-22）

1. **RGB 绝对信用没有上升**：smoke 实测 `rgb_encoder_credit` **0.002435／0.001256／0.000635**，对 U.1 的 0.0025 持平或更低；question／rgb 比值由 ~29× 变为 **2.55／3.74／10.50×**，改善**全部来自 question 信用下降**（0.073→0.006208／0.004692／0.006663，约 10×）。故 §1／§2 的"把梯度路由回 RGB 内容"只有**结构证据**（列 key 经 units 依赖 RGB、全参数有梯度），**没有幅度证据**；本修订仍可能因同一信用饥饿失败，不得据 smoke 绿声称信用路已修好。
2. **出生控制读数不是门读数**：`rgb_zero_agreement` 1.0／0.0／1.0、`fixed_question_agreement` 1.0×3 取自未训练图，只说明出生读出对 RGB 不敏感；控制门（agreement<0.5）只在训练后判。
3. **无显式注意力监督**：为守"唯一结构变量"，本卡不加位置聚焦损失，CE 须间接把注意力 sharpen 到被查询列。同族前车：C.1 注意力近均匀（mass 0.157）失败，C.2 加监督后 mass 0.96 而值仍不可解码。run 记注意力熵诊断；若负结果，须区分"注意力没聚焦"与"聚焦了但值不可读"，不得直接归因剂量。
4. **定价分辨率不足**：`per_update=0.01562` 恰在本机 `process_time` 刻度上，推导 17.99424 对帽 18.0 只余 0.032%；帽是硬约束，估计不是。U.1 同帽实耗 6.375 CPU，实践中偏保守。
5. **独立复核结论（零优化器步）**：8 source pin＋3 起点 bundle（各 27,366 字节）＋编码合同 digest＋`value_digest(criteria)`＋`value_digest(data)`（train 64／fit 32／price 8，无 check/final/confirm 键）逐项匹配；`guards(run)` 在 `training_authority=null` 下拒绝、补字段后通过（两支都走到、未写盘）；pin／编码／bundle 三条篡改负对照全部命中；契约测试 8/8；ruff／B,SIM／black rc 全 0。复核者首版用整文件 sha 比 `data_digest`、自造 JSON 序列化比 `criteria_sha256` 造出两处假红，改用被核验件自己的 `value_digest` 后均匹配。

## 6 当前状态与下一步

设计＋定价冻结、smoke 全过、契约测试绿。**停在训练授权是非题**。授权后由我执行 --run（帽内）→--recover→--verify，按冻结出口结案。正结果→U.2 单位纠正另行设计授权；负结果→修订上限 2 用满 1，剩 1 次，按程序裁决（修订 2 或单位核心 B 读路径收束）。