# M5 / R2 内容绑定实施合同 v2（配对绑定目标，冻结版）

2026-09-19。状态：**已冻结**。冻结出处：v1 判停后用户在路线裁决中选择「配对绑定目标 v2」（问答记录：选择即批准对应代价，预算沿用 2026-09-19 已批上限）。依据：[v1 标定判停备忘录](M5_R2_CONTENT_BINDING_CALIBRATION_STOP_20260919.md)。本文件以 [v1 FROZEN 合同](M5_R2_CONTENT_BINDING_CONTRACT_FROZEN_20260919.md) 为基线，**只改一个主设计变量**（§D1），其余条款逐字继承。

## §D1 主设计变量：组内反事实对比目标（pair-contrastive auxiliary）

**假设 H-P**：v1 的逐题 CE 在 train 上可被逐题记忆满足，绑定没有学习期压力；在组内两成员互为反事实的结构下，要求模型在**同一前缀下把自己成员的答案似然排到交叉成员答案之前**，会在训练早期（记忆化定型前）对 object_swap（问题→值选择）、relation_flip（相同/不同）、negation_scope（是/否）这些失败类产生直接对比压力，使未见实例的绑定泛化改善。若 v2 标定仍不达门槛，则"目标缺对比压力"假设被削弱，残余解释收敛到表示/容量/数据配方。

**定义（冻结）**：微批内每组 (a,b) 四个序列对数似然（TF、逐位求和）：s_xx = logP(resp_a|q_a,mat_a)、s_yy = logP(resp_b|q_b,mat_b)、s_yx = logP(resp_b|q_a,mat_a)、s_xy = logP(resp_a|q_b,mat_b)。辅助项

    L_pair = mean_over_groups [ softplus(s_yx − s_xx + γ) + softplus(s_xy − s_yy + γ) ]

其中 γ = 1.0（nat）、λ_pair = 1.0，总 loss = CE项 + λ_pair × copyNLL项（v1 复合，不变）+ λ_pair × L_pair。等答案类（distractor_invariant/unknown_preserved）该项自动退化为常数（两成员同串，s_yx=s_xx），无梯度、无副作用。

**不引入**：金槽/金关系监督、答案类型字段进推理、外部解题器、任何 sealed 路径。交叉项只使用同组两成员的 train 字符串与答案（监督用途），运行时推理不可见。

## §D2 继承不变（v1 FROZEN §A 全部数值沿用）

数据（digest `377a7391…` 三件套）、A/B 两臂计算图（`taiji/sequence_content_workspace.py` 不改图）、微批 4 组=8 题（组不拆）、两配方 cal_lr1/cal_lr3、调度、检查点/恢复机制、选择规则（复制≥0.90 且未知≥0.90 且值有限 → 三类成对宏最大；并列低 lr→早更新）、停止条款、产物目录纪律。标定 seed 20260920、正式 seed 20260921/22/23、评估点 500/1000/1500/2000 全部不变。

## §D3 v2 执行顺序与身份

1. 实现门（零训练）：对比项梯度到达两臂内容模块与 copy 通路；构造用例验证 margin 数学（同串→常数零梯度；翻转→正确符号）；v1 checkpoint 仍可加载（向后兼容）；v2 checkpoint 记录 `trainer_revision="v2-pair-contrastive"` 与 γ/λ_pair。
2. 标定 4 次（产物 `reports/r2_content_binding_v2/<arm>/<seed>/<config>/`）→ 按 §D2 选择规则判读：任一臂有合格候选 → 该臂进正式三 seed；两臂皆无 → v2 判停结案（同 v1 纪律，不调 γ/λ 再试）。
3. 正式与确认批按 v1 FROZEN §A/§B 原文执行（含 sealed 一次授权批次）。
4. γ/λ_pair 在 v2 标定前冻结（本文件即为冻结时点），不得看 v2 结果后调整。

## §D4 判定边界

v2 判停不证明"配对目标无效"，只结束该配置投入；v2 通过标定也不构成能力结论——能力结论只出自确认批各门（BIND/DELTA/COPY/其他切片/旧 dev 回归/内容消费/真实性恢复）。全程不晋级 L2/M5、不改默认、不重启 P3b。
