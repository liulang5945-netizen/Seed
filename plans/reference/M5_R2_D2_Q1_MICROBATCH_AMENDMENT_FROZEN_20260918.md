# R2-D2 修订四：Q1 微批梯度累积（v4 A2 图不变，冻结版）

2026-09-18。状态：**已冻结**（用户在[A族评审备忘](M5_R2_D2_A_FAMILY_REVIEW_MEMO_20260918.md)§4 二次评审裁决 Q1 后提交即冻结）。
定位：**优化器步构造合同**。图（graph v4 A2，per_position+copy，常量 start，82,658 参数）、D1 数据与顺序、lr、epoch、probe 门全部不变；不叠加均衡（均衡与否留待 Q1 稳定后另议）。

## §1 假设

四次 learnability 运行中唯一稳定构型是 v4 A2@lr0.01=平台 0.667；三次失稳（0.05、v5 晚期、均衡全程）都与梯度扰动增大相关。假设：**batch=1 Adam 的梯度方差是反复失稳的共同约束**；把每步梯度改为 8 个 episode 的平均（梯度累积），在不引入任何表征/数据变化的情况下，训练应全程稳定下降。

## §2 冻结算法

1. 训练器新增 `train_microbatch(batch)`：对一批 episode 逐个算 `sequence_loss`（各 episode 位置等权平均后再对 batch 等权平均），求和反传，**一次** Adam 步骤；每批前 `optimizer.zero_grad()`；辅助损失（contrastive/scheduled/first-byte 权重）全部关闭。
2. 顺序：沿用冻结 data_order（固定文件顺序，不 shuffle、不均衡），每 epoch 174 episode 切成 21 个 8 集批 + 1 个 6 集尾批 = 22 个优化步；30 epochs = 660 步；episode 暴露总数仍 5,220。
3. Adam lr=0.01、其余优化器状态/seed 规则不变；30 epochs、wall 20 min、preflight 先行（保存/恢复口径不变）。
4. 评估与门仍在**未改动**的原始 train split 上自由生成。

## §3 门（冻结；三分支出口）

1. **稳定性门（Q1 自身判据）**：e5→e30 每次观测 loss 不高于前一观测 +15%，且 e25→e30 M1 回撤 ≤0.10；不得出现 P1 式终局发散。
2. **学习门（沿用）**：copy-supported M1≥0.90，negation M1>0，fact/sof 不回落；finite/declining、preflight、wall cap 同前。

出口预承诺：

- **稳定 + M1≥0.90**：Q1 支持（方差与首字节决策共同成因）→ 才允许 matched dev（A2-microbatch vs C1-microbatch，3 seeds，G1–G6 原样）。
- **稳定 + M1 仍≈0.667**：梯度方差只是稳定前提，不是首字节解；停止微批路线，回二次评审（此时「微批+均衡」才是可讨论的新合同，不得自动执行）。
- **不稳定**：该形式下梯度方差假设被否，回二次评审 Q2/Q3/Q4。

## §4 边界

只做一次；不改图、不换数据顺序、不叠加均衡、不调 lr/epoch；不读 dev/final；`growth_admitted=false`、`can_promote=false`；train-only 结果不是 L2 或内容能力主张。
