# 10.7e 输入条件读出：把目标绑到动作上（一次有界训练，≤120 CPU 秒）

> 文档状态：[合同] — owner 已于 2026-11-01 裁定「收口 10 ＋ 立项读出实验」。本合同按 10.7b／10.7c／10.7d 的三条读数设计：共享参数的一阶更新要么共模（0.99998 余弦）、要么成对相消（均值只剩 0.56%），能承载条件化的是**梯度逐样本不同**的参数。

日期：2026-11-01。候选＝09.8 主体（父摘要 `921dc998…`）。**`taiji/living_core.py` 与主体参数全部冻结**。

## 1. 机制

在候选内新增一个零初始化的**秩 4 双线性读出头**：

```
score(goal, action) = (Uᵀ g) · (Vᵀ a),   U, V ∈ R^{128×4},   1,024 参数
g = encoder.goal_vector(目标句字节)           # 主体自己的目标编码，128 维
a = mean(encoder.context(动作句字节))         # 主体自己的动作编码，27×128 池化
```

`V` 零初始化、`U` 小幅随机（std 0.01）⇒ **起点对任何输入都输出 0**，不改变候选任何现行行为；不能两边都置零——那样 `∂score/∂V = (gᵀU)a` 也恒为 0，梯度死锁。只有 U／V 两组参数参与训练。

## 2. 训练与对照（各一条臂，同预算）

- 样本：与 10.7 完全相同的 12 个 (局面, 目标) 样本，**按局面 6 训练／6 留出**。
- 目标函数：成对 margin `softplus(-(score(对) − score(错)))`，每样本一项，取平均。
- 优化：Adam，只更新 U／V；步数 300，lr 0.05，固定种子。
- **对照臂**：同样的头、同样的步数，但把 6 个训练样本的对／错标签**全部对调**（对抗常量）。留出应 ≤3/6。

## 3. 判据（预注册，能翻）

1. **冻结**：训练前后主体的 `career_nll` 在 6 个留出样本上**逐位相同**；否则 `backbone_mutated`。
2. **主读**：留出 ≥5/6 ⇒ 有条件化；≤3/6 ⇒ `readout_binding_absent`；4/6 ⇒ `readout_inconclusive`。
3. **对照**：标签对调臂留出 ≤3/6，否则 `control_not_null`。
4. 未训练的同头（全零）平局随机 ⇒ 记录即可，不作判据（2 个候选下平局必 3/6）。
5. 预算：≤120 CPU 秒／240 墙钟秒；超限 `budget_stop`；异常 `engineering_failure`。

出口：`readout_binding_supported`／`readout_binding_absent`／`readout_inconclusive`／`control_not_null`／`backbone_mutated`／`budget_stop`／`engineering_failure`。

## 4. 可声称与不可声称

- **可声称**（若过门）：在这套合成夹具上，冻结主体的既有编码加上 1,024 个新参数，就能按目标从两个合法动作里挑对那个，且换标签的对照做不到。
- **不可声称**：不表示主体自己会走路径（读出头只吃编码、不读循环状态）；不表示自然语言连续任务；不表示 P2 满足（那要按 11 的准入判据另判）；不表示其余参数可训；不表示样本外推。
- 权重只落 `output/taiji_core_10_7e/`，不发布、不替换默认。

## 5. 交付

本合同、`taiji/goal_action_readout.py`、`scripts/training/verify_taiji_readout_10_7e.py`、`tests/taiji_native/test_goal_action_readout.py`、`reports/taiji_core_readout_10_7e_20261101.json`、MAINLINE／03／索引同步。
