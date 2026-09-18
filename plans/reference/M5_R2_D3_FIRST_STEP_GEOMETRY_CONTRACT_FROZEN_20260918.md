# M5 R2-D3：首步读取几何假设包（多头位置感知读取，冻结版）

2026-09-18。状态：**已冻结**（用户在[A 族评审备忘 §6](M5_R2_D2_A_FAMILY_REVIEW_MEMO_20260918.md)战略裁决 S1 后提交即冻结）。
前置：R2-D2 六构型收敛结论——首回答字节内容决策对证据来源/起点（单头）/梯度方差/类别边缘均不敏感；copy 五形状 train exact=1.0，「不」首字节两类（negation、combination_different）恒为 0。本包是**新假设、新图族**，不是 A 最小切片的再调参。

## §1 假设 H-G（冻结）

> 在回答生成**尚未输出任何字节时**（第一步），单头、对位置无感知的软寻址无法把「问题类型」与「材料内位置」组合判别；提供 (a) 多个各自有查询投影的读取头（可分别特化到材料从句首/尾等角色位置）、(b) 证据键上的固定位置编码后，问题条件化起点（v5 已实现，单独不充分）能够驱动第一步选择正确的读取角色，使 negation 首字节「不」与 combination_different 首字节「不」在训练集可拟合，并在 D1 dev 反事实对上随事实正确翻转。

## §2 graph v6 精确规格

在 v5（per_position 证据 + copy + question-conditioned start）基础上：

1. **位置编码**：每个证据行 i 的键附加固定正弦位置向量 PE(i, sw=48)（sin/cos 多频，不可训练，0 参数）：`K_i = h_i·evidence_key + PE_i`；值 `V_i = h_i·evidence_value` 不变。PE 只影响寻址，不进入读出值内容。
2. **H=4 读取头**：每头独立查询投影 `head_query_h (rw,sw)`（4×3,072=12,288 新参数）；头 h 的权重 `w_h = softmax(K·(r·head_query_h)/√sw)`。
   - 生成读出：`read = (1/H)·Σ_h w_h·V`（维度保持 sw=48，decoder 不扩宽）；
   - copy 分布：`p_copy = (1/H)·Σ_h scatter(w_h, entry_bytes)`（boundary 仍只由 vocab 给）；
   - copy gate π 仍为单标量（v4 既有 copy_gate 参数）。
3. 问题条件化起点沿用 v5（answer_start_weight/bias，4,160 参数）；证据/值投影、decoder、训练目标（真实下一 byte NLL）全部不变。
4. 参数清单：v4 copy 82,658 中 address_query(3,072) 被四个头查询替换（−3,072+12,288），加 v5 起点 4,160 = **96,034**；配置 `readout_heads: int = 1`（1 时沿用 address_query，等价 v5 图，行为逐位不变；4 为 H-G 臂）与 `positional_keys: bool = False`（PE 零参数）。checkpoint 版本 5→**6**，载荷含两标志与 PE 规格；旧版本继续可读。
5. 训练：Q1 验证过的**微批 8**（660 步/30 epoch）+ **原始固定顺序**（本包先不叠加均衡——几何变量必须单独检验）；lr0.01、seed 20260917、wall 20 min、preflight 先行；评估在未改动 train split。

## §3 实现门（零训练，新增）

1. H=1 时 v5/v4 图逐位不变（logits/混合分布/生成相同）；H=4 清单 99,106。
2. PE 有界、仅进键：同前缀各行 PE 随位置不同；改 PE 只改寻址不改 V/read 值的构造（结构性断言）。
3. 多头权重各自归一化、梯度到每个 head_query 有限非零；数值差分；单头消融（屏蔽 3 头）改变读出。
4. v5 材料隔离性质保持（r_0 不含材料；同题干材料变化只经读出影响首步）。
5. v6/v5/v4/v3/v2 checkpoint 恢复矩阵；fresh-process preflight。

## §4 probe 与门（一次）

- A-G probe（H=4，1 运行，seed 20260917，上述冻结训练）：copy-supported（144）M1 **≥0.90**，negation M1>0；稳定性沿用 Q1（观测 loss 上升≤15%、e25→30 M1 回撤≤0.10）；描述 combo（combo_different 首字节同为「不」）。
- 通过才进入 matched dev：H-G(H=4) vs 单头位置编码对照（H=1+PE，隔离「多头」与「位置编码」两贡献，见 §5），3 seeds，微批8；G1–G6 阈值沿用（M3≥0.26/δ≥0.20；M4≥0.50/δ≥0.25、fact_flip≥2/6；misbind ΔM4≥0.50；上下文 ΔM3≥0.30；seed；boundary/恢复）。

## §5 对照与失败路由（预承诺）

| probe 结果 | 路由 |
|---|---|
| 过（稳定+M1≥0.90+negation>0） | matched dev；dev 上过门才支持 H-G |
| 稳定但仍 0.667 | 多头几何也不充分；补 H=1+PE 对照若同样失败 ⇒ 读取几何假设在当前尺度判否，转评审（目标函数/B/规模），不再追加几何变体 |
| H=4 过、H=1+PE 不过 | 多头是关键变量；若反之则位置编码是关键 |
| 不稳定 | 与 Q1 矛盾（微批已稳），按实现缺陷回实现门，不放预算 |

## §6 边界

不读 dev/final；D1 仪器与 M1–M5 口径不变；一次 probe；`growth_admitted=false`、`can_promote=false`；train-only 成功不是 L2 主张。
