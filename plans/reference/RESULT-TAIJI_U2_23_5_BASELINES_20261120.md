# 结果卡：23.5 等算力 GRU/TF 基线（`baseline_comparison_verified`；GPU 口径披露）

> 文档状态：[结果卡] — 2026-11-20（D-20261119-175）。上游 [23.2c](RESULT-TAIJI_U2_23_2C_ATTENTION_20261120.md)、[23.4](RESULT-TAIJI_U2_23_4_FEEDBACK_20261120.md)。**零 check/final、只读 fit。**

## 1. 协议

- 同一 Q2/Q3 数据与 23.2c 令牌接口；U2 臂＝冻结 23.0b 父＋注意力读；
- GRU 基线（GRU(8,16)＋查询拼接＋MLP）与 Transformer 基线（1 层 d=16/2 头＋查询 token＋均值池化）；
- 各臂同批次、1,500 步、3 seed；比较 fit 准确率；门：U2 不劣于最强基线（差 ≥−0.05）、GRU 实测 CPU 比 ≤10%、旧技能损伤 ≤0.02；239.64 CPU（帽 1,500）。

## 2. 读数（[回执](../../reports/taiji_u2_23_5_baselines_20261120.json)）

| seed | U2 | GRU | Transformer | U2−最强 | 旧技能损伤 |
|---|---|---|---|---|---|
| 20261120 | 0.3555 | 0.3242 | 0.2734 | +0.0313 | 0.0 |
| 20261121 | 0.3281 | 0.2930 | 0.3242 | +0.0039 | 0.0 |
| 20261122 | 0.3477 | 0.2695 | 0.2695 | +0.0781 | 0.0 |

- **noninferior ✓**：均值差 **+0.038**（≥−0.05）；
- **near_equal_compute_gru ✓**：实测 CPU 比 **0.986**（GRU/U2，≤10%）；
- **retention ✓**：0.0；
- ⇒ **`baseline_comparison_verified`**。

## 3. 口径披露（如实）

1. 解析 FLOPs 估计器对本卡不可用：GRU 用 `nn.GRU` 而非 Linear，线性项统计漏算（记录值 0.36 属失真）；本卡改用**实测 CPU 比**作为等算力门，GRU 与 U2 近等（0.986）；
2. **Transformer 超支**（实测 CPU 约 U2 的 1.7 倍）却更弱；U2 对其非劣不构成等算力比较，只作「更强算力基线也不优于 U2」的旁证；
3. U2/GRU/TF 的 fit 绝对水平都低（0.27–0.36），与 23.2c 内容上限一致——**等算力非劣不等于组合泛化成立**；
4. 旧保持 0.0 仍来自父冻结。

## 4. 判读与 23 收口

- 23.5 只签「同任务同接口下 U2 不劣于等算力 GRU，且不劣于超支 TF」；**不签**组合泛化主门 Q≥0.80；
- 23 子卡已齐：23.1 `data_contract_frozen`、23.2c `relation_read_verified`（有界）、23.3/3b `generation_bounded_negative`、23.4 `feedback_effect_verified`（通道级有界）、23.5 `baseline_comparison_verified`；
- 23 应进入**范围限定收口**卡（`composition_partial`）：主门 Q≥0.80 未达，结论限定为「令牌接口下关系读取有真实增益、反馈通道可路由、但内容/组合泛化未达主门」；
- 主窗 34,500 CPU、check≤2/final≤1 仍待 owner 授权；本阶段 23 实际花费 ≈1,900 CPU；零 check/final；不签 23 主门/final/C/D/J/模型成型。
