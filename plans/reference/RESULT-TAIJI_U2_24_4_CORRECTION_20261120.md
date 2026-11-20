# 结果卡：24.4 四类纠错探针（`correction_partial`；稳定性子项设计缺陷已披露）

> 文档状态：[结果卡] — 2026-11-20（D-20261119-182）。上游 [24.0 预注册](DESIGN-TAIJI_U2_24_0_EXPERIENCE_20261120.md)、[24.2](RESULT-TAIJI_U2_24_2_CONSOLIDATION_20261120.md)、[24.3](RESULT-TAIJI_U2_24_3_PREFIX_20261120.md)。**零 check/final、只读 fit。**

## 1. 协议

- 冻结 23.0b 父；10 会话×20 事件（无重复抽样）；会话 1–8 训练关系读取头，会话 9–10 的 40 事件分四子集各 10 条：stale（标签翻转）、revocation（回退目标）、misleading、contradiction；
- 每类给 **2 次有效交互**（2 个 batch 的重放）；
- 门：stale/revocation 目标恢复 ≥0.50；misleading/contradiction 原标签损伤 ≤0.05；旧技能损伤 ≤0.02；3 seed、200.08 CPU（帽 4,000）。

## 2. 读数（[回执](../../reports/taiji_u2_24_4_correction_20261120.json)）

| seed | stale 恢复 | revocation 恢复 | misleading 原标签 | contradiction 原标签 | 旧技能损伤 |
|---|---|---|---|---|---|
| 20261120 | 0.700 | 0.600 | 0.200→0.600 | 0.100→0.300 | 0.0 |
| 20261121 | **0.400** | 0.600 | 0.300→0.500 | 0.300→0.600 | 0.0 |
| 20261122 | 0.600 | 0.800 | 0.300→0.600 | 0.100→0.400 | 0.0 |

- **stale 恢复 ✗**（seed1 0.40 <0.50；另两 seed 0.60/0.70）；
- **revocation 恢复 ✓**（0.60/0.60/0.80）；
- **stability «✓» 但不可信**：misleading/contradiction 的「原标签」在 2 次交互后反而**上升**（0.2→0.6 等），因为这两臂的「反馈」实为**原标签目标训练**——不是冲突/误导反馈，**没有测到抗误导**；
- **retention ✓**（0.0）；
- ⇒ **`correction_partial`**。

## 3. 判读（如实）

1. **回退类纠错在 2 次交互内可恢复**（revocation 3/3、stale 2/3）；stale 有一 seed 未过门；
2. **稳定性子项设计失败**：misleading/contradiction 臂实际训练回原标签，未构造真正的冲突/误导反馈，因此 stability=✓ 是**无效读数**，本卡不得据此签「抗误导/抗矛盾」；
3. 旧保持 0.0 仍来自父冻结；
4. 本卡不签 24.4 完整、不预判 24 收口。

## 4. 后继

- 24.5（skill×session 总账）只剩汇总；建议直接对 24 做**范围限定收口**（`experience_learning_partial`）：24.1 接口有界、24.2/24.3 无净益/无成本收益、24.4 仅回退类 2 次交互内可恢复、稳定性未测；
- 「读取结构升级卡」仍列为 24 后第一优先；主窗 17,000 CPU 仍待 owner 授权；累计 24 实际 ≈870 CPU；零 check/final；不签 24 通过/final/C/D/J/模型成型。
