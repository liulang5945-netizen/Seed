# 结果卡：23.1 数据合同冻结（`data_contract_frozen`）

> 文档状态：[结果卡] — 2026-11-20（D-20261119-167）。上游 [23.1 合同草案](DESIGN-TAIJI_U2_23_1_DATA_CONTRACT_20261120.md)、[23.0b 三技能父](RESULT-TAIJI_U2_23_0B_JOINT_20261120.md)。**零训练、零 check/final。**

## 1. 内容

- 新增 `taiji/native_composition.py`（fixture `taiji-23-composition-v1`）与 `scripts/training/run_s2_23_1_contract_20261120.py`；
- 896 world × 3 题 = **2,688 条**：train 1,536（Q1/Q2/Q3）、fit 384（Q2/Q3/Q4）、check 384、final 384（同构，封存只记 digest）；
- 分区合同写入 `plans/manifests/taiji_u2_23_1_composition_contract_20261120.json`（含 `criteria_sha256`）。

## 2. 验收（[回执](../../reports/taiji_u2_23_1_contract_20261120.json)，244.3 CPU）

| 项 | 结果 |
|---|---|
| C1 分区规模 | ✓ 1,536 / 384 / 384 / 384（共 2,688） |
| C2 三轴无泄漏 | ✓ 训练与 fit/check/final 的 `(kind, template_group, direction, pair_order)` 集合不交 |
| C3 oracle | ✓ **1.0**（独立重算 Q2/Q3 答案与标签逐条一致） |
| C4 digest 可复现 | ✓ 两次生成一致 |
| C5 中文/答案空间 | ✓ UTF-8 正常、答案枚举完整 |

**分区 digest**（冻结值）：

| 分区 | sha256 |
|---|---|
| train | `5297bde6…f259be4` |
| fit | `b24abb21…08f643` |
| check | `99c50717…c701ab5` |
| final | `6667dbe7…3478d` |

⇒ **`data_contract_frozen`**；check/final 未读（digest 仅由生成器计算，未暴露标签到训练路径）。

## 3. 判读

1. 23 的数据面可执行、可复现、无泄漏、oracle 符号真值 1.0；
2. 2,688 条与 MAINLINE「2,688 world」口径对齐（本卡按 896 world×3 题实现，world 数在合同 manifest 中显式记录，偏差披露：MAINLINE 原文的 "world" 一词按题项计）；
3. 数据合同冻结后，任何模板/方向/顺序改动即作废，须新卡；
4. 23.2（2 跳关系读取）可进入实现；主窗训练预算仍按 owner 授权口径执行。

## 4. 状态

- 23.0b 三技能父（`joint_parent_ready`）＋23.1 数据合同（`data_contract_frozen`）已就绪；
- **23 主窗（34,500 CPU、check≤2/final≤1）仍待 owner 明示授权**；本卡零训练、零 check；
- 不签 23 通过/final/C/D/J/模型成型；不改包 22 判词。
