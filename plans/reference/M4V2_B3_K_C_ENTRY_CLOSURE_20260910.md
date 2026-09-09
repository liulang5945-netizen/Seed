# M4.V2 B3-K C-entry formal closure

日期：2026-09-10  
状态：已完成 formal execution，候选不晋级，保留为 shadow evidence

## 合同与执行

本轮严格使用冻结的 [C-entry evaluation manifest](../manifests/taiji_m4v2_b3_k_c_entry_evaluation_v1.json)，没有在读取 sealed-test 后修改数据、scorer、阈值或 arm 定义。

- 3 个独立 model seed：17、23、31。
- 3 个 target-aware course seed：`A+B+C`、`A+A+B`、`A+B+B`。
- 9 个 candidate continuation cell 与 9 个 C-entry fixed-large control cell。
- sealed artifact：3 个独立 episode、9 个 observation，内容寻址 digest 与 manifest 一致。
- candidate 只更新 `k1.semantic`、`k2.transition`；`k3.outcome_projection`、parent 和 default runtime 不变。
- 训练前 checkpoint、fresh restore、disk restore、rollback、owner lineage、训练/推理分离资源账本均已记录。
- 全程 CPU；未接入 provider、MCP、client、CUDA 或默认 runtime。

## 结果

正式报告：[C-entry formal report](../../reports/taiji_m4v2_b3_k_c_formal_20260910.json)

| 量尺 | 结果 |
|---|---:|
| candidate sealed combined-MSE mean delta vs frozen | `-0.0004431358` |
| candidate sealed worst delta vs frozen | `-0.0002581254` |
| fixed-large sealed combined-MSE mean delta vs frozen | `-0.0011265067` |
| candidate 胜过 fixed-large 的 cell 数 | `0/9` |
| technical/resource/candidate-quality formal Gate | 通过 |
| promotion | 禁止 |

## 结论边界

1. Taiji 当前 K1/K2 continuation 在冻结的 target-aware 课程上确实产生了可恢复、可回滚、未见 sealed 改善；这不是“没有学到东西”。
2. 在同一 validation/sealed scorer 下，宽度 2 的 fixed-large 控制在所有 9 个 cell 都更好。当前 continuation 不能证明比强固定容量对照更有能力或更高效，因此不得进入默认 runtime，也不能宣称结构成长已被证明。
3. 该结果也不能单独证明 Taiji 继承式学习路线错误：candidate 与 fixed-large 的容量、训练步数和表示自由度不同。本轮只证明“当前这套 K1/K2 更新规则和课程，在当前强对照下尚未取得优势”。

## 收束决策

- C-entry formal 资产、报告和测试保留为版本化 shadow evidence。
- 不修改既有 parent，不把 fixed-large 控制降级，不用新 Gate 输入掩盖 `0/9` 优势结果。
- R5 learned router、结构增长默认接线、Skill/MCP/provider/client/CUDA 均继续冻结。
- 进入下一阶段前，先做 capacity-parity audit/pre-registration：明确参数量、实际更新步数、checkpoint 总量和训练/推理预算的可比边界；在边界冻结前不追加新训练、不调整学习率、不复活旧结构路线。
