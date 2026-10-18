# PLAN-N1-118：relative-cap normalized joint RWL trust region

**状态：按 owner 持续推进授权进入工程、zero恢复前置及同配方训练。** N1-117已完成，但seed13/position仍未过原joint-four门；本方案只修订RWL共享位移的尺度归一化，不换数据、训练目标、模型或判据。

## 证据与假设

- N1-117完成221,184次更新、18/18 zero/final独立恢复；position seed11/12为48/48、48/48，seed13为24/48、23/48，rename transition为41/48。相较N1-110 seed13的5/48、4/48与38/48，联合共享半径有实际改进，但仍不足。
- N1-117逐周期归因发现，seed13早期writer/local在N1-110各自relative clip约0.52/0.23与0.74/0.30；N1-117的绝对RWL半径受reader的大参数范数主导，共享scale约0.98/0.99，早期几乎不触发。seed13 QA loss在四周期均高于N1-110。逐父项对账显示原始救回19项且无回退，改名救回20项但回退`train-078`，两视图仍各有24项joint-four全错。
- 解释：单一绝对半径让大范数reader主导了writer/local的更新上限，弱化原先各组相对尺度保护。新方案在耦合RWL向量中按每组自身相对cap归一化，维持共同方向，同时守住与三个独立relative cap合计相同的位移能量预算。

## 唯一配方改动

- 对每组 `g ∈ {reader, writer, local}` 定义 `c_g = 0.001 × max(||θ_g||₂, 1)`，`r_g = ||Δ_g||₂ / c_g`，其中 `Δ_g` 是未投影Adam提案。
- 采用单一共享比例 `s = min(1, √3 / sqrt(r_reader² + r_writer² + r_local²))`，三组所有参数的Adam位移统一乘 `s`。因此归一化总半径满足 `sum((s r_g)²) ≤ 3`，不超过三组各自在既有relative cap上的总平方位移预算；不同组不能再独立缩放。该联合半径允许一个组局部超过1个旧cap，但上限受共同归一化能量约束。
- context/relation仍按现行PLAN-N1-100的独立relative cap处理；Adam梯度、moment、RNG、目标函数与checkpoint bundle不变。

## 固定训练协议

- 沿用N1-110/N1-117同一长度平衡schedule、同一192项train来源、同一paired-question目标、同一fresh seed11/12/13 × position/terminal/cut九端点；每端点24,576更新、总221,184。不得生成新材料或改变观测/名称分配。
- 使用新write-once输出根 `data/checkpoints/n1_normalized_coupled_rwl_length_balanced_20261018/` 与独立报告/attempt/progress路径；保留N1-116失败空目录和N1-117九端点。
- 先通过归一化半径纯计算合同、九端点schedule readiness与完整SHA-256 source binding，并完成9个fresh-zero完整bundle保存及独立进程恢复（模型、Adam、RNG、history、policy identity）。恢复或守卫失败，不训练。
- zero前置CPU≤300秒；正式批次总CPU≤18,000秒。每端点zero/final恢复完整检查，单批一次，不追加或重试。

## 判据与隔离

- 完整复用PLAN-N1-110的原 `training_sufficient` predicate：每个position seed original/rename joint-four各≥44/48，rename transition free-exact≥40/48。不得按终点趋势改门。
- dev、holdout、confirmation查询均为0；训练失败则不生成或查询v12确认面。训练门通过也不自动创建确认面；需要后续独立计划。`can_promote=false`。
- N1-117离线读数只用于这次参数尺度修订，不代表因果确认或未见泛化。N1-118报告必须绑定N1-110和N1-117来源报告、六个位置训练shard、长度平衡schedule、失败/归因报告和全部新代码SHA。

## 唯一下一步

实现并测试normalized joint-RWL projection与独立N1-118 runner；完成零模型schedule readiness和source binding后，执行九个zero checkpoint保存/恢复前置。全部通过后，按固定协议运行一次221,184更新训练，结束时只按原门结案。
