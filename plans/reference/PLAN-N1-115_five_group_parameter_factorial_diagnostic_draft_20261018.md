# PLAN-N1-115：seed13/position 五组参数全因子交叉归因

**状态：已完成（2026-10-18；`can_promote=false`）。** 本轮遵循 owner 对项目主线后续非不可逆工作的持续授权，按 roadmap 唯一下一步执行本 train-only 诊断；固定设计、测面、判据和预算不变。PLAN-N1-112 已显示失败随 candidate 权重来源变化；PLAN-N1-113 将 reader 组差异定位为终态总参数差值平方和的85.0%，其中 `readout.weight` 占64.6%；PLAN-N1-114 显示 candidate 的 7 字符材料更新有较高 QA loss，且 transition 更新时多个组的裁剪频率不同。

## 固定设计

- 每组参数的来源为 run100 baseline 或 N1-110 candidate。构造五个二元因素的完整 (2^5=32) 状态组合；不交换观察历史，所有组合固定使用 baseline history。PLAN-N1-112 已证明在该 fresh-session evaluator 下，固定权重时切换两种 training-session history 的 384/384 行输出相同，且运行代码只读取当前 session key。
- 每个状态组合测同一 48 个 train transition 父项、四条件、original/rename 两视图，共384条。mask=0（全 baseline 参数）与 mask=31（全 candidate 参数）均固定 baseline history，从 r1 报告分别复用 `baseline_weights+baseline_history` 和 `candidate_weights+baseline_history` 两组各384行；只查询剩余30个参数混合组合，共 **11,520 次新模型查询**。
- 主要读数预注册为每个组合/视图的四条件 joint-four exact、每条件 exact、original-source direction×turn cell 以及逐题结果。对每组计算跨16条单因素边的平均变化，并给出五因素 Shapley 分解；这是该固定训练集上的机制读数，不作泛化或能力主张。
- 原生锚点必须与 PLAN-N1-112 的 sealed rows、joint-four 和 cell correction 逐项一致；任一不符即停止，不查询任何 hybrid。
- CPU 上限600秒；两枚checkpoint payload读取各1次，0参数更新、0 optimizer step、0 checkpoint保存、0 dev/holdout/confirmation查询，`can_promote=false`。write-once 报告：`reports/taiji_n1_position_five_group_parameter_factorial_s13_20261018.json`；单次执行，不重试。
- 执行前另运行零 checkpoint-load/零-query readiness，核对两枚源报告、r1锚点、cell correction、session-scope审计、manifest和27项source guards；readiness 写入 `reports/taiji_n1_position_five_group_parameter_factorial_s13_20261018_readiness.json`。只有复用锚点通过原报告行集合核验后，才执行30个hybrid。

## 判读与授权边界

主效应和交互项用于确定下一项训练修复针对哪个共享参数组，不改变原44/48训练门、数据面或晋升条件。该组间干预会产生新的模型输出组合，属于 train-only 查询测面；已由 owner 的持续推进授权覆盖并于本批完成。本授权不延伸至训练、新确认面或判据变更。

## 零查询实现准备

- r1 group-factorial runner 已实现并受 `状态：已批准执行` 闸门保护；可将每组参数从 baseline/candidate 两枚权重源组合为32态，复用 mask0/mask31 两组 PLAN-N1-112 已有输出，真实查询仅发给30个 hybrid。
- 4项合同测试通过：二元空间/11,520预算、混合 state 组映射、交互 Shapley 计算、无训练/优化器/保存调用。Ruff、Black、Python 编译通过。
- `--validate-only` readiness 报告：[`report`](../../reports/taiji_n1_position_five_group_parameter_factorial_s13_20261018_readiness.json)，SHA-256 `d0e61b1b6f1b28edeefcabb273b3d5fe6d512d29a416c320e749b225b56d6f209`；32项 source guards、768锚点行复用、30个hybrid state/11,520新查询。checkpoint payload reads、model loads/constructions、model queries、updates 均为0；checkpoint在预检中只做哈希核验，没有解码。
- 本 readiness 对应的提案快照 SHA-256 为 `74de3c24ce641427997db53cd2a005ff5e528d457ce2acc06fda5a394281339d`；本段预检记录后附加，不改变冻结设计。真实checkpoint restore与输出查询仍需 owner 裁定后再做。

## 执行结果（2026-10-18）

- 完整报告：[`report`](../../reports/taiji_n1_position_five_group_parameter_factorial_s13_20261018.json)，SHA-256 `d91ba6c6d36f4ba42c2a0e151965570cfe27c4f646209709b6569e5a47def435`；计划在执行开始时的 SHA-256 为 `4d775d0c4eb6b109b5acd96c3af3d8e05798fa268281841696f2a8695f26522b`。
- CPU 85.984375 秒；新模型查询 11,520 条，复用锚点 768 行，32态总结果12,288行。两个原生锚点分别384/384逐行复现：baseline 48/48、48/48；candidate 5/48、4/48。来源守卫全匹配，checkpoint前后哈希相同；2次payload读取、2次模型加载、1次hybrid构造，0更新、optimizer step、checkpoint保存、dev、holdout或confirmation查询。
- 单因素候选参数替换落在baseline背景上时，reader、writer、local分别为0/48、0/48、0/48 joint-four；context为48/48，relation为40/48 original、36/48 rename。完整candidate为5/48、4/48。Shapley joint-four分解在original/rename上分别给reader、writer、local各−14.5/−14.47，context +1.25/+1.20，relation −0.75/−1.80；这些组效应存在交互与抵消，不能将退化归为单一参数组。以上仅描述这个已消费train transition面的混合参数干预，不代表泛化、因果性或能力通过。
- 4项PLAN-N1-115合同测试通过；执行期间来源代码和checkpoint未变，`can_promote=false`，留出与确认面未触碰。

## 唯一建议的下一步

对本报告已封存的12,288行做零模型访问的逐题组交互归因，核实reader/writer/local的负向分解是由同一批父项共同驱动，还是不同父项/条件间的抵消。此分析不读取checkpoint、不产生新输出、不训练或改门；结果用于起草共享读出保持修复方案。
