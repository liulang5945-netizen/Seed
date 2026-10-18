# PLAN-N1-115：seed13/position 五组参数全因子交叉归因

**状态：提案，待 owner 裁定。** PLAN-N1-112 已显示失败随 candidate 权重来源变化；PLAN-N1-113 将 reader 组差异定位为终态总参数差值平方和的85.0%，其中 `readout.weight` 占64.6%；PLAN-N1-114 显示 candidate 的 7 字符材料更新有较高 QA loss，且 transition 更新时多个组的裁剪频率不同。现需区分 reader/writer/local/context/relation 中哪些权重变化控制已观测到的训练 transition 退化。

## 固定设计

- 每组参数的来源为 run100 baseline 或 N1-110 candidate。构造五个二元因素的完整 (2^5=32) 状态组合；不交换观察历史，所有组合固定使用 baseline history。PLAN-N1-112 已证明在该 fresh-session evaluator 下，固定权重时切换两种 training-session history 的 384/384 行输出相同，且运行代码只读取当前 session key。
- 每个状态组合测同一 48 个 train transition 父项、四条件、original/rename 两视图，共384条。mask=0（全 baseline 参数）与 mask=31（全 candidate 参数）均固定 baseline history，从 r1 报告分别复用 `baseline_weights+baseline_history` 和 `candidate_weights+baseline_history` 两组各384行；只查询剩余30个参数混合组合，共 **11,520 次新模型查询**。
- 主要读数预注册为每个组合/视图的四条件 joint-four exact、每条件 exact、original-source direction×turn cell 以及逐题结果。对每组计算跨16条单因素边的平均变化，并给出五因素 Shapley 分解；这是该固定训练集上的机制读数，不作泛化或能力主张。
- 原生锚点必须与 PLAN-N1-112 的 sealed rows、joint-four 和 cell correction 逐项一致；任一不符即停止，不查询任何 hybrid。
- CPU 上限600秒；两枚checkpoint payload读取各1次，0参数更新、0 optimizer step、0 checkpoint保存、0 dev/holdout/confirmation查询，`can_promote=false`。write-once 报告：`reports/taiji_n1_position_five_group_parameter_factorial_s13_20261018.json`；单次执行，不重试。
- 执行前另运行零 checkpoint-load/零-query readiness，核对两枚源报告、r1锚点、cell correction、session-scope审计、manifest和27项source guards；readiness 写入 `reports/taiji_n1_position_five_group_parameter_factorial_s13_20261018_readiness.json`。只有复用锚点通过原报告行集合核验后，才执行30个hybrid。

## 判读与授权边界

主效应和交互项用于确定下一项训练修复针对哪个共享参数组，不改变原44/48训练门、数据面或晋升条件。该组间干预会产生新的模型输出组合，属于新的 train-only 查询测面；需要 owner 明确裁定后才实现并运行。它不会授权训练或新确认面。

## 零查询实现准备

- r1 group-factorial runner 已实现并受 `状态：已批准执行` 闸门保护；可将每组参数从 baseline/candidate 两枚权重源组合为32态，复用 mask0/mask31 两组 PLAN-N1-112 已有输出，真实查询仅发给30个 hybrid。
- 4项合同测试通过：二元空间/11,520预算、混合 state 组映射、交互 Shapley 计算、无训练/优化器/保存调用。Ruff、Black、Python 编译通过。
- `--validate-only` readiness 报告：[`report`](../../reports/taiji_n1_position_five_group_parameter_factorial_s13_20261018_readiness.json)，SHA-256 `d0e61b1b6f1b28edeefcabb273b3d5fe6d512d29a416c320e749b225b56d6f209`；32项 source guards、768锚点行复用、30个hybrid state/11,520新查询。checkpoint payload reads、model loads/constructions、model queries、updates 均为0；checkpoint在预检中只做哈希核验，没有解码。
- 本 readiness 对应的提案快照 SHA-256 为 `74de3c24ce641427997db53cd2a005ff5e528d457ce2acc06fda5a394281339d`；本段预检记录后附加，不改变冻结设计。真实checkpoint restore与输出查询仍需 owner 裁定后再做。

## 唯一建议的下一步

### 零查询实施准备（2026-10-18）

- r1 factorial runner 的纯状态混合、Shapley 计算、锚点复用、session-neutral query contract 已实现；4项合同测试、Ruff、Black、编译均通过。
- `--validate-only` readiness：`reports/taiji_n1_position_five_group_parameter_factorial_s13_20261018_readiness.json`，SHA-256 `d0e61b1b6f1b28edefcabb273b3d5fe6d512d29a416c320e749b225b56d6f209`。32项来源守卫全匹配，原始/改名 views 各192项，sealed anchors 768行完全复用且逐项对账；30个hybrid状态每态384项，共11,520计划查询。
- readiness 阶段 0 checkpoint payload read、0 model load/construction/query、0 updates/save；两个 checkpoint 只做SHA检查，没有解码。预检提案快照SHA-256 `74de3c24ce641427997db53cd2a005ff5e528d457ce2acc06fda5a394281339d`，本结果追加未改测面或判据。
- 当前代码在 `--execute-approved` 前要求本提案状态为“已批准执行”；此状态仍未设置，因此 runner 不会自行越过裁定闸门。

唯一下一步：owner 批准或终止本五组全因子诊断。批准后只运行一次；若停止，则依据 PLAN-N1-112/113/114 的现有静态证据改写共享读出保持方案，不自动改 gate。
