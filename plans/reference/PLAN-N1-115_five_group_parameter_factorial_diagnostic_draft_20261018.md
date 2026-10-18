# PLAN-N1-115：seed13/position 五组参数全因子交叉归因

**状态：提案，待 owner 裁定。** PLAN-N1-112 已显示失败随 candidate 权重来源变化；PLAN-N1-113 将 reader 组差异定位为终态总参数差值平方和的85.0%，其中 `readout.weight` 占64.6%；PLAN-N1-114 显示 candidate 的 7 字符材料更新有较高 QA loss，且 transition 更新时多个组的裁剪频率不同。现需区分 reader/writer/local/context/relation 中哪些权重变化控制已观测到的训练 transition 退化。

## 固定设计

- 每组参数的来源为 run100 baseline 或 N1-110 candidate。构造五个二元因素的完整 (2^5=32) 状态组合；不交换观察历史，所有组合固定使用 baseline history。PLAN-N1-112 已证明在该 fresh-session evaluator 下，固定权重时切换两种 training-session history 的 384/384 行输出相同，且运行代码只读取当前 session key。
- 每个状态组合测同一 48 个 train transition 父项、四条件、original/rename 两视图，共384条。两端纯 native 锚点的768行从 r1 密封报告与原生复现结果复用；只查询剩余30个 hybrid 组合，共 **11,520 次新模型查询**。
- 主要读数预注册为每个组合/视图的四条件 joint-four exact、每条件 exact、original-source direction×turn cell 以及逐题结果。对每组计算跨16条单因素边的平均变化，并给出五因素 Shapley 分解；这是该固定训练集上的机制读数，不作泛化或能力主张。
- 原生锚点必须与 PLAN-N1-112 的 sealed rows、joint-four 和 cell correction 逐项一致；任一不符即停止，不查询任何 hybrid。
- CPU 上限600秒；两枚checkpoint payload读取各1次，0参数更新、0 optimizer step、0 checkpoint保存、0 dev/holdout/confirmation查询，`can_promote=false`。write-once 报告：`reports/taiji_n1_position_five_group_parameter_factorial_s13_20261018.json`；单次执行，不重试。

## 判读与授权边界

主效应和交互项用于确定下一项训练修复针对哪个共享参数组，不改变原44/48训练门、数据面或晋升条件。该组间干预会产生新的模型输出组合，属于新的 train-only 查询测面；需要 owner 明确裁定后才实现并运行。它不会授权训练或新确认面。

## 唯一建议的下一步

批准或终止本五组全因子诊断。批准后才进入一次实现/preflight/run；若停止，则根据 PLAN-N1-112/113/114 现有静态证据另提共享读出保持方案，不自动改 gate。
