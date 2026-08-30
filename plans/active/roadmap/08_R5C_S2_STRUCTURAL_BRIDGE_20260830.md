# W7-R5C-S2：candidate-only structural bridge

## 目标

把 R5C-S1 的跨任务 `StructuralGrowthEvidenceProjection` 单向接入已有
`AdaptiveStructuralGrowthController`，但把“发现成长压力”和“改变模型结构”明确拆开。
S2 只允许生成可恢复的候选，不允许在 pressure projection 阶段提交拓扑、消耗结构预算或执行
神经元/突触增长。

## 已实现

- `TSKV8Adapter.propose_structural_candidate_from_pressure()` 接受带有 holdout 证据的
  content-addressed projection，并把 controller 的判定封装成现有
  `StructuralProposalCandidate`。
- 同一 projection digest 只能消费一次；重复 projection fail-closed，不重复改变 controller
  状态，也不重复排队候选。
- 候选绑定 projection evidence、controller region、资源成本和 parent checkpoint digest。
- 外部封存 projection 的 `last_tick` 会推进 adapter 的 structural runtime clock，保证候选
  与 checkpoint 的 source tick 一致；这只更新证据时钟，不改变 topology 或 budget。
- candidate materialization 仍只创建 pending topology proposal；真实 admission、shadow、
  holdout/retention、lesion、rollback 均留给后续 Gate。

## Gate 证据

- `tests/taiji_native/test_structural_pressure.py`：4 个 S1/S2 定向测试通过。
- `scripts/training/eval_taiji_structural_bridge.py`：验证候选生成、projection 去重、父
  checkpoint 绑定、checkpoint 恢复、materialization 后仍不改 topology、不消耗 budget。
- 报告：`reports/taiji_w7_r5c_s2_structural_bridge_20260830.json`，`gate.passed=true`。

## 明确未做

- 没有自动 admission、没有真实突触/神经元增长、没有删除或替换旧结构。
- 没有把 holdout 标签写入 train replay，也没有把 candidate-only 证据扩大为开放域智能证明。
- CI 全量回归仍按用户决定暂缓；本 slice 只运行与改动范围匹配的定向验证。

## 下一阶段准入

R5C-S3 只能在候选层建立 shadow/holdout/retention/lesion 的结构验证 Gate，并且必须保存
trial checkpoint、保留 parent active、让失败候选原子拒绝和资源归还；通过前不能调用真实
topology admission。
