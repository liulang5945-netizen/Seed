# R5C-S43：多 candidate artifact 引用预解析与原子性 Gate

## 目标

验证 SeedRuntime artifact store bridge 在多 candidate 引用场景下先完成全部外部 artifact 解析，再进入 native batch contract；第二个 artifact 缺失、篡改或非法时，第一个已解析 artifact 也不能提前触发 admission。

## 计划边界

- 创建一个真实 Workbench 双 candidate batch，准备一个合法外部 artifact 和一个不可解析引用。
- 通过 bridge 提交两项引用，验证 store 预解析失败时 runtime topology、budget、candidate/batch 状态和 provenance 全部不变。
- 在同一 checkpoint 上只提交合法 artifact，完成一次 admission 与重复幂等，证明失败预解析不会污染后续合法 sibling。
- 不改变 native batch 的候选级 replay 失败语义，不引入批次外的自动重试或隐式回滚。

## Gate

- 多 candidate 引用必须 all-or-nothing 通过 store resolution；任一引用失败时不得部分消费其他 artifact。
- 预解析失败后合法 candidate 仍可在 checkpoint 上 admission，预算只扣一次，重复提交幂等。
- store bridge 的 atomicity 不改变 parent、replay、batch、retention 或 rollback 所有权。

## 明确不声明

本 slice 只覆盖 native CPU SeedRuntime 多 artifact store 预解析，不声明多 batch 自动编排、无限结构扩张、开放域收益、CUDA、前端、Windows shell、CI 或通用智能。

## 证据交付

新增定向测试、独立 evaluator、JSON Gate 报告，并同步 active/reference/manifest；按用户决定不在本 slice 处理 CI，也不提交或推送。
