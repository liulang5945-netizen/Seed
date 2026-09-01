# R5C-S42：SeedRuntime artifact store bridge 与原子交接 Gate

## 目标

把 S41 的外部 artifact store 接入 SeedRuntime 的显式 batch API：调用方只提供 batch-bound candidate 到 artifact digest 的引用和 replay，runtime 负责先完成引用解析与 key 校验，再一次性进入现有 native artifact/admission contract。

## 计划边界

- 新增明确的 SeedRuntime store bridge，不让调用方直接拼接对象、绕过 batch/candidate/parent/replay 校验，也不把 store 变成 runtime 的认知或 retention 所有者。
- 所有 artifact 引用先在 runtime 状态改变前完成 digest 解析；未知 candidate key、非法 digest、缺失文件和篡改文件必须 fail-closed 且 runtime 全局状态不变。replay 错配继续遵守 native 的候选级 fail-closed 合同，不污染 sibling、topology 或全局预算。
- 合法外部 artifact 继续复用现有逐 candidate/batch 生命周期；重复提交保持既有 `already_applied` 语义，artifact provenance 与 checkpoint 保持。
- bridge 不执行物理删除、不自动清理 store、不修改 provider/Transformer/CUDA/前端边界。

## Gate

- 通过 store digest 引用的 artifact 可在 SeedRuntime checkpoint 重启后完成 admission，并能幂等重复。
- unknown key 在解析前原子拒绝；非法/缺失/篡改 artifact 的 store 解析失败时 runtime 状态不变。
- 现有 batch reservation、budget、topology、parent digest、artifact provenance 语义不改变；store 读取失败不会留下部分 batch 更新，replay 错配只允许 native contract 记录对应候选的终态，不得波及 sibling、topology 或全局预算。

## 明确不声明

本 slice 只覆盖 native CPU SeedRuntime 的 artifact-store bridge，不声明 store 自动垃圾回收、无限结构扩张、开放域收益、CUDA、前端、Windows shell、CI 或通用智能。

## 证据交付

新增 API bridge、定向测试、独立 evaluator、JSON Gate 报告，并同步 active/reference/manifest；按用户决定不在本 slice 处理 CI，也不提交或推送。
