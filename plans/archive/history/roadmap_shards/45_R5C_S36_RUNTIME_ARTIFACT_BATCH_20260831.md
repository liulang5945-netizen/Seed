# R5C-S36：SeedRuntime artifact batch 投影与稳定结构绑定

## 目标

让 replay-bound artifact batch 从 native `TSKV8Adapter` 合同正确进入 `SeedRuntime` 工作台边界：运行时只做线程安全、稳定 payload 投影，不重新计算 measured facts，不绕过 Taiji 的 candidate、artifact、validation、admission、rollback 和 retention 所有权。

## 计划边界

- 审计 `SeedRuntime` 已有的单候选/批次 artifact wrapper，明确输入、返回值、锁、异常和 checkpoint 边界；缺失的 wrapper 只做最小补齐。
- 验证 SeedRuntime 在 runtime checkpoint 重启后能消费外部保存的 measured artifact/replay，并维持 candidate/artifact batch digest、状态和幂等语义。
- 验证未知 artifact/replay key、malformed artifact、stale parent 和跨 batch 输入仍由 Taiji 原子拒绝，runtime 不吞错、不偷偷重算指标。
- 若 runtime provider 初始化会改变 structural parent digest，必须把它作为稳定结构绑定问题定位并修正；不以放宽 digest 校验掩盖 provider 或 checkpoint 漂移。
- 维持 native CPU 边界，暂不引入 CUDA、后台自治、开放域数据或前端状态。

## Gate

- SeedRuntime wrapper 与 native adapter 返回语义一致，且调用后只出现预期的结构/ledger 变化。
- 外部 artifact 保存后，runtime 重启能完成 measured admission；重复消费幂等。
- 未知 key、篡改、stale parent、跨 batch 输入 fail-closed 且状态原子。
- checkpoint 前后 artifact provenance、candidate state、topology、budget 和 provider/structural digest 绑定明确可审计。

## 明确不声明

本 slice 只覆盖 SeedRuntime/native CPU artifact batch contract，不声明开放域收益、无限扩张、自动增预算、CUDA、前端、Windows shell、CI 或通用智能。

## 证据交付

新增定向测试、独立 evaluator、JSON Gate 报告，并同步 active/reference/manifest 计划事实。
