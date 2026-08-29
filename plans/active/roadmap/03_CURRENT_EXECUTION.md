# Seed / Taiji 当前执行状态

> 本文件是当前执行状态的唯一实时摘要。详细历史记录按日期保存在 `plans/archive/history/`，不从历史文档恢复下一步。

## 2026-08-29 状态快照

### 已闭合的执行层

- W0–W3 的 native Workbench 版本合同、真实 workspace 只读/受控写入、语言证据与 IDE 自主切换、终端、审批、MCP registry、有限循环、checkpoint continuation 和真实跨文件任务闭环已具备证据。
- W4 的 HF/GGUF/Transformer/Legacy 产品语义清理已完成；合法的 provider 只负责语言 realization，不选择工具、不拥有 Taiji cognition。
- W5 的客户端真实性接入已完成；前端以 native capability、runtime/provider/homeostasis/training/knowledge evidence 为状态来源，旧 Legacy 调用不能回流。其中 homeostasis 一路直到 2026-08-29（提交 `cd39632`）才真正接通：此前 `taiji/adapter.py` 没有 homeostatic 读访问器，`api/models_runtime.py` 的 `LifeNeedsPayload` 又给四个需求字段各设了 `default: 50.0`，于是空 `needs: {}` 在传输层被补成四个编造值、原生 `stress` 被静默丢弃——**客户端显示「已接入原生」而数据全是假的，且门禁全绿**。现已改为 `dict[str, float] = {}`（缺测就不出现，不再编造），并由 `tests/seed/test_native_life_status.py` 的反编造断言看守。
- W6 的 typed native facade 和产品页拆分已完成至 Settings 共享面板收口；组件不越权持有 native API 副作用，前端回归保持可见。
- current Gate（recovery portfolio 客户端审计回放）的 S0/S1 已在代码层闭合，S2 packaged-client 现场取证也已完成：只读绑定键 `GET /taiji/recovery-branch/context`、结构化错误码、`RecoveryPortfolioAuditPanel`（右栏属性检查器，事件投影驱动，stale-keep-last / 切 loop 清空 / 只读）在最终 Legacy-off 客户端真实 Workspace 路径可见。客户端实际观察到 native checkpoint `seed:seed_corpus.pt` 和 capability snapshot revision `4`；所有关键 API 请求 8138/200，无页面错误或 Legacy 标记。详见 [02_GATES_AND_CI.md §14.20](02_GATES_AND_CI.md) 与 [S2 证据](../../../reports/packaged_client_s2_20260829.json)。
- P6 provider artifact、provider startup、客户端观测和训练/回滚合同已接通；P7 executive、grounding、world evidence、bounded successor graph 和 recovery portfolio 已形成可恢复只读闭环。

### 必须保持的边界

- 当前默认自主执行只覆盖 Taiji-owned、freshness-valid、受能力快照约束的只读 Workbench 路径；写入自治、开放域自然语言工具选择和外部 MCP 生命周期仍未宣称完成。
- recovery portfolio 审计 Gate 的 S2 packaged-client 现场取证已完成；本次启动展示的是无持久化 portfolio 的结构化空态，非空 branch/tombstone 排序仍由 S0/S1 replay 证据覆盖，不把空态 canary 宣称为非空恢复演示。最终客户端同时修复了 Qt 无 GPU 启动降级、真实后端端口透传、受限数据目录的非阻塞降级和 `/api/health.taiji_available` 状态不一致。
- provider watchdog、interaction-group、视觉/桌面体验、CUDA、开放域学习和结构自进化没有取消，只能按 W7 顺序推进；CUDA 在当前 CPU-only 主机上保持 `hardware-blocked`。
- 训练前必须先验证 checkpoint 能保存、恢复并继续产生等价的 lineage、预算、结构和 provider artifact 状态；任何只在内存中成立的训练结果不算 Gate 证据。该往返等价性准入已由 [04_EXECUTION_PLAN.md §3](04_EXECUTION_PLAN.md) 的 `test_checkpoint_roundtrip_contract.py`（3 例）满足。

## 当前唯一下一步

进入 **W7-R1-S2 provider watchdog packaged-client 观测**：S0/S1 已通过，沿已冻结的 [R1 manifest](../../manifests/taiji_w7_r1_provider_watchdog_v1.json) 在 Legacy-off 的真实打包客户端记录 provider artifact digest、健康状态、结构化降级和 backend/network 绑定；本步只观测服务端真实投影，不让客户端自行切换 provider。

## 后续唯一顺序

1. 完成 W7-R1：provider watchdog 的 S2（S0/S1 已通过）。
2. 按 R2 → R3 → R4 → R5 推进；每个方向先做可证伪 Gate，再接入真实运行时，最后才更新产品展示。

## 更新规则

- 当前快照只保留已验证的能力和明确的未完成边界；历史数字、实验过程和一次性失败原因移入 archive。
- 新能力必须在这里留下“owner、真实输入、结构化输出、checkpoint 归属、失败模式和 Gate”；否则只能算实验记录。
- 若实现与本文件、架构文档或 CI 事实冲突，先暂停下一步，修正唯一事实源并提交，再恢复执行。
- W7-G0 的五份 manifest 与结构门禁已提交；R4 当前硬件状态仍为 `hardware-blocked`，不可用 CPU 结果替代 CUDA 证据。
