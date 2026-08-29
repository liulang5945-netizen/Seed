# Seed / Taiji 当前执行状态

> 本文件是当前执行状态的唯一实时摘要。详细历史记录按日期保存在 `plans/archive/history/`，不从历史文档恢复下一步。

## 2026-08-29 状态快照

### 已闭合的执行层

- W0–W3 的 native Workbench 版本合同、真实 workspace 只读/受控写入、语言证据与 IDE 自主切换、终端、审批、MCP registry、有限循环、checkpoint continuation 和真实跨文件任务闭环已具备证据。
- W4 的 HF/GGUF/Transformer/Legacy 产品语义清理已完成；合法的 provider 只负责语言 realization，不选择工具、不拥有 Taiji cognition。
- W5 的客户端真实性接入已完成；前端以 native capability、runtime/provider/homeostasis/training/knowledge evidence 为状态来源，旧 Legacy 调用不能回流。
- W6 的 typed native facade 和产品页拆分已完成至 Settings 共享面板收口；组件不越权持有 native API 副作用，前端回归保持可见。
- P6 provider artifact、provider startup、客户端观测和训练/回滚合同已接通；P7 executive、grounding、world evidence、bounded successor graph 和 recovery portfolio 已形成可恢复只读闭环。

### 必须保持的边界

- 当前默认自主执行只覆盖 Taiji-owned、freshness-valid、受能力快照约束的只读 Workbench 路径；写入自治、开放域自然语言工具选择和外部 MCP 生命周期仍未宣称完成。
- provider watchdog、interaction-group、视觉/桌面体验、CUDA、开放域学习和结构自进化没有取消，只能按 W7 顺序推进；CUDA 在当前 CPU-only 主机上保持 `hardware-blocked`。
- 训练前必须先验证 checkpoint 能保存、恢复并继续产生等价的 lineage、预算、结构和 provider artifact 状态；任何只在内存中成立的训练结果不算 Gate 证据。

## 当前唯一下一步

建立 recovery portfolio 的客户端审计回放视图 Gate：在已有 native projection 消费层上增加只读审计模型/视图，按 revision 展示 branch 生命周期、容量压力、source evidence/after-state lineage 和 eviction tombstone；视图不得触发 maintain/select/execute，也不得显示可直接复用的 parameters。通过前不进入写入自治、开放域自然语言工具选择、CUDA kernel 或视觉包装。

## 后续唯一顺序

1. 完成上述 recovery portfolio 只读审计视图，并用故意破坏 revision、lineage、eviction 和参数脱敏的测试证明它会变红。
2. 进入 W7-G0，先把 provider、interaction-group、视觉、CUDA 和开放域 R5 的输入/输出合同冻结到真实 capability、trace、资源预算和 checkpoint 版本上。
3. 按 R1 → R2 → R3 → R4 → R5 推进；每个方向先做可证伪 Gate，再接入真实运行时，最后才更新产品展示。

## 更新规则

- 当前快照只保留已验证的能力和明确的未完成边界；历史数字、实验过程和一次性失败原因移入 archive。
- 新能力必须在这里留下“owner、真实输入、结构化输出、checkpoint 归属、失败模式和 Gate”；否则只能算实验记录。
- 若实现与本文件、架构文档或 CI 事实冲突，先暂停下一步，修正唯一事实源并提交，再恢复执行。
