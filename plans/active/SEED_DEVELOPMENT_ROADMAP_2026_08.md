# Seed / Taiji Native v1 当前开发路线入口

> 本文件是当前执行路线的唯一入口，不再承载逐次实现日志。详细内容按职责拆分到 `active/roadmap/`；历史执行记录位于 `archive/history/`。

## 当前项目口径

- **Taiji** 是完整的原生认知架构与模型；**Seed** 是项目、产品和运行时。
- `taiji/` 当前是 TSK-v8 substrate/kernel 及其 native 纵切片；它是 Taiji 的可执行基础，不等于完整认知架构。
- Legacy NeuroPlex 是冻结的 Transformer 离线对照，不进入 Taiji cognition。
- HF、Qwen、Transformers、GGUF 等只在语言/数据/provider 适配边界出现，不作为 Taiji 核心认知主体或格式切换。

## 详细路线

| 文件 | 责任 | 状态 |
|---|---|---|
| [01_SCOPE_AND_PHASES.md](roadmap/01_SCOPE_AND_PHASES.md) | 目标、原则、P0–P7 阶段和阶段退出门槛 | 当前参考 |
| [02_GATES_AND_CI.md](roadmap/02_GATES_AND_CI.md) | 当前仍有效的门禁、CI 纪律、停止项和阻塞边界 | 当前参考 |
| [03_CURRENT_EXECUTION.md](roadmap/03_CURRENT_EXECUTION.md) | 当前实现快照、唯一下一步、W0–W7 顺序 | 当前执行 |
| [04_EXECUTION_PLAN.md](roadmap/04_EXECUTION_PLAN.md) | C0–C8 后续工作包、依赖、产物、Gate 和工程纪律 | 当前参考 |
| [IMPLEMENTATION_STATUS_2026_08.md](../reference/IMPLEMENTATION_STATUS_2026_08.md) | 代码事实、owner 和能力声明 | 当前参考 |

## 固定执行规则

1. 只从 [03_CURRENT_EXECUTION.md](roadmap/03_CURRENT_EXECUTION.md) 的“当前唯一下一步”开始；归档文档中的下一步全部是历史语境。
2. 每完成一个 slice，先更新实现状态、相关架构合同和本路线入口，再提交；CI 红线、checkpoint 往返和真实 canary 不能被“已完成”文字替代。
3. 研究 Gate、产品能力和演示包装必须分开标注；小型模拟只证明 S0 机制，不替代 replay、真实工作台或 packaged-client 证据。
4. CUDA、视觉体验、provider watchdog、interaction-group、开放域学习和结构自进化均保留在路线中；硬件不可用时只能标记 `hardware-blocked`，不得伪造完成。

## 收敛后的路线顺序

W0–W6 与 W7-R1/R2 已形成基线。W7 不再把彼此无实现依赖的验证线伪装成严格串行：

- W0–W3：原生 Workbench 合同、语言/IDE、受控执行、MCP-shaped 工具和有限循环。
- W4–W6：产品语义/Legacy 边界、客户端真实性、provider 与前端 native facade 收口。
- W7-R1/R2：provider watchdog 与 interaction-group 已完成 S0/S1/S2 基线。
- W7-R3：页面层已通过，Windows shell 现场证据保持 `tool-blocked` 独立补证。
- W7-R4：真实 CUDA 主机上的 profile、跨设备 checkpoint 和数值一致性保持 `hardware-blocked`。
- W7-R5：当前活跃主线，按 **G1 合同分离 → R5A 知识内化 → R5B 效应器成长 → R5C 结构自进化** 推进。

R3/R4 未关闭时不得宣称对应能力完成；但工具/硬件阻塞不再冻结与它们无依赖的 R5 CPU/native 工作。R5 也不得反向替代 R3/R4 证据。

## 历史记录入口

- [P8 产品与发布记录](../archive/history/SEED_ROADMAP_RELEASE_LOG_2026_08.md)
- [前期 Gate 与路线校准记录](../archive/history/SEED_ROADMAP_EXECUTION_FOUNDATION_2026_08.md)
- [Workbench Closure 规划记录](../archive/history/SEED_ROADMAP_WORKBENCH_PLAN_2026_08.md)
- [Workbench Closure 进展记录](../archive/history/SEED_ROADMAP_WORKBENCH_PROGRESS_2026_08.md)
- [旧入口与状态快照](../archive/history/PLANS_INDEX_STATUS_SNAPSHOT_20260829.md)
- [Gate 与 CI 历史](../archive/history/SEED_GATE_CI_HISTORY_2026_08.md)
- [W7 执行蓝图快照](../archive/history/SEED_W7_EXECUTION_PLAN_SNAPSHOT_20260829.md)

## 维护边界

核心需求、Taiji native 架构、身份与产品边界仍分别维护在 active 根目录的四份架构文档中；测试调试日志、已完成执行记录和被替代路线只能进入 archive。任何新计划文件必须先说明 owner、输入/输出合同、checkpoint 边界、Gate 和失败时的回滚路径，避免再次形成第二个“唯一下一步”入口。
