# Seed / Taiji 计划与架构入口

> 这是计划资料的总导航。路线只从 [active/SEED_DEVELOPMENT_ROADMAP_2026_08.md](active/SEED_DEVELOPMENT_ROADMAP_2026_08.md) 进入，即时动作只从 [03_CURRENT_EXECUTION.md](active/roadmap/03_CURRENT_EXECUTION.md) 读取；历史文档中的“下一步”一律不再生效。

## 当前收敛结论（2026-08-31）

- W0–W6、W7-R1/R2 已形成基线；R3 页面层已有证据但 Windows shell 为 `tool-blocked`，R4 CUDA 为 `hardware-blocked`。
- R5A/R5B 和 R5C-S0–S52 已建立知识、效应器与结构候选的验证、准入、rollback、checkpoint、lineage，以及显式 artifact consumption policy；新运行时默认 `verified-only`，历史 replay 只能显式使用 `legacy-compatible`。
- 当前最大能力缺口已经从“结构机制不存在”转为“自然语言任务虽可形成 Goal evidence、resolved evidence、语言 evidence、受限语义分解，provider 边界、确定性 lifecycle seam、真实 Workbench interaction-group 闭环、互补组收益、train-only 选组稳定性、同 capability 对的 context 留出迁移、异质成员/未见组合的受限 transfer、三轮 future Workbench 对照、三轮真实在线 Outcome 写回/准入/回滚、在线反馈到结构候选的受控桥接、首次结构扩容净收益、两个独立周期的连续结构增长、editor+MCP 跨域结构收益与旧 workspace 保留，以及 terminal 三域治理/审批/资源/失败恢复已建立；P2-8 完成单步闭环，P2-9 完成声明式 semantic grounding，P2-10 完成无外部绑定的多步 grounding/recovery，P2-11 完成无外部最终绑定的 IDE 语言链，P2-12 完成 Taiji-owned digest-checked 自然语言受控写入，P2-13 完成 API/OpenAPI/前端两阶段 transport，P5-1 完成协议编排模块化，P5-2 完成 grounding engine 模块化，P5-3 完成执行边界模块化，但真实聊天 UI 用户旅程、真实 provider 质量和更广开放域收益仍未验收”；S52 已收口 artifact 基础设施线，主线转入可验证的学习/自进化收益，不继续无限增加 artifact-store 微分片。
- P6-1a 已建立独立 `SemanticEvidenceProvider` / `SemanticProviderRequest` seam，并通过内容寻址、无执行字段、Taiji admission 前置和无 Workbench 副作用 Gate；P6-1b 已用测试注入 provider 验证聊天端 `interpret → plan → execute` 只读 Workbench 旅程，真实 provider artifact / 浏览器现场仍进入 P6-1c 单独验收。
- provider watchdog、interaction-group、小型模拟 Gate、Windows 客户端、Legacy 残留清理和 CUDA 均保留在阶段总计划中；CI 按用户决定暂缓，不得把未运行写成通过。
- 核心架构讨论继续留在 active/reference；完成 Gate 的过程文档将按新的收敛计划批量归档。

## 目录职责

| 目录 | 只允许放什么 |
|---|---|
| `active/` | 当前执行入口与当前路线分片；根目录保留项目身份和架构合同 |
| `active/roadmap/` | 当前仍有效的阶段、门禁和执行摘要 |
| `reference/` | 当前代码事实、owner、能力声明和边界参考；不提供独立执行顺序 |
| `archive/` | 已完成、被替代或仅用于追溯的设计、审计、实现和历史记录 |
| `manifests/` | 可执行 Gate、runtime 结构和实验合同的版本化 manifest |

## 当前架构合同

| 文档 | 权威范围 |
|---|---|
| [TAIJI_CORE_REQUIREMENTS.md](active/TAIJI_CORE_REQUIREMENTS.md) | 长期使命、CR-1–CR-10 和不可归档的核心需求 |
| [TAIJI_NATIVE_ARCHITECTURE_V1.md](active/TAIJI_NATIVE_ARCHITECTURE_V1.md) | Taiji 感知、世界状态、工作空间、记忆、推理、规划、生成、学习和硬编码治理 |
| [ARCHITECTURE_DIRECTION_2026_08.md](active/ARCHITECTURE_DIRECTION_2026_08.md) | 词表、身份、成熟技术采纳规则、Transformer/Legacy 边界 |
| [SEED_ARCHITECTURE.md](active/SEED_ARCHITECTURE.md) | Seed 产品/runtime 所有权、API、Workbench 和 checkpoint 边界 |
| [IMPLEMENTATION_STATUS_2026_08.md](reference/IMPLEMENTATION_STATUS_2026_08.md) | 当前代码事实和已验证/未验证能力的短摘要 |

## 当前执行路线

| 文档 | 作用 |
|---|---|
| [SEED_DEVELOPMENT_ROADMAP_2026_08.md](active/SEED_DEVELOPMENT_ROADMAP_2026_08.md) | 唯一路线入口和 W0–W7 / R1–R5 顺序 |
| [01_SCOPE_AND_PHASES.md](active/roadmap/01_SCOPE_AND_PHASES.md) | 目标、原则、P0–P7 阶段和退出门槛 |
| [02_GATES_AND_CI.md](active/roadmap/02_GATES_AND_CI.md) | 当前仍有效的门禁、CI 纪律、停止项和阻塞边界；不再承载事故日志 |
| [03_CURRENT_EXECUTION.md](active/roadmap/03_CURRENT_EXECUTION.md) | 当前实现快照和当前唯一下一步 |
| [04_EXECUTION_PLAN.md](active/roadmap/04_EXECUTION_PLAN.md) | W7-R5 已完成分片、原始 Gate 与历史阶段定义的证据索引 |
| [62_POST_S51_PROJECT_CONVERGENCE_20260831.md](active/roadmap/62_POST_S51_PROJECT_CONVERGENCE_20260831.md) | S52 后的端到端 Workbench、语言、自进化收益、工程、产品、CI 与 CUDA 顺序 |

## 归档原则

- 核心需求、Taiji 原生架构、身份边界和未关闭缺口必须留在 active/reference，不能只留在 archive。
- 测试调试过程、一次性探针、已完成执行日志、被替代路线和旧 Transformer 设计进入 archive。
- 归档不等于删除：所有历史内容都保留可追溯路径，但不得成为新的执行入口。
- 任何新增计划必须明确 owner、输入/输出合同、checkpoint 归属、Gate、失败模式和回滚路径；能合并到现有文件就不新增文件。
- 每次路线决策或实现收口后，先更新本目录的事实源，再提交代码；不得并列维护两个“当前唯一下一步”。

## 归档索引

完整清单和归档使用规则见 [archive/README.md](archive/README.md)。路线执行的大型原文已按产品发布、前期校准、Workbench 规划/进展、Gate/CI 历史和 W7 蓝图快照拆分；旧架构妥协审计按来源、缺口和候选方向拆分。
