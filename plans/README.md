# Seed / Taiji 计划与架构入口

> 这是计划资料的总导航。路线只从 [active/SEED_DEVELOPMENT_ROADMAP_2026_08.md](active/SEED_DEVELOPMENT_ROADMAP_2026_08.md) 进入，即时动作只从 [03_CURRENT_EXECUTION.md](active/roadmap/03_CURRENT_EXECUTION.md) 读取；历史文档中的“下一步”一律不再生效。

## 当前收敛结论（2026-08-30）

- W0–W6、W7-R1/R2 和 R5-S0 已形成当前基线；R3 页面层通过但 Windows shell 为 `tool-blocked`，R4 为 `hardware-blocked`。
- R3/R4 保持独立补证线，不再把工具或硬件缺失当作整个项目停工条件；它们未通过时仍不得发布对应声明。
- R5-G1 已分别冻结知识内化与效应器成长 manifest；当前唯一开发动作是 R5A-S0 转换器 DTO 与确定性 replay Gate。
- 核心架构讨论继续留在 active/reference；2026-08-29 及以前的 Gate 过程和旧 W7 蓝图已归档。

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
| [04_EXECUTION_PLAN.md](active/roadmap/04_EXECUTION_PLAN.md) | C0–C8 后续工作包、依赖、产物和验收；不替代唯一下一步 |

## 归档原则

- 核心需求、Taiji 原生架构、身份边界和未关闭缺口必须留在 active/reference，不能只留在 archive。
- 测试调试过程、一次性探针、已完成执行日志、被替代路线和旧 Transformer 设计进入 archive。
- 归档不等于删除：所有历史内容都保留可追溯路径，但不得成为新的执行入口。
- 任何新增计划必须明确 owner、输入/输出合同、checkpoint 归属、Gate、失败模式和回滚路径；能合并到现有文件就不新增文件。
- 每次路线决策或实现收口后，先更新本目录的事实源，再提交代码；不得并列维护两个“当前唯一下一步”。

## 归档索引

完整清单和归档使用规则见 [archive/README.md](archive/README.md)。路线执行的大型原文已按产品发布、前期校准、Workbench 规划/进展、Gate/CI 历史和 W7 蓝图快照拆分；旧架构妥协审计按来源、缺口和候选方向拆分。
