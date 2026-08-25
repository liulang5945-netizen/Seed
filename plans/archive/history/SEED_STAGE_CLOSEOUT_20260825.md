# Seed 当前阶段收束记录

日期：2026-08-25

基线：`main`，提交 `2395bc8` 及其后续整理提交

范围：Taiji 原生边界、硬编码容量规划、Transformer 清理边界、S1 产品体验、计划体系整理

## 已收束成果

- Taiji 原生核心不导入 `seed`、`neuroplex` 或 `transformers`，raw-byte 感知—状态—记忆—行动闭环保持独立。
- Transformer/NeuroPlex 已被定义为冻结的离线对照与显式兼容扩展；当前不删除 `neuroplex/`，避免破坏产品兼容契约和同预算对照证据。
- `CapacityPolicy` 和参数预算已经接管区域、稀疏连接与 episodic memory 相关容量，不再依赖散落的结构魔法数字。
- N0–N11、M5–M7 以及阶段性自我评估/睡眠/探索判据已有通过记录；当前诚实边界仍是 raw-byte 自由生成尚未达到稳定人工可读。
- S1 产品断点已完成：知识库、生命状态、设置、聊天 composer、工作区重命名均有真实行为或明确门控。
- S1 验收：后端 282 passed/5 skipped，前端 160 passed，Playwright 22/22，生产构建通过；详细证据见 [S1 验收报告](../../../reports/seed_s1_acceptance_20260825.md)。
- 本次归档校验：`plans/active` 仅保留四份权威文档，旧 active 路径无残留引用，所有归档内相对 Markdown 链接可解析；项目身份/架构/命名边界回归 17 passed。

## 本次归档

active 目录只保留四份当前有效文档：

1. `SEED_DEVELOPMENT_ROADMAP_2026_08.md`：唯一执行顺序。
2. `SEED_ARCHITECTURE.md`：Seed/Taiji/Legacy 所有权与产品边界。
3. `TAIJI_SUBSTRATE_ARCHITECTURE.md`：算法、状态、局部学习和机制合同。
4. `ARCHITECTURE_DIRECTION_2026_08.md`：命名和不可回退决策。

以下文件移动到 archive，仅保留追溯价值：旧 M7 失败口径、800K 公测路线、容量/Legacy 实施过程、M4 前端检查清单。

## 延期事项

- CI 固定 Black 24.12.0 的最终复验是 R0 退出门，不把本机 Black 26.5.1 的异常退出当成通过或失败。
- S2 工程质量尚未开始：Legacy extra CVE、覆盖率盲区、前端 warning 棘轮和带后端浏览器链路仍待执行。
- 真实 CUDA 吞吐、显存和稀疏算子收益尚未测量；没有 GPU 证据前不改 Taiji 学习方程，也不写自定义 CUDA kernel。
- 16M 检查点仍需冻结 holdout 的完整质量审计；不直接恢复 100M 长训。

## 恢复执行时的唯一入口

先在 CI 固定 Black 24.12.0 环境完成 R0 复验；通过后从统一路线的 S2 开始。归档中的历史“下一步”全部失效。
