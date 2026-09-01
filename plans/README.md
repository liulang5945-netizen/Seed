# Seed / Taiji 计划与架构入口

> 当前只有一份执行计划：[模型优先统一开发计划](active/roadmap/03_CURRENT_EXECUTION.md)。历史文档中的“下一步”一律失效。

## 当前收敛结论（2026-09-01）

- Taiji 当前是原生学习机制原型，基础认知能力尚未得到可信证明；已有 checkpoint、局部学习、结构成长、Skill/MCP 投影和客户端隔离成果全部保留为训练底座。
- 当前唯一主线是 `M0 CPU 五项基础能力真实性基线 → M1 foundation 训练 → M2 世界/行动/语言后训练 → M3 综合晋级 → M4 持续进化 → M5 Skill/MCP 与客户端身体 → M6 provider/产品体验 → M7 CI/发布 → M8 CUDA`。
- M0 完成可信测量后，无论模型分数高低都必须进入 M1；低分是训练目标，不是继续外围建设的理由。
- MiniMind 只作为小模型训练工程参考：采纳分阶段训练、分级数据、checkpoint/resume、独立评估和推理入口，不把 Transformer 复制成 Taiji 核心。
- E1～E7 的完成状态保持有效但不代表通用能力；E8 bounded replay 冻结到 M4；真实第三方 MCP 连接冻结到 M5；CUDA 保持 `hardware-blocked`。

## 当前权威文档

| 文档 | 唯一职责 |
|---|---|
| [TAIJI_CORE_REQUIREMENTS.md](active/TAIJI_CORE_REQUIREMENTS.md) | 长期使命、CR-1～CR-10 和不可归档需求 |
| [TAIJI_NATIVE_ARCHITECTURE_V1.md](active/TAIJI_NATIVE_ARCHITECTURE_V1.md) | Taiji 目标架构、核心对象、学习体系和硬编码治理 |
| [ARCHITECTURE_DIRECTION_2026_08.md](active/ARCHITECTURE_DIRECTION_2026_08.md) | 身份、成熟技术采纳、Transformer/Legacy 边界 |
| [SEED_ARCHITECTURE.md](active/SEED_ARCHITECTURE.md) | Seed runtime、Workbench、provider、客户端和副作用所有权 |
| [03_CURRENT_EXECUTION.md](active/roadmap/03_CURRENT_EXECUTION.md) | **唯一执行计划、优先级、日程、Gate 和当前下一步** |
| [IMPLEMENTATION_STATUS_2026_08.md](reference/IMPLEMENTATION_STATUS_2026_08.md) | 当前代码事实和能力声明边界，不决定顺序 |

`active/roadmap/01`、`02`、`04` 与 `active/SEED_DEVELOPMENT_ROADMAP_2026_08.md` 只保留历史路径兼容入口，不能形成第二份计划。纠偏前全文已完整归档到 [roadmap_convergence_20260901](archive/history/roadmap_convergence_20260901/README.md)。

## 目录职责

| 目录 | 内容边界 |
|---|---|
| `active/` | 当前架构合同、唯一执行计划和短兼容入口 |
| `reference/` | 当前代码事实、owner 和能力边界；不提供执行顺序 |
| `archive/` | 已完成、被替代或仅用于追溯的设计、计划和调试记录 |
| `manifests/` | 可执行 Gate、数据、runtime 和实验合同 |

## 维护规则

1. 优先级、日程、当前阶段和下一步只能修改 `active/roadmap/03_CURRENT_EXECUTION.md`。
2. 架构身份变化同步更新对应架构合同；代码事实只写入 `reference/IMPLEMENTATION_STATUS_2026_08.md`。
3. 每个开发 slice 都必须更新统一计划、运行相关 CI、检查 checkpoint/数据边界并提交。
4. 完成过程进入 archive；归档文档不重新获得执行权。
5. 不新增并行路线文件、阶段分片或第二份“当前下一步”。

## 归档入口

- [归档索引](archive/README.md)
- [2026-09-01 模型优先纠偏前路线快照](archive/history/roadmap_convergence_20260901/README.md)
- [W7-R5 已完成分片](archive/history/roadmap_shards/)
- [历史 Gate/CI 记录](archive/history/SEED_GATE_CI_HISTORY_2026_08.md)
