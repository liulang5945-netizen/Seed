# W7-R5B L4 纯计算执行体架构评审

> 评审日期：2026-08-30。本文是 L4 的 active 架构边界，不是实现承诺。

## 1. 评审结论

结论：**当前不批准新增 L4 执行体，实现保持 No-Go；允许后续提交满足本文件的 candidate package。**

原因不是能力不足，而是现有 Workbench 能力都依赖外部世界、用户界面、终端或远程通道，尚没有一个同时满足以下三个条件的现成候选：

1. 输入只由显式 bytes/JSON 构成；
2. 执行完全确定、无外部副作用；
3. 可以由独立 oracle 在不复用候选实现的情况下完整验证。

L4 不能把普通的文件读取、语言识别、IDE 操作或 MCP 调用包装成“纯计算”，也不能因为 executor 已经有版本号就自动获得 L4 资格。

## 2. 当前 capability inventory

| 能力集合 | 当前性质 | L4 结论 |
|---|---|---|
| `workspace.list/read/stat/search` | 读取可变文件系统，结果依赖 workspace after-state | 不准入 |
| `workspace.programming_language.resolve` | 依赖扩展名、shebang、manifest、邻近文件、toolchain/LSP | 不准入 |
| `editor.open/set_language` | 修改或读取 IDE/UI 状态 | 不准入 |
| `workspace.apply_patch/create/rename/delete/undo` | 文件系统副作用，需要 disposer 和 approval | 不准入 |
| `terminal.run` | 外部进程、环境和副作用 | 不准入 |
| `mcp.list/invoke` | 外部连接和远程状态 | 不准入 |

因此，L4 评审没有把任何现有 Workbench capability 偷换成纯计算能力；当前能力仍由 R5B-L0/L1/L2/L3 合同治理。

## 3. 未来 candidate 的强制准入条件

候选必须先作为 R5B-L1 package 提交，并同时满足：

- 输入只允许显式 JSON、bytes 或有限标量；禁止路径、文件句柄、环境变量、当前时间、随机数、网络、进程和隐式全局状态；
- 对同一 canonical input 必须产生同一 output 或同一结构化 error；
- candidate artifact 只能保存逻辑 executor identity，不能保存源码、import path、shell command 或自动激活字段；
- 至少一个独立 oracle：参考实现、数学不变量、标准算法向量或第三方验证器；oracle 不能只是复制 candidate executor；
- 必须有 exact-output、property/metamorphic、边界输入和错误输入测试；
- shadow 阶段必须 output 等价、after-state 不变，并记录 resource delta；
- 仍需经过 policy、resource reservation、checkpoint 和可逆 lifecycle；“纯计算”不等于绕过 registry 或审批；
- 只有跨 seed/task slice 稳定且 rollback/retention 通过，才可进入 active。

## 4. 禁止的伪 L4 路径

- 把 `workspace.read` 的缓存结果当成纯函数；
- 把 provider、Transformer、LLM 或 prompt 作为独立 oracle；
- 以“只读”名义隐藏 MCP、终端、模型调用或文件访问；
- 让 Taiji、provider 或 frontend 直接注册、激活、替换 executor；
- 用一次 shadow 通过替代长期 holdout、lesion、rollback 和资源证据；
- 因为本机没有 CUDA 就把 CPU 纯计算结果宣称为 CUDA 执行体能力。

## 5. 评审输出与下一入口

- L4 评审输出：`architecture_review_required`，当前无候选执行体获准实现；
- R5B 已完成 L0/S1、L1、L2、L3 的 registry、candidate、shadow、resource 和 rollback 基础；
- 完整 Workbench CI 仍按用户决定暂缓，不能被本评审标记为通过；
- 下一入口转入 R5C：把长期真实 evidence 接入已有 structural growth proposal/holdout/lesion/rollback substrate，仍然不按 action/intent 名称硬编码结构角色。
