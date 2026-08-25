# Seed S1 产品断点验收报告

日期：2026-08-25

基线：`main` / `470f2af Refactor project structure and remove obsolete code`

范围：知识库、生命状态、设置、聊天 composer、工作区重命名，以及前端路由和桌面发布前的基础构建

## 结果

S1 功能验收通过。现有实现已接通真实后端能力，未发现需要扩大到 Taiji 学习方程或 Transformer 边界的产品问题。

| 检查 | 结果 |
|---|---:|
| 后端全量 pytest | 282 passed, 5 skipped |
| S1 后端路由回归 | 54 passed |
| 前端全量 Vitest | 19 files / 160 tests passed |
| S1 前端视图回归 | 51 tests passed |
| ESLint | 0 errors, 17 existing warnings |
| Ruff | passed |
| Mypy `seed taiji` | passed |
| 前端 production build | passed |
| Playwright smoke | 22 assertions, 0 failures |

## 已验收的产品闭环

- 知识库：上传队列、文件大小/更新时间/索引状态、预览、清空确认和状态刷新。
- 生命状态：使用真实运行时指标，feed/sleep/play/evolve 通过原生生命端点，支持 JSON 状态快照导出。
- 设置：设置读取、持久化、失败回滚、数据导出和带确认的会话重置。
- 聊天：快捷、代码、总结、翻译和附件入口有真实行为；未就绪能力不保留假按钮。
- 工作区：安全路径校验、重命名冲突、源不存在处理和已打开编辑标签同步。
- 浏览器：聊天、训练、Agent、工作区、生命状态、设置和知识库路由均可加载，移动端 `#app` 可渲染。

## 验收修正

Playwright 原断言要求“输入文本后立即允许发送”，但 `ChatView` 的正式契约是“输入非空、运行时已连接、模型已加载”三个条件同时满足才允许发送。已将 smoke 改为验证运行时门控：运行时未就绪时按钮保持禁用，运行时就绪时按钮必须可用。没有放宽产品安全条件。

## 遗留门项

- Black 首次检查只发现 `tests/test_workspace_routes.py`，该文件已按格式规范整理。当前开发机的 Black 26.5.1 编译运行器在检查完成后长时间不退出，无法把本地进程退出状态作为可靠证据；CI 使用固定的 Black 24.12.0，R0 关闭前必须在固定版本上复验 `black --check .`。
- ESLint 的 17 条 warning 和 production build 的 chunk size/plugin timing warning 不阻断 S1，但进入 S2 门禁收紧清单。
- 本报告不宣称 API-backed 浏览器数据操作已经完成真实后端 E2E；这些行为目前由前端组件测试和后端 TestClient 路由测试覆盖，S2 再补带后端的浏览器链路。
