# M6 前端 DSH 式工作区 Dock 改造方案（冻结）

- 日期：2026-09-21
- 状态：**方案冻结（三项关键决策已由所有者裁定）**，实施进行中
- 触发：所有者提出「IDE 等可以整合进对话页面，做成主流的侧面展开，依照 DeepSeek Harness 的内核和 UI 设计对前端和 UI 进行调整」

## 1. 所有者裁定（2026-09-21，AskUserQuestion 三问三答）

| 决策点 | 裁决 | 含义 |
|---|---|---|
| Dock 挂载层级 | **全局 Dock（App 壳层）** | 工作区面板挂在 `App.vue` 的 `.app-body`，训练/知识库/生命状态等任何页面都能侧拉出 IDE（上限更高方案） |
| 原 `/workspace` 独立页 | **移除，面板可全宽** | 单一入口；dock 支持 收起/分栏/全宽 三档 + 拖拽调宽，全宽时体验等同原独立页 |
| dsh 改造深度 | **布局 + 会话流 + 视觉** | ① IDE 侧栏整合；② 对话流 Trajectory 化（append-only 时间线）；③ dsh 式视觉刷新，**保留现有 5 套主题 CSS 变量体系** |

## 2. 参照对象：DeepSeek Harness（dsh）的手感来源

决策简报（FRONTEND_TS_VS_HARNESS_ADOPTION_DECISION_BRIEF_20260921.md §2）已考察：dsh 的使用手感来自 harness 设计——**append-only session log + Trajectory 视图**（工具调用/执行步骤以时间线流入对话）、一切皆插件、四档 preset。本改造取其 UI 手感与组织思想，不引入其运行时依赖（Cordis 内核不采纳，理由见决策简报结论 A：语言/内核与效果正交，借鉴架构思想即可）。

## 3. 现状（2026-09-21 侦察）

- `App.vue`：`.app-body`（flex 行）= `AppSidebar` + `.router-wrapper`（全应用唯一外围边框+圆角，真源在 `styles/shell.css`）
- 路由：`/`（ChatView）、`/kb`、`/train`、`/agent`、`/workspace`（WorkspaceView，1032 行完整 IDE）、`/life`、`/settings`；keep-alive include 含 `WorkspaceView`
- WorkspaceView：文件树（WorkspaceFileTree）+ 编辑器（WorkspaceEditorPane → MonacoEditor，openTabs/activeTab 状态在组件内）+ 终端（WebTerminal）+ 属性面板 + quickOpen（Ctrl+P）+ 右键菜单 + 双对话框；数据流经 `useWorkbenchProjection`（listDirectory/readFile/setWorkspaceRoot/executeNativeMutation：workspace.create/rename/delete、terminal.run）
- ChatView：topbar + ChatMessageList（已含 `workbench-trace` 行，msg.workbenchEvents）+ 悬浮 WorkbenchTaskCard（解释→计划→审批→执行）
- AppSidebar：navGroups「工作台」组含 `/workspace` RouterLink

## 4. 目标架构（冻结）

```
.app-body
├── AppSidebar（IDE 项改为 dock 开关 button；Ctrl+P/Ctrl+` 全局快捷键移入 dock）
├── .router-wrapper（flex:1；dock 分栏时收缩，全宽时让位）
└── WorkspaceDock（全局单实例常驻，v-show 控制显隐 → Monaco 状态天然保持）
    ├── dock 头部：路径 + 状态点 + 打开文件夹/搜索/终端/运行/保存（原 WorkspaceView 顶栏紧凑化）
    ├── 三档：collapsed（width 0 不渲染内容区）/ split（480–960px 拖拽）/ full（独占，router-wrapper 让位）
    └── 主体：WorkspaceFileTree | WorkspaceEditorPane+WebTerminal | 属性面板（原三栏结构整体迁入）
```

- **store**：新增 `stores/workspaceStore.js`（pinia）：`dockMode: 'collapsed'|'split'|'full'`、`dockWidth`（localStorage 持久化 `taiji_dock_mode/width`）；IDE 内部状态（fileTree 等）留在 WorkspaceDock 组件内（单实例无需提升）
- **移除**：`/workspace` 路由、keep-alive include 的 `'WorkspaceView'`、`WorkspaceView.vue` 文件、侧栏旧入口
- **对话流 Trajectory 化**：`WorkbenchTaskCard.vue` 重构为 `WorkbenchTimeline.vue`——从悬浮 aside 卡片改为 chat-thread 流内的 append-only 垂直时间线（目标证据 → 语义步骤 → 计划 → 审批 → 执行 → 结果，随阶段推进逐节追加，左侧连线 + 状态点）；审批/执行交互保留；`ChatMessageList` 的 `workbench-trace` 行样式统一到同一时间线语言
- **视觉刷新（dsh 式，全部走现有 CSS 变量）**：消息气泡扁平化（用户浅底圆角块、助手全宽 markdown 无气泡）；nav/工具卡紧凑化；路径/代码等宽字体；状态点语言统一

## 5. 风险与边界（冻结）

1. **Monaco 在 display:none 下初始化会 layout 异常**：dock 首次展开后再让编辑器 pane 挂载（v-if 由 dockMode 驱动，首次展开即挂载且此后常驻），展开时 nextTick 触发一次 layout
2. **Ctrl+P / Ctrl+` 生命周期**：从 WorkspaceView 的 onMounted/onActivated 移到 WorkspaceDock 常驻监听；dock 隐藏时 Ctrl+P 先展开 dock 再弹 quickOpen
3. **keep-alive**：dock 在壳层不受路由 keep-alive 管理，原 onActivated 懒加载语义改为「首次展开时 loadTree」
4. 5 套主题（classic/dark/teal/violet/warm）+ light/dark 全部经由既有 CSS 变量生效，本改造不新增硬编码色值（占位灰度可例外）
5. Electron 壳零改动（本改造纯前端；`frontend/dist` 重建后打包版自然生效）

## 6. 验证门（冻结）

1. `npm run build` 过（frontend 目录）
2. dev 起后目测：三档切换与拖拽调宽、Ctrl+P/Ctrl+`、文件打开/编辑/保存/运行、对话发送与轨迹时间线、5 套主题抽查 2 套、窄窗（<880px）dock 行为
3. `WorkspaceView` 全库 grep 残留清零（路由/侧栏/keep-alive/文件本体）
4. 红绿纪律：先起 dev 目测改造前基线，改造后复测同一清单
