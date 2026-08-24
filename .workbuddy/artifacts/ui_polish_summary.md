# 前端界面与交互优化概览

时间：2026-08-24
目标：对齐主流桌面应用（Claude / Codex / Trae）的界面成熟度与一致性。

## 已完成的 5 项 + 1 个顺带修复

### 1. 移除侧边栏滑动效果
- 文件：`frontend/src/assets/app.css`
- 改动：`.sidebar` 的 `transition: width 0.3s …` 移除 `width` 过渡，仅保留背景过渡。
- 效果：拖拽改宽即时跟随鼠标，消除“滑动拖尾”的迟滞感；移动端折叠也即时生效。

### 2. 取消客户端启动载入动画
- 文件：`desktop/main.py`
- 改动：删除 `_show_loading`（loading.html 旋转/呼吸/进度条动画）流程，窗口创建后直接加载前端。
- 依据：`backend.start()` 与 `ws_server.start()` 在窗口创建前已阻塞至就绪，动画本就多余。
- 顺带修复：原有两处重复的 `QTimer.singleShot(300, self._show_loading)`（重复调用 bug）已合并为一处直接加载。
- 预渲染背景设为亮色（`#f6f7f9`，匹配默认亮色主题），避免白屏闪烁。
- 注：`desktop/loading.html` 现已成为未引用文件（未删除，保留以备需要）。

### 3. 系统托盘图标改为 logo 样式
- 文件：`desktop/main.py`、`api/run_app.py`
- 说明：用户澄清原意是「改为 **logo** 样式」（原描述中「非」为误打）。
- 改动：托盘图标统一使用应用 logo（与窗口图标一致）。`main.py` 用 `frontend/public/favicon.ico`（缺则回退 `icon.ico`）；`run_app.py` 沿用打包 `app_icon`（icon.ico）。窗口与托盘图标现在同为品牌 logo，视觉统一。

### 4. 底部状态栏补充图标
- 文件：`frontend/src/views/WorkspaceView.vue`
- 改动：状态栏各状态项补充对应 lucide 图标（行列→Crosshair、UTF-8→FileText、语言→FileCode、未保存→圆点、神经元同步→Activity），新增 `.status-item` 对齐样式，统一 12px 图标，对齐主流 IDE 状态栏观感。

### 5. 去掉系统原生标题栏（无边框窗口）
- 文件：`desktop/main.py`、`api/run_app.py`
- 改动：
  - 窗口设为 `FramelessWindowHint`（保留 Window 类型，不影响任务栏/最小化）。
  - 新增自绘极简标题栏（36px）：标题 + 最小化 / 最大化·还原 / 关闭；可拖拽移动（startSystemMove）、双击切换最大化。
  - 标题栏配色**跟随前端主题**：默认亮色，前端就绪后用 JS 读取 `data-theme` 同步（暗色→深色）。
  - 新增 `_EdgeResizeFilter` 应用级事件过滤器：光标进入窗口边缘 6px 热区并按下左键时交给系统缩放（startSystemResize），保留边缘调整大小能力。

## 验证
- `desktop/main.py`、`api/run_app.py` 均通过 `py_compile`。
- 前端 `vite build` 成功（`.build-verify` 临时目录验证后已删除）。注：直接构建到 `dist/` 会被沙箱拦截清空操作，属环境限制，非代码问题。

## 已知限制 / 后续可选项
- **Windows 贴边分屏（Aero Snap）与原生阴影**：无边框窗口默认失去。如需恢复，需引入 DWM/WS_THICKFRAME 处理（较复杂），可后续单独做。
- **标题栏主题同步时机**：在前端加载完成后一次性读取；运行中切换主题需刷新页面后标题栏才会跟随（如需要可再加 QWebChannel 实时桥接）。
- 各页面顶部的页头栏（`.topbar`，如「IDE 工作区」标题行）按你的选择**保留**（仅无边框化窗口，未删页头）。

## 涉及文件
- `frontend/src/assets/app.css`
- `frontend/src/views/WorkspaceView.vue`
- `desktop/main.py`
- `api/run_app.py`
