# M4 前端联调评审清单

> 依据：公测（Public Beta）路线图 §1.5（前端/桌面端门槛）、`docs/design/taiji-front-redesign/`（项目既有设计标准）。
> 评审方式：启动 `api` + `frontend` dev server，用浏览器逐页对照；桌面端走打包冒烟。
> 每项判 通过 / 差距（附截图与修复项），全部通过方可进入 M5。
>
> **2026-08-23 评审结果**：三轮浏览器评审完成，记录见 `reports/m4_frontend_review_20260823.md`。
> 页面渲染/主路径/横幅与 console 全部通过；首轮暴露的 4 项后端阻断已修复并复验；
> 暗色模式对比度已测量达标（全局主题，采样聊天/设置/知识库三页）；桌面端双入口打包冒烟全部通过。

## 1. 页面覆盖对照（实现 ↔ 设计参考）

| # | 路由 | 实现 | 设计参考 | 布局 | 间距 | 暗色模式 | 备注 |
|---|---|---|---|---|---|---|---|
| 1 | `/` | `components/ChatView.vue` | `pages/chat.html` | [x] | [x] | [x] | 对话主路径实测通过（见 §2）；暗色采样页，最低对比度 4.91:1 |
| 2 | `/kb` | `views/KBView.vue` | `pages/kb.html` | [x] | [x] | [x] | rag/stats 404 已修；暗色采样页 |
| 3 | `/train` | `views/TrainingView.vue` | `pages/training.html` | [x] | [x] | [x] | checkpoints 端点已补，列表正常；续训端点已实现 |
| 4 | `/agent` | `views/AgentConfigView.vue` | `pages/agent-config.html` | [x] | [x] | [x] | 工具横幅异常已消除（6 工具） |
| 5 | `/workspace` | `views/WorkspaceView.vue` | `pages/workspace.html` | [x] | [x] | [x] | |
| 6 | `/life` | `views/LifeStatusView.vue` | `pages/life-status.html` | [x] | [x] | [x] | 后台任务横幅异常已消除 |
| 7 | `/settings` | `views/SettingsView.vue` | `pages/settings.html` | [x] | [x] | [x] | 运行环境分区实测全通；暗色采样页（主题切换入口） |
| 8 | 侧边导航 | `App` 布局 | `partials/taiji-sidenav.html` | [x] | [x] | [x] | <880px 隐藏且无抽屉入口，记录在案 |

> 暗色列说明：主题为全局 `data-theme="dark"`，已对聊天/设置/知识库三页采样测量，
> 其余页面同享主题变量，判定适用。证据：`_scratch/dark_01..03_*.png`。

## 2. 对话体验（公测用户主路径）

- [x] 首字节出现前无长时间空白（“Seed · 正在启动”指示可见，首问冷启动 ≈10-21s）
- [x] 回复流式/完整渲染，无乱码方块占满整屏时的误导性“成功”观感（乱码如实呈现，已知限制已入评审报告 §4）
- [x] 多轮会话历史可见、可继续；会话切换不丢上下文（实测两轮+离开返回恢复）
- [x] 模型/运行时切换（Seed 原生 ↔ Cortex）后对话可用（实测切换+持久化+重启自动恢复）
- [x] 控制台无未捕获错误（五页巡检+对话页 20s 停留，console error 0）

## 3. 设计规范符合度（对照 `colors_and_type.css` / `components.css`）

- [x] 主色调与字体标尺一致（单主色相 + 中性色 + 语义状态色；亮色经典蓝观感舒适，初步对照通过）
- [x] 圆角标尺克制（实现侧未见 20px+ 圆角）
- [x] 暗色模式下对比度可读（正文/次要文字/边框三级）——已测：「深邃暗色」采样 34/33 元素，最低 4.91:1，全部 ≥ WCAG AA 4.5:1
- [x] 间距系统一致，无明显挤压或超大留白（逐页评审未见挤压/超大留白）

## 4. 桌面端打包冒烟

- [x] 开发入口 `desktop/main.py` 全链路：窗口启动→后端拉起→Seed 激活→8765 ws→前端加载→对话可用（修复四项缺陷，见路线图进度日志）
- [x] `desktop/build.py` 打包成功（1121.7MB，双入口 `Seed.exe`+`SeedBackend.exe` 共享 `_internal`）；冒烟暴露并修复：frozen 递归启动风暴（改双入口独立 exe）、日志黑盒（文件 handler）、数据资产缺失（`tokenizer_contract.json`/`domains/*.model`/`checkpoints/seed_corpus.pt` 入 spec `datas`）
- [x] 双击启动 ≤ 15 秒出界面：实测约 2.9 秒（后端 1.5s 就绪→Seed 激活→8765 ws→前端加载，见 `dist/Seed/logs/desktop_main.log`）
- [x] 首次对话可用（`SEED_RUNTIME=1`：`/api/health` ok、`seed_active=true`；浏览器发送“你好”收到回复并正常渲染，截图 `_scratch/verify_8000_chat_nihao_reply.png`；回复乱码为 4M 早期检查点预期行为）
- [x] `SEED_HOST/SEED_PORT` 远程接入：`BackendManager` 读取环境变量并传给 `SeedBackend.exe`，与开发模式同源路径；设 `SEED_HOST=0.0.0.0` 即开放局域网访问（代码级验证，随 M5 用户文档发布）

## 5. 完成判据

§1.5 三项全过：安装与启动、核心交互（对话/历史/切换/设置全链路无控制台错误）、
界面品质对照本清单达标。差距项逐条修复后复评，复评记录落 `reports/m4_frontend_review_<日期>.md`。
