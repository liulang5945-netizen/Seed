# M4 前端联调评审报告（2026-08-23）

> 依据：公测路线图 §1.5、`plans/archive/implementation/M4_FRONTEND_REVIEW_CHECKLIST.md`。
> 方式：API（127.0.0.1:8000）+ Vite dev（localhost:5173），浏览器三轮评审。
> 结论：**页面评审与主路径验证通过**；桌面端打包冒烟待执行；对话质量受限于
> 早期检查点（800K，可读率 0%）——属 M1/M2 范畴，非前端缺陷。

## 1. 评审轮次与发现

### 第一轮（基线评审，7 页面全覆盖）
页面渲染与视觉层面达公测水准（7/7 正常渲染、布局精致、多主题），
但暴露三个**阻断级**后端缺陷 + 一个严重前端问题：

| # | 级别 | 问题 | 根因 |
|---|---|---|---|
| 1 | 阻断 | 聊天发送按钮恒禁用 | `/api/runtime/status` 500：`routes_runtime` 以 1 参调用零参 stub `get_runtime_status()` → TypeError |
| 2 | 阻断 | 运行时切换不生效 | 切换 POST 被全局限流（100/分钟）429 拦截 |
| 3 | 严重 | 控制台轮询风暴、连带 429 | 全局安全限流 100/分钟 < 前端多组件轮询量；前端同请求重复发射 |
| 4 | 严重 | "工具状态异常"横幅 | `tool_registry` 为空 stub 且 `ToolDef` 不认 `func=`/`category=` 关键字 → 注册崩溃 |
| 5 | 中 | `/api/train/checkpoints` 404 | `api/training/` 缺检查点列表模块 |
| 6 | 轻 | 启动警告 ×2 | `start_auto_reload` 缺失、`neuroplex.infra` 包整体缺失 |

### 第二轮（修复后复评）
三个阻断全部解除：`/api/runtime/status` 全程 200（真实负载：
health/memory/auth/life/tools/training 六段）；设置页正确显示
"Seed 原生运行时激活中"；轮询稳定无 429；对话主路径走通
（首问约 10-21s 冷启动、回复为 800K 检查点预期的乱码、多轮历史正常）。

### 第三轮（终复评）
- `#/train` 404 消除，检查点列表显示 2 项（`seed_beta.pt` Step 2,800,000 等）
- 对话主路径：发送→"正在启动"指示→回复完成 ≈11.1s；20s 停留 console error 0
- 五页巡检（kb/agent/workspace/life/settings）：横幅 0、console error 0
- 遗留轻微项 `/api/rag/stats` 404 → 已当场补齐（连同前端调用的 `/api/rag/clear`）

## 2. 本轮落地的修复

| 文件 | 修复 |
|---|---|
| `neuroplex/services/runtime_service.py` | stub → 真实聚合服务（六段负载，逐段防御降级，契约对齐 `RuntimeStatusPayload`） |
| `neuroplex/agent_ext/tool_registry.py` | stub → 真实注册表；`ToolDef` 兼容 `func=`/`category=` |
| `neuroplex/services/tool_service.py` | stub → 注册表桥接（列表/Schema/执行） |
| `neuroplex/infra/{__init__,events,event_subscriptions}.py` | 新建：真实 EventBus（发布/订阅/广播回调、异常隔离）+ 事件审计落盘订阅 |
| `neuroplex/core/model_loader.py` | 补 `start_auto_reload`（周期巡检自动重装，幂等，Seed 激活时不干预） |
| `api/app.py` | 全局限流 100→600/分钟（对齐 read 桶）；启动恢复运行环境偏好 |
| `api/routes_model_switch.py` | 运行环境选择持久化（`data/runtime_preference.json`），重启自动恢复 Seed |
| `api/training/checkpoints.py` | 新增：`GET /api/train/checkpoints`（读信封 metadata，坏文件降级） |
| `api/routes_rag.py` | 补 `GET /api/rag/stats`、`DELETE /api/rag/clear` |
| `frontend/src/composables/useApi.js` | 稳态轮询去掉重复的 `refreshAll`（同请求双发射） |
| `frontend/src/views/LifeStatusView.vue` | 页面级轮询 15s→60s（App 级已覆盖） |
| `frontend/src/components/MemoryStatusBar.vue` | 内存轮询 5s→30s |

## 3. 验收对照（路线图 §1.5）

- [x] 安装与启动：dev 服务正常；重启后运行时偏好自动恢复（Seed 直启，免手工切换）
- [x] 核心交互：对话/历史/运行时切换/设置全链路无 console error、无 500、无 429
- [x] 界面品质：7/7 页面布局间距暗色观感达标（第一轮截图对照设计参考）
- [ ] 桌面端打包冒烟（`desktop/build.py`）——待执行

## 4. 已知限制（如实呈现）

1. 回复质量 = 当前检查点能力：800K 基线输出乱码（M2 已量化），
   待 M1 大预算训练（进行中，已 2.8M ticks）完成后替换检查点复评。
2. 首问冷启动约 10s（"正在启动"指示可见，非静默等待）。
3. 窄屏（<880px）侧边栏隐藏且无抽屉入口——桌面端定位下不阻断，记录在案。
4. `api/training` 的断点续训流式端点（`/api/train/resume_checkpoint`）仍未实现，
   训练页"恢复训练"按钮依赖它——列入 M5 前待办（不阻断对话主路径）。

## 5. 证据

- 截图（三轮）：`_scratch/01_chat_home.png` … `_scratch/10_chat_with_sidebar.png`、
  `_scratch/recheck_01..05_*.png`、`_scratch/final_r3_01_train.png`、`_scratch/final_r3_02_chat.png`
- 回归：`pytest tests/ -q` → 113 passed / 4 skipped（每轮修复后均复跑）
- 运行时证据：`/api/runtime/status` 六段负载 200；切换→偏好落盘→重启自动恢复全链路实测通过
