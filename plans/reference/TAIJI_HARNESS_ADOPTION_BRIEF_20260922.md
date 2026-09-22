# Taiji Harness 立项决策简报（dsh 整仓 fork）

- 日期：2026-09-22
- 状态：**九项裁决全部闭合（所有者逐项选择），进入执行**
- 触发：所有者对 Vue 前端两轮改造（dsh 式 Dock → Artifacts 工件面板）后仍不满意，裁定「直接把 DeepSeek Harness 改造成 Seed，也就是 Taiji Harness」
- 前史：本简报**推翻** [FRONTEND_TS_VS_HARNESS_ADOPTION_DECISION_BRIEF_20260921.md](FRONTEND_TS_VS_HARNESS_ADOPTION_DECISION_BRIEF_20260921.md) 的结论 A 处置（当时选「只借鉴思想、自研 UI」）；语言≠效果的论证仍成立，但所有者要的是 harness 本体而非仿制品。同时**取代** [M6_FRONTEND_DSH_WORKSPACE_DOCK_PLAN_20260921.md](M6_FRONTEND_DSH_WORKSPACE_DOCK_PLAN_20260921.md)（v1/v2 均作废）。

## 1. 九项裁决（2026-09-22 逐项问答）

| # | 决策点 | 裁决 |
|---|---|---|
| 1 | 采用策略 | **整仓 fork 改名，自主迭代**（接受 dsh v0.1 破坏性变更由自己消化） |
| 2 | 前端形态 | **dsh 自带前端为主**（「已经很不错」），做 Taiji 风格调整；缺的是生命系统，训练与知识并入生命系统 |
| 3 | Taiji 位置 | **Taiji runtime = 模型**（硬接，语言器官即 provider） |
| 4 | 硬接策略 | **纯硬接，失败即读数**——无回退无兜底，harness 成为 M5 能力的实时考场，每个失效模式进 Trajectory 可回放 |
| 5 | fork 位置 | **本仓子目录 `taiji-harness/`**（与 taiji/、scripts/ 同层，单仓开发） |
| 6 | 旧线处置 | **fork 跑通后立即删除** `frontend/` 与 `desktop-electron/`（git 历史可找回；Electron 壳职责由 harness 的 host/client 包承担） |
| 7 | 生命系统 | **侧栏面板 + 上下文注入**双层：面板可见（状态/训练控制/知识管理），生命状态注入每轮对话（模型感知自身状态） |
| 8 | 包命名 | **`@taiji/*`** scope（全仓机械 rename，如 @taiji/dsh、@taiji/dsh-session） |
| 9 | 视觉标识 | **沿用现有水墨太极图标**（icon.ico / seed-taiji-network.png） |

## 2. 目标架构

```
taiji-harness/                      ← dsh fork（MIT 合规：保留上游 LICENSE 与出处声明）
├── packages/ @taiji/dsh-*          ← 50+ 包 rename（core/agent/session/llm/host/client/...）
├── vendor/cordis                   ←  vendored Cordis 内核（随 fork 保留）
└── packages/taiji/                 ← 新增 Taiji 插件组（本项目的全部增量）
    ├── dsh-llm-taiji/              ← provider 插件：dsh 模型接口 → Taiji runtime HTTP(8000)
    │                                  纯硬接：Taiji 输出即回合，工具调用发不出＝失败进 Trajectory
    ├── dsh-life/                   ← 生命系统插件：侧栏面板（需求/惊奇/可塑性/训练进度）
    │                                  + 上下文注入（生命状态→每轮 system context）
    │                                  + 训练控制与知识库管理（并入原 Vue 页职责）
    └── dsh-boot-taiji/             ← boot 插件：拉起/守护 SeedBackend.exe（现 desktop-electron
                                       backend.ts 的 Python 子进程管理职责迁入）
```

- **模型接口适配**：dsh provider 契约（chat completion + tool calls 结构化）→ Taiji runtime 现只有纯文本语言器官；适配层**不做美化**——把 dsh 的回合请求（含工具 schema）序列化进 Taiji 输入流，把 Taiji 的纯文本输出原样交回解析器，解析失败即按失败记录。这是刻意的：失效模式即 M5 训练的需求清单。
- **开发期模型**：日常开发 harness 本身时用 deepseek-official（凭证已在 ~/.dsh）；**验收读数一律 Taiji provider**。两条路径不混淆：前者是工程工具，后者是产品定义。
- **桌面交付**：harness `dsh web`（127.0.0.1:3080）+ host/client 包；旧 Electron 壳删除后如需桌面窗口，用 harness 自带机制或最小新壳，不复活旧壳。

## 3. 里程碑（验收门，顺序执行）

| 门 | 内容 | 通过判据 |
|---|---|---|
| **G1 fork 健康** | clone dsh → `taiji-harness/` → 上游原样 install/build/test → `dsh web` 起 UI、deepseek-official 对话可用 | build 绿 + web 可用 + 记录上游基线测试通过数 |
| **G2 rename** | `@deepseek-ai/*` → `@taiji/*` + 品牌化（标题/图标/文案 Taiji Harness） | typecheck/build/test 与 G1 基线一致；UI 无 deepseek 品牌残留（保留 LICENSE 出处） |
| **G3 Taiji provider** | `dsh-llm-taiji` 插件：硬接 8000 runtime；纯硬接语义落地 | 对话回合由 Taiji 回答（哪怕质量差）；Trajectory 完整记录每回合与失效模式；**旧线删除提交在此门后** |
| **G4 生命系统** | `dsh-life` 面板 + 上下文注入 + 训练/知识并入 | 面板实时反映 runtime 状态；生命状态出现在请求上下文；训练可在面板启停 |
| **G5 交付** | 打包（harness 自带机制）+ 桌面可用 | 一条命令装起即用，Taiji provider 为默认 |

- 每门独立提交、可回退；G1 基线测试通过数是 G2 的对照尺（rename 不得引入回归）。
- 旧线删除（裁决 6）固定在 **G3 通过后**执行。

## 4. 风险与边界

1. **dsh v0.1 API 不稳**：fork 自主迭代＝自己消化上游变更；策略=锁定 G1 的 commit 为基线，上游更新按需 cherry-pick，不做定期 merge（避免破坏性变更风暴）。
2. **纯硬接的产品可用性为零期**：G3 后、Taiji 语言能力上来前，产品对话质量受 M5 现状约束（成句率 0.60、无工具调用格式能力）——这是已接受的代价，失效读数反哺 M5。
3. **仓体量**：50+ TS 包进主仓（node_modules 除外）；`.gitignore` 与目录守卫（tests/test_folder_structure_guard.py）需同步认 `taiji-harness/`。
4. **Node/工具链**：dsh 要求 node ^22.19 || >=24（本机 v24.15 ✓）、pnpm workspaces（需装 pnpm）、部分 gate 需 git/gh；Windows 专属 gate（wine）跳过并记录。
5. **MIT 合规**：保留上游 LICENSE 文件与出处声明（README 注明 fork 自 deepseek-ai/deepseek-harness@<commit>），仅改品牌与代码。
6. **R2/M5 训练线不受影响**：判决已出（第十八批），三臂资产封存；Taiji 能力增长（M1 缺口）与 harness 互为考场/考生。

## 5. 执行记录

- G1 启动：2026-09-22（本简报落盘后）
- **G1-1 源码落地 ✓**：git clone 两次被网络掐（Connection reset）→ 改走 codeload tarball（master @ 2026-09-22，27.1 MB）解包到 `taiji-harness/`；15 个符号链接（CLAUDE.md→AGENTS.md 等）在 Windows 沙箱无法创建 → 全部物化为副本；上游版本 `@deepseek-ai/dsh-root 0.1.7-alpha.1`，pnpm 钉 11.7.0（corepack 缓存被沙箱拦 → `npx pnpm@11.7.0` 走 npm 缓存成功）
- **G1-2 install ✓**：`Done in 2m26.1s`（bin 警告＝lib 未构建，预期）
- G1-3 build ✓：`pnpm run build` 通过（tsc + tsdown，客户端 263 artifacts，末尾 `✓ built`）
- G1-4 web ✓：`pnpm dsh web --no-open` 起服（沙箱 HOME 重定向到 `E:\Seed\.dsh-sandbox-home` 绕开 `~/.dsh` 锁），3080 监听、token URL 浏览器打开 → **UI 完整渲染**（中文本地化「DSH 本地构建」：新建会话/工作区/插件/设置/模型选择器 DeepSeek-V41-Flash/composer `/`指令+`@`文件/访问模式/内测声明弹窗）。**fork 从源码跑通，G1 通过**。
- G1-5 test 基线：延后到 G2 前单独跑（vitest 全量耗时；rename 回归对照用 build+web 已足够，test 基线数在 G2 提交前补记）。
- **G1 结论：通过。** fork 源码 12697 文件（纯源码树，node_modules/lib/tsbuildinfo 零命中，内嵌 .gitignore 生效）落 `taiji-harness/`。
