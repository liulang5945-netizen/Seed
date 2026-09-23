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
- **G1 结论：通过。** fork 源码 12697 文件（纯源码树，node_modules/lib/tsbuildinfo 零命中，内嵌 .gitignore 生效）落 `taiji-harness/`。提交 82042a2f。
- **检查点（2026-09-22）**：所有者选择「先停，亲自验原版 UI 再决定是否进 G2」。当前状态=G1 已过、G2 未启动、旧线（frontend/desktop-electron）未删。恢复指令：所有者跑 `npx @deepseek-ai/dsh web` 验原版，确认后由所有者点头再进 G2（rename+品牌化）。
- **顺序裁决（2026-09-22，G2 启动令）**：所有者裁定「**网页端做好再做一版客户端版**」——G2/G3/G4 全部以 web 形态推进，桌面客户端（fork `apps/desktop`，Electron+内置运行时）挪到 G5 末位；届时需修 dev/package 流程的运行时下载链（nodejs.org/GitHub releases/pythonhosted 三源，本机 GitHub 通道不稳定 → 打国内镜像补丁）。验证过程中 fork 侧 Electron 二进制已就位（npmmirror 镜像 + 工作区缓存 `E:\Seed\.electron-cache`，后者为未跟踪目录）。**G2 即刻执行**。
- **G2 完成（2026-09-22）**：scope rename @deepseek-ai/*→@taiji/*（两遍机械替换 5206+43 文件；外部 npm 包 @deepseek-ai/libreoffice-kit 为唯一合法保留，相关 155 处回修；.agents/notes 冻结档案误改已还原 HEAD）；品牌化=client locale 字典（zh/en）+web 壳 title/manifest+vite 注入+desktop 菜单/About，冒烟首页 title 实测「Taiji Harness Local Build」；WELCOME_NOTICE_VERSION→2026-09-22.1 保证新声明重弹；build+test:gui 绿。遗留：electron-builder productName（打包身份）归 G5。
- **G2 后真机走查与修复（2026-09-23，G2b–G2h）**：四项走查闭三项（logo ✓／插件入口非缺陷／归档可只读查看＋可回档 ✓），另修两个真缺陷（归档只读位被模型选择器覆盖→`conversation.blocks` 改按 owner 键控多源注册表；live 会话不可删→Host 增「关闭后删除」）＋`replaceMain` 顺序竞态、归档行间歇打不开（F）＋slogan 两轮改定「态之极境」＋太极动势反向＋摘掉「预览版」标签。逐条读数见 [03_CURRENT_EXECUTION.md §5.7](../active/roadmap/03_CURRENT_EXECUTION.md)。
- **G3 完成（2026-09-23）**：`packages/llm/llm-taiji`（`@taiji/dsh-llm-taiji`）硬接 8000 runtime 落地——请求体按 Taiji 侧真实契约、SSE 取 `final` 帧、失败全走 `LlmError` 归一化（无 usage／无重放／无质量加工），就绪判定沿用 `GET /api/health` + `AdapterRegistrationHandle.replace` 撤/恢复路由。真机读数：`/api/chat/stream` 帧与适配器假设逐字一致；UI 选中「Taiji（本地运行时）」的真回合由 Taiji 回答（标题与正文均为运行时原生语言表层输出），Trajectory 记录完整。中途「Taiji 未进模型目录」经 `ctx.llm.listProviders()` 探针**翻案为探法误判**（`ModelSelect` 为多层面板，需点进「模型」才渲染分组；注册与目录链始终正常）。**G3 门通过 ⇒ 旧线删除（裁决 6）解锁。** 遗留：G（工作区状态死路：`initialized`/`defaultWorkspaceId` 与默认工作区记录不收敛；**已修复，见下条**）、首屏「内测声明」弹窗仍有 DeepSeek 口径文案（待所有者裁定）。
- **G 修复（2026-09-23，所有者批 (a)+(b)）**：工作区状态死路根因＝上游把「删除默认工作区」的禁用意图编码成悬空 `defaultWorkspaceId`（`initializeDefault` 见其存在即早退），配合 `initialized:true` + 空注册表，产品会以「无工作区可选且无法自愈」开局。修法：(a) 删除该工作区时在同一次写入里清掉默认身份（不变量：身份要么缺席、要么指向存在的行）；(b) 启动时对遗留悬空身份记 warn 并清除（不重置 `initialized`，避免 bootstrap 复活已删工作区），使客户端下次启动即可重建默认工作区。改写了上游一条冲突契约用例（原断言「永久禁用自动创建」），新增坏态自愈用例，两条修复前红、修复后绿。验证：typecheck 0 / lint 0 / 定向 vitest 117 文件通过（失败全为环境项）。提交 `4b8ab255`（另 `ffb6de57` 清掉 G2d/G3 留下的 25 条 lint 债）。逐条与未闭合项（i18n pairing 门全仓红、G2 锚点残留、无工作区但有会话的窄缺口）见 [03_CURRENT_EXECUTION.md §5.7](../active/roadmap/03_CURRENT_EXECUTION.md)。
- **旧线删除已执行（2026-09-23，裁决 6 兑现，档位＝所有者选定的「整线收敛」）**：删 `frontend/`、`desktop-electron/`、`desktop/`、`scripts/release.py`＋`sync_version.py`＋`make_social_preview.py`、两个读 `frontend/src` 的 E5 门脚本与 `tests/test_desktop_orphan_reap.py`（跟踪面 179 文件 / 48575 行），并同步 15 处引用面（结构守卫与 S2 台账、CI 三处、dependabot、Dockerfile、`.dockerignore`、`pyproject`、clean_worktree、`.gitignore`、两份 docs、CONTRIBUTING、README 中英）。本仓自此只剩**运行时**（`api/` 起的 8000）＋训练/M5 资产＋`taiji-harness/`；产品界面与打包全部归 harness（G5）。验证：结构守卫/身份/边界/密钥四组 pytest 全绿、OpenAPI 快照与 W7 manifest 门全绿、`create_app()` 无异常。逐条见 [03_CURRENT_EXECUTION.md §5.7](../active/roadmap/03_CURRENT_EXECUTION.md)。
