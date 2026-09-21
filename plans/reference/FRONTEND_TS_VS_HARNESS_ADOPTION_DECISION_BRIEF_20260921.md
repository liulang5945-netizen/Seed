# 前端 TS 化 vs 抄开源 Harness 决策简报

- 日期：2026-09-21
- 状态：**选项甲已由所有者批准并执行**（2026-09-21）；执行结果与红/绿取证见 §7，实施中有一处原设计被实测否决（见 §5.1）。
- 新增待裁定项：**Electron 迁移**（所有者本轮提出），评估见 §8。
- 触发：所有者观察到主流开源 agent 平台（DeepSeek Harness、ZCode）以 TypeScript 为主，提问是否应将本项目前端改用 TS，或转向采用这些开源项目做 taiji 化改造。
- 类型：决策简报（新增文件，不改动任何既有冻结文档）

---

## 1 前提核实（取证时刻 2026-09-21 10:30 GMT+8）

| 项目 | 事实 | 对本决策的含义 |
|---|---|---|
| ZCode（智谱 ADE） | 仓库 `zai-org/ZCode` 于 **2026-09-20 20:01** 开源，Apache-2.0，7364 文件，TypeScript 97.6%，**提交仅 2 条**（Initial commit / feat: open source），**无 release**；正处「工作区快照静默上传阿里云 OSS」安全事故整改期（9-19 v3.14.0 移除 Repo Wiki，9-21 公告整改完成并致歉） | **不可作为 fork 基座**：无历史、无发布物、安全叙事未闭环 |
| DeepSeek Harness（dsh） | MIT，TypeScript，构建于 Cordis 插件内核，**v0.1 开发者预览**，官方明言会有兼容性破坏变更；四档 preset：Standard / PTC（Code Mode SDK）/ Minimal（两工具裸 harness，用于在裸环境 benchmark 模型）/ Create | 可借鉴**架构思想**；作为运行时依赖需承担持续对齐成本 |
| 反例 | Aider、OpenHands、AutoGen 均为 Python；Claude Code 为闭源 TS | 「实现语言」与「agent 效果」无因果关系，样本两侧都有 |

**结论 A：语言不是效果变量。** dsh/ZCode 的使用手感来自 harness 设计——一切皆插件、append-only session log + Trajectory 视图、四档 preset、子智能体与上下文注入落账——这些与实现语言正交。

---

## 2 本项目前端现状（实测）

- `frontend/src`：**124 文件 / 21,117 行**。`.vue` 45（**全部 `<script>` 为 js，无 `lang="ts"`**）、`.js` 69、`.css` 10。
- **无 `tsconfig.json`**；仅有 `jsconfig.json`（只配 `@/*` 路径别名，`compilerOptions.paths`）。
- 依赖：Vue 3.5.32 · Vite 8.0.8 · Pinia 3 · vue-router 4 · naive-ui · Monaco · xterm。无 `typescript`、无 `vue-tsc`。
- 既有质量设施：49 个 vitest 用例、Playwright e2e smoke、`scripts/check-native-boundary.mjs`、`scripts/check-api-contract.mjs`。

**关键发现（决定本决策的性质）**：`src/composables/nativeApi.js` 已是**类型化边界门面的雏形**——冻结的 `nativeApiPaths` 路径表（runtime / auth / settings / chat / training / workbench / clientExtensions），且其文件头注释明确写道：

> "The endpoint paths remain explicit here so the OpenAPI contract checker has one client-side source to audit; request payloads stay plain objects and are serialized only at this boundary."

即：**契约单点化这件难事，本项目已经做完了；缺的只是类型层。前端是「TS 形状的 JS」，不是「需要重构成 TS 的 JS」。**

---

## 3 两条路的对照

### 选项甲：前端增量 TS 化（采纳建议）

真实收益，逐项定位：

1. `nativeApi.js` 的 payload 目前是 plain object → 后端字段一旦变更，前端静默出错；类型化后为编译期错误。
2. `stores/runtimeStore.js`(589) / `chatStore.js`(420) / `appStore.js`(233) 的状态形状 → 重构中最易静默损坏的位置，TS 收益最大。
3. `useWebSocket.js` + `/api/workbench/events` → SSE/WS 事件以 discriminated union 表达。
4. `check-native-boundary.mjs`（71 行，**正则扫描源码文本**）与 `check-api-contract.mjs`（227 行，文本检查）→ 其对象从「文本」升级为「类型」，部分检查由编译期自动承担。
5. 后端具备 OpenAPI 快照纪律（`api/app.py` 明言 schema 快照）→ 可 `openapi-typescript` 由后端 schema 生成前端类型，**Python 保持单一真源**。

诚实标注无法获得的部分：

- 桌面壳为 PyQt6（`desktop/`），**不是 Electron** ⇒ 拿不到「全栈同语言共享类型」的最大红利，只能走 OpenAPI schema 单点。
- 对 agent 能力/效果 **0 收益**。本选项的定位是**契约强度与可维护性**，不是效果。
- 21K LOC 全量迁移是长期工程，必须增量，不得一次性重写。

### 选项乙：采用 dsh/ZCode 做 taiji 化改造（否决建议）

| 否决理由 | 具体内容 |
|---|---|
| 层次错位 | dsh 是「运行编程 agent 的 harness」，本项目是「预测基底」。把 taiji 做成 dsh 的 model adapter 插件，等于把根做到别人的插件槽里，差异化被降格为适配器 |
| 架构倒置 | `taiji/` 的架构契约要求自足、对外零依赖、语义 encoder 的 embedder 显式注入（fail closed）。接入 dsh 需在 Node 与 Python 之间插入 IPC/HTTP 层——方向与「Python 为本体、前端为薄壳」相反 |
| 成本真实、收益间接 | dsh 处于 v0.1 预览并明言破坏性变更，每次升级需重新对齐插件 API；而它带来的价值主要是**思想**，思想无需其代码即可借用 |
| 可行性不足 | ZCode 无法 fork（2 条提交、无 release、安全事故整改期） |
| 与当前主线冲突 | R2 受控重训仍在执行、M5 才刚限定退出、`taiji/` 尚未接入产品后端；此时代替内核等于把主线资源挪去装修门面 |

---

## 4 应借用的思想（借架构，不借代码）

1. **append-only session log + trajectory**：本项目 `RuntimeEvidenceStrip` / `useWorkbenchProjection` 已是雏形，但缺「每一次上下文注入都落账」。
2. **Code Mode / PTC**：把 `nativeApiPaths.workbench` 的 12 个端点 + 3 个 natural-language 端点，从「模型多次 round-trip」收敛为「模型写一段程序」。
3. **Minimal preset（两工具裸 harness）**：dsh 专为「在裸环境 benchmark 模型」设计的最小形态，**正是 R2 所需的裸环境对照臂形态**。这是 dsh 对本项目最直接的可用价值，且不需要其任何代码。

---

## 5 建议与下一步（已于 2026-09-21 执行）

**建议采纳选项甲，否决选项乙。**

**唯一下一步**：为前端建立 TypeScript 地基。

### 5.1 执行中的一处设计修正（原文写错，原值保留于此）

原文写的是「将 `nativeApi.js` 更名为 `nativeApi.ts` 并引用生成的类型」。**该做法经实测否决，不得沿用。**

- 实测（2026-09-21）：把文件改名为 `.ts` 后跑 `npx vitest run`，**24 个测试文件 / 29 个用例转红**，
  失败原因一律是 `Failed to resolve import "../composables/nativeApi.js"`。
- 结论：**Vite / Vitest 不会把 `./x.js` 说明符重写到 `x.ts`。** 要 rename 就必须同时改 **20 处 import 点**
  （其中 13 处在 `.vue` 里），与「零 `.vue` 改动」直接冲突。已回滚。
- 改用方案：路径表**上移**到新建的 `src/api/paths.ts`。理由：只有 TS 文件里的 `const` 类型参数能保留
  字符串字面量类型（`Object.freeze` 会把每个属性拓宽为 `string`），而 JS 文件做不到。
  `nativeApi.js` 的文件名、20 处 import 点、运行时语义全部不变（仍逐组冻结、同名导出）。
- 附带收益：路径表从 `composables/` 归位到 `api/`，分层更正确。

### 5.2 另外两处与原文不同

1. 原文未提 `@typescript-eslint/parser`，实际必需：`.eslintrc.cjs` 的 root parser 是 `vue-eslint-parser`，
   不解析 TS 语法（`import type`、`<const T>` 直接报 `Parsing error`）。
2. 曾计划在 `contract.ts` 里用类型系统复刻 `check-native-boundary.mjs` 的遗留边界门。**实测后撤掉**：
   该断言自身的字面量（`'/api/taiji'` 等）会命中那条守卫正则，迫使守卫豁免 `contract.ts`，
   反而削弱刚给 `.ts` 加上的覆盖。改为把 `check-native-boundary.mjs` 的扫描范围扩到 `.ts`（净增强）。

### 5.3 验收门（已执行，读数见 §7.4）

1. `frontend` 现有 vitest 全绿；
2. `check:native-boundary` 与 `check:api-contract` 仍 PASS；
3. `npm run typecheck`（`vue-tsc --noEmit`）在 `tsconfig.json` include 白名单内零错误；
4. `npm run check:api-types` 证明生成物与冻结快照逐字节一致；
5. `npm run build` 成功，`npx eslint src --ext .js,.ts,.vue` 0 errors。

---

## 6 附：本简报未涉及

- 前端语言是否写入 `plans/active/roadmap/02_GATES_AND_CI.md`：**已核，不需要**。该文档是能力晋级门禁
  与认知闭环验收规则，不是 CI 步骤清单，前端 CI 门不属于它。
- 是否写入 `03_CURRENT_EXECUTION.md` 的「唯一下一步」：**已核，不可写**。该页按纪律只保留研究主线
  唯一队首（当前被 R2 三臂训练堵住），产品侧工程支线不得与之并列。

---

## 7 执行记录（2026-09-21）

### 7.1 落盘改动

| 文件 | 性质 |
|---|---|
| `frontend/tsconfig.json` | 新增。`strict` + `allowJs` + `checkJs: false`；`include` 为**显式白名单**（棘轮，只增不减） |
| `frontend/src/api/schema.d.ts` | 新增（生成物，7,144 行 / 202,933 字节）。源：`tests/snapshots/openapi_baseline.json` |
| `frontend/src/api/paths.ts` | 新增。路径表上移，用 `const` 类型参数保留字面量类型 |
| `frontend/src/api/contract.ts` | 新增。编译期契约门 |
| `frontend/src/composables/nativeApi.js` | 表上移，改为 import + 转出；文件名与 import 点不变 |
| `frontend/scripts/check-api-types.mjs` | 新增。生成物漂移门 |
| `frontend/scripts/check-native-boundary.mjs` | 扫描范围 `js\|vue` → `js\|ts\|vue` |
| `frontend/scripts/check-api-contract.mjs` | `facadeFile` → `api/paths.ts`；扫描加 `.ts`；排除生成物 |
| `frontend/.eslintrc.cjs` | `.ts` 的 `@typescript-eslint/parser` override；`ignorePatterns` 排除生成物；resolver 扩展名加 `.ts` |
| `frontend/.prettierignore` | 新增。必须排除 `src/api/schema.d.ts`——格式化会让漂移门失效 |
| `frontend/package.json` | 四个工具**精确钉版本**；新增 `typecheck` / `gen:api-types` / `check:api-types` |
| `.github/workflows/ci.yml` | `build-frontend` 加 `typecheck` + `check:api-types` 两步；eslint 扩展名加 `.ts` |

### 7.2 工具版本（精确钉；依据 ci.yml 的 R7 教训——拿输出当阈值的工具必须钉版本）

`typescript@5.9.3` · `vue-tsc@3.3.11` · `openapi-typescript@7.13.0` · `@typescript-eslint/parser@8.70.0`

**踩到的坑（值得记住）**：第一次 `npm i -D typescript` 不带版本，npm 写入 `"*"`，解析到
**typescript@7.0.2（原生 Go 版）**，其 `ts.factory` 为 `undefined`，`openapi-typescript` 直接崩溃。
这也是 `"*"` 这种浮动范围本身的违规——已改为精确钉。

### 7.3 守卫红/绿验证（纪律：每个新门、每个改过作用域的门，都必须各跑一次证明能响）

| 门 | 红测手段 | 实测结果 |
|---|---|---|
| `vue-tsc` 契约门 | `paths.ts` 的 `bootstrap` 改为 `/api/runtime/bootstrap_PROBE` | **红**：`contract.ts(43,54) error TS2344: Type '"/api/runtime/bootstrap_PROBE"' does not satisfy the constraint 'never'` |
| `check:api-contract` | 改为 `/api/taiji/probe` | **红**：`api/paths.ts: /api/taiji/probe is absent from OpenAPI snapshot` |
| `check:native-boundary` | 同上 | **红**：`api/paths.ts: forbidden native-boundary residue` |
| 全部 | 恢复原值 | **绿**（见 §7.4） |

三个红测一次探针同时证明：`contract.ts` 确实在类型程序内（不是空跑）、`check-api-contract` 的
`facadeFile` 指针迁移正确、`.ts` 新覆盖真实生效。

### 7.4 最终门禁读数

```
typecheck              rc=0
check:api-types        PASS: schema.d.ts 与 openapi_baseline.json 逐字节一致（202933 字节）
check:api-contract     PASS: 55 API literals match OpenAPI paths
check:native-boundary  PASS: 1 authoritative evidence entrypoint and Legacy boundary clean
vitest                 47 files / 267 tests passed
eslint                 0 errors / 13 warnings（13 条全部来自未改动文件的存量）
vite build             成功（4830 modules transformed）
```

### 7.5 未做与已知边界

- **未改动任何 `.vue` 文件**；45 个 `.vue` 的 `<script>` 仍为 js，`tsconfig.json` 的 include 也尚不含 `.vue`。
- `nativeApi.js` 的 `@typedef NativeApiFacade` 仍是裸 `Object`，且**已与实现脱节**：实现里的
  `chatWorkbenchInterpret`、`taijiWorkbenchRecoveryPortfolio`、`clientExtensionsDependency` /
  `Rollback` / `BeginCall` / `EndCall` / `Retire` / `Quarantine` 在 typedef 中均缺失。
  这是下一步的入口（手写类型层会腐坏，正是要换成生成型层的直接证据）。
- 将来 `.vue` 写 `lang="ts"` 时，`.eslintrc.cjs` 的 root 还需补
  `parserOptions.parser: '@typescript-eslint/parser'`（已在配置中留注释）。
- 有意留作后续棘轮的严格度：`exactOptionalPropertyTypes`、`noUncheckedIndexedAccess` 未开。

---

## 8 Electron 迁移评估（2026-09-21 由所有者提出，待裁定）

### 8.1 桌面壳实际承担了什么（实测）

`desktop/main.py` 共 **1,311 行**，职责为：QWebEngineView 嵌入前端（标题栏亦由前端 DOM 承载）、
系统托盘（最小化到托盘）、窗口尺寸/位置记忆、`subprocess` 起 uvicorn（8000）与进程内
WebSocket 服务（8765）、子进程崩溃自动重启并以 job object 保证随父进程退出。
另含 `backend_worker.py`(36) · `seed.spec` PyInstaller 双入口(166) · `installer.nsi`(92)。

**关键结构事实**：前端由后端自身托管——`api/app.py::_mount_static_assets` 把 `frontend/dist` 挂载
并带 SPA catch-all，还支持一个外置 `update_frontend` 覆盖路径。所以桌面壳本质上只是
**webview + 进程监管 + 托盘**，在 Electron 里是 `BrowserWindow.loadURL` + `child_process` + `Tray`。

### 8.2 支持迁移的实测证据

1. `main.py` 第 46–113 行（约 70 行）是纯 Qt/PyInstaller 战场疤痕：
   `_prepare_frozen_qt_dll_path()`、`QTWEBENGINEPROCESS_PATH`、`_QT_DLL_DIRECTORY_HANDLES`、
   `_QT_PRELOADED_LIBRARIES`——这一段是 PyInstaller 找不到嵌套 `Qt6/bin` 导致的 DLL 加载顺序问题，
   Electron 侧不存在对应概念，**整段消失**。
   **但需订正一处我先前讲过头的话**：同段里还有 `QTWEBENGINE_CHROMIUM_FLAGS=--disable-gpu
   --single-process` / `QT_OPENGL=software`，"整段消失"对这部分**不成立**。§9.4 实测显示，
   在受限会话里 Electron 同样需要 `--no-sandbox --disable-gpu --disable-gpu-compositing
   --in-process-gpu` 才能起来（否则 `FATAL: GPU process isn't usable. Goodbye.` 直接 abort）。
   差别在于：Qt 侧只能把这类开关写死在 70 行 ctypes 里，Electron 侧它是启动参数/环境变量。
2. `frontend/vite.config.js` 里有个 `strip-crossorigin` 插件，注释写明是「QWebEngineView 兼容」——
   又一处**只为 Qt 壳存在**的构建期 hack。
3. PyInstaller bundle 会因此卸掉整个 Qt6/QtWebEngine（体积主项），Electron 侧增量大致抵消；
   换来的是最脆的那一环（frozen Qt DLL 加载）彻底不存在。
4. 它把 §3 里最大的减分项消掉：**Electron 主进程天然是 TS ⇒ 前端 + 主进程 + preload 同语言**，
   OpenAPI 类型从 Python 一路贯到壳，不再只靠 schema 单点。

### 8.3 不建议低估的成本

1. **两个运行时**：Electron(Node) + Python，启动编排/生命周期/崩溃恢复要重写（`BackendManager`
   `_reap_orphan_listener`、`tcp_listener_pid`、`should_reap_listener` 这套孤儿监听进程回收是有分量的工程）。
2. `dist/Seed/SeedBackend.exe` 仍然要 PyInstaller——**PyInstaller 不退场，只是变瘦**。
3. `seed.spec` + `installer.nsi` 换 electron-builder 配置；签名/SmartScreen 需重做一遍。
4. 迁移期双轨（NSIS 产物与 Electron 产物并存）。

### 8.4 结论与顺序

**建议采纳 Electron**，但顺序必须是 **TS 地基 → Electron 壳**，理由：Electron 主进程要写成 TS，
若反过来先做壳，等于先用 JS 写一遍再转 TS。§7 已完成的 TS 地基正是这次迁移的前置（本无需返工）。

### 8.5 与选项乙（抄 dsh/ZCode）的关系

Electron 迁移**不改变** §3 对选项乙的否决。它解决的是「壳归到哪个生态」，不解决「agent 效果」；
后者在 ② 层（Python），不属于任何壳语言的作用范围。

---

## 9 Electron 壳迁移 —— 执行记录（2026-09-21）

### 9.1 落盘（新增目录，**未改动 `desktop/` 任何文件**）

| 文件 | 职责 |
|---|---|
| `desktop-electron/src/config.ts` | 端口/路径/环境契约，逐项标注对应的 `main.py` 常量 |
| `desktop-electron/src/logger.ts` | 文件 sink + 控制台；子进程输出用文件 fd（绝不用 pipe，原因见 main.py 注释） |
| `desktop-electron/src/settings.ts` | 窗口几何持久化，**复用同一份 `desktop/settings.json`** |
| `desktop-electron/src/backend.ts` | `BackendManager` 移植 + owner record 孤儿回收 |
| `desktop-electron/src/websocket.ts` | `WebSocketManager` 移植（8765） |
| `desktop-electron/src/preload.ts` | **QWebChannel 形状兼容层** |
| `desktop-electron/src/main.ts` | 应用生命周期/窗口/托盘/看门狗/桥接自检 |
| `desktop-electron/{package.json,tsconfig.json,electron-builder.yml,.gitignore}` | 工具链与打包配置（替代 `seed.spec` + `installer.nsi`） |

### 9.2 冒烟测试：端到端通过（本机实测）

```
2026-09-21 03:34:04.901 - SeedDesktop - INFO - Backend started on port 8000 (PID: 28676)
2026-09-21 03:34:15.021 - SeedDesktop - INFO - Backend is ready
2026-09-21 03:34:15.349 - SeedDesktop - INFO - WebSocket server started on port 8765 (PID: 9124)
2026-09-21 03:34:15.852 - SeedDesktop - INFO - WebSocket server ready on port 8765
2026-09-21 03:34:15.965 - SeedDesktop - INFO - Loading frontend: http://127.0.0.1:8000/#/?taiji_client=desktop
2026-09-21 03:34:17.235 - SeedDesktop - INFO - Frontend loaded successfully
2026-09-21 03:34:17.243 - SeedDesktop - INFO - Window bridge self-check passed (qt.webChannelTransport + QWebChannel present)
```

最后一行是本次移植**唯一被标注为「离线无法证明」的接口**：`contextBridge` 能否承载
「可 new 的函数 + 回调内传函数」。实测成立 ⇒ `AppTitlebar.vue` 能找到 `channel.objects.seedWindow`，
窗口控制按钮可用，**且 `.vue` 零改动**。

停止路径亦已验证：TaskStop 硬杀后端口 8000/8765 全部释放，`logs/seed_backend.owner.json`
按设计留存且指向已死进程（下次启动回收），清理后重跑正常。PyQt6 同名日志
（`logs/desktop_main.log`，含 2026-08-28 的 `Child job object armed` 记录）与 Electron 记录同文件共存，
双轨可直接并排对照。

### 9.3 与 PyQt6 的逐项对照

| 项 | 结论 |
|---|---|
| frozen Qt DLL 加载（`_prepare_frozen_qt_dll_path` 等） | **消失**（PyInstaller 特有问题） |
| `_EdgeResizeFilter`（约 50 行） | **消失**（`frame:false` 由系统原生处理边缘缩放） |
| `_apply_window_shape`（QRegion 圆角遮罩） | **消失**（透明窗 + 前端 CSS 圆角） |
| `_RestartWorker(QThread)` | **消失**（Node 异步，直接 await） |
| 注入 `qwebchannel.js` 资源 | 换成 preload 里的形状兼容层，**前端契约不变** |
| 拖拽 | 新增：主进程注入 `-webkit-app-region: drag`（Electron 无 `startSystemMove`） |
| toggle-maximize | 新增 300ms 去抖（原生标题栏双击与前端 `@dblclick` 会互相抵消） |
| 桥接自检 | 新增 `verifyBridge()`，失败留明确 error 而非静默失效 |
| 孤儿回收 | 由 Job Object 改为 **owner record**（见 9.5.6） |

### 9.4 复现命令

```bash
cd desktop-electron
npm install          # 若 electron 二进制未下载：见 9.5.3
unset ELECTRON_RUN_AS_NODE
SEED_DISABLE_GPU=1 \
SEED_PYTHON="C:/Users/23747/AppData/Local/Programs/Python/Python312/python.exe" \
  ./node_modules/electron/dist/electron.exe . \
  --no-sandbox --disable-gpu --disable-gpu-compositing --in-process-gpu
```

`SEED_PYTHON` 是必需的：本机 PATH 上的 `python` 是 managed 3.13（无 uvicorn），
Python312 才有 `uvicorn 0.52.1 / fastapi 0.141.1`。
后四个 Chromium 开关是**受限会话下**的必需项，普通桌面环境可能不需要，且 `--no-sandbox`
是安全降级——因此它们只出现在启动命令里，**不写进应用**。

### 9.5 未闭合项（诚实清单）

1. **`SeedWs.exe` 缺失**：`main.py` 的 frozen 分支用进程内守护线程跑 8765 WS，Electron 无法在
   Node 里跑 Python，改为独立子进程，但 `seed.spec` 目前只有 `Seed.exe` / `SeedBackend.exe`
   两个入口。frozen 分支会**明确报错并记日志**，不会静默降级。补第三个入口即可闭合。
2. **frozen 路径整体未验证**：本机无法构建 `SeedBackend.exe`，dev 路径已实测，打包路径未测。
3. **`ELECTRON_RUN_AS_NODE=1` 是本机环境变量，不是仓库问题**。未清掉它时 `electron.exe --version`
   打印 `v24.21.0`（Node 版本）而非 `v44.4.3`，`require('electron')` 退化为返回路径字符串，
   表现为 `TypeError: Cannot read properties of undefined (reading 'isPackaged')`。
   **症状与二进制缺失高度混淆，值得记住这一条判据**。
4. **electron 二进制未随 `npm install` 下载**（287 包 29 秒装完但无 `dist/`）。
   补下命令：`ELECTRON_MIRROR=https://npmmirror.com/mirrors/electron/ node node_modules/electron/install.js`。
5. **未验证**：UI 实际交互（三个窗口按钮、拖拽、托盘菜单、关闭到托盘）。日志只能证明桥已就位。
6. **孤儿回收语义变更**：Node 无法调 Win32，Job Object 的「主进程怎么死都回收」不成立。
   改为启动子进程时把 `(electronPid, backendPid)` 写入 `logs/seed_backend.owner.json`，
   判定变成「electronPid 存活 ⇒ 别的实例在用，不碰；已死而 backendPid 存活 ⇒ 确凿孤儿，回收」。
   精度不低于原来的 image+ppid 规则，但**跨壳有缺口**：PyQt6 侧不写这份记录，它留下的
   `SeedBackend.exe` 孤儿 Electron 认不出来（反方向由 PyQt6 自己的规则兜底，故任一侧启动都自愈）。
   要彻底闭合需两侧共用同一份 record。
7. **本机未做**：`npm run lint`（desktop-electron 无 eslint 配置）、单元测试（无测试框架）。
   目前唯一的门是 `npm run typecheck`，且它已用 13 个真实错误证明过会响。

### 9.6 待所有者裁定

1. **GPU/沙箱降级开关是否成为默认**：§8.2 的订正说明受限环境下 Electron 也需要降级开关。
   默认保留 GPU（迁移收益之一）还是默认 `--disable-gpu`（与 Qt 基线对齐）？
2. **是否加单实例锁**：PyQt6 允许开多个实例（第二个的后端会 bind 失败、白窗重试）。Electron 可用
   `requestSingleInstanceLock()` 消除该失效模式，但会引入与 PyQt6 的行为差异，故未擅自加。
3. **`SeedWs.exe` 第三入口的排期**（§9.5.1）。

---

## 10 SeedWs 第三入口与 frozen 路径验证（2026-09-21 续）

**本轮闭合了 §9.5 的 1、3、4 三项**；2、5、6 与 §9.6 的三项裁定仍在。

### 10.1 闭合 §9.5.1：`SeedWs.exe` 已产出并被真实使用

- 新增 `desktop/seed_ws.py`：8765 的独立进程入口，用法 `SeedWs.exe [port]`。与既有
  `desktop/backend_worker.py` 完全同构（函数级 import、依赖 PyInstaller 的 `pathex=[ROOT]`），
  故 `python desktop/seed_ws.py` 直接跑会 `ModuleNotFoundError: neuroplex` —— **这是预期的，
  不是缺陷**；dev 路径走的是 `python -m neuroplex.core.websocket_server`（已由 §9.2 验证）。
- `desktop/seed.spec` 改为**三入口**：新增 `a_ws` Analysis、把 `a_ws` 纳入 ICU 过滤循环、
  `MERGE` 三项、`pyz_ws` / `exe_ws`、`COLLECT` 补齐 `exe_ws + a_ws.{binaries,zipfiles,datas}`。
  PyQt6 出货路径**行为零变化**（它用进程内线程跑同一模块，不需要该入口）。
- `scripts/release.py` 的 `_verify_artifacts` 补上 `SeedWs`（否则新入口缺失时无人报错），
  并同步更新两处「双入口」措辞。
- 实测（`python -m PyInstaller --clean --noconfirm desktop/seed.spec`，与 release.py 同参）：
  三个 PYZ / 三个 PKG / 三个 EXE 全部构建成功，`COLLECT` 完成；
  `dist/Seed/` = `Seed.exe` + `SeedBackend.exe` + `SeedWs.exe` + `_internal/`（9,293 文件，1.45 GB）。

### 10.2 frozen 分支端到端验证通过

先加了验证接缝（§10.3），再对着上述真实产物跑 Electron：

```
Backend worker started on port 8000 (PID: 32936)      ← "worker" = 确认走的是 frozen 分支
Backend is ready
WebSocket server started on port 8765 (PID: 10848)    ← SeedWs.exe
WebSocket server ready on port 8765
Loading frontend: http://127.0.0.1:8000/#/?taiji_client=desktop
Frontend loaded successfully
Window bridge self-check passed (qt.webChannelTransport + QWebChannel present)
```

全链路零告警。三条结论：frozen 分支的 `SeedBackend.exe` / `SeedWs.exe` 解析与拉起成立；
打包后端确实托管了前端（`api/app.py::_mount_static_assets` 的 `_internal/frontend/dist`）；
桥接自检在 frozen 下同样通过。

### 10.3 新增验证接缝：`SEED_FORCE_FROZEN` / `SEED_ROOT_DIR`

`SeedBackend.exe` / `SeedWs.exe` 只有 packaged 才走得到，而 electron-builder 打一次包成本很高。
这两个环境变量允许在源码树上直接跑 frozen 分支、对真实 PyInstaller 产物做端到端验证。
默认关闭，出货路径不受影响。

**注意**：改动 `desktop-electron/src/*.ts` 后**必须先 `npm run build`** 再跑 `electron.exe`，
否则用的是旧 `dist/`。本轮曾因此得到一个假读数（新加的环境变量看起来"没生效"），见 §10.5。

### 10.4 本轮抓到的两个真 bug（都是 frozen/repeat 运行才暴露的）

1. **`reapOrphanedBackend` 漏了自我排除**。`main.py` 原有 `owner == os.getpid()` 一条，移植时丢掉。
   后果：看门狗重启时把**自己上一轮的 child** 当成「另一个客户端实例」而跳过回收，随后在同一
   端口上再起一个后端，两个必然互相 bind 失败。日志里报出的 PID 就是自身。已修为四态判定
   （自己 / 别人的活实例 / 孤儿 / 无可回收）。
2. **`findBrandIcon()` 的 frozen 候选路径错**。沿用了 Electron 的 `process.resourcesPath` 概念，
   而本项目 frozen 布局是 PyInstaller onedir，datas 在 `_internal/` 下：
   `dist/Seed/_internal/frontend/dist/seed-taiji-network.png`。后果是**托盘被直接禁用**、
   窗口图标缺省（日志 `Brand icon not found; tray disabled`）。已改为两套根都探测，并在落空时
   打印候选数，使再次失败可诊断而非静默降级。

### 10.5 我自己的流程错误（如实记录）

首次 frozen 探针**无效**：我直接跑 `node_modules/electron/dist/electron.exe .`，绕过了
`npm run build`，于是 `dist/` 是旧的、`SEED_FORCE_FROZEN` 根本没进产物，探针实际跑的是 dev 分支。
这个无效探针意外暴露了 §10.4.1 那个 bug（因为 dev 分支也会走 `reapOrphanedBackend`）。
结论：**`npm start` 的 `build && electron .` 顺序不是装饰，别绕过它。**

### 10.6 更新后的未闭合清单

| §9.5 项 | 状态 |
|---|---|
| 1 `SeedWs.exe` 缺失 | **已闭合**（§10.1） |
| 2 frozen 路径未验证 | **已闭合**（§10.2） |
| 3 `ELECTRON_RUN_AS_NODE` 判据 | 已记录，未闭合（环境侧事项） |
| 4 electron 二进制需镜像补下 | 已记录，未闭合（环境侧事项） |
| 5 UI 实际交互未验证 | **仍未验证**（按钮/拖拽/托盘菜单/关闭到托盘；日志只能证明桥已就位） |
| 6 孤儿回收跨壳缺口 | 仍未闭合（PyQt6 不写 owner record） |

`dist/`（1.45 GB）为构建产物且已被 `.gitignore` 忽略；下一次 `release.py` 会经 `clean_outputs()`
重建它。若要回收磁盘可直接删除 `dist/` 与 `build/`。

### 10.7 补做的回归检查：PyQt6 出货路径未被破坏

我改的是**出货路径的** `desktop/seed.spec`，但 §10.1/§10.2 只跑了 `SeedBackend.exe` 与
`SeedWs.exe`。构建成功 ≠ 运行正常，因此补跑了一次 `dist/Seed/Seed.exe`：

```
Backend worker started on port 8000 (PID: 28984)
Child job object armed (kill-on-close)            <- Job Object 路径仍工作
Backend is ready
WebSocket server started in-process on port 8765  <- 进程内线程，与原设计一致
WebSocket server ready on port 8765
Loading frontend: http://127.0.0.1:8000/#/?taiji_client=desktop
Page loaded: http://127.0.0.1:8000/#/?taiji_client=desktop (ok=True)
Frontend loaded successfully
websockets.server - connection open
Taiji.WebSocket - 新客户端连接: ('127.0.0.1', 49542)   <- 前端 useWebSocket 已连上 8765
```

结论：三入口改造与 `release.py` 的校验补齐**未破坏 PyQt6 出货路径**——不只窗口起来，
前端与 8765 的 WebSocket 通道也真实建立。

QtWebEngine 打出的 `Failed to create GLES3 context, fallback to GLES2` 与
`Cannot use V8 Proxy resolver in single process mode` 是受限桌面下的既有噪声
（正对应 main.py 里 `--disable-gpu --single-process` 那段注释），不影响启动。
硬杀后 8000/8765 全部释放、无残留 Seed 进程 —— Job Object 的 kill-on-close 语义仍然成立。

与本次改动无关的既有现象（仅留痕）：spec 的 datas 中 `(ROOT/"version.json", ".")` 被
`if src.exists()` 静默跳过（`dist/Seed/_internal/version.json` 不存在），因仓库根无该文件。
