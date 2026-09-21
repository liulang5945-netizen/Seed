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
   `QTWEBENGINE_CHROMIUM_FLAGS=--disable-gpu --single-process`、`QT_OPENGL=software`、
   `_prepare_frozen_qt_dll_path()`、`QTWEBENGINEPROCESS_PATH`、`_QT_DLL_DIRECTORY_HANDLES`、
   `_QT_PRELOADED_LIBRARIES`——Electron 的 Chromium 自带解决，**整段消失**。
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
