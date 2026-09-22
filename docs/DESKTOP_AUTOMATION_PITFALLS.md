# 桌面 / 前端工具链与打包的坑（本机实测记录）

> 2026-09-22 由 `.workbuddy/memory/MEMORY.md` **单方合并瘦身**时移出（原文逐条保留，未改写）。
> MEMORY.md §4 只留一行指针：**细节在本文**。每条都是**本机实测**，不是推断或传闻。
> 适用范围：产品侧工程支线（前端 TS 地基、`desktop-electron/` 壳、Electron 打包），**不入研究主线队列**。

## 前端工具链（2026-09-21 实测）

- Vite / Vitest **不会**把 `./x.js` 说明符重写到 `x.ts` ⇒ 任何 JS→TS 改名前必须先验解析
  （本仓 20 处 import 会全断；实测 24 个测试文件 / 29 个用例转红）。
- `Object.freeze` 会把属性**拓宽为 `string`** ⇒ 需要字符串字面量类型时，表必须在 `.ts` 里用
  `const` 类型参数，JS 侧做不到。
- `npm i -D <pkg>` 不带版本会写入 `"*"`，实际解析到 `typescript@7.0.2`（原生 Go 版，
  `ts.factory` 为 `undefined`）⇒ **工具一律精确钉版本**。

## Electron 壳（2026-09-21 实测）

- 本机环境设了 `ELECTRON_RUN_AS_NODE=1` ⇒ `electron.exe --version` 打印 **Node** 版本（`v24.x`）
  而非 `v44.x`，且 `require('electron')` 退化为返回路径字符串，报
  `Cannot read properties of undefined (reading 'isPackaged')`——**症状与「二进制没装」极易混淆**；
  启动前必须 `unset`。
- Electron 二进制**不随** `npm install` 下载（287 包 29 秒装完但无 `dist/`）。补下：
  `ELECTRON_MIRROR=https://npmmirror.com/mirrors/electron/ node node_modules/electron/install.js`。
- 受限会话下另需 `--no-sandbox --disable-gpu --disable-gpu-compositing --in-process-gpu`
  （只给 `--disable-gpu` 不够：Chromium 仍会起软件合成 GPU 进程，被自身沙箱挡住 ⇒
  `FATAL: GPU process isn't usable. Goodbye.`）。
- **圆角只归一层**：壳与应用各画一层圆角必出缝隙（实测直角残留 + 双层框）。壳只负责把
  `html/body` 挖透明，圆角/边框/最大化交还应用自己的 `.app-wrapper`。
- 验证产物里「是否包含某改动」**不能靠哈希**（本轮两次不同构建 sha256 前缀曾相同）⇒
  应 grep 产物里的特征字符串（如 `app.asar` 里 `in-process-gpu` 的计数）。

## 移植「进程守卫」类逻辑必查自我排除

- `desktop/main.py` 原本有 `owner == os.getpid()` 一处；移植时漏掉后，看门狗重启会把
  **自己上一轮的 child** 误判成「另一个实例」而跳过回收，随后同端口再起一个 ⇒ 两个互相 bind 失败。
- **判据**：凡「按持有者判孤儿」的逻辑，先问「持有者可能是我自己吗」。
- 另：改动 `desktop-electron/src/*.ts` 后**必须先 `npm run build`**——直接跑 `electron.exe .`
  会用到旧 `dist/`，导致新加的环境变量看起来"没生效"（实测踩过）。

## electron-builder 打包（2026-09-21 实测）

三个卡点都要绕：

1. **工具链下载挂死**：从 `app-builder-lib/out/toolsets/windows.js` 读确切版本 + **官方 sha256**，
   手工抓取校验后放入 `%LOCALAPPDATA%/electron-builder/Cache/<releaseName>/`。
2. **`unpacking default Electron distribution` 死住** ⇒ 配 `electronDist: node_modules/electron/dist`。
3. **`release/` 已存在时 `copying unpacked Electron` 死住**（32/75 文件后零增长）⇒ 每次构建前清空
   `release/`（已固化为 `scripts/clean-release.mjs` 并接进 `npm run dist`；该脚本后来改为
   **只重命名不删除**，因为关键路径上 `fs.rmSync` 删陈旧目录会随机挂住 0.2 s～22 min）。

**根因未定性**（工具缺陷 vs 本机 VM 大文件 I/O 不稳）——只记可复现的最短操作序列，不下结论。

另两条：

- Electron 包默认**只含壳**，Python 侧载荷（`SeedBackend.exe` / `SeedWs.exe` / `_internal`）
  需另行以 `extraFiles`/`extraResources` 打进包，否则装完是空壳。用 `extraFiles` 时 filter 必须
  排除 `Seed.exe`，否则 PyInstaller 里的 PyQt6 GUI 会**覆盖掉 Electron 自己的 Seed.exe**。
- **NSIS `/D=` 自定义路径在 bash 里必须加引号** —— `/D=E:\xxx` 未加引号时 `\_` 被当转义吃掉 `\`，
  安装器把畸形路径写进 `InstallLocation`/`UninstallString`，表现为
  「安装成功、运行正常、卸载却什么都不删」（rc=0），**极易误诊为产品缺陷**。实测踩过，
  结论曾一度误判为产品缺陷并反转。

## 相关文档

- 决策与评估：`plans/reference/FRONTEND_TS_VS_HARNESS_ADOPTION_DECISION_BRIEF_20260921.md`
- 仓库卫生：`docs/REPO_HYGIENE_RULES.md`　目录结构：`docs/FOLDER_STRUCTURE_RULES.md`
