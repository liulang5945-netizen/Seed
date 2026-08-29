# Seed / Taiji 路线执行记录：P8 产品与发布

> 本文由原总路线图按职责拆分而来。原始行号：496–992；本文件属于当前计划资料或历史证据，具体身份以本文件所在目录为准。
> 这是已完成产品、桌面、打包和发布工作包的历史证据，不是当前执行入口。

## 13. P8：产品与发布

- Seed UI/API 只展示已通过 Gate 的 Taiji 能力。
- S2 安全、覆盖率和门禁继续完成；S3 打包、版本、更新和回滚在 v1 API 稳定后收口。
- 默认发行不安装 Legacy 重依赖；Legacy 只保留离线对照和显式兼容构建。
- 发布物包含模型卡、数据卡、能力 Gate、失败边界和恢复方式。

### 13.1 桌面客户端 UX 修复轮（2026-08-27）

实测澄清的运行形态：桌面端（`desktop/main.py`，PyQt6 无边框窗）= 子进程 uvicorn `api.app:app`(8000，同时服务 REST 与 `frontend/dist` 静态前端) + 子进程 WS 服务器(8765)；聊天走 Seed 原生运行时（`checkpoints/seed_corpus.pt`，**0.51 M 可学习权重 / 960 神经元式单元**，详见 §13.3.1 规模勘误；底层仍为 byte predictor，本轮当时的语言器官是 `structured-stub`，产品表层已在后续 P6 Gate 改为 `native-readable`）。本轮十项修复：

| # | 问题 | 根因 | 修复 |
|---|---|---|---|
| 1 | 外框边框不跟主题 | 标题栏 QSS 只在加载后同步一次 | `desktop/main.py` 1s 轮询 `data-theme`，变化才重设 QSS |
| 2 | 进入页面弹「已刷新」 | `AgentConfigView.onActivated` 调带 toast 的刷新 | 自动刷新静默化，仅手动点击提示 |
| 3 | 页面切换生硬 | router-view 无过渡 | `App.vue` 增加 `route` 过渡（out-in，220ms，reduced-motion 降级）。**⚠ 本项引入白屏回归，已在 §13.3 推翻重做** |
| 4 | IDE 无法唤起系统文件管理器 | Web 沙箱无原生对话框 | 后端 `POST /api/workspace/pick_folder`（PowerShell STA BrowseForFolder）+ 前端「浏览系统目录」。**⚠ 仅解决"选得到"，对话框仍弹在主窗后面，见 §13.3** |
| 5 | IDE 简陋 / 终端不可用 | 终端 WS 在 auth 关闭时默认拒绝 | 终端默认放行（与全局 JWT 中间件一致，可配置收紧）；新增 Ctrl+\`、Ctrl+P 快速打开、新建文件夹、刷新树、「在资源管理器中显示」(`/api/workspace/reveal`)、`/api/workspace/mkdir`。**⚠ "默认放行"是局域网免鉴权 shell 漏洞，已在 §13.3 收紧为对端地址感知** |
| 6 | 侧边栏搜索右侧不明符号 | macOS 专用 `⌘K` 硬编码 | 平台感知提示（Win/Linux: `Ctrl K`），并真正绑定 Ctrl+K 聚焦 |
| 7 | 「你好」回复乱码 | **模型真实输出**：0.51 M 权重的 byte 级基底 + raw prediction 未经过语言器官；旧 `structured-stub` 只会做无损结构序列化，不会形成可读语言 | 本轮先做诚实呈现：后端 final 事件标注 `readable`（U+FFFD/控制符占比启发式），前端以「RAW 原始字节输出」卡片呈现而非伪装成正常回复，历史消息同启发式。**根治已在 P6 语言表层 Gate 落地**（见 §16）：聊天路径构造 Taiji-owned `ExpressionPlan`，先经本地 `native-readable` 表层，有效候选保留、不可读 prediction 转为诚实可读状态文本，final event 暴露 `language_backend`，前端只在真正不可读时显示 RAW 调试卡片 |
| 8 | 输入栏按钮「没用」 | 按钮实际可用（Chromium 实测全通过）；体感来自发送按钮 disabled 且无反馈 | 发送门控保留但移除 disabled，点击未就绪时 toast 明确原因（连接中/模型未装载/生成中） |
| 9 | 生命状态数据来源存疑 | needs 数据源是 Cortex legacy `life_scheduler`；Seed 运行时下后端返回空（无假数据） | `LifeStatusView` 增显式 DATA SOURCE 说明卡；`is_seed` 透传至前端；Seed 下生命活动按钮给真实提示 |
| 10 | 对话页面无法上下滑动 | `.chat-stage` 为 `flex:1; min-height:0`，在 flex 列滚动容器中被压缩到小于内容高度；内容以 `overflow:visible` 溢出绘制，但父级 `scrollHeight` 仍按 stage 盒子计算 ⇒ 滚动条永不出现，内容被 sticky 输入栏遮挡 | `.chat-stage` 改为 `flex:1 0 auto`（可涨不可缩，去掉 `min-height:0`）；`.composer-wrap` 加 `flex:none; z-index:2`；`.msg` 的 `contain-intrinsic-size` 由 80px 提到 140px 以减少 `scrollHeight` 失真 |

滚动修复的实测证据（Chromium，注入内容后量测）：修复前 h=610 时 `stageScroll 433 > stageBox 397` 而 `saScroll === saClient(558)`、滚轮无效；修复后 `stageBox === stageScroll(449)`、`saScroll 610 > saClient 558`、滚轮生效；12 条真实消息场景 `saScroll 1565`、可滚到底且末条消息 bottom(522) < 输入栏 top(538) 不被遮挡。

配套：OpenAPI 基线快照已更新（新增 3 个 workspace 端点）；vitest 160/160、e2e 冒烟 22/22 通过；`frontend/dist` 已重建。

遗留（下一轮候选）：native-readable 已解决产品乱码与 structured-stub 误用，但它不是开放域语言模型；下一轮需为 Taiji-owned `ExpressionPlan` 建立真实语言表达训练/holdout Gate。终端默认 shell 仍是 cmd.exe；侧边栏搜索框尚未接线为会话过滤。

### 13.2 打包链收敛与客户端重打包（2026-08-27）

上一轮十项修复提交后，用户实测反馈「没有重新构建前端和打包客户端」。核查结论一分为二：`frontend/dist` 确已重建（构建产物含全部修复），但 `dist/Seed/Seed.exe` 仍是 08-24 17:40 的旧包（45.43 MB），内置 `index.html` 哈希与源码构建不一致 ⇒ 客户端里跑的是三天前、不含任何修复的前端。

**机制层根因**：存在两条重叠的桌面打包入口，而被文档推荐的那条恰好缺少防漂移断言。

| 入口 | 独有能力 | 缺陷 |
|---|---|---|
| `scripts/release.py`（CONTRIBUTING 推荐） | 前端 + PyInstaller + NSIS 编排、产物验证 | **缺少**「源码 dist == 客户端内置 dist」字节断言；无旧产物清理 |
| `desktop/build.py` | 字节级前端一致性断言、dist/build 清理、运行时可写目录后处理 | 未被文档与 CI 之外的任何流程调用 |

按「机制演化时收敛、清理旧的」收敛为**唯一入口** `scripts/release.py`：

- 并入 `_verify_packaged_frontend()`——对比 `frontend/dist/index.html` 与 `dist/Seed/_internal/frontend/dist/index.html` 字节，把「改了前端却打出旧包」从静默漂移变成显式构建失败；在 PyInstaller 之后作硬门禁，并复用于 `_verify_artifacts()`。
- 并入 `clean_outputs()`（dist/build 清理，新增 `--no-clean` 供增量调试）与 `postprocess()`（随包复制 `knowledge_store/`、`user_data/`、`security/`，创建 `agent_workspace/`、`taiji_data/{feed,sleep,life,evolution}_data/`）。
- 步骤重编号为 [1/4]…[4/4]；`_verify_artifacts()` 修正为按 `seed.spec` 的 `COLLECT name="Seed"` 检查 `dist/Seed/{Seed,SeedBackend}.exe`（非 Windows 跳过 SeedBackend）。
- 删除 `desktop/build.py`；同步更新 `ci.yml` F05 步骤（不再 py_compile 已删文件）与 `seed.spec` 文档字符串。

顺带修掉三个会让完整发布必然失败的 NSIS 缺陷（`makensis` 的 `OutFile`/`File` 相对**工作目录** `desktop/` 解析，而非 .nsi 所在目录）：

| 缺陷 | 现象 | 修复 |
|---|---|---|
| `OutFile "SeedSetup.exe"` | 装机包落在 `desktop/`，而验证步骤查 `dist/SeedSetup.exe` ⇒ 永远失败 | `OutFile "..\dist\SeedSetup.exe"` |
| `File /r "dist\Seed\*.*"` | 去找不存在的 `desktop/dist/Seed`，与同文件 `..\icon.ico` 自相矛盾 | `File /r "..\dist\Seed\*.*"` |
| `APP_VERSION "1.6.0\"` 多余反斜杠 | 版本串被污染 | 源头在 `scripts/sync_version.py` 的 raw f-string `rf"...\""`（raw 串里 `\"` 会把反斜杠写进文件），改为单引号 f-string `rf'\g<1>{ver}"'`，杜绝再生 |

**重打包实测证据**（`python scripts/release.py --skip-nsis`，本机无 NSIS）：

| 指标 | 结果 |
|---|---|
| 流水线 | 清理 dist/build → 前端构建 → 一致性校验通过 → 后处理 → 产物验证通过，`Seed v1.6.0 构建完成`，1273.7 MB / 9111 文件 |
| `dist/Seed/Seed.exe` | 69.14 MB @ 2026-08-27 23:46:56（旧包 45.43 MB @ 08-24 17:40:58）|
| `dist/Seed/SeedBackend.exe` | 69.07 MB，同时间戳 |
| index.html 哈希 | 源码 == 打包 == `76E4B2B8…17BA`，**MATCH** |
| 打包内资产抽查 | CSS `flex:1 0 auto`、`contain-intrinsic-size:auto 140px`、JS `raw-output` 均命中；`seed_corpus.pt`、`tokenizer_contract.json`、`agent_workspace/`、`taiji_data/life_data/` 就位 |
| 冷启动冒烟 | `Seed` + `SeedBackend` 双进程存活，`GET /api/health` 200，`model_loaded:true`、`seed_active:true`、`security_middleware:true` |

冒烟返回的 `language_provider.backend_id = "structured-stub"` 再次确认 §13.1 第 7 项（乱码）的根因仍在语言器官，**下一步应做 P6 真实语言器官接入**，而非继续在 UI 侧修补。

附注：`seed.spec` 的 `_datas` 用 `if src.exists()` 软条件，`version.json` / `app_settings.json` 在仓库中本就不存在且无任何代码读取，被静默跳过属预期，不是本次打包缺陷。

### 13.3 白屏回归根治、原生对话框前台化、终端鉴权收紧与规模勘误（2026-08-28）

用户实测反馈四件事：① 各页面点着突然全变空白（最严重）；② IDE 能选文件了但仍拉不起系统文件管理器；③ 终端和文件各有两个重复按钮；④ 追问模型真实规模。前三项均是 §13.1 修复本身的回归或未彻底，按「机制演化时收敛、清理旧的」逐项推翻重做。

#### 13.3.1 模型规模勘误（口径统一）

`checkpoints/seed_corpus.pt` 是自研 `seed-native-v1` 格式，**不是** PyTorch `state_dict`；稀疏突触以 `pre_index`（拓扑，整型索引）+ `edge_weight`（权重）成对存储，直接 `sum(numel())` 会把拓扑当参数一并计入。

| 口径 | 数值 |
|---|---|
| **可学习权重** | **509,521 元素（≈0.51 M），40 个张量** |
| 拓扑索引 `pre_index` | 506,768（不是参数） |
| 其他状态量 | 13,539 |
| 张量元素合计 | 1,029,828 |
| 文件体积 | 3.95 MB |
| **神经元式单元** | **960** = 皮层 `[256,192,128]`=576 + `memory_units` 384 |
| 字符表 / 训练 tick | 257 / 4,800,000 |
| 语料 / 训练器 | `simple_zh_texts.jsonl`（1,394,775,610 B）/ `train_seed_corpus`，存档 2026-08-23T09:28:23Z |

**结论**：此前记载的「43.7 万参数」是把 `pre_index` 与 `edge_weight` 混算所致，准确数字是 **0.51 M 可学习权重**。这是微型类脑基质，不是 Transformer 量级 LLM——§13.1 第 7 项乱码属该规模下的预期行为。§13.1 相关表述已同步勘误。

#### 13.3.2 全页白屏：`out-in` + `keep-alive` + `:key` 三者互斥

> **诊断范围勘误（2026-08-28，§13.8）**：本节修掉的是**真实存在的过渡竞态**（`:key` 与 keep-alive 语义冲突、`delayedLeave` 持旧 vnode），这部分结论与收敛依然有效。但当时把它当作用户所报白屏的**唯一**根因，是**推理而非观测**——没有打开真实浏览器控制台看有无异常。用户随后二次上报同一现象，§13.8 用远程调试实测到真正的致命项是 `FileUploadQueue.vue` 把 emoji 字符串喂给 `<component :is>`，在 Blink 下抛 `InvalidCharacterError` 并摧毁整个 router-view 子树。两者是**不同层的两个缺陷**，本节不构成对用户所报白屏的完整解释。

根因在 §13.1 第 3 项引入的 `App.vue` 过渡结构 `<transition mode="out-in"> → <keep-alive> → <component :is :key="$route.path">`，三个因素叠加致命：

1. `:key="$route.path"` 强制每次导航销毁重建，**使基于组件 name 的 keep-alive 缓存永不命中**，且同一次更新里 `KeepAlive` 返回全新 vnode；
2. `mode="out-in"` 把 enter 阶段推迟到 leave 完成后，经由绑定**旧 vnode** 的 `delayedLeave` 回调触发；
3. 用户行为恰是快速连点切页 ⇒ 下一次导航在上一次过渡未结束时抵达。

结果：`delayedLeave` 持有旧 vnode，而 `:key` 已变的新元素拿不到 enter 钩子，**停留在 `.route-enter-from` 的 `opacity: 0`**——DOM 完整存在、只是全透明。这解释了为何白屏无任何报错、也不触发 `RouteErrorView`（后者渲染可见 UI）。时间线亦吻合：`b656ff5` 只改了 `App.vue`，两个 CSS 文件未动。

修法（不是加补丁，而是拆掉互斥前提）：

- 删除 `:key="$route.path"`——它与 keep-alive 语义冲突，是纯冗余；
- **彻底删除离场过渡 CSS**（`.route-leave-active` / `.route-leave-to`），只保留淡入。`out-in` 下 leave 因检测不到 CSS 过渡而同步结束，`delayedLeave` 竞态窗口归零；enter 即使被打断，元素也只是丢掉 class 回落到自然的 `opacity: 1`，**物理上不可能卡在透明态**。

排除过程（逐一实证否定）：`.router-wrapper` 重复声明（`index.css` 导入序 shell→app，且级联按属性生效，后者不含 `flex` 无法取消前者）、`views/ChatView.vue` 缺失（自查误报，实际在 `components/`，路由引用正确）、多根模板、chunk 加载失败（`npm run build` exit 0，7 个 chunk 齐全）、`animations.css`/`overrides.css` 冲突关键帧（零匹配）、`appStore.applyBgImage()`（只改背景图）、`product.css`（只改背景色）——全部排除后嫌疑完全收敛到过渡组合。

顺带收敛：`.router-wrapper` 从 `app.css` + `styles/shell.css` 两处重复声明合并到 `shell.css` 单一定义（合并前先把 `app.css` 独有的 `background: var(--bg)`、`min-height: 0` 迁入，避免静默丢样式），`app.css` 处留指向注释。

#### 13.3.3 原生目录对话框：不是没创建，是没有宿主窗口

§13.1 第 4 项的 `Shell.Application.BrowseForFolder(0, ...)` 传 **hwnd = 0（无归属窗口）**。实证探针显示子进程阻塞 6.1 秒并生成 Explorer iconcache 临时文件 ⇒ **对话框确实被创建了**，只是拿不到前台激活，弹在无边框 Qt 主窗**后面**，用户完全看不见。诊断由此从「未创建」反转为「创建了但没前台化」。

修法：改用 WinForms `FolderBrowserDialog`（BIF_NEWDIALOGSTYLE 可缩放树），并先创建一个 `TopMost=$true`、`Opacity=0`、1×1、`ShowInTaskbar=$false` 的宿主窗体，`Show()` + `Activate()` 后以它为 owner 调 `ShowDialog($owner)`，用完 `Close()`/`Dispose()`。同时移除 `-NonInteractive`（本调用的全部目的就是展示交互式 UI），保留 `-STA`（COM 对话框需单线程套间）。

可行性交叉验证：`desktop/main.py:601` 仅设 `Qt.WindowType.Window | FramelessWindowHint`，**无 `WindowStaysOnTopHint`** ⇒ TopMost 宿主窗必然压在主窗之上。新脚本探针复测：脚本长度 693、8 秒超时未返回（对话框正常等待输入）+ iconcache 副作用，语法与行为均成立。

#### 13.3.4 重复按钮收敛（各留视觉层级最高的那个）

| 功能 | 保留 | 移除 | 理由 |
|---|---|---|---|
| 终端 | 顶栏 `终端` 按钮 | 右栏「快捷操作」分组内的 `quick-btn` | 顶栏项有 `active` 状态、与 运行/保存 同组、图标+文字，视觉层级最高；右栏那个是分组里唯一的孤立填充 |
| 目录 | 顶栏 `打开文件夹` | 空树状态里的 `切换目录` 按钮 | 顶栏常驻可见；空态按钮只在空态出现，改为指向顶栏的文案，避免死路 |

配套清理：右栏「快捷操作」分组整体删除（其唯一子项已移除）、`.quick-btn` 相关死 CSS 删除并留注释；`Terminal` 图标 import 保留（顶栏与文件图标映射 `sh: Terminal` 仍在用）。

#### 13.3.5 终端免鉴权漏洞：判定依据从配置项改为对端地址

§13.1 第 5 项把 `terminal_allow_unauthenticated` 默认为 `True`，前提写的是「默认 127.0.0.1」；但 README 推荐用 `SEED_HOST=0.0.0.0` 让手机连电脑（`901a8c5`），两者叠加**在局域网上暴露一个免鉴权 shell**。且 `_verify_ws_token` 的 docstring 写「默认不允许」，与代码相反。

修法上取上限更高的方案：**不读 `SEED_HOST` 之类的服务端声明，而是判定这条连接的真实对端地址**——绑定 `0.0.0.0` 时回环与局域网请求走同一个监听套接字，只看绑定值根本无法区分风险来源。新增 `_is_loopback_peer(ws)`，兼容 IPv6 回环 `::1` 与 IPv4-mapped `::ffff:127.0.0.1`，地址缺失（反向代理剥离）按不可信处理。策略变为：认证启用→必须有效 token；认证未启用→仅放行回环对端，非回环一律拒绝并给出「请先启用 JWT 认证」的日志。`terminal_allow_unauthenticated` 语义收窄为**只能收紧不能放宽**（置 false 时连本机也要求鉴权），无法再用来给局域网开后门。模块 docstring 与函数 docstring 同步勘误。

边界实测（9/9 正确）：`127.0.0.1`/`::1`/`::ffff:127.0.0.1`/`localhost` → 放行；`192.168.1.7`/`10.0.0.5`/`0.0.0.0`/`None`/`""` → 拒绝。

#### 13.3.6 验证与产物

| 项目 | 结果 |
|---|---|
| vitest | **19 文件 / 160 用例全通过** |
| `npm run build` | exit 0，7 个 view chunk 齐全（ChatView 1,001.84 kB），945 ms |
| 构建产物断言 | `route-leave` **0 次**（竞态窗口消失）、`route-enter` 3 次、`quick-btn` **0 次**（死码清除）、`.router-wrapper` **1 次**（两处收敛为一处）|
| 重新打包 | `Seed.exe` 69.14 MB、`SeedBackend.exe` 69.07 MB，均为 2026-08-28 01:06:11 |
| 内置前端一致性 | `frontend/dist/index.html` 与 `dist/Seed/_internal/frontend/dist/index.html` SHA256 同为 `DF4069E4…790D`，**MATCH** |
| 打包内 CSS 断言 | `route-leave=0`、`quick-btn=0`、`.router-wrapper=1`，与源码构建一致 |

`python scripts/release.py` 在本机以 exit 1 结束，但**打包主体成功**：前端一致性字节门禁通过两次、PyInstaller 报告 `Build complete!`、后处理已复制 `user_data/` 与 `security/`。失败只在最后 `_verify_artifacts()` 检查 `dist/SeedSetup.exe`——本机没有 `makensis`，NSIS 步骤被跳过而验证仍要求安装包。

> **本条已过时（2026-08-28，§13.8）**：当时的处置是「本机执行必须加 `--skip-nsis`」，即用人的记忆绕过脚本缺陷；该缺陷已在 §13.8 修掉——`build_nsis()` 改为回传「本机是否真的编译出安装包」这一事实供验证消费，因此现在**不需要任何标志**，`python scripts/release.py` 在无 makensis 的机器上也会如实以 exit 0 结束。

改动文件（6 个）：`frontend/src/App.vue`、`frontend/src/assets/app.css`、`frontend/src/assets/styles/shell.css`、`frontend/src/views/WorkspaceView.vue`、`api/routes_agent_workspace.py`、`api/routes_terminal.py`。

**方法论沉淀**：本轮三个问题全部是「上一轮修复引入的新缺陷」，且两个的初诊都是错的（白屏一度归因 CSS 重复、对话框一度归因未创建）。有效手段是**实证否定**而非推理：CSS 用导入序+级联语义排除、对话框用子进程阻塞时长与文件系统副作用反证「已创建」、鉴权用穷举边界地址验证。凡涉及「看不见」的失败（透明元素、隐藏窗口），必须找到能观测的侧信道。

遗留：终端默认 shell 仍是 cmd.exe；侧边栏搜索框尚未接线为会话过滤；P6 真实语言器官接入仍是消除乱码的唯一根治路径。

### 13.4 `ChatView` chunk 拆分与「假测试」收敛（2026-08-28）

起因是 §13.3 的遗留项「`ChatView` chunk 已达 1 MB 需拆分」。拆分本身顺利，但过程中**顺带查出三个一直存在、且被测试全绿掩盖的生产 bug**。这一轮的真实价值在后者。

#### 13.4.1 体积构成：唯一大头是 highlight.js 全量语法

`frontend/src/` 中只有 `composables/useMarkdown.js` 一处 `import hljs from 'highlight.js'`，该默认入口静态注册全部语法。实测 `highlight.js` 目录构成（此前笔记里的「384 种语言」是勘误）：

| 事实 | 数值 |
|---|---|
| 真实语法数 | **192** |
| 名称总数（含别名） | 371（其中纯别名 179） |
| `es/languages/` 文件数 | 384 = 192 真实语法 + 192 个 `<name>.js.js` 兼容 shim |
| 单个语法极端体积 | `mathematica` 109,852 B（约 107 KB，此前白吃） |

`highlight.js` 的 `exports` 条件映射在 `import` 条件下会把 `./lib/core` 解析到 `es/core.js`、`./lib/languages/*` 解析到 `es/languages/*.js`，所以 bare specifier 可直接用于 ESM 按需加载。

#### 13.4.2 方案：core + 192 语法按需加载，而非静态挑选子集

选择上限更高的方案：只静态引入 `highlight.js/lib/core`，**192 种语法一个不减**，全部改为按需动态加载。被否决的方案是「静态注册十几种常用语法」——那是能力降级。

三个必须解决的技术前提，均以探针实测确认（探针用完即删）：

1. **别名要在加载前就能解析。** hljs 的别名（`py`→`python`）只在语法注册后才生效，而 fence 标记恰恰在注册前到达；未注册语言调 `hljs.highlight` 会**抛异常**。因此在构建期生成 `frontend/src/composables/hljsAliases.js`（179 条纯别名，3,318 B）做前置映射。
2. **`renderMarkdown` 不能变成 async。** 模板里是 `v-html="renderMarkdown(msg.content)"` 同步调用。解法：模块级 `grammarVersion = ref(0)`，`renderMarkdown` 内 `void grammarVersion.value` 读一次让 Vue 记为渲染依赖，语法到位后自增即触发重渲染 —— `ChatView.vue` **零改动**。
3. **shim 不能进产物。** 曾断言「用精确路径 `import(\`…/${name}.js\`)` 就不会展开 shim」，**实测被证伪**：Rollup 把 `${name}` 当 `[^/]*` 匹配，`python-repl.js` 同样命中，产出 192 个永不加载的死 chunk（54,201 B）。改用 `import.meta.glob` 的否定模式 `'!…/*.js.js'` 后归零；该 record 同时充当白名单，未知语言名可同步拒绝而不发起失败请求。

#### 13.4.3 三个被「假测试」掩盖的生产 bug

`src/__tests__/useMarkdown.test.js` 原本**复制了一份自己的 `parseMessageContent`**（注释写明 "Simplified … for testing core logic"），从不 import 真模块。11 个断言长期全绿，而真实模块同时坏着三处：

| bug | 现象 | 根因 |
|---|---|---|
| 代码块内容全丢 | 每个 fence 渲染成 `[object Object]`，语言标签恒为 `text` | marked v13+ 把 `renderer.code` 改为接收 **token 对象**，代码仍用 v12 的位置参数 `code(code, lang)`，模板插值把对象字符串化 |
| 答案标签残留 | 「思考过程：…\n最终答案：…」的正文开头留着「最终答案：」 | 清理正则 `/^(?:最终)?(?:回答\|答案)[：:]/` 缺少前导 `\s*`，而 lookahead 未消费的 `\n` 就在开头；且只认中文前缀，`Answer:`/`Final:` 完全没清 |
| 复制按钮点不动 | 代码块「📋 复制」按钮全程无响应 | 钩子用 `data-action="copy-code"`，但 `purifyConfig` 里 `ALLOW_DATA_ATTR: false`，DOMPurify 每次都把它剥掉，而事件委托正以该属性为选择器 |

三处均已修复。第三处按收敛原则**不放宽 `ALLOW_DATA_ATTR`**（那是有意的安全姿态），而是删掉多余标记、统一以已存活的 `.code-copy-btn` 类为锚点，取文本改走 `.code-block-wrapper > pre`。

测试文件重写为 import 真模块，用例 11 → 27，新增覆盖：token 对象渲染器（正文非 `[object Object]`、语言标签正确、无语言回落 `text`）、未加载语法路径的 HTML 转义、未知 fence 标记的安全降级、`<推理>` 中文标签、中文标签不残留、markdown 标题分隔、`formatDuration`。**复制按钮的断言不测字符串而测契约**：把 HTML 塞进真实 DOM，验证委托选择器 `.code-copy-btn` 能选中、且 `.code-block-wrapper > pre` 可达。

#### 13.4.4 验证与产物

| 项目 | 结果 |
|---|---|
| `ChatView` chunk | **1,001.84 kB → 132.55 kB（−86.8%）**，gzip 44.03 kB；500 kB 警告消失 |
| chunk 总数 / 死 shim | 207 个 / **0 个**（修正前为 399 / 192） |
| 语法体外置校验 | ChatView 内 `LiveScript`/`Mathematica`/`PostgreSQL` 命中 **0** 次；对应语法各自成独立 chunk（`python` 3,258 B、`typescript` 7,590 B、`rust` 2,667 B、`x86asm` 19,007 B、`mathematica` 109,852 B）|
| vitest | **19 文件 / 175 用例全通过**（原 160，useMarkdown 由 11 假断言 → 27 真断言）|
| 重新打包 | `Seed.exe` 69.14 MB、`SeedBackend.exe` 69.07 MB（位于 `dist/Seed/`，**不在** `_internal/`），均为 2026-08-28 02:25:01 |
| 内置前端一致性 | `frontend/dist/index.html` 与 `dist/Seed/_internal/frontend/dist/index.html` SHA256 同为 `5BE51F49FE10…`，**MATCH**（release.py 内部亦自校两次）|
| 打包内产物断言 | 包内 `ChatView-C5zeGOa0.js` 129.45 KB、207 个 js chunk、`data-action` **0** 次 |

`scripts/release.py --skip-nsis` 以 exit 1 结束，但**构建完整成功**（输出 "Seed v1.6.0 构建完成"、总大小 1273.7 MB、前端一致性两次通过）。失败来自沙箱拦截 PyInstaller 分析阶段对 `Python312/Lib/**/__pycache__/*.pyc.<pid>` 临时文件的写入，与构建结果无关 —— 这是本机第二类已知的「假红」（第一类见 §13.3.6 的 NSIS 缺失）。**判定打包成功必须看产物本身，不能看 release.py 的退出码。**

改动文件（3 个）：`frontend/src/composables/useMarkdown.js`、`frontend/src/composables/hljsAliases.js`（新增，构建期生成）、`frontend/src/__tests__/useMarkdown.test.js`。提交 `e564029`（含本节 plans，4 文件 +445/−92）。构建产物 `frontend/dist/`、`dist/` 均在 `.gitignore` 内，不入库。

**方法论沉淀**：本轮最大教训不是体积，而是**「测试复制被测逻辑」等于零覆盖且伪装成满覆盖**——`[object Object]` 这种毁灭级 bug 与 160 全绿共存了很久。凡是 `__tests__` 里出现被测函数的本地副本（尤其带 "Simplified"/"for testing" 字样），一律视为门禁缺口。其次，本轮我两次把推断写进代码注释（shim 是否展开、`${name + '.js'}` 是否改变模式），两次都靠实测产物计数才被纠正：**注释里不能出现未实测的构建行为断言**。

遗留：终端默认 shell 仍是 cmd.exe；侧边栏搜索框尚未接线为会话过滤；P6 真实语言器官接入仍是消除乱码的唯一根治路径。

### 13.5 `hljsAliases.js` 再生成门禁（2026-08-28）

清偿 §13.4 的遗留项。别名表是构建期固化的静态产物，其生成器随探针一起删掉了：highlight.js 升级新增语言（如 `zig`）后，用户写 ```` ```zig ```` 会**静默退化成无高亮纯文本，而全部测试依然全绿**——正是 §13.4 刚付过学费的失效模式。

#### 13.5.1 四个必须先实测的前提

动手前用两个一次性探针把设计前提全部测出来，因为 §13.4 证明了「把推断写进代码」会被产物打脸：

| 实测结论 | 数据 | 对设计的约束 |
| --- | --- | --- |
| `highlight.js` 与 `highlight.js/lib/core` **是同一个单例** | `before=0 → afterFullImport=192`，`sameObj=true`（`lib/index.js` 对 `require('./core')` 注册 192 个语法；`lib/core.js:2589` 导出单例） | 门禁不能与 `useMarkdown.test.js` 同文件，且生成时必须用 `newInstance()` 隔离，否则会把 192 个语法注册进被测单例，让按需加载测试失去意义 |
| 别名存在**真实冲突** | `ls`: lasso vs livescript；`ml`: ocaml vs sml。上游 `registerAliases` 直接覆盖（后注册者胜），实测 `lasso@93 < livescript@100` → `ls`=LiveScript，`ocaml@124 < sml@162` → `ml`=SML | 必须解析 `lib/index.js` 复刻注册顺序 |
| 注册顺序**不是文件名字典序** | `REG_ORDER isSorted=false` | 现有表与字典序恰好吻合是**巧合**；自行排序在未来某次升级会静默产生错误归属 |
| `spec.name` 是**展示名而非键** | 几乎每个语法都不同（`1c` → `1C:Enterprise`），且 `python-repl` **根本没有 `name` 字段** | 只能以文件名为唯一标识，反查校验不可用展示名 |

行尾另需注意：无 `.gitattributes`、`core.autocrlf=true`、源文件磁盘上 100% CRLF，故生成器写 LF、`--check` 比较前归一化 CRLF→LF，避免制造全文件 diff。

#### 13.5.2 结构：生成逻辑只存在一份

`frontend/scripts/gen-hljs-aliases.mjs` 既是 CLI 也是模块，导出 `buildAliasMap`/`renderModule`/`listGrammarFiles`/`resolveHljsRoot`，由 `src/__tests__/hljsAliases.test.js` 直接 import。**测试不重算任何别名**——若在测试里重写一份「简化版」推导，就是 §13.4「假测试」的原样重犯。CLI 入口用 `import.meta.url === pathToFileURL(process.argv[1]).href` 守卫，保证被 vitest import 时不触发写盘。

新增 `npm run gen:aliases`（重生成）与 `npm run check:aliases`（`--check`，过期即 exit 1）。

#### 13.5.3 验证：门禁必须被证明能变红

| 验证项 | 结果 |
| --- | --- |
| 生成结果与既有表一致 | 重新生成后 `git diff` 仅 **2 insertions**（新增的「生成勿改」头注释），179 条别名逐字节复现 |
| 幂等 | 连续两次生成后 `--check` 均 exit 0 |
| **注入缺失别名** | 删掉 `yml` → 精确报 `missing: ['yml']`，红 |
| **注入错误归属** | `py: 'ruby'` → 报 `py: 文件=ruby 实际=python`，红 |
| 全量回归 | **20 文件 / 181 用例**全绿（原 19/175，新增 1 文件 6 用例）|
| 脚本未泄漏进前端产物 | `dist/assets/*.js` 中 `gen-hljs-aliases`/`node:fs`/`node:module` 均 **none**；唯一命中的 `registerLanguage(` 是 `useMarkdown.js` 的按需注册。ChatView 仍 132.55 kB |

**方法论沉淀**：一次红色验证比十次绿色更有信息量。首轮篡改时键集断言没红，我一度以为是漏判，实测发现是 PowerShell `Set-Content -NoNewline` 注入了 BOM 且使 CRLF 正则失效、`yml` 实际未被删除——**验证手段本身也会假**，改用 node 精确改写后立刻变红。因此「新增门禁」的完成标准不是它通过，而是它在人为破坏下必定失败。

改动文件（3 个）：`frontend/scripts/gen-hljs-aliases.mjs`（新增）、`frontend/src/__tests__/hljsAliases.test.js`（新增）、`frontend/package.json`（两个 script）；`frontend/src/composables/hljsAliases.js` 补头注释 2 行。临时探针 `probe_alias.mjs`/`probe_alias2.mjs` 已删除，不留残留。提交 `a2a4488`（含本节 plans，5 文件 +258/−2）。

### 13.6 把别名门禁接到发版必经路径（2026-08-28）

13.5 建成的门禁只在 `npm test` 时生效，而真正会出事的场景恰好绕过它：升级 highlight.js 后直接打包发版。门禁存在但不在关键路径上，等于不存在。§13.2 已把打包收敛到 `scripts/release.py` 这唯一入口，因此把检查插在它的构建之前。

`scripts/release.py` 新增 `check_generated_sources()`，执行 `npm run check:aliases`，失败即 `sys.exit(1)` 并打印修复命令。三个不显然的设计决策：

| 决策 | 理由 |
| --- | --- |
| **不受 `--skip-frontend` 影响** | 跳过的是构建，不是校验。`frontend/dist` 正是由这份可能已过期的源码产出的，跳过构建时打进安装包的 dist 同样过期。 |
| **置于 Step 0 清理之前** | 脱同步是必然中止的错误。若先清理再报错，会把上一版可用产物白删一遍。 |
| **`--check-only` 不跑该检查** | `--check-only` 的语义是「验证已有产物」，别名表属于源码而非产物。混进去会模糊两者职责。 |

顺带收敛：Windows 上 `npm` 实为 `npm.cmd`（直接调 `"npm"` 会 WinError 2）这一特例原本要在两处重复，抽成 `_npm()` helper。构建标号随之从 `[1/4]~[4/4]` 统一为 `[1/5]~[5/5]`。

实测（两条路径都验过，不止验绿）：

| 场景 | 结果 |
| --- | --- |
| 正常状态 | `[1/5] 生成式源码同步门禁` → `179 条别名一致`，放行进入后续步骤 |
| 删掉 `yml:` 一行后 `python scripts/release.py --skip-nsis` | `EXIT=1`，`hljsAliases.js 已过期` → `生成式源码已与依赖脱同步，构建中止`；**「清理旧产物」一行从未打印**（本次特意不加 `--no-clean`），证明产物未被删 |
| 恢复后 | `git status` 仅 `M scripts/release.py`，`check:aliases` 转绿，`--check-only` exit 0 |

改动文件（1 个）：`scripts/release.py`（+41/−11）。提交 `878316f`（含本节 plans 与 §14 修订，2 文件 +66/−12）。

### 13.7 解除 CI 下游 job 的 `needs: test` 挟持（2026-08-28）

本轮起点是一个**被推翻的假设**。上一轮收尾时我建议"把 `check:aliases` 与 `npm test` 接进 CI 前端 job"，动手前通读 `ci.yml`（373 行，不做局部读——`needs` 链局部读极易误判）才发现：`build-frontend` 第 204-206 行**早已有 `npx vitest run`**，而 `hljsAliases.test.js` 的断言里就含"磁盘文件与生成器输出逐字节一致（等价于 `--check`）"。即别名门禁自 `a2a4488` 起就已在 CI 生效。实跑 `npx vitest run` 确认收集到该文件（20 文件 / 181 测试全绿，含"别名键集合与当前 highlight.js 完全一致"）。**按收敛原则，不新增任何重复步骤。**

同时排除了第二个疑似风险：`highlight.js` 声明为 `^11.11.1`，但 CI 用 `npm ci` 按 lockfile 装（钉死 11.11.1，与本地 `node_modules` 一致），caret 并非静默漂移口子——真正刷新 lockfile 的时刻（`npm update` / dependabot）会让 vitest 那条断言当场变红，该路径已被封住。

真问题在依赖图上：`build-frontend` 与 `docker-build` 都挂着 `needs: test`，而 §14.8 已实测过其后果——`test` 连红 7 次期间这两个 job 一直是 `skipped` 而非 `failure`，**从未执行**。这与 §13.6 治的是同一个病（门禁不在必经路径上），病灶换到了 CI 的依赖图里：一个只改前端的 PR，若 Python 矩阵因无关原因（含 flaky）变红，eslint / vitest 别名门禁 / npm audit / E2E 全部静默失效一次。

解除依赖前逐条排除了耦合：

| 核查项 | 结论 |
| --- | --- |
| `build-frontend` 是否消费 `test` 的产物 | 否。10 个步骤全部自给（`npm ci` 起链） |
| `e2e/smoke.cjs` 是否需要后端 | 否。只依赖 `vite preview`（`BASE_URL` 默认 5173，CI 传 4173），不访问 `/api/*` |
| `docker-build` 是否消费 `test` 的产物 | 否。镜像构建自带完整依赖安装 |
| 其余 job 的写法 | `startup-smoke`、`test-windows` 本就无 `needs`——只有这两个挂着，不一致本身即线索 |

`docker-build` 一并解除的理由更强：§14.8 记载的两个缺陷（Dockerfile 缺 `data/` 目录、`seed_platform` 未随包安装导致启动 `ModuleNotFoundError`）都是它**独家**发现的，Python 测试矩阵抓不到。把一个具备独立发现能力的 job 挂在另一个 job 之后，等于让这份能力随上游一起失效。

验证（本地解析真实依赖图，而非只验 YAML 能否 parse）：

| 手段 | 结果 |
| --- | --- |
| `yaml.safe_load` 后枚举 `jobs` 的 `needs` | 5 个 job 全部 `needs = None`，依赖图扁平化，无悬空引用 |
| 同时输出各 job 步骤数 | 26 / 10 / 5 / 7 / 8，与改前一致——只删了 `needs` 行，未误伤步骤 |
| 枚举 `build-frontend` 十步的 `run`/`uses` | `npx vitest run` 在位，别名门禁执行点未动 |
| `npx vitest run` 实跑 | 20 文件 / 181 测试通过，`hljsAliases.test.js` 6 测试全绿 |

并发面变化：原先 `test`（2 矩阵）跑完才轮到 2 个下游，现在 5 个 job 立即并发，峰值 7（2+1+1+2+1），远低于公开仓库 20 的并发上限；副作用是反馈更快。

**本轮刻意未做**：`ci.yml` 既无 `concurrency` 也无 `timeout-minutes`（见 §14.14）。二者是真实欠账，但本轮意图是"解除门禁挟持"，混入并发治理会让这次提交不可审计。

改动文件（1 个）：`.github/workflows/ci.yml`（删 2 行 `needs: test`，加 8 行理由注释）。提交 `9dab2e5`（含本节 plans，2 文件 +53/−3）。

### 13.8 知识库白屏根治、jsdom-Blink 门禁与子进程内核级回收（2026-08-28）

本轮由四条客户端反馈驱动，其中一条是**同一现象的第二次上报**——"点击知识库后标签页内容全白屏，这个问题还是没解决"，并附带一条流程质问："为什么不启动开发者模式调试好了再打包"。后者是本轮最有价值的输入：§13.3.2 那次"白屏已修"是在没有真实浏览器控制台的前提下宣布的，用的是推理而非观测，所以修错了层。

**白屏真因（与 §13.3.2 的过渡动画完全无关）**：`FileUploadQueue.vue` 把 `icon` / `uploadIcon` 两个 prop 声明为 `String` 且默认值是 emoji（`📤`），又交给 `<component :is="uploadIcon">`。Vue 对字符串型 `is` 的处理是"解析不到组件就当原生标签",于是执行 `document.createElement('📤')`。Blink 对标签名做严格校验，emoji 不是合法标签名，**同步抛出 `InvalidCharacterError`**；该异常发生在 `keep-alive` / `router-view` 的渲染过程中，导致整棵子树被销毁——表现为点进知识库后**所有**路由都白屏，且不可恢复。修法是把 prop 类型改为 `[Object, Function]`、默认值换成 lucide 组件（`FileText` / `Upload`），并加 `asComponent()` 归一化 computed 兜住历史调用方。

**为什么 181 个前端测试全绿却放过了它**：`jsdom` 的 `createElement` 不校验标签名，`createElement('📤')` 在 jsdom 里合法，在 Blink 里抛异常。这是**环境差异造成的门禁盲区**，不是用例写少了。故新增 `frontend/src/__tests__/setup/blinkDom.js`，在 vitest `setupFiles` 里给 `Document.prototype.createElement` 打上 Blink 同级的标签名正则校验（`/^[A-Za-z][^\0\t\n\f\r >/]*$/`），不合法即抛 `InvalidCharacterError`。配套 4 个用例。**门禁必须能变红**：临时把修复回退，新用例当场以客户端里那条一模一样的错误失败；恢复后 185/185 全绿（181 → 185）。

**另三条反馈**：`WorkspaceView.vue` 去掉「项目文件」文字；托盘通知图标改为 `self.tray.icon()`（原先传 `MessageIcon.Information`，那是系统蓝色 i 图标，与 taiji logo 无关）；`MonacoEditor.vue` 的纯图标保存按钮确认与顶栏「运行/保存」功能重复，按 §13.3.4 同一原则删除视觉层级低的那个。

**验证方式改为真实观测**：QtWebEngine 不支持 Playwright 的 `connectOverCDP`，故用 `QTWEBENGINE_REMOTE_DEBUGGING=9222` + 裸 CDP over WebSocket 驱动。11 次路由跳转 + 3 个知识库标签页逐一断言 `routeError` 与容器内容长度：修复前 `nav-kb` 的 `len: 0`，修复后 `len: 205`，全程零异常零 console error。**这才是"调试好了再打包"该有的证据形态。**

**顺带暴露并根治的隐患：子进程在主进程被强杀后独活占用端口。** 排查白屏时两次遇到"代码改了但行为像旧的"，实测是上一轮的后端 worker（PID 20488、4944）仍在监听 8000，而就绪探测只看 `/api/health` 是否响应、不校验持有者是不是自己的子进程，于是静默接管了陈旧后端。WebSocket 服务（8765）也有同一现象（PID 11636）。清理路径 `_quit() → backend.stop()` 只在优雅退出时执行，强杀/崩溃时根本不跑。**不在 Python 层再加 `try/finally` 或 `atexit`（强杀时同样不执行）**，改用内核级 Windows Job Object：`JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE` 使 Job 句柄随主进程消亡时内核自动终止 Job 内全部子进程，与主进程如何死亡无关；三处 `Popen` 之后一律 `adopt_child()`。另加 `_reap_orphan_listener()` 处理"上一轮遗留的孤儿"，回收判定要求两个条件同时成立——映像名恰为 `SeedBackend.exe`（该名字只存在于本产品包里）且其父进程已不在系统中——避免误杀用户自己的服务或第二个客户端实例；5 个契约测试锁住这条边界（`tests/test_desktop_orphan_reap.py`，本仓库第一个 desktop 层 python 测试）。

**一次假警报及其教训**：强杀主进程后子进程存活，我一度判定 Job Object 未生效。升级排查（读日志 → `IsProcessInJob` 直查成员归属 → `QueryInformationJobObject` 回读 `LimitFlags` 确认 `0x00002000` 排除 ctypes 结构体错位 → 无 shell 中间层的隔离父子实验），逐项证明机制正确。真因是**我的验证方法有缺陷**：按命令行文本匹配挑进程，选中的是 shell 包装层，真正持有 Job 句柄的父进程（由子进程 `ParentProcessId` 反查得到）还活着。改按子进程反查父进程后复验通过。**记入方法论：进程身份不能靠命令行文本匹配确定，必须由子进程的 `ParentProcessId` 反向确认。**

**排查过程中发现并消除的潜伏缺陷**：`desktop/__init__.py` 里有 `from desktop.main import main`。`python -m desktop.main` 时 runpy 先导入 `desktop` 包 → `__init__` 把 `desktop.main` 载入 `sys.modules` → runpy 再把同一份源码作为 `__main__` 执行一遍。症状不只是日志 handler 装两遍导致每行重复（已观测），更严重的是**模块级全局出现两份副本**（包括 `_CHILD_JOB` 这个 Job 句柄本身，日志显示两次 armed），以及 `BackendManager` 类对象重复使 `isinstance` 失效。这正是它让上面那次假警报更难排查的原因。已删除该导入并在 docstring 里记录此陷阱；grep 确认主线无 `from desktop import main` 依赖，打包 spec 用脚本路径而非包导入；重启验证 runpy warning 消失、日志不再重复、Job 只 armed 一次。

**`scripts/release.py` 的自相矛盾**：`build_nsis()` 在 makensis 缺失时打印警告并 `return True`（判为非致命），而 `_verify_artifacts()` 仍按 `--skip-nsis` 标志硬性要求 `dist/SeedSetup.exe` 存在。后果是无 NSIS 环境下一次完全健康的构建必然以「产物验证失败」收尾——这就是 §13.3.6 记的"第一类假红"，当时的处置是"记住要加 `--skip-nsis`"，属于用人的记忆绕过缺陷。本轮按收敛原则改为**事实回传**：`build_nsis()` 返回 `(是否可继续, 是否应产出安装包)`，验证消费后者而非猜标志；`--check-only` 走同一套判定以免两条路径对同一产物给出不同结论；并补 `_find_makensis()` 兼查 NSIS 默认安装位置（NSIS 安装器不写 PATH，只查 PATH 会把"已装"误判成"未装"）。

**仓库卫生**：`.gitignore` 只有 `.codex_tmp/`，匹配不到 `.codex/`（gitignore 无前缀通配语义），而后者含 29 张 QA 截图、若干 CDP 探针，以及**两份活跃 git worktree 副本**（`git worktree list` 确认），副本里有同名的 `plans/ tests/ scripts/ frontend/`，会被仓库级 Grep / ruff / vitest 一并扫到而使统计基数失真。已补 `.codex/` 为**独立一行**（不删 `.codex_tmp/`——两者无覆盖关系，删了会让它重新被跟踪）。worktree 属活跃工作树，须走 `git worktree remove` 而非直接删目录，本轮不动。

验证与产物：

| 手段 | 结果 |
| --- | --- |
| vitest 全量 | 185/185（原 181，新增 4 个 Blink DOM 用例） |
| 门禁变红验证 | 回退 `FileUploadQueue.vue` 修复 → 新用例以 `InvalidCharacterError` 失败 |
| 裸 CDP 实测 | 11 次路由跳转 + 3 个知识库标签页，`routeError` 全为 `null`，零 console error；`nav-kb` len 0 → 205 |
| `tests/test_desktop_orphan_reap.py` | 5 passed |
| Job Object 机制隔离验证 | `LimitFlags = 0x00002000`、`IsProcessInJob(mine) = True`、隔离父进程强杀后子进程 alive = False |
| **打包模式端到端强杀** | 强杀 `Seed.exe`(25308) → `SeedBackend.exe`(25044) alive = False，8000/8765 全部释放 |
| `python scripts/release.py --check-only` | 全绿（修复前必然报 `✗ dist/SeedSetup.exe 不存在`） |
| 打包产物 | `Seed.exe` 72,507,172 B / `SeedBackend.exe` 72,422,700 B，前端一致性校验通过 |

**本轮方法论沉淀**（三条，均已在本轮内被实测检验过）：

1. 宣布"UI 缺陷已修"之前必须有真实浏览器控制台的观测证据；推理修出的是另一个 bug。
2. 单元测试环境（jsdom）与生产环境（Blink）的能力差异本身是门禁盲区，发现一例就要把校验补进 setup 层，而不是只补一个用例。
3. 机制看起来"没生效"时，先怀疑验证手段。本轮"Job Object 失效"与 §13.5 "PowerShell 注入 BOM"是同一类错误的两次发作。

## 13.9 外壳边框收敛为"整体圆角卡片内嵌"、标签页"常驻 + 零动画 + URL 同步"（2026-08-28）

用户对照主流客户端（TRAE/Doubao）截图提出四个互相关联的质疑：(1) 主流客户端是一条外围边框整体包裹、标签页嵌入其中，而本应用是"顶部边框与下方边框分割、两段对不齐"；(2) 标签页切换不如主流客户端丝滑，像"刷新显示"；(3) 这是否也是白屏的原因；(4) 商用前端是不是不用 Vue 这类平台、自己直接写的。四个问题逐一回答并落地实现（用户已确认目标形态：整体圆角卡片内嵌 + 常驻/零动画/URL 同步）。

**(1) 边框分割与 Vue 无关，是"边框所有权"颠倒。** 主流客户端由 shell（外壳）持有唯一边框，内容视图只是填充；本应用反了过来——每个视图自己画 `border-bottom`，且 `view-header` 的 `max-width: 800px` 使"线"的宽度永远取决于各视图内容宽度，全局无法对齐。React/Svelte/原生 DOM 会同样出错，框架无关。**修法**：收敛到 `styles/shell.css` 单一真源——`.app-wrapper` 降级为窗口/背景宿主（保留 `appStore.applyBgImage()` 的挂载点，用 `--bg-base` 暗部营造"卡片内嵌"的亮度差），`.router-wrapper` 成为**全应用唯一外围边框**（`border + border-radius: var(--radius-lg) + box-shadow + margin`），`.sidebar` 变为无边框透明面板；全部 5 处 `.topbar`、3 处 `.tabs`、`.view-header` 的 `border-bottom` 全部移除，并附注释禁止回潮。同时消除三份重复定义的级联债：`.app-wrapper` 原先在 shell.css（flex）/ app.css（grid）/ product.css（background，且最后加载会盖掉亮度差）各一份，`prefers-reduced-motion` 有两份全局块，响应式断点 768/880 冲突——统一为 880 + 560 两级。

**(2) "刷新显示"的根源是动画过多而非少了动画。** 面板用 `v-if` 切换 = 卸载 → 重建 → 重跑 setup → 重取数 → 从 `opacity: 0` 淡入，滚动位置、展开状态、输入内容全丢。主流客户端标签切换是 0ms（VS Code/Chrome 皆如此）——"丝滑"指的就是瞬时，动画只会让它看起来像刷新。

**(3) 白屏的直接成因仍是一行确凿的渲染异常（§13.8 已根治）；但 `v-if` + 淡入确实制造了"白屏易感体质"。** 判别法：量 `container.innerHTML.length`，0 = DOM 被清空（真白屏），非 0 + 透明 = 动画卡住。两者结论完全相反，先量再猜。把面板改为常驻后，任何真实渲染错误都会立即以可见形态暴露，降低再误诊概率。

**(4) 商用客户端没有"不用 Vue"。** TRAE/Doubao、VS Code、Slack、Notion、Linear 全是 Electron + Web 技术栈；VS Code 工作台是手写 TS + 直接 DOM，但它的两个要点（单一自顶向下的布局真源、视图永不销毁）在本项目用 Vue 完整可达——本轮同时落地了这两点。

**实现**：新增 `composables/useTabs.js`（唯一实现，三视图复用）：`activeTab` 写入 `?tab=`（`router.replace` 防污染后退栈），URL → 状态（前进后退/深链/刷新保持），`onActivated` 应对 keep-alive 下 `onMounted` 只触发一次的问题，方向键/Home/End + roving tabindex + `aria-selected/aria-controls/role=tabpanel`（WAI-ARIA tablist 手动激活模式），无 vue-router 环境（单元测试）自动降级为纯状态模式。`KBView`（白屏案发地，三个 `v-if` 面板）`TrainingView`（四个面板）`AgentConfigView`（三个 `v-if` 面板，且原本连 `role="tab"` 都没有）全部改为 `display` 切换；AgentConfig 原先内联在 `@click` 的 `loadInstalled()/loadMarketplace()` 收敛为对 `activeTab` 的 watch，深链直达也能触发加载（比内联点击更高上限）。`AppSidebar.vue` 响应式升级：880px 以下压缩为 56px 图标轨道（`!important` 压内联 `width`，仅此能赢），560px 以下才隐藏。

验证与产物：

| 手段 | 结果 |
| --- | --- |
| vitest 全量 | 185/185 全绿 |
| eslint（改动的 4 个 vue/js 文件） | 0 error 0 warning（`--fix` 属性换行） |
| ruff check（根目录） | All checks passed |
| `npm run build` | ✓ built，无编译级遗漏 |
| 上线请求 | 5 个 `.topbar` + 3 个 `.tabs` + `.view-header` 边框全部收敛到 `.router-wrapper` 唯一外围边框 |

**实机观测（QtWebEngine 裸 CDP @9222，source 模式 `python -m desktop.main`）**：

| 测量项 | 结果 |
| --- | --- |
| 三个视图 9 次标签切换耗时 | 3–22ms，全部同帧内完成（DNS 语义上的 0ms；旧 v-if+fade 需等整帧动画） |
| 面板显隐 | 恒为「1 显示 + N 隐藏」，DOM 常驻（切换无白屏帧、无重建） |
| URL 同步 | 每次切换 `#/kb/train/agent` 后附 `?tab=`，前进后退/刷新可还原 |
| 深链直达 | `location.hash='#/train?tab=dataset'` → 面板直接是高亮「数据集」 |
| keep-alive 折返保持 | KB 选「检索配置」→ 切 agent → 折返 KB，标签仍为「检索配置」 |
| 边框形态 | `.router-wrapper` 恒为 `1px solid` + `19.2px` 圆角；`.topbar/.tabs/.view-header` 边框全为 0 |
| 异常监控 | 全程 `Runtime.exceptionThrown` 与 console error 零触发 |
| 截图（card-kb.png） | 外壳灰底 + 大圆角卡片内嵌 + sidebar 独立间隙，内外部无任何分割边框错位 |

## 13.10 标题栏所有权移交前端、系统通知署名收敛为 AppUserModelID（2026-08-28）

用户附四张截图提出两条诉求：(1) 系统通知左上角不是应用 logo，而是紫色占位方块加字面量 `Seed.exe`——"用正确的 logo，或者直接不给这个弹窗提示"；(2) 顶部栏与下方"还不是一体的"，参照图 3（TRAE/Doubao）与图 4（Codex）——顶部一条与主体是同一个连续平面。

**(1) 通知署名不由 QIcon 决定，改 icon 永远无效。** Windows 通知左上角的归属槽取的是**进程的 AppUserModelID**；未声明时系统回退到 exe 身份，于是渲染占位方块 + `Seed.exe`。§13.8 那次"改为 `self.tray.icon()`"只换了通知体内的图标，署名槽根本不在该 API 的作用域内，所以用户看到的问题原封不动。**修法采纳"两条都做"的高上限组合**：一是声明 `APP_USER_MODEL_ID = "Seed.Desktop.Shell"`，经 `ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID` 在**任何窗口创建之前**设置（`desktop/main.py` 与 `api/run_app.py` 两个入口各一处，就在 `QApplication` 构造前）；二是**彻底删除 `closeEvent` 里的 `tray_icon.showMessage(...)` 气泡**——"已最小化到托盘"这类信息价值极低，而托盘图标本身就是可见反馈，改由 tooltip「Seed — 双击图标恢复窗口」承载。两者叠加后，即便未来别处需要弹通知，署名也已经是正确的应用身份。

**(2) "不一体"的物理根因：两个渲染平面各自画自己的背景，永远拼不成一个面。** 旧实现的顶部条是 Qt 控件（`QWidget` + `QHBoxLayout` + `QLabel` + 三个 `QPushButton`，36px，QSS 上色），下方才是 `QWebEngineView`；两者是不同的绘制宿主，无论把颜色调得多接近，接缝处的抗锯齿、DPI 缩放取整和主题时序差都会显形——这也是 §13.1 那条"1s 轮询 `data-theme` 重设 QSS"补丁的存在理由，它本身就是这个错误架构的并发症。**修法是把标题栏所有权整体移交前端**：中央区域改为 `QWebEngineView` 独占，标题栏成为 DOM 的一部分，与 sidebar 共享同一个 `.app-wrapper` 背景宿主，且中间**不存在任何 border**——由于 `.sidebar` 本就透明地坐在 `.app-wrapper` 上，一个同样透明的 `.app-titlebar` 自动就是同一个平面，"一体化"从此是结构保证而非调色结果。

**实现**：

- **窗口控制桥**：新增 `_WindowBridge(QObject)`，以 `pyqtSlot` 暴露 `minimize` / `toggleMaximize` / `close` / `startDrag` / `isMaximized`，经 `QWebChannel` 注册为 `seedWindow`。拖拽走 `windowHandle().startSystemMove()`（系统级移动，比手算 `globalPos` 增量更跟手且不丢焦点）。
- **客户端库注入**：页面由 http 提供，无法引用 `qrc:`，故把 `:/qtwebchannel/qwebchannel.js` 读出后注册为 `QWebEngineScript`（`DocumentCreation` + `MainWorld`，`seed_qwebchannel` 幂等去重），前端拿到的是原生 `QWebChannel` 全局。
- **最大化状态通道**：`resizeEvent` / `changeEvent` → `_sync_window_state()` 把 `data-maximized` 写到 `document.documentElement`；CSS 用 `:root[data-maximized='true']` 抹掉圆角与边框，Vue 侧用 `MutationObserver` 切换按钮图标。单向数据流，无轮询。
- **外框所有权也一并下移**：`.app-wrapper` 接管 `border: 1px solid var(--border)` + `border-radius: 18px`，Qt 只保留 `QRegion` 圆角裁切（`WINDOW_RADIUS = 18`，与 CSS 同值）防白直角露出。
- **前端**：新增 `components/AppTitlebar.vue`（无 `<style>` 块，样式全部落在 `shell.css` 单一真源），`App.vue` 增 `.app-body` 与 `sidebarCollapsed`（持久化到 `taiji_sidebar_collapsed`）。标题栏刻意**不放搜索框**——`AppSidebar.vue` 已持有 `.search-field` 与全局 Ctrl/⌘K，再加一个就是第二套入口。

**旧机制清理（收敛而非叠加）**：`desktop/main.py` 与 `api/run_app.py` 两处各删除 `_titlebar_qss`、`_window_frame_qss`、`_apply_titlebar_theme`、`_sync_titlebar_theme`、`_build_titlebar`、`_titlebar_mouse_press`、`_titlebar_double_click` 共 7 个方法（约 120 行/处）、§13.1 的主题轮询定时器、以及随之失效的 `QVBoxLayout/QHBoxLayout/QLabel/QPushButton/QWidget` 导入。净机制数下降。另确认 `desktop/seed.spec` 打包的是 `desktop/main.py`（`run_app.py` 当时的文件头"打包环境"注释是错的，已在 §13.10.2 改掉），但后者仍是可运行入口，按"清理旧的以免残留干扰"原则同步收敛，否则就留下第二套标题栏；`seed.spec` 的 `hiddenimports` 补 `PyQt6.QtWebChannel`。

**实机观测（QtWebEngine 裸 CDP @9222，先调试后打包）**：

| 测量项 | 结果 |
| --- | --- |
| 桥与客户端库 | `hasQt/hasTransport/hasQWebChannel` 全 true；`objects: ["seedWindow"]`，五个 slot 全部可达 |
| 一体化度量 | `.app-titlebar` 背景 `rgba(0,0,0,0)`、`border-bottom: 0px none`、高 40；`gap_bar_to_body = 0` |
| 接缝取色 | 跨越标题栏下沿的 4 个采样点（y=35/40/42/47）**实际背景全为 `rgb(241,243,245)`**——同一个平面，无缝 |
| 侧边栏收起 | false→true→false 双向可用，`taiji_sidebar_collapsed` 正确持久化 |
| 最大化联动 | `data-maximized` true 时圆角 `0px`/边框 `0px`/2560×1392；还原为 false/`18px`/`1px`/1280×800 |
| 截图（topleft/topright 2× 裁切） | 左上「收起按钮 → 太极 logo → Seed/在线」同一背景连续过渡、零分隔线；右上三个窗口按钮直接坐在窗口背景上，下方才是内容卡片圆角——与参考图 3/4 结构一致 |
| ruff / vitest 边界 | `ruff check` All passed；`tests/seed/test_platform_boundary.py` 9 passed |

**顺带收敛的悬空引用**：`frontend/public/` 下 `logo.svg` / `favicon.svg` / `icons.svg` 三个文件在前几轮已被删除，但 `frontend/index.html:5` 仍在 `<link rel="icon" type="image/svg+xml" href="/logo.svg?v=ink-20260624">` 引用 `logo.svg`，构成一个每次加载都 404 的悬空引用。已删除该行——`favicon.ico`（同文件第 6 行，文件实际存在）单独就足以承担 favicon 职责，且 taiji logo 在应用内由 `logo-taiji-ink.jpg` 提供。

### 13.10.1 冻结产物复验（`python scripts/release.py`，2026-08-28）

上面那张表是**源码模式**的观测，不足以结案：本轮新增的 `PyQt6.QtWebChannel` 是运行时依赖，而 `qwebchannel.js` 是**编译进 Qt 的 qrc 资源、不是磁盘文件**，`Get-ChildItem` 永远找不到它；更要紧的是 `_inject_webchannel_client()` 读取失败只打一条 `logger.warning`（"前端标题栏将退化为无窗口控制"），`_set_windows_app_identity()` 同样只 warning——**两条都是静默降级**，应用照样启动、外观几乎正常。所以必须在真实 exe 里把资源读出来才算证明。

打包产物：`dist/Seed/Seed.exe` 69.2 MB + `dist/Seed/SeedBackend.exe` 69.1 MB。经 `QTWEBENGINE_REMOTE_DEBUGGING=9333` 启动后裸 CDP 复验：

| 复验项 | 结果 |
| --- | --- |
| 依赖收集 | `_internal/PyQt6/QtWebChannel.pyd`、`Qt6/bin/Qt6WebChannel.dll`、`Qt6WebChannelQuick.dll`、`Qt6/qml/QtWebChannel/webchannelquickplugin.dll` 四件齐备 |
| qrc 资源实读（充分条件） | 冻结进程内 `typeof window.QWebChannel === 'function'` 为 true，`window.qt.webChannelTransport` 存在，`Object.keys(channel.objects) === ["seedWindow"]` |
| 一体化度量 | 与源码模式**逐项相同**：标题栏 `rgba(0,0,0,0)` / `border-bottom: 0px none` / 高 40 / `gap_bar_to_body = 0`；wrapper `1px solid rgb(231,234,239)` + `18px` |
| 接缝取色 | y=30/38/40 命中 `.titlebar-drag`、y=42/50 命中 `.sidebar-header`，五点背景**全为 `rgba(0,0,0,0)`**，统一落在 `.app-wrapper` 的 `rgb(241,243,245)` 上——纵向连续，无缝 |
| 最大化往返 | `false/18px/1px/1280×800` → `true/0px/0px/2560×1392` → 还原完全一致 |
| 冻结日志负向证据 | `dist/Seed/logs/desktop_main.log` 无 `qwebchannel.js 资源读取失败`、无 `AppUserModelID 设置失败`，两条降级分支均未走到 |
| 子进程生命周期 | `Stop-Process Seed` 后 `SeedBackend` 同步消失，job object 的 kill-on-close 在打包产物中同样生效 |
| 截图（full/topleft/topright） | 整窗一张外框，标题栏与 sidebar 同底、零分隔线，白色对话区圆角内嵌——与参考图 3/4 形态一致 |

**AUMID 的验证边界（记录方法论，避免下次误判）**：Win32 **没有**读取其他进程 explicit AUMID 的 API。`SHGetPropertyStoreForWindow` + `PKEY_AppUserModel_ID` 读的是**窗口级**属性存储，而 `SetCurrentProcessExplicitAppUserModelID` 设的是**进程级**值——实测目标窗口（`hwnd=2950018`，标题 `Seed - AI 生命体`）返回 `VT_EMPTY`，这是**预期结果**，不代表修复失效，通知系统会回退到进程级值。因此改用三条合证：(1) 同段代码在进程内 set/read-back，`E_FAIL` → `S_OK` → `'Seed.Desktop.Shell'`，机制有效；(2) 调用点在 `QApplication(sys.argv)` 构造**之前**（`desktop/main.py` L773 紧邻 L774），满足"任何窗口创建前"的时序要求；(3) 冻结日志无失败 warning。另：`HKCU:\...\Notifications\Settings` 下**不存在** `*Seed*` 项，这恰好侧面印证"直接不给弹窗提示"那一半生效了——`showMessage` 已删，进程从未发通知，系统自然不会建项。

**一个会反复踩的构建陷阱**：`python scripts/release.py 2>&1 | Tee-Object -FilePath build_release.log` 返回 exit 1，但日志尾部是 `Seed v1.6.0 构建完成`。原因是 PyInstaller 把全部 INFO 写 stderr，PowerShell 在管道中遇到原生命令写 stderr 会抛 `NativeCommandError`，**掩盖 python 的真实退出码**；日志里的 `ModuleNotFoundError: No module named 'tensorboard'` 也只是 PyInstaller 的可选导入探测，无害。构建是否成功的权威判据是脚本自己的 `python scripts/release.py --check-only`（本轮返回 0，含"前端一致性校验通过（源码 dist = 客户端内置 dist）"），而不是 shell 的 `$LASTEXITCODE`。

### 13.10.2 入口所有权收敛：消除"`run_app.py` 是打包入口"的错误共识（2026-08-28）

§13.10.1 里记了一句「`api/run_app.py` 文件头『打包环境』注释已过时」，本轮把它真正改掉。这不是措辞问题：两个文件头**互相印证**了一个反的事实——`api/run_app.py` 自称 `[打包入口] PyInstaller 桌面客户端`，`desktop/main.py` 自称 `[产品入口] … 开发环境版本` 并写着「api/run_app.py：打包环境」「未来计划：合并为一个入口，**以 api/run_app.py 为基础**」。任何人（包括我自己）照此改桌面行为，都会把改动落在一个**既不被打包、也不被版本同步覆盖**的文件上，然后打出旧行为的包。

判定入口身份用的是证据而不是注释：

| 证据 | 结论 |
| --- | --- |
| `desktop/seed.spec` L60 `a_main = Analysis([str(ROOT / "desktop" / "main.py")], …)` | 打包入口是 `desktop/main.py`，产物 `dist/Seed/Seed.exe`（spec 自己的文件头 L4 早就写对了） |
| `scripts/sync_version.py` 的同步清单 | 只覆盖 `frontend/package.json` / `desktop/installer.nsi` / `desktop/main.py` / `desktop/loading.html`，**没有 `api/run_app.py`** |
| grep `setApplicationVersion|SeedDesktop/|1\.\d+\.\d+` on `api/run_app.py` | 零命中。它连版本号都报不出来——真产品入口不可能如此 |
| glob `docs/ENTRYPOINTS.md` | **文件不存在**。两处文件头都在把读者指向一份不存在的文档 |

最后一条把问题性质升级了：这与上一轮删掉的 `logo.svg` 是同一类**悬空引用**，只是指向文档而非资源。已一并清除，仓库内（plans 外）`ENTRYPOINTS` 命中归零。

改法上没有写"以后再合并"这种会再次腐烂的承诺，而是各自写死当前事实：`desktop/main.py` 改为 `[唯一产品入口] … 开发与打包共用`，直接点名 `seed.spec` 的 `a_main` 与产物路径，顺手把功能描述校正到现状（标题栏由前端 DOM 承载、不发气泡通知、进程内 WebSocket、job object）；`api/run_app.py` 改为 `[历史入口·非打包]`，开头即**否定式断言**「**本文件不是打包入口。**」，并说明**保留理由**（依赖自检自动安装、`HotUpdateImporter` 热更新——这两项 main.py 没有），避免下次有人把"过时"误读为"可删"。

门禁与一处判断边界：`tests/seed/test_platform_boundary.py` 9 passed（该测试只断言 `run_app.py` 的 import 边界与 `CORE_DEPENDENCIES`/`transformers` 两个字面量不出现，文件头改写安全）、`ruff check` All passed、`py_compile` 0，并用 AST `get_docstring()` 反读确认两个 docstring 仍是模块首语句、内容正确。`black --check` 报这两个文件 `would reformat`——`git stash` 后复跑**基线同样 exit 1、同样这两个文件**，且 git 提示 `LF will be replaced by CRLF`，属既有换行符交互，非本轮引入，不扩大范围。

**遗留观察（本轮不动，记录以免丢失）**：(1) `test_desktop_entrypoint_keeps_transformer_dependencies_opt_in` 函数名断言的却是 `api/run_app.py`，是同一错误共识的命名残留；(2) 上述 black/CRLF 基线。（原第 (3) 条「本地与 `origin/main` 分叉」已由 §13.10.3 解决，故删除。）

### 13.10.3 分叉归零：桌面壳层三提交 rebase 到 P6 provider 四提交之上（2026-08-28）

`git stash` 时暴露出本地与 `origin/main` 分叉（本地 3 / 远端 4）。这是当时唯一的阻塞项，理由不是"分叉本身难看"，而是**远端那 4 个提交内容未知，一旦触碰 `desktop/` 或 `frontend/`，§13.10.1 的冻结产物验证就不再代表合并后的代码**——而那份验证是前两轮的全部结论依据。

**先按文件求交集再决定策略，不靠提交信息猜。** 提交标题（`provider registry rotation` 等）看起来与桌面无关，但"看起来无关"不是判据。用 `merge-base` 分别取两侧 `diff --name-only` 后做 `Compare-Object -IncludeEqual -ExcludeDifferent`：

| | 文件 |
| --- | --- |
| 本地 11 个 | `desktop/main.py`、`desktop/seed.spec`、`api/run_app.py`、`frontend/` 6 个、路线图 |
| 远端 11 个 | `seed/config.py`、`seed/language_provider.py`、`taiji/{__init__,adapter,language_organ}.py`、`api/seed_runtime.py`、2 个测试、`plans/README.md`、`scripts/training/…`、路线图 |
| **交集** | **只有路线图 1 个** |

且两侧在路线图内的 hunk 也不重叠：本地 `@@ -889,0 +890,74 @@`（§13.10 区），远端 `@@ -1346,3 +1346,22 @@`（§16 P6 区）。**远端 4 个提交完全不触碰 `desktop/` 与 `frontend/`**，因此冻结验证结论无需重新打包复验——这是本轮最重要的判定，它把"必须重跑 69 MB 打包"降为"跑静态门禁即可"。

**选 rebase 而非 merge**：本地 3 个提交是尚未共享的线性叙事（标题栏移交 → 冻结复验 → 入口收敛），三者有明确因果顺序；merge 会插入一个无信息量的 merge commit 并把这条因果链打散在图里。rebase 前先建 `backup/pre-rebase-20260828` 分支作为可回退锚点。结果：`Rebasing (1/3)(2/3)(3/3)` **零冲突**，`9fc7ecf/e7680c2/ccf0167` → `f82b169/0b59da2/ad47075`。

**合并后复验（不是"应该没问题"）**：

| 项 | 结果 |
| --- | --- |
| 路线图两侧内容并存 | §13.10.1/§13.10.2 在 922/943 行，远端 P6 provider 段在 1431-1439 行，共 1441 行，无一方被吞 |
| 远端文件完整落地 | `git diff --name-only origin/main HEAD` 在 push 前为 11（即本地三提交的改动），无远端文件丢失 |
| 联合测试 | `test_platform_boundary` + `test_language_provider_runtime` + `test_p6_language_organ_boundary` = **34 passed**（我方 9 + 远端 25，与远端提交声明的"25 passed"吻合） |
| 静态 | `ruff check desktop/ api/ seed/ taiji/` All passed；`py_compile` 0 |
| 跨层风险点 | 远端改了 `api/seed_runtime.py`（后端运行时，桌面壳层唯一可能被跨越隔离影响的地方），`importlib` 实导 `api.seed_runtime`/`seed.language_provider`/`taiji.language_organ` 三模块 → `backend_import_ok` |
| 产物一致性 | `python scripts/release.py --check-only` → 0，含"前端一致性校验通过（源码 dist = 客户端内置 dist）"，证明既有打包产物在合并态仍有效 |

`git push origin main` → `634c15a..ad47075`，`git status -sb` 回到无 ahead/behind 的 `## main...origin/main`。另：`git rebase` 与 `git push` 都再次触发 PowerShell 的 `NativeCommandError`（git 把进度写 stderr），真实 `$LASTEXITCODE` 均为 0——与 §13.10.1 记录的 PyInstaller 陷阱同一根因，**凡在 PowerShell 里判断原生命令成败，都必须看退出码而不是有无 stderr 输出**。
