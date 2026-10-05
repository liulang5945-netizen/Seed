# Seed / Taiji 长期工作记忆

> 只留跨轮次仍有效的契约/纪律/判据；细节见 §5 与 `docs/*_RULES.md`。整理 2026-09-22。

## 1 架构契约（机器强制）

- `taiji/` 自足：禁 import seed / seed_platform / neuroplex / transformers（含传递性）；AST 级强制于
  `tests/taiji_native/test_{architecture,naming_boundary}_contract.py`。
- `instruments/` → taiji 单向（仅 `content_digest`）；`taiji/` 对 instruments 零引用。语义 encoder 的
  embedder 必须显式注入（fail closed）。`neuroplex/` 不得 import seed/taiji。
- 红线：`taiji-document-embedder-v1` 的 payload 格式与 digest 锚不得变更。

## 2 环境硬约束

- Python 用 `C:/Users/23747/AppData/Local/Programs/Python/Python312/python.exe`（managed 3.13 无 torch/ruff）。
- **bash 部分命令缺失**：实测**有** `du`/`find`/`ls`（`/usr/bin/`），别再假设全无；仍按缺处理
  cat/grep/head/tail/mkdir/rm。文件用 Read/Write/Edit/Glob/Grep，目录/过滤用 `python -c`
  或 `| python.exe -c "…"`。**管道里缺失命令会 SIGPIPE 杀掉上游 Python**。
- **统计大目录体积一律用 `du`，禁用 Python `os.walk`**：harness 的 node_modules 用 Python 遍历
  36 min 未出结果，`du -d 1` 只用 56 s（差一个数量级以上）。
- **反引号在任何 shell 引号里都会被命令替换**（同一天内已踩 **8** 次，规则写着照样犯）⇒
  **多行文本一律先 Write 成文件再合并，禁止写进 `python -c` 的字符串**（连"追加一段 Markdown"都会中招：
  标识符被吃空、三引号串被打断）。严禁 heredoc。
- **跑 `tests/` 与任何大批删除的构建步骤必须 `CODEBUDDY_SAFE_DELETE_ENABLED=0`**：批删守卫劫持
  `Path.unlink`/`os.remove` ⇒ 中止或长时间无进展，**极易误诊为磁盘 I/O 卡死**。
  全量 pytest ~15 min 会 SIGTERM。读仓库文件的 gate 必须显式传
  `SeedRuntime.load(..., workspace_root=PROJECT_ROOT)`。
- **harness/taiji-harness 前端**：`pnpm` **不在 PATH**，任何脚本前须
  `export PATH="$PWD/node_modules/.bin:$PATH"`。**仓库根 oxlint 在本机起不来**
  （`tsgolint` spawn 报 `os error 231`＝Windows 命名管道耗尽，与代码无关）；可复现替代＝
  `pnpm exec oxlint -c <临时cfg> <包>`，临时配置**只写** `{"categories":{"correctness":"error"}}`
  （写 eslint 风格规则名报 `not found in plugin 'eslint'`，加 `plugins` 报 `Unknown plugin`）。
  `tsc --noEmit` 要加 `--listFiles` 自证编译到了目标文件，否则「0 error」可能只是没编译到。
- ⚠️ **node 同步子进程创建被系统层拦截（2026-10-03 实测）**：`execFileSync`/`spawnSync`
  一律 `EBUSY`，**对任意目标都一样**（git / cmd.exe / node 自身 / 甚至一个 .txt），
  且与 node 版本（22 与 24）、worker 线程、`dangerouslyDisableSandbox`、
  清空 `NODE_OPTIONS` **全都无关**；而**异步 `spawn` 正常**。所以 pnpm、vitest
  （threads 池）能跑，**任何需要同步 spawn 的脚本必然失败** —— 桌面打包
  （`package:win:x64:unsigned` → `readDesktopBuildCommit` 用 `execFileSync('git')`）
  在助手环境内**做不了**，且同步 spawn 点遍布仓库（release/process、native build、
  primary-runtime prepare 等几十处），改代码绕不过去。**结论：打包/发布类重活交
  自己在普通 cmd/PowerShell 里跑**（脚本模板见
  `C:/Users/23747/Desktop/build-life-update.cmd`：PATH 补 node + node_modules/.bin、
  `CODEBUDDY_SAFE_DELETE_ENABLED=0`、`DSH_DESKTOP_NPM_REGISTRY` 指国内镜像、
  `DSH_DESKTOP_BUILD_VERSION`、`DSH_CLIENT_COMMIT_HASH` 钉死 HEAD）。
  诊断口诀：先测 `node -e "execFileSync('cmd.exe',['/c','echo',''])"`，
  EBUSY 即环境级拦截而非代码问题。
- **桌面打包三个必踩点**（2026-10-03 实测，各踩过一次）：① **build version 只能走 CLI
  `--build-version`，环境变量 `DSH_DESKTOP_BUILD_VERSION` 无效**（`resolveRequestedBuildVersion`
  只读 `invocation.requestedBuildVersion`）⇒ 不传就退回 productVersion `0.1.7-alpha.1`，
  **比已装的 `.2026MMDD.N` 更旧** ⇒ 客户端报"已是最新"；**绝不能靠改 exe 文件名冒充**，
  那会造成"文件名新、内部版本旧 ⇒ 反复提示更新"的死循环。② **`--build-version auto`
  在本机不可用**：`apps/desktop/.env.windows` 设了 `DSH_DESKTOP_AUTO_UPDATE_ENV=test` +
  `DOWNLOAD_TEST_ORIGIN`，于是 `remoteVersions` 必查云端 bucket，而
  `DOWNLOAD_TEST_COS_BUCKET` 未配 ⇒ 抛 `must be set to a non-empty value`，**走不到
  本地扫描那条降级路径**（判据是 `=== undefined`，空串也会进去）。③ **末步
  `smoke-packaged-runtime` 必红＝既存 R7**（LibreOffice xlsx→PDF，
  `loadComponentFromURL returned an empty reference`，docx 过）⇒ 打包退出码 1 是常态，
  **发布不能以退出码为门槛**，改以"产物文件名带当日 `YYYYMMDD`"为准。
  打包实测约 11 分钟（非 30–60）。

## 3 工作流与文档纪律

预注册 → 实现 → 门禁 → 报告 → 提交；负结果**如实落账，绝不改绿**。文档四类：冻结判据/预注册与
冻结证据**只追加**（新结论写新预注册；被覆盖则恢复归档版+另存）；导航/状态文档（03 的"唯一下一步"）
**必须改**；原文写错**改+注明原值**。

- 归属证明用引用图检查；回归测试**双向钉住**；**守卫必须红/绿各跑一次证明能响**（否则是装饰）。
  ⚠️ 推论：**恒真式断言不构成验证**。踩过的实例：想用 vitest 断言
  `typeof css.organ === 'string'` 证明样式表没坏 ⇒ **注入未闭合块后仍然绿**
  （Vite 的 CSS 处理对未闭合块宽容）⇒ 该探针是装饰、已删。**探测目标要选对：
  "类名能解析" ≠ "CSS 语法正确"**。本机可用的 CSS 语法验证＝**postcss**
  （`node_modules/.pnpm/postcss@<ver>/node_modules/postcss`，**未 hoist，`require('postcss')`
  会 MODULE_NOT_FOUND**）；**lightningcss 的 Node 绑定在本机初始化失败，不可用**。
- 测试改完读回全文（Edit 可能匹配错缩进而静默失效）；同一文件多处替换**必须串行**。探针用毕即删。
- 链接前缀：`plans/reference/*`→`../../`；`plans/active/roadmap/*`→`../../../`；改完跑校验。
- 正结果先问"相邻设置能否复现"；交互效应声明前做规模扫描。提交信息写 `.git/COMMIT_MSG_*` 再 `-F`。
- ⚠️ **本仓常有并行会话**：结论标取证时刻、提交前重跑 `git log`；别清理未跟踪内容。

## 4 可复用判据

- **边际退化**：不同输入生成同一字节串且等于 train 众数 ⇒ 学到边际分布，前向/机制层解释全不成立。
- **预算分层**：否决只在「对照臂在该预算下已能产出非零 exact」时有效；"改动生效但两臂逐位相同"
  ⇒ 是**预算不足以表达**，不是假设被否决。
- **抖动 vs 系统性**：1/N seeds ⇒ 抖动可放行；N/N 且机制读数退化 ⇒ 系统性不得放行。
- **写"无差别"类阈值前先定双侧还是单侧**：把 |Δ| 写成 Δ 会把"显著更差"误判成满足（已踩两次）。
- **长跑/批处理**：幂等重跑要用**多臂**测试证明；时刻进文件名换安全字符（`:` 在 Windows 非法）；
  driver 分臂记状态；**报速率报中位窗口不报累计**；空窗口指标写 `null`。
- **`counterfactual._run_scripted` 不能判合同合法性**：绕过 `policy_for` 的拦截。
- **写进代码的预期值必须有可否决通路**（否则是装饰）。
- **仓库卫生**（全文 `docs/REPO_HYGIENE_RULES.md`）：判泄漏比**在位值 vs 历史 blob 指纹**（≠数提交、
  ≠看 ignore）；目录规则不覆盖子文件。历史重写用 `git clone --mirror` 镜像隔离、**永不**原地做；
  `filter-repo` 不重写自定义 ref 与远端 `refs/pull/*`。
- **桌面/前端工具链与打包**：见 `docs/DESKTOP_AUTOMATION_PITFALLS.md`。
- **UI/CSS 三条静默失效**（写对也看不出错的那类，务必查）：① **伪元素不能嵌套**，
  `::before::before` 被解析器**直接丢弃、不报错** ⇒ 骨架/装饰必用真实空节点承载；
  ② **Chromium 的 `<summary>` 不暴露 `aria-expanded`**，折叠态只能
  `details[open] > .summary::after` 驱动，写 `[aria-expanded='true']` 是永不命中的死规则；
  ③ **整体替换 CSS 必须做类名双向核对**（TSX 引用集 A / CSS 定义集 B，报 A−B 与 B−A），
  且正则要匹配**嵌套选择器**（`.actions .dangerAction` 不在行首，按 `^\s*\.` 扫会误报缺失）。
- **`<summary>` 内不得有直接文本子节点**：会让 `getByText(区块名)` 由单点命中变多点命中
  直接叫红；同理 armed 确认清单**整份只渲染一次**，不要每行渲染。
- **双产出文档互相引用、不互相复制**（CSS 归 A、DOM 归 B，附录写分工）：两份可抄的 CSS
  迟早写歪，而交叉复核能抓到写的人不会回头自查的规范边界（本轮即靠此抓到二级伪元素）。
- **字级分裂**：同一页面并存两套小字号时，用户说的"挤"往往不是那一块的问题，而是
  **旁边那块已经放宽、它没跟上**。分档判据＝"这是要读的正文/一个可点的标题" vs
  "这是一个标签/徽标/表格单元"；后者（标签/仪表/徽标/表格）就该更小更密。
- **别把 `<ul>` 改 flex 来加行距**：flex 父容器会把 `li` blockify，**项目符号整体消失**。
  用 `.x li + li { margin-top: 4px }` 代替。

## 5 当前状态与归档索引

- **产品默认基座 `checkpoints/seed_beta.pt`**（16M tick，来源登记 v2）；DEBT-I9 未结项。H 阈值绑
  `(设备,链路,checkpoint)`。**M5 限定退出已获批准**（2026-09-20），**R2 语言能力是其显式排除项**。
- **R2**：受控重训三臂各 16M（M1/M2 见合同）。**逐区审计：槽结构只在区 0（16M 仍 0.729/0.709），区1/2 从未有过**；
  T5 读出只吃区0 ⇒ cue 水平↑(0.42→0.74) 但训练期游走依旧；T6 CAP 四模型全 0；
  **归纳探针实测：S+S 首字/整串命中 0/30×3模型 ⇒ 无归纳/复制机制** ⇒
  **CAP D+E=0/36 是架构缺归纳/复制的结构性缺口**，与预算/读法/写入强度/cue 可分离性全部无关；
  修复需架构级立项（所有者定）。**长跑必须保号存档**；**提>几小时的实验前先过零训练诊断**。见 `M5_R2_*`。
- **产品侧工程支线**（不入研究主线）：前端 TS 地基 + Electron 壳 + 打包链路，见
  `plans/reference/FRONTEND_TS_VS_HARNESS_ADOPTION_DECISION_BRIEF_20260921.md`。
- **Seed 品牌 logo 已定稿（2026-09-28，owner 认可）**：AI 原创重绘（水滴壳+满冠树+Seed 衬线署名，
  构图对齐 owner 参考图），画布 `E:/Seed/seed-logo.miora`，定稿文件
  `E:/Seed/seed-logo_assets/1044f28f-miora_edit_image-1790567714143-0-bb84fe254933.png`（1024×1024）。
  色板：深墨绿 #144235 系 / 清新绿 #799D54 系 / 象牙底。**教训：owner 给参考图并反复指向时，直接
  按参考 AI 生成成品，勿走几何拆解**。后续物料以此为锚点派生。
  **外壳比例定稿（同日，owner 选方案3）+ 图标细节四修（同日续）**：原泪滴壳 358×552（宽高比 0.649）装进
  正方形图标左右留白过多 ⇒ **保留树的像素不动**、只按原轮廓同族曲线重画外壳为宽高比 **0.94** 的饱满蛋形
  （顶部尖头保留）。随后按 owner 四点反馈再修：图标母版 **`MARK_RATIO` 0.72→0.84**（记号太小，且一颗
  "坐标漏乘超采样倍数"的假杂点曾把包围盒撑大、令图标里记号缩小偏心）；**树根用同色补块桥接进壳底笔画**
  （原来树干平截、与壳底留白缝）；**树自动适配放大**到"不碰壳的最大尺寸"（scale 1.24／横向 1.18，实测
  越界 0px、最小间隙 23px）；**配色按亮度重映射变嫩**（结构深绿 `#124A38`＋嫩叶高光 `#AAD66A`）。重生成器
  `E:/Seed/design/round_shell.py`；图标包 `E:/Seed/design/icons/` 按新母版重出（`build_icons.py`）。
  **应用内接线已收官（2026-10-02）**：记号像素迹线矢量化（欧拉边消费＋闭合环 RDP 最远点拆分，彩色层必须 evenodd）换入 FishLogo／BrandWordmark／web 与官网 favicon／wordmark／桌面三套 icon.svg；主题 `--dsw-static-seed-*` 色阶重锚（300=#AAD66A、800=#124A38，amber 删）；安装器横幅／卸载器侧栏／skill 徽章／icon*.png 由母版合成（`design/build_brand_assets.py`）；身份句按「Seed=客户端、Taiji=模型」落地（700cee0da）。读数：primitives+brand-official 56 文件 1198 绿；sidebar 快照重录 5/5；app-boot+ui-conversation 5928 过/1 环境红。原尚欠行作废：应用内记号（`ui-primitives/src/FishLogo.tsx` 等 v5「圆＋树」几何记号）与桌面/favicon/安装器资产曾是旧记号，
  与新锚点不一致，接线未做。**
- 回退备份 `E:/Seed-backup-{git,secrets}-20260919`；未确认前别跑 `git gc`/`prune`。
- **仓库体积账（2026-09-27 只读扫描）**：全仓 ~121 GB。零风险可删 1.86 GB（`scripts/clean_worktree.py`
  dry-run，程序自建）；harness 构建/依赖缓存 7.3 GB（可重建，重编译成本）；实验产物 output+outputs
  ≈51 GB（**删=重跑 GPU，需 owner 裁决**）；data 60 GB 不删。
  **git 不需要瘦身**（已入库仅 213.78 MiB、无 .pt）。详见
  `plans/reference/REPO_CLEANUP_ASSESSMENT_20260927.md`。
