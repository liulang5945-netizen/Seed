# Seed 开发计划索引（2026-09-25 v3，编号体系规范化）

> 用途：**唯一的计划入口**。v3 新增**编号体系**（一个前缀一个意义，§0）；
> v2 修正层级：**主线是 M 系列（里程碑）**，A/B 是**支线**（欠账修复 / 架构债），与主线**并行、互不阻塞**。
> 历史文档**冻结、不重命名**（改名会断日志与提交的链接），用 §4 的**新编号 ↔ 旧文件名**映射检索。
> 冻结判据/预注册/冻结证据只追加；本索引随状态**就地修订**。

## 0. 编号体系（**一个前缀一个意义**；新增文档强制，存量不动只映射）

**格式**：`<类别>-<归属>-<序号>[_<主题>][.<日期>]`
—— **类别**说"这是什么性质的文件"，**归属**说"属于哪条线/哪个里程碑"，**序号**是该归属内的顺序。

| 类别前缀 | 意义（只放这一类东西） | 旧名对应 |
|---|---|---|
| `PLAN` | 计划/提案/路线（**非判决**，可修订） | *_PROPOSAL、*_BRIEF、*_MECHANISM_DESIGN |
| `SPEC` | 判据/预注册/合同（**冻结件，只追加**） | *_PREREG、*_CONTRACT、*_BUDGET |
| `EXP` | 实验执行与判决（训练臂/评测/计分的**落盘件**） | eval_taiji_*、train_taiji_*、reports/*.json 判决件 |
| `DIAG` | 诊断与归因（探针/审计，**非判决**） | probe_taiji_*、audit_taiji_*、diag_* |
| `FIX` | 缺陷修复与守卫 | *_fixes、tests/*_contract |
| `MS` | 里程碑判定/批准 | M5_EXIT_APPROVAL |
| `CONV` | 口径/约定 | metrics_conventions、FOLDER_STRUCTURE_RULES、REPO_HYGIENE_RULES |

**里程碑**：`M1..M6` 不变（大阶段主线，见 §1）。
**归属**：`R2`（M5 排除项调查，已收束）/ `A`（修复线，见 §2）/ `B`（架构债线，见 §3）/ `M6`（产品交付，见 §1）。
**脚本前缀**（代码件，另有约定）：`train_/eval_/verify_/build_/probe_/diag_/audit_/score_` 说"脚本干什么"，
与文档类别前缀**互不混用**。

**存量处置**：历史文件**冻结、不重命名**（改名断日志与提交链接）；§4 给**旧名 → 新编号**的映射；
**新增文档一律按本表**（例：T8 若今天新建应为 `PLAN-A-08_mechanism-design`）。

## 1. 主线：M 系列（里程碑）

| M | 名称 | 状态 |
|---|---|---|
| M1–M4 | 早期里程碑 | ✅ 已完成（详见 01_SCOPE_AND_PHASES） |
| **M5** | 限定退出 | ✅ **2026-09-20 批准落盘**（[批准书](../reference/M5_EXIT_APPROVAL_20260920.md)） |
| **M6** | **产品交付（当前大阶段）** | 🔄 **界面载体＝`taiji-harness/`**（dsh 整仓 fork）；原 `desktop-electron/`＋`frontend/`＋`desktop/` 与发布链 **2026-09-23 随裁决 6 退役**（G3 通过后删旧线）。里程碑：**G1 ✓ G2 ✓（含 G2b–G2h 修复轮）G3 ✓**（`dsh-llm-taiji` 硬接 8000，真机回合由 Taiji 应答）→ **G4 进行中**（六步全落地；判据①②④⑤⑥ 已取到本轮读数（2026-09-26：①四连 rc 全 0、②③ 43 叶 40 绿、⑤ 真机 UI 三取数、⑥ lint 与基线一致），**判据③ 真机训练回合挂账待裁定**）→ **G5 交付**（web 先行，客户端版后做；**2026-09-26 已出只读就绪清点 [G5 就绪清点](../reference/TAIJI_G5_READINESS_20260926.md)**：打包机制存在但未实跑；装机默认 provider 为 `deepseek-official/deepseek-flash`（`packages/bundle/base/cordis.patch.yml:82-86`）而**非 Taiji**，改它属产品默认变更；**2026-09-26 这条已在本 fork 真机复验为"路由有服务"**（跑上游已有 lane `apps/web/tests/shipped-composition.e2e.ts` 的 keyless 模式：`currentSelection()` 命中该默认值＋`providerRetryPolicy('deepseek-official')` 取到值＝注册确实存在；首启动无凭据时 DeepSeek 行显示“API 密钥缺失”且密钥框可输入，已同日从 ARIA 金样取得表层读数；同批 8 条 keyless 用例红的原因是首启动“添加一个 API Key”卡片不再出现，最可能因为 Taiji 免凭据路由在场（08 §6 ⑪，因果待判别实验）），同一条 lane 首跑另出一条宿主 shell 的环境性红（08 §6 ⑩）；我方六包只在 `base`＋`web-app` 装配里，四个 bundle 为 0；**但"桌面走另一条链"这个推断已被更正**——`apps/desktop/src/main.ts:502` 服务的是 `@taiji/dsh-web-frontend`，而它就是本仓 `apps/web`，所以桌面渲染的正是带我方行的那份装配；未测的问题换成"打包暂存有没有把我方六包放进 `resources.dsh/node_modules`"。另：`build:desktop` 与 `package:desktop:dir` 已实跑；**2026-09-26 下午续**：web 表层 lane 从零覆盖推到可分层当门（helper 真因＝点错元素，H3t；42 条失败已分堆：18 纯金样差异／6 纯超时／5 两者／7 其他），但**那份 33/42 是在过期 client 产物上取的、不合格**，补跑根 `pnpm run build` 后需在新鲜 build 上重跑才算基线；新增裁定项 **R5**（交付面三处上游品牌名，默认工作区目录那处已查实为一行、不需迁移）；`pnpm run typecheck` 现为 0 错误。待裁仍为 **D2／R4／R5／金样三件事**＋重录（`corepack pnpm` 可起钉住的 pnpm@11.7.0，"本机无 pnpm"作废）——前者 **rc=0**，后者**在拉任何二进制之前**先撞本地 `.env.windows`（R4 的产品决定），⇒ 待裁三条现为 **D3／D2／R4**，原先那条"打包在哪台机器实跑"不是被裁定消掉的，是被一次实跑消掉的） |

### M6 开放项（2026-09-26 就地修订；细节与读数在 03 §5.7 的 Taiji Harness 活动卡，交付就绪与裁定单在 [TAIJI_G5_READINESS_20260926](../reference/TAIJI_G5_READINESS_20260926.md)）

> **2026-09-27 裁定状态**：七项 owner 裁定经弹窗全批（回执在 G5 §8 顶部）。执行状态：**D3 已落地**（默认 provider→taiji-local，keyless 四条 overlay 钉回 DeepSeek 前提，13 用例全绿）；**R4 已落值**（`.env.windows`＝com.taiji.harness＋无自动更新声明，unsigned 打包推进至外部下载阶段，揪出并修复 ui-life 版本欠账）；**金样 normalize→refresh 已执行**（(丙)三件平台处置＋plugin-manager {{home}} 破案＝ariaSnapshot 反斜杠翻倍；8 金样保留 2 份污染回退；验证批 23 红｜50 过——9 lane 脱红）；**判据③ 已收官**（六步全通，G4 完成）；**C6 P1 已实现收官**（适配器＋白名单＋no_prose＋生产者/消费者全链：16 条能力折进 consolidated 语料、native_trainable、测试 18 绿）；**R5 已落地**（三处品牌名改 taiji-harness/taiji-harness-acp，断言测试与 README 中英同批，pairing 1100 对一致）；**D2 四刀实现＋产物级四连验证**（wheelhouse 115 轮子/离线安装/幂等/产物内 host 读取形状，unsigned 产物实出；设计=TAIJI_D2_BACKEND_CHANNEL_DESIGN_20260927.md：wheelhouse/离线安装器＋DesktopBackendHost 生命周期＋P1-①…④）；回放件＝暂缓。六条已登记红全部判读（08 ⑲：server-restart＝D3 涟漪已修、goal-bar 负载加固、stats 两因含 H3w 已修、chat/steering＝⑭ 类不计门、menu 多一行＝新裁定）——**新增待裁 R6（引用枚举含空白自动会话）／R7（packaged office 冒烟首达红，unsigned 产物已实出、打包门红在此）／R8（真启动打包产物＝打不开，`ERR_MODULE_NOT_FOUND: @taiji/cordis`；见下方第 10 条与 G5 §8／§9）**。**R8 已裁＝乙并已落地**（外壳自包含，`alwaysBundle: [/^@taiji\//]`；构建层＋系统级均验证通过——沙箱外重打包出干净产物、真启动不再崩、host 拉起；详第 10 条／G5 §10／08 ㉓）。**R7 同轮收窄为只卡 xlsx**（packaged office 冒烟 `docx` 已过）。**品牌收口已落地**（owner 裁「甲＋改叫 Seed＋尽量去 DeepSeek」）：客户端记号统一为新的「种子→小苗」、产品名→`Seed`、DeepSeek 身份残留剥离（保留模型 provider／第三方包名）——详 08 ㉕；**尚欠**：根快照重录＋README 配对指纹重录＋agent 身份（system-prompt）那条线未动＋产物改名待非沙箱复跑验证。**新增待裁 R9**（全新装机首屏被「登录/API Key」欢迎窗挡住，与 D3 免凭据默认相冲突；见第 11 条与 G5 §8／08 ㉔）——**同日已裁＝甲并落地**：客户端**首启动直进工作区**、首页再无账号/API Key 门（owner：「客户端只是用来装载 taiji，登录界面也不需要」）。**品牌记号同日再改**：owner 反馈"太丑／就要一棵树"，定稿为**极简单色"一棵树"**（三瓣树冠＋锥形树干，非零填充），矢量与栅格资产全部重出。 **再续（同日深夜，owner 给 10 张参考＋三点指令）**：**记号 v5＝「圆（种子）里一棵树」**（琥珀圆 `#E8A33C`＋绿树 `#3F8F45`；in-app 用 evenodd 把树从圆里抠出，彩色资产两段 path）；**主题换 Seed 配色**（新增 `--dsw-static-seed-*` 色阶，品牌 token 指绿，`--dsw-static-deepseek-*` 整套删净）；**DeepSeek 登录那套彻底删除**（桌面欢迎/登录窗＋`ui-settings-account` 整包＋相关 IPC/preload/词条/装配面，启动语言偏好改走 Host 现成 RPC）——**模型供应商与相关包全部保留**。读数：目标改动文件 5 文件/100 全绿；apps/desktop 1048 过/9 红（全为既存+环境）；客户端 7073 过/7 红（5 个套件为既存 roster 环境红）；ui-sidebar 48/48 绿。**尚欠**：`pnpm-lock.yaml` 未重生成、desktop README 欢迎窗段落待删并重录配对。详 08 ㉕⑨。**本机验证**（`node node_modules/vitest/vitest.mjs` 直跑）：客户端 81 文件／1654 用例全绿；desktop 1093 过／8 红＝7 条既存红（`upload-with-credentials`，未触碰）+1 条并行 flake（`installed-update-packaging` 单独跑全绿）；并顺带补掉 `app.getPath` 的既存 mock 缺口（两个 spec 转绿）。

1. ~~**G4 判据③**（面板点真训练→进度流→停止→检查点新增复验；再真续训、真激活各一回合）~~ ⇒ **2026-09-27 真机回合收官（03 §5.7）**：六步全通——启动训练（面板转训练中）→进度流接通→停止（回空闲）→新检查点 `seed_native.pt` 复验→真续训（产物 `resumed_seed_native.pt`）→真激活（「再次点击确认」机制，active 由 seed_beta.pt 翻转）。**G4 六判据①②③④⑤⑥全数有真机读数 ⇒ G4 完成**。
2. ~~**未闭合 I**：模型面板缺 Taiji 组 ⇒ 默认模型不可用、composer 被挡、真机 UI 回合取不到~~ ⇒ **2026-09-25 已定位、修复并产品级复验**：根因是就绪只在 load 与 `loader/volatile-update` 采样一次，运行时冷启动晚于 harness 启动即**永久**不注册（`listProviders()` 只列有 adapter 的 provider，而 `buildModelCatalog` 只读它）；改为按 `readinessPollMs`（默认 5s）在插件存活期内再探测，单元与真组合红绿各跑。**真机红绿（同一 web 实例、未重启）**：路由指向空端口启动 ⇒ 模型面无 Taiji 组、composer 停在「当前模型不可用」；假运行时就绪 ⇒ `group Taiji（本地运行时）` 自行出现且 radio `checked`。判据⑤ 的另两半（启动即就绪时 `taijiAtBoot=true`、**发一个普通对话回合成功**「已完成工作 用时 11 秒／1 轮 1 步」）同批复验 ⇒ C2 批「真机 UI 回合未取得」补上（详见 03 §5.7）。
3. ~~**i18n pairing 全仓欠账**（约 252 对 out-of-sync，G2 rename 改了两侧 `.md` 未重录 `.i18n.yaml`）~~ ⇒ **2026-09-25 全部销账**：按裁定分两批「先核对后重录」（第一批 `docs/subsystems` 19 对＝`f6f9eb6a`；第二批余 233 对＝`5d7ed3e4`）。核对判据＝逐对取「该对 `.i18n.yaml` 最后一次被确认的提交」到 HEAD 的**两侧** diff，断言每处改动只是 `@deepseek-ai/`→`@taiji/` 更名、且两侧更名次数逐对相等；**251 对纯机械、1 对（`CONTRIBUTING` 的 slogan 改写）逐字人工读过** ⇒ 零文档正文改动，全仓门转绿（**1099 对全一致**）。
4. ~~**6 个在飞文件**~~ **已查清为行尾幻影（2026-09-25 实测）**：`life-context` ×4、`session-memory-taiji` ×2 在 `git status` 显示 `M`，而 `git diff HEAD --numstat` 为空 ⇒ 与 HEAD 零内容差（`core.autocrlf=true` 所致）；**无待裁定的在飞改动，升级重放不被它阻塞**（08 §6）。
5. **既存红门**：~~`verify-concrete-terms` 3 处 `provenance` 命中~~ **2026-09-25 已清、该门转绿**（改为点名实际字段 `count`/`revision`/`source`/`owner`，提交 `9a4be7ab`）；~~余 `verify-plugin-packages` 1 处（产物在他人包，未处置）~~ ⇒ **2026-09-26 已清、该门转绿**（`fe1dc376`）：落后的其实是 agent-preset 的组合参考包表，重生成后差异**恰为我方新增六包各一行**——是我方 G3/G4 漏跑生成器的欠账，该生成器已补进 08 §2 第 6 步与 §5 判据②。
6. ~~**J（2026-09-25 新发现）**：native 链路收到的 `prompt` 不是用户提问~~ ⇒ **同日已钉死并修复**：agent-loop 的 runtime-context 快照是 durable **user** 消息（`source.kind='runtime-context'`，`agent-loop/src/runtime-context.ts:14/20`）且按设计追加在本轮**之后**，而 `llm-taiji/src/chat.ts` 取「最后一条 user-role 消息」当运行时唯一的 `prompt` ⇒ 用户提问被降级进 `history`。修法＝取最后一条**属用户本人输入**的 user 消息（不带 source 或 `source.kind==='user'`，口径同 `memory-context/src/index.ts:95`），全为 harness 自有时**兜底**取最后一条 user-role（辅助调用 `session-title` 依赖该兜底，否则发空 prompt）。取证＝8098 转发代理抓真实请求体（零改仓、零插桩）：修前 `prompt` 496 字符快照／history 3 对，修后 `prompt` 17 字符提问／history 2 对；红绿＋真机各一次，提交 `9155fca0`。**连带提醒 A 线自查**：此前凡经 web 回合取过的读数，其 prompt 都不是题面（直接 curl runtime 的读数不受影响）。
7. **doc-sync 批余 5 条既存红（2026-09-25 逐条定性，均非本轮引入；**2026-09-26 收为 1 条**）**：①`verify-dependency-catalog`＝**跨平台排序假红**（本机重生成后与已提交版**字节数相同、561 包集合与逐字段全同、按名排序后整份逐字相同**；差异只是 `localeCompare` 在 darwin/node 24.19 与 win32/node 24.15 下的条目顺序）⇒ 重生成产物已回滚，要真绿须把比较器改为与 locale 无关（上游脚本改动＋一次性重排，**需裁定**）——**此判断已过时：2026-09-26 直接改掉比较器（4 处 `localeCompare`→码点序，登记为 08 的 H3n）并重生成产物，门 rc=0；实测产物与 darwin 旧版是同一批 7312 行的重排、spec 不锁 locale 序，故"需裁定"这一项从裁定单上划掉**；②③`verify-persistence-catalog` 与 `verify-persistence-changes` 是**同一根因**（2026-09-26 用门自己的分类器实测；**更正此前归给 H3c 的判断**——门只追踪 `SessionHeader`／`JsonlHeaderLine`／`SessionEventEnvelope`＋57 个 `event:*`，服务接口不在其中）：真凶＝**H1c/H1d** 的 `life-context`／`memory-context` 各自 declare-merge 新 `MessageSourceMap` source kind，**加宽了 durable 会话事件里被持久化的 `source` 联合** ⇒ 8 处变化／5 个 root，其中 **7 处 `requiresVersionBump=true`**（另 1 处 `event:subagent/catalog.data` 疑上游漂移）。当前 `SessionHeader.version`=4 与 `docs/persistence-changes/finalized/` 的已接受基线 4 相等 ⇒ 门要求升到 **5**。**合规处置只有一条（此判断已于 2026-09-26 被实测推翻，见本条末尾）**：bump 到 5 ＋ `persistence-changes --record <id> --decision … --prose FILE` 立承认记录（中英散文＋schema 快照）＋ 重生成 `gen-persistence-catalog`；「留在 4 版声称兼容」门本身不允许 ⇒ **需所有者裁定**（格式版本牵动已发布日志的读取与迁移：不认识某 source kind 的构建会**拒读日志**，除非事件带 `ignorable: true`）——**括号里这句是错的，本轮（2026-09-26）读到真实校验点后更正**：`packages/core/session/src/index.ts:352-357` 对 `source` 只做**结构**校验（必须是对象且 `kind` 为非空字符串），**不枚举已知 kind** ⇒ 未知 `source.kind` 不会被拒读；"拒读未知"发生在**未知事件 type** 那一层，那才需要 `ignorable: true`。这条错判曾让"升格式 5"看起来比实际危险，实际上是反向也错：升版会让新构建读不了旧档。。另注：这条门属 doc-sync 批，而聚合入口需 pnpm、本机跑不了 ⇒ **被环境遮蔽了约两个月**；④~~`verify-archived-agent-notes`＝fork 布局假红~~ **2026-09-25 已修、门转绿**（`ls-tree` 的 pathspec 相对 cwd 而 `git show <ref>:<path>` 相对仓库根，改为按 `git rev-parse --show-prefix` 分别拼；`1917 frozen artifact(s)／6 kind(s)`，无真实违规；上游布局下前缀为空 ⇒ 零行为变化；提交 `7b6687eb`，登记为 08 的 **H3k**）；⑤~~`verify-plugin-packages` 1 处（产物在他人包）~~ **2026-09-26 已清、且原归属判断是错的**：落后的是 agent-preset 的组合参考包表，重生成后差异**恰为我方 G3/G4 新增六包各一行** ⇒ 我方漏跑 `gen-plugin-packages`（提交 `fe1dc376`）；该生成器已补进 08 的 M3 清单与 §5 判据②（此前不在其中，与 H3j 同型的清单盲区）。**doc-sync 批的既存红至此只剩一条**（`dependency-catalog` 跨平台排序假红）。另：聚合入口 `run-gates.ts` 要求经 pnpm 调用而本机 pnpm 不在 PATH，但**叶子本身只是 `tsx <脚本>`** ⇒ 2026-09-26 以 `node node_modules/tsx/dist/cli.mjs …` 直跑此前记为「跑不了」的三叶：**`docs:build` 与 `verify-doc-site-fragments` 本就绿**（rc=0／4920 片段可解析），**`doc-typecheck` 曾红 5 处 TS2307、当日修根因转绿**（＝H1a 作用域改名漏扫未归档 notes 的 `ts` 围栏；10 文件两侧同改＋重记 5 对配对指纹，提交 `d25c8301`，登记为 08 的 **H3l**）。⇒ 「两叶未跑」这条欠账消掉，**同日全量重跑 43 叶＝39 绿／4 红**（红＝`dependency-catalog` locale 假红 ＋ 持久化两条等格式 5 裁定 ＋ `docs-site-projection` 里一条本机不可建符号链接的用例，`fs.symlinkSync` 同机探测亦 EPERM）。重跑另抓到**一条我方真实欠账并已修**：`doc-standard-tests` 报我方包 README 不合规——全仓 626 份包 README 里 7 份不合规、**全部落在 G3/G4 六包内**（3 份 EN 缺 `### Dev Note`，4 份 zh 用英文标题或写作「开发者备注」而门要求「开发备注」），按 `life-controller` 形状补齐＋两侧逐行等长＋zh 内链改指 `.zh.md`＋重记 4 对 pairing ⇒ 该叶转绿（提交 `26539ade`；此前一句"只剩一条待裁"漏了持久化两条，一并更正）。**②③持久化两条红已于同日在根因上消掉，"格式 5"这条裁定不再需要**：真因是**我方 `life-context`／`memory-context` 两个包漏了上游对同类贡献要求的 `@persistenceAttribution` 标注**（同形的上游包 `packages/context/tmux-context/src/index.ts:33` 就有）。补上标注后同一批变化的分类从 `union-variants-changed`（要升版）变成 `attribution-kind-added`（`requiresVersionBump=false`），于是按门自己的处方走账即可：再生成 catalog → `persistence-formats --write` → `persistence-changes --record --decision same-version` 立**同版本**承认记录 → 因源文件行号位移重生成 config-catalog 并重录 pairing。读数：`SESSION_FORMAT_VERSION` 仍为 **4**、`persistence-changes --check` rc=0（62 roots／7 条历史纪录）、两包 40 条用例全绿，提交 `ec2e72b2`。**教训**：门说"必须升版"时先怀疑我方是否漏声明了它认识的兼容策略，别直接接受这个二选一。
8. **C6（审计登记的 P2）⇒ 提案已交付、待批**：[PLAN-M6-01](../reference/PLAN-M6-01_experience-projection-producer_20260926.md)。**2026-09-26 只读取证把前提部分否证**——缺口不是一条而是三条：①**生产者缺**（`project_to_ledger` 仍只被训练脚本与测试调用、`api/` 零引用）；②**消费者也缺**（E1 ledger＝`seed_platform/evolution_ledger.py:73` 的 `EvolutionExperienceLedger`，其 `training_view` 在产品侧无人读 ⇒ 只补生产者会得到一个只写不读的日志，正是审计对 C3/C4 的「只补一端无效」同型）；③**物料错配**（现成三适配器是 skill/mcp/client-plugin，而真机 `GET /api/mcp-client-capabilities` 的 `records=[]`，skill 与 client-plugin 在 native 侧无运行时来源；真正有料的 `GET /api/workbench/capabilities`＝16 条能力、内容寻址 `snapshot_id`、`revision=6`，**却没有对应适配器**）⇒ **照原样接线今天投影 0 条**（给空集铺管道）。提案＝新适配器 `WorkbenchCapabilityAdapter` ＋ 生产者挂 `sleep_pass` 的 project 段 ＋ **消费者同批**（折进 `data/consolidated/corpus-*.jsonl` 并进 `spec.datasets`）；前置否证门 **Z1**＝先只写适配器＋离线脚本对活运行时投影一次，判据「单元数 > 0 且 `native_trainable`」，**不过即停报**（记「物料不足降级」，不铺运行时接线）。**Z1 已于 2026-09-26 跑完**（零改产品码：新诊断件 `scripts/training/diag_taiji_c6_workbench_capability_projection.py` ＋ 报告 `reports/taiji_c6_workbench_capability_projection_20260926.json`）——物料判据**过**（16 条全渲染、`invalid_records=0`、`native_trainable=true`），但带一条不利读数（**CJK 占比中位 0.243** ⇒ 直译行约四分之三非中文字符，按 D 批理由不能就这么进语料），并跑出**两块初稿没料到的硬约束**：P1 撞 `EvolutionCorpusArtifact` 的**闭合 `source_kind` 白名单**（`taiji/evolution_experience.py:245-251` ＋ `evolution_adapters.py:36`，受 `EVOLUTION_CONTRACT_VERSION` 守护）；**缺口从三条变四条**——多出的一块是**渲染器**（`training_view()` 给结构化记录，训练器只认 `{"text": …}`）。量级如实：物料只有 16 条、训练权重≈0，本件定位是「补齐数据环第四块并留接口」。**决策点重排为六条**（kind 白名单三选一／渲染模板与语言／现在做还是等面扩容／生产者落点／消费者是否同批／面板行）在提案 §8；**P1 一行代码未写**（含一个不该由我裁的契约问题）。**Z1b 同日加测**（同一支脚本内换三种渲染模板，同快照 16 条）：`literal` 直译 CJK 中位 **0.243**／行长中位 197，`no_prose` 去英文描述 **0.503**／86.5，`id_only` 只投 id **0.630**／55——三者 `invalid_records` 皆 0、皆 `native_trainable=true` ⇒ **该判据在模板选择上无区分度**；结论是英文描述占主要负载、**不作者中文能力表就到不了中文主导且保住信息量**（`id_only` 靠"少投"换占比、行内已无"这个能力做什么"）⇒ 决策点 2 重排为四选一（作者中文表／`no_prose`／`id_only`／接受分布外行），见提案 §5、§8。**Z1c 同日再跑**（决策点 1 定价；新诊断件 `scripts/training/diag_taiji_c6_kind_widening_impact.py`＋报告 `reports/taiji_c6_kind_widening_impact_20260926.json`，提交 `f2d4a6ad`）：**扩 kind 的真实代价比初稿写的小**（事件侧 kind 早已含 `workbench`，关着的只有语料侧 `taiji/evolution_experience.py:245-251` 那 1 处**行内字面集**；按模式扫 5 个模块得 22 处 kind 门，其中在 C6 路径上的只有 4 处），**而「复用 `verified_domain_material`」被量出不可行**——用 `workbench` 桩适配器实跑注册表生命周期：新 kind 在 `register()` 当场响亮拒绝；复用则 `register()` 过、`from_checkpoint()` 抛 `source registry artifact kind mismatch` ⇒ **跑得起来、重启就读不回来**（只有绕开 `DeclarativeSourceRegistry`、放弃 `snapshot_id`/`revision` 生命周期与增量去重才成立）。另量到**未知 kind 的 durable 保护本就由白名单提供**（往已落盘 checkpoint 注入未知 kind 并重算 digest ⇒ 整档拒绝、无逐条跳过）⇒ **`EVOLUTION_CONTRACT_VERSION` 建议维持 1**（升版连旧档也读不回来）。顺带更正初稿两处措辞：适配器 `source_kind` 应为 `workbench`（命名规则要求语料条目 kind ＝ `<registry.source_kind>_artifact`），以及「两处闭合白名单」的说法。
9. **08 §5 复验集本轮读数（2026-09-26）**：判据①（tsc→tsdown 四连）**全绿 rc 全 0**；判据⑥（lint 只允许登记基线）**0 warnings／7 errors，7 条全在 `ui-life/src/client/LifePanel.tsx`＝与 09-24 基线一致，无新红**；判据②③（doc-sync 43 叶）**39 绿／4 红**（红＝locale 假红＋持久化两条等裁定＋一条 Windows 符号链接权限）；判据⑤（真机 UI 三取数）已在上一轮闭合；**判据④（定向 vitest）2425 过／3 败／11 skip**，3 败＋8 个 collection 挂的文件**逐条归因＝环境前提而非丢改**（bundle 名册需根 `node_modules/@taiji/` 有 workspace 行⇒待有 pnpm 的机器复跑；2 条需符号链接权限；1 条是并发超时），详见 08 §6 ⑧。**剩余待办**：判据③ 真机训练回合（等 R2 收束＋你触发）；~~在有 pnpm 的机器上复跑判据④ 的 8 个 client spec~~ **2026-09-27 已复跑并结清（08 §6 ⑧）**：`corepack pnpm exec vitest run packages/api/session-controller/tests`＝41 文件 839 过／1 红（＝Windows 符号链接权限环境类）——8 个 collection 挂的 client spec 全绿，(i)(ii) 两条出路撤销、无合同缺口。**Z1d 同日补分母**（决策点 3）：名册 68 档共 **2,074,007 行／928,766,482 字符** ⇒ 16 条＝行数 **0.00077%**、字符 **0.00034%**，但 `data/consolidated/` 当前**为空** ⇒ 作为"那一趟巩固产物"它是 100% ⇒ 三条理由要分开：为**环的完整性**做（提案定位）／为**训练效果**做（读数不支持，别按这条批）／**等面扩容再做**。
10. **打包产物真启动（2026-09-27，收官清单第 1 刀）⇒ 产物不能启动，根因已定位、已在产物内复现（转裁定 R8，见 G5 §9）**：真启动 unsigned 产物（win-unpacked 的 `DeepSeek Harness.exe`）时，主进程在模块加载阶段抛 `ERR_MODULE_NOT_FOUND: Cannot find package '@taiji/cordis'`，弹出一个标题为 `Error` 的原生对话框（正文经 UI Automation 在**锁屏下**读出；**锁屏只挡画面、不是原因**），进程存活但不进界面、不写日志。**触发链**：`lib/main.js` →（外部依赖）`@taiji/dsh-api-gateway/stream-protocol` → `@taiji/dsh-typert-protocol` → `@taiji/cordis`。**根因**：`@taiji/cordis` 在链上两包里**只声明为 peerDependency**（`packages/typert/protocol/package.json:37-42`＋`packages/api/gateway/package.json:64-67`），而打包收集"要带的依赖"的环节**只收 dependencies、不收 peers** ⇒ 产物**顶层** `node_modules/@taiji/` 只有 7 个包、**缺 cordis**；完整的 282 包运行时树在 `app.asar/dsh/node_modules/@taiji/`（含 cordis），但 Node 解析从顶层逐级向上、**不会回落到 `dsh/` 子树**（用 `@electron/asar` 列 17734 条实测）。**归属**：这几个 `package.json` 只被 G1／G2 两次提交碰过 ⇒ **随 fork 继承的上游结构**，非 G2 改名所致。**影响**：`package:win:x64:unsigned` 的产物"存在"但**打不开** ⇒ **G5-D1「打包跑通」、D2「装起即用」、D4「桌面可用」在系统级均为红**（此前只有部件级读数）；**R7 的现场位置被前置**（R7 记的是打包末步 office 冒烟红，而崩溃发生在更早的启动路径）。**修法两条（转裁定 R8）**：(乙) 外壳自包含——`apps/desktop/tsdown.config.ts` 加 `deps.alwaysBundle: [/^@taiji\//]` 把 `@taiji/*` 打进 `main.js`，一次消掉整类／(甲) 给 `apps/desktop` 的 `dependencies` 补 `@taiji/cordis`（一行＋一次 install）；两者都需重打包。 ⇒ **R8 已裁＝乙并已落地**（2026-09-27 深夜）：`tsdown.config.ts` main 配置改成 `deps: { neverBundle: ['electron'], alwaysBundle: [/^@taiji\//] }`，**构建层已验证**（重编译后 `lib/main.js` 顶层外部 import 只剩 `node:*`＋`electron`/`semver`/`ws`/`electron-updater`，**零 `@taiji/*` 外部 import**）。**但重打包被一个与 R8 无关的环境问题阻断**：`prepare:dsh` 的 `runtime:smoke` 报 `Cannot launch conpty`——该冒烟今日 16:44/17:01/17:26 还 `success:true`、19:38 起失败；独立探针（新装 node-pty＋普通 node）在 agent 上下文同样失败、经 `explorer.exe` 绕出沙箱则成功 ⇒ **Trae 受管进程上下文创建不了 ConPTY**；绕沙箱跑打包两次均不稳（pnpm 退出 abort／Ctrl+C）。⇒ **干净新产物未产出，清单第 1 刀的系统级读数仍缺**，须在**不受 Trae 沙箱约束的普通终端**补跑一次 `package:win:x64:unsigned` 后续取（详 G5 §10、08 ㉒）。 ⇒ **同日深夜已续上（08 ㉓、G5 §10）**：沙箱外一次运行（run `12-13-23.810Z-GXstl6`，commit＝含 R8＝乙）**除末步外全绿，干净 unsigned 产物实出**；真启动该产物——**`ERR_MODULE_NOT_FOUND: @taiji/cordis` 错误框消失**，用 `DSH_HOME=<工作区内可写路径>` 重跑后**无任何错误框**：主窗口 `Taiji Harness` visible、**应用亲手拉起打包自带的 `dsh-desktop-host` 子进程**、首启动生命周期走完（profile／默认工作区／凭据）、host 监听 `127.0.0.1:19387` 且 `GET /`＝401 鉴权门正常 ⇒ **系统级「装起即用」＝绿（backend host 拉起 ✓），装机默认 provider＝`taiji-local` ✓**。（不重定向家目录时弹的是沙箱 `EPERM mkdir C:\Users\23747\.dsh\...`＝环境，非产品缺陷。）**唯一仍缺＝"发一个回合"**（需 host 打印的带 token URL 或点 UI）。**R7 同轮被收窄：packaged 冒烟载荷层全过、Host/前端/外挂插件 peer 均正常，office 只卡 `xlsx→PDF`（`docx→PDF` 已通过）**——D1「打包跑通」仍被 R7 挡最后一步。

11. **品牌收口＋两项只读发现（2026-09-27 深夜）**。**(a) 登录门与 D3 矛盾（转裁 R9）**：全新装机首启动弹的是**应用自己的欢迎窗**（`apps/desktop/src/welcome-api.ts:58-60` `needsWelcome = !loggedIn && !hasApiKey`；`main.ts:951-963` 触发；欢迎页只有「登录」「API Key」两个出口），而 **G5-D3 已把默认路线改成免凭据的 `taiji-local`** ⇒ 首屏被登录窗挡住，与「装起即用/默认免凭据」冲突（**那条 401 只是本地访问门、纯文本、桌面下自动满足，不是用户看到的界面**）。**(b) 品牌收口已落地（详 08 ㉕）**：客户端记号统一换成新的**「种子→小苗」**（24×24 单色、非零填充、路径常量在 `ui-primitives/src/FishLogo.tsx` 的 `FISH_LOGO_PATH`，导出名保留），矢量资产（web/官网/桌面图标/welcome-brand）全部重写、栅格资产按原尺寸重出（icon 1104²/1024²、安装器 brand 600×196 与 1200×392、uninstaller-sidebar 164×314、徽章 726×120；.ico/.icns 由 electron-builder 生成）；产品展示名 `Taiji Harness`／`DeepSeek Harness` → **`Seed`**（桌面 locale 24 处＋about、electron-builder 的 productName/artifactName(`seed-…`)/protocols.name、安装器 strings、web manifest/title、官网 title、客户端 locale 十余处＋连带 expected/snapshot 同步）；**有意保留**：DeepSeek 模型 provider／`@deepseek-ai/libreoffice-kit`／python sdk 包名／内部包名与键名／`repository` 字段。**(c) 尚欠（本机做不了）**：`snapshots/**` 里 GUI 文案相关的 Web e2e 金标需 `DSH_SNAPSHOT=refresh` 重录；被改过的 README 需 `verify-translation-pairing --write` 重录 `*.i18n.yaml`；**agent 身份/system-prompt 仍是 `Taiji Harness`**（`packages/core/system-prompt`＋~60 份 `snapshots/**/system-prompt.expected.md`）属另一条线、未动；`productName` 改名对产物名的实际效果待在非沙箱机复跑 `package:win:x64:unsigned` 验证。

### 已作废：M6 desktop 线未闭合五项（2026-09-21 立卡）
①logo 判定、②托盘退不出、⑤单实例锁**均已闭合**（所有者裁决＋根因修复＋实测）；③UI 交互其余项、④跨壳孤儿回收**随该线退役作废**（载体已删）。原活动卡见 03 §5.7「M6 desktop 交付线」（线状态＝已退役，卡为历史记录）。

## 2. 支线 A：R2 欠账修复线（语言读出/答对）

> 来历：M5 退出时 R2 语言能力列为**显式排除项**（欠账保留在账）。调查已收束（归因定案），
> 修复由**并行会话**以 A2 复制回路推进。**与 M6 并行、互不阻塞**（负载边界：R2 训练等待期
> 不并行 ≈30 min 级重构建/重评测）。

| 步 | 名称 | 状态 | 判据 |
|---|---|---|---|
| **A-1** | R2 归因 | ✅ **定案** | CAP D+E=0/36 是「先告知→后提问」与"无归纳机制"的结构性错配 |
| **A-2** | 复制回路（A2.1 写入门 + A2.2 F1 内容直读，gate 零初始化位级不变） | ✅ **落地** | §4.1 结构存在性过（greedy 首字节=0xE9）；合同 6 绿 |
| **A-3** | M-3：问答结构语料 + 召回条件发射训练 | ✅ **并入 A2.3** | 立项 §5：A0 证伪"甲单独走"（无通道则无物可学） |
| **A2.3** | 召回条件发射训练（裸格式） | ✅ 判据过 | 寻址 0.86；CAP 抽查 5/36 ⇒ **严格口径 4/16**（见下） |
| **A2.4** | 协议开闸（chat 链入库＋显式挂载，默认关） | ✅ 落地 | chat 形态 3/36 ⇒ **严格 2/16**；真增量 D09"42"/D12"23" |
| **A2.3b** | **协议格式对齐重训**（问：/答：壳＋多事件＋自历史上文） | ✅ **判据②过** | chat 形态严格真命中 **6–7/16**（两个独立初始化电路：seed-A 7、seed-B 6；线 >2/16；对照 0/16、格式错配臂 2/16、裸格式抽查 4/16）；S2 寻址 0.848、基座 sha256 未动 |
| **GATE** | 立项 §4.3 回归门（装上有没有让 A/B/C 变差） | ✅ **收口：无代价、方向为正**（表层在扩展分母上判 `improved`） | 能力分没劣化：A 维 9 项全不掉、C 维 0.0→**0.0714**（升）、B 维 0 项可机检记 `unverified`。表层成句率 0.0612→0.0408（49 句里 3 句变 2 句）判"未通过"，而同一电路在 D/E 原始字节链上成句率**升**（0.0278→0.0833）⇒ 原条文没钉链路/题集，且 36 题、49 句两个分母上"≥3"阈值都要 8% 摆动＝**结构性不可判**。修订见立项件 §9＋[`SPEC-A-21`](../reference/SPEC-A-21_r2_surface_extension_prereg_20260925.md)（104 题≈260 条文本、两独立电路同向才判） |
| **A2.5?** | 事件选择（"挑哪条记忆"变可学） | 📐 **已定价＋已钉下限线，排在键/续接之后** | §10 天花板只否证了**单告知分布**（CAP 替它挑对净 +2）；扩展集带干扰 48 题上摘掉干扰值 **+17/48**（§13）。§15 把 `best_match` 换成"与提问共享字符"这把**免训练**键 ⇒ v1 表层 **23→39、22→40**，且救回的题与天花板**逐题相同**。§16 在 v2（同谓语难干扰）上规则键掉回 **15／14**、天花板仍 **43／39** ⇒ **缺口存在：上限＋27、下限线＝必须超过现状 16**；但选择完美仍剩 **61 题**答不出 ⇒ 先做键/续接（`SPEC-A-17` §13/§15/§16） |
| **A2.6** | 失败题的**逐答案步轨迹**（内容在库里，为什么发不出来） | ✅ **读数收口（三档，零训练）** | 81 道未命中题：scored 链 81/81 复现未命中；三成答案步挑的是**另一条告知**（带干扰家族 110/342）；摘掉干扰后残余 64 题里 `address_miss` 39／`continuation_slips` 29／`emission_loses` 11 ⇒ 键/续接是残量大头，选择是**可摘的一块增量**。件见 `SPEC-A-17` §13 表 |
| **A-4** | 可分离读法推广到主训练线 | ⏳ 前件已满足，排在门补完之后 | `predictive_context_region0_only`（默认关、掩码式、守卫 4 绿） |

**计分口径已收紧（2026-09-25，读旧结论时必带）**：`expected_contains` 里混着通用词，
旧口径把"没装电路也能命中"算成了能力。严格口径＝**期望词必须在更早的用户轮原文里逐字出现**：
裸格式 5/36 ⇒ **4/16**，chat 对照 2/36 ⇒ **0/16**，chat 电路 3/36 ⇒ **2/16**；
且 **E 维严格可命中数恒 0** ⇒ 旧判据"CAP D+E>0"里真正承载复制的只有 D 维。
复算件 `reports/taiji_r2_copy_strict_cap_recompute_20260925.json`。

**支线 A 已落的两刀（零训练）**：§14 静态可分离——扩展集 48 道带干扰未命中题上，字符重叠键挑对
**48/48**、现状神经 cue 37/48；§15 定价——把生产发射消费的 `ToldContentStore.best_match` 在进程内换成
这把键，表层严格命中 **23→39、22→40**（两臂同向、差 ≥3 ⇒ 可判），且 seed-A 救回的 17 题与 §13
"只摘掉库里那条干扰"救回的 17 题**逐题相同**（一个改库、一个改查询，指向同一批题）。
⇒ **选择侧的价格免训练就能全部拿到**：`A2.5`（可学选择器）此刻**不占训练预算**；能声称的只是
"**由提问决定、无状态的选择**"这一机制值钱，**不是**"字符重叠就是最终机制"。

**v2 已造出并取完三档（`SPEC-A-17` §16）**：干扰换成**同谓语不同实体**（"我表哥住在无锡。"配"我住哪？"），
构造层面规则键只能挑对 40.4%。表层严格命中：现状神经 cue **16/104**（两臂四次取数一致）、
规则键 **15／14**（差 −1／−2 ⇒ 与 v1 的＋17 相反，题集修对了）、天花板（替它选对事件）**43／39**
（＋27／＋23，两臂同向 ≥3 ⇒ 可判）。
⇒ **可学选择器的缺口成立，上限＝＋27；下限线＝在 v2 上必须超过 16**（免训练规则只到 15，
"随便加个相似度阈值"不算超线）。**但选择完美也只剩 61 题答不出**，且"规则挑对事件"的 42 题里
表层只出来 15 题 ⇒ **27 题是"选对了仍然发不出来"**——A2.5 立项报收益只能报"16→~40 这一档"。

**§20 已把那一问量清（on-policy 重测，零训练）**：同一档寻址，**离线**指对 94.49%（在旧轨迹的状态
点上量），**自己驱动解码**后只剩 **23.78%**（基线 16.06%）⇒ 表观增益 78.4 个点里只有 7.7 个点活下来
（**九成被分布位移吃掉**）。生效与锚点都自证过：635 次寻址调用里 353 次选了不同位置；
覆写设回训练取值时与原实现同 argmax、逐位差 <1e-4。

**还有一件更难受的事（措辞已由 §23 更正）**：前缀内它确实变好了（发出率 17.5%→23.5%），表层却从
23/104 掉到 12/104 —— 盲区不在"前缀 vs 尾巴"（尾巴那一支已否证，两档 `tail_destroyed` 都是 0），
而在**逐字节指点 vs 把整串连续走完**。
⇒ 制度性两条（后续任何改寻址/改发射的方案都要带）：**评测必须 on-policy**、**结果指标必须覆盖整句**。

**A2.8-3 已把"聚合方式"这一支否证（§23）**：`mean`（÷位置数，公平检验）整串指对只有 **1.11×**
（判读线 ≥1.5×），而**答案首字节 30.9%→27.2%（降）**；`max` 更差（丢重数，不算公平检验）。
屏幕不动 ⇒ 按约定不花那次表层整跑。**键/寻址那侧今天连排三支**：幅度（§17）、聚合（§23）、
离线好解 transplant（§19/§20）——只剩"要不要动内容表征"这一条**需要签字**的路。

**队首（2026-09-28 rev20）：两臂三点归因已出 ⇒ 复述命中几乎不动、"给复述训一条通路"未抬过历史档；请 owner 裁"接受上界 vs 回主线"**——
两臂各跑满预算（`pos_on` 3700 ep／`pos_off` 2700 ep，均 90 分钟），三条判读链全落盘。**严格 /16：历史 A2.3b 6–7 ／ `pos_off` 6 ／ `pos_on` 7**（差 1，远低于 house rule 的 ≥3 ⇒ `not_resolved`）；**v3 严格 /104：17 对 18**。**位置输入的贡献集中在另一条轴**：裸基底真非法率 **99.0% → 0%**、成句 **0 → 68**（裸）／**1 → 32**（挂电路）——决定性但不是复述。**复述已近结构性上界**：v3 18 对历史 oracle 19–21 ⇒ 约 90%；剩余缺口不在选择（近 oracle）、不在合法性（已修）⇒ 落到底座"说不出有内容的话"（主线层级）。**建议（单一）**：先不花 3 小时做 G3（收益空间已量到只剩 ~2 题），请裁 **"接受当前上界" vs "回主线（数据/目标/规模）"**。见 [PLAN-A-24 rev20](../reference/M5_R2_A_BRANCH_PLAN_REV2_20260926.md)。
**PLAN-A-25 已按"实现+守卫"档落地并做完零训练验证（`2d2db87c` 实现；判读件 `taiji_a25_gate_on_pos_on_{surface_v3,cap,strict}_20260928.json`）**：把复制回路的加性证据按 UTF-8 位置状态门控（默认关、逐位不变；评测期走 `set_copy_evidence_utf8_gate` 运行时覆写，因为**身份器官 lineage 守卫拒绝了手改 config 的档**）。**主判据按字面冻结口径（整句可解码率 ≥0.5）未过**（0.1058），但同日发现该口径**在固定字节预算下把"缓冲切在字中间"算成了模型缺陷**（本仓第三次踩同一坑）⇒ 复算"切尾感知的真非法率"：对照（裸基底）**1.0000**、门关 **0.0673**、门开 **1.0000** ⇒ **门的机制完全成立：真非法率 93% → 0%**。归因同向为正（/104 14→18、/16 7→8、成句 32→33）。**两处更正**：①"裸基底在 v3 只有 0.10 ⇒ 合法性不跨题面分布"**是错的**（那 0.17 全是截断，裸基底在 v3 一条非法字节都没有）；② rev18"电路会打破合法性"**仍成立**。**新增 owner 口径裁定项**：§5d 的"整句可解码率"应改为切尾感知口径。见 [PLAN-A-25 §6](../reference/PLAN-A-25_copy-evidence-utf8-gate_20260928.md)。
**队首（2026-09-28 rev14）：按 rev13 结论起"给复述训一条通路"两臂（A2 电路 · 合法通道 vs 同配方对照，各 ≤90 分钟），已起跑**——
rev13 判读是"缺一条被训练过、朝复述方向的通路"，故下一刀＝在合法通道上把 **A2 电路**训起来（它迄今是唯一被证明能把实体带到出口的机构）。开跑前被一处漏接线挡下并已修（`42edda93`）：**A2 训练器直接调 `readout.probabilities(ctx,…)` 重算当前状态分布却没带位置列** ⇒ 在开启位置输入的基底上被读出自己的守卫当场抛错（**守卫奏效**）；6 分钟 judge 探针在 `pos_on` 基底上通过（address_top1 0.7753／gate_abs_mean 25.14／selection_pick 1.0／`criteria_pass=true`）。**两臂**＝`--stage judge --protocol chat`，唯一变量＝基底：`pos_on`（合法通道）vs `pos_off`（同配方对照）⇒ 与历史 A2.3b（seed_beta 基底、CAP 严格 6–7/16）三点并列，可把"读出重训"与"位置输入"分开。**不预设方向**：通道变合法后"命中"含义会变，报告须同时给可解码率/成句率（§6 制度 7）。见 [PLAN-A-24 rev14](../reference/M5_R2_A_BRANCH_PLAN_REV2_20260926.md)。
**首臂终件读数（03:10 跑满 3700 episodes）＝两条更正 + 一条新机制发现**：严格 **7/16**（对照 0/16，落在历史档 6–7/16 上沿）、v3 严格 **14/104**（历史 20）、表层成句 32（对照 68）、**v3 可解码率仅 1.28%**。⇒ ① rev17 那个"中途 9/16 越过历史档"是**暂态**，终件回落（已就地更正）；② **"合法通道会顺带把输出变合法"被否**——位置输入保住的字节合法性只在**裸读出**那条链上成立，**A2 电路的加性证据一旦加到出口 logit 上，合法性就被打回去了**（电路的证据不受当前 DFA 位置约束）⇒ **下一刀候选（新）：按 UTF-8 位置状态门控电路的加性证据**。**归因仍未做**（`pos_off` 对照臂在跑，03:10 起 ≤90 分钟）；在它出数前不得把任何增益/损失记给位置输入。
同时（等待期）闭掉了 rev13 明写的那条未决项 **"token 能否被线性读出"**（新探针 `01e2e90f`）：载体做干扰变量的配对状态距离 + **整表重排**置换零假设，三臂逐位相同（第三次独立确认 `motor_context` 不随读出训练改变）⇒ **7 字节 ratio 0.8554 对零分布 0.9983±0.0203 ＝低 7.05σ、18 字节低 4.25σ、48 字节反超（−7.76σ）** ⇒ **实体身份在近中距可读、远场消失**。这把"表征可读、缺的是通路"钉实，也把 rev12 的"语境对实体名不敏感"彻底否掉。
此前（rev13）：保持曲线 v2 跑通 ⇒ **"复述"缺的是一条被训练过的通路**，不是表征、也不是语境产出方（三臂一致 `exit_responds_but_no_copy_path`：7–60 字节 `alive_rate`/`l0_hit_rate` 全 0，而换名字让状态相对 L1 变 15–36%、出口 KL 变 0.006–0.076 ⇒ 信息在场且到得了出口）。
此前：§5b 队列全执行→F0 判 fail（无掩码可解码全 0；M1 成句 0.87 真相＝判读仪器带掩码扶手）→
SPEC-R2-01 训练目标方向判分支 3（加权在训练分布有效 0.3513/0.2985 但不转移，第四例）。
**2＝`SPEC-R2-02` 解码掩码产品化（已落地）**：主判据过——产品表层占位句率 **32/32→0/32**、
被接受表层 100% 可解码、真 chat 4 题复核一致；实现＝`taiji/utf8_state.py` 共享状态机＋
`generate/generate_input/chat` 链 `utf8_strict`（默认 False⇒逐位不变，golden+F0 复跑双重钉）＋
产品 chat 传 True（owner 授权的默认变更）。**如实登记**：掩码修"合法"不修"成句"——
产品通道 T2/T3/T4 命中全 0、成句率 0.31–0.50 ⇒ 产品从"占位句"变"合法的字汤"，无掩码地板线仍 fail。
**3 的判决前置＝A3 探针（零训练，已跑）**：`motor_context` 线性读"下一字节 UTF-8 位置类" 75.7%
（多数类 64.4%／DFA-state oracle 上界 96.6%）⇒ 位置信息**部分在场但弱**（离上界 20.9 点，
判"分层报数不裁定"）——恰好指向"4 维 DFA 位置状态作读出的显式输入"这最后一块便宜砖 ⇒
**`PLAN-R2-01`（决策就绪，待 owner 批）**：主判据钉死**无掩码 F0 ≥0.5**＋**转移自证**
（teacher-forced 涨/自由生成不涨＝第五次不转移，判据先拦住）；批三项＝架构改动进主干（默认关）＋
两臂×2M 训练预算＋分支 2 的追加权。**诚实预期（PLAN-R2-01 §7）**：若分支 3，问题定级为
"该量级架构装不下守序列又成句的自由生成"，转 M5 主线数据/目标/规模裁决。
A 支线状态不变：表层判据冻结、L1/L2 诊断可用；G8/后果语义/巩固仍归 owner。
（§5c 已按 L0/L1/L2 回填：B1 降"待重测"、真 oracle 限表层上界、乙-1/C3/A1/A2/头权重五条保持；
人脑对照表成表于 PLAN-A-24 §3c。）
§5c 判读按 L0/L1/L2 三级回填：B1 否证降级"待重测"（且其重测须等地板过线）、真 oracle 限定为表层上界；
乙-1 机制诊断（动作-奖赏绑定、无文本槽位）、C3、A1、A2、选择头权重五条保持（L1/L2）。
人脑对照表（用户五行原话）成表于 §3c，逐行给在册指针。

**已排除的五支（别再回头试；rev7 后效力范围注明）**：选择侧已定价（§16）；前驱约束的幅度（§17）；
离线好解 transplant（§19/§20）；**尾部毁掉答案**（§21）；
**可训内容表征**（丁2：v3 3/2 崩塌——rev7 降级为"待重测"：崩塌是 L0 读数，
可能死在发射段；该否证只在"语言地板过线后的通道"里重新有效）。

**已排除／已定的三支**：① 选择侧已定价（§16：v2 上限＋27、下限线＝超过现状 16；v1 上免训练规则键
就能拿到 23→39）；② 前驱约束的**幅度**已排除（§17：钳位值放到≈硬掩码指对率 14.0%→13.1%）；
③ **离线好解直接 transplant** 已否证（§19）。天花板档是 oracle 不是机制（v2 题形里"最早入库"
必然等于含答案的告知，照抄＝给我的题集打分）。


**已落定的两条**：① 能力侧增益可重复——seed-A 严格 **7/16**、seed-B **6/16**，对照均 0/16、格式错配基线 2/16
⇒ 此后一律按 **6–7/16 这一档**引用，不写单点（两电路命中集不同，seed-B 拿下带干扰的 D06）；
② 回归门表层已判 `improved`（扩展分母 104 题／260 条文本，两独立电路同向）——
判法固定：两电路同向且成句差 ≥3 条才判劣化/改善，否则 `not_resolved`，**"没判出劣化"不等于"无代价"**。
**A-4**（可分离读法推广到主训练线）排在键/续接那一刀之后。

## 3. 支线 B：架构债线（审计修复）

| 步 | 名称 | 状态 | 判据 |
|---|---|---|---|
| **B-0** | P0+P1+P2 三轮修复 | ✅ 完成（116 测试） | `project_audit_fixes{,_p2}_2026-08-23.md` |
| **B-1** | `detect_modality` 抽为纯函数（无 self） | ✅ 完成 | 等价守卫 3 绿（`a6895b6a`） |
| **B-2** | `_infer_domain` ✅ 黄金 330 格**真的**等价（该判据当时恒真、一条断言都没跑，2026-09-26 修好并全绿）；`_reencode` ✅ 随 C-2 抽离（黄金全量一致）、`nll_quality` ✅ 随 C-1 抽离并以 neuroplex Cortex 真夹具跑通等价（skip 已结清） | ✅ 完成（挂起项随 B-3 结清） | 见 §6 与 PLAN-B-03 §6 |
| **B-3** | Cortex 神对象完整拆分（签字 2026-09-25）：**已按预注册范围收口 2026-09-26**。C-1 质量/评分 ✅（含第二刀 `_rolling_nll_quality`，`e3a6e0a4`，迁移前黄金 10 格）、C-2 tokenizer/对齐 ✅（黄金 50000 条）、C-3 路由 ✅第一刀 `_fingerprint_route`＋其余三成员**留壳**（调 `think()`+EMA 状态 / 依赖 `ensemble`）、C-4 单步解码 `decode_step` ✅（`0ef93b0b`）；§4 **唯一声明的行为变更已落地**（`1da2877e`：兜底 `next(iter(set))`→`min()` 前缀，实测 330 格里 21 格随 `PYTHONHASHSEED` 翻转 ⇒ 改后 0 格，且 21 格确定值逐格等于原黄金）；同件结清 **B-2 的黄金守卫空跑**（跳过判据对 330 格恒真 ⇒ 一条等价断言都没执行）。逐成员处置表见 PLAN-B-03 §6，C-4 内联纯段后续刀清单见 §7 | ✅ 收口 | 每迁一簇配等价测试；冻结基线，禁静默漂移 |
| **B-4** | 核心推理路径覆盖率 ≥60%（审计 §7-16）：**达标**。度量面按审计定位钉死（brain/cortex + `_cortex_*` + resonance 的 ensemble/continuous/field + layers.py；`working_memory` 作为"仅注册未接入"的死模块排除），门 `scripts/training/verify_core_reasoning_coverage.py` 按文件过滤分母、面内文件缺席即红，**已接入 CI 且阈值直接取 60%**（只在套件步成功时判，理由见债册 DEBT-B4-1/5）。读数（全量单次收集、同树两次）：**65.03% / 65.25%**（2427-2435 / 3732）；分文件 cortex.py 20.6→65.7%、ensemble 869→962 行、layers 88%、field 83%、`_cortex_generation`/`continuous` 100%。本轮 B-lane 测试件 **14 个文件 / 113 条**（其中 12 个文件为新建，另 2 个是改写既有守卫），过程中修掉 4 条真实缺陷（隔离池 id 复用覆盖 ckpt／`set_neuromodulator(None)` 半装配异常／聚合加载静默吞 field／`_get_neuron_tokenizer` 对带后缀 nid 的死回退）并结清门自身两处"永远红/读不准" | ✅ 完成 | 门随 CI 常态执行；剩余缺口集中在 ensemble 的前向编排（未覆盖 ~790 行） |

## 4. 存量映射（**新编号 ↔ 旧文件名**；旧文件冻结不改名）

| 新编号 | 旧文件（`plans/reference/` 与 `scripts/training/`） |
|---|---|
| `MS-M5-01` | `M5_EXIT_APPROVAL_20260920.md` |
| `EXP-A-01` | `M5_R2_READOUT_RETRAIN_*`（A/B/C 合同、判决与 runner） |
| `PLAN-A-02` | `M5_R2_COMPOSITION_BINDING_BUDGET_PROPOSAL_20260923.md` |
| `SPEC-A-03` / `EXP-A-04` | `M5_R2_T1_T2_PREREG_20260923.md`（预注册）与其轨迹/判决件 |
| `SPEC-A-05` / `EXP-A-06` | `M5_R2_T3_FABRIC_WRITE_PREREG_20260923.md` |
| `SPEC-A-07` / `EXP-A-08` | `M5_R2_T4_SLOW_ALL_PREREG_20260924.md` |
| `SPEC-A-09` / `EXP-A-10` | `M5_R2_T5_REGION0_CUE_PREREG_20260924.md` |
| `SPEC-A-11` / `EXP-A-12` | `M5_R2_T6_CAP_PREREG_20260924.md`（判决件 `reports/taiji_r2_cap_checkpoint_scores_20260924.json`） |
| `SPEC-A-13` / `EXP-A-14` | `M5_R2_T7_MASKED_A_PREREG_20260924.md`（判决件 `reports/taiji_r2_masked_arm_a_20260924.json`） |
| `PLAN-A-15` | `M5_R2_T8_MECHANISM_DESIGN_20260925.md` |
| `PLAN-A-16` | `M5_R2_ARCH_LEVEL_PROPOSAL_20260925.md` |
| `SPEC-A-17` | `SPEC-A-17_r2_a2_3b_format_align_prereg_20260925.md`（**新建即按本表编号**） |
| `PLAN-A-18` | `M5_R2_ARCH_COPY_CIRCUIT_PROJECT_20260925.md`（A2 立项） |
| `SPEC-A-19` | `M5_R2_A2_3_PREREG_20260925.md`（含 §6 rev2–rev4 修订与 §7 读数） |
| `PLAN-A-20` | `M5_R2_A2_4_PROPOSAL_20260925.md`（协议开闸提案＋chat 形态读数） |
| `SPEC-A-21` | `SPEC-A-21_r2_surface_extension_prereg_20260925.md`（**新建即按本表编号**；§4.3a 表层子判据的扩展分母） |
| `SPEC-A-22` | `SPEC-A-22_r2_a2_5_query_conditioned_selector_prereg_20260926.md`（**新建即按本表编号**；A2.5 开案预注册：判据与评测协议先于代码冻结） |
| `PLAN-M6-01` | `PLAN-M6-01_experience-projection-producer_20260926.md`（**新建即按本表编号**；C6 经验投影的生产者／消费者／物料／**渲染器**四缺口、`WorkbenchCapabilityAdapter` 与已跑的 Z1／Z1b 前置否证门＋其读数） |
| `PLAN-A-25` | `PLAN-A-25_copy-evidence-utf8-gate_20260928.md`（**新建即按本表编号**；按 UTF-8 位置状态门控复制回路的加性证据——由来是 rev18 实测"位置输入保住的合法性被电路证据打回去"，默认关、零训练可判） |
| `FIX-B-01` | `project_audit_fixes_20260823.md` / `project_audit_fixes_p2_20260823.md` |
| `CONV-B-02` | `docs/metrics_conventions.md` / `docs/REPO_HYGIENE_RULES.md` / `docs/FOLDER_STRUCTURE_RULES.md` |

## 5. A-3（M-3）详细设计：**问答结构语料 + 召回条件发射训练**

> **本节已被 A2.3 取代（2026-09-25）**：A0 重判证明"无内容通道则无物可学"，M-3 的语料与制度
> 设计已并入 A2 立项 §3-A2.3 与 `SPEC-A-19`（学习规则、实体表防泄漏、预算止损全部照彼件冻结）。
> 下文保留作设计依据与判读线出处，**不再作为待办条目**。

**目标**：让 F1 读出在"问题条件"下学会**发射答案字节**（T8 已判 (c)：读出没学过"由召回内容发射答案"）。

* **语料来源**：`data/simple_zh/dialogue_extended_clean.jsonl`（已有 问：/答： 结构）。
* **语料格式（设计）**：每行 JSON
  `{"text": "<告知>。<提问>？<答案>。"}`
  例：`{"text": "我叫阿岩。我的名字是什么？阿岩。"}`。
  由 dialogue 的 问：/答： 对改写成此形，**再注入合成的 given-then-ask 对**（不同名字/地点/物品），
  避免只学表面问答形状。
* **⚠️ 防泄漏（沿用指标白皮书）**：合成对的"告知内容"**必须与 CAP-43 评价集的答案不相交**
  （否则 CAP>0 只是测试集记忆）。评价集 `cap0_eval_set_v2.json` 仍作**持有外**基准，不进训练。
* **训练制度**：F1 读出（`readout="predictive"`）全程 `learn=True`；
  `use_memory=True, use_identity=True`（召回条件存在）；写入面开放（Fabric 写入允许）。
* **判读线（冻结）**：CAP D+E 机器计分 **> 0**（对照 `t1@5M=0`、`base=0`）；成句率只报不判；
  0 ⇒ 记"读出对问答条件也无通路"（升级 B/架构侧）。
* **预算**：5M 判读 ≈ 5.5 h → 16M 定论 ≈ 22 h（视判读线定）。

## 6. B-2 / B-3 详细设计：Cortex 神对象拆分

**模式**（已建立样板 `B-1`）：纯函数抽离 + staticmethod 委托 + 黄金向量等价测试。

* **成员**：`detect_modality`（已迁，无 self）✓；`_infer_domain`（依赖 `self.neurons`）；
  `_reencode_domain_generation_context`（依赖 `self.neurons`）。
* **方法**：把 `self.neurons` 提升为参数 ⇒ **签名变 ⇒ 真重构 ⇒ 需负责人签字**。
  每迁一簇，先在 `tests/` 立**黄金向量等价测试**：同一输入下 `Cortex.方法(x)` == `helpers.函数(neurons, x)`
  **逐位相同**，绿后才迁；迁后立刻跑全量测试。
* **顺序**：B-1 `detect_modality` ✓ → B-2 `_infer_domain` → B-2 `_reencode_domain_generation_context`
  → B-3 完整拆分（路由集群 / 域推断集群 / 生成集群）。
* **风险**：冻结基线的静默数值漂移 ⇒ **等价测试先立、不绿不迁**；不签字不动。

## 7. 里程碑索引

| M | 名称 | 状态 |
|---|---|---|
| M5 | 限定退出 | ✅ 2026-09-20 批准落盘 |
| M6 | 产品交付（界面＝Taiji Harness） | 🔄 G1–G3 ✓／G4 判据①②④达成、③待裁定／G5 待做 |
7. **G5 交付（打包＋桌面可用／一条命令装起即用，Taiji 为默认）——四项判据都缺裁定而非缺工**：**D2** 同日估工后从三选一改收成二选一（(b′) 载荷随包＋桌面拉起常驻进程 / (c′) 用户自备并启动，我先前"扩清单基本排除"的依据——发布工作流体积上限——在 workflows 里搜不到证据，已推翻）；**D3** 装机默认仍是 `deepseek-official`（改两行 config 属产品默认变更，已真机验）；**R4** `apps/desktop/.env.windows` 那组产品值给不出则 `package:desktop:dir` 一步不动；**R5** 三处产品身份（默认工作区目录名已核**不需迁移**）。**八项裁定的可回复形状见 G5 §8 裁定单**（含可照抄的最小回复格式）。
8. **web 表层 lane（08 §5 第 7 条）今日净进展：红文件 40 → 32**，全部可归因到两条 fork 补丁 **H3u**（路径分隔符锚定替换，POSIX 上是恒等式、金样一字未动）与 **H3v**（会话行从位置索引改成按 basename 选，覆盖 7 个文件；`markdown-images` 不属该族、已撤回）。最小面 4 个 keyless 文件在 HEAD 上第二次独立验证 **4 文件／11 用例全绿（27.11 s）**，可当门；门前提三条已写进 08 §5：**build 与跑门互斥**、**批内不含 spawn watcher 的 lane**（`hmr-live` 会改写共享 `apps/web/dist`）、**起止 mtime 存档且取证在下一次 build 之前**。
