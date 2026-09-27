# G5 交付｜就绪清点（只读，未改任何产品默认）

> **为什么先做这份清点**：M6 的最后一个里程碑是 **G5 交付**，计划里的判据原文只有一句
> 「**打包（harness 自带机制）＋ 桌面可用 ｜ 一条命令装起即用，Taiji provider 为默认**」
> （[采纳简报 §里程碑表](TAIJI_HARNESS_ADOPTION_BRIEF_20260922.md:50)）。
> 这句话在动手前必须被拆成"哪些机制已经有、哪些是配置、哪些要人拍板"，否则"推进 G5"
> 会直接变成"擅自改产品默认"。本件只做**只读清点＋逐条挂证据**，不含任何改动。
> 生成时间：2026-09-26。所有行号/命令均为当轮实测。

## 1 · 判据拆成四件事

| # | 判据里的短语 | 它其实要求什么 | 本轮实测状态 |
|---|---|---|---|
| D1 | "harness 自带机制" | 打包命令存在且可用，不需要我们自造发布链 | **已具备，且第一段已实跑**：`package.json` 里有 `build:desktop`／`package:desktop`／`package:desktop:dir`，以及 `package:desktop:win:x64`、`:mac:arm64`、`:mac:x64`（含 `:dir` 与 `:unsigned` 变体）——全部经 `pnpm --filter @taiji/dsh-desktop run …` 驱动。**2026-09-26 经 `corepack pnpm` 实跑第一环**：`apps/desktop` 的 `build`（＝`tsc -b && tsdown`）**rc=0**（末段产物 `lib/welcome/welcome.js 151.23 kB`，`✓ built in 362ms`）⇒ 桌面外壳在本机可编译。**仍未实跑**：`package:desktop:*`。**2026-09-26 也实跑了一次 `package:desktop:dir`，在拉任何二进制之前就停了**，报错精确且可操作：`desktop package: cannot read apps\desktop\.env.windows; copy .env.windows.example and fill in the local settings`（rc=1，`ERR_PNPM_RECURSIVE_RUN_FIRST_FAIL`）⇒ **D1 的真实阻塞不是 electron 下载，而是一份未提交的本地打包配置**，里面装的全是**产品级决定**（见 §4-R4）。同一条注释还暴露了客户端版的结构性前提：该文件里有 `DSH_DESKTOP_NPM_REGISTRY`——"Optional npm registry for **the bundled dsh runtime install**"，而 `prepare-package-set.ts:141-151` 是按每个包自己的 `name`＋`version` 组一份 package set 再 `pnpm install` ⇒ **默认按名字到 registry 取包（registry.npmjs.org）**；本 fork 的 `@taiji/*` 并未发布，所以客户端交付还需要一条"包怎么到达安装现场"的通道（发布／镜像／本地 packed tarball，`release:pack` 是否喂给它尚未查清） |
| D2 | "一条命令装起即用" | 装完之后第一次启动不需要用户手工装配运行时/凭据 | **未达成；但我上一版说"打包里没有随包带运行时的环节"说过头了，已按代码更正**：桌面打包链**本来就自带一条独立 Python 分发通道**——`apps/desktop/README.md:31` 明写 "Desktop carries independent Python, Node.js and pnpm distributions"（Python 里含 numpy/pandas/python-docx/python-pptx/openpyxl/Pillow/lxml/XlsxWriter），`apps/desktop/scripts/prepare-primary-runtime.ts:29` 会**要求 `manifest.pythonPackages` 非空**才继续，`apps/desktop/scripts/smoke-runtime.ts:39` 会真的 `exec` 那份解释器做冒烟 ⇒ **载体、版本钉、签名与冒烟流程都是通的**。所以 D2 缺的不是"怎么随包带 Python"，而是两件更小的事：① 把我们的后端依赖（服务框架与推理依赖）**加进那份 Python 清单**（体积与许可要重新算，torch 那一类明显不是 office 库量级）；② **谁在装机后把它拉起来并等它就绪**（现通道服务的是 office 技能库的"按需调用"，不是常驻服务进程；设计沿革见 `.agents/notes/implemented/architecture/2026-09-17-shared-office-runtime.md`——**该原文本轮已读，结论见下一段**）。。⇒ D2 从"要造机制"降级为"要扩清单＋加一条生命周期"。**但同轮读完那份设计原文后又把估计往回抬了一格，且方向变清晰**（`.agents/notes/implemented/architecture/2026-09-17-shared-office-runtime.md`）：那条通道**是通用 Python 载荷机制，不是 office 专用**——第 19 行：`runtime.json` 顶层存 Python/Node/pnpm 版本、**所有 Python 发行版版本一律进 `pythonPackages` 这张完整映射**，且"Build metadata does not single out numpy or pandas"；第 31 行还把"每个库一个 component 字段"当反例否掉；第 15 行：共享构建入口 `scripts/primary-runtime/prepare.ts` 明文允许"其他载体自选输出目录、**可只建 Python 载荷**"；第 13 行：`DSH_PRIMARY_RUNTIME` 可覆盖路径、置空即退出。**真正卡住我们的是第 21 行的两条**：载荷是**随发布物走的资源目录**，且"public-index size limits remain enforced by the release workflow"⇒ **torch 那个量级塞不进同一条通道**；并且这套语义是"按需调用的解释器＋技能"，不是**常驻服务进程**。⇒ D2 的可选路收敛为三条，第一条基本可排除：**(a) 扩 `pythonPackages` 装后端**＝撞发布物体积上限；**(b) 给后端开一条独立分发通道**（自带 wheelhouse／离线安装器，可复用 `DSH_PRIMARY_RUNTIME` 这类载体覆盖口子）；**(c) 文档要求用户自备并启动后端**（最省，但"一条命令装起即用"这句就得改口径）。**建议按 (b) 估工、(c) 兜底**；(a) 只有在我们后端能瘦到无重型张量依赖时才成立。 |
| D3 | "Taiji provider 为默认" | 新装 Profile 的**默认模型选择**落在 Taiji 路由上 | **未达成，旋钮已定位到具体两行**：默认选择归 `@taiji/dsh-agent-default-model`（`packages/core/agent-default-model/src/index.ts:24-32` 的 `Config{provider, model, reasoningEffort?}`，三项都 `volatile`，实际值优先从 `settings` 读），而交付装配里**确实设了它**——`packages/bundle/base/cordis.patch.yml:82-86`：`config: {provider: deepseek-official, model: deepseek-flash}`。⇒ **装机默认是 `deepseek-official`，不是 Taiji**（我初稿写"没有任何 profile 设它"是错的，已在本文更正）。**再往下一层未测**：`deepseek-official` 这条路由在 web 装配里是否真有 adapter 服务、是否需要凭据——`base`（2 处）与 `acp-app`（1 处）里出现的是**引用该 provider 名的配置**，`packages/bundle/web-app/cordis.patch.yml` 里为 **0 处**；一个已被测试脚手架承认的风险写在 `apps/web/tests/default-model.overlay.yml` 的注释里："shipped deepseek-official default would be a route nothing serves — which the composer refuses to type into"（所以 fixture 自己改设了路由）。⇒ 新装用户第一眼面对的默认模型**可输入但发不出**（凭据缺失，见 §3.5），而 Taiji 路由按 `llm-taiji` 的合同**不需要凭据**（运行时在本机）。改那两行属**产品默认变更，需所有者拍板**（§4-D3），且**依赖 D2 先定**（§3.5 末段）。 |
| D4 | "桌面可用" | 桌面外壳里我方六个包真在装配内 | **我上一版的推断是错的，已按代码更正**：我原写"桌面走的是 `acp-app`／`headless` 那条不含我方行的链"（依据是 bundle 行数计数），但 `apps/desktop/src/main.ts:502` 实际服务的是 `resources.dsh/node_modules/@taiji/dsh-web-frontend/dist…`，而 **`@taiji/dsh-web-frontend` 就是本仓的 `apps/web`**（`apps/web/package.json` 的 name，且根 `build:web` 也指向它）⇒ 桌面外壳渲染的正是携带我方 10 行的 web 装配，**行数计数那条论证不成立**。**真正未测的问题换成了另一个**：打包暂存（`release:pack`／`prepare:desktop`／vendor 链）有没有把我方六个包放进 `resources.dsh/node_modules`。**本轮同日已答，答案是"有"**：`apps/desktop/scripts/prepare-dsh.ts` 的装法是**依赖闭包安装**（stage 一份 package set → `pnpm install --prod --frozen-lockfile` → 把 `node_modules` 整体拷进产物），而闭包根 `@taiji/dsh-web-frontend`（＝`apps/web`）经 `@taiji/dsh-web-app` 声明了**我方 6 包中的 5 个**（`dsh-api-life-controller`／`dsh-client-ui-life`／`dsh-life-context`／`dsh-memory-context`／`dsh-session-memory-taiji`），第 6 个 `@taiji/dsh-llm-taiji` 由 **`@taiji/dsh-base` 直接声明**（我一度怀疑它漏了，查后是本仓库内可达，不是缺口）⇒ **桌面交付的软件面覆盖我方全部六包**。于是"客户端版后做"真正剩下的只有两件事：**D2（python 运行时怎么随包）** 与 **`package:desktop:*` 实跑（要拉 electron 二进制＋签名）**，不再是"装配/包集缺东西"。 |

## 2 · 与 G5 直接相关的本轮欠账（已消掉）

* **doc-sync 全量 43 叶重跑＝39 绿／4 红**（`0a5ba891`／`61040729`）；红＝`dependency-catalog` locale 假红、持久化两条等格式 5 裁定、一条 Windows 符号链接权限用例。
* **我方包 README 骨架 7 份不合规已补齐**（`26539ade`）——这类"包文档不合格"在交付审阅时最容易被点名，且此前被"这叶跑不了"的错判遮着。
* **判据①（tsc→tsdown 四连）与判据⑥（lint 基线）本轮取到读数**（判据⑥ 7 条 error 全在 `ui-life/src/client/LifePanel.tsx`＝与登记基线一致，无新红）。

## 3 · 明确不声称

* **不声称桌面打包可跑通**：实跑的只有第一环 `build:desktop`（`apps/desktop` 编译 rc=0）；`package:desktop:*`（拉 electron 二进制＋签名＋写 `.desktop-build` 暂存树）**一条都没实跑**。"已具备"到本轮为止指**命令存在、指向真实脚本、且桌面外壳可编译**。另：本机 `corepack` 能起钉住的 `pnpm@11.7.0`，所以旧的"本机无 pnpm"措辞已作废——没跑的原因改成"会拉外部二进制并写暂存树，等 D1 排期"。
* **不声称 G5 已就绪**：D2／D3 未达成，D4 只有 web 侧达成。
* **不声称桌面产物一定缺我方行**（**本轮已查证，结论是不缺**）：桌面暂存走依赖闭包安装，我方六包全在闭包内（5 个由 `dsh-web-app` 声明、`dsh-llm-taiji` 由 `dsh-base` 声明）⇒ 客户端版剩下的只有 D2 与 `package:desktop:*` 实跑两件事。

## 3.5 · D3 的下一层已测清（同日补做，静态读码＋待一次真机复验）

「装机默认是不是没人服务的路由」这条我原本记成"未测"，现已读出答案：

* **提供方在装配里**：`deepseek-official` 由 `packages/llm/llm-deepseek/src/index.ts:57` 注册，而该包在 `packages/bundle/base/cordis.patch.yml` 里有 **3 行**装配（`web-app` 0 行，但 web 走 base 的继承链）。
* **adapter 是无条件注册的**：`registerAdapter([PROVIDER], adapter)` 在同一 `apply` 里直接调用（`:114`），前面没有凭据判断；凭据是**每次请求**才解析的（`:65-81`），拿不到时抛的是可操作的文案：`llm-deepseek: no API key for provider route "deepseek-official"; store <ref> through the credentials`。
* **⇒ 结论与初稿的担心相反**：新装用户**能**在默认模型上输入（路由有服务），但**发不出去**，直到存进一个 DeepSeek key。上游测试注释里那句"a route nothing serves"只适用于它自己的 fixture（那个 scaffold 不注册 adapter），不适用于交付装配——这条我按实测更正，不再当风险引用。

**因此 D3 依赖 D2，三条裁定不是并列的**：Taiji 路由按合同不需要凭据，但它要求本机 8000 端口的 python 运行时在跑（`llm-taiji` 走 HTTP，未就绪时按设计不注册 adapter）。⇒ **在"运行时怎么随包"（D2）定下来之前，把装机默认改成 Taiji 只会把"要 key 才能答"换成"要另起进程才能答"**，两者都不是"装起即用"。所以合理排期是先裁 D2、再随它一起裁 D3，而 D1（打包实跑）可以现在就做以拿最便宜的失败信号。

**仍待做的一次真机复验**（不需裁定，需要一次前端构建＋一次本机启动）：干净 profile 起一次 web，读 `ctx.agentDefaultModel.currentSelection()` 与 `listProviders()` 的交集，确认上面这条静态结论在真装配里成立（默认=可输入、首轮请求=报缺 key）。**本轮试过一条捷径并失败**：`vitest.e2e.config.ts` 的 `include` 只有 `packages/*/*/tests/**/*.e2e.ts` 与 `apps/cli/tests/**/*.e2e.ts` ⇒ 直接跑 `apps/web/tests/default-model.e2e.ts` 报 `No test files found`；该文件属 `test:web:built` 那条 lane（要先 `pnpm run build` 出 web dist）。**这一段原先写的"并入 §4-R1、在有 pnpm 的机器上做"是错的**：那来自同一天被否证的"本机跑不了 pnpm"错误前提（corepack 0.34.6 可跑锁定的 pnpm 11.7.0，本轮已用它实跑构建与安装，见 08 §6 ④），所以这条复验的技术前提本机具备，**不并入 R1**；它剩下的唯一门槛是要不要在这台共享 worktree 上起一次长驻服务。因此 §3.5 的结论当前标注为**静态读码所得（行号可查）**，不是真机读数。

**2026-09-26 同轮补：这条结论现在有三条互相独立的静态支撑，而且"怎么真机验"也定了**：
① 装配里有这个适配器——`packages/bundle/base/cordis.patch.yml:524-525` 是真挂载条目
（`- id: llm-deepseek` / `name: '@taiji/dsh-llm-deepseek'`），其上方注释写明"键与端点都不内联，
按请求从 `llm-deepseek:` 设置节＋凭据存储解析"；
② 注册无条件——`packages/llm/llm-deepseek/src/index.ts:114` 在 `apply` 里直接
`ctx.llm.registerAdapter([PROVIDER], adapter)`，前面没有凭据判断；
③ 凭据按请求解析——同文件 `:65-81` 拿不到 key 时抛可操作文案。
⇒ "装机默认**可输入、首发报缺 key**"这条判断的证据链是完整的，只差一次真实启动。

**（同日更正）上一条里我写的"现有测试 lane 按设计不注册 adapter，拿它测默认路由会得出假阴性"是错的**，那句是从
`apps/web/tests/default-model.e2e.ts:10` 的注释推广来的，而它只描述 `launchWebScaffold()` 的**默认**配置。同一个
scaffold 有一个专门的键控参数：`apps/web/tests/scaffold.ts:390-395` 的 `deepSeekMissingCredential?: boolean`，文档
原文是"**Keep the shipped DeepSeek adapter mounted** while masking the process environment's `DEEPSEEK_API_KEY` for
this scaffold lifetime. **This is the keyless first-run configuration lane**; the default disables the adapter."；
再往装配面看，`:752` 加载的是 **`@taiji/dsh-base` ＋ `@taiji/dsh-web-app`**（真实交付链，不是替身），`:866` 的分支
在 keyless 模式下**跳过**注册假的 `RouteOnlyAdapter`。⇒ **"新装、无凭据、走交付装配"这个场景上游已有 lane 覆盖**，
用到它的有 5 处（`shipped-composition.e2e.ts:524`、`deepseek-messages-settings.e2e.ts:22`、
`onboarding-deepseek-config.e2e.ts:36`、`onboarding-native.e2e.ts:24`、`onboarding-usable-provider.e2e.ts:31`）。

**更强的一条：`shipped-composition.e2e.ts:523-527` 断言的正是我要取的那两个读数**——
`ctx.agentDefaultModel.currentSelection()` **等于** `{ provider: 'deepseek-official', model: 'deepseek-flash' }`，
并对 `ctx.llm.providerRetryPolicy('deepseek-official')` 取了内联快照（能取到值 ⇒ 该 provider 在这份装配里确有注册）；
`onboarding-usable-provider.e2e.ts` 头部还写明它让真实 DeepSeek 适配器整程挂在无凭据状态，浏览器断言的是首启动的
"添加一个 API Key 开始使用"卡片。⇒ **D3 的证据等级从"静态读码"升到"上游自己维护的断言（本 fork 尚未实跑）"**。

**于是"首选动作"换掉**：先前写的是"起一次 ship 配置的 `dsh web` 再读交集"，现在改为**跑已有 lane**——同一目的、
更便宜、且判据面由上游维护（我另起探针只会造一份要自己养的第二判据）。本机实跑的前提只剩两个：chromium 已在
（`~/.cache/ms-playwright/chromium-1223`），缺的是 `apps/web/dist`（`apps/web/tests/support.ts:93-97` 的
`requireDist()` 会拒跑）⇒ `corepack pnpm run build:web` 之后单跑 `shipped-composition.e2e.ts` 那条 `it(...)` 即可
拿到**本 fork 的真机读数**（顺带覆盖我方六包是否在交付装配里）。

**2026-09-26 同日实跑读数（`build:web` rc=0 ⇒ 该 lane 在本 fork 首次跑通）：D3 不再是静态推断**。跑的是一条已存在的上游 lane，不新立判据：`corepack pnpm exec vitest run --config vitest.web.config.ts apps/web/tests/shipped-composition.e2e.ts -t 'assembles the shipped Web transport'`。它跑到该 `it` 的 **`:603` 才失败**，而**失败点之前的两条 D3 断言都已通过**：① `ctx.agentDefaultModel.currentSelection()` **等于** `{ provider: 'deepseek-official', model: 'deepseek-flash' }`（`:526`）；② `ctx.llm.providerRetryPolicy('deepseek-official')` 取到了内联快照（`:534`）——这条算"有注册"的证据是因为 `packages/llm/llm/src/index.ts:969-973` 的 `registration()` 在未注册时直接抛 `NO_ADAPTER`，**取到值就意味着该路由挂着真适配器**。整条 lane 的装配面＝`launchWebScaffold({ deepSeekMissingCredential: true })`：加载 **`@taiji/dsh-base`＋`@taiji/dsh-web-app`**（`scaffold.ts:752`）、**屏蔽进程环境的 `DEEPSEEK_API_KEY`**（`:390-395`）、且**不**注册假的 `RouteOnlyAdapter`（`:866`）。
⇒ **口径要分层，不许把这条说成"D3 全验完"**：**已验**＝交付装配里默认路由**有服务**（"composer 拒绝输入"的那个担忧在本 fork 不成立）；**未跑**＝首发请求的实际报错文案与首启动卡片（那是 `onboarding-*.e2e.ts` 那几条浏览器 lane，本轮被 `-t` 过滤掉了）、以及"把默认换成 Taiji"之后的行为（属 §4-D3 的产品意愿，不是未知）。
**附带读数（对 §6 ⑨ 的锁文件同步门有用）**：`build:web` 触发了一次 pnpm 隐式安装，**`pnpm-lock.yaml` 零漂移**（`git status --porcelain -- pnpm-lock.yaml` 为空）⇒ `bc16104e` 补齐 5 个 importer 之后，先前"每次安装都静默改写锁文件"的现象已消失。
**同一条 lane 暴露的一处环境性红（登记为 08 §6 ⑩）**：`:603` 的工具花名册内联快照期望含 `bash`、Windows 本机实得 `pwsh` ⇒ 该断言把**宿主 shell** 烤进了快照，与 `project-doc-site.spec` 的 symlink EPERM 同类（环境依赖，非 fork 回归）；本轮未为它改判据，也未跑该文件其余 6 条。

**2026-09-26 同日第二批真机读数（装上钉住的 `chromium_headless_shell` 后跑那 4 条 keyless 浏览器 lane）**：跑的是 `vitest run --config vitest.web.config.ts` 下的 `onboarding-usable-provider`／`onboarding-deepseek-config`／`onboarding-native`／`deepseek-messages-settings`，共 **8 条用例红**（其余通过）。**红里带出三条我们要的表层读数**：
① **Taiji 组在交付装配的"设置 › 模型"里在场且可编辑**——ARIA 金样 diff 里是新增行 `+ text: Taiji（本地运行时）`／`+ button "编辑 Taiji（本地运行时） (taiji-local)"`；这条的结构性支撑是 `packages/client/ui-settings-models/src/client/store.ts:183-185` 把 `llm.listProviders()` 与 `listConfigurableProviders()` join 成行 ⇒ **行在场＝注册面在场，不是 UI 自己造的**。运行时前提本轮独立量过：`GET http://127.0.0.1:8000/api/health` **HTTP 200**、`model_loaded`／`taiji_available`／`seed_active` 全 true。⇒ **"启动即得 Taiji 组"在"运行时在场"这个前提下已有表层读数**；它当时把 D2 的口径写成「运行时不在场时这一行按设计不出现」——**这句同日实测被推翻，见下方第四批**（`packages/llm/llm-taiji/src/index.ts:26,66-70` 的就绪探测），所以 D2 三条选项里只有"随包带并拉起"或"引导用户启动"能让这行在装机首启出现。
② **D3 的表层半句验到了**：同一次 diff 里 DeepSeek 行的状态图标从 `API 密钥已配置` 变成 **`API 密钥缺失`**，并且其下 `textbox "API 密钥"`（placeholder 输入 API 密钥）在场 ⇒ 默认路由**可输入、无凭据**这条不再只有 API 层证据。
③ **8 条红的共同形态**＝全在等"添加一个 API Key 开始使用"这张首启动卡片或其字段（`locator.waitFor` 15 s／30 s 超时）。与上游设计自洽的解释是：该卡片只在"没有任何可用路由"时出现，而本 fork 在运行时在场时**已经有一条免凭据可用路由**（`onboarding-usable-provider` 这条测试的名字本身就说明"配好另一个可用提供商就会结束 onboarding"）。⇒ **这条因果本轮没有钉死**：判别实验要把 8000 端口的运行时停掉再跑一次，而那是并行会话共用的服务，**停它由所有者排期，我不做**。登记为 08 §6 ⑪。

**判别实验已跑，因果钉死（2026-09-26 同日第三批，本 fork 新增两个文件）**：不改上游任何一条 lane，而是新写一条最小实验（`apps/web/tests/taiji-runtime-absent.e2e.ts` ＋ 同目录 `taiji-runtime-absent.overlay.yml`），走 scaffold 已有的 overlay 入口（`scaffold.ts:554-556`）把 `llm-taiji` 的 `baseURL` 指到死端口 `http://127.0.0.1:9`（**共享运行时一次都没停**）。读数 **1 passed／4.18 s**，三条断言同时成立：`ctx.llm.listProviders()` **不含** `taiji-local`、**含** `deepseek-official`、且首启动的「添加一个 API Key 开始使用」卡片**回来了**（卡内 `API 密钥` 输入框在场）。⇒ 上一批那 8 条红的主因确认：**本地运行时在场时本 fork 多了一条免凭据可用路由，首启动因此不再索要 key**——这条产品行为正面支持 G5 的「装起即用」，也正面解释那 4 条上游 lane 为何在 fork 侧结构性对不上。**留一处边界**：本实验钉的是「路由撤回 ⇒ 要 key 卡片出现」这一对关系，**没有**区分撤回的具体分支（健康探测不通过，还是 overlay 整行替换配置导致该包失效）；钉死只需再加一条 `expect(ctx.llm.listConfigurableProviders())` **同日已补并复跑，仍 1 passed**：死端口下 `listConfigurableProviders()`（声明面）**仍含** `taiji-local`、`listProviders()`（注册面）不含 ⇒ **撤回发生在就绪探测这一层，`llm-taiji` 包仍挂载并声明该路由（dormant）**。这条排除的是「包没挂载／声明被抹掉」那一支；config 行被 overlay 替换是实验手段本身，不是缺陷。

## 4 · 需要所有者拍板的三条：D3／D2／R4（按能声称的最强结论排；原 R1 已并入 R4）

* **R5｜交付面里剩下的上游品牌名（2026-09-26 新发现，G2 改名的漏网面）**：全仓交付源码里 `deepseek-harness`／`deepseek-ai` 共 **26 处／19 个文件**，但**大部分不能改**——`@deepseek-ai/libreoffice-kit` 是第三方包名（`packages/document/office-to-pdf/src/index.ts:7`、`apps/desktop-host/src/office-engine.ts:27-32`、`webworker-runtime` 的替身表），`x-deepseek-harness-user-id`／`-session-id`／`-compact` 是服务商侧认的 HTTP 头（`packages/llm/llm-deepseek/src/adapter.ts:121-123`）。**真正属于产品身份、且改法需要你点头的只有三处**：① **用户磁盘上的默认工作区目录** `packages/api/workspace-controller/src/default-directory.ts:78` （`~/Documents/deepseek-harness/<名称>`——本轮 `github-ready-review` 的插桩读数里就印着这个路径，是装机后第一眼能看到的东西）；② **ACP 握手里的 agent 名** `packages/acp/acp/src/index.ts:182,378`（`deepseek-harness-acp`，对端可见）；③ **归因元数据** `packages/llm/llm/src/attribution.ts:41,43`（`product: deepseek-harness` ＋ 上游 GitHub URL）。⇒ **代价更正（同日读码，我先前把 ① 估高了）**：`defaultWorkspaceDirectory` 全仓**只有一个调用点**（`packages/api/workspace-controller/src/index.ts:108`，在 `initializeDefault` 内），且**没有任何代码拿这个路径做比较**（grep 只命中它自己与它的测试）⇒ 改 ① **只影响之后新建的默认工作区**，既有安装的工作区存的是各自的绝对路径、继续有效，**不需要迁移**。唯一会跟着红的是 `packages/api/workspace-controller/tests/default-directory.host.spec.ts`（它按字面断言这个目录名，属"测试随有意变更一起改"）。②③ 仍是对外可见身份的改变。**三条都是一行级改动，我不自行改，等你一句方向。**

* **R1｜打包实跑在哪台机器**：**这条的前提已被同日实跑否证，实际并入了 R4**。原写法假设"要先装 pnpm 与桌面工具链"；事实是 corepack 无需全局安装即可跑锁定的 pnpm 11.7.0（08 §6 ④），且本机已实跑 `package:desktop:dir` 一次——**它在拉任何二进制之前**先撞 R4 的 `.env.windows`（`build:desktop` 侧 rc=0）。⇒ 真问题不再是"哪台机器能跑"，而是"填不出 R4 那组产品值就一步也往前走"；可做的最强动作相应变为：**R4 落定后在本机跑 `package:desktop:dir` 的 unsigned 变体**，拿到"能出 dir 产物"的证据（不必换机器、不必降级到只跑 `build:desktop`）。
* **D3 默认 provider 怎么定**：(a) 装机默认＝Taiji（改 profile 配置，产品默认变更）；(b) 默认仍是上游 provider，Taiji 作为可选组（现状）；(c) 首启动做一次引导选择。**这条决定"Taiji provider 为默认"这句话能不能说**，我不自行改。
* **D2 运行时怎么随包**：随包附带并拉起／首启动引导用户启动／文档要求自备。**这条决定"一条命令装起即用"能不能声称**，也决定客户端版的工作量排序。**本轮读原文后已把选项收敛**：(a) 扩 `pythonPackages`＝撞发布物体积上限（基本排除），**(b) 给后端开独立分发通道（建议按此估工）**，(c) 要求用户自备并启动（兜底，但要改判据口径）。详见 §1 的 D2 行。
* **R4｜桌面打包需要一份本地 `.env.windows`，里面是产品决定**（2026-09-26 实跑 `package:desktop:dir` 撞出来的第一道门，在拉任何二进制之前）：`.env.windows.example` 要求填 `DSH_DESKTOP_APP_ID`（**示例默认值是 `com.deepseek.harness`＝上游身份，fork 要用就得改成自己的 app id**）、自动更新环境 `DSH_DESKTOP_AUTO_UPDATE_ENV`、**强制更新端点** `DSH_DESKTOP_MANDATORY_UPDATE_TEST_ORIGIN`／`_PROD_ORIGIN`（现在为空）、更新回退页 JSON（含 `allowedAuthOrigins` 登录白名单）、可选 `DSH_DESKTOP_NPM_REGISTRY`（**"for the bundled dsh runtime install"**），以及签名三件套（`WINDOWS_CER_FILE`／`SIGNTOOL`／`KEY_CONTAINER`，本机构建可走 `:unsigned` 变体）。⇒ **客户端版（G5 后半）真正的前置是三件**：① 定下 Taiji 版的应用身份与更新端点（或明确"客户端不做自动更新"）；② 回答"包怎么到达安装现场"（发布 `@taiji/*`／指镜像／用 packed tarball，`release:pack` 是否喂给这一步尚未查清）；③ 再谈签名链。这三件都不是我能在只读清点里替你定的，但没有它们 `package:desktop:*` 连跑都跑不起来——**这条比"缺 electron"更靠前，也更贵**。

## 4.5 · D2 的估工（同日读码所得；**它推翻了我先前把选项 (a) 判为"基本排除"的依据**）

先前写的是"(a) 扩清单＝基本排除；(b) 独立分发通道（建议按此估工）"。把桌面载荷这条链读完后，**这个排序的前提站不住**：

* **载荷通道是现成的，而且是打包期下载**：`apps/desktop/scripts/prepare-primary-runtime.ts` 调 `scripts/primary-runtime/prepare.ts`，
  按 target 组装 runtime 目录，**带下载缓存**（`cache: paths.downloads`），产出 `runtime.json`（版本／平台／架构／Python 发行版映射／摘要）。
* **安装与升级语义也已实现**：`apps/desktop/README.md:35` 写明"匹配的既有安装会被复用；依赖或归档变化时**先完整暂存再整目录替换**，替换失败保留旧安装；
  Windows 上解释器仍在跑时可能拒绝替换"。⇒ 这不是"塞不进发布物"，而是**一条已经跑通的分发＋替换机制**。
* **要加包，落点是三处声明面**：`packages/skill/tool-workspace-dependencies/src/index.ts:107` 的 `['numpy','pandas'] as const`、
  同文件 `:251` 工具描述里那句"Python 含 numpy、pandas、python-docx、…（这是**面向模型的工具说明**，改包必须同步改它）"、
  以及打包期冒烟 `scripts/primary-runtime/prepare.ts:183`（硬编码 `import … numpy, pandas` 并断言）。
  外加该包双语 README 与生成目录 ⇒ **属"改上游一处声明面"的 fork 补丁（登记 H 项），不是架构上做不了**。

**我先前当作前提、这次找不到证据的一句话要降级**：**"发布工作流强制 public-index 体积上限"**——我在 `.github/workflows/*.yml` 里按
`size limit`／`MAX_`／`public-index` 等关键词**没有搜到任何体积上限**（只搜到 `DSH_*_MAX_WORKERS` 这类并发变量）。
⇒ **这条要么另找出处（ADR／发布文档），要么从 D2 的裁定依据里划掉**；我先前"torch 量级塞不进去"的判断有一半是压在它上面的。

**仍然成立、且是真正成本所在的那半句**：载荷解决的是"**文件在不在盘上**"，而 `taiji-local` 要的是一个**开机即监听 `127.0.0.1:8000` 的常驻进程**
（默认端口 `packages/llm/llm-taiji/src/defaults.ts:14`）。`apps/desktop/src` 里**没有任何 `python` 字样** ⇒ 载荷里的 Python 现在只被工具按需调用。

**好消息是常驻这块也有现成接缝**：`apps/desktop/src/backend-controller.ts:1-24` 是一个**与具体后端无关的泛型生命周期控制器**
（`DesktopBackendController<Host extends DesktopBackendHost>`，状态只有 `starting`／`ready`／`error`，`start()` 的契约就是"子进程接受应用请求后的就绪"），
node 后端就是它的一个实现（`apps/desktop/src/host-process.ts:3,139` 用 `spawn` 起子进程）。
⇒ **(b′) 的真实工作量＝** 再写一个 `DesktopBackendHost` 实现（用载荷路径起 Python 后端）＋ 接就绪探测 ＋ **补控制器现在没有的东西：崩溃重启／退避与端口占用处置**（该文件里搜不到 `restart`／`backoff`）。

**因此 D2 收成一个二选一，价格已明**：
* **(b′) 载荷随包＋桌面拉起常驻进程**：三处声明面改动（＋双语 README＋生成目录）＋ 一个 `DesktopBackendHost` 实现 ＋ 重启/退避/端口策略 ＋ 首启时序
  （`taiji-local` 就绪失败时**静默撤回路由**，见 §3.5 与 H3o——所以"起了但没就绪"在表层是看不见的）。**这是唯一能让"装起即用"与"Taiji 为默认"同时成立的路径。**
* **(c′) 要求用户自备并启动**：零代码，但**交付判据那句"装起即用"要改写**（装机首启看到的是"存在但不可路由"的 Taiji 行）。
* 原 (a) 不再单列——它就是 (b′) 的前半段。

**待所有者拍的只剩一句**：**(b′) 还是 (c′)**。若 (b′)，还需要一条产品口径：**后端崩溃/端口被占时桌面表现成什么样**（现在控制器只有 `starting`／`ready`／`error` 三态）。

## 5 · 我可以在裁定前继续做的（无需批）

* ~~把 `prepare:desktop` / `release:vendor` 的**包来源**查清：本机 gitignored 产物 `apps/desktop/.desktop-build/development/project/desktop-runtime.json` 里出现 `@deepseek-ai/dsh-agent-default-model` 这类**改名前的包名**，而已跟踪文件里 `@deepseek-ai/dsh-` 为 **0 处**（2026-09-26 复算）⇒ 疑点：桌面暂存可能拉的是**已发布上游包**而非本 fork 产物。~~ **【同日已否证】**：该产物的 mtime 是 **09-22 16:26**，而改名提交 `88ad3040e`（G2 rename，5206＋43 文件）落在 **09-22 18:51** ⇒ 产物比改名**早 2.5 小时**，它是**改名前的本机构建残留**，不是"桌面线拉已发布上游包"的证据。副产品事实：这条暂存链在 G1 期就真跑通过一次（`schemaVersion 1`、`release.version 0.1.7-alpha.1`、`nodeVersion 24.18.1`、**`sharedPackages` 1495 项**）⇒ 桌面打包的**起点机制是存在的**，只是这台机器上的产物是旧的。
* **在真实 web 装配里查"装机默认是否可输入"**（零改产品码的只读探针，与判据⑤ 同一条取数路径）：起一次不带 settings 的干净 profile，读 `ctx.agentDefaultModel.currentSelection()` 与 `listProviders()` 的交集，判 `deepseek-official/deepseek-flash` 是否真在服务、composer 是否可输入。它直接决定 §4-D3 三个选项里哪个是"最小可用交付"，且**不需要任何裁定**就能先拿到读数。
* ~~把 web-first 的"启动即得 Taiji 组"写成可复验的验收脚本（复用判据⑤ 已有的 Playwright 取数路径），为 D2 的三个选项各留一条读数口径。~~ **按同日教训改口径：不新写脚本**——`apps/web/tests` 里已有整条 Playwright＋真实交付装配的 lane（见 §3.5 的 keyless 参数），**再立一份验收脚本只会多养一套判据**。改成：把这条 lane 在本机跑起来（缺的是 `apps/web/dist` 与钉住的 `chromium_headless_shell` 二进制——**dist 本轮已 build 出来 rc=0，浏览器二进制本轮正在装**；无浏览器依赖的那半已经真机跑过，见 §3.5），D2 三个选项各自的读数口径挂在它上面。
* **本件自身的复验纪律（D2 一条估计连改三次的教训）**：先写"要造机制"，读到打包链已有 Python 通道后改成"只是接清单"，再读到该通道**载荷随发布物走、release workflow 强制体积上限**后才定成"接不下、要开新通道"。⇒ **估成本必须读到约束条款（体积上限、是否常驻、许可面），不能只读到"机制存在"就下结论**；引用设计记录时先读原文（本行就是补读原文后重写的）。
## 6 · 下一轮队首（交接，2026-09-26 本轮预算用尽处）

本轮把 M6/G5 的 D3 从静态推断推到真机读数并把因果钉死（提交 `676c2fea`／`3951a5c9`／`fd1d4b39`／`8f204250`／`a330d51a`／`b0a49435`／`f40c040c`／`2d1228bb`／`b9961315`）。下一轮按能声称的最强结论排序：

1. ~~把 §5 第 7 条的最小面扩成「运行时在场／不在场」两套前提~~ **同日已完成（H3p ＋ H3q）**：4 个 keyless 文件的 11 条用例现在全绿（起点 8 条红），工作区流程改走宿主侧 `connectFreshWorkspaceViaHost`；残余红只剩**金样与计数差异**那一类，等下面的金样裁定。新登记的两条要接着判：08 §6 ⑭（回放型 lane 的录制件不入库，要 `test:snapshot:record` ＋ key 重录）与 §6 ⑬ 的教训（新增 e2e 要同登记 exclude）。
2. **需裁定（D2，拍了它 D3 才有意义）**：本轮 overlay 实验已把口径钉过一次，但**第四批实测更正了它**：运行时不在场时 `taiji-local` **只是从注册面撤回，设置里的行仍在**（声明面 `listConfigurableProviders()` 还带着它）；整行消失要把装配行 `disabled: true`。所以装机首启用户看到的是一行**存在但不可路由**的 Taiji，而不是「没有 Taiji」——D2 的三条选项仍成立，但措辞要按这个分层重写；「文档要求自备」等于放弃 G5 判据里「Taiji provider 为默认」这句。请在三者里拍一个，并给出可接受的包体积涨幅上限。
3. **需裁定（R4，比缺 electron 更靠前）**：`apps/desktop/.env.windows` 的产品值——应用身份（示例仍是上游的 `com.deepseek.harness`）、自动更新环境与强制更新端点、是否声明「客户端不做自动更新」。这三件不落定，`package:desktop:*` 连跑都跑不起来 ⇒ 「打包」这半句取不到任何读数。落定后本机可直接跑 unsigned 变体（corepack 已实证可用，不必换机器）。
4. **挂账但不自动触发**：G4 判据③ 真机训练回合（真训练→进度流→停止→检查点复验→真续训→真激活），等 R2 收束与所有者放行，不因「继续推进」字样而起。
5. **方法债（别重复踩）**：要区分 provider 的「撤回原因」，直接复用 H3o 那条 lane 的形状——**声明面 `listConfigurableProviders()` 与注册面 `listProviders()` 分开断言**，本轮就靠这一对把「包没挂载」那一支排除掉的；不要再自建第二判据（本轮已因此写错过一次「现有 lane 测不到」）。

**收口读数（本轮最后一次全量叶批，2026-09-26 21:13）**：doc-sync 复跑＝**43 叶／唯一红 `docs-site-projection`**（即 `project-doc-site.spec.ts` 需 `symlink()` 而本机 EPERM 那条已登记环境红，见 08 §6 ④）⇒ **今日全部改动**（含 H3o 新增的那条 lane 与它的 overlay 行）**未引入新红**；新增文件另过 oxlint **0 error／0 warning**（90 条规则）。**一处口径提醒**：本仓没有可直接整体调用的聚合入口跑这 43 叶，我用的是仓外批跑脚本，它把嵌套的 `verify-doc-site-fragments` 双计成 44 叶——**以仓库自身聚合报告的 43 为准**。
**第四批（同日，2026-09-26）：把「无可用路由」写成显式前提，8 条红降到 2 条**：不再让这 4 条 keyless lane 的前提取决于本机恰好有没有跑着运行时。做法是新增 `apps/web/tests/taiji-row-absent.overlay.yml`（内容 `- id: llm-taiji` ＋ `disabled: true`），在 `apps/web/tests/scaffold.ts` 导出 `TAIJI_ROW_ABSENT_OVERLAY`，四条 lane 的 `launchWebScaffold(...)` 统一带上 `extraOverlayPath`（`onboarding-native` 与它原有的 cordis.patch.yml 合并成数组）。**读数**：同 4 文件 `Test Files 2 failed｜2 passed`、`Tests 2 failed｜9 passed（11）`，107 s ⇒ **红从 8 降到 2**，验证了「这些红的前提被运行时会话污染」这一判断。**关键更正**：只把 baseURL 指到死端口时，设置里的 Taiji 行**依然出现**（注册面撤回、声明面仍在），必须 `disabled: true` 才整行消失——第三批我据此写的 D2 口径已在上面就地改掉。**剩余 2 条（未归因）**：`deepseek-messages-settings > offers one DeepSeek card...` 等 `getByRole('textbox',{name:'选择工作区'})` 32 s 超时；`onboarding-deepseek-config > configures arbitrary DeepSeek models...` 33 s 超时（等待对象待取）。这两条不再像是运行时污染，下一条线索是工作区选择与「删掉选中模型后」的表层差异。**同日对照实验把范围改大了（重要）**：跑一条同样调用 `connectFreshWorkspaceZh` 但不带本轮 overlay 的对照 lane （`access-confirmation.e2e.ts`），它在 helper 的**下一步**红——`选择工作区目录` 对话框 10 s 不出现；而 `apps/web/tests/scaffold.ts:660-665` 是**无条件**把页内 browse picker 钉住的（`{ id: directory-picker, disabled: true }` ＋ insert 两行），装配也没报错。⇒ 这不是"某条 lane 忘了带 overlay"，而是**页内目录选择器在本 fork 实际弹不出来**，它横跨 12 条共用该 helper 的 lane。**"剩余 2 条"这个说法要改，但不是我当时写的那个方向**：这条 lane 在 CI 里 0 引用（`.github/workflows/` 中 `test:web`／`run-web-snapshots` 出现 0 次，只 `playwright install chromium` 给别的门用），所以它没有上游拥有的绿色基线 ⇒ 红既不能算 fork 回归也不能算已修；要用它当门，先做一次失败瞬间的 ARIA 判别（08 §6 ⑫ 已按此降级）。**本轮批处理自己的两处缺陷（已修，记法在此）**：统一正则插入造成 `extraOverlayPath` **重复键**（一处文件）与**后写覆盖前写**（`onboarding-native` 里 `...desktop ? {} : { extraOverlayPath }` 会盖掉新键，属会静默丢前提的那一类）；另有一条 max-len 149＞140。跑之前逐文件核对落点形状后才修掉。


## 6.5 · 第五批之后的队首（同日收口，接手的人从这里开始）

**（2026-09-27 续刀）⑫/⑮ 家族两 lane 修进**：goal-bar＝按名点开 New Session 行（hero composer 不处理 /goal）＋把"agent 计数=1"换成"恰好一个 agent 持有 armed goal"（自动打开机制会多起 agent）⇒ **单跑两次 2/2 全绿**；sidebar-right＝settled beforeAll 按标题重开会话行 ⇒ 7 红降 2；余 2 条经**探针 v4**（复刻 lane 完整 beforeAll）定案＝新页会话行标签仍是消息标题、但藏在折叠的 `Default workspace` 组里 ⇒ 组展开＋选行后 **sidebar-right 15/15 全绿（两次复跑）**，该 lane 归零（08 §6 ⑱）。

**（2026-09-27 收口）队首四条已全部判读，读数与证据链在 08 §6 ⑱**：① 75 条全并发批（baseline79）补上 agent-team-panel 批内读数＝**4 条全绿**，转换成立；文件计数因两条翻动（access-confirmation 偶发红、sidebar-subagent-activity 翻绿）仍是 32，**总数不是稳定指标**。② markdown-images 两层定案：选行层已修（懒构建树＋Ungrouped 默认折叠＋"New Session" 占位——按名字锚定展开、按 basename 选行，`045be75e` 的"没有会话行"就此更正）；剩余红＝**产品层 Windows 限制**（`ui-chat/AssistantMarkdown.tsx:21` 的 `localPathMediaUrl` 拒绝盘符路径 ⇒ 绝对路径图片不发 /api/file 请求，金样期望 img；preview 侧 `path-images.ts` 已有正确样板）——**转所有者裁定，修法＝聊天侧对齐 preview 侧**。③ 五条超时 lane 逐条归因完成：agent-preset-selection（G2 品牌化金样差异＋级联）、models-settings（Taiji 行金样差异＋级联）、plugin-manager（{{home}} 归一化未命中 ⇒ 错误态 bundle 未清除 ⇒ 三条级联）、goal-bar（helper 连上但主区停在 hero，/goal 进了 hero composer 无效）、sidebar-right（自动建/自动打开的空白 New Session 顶掉 lane 会话）——**每条一个根，全是已登记类别的新实例，无新未知类**。④ basename 样板落 markdown-wide-table（主流程＋hidpi，后者接受 basename/标题双标签）＝**单跑 10/10 全绿**；其余旧 helper lane 全绿，刻意不动。

1. **⑮ 的对照同日已做完（见 §7.2 末）**：`firstUse: true` 档下树逐字相同、标题两档都不进树 ⇒ **⑮ 是真实的**，
   "默认工作区多占一行"这个替代解释已被否证。**剩下的那一层未判**：缺失发生在宿主索引还是前端渲染，
   需要一条**宿主侧列出会话标题**的读法（我第一次试的 `workspaceRegistry.list()` 写法报错，未跑通）。
**（2026-09-26 收口时更新）第 1 项已完成、且被扩展**：`agent-team-panel` 经转宿主侧 helper 实验证实同因，两次单跑 `4 passed`（08 §5 末段）；⑮ 的"层次"问题也已改述为"头记录不携带标题"。
**下一轮的队首（按序，全部不需裁定）**：① **重跑一次 75 条全并发**，把 `agent-team-panel` 转换后的批内读数补上——目前"32 → 31"只有单跑证据，**批内未验**；② `markdown-images` 单独判（它不属"点错行"一族，树里没有会话行）；③ 剩余超时 lane 逐条取证（10 条、10 种不同等待对象，"共同因"已否证）；④ 把 basename 选行样板套到其余仍用旧 helper 的 lane（现存调用点见 08 §6 ⑫）。**
2. 若 1 判为"位置假设"：把 DONE 一族 6 条 lane（7 条断言）**改成按内容选行**（`getByRole('treeitem', { name: … })`），
   **不要**改成另一个位置索引——那只是把同一个假设换个写法。**这一步之前不要动那 6 个文件**。
3. **18 条超时逐条取证**（文件清单：`node .dsh-sbx2/analyze-web-log.mjs .dsh-sbx2/baseline76c.log` 的 `[timeout]` 行）。
   **自改写 lane 这个解释已排除**（去掉 `hmr-live` 的 A/B 红绿不变）；**负载这个解释仍未排除也未证实**——唯一那次 11 条小批重测被并行的 build 污染而作废（08 §5 第 7 条同日更正）。
4. **金样这件事已经定价完**（§7／§7.1／§7.2）：refresh 能救的只有 **15 条整棵 ARIA 快照差异**；
   (乙) 分隔符**已实施为 H3u**；剩下要裁的只有 (甲) 该不该在非本机 refresh、(丙) 时区与 shell 两处在平台条件 skip 还是改断言。
5. **批跑口径三条**（已写进 08 §5 第 7 条，别再重新发现一遍）：完整 build 必须在跑门**之前**；批内不含 spawn watcher 的 lane；
   起止 mtime 要存档，且**任何"谁改了产物"的问题必须在下一次 build 之前问**。
6. 仍卡在所有者手上的：D2（运行时怎么随包）／R4（`.env.windows` 那组产品值）／R5（三处产品身份）／(甲)(丙)／重录回放件（要 `DEEPSEEK_API_KEY`）／符号链接那条测试的处置。**判据③ 等 R2 收束，不因"继续推进"字样自动开跑。**


## 7 · 金样 refresh 会烤进什么（2026-09-26 逐条取证，给"要不要 refresh"这条裁定用）

跑那 10 条红（`smoke-real` 因会另起 CLI 进程未列入本轮），把每条 ARIA 差异归到具体 lane 后，**差异只有四种来源，而其中一种不该进基线**：

| lane | 差异内容 | 类别 | refresh 的后果 |
| --- | --- | --- | --- |
| `agent-preset-authoring` | `扩展 DSH 的能力` → `扩展 Taiji Harness 的能力` | **fork 有意变更**（G2 品牌化） | 烤进去＝正确，且这是唯一让它变绿的做法 |
| `models-settings-recovery` | 模型列表多出一行 `Taiji（本地运行时）` ＋ 其编辑按钮 | **fork 有意变更**（我方 `llm-taiji` 在装配里） | 烤进去＝正确，但**这台机器上录的会连带下一行的宿主差异** |
| `plugin-install-registry` | `安装位置：{{cwd}}/.dsh-home/profiles/scaffold` → `{{cwd}}\.dsh-home\profiles\scaffold` | **宿主路径分隔符**（Windows） | **不该烤**：在本机 refresh 会把反斜杠固化进基线，macOS/Linux 上立刻变红 |
| `github-ready-review` **【已定案，见本节末】** `expect(ctx.agents.list()).toHaveLength(before + 1)` 实得**多一个 agent**（`e2e.ts:175`） | **唯一仍未归类的一条** | 候选：我方某个包在启动时挂了 agent，或同文件前一条测试的 agent 未释放；要单独判，**refresh 与它无关**。**同日再收窄（单跑复现）**：把这一条**单独跑**仍然红（200 ms，`expected [ReactLoopAgent, …(1)] to have a length of 1 but got 2`）⇒ **排除跨文件状态泄漏**；真实形状是"一次 ingress 之后存在 2 个 `ReactLoopAgent`，而 `before` 记的是 0"。下一步的第一个探针很具体：把这两个 agent 的 `session.header.cwd`／创建栈打出来，看多出来的那个是谁挂的（我方六个包里 grep 不到 `agents.create`，所以候选在核心/其他包，或 `before` 取样点晚于第二个 agent 的创建）。**未判** |**同日再排除三项（负结果也是账）**：① 中途那次 `toHaveLength(before)` 断言**是过的**，失败只发生在 `turn/end` **之后** ⇒ 第二个 agent 是回合结束时才被拉起来的；② 我方六个包里 grep 不到 `agents.create`／`ctx.agents.` 用法，`session-memory-taiji` 只监听 `agent/turn-stopping` 并使用传入的 agent（`src/index.ts:240`）⇒ "我方包在回合结束时造 agent"这个假设**被否证**；③ 单跑复现排除跨文件泄漏。⇒ 候选收窄到核心／subagent／webhook 路径，**要定位必须插桩打印两个 agent 的 `session.header.cwd` 与创建栈**（下一轮的独立小实验）。**同日再查两个嫌疑，都被否证（负结果入册）**：① fork 的 G2d（`5f5158ca4`）确实在会话生命周期上有非品牌改动，但它在 session-controller 里**只是新增 `await this.agents.closeSession(sessionId)`**（关闭时收起 agent，不是创建），而改动主体所在的 `packages/workspace/workspace/src/index.ts` **通篇不引用 `agents`**（grep 只命中一条注释）；② `ctx.agents.get(id)` 的签名是 `Agent | undefined`（`packages/core/agent/src/index.ts:566`），**只取不造**，所以"某处 liveness 复查顺手 get 出 agent"也不成立。⇒ 现在最可能的解释换成了**产品行为**：web 客户端在会话列表变化时自动选中并打开新会话，宿主因此为它起 agent（本 lane 的浏览器是连着的）。**下一个具体探针**：`vi.spyOn(ctx.agents, "create")` 打印调用栈，一眼就能看出是谁、因何创建第二个 agent；若确认是自动选中，则这条红属"lane 的预期与当前表层不符"，**不是缺陷**。**同日按这个探针跑了一次，结果是不确定（如实记）**：在 webhook 之前对 `ctx.agents.register` 装 spy（`vi.spyOn` 未报错、测试仍在同一处断言失败），但 **spy 一次都没被调用**，而结束时确实有 2 个 agent ⇒ **`register` 不是这条路径上的创建缝**，要挂得挂在工厂侧的 `createAgent`／`resume`（`packages/core/agent/src/index.ts:190,202`）。**同日找到具体嫌疑代码（不必再插桩就有靶子）**：G2d（`5f5158ca4`）在 `packages/client/ui-workspace/src/client/navigation.ts` 改的正是**"选中→打开"的时序**——它把 `archivedView` 的赋值提到 "任何 selection/retention 写入之前"，注释自己写着 *"a reconcile these writes trigger must see the view as deliberate, or it reads the new selection as an archive event and **closes it again**"*。而**打开会话才会起 agent**，所以"多出来的那个 agent 属于可浏览工作区会话"与这段改动是同一条路径。**下一步的判定实验（比插桩便宜且决定性）**：把这条 lane 的浏览器页在 webhook 之前关掉（或改成不建页），若 `agents.list()` 就只剩 1 个 ⇒ 第二个 agent 是**客户端连接后的自动选中/打开**造成的，属"lane 的预期与当前表层不符"；若仍是 2 个 ⇒ 才需要往宿主侧查。当前证据下我倾向前者，但**没有做这个实验就不算定案**。**实验已做，定案（2026-09-26 07:17）**：把这条 lane 的浏览器页在 webhook **之前**关闭后跑，读数 `PROBE agents-after=1 before=0`——**宿主侧每次 ingress 只起 1 个 agent，正是 lane 期望的值**；之前那个第 2 个 agent 完全来自**有客户端连着时，新建的可浏览会话被自动打开**（打开才起 agent）。⇒ 归类完成：**不是宿主缺陷，也不是金样问题**，而是"客户端在场会自动打开新会话"这一条表层行为与该 lane 的预期不符。**fork 还是上游 drift 仍未分**（嫌疑代码是 G2d 在 `ui-workspace/navigation.ts` 改的选中→打开时序），但**这条对交付本身是个事实**：webhook 建出来的新会话会被自动打开并起 agent，不是躺在列表里等用户点。插桩当场撤回，`apps/web/tests/` 树已确认干净。**插桩已当场撤回**（`git checkout` 后 `git status` 对 `apps/web/tests/` 干净、文件内 `PROBE` 计数 0）。**过程教训一条**：撤销命令的路径要按仓根写——我第一次写成了 `apps/web/tests/...` 少了 `taiji-harness/` 前缀，`git checkout` 直接报 pathspec 不匹配，探针因此在树上多留了一轮。
| `support-timezone` | `expected 'Asia/Shanghai' to be 'UTC'`（第 9 行的**对照页**断言） | **Windows 不认 `TZ` 环境变量**：lane 用 `chromium.launch({ env: { TZ: hostTimeZone } })` 造"宿主时区"，但本机 Chromium 仍报宿主真实时区 | **前提在本机不可能成立**（与 ⑩ 同类，非 fork 回归）；`newEnglishPage` 那条固定 `Asia/Shanghai` 的断言本身是好的 |
| `cold-blank-session`／`startup-auto-selection` | 等待对象根本不出现 | **lane 的状态假设与当前表层对不上**（§6 ⑮ 那条） | refresh 无效——它们不是快照差异 |

**（同日补判）**上表最后两行已从"未判"改成定案：`support-timezone` 是 Windows 不认 `TZ`，`github-ready-review` 是多出一个 agent（唯一仍未归类）。⇒ 11 条红的归属现在是：**2 类 fork 有意变更（该 refresh，但不该在本机）＼1 类宿主路径分隔符（该 normalize，不该 refresh）＼1 类宿主时区（本机前提不成立）＼2 条 lane 状态假设（refresh 无效）＼1 条待判＼其余为同类快照差异。**

**给裁定的三句话**：① 前两类**该** refresh，但**不该在本机做**（第 3 类会一起被烤进去），要么在 macOS/Linux 上 refresh，要么先把那两处宿主依赖值 normalize（`{{cwd}}` 已经是占位，分隔符却没有）；② 第 4、5 类要先各判一次，别混进 refresh；③ 第 6 类 refresh 救不了，得改 lane。⇒ **"refresh 与否"不是一个开关，而是三件事**。



## 7.2 · 那 22 条"普通断言红"不是金样差异，其中 7 条（6 条 lane）看着与已知的登记阻塞同源

用分账器（`.dsh-sbx2/analyze-web-log.mjs`，按每个 `FAIL` 块的**第一条错误行**分类，不读宿主 stderr）重算 `baseline76.log`：
79 条失败用例＝**超时 26 ＋ 快照/ARIA 差异 17 ＋ 普通断言 22 ＋ 运行时错误 5**（合计 70，余 9 条错误行形状未识别）。
⇒ **更正我今天写进 08 §5 的"金样/ARIA 差异 39 条"**：那 39 是"快照差异 17 ＋ 普通断言 22"的和，**只有 17 条是整棵 ARIA 树的差异、才是 refresh 能救的**；
22 条普通断言里最大的一族是 **`expected +0 to be 1` 共 7 条，横跨 6 条 lane**
（`clickable-links-gallery`、`markdown-cjk-strong`、`markdown-images`、`markdown-inline-code-links`、`math-rendering`、`produced-file-mentions`），
它们的等待对象都是**同一个东西：会话正文里那条 DONE 哨兵文本**（各 lane 自带的 `*_DONE` 字串，15 s `expect.poll`）。

**取证方式与读数（两张失败瞬间截图，非日志相邻文本）**：`markdown-cjk-strong` 与 `markdown-images` 在
**完整新鲜 build 之后**（`.artifacts/web-e2e-markdown-*.png`，19:51）的红**同形**——
侧栏 `Workspaces` 下只有 **`Default workspace`**，播了会话的那个临时工作区**没有作为工作区出现**，其会话行落在
**`Ungrouped`** 下且**行名是临时目录名**（`dsh-web-e2e-ws-XXXX`）而不是 lane 播种的标题；主区仍是 hero（`State at Its Utmost`）。
⇒ 这**不是** markdown 渲染缺陷：DONE 文本来自 lane 自己写进盘上的会话件（`source: {kind:'model', provider:'fixture'}`），
红的是**那条会话根本没进表层**。也**不是**产物过期：这两张图就是新鲜 build 上取的。

**我给出的第一个解释已被读码否证（同日，记下以免重犯）**：我先前写"这族 lane 走的是页内目录选择器那条旧 helper（`connectFreshWorkspace`，56 个调用点），
而该选择器在本 fork 开不出菜单 ⇒ 工作区从未被登记"。**打开 `markdown-cjk-strong.e2e.ts` 逐行看，它根本不调任何选择器 helper**——
`beforeAll` 是 `launchWebScaffold({})` → `seedSession(...)` → 启浏览器 → `goto`（`:95-104`），红的那段是在树里点两行（`:113-119`）。
⇒ **"未登记"这个说法对它不成立**，我把"截图里工作区没出现"直接接到了一个我没核对过的机制上。**又一次：先读那条 lane 的代码，再谈它的因**（同族第四次）。

**读码后剩下的两个候选（都还没判）**：① **⑮ 那一族**——`seedSession` 发生在宿主 `launchWebScaffold` **之后**，
而工作区的会话头部索引是**启动时建一次**（08 §6 ⑮），所以树里那行只有目录名、没有播种的标题，点开也就没有正文；
② **⑭ 那一族**——播种件被会话校验器拒收（`system/message ... must have system-prompt source`，`packages/core/session/src/index.ts:362-364`），
装载阶段就没东西可渲染。**两者区分办法（下一步做，一条 lane 就够）**：在 `seedSession` 之后**重启一次宿主**（或显式触发 re-index）再 `goto`——
若 DONE 出现＝⑮；仍不出现＝把当时的宿主 stderr 与 `readSessionHeader` 的返回值取出来，按 ⑭ 查。
**这条探针同日已跑（插桩在 DONE 轮询前打印 `[role=treeitem]` 的全部文本，跑完立刻 `git checkout --` 撤回，`git status` 已确认无残留）。读数把两个候选都替掉了**：
`rows = ["Default workspace", "New Session", "Ungrouped", "dsh-web-e2e-ws-xGXW9l" + "now"]`（`url` 是该次的随机端口）。
⇒ **树里有四行，而 lane 点的是 `.first()` 与 `.nth(1)`**——即 `Default workspace` 与 **`New Session`**，
**播种的那条会话在第 4 行（index 3），从来没被点到**。DONE 不出现不是"渲染不出"，是**点错了行**。
**两件事因此同时成立**：① **位置假设**（`nth(1)`）在本装配里错位——`Default workspace` 下面多出一个 `New Session` 行（`scaffold.ts:810` 会播种一个默认工作区，除非 `firstUse: true`）；
② **⑮ 仍然真实**——那行的标签是**临时目录名**而不是播种的标题（`CJK strong emphasis`），说明宿主启动后写入的会话件没有把标题投影进树。
⇒ **处置不是"改产品"，是给这族 lane 换成按内容选行**（例如 `getByRole('treeitem', { name: <标题或目录名> })`），
**但先别改**：② 那半句要单独判——"标题没进树"到底是**上游 lane 的既有假设**（它们在本仓从没跑过，CI 也不跑）还是**fork 的会话投影缺陷**，
需要一次"同样的播种件、在 `firstUse: true` 且只有一个工作区"的对照，**这条排在下一轮队首**。
**顺带钉一条方法**：这次是一个 `console.log` 加一次单跑（约 20 s）解决的，比我先前设想的"重启宿主／查校验器"两条路都便宜——**红在一句等待上时，先把它等待的那个容器打印出来**。

**对照实验同日跑完（两档单跑，插桩均已 `git checkout --` 撤回、`git status` 干净），队首第 1 项有答案了**：
① **`launchWebScaffold({ firstUse: true })` 下树逐字相同**（`["Default workspace","New Session","Ungrouped","dsh-web-e2e-ws-…now"]`）
⇒ `Default workspace`／`New Session` **不是 scaffold 播种出来的**（`scaffold.ts:838` 的播种条件是 `firstUse !== true` 且注册表为空），
**我上一条写的"默认工作区多占了一行"因此不成立**，位置错位不是 scaffold 造成的（那两行由 scaffold 之外的路径创建，最可能是产品自己的默认工作区初始化，**未核对**）。
② 改成**按内容选行**（`locator('[role="treeitem"]').filter({ hasText: /dsh-web-e2e-ws-/ })`）后 `waitFor` 超时，错误文本是 **waiting for ... to be visible**
⇒ **那一行在 DOM 里存在但不可见**（它在未展开的分组下）；而第一次插桩用的 `allTextContents()` **不要求可见**，所以"读得到四行"与"点不到"同时为真。
③ **两档下播种标题都没进树**（行文本始终是临时目录名，而不是会话件里的 `CJK strong emphasis`）⇒ **⑮ 是真实的**，不再能被当成"位置假设的副作用"打发掉。
**仍未判**：标题缺失发生在**宿主索引**还是**前端渲染**——我试的 `scaffold.ctx.workspaceRegistry.list()` 探针本身报错（不是产品失败），**下一步要用一条能列出会话标题的宿主侧读法，而不是继续改 lane**。
**处置边界（据此更新）**：把 6 条 lane 改成"先展开分组、再按内容选行"确实能让它们跑起来，**但那会掩盖 ⑮**；
⇒ 顺序必须是**先定 ⑮ 的层次，再决定 lane 怎么改**。

**机制的最后一格是插桩换个位置读出来的（同日，两档探针均已 `git checkout --` 撤回、`git status` 无残留）**：上一段"树里四行、`nth(1)` 落在 `New Session`"那个读数是**在两次点击之后**取容器的。
改成**在点击之前**取，读数是 **`before=["Ungrouped"]`——测试开始时整棵树只有一行**（折叠态分组）；点一下那一行之后才展开成四行。
⇒ **DONE 一族真正的机制是"树默认折叠成一行，而 lane 的 `.first()`／`.nth(1)` 是按展开后的行序写的"**：`.first()` 点到的就是那唯一一行，`.nth(1)` 命中的是展开后新出现的 `New Session`，
**播种会话所在的行从头到尾没被点到**。这比"多占了一行"准确，也解释了为什么这些 lane 单跑小跑都同形。
**另一半仍然成立**：展开后按标题找行（`hasText: /CJK strong emphasis/`）**可见性等待失败**（`titleRow=false`）⇒ 播种件里那条 `session/title` 没有成为行标签；
`ui-workspace` 自己写明行标题是**三级回退**"持久标题 → 项目 basename → 会话 id"（`packages/client/ui-workspace/src/client/contract/slots.ts:71`、`client/tree.ts:131-134`），
实得的是 **basename 那一级** ⇒ ⑮ 仍然真实。
**未判**：标题缺失发生在**宿主索引**还是**前端渲染**（需要一条能列出会话标题的宿主侧读法；我第一次试的 `ctx.workspaceRegistry.list()` 写法没跑通，那是仪器错误、不是产品失败）。
**处置边界不变**：先定这一层，再决定那 6 条 lane 怎么改——按内容选行能让它们跑绿，但会把 ⑮ 盖住。

**想用房侧读法定位 ⑮ 的层次，结果撞出一个新的仪器问题（同日；探针已撤、`git status` 干净）**：在 `beforeAll` 之后直接读 `scaffold.ctx`——
`ctx.sessions.list()` 得 **`[]`**、`ctx.workspaceRegistry.list()` 得 **`[]`**，而**同一时刻页面侧边栏显示着两个工作区和一行带 `now` 的会话**。
⇒ **但这个读法本身是错的，**我把"对不上"那句收回**：`ctx.sessions` 是**客户端作用域的访问器**（`packages/api/session-controller/src/client/sessions/manager.ts:403` 里它是 `await this.remote.session.list({})`），空数组不证明"宿主没有这些会话"；而 `seedSession` 走的是**另一个独立 `new Context()`**（`scaffold.ts:1486` 的 `seeder`，只为往 `persistenceRoot` 写件）——两者本就不是同一视图。
⇒ **⑮ 的层次仍未判，但判它不能靠 `ctx.*`**：要么按 `seeder` 那样另开一个 Context 去 `list`，要么把 `persistenceRoot` 下那份 JSONL 与表层实际取的那条 HTTP 响应对账。**教训（同族第五次）**：`[]` 是一个**访问器的返回值**，不等于"系统里没有"——用它下结论前先读该访问器怎么取数。

**⑮ 的层次同日判完了（改用 `seeder` 那条正确读法；探针已 `git checkout --` 撤回、`git status` 干净）**：另开一个 `Context` 挂 `JsonlSessionPersistence`（root 用 `scaffold.persistenceRoot`）调 `sessionPersistence.list()`——
盘上**确实有这条会话**，但它的**头记录里没有 title 字段**：
`header = {version:4, id:"markdown-cjk-strong-web-e2e", createdAt:1768449600000, cwd:"C:\…\dsh-web-e2e-ws-96fxyq", isSeeded:false, delegationDepth:0}`（外加 `revision`、`sizeBytes:797`）。
⇒ **树只读头记录**，头里没有标题，于是回退到 `cwd` 的 basename（就是 `ui-workspace` 那三级回退里的第二格）。
lane 追加的那条 `session/title` 是**事件**、不是**头字段**，所以从来没进过行标签。
**另一处相关事实**：scaffold 自己把 `session-title-llm` 关掉了（`scaffold.ts:630`，理由在 `:19`——它那个 fire-and-forget 的标题调用会与回放游标竞争），
⇒ **在这条测试面上没有别的机制会把标题写回头记录**。
**所以 ⑮ 的措辞要换**：从"工作区索引启动时建一次"改成更准的一句——**行标签的数据源是持久化头记录，而头记录不携带标题**。
**这不需要任何 fork 缺陷来解释**；但也**不能反过来说"上游一定是对的"**——这条面 CI 从不跑，没有"曾经绿过"的证据（同 §6 那条"没有上游绿色基线"的事实）。
**处置现在可以动了**：那 6 条 lane 的修法＝**先点那唯一一行把树展开、再按 `cwd` 的 basename 选行**（不是按标题）；
另一种做法是让 `seedSession` 把头记录写成带标题的形状——那动的是测试夹具，影响面更大，**不选**。
**并且改完要保留一条断言**："行标签当前是 basename"这件事要留在断言里，否则将来标题真的进了头记录，这条 lane 会**静默改变含义**。

**同日按此改完并实测（登记 08 的 H3v）**：6 条 lane 全部改成"展开后按 `cwd` basename 选行"，
**读数 `Test Files 5 passed｜1 failed (6)`、`Tests 6 passed｜1 failed (7)`、55.75 s**，oxlint 6 文件 0／0。
⇒ **这一族确实就是"点错行"**：改完 5 条整文件绿（它们后续的 strong／链接／公式／提及断言都过），
**这条面上"未归因的普通断言红"从 7 条降到 1 条**——`markdown-images` 越过了 DONE 断言、红在后面的一个 `locator.click` 超时，**属新形态、单独判**。
**我先前那句"改 lane 会掩盖 ⑮"要按结果修正**：⑮ 已被读数定性为"头记录不带标题"这一持久化事实（不是缺陷），所以改 lane 不会掩盖它；
而**"再补一条 basename 断言"这件事其实不需要**——我先前把它列为欠账是想多了：改动本身用的就是`filter({ hasText: /dsh-web-e2e-ws-/ })`，**将来标题真的进了头记录、行标签变成标题时这个定位器会直接找不到元素**，失败是响亮的而不是静默的 ⇒ **定位器本身就是那条守卫**，不必再加断言。
**这与 H3q 的既有结论不冲突**：`connectFreshWorkspaceViaHost` 算"已验证"是因为它的效果**在页面上看得见**（工作区出现、composer 就绪），不是靠 `ctx` 读出来的——
这条新读数反而说明**当时不该改用 `ctx` 断言**。
**我自己的两处仪器错误一并记下**：第一次写 `workspaceRegistry.list().then(...)`（它不是 Promise）、第二次把 `existsSync` 的 import 插在该文件不存在的那一行上；
**两次都是仪器错误，不是产品失败**。
**下一步（改法已定，未做）**：先确认 `ctx` 与页面服务是否同一进程（看 `scaffold.ts` 里 host 的启动方式），再选 ⑮ 的读法；
在此之前 **⑮ 的层次保持"未判"**，而 DONE 一族的机制（**树默认折叠成一行，位置点击却按展开后的行序写**）**已定案**。
**其余 15 条普通断言**形状各异（`plugin-config` 三条都是毫秒值对不上：`'120000'` 对 `'12000'` 两条、对 `'60000'` 一条；
`sessionless-header` 三条是计数对不上：`40` 对 `+0` 两条、`1` 对 `+0` 一条；另有 `'/go' to be '/goal '`、`Asia/Shanghai` 对 `UTC` 等），
**逐条取证前不要并进"金样"那一堆**。

**其中 3 条已定案（`plugin-config`，同日读码＋历史双向核对）**：该 lane 三条红（`'120000'` 对 `'60000'` 一条、`'120000'` 对 `'12000'` 两条）**是同一个因的级联**——
产品默认命令超时是 **120_000 ms**（`packages/shell/bash-local/src/index.ts:102` 与 `packages/shell/pwsh-local/src/index.ts:128` 两处默认值相同），
而 lane 第一句 `expect(timeout.inputValue()).toBe('60000')`（`plugin-config.e2e.ts:211`）就红，后面两条"填 12000 再读回"的用例因此从未持久化过 ⇒ 读回的是默认。
**归因为什么可信**：① 两个 shell 的默认一致 ⇒ **与本机用哪个 shell 无关**，不是宿主类；② `git log -S"60_000"` 在该文件为空、该文件自 fork 导入起只被 G1/G2 两次改名动过 ⇒ **不是我方改的**，是 lane 的期望与上游当前默认脱节。
**处置**：改 lane 期望属"动上游测试"，与符号链接那条同批裁定；**refresh 对它无效**（不是快照断言）。

**另有 4 条定案为"lane 的平台分支与本装配不符"（`sessionless-header`）**：**先记一次我差点写错的归因**——我第一版把 `expected 40 to be +0` 读成"本机是 win32、lane 假设非 darwin 就 0"，
读了源文件才知道 `platform` **不是宿主 OS**：该 lane 是 `it.each(['web','win32','linux','darwin'])`（`sessionless-header.e2e.ts:21`），
用 init script 往 `documentElement` 写 `data-platform` 来**仿真**四个平台（`:24-29`），断言"无选中会话时 `conversation.session.header` 槽位不存在"（`:35`）与"空头部高度＝`platform === 'darwin' ? 40 : 0`"（`:37`、`:44`）。
**四档全部红，且两两形态不同**（档名就在 FAIL 头行里，`it.each` 的参数是测试名的一部分）：
`web` 与 `darwin` 报 **`expected 1 to be +0`**＝槽位**在场**（`count()` 实得 1、期望 0）；`win32` 与 `linux` 过了槽位那条、报 **`expected 40 to be +0`**＝空头部**实得 40 px 而期望 0**。
⇒ 两件事各自成立：**（一）** 那个头部在本装配里**与平台无关地占 40 px**（四档同一浏览器同一宿主，只有注入属性不同），lane 的 `darwin ? 40 : 0` 分支因此只对 `darwin` 那一档"碰巧可能对"；
**（二）** 无选中会话时槽位**在 `web`／`darwin` 两档里仍然渲染**。**都不是金样**（refresh 无效），也**不是宿主类**（同一浏览器仿真的四档）。
**未判**：这是"上游 lane 过期"还是"fork 的头部布局变了"——要一次同装配的对照（把 `data-platform` 注入去掉看默认值）才能分，**登记为待办、不在本轮**。
**顺带一条读法教训**：`it.each` 的参数名在日志的 FAIL 头行里，**先 grep 头行再谈"哪几档红"**；我差点在这条上写出一段"本机 win32"的假归因（同族第三次：错误文本里的数字看着像宿主差异，实际要读断言那段代码）。

**给裁定的影响**：§7 那句"金样差异的真实规模是 39 条"要按 **17 条**引用；refresh 一刀切能救的面比我今天白天说的小一半以上。

## 7.1 · "先 normalize 再 refresh" 这条路的价格（同日实测，给 §7 那句"要么先 normalize"定价）

§7 的三句话里第 ① 句留了一个未定价的选项：「先把那两处宿主依赖值 normalize（`{{cwd}}` 已经是占位，分隔符却没有）」。本轮把它的**可行形状与半径**量清了：

* **公共出口只有一处**：`apps/web/tests/scaffold.ts:1540` 的 `normalizeAria(snapshot, workspaceCwd, age)`，被 `captureStableAria`（`:1606`）在套完 lane 自带 `replacements` 之后调用；它现有 **18 条归一规则**（uuid／时长／时钟／日期／`{{workspace}}` 基名／吞吐／tokens／`{{cwd}}` 本身），**没有一条管路径分隔符**。金样比对本身在 `compareOrRefreshGolden`（`:1675`），它是**纯字节比较**、不做二次归一。
* **通用替换 `\` → `/` 已证不可行**：入库的 **119 份**金样里有 **6 份**的反斜杠是**被展示的内容**而非路径——aria 里的 JSON 负载转义引号（`expected/clickable-links-gallery/ui.expected.md:41` 的 `\"command\"`、`expected/cordis-history/ui.expected.md:27`、`expected/models-settings-recovery/stored-error.expected.md:17`、`expected/plugin-manager/missing-bundle.expected.md:7`、`expected/session-archive-active/dialog.expected.md:7`、`expected/steer-all/replay.override.json`）。在公共出口做 blanket 替换会把这 6 份的内容改坏 ⇒ **这 6 条 lane 在所有平台都会变红**，不是本机问题。⇒ **"一行通用归一"这个便宜选项不存在**，别按它估工。
* **锚定式的真实半径＝1 份金样／1 行 lane 补丁**：含路径占位的金样只有 4 份（`{{cwd}}` 3 份 ＋ home 令牌 1 份），其中**会因平台翻转的只有 1 份**——`expected/plugin-install-registry/installed.expected.md` 的"安装位置"行（值由 app 自己 `join` 出来）。`markdown-images` 与 `reference-composer` 里的 `/` 分别来自 markdown 源文本与 cwd 本体，不经 join、不翻转。**修法**＝在 `plugin-install-registry.e2e.ts:75-77` 那条链式替换前加一条整串替换（把 `join(workspaceCwd, '.dsh-home', 'profiles', 'scaffold')` 直接映射成 `{{cwd}}/.dsh-home/profiles/scaffold`），**在 darwin/Linux 上是恒等式** ⇒ 不动任何入库金样、不引入跨平台漂移；代价＝1 行 fork 补丁 ＋ **0 份金样改动**。**同日已实施并登记为 H3u**：本机该 lane 从红转 `1 passed`（15.87 s）、oxlint 0／0、`expected/` 一个文件都没动。
* **另两处宿主依赖不是 normalize 能覆盖的**，要分开裁：① `support-timezone` 的 2 条断言（`expected 'Asia/Shanghai' to be 'UTC'`／`to be 'America/Los_Angeles'`）前提是「Chromium 认 `TZ` 环境变量」，Windows 上不成立 ⇒ 出路是 `platform` 条件 skip（动上游 spec）**或**把对照页断言改成取宿主真实时区；② `shipped-composition.e2e.ts` 的工具花名册**内联快照**把宿主 shell 烤了进去（本轮该 diff 为 1 行 `+ "pwsh"`，位于字母序 `present` 与 `read` 之间）⇒ 同属 skip-or-normalize，且它和 `plugin-install-registry` 的**归因方法要记清**：我是按"该 diff 之前最近出现的 lane 名"归的，不是按 FAIL 头行（ANSI 色码使头行匹配不上）。

⇒ **对 §7 裁定的净影响**：金样这件事不是"refresh 一刀切"，而是**三个各自独立的小裁定**——（甲）fork 有意变更（品牌字串、Taiji 行、我方 bundle 计数）**该** refresh，且**该**在非本机做；（乙）分隔符按锚定式 1 行 lane 补丁消掉，refresh 不该把它烤进去；（丙）时区／shell 两处属"环境前提在本机不可能成立"，与 `docs-site-projection` 的符号链接 EPERM 同族（08 §6 ⑦），三件可以一次裁定「在本机按平台 skip」。**在（乙）（丙）落定前做任何 refresh，都会把本机指纹固化进基线。**

## 8 · 裁定单（一次性总表；每项都是"问题一句／选项按能声称的最强结论排序／价格／不裁的后果"）

细节与证据在 §3.5／§4／§4.5／§6.5／§7／§7.1／§7.2，这里只收口成可回复的形状。

| # | 问题（平实一句） | 选项（强→弱） | 价格 | 不裁的后果 |
| --- | --- | --- | --- | --- |
| **D2** | 训练后端（Python＋torch 量级）怎么到用户机器上 | (b′) 随包并由桌面拉起常驻进程 → (c′) 用户自备并启动 | (b′)＝三处声明面加包（`tool-workspace-dependencies/src/index.ts:107`、`:251`、`prepare.ts:183`）＋一个 `DesktopBackendHost` 实现＋重启/退避/端口策略；(c′)＝零代码但**要改写"装起即用"** | G5 交付判据里"一条命令装起即用"与"Taiji 为默认"**无法同时成立**，桌面装配这条只能停在"存在但不可路由" |
| **D3** | 装机默认 provider 改不改 Taiji | 改（`bundle/base/cordis.patch.yml` 两行）→ 不改（保持 `deepseek-official`） | 两行配置，但属**产品默认变更**；已真机验过默认不是 Taiji（§3.5） | 不改则 D2 选 (b′) 也拿不到"默认即 Taiji" |
| **R4** | 桌面打包要的那组产品值 | 给值 → 明确"暂不打包" | `apps/desktop/.env.windows`：App ID（示例值仍是上游 `com.deepseek.harness`）、自动更新环境、强制更新端点、登录源白名单 JSON、npm registry、签名三件套 | `package:desktop:dir` 在下载任何二进制之前就失败，**打包实跑一步也走不了** |
| **R5** | 交付面上残留的三处产品身份改不改 | 三处都改 → 只改用户看得见的那一处 → 都不改 | ①默认工作区目录名 `packages/api/workspace-controller/src/default-directory.ts:78`（**已核：不需迁移**，只有一个调用点、无路径比较）；②ACP agent 名 `packages/acp/acp/src/index.ts:182,378`；③归因元数据 `packages/llm/llm/src/attribution.ts:41,43`。其余 23 处是第三方包名与服务商认的 HTTP 头，**不能改** | 装机后用户在文件管理器里看到的仍是 `deepseek-harness` 目录名 |
| **金样(甲)** | 上游那批 ARIA 金样要不要把我方差异烤进基线 | 在 macOS/Linux 上 `DSH_SNAPSHOT=refresh` → 本机 refresh（**会连带烤进宿主依赖值**）→ 不 refresh | 真实规模是 **15 条整棵快照差异**（不是先前说的 39），面是 139 文件／321 用例那套 | 这 15 条会一直红；但**先裁 (丙)**，否则本机 refresh 会把分隔符/时区/shell 固化进基线 |
| **金样(丙)** | 三处"环境前提在本机不可能成立"的红怎么处置 | 平台条件 skip（动上游 spec）→ 改断言取宿主真实值 → 保持红并标注 | `support-timezone`（Windows 不认 `TZ`）、`shipped-composition`（快照烤进宿主 shell `pwsh`）、`docs-site-projection`（符号链接 EPERM）——**同一族，建议一次裁** | 复验集里永远挂 1–3 条"环境红"，掩盖真信号 |
| **判据③** | 真机训练回合（真训练→进度→停止→检查点复验→真续训→真激活）何时开跑 | 等 R2 收束 → 现在开 | 需要独占运行时窗口；**已裁定不因"继续推进"字样自动触发** | G4 的复验集里这条一直无读数 |
| **回放件** | 缺的录制件要不要重录 | 重录（要 `DEEPSEEK_API_KEY` ＋ `test:snapshot:record`）→ 承认回放型 lane 不可当门 | 盘上那份是 09-22 本机录的，`.gitignore` 有 `*.jsonl` ⇒ **干净检出跑不出它们** | 若干 lane 长期"看着像红其实缺证据" |

**最小回复格式（照抄即可）**：
`D2=b′|c′；D3=改|不改；R4=给值|暂缓；R5=全改|只改①|都不改；金样甲=非本机refresh|本机refresh|不refresh；金样丙=平台skip|改断言|保持红；判据③=等R2|现在开；回放件=重录|承认不可当门`

**不需要裁定、下一轮我按序做的**（§6.5）：`agent-team-panel` 的 setup 双形态（先查 `Ready.` 为何有时 10 s 不来）→ 那条的可访问名判定 → 10 条超时逐条取证 → 把 basename 选行样板补到剩下用例。

