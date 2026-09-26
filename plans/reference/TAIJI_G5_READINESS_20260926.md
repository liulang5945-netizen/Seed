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

