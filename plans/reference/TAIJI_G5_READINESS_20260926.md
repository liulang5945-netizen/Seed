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
| D1 | "harness 自带机制" | 打包命令存在且可用，不需要我们自造发布链 | **已具备，且第一段已实跑**：`package.json` 里有 `build:desktop`／`package:desktop`／`package:desktop:dir`，以及 `package:desktop:win:x64`、`:mac:arm64`、`:mac:x64`（含 `:dir` 与 `:unsigned` 变体）——全部经 `pnpm --filter @taiji/dsh-desktop run …` 驱动。**2026-09-26 经 `corepack pnpm` 实跑第一环**：`apps/desktop` 的 `build`（＝`tsc -b && tsdown`）**rc=0**（末段产物 `lib/welcome/welcome.js 151.23 kB`，`✓ built in 362ms`）⇒ 桌面外壳在本机可编译。**仍未实跑**：`package:desktop:*`（要拉 electron 二进制与签名，且会写 `.desktop-build` 暂存树） |
| D2 | "一条命令装起即用" | 装完之后第一次启动不需要用户手工装配运行时/凭据 | **未达成，且缺口不在打包**：Taiji 侧一切读数都来自本机 8000 端口的 python 运行时（`llm-taiji` 走 HTTP，`life-controller` 轮询它）。打包产物里**没有**"把运行时一起拉起"的环节 ⇒ 装机后面板可开，但 Taiji 组不可用。这是一个**设计决定**（随包附带运行时／首次启动引导用户起／明确"需要自己起运行时"），不是缺陷修 |
| D3 | "Taiji provider 为默认" | 新装 Profile 的**默认模型选择**落在 Taiji 路由上 | **未达成，旋钮已定位到具体两行**：默认选择归 `@taiji/dsh-agent-default-model`（`packages/core/agent-default-model/src/index.ts:24-32` 的 `Config{provider, model, reasoningEffort?}`，三项都 `volatile`，实际值优先从 `settings` 读），而交付装配里**确实设了它**——`packages/bundle/base/cordis.patch.yml:82-86`：`config: {provider: deepseek-official, model: deepseek-flash}`。⇒ **装机默认是 `deepseek-official`，不是 Taiji**（我初稿写"没有任何 profile 设它"是错的，已在本文更正）。**再往下一层未测**：`deepseek-official` 这条路由在 web 装配里是否真有 adapter 服务、是否需要凭据——`base`（2 处）与 `acp-app`（1 处）里出现的是**引用该 provider 名的配置**，`packages/bundle/web-app/cordis.patch.yml` 里为 **0 处**；一个已被测试脚手架承认的风险写在 `apps/web/tests/default-model.overlay.yml` 的注释里："shipped deepseek-official default would be a route nothing serves — which the composer refuses to type into"（所以 fixture 自己改设了路由）。⇒ 新装用户第一眼面对的默认模型**可输入但发不出**（凭据缺失，见 §3.5），而 Taiji 路由按 `llm-taiji` 的合同**不需要凭据**（运行时在本机）。改那两行属**产品默认变更，需所有者拍板**（§4-D3），且**依赖 D2 先定**（§3.5 末段）。 |
| D4 | "桌面可用" | 桌面外壳里我方六个包真在装配内 | **我上一版的推断是错的，已按代码更正**：我原写"桌面走的是 `acp-app`／`headless` 那条不含我方行的链"（依据是 bundle 行数计数），但 `apps/desktop/src/main.ts:502` 实际服务的是 `resources.dsh/node_modules/@taiji/dsh-web-frontend/dist…`，而 **`@taiji/dsh-web-frontend` 就是本仓的 `apps/web`**（`apps/web/package.json` 的 name，且根 `build:web` 也指向它）⇒ 桌面外壳渲染的正是携带我方 10 行的 web 装配，**行数计数那条论证不成立**。**真正未测的问题换成了另一个**：打包暂存（`release:pack`／`prepare:desktop`／vendor 链）有没有把我方六个包放进 `resources.dsh/node_modules`——没有的话桌面装机会在加载期缺行。这一条要跑 `package:desktop:*` 或读 `prepare` 的清单才能定，本轮未做（会拉外部二进制）。web 侧事实不变：我方行在 `bundle/base`（2）＋`bundle/web-app`（10） |

## 2 · 与 G5 直接相关的本轮欠账（已消掉）

* **doc-sync 全量 43 叶重跑＝39 绿／4 红**（`0a5ba891`／`61040729`）；红＝`dependency-catalog` locale 假红、持久化两条等格式 5 裁定、一条 Windows 符号链接权限用例。
* **我方包 README 骨架 7 份不合规已补齐**（`26539ade`）——这类"包文档不合格"在交付审阅时最容易被点名，且此前被"这叶跑不了"的错判遮着。
* **判据①（tsc→tsdown 四连）与判据⑥（lint 基线）本轮取到读数**（判据⑥ 7 条 error 全在 `ui-life/src/client/LifePanel.tsx`＝与登记基线一致，无新红）。

## 3 · 明确不声称

* **不声称桌面打包可跑通**：实跑的只有第一环 `build:desktop`（`apps/desktop` 编译 rc=0）；`package:desktop:*`（拉 electron 二进制＋签名＋写 `.desktop-build` 暂存树）**一条都没实跑**。"已具备"到本轮为止指**命令存在、指向真实脚本、且桌面外壳可编译**。另：本机 `corepack` 能起钉住的 `pnpm@11.7.0`，所以旧的"本机无 pnpm"措辞已作废——没跑的原因改成"会拉外部二进制并写暂存树，等 D1 排期"。
* **不声称 G5 已就绪**：D2／D3 未达成，D4 只有 web 侧达成。
* **不声称桌面产物一定缺我方行**：只声称在 `packages/bundle/*/cordis.patch.yml` 这一层计数为 0；桌面装配是否另有注入点未查。

## 3.5 · D3 的下一层已测清（同日补做，静态读码＋待一次真机复验）

「装机默认是不是没人服务的路由」这条我原本记成"未测"，现已读出答案：

* **提供方在装配里**：`deepseek-official` 由 `packages/llm/llm-deepseek/src/index.ts:57` 注册，而该包在 `packages/bundle/base/cordis.patch.yml` 里有 **3 行**装配（`web-app` 0 行，但 web 走 base 的继承链）。
* **adapter 是无条件注册的**：`registerAdapter([PROVIDER], adapter)` 在同一 `apply` 里直接调用（`:114`），前面没有凭据判断；凭据是**每次请求**才解析的（`:65-81`），拿不到时抛的是可操作的文案：`llm-deepseek: no API key for provider route "deepseek-official"; store <ref> through the credentials`。
* **⇒ 结论与初稿的担心相反**：新装用户**能**在默认模型上输入（路由有服务），但**发不出去**，直到存进一个 DeepSeek key。上游测试注释里那句"a route nothing serves"只适用于它自己的 fixture（那个 scaffold 不注册 adapter），不适用于交付装配——这条我按实测更正，不再当风险引用。

**因此 D3 依赖 D2，三条裁定不是并列的**：Taiji 路由按合同不需要凭据，但它要求本机 8000 端口的 python 运行时在跑（`llm-taiji` 走 HTTP，未就绪时按设计不注册 adapter）。⇒ **在"运行时怎么随包"（D2）定下来之前，把装机默认改成 Taiji 只会把"要 key 才能答"换成"要另起进程才能答"**，两者都不是"装起即用"。所以合理排期是先裁 D2、再随它一起裁 D3，而 D1（打包实跑）可以现在就做以拿最便宜的失败信号。

**仍待做的一次真机复验**（不需裁定，但**需要 pnpm 与一次前端构建**）：干净 profile 起一次 web，读 `ctx.agentDefaultModel.currentSelection()` 与 `listProviders()` 的交集，确认上面这条静态结论在真装配里成立（默认=可输入、首轮请求=报缺 key）。**本轮试过一条捷径并失败**：`vitest.e2e.config.ts` 的 `include` 只有 `packages/*/*/tests/**/*.e2e.ts` 与 `apps/cli/tests/**/*.e2e.ts` ⇒ 直接跑 `apps/web/tests/default-model.e2e.ts` 报 `No test files found`；该文件属 `test:web:built` 那条 lane（要先 `pnpm run build` 出 web dist）⇒ **这条复验并入 §4-R1（在有 pnpm 的机器上做）**，不在本机强跑。因此 §3.5 的结论当前标注为**静态读码所得（行号可查）**，不是真机读数。

## 4 · 需要所有者拍板的三条（按能声称的最强结论排）

* **R1｜打包实跑在哪台机器**：`package:desktop:win:x64` 需要 pnpm 与桌面工具链。可选：① 本机装 pnpm 后我跑（会给本机加全局依赖，且要下载 Electron 等）；② 所有者在能跑的机器上执行并把产物/日志回传；③ 先只跑 `build:desktop`（不签名）拿"能构建"的证据，再谈装机即用。**推荐 ③→②**：先要最便宜的失败信号。
* **D3 默认 provider 怎么定**：(a) 装机默认＝Taiji（改 profile 配置，产品默认变更）；(b) 默认仍是上游 provider，Taiji 作为可选组（现状）；(c) 首启动做一次引导选择。**这条决定"Taiji provider 为默认"这句话能不能说**，我不自行改。
* **D2 运行时怎么随包**：随包附带并拉起／首启动引导用户启动／文档要求自备。**这条决定"一条命令装起即用"能不能声称**，也决定客户端版的工作量排序。

## 5 · 我可以在裁定前继续做的（无需批）

* ~~把 `prepare:desktop` / `release:vendor` 的**包来源**查清：本机 gitignored 产物 `apps/desktop/.desktop-build/development/project/desktop-runtime.json` 里出现 `@deepseek-ai/dsh-agent-default-model` 这类**改名前的包名**，而已跟踪文件里 `@deepseek-ai/dsh-` 为 **0 处**（2026-09-26 复算）⇒ 疑点：桌面暂存可能拉的是**已发布上游包**而非本 fork 产物。~~ **【同日已否证】**：该产物的 mtime 是 **09-22 16:26**，而改名提交 `88ad3040e`（G2 rename，5206＋43 文件）落在 **09-22 18:51** ⇒ 产物比改名**早 2.5 小时**，它是**改名前的本机构建残留**，不是"桌面线拉已发布上游包"的证据。副产品事实：这条暂存链在 G1 期就真跑通过一次（`schemaVersion 1`、`release.version 0.1.7-alpha.1`、`nodeVersion 24.18.1`、**`sharedPackages` 1495 项**）⇒ 桌面打包的**起点机制是存在的**，只是这台机器上的产物是旧的。
* **在真实 web 装配里查"装机默认是否可输入"**（零改产品码的只读探针，与判据⑤ 同一条取数路径）：起一次不带 settings 的干净 profile，读 `ctx.agentDefaultModel.currentSelection()` 与 `listProviders()` 的交集，判 `deepseek-official/deepseek-flash` 是否真在服务、composer 是否可输入。它直接决定 §4-D3 三个选项里哪个是"最小可用交付"，且**不需要任何裁定**就能先拿到读数。
* 把 web-first 的"启动即得 Taiji 组"写成可复验的验收脚本（复用判据⑤ 已有的 Playwright 取数路径），为 D2 的三个选项各留一条读数口径。
