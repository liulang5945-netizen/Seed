# 08 · Taiji Harness 改造清单与上游同步手册

状态：**定型生效**（2026-09-24 落盘；每次对 `taiji-harness/` 的成体系改造后必须回本文件增改条目——清单与仓库同步演化，否则机制失效）。
本文件是「Taiji 原生改造三件事」的第 ③ 项产出物，与[执行记录](03_CURRENT_EXECUTION.md)互指：03 记过程与证据，本文件记**可重放的现状**。

## 1 · 前提与范围

- **fork 无共同祖先**：G1 以 tarball 落地（非 clone），本仓从未有上游 git 历史与 remote ⇒ `git merge/cherry-pick upstream` **先天不可用**。同步机制定型为：**上游快照 + 本清单重放**（审计四已裁定，此处不重开）。
- **重放方向**：永远以**上游新版为基底**、把本清单条目应用上去；不是把我们历史的改动反向 diff。因此清单必须完整——**漏一条＝升级时静默丢失一条能力**。
- **范围**：仅 `taiji-harness/`（dsh fork）。外层 Seed 仓的 Python 侧改造（C1/D 记录环、C2 载荷字段、C3/B 的 `memory_store/sleep_pass/routes_*`、数据集扫描修正）**不属于本手册**——外层仓本来就是我们的原生仓，没有"上游"要同步。
- **基线身份**：上游 `@deepseek-ai/dsh-root` **0.1.7-alpha.1**（fork 落地即此版，至今 `taiji-harness/package.json` 未升版 ⇒ 当前所有"对基线的改动"= 基线→HEAD 的全部差异）。落地提交 `82042a2f6`（2026-09-22）。上游源：`https://github.com/deepseek-ai/deepseek-harness`（各包 `repository` 字段仍指它——**这是刻意的**，见 M4/D2）。

## 2 · 重放管线（升级一轮的操作顺序）

1. **取快照**：下载上游新版 tarball 到**仓外**（`E:\Seed\.fork-sync\dsh-<ver>\`，勿入本仓；`.dsh-sbx*` 同款隔离原则）。
2. **换基底**：`taiji-harness/` 内容整体替换为新版（git 跟踪目录，`git status` 天然给出上游侧 diff 面）。
3. **机械项**：按 §3 M 组逐条执行（M1 rename 脚本 → M2 品牌表 → M3 生成物重刷）。
4. **手写项**：按 §3 H 组逐包/逐点重放——新建包整目录拷回（它们在上游不存在，冲突为零）；上游文件修改按条目锚点重落（冲突面只可能出现在上游恰好同文件同区域改动时，逐条人工合并）。
5. **删除项**：按 D 组核对"新版不得带回"的内容。
6. **装配与安装**：沙箱外 `pnpm install --no-frozen-lockfile`（新包软链 + lockfile），随后 `tsc -b tsconfig.host.json` → `tsdown --env.DSH_BUILD_FACE host`（typert 契约）→ `tsc -b tsconfig.client.json` → `tsdown --env.DSH_BUILD_FACE client`。
7. **门禁**：§5 复验集全绿。
8. **回写清单**：基线版本号、以及"本轮上游改动与 H 组锚点相撞并已合并"的条目注记。

## 3 · 清单

图例：**M** 机械（脚本/表可重放）｜**H** 手写（人判断，锚点已列）｜**D** 删除/不落地。证据＝本仓提交哈希（历史在本仓，永久可查）。

### M 组 · 机械变换

| ID | 内容 | 重放动作 | 证据 |
| --- | --- | --- | --- |
| M1 | scope 更名 `@deepseek-ai/*` → `@taiji/*`（包目录名不变，`dsh-` 前缀保留） | 全文本替换 `@deepseek-ai/`→`@taiji/`；**豁免仅限包名/发布身份**（`python/sdk*` 的 `deepseek_harness*` 不改——D2），其**串值**照改。三种形态都要盯：带斜杠 `@taiji/`、**独立段** `'@taiji'`（python/sdk 测试、native/system、boot 与 cli 测试、THIRD_PARTY_NOTICES、desktop-host 安装锚）、**转义正则** `/^@taiji\//`（`packages/client/tsdown.client.ts`、`benchmarks/tsdown.config.ts`）；改名会连带 lockfile/registry URL 区——以替换后 `pnpm install` 重建为准 | `88ad3040`（5206+43 文件）、清单见 `908a04f3` 尾部五类残余；文件级审计 2026-09-24 登记残差 22 条均属此族 |
| M2 | 品牌文案对（固定字符串替换表）：`DeepSeek Harness`→`Taiji Harness`（身份句唯一定义处 `packages/core/system-prompt/src/index.ts` 的 `includeHarnessIdentity`，其余全部机械跟随）；**上游缩略语 `DSH` 单独出现处**→`Taiji`／`Taiji Harness`（locale `brand.localBuild`、`apps/web/index.html` 标题与 `vite.config.ts` 注入、`manifest.webmanifest` short_name、ui-agent-preset 向导文案、ui-sidebar-browser 错误文案）；首屏 slogan 现值「态之极境 / State at Its Utmost」（zh/en locale 两侧）；「探索未至之境/Into the Unknown」全仓禁再现 | 按表逐对替换；同步面：locale 字典、`skeleton.client.spec` 断言、web 快照期望（`snapshots/web/*`）、6 个 web e2e needle、`CONTRIBUTING(.zh).md` | `908a04f3`、`19fb5130`、`3c852a97`、`d4987c05`（预览版徽标摘除）、`88ad3040`（DSH 缩略语面） |
| M3 | 生成目录与锚：`config-catalog(.zh)`、`tool-catalog(.zh)`、`api-catalog.ts`、subsystem cordis-surface 区、slot-catalog（`packages/extensions/cordis-client-runner/src/client/slot-catalog.ts`）、`apps/cli/composition.md`、doc-graphs、tsconfig paths——全部由 `scripts/gen-*`／装配变化按**当前包名**产出（M1 之后自然翻成 `taijidsh-*` 锚；H 组每片装配都会连带刷新 slot-catalog 与 composition.md） | 跑 §2 第 6 步后依次 `gen-cordis-catalog`／`gen-config-catalog`／`gen-config-catalog-zh`（zh 镜像同为生成物，见 §6 收敛记录）／`gen-tool-catalog`／`gen-client-catalog`／`gen-doc-graphs`／`gen-tsconfig-paths`，再 `--check`；两份 catalog 改动后重录配对 `verify-translation-pairing --write docs/config-catalog.md` | 锚收敛先例 `eab07afe`（100 文件 642 处一次性对齐）、`b90c0de2`（ui-life 装配连带 slot-catalog） |
| M4 | `package.json` 的 `repository/homepage/bugs` **保持指上游**——这是上游事实引用（THIRD_PARTY 同款），不是品牌残余，不改 | 无动作（防误改条目） | 边界裁定见 `908a04f3` 尾部⑤类分析 |

### H 组 · 手写增量（能力包与上游文件修改）

**H1 · 本仓新建包**（上游不存在 ⇒ 整目录拷回，零冲突；每包都带 README 中英对 + i18n.yaml + package.json `dsh.client` manifest）：

| ID | 包 | 作用（一句话） | 装配面（必须同落，见 H2） | 证据 |
| --- | --- | --- | --- | --- |
| H1a | `packages/llm/llm-taiji` | `taiji-local` 路由硬接 8000 `/api/chat/stream`；失败全走 `LlmError` 归一化；health 判可路由，并在插件存活期内**按 `readinessPollMs`（默认 5000、下限 250，非 volatile＝装配选择）再探测**——加载时不可达的运行时此后会被路由，探测串行化以防两次首注册撞 `DUPLICATE_ADAPTER`，停表挂 `ctx.effect`。重放时**勿退回旧形**（「只在 load 与 `loader/volatile-update` 采样」＝运行时冷启动晚于 harness 启动就永久缺席模型目录） | base patch `llm-taiji` 行、`tsconfig.host.json` 引用 | `d59d918f`、`8210a077`（轮询落地；该提交由并发会话的不带 pathspec 提交吞并，故哈希归属混装——按文件族取，别按信息取）、`31d47384`（真组合用例） |
| H1b | `packages/api/life-controller` | 单轮询器读 runtime：快照/替换帧流 + 12 个 Remote 动词（train/consolidate/activate/legacy 族） | web-app host `life-controller` + **client `ui-life` 行的前置**、remotes 挂清单（`lifeRemote`）、`RemoteErrorDetailsMap` 的 `life/*` declare-merge | `b61d1656`、`8924e2d4`、`fdd…` 等三片 |
| H1c | `packages/context/life-context` | `life:policy` 静态段 + `agent/pre-step` **前置**注入 life-state（节流/新鲜门） | web-app host `life-context` 行、system-prompt `SECTION_ORDERS.LIFE_POLICY`（H3e） | `5d46e522`、`ec8baac1`（注入位置修正） |
| H1d | `packages/context/memory-context` | 每回合一次 `GET /api/memory/recall` 前置注入（零命中不注、失败整块弃） | web-app host `memory-context` 行 | `ec8baac1` |
| H1e | `packages/session/session-memory-taiji` | `agent/turn-stopping` 上报回合到 `POST /api/memory/record`（问/答同形语料、幂等、best-effort） | web-app host `session-memory-taiji` 行 | `ad8a3ce3` |
| H1f | `packages/client/ui-life` | 侧栏 Life 入口 + **六分区**主面板（来源/生命/训练+名册+检查点行续训激活/知识/记忆巩固/宿主含能力鉴权） | web-app client `ui-life` 行、ui-layout `MainPanelId` 联合（H3d） | `9091906d`、`d24b22e0`、`0d6b994f`、`49d7c257`、`16c76667`、`12001360` |

**H2 · 装配行与配置**（散在上游文件里的"插线"，升级时逐行比对）：

| ID | 文件 | 我们的行 |
| --- | --- | --- |
| H2a | `packages/bundle/base/cordis.patch.yml` | `- id: llm-taiji`（紧跟 `llm-deepseek`，deepseek 原行保留） |
| H2b | `packages/bundle/web-app/cordis.patch.yml`（host 面） | `life-controller`、`life-context`、`memory-context`、`session-memory-taiji` 四行（各带注释段）；client 面 `ui-life` 行 |
| H2c | 两 bundle 的 `package.json` | 对应依赖行（`workspace:^`） |
| H2d | `tsconfig.host.json` / `tsconfig.client.json` | 新包项目引用；`tsconfig.base.json` 别名由 `gen-tsconfig-paths` 生成（M3） |
| H2e | `packages/api/remotes/src/index.ts` | `lifeRemote` 的 import、type re-export 与 **mount 清单行**（漏了＝浏览器 `remote.life` 永不激活）；其 `package.json` devDependencies + `tsconfig.client.json` 引用 | `8924e2d4` |
| H2f | 根 `AGENTS.md`/文档站点登记 | `SERVICE_PAGE`/`SERVICE_ROLES`/`SENTENCE_MODEL_EXPERIENCE`/`linkedTypePages`（`gen-cordis-catalog.ts` 的类型链接表：`Life*` 全系）随每片追加 | 各片 |
| H2g | `.agents/notes/implemented/feature/**`（目录上游本有） | 我方每个行为改动按纪律增/改中英 note 对——决策反转须**新 Note＋旧 Note 原地更新事实并交叉链接**，不许只改旧文件（实证：first-use 资格 2026-09-23 新增 ＋ default-workspace 2026-09-20 更新＝`e8433b39` 配套）。重放：新增 note 整文件拷回（零冲突）；上游若动同文件人工合并 | `e8433b39`、`8955806a` |

**H3 · 上游文件的功能性修改**（真正的 merge 冲突面；逐条给锚点）：

| ID | 上游文件/区域 | 改动 | 证据 |
| --- | --- | --- | --- |
| H3a | `packages/client/ui-conversation` `conversation.blocks` | 每会话单槽 → **按 owner 键控多源注册表**（`COMPOSER_BLOCK_OWNERS`，归档只读 > 模型不可路由） | `5f5158ca` |
| H3b | `packages/workspace/workspace`（`WorkspaceRegistry`；**勘误：原文写的 `packages/core/workspace` 不存在**——文件级审计 2026-09-24 钉正） | ① `deleteSession`：摘 archive/pin + 关闭 live（`ctx.parallel('workspace/session-close')` 后复查）+ 物理删；② 默认身份不变量（删默认工作区同写清 `defaultWorkspaceId`）+ 启动自愈 `healStaleDefaultIdentity()`（`workspace.spec.ts` 的 `beforeInit` 日志观察缝为其测试配套）；③ 首次创建资格放宽为"注册表为空"。**装配面必同落**：`packages/api/workspace-controller/**` 8 文件（client model/service + commands + types + index + tests×2：`WorkspaceDeleteSession*` 动词、`WorkspaceSessionDeleteError`、`workspace/session-open` 拒绝读数） | `8955806a`、`4b8ab255`、`e8433b39` |
| H3c | `packages/api/session-controller` + `packages/session/session-persistence`（接口）+ `session-persistence-jsonl` | `ApiSessionAgentController` **保留 AgentHandle** + `closeSession`（cancel→whenIdle→flush→dispose）+ `closing` 集合；`SessionPersistence` 接口增 `delete` 契约 + `SessionPersistenceDeleteOptions`（不存在抛 `SessionPersistenceNotFoundError`）——**连带 7 个合同测试桩补 `delete`**（feedback helpers、schedule plugin.spec、session-query×4、session-checkpoint-policy）；jsonl `delete` 首组测试与并发不变式 | `8955806a`、`9619bf99` |
| H3d | `packages/client/ui-layout` + `ui-workspace` | `MainPanelId` 加 `'life'`；归档行只读打开（标志**只由本服务动作结束、快照不得改写**）、回档入口、删除入口与确认框、`replaceMain` 写入顺序修复 | `9091906d`、`3c852a97`、`d4987c05` |
| H3e | `packages/core/system-prompt` | `SECTION_ORDERS.LIFE_POLICY: 700`（TEAM_POLICY 与 PTC_ONLY 之间）；身份句文案在 M2 | `5d46e522` |
| H3f | 品牌资产（上游文件替换） | `FishLogo`→几何太极 mark、favicon 族、字标、welcome 页（`lib/welcome/*` 的构建源，**连带 `packages/client/ui-settings-models/src/onboarding-copy.ts` 的 `WELCOME_NOTICE_VERSION` bump**）；swim 动画→旋转动势。路径清单（文件级审计补登）：`apps/web/{index.html, vite.config.ts, public/manifest.webmanifest, public/favicon.svg, public/favicon-dark.svg}` | `44d1d2af`、`d4987c05`、`88ad3040` |
| H3g | 首屏空态（EmptyHero 所在包） | 徽标节点删除 + `.previewBadge/.titleGroup` CSS 移除 | `d4987c05` |
| H3h | `docs/subsystems/*` + 各包 README | life.md 对新建、workspace 等页随 H3 各条**重生成**（gen 产物，归 M3 校验） | 各片 |
| H3i | `.gitattributes`/行尾 | fork 仓内 ts 源以 LF 入库（`eol=lf`）；生成器 verbatim 归一（`gen-config-catalog` pasteText）——防 CRLF 进双语门 | `905d29f5` |
| H3j | **中文镜像工具链**（zh catalog 生成化，2026-09-24） | 新建两文件整拷回：`scripts/gen-config-catalog-zh.ts`（与 en 同源渲染＋en/zh 围栏逐字对拍）＋`scripts/gen-config-catalog-zh.spec.ts`（5 用例）。**改三个上游文件**：`scripts/gen-config-catalog.ts`（导出 `FENCE`/`TypeRef` 两符号，零行为改动）、`scripts/run-gates.ts`（doc-sync 批 +1 门 `config-catalog-zh`）、根 `package.json`（+`gen-config-catalog-zh`/`verify-config-catalog-zh` 两 script 行）。产物连带：`docs/config-catalog.zh.md` 转为生成物（勿手改）、`docs/config-catalog.i18n.yaml` 重录。**登记说明**：审计脚本白名单把整个 `scripts/` 归我方，此改动不产残差但必须有条目——2026-09-25 增量登记时发现并补 | `8f65dfd8` |

### D 组 · 删除/不落地

| ID | 内容 | 规则 |
| --- | --- | --- |
| D1 | 上游 CI：`.github/workflows/*`（publish/release 流水线指向 DeepSeek 基础设施） | 保留文件**不接线、不执行**（fork 无发布通道）；新版带回变更直接接受，不影响本地门禁 |
| D2 | 上游 `python/sdk*` 包名、THIRD_PARTY_NOTICES、vendor 归属 | **不得**品牌化（M4 同源边界） |
| D3 | `.agents`/`.claude` skills | G2 裁定为**冻结档案**：不追改，也不在上游重命名它们时跟着动 |
| D4 | 上游若重新引入「探索未至之境」类 DeepSeek 口径文案 | 按 M2 表再次替换（不是回退清单） |
| D5 | 外层仓旧线（`frontend/`、`desktop-electron/`、`desktop/`、发布链）**不属于本清单**——它们在 Seed 外层仓已物理退役（2026-09-23），与 dsh 基线无关 |

## 4 · 环境与工具事实（重放执行时的操作层，不属清单条目）

- 沙箱内构建链：`node node_modules/typescript/bin/tsc -b <tsconfig>`；`node node_modules/tsdown/dist/run.mjs --env.DSH_BUILD_FACE <host|client>`；vitest `node node_modules/vitest/vitest.mjs run <path> --pool=threads`；tsx 直调脚本。
- 沙箱 HOME 隔离：`$env:USERPROFILE='E:\Seed\.dsh-sbx2'`；workspace 包以 **junction** 链接（缺链用 `.dsh-sbx2/sbx-link.mjs` 补，targets 需含新包）；`python -m black` 需 `$env:BLACK_CACHE_DIR` 指沙箱内否则永久挂起。
- **顺序铁律**：改 `src` 后必须先 `tsc -b` 再 tsdown（client bundle 入口是 `lib/types/**`）；typert 新 verb 必须先 host face 构建生成 `lib/typert.remote-client.d.ts`，client 面 tsc 才可能过。
- 编辑与编译校验**勿混同一并行批**（半写文件假红）。

## 5 · 复验集（一轮重放后的通过判据）

1. §2 第 6 步四连（tsc host → tsdown host → tsc client → tsdown client）零错。
2. 文档/生成物 `--check`：`gen-cordis-catalog`／`gen-config-catalog`＋`verify-config-catalog-zh`（zh 侧由 `gen-config-catalog-zh` 生成，跑完重录 pairing `--write`）／`gen-doc-graphs`／`gen-tsconfig-paths`。
3. 17 项门禁批（md-wrap／md-links／doc-refs／subsystem-pages／summaries／model-experience／limitations／invariants／meta／dependencies／cordis-config／doc-budgets／export-jsdoc／translation-pairing…）＋`verify-cordis-catalog/inspect`；`verify-client-ui-i18n` 唯一合法红＝`BrandWordmark`（既存）。
4. 定向 vitest：`llm-taiji`、life 五包 + `packages/core/system-prompt`、`ui-conversation`+`ui-workspace`+`session-controller`+`session-persistence-jsonl`（H3 契约用例——**这些用例是行为合同的守卫，红了就是重放丢了条目**）。
5. 真机读数（G3/G4 判据）：web 起服 → 模型面板出现 Taiji 组（**启动顺序不再敏感**：运行时晚于 harness 就绪时，该组在一个 `readinessPollMs` 内自行出现，目录按需从活注册表计算故无需重启）→ 发真回合由 Taiji 应答（约束在语料内，不拼 prompt）→ Life 面板六分区读数与 `GET /api/runtime/status`/`/api/artifacts`/`/api/consolidation/status` 逐项一致 → 记忆写入/召回闭环（journal 计数增长、下一回合可见注入块）。
6. lint：仅允许**登记基线**（ui-life errorText 7 条；重测于 2026-09-24 恰为 7）内条目；出现新红＝丢改。

## 6 · 已知薄弱点（诚实登记）

- ~~中文 `config-catalog.zh.md` 是手工镜像~~ **已收敛（2026-09-24）**：`scripts/gen-config-catalog-zh.ts` 与英文侧从**同一份** `collectConfigCatalog` 渲染（锚/标题/逐字围栏/`来源：` 行号结构性一致，翻译骨架内聚为脚本常量），`verify-config-catalog-zh` 已入 doc-sync 门与 §5 判据②；main() 另有 en/zh 围栏逐字对拍，双渲染器一旦漂移即炸。旧的手工镜像叙事（含 `H3i` 之外的“先跑 gen 再手对 zh”流程）全部作废，防重放时误走旧路。
- H3 各条的锚点随上游重构漂移的风险：本清单以"功能点 + 文件族"定位，不背行号；若上游整文件重写对应功能（如 blocks 合同再变），该条**升级重设计**而非硬贴。
- ~~并行工人在飞文件（system-prompt 回退、host.spec `:225`）~~ **已收敛（2026-09-24 所有者裁定回滚）**：system-prompt 两文件恢复 Taiji 身份句；`:225` 查实是 HEAD 既有类型错（非在飞改动），按仓内惯用法 `failure?.message` 修复——**全量 `tsc -b tsconfig.host.json` 首次 0 错**。仍在飞＝life-context ×4 与 session-memory-taiji ×2（六文件 lint/tsc 零新增，未裁定，升级前仍需收敛）。
- 本清单随改造增长：**每片合入即追加/修订对应行**（新增 H 条、更新证据哈希）；"清单完整性"就是同步机制的全部安全性所在。
- **共享 worktree 的提交归属风险（2026-09-25 实测）**：并行会话执行**不带 pathspec** 的 `git commit` 时，会把本会话已 `git add` 的暂存面一并提交进它自己的信息里——llm-taiji 的就绪轮询修复即因此挂在 `8210a077`（`feat(taiji): 复制回路补 init_seed 通道`）名下：内容无损、归属混装。**纪律**：本仓一律用 `git commit -m … -- <pathspec>`（不依赖暂存区，也不吞别人的暂存），与 03 §四「提交一律带 pathspec」同源；重放与取证按**文件族**定位改造，不按提交信息。
- **文件级完整性审计已跑（2026-09-24，2026-09-25 复跑增量）**：对 `82042a2f6..HEAD -- taiji-harness` 的内容差异文件做双规则归并（M 内容模式 ∧ 路径白名单）⇒ 首轮 5065 文件/57 残差逐条人工判定：**真清单缺口已全部修条**——H3b 路径勘误（`packages/core/workspace` 不存在）＋ controller 装配面 8 文件、H3c 接口契约与 7 个连带测试桩、M2 缺 `DSH` 缩略语对、M3 缺 `apps/cli/composition.md`、H2g note 纪律未登记、H3f 资产路径与 welcome 版本 bump；余 22 条为 M1 变体形态（独立段/转义正则），已登记进 M1。**09-25 复跑**（5068 文件/57 残差，与首轮持平）暴露白名单盲区：`scripts/` 整目录前缀把我方对上游脚本的改动全部吞进 h-whitelist（zh 工具链 3 个上游文件改动不产残差）⇒ 补 **H3j** 登记，且此后**审计绿≠清单全**——凡动 `scripts/`、根 `package.json` 的上游文件改动必须人工对到 H 条目。脚本与输出：`E:\Seed\.dsh-sbx2\audit_fork_diff.py`／`audit-out.txt`（仓外，不入 git）。
