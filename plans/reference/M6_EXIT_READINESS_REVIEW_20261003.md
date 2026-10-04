# M6 收官对账（2026-10-03）

> **性质**：只读对账 + 当日实测增量登记。**不是收官批准书**——按 `02_GATES_AND_CI.md:145`，M6 采用评审
> 需**所有者批准**才能声明通过；本件只把「判据原文 / 当前证据 / 缺口 / 能否声称」四栏摆齐，供裁定。
> 证据搜集为只读，未修改任何产品代码；当日增量部分的读数全部来自本机实跑并逐条标注了命令与出处。

## 0 · 一句话结论

**M6 现在不能收官，也不建议进 M7。** 三条独立理由，任一条单独成立即足以阻塞：

1. ~~**六项退出条件里，前两项（采用 artifact 清单、shadow/canary）在仓库里连一条 M6 期的执行记录都没有**，
   而归档件已把这两步写成产品采用的必经环节——**流程要求存在、执行记录缺失**。~~
   **【2026-10-03 已处置】**所有者批准**列入「未完成且从交付范围排除」**，批准书
   [M6_SCOPE_EXCLUSION_APPROVAL_20261003](M6_SCOPE_EXCLUSION_APPROVAL_20261003.md)。
   ⚠️ **批准不改变这两项的未完成状态**，也不等于 M6 收官——见批准书 §2.1／§2.3／§3。
   ⇒ **本条不再是阻塞理由**；理由①②③ 之外，判据⑦ 重取与 B／C 档各项仍在。
2. **判据⑦ 的可引用数字已过期**：现台账引的「35 红 / 注入池 20 文件」出自 2026-09-30，而
   `SPEC-M6-01` 的注入折叠**已于 10-01 落地并部分验收（12 条纯注入 lane 里 8 条转绿并通过 replay 复验）**。
   拿过期数字判收官，会把一个已经收口一半的族算成未收口。
3. **台账未更新**：M6 活动卡停在 09-24、G5 就绪清点停在 09-30、PLAN_INDEX 开放项停在 10-02 第三批；
   **10-03 当日的全部工作（两批共 8 处 UI 修复、训练区拆块、四个版本出包、离线更新通道验证、产物清理）
   一行未入账**。

## 1 · 六项判据逐项对账

判据原文出自 `01_SCOPE_AND_PHASES.md:30`：
> 采用 artifact 清单、shadow/canary、持久化/恢复、provider 隔离与故障降级、客户端真实能力对齐、桌面体验与端到端验收

**状态四档要分清**：「已达成」／「部分达成」／「未达成（文档明确说不达成）」／「**未找到任何记录**」。
最后两档性质不同，不可混称——前者是结论，后者是**空白**。

| # | 判据 | 状态 | 证据锚点 | 缺口 |
|---|---|---|---|---|
| 1 | 采用 artifact 清单 | **未找到任何记录** | 判据三处定义齐：`01:30`、`01:112`、`02:145`（后者是唯一定义了 M6 退出判据的正式门禁行，**至今未被任何文档声明通过**） | 无 M6 期的采用清单。`plans/manifests/` 39 份 manifest 中最接近的是 M5 期 K 轴默认运行时附着（`M5_K_DEFAULT_RUNTIME_ROLLOUT_REVIEW_PREREGISTRATION_20260911.md:77`），**不是** M6 采用清单。**注**：此处 artifact 指**模型工件/行为范围**（同 `01:112`「artifact/行为范围」），构建产物语义属 M7（`01:31`） |
| 2 | shadow/canary | **未找到任何记录** | 流程规范在归档件：`archive/history/20260913_post_route_a/EXECUTION_BEFORE_REVIEW.md:176`（依赖验收→shadow 旁路→opt-in canary→持久化/恢复合同→默认行为评审）、`M5_B0_CLOSEOUT_AND_PLAN_REVISION_20260914.md:180` | 流程已定义，**M6 期无任何 shadow 或 canary 执行记录**。中文词族（灰度/影子/试点/预发）全域零命中。`canary` 的命中均同名不同物：`output/manual-r5-canary/` 测试残留目录（`05:12` 等 12 处）、M4V2/M5 研究期实验 canary |
| 3 | 持久化/恢复 | **部分达成** | 门禁已闭合：`03:1250`（归属更正）→`03:1268`（`ec2e72b2` 根因修复，8 处新增判为 `attribution-kind-added`）→ PLAN_INDEX 开放项⑦ 销账 → G5 §11 记 `doc-sync 43/43` | **门禁闭合 ≠ 退出条件达成**。`回滚/rollback` 在 03 的 8 处命中（L1012/1152/1168/1176/1178/1218/1242/1250）逐条读过，**全是别的族**（DONE 族修法、锁文件回退、金样回退、单实例锁、生成物回滚），无一条是恢复验收。`重启恢复/会话恢复` 在 M6 语境零命中【**2026-10-04 ㊵-218 更正：这句是我搜错了面**——当时搜的是**文档词**，没有枚举**测试面**。枚举之后：本机有 **8 条恢复/重启 lane 为绿**——web 侧 `connection-recovery`(2 用例)／`models-settings-recovery`(2)／`server-restart`(1)，host 侧 `agent-loop/resume`(45｜1 skipped)／`schedule/jsonl-restart`(1)／`v3-restart-migration`(4)／`life-controller/resume-training.host`(6)；**真缺的只有硬崩溃那一支**：`session-checkpoint-policy/tests/crash-recovery.e2e.ts:91` 整段是 `describe.skipIf(process.platform === 'win32')`，隔离趟里该文件**根本没出现**（未进分区）。⇒ 本格从「零记录」改判为「**有可执行验证且本机为绿，缺硬崩溃验收＋缺文档引用**」】 |
| 4 | provider 隔离与故障降级 | **部分达成**（隔离成立，**降级零记录**） | 隔离面已证一条真事实：`03:1286`（H3p 判别实验）——`listProviders()`（注册面，有 adapter 才列）与 `listConfigurableProviders()`（声明面，配置即可列）是**两张表**，已被 08 的 H3o 复用为判读方法 | **`故障降级/failover` 在 M6 语境零命中【**2026-10-04 ㊵-219 更正：与 ㊵-218 同一族错——我当时搜的是文档词，没枚举测试面。**隔离趟（无并行）里重试/隔离行为有三条 lane 全绿：`llm-retry/tests/retry.spec.ts`(31 用例)／`llm/tests/retry-policy.spec.ts`(20)／`api/gateway/tests/control-retry.client.spec.ts`(15)【㊵-219 落库后自查：我第一次把这两条的用例数对调了，实测取自隔离趟的每文件 `✓` 头行】，共 66 个用例。**真正缺的那半句仍然成立、但要说准**：降级没有用户可见文案——唯一上报点是 `packages/llm/llm-retry/src/index.ts:207` 的 `ctx.logger.warn`，该包的 `inject` 只有 `['agents','sessionProjections']`（无通知/插槽面），`agent-loop/src/agent.ts` 里也搜不到 error/notification 命中。⇒ 本格由「零记录」改为「**重试/隔离有绿测试面，缺用户可见降级文案与跨 provider 改道实现**】】**。`降级` 在 08 的 7 处（L81/221/230/252/297/588/1046）逐条读过，**全是「把结论降级为过程读数」的记账用法**——同名陷阱。历史有工程资产（`archive/history/SEED_ROADMAP_EXECUTION_FOUNDATION_2026_08.md:265-278` watchdog＋自动回退＋隔离版本不复活），但 `01:30` 自己写着「已有 provider/客户端工程资产，**未据本轮整体重验**」 |
| 5 | 客户端真实能力对齐 | **部分达成** | `03:1230`（未闭合 I 收口，**M6 线**，2026-09-25）：根因＝就绪只在 load 与 volatile-update 采样一次、运行时冷启动晚于 harness 即**永久**不注册 Taiji 组、composer 被挡；修法＝按 `readinessPollMs` 轮询；**真机红绿**（空端口⇒无 Taiji 组；假就绪⇒组自行出现且 radio checked） | 这是**单点缺陷修复，不是逐项裁决**。反证仍在：PLAN_INDEX 开放项 13 的 owner 反馈⑧「生命系统各按钮是摆设没真正接线」，到 10-02 才定位根因（随包安装器无后端通道）并修复。`能力对齐` 在 M6 语境外命中均为 2026-08 期前端条款 |
| 6 | 桌面体验与端到端验收 | **部分达成** | 专有台账 `TAIJI_G5_READINESS_20260926.md` §11：D2 装起即用／D3 默认 provider＝Taiji／D4 桌面装配**均成立**且有真机读数（`03` §5.7 的「真机发一回合」：默认链免凭据被服务、durable 记录自己命名 `taiji-local`） | **D1 后半未闭合**：打包末步 office 冒烟仍红（R7）。**判据⑥ 已满足（2026-10-04）**：`duplication` rc=0／Found 0 clones，他线带入的 231 条缩进亦已代修⇒已入库面零 lint 红（08 ㊵-159／179）；本行原文那句「未满足」降为当日历史。**判据②③** 余 1 红（`verify-doc-graphs`，归并行会话在飞件）。**判据⑦ 数字过期**（见 §2）。**品牌**：安装器／卸载器／skill 徽章的字标栅格待排（构图决定，待确认） |

**四项「部分达成」里没有一项可以写成「已达成」。** 按 `01:30` 的纪律（`03:486`：
「G5 四条判据没有一条被『整句』勾掉」），本件沿用同一口径。

## 2 · 判据⑦ 的数字必须重取（这是最容易误判的一格）

| 时点 | 读数 | 出处 |
|---|---|---|
| 2026-09-30 | `Test Files 36 failed｜104 passed｜4 skipped (144)`；池＝注入过期 20／win32 工具名偏斜 8 | 08 ㊵-86／㊵-91（`rerun8`） |
| 2026-09-30 | `35 failed｜105 passed｜4 skipped (144)` | 08 ㊵-95（`rerun11`，五条预注册判据逐条 PASS） |
| **2026-10-01** | **`SPEC-M6-01` 落地并验收：12 条纯注入 lane refresh 后 8 条转绿并通过 replay 复验**；4 条仍红（`lifecycle-chrome`／`ptc-round`／`present-svg`／`question-composer`）＝另有其因，如实归因不硬凑，refresh 产物已回退不提交 | `SPEC-M6-01_replay_injection_fold_prereg_20261001.md:57-59` |
| 2026-10-04（`rerun13`） | **`Test Files 1 failed｜139 passed｜4 skipped (144)`**／`Tests 469 passed｜32 skipped (501)`（四批 1/0/0/0，逐批“汇总 failed＝解析 FAIL 头数”闭合；dist 与 client 两平面内容哈希前后相同） | 08 ㊵-181／㊵-183／㊵-186；红文件清单 `rerun13-redfiles.txt` |

**历史面数显式作废（本件 §6 要求的那一步，2026-10-04 执行）**：`35／36／42／44／46／49／50／65／69／81` 这一整串以及 `rerun12` 的 `33 failed` 全部降为**过程读数**，只用于记录收口过程，**不得再被任何收官件引用为当前状态**；当前可引用的面读数自此只有 **`1 failed｜144`**（`rerun13`）。两条限制必须与该数一起引用：①对 `rerun12` 给不出名字级集合差（那轮批日志与清单都不在盘上，08 ㊵-167 跑前注册），故本轮只报并集；②树上按 owner 裁保持未提交的两枚件之一（`clickable-links-gallery` 的期望色值）**在面内**，所以这**不是“干净检出”的数**（该改动与 HEAD 的 `--dsw-static-seed-500: rgb(86, 158, 88)` 同源，见 08 ㊵-186）。**同时更正本件 §2 的那句“当前数字必须重取一次整面才能引用”**：重取已完成，且结论比预期更好——剩余那**一枚**红的成因是录制机与本机 PowerShell 输出语言不同（08 ㊵-186），**不在注入族**，所以本件 A 档里“R-1 剩余 4 条”已结清、§2 表格所依赖的“按平台分档重录（乙档）”前提被正面否证、应撤。



`SPEC-M6-01` 的处置方式（**乙「授权有原则归一化」**）已于 2026-10-01 由 owner 弹窗授权，判据冻结在 §2：
在 `apps/web/tests/scaffold.ts` 的 `normalizeWebSessionVolatiles` 里对
`data.source.kind === 'life-context' | 'memory-context'` 的 `user/message` 行做**逐行 JSON 折叠**
（`{{lifeContext}}`／`{{memoryContext}}`），**不丢弃整行**（丢弃会让注入逻辑退出断言面）。
附录另记两层值层折叠（`{{harnessHome}}`／`{{tempWorld}}`）与一条绝对路径守卫。

⇒ **注入族已从「完全未收口」推进到「12 条里 8 条收口」**，台账引的 20／35 是收口**之前**的数。
**当前数字必须重取一次整面才能引用**，否则要么把已收口的算成未收口，要么反过来。

## 3 · 2026-10-03 当日增量（尚未入账，本件首次登记）

以下全部为本机实跑，读数逐条可复核：

| 项 | 读数 | 出处 |
|---|---|---|
| **桌面出包链路走通** | 连续出包并发布四个版本 `.20261003.1` → `.4`；`--build-version` 是 CLI 选项而非环境变量（只给环境变量会退回产品版本 `0.1.7-alpha.1`，**比已装的更旧**） | 本会话；`apps/desktop/scripts/package-target.ts:316-330` |
| **R7 归因确定** | 打包末步 office 冒烟红在 **`xlsx→PDF`**，完整错误 `LibreOffice native conversion failed: loadComponentFromURL returned an empty reference`；**`docx` 已过**。三次出包（10-02 两次、本日一次）均在同一条 ⇒ 属稳定既存缺陷，不是本次改动回归 | `packaging-runs/2026-10-03T02-47-22.648Z-eY7Nfc/stderr.log` |
| **离线更新通道验证** | `.4` 内部 app 版本 `0.1.7-alpha.1.20261003.4`（从 asar 读出）＝文件名＝频道 yml；**sha512 复算逐字符相等**（`0Eck2z2lA9pyr…`）、size 848,156,344 相等、blockmap 在位、通道只余三件；asar 内标记全命中（含新增 `life-training-data`／`life-checkpoints`） | 本会话独立复核，未采信发布脚本输出 |
| **生命页 UI 两批共 8 处修复** | 上传按钮"没渲染"根因＝`ui-primitives` 的 `Button` 默认 `ghost` 是透明底无边框（`Button.module.css:9-15`）；其余为分组依据说明、组界分层、字级统一 | 提交 `0fb45e14`／`970b616e`／`adade3bb`／`266289e1` |
| **训练区拆三块** | 六区块→八区块（`life-training`／`life-training-data`／`life-checkpoints`）；既有 23 用例零修改通过，新增 1 用例钉住结构且**实测能红** | 提交 `266289e1` |
| **产物清理** | 删旧版本安装包 11 个、**3.95 GiB**、零失败；保留 `.4` 与 `win-unpacked`（1.7 GB，当前版本 asar 在其中） | 本会话 |
| **§附 那格「只差一次真实启动」结清** | `TAIJI_G5_READINESS:531-543`（2026-10-02 23:45）要求 owner 装版后发一句话并同时看 UI 与进程日志两处。**owner 2026-10-03 实际完成**：装 `.2`→`.3`→`.4`、点更新、实际使用并给出六条 UI 反馈 | 本会话；该件不再欠 |
| **打包环境限制（新登记）** | 助手环境内 node **同步子进程创建被系统层拦截**（`execFileSync`/`spawnSync` 一律 `EBUSY`，对任意目标包括 `.txt` 皆同；与 node 版本、worker 线程、`dangerouslyDisableSandbox`、清空 `NODE_OPTIONS` 均无关），而**异步 spawn 正常** ⇒ 桌面打包**不能在助手环境内完成**，须在普通终端跑 | 本会话实测六种组合；诊断口诀：`node -e "execFileSync('cmd.exe',['/c','echo',''])"` |

## 4 · 未闭合清单（按能否声称分级）

**A 档 · 阻塞收官，且需所有者动作**
1. ~~**第 1、2 项判据（artifact 清单、shadow/canary）**~~ ⇒ **2026-10-03 所有者批准列入「未完成且从交付
   范围排除」**（[批准书](M6_SCOPE_EXCLUSION_APPROVAL_20261003.md)）。**批准不改变其未完成状态**，
   欠账保留，重启条件见批准书 §5。本条**已结**。
2. **判据⑦ 当前数字重取**——一次新鲜整面（`test:web:built` 144 文件口径），才能引用于收官件。
3. **R-1 剩余 4 条**（`lifecycle-chrome`／`ptc-round`／`present-svg`／`question-composer`）——`SPEC-M6-01` §3.3
   明确要求"如实归因不硬凑"，**这 4 条不属于注入族**，须各自定因。

**B 档 · 阻塞收官，我方可自办**
4. **判据⑥**：`duplication` 剩 1 枚克隆，落地步骤已由 09-30 裁定写明（`scripts/package-dependency-policy.ts` 的
   `SAFE_HOST_DEPENDENCY_EXPORTS` 加两条 → 等待器落进 `@taiji/dsh-api-gateway`＋`life-controller` 加依赖＋同提交改锁
   → 预期 `Found 0 clones`）。**未做。**
5. **D1 后半 / R7**：`xlsx→PDF` 转换失败。**已确定是 LibreOffice 原生转换的空引用，且只卡 xlsx、docx 已过。**
   修法需要 LibreOffice 侧结论，不是重跑能解决的。
6. **字标栅格**：安装器 `brand*`／`uninstaller-sidebar`／`skill-badge` 按新锚点重排（构图决定，待确认）。
7. **台账回填**：本件 §3 的当日增量进 `03 §5.7` 活动卡与 `PLAN_INDEX` 开放项（13 续三）。

**C 档 · 不阻塞但须披露**
8. **判据②③** 余 1 红（`verify-doc-graphs`），归并行会话在飞件（`docs/event-producer-consumer.md`），我不替其跑生成器。
9. **148 条 web 面红**中「两池都不沾」的 16 个已分档（平台 2／等元素超时 3／金样内容 5／夹具链 1／
   产品侧平台缺口 1／起进程与 boot 3／文案过期 1），**尚未逐条收口**。
10. **runtime 侧待补**：`training.paused` 字段缺失 ⇒ 生命页「暂停」与「继续」两按钮禁用条件完全相同
    （UI 侧不猜，已登记为后端待补）。**（2026-10-04 就地收窄：两按钮互斥这一半用现有 `training.pause_requested` 即可解，属 fork 内 UI 条件；而名为 `paused` 的**状态**字段确实没有⇒若判据要真实暂停态，后端仍缺。见 08 ㊵-196／㊵-197）**

## 5 · 与 M7 的边界

`01_SCOPE_AND_PHASES.md:31` 对 M7 的判据是：**实际发布 workflow 通过；代码/数据/model/package manifest 对齐；
安装启动、回滚、安全和发行包验收**；并明写「**满足并批准后才能声明发布就绪，不以『零新增』代替绿 CI**」。

当前与 M7 判据的差距：
- 「安装启动」**成立**（今日真机：四个版本装起即用、点更新链路走通）；
- 「回滚」**未对齐**——M6 第 3 项的恢复/回滚验收本就缺记录，而 M7 把它列为必达 ⇒ **这一格会同时卡住两个阶段**；
- 「实际发布 workflow」——M6 用的是**本地更新通道**（环回 ＋ 合成频道），不是 CI 驱动的正式发布；
  M7 判据要的是后者。按 `02` 的口径，**本地通道不能冒充正式发布 workflow**。

⇒ 即使 M6 收官，M7 也不是「顺手就进」：回滚验收与正式发布 workflow 是两块独立工作。

## 6 · 唯一下一步

**A 档第 1 条已于 2026-10-03 结清**（批准列入交付范围外，见
[批准书](M6_SCOPE_EXCLUSION_APPROVAL_20261003.md)）。**剩下的唯一下一步＝重取判据⑦ 的当前数字。**

理由：它是当前**唯一挡住"已落地的部分被正确计价"**的东西。台账引的 35 红出自 09-30，而
`SPEC-M6-01` 的注入行折叠 10-01 已落地（12 条纯注入 lane 里 8 条转绿）。**在重取之前，
判据⑦ 既不能判绿也不能判红**——引旧数会把已收口的一半算成未收口。

### 执行口径（照此跑，不许临时改）

- **分母必须写 144**：`test:web:built` 收 144 个文件（`08 ㊵-86` 确立）。此前台账里的 75 条是历史子集，
  **与 144 不可相减互比**。
- **前置**：`corepack pnpm run build`（判据①，client 产物必须新鲜）→ `build:web`。
  只跑 `build:web` 会整批测到旧客户端。
- **产物同一性自证**：起跑与收尾各取一次 `apps/web/dist` 的文件摘要，两次相同才作数；
  `dist` mtime 起止同值只算辅助证据，不足以作结论（08 ㊵-50 记过这个坑）。
- **只启一个驱动**：`run-face.py` 同参数两跑会改变红绿（09-30 实测批 00 差 1 个文件，13 对 14），
  且曾因误判进程已死而并发出两棵树。存活证据＝日志字节增长＋**按进程名过滤**的计数为 1
  （按命令行子串数会把 bash 包装算成驱动）。
- **池解析用块不用行**：按文件级分族（注入过期／win32 工具名偏斜／两池都不沾），
  分类器**须先自报覆盖度再报结论**，零命中即报错（`classify-face.py`）。
- **禁止**：引用任何未实测的数；改未入库的 `.jsonl` 刷绿；把「面内子集」当门定义。

### 收口后要做的一件事

把新数回填到本件 §2 的三行时序表（G5 §11 判据⑦ 行、08 §5、PLAN_INDEX 13 续三），
并**显式作废 35／36／44／65／69 这一串历史数**——它们的唯一价值是记录收口过程，
引用面一律只引新数。
## 7 · 第 3 项与第 4 项的读数改写（2026-10-04，M6 线第二十六段追加）

本节只追加，不改写上面任何一行的原文。两格的证据底座在 `08_UPSTREAM_SYNC_PLAYBOOK.md` ㊵-184 与 ㊵-185，其中承重的几条我回读过原文（`crash-recovery.e2e.ts:91` 是 `describe.skipIf(process.platform === 'win32')`；`server-restart.e2e.ts:187-198` 是同端口重启后 `graphs` 深相等、停机为 SIGTERM；`llm-retry/tests/retry.spec.ts:460` 断言 `no adapter registered for provider`；`packages/bundle/base/cordis.patch.yml:526-527`／`:535-536` 默认同挂 deepseek 与 taiji）。

- **第 3 项（持久化/恢复）**：从“部分达成＋无验收记录”改写为**实现与验收件都在、本机平台档不可执行**——恢复分支在 `agent-loop/src/index.ts:807` 一路到 `:856`，选份逻辑在 `session-persistence-jsonl/src/index.ts:1525/:1538`，真写盘读回的测试有六处（见 ㊵-184），但唯一的 SIGKILL 验收 `crash-recovery.e2e.ts` 在 win32 被跳过，且它读回走手工 `open(...,'read')` 而**不经产品 `resume()`**；`server-restart` 只作干净重启。⇒ 仍判**部分达成**；要收这一格缺一条跨平台、真 kill -9、经 `ctx.agents.resume()` 读回的验收。
- **第 4 项（provider 隔离与故障降级）**：隔离面成立且更强（两张表＋无 adapter 即抛，代码与测试都在）；**“降级”查到的结论是“harness 里没有跨 provider 改道实现”，不是“有实现没测”**。归档件那条 watchdog 今天仍活着但在 **Python 运行时平面**（`seed/language_provider.py:742/:866` 等），与 harness 路由不相交，不能充当本项执行记录。⇒ 仍判**部分达成**，且**必须先裁口径**：要么改为“故障⇒撤路由＋给用户一句错误”（可即刻测，但用户可见那句话现在也确实没有），要么先补改道实现。

本节的追加方式说明（避免下轮误改）：本文件在盘上是 core.autocrlf 的 CRLF 检出态，所以追加按文件自带的行尾写回，**没有整档重写行尾**；上面各行的字节未动。
## 8 · 第二十六段终账追加（2026-10-04，判据⑦ 的那枚红不在注入族）

只追加，不改上面各行原文。要点（全量证据在 08 ㊵-186）：判据⑦ 重取后 144 文件里**只剩一枚红**，而它的成因是**录制机与本机 PowerShell 输出语言不同**（同一 `toolCallId` 的 `tool/result`，第 234 字节起分叉于中英文 stderr），既非注入位移亦非工具名偏斜。⇒ **A 档里“R-1 剩余 4 条”结清（四条 lane 整面脱红），而 §2 那张时序表所等的“注入族证据”此路不通**：乙档（按平台分档重录）被正面否证，应撤或改判；本机 `refresh` 会把中文 stderr 烤进夹具，反而在英文机器上制造新红，故 refresh 不是这条的修法。诚实路线两条：把宿主 shell 的本地化 stderr 做有原则的令牌折叠，或把该 lane 的关闭期自校验钉成英文语言模式限定档——**都未动，等裁**。

另两笔：`clickable-links-gallery` 的未入库期望色与 HEAD 的 `--dsw-static-seed-500: rgb(86, 158, 88)` 同源（该文件已不含旧值 63, 143, 69，实测 grep 计数 0），单跑 `1 passed (1)`；R7 在开发树实测 `backend = native` 且 **xls 转换 OK**，于是收窄为“打包载荷／环境层”与“那份 openpyxl 产物特有”两支，本机分不开（无明文 xlsx 夹具、助手 python 无 openpyxl）。

覆盖率：`check:ci:coverage` rc=1、`All files 99.29｜99.19｜98.93｜99.38`、48 个文件低于 100% 阈值；红分争用（`change-scope` ×3 全 5000ms 超时、`migrate-sessions-to-v4` ×6、`dev-web` ×1）／权限（`fs-local` symlink EPERM）／真缺陷候选（那 48 个，扣除 `test:coverage-exempt-heavy` 失败所欠的部分本机判不了）三档。**判读件与“跑期不提交”冲突时该守哪条，已作为一问交 owner**；本轮我选择了入库，因此那趟读数同时被我污染，不作门裁决。

## 9 · 未闭合清单的状态差（2026-10-04 追加，只标注不覆盖上面原文）

- **§4 A 档第 2 条（重取判据⑦）＝已结**：`rerun13` ＝ 1 failed／144，读数与口径见 08 ㊵-181／㊵-183，已回填本件 §2 并作废历史数。
- **§4 A 档第 3 条（R-1 剩余 4 条须各自定因）＝已结**：四条 lane 在 `rerun13-face-01.log` 文件级为 ✓（15／7／2／4 用例）。
  同轮改判：整面剩下的那一枚红成因是 PowerShell stderr 的中英文本地化差异（08 ㊵-186），**不在注入族**
⇒ 本件原先依赖的 R-1 乙（按平台分档重录）前提被正面否证、应撤。
- **§4 B 档第 4 条（判据⑥ 剩 1 枚克隆）＝已结**（08 ㊵-159：duplication rc=0、Found 0 clones）；本轮另把他线带入的 231 条缩进代修掉，
  判据⑥ 现为**已入库面零 lint 红**，整入口余 1 条 eol-last 属 owner 裁保持未提交的那枚件（08 ㊵-179／㊵-186）。
- **§4 B 档第 5 条（R7）＝前进但未结**：开发树实测 backend = native 且 preview.xls 转换 OK（08 ㊵-186）⇒ 收窄为
  打包载荷／环境层 与 那份 openpyxl 产物特有 两支；本机分不开（仓内无明文 xlsx 夹具、助手 python 无 openpyxl）。
- **§4 B 档第 7 条（台账回填）＝已结**：本件 §3 当日增量已在 03 §5.7 与 PLAN_INDEX 13 续三，本轮再补 08 ㊵-179…187 与 G5 §11／§12。
- **§4 C 档第 8 条（doc-graphs 1 红）＝已消**：`doc-sync` 三次实测 43 passed／0 failed（最终提交态复跑 90.96s、rc=0）。
- **§4 C 档第 9 条（两池都不沾的 16 个未逐条收口）＝被本轮整面读数作废**：144 口径现在只剩 1 枚红，该分档不再需要逐条追。
- **§4 C 档第 10 条（`training.paused` 缺失）＝归属被本轮读码否证**：字段在 `api/models_runtime.py:96` 且已挂在 `GET /status` 的载荷上，Host 在 `runtime-client.ts:695` 已发布 `pauseRequested`，真正缺的是 `LifePanel.tsx:762/:763` 两行按钮条件从不读它⇒不是后端欠账，是本线一小包（2 行＋配套 100% 用例＋文档同改）。**我先前在此照抄的「属后端待补」是错的，原文不删、就地更正；全量证据见 08 ㊵-196。**
- **§1 表格第 3／4 项＝从「零记录」改为有出处**（08 ㊵-184／㊵-185）：第 3 项实现与验收件都在但唯一 SIGKILL 验收 skipIf(win32) 且不经产品 resume()；
  第 4 项隔离更强，而降级在 harness 里**没有实现对象**（agent.ts:488-490 抛错沿用同一 provider；retry.spec.ts:460 把无 adapter 报错钉成现行行为）。
  两项仍判部分达成，本件 §2 那句「四项没有一项可写成已达成」继续有效。


## 10 · 一页裁定单（2026-10-04，本轮实测后重列；**只读这一页即可批**）

取代 G5 §17 那页（它的 42 红／注入池 31 都是旧口径）。每条给：问题一句／已量到的代价／档位按**能声称的最强结论**排序／我推荐／批完可核判据。

**R-1 判据⑦ 剩的那一枚红怎么结**（成因＝录制机与本机 PowerShell 输出语言不同；77 份语料仅此 1 枚，见 08 ㊵-186／187） 【13:13Z 本条已不需要新裁：owner 已于同日裁走 甲；且本行的成因句已被否证】"成因＝录制机与本机 PowerShell 输出语言不同"作废——㊵-238 证本机侧**不是中文文案**而是 cp936 字节被按 UTF-8 解出的 U+FFFD；㊵-240 把失效点收到具体处（`pwsh-local/src/index.ts:48-49` 的 `ENCODING_PREAMBLE` 已在每条命令前强制 UTF-8，真凶是该前缀里的类型构造被 ConstrainedLanguage 禁止 ⇒ 强制静默失败后回落 cp936）；㊵-244 进一步证受限模式是**我们沙箱的 read-only 档**施加的，不是宿主显示语言。⇒ 丙 那句"英文语言模式限定档"的键也写错了（该键＝沙箱 read-only × 遗留 shell），而 乙 早被 ㊵-186 正面否证（全仓仅 1 枚、且无注入族红）。**现行修法与排序**：㊵-239① 已裁 甲＝按控制台代码页解码（未动码，实施约束见该条"其一/其二/其三"），② owner 选"⑦ 与 pwsh 7 同批定"、在此之前不动 `scaffold.ts`；有序两步＝甲（按 OEM 码页解码，需 `936→gbk` 标签表）→ 该 lane 局部折叠，**不动金样**（㊵-246 证两侧是同一条执行路径；㊵-238 证当下折叠做不出来、refresh 只会把问号烤进基线）。详见本件 §12 S-6 与 §13 判据⑦ 行。
- 丙 **该 lane 关闭期自校验记为英文语言模式限定档**：零共享面、只加一处判定与一行口径；批完判据＝整面 144 口径红数到 0，且 G5 §11 写明 win32 中文机不计入。**我推荐这条**。
- 乙 给 `assertReplaySession` 新增可选的 per-lane 折叠入参：仍是共享管子的加法（本仓折叠表现在只挂在 `captureStableAria`，`scaffold.ts:1768`），代价＝改 `scaffold.ts` ＋配套用例。
- 甲 改共享归一化 `normalizeWebSessionVolatiles`：为 1 枚夹具改整张 144 文件的比较面，与 AGENTS.md「用窄而正当的例外」相反，**不推荐**。
- 不可选：本机 `refresh`——会把中文 stderr 烤进入库夹具，英文机上反红。

**R-2 判据⑤ 第③子件的口径**（现措辞「面板读数与三个 GET 逐项一致」已与实现不同步：面板只吃 `life.getSnapshot`，Host 在 `runtime-client.ts:146-148` 刻意单源）
- 甲 把 ⑤③ 正式改成**两跳各判**：Host snapshot 对三 GET（需一次对活运行时读数）＋面板渲染对该 snapshot（ui-life 单元面已有）；可声称「按当前架构逐项核过」。**我推荐这条**。
- 乙 维持原文并补一条端到端件：但**新增 lane 会改动门定义面**（144→145 就是上一轮的教训），要先接受面变大。
- 丙 记为「不可判定」并从判据里摘出：最省事，但等于承认这一格永不验收。

**R-3 C 档第 10 条（生命页「暂停／继续」禁用条件相同）归谁**（㊵-196／197 已切分：`pause_requested` 已在载荷且 Host 已发布 `pauseRequested`；名为 `paused` 的状态字段确实没有）
- 甲 **本线做 2 行 UI 条件**（暂停在 `pauseRequested` 为真时禁用、继续在其为假时禁用）＋配套用例（每文件 100% 门要两档各能为真）＋README/JSDoc 同改：解掉症状，不动后端。**我推荐这条，1 轮可完**。
- 乙 另外要「真实已暂停」状态：那需 runtime 增加 `training.paused`，属后端（A 线那块面），要单独排期。
- 丙 只改文档不修：症状继续挂着，但零风险。

**R-4 判据④ 的覆盖率档怎么收尾**（**本页原来的前提已被否证**：那句"两趟都被并行提交挡住"只适用于超时族。独立检出跑完了，无并行仍不合格——`EPERM` 权限族 24 个文件（㊵-222 更正枚数：按完整路径重算，先前按 basename 并掉同名的一枚）两趟零变化、超时族 26→7、断言型 15→18、门自述 `1 passed, 2 failed in 1247.10s`；08 ㊵-202）【2026-10-04 ㊵-231 更新：第三趟**合格**读数已到手（窗口 05:44:33Z..06:04:09Z 内提交数 0、`run-gates: 1 passed, 2 failed, 0 skipped in 1176.64s`、rc=1）——本段那句`EPERM 24 枚两趟零变化／断言型 15→18` 是**旧两趟的文件级读数**，引用以 ㊵-231 为准：本趟 `EPERM` 头三趟同为 171、红文件 41、真候选面重算＝9 枚（含新增并已定因的 workspace 跨盘符一枚）。】 【13:13Z 本段两处数已被更新】(1) 那句"真候选面重算＝9 枚"已收到 **3 枚**且全部 pwsh 族（㊵-237；㊵-239④ 记名＝`runner`／`sandbox`／`upload-with-credentials`，它们与 ⑦ **共用同一前置动作**＝装 pwsh 7 ⇒ 一次安装同时推进两条判据）；本段带戳的那句"引用以 ㊵-231 为准"请按时刻取，㊵-231 之后又有 237／239 两次收窄。(2) 下面 丁 那条"按平台档豁免这 24 个 `EPERM` 文件"的**形状已被仓内既有机制改写**，读 §12「S-3 补」再选档（㊵-231 已证"按文件整份豁免"的论据不可用）。
- 甲 ~~在固定提交的独立检出里跑一次~~ **已执行完毕且不足以拿到合格读数**（2026-10-04，`/c/Users/23747/wt-m6` 钉 `7e9c3e84`，install／build／coverage rc＝0／0／1）：它确实消除了污染类红（超时族 26→7），但 `EPERM` 权限族 24 个文件（㊵-222 更正枚数：按完整路径重算，先前按 basename 并掉同名的一枚）一个没少，门仍红。**机时已花掉，本选项不再是可选项**；代价实测＝install＋全量 build＋20.8 分钟。
- 乙 等并行会话自己停（现在仍未停：a30 于 01:33 前还在提交）：零额外成本，时间不可控。
- 丙 判据改记「本机不可合格取得、由 CI 拥有该信号」：与仓内既有先例一致（CI 拥有的信号不本地冒充）。**甲已花掉，所以默认仍是丙**——但丙与丁的区别要说清：丙是"整个档不在本地判"，丁是"豁免已知平台族后在本地判其余"。
- 丁 **按平台档豁免这 24 个 `EPERM` 文件（㊵-222：先前写 23 枚是把两个同名 `workspace.spec.ts` 按 basename 并成了一个）、覆盖率档只判其余并记 `unverified`**：与仓内既有口径一致（CI 拥有的信号不在本地冒充），**残余风险必须同写**——若该族里藏着真缺陷，豁免会让它永久隐身；现在既没有证据说这 23 个里有真缺陷，也没有证据说没有。
- 戊 **先给本机符号链接权限再重跑该门**（开开发者模式，或在已有 `SeCreateSymbolicLinkPrivilege` 的终端里跑）：㊵-214 实测这 23 枚的失败操作**全部**是 `symlink`（两趟各 171 条，无第二种操作），所以一次本机设置可能同时消掉整族，覆盖率档**在本地就合格**，丙／丁 都不必选。代价＝本机设置一次＋约 21 分钟机时；**未验证**（我不代你改本机权限）。**㊵-216 把这条的代价精确化了**：实测本机 `file`（含 type 省略）与 `dir` 两类符号链接**都** EPERM，只有 `junction` 可以 ⇒ 需要的是符号链接创建特权（Windows 开发者模式，或让跑门的进程持有 `SeCreateSymbolicLinkPrivilege`／以管理员身份跑），**不是**某一种链接的个别问题；也因此 戊 一旦成立，受益面是那 23 枚文件／171 条断言**同时**恢复。仍**未验证**（我不代你改本机权限）。

**R-5 流程冲突（影响我以后每一轮的做法）**：「门在跑时不提交」与「判读件必须入库」相冲。本轮我在覆盖率跑期提交了两次，那趟读数因此作废（已自陈）。
- 甲 **判读件先落仓外 `C:/…`，门跑完再入库**（本轮已按此执行：㊵-200／201／202 的日志与脚本都在仓外，门后一次性入库）；乙 允许跑期提交，但该趟读数一律标注为污染；丙 长门一律走独立检出（**R-4 甲的实测修正**：独立检出能消除污染类红，但它不能消除平台档红，所以"走独立检出"不再等于"能拿到合格读数"）。

**R-7 本轮新增的一枚已入库红（㊵-221）：依赖分类器的顺序 pin 与实现不一致**（`scripts/verify-package-dependencies.spec.ts:623` 要 `@taiji/dsh-lazy-require` 两条在期望数组开头，当前实现把它们放在结尾；**集合同为 12 项、完全相同**）。
- 甲 把期望那两条挪到末尾（恢复门为绿，但把顺序固化成合同）；乙 改顺序不敏感比较（贴合该测试自写的意图「识别精确运行时导出、类型导入不算值」，**但削弱顺序判别力，需同报测不到的那半边**）；丙 先查实现为何把这两条挪后（若顺序对其他消费者有意义，问题在实现侧）——**我推荐 丙→乙**；本轮未改 spec 或实现。归属：impl 与 spec 最后修改同属 `88ad3040e`（G2 那笔 5206＋43 文件），即那笔落地时这扇门没被重跑。另有一条同 spec 的失败 `keeps generated Host schema imports…` 经三方对比（共享 2 条／隔离 1 条／单跑 1 条）判为**他线在飞件引起的争用族**，不入本裁定。**【㊵-225：丙 已由读码回答——顺序不是契约；甲 已做并验毕，本条结】**
**R-8 沙箱受限执行到底允不允许 Windows PowerShell 5.1 回退（㊵-226；这一条把两枚「平台档」合并成一个契约问题）** 三条读到的原文彼此冲突：① `packages/shell/pwsh-local/README.md:32` 把解析顺序写成「`pwshPath` → 已知安装位置 → PATH → **Windows PowerShell 5.1 作为最后手段**」，且 `:60` 明确「即使走 5.1 回退也先钉 UTF-8 输出」⇒ **回退是文档化契约**；② `packages/shell/pwsh-sandbox/tests/sandbox.spec.ts:278` 却断言受限 argv[0] `toMatch(/pwsh(\.exe)?$/u)`（本机实得 `C:\WINDOWS\System32\WindowsPowerShell\v1.0\powershell.exe`）⇒ 按 ① 这枚期望**在没装 pwsh 7 的机器上必红**；③ `sandbox-windows-acl/tests/runner.spec.ts:24` 的可用性探针是**功能性**的（真跑一次 `resolvePwshPath()`），于是它在缺 pwsh 7 的机器上判「可用＝真」、`skipIf` 不生效，而该族内部用**字面命令名 `pwsh`** 经 `CreateProcessAsUserW` 启动，Win32 2 当场失败（单跑 10 条红，㊵-223 已复现）。
- **甲 受限面只认真 pwsh 7**：`sandbox.spec.ts:278` 不动；改 `pwshAvailable()` 为「解析结果必须以 pwsh 结尾」⇒ 缺 pwsh 7 时整族**显式 skip**（不再跑 10 条红），并在 README 给 pwsh-sandbox 补一句「不使用 5.1 回退」；代价＝这台机器上 Windows 沙箱 confinement **永远没有本地验证**（除非装 pwsh 7）。 【13:13Z 本条代价句需按新读数收窄】"这台机器上 Windows 沙箱 confinement **永远没有本地验证**（除非装 pwsh 7）"已被部分否证：`packages/shell/pwsh-sandbox/tests/acl.e2e.ts` 在**未装 pwsh 7** 的本机走到真 ACL 受限启动并 `Tests 2 passed (2)`（真因是缺 `SessionProjectionRegistry` 的 plugin apply，与 shell 版本无关，㊵-268）⇒ 甲 的真实代价是"**`sandbox-windows-acl/runner.spec.ts` 与 `upload-with-credentials.spec.ts` 两族**在缺 pwsh 7 的机器上没有本地验证"（那两族仍按 ㊵-223 的 10 条红计）。另 ㊵-244 已证受限模式由我们沙箱的 read-only 档施加，选甲／乙 时把这一条当作契约事实而不是环境噪声。
- **乙 受限面接受契约里的回退**：argv 用 `resolvePwshPath()` 的结果（acl 族不再写死 `pwsh`），`sandbox.spec.ts:278` 改成 PowerShell 族（`/(pwsh|powershell)(\.exe)?$/u`）⇒ 本机两族红一起消失，**但要说清代价**：5.1 与 7 在受限环境下的行为差异由**产品承担**，且 README 已登记的 5.1 已知缺陷（非 ASCII stdin 可能错解码）会在沙箱路径上成为受支持行为，需要一处书面确认。
- **㊵-226 补 census：三处文件对同一问题有三种不同政策**（本轮逐个数过）：`apps/desktop/tests/upload-with-credentials.spec.ts` 用**裸 `pwsh`**（:70），守卫只有 `skipIf(process.platform !== 'win32')`（:117）——**既不走解析器也没有可用性探针**，所以本机必红；`packages/shell/pwsh-sandbox/tests/sandbox.spec.ts` 混用（裸 `pwsh` 7 处、`resolvePwshPath` 2 处、可用性守卫 2 处）；`packages/sandbox/sandbox-windows-acl/tests/runner.spec.ts` 裸 `pwsh` 11 处＋解析器 2 处＋**功能性**守卫 2 处，而那个守卫会被 5.1 回退喂成「可用＝真」。⇒ 这不是某一条断言写旧了，是**同一契约在三处各写一遍**；甲／乙 任选一条都要**三处一起对齐**，否则会留下第四种做法。
- 两条都要动的都不是测试断言本身，而是「沙箱用什么解释器」这条产品事实 ⇒ **我不自选**；上一轮我列的 丙-1／丙-2 其实是同一条，故合并为本条。
**R-6 维持不变、仍在你手上的四件**：H1 带凭据全量重录；R7／H2 沙箱外打包复验（R7 已收窄为「打包环境」vs「那份 openpyxl 文件」两支，本机分不开——要一份明文 xlsx）；安装器／卸载器／skill 徽章字标栅格；两条 R 级（显式删除默认工作区是否尊重、pwsh 行渲染）。【2026-10-04 ㊵-239：本行的「字标栅格」已由 owner 定为「当前 726×120 资产为最终」⇒ skill-badge 的重钉即定稿、该格可勾；其余三件仍开。】

**引用本页时的强制边界**（不因批准而改变）：M6 **仍未收官**，六项退出条件里四项「部分达成」一件都没有被批准；范围排除批准书批准的是排除不是收官；M7 判据一条都未满足（回滚与判据③同源，本地更新通道不构成发布 workflow）。

## §11 · 净结果（2026-10-04，覆盖 §1 表与 §4 清单里被本轮读数作废的句子）
> **性质**：本轮（2026-10-04）读数把 §1 表与 §4 清单里的若干句子作废。下表只登记**当前可引用**的状态与被替换的句子；§1 原文不动，引用时以本表为准。
> 范围排除批准书的三条强制义务在此一并适用：批准不改变被排除项的未完成状态；本件不构成「M6 退出判据已过」；M7 判据一条都未满足。

| 判据 | 现可主张 | 明确不可主张 | 出处 |
|---|---|---|---|
| ① 采用 artifact 清单／② shadow·canary | 已列入交付范围外并保留欠账（批准书 §2.1／§2.2） | **不得**引为「已完成」，也不得引本件任何一行作为收官依据 | `M6_SCOPE_EXCLUSION_APPROVAL_20261003.md` |
| ③ 持久化／恢复 | `doc-sync` 43 叶今日实测 `43 passed, 0 failed`（rc=0），该文档门红已结；恢复验收此前已有 ㊵-218 枚举的 8 条本机为绿的 lane，本轮再补上**硬崩溃那一支的首次真覆盖**（2026-10-04 ㊵-251／提交 `f85c6c1e`）：`crash-recovery.e2e.ts` 那层上游自带的 `describe.skipIf(win32)` 已摘、子进程自举改成 `pathToFileURL(tsxLoader).href`，`vitest.e2e.config.ts` 面实测 `Test Files 1 passed (1)`／`Tests 2 passed (2)`、1169ms（改前两条用例各 `90076ms (retry x2)` 后失败，原文 `ERR_UNSUPPORTED_ESM_URL_SCHEME`） | 不得据此说 ③ 达成：(1) 缺的是**两者的合取**：真 kill -9 产物没有被喂进产品 `ctx.agents.resume()`——这条硬崩溃 lane 的读回走手工 `open(…,'read')`＋`interruptedTurnClosers`；而 `resume()` 侧的崩溃形态是合成件（`packages/core/agent-loop/tests/resume.spec.ts:479`／`:504`／`:543` 三支，`:543` 是「torn physical tail」半条记录），㊵-218 已枚举该 spec 本机 `45 tests | 1 skipped` 为绿 ⇒ 准确说法是「缺 真崩溃产物 × `resume()` 那一条」，**不是**「缺 `resume()` 读回验收」（我先前那样写属过头话，此处就地更正）；(2) **回滚**那一格按 08 ㊵-275 改记：它是训练侧「更新·保持·回滚」门族（`02_GATES_AND_CI.md:37`／`:120`），本仓没有可补的对应验收 ⇒ 先需您裁 M6 是否声明「在线参数学习／持续适应」；(3) 这条 lane **不在任何 CI 面上被跑到**：`test:e2e` 的 runner 是 Linux（`.github/workflows/e2e.yml:56-57`＝`ubuntu-latest`/blacksmith linux），Windows 门里唯一用 e2e config 的是 `builtBinSmokeGate` 写死的 5 枚文件（`scripts/run-gates.ts:850-860`），而覆盖率分区用 `vitest.config.ts`（不收 `*.e2e.ts`）⇒ **本机执行过 ≠ 有门守着**；(4) 门闭合 ≠ 退出条件达成 | 08 ㊵-199／251；本轮 §12 第二十七段（那两段里"隔离趟该文件未进分区"一句已更正为收集面不收 `*.e2e.ts`，见 ㊵-251 ⑤）；先前"整段 skipIf、本机永不执行"的描述现应读作已结 | 【13:22Z 本行"不可主张 (1)"已结，第一列的数也要换】"缺的是两者的合取（真 kill -9 产物 × 产品 `ctx.agents.resume()`）"这一格本轮已补上并带负对照：`crash-recovery.e2e.ts` 现有两支产品读回件（`request` 派发前／`tool` 副作用前各一），且 resume 之前落盘日志里没有 `turn/end` ⇒ closer 确由产品写入，实测 `Tests 4 passed (4)`（win32 keyless 档，08 ㊵-272／273）⇒ 本行第一列那句 `Tests 2 passed (2)`、1169 ms 是 ㊵-251 时刻的读数，引用请取 4/4。本行仍然成立的是 (2) 回滚那一格（待您裁，见本页"仍在你手上的裁定"h 项）、(3) 该 lane 不在任何 CI 面上、(4) 门闭合≠退出条件达成；两行并读时以 §13 判据②③ 行为准（它已是补上后的版本）。
| ④ provider 隔离与故障降级 | 客户端四包面首次零红（`60 passed (60)`／`1079 passed／1 skipped`）；重试/隔离面㊵-219 枚举为绿（`llm-retry/retry` 31／`llm/retry-policy` 15／`gateway/control-retry` 20 用例）；**判据④ 的覆盖率档已拿到合格读数**（第三趟全跑，窗口 `05:44:33Z..06:04:09Z` 内提交数 **0**、起 `bb44585f`、门自述 `run-gates: 1 passed, 2 failed, 0 skipped in 1176.64s`、rc=1）：红文件 **41**（65→51→41）、`FAIL` 头 289／用例 97、`EPERM` 头三趟同为 **171**；㊵-231 的 A4 复验＝**5 枚"已修"全绿**；`negotiation-lifecycle` 三条件（聚合／单跑 `5 passed (5)`／按包 `144 passed (144)`）皆绿⇒㊵-211 的"只有单独跑才红"降级；新增那枚 `workspace.spec.ts` 已定因成跨盘符测试前提缺陷并修掉（`3516c304`，负对照保留判别力） | **不得**写「本机已拿到合格覆盖率读数」**这句已作废**（2026-10-04 ㊵-231／232：合格读数到手，本句写在它之前的两趟确实不合格——引用时按时刻取，别拿这句否证现在的数）；此刻仍**不可**主张的三件：(1) 判据④ 整句达成——门 rc 仍 **1**，阈值面每文件 100% 未达（本趟阈值 `ERROR` 行 **151**，旧隔离趟 135）；(2) 真候选面 **3 枚**（㊵-237：`tar` 盘符与字标钉值两枚已按裁修掉 `7929663ed`；下面列举的①②即那两枚，现应读作已结）（2026-10-04 ㊵-235：八枚里已按裁结掉三枚——notices 重生成 `cedfac08`、样式两处收敛＋一处窄档豁免 `ebb882e81`；余下＝pwsh 族三枚、`tar` 盘符一枚、字标钉值一枚）全部"等一刀"而非"未证"（notices 重生成、样式两枚、pwsh 族三枚、`tar` 盘符一枚、字标钉值一枚），收尾待你在 丙／丁 之间裁，且 ㊵-231 已证"按文件整份豁免"的论据不可用（本趟 EPERM 24 枚里 23 枚纯符号链接）；(3) `故障降级` 的零记录问题仍未处置（`packages/llm/llm-retry/src/index.ts:207` 只有 `ctx.logger.warn`）。"等并行静默或独立检出"这条路仍被否证 | 08 ㊵-201／202／203／204／230／231／232 | 【13:22Z 本行"收尾待你在 丙／丁 之间裁"是重复索要裁定】覆盖率档 owner 已裁 **戊＝给符号链接特权重跑**（08 ㊵-235①，同批还裁了 ⑦＝该 lane 局部折叠、样式＝乙、notices＝批准重生成）⇒ 本行等的是那个本机动作，不是第二句话；若特权不打算给，才回到 丁（探针条件式豁免＋记 `unverified`，其形状已被仓内既有机制改写，见 §12「S-3 补」）。
| ⑤ 客户端真实能力对齐 | 本轮补上了此前**缺失的判据⑤ 状态行**并按子件取证 | 仍是**单点缺陷修复的集合**，不是逐项裁决；口径措辞与实现不同步那条待裁 | G5 §11 判据⑤ 行 |
| ⑥ 桌面体验与端到端验收 | `duplication` 那 1 枚克隆已落地清零（`Found 0 clones`）；装机可用、默认 provider＝Taiji、离线更新通道可点升级 | R7 打包末步 `xlsx→PDF` 仍红（收窄为两支，本机分不开，需一份明文 xlsx）；字标栅格待排；**打包门不得称全绿** | G5 §12 收口段 | 【13:22Z 本行第二列"字标栅格待排"已结】owner 已裁「当前 726×120 资产为最终」（08 ㊵-239③）⇒ `skill-badge` 的重钉（`7929663ed`）即定稿、该格可勾（本页 R-6 那行已打同一戳）。其余各句不变：`duplication` 清零、R7 的 `xlsx→PDF` 仍红（需一份明文 xlsx）、以及"打包门不得称全绿"。
| ⑦ web 面 | 整面基线已重取（`rerun13`），历史数 35／36／42／44／46／49／50／65／69／81／33 **一律作废** | 剩余 1 枚红的修法等你在两条里选；未采任何边、未刷任何金样 | 本件 §2 时序表末行 |【2026-10-04 ㊵-246 更新本行的"两条里选一条"：这一刀已裁并执行（owner 先选 丁＝修好已有的 UTF-8 强制，已入库 `9df0a1c75`／`33f829a2`／`d4a2edcdb`，实测伪错误消失、U+FFFD 仍 26），本行第二列那句"两条里选一条"**已过期**；现行状态＝修法已定为有序两步（甲 按 OEM 码页解码 → lane 级折叠，见 08 ㊵-246 与退出件 S-6），**不需要动金样**；owner 已裁"等 pwsh 7 一起定"，故甲 排在其后，本行仍记 1 红。】

**本轮被本表替换掉的旧句子（原文仍在 §1／§4，只作当日历史）**：④「本机两趟都被并行提交挡住」（超时族成立、权限族不成立）；⑥「`duplication` 剩 1 枚克隆」；⑦ 那串历史红数与 §4 A 档第 2 条「判据⑦ 待重取」；§4 A 档第 1 条（已结）。
**仍在你手上的裁定**（本页 §10 R-1／R-4／R-6 与 G5 §12 第二十七段）：覆盖率档走 丙 还是 丁；⑦ 最后一枚红的修法；⑤③ 判据口径措辞；生成物／钉值那 4 件要不要刷、**依赖分类器的顺序 pin 走 甲／乙／丙（R-7，㊵-221）已由 `2fc246b2` 结掉**（读码证明顺序无消费者 ⇒ spec 期望对齐已发货顺序，只改 2 行；我 12:42Z 复跑 `Tests 42 passed (42)`，㊵-276）；C10 的 2 行 UI 修法还是后端 `training.paused`；H1 带凭据重录——其前提待您一句：本机 `.env` 里那把 `DEEPSEEK_API_KEY` 当下是否有效（08 ㊵-256：`vitest.snapshot.config.ts:36` 会 loadEnvFile，故"无 key 重录失败"那次实测并非无凭据档）；R7／H2 沙箱外复验（需一份明文 xlsx）；字标栅格。【2026-10-04 ㊵-239 更新：本行大幅收缩——覆盖率档裁 **戊**（等符号链接特权，本轮现测仍 file/dir EPERM、junction OK）、⑦ 裁「与 pwsh 7 同批定」、notices 批准并已重生成 `cedfac08`、样式裁 乙 已入库 `ebb882e81`、`tar` 盘符已按最小形态修掉、字标钉值裁「当前资产为最终」并已重钉（`7929663ed`）；**新裁一条：产品侧子进程解码走 甲（按控制台代码页），本轮未实施，三条硬约束与接口选型见 08 ㊵-239**。真正还开着的＝符号链接特权与 pwsh 7 安装（同批可推进 ④⑦ 两条判据）、H1 带凭据重录、R7 那份明文 xlsx、⑤③ 措辞、C10 修法、两条 R 级。】【2026-10-04 12:45Z 逐条核过，上面这一串里有三项已结，请以下面这份为准】仍在您手上的：**a** 覆盖率档走 丙 还是 丁（门 rc 仍 1）；**b** ⑦ 最后一枚红的修法（甲需 `936→gbk` 标签表；您已裁"与 pwsh 7 同批"）；**c** ⑤／③ 判据口径措辞；**d** C10 —— 本轮核实仍未结：`packages/client/ui-life/src` 里 `pauseRequested` **零命中**（该字段确已发布：`packages/api/life-controller/src/types.ts:186`、`runtime-client.ts:695`），且真 `paused` 状态字段仍不存在 ⇒ "面板不读"这半句成立，选 UI 两行还是后端字段仍归您；**e** H1（其前提另见 ㊵-256：需您一句 `.env` 里那把 key 当下是否有效）；**f** R7／H2 沙箱外复验（需一份明文 xlsx）；**g** `02` 里那一项签字本身；**h**（本轮新增）M6 对外声明里有没有"在线参数学习／持续适应"——它决定"回滚"是 M6 欠账还是 M7 训练侧另立项（㊵-275）；**i**（本轮新增）是否允许再花真模型配额复跑 `github-webhook-real`／`hooks` 两枚（不花配额的办法＝强制 keyless 档，但它们会变成 skipped，㊵-270）。**已核为结、不必再裁**：生成物／钉值那 4 件（skill-badge 钉值已重钉于 `7929663e` 且现值在 `skill-badge.spec.ts:37`、notices 已重生成、样式走乙已入库 `ebb882e8`、`tar` 盘符已修）、字标栅格（定为最终）、依赖分类器顺序 pin（`2fc246b2`，复跑 `42 passed`，㊵-276）、e2e 面六枚红文件（已修绿，㊵-258…268）。 【13:22Z 上面这份 a–j 里 a／b 两项要降级为"待前置动作"】a 覆盖率档＝**已裁 戊**（08 ㊵-235①；缺的是您给符号链接特权或开开发者模式，不给才回到 丁）；b ⑦ 那枚红＝**修法已裁两遍**（㊵-235① 裁 lane 局部折叠、㊵-239 新裁产品解码走 甲 并把 ⑦ 与 pwsh 7 同批定、㊵-246 定成有序两步且不动金样 ⇒ 缺的是"装 pwsh 7"这一个动作）。⇒ 真正还等您**一句话**的是：c ⑤／③ 口径措辞、d C10（UI 两行 vs 后端 `training.paused`）、e 那把 `DEEPSEEK_API_KEY` 当下是否有效、f 一份明文 xlsx、g `02` 签字本身、h M6 是否声明"在线参数学习／持续适应"、i 是否允许再花真模型配额复跑那两枚；等您**动手**的是 a（符号链接特权）、b（pwsh 7）与 ⑤ 活侧的本地 runtime／有效 key。

## §12 · 四条待裁的执行方案（本轮**不执行**；每条给了改动点、命令、实测基线与预期读数）
本节只为把「批一条就能落地」这件事做扎实：下面每条都给出**改哪个文件的哪一行**、**跑哪条命令**、**本轮实测到的基线**、**预期读数**，以及**未验证项**。本轮**一条都没有执行**。 【13:33Z 本段那句"本轮一条都没有执行"已被后续执行取代，引用时按时刻取】㊵-235／237／239 那批裁定里有几条**已落地入库**：`7929663ed` 收掉 S-5 的两枚（外加 `out-of-process.spec.ts` 把夹具根挪进 cwd＋`isAbsolute(relativeCwd)` 前置守卫，实测 `Tests 11 passed | 1 skipped (12)`，该分支此前从未被执行）、`cedfac08` 重生成 notices、`ebb882e81` 样式走乙。⇒ 本节的"待执行"集合只剩 S-1／S-2（互斥，等 pwsh 7 那条动作）、S-3 的 戊（已裁、等符号链接特权）、S-4 乙′（一行，未做）与 S-6（甲，已裁未实施）。

### S-1 R-8 甲（沙箱只认真 pwsh 7）
- 改动点：① `packages/sandbox/sandbox-windows-acl/tests/runner.spec.ts:24-26` 的 `pwshAvailable()`—现值为 `spawnSync(resolvePwshPath(), […]).status === 0`，因 `resolvePwshPath()` 允许 5.1 回退（`packages/shell/pwsh-local/README.md:32` 明写），本机**判真**；改为再核一句「解析结果必须以 `pwsh` 结尾」⇒ 缺 pwsh 7 时 `describe.skipIf`（`:35`）真正生效。② `apps/desktop/tests/upload-with-credentials.spec.ts:70` 用**裸 `pwsh`** 且守卫只有 `skipIf(process.platform !== 'win32')`（`:117`）⇒ 同一把可用性守卫补进去。③ `pwsh-sandbox` 侧不改（`:278` 的 `/pwsh(\.exe)?$/u` 正是甲的口径）。
- 命令与基线：`corepack pnpm exec vitest run packages/sandbox/sandbox-windows-acl/tests/runner.spec.ts`（本轮实测 rc=1、`Test Files 1 failed (1)`、10 条 FAIL 头）；`corepack pnpm exec vitest run apps/desktop/tests/upload-with-credentials.spec.ts`（本轮实测 7 条 FAIL 头）。
- 预期读数：两趟都从「红」变成 `skipped` 计数增加、FAIL 头 0；**代价**：本机对 Windows confinement 与凭据隔离**零验证**（acl 那 10 条正是删除逃逸／跨根删除／mode-downgrade 泄漏等围栏回归）。 【13:33Z 本条"代价"那句需收窄】"本机对 Windows confinement …零验证"作为绝对句已不成立：`packages/shell/pwsh-sandbox/tests/acl.e2e.ts` 在**未装 pwsh 7** 的本机走到真 ACL 受限启动并 `Tests 2 passed (2)`（㊵-268）。⇒ 甲 的准确代价＝**`sandbox-windows-acl/tests/runner.spec.ts` 那 10 条围栏回归（删除逃逸／跨根删除／mode-downgrade）与 `upload-with-credentials` 的凭据隔离**在缺 pwsh 7 的机器上不再有任何本地验证，而不是"confinement 整体零验证"。
- 未验证：装 pwsh 7 后两族是否直接转绿（未测）。

### S-2 R-8 乙（受限面接受契约内的 5.1 回退）
- 改动点：① 受限 argv 改用 `resolvePwshPath()` 的结果而非字面 `pwsh`：`packages/sandbox/sandbox-windows-acl/tests/runner.spec.ts`（裸 `pwsh` 11 处）、`apps/desktop/tests/upload-with-credentials.spec.ts:70`（1 处）；② `packages/shell/pwsh-sandbox/tests/sandbox.spec.ts:278` 放宽到 `/(pwsh|powershell)(\.exe)?$/u`。
- 命令与基线：同上两条单跑，另加 `corepack pnpm exec vitest run packages/shell/pwsh-sandbox`（本轮整趟里该 spec 记 1 条断言型红）。
- 预期读数：三处红同消；**代价**：5.1 与 pwsh 7 的受限差异由产品承担，且 `pwsh-local/README.md:152` 已登记的「5.1 下非 ASCII stdin 可能错解码」成为沙箱路径上的受支持行为，需要一处书面确认。
- 未验证：5.1 在该 confinement 路径上的真实行为（本轮未跑过任何 5.1 受限用例，因为字面 `pwsh` 先失败）。

### S-3 R-4 覆盖率档三条路 【13:33Z 本条三处要更新】① **戊 已裁**（㊵-235①）⇒ 它不是"待您选路"，是"等您给符号链接特权／开发者模式，给了我就跑"；只有您明确不给，才回到 丁／丙。② 本条写的基线（隔离趟 51 个失败文件、阈值面 47 文件 135 条 ERROR、`1 passed, 2 failed in 1247.10s`）是**旧一趟**；现行合格读数是第三趟 `run-gates: 1 passed, 2 failed, 0 skipped in 1176.64s`、红文件 **41**、阈值 `ERROR` **151** 条（㊵-231）⇒ "失败文件 −24"这个预期要按 41 重算，别照 51 报。③ 丁 的可执行形状已被仓内既有机制改写（不是自由裁量），见下面「S-3 补 B」；本条 丁 那段"豁免清单"读法已不适用。
- 戊（先给符号链接特权再重跑）：本轮实测**整族 24 枚文件的 EPERM 全部来自 `symlink` 一种操作**（共享与隔离两趟各 171 条原文），且 102 处 `symlink(` 调用里 85＋1 处指向**文件**（junction 只支持目录，㊵-222／217）。⇒ 开开发者模式（或让跑门进程持有 `SeCreateSymbolicLinkPrivilege`）后：`corepack pnpm run check:ci:coverage`，基线＝隔离趟 `run-gates: 1 passed, 2 failed in 1247.10s`、51 个失败文件（`EPERM` 24／超时 7／断言 18／无错误体 2）、阈值面 47 文件 135 条 ERROR；预期＝失败文件 −24、阈值 ERROR 显著下降。**代价**：约 21 分钟机时（实测 1247.10 秒）。**未验证**：是否全消。
- 丁（按族豁免＋记 `unverified`）：豁免清单必须是**两族分开**——`EPERM`/symlink 族 24 枚（分布 17 个区，最多 `packages/api/workspace-files` 4、`fs-sandbox` 2、`fs-local` 2、`scripts` 2），与 pwsh 契约族 3 枚（R-8）。**残余风险要同写**：前者含路径围栏/原子写/持久化，后者含 10 条 confinement 围栏回归；一句话版＝**豁免之后判据③ 与判据⑥ 的 Windows 侧在本地不可证**。
- 丙（整档交 CI，本机不冒充）：与仓内既有先例一致（CI 拥有的信号不本地冒充），零机时代价，但判据④ 的覆盖率一格将永远只有 CI 读数。

### S-4 两枚测试可靠性缺陷（`dsh-ci-test-reliability` 规程）
- `packages/mcp/mcp-client/tests/negotiation-lifecycle.spec.ts:127`：单跑绿（`5 passed (5)`）、跟队红；give-up 首现 635 ms（探针实测）⇒ 甲 显式 `timeout` 把假设写进参数（不推荐，把不稳定钉成合同）／乙 让它自带时序前提或进独占 lane（推荐）。
- ~~`scripts/verify-package-dependencies.spec.ts` 的 `keeps generated Host schema imports…`~~ **㊵-228 撤出本条**：它的失败原文是 `Error: Test timed out in 5000ms`（不是断言不等），单跑该条只花 483 ms ⇒ 属**并发超时的争用族**（㊵-221 的原判方向正确），不是套件内状态共享；S-4 只剩 `negotiation-lifecycle` 一条。
【本段是 ㊵-227 那枚被 ㊵-228 撤出条目的**残尾**，留此仅作过程记录，其中两句已不成立】原句「单跑与干净整趟都绿（㊵-227）」**按 ㊵-230 ① 否证**：该文件在覆盖率面里确有断言型失败，只是红在同文件**另一条**用例（`identifies exact runtime exports without treating type imports as values`，帧 `:606`／期望数组起 `:623`），而 ㊵-227 谈的是超时那条（`keeps generated Host schema imports…`）。⇒ 文件级"都绿"这个说法从未成立过；"具体共享了哪一个模块级状态"这一问随本条**作废**（㊵-228 已改判为争用超时），`policy()` 纯字面量与夹具自 `mkdtemp` 两条否证仍然有效，但它们的对象是"套件内状态共享"这个我当时下的错因，不是任何现行结论。S-4 现行成员只有 `negotiation-lifecycle` 一条。

### S-5 判据④ 面里两枚"新定因、等一刀"的缺陷（本轮读到实现层，未改码） 【13:33Z 本条两枚均已结，且落地形态既不是 甲 也不是 乙——别再照本条的"推荐乙"实施】实测出处 `7929663ed`（commit message 自带读数）：① `document-preview-license-bundle` 的 tar 调用改成**第三形**＝`:58`/`:59`/`:68` 传 `basename(packed.filename)` 并把 cwd 换成 `output`，从而不让 GNU tar 见到含盘符的操作数；**甲（`--force-local`）没采用，乙（Node 侧解包）也没采用**——理由写在那笔提交里：引入 tar 包要过依赖目录门并动 manifest（属另一刀），仓内 `scripts/primary-runtime/prepare.ts:12` 留有可日后走的先例；该枚验到 `Test Files 1 passed (1)`／`Tests 1 passed (1)`。② `skill-badge` 的 `:36` 钉值已重钉为**已发货资产** `582267ae…`（盘上与 `git show HEAD:` 逐字节相等、11777 字节；owner 裁「当前 726×120 资产为最终」＝㊵-239③），该枚 `Tests 2 passed (2)`。⇒ 本条两档（甲／乙）与"本线不代裁、不机械重钉"的悬置状态一并结束，引用 ④ 面候选数时按 3 枚（㊵-237）。
- `packages/client/ui-sidebar-documentpreview/tests/document-preview-license-bundle.client.spec.ts`：**win32 测试仪器缺陷**。同一用例两趟换因——隔离趟 `AssertionError: tar: Cannot connect to C: resolve failed`，共享趟 `AssertionError: expected Error: spawnSync D:\node.exe ETIMEDOUT`（后者的 `D:\node.exe` 就是 `process.execPath`，属负载）。取码链已读到行：`:49` 把 `pnpm pack --pack-destination` 指到 `mkdtempSync(join(tmpdir(), …))`（本机 `tmpdir()` 在 C: 盘），`:58` 却用 `resolve(packageRoot, packed.filename)` 组给 `tar -xOf` 的档案参数，而 pnpm 该字段是**含盘符的绝对路径** ⇒ GNU tar 把开头 `C:` 读成 `host:path`。**两档**：甲 给那次 `tar` 调用加 `--force-local`（已按 spec 同一条 spawn 路径实测本机解析到 `tar (GNU tar) 1.35`、其自身 `--help` 命中该旗标 1 次，对照组 `--force-notreal` 命中 0；**"加了就绿"未跑**，那要一次真 pack 出 tgz 的复跑）／乙 Node 侧解包、不依赖外部 tar。**推荐乙**（甲把"哪支 tar 在 PATH 上"变成第三个隐式依赖——这台机器上 `where tar` 与 `spawnSync('tar')` 已足以互相打架）。
- `packages/skill/skill-badge/tests/skill-badge.spec.ts:36`：**"改了被钉物、没改钉"同族第三例**（前两例＝`shell-env.spec.ts:80`/`9212c868b` 已修、`otel.spec.ts:168` 已修）。盘上资产与 `git show HEAD:` 逐字节相等（`582267aed82af7a6973bf83f6ca359199833c254825be8e0cf5deeaeadcc3e4f`、11777 字节）而钉值是 `f2c4f5ec9cbe847c0c763545c4d839efa8485bc74203733d0a0e8259f233c653`；该资产最后一笔 `a8d522882`（10-02 13:08 品牌收官，33 枚文件里落在 skill-badge 包内的只有 `assets/dsh-badge.png`）。**这一枚不是覆盖率缺陷而是 R-6 字标裁定的下游**：裁"新资产为真"⇒ 重钉一行；裁"资产应回退"⇒ 重钉无效且要走品牌面。**本线不代裁、不机械重钉**。

### S-3 补（丁 的形状已被仓内既有机制改写）／S-4 补（乙 被降级成一行）—— 2026-10-04 05:53 读到实现层，本轮未改码
**A. `negotiation-lifecycle` 的"独占 lane"其实已有现成机制，我先前把 乙 估贵了。**
`vitest.config.ts:147-148` 写着的理由与这一枚**逐字同形**：「These suites exercise process-global state, process APIs, or **timing-sensitive process I/O that worker threads cannot isolate reliably under aggregate gate contention**.」——而该处的处置不是"改用例"，是把它列进 `processBoundTests`（`:146` 起，成员含 `packages/context/time-context/tests/time-context.spec.ts`、`packages/subprocess/subprocess-local/tests/spawn.spec.ts` 等），注释同时限定「Keep the **narrow** exception in forks」。⇒ 于是 S-4 的档位重排：**乙′＝把这枚 spec 加进 `processBoundTests`（一行、有已文档化的准入理由、与本枚症状精确匹配）**，比我原先推荐的"进独占 lane/新建 config"便宜一个量级；甲（加 `timeout`）仍然最差（把不稳定钉成合同）；原 乙（新建 lane）降为不必要。**残余风险要同写**：`fileParallelism`／`maxWorkers` 全仓只有 bench／web／e2e／snapshot 几支配法（`vitest.bench.config.ts:21-22`、`vitest.web.config.ts:40`、`vitest.snapshot.config.ts:69`），**没有** `isolate`／`poolOptions` 的任何用法 ⇒ 这条"每文件独占"的路在本仓没有第二先例，只能走 `processBoundTests`。
**B. 覆盖率档"丁"该走哪种豁免，仓内已经给出准入规则，不是自由裁量。**
`scripts/coverage-exempt.ts` 的 JSDoc 把 membership rule 写死了：一支 suite 只有当**它在进程内执行的每一个被测量文件都已被别的 suite 完全覆盖**时才可豁免，"removing it from the instrumented run changes no threshold outcome"。⇒ 由此可判定：**符号链接族不能走这条**（那些 spec 所测的 src 路径正是别处覆盖不到的，豁免会把真阈值洞藏起来）；**对应先例是 `vitest.config.ts` 里按"宿主能力探针"分支的 src 路径排除**——三种既有形状：`:82` `windowsOnlyCoverageExclusions`（按 `process.platform` 静态分支）、`:97` `windowsRunnerCoverageExclusions`（同上，含"v8 coverage 从不计量子进程"这类成文理由）、`:116` `pwshCoverageExclusions`（**探针条件**：`spawnSync(resolvePwshPath(),[…]).status === 0 ? [] : [两枚 src 路径]`）。而 `:109-115` 那段注释连同一条**警告**一起给了：**"the exemption is active exactly when the suites skip — a mismatched narrower probe could exempt the file on hosts whose suites actually run"**，且探针必须跑"suites 自己的解析路"（`resolve.ts` 那个无依赖模块）。
⇒ 所以 丁 的可执行形状＝新增一组 `symlinkCoverageExclusions`，条件写成**与那 24 枚 spec 自己创建符号链接所用同一取路**（不是 `os.platform()`，也不是另起一次 `fs.symlink` 猜测），成员是那 24 枚对应的 `src/**` 路径；并按 PWSH 那段的样式写明"CI 有特权时仍执行全量线"。**这条同时回答了那个挂在 owner 手上很久的问题**（"缺符号链接权限时该叶算红还是算跳过"）：本仓既有答案是**按宿主能力探针豁免、响亮地保留 CI 侧阈值**，而不是把断言改成 skip。
**C. 一处要在读数落地时一起写的自我更正（预注册件 D 段）**：㊵-230 与 G5 第三十段那句"按当前 HEAD 复算＝13 枚文件、5 枚已修／9 枚待办"混了两个键——5＋9＝14 就是 ㊵-229 的测试级集合本身；「13」只能作为 ㊵-202 的**文件级历史读数**引用。正解：两趟时刻 14 枚（测试级）⇒ 当前 HEAD 5 已修／9 待办。

### S-6 判据⑦／④ 的最后一格：甲（按宿主码页解码）的可执行设计与它的前置
**为什么要它**：丁 已把两处内联 UTF-8 强制条件化（`9df0a1c75`／`33f829a2`／`d4a2edcdb`），实测效果＝受限宿主不再产生 `Cannot create type` 伪错误，**但输出仍是宿主码页**（该 lane 的 received 侧 `U+FFFD` 稳定 26 个：interim 纯 chcp 版一次（`d_lane.log`）＋入库条件式两次（`RC_LANE2`、`RC_LANE3`，后者在重构建之后））。机制已在仓内成文：`sandbox-windows-acl/README.md:181` 说明 `read-only` 档下 PowerShell 建不出 AppLocker 探针文件而**保守进入 ConstrainedLanguage**，此时 `[Console]::OutputEncoding` 的赋值被禁止——这正是唯一能改变写出字节的开关。

**设计（三处，全部可选参数默认不变）**
1. `subprocess-local`：给 `OutputCollector`（`src/output.ts:57` 构造子，两处解码点在 `:166` 与 `:207`）加一个**可选**的解码标签，并**统一走一个私有 `decode()`**——不要像现在这样在两处各写 `toString('utf8')`，否则会出现"读时正确、结算又变坏"的半修。默认无标签＝`utf8`，其余所有调用点零行为变化。
2. `pwsh-local`：它已经知道两件事——起的是不是 5.1（`resolvePwshPath()` 的回退），以及**这一次 pin 有没有生效**（守卫条件 `LanguageMode -eq "FullLanguage"` 的**否分支就是信号**）。把信号带出来的最小办法不是猜，而是让前缀在守卫为假时**自己印一行标记**（`else { 'PIN:skipped' }` 之类，纯 ASCII、不受码页影响），executor 在**剥掉该行之后**决定用哪个标签解码。这样"pin 成功却按 cp936 解"的主动破坏不会发生（那是 ㊵-239 ③ 指出的硬冲突）。
3. 标签值：受限宿主需要的是**控制台输出码页**，不是硬编码。Node 无 `GetOEMCP` 绑定，`chcp` 的输出是 ASCII 数字可安全解析，但**必须只解析一次并缓存**，且在无控制台／解析失败时回落 UTF-8 并**在自述里写明回落发生了**（不许静默）。**标签名已实测（本机 `v24.15.0`、`icu_small=false`）**：`new TextDecoder('cp936')` **抛 `The "cp936" encoding is not supported`**，而 `'gbk'`／`'gb2312'`／`'big5'`／`'shift_jis'`／`'windows-1252'` 均可构造并正确解出同一批字节（`d5 d2 b2 bb` → `找不`）⇒ **必须带一张 OEM 码页→WHATWG 标签的映射表**（至少 `936→gbk`、`950→big5`、`932→shift_jis`、`1252→windows-1252`、`65001→utf-8`），直接拿数字当标签会在第一步就抛；表外码页走回落并披露。

**覆盖与验收（前置，不是配套）**：本仓 `test:coverage` 按**每文件 100%** 判定 ⇒ 新增的每条分支（有标签／无标签／标签不被支持／回落发生／PIN:skipped 行被剥掉）都要有测试成员，缺一支那趟覆盖率就红。收口顺序＝定向 spec → `pnpm run build` → 单跑 `approval-composer`（判据＝received 侧 `U+FFFD` 计数**降为 0**，且 `Cannot create type` 仍为无）→ `check:ci:coverage`（跑期不提交）。**还欠一步**：金样 `snapshots/web/approval-composer/session.v4.jsonl` 含 1 行旧伪错误文本 ⇒ 无论甲 是否修好乱码，这条 lane 都要一次 `DSH_SNAPSHOT=refresh` 才可能转绿——重录属 owner 动作，我不擅自刷。 【13:38Z 本段末尾那句"这条 lane 都要一次 `DSH_SNAPSHOT=refresh` 才可能转绿"与 §13 判据⑦ 行的"不动金样"是**一条未判的冲突**，实施 甲 时必须顺带判掉】现状：丁 之后 received 侧不再产生 `Cannot create type` 伪错误，而金样 `snapshots/web/approval-composer/session.v4.jsonl` 含 1 行该文本 ⇒ 甲 修好乱码之后，这一行差到底是**"两侧同字段文本不同"（子串折叠可消，不需 refresh）**还是**"金样多一整行、received 该字段无对应内容"（折叠消不掉，必须 refresh＝owner 动作）**，本轮**没有读数**，因为 甲 未实施。⇒ **预注册（甲 落地后第一次单跑 `approval-composer` 时按此判，不许看完数再选解释）**：取 received 侧那条 `tool/result` 的字段原文两支分岔——字段存在但文本不同 ⇒ 走 lane 级折叠、不动金样；该字段整行缺席 ⇒ 折叠判为不可行，改为向 owner 请示 refresh，且**在拿到授权前不得**用 `DSH_SNAPSHOT=refresh`（㊵-238：那会把 U+FFFD 烤进入库基线，在英文机上反红）。两支都要在台账留下该次读数原文，并按 ㊵-246 的手法复核"金样两侧是否同一条执行路径"。

**要不要做的判断依据（给 owner 的一句话）**：甲 修的是"**受限档下非 ASCII 输出进入模型上下文与日志**"这一条产品缺陷；它不影响任何计数类判据的达成，但影响"模型可见内容可重建"这条仓规不变量。若你已决定给用户默认 `workspace-write`（该档为 FullLanguage，缺陷不出现），甲 的紧迫度就降为"加固"；若 `read-only` 是会被实际使用的档，甲 是缺陷修复而非优化。

**引用本节的边界**：以上是方案，不是结果；本节没有任何一条已执行，因此**不构成任何判据达成的证据**。 【13:38Z 本行的范围要限定】"本节没有任何一条已执行"只对 **S-6 自身**成立；§12 的其他条目已有落地（`7929663ed` 收掉 S-5 两枚、`cedfac08` notices、`ebb882e81` 样式＝㊵-235／237／239）⇒ 引用时别把整节读成"全未执行"，详见本节抬头 13:33Z 那条戳。

## §13 · 签字与裁定所需证据一页（2026-10-04 收束段；只读这一页即可决定要不要签，以及先给我哪一句）

**读数时刻与范围**：本段每条都标了取数命令与提交号；产品侧最后一次源码改动是 `d4a2edcdb`，其后的三笔（`0d35d5fc`／`81348a5f`／`987d5689`／`29f2bf36` 等）全在 `plans/**`。构建证据＝`corepack pnpm run build` 于 `START_UTC=2026-10-04T09:26:59Z HEAD=0d35d5fc` 那趟（其 `&&` 链后续行存在 ⇒ build 以 rc=0 通过），且 `git merge-base --is-ancestor d4a2edcdb 0d35d5fc` 返回**是** ⇒ 这次绿**覆盖**了最后一笔源码改动，不是沿用早先读数。**本段之后又入库一笔 `f85c6c1e`**：`git show --name-only` 只有两份件——`plans/active/roadmap/08_UPSTREAM_SYNC_PLAYBOOK.md` 与那份测试 `packages/session/session-checkpoint-policy/tests/crash-recovery.e2e.ts`，产品 `src/**` 零改动 ⇒ 上面的构建与覆盖率读数不必为此重取，只有 ③ 那一格按新读数改写。本段之后入库一笔 `f85c6c1e`：`git show --name-only` 只有两份件——`plans/active/roadmap/08_UPSTREAM_SYNC_PLAYBOOK.md` 与那份测试 `packages/session/session-checkpoint-policy/tests/crash-recovery.e2e.ts`，产品 `src/**` 零改动 ⇒ 上面的构建与覆盖率读数不必为此重取，只有 ③ 那一格按新读数改写。【2026-10-04 复核到 12:38Z】这句已被后续提交超出：`0d35d5fc..HEAD` 现在有 12 笔，其中**动到 `taiji-harness` 的只有 8 个文件、全部是 `tests/**`**（`built-bin`／`web-agent-presets`／`web-auth`／两枚 `built-lib`／`typescript-server`／`crash-recovery`／`acl`），而 `git log 0d35d5fc..HEAD -- 'taiji-harness/packages/**/src/**' 'taiji-harness/apps/**/src/**' '…/vendor/**' '…/scripts/**' package.json pnpm-lock.yaml` 计数 **0** ⇒ **判据① 的 `build` 读数未过期、仍是当前可引用底数**（这条是按"红归因要在固定面上、状态列会比证据先过期"的规矩复核的，不是沿用旧断言）。`doc-typecheck` 之后被证明**也计量测试文件**（㊵-274），所以这 8 个测试件由 `6056151b` 那次 `doc-sync 43 passed, 0 failed` 覆盖，而非 build 面。

| 退出条件／判据 | 此刻**可以**主张（带出处） | 此刻**不可**主张 | 需要您的一句 |
|---|---|---|---|
| 判据① 构建面 | `build` rc=0（时刻见上，覆盖 `d4a2edcdb`） | 不得说"整仓门全绿"——lint 整入口仍有 1 条 `eol-last`，属他线未入库件（owner 已裁保持未提交） | — | 【14:08Z 本行"读数未过期"从现在起是可复算的，不再是断言】命令＝在仓根跑 `git diff --name-only d4a2edcd..HEAD -- taiji-harness`（HEAD 现为 `b413acef`）⇒ 实测 **8 枚文件，全部落在 `tests/`**（`apps/cli/tests/built-bin.e2e.ts`／`web-agent-presets.e2e.ts`／`web-auth.e2e.ts`、`packages/experimental/{inspector,webworker-packer}/tests/built-lib.e2e.ts`、`packages/lsp/lsp-stdio/tests/typescript-server.e2e.ts`、`packages/session/session-checkpoint-policy/tests/crash-recovery.e2e.ts`、`packages/shell/pwsh-sandbox/tests/acl.e2e.ts`），**非测试件 0 枚** ⇒ 自那次 `build` rc=0 起**没有任何构建输入（`src`／配置／`package.json`／`pnpm-lock.yaml`）变动**，所以本行既不必重跑 build，也不得被读成"旧读数"。范围限定照写：这条只覆盖**已提交树**——本机此刻另有 2 枚他线未提交的 `taiji-harness` 件（`apps/web/tests/clickable-links-gallery.e2e.ts`、`packages/api/life-controller/tests/resource-cleanup.host.spec.ts`），我没有把它们算进任何读数。那 8 枚测试文件的类型侧由本行第三列那句 `doc-sync` `43 passed, 0 failed`（含叶子 `doc-typecheck`；最近一趟 90.06 s，实际跑在含上述 2 枚在飞件的树上）覆盖。
| 判据②③ 文档面与硬崩溃恢复 | `doc-sync` 今日三次 `43 passed, 0 failed`（83.84s／84.41s／81.95s），`hygiene` `18 passed, 0 failed`（19.58s）；**③ 的"硬崩溃恢复"那一支已在 win32 取到真覆盖**：`crash-recovery.e2e.ts` 的上游平台 skip 已摘、子进程自举改成 `pathToFileURL(tsxLoader).href`，实测 `Test Files 1 passed (1)`／`Tests 2 passed (2)`、1169ms（提交 `f85c6c1e`，成因与逐条读数见 08 ㊵-251） | 不得据此说 ③ 达成：这条只覆盖 SIGKILL-at-failpoint 的两支（request 派发前／tool 副作用前），③ 的"真 kill -9 × 产品 `resume()`"合取**已补上且带负对照，且两个 failpoint（request 派发前／tool 副作用前）各有验收**（`Tests 4 passed (4)`；非 win32 侧仍只有合成件，不许写成整格已勾）（`Tests 3 passed (3)`，win32 keyless 档；pre-resume 原始日志无 `turn/end` ⇒ closer 由产品写入，㊵-272），仍缺的是**回滚**那一格——㊵-275 核对原文后改记：那是训练侧参数回退门（`02:37`／`:120`），不是本仓一条 e2e 能补的 ⇒ 请先裁 M6 是否声明「在线参数学习／持续适应」，不声明则该格移出 M6 欠账、声明则它是训练侧另立项；已补两支修的是测试自举与新增验收件，产品侧行为未变；再加一条限定：这枚 lane 不属于任何 CI 面（`test:e2e` 只在 Linux runner，`e2e.yml:56-57`；Windows 侧用 e2e config 的只有 `builtBinSmokeGate` 那 5 枚写死文件，`run-gates.ts:850-860`）⇒ 现在可主张的是"本机真执行过"，不是"门会替我们守这条" | 第 3 项余下的"回滚"要不要我补一条验收（§10 R 级仍在） |
| 判据④ provider 隔离与故障降级 | 覆盖率档**首份合格读数**到手（窗口内提交数 0、`run-gates: 1 passed, 2 failed, 0 skipped in 1176.64s`、rc=1、红文件 41）；真候选面从 14 收到 **3 枚**（全部 pwsh 族）；五枚已修各有独立读数（㊵-231／232／237／239／7929663ed／cedfac08／ebb882e81） | **不得**说判据④ 达成：门 rc 仍 1，阈值面每文件 100% 未达（阈值 `ERROR` 行 151 条）；符号链接 24 枚（其中纯符号链接 23 枚）需特权；**退出条件第 4 项的两半都不成立**——「撤路由」**在架构层表达不出来**：`agent/request-error` 的动作联合是 `RequestErrorAction = { kind: 'retry' } | undefined`（`packages/core/agent/src/runtime-types.ts:122`），处理者只能答"再试一次"或"不处理"；src 侧挂点仅两个 compaction 与 `llm-retry`，全仓也无任何 provider 健康／摘除状态（详见 08 ㊵-248／㊵-249）⇒ 实现它要扩事件契约＋改 `agent-loop`＋同步 `docs/architecture.md`，不是加插件的量级；「给用户一句错误」也只有 `packages/llm/llm-retry/src/index.ts:207` 的 `ctx.logger.warn`，无用户可见文案 ⇒ 本项要么实现改道＋文案（文案要过 `verify-client-ui-i18n` 的字典面），要么由您把口径改写为"重试到上限＋日志"并记为有意的范围收缩 | 覆盖率档走 戊（您给特权我重跑）还是 丁（探针条件式豁免＋记 unverified）；以及"故障⇒撤路由＋给用户一句错误"这一格要不要我先补文案 | 【13:22Z 本行最后一列同样是在重复索要已给的裁定】"覆盖率档走 戊 还是 丁"——戊 已于 ㊵-235① 裁出（给符号链接特权重跑，且本轮现测 `{"file":"EPERM","dir":"EPERM","junction":"OK"}` 正是它等的那件事）⇒ 本列请读作"等动作"。本行第二列那句"退出条件第 4 项的两半都不成立"仍是**真待裁**（扩 `RequestErrorAction` 事件契约 vs 把口径改写为"重试到上限＋日志"，㊵-248／249），这一条没收回。
| 判据⑦ web 面 | 面基线 `1 用例红／144`，且**另有 1 枚文件级红已用单跑对照排除**（`server-restart` 的 `EBUSY rmdir`＋worker fork 崩溃，单跑 `1 passed (1)`、FAIL 头 0） | 不得说 ⑦ 收口：那 1 枚仍在；也不得说"重录就能好"——㊵-246 已否证该推论（金样两侧是同一条执行路径） | 无需新裁：修法已定为有序两步（甲 → lane 级折叠，不动金样），但按您的"等 pwsh 7 一起定"排队 |
| 判据⑤ 装机首启 | 判定式与观测点已写成可执行（本件"附"节） | 活侧不可取：`127.0.0.1:8000` 与 `localhost:8000` 两次探测均 `TimeoutError` 无监听；我没有擅自起后端；另已取到一条具体形态：keyless 且本地 runtime 不在场时，默认链（`base/cordis.patch.yml:87-88` 把默认 provider/model 都设成 `taiji-local`）抛 `NO_ADAPTER`、无可行动文案（㊵-262）⇒ 您那次观察请连这一处一起看 | 您那一次真机观察（输入一句话并看 UI 提示卡与进程侧日志两处） |【12:47:52Z 复探，措辞按当前实测改】`socket.create_connection` 对 `127.0.0.1:8000` 与 `localhost:8000` 均返回 **`ConnectionRefusedError`**（此前记的是 `TimeoutError`）⇒ 语义是"端口可达但无进程在听"，不是防火墙丢包或进程挂起；结论不变（活侧不可取，我没擅自起后端——本机另有 python 71 个、node 40 个在跑，起服务会撞训练资源，㊵-277）。
| 范围外／批准书 | 六项退出条件里四项"部分达成"的清单与缺口逐条有名有姓（§11） | **M6 未收官**；本件任何一行都不构成收官依据；另要计入一条**本轮新出现的未判读面**——win32 整条 `test:e2e` 面首次执行，终数 `Test Files 10 failed｜55 passed｜20 skipped (85)`／`Tests 19 failed｜183 passed｜76 skipped (278)`、rc=1：其中 4 枚已隔离定因（built 产物新鲜度档两枚同根因、`tar: gzip: stdin: unexpected end of file`；pwsh 沙箱平台档一枚，`:80`/`:110` 两条断言同时倒；`NO_ADAPTER: no adapter registered for provider "taiji-local"` 一枚，属产品默认链缺件而非仪器坏），**10 枚已全部有名有姓：6 枚已修至 win32 绿（新增 `pwsh-sandbox/acl` 2/2：补上缺失的 `SessionProjectionRegistry` apply ⇒ windows-acl 两模式首次为绿，㊵-268）（`web-agent-presets` 22/22、`web-auth` 1/1、两枚 `built-lib` 各 1/1、`typescript-server` 4/4 ⇒ **win32 的 LSP 兼容性验收本机首次真正跑到**，㊵-265），`built-bin` 由 2 降到 1，**整面已按强制 keyless 档重跑：红文件 10 → 2、`Tests 2 failed｜159 passed｜117 skipped (278)`、59.42 s、零 API 配额**（㊵-270；余 2 枚＝`agent-team-headless`＋`built-bin`，同因 `NO_ADAPTER "taiji-local"`；`github-webhook-real`／`hooks` 在该档转 skipped＝档变不是我修的，故两档读数不可相减，㊵-252 保留作带凭据档历史）＝2 枚 `NO_ADAPTER`（**keyless × 本地 runtime 不在场的环境档，非 win32、非测试坏**，㊵-262）、2 枚带凭据真实模型、1 枚沙箱/pwsh、1 枚产品侧 `subprocess-local` 绝对路径分支不补 PATHEXT**（08 ㊵-252…261；面终数要重跑才更新，而重跑会再花真模型配额 ⇒ 未经您允许不跑；余 2 枚待您点头复跑：`web-auth` 的 `expected 438 to be 384`、`hooks` 的 `promise resolved "undefined" instead of rejecting`——后者若稳定即产品缺陷，但属带凭据真实模型档、会花您的配额）。**该面读数的前提**：本机 `taiji-harness/.env` 含 `DEEPSEEK_API_KEY`，`vitest.e2e.config.ts:12` 会 `loadEnvFile` ⇒ 整面是**带凭据档**，`describe.skipIf(!key)` 不生效（㊵-255）：10 枚名单已补齐——第 10 枚不是以 `❯ … failed` 报出，而是 `Failed Suites 1` 里的 `packages/lsp/lsp-stdio/tests/typescript-server.e2e.ts`，其 win32 假因 `new URL(..).pathname` 已修，真因指向产品侧 `subprocess-local` 绝对路径分支不补 PATHEXT） | `02` 里那一项的签字本身 |【12:56Z 四处更正，以 08 ㊵-261／265／268／270 为准】① 本行"其中 4 枚已隔离定因"括注里的**"built 产物新鲜度档两枚同根因"已被否证**：单变量实验（`corepack pnpm run build` rc=0 后重跑，两枚仍同一句错）判掉了该候选，真因是本机 msys/GNU tar 1.35 把**含盘符的绝对路径**操作数按 `host:path` 解析（档案名、`-C`、反斜杠相对路径三形同源），修法＝两处都不给绝对路径 ⇒ 两枚各 `Tests 1 passed (1)`（㊵-261）。② 本行"面终数要重跑才更新，而重跑会再花真模型配额 ⇒ 未经您允许不跑"与"余 2 枚待您点头复跑：`web-auth` 的 `expected 438 to be 384`、`hooks` 的 …"**两句均已作废**：`web-auth` 已按 win32 权限位分档修绿 `1 passed (1)`（已计入 ㊵-261 的 4 枚之内），`hooks`／`github-webhook-real` 在强制 keyless 档转 skipped，而 keyless 档整面重跑**不花配额**（59.42 s、rc=1，㊵-270）⇒ 现行面终数是 `Test Files 2 failed｜43 passed｜40 skipped (85)`／`Tests 2 failed｜159 passed｜117 skipped (278)`；本行前段的 `10 failed｜19 failed` 保留作**带凭据档历史读数**，两档 skipped 数 76 与 117 差 41 枚 ⇒ **不可相减**。③ 本行末段"1 枚沙箱/pwsh、1 枚产品侧 `subprocess-local` 绝对路径分支不补 PATHEXT"**两句同归作废**：`pwsh-sandbox/acl` 的真因是缺 `SessionProjectionRegistry` 的 plugin apply、**不依赖 pwsh 7**，补上后 windows-acl 两模式本机首次 `Tests 2 passed (2)`（㊵-268）；`typescript-server` 已改成 `command: process.execPath` ＋ `args: [cli.mjs, '--stdio']` 修至 `Tests 4 passed (4)`（㊵-265），而我当时据此提的"给绝对路径分支补 PATHEXT"也被实测否证（`.CMD` spawn 直接 `EINVAL (-4071)`）⇒ 该格不需要产品改动。④ 于是本行的**有效缺口只剩两句**：余 2 枚 `NO_ADAPTER`（keyless × 本地 runtime 不在场的环境档，需您起后端或给有效 key，㊵-262／277）＋ `02` 里那一项的签字本身 ⇒ 本行由"本轮新出现的未判读面"转为"已判读、缺口两条且都在您侧"。

**需要您动手的两件（都在系统层，我不代改本机权限）**：① 装 pwsh 7（同一动作可同时推进 ④ 的 3 枚与 ⑦ 的形态判断）；② 开开发者模式或给跑门进程 `SeCreateSymbolicLinkPrivilege`——本轮现测仍 `{"file":"EPERM","dir":"EPERM","junction":"OK"}`。两件任一到位，我立刻重跑 `check:ci:coverage` 与那条 lane。

**需要您点头的两件（属"刷/重录"类，我不擅自）**：① `packages/shell/pwsh-local/README.md` 与其孪生 `README.zh.md` 两侧同改后跑 `verify-translation-pairing -- --write` 重录配对记录（本轮我改的源码功能已入库，文档欠项在 ㊵-243）；② 若将来要动 `approval-composer` 的金样，须先做完 甲，否则会把问号文本烤进基线。

**本轮我明确**不做**的两件及理由**：① 不实施 甲——它动模型可见文本与会话日志内容，且 pwsh 7 装上后这一族形状可能整体改变，您已裁"等前置拿真读数"；② 不把任何绿色 lane 改红（`out-of-process.spec.ts` 那枚"绿而无效"已按您裁加了前置守卫，若守卫变红才说明有产品缺陷——实测 `11 passed | 1 skipped` 未复现）。

## §14 · 预算用尽时的续跑清单（2026-10-04 收束段；每条＝命令／复验判据／风险，可直接复制）

1. **e2e 面判读进度：10 枚全部有名；6 枚已修至 win32 绿（新增 `pwsh-sandbox/acl` 2/2：补上缺失的 `SessionProjectionRegistry` apply ⇒ windows-acl 两模式首次为绿，㊵-268）（含 `typescript-server` 4/4，㊵-265）、`built-bin` 2→1，**整面已按强制 keyless 档重跑：红文件 10 → 2、`Tests 2 failed｜159 passed｜117 skipped (278)`、59.42 s、零 API 配额**（㊵-270；余 2 枚＝`agent-team-headless`＋`built-bin`，同因 `NO_ADAPTER "taiji-local"`；`github-webhook-real`／`hooks` 在该档转 skipped＝档变不是我修的，故两档读数不可相减，㊵-252 保留作带凭据档历史）**（08 ㊵-252…261）。`apps/cli/tests/web-auth.e2e.ts` 已按此复跑（08 ㊵-257：稳定红、内层断言帧 `:200:30`、`384`／`438` 非 HTTP 状态码 ⇒ 待判的那半是"哪一条长度断言"）；另 `web-agent-presets.e2e.ts` 的 9 条已修（shell 工具名按平台派生＋期望数组 `.sort()`，POSIX 恒等已本地证明），win32 单跑 `Tests 22 passed (22)`，其间的 `SessionAlreadyExistsError` 放大效应属未修的仪器缺陷（08 ㊵-258）；`apps/cli/tests/profiles/acp/tests/hooks.e2e.ts` 涉真模型，**会花您的配额 ⇒ 我不自跑**。命令：在 `taiji-harness/` 内 `corepack pnpm run test:e2e <完整路径>` 逐枚隔离复跑。复验判据＝单跑 rc 与同形错原文逐字对齐后再分档（同形＝一根因）。**便宜的整批复验命令（零 API 配额，㊵-269 实测 `Test Files 7 passed (7)`／`Tests 33 passed (33)`、10.75 s）**：在 `taiji-harness/` 内一次带上七个路径——`packages/session/session-checkpoint-policy/tests/crash-recovery.e2e.ts`、`packages/shell/pwsh-sandbox/tests/acl.e2e.ts`、`packages/experimental/webworker-packer/tests/built-lib.e2e.ts`、`packages/experimental/inspector/tests/built-lib.e2e.ts`、`packages/lsp/lsp-stdio/tests/typescript-server.e2e.ts`、`apps/cli/tests/web-agent-presets.e2e.ts`、`apps/cli/tests/web-auth.e2e.ts`。风险两条：用 `pnpm exec` 会因缺 `npm_execpath` 造 built 型 spec 的假红；隔离日志文件名必须用完整路径——本轮两枚同名 `built-lib.e2e.ts` 被我按 basename 覆盖了一份读数。
2. **built 产物档那两枚**（`webworker-packer`／`inspector` 的 `built-lib.e2e.ts`）。命令：先 `corepack pnpm run build`，再复跑这两枚。复验判据＝`gzip: stdin: unexpected end of file`／`tar: Error is not recoverable … expected 2 to be +0` 这组同形错消失。风险＝整面跑前的产物新鲜度（`build:web` 只打旧 lib 那条已登记）。 【13:02Z 本条已结，处方作废】两枚的真因不是产物新鲜度：`corepack pnpm run build` rc=0 后重跑仍同一句错，㊵-261 已把该候选判掉。真因＝本机 msys/GNU tar 1.35 把**含盘符的绝对路径**操作数按 `host:path` 解析（档案名、`-C`、反斜杠相对路径三形同源），修法＝两处都不给绝对路径 ⇒ 两枚各 `Tests 1 passed (1)`。除非改过 src，复跑不必再先 `build`。
3. **pwsh 族 4 枚**（判据④ 余 3 枚＋`pwsh-sandbox/tests/acl.e2e.ts` 1 枚，DENIED 与 OK 两条断言同时倒、9ms 即失败）。命令（需 owner 先装 pwsh 7）：`corepack pnpm run check:ci:coverage`，**跑期不提交**。复验判据＝pwsh 契约族红消失，且阈值面 `ERROR` 行数按趟记录不互抄。 【13:02Z 本条从 4 枚减为 3 枚】`packages/shell/pwsh-sandbox/tests/acl.e2e.ts` 已从本条移出：真因是缺 `SessionProjectionRegistry` 的 plugin apply，**与 pwsh 7 无关**，补上后本机 `Tests 2 passed (2)`（㊵-268）。④ 余下的 pwsh 族 3 枚仍按本条命令走（需 owner 先装 pwsh 7、跑期不提交）。
4. **符号链接特权**：本轮现测 `{"file":"EPERM","dir":"EPERM","junction":"OK"}`。到位后同样走 `check:ci:coverage`；纯符号链接那 23 枚的归属才有意义。二者任一到位我都立刻重跑，不改本机权限、不代裁。
5. **判据③ 的产品读回验收（已完成，win32 档）**：真 SIGKILL 产物接进产品 `ctx.agents.resume()` 这件事**已落地**——`packages/session/session-checkpoint-policy/tests/crash-recovery.e2e.ts` 现含两支产品读回件（`request` 派发前／`tool` 副作用前各一），都带负对照（resume 之前落盘日志里没有 `turn/end`）⇒ `Tests 4 passed (4)`；装配配方是现成的（同包 `tests/fixtures/crash-child.ts:1-45`），**不加依赖、不动锁**（㊵-272／273）。仍待的两件：这枚 lane 不在任何 CI 面上（`test:e2e` 只有 Linux runner，Windows 门里用 e2e config 的只有 `builtBinSmokeGate` 写死的 5 枚文件），以及「回滚」那一格——按 ㊵-275 它属训练侧「更新·保持·回滚」门族（`02_GATES_AND_CI.md:37`／`:120`），需您先裁 M6 是否声明在线参数学习／持续适应。
6. **判据⑦ 最后 1 枚红**：有序两步不变——先 甲（按宿主 OEM 码页解码，需 `936→gbk` 等标签表；`new TextDecoder('cp936')` 本机抛不支持），再该 lane 局部折叠；**不动金样**（㊵-246 已证两侧同一条执行路径，refresh 只会把 U+FFFD 烤进基线）。按您的"等 pwsh 7 一起定"排队。
7. **判据⑤ 装机首启**：`127.0.0.1:8000`／`localhost:8000` 两次探测均无监听，我没擅自起后端。需要您那一次真机观察，看两处（UI 缺 key 提示卡 ＋ 同刻进程侧日志原文）。
8. **属"刷/重录"类、等您点头的两件**：`verify-translation-pairing -- --write` 重录 pwsh-local README 双语对（两侧须同改）；H1 带凭据重录 `session-snapshot` 的 replay 场景——**但这条的实测前提已失效**：`vitest.snapshot.config.ts:36` 会 `loadEnvFile` 本机 `taiji-harness/.env`（其中有 `DEEPSEEK_API_KEY` 键名，我只按名核、未取值），所以"无凭据重录只改写半截夹具"那趟其实带着一把**有效性未知**的 key ⇒ 53 条红分不开"key 无效／端点无响应／确需重录"三种解释。先请您给那一句（那条 key 当下是否有效），再决定 H1 是执行还是改记"本机不可判"（08 ㊵-256）。
9. **仍要您裁的口径两件**：退出条件第 4 项"故障⇒撤路由＋给用户一句错误"——现契约 `RequestErrorAction = { kind: 'retry' } | undefined`（`packages/core/agent/src/runtime-types.ts:122`）表达不出来，要么批准扩事件契约＋改 `agent-loop`＋同步 `docs/architecture.md`，要么把口径改写为"重试到上限＋日志"并记为有意收缩；以及 e2e 面这 6 枚未判读红要不要进 M6 还是留 M7。 【13:02Z 后半句的数已变】"e2e 面这 6 枚未判读红"已由 08 ㊵-265…270 全部判读、其中 6 枚修至 win32 绿 ⇒ 现行待裁的是"整面 keyless 档余 2 枚 `NO_ADAPTER`（`agent-team-headless`＋`built-bin`，keyless × 本地 runtime 不在场的环境档）要不要算进 M6"，它落在您"起后端或给有效 key"那一句上，不属仪器侧。

10. **产品侧一条不对称（本轮新找到、代价已定价、未动码）**：`packages/subprocess/subprocess-local/src/index.ts:151` 的绝对路径分支不补 PATHEXT，而裸名分支 `:170-172` 补。命令：让绝对路径也走 `executableCandidates` 后复验 `corepack pnpm run test:e2e packages/lsp/lsp-stdio/tests/typescript-server.e2e.ts`。复验判据＝那 4 条 `spawn …\.bin\typescript-language-server ENOENT` 转绿（或给出 `.CMD` 解析成功的具体读数）。风险三条：本仓 `test:coverage` 按每文件 100% 判 ⇒ 每条新分支都要测试成员；这是 `shell`／`ssh`／`terminal`／`lsp` 全部绝对路径 spawn 的公共入口，须同提交改 `subprocess-local/README.md` 与 JSDoc；`check:ci:coverage` 一趟约 1176 s 且跑期不得提交。另记同形写法第二处 `packages/ptc-runtime/ptc-runtime-node/src/launch.ts:30`（本机无害，路径含空格会被百分号编码弄失真）。详见 08 ㊵-254。 【13:02Z 复验判据作废，登记本身保留】本条给的复验命令已不可用：`typescript-server` 已由测试侧改成 `command: process.execPath` ＋ `args: [cli.mjs, '--stdio']` 修至 `Tests 4 passed (4)`（㊵-265），那 4 条 ENOENT 已经不红，不会因产品改动"转绿" ⇒ 该 lane 不再能判这条产品改动的真伪。而我据"补 PATHEXT"提的形状也被实测否证（`.CMD` spawn 直接 `EINVAL (-4071)`）⇒ 真要动这条分支须另立判据（例如绝对路径命中 `.EXE` 的单件验收）。本条作为"不对称已定价、未动码"的登记继续保留。

**M6 未收官**：`02` 里那一项签字本身仍未给；本清单不改变任何一格的"可主张／不可主张"。


> **⚠ 一条影响本页多处表述的实测（2026-10-04 12:49Z，08 ㊵-278）**：`taiji-harness` 没有独立 `.git`（toplevel ＝ `E:/Seed`），其 20 个 workflow 文件位于**子目录** `.github/workflows/`，而 GitHub 只加载**仓根**的 workflows；仓根只有 `ci.yml` 且其中 `taiji-harness` **零命中**，`gh workflow list` 在该 remote 上只返回 "CI" 与 "Dependabot Updates"。⇒ **本仓当前没有任何 harness 门在 CI 上执行**：本页出现的"CI 会跑／CI owns the platform matrix／`test:e2e` 只在 Linux runner／`dsh-win-ci` 跑 windows-complete"等表述，都应读作**上游带过来的惰性文件**；所有 harness 门读数（build、doc-sync、hygiene、coverage、web/e2e 面）都只有**本机证据**。判据⑥ 与"绿 CI"类结论请按此重述，收官声明里不要写"CI 已验证"。是否把门真接上 CI（甲＝移出仓根或独立仓／子模块；乙＝根 `ci.yml` 加 job 进 `taiji-harness/` 跑 `check:ci:*`）＝新增给您的一项决定。

> 【12:52Z 收窄，08 ㊵-279】上面这条**不新增 M6 欠账**：`02_GATES_AND_CI.md:14`／`:146` 把"端到端／正式 CI"列在**产品采用/发布与 M7 发布**行，不在 M6 六项退出条件里。它推翻的是"把本机实测写成 CI 已验证"这类**表述**（属过度声称修正，不是缺口新增）；也不许反过来拿"CI 没接线"解释 harness 门的 rc=1——那些门由我在本机跑，红了就是红了。甲／乙两条接线方案记为 **M7 前置工程**。 【13:51Z 本条的处置已落进 02 本体，本块自此有仓内正式出处】`02_GATES_AND_CI.md` §4 新增一条（提交 `8264aaad`，紧跟 `:155` 那条历史 CI 读数之后）：按该页 `:154`「正式发布须实际 workflow 通过」与 `:153`「三套状态分账」写明 **harness 的远端 workflow 一套是空集**，并把两句禁止钉在门定义页上——不得写"CI 已验证"、也不许反向拿"CI 没接线"解释 harness 门的 rc=1；甲（把 workflows 移出子目录）／乙（根 `ci.yml` 加 job）同处记为 M7 前置工程、未裁。⇒ 上面 ⚠ 块那句"应读作上游带过来的惰性文件"现在不再只由本件背书。
