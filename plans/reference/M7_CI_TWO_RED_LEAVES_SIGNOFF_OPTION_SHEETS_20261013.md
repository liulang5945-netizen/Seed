# `m7-ci` 两枚红叶子的签字选项（十裁⑷ 交付物；一页一件，勾选权在 owner）

现读出处＝run 38397747172（本批 push 后，㊵-776）汇总行 `run-gates: 41 passed, 2 failed, 0 skipped in 239.32s`，
未过叶子逐字＝`translation pairing (74.91s, exit 1)` 与 `package README model experience (16.95s, exit 1)`。
两枚都与 ㊵-766 那一次同名 ⇒ 本批没新增红，也没有漏修：红的是**要人签的主张**，不是我能力半径里的缺陷。
本地复现不需要 pnpm：在 `taiji-harness/` 里直接 `./node_modules/.bin/tsx scripts/<verify-…>.ts`（本轮已实测）。

## 第一件：`package README model experience`

**门现在说什么**（原文，一行）：`packages/web/web-search-scraper/README.md: must contain one or more complete model-context entries`。

**这张门的形状**（现读 `scripts/verify-package-readme-model-experience.ts`）：两张登记表＋一条互斥。

* `NO_MODEL_EXPERIENCE_SECTION`（第 32-39 行）：主张"该包没有任何模型面章节"，每条必须留一句审计理由。
* `SENTENCE_MODEL_EXPERIENCE`（第 46 行起）：主张"模型面只有一句话规模"，取值 `kind: none` 或 `indirect` ＋理由。
* 第 323-333 行的三条守卫：条目必须指向**被扫到的包**、理由不许为空、**同一包不许同时出现在两张表里**。

**三个选项（我不自选，因为三条都是对包主主张的签署）**

| 选项 | 写下去的含义 | 代价／半径 | 我先要看到的东西 |
|---|---|---|---|
| 甲 进 `SENTENCE_MODEL_EXPERIENCE`（`kind: indirect` ＋理由） | 承认"抓取结果会经别的包进模型上下文，渲染责任在消费者" | 只改那张登记表一行；跑该叶子＋`doc-standard` 册约 15 分钟，零产品码 | 「scraper 的输出到底进不进模型请求」的出处（包内消费者或调用点） |
| 乙 进 `NO_MODEL_EXPERIENCE_SECTION` | 主张更强：这包**完全不**碰模型面 | 同上，但一句错话会长期留在审计表里 | 同甲，且要能解释为什么没有消费者把它的内容送进上下文 |
| 丙 在该包 README 里补完整 model-context 条目 | 不改登记表，文档自己承担主张 | 代价最大（要写清哪些内容进模型、谁渲染），但主张最弱最稳 | 我可以起草，但"条目算完整"的判定权在门＋包主 |

**我不做的事**：不为了让这条叶子变绿而填一张表——那等于我代包主签署一个我没核对的模型面主张。
勾选后我执行：改那一行／那一节 → 本地跑 `./node_modules/.bin/tsx scripts/verify-package-readme-model-experience.ts` → 再跑 `doc-standard` 册（`node_modules/.bin/vitest run scripts/doc-standard.spec.ts`）→ 提交推送后取 CI 读数。

## 第二件：`translation pairing`

**门现在说什么**（本地复现原文，rc=1）：

* `docs/config-catalog.md: out of sync — content no longer matches the pair's last confirmed-consistent state in docs/config-catalog.i18n.yaml (bring the other side along, then re-record with --write)`
* `docs/config-catalog.zh.md:` 同一句，指向同一份 `.i18n.yaml`

**为什么两侧同时红**（现读，不是猜）：`config-catalog` 是**派生文档**——本仓已记过这个耦合：改包内 JSDoc 会红在派生的 config-catalog（`plans/active/roadmap/08` 里那条"中文孪生件按逐块同码豁免，两侧同改＋按对重录 pairing"）。两侧都改了、而"上次确认一致状态"还是旧的 ⇒ 门拦的是**签名过期**，不是内容错。

**三个选项**

| 选项 | 动作 | 代价／半径 | 需要先证明的事 |
|---|---|---|---|
| 甲 对齐后重录（门自己给的路） | 先把 en／zh 两侧都重生成为当前源码状态，再 `--write` 重录确认状态 | 一次 `--write` 改写的是"已确认一致"的凭证；跑 pairing 册约 1-2 分钟 | **我必须逐块 diff 两侧**才敢称"内容等价"——我不为没核对的内容签一致性 |
| 乙 只重录不核对 | 直接 `--write` | 最快，但把"可能不等价"的两侧钉成"已确认一致" | 我认为不该选：这条门的价值就在凭证上 |
| 丙 登记为已知不修（十裁⑷ 的另一路） | 写进 09／债表，CI 面不再算阻塞 | 绿灯信号对本仓永久失效（现状本就已被这两枚红拦住） | 无需内容判断，纯粹是排序决定 |

**一条要防止的混淆**：本仓还挂着一条"H1 的重录需要真凭据"的说法，而它**未被证实**（08 ㊵-255／256）。本件这条 pairing 与任何 API 凭据无关——两件事不要写在一起，也不要互相代答。

**授权点**：甲／乙／丙 选一个；若选甲，是否授权我在两侧 diff 核对完成后执行一次 `--write`（一次授权不外推到其他件）。
## 更正（2026-10-13 ㊵-786，十一裁之后按实测回填；原句一字不删）

第一件「代价／半径」那行当时写「只改那张登记表一行；跑该叶子＋`doc-standard` 册约 15 分钟，零产品码」——**这句把半径估小了**，实测半径是四处改动加两次重录：登记表一行、`README.md` 与 `README.zh.md` 各改写为"已审短句形"、下一节 `<a id=…>` 锚点位置（该文件把锚点写在标题**之前**，会被算进 `## Model Experience` 块，而门要求该块恰好三行）、以及**两次** `--write`（config-catalog 一对、scraper README 一对；第二次另向 owner 取过授权，没有外推第一次）。原句保留，因为"我当初怎么估的"本身是要留的证据。

第二件「需要先证明的事」那句仍然成立，但核对只做到机器可证那一层：两台生成器 `gen-config-catalog.ts`／`gen-config-catalog-zh.ts` 重跑后 `git status` 不列那两份件（＝两侧本来就在当前源码态，缺的只是签名），逐块核对结果是 148 枚 `## ` 级标题两侧同一、18 枚点号配置键 `keys_only_en=0`／`keys_only_zh=0`、标题深度零错位——**这不证中文译文句子级对等**，我没有逐句读那 148 段的译文。勾"甲"时请把这一层限制算进你的判断。

一句话结论：两件都已落地（本地两枚 verify 全 rc=0），但**本件不声称 CI 面已绿**——本机没有 pnpm，聚合门 `doc-sync` 未跑，`m7-ci` 的读数要推送后现取并抄 run 号与时刻。
