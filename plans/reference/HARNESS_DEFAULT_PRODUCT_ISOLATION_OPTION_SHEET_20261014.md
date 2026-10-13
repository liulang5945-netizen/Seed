# `default product isolation` 那枚红——第二份一页纸（十二裁 ② 的交付物；决定权在 owner）

现取时刻＝2026-10-13T16:46Z，HEAD `435d7877e`＋本轮未提交的文档改动。**这份件只报证据与两条互斥修法，不替你选**；件里每个结论都带 `文件:行`，每条都我自己读过。

## 一、红是什么（CI 现读）

- 位置＝`m7-ci` run **38431031552**（2026-10-13T16:30:14Z 触发，9m7s），job `doc-sync + hygiene (ubuntu)`：`doc-sync (43 leaves)` **通过**，`hygiene (18 leaves)` ＝ `16 passed, 2 failed`。
- 失败叶子逐字＝`FAILED default product isolation (15.26s, exit 1)` 与 `FAILED package invariants (2.98s, exit 1)`；前者消息＝
  `verify-default-product-isolation: @taiji/dsh-desktop optionalDependencies -> @taiji/dsh-experimental-schedule-bundle: default product must not include experimental packages`。
- **这两枚不是本轮造成的**：上一趟 run 38420482505 的 step 表里 `hygiene (18 leaves)` 是 `-`（未跑，被 `X doc-sync` 挡在前面）。十一裁 ①② 把 doc-sync 修绿之后 hygiene 才第一次被执行到 ⇒ **newly visible，不是 newly caused**。
- 第二枚（`package invariants`）我已按你授权修掉：缺的是"本包为何不发布不变量配套入口"那句理由，`companion` 一词在我提交前后的同一份 `README.md` 里都不存在（`git show 435d7877e~1:<path> | grep companion` ＝ 空），所以那枚包 10-09 建起来时就漏着。补完两侧后本地读数＝`verify-package-invariants` rc=0、`verify-package-readme-model-experience` 仍 rc=0、`doc-standard`＋`package-invariants` 两本 spec 39 passed、`verify-translation-pairing` rc=0（1104 对全一致）。

## 二、这枚红为什么不是"顺手删一行"

门只给了**一条**声明过的豁免路，而这条已经用上了：

- `scripts/verify-default-product-isolation.ts:3`（政策句）：唯一例外是 launcher 在 `OPTIONAL_BUNDLES` 里点名的 bundle——"shipped for the person to switch on, selected by no shipped template, its own dependency graph outside the default product's"。
- `packages/boot/app-boot/src/profile.ts:188-193`：`OPTIONAL_BUNDLES` 现在确实含 `@taiji/dsh-experimental-schedule-bundle`（第 192 行）。
- `scripts/verify-default-product-isolation.ts:86-93`：门对名单里每枚 bundle 只校验两件事——它必须是 `apps/cli` 的运行时依赖（`apps/cli/package.json` 里有）、且必须声明 `dsh.bundle.patch`。
- **但它照样红**，因为遍历那条腿不看名单：`:100-108` 的 `add()` 只要从默认产品图里碰到 experimental 包就 push 失败，而 `:133`（`verify-default-product-isolation.spec.ts`）用 `it.each(['dependencies', 'optionalDependencies', 'peerDependencies'])` 把三个 section **全部**当作"默认产品包含"——这是被测过的意图，不是漏洞的副作用。
- 那行依赖也不是冗余声明：桌面构建产物里**真的装出了这枚 bundle**——`apps/desktop/.desktop-build/targets/win-x64/dsh/desktop-runtime.json:609`、`.../dsh/package.json:126`（版本 `0.1.7-alpha.1`）、`.../package-set/desktop-packages.json:838-840`（带 `.tgz`）。出处：`apps/desktop/package.json:46`。
- 那行是谁加的：`9d3b77266`（2026-10-09，「发布 Schedule 可选 bundle——客户端里开启即得 Automation 能力」）；更早的对齐是 `6e1295986`（2026-10-02，官方可选插件包注册 schedule-bundle）。我本轮没碰过 `apps/desktop/**`。

⇒ 结论：**发布侧按 `OPTIONAL_BUNDLES` 的语义把能力发给了桌面，门侧的意图却是"app 的 optionalDependencies 也算默认产品"**。这两句现在互相矛盾，且矛盾一直存在，只是没人跑过 hygiene。

## 三、两条互斥修法（各自都会改掉一句主张）

| 选项 | 动作 | 它承认了什么 | 半径／代价 | 我先要看到的东西 |
|---|---|---|---|---|
| 甲 改发布侧 | 撤掉 `apps/desktop/package.json:46` 那行 optionalDependencies，桌面改走别的路径拿这枚 bundle | 承认"桌面默认产品不该带 experimental 包"这一句优先 | 桌面失去"开启即得 Automation"这条现成路径（构建产物里那三处会消失）；要另找发货通道，动的是打包链不是文档 | 桌面端有没有第二种装载入口（launcher 侧名单能否直接驱动桌面打包），我没查到，需要一次真实构建验证——**这一条会花机时，且改的是 `apps/desktop`，属产品可见改动** |
| 乙 改门侧意图 | 在 `add()` 那条腿上给"launcher 名单里的 bundle ＋ app 的 optionalDependencies"开豁免，并同步 `:3` 政策句与 `spec:133` 的 `it.each` | 承认"名单点名的可选 bundle 即使是 app 的 optionalDependency，也不算默认产品包含" | 这是**主张变更**：要改政策句、要改一条被测过的断言的覆盖面（三个 section 不再同等对待），并同步 `docs/architecture.md`。改动面小（一个门＋一个 spec＋文档），但语义上是你替"默认产品"重新划界 | 你的一句话：豁免只给"在 `OPTIONAL_BUNDLES` 里且 app 级 optionalDependencies"这一形，还是给所有 optionalDependencies（后者会让 `it.each` 那三支断言的覆盖面缩掉） |
| 丙 不修，登记 | 新债进 `05_TECH_DEBT_REGISTER.md`，带本件的 run 号与出处 | 承认这是排序决定，不是缺陷判断 | CI 的 hygiene 面继续红这一枚；下一次任何提交都要重新解释"这枚红不是新造成的" | 无 |

## 四、这份件不声称的三件事

1. 不声称"门错了"或"发布错了"——两边各有出处支撑，判定需要一句产品主张，那是你的。
2. 不声称甲的成本：我没跑过桌面构建（本机没有 pnpm，且构建属重型动作），所以"撤掉后还有没有别的路"是**未测**，不是"没有"。
3. 不声称 `package invariants` 那枚的修法覆盖面：我只补了 `web-search-scraper` 这一包的理由句，**没扫全仓**还有没有别的包缺同一句（门现读只点名这一枚 ⇒ 至少 CI 面上只这一处，但那是门的判定面而不是我的普查）。
