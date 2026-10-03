# Seed 桌面端「生命」页面体验规范（2026-10-03）

> 本文件是**体验/交互/信息架构**层的落地规格，承接已定稿的
> `UI_LIFE_VISUAL_DIRECTION_20261003.md`（视觉 token 与形状不再讨论，只在使用性上提出异议）。
> 目标读者：负责 Phase 3 落地的前端工程师。所有结论都可直接编码，不需要再回来问。
>
> 输入文件：`src/client/LifePanel.tsx`（1103 行）、`src/client/LifePanel.module.css`（341 行）、
> `src/client/locales.ts`（346 行）、`tests/panel.client.spec.tsx`（740 行，23 个用例）。
>
> **本轮硬性原则：默认不新增 locale key。** 全部方案优先通过「复用既有文案 + `aria-labelledby` 指向既有标题 +
> CSS」实现。确实绕不开的新文案集中在附录 A，共 **0 条必需、3 条可选**。

---

## 0. 先做三处事实更正（简报与本仓实况不符）

这三条会直接改变落地清单，请先按实际情况核销：

| # | 简报里的说法 | 实际情况 | 影响 |
|---|---|---|---|
| 1 | 「表格 `<th>` 没有 scope」 | `LifePanel.tsx:450-455` **六列全部已有 `scope="col"`** | Q6 的表格语义缺口不在表头，而在**行标题**：每行第 2 格（文件名）应承担 `scope="row"`，让「选择 seed_beta.pt」这类复选框名称有行上下文。不需要去补 col scope。 |
| 2 | 「confirmLine 渲染在区块底部，需要把它显示出来」 | 确认文案其实**两处都有**：按钮原地变文案（`LifePanel.tsx:754-756 / 822-824 / 969-971`）+ 页面最底部 `.content` 末尾的 `.confirmLine`（`:215`） | 真正的修复不是"把 confirmLine 挪到看得见的地方"，而是**删掉 `:215` 那一行**——它是纯重复且永远看不见；按钮本身就在用户视线里，已经是最佳位置。强行再复制一份到顶部会叫 `getByText(en.confirmStop)` 从 1 个命中变成 2 个，直接叫红用例。详见 §1-Q3。 |
| 3 | 「act 行是自定义栅格不是 dl」 | 是的，但 `<div class=facts>` 是**网格容器**，10 处调用点，可直接替换为 `<dl>`；每格一个独立 `<dl>` 反而是错的（会把一个 12 项的事实网格拆成 12 个描述列表） | Q6 的正确改法是：容器 `<dl class=facts>` + 每格 `<div class=fact>` 包裹 `<dt>`/`<dd>`。HTML5 允许 `<div>` 作为 `<dl>` 的直接子节点分组 dt/dd。 |

补充一条**简报未提及但会卡住重排**的硬约束：

> `tests/panel.client.spec.tsx:186-197` 用 `filePickers()[0]` 取训练集 picker、`[1]` 取知识 picker。
> **「训练」在文档顺序上必须先于「知识」**，任何把 ④知识 提到 ③训练 之前的重排都会叫红
> 「uploads a picked dataset…」与「uploads a knowledge document…」两个用例。这是 §1-Q1 不重排的决定性理由之一。

---

## 1. 七个问题的结论

### Q1 · 信息架构与优先级

**结论：六个区块顺序不动（来源 → 生命读数 → 训练 → 知识 → 记忆与巩固 → 宿主）；不引入"默认折叠"；改为「六个区块全部可折叠 + 全部默认展开 + 折叠状态在会话内保持」。训练区内部补为四个具名子组，不再是一片平地。**

理由三条：

1. **顺序不动**：(a) 来源→生命→训练→知识→记忆→宿主 是"这份数据从哪来 / 现在什么样 / 我做了什么 / 它记住了什么"的因果链，视觉方向已把它作为锁定前提；(b) 上面的 filePicker 顺序约束；(c) 重排解决不了任何用户抱怨——owner 抱怨的是"不好读、找不到、操心"，不是"顺序错"。
2. **不默认折叠**：诊断性区块（①来源、⑥宿主）承载的恰恰是"这份读数能不能信"的信号——`staleBadge`、`unavailableTitle`、`downBadge`、`startupIncomplete`。**默认折叠等于把最该先看到的信用信息藏起来**，与"少操心"的目标正好相反。而且算笔账：⑥宿主在视觉方案下就是**一个 facts 网格（约 72px 高）**，折叠它省下的滚动距离不到全页的 3%，代价却是掩盖 `健康/模型/鉴权` 三个决定整页可信度的读数——收益为负。同理①来源只有 3 个 fact。
3. **改为"可折叠且默认展开"**：真正该减的不是可见性，是"每次进来都要重新滚过同样的东西"。给用户一次折叠操作、之后一直生效，比替用户决定藏什么更符合"用户掌控"。

**训练区内部分层（本轮最实质的 IA 改动）**——现有 DOM 顺序已经基本正确，问题在于**四个子组没有名字、没有视觉边界、权重全一样**：

```
③ 训练（<details open>，可被整体折叠）
   ├─ A 运行状态          ← 不可折叠。状态 tags + 进度条 + 运行警告。最高频。"现在在干嘛"
   ├─ B 训练数据          ← 上传 / 目录名单 / 选中计数 / 删除所选。次高频。"喂什么"
   ├─ C 运行控制          ← 启动·暂停·继续 | 停止·强制解锁。动作行。"干什么"
   └─ D 检查点            ← 沿用现有「收起/展开」按钮。低频、最长。"从哪续"
```

落地方式：四个子组各自用一个 `<div className={css.organ}>` 包裹、组内第一个元素是 `<h4 className={css.organTitle}>`，
动作行通过 `aria-labelledby` 指向同级 `<h4>`。**不新增 DOM 层级嵌套超过一层，不移动任何现有节点的相对顺序**（一旦移动节点顺序，就会连带碰到 §5 里那 3 个由「行内按钮同名」引起的用例）。唯一的顺序微调见 Q3 / Q7 的「选中计数移入动作行」。

---

### Q2 · 长页面导航

**结论：不加侧边目录，不加顶部 chip 导航。只做两件零成本的事——给六个区块加稳定 `id` 供外部深链直达，以及让"折叠 + 状态带 + sticky header"承担导航职责。**

理由（针对桌面应用右面板）：

1. **宽度账算不过来**。右面板常见有效宽度 360–520px。侧边 TOC 至少要 120–160px（含中文区块名 4–7 字），等于**拿走 30–40% 的阅读宽度**，去换 6 个用户第二次几乎不再访问的锚点。这在 960px 上限的内容区里是净亏。
2. **chip 导航会制造三重重复**。`数据源 / 生命读数 / 训练 / 知识 / 记忆与巩固 / 宿主` 这 6 个字符串已经在 `<details>`（作为 summary 的可访问名）和 `<h3>` 里各出现一次；再加 chip，读屏用户在页面顶端会连听三遍同一组名词。更麻烦的是它会和可访问性树里的 summary/h3 争同一个名字，最后只能靠 `aria-hidden` 打补丁——用隐藏来掩盖自己造的噪音，不是解法。
3. **页面长度的主因已经被处理掉了**。长度来自训练区，而训练区已有「目录折叠」「检查点折叠」两级；本轮再把六个区块本身变成 `<details>` 后，用户可以把不关心的部分压缩到只剩标题行——这是比跳转更符合"信息已经在手边，只是暂时收起来"的形态。
4. **跳转目标本来就是一次性的**。用户来这一页的真实任务是"看一眼状态 / 启动或停止训练 / 上传文件 / 删点东西"，都是**先扫描后定位**，不是"我知道我要去第 5 块"。sticky header + 状态带 + 折叠足够。

需要做的两件小事：

```css
/* 让键盘 Tab 到任何元素时不会被 sticky header 盖住 */
.page { scroll-padding-top: 96px; }
/* 让 #life-host 这类深链跳转/锚点定位不被 sticky header 吃掉标题 */
.section { scroll-margin-top: 96px; }
```

`id` 取值固定为 `#life-source` / `#life-readings` / `#life-training` / `#life-knowledge` / `#life-consolidation` / `#life-host`，
供宿主 Agent 或其它面板以 `location.hash` 直达，不依赖任何新增文案。

---

### Q3 · 状态操作分区与二次确认

#### 3.1 三级分组与排序规格

| 级别 | 成员 | 视觉 | 排序规则 |
|---|---|---|---|
| **主操作** | `trainStart`、`uploadSend`（选中文件后）、`runConsolidate` | primitives 的 `variant="primary"`，**每组最多一个** | 永远排在该组最左 |
| **次要** | `uploadPick`、`uploadCancel`、`trainPause`、`trainResume`、`activateRow`、`activateBuiltin`、`checkpointsFold/Unfold`、`lifeStart`、`actionFeed/Sleep/Play` | 默认 ghost | 紧跟主操作 |
| **危险** | `trainStop`、`trainReset`、`lifeStop`、`deletePickedDatasets`、`deletePickedCheckpoints`、`deletePickedKnowledge` | `.dangerAction`（视觉方向已定） | **推到该动作行的右端**，并与前面用一条竖分隔线隔开 |

**训练「C 运行控制」行的具体顺序（不重排成员，只改视觉分组）**：

```
[ 启动训练 ](primary)  [ 暂停 ]  [ 继续 ]   │   [ 停止 ](danger)  [ 强制解锁 ](danger)
                                          ↑
                            右侧危险簇，宽屏 margin-left:auto 贴右缘，
                            窄屏随 flex-wrap 掉到下一行
```

理由：`暂停/继续` 是**同一状态的互斥开关**，必须紧邻（现状已是）；`停止/强制解锁` 语义完全不同——一个是"结束这次训练"，一个是"撬锁"，且都不可撤销。把危险簇与前面用**空间**隔开，比靠颜色更符合"防错"启发式：用户手指已经形成了"左边是开跑、右边是毁掉"的位置记忆后再也不需要读文案。

**已知数据缺口（不本轮解决，需提给 runtime）**：`LifePanel.tsx:766-767` 的 `trainPause` 与 `trainResume` 的 `disabled` 条件完全相同（`busy || !active || stopRequested`），
因为快照里没有 `paused` 字段，所以**运行中与已暂停两态下这两个按钮都同时可点**。这是「暂停 / 继续」这一组按钮唯一的信息缺口；建议 runtime 增加 `training.paused: boolean`，届时补成互斥 disabled。**本轮不要靠前端猜 UI 状态来掩盖它。**

#### 3.2 二次确认：保留两阶段（原地变文案），不换 dialog

**结论：保留。理由：①这是 Electron 右面板，居中对 dialog 没有空间、且焦点陷阱在窄面板里体验更差；②现有 4 个用例完整钉住了"第一次点击进入待确认、第二次执行"的契约，换成 dialog 等于把交互契约和 4 个用例一起重写，收益却只是"更醒目"——而醒目的需求已经由下面的修复解决。**

#### 3.3 确认态的可见性：靠"位置"而不是"复制"

现状的真实问题是 **`LifePanel.tsx:215` 那行 `.confirmLine` 永远在页面最底部，用户在中段按下按钮时根本看不到它**。
修复方案不是把它挪到顶部（会撞-guard `getByText` 唯一性），而是：

1. **删除 `LifePanel.tsx:215` 的 `.confirmLine` 与 `:197` 的 `confirmKey` 变量**（连带删以避免 unused 报警）。
   保留按钮原地变文案——按钮就在用户光标下、就在焦点里，是唯一不可能丢失的位置。
2. **危险动词的影响清单内联到按钮旁边**（见 Q7），`aria-describedby` 关联，随 armed 状态出现/消失。
3. **全局失败条 `failureText`（`:214`）改为 sticky 底部停靠条**——这是唯一真正需要全局可见的反馈，
   且它是**纯 CSS 改动**（`position: sticky; bottom: 0`），不移动 DOM、不复制字符串、不产生布局跳动（见 V-2 异议）。

```css
/* 停靠在滚动容器底部，出现时不推动任何内容 */
.content > .errorLine {
  position: sticky;
  bottom: 0;
  z-index: 6;
  width: min(100%, 960px);
  margin: 0 auto;
}
```

**明确不做**：不给二次确认加自动超时撤销（"我刚才点过吗"比多按一次更糟）；不改 `successLine` 的既有位置（它就在动作旁）。

---

### Q4 · 空 / 加载 / 错误 / 不可用 四态统一规格

统一成一张决策表，**四个层级各有一种形态，不允许同一层级出现两套视觉**：

| 层级 | 触发 | 形态 | 是否带语义角色 | 实现位置 |
|---|---|---|---|---|
| **L1 首屏加载** | `state.state === 'loading'` | 单张卡片 + 两条骨架条 + 一行说明文案，**不用六张分区骨架**（快照是一次性整体返回，分块骨架没有真实对应物） | `role="status"` | `LifePanel.tsx:174-176` |
| **L2 流中断** | `state.state === 'error' \|\| snapshot === undefined` | 独立 `.statePanel` 容器（错误带 + 重试按钮居中），替换掉脆骨头的相邻兄弟选择器 | `role="alert"` | `LifePanel.tsx:178-193` |
| **L3 子系统未应答** | `datasetsUnavailable` / `knowledgeUnavailable` / `knowledgeFilesUnavailable` / `artifactsUnavailable` / `consolidationUnavailable` | 统一的 `.unavailableNote`：警告浅底 + 1px 状态边界 + **主文本色**（不是 tertiary 灰字） | 无（不是 alert，它描述的是状态不是突发事件） | 见 §4 清单，5 处 className 替换 |
| **L4 空态** | `checkpointsEmpty` / `datasetsEmpty` / `knowledgeEmpty` / `noProgress` / `noReport` / `specNotReady` / `noReadings` / `noWeaknesses` / `noNotes` / `notYet` | 保持纯文本 `.muted`，**不做插图、不做占位卡片** | 无 | 现状保留 |

三条补充规则：

- **L3 与 L4 不得混用**。凡带"不可用 / 未应答"语义的（来源是运行时没答上来）一律走 `.unavailableNote`；
  凡带"还没有 / 尚无"语义的（运行时答上来了，只是为空）一律走 `.muted`。这两者今天全混在 `.muted` 里，
  导致"能力没起来"和"暂时没数据"长得一模一样——这是 owner 说的"操心"的一个具体来源。
- **空态必须带行动出口**。现已满足：`datasetsEmpty` 上方就是上传区（`LifePanel.tsx:678` 在 `:702` 之前）；
  `knowledgeEmpty` 同理。**落地时不要把这个顺序反过来。**
- **pending（单次动作进行中）不给骨架、不改按钮文案**。只给按钮加 `aria-busy="true"`，其余靠既有 disabled。
  理由：改按钮文案会破坏 5 个用例的 `getByRole('button', { name: en.trainStop })` 类断言，而收益几乎为零。

---

### Q5 · 响应式与窄面板

#### 5.1 表格：拍板选 **`.tableScroll` 包裹层**

**结论：采用 `.tableScroll` wrapper，否决 `.table { display: block; overflow-x: auto }`。**

四条理由：

1. `display: block` 会让 `<table>` 退出表格盒模型：`border-collapse: collapse` 失效、列宽由最长内容自由决定、
   之后想sticky 表头不可行；且滚动容器是表格自身，横向滚动条会跟着内容一起滚走。
2. `display: block` 上的 `overflow-x: auto` 会让 `<table>` 不再扮演 `<table>` 的布局上下文，**列**的意义只靠 `scope="col"` 撑着，容错面变窄。
3. wrapper 方案保留 `display: table`，**表格语义、单元格角色、列计数、DOM 顺序全部不变**，`getAllByRole('button')`、`getByRole('checkbox')`、`getByRole('rowheader')` 全部不受影响。
4. **测试影响：零。**

```css
.tableScroll {
  width: 100%;
  overflow-x: auto;
  overscroll-behavior-x: contain;
}

.tableScroll .table {
  display: table;         /* 关键：撤销视觉方案里的 display: block */
  width: 100%;
  min-width: 720px;       /* 6 列自然最小宽实测约 738px；720 是留有一点压缩余量的稳妥值 */
  overflow: visible;
  border-collapse: collapse;
  font-size: 12px;
  white-space: nowrap;    /* 只有第 2 列（文件名）覆写为 normal */
}
```

wrapper 同时承担键盘可达性（见 Q6-5）。

#### 5.2 facts 网格：**把单列断点从 420px 提到 560px（这是对视觉方案的修订，见 §6-V4）**

推导：容器宽度 = 面板宽 − page padding(32×2) − section padding(20×2) = **面板宽 − 104px**（≤640px 时是 − 64px）。
视觉方案的 `minmax(min(100%,180px), 1fr)` 意味着两列需要 ≥ 368px 容器宽，即**面板宽 ≥ 472px** 就会出现两个只有 180px 宽、里面还塞着等宽字体文件路径的格子。

结论：**把 `@container (max-width: 420px)` 改成 `@container (max-width: 560px)`**，
让二列只在面板宽 ≥ 664px 时才出现。窄面板下宁可一行一个 fact，也不要两个都在换行的格子。

#### 5.3 meter（三列 88px / 1fr / 48px）

推导：最小占用 = 88 + 8 + 80 + 8 + 48 = **232px**。面板宽 ≤ 296px 时才挤压——桌面客户端几乎不会出现。
所以：**保留视觉方案 `@container (max-width: 420px)` 里缩到 `72px / 1fr / 42px` 的那一档作为中段，
只在 `@container (max-width: 240px)` 以下才切成两行兜底**（标签 + 数值同行两端对齐，轨道独占第二行）。这条只是保险，不必优先实现。

#### 5.4 面板 > 960px 时的处理

`.content { width: min(100%, 960px); margin: 0 auto }` 保持。不要为了让六张卡撑满而再放宽——超过 960px 后
fact 网格会出现 5 列以上，扫读路径从"Z 形"退化成"报表"。多出来的宽度留给滚动条和留白是正确的。

---

### Q6 · 无障碍与键盘（最小改动，优先原生语义）

全部改动的共同原则：**能去掉 `role` 就去掉，能用原生标签就用原生标签，名字优先靠 `aria-labelledby` 指向页面上已经存在的标题，实在没有才用 `aria-label`。**

| # | 改动 | 位置 | 为什么 |
|---|---|---|---|
| 1 | `.facts` 容器 → `<dl>`，每格 `.factLabel` → `<dt>`、`.factValue` → `<dd>`（外层保留 `<div class=fact>`） | 10 处 `.facts` + `Fact()`（`:304-311`） | 这是"标签—值"对的原生语义。`aria: none` 也能读，但读屏用户失去"这是定义列表、有 12 项"的整体感 |
| 2 | `<section aria-label={t('sectionX')}>` → `<details><summary><h3 id>` | 6 处 section 根（见 §4） | 现状 section 的 `aria-label` 与内部 `<h3>` 文本**完全重复**，读屏会连念两遍且标题与区域无关联 |
| 3 | 六个 action 行的 `role="group"`：能删则删，保留的一律改 `aria-labelledby` 指向同级 `<h4>` | `:423` `:679` `:749` `:759` `:817` `:921` `:967` `:1042` | `:759` 的 `aria-label={t('sectionTraining')}` 与外层同名；`:679` 的 `aria-label={t('uploadTitle')}` 与按钮文案 `uploadPick` 在 zh 下**是同一个字符串**「上传训练文件」；`:749` 与 h4 `datasetsTitle` 同名 |
| 4 | meter 轨道加 `role="meter"` + `aria-valuenow/min/max` + `aria-label={label}` | `:318` | 现在只是一根有宽度的 `<span>`，读屏读到的是空白；`42.5` 这个数旁边没有任何"这是 0–100 的量表"信息 |
| 5 | 进度条轨道加 `role="progressbar"` + `aria-valuenow`，用 `aria-labelledby` 指向所在区块 `<h3>` 的 id | `:518` | 同上；名字取「训练」而非新增文案 |
| 6 | 表格滚动容器 `role="group"` + `aria-labelledby` + `tabIndex={0}` | `Checkpoints()` `:446-447` | 横向滚动区必须可聚焦，否则纯键盘用户读不到右侧列（WCAG 2.1.1） |
| 7 | 每行第 2 格 `<td>` → `<th scope="row">` | `:476-482` | 见 §0-1。让「选择 seed_beta.pt」这类名称在行内有归属 |
| 8 | 行内按钮加文件名后缀：`aria-label={\`${t('resumeFrom')} ${cp.filename}\`}`、`${t('activateRow')} ${cp.filename}` | `:488` `:493` | 现状 6 行会渲染 6 个**同名**「从这里续训」按钮，读屏列表里完全无法区分。与文件自身的复选框命名约定（`${t('colSelect')} ${cp.filename}`，`:471`）保持一致。**这是本轮唯一会叫红用例的 a11y 改动，见 §5** |
| 9 | `<details>` 的 `<summary>` 需要 `list-style: none` + `::-webkit-details-marker{display:none}` + `cursor: pointer` + focus ring | 新增 CSS | `<summary>` 原生可键盘操作，不需要任何 JS 状态 |
| 10 | `.page { scroll-padding-top: 96px }`、`.section { scroll-margin-top: 96px }` | CSS | 防止 Tab 到的元素被 sticky header 盖住 |

**明确不做的**（避免"为了 a11y 而破坏可用性"）：

- 不把 `.fileInput` 从 `display:none` 改成视觉可见。现状是「可见 `<Button>` → 程序化 `.click()`」，
  键盘触发 Button 时同样会打开文件选择器，链路完整且没有幽灵 tab 停靠点。
- 不把禁用态从 `disabled` 改成 `aria-disabled`。`disabled` 有一个真实代价（不进 tab 顺序、读屏可能跳过），
  但 3 个用例直接断言 `.disabled === true`，且本页禁用态旁边都有原因文案（`checkpointLocked`）或明显的分组上下文。改为 P2。

---

### Q7 · 风险动作的影响范围说明

**结论：需要，而且必须把被删对象的名字列出来；但清单要放在按钮旁边而不是单独一段提示。**

理由：删除不可恢复（`confirmDelete*` 文案已含「不可恢复」），而按钮文案里只有**数量**。
让用户按下"再确认一次"时自行回忆"我刚才勾了哪两个"是典型的 recall-over-recognition，
尤其在 5 秒一次快照刷新、勾选项会跨刷新保留（`LifePanel.tsx:243-283` 刻意保留用户勾选）的情况下——
**用户完全可能在不知情时勾着三个数据集过了两分钟，然后按了删除**。

实现约束（关键）：**不要把文件名拼进按钮文案**。`getByRole('button', { name: t('confirmDeleteDatasets', {count:'2'}) })`
是对**完整按钮名**的精确断言，改按钮文案会一次性叫红 3 个用例（`:435` `:470` `:524`）。

正确做法：

```
┌─ B 训练数据 ──────────────────────────────────────────┐
│ 2 项已选                                              │  ← 从下方的灰色 <p> 移到这里，紧邻删除按钮
│ ⚠ 将删除：consolidated/night-1.jsonl                  │  ← armed 时才出现，≤3 项全列，>3 项限高滚动
│          simple_zh/dialogue_extended_clean.jsonl      │
│ [ 再次点击确认删除所选 2 个数据集（不可恢复） ](danger) │
└───────────────────────────────────────────────────────┘
```

- 清单只在 `confirming === 'deleteDataset' | 'deleteCheckpoint' | 'deleteKnowledge'` 时渲染，
  通过 `aria-describedby` 关联到按钮，**不新增任何文案**（列表项就是路径/文件名本身，不加前缀词）。
- 不做数量截断：`选了 12 个时列前 3 个 + "还有 9 个"`需要一个新 locale key，而直接全列 + `max-height: 88px; overflow-y: auto` 零成本且不丢信息。
- **不做假撤销**：后端不支持恢复，UI 不得给出"已撤销"暗示。成功文案 `deleteDone` 保持现状。
- 另一个真实误触面（一并修）：把 `{count} 项已选` 从 roster 下方的灰色 `<p>`（`:739-742`）移到删除按钮**同组的最左侧**，
  让"我勾了几个"和"我要删"在空间上相邻。纯位移，文本不变，用例 `getByText(/2 selected/)` 仍绿。

---

## 2. 任务流程（两个主任务）

```
【任务 A：喂一份语料并开跑】
落在 #life-training（默认展开）
  ① 看 A 运行状态 ── "空闲 / 训练中" tag ──────────────┐
                                                        │ 训练中则上传与勾选项全部 disabled
  ② B 训练数据 → [上传训练文件] → 系统文件管理器        │
         └─ 选中后按钮行变 [上传][取消]，下方给出已选文件名+大小
         └─ 超过 200MB：立即 alert（role=alert），不读文件、不发请求
         └─ 上传成功 → 成功文案 + 自动勾选该行 + 刷新名单
  ③ 勾选/取消勾选数据集（目录可折叠，不折叠其它目录）
         └─ 0 项已选 → 提示"按运行时默认语料训练"
  ④ C 运行控制 → [启动训练](primary) ──────────────────┘
  ⑤ 训练中：A 区出现进度条 + 六个数（损失/步数/轮次/剩余/速度）


【任务 B：清理旧检查点】
落在 #life-training → D 检查点
  ① D 区标题行右侧 [收起/展开（N）]
  ② 勾选目标行        ← 使用中/下次启动的两行 checkbox 直接 disabled + title 给出原因
  ③ [删除所选检查点](danger)
         └─ 第一次点击：按钮原地换成"再次点击确认删除所选 N 个检查点（不可恢复）"，
            同时按钮上方出现影响清单（列出文件名），焦点保持在按钮上
         └─ 第二次点击：执行 → 成功文案 + refresh + 勾选清空
         └─ 中途点了别的动作 → 自动解除 armed（现有 `setConfirming(null)` 行为）
  ④ 失败 → 全局 `.errorLine` 停靠在滚动容器底部（role=alert），正文是 Host 的稳定错误码，不是 RPC 原文
```

---

## 3. 界面线框要点（按区块）

```
┌ .page ──────────────────────────────── W−64 ───────────────────────────┐
│ ┌ .header (sticky) ──────────────────────────────────────────────────┐ │
│ │ ▎生命                                                    22/30 600 │ │
│ │   Taiji 本地运行时的读数与控制                           13/20 400 │ │
│ └────────────────────────────────────────────────────────────────────┘ │
│   gap 16                                                               │
│ ┌ .section / <details open> ─────────────────────────────────────────┐ │
│ │ ▸ ▎区块标题（summary，整行可点击，含 3px Seed 绿短标）     16/24 600│ │
│ │   ── content（gap 16）──────────────────────────────────────────── │ │
│ │   <h4> 子组名（可选）                                    13/20 600 │ │
│ │   [fact][fact][fact]   每格 label 12/18 secondary 上 / value 13/20 │ │
│ │                        primary 下，里宽 10×12，圆角 8，弱交互底    │ │
│ │   <h4> 子组名                                                      │ │
│ │   [ 主操作 ]  [ 次要 ]  [ 次要 ]  │  [ 危险 ]  [ 危险 ]            │ │
│ └────────────────────────────────────────────────────────────────────┘ │
│   ×6（卡片间距 16，无阴影，0.5px border-l2，圆角 12，bg-layer-1）      │
│ ┌ .errorLine (position: sticky; bottom: 0) ───────────────────────────┐ │
│ │ 操作失败文案（role=alert，1px error 边界 + error 浅底 + primary 字）│ │
│ └────────────────────────────────────────────────────────────────────┘ │
└────────────────────────────────────────────────────────────────────────┘
   滚动寄主：.page 是唯一滚动容器；纵向滚动条贴右缘（已修）
```

**必须保持的三条空间纪律：**

1. **一张卡只有一层结束**。六个 `<details>` 是唯一的卡片层；内部的 organ/fact/uploadBlock/progress 都用**弱交互底 + 无阴影 + 最大 8px 圆角**，绝不再加 border + shadow。
2. **动作行永远属于它作用的对象**。`删除所选数据集` 必须在 roster 正下方、`删除所选检查点` 必须在表格正下方、`启动/停止` 必须在同一个"运行控制"子组里。今天它们散在一行一行里，用户必须靠读文案猜作用范围。
3. **53–68ch 是上限**。所有说明性段落（`uploadHint`、`memoryUsed`、运行时警告 `runWarning`）都要限制宽度，不要横跨 960px。

---

## 4. 落地清单

### 4-A · CSS 改动（`LifePanel.module.css`）

> **权威版本是视觉方向 v2.1 §7，请以那份为准。** 视觉的 v2.1 已经把本表的大部分条目直接并入
> （A1 / A2 / A3 / A4 / A5 / A6 / A7 / A8 / A9 / A10 / A13 / A14），所以下表的作用是**让你核对它们为什么存在、
> 以及哪些还需要补**，而不是让你照抄第二份 CSS。唯一没进 v2.1 的是 A12（优先级最低，可最后做）。
> 编号 V-* 的另见 §6 异议。

| # | 选择器 | 改动 | 说明 |
|---|---|---|---|
| A1 | `dl.facts` | `margin: 0`；`dd.factValue { margin: 0 }` | `<dl>/<dd>` 有 UA 默认 margin，不重置会让网格错位。**这是 Q6-1 的必需配套** |
| A2 | `.statePanel` | 新增：`width: min(100%, 960px); margin: 24px auto 0; display:flex; flex-direction:column; gap:12px; align-items:flex-start` | 替换视觉方案里 `.page > .errorLine` 与 `.page > .errorLine + button` 的兄弟选择器（见 V-2） |
| A3 | `.content > .errorLine` | 新增：`position: sticky; bottom: 0; z-index: 6; width: min(100%,960px); margin: 0 auto` | Q3.3 的底部停靠条 |
| A4 | `.tableScroll` | 新增：`width:100%; overflow-x:auto; overscroll-behavior-x:contain`；`.tableScroll .table { display:table; width:100%; min-width:720px; overflow:visible }` | Q5.1；**同时删掉视觉方案里 `.table { display:block; overflow-x:auto }`** |
| A5 | `.table thead th` | 把视觉方案里 `.table th { background/font-size/color }` 的三条收窄到 `thead th` | Q6-7 加了 `tbody th`，否则表头样式会污染行标题 |
| A6 | `.table tbody th` | 新增：`font-size:12px; font-weight:500; color: label-primary; background: transparent; text-align:left` | 行标题要与数据行同权，不能是 11px tertiary |
| A7 | `.sectionSummary` | 新增：`display:flex; align-items:center; gap:8px; margin:0; padding:2px 0; cursor:pointer; list-style:none` + `&::-webkit-details-marker{display:none}` + `&::marker{content:''}` + chevron 伪元素随 `details[open]` 旋转 | Q6-9 |
| A8 | `.sectionSummary:focus-visible` | 并入现有 focus ring 规则：`outline: 2px solid business-primary; outline-offset: 2px` | 现状 focus 规则只覆盖了 button/`.groupToggle`/`.datasetRow input`/`.table input` |
| A9 | `.impactList` | 新增：`margin:0; padding-left:16px; max-height:88px; overflow-y:auto; font-size:12px; list-style:disc`；路径走 `--ds-font-family-code` | Q7 |
| A10 | `.page` | 加 `scroll-padding-top: 96px`；`.section` 加 `scroll-margin-top: 96px` | Q2 / Q6-10 |
| A11 | `@container (max-width: 420px)` | **改为 `560px`** | Q5.2（V-4） |
| A12 | `@container (max-width: 240px)` | 新增：`.meter { grid-template-columns: 1fr; }`，标签+数值同行（`.meterLabel`/`.meterValue` 靠 `justify-content: space-between` 包一层），轨道独占下一行 | Q5.3 兜底，优先级最低 |
| A13 | `.unavailableNote` | 新增：`:extends` 视觉方案 `.unavailable` 的配色（warn 浅底 + 1px warn 边界），但 `padding: 6px 10px`、`font-size:12px`，`.unavailableNote .muted { color: label-primary }` | Q4-L3。与 Source 那一块多行 `.unavailable` 区分为「块」与「注」两级 |
| A14 | `.actions .actionCluster` | 新增：`display:flex; align-items:center; gap:8px; margin-left:auto` + 前置 1px 竖向分隔伪元素 | Q3.1 危险簇推右端；窄屏靠父类 `flex-wrap` 自然掉行 |

> `.dangerAction` 的用法与按钮清单沿用视觉方向 §8.1，**一个字不改**。

### 4-B · TSX DOM 改动（`LifePanel.tsx`）

> 详细代码片段见 §5。这里只列清单与风险等级。
> 🟢 = 不会叫红任何用例　🟡 = 不叫红但需冒烟复核　🔴 = 会叫红，必须同步改测试

| # | 位置 | 改动 | 风险 |
|---|---|---|---|
| B1 | `:12` import | 增加 `useId` | 🟢 |
| B2 | `:174-176` loading 分支 | `<p class=loading>` 加 `role="status"`，并插入两个空的 `aria-hidden` `<span className={css.loadingCard} />` / `<span className={css.loadingCardWide} />`（视觉 v2.1 §8.4） | 🟢 |
| B3 | `:178-193` error 分支 | 包一层 `<div class={css.statePanel}>`；`<p class=errorLine>` 加 `role="alert"` | 🟢 |
| B4 | `:197` + `:215` | **删除** `confirmKey` 变量与 `<p class=confirmLine>` 那一行 | 🟢（见 §0-2，按钮已承载该文案） |
| B5 | `:214` | 位置不动（保持在 `.content` 最后一个子节点） | 🟢（靠 CSS A3 停靠到底部） |
| B6 | 6 处 section 根：`:350` `:413` `:657` `:908` `:998` `:1065` | `<section aria-label>` → `<details className={css.section} id="life-*" open>`，首个子节点 `<summary className={css.sectionSummary}><h3 className={css.sectionTitle} id={headingId}>`，收尾 `</details>`；同步修改各自 `return` 的开闭 `:349/:412/:656/:907/:997/:1064` 与挂载处 `:207-212` | 🟡（`<summary>` 无直接文本子节点，`getByText` 仍只命中 `<h3>`；需冒烟复核 6 个 `getByText(en.sectionX)` 仍唯一） |
| B7 | `:304-311` `Fact()` | `.factLabel` span → `<dt>`，`.factValue` span → `<dd>` | 🟢 |
| B8 | 10 处 `<div className={css.facts}>`：`:352` `:377` `:394` `:659` `:782` `:914` `:1004` `:1021` `:1033` `:1067` | → `<dl className={css.facts}>`（收尾同步） | 🟢 |
| B9 | `:316-321` `Meter()` | 轨道 `<span className={css.meterTrack}>` 加 `role="meter" aria-label={label} aria-valuenow={value} aria-valuemin={0} aria-valuemax={100}` | 🟢 |
| B10 | `:518-519` `Progress()` | 轨道加 `role="progressbar" aria-labelledby={labelId} aria-valuenow={Math.round(fraction*100)} aria-valuemin={0} aria-valuemax={100}`；新增 `labelId` prop，由 `:671` 处传入「训练」`<h3>` 的 id | 🟢 |
| B11 | `:435-444` `Checkpoints()` 签名 + `:446-447` / `:501` | 新增 `labelId: string` prop；`<table>` 外包 `<div className={css.tableScroll} role="group" aria-labelledby={labelId} tabIndex={0}>` | 🟢 |
| B12 | `:476-482` | 文件名 `<td>` → `<th scope="row">` | 🟢 |
| B13 | `:488` `:493` | 两个按钮加 `aria-label` 后缀文件名 | 🔴 **叫红 3 个用例** |
| B14 | `:423` | 删除 `role="group" aria-label={t('organLegacy')}`（按钮名已自解释） | 🟢 |
| B15 | `:679` | 删除 `role="group" aria-label={t('uploadTitle')}`（与按钮文案在 zh 下完全同名） | 🟢 |
| B16 | `:759` | 删除 `role="group" aria-label={t('sectionTraining')}`（与外层区块同名） | 🟢 |
| B17 | `:749`、`:817`、`:967`、`:1042` | `role="group"` 保留，`aria-label` → `aria-labelledby` 指向对应 `<h4>` / `<h3>` 的 id | 🟢 |
| B18 | `:921` | 删除 `role="group" aria-label={t('knowledgeUpload')}` | 🟢 |
| B19 | `:739-742` → `:749` 组内 | `{count} 项已选` 的 `<p>` 移进删除按钮所在的 `.actions`，置于按钮左侧 | 🟢 |
| B20 | `:749-758`、`:816-827`、`:967-973` | 三处删除按钮：armed 时在按钮前渲染 `<ul className={css.impactList} id={impactId}>`，按钮加 `aria-describedby={impactId}` | 🟢（见 §7 冲突核查） |
| B21 | `:757-769` | 运行控制行拆为两个 flex 子容器：次要簇 + `<div class={css.actionCluster}>`（停止 / 强制解锁）。`trainStop`、`trainReset` 加 `className={css.dangerAction}` | 🟢 |
| B22 | 5 处不可用文案：`:702` `:780` `:911` `:945` `:1001` | `className={css.muted}` → `className={css.unavailableNote}`，内层 `.muted` 保留 | 🟢 |
| B23 | 全部 `<Button>` 的 pending 态 | 传入 `aria-busy={pending === verb}`（仅在按钮已有 `disabled={busy...}` 的位置加，共约 12 处） | 🟢 |
| B24 | — | **不向 `locales.ts` 增 key** | — |

### 4-C · 需要同步修改的测试（`tests/panel.client.spec.tsx`）

只有 **3 个必须改**（都是 B13 引起），**2 个需冒烟复核**（B6 引起），其余 18 个零改动。详见 §5。

---

## 5. TSX 改动逐条代码 + 测试影响

### B1 · import（`:12`）

```tsx
import { useCallback, useEffect, useId, useRef, useSyncExternalStore, useState, type ChangeEvent, type ReactNode } from 'react'
```

### B2/B3 · 两个终态分支（`:174-193`）

```tsx
  if (state.state === 'loading') {
    return (
      <div className={css.page}>
        {/* L1: two section outlines previewing the cards that are about to arrive.
            Both nodes are empty and aria-hidden, so getNodeText still reads only
            the caption and the loading query stays unique.
            CSS lives in visual direction v2.1 §7 (.loadingCard / .loadingCardWide). */}
        <p className={css.loading} role="status">
          {t('loading')}
          <span className={css.loadingCard} aria-hidden="true" />
          <span className={css.loadingCardWide} aria-hidden="true" />
        </p>
      </div>
    )
  }

  if (state.state === 'error' || state.snapshot === undefined) {
    return (
      <div className={css.page}>
        {/* L2: one state panel, replacing the sibling-selector layout. */}
        <div className={css.statePanel}>
          <p className={css.errorLine} role="alert">{t('errorTitle')}</p>
          <Button
            disabled={refreshing}
            onClick={() => {
              setRefreshing(true)
              void life.refresh().catch(() => {}).finally(() => { setRefreshing(false) })
            }}
          >
            {t('retry')}
          </Button>
        </div>
      </div>
    )
  }
```

🟢 影响：`'shows the loading state first and the error state with a retry that re-reads'` 里
`getByText(en.loading)` / `getByText(en.errorTitle)` / `getByRole('button', { name: en.retry })` 全部不变。

### B4 · 删除 confirmKey 与 confirmLine（`:197`、`:215`）

```diff
   const snapshot = state.snapshot
   const shared = { t, snapshot, pending, confirming, run, life }
-  const confirmKey = confirming === null ? undefined : CONFIRM_COPY[confirming]
@@
         <ConsolidationSection {...shared} />
         <HostSection t={t} snapshot={snapshot} />

         {failureText !== null && <p className={css.errorLine} role="alert">{failureText}</p>}
-        {confirmKey !== undefined && <p className={css.confirmLine}>{t(confirmKey)}</p>}
       </div>
     </div>
   )
```

🟢 影响：**零**。逐个核过：

- `:705` `getByText(en.confirmStop)` —— 动手前**先跑一次基线**确认现状：`.confirmLine` 的 `<p>` 与 armed 后的
  `<button>` 都是这句话的直接文本宿主，按 RTL 的 `getNodeText` 规则两者会同时命中，这个断言今天很可能已经是
  "多处命中"而不是 1 处。删除 `.confirmLine` 之后只剩 `<button>` 一个命中点。
  - 若基线已是多处命中 → 本改动顺手修好它。
  - 若基线是 1 处 → 说明另有门道，请在改动前后各跑一次该用例对比，保持 1 处即可。
- `:435` / `:470` / `:524` 用 `getByRole('button', { name: confirmCopy })` —— 只匹配 `button` 角色，🟢 不受影响。
- `CONFIRM_COPY` 常量本身仍在：`:754` `:822` `:969` 是用 `confirming === verb ? t(...) : t(...)` 三元表达的，不读 `CONFIRM_COPY`。删除后若出现 unused 报警，按项目 lint 规则处理。

> 落地时若发现 `getByText(en.confirmStop)` 因删除而仍能命中 1 个 `<button>`：正常，留着。
> 若发现 `:705` 原本就返回 2 个：这是既有缺陷，本改动顺手修好。

### B6 · `<details>` 化六个 section（以 TrainingSection 为例，`:656-658` / `:831`）

```tsx
// TrainingSection, line ~533
  const headingId = useId()          // 供 Progress 的 aria-labelledby 使用
  const checkpointsId = useId()      // 供 <h4>检查点</h4> 与表格滚动容器共用

  return (
    <details className={css.section} id="life-training" open>
      <summary className={css.sectionSummary}>
        <h3 className={css.sectionTitle} id={headingId}>{t('sectionTraining')}</h3>
      </summary>
      {/* 以下所有子节点保持原有顺序，一个都不移动 */}
      <div className={css.facts}>…</div>
      …
    </details>
  )
```

`id` 分配：`life-source` / `life-readings` / `life-training` / `life-knowledge` / `life-consolidation` / `life-host`。

🟡 需复核：

- **`<summary>` 必须是 `<details>` 的第一个子节点**，否则六个区块内部的文档顺序会错。
- **`<summary>` 内不要再写任何直接文本**（`{" "}`、图标请用 `<span>` 包裹），否则
  `'renders a native reading across all six sections'` 的 6 个 `getByText(en.sectionX)` 会变成多点命中。
- `'states workbench capabilities and the three authentication outcomes'`：宿主用 `<details open>`，
  内容是留在 DOM 里的（`<details>` 隐藏子节点但**不卸载**），而 RTL 的 `getByText` 默认不校验可见性 → 预期仍绿；
  但**这条结论依赖该默认行为，请前端落地后单独跑一次这个用例确认**。

### B7/B8 · Fact 语义化（`:304-311`，10 处调用点）

```tsx
function Fact({ label, children }: { label: string; children: ReactNode }): ReactNode {
  return (
    <div className={css.fact}>
      <dt className={css.factLabel}>{label}</dt>
      <dd className={css.factValue}>{children}</dd>
    </div>
  )
}
```

```diff
-    <div className={css.facts}>
+    <dl className={css.facts}>
       <Fact label={t('sourceLabel')}>…</Fact>
       …
-    </div>
+    </dl>
```

> 10 处调用点全在文件里列出来了（§4-B B8）。**注意 `:506-511` 的 `ProgressMetric` 内部那个
> `<span className={css.factLabel}>` 不要动**——它在 `.progressMetrics` 里，不是 `<dl>` 的后代，
> 改成 `<dt>` 反而制造非法嵌套。

🟢 影响：`Fact` 里 label/value 仍是两个独立元素的直接文本，`getByText('curiosity')`、`getByText('42.5')`、
`getAllByText('energy')`（≥2 次）断言全部不变。`dl/dt/dd` 在 ARIA 映射为 term/definition，不影响任何 `getByText`。

### B9/B10 · 量表与进度条语义

```tsx
// Meter, :316-321
    <div className={css.meter}>
      <span className={css.meterLabel}>{label}</span>
      <span
        className={css.meterTrack}
        role="meter"
        aria-label={label}
        aria-valuenow={value}
        aria-valuemin={0}
        aria-valuemax={100}
      >
        <span className={css.meterFill} style={{ width: `${clampPct(value)}%` }} />
      </span>
      <span className={css.meterValue}>{value.toFixed(1)}</span>
    </div>
```

```tsx
// Progress, :515-530 —— 新增 labelId prop
function Progress({ t, progress, labelId }: { t: LifePanelProps['t']; progress: LifeProgressView; labelId: string }): ReactNode {
  return (
    <div className={css.progress}>
      <span
        className={css.progressTrack}
        role="progressbar"
        aria-labelledby={labelId}
        aria-valuenow={Math.round(progress.fraction * 100)}
        aria-valuemin={0}
        aria-valuemax={100}
      >
        <span className={css.progressFill} style={{ width: `${clampPct(progress.fraction * 100)}%` }} />
      </span>
      …
```

```diff
-      {training.progress !== undefined ? <Progress t={t} progress={training.progress} /> : <p className={css.muted}>{t('noProgress')}</p>}
+      {training.progress !== undefined
+        ? <Progress t={t} progress={training.progress} labelId={headingId} />
+        : <p className={css.muted}>{t('noProgress')}</p>}
```

`aria-labelledby` 指向「训练」`<h3>` — **零新增文案**，名字落成"训练 25%"。

🟢：`role="meter"` / `role="progressbar"` 不是任何现有断言的目标角色；可见文本节点一个没动。

### B11/B12 · 表格滚动容器 + 行标题

```tsx
// Checkpoints, :435-447 / :501
function Checkpoints({ t, checkpoints, artifacts, busy, selected, onToggleSelect, onActivate, onResume, labelId }: {
  …
  labelId: string
}): ReactNode {
  if (checkpoints.length === 0) return <p className={css.muted}>{t('checkpointsEmpty')}</p>
  return (
    <div
      className={css.tableScroll}
      role="group"
      aria-labelledby={labelId}
      tabIndex={0}
    >
      <table className={css.table}>
        …
      </table>
    </div>
  )
}
```

```diff
-              <td>
+              <th scope="row">
                 {cp.filename}
                 {artifacts?.activeId === cp.filename && <Tag tone="solid">{t('artifactsActiveBadge')}</Tag>}
                 …
-              </td>
+              </th>
```

> `labelId` 由 `TrainingSection` 传入，指向 `:772` 的 `<h4 id={checkpointsId}>{t('checkpointsTitle')}</h4>`。

🟢：`tabIndex={0}` 会新增一个 tab 停靠点，这是 WCAG 2.1.1 对滚动区的通行做法，无用例受影响。

### B13 · 行内按钮命名唯一化 —— 🔴 **唯一会叫红用例的改动**

```diff
                 <div className={css.actions}>
-                  <Button disabled={busy} onClick={() => { onResume(cp.filename) }}>{t('resumeFrom')}</Button>
+                  <Button
+                    disabled={busy}
+                    aria-label={`${t('resumeFrom')} ${cp.filename}`}
+                    onClick={() => { onResume(cp.filename) }}
+                  >
+                    {t('resumeFrom')}
+                  </Button>
                   <Button
                     disabled={busy || artifacts?.activeId === cp.filename}
+                    aria-label={`${t('activateRow')} ${cp.filename}`}
                     onClick={() => { onActivate(cp.filename) }}
                   >
                     {t('activateRow')}
                   </Button>
                 </div>
```

> 可见文案保留（`aria-label` 会取代可见文案成为可访问名），所以页面外观零变化，只是读屏与测试查询到的名字变了。

**会变红的用例及改法：**

| 用例 | 原文 | 改为 |
|---|---|---|
| `'resumes a training run from a checkpoint row with the selected datasets'`（`:545`） | `screen.getByRole('button', { name: en.resumeFrom })`（`:569`、`:579`） | `screen.getByRole('button', { name: \`${en.resumeFrom} seed_beta.pt\` })` |
| `'folds the checkpoint block away and back'`（`:479`） | `getByRole/queryByRole('button', { name: en.resumeFrom })`（`:493` `:495` `:499`） | `getByRole/queryByRole('button', { name: \`${en.resumeFrom} old_run.pt\` })` |
| `'shows the publish state on the checkpoint rows and activates with confirmation'`（`:585`） | `screen.getAllByRole('button', { name: en.activateRow })`（`:611`） | `screen.getAllByRole('button', { name: new RegExp(\`^${en.activateRow} \`) })` |

> 若不想动测试，唯一的选择是保留同名的 `resumeFrom`/`activateRow` —— 那意味着 6 行检查点出现 6 个同名按钮，
> 读屏列表里完全不可分辨（WCAG 2.4.6 / 1.3.1）。**本规范选择同步修改这 3 个用例，因为这才是修缺陷而不是让可用性去迁就缺陷。**

### B14–B18 · 消除重复的可访问名（全部 🟢）

规则：**能删则删；保留的一律 `aria-labelledby` 指向同级标题。**

```tsx
// :423 —— 删除 role/aria-label，按钮名自解释（启动调度/停止调度/喂食/睡眠/玩耍）
-      <div className={css.actions} role="group" aria-label={t('organLegacy')}>
+      <div className={css.actions}>

// :679 —— 删除；zh 下 t('uploadTitle') 与 t('uploadPick') 都是「上传训练文件」，会与按钮同名
-        <div className={css.actions} role="group" aria-label={t('uploadTitle')}>
+        <div className={css.actions}>

// :759 —— 删除；t('sectionTraining') 与外层 <details> 同名
-      <div className={css.actions} role="group" aria-label={t('sectionTraining')}>
+      <div className={css.actions}>

// :921 —— 同上，删除
-              <div className={css.actions} role="group" aria-label={t('knowledgeUpload')}>
+              <div className={css.actions}>

// :749 / :817 / :967 / :1042 —— 保留 group，改指向同级标题的 id
-      <div className={css.actions} role="group" aria-label={t('datasetsTitle')}>
+      <div className={css.actions} role="group" aria-labelledby={datasetsId}>
-            <div className={css.actions} role="group" aria-label={t('checkpointsTitle')}>
+            <div className={css.actions} role="group" aria-labelledby={checkpointsId}>
-            <div className={css.actions} role="group" aria-label={t('knowledgeFilesTitle')}>
+            <div className={css.actions} role="group" aria-labelledby={knowledgeFilesId}>
-            <div className={css.actions} role="group" aria-label={t('sectionConsolidation')}>
+            <div className={css.actions} role="group" aria-labelledby={consolidationHeadingId}>
```

配套给四处标题加 `id`：`:677` `<h4 datasetsTitle>`、`:772` `<h4 checkpointsTitle>`、`:943` `<h4 knowledgeFilesTitle>`、
以及各 section 的 `<h3>`（B6 已生成 `headingId`，直接复用）。

🟢：spec 中没有任何 `getByRole('group'…)` 断言。

### B19/B20 · 选中计数前置 + 影响清单（以训练区为例，替换 `:749-758`）

```tsx
      {/* 「{count} 项已选」从 roster 下方的灰色 <p> 移到这里，紧贴操作（B19 的原行 :739-746 移除） */}
      <div className={css.actions} role="group" aria-labelledby={datasetsId}>
        {confirming === 'deleteDataset' && (
          <ul className={css.impactList} id={datasetImpactId}>
            {[...selected].map(path => <li key={path}>{path}</li>)}
          </ul>
        )}
        <Button
          className={css.dangerAction}
          disabled={busy || active || selected.size === 0}
          aria-describedby={confirming === 'deleteDataset' ? datasetImpactId : undefined}
          onClick={() => { run('deleteDataset', deleteDatasets) }}
        >
          {confirming === 'deleteDataset'
            ? t('confirmDeleteDatasets', { count: String(selected.size) })
            : t('deletePickedDatasets')}
        </Button>
      </div>
```

同样的 `<div className={css.actions}>` 包裹适用于另外两处删除按钮（`:816-827` 检查点、`:967-973` 知识文件），
三处都记得给 `<Button>` 加 `className={css.dangerAction}`。

### B21 · 运行控制行分组（替换 `:759-770`）

```tsx
      <div className={css.actions}>
        <Button
          variant="primary"
          disabled={busy || active}
          onClick={() => { run('trainStart', () => life.trainStart(selected.size > 0 ? { datasets: [...selected] } : {})) }}
        >
          {t('trainStart')}
        </Button>
        <Button disabled={busy || !active || training.stopRequested} onClick={() => { run('trainPause', () => life.trainPause()) }}>{t('trainPause')}</Button>
        <Button disabled={busy || !active || training.stopRequested} onClick={() => { run('trainResume', () => life.trainResume()) }}>{t('trainResume')}</Button>
        {/* 危险簇：宽屏贴右缘、窄屏自然换行，视觉上用竖线与前面隔开 */}
        <div className={css.actionCluster}>
          <Button className={css.dangerAction} disabled={busy || !active} onClick={() => { run('trainStop', () => life.trainStop()) }}>{t('trainStop')}</Button>
          <Button className={css.dangerAction} disabled={busy || !active} onClick={() => { run('trainReset', () => life.trainReset()) }}>{t('trainReset')}</Button>
        </div>
      </div>
```

> **注意**：`variant="primary"` 是 primitives `Button` 已有 prop（视觉方向 §8.1 确认），
> 但 `:759` 这一行原本没有 `variant`。加上后不影响 `getByRole('button', { name: en.trainStart })`，🟢。
> `:423` 的 `lifeStop`、`:1042` 的 `runConsolidate` 同理处理 `className`/`variant`。

### B22 · 5 处「不可用」文案升级为卡片（Q4-L3）

```diff
-        ? <p className={css.muted}>{t('datasetsUnavailable')}</p>
+        ? <p className={css.unavailableNote}>{t('datasetsUnavailable')}</p>
```
同样处理 `:780 artifactsUnavailable`、`:911 knowledgeUnavailable`、`:945 knowledgeFilesUnavailable`、`:1001 consolidationUnavailable`。

🟢：只有 className 变，文本一字未改。唯一需要确认的是 `'says honestly when the publish surface did not answer'`（`:635`）
里 `expect(screen.queryByText(en.activeCheckpointLabel)).toBeNull()` —— 逻辑分支不变，仍为 null。🟢

### 冲突核查（B20 引入的新 DOM 与既有断言）

| 断言 | 会不会因影响清单而变多 | 结论 |
|---|---|---|
| `getAllByText('consolidated/night-1.jsonl').length).toBeGreaterThanOrEqual(1)`（`:251`） | 该用例未进入 armed 状态 | 🟢 |
| `getAllByText('seed_beta.pt')).toHaveLength(2)`（`:604`） | 该用例 armed 的是 `activateCheckpoint` 而非 delete，清单不渲染 | 🟢 |
| `getByText('ckpt-000005.pt')`（`:700`） | 未 armed | 🟢 |
| `getByText(/2 selected/)`（`:431` 前后） | 移了位置但文本未变，仍唯一 | 🟢 |
| `getByRole('checkbox', { name: 'loose.txt' })`（`:522`） | 清单用 `<li>` 不是 checkbox；且arming 发生在 click 之后 | 🟢 |

---

## 6. 对视觉方向的异议（已由 visual 全部采纳）

**状态更新**：V-1 至 V-6 全部采纳，已并入视觉方向 **v2.1**（§7 为唯一权威 CSS 版本、§9 为变更登记）。
下表保留原始论证以便回溯；**具体以 v2.1 §7 为准，不要再按本节文字改 CSS。**

| 编号 | 结论 | 在 v2.1 的处理 |
|---|---|---|
| V-1 | 骨架形态不成立 | 采纳。改为两张卡片轮廓 56 / 88px + 内部占位条；文案居中放在轮廓下方。**过程中发现一处无效 CSS，见下方 V-1 补记** |
| V-2 | 兄弟选择器脆骨头 | 采纳（必须改）。删两条兄弟选择器，新增 `.statePanel`；运行期失败带走 `.content > .errorLine` 底部停靠 |
| V-3 | `.table th` 污染行标题 | 采纳（必须改）。拆成 `thead th` / `tbody th`，并显式 `font-weight: 500` 挡 UA 默认值 |
| V-4 | facts 单列断点太晚 | 采纳。`420px` → `560px`；420px 档保留给 meter，另加 240px 保险档 |
| V-5 | 动作行缺分组容器 | 采纳。新增 `.actionCluster`，分隔符改用 `border-left`（不占内容宽、不参与 flex 收缩），容器 ≤560px 时移除分隔与 `margin-left:auto` |
| V-6 | sticky 纪律提醒 | 采纳为纪律，写入 §2 与 §6 |

V-2 还被升格成了通用纪律（**不用相邻/兄弟选择器表达布局**），这比逐条修复更彻底。

### V-1 补记 · 修正过程中出现的一处无效 CSS（已修）

v2 第一版把占位条写成 `.loading::before::before` / `.loading::after::before` / `.loading::after::after`。
CSS 规范限定**每个宿主元素只能生成一个 `::before` 和一个 `::after`，伪元素不能嵌套**，
这三条会被解析器静默丢弃——不报错、不报警，结果是两张卡片变成**空框**，一条占位条都不出现。

修法已在 v2.1 §8.4 落地：两张轮廓改为两个空的 `aria-hidden` `<span>`（`.loadingCard` / `.loadingCardWide`），
占位条走各自的一级 `::before/::after`，从而保住 `border-radius: 999px` 的胶囊圆角
（`background-image` 多层渐变虽然能做到零 DOM，但画不出圆角，故未采用）。
对应 TSX 改动已同步进本文件 **B2**。

**测试影响零**：`<span>` 无文本，RTL 的 `getNodeText` 只拼直接文本子节点，`getByText(en.loading)` 仍唯一命中那个 `<p>`。

**验收哨兵**（抄自 v2.1 §10）：56px 卡内必须有 1 条、88px 卡内必须有 2 条占位条；
如果只剩两个空框，说明有人把二级伪元素写回去了。

### V-1 原文（保留论证，勿据此改代码 —— 请看上方表格与 v2.1 §7）

视觉方案初版把两条 10px 高的圆角线用 `::before/::after` 放在 `.loading` 的 top 18/38px，
而真实文案被 `padding-top: 58px` 推到下方。结果是"两根细条 + 一行字"并排，
**看起来像已经加载好的一个标题加一段摘要，而不是尚未到达的内容**——骨架屏的全部价值在于"预告即将出现的结构"，
这两条细线预告的结构与被替换的六个卡片毫无相似度。

当时的建议：改成**两张不同高度的卡片轮廓**（约 56px 与 88px、`border-radius: 12px`、
`0.5px border-l2`、`background: bg-layer-1`，内部再表示标题与正文位置），文案居中放在轮廓下方。
> 注意：这条建议里"保留伪元素（零 DOM 成本）"的部分**已被 §6 开头的 V-1 补记推翻**——
> 占位条无法用嵌套伪元素实现，最终采纳了 v2.1 §8.4 的两个空 `<span>` 方案。

### V-2 · `.page > .errorLine` + `.page > .errorLine + button` 是脆骨头选择器（必须改）

视觉方案第 587-596 行用相邻兄弟选择器把错误态居中：

```css
.page > .errorLine { width: min(100%, 960px); margin: 24px auto 12px; }
.page > .errorLine + button { display: flex; margin: 0 auto; }
```

这把布局**绑死在"错误带必须紧挨着按钮的前一个子节点、且二者都是 `.page` 直接子节点"**上。
本规范 B3 就要给这个分支套一层 `.statePanel` 容器（为了让它拥有和加载态一致的卡片形态），
套完这两条规则立刻全塌。而且它与本文件另一条—— `.content > .errorLine` 的 sticky 停靠 —— 语义打架，
同一个 class 在两处被两套位置规则描述，后续任何改动都会踩雷。

建议：**换成 `.statePanel` 容器定位**，把居中职责交给容器的 flex，删掉这两条兄弟选择器。（CSS-A2、TSX-B3）

### V-3 · `.table th` 会污染新增的行标题（必须改）

视觉方案第 620-627 行：

```css
.table th { border-bottom-color: …; background: var(--dsw-alias-bg-layer-1); color: …secondary; font-size: 11px; font-weight: 600; }
```

本规范 Q6-7 要求把每行第 2 格从 `<td>` 改成 `<th scope="row">`（这是"选择 seed_beta.pt"这类复选框名称能关联到行的前提）。
不收窄的话，行标题会变成 11px tertiary 字 + `bg-layer-1` 实底，在 hover 时不跟随行底色，视觉上像一行"被选中的表头"。

建议：`.table thead th { … }` + 新增 `.table tbody th { font-size:12px; font-weight:500; color: label-primary; background: transparent }`。（CSS-A5/A6）

### V-4 · facts 单列断点 420px 太晚（建议改）

见 Q5.2 推导：`container-type: inline-size` 查询的是 content box，
面板宽 472px 时容器宽就已经到了 368px = 两列门槛。也就是说**在最常见的窄面板宽度下，
用户拿到的是两个只有 180px 宽的格子、里面塞着等宽字体的长路径**（如 `simple_zh/dialogue_extended_clean.jsonl`），
换行到 3 行，扫读成本比单列更高。

建议：`@container (max-width: 420px)` → **`@container (max-width: 560px)`**，
让两列只在面板宽 ≥ 664px 时才出现。（CSS-A11）

### V-5 · 动作行缺少分组容器，`margin-left:auto` 无处安放（新增请求）

视觉方案的 `.actions { display:flex; flex-wrap:wrap; align-items:center; gap:8px }` 是一维的，
无法表达 Q3.1 要求的"次要簇 | 危险簇、危险簇贴右缘"。本规范为此新增了 `.actionCluster`（CSS-A14），
只有一条 `margin-left:auto` 和一个 1px 竖线伪元素，不引入任何新的颜色或形状 token，请一并纳入。

### V-6 · 不是异议，是提醒

`.section { container-type: inline-size }` 隐含 `contain: layout inline-size`，会为每个区块创建新的 BFC。
这不影响 `.page` 的纵向滚动、也不影响 `.tableScroll` 的横向滚动，但如果后续有人想给某个 section 内的元素做
`position: sticky`，会以 section 而不是 `.page` 为滚动上下文 —— **请勿在 section 内再使用 sticky**，
所有吸顶/吸底都只能在 `.page` / `.content` 这一层做（本规范的 `.content > .errorLine` 停靠条正遵循此约束）。

---

## 7. 交付自查

- [x] 七个问题全部给出了单一推荐方案，没有让用户二选一
- [x] 落地清单按 CSS / TSX / 测试三类分开
- [x] 每条 TSX 改动都带位置（组件名 + 行号）与代码片段
- [x] 明确列出会叫红的 3 个用例名称 + 改法
- [x] 未修改本任务仓库（`packages/client/ui-life`）任何文件，只写了本 md
- [x] 未运行任何构建/测试命令
- [x] 未推翻视觉方向的 token 决策；异议单独成节
- [x] 零新增 locale key

## 附录 A · 可选新增文案（本轮**不需要**）

| key | 若将来需要 | 本轮的规避方式 |
|---|---|---|
| `sectionFold` / `sectionUnfold` | 区块折叠按钮的无障碍名 | 用 `<details>/<summary>`，`<summary>` 的可访问名直接来自 `<h3>` 的既有文案 |
| `trainingProgressLabel` | 进度条的 aria-label | 用 `aria-labelledby` 指向区块 `<h3>`「训练」 |
| `impactLeadIn` | 影响清单的前导语（如「将删除：」） | 清单不加前缀词，直接列文件名，靠位置与周围上下文表达语义 |

## 附录 B · 验收清单（给 Phase 3 前端）

**功能不回归（23 个用例）**

- [ ] 20 个用例保持全绿
- [ ] 3 个用例按 §5-B13 的表格改完后全绿
- [ ] `'states workbench capabilities and the three authentication outcomes'` 单独跑一次（B6 引入 `<details>`）
- [ ] `'renders a native reading across all six sections'` 单独跑一次（确认 `<summary>` 无直接文本）

**体验验收（人工）**

- [ ] 面板宽 360 / 420 / 520 / 960 / 1400px 五档：页面无横向滚动；检查点表格横向滚动只在表格内
- [ ] facts 在面板宽 < 664px 时是单列
- [ ] 六个区块标题行可点击折叠/展开；刷新快照后用户的折叠状态不丢
- [ ] Tab 遍全页，焦点永远不被 sticky header 盖住；焦点环是 2px Seed 绿且可见
- [ ] 训练区「停止 / 强制解锁」在操作行右端并与前面有分隔；两者是 danger 样式
- [ ] 按下删除类按钮 → armed → 按钮上方出现文件名清单 → 第二次点击执行
- [ ] 任意控制动作失败 → 错误带停靠在滚动容器底部（滚到页面任意位置都可见），文案是稳定错误码不是 RPC 原文
- [ ] 五个"不可用"文案现在是统一的警告浅底卡片，不再与"尚无数据"的灰字混淆
- [ ] 训练区的四个子组都有 `<h4>` 小标题，且每个动作行属于它作用的对象
- [ ] 读屏逐区块朗读：没有听到重复的区块名；检查点表格每行能读出"行 X / 文件名"
- [ ] **加载态哨兵**：56px 卡内有 1 条、88px 卡内有 2 条占位条；只剩两个空框说明有人写回了二级伪元素（见 §6 V-1 补记）

## 附录 C · 两份文档的分工（避免版本漂移）

本轮产出两份文档，**刻意不做内容复制**，请按用途各取所需：

| 文档 | 版本 | 负责什么 | 什么时候读 |
|---|---|---|---|
| `UI_LIFE_VISUAL_DIRECTION_20261003.md` | **v2.1** | CSS 的唯一权威版本（§7 整份替换片段、§8.4 唯一的骨架 DOM、§9 变更登记、§10 验收清单） | 写 `LifePanel.module.css` 时，**只**照 §7 |
| `UI_LIFE_UX_SPEC_20261003.md` | 本文件 | TSX 的 DOM 结构与行号（§4-B / §5）、七个问题的结论与理由（§1）、以及对测试的影响判定（§5） | 写 `LifePanel.tsx` 与改测试时 |

三条纪律：

1. **CSS 以 v2.1 §7 为准。** 本文件 §4-A 那张表已经改成"核对清单"，只说明每条为什么存在、以及哪些还没并入，**不要照它另写一份 CSS**。
2. **DOM 以本文件为准。** v2.1 §8.4 只声明了加载骨架那两个 `<span>`，其余 TSX 改动都在本文件 §5。
3. **视觉 v2.1 反过来引用了本文件的 §6 与 §4-A**（见它 §7 开头的版本说明与 §9 变更登记）。两份互相引用而非互相复制，任何一方改动请通知另一方同步——那处无效 CSS 正是交叉复核读出来的，不是各自审自己能发现的。
- [ ] 读屏逐区块朗读：没有听到重复的区块名；检查点表格每行能读出"行 X / 文件名"
