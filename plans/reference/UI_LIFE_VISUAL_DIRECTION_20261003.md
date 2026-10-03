# Seed「生命」页视觉方向（2026-10-03）

> 范围：仅定义视觉与 CSS 落地方向；不改业务逻辑、文案语义、数据流或 DOM 结构。本规范基于 `LifePanel.tsx`、`LifePanel.module.css`、`locales.ts` 全量阅读，以及全仓主题 token 定义检索。目标运行环境为 Electron / Chromium，默认 IDE 主题为 light，同时覆盖 dark。

## 1. 视觉方向陈述

将页面从“连续文本流”改成**克制的生命运行台**：保留从「来源」到「宿主」的纵向时间/因果顺序，以 6 个同层级、低浮雕的平面区块承载内容；区块内部再用紧凑的信息格、发丝分隔和状态带组织扫描路径。页头固定在滚动容器顶部，Seed 绿只用于“活跃/进行中/焦点”，成功、警告、失败严格使用语义状态色，避免整页泛绿。整体不是营销型大卡片，也不做霓虹科幻；浅色以白底、黑灰文字和 0.5px 边界为主，深色由同一 alias token 自动切层。

## 2. 设计原则

1. **一层卡片，内层平面分组**：六个 `.section` 是唯一主卡片层；内部 fact 使用低对比底色而非再加阴影，避免“卡片套卡片”。
2. **固定上下文，压缩跳读成本**：`.header` sticky；主区块间距 16px，区块内间距 16px，子组间距 8–12px。标题左侧 3px Seed 色短标记建立稳定锚点。
3. **状态色只表达状态**：普通生命 meter=success，训练 progress=business，warning/error/success 消息用对应浅底+边界；正文始终用主/次文字色，避免浅色主题中低对比彩色小字。
4. **数据优先于装饰**：数值统一 `tabular-nums`；路径使用代码字体；表头与辅助文案至少使用 `label-secondary`，不再用 light 下偏淡的 tertiary 承担 12px 正文。
5. **窄面板不牺牲可读性**：内容上限从 760px 放宽到 960px；fact 单格最小 180px、标签和值上下排，但**两列只在容器宽 ≥ 560px（面板宽 ≥ 664px）时才出现**——低于该宽度宁可一行一个 fact，也不要两个都在换行的窄格。表格由 `.tableScroll` 局部横向滚动，页面本身不横向溢出。

### 明确避免

- 不使用仓库不存在的 `--dsw-alias-fill-accent`、`--dsw-alias-text-tertiary`；全仓精确检索均为 **No matches found**。
- 不用硬编码主题分支、图片、图标字体、渐变霓虹、重阴影或彩色大面积底。
- 不依据 meter 数值高低猜测 warning/error：`needs` / `drives` 的高低不天然等同好坏，当前数据没有 tone 语义。
- 不用 `nth-child` 或中文按钮文本选择器识别危险操作；这会被中英文切换或按钮增删破坏。
- 不用相邻/兄弟选择器表达布局（`.page > .errorLine + button` 之类）：分支 DOM 一变就塌，位置职责交给容器。
- 不给 section 同时使用实线 border 与 elevation shadow；当前方案选择 0.5px 平面边界，不叠 elevation。
- 不在 `.section` 内部使用 `position: sticky`（`container-type: inline-size` 会创建 BFC，滚动上下文变成 section 自身）。

## 3. 色板（解析值 + CSS 变量）

下列 HSL/HEX 仅用于设计审阅；生产 CSS 一律消费 alias token，并带 fallback。

| 角色 | Light | Dark | CSS token | 使用边界 |
|---|---:|---:|---|---|
| 页面底 | `#FFFFFF` / HSL 0 0% 100% | `#151517` / HSL 240 5% 9% | `--dsw-alias-bg-base` | 页面、sticky header |
| 区块底 | `#FFFFFF` | `#232324` / HSL 240 1% 14% | `--dsw-alias-bg-layer-1` | 六个主区块 |
| 主文字 | `#0F1115` / HSL 220 17% 7% | `#F9FAFB` / HSL 210 20% 98% | `--dsw-alias-label-primary` | 标题、数值、消息正文 |
| 次文字 | `#61666B` / HSL 210 5% 40% | `#CFD3D6` / HSL 206 7% 83% | `--dsw-alias-label-secondary` | 辅助文案、标签、表头 |
| 弱文字（审计保留） | `#81858C` | `#ADB2B8` | `--dsw-alias-label-tertiary` | 仅非关键装饰；不承担 12px 正文 |
| Seed / 进行中 | `#569E58` / HSL 121 30% 48% | `#7CBB67` / HSL 105 38% 57% | `--dsw-alias-state-business-primary` | section 标记、训练进度、focus ring |
| 正常 / 成功 | `#22C55E` / HSL 142 71% 45% | `#22C55E` | `--dsw-alias-state-success-primary` | 普通 meter、成功边界 |
| 成功浅底 | `#E6FAED` | `#233C2C` | `--dsw-alias-state-success-tertiary` | 成功消息背景 |
| 警告 | `#F59E0B` / HSL 38 92% 50% | `#F59E0B` | `--dsw-alias-state-warn-primary` | 警告边界/标识，不作小字正文色 |
| 警告浅底 | `#FEF5E7` | `#27241F` | `--dsw-alias-state-warn-tertiary` | 二次确认、不可用来源背景 |
| 失败 | `#EC1313` / HSL 0 85% 50% | `#F25A5A` / HSL 0 85% 65% | `--dsw-alias-state-error-primary` | 错误边界、危险按钮文字 |
| 微弱交互底 | `rgba(38,49,72,.06)` | `rgba(255,255,255,.08)` | `--dsw-alias-interactive-bg-hover` | fact、hover、进度块 |
| 轨道/按下底 | `rgba(38,49,72,.10)` | `rgba(255,255,255,.14)` | `--dsw-alias-interactive-bg-active` | meter/progress track |
| 危险 hover 底 | `rgba(236,19,19,.05)` | `rgba(242,90,90,.15)` | `--dsw-alias-interactive-bg-hover-danger` | 危险按钮、错误消息底 |
| 卡片边界 | `rgba(0,0,0,.10)` | `rgba(255,255,255,.12)` | `--dsw-alias-border-l2` | section / dataset group |
| 强分隔 | `rgba(0,0,0,.16)` | `rgba(255,255,255,.20)` | `--dsw-alias-border-l4` | 表头/关键行分隔 |

### 对比度策略

- 文字主体只用 `label-primary` / `label-secondary`；语义浅底上的正文仍用 `label-primary`，状态色只做底、边界或大于 3:1 的 UI 图形。
- `label-tertiary` 在 light 的 12–13px 文本上偏弱，因此现有 `.subtitle`、`.factLabel`、`.muted`、`.datasetSize`、`.table th` 建议统一提升为 `label-secondary`。
- `business/success/warn` 的亮色值不直接用于小号正文；这样保证 light 为系统默认主题时仍可读。

## 4. Token 取用与全仓校对

### 当前文件已用 token（逐项核实）

| 用途 | token | 权威定义位置（light / dark） | 结论 |
|---|---|---|---|
| 弱文字 | `--dsw-alias-label-tertiary` | `packages/client/ui-theme/src/styles/design-platform.css:209 / 309` | 存在；但不建议继续承担小号正文 |
| 次文字 | `--dsw-alias-label-secondary` | `packages/client/ui-theme/src/styles/design-platform.css:208 / 308` | 存在，继续使用 |
| 业务/Seed 状态 | `--dsw-alias-state-business-primary` | `packages/client/ui-theme/src/styles/design-platform.css:223 / 323` | 存在，限定进度/焦点/强调 |
| 错误状态 | `--dsw-alias-state-error-primary` | `packages/client/ui-theme/src/styles/design-platform.css:225 / 325` | 存在 |
| 警告状态 | `--dsw-alias-state-warn-primary` | `packages/client/ui-theme/src/styles/design-platform.css:232 / 332` | 存在 |
| 交互按下/轨道 | `--dsw-alias-interactive-bg-active` | `packages/client/ui-theme/src/styles/design-platform.css:196 / 296` | 存在 |
| 强边界 | `--dsw-alias-border-l4` | `packages/client/ui-theme/src/styles/design-platform.css:176 / 276` | 存在 |

### 本轮新增取用 token

| 用途 | token | 权威定义位置（light / dark） | 结论 |
|---|---|---|---|
| 页面背景 | `--dsw-alias-bg-base` | `packages/client/ui-theme/src/styles/design-platform.css:155 / 255` | 存在 |
| 卡片背景 | `--dsw-alias-bg-layer-1` | `packages/client/ui-theme/src/styles/design-platform.css:158 / 258` | 存在 |
| 骨架底 | `--dsw-alias-bg-skeleton` | `packages/client/ui-theme/src/styles/design-platform.css:169 / 269` | 存在 |
| 主文字 | `--dsw-alias-label-primary` | `packages/client/ui-theme/src/styles/design-platform.css:207 / 307` | 存在 |
| hover / 弱底 | `--dsw-alias-interactive-bg-hover` | `packages/client/ui-theme/src/styles/design-platform.css:200 / 300` | 存在；`ui-plugin-manager` 有局部重绑，不影响本页继承 |
| 危险 hover / 错误浅底 | `--dsw-alias-interactive-bg-hover-danger` | `packages/client/ui-theme/src/styles/design-platform.css:198 / 298` | 存在 |
| 普通边界 | `--dsw-alias-border-l2` | `packages/client/ui-theme/src/styles/design-platform.css:174 / 274` | 存在 |
| 正常/成功 | `--dsw-alias-state-success-primary` | `packages/client/ui-theme/src/styles/design-platform.css:228 / 328` | 存在 |
| 成功浅底 | `--dsw-alias-state-success-tertiary` | `packages/client/ui-theme/src/styles/design-platform.css:230 / 330` | 存在 |
| 警告浅底 | `--dsw-alias-state-warn-tertiary` | `packages/client/ui-theme/src/styles/design-platform.css:234 / 334` | 存在 |
| 空闲状态（状态表预留） | `--dsw-alias-state-idle-primary` | `packages/client/ui-theme/src/styles/design-platform.css:227 / 327` | 存在；当前不强行映射 meter |
| 字体族 | `--dsw-font-family` | `packages/client/ui-theme/src/styles/base.css:7-8` | 存在 |
| 路径/代码字体族 | `--ds-font-family-code` | `packages/client/ui-theme/src/styles/base.css:9-10` | 存在 |
| 动效时长/曲线 | `--ds-transition-duration` / `--ds-transition-duration-slow` / `--ds-ease-in-out` | `packages/client/ui-theme/src/styles/base.css:11-14` | 存在 |

补充：全仓精确检索 `--dsw-alias-fill-accent:`、`--dsw-alias-text-tertiary:` 均无定义，不得使用。主题权威文件是 `packages/client/ui-theme/src/styles/design-platform.css`；`README.zh.md:101-105` 也明确“token 样式表是颜色值的唯一权威来源”。

## 5. 字号阶梯与排印

| 层级 | 字号 / 行高 | 字重 | 应用 |
|---|---:|---:|---|
| 页面标题 | 22 / 30px | 600 | `.title`，唯一一级视觉标题 |
| 区块标题 | 16 / 24px | 600 | `.sectionTitle`（挂在 `<summary>` 内） |
| 子组标题 | 13 / 20px | 600 | `.organTitle` |
| 正文/值 | 13 / 20px | 400–500 | `.factValue`、`.muted`、列表 |
| 标签/表格/度量 | 12 / 18px | 400–600 | `.factLabel`、`.meter*`、`.table` |
| 表头 | 11 / 16px | 600 | `.table thead th`（行标题 `.table tbody th` 走 12/18 500） |

- 中英文均沿用 `--dsw-font-family` 系统字体栈；路径与文件名用 `--ds-font-family-code`。
- 数值、步数、容量、时间统一 `font-variant-numeric: tabular-nums`。
- 不引入新字体，不用全大写，不以字距拉开中文标题。

## 6. 关键组件视觉规范

### 区块

- `.section`（`<details>`）：`20px` 内边距、`12px` 圆角、0.5px `border-l2`、`bg-layer-1`；卡片间 `16px`。
- section 内：标题→内容 `16px`，同组行 `8px`；第二个 organ 用顶部发丝线分开。
- 折叠态的节奏补偿：`<details>` 折叠后 `gap: 16px` 不再作用于首个子节点，标题会比其他状态贴紧 4px，因此 `details:not([open]) > .sectionSummary` 补 `padding-bottom: 4px`，让展开/折叠两态标题到内容（或到下沿）的距离一致。
- hover 不抬升整张 section，避免长页每块都“可点击”的错误示能。

### Fact

- 自动列宽 `minmax(min(100%, 180px), 1fr)`，比当前 220px 更晚掉到单列。
- 标签和值纵向排布，单格 `10px 12px`，使用微弱交互底；窄屏成为独立可扫读条目，而不是连续的“标签 值”长串。

### Meter / Progress

- Meter：8px 轨道、胶囊圆角；普通生命读数使用 `state-success-primary`。
- Progress：10px 轨道、`state-business-primary`，明确表示“过程进行中”，与稳定读数分开。
- warning/error 不根据数值阈值自动推导；只能在运行时提供语义后再加状态 class。

### 表格

- 采用 UX 拍板的 `.tableScroll` wrapper：`.table` 保持 `display: table` + `border-collapse: collapse`，`min-width: 720px`，横向滚动只在 wrapper 内，页面本身不横向溢出（初版写在 `.table` 上的 `display:block; overflow-x:auto` 已撤销）。
- 表头样式收窄到 `thead th`；`tbody th`（行标题，文件名）按数据行处理：12/18、500、`label-primary`、`background: transparent`，hover 跟随行底色。
- 行高约 36px，hover 使用 `interactive-bg-hover`；第 2/5/6 列给 `min-width` 防止挤扁。

### 按钮

- 普通次级动作沿用 primitives 的 ghost；每组最多一个主动作可用现有 `variant="primary"`。
- 停止调度、停止训练、强制解锁、删除所选数据集/检查点/知识文件使用新增 `.dangerAction`：默认错误色文字+1px 状态边界，hover 才出现危险浅底。
- 禁用沿用 primitive 的 `opacity: .4`；focus-visible 统一 2px Seed ring。

### 消息、加载、空态

- error / confirm / success 都是 10×12px 的状态带，正文使用 `label-primary`，状态由背景与 1px 边界表达。
- 流中断态用新增 `.statePanel`（`960px` 上限 + flex 列 + 左对齐）承载错误带与重试按钮，居中职责归容器；不使用相邻兄弟选择器。
- 运行期失败带用 `.content > .errorLine`（`sticky; bottom: 0`）停靠在滚动容器底部。禁止在 `.section` 内部使用 sticky——`container-type: inline-size` 隐含 `contain: layout`，会把 sticky 的滚动上下文变成 section 自身。
- 加载态是两张区块轮廓（56px / 88px、`border-radius: 12px`、0.5px `border-l2`、`bg-layer-1`）+ 说明文案，预告即将到达的卡片结构，而不是两根孤立细条。轮廓用两个空 `aria-hidden` `<span>` 承载（见 §8.4），占位条走它们的 `::before/::after`。
- 不可用注记用 `.unavailableNote`（警告浅底 + 1px 边界 + 主文本色），与"尚无数据"的 `.muted` 明确区分；空态不做插图或占位卡片。

## 7. 可直接粘贴的 CSS

> **v2.2（2026-10-03 夜）**：本节已并入体验规范 `UI_LIFE_UX_SPEC_20261003.md` §6 的 V-1～V-5 与 V-6 提醒、UX §4-A 的 A1/A7/A8/A9/A10/A12/A13，并修正了 v2 里三处无效 CSS（二级伪元素）；v2.2 另覆盖 fe 第一轮 TSX 语义改造挂上的 7 个新类名与 `<details>/<summary>` 化后新暴露的两条规则。本节是落地的唯一权威版本，请以本节为准，不要混用初版片段。变更登记见 §9。

以下为 **`LifePanel.module.css` 整体替换片段**（v2.2）；保留全部现有 class 名。`.dangerAction`、`.actionCluster`、`.statePanel`、`.unavailableNote`、`.impactList`、`.sectionSummary`、`.tableScroll` 是新增规则（fe 第一轮已把这些类名挂到 TSX 上）；`.loadingCard` / `loadingCardWide` 需要两个空的 `aria-hidden` `<span>`（见 §8.4，本规范唯一新增 DOM）；`.table` 的 `display:block/overflow-x:auto` 已按 UX 结论撤销并改用 wrapper。

```css
/* Life: a quiet runtime console. All colours resolve through verified theme
   aliases and include fallbacks for first-paint resilience. */

.page,
.page * {
  box-sizing: border-box;
}

.page {
  height: 100%;
  overflow-y: auto;
  overflow-x: hidden;
  overscroll-behavior-y: contain;
  scrollbar-gutter: stable;
  /* Keep tab stops and #life-* deep links clear of the sticky header. */
  scroll-padding-top: 96px;
  padding: 0 32px 48px;
  background: var(--dsw-alias-bg-base, #ffffff);
  color: var(--dsw-alias-label-primary, #0f1115);
  font-family: var(--dsw-font-family, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif);
}

.content {
  display: flex;
  width: min(100%, 960px);
  margin: 0 auto;
  flex-direction: column;
  gap: 16px;
}

.header {
  position: sticky;
  top: 0;
  z-index: 5;
  display: flex;
  flex-direction: column;
  gap: 2px;
  padding: 20px 4px 14px;
  border-bottom: 0.5px solid var(--dsw-alias-border-l2, rgba(0, 0, 0, 0.1));
  background: var(--dsw-alias-bg-base, #ffffff);
}

.title {
  margin: 0;
  color: var(--dsw-alias-label-primary, #0f1115);
  font-size: 22px;
  font-weight: 600;
  line-height: 30px;
}

.subtitle {
  max-width: 68ch;
  margin: 0;
  color: var(--dsw-alias-label-secondary, #61666b);
  font-size: 13px;
  line-height: 20px;
}

.section {
  container-type: inline-size;
  display: flex;
  flex-direction: column;
  gap: 16px;
  padding: 20px;
  scroll-margin-top: 96px;
  border: 0.5px solid var(--dsw-alias-border-l2, rgba(0, 0, 0, 0.1));
  border-radius: 12px;
  background: var(--dsw-alias-bg-layer-1, #ffffff);
}

.sectionTitle {
  display: flex;
  align-items: center;
  gap: 8px;
  margin: 0;
  color: var(--dsw-alias-label-primary, #0f1115);
  font-size: 16px;
  font-weight: 600;
  line-height: 24px;
}

.sectionTitle::before {
  width: 3px;
  height: 16px;
  flex: 0 0 3px;
  border-radius: 999px;
  corner-shape: round;
  background: var(--dsw-alias-state-business-primary, #569e58);
  content: "";
}

.organTitle {
  margin: 0;
  color: var(--dsw-alias-label-secondary, #61666b);
  font-size: 13px;
  font-weight: 600;
  line-height: 20px;
}

.organ {
  display: flex;
  flex-direction: column;
  gap: 10px;
}

.organ + .organ {
  padding-top: 16px;
  border-top: 0.5px solid var(--dsw-alias-border-l2, rgba(0, 0, 0, 0.1));
}

.facts {
  display: grid;
  margin: 0;
  grid-template-columns: repeat(auto-fit, minmax(min(100%, 180px), 1fr));
  gap: 8px;
}

.fact {
  display: grid;
  min-width: 0;
  grid-template-rows: auto minmax(20px, auto);
  align-content: start;
  gap: 2px;
  padding: 10px 12px;
  border-radius: 8px;
  background: var(--dsw-alias-interactive-bg-hover, rgba(38, 49, 72, 0.06));
}

.factLabel {
  min-width: 0;
  color: var(--dsw-alias-label-secondary, #61666b);
  font-size: 12px;
  font-weight: 400;
  line-height: 18px;
}

.factValue {
  min-width: 0;
  margin: 0;
  overflow-wrap: anywhere;
  color: var(--dsw-alias-label-primary, #0f1115);
  font-size: 13px;
  font-weight: 500;
  line-height: 20px;
  font-variant-numeric: tabular-nums;
}

.badgeText {
  white-space: nowrap;
}

.groupLabel {
  margin: 0;
  color: var(--dsw-alias-label-secondary, #61666b);
  font-size: 12px;
  font-weight: 500;
  line-height: 18px;
}

.meters {
  display: flex;
  flex-direction: column;
  gap: 8px;
}

.meter {
  display: grid;
  grid-template-columns: minmax(88px, 128px) minmax(80px, 1fr) 48px;
  align-items: center;
  gap: 10px;
  min-width: 0;
  font-size: 12px;
  line-height: 18px;
}

.meterLabel {
  overflow: hidden;
  color: var(--dsw-alias-label-secondary, #61666b);
  text-overflow: ellipsis;
  white-space: nowrap;
}

.meterTrack {
  display: block;
  height: 8px;
  overflow: hidden;
  border-radius: 999px;
  corner-shape: round;
  background: var(--dsw-alias-interactive-bg-active, rgba(38, 49, 72, 0.1));
}

.meterFill {
  display: block;
  height: 100%;
  border-radius: 999px;
  corner-shape: round;
  background: var(--dsw-alias-state-success-primary, #22c55e);
  transition: width var(--ds-transition-duration, 0.2s) var(--ds-ease-in-out, ease-in-out);
}

.meterValue {
  color: var(--dsw-alias-label-primary, #0f1115);
  text-align: right;
  font-weight: 500;
  font-variant-numeric: tabular-nums;
}

.actions {
  display: flex;
  flex-wrap: wrap;
  align-items: center;
  gap: 8px;
}

.actions button:focus-visible,
.groupToggle:focus-visible,
.datasetRow input:focus-visible,
.table input:focus-visible {
  outline: 2px solid var(--dsw-alias-state-business-primary, #569e58);
  outline-offset: 2px;
}

.muted {
  margin: 0;
  color: var(--dsw-alias-label-secondary, #61666b);
  font-size: 13px;
  line-height: 20px;
}

.uploadBlock {
  display: flex;
  flex-direction: column;
  gap: 6px;
  padding: 12px;
  border-radius: 8px;
  background: var(--dsw-alias-interactive-bg-hover, rgba(38, 49, 72, 0.06));
}

.datasetGroup {
  display: flex;
  overflow: hidden;
  flex-direction: column;
  gap: 0;
  border: 0.5px solid var(--dsw-alias-border-l2, rgba(0, 0, 0, 0.1));
  border-radius: 8px;
}

.groupToggle {
  display: flex;
  width: 100%;
  min-height: 34px;
  align-items: center;
  gap: 8px;
  padding: 7px 10px;
  border: 0;
  background: var(--dsw-alias-bg-layer-1, #ffffff);
  color: var(--dsw-alias-label-primary, #0f1115);
  font: inherit;
  font-size: 12px;
  font-weight: 600;
  line-height: 18px;
  cursor: pointer;
  text-align: left;
}

.groupToggle:hover {
  background: var(--dsw-alias-interactive-bg-hover, rgba(38, 49, 72, 0.06));
}

.groupToggle .datasetSize {
  margin-left: auto;
  font-weight: 400;
}

.groupChevron {
  width: 12px;
  flex: 0 0 12px;
  color: var(--dsw-alias-label-secondary, #61666b);
}

.subHeader {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 12px;
  margin-top: 4px;
  padding-top: 16px;
  border-top: 0.5px solid var(--dsw-alias-border-l2, rgba(0, 0, 0, 0.1));
}

.datasetList {
  display: flex;
  margin: 0;
  padding: 4px 10px 6px 30px;
  flex-direction: column;
  gap: 0;
  color: var(--dsw-alias-label-primary, #0f1115);
  font-size: 12px;
  line-height: 18px;
  list-style: none;
}

.fileInput {
  display: none;
}

.datasetRow {
  display: grid;
  min-width: 0;
  min-height: 32px;
  grid-template-columns: minmax(0, 1fr) auto auto;
  align-items: center;
  gap: 8px;
  padding: 6px 0;
  border-bottom: 0.5px solid var(--dsw-alias-border-l2, rgba(0, 0, 0, 0.1));
}

.datasetRow:last-child {
  border-bottom: 0;
}

.datasetRow label {
  display: inline-flex;
  min-width: 0;
  align-items: center;
  gap: 8px;
}

.datasetRow input,
.table input {
  accent-color: var(--dsw-alias-state-business-primary, #569e58);
}

.datasetPath {
  min-width: 0;
  overflow-wrap: anywhere;
  font-family: var(--ds-font-family-code, Consolas, "Microsoft YaHei", sans-serif);
}

.datasetSize {
  flex-shrink: 0;
  color: var(--dsw-alias-label-secondary, #61666b);
  font-variant-numeric: tabular-nums;
}

.unavailable {
  display: flex;
  flex-direction: column;
  gap: 6px;
  padding: 10px 12px;
  border: 1px solid var(--dsw-alias-state-warn-primary, #f59e0b);
  border-radius: 8px;
  background: var(--dsw-alias-state-warn-tertiary, #fef5e7);
}

.unavailable .muted {
  color: var(--dsw-alias-label-primary, #0f1115);
}

.unavailableList {
  margin: 0;
  padding-left: 18px;
  color: var(--dsw-alias-label-secondary, #61666b);
  font-size: 12px;
  line-height: 18px;
}

.trainingBadges {
  display: flex;
  margin: 0;
  flex-wrap: wrap;
  align-items: center;
  gap: 6px;
}

.progress {
  display: flex;
  flex-direction: column;
  gap: 10px;
  padding: 12px;
  border-radius: 8px;
  background: var(--dsw-alias-interactive-bg-hover, rgba(38, 49, 72, 0.06));
}

.progressTrack {
  display: block;
  height: 10px;
  overflow: hidden;
  border-radius: 999px;
  corner-shape: round;
  background: var(--dsw-alias-interactive-bg-active, rgba(38, 49, 72, 0.1));
}

.progressFill {
  display: block;
  height: 100%;
  border-radius: 999px;
  corner-shape: round;
  background: var(--dsw-alias-state-business-primary, #569e58);
  transition: width var(--ds-transition-duration-slow, 0.3s) var(--ds-ease-in-out, ease-in-out);
}

.progressMetrics {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(96px, 1fr));
  gap: 8px 16px;
}

.progressMetric {
  display: inline-flex;
  min-width: 0;
  flex-direction: column;
  gap: 1px;
  color: var(--dsw-alias-label-primary, #0f1115);
  font-size: 12px;
  font-weight: 500;
  line-height: 18px;
  font-variant-numeric: tabular-nums;
}

.errorLine,
.confirmLine,
.successLine {
  margin: 0;
  padding: 10px 12px;
  border-radius: 8px;
  color: var(--dsw-alias-label-primary, #0f1115);
  font-size: 12px;
  line-height: 18px;
}

.errorLine {
  border: 1px solid var(--dsw-alias-state-error-primary, #ec1313);
  background: var(--dsw-alias-interactive-bg-hover-danger, rgba(236, 19, 19, 0.05));
}

.confirmLine {
  border: 1px solid var(--dsw-alias-state-warn-primary, #f59e0b);
  background: var(--dsw-alias-state-warn-tertiary, #fef5e7);
}

.successLine {
  border: 1px solid var(--dsw-alias-state-success-primary, #22c55e);
  background: var(--dsw-alias-state-success-tertiary, #e6faed);
}

/* Wrapper for the initial loading / stream-error branches: owns the measure
   and the centring, so no sibling selector depends on branch DOM order. */
.statePanel {
  display: flex;
  width: min(100%, 960px);
  margin: 24px auto 0;
  flex-direction: column;
  align-items: flex-start;
  gap: 12px;
}

.statePanel .errorLine {
  width: 100%;
}

/* Docked failure band: stays inside .content, never a section. */
.content > .errorLine {
  position: sticky;
  bottom: 0;
  z-index: 6;
  width: min(100%, 960px);
  margin: 0 auto;
}

.tableScroll {
  width: 100%;
  overflow-x: auto;
  overscroll-behavior-x: contain;
}

.table {
  display: table;
  width: 100%;
  min-width: 720px;
  border-collapse: collapse;
  border-spacing: 0;
  color: var(--dsw-alias-label-primary, #0f1115);
  font-size: 12px;
  line-height: 18px;
  white-space: nowrap;
}

.table th,
.table td {
  padding: 8px 10px;
  border-bottom: 0.5px solid var(--dsw-alias-border-l2, rgba(0, 0, 0, 0.1));
  text-align: left;
  font-variant-numeric: tabular-nums;
}

.table thead th {
  border-bottom-color: var(--dsw-alias-border-l4, rgba(0, 0, 0, 0.16));
  background: var(--dsw-alias-bg-layer-1, #ffffff);
  color: var(--dsw-alias-label-secondary, #61666b);
  font-size: 11px;
  font-weight: 600;
  line-height: 16px;
}

/* Row headers (file name) must read as data, not as a second table head. */
.table tbody th {
  background: transparent;
  color: var(--dsw-alias-label-primary, #0f1115);
  font-size: 12px;
  font-weight: 500;
  line-height: 18px;
}

.table th:nth-child(2),
.table td:nth-child(2) {
  min-width: 220px;
  white-space: normal;
  overflow-wrap: anywhere;
}

.table th:nth-child(5),
.table td:nth-child(5) {
  min-width: 156px;
}

.table th:nth-child(6),
.table td:nth-child(6) {
  min-width: 220px;
}

.table tbody tr:hover {
  background: var(--dsw-alias-interactive-bg-hover, rgba(38, 49, 72, 0.06));
}

.table .actions {
  flex-wrap: nowrap;
}

.loading {
  position: relative;
  width: min(100%, 960px);
  min-height: 184px;
  margin: 24px auto 0;
  padding: 164px 4px 0;
  color: var(--dsw-alias-label-secondary, #61666b);
  font-size: 13px;
  text-align: center;
  line-height: 20px;
}

/* Two section outlines (56 / 88) preview the cards that are about to arrive.
   They are real (empty, aria-hidden) nodes because CSS allows one ::before and
   one ::after per host: nested pseudo-elements (::before::before) are invalid
   and get dropped silently, which would leave two empty frames. */
.loadingCard,
.loadingCardWide {
  position: absolute;
  left: 0;
  width: 100%;
  border: 0.5px solid var(--dsw-alias-border-l2, rgba(0, 0, 0, 0.1));
  border-radius: 12px;
  background: var(--dsw-alias-bg-layer-1, #ffffff);
  animation: lifeSkeletonPulse 1.2s var(--ds-ease-in-out, ease-in-out) infinite alternate;
}

.loadingCard {
  top: 0;
  height: 56px;
}

.loadingCardWide {
  top: 72px;
  height: 88px;
}

/* Placeholder bars ride the outline's opacity pulse. */
.loadingCard::before,
.loadingCardWide::before,
.loadingCardWide::after {
  position: absolute;
  height: 10px;
  border-radius: 999px;
  corner-shape: round;
  background: var(--dsw-alias-bg-skeleton, rgba(0, 0, 0, 0.04));
  content: "";
}

.loadingCard::before {
  top: 18px;
  left: 20px;
  width: min(34%, 180px);
}

.loadingCardWide::before {
  top: 22px;
  left: 20px;
  width: min(62%, 360px);
}

.loadingCardWide::after {
  top: 48px;
  left: 20px;
  width: min(42%, 240px);
}

/* New visual hook. Apply only to stop / force-release / delete actions. */
.actions .dangerAction {
  border: 1px solid var(--dsw-alias-state-error-primary, #ec1313);
  background: var(--dsw-alias-bg-layer-1, #ffffff);
  color: var(--dsw-alias-state-error-primary, #ec1313);
}

.actions .dangerAction:hover:not(:disabled) {
  background: var(--dsw-alias-interactive-bg-hover-danger, rgba(236, 19, 19, 0.05));
}

.actions .dangerAction:focus-visible {
  outline-color: var(--dsw-alias-state-error-primary, #ec1313);
}

/* Destructive cluster: pushed to the trailing edge of the action row and cut
   off by one hairline so position, not just colour, carries the warning. */
.actions .actionCluster {
  display: flex;
  min-width: 0;
  align-items: center;
  gap: 8px;
  margin-left: auto;
  padding-left: 8px;
  border-left: 1px solid var(--dsw-alias-border-l2, rgba(0, 0, 0, 0.1));
}

/* Inline note for surfaces that did not answer: a note, not a block. */
.unavailableNote {
  margin: 0;
  padding: 6px 10px;
  border: 1px solid var(--dsw-alias-state-warn-primary, #f59e0b);
  border-radius: 8px;
  background: var(--dsw-alias-state-warn-tertiary, #fef5e7);
  color: var(--dsw-alias-label-primary, #0f1115);
  font-size: 12px;
  line-height: 18px;
}

.unavailableNote .muted {
  color: var(--dsw-alias-label-primary, #0f1115);
}

/* Names of the rows a destructive verb is about to remove. */
.impactList {
  margin: 0;
  padding-left: 16px;
  max-height: 88px;
  overflow-y: auto;
  font-size: 12px;
  font-family: var(--ds-font-family-code, Consolas, "Microsoft YaHei", sans-serif);
  line-height: 18px;
  list-style: disc;
}

/* Collapsible section head: summary carries the h3, marker is ours. */
.sectionSummary {
  display: flex;
  margin: 0;
  align-items: center;
  gap: 8px;
  padding: 2px 0;
  cursor: pointer;
  list-style: none;
}

.sectionSummary::-webkit-details-marker {
  display: none;
}

.sectionSummary::marker {
  content: "";
}

/* Fold chevron: shape only, so it never enters the summary's text. Zero
   border-radius means no circular radius, so no corner-shape pairing needed. */
.sectionSummary::after {
  width: 0;
  height: 0;
  flex: 0 0 auto;
  margin-left: auto;
  border-top: 4px solid transparent;
  border-bottom: 4px solid transparent;
  border-left: 6px solid var(--dsw-alias-label-secondary, #61666b);
  content: "";
  transition: transform var(--ds-transition-duration-fast, 0.1s) var(--ds-ease-in-out, ease-in-out);
}

/* <summary> never exposes aria-expanded in Chromium; [open] on the parent is
   the rule that actually fires. */
details[open] > .sectionSummary::after {
  transform: rotate(90deg);
}

/* A folded section loses the 16px gap to its first child, which would make the
   title sit 4px tighter than in an open one — the rhythm must not move. */
details:not([open]) > .sectionSummary {
  padding-bottom: 4px;
}

.sectionSummary:hover {
  border-radius: 8px;
  background: var(--dsw-alias-interactive-bg-hover, rgba(38, 49, 72, 0.06));
}

.sectionSummary:focus-visible {
  outline: 2px solid var(--dsw-alias-state-business-primary, #569e58);
  outline-offset: 2px;
}

.sectionSummary .sectionTitle::before {
  background: var(--dsw-alias-state-business-primary, #569e58);
}

@keyframes lifeSkeletonPulse {
  from { opacity: 0.55; }
  to { opacity: 1; }
}

@container (max-width: 560px) {
  .facts {
    grid-template-columns: 1fr;
  }

  .subHeader {
    align-items: flex-start;
    flex-direction: column;
  }

  .datasetRow {
    grid-template-columns: minmax(0, 1fr) auto;
  }

  /* Wrapped clusters start a fresh row; a leading divider reads as noise. */
  .actions .actionCluster {
    margin-left: 0;
    padding-left: 0;
    border-left: 0;
  }
}

@container (max-width: 420px) {
  .meter {
    grid-template-columns: minmax(72px, 96px) minmax(72px, 1fr) 42px;
    gap: 8px;
  }
}

/* Insurance tier: below this the three-column meter cannot hold at all. */
@container (max-width: 240px) {
  .meter {
    grid-template-columns: 1fr;
    gap: 4px;
  }

  .meterLabel,
  .meterValue {
    display: inline-flex;
    justify-content: space-between;
  }
}

@media (max-width: 640px) {
  .page {
    padding: 0 16px 32px;
  }

  .content {
    gap: 12px;
  }

  .header {
    padding: 16px 2px 12px;
  }

  .section {
    gap: 14px;
    padding: 16px;
    border-radius: 10px;
  }
}

@media (prefers-reduced-motion: reduce) {
  .meterFill,
  .progressFill,
  .sectionSummary::after {
    transition-duration: 0.01ms;
  }

  .loadingCard,
  .loadingCardWide {
    animation: none;
  }
}
```

## 8. 工程接线建议（与 CSS 分开，不属于本阶段源码改动）

### 8.1 危险操作 class

`Button` 已支持 `className`（`packages/client/ui-primitives/src/Button.tsx:18-26`），因此不需要新增组件 prop 或业务数据流。工程阶段只给以下既有按钮加 `className={css.dangerAction}`：

- `lifeStop`
- `trainStop`
- `trainReset`
- `deleteDataset`
- `deleteCheckpoint`
- `deleteKnowledge`

普通主动作最多每组一个，可直接使用 primitives 已有的 `variant="primary"`；建议优先级：`trainStart`、用户选中文件后的 `uploadSend`、`runConsolidate`。其余保持默认 ghost。

### 8.2 表格 wrapper（已在本规范内拍板）

UX §5.1 已否决 `.table { display:block; overflow-x:auto }`，改为 `.tableScroll` wrapper。**§7 的 CSS 已按此写入**，不再需要二选一：

```tsx
<div className={css.tableScroll} role="group" aria-labelledby={labelId} tabIndex={0}>
  <table className={css.table}>…</table>
</div>
```

```css
.tableScroll { width: 100%; overflow-x: auto; overscroll-behavior-x: contain; }
.table { display: table; width: 100%; min-width: 720px; border-collapse: collapse; }
```

这不改变 table 的语义或数据流；wrapper 本身是纯布局容器，由 UX 决定 `role="group"` / `tabIndex={0}` 的可达性属性。

### 8.3 meter 语义状态（暂不实施）

当前 `Meter` 只有 `label` 和 `value`，无法可靠知道“警告/失败”；高 needs 或 drives 可能只是强度而非故障。不要按阈值写 `nth-child` 或 CSS 猜测。若未来 runtime 提供 tone，可再映射：normal→`state-success-primary`，ongoing→`state-business-primary`，warning→`state-warn-primary`，failure→`state-error-primary`。本轮只把稳定生命读数与训练过程从视觉上分开。

### 8.4 加载骨架的两个空节点（唯一新增 DOM）

CSS 限定每个宿主元素只能生成一个 `::before` 和一个 `::after`，伪元素上不能再挂伪元素。要让占位条保住 `border-radius: 999px` 的胶囊形，两张轮廓必须是真实节点。这是本规范**唯一**新增的 DOM，且两个节点都是空的、`aria-hidden`：

```tsx
/* LifePanel.tsx:174-176 */
<p className={css.loading} role="status">
  {t('loading')}
  <span className={css.loadingCard} aria-hidden="true" />
  <span className={css.loadingCardWide} aria-hidden="true" />
</p>
```

测试影响：零。`<span>` 无文本子节点，RTL `getNodeText` 只拼直接文本子节点，`getByText(en.loading)` 仍唯一命中该 `<p>`；`aria-hidden` 保证 `role="status"` 不朗读多余内容。

**DOM-free 备选**（不建议，仅在绝对不能加节点时用）：把占位条改画成同一个伪元素上的多层 `linear-gradient`，用 `background-position`/`background-size` 定位。代价是渐变成不了胶囊圆角，只能是直角。

```css
.loading::after {
  background-image:
    linear-gradient(var(--dsw-alias-bg-skeleton, rgba(0, 0, 0, 0.04)) 0 100%),
    linear-gradient(var(--dsw-alias-bg-skeleton, rgba(0, 0, 0, 0.04)) 0 100%);
  background-repeat: no-repeat, no-repeat;
  background-position: 20px 22px, 20px 48px;
  background-size: min(62%, 360px) 10px, min(42%, 240px) 10px;
}
```

## 9. 变更登记（v2.2 · 并入 UX 异议 V-1～V-6 + 骨架 CSS 修正 + fe 第一轮 TSX 语义改造）

> v2.2 面向 fe 的合并轮：§7 的 CSS 已覆盖 fe 在 TSX 上挂的全部 7 个新类名（`.sectionSummary`、`.tableScroll`、`.dangerAction`、`.actionCluster`、`.impactList`、`.statePanel`、`.unavailableNote`），并补齐 `<details>/<summary>` 化之后才暴露的两条视觉规则。

| 编号 | UX 异议 | 结论 | 本文件改动 |
|---|---|---|---|
| V-1 | 双骨架形态不成立 | **采纳** | `.loading` 从"两根细条"改为两张区块轮廓（56px / 88px + 内部标题/正文占位条），说明文案居中放在轮廓下方。**v2.1 修正**：占位条最初写成 `.loading::before::before` 等二级伪元素（无效 CSS，会被静默丢弃），改为两个空的 `aria-hidden` `<span>`（`.loadingCard` / `.loadingCardWide`）承载轮廓，占位条走各自的 `::before/::after`，胶囊圆角得以保留 |
| V-2 | `.page > .errorLine` + `.page > .errorLine + button` 兄弟选择器 | **采纳（必须改）** | 删除两条兄弟选择器；新增 `.statePanel` 承担宽度上限与居中；另新增 `.content > .errorLine`（sticky 底部停靠）承担运行期失败条，两套位置规则不再冲突 |
| V-3 | `.table th` 会污染行标题 | **采纳（必须改）** | 拆为 `.table thead th`（11/16 600、secondary、实底）与 `.table tbody th`（12/18 500、`label-primary`、`background: transparent`）；列 `min-width` 规则同时覆盖 `th` 与 `td` |
| V-4 | facts 单列断点 420px 太晚 | **采纳** | `@container (max-width: 420px)` → `560px`（两列仅在面板宽 ≥ 664px 出现）；420px 档保留给 meter 收窄，另加 240px 保险档 |
| V-5 | 动作行需要分组容器 | **采纳** | 新增 `.actions .actionCluster`（`margin-left:auto` + 1px `border-l2` 竖分隔），不引入新颜色/形状 token；容器宽 ≤ 560px 时去掉分隔与 `margin-left:auto`，避免换行后新行首出现孤立竖线 |
| V-6 | `container-type` 隐含 BFC | **采纳为纪律** | §2「明确避免」与 §6 均写入"禁止在 section 内使用 sticky"；本规范的 sticky 只出现在 `.header` 与 `.content > .errorLine` |

同时并入 UX §4-A 的 A1（`dl.facts`/`dd.factValue` margin 复位）、A7/A8（`.sectionSummary` 与 focus ring）、A9（`.impactList`）、A10（`.page` `scroll-padding-top: 96px`、`.section` `scroll-margin-top: 96px`）、A13（`.unavailableNote`）。

### v2.2 增量（fe 第一轮 TSX 落地后）

| 改动 | 原因 |
|---|---|
| `.sectionSummary:hover`（`border-radius: 8px` + `interactive-bg-hover`） | `<summary>` 整行可点，必须有 hover 反馈；否则用户看不出标题行是控件 |
| `details:not([open]) > .sectionSummary { padding-bottom: 4px }` | 折叠后 `gap: 16px` 不再作用于首个子节点，标题会比展开态贴紧 4px，节奏会跳；补 4px 让两态一致 |
| chevron 的 `border-left` 5px → 6px | 与 ui-primitives `JsonTree.module.css:314-323` 的既有三角形尺寸对齐，保持一致 |
| §5 字阶表、§6 区块小节同步 | 明确 `.sectionTitle` 现在挂在 `<summary>` 内、`.table thead th` 与 `tbody th` 分行 |

## 10. 验收清单

- light：所有 12–13px 信息文本为 primary/secondary，不以 tertiary 或亮状态色承载正文。
- dark：页面 `bg-base`、section `bg-layer-1` 自动形成层级，状态浅底来自各自 tertiary token。
- facts：容器宽 < 560px（面板宽 < 664px）必须是单列；≥ 560px 才双列以上；路径可断行；meter 轨道不塌陷。
- checkpoints：6 列不挤扁，横向滚动只在 `.tableScroll` 内；行标题（文件名）与数据行同权，不呈现成第二排表头。
- sticky header：滚动六区块时标题与副标题持续可见，不遮挡首个 section；没有任何 sticky 写在 section 内部。
- destructive actions：不再与上传/喂食/睡眠等同权；不得用位置或文案选择器实现，只靠 `.dangerAction` + `.actionCluster`。
- loading：两张轮廓内**各能看到占位条**（56px 卡内 1 条、88px 卡内 2 条，均为胶囊形）——若只剩两个空框，说明有人把占位条写回了二级伪元素。
