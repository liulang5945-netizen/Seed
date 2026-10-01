# SPEC-M6-01 · replay 注入上下文行归一化预注册（冻结判据，只追加）

> **性质**：预注册（判据先于实现冻结）。2026-10-01 由 owner 弹窗裁定授权（处置方式＝乙"授权有原则归一化"）。
> 背景与候选账目见 08 ㊵-45／㊵-48；本件冻结后实现只许照此执行，偏离即回退重注册。

## 1. 问题陈述

2026-09-28 起产品在回合内注入两类上下文 user 消息（`source.kind=life-context` 回路状态快照、
`source.kind=memory-context` 记忆召回），其载荷含运行时波动值（`age=`、`tick=`、`needs[...]`、记忆查询
原文含宿主路径）。web 面 replay 语料（09-28 录）不含这些行 ⇒ 持久回放比较（`assertReplaySession` 的
`persisted replay` 断言）在受影响 lane 必然失配。方法学候选 ~25 文件（rerun3 口径 27，逐文件因果未确证；
rerun6 面跑后将按新鲜日志重推导，见 §4）。

## 2. 冻结的折叠形状

在 `apps/web/tests/scaffold.ts` 的 `normalizeWebSessionVolatiles`（web 面逐行规范化层，actual 与
expected 两侧同用）增加一条**逐行 JSON 折叠**规则：

- 命中条件：该行解析为 `user/message` 事件，且 `data.source.kind === 'life-context'` 或 `'memory-context'`。
- 折叠动作：`data.content[*].text` 与 `data.source.sections[*].text` 全部替换为稳定 token
  `{{lifeContext}}`／`{{memoryContext}}`（kind 一一对应），**行的其余结构（type/turn/id 令牌/surfaceOp）原样保留**。
- 两侧同折 ⇒ 折叠后比较的是"注入行**存在**、位置正确、结构正确"，波动载荷不进基线。
- 不折叠其它任何行；`sections[*].name` 保留（结构面）。

**为什么不是整行丢弃**：丢弃会让注入逻辑（来源、轮次位置、行数）退出断言面——注入回归将无门可拦。
**为什么不是 refresh 不加折叠**：直接 refresh 会把 `tick=384` 等波动值烤进语料，下一 tick 必红。

## 3. 语料刷新半径与验收判据

1. 候选集＝rerun6 面跑日志中，FAIL 窗口内 `life-context`/`memory-context` 出现于 `+` 侧的文件
   （方法同 ㊵-45，ANSI 剥离；**名单随 rerun6 落盘为准**）。
2. 对候选集逐 lane `DSH_SNAPSHOT=refresh` 单跑：重写 `session.v3.jsonl`（及该 lane 的 .md 金样，若有变化）。
3. **验收（全部满足才提交）**：
   - 候选集逐 lane replay 单跑转绿（若某 lane refresh 后仍红 ⇒ 该 lane 另有其因，如实归因不硬凑）；
   - 抽 3 条已绿 lane（sidebar-right、seeded-history、plugin-install-registry）单跑不回退；
   - refresh 的 diff 逐文件审：**新增行只许是折叠后的注入行**，已有行不许出现非令牌化改动。
4. 失败出路：若折叠导致两侧对不齐（如期望侧结构与实际侧同形但错位），回退折叠、保留候选账目、重注册。

## 4. 边界

- 不动产品注入行为；不动语料的非注入行；不声称"注入族 25 文件全部由此转绿"——逐 lane 实测为准。
- aria 面若受注入行可见性影响（会话 UI 渲染注入消息），由同一 refresh 批次顺带重写，diff 审查同 §3.3。

## 5. 附录（2026-10-01 验收期发现，同日追加；记录最终实现形状）

验收过程暴露两处叠加的规范化缺口（与注入行无关但同属"宿主波动值不得进语料"），最终实现比 §2 多一层：

1. **值层世界根折叠**：临时世界内的一切路径都随 `dsh-web-e2e-ws-<随机段>` 逐次漂移，且同一路径以
   不同转义深度出现（tool arguments 是 JSON 套 JSON）、不同分隔符拼法（tool result 文本用正斜杠）。
   在 `normalizeWebSessionVolatiles` 的**值层**（字符串尚未 JSON 序列化处）折叠：`harnessHome`
   ＝`<世界根>\.dsh-home` → `{{harnessHome}}`、世界根 → `{{tempWorld}}`（原生＋正斜杠两种拼法，最长
   优先；会话 cwd 的 `{{cwd}}` 形状由既有 cwdSpellings 先行折叠）。值层折叠一次性覆盖所有嵌套深度。
2. **绝对路径守卫**：expected 侧语料 header 的 cwd 已是令牌（`{{cwd}}`），`dirname(令牌)`＝`.`——
   不加守卫会把整个文件里的句点全折成 `{{tempWorld}}`（实测发生过并已修）。仅当 session cwd 为
   绝对路径时才推导世界根。

行层 `{{harnessHome}}` split 链保留为冗余防线。验收：12 条纯注入 lane refresh 后 8 条转绿并在 replay
复验通过（file-upload-round 经附录补丁后 refresh＋replay 双绿）；4 条仍红（lifecycle-chrome／ptc-round／
present-svg／question-composer）＝另有其因，按 §3 如实归因不硬凑，其 refresh 产物已回退不提交。
