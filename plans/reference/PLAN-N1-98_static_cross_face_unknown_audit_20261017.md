# PLAN-N1-98 · 已消费 v5/v6 unknown 题面的静态结构审计 · 2026-10-17

**状态：owner 已批准范围；审计已完成，判读见 [EXP-N1-98](EXP-N1-98_static_cross_face_unknown_audit_adjudication_20261017.md)。**本件只覆盖 §1–§5。它落实 [HANDOVER-N1-AFTER-V6 §2 甲](HANDOVER_N1_AFTER_V6_20261017.md) 后续的静态题面核查；未重开任何已消费确认面，也未修改此前冻结的训练目标、确认判据或产品默认。

## 0. 问题与边界

[EXP-N1-97](EXP-N1-97_unknown_abstention_cross_face_census_20261017.md) 发现同九枚终端在两张已消费面上的 position `unknown` 精确输出率为 v5 23/24、v6 0/15。该差异指向题面条件变化，但当前报告行没有题面结构字段。本审计只描述可观察输入结构与已出版输出的对应关系，不检验能力、不估计因果、不挑选新训练配方。

允许读取两个已消费清单及其已出版报告：

- v5：`plans/manifests/n1_fresh_confirmation_v5_20261016.json`；报告绑定摘要 `22ba37976bc6eaaaf2425b84714a6efa6e95e7e5aa05f32eb3a7c0c10d9ffc2d`，32 个 episode。
- v6：`plans/manifests/n1_fresh_confirmation_v6_20261017.json`；确认报告登记原始 SHA-256 `7dddf00f3f2e24381728855b218e67bc8160fd6badeb1a7a933937d999766d49`，面含牺牲与确认两个子集；只分析确认报告已有的 unknown 行，不扩展到牺牲子集的能力读数。
- 连接的报告分别是 `reports/taiji_n1_transition_factorial_confirmation_run89.json` 与 `reports/taiji_n1_transition_factorial_confirmation_v6.json`。按 `(face, item_id, seed, arm)` 连接，不跨题面匹配不同 ID；每题保留既有 arm、`free_exact` 与逐字输出。

## 1. 固定描述字段

只对清单中标记为 `unknown` 的项目提取并汇总：

1. 原始问题文本的 UTF-8 字节数与问题中实体名的原文；输出报表不复述完整合成题面。
2. 按清单原序排列的观测条数、实体名、候选值，以及问题实体名是否在观测实体中精确出现、出现位置和次数。
3. 对应报告行中输出是否逐字等于 `unknown`、既有 `free_exact` 值及输出字节串；只引用原行，不重新加载或评分模型。

不计算新距离阈值、不做语义标签、不做显著性检验、不重新运行四门。v5 与 v6 的比例不合并；三臂读数作为描述性背景，重点报告 position 的跨面变化。

## 2. 完整性与停止条件

- 读取清单前后均核对 SHA-256；其值须与对应已出版报告一致。报告状态、查询数、item ID 唯一性、unknown 行数和九枚终端身份须与 [EXP-N1-97](EXP-N1-97_unknown_abstention_cross_face_census_20261017.md) 及已消费回执相符。
- 每张面内，清单 unknown 项须与该面每个 seed/arm 的 unknown 报告行一一对应；缺行、多行、ID 冲突或摘要不符即 `stopped`，不补行、不改清单、不再查询。
- 只允许 JSON/Markdown 报告写入各自唯一的新路径。原清单、既有报告、checkpoint、训练代码和判据页均只读。

## 3. 明确禁止与交付

新增模型/checkpoint 加载、推理、训练、参数更新、新题生成、新查询、重测 v5/v6、访问 dev/其他留出面、修改 gate/预注册、产品行为或远端状态均为 **0**。本地普通 CPU 只用于清单解析与表格汇总，不设训练预算，也不把它计为能力计算。

交付唯一一份结构普查 JSON `reports/taiji_n1_unknown_face_structure_audit_20261017.json` 与 `plans/reference/EXP-N1-98_static_cross_face_unknown_audit_adjudication_20261017.md` 判读；结论仅回答两张已消费面在上述固定字段上的差异。若数据不能把 0/15 的失效定位到可复核的结构差异，则明确保留“原因未知”，再提交下一项计划范围供 owner 裁定。
