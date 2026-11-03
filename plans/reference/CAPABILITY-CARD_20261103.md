# 能力边界卡（唯一对外表述源）

> 文档状态：[当前·能力表述权威]。日期 2026-11-03，Asia/Shanghai。本卡只回答一件事：**现在这台模型会什么、不会什么、哪些说法已被实验否掉**。README、MAINLINE、复审包等对外文字必须与本卡一致；冲突时以本卡为准。
>
> 每条都带证据指针（报告文件 + JSON 字段）。指针里的报告都已入库，可直接打开核对。

## 0. 先说人话（首次出现的术语）

- **主体（subject）**：同一个模型实例带着自己的状态、记忆继续做事，不是每道题重新开一个模型。
- **原生字节**：模型直接读写原始 UTF-8 字节，不经过分词器，也不把思考外包给另一个大模型。
- **选择头（selection head）**：一个很小的附加模块（37,120 个参数），负责“从几条候选记录里挑一条”。
- **位置监督（position supervision）**：训练时额外告诉模型“这个字应当从记录的第几个字抄”，属于辅助信号，不是独立能力。
- **confirm / 确认面**：留到最终验收才允许读的题集。提前读一次就算污染，所有阶段都必须为 0。
- **臂（arm）**：一次对照实验里同时跑的一个配置，例如 A=锁定路线、B=只锁停梯度、C=不锁。
- **判负（negative）**：预先写好的门没有过。它是实验结果，不是工程故障。

## 1. 已证（有实验证据、当前可对外说）

| # | 能力/事实 | 证据指针（报告 + 字段） | 一句话白话 |
|---|---|---|---|
| C1 | 原生字节读写在同一张图内闭环：解码逐字节复制或生成，全程 257 字节 + EOS | `reports/taiji_core_record_relation_14_4_20261102.json` → `content`（`correct`/`wrong`/`null` 三个分支）；`reports/taiji_binding_copy_15_1_compare_20261102.json` → `gates.task` | 模型确实自己写字，不是让外部模型代写 |
| C2 | 内容通路是必要的：去掉来源/换错记录/换错位置，分数会明显下降 | `reports/taiji_core_record_relation_14_6_20261102.json` → `gates.content.conditions`（8 例固定对照下降 0.75 / 0.75 / 0.375） | 模型确实在读提供的记录，不是背模板 |
| C3 | 保存—新进程加载—继续训练，参数/优化器矩/后继分布逐位一致 | `reports/taiji_core_record_relation_14_6_preflight_20261102.json` → `checks`（6/6）；`reports/taiji_core_r027_subject_gates_20261101.json` → `boundaries`、`checks`（8/8） | 训练可以中断再续，结果不会漂 |
| C4 | 对照公平：同父、同数据、同剂量、同预算 | `reports/taiji_binding_select_16_1_compare_20261103.json` → `configs`（2 seed × 2 臂 × 300 更新，240 开发例 = 12 world × 20） | 比较的是机制，不是谁多训了 |
| C5 | 新增参数很小：选择头 37,120，容器 3,563,601 / 上限 3,567,795 | `reports/taiji_selection_routing_17_3_verify_20261103.json` → `t7_parameters` | 没有靠堆一个大模块来换分数 |
| C6 | 有序编码能把“字序不同”送到选择打分：20 对换序句最小差 0.428526 / 0.052838，旧头 0.0 | `reports/taiji_selection_routing_17_3_verify_20261103.json` → `r1_calibration`、`r1_calibration.old_head_witness` | 同一批字换个顺序，模型现在能看出不同 |
| C7 | 确认面零污染：14.4–17.3 全部 `confirm` 读取 0 次 | `...14_6...json` → `accounting`；`...15_1_compare...json`、`...16_1_compare...json` → `accounting.consumed_confirm_examples=0`；`...17_3_verify...json` → `confirm_reads`（顶层）与 `t11_api.confirm_reads` | 最终考题一次都没提前看 |
| C8 | 同形定价已完成：三臂 13,092.25 CPU / 1,800 Adam / 495.0 MiB；二臂 8,750.34 / 1,200 / 330.0 MiB | `reports/taiji_selection_routing_17_3_price_20261103.json` + `plans/manifests/taiji_selection_routing_17_3_price_20261103.json` | 要花多少钱已经算清，尚未开跑 |

## 2. 未证（还没有实验能支持，对外不得声称）

| # | 待证能力 | 现状读数 | 证据指针 | 一句话白话 |
|---|---|---|---|---|
| U1 | 选择—解码绑定能提升任务成绩 | 16.1 总体完成率 0.0500 / 0.0167 | `...16_1_compare...json` → `configs.*.completed_rate` | 模型挑得不准，写完更不准 |
| U2 | 新描述 / 新输出名字的泛化 | 16.1 新名字 0/60、新描述 0/60、未见组合 0/36；15.1 新名字/新描述 0 | `...16_1_compare...json` → `configs.*.subsets`（`subset_denominators` 60/60/36/15）；`...15_1_compare...json` → `gates.task` | 没见过的说法基本写不出来 |
| U3 | 选择本身要准（教师条件下） | top-1 0.3833 / 0.3542，门 0.90 | `...16_1_compare...json` → `configs.*.selection_top1` | 给它答案线索，也只有三成多挑对 |
| U4 | 未见实体 | 14.6 `unseen_entity` 0.2125，门 0.60 | `reports/taiji_core_record_relation_14_6_20261102.json` → `gates.task` | 换一批没见过的名字就基本失效 |
| U5 | 旧能力保持 | `retention_not_admissible`（当前没有合格父） | `...17_3_verify...json` → `t10_retention`；`INVARIANT-LEDGER` L9 | 还没资格谈“学新不忘旧” |
| U6 | 跨会话经验写入 / 长期能力 | `experience_surface_absent`（声明缺失，不是能力） | `...17_3_verify...json` → `t11_api`、`applicability.T11`；`INVARIANT-LEDGER` L13 | 交互结束后不会自己变强，只能靠重新训练 |
| U7 | 路线锁定（最终 Q 用选择结果锁复制分布）带来的任务增益 | 未测，17 未准入 | `PRICING-17_3_SELECTION_ROUTING_20261103.md` §0、§8 | 这是 17 唯一要回答的问题，钱还没花 |
| U8 | 一般中文写作能力 | 14.6 完整动作 0.5844，门 0.80 | `...14_6...json` → `gates.task` | 能写对一半多，离可用还远 |

## 3. 已证否（在已测条件下不成立，正文不得再宣称）

| # | 被否说法 | 实测 | 证据指针 | 白话解释 |
|---|---|---|---|---|
| N1 | “字节均值嵌入足以分辨同字不同序” | 旧头换序差 0.0 / 0.0 | `...17_3_verify...json` → `old_head_control`；16.1 P1 | 只取平均值，顺序信息被抹平了 |
| N2 | “锁定复制比不锁更好”（A−C ≥ 0.10） | 14.6 A−C +0.1063，但 bootstrap 下界 −0.0375 | `...14_6...json` → `gates.task.structure` | 点估计为正但统计上站不住 |
| N3 | “选择支路代理信用带来额外收益”（A−B） | 16.1 A−B −0.27708 | `...16_1_compare...json` → `contrast_A_minus_B` | 加了这条信用反而更差 |
| N4 | “位置监督是必需项”（旧 P4 的推广） | 40 例中 7/40 对象字节已全对，错在后续句长；强制只在 alpha/beta 层 | `...14_6...json` 原始 0/40 保留；`ATTRIBUTION-ARCH-VS-ALG` §2 | 旧归因把“句子长度没对上”误记成“不会复制” |
| N5 | “反序配对率可作任务门” | 16.1 原始字节不足，配对率撤回/不可判 | `SPEC-N16-01` §4 仪器缺陷；`...16_1_3_closeout...json` | 没存原始输出，配对重算不可信 |
| N6 | “六臂可塑性选择能转为后继净收益” | ΔNLL_holdout −0.1370，未优于对照 −0.1402；六臂全超保持容差 | `reports/taiji_core_r026_plasticity_arms_20261031.json` | 合成夹具上，出生/生长方向 8 步内没回本 |
| N7 | “出生方向能自我偿付” | 同上，按预注册判负 | `...r026...json` | 同上 |
| N8 | “自演进 / 终身学习 / 自我改进已经接通” | 从未落地；17 也没有该合同 | `INVARIANT-LEDGER` L13；`...17_3_verify...json` → `t11_api`、`applicability.T11` | 这是长期目标，不是现状 |

> N2/N3/N6/N7 的“否”都限定在当时的夹具、剂量与预算；它们不证明原理在任何规模都不成立，但足以禁止把“已经证明有效”写进正文。

## 4. 措辞审计

| 禁用表述 | 现状 | 处置 |
|---|---|---|
| 自演进、自我演进、自我改进、终身学习、self-evolving、self-improving、lifelong | README 曾有 “Lifelong development”；MAINLINE/03 仅在“禁止当作结论”的警示句里出现 | README 已改为“long-horizon development (research goal, not demonstrated)”；其余保留警示用法 |
| “已实现 / 已接通 / 已具备”用于 U1–U8 | 未发现正文宣称 | 本卡 §2 固化为“未证” |
| 把工程门通过写成任务能力 | 14.4/14.5 报告均写 `engineering_only` | 保持 |

## 5. 与对外文档的对应

- `README.md` / `README.zh-CN.md`：第 2 节能力清单只保留 C1–C7，明确写出 U1–U8 未证、N8 不存在。
- `plans/MAINLINE.md`：只引用本卡编号，不另写能力结论。
- `plans/active/roadmap/03_CURRENT_EXECUTION.md`：执行流水，不承担能力表述。
- 本卡是能力边界唯一权威；新增能力必须先有报告字段，再进本卡。
