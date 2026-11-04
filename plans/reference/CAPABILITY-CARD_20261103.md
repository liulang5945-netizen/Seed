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
| C1 | 原生字节读写在同一张图内闭环：解码逐字节复制或生成，256字节值＋EOS，共257类 | `reports/taiji_core_record_relation_14_4_20261102.json` → `content`（`correct`/`wrong`/`null` 三个分支）；`reports/taiji_binding_copy_15_1_compare_20261102.json` → `gates.task` | 模型确实自己写字，不是让外部模型代写 |
| C2 | 内容通路是必要的：去掉来源/换错记录/换错位置，分数会明显下降 | `reports/taiji_core_record_relation_14_6_20261102.json` → `gates.content.conditions`（8 例固定对照下降 0.75 / 0.75 / 0.375） | 记录信息对该范围任务有因果影响 |
| C3 | 保存—新进程加载—继续训练，参数/优化器矩/后继分布逐位一致 | `reports/taiji_core_record_relation_14_6_preflight_20261102.json` → `checks`（6/6）；`reports/taiji_core_r027_subject_gates_20261101.json` → `boundaries`、`checks`（8/8） | 训练可以中断再续，结果不会漂 |
| C4 | N16同父/剂量等条件有记录，目标处理存在D16-4差异 | reports/taiji_binding_select_16_1_attribution_20261103.json → contract_fault | B触发自模仿、A未触发，增量需保留目标差异解释 |
| C5 | 有序头37,120参数；容器3,563,859（binding 3,526,481＋head 37,120＋span 258），冻结为单一**语义身份**`candidate_sha_v2` 773dcec2…（旧字节绑定摘要964a8c68…仅作历史） | `taiji/candidate_identity.py`；`reports/taiji_selection_routing_17_2r_identity_20261104.json`；`reports/taiji_selection_routing_17_2r_20261103.json` → `candidate`、`carriers` | 参数版本已收成一个身份，格式改动经零更新探针验证不改变身份；实际能力仍待任务比较 |
| C6 | 有序编码能把“字序不同”送到选择打分：20 对换序句最小差 0.428526 / 0.052838，旧头 0.0 | `reports/taiji_selection_routing_17_3_verify_20261103.json` → `r1_calibration`、`r1_calibration.old_head_witness` | 同一批字换个顺序，模型现在能看出不同 |
| C7 | 确认面零污染：14.4–17.3 全部 `confirm` 读取 0 次 | `...14_6...json` → `accounting`；`...15_1_compare...json`、`...16_1_compare...json` → `accounting.consumed_confirm_examples=0`；`...17_3_verify...json` → `confirm_reads`（顶层）与 `t11_api.confirm_reads` | 最终考题一次都没提前看 |
| C8 | 改前span/参数配置的价格分项已交付并`superseded`（13,092.25 / 8,750.34 CPU，不得当作当前价）；当前价为三臂15,018.52 / 二臂10,182.68 CPU | `reports/taiji_selection_routing_17_3_price_20261103.json.superseded.json` → 新价指针；`reports/taiji_selection_routing_17_2r_price_20261103.json` → `frozen` | 旧价只作差额对照 |
| C9 | 新图精确复制控制720/720通过（旧文件701/720为改前基线，已`superseded`） | `reports/taiji_selection_routing_17_2_t3_supplement_20261103_v3.json` → `forced_gold_probability` | 通过范围为gold控制下f64及reader-f32，正常定位/完整主体另验 |
| C10 | 当前词表（3物件×5地点）下，模型对**未见过的新名字**逐字节正确率 0.000，且删记录/换错记录/换位置四种条件下新名字从未逐字出现；γ≈0.92（生成权重），复制权重约 0.08 | `reports/taiji_memory_dominance_probe_20261104.json` → `pooled`、`verdict`、`cells[*].criterion` | 训练从未需要复制：15 种组合可背诵，输出在已背名字池内换，内容不来自记录 |
| C11 | 方案 A 的**数据前提已构建但未训练**：物件/地点名字池各 220（≥200），按**名字串**三向互斥切 160/30/30，train/dev 的（物件×地点）组合交集 0，八项跨向名字出现计数全 0；参数增量 0（名字是记录字节，不是嵌入），容器仍 3,563,859 参数 | `plans/manifests/taiji_data_premise_18_0_manifest_20261104.json` → `vocabulary`、`proofs`、`capacity`；`plans/manifests/taiji_data_premise_18_0_prereg_20261104.json` → `gates`、`axes` | 只证明“考题不再可背”，**不证明模型已能抄新名字**；能力结论要等 18.1 结果 |

## 2. 未证（还没有实验能支持，对外不得声称）

| # | 待证能力 | 现状读数 | 证据指针 | 一句话白话 |
|---|---|---|---|---|
| U1 | 选择—解码绑定能提升任务成绩 | 16.1 总体完成率 0.0500 / 0.0167 | `...16_1_compare...json` → `configs.*.completed_rate` | 模型挑得不准，写完更不准 |
| U2 | 新描述/新名字泛化 | N16 A新名字0/60，新描述生成3/60、1/60，选择新描述0/60 | reports/taiji_binding_select_16_1_attribution_20261103.json → gates/readings | 生成与选择两个端点分开报告 |
| U3 | 选择本身要准（教师条件下） | top-1 0.3833 / 0.3542，门 0.90 | `...16_1_compare...json` → `configs.*.selection_top1` | 给它答案线索，也只有三成多挑对 |
| U4 | 真正未见输出名字 | 14.6两seed新名字各0/40；原unseen_entity主要为新描述轴 | reports/taiji_post14_algorithm_review_20261102.json | 未见名字和新描述分轴，不复用含混字段 |
| U5 | 旧能力保持 | `retention_not_admissible`（当前没有合格父） | `...17_3_verify...json` → `t10_retention`；`INVARIANT-LEDGER` L9 | 还没资格谈“学新不忘旧” |
| U6 | 本候选跨session经验/长期学习 | experience_surface_absent描述当前17任务合同范围 | 17验证的t11_api；系统计划19–22 | 09及旧记忆有工程资产，当前共同任务还未完成该范围验证 |
| U7 | 17路由/信用的任务增益 | 17.5 已判负：overall 0.000–0.200（门 0.80）、selection_object 0.338–0.750（门 0.90）、A−C +0.017/−0.050 | `reports/taiji_selection_routing_17_5_compare_20261104.json` → `configs`、`gates` | 工程成本已发生，任务收益未出现 |
| U8 | 一般中文写作能力 | 14.6 完整动作 0.5844，门 0.80 | `...14_6...json` → `gates.task` | 能写对一半多，离可用还远 |
| U12 | 自由运行时把未见过的新名字抄进输出 | 0.000（新名字逐字抄回率，六格全 0） | `reports/taiji_memory_dominance_probe_20261104.json` → `pooled.*.copy_rescue` | 只有给死路线时（T3）才抄得出，正常运行时不会抄 |

| U9 | 回答代理信用的额外收益 | N16无该ST路径且A−B有目标处理混杂，17待测 | 16归因与17正式图 | 不能以N16负增量证否17的ST信用 |
| U10 | 位置辅助监督是否必要 | 旧P4控制/长度条件不足以裁决必要性 | ATTRIBUTION-ARCH-VS-ALG及系统计划§4 | 先分层定位，再立学习对照 |
| U11 | 正常pointer的自主定位/软硬泛化 | 工作树改变了读核，完整任务及新恢复/价未验 | 当前源码与17局部回执 | gold可达性和自主学得分开验收 |

## 3. 已证否（在已测条件下不成立，正文不得再宣称）

| # | 被否说法 | 实测 | 证据指针 | 白话解释 |
|---|---|---|---|---|
| N1 | “字节均值嵌入足以分辨同字不同序” | 旧头换序差 0.0 / 0.0 | `...17_3_verify...json` → `r1_calibration.old_head_witness`；16.1 P1 | 只取平均值，顺序信息被抹平了 |
| N2 | “锁定复制比不锁更好”（A−C ≥ 0.10） | 14.6 A−C +0.1063，但 bootstrap 下界 −0.0375 | `...14_6...json` → `gates.task.structure` | 点估计为正但统计上站不住 |
| N5 | “反序配对率可作任务门” | 16.1 原始字节不足，配对率撤回/不可判 | `SPEC-N16-01` §4 仪器缺陷；`...16_1_3_closeout...json` | 没存原始输出，配对重算不可信 |
| N6 | “六臂可塑性选择能转为后继净收益” | ΔNLL_holdout −0.1370，未优于对照 −0.1402；六臂全超保持容差 | `reports/taiji_core_r026_plasticity_arms_20261031.json` | 合成夹具上，出生/生长方向 8 步内没回本 |
| N7 | “出生方向能自我偿付” | 同上，按预注册判负 | `...r026...json` | 同上 |
| N8 | “自演进 / 终身学习 / 自我改进已经接通” | 从未落地；17 也没有该合同 | `INVARIANT-LEDGER` L13；`...17_3_verify...json` → `t11_api`、`applicability.T11` | 这是长期目标，不是现状 |

> N2/N6/N7 的“否”都限定在当时的夹具、剂量与预算；它们不证明原理在任何规模都不成立，但足以禁止把“已经证明有效”写进正文。

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

## 6. 系统计划采用范围（2026-11-03）

本轮撤回N3把16负增量当ST信用否证、N4把位置监督未证当证否、N9把旧前置锚点测试当最终Q架构失败的表述，分别纳入U9–U11。工作树新T3按C9限定范围记录。原报告与旧卡Git版本保留，当前算法/状态/价格/恢复由同候选manifest逐项核对。

后续[系统计划](PLAN-NATIVE-COGNITIVE_SYSTEM_DEVELOPMENT_20261103.md)覆盖17–22，远期能力均为目标和待验证责任。现阶段的值恢复、有序信号及gold控制结果不代签整合后的任务、持续经验或规模生长能力。

## 7. 数据前提修订三选项（2026-11-04，待 owner 裁定，不得当作已批准）

C10 把 17.5 的失败归因从“算子坏了”改成“任务前提允许背诵”。据此有三条互斥路径，裁定前 18–22 不启动训练：

| 选项 | 内容 | 成本量级 | 能力上限 | 局限 |
| --- | --- | --- | --- | --- |
| A | 扩大物件/地点词表，并按（物件×地点）组合做 held-out 划分 | 数据≈0；训练 6 格 × 300 更新 ≈17k CPU | 上限最高：唯一能证明“换新名字也能抄对”的泛化 | 选择头（现 0.75）可能成为新瓶颈 |
| B | 在物件/地点字位把 γ 压到 0，保留模板生成 | 工程改动约 1 天 + 同剂量 ≈17k CPU | 中：复制路径必然承重 | 仍可背词表，只证机制、不证泛化；可能伤模板 |
| C | A+B 同时做 | ≈20–22k CPU | 最高 | 双变量同时改，读数无法归因 |

建议**先 A**：它是根因（数据前提）的直接解，也是 B 能否被解释的前置条件（无 held-out 时 B 的增益与背诵不可区分）。C 仅在 A、B 各自读数清楚后才合并。

**18.0 落地（2026-11-04，仍待 owner 批准）**：A 的非训练部分已完成——数据前提（C11）、定价（六格裸价 9,109.4 / 保护价 19,220.8 CPU；step150 前段 3,088.9 / 6,517.5，后段 6,040.2 / 12,744.9）、判据预注册（10 门，其中 `new_name_copied ≥ 0.60` 为**独立门**，overall 过门但该门不达标一律标 `recitation_remains`；每格须 γ 与 1−γ 双报）。18.1 正式比较**未启动**，须 owner 明确批准 prereg 与预算。证据：`plans/manifests/taiji_data_premise_18_0_prereg_20261104.json`、`plans/manifests/taiji_data_premise_18_0_pricing_20261104.json`。

**18 号立项前提（钉住，不可回退）**：新字节串 + 大词表（各 ≥200）+ 组合 held-out。只满足其中一条的“扩词表”不算 A，读数不得用于泛化结论。

