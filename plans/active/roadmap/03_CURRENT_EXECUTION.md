# Seed / Taiji 模型优先统一开发计划

> 修订：2026-09-07。状态：R0 测量链完成；R1～R3 首轮能力报告因数据重叠被撤回，record-disjoint v2 数据链已修复；seed11/29/47 的 R1 formal aggregate 已完成但不晋级。R2.R0～R2.R1 时间表征候选已完成并否决 promotion；M2.R3.R0～R3.R4 的 structured semantic CPU canary、runtime owner/checkpoint、多实体/关系/约束多 seed 组合训练、多步持久 WorldState 转移及 runtime adapter 接线均已完成；M2.R4 联合课程与能力保持也已完成：事实、Goal、ContentPlan、未知/冲突/澄清、事件删除/顺序、持久性、owner lesion、旧 adapter 行为保持、新进程恢复、detach 清理、双 owner 组合和受保护能力保持全部通过。M3.R0 的 project-disjoint Workbench 只读边界 canary 已完成；它证明的是 provider-assisted 实时文件/语言证据、结构化 Goal/ContentPlan、checkpoint/restart 和 fail-closed 边界，不是 native raw-text learning 或自然语言流畅。下一步进入 M3.R1 native semantic observation canary，不改变 provider/native-only 证据边界。连续流切块、资源遥测和中断恢复均已通过。本文是唯一执行顺序与“下一步”来源。
>
> 本轮已完成数据契约代码、审计报告和回归测试；不会改写旧报告，下一步只从原始 child 重新生成独立证据。历史 M0/M1/M2-2a～2ae 的有效成果保留；旧“下一步”全部失效。重审依据见 [研究审视](../../reference/TAIJI_RESEARCH_REVIEW_2026_09_06.md)，原文见 [历史快照](../../archive/history/research_review_20260906/README.md)。
>
> **审计范围声明：** 本版不是只读 plans 得出的排程。它已对当前 `main` 的关键模型、数据、训练和 evaluator 代码、实际 JSON 报告与现存 checkpoint 做定向交叉核对，并对 active/protected 评分调用链和 B5 数据流做最小复现；但没有逐行审计仓库全部客户端、前端、CI 和历史模块。因此“当前证据”有代码或产物支撑，“后续候选/待验证”仍是计划假设，不能提前当作已实现能力。

## 1. 当前定位与本次调整

Taiji 要形成拥有持续状态、异质群体、可学习表征、记忆、世界模型、目标、行动和终身适应的原生认知架构。Seed 承载训练、设备、工作台、权限、客户端和发布。核心依据仍是 [CR-1～CR-10](../TAIJI_CORE_REQUIREMENTS.md)，完整目标仍是 [Native Architecture v1](../TAIJI_NATIVE_ARCHITECTURE_V1.md)。

目前已完成真实小规模训练，但仍是具有窄任务学习证据的研究原型。不能再笼统说“完全没训练”，也不能把它称为成熟语言大模型或自主进化系统。当前最重要的问题是：训练是否写到了正确的器官、评估是否读到了同一器官、对未见内容的能力是否提高。

本次收敛为以下决定：

1. **先修量尺，再立即训练。** active 评分错路由、旧 B5 课程重复和 Gate 语义混用是近期前置工作；不继续新增完整的外围治理系统。
2. **保留已有知识和谱系。** 从审计过的 child 继续；旧父模型用于因果对照。重初始化只允许出现在明确标注的对照实验。
3. **分开能力达标、能力保持、本轮收益。** 一个已达 100% 且继续保持的旧任务，不要求本轮再提高；新能力必须有独立证据。
4. **将读出隔离定位为受控适应基线。** 冻结旧分支、人工指定 active 不等于自主路由、知识融合或神经网络开放成长。
5. **训练表征和跨时间信用，而非只磨最后一层。** 先测当前规则的学习曲线，再依据证据选择一个可塑上下文/成熟递归组件；避免同时更换多个机制。
6. **保留模型目标，缩小每轮研究问题。** 近期只完成“可信测量 → 真实续训 → 表征/语义训练 → 联合保持”。功能、插件和动画数量不作为进展主指标。

## 2. 事实基线与待纠正声明

| 项目 | 当前证据 | 本轮认可的结论 |
|---|---|---|
| B1 字节预测 | 三 seed v4 child holdout BPB 约 4.288/4.384/4.373，优于约 5.94 的 unigram | 窄语料统计预测有学习收益；不等于可读语言、语义或推理 |
| B2 延迟记忆 | identity generation 的三 seed 最差 recall 0.99，retention 0.975 | 已写入关联的受干扰检索有效；不等于未见概念泛化 |
| B3 世界预测 | 当前主要是 position → position+1；误差约 3.62e-8 | 简单转移拟合有效；缺少多对象、变化规则和多步后果证据 |
| B4 行动 | 两个 cue 对应两个 action；成功率 1.0 | 简单奖励关联有效；缺少未见目标与计划组合 |
| B5 持续学习 | 旧 replay 可减轻遗忘，最差 BWT -0.262874 | 原设置中仍有遗忘；新课程“holdout”重复，泛化证据不足 |
| M2-2ad active | 权重改变、owner 隔离、保存恢复检查通过；评分却全走 protected | active 能力结论撤回，状态为 measurement-invalid，不能归因为重复训练或容量不足 |
| M2-2ae phase-C | 按完整记录排除 A/B，可提供 C 数据 | 整条记录不重复；尚未证明近重复隔离、领域变化或语义新颖性 |
| M2.R1 v2 seed11 pilot | 64 KiB C 训练、32 KiB 评估，五臂均完成；no-update/protected/active/replay/cascade 技术检查分别为 2/2、4/4、14/14、14/14、12/12 | 数据契约、owner 路由和 fresh checkpoint 均通过；active `-0.0594` BPB、replay `-0.0607`、cascade C′ `-0.0480`，当前预算下未见留出收益，不能晋级 |
| M2.R1 v2 seed11 formal active | 1 MiB C 训练、128 KiB 评估，16 chunks；14/14 技术检查通过，训练约 1,220 秒、评分约 403 秒、约 860 B/s、峰值工作集约 368 MB | C holdout protected `4.2194` → active `4.1950 BPB`，增益 `+0.02444`；三 seed 对照已完成，aggregate 仍不晋级 |
| M2.R1 v2 seed11 formal replay | 同一 untouched parent、同一 1 MiB C 预算、C→replay 两周期、32 chunks；14/14 技术检查通过，训练约 2,579 秒、评分约 414 秒、约 813 B/s、峰值工作集约 369 MB | C holdout protected `4.2194` → replay `4.2784 BPB`，增益 `-0.05893`；等预算 replay 未复现 active-only 正收益，不能晋级 |
| M2.R1 v2 seed11 formal cascade | 同一 untouched parent、1 MiB C→C′ 两周期、32 chunks；12/12 技术检查通过，训练约 2,519 秒、评分约 1,945 秒、约 832 B/s、峰值工作集约 369 MB | C′/C2 protected `4.2925` → active `4.2866 BPB`，新 holdout 增益 `+0.00592`；但 C 在 cycle1 `4.1950` → cycle2 `4.2132 BPB`，退化 `+0.01827`，第二周期保持失败，不能晋级 |
| M2.R1 v2 seed29 formal active | 1 MiB C 训练、128 KiB 评估，16 chunks；14/14 技术检查通过，训练约 1,020 秒、评分约 319 秒、约 1,028 B/s、峰值工作集约 370 MB | C holdout protected `4.3082` → active `4.2547 BPB`，增益 `+0.05344`；独立 seed 的 active-only 结果为正，但 aggregate 不晋级 |
| M2.R1 v2 seed29 formal replay | 同一 untouched parent、同一 1 MiB C 预算、C→replay 两周期、32 chunks；14/14 技术检查通过，训练约 2,047 秒、评分约 326 秒、约 1,025 B/s、峰值工作集约 370 MB | C holdout protected `4.3082` → replay `4.4004 BPB`，增益 `-0.09228`；A retention `4.4446 BPB`，等预算 replay 明显未复现 active-only 正收益 |
| M2.R1 v2 seed29 formal cascade | 同一 untouched parent、1 MiB C→C′ 两周期、32 chunks；12/12 技术检查通过，训练约 2,513 秒、评分约 1,933 秒、约 835 B/s、峰值工作集约 369 MB | C′/C2 protected `4.3551` → active `4.3292 BPB`，新 holdout 增益 `+0.02592`；但 C 在 cycle1 `4.2547` → cycle2 `4.2656 BPB`，退化 `+0.01088`，第二周期保持失败，不能晋级 |
| M2.R1 v2 seed47 formal active | 1 MiB C 训练、128 KiB 评估，16 chunks；14/14 技术检查通过，训练约 1,238 秒、评分约 394 秒、约 847 B/s、峰值工作集约 368 MB | C holdout protected `4.3199` → active `4.2447 BPB`，增益 `+0.07520`；第三个 seed 的 active-only 仍为正，但 aggregate 不晋级 |
| M2.R1 v2 seed47 formal replay | 同一 untouched parent、同一 1 MiB C 预算、C→replay 两周期、32 chunks；14/14 技术检查通过，训练约 2,479 秒、评分约 396 秒、约 846 B/s、峰值工作集约 369 MB | C holdout protected `4.3199` → replay `4.3083 BPB`，增益 `+0.01154`；但 A retention `4.3844`，相对 protected 退化约 `+0.06451 BPB` |
| M2.R1 v2 seed47 formal cascade | 同一 untouched parent、1 MiB C→C′ 两周期、32 chunks；12/12 技术检查通过，训练约 2,461 秒、评分约 1,910 秒、约 852 B/s、峰值工作集约 370 MB | C′/C2 protected `4.3779` → active `4.3281 BPB`，新 holdout 增益 `+0.04975`；但 C 在 cycle1 `4.2447` → cycle2 `4.2489 BPB`，退化 `+0.00417`，第二周期保持仍未通过 |
| M2.R1 v2 three-seed aggregate | 9 个正式臂全部 `status=passed`；active/replay 各 3×14/14，cascade 各 3×12/12；来源报告 SHA-256 已写入 aggregate | active-only 均值/最差 `+0.05102/+0.02444 BPB`（3/3 正）；replay `-0.04656/-0.09228`（1/3 正）；cascade C2 `+0.02720/+0.00592`（3/3 正），但 C cycle2 delta 均值 `+0.01110 BPB` 且 3/3 退化；测量链完成但不晋级 |
| M2.R2.R0 seed11 smoke | 固定 seed11 parent，frozen/active_readout 两臂、4/16 KiB、4 点；20/20 技术检查通过；曾捕获默认绝对 corpus path 导致 lineage digest 不一致，已修正为 child 生成时的相对 canonical path | checkpoint preflight digest 一致；active owner 只写 active slot，但 holdout gain 为 `-0.06255/-0.14109 BPB`；smoke 仅作执行链证据，不晋级 |
| M2.R2.R0 seed11 formal curve | 固定 seed11 parent，frozen/active_readout/predictive_context/joint_predictive 四臂、4/16/64 KiB、12 点；60/60 技术检查通过；preflight checkpoint `33,174,421` bytes，峰值工作集约 `535–569 MB` | protected holdout baseline `4.049390 BPB`；context gain 为 `+0.001515/+0.023738/-0.016105`，16 KiB 之外不稳定；active gain `-0.062549/-0.141091/-0.165254`，joint gain `-0.058237/-0.120990/-0.153693`；owner 写入集合与 read-only scoring 全部正确，当前不引入新架构 |
| M2.R2.R0 context aggregate | seed11 formal + seed29/47 context 复现共 24 点、120/120 技术检查通过，来源报告 SHA-256 已写入 aggregate | 4 KiB gain 均值 `+0.007681`（3/3 正）；16 KiB `+0.018081`（2/3 正）；64 KiB `-0.010650`（1/3 正）；结论是短/中预算有信号但长预算不成立，先做消融与参照，不做时间架构晋级 |
| M2.R2.R0.1 seed11 context/ordering ablation | 固定 seed11 parent，frozen/normal_context/shuffled_context/readout_only 四臂、4/16/64 KiB、12 点；60/60 技术检查通过；包含 temporal-residual clone lesion、unigram/2-gram/3-gram 参照 | normal context gain `+0.001515/+0.023738/-0.016105`；shuffle gain `-0.240627/-0.532733/-0.697068`；readout-only `-0.062549/-0.141091/-0.165254`；正常 residual lesion 后 holdout `4.718034 BPB`，说明有序输入和 residual 有因果参与，但尚未测直接依赖跨度，不扩架构 |
| M2.R2.R0.2 seed11 delay probe v2 | 受控 record-disjoint copy 任务，distance `1/8/32/128/512`，frozen/joint/shuffled/readout-only 四臂，3 epochs、32/16/16 records；20 点、100/100 技术检查通过；v1 低剂量报告已删除，不作为证据 | joint test accuracy：`0.938/0.062/0/0/0`；readout-only：`0.938/0.062/0/0/0`；frozen 全 0；shuffled joint：`0.625/0/0/0/0`；joint temporal lesion 与 joint 相同，说明当前 candidate 在此 copy probe 上没有被证明提供超出 readout 的长程 credit，当前不宣称 32～512 距离能力 |
| M2.R2.R1 gated temporal candidate probe | 在旧 checkpoint 上启用零初始化 fast/slow gated residual；smoke 8 点/40 检查、full 20 点/100 检查均通过；candidate owner、checkpoint round-trip、lesion 与 read-only scoring 全部闭合；报告 `taiji_m2r2_r1_candidate_smoke_seed11_20260907.json` / `taiji_m2r2_r1_candidate_probe_seed11_20260907.json` | gated_joint 在 distance=1/8 与 joint/readout-only 同为约 `0.938/0.062`，distance≥32 均为 `0`；full probe 没有形成额外长程能力，且部分 BPB 更差；`can_promote=false`。实现保留为默认关闭的迁移/反事实资产，不作为当前主架构 |
| M2.R3.R0 structured semantic CPU canary | 新增 `taiji/semantic_training.py` 与 `eval_taiji_m2r3_r0_structured_semantics.py`；8/4/4 record-disjoint train/dev/test，122 个参数标量；15/15 Gate 检查通过；报告 `taiji_m2r3_r0_structured_semantics_20260907.json` | train/dev/test fact F1、Goal accuracy、ContentPlan accuracy 均 `1.0`；未知输入 `unknown`、互斥事实 `conflict`、blocked 目标 `clarify`；checkpoint 输出一致，fact-owner lesion 使 test F1/Goal accuracy 降为 `0`；`can_promote=false`，不宣称自然语言能力 |
| M2.R3.R1 runtime semantic owner | `TSKV8Adapter` 增加显式 attach/detach、read-only inference、独立 `structured_semantic` native checkpoint component 与恢复；8/8 Gate 检查通过；报告 `taiji_m2r3_r1_runtime_owner_20260907.json` | 未 attach 的旧 adapter 不产生新 component；推理不改变 `CognitiveState`、不产生 TaskInterpretation/ActionIntent；新进程恢复 learner 与上次结果，detach 后无残留；`can_promote=false` |
| M2.R3.R2 multi-seed semantic composition | 3 个独立 feature-layout seed；每个 36/10/8 record-disjoint split，11 个关系事实、3 个 Goal、18 个 ContentPlan、482 个参数标量；每个 seed 14/14 检查通过，报告 `taiji_m2r3_r2_multiseed_20260907.json` | 三 seed test fact F1/Goal/ContentPlan accuracy 均 `1.0`；未知、互斥 constraint、blocked 澄清、pending plan、checkpoint 和 fact-owner lesion 全通过；`can_promote=false`，仍不宣称文本语言能力 |
| M2.R3.R3 multi-step semantic transition | 3 个独立 feature-layout seed；每个 32/8/8 record-disjoint transition split，7 个事实、3 个 Goal、6 个 ContentPlan；transition v2 使用通用 fact×event interaction basis，594 个参数标量；每个 seed 14/14 检查通过，报告 `taiji_m2r3_r3_multistep_20260907.json` | 三 seed train/dev/test fact F1、Goal、ContentPlan 均 `1.0`；`release → move → block` 持久序列、事件删除/顺序、mid-sequence checkpoint、unknown/conflict fail-closed、transition lesion 与三个 owner 变化全部通过；`can_promote=false`，证明的是受控结构化多步状态转移，不是开放域世界模型 |
| M2.R3.R4 runtime transition owner | 复用 R3.R3 三个 seed 的 transition v2 learner；通过 `TSKV8Adapter` 显式 attach/detach、read-only inference、独立 `structured_semantic_transition` native component 和新进程恢复；每个 seed 10/10 runtime checks 通过，报告 `taiji_m2r3_r4_runtime_transition_owner_20260907.json` | 三 seed 的组件可选性、三步序列、checkpoint round-trip/source digest、CognitiveState 只读、无执行副作用、transition lesion、detach 清理和旧 adapter 无组件全部通过；`can_promote=false`，只证明 runtime 所有权和恢复边界 |
| M2.R4 joint course and retention | 三个 seed；静态 state-encoding 24/12/12、transition 32/8/8 record-disjoint；frozen-parent/static-only/transition-only/joint-native 四条控制路径；每个控制 13/13 检查通过，报告 `taiji_m2r4_joint_course_20260907.json` | joint-native 静态 test fact F1=`1.0`、transition test fact F1=`1.0`、三步组合=`3/3`；双 owner 同时写入、native 双组件 checkpoint 恢复、CognitiveState 只读和旧能力保持全部通过；`can_promote=false`，这是独立 owner 的联合课程/组合证据，不是共享权重联合优化 |
| M3.R0 Workbench read-only boundary | 三个互不重叠的微型项目族（Python/TypeScript/缺 toolchain Rust）；实时 `workspace.read` + `workspace.programming_language.resolve`，另含 `.h` 歧义、无效路径、stale snapshot、断开 diagnostics 控制；13/13 Gate 指标通过 | Goal/ContentPlan 存在，provider 未提交最终语言 id，语言/文件 digest 来自当前 Workbench；只执行两个 read-only capability；三项目 checkpoint/restart 恢复；歧义/无效目标在 ActionIntent 前澄清；语言 registry 随 checkpoint 恢复；`can_promote=false`，provider-assisted 边界证据，不等于 native raw-text learning |
| M2.R1～R3 首轮报告 | 仅用不同 `partition_seed`，A/C 交集 `1597`、A/C′ `1577`、C/C′ `1563` | 技术 owner/保存检查仍可留作诊断；所有 phase-C 能力与多周期结论撤回，不能聚合或晋级 |
| M2-2af 草案 | 继承评分错误、训练后才保存、仅 fresh digest、可单 seed promote | 中止且无正式报告；退出可执行主线，保留归档供重构参考 |
| 完整认知层 | joint runner 使用 Taiji；Seed runtime 使用包含更多器官的 TSKV8Adapter | 不能把 kernel child 的成绩归给所有 adapter 器官，需逐 owner 训练覆盖映射 |

计量也必须绑定具体 checkpoint：本轮只读检查的 seed11 sequence child 有 **193,586** 个 parameter_tensors 标量、**774,344** 字节参数张量，joint 文件 **18,771,087** 字节；后续 identity-growth child 为 **476,978** 个标量、**1,907,912** 字节参数张量，joint 文件 **42,908,047** 字节。这不是全部 runtime 状态大小，也不是每步更新量，更不是与同能力其他模型相比的硬件优势。

## 3. 唯一执行顺序

阶段名 M0～M8 保留，近期只使用 M2.R0～R4 拆解本轮研究；不再递增 M2-2ag/ah 等一次性实验编号。

| 顺序 | 阶段 | 状态与交付 | 退出条件 |
|---|---|---|---|
| 0 | M0 / M1 | 历史基线与首轮训练已执行 | 历史证据按任务范围保留，不重跑整段流程 |
| 1 | M2.R0 测量纠正 | 已完成；评分 owner、数据排除基础、Gate/谱系合同修正 | 正反例能识别错 owner、重复、回归和无效恢复 |
| 2 | M2.R1～R3 证据重建 | **已完成本轮 R1 证据**；三 seed active/replay/cascade 全部正式臂和 aggregate 均完成，active 分支有受控留出收益但 replay/第二周期保持不稳定，不能晋级 | 技术合格报告 + 独立 C/C′ 能力结果，不要求实验一定成功 |
| 3 | M2.R2 表征与时间学习 | **R2.R0～R2.R1 已完成；gated multi-timescale temporal candidate 技术闭环通过但 promotion 否决**，候选默认关闭并保留为可回滚实验资产，不继续调参 | 已通过旧输出保持、owner、保存恢复和移除/冻结反事实；未通过“同一 delay probe 上形成额外长程能力”，不替换默认结构 |
| 4 | M2.R3 语义与表达训练 | **R3.R0～R3.R4 已完成**：结构化语义训练合同、可选 runtime owner/checkpoint、多实体/关系/约束多 seed canary、多步事件到持久 WorldState/Goal satisfaction、runtime adapter 接线已闭合；当前进入 M2.R4 联合课程与能力保持 | runtime 输入只来自当前 `PerceptEvent`；事实/Goal/ContentPlan/WorldState 结果可审计；provider 表达收益与 native-only 分开统计 |
| 5 | M2.R4 联合课程与保持 | **已完成**：frozen/static-only/transition-only/joint-native 对照、双 owner 组合、runtime 双组件 checkpoint 和 protected retention 已闭合；不把独立 owner 的串行课程称为共享权重联合优化 | 三 seed 正式报告；joint-native 新组合通过且两个受保护 owner 均保持 |
| 6 | M3 最小真实任务验证 | **M3.R0 已完成；下一步 M3.R1**：R0 已在 project-disjoint 微型项目上闭合只读、语言证据、失效澄清、断开 diagnostics 和 restart；R1 必须把当前 Workbench observation 接入 Taiji native semantic owner，再决定是否开放 preview/approval 写入 | R1 需在未见项目上形成 native owner 的可复现实证，并与 provider-assisted、static-only、空计划对照；模型、客户端、权限、Outcome 同一 runtime 可追溯 |
| 7 | M4 连续成长 | 多轮续训、巩固、必要时结构增长与压缩 | 优于固定容量和等预算对照，且保持/成本/回滚达标 |
| 8 | M5 知识与身体 | Skill/MCP 数据内化、真实调用与客户端插件 | 认知与执行收益可分别归因，权限/撤销闭合 |
| 9 | M6 产品收口 | provider 稳定性、UI/桌面、遗留格式清理 | packaged client 与真实能力一致 |
| 横向 | M7 工程质量 | 每轮相关检查，阶段末全矩阵，发布时集中核验 | 无新增 CI 退化；正式发布绑定代码/数据/模型/包 |
| 条件阶段 | M8 CUDA | 本机硬件条件未变，继续搁置设备实测 | 有真实硬件后测量等质量性能，不阻塞前述 CPU 工作 |

研发日是估算，不是自动执行承诺：R0 2～3 日、R1 2～4 日加训练、R2 5～10 日加训练、R3 5～10 日加训练、R4 3～5 日；M3 约 1～2 周，M4 约 2～4 周起步，M5/M6 各约 1～2 周。实际训练时长先由吞吐探针估计，不能由“小参数量”推定。

## 4. 当前唯一下一步

**唯一下一步：进入 M3.R1 native semantic observation canary。** M3.R0 已经用三个互不重叠的真实微型项目族闭合了 Workbench 只读边界：模型/Provider 只提交 `read` 与 `resolve-language` 语义，当前文件内容、语言 id、file digest、toolchain 可用性和 capability 状态由实时 Workbench 产生；`.h` 歧义、无效路径、stale snapshot、断开 diagnostics 都在执行前或权限边界上 fail-closed；语言 registry、选择和 runtime 状态可重启恢复。下一步不开放 patch、terminal、MCP，也不把 R0 的 provider-assisted 结果算作 native 学习。建立一个版本化 `WorkbenchObservation -> PerceptEvent` 适配层：输入当前已验证的文件/语言/工具链证据与用户目标，输出 Taiji native semantic owner 可消费的 observation；先用已有 `StructuredSemanticLearner`/transition owner 的 checkpoint 合同做保存恢复预检，再在 project-disjoint 未见项目上比较 native owner、provider-assisted、static-only、空计划四臂。验收事实支持率、Goal/ContentPlan 正确率、歧义/缺 toolchain/stale snapshot 澄清率、provider 是否仍不能提交最终绑定、owner lesion、checkpoint/restart、CPU 成本和工作区零副作用。只有 native owner 在未见项目上形成可复现净收益且受保护能力保持，才进入 M3.R2 preview/approval；R1 aggregate、R2.R0 aggregate、R2.R0.1、R2.R0.2、R2.R1、R3.R0、R3.R1、R3.R2、R3.R3、R3.R4、M2.R4 与 M3.R0 报告均保留为基线。

2026-09-06 实际审计已证明首轮报告不能作为能力证据：旧 evaluator 的 C/C′ 只是换 partition seed，不是新记录。当前已落地的修复为 `scripts/training/eval_taiji_m2r1_phase_c_canary.py` v2、`scripts/training/audit_taiji_m2r1_data_contract.py` 和 `reports/taiji_m2r1_data_contract_20260906.json`：

- 对 seed `11/29/47` 分别重建与 checkpoint 对齐的 phase-A protected 和 phase-B active；A/B digest 与现有 child lineage 逐一相等。
- 共同 phase-C 排除六份 A/B 记录并集，phase-C′ 再排除 phase-C；相对路径 canonical corpus 下 C digest=`ac97455f…6476bd`、C′ digest=`13dee7b2f…3598c1`，选中记录数 `1567/1555`，所有 lineage↔C/C′ 与 C↔C′ overlap 为 `0`。
- v2 aggregator 已取消硬编码旧 C digest，并拒绝缺少 `source_lineage`、`record_disjoint` 或旧 v1 格式的报告；旧 JSON 保留为历史失效证据，不原地改绿。
- 首轮数值暂不保留为“第一周期增益上限”或“第二周期失效”；重跑前只能说它们是测量管线通过、数据契约不成立的诊断结果。重跑仍必须 `can_promote=false`，即使出现正增益也需先完成预注册的三 seed 判定。

## 5. 近期实施规格

### M2.R0：恢复可信的测量链

**状态：已完成（2026-09-06）。** 交付物：`taiji/measurement_verdict.py`（版本化三类判定）、`scripts/training/inventory_taiji_checkpoints.py` + `reports/taiji_m2_r0_checkpoint_inventory_20260905.json`、`scripts/training/reeval_taiji_r0_verdicts.py` + `reports/taiji_m2_r0_reevaluation_20260905.json`、m2ad loader 硬编码修复、R0.2-R0.5 回归测试。

**研究问题：** 当前每个分数测了哪个 checkpoint、哪个 owner、什么数据、什么能力？

**改动范围：**

- 核心：`taiji/model.py` 的 `score_bytes` 与 readout 选择；`taiji/workbench_boundary.py` 的现有边界消费。
- 评估：`taiji/foundation_evaluation.py`、`scripts/training/eval_taiji_foundation_baseline.py`、M2-2ad helper。
- 数据：`taiji/foundation_training.py`、旧 B5 stream 构造器；重用现有 manifest。
- 回归：workbench boundary、foundation child evaluation/training 的相关测试，避免再建一套通用评测框架。

**按依赖完成：**

1. 为评分增加显式 owner/scope 路径，默认 protected 兼容；受控 active 与生成使用同一校验规则。评分返回或可审计记录 effective owner、readout digest、boundary digest。闭合 checkpoint 恢复后的评分。
2. 回归必须让 active 和 protected 的输出分布显著不同：分别打分应产生不同结果；仅改 active 不应改 protected 分数。缺 active、过期 boundary、foreign registry boundary、scope 不匹配必须明确拒绝。校验 registry 挂载的 boundary 与本次消费一致，不能只验证传入 token 自洽。
3. 验证评分完成及异常时全部模型状态/RNG/学习计数恢复。记录真实 owner 调用计数，不能用 `holdout_updates=0` 替代量尺正确性。
4. 将旧 B5 周期流降为遗忘压力测试；另建不同规则/组合的泛化分区。检测整段、足够长 n-gram 和记录/模板族重复；已写入关联的 B2 recall 允许查询训练中存入的记忆，但必须标注其记忆而非泛化语义。
5. 新增版本化的绝对能力、保持、增量三类判定，保留旧 JSON 和旧 manifest 的历史语义；不得原地修改旧报告使其变绿。
6. 建立一个简短的 checkpoint/owner 清单：直接父代、认证祖先、训练相位、数据地址、器官训练覆盖、文件是否存在、模型格式。确认 memory-growth child 不因 loader 硬编码 `training_phases=[sequence]` 被排除；保持严格的 schema/digest/owner 校验。
7. 对现有 child 先进行只读复评；此步骤可以重新解释“保持但没新增收益”，不能据此宣布旧 B5 通过。

**验收与产物：**

- 一个版本化 measurement contract 和一个重评报告，包含故意错 owner、错数据、真回归、能力饱和不回归四类测试。
- 清楚列出 M2-2ad 的失效范围：隔离/序列化检查保留，active gain/retention 的能力比较无效。
- 调整后的总 Gate 不要求旧任务每轮创造增益；也不允许所有分支冻结就被算作学习成功。
- 若发现更多不影响本实验的外围问题，记入后续阶段，R0 不无限扩张。当前训练 owner 的错误则必须先修。

### M2.R1：可恢复的真实 phase-C 续训

**目的：** 测出已有模型在新记录上的真实学习收益、遗忘和 CPU 成本，结束围绕无效零增益的推测。

**谱系选择：** 先选保留已确认 B2 增长成果的 seed11 identity-generation child；旧 sequence child 只作为配对控制。若 loader/迁移暂不能保持对应 owner，不静默回到较老父代，先完成 R0 的兼容性修正。随后才扩展 seed29/47。

**数据与实验臂：**

- v2 采用 cohort 级记录链：每个 seed 的 A=`partition_seed=seed`、B=`10000+seed` 且 B 排除 A；共同 C 排除所有六份 A/B 的完整记录，共同 C′ 再排除 C。记录文本抽取、规范化、全记录排除、截断边界和 digest/overlap 必须写入报告；同源 C 只能称为未见记录，不能称为新领域。
- 同一父代复制四个顺序执行的臂：不更新、单分支可塑、exact train-only replay、protected+active readout。使用同一数据和评估切片。
- 4 KiB train / 1 KiB evaluation 技术 canary 已通过；seed11 的 64 KiB train / 32 KiB evaluation 五臂 CPU pilot 也已通过全部技术 Gate，但 active/replay/cascade 的 C/C′ 留出增益均为负，不能当作能力晋级。
- v2 evaluator 已记录训练/评分/保存/恢复耗时、吞吐、峰值工作集、torch CPU 线程数，并按连续字节流边界保存可续接游标；4 KiB active chunked canary、`--resume` canary 和 seed11/29/47 的 1 MiB active/replay/cascade 正式臂均通过技术 Gate。三 seed aggregate 已固定为 R2 基线；后续缩减运行必须显式报告，不冒充 full foundation。
- active-only、protected-only 是器官能力测量；显式 boundary 组合是已知任务边界下的系统测量。无任务边界的自主路由单列未评估。

**训练执行：**

1. 在目标输出目录先保存初态，关闭/全新子进程加载并比对 digest、下一步输出及 owner；再做一次小更新、保存、恢复和续步一致性验证。失败不进入训练。
2. 为每个 run 创建独立目录和配置摘要；复用可恢复 trainer 的 chunk/epoch 游标、RNG、局部状态与 checkpoint 间隔。中断可从 last 恢复。
3. 保留 parent、last、final 和报告关联；验收后的 active checkpoint 是主要产物，不得只保存探针再删除。
4. 训练后的 fresh process 不仅核对 digest，还重跑 A retention / B holdout / C holdout 的指定 owner 分数和固定输出 canary。
5. phase-C 训练期间只准预声明 owner 更新；持久 owner 不变和能力保持是两种证据，分别报告。

**指标与结果解释：**

- BPB、accuracy、每阶段训练曲线、A/B/C 配对变化、参数/状态/磁盘字节、byte/s、峰值进程内存、训练/评分/保存/恢复时间。
- replay 与 no-replay 同时报新增字节量；既做等样本对照，也做等总更新/墙钟预算对照，不能把多训练的收益直接归因于 replay。
- 单 seed 只产生 candidate，`can_promote=false`。任何正增益必须在预注册的验证条件和后续多 seed 确认，浮点正数不自动算有效收益。
- R1 即使得到可靠负结果也允许转入 R2；不能不断更换 C/D/E 数据种子追求通过。仍存在测量错误时回 R0。

### M2.R2：把研究推进到表征、时间跨度与学习效率

**研究问题：** 当前上下文是否能表达任务、训练规则是否能学到这些信息，而非仅仅改变 bias 或最近字符统计？

**第一轮固定当前结构，建立学习曲线：**

- 数据分预算至少三个点，保持独立评估集；模型 scale、学习率与数据规模每次只变一项。
- **量尺修正：** 训练字节预算曲线只能说明样本量/过拟合行为，不能直接当作上下文跨度。R2.R0.1 已证明顺序和 residual 有作用；R2.R0.2 必须用已知延迟距离的合成/受控任务直接测信用跨度，再决定时间架构。
- 对照至少包含 byte unigram、经 dev 选择阶数的 n-gram，以及一个相近参数量的小型门控循环模型。循环模型是训练能力与成本参照，不接管 Seed 的认知。
- 同时报等参数、等训练数据和等 CPU 时间的可比结果；不同 tokenizer 的模型按原始字节或同一语义任务衡量，不能比较不兼容的 token loss。
- 检验上下文长度 32/128/512 的依赖任务、顺序打乱、上下文清空、固定随机表征、只训 readout 等消融。长度可由资源预检缩减并注明，不能改变题目后沿用成绩。
- 用组合族/文档源/事件规则隔离 test，而非只改变样本 ID；确认证据写入后查询和学习新规则是不同任务。

**训练对象与上限改进：**

- 当前优先对 `BytePredictiveContext`、`BytePredictiveReadout` 的可塑上下文和信用跨度做诊断；绘出 train/dev 曲线，检验训练本身欠拟合还是只泛化失败。
- R2.R0.2 已把“训练预算”与“信用跨度”分开：当前 joint/readout-only 在 distance=1 可学、distance=8 仅少量、distance≥32 未学会；因此后续候选必须在距离 probe 上验收，不能再用大字节量 BPB 代替长程证据。
- 若继续缺少时序表达能力，优先增加一个 Taiji-owned gated multi-timescale temporal residual/小型递归状态候选，零初始残差或保持函数的迁移尽量保留既有输出；不同时扩 memory、router 和区域数。
- 发展训练允许评估成熟 autograd/optimizer、截断跨时间信用等实现；在线局部更新仍独立审计。当前代码的 AST 禁令保持到迁移方案通过评审，不能为比较实验静默删除它。
- 手写梯度与 autograd 可数学等价；“手写”本身不证明更原生或更高效。候选须有梯度对照、checkpoint 转换和移除/冻结后行为证据。
- CPU 首轮使用可运行的递归参考；SSM/Mamba 是后续候选来源，不在硬件不可用时强装依赖或宣称获得其性能。

**出口与决策点：**

- 若现有结构随数据量增长仍持续改善，延长训练；若只在训练集改善，先改课程/正则/切分；若训练集也不改善且参照可学，进入一次有证据的更新规则/表示选择。
- 同一假设两轮无净收益后停止微调该路线，提交曲线、消融和成本比较，再决定迁移；不追加多个命名模块掩盖瓶颈。
- 只有计划中的候选验证；默认架构替换、学习约束修改和不兼容 checkpoint 迁移在该节点单独评审。此阶段交付一个实际训练后的候选 checkpoint，而非又一层空接口。

### M2.R3：语义、目标与表达的受控训练

**研究问题：** Taiji 能否从输入形成可验证内部内容，并通过语言或结构化行动表达？

**课程按依赖递进：**

**R3.R0 已完成（2026-09-07）：** 新增 `taiji/semantic_training.py` 的三段 native learner。事实 head 使用 `PerceptEvent.features + confidence/boundary metadata` 预测 relation facts 并物化 `WorldState`；Goal head 只消费事实概率；ContentPlan head 只消费事实与 Goal 概率。训练使用 detached local-delta，checkpoint 包含 corpus source digest、三个 owner 的参数和训练步数。CPU canary 使用 8/4/4 record-disjoint splits，并覆盖未知、互斥事实冲突、澄清计划、owner lesion 与恢复一致性；这一步不输入文本，不加载 Qwen/provider，不执行工具。

**R3.R3 已完成（2026-09-07）：** 新增独立 `StructuredSemanticTransitionLearner`，输入为当前 `WorldState` 与当前 `PerceptEvent`，先学习事实 delta，再物化下一时刻 `WorldState`，最后读取 Goal 与 ContentPlan；transition 输入使用通用的当前事实×事件上下文交互基，以表达状态条件事件，而不是把 `release/move/block` 写成模型外规则。三 seed 各使用 32/8/8 record-disjoint episodes，训练/验证/测试事实、Goal、ContentPlan 均为 `1.0`；事件删除、事件顺序、三步持久性、mid-sequence checkpoint、unknown/conflict fail-closed 与 transition owner lesion 全部通过。由于 checkpoint 输入形状发生变化，transition schema 升为 v2；该结果仍是 native-only 结构化 CPU canary，`can_promote=false`，不等价于自然语言或开放域世界模型。

**M2.R4 已完成（2026-09-07）：** 以 transition v2 的 7 个事实目录为统一 vocabulary，构造静态 state-encoding 24/12/12 split（每个 state/resource 组合有不同 metadata variant）并与 32/8/8 transition split 组成一门联合课程；frozen-parent、static-only、transition-only、joint-native 四个控制路径分别验证写入范围和负对照，joint-native 再通过 `TSKV8Adapter` 同时挂载两个 owner，执行静态初态感知与三步持久转移。三 seed 的 joint static/transition test fact F1 均为 `1.0`，联合序列 `3/3`，双组件 checkpoint、CognitiveState 只读、旧能力保持和无执行副作用全部通过；这证明的是模块组合，不是共享权重优化、自然语言表达或真实工具成功。

1. 中文短指令、事实提取、实体/关系与约束：同义改写和模板族严格跨 split，包含歧义、未知和冲突。
2. 上下文事件 → 持续世界状态 → 目标满足条件；用延迟查询和事实变更检验记忆更新，加入拒答/澄清。
3. Goal/WorldState → 有依据的 ContentPlan；检验字段事实覆盖、无根据内容、约束违背、不同表述下语义一致。
4. ContentPlan → 可读表达；先以现有 native-readable 确保结构可解释，再接训练过的 native 输出或受控语言器官。模板可读与模型语言能力分别记录。
5. 使用已存在的 Workbench 离线 trace 做计划数据，先训练选择和结果预测，副作用执行留给 M3。

**接线要求：**

- 输出逐 owner 的训练覆盖表：kernel private context、adapter perception/world/executive/content 等哪些被训练、冻结或未接线。
- kernel checkpoint 导入 runtime 时，必须保存/恢复对应 adapter 器官参数，禁止随机初始化语义器官后对外声称加载了完整能力。
- loss/局部目标包括序列预测、事件预测、目标满足、计划事实约束；不预设固定比例，每个目标有独立 dev 指标和消融。
- Skill/MCP 的本地说明、schema、示例可作为这一阶段的有来源语料；来源清洗和模型学习先做，真实第三方连接继续留到 M5。

**teacher/provider 边界：**

- 教师只离线提供带来源的标签/解释/候选，数据经验证；不把教师输出当未经核对的事实。
- 独立区分 native-only、teacher-assisted training、runtime expression provider、runtime semantic evidence。后两种表现不得计入 native-only 成绩。
- 训练完成后关闭教师复测目标/事实/计划；表达器可替换或损伤，内部内容应保持可审计。
- 没有真实 provider checkpoint 时照常推进核心结构化训练；不为了“嘴巴”停下模型训练，也不声称语言已自然流畅。

**验收：** 未见模板上的目标/约束准确率、事实支持率、无依据内容率、澄清校准、持续状态一致性、UTF-8 有效率和可读性各自达标；阈值使用 train/dev 确立后冻结，再用最终 test 验收。至少有一个超出字符统计与固定模板的模型净收益。

### M2.R4：统一多器官课程与能力保持

- 使用通过验证的最新 child 串行续训，定义每个阶段可写 owner、数据游标、更新次数和目标能力。
- 以冻结父代、固定容量续训、train-only replay 为对照；branch 方案必须同时评估单分支能力和实际部署路由。
- 构造跨阶段能力矩阵：阶段完成后测全部旧任务和当前新任务。BWT、旧任务最差变化、新任务收益及成本分别呈现。
- B3 引入不同动作效果/多对象/噪声和多步预测；B4 引入未见目标、变化动作映射和多步成功，避免天花板任务持续占据训练预算。
- 绝对能力对固定基准与认证祖先，保持对直接父代，本轮增量对预注册目标；不得因换了 parent 让“已会的能力”自动变成不具备。
- R1 已按固定协议完成 seed11/29/47；aggregate 已报告每 seed 成对差值、均值、最差值和技术 Gate，但尚未做按记录/任务重采样的不确定性。R2 必须把这些不确定性与 owner/上下文消融分开，不能只交叉比较最强控制与最弱模型。
- M2 完成条件是至少一项目标有可信新收益且全部受保护项保持，不是五个数字每一轮都提高。正式通用能力声明仍等待更广泛评估。

## 6. 公共评测与防误判规则

| 维度 | 衡量什么 | 不能替代它的证据 |
|---|---|---|
| technical validity | owner 正确、无泄漏、可恢复、异常也不污染 | 测试数量、digest 存在 |
| absolute capability | 在定义清楚的任务上达到阈值并胜过相应基线 | 仅相对随机初态改善 |
| retention | 同一旧任务在直接父代/child 的配对不退化 | 旧权重文件没变化 |
| incremental gain | 本轮指定新目标的有效提升 | 每项旧能力都必须提高 |
| generalization | 未见模板/规则/项目/依赖长度上的表现 | 新 ID、同源换 seed、同周期平移 |
| resource efficiency | 等质量/等预算下的吞吐、内存、状态和训练成本 | 参数少、文件小或稀疏名称 |

- 所有误差方向与计量单位固定；BPB 越低越好、成功率越高越好，BWT 采用有符号定义并明确分母。
- 保持容差按任务数值噪声与业务可接受退化预注册。旧 0.05 BPB 仅为历史实验容差，不能在看到新 test 结果后调整。
- 开发验证、最终 test、旧能力 retention 分开；test 不参与路由阈值/超参选择，多次实验消耗后的 test 必须更换为新保留集。
- 无标签路由不得读取当前 target、未来 token 或 evaluator 提供的最佳分支。任务边界可作为现实输入，但必须标注 task-aware；它不能证明 task-free 自主适应。
- 结构成长验收必须有固定大模型、固定小模型、单纯续训/记忆/路由和结构候选对照；扩容本身不计成功。
- 不在普通 CI 跑长期能力训练；CI 检查测量合同和小型因果反例，正式训练报告绑定代码/数据/模型版本后独立验收。

## 7. M3：最小真实 Workbench 任务

### M3.R0：project-disjoint 只读边界（已完成，2026-09-07）

交付物：`scripts/training/eval_taiji_m3_workbench_readonly.py`、`tests/taiji_native/test_m3_workbench_readonly.py`、`reports/taiji_m3_workbench_readonly_20260907.json` 与 `tests/fixtures/m3_workbench_projects/`。三个互不重叠项目族覆盖 Python、TypeScript、缺失 toolchain 的 Rust；控制样例覆盖共享 `.h` 歧义、无效目标、陈旧 capability snapshot 和没有真实编辑器连接时的 `editor.diagnostics.read`。13/13 Gate 指标通过：结构化 Goal/ContentPlan 存在，provider 没有提交最终语言 id，文件/语言 digest 来自 live Workbench，只执行 `workspace.read` 与 `workspace.programming_language.resolve`，三项目 checkpoint/restart 恢复，歧义/无效目标在 ActionIntent 前澄清，工作区保持不变。

本轮同时修复三个实际边界缺口：自然语言 grounding 在规划前验证 read/resolve 的 live target 和语言置信状态；`ProgrammingLanguageRegistry` 以完整 revision payload 写入并恢复 runtime checkpoint；有明确扩展名证据时，toolchain 可用性不再跨过扩展名冲突（`.ts` 不因 node 存在而误判成 JavaScript）。`editor.diagnostics.read` 仍保持 disabled，这是无真实编辑器连接时的正确 fail-closed 结果。M3.R0 是 provider-assisted native Workbench boundary evidence，`can_promote=false`；不证明 raw-text native semantic learning、开放域语言能力或可执行副作用。

### M3.R1：native semantic observation（唯一下一步）

R0 只证明语义 provider 能把当前 Workbench 证据安全送进 runtime，不能把 provider 的结构化步骤当成 Taiji 已经从 IDE 快照自行学习。R1 先定义版本化 `WorkbenchObservation -> PerceptEvent` 适配层：仅消费已经通过 Workbench boundary 的文件 digest、语言 assessment、toolchain availability、目标摘要、snapshot/boundary lineage 和明确的未知/冲突状态；不消费 provider 最终 language binding，不生成 patch/terminal/MCP 参数。使用现有 `StructuredSemanticLearner` 与 transition owner 的 checkpoint contract 做 fresh-save/restore/next-output preflight，再做 project-disjoint 未见项目的 native owner 训练或续训。

R1 必须并列 provider-assisted、native observation、static-only 和空计划四臂；指标为事实支持率、Goal/ContentPlan 正确率、ambiguity/unknown/缺 toolchain/stale snapshot 澄清率、provider final-binding absence、owner lesion、checkpoint/restart、CPU 时间/内存和工作区零副作用。三 seed 或等价 project split 前不开放 preview/approval；native owner 没有相对 static-only 的留出净收益时停止该适配层，不以增加 head、手动项目标签或每项目新表替代学习证据。

**解冻条件：** M2 结构化目标/计划可信且旧能力保持通过。提前可使用现有 Workbench 只读轨迹作训练数据，不能扩建完整客户端来替代能力验证。

- 先选一个任务族：在留出的微型项目中定位文件、判断语言、解释诊断并提出可逆补丁；依次验收只读、语言切换、preview/approval 写入、运行测试、失败恢复。
- Taiji 形成目标/内容/意图；Seed 只执行能力、权限和事务；provider 不返回未审计最终命令。
- 数据按项目划分，覆盖扩展名与内容冲突、用户指定语言、缺 toolchain 和陈旧快照。
- 模型接通客户端后，真实按钮、任务进度、语言选择和 Outcome 必须来自同一 runtime 状态。
- 成功率、无效工具调用、恢复率、审批遵从、重启续接和耗时共同验收；人写好的 intent 只算执行基础设施对照。
- 通过后再考虑扩大任务族，不同时启动浏览器、外部 MCP 和插件市场。

## 8. M4：在原有知识上成长的研究日程

目标是持续利用已有参数与状态，按证据扩展容量；不是每次训练从零开始，也不是无限追加互不协作的副本。

1. **固定容量多周期学习**：先至少三个顺序任务周期，测 train/dev/test、旧能力矩阵与写入成本。
2. **巩固**：复用 bounded replay，只读取已批准 train/Outcome；尝试把短期适应合并进慢参数，保护稳定知识。
3. **容量诊断**：仅在固定容量学习/重放仍有证据支持的瓶颈时提出结构候选；保留“不是容量问题”的否决出口。
4. **继承式增长**：零初始残差、保持函数的扩宽、群体/连接增量等一次一个候选；旧权重、身份、记忆和状态可迁移，新增参数受资源预算控制。
5. **自主路由与协作**：区分已知任务边界和未知切换；训练路由，用最强单体/随机组/平均组/路由损伤证明协作。
6. **压缩与回收**：有无增长、固定大容量、合并/剪枝前后的能力和成本配对；失败可回到上代 checkpoint。
7. **自主选择干预**：最后让模型基于已校准的失败/不确定性选择继续学、重放或申请增长；外部脚本触发不冒充自主进化。

退出须展示连续多轮净收益与受控资源增长。不得以“主干永久冻结 + 手动任务 ID + 每任务新头”宣布开放式成长完成。

## 9. M5～M8 外围任务的具体安排

| 工作 | 可提前做的最小范围 | 正式解冻时机与交付 |
|---|---|---|
| Skill/MCP 知识 | M2.R3 使用本地有许可文本/schema/示例 | M5 做去来源测试、知识内化和真实 Outcome 回流 |
| MCP 硬件/连接继承 | 保留现有 capability/ledger | M5 连接器、executor、权限、resource/UI 独立候选；与认知 artifact 共享来源而独立回滚 |
| 客户端插件热插拔 | 维护现有静态功能与缺陷 | M5 实现受控 Vue route/panel/command 与后端 capability 生命周期；根壳/托盘/进程更新需重启 |
| interaction-group | M2 仅测训练相关 trace | M3 做归因对照，M4 做自主路由/结构成长；无收益不扩系统 |
| 小型模拟 Gate | R0 起保留快速回归与泄漏反例 | M3 加未见项目；模拟绿不代替真实任务绿 |
| provider watchdog | 使用 provider 训练/实验时维护最小健康和回退 | M6 做真实 artifact 轮换、cooldown、故障、重启重绑与打包验收 |
| HF/Transformer 残留 | 维持 native import 与 artifact 类型边界 | M6 清理 live 产品混淆项；必要迁移 tombstone 有版本与移除条件 |
| 视觉/桌面 | 修复影响训练/运行的崩溃即可 | M6 统一水墨 Taiji logo、任务栏/托盘/通知、圆角/DPI、侧栏无需常态滚动、生命状态入口去重与雷达图 |
| CI/仓库/发布 | 每轮定向回归、lint、diff，阶段末全量 | M7 绑定 release manifest，验证 Windows/Linux、API、前端、Legacy-off、打包与 main 状态 |
| CUDA | 当前仅记录 CPU workload/吞吐，保持硬件阻塞 | M8 实机 CPU↔CUDA checkpoint、数值与成本对照，再做热点优化 |

第三方服务的网络、凭据和副作用权限在实际接入时单独确认；本计划不授权购买算力或发布/推送。现有第三方/客户端成果继续保留为底座。

## 10. 研究与工程纪律

- 每轮开始确认工作树、checkpoint、现行计划和用户已有改动；不因旧标题“下一步”恢复过时任务。
- 一轮只检验一个主要假设，记录输入、预期反证、可写 owner、资源预算、退出/停止条件。
- 训练前保存恢复预检必须发生在第一次正式更新前；报告无保存证据不准长跑。
- 复用现有 trainer/evaluator，共同能力进入稳定模块；历史脚本保留可追溯，新增脚本须说明无法复用的原因。
- 代码改动运行相关 pytest、ruff/格式/类型及涉及的 API/前端检查；本轮仅文档不宣称全仓 CI 已绿。
- 普通 bug 修复、checkpoint 合同、泄漏检查是前置质量，不要求把整个仓库所有历史债先清零才能继续训练。
- 跨阶段跑现有 CI 矩阵；只统计实际执行结果。报告明确 tests passed、实验失败、未执行、硬件阻塞四种情况。
- 每个完成单元提交精确范围，不用 `git add .`；不自动推送。清理前确认产物所有权，checkpoint/原始报告不随便删除。
- 核心讨论保留 active；证据解释放 reference；重复日志归 archive。活动执行计划控制在约 30 KiB，超出时先移走已完成日志，不拆出第二份路线。

## 11. 维护与索引

- [研究审视与证据](../../reference/TAIJI_RESEARCH_REVIEW_2026_09_06.md)：事实、复现、失效结论、技术参考，无独立执行顺序。
- [实现事实](../../reference/IMPLEMENTATION_STATUS_2026_08.md)：当前代码能力边界。
- [本次归档](../../archive/history/research_review_20260906/README.md)：原计划全文与中止草案。
- [2026-09-01 旧收敛记录](../../archive/history/roadmap_convergence_20260901/README.md)：更早设计和事件。
