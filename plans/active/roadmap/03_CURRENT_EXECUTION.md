# Seed / Taiji 唯一执行计划

> 修订：2026-09-10；实际代码/报告基线 917c8bf9。本文覆盖所有旧文档中的执行许可和“下一步”。
> 本轮任务是根据新增结果修订方案；训练与实现按下述验收顺序在后续开发中执行。
> 研究依据：[本轮源码与结果复审](../../reference/M4V2_POST_C_STAGE_RESULT_REVIEW_20260910.md)；[历史执行记录](../../archive/history/20260910_result_review/EXECUTION_HISTORY.md)。

## 阶段收束：完成研究审计，不等于完成模型验收

本轮从 601413cd 的收束基线继续完成了 P0 等 replay validation-only 诊断、P1 失败审计、P1.1 数据契约修复、P2 小预算 validation pilot 和 P2.1 只读输出/行动链诊断；没有读取新的 sealed payload，也没有 promotion 成绩。代码/报告证据以 917c8bf9、P0 报告、P1 v1/v2 报告、P2 pilot/P2.1 报告及本结果复审为准。

| 工作线 | 收束状态 | 后续处理 |
|---|---|---|
| 五类 K1/K2 学习器 | 实现资产保留；共 5,648 有效参数，不代表通用语言或完整认知能力 | 用作同父代持续学习基线，先不扩参 |
| fast/slow 与 replay | 等 replay 下与直接 continuation 等价；拆分独立贡献未证实 | replay 作为效果基线；FS 只保留为状态实现候选 |
| widened / 旧 parity | 当前合成路线关闭；错误计数结论撤回 | 保留失败证据和 XL 对照，不继续补次数或凑容量 |
| C-stage / scorecard | 报告入账完成；覆盖范围有限，can_promote=false | 不追加同质 formal；换成五类、同预算、同 artifact 验证 |
| S/G/K 连续整合 | v1 仅预注册且暂停，不视为已实现 | P3 基于学习结果重订 v2；禁止先做 lineage 再补能力 |
| Seed / IDE / provider / 插件 | 已有工程资产保留；本轮未重新验收客户端全链路 | 仅修阻塞主线的故障；新能力按 P5 的依赖解冻 |
| CI、临时目录与发布 | 不把历史局部测试当当前全仓通过 | 变更相关检查随步执行；发布另验收，不批量删除未知资产 |

长期核心目标不变：Taiji 拥有认知状态与行动选择，继承已有权重和学习状态持续成长，并可使用成熟技术。当前五类任务只是实验载体，不应被固化成架构能力上限。神经群体协作、开放式成长、跨域迁移和自主进化仍是待验目标，不能从模块存在或 checkpoint 数量推断完成。

### 下一阶段唯一交付目标

**先完成 P2.2 安全 abstention 与 recovery bridge canary，再决定是否继续学习或进入 P3。** P0 已完成，P1.1 已通过数据 Gate，P2 pilot 已真实训练但未通过能力晋级；P2.1 证明连续 MSE 下降没有转成更高离散命中，6/10 validation 行在输入置信度低于 `0.55` 时安全 abstain，K1→K2→planner→Workbench 的 wake-only/replay 成功率为 4/10，R 的 `content:recover-target` 没有既有只读 route。P3–P5 是后续路线，不是当前并行待办。不以添加新器官、新 Gate、更大模型或训练次数代替输出、保持和资源验收。

- P0 已确定当前实现的效果基线：在 model17/course0、150 条 wake＋50 条固定 replay 上，FS 与 C-replay、FS-no-replay 与 C 的有效状态峰值差均为 `4.76837158203125e-7`，低于预先冻结的 `1e-5`；checkpoint preflight 通过。当前数据覆盖的验证类为 A/B/C，D/R 留给 P1。
- P1 v1 失败审计确认了根因：450 条 train 记录中每类表面 observation digest 为 90 个，但实际 K1/K2 mask-visible input 各只有 1 个；validation 缺 D/R、无 project 隔离。报告保留为失败证据，不覆盖。
- P1.1 已通过：修复后的 450 条 train + 10 条 validation 中，A/B/C/D/R 每类均有 2 个 K1/K2 visible input；course seed 改变可见序列；validation 覆盖五类，并与 train 在 project/template 上隔离。报告见 [P1 v2 数据契约审计](../../../reports/taiji_m5_k_p1_data_contract_audit_v2_20260910.json)，清单见 [P1 v2 manifest](../../manifests/taiji_m5_k_p1_data_manifest_v2.json)。model17/23/31 state_dict 仍相同，但本阶段只做单父代继承学习，该旁证不再作为 P1 Gate。
- P2 pilot 已完成：P1 v2 的 460 条记录重建为 0 mismatch；50 条均衡 wake（A/B/C/D/R 各 10）+ 10 条固定 replay；checkpoint 保存和独立进程恢复均通过。报告见 [P2 v2 pilot](../../../reports/taiji_m5_k_p2_validation_pilot_v2_20260910.json)，机械失败保留在 [P2 failure report](../../../reports/taiji_m5_k_p2_validation_pilot_failed_20260910.json)。
- P2 结果：frozen macro MSE `0.156574`；wake-only `0.029237`（Δ `-0.127337`）；wake-replay `0.030520`（Δ `-0.126053`）。但三臂宏观 semantic/transition goal/content 命中均为 `0.4`，R 类命中为 `0.0`；replay 相比 wake-only 反而使宏观和最坏类 MSE略差。因此只能确认连续输出拟合和 checkpoint 链路有效，不能确认离散输出、行动成功、抗遗忘或 replay 独立收益。
- P2.1 已完成只读诊断：10 条 validation 重新构建为 0 mismatch；三臂均为 5,648 有效参数，checkpoint 独立恢复通过。frozen/wake-only/wake-replay 的 K1→K2→planner→隔离 Workbench 成功率分别为 3/10、4/10、4/10；6/10 行因输入 confidence 低于 K1/K2 的 `0.55` floor 输出 `unknown`，不是 argmax 读出错；frozen 另有 1 条因 `stale_world_observation` 被 planner 拒绝。报告见 [P2.1 诊断](../../../reports/taiji_m5_k_p2_output_action_diagnostic_20260910.json)。
- P2.1 还确认既有 `READ_ONLY_ROUTES` 没有 `content:recover-target`→只读能力的路由。这个缺口不能用降低 confidence floor 或给缺失文件直接执行来掩盖；下一步必须先做安全 recovery bridge canary。
- P2 只允许使用 P1 v2 manifest、冻结的 candidate/scorer/threshold/resource contract；训练前必须重新通过 checkpoint 保存/独立恢复 preflight。当前不进入 P3。
- 阶段完成必须同时给出训练基线、可重建数据清单、不可变候选、五类结果及失败分析。没有收益或保持失败也是可收束的研究结论，但不得因此解冻 P3 的能力整合或晋级。
- 按验收事件排期，不承诺缺乏运行时间依据的日历日期；同一时间只推进一个研究问题。

### P0 已完成：等 replay 机制归因

P0 的可重建入口为 [等 replay 诊断脚本](../../../scripts/training/eval_taiji_m5_k_p0_equal_replay_diagnostic.py)，结果见 [诊断报告](../../../reports/taiji_m5_k_p0_equal_replay_diagnostic_20260910.json)。它从同一 model17 v4 worker 快照派生 C、C-replay、FS、FS-no-replay，消费同一 150 条五类经历和同一 50 个 replay 索引，并逐 wake/replay/consolidate 记录轨迹与 A/B/C 验证类 MSE。结果完成“直接 continuation＋replay 是效果基线”的归因；它没有证明五类泛化、独立模型泛化、完整闭环或晋级。

因此不再以 FS 相对 C-replay 的效果差作为成长证据。FS 的剩余价值转为状态拆分、保存和恢复接口候选，P3 仍需独立进程中断续训验证。

### 自动推进与讨论边界

P1 合同完成后可做 P2 validation pilot。最终测试前冻结主指标、最低有意义收益、各类允许遗忘界和资源预算，注明各值的依据，禁止根据测试成绩放宽。

出现下列情况应提交已有成果并停在决策点：有效信号不足需要改变任务定义；公平对照后仍无收益需要改变学习机制；资源约束迫使缩减目标；或准备改变认知所有权/默认发布模型。讨论时给出证据、保留方案与替代方案的收益和代价，再更新唯一计划；不自动扩展训练规模或购买算力。

本轮已完成 P0/P1 validation-only 诊断、P1.1 修复、P2 小预算 pilot 和 P2.1 输出/行动链诊断；下一步只做安全输出/recovery bridge canary，不读取 sealed、不扩展 formal 矩阵、不进入结构成长。

## 当前判断

K 信号空间已扩至五类；fast/slow＋真实 replay 已实现并通过基础机制测试。C-stage v2 的 D/R/A 小型评估中，FS 相对 C 平均 MSE 改善 0.001868，D/R 改善 0.002495，原报告四门通过。该结果包含额外 replay 的收益；跨模型独立性、全五类保持、等预算机制优势和 S/G/K 整合仍未验证。can_promote=false。

widened 合成路线已收束。先前“14,252 本轮更新”“v4 worker 三 seed 独立”“新 seed 即新课程”不再作为设计依据。P0 已把效果基线收敛为直接 continuation＋replay；P1 则证明当前课程的表面变化没有进入 K 的有效输入。旧 SGK v1 预注册待修订，暂停其 lineage-first 执行顺序。

P1.1 已把数据入口修复为可见状态优先的合同：A/B/C 通过语言证据的 resolved/ambiguous 变化，D 通过 header 语言证据的 ambiguous/resolved 变化，R 通过显式 recovery-language hint/no-hint 变化。P2 pilot 证明同一父代上的 K1/K2 连续 MSE 可显著下降，但离散 readout 命中不随之提升；这把问题从“有没有训练信号”推进到“输出阈值/目标/行动桥是否正确”。

## 唯一下一步：P2.2 安全 abstention 与 recovery bridge canary

目的：把 P2.1 暴露的低置信度 abstention、K2 世界对齐和缺失目标恢复问题收敛成不越权、可测的输出/行动合同；完成前不继续扩大训练，不降低全局 confidence floor。

1. 冻结 P2.1 的安全边界：输入 confidence 低于 `0.55` 时不强行产生可执行 Goal/ContentPlan；定义 typed abstention/clarification 结果，要求下游显式知道“无动作”而不是把 `None` 当异常。不得通过降低全局 floor 换取命中率。
2. 为 `content:recover-target` 设计并实现显式只读 recovery contract：只允许列出受限 workspace 候选（例如 `workspace.list`），不对缺失路径直接 `read`，不写文件、不执行 terminal；route、参数、能力快照和失败原因都必须可审计。
3. 对 D 的 `stale_world_observation` 保持拒绝语义，增加 K1 预测世界→K2 级联世界→observation 的对齐 canary；对齐失败时输出结构化拒绝原因和安全下一步，不修改 K 权重掩盖 bridge drift。
4. 在相同 10 条 validation 上做无训练 P2.2 canary：比较原 route、typed abstention、recovery route、级联对齐四层；记录模型输出、planner admission、Workbench 成功、失败恢复和 route/快照/参数漂移。不得读取 sealed，不新增 replay，不改 P2 scorer。
5. 结果出口固定：安全 abstention、R recovery、K2 对齐和原有成功路径均通过才进入下一次 targeted learning；否则按失败层修 contract。只有输出/行动、逐类保持、checkpoint 和资源同时通过才可解冻 P3。

## 后续依赖顺序与验收

| 顺序 | 工作重点 | 进入下一步的条件 |
|---|---|---|
| P0 | 相同 replay 的机制归因 | **已完成**：两组轨迹均在 `1e-5` 内等价，checkpoint preflight 通过 |
| P1 | 真实学习信号与五类数据合同 | **已通过 P1.1**：五类各有至少 2 个 K1/K2 visible input，validation 五类覆盖且 project/template 隔离 |
| P2 | 五类学习及保持的独立验证 | **pilot 与 P2.1 已完成但未晋级**：连续 MSE 改善，命中不变；先执行 P2.2 安全 abstention/recovery bridge canary |
| P3 | 中断续训与 S/G/K 联合状态整合 | joint checkpoint、逐 phase retention、rollback 可复现 |
| P4 | 结构成长必要性与收益验证 | 容量压力真实存在，增长收益优于强固定基线 |
| P5 | 知识来源、IDE、客户端、硬件发布 | 各项按所需模型能力与接口成熟度解冻 |

### P1：有效信号与评估数据（P1.1 已通过）

- 审计现有五类课程；按 K1/K2 typed mask 可见输入、目标张量、时序组合生成签名。分别统计经历数、唯一文件数、有效类数、模板族数。注释/文件名变化允许作为同分布扰动，不能充当新能力或新独立样本。
- train/validation/test 以项目或任务模板分组，完整覆盖 A/B/C/D/R；每类包含多个不同可见状态/组合，记录数量与重复率。测试设计应同时测同分布泛化和未见组合，不能每轮仅更换 seed。
- 父 worker 沿用现有权重即可做机制诊断；若主张跨模型泛化，则以实际初始化/训练流变化构建父 worker，并核验 state_dict 差异。所有独立性结论由实测决定，不强制为了 n=9 重建模型。
- 将 v1–v5 已读 sealed 登记为 consumed；后续开发用 validation。候选、scorer、阈值、资源预算和最终输入在读取下一份测试成绩前锁定。
- v1 失败合同保留为 [P1 失败审计](../../../reports/taiji_m5_k_p1_data_contract_audit_20260910.json)；修复后的合同状态为 `passed`，见 [P1 v2 审计](../../../reports/taiji_m5_k_p1_data_contract_audit_v2_20260910.json) 与 [P1 v2 manifest](../../manifests/taiji_m5_k_p1_data_manifest_v2.json)。P2 只能使用 v2 入口，不能用 v1 的一类一签名课程训练。

### P2：验证学习与保持

- 主要因果臂为 P0 选定流程与同新增训练预算的直接学习对照；F 测零更新漂移。FS 若仅是等价状态实现，无须继续宣称胜过同 replay 的基线。XL 作为容量参考，动态增长阶段再做最终容量对齐。
- 用五类宏平均和每类 MSE；D/R 弱类单列，A/B/C 旧强类非劣逐类检查，报告最坏模板结果。整体均值不能掩盖遗忘。
- 增加真实预测链：K1 预测→K2 多步状态→隔离 Workbench 执行。真实任务成功率/失败恢复与局部 MSE 分账；低置信度和 D/R 路径必须有可解释出口。
- 使用 validation-only pilot 确定样本量、最低有意义改善、数值容差和逐域非劣界。浮点噪声容差与“允许遗忘多少”分别定义；不能用 candidate 退化方差自动放宽所有门槛。
- 先保存候选和 presealed 合同，随后同一个 artifact 只读评分，不在第二阶段重训候选。记录代码版本、数据/参数/状态摘要和所有失败。
- 结果出口：收益/保持/资源通过→P3；无收益→回对应反馈或数据根因；数值等价→保留成本更合理的基线；机械错误→修复后重做技术预检。不得看测试成绩改当前版本阈值。
- P2 pilot 已执行上述最小预算和独立 checkpoint preflight；P2.1 又完成了 K1→K2 级联与隔离 Workbench 诊断。当前结论仍仅为连续 MSE 拟合改善；goal/content 命中率未改善，6/10 行低于 confidence floor，R route 缺失，wake-replay 不优于 wake-only，因此按 P2.2 先修安全输出/recovery bridge，不进入 P3。

### P3：状态整合与同一父代连续课程

P3 前先发布 SGK v2 替代 [暂停的 v1](../../reference/M4V2_SGK_PROMOTION_COURSE_PREREGISTRATION_20260910.md)，具体要求：

1. 完整 checkpoint 含 parent、worker、slow/fast、replay payload 或可解析内容地址、游标、采样/RNG 状态、预算计数、origin/attached lineage；缺失/篡改内容拒绝恢复。
2. 比较 uninterrupted 与 wake 中段、replay 中段、consolidate 前后三种中断恢复轨迹，核验后续更新/选样/最终预测。同进程 from_checkpoint 不替代独立进程复验。
3. lineage re-binding 仅是元数据整合；保留 origin_parent 与逐 phase chain，验证错误 parent/缺链/换 worker 失败。S/G 尚不改变 K 输入，不能将联合归档宣称迁移。
4. 先单 cell S→G→K，阶段后完整评估旧域。统一 loss/surprise 的 delta=after−before，保持条件为 delta≤epsilon；S 遗忘以 P1→P2 衡量，总改善以 P0→P2 衡量。K 不写 parent 的零差只能证明隔离。
5. 资源分别列有效推理参数、学习持久状态、replay、临时 scratch、进程峰值、保存内容和实际 I/O。FS 有效参数 22,592 字节，slow＋fast 45,184 字节；真实总成本更高，预算经预检后冻结，不要求靠填充文件凑 checkpoint 数。
6. 单 cell 通过后按 P1 定义的真实统计单位扩展；通过只取得联合学习证据。再更新 scorecard 的可验证 veto，默认 runtime rollout 另设灰度与回滚验收。

### P4：回归态极的长期目标——继承式结构成长

在固定容量持续学习和保持成立后，使用多任务干扰、容量扫描与长序列退化确定扩容压力。增长从同一模型继承有效权重和学习状态，新增结构零影响出生，并有可测 credit/活动/贡献。

对照至少包含最强固定容量学习流程、同最终有效容量的 fixed-large、动态增长及结构 lesion；同时比较质量、旧域保持和累计资源。不以 replica 翻倍、元数据 lineage 或单纯保存更多权重替代神经结构成长。是否引入成熟网络/表征组件，以项目需求和对照证据决策，避免原始从零重造与为避 Transformer 而降低能力。

### P5：外围开发顺序

1. 模型知识来源：Skill/MCP 文本、文档、成功/失败轨迹形成有来源语料；与普通数据同预算比较内化和未见任务收益。工具知识与宿主执行权限分离。
2. IDE/interaction-group/小型模拟：先支持模型产出类型化动作、识别文件语言、解释语言切换、preview/执行/undo；模拟和真实 Workbench 使用一致的动作合同。
3. 客户端插件热插拔与 provider watchdog：接口稳定后实现能力发现、兼容性、设备适配继承、故障隔离和回滚。语言 provider 可作语言器官，成绩不混入原生学习收益。
4. CUDA 待硬件具备后做 CPU 一致性与吞吐对照；产品视觉在接口稳定后统一 logo/托盘/任务栏、圆角、状态页和旧 HF 入口。严重客户端故障及本次变更引入的 CI 错误优先修复。

## 工程与文档纪律

每步完成后只更新本文当前状态和一份必要证据；归档调试流水，核心架构/需求持续留在 active。失败报告不覆写、不改绿。已否决 widened/旧 parity 不再获运行许可，产物保留可追溯性，确认无引用的临时产物才清理。

上次结果复审运行的 5 项 FS 基础测试通过；本次收束不重报为新增测试成绩。未运行新的研究训练，未宣称全仓 CI 通过。实现阶段按实际 workflow 运行相关 pytest、Ruff、B/SIM、Black、core mypy；脚本改变必须核对 CLI/输出合同与负例。计划/链接变更仅做 JSON、引用和 diff 校验，不新增镜像文档内容的测试。

文档仅保留一个执行入口，不再新增平行总计划：本文记录阶段状态、下一步和验收；结果复审记录证据解释；核心需求与架构常驻 active；已有历史流水保留在 archive。旧预注册即使留在 reference 也不重新获得执行许可。当前收束不移动有引用的研究资产，不覆盖旧报告或删除 checkpoint。根目录存在部分无读取权限的临时路径，未证明其为空或无用；后续清理须逐项验证绝对路径、引用和可恢复性，不能把它们报作已清理。

本轮 P2 pilot、P2.1 报告与计划同步已提交本地 main。后续唯一入口是 P2.2 安全 abstention/recovery bridge canary；不跳到 P3/lineage 实现，也不按历史“下一步”自动开跑。
