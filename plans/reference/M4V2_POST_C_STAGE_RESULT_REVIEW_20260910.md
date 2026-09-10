# 2026-09-10 后续结果复审与路线修订

依据：本地 main 提交 4629bc8c；比较 da8a16e1 之后的 51 个变更文件，重点读取实际 learner、runner、课程生成器、checkpoint 和 formal/scorecard。没有重跑正式训练，没有覆盖历史评分。

机器复核：[结果审计](../../reports/taiji_m4v2_plan_result_review_20260910.json)。本次检查通过 systematic-debugging 的源码追踪与最小反例定位问题。当前唯一执行顺序见 [执行计划](../active/roadmap/03_CURRENT_EXECUTION.md)。

## 1. 实际进步及证据范围

| 项目 | 已有证据 | 不能推导的结论 |
|---|---|---|
| widened | 等容量比较已执行；相加导致校准问题，平均修复后收益接近 XL | 没有显著胜出不等于已证明统计等价，也不能否定所有扩容设计 |
| 5 类 K worker | fact 14；K1/K2 共 5,648 有效参数；D/R 可训练 | 与旧 3 类不同父代、不同测试集的 −0.002/−0.129 不能作为同口径性能增幅 |
| FS 机制实现 | wake 写 fast、replay 写 slow、consolidate、恢复测试 5/5 通过 | 状态拆分自身具有独立学习优势 |
| C-stage v2 | 报告 G1–G4 通过；D/R 三课程均胜 C | 全五类泛化、跨独立模型稳健性、同预算机制优势 |
| scorecard v3 | 能力/机制证据入账，promotion=false | 默认模型已经成长或 S/G 学习已经迁移到 K |
| S/G/K | 只有预注册；re-binding/joint checkpoint 尚待实现 | 连续课程已经执行、晋级已经具备条件 |

按 C-stage v2 逐课程值重新计算：FS 相对 C 的 sealed combined-MSE 平均差为 **−0.0018684537**，D/R 平均差为 **−0.0024950897**。负值是改善，原始四门通过的报告保持原判；新的解释是“FS 加 replay 的组合在当前小型 D/R/A 评估上有收益”。

## 2. 会改变开发顺序的实际问题

### 2.1 上一轮我给出的 14,252 更新步结论需要撤回

旧审计把继承累计计数当成本轮新增更新；14,252 中 14,240 来自父代。用户后续代码已修正，双通道与双 replica 原构建均新增 12 个 worker-example 步。当前 C-stage 是 C 300、FS 300 wake＋100 replay，后者多 1/3 新增工作量。继续保留旧 JSON，但不再按旧审计扩充 14,252 次新训练或强凑 9 个 checkpoint。

后续计量分别记录：继承计数、本轮 experience 消费、worker-example 步、fit 调用/epoch、head-update 次数、replay 消费、实际写盘次数及文件字节。现有 fit 的 training_steps += epochs × len(examples) 是样本暴露计数；batch size 改变时不等同参数更新调用次数。

### 2.2 缺少直接 continuation＋相同 replay 对照

源码 taiji/k_fast_slow.py 中有效参数 W=slow+fast。wake 把 Δ(W,x) 写入 fast，sleep 把 Δ(W,x) 写入 slow；consolidate 保持 W 不变。在同一经历、顺序、学习率下，有效轨迹与直接 continuation 依次消费 wake＋replay 在精确算术下相同，浮点实现可能出现舍入差异。

因此预期 FS 与“C＋完全相同 replay”近似一致，这是待验证的机制诊断。它不否定 FS 的状态管理价值，却意味着仅对比 C300 与 FS400 无法把收益归因于拆分。新增 C-replay400 为主因果对照；保留 C300 测 replay 总体价值。replay-lesion 预期接近 C300，可作一致性探针，不能取代 C-replay400。

### 2.3 独立性仍未成立

直接读取 checkpoints/taiji_k_workers_v4/model_17、23、31 的 K1/K2 state_dict：四个对比的最大绝对差均为 **0.0**。manual_seed 后 learner 显式零初始化、同课程确定性训练，不能制造独立模型。SGK v1 §3 的“v4 重建后 worker 互异”与实测矛盾。

_build_course_v4 使用 _class_schedule(seed % 3)，故 3/4/5 与 0/1/2 的类序列分别相同。文件注释可能改变元数据/掩外输入；这不足以证明有效学习信号新颖。须核查应用 typed masks 后输入、target、实际类序列，而非只看文件或 artifact digest。当前 n=3 仅指三个课程条件，不视为三个独立抽样总体，更不能算九个独立模型。

### 2.4 测试覆盖与版本边界

sealed v5 只有 D/R/A 各一个 episode，缺 B/C；G4 的“整体非劣”仅覆盖这三条记录。v1→v2 修改 G1 并换文件 seed 是重新预注册的窄复验，不能自动获得新分布独立性。后续全五类分别报告、按模板/项目族隔离，训练/validation/最终测试都要覆盖旧强类与弱类。

当前 _loss_score 的 K2 输入来自 example.before 的真实状态，尚不是 K1 预测→多步 K2→真实 Workbench 执行的完整闭环。MSE 与任务成功率必须分别验证。多个 sealed 被评分后均标为 consumed；下一轮先锁模型/阈值/输入，再由同一 checkpoint 评分，避免当前 runner 在 validation 与 sealed 阶段重复训练却缺少候选摘要一致性凭证。

### 2.5 SGK retention 方向和资源合同不可原样执行

mean_surprise 越低越好；统一 Δ=after−before，应验 **Δ≤epsilon**。旧 §5 G3 写成 Δ≥−epsilon：1.0→1.1、epsilon=0.01 会放过遗忘，反而可能拒绝改善。下一版同时测 P0、S 后 P1、G 后 P2；相对 P1 测 S 遗忘，相对 P0 测总改善。

v4 有效参数 22,592 字节；FS slow＋fast 为 45,184 字节，尚未包含 parent payload、scratch、replay 和临时张量。有效推理容量相同与训练状态内存相同是不同条件。旧“worker 字节相等、checkpoint≤1.05×C、RSS≤1.25×frozen”缺少统一计量与可行性前测。下一版先测真实资产和隔离进程资源再冻结预算；不能填充 C 文件强凑等价。

FS checkpoint 保存 slow/fast 和 replay digest，但不包含 replay 的可重建经历 payload、课程游标及独立采样状态契约。有效权重恢复测试通过不足以证明半程中断续训一致。joint checkpoint 之前补边界恢复、经历解析、摘要校验、缺失/篡改拒绝、最终轨迹等价。

## 3. 修订决策

保留 FS 实现和弱类收益，把其状态写为“有收益的候选训练流程；状态拆分贡献待隔离”。widened 继续关闭，XL 保留为容量参考。先完成等 replay 的最小机制实验与可见数据审计，再决定是否值得整合 lineage；项目上限来自可验证的持续学习、知识迁移和结构成长，参数字节凑倍数不是目标。

旧 SGK v1 标记为待修订、禁止按原 §9 开跑；已有报告/权重不改写。本轮仅更新研究依据、计划和文档入口。P0 诊断完成后，唯一下一步转为 P1 的 validation-only 有效信号与五类数据合同审计。

## 4. 阶段收束补记

在 601413cd 之后进行计划收束，未发现新增提交或实验成绩。本节不构成另一轮性能验证，前述数值和限制不变。已有五类 worker、FS/replay、课程和评分工具作为研究资产保留；widened 当前合成路线关闭；SGK v1 暂停；默认晋级仍不获许可。

下一阶段不再同时推进结构成长、整合和客户端扩展，而以 P1–P2 建立固定容量持续学习基线为单一目标。P0 的公平归因已完成；正面结果必须继续经过五类数据隔离及保持检验，负面或等价结果允许收束，但不能包装为能力通过。长期成长目标保留，不将当前小型任务集合当作 Taiji 的最终结构定义。

本次同步检查核心需求、Seed 架构文档、目录与工作区状态；旧架构文档中的客户端“当前状态”未经本轮运行验证，故不据此宣称功能已完成或缺失。部分本地临时目录读取受限，未做删除。执行边界与下一步统一维护在当前计划，不再另建并行收束方案。

## 5. P0 等 replay 诊断结果（2026-09-10）

随后按唯一计划实际执行了 P0 validation-only 诊断，完整报告为 [taiji_m5_k_p0_equal_replay_diagnostic_20260910.json](../../reports/taiji_m5_k_p0_equal_replay_diagnostic_20260910.json)，入口为 [eval_taiji_m5_k_p0_equal_replay_diagnostic.py](../../scripts/training/eval_taiji_m5_k_p0_equal_replay_diagnostic.py)。本轮未读取 sealed test，也未附着或晋级候选。

| 检查 | 实测结果 | 解释 |
|---|---:|---|
| checkpoint 保存/恢复 preflight | 通过 | parent K1/K2 与 FS birth 都可写盘并恢复；这是训练前技术门，不是能力通过 |
| wake 课程 | 150 条 | A/B/C/D/R 各 30 条，使用 model17/course0 v4 worker |
| 固定 replay | 50 条 | 一次生成索引清单，由 C-replay 与 FS 共同消费、顺序一致 |
| FS vs C-replay 轨迹峰值差 | `4.76837158203125e-7` | 小于预冻结 `1e-5`，等价 |
| FS-no-replay vs C 轨迹峰值差 | `4.76837158203125e-7` | 小于预冻结 `1e-5`，等价 |
| validation 类覆盖 | A/B/C | 既有 helper 的最小验证覆盖；D/R 不据此宣称已验证 |

该结果把当前效果归因收束为“直接 continuation＋replay”；在本地 delta 语义和相同经历下，fast/slow 状态拆分没有可分离的有效权重更新收益。FS 仍可作为可恢复状态实现候选，但它的资源成本、半程中断恢复和 replay payload 完整性必须在 P3 单独验证。C-stage v2 的小型改善不能再描述为 FS 独立优势，也不能推导五类泛化、真实 Workbench 成功率、结构成长或自主进化。

因此当前唯一下一步已从 P0 前进到 P1：审计五类实际 masked input/target、模板/项目隔离、统计单位和独立性；在该合同通过前不追加正式效果训练。
