# 2026-09-10 后续结果复审与路线修订

初始复审依据：本地 main 提交 4629bc8c；比较 da8a16e1 之后的 51 个变更文件，重点读取实际 learner、runner、课程生成器、checkpoint 和 formal/scorecard。随后追加的 P0/P1/P1.1/P2/P2.1 validation 结果落在本地 main 提交 917c8bf9；P2.2 安全 bridge 结果落在 7a9e9a01；没有读取新的 sealed payload，没有 promotion 成绩。

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

旧 SGK v1 标记为待修订、禁止按原 §9 开跑；已有报告/权重不改写。本轮仅更新研究依据、计划和文档入口。P0 诊断完成后，唯一下一步曾转为 P1 的 validation-only 有效信号与五类数据合同审计；该审计结果见本文件 §6，P1.1 修复结果见 §7，当前入口已转为 P2 validation pilot。

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

因此当时的唯一下一步已从 P0 前进到 P1：审计五类实际 masked input/target、模板/项目隔离、统计单位和独立性；在该合同通过前不追加正式效果训练。该审计随后在 §6 执行，P1.1 修复结果见 §7。

## 6. P1 数据契约审计结果（2026-09-10）

随后按计划运行了 validation-only 的 [P1 审计脚本](../../scripts/training/audit_taiji_m5_k_p1_data_contract.py)，完整结果见 [P1 数据契约审计报告](../../reports/taiji_m5_k_p1_data_contract_audit_20260910.json)，可重建清单见 [P1 manifest](../manifests/taiji_m5_k_p1_data_manifest_v1.json)。本轮恢复了现有 model17 worker、构造了三组 course seed 的五类课程和既有 validation；没有调用 `fit`，没有读取新的 sealed payload。

| 检查 | 实测结果 | 结论 |
|---|---:|---|
| typed mask 合法性 | K1/K2 均通过；K1 输入 23 维、K2 输入 359 维、各 14 行 | 技术结构合法，但不等于信号有变化 |
| train 规模与类别 | 3 个 course seed × 5 类 × 90 = 450 条 | 表面数量完整 |
| 每类 observation digest | 90 个 | 包含文件字节/元数据变化，不能直接视为可见样本 |
| 每类 K1/K2 mask-visible input | 各 1 个；重复可见输入计数各 89 | 当前课程退化为按类查表，未证明类内学习 |
| 每类 target digest | K1/K2 各 1 个 | 目标没有提供额外类内区分 |
| course seed 有效变化 | K1/K2 的 0/1、0/2、1/2 均为 false | seed 变化没有改变实际可见签名 |
| split | train 只有 `m5k1-project`；validation 只有 A/B/C，缺 D/R | 无 project 隔离，不能做五类泛化/保持结论 |
| model17/23/31 state_dict | K1/K2 两两最大绝对差均为 0.0 | 不能把复制的同状态 worker 计为独立模型 |

报告状态为 `needs_data_revision`，`can_start_p2=false`。因此 P2 训练/效果验证被数据 Gate 阻塞，不能用更多 fit、重复 seed 或放宽 scorer 来掩盖问题。下一步只修复课程生成与 split contract：把变化放入 typed mask 可见的合法状态/时序组合，令每类至少有多个 K1/K2 可见状态、validation 覆盖 A/B/C/D/R 并按 project/template 隔离；随后重跑同一审计。该 Gate 通过前不读取新的 sealed 测试，也不进入正式训练。

## 7. P1.1 数据契约修复结果（2026-09-10）

针对 §6 的失败 Gate，新增 [P1 数据构建器](../../scripts/training/build_taiji_m5_k_p1_data.py)，并用同一 [审计入口](../../scripts/training/audit_taiji_m5_k_p1_data_contract.py) 重跑；旧 v1 失败报告和 manifest 保留不覆盖。通过结果见 [P1 v2 审计报告](../../reports/taiji_m5_k_p1_data_contract_audit_v2_20260910.json)，合同见 [P1 v2 manifest](../manifests/taiji_m5_k_p1_data_manifest_v2.json)。本轮仍未调用 `fit`，未读取新的 sealed payload。

| Gate | 修复后实测 | 结论 |
|---|---:|---|
| train/validation | 450 train + 10 validation；A/B/C/D/R 各有记录 | 五类覆盖完整 |
| 类内 K1/K2 visible input | 每类均为 2 个；不是文件字节/注释的唯一变化 | typed mask 真正看到状态差异 |
| A/B/C 状态 | resolved-language / ambiguous-language | 语言证据导致可解释的 selection/goal 差异 |
| D 状态 | ambiguous-header / resolved-header | header 语言证据改变 selection 状态 |
| R 状态 | recovery-no-selection / recovery-selection-hint | 缺失目标下的显式恢复语言上下文改变 selection 状态 |
| course seed | 可见状态集合相同但有序可见序列三组两两不同 | 计为课程顺序差异，不计为独立模型 |
| split | validation 五类覆盖；project 和 template 均与 train 隔离 | 可以做 validation pilot |
| model17/23/31 | state_dict 仍两两相同 | 仅旁证；单父代 P1 Gate 不要求复制出独立模型 |

报告状态为 `passed`，`can_start_p2=true`。这只证明数据入口和统计合同成立，不证明 K1/K2 已经学会或能泛化。P2 pilot 结果见 §8。

## 8. P2 validation pilot 结果（2026-09-10）

按当前计划运行了 [P2 pilot v2](../../scripts/training/eval_taiji_m5_k_p2_validation_pilot.py)，结果见 [P2 v2 报告](../../reports/taiji_m5_k_p2_validation_pilot_v2_20260910.json)。首次评分字段适配错误保留在 [失败报告](../../reports/taiji_m5_k_p2_validation_pilot_failed_20260910.json)；修复后没有覆盖失败证据。本轮从 P1 v2 manifest 重建 460 条记录，`mismatch_count=0`，使用 50 条均衡 wake（A/B/C/D/R 各 10）和 10 条固定 replay；没有读取新的 sealed payload。

| 检查 | frozen | wake-only | wake-replay |
|---|---:|---:|---:|
| macro combined MSE | 0.156574 | 0.029237 | 0.030520 |
| 相对 frozen 的 Δ | — | −0.127337 | −0.126053 |
| worst-class MSE | 0.188369 | 0.102619 | 0.113938 |
| macro goal/content 命中 | 0.4 | 0.4 | 0.4 |
| R 类离散命中 | 0.0 | 0.0 | 0.0 |
| checkpoint 独立进程恢复 | — | 通过 | 通过 |

结论分两层：连续回归输出的拟合明显改善，说明 P1 数据、fit、保存/恢复链路可运行；但离散 goal/content 输出没有改善，R 类没有命中，且 replay 在本 pilot 略差于 wake-only。`can_promote=false`，不能把这次 MSE 下降称为泛化、智能、真实 Workbench 成功或 replay 独立收益。P2.1 诊断结果见 §9；在输出/行动命中与逐类保持未验收前不进入 P3。

## 9. P2.1 输出与行动链诊断（2026-09-10）

按计划运行了 [P2.1 诊断脚本](../../scripts/training/eval_taiji_m5_k_p2_output_action_diagnostic.py)，报告见 [P2.1 诊断报告](../../reports/taiji_m5_k_p2_output_action_diagnostic_20260910.json)。它只读 P2 已保存的 frozen/wake-only/wake-replay checkpoint 和 P1 v2 validation，`mismatch_count=0`，K2 使用 K1 预测 world 进行级联；没有调用 `fit`，没有读取 sealed payload，`can_promote=false`。

| 检查 | frozen | wake-only | wake-replay |
|---|---:|---:|---:|
| 有效参数（K1+K2） | 5,648 | 5,648 | 5,648 |
| checkpoint 独立恢复 | 通过 | 通过 | 通过 |
| K1→K2→planner→Workbench 成功 | 3/10 | 4/10 | 4/10 |
| K1/K2 输入低于 `0.55` floor | 6/10 | 6/10 | 6/10 |
| planner 因 `stale_world_observation` 拒绝 | 1 | 0 | 0 |
| failure recovery 触发 | 0 | 0 | 0 |

P2.1 把问题进一步分层：6/10 行在 native readout 之前就因输入证据置信度低于 `0.55` 而安全返回 `unknown`，不是简单的 argmax 错误；frozen 另有一条 K2 预测 world 与 observation 不对齐而被 planner 正确拒绝。wake-only/replay 能在 4/10 行完成真实只读 Workbench 动作，但不构成能力晋级。既有 `READ_ONLY_ROUTES` 还没有 `content:recover-target` 路由，因此 R 类即使后续产生 recovery 内容，也没有安全的缺失目标恢复桥。当前唯一下一步改为 P2.2：实现 typed abstention/clarification、受限 `workspace.list` recovery contract 和级联 world-alignment canary；不降低全局 confidence floor，不新增 replay，不进入 P3。

## 10. P2.2 安全 abstention 与 recovery bridge canary（2026-09-10）

随后按计划实现并运行了 [P2.2 canary](../../scripts/training/eval_taiji_m5_k_p2_2_safety_bridge_canary.py)，结果见 [P2.2 报告](../../reports/taiji_m5_k_p2_2_safety_bridge_canary_20260910.json)。本轮只读 P2.1 checkpoint 和 P1 v2 manifest，重建 460 条记录，`mismatch_count=0`；没有调用 `fit`，没有读取 sealed payload，`can_promote=false`。

| 检查 | 三臂结果 | 解释 |
|---|---:|---|
| typed abstention 数量 | 6/10/臂 | 对应低于 `0.55` 的低证据输入，不强行制造可执行动作 |
| abstention roundtrip | 6/6/臂 | `ReadOnlyAbstention` 序列化/反序列化一致 |
| abstention 非执行 | 6/6/臂 | payload 明确携带 `action_intent: null` |
| `workspace.list(path=".")` recovery 控制 | 2/2/臂成功 | 仅是 oracle-labeled route control，根目录参数和快照绑定均通过 |
| world-alignment 控制 | 4/4/臂成功 | 使用 observation 对齐 world 的 oracle control，不是 K2 学习命中 |
| 原始 learned route | frozen 3/10；wake-only 4/10；wake-replay 4/10 | 与 P2.1 一致，未产生模型能力晋级 |

这次结果解决了 P2.1 的安全边界缺口，但没有解决模型如何在缺失目标后获得新证据的问题。P2.2 的 recovery 与 alignment 都明确标为 `oracle_control`，不能计入 K1/K2 能力、真实 recovery 命中或 P3 解冻条件。下一步改为 P2.3：建立“缺失目标安全 abstention→根目录列举→新候选观察→可执行只读动作”的可重建 continuation 数据合同，先通过数据审计，再做同一父 checkpoint 的 targeted learning；不降低 confidence floor、不把 host policy 内化成绩当作模型能力。

## 11. P2.3 recovery continuation 数据合同（2026-09-10）

按 §10 的边界新增 [P2.3 continuation builder/auditor](../../scripts/training/build_taiji_m5_k_p2_3_recovery_continuation.py)，产出 [P2.3 manifest](../../plans/manifests/taiji_m5_k_p2_3_recovery_continuation_manifest_v1.json) 和 [P2.3 contract report](../../reports/taiji_m5_k_p2_3_recovery_continuation_contract_20260910.json)。本轮只构建并审计数据，没有调用 `fit`，没有读取 sealed payload，`can_promote=false`。

| 检查 | 实测结果 | 结论 |
|---|---:|---|
| train continuation | 6 条 | 通过；初始 recovery 不进入 fit，候选观察标为 fit-eligible |
| validation continuation | 2 条 | 通过；候选路径与 train 不重叠 |
| 初始缺失事件 | `read_success=false`、Percept confidence `0.0` | 只能走 typed abstention→`workspace.list(path=".")` |
| 列举后候选事件 | Percept confidence `0.99`、resolved 文件 | 可作为 K1/K2 读出学习目标 |
| K1/K2 示例 | payload 往返、input digest 一致 | 通过；没有把字节变化当作输入变化 |
| 隔离与寻址 | project/path/template disjoint；record/manifest digest 通过 | 可以进入训练前 checkpoint Gate |

这一步修复的是数据入口，不是模型能力。下一步唯一执行项是 P2.3 targeted learning pilot：从同一 P2 parent 继承，先做 checkpoint 保存和独立进程恢复，再只用 train candidate examples 学习；原 P1 v2 五类 validation、P2.3 validation candidate、6 条低证据安全 abstention 和 Workbench 结果分账评分。host `workspace.list` 不计入 K1/K2 命中，也不降低全局 `0.55` floor。

## 12. P2.3 targeted learning 结果（2026-09-10）

随后运行了 [P2.3 targeted learning pilot](../../scripts/training/eval_taiji_m5_k_p2_3_targeted_learning.py)，报告见 [P2.3 targeted pilot report](../../reports/taiji_m5_k_p2_3_targeted_learning_pilot_20260910.json)。它从同一 P2 parent 构造 `parent-frozen`、`continuation-targeted` 和已保存的 `p2-wake-only-reference` 三臂；训练前 parent checkpoint 和三臂保存后的独立进程恢复均通过。只对 P2.3 的 6 条 train candidate 做 fit，2 条 continuation validation 和原 P1 v2 validation 均只读；没有读取 sealed payload。

| 指标 | parent-frozen | continuation-targeted | p2-wake-only-reference |
|---|---:|---:|---:|
| P2.3 continuation K1/K2 goal 命中 | 2/2 | 2/2 | 2/2 |
| P2.3 continuation Workbench 成功 | 2/2 | 2/2 | 2/2 |
| 原 P1 v2 高/低证据 K1 goal 命中 | 4/10；安全 abstain 6/10 | 1/10；安全 abstain 6/10 | 4/10；安全 abstain 6/10 |
| 原 P1 v2 K2 goal 命中 | 4/10 | 3/10 | 4/10 |
| 原 P1 v2 Workbench 成功 | 3/10 | 3/10 | 4/10 |

结论不是“训练成功”：parent 已经能完成新 candidate validation 的 2/2，candidate-only fit 没有增加可测能力，却使 B/C/D 等旧高证据读出发生干扰；安全 abstention 6/6 保持。故 `can_promote=false`，也不进入 P3。当前失败层已从数据合同推进到 online update 的 retention/objective：下一步唯一执行项是 P2.4，用 P2 合同中固定的 50 条均衡 rehearsal 与 6 条 continuation 做确定性交错 canary，先验证不遗忘，再判断是否存在真实新增能力。

## 13. P2.4 retention-preserving objective canary（2026-09-10）

按 §12 的失败归因运行了 [P2.4 retention canary](../../scripts/training/eval_taiji_m5_k_p2_4_retention_canary.py)，报告见 [P2.4 报告](../../reports/taiji_m5_k_p2_4_retention_canary_20260910.json)。它重新构建 P1 v2 的 460 条记录，校验 P2 合同的 50 条均衡 wake 的 experience digest、顺序和 A/B/C/D/R 各 10 条，然后比较 parent、rehearsal-only、50 条 rehearsal 与 6 条 continuation 的确定性交错 stream、已有 P2 wake-only reference；训练前后 checkpoint 独立恢复均通过。

| 指标 | parent-frozen | rehearsal-only | interleaved rehearsal + continuation | P2 wake-only reference |
|---|---:|---:|---:|---:|
| 原 P1 v2 K1 goal 命中 | 4/10 | 4/10 | 4/10 | 4/10 |
| 原 P1 v2 K2 goal 命中 | 4/10 | 4/10 | 4/10 | 4/10 |
| 原 P1 v2 Workbench 成功 | 3/10 | 4/10 | 4/10 | 4/10 |
| 原 P1 v2 安全 abstention | 6/6 | 6/6 | 6/6 | 6/6 |
| P2.3 continuation Workbench 成功 | 2/2 | 2/2 | 2/2 | 2/2 |

P2.4 的 retention Gate 通过，证明固定 rehearsal 能阻止 P2.3 candidate-only 的旧类干扰；但 continuation 仍与 parent 一样是 2/2，没有新增能力。因此这不是 promotion 或 P3 解冻证据。下一步改为 P2.5：保留 recovery→list→candidate 顺序，但换成训练未出现的语言/工具链组合（优先 TypeScript＋可用 toolchain 的 inspect 组合），先对 frozen parent 做 validation-only novelty probe；只有 parent 对该真实未见组合失败，才有必要制定下一轮学习。

## 14. P2.5 novel-composition probe（2026-09-10）

按 §13 的停止条件运行了 [P2.5 novel-composition probe](../../scripts/training/eval_taiji_m5_k_p2_5_novel_composition_probe.py)，产出 [P2.5 manifest](../../plans/manifests/taiji_m5_k_p2_5_novel_composition_manifest_v1.json) 和 [P2.5 报告](../../reports/taiji_m5_k_p2_5_novel_composition_probe_20260910.json)。本轮只做 validation-only 评分，没有调用 `fit`，没有读取 sealed payload，`can_promote=false`。候选严格保留 recovery→list→candidate 顺序，使用 TypeScript、toolchain available、selection resolved、`content:inspect-language` 的组合；2 条候选路径与 P1/P2 路径 disjoint，合同和 checkpoint Gate 均通过。

| 指标 | frozen parent | P2 wake-only reference |
|---|---:|---:|
| 新组合 K1 goal/content | 2/2；2/2 | 2/2；2/2 |
| 新组合 K2 goal/content | 2/2；0/2 | 0/2；0/2 |
| 新组合 Workbench 成功 | 2/2 | 2/2 |
| 新组合 K2 状态 | `ambiguous`，content `None`（2/2） | `clarify`，goal/content 不命中（2/2） |

checkpoint preflight 在训练前和保存后的独立进程恢复均通过；P2.5 只保存验证 arm，不产生训练权重或 promotion 资格。这个结果把缺口从“新组合是否完全不会”收窄为“模型已识别目标并能执行，但 K1→K2 的内容承接仍不能稳定输出 `content:inspect-language`”。因此不能把 Workbench 2/2 或 K2 goal 2/2 误写成完整能力，也不能继续在已经通过的 K1/执行目标上重复训练。

当前唯一下一步改为 P2.6：固定 P2.5 的 novel tuple，构造 6 条 train candidate + 2 条 validation candidate，使用 P2.4 已通过的 50 条均衡 rehearsal 与 novel candidate 交错学习；只把新组合 K2 content 作为新增目标，同时把旧五类、低证据 abstention、K1/K2 goal、Workbench 和 checkpoint restore 设为非劣约束。若 K2 content 仍为 0/2，或旧类退化，保持 `can_promote=false` 并停止扩展；不进入 P3。

## 15. P2.6 retention-preserving novel K2 learning（2026-09-10）

按 §14 的边界运行了 [P2.6 novel learning](../../scripts/training/eval_taiji_m5_k_p2_6_novel_learning.py)，产出 [P2.6 manifest](../../plans/manifests/taiji_m5_k_p2_6_novel_learning_manifest_v1.json) 和 [P2.6 报告](../../reports/taiji_m5_k_p2_6_novel_learning_20260910.json)。实验从同一 parent checkpoint 开始，构造 6 条 TypeScript＋可用 toolchain＋resolved 的 novel train candidate 与 2 条全新 validation candidate；路径与 P1/P2.5 disjoint。固定 50 条 P2 rehearsal 按 A/B/C/D/R 各 10 条复现并与 novel candidate 交错。训练前 parent、三个 arm 保存后的 checkpoint 和独立进程恢复均通过，validation 未参与 fit，参数量未增长，未读取 sealed payload。

| 指标 | parent-frozen | rehearsal-only | interleaved-rehearsal-novel |
|---|---:|---:|---:|
| 新组合 K2 goal/content | 2/2；0/2 | 0/2；0/2 | 2/2；2/2 |
| 新组合 K1 goal/content | 2/2；2/2 | 2/2；2/2 | 2/2；2/2 |
| 新组合 Workbench 成功 | 2/2 | 2/2 | 2/2 |
| 原 P1 v2 K1/K2 content | 4/4；4/4 | 4/4；4/4 | 4/4；4/4 |
| 原 P1 v2 安全 abstention | 6/6 | 6/6 | 6/6 |
| 原 P1 v2 Workbench 成功 | 3/10 | 4/10 | 4/10 |

P2.6 的 Gate 全部通过：旧类相对 P2.4 parent 不下降，novel K2 content 达到 2/2，Workbench 不下降，参数计数稳定。这个结果第一次给出“某个具体 K2 内容承接目标可以在保持约束下被学习”的证据，但它仍只有 2 条 validation，且训练/验证共享同一 tuple，不能宣称跨路径泛化，也不能解冻 P3；报告保持 `can_promote=false`。

当前唯一下一步改为 P2.7：加载 P2.6 的交错学习 checkpoint，在至少 2 个全新 project、至少 4 条全新 path 上做 validation-only holdout；不调用 `fit`。P2.6 validation 只作 learned-arm sanity check，新 holdout 才计泛化 Gate。若 holdout 达到 K2 content 与 Workbench `4/4` 且已学验证 `2/2`，再讨论局部泛化和 P3；否则保持 `can_promote=false`，不靠追加同质 epoch 掩盖边界。

## 16. P2.7 independent holdout generalization（2026-09-10）

按 §15 的停止条件运行了 [P2.7 generalization probe](../../scripts/training/eval_taiji_m5_k_p2_7_generalization_probe.py)，产出 [P2.7 manifest](../../plans/manifests/taiji_m5_k_p2_7_generalization_manifest_v1.json) 和 [P2.7 报告](../../reports/taiji_m5_k_p2_7_generalization_20260910.json)。本轮加载 P2.6 `interleaved-rehearsal-novel` checkpoint，不调用 `fit`；先对源 checkpoint 做 content-addressed digest 与独立进程恢复，再构造 4 条全新 path、2 个全新 project 的 TypeScript holdout。holdout 与 P1/P2.5/P2.6 全部 disjoint，合同通过。

| 指标 | frozen parent | P2.6 learned arm |
|---|---:|---:|
| P2.6 已学 sanity K2 content | 0/2 | 2/2 |
| 全新 holdout K1 goal/content | 4/4；4/4 | 4/4；4/4 |
| 全新 holdout K2 goal/content | 4/4；0/4 | 4/4；4/4 |
| 全新 holdout Workbench 成功 | 4/4 | 4/4 |
| P1 旧类安全 abstention | 6/6 | 6/6 |

P2.7 的 generalization Gate 全部通过：learned arm 的已学 sanity `2/2`、holdout K2 content/Workbench `4/4`、P1 旧类相对 P2.4 parent 不下降、参数计数稳定、P2.6 源 checkpoint 独立恢复再次通过。父模型在同一 holdout 的 K2 content 为 `0/4`，因此这不是 parent 原有的泛化能力。这个结果支持“局部跨项目/路径泛化”，但仍不是通用语言能力或 promotion 证据；`can_promote=false` 保持不变。

P2 阶段因此满足进入 P3.0 的证据条件。当前唯一下一步改为 P3.0 checkpoint/interrupt-resume contract：固定 P2.6 learned checkpoint 为 parent，建立 SGK v2 最小内容寻址状态合同，验证 wake/replay/consolidate 边界的独立进程中断恢复、错误 parent/篡改拒绝、phase cursor/RNG/experience digest 和 uninterrupted 轨迹一致性。在 P3.0 通过前，不新增 S/G worker、不增长参数、不把 lineage 元数据当作能力。

## 17. P3.0 checkpoint/interrupt-resume contract 结果（2026-09-10）

按 §16 的入场条件运行了 [P3.0 checkpoint contract](../../scripts/training/eval_taiji_m5_k_p3_0_checkpoint_contract.py)，产出 [P3.0 manifest](../manifests/taiji_m5_k_p3_0_checkpoint_contract_manifest_v1.json) 和 [P3.0 报告](../../reports/taiji_m5_k_p3_0_checkpoint_contract_20260910.json)。本轮固定 P2.6 `interleaved-rehearsal-novel` learned checkpoint 为 parent，真实执行 K1/K2 continuation update；没有新增 S/G worker、没有增长参数，没有读取新的 sealed payload，`can_promote=false`。

| Gate | 结果 | 证据含义 |
|---|---:|---|
| source parent digest / 独立 restore | 通过 | P2.6 parent 文件、manifest/parent digest 和 K1/K2 payload 可验证、可独立加载 |
| uninterrupted 轨迹保存/恢复 | 通过 | 2 wake + 2 replay + 2 consolidate phase item 完整保存，最终 worker/budget/RNG/stream digest/cursor 可恢复 |
| wake 中段中断恢复 | 通过 | 在 wake `1/2` 保存后独立进程恢复，继续轨迹与 uninterrupted 的 worker、budget、RNG、stream digest、final cursor 一致 |
| replay 边界中断恢复 | 通过 | 在 replay `2/2` 边界保存后独立进程恢复，继续轨迹与 uninterrupted 一致 |
| 篡改/错误 parent/缺 lineage 拒绝 | 通过 | tampered cursor、wrong parent、missing lineage 全部拒绝，不降级加载 |
| rollback | 通过 | 回滚到 P3 parent 后 K1/K2 source digest 与原始 parent 一致，独立 restore 通过 |

P3.0 的结论是“当前 K1/K2 continuation 状态可以被内容寻址、独立中断恢复、校验和回滚”，不是“Taiji 已经完成 S/G/K 联合学习、结构成长或自主进化”。下一步唯一执行项改为 P3.1：在 P3.0 parent 上建立真实 S/G/K 单 cell 的 schema/owner/mask/事件合同，先做接口回放和独立恢复；没有真实 S/G 持久状态时必须显式标记 `absent`，不得用空字段包装成联合能力。P3.1 仍不扩参、不进入结构成长、不接入 CUDA/客户端外围。

## 18. P3.1 S→G→K single-cell state preflight 结果（2026-09-10）

按 §17 的入场条件运行了 [P3.1 single-cell preflight](../../scripts/training/eval_taiji_m5_k_p3_1_single_cell.py)，产出 [P3.1 manifest](../../plans/manifests/taiji_m5_k_p3_1_single_cell_manifest_v1.json) 和 [P3.1 报告](../../reports/taiji_m5_k_p3_1_single_cell_20260910.json)。本轮固定 P3.0 continuation parent，复用 P2.7 的 4 条 holdout，形成每例 observation→S update→G selection→K readout→action 五阶段、共 20 个事件；没有调用 `fit`、没有新增参数、没有读取 sealed payload。

| Gate | 结果 | 证据含义 |
|---|---:|---|
| schema / owner / event mask | 通过 | S=runtime evidence，G=control-only external selection，K=既有 K1/K2 learned readout；K readout 实际只读取 S，action 读取 G/K |
| uninterrupted / event boundary / phase boundary | 通过 | 五条轨迹均能生成 content-addressed owner/event checkpoint |
| 独立进程恢复 | 通过 | event/owner/worker/budget/RNG/cursor/logical digest 与 uninterrupted 全部一致 |
| tamper / wrong base / wrong manifest | 通过 | 篡改 cursor、错 P3.0 base、错 P3.1 manifest 均 fail-closed |
| rollback / parameter growth | 通过 | rollback parent 指向 P3.0，K1/K2 digest 未变，参数未增长 |

P3.1 的边界结论是“单 cell 状态合同和恢复链路成立”，不是“已经产生 S/G 学习能力”。实现审计还发现：当前 K1 的既有输出同时包含语义、goal 和 content，G 仍由外部 target 驱动；若直接训练新的 G 而不先处理所有权，会复制能力并让指标归属失真。因此下一步唯一执行项改为 P3.2：沿用 K1/K2 的有效权重，建立 S/K/G owner-transfer adapter 和 `GSelectionState`，先做 K-only 对照、无新增参数的迁移恢复和 fail-closed 负例；只有迁移不劣且 G 不再读取外部 target，才允许设计 P3.3 的 G 小步学习。P3.2 不进入结构成长、CUDA、IDE/provider 或客户端视觉路线。

## 19. P3.2 K→G owner-transfer preflight 结果（2026-09-11）

按 §18 的入场条件运行了 [P3.2 owner-transfer preflight](../../scripts/training/eval_taiji_m5_k_p3_2_owner_transfer.py)，产出 [P3.2 manifest](../manifests/taiji_m5_k_p3_2_owner_transfer_manifest_v1.json) 和 [P3.2 报告](../../reports/taiji_m5_k_p3_2_owner_transfer_20260910.json)。本轮固定 P3.0 continuation parent、P3.1 manifest 和 P2.7 的 4 条 holdout；K1 只产生 inherited goal/content candidate，`GSelectionState` 持有最终选择，外部 goal/content target 只作为标签审计，未进入运行时。没有调用 `fit`、没有新增参数、没有读取 sealed payload。

| Gate | 结果 | 证据含义 |
|---|---:|---|
| K-only 与 owner-transfer 的选择/输出一致 | 通过 | 4/4 holdout 的 K1 selection、K2 output、safe abstention、Workbench 全部等价，owner-transfer 没有破坏已有能力 |
| 运行时外部 target 隔离 | 通过 | G 只消费 K candidate/evidence；`external_target_used=false`，目标标签没有成为模型输入 |
| checkpoint / 独立恢复 | 通过 | uninterrupted、事件边界、case 边界和两条 resumed 轨迹的 worker/budget/RNG/cursor/logical digest 一致 |
| fail-closed / rollback | 通过 | tampered cursor、错误 P3.0 base、错误 P3.2 manifest、错误 owner mask 全部拒绝，rollback 可恢复 |
| 训练与参数边界 | 通过 | `fit_called=false`、`training_performed=false`、K 参数未增长、`can_promote=false` |

P3.2 的结论是“选择所有权已经可以从 K 迁移到 G，且不损害当前行为”，不是“G 已经学会选择”。因此下一步唯一执行项改为 P3.3：冻结 P3.2 的 K，只建立有正/干扰/拒绝候选的 G candidate-set contract，先做 data-signal canary，再决定是否允许 G-only 小步更新。P2.7 holdout 必须保持 untouched test；训练前后都必须通过 checkpoint 保存、独立恢复、lineage、rollback 和篡改拒绝。若候选标签只是复制 K1 原选择，必须停止并重设计目标，不以 identity fit 冒充能力提升。

## 20. P3.3 G candidate data-signal canary 结果（2026-09-11）

按 §19 的入场条件运行了 [P3.3 G candidate data-signal canary](../../scripts/training/eval_taiji_m5_k_p3_3_g_signal_canary.py)，产出 [P3.3 manifest](../manifests/taiji_m5_k_p3_3_g_candidate_manifest_v1.json) 和 [P3.3 报告](../../reports/taiji_m5_k_p3_3_g_signal_canary_20260911.json)。本轮固定 P3.2 的 K worker，重建 P1 v2 的 460 条数据但只选 40 条 train、10 条 validation 进入 G 合同；P2.7 四条 holdout 没有进入 fit，未读取 sealed payload，也没有调用任何 fit。

| Gate | 结果 | 证据含义 |
|---|---:|---|
| 数据重建 / split 隔离 | 通过 | P1 期望/重建均为 460 条、`mismatch_count=0`；G train/validation 为 40/10，project/path 分离 |
| 候选完整性 | 通过 | 每条至少两个候选；低证据行使用 `abstain` 与 `reobserve`，高证据行保留 K 提案与 `abstain` |
| 目标与信号 | 通过 | target coverage 全部通过，目标同时包含 `pair` 与 `abstain`，20 条有非零 K score margin，30 条有竞争干扰项 |
| 运行时标签隔离 | 通过 | inference payload 不含 `target_candidate_id`/`target_kind`；`external_target_used=false` |
| checkpoint 入场 | 通过 | P3.2 K1/K2 保存、独立进程 restore 和 digest 校验通过；没有 K 参数变化 |

本轮只证明“可以开始设计 G-only 学习”，不证明 G 已产生能力增益。候选集合目前主要是“K 提案 vs 安全 abstain/reobserve”的二选一，避免把简单的候选复制误报为新能力。下一步唯一执行项改为：冻结 P3.2 K，保存并独立恢复空白 G child，只训练 G 的候选选择器，再以 K-only、zero-step owner-transfer、trained-G 三臂做保持与动作验证；任何 checkpoint、旧类保持或安全出口退化都必须回滚。

## 21. P3.3 G-only learning Gate 结果（2026-09-11）

按 §20 的 data-signal 入场条件运行了 [P3.3 G-only learning Gate](../../scripts/training/eval_taiji_m5_k_p3_3_g_learning.py)，产出 [P3.3 G-only 报告](../../reports/taiji_m5_k_p3_3_g_learning_20260911.json)。本轮继续冻结 P3.2 的 K1/K2，只对 G 做 320 steps、13 参数的小步 fit；train/validation 使用 P3.3 candidate manifest，P2.7 的 4 条 holdout 只做最终验证，没有进入 fit，未读取新的 sealed payload。

| Gate | 结果 | 证据含义 |
|---|---:|---|
| K parent 保存、独立恢复、训练前后 digest | 通过 | K1/K2 的 checkpoint 在 G 训练前后均可独立恢复，K1/K2 digest 完全不变 |
| G zero-step / trained checkpoint | 通过 | 空白 G 与训练后 G 均可内容寻址、独立恢复；训练后 digest 与 zero-step 不同 |
| G-only 训练边界 | 通过 | 仅 fit 40 条 train candidate set、320 steps、13 个 G 参数；validation/holdout 未 fit，外部 target 未进入 runtime |
| lineage / 篡改 / 错误 parent 拒绝 | 通过 | tampered checkpoint、错误 K lineage 均 fail-closed |
| P2.7 holdout 与 Workbench | 通过但无增益 | 4 条 holdout 的 fit count 为 0，四类 Workbench 行为保持通过 |
| 行为增益 | 未通过 | train 40/40、validation 10/10、holdout 4/4 上，K-only、zero-step、trained-G 的选择与动作均完全一致 |

本轮的正确结论是“G 的工程学习闭环成立，但当前候选信号没有让 G 改变行为”，不是“G 已经产生选择能力”，也不是“继续堆同样 epoch 就会自然获得能力”。当前候选主要仍是 K proposal 与 `abstain`/`reobserve` 的安全二选一；G 的更新虽然改变了参数，却没有改变三臂轨迹，因此 `can_promote=false`，不追加同质训练轮数。

这次实验同时修复了候选身份的审计漏洞：candidate identity 已绑定 split/course/index/project/template/input digest，40 条 train 不再因重复观测折叠成少数候选集合；manifest 的 `experience_identity_unique` 与 `candidate_set_digest_unique` 均通过。该修复证明数据没有被错误去重，但仍没有证明候选之间存在足够的独立行为差异。

因此下一步唯一执行项改为 P3.4 行为差异候选信号 Gate：冻结当前 K、zero-step G 和 trained-G，不追加同质 epoch；重新构造包含 K1/K2 proposal、合法 score-grid counterfactual、`abstain`/`reobserve` 及按场景启用的 recovery/clarify 候选集合，用隔离 Workbench/action contract 的 snapshot match、route/parameter validity、执行成功、世界状态一致性和安全出口生成 utility。先只做 train/validation data-signal canary，要求出现非零 utility margin 和 K/G 可解释分歧；信号不足则停在数据/目标重设计，不进入 fit。只有 P3.4 通过后，才允许再次运行 G-only fit，并用 K-only、zero-step、trained-G 三臂验证 contested cohort 的真实收益、旧类保持、安全出口和恢复链路。

## 22. P3.4 behavior signal Gate 结果（2026-09-11）

按 §21 的停止条件运行了 [P3.4 behavior signal canary](../../scripts/training/eval_taiji_m5_k_p3_4_behavior_signal_canary.py)，产出 [P3.4 manifest](../manifests/taiji_m5_k_p3_4_behavior_manifest_v1.json) 和 [P3.4 报告](../../reports/taiji_m5_k_p3_4_behavior_signal_20260911.json)。本轮固定 P3.2 K1/K2，未调用 `fit`，未读取 sealed payload，也没有把 P3.3 trained-G 覆盖为新的能力基线；候选行为由隔离 native Workbench/action contract 的快照、路由/参数、安全出口和实时 observation 一致性组成。

| Gate | 结果 | 证据含义 |
|---|---:|---|
| 数据重建 / split | 通过 | P1 v2 重建合同通过；40 条 train、10 条 validation，project/path 隔离 |
| 候选覆盖 | 通过 | 每条 2–6 个候选；角色为 `abstain`、`proposal`、`reobserve` |
| 行为信号 | 通过 | 40/50 条有非零 utility margin；40/50 条与 K-only 行为分歧；行为 digest 唯一 |
| reobserve 目标 | 通过 | 30 条 target 为 typed `reobserve`；它表示安全的重新观察步骤，不表示 proposal 已执行成功 |
| checkpoint / 运行时隔离 | 通过 | K parent 独立恢复、K digest 不变、runtime 不带 target/utility、P2.7 holdout untouched |
| 训练 / promotion | 未执行 | `training_performed=false`、`fit_called=false`、`can_promote=false`；`can_start_g_fit=true` 只表示满足进入下一 Gate 的条件 |

审计发现：10 条 B/D 场景的候选静态 semantic slots 与实时 Workbench observation 不一致，导致候选 utility 同为 0、`utility_margin=0`；它们通过了 artifact 完整性，但只是确定性 tie-break，不能作为 G 的监督信号，也不能据此宣称行为学习。剩余非零 margin cohort 才是 P3.5 的 fit-eligible 集合。`reobserve` 目标必须在动作边界投影为不可写、可往返的 `ReadOnlyAbstention(next_step="workspace.list")`，不能把 list 结果直接当作当前样本的 oracle target。

P3.4 的结论是“行为标签合同终于能产生可解释分歧”，不是“G 已经学会行为选择”。因此 `can_promote=false` 保持不变，不能追加同质 epoch。下一步唯一执行项改为 P3.5 reobserve-aware G-only learning Gate：仅用非零 margin cohort 训练 G，训练前验证 checkpoint 保存/独立恢复，训练后以 K-only、zero-step G、trained-G 三臂验证 contested cohort 的真实收益、旧类保持、安全 reobserve projection、P2.7 holdout、K digest 和 lineage；任一安全/保持/恢复条件退化则回滚并停止 promotion。

## 23. P3.5 reobserve-aware G-only learning Gate 结果（2026-09-11）

按 §22 的入场条件运行了 [P3.5 G-only learning](../../scripts/training/eval_taiji_m5_k_p3_5_g_learning.py)，产出 [P3.5 fit manifest](../manifests/taiji_m5_k_p3_5_g_learning_manifest_v1.json) 和 [P3.5 报告](../../reports/taiji_m5_k_p3_5_g_learning_20260911.json)。本轮加载 P3.4 behavior artifact，只选择 `utility_margin > 1e-9` 的 32 条 train candidate 做 G-only fit；8 条非零 margin validation、8 条 zero-margin train、2 条 zero-margin validation 和 P2.7 四条 holdout 均未进入 fit。

| Gate | 结果 | 证据含义 |
|---|---:|---|
| G-only 训练边界 | 通过 | 13 参数 G、256 steps/8 epochs；K1/K2 不可训练，validation/holdout 不 fit，runtime 不读 target/utility |
| checkpoint / lineage | 通过 | G zero-step/trained checkpoint 独立恢复；tamper/wrong-parent 拒绝；训练前后 K1/K2 digest 完全不变 |
| zero-margin 排除 | 通过 | train 8 条、validation 2 条 tie 样本均排除；fit manifest 只含 32 条非零 margin train digest |
| reobserve action boundary | 通过 | 30/30 behavior target、所有实际 reobserve selection 均成为可往返 `ReadOnlyAbstention(next_step="workspace.list")`，无 `ActionIntent` |
| P2.7 holdout / Workbench | 通过 | holdout fit count 为 0；trained-G 的四条 Workbench 结果保持 `4/4` |
| contested behavior gain | 通过 | contested utility 从 zero-step `16.5` 到 trained-G `30`；behavior target hit 从 `0` 到 `30` |
| promotion | 未通过/未开放 | `can_promote=false` 保持；当前收益仍需独立 project/path holdout 验证，不能直接进入 P4 |

P3.5 的结论是“在冻结 K、排除零边际平局并保留安全动作边界的条件下，G-only 学习确实改变了 contested 行为”，这已经超出 P3.3 的参数变化但行为不变；它仍不是通用智能、结构成长或开放泛化证据。下一步唯一执行项改为 P3.6 独立行为 holdout 与保持 Gate：至少 2 个新 project、4 条新 path，validation-only，重新计算 behavior utility 和 reobserve projection，并以 P2.7、P1 五类旧类和安全出口做非劣对照。若新身份上无收益或旧类退化，保持 `can_promote=false`，不追加同质 epoch、不进入 P4。

## 24. P3.6 独立行为 holdout 与保持 Gate 结果（2026-09-11）

按 §23 的入场条件运行了 [P3.6 behavior holdout](../../scripts/training/eval_taiji_m5_k_p3_6_behavior_holdout.py)，产出 [P3.6 manifest](../manifests/taiji_m5_k_p3_6_behavior_holdout_manifest_v1.json) 与 [P3.6 报告](../../reports/taiji_m5_k_p3_6_behavior_holdout_20260911.json)。本轮不调用 `fit`，只加载 P3.5 zero-step/trained-G，在两个新 project、四条全新 path 上重新生成候选并通过真实只读 Workbench 计算 behavior utility；P3.4/P3.5 train/validation、P2.7 holdout 均未作为新训练输入。

| Gate | 结果 | 关键实测 |
|---|---:|---|
| 新身份与 artifact 隔离 | 通过 | 2 个新 project、4 条新 path；candidate-set、behavior、observation digest 唯一，utility-margin record 内容寻址 |
| 三臂行为泛化 | 通过 | K-only/zero-step utility `2.65`、target hit `1/4`；trained-G utility `4.0`、target hit `4/4`；trained-G 改变 3 条 contested 行为 |
| reobserve action boundary | 通过 | 3 条 target reobserve、3 条 selected reobserve 均为可往返 `ReadOnlyAbstention(next_step="workspace.list")`，无 `ActionIntent`，snapshot match 全部通过 |
| 五类旧类与安全保持 | 通过 | P1 A/B/C/D/R 五类均出现；trained-G target hit 不低于 zero-step；低证据 proposal 违规 `0` |
| P2.7 / checkpoint / lineage | 通过 | P2.7 trained-G Workbench `4/4`；K1/K2 digest 前后相同；K/G 独立恢复、lineage、rollback 均通过 |
| promotion | 未通过/未开放 | `can_promote=false` 保持；这是局部行为选择跨身份泛化，不是通用智能或结构成长证明 |

P3.6 的结论是：P3.5 的 G-only 行为变化不只在原 artifact 上重放，在这组未见 project/path 上也能复现，同时安全 reobserve 和旧类保持没有退化。但样本规模仍是 4 条 holdout，不能直接解冻 promotion 或宣称开放泛化。P3 阶段收束，下一步唯一执行项改为 P4 容量压力与继承式结构成长 Gate：先用 validation-only 扫描证明固定容量存在可重复瓶颈，再比较继承式 dynamic-growth、强 fixed-large 与结构 lesion；若无真实压力则停止结构增长，不能为了“神经元扩张”而人为制造需求。

## 25. P4.0 固定容量压力扫描结果（2026-09-11）

按当前执行计划运行了 [P4.0 capacity-pressure scan](../../scripts/training/eval_taiji_m5_k_p4_0_capacity_pressure.py)，产出 [P4.0 manifest](../manifests/taiji_m5_k_p4_0_capacity_pressure_manifest_v1.json) 与 [P4.0 报告](../../reports/taiji_m5_k_p4_0_capacity_pressure_20260911.json)。本轮严格 validation-only：没有调用 `fit`，没有修改 K1/K2，没有新增 G 参数，没有读取 sealed payload；前一轮失败的临时 run 目录已清理，最终 run 目录按 `.gitignore` 保留为本地可复核产物。

| Gate | 结果 | 关键实测 |
|---|---:|---|
| 新身份与候选合同 | 通过 | 5 个新案例、3 个新 project、20 个 candidate/behavior digest 唯一；宽度 `2/4/8/12` 各有 5 个案例 |
| checkpoint / lineage | 通过 | K1/K2、zero-step G、trained-G 独立恢复；G 对 P3.2 manifest lineage 校验通过；K1/K2 digest 前后不变 |
| 固定 G 宽度压力 | 观察到 | trained-G residual 分别为 width 2=`0.32`、4=`0`、8=`0.54`、12=`0.59`；target hit 为 `0.6/1.0/0.4/0.2` |
| 长序列压力 | 未观察到 | sequence length `1/4/16` 的 utility 均为 `0.6375`，没有随重复长度下降 |
| 特征碰撞 | 未观察到 | 四个宽度的 feature collision rate 均为 `0` |
| 结构成长 / promotion | 未开放 | `growth_admitted=false`、`can_promote=false`；fixed-large 尚未具备公平的 G owner/readout 合同 |

这个结果只能证明“固定 G 在候选集合变宽、跨案例 proposal 竞争时出现可重复的选择压力”。width 4 完整通过而 width 8/12 退化，且序列长度没有进一步退化，说明当前首先要排查候选集上下文/竞争特征、排序与归一化合同；它还不能单独证明增加神经元或扩大拓扑会解决问题。P4.0 的 `scan_passed=true` 只表示验证合同、隔离、恢复和诊断运行完成；`growth_admitted=false` 是有意保持的安全状态。

因此下一步从“直接进入结构成长”修订为 **P4.1 压力归因与公平容量对照 Gate**：在相同 P4.0 candidate/behavior artifact 上建立同输入/读出合同的 fixed-large、结构 lesion 和候选集 context-aware 对照，validation-only 比较 width `2/4/8/12` 的 utility、target hit、旧类保持、安全投影、Workbench、参数/资源和独立恢复；至少两个 deterministic seed。只有 fixed-large/lesion 排除上下文缺陷并稳定证明 fixed-small 瓶颈后，才允许设计继承式结构成长；否则先修 G 的输入合同，不扩拓扑。

## 26. P4.1 候选集上下文与公平容量合同预检结果（2026-09-11）

按 P4.0 结果修订后的计划运行了 [P4.1 context-contract preflight](../../scripts/training/eval_taiji_m5_k_p4_1_context_contract.py)，产出 [P4.1 manifest](../manifests/taiji_m5_k_p4_1_context_contract_manifest_v1.json) 与 [P4.1 报告](../../reports/taiji_m5_k_p4_1_context_contract_20260911.json)。这一步仍是 validation-only：没有调用 `fit`，没有改写 P3.5 G/K parent，参考 arm 只做零初始化合同和独立恢复。

| Gate | 结果 | 关键实测 |
|---|---:|---|
| 来源与 lineage | 通过 | P4.0 manifest/report digest 一致；P3.2 G lineage、trained-G 独立恢复通过；`growth_admitted=false` 保持 |
| 候选集上下文 | 通过 | 20 个集合，width `2/4/8/12` 各 5 个；9 个 context feature；每集合内部 context collision `0`；输入显式排除 target/utility |
| fixed-large reference | 通过预检 | 22 个参数（当前 trained-G 为 13）；candidate 12 维 + context 9 维 + bias；独立进程 roundtrip 通过 |
| context lesion | 通过预检 | 屏蔽 context 后 effective 参数为 13；独立恢复通过；20 个集合选择与当前 G 完全一致 |
| 归因 | 未定 | `inconclusive`；零 fit 参考臂不能从同一验证集合重放推断学习收益或容量上限 |

P4.1 的实际价值是把 P4.0 模糊的“候选变宽”拆成两个可审计对象：G 当前的逐候选 12 维输入，以及候选数量/角色比例/score 分布/相对排名组成的 9 维候选集上下文。22 参数 reference 只是一个可恢复容量合同，不是已训练模型，也不能当作神经元成长。下一步因此改为 **P4.2 隔离训练与公平容量归因 Gate**：在与 P4.0/P4.1 全部 disjoint 的 train/validation/holdout 上，分别训练 fixed-small、context-aware-small 和 fixed-large；只有新的 holdout 与 fixed-large/lesion 对照共同支持固定容量瓶颈，才保留结构成长假设。

## 27. P4.2 隔离训练与公平容量归因结果（2026-09-11）

按 §26 的入场条件运行了 [P4.2 capacity attribution](../../scripts/training/eval_taiji_m5_k_p4_2_capacity_attribution.py)，产出 [P4.2 manifest](../manifests/taiji_m5_k_p4_2_capacity_attribution_manifest_v1.json) 与 [P4.2 报告](../../reports/taiji_m5_k_p4_2_capacity_attribution_20260911.json)。本轮第一次允许在新 child 上 fit，但仍冻结 P3.5/P4.0 parent、K1/K2 和结构成长；两个 deterministic seed 使用完全 disjoint 的 A/B/C/D/R train/validation/holdout，14 条新 train candidate set 进入 fit，三臂均做零步/训练后 checkpoint 保存和独立进程恢复。

| Gate | 结果 | 证据含义 |
|---|---:|---|
| 数据、身份与训练边界 | 通过 | 五类 train/validation/holdout 均存在，project/path/candidate/behavior digest 与旧实验隔离；validation、holdout、P4.0/P4.1 和外部 target/utility 没有进入 fit |
| checkpoint / lineage | 通过 | fixed-small、context-aware-small、fixed-large 的零步和训练后 checkpoint 全部 roundtrip/独立恢复；parent 未覆盖；K1/K2 digest 不变 |
| 新 holdout 容量收益 | 未通过 | fixed-small 与 context-aware-small 平均 utility 均为 `0.68`，fixed-large 为 `0.65875`；没有稳定容量增益，seed 1 fixed-large 还有 6 次 safe-selection violation |
| 旧行为保持 | 未通过 | parent retention 为 `4/4`、utility `4.0`；fixed-small/context-aware-small 训练后两个 seed 都降到 `3/4`、utility `0.8`；fixed-large 只在 seed 1 恢复到 `4/4`，跨 seed 不稳定 |
| 归因与结构成长 | 关闭 | `attribution=inconclusive`、`experiment_passed=false`、`growth_admitted=false`、`can_promote=false` |

P4.2 的正确解释是：当前实验还不能区分“上下文表征没有收益”和“固定容量不足”，因为所有训练臂先在保持合同上失败；更直接暴露的工程/学习问题是增量更新造成旧行为遗忘。22 个存储参数并未自动带来能力，参数规模也不能替代公平输入、训练保持和稳定 seed。下一步从容量归因改为 **P4.3 保持约束下的增量学习 Gate**：在 13 参数 fixed-small 上比较 new-only 与旧行为 rehearsal，使用重新生成的 fresh retention holdout；保持恢复前不进入 dynamic growth。

## 28. P4.3 保持约束下的增量学习结果（2026-09-11）

按 §27 的修订条件运行了 [P4.3 retention incremental](../../scripts/training/eval_taiji_m5_k_p4_3_retention_incremental.py)，产出 [P4.3 manifest](../manifests/taiji_m5_k_p4_3_retention_incremental_manifest_v1.json) 与 [P4.3 报告](../../reports/taiji_m5_k_p4_3_retention_incremental_20260911.json)。本轮固定 13 参数 P3.5 trained-G parent，生成与 P4.0/P4.2/P3.6 全部 disjoint 的新 train/validation/holdout/fresh-retention；P3.6 的 4 条旧 holdout 只转换为显式 rehearsal source，未在训练后再作测试。两种 child training 使用相同的新 train、epoch/学习率/seed/CPU 预算。

| Gate | 结果 | 证据含义 |
|---|---:|---|
| 数据与训练边界 | 通过 | 新四类 split 各 20 条、14 条新 train fit；project/path/candidate/behavior digest 隔离；validation、new holdout、fresh retention 均未 fit |
| checkpoint / lineage | 通过 | 两臂零步/训练后 checkpoint 独立恢复；tamper、wrong parent、rollback 均拒绝/通过；13 参数与 K1/K2 未改变 |
| 新任务与 fresh retention | 通过保持 | 两个 seed 的 new-only 与 rehearsal-mix 都为 new holdout utility `0.68`、target hit `0.6`；fresh retention 都为 utility `0.68`、target hit `0.6`，safe selection violation `0` |
| rehearsal 机制归因 | 未通过 | new-only 与 rehearsal-mix 在 holdout、fresh retention、safe projection 上逐 seed 完全相同；`rehearsal_specific_gain=false` |
| 结果出口 | 未晋级 | `outcome=signal_insufficient`、`experiment_passed=false`、`growth_admitted=false`、`can_promote=false` |

P4.3 不能解释为“rehearsal 已修复 P4.2 的遗忘”：它只证明在本轮 fresh retention 分布上，训练后没有出现低于 parent 的保持退化；new-only 也获得了完全相同的结果，因此没有 rehearsal 的可分离贡献。P4.2 中基于旧 P3.6 retention 的 `3/4` 退化不能直接拿来与本轮 fresh retention 混合比较，因为旧集合已经作为 rehearsal 输入。下一步改为 **P4.4 保持身份/结构校准 Gate**：使用候选数量、角色组成和难度同构但 project/path 全新的 sibling retention，validation-only 评估 parent 与 P4.2/P4.3 child，判断退化能否跨身份复现；在此之前不扩容、不 promotion。

## 29. P4.4 保持身份与结构校准结果（2026-09-11）

按 §28 的边界运行了 [P4.4 retention identity calibration](../../scripts/training/eval_taiji_m5_k_p4_4_retention_identity_calibration.py)，产出 [P4.4 manifest](../manifests/taiji_m5_k_p4_4_retention_identity_calibration_manifest_v1.json) 与 [P4.4 报告](../../reports/taiji_m5_k_p4_4_retention_identity_calibration_20260911.json)。本轮严格 validation-only：只从 P3.6 提取候选数量、角色组成、置信度分桶和安全投影类型，生成 2 个新 project、4 条新 path 的 sibling retention；没有复制 P3.6 的 path、target、utility 或 exact candidate digest，没有调用 `fit`，也没有覆盖历史 checkpoint。

| Gate | 结果 | 关键实测 |
|---|---:|---|
| 来源、结构与身份 | 通过 | P3.6/P4.2/P4.3 source chain、manifest/report digest 通过；4 条 sibling 结构均为一条 6-candidate `proposal×4 + abstain + reobserve` 和三条 2-candidate `abstain + reobserve`；2 个 project、4 条 path 全部为新身份 |
| checkpoint / lineage | 通过 | P3.5 parent、P4.2 三臂、P4.3 两臂的全部 seed checkpoint 均独立恢复且 lineage 有效；P4.2/P4.3 parent 未被覆盖 |
| parent sibling 基线 | 通过 | parent utility `1.0`、target hit `4/4`、safe violation `0`、reobserve projection 通过、Workbench success `1` |
| P4.2 退化复现 | 通过 | P4.2 历史退化的 5 个 arm/seed 在 sibling 上全部复现；对应多数 arm/seed utility `0.8`、target hit `3/4`，P4.2 seed-1 fixed-large 为非退化对照 |
| P4.3 退化复现 | 通过 | new-only 与 rehearsal-mix 两个 seed 均为 utility `0.8`、target hit `3/4`；rehearsal 没有消除同构 sibling 上的保持退化 |
| 结果出口 | 已分类但未晋级 | `outcome=retention_failure_reproduced`、`training_performed=false`、`fit_called=false`、`experiment_passed=false`、`growth_admitted=false`、`can_promote=false` |

P4.4 的证据把 P4.3 的表面矛盾拆开了：P4.3 fresh retention 的 `0.68` 通过不能归因给 rehearsal，也不能代表保持问题不存在；当评估集合恢复为与 P3.6 相同的候选结构/难度而换成全新身份时，P4.2 历史退化和 P4.3 两臂退化都出现。当前最强结论是“训练后更新会破坏这一类保持合同，且现象可跨身份复现”，而不是“需要增加神经元”。下一步只允许进行 P4.5 保持约束与更新规则对照；dynamic growth、promotion 和 P5 外围继续冻结。

## 30. P4.5 保持约束与更新规则对照结果（2026-09-11）

按 §29 的唯一下一步运行了 [P4.5 update-rule Gate](../../scripts/training/eval_taiji_m5_k_p4_5_update_rule_gate.py)，产出 [P4.5 manifest](../manifests/taiji_m5_k_p4_5_update_rule_gate_manifest_v1.json) 与 [P4.5 报告](../../reports/taiji_m5_k_p4_5_update_rule_gate_20260911.json)。本轮固定 13 参数 G、K1/K2、候选输入和 selection threshold；P4.4 sibling 只提供结构合同，P4.5 重新生成与历史 project/path 全部 disjoint 的 20 条 train、20 条 validation、20 条 holdout、4 条 rehearsal 和 4 条 fresh retention。两个 deterministic seed 比较 `new-only`、`rehearsal-interleaved` 与 parent 范数 15% trust-region 的 `constrained-update`。

| Gate | 结果 | 关键实测 |
|---|---:|---|
| 来源、身份与结构 | 通过 | 五类 train/validation/holdout 覆盖；retention 结构 digest 与 P4.4 相同；所有新 project/path/candidate/behavior digest 隔离；retention 与 P4.4 sibling 均未进入 fit |
| checkpoint / lineage | 通过 | 三臂两个 seed 的零步/训练后 checkpoint 全部独立恢复；tamper 拒绝；parent 未覆盖；参数始终 13，K1/K2 digest 不变 |
| new-only vs rehearsal | 无差异 | 两 seed 两臂 new holdout 都为 utility `0.68`、target hit `0.6`、safe violation `0`；fresh retention 都为 utility `0.8`、target `3/4`，没有 rehearsal-specific gain |
| constrained seed-0 | 局部保持通过 | fresh retention utility `1.0`、target `4/4`、safe violation `0`；但 new holdout 只有 parent 的 utility `0.6375`、target `0.55`、safe violation `6` |
| constrained seed-1 | 局部新任务通过 | new holdout utility `0.68`、target `0.6`、safe violation `0`；fresh retention utility `0.8`、target `3/4`，仍低于 parent |
| 结果出口 | 未晋级 | `outcome=update_rule_unresolved`、`training_performed=true`、`experiment_passed=false`、`growth_admitted=false`、`can_promote=false` |

P4.5 的证据说明，简单把旧样本交错进训练仍没有独立贡献；固定参数距离的 trust-region 可以在一个 seed 上回到 parent 的保持表现，但在该 seed 上失去新任务增益，另一个 seed 则在保持和新任务之间重新出现冲突。因此当前待解决的不是“有没有一个更小的固定半径”，而是如何在函数行为层面同时约束 parent 能力和学习新任务。下一步改为 **P4.6 功能性 parent-preserving objective 对照 Gate**：使用与 P4.5 全部 disjoint 的 constraint-cohort，以 parent 输出作功能性保持项，不把评估 target/utility 写入输入；若两个 seed 仍不能同时满足新任务与保持，停止继续调参并维持 dynamic growth/promotion/P5 冻结。
