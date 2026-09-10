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
