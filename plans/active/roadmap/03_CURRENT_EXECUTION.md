# Seed / Taiji 唯一执行计划

> 最新修订：2026-09-10；依据当前工作树基线 `a78eae17` 及实际实验报告。
>
> **执行优先级覆盖声明：仅下方「2026-09-10 修订执行序列」决定当前顺序。其后旧正文保留为设计/推进历史，其中所有“当前唯一下一步”、formal 开跑许可和完成状态必须结合本次复审解释，不再单独授权执行。**
>
> 当前主线：**M4.V2 继承式成长重构**。M4.R0～R12 已作为 F1 fixed-capacity 实验证据归档，但其对“容量、分布和全部学习规则”的过度外推已撤销。当前架构依据为 [Taiji 继承式成长架构 v2](../architecture/TAIJI_CONTINUAL_DEVELOPMENT_V2.md)，历史证据边界见 [M4 固定容量证据链复盘](../../reference/M4_FIXED_CAPACITY_EVIDENCE_REVIEW_2026_09_09.md)。
>
> 本文是唯一执行顺序和唯一“下一步”来源。其他 active 文档定义长期需求或架构，不发布并行任务；archive/reference 中的旧“下一步”全部失效。

## 2026-09-10 修订执行序列（唯一有效）

详细证据与源码定位见 [M4 v1/v2 实际结果复审](../../reference/M4_V1_V2_RESULT_REVIEW_2026_09_10.md)、[B3-K C-entry formal closure](../../reference/M4V2_B3_K_C_ENTRY_CLOSURE_20260910.md) 和 [B3-K capacity-parity audit](../../reference/M4V2_B3_K_C_CAPACITY_PARITY_AUDIT_20260910.md)。当前判断：v2 有机制和窄任务进步；未证明总体超过 v1。R4 比小容量对照好但未稳定超过 fixed-large，R5 候选否决；C-entry formal 已完成，candidate quality/resource Gate 通过，但 fixed-large 在 `0/9` 格更强。后续审计确认这次 strong-control 比较还混有容量和 update budget 差异，保持 `can_promote=false`，不把候选接入默认路径。

### A. 已完成：修正 R6 学习对照的评分语义

本轮没有重新训练，也没有覆盖旧报告；目标是让同一组行为证据得到诚实评分，并让学习实验必须真的更新模型。

1. `eval_taiji_m4v2_r6_formal_single_cell.py` 已拆开 `task_success`、`feedback_admitted`、`learning_update_applied` 与 `learning_eligible`。matched 的真实 Workbench 成功不再因为不接纳反馈而被记为 0；未执行的 frozen parent 记为 `None`，不伪造失败。
2. 新增 `test_m4v2_r6_measurement_semantics.py`，覆盖反馈解耦、未执行不归零、零更新不算学习和 wiring-canary 不得进入 learning-formal Gate；本轮 7 条语义测试及相关 18 条测试通过。
3. runner 已声明 `measurement_semantics` 与 `run_kind=wiring-canary`；aggregate 不再把 K3 lesion 的任务成功差当作学习增益，并会拒绝没有真实 candidate 参数更新的 formal 输入。旧九 cell 报告和 digest 未覆盖。
4. Workbench controlled canary 与 fixed-large canary 已显式记录 `task_executed`；stage/exchange 仍不算参数学习。Ruff、`py_compile`、`git diff --check` 已通过；未宣称远端 CI 全绿。

交付：评分实现、失败回归测试和计划状态已提交。旧报告尚未生成新版本解释报告；它们仍只能作历史证据，不能作学习 formal 证据。A 完成，进入 B；不单开新外围路线。

### B. 真实固定容量连续学习 pilot（当前阶段）

**B2 已完成：训练前 checkpoint 保存恢复预检通过。** `reports/taiji_m4v2_checkpoint_preflight_20260910.json` 使用 CPU 上的 model 17 parent 做了一个仅用于预检的小步 developmental update；它验证了 fast/slow、replay、owner lineage、内部 RNG、课程游标、无 optimizer 的局部更新状态、原子保存、fresh restore、断点续训一致性和 parent rollback。`can_start_training_pilot=true`，但 `research_course_executed=false`、`can_promote=false`。系统 TEMP/ACL 暴露的两个入口问题已修复，临时 checkpoint 已清理。

**B3-S/G slice 已完成，完整 B3 尚未完成。** 使用同一个不可变 model 17 parent（digest `3e1b39b68be25de672535939401f3c2cf0fb9229c99458237c888be9a50f6675`）在 CPU 上执行了真实 fast/slow 学习与 replay；报告为 [B3 pilot probe](../../../reports/taiji_m4v2_b3_pilot_probe_20260910.json)。fast+replay 消费 424 个真实 wake replay event，replay 后 fast 清零；相对 parent 的 S/G delta 为 `−0.300456/−0.083226`，相对 slow-only 为 `−0.005738/−0.002625`，相对 fast-only 为 `−0.007025/−0.004870`（mean-surprise 越低越好）。fresh restore、read-only、old owner 未写入和 rollback 均通过，`can_promote=false`。

这只证明当前 inherited fixed-capacity S/G 学习链在小型 synthetic course 上能产生真实更新和窄 holdout 改善；它没有 K phase、没有同权限 frozen K arm，也没有独立大规模 holdout，因此不能称完整 B3 或自然自进化。

**K continuation-learning contract 已完成，完整 B3-K 尚未完成。** 新增 `taiji/k_continuation.py`，把一条真实 K episode 固化为 content-addressed experience，并强制同一 parent、worker bundle、source manifest；train/holdout 按 experience、输入 digest 和 family 隔离。v1 只允许 `k1.semantic` 与 `k2.transition` 改变，`k3.outcome_projection` 与 Taiji parent 保持不变；当前 worker 的 detached local-delta 语义明确记录为无 optimizer state。新增 course roundtrip、family leakage、update receipt/K3 不可变测试，相关测试与 Ruff 通过。K1/K2 example 现已具备可恢复 payload 解析，供 continuation artifact 使用。

**B3-K 单步 pilot 已完成技术 Gate，但没有能力晋级。** 报告为 [B3-K single-step pilot](../../../reports/taiji_m4v2_b3_k_single_step_20260910.json)：同一 model 17 parent 上真实消费 1 条 train experience，K1/K2 各更新 1 步（receipt 合计 2），K3 未变；candidate checkpoint 保存/fresh-restore、parent 不变、adapter stage/rollback、train/holdout 隔离均通过。独立 holdout 的结构化准确率已经是 `1.0`，更新前后仍为 `1.0`，所以这次只证明 continuation 写入链路可用，不能证明泛化收益，`can_promote=false`。

**B3-K 非饱和 structured-loss diagnostic 已完成，仍不能晋级。** 报告为 [B3-K loss diagnostic](../../../reports/taiji_m4v2_b3_k_loss_diagnostic_20260910.json)：严格排除 M5.K2 第四个重复 observation 后使用 3 条 holdout；K1 fact/goal/content 与 K2 transition/goal/content 六个连续 MSE 分量全部下降，combined MSE 从 `0.01677758` 降到 `0.01569742`，delta `−0.00108017`。K1/K2 更新、K3/parent 不变、candidate 保存/fresh-restore、holdout 未训练和 rollback 均通过，但这仍是单 model seed、单 course seed 的窄证据，`can_promote=false`。

**B3-K 多 course-seed 稳定性复验已完成，但性能 Gate 未通过。** 修正后的 runner 固定同一 workspace/三条 holdout，只改变实际被消费的 train episode；course seed `0/1/2` 的 train episode digest 均不同，parent digest 相同，K1/K2 更新、K3 冻结、fresh restore、rollback 和 contract 均通过。真实 combined-MSE delta 分别为 `−0.00108017`、`+0.00005455`、`+0.00033079`；均值虽为 `−0.00023161`，最坏值为正，因此不能用均值掩盖课程敏感性。报告为 [B3-K loss stability](../../../reports/taiji_m4v2_b3_k_loss_stability_20260910.json)，`technical_gate_passed=true` 但 `performance_gate_passed=false`、`stability_gate_passed=false`，仍 `can_promote=false`、`can_start_r6_formal=false`。这次结果把问题从“是否能更新”收敛为“单条课程更新对 train episode 选择过敏”，不是 formal 证据。

**B3-K 课程敏感性单变量诊断已完成。** 在相同 model 17 parent、同一 workspace、同一 3 条 holdout 下，三条不同 train episode 的 train combined-MSE 都下降（`−0.00050056/−0.00021416/−0.00022227`），K1/K2 参数 delta norm 均非零；无更新对照的 holdout delta 全为零，排除了保存链或 scorer 漂移。结合稳定性结果，当前最合理解释是单条样本更新存在过拟合/跨 episode 干扰或梯度方向不一致；这仍是窄诊断，不宣称已区分全部原因。诊断字段已写入 [B3-K loss stability](../../../reports/taiji_m4v2_b3_k_loss_stability_20260910.json)，继续保持 `can_promote=false`。

**B3-K bounded multi-example 诊断已完成，但仍未过性能 Gate。** 只把每个 course 扩为 2 条真实、互不重复的 train experience，seed 组合为 `[0,1] / [1,2] / [2,3]`；parent、worker owner、学习率、holdout 和回滚边界保持不变。combined-MSE delta 为 `−0.00053848/+0.00019021/−0.00040499`，2/3 course 改善，最坏退化从单样本的 `+0.00033079` 降到 `+0.00019021`，均值为 `−0.00025109`，但仍不能用均值掩盖 seed 1 退化。报告为 [B3-K bounded batch](../../../reports/taiji_m4v2_b3_k_bounded_batch_20260910.json)，技术 Gate、no-update、保存恢复、K3 冻结和 rollback 通过，`performance_gate_passed=false`、`stability_gate_passed=false`、`can_promote=false`。

**B3-K bounded 三样本诊断已完成，但证据非判别。** 课程组合 `[0,1,2] / [1,2,3] / [2,3,4]` 的 train input/target digest 均不同，三组 raw combined-MSE delta 都是 `−0.00025813`；但三组 candidate worker bundle/参数 digest 完全相同，`candidate_updates_distinct=false`。因此技术链、保存恢复、K3 冻结、rollback 和 no-update 通过，但这不是 3 个独立学习结果，`performance_gate_passed=false`、`stability_gate_passed=false`、`can_promote=false`。报告为 [B3-K three batch](../../../reports/taiji_m4v2_b3_k_three_batch_20260910.json)。当前解释是 batch 局部更新在这组 synthetic 组合上出现梯度/表示聚合对称性，尚不能判断是数据对称还是 worker 更新规则过度压缩。

**B3-K 非滑动课程组合诊断已完成，仍未形成稳定证据。** 预注册组合 `[0,2,4] / [1,3,5] / [0,3,5]` 的 train input/target digest 都不同，三组 holdout delta 为 `−0.00025813/−0.00025813/−0.00063627`；但前两组 candidate update digest 完全相同，只有第三组不同。raw performance 全部改善，却因 `candidate_updates_distinct=false` 不能通过稳定性 Gate。报告为 [B3-K non-sliding](../../../reports/taiji_m4v2_b3_k_non_sliding_20260910.json)，当前结论是部分课程组合在现有 K1/K2 局部更新中不可辨识，不能继续用组合数量替代机制定位。

**B3-K 逐 episode update-signature 审计已完成，定位到 K1/K2 共同的更新碰撞。** 6 条 episode 的 semantic/transition 输入 digest 全部不同，但单步 candidate 成对碰撞：`0=3`、`1=4`、`2=5`；K1 与 K2 两个 worker 的 candidate checkpoint digest 都呈现同样的三组成对碰撞。三样本 batch candidate 均不等于任一单 episode candidate，说明 batch 还叠加了聚合效应。报告为 [B3-K update signature audit](../../../reports/taiji_m4v2_b3_k_update_signature_audit_20260910.json)，technical Gate 通过但 `can_promote=false`。

**B3-K K1/K2 feature-target collision audit 已完成，根因收敛到课程 target 多样性不足。** 6 条 episode 的实际 K1/K2 input tensor digest 全部不同，但 K1 fact/goal/content target tensor 与 K2 delta/goal/content target tensor 都按 `0=3、1=4、2=5` 成对重复；对应 K1/K2 parameter delta 也按同样分组碰撞。报告为 [B3-K feature-target audit](../../../reports/taiji_m4v2_b3_k_feature_target_audit_20260910.json)，technical Gate 通过但 `all_fit_tensor_signatures_distinct=false`、`can_promote=false`。因此先前 course-seed 结果测到的是 3 类 target 组合的排列/重复，不是 6 个独立学习目标；当前没有证据要求修改 local-delta 算子或 typed mask。

**B3-K target-aware Gate v1 已被撤销并由 v2 取代。** v1 使用 `experience.target_digest` 判定 target 多样性，而该摘要包含 episode/example ID；feature-target audit 已证明实际 K1/K2 target tensor 仍存在 `0=3、1=4、2=5` 碰撞，因此 v1 的“多样性通过”是语义误判。v1 报告保留为历史产物，不得作为 C 入口证据。

**B3-K target-aware course diversity Gate v2 已通过。** v2 只使用每条 train experience 的实际 K1 semantic target tensor digest 与 K2 transition target tensor digest 的组合摘要，另行保留 ID-bearing `train_experience_target_digests` 供审计。在 model 17 parent、固定 holdout、相同学习率和 worker owner 下，真实 target multiset 为 `A+B+C / A+A+B / A+B+B`；三组 candidate update digest 不同，combined-MSE delta 为 `−0.00025813/−0.00072424/−0.00034705`，均值 `−0.00044314`、最坏 `−0.00025813`，`technical_gate_passed=true`、`performance_gate_passed=true`、`stability_gate_passed=true`。报告为 [B3-K target-aware v2](../../../reports/taiji_m4v2_b3_k_target_aware_v2_20260910.json)。这证明课程多样性 Gate 在真实 tensor 语义下成立，但仍只有窄任务/validation 证据，`can_promote=false`，不能等同 C formal。

**B3-K target-aware model-seed v2 稳定性复验已完成。** 真实 model seed `17/23/31` 的 parent digest 分别为 `3e1b39…6675`、`8b46f3…822e4`、`c3e21d…7732`，三者独立；每个 seed 的 target-aware 三组 target-tensor composition、candidate namespace 与 worker attachment 正确，9 个 cell 的 technical/performance/stability Gate 全通过。所有 combined-MSE delta 均为负，整体均值 `−0.00044314`、最坏 `−0.00025813`；三个 seed 仍呈同构数值轨迹，因此还必须通过冻结后的 sealed-test 与 fixed-large 强对照，不能直接 promotion。报告为 [B3-K target-aware model seeds v2](../../../reports/taiji_m4v2_b3_k_target_aware_model_seeds_v2_20260910.json)，继续 `can_promote=false`、`can_start_r6_formal=false`。

**C 入口 manifest、sealed-test 与 fixed-large 控制已冻结/材料化。** 合同已写入 [C-entry evaluation manifest v1](../../manifests/taiji_m4v2_b3_k_c_entry_evaluation_v1.json)，sealed artifact 为 3 个独立 episode、9 个 observation，旧 R6 fixed-large 未被冒充为本次控制。新建的 `build_taiji_m4v2_b3_k_c_fixed_large.py` 已按 3 个 model seed × 3 个 course seed 生成 C-entry 专用控制：每格绑定同一 parent、同一 `A+B+C / A+A+B / A+B+B` target-tensor multiset、同一 validation/sealed artifact 和 C-entry source manifest；prefit save/fresh-restore、disk restore、K1/K2 两个顺序不同的 replica 及 replica update distinct Gate 均通过。输入预检报告 [C-entry input preflight](../../../reports/taiji_m4v2_b3_k_c_entry_input_preflight_20260910.json) 现为 `status=ready`、`fixed_large_ready=true`、`formal_input_ready=true`，但仍 `can_start_formal=false`、`can_promote=false`。

**C-entry sealed scoring canary 已完成，但不是 formal。** 只读脚本 `eval_taiji_m4v2_b3_k_c_sealed_scoring.py` 消费冻结后的 parent、candidate 和 C-entry fixed-large artifact，使用同一六分量 structured-loss scorer 对 validation 与 sealed 各 9 格评分；报告为 [C-entry sealed scoring canary](../../../reports/taiji_m4v2_b3_k_c_sealed_scoring_20260910.json)。candidate 相对 frozen parent 的 sealed combined-MSE 平均 delta 为 `−0.0004431358`，最坏为 `−0.0002581254`，说明 continuation 仍有真实改善；但 fixed-large 平均 delta 为 `−0.0011265067`，candidate 在 `0/9` 格胜过 fixed-large。因此“候选有效”成立，“候选胜过强容量对照”不成立；没有改阈值，也没有把 sealed 结果改写成 promotion 依据。该 canary 明确 `training_performed=false`、`resource_measurement_complete=false`、`formal_gate_passed=false`、`can_start_formal=false`、`can_promote=false`。

**C-entry formal execution 已完成并收束。** 报告为 [C-entry formal report](../../../reports/taiji_m4v2_b3_k_c_formal_20260910.json)：9/9 cell 的 technical、resource、candidate-quality Gate 全部通过；candidate sealed combined-MSE 相对 frozen parent 的平均 delta 为 `−0.0004431358`，最差为 `−0.0002581254`。但 fixed-large 平均 delta 为 `−0.0011265067`，candidate 在 `0/9` 格胜过 fixed-large，因此 `formal_gate_passed=true` 只表示 candidate 自身学习合同闭合，不表示 promotion；`can_promote=false` 保持。完整边界见 [B3-K C-entry formal closure](../../reference/M4V2_B3_K_C_ENTRY_CLOSURE_20260910.md)。

**C-entry capacity-parity audit 已完成。** 机器报告为 [capacity-parity audit](../../../reports/taiji_m4v2_b3_k_c_capacity_parity_audit_20260910.json)，结论为 `blocked-current-comparison-confounded`：9 格课程与推理 trace 一致，但 candidate/fixed-large 分别为 `19,332/38,664` 参数字节、`6/14,252` 实际更新步数、`2/9` 逻辑 checkpoint 写入数。candidate 的自身学习 Gate 仍成立，strong-control 因果解释暂不成立。

**上调 candidate 容量的 parity input preflight 已完成。** 合同为 [capacity-parity manifest](../../manifests/taiji_m4v2_b3_k_c_capacity_parity_v1.json)，机器报告为 [capacity-parity input preflight](../../../reports/taiji_m4v2_b3_k_c_capacity_parity_input_preflight_20260910.json)：9/9 fixed-large cell 满足 38,664 参数字节、14,252 实际更新步、9 个逻辑 checkpoint、同 parent/course 和 fresh-restore；candidate artifact 尚未生成，且不能直接复制 fixed-large replica。

**widened-candidate 路线设计已冻结：[M4V2_B3_K_C_WIDENED_CANDIDATE_DESIGN_20260910.md](../../reference/M4V2_B3_K_C_WIDENED_CANDIDATE_DESIGN_20260910.md)。** 路线为单实例双通道分解——每个可学习头（K1 三头 + K2 三头）从单 delta 通道扩展为「supervised（全量 train experience 监督 delta）+ k3-feedback（K3 admitted outcome projection delta）」双通道，readout 为异构信号通道之和；通道与 supervised 同形状，参数精确 ×2 = 9,666（38,664 字节，ratio 0.0%）。与 fixed-large 的路径差异：owner（feedback 通道绑定 K3 admitted outcome）、update（7,126 ticks × 2 通道 = 14,252 步精确相等）、readout（通道和 vs replica 平均）。K continuation contract 升 v2（update 触发视图扩展，仍无 optimizer state）。硬性 distinct 证据门：逐 cell 报告两通道参数差分范数与反馈子集覆盖率，退化（零差分或覆盖率 <50%）即停。

**当前唯一下一步**：实现 K continuation contract v2 与 widened-candidate artifact builder——(1) `taiji/k_continuation.py` v2 双通道语义 + receipt 扩展 + 定向测试；(2) 按 manifest `taiji-k-c-entry-parity-candidate-v1` 生成 artifact 并出参数核算表（核验 9,666 参数 / 38,664 字节）；(3) 三硬门机器核验（参数字节 / 14,252 更新步 / 9 checkpoint + distinct 证据 + fresh restore/rollback/parent/K3 不变）；核验通过前不训练、不读 sealed formal、不调学习率。

widened-candidate 已实现并构建（`taiji/k_widened.py` 的 `WidenedKBundle` + `build_taiji_m4v2_b3_k_c_widened_candidate.py`；通道 2 按设计 §4 修订条款落地为 **K3-anchored permuted-order**——projection 事实不在现有 13-fact 词汇内，扩词汇破坏精确 ×2，K3 语义如实降级为排序见证）。构建结果 `reports/taiji_m4v2_b3_k_c_widened_candidate_build_20260910.json`：**8/9 cell 通过**（38,664 字节 ratio 0.0%、14,252 步精确、fresh restore/parent/K3 全过、通道差分非零），**model31/course1 的通道差分为精确 0.0 → distinct 门诚实失败**。归因：(1) 该 cell 的 parent 在课程语料上已收敛，顺序通道无信息；(2) 口径发现——14,252 中 14,240 是 parent 继承步数（fixed-large 同口径），真实新增更新每 cell 仅 12 步，parity 的「更新步数」门本身混淆了继承与新增。按停止线保留 8/9 证据，不放宽 distinct 门。

**当前唯一下一步**：parity 口径修正决策——先冻结「实际新增参数更新步数」的独立计量（把继承步数从 parity 硬门中分离，fixed-large 与 candidate 同口径重述），再对 model31/course1 的零差分归因（parent 收敛 → 顺序通道无信息）选择：失败 episode 进入课程（S6B 失败准入遗产）或更强的新增预算课程；两者都需 manifest 修订预注册。决策前不进入 sealed 评分、不训练、不改 distinct 门。

**B1 数据与量尺冻结。** 复用 R2 fast/slow + replay 和现有 K worker 训练路径，先梳理参数 owner、调用点、训练反馈到数值更新链。不接外部 provider 代替 Taiji 学习。建立 train/validation/sealed-test 三份分离集合；K 按项目/任务模板隔离，不能仅改文件名。逐 phase 记录实际消费内容 digest。课程 seed 必须改变实际经历顺序或组合，不能只改变标签。

**B2 保存先于训练。** 先在 CPU 检查父 checkpoint 可恢复；保存 fast/slow、replay、学习器状态（若使用 optimizer 则含其状态）、RNG、课程游标和 owner lineage。新进程恢复后预测等价；单步更新后保存/恢复/再更新，与不中断轨迹在预先声明容差内一致。检查磁盘空间、原子写和 rollback；任何失败禁止长跑，不覆盖父代。

**B3 一个模型 seed、一个课程的诊断 pilot。** 同一不可变父代派生三臂：冻结学习但保持同等执行权限；现有固定容量 continuation；固定容量 fast/slow + 真实 replay。共同任务、输入、解码与训练预算；replay 消费量另外计入成本。逐 S→G→K 阶段训练，不从头重建父模型；评分时只读且关闭测试反馈写入。

每阶段记录：S/G/K 累计 holdout 矩阵、parent 与上一阶段差值、新任务收益、最坏旧域退化、实际更新步数、owner 参数数值差、checkpoint 恢复结果、训练/推理分开耗时和内存。S/G 使用各自声明的 surprise 指标；K 使用独立行为成功率，不与反馈接纳混合。

验收：能展示“训练前→更新→未见任务评分→保存恢复”的完整链。pilot 仅用于检查可学习性和估算预算，不晋级。若有更新无泛化收益，先区分标签/反馈质量、表征、优化或遗忘，单变量修订；不可直接追加结构、provider 或九 cell 报告。

### C. 冻结合同后开展正式学习比较（B 后）

1. 用 pilot/validation 确定任务难度、最小有意义收益、每域非劣界、灾难退化上限和可承受 CPU 预算，然后锁定；不得看 sealed-test 结果后调整门槛。无训练保持不是通过依据。
2. 使用至少三个实际模型训练 seed、多个真实课程顺序；记录父参数 digest 与随机初始化/训练差异。重复同一已训练 worker 的运行不能充作独立模型训练样本。统计按独立 seed/项目聚类，列逐 seed 结果，不将相关 cell 机械当成九个独立样本。
3. 主要比较 candidate 与最强固定容量学习臂；冻结臂只用于验证是否学到新能力。评价新任务收益与全部旧域保持，禁止只选最终一个域。
4. 增设 v1 兼容对照切片：在同一数据、预算、scorer、可比较 owner 边界重跑 v1 continuation 与 v2 学习器。若 checkpoint/owner 无法公平对齐，记录不兼容原因并建立共同父基线；历史 R7 只作参考，不能直接拼成绩。
5. 资源按相同硬件/线程/实际样本比较，分训练与推理，隔离进程并平衡运行顺序；测真实峰值或明确仅下界。同时报告全部模型、附加学习状态、checkpoint 总量，不能仅报 K worker 参数 bytes。

通过条件：冻结的新任务收益、旧域非劣与最坏域上限、恢复/回滚和相关 CI 均满足；报告失败也必须完整。若不满足，保留最强固定容量基线，回到已定位的学习问题；不因 formal 名称或 aggregate passed 晋级。

### D. 有依据地恢复结构成长研究（C 后）

固定容量学习成立后，通过误差/干扰与容量扫描确认增长需求。复用 R3/R4 主路径 residual、零影响出生、局部 credit 和 rollback；R5 路由候选不默认复活。新增结构必须继承原模型、迁移学习状态，不能以独立新模型替代成长。

比较 frozen、最强固定容量、动态成长、从开始就具同等最终容量的 fixed-large，并分别报告训练量与总成本。lesion 移除学到的结构贡献但不改变执行权限、评分器或接线。成熟准入要求未见任务收益/保持及预注册质量—资源目标；没有优于强对照的证据就不开放默认成长，也不外推成所有成长设计无效。

### E. 后续能力和客户端日程（保留，不并行抢占本体训练）

| 顺序 | 解冻条件 | 具体交付与边界 |
|---|---|---|
| E1 Skill/MCP 知识学习 | C 的有效学习基线成立 | 来源许可、去敏、正确/失败轨迹、去重与项目隔离；与普通语料对照，验证内化后的未见任务收益。学习工具知识不赋予执行权限 |
| E2 Workbench / interaction-group / 小型模拟 Gate | E1 有可用行为能力，或为 B 的隔离实验所必需 | 模型生成结构化意图、IDE 语言选择来自任务/文件内容、preview/批准/执行/undo、跨步骤状态；与真实客户端安全边界一致，不靠语言下拉框注入答案 |
| E3 客户端插件热插拔与 provider watchdog | E2 接口和所有权稳定 | 工具/设备能力声明、兼容性、热插拔/回滚、watchdog 故障隔离；客户端继承硬件适配，模型不获得任意宿主权限；provider 只按明确语言器官合同接入 |
| E4 CUDA、产品视觉与发布 | 硬件可用再启动 CUDA；接口稳定后安排视觉发布 | CUDA 做 CPU 数值/性能对照；视觉包括原 logo、托盘通知/任务栏、圆角和生命状态入口一致性；清除确认失效的 HF/旧架构入口。无硬件不阻塞 A～E3 |

严重客户端错误、安全问题和当前修改导致的 CI 失败可优先止损；这不授权平行扩展外围功能。日程以 A→B→C→D→E 的验收依赖驱动；B 吞吐测得前不承诺训练天数。

### 文档和决策规则

每个阶段结束把结果更新到本节并精确提交；新实验运行失败时不能仅写“已推进”。旧预注册和原始报告是历史证据，不覆盖、不为变绿放宽旧门槛。核心需求/架构继续留 active；历史“下一步”不再复制到新入口。下一轮只执行 A，之后依上述依赖推进；训练预算超出本机可承受范围、需要付费/外部权限或改变架构目标时停下讨论。

---

## 历史设计及推进记录（下列顺序已被上节取代）

## 1. 当前事实基线

### 1.1 项目定位

Taiji 是唯一认知主体，Seed 是产品/runtime、Workbench、设备、权限和发布载体。Taiji 可以复用 embedding、SSM、attention、MoE、autograd、optimizer、CUDA 和成熟语言器官，但持续世界状态、记忆、目标、路由、发展选择与行动决定必须由 Taiji 拥有。

目前是有真实小规模训练和多个窄能力 Gate 的研究原型，不是成熟语言大模型，也不是已经完成自主进化的系统。

### 1.2 已完成且仍有效

| 范围 | 有效成果 | 不得外推为 |
|---|---|---|
| M0/M1 | checkpoint、局部学习、字节预测、延迟记忆、简单世界转移与行动基线 | 通用语言、推理或完整神经认知架构 |
| M2 测量与训练 | record-disjoint 数据、owner 路由、read-only score、fresh restore；结构化事实/Goal/ContentPlan、多步状态转移、双 owner 组合与保持 | 开放域语义、共享表征联合优化或自然语言能力 |
| M3 Workbench | project-disjoint observation、task-state sequence、只读 ActionIntent、preview/approval、隔离执行与 undo | 未经授权的真实客户端自治写入 |
| M4 v1 技术链 | 三模型 seed 的数据 lineage、原子保存、fresh restore、owner 差分、资源测量和可重复 F1 cascade | CR-4/A8 结构成长已完成或已被否决 |
| 结构资产 | `AdaptiveNeuronRegion/Network`、proposal、budget、validation、lesion、rollback、lineage | 已自动进入普通 observation/decision 主路径 |
| 产品外围 | provider artifact/watchdog、Skill/MCP、插件、客户端与 CI 的既有实现 | 模型本体能力已随外围功能增长 |

### 1.3 M4 v1 实验的有效读法

- R7 scale `0.5` 的 C3 新分布 gain 均值 `+0.152757 BPB`、3/3 为正；C cycle3 平均 delta `−0.005050 BPB`，但 seed11/47 分别 `+0.002126/+0.009497`，因此未通过当时预注册的 exact-zero all-seed Gate。Gate 结果保留，不能把它外推成“方法没有稳定—可塑信号”。
- R10 证明的是 readout-only 路径上固定 logits preservation 不通过 Gate；它没有检验真实经历 replay、importance 或 fast/slow consolidation。
- R2 的容量臂是“零 readout slot + fixed 50/50 概率平均”，没有 learned router、表征扩宽或零影响成熟期；负结果不再作为容量压力的总否决。
- R7 更新 protected context+readout，R10/R12 在 active Workbench boundary 下只更新 readout。跨轮不能把差异全归因到 consolidation 或数据。
- R12 的 A `+1.62 BPB` 是 UltraData 臂相对 simple_zh continuation 臂的差值，不是报告内直接计算的 parent→child forgetting；它只支持“abrupt replacement 高风险”。
- C/C2/C3 的 record-level disjoint 有效，但三者仍来自同一语料；R9 的 byte marginal JS 不能代表高阶序列、语义或技能边界。

## 2. 本次架构决定

1. **M4 v1 结束，M4 v2 重开。** 不新增 R13，也不沿旧 Gate 直接跑混合语料。
2. **先量尺、后模型。** 先让每个指标都声明方向、父基线、单位、delta 和能力 owner；缺基线不得写“遗忘”。
3. **在原 checkpoint 上迁移。** 旧权重成为 `slow_weight`，新增 `fast_delta=0`；迁移边界输出必须等价，不重新开始训练。
4. **快适应与慢巩固分开。** wake 写 fast/episodic，sleep 用真实经历 replay、importance 与验证写 slow；protected snapshot 只是对照/恢复点。
5. **结构必须进入主路径。** `AdaptiveNeuronNetwork` 只有被 observation→prediction/workspace→credit 实际消费，才算成长 substrate；显式旁路调用只算 shadow 实验。
6. **增长必须函数保持。** 新 population/region 以输出 gate=0 出生，学习 residual 和 router，通过 Gate 后开放；固定平均或每任务新 head 禁止作为正式方案。
7. **路由不看任务 ID。** 外部 boundary 继续控制权限，不控制认知知识选择；路由从内容、状态、不确定性、目标和资源学习。
8. **成熟技术可用于发展训练。** 原生身份按认知所有权判断。当前 AST 禁令在正式对照与合同迁移前保持，不能静默绕过。

## 3. 唯一执行顺序

| 顺序 | 阶段 | 状态 | 完成条件 |
|---|---|---|---|
| 0 | M0/M1 基线 | 已完成 | 历史能力按原边界保留，不重跑整段 |
| 1 | M2 可信训练 | 已完成当前范围 | owner/data/checkpoint/语义与多器官窄 Gate 可复用 |
| 2 | M3 Workbench 因果边界 | R0～R5 已完成，真实客户端写入仍需单独授权 | 当前不扩大副作用范围 |
| 3 | M4 v1 fixed-capacity F1 | R0～R12 已归档 | 候选与报告保留；不再发布旧方向下一步 |
| 4 | **M4.V2 继承式成长** | **当前阶段** | 依次完成 R0～R6；禁止跳级 |
| 5 | M5 知识与身体 | 冻结等待 M4.V2.R4 | Skill/MCP 成为有来源经历；硬件 capability 由客户端继承，认知/执行收益分开归因 |
| 6 | M6 provider 与产品收口 | 冻结等待 M4.V2.R4 | provider watchdog、HF 残留、客户端真实能力对齐、桌面体验一致 |
| 7 | M7 CI/发布 | 每轮相关检查，阶段末全量 | 无新增 CI 退化；release manifest 绑定代码/数据/model/package |
| 8 | M8 CUDA | 本机硬件阻塞，不阻塞 CPU 主线 | 有真实设备后做同 checkpoint 数值/收益/成本对照 |

### 3.1 M4.V2 里程碑

| 里程碑 | 单一研究问题 | 主要交付 | 退出 Gate |
|---|---|---|---|
| R0 量尺合同 | 当前结论是否由指标语义和 Gate 偏差造成 | course manifest、metric spec、cumulative scorecard、非劣界校准、legacy report audit | 能识别绝对值/delta、owner 混用、缺父基线、跨域 BPB 和 exact-zero 偏差；不训练 |
| R1 零变化迁移 | 旧 checkpoint 能否无损进入 fast/slow 状态 | versioned developmental synapse payload、migration、fresh restore | `slow=old/fast=0`；输出、score、owner、参数计数与恢复在容差内一致 |
| R2 快—慢学习 | 真实 replay consolidation 是否改善稳定—可塑性 | wake fast update、episodic sampling、sleep slow write、rollback | S/G 课程优于 slow-only/fast-only/旧 preservation；不过则不扩容 |
| R3 主路径结构桥 | adaptive region 能否真实参与计算且关闭时不改变父代 | zero-gated residual region、forward/credit 接线、owner trace | gate=0 等价；打开后有可测贡献；关闭/lesion/restore 完整 |
| R4 shadow 生长 | 容量压力能否触发有价值的新结构 | residual/conflict pressure、shadow train、admission/rollback | 优于 fixed-capacity、random growth、等参数预分配；保持与资源 Gate 通过 |
| R5 自主路由 | 无任务 ID 时能否形成专门化与协作 | content/state/uncertainty/resource router、load/credit trace | 优于最强单体、固定/随机/平均路由；router lesion 破坏组合能力 |
| R6 A8 formal | 是否展示可重复、受控资源的连续净成长 | 3 model seeds × ≥3 course seeds/orders 的 S/G/K 矩阵 | 新能力置信下界为正、旧能力非劣且无灾难域、结构 lesion 有因果、可回滚 |

依赖是严格的：`R0 → R1 → R2 → R3 → R4 → R5 → R6`。R2 失败时回到学习/记忆设计，不允许用增长掩盖；R3 失败时不能运行 R4；R4 未胜过 matched-capacity baselines 时不能称自进化。

## 4. 当前唯一下一步

M4.V2.R0、M4.V2.R1、R2 与 R3 已完成当前阶段的量尺、迁移、快/慢学习和第一条主路径结构桥 Gate；R2 候选被接受，R3 canary 通过；R4 pressure 已由 R3 bridge 的真实 predictive observation tick 驱动，candidate artifact 已被固化，独立 shadow materialization、candidate-only causal training smoke 和首个短 S/G 五臂对照已通过技术 Gate，但 pressure-driven growth 没有稳定优于 R3 fixed-capacity，candidate lesion 在 S/G 方向不一致。随后补上的逐 tick activity、candidate residual、credit、projection update 和 trace digest 证明候选确实被激活并持续收到 causal credit；把候选 projection 的学习资格从瞬时 activity 扩展到已有的 candidate eligibility trace 后，候选残差可测增强，G lesion 增益变为正，但 S lesion 仍为负，仍不能宣称增长有效。审计发现首轮五臂没有严格共享同一 parent/owner/训练边界，已在本轮修正为同一 `pressure_parent`、同一 parent-bridge 学习边界和同一课程预算；candidate-only smoke 独立保留并通过。校正后的复跑中 pressure growth 相对 fixed-capacity 在 G 改善约 `0.04233`、S 退化约 `0.00033`，相对 fixed-large 仍未占优，S lesion 仍为负。固定边界的 3 model-seed × 3 course-seed/order 矩阵证明 pressure growth 已相对 fixed-capacity 稳定改善（G `9/9`、S 均值改善约 `0.04915`），但相对 fixed-large 在 S 为 `0/9` 占优，问题从 credit 断路收敛为出生选择质量不足。单-anchor pressure birth 已修复 S 但专门化 G；top-2 birth mixture 的 9-cell 结果为：相对 fixed-capacity 的 S `9/9`、G `8/9` 改善，S lesion `9/9` 为正但 G lesion 仅 `2/9` 为正且均值为负，相对 fixed-large 的 S `8/9`、G `4/9` 占优。utility trace 证明 canonical G 的平均 residual·feedback 约 `0.0409` 不能解释 G holdout lesion `-0.00555`；第一轮 gate 的近似 credit 已证明不足。exact counterfactual、direct exact gate credit、dual-state、three-source gate 与 aligned pressure birth 已按 canonical→9-cell 完成：aligned output 使 formal G lesion 均值从约 `-0.00199` 收敛到 `-0.000008`，`7/9` cell non-worse，S 仍 `9/9` 正且均值 `+0.07515`；但 pressure 对 fixed-large 的 G 仍平均劣化 `0.00287`，所以这是“功能伤害基本消除”而不是“结构增长晋级”，整体保持 `can_promote=false`。当前唯一下一步是 **M4.V2.R4：对 aligned anchor mixture 做 deterministic birth homeostasis，使 incoming/recurrent/output 三类 candidate mapping 的有效范数与 anchor mixture 的加权二阶范数一致，再跑 canonical→9-cell**；不改变 gate credit、pressure、parent、课程、预算或对照，不进入 R5 learned router。

R0 的实现与审计产物如下：

- `taiji/continual_evaluation.py`：完成 versioned/content-addressed 的 metric spec、课程 manifest、累计 snapshot/scorecard；绝对值、parent delta、comparison delta、owner 和 read-only 输入 digest 已分离；缺少 parent baseline 时非劣判定 fail-closed；跨域 raw BPB 不可隐式聚合；phase 标签不进入 model context。
- `scripts/training/audit_taiji_m4v2_measurement_contract.py`：只读重审 R7/R10/R12，生成 `reports/taiji_m4v2_measurement_audit_20260909.json`；R10 A 被标为 absolute BPB，R12 `+1.62` 被标为 arm-vs-arm，历史 exact-zero Gate 保留但不再作为全局架构否决。
- `tests/taiji_native/test_continual_evaluation.py`：13 项定向测试通过；目标模块 ruff 与 `git diff --check` 通过；没有训练、checkpoint 权重或旧 JSON 改动。

R0 的技术结论是“量尺语义审计通过”，不是模型晋级：审计报告仍保持 `can_promote=false`。

### 4.1 R1：零变化迁移（已完成）

本步骤只迁移已有 F1 checkpoint 的表示，不写入新学习权重，不改变 forward：

- 新建 versioned developmental-synapse payload：`slow_weight=old checkpoint`、`fast_delta=0`，并保存 eligibility/importance/usage/age/plasticity 元数据的默认值；
- 对旧 F1 checkpoint 做迁移、保存、fresh restore、二次迁移四路一致性比较；
- 固定同一输入流和同一 owner graph，比较输出 logits、loss/BPB、参数计数、checkpoint digest、评估输入 digest；
- `taiji/developmental_synapse.py`：完成 F1 两个稀疏 owner 的 versioned bundle；旧 edge topology/weight 进入 `slow_weight`，`fast_delta=0`，eligibility/importance/usage/age/plasticity 显式保存；有效 forward 为 `slow + fast`。
- `scripts/training/migrate_taiji_f1_checkpoint.py`：支持裸 `taiji-native-v10` 和外层 `taiji-native-joint-training-v1`；源文件不覆盖，外层 digest 重算，磁盘 save/load/fresh restore 逐项校验。
- `taiji/model.py` / `taiji/organs.py`：R1 overlay 只读接入 F1 predictive context/readout；默认无 overlay 路径不变，挂载态写入明确拒绝。
- 真实 `output/taiji-m2-f5-seed11-private-context-20260905/last.pt` smoke 通过：`source_unchanged=true`、`source_checkpoint_digest_valid=true`、`restored_outer_digest_valid=true`、`fresh_restore_matches=true`、`fast_is_zero=true`；smoke 产物已清理。
- `tests/taiji_native/test_developmental_synapse.py`：磁盘迁移、forward/score/generate 等价、fresh restore、只读停止线均覆盖。

R1 完成条件：

1. 旧 F1 权重、topology、owner graph 无损可寻址；
2. `slow=old/fast=0` 时输出、score、generate、参数计数和 restore 在容差内等价；
3. source checkpoint 不被覆盖，outer/inner digest 都能验证；
4. R1 overlay 不允许写入，未通过 R2 前不启动真实 fast/slow 学习。

### 4.2 R2：快适应—慢巩固（实现合同与单 CPU smoke 已完成）

本步骤第一次允许写 developmental state，但仍不扩容、不接入新的 adaptive region：

- 先做 S 同分布 smoke，再做 G 渐进混合 canary；每个 checkpoint 都生成绝对能力、parent delta、comparison delta 和资源记录；
- 固定同一 parent/owner graph，比较 slow-only、fast-only、fast + 真实经历 replay + consolidation；禁止把旧字节前缀或静态 preservation logits 当作经历 replay；
- wake 只写 fast/episodic evidence，sleep/replay 才允许写 slow；记录 eligibility、importance、usage、age、plasticity 的变化；
- 通过 calibrated epsilon、worst-domain catastrophe、fresh restore、rollback 和 read-only Gate 后，才允许保留 R2 候选；失败则恢复 R1 parent，不运行 R3/R4。

已完成的实现与单 CPU smoke：

- `taiji/developmental_synapse.py`：发展突触提供 `slow_weight + fast_delta` 的固定拓扑 forward/backproject、fast/slow 局部更新、eligibility/importance/usage/age 更新、慢巩固和边界约束；另有 versioned 的真实 wake replay event/buffer。
- `taiji/model.py` / `taiji/organs.py`：R1 overlay 继续默认只读；R2 只能通过显式 mode 进入，`fast`/`slow`/`fast_slow` 的写入 owner 明确，replay 只消费实际 wake 局部 credit trace，静态 preservation logits 被拒绝；checkpoint 保存 developmental state 与 replay evidence，fresh restore 强制回到 read-only。
- `scripts/training/eval_taiji_m4v2_r2_canary.py`：固定同一 parent、同一 owner graph，对比 `slow_only`、`fast_only`、`fast_replay`；S 后接 `80/20 → 60/40 → 40/60 → 20/80` G 课程，生成 manifest、scorecard、parent delta、资源和恢复/回滚 Gate。
- `reports/taiji_m4v2_r2_sg_canary_20260909.json`：单 CPU smoke 的三臂合同 Gate 全通过，S/G 的本次样本相对 parent 均为 lower-surprise 改善；这只证明实现边界和一次小课程闭合，不证明正式晋级。

### 4.3 R2.Formal：多 course-seed/order S/G Gate（已完成）

只复用 R2 的实现，不改 owner、不扩容、不接 R3。已先以 parent 独立、等比例、等长度 block 的波动校准 epsilon，再运行 3 个 course seed/order 的 S/G 短课程；每个 arm 都保留 slow-only、fast-only、fast+real replay+consolidation 对照。正式报告为 `reports/taiji_m4v2_r2_formal_sg_20260909.json`：初版因把超过上限的 `0.4629` 波动截断为 `0.05` 而被 fail-closed；修正校准 block 后 `max_deviation=0.0176304`，epsilon=`0.0176304`，所有 Gate 通过。replay 相对每个 course 的最强 fixed-capacity arm 在 S/G 均不劣且严格更优，旧 F1 owner 未被写入，fresh restore/rollback/read-only 全通过。该结果接受 R2 候选，但不代表 R3 已通过，更不代表 A8 晋级。

### 4.4 R3：主路径 zero-gated adaptive residual bridge（已完成）

R3 只实现一个最小结构桥，不创建多个专家、不引入任务 ID 路由、不接 Skill/MCP 或客户端外围。候选从当前 `observation → predictive context/readout → credit` 路径获得输入和误差；稳定 F1 trunk 保持原 owner；新增 residual population 以输出 `gate=0` 出生；forward、credit、checkpoint、lesion 和 rollback 消费同一 owner graph。

已完成的实现与 Gate：

- `taiji/adaptive_residual_bridge.py`：以 `AdaptiveNeuronRegion` 为底层的稀疏 residual population，稳定 unit identity、显式 `predictive_context` 输入源、gate、lesion、局部 credit、runtime state 和 versioned payload；recurrent fan-in 对自连接约束做了小容量边界裁剪。
- `taiji/model.py`：把 bridge 接入 native predictive forward；gate=0 时不 step、不写 credit、不改变原预测；显式 `learn_adaptive_residual_bridge` 允许在结构 admission 试验中冻结成熟 F1 context/readout，只把同一 causal error 送入新桥；checkpoint、identity lineage、参数清单、reset、fresh restore 和 rollback 均保留 bridge。
- `taiji/__init__.py`：导出 bridge 的 format/version/类型，保持类型边界可寻址。
- `scripts/training/eval_taiji_m4v2_r3_bridge.py`：先做 checkpoint preflight，再做 zero-gate 等价、active residual、local credit、mature F1 owner 不变、lesion、fresh restore 和 rollback canary；报告为 `reports/taiji_m4v2_r3_bridge_canary_20260909.json`。
- `tests/taiji_native/test_m4v2_r3_bridge.py` 与 `test_m4v2_r3_bridge_canary.py`：覆盖默认无 bridge、gate=0 主路径等价、显式 bridge credit、owner freeze、lesion、fresh restore、rollback 与报告 Gate。

R3 canary 结果：所有 12 项 Gate 通过；`gate=0` 输出与 parent 完全一致，active residual `L1=1.30631685`，概率最大变化 `0.00535537`，local credit 改变 bridge payload，mature F1 两个 owner 未变，fresh restore/lesion/rollback 通过。该结果只接受“结构候选已进入主路径”的技术事实，不接受结构成长或 A8 晋级，`can_promote=false` 保持不变。

### 4.5 R4：shadow 生长与 matched-capacity 对照（阶段收束，未晋级）

R4 要回答的不是“能否再创建一个区域”，而是“固定容量已出现可观测压力时，按真实压力出生的候选是否比随机增长和从一开始就等量预分配更有用”。只允许沿 R3 bridge 这一个结构入口做第一轮 shadow growth；不引入 learned router、不接 Skill/MCP、不接客户端外围。

实施顺序固定为：

1. **压力合同与主路径观测（已完成）**：`taiji/adaptive_residual_growth.py` 已定义 residual error、fast/slow conflict、activity saturation、utility gap、resource state 的 versioned/content-addressed observation；`AdaptiveResidualGrowthTrigger` 提供 EMA、连续压力、预算、parent digest、重复 evidence 和 checkpointable decision。`Taiji.observe(readout="predictive")` 只在同一 native observation→prediction/credit tick 已产生 prior error 且 R3 bridge 作为当前 owner 时记录 pressure；没有 task ID，没有 synthetic evaluator 路由，也不自动改拓扑。decision digest、last pressure、trigger checkpoint 和 model fresh restore 已闭合。
2. **候选提议（已完成）**：`should_propose=true` 的 decision 已固化成 content-addressed zero-impact candidate artifact，在 bridge 上按稳定身份描述一个最小 unit，保存 parent/source digest、proposal evidence、topology diff、结构预算和 pending topology proposal；artifact 创建没有改变 parent 函数、bridge unit 数量或 forward/credit substrate。
3. **shadow materialization（已完成）**：从 candidate artifact 复制 bare parent 到独立 shadow 实例，在独立 checkpoint 中应用 pending topology proposal；旧 unit 的 incoming/recurrent 支持、权重、runtime state 逐项保持，新增 unit 初始化可寻址，shadow 的 identity-preserving sparse projection 对齐原输出，新增 unit projection edge 初始为零，shadow 为 `gate=0`；fresh restore 后 candidate/parent digest、bridge identity 和 shadow payload 一致。materialization 失败不会改变原 parent 且不消耗正式结构预算。
4. **shadow 训练（最小 Gate 已完成）**：先保存 shadow bare checkpoint，在 fresh restore 后只打开 shadow gate，冻结成熟 F1 trunk 与 parent readout，只给 shadow residual/projection/候选 region credit 做一次最小更新；已经验证候选 projection 会在真实活动 tick 改变、旧 region 前缀不被写入、再次保存/fresh restore/续步有效，失败可回滚到 bare shadow。该 Gate 只证明训练边界，不证明能力收益。
5. **短课程与准入对照（已完成首轮，但结果未通过晋级）**：已用同一 parent、同一 S/G 课程、同一训练预算比较 frozen parent、R3 fixed-capacity bridge、pressure-driven growth、random growth、同等最终参数量的 fixed-large；五臂都完成 checkpoint preflight、短训练/holdout、candidate lesion、fresh restore、rollback 和资源记录。报告中 pressure-driven growth 的 S/G 结果没有稳定优于 R3 fixed-capacity，candidate lesion 的方向不一致，因此只接受“对照管线闭合”，不接受“生长有效”。
6. **candidate credit/activation 修订（已完成一轮，未晋级）**：已增加逐 tick candidate activity、projection credit、candidate-vs-parent residual、eligibility 和 lesion trace，并把候选 projection 的资格改为使用已有 candidate trace；成熟 F1 trunk、pressure contract、课程和对照数量未改变。新报告显示 pressure candidate 的平均 candidate/parent residual ratio 约 `0.00981`，G lesion delta `+0.00474`，S lesion delta `-0.00027`；候选 credit 通路有效，但跨 S/G 不一致，因此 R4 仍未通过。
7. **matched-capacity 边界校正（已完成）**：五臂现在统一从 `pressure_parent` 进入；fixed-capacity 与 growth efficacy 都使用 parent bridge 可学习边界；candidate-only 版本单独作为冻结 parent 的 shadow smoke。校正后技术 Gate 全通过，但 pressure growth 只在 G 改善、S 轻微退化，且相对 fixed-large 未占优，不能晋级。
8. **多 seed/order 复测（已完成，未晋级）**：固定校正后的五臂、eligibility 规则、pressure contract、课程内容和预算，完成 `3 model seeds × 3 course seeds/orders = 9` 个 cell；9/9 的 pressure growth G 相对 fixed-capacity 改善，S 均值改善 `0.04915`，但 pressure 相对 fixed-large 在 S 为 `0/9` 占优，故不能把收益归因于压力驱动出生已经优于等量预分配。正式报告为 `reports/taiji_m4v2_r4_shadow_formal_20260909.json`。
9. **pressure-conditioned candidate birth（已完成一轮，未晋级）**：候选 artifact 仍保持零影响和 content-addressed；shadow materialization 根据同一 `pressure_parent` 的 native activity+eligibility 选择单 anchor，再升级为固定 fan-in/预算内的 top-2 activity+eligibility birth mixture；anchor identities、归一化权重和 parent digest 可 fresh restore 重算。两轮都证明 S 能力可改善，但 G 的 candidate lesion 不稳定，不能晋级。
10. **candidate utility/selectivity 诊断（已完成）**：逐 tick 已保存 candidate residual、同一 causal feedback、candidate utility、candidate/parent residual ratio 和 S/G phase。canonical 中 pressure candidate 的 G utility 均值为正但 G holdout lesion 为负，证明必须引入上下文选择性，不能用全局 utility 标量直接开关。
11. **candidate output selectivity gate（已完成一轮，未晋级）**：增加了从 parent eligibility trace 到 candidate gate 的 1×fan-in 稀疏 projection；所有 growth/fixed-large 臂均携带 gate 以保持参数匹配，gate bias/weights、utility baseline 和 runtime state 可 checkpoint，parent freeze/restore/rollback 全通过。但 gate 值仍在约 `0.50` 附近，G lesion 仍不稳定，因此不继续调学习率。
12. **exact counterfactual gate credit（已完成一轮，未晋级）**：shadow 在 candidate forward 后保存同一 readout/evidence 下的 candidate-off 概率，下一 observation 用真实 target 在 shadow learn 之前计算 `loss_off - loss_on`；9-cell 技术 Gate 全通过，但 candidate lesion 的 G 只有 `2/9` 正，不能晋级。
13. **direct exact gate credit（已完成一轮，未晋级）**：exact delta 已直接用于连续 gate 的 local update，EMA 只保留为诊断；pressure 训练期 G gate 均值约 `0.46`，但 S/G holdout gate 都约 `0.42`，G lesion 仍为负，说明瓶颈转为 gate 输入表征不足，不能晋级。
14. **dual-state context gate（已完成一轮，未晋级）**：gate projection 的输入改为 parent instantaneous activity 与 parent eligibility 的拼接；所有 growth/fixed-large 臂使用相同新 topology，技术 Gate、checkpoint、restore、rollback 全通过。canonical holdout 的 S/G gate 约 `0.451/0.429`，formal G lesion 仍仅 `2/9` 正，说明局部双状态略有区分但不能支撑未见 G 泛化。
15. **three-source context gate（已完成一轮，未晋级）**：gate 输入扩展为 parent bridge input context + instantaneous activity + eligibility；所有 growth/fixed-large arm 使用相同 topology，技术 Gate、checkpoint、restore、rollback 全通过，但 canonical holdout S/G gate 约 `0.419/0.422`，formal G lesion 仅 `2/9` 正，说明继续扩 gate 输入不能修复出生功能未对齐。
16. **aligned pressure birth output（已完成一轮，部分通过）**：pressure anchor/mixture 在 materialization 时同步初始化 candidate→context output projection，使候选输出映射继承同一 anchor mixture；formal G lesion 均值约 `-0.000008`、`7/9` non-worse，S lesion `9/9` 正，但 pressure 相对 fixed-large 的 G 仍平均劣化 `0.00287`，只能接受“功能伤害基本消除”，不能晋级。
17. **birth homeostasis（已完成一轮，部分通过）**：对 aligned anchor mixture 的 incoming/recurrent/output candidate mapping 做 deterministic weighted-second-moment norm matching，避免新单元因稀疏截断和 mixture 权重而出现过强或过弱的有效范数；保持三源 gate、direct exact credit、parent、pressure、课程、预算、projection 和 fixed/random 对照不变，完成 canonical→9-cell。formal 中 G candidate lesion 均值 `+0.003081`（`6/9` 胜出、`9/9` 不劣），S 均值 `+0.080387`（`9/9` 胜出）；相对 fixed-capacity 的 G/S 均为 `9/9` 改善。它证明 homeostasis 修复了出生映射的量级失配并恢复候选的可观测因果贡献，但相对同等最终参数量的 fixed-large，G 均值仍为 `+0.001002`、仅 `4/9` 胜出，不能证明压力驱动生长优于从一开始预分配。
18. **因果 Gate 与 R4 收束（已完成判定，未晋级）**：candidate lesion、fresh restore、rollback、owner 隔离、matched parent boundary 和 9-cell 技术 Gate 全通过；但 fixed-large 的 G 非劣/优效条件未通过，故只能接受“候选确实产生因果增益且 homeostasis 降低了新增伤害”，不能接受“压力驱动结构生长已优于等量预分配”。`can_promote=false` 保持不变，默认能力继续使用 R3/pressure parent，homeostasis 只保留在 shadow 实验路径，不进入 R5 learned router。

R4 的最小交付 `pressure/proposal → shadow materialize → shadow train → validate → lesion/admit/rollback` 已形成可复现 CPU canary 和 versioned report；birth homeostasis 已把 G candidate lesion 从此前的负均值修复到正均值，但相对 fixed-large 的 G 仍未达 Gate，因此 R4 以“技术闭合、结构不晋级”收束。后续只允许做收束审计、默认路径保护和计划证据整理，不得继续用 gate 输入、出生缩放或新语料无边界试探来掩盖 fixed-large 反证；R5 自主路由、Skill/MCP/provider/客户端外围继续冻结，直到新的架构决策明确解除停止线。

R4 收束审计已完成其研究边界部分：R4 定向 6 项测试通过，homeostasis 只存在于独立 shadow，materialize 不改变默认 parent checkpoint/forward，版本一致性检查通过，本轮涉及文件的 ruff、B/SIM 和 black 约束通过。全量 native 回归得到 `627 passed, 17 failed, 15 errors, 1 skipped`；15 个 error 主要是本机历史 pytest 临时目录锁权限，切换到仓库可写 basetemp 后代表性测试可正常执行；剩余失败集中在既有 context/delayed memory、Workbench neutral baseline、interaction structural gate 和 synapse longevity 基线，不属于本轮 R4 代码路径。tracked source 的 ruff、B/SIM 与 core mypy 在本轮已分别收敛到 0，但全量 native 回归仍未绿，不能把仓库 CI 写成已通过。

M7 CI 基线收敛第一批已完成：tracked source 的主 ruff 门禁从 9 项降为 0，安全自动修复涉及的 14 个脚本/平台/测试文件已通过 `py_compile`、ruff、black diff 和 `git diff --check`；R4 代码路径与默认 parent 未改变。随后完成 B/SIM 显式契约审计，`51 → 42 → 0`，所有 `zip` 改动都基于已确认的等长输入，未使用 unsafe 批量改写；受影响模块回归 `23 passed`。本轮完成 core mypy 棘轮：`mypy --follow-imports=silent seed taiji` 从 `175 errors / 24 files` 收敛到 `0 errors / 95 files`，同时通过源码范围 ruff、B/SIM、py_compile；与类型边界相邻的回归分组共 `85 passed`，Qwen provider 的 `tmp_path` 组受本机 pytest 临时目录权限阻塞，代码未出现断言失败。

`context/delayed memory` 两个 native 失败已复现并归因：`learn_bytes()` 的现行合同训练 F1 predictive readout，而旧测试仍用默认 action readout 评估，因而得到 `0.0`；没有改模型或放宽阈值。测试现已显式选择 `readout="predictive"` 并关闭 identity 污染，context 为 `1.0 vs 0.5`，delayed 为 `1.0 / 0.5 / 1.0 / 0.5`，原 Gate 仍成立。

`test_synapse_longevity.py` 的失败也已完成归因：`SparseSynapses.local_update()` 已按接触资格门控 decay，decoder mass 没有发生蒸发；旧测试在 `learn_bytes()` 之后通过 `Seed.observe()` 默认进入 action readout，并继续学习共享 fabric，使 predictive decoder 的输入表征漂移，surprise 从 `2.481` 升到 `2.704`。测试现已沿 M2 合同显式使用 `readout="predictive"`、`use_memory=False`、`use_identity=False`；150 轮后 decoder mass 保持且 surprise 降至约 `1.356`，原阈值未放宽。

`Workbench neutral baseline` 已完成归因：新的 Workbench task-boundary 合同要求 `execute_taiji_workbench_task()` 携带显式 token，运行时拒绝无 token 是安全行为，不是要放宽的基线；旧 longitudinal evaluator 没有打开 boundary。评估器现已为每个 episode 打开覆盖实际 capability 的 boundary，并让 neutral、正常动作和 recovery 共享同一 token，Workbench longitudinal、interaction-group Workbench 与 structural bridge 相关回归共 `24 passed`。

随后按同一失败账本继续验证时，首个真实失败收敛到 `test_multistep_grounding_recovery.py`：负向场景使用不存在路径，但当前 live semantic grounding 会在执行前按设计返回 `workspace_target_not_found`，因此旧评估器错误地期待已保存的失败 checkpoint。没有放宽生产 grounding；评估器现仅在这个受控负向探针中传入显式 `parameter_bindings`，让故障落到真实执行层并验证 checkpoint/fresh recovery，成功路径仍保持 Taiji-owned semantic grounding。相关 memory/objective/naming/multistep 分组现为 `16 passed`，评估脚本 Ruff/编译/diff 检查通过。

**当前唯一下一步**：继续在仓库可写 basetemp 下刷新完整 `tests/taiji_native` 的失败账本；先用不触发 pytest 临时目录 fixture 的分组锁定首个真实失败，再对需要 `tmp_path` 的失败单独处理 Windows 清理权限噪声。保持 R4 shadow/默认 parent 不变，不通过放宽阈值消除失败，R5 learned router 及 Skill/MCP/provider/客户端外围继续冻结。

R0 完成条件（已满足）：

1. 所有合同 versioned、content-addressed、checkpoint-independent 且 Python 3.10 兼容；
2. 旧 JSON 不改写，audit 生成新的版本化报告；
3. 技术结论、历史 Gate 结论和架构结论分字段输出；
4. 对 R7/R10/R12 的重审结果与本计划 §1.3 一致；
5. 定向 pytest、ruff、`git diff --check` 通过；
6. 仍保持 `can_promote=false`。

R1 通过后的唯一下一步是 R2 S/G canary；R2 formal 通过后的唯一下一步是 R3 主路径结构桥；R3 canary 通过后的 R4 先把 bridge pressure 接入真实 tick，再固化 zero-impact candidate，经过独立 shadow materialization 后才允许做一次最小训练步。R4 未通过则恢复 R3 parent，不得用 R5 路由、客户端外围或新语料掩盖容量/准入失败。

## 5. v2 课程与 Gate

### 5.1 课程轴

- **S 同分布续训**：同源 record-disjoint，多 course seed/order；检验吸收新记录。
- **G 渐进分布**：旧/新来源按 `80/20 → 60/40 → 40/60 → 20/80` 过渡；abrupt replacement 只作为压力上界。
- **K 技能组合**：结构化语义 → 世界转移 → Workbench 只读意图 → 隔离执行；检验累计技能和真实 outcome。

S/G 的 BPB 只证明低层预测变化，K 是 A8 的主要能力证据。正式候选按 `S smoke → G canary → K canary → multi-seed/order formal` 推进。

### 5.2 必须同时存在的对照

1. frozen parent；
2. fixed-capacity slow-only；
3. fast-only + rollback；
4. fast + real replay + consolidation；
5. random growth；
6. 与成长后参数量相同、从开始就预分配的 fixed-large；
7. 候选 growth + learned router；
8. growth/router/replay lesion。

缺 matched-capacity baseline 时，参数增多带来的收益不能归因于“生长方式”；缺 lesion 时，候选结构存在不能归因于能力。

### 5.3 正式晋级条件

- 所有技术/来源/owner/checkpoint/read-only/副作用检查全过；
- 新能力相对父代和最强 fixed-capacity baseline 的置信下界为正；
- 旧能力平均非劣，任一关键域不越灾难上限；
- exact-zero 作为额外观测报告，但非劣 `epsilon` 必须在看候选结果前由父代 block/course 变异校准并封顶；
- 无 evaluator task ID 的路由通过；
- 新结构 lesion 消除其增益，rollback 恢复父代函数；
- 收益/参数、收益/时间、峰值内存均在预算内；
- `can_promote` 只由 aggregate 计算，单 seed 永远不能晋级。

## 6. M5～M8 解冻顺序

| 工作 | 当前允许 | 解冻点 |
|---|---|---|
| Skill/MCP 语料 | 维护已有 provenance/schema，不扩大采集 | R4 后作为 K 课程的新经历来源；去来源 holdout 证明内化 |
| MCP 硬件与客户端插件 | 修阻塞 bug、保留 capability/ledger | R4 后验证身体能力继承、热插拔、撤销与认知 checkpoint 独立回滚 |
| provider watchdog / language organ | 保持可运行与显式回退 | R4 后做真实 artifact 轮换、故障恢复与语言效果；不替模型思考 |
| interaction-group | 保留已有实验资产 | R5 作为 learned router/协作对照，不提前扩系统 |
| 小型模拟 Gate | 继续用于快速回归 | 每个 M4.V2 里程碑先 smoke；模拟绿不替代真实 K 课程 |
| HF/Transformer 残留 | 保持 native/organ 类型边界 | M6 清理产品混淆与迁移 tombstone，不因名字做破坏性删除 |
| 视觉/桌面 | 只修崩溃和阻塞使用的问题 | M6 统一 logo、托盘/通知/任务栏、圆角/DPI、导航和生命状态入口 |
| CUDA | 记录 CPU 吞吐与内存 | M8 有真实硬件后验证同 checkpoint 等质量加速 |

## 7. 研究、CI 与提交纪律

- 每轮开始检查 working tree、现行 checkpoint、报告、active plan 和用户已有改动；不覆盖无关改动。
- 每轮只检验一个主假设，预先写输入、允许写入的 owner、基线、反证、预算、停止线和报告 schema。
- 任何训练前先在目标目录做初态保存、fresh-process 恢复、一次小更新、再保存/恢复/续步一致性；失败不训练。
- 旧报告不可改绿；更正通过新 audit/versioned report 表达。
- 定向测试在每个提交前执行；触及公共模块时运行相邻回归，阶段末再跑全量 CI。不得把“以后统一修 CI”当作接受当前新增错误的理由。
- 代码需满足 Python 3.10、ruff、black/mypy 现行配置和 `git diff --check`；只报告实际执行的检查。
- 每个完整单元精确 `git add` 并本地提交；不自动 push。训练 artifact、checkpoint、原始报告不随意删除。
- 核心讨论与当前设计留 active；证据解释放 reference；重复日志和失效执行顺序放 archive。本文保持约 30 KiB 内，已完成明细不再回填正文。
- 本计划不授权购买算力、访问新外部服务、真实客户端写入或发布。

## 8. 权威索引

- [Taiji 核心需求](../TAIJI_CORE_REQUIREMENTS.md)：CR-1～CR-10，长期不变目标。
- [Taiji Native Architecture v1](../TAIJI_NATIVE_ARCHITECTURE_V1.md)：完整认知架构与 A0～A9。
- [Taiji 继承式成长架构 v2](../architecture/TAIJI_CONTINUAL_DEVELOPMENT_V2.md)：M4 v2 的状态、主路径、学习与结构准入设计。
- [M4 fixed-capacity 证据复盘](../../reference/M4_FIXED_CAPACITY_EVIDENCE_REVIEW_2026_09_09.md)：R0～R12 数值、资产及复审后的解释边界。
- [M4.V2.R4 收束决策](../../reference/M4V2_R4_CLOSURE_DECISION_20260909.md)：结构增长“技术闭合、不晋级”的精确否决与唯一后续假设。
- [M4.V2.R5 预注册草案](../../reference/M4V2_R5_CONDITIONAL_MODULARITY_PREREGISTRATION_20260909.md)：资源归一化条件模块化的指标/对照/停止线，待三确认点拍板。
- [研究审视](../../reference/TAIJI_RESEARCH_REVIEW_2026_09_06.md)：M0～M3 的事实、失效结论和复现。
- [实现事实](../../reference/IMPLEMENTATION_STATUS_2026_08.md)：代码能力边界；更新晚于该文档的事实以本计划和版本化报告为准。
完整 native 回归在前 296 项均通过后，首个真实失败为 `test_m3r5_isolated_approved_execution_gate`：评估器内部 `tempfile.TemporaryDirectory` 在本机创建 0700 临时根，导致隔离 workspace 尚未复制就被 Windows ACL 拒绝；不是 Workbench 执行或审批 Gate 失败。M3.R5 现改为在 `SEED_M3R5_TMPDIR` 或系统临时父目录下用普通目录创建唯一隔离根并显式清理，测试把该父目录绑定到可写 fixture；隔离执行、审批、撤销和清理语义不变。M3.R1～R5 相邻回归为 `8 passed`，相关脚本 Ruff/编译检查通过。

**当前唯一下一步**：在同一仓库可写 fixture 机制下从头重跑完整 `tests/taiji_native`，继续按首个真实失败刷新账本；不修改 R4 shadow、默认 parent 或任何执行安全边界。
完整 native 回归已在仓库可写 fixture 下闭合：`659 passed, 1 skipped`，无真实失败；唯一 warning 为 FastAPI/Starlette 的既有兼容提示。标准 pytest 内置 `tmp_path` 在本机仍会因 0700 临时目录触发 `WinError 5`，这属于执行环境 ACL，不计入代码失败；M3.R5 评估器已不再依赖该创建方式。native 失败账本现清零，R4 shadow/默认 parent 仍未改变。

**当前唯一下一步**：审计并执行仓库 CI 交付层的精确门禁（tracked Ruff、B/SIM、core mypy、相关 pytest 命令和 workflow 约束），只修实际失败项并把可复现命令写入计划；不借 native 绿灯解冻 R5 或外围系统。
CI clean-checkout 静态门禁审计完成一轮：版本一致性通过；主 Ruff `0`、B/SIM `0`、core mypy `0 errors / 95 files`、相关脚本 `py_compile` 和 M3.R5 回归通过。审计发现 Git 跟踪的 `direct-*` Workbench 夹具包含故意的跨语言/坏语言样本，原 `ruff check .` 与 `black --check .` 会把它们误当源码；现已在 `pyproject.toml` 的 Ruff/Black 边界中排除，夹具内容不变。修正后 Ruff 两个门禁均返回 0；本机 Black 全量/定向进程仍出现无输出的 Windows worker/ACL 异常，未将 Black 写成已通过，需在可复现环境继续核验。

**当前唯一下一步**：执行 CI workflow 中剩余的 native verify 脚本和 `tests/seed` 回归，继续区分真实代码失败与本机临时目录噪声；不解冻 R5。
CI 动态 verify 首批已修复并通过：v7 原因是 `generate()` 后仍处于 predictive dynamics episode，旧脚本切换默认 action readout；现显式续接 predictive，v7 全部 Gate 通过。N7 原因是 `learn_bytes()` 的 F1 predictive 训练与旧 action 评估不一致；现显式使用 predictive、关闭 memory/identity 干扰，二阶 ambiguous accuracy `1.0`，相对一阶与 full-state lesion 均提升 `0.5`，全部 Gate 通过。模型、训练阈值和 R4 路径未改。

**当前唯一下一步**：继续执行 workflow 的 N8 delayed trace verify；沿用显式 readout/organ 边界，首个真实失败才允许修改对应评估契约。
N8 delayed trace verify 已通过：旧脚本原先用 action readout 评估 F1 训练结果，现让完整流、no-trace、trace-only 与 all-state lesion 统一显式走 predictive 且关闭 memory/identity。full 与 trace-only accuracy 均为 `1.0`，no-trace/all-state 为 `0.5`，trace necessity/sufficiency gap 均为 `0.5`，未改模型或门槛。

**当前唯一下一步**：继续执行 workflow 的 N9 long free-running stability verify。
N9 long free-run verify 已通过：旧脚本的 instrumented loop 原先仍走 action readout，修正为 predictive 且关闭 memory/identity 后，128 步 exact cycle、accuracy `1.0`、无 invalid/boundary action，膜电位/trace/threshold 边界、finite-state 和 public generate 对齐全部通过。

**当前唯一下一步**：继续执行 workflow 的 N10 sparse-kernel migration verify。
N10 sparse migration verify 已通过：forward/backproject/local_update 对 dense reference 的最大误差分别为约 `2.98e-8/0/0`；当前 v7、N7、N8、N9 行为均通过，checkpoint format、sparse storage 和无 dense synapse Gate 全绿。未改 kernel 或模型参数。

**当前唯一下一步**：继续执行 workflow 的 N11 active environment verify。
N11 active environment verify 已通过：learned policy 最终窗口准确率 `1.0`、总体 `0.965`、`200` 次局部 reward updates，较随机策略提升 `0.5`、较 action-learning lesion 提升 `0.375`；环境转移依赖动作、deterministic policy 解两 cue、pending action/experience 清零均通过。

**当前唯一下一步**：继续执行 workflow 的 M5 episodic field verify。
M5 episodic field verify 已通过。首个失败由两层评估错误组成：recurrent lesion 直接篡改 checkpoint payload，触发 v10 identity lineage 防篡改；改为 fresh restore 后在内存对象上 lesion。随后 identity organ 旁路使 lesion action 仍为 `1.0`，已在 M5 写入/查询统一 `use_identity=False`，保持 episodic field 独立。最终 full action `0.875`，trace-only/recurrent lesion `0.25`，recurrent causal gap `0.625`，outcome/provenance/time/episode/transaction checkpoint/fixed topology 全部通过。

**当前唯一下一步**：继续执行 workflow 的 M6 endogenous replay consolidation verify。
M6 endogenous replay verify 已通过。评估器原先对 action/outcome/association 做 checkpoint payload lesion，触发 identity lineage 防篡改；现统一 fresh restore 后在内存对象施加 lesion。full replay contingency accuracy `0.75`，no-replay `0.25`，content/recurrent lesion 均 `0.25`；sleep 只改变 cortex，memory field topology/write count 不变，settled-state/written-field guard 和无 episodic readback 全部通过。

S1 grounded internalization canary 已通过：native consolidation 对未见 grounded holdout 的 loss gain 为约 `0.359993`，internalized lesion loss `0.36`、grounding lesion loss `0.040714`，旧任务 retention、checkpoint roundtrip 和生命周期 `internalized` 均通过；该结果仍只证明 synthetic native canary，不授予物理删除外部描述、provider 执行或结构增长权限。

随后按 workflow 运行 `tests/seed`，首轮在仓库可写 fixture 下得到 `82 passed, 1 failed`；唯一真实失败为 `test_judge_ranks_learned_text_above_noise`。根因是 `Seed.learn_bytes()` 的 F1 `predictive` readout 与 `SeedJudge.score()` 默认 `action` readout owner 错位，judge 因而没有读取已训练语言器官。`seed/judge.py` 已将只读观测显式对齐 `readout="predictive"`，并关闭 memory/identity 旁路，保持 F1 语言质量评分不被 F2/identity shortcut 污染；没有放宽质量阈值或修改训练参数。修复后完整 `tests/seed` 在同一仓库可写 fixture 下为 `114 passed`，目标文件 ruff、py_compile 和 `git diff --check` 通过。

标准 pytest 内置 `tmp_path` 在本机仍可能创建 Windows 0700 临时目录并触发 ACL 错误；可写 fixture 版本是本轮真实失败账本依据，不能把环境清理噪声写成代码失败或代码通过。

**当前唯一下一步**：继续执行 workflow 的完整 `tests/` 回归，仍使用仓库可写 basetemp 区分真实失败与 Windows 临时目录 ACL 噪声；保持 R4 shadow/默认 parent 不变，R5 learned router、Skill/MCP/provider、客户端与 CUDA 继续冻结。

完整 `tests/` 回归已在同一仓库可写 fixture 下闭合：`1070 passed, 6 skipped, 1 warning`，没有真实测试失败；warning 是既有 FastAPI/Starlette 与 httpx 兼容提示。该结果覆盖 `tests/seed`、`tests/taiji_native` 及其余仓库测试，但本次诊断命令尚未带 CI 的 coverage/junit 参数。

**当前唯一下一步**：执行 workflow 完整回归对应的 coverage/junit 门禁，确认 `fail_under` 与报告生成在同一可写 fixture 下通过；若失败，只处理真实覆盖率/报告问题，不改变模型或 R4 shadow。

coverage/junit 门禁已闭合：同一仓库可写 fixture 下 `1070 passed, 6 skipped, 1 warning`，`coverage.xml` 成功生成，总覆盖率 `57.73%`，超过 pyproject 的 `21.8% fail_under`，退出码为 `0`。warning 仍为 FastAPI/Starlette 与 httpx 的既有兼容提示，不是本轮代码失败。

**当前唯一下一步**：对提交后的工作树执行最终静态 CI 门禁（版本一致性、主 Ruff、B/SIM、core mypy、变更脚本编译和 diff 检查）；Black 全量若再次受本机 Windows worker/ACL 阻塞，只记录为环境限制，不伪报通过，也不借机修改无关代码。

提交后静态门禁复核结果：版本一致性、主 Ruff、B/SIM、core mypy（`0 errors / 95 files`）、`seed/judge.py` 编译和 `git diff --check` 均通过；Ruff 扫描历史不可读目录时只产生环境 warning。Black 全量明确失败于扫描历史 `.m0-checkpoint-a4x_ye4_` 目录的 `WinError 5`，定向 Black 又出现无输出 worker 阻塞并已终止；因此 Black 在本机记为“环境受阻、未验证”，不是代码通过或代码失败。

**当前唯一下一步**：执行 CI 独立的前端 job（npm 依赖、ESLint、native boundary、API contract、Vitest、build 和 dist 存在性），继续只修真实失败；模型、R4 shadow/默认 parent 与 CUDA 不动。

前端 job 已完成本地等价审计：native boundary PASS，API contract PASS，ESLint `0 errors / 13 warnings`，Vitest `47` 个文件、`267` 个测试通过，Vite production build 通过且 `dist/index.html` 存在。精确 `npm ci` 两次均被本机 Windows cache/子进程 `EPERM` 阻塞；改用仓库可写 npm cache 并加 `--ignore-scripts` 后成功安装 `548` 个包，lockfile 与源码无改动。该安装 workaround 只用于本机验证，不能改写 CI 的 `npm ci` 语义；npm audit 的 `11 vulnerabilities` 是安装报告，CI 当前没有把它作为独立 blocking gate。

CI 交付层当前事实：后端完整测试/coverage、前端边界/测试/build、版本/Ruff/B-SIM/core mypy/编译/diff 均通过；本机仍无法验证 Black 全量（历史目录 ACL/worker），精确 npm ci 仍无法验证（Windows spawn/cache ACL）。这些是环境差异，不是模型或前端源码失败。

**当前唯一下一步**：进入 M4.V2.R4 收束决策审计，把 fixed-large 对照下结构增长未晋级的反证、可保留的 causal contribution、默认 parent 保护和下一版容量假设写成一份版本化架构决策；在该决策前不运行 R5 learned router、不继续调 Gate 输入、不引入 Skill/MCP/provider/client/CUDA 新变量。

R4 收束决策记录已完成：[M4V2_R4_CLOSURE_DECISION_20260909.md](../../reference/M4V2_R4_CLOSURE_DECISION_20260909.md)。它确认 9-cell 技术 Gate 全过，candidate lesion G/S 均为正，但 pressure 相对 fixed-large 的 G 均值仍劣化 `+0.001002`、仅 `4/9` 不劣，因此 `can_promote=false` 和默认 parent 保护不变。唯一推荐的后续假设是“资源归一化的条件模块化”：检验增长是否通过内容/状态路由节省早期活跃容量并改善未见组合 transfer，而不是继续微调当前 residual gate；在指标、对照和停止线预注册前不写实现、不训练、不解冻 R5。

**当前唯一下一步**：在该决策节点确认“条件模块化 + 资源归一化”假设的指标、对照和停止线；确认前保持代码、模型权重、R4 Gate 和外围系统冻结，避免路径漂移。

R5 预注册草案已完成：[M4V2_R5_CONDITIONAL_MODULARITY_PREREGISTRATION_20260909.md](../../reference/M4V2_R5_CONDITIONAL_MODULARITY_PREREGISTRATION_20260909.md)。它把 R4 决策 §4 的假设落成可证伪设计：三臂（A fixed-capacity / B fixed-large 不可删除 / C conditional module）× 同一 3×3 matrix；路由输入白名单（内容/状态/不确定性/目标/资源，禁止 task ID 与 phase 标签）；资源归一化协议（累计更新预算、峰值内存 ≤1.25×、墙钟 ≤1.5×，超限 cell 判无效）；主指标为未见组合 transfer（C 相对 B `non-worse ≥ 7/9` 且 mean 更优）+ route lesion 因果证明 + resource-normalized utility + 旧能力非劣；停止线为 canary 先行、formal 二分出口（晋级讨论 vs 主线收束转 M5）、全程 `can_promote=false`、默认路径保护。

**当前唯一下一步**：用户对草案 §7 的三个确认点拍板——(1) 资源上限初值（内存 1.25×、墙钟 1.5×）；(2) 主指标阈值（7/9 non-worse 且 mean 更优）；(3) 出口二分。确认后按 R4 同节奏进入实现（zero-gated shadow 外壳 → route learner 最小实现 → canary）；确认前不写任何实现代码。

三个确认点已获用户批准，R5 进入实现。已交付：`taiji/conditional_module.py` 的 `ConditionalRouteLearner`（白名单路由输入：content bucket / surprise EMA / parent residual norm / candidate activity / 常量 resource；局部可审计规则训练，无 autograd；checkpointable）；`scripts/training/eval_taiji_m4v2_r5_conditional_canary.py` 复用 R4 合成 parent 与 canonical 课程，把 shadow 的 `set_gate(1.0)` 替换为每 tick 路由动态 gate。实现教训：评分必须走路由流——R4 `_score` 每 tick 强制 `set_gate(1.0)` 会抹掉条件化，已改为全部评分经 `_r5_stream(learn=False)`；route lesion 与 candidate lesion 的探测顺序必须先 unlesion 路由再删候选，否则两个因果探针混淆。

`reports/taiji_m4v2_r5_conditional_canary_20260909.json`：11/11 技术 Gate 全过。关键因果证据——route lesion 使 G holdout surprise 恶化 `+0.08245`、S 恶化 `+0.00383`（未见组合对条件路由的依赖显著大于同分布），candidate lesion 独立可观测（S `+0.03804`、G `+0.01728`）；route gate 非常量（std `0.0385`）且全程在 [0,1]；parent substrate 与 mature owner digest 不变；shadow/route checkpoint 往返一致；route+candidate 活跃参数字节已记录。R4/R3/adaptive region 相邻回归 `8 passed`。

**当前唯一下一步**：按预注册草案把 canary 扩展为 3×3 formal——model seeds `71/83/97` × course seeds `101/202/303`，补 fixed-large 臂与资源归一化记录（累计更新预算、峰值内存 ≤1.25×、墙钟 ≤1.5×，超限 cell 判无效），按 §4 主指标（C 相对 B `non-worse ≥ 7/9` 且 mean 更优、route lesion、旧能力非劣）出二分判定。不调路由输入白名单、不改 Gate 阈值、不引入外围系统。

R5 3×3 formal 已完成：`reports/taiji_m4v2_r5_conditional_formal_20260909.json`，**假设否决（hypothesis_supported=false）**。资源归一化干净（0 违例）；C 相对 fixed-large 的 G delta 均值 `+0.00104`（正=更差）、non-worse `4/9`（需 ≥7）；S 非劣 `6/9`（需 9/9）。7 个技术有效 cell 中 route lesion 在 G 上 `5/7` 正，但 model97 的 202/303 两 cell 路由断开反而更好（G `-0.042/-0.008`）——条件路由在未见组合上不能稳定胜过常开。判定与 R4 的 fixed-large 反证同构：**条件化没有解开“等量预分配”的死结，结构增长主线按预注册出口收束，剩余路线转 M5 外围**。全部 `can_promote=false`；R5 代码与报告保留为可回滚 shadow 资产，默认 parent 未被污染。

**当前唯一下一步**：按 §6 解冻顺序启动 M5 外围第一项（Skill/MCP 成为有来源经历的知识内化 canary）；量尺从三周期 retention Gate 切换为 M5 的知识内化/真实任务 Gate，M4.V2 的 R4/R5 shadow 资产保持冻结且可回滚，不携带任何未晋级结构进入默认路径。

M5.S2 已完成第一轮：`taiji/internalization.py` 扩展 provenance 白名单（新增 `document-grounding` 来源族，lineage 前缀 `document:` 必须与 provenance 配对，防止跨族伪造；world-state 族行为不变，S1 回归 21 passed）；`scripts/training/eval_taiji_m5_s2_document_internalization.py` 把 M4.R12 的 UltraData 转换语料（真实开源记录，SHA-256 锚定）特征化为 12 维确定性文本统计（版本化 `m5s2-text-stats-v1`），以 **域级去来源 holdout** 设计通过内化 canary：训练 Knowledge+IF 各 3000 条真实来源经历，holdout 为完全未见来源域 Chinese-general 2000 条——holdout loss `1.0 → 0.000179`（gain `0.9998`）、internalized lesion `→1.0`（内化必要）、grounding lesion `→0.607`（来源必要）、retention `≈0`（旧任务保持）、checkpoint 往返一致，11 项检查全过。实现教训：M4.R12 manifest 的 digest 是裸 sha256 而非 canonical content digest，校验必须按同口径；replay_budget 是经历池容量参数，大批量来源经历需同步扩容（本 canary 8000）。边界如实声明：reward 为占位常量（真实 outcome 需 MCP/执行回路），特征为确定性统计而非语义 embedding——S2 证明的是“真实来源经历可进入内化管线并在未见来源域上泛化”，不是语义理解。

**当前唯一下一步**：M5.S2 的语义升级设计决策——(a) 特征化从统计特征升级为 embedding（需预注册 embedding 来源与依赖边界，v2 架构允许复用 embedding）；(b) reward 从占位常量接入真实 outcome（依赖 MCP/执行回路）；(c) 扩展 9-cell/多源矩阵。三项均未预注册，选择后按 R4/R5 同节奏（预注册 → canary → formal）推进；选择前 M5 停留在 S2 已验证边界内。

设计决策 (a) 已批准并执行：预注册 [M5_S3_SEMANTIC_INTERNALIZATION_PREREGISTRATION_20260909.md](../../reference/M5_S3_SEMANTIC_INTERNALIZATION_PREREGISTRATION_20260909.md)（模型 `paraphrase-multilingual-MiniLM-L12-v2` 384 维、纯 transformers mean pooling 零新依赖、revision/config digest 锚定、禁止云端推理）；`taiji/document_embedding.py` 的 `DocumentEmbedder` 与 `scripts/training/eval_taiji_m5_s3_semantic_internalization.py` 落地。`reports/taiji_m5_s3_semantic_internalization_20260909.json` 技术门 6/6 全过（域级去来源 holdout、checkpoint 往返、embedder 锚定）。**但须如实记录因果区分度退化**：384 维语义 embedding 下 loss 饱和到机器精度（holdout/retention loss 全为 0.0），grounding lesion 仅 `4e-06`（S2 统计特征下为 0.607）——来源必要性只剩名义边际；语义探针 same-domain 0.651 vs cross-domain 0.678，域分离度弱（三域均为中文问答，语义重叠大）。三个信号同向说明：**特征维度↑使内化任务过易，因果控制的区分度反而在 S2 统计特征上更健康**。S3 字面通过（预注册 Gate 全满足），区分度退化登记为开放观察。

**当前唯一下一步**：M5 内化的区分度恢复设计——预注册强化版 Gate（grounding lesion 边际下限、holdout 难度提升：同域细分来源或跨域真正异质来源、learner 容量与特征维度的配比约束），使语义 embedding 与统计特征两个版本都能产出可区分的因果证据；随后再进入 reward 真实化（决策 (b)）或多源矩阵（决策 (c)）。强化设计确认前不跑新 formal。

区分度恢复设计按 advisor 纠正后的路径执行，结论推翻了"直接强化 Gate"：

- **M5.S4（异质 holdout，排除"数据太易"）**：`scripts/data_prep/convert_ultradata_hetero_holdout.py` 把本地 UltraData 的 RL-2609 Math + SFT-Agent-2609 Code_Agent 各 2000 条转成 simple_zh 同构异质 holdout（SHA-256 `79c58330…`）；`scripts/training/eval_taiji_m5_s4_hetero_internalization.py` 严格同规格于 S3（384 维、train 6000 中文 QA、holdout 2000、retention 2000、S1 Gate），唯一变量是 holdout 换成 Math+Code。语义探针证实异质性成立（train↔Math 质心距离 `0.433`、train↔Code `0.369`，是 S3 域间距离的 ~15×），但 **grounding lesion 边际仍坍缩到 `4.23e-06`**、holdout_loss `1.2e-07`。→ "数据太易"不是根因。
- **M5.S5（隔离诊断，锁定根因）**：`scripts/training/diagnose_taiji_m5_s5_grounding_collapse.py` 固定 learner、只变 (a) target 是否常量 与 (b) 特征几何，测 grounding margin。结果决定性：**常量 target 下 margin 为负**（`-5.4e-09`/`-1.2e-09`，bias→1.0、weights→0）；**target 携带信息时 margin 恢复**（归一化 `0.0026`、未归一化 `0.464`，bias→0、weights→0.7~1.0）。机制：`score(grounding_enabled=False)` 只返回 bias，常量 outcome 完全由 bias 预测，grounding 不携带模型需要的信息。
- **必须上报的负面结论**：当前基于常量 reward 的 grounding lesion 探针**测的不是来源必要性**。S2 报告的 `0.607` 是"常量 target + 12 维小尺度统计特征"下 bias/weights 学习速率竞争产生的分布漂移伪影，不代表 grounding 携带 outcome 信息。S3/S4 的坍缩不是 embedding 吞掉信号，而是探针在常量 target 下结构性退化。`can_promote=false` 不变，S2/S3 的 holdout 泛化与 checkpoint 结论仍有效，仅 grounding lesion 一项的语义被本诊断撤销。

**当前唯一下一步**：reward 真实化（原设计决策 (b)）——诊断证明这是让 grounding/internalized lesion 探针重新有意义的唯一途径，且它本就是 M5 的本义（"Skill/MCP 成为有来源经历"要求真实 outcome，不是常量占位）。需预注册：outcome 来源（M3 Workbench 执行回路的真实 success/reward）、target 如何从 outcome 派生、grounding lesion 在真实 outcome 下的新判据。确认 outcome 来源前不写实现、不跑新 canary；不通过给常量 target 加 lesion 下限来"修"探针。

outcome 来源经用户确认为「M3 隔离工作区真实执行」，已预注册 [M5_S6_REAL_OUTCOME_PREREGISTRATION_20260909.md](../../reference/M5_S6_REAL_OUTCOME_PREREGISTRATION_20260909.md)（选项 A：成功执行上派生分级 reward、不改"拒绝失败证据"边界；选项 B：允许失败证据以负 reward 进入，属独立执行安全边界变更，A 未证明不足前不动）。探索确认链路基本现成：`WorkbenchEnvironment.read_workspace_evidence` 是真实无副作用只读传感器，`project_workbench_outcome_for_internalization` 已支持外部注入分级 reward，`WorldAffordanceGroundingProducer` 提供 17 维真实 grounding；fixture 仅 7 文件不足以支撑多任务，canary 用进程私有临时目录生成 360 文件。

`scripts/training/eval_taiji_m5_s6_real_outcome.py` 选项 A canary 已通过：360 条真实 `workspace.read` 执行，grounding 特征取真实读取结果（byte_length / digest buckets / content 统计），reward 由"读取内容是否命中声明目标 digest"派生（命中/部分/非目标三档，全部 success=True）。结果 `reports/taiji_m5_s6_real_outcome_20260909.json`：reward_variance `0.944`（target 真实变化）、**grounding lesion margin `0.2085`（远超预注册下限 0.05）**、holdout loss `1.0→0.848`（真实非饱和改善，与 S3/S4 的机器精度饱和形成对照）、internalized lesion 必要、retention 保持、lifecycle 达 `internalized`，5/5 检查全过。**S5 预测的"变化 target 恢复 lesion"在真实执行 outcome 上被证实。** 边界如实声明：read 执行真实、特征与 reward 真实派生自读取内容，但"任务目标"仍是构造的分级函数，尚未接入真实用户目标与 success/failure（那是选项 B / 完整执行回路）。

**当前唯一下一步**：S6 选项 A 已证明 lesion 恢复且幅度充足（0.209 ≫ 0.05），按预注册 §6 进入多 seed formal——固定 learner/课程、只变任务生成 seed（如 3×3），验证 grounding margin 恢复的稳健性（非单次抽样运气）；formal 通过后再评估是否值得为"真实 success/failure 进入学习"单独提选项 B 的边界变更预注册。formal 前不改执行安全边界、不引入 provider/联网/真实客户端写入。

S6 选项 A formal（3×3，task_seed × learner_seed）已完成：`reports/taiji_m5_s6_real_outcome_formal_20260909.json`，**结果不稳健（margin_recovery_robust=false）**：6/9 cell 过 floor、仅 5/9 过完整 causal gate，margin 均值 `0.189`、min `−0.613`。canary 的 0.2085 部分依赖 learner_seed=17 的有利抽样——task_seed=0 下 margin 随 learner_seed 剧烈摆动（17→0.209、23→−0.613、31→0.011），task_seed=1 三 seed 全正（0.255~0.710）、task_seed=2 混合。**关键疑点**：reward_variance 在全部 9 cell 完全相同（0.944121），而 learner 是零初始化确定性、`learner_seed` 只进 `InternalizationConverter(seed=…)`——它不该改变数值却导致 margin 剧变，指向 converter 内一条未受控的 seed→数值路径。这可能是 harness bug（可定位修复），也可能是恢复本身是脆弱的真变异性。

**当前唯一下一步**：定位 `InternalizationConverter` 的 seed 数值路径（读 `convert`/replay 采样，确认 learner_seed 是否经由 example 顺序、target 变换或 replay 抽样影响零初始化 learner 的 consolidate）。若为 bug → 修复后重跑 formal（同预注册判据，不放宽）；若为真变异性 → 如实判定选项 A 的 lesion 恢复不可信，回到 reward 派生设计（更宽 reward 带或更多 passes 的容量配比），不通过挑 seed 制造通过。解释这条 seed 敏感性之前不进入选项 B 边界变更。

seed 敏感性已定位并修复，S6 formal 重跑通过：

- **根因（非 converter bug，是 reward 设计缺陷）**：`_examples`（`internalization_learner.py:35`）按 `example_id` 排序，而 `example_id` 含 `converter.seed`，故 `learner_seed` 改变训练**顺序**；在线 SGD 对顺序敏感。真正的病灶是原分级 reward 由 `index%3` 的 band 决定，但 band 在 grounding 特征里几乎不可读（content-token 特征只能区分一档、exact_match 恒真）——target 只能被特征**弱预测**，顺序就决定了线性 learner 过/欠拟合，margin 随 seed 剧变（0.209→−0.613→0.011）。canary 的 0.2085 是"弱可学习 target + 恰好有利顺序"的假稳健。
- **修复**：`_graded_reward` 改为真实读取数值属性（byte_length、content-token 密度）的固定 tanh 组合，落在 [−1,1]——即 S5"信息 target"情形的诚实版本，只是特征/reward 全部来自真实 `workspace.read` 执行。同时删除已失效的 band/`TARGET_BANDS` 脚手架（收敛清理）。
- **验证**：task_seed=0 三 learner_seed margin 稳定 0.090~0.104（不再摆动）；完整 3×3 formal `reports/taiji_m5_s6_real_outcome_formal_20260909.json` **9/9 过 floor、9/9 过完整 causal gate**，margin 均值 `0.1067`、min `0.0904`、max `0.1232`，reward_variance 稳定 ~0.338。**S6 选项 A 的 lesion 恢复被证实为稳健**：真实执行 outcome 驱动的分级 reward 让 grounding/internalized 探针在跨 seed 下一致有意义。

**当前唯一下一步**：S6 选项 A 已稳健通过，按预注册 §6 评估是否进入选项 B（允许真实 success/failure 以负 reward 进入内化）——这是一个**独立的执行安全边界变更**（`project_workbench_outcome_for_internalization` 现拒绝 `success=False`），需单独预注册评审，不在本轮自动解冻。若保持选项 A，则 M5 内化证据线已闭合到"真实只读执行 + 分级 reward 稳健恢复因果探针"，下一步转向 M5 的第二项（MCP 硬件/客户端能力继承验证）或 K 课程（技能组合，A8 主证据）。选择前不引入 provider/联网/真实客户端写入。

用户已确认选项 B。预注册 [M5_S6B_FAILURE_ADMISSION_PREREGISTRATION_20260909.md](../../reference/M5_S6B_FAILURE_ADmission_PREREGISTRATION_20260909.md) 并实现边界变更——探索发现旧政策是**四点协同设计**（workbench.py 的 projection_bindings 失败返回空 + to_taiji_affordances 失败返回空、seed_runtime 的 reproject 空即 raise + 内化投影拒绝失败），其中 affordance 链承担"下一步建议"语义不应改动。最终变更范围：(1) `taiji/internalization.py` converter 显式失败准入政策（failure ⇒ reward ≤ 0，违反即 `failure_with_positive_reward` 拒绝；lifecycle 新增 `failure_admitted` 事件）；(2) `seed_runtime.py` 内化投影对齐同一政策，失败证据经**同一 world-state grounding producer、同一 17 维契约**自 grounding（affordance 链零改动，失败 affordance 不进入下一步建议）；(3) metadata 加 `failure_admitted` 审计标记。新增双向单元测试钉住政策（失败+负 reward 准入、失败+正 reward 拒绝），`tests/test_workbench_contract.py` 48 passed、core mypy 0（97 文件）。

`scripts/training/eval_taiji_m5_s6b_failure_admission.py` 走**完整 runtime 链**（SeedRuntime → 真实混合执行 → 投影 → 内化）：canary 一次通过（120 条失败证据全部准入、margin `0.363`、holdout loss `1.0→0.043`）；3×3 formal `reports/taiji_m5_s6b_failure_admission_formal_20260909.json` **9/9 稳健**（margin `0.317~0.400`、均值 `0.367`，跨 task/learner seed 无摆动——比选项 A 更稳，真实失败携带强且清晰可读的信号）。**M5 内化的失败半边闭合：真实成功与真实失败都是有来源经历，因果探针在两类证据上稳健有意义。** 顺带修复 S2 遗留的 mypy 类型冲突（`provenance` 变量名 str/tuple 撞名，当时只查了脚本未查库文件）。

**当前唯一下一步**：M5 内化证据线（S2 统计特征 → S3 语义 embedding → S4 异质 holdout → S5 根因诊断 → S6 真实执行 → S6B 失败准入）已完整闭环，全部预注册、全部可回滚、默认路径未被未晋级结构污染。下一步二选一：(a) K 课程（技能组合，A8 主证据——结构化语义 → 世界转移 → Workbench 只读意图 → 隔离执行，每阶段未见组合+真实 outcome，S6B 的真实 outcome 链路可直接复用）；(b) M5 第二项（MCP 硬件/客户端能力继承验证）。建议 (a)，理由：内化管线的真实 outcome 来源已闭合，K 课程正是消费它的主场。选择前不引入 provider/联网/真实客户端写入。

用户已确认 (a)。K 课程预注册 [M5_K1_SKILL_COMPOSITION_PREREGISTRATION_20260909.md](../../reference/M5_K1_SKILL_COMPOSITION_PREREGISTRATION_20260909.md) 已完成：可证伪假设 = 同一 runtime 上串联四阶段后，**未见任务组合**（训练未见过的 目标×文件×语言 三元组，每个元素都已见——严格组合泛化定义，吸取 S3"holdout 太易"教训）上完整链的真实执行成功率显著高于冻结语义对照，且语义 lesion 使其崩塌。三臂（full-chain / frozen-semantic / semantic-lesion）+ 主指标（A 成功率 ≥0.8、A−B ≥0.3、A−C ≥0.3）+ 真 outcome 全走 S6B 政策。资产盘点结论：四阶段 learner/planner/执行 API 全部现成（M3.R3 已串 1+2→3），唯一新写胶水是参数化任务组合生成器与 3→4 的真 outcome 闭环（M3.R3 native 臂产出 `decision.action_intent` → M5.S6B 的 `execute_workbench_intent`→project→ingest）。

**当前唯一下一步**：实现 K1 canary——(1) 读 M3.R3 的 corpus/sequence 构造与 M3.R1 的 `build_course`（组合生成器的基底）；(2) 写参数化 (文件×语言×目标) 网格生成器（train/holdout 按组合维分割）；(3) 串联：语义训练 → 转移训练 → 未见组合上 planner.propose → SeedRuntime 真实执行 → 真 outcome 成功率；(4) 三臂对照 + lesion。先 canary 后 formal；A 臂成功率 <0.5 先归因不调阈值；全部执行在进程私有临时工作区；`can_promote=false` 固定。

K1 canary 已实现并运行（`scripts/training/eval_taiji_m5_k1_skill_composition.py`，报告 `reports/taiji_m5_k1_skill_composition_20260909.json`），**Gate 未通过（honest fail）**：A 臂 unseen 成功率 `0.0`（需 ≥0.8），B/C 正确崩塌（`conflict` fail-closed），Stage-2 接口贯通 6/6，全链在已见组合上真实闭合（python inspect 成功+S6B 准入、rust clarify 成功）。按停止线完成归因而非调阈值：失败精确定位在 Stage-1 fact head 的属性组合——训练边际完全相关（python↔available、ts/rs↔missing）使线性欠定 fact head 把状态属性绑到身份特征（实测 ts05 的 `toolchain::available=0.135`，而其因果特征 toolchain scalar=1.0；`language::typescript=0.830` 组合成功），planner 对错误世界 fail-closed 正确。训练 registry 中 ts 工具链必然不可用，该交叉训练不可达，harness 数据无解。实现修正（holdout 全覆盖、拆分防泄漏、合成内容去污染、Stage1/2 分离）与完整归因见预注册 §6 执行记录。

**当前唯一下一步**：K1.1 预注册确认——类型化 fact–feature 绑定（fact 按 attribute 掩码读取 `WorkbenchObservationSchema.feature_names`，goal/content 读出掩码到状态类 facts，身份 facts 走世界与 content.semantic_slots）＋掩码的 checkpoint/恢复契约；确认后实现并原样重跑 K1（同判据，不放宽）。确认前不改 `taiji/semantic_training.py`、不重跑 K1、不引入新变量。

用户已确认 K1.1。实现与重跑完成，**K1 canary Gate 通过**：`taiji/semantic_training.py`（`SEMANTIC_TRAINING_VERSION 1→2`）给 `StructuredSemanticLearner` 加可选 `fact_feature_masks`（精确覆盖校验，fail-closed；初始化/每 epoch/checkpoint 恢复三处重执行掩码）与 `readout_excluded_facts`（goal/content 读出剔除身份 facts，fit/predict 两侧一致）；未提供时行为与 v1 逐位一致。K1 脚本由 schema 规范派生绑定（`language::*`→语言 one-hot、状态谓词→同名 scalar），绑定写入报告 `typed_binding`。重跑结果（同判据未放宽，单 seed）：**A unseen `1.0`（3/3 未见三元组 `resolved → workspace.read → 真实执行成功 → S6B 准入`）、A−B=`1.0`、A−C=`1.0`（B/C conflict fail-closed）、Stage-2 接口 6/6**，4 项 Gate 全过，`can_promote=false` 固定。M2.R3 回归 5 passed、M3.R1 回归 2 passed、mypy 0。边界如实声明：fact→feature 掩码是 harness 依 schema 先验派生的表征约束（learner 学的是掩码内校准），约束本身的普适化属后续课程问题；resolve 成功 outcome 的 S6B 投影 stale 限制为已知次要项。详见预注册 §7。

**当前唯一下一步**：K1 multi-seed formal 预注册——把 canary 扩展为 model/course seed 矩阵（K 量尺=任务成功率，量尺合同沿用 M4.V2.R0 的 metric spec：绝对成功率、parent delta、comparison delta 分离），验证组合成功率的跨 seed 稳健性；formal 判据在看结果前预注册，不挑 seed、不改判据；`can_promote=false` 直到 aggregate 计算。

K1 formal 预注册已完成：[M5_K1_FORMAL_PREREGISTRATION_20260909.md](../../reference/M5_K1_FORMAL_PREREGISTRATION_20260909.md)。矩阵 `3 task_seed × 3 learner_seed = 9 cell`，每 cell 复用 canary `run_cell`（A/B/C 三臂、K1.1 类型化绑定零改动）；量尺合同明确 K1 为全新能力轴——链不写任何 F1/记忆 owner，故 parent-retention 以 `null` 显式缺省（不伪造父代基线），主量尺为绝对任务成功率 + 臂间 delta。判据（看结果前冻结）：技术门（canary 4 检查 + Stage-2 接口 + 全行真实执行）；主判据 `9/9` cell A unseen ≥0.8 且均值 ≥0.9、`9/9` cell A−B ≥0.3 且 A−C ≥0.3、`9/9` cell 成功 read outcome S6B 准入率 =1.0（resolve stale 为已知次要项不计入）、`9/9` cell reward_variance >0。停止线：任一门失败即停并逐行归因，不调掩码/阈值/不挑 seed。报告 `taiji-m5-k1-skill-composition-formal-v1`，`can_promote=false` 固定；formal 通过仅闭合 K1 证据线，不等于 A8 晋级。

**当前唯一下一步**：实现 K1 formal 脚本（`eval_taiji_m5_k1_skill_composition_formal.py`，逐 cell import 复用 canary `run_cell`，串行 9 cell + aggregate，零逻辑复制、零判据改动），运行并产出 `reports/taiji_m5_k1_skill_composition_formal_20260909.json`；任一门失败按预注册 §5 停止线处理。

K1 formal 已实现并运行，**9/9 cell 全过，K1 证据线闭合**。首轮在 task_seed=2 的 corpus 构造阶段暴露 harness 缺陷（`variant=(index+task_seed)%8` 的数字宽度随 task_seed 变化，task_seed=2 时 py04/py05 byte_length 碰撞 → dev/test `input_digest` 跨 split 泄漏崩溃）；按预注册 §5 停止线归因为 harness 参数化缺陷（运行前崩溃，非判据结果），确定性修复为 pad 间距 2 字节/index（同语言文件长度对任意 task_seed 严格互异），修复后 canary 重跑 gate 仍全过。formal 结果（判据冻结）：A unseen min=mean=max=**1.0**（9/9 ≥0.8，均值 1.0 ≥0.9）、A−B=A−C=**1.0**（9/9 ≥0.3）、成功 read outcome S6B 准入率每 cell `4/4`（9/9 =1.0）、reward_variance 9/9 >0、技术门 9/9、parent_retention 显式 `null`（无父代基线，不伪造）；`status=passed`、`robust=true`、`can_promote=false` 固定。报告 `reports/taiji_m5_k1_skill_composition_formal_20260909.json`，修正与结果记录在预注册 §8。

**当前唯一下一步**：K 课程的下一步设计确认——(a) **K2**（更深组合维度：多步任务序列/任务间依赖，如 recover → clarify → inspect 链式未见组合，要求转移头进入主路径的组合）；或 (b) **A8 K 轴 formal**（把 S/G/K 矩阵纳入 R6 候选的 aggregate scorecard 设计讨论）。两者均需先预注册；选择前不改判据、不引入新变量，`can_promote=false` 直到 aggregate 计算。建议 (a)：K1 证明了单步属性组合，K2 是把转移头（现仅为接口贯通）提升为主路径组合参与者的自然下一步。

K2 canary 预注册已完成：[M5_K2_MULTISTEP_COMPOSITION_PREREGISTRATION_20260909.md](../../reference/M5_K2_MULTISTEP_COMPOSITION_PREREGISTRATION_20260909.md)。可证伪假设 = 转移头 autoregressive 世界驱动的完整链（转移世界 → Stage-1 语义 → 只读意图 → 隔离执行）对**未见 (任务序列 × 语言 × 文件) 组合**的序列级成功率显著高于冻结对照且转移 lesion 崩塌。两个关键设计：(1) **K2.1 转移头类型化绑定**——delta(fact) 行掩码只允许「before 同 fact 持久性 + event 同属性特征指示 + 其交叉」，结构性消除 K1 §6.3 的块级捷径（`SEMANTIC_TRANSITION_VERSION` bump，掩码 init/每 epoch/checkpoint 恢复三处重执行，向后兼容）；(2) **主路径世界流**——初始世界为中性 M 块锚（missing_00，所有 episode 共用），每步 `after_world = transition.predict(current, event)`，planner 消费 after_world + Stage-1 的 goal/content（Stage-1 零改动），纯 autoregressive，outcome 不反馈世界（K3 范围）。计量：训练 episodes ≥6 个 3 步序列，holdout 4 个未见序列，「序列成功」= 全部步接受且真实执行成功（episode 粒度，严于 K1 单步）。Gate：A ≥0.75、A−B ≥0.25、A−C ≥0.25，技术门含「训练 episodes 上 A 序列成功率 =1.0」与「K2.1 掩码 checkpoint 往返后仍强制执行」；A <0.5 先归因不硬凑。明确不做：outcome→世界反馈、任务间依赖（K3）、formal、改 K1 资产、转移 learner 的 goal/content 读出接进主路径。

**当前唯一下一步**：实现 K2——(1) `taiji/semantic_transition.py` 的 K2.1 转移头行掩码与 checkpoint 契约（版本 bump）；(2) 新脚本 `eval_taiji_m5_k2_multistep_composition.py`（复用 K1 的 workspace/registry/schema/绑定派生/planner/执行/准入链）；(3) 运行 canary 产出 `reports/taiji_m5_k2_multistep_canary_20260909.json`；任一门失败按 §4 停止线逐行归因。
K2 canary 已完成：`reports/taiji_m5_k2_multistep_canary_20260909.json` 通过；A full-chain 未见三步 episode `4/4` 成功、B frozen/C transition-lesion 为 `0/4`，A 训练 `6/6`，S6B 成功 read 准入、K2.1 checkpoint/篡改防护、reward variance 全部通过。首轮暴露的 executor 恒定 reward 与 Windows 临时目录 ACL 已分别按既有 S6 graded-outcome 口径和 repo-writable temp 模式收敛，未改变模型判据；`can_promote=false` 不变。

**当前唯一下一步**：先完成并冻结 K2 formal 预注册（`task_seed=0/1/2 × learner_seed=17/23/31`、逐 cell 技术门、A 绝对成功率与 A-B/A-C 分离阈值、read admission、reward variance、停止线），再实现 formal runner；预注册完成前不跑 formal、不引入 K3 的 outcome→world 或任务依赖变量。

K2 formal 已完成：`reports/taiji_m5_k2_multistep_formal_20260909.json`，9/9 cells 通过，A holdout/A-B/A-C 的 min/mean/max 均为 `1.0`，训练、checkpoint/mask、S6B read admission、reward variance 全部 `9/9`；K2 转移头只作为 shadow 资产保留，`can_promote=false`。

**当前唯一下一步**：将 M5.K1/K2 结果整理进 A8 K 轴 scorecard，明确「K2 结构是否有资格进入默认 Taiji 路径」的 aggregate 入口与否决条件；在 scorecard 冻结前不启动 K3、不引入 MCP/provider/client/CUDA，也不把 shadow 结构接入默认 runtime。

A8 K 轴 scorecard 已完成：`reports/taiji_m5_k_axis_scorecard_20260909.json`。K1/K2 证据均闭合，content-addressed absolute snapshots 与 A-B/A-C comparison evidence 已分离；但 parent retention 缺失、learner 仍是 standalone shadow、默认 runtime 未接入，故 `promotion_gate=false`、`can_promote=false`。

K3 预注册已完成：[M5_K3_OUTCOME_WORLD_DEPENDENCY_PREREGISTRATION_20260909.md](../../reference/M5_K3_OUTCOME_WORLD_DEPENDENCY_PREREGISTRATION_20260909.md)。实际缺口已冻结为：runtime 已记录 `WorkbenchTaijiEvidence → WorldEvent`，但 K2 只把真实 outcome 用于 S6B 准入，下一步仍消费预测 `after_world`；K3 必须验证 typed outcome/dependency projection 是否真正进入下一步 world、任务依赖和 lineage gate。三臂固定为 A full-feedback、B no-feedback、C outcome-lesion；真实 success/failure、checkpoint/restore、stale/duplicate/cross-episode fail-closed 都是技术门，`can_promote=false` 固定。

K3 projection contract 已实现并单测：`taiji/outcome_dependency.py` 提供 content-addressed outcome/dependency projection、同 tick world enrichment、checkpoint/restore 和 lesion/stale/duplicate/cross-scope fail-closed；定向 7/7、K2/执行回归 13/13、ruff/py_compile/mypy 通过。工作台全量回归的剩余阻断是本机 pytest temp ACL，不是 projection 断言失败。projection 仍是 shadow，尚未接入 runtime 或执行 canary。

K3 单 cell canary 已完成：`scripts/training/eval_taiji_m5_k3_outcome_dependency.py` 与 `reports/taiji_m5_k3_outcome_dependency_20260909.json`。首轮暴露并修正了三类 harness 问题：holdout 类型未进入 transition vocabulary、train/holdout 工作区内容未分离、runtime workspace root selector 未随隔离 root 切换；随后补齐跨语言训练组合并把固定 transition 训练预算从 320 提到 1280，未改变阈值或 A/B/C Gate。正式单 cell 结果为 A `1.0`（4/4）、B `0.0`、C `0.0`，A-B/A-C 均 `1.0`，训练 A `6/6`，probe admission、lineage、checkpoint、feedback fact consumption 全部通过；`can_promote=false` 仍固定，默认 runtime 未接入。

**当前唯一下一步**：冻结并实现 K3 formal 预注册——先把单 cell 已验证的 workspace 分离、全语言 vocabulary、跨语言 sequence coverage、训练预算和三臂技术 Gate 写成 `3 task_seed × 3 learner_seed` 的 formal 合同；预注册完成前不跑 formal、不接默认 runtime、不引入 MCP/provider/client/CUDA。

K3 formal 合同已冻结：[M5_K3_FORMAL_PREREGISTRATION_20260909.md](../../reference/M5_K3_FORMAL_PREREGISTRATION_20260909.md)。矩阵固定为 task seed `0/1/2` × learner seed `17/23/31`，逐 cell 复用 canary `run_cell`，主判据为 A holdout `≥0.75`、A-B/A-C 各 `≥0.25`，并保留 probe admission、feedback lineage、checkpoint/lesion、reward variance 和 feedback fact consumption 技术门；单 cell 失败不由 aggregate mean 抵销，`can_promote=false` 不变。

**当前唯一下一步**：实现 `scripts/training/eval_taiji_m5_k3_outcome_dependency_formal.py`，只 import canary 的 `run_cell`，先做 py_compile/ruff/mypy，再串行运行 9 个 cell；formal 前不改 canary、不接默认 runtime、不引入 MCP/provider/client/CUDA。

K3 formal 已完成：`reports/taiji_m5_k3_outcome_dependency_formal_20260909.json`，9/9 cell 通过。A holdout、A-B、A-C、A train 的 min/mean/max 均为 `1.0`；technical、probe admission、feedback lineage admission 均 `9/9`；feedback reward variance 全部为正。首轮 `0/9` 是 runner 字段契约误判，已在不改 canary/判据的前提下修正并用同一矩阵重跑。K3 仍是 standalone shadow，`can_promote=false`，默认 runtime 未接入。

**当前唯一下一步**：把 K3 formal 与已有 K1/K2 formal 一起纳入新的 content-addressed K 轴 scorecard 版本，显式记录 K3 的 outcome/dependency 能力、三臂因果分离和仍缺失的 same-parent/default-runtime promotion Gate；scorecard 更新前不解冻任何 shadow owner，不引入 MCP/provider/client/CUDA。

K 轴 scorecard v2 已完成：`reports/taiji_m5_k_axis_scorecard_v2_20260909.json`，K1/K2/K3 source digest、absolute snapshots、A-B/A-C comparison 和 K3 admission/lineage evidence 均已纳入。`k_evidence_closed=true`，但 `promotion_gate=false`、`can_promote=false`：same-parent retention、default runtime owner、resource/rollback/old-capability Gate 均明确未通过且未伪造。

**当前唯一下一步**：为同一 parent 的 A8/R6 promotion course 冻结独立预注册合同，定义如何把 K1/K2/K3 shadow evidence 接到连续训练/运行时，同时验证 parent retention、资源等价、rollback 和旧能力非劣；合同冻结前不解冻任何 shadow owner，不引入 MCP/provider/client/CUDA。

同一 parent 的 A8/R6 promotion formal 已冻结：[M4V2_R6_A8_PROMOTION_PREREGISTRATION_20260909.md](../../reference/M4V2_R6_A8_PROMOTION_PREREGISTRATION_20260909.md)。合同明确当前真实入口：R4 相对 fixed-large 的 structural growth 仍未晋级、R5 router 尚未解冻、K1/K2/K3 仍是 standalone shadow；所以 R6 不能直接运行。其 9-cell 合同先要求 parent checkpoint/owner/source/resource/rollback preflight，再要求 native K adapter 与 S→G→K 同一 parent 连续课程；`can_promote=false` 继续固定。

same-parent K adapter 已实现并完成 preflight：`reports/taiji_m4v2_r6_k_adapter_preflight_20260909.json`，13/13 checks 通过；新增 `tests/taiji_native/test_m4v2_r6_k_adapter.py`，3 tests 通过。验证内容包括 parent/candidate namespace、owner/source/resource digest、K3 dependency projection、prefit/staged/rollback checkpoint fresh restore、同一 parent 绑定和 explicit rollback。全程 `training_performed=false`、`default_runtime_attached=false`、`candidate_promoted=false`，所以这只是 R6 入口技术 Gate，不是 formal/promotion；R4/R5/parent baseline/resource/old-capability retention 仍未解冻，`can_promote=false` 不变。

只读 admission audit 已完成：`reports/taiji_m4v2_r6_admission_audit_20260909.json`。审计 content-addressed 读取 R4/R5/K v2/R6 preflight；R4 technical evidence、R5 rejection、K evidence closed、R6 adapter preflight 和 R5 resource caps 已确认，但 R4 structural admission、R5 router/no-router 边界、same-parent retention baseline、完整 S→G→K、全 arm resource/old-capability Gate 与 CI/native ledger 确认仍缺失。结论 `blocked_shadow_only`，`can_start_r6_formal=false`、`can_promote=false`。

R6 fixed-capacity parent admission addendum 已冻结：[M4V2_R6_FIXED_CAPACITY_ADMISSION_ADDENDUM_20260909.md](../../reference/M4V2_R6_FIXED_CAPACITY_ADMISSION_ADDENDUM_20260909.md)。它明确选择 fixed-capacity parent continuation：R4 growth、R5 learned router 和 K standalone learner 都保持 shadow；R6 固定 no learned router；先做 `9 cells × 3 baseline repeats` 校准 epsilon，再做 adapter/all-arm checkpoint Gate，未通过不训练、不接 default runtime。CPU 是唯一执行设备，CUDA/provider/MCP/client/联网不进入变量。

parent baseline/preflight 已实现并运行：`reports/taiji_m4v2_r6_parent_baseline_preflight_20260909.json`。S/G 的 `9 cells × 3 repeats`、epsilon 校准和 6 类 arm 的 checkpoint/fresh restore/rollback 机械 Gate 全部通过，所有 arm `training_performed=false`。但按真实边界只测得 S/G，K parent retention 没有从 standalone K1/K2/K3 推断，报告明确 `baseline_complete=false`、`can_start_r6_formal=false`；因此仍不能训练或接 default runtime。

单 cell controlled adapter smoke 已完成：`reports/taiji_m4v2_r6_adapter_controlled_smoke_20260909.json`，16/16 checks 通过。`KAdapterInput/Output/Exchange` 已进入 native adapter，same-parent、K3 dependency/projection lineage、stage/fresh restore/rollback、S/G old-capability retention 全部通过；但这是 `fixture_outcome_only` 的 transport smoke，不是 K learner 能力结果，`k_learner_owner_attached=false`、`k_learner_training_performed=false`、`baseline_complete=false`、`can_start_r6_formal=false`。

K learner-owner attachment contract 已冻结：[M4V2_R6_K_LEARNER_OWNER_ATTACHMENT_CONTRACT_20260909.md](../../reference/M4V2_R6_K_LEARNER_OWNER_ATTACHMENT_CONTRACT_20260909.md)。它把 adapter 定义为唯一 candidate/rollback owner，K1 semantic、K2 transition 为 subordinate worker，K3 为 deterministic projection；明确了真实 checkpoint/owner manifest、joint digest、训练前 restore、原子 rollback 和禁止把 standalone report 冒充 worker checkpoint 的规则。

K worker manifest + joint checkpoint attachment preflight 已实现并运行：`taiji/k_worker_manifest.py` 固化 K1/K2/K3 manifest、输入/输出合同 digest、同 parent/source/resource 的 worker bundle 与 owner graph；`KContinualAdapter` 已支持 bundle 原子挂接、joint checkpoint/fresh restore、typed exchange ledger 与 rollback 保留。首次扫描没有真实 artifact，报告曾诚实为 `status=artifact_missing`；该报告与入口 contract 已提交留痕。

随后实现并运行 `scripts/training/build_taiji_m4v2_r6_k_worker_artifacts.py`：复用冻结 M5.K2 course，在 CPU 上生成 K1 semantic（`2080` steps）、K2 transition（`5040` steps）和 K3 deterministic projector（`0` steps）三类 artifact。训练前 checkpoint 落盘/回读、训练后 fresh restore 全部通过，报告为 `reports/taiji_m4v2_r6_k_worker_artifact_build_20260909.json`。构建首轮发现 preflight/builder 的 parent episode id 不一致，已统一复用 R6 baseline factory，并加入回归测试。

重新运行 attachment preflight 后，`reports/taiji_m4v2_r6_k_worker_attachment_preflight_20260909.json` 为 `status=passed`：同 parent bundle、owner graph、typed exchange、S/G retention、candidate stage、rollback 与 joint checkpoint 全部通过；`attachment_gate_passed=true`、`can_start_k_controlled_canary=true`，但 `can_start_r6_formal=false`、`can_promote=false`。没有 default runtime/provider/MCP/client/CUDA，也没有把 worker artifact 当成 promotion evidence。

single-cell controlled K canary 已实现并运行：`scripts/training/eval_taiji_m4v2_r6_k_worker_controlled_canary.py`，预注册合同为 `plans/reference/M4V2_R6_CONTROLLED_K_CANARY_PREREGISTRATION_20260909.md`，报告为 `reports/taiji_m4v2_r6_k_worker_controlled_canary_20260909.json`。首轮失败定位为隔离 Workbench root 未覆盖 `get_setting("workspace_path")`，导致默认 workspace 对真实路径返回 `not_found`；按已通过的 M5.K2 runner 模式修正 root selector 后，未改模型、artifact 或判据，重跑通过。

最终结果 `status=passed`：K1/K2 typed result 均 `resolved`；`typescript_05.ts` 真实 `workspace.read` 成功、reward `1.0`；K3 接受并应用真实 outcome/dependency projection；exchange 保存三类 worker manifest lineage；joint checkpoint、candidate stage、explicit rollback、S/G retention 全通过。`training_performed=false`、`candidate_training_performed=false`、`candidate_promoted=false`，provider/MCP/client/CUDA/default runtime 均未接入；`can_start_r6_formal=false`、`can_promote=false` 保持。

R6 single-cell controlled K canary 的证据边界已经审阅并冻结：[M4V2_R6_FORMAL_RUNNER_INPUT_CONTRACT_20260909.md](../../reference/M4V2_R6_FORMAL_RUNNER_INPUT_CONTRACT_20260909.md)。真实 Workbench outcome、K3 dependency projection、typed worker lineage、joint checkpoint 和 rollback 已在一个 cell 通过；但 causal 对照/lesion、peak resource 和 all-arm side-effect ledger 尚未具备，不能把 canary 当成 formal 或 promotion。

formal runner 现在被限制为只消费显式 content-addressed input manifest：固定 `model_seed=17/23/31`、`course_seed=0/1/2`、baseline repeats `401/503/607`、`S→G→K`、CPU、parent/worker/course registry；禁止隐式 parent factory、目录猜 artifact 或按结果重选 seed。失败必须按 input/lineage/checkpoint/course/environment/worker/outcome/projection/causal/retention/resource/side-effect/aggregate 的优先级写入 cell/arm/step/digest 归因。

R6 formal input manifest preflight 已实现并运行：`plans/manifests/taiji_m4v2_r6_formal_input_v1.json` 已生成，model 17/23/31 三份 parent checkpoint 和各自 K1/K2/K3 worker bundle 均完成 save/fresh-restore/digest Gate；course/seed/arm/resource/side-effect/failure contracts 全部通过。报告 `reports/taiji_m4v2_r6_formal_input_manifest_preflight_20260909.json` 为 `status=passed`、`formal_input_ready=true`。这仍只是输入 Gate，没有运行 S→G→K full formal，也没有把 `can_start_r6_formal` 改为 true。

R6 formal runner 入口层已实现并通过静态/输入 Gate：`scripts/training/eval_taiji_m4v2_r6_formal.py` 重新验证 manifest/input preflight，生成 `9 cells × 5 arms = 45` 个 `not_started` ledger row；报告 `reports/taiji_m4v2_r6_formal_preflight_20260909.json` 明确 `status=input_ready`、`formal_input_ready=true`，但 `course_executed=false`、`can_start_r6_formal=false`、`can_promote=false`。没有执行 S→G→K，没有接 default runtime/provider/MCP/client/CUDA。

R6 首轮 single-cell 曾因 fixed-large 只有旧 structural shadow 而以 `input_contract / fixed_large_k_control_required` 阻断；该结果被保留为历史证据，不把 structural shadow 冒充 K 对照。candidate 的真实 Workbench read→K3 accepted→dependency→rollback 与 S/G retention、同路径 K3 lesion 的 fail-closed 边界均已通过。

fixed-large 设计已冻结：[M4V2_R6_FIXED_LARGE_K_CONTROL_PREREGISTRATION_20260909.md](../../reference/M4V2_R6_FIXED_LARGE_K_CONTROL_PREREGISTRATION_20260909.md)。它使用两个独立 native K1/K2 worker replica（worker-training task seed `3/4`）和固定 arithmetic ensemble；旧 R4 structural shadow 不再作为 R6 K control。

native fixed-large builder 已通过：宽度 `2`、参数量 `9666`、训练 task `3/4` 与 formal holdout `0/1/2` 的路径/semantic/transition digest 交集为空，ensemble/K3 fresh restore 通过。其 single-cell comparator 已接入正式五臂 runner；当前报告 `status=passed`，fixed-large 真实 read、K3 lineage、rollback 和 branch lesion 均通过，`can_start_r6_formal=false`、`can_promote=false` 仍保持。

paired resource ledger 已闭合：candidate/fixed-large 均有同方法 peak-RSS、checkpoint-write、parameter/inference、wall-clock 和 measurement-complete，K 单步均为 `1/1`；本 cell 的 paired delta 已记录，仍未改变 promotion 阈值。

9-cell resource validity/aggregate 合同已冻结：[M4V2_R6_RESOURCE_AGGREGATE_PREREGISTRATION_20260909.md](../../reference/M4V2_R6_RESOURCE_AGGREGATE_PREREGISTRATION_20260909.md)。它沿用 CPU、peak `1.25×`、wall-clock `1.5×`、完整 arm 字段和不可删除无效 cell 规则，并冻结 mean/min/max 与一侧 95% Student-t lower bound；不改变 candidate 主阈值，也不把 fixed-large 事后变成晋级阈值。

model23/31 的 native fixed-large ensemble 已生成并通过与 model17 相同的 artifact、
parent/source/resource、fresh-restore 和 source/holdout non-overlap Gate；三 seed
`fixed_large_registry` 已由 preflight 从 artifact 自动重建并纳入 content-addressed
formal input manifest。formal input report 为 `status=passed`、`formal_input_ready=true`，
formal runner 为 `status=input_ready`，9×5 ledger 已携带每个 model 的 fixed-large
artifact/ensemble digest；model17/course0 single-cell 仍通过，paired resource ledger
仍闭合。所有 execution/promotion 状态保持 `course_executed=false`、
`can_start_r6_formal=false`、`can_promote=false`。

**当前唯一下一步**：实现 model17/course0 的单 cell execution contract，把五臂
S→G→K 的 baseline、K capability、causal lesion、resource、side-effect、
checkpoint/rollback 和 failure attribution 写入冻结 ledger。该 cell 现已真实重跑并通过：
五臂 `resource_gate=true`，每个 arm 均有 CPU/resource digest、peak RSS、参数/字节、
checkpoint path/bytes、inference trace、training steps；candidate/fixed-large K
success 为 `1/1`，lesion 与 rollback/side-effect Gate 通过。**当前唯一下一步**：将
这份五臂执行合同抽为 formal runner 可复用的 cell executor，以同一 model17/course0
重放并做字段级一致性检查；通过前不扩大到其余 8 cells、不运行 9-cell aggregate、
不接 default runtime/provider/MCP/client/CUDA。

executor 现已实现为 `run_cell(model_seed, course_seed, …)`，默认 17/0 入口保持兼容；
同一 manifest 连续两次 replay 的 `execution_contract_digest` 均为
`8d670a310c08a81e024b3c8f2256ba5afa4808491871e67637205d6156c5d22c`，五臂 resource
Gate 全部通过。formal runner 现已消费并写回 `model17/course0` 与
`model17/course1` 两个 row，均为 `executed_passed`；prior ledger digest 校验保证
course0 不会被覆盖；`model17/course2` 也已成功追加，model17 三个 row 均为
`executed_passed`，其余 6 rows 仍为 `not_started`，逐臂 resource/side-effect/checkpoint/
rollback/failure ledger 均已落盘。runner 现固定前序顺序
`17/0 → 17/1 → 17/2 → 23/0 → 23/1 → 23/2 → 31/0 → 31/1 → 31/2`，禁止跳过未通过
row。`model23/course0`、`model23/course1` 与 `model23/course2` 均已执行并通过，
candidate/fixed-large K 均为 `1/1`、lesion 均为 `0`，五臂 resource Gate 全部通过；当前
累计 6 个 row 为 `executed_passed`，其余 3 个仍为 `not_started`，五臂
resource/side-effect/checkpoint/rollback ledger 完整。`model31/course0` 已执行并通过，
累计 7 个 row 为 `executed_passed`，model31/course1/2 仍为 `not_started`。`model31/course1`
已执行并通过，累计 8 个 row 为 `executed_passed`，仅 course2 为 `not_started`。`model31/course2`
已执行并通过，9/9 cell 均为 `executed_passed`，五臂 resource Gate、candidate/fixed-large
K success 和 lesion ledger 全部闭合。**当前唯一下一步**：只运行冻结的 9-cell aggregate
calculation，完成后再按结果审计 promotion，不引入任何未预注册变量、
不接 default runtime/provider/MCP/client/CUDA。

aggregate 已完成并生成
reports/taiji_m4v2_r6_formal_aggregate_20260909.json。结果为
blocked_aggregate，不是执行账本失败：9/9 cell execution 通过，candidate 为
1.0、lesion 为 0.0，K3 causal 与 peak resource 通过；但 candidate/matched
wall multiplier 为 5.9264×–6.8290×（mean 6.5769×，上限 1.5×），且当前
matched/frozen 都是 detached K，无法计算两个预注册 capability delta。**当前唯一下一步**：
先新增并冻结 matched-control revision，再实现同一 K bundle/restore/输入/trace/参数预算、
no-update/no-feedback 的 matched K；只重跑受影响的 paired resource/capability slice，
不放宽阈值、不把 fixed-large 改成 promotion threshold、不接 default runtime/provider/
MCP/client/CUDA。

revised formal runner 已建立独立 monotonic ledger：
reports/taiji_m4v2_r6_matched_control_execution_20260910.json。model17/course0 和
course1 已连续通过，累计 2/9；两个 cell 都是 frozen=0、matched=0、candidate=1、
lesion=0，两个 paired delta=1.0，旧 R6 ledger 未覆盖。**当前唯一下一步**：只执行
revised runner 的 model17/course2，继续要求 predecessor、manifest、resource 和
no-feedback matched contract 全部通过，不运行 aggregate、不改变阈值。

matched-control revision 已生成新 manifest
plans/manifests/taiji_m4v2_r6_matched_control_v2_20260910.json（digest
9221b0920413e7d470821dc66b7daa1a2769ba2502c6b6af1801eb5ad1555ae7），并完成
model17/course0 revised cell：frozen-parent=0.0、matched no-feedback=0.0、
candidate=1.0；matched/candidate 参数和 trace 相同，wall=1.2724×、peak=1.0024×，
旧 runtime/provider/MCP/client/CUDA 仍未接入。**当前唯一下一步**：只执行 revised
model17/course1；成功后按原固定顺序继续，任一失败停止并保留证据，不修改旧 aggregate
或资源阈值。

revised model17 slice 已闭合：course0/course1/course2 连续通过，独立 ledger 累计
3/9，三格均为 frozen=0、matched=0、candidate=1、lesion=0，两个 paired delta=1.0，
wall/peak 和 same-capacity 参数/trace Gate 均通过。**当前唯一下一步**：只执行 revised
runner 的 model23/course0；保持 prior-ledger 顺序，9/9 前不运行 aggregate、不接默认
runtime/provider/MCP/client/CUDA。

model23/course0 已由 revised runner 追加并通过，独立 ledger 累计 4/9；该格仍为
frozen=0、matched=0、candidate=1、lesion=0，candidate-frozen 与
candidate-matched delta 均为 1.0，matched/candidate 参数与 inference trace 相同，
fixed-large K-task-equivalent control 通过。**当前唯一下一步**：只执行 revised runner
的 model23/course1；不运行 aggregate、不改变阈值、不接入 default runtime/provider/
MCP/client/CUDA。

model23/course1 已由 revised runner 追加并通过，独立 ledger 累计 5/9；该格仍为
frozen=0、matched=0、candidate=1、lesion=0，两个 capability delta 均为 1.0，
matched/candidate 参数与 inference trace 相同，wall=1.1905×、peak=1.0040×，
fixed-large K-task-equivalent control 通过。**当前唯一下一步**：只执行 revised runner
的 model23/course2；不运行 aggregate、不改变阈值、不接入 default runtime/provider/
MCP/client/CUDA。

model23/course2 已由 revised runner 追加并通过，独立 ledger 累计 6/9；该格仍为
frozen=0、matched=0、candidate=1、lesion=0，两个 capability delta 均为 1.0，
matched/candidate 参数与 inference trace 相同，wall=1.2714×、peak=1.0025×，
fixed-large K-task-equivalent control 通过。model23 slice 已闭合。**当前唯一下一步**：
只执行 revised runner 的 model31/course0；不运行 aggregate、不改变阈值、不接入
default runtime/provider/MCP/client/CUDA。

model31/course0 已由 revised runner 追加并通过，独立 ledger 累计 7/9；该格仍为
frozen=0、matched=0、candidate=1、lesion=0，两个 capability delta 均为 1.0，
matched/candidate 参数与 inference trace 相同，wall=1.1799×、peak=1.0024×，
fixed-large K-task-equivalent control 通过。**当前唯一下一步**：只执行 revised runner
的 model31/course1；不运行 aggregate、不改变阈值、不接入 default runtime/provider/
MCP/client/CUDA。

model31/course1 已由 revised runner 追加并通过，独立 ledger 累计 8/9；该格仍为
frozen=0、matched=0、candidate=1、lesion=0，两个 capability delta 均为 1.0，
matched/candidate 参数与 inference trace 相同，wall=1.1603×、peak=1.0030×，
fixed-large K-task-equivalent control 通过。**当前唯一下一步**：只执行 revised runner
的 model31/course2；不运行 aggregate、不改变阈值、不接入 default runtime/provider/
MCP/client/CUDA。

model31/course2 已由 revised runner 追加并通过，revised ledger 已完成 9/9，全部 row
均为 executed_passed、failures 为空；该格 frozen=0、matched=0、candidate=1、lesion=0，
两个 capability delta 均为 1.0，matched/candidate 参数与 inference trace 相同，
wall=1.2513×、peak=1.0027×，fixed-large K-task-equivalent control 通过。**当前唯一
下一步**：实现并运行 matched-control revision 专用 aggregate validator；先验证
9-cell causal/resource/retention 合同，再决定是否允许 formal/promotion，期间不接入
default runtime/provider/MCP/client/CUDA、不修改阈值。

matched-control revision aggregate validator 已实现并通过，报告为
reports/taiji_m4v2_r6_matched_control_aggregate_20260910.json；9/9 cell、candidate
floor、K3 lesion、wall/peak budget、frozen-parent paired delta 和 matched-capacity
paired delta 全部为 true，两组 paired delta 的均值及单侧 95% lower bound 均为 1.0，
blocking_failures 为空。**当前唯一下一步**：实现独立的 R6 formal-admission gate，
只消费该 aggregate 与 revision manifest，验证放行前的完整 lineage/side-effect/
checkpoint 约束；在该 gate 通过前不启动 formal/promotion，不接 default runtime/
provider/MCP/client/CUDA。

独立 formal-admission gate 已实现并通过，报告为
reports/taiji_m4v2_r6_matched_control_admission_20260910.json；它只消费 revision
manifest 与 matched-control aggregate，重新核验 registry/owner graph、checkpoint
路径、9-cell aggregate lineage、资源 owner、side-effect 和阈值不变性。结果为
`ready_to_start_formal`、`can_start_r6_formal=true`、`can_promote=false`，且没有
训练、runtime/provider/MCP/client/CUDA 副作用。**当前唯一下一步**：把 admission
报告接入 formal runner 的输入合同，要求后续 formal 执行必须携带同一 manifest digest
和通过的 admission report；先做 admission-gated preflight，不直接 promotion。

formal runner 已接入 admission report 硬性输入；独立 gated preflight
reports/taiji_m4v2_r6_matched_control_formal_admission_preflight_20260910.json
通过，manifest digest 一致、admission status 通过、formal_input_ready=true、
can_start_r6_formal=true，且 course/training/promotion/runtime/provider/MCP/client/CUDA
均未发生。旧 execution/aggregate evidence 未被改写。**当前唯一下一步**：在该
gated preflight 基础上启动同一 manifest 的正式执行入口；保持 promotion=false，先
记录 formal start/执行结果，不接入默认 runtime/provider/MCP/client/CUDA。
