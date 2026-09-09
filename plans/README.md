# Seed / Taiji 计划与架构入口

> 最新修订：2026-09-10。当前唯一执行计划是 [Seed / Taiji 唯一执行计划](active/roadmap/03_CURRENT_EXECUTION.md) 顶部的「2026-09-10 修订执行序列」。旧正文及归档中的“下一步”均失效。

M4.R0～R12 已完成并归档，但 2026-09-09 的代码/报告复审发现：该系列主要测量 F1 byte-prediction 的固定容量 continuation，R2 的固定等权读出槽、R7/R10 不一致的 owner 图、未进入普通主路径的 adaptive network，以及 exact-zero Gate 都不足以代表 CR-4/A8 的继承式结构成长。历史报告保留，过度外推已撤销。

最新实际状态：v2 R2 的 fast/slow 与真实 replay 有小规模 S/G 证据；R4 formal 在 S/G 上优于小容量对照，但未稳定超过 fixed-large；R5 条件路由候选被否决。R6 九 cell 接线运行通过，但全部无训练，且旧评分把“不接纳反馈”计为任务成功率 0，不能当作学习提升。评分语义修正已完成：任务成功、反馈准入和参数更新现在分离，wiring-canary 不能进入 learning-formal Gate；旧报告仍保留为历史证据。整体保持 `can_promote=false`，暂停直接沿旧 admission 启动学习 formal。完整依据见 [M4 v1/v2 实际结果复审](reference/M4_V1_V2_RESULT_REVIEW_2026_09_10.md)。

K continuation-learning contract 已完成，但完整 B3-K 尚未完成：`taiji/k_continuation.py` 现在把真实 K episode、train/holdout 隔离、K1/K2 可更新边界、K3/parent 不可变边界以及 checkpoint/rollback receipt 固化为 content-addressed 合同；当前 detached local-delta worker 明确不带 optimizer state。B3-K 单步 pilot 已在同一 inherited model 17 parent 上真实更新 K1/K2 各 1 步，K3、parent、fresh-restore 和 rollback 均通过；但 holdout 结构化准确率已在更新前饱和为 1.0，更新后没有可测增益，不能晋级。

B3-K 的非饱和 structured-loss diagnostic 已通过：同一 model 17 parent、1 条 train、3 条真正 record-disjoint holdout 上，K1/K2 六个连续 MSE 分量全部下降，combined MSE `0.01677758 → 0.01569742`；K3、parent、fresh-restore、rollback 全部通过，但仍不是晋级证据。随后完成的多 course-seed 复验真正切换了 train episode（3 个 train digest 均不同）并固定 holdout：seed 0 delta `−0.00108017`，seed 1 `+0.00005455`，seed 2 `+0.00033079`。技术链通过，但最坏退化为正，`performance_gate_passed=false`、`stability_gate_passed=false`，所以不能把平均改善当作稳定学习，也不能进入 formal。补充的课程敏感性量尺显示三条 train combined-MSE 都下降（`−0.00050056/−0.00021416/−0.00022227`），参数 delta norm 非零且 no-update scorer 对照为零；问题收敛到单条样本更新的过拟合/跨 episode 干扰候选，而非保存或评分器漂移。随后仅把 train 扩为 2 条的 bounded batch 结果为 `−0.00053848/+0.00019021/−0.00040499`：2/3 改善且最坏退化收窄，但性能 Gate 仍失败。三样本连续组合的 raw delta 都是 `−0.00025813`，但 candidate worker/参数 digest 三组完全相同；非滑动组合进一步显示前两组仍相同、第三组才不同，已标记为 `candidate_updates_distinct=false` 的非判别证据。逐 episode 审计进一步确认：6 条输入 digest 全部不同，但单步 K1/K2 candidate 都出现 `0=3、1=4、2=5` 成对碰撞，batch candidate 又不等于任何单步 candidate。feature-target audit 最终发现 K1/K2 input tensor 虽不同，target tensor 却按同样的 `0=3、1=4、2=5` 重复，说明先前课程的独立性被路径差异伪装了。target-aware Gate 随后使用 `A+B+C / A+A+B / A+B+B` 三种真实 target 组成，三组 candidate digest 分离且 loss delta 为 `−0.00025813/−0.00072424/−0.00034705`，性能与稳定性 Gate 均通过，但仍只是 model 17 的窄证据。真实 model seed 17/23/31 的 9 个 target-aware cell 也全部通过，整体均值 `−0.00044314`、最坏 `−0.00025813`，但三个 seed 轨迹数值同构，仍不能替代 sealed-test/强对照。

**当前唯一下一步：冻结 C 入口 evaluation manifest。** 把 target-aware train composition、validation holdout、sealed-test holdout、target multiplicity、loss scorer、最小有意义收益、最坏退化上限、fixed-capacity 强对照、model seed 列表和资源预算写成不可变 manifest；先不看 sealed-test 便锁定阈值和比较方法。当前 3 条 holdout 只作 validation，未冻结前不启动 formal、不 promotion、不扩结构。Skill/MCP、Workbench、插件、provider、CUDA 和视觉均已排入计划，不并行抢占模型学习主线。

## 权威文档

| 文档 | 唯一职责 |
|---|---|
| [TAIJI_CORE_REQUIREMENTS.md](active/TAIJI_CORE_REQUIREMENTS.md) | 长期使命与 CR-1～CR-10，不随实验归档 |
| [TAIJI_NATIVE_ARCHITECTURE_V1.md](active/TAIJI_NATIVE_ARCHITECTURE_V1.md) | 目标架构、对象、学习体系与设计约束 |
| [TAIJI_CONTINUAL_DEVELOPMENT_V2.md](active/architecture/TAIJI_CONTINUAL_DEVELOPMENT_V2.md) | M4 v2 的快/慢学习、主路径结构成长、课程与准入 Gate |
| [ARCHITECTURE_DIRECTION_2026_08.md](active/ARCHITECTURE_DIRECTION_2026_08.md) | 原生身份、成熟技术与 Legacy 边界 |
| [SEED_ARCHITECTURE.md](active/SEED_ARCHITECTURE.md) | Seed、Workbench、provider 与副作用所有权 |
| [03_CURRENT_EXECUTION.md](active/roadmap/03_CURRENT_EXECUTION.md) | **唯一执行顺序、详细任务、验收、日程和下一步** |
| [TAIJI_RESEARCH_REVIEW_2026_09_06.md](reference/TAIJI_RESEARCH_REVIEW_2026_09_06.md) | 本次审视、最小复现、证据失效范围与技术参考，不另设执行路线 |
| [M4_FIXED_CAPACITY_EVIDENCE_REVIEW_2026_09_09.md](reference/M4_FIXED_CAPACITY_EVIDENCE_REVIEW_2026_09_09.md) | R0～R12 原始数值、有效技术资产与复审后的解释边界 |
| [M4_V1_V2_RESULT_REVIEW_2026_09_10.md](reference/M4_V1_V2_RESULT_REVIEW_2026_09_10.md) | v1/v2 实际对比、R6 评分混淆与本次修订依据，不另设执行路线 |
| [IMPLEMENTATION_STATUS_2026_08.md](reference/IMPLEMENTATION_STATUS_2026_08.md) | 当前实现事实与能力声明边界 |

## 目录与维护

- `active/` 保留根需求、当前架构和唯一计划。兼容入口 `roadmap/01、02、04` 不决定顺序。
- `reference/` 保存当前事实、证据解释和 owner 边界。
- `archive/` 保存已完成/被替代的计划、实验讨论和调试过程；仍有效的设计结论先提炼到 active。
- `manifests/` 保存版本化数据、评估与实验合同；旧报告不因新判定而被原地改绿。
- 每轮更新唯一计划并提交精确范围；不以新增编号文档、checkpoint 文件大小或测试数量代替能力收益。
- 历史入口：[本次研究重审归档](archive/history/research_review_20260906/README.md)、[2026-09-01 收敛归档](archive/history/roadmap_convergence_20260901/README.md)、[总归档索引](archive/README.md)。
