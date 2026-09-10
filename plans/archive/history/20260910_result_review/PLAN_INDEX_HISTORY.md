# 历史计划入口（截至 4629bc8c）

> 仅作历史证据，当前入口见 [plans/README](../../../README.md)。

# Seed / Taiji 计划与架构入口

> 最新修订：2026-09-10。当前唯一执行计划是 [Seed / Taiji 唯一执行计划](../../../active/roadmap/03_CURRENT_EXECUTION.md) 顶部的「2026-09-10 修订执行序列」。旧正文及归档中的“下一步”均失效。

M4.R0～R12 已完成并归档，但 2026-09-09 的代码/报告复审发现：该系列主要测量 F1 byte-prediction 的固定容量 continuation，R2 的固定等权读出槽、R7/R10 不一致的 owner 图、未进入普通主路径的 adaptive network，以及 exact-zero Gate 都不足以代表 CR-4/A8 的继承式结构成长。历史报告保留，过度外推已撤销。

最新实际状态：v2 R2 的 fast/slow 与真实 replay 有小规模 S/G 证据；R4 formal 在 S/G 上优于小容量对照，但未稳定超过 fixed-large；R5 条件路由候选被否决。C-entry formal 已完成：candidate 相对 frozen parent 的 9 格质量与 CPU 资源 Gate 通过，但 candidate 在 `0/9` 格胜过 fixed-large，因而不晋级。随后完成的 capacity-parity audit 确认 formal 对照存在 19,332 对 38,664 参数字节、6 对 14,252 更新步的容量/预算混淆，因此暂不能作学习规则因果结论。完整依据见 [M4 v1/v2 实际结果复审](../../../reference/M4_V1_V2_RESULT_REVIEW_2026_09_10.md)、[B3-K C-entry formal closure](../../../reference/M4V2_B3_K_C_ENTRY_CLOSURE_20260910.md) 和 [B3-K capacity-parity audit](../../../reference/M4V2_B3_K_C_CAPACITY_PARITY_AUDIT_20260910.md)。

K continuation-learning contract 已完成，但完整 B3-K 尚未完成：`taiji/k_continuation.py` 现在把真实 K episode、train/holdout 隔离、K1/K2 可更新边界、K3/parent 不可变边界以及 checkpoint/rollback receipt 固化为 content-addressed 合同；当前 detached local-delta worker 明确不带 optimizer state。B3-K 单步 pilot 已在同一 inherited model 17 parent 上真实更新 K1/K2 各 1 步，K3、parent、fresh-restore 和 rollback 均通过；但 holdout 结构化准确率已在更新前饱和为 1.0，更新后没有可测增益，不能晋级。

B3-K 的非饱和 structured-loss diagnostic 已通过：同一 model 17 parent、1 条 train、3 条真正 record-disjoint holdout 上，K1/K2 六个连续 MSE 分量全部下降，combined MSE `0.01677758 → 0.01569742`；K3、parent、fresh-restore、rollback 全部通过，但仍不是晋级证据。随后完成的多 course-seed 复验真正切换了 train episode（3 个 train digest 均不同）并固定 holdout：seed 0 delta `−0.00108017`，seed 1 `+0.00005455`，seed 2 `+0.00033079`。技术链通过，但最坏退化为正，`performance_gate_passed=false`、`stability_gate_passed=false`，所以不能把平均改善当作稳定学习，也不能进入 formal。补充的课程敏感性量尺显示三条 train combined-MSE 都下降（`−0.00050056/−0.00021416/−0.00022227`），参数 delta norm 非零且 no-update scorer 对照为零；问题收敛到单条样本更新的过拟合/跨 episode 干扰候选，而非保存或评分器漂移。随后仅把 train 扩为 2 条的 bounded batch 结果为 `−0.00053848/+0.00019021/−0.00040499`：2/3 改善且最坏退化收窄，但性能 Gate 仍失败。三样本连续组合的 raw delta 都是 `−0.00025813`，但 candidate worker/参数 digest 三组完全相同；非滑动组合进一步显示前两组仍相同、第三组才不同，已标记为 `candidate_updates_distinct=false` 的非判别证据。逐 episode 审计进一步确认：6 条输入 digest 全部不同，但单步 K1/K2 candidate 都出现 `0=3、1=4、2=5` 成对碰撞，batch candidate 又不等于任何单步 candidate。feature-target audit 最终发现 K1/K2 input tensor 虽不同，target tensor 却按同样的 `0=3、1=4、2=5` 重复，说明先前课程的独立性被路径差异伪装了。target-aware Gate v1 随后被撤销：它使用包含 episode/example ID 的 `experience.target_digest`，因此把 ID 差异误当作 target 差异；v1 报告仅保留历史审计用途。修正版 target-aware v2 改用真实 K1/K2 target tensor digest 组合，确认 `A+B+C / A+A+B / A+B+B` 的三种 target multiset 真正不同，三组 candidate digest 分离且 loss delta 为 `−0.00025813/−0.00072424/−0.00034705`；model seed 17/23/31 的 9 个 cell 也全部通过，整体均值 `−0.00044314`、最坏 `−0.00025813`。这仍是 validation/窄任务证据，不能替代 sealed-test、fixed-large 强对照或 promotion。

**C 入口 manifest、sealed-test 与 fixed-large 控制已冻结/材料化。** 新控制按 3 个 model seed × 3 个 course seed 生成，绑定同一 parent、target-aware target multiset、validation/sealed artifact 和 C-entry source manifest；prefit/fresh-restore、disk restore、顺序 replica distinct 和输入预检均通过，报告为 [C-entry input preflight](../../../../reports/taiji_m4v2_b3_k_c_entry_input_preflight_20260910.json)。

**C-entry sealed scoring canary 已完成，但不是 formal。** 报告为 [C-entry sealed scoring canary](../../../../reports/taiji_m4v2_b3_k_c_sealed_scoring_20260910.json)：candidate 相对 frozen parent 的 sealed combined-MSE 平均 delta 为 `−0.0004431358`、最坏 `−0.0002581254`；fixed-large 平均 delta 为 `−0.0011265067`，candidate 在 `0/9` 格胜过 fixed-large。技术评分通过，但 canary 没有训练/推理资源账本，所以 `formal_gate_passed=false`、`can_start_formal=false`、`can_promote=false`。

**C-entry formal execution 已完成并收束。** 报告为 [C-entry formal report](../../../../reports/taiji_m4v2_b3_k_c_formal_20260910.json)：9/9 technical、resource、candidate-quality Gate 通过；candidate sealed mean/worst delta 为 `−0.0004431358/−0.0002581254`，fixed-large mean 为 `−0.0011265067`，candidate 胜出 `0/9`。因此 candidate 可学习但未胜强对照，`can_promote=false`，不接入默认路径。

**C-entry capacity-parity audit 已完成。** 报告为 [capacity-parity audit](../../../../reports/taiji_m4v2_b3_k_c_capacity_parity_audit_20260910.json)：9 格的训练 episode 与推理 trace 一致，但 candidate/fixed-large 的参数字节为 `19,332/38,664`、实际更新步数为 `6/14,252`、逻辑 checkpoint 写入数为 `2/9`。因此 candidate 的学习证据有效，当前 strong-control 比较却被标记为 `blocked-current-comparison-confounded`，不改变 `can_promote=false`。

**上调 candidate 容量的 parity input preflight 已完成。** 合同为 [capacity-parity manifest](../../../manifests/taiji_m4v2_b3_k_c_capacity_parity_v1.json)，输入报告为 [capacity-parity input preflight](../../../../reports/taiji_m4v2_b3_k_c_capacity_parity_input_preflight_20260910.json)：9/9 fixed-large cell 均满足 38,664 参数字节、14,252 实际更新步、9 个逻辑 checkpoint、同 parent/course 和 fresh-restore；candidate parity artifact 尚未生成，且不能直接复制 fixed-large replica。

**widened-candidate 路线设计已冻结**：[widened-candidate design](../../../reference/M4V2_B3_K_C_WIDENED_CANDIDATE_DESIGN_20260910.md)——单实例双通道分解（supervised + k3-feedback），每个可学习头精确 ×2 = 9,666 参数（38,664 字节，ratio 0.0%），update 7,126 ticks × 2 通道 = 14,252 步精确对齐 fixed-large，readout 为异构通道和而非 replica 平均；K continuation contract 升 v2，含硬性 distinct 证据门（通道差分范数/反馈覆盖率）。设计未实现前不训练、不读 sealed formal。

**parity 口径修正与修复路线已冻结**：[caliber revision](../../../reference/M4V2_B3_K_C_PARITY_CALIBER_REVISION_20260910.md)——机器证据钉死两臂真实新增均为 12 步/cell（14,252 中 14,240 是 parent 继承步数）；新口径只用 `new_update_steps` 比较（inherited/new 分离，旧 JSON 保留）；修复路线 = 更强新增预算课程（每 cell 150 个未见过的新 experiences，双臂同流新增 600 步且自动相等），失败 episode 不引入。

**v2 修复按修正后根因执行，9/9 cell 通过**：类平衡课程（A/B/C 各 50）+ 类模式守卫（anchored 排列确定性重生成直到模式改变）+ 双臂同流 600 新增步（new_update_steps 口径）+ 每实例 9 逻辑 checkpoint。**预注册准备期 digest 固定又发现并修正一个独立性问题**（类序列不随课程变化 → 权重逐位相同）：类块内排列随课程变化（ABC/ACB/BAC）+ 课程独立性硬门（权重 state_dict digest 两两互异），重跑后 ensemble digest 恰 3 份（每课程一份）——独立样本结构机器成立。通道差分 9/9 严格正（K1 0.035~0.264），原失败 cell 31x1 从精确 0.0 → 0.1457，机制修复直接确认。v1 证据不覆盖，sealed 仍锁定。

**C-entry parity formal 预注册已冻结**：[parity formal preregistration](../../../reference/M4V2_B3_K_C_PARITY_FORMAL_PREREGISTRATION_20260910.md)——评分输入 digest 固定、6 分量 MSE 同口径、candidate readout = 权重相加（与 logit 和数学等价，诚实声明）、判据 G1 质量门 / G2 灾难界（validation 派生先行冻结）/ G3 主判据（sealed 上 ≥2/3 课程胜出且均值更优）、统计单元 course n=3、sealed 读取后禁止调参。

**parity formal v3 已执行（readout 修订 + 全新 sealed v2），widened 路线按预注册关闭**：G1/G2 通过（权重平均 readout 消除了 v1 的 +0.107 灾难，sealed 三课程全部为负），G3 极小差距失败（1/3 课程胜出，均值 −0.00173 vs −0.00188）。**结论：等预算等容量下，顺序分化 + 权重平均不优于同构 replica 概率平均——两种合成在同一表现带（±0.0002）。** fixed-large 概率平均保留为 C-entry strong arm。学习问题收敛点：K 课程可见信号空间只有 3 类，任何课程/合成变化都是二阶差异。

**K 信号空间扩展预注册已冻结**：[signal-space expansion](../../../reference/M5_K_SIGNAL_SPACE_EXPANSION_PREREGISTRATION_20260910.md)——fact 词汇 13→14（+`language_state::ambiguous`）、K1 goal 3→4、K2 goal 2→4，可见类空间 **3→5 类**（+D=.h clarify-language、+R=recover）；父代 K workers 按 M5.K2 课程+D/R episodes 重建至 `taiji_k_workers_v4/`（K3/掩码机制/学习率不动）；重跑合同 = 150 experiences 5 类平衡、双臂同流 600 新增步、fresh sealed v3、G1/G2/G3 同构。可证伪点：扩展后合成差异是否超出二阶带。

**K 信号空间扩展 §8 全链已执行，G3 失败——合成方式不可分确证**：父代重建（14 facts/4 goals，参数 K1 472/K2 5176）→ 双臂 build 9/9（通道差分 K1 0.21~0.41，比 3 类空间再升一个量级）→ sealed v3 → formal v4（教师强制镜像评分修复评分不对称后）：G1/G2 通过，G3 失败（1/3 胜出，均值差 4e-5 ≪ ±0.0002 带）。**两个结论并存**：信号空间扩展本身成功（sealed 改善 −0.002 → −0.129，学习效应高度可测）；但合成方式在该 harness 内不可分——fixed-large 概率平均保留为 C-entry strong arm。C-entry 证据线收束（容量/信号空间/合成方式三维度均检验）。

**B3 pilot（K 相位）预注册已冻结**：[B3 K pilot preregistration](../../../reference/M4V2_B3_K_PILOT_PREREGISTRATION_20260910.md)——三臂从同一 v4 父代派生：F frozen / C continuation（300 新增步）/ FS fast+slow+replay（wake 写 fast_delta、sleep 对 b=50 真实经历 replay 更新 slow 后 consolidate 清零，replay 100 步单列成本）；机制门 6 项为主交付；validation-only（sealed v3 留给 C 阶段）。

**B3 K-phase pilot 已执行，机制门 12/12 全过**：`taiji/k_fast_slow.py`（slow+fast 拆分、wake 写 fast、真实经历 replay 写 slow、consolidate 清零且保持有效权重）+ 定向测试 5/5 + 三臂 pilot。诊断读数：整体 validation FS（−0.000335）略逊 C（−0.000455），但 **D/R 弱类探针 FS 优于 C**（−0.1928 vs −0.1914）——replay 收益是类选择性的。修订注记：wake 轨迹位等价断言在浮点下不可达，门改容差（实测 2.38e-7）。

**C 阶段正式比较预注册已冻结**：[C-stage formal preregistration](../../../reference/M4V2_C_STAGE_FORMAL_PREREGISTRATION_20260910.md)——四臂零重训（F / C 300 步 / **FS candidate** 300 wake+100 replay 单列 / XL fixed-large 2× 容量次要-bar），统计单元 course n=3；判据 G1 学习门 / G2 灾难界 / **G3 弱类主判据**（D/R sealed：FS < C 于 ≥2/3 课程）/ **G4 整体非劣**（epsilon 从 validation paired diff 派生）；sealed v4 全新 materialize（覆盖 D/R/A）。结果映射：全过 → FS 为 K 相位默认机制候选；G3 败 → continuation 收束为默认；G4 败 → replay 回炉。

**C 阶段 formal 已执行，G3/G4 通过、G1 失败——冻结映射出现缺口**：sealed v4（task_seed=101，D/R/A 覆盖）四臂两阶段执行。**G3 弱类主判据 3/3 全胜**（FS −0.1931~−0.1946 vs C −0.1913，每课程好 ~0.0022）、**G4 通过且 FS 实际更优**（整体 −0.1307 vs −0.1288，还优于 XL −0.1287）、G2 通过；但 G1 失败（FS validation delta 3/3 课程 ≥ 0——validation 小样本高方差与 sealed 强负方向相反）。判定按冻结门如实为 failed，映射缺口（未定义「G1 败 + G3/G4 过」）记录 §7.2。

**当前唯一下一步（需拍板）**：(a) 新预注册修正 G1 操作化 + fresh sealed v5 重跑（主假设已 3/3 确证）；(b) 接受 failed 判定收束。拍板前不读新 sealed。

## 权威文档

| 文档 | 唯一职责 |
|---|---|
| [TAIJI_CORE_REQUIREMENTS.md](../../../active/TAIJI_CORE_REQUIREMENTS.md) | 长期使命与 CR-1～CR-10，不随实验归档 |
| [TAIJI_NATIVE_ARCHITECTURE_V1.md](../../../active/TAIJI_NATIVE_ARCHITECTURE_V1.md) | 目标架构、对象、学习体系与设计约束 |
| [TAIJI_CONTINUAL_DEVELOPMENT_V2.md](../../../active/architecture/TAIJI_CONTINUAL_DEVELOPMENT_V2.md) | M4 v2 的快/慢学习、主路径结构成长、课程与准入 Gate |
| [ARCHITECTURE_DIRECTION_2026_08.md](../../../active/ARCHITECTURE_DIRECTION_2026_08.md) | 原生身份、成熟技术与 Legacy 边界 |
| [SEED_ARCHITECTURE.md](../../../active/SEED_ARCHITECTURE.md) | Seed、Workbench、provider 与副作用所有权 |
| [03_CURRENT_EXECUTION.md](../../../active/roadmap/03_CURRENT_EXECUTION.md) | **唯一执行顺序、详细任务、验收、日程和下一步** |
| [TAIJI_RESEARCH_REVIEW_2026_09_06.md](../../../reference/TAIJI_RESEARCH_REVIEW_2026_09_06.md) | 本次审视、最小复现、证据失效范围与技术参考，不另设执行路线 |
| [M4_FIXED_CAPACITY_EVIDENCE_REVIEW_2026_09_09.md](../../../reference/M4_FIXED_CAPACITY_EVIDENCE_REVIEW_2026_09_09.md) | R0～R12 原始数值、有效技术资产与复审后的解释边界 |
| [M4_V1_V2_RESULT_REVIEW_2026_09_10.md](../../../reference/M4_V1_V2_RESULT_REVIEW_2026_09_10.md) | v1/v2 实际对比、R6 评分混淆与本次修订依据，不另设执行路线 |
| [M4V2_B3_K_C_ENTRY_CLOSURE_20260910.md](../../../reference/M4V2_B3_K_C_ENTRY_CLOSURE_20260910.md) | C-entry formal 的真实结果、promotion 边界与下一阶段前置审计 |
| [M4V2_B3_K_C_CAPACITY_PARITY_AUDIT_20260910.md](../../../reference/M4V2_B3_K_C_CAPACITY_PARITY_AUDIT_20260910.md) | C-entry formal 的容量/训练预算混淆审计与 parity contract |
| [IMPLEMENTATION_STATUS_2026_08.md](../../../reference/IMPLEMENTATION_STATUS_2026_08.md) | 当前实现事实与能力声明边界 |

## 目录与维护

- `active/` 保留根需求、当前架构和唯一计划。兼容入口 `roadmap/01、02、04` 不决定顺序。
- `reference/` 保存当前事实、证据解释和 owner 边界。
- `archive/` 保存已完成/被替代的计划、实验讨论和调试过程；仍有效的设计结论先提炼到 active。
- `manifests/` 保存版本化数据、评估与实验合同；旧报告不因新判定而被原地改绿。
- 每轮更新唯一计划并提交精确范围；不以新增编号文档、checkpoint 文件大小或测试数量代替能力收益。
- 历史入口：[本次研究重审归档](../research_review_20260906/README.md)、[2026-09-01 收敛归档](../roadmap_convergence_20260901/README.md)、[总归档索引](../../README.md)。
