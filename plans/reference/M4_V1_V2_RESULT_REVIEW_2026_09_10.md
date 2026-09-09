# M4 v1 / v2 实际结果复审（2026-09-10）

审视基线：`672adc9a`。本次读取实现、原始 JSON、九个 cell 和测试源码，不仅依据 plans；未重新训练、未重跑历史实验，未验证远端 CI。执行顺序只由 [当前计划](../active/roadmap/03_CURRENT_EXECUTION.md) 决定。旧报告保持原样，不把解释修订伪装成实验重跑。

## 1. 回答：有进步，但尚未证明整体能力超过第一版 M4

“第一版”同时覆盖最初 R0 和其后收敛的 R0～R12，不能只挑最弱 R0 比较。v2 增加了实际 fast/slow、replay、主路径 residual 和结构训练；有窄任务收益。最新 R6 则主要增加了 Workbench/反馈接线与运行证据，不是新一轮持续学习成功。当前不能宣称通用智能、自然生长或训练效率优势。

| 证据 | 实际结果 | 能支持什么 / 不能支持什么 |
|---|---|---|
| v1 R0 | C″ gain −0.01078 BPB，cycle2/3 退化 3/3 | 最初固定容量 cascade 未过能力门槛 |
| v1 R7 | C3 gain +0.152757 BPB，3/3 正；旧 C cycle3 退化 2/3 | 已有真实新知识收益，但未满足当时 exact-zero 保持规则；并非完全不会学习 |
| v2 R2 | `r2_candidate_accepted=true`；三课程顺序的 replay/状态变化/恢复 Gate 通过 | 快慢状态及回放有小规模 S/G 证据；课程训练量约 419 bytes，不等于 v1 的 64 KiB/phase 任务 |
| v2 R4 formal | 相对 fixed-capacity，S/G mean-surprise 差 −0.086353/−0.019313，均 9/9 更好 | 结构候选有窄任务收益与因果贡献，不再只是空结构 |
| v2 R4 formal | 相对 fixed-large，G 差 +0.001002，仅 4/9 更好；`can_promote=false` | 尚不能证明动态成长优于一开始给足容量 |
| v2 R5 formal | `rejected`；G 非劣 4/7、S 非劣 6/7；G 相对 fixed-large 均值 +0.001036 | 该条件路由候选未解决强对照差距；不能由此否定一切结构成长 |
| 最新 v2 R6 | 九 cell 均 `training_performed=false`、`course_executed=false`、训练步数 0 | 接线运行验证，不是九次连续学习复现 |

数值方向：BPB/mean-surprise 越低越好；v1 的 gain 是改善量，v2 表中的差值是 candidate−control。二者任务、预算、单位和父代不同，禁止把两列直接相减作为性能提升百分比。

原始来源：

- [v1 R0～R12 复盘](M4_FIXED_CAPACITY_EVIDENCE_REVIEW_2026_09_09.md)、[R7 aggregate](../../reports/taiji_m4r7_formal_aggregate_20260908.json)。
- [R2 S/G formal](../../reports/taiji_m4v2_r2_formal_sg_20260909.json)、[R2 实现](../../scripts/training/eval_taiji_m4v2_r2_canary.py)。
- [R4 formal](../../reports/taiji_m4v2_r4_shadow_formal_20260909.json)、[R5 formal](../../reports/taiji_m4v2_r5_conditional_formal_20260909.json)。
- [R6 aggregate](../../reports/taiji_m4v2_r6_matched_control_aggregate_20260910.json)、[首个 cell](../../reports/taiji_m4v2_r6_matched_control_cell_model_17_course_0_20260910.json)；其余八个同前缀 cell 已逐一读取核对。

## 2. 最新 R6 的关键混淆：反馈准入被当成任务成功

根因可直接在 [single-cell runner](../../scripts/training/eval_taiji_m4v2_r6_formal_single_cell.py) 的 `_candidate_arm` 查到：

- `admit_feedback=False` 且非 lesion 时，运行通过就赋 `capability_rate=0.0`。
- candidate 则根据 canary 通过、真实执行成功、projection 接纳赋 `1.0`。
- 每臂 `sample_count=1`。九个 cell 的 matched `real_workbench_success=true`，却全部记作任务成功率 0；candidate 全部为 1。
- 九个 cell 的 candidate K observation 都是 `typescript_05.ts`。文件名相同本身不能证明内容完全相同，但加上没有课程训练，不能将这些运行当成九组独立学习泛化证据。
- runner 直接记录 `training_performed=False`、`candidate_training_performed=False`；S/G 阶段是评分观测，不是完整 S→G→K 参数更新课程。

因此 aggregate 的差值 +1、零方差及其置信下界，只描述当前评分规则，不能证明“反馈训练带来 100% 能力提升”。反馈接纳是机制状态；任务成功必须由执行结果或独立评估器判定。关闭学习但正常完成任务的对照，必须保留其成功成绩。无权执行/未接入 worker 应单列覆盖率或准入拒绝，不能冒充同能力基线失败。

无更新时旧 S/G 不变，只证明只读/隔离保持，不证明学习之后没有遗忘。项目以前确实训练过模型和 worker；本结论仅限定这一轮 R6。

## 3. 资源和测试结论降回实际范围

- 新 matched aggregate 的 candidate/matched wall-clock 均值约 1.2202，表示候选慢约 22%，不是加速 22%。旧 detached 对照换成同 worker 对照后比值改善，首先是比较口径变化。
- RSS 比值约 1.0026，但方法明确是 `process_rss_before_after_lower_bound`，不是运行期间真实峰值。训练步数为 0，不能说明训练内存优势。
- K worker 4833 参数、19332 参数 bytes，fixed-large 为 9666/38664；这些不是整个 Taiji 的参数量、checkpoint 总量或运行内存。相同单题成功下，fixed-large 平均还比 candidate 快约 0.302 秒，不能据参数更少宣称整体资源胜出。
- [admission 测试](../../tests/taiji_native/test_m4v2_r6_matched_control_admission.py) 读取已保存 JSON 断言状态，未直接执行 admission runner 的拒绝分支。它是报告回归检查，不是抗篡改、失败闭锁或训练正确性的充分验证。
- 已存 `can_start_r6_formal=true` 仅是旧合同下的准入结果；本次撤销其作为“可直接开始学习 formal”的计划依据，不改写历史 JSON。

## 4. 设计修订依据

保留 v2 快慢状态、真实 replay、函数保持出生、owner 隔离、checkpoint/rollback。否决“多跑无训练九宫格即可证明成长”的解释，不删除负结果或推倒可复用底座。

当前最重要的研究问题是：**同一父代、同等权限与预算下，实际吸收新的训练经历，是否使未见任务得分提高，并在后续课程保持旧能力？** 先固定容量解开学习与接线的混淆；这不是放弃高上限结构成长，而是避免用扩容掩盖学习链未建立。

选择说明：直接再跑旧 formal 只能重复接线事实；马上扩大网络无法分离容量和训练作用；先完成真实固定容量学习对照，能复用最多已有资产，并为后续增长提供可证伪基线。采用第三条。

## 5. 解释边界和待验证事项

尚无同父代、同数据、同预算、同 scorer 的 v1/v2 正面对照；尚无此次 R6 的实际持续学习收益；不能据此计算总体提升比例。R2/R4 的小任务收益应保留，不外推到自然语言能力。R5 报告中的“structural-growth mainline closes”只按该候选与该课程的否决理解。

本次只修订计划与证据解释。评分实现、admission 故障路径、完整训练状态恢复以及独立课程消费审计均列为后续任务，不宣称已经修好。
