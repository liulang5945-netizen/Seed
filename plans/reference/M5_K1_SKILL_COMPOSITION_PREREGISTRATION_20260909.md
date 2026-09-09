# M5.K1 预注册：技能组合课程 canary（A8 主证据第一步）

> 起草日期：2026-09-09。K 课程定义见 [架构 v2 §6.1](../active/architecture/TAIJI_CONTINUAL_DEVELOPMENT_V2.md)：结构化语义 → 世界转移 → Workbench 只读意图 → 隔离执行，每阶段未见组合 + 真实 outcome。本文是 K1 canary 的预注册；确认后实现。执行顺序以 [03_CURRENT_EXECUTION.md](../active/roadmap/03_CURRENT_EXECUTION.md) 为准。

## 1. 可证伪假设

> 在同一 runtime/checkpoint 上串联四阶段（语义 → 转移 → 只读意图 → 隔离执行），对**未见任务组合**（训练中未出现过的 目标×文件×语言 组合），完整链产出的意图经真实隔离执行的成功率显著高于冻结语义对照，且语义 lesion 使成功率崩塌——即组合能力来自学习的语义表征，不是 planner 常量或执行运气。

## 2. 串联路径（全部复用既有资产）

| 阶段 | 资产 | 本 canary 的用法 |
|---|---|---|
| 1 语义 | `StructuredSemanticLearner`（semantic_training.py） | 在**已见**组合（文件×语言×目标）上训练 observation→facts/Goal/ContentPlan |
| 2 转移 | `StructuredSemanticTransitionLearner`（semantic_transition.py） | 在多步状态序列上训练（K1 先最小化：单步验证接口贯通） |
| 3 意图 | `NativeReadOnlyIntentPlanner.propose`（read_only_intent.py） | 消费 1+2 输出，对未见组合产出 `ActionIntent`（M3.R3 native 臂路径） |
| 4 执行 | `execute_workbench_intent` → `project_workbench_outcome_for_internalization` → ledger（M5.S6B 链） | 真实隔离执行，真 outcome 按 S6B 政策准入（成功分级/失败负） |

新增胶水（唯一新写）：**参数化任务组合生成器**——生成 (文件, 语言, 目标) 三元组网格，train/holdout 按组合维分割（holdout 组合的每个元素都已在训练出现，但**三元组本身未见**）——这是"组合泛化"的严格定义，避免 S3 的"holdout 太易"教训。

## 3. 臂与判据

| 臂 | 语义 learner | 预期 |
|---|---|---|
| A full-chain | 训练 | 未见组合执行成功率高 |
| B frozen-semantic | 冻结（零训练） | 成功率显著低 |
| C semantic-lesion | 训练后 lesion | 成功率崩塌（因果证明）|

Gate（全部预注册）：

1. **主指标**：A 臂未见组合真实执行成功率 ≥ 0.8，且 A−B ≥ 0.3、A−C ≥ 0.3；
2. 每阶段的既有 Gate 保持（语义 fact F1、unknown fail-closed、planner stale-snapshot、执行 approval/digest/undo）；
3. 真实 outcome 全部经 S6B 政策准入（成功分级、失败负 reward），reward_variance > 0；
4. checkpoint/fresh-restore、core mypy、相关回归不退化。

## 4. 停止线

- canary 先行（单 seed、小网格）；A 臂成功率 < 0.5 → 先归因（语义阶段？planner？执行？），不调阈值硬凑；
- 不引入 provider/联网/真实客户端写入；执行全部在进程私有临时工作区；
- `can_promote=false` 固定——K1 证明的是组合链闭合，不是 A8 完成。

## 5. 明确不做

- 不在本 canary 训练字节预测/fabric（K 的量尺是任务成功率，不是 BPB）；
- 不做多 seed formal（canary 通过后另行预注册）；
- 不把 provider 文本当语义特征。
