# M5.K2 预注册：多步任务序列的组合课程（转移头进入主路径）

> 起草日期：2026-09-09。前置：[K1 canary 预注册 §6/§7](M5_K1_SKILL_COMPOSITION_PREREGISTRATION_20260909.md)（K1.1 类型化绑定后 canary 通过）与 [K1 formal 预注册 §8](M5_K1_FORMAL_PREREGISTRATION_20260909.md)（9/9 robust，证据线闭合）。K1 证明了**单步** (身份×状态) 属性组合；K2 的增量问题是**序列级组合**——转移头的 autoregressive 世界成为 planner 的主路径输入，组合错误会在步间传播并被测量。本文只预注册 K2 canary；formal 通过后另行预注册。执行顺序以 [03_CURRENT_EXECUTION.md](../active/roadmap/03_CURRENT_EXECUTION.md) 为准。

## 1. 可证伪假设

> 在多步任务 episode 上，由转移头 autoregressive 世界驱动的完整链（转移世界 → Stage-1 语义 → 只读意图 → 隔离执行）对**未见 (任务序列 × 语言 × 文件) 组合**的序列级真实执行成功率显著高于冻结对照，且转移 lesion 使其崩塌——即序列组合能力来自学习的转移表征与语义表征的分工，不是 planner 的观察一致性检查（`_world_matches_observation` fail-closed）单独兜底的结果。

## 2. 串联路径（复用 K1 全部资产 + 两处新增）

| 项 | 资产 | K2 用法 |
|---|---|---|
| Stage 1 语义 | `StructuredSemanticLearner` v2（K1.1 类型化绑定） | 与 K1 完全相同：事件接地 facts/Goal/ContentPlan；**零改动** |
| **K2.1 转移头类型化绑定** | `StructuredSemanticTransitionLearner`（`SEMANTIC_TRANSITION_VERSION` bump） | **新增**：消除 K1 §6.3 的 before-block × event-attribute 交叉捷径（见 §2.1） |
| Stage 2→3 主路径 | 转移头 autoregressive 世界 | **新增**：planner 消费转移头预测的 after-world（K1 中为接口检查） |
| Stage 3/4 | planner + `execute_workbench_intent` + S6B 准入 | 与 K1 相同 |

### 2.1 K2.1：转移头 delta 行掩码（精确语义）

转移头输入 = `[before_fact_vector] + [event_context] + [before ⊗ event_context]`，输出为每 fact 的 delta。K2.1 为每条输出 fact f 定义行掩码，只允许非零权重位于：

1. before 段中 **fact f 自己**的位置（持久性）；
2. event_context 段中 **属性(f) 自己的特征**（事件指示；复用 K1.1 的 fact→attribute 特征绑定表）；
3. 交叉段中 **(fact f 自己) × (属性(f) 特征)** 的列（before 值条件下的属性更新）。

其余列强制为零（init / 每训练 epoch / checkpoint 恢复三处重执行，沿 K1.1 模式）。提供掩码时必须**精确覆盖全部 fact keys**（fail-closed）；未提供时行为与现行版本逐位一致（向后兼容）。该绑定使「M-block × ts-event → toolchain missing」类捷径结构性不可表达，序列组合必须经由「持久性 + 事件属性指示」两通道。

### 2.2 多步 episode 世界流（K2 主路径语义）

- 每个 episode = 3 步任务序列；**初始世界 = 中性任务起点锚（missing_00 的世界，M 块），所有 episode 共用**——episode 的每一步世界都由转移头驱动，包括第一步（C 臂 lesion 后世界永远停在锚上，除锚本身外全部一致性失败，崩塌语义干净）；
- 每步 t：`after_world_t = transition.predict(current_world, event_t)`；planner 消费 **after_world_t**（不是 Stage-1 事件接地世界）+ Stage-1 的 goal/content；接受则真实执行；
- 执行后 `current_world = after_world_t`（纯 autoregressive；**真实 outcome 不改写世界**——outcome 只进 S6B 准入记录，outcome→世界反馈属 K3 范围，本 canary 明确不做）；
- 转移头某步预测的世界与观察不一致 → planner fail-closed 拒绝 → 该步失败 → episode 失败（这是被测量的转移能力，不是 harness 容错）；
- Stage-1 的 goal/content 读出掩码（剔除语言身份 facts）与 K1 完全相同。

## 3. 臂与判据

| 臂 | 语义 learner | 转移 learner | 预期 |
|---|---|---|---|
| A full-chain | 训练（K1.1 绑定） | 训练（K2.1 绑定），主路径 | 未见序列成功率高 |
| B frozen | 冻结 | 冻结 | 显著低 |
| C transition-lesion | 训练 | 训练后 `zero_transition_head` | 崩塌（世界不更新→一致性失败） |

**序列与计量**：训练 episodes（train registry，ts 工具链不可用）≥6 个 3 步序列，覆盖 recover/inspect/clarify 起始与块间转移；holdout = **4 个未见 (序列 × 语言 × 文件) 组合**（holdout registry，ts 可用；每步元素已见、组合未见）。「序列成功」= episode 内全部步 planner 接受且真实执行成功；主指标 = 序列成功率（episode 粒度，比 K1 单步严格）。

**Gate（全部预注册）**：

1. **主指标**：A 未见序列成功率 ≥ 0.75（4 中允许至多 1 个 episode 失败），且 A−B ≥ 0.25、A−C ≥ 0.25；
2. 技术门：训练 episodes 上 A 序列成功率 = 1.0（先证明已见序列闭合）；每 episode 全部步产出结构化行；K2.1 掩码在 checkpoint 往返后仍被强制执行；
3. S6B 准入：A 臂全部成功 read outcome 准入率 = 1.0（resolve stale 为已知次要项，不计入）；
4. checkpoint/fresh-restore、core mypy、相关回归（M2.R3/M3.R1/transition 既有测试）不退化；
5. 真实 outcome 全走 S6B 政策，reward_variance > 0。

## 4. 停止线

- A < 0.5 → 先归因（转移头世界错在哪一步？Stage-1 goal？planner？执行？），不调阈值硬凑；
- K2.1 掩码下转移训练不收敛（训练 episodes 技术 Gate 失败）→ 归因绑定设计，不放宽掩码、不改判据；
- planner fail-closed 拒绝激增 → 逐行输出 `reason_code` 分布再归因，不绕过一致性检查；
- 不引入 provider/联网/真实客户端写入；执行全部在进程私有临时工作区。

## 5. 明确不做

- 不改 Stage-1 的 `SEMANTIC_TRAINING_VERSION`、K1.1 掩码语义与 K1 判据（K1 资产冻结复用）；
- 不做 outcome→世界反馈、不做任务间依赖/子目标分解（K3 范围）；
- 不做 K2 formal（canary 通过后另行预注册）；`can_promote=false` 固定；
- 不把转移头的 goal/content 读出接进主路径（K2 分工：转移头管世界，Stage-1 管语义；转移 learner 的 goal/content 读出仅保留接口校验用途）。

## 6. 实现盘点与产物

- `taiji/semantic_transition.py`：K2.1 行掩码 + checkpoint 携带（版本 bump）；
- 新脚本 `scripts/training/eval_taiji_m5_k2_multistep_composition.py`：多步 episode harness（复用 K1 的 workspace/registry/schema/绑定派生与 planner/执行/准入链）；
- 产物：`reports/taiji_m5_k2_multistep_canary_20260909.json` + 路线图执行记录 + 独立提交。
