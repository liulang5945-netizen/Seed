# M5.K3 formal 预注册：outcome→world 任务依赖的跨 seed 稳健性

> 注册日期：2026-09-09。前置：
> [M5_K3_OUTCOME_WORLD_DEPENDENCY_PREREGISTRATION_20260909.md](M5_K3_OUTCOME_WORLD_DEPENDENCY_PREREGISTRATION_20260909.md)
> 的单 cell canary 已通过。本文在 formal 运行前冻结矩阵、量尺、技术门和停止线；formal runner 只能复用 canary 的 `run_cell`，不得复制或改写 cell 逻辑。

## 1. formal 要回答的问题

单 cell 已证明在一个 task/learner seed 上，真实 Workbench outcome 可以经过 typed projection 进入下一步 world/dependency context，并由 dependency lineage gate 约束后续动作。formal 只回答一个更窄的问题：

> 这条 outcome→observed-world→dependency-gated action 链，是否在不同工作区内容和 learner seed 下保持稳健？

formal 不重新解释 K3 的能力范围，不把结果扩展为通用规划、自进化、开放域语言理解或默认 runtime 晋级证据。

## 2. 固定实现和不变项

每个 cell 必须直接调用：

```python
cell = run_cell(task_seed=task_seed, learner_seed=learner_seed)
```

保持不变：

- `taiji/outcome_dependency.py` 的 typed projection、content address、checkpoint/restore、lesion 与 fail-closed 规则；
- K3 canary 的两套隔离 workspace、holdout task-seed 偏移、全语言 fixture vocabulary、跨语言训练序列和 1280 轮 transition 训练预算；
- K1.1 semantic fact mask、K2.1 transition row mask、semantic/transition/planner/S6B owner；
- 三步 episode、probe 真实 `workspace.read`、failure branch 的真实删除/恢复 fixture、follow-up/verification 的 dependency digest echo；
- A/B/C 三臂、`learn=False`、planner stale-world gate、进程私有工作区和 `can_promote=false`；
- 不改变 materialization threshold、confidence/ambiguity threshold、A/B/C 判据或 holdout episode 数量。

formal 不测：provider/联网、MCP、真实客户端写入、CUDA、视觉 UI、语言 provider artifact、默认 runtime 接入或结构 promotion。

## 3. 矩阵和执行协议

固定 `3 × 3 = 9` 个 cell：

- `task_seed = (0, 1, 2)`：改变训练与 holdout workspace 的内容变体；
- `learner_seed = (17, 23, 31)`：改变 Taiji/runtime episode seed；
- 顺序固定为 task seed 外层、learner seed 内层，串行执行，避免全局 workspace selector 与 Windows 临时目录相互污染；
- runner 只 import canary 的 `run_cell`，不复制 `_run_episode`、projection、planner 或 outcome admission 逻辑；
- 任一 cell 返回异常时立即停止，报告为 harness/environment failure，不跳过该 cell、不换 seed、不把部分结果聚合成通过。

## 4. 每 cell 指标和技术 Gate

### 4.1 主指标

- `A_holdout_success`：A 臂 4 个 holdout episode 中三步均通过 planner、真实执行和 lineage 的比例；
- `A_minus_B`：A full-feedback 减 B no-feedback 的 episode success 差值；
- `A_minus_C`：A full-feedback 减 C outcome-lesion 的 episode success 差值。

### 4.2 技术 Gate

每个 cell 必须全部满足：

1. canary `checks` 全部为 true；
2. A 训练 episode success `= 1.0`，训练 episode 数 `≥ 6`；
3. A 的每个 probe 都产出真实 success/failure outcome，且 S6B admission 全部通过；
4. A 的每个实际进入 dependency gate 的 follow-up/verification 都有匹配的 boundary token lineage；
5. feedback reward population variance `> 1e-12`；
6. pre-fit checkpoint round-trip、post-fit owner/mask/projector checkpoint gate 全部通过；
7. A 的每个依赖步骤都消费 outcome fact，不能只凭当前 observation 通过；
8. canary 的逐行报告完整产生 A/B/C 三臂，不能因早停而把空行当成成功。

projection 的 stale event、duplicate event、cross-scope event、wrong outcome 和 tampered checkpoint 由 K3 定向单测固定验证；formal runner 不复制第二套 projection 测试逻辑。

### 4.3 看结果前冻结的 formal 判据

每个 cell 同时满足：

- `A_holdout_success ≥ 0.75`；
- `A_minus_B ≥ 0.25`；
- `A_minus_C ≥ 0.25`；
- §4.2 的全部技术 Gate 通过。

只有 `9/9` cell 全部通过时，才记录 `robust=true`。报告必须给出 A/B/C 的 min/mean/max、A-B/A-C 的 min/mean/max、训练成功率、probe admission、lineage admission、reward variance 和技术 Gate 通过数。不能用 aggregate mean 抵销单 cell 失败。

无论 formal 是否通过，`can_promote=false` 固定；formal 通过只闭合 K3 的跨 seed 证据线，不授予默认 runtime 接入权。

## 5. 停止线和归因顺序

- 任一 cell 的 A `< 0.50`：先按 probe/follow-up/verification 逐行输出 predicted world、transition/semantic status、planner reason、projection reason、真实 outcome 和 lineage；不调阈值；
- A `≥ 0.50` 但低于 `0.75`：保留该 cell，按跨语言序列、workspace digest、学习预算和 world conflict 归因，不挑 seed、不删除失败行；
- 任一 probe admission、lineage、checkpoint 或 lesion gate 失败：停止 formal，先修复对应 owner/contract，不能把失败降级为“仅诊断”；
- B/C 仍通过：优先检查 outcome branch、dependency digest 和当前 observation 是否发生静态泄漏；不能据此宣称 outcome feedback 已证明必要；
- Windows pytest/temp ACL、权限、workspace selector 或外部环境错误：单独标记 harness/environment，不能转成模型分数，也不能放宽主判据；
- 9/9 通过：只记录 K3 formal 证据闭合，继续保持 shadow，不自动接入 default runtime、MCP/provider/client/CUDA 或 structural growth。

## 6. 产物和唯一后续动作

- runner：`scripts/training/eval_taiji_m5_k3_outcome_dependency_formal.py`；
- report：`reports/taiji_m5_k3_outcome_dependency_formal_20260909.json`；
- 计划同步：执行完成后把逐 cell 结果、aggregate、失败归因和 `can_promote=false` 写回本文件与 `plans/active/roadmap/03_CURRENT_EXECUTION.md`；
- 当前唯一允许的下一步：实现上述 runner，先静态检查，再按固定顺序串行运行 9 个 cell；formal 运行前不修改 canary、不接默认 runtime、不引入外围变量。

