# M5.K2 formal 预注册：多步组合的跨 seed 稳健性

> 注册日期：2026-09-09。前置：
> [M5_K2_MULTISTEP_COMPOSITION_PREREGISTRATION_20260909.md](M5_K2_MULTISTEP_COMPOSITION_PREREGISTRATION_20260909.md)
> 的 canary 已通过。本文只冻结 formal 的矩阵、统计口径和停止线；实现和运行必须在本文落盘后进行。

## 1. 研究问题与不变项

formal 只回答一个问题：K2 canary 观察到的「转移头 autoregressive world → Stage-1 goal/content → read-only intent → 隔离 Workbench 执行」是否跨 task/course seed 与 learner seed 稳健。

保持不变：

- Stage-1 `StructuredSemanticLearner`、K1.1 fact-feature mask、readout exclusion 和 K1 资产；
- K2.1 transition row mask、三步 episode、missing_00 初始世界、纯 autoregressive world 流；
- planner fail-closed 一致性检查、S6B read admission、`learn=False`、进程私有工作区；
- canary 的 S6 `s6-graded-v1` read reward 口径；
- 所有 arm 和指标定义；formal runner 只能 import canary 的 `run_cell`，不得复制或改写 cell 逻辑。

不测：provider/联网/真实客户端写入、CUDA、K3 outcome→world、任务依赖/子目标分解、K1 判据重算或任何结构 promotion。

## 2. 矩阵与复用协议

固定 `3 × 3 = 9` cells：

- `task_seed = (0, 1, 2)`：改变工作区内容与组合课程样本；
- `learner_seed = (17, 23, 31)`：改变 runtime/episode seed；
- 运行顺序：task seed 外层、learner seed 内层，串行执行，避免临时工作区/全局设置交叉污染。

每个 cell 必须直接调用：

```python
cell = run_cell(task_seed=task_seed, learner_seed=learner_seed)
```

不得在 formal runner 中筛选 seed、重训、改学习率、改阈值、改 holdout、移除失败行或重新解释 resolve stale。

## 3. 每 cell 指标与 Gate

### 3.1 主指标

- `A_holdout_success`：4 个 holdout episode 中三步全部 planner 接受且真实执行成功的比例；
- `A_minus_B`、`A_minus_C`：A 与 frozen/transition-lesion 的 episode success 差值。

### 3.2 每 cell 必须通过的技术门

1. canary technical checks 全部通过；
2. A 训练 episode success `= 1.0`，训练 episode 数 `≥ 6`；
3. A dev/test episode 完成；
4. A 臂所有成功 `workspace.read` outcome 的 S6B admission rate `= 1.0`；
5. A reward population variance `> 1e-12`；
6. K2.1 checkpoint round-trip、篡改禁用列防护、post-fit mask enforcement 全部通过；
7. core mypy/ruff/相关 transition 回归不退化。

### 3.3 formal 判据（看结果前冻结）

每个 cell 必须同时满足：

- `A_holdout_success ≥ 0.75`；
- `A_minus_B ≥ 0.25`；
- `A_minus_C ≥ 0.25`；
- 上述 3.2 技术门全部通过。

formal aggregate 只有在 `9/9` cells 通过时才记为 `robust=true`；同时报告 A 的 min/mean/max、两项分离差的 min/mean/max、技术门通过数和 read admission 通过数。这里不另设事后均值补偿：单 cell 失败即停止并逐行归因。

## 4. 停止线与解释顺序

- 任一 cell 的 A `< 0.5`：先输出每一步的 predicted world、planner `reason_code`、Stage-1 status、transition status 和真实 outcome，区分绑定/转移/语义/执行；不调阈值；
- 任一 cell 的 checkpoint 或 mask 门失败：停止 formal，不放宽 mask；
- A≥0.5 但未达到 0.75：按 cell 归因，不挑 seed、不删除 cell、不把 B/C 的失败解释成 A 的成功；
- 9/9 通过也只闭合 K2 证据线，`can_promote=false`，不将 transition 结构带入默认路径；
- 任意外设或环境错误必须单独标为 harness/environment，不得转成模型分数。

## 5. 产物与唯一后续动作

- runner：`scripts/training/eval_taiji_m5_k2_multistep_formal.py`；
- report：`reports/taiji_m5_k2_multistep_formal_20260909.json`；
- 执行后将逐 cell 结果、aggregate 与归因写回本文件 §6，并同步 `plans/active/roadmap/03_CURRENT_EXECUTION.md`。

**formal 之前唯一允许的下一步**：实现上述 runner，零逻辑复制地 import canary `run_cell`，先运行静态检查再串行运行 9 cells。formal 期间不进入 K3、不引入 MCP/provider/client/CUDA 变量。
