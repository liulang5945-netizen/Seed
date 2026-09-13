# M5.K 默认 runtime 附着工程预注册：owner/学习机制接入产品 runtime

> 冻结日期：2026-09-12。前置：[A8 评审合同 §7](M5_K_A8_PROMOTION_REVIEW_20260912.md)（附着授权已独立批准 + 「优先上限更高」常设准则）与 [scorecard v7](M5_K_AXIS_SCORECARD_V7_CONTRACT_20260912.md)。本文冻结附着步的**范围、设计、验收门、结果映射与 fail-closed 边界**；冻结后按 §7 执行。本步零训练（`fit_called=false`）、不改任何已冻结判据；设计按上限更高准则取**真实 runtime 集成**（非沙箱旁路）。

## 1. 可证伪假设

rollout review 在沙箱 runner 中验证了消费合同；本步假设：**同一合同可在产品 runtime 内以 runtime 自有代码实现**——`SeedRuntime` 获得显式、可选、fail-closed 的附着入口，附着后的 consumer 状态在 runtime 路径上逐字段复现 P4.14/review 记录值，且对产品默认行为与全部研究产物零干扰。

## 2. 设计（冻结；上限更高选项）

- **新模块 `api/taiji_runtime_attachment.py`（runtime 自有，不得 import `scripts.*`）**：以 `taiji` 包类（`StructuredSemanticLearner`/`StructuredSemanticTransitionLearner`/`ExtendedGSelectionLearner`）实现消费合同 `taiji-default-runtime-rollout-attachment-v1` 的独立 fail-closed loader——load 顺序与 review 冻结值一致（manifest 自校验 → P4.14 lineage → 逐位 digest → 嵌入不变量 → 独立进程恢复 → 篮改/错 cell 混装拒绝）；G digest 采用排除内嵌 `checkpoint_digest` 键的约定，K2 `fact_threshold=0.55` 与 K1 `0.65` 按角色钉定；loader 无任何 override 通道；
- **`SeedRuntime.attach_k_g_state(attachment_path, cell_index)`（显式 opt-in）**：经模块原子附着（任一校验失败则整体拒绝、runtime 保持未附着并记录拒绝原因）；`detach_k_g_state()` 仅内存卸载；runtime status 新增 `k_g_attachment` 段（attached/cell/manifest digest/artifact digests/attached_at）；拒绝与附着事件进 runtime 状态，能力面可见（capability surface 注册）——**不改默认 chat/workbench 行为路径**（默认路径采用该机制属 promotion 后的独立决策）；
- **typed 读出面**：附着状态暴露 `k1_predict(percept)` / `k2_predict(world, event)` / `g_select(candidate_set)` 与 cell 内嵌安全不变量只读视图；
- **独立进程校验模式**：模块提供 `--verify-attachment <manifest> <cell>` 子命令（fresh 进程完整 preflight + K/G 恢复 + 行为快照），供附着与验收 runner 双向调用。

## 3. 验收门（全部冻结；4/4 cell 全部适用，无子采样）

| 门 | 冻结判据 |
|---|---|
| 附着 preflight | 每细胞经 runtime 模块附着成功；digest/不变量/lineage 全过；独立进程校验模式通过 |
| 消费等价（runtime 路径） | 冻结 cohort 经 **runtime 暴露的读出面**（非研究 consumer）重评：Phase K（novel 2/2、旧类 4/4、安全 6/6）与 Phase G（validation/holdout/retention-newtask `0.8/0.8675/0sv`、sibling `1.0/1.0`）逐字段等于 P4.14 记录值 |
| fail-closed 拒绝 | 篡改 payload、错 cell 混装、错 lineage、缺文件：附着被拒绝且 runtime 保持未附着、拒绝原因入状态 |
| rollback / detach | detach → re-attach 后行为逐数值等价；附着全程 `fit_called=false` |
| 非干扰 | `checkpoints/seed_corpus.pt` sha256 前后一致；全部研究 artifact digest 前后一致；默认 chat/workbench 行为零变化（status 之外的运行时行为不受附着影响） |
| 资源绝对预算 | 每细胞 attach ≤ `60s`、验收运行总 ≤ `600s` |
| 静态纪律 | 新模块 + 定向测试通过 py_compile/ruff/black/mypy 与相关 pytest |

## 4. 矩阵与产物顺序

- **矩阵**：P4.14 全部 4 cell（消费对象与 review 完全相同；附着 manifest 复用 `taiji_m5_k_default_runtime_rollout_attachment_v1.json`，digest 钉定）；
- **顺序**：(1) `api/taiji_runtime_attachment.py` + 定向测试；(2) `SeedRuntime.attach_k_g_state/detach/status` 接入；(3) 验收 runner `scripts/training/eval_taiji_m5_k_runtime_attachment.py`（py_compile/ruff/black/mypy 先行）；(4) 执行落盘 `reports/taiji_m5_k_runtime_attachment_20260912.json`；任一停止线触发即停。

## 5. 结果映射（执行后禁止调门）

| 分支 | 条件 | 判定 |
|---|---|---|
| `runtime_attachment_supported` | 4/4 cell 全门（附着 + runtime 消费等价 + fail-closed 拒绝 + rollback + 非干扰 + 预算） | scorecard v8 翻转 `default_runtime_owner_attached` 与 `learning_mechanism_attached_default_runtime`——届时五项 gate 全 true，`promotion_gate`/`can_promote` 机械置 true，**最终晋级仍须独立批准** |
| `runtime_attachment_consumption_gap` | 任一 cell runtime 路径行为 ≠ 记录值 | 停止并归因（合同实现偏差）；不调门、不重训 |
| `runtime_attachment_refusal_gap` | fail-closed 拒绝门失败（该拒的没拒/不该拒的拒了） | 停止修 loader，诚实入账 |
| `rollback_unverified` / 机械失败 | detach-reattach 不等价 / digest/预算失败 | `status=failed` 停止 |

## 6. 边界（诚实声明）

- 附着 ≠ promotion：默认任务解释路径不改写（那是 `can_promote` 置位且独立批准后的决策）；
- 附着状态为显式 opt-in、进程内内存态（runtime 重启后需重新附着——持久化附着属后续设计，不在本步）；
- 五类合成课程仍是唯一载体；不读取 sealed；`growth_admitted=false` 贯穿。

## 7. 执行记录（2026-09-12，附着已执行）

1. **实现**：`api/taiji_runtime_attachment.py`（runtime 自有消费合同实现，零 `scripts.*` import；load 顺序与 review 冻结值一致；G digest 排除内嵌 `checkpoint_digest` 键约定；K1/K2 fact_threshold 按角色钉定 0.65/0.55；G parent_manifest_digest 对 P4.1 `source_p3_2_manifest_digest` 校验；篡改/混装探针内建）+ `SeedRuntime.attach_k_g_state/detach_k_g_state/k_g_attachment_status`（显式 opt-in、原子拒绝、拒绝原因入 status、`status()` 新增 `k_g_attachment` 段）+ 独立进程校验子命令 `--verify-attachment`。静态纪律：py_compile/ruff/black/mypy 全过（`seed_runtime.py` mypy 错误数 24 = 改动前基线，新代码零新增）；定向测试 15/15 通过（`tests/test_taiji_runtime_attachment.py`）。
2. **验收 runner**：`scripts/training/eval_taiji_m5_k_runtime_attachment.py`（py_compile/ruff/black/mypy 全过）。执行产出 `reports/taiji_m5_k_runtime_attachment_20260912.json`：**`outcome=runtime_attachment_supported`，4/4 cell 全门**。
   - **消费等价（runtime 路径）**：Phase K（novel 2/2、旧类 4/4、安全 6/6）与 Phase G（validation/holdout/retention-newtask `0.8/0.8675/0sv`、sibling `1.0/1.0`）经 runtime 附着 consumer 逐字段复现 P4.14 记录值；cohort 重物化 digest 与 P4.14 manifest 逐位一致（per-batch 第一 cell 用 runtime 附着 learners 物化）。
   - **fail-closed**：篡改 digest / 缺文件 / 错 cell 混装三个探针全部拒绝，拒绝后 runtime 保持未附着且原因入 status。
   - **rollback**：detach → re-attach 后 Phase K/G 指标逐数值相等（4/4）。
   - **非干扰**：`checkpoints/seed_corpus.pt` sha256 不变、model tick/参数计数不变、P4.14 报告不变、全部研究 artifact 复验通过；附着为 opt-in，不进入默认 chat/workbench 路径。
   - **资源**：attach 1.32–1.36s ≪ 60s/cell；总 265.7s ≪ 600s；`fit_called=false`、`training_performed=false`。
3. **scorecard v8**：[M5_K_AXIS_SCORECARD_V8_CONTRACT_20260912.md](M5_K_AXIS_SCORECARD_V8_CONTRACT_20260912.md) 冻结并运行——两项附着门翻转，五项晋级 gate 全 true，`promotion_gate=true`、`can_promote=true`（机械置位）；**最终晋级宣布仍须独立批准**。
4. `growth_admitted=false` 贯穿；无 sealed 读取；默认行为路径零改动。唯一下一步 = 晋级宣布的独立批准。
