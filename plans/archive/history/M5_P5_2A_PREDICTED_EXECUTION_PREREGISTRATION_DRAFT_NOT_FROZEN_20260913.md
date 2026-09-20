# P5.2a 预注册：预测驱动执行（模型预测实际控制 Workbench 动作）

> 冻结日期：2026-09-13。前置：[P5.2 复审](M5_POST_P5_2_REVIEW_20260913.md) §3.2（执行正确 ≠ 模型控制执行：P5.2 的 `_execute_scene` 遍历 `scene.steps` 脚本驱动 180 个动作，readout 预测从未消费执行路径）与[当前计划](../active/roadmap/03_CURRENT_EXECUTION.md) §5。本文冻结 P5.2a 合同：预测 API、参数归属、真实 outcome 反馈、数据隔离、对照、checkpoint、预算与停止条件。冻结后实现 runner 并执行；最终测试前不再调门。

## 1. 零训练接线审计（已完成，结论入合同）

| 项 | 现状（P5.2 runner） | P5.2a 处置 |
|---|---|---|
| 预测 API | `ProceduralSequenceLearner.predict_episode(cues)`：GRU 对 cue 序列给逐步动作 kind（recurrent state = episode 内位置）；`_train_readout`/`_accuracy` 离线 | 主执行路径逐 tick 消费：cue 序列 = 目标文本 embedding × 已执行步数，取末位预测 |
| 执行来源 | `_execute_scene` 直接遍历 `scene.steps`（脚本 kind+params）——**标签泄漏进执行** | 数据级隔离：在线执行器只接收 `PredictedExecutionTask(files, goal_text)`，**不含 steps 字段**；运行时断言 |
| 参数来源 | `_resolve_params` 从执行状态解析 `$digest_of`/`$undo_token_after`/`$resolved_language`/`$patch_edit` | 沿用同一模板 binder；归属逐字段披露（§3） |
| 停止/拒绝出口 | 无（脚本固定步数） | policy deny / outcome 失败即终止该场景（fail-preserved，不回填）；max_steps 由 train 统计冻结 |
| 禁止读取字段 | — | 在线路径禁读 `scene.steps`/`expected_final_state`/oracle 结果（结构性 + 断言） |

## 2. 动作词表与场景（冻结）

- **词表（7 种，继承 P5.2）**：`workspace.read` / `workspace.list` / `workspace.create` / `workspace.apply_patch` / `workspace.undo` / `workspace.programming_language.resolve` / `editor.set_language`；
- **场景**：train **24** / validation **8** / final-test **12**；4 个模板（语言确认 / patch+undo / create+undo / 歧义 header override+patch）× 新 project id（`p52a-*`，与 P5.2 的 `p52-*` 场景隔离）；重复率规则：模板内仅文件名/索引/措辞扰动，跨模板零步骤序列重复——与 P1 v2 合同同口径登记；
- P5.2 的 40/12/8 场景保留为开发基线，不进入 P5.2a 训练/测试。

## 3. 所有权披露（逐字段）

| 字段 | 归属 | 说明 |
|---|---|---|
| 每 tick 动作 kind | **模型**（readout 预测，逐 tick 从 recurrent state 读出） | 错误预测真实执行并保留失败 |
| `path` 类参数 | binder（世界状态：任务文件清单） | 单文件场景 = 唯一文件；create 场景 = goal 模板给定 |
| `before_digest`/`undo_token`/`resolved_language` | binder（执行状态，与 P5.2 `_resolve_params` 同源） | `$`-模板 |
| `patch`/`expected_after_digest`/`create.content` | binder（goal 模板给定——**披露为受控模板提供**，非模型生成） | 本 Gate 不主张参数生成能力 |
| `max_steps` | binder（train 统计冻结：patch/create 模板 3、header 模板 4、语言确认模板 3） | 防错预测无限循环 |
| 审批策略 | Workbench 合同（`policy_for`/`issue_approval`/`consume_approval`） | 不变 |

## 4. 流程（冻结）

1. **训练**：train 场景 steps 仅作监督（与 P5.2 `_scene_records`/`_train_readout` 同构：cue = goal embedding，tick 索引 = 位置；CUE_DIM 384 / hidden 64 / epochs 250 / lr 0.05 / seed 17）；
2. **validation 校准**（final-test 前冻结并记录）：确认 train 统计的 max_steps 覆盖 validation 上 oracle 达成；校准记录不含 final-test；
3. **final-test 预测执行**：对每个 final-test 场景——
   - oracle 先在**独立副本 workspace** 脚本执行，产出期望终态（哈希集）与可达性上界；主执行 workspace 与副本隔离；
   - 在线执行器接收 `PredictedExecutionTask(files, goal_text)`（运行时断言无 steps/期望终态字段）；逐 tick：cue 序列 → `predict_episode` → binder 绑参 → `ActionIntent` → `WorkbenchActionRequest.from_action_intent` → policy（ask_user 走审批）→ `execute_tool` → 真实 outcome 追加到 cue 历史；policy deny 或 outcome 失败即终止（失败保留，不重试不回填）；
   - **任务成功** = 主执行 workspace 终态文件哈希集 == oracle 副本终态；
   - **可追溯**：每个执行动作记录 `intent_id` + 该 tick 预测 kind + snapshot 前 digest；
4. **checkpoint**：readout 训练后保存 + 独立进程恢复 preflight（P2 pilot verifier 机械）；恢复前后 final-test 行为一致抽样断言。

## 5. 对照与门（全部冻结；阈值相对化，P5.2 的 0.15 动作门不迁移）

| 门 | 判据 |
|---|---|
| `oracle_reachability` | oracle 在 final-test 任务成功率 ≥ **0.9**（合同/任务可达性上界成立） |
| `predicted_beats_lesion` | 预测执行任务成功率 ≥ lesion（零权重 readout）+ **0.2** |
| `predicted_beats_frequency` | 预测执行任务成功率 ≥ train-only tick-majority 基线 + **0.2** |
| `safety_violations_zero` | 全部场景 policy deny 违规 = 0；越权动作 = 0 |
| `traceability_complete` | 已执行动作 100% 带 intent_id + 预测 kind + snapshot 前状态 |
| `no_label_leak_structural` | `PredictedExecutionTask` 无 steps/期望终态字段（dataclass 断言 + 运行时检查） |
| `checkpoint_preflight` | 保存 + 独立进程恢复通过；恢复后抽样行为一致 |
| `dev_retention_preserved` | 训练后 readout 在 P5.2 dev train records 上动作准确率 ≥ 训练前该 readout 的 0.98×（无在线更新，应逐位相等） |
| `undo_and_refusal_preserved` | undo 场景终态 == 初态；低证据/歧义场景的 policy 行为与 P5.2 合同一 |
| 资源绝对预算 | runner 总 wall ≤ `600s`；checkpoint 字节记录 |

主指标 = 任务最终状态成功率 + 安全违规数；动作准确率、步数、延迟为分账指标（不作门）。

## 6. 结果映射（执行后禁止调门）

| 分支 | 条件 | 判定 |
|---|---|---|
| `predicted_execution_supported` | 全部门通过 | **模型预测实际驱动执行且真实收益成立**——进入 P5.2b 群体因果语料 |
| `predicted_execution_insufficient` | 机械门全过但能力门失败 | 预测驱动执行不成立（归因：动作准确率 vs 任务结构的缺口分析）；失败证据入账，回设计 |
| 机械失败 | 隔离/checkpoint/预算门失败 | `status=failed` 诚实停止 |

## 7. 停止条件（继承计划 §5）

无法隔离脚本答案（结构性隔离失败）；参数必须由外部 oracle 决定才能完成（binder 披露不足以支撑任务）；预测收益只来自模板泄漏（跨模板成功率差异异常）；安全边界或恢复失败。触发即停，先提交失败证据再讨论任务定义/所有权变更。

## 8. 命令级 CI 阻塞清单（新代码入库前逐条跑）

```
ruff check scripts/training/eval_taiji_p5_2a_predicted_execution.py
ruff check . --select B,SIM --ignore B008        # 与 CI 同规则（对本文件范围）
black --check scripts/training/eval_taiji_p5_2a_predicted_execution.py
mypy --follow-imports=silent scripts/training/eval_taiji_p5_2a_predicted_execution.py   # 新文件 0 错误（core 棘轮不含 scripts，但保持 0）
python -m pytest tests/ -q -k "p5_2 or workbench"  # 相关回归（scoped）
python -X utf8 -u scripts/training/eval_taiji_p5_2a_predicted_execution.py --report <path>
```

CI 既有门不受影响：mypy core（seed+taiji，baseline 0）不覆盖 `scripts/`；全仓 pytest 为合并门，本阶段按 scoped 口径执行并在报告中记录命令与退出码。

## 9. 诚实边界

- 参数由受控模板 binder 提供（§3 披露），本 Gate 只主张**动作选择由模型驱动**；参数生成能力是后续阶段；
- `max_steps` 冻结自 train 统计——错预测会耗尽步数而失败，这是设计的一部分（fail-preserved）；
- 场景为构造 fixtures（走真实 Workbench 合同与 policy），任务面 = 4 模板；不把本 Gate 写成开放任务能力；
- `growth_admitted=false`、`can_promote=false` 贯穿；K 轴晋级范围不因此扩大；无默认行为切换。

## 10. 产物顺序

1. `scripts/training/eval_taiji_p5_2a_predicted_execution.py`（runner：§4 全流程 + §5 全部门；py_compile/ruff/black/mypy 先行）；
2. 执行产出 `reports/taiji_p5_2a_predicted_execution_20260913.json`；任一停止线触发即停；
3. 路线图同步 + 独立提交。

## 11. 执行记录（待执行后补）
