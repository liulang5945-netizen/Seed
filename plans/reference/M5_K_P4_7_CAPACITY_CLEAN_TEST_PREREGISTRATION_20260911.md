# M5.K P4.7 预注册：容量假设干净检验（继承式 22 参数 + functional 保持协议）

> 冻结日期：2026-09-11。前置：[结果复审 §32](M4V2_POST_C_STAGE_RESULT_REVIEW_20260910.md)（P4.0–P4.6 机制综合与决策分析）、路线图「当前决策点」。本文在 runner 实现与任何 fit 之前冻结臂、门、阈值与结果映射；冻结后按 §7 顺序执行。

## 1. 可证伪假设

P4.3–P4.6 钉死：13 参数 G 上「保持 parent 选择面」与「学习新任务选择面」在 seed 间系统性互斥，rehearsal（数据层）、trust-region（参数空间）、functional teacher（函数空间）三类干预均无法同时满足。P4.0 确认容量压力真实（width 8/12 residual 0.54/0.59），但 P4.2 的「fixed-large 无收益」结论受坏协议污染（连 fixed-small 都保持退化时测得），容量假设从未被干净检验。

**假设**：若 13 参数互斥由容量引起，则在**完全相同的 functional 协议**下，把 G 从 13 参数继承式扩展到 22 参数（出生时行为与 parent 逐位等价）将使两 seed 同时通过保持与新任务门。若 22 参数臂仍互斥，则容量假设关闭——互斥是更新规则/表示问题，与容量无关。

## 2. 单变量设计与臂（恰好两臂）

单变量 = **容量 13 → 22**。更新规则、协议、数据、teacher 完全相同。

1. **`functional-13`**（P4.6 复现基线）：13 参数 G 从 P3.5 trained-G parent 继承，functional teacher 协议与 P4.6 逐参数相同（epochs=8、lr=0.15、functional_weight=1.0、order_seed=seed、同一 train fit 集、同一 constraint cohort、同一交错序列——每步 task delta 后接 constraint delta，`training_steps += 2`）。预期复现 P4.6 的 seed 间张力。
2. **`functional-22-inherited`**：22 参数 context G（`taiji/g_selection_context.py`）——`nn.Linear(21, 1, bias=True)`，输入 = 12 维 candidate features + 9 维候选集 context features（P4.1 合同名集与语义）；**继承式初始化**：candidate 权重与 bias 从 P3.5 parent 逐位复制、9 维 context 权重零初始化。functional teacher 仍锚定 **P3.5 parent 的 13 参数输出**（两臂同 teacher 值——同一 constraint cohort 上 parent 对 candidate 部分的打分）。

不做第三臂、不同时改更新规则、不自动扩到 9-cell。`new-only` 不作为臂——其 frozen 基线数值（P4.3/P4.5/P4.6 三轮稳定复现的 holdout utility `0.68`、target hit `0.6`）直接作为新任务门阈值。

## 3. 出生等价门（继承式初始化的干净性）

零 fit 的 22 参数臂在**全部评估 split**（validation / holdout / retention-sibling / retention-newtask）上与 parent 的选择完全一致（selection_status + selected_candidate_id 逐 record 相同；score 偏差作为诊断记录）。这是 P4.1 context-lesion 合同（「零 fit 下选择与当前 G 完全一致」）在继承式初始化上的重验。门失败 = 扩展算子不干净，实验无效（status=failed）。

## 4. 数据与身份

- 与 P4.1–P4.6 全部 manifest 的 project/path/candidate/behavior digest 隔离（p47 前缀、全新 task seeds）；
- train / validation / holdout / retention-newtask：各 20 records（5 类 × 宽度 2/4/8/12，P4.6 压力课程同构），train 仅 margin>0 者可 fit；
- constraint cohort 与 retention-sibling：各 4 records，P4.4 结构合同行（1×6-candidate + 3×2-candidate）逐行同构、全新身份；retention-sibling 结构 digest 与 P4.4 合同逐位校验；
- constraint cohort 的 behavior target/utility 不进入 fit（fit 只读 candidate feature vectors 的 21 维拼接）；retention 不进入 fit；
- 不读取任何 sealed payload（本实验无 sealed 环节）。

## 5. 门（全部在执行前冻结）

**新任务门**（逐 arm 逐 seed，holdout 上）：

- `selected_utility_mean ≥ 0.68 − 1e-9`（frozen new-only 基线，P4.3/P4.5/P4.6 三轮稳定）；
- `behavior_target_hit_rate ≥ 0.6 − 1e-9`；
- `safe_selection_violations == 0`；
- `reobserve_projection_passed == true`。

**保持门**（逐 arm 逐 seed，**两个 retention split 都必须通过**——P4.4 教训的落地）：

- retention-sibling（保持分布）：utility / target hit 非劣于 **in-run parent** 在同一 split 的实测（parent 为 P3.5 trained-G，预期 1.0/4-4 量级），且 safe violations == 0、reobserve projection 通过；
- retention-newtask（新任务分布）：同上非劣于 parent 在该 split 的实测（ catastrophic forgetting 检查——防止「在新任务分布自身上退化到 parent 之下」）。

**机械门**：两 arm × 两 seed 的零步/训练后 checkpoint 保存 + 独立进程恢复 + tamper 拒绝 + parent 未覆盖；22 参数臂 parameter_count == 22、13 参数臂 == 13；K1/K2 digest 不变；identity/结构门全过。

## 6. 结果映射（全分支；执行后禁止调门/阈值）

| 分支 | 条件 | 判定 |
|---|---|---|
| `capacity_bottleneck_supported` | 出生等价过 ∧ 22 参数臂两 seed 同时过新任务门 + 双保持门 ∧ 13 参数臂在 ≥1 seed 上未同时通过（张力复现） | **容量瓶颈假设成立** → 进入继承式结构成长预注册（出生零影响 + 可测新增贡献；P4.1 复制算子即最小 growth 机制） |
| `capacity_hypothesis_closed` | 出生等价过 ∧ 22 参数臂在 ≥1 seed 上未同时通过 ∧ 13 参数臂张力复现 | **容量假设关闭** → 互斥定性为更新规则/表示问题；固定容量路线整体收束，进入表示合同重设计 |
| `tension_not_reproduced` | 13 参数臂两 seed 全过（P4.6 互斥在 fresh 数据上不复现） | 容量比较失效 → 回数据敏感性分析；22 参数臂结果仅描述性记录 |

- 机械/身份/checkpoint 任一失败 → `status=failed` + 错误归因，停止；
- `growth_admitted=false`、`can_promote=false` 贯穿所有分支（`capacity_bottleneck_supported` 打开的是**下一份预注册**的入口，不直接解冻任何 runtime/promotion）；
- dynamic growth、P5、CUDA、IDE/provider 在判定前继续冻结。

## 7. 产物顺序

1. `taiji/g_selection_context.py`（ContextGSelectionLearner + context 特征计算）+ 定向测试（出生等价/上下文影响/checkpoint 往返/tamper/lineage）；
2. `scripts/training/eval_taiji_m5_k_p4_7_capacity_clean_test.py`（py_compile/ruff/mypy 先行）；
3. 执行产出 `plans/manifests/taiji_m5_k_p4_7_capacity_clean_test_manifest_v1.json` + `reports/taiji_m5_k_p4_7_capacity_clean_test_20260911.json`；
4. 路线图/记录文档同步 + 独立提交。

## 8. 执行记录（2026-09-11，已运行，容量假设关闭）

1. 实现顺序：`taiji/g_selection_context.py` + 定向测试 6/6（出生等价/上下文影响/显式线性组合/往返 tamper/functional fit 边界/父代容量拒绝）+ py_compile/ruff/mypy 通过后运行 runner。首轮失败归因：`_materialize_case` 硬编码只对 `p40_` 前缀路径写文件内容，`p47_` 前缀导致 proposal 池为空——路径改为 `p40_p47_`（保持内容生成触发，身份仍隔离）后重跑，未改任何判据。
2. 报告 `reports/taiji_m5_k_p4_7_capacity_clean_test_20260911.json`：`status=completed`、`experiment_passed=true`（机械门全过：出生等价 64 record 0 mismatch / 最大 score 偏差 0.0、身份与 P4.1–P4.6 隔离、checkpoint 独立恢复 + tamper 拒绝、参数 13/22 精确）。
3. 结果：**`outcome=capacity_hypothesis_closed`**。13 参数臂逐数值复现 P4.6 张力（seed-0 败新任务 `0.585/0.45`+6 violation、过保持 `1.0/1.0`；seed-1 过新任务 `0.68/0.6`、败保持 `0.8/0.75`）；22 参数臂未改变定性形态（seed-0 `0.625/0.45` 仍败、seed-1 与 13 参数逐数值相同）。`growth_admitted=false`、`can_promote=false`。
4. 按 §6 冻结映射：固定容量路线（P4.0–P4.7）整体收束，进入表示合同重设计；判定前冻结的外围（dynamic growth、P5、CUDA、IDE/provider）继续冻结。
