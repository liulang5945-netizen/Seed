# M5 K 轴 scorecard v8：默认 runtime 附着入账与机器晋级边界闭环

> 注册日期：2026-09-12。前置：[默认 runtime 附着工程预注册 §7](M5_K_DEFAULT_RUNTIME_ATTACHMENT_PREREGISTRATION_20260912.md)（`runtime_attachment_supported`——4/4 cell 全门通过）与 [scorecard v7 合同](M5_K_AXIS_SCORECARD_V7_CONTRACT_20260912.md)。本 v8 在不重训、不重算任何 cell 的前提下新增第七条证据线 **`runtime_attachment_evidence`**（附着验收只读转录 + digest 校验），并把两项附着门机械翻转为 `true`。执行顺序以 [03_CURRENT_EXECUTION.md](../active/roadmap/03_CURRENT_EXECUTION.md) 为准。

## 1. 目的与唯一变更

- **唯一变更**：新增 `runtime_attachment_evidence`（来源 = 附着验收报告 + 预注册文档 sha256，只读转录）；全部证据线机械复用 v7 reducer（`build_v7` 原样调用），v7 source digest 逐位校验零漂移。
- **唯一门翻转（2 项）**：`default_runtime_owner_attached`、`learning_mechanism_attached_default_runtime`：`false → true`。
- **机器边界闭环**：v7 合同 §4 冻结的映射——两项附着门翻转后五项晋级 gate 全 true，`promotion_gate` 与 `can_promote` 机械置 true。**`can_promote=true` ≠ 已晋级**：最终晋级宣布仍须独立批准，且不得被读作完整认知能力或结构成长主张（`growth_admitted=false` 贯穿全部实验报告）。

## 2. 固定输入与内容寻址

- K1/K2/K3、C 阶段、v2–v6 scorecard、P4.12/P4.13/P4.14、rollout review、attachment manifest、A8 评审合同、SGK v1 合同：与 v7 完全相同（v7 报告为 digest 一致性校验基准）；
- **默认 runtime 附着验收报告**：`reports/taiji_m5_k_runtime_attachment_20260912.json`（校验 `format=taiji-m5-k-runtime-attachment-v1`、`status=completed`、`outcome=runtime_attachment_supported`、`experiment_passed=true`、`can_promote=false`、`growth_admitted=false`、`fit_called=false`、4/4 cell `passes_all`、机械/消费/非干扰三组 gate 全 true、三个拒绝探针全过）；
- **附着预注册文档**：`plans/reference/M5_K_DEFAULT_RUNTIME_ATTACHMENT_PREREGISTRATION_20260912.md`（sha256 记录入证据线）。

## 3. runtime_attachment_evidence 字段（只转录，不重算）

- `source_report` / `source_preregistration` / `source_preregistration_sha256` / `outcome`；
- `attachment_design`：`api/taiji_runtime_attachment.py`（runtime 自有 fail-closed 消费合同、零研究脚本 import）+ `SeedRuntime.attach_k_g_state/detach_k_g_state`（显式 opt-in、原子拒绝、status 面）+ typed 读出面（`k1_predict`/`k2_predict`/`g_select`）；
- `consumption_equivalence`：Phase K（novel 2/2、旧类 4/4、安全 6/6）与 Phase G（`0.8/0.8675/0sv`、sibling `1.0/1.0`）在全部 4 cell 经 runtime 附着 consumer 逐字段复现；detach/re-attach 行为逐数值等价；
- `fail_closed_summary`：篡改 digest / 缺文件 / 错 cell 混装全部拒绝，拒绝后 runtime 保持未附着且原因入账；
- `non_interference`：`checkpoints/seed_corpus.pt` sha256 不变、model tick/参数计数不变、默认 chat/workbench 行为零变化（opt-in 附着不进入默认路径）；
- 资源绝对预算全过（attach 1.3s ≪ 60s/cell、总 266s ≪ 600s）；`fit_called=false`、`training_performed=false`；
- `runtime_owners_attached=true`（机器结论）。

## 4. promotion gate（v7 全部保留 + 两项翻转 → 全 true）

`promotion_gate = all(promotion_gates) = true`、`can_promote = true`（机械置位，v7 合同 §4 冻结映射）。**边界（冻结）**：最终晋级宣布须独立批准；批准文本必须保留以下诚实边界——(a) 五类合成课程仍是唯一实验载体；(b) 结构成长未被触发（`growth_admitted=false`），A8 增长子句为「未被触发」而非「已满足/已跳过」；(c) 附着是显式 opt-in、进程内内存态，默认任务解释路径不变（默认路径采用属独立决策）；(d) S 仍为架构性 control-only evidence。

## 5. 不做的事

- 不重训、重跑、不触碰任何 frozen 报告语义；本 v8 不宣布晋级、不改产品代码、不做默认行为切换；
- 不修改 v7 报告与更早历史产物；附着预注册文档只补 §7 执行记录。

## 6. 产物与唯一后续动作

- auditor：`scripts/training/audit_taiji_m5_k_axis_scorecard_v8.py`；
- report：`reports/taiji_m5_k_axis_scorecard_v8_20260912.json`；
- 路线图同步 + 独立提交；v7 报告保留为历史、不可覆盖；
- 唯一后续动作：**最终晋级宣布的独立批准**（机器边界已闭环；批准与否均落盘并同步唯一计划；批准前不做默认行为路径切换、不训练、不读取 sealed）。

## 7. 执行记录（2026-09-12，v8 已运行）

1. 机械纪律：`py_compile` / `ruff` / `black` / `mypy --follow-imports=silent` 通过后运行 auditor；v7 source digests 重算后与 v7 报告逐位一致（漂移校验零触发）；翻转前断言两项附着门在 v7 中确为 false（转录防呆）。
2. 报告 `reports/taiji_m5_k_axis_scorecard_v8_20260912.json`：10 份 source digests（v7 全部 + `RUNTIME_ATTACHMENT`）；`runtime_attachment_evidence` 完整转录附着验收。
3. 门翻转核验：v7→v8 恰好两项翻转；13 项 gate 全 true；`promotion_gate=true`、`can_promote=true`（机械）。
4. v7 报告与全部历史产物未覆盖；默认行为路径零改动、无训练、无 sealed 读取。
