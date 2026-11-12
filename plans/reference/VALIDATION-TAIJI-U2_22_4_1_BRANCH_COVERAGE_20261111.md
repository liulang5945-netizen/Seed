# 分支覆盖台账：22.4.1 执行器（`scripts/training/run_taiji_u2_22_4_1.py`）

**建立时间**：2026-11-11｜**对应自检**：`python -m scripts.training.run_taiji_u2_22_4_1 --self-check`（**62 条**，`optimizer_step_calls=0`，cpu 7.48 s／wall 19.86 s；臂参数化后由 36 条增至 43 条，u2 臂行回填后增至 46 条，u2 执行图改为基线并落 F2 逐字项后增至 51 条，u2 一步执行路径接入后增至 56 条，u2 的 R05 父身份口径接入后增至 61 条，F-01 改实后补 1 条 R06 正向见证增至 62 条，见 D-20261111-18／-19／-20／-21／-23／-25）。
**规则**：执行器每新增一个可退出分支，必须在本表加一行；只加代码不加账视为未完成。标"未覆盖"的不得写成"已验证"。

## 1. 退出码 5（授权、身份、合同）

| # | 分支 | 覆盖 | 覆盖者／未覆盖原因与风险 |
|---|---|---|---|
| 1 | R01 `update_id` 与 `attempt_id` 不自洽，或不是本 run 所引用执行 manifest 的臂（不再比对常量） | 已覆盖 | `update_id_mismatch`（拉臂绑定分支）；"自洽但 manifest 无此臂"子分支见 #49 |
| 2 | R01 `run_id` 与目录名不一致 | **未覆盖** | 需真跑（`execute` 拿 `root.name` 比对）；触发：run_root 被改名/复制。风险：低——只是命名校验，误放过也不放行步进 |
| 3 | R02 `authorized!=true` | 已覆盖 | `authorization_missing` |
| 4 | R02 `owner_approval_ref` 为空或不含本 run 的 `update_id`（取自 run.json） | **未覆盖**（同函数相邻分支） | 触发：手改草稿抹掉绑定段。风险：中——授权凭据失效却仍可开跑 |
| 5 | R03 执行 manifest 哈希不符 | 已覆盖 | `manifest_hash_tampered` |
| 6 | R04 准入 manifest 哈希不符 | 已覆盖 | `admission_hash_tampered` |
| 7 | R05 accepted parent 状态/profile/seed/精度/参数数不符 | **未覆盖** | 需篡改 `taiji_u2_22_3_accepted_parent_20261111.json`（冻结件，本卡不动）。真实运行走过通过分支 |
| 8 | R05 重建基座与 accepted 身份不符 | 已覆盖（隐式） | 真实运行 R05 通过；自检里基座摘要由 `fork_evidence` 现算，未注入错误摘要 |
| 9 | R05 接口回执 verdict／记过的 pin 不一致 | **未覆盖** | 触发：回执被改。风险：中——pin 漂移会被磁盘 pin 校验兜住 |
| 10 | R06 `cpu_cap>120`、`wall_cap>240`、精度不符、`steps!=1` | 已覆盖 | 四条独立用例，全部用**内存副本**触发、不碰冻结草稿：`r06_cpu_cap_over_price`、`r06_wall_cap_over_price`、`r06_precision_mismatch`、`r06_optimizer_steps_not_one`；每条退 3 且见证账本字节不变、`consumed=0` |
| 11 | R06 账本已有 attempt | 已覆盖 | `attempt_already_consumed_r06` |
| 12 | R07 六个 false 标志被改 | 已覆盖 | `data_scope_flag_mutated` |
| 13 | R07 分区不是 `engineering_slice` | **未覆盖**（同函数相邻分支） | 触发：改草稿 |
| 14 | R07 digest 为空 | **未覆盖**（同函数相邻分支） | 触发：开跑前未写 digest |
| 15 | R07 digest 与生成片段不符 | 已覆盖 | `data_digest_mismatch` |
| 16 | R08 Adam 超参不符 | 已覆盖 | `optimizer_hyperparameter_changed` |
| 17 | R08 默认臂（u1）权重不是 F1 | **未覆盖** | 触发：改执行 manifest。风险：中——权重改动会静默改变"学什么"。非默认臂的对应分支见 #53 |
| 18 | R08 warmup／clip 被打开 | **未覆盖** | 同上 |
| 19 | R08 声明的活性参数不在模型里 | 已覆盖 | `declared_active_parameter_absent` |
| 20 | R08 群单元扩展改变了活性清单 | **不可达** | 只有 `group_units != None` 才进入；本卡模型固定无群单元 |

## 2. 退出码 3（账本、额度、worker、入口）

| # | 分支 | 覆盖 | 覆盖者／未覆盖原因与风险 |
|---|---|---|---|
| 21 | 账本首行不是 run_start | **未覆盖** | 触发：删首行。`read` 会先报"authorization header missing" |
| 22 | run_start 不唯一 | 已覆盖（`read` 层） | `ledger_run_start_not_unique`：注入第二条 run_start 被状态机拒绝（Unexpected ledger event，退 3）。R10 自己的"single line"文案分支因此**不可达** |
| 23 | 哈希链断裂／行摘要不符 | **未覆盖** | 触发：手改任意一行。风险：高——这是账本的根信任，只靠真实账本的独立复算兜底（本轮 4 行已复算） |
| 24 | attempt 已消耗（open 二次） | 已覆盖 | `attempt_already_consumed` |
| 25 | 额度将超 `budget_max` | 已覆盖 | `budget_max_zero_gate`：假 precontract 把 `budget_max` 压到 0，`open` 退 3，且账本**字节不变**（没有多出计费行） |
| 26 | 截断尾 `TailTruncated` | 已覆盖 | `tail_truncated_detected` |
| 27 | `repair_tail` 修复截断尾 | 已覆盖 | `repair_tail_restores_chain` |
| 28 | `reconcile` 处理 open 无 close | 已覆盖 | `reconcile_open_without_close` |
| 29 | 账本 run／source identity 不符 | 已覆盖 | `ledger_run_identity_mismatch` |
| 30 | open 后任意异常收尾（通用） | 已覆盖 | `generic_exception_after_open` |
| 31 | R11 读回失败收尾（退 4） | 已覆盖 | `r11_readback_failure_closeout` |
| 32 | R12 活性门失败收尾（退 6） | **未覆盖**（`reject_attempt` 的 exit 6 分支） | 活性门本身由 `activity_gate_not_live` 覆盖，但那次是自检手工 close；`perform` 抛 `ActivityGateError` 走 `reject_attempt` 的路径未走。风险：中 |
| 33 | 独立 worker 未通过 | 已覆盖（判定函数） | `worker_failed_exit3` + `worker_failure_does_not_reclose`；子进程本身（超时 180 s、返回码语义）未测 |
| 34 | 同一 run_root 被第二个 run 复用 | 已覆盖 | `run_root_reused_by_second_run`（本轮新增 `bind_run` 守卫） |
| 35 | `--run` 缺 `--run-root` | 已覆盖 | `cli_run_without_root` |
| 36 | `main` 捕获非账本异常退 3 | **未覆盖** | 触发：未预料的代码错误。风险：低——只影响退出码呈现 |
| 37 | `within_cap=false`（真超帽） | **未覆盖** | 需真跑；触发：CPU>120 或 wall>240 |

## 3. 退出码 4／6／0

| # | 分支 | 覆盖 | 覆盖者 |
|---|---|---|---|
| 38 | R09 父 checkpoint 读回不符 | 已覆盖 | `parent_readback_mismatch`（经 `readback_or_exit`） |
| 39 | R09 父 checkpoint 不存在 | **未覆盖** | 触发：run_root 被清空 |
| 40 | R12 活性回执为空行 | **未覆盖** | 触发：执行 manifest 活性清单为空 |
| 41 | 成功路径（回执→读回→close） | 已覆盖 | `accepted_path_order`（假优化器、不消耗额度） |
| 42 | 重复 close | 已覆盖 | `duplicate_close_rejected` |
| 43 | 全程不得出现 `optimizer.step` | 已覆盖 | `no_optimizer_step` |
| 44 | 回执守卫：close 行没有 `s12_sha256` | 已覆盖 | `close_missing_s12_sha256`（`close_receipt_guard` 退 3） |
| 45 | 回执守卫：回执里没有实测 `cpu/wall` | 已覆盖 | `s12_without_cpu_wall`（退 3；空值、布尔、字符串都算缺） |
| 46 | rejected 路径的回执顺序 | 已覆盖 | `rejected_s12_precedes_close`：S12 落盘→读回→close，close 带 `s12_sha256`，回执内 `ledger_seq_close=null` |
| 47 | rejected 路径回执根本写不出来（降级 close） | **未覆盖** | 需让 `write_once` 失败（磁盘满／权限）；触发：写盘故障。风险：低——降级时 close 行记 `s12_error`，账本仍结算，不悬空 |
| 48 | R01 `update_id` 为空／非字符串 | 已覆盖 | `update_id_mismatch` 与 u2 草稿用例共用同一判据；空串分支由 R01 的类型检查拦住 |
| 49 | R01 `update_id` 自洽（==`attempt_id`）但不是本 run 所引用执行 manifest 的臂 | **未覆盖** | 触发：伪造自洽但 manifest 无此臂的 run.json。风险：低——R03/R04 哈希与 R06 准入行仍会拦 |
| 50 | R01 `run_id=null`（未授权草稿形态） | 已覆盖 | `u2_draft_r01_rejects_null_run_id`（u2 草稿原样，退 5） |
| 51 | R02 未授权草稿形态（`authorized=false`、`owner_approval_ref=null`） | 已覆盖 | `u2_draft_r02_rejects_missing_authorization`（退 5） |
| 52 | R01–R05 对 u2 臂全通过（仅在内存绑定 run_id／授权／额度，不落盘） | 已覆盖 | `u2_bound_r01_identity_binds_arm`、`u2_bound_r02_authorization_binds_arm`、`u2_bound_r03_execution_manifest_sha`、`u2_bound_r04_admission_and_identity_sha`、`u2_bound_r05_frozen_identity`（R05 走 `check_r05_baseline`，逐项返回 10 项读数） |
| 53 | R08 非默认臂（u2）权重与臂声明的 `weight_profile` 漂移 | 已覆盖 | `u2_r08_weight_drift_rejected`（内存副本把 `verbatim` 改成 0.5，仍留 `F2-style` 名 → 退 5）；**该分支原为同义反复**（`weights != dict(row["weights"])` 恒假），D-20261111-19 改为对照冻结表 `F1-style`／`F2-style` 后才可覆盖 |
| 54 | R08 非默认臂声明未知 `weight_profile` | 已覆盖 | `u2_r08_unknown_profile_rejected`（内存副本把 profile 改成 `F9-style` → 退 5） |
| 55 | u2 执行图构造：非默认臂不再走 U2 元数据候选，改按身份清单重建基线并核对 21,977/35＋参数清单摘要 | 已覆盖 | `u2_execution_graph_is_frozen_baseline`（`transformer_sequence` 21,977/35，`inventory=8c041a19…4569`） |
| 56 | u2 角色槽缺位：`predictive_head`／`recurrent_core` 在该图不存在，照实标并跳过，不用 U2 参数冒充 | 已覆盖 | `u2_role_slots_backfilled_from_graph`、`u2_no_u2_parameter_names_on_arm` |
| 57 | u2 F2 verbatim 项：按臂取片段、`byte_head` 逐字教师强制（非零），且全参数有梯度 | 已覆盖 | `u2_verbatim_term_positive`（实测 5.5150）、`u2_f2_gradients_alive`（35 项无零梯度） |
| 58 | u2 一步执行拒绝：基线尚未接入 `perform`／checkpoint／迁移，非默认臂在 R08 后退 5 | **已删除**（D-20261111-21） | 该分支随一步执行路径接入而移除，由 #64 端到端承接；原先“未覆盖”的风险不再适用（留痕行，不重复计数） |
| 59 | 基线 checkpoint 写盘往返：`save_baseline` 写 `parent.bin` 后立即用新实例读回五项（`state_sha256`／`parameter_sha256`／`moment_sha256`／`history_reference`／`revision`） | 已覆盖 | `u2_baseline_parent_checkpoint_round_trip`（revision=0、99,958 字节） |
| 60 | 基线一步前必须账本 open：`StepCounter` 未 open 时抛 `LedgerError` 退 3，且不产生任何 step | 已覆盖 | `u2_baseline_step_requires_open_attempt`（step calls 仍 0） |
| 61 | 基线首步位移符合 Adam 零动量首步公式：`max|Δ|≤lr`，且 `|g|≥1e3·eps` 处 `|Δ|≈lr` | 已覆盖 | `u2_baseline_first_step_displacement_is_lr`（max 1.000000e-3、shortfall 9.922e-7、elements 21,977、below_eps 1,392） |
| 62 | 基线 policy 为结构空、`next_output` 为 `byte_head` 257 类 | 已覆盖 | `u2_baseline_policy_is_structural_zero` |
| 63 | 基线 rejected 回执形状（`status=rejected`、`execution_graph.param_total=21,977`、`policy=[]`） | 已覆盖 | `u2_baseline_rejected_result_shape` |
| 64 | u2 全链路：`check_r09_baseline`→`perform_baseline`（梯度→Adam 单步→35 项活性回执→写盘读回）→`check_r11_baseline`→`s12_payload`→worker `verify_baseline` 独立复算 | **未覆盖**（自检只覆盖分段 #59–#63；端到端由 evidence-only 一次性证据在自检外复算） | 分段覆盖见 #59–#63。完整链路由 `scripts/training/evidence_taiji_u2_end_to_end_22_4_1.py` 走真账本（`run_start→open→close`，每次 append fsync）跑通一次：35/35 参数更新、worker `verdict=pass`、`adam_equations=35`，读数见 `reports/taiji_u2_22_4_1_end_to_end_evidence_20261112.json`（D-20261112-24）。它**不入自检**（自检须 0 步），也不改本行"未覆盖"判定；create-once 拒绝、schema/arm 不符等子分支仍随通用异常路径兜底 |
| 65 | R05-u2 身份清单哈希与 run／臂 pin 不一致 | 已覆盖 | `u2_r05_identity_hash_drift_rejected`（内存副本改 `references.baseline_identity_sha256` → `ContractError: R05 baseline identity hash does not match the run/arm pin`，退 5） |
| 66 | R05-u2 父带训练祖先（`training_runs`／`optimizer_steps` 非 0） | 已覆盖 | `u2_r05_training_ancestor_rejected`（内存副本把 `conditions.training_runs` 改成 1 → `R05 baseline parent is not a training-free initialization`，退 5） |
| 67 | R05-u2 构造件模块源码漂移 | 已覆盖 | `u2_r05_module_pin_drift_rejected`（内存副本改 `architecture.module_sha256` → `R05 frozen baseline module source changed`，退 5） |
| 68 | R05-u2 测量报告未选中本图或选择路径不是 `tie_break`／L1 | 已覆盖 | `u2_r05_measurement_not_selected_rejected`（内存副本改 `selection.selected` → `R05 identity measurement selection differs from the identity`，退 5） |
| 69 | R05-u2 线程数：只记录、不判假（`threads≠1` 不得拒绝） | 已覆盖 | `u2_r05_threads_recorded_not_judged`（实测 `threads=4` 仍通过，读数原样记录） |
| 70 | R06 正向：准入 u2 行 `precision` 与 run 逐字一致时**放行**（F-20261112-E2E-01 改实的见证） | 已覆盖 | `u2_bound_r06_real_precision_accepted`（读真实准入文件与 u2 绑定副本，`consumed=0`，pass；判据代码未动） |

### 3a. 2026-11-12启动与零步恢复补充

| # | 分支 | 覆盖 | 覆盖者／未覆盖原因与风险 |
|---|---|---|---|
| 71 | 显式合同创建并再次绑定同一run | 已覆盖 | `test_selected_contract_binds_new_run` |
| 72 | 显式合同遇到其他run_id拒绝复用根 | 已覆盖 | `test_selected_contract_rejects_reused_root`，原run字节保留 |
| 73 | 默认绑定保持u1草稿行为 | 已覆盖 | `test_default_binding_preserves_u1` |
| 74 | CLI将`--run-contract`传递到执行器 | 已覆盖 | `test_cli_passes_selected_contract`及正式u2运行 |
| 75 | 无迁移条件核对元数据图与实际宽度 | 已覆盖 | `test_dense_metadata_graph_accepts_no_migration`、`test_original_graph_cannot_substitute_metadata_graph`及u3/u4实际恢复 |
| 76 | 梯度比较包含unused与零参数 | 已覆盖 | `test_comparison_includes_unused_and_zero_parameters`及776项实际比较 |
| 77 | 单参数微小梯度漂移被拒绝 | 已覆盖 | `test_comparison_rejects_small_parameter_credit_change` |
| 78 | source pin路径缺失于open前退出 | 已覆盖 | `no_migration_preopen_failure_20261112`保留v1失败，v2全部16项source pin通过 |
| 79 | u3/u4两名额按顺序关闭、独立进程接续一致 | 已覆盖 | `no_migration_summary_20261112`，5行账本、0 optimizer step、194×4逐参数比较 |
| 80 | CLI仅给`--run-contract`而缺`--run` | 未覆盖 | 新增argparse退出分支，风险低；启动参数错误时在模型与账本之前退出 |

本轮相关回归124项通过，原未覆盖18行与新增#80仍明确登记，合计19行。源码路径失败发生在计费前。u2与u3/u4真实结果及历史u1限制归[22.4.1结案](VALIDATION-TAIJI-U2_22_4_1_CLOSEOUT_20261112.md)。

## 4. 未覆盖项的合计风险

高风险未覆盖项只剩"账本根信任"#23（手改任意一行）：只能靠独立复算兜底，注入断链会污染真账本，故不在自检内做。原 #10（R06 帽/精度/步数）与 #25（额度闸门）均已覆盖（#25 与 #10 分别见 D-20261111-14／D-20261111-15）。臂参数化（D-20261111-18）新增 #48–#53 六行、臂行回填（D-20261111-19）新增 #54、执行图改基线（D-20261111-20）新增 #55–#58 四行、一步执行路径接入（D-20261111-21）新增 #59–#64 六行并删除 #58 的分支（留痕不计数）、R05-u2 父身份口径接入（D-20261112-23）新增 #65–#69 五行。当前共 **69 行**（含 1 行已删除留痕），未覆盖 **18 行**、另有 **2 行不可达**（#20 群单元扩展、#22 中 R10 自身文案分支），全部是"要真跑、要造故障或要篡改账本"才能触发的分支。

**u2 臂的现状（R05 已按预注册改查执行图身份，C09 待 owner 放行）**：自检把 `u2_binding_preflight` 写进 `selfcheck.json`——执行图 `transformer_sequence` **21,977 标量／35 张量**（`inventory=8c041a19…4569`，与身份清单逐项一致）；`dtype_field="float32"`、`active_parameters_count=35`、`active_set_equals_graph_inventory=true`、`missing_in_model=[]`、`roles_match_draft=true`、`dead_gradient_parameters=[]`、`executable=true`、`step_path_implemented=true`。三槽：`output_head`=`byte_head.weight/bias`（2 项），`predictive_head`／`recurrent_core` **该角色不存在**（原因已写入臂行）。F2 实测：predictive 5.3683、verbatim 5.5150、actor/critic 为结构零。**仍记着三条 `semantic_gaps`**：actor/critic 在该图上没有对应头（结构零，已声明）；u2 的父是冻结初始化、R05 只签"更新能落地／状态能往返／独立进程能复现"（不签续训，见预注册与 #65–#69）；policy/state 为结构空（`policy=[]`、`state_tensors=0`，迁移记 `not_applicable_baseline_has_no_subject_state`，不伪造迁移）。

**守卫与现存证据的差距（必读）**：`close_receipt_guard` 是**向前生效**的。已落盘的那条 accepted close（`reports/taiji_u2_22_4_1_attempts_20261111T142919Z.ndjson` 第 2 行）**没有** `s12_sha256`，对应的补算 S12 也**没有** `cpu_seconds`/`wall_seconds`（两个字段为 null）——它是在守卫和顺序契约存在之前写下的。不能用新守卫反判旧证据，也不能改写旧行（改写即破坏哈希链）；差距记在 `DECISION-LEDGER` 的 D-20261111-14，等下一次真实 attempt 用新代码自然补齐。

## 5. 维护约定

- 新增分支 → 同一提交内更新本表，写清覆盖者或未覆盖原因。
- 自检用例改名或删除 → 本表对应行必须同步，否则视为失效覆盖。
- 本表不替代 `DECISION-LEDGER`；额度与结论性判断记在台账。
