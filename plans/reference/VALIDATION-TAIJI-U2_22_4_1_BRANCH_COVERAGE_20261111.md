# 分支覆盖台账：22.4.1 执行器（`scripts/training/run_taiji_u2_22_4_1.py`）

**建立时间**：2026-11-11｜**对应自检**：`python -m scripts.training.run_taiji_u2_22_4_1 --self-check`（32 条，`optimizer_step_calls=0`，cpu 4.92 s／wall 5.80 s）。
**规则**：执行器每新增一个可退出分支，必须在本表加一行；只加代码不加账视为未完成。标"未覆盖"的不得写成"已验证"。

## 1. 退出码 5（授权、身份、合同）

| # | 分支 | 覆盖 | 覆盖者／未覆盖原因与风险 |
|---|---|---|---|
| 1 | R01 `update_id` 不是 u1-candidate-float32 | 已覆盖 | `update_id_mismatch` |
| 2 | R01 `run_id` 与目录名不一致 | **未覆盖** | 需真跑（`execute` 拿 `root.name` 比对）；触发：run_root 被改名/复制。风险：低——只是命名校验，误放过也不放行步进 |
| 3 | R02 `authorized!=true` | 已覆盖 | `authorization_missing` |
| 4 | R02 `owner_approval_ref` 为空或不含 update id | **未覆盖**（同函数相邻分支） | 触发：手改草稿抹掉绑定段。风险：中——授权凭据失效却仍可开跑 |
| 5 | R03 执行 manifest 哈希不符 | 已覆盖 | `manifest_hash_tampered` |
| 6 | R04 准入 manifest 哈希不符 | 已覆盖 | `admission_hash_tampered` |
| 7 | R05 accepted parent 状态/profile/seed/精度/参数数不符 | **未覆盖** | 需篡改 `taiji_u2_22_3_accepted_parent_20261111.json`（冻结件，本卡不动）。真实运行走过通过分支 |
| 8 | R05 重建基座与 accepted 身份不符 | 已覆盖（隐式） | 真实运行 R05 通过；自检里基座摘要由 `fork_evidence` 现算，未注入错误摘要 |
| 9 | R05 接口回执 verdict／记过的 pin 不一致 | **未覆盖** | 触发：回执被改。风险：中——pin 漂移会被磁盘 pin 校验兜住 |
| 10 | R06 `cpu_cap>120`、`wall_cap>240`、精度不符、`steps!=1` | **未覆盖** | 触发：改草稿。风险：高——帽是资源闸门，放行等于授权超支 |
| 11 | R06 账本已有 attempt | 已覆盖 | `attempt_already_consumed_r06` |
| 12 | R07 六个 false 标志被改 | 已覆盖 | `data_scope_flag_mutated` |
| 13 | R07 分区不是 `engineering_slice` | **未覆盖**（同函数相邻分支） | 触发：改草稿 |
| 14 | R07 digest 为空 | **未覆盖**（同函数相邻分支） | 触发：开跑前未写 digest |
| 15 | R07 digest 与生成片段不符 | 已覆盖 | `data_digest_mismatch` |
| 16 | R08 Adam 超参不符 | 已覆盖 | `optimizer_hyperparameter_changed` |
| 17 | R08 u1 权重不是 F1 | **未覆盖** | 触发：改执行 manifest。风险：中——权重改动会静默改变"学什么" |
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

## 4. 未覆盖项的合计风险

高风险未覆盖项只剩"账本根信任"#23（手改任意一行）：只能靠独立复算兜底，注入断链会污染真账本，故不在自检内做。原 #10（R06 帽/精度/步数）与 #25（额度闸门）中，#25 本轮已覆盖；#10 仍未覆盖——它要改的是**冻结草稿里的执行参数**，改草稿本身就会污染准入物，故登记为未覆盖，风险由真实运行的 `cpu_cap`/`within_cap` 兜最后一道。

**守卫与现存证据的差距（必读）**：`close_receipt_guard` 是**向前生效**的。已落盘的那条 accepted close（`reports/taiji_u2_22_4_1_attempts_20261111T142919Z.ndjson` 第 2 行）**没有** `s12_sha256`，对应的补算 S12 也**没有** `cpu_seconds`/`wall_seconds`（两个字段为 null）——它是在守卫和顺序契约存在之前写下的。不能用新守卫反判旧证据，也不能改写旧行（改写即破坏哈希链）；差距记在 `DECISION-LEDGER` 的 D-20261111-14，等下一次真实 attempt 用新代码自然补齐。

## 5. 维护约定

- 新增分支 → 同一提交内更新本表，写清覆盖者或未覆盖原因。
- 自检用例改名或删除 → 本表对应行必须同步，否则视为失效覆盖。
- 本表不替代 `DECISION-LEDGER`；额度与结论性判断记在台账。
