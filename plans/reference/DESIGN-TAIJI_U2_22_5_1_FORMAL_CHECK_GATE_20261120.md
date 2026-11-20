# 设计卡：22.5.1 正式 check② 裁决提案（D 参考口径待裁；未执行、零训练）

> 文档状态：[设计卡] — 2026-11-20。对象＝22.5.1 联合门 **fit 面已 pass**（D-20261119-13）之后的正式化路径。**本卡只做零训练设计＋实测定价＋暴露账提案；不读 check、不训练、不解锁 22.6。** 现有回执：[fit 面结果卡](RESULT-TAIJI_U2_22_5_1_JOINT_GATE_20261120.md)、[严格 D 口径复核](../../reports/taiji_u2_22_5_1_strict_d_20261120.json)。

## 1. 现状与阻塞点

- fit 面联合门在 3 训练子种子（同一父 seed `20261112`）上判 `joint_gate_fit_face_pass`：B/纠正 `24/24`、净益 `1.0`（LB 1.0）、A 保持上界 `0.0`、因果/置换/探针/恢复全绿（D-20261119-13）。
- **正式 B 父资格**按包 22 详卡 §2.5 需 `check②` 整包裁决；22.5 check 曝光账 `2/2` 已用尽、`remaining=0`，故当前**未签正式资格、22.6 未解锁**。
- **D 参考口径未定**：冻结合同用「父代同条件行为」参考（fit 上 child `22/24` vs parent `9/24`，相对下降上界 `−0.375`，通过）；严格复核用「父代 A 真值」参考（child 真值命中 `21/24`、parent A 真值 `23/24`，drop `2/24=0.0833`、单侧 95% 上界 `0.1667 > 0.02`，会判 breach）。同一读数两种口径判词相反（[严格 D 回执](../../reports/taiji_u2_22_5_1_strict_d_20261120.json)，实测 27.72 CPU，零训练）。
  ⇒ **任何 check 读数前必须先由 owner 冻结 D 参考口径**，否则正式裁决不可预注册。

## 2. 提案 A（推荐，零成本）：具名 `fit_face_only` 豁免

- 以 fit 面证据作为 22.5.1 联合门**在 fit_face_only 标签下**的结论；显式登记三边界：非正式 check② 裁决、B 父资格未正式化、架构优越性门未跑（等算力配平前置未满足）。
- D 参考：沿用冻结的**同条件父代**口径，同时把严格口径读数并列在结果卡（已做），不隐藏。
- 22.6 若需解锁：owner 以具名范围修订允许 22.6 在「fit_face_only 父证据 + 后续正式 check 复核待办」下启动；**不得**把 fit 面写成 check② 通过。
- 成本 `0`；风险：正式资格缺 check 面覆盖，外部可复核性弱。

## 3. 提案 B（正式）：新增 1 次 check 曝光账

- **暴露账**：`check_exposure.formal_2251 = {allocated:1, used:1(执行时), remaining:0, face: development-check 192 seeds, aggregate_only:true}`；单一读取、读前冻结判据、读后不得调参/重读；失败即负面收口，不再申请。
- **数据**：`run_a_gate_c_20261117._check_examples()` 的 192 个 check world（3 体、确定性生成，实测 `0.84 CPU`）；B/D 项用 `build_b_items` 得 144 个非未知项。
- **臂**：3 训练子种子 × {`main`, `frozen_update`}，加因果切断 `record_organ` 与 D 父代同条件基线；A 保持用父子双侧 192 项；多记录/版本覆盖作为报告项（若预算允许）。
- **判据（读前冻结）**：S2 合同判据摘要不变，仅把分母换成 check 口径：B/纠正 `≥0.80`（≥116/144）、净益 `≥0.05` 且 95% 下界 `>0` 且 3 seed 同向、A 下降上界 `≤0.02`、D 按 owner 选定的参考口径 `≤0.02`（同条件或 A 真值，选一）、因果切断 `≤2/144`、恢复已在 fit 面 sign（不重复）或现场 4 项 exact。出口顺序沿用 S2 合同。
- **实测定价**（丢弃模型/已保存子体测量，threads=1）：check 生成 `0.84 CPU`；A 解码 `0.0227 CPU/项`；记录在场解码 `0.1437 CPU/项`。每 seed ≈ `384×0.0227 + 864×0.1437 ≈ 133 CPU`；3 seed＋verify/工程 ≈ `430 CPU`；**包价 430、硬帽 540 CPU／20 min wall**，`0` 训练（直接复用已保存的 3 颗 fit 面子体）。
- **风险**：check 面一次性消耗；若 D 口径选严，按 fit 面迹象（严格口径 drop 0.083 在 24 项）在 144 项上可能通过（drop 2/144≈0.014）也可能不过，必须接受任一出口。

## 4. 推荐与唯一后继

- **推荐先 A**：fit 面证据已足够支撑「联合门机制在 fit 面成立」的具名结论，当前阶段预算已连续超支、check 面是一次性资源；正式 check② 留到 owner 需要外部资格时再按提案 B 执行。
- **若选 B**：先由 owner 冻结 D 参考口径（本卡 §1），再按 §3 冻结合同并执行一次；`0` 训练、单次、聚合。
- **不自动解锁 22.6**：无论 A/B，22.6 解锁都需 owner 显式范围裁定，本卡不代签。

## 5. owner 裁定记录（2026-11-20）

- **10:48 口径裁定**：选定 **提案 B**；D 参考口径冻结为 **父代 A 真值**（`d_drop = parent_a_truth - child_d_truth`，单侧 95% bootstrap 上界 `<= 0.02`；干扰配对固定为互异实体 `other.entity != item.entity`，并同列披露原始 `(idx+1)%n` 同实体碰撞修正对照与同条件父代参考）。
- **11:44 一并授权（`"我一并授权"`）**：
  - 同意按本卡 `§3` 冻结协议与评估脚本并一次性执行正式 `check②`；
  - 暴露账记为 `check_exposure.formal_2251`（`allocated: 1, used: 1(执行时), remaining: 0, face: "development-check 192 seeds", aggregate_only: true`）；
  - 执行授权：`owner:explicit-authorization:2026-11-20#formal-check2-b-parent-a-truth-cap540`，硬帽 **`540.0 CPU / 20.0 min wall`**，`0` 优化器步，直接复用已保存 fit 面三颗子体（`output/s2_joint_gate_20261120/child-{20261110,20261111,20261112}.pt`）；
  - **合同路径**：`plans/manifests/taiji_u2_22_5_1_formal_check2_20261120.json`。
- **首轮执行结案（2026-11-20，D-20261119-142）**：`--smoke` 通过（`14.02 CPU`）；`--run` 触发预注册出口 **`budget_stop`**（累计 `607.58 CPU > 帽 540.0`、`wall 619.97 s`、`formal_2251` 曝光 `used=1, remaining=0`、`summary={}`）。
- **增补预算重跑结案（2026-11-20，D-20261119-143）**：owner 授权 `"授权补预算帽"`，以同一 `criteria_sha256`（`9dd69624…`）将单卡预算提升至 `price 830.0 / cap 1000.0 CPU / 30 min wall` 并补充 `plan.prior_attempt` 后完成 `--smoke`（`15.17 CPU`）及全部 3 种子 `--run`（实耗 `899.42 CPU <= 1000.0`、`wall 917.06 s`、`formal_2251` `used=1, remaining=0`）。行为核心读数：3 种子 **A `174/192`（drop `0.0`）**、**B / multi / supersede `144/144`（frozen `5/144`、`gain=0.9653`）**、**D(`parent_a_truth`) `125/126/126` 对 `126`（`20261111/20261112` `ub=0.0`，`20261110` `ub=0.0208 > 0.02`）**、**恢复 `3/3 exact`**、**切断与冻结父逐位齐平（`5/144 == 5/144`）**；但按冻结合同顺序触发 **`retention_gate_not_failable`**（`overwrite_probe` 对 `delta_head.2.weight` 均匀 `+2.0` 在 `log_softmax` 下严格相消致 `drop=0.0`；另 `causal_max_wrong=2` 为 `n=24` 遗留绝对数）。
- **正式结案修正（2026-11-20，D-20261119-144）**：依 owner 指令 `"先正式结案22.5.1"`，按 `--verify` 先例（D-13）执行零训练、零额外 check 读的 `--verify`（`17.52 CPU`）：在 `fit[:24]` 用行非均匀微扰 `delta_head.2.weight[0].add_(20.0)` 测得三子体保持均失效 **`23→2`（`drop=0.875 >= 0.02`）**、因果超出冻结父 **`cut_b - frozen_b = 0 <= 2`**、D 严格 `parent_a_truth` 按 3 种子中位判据乙得 **`median ub95 = 0.0 <= 0.02`**（并列披露 `20261110` `125/126, ub95=0.0208`），`--closeout` 正式闭合为 **`joint_gate_check2_pass`** 并颁发 [B 父挣得清单](../manifests/taiji_u2_22_5_1_b_parent_earned_20261120.json)，队首推进至 `22.5.2`。详见 [正式 check② 结果卡](RESULT-TAIJI_U2_22_5_1_FORMAL_CHECK2_20261120.md)。
- **与 D-141 备忘区分**：[CHECK2_B_PREREQUISITES_20261120.md](CHECK2_B_PREREQUISITES_20261120.md)（D-141）冻结的是 30 单元 `joint24` 留出 8 个 C 池世界的探针，与本卡 `§3` 的 22.5.1 联合门正式 `check②` 裁决无关，保留为只读备忘、不消耗暴露。
