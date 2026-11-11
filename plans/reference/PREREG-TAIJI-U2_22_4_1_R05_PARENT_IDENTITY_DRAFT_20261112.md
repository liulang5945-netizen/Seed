# 预注册草案：u2 的 R05 父身份口径（22.4.1）

**状态**：`draft_unfrozen`——**未冻结、未获 owner 批准、未改代码**。本文件只定义“u2 的父是谁、R05 要检什么”，不改变 12 条退出码契约的编号、文本与退码，也不触碰 `check_r05`。
**关联**：[D-20261111-21](DECISION-LEDGER_20261109.md)（u2 一步执行路径接入；R05 口径缺口登记）、[PREREG-TAIJI-U2_22_4_4_BASELINE_IDENTITY_20261111.md](PREREG-TAIJI-U2_22_4_4_BASELINE_IDENTITY_20261111.md)（执行图身份）。
**冻结条件**：owner 书面批准 + 本文件规范化 sha256 写入 u2 `run.json` 的 `references`；在此之前任何实现都不得启用。

---

## 1. 父的精确定义（一句话）

> **u2 的父是 `plans/manifests/taiji_u2_22_4_4_baseline_identity_20261111.json` 这份 `status=frozen` 的身份清单所唯一确定的 `TransformerSequenceBaseline`（`architecture_id=transformer_sequence`，21,977 标量／35 张量，`param_inventory_sha256=8c041a19…4569`，模块 sha `384b4045…45cf`），按 `init.rule` 与 `init.seed=20261110` 在 CPU/float32 下重建，`training_runs=0`、`optimizer_steps=0`；它不是任何训练过程的产物，`parent.bin` 是本 run 内按 R09 首次写出的 `revision=0` 只读快照。**

拆成四个可判真假的要素：

| 要素 | 具体所指 |
| --- | --- |
| 谁 | `TransformerSequenceBaseline` 类（`scripts/training/baseline_identity_22_4_4.py`）实例化出的执行图，不是 `U2Subject`，也不是它的元数据候选 |
| 在哪 | 身份清单 `plans/manifests/taiji_u2_22_4_4_baseline_identity_20261111.json`；运行时物化在 `output/taiji_u2_22_4_1_<run_id>/parent.bin`（`revision=0`） |
| 由哪条规则构造 | `prepare_parent` 对非默认臂走 `baseline_identity_ref`：读身份清单 → 按 `architecture.module_path/class_name` 重建 → 按 `init.rule`（`normal_(std=0.3/sqrt(fan_in_per_row))`、bias=0.01、其余 torch 默认）与 `init.seed=20261110` 初始化 → 逐项核对参数量、张量数、参数清单摘要 |
| 包含哪些字段 | 身份字段：`schema`、`architecture_id`、`status`、`param_total`、`param_tensors`、`param_inventory`、`param_inventory_sha256`、`init`（含 `rule`/`seed`/`post_init_state_dict_sha256`/`second_construction_sha256`）、`architecture`（含 `module_path`/`module_sha256`/`class_name`/层宽）、`conditions`（`dtype`/`device`/`threads`/`torch_version`/`training_runs`/`optimizer_steps`）、`seeds`、`limitations`、`approval`；运行时五字段：`state_sha256`（执行图身份摘要）、`parameter_sha256`、`moment_sha256`、`history_reference`（`baseline:<identity_sha[:16]>`）、`revision` |

u1 的父**原样保留**：`plans/manifests/taiji_u2_22_3_accepted_parent_20261111.json`（`status=accepted_engineering_initialization`、191 参数、22.3 训练血脉）。两臂各查各的父，互不覆盖。

---

## 2. R05 检查项清单（u2 口径）

判假一律抛 `ContractError`，**退出码 5**（与现行 R05 相同）。检查顺序即下表顺序，任一失败立即返回。

| # | 检查项 | 判真条件 | 判假（可观察后果） |
| --- | --- | --- | --- |
| 1 | 父对象存在且可用 | 身份清单可读、可解析；`schema=="taiji-u2-22.4.4-baseline-identity-v1"`；`status=="frozen"` | 拒绝开跑，退 5；不建 run 目录、不写账本 |
| 2 | 父身份绑定 | 身份清单文件规范化 sha256 == `run.json.references.baseline_identity_sha256` | 退 5（防“换了清单再跑”） |
| 3 | 架构唯一 | `architecture_id=="transformer_sequence"`，且 `architecture.module_path/class_name` 与重建类一致 | 退 5（防“换图冒充”） |
| 4 | 规模与形状 | `param_total==21977`、`param_tensors==35`，且重建图逐项相等 | 退 5 |
| 5 | 参数清单一致 | 清单 `param_inventory_sha256==8c041a19…4569`，重建图按同规则（按 name 排序的 `[name, shape, dtype]` 规范 JSON）重算 == 该值 | 退 5 |
| 6 | 初始化可复现 | 本次重建的 `state_dict` 摘要 == 清单 `init.post_init_state_dict_sha256`；清单内 `post_init_state_dict_sha256==second_construction_sha256`（两次构造同值） | 退 5（防非确定性初始化） |
| 7 | 构造件未变 | `architecture.module_sha256==384b4045…45cf` == 磁盘 `scripts/training/baseline_identity_22_4_4.py` 的规范化 sha256 | 退 5（防“改了类再说同一张图”） |
| 8 | 无训练祖先 | 清单 `conditions.training_runs==0` 且 `optimizer_steps==0`；重建 runtime `revision==0` | 退 5（**u2 新增项**：父必须是纯初始化） |
| 9 | 测量回执通过 | `reports/taiji_u2_22_4_4_identity_measurement_20261111.json` 的规范化 sha256 == `run.json.references.measurement_report_sha256`；`verdict=="selected"`；`selection.selected=="transformer_sequence"`、`selection_path=="tie_break"`、`tie_break_level=="L1"` | 退 5（防“没选上却被当父”） |
| 10 | 源件未漂移 | 执行 manifest `authority.source_pins` 逐项 == 磁盘规范化 sha256；测量报告记录的 `execution_v1_sha256/execution_v2_sha256` 仅作历史记录（`matches`／`historical` 标注），**不参与判真假**——执行 manifest 的等值由 R03 独立承担 | 任一 source pin 漂移退 5；历史哈希不判假 |

**关于第 10 项的边界**：测量报告生成于身份清单追加 `limitations` 之前，其 `outputs.identity_sha256=820646c7…6f75`、`execution_v2_sha256=56017e2f…` 都是历史值（现行清单 `e5c776c4…5b29`、v2 `4fc2d042…4965`）。把历史值当等值条件会误杀合法重签；把它从判据里去掉又会让报告“无人核对”。故：报告文件哈希本身被钉死（第 9 项），其中数值只登记、不复核——这不是放宽，而是把复核责任交给钉死文件哈希与 R03。

---

## 3. 局限（原文，必须随任何 u2 读数引用）

> **u1 的父是 22.3 训练出的接受父：它有一条真实的训练血脉——191 个参数由 22.3 的优化过程产出，接受记录、接口回执与初始参数摘要共同证明“从哪个已训练状态继续”。u2 的父是 22.4.4 冻结的初始化：它没有任何训练祖先（`training_runs=0`、`optimizer_steps=0`、`revision=0`），因此 u2 的读数只签三件事——更新能在该图上落地、checkpoint 能写出并读回、独立进程能复现同样的梯度与回执；它不签“从已训练状态继续”，也不得被任何人引用为续训证据。任何把 u2 的单步结果当作“U2 主体被续训”或“训练可恢复”的推论，都超出本预注册的授权范围。**

补充两条同一性质的声明：

- u2 的 `parent.bin` 不是“检查点恢复”意义上的历史状态，只是本 run 内 `revision=0` 的只读快照；删掉它可以从身份清单无损重建，这一点与 u1 的父不同。
- u2 的 `migration` 记 `not_applicable_baseline_has_no_subject_state`：没有主体 state/history 可迁移，不伪造迁移成功。

---

## 4. 对照口径（两臂怎么比）

> **两臂的父不是同一个对象：u1 的父是训练产物，u2 的父是冻结初始化。因此两臂之间只比机制——更新是否落地（声明参数是否有位移）、checkpoint 是否可读回、是否只动了声明参数；不比损失数值、不比位移绝对值、不比下降幅度。u2 的 loss 从 6.0518 到 5.9826 只说明“梯度方向在该图上可走”，不是收敛或能力的证据。**

---

## 5. 为什么这是“改检查对象”而不是“放宽阈值”

证据一：**契约文本未变**。`run.json` 的 `run_contract` 里 R05 仍是“接受父存在；接口回执通过；source pins 与执行 manifest 一致（C01/C02）”，`fail_exit` 仍是 5，编号仍是 R05；12 条判据的编号、检查内容与退出码一个字未改（D-20261111-21 已记录）。

证据二：**实现方式与 R09/R11 同模式**。u1 的 `check_r05` 一行不动；冻结后新增独立的 `check_r05_baseline`，由 `execute()` 按 `isinstance(runtime, BaselineRuntime)` 分派——与 R09/R11 的基线实现完全同构。u1 路径逐字不变，其产物哈希不受影响。

证据三：**没有删项，反而多一项**。u1 的 10 类检查在 u2 上逐项有对应（对象从“22.3 接受父＋接口回执”换成“冻结身份清单＋身份测量报告”）；第 8 项“无训练祖先”是 u2 独有的**新增**检查，u1 不检、u2 必检。检查强度只增不减。

证据四：**“任意图”不能冒充父**。父被四重钉死：文件哈希（第 2 项）、架构与类（第 3 项）、参数量／张量数／参数清单摘要（第 4、5 项）、初始化摘要与构造件哈希（第 6、7 项）；再加“必须零训练”（第 8 项）与“必须是被选中的那一个”（第 9 项）。换一张图、改一次初始化、加一次训练，都会在开跑前退 5。

证据五：**阈值一个没动**。本草案不修改梯度门、变化门、`barely_live` 判据、Adam 超参、cpu/wall 帽，也不改 R12 的任何数值门限；只规定 R05 在 u2 上“看哪个对象”。

---

## 6. 冻结后的实施边界（本次不做）

1. 新增 `check_r05_baseline`（上述 10 项），复用 `baseline_identity` 重建，不复制 191 参数逻辑。
2. `execute()` 增加 R05 的按臂分派；u1 分支调用原 `check_r05` 不变。
3. 自检新增用例：身份清单哈希漂移、参数量漂移、模块 sha 漂移、`training_runs>0`、测量报告未被选中——各断言退 5 且不写账本；另加 u1 回归，确认五个产物哈希不变。
4. `run.json` 草稿的 `references` 增 `r05_prereg_sha256`（本文件冻结后的哈希）；`open_items` 的 R05 缺口条目转已清。
5. 覆盖台账新增对应行（含“历史哈希只登记”分支）。

---

## 7. 未决点（冻结前须 owner 裁定）

1. 身份测量报告是否需要在冻结前重签一次，使其 `outputs.identity_sha256/execution_v2_sha256` 追上现行值？重签会把 `measurement_report_sha256` 从 `4c77d318…` 改掉，并连锁改 run 草稿哈希；不重签则按第 10 项“只登记”。
2. 第 8 项是否也要检查 `conditions.threads==1`（当前清单记录 1；u2 的 `run_conditions` 已单独记录线程环境）。建议只记录、不判假，避免把“线程数不同”误判成“父不对”。
