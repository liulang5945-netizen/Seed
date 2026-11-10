# 证据协议记录：字节冻结重基线（6 个 taiji 模块的类型修复）

**日期**：2026-11-10（记录文件名与重跑产物使用 `20261111` 作为序列标签，见 §8 说明）
**类型**：证据协议变更（不是读数修正；旧记录一个字节未改）
**授权**：D-20261110-04

## 1. 为什么需要新记录

6 个模块被 9 份已入库 manifest 用 `frozen_files` 按严格字节钉住（`sha256_of`，任何编辑都会触发 `frozen input drift`）。修 CI 的类型错误必须编辑这 6 个文件，因此旧 manifest 在改后无法再通过自身校验。协议不允许覆盖旧记录，所以：

- **旧 manifest／旧报告／旧 progress 全部原样保留**，仍指向旧字节，仍可自证当时的状态；
- **旁侧新增一套新记录**（新 manifest＋新报告＋新 progress），指向新字节；
- 两套记录并存，用本文件说明映射关系与撤回路径。

## 2. 改了哪 6 个文件（旧哈希 → 新哈希）

| 文件 | 旧 sha256 | 新 sha256 |
| --- | --- | --- |
| `taiji/binding_copy.py` | `37f4885e8dc33a41f6f701e3d0e87daf15c126be735acceea02098e5e39c045c` | `ce74e47f5e9b28e20b330b1c3438603bfe91d9822b107c981e2193b43068f19f` |
| `taiji/record_relation_core.py` | `8318dde5c3d3b3ff42a2c94e528aec7552363d4f9aef2c11ee95b8e868ef64a2` | `3281f2599eaa6208bf54a61e14f4ced33151f440f903b4a75450158fbcc43256` |
| `taiji/living_core.py` | `6c0c53a2a6465627918721da83b1e0fb80a04c262f95c97d9b825ee858d0785d` | `c9df8b7a99c7950e0b69f0881c96d3ac3354dba548ccdede8b1902b6b9ddba38` |
| `taiji/data_premise_18_0.py` | `b98b68534d26773091b234d5b7c92220def1059be56ab2c2bb0f3f8cc5791204` | `0d849106fbadd0d46f4c2e06853eea98a6606e3067433ac8db6ba452d84587e4` |
| `taiji/selection_routing.py` | `3b80ed033055ed9eb15afc386befae7d6b73ad829844c68a59a89f20b2aa5bde` | `b90c410b844362960954b1fd1085ca86eb923e20254d2d0f0501ae2e8cac7963` |
| `taiji/selection_routing_checkpoint.py` | `606cc68cf1c5a628d3b91c494258c9f4b20084fa638eb88615e8e60dfcd5ce61` | `80aa1dc300bec6b3384b149e11ffb4fa2377edffd2cc427f61a90cb5ea4a0a20` |

### 2b. 146 条按根因分组（是否必须动签名）

在 HEAD（`66af8896f`）的旧字节上重跑 `mypy --follow-imports=silent seed taiji`，得到 146 条，与改前读数逐条对齐；按 mypy 报文归因如下（`A`–`H` 互斥，合计 146）：

| 根因 | 条数 | 文件分布 | 采用的手段 | 是否动签名 |
| --- | --- | --- | --- | --- |
| A. `nn.Module.__getattr__` 把动态属性解析成 `Tensor \| Module`（含由此派生的 "Tensor not callable"、"Unsupported operand types … Module"） | 101 | binding_copy 57／record_relation_core 36／living_core 4／selection_routing_checkpoint 4 | 类级属性注解（让动态属性有确定类型）＋调用点 `cast` | 否 |
| B. 状态字段声明为 `object`（`sc_bridge`／`sc_sfk`） | 10 | living_core 10 | `cast(BridgeTable/AddressTable, …)` | 否 |
| C. 两个候选类不兼容（`BindingCopyCandidate` vs `RecordRelationCandidate`） | 5 | binding_copy 4／selection_routing_checkpoint 1 | `cast` | 否 |
| D. `no-any-return`（`sum(...)/len(...)` 等表达式退化为 `Any`） | 8 | living_core 4／selection_routing 4 | `cast(torch.Tensor, …)` | 否 |
| E. 容器字面量推断不出元素类型 | 2 | living_core 2 | 局部变量注解 | 否 |
| F. 元组形状／解包不一致（2 元 vs 3 元） | 5 | record_relation_core 4／binding_copy 1 | 局部注解／解包修正 | 否 |
| G. `Optional` 字段在分支内未收窄 | 14 | living_core 11／selection_routing 3 | 6 个 `assert`（收窄）＋3 处类级 `\| None` 注解 | 否 |
| H. 注解笔误（`dict[str, dict[str, int]]` 实为 `dict[str, int]`） | 1 | data_premise_18_0 1 | 改正注解 | 否 |

**结论：146 条里 0 条需要改公开签名**；全部在函数体内或类级属性声明上完成。其中 A–F、H 共 132 条是纯静态手段（注解／`cast`），G 的 6 个 `assert` 是唯一会在运行时执行的语句。

其中只有 2 处触及运行时可观测行为：

1. `living_core.py` 新增 4 个 `assert`（`python -O` 下会被移除；只在 store 四字段不一致时把异常类型从 `TypeError` 变为 `AssertionError`）；
2. `record_relation_core.py` 把 `RecordRelationCandidate.subject` 声明为 `Any`（弱化静态检查，运行时零差异）。

没有任何公开签名被改动。

## 3. 受影响的旧记录（9 份 manifest）

`taiji_stop_shape_r58_20261109.json`、`taiji_stop_shape_r59_20261109.json`、`taiji_object_length_r60_20261109.json`、`taiji_object_length_dense_r60b_20261110.json`、`taiji_object_edge_r61_20261109.json`、`taiji_object_edge_r61_20261109_v3.json`、`taiji_object_edge_intervention_r62_20261109.json`、`taiji_object_edge_char_r63_20261110.json`、`taiji_stop_boundary_20261109.json`。

前 8 份钉住 6 个文件中的 6 个；`stop_boundary` 钉住其中 4 个（不含两个 `selection_routing*`）。

## 4. 新增的旁侧记录

`scripts/training/replay_byte_freeze_rebaseline_20261111.py`（回放驱动，不改任何 runner 源码，只在导入后重绑模块级 `MANIFEST`／`REPORT`／`PROGRESS` 三个全局再调用 `module.main()`）产出：

- `plans/manifests/*_rebaseline_20261111.json`：8 份（`r61` 的脚本本身把 manifest 写到 `_v3`，故其新 manifest 名为 `taiji_object_edge_r61_20261109_v3_rebaseline_20261111.json`）；外加 R57 的 `taiji_stop_boundary_20261109_rebaseline_20261111_result.json`（登记本次 confirm 读的 `report_sha256`）；
- `reports/*_rebaseline_20261111.json`：8 份；
- `reports/*_progress_rebaseline_20261111.jsonl`：8 份，但 `.gitignore: *.jsonl` 使它们与旧 progress 一样**只存在于工作区、不入库**（旧 `*_progress.jsonl` 也从未进过历史）。

新 manifest 的 `definition_digest` 与旧 manifest 相同——判定口径未变，变的只是被钉字节。

## 5. 重跑与逐项比对结果

7 份 dev 报告（R58–R63）**逐行 diff 只有两类差异**：

1. `preregistration` 路径与 `preregistration_sha256`（新 manifest，必然不同）；
2. `cost.cpu_seconds`／`wall_seconds`（机器负载不同，非语义字段）。

除此之外**判定名、核心计数、各类分布、逐格行、锚点字段数全部逐项相同**：

| 步 | 判定 | 关键读数（旧＝新） |
| --- | --- | --- |
| R58 | `stop_rule_partial_general` | 与旧一致 |
| R59 | `object_slot_short` / `word_effect_weak` | 4 格 `byte_exact_by_cell` 一致；词效统计一致 |
| R60 | `stem_dependent` / `limit_not_relaxed` | anchor 15 字段、`structural_exact 12/12` 一致 |
| R60b | `no_stable_boundary` | anchor 74 字段、21/21 三组零多余/零缺失一致 |
| R61 | `argmax_predicts_content` | 29/48、Type1 0、Type2 0、McNemar p=3.814697265625e-06、anchor 128 字段一致 |
| R62 | `causal_one_direction` | rescue 18/19、break 29/29、anchor 1890 字段 0 不符、sham 恒等一致 |
| R63 | `char_handle_complete` | rescue 19/19、break 29/29、`put_char_exact` 48、`yu_char_exact` 0、`put_lead_same_source` 47、anchor 1890 字段一致 |

### R57 `stop_boundary` 的差异（单列，不抹平）

新报告与旧报告有 1 处**结构性**差异：`read_register.recovery_rerun` 由 `true` 变 `false`，因此旧报告里的 `recovery_check{compared:192, mismatches:[], identical:true}` 块在新报告里不存在。

原因：该字段只表示"本次运行前 progress 里已有一遍逐例日志，故需要逐行复算"，它是**崩溃恢复**标记，不是读数。旧记录产生时脚本先跑过一遍（progress 里已有 192 行），于是第二遍做了 192 行复算；回放用的是全新 progress，只跑一遍，所以没有恢复块。

补偿取证：把旧 progress 的第一遍 192 行、第二遍 192 行与新 progress 的 192 行三方向逐例比对，`byte_exact`／`answer_bytes`／`terminated` **0 处不同**（192/192 `byte_exact=true`）。因此判定 `stop_edge_probability_side`、`completion 1.0`、`gate_a`／`gate_b`、`eos_argmax=false`、`read_integrity=true` 均未变。该比对依赖本地 progress（不入库），但同一 6 字节下重跑即可复现。

**协议代价（必须记账）**：R57 是 confirm 分区的第 2 次读（`confirm_read_index: 2`，授权 D-20261109-60）。本次重跑构成**第 3 次读**。结果与第 1／2 次逐例相同，故不产生任何新声明；但这是本次授权重跑带来的不可撤回的读取成本，后续任何步骤不得再读 confirm 分区。

## 6. 重跑成本

CPU 合计 337.29 s、墙钟 345.87 s（旧记录合计 CPU 263.02 s、墙钟 270.84 s；差额来自机器负载，单步均未越过各自 cap）：

| 步 | 旧 cpu | 新 cpu |
| --- | --- | --- |
| R58 | 6.39 | 6.38 |
| R59 | 21.91 | 17.64 |
| R60 | 3.31 | 6.89 |
| R60b | 8.16 | 12.22 |
| R61 | 21.44 | 24.91 |
| R62 | 52.66 | 76.56 |
| R63 | 81.53 | 114.08 |
| R57 | 68.62 | 78.61 |

## 7. 撤回路径

1. 新记录全部是**新增文件**，删除 `plans/manifests/*_rebaseline_20261111.json`、`reports/*_rebaseline_20261111.json`、`reports/*_progress_rebaseline_20261111.jsonl` 与 `scripts/training/replay_byte_freeze_rebaseline_20261111.py` 即回到纯旧状态；
2. 若同时要撤回字节改动：`git revert` 本提交，6 个文件恢复旧哈希，9 份旧 manifest 重新自洽；
3. 旧记录从未被写入，所以撤回不需要恢复任何被覆盖的内容（唯一不可撤回的是 §5 的第 3 次 confirm 读）。

## 8. 命名说明与限制

- 产物后缀写作 `20261111`，是本轮任务规划时确定的序列标签；实际执行日期为 2026-11-10。该标签不构成日期声明，产物内部的 `preregistration`／`preregistration_sha256` 与文件名自洽。
- 新记录只覆盖 8 个已经存在的探针步骤；它不新增任何判定口径，也不改变 192/192 的形状限制。
- 换行符：`binding_copy.py`／`record_relation_core.py`／`living_core.py`／`selection_routing.py`／`selection_routing_checkpoint.py` 现在显式声明 `text eol=lf`（除 `living_core.py` 外是本轮补的）。在 Windows 上 `core.autocrlf=true` 的全新检出原先会把它们变成 CRLF，使 `frozen_files` 在干净克隆上报 drift；读数是按 LF 字节录的，CI 也在 LF 上跑，所以这条只补可复现性、不改任何字节。
- **`data_premise_18_0.py` 是例外，且必须保持例外**：它被钉的是维护机工作区的 CRLF 字节（8 份 manifest 全部按这个值记录）。把它改声明为 `text eol=lf` 会让它在下次检出时变成 LF、哈希变化，从而让 8 份新 manifest 全部失效；要重新取哈希就得重冻重跑全部 8 步，其中包含 R57，等于**第 4 次读 confirm 分区**，协议禁止。因此该文件不声明 EOL 属性：维护机上字节不变、记录自洽；Linux 检出是 LF，无法逐字节复跑这 8 步——这是本次之前就存在的限制，不是本次引入。
- 这 8 份新报告里的 `cost` 数字是本次机器状态下的实测值，不与旧值做"择优保留"。
