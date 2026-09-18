# CAP-0 加载策略**决策简报**：训练态装不上产品入口（含落地清单与时序约束）

日期：2026-09-15（本地 01:54）。状态：**未拍板、未改码**。本文只做一件事——把这个已被三处
文档反复引用的决定，写成**可一次性执行**的形态：选什么、为什么现在不能做、做的顺序是什么、
验收看哪几条。所有行号与断言均为**本轮实测核对**（非引用转述）。

相关既有文档：[盘点结果 §处置](M5_CAP0_INVENTORY_RESULT_20260915.md) §「需要显式处置」三选一、
[加载反事实](M5_CAP0_LEGACY_LOAD_COUNTERFACTUAL_20260915.md) §2/§4、
[P1 诊断 §7](M5_CAP0_P1_LANGUAGE_SUPERVISION_DIAGNOSIS_20260915.md)。

## §1 问题的一句话版本

`checkpoints/seed_beta.pt`（`taiji-native-v8`，`tick = 16,000,000`）是**这轮训练真正的产物**，
但产品的装载路径**拒绝**它，报 `ValueError: enabled identity organ checkpoint payload is missing`
（`taiji/model.py:2732`）。⇒ 今天"能不能用训练出来的模型"这个问题，答案是**装不上**。

**并且这两件事已被实测拆开**（反事实件 §1）：放宽守卫后它确实装得上、读回真实 tick，
但输出与 `tick = 2` 的未训练基底**逐字相同** ⇒ **装载缺陷是正确性问题，语言能力缺口是另一件事**。
不要把修守卫当成"让它会对话"。

## §2 为什么它是缺陷而不是设计（本轮核对的代码事实）

| 事实 | 出处（已核对） |
|---|---|
| v8/v9 **被声明为支持格式** | `taiji/model.py:78` `LEGACY_CHECKPOINT_FORMATS = frozenset({"taiji-native-v8","taiji-native-v9"})`；格式闸 `:2597-2602` 放行 |
| 同类载荷**都有** legacy 宽容分支 | `:2611-2629` 形如 `is None and not is_legacy_checkpoint` + `load_legacy_motor_payload` 迁移（注释里写明 M2-2h / M2-2f 迁移） |
| **只有身份器官这一支没有** | `:2726-2732`：载荷不是 Mapping 就 `raise`，无 `is_legacy_checkpoint` 例外 |
| 类注释本身承诺"v8/v9 仍可读，只是要走那条单向迁移" | `:74-76` |
| 仓库自己的诊断文字把它定性为**政策未定**，不是"应该拒" | `reports/taiji_cap0_inventory_20260915.json` 的 `stranded_defect_diagnosis`："whether to apply it is a refuse-vs-migrate policy decision" |

⇒ 严格性不是有意的边界：**其余分支都迁移了，这一支漏了**。这是把它归为"正确性修复"而非
"能力/政策扩张"的依据。

## §3 三个选项与当前证据状态

| 选项 | 内容 | 证据状态 | 上限（能声称什么） |
|---|---|---|---|
| **(a) 补迁移**（推荐） | `:2731-2732` 那一支加 `and not is_legacy_checkpoint` + 一条迁移说明（器官以**全新实例**挂回，沿用文件内既有模式） | 语义已被在跑实验**实证 20 余次**：`probe_taiji_cap0_legacy_load.py:57-87` 的进程内包装做的正是这件事，P3a/P3b 全程用它 | "训练态可被装载；产物与 tick=2 基底同输出"（正确性 + 一条负面能力事实） |
| (b) 重导出为 v10 | 用当前格式重新保存训练态 | **已证不成立**：第二道守卫 `:2604-2606` 比较整个 config 相等（`checkpoint configuration does not match architecture`），且实测没有任何 `--scale` 画像等于 v8 载荷 ⇒ 需要显式"从信封重建配置"的流程（P3b 训练器就是这么绕的） | 不比 (a) 更强，成本更高 |
| (c) 明示降级 | 保持严格，产品入口声明"当前无可用训练态" | 可行 | 只到"如实告知"，不解决正确性 |

**上限排序 = (a) > (c) > (b)**。按"决策点默认取上限"的既有指示，**(a) 是推荐取值**；
但它是**改 `taiji/model.py` 的行为**，属需用户许可的一类改动，本文只把选项与后果摆清。

## §4 为什么**现在**不能做（时序约束，硬事实）

双臂 campaign 在跑（预计本地 2026-09-18 傍晚结束）。驱动是**逐阶段新起子进程**去跑
`eval_taiji_cap0_baseline.py`，而该评测的链路就是"放宽守卫 + 约束解码"
（`run_p3b_campaign.py:53 REQUIRED_CHAIN`）。改 `taiji/model.py` 会：

1. **中途改变后续阶段的评测语义** —— 前 N 个阶段与之后的阶段不再是同一条装载路径 ⇒
   同一 tick 的 `treatment − control` 失去可比性，直接废掉这 54 小时的主效应；
2. 与**已封存基线的自述冲突** —— `test_cap0_baseline_contract.py:325` 钉住 P3a 报告的
   `chain == {"relax_legacy_guard": true, "constrained_decode": true}`；落地后若以
   `relax_legacy_guard: false` 复评，J1（同链路）**直接不通过**，前后不可比；
3. 让一批**读封存报告的契约测试**与新现实脱节（见 §5 第 3 步必须同批改的东西）。

⇒ 结论：**落地时机 = campaign 结束（或明确止损）之后**，与 WP-6 的 `binder_version=v2` 预注册
同处"待 P3b 结论"这一门后。本文写在这里是为了让那一天不需要再重新调研。

## §5 落地清单（一旦解禁，按此顺序做）

1. **改一处**：`taiji/model.py:2731-2732` 的器官分支加 legacy 例外 + 迁移说明注释
   （照 `:2614-2616` 的 M2-2h 写法），**不改**镜像分支 `:2727-2729`、**不改**
   血统摘要校验 `:2733-2763`（缺载荷与错父体是两件事，后者继续 fail-closed）。
2. **加"迁移确实发生"的活体测试**：直接 `pytest.raises` 的对面——用真实 `seed_beta.pt` 断言
   原生 `Taiji.restore` 现在成功、且 `identity_organ` 是**新初始化**实例（不是伪造载荷）、
   且**没有**绕过血统校验。目前**没有任何活体测试**断言这一支的语义（拒绝只由"读封存报告"的
   契约测试钉住，见第 3 步）——这是本决策最大的证据缺口。
3. **同批更新三处会被静默留旧的东西**（缺一不可）：
   - `reports/taiji_cap0_legacy_load_probe_20260915.json` 与 `..._cap0_inventory_20260915.json`
     **必须重跑探针再生**；否则 `test_cap0_legacy_load_contract.py:44-48`（`load_ok is False`）与
     `test_cap0_inventory_contract.py:59-66` 仍读**旧报告**而继续通过 ⇒ **代码已放宽、测试仍绿**
     的静默失真（这些测试"读提交报告且从不重跑"是设计如此，见其 docstring）。
   - `eval_taiji_cap0_inventory.py:350-356` 的诊断文字里**硬编码了行号** `"2726-2732"`，
     并由 `test_cap0_inventory_contract.py:102` 断言该字面串。行号会随任何上方编辑漂移，
     而测试**不会失败**——它只查字符串在不在。⇒ 顺带改为断言**错误文本**
     （`"enabled identity organ checkpoint payload is missing"`）而不是行号；已另登记 DEBT-I5。
   - `probe_taiji_cap0_legacy_load.py:57-87` 的进程内包装：落地后应改为**验证原生路径**
     （否则探针测的是"包装还能用"，而它已经不需要了），并保留 `--child` 短命进程隔离。
4. **重跑 P3a 基线一次**（新链路 `relax_legacy_guard: false` + `constrained_decode: true`），
   作为"落地后世界"的新基线；**不要**把它和已封存的 `relax_legacy_guard: true` 基线混用。
   旧的 P3b 结论仍只在旧链路下成立（07 §4.1 的披露纪律）。
5. 全量套件 `--junitxml` 后台跑，对照 0 失败基线（本轮基线 **1543 passed / 6 skipped / 0 failed**）。

**出口判据（全过才算落地）**：① 原生 `Taiji.restore(seed_beta.pt)` 成功且 tick=16,000,000；
② `identity_organ` 为新初始化实例、血统校验未被放宽（用一个真·错父体样本证明仍拒）；
③ 三份受影响的封存报告已再生且契约测试读**新**报告；④ 诊断文字不再依赖行号；
⑤ 新基线已生成且 `chain` 字段如实为 `relax_legacy_guard: false`；⑥ 全量无新增失败。

## §5b 落地记录（2026-09-18，提交 `30660167`）

**已落地**：§5 第 1–3 步 + 第 4 步的复现部分。编号 M2-2i。

| 出口判据 | 证据 |
|---|---|
| ① 原生 `restore` 成功、tick 真实 | `test_cap0_legacy_organ_migration.py` + 再生报告 `trained_current_guard.load_ok = true`、`tick = 16000000` |
| ② 器官为新初始化、**未放宽血统** | 同一文件 4 支断言：错父体仍 `lineage does not match`、v10 缺载荷仍抛原文本；镜像分支未动 |
| ③ 受影响封存报告已再生且测试读**新**报告 | `taiji_cap0_legacy_load_probe_20260915.json`、`taiji_cap0_inventory_20260915.json` 再生；`test_cap0_legacy_load_contract.py` / `test_cap0_inventory_contract.py` 的旧世界断言按新事实**重推**（未放宽，见下） |
| ④ 诊断文字不再依赖行号 | DEBT-I5 结项：改按错误文本 + 新增"报告里不许出现行号"断言 |
| ⑤/⑥ 新基线与全量 | 全量套件与不放宽守卫的新基线在跑；**注意**：`relax_legacy_guard: false` 基线只在守卫成为 no-op 后才有意义 |

**行为未变的证明（不是假设）**：把 P3a 冻结链路整体重跑
（`reports/taiji_cap0_baseline_repro_20260918.json`）与已封存基线**逐题**比对——
评测面四字段全等、题序全等、**分数 0 处差异、100 题输出逐字节相同**；该比对已升成契约测试，
今后"照样能装载但模型行为变了"会直接红。装载缺陷修复 ≠ 能力修复：输出仍是同一模板
（`templated: true`、`distinct_signatures = 1`），本文件 §4 的"装载成功不等于内容能力"继续成立。

**路上另外挖出并修掉的三个缺陷**（都不在原清单里，都是真 bug）：
① 约束解码 copy-patch 的锚点钉在 `Taiji.generate` 内部一行，R2 改动实现后**整条 CAP-0 语言测量静默罢工**
（唯一的保护是一句子串 grep）⇒ 锚点改钉包装真正消费的接口，并加活体安装测试（DEBT-I10）；
② 子机管道 `ensure_ascii=False` 在中文 Windows（cp936）下让父进程解码失败、`stdout` 变 `None`；
③ 探针带 `error` 时**覆盖自己的封存报告**（本轮真发生：115 行证据被 4 行存根替换，已还原）
⇒ 改为写 `.error.json`（DEBT-I6 部分处置）。

**明确未完成**：DEBT-I7 的测试隔离（其后果已升级为 DEBT-I9：默认基座被写成 tick=36 且不入 git，
"未训练基座"这一不变量目前只能 xfail 挂着，**没有把期望值改成 36 去迁就污染**）、
阶段报告原子写与复用前校验、普查余项 B4/B5/C1/C2、§3"读封存 vs 复现封存"结构升级。



## §6 本文不做什么

- **不**改 `taiji/model.py`、**不**改加载器行为、**不**重跑探针（都在 campaign 之后）；
  ⇒ **本条已被 §5b 取代**（2026-09-18 落地）；文中其余表述描述的是落地前的世界，保留不改写。
- **不**声称修守卫会提升语言能力——反事实件已证"装得上但输出同 tick=2"；
- **不**把本简报当批准；批准后的第一动作是 §5 第 1 步，且**必须**连着做完 §5 第 3 步，
  否则会造成本仓库最坏的一类状态：**代码变了，钉它的测试还在读旧证据**。
