# 22.5.B.1 设计卡：把纠正的更新路径与 emission 参数分离

> 卡号：`22.5.B.1`｜日期：2026-11-16｜状态：**已结案（负，有定位）**——见[结果卡](RESULT-TAIJI_U2_B1_SEPARATION_20261116.md)（D-20261116-09，`retention_broken_under_separation`）；本卡保留冻结时的设计原文。
> 契约 [`taiji_u2_b1_separation_20261116.json`](../manifests/taiji_u2_b1_separation_20261116.json)（`criteria_sha256=b9a3842f…`，修订 v2：控制臂曝光按方案参数化，仍先于任何正式读数）；[smoke 回执](../../reports/taiji_u2_b1_separation_smoke_20261116.json)、[定价回执](../../reports/taiji_u2_b1_separation_price_20261116.json)。
> 前置：[A 线收束卡](CLOSEOUT-TAIJI_U2_A_LINE_20261116.md)（D-20261116-07）——联合父门不可达的卡点是**纠正与保持不能同时成立**，因为 B 的更新直接落在 A 也在用的 `byte_head`／`joint_readout` 上。owner 已裁决**立项 B 线**（D-20261116-08）。

## 1 这张卡问什么

A 线把失败定位到了**更新落点**，不是预算、不是协议、不是读的强弱。于是 B 线的第一问只有一个：

**把 B 的学习更新从 emission 参数上移开之后，纠正与保持能不能同时成立？**

这是结构卡，不是协议卡。它新增一个专用可塑性靶点，冻结全部 emission 参数，其余（读、数据、释放协议、门阈值）逐字继承 A 线最强配置。

## 2 为什么不是"只冻结 emission、只训 record 通道"

那是最省事的做法，但会重蹈 A.5 的覆辙。A 回合无记录，且：

- read-code 钩子把 `joint_readout` 的输入替换成注入码 ⇒ record 通道经 `h/o/fast` 的路已断；
- copy 分支在无有效记录时结构性惰性（A.2 实测 448 位置零梯度）；
- A.5 的 `train_step` 注释与实测都确认：**A 任务里 record 通道整条在 A 图之外，梯度为 None**。

所以只训 record 通道时，A 的输出逐位不变，**保持门按构造不可能失败**——收束卡明令这种通过不得引用为"保持被挣到"。要让保持门可失败，纠正通路必须**物理上能影响 A 的计算**。

## 3 结构：`correction_adapter`（身份 `taiji-u2-subject-v6`）

- **位置**：`output()` 内，`context = context + adapter(cat([record, context]))`，在 `byte_head` 与 `copy_gate` 之前。`record` 是 `record_read` 的值读出——它在 v4 输出路径里是**死信号**（算了不用），adapter 给了它第一个因果消费者。
- **形状**：`MLP(2d→2d→d)`＝16→16→8，**4 张量／408 标量**（`correction_adapter.first/last.weight/bias`）。
- **出生**：last 层零初始化 ⇒ **出生精确中性**，v6 继承权重后逐位复现 v4；first 层取自种子流延续，194 个基张量逐位继承 A.5 反馈前 bundle。
- **无记录门控**：adapter 在 A 回合（无记录）照样计算 ⇒ 学坏的 adapter **能**伤 A，保持门可失败。smoke 双层证明：+0.05 扰动改变 context（数值活性），+2.0 扰动把 A 精确从 32/32 打到 8/32（行为活性）。
- **敲除**：`_verification_cuts={"correction"}` 精确还原 v4 计算（smoke 逐位验证）。
- **图账**：22,742→**23,150** 标量、194→**198** 张量、**42 状态字段不变**（request profile）；`to_bundle` 新 schema `taiji-u2-subject-v6`，旧 v2–v5 bundle 不受影响（契约测试钉住）。

**可训集（反馈期）**：record 18 张量／762 ＋ adapter 4 张量／408 ＝ **22 张量／1,170 标量**。emission 8 张量／3,919 与其余 168 个基张量**全程冻结**；跑后逐张量比对 start bundle（`path_purity`，176 张量须逐位相同）。

## 4 协议（A.6 修正案，D-20261116-05/06）

一次释放一错、每错 **k=8** 更新、**不回滚**、**轮末全量重问**进门；立即重问只作诊断（full 方案；选定的 lean 不做，省 72 回合）。释放仍走 `feedback_record_ids` 合法闸（非法 id 被拒，smoke 验证）。k=8 是 A.6 唯一产生过纠正的剂量，本卡不重测剂量曲线。

## 5 门（读数前冻结，阈值＝22.5.1 父门自身口径）

| 判据 | 阈值／定义 |
|---|---|
| 主项 | A fit 32 全应答精确 ≥ **0.80**（反馈后、adapter 在线） |
| 纠正率 | 24 首过错例轮末精确 ≥ **0.80**，**3 seed 全部** |
| 保持／损伤 | A 精确损伤 ≤ **0.02**，逐 seed；**可失败**（§2/§3） |
| 反馈增益 | learned 二遍 − control 二遍 ≥ **0.05** |
| 95% 下界 | world 配对（按例，learned−control）72 对、2000 次重采样、seed 20261116，下界 **>0** |
| 稳定性 | 首过对→二遍错 ≤ max(1, 0.10×首过对数) |
| 反事实臂 | 指认第三位置的 record 须被跟随 ≥ **0.80**（凡纠正过门的 seed）；定偏移拷贝／记录盲捷径不能过 |
| 敲除臂 | correction cut 下：B ≤ 首过+0.05 **且** A 摘要 == 基线摘要（纠正由 adapter 承载的直接证据） |
| 路径纯度 | 176 个非可训基张量逐位 == start bundle |
| 控制臂 | 零更新对照：≥3 例合法释放曝光（lean；full 为 24），参数逐位不变、全 24 题二遍 == 首过 |
| 恢复 | 独立进程重载反馈后 bundle，32 A＋24 B 逐字节一致 |
| 不可辨识 | 首过错例 <4 ⇒ `not_identifiable_by_design` |

评测贪婪、精确字节＋EOS＋UTF-8、65 字节帽；教师强制只用于损失（声明同 A.5/A.6）。

## 6 出口（优先序）

1. `retention_broken_under_separation`：任一 seed 损伤 >0.02（先扣代价再谈收益）；
2. `knockout_failed`：敲除不塌缩或不还原 ⇒ 归因无效；
3. `engineering_failure`：纯度／恢复／控制任一不成立；
4. `counterfactual_failed`：纠正过门但反事实不过 ⇒ 捷径嫌疑，不算读记录；
5. **`separation_earns_joint_gate`**：全部门项 3 seed 成立 ⇒ **联合父门在分离路径下可达**（机制可达性，非父、非能力）；
6. `correction_absent_under_separation`（全 seed ≤0.05）／`correction_partial_under_separation`；
7. `budget_stop`、`not_identifiable_by_design`、`pricing_over_budget`。

## 7 能签与不能签

**能签**：分离更新路径这一结构干预下，纠正与保持能否同时成立，以及敲除／纯度／反事实给出的归因。

**不能签**：`accepted_parent`（归 22.5.0/22.5.1 正式父卡）；任何能力声称；check② 未读；22.6 不解锁。正结果的后继是**正式父卡**，不是自动放行。负结果的后继是 B.2 裁决（容量／检索），不自动加剂量。

**局限（预注册）**：read-code 钩子仍切断 fast（A.2），本卡不修读缺口；主项骑在注入的校正读上（A.1b-cal，后验，独立交叉半最低 0.844）；seed 只变缓存位置消费顺序（弱重复）；adapter 是"可塑性定向"的第一个实例，正结果只签本配置下的机制可达性。

## 8 预算（实测单位，非估计）

定价在最终钉死文件上实测（8 次取中位，定价模型用后丢弃）：

| 单位 | CPU |
|---|---:|
| 带合法释放的记录条件更新（含 episode 重建＋adapter） | **0.507812** |
| B 回合 | 0.265625 |
| A 回合 | 0.0625 |

| 方案 | 内容 | 估计 | 帽 |
|---|---|---:|---:|
| `full_3seed` | 3 seed × 24 错 × k8 ＋ 诊断重问＋24 例控制曝光 | 544.5 | 626.175／1252.35 ← **超设计帽 600，按 `pricing_over_budget` 不出** |
| **`lean_3seed`（选定）** | 同上，无诊断重问、控制曝光 3 例（零更新不变性证明不需要 24 例） | 489.515 | **562.943 CPU／1125.886 wall** |

设计帽 **600 CPU／1200 wall** 不上调；定价门在最终测量上拒了 full（上一轮测量 0.4375 时 full 曾在帽内——单价随机负载波动，门按最终测量判，这正是 A.6 立下的规矩）。选 `lean_3seed`：诊断重问只买逐错归因便利，不进门；控制臂的零更新不变性 3 例曝光即足以证明。wiring 另账：smoke **81.14**、定价 **8.16**（0 正式训练步；smoke 0 优化器步，定价 8 步于丢弃模型）。早先 smoke/price 对被 runner 编辑取代后重跑；最终归档对之后仅有一处 run-mode-only 编辑（控制曝光参数化），不触及 smoke/price 代码路径，已在契约 note 声明。

## 9 工程附注

本轮开工时发现 **HEAD 的 mypy 核心棘轮已红 35 条**（P3.9/binding-v2/A 线提交引入，origin 落后 17 提交未暴露；本地工具链与 CI 钉版本逐项一致）。已按纯类型修复归零（`197fff683`，346 项 u2/契约/bind2 回归全过，RNG 流不变），B.1 实现随该提交入库。契约测试 `test_b1_separation.py` **9 项**钉住图账／继承／中性／敲除／活性／梯度路／schema。

## 10 唯一的请求（是非题）

**是否授权按 `lean_3seed` 实价执行 `22.5.B.1`（帽 562.943 CPU／1125.886 wall，576 更新，3 seed，无诊断重问、控制曝光 3 例）？**

目前 **0 正式优化器调用、0 训练步、0 checkpoint、0 能力声称、0 A 父**。