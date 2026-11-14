# P3.8 全局收束卡（2026-11-14）

> 22.5.P3.8.4。机读总账：[reports/taiji_u2_p3_8_4_global_closeout_20261114.json](../../reports/taiji_u2_p3_8_4_global_closeout_20261114.json)，`closeout_sha256 = 056e2a7bb2fda49f7087cc7319bd40e5f389783ec03ae90cbc6c7d06823d437a`（canonical JSON 可重算）。本卡只**索引**该总账的判据、门值与成本数字，不复制第二套；守卫见 [verify_taiji_p3_8_4_global_closeout_20261114.py](../../scripts/training/verify_taiji_p3_8_4_global_closeout_20261114.py)。

## 0. 为什么现在收束（准入门）

[执行包 22 §P3.9](PLAN-TAIJI-CLOSED_LOOP_DEVELOPMENT_20261110.md) 明文规定：「P3.8 的负出口或预算出口也先结案，不能反复加剂量把单位设计无限延期」，并把 22.5 内部顺序固定为 `P3.8.2b → P3.8.3 → P3.8.4 → P3.9.1–4`。[P3.8 计划 §3](PLAN-TAIJI_U2_LEARNING_CALIBRATION_20261113.md) 给 P3.8.4 的交付是「来源／checkpoint／moment、收益／失败／费用总账及全局收束」，出口为「成功、负、不可判、工程或预算均有限结案」，审计草案 60 CPU／120 wall。

P3.8.2b 已由独立审计结案 `engineering_transfer_verified`；P3.8.3 已按冻结合同执行并由独立审计结案 `budget_cannot_express`。**两张后继卡都已到有限出口**，故本卡不再产生新读数、不再补训、不再改门值，只做合并与交接。

## 1. 一句话结论

**P3.8 按「负结果 ＋ 预算停止」有限出口正式关闭**：本包唯一学成的正向结果是「单对象／六固定问题／三几何族／单初始化的首问神经参照」（train/dev 各 179/180），**不是** Native 主体能力；Native 工程接口与完整学习包恢复一致性已通过，但同一 Native 父上的整体配方行为比较因 A 臂无动态范围而判 `budget_cannot_express`。`native_parent_qualified=false`，能力账无变化，0 能力声称。

**这是卡 22.5.P3.8 的关闭，不是 M 线或里程碑关闭，也不是「范围排除批准」，更不是配方被否决。** 按原冻结口径，预算不足以表达假设既不构成假设被否决，也不构成加剂量理由。

## 2. 全阶段总账（数值以机读总账 `stages` 为准）

| 卡 | verdict | 关键读数 | 机器 CPU |
|---|---|---|---:|
| P3.8.1 原字节 CE | `first_round_not_calibrated` | train/dev 50%/50%，2000 Adam／360000 梯度样本呈现；负结案 | 29.000000 |
| P3.8.1b 内容监督原 Adam | `content_credit_not_calibrated` | train 66.11%／dev 63.33%，去头等值、固定末点未过门 | 38.296875 |
| P3.8.2a 受验收更新 | `first_round_calibrated_stable_credit` | train/dev 99.44%；1845 Adam，隔离超额一次，1844 尝试／1715 参数接受 | 83.343750 |
| 阶段审计 | `passed_calibration_evidence_audit` | 126 份中间／终点／隔离文件，3 次 teacher 梯度复核，0 optimizer | 5.296875 |
| 参照重建 | `regenerable_exact` | 1844 次 Adam 复算单列，推理包 SHA／全部生成精确一致 | 32.515625 |
| P3.8.2b 预检 | `precheck_passed_admitted` | 完整包保存／独立进程恢复／篡改拒绝，0 Adam | 1.9688 |
| P3.8.2b duty v1 | `engineering_failure` | v1 容差低于 float32 机器精度，PASS 分支算术不可达；原读数保留 | 2.2188 |
| P3.8.2b duty v2 | `duty_verified` | 另出 v2 预注册后 0 失配，`J_content` 实算 | 2.6875 |
| P3.8.2b 提案 v1 | `engineering_transfer_verified` | runner 自报，后被独立审计否决 | 4.0625 |
| P3.8.2b 审计 v1 | `inconclusive_coverage_gaps` | 恢复覆盖 1/8，4 类证据未覆盖 | — |
| P3.8.2b 提案 r2 | `engineering_transfer_verified` | 6 次真实 Adam、u6/v4、9 次独立恢复、4 份 committed 事务回执 | 5.0312 |
| P3.8.2b 审计 r2 | `engineering_transfer_verified` | 13/13、`uncovered=[]`、恢复 9/8；`rebuild.entries=0` 登记为范围限制 | — |
| P3.8.3 定价 | `price_measured_admission_possible` | 128 次配对更新估 461.7749 CPU，落在 600 帽内 | 4.5625 |
| P3.8.3 训练前预检 | `preflight_passed` | 两臂各一次独立进程完整包恢复，0 优化器步 | — |
| P3.8.3 执行 | `budget_cannot_express` | 两臂各 128 次配对更新，before/after 全 0.5，配对 Δ=0、CI [0,0]，控制也 0.5 | 271.3750 |
| P3.8.3 独立审计 | `budget_cannot_express` | 12/12、0 缺口；给 A 臂注入动态范围后 verdict 改变 ⇒ 守卫非恒真 | — |

`—` 表示该报告未记录 CPU 字段，**不得估填**；这些审计／预检脚本的成本归入 §3 的「未计账类别」。v1 与 r2 是两次真实执行，两次 CPU 都计入实耗，v1 verdict 原样保留不被 r2 覆盖。

## 3. 成本总账

| 项 | CPU | 说明 |
|---|---:|---|
| P3.8 原始额度 | 1110 | 原计划总额 |
| 11-13 已结案预留 | 320 | 首卡封账 30 ＋ 重建 90 ＋ 内容对照 90 ＋ 更新校准 90 ＋ 阶段审计 20 |
| 未分配额度 | 790 | 工程 130 ＋ runner 600 ＋ 最终审计 60 |
| 11-13 入库读数合计 | 188.453125 | 上表前五行 |
| 11-14 工程实测合计 | 15.9688 | 帽 130 |
| 11-14 runner 实测合计 | 275.9375 | 帽 600；其中 P3.8.3 正式执行 271.375 / wall 290.75 |
| **入库读数总计** | **480.359425** | 三组相加 |
| 790 中未使用 | **498.0937** | 含未动的 60 最终审计帽；**不转移给 P3.9** |

P3.8.3 正式执行的 `cpu_ratio = 1.1530831714344465` 在冻结带内；0 check／final／confirm 读取。

**必须随成本一起引用的两个事实**：① 冻结定价把「单次前向 0.2031 CPU」当评估代理，而实测完整 dev `evaluate`（180 episodes × 2 问自由生成）为 **23.6562 CPU**，低估 **116.48 倍**——预算结论未翻转，因为 B 臂每次只做 1 次 α 评估而非保守 8 次，且实际总费仍远低于 600 帽；② 临时探针、守卫、纯 JSON 审计、ruff/black/pytest 门禁与本卡自身的 CPU **未 instrumented**，如实列为未计账类别，不并入 480.359425，也不编造数字。本卡守卫自身实耗写在守卫回执里。

## 4. 父身份与谱系

唯一父仍是 **P3.7 问题 Fusion `update-000758.bin`**，路径见机读总账 `parent_identity.path`，SHA-256 `c8f2359b0bef3085bce88a0a0a8ee7eff5f66a0bc614d2bf52e7ef361638b017`，`native-joint-query`、`update = cursor = 758`、25086 参数／214 张量／optimizer 191 state。

- P3.8.2b 的参数迁移事务只在**隔离影子 runtime** 提交，未写出新接受父；4 份回执 `rebuild.entries=0`，因影子 runtime 的 `CausalHistory` 为空。
- P3.8.3 两臂都在冻结父的**内存副本**上更新，0 checkpoint 写、0 产生父。
- 故 `native_parent_qualified=false`；正式 A 的多实体／三 seed／独立 check 仍待对应父，本包未准入。

## 5. 能力账与负结果

唯一正向结果是**校准后的首问神经参照**：去辅助头后 train/dev 各 179/180 的真实字节生成，最低问题组 29/30，两输入控制 50%。它的范围是单对象、六固定问题、三几何族、单初始化；与 Native 架构、参数量、历史处理和费用都不同，**不得记入 Native 主体能力**。原字节 CE 与内容监督原 Adam 的负结果及固定末点全部保留。

**输入不敏感线索（四条独立证据同向）**：P3.8.3 父在 dev 上对 RGB 置零、问题置空均保持 0.5；P3.7「Native 恒答未知」；21.3「表示轴对此图关闭」；FEC-1 旧图写键函数类塌缩。本卡**不判定架构不可能**，只把它登记为架构级缺口候选，交 P3.9 的 A 兼容嵌入／B 单位原生核心比较裁决。

同时必须保留**预算混淆**：128 次更新内两臂逐位相同，也可由预算不足以表达解释；架构解释与预算解释**不互相代签**。P3.8.3 是整体配方比较、无消融臂，只能签配方级结论，不单独归因内容监督或 α 搜索。

## 6. 限制与未覆盖范围（逐条可查）

1. `rebuild.entries=0`：非空主体历史按新 θ 重编码**未覆盖**，归 22.3／22.5 运行时历史链；单写者保证由库事务提供，未独立测量。
2. 评估单价低估 116.48 倍（见 §3）。
3. v1 判据 `atol=1e-12` 低于 float32 机器精度 1.19e-7，PASS 分支算术不可达；v1 读数保留，另出 v2 预注册，只追加不改写。
4. P3.8.2b runner 首份自报 `engineering_transfer_verified` 属过度声称，已由独立审计改判 `inconclusive_coverage_gaps`，补齐 4 项证据后才由 r2 重判通过。
5. A 臂无内容头（214 张量），原 Capsule registry 硬编码 216 会拒收；按 owners 求和泛化（B=216／A=214）并加绿／红契约测试——**修实现以符合合同，未改判据**。
6. P3.8.3 训练前预检回执 `dev_reads` 曾误记 1，已就地更正为 0 并注明原值。
7. 0 check／final／confirm 读取，未触及任何封存面。
8. 本包未验跨轮、纠正、保持、自驱、成熟成长与长期物理理解；这些仍按各自后继合同与能力父门验证。

## 7. 有限出口裁定

机读总账 `closeout_conditions` 列出 **C1–C8 共 8 项结项条件，全部通过**：三项校准与阶段审计／重建已结案、P3.8.2b 独立审计通过、P3.8.3 冻结合同执行并独立重判、v1 不可达与过度声称如实保留、16 份报告逐字节绑定 sha、父身份唯一且无新接受父、能力账无变化且正式 A 未准入、成本与预算边界落账且不转移额度。守卫逐项从原始报告重读 verdict 与 sha，不是只读这 8 个布尔。

裁定 **`p3_8_closed_limited_exit`**，出口类 `negative_result` ＋ `budget_stop`。按出口语义：**不改门值、不加剂量、不缩数据重跑、不新开包、0 能力声称**；收束 ≠ 里程碑关闭 ≠ 范围排除批准 ≠ 配方被否决。

## 8. 交接 P3.9.1

按执行包 22 §P3.9 与 [路径备案 D-20261114-01](DECISION-TAIJI_NEURAL_CORE_PATH_20261114.md)：

- **先按两图实价定价，再冻结** A 兼容映射与 B 完整输入／通信／直接查询／消费接口；原 120 CPU／240 wall 草案未经冻结不得执行或自动翻倍。
- 工程父取 P3.8 最后可恢复、来源完整的 Native 研究父；**无新接受父则使用其冻结起点**，manifest 必须注明 `native_parent_qualified=false`。
- **不转移 P3.8 的 790 CPU**，P3.9 在定价冻结前没有执行预算；不触及 sealed check／final／confirm。
- D-20261114-01 的复核点「P3.8.4 全局结案」由本卡达成；余下复核点为 P3.9.1 图冻结与 P3.9.4 路线裁决，随后每包收束检查目标偏移。
- P3.9.3 之前不得以局部数学核验（41 项兼容算子）或本工程通过代签 A/B 行为比较；22.6／22.7 仍受各自能力父门约束。

## 9. 判据、出口与验收（索引）

- 冻结判据与全部数值：[P3.8.3 预注册 manifest](../manifests/taiji_u2_p3_8_3_prereg_20261114.json)（`criteria_sha256=b88b3381…`）、[P3.8.2b v2 合同](../manifests/taiji_u2_native_credit_transfer_proposals_v2_20261114.json)（`c26b7822…`）、[P3.8.4 机读总账](../../reports/taiji_u2_p3_8_4_global_closeout_20261114.json)（`closeout_sha256=056e2a7b…`）。
- 出口语义：本件 §7；执行包 22 §P3.9 的「负／预算出口先结案」条款；P3.8 计划 §3 的 P3.8.4 行。
- 验收：C1–C8（机读总账 `closeout_conditions`），由 [守卫](../../scripts/training/verify_taiji_p3_8_4_global_closeout_20261114.py) 从原始报告重算并做红绿双向演练，回执 `reports/taiji_u2_p3_8_4_global_closeout_guard_20261114.json`。
- 本卡 0 训练／0 优化器／0 前向／0 dev／0 check／0 final／0 confirm／0 新父，只读已入库报告与冻结父 blob 的 sha。

## 10. 队首（收束后）

唯一队首已同步至 [MAINLINE](../MAINLINE.md)／[MASTER-PLAN](../MASTER-PLAN.md)／[ROADMAP-NEXT](ROADMAP-NEXT_20261104.md) 三处逐字一致，流水见 [03_CURRENT_EXECUTION](../active/roadmap/03_CURRENT_EXECUTION.md)：**P3.8 已按负结果＋预算停止有限出口全局收束；唯一后继为 P3.9.1 的 A/B 双图实价定价与冻结，先定价后执行，不转移 790 CPU、不触及封存面。**
