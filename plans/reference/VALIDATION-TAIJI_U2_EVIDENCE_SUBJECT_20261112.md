# 请求条件证据读取的主体集成回执

> 22.5.P3.3，2026-11-12。passed_engineering_only。新候选具有23,326参数/201张量/44状态字段，新增584参数/7张量及证据目录、撤销登记两项状态。任务学习收益待P3.4。

## 1. 正常入口及来源

`EvidenceSubject`与`EvidenceRuntime`为显式候选类型，读取接入实际branch_context，正常感知、request、表达、字节接续复用U2入口。持续goal和当前request按原合同独立。读取公开的sensor/world包，登记包id与规范内容摘要，核对seen_packets、settled receipt和时钟。字段来自已经验证的接收入口，读取时再核验内容与来源。空、未结算、缺失和撤销分别覆盖。

规范证据保存在原evidence中，目录保存摘要。每次调用按当前编码器重建，无派生K/V持久缓存。最多4个活跃包/128token，超限具名停止，接收事务保持原目录。撤销的合同范围是后续直接证据读取权限。已形成的H/F影响作为历史状态登记，知识纠正由后继B卡验证。

集成时纠正了元数据位的解释。RGB的第6位描述坐标类型，不能用作有效token掩码。合法模态由接收入口与proprio有效位决定，实际正常包保留16 RGB+7 audio+1 proprio，共24token。读取元数据的两个相对时间值用包t_end及当前world_time重新定位，原始包不发生修改。

## 2. 覆盖矩阵

| 方程/合同 | 正常源码 | 参数/状态 | 核验证据 |
|---|---|---|---|
| 请求直接查询、两步共享Q/K/V | `u2_evidence_read.py`、`EvidenceSubject.branch_context` | 7个reader张量、原request/H | 完整回答损失对7张量及RGB encoder有梯度；第一步V到第二步attention导数非零 |
| 规范证据重编码 | `EvidenceSubject.evidence_tokens` | 原evidence、目录摘要、当前encoder | 24token、两次编码迁移、原始包摘要不变 |
| 来源/时间/容量/撤销 | `EvidenceSubject`、`EvidenceRuntime.revoke_evidence` | seen/settled、目录、撤销历史 | 非法source/scope、缺包、篡改、未结算、重复包、时钟、非法模态、129token及第五包检查 |
| 同一候选部分表达接续 | typed runtime factory、候选bundle schema | 201张量、44状态、raw history | float32/64共6个独立进程接续，覆盖原版及每种精度两次编码迁移 |
| 完整有限窗口信用重建 | `u2_recovery.py`、`u2_learning_transaction.py` | 参数、规范历史、原输出及资格状态 | 1,206个逐参数梯度条目一致，未用参数的None模式同时比较；最大差0 |
| 历史父继承 | `load_legacy_learner`、`EvidenceSubject.from_parent` | P2 Native-conditional父、源码快照 | 每个旧pin按现源或登记快照验证，194张量逐位继承，旧认知源码哈希继续匹配 |

正式[独立进程报告](../../reports/taiji_u2_evidence_subject_20261112.json)包含源身份、父摘要、6例和资源。新增12个案例及必要既有请求、基线、物理、恢复/重编码回归共108项通过。Ruff/Black/mypy局部检查通过。0优化器更新，check/final读取0。

## 3. 资源与证据修订

正式主/worker共18.8125 CPU，准备阶段两项脚本接口错误以30 CPU保守预留记账，合计48.8125，符合240 CPU帽。失败分别为重复创建路径和将迁移API的committed误写为accepted，均为核验脚本假设问题。开发工具及回归测试资源独立于此计量范围。

[成本v2](../../reports/taiji_u2_evidence_read_cost_20261112_v2.json)测得单事件24token读取夹具中位0.0029296875 CPU秒，整进程峰值RSS239,419,392字节，含Python/PyTorch。计时调用含一次独立证据重建及一次branch_context内重建，未代表训练单步价格。四次规范历史重建为0.171875/0.3125/0.171875/0.296875 CPU秒，限本夹具历史长度。

成本v1错误匹配audit操作名，漏提取迁移记录，原报告保留，v2更正并按dtype/revision去重。两次成本程序CPU4.71875+4.328125，纳入另预留60 CPU。已冻结报告的确切脚本快照及旧运行时源码由[迁移清单](../manifests/taiji_u2_p3_source_migration_20261112.json)登记。当前price脚本仅补类型标注，其运行证据对应登记的v2快照。

P3.3工程/成本预留240+60，从22.5未分配17250中登记，余16950。上述可核验程序及准备预留合计57.859375 CPU。本轮任务能力资格为空。

## 4. 唯一后继

进入[P3.4配对架构实验准备](PLAN-TAIJI_U2_EVIDENCE_LEARNING_20261112.md)，先交付参数配平控制、新父checkpoint身份与实价。训练剂量、正式执行清单和开发fit分区在新结果前冻结。本回执只签主体工程，实际中文问答、保持、自驱与成长收益按对应任务验收。
