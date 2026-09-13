# M5 P5.2b — 群体因果语料 Gate 预注册（冻结版）

日期：2026-09-13。状态：**已冻结**（本文提交即冻结；冻结后不接预注册之外数据源、不读取 sealed）。
上游：[推进方案](../active/roadmap/03_CURRENT_EXECUTION.md) §6、P5.2a（`predictive_execution_insufficient`，提交 `eff6e1d1`；动作头边界 = 单模板记忆 + 位置先验水平的组合泛化，探针 `b63eb483` 已排除 cue 措辞因素——群体归因面不依赖动作头泛化）。

## §1 研究问题

哪些**实际认知成员**在相同任务条件下存在超出单体的联合贡献（S0 pair interaction）？矩阵与记录能否满足 `InteractionGroupEvaluator` 的 factorial 估计合同？

## §2 可干预成员（真实存在，非虚构）

**成员 = 4 个 family-specialist procedural readout**：同一 `ProceduralSequenceLearner` 构造（hidden 64、epochs 250、seed 独立），分别只在 P5.2a train 场景的一个模板家族（每家族 10 场景）上训练。每个成员是完整可运行实例——**干预 = 从执行环移除该实例**（真实可执行，非标签虚构）。成员训练数据互斥、均属 train 分区。

- opaque 命名：`member-a/b/c/d`（按训练家族索引字母序分配）；语义映射表只存于 runner 配置节，**不进入任何 learner/evaluator 输入**（§6.3 无标签泄漏）。
- 成员 checkpoint 独立保存并独立进程恢复验证（§11.2）。

## §3 干预矩阵（factorial，合同由 `_estimate_pair` 语法决定）

- **context × cells**：12 个 context（4 模板 × 3 个新场景，goal 非平凡持久目标——沿 P5.2a validation 构造，undo 类排除）× 4 cells × 2 次重复 = **96 episodes**。
- **cells**（对每个候选 pair (a,b)）：`(F,F)` inactive baseline——无认知成员则执行环不产生任何动作，episode 记为空事件、outcome 失败（真实语义：无决策来源）；`(T,F)`/`(F,T)` 单成员；`(T,T)` 两成员组合。
- **组合机制（冻结；实现期修订已披露）**：原冻结构造为顺序 fallback——实现时发现其与估计合同存在结构性错配：episode 的成员集合由事件 owner 派生，纯 fallback 链下未被调用的成员不产生事件，pair cell 永远无法成形。修订为 **dual-predict-select**：每 tick 所有在场成员都被真实调用（各自 readout 预测 + 参数绑定尝试，逐成员留痕），冻结选择规则 = 字典序第一个绑定成功者执行、其余记为 dissent；episode 成员集合 = 干预配置集合。此为构造机制修订，九门判据零改动；未被调用即不产生事件的原则不变（每事件对应一次真实认知调用）。
- **执行路径**：与 P5.2a 完全相同的合同链（policy→approval→execute→undo）与参数绑定（goal/world/token 三来源、provenance 逐动作记录）。
- **episode 记录**：outcome = 任务成功 +1.0 / 失败 −1.0（真实执行）；`recovery_effect` 本轮恒 0（如实：恢复交互不在本 gate 度量范围，缺省记录不伪造）；`resource_cost` = 该 episode 实际执行的合同动作数 × 1.0。
- **trace 分区**：前 8 context → `InteractionTraceCorpus.train`，后 4 context → `holdout`（成员已见——训练家族场景、组合×context 未见）；holdout 不进任何 fit/observe。

## §4 九门

| # | 门 | 判据 |
|---|---|---|
| 1 | static_checks | scoped py_compile/ruff/black + 全仓 ruff 通过 + pytest 无新增失败（基线 2 failed 既有债务）+ mypy 61 基线持平 |
| 2 | member_integrity | 4 成员独立训练（数据互斥断言）、独立 checkpoint 独立恢复、恢复后预测一致、篡改 format 拒绝 |
| 3 | cell_completeness | 每个候选 pair 的每个 usable context 四 cells 齐（ episodes ≥ 2/cell）；trace corpus 被 `InteractionGroupEvaluator` 消费不 raise |
| 4 | pairing_identity | 同 context 各 cells 的初始文件/目标/步数上限逐位一致（构造断言） |
| 5 | label_opaqueness | 成员 id opaque；映射表只存 runner 配置节，evaluator/learner 输入不含映射 |
| 6 | real_execution | 全部 outcome 来自真实合同执行；安全违规 0；provenance 三来源记录完整 |
| 7 | holdout_independence | holdout context 不进任何 fit/observe；train/holdout episode id 不相交 |
| 8 | nonempty_records | `InteractionGroupMemberEvidence`（经 `build_member_evidence`）与 evaluator 输出非空；全 rejected 时如实记录拒绝原因清单 |
| 9 | deterministic_and_budget | replica 全流程一致（排除 sensation 随机派生量，披露）；wall ≤ 900s |

## §5 三态

- `group_causal_corpora_supported`：九门全过（矩阵与记录成立；pair interaction 的符号与显著性是**科学结果**如实报告，不作门）。
- `group_signal_constant`：矩阵成立但所有 pair 的 cells outcome 恒定（无差异可测）——触发 roadmap §6 停止点「先设计可干预任务/成员」。
- `failed`：任一机械/合同门失败。

## §6 停止点（沿 roadmap §6）

实际认知成员不足、对照无法执行、所有结果恒定、或不能留出未见组合——不创建虚构群体、不靠扩大重复场景制造样本量。

## §7 产物顺序

1. `scripts/training/eval_taiji_p5_2b_group_causal_corpora_gate.py`（静态检查先行）。
2. 执行落盘 `reports/taiji_p5_2b_group_causal_corpora_20260913.json` + 九门对账。
3. roadmap 状态表与唯一下一步更新 + 独立提交。

## §8 纪律

门值以报告为准逐项对账；禁止反向改判据、剪枝凑数、重跑挑结果；报告不覆写不改绿；`growth_admitted=false`、`can_promote=false` 贯穿；临时脚本用毕即删。
