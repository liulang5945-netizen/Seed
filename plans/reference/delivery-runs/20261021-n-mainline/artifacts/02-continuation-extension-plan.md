# R2续训与评测活动技术摘要

更新：2026-10-24。当前工作归入持续计划CC-1，不再保留一套独立阶段顺序。实时运行/额度看[03](../../../../active/roadmap/03_CURRENT_EXECUTION.md)，详细排工看[10](../../../../active/roadmap/10_CONTINUOUS_CAPABILITY_EXECUTION_PLAN.md)。

| 项目 | 当前有效语义 |
|---|---|
| 研究目的 | 原生中文多轮学习充分性与初始化/容量对照；尚无新语言能力结论 |
| 身份 | 各臂从自己的已验证耐久parent/最新receipt续接；权重、优化器、RNG、游标与来源保留，不合并权重 |
| 当前范围 | D1已完成、v1.5已激活；warm_start已有完整终态，其余两臂沿03既有最多两臂并行安排；实际进程/进度见03 |
| 训练终点 | 每臂30,096 pretraining episodes / 14,517,912 updates，再525 SFT episodes / 113,209 updates；partial不冒充final |
| 版本化累计上限 | 当前v1.5 sidecar为1,590,000 CPU秒/臂、4,560,000总计；旧实耗、reservation与未知尾账计入，来源审批历史不改写 |
| 活动入口 | `scripts/training/run_taiji_r2_fast_ledger_resume_v4.py --arm <arm>`，并发与逐臂writer规则按03；历史resume入口不覆盖当前实际运行身份，不重复启动同臂 |
| 必要保护 | writer互斥、NTFS耐久发布、receipt/bundle与祖先链、完整独立恢复、实际CPU硬上限和停止reserve；内存策略见单入口精简记录 |
| 额度/终点冲突 | 沿实际绑定版本的预算/恢复守卫运行；D1已完成，本轮不自动再提额。必需臂触顶/资源拒绝则停后续启动并按03封存，不改终点或分母 |
| 终点与评价 | final/readiness、唯一completion、source/launch/ledger全链、无writer/reservation及独立恢复齐备后才准入；准备器不等于worker恢复 |
| 原W5b | parent+三臂、39×4=156、固定前12-family人工、CPU与wall各14,400秒、跨入口共享一次性消费；完整前置未过不消费保留题面 |
| 本轮结束 | 依03的R1–R6与终态表；完整流程可得正/负/混合或不可判结论，预算/资源不足及工程失败分别封存；正式信息/信用实验另属后继包 |

## 技术证据与冻结原件

[extension原架构](04-continuation-extension-architecture.md)、[v1.1来源兼容](04-continuation-source-revision-v1-1.md)、[v1.2预算修订](04-continuation-source-revision-v1-2.md)、[v1.3会计校正](04-continuation-boot-accounting-v1-3.md)是历史冻结身份；其中旧额度只解释当时版本，不能改字节。v1.4及当前v1.5单独sidecar继承旧账，[18裁决](18-d1-budget-decision.md)记录v1.5，[15精简](15-training-system-simplification.md)记录早期入口精简。父谱系与旧失败不核销。

[04-v2评测架构](04-v2-evaluation-architecture.md)保留原消费/预算合同；[extension适配架构](04-continuation-evaluation-adapter.md)定义provider与worker的贯通要求，其中旧预算描述属于当时快照，实施时从有效修订链读取，不修改原冻结件。当前适配实现、合成与独立审查已完成，真实端点/生成贯通尚未验证。

恢复、来源和生成链未闭合时，按CC-1对应包修复；真实输出归因按CC-2。终点/三臂费用不足不推导新能力胜败，更不自行把原四端点评测缩成单臂。

旧活动摘要与先前快照从Git读取，见[历史全文恢复](../../../PROJECT_HISTORY_CONSOLIDATION_20261023.md)。
