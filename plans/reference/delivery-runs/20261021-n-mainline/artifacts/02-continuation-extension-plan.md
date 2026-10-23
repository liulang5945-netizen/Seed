# R2续训与评测活动技术摘要

更新：2026-10-23。当前工作归入持续计划CC-1，不再保留一套独立阶段顺序。实时运行/额度看[03](../../../../active/roadmap/03_CURRENT_EXECUTION.md)，详细排工看[10](../../../../active/roadmap/10_CONTINUOUS_CAPABILITY_EXECUTION_PLAN.md)。

| 项目 | 当前有效语义 |
|---|---|
| 研究目的 | 原生中文多轮学习充分性与初始化/容量对照；尚无新语言能力结论 |
| 身份 | 各臂从自己的已验证耐久parent/最新receipt续接；权重、优化器、RNG、游标与来源保留，不合并权重 |
| 当前范围 | warm_start单臂运行；full_turn_1m、fresh_same_capacity暂缓；它们仍是原配对合同的端点 |
| 训练终点 | 每臂30,096 pretraining episodes / 14,517,912 updates，再525 SFT episodes / 113,209 updates；partial不冒充final |
| 版本化累计上限 | v1.4 sidecar为1,550,000 CPU秒/臂、4,400,000总计；旧实耗、reservation与未知尾账计入，来源审批历史不改写 |
| 活动入口 | `scripts/training/run_taiji_r2_resume.py`，现限warm；新增其他臂入口须在同合同与预算裁决后实施，不把只读预检当续训 |
| 必要保护 | writer互斥、NTFS耐久发布、receipt/bundle与祖先链、完整独立恢复、实际CPU硬上限和停止reserve；内存策略见单入口精简记录 |
| 额度/终点冲突 | 运行中估计仅作告警，不恢复昂贵预测否决；新包开工完整规划，触顶停止后保留partial并提交D1完整缺口材料，不自动循环提额 |
| 终点与评价 | final/readiness、唯一completion、source/launch/ledger全链、无writer/reservation及独立恢复齐备后才准入；准备器不等于worker恢复 |
| 原W5b | parent+三臂、39×4=156、固定前12-family人工、CPU与wall各14,400秒、跨入口共享一次性消费；完整前置未过不消费保留题面 |

## 技术证据与冻结原件

[extension原架构](04-continuation-extension-architecture.md)、[v1.1来源兼容](04-continuation-source-revision-v1-1.md)、[v1.2预算修订](04-continuation-source-revision-v1-2.md)、[v1.3会计校正](04-continuation-boot-accounting-v1-3.md)是历史冻结身份；其中旧额度只解释当时版本，不能改字节。当前v1.4单独sidecar继承旧账，[15精简](15-training-system-simplification.md)说明活动入口。父谱系与旧失败不核销。

[04-v2评测架构](04-v2-evaluation-architecture.md)保留原消费/预算合同；[extension适配架构](04-continuation-evaluation-adapter.md)定义provider与worker的贯通要求，其中旧预算描述属于当时快照，实施时从有效修订链读取，不修改原冻结件。当前适配只是局部实现，不能称全链已通过。

恢复、来源和生成链未闭合时，按CC-1对应包修复；真实输出归因按CC-2。终点/三臂费用不足不推导新能力胜败，更不自行把原四端点评测缩成单臂。

旧活动摘要与先前快照完整见[过渡前全文](../../../../archive/history/CONTINUOUS_TRANSITION_R2_SUMMARY_BEFORE_20261023.md)。
