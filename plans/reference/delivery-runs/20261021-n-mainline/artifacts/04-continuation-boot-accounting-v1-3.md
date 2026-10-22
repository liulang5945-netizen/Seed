# v1.3 extension boot身份与尾账审计校正

已批准预算保持每臂1.5M、aggregate4.3M。只对v1 continuation_prepared事件之后的浮点身份误判作审计校正，不重算原parent尾账。基线三臂合计3,949,572.172730 CPU秒完整保留。

证据为boot标识1792002779.5610702与1792002779.56107相差约0.2微秒。v1 parent标识1792002780.622641相差约1.06秒，不能用本修正规则核销。新helper只接受boot相差≤1ms且host CPU计数单调不减；其他情况拒绝credit。每臂保守上界取prepared charged加全部host CPU增量，包含其他进程CPU，仍是该arm全部后续CPU的上界。

reports/taiji_r2_daily_dialogue_recovery_v2_20261021_boot_accounting_audit_v1.json以原ledger chain内prepared event sequence/SHA、snapshots、当前host counter、progress SHA、v1.2 amendment SHA及ledger tip为锚。报告仅为提案，尚未改动charges。审计时现账4,175,331.822847，修正后的host上界4,034,133.407105；4项针对测试通过，覆盖微小浮点差、真实boot变化、host计数回退和非有限值。

唯一执行路径：发布链接v1.2的v1.3来源与会计修订，明确准入的审计校正event；原事件保留，只有可复算的本extension prepared基线之后错误尾账可被追加校正。校正前验证全链、无pending/活writer、无restore/训练episode/commit。随后严格按1.5M/4.3M、原数据/episode目标恢复训练。不得删除旧event、直接覆盖summary或静默降低cap/预测安全系数。
