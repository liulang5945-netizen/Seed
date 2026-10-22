# R2 continuation v1.1 来源修订合同

本补充仅修复真实run在训练前遭冻结base validator拒绝的组件衔接；原04架构、v1 contract、sidecar、progress及checkpoint均保持原字节。最新授权仍为累计每臂1,450,000、聚合4,200,000 process-CPU秒。

唯一方案是write-once source amendment，继续使用v1同一CPU账本和已prepare的父副本，不重新初始化。修订绑定原contract SHA、三臂attempt SHA、parent readiness/receipt SHA、发布时prepared progress SHA、固定原v1 runner/backend/两测试SHA、新runner/validator/helper/tests SHA和本补充SHA。旧来源字段保留；仅在修订记录精确匹配旧来源及当前绑定时，在内存中使用新来源身份。其他父来源、数据、谱系字段不得放宽。

发布只允许三臂均已prepared、progress只有prepare类事件、没有start/resume/restore/training/stop/final、没有pending CPU reservation、root及stage不存在writer.lock/reclaim.guard、parent副本精确且无terminal。失败不发布、不训练；修订记录原子write-through、不可覆盖。发布不更改旧progress或账本，实际修订和失败preflight成本由随后start从最后extension accounting snapshot保守累计，不能重新从原parent wall×24重启时间窗口或清零费用。

extension训练校验完整覆盖base原12项来源：builder只接受已验证compatibility report中精确parent/current SHA与Git provenance证明；其余数据、plan、manifest、readiness、reports和baseline checkpoint保持冻结原SHA。继续复用base corpus加载器、manifest/readiness的episode/family/role/target-byte不变量，仍执行frozen probe face。旧base validator及其常量不改，不monkeypatch。

真实run必须在当前来源/版本化绑定、资源、预算、吞吐和计费父checkpoint fresh-process恢复门依次通过后才能更新权重。W5b、dev/reference、selector与晋级保持冻结。独立审查和实际来源衔接回归通过前不得发布amendment或启动训练。

发布前必须在同一aggregate OS byte-range锁保护下全链回放账本，确认三臂完整、各臂≤1.45M、aggregate≤4.2M且无reservation；使用candidate amendment完整验证prepared sidecars、source绑定与checkpoint/receipt/bundle后才发布，锁保持到write-once发布完成。校验失败不得留下修订文件。prepared_progress_sha256是发布时审计锚，后续训练progress按原append-only合同延续。

同锁内还须检查已验证全量账本历史：仅允许初始化、work_unit_reserved与三类prepare结算事件。即使reservation已结清、progress尚未写入start，也必须拒绝包含start/restore/training/stop/final或未知事件的历史，防止真实执行故障窗口被误收为prepared。
