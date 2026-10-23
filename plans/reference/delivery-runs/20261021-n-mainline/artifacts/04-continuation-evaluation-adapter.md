# continuation extension 配对评测适配架构 v1

## 结论
架构通过，仅准入独立适配实现与合成测试；真实评测仍由三臂完成门阻断。保持parent_baseline + full_turn_1m + warm_start + fresh_same_capacity四端点顺序、39×4=156一次调用、固定同面前12人工样例及CPU/wall各14400秒双预算。extension不可标为restart_from_zero，不修改训练绑定源码、测试、冻结报告或历史评测文件。

## 所审文件
读取本运行board、02-plan、02-continuation-extension-plan、04-continuation-extension-architecture、04-continuation-source-revision-v1-1、04-continuation-boot-accounting-v1-3；静态审查scripts/training/evaluate_taiji_r2_daily_dialogue_recovery_v2.py、prepare_taiji_r2_continuation_endpoints.py、validate_taiji_r2_continuation_endpoint.py及评测器内worker协议。未读保留题面/dev/reference，未执行模型、评测或训练。

## 最小适配边界
新增独立版本化extension evaluator入口及契约测试，复用既有受监督Job Object、durability、one-shot claim、IPC与生成逻辑；显式依赖注入endpoint metadata provider和独立output paths。不得运行时monkeypatch旧模块全局、放宽旧restart校验或构造伪旧result。若共享worker必须改协议，新增带版本的worker入口/dispatcher，使parent与worker使用完全相同的extension provider，保留旧worker分支原合同。

当前旧评测器_arm_endpoint校验restart_from_zero=true，worker restore又内部直接调用旧metadata_preflight及旧_evaluation_paths；因此仅把extension manifest传给父进程或改路径不足以适配，必须同时调整独立worker授权与restore路径。旧复原语义仍用assistant train corpus与完整trainer checkpoint，禁止参数部分载入或丢弃历史冒充恢复。

## 接口与证据映射
|边界|具体输入与验证|适配输出|
|---|---|---|
|metadata-only provider|prepare_taiji_r2_continuation_endpoints.metadata_preflight；三臂result/readiness齐全、完整aggregate链验证、无reservation/活writer、每arm唯一continuation_completed|extension-endpoint snapshot，0prompt/0模型|
|父baseline|沿用旧评测器固定baseline provenance、文件SHA、checkpoint/model digest、corpus绑定|endpoint_id=parent_baseline；基线既有0/0恢复合同不改|
|extension三臂|validator核验extension schema/arm/complete、result→readiness SHA、parent attempt/receipt/bundle、两stage final路径SHA及恢复receipt、30096/14517912与525/113209|endpoint_kind=continuation_extension，origin=resume_from_verified_parent；保留原训练血缘，禁止restart_from_zero=true|
|launch provenance|原attempt不可变SHA、current/source_revision_v1_3与v1.1→v1.2→v1.3修订链、实际launch时runner/module/test SHA及精确Git/external分类兼容证据|source_hashes分为parent bindings、launch bindings、current eval bindings，精确映射变化而非要求全等或忽略漂移|
|会计证据|原ledger全链含获准boot校正事件、唯一completion event的extension_id/arm/parent SHA/details.result/readiness SHA；tip与charges|同时保留result发布前下界、completion较晚下界、reservation上界；不把修正credit当负训练CPU或零旧账|
|endpoint map|parent+三arm，固定path、sha256、checkpoint_digest、model/corpus配置、阶段计数、parent lineage与completion SHA|独立schema taiji-r2-continuation-evaluation-endpoints-v1；canonical snapshot SHA，前后重验|
|worker授权|独立attempt/ledger/readiness绝对固定路径、schema+adapter_version、endpoint snapshot SHA、task_id、单次token SHA、operation/index/prompt或checkpoint SHA、durable intent与claim|Job membership及父监督确认后才restore/generate；不准旧路径回退，不准任意checkpoint请求|
|worker restore|再次运行同一extension metadata provider，比对attempt/readiness/report/source快照；绑定authorized checkpoint SHA|complete/equal、endpoint_id/stage/path/SHA、checkpoint/model/corpus digest、525/113209（parent仍0/0）、active_parameters、0更新/查询、测量CPU；另带endpoint_kind与lineage snapshot SHA|

manifest准备器是元数据门，不是worker完整恢复门。当前manifest未自带完整launch字段与parent baseline映射，适配provider须补齐并绑定原证据，不能凭manifest名字推断来源。训练预算以当前获准v1.3每臂1.5M/aggregate4.3M为实际执行合同；先前1.45M/4.2M仅历史，不硬编码为当前预算。精确launch身份须来自实际launch/source amendment报告与ledger事件，而非当前工作区SHA猜测。

## 一次性与资源合同不变
独立输出前缀与schema标extension evaluation；共享同一39-family消费面锁/旧attempt检查，防止换入口或换文件名获得第二次查询。旧评测已消费则拒绝新执行，失败同样消费，不自动重试。启动前恢复四端点且receipt全部有效后才触及selector；读取prompt/history不读reference。每次query_started先耐久落盘，再一次生成并逐条fsync，完成156且39个唯一family。固定前12同family人工复核中文可读性/贴题/历史连贯/重复，描述统计不作语义通过。

CPU≤14400且wall≤14400，包含metadata/source校验、启动/四端恢复、Job树、IPC、生成、终止/失败、已有该评测生命周期预检与结果发布reserve；训练累计账另列，不从训练cap凭空借额度。cost_receipt明确结果/receipt发布排除与预留边界，unknown退出成本不得当0。超限或无法确认worker/IPC终止即停止留证，不发下一题。

## 未完成项与给下游
唯一下一步：实现独立extension adapter及worker provider贯通的合成契约测试，尚不运行真实metadata准备、selector或生成。覆盖旧restart结果仍严格、extension不伪造zero、parent+三臂计数、launch修订链/boot校正链漂移、重复completion/活writer/pending拒绝、worker拿旧readiness/错adapter路径/错token拒绝、恢复receipt错谱系拒绝、39×4及固定12、one-shot跨入口消费、双预算与失败树终止。实现后独立代码审查；三终点完整、来源门、fresh restore及资源预算全部通过后才可执行。

## 回流判定
- 结论：通过（仅架构/实现准入，不放行真实评测）
- 回流目标：无
- 返工原因：无；worker硬编码及launch证据补齐是本阶段定义的下游交付。

## 2026-10-23 实施状态（CC-1.2/1.3 收口）

- 独立入口、versioned worker、适配层已落地并通过合成合同（`612c2dfe4`／`b1817cbcf`／`9ec663c25`／`112caed7b`／`d39f6c8af`／`1fc8bac40`）；冻结 recovery-v2 评测器未改。
- launch provenance、aggregate ledger tip/charges、批准额度已绑入 extension 快照；端点清单与 worker 恢复收据携带 `endpoint_kind` 与 `lineage_snapshot_sha256`；评测源码闭包（冻结源＋extension seam）进入 `source_hashes`。
- 训练与评测预算以 v1.4 **1,550,000 CPU秒/臂、4,400,000总计**为准；本文第26行 v1.3 数值为历史。
- 真实评测仍待三臂完整终点；下一步为 CC-1.5 D1 材料（缺口、总成本与唯一推荐）。
