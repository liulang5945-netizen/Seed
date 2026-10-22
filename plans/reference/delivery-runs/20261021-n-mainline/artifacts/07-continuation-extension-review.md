# R2 continuation extension v1：Phase 7 独立代码审查

审查对象为预算后端、extension runner、两份合同测试；对照R2EXT-01..08、需求与架构合同。冻结提交：ad18ce7c7。

| 文件 | SHA-256 |
|---|---|
| scripts/training/taiji_r2_dialogue_continuation_extension.py | 5BC1787499BEE803A8E5310349672E40B038205CED35F55D28B0C7A7893C72D9 |
| scripts/training/run_taiji_r2_daily_dialogue_continuation_extension.py | 58A96D63186681E7CDAAA1034F4E0A68EE14B20C22050BAD214CD12EEC3EF18F |
| tests/taiji_native/test_r2_dialogue_continuation_extension_contract.py | 09E67815752B23D9DB1797B1EE640056D931F80168A851954F78E4DA68488469 |
| tests/taiji_native/test_r2_dialogue_continuation_extension_runner_contract.py | 3B25E646E1F823C025FB51FF89DE0FB3DE930E396B8AB9FA8834176E025CD39A |
| artifacts/03-requirements.md | 80D09EA0851D28A051E8F64011372C12B5B512C8C2F0938CE3538A1B40B7409F |
| artifacts/04-continuation-extension-architecture.md | 7BBBA63EAA0D7509BA14FCCED09645DCADB168CF7626F85BD8FFD3AD4595D999 |

## 门槛判断

| 级别 | 数量 | 结论 |
|---|---:|---|
| P0 | 0 | 无阻断问题。 |
| P1 | 0 | 无阻断问题。 |
| P2 | 0 | 无需实现回流的缺陷。 |

以上是冻结快照的静态代码审查结论，形成于真实run之前；后续执行回流见本报告末尾，并覆盖当前执行门判断。

独立合同测试72/72通过（33.60秒）；py_compile、Ruff check/format、Black、CLI帮助和目标文件diff-check通过。

复核确认：reservation/settlement以OS锁串行化并校验token，未结reservation按上界计入1.45M/臂、4.2M aggregate；同臂根writer lock先于账本与progress变更，死owner需PID/create-time验证、耐久归档后才能resume。父attempt、receipt/bundle、source lineage、cursor与copy-only checkpoint分支均绑定校验；每episode前reserve，bounded restore、stop reserve、全臂吞吐投影及终点结果/唯一completion ledger事件恢复均有代码与合同测试覆盖。旧证据不回写，预算拒绝不能跳过来源/readiness校验。

## 放行边界与风险

代码审查通过。随后主任务报告真实固定本地NTFS prepare-only exit 0，三臂均为prepared；准备后账本累计为1,317,124.782691 / 1,316,636.232466 / 1,315,811.157573秒，aggregate 3,949,572.172730秒。剩余训练投影186,719.457141秒，aggregate headroom约250,427.827270秒。19:41:41 CPU/内存样本早于prepare，已是历史信息；每个run仍须实时重算预算、资源及吞吐。真实父checkpoint恢复与训练尚未执行，工程审查和prepare结果都不构成模型能力证据。

真实NTFS prepare-only通过后，原静态审查允许进入受控run入口；但三臂实际run随后在来源校验衔接处拒绝，当前run门关闭。原代码路径要求的根writer lock、实时资源/预算/吞吐门及reservation保护的bounded fresh-process restore仍是后续必需门。本审查未执行真实checkpoint恢复、训练、dev/reference/selector访问或execute。

## 给下游的唯一事项

按版本化合同实现extension专用数据来源验证并加入红绿集成回归；独立复审通过后，才可在既有账本连续计费下重新进入run。

## 回流判定

初次静态代码审查通过，但真实run暴露P1来源验证集成缺口，当前回流到版本化实现与合同复审。真实prepare保留；父checkpoint恢复、训练和W5b仍未通过。不得改写v1 prepared sidecar/ledger、不得重置预算；04 v1保持原字节，新合同另行版本化。

## 19:44真实run回流：base validator兼容衔接缺口

| 当前级别 | 数量 | 结论 |
|---|---:|---|
| P0 | 0 | 拒绝发生在权重更新前，未发现证据破坏。 |
| P1 | 1 | 来源兼容层没有传递到冻结base validator，三臂run均无法进入执行。 |
| P2 | 0 | 其余项目不构成新增非阻断缺陷。 |

主任务报告三臂真实run均exit 1，错误为base._validate_sources拒绝full-turn builder的raw SHA变化。拒绝发生在根writer lock、start intent、reservation与任何训练更新之前；无权重更新，无start event。v1 prepare与账本保留，预算不重置。三次失败尝试的进程CPU必须从最后durable accounting snapshot以同boot host-CPU delta或跨boot wall×max(logical CPUs)计入后续累计，不得从旧父SHA或新budget起点重算。

代码根因：extension先由validate_source_compatibility接受精确审核过的parent/current SHA与Git blob来源，但execute_arm随后直接调用base._validate_sources()。base validator仍逐项按旧常量raw SHA硬拒绝；当前builder SHA与其parent SHA不同，扩展兼容报告没有传入该validator。触发点位于runner第2600行、open root writer store之前；旧validator的拒绝点为full-turn pretraining runner第284-286行。现有execute合同测试在runner测试第394、537、646、756及相邻用例直接monkeypatch base._validate_sources，因此72项通过没有覆盖此真实组件边界。

最小安全设计建议：保留base旧常量与默认strict validator，不调用base._validate_sources，也不monkeypatch hash。新增版本化extension validator，完整覆盖base validator的冻结文件SHA表；每项只接受精确父字节，或已审查兼容报告中parent SHA/current SHA/Git commit+blob完全匹配的条目。未跟踪data仍要求parent attempt/receipt/bundle SHA精确绑定。来源门通过后复用base._load_pretrain_corpus、base._load_assistant_train_corpus和base._read_json，并保留readiness/manifest字段、split/family/role计数及pretraining target-byte总量检查，向原probe/trainer路径返回相同数据形状；不得加载dev/reference/selector。新增helper、runner调用路径、v1账本charged-tail carry-forward与拒绝未批准源/坏manifest的红测都纳入新版本SHA及独立审查；04 v1不改。
