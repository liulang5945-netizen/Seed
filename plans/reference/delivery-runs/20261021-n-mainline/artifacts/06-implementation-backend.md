# Phase 6 补充：R2 continuation extension v1 账本实现

日期：2026-10-22
角色：backend-architect
范围：只实现CPU预算账本与路径契约；未启动训练，未读取dev/reference或selector，未改旧attempt、checkpoint或result。

## 结论

后端账本子模块已完成，供新continuation runner集成。它把父三臂累计CPU上界固化为write-once合同；每个有界工作单元开始前，`guard`在跨进程原子锁内检查当前charge与所有活动reservation，写入保留事件并返回token；settlement必须持同一token，在一个durable事件中写入累计CPU并释放reservation。首次实现使用每臂1,000,000秒/聚合3,000,000秒；同一续训窗口随后修订为1,200,000/3,600,000，并于2026-10-22 17:50按用户批准更新为当前每臂1,300,000秒、三臂聚合3,900,000秒。旧charge与未知尾账保留且不可重置。父charge必须完整，来源是闭合值或按冻结公式算出的保守上界；未知或部分尾账拒绝初始化。

该实现只完成Phase 6 backend子项，不代表新runner、端到端恢复或真实训练门通过。新runner仍须在任何parent restore前调用`initialize_ledger`，并在每个恢复/训练进程启动时调用`latest_charges()`完整校验事件链。

## 接口实现清单

模块：[taiji_r2_dialogue_continuation_extension.py](../../../../../scripts/training/taiji_r2_dialogue_continuation_extension.py)

| 接口 | 责任 |
|---|---|
| `paths(arm) -> dict[str, Path]` | 给出parent attempt/readiness/progress/result/checkpoint root、extension attempt/progress/result/checkpoint root、全局合同和aggregate ledger路径；拒绝未知arm。`aggregate_ledger_path`是编号事件文件的公共名前缀。|
| `initialize_ledger(parent_records, source_report_sha256, approved_utc) -> dict` | 要求三臂parent记录齐全，验证父attempt/receipt/bundle哈希、累计CPU、tail状态和CPU/boot/wall快照；依次以write-through写出全局合同、初始化事件和派生summary。合同或账本已存在时拒绝重启建账。|
| `latest_charges() -> dict[str, float]` | 完整校验append-only事件链，返回每臂已结charge加未结reservation上界；必要时重建落后的派生summary。缺项、序列间隙、哈希链不符或temp残留均拒绝。|
| `latest_reservations() -> dict[str, dict]` | 完整校验后返回每臂活动reservation的`reservation_id`与`reserved_cpu_seconds`；进程恢复时用于将未结上界按`continuation_resumed`结算。|
| `append_charge(arm, event, charged_cpu_seconds, details) -> dict` | settlement必须在`details["reservation_id"]`提供该arm唯一活动token。一次durable事件原子更新实际累计charge并释放reservation；缺token、错臂、旧token或重复settlement拒绝。|
| `guard(arm, next_charge, reserve_cpu_seconds) -> str` | 在跨进程锁内原子检查charge+所有活动reserve，拒绝已有同臂活动reservation或任何per-arm/aggregate超额；write-through追加`work_unit_reserved`后返回32位hex token。调用者须在任何work unit前调用，并用同token settlement。|

事件数据写在既有`reports/`目录的编号JSON文件中，例如`..._aggregate_cpu_ledger.00000000.json`、`.00000001.json`；该目录本身不由模块创建。摘要文件仅为可重建的派生索引，事件文件才是账本事实源。合同、初始化事件及每个后续事件以独占同目录临时文件、flush、`FlushFileBuffers`与`MoveFileExW(MOVEFILE_WRITE_THROUGH)`发布。运行时只接受本地固定NTFS；网络、ReFS、可移动盘和未经验证文件系统fail closed。

初始化时，每臂parent记录至少有`parent_attempt_sha256`、`parent_receipt_sha256`、`parent_bundle_sha256`、`charged_cpu_seconds`、`tail_status`与`accounting_start`。`tail_status`必须是`closed`或`conservative_upper_bound`；`accounting_start`必须包括`snapshot_utc`、`boot_identity`、`host_cpu_seconds`和`logical_cpu_upper`。新额度只作总累计上限，不清零旧账，也不因恢复次数增加。

## 数据模型与迁移

- 全局write-once合同schema：`taiji-r2-dialogue-continuation-extension-v1-budget-contract`。合同包含批准日期、来源报告SHA、三臂parent identity、起始charge、1M/臂与3M聚合上限、文件系统/发布资格和合同digest。
- append-only事件schema：`taiji-r2-dialogue-continuation-extension-v1-budget-event`。首事件记录三个起始charge；`work_unit_reserved`事件原子保存arm、累计charge、reservation id和reserve upper；settlement事件必须引用活动token、记录实际charge并释放reserve；所有事件包含parent SHA、序号和前序digest。
- 派生summary schema：`taiji-r2-dialogue-continuation-extension-v1-budget-summary`。它包含每臂settled charge与活动reservation，由完整事件链重建；未结reservation在`latest_charges()`中按reserve upper计入。summary缺失或落后时可以恢复，summary失配或指向不存在事件则拒绝。
- 未迁移、未修改任何旧schema或旧训练文件。extension runner须把自己与兼容性报告的SHA另记入extension manifest。

## 自测结果

新增合成合同测试：[test_r2_dialogue_continuation_extension_contract.py](../../../../../tests/taiji_native/test_r2_dialogue_continuation_extension_contract.py)。

- `python -m pytest --basetemp .seed_test_tmp/r2-extension-budget tests/taiji_native/test_r2_dialogue_continuation_extension_contract.py -q`：8 passed。
- Ruff检查：通过；Ruff格式检查：通过；`py_compile`：通过。
- 覆盖路径契约、write-once初始化、单调charge、错token/重复settlement拒绝、未结reserve重启保守计费与显式reconcile、未闭合尾账拒绝、per-arm上限、两个独立进程对aggregate headroom并发争用（唯一序号、一方获批、一方拒绝）、temp残留和事件digest漂移。
- 测试注入durability adapter验证发布顺序；真实路径运行时仍须现场探测NTFS并用Win32 API进行write-through，不把合成测试伪称为断电测试。

## 遗留风险

- 尚未与extension runner联调；runner必须确保prepare-only只初始化一次，并在任何parent恢复或训练前完成耐久意图写入。
- 每次启动/恢复需先调用`latest_charges()`和`latest_reservations()`做全链核验；每个episode或restore前用`guard()`获取token，结束后立刻用该token调用`append_charge()`settle；同一budget对象再次guard前必须先结清上一token。进程中止后的未结reservation在resume时按上界入账并由`continuation_resumed`显式释放。
- 锁、write-through和summary恢复需在目标固定NTFS目录进行启动前验证；本子项未验证真实checkpoint恢复、训练吞吐或最终结果。

## 给下游的唯一事项

由ai-integration runner调用本模块完成一次三臂全局建账；随后每臂复用相同账本，写各自extension sidecar与版本化checkpoint分支。runner实现和独立代码审查通过前不得启动真实训练。

## 回流判定

- 结论：backend账本子项通过，允许进入runner集成；不放行真实训练。
- 回流目标：无。
- 返工原因：无；runner集成与独立复审仍未完成。
