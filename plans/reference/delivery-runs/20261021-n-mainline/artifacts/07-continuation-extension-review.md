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

独立合同测试72/72通过（33.60秒）；py_compile、Ruff check/format、Black、CLI帮助和目标文件diff-check通过。

复核确认：reservation/settlement以OS锁串行化并校验token，未结reservation按上界计入1.45M/臂、4.2M aggregate；同臂根writer lock先于账本与progress变更，死owner需PID/create-time验证、耐久归档后才能resume。父attempt、receipt/bundle、source lineage、cursor与copy-only checkpoint分支均绑定校验；每episode前reserve，bounded restore、stop reserve、全臂吞吐投影及终点结果/唯一completion ledger事件恢复均有代码与合同测试覆盖。旧证据不回写，预算拒绝不能跳过来源/readiness校验。

## 放行边界与风险

代码审查通过。随后主任务报告真实固定本地NTFS prepare-only exit 0，三臂均为prepared；准备后账本累计为1,317,124.782691 / 1,316,636.232466 / 1,315,811.157573秒，aggregate 3,949,572.172730秒。剩余训练投影186,719.457141秒，aggregate headroom约250,427.827270秒。19:41:41 CPU/内存样本早于prepare，已是历史信息；每个run仍须实时重算预算、资源及吞吐。真实父checkpoint恢复与训练尚未执行，工程审查和prepare结果都不构成模型能力证据。

prepare门通过后允许进入受控run入口，但每臂必须先获取根writer lock，实时通过资源/预算/吞吐门，并在保留reservation下由bounded fresh-process restore核对latest receipt；任何一门失败都不得进入训练更新。本审查没有执行真实checkpoint恢复、训练、dev/reference/selector访问或execute。

## 给下游的唯一事项

在实时资源、预算与吞吐门通过后，启动三臂run入口并遵守各臂根锁、计费父checkpoint恢复及逐episode预算守卫；未通过的arm立即停止。

## 回流判定

无代码或需求回流。Phase 7审查与真实prepare门均通过；Phase 8主指针保持不变。父checkpoint恢复、训练和W5b仍未通过。
