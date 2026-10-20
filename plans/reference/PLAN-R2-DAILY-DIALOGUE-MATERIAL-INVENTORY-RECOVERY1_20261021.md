# PLAN-R2-DIALOGUE-05R1：材料清点报告来源与盲区修复

**状态：已冻结；替代原DIALOGUE-05清点attempt作为有效读数。** 原计划、数据边界与300秒CPU上限不变。

## 已知初稿问题

初稿`reports/taiji_r2_daily_dialogue_material_inventory_20261021.json` SHA-256为`19ec1cad3d98b1802066f275c8a96224ce0fd2acb4091e284ae50094362d0280`，但报告绑定的runner SHA为`142cfe92f1867374e36e66003cdb016965b51285e006380b16e3ac9489a1e04a`，当前修订runner SHA已不同。初稿代码还对所有source行调用完整`json.loads`，因此dev/final文本曾进入清点进程内存；没有打印、保存、评分或送入模型，但这违反了DIALOGUE-05更严格的“dev/final只读family元数据”冻结边界。初稿保留追溯、不得作为正式数据审计结论；不删除、不覆盖。

## 最小恢复

- 输出移到独立`recovery1`路径，绑定本恢复方案、脚本、所有输入和初稿SHA；对旧final树只计metadata，绝不解码文本字段；dev仅计family metadata，不访问文字字段。
- 对train树仍严格复用原OASST2 `ready_for_export`、review、deletion、synthetic、PII/安全flag、邮箱/电话和reserved-marker过滤；只输出计数、rank/长度分布、hash，不落原文、消息ID或用户ID。
- 新回归测试覆盖脚本从仓库外启动及带默认空格JSON的episode元数据解析。源码检查、格式检查、编译与测试全部通过后执行一次。
- 初稿CPU已耗12.53125秒；恢复运行与初稿合计不得超过300秒。0模型加载/查询、0训练/更新、0checkpoint写入；新报告如仍未完成则停止，不再重试。

## 产物

- 修复后runner：`scripts/data_prep/audit_taiji_r2_daily_dialogue_material_inventory.py`
- 合同测试：`tests/taiji_native/test_r2_daily_dialogue_material_inventory_contract.py`
- 有效报告：`reports/taiji_r2_daily_dialogue_material_inventory_recovery1_20261021.json`
- 中文结案：`plans/reference/PLAN-R2-DAILY-DIALOGUE-MATERIAL-INVENTORY-RECOVERY1_execution_closeout_20261021.md`
