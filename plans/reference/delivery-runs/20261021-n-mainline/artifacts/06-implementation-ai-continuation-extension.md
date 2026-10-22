# Phase 6：R2 continuation extension v1 AI runner实现

日期：2026-10-22。结论：runner实现与合成验证完成；真实恢复和训练仍未放行。

## 交付

- 新入口：`scripts/training/run_taiji_r2_daily_dialogue_continuation_extension.py`
- 合同测试：`tests/taiji_native/test_r2_dialogue_continuation_extension_runner_contract.py`
- 预算后端由并行 backend 子项提供：`scripts/training/taiji_r2_dialogue_continuation_extension.py`

runner使用独立的 `--prepare-only`、`--run --arm <arm>` 与 `--resume --arm <arm>` 模式。全局prepare只初始化一次三臂contract和aggregate ledger，校验parent身份与source compatibility，使用版本化checkpoint root逐字节复制parent latest bundle/receipt；旧attempt、progress、result、checkpoint和source hash保持只读。已有或不完整sidecar不会被重建或覆盖，验证不完整时fail closed。

每个bounded work unit通过后端原子guard取得reservation token，settlement必须使用同token。prepare阶段逐臂预留120 CPU秒；fresh-process restore具有120秒硬deadline，并额外预留2秒账本settlement开销。worker失败、超时或receipt无效时，runner记录host CPU delta；失败收敛用实际累计charge和活动reservation upper中较大值结算。resume先全链审计，未结reservation按有效上界记录 `continuation_resumed` 后才申请新工作额度。终点readiness/result写入由30秒final-publication预留覆盖，completion event在result耐久发布后记录较晚的累计charge下界；result明确标注其CPU字段的pre-publication边界。

训练分母与stage顺序沿用冻结合同。全过程dev/reference/generation查询计数为0，`can_promote=false`；代码实现、准备或后续训练均不能作为能力提升或W5b放行证据。来源兼容性使用固定7项parent/current SHA与Git revision表，漂移或无法解释时拒绝执行。

## 验证

- runner + backend synthetic tests：20/20通过；其中runner测试12项覆盖source drift、父checkpoint副本逐字节一致、prepare幂等、reservation token settlement、resume恢复、fresh-process失败/超时及未结上界计费。
- `py_compile`、Ruff、Black check通过。
- CLI `--help`展示三种互斥模式及单臂选择；没有执行prepare、恢复或训练命令。
- 未加载或恢复真实checkpoint，未访问dev/reference/selector，未运行`--execute`，未启动训练。

## 风险与下游

本次只完成实现岗位子项。runner/backend仍需独立代码审查，并在后续准入时重新计算wall×24未知尾账、核对固定NTFS durability、父来源兼容性、资源与可用预算；当前文档中的历史CPU快照不能代替执行前实测。若任一门失败或下一有界工作单元reserve不足，必须停止且保留账本，不得重置累计额度。

唯一下一步：由独立code-reviewer审查runner与账本集成及失败收敛合同；审查通过前不启动真实续训。

## 18:02 实现修订与最终复审准备

用户于17:50将当前累计窗口批准为每臂1,300,000、三臂聚合3,900,000 process-CPU秒；旧实际charge与wall×24未知尾账继续计入，未清零。

在原Phase6初版基础上，当前冻结实现新增了：

- Git-tracked source用parent/current SHA、commit/blob OID和Git clean-filter证明；Git未跟踪的`data/`输入仅在当前SHA与parent SHA精确相同时，以parent attempt/receipt/bundle谱系绑定。
- 本地固定NTFS目标卷校验与write-through发布覆盖sidecar、progress、parent fork、RecoveryStore checkpoint/receipt/latest、final checkpoint和aggregate ledger。
- 合同写入后初始event缺失、部分prepare、parent copy中断和ledger已settle但progress缺失均可在字节校验后幂等补齐；不覆盖既有内容或重复收费。
- completion readiness/result已发布而ledger/progress未结算时幂等收敛；final restore worker有120秒deadline；guard预留terminal stop空间并持久化预算/速率拒绝；按parent与扩展运行历史作滚动CPU/episode投影，只有剩余预算覆盖全部剩余work时才派发下一有界单元。
- 32-episode checkpoint边界synthetic续跑验证父cursor后下一episode、RNG及训练终点摘要等价，并在latest已durable而进度事件缺失后恢复时不重训已提交episodes。

冻结候选验证：backend+runner联合测试 **58/58 passed**；Ruff check、Ruff format、Black、`py_compile`、CLI `--help`及`git diff --check`通过。18:02只读parent audit的source binding为26/29/29项通过；当前保守累计上界为1,170,919.559/1,170,469.115/1,169,679.911秒，aggregate 3,511,068.585秒。以上不代表真实extension `prepare-only`、checkpoint restore或训练已启动。

唯一下一步：由独立Phase7 reviewer按冻结runner/backend/test/architecture SHA审查P0/P1/P2；通过后立即刷新预算和资源，运行NTFS `prepare-only` durability门，再从三臂各自原checkpoint并发续训。`can_promote=false`、W5b与N1-166冻结维持。


## 最新累计预算与恢复验证

最新用户批准同一窗口每臂1,450,000、三臂总计4,200,000 process-CPU秒，覆盖此前1.38M/4.05M及更小窗口；旧实耗与未知尾账持续累计。runner与backend共用该批准常量。

恢复路径已改为先持有根writer lock再进行账本/进度修改；writer.lock与reclaim.guard均使用完整JSON的原子write-through发布。死owner归档、guard恢复及发布失败均保留精确来源证据，OS自动释放互斥锁保护guard恢复。组合故障回归覆盖死writer归档成功、替代writer发布失败、guard清理后仅凭归档证据再次恢复。活owner拒绝路径不修改账本和progress。

当前backend+runner合成测试72/72 passed（33.09秒），Ruff通过。合成恢复测试包含真实父游标形状9472/epoch2/index1948与32-episode提交/恢复等价；这不代表实际parent restore或模型能力已通过。

result/readiness中的累计CPU为各自发布前的下界。消费者应在已验证的append-only账本中按extension_id、arm、parent_attempt_sha256及details.result_sha256/readiness_sha256定位唯一continuation_completed事件；该事件charged_cpu_seconds是更晚的累计下界，仍不含自身事件写入及进程退出CPU。reserved_cpu_seconds为预留上界，不能作为实际消耗。progress的ledger序号与hash用于定位，账本为权威来源。

19:19历史只读快照aggregate为3,805,047.806345秒、剩余训练投影186,719.457141秒；执行前必须重新计算。此前正式资源门仅7.50GiB可用内存而拒绝，不能据此声称资源门已过。当前仍待独立审查、实时资源/预算检查及本地NTFS prepare-only保存门；尚未启动真实extension训练。

19:31:42刷新：各臂保守累计1,300,114.326747/1,299,649.842946/1,298,841.390844秒，合计3,898,605.560537秒；当前4.2M余量301,394.439463秒，剩余训练预测186,719.457141秒，全部臂与aggregate投影可行。正式资源门CPU15.9/18.5%、可用内存27.669GiB通过，开跑仍须再次刷新。
