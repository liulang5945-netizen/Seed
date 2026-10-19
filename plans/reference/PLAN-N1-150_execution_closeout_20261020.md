# PLAN-N1-150 执行结案：匹配长上下文训练续接

**状态：训练批次完整结束；原 N1-132 训练门六个 endpoint 全部通过。`can_promote=false`，不代表留出泛化或能力晋级。**本结案绑定 N1-150、N1-150A 计分分母补充、N1-150B CPU 上限修订。

## 执行核验

- 训练前 readiness：192 个 v3 train 父项；每模型9,216个 slot。candidate 的8/16/32行分别3,072次，8行 control 为9,216次；四类各2,304次，辅助绑定项11,520次。source/schedule CPU 6.656秒，0模型加载、0训练、0查询。
- Zero bundle 前置：六份完整 bundle 保存并由独立进程恢复6/6；每 seed 的 candidate/control 初始 fingerprint 相同，模型、41个 Adam 参数状态、RNG 和 history 均核对相等。CPU57.125秒，累计63.781/18,000秒；0更新、0查询。
- 六个 endpoint 各训练9,216次，总计55,296次更新；每个 final bundle 独立恢复6/6。累计 CPU 6,956.094/18,000秒，来源及 checkpoint SHA 在执行结束时全部通过。
- 计分共7,488条：每 endpoint 原 N1-132 门960条，另有8/16/32行长度分层各96条，共288条。六个 endpoint 的全部原门均通过；每个 endpoint 的训练充分性为 true。
- 0 dev、0 holdout、0 confirmation 查询；没有改 N1-132 原门；`can_promote=false`。

## train-only 长度读数

以下是同一 v3 train 任务上的描述性结果，不构成独立泛化验证。已知类合计 lookup/revision/transition 216项；unknown 数值为字面 `unknown` 正确弃答数，分母72项。

| 分支 | 已知类正确 8行 | 16行 | 32行 | unknown正确弃答 8行 | 16行 | 32行 |
|---|---:|---:|---:|---:|---:|---:|
| 长度混合 candidate | 216/216 | 216/216 | 215/216 | 42/72 | 60/72 | 65/72 |
| 8行 control | 215/216 | 210/216 | 190/216 | 72/72 | 69/72 | 70/72 |

长度混合分支在该 train-only 描述面上保持了长上下文的已知类命中；它的短上下文 unknown 弃答读数低于 control，16/32行时差距缩小。该读数只用于描述本训练集上的长度行为。

N1-132 原门逐 endpoint 的 `free_exact` 为：seed11 candidate/control 189/192、192/192；seed12为192/192、192/192；seed13为190/192、192/192。六个 endpoint 的重命名、paired、类别与联合条件均由原 scorer 判为充分；完整字段见结果 JSON。

## 逐题日志留档偏差

训练与计分已按冻结分母完成，但 progress JSONL 没有保留全部逐题评分明细。最终文件可完整解析，共56,544条：55,296条训练更新，以及最后一个 endpoint（seed13/eight_row_control）的960条原门行和288条长度评分行；前五个 endpoint 的6,240条评分行被后续训练更新覆盖。独立审计确认最终文件没有格式错误，且结果 JSON 中六个 endpoint 的计分分母、原门摘要和长度分类汇总完整。

原因是训练时打开的 progress 写句柄一直保持旧文件位置，评分另用追加句柄写入；下一分支训练继续从旧位置写入。结案不把缺失的逐题明细称为已保存，也不重跑 N1-150 的 one-shot 训练。该运行的方案、源码、result 和现存 progress 均保留 SHA 绑定。

审计回执：[progress 完整性审计](../../reports/taiji_n1_plan150_v27_length_progress_integrity_audit_20261020.json)。

## 结果与下一步

- 完整运行结果：[N1-150 result](../../reports/taiji_n1_plan150_v27_length_result_20261020.json)；attempt：[运行 attempt](../../reports/taiji_n1_plan150_v27_length_attempt_20261020.json)。
- Readiness：[source readiness](../../reports/taiji_n1_plan150_v27_length_readiness_20261020.json)；检查点前置：[zero bundle preflight](../../reports/taiji_n1_plan150_v27_length_checkpoint_preflight_20261020.json)。六个 final checkpoint 位于计划指定目录，其路径、SHA 和独立恢复结果都写在 result JSON 中。
- 唯一下一步：批准并实施 PLAN-N1-151，对六个既有 final bundle 重评同一批 train-only 的960条原门和288条长度分层任务（合计再计分7,488条），逐题结果直接写入一个独立结果文件；不训练、不新增题面、不访问 dev/holdout/confirmation、不改门或晋级状态。此步骤只补齐 itemwise 对账并核对原聚合值，不作独立泛化或能力主张。
