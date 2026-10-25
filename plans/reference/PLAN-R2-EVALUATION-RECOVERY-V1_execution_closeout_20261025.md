# R2 评价工程恢复 v1 执行结案

日期：2026-10-25，Asia/Shanghai。状态：**CLOSED_FAILURE**。用户批准的一次正式恢复已实际启动并失败停止；未重试，三臂不重训。原CC-1失败与本次恢复失败分别保留。

## 实际执行与缺项

用户明确批准[恢复合同v1](delivery-runs/20261021-n-mainline/artifacts/04-evaluation-engineering-recovery-v1.md)，授权提交`c624eacf3`在启动前落盘。metadata预检通过，固定四端身份/源码与156分母一致。2026-10-25 12:14:35形成attempt `8a4dfd7ee2cb0ecbba4573892a62e139b690af182064803b`；12:14:37真实握手成功后，首个parent恢复任务在授权阶段被拒绝：

`Refuse: worker attempt/ledger path is outside the fixed evaluation contract`

该路径校验在corpus/checkpoint读取前，因此本次实际模型恢复**0/4**，query **0**、耐久输出**0/156**；未发布face/generations/review，无48回答可判。失败是工程接线缺陷，不是模型行为负结果。

## 根因与责任

上一版恢复入口只将provider/preflight及readiness路径传到worker，漏了冻结`_authorize_worker_task`仍默认绑定旧attempt/ledger；同一审查还发现`_read_worker_prompt_row`默认旧face路径。启动握手通过不覆盖这些后续路径；原合成测试替换了authorize函数，未验证真实token与自定义命名空间组合。这是本任务实现和覆盖的遗漏，不归因于模型、额度或用户批准。

本次失败发生在授权路径处，face路径缺口尚未实际触发；两者修复均在本次attempt终态之后完成，没有在运行中改绑定源，也没有刷新旧批准合同的source SHA掩盖版本变化。

## 修复与独立于正式题面的验证

- extension worker从可信provider路径表派生expected attempt/ledger，继续调用原真实授权器；face读取也显式传同一provider的固定face路径。不是把请求自报路径当允许路径，冻结evaluator与token/来源/单次claim检查不改。
- 新集成测试先复现真实旧路径拒绝，再用全合成数据走完4次恢复、156次生成；真实token授权器和face读取器保留，只替换外部模型/语料、Job成员观察及存储API边界。验证重复claim、越界attempt和异面拒绝，未加载真实checkpoint或题面。
- [真实授权探针](../../reports/taiji_r2_evaluation_recovery_v1_authorization_probe_20261025.json)进一步在实际Windows Job/IPC和本地NTFS中消费合成parent token、发布一次claim，并在模型加载前主动抛出指定sentinel。预期sentinel确认路径到达该点，随后验证树退出；0模型/题查询。它不是本次正式评价的重试或恢复。
- 探针结束时Python解释器清理输出了已忽略的Job对象析构警告；前序显式树退出和handle关闭已通过。该警告不作新模型故障，也不据它声称未执行的正常四端shutdown。

## 费用与退出

| 范围 | 已测读数与限制 |
|---|---|
| 独立validate-only | 40.703125 CPU秒，43.891 wall秒；监测进程另0.015625 CPU秒 |
| 正式失败发布时 | parent31.046875＋child1.796875＝32.843750 CPU秒，35.375 wall秒；raw仍`unknown_tail=true` |
| 正式父进程完整退出 | [退出回执](../../reports/taiji_r2_evaluation_recovery_v1_process_exit_20261025.json)为33.328125 parent CPU秒，38.032 wall秒，返回2，未触发外层watchdog；不把parent单项当子树全成本 |
| 修复后真实授权probe | 导入依赖后parent0.031250、child1.828125 CPU秒，2.141 wall秒；导入/退出未测部分留在准备预留，不宣称全进程精确成本 |

ledger的`restore_failed.worker_tree_stopped=true`，操作后进程扫描未发现本恢复入口/worker存活；未知成本标记仍按raw保留，不清零、不重计训练账。原3400历史/240准备预留、10500正式阶段额度和CPU/wall各14400同账纪律不改，本次未追加任何查询或预算。

## 原始证据与结案

机器事实位于`reports/taiji_r2_daily_dialogue_recovery_v2_20261021_extension_v1_paired_eval_recovery_v1_*`的attempt、readiness、endpoints、ledger、progress、failure、result；validation/exit/probe另外按上述链接记录。ledger/progress遵循仓库`*.jsonl`运行态忽略规则留在本机，[版本化执行审计](../../reports/taiji_r2_evaluation_recovery_v1_execution_audit_20261025.json)保留其SHA与关键事件，未强制把日志类型加入版本库。原文件不覆写；本次批准的一次资格已消费，R3–R5未完成，按工程失败出口结案，不能声称三臂比较已有结论。

唯一后继建议：先裁决“工程准入与正式题面消费是否分开”的协议边界，再决定新的有界评价包；本轮不再靠换名或刷新source快照重新启动。该讨论只涉及试验资格与工程流程，不改模型、39-family开发面权限、能力评分或既有失败历史；不把反复人工批准当作修复工程缺陷的替代品。
