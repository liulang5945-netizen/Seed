# PLAN-R2 续训配对验证（CC-1）执行结案

日期：2026-10-25，Asia/Shanghai。状态：**CLOSED_FAILURE（工程失败封存）**。本轮按 [03](../../active/roadmap/03_CURRENT_EXECUTION.md) 的 R1–R6 结案；工程故障不写作模型能力负结果。

## 结论

三臂续训全部到达原终点，训练侧 R1/R2 完成；四端点准入（R3）通过并持久化；**一次性 W5b（R4）在受监督 worker 启动握手前失败，0/156 输出**，按原 one-shot 合同消费一次资格、不重试。失败根因是我方 worker 入口接错模块（库模块没有 `__main__`），属可定位、可修复的工程缺陷，不是模型行为或语言能力结果。R5 无可判读输出，未执行。

## R1–R6 实际状态

| 条件 | 状态 | 证据 |
|---|---|---|
| R1 三臂到原终点 | ✅ 完成 | 每臂 30,096 episodes / 14,517,912 updates＋525 / 113,209；3 份 result、3 份 readiness、6 个阶段 final，独立恢复均 `complete=true`、`equal=true` |
| R2 身份、终态与费用闭合 | ✅ 完成 | 各臂唯一 `continuation_completed`；无活动 reservation、无活 writer；v1.5 修订链 `21e0497e` 绑定 sequence 66,699 |
| R3 四端点准入 | ✅ 完成 | `…_extension_v1_endpoint_manifest_v1.json` 与 `…_paired_eval_endpoints.json`、`…_paired_eval_readiness.json`；`can_query_dev=false`、`can_promote=false`；端点顺序 parent_baseline→full→warm→fresh；`expected_generation_outputs=156` |
| R4 一次固定 W5b | ❌ 工程失败 | attempt `abffec0c…e0080e` 已形成；`query_attempts=0`、`persisted_generation_outputs=0`；`worker exited before authenticated bootstrap-ready handshake` |
| R5 固定人工判读 | — 未执行 | 无 48 个端点回答可判；不以空白表冒充判读 |
| R6 结案落盘 | ✅ 本文件 | 汇总完成项、缺项、费用、失败与唯一后继建议 |

## 失败根因与已落地的修复

- 现象：worker 于 11:18:01 启动、11:18:03 退出，父进程未收到 `ready` 握手；失败记录 `WorkerTaskFailure`，`unknown_tail=false`、`worker_tree_exit_unverified=false`、`ipc_writer_exit_unverified=false`。
- 根因：`taiji_r2_continuation_evaluation_adapter.WORKER_MODULE` 指向 worker **库模块** `evaluate_…_worker_v1.py`，该文件只有函数、没有 `__main__` 入口；`python -m` 因此立即静默退出（退出码 0，无 stdout）。实测 `python -m scripts.training.evaluate_…_worker_v1` 无任何输出、退出 0，确认假设。
- 修复：`WORKER_MODULE` 改为版本化入口 `scripts.training.evaluate_…_extension_v1`（其 `--worker` 分支才是文档化的私有 IPC 循环，参数与命令完全匹配）。冻结评测器源码未改。
- 回归：新增 `test_worker_command_module_has_a_runnable_entry_point`（对命令里的 `-m` 模块执行 `--help`，要求退出 0 且出现 usage）；adapter 合同 20 支通过。该修复只阻止同一缺陷复现，**不构成对已消费 one-shot 的重跑授权**。

## 费用

- 训练累计（v1.5 口径）：fresh **1,410,277.610698** ＋ full **1,521,529.860816** ＋ warm **1,580,449.419966** ＝ **4,512,256.891** CPU 秒，占 4,560,000；训练余量 **47,743.109**。无未知尾账加记。
- 评价：本次 one-shot 实测父进程 **59.094**＋子进程树 **1.688** ＝ **60.781** CPU 秒、**63.95** wall 秒，均为独立 14,400 秒额度内；`cpu_measurement_valid=true`。attempt 形成前记账 59.031 秒。
- 口径说明：发布前 receipt 不等于进程退出全成本；本轮不把上述数字扩写为未实测成本。

## 消费范围与不可重试

已落盘：`…_paired_eval_readiness.json`、`…_paired_eval_endpoints.json`、`…_paired_eval_attempt.json`、`…_paired_eval_ledger.jsonl`、`…_paired_eval_progress.jsonl`、`…_paired_eval_failure.json`、`…_paired_eval_result.json`。
未产生：face、generations、cost_receipt、review。

按合同：attempt 一旦形成即视为消费一次资格；失败不重试、不换入口/名字、不补题、不续跑，也不得通过改名重开一次性消费。下一轮如需语言能力读数，必须作为**新的有界包**重新登记合同、预算与停止线，不能沿用本次资格。

## 能力判定

本轮**不能**给出“新配方改善/无收益/退化”的结论：生成输出为 0，判读分母为空。可陈述的只有：三臂在原终点上具备完整、可恢复的训练终态；评价链路的准入与元数据已跑通，worker 启动接线缺陷已被定位并修复。旧能力保持（warm/fresh/full 各自的旧配方对照）未在本轮被测量，不应推断。

## 唯一后继建议

进入 **CC-2/VX0**：先整理已消费证据与错误表，再选择唯一必要诊断（信息是否进入决策路径）；新实验作为独立有界包登记问题、对照、固定分母、完整预算与停止线，不在本轮追加训练或复用已消费评价资格。
