# 11.7 候选仪器读数：重复运行、峰值内存、p50/p95 与评分路径副作用（不授予训练）

> 文档状态：[合同] — 11.4 要求核对"来源、scope、缺权重、原始生成、重复运行、中断恢复、峰值内存及 p50/p95"，11.1 要求"冻结评价禁写"。本合同只把这两句变成读数；**不训练、不改默认、不发权重**。

日期：2026-11-01。候选＝09.8 主体（父摘要 `921dc998…`），路径＝10.7 的判定打分（`taiji/action_verdict_scorer.answer_nll`）。

## 1. 测量（四件）

1. **重复运行**：同一前缀连续 20 次打分，记录每次墙钟；第 1 次与第 20 次的返回值必须**逐位相同**。
2. **p50／p95**：按最近秩（nearest-rank）从 20 次墙钟算出。
3. **峰值内存**：进程 `PeakWorkingSetSize`（Windows `GetProcessMemoryInfo`；非 Windows 用 `ru_maxrss`），起步记一次、20 次后记一次，报增量与峰值。
4. **评分路径副作用**：打分期间挂钩 `builtins.open`／`io.open`／`socket.socket`／`subprocess.Popen`／`os.system`，记录调用。**允许**：只读打开、`__pycache__` 写入。**不允许**：其他写模式、任何 socket／subprocess／os.system ⇒ `unsafe_side_effect`。

## 2. 判据与出口

- 20 次里任一次抛错 ⇒ `engineering_failure`。
- 第 1／第 20 次返回值不逐位相同 ⇒ `nondeterministic_output`。
- 出现不允许的副作用 ⇒ `unsafe_side_effect`。
- 否则 `instrument_measured_quiet`；超预算 `budget_stop`。
- 判定优先级：`budget_stop` ＞ `unsafe_side_effect` ＞ `nondeterministic_output` ＞ `engineering_failure` ＞ `instrument_measured_quiet`（一次运行同时触发多项时按此取最严重）。
- 预算：≤240 CPU 秒／480 墙钟秒。

## 3. 可声称与不可声称

- **可声称**：在这台机器、这一个进程、这条打分路径上，20 次打分逐位可重复；p50/p95 与峰值内存有读数；评分期间没有写仓库文件、没有联网、没有起子进程。
- **不可声称**：不表示主体"安全"（它没有动作通道，这条面上的风险面本来就小）；不表示其他路径（训练、生长、恢复、迁移）无副作用；不表示 p50/p95 与内存可外推到其他设备或更大输入；不表示任何任务能力。**训练仍未准入。**

## 4. 交付

本合同、`scripts/training/verify_taiji_instrument_11_7.py`、`tests/taiji_native/test_instrument_11_7.py`、`reports/taiji_core_instrument_11_7_20261101.json`、11 准入报告的 v2（把 11.7 读数并入 P3 的安全项）、MAINLINE／03／索引同步。
