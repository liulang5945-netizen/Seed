# 立项结论：跨进程"单独重跑就过"的用例（四例）

**日期**：2026-11-10（2026-11-10 补第 4 例）｜**类型**：测试缺陷判定（不是噪声、不是类型改动引入）
**背景**：修类型前跑 `tests/taiji_native` 子集（167 passed / 3 failed），三条失败单独重跑全部通过。本项目把可信度压在"同字节同摘要"上，因此不接受"重跑就过"的结案，逐条判定。CI 全量回归的 junit 又暴露同一形态的第 4 条（`test_content_binding_runner_contract`），一并立项。

## 结论速览

| 用例 | 涉及量 | 两次输入是否相同 | 判定 | 处置 |
| --- | --- | --- | --- | --- |
| `test_native_content_training.py::test_fresh_process_restores_pricing_checkpoint` | `NativeRunTrainerV1.probe_digest()` 摘要 | 相同（同一 checkpoint 字节＋同一查询） | **测试写错了** | 子进程钉父进程的线程预算 |
| `test_corrected_old_baseline.py::test_zero_step_bundle_restores_in_process_and_fresh_process` | `CorrectedOldBundleV1.probe_digest()` 摘要 | 相同 | **测试写错了** | 子进程钉父进程的线程预算 |
| `test_r029_engineering_gaps.py::test_r029_worker_accounts_its_own_cpu` | `worker_cpu_seconds`／`worker_wall_seconds` | 相同（同一 bundle、同一 microstep） | **测试写错了** | 允许一个时钟 tick；子进程钉 1 线程 |
| `test_content_binding_runner_contract.py::test_fresh_process_continuation_matches_uninterrupted` | `SequenceContentTrainer` 的 `content_digest` 摘要 | 相同（同一 3 步 checkpoint、同一续跑 1 步） | **测试写错了** | 子进程钉父进程的线程预算 |

四条都**不是实现硬伤**，也都**与本次类型改动无关**（§4 给出证明）。共同根因已升格为项目判据，见[台账 D-20261110-05](DECISION-LEDGER_20261109.md)与[能力卡 §13](CAPABILITY-CARD_20261103.md)。

## 1 & 2：跨进程摘要比对

**失败形态**：父进程算 `probe_digest()`，子进程（`subprocess` 起一个全新解释器）从同一 checkpoint 重新加载后再算一次，两个字符串直接相等比较。

**失败原文（如实说明）**：最初那次子集运行的完整 pytest 输出没有落盘留存，因此无法逐字引用当时那两行 `AssertionError`。定位依据是：该子集里"父进程算一次、子进程另算一次再直接比较"的断言只有两条，且改回旧写法后按"先跑钉 1 线程的用例"的顺序可稳定复现失败——

- [test_native_content_training.py](file:///e:/Seed/tests/taiji_native/test_native_content_training.py#L227)：`assert completed.stdout.strip().splitlines()[-1] == digest`
- [test_corrected_old_baseline.py](file:///e:/Seed/tests/taiji_native/test_corrected_old_baseline.py#L333)：`assert result.stdout.strip() == expected`

复现条件：父进程被同会话早先用例钉在 1 线程、子进程从机器默认（本机 12）起步；改后（子进程钉同一预算）稳定通过。

**涉及摘要**：`probe_digest` 把 `logits` 四舍五入到 7 位小数后与答案、选择拼成一个 payload，再走 `content_digest`（SHA-256）。`content_digest` 本身是顺序无关的（Mapping／set 都排序），所以差异只能来自 logits 的末位。

**两次输入是否真的相同**：**是**。同一 checkpoint 文件、同一 `PROBE_QUERY`、同一份权重。唯一不同的是 **torch 线程预算**：pytest 会话里大量用例（`new_model`、各探针脚本）都会 `torch.set_num_threads(1)` 且**不恢复**，于是父进程在跑到该用例时是 1 线程，而 `spawn`／`subprocess` 出来的子进程从机器默认（本机 12）起步。

**判定实验（同进程、同一 checkpoint、只改线程数）**：

```
threads= 1 digest=9d227be8fa4b86d8b352c5f1805e44dbcedfe97dc9a7c3db7d77901805020c53
threads= 4 digest=9d227be8fa4b86d8b352c5f1805e44dbcedfe97dc9a7c3db7d77901805020c53
threads=12 digest=049288278a0c3c6f96cf7956c2dc408135a585d6beeb4c064e5922f52d0449da
repeat@1   digest=9d227be8…（重复同值）
```

**判定**：属于第一类——**测试写错了**。摘要是 `(权重, 输入, 线程预算)` 的函数，不是 `(权重, 输入)` 的函数：并行归约会改浮点求和顺序，7 位小数的 logits 在末位变化，摘要就变。1 线程与 4 线程同值、重复同值，说明**同预算下逐位确定**，不存在"同输入摘要随机"。

**为什么这个量本来就不该跨进程相等**：两个进程的线程预算不同，等于在两份不同的执行配置上比摘要；这测的是机器默认线程数，不是"checkpoint 是否被正确恢复"。

**改了什么**：
- 父进程在算摘要前先记下 `torch.get_num_threads()`；
- 子进程代码前插 `torch.set_num_threads(<该值>)`；
- 注释写明"摘要含 7 位小数 logits，浮点归约顺序随线程预算变，故必须钉同一预算"。

**留下的限制（必须记住）**：任何跨进程／跨机器的摘要比对都必须显式钉线程预算。所有产出报告读数的脚本（R5b 训练报告、R58–R63 探针）在模块顶部就 `torch.set_num_threads(1)`，所以**已入库读数可复现**；漏钉的只有本立项里的四条测试。

## 3：worker CPU 记账

**失败原文（本轮在同一机器上复现）**：

```
FAILED tests/taiji_native/test_r029_engineering_gaps.py::test_r029_worker_accounts_its_own_cpu
assert 0.046875 <= 0.04587229999015108
```

**涉及量**：不是摘要，是子进程自报的 `worker_cpu_seconds`（`time.process_time()`）与 `worker_wall_seconds`（`perf_counter()`）。

**两次输入是否真的相同**：**是**。同一 bundle、同一 `microstep`、同一配置；两次差异不是数据，是计时器分辨率。

**判定**：属于第一类——**测试写错了**。`time.process_time()` 在 Windows 上按 OS 时钟 tick 记账（1/64 s = 15.625 ms），一个约 40 ms 的 microstep 只能报出 0.03125／0.046875 这样的台阶值；墙钟是连续的。0.046875 对 0.04587 只是**同一个测量向上取整**，不是"worker 花的 CPU 比时间还多"。区间越短、越贴近 tick 边界，翻转概率越高。

同一用例里还有一条同类的不可比断言：`parent_cpu < worker["worker_wall_seconds"]`。父侧 `parent_cpu` 覆盖 `process.start()` **加上** 对 worker 输出做 `torch.load`，子侧 `worker_wall` 只覆盖 microstep（本机约 32–46 ms），两者不是同一个区间。

**改了什么**：
- `worker_cpu <= worker_wall` 放宽一个 tick（`_CPU_CLOCK_TICK_SECONDS = 1.0 / 64.0`），并注释说明断言的目的是"抓冒名顶替的 CPU 字段"，不是测时钟分辨率；
- `parent_cpu < worker_wall` 改为 `parent_cpu <= parent_wall` ＋ `worker_cpu < parent_wall`（worker 的 CPU 必须落在父进程整个 spawn-and-join 窗口内）；
- `spawn_worker` 改用模块内的 `_worker_entry`，在子进程里先 `torch.set_num_threads(1)` 再 `run_worker_job`——与父进程 `new_model` 的参考实现同一预算，顺带消除"多线程子进程 cpu>wall"的第二条翻转路径。

## 3b：第 4 条——`content_binding` 的跨进程续跑

**失败原文（CI junit，`test-results-3.12`）**：

```
AssertionError: assert '624f786958aa...145de645a5395' == 'da8bbaac3c4b...a038aea26ead7'
  - da8bbaac3c4bea69ab3b166606671ad6e74ac820ea23bf18e49a038aea26ead7
  + 624f786958aa7b8be7221b6a08bacc9b6f754aa63037cc0cf93145de645a5395
```

**涉及摘要**：`SequenceContentTrainer` 的 checkpoint `content_digest`（权重＋续跑两步后的状态）。父进程在 pytest 会话里连跑 2 步，子进程从同一 checkpoint 起再跑 1 步，两边各算一次摘要。

**两次输入是否真的相同**：**是**。同一 checkpoint 文件、同一批数据、同一 seed。不同之处仍只有线程预算：同一 pytest 进程里先跑的用例把 `torch.set_num_threads(1)` 留在全局，父进程以 1 线程续跑；`subprocess` 子进程是新解释器，从 runner 默认核数起步。单独跑该用例时两侧都是 12 线程，所以本地恒过——这正是"单独重跑就过"的形态。

**判定**：第一类——**测试写错了**。断言本身没写运行条件，等于拿两个不同执行配置的摘要做相等比较。

**改了什么**：父进程先取 `threads = torch.get_num_threads()`，在子进程脚本首行插入 `torch.set_num_threads(threads)`。断言、判据、期望值均未改。

**为什么与类型改动无关**：该用例的摘要路径是 `SequenceContentTrainer` 前向＋`content_digest`，本轮没有改动其中任何表达式；把 `torch.set_num_threads` 改成同一值即可在不碰那 6 个文件的情况下让摘要重合。

## 4：为什么这些失败都与类型改动无关

1. **路径不重叠**：三条摘要用例走 `native_content_core*`／`corrected_old_baseline`／`sequence_content_workspace` 的前向与序列化，摘要函数里没有任何一个表达式被本轮改过；r029 的断言对象是计时器字段，不是被改的类型面。
2. **可独立重现**：摘要差异在同一进程内仅靠 `torch.set_num_threads` 就能复现（1／4 线程同值、12 线程不同值），完全不需要碰那 6 个文件；r029 的失败差值恰为一个 1/64 s tick，与源码无关。
3. **时间顺序**：前三条失败在**改类型之前**的那次子集运行里就已经出现；第 4 条在改类型后的 CI 全量回归里出现，且单独运行恒过。改完测试后同类用例在"先跑钉 1 线程用例"的敌意顺序下 30/30 通过。
