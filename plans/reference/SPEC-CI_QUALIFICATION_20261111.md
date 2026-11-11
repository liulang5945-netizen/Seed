# SPEC：CI 8 条红中 5 条的定性 + 修复（2026-11-11）

**状态**：待批准
**范围**：只碰 CI 判据与 3 个测试文件、1 个 runner、2 个关系契约测试。不跑正式训练、不读 confirm、不动无关冻结读数。
**不纳入本轮硬目标**：620 份报告豁免、278 条 mypy advisory、158 次 deselect 全量恢复（deselect 只做清单与「顺手恢复」）。

---

## 1. 实测证据（原始读数，不替代）

### 1.1 本机重跑（Windows，HEAD 4bb053886，同一 pytest 会话，r022 先把线程钉到 1）

命令：`pytest test_living_core_credit_r022.py test_native_content_core.py test_ordinary_baseline.py test_conditional_language_candidate_v3.py test_native_content_r5_runner.py -p no:cacheprovider -q`

结果：`1 failed, 36 passed in 128.27s`

- 失败原文：`tests\taiji_native\test_native_content_r5_runner.py:62: AssertionError` / `assert 0 == 1`（`record["steps_executed"]` 为 0）
- 另外 4 条（3 条 checkpoint 往返 + r022）在本机通过。

### 1.2 CI 原始失败行（run 42979760915，`gh run view --log-failed`）

| 测试 | 行 | 原始断言 |
| --- | --- | --- |
| `test_conditional_language_candidate_v3.py` | 200 | `assert '70f81d0f18a0…' == '999d315d509a…'` |
| `test_native_content_core.py` | 360 | `assert '206dda585edd…' == '82e642bd338f…'` |
| `test_ordinary_baseline.py` | 122 | `assert '5f32a4733352…' == '13fc9cce068c…'` |
| `test_living_core_credit_r022.py` | 859 | `assert False is True` |
| `test_native_content_r5_runner.py` | 62 | `assert 0 == 1` |

三条 checkpoint 测试的失败点都只落在「子进程再算一次摘要」这一行；同一条测试里「同进程 `load` 后摘要相等」的断言（v3:198、core:343、ordinary:105）在 CI 也通过。

---

## 2. 逐条判定

### 2.1 三条 checkpoint 往返（v3 / native_content_core / ordinary_baseline）：判据本身写错

**证据**：失败行全是 `result.stdout.strip() == expected`，即父进程摘要 vs 全新解释器摘要。`probe_digest` 把 logits 四舍五入到 7 位小数后做 SHA-256；torch intra-op 线程预算改变浮点归约顺序，末位可跨过舍入边界，摘要随之改变。

**既有判据（不重复发现）**：`plans/reference/FINDING-CROSS_PROCESS_EVIDENCE_20261110.md` 已用同一模型实测：`threads=1` 与 `threads=4` 摘要相同（`9d227be8…`），`threads=12` 不同（`04928827…`）；并已按「子进程钉父进程线程预算」修过 4 条同类测试。台账 `D-20261110-05` 已把「跨进程摘要比对必须显式钉线程预算」升格为项目判据。

**本机为何通过**：本机这三组配置在 1 vs 12 线程下 7 位小数恰好一致；这是运气，不是判据成立。CI（Ubuntu，不同 BLAS/核数）越过了舍入边界。

**判定**：判据本身写错（不是实现硬伤，不是不可修的「环境决定」）。**修法**：按既有协议，父进程取 `torch.get_num_threads()`，子进程脚本首行 `torch.set_num_threads(该值)`。断言、期望值、容差一律不改。

### 2.2 R5 runner `test_train_run_completes_restores_and_keeps_frozen_groups`：真错（出厂 runner 缺陷）

**证据**：`scripts/training/run_native_content_core_r5.py` 的 `_fresh_probe()`（L155-170）用 `[sys.executable, __file__, "--restore-probe", path]` 起子进程，**没有把父进程线程预算传下去**。父进程在 pytest 会话里被早先用例钉到 1 线程，子进程从机器默认（本机 12）起步 → `zero_restore.fresh_process_equal=False` → `train_run` 在 L244 提前 return，`steps_executed` 保持初始值 0，于是 `assert 0 == 1`。

**与 2.1 的区别**：这不是测试写法问题，而是**出厂训练 runner** 的恢复自检没有控制运行条件；任何宿主上线程预算会改变末位，runner 都会误判恢复失败并中止整轮。**判定：真错。**

**修法**：`_fresh_probe` 把父进程 `torch.get_num_threads()` 通过新参数 `--threads` 传给子进程；`main()` 在 `--restore-probe` 分支先 `torch.set_num_threads(args.threads)`。报告的 `environment.torch_threads` 已记录父进程线程数，新增参数不改变报告字段。

### 2.3 `test_r022_declared_credit_gate`：环境决定（失败字段待定位，先加可诊断性，不放宽）

**证据**：CI 失败在 L859 `gate["passed"] is False`；其前 5 条集合断言（L854-858：`unexpected`／`below_noise`／`fd_disagreements`／`directional_disagreements`／`second_disagreements` 皆空）**已通过**。因此失败成分只能是 `fp32_violations`／`probe_failures`／`control_failures`／`clamp_margin<=1`／`moved=False`／`inert_updates_negligible=False` 之一。

本机同一 `compute_all()` 读数（float64/float32 同源码、`torch.set_num_threads(1)`）：

```
passed True
unexpected=[] below_noise=[] fd_disagreements=[] directional_disagreements=[] second_disagreements=[]
fp32_violations=[] probe_failures=[] control_failures=[]
clamp_margin=14.775532408851708        (阈 >1.0)
moved=True  inert_updates_negligible=True
inert_update_max=1.2197187704288126e-12  (阈 <=1e-11)
inert_post_gradient_max=2.1955093992831465e-18  (阈 <=1e-14)
```

本机所有量离阈值都远（clamp 14.78 vs 1，inert 1.2e-12 vs 1e-11），说明不是「差一点点」。两腿 torch 同为 `2.13.0+cpu`（ci.yml L257/L780/L919），差异只能来自 OS/CPU/BLAS。

**判定**：环境决定。**但具体是哪个 gate 字段失败，现有日志没有携带**——这是证据缺口，不是可以猜的读数。

**处置（不猜测、不放宽）**：把 L859 改成带载荷的断言 `assert gate["passed"] is True, gate`，使下一次 CI 红直接打印失败字段与数值。**在此之前不 rebaseline、不改任何阈值。**

### 2.4 两条关系契约 float 翻转（`test_native_local_relation_contract.py:72`、`test_native_relation_confirmation_contract.py:72`）：判据本身写错

**证据**：两条断言都是 `all(torch.equal(b, b[:1].expand_as(b)) for b in blocks)`，且测试内已 `torch.set_num_threads(1)`。同进程、同预算下仍要求**逐位相等**，跨 BLAS 分块实现不成立（同一输入的行在不同分块核里可能差 1-2 ulp）。CI 原文：`assert all(torch.equal(...))` → `assert False`。

**判定**：判据本身写错（把「行间功能等价」写成了「跨 BLAS 逐位相等」）。

**修法**：改为带自缩放容差的等价判断，并在失败信息里打印实测行间最大偏差：

```
tol = 8.0 * torch.finfo(block.dtype).eps * max(1.0, float(block.abs().max()))
spread = float((block - block[:1]).abs().max())
assert spread <= tol, (spread, tol)
```

`8 ulp × 量级` 是 float 累加误差的保守界，不是随手放宽；若实测 spread 超过该界，则说明确有实现问题，需升级为真错处理（此时断言会打印具体数值供定位）。

### 2.5 文档链红：已在 HEAD 修复，本轮不动

`reports/taiji_scale_20261108_outputs.jsonl` 被 `.gitignore:157:*.jsonl` 忽略，CI 干净检出不存在；当前 HEAD 的 `plans/active/**` 已无任何指向它的 markdown 链接（全树 grep 0 命中），本机 `pytest tests/seed/test_project_identity.py` = `4 passed`。CI 那行红来自修复前的提交，本轮不复改。

---

## 3. 精确改动清单

| # | 文件 | 改动 |
| --- | --- | --- |
| C1 | `tests/taiji_native/test_conditional_language_candidate_v3.py` | `test_v3_step_update_and_fresh_process_checkpoint_restore`：算摘要前取 `threads = torch.get_num_threads()`；子进程 `code` 前缀加 `f"import torch; torch.set_num_threads({threads}); "` |
| C2 | `tests/taiji_native/test_native_content_core.py` | `test_zero_step_bundle_restores_in_process_and_fresh_process`：同上，子进程 `code` 前缀加线程钉定 |
| C3 | `tests/taiji_native/test_ordinary_baseline.py` | `test_zero_step_bundle_restores_in_process_and_fresh_process`：同上 |
| C4 | `scripts/training/run_native_content_core_r5.py` | `_fresh_probe` 传 `--threads <父进程线程数>`；`main()` 增 `--threads`，`--restore-probe` 分支先 `torch.set_num_threads` |
| C5 | `tests/taiji_native/test_living_core_credit_r022.py` | L859 改为 `assert gate["passed"] is True, gate`（只加诊断，不改判据） |
| C6 | `tests/taiji_native/test_native_local_relation_contract.py` | L72 改为 `8 ulp × 量级` 自缩放容差 + 打印 spread |
| C7 | `tests/taiji_native/test_native_relation_confirmation_contract.py` | L72 同上（用 `transfer.torch`） |

不改任何 `taiji/` 主体实现，不改任何既有报告/阈值/期望摘要。

## 4. 验证

1. 单测：以上 7 个文件逐一 `pytest -p no:cacheprovider -q`，全绿；再按「r022 先跑」（钉 1 线程）的敌意顺序合跑一次，确认 C1-C4 稳定。
2. 一致性：`pytest tests/seed/test_project_identity.py` 仍 `4 passed`。
3. 台账/计划：把本轮判定写入 `plans/reference/DECISION-LEDGER_20261109.md` 新增一条（编号续 `D-20261111-xx`），并在 `plans/reference/FINDING-CROSS_PROCESS_EVIDENCE_20261110.md` 补上这三条新实例；更新 `plans/active/roadmap/03_CURRENT_EXECUTION.md` 的当前动作。
4. 提交并推送。

## 5. 后续块（本 spec 不实施，仅登记）

- 158 次 deselect 清单：按「已知环境不可跑 / 被顺手排除」二分，后者恢复。
- r022 gate 字段定位：等下一次 CI 红的 `gate` 载荷。
- 620 份豁免、278 条 advisory：不作硬目标。

## 6. 需要批准的决策

1. C4 修改出厂 runner（非测试）是否接受？理由：这是唯一能让 R5 恢复自检在任意宿主成立的方式，且不改变报告字段。
2. C6/C7 使用 `8 ulp × 量级` 容差是否接受？备选：先只加 spread 打印、保持红，等下一轮拿到实测偏差再定容差。
3. r022 本轮只加诊断、暂不 rebaseline，是否接受？
