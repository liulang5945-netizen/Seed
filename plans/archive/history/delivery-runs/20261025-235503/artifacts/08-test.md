# Phase 8 前置取证 · 根因同构复现与定因（api-tester 查必达）

- 运行目录: `E:/Seed/delivery-runs/20261025-235503/`
- 取证时间: 2026-10-25 23:58 – 2026-10-26 00:35 (GMT+8)
- 取证性质: **只读取证**，未修改任何被追踪文件
- 本文件正文＝取证章节（复现方法 / 复现结果 / 三子条件具体值 / 归族与重启条件 / deselect 核账）

---

## 1 结论（先给答案）

1. **R1 已解除——本机同构复现方法已建立，两条失败路径都已在本机原样复现成功**，
   不再需要"推上去看 CI 变不变绿"。
2. **CI 失败 run `40203419431` 的 head sha 是 `29980ee3b`，不是 `fbc212397`**（01-intake §2.2 记载有误，
   已用 `gh run view --json headSha` 复核）。这处更正 **改变了整个难度**：
   `fbc212397` 已经是修复之后的提交，在它上面怎么试都复现不出来；
   必须回到 `29980ee3b` 才站在真正的红点上。本机实测也印证了这点
   （见 §3.3 复现矩阵的 `cimirror-fbc-lf` 行：干净检出下 import 通过）。
3. **红点 B（19 模块 collection error）在干净检出里 100% 复现**：
   `Interrupted: 19 errors during collection`，失败模块清单与 CI 一致，抛出点
   `scripts/training/taiji_r2_dialogue_continuation_extension.py`（在 `29980ee3b` 上为第 755 行）。
4. **失败子条件是 2 个，不是 1 个**，且已定因到具体值：
   - `boot_audit_path` 判据 **False**（维护机绝对路径轴）；
   - `boot_accounting_sources` 字典全等判据 **False**（绝对路径轴 **且** 工作树字节渲染轴同时打偏）；
   - `boot_audit_sha256` 判据 **True**（该文件被 `.gitattributes` 钉成 `eol=lf`，跨机字节一致，侥幸躲过）。
   详见 §4。
5. **归族**：落入"维护机绝对路径"族（次轴＝CRLF 渲染未归一），**不是**"缺未入库产物"族
   （全部相关文件经 `git ls-files --error-unmatch` 验证均已 Tracked，见 §4.4）。
6. **红点 A 是 B 的下游症状——本机实测闭环**：同一棵 `29980ee3b` 干净检出里，
   deselect 清单落空 **12 条 / 93 条，覆盖 5 个 r2_\* 文件**，与 CI 日志打印的 24 行 error
   （12 条 × 2 腿）**逐条一致**。
7. **当前 HEAD（`979217e93`）在 CI 同构检出（LF）下已经全绿**：extension import 通过、
   collection 无 error、deselect 86 条全命中。**修复早已由 `565604a9e` 落地**，
   但迄今**没有任何一次 CI run 给它出过完整结论**（近期 run 全被 cancel-in-progress 干掉）。
8. **但 S1 没有死干净，换了个灶台还在烧**：当前 HEAD 在 **CRLF 渲染**的干净检出里，
   仍然 `Interrupted: 19 errors during collection`，同样是 19 个模块，
   但这回炸在更前一级的 **v1.1 review report 严格字节比较**（extension.py:678）。
   → 今天 CI 之所以躲过去，**唯一原因**是 Windows 腿把 `git config --global core.autocrlf false`
   排在 `actions/checkout` **之前**跑。这是活雷，详见缺陷 D-01。

---

## 2 产物路径

| 类别 | 路径 |
|---|---|
| 本阶段产物（取证章节） | `E:/Seed/delivery-runs/20261025-235503/artifacts/08-test.md`（本文件） |
| 复现探针 · HEAD 版判据全链 | `E:/Seed/.workbuddy/tmp/probe_chain.py` |
| 复现探针 · `29980ee3b` 版旧严格判据 | `E:/Seed/.workbuddy/tmp/probe_v13_old.py` |
| 复现探针 · v1.3 三子条件细粒度（首轮） | `E:/Seed/.workbuddy/tmp/probe_v13.py` |
| deselect 核账脚本（两腿比对＋逐条命中） | `E:/Seed/.workbuddy/tmp/audit_deselect.py` |
| 干净检出树 @CI 失败点 | `E:/Seed/.workbuddy/tmp/cimirror-299-lf`（钉在 `29980ee3b`，LF） |
| 干净检出树 @HEAD LF | `E:/Seed/.workbuddy/tmp/cimirror-head-lf`（钉在 `979217e93`，LF） |
| 干净检出树 @HEAD CRLF | `E:/Seed/.workbuddy/tmp/cimirror-head-crlf`（钉在 `979217e93`，CRLF） |
| 干净检出树 @`fbc212397`（对照，已含修复） | `E:/Seed/.workbuddy/tmp/cimirror-fbc-lf` |
| 干净检出树 @`60f32a95f` CRLF（首轮，已过时） | `E:/Seed/.workbuddy/tmp/cimirror-head` |

> 上述 worktree 均为临时，共 5 棵：**不删即是复现环境**（保留便于后续阶段复跑）。
> 清理方式：`git worktree remove --force <dir>`（本阶段未执行，避免复用时重建 2 万个文件）。

---

## 3 复现方法与复现结果矩阵

### 3.1 环境（写死，避免后续误判）

| 项 | 值 |
|---|---|
| 解释器 | `C:/Users/23747/AppData/Local/Programs/Python/Python312/python.exe`（3.12.10；pytest 9.1.1、torch 2.13.0+cpu、psutil **7.2.2**、numpy 2.5.1、black 26.5.1） |
| 必需环境变量 | `CODEBUDDY_SAFE_DELETE_ENABLED=0`（批删守卫会劫持 `Path.unlink`/`os.remove`，不设会误诊为磁盘卡死）＋ `PYTHONUTF8=1` |
| git | PortableGit（随 WorkBuddy 携带），**系统 gitconfig 里 `core.autocrlf=true`**（`D:/WorkBuddy/resources/vendor/PortableGit/etc/gitconfig`） |
| 本机 psutil 版本 | 7.2.2 —— 与 CI 约束 `psutil>=5.9.0,<7.0.0` 不一致，登记为一节遗留风险（见 §8 L3） |

### 3.2 复现配方（三步，可重复）

```bash
# ① 干净 worktree 模拟 CI 检出。
#    关键点是 -c core.autocrlf=false：CI 两腿都是 LF（Windows 腿先 git config --global core.autocrlf false
#    再 checkout；Ubuntu 腿默认即为 LF），而本机散发的 PortableGit 把它设成了 true。
git -c core.autocrlf=false worktree add --detach E:/Seed/.workbuddy/tmp/cimirror-299-lf 29980ee3b

# ② 单进程定因：不必跑 pytest，换哪棵树只用 SEED_ROOT 指定
cd E:/Seed/.workbuddy/tmp/cimirror-299-lf
export CODEBUDDY_SAFE_DELETE_ENABLED=0 PYTHONUTF8=1 SEED_ROOT=$PWD
C:/Users/23747/AppData/Local/Programs/Python/Python312/python.exe E:/Seed/.workbuddy/tmp/probe_v13_old.py

# ③ collection 层面的复现（与 CI 同构）
C:/Users/23747/AppData/Local/Programs/Python/Python312/python.exe -m pytest tests/ --co -q \
    -m "not slow" -p no:cacheprovider
```

> **为什么用 `SEED_ROOT`**：探针不硬编码根目录，由环境变量指定"要看哪棵树"。
> 同一份脚本可在主树 / 任一 worktree 上对同一批桩点与判据逐一比对（`probe_chain.py` 同理）。

### 3.3 复现结果矩阵（全部实测）

| 被测树 | commit | 检出渲染 | extension import | pytest collection（全 `tests/`） |
|---|---|---|---|---|
| `E:/Seed` 主树（含未跟踪产物） | `979217e93`（起手时 `60f32a95f`） | 混合（v1-1 md 恰为 LF） | **OK** `(1500000.0, 4300000.0)` | 未跑（本阶段禁全量） |
| `cimirror-head` | `60f32a95f` | CRLF | **FAIL** `v1.1 source revision review report binding differs` | 未跑（已由 `-crlf` 树代庖） |
| `cimirror-head-lf` | `979217e93` | **LF** | **OK** | 无 error（单模块抽样 11 collected） |
| `cimirror-head-crlf` | `979217e93` | **CRLF** | **FAIL** `v1.1 ... review report binding differs` | **`Interrupted: 19 errors during collection`**，4482/4485 collected，28.29s |
| `cimirror-fbc-lf` | `fbc212397`（intake 误记的 commit） | LF | **OK** ← 该 commit 已含修复，复现不出来 | — |
| **`cimirror-299-lf`** | **`29980ee3b`（= CI 失败 run 真身）** | **LF** | **FAIL `v1.3 boot audit or source binding differs`** | **`Interrupted: 19 errors during collection`**，4474/4477 collected，28.38s |

失败模块清单（`cimirror-299-lf`，与 CI Windows 腿等量，共 19 个，均在 `tests/taiji_native/` 下）：

```
test_r2_continuation_budget_amendment_v1_4_contract     test_r2_dialogue_continuation_extension_contract
test_r2_continuation_budget_amendment_v1_5_contract     test_r2_dialogue_continuation_extension_runner_contract
test_r2_continuation_endpoint_preflight_contract        test_r2_dialogue_continuation_source_revision_contract
test_r2_continuation_evaluation_adapter_contract        test_r2_engineering_recovery_entry_contract
test_r2_continuation_launch_provenance_contract         test_r2_evaluation_recovery_v2_contract
test_r2_dialogue_checkpoint_resume_adapter_contract     test_r2_fast_ledger_resume_contract
test_r2_fast_ledger_resume_v2_contract                  test_r2_fast_ledger_resume_v3_contract
test_r2_fast_ledger_resume_v4_contract                  test_r2_resume_contract
test_r2_resume_ledger_anchor_contract                   test_r2_target_update_projection_contract
test_r2_worker_namespace_integration
```

出错调用链末帧（与 CI traceback 同形）：
`tests → adapter / prepare_endpoints → extension.py:1532 <module> → effective_budget_limits() → raise`。

---

## 4 定因：三个子条件的**具体值**

### 4.1 `29980ee3b`（CI 失败真身）上的判据——三字段全部机器局部严格比较

该 commit 的 `effective_budget_limits()` v1.3 分支（`git show 29980ee3b:…`，第 745–755 行）：

```python
if (
    revision.get("boot_audit_path") != str(audit.resolve())                  # ① 绝对路径严格相等
    or revision.get("boot_audit_sha256") != _sha256_file(audit)              # ② 字节严格相等
    or revision.get("boot_accounting_sources") != expected_boot_sources      # ③ dict 全等（键是绝对路径，值是本机工作树 sha）
):
    raise ExtensionLedgerRefuse("v1.3 boot audit or source binding differs")
```

| # | 判据 | recorded（`…_source_revision_v1_3.json` 里的记录值） | actual（干净检出实测） | 结果 |
|---|---|---|---|---|
| ① | `boot_audit_path` | `E:\Seed\reports\taiji_r2_daily_dialogue_recovery_v2_20261021_boot_accounting_audit_v1.json` | `E:\Seed\.workbuddy\tmp\cimirror-299-lf\reports\…json`（CI 上则是 `D:\a\Seed\Seed\reports\…json`） | **False** |
| ② | `boot_audit_sha256` | `897adda1e1882092f941470425ae737459dfb5318106ac3062c3c8ae941d67d7` | `897adda1e1882092f941470425ae737459dfb5318106ac3062c3c8ae941d67d7`（LF，2612 B） | **True** |
| ③ | `boot_accounting_sources` | 键 `E:\Seed\scripts\training\taiji_r2_dialogue_boot_accounting.py` → `73facfe1cd13f3f4eb0f6a752af6d43e2ba241fdb5845c47563509ba084ba25b`；<br>键 `E:\Seed\tests\taiji_native\test_r2_dialogue_boot_accounting_contract.py` → `fdd19dd18cf569834076e4964b2b69d824afe58dd2d2e23a17b3a9bc04cc9020` | 键 `…\cimirror-299-lf\scripts\training\taiji_r2_dialogue_boot_accounting.py` → `d540c6949aacda8eb8f1e39b2ff8dae4e2bc5746ce71c57f173166646c609c43`；<br>键 `…\cimirror-299-lf\tests\…test_r2_dialogue_boot_accounting_contract.py` → `6b21f81ce1f7e2444a10ab0b93b30d93d11bf43602e644b996e392db9c757db7` | **False**（键轴 False，值轴也 False） |

**值轴旁证（证明 recorded 值是 CRLF 渲染）**：同一份 `scripts/training/taiji_r2_dialogue_boot_accounting.py`
在 **CRLF 渲染**的干净检出（`cimirror-head`）里实测 sha256 = `73facfe1cd13f3f4…` ＝ recorded 值；
在 LF 检出里 = `d540c6949aacda8e…`。⇒ recorded 取自 CRLF 时代的维护机工作树，而不是 git blob。

⇒ **① 与 ③ 同源**：v1.3 修订单把"某台机器的绝对路径 + 那一刻的工作树字节"写成了冻结证据，
并在**导入期**用 `!=` 搬回来比。`raise` 的触发条件是 `①False or ②False or ③False` ⇒ 成立 ⇒ 19 个模块连带。

### 4.2 HEAD（`979217e93`）上的三子条件（修复后）——全部 True，附值

`565604a9e` 把 ①③ 换成 `seal_ast` 身份映射、② 换成渲染容忍，并让 ③ 逐件走 `verify_path_pin`：

| # | 判据 | 具体值 | 结果 |
|---|---|---|---|
| C1 | `boot_audit_identified` | `repo_relative(recorded='E:\Seed\reports\…json')` → `reports/taiji_r2_daily_dialogue_recovery_v2_20261021_boot_accounting_audit_v1.json`；<br>`repo_relative(actual='…\cimirror-head-lf\reports\…json')` → 同上 | **True** |
| C2 | `rendering_tolerant_match` | recorded `897adda1…67d7` ＝ raw sha(LF) `897adda1…67d7`；LF→CRLF 变体 `9aec4b44…300d` | **True**（还是严等命中） |
| C3 | `boot_sources_match` | 两侧 key 均映射成 `scripts/training/taiji_r2_dialogue_boot_accounting.py` 与 `tests/taiji_native/test_r2_dialogue_boot_accounting_contract.py`；<br>在位字节比：**False**（recorded `73facfe1…` / `fdd19dd1…` vs LF 实测 `d540c694…` / `6b21f81c…`）；<br>回落 `git_history_attests`：**True**（两个文件各自只被 1 个 commit 改动过：`622bde9ef`） ⇒ `verify_path_pin` **True** | **True** |

> C3 的绿 **不在位字节通道上**，是靠 git 历史取证（渲染变体）兜住的。这是一条隐性依赖：
> 一旦走浅克隆到连 HEAD blob 都取不到，或相关文件的历史被改写，这条会转红。登记进 §8 L4。

### 4.3 仍然活着的那一个：v1.1 review report 严格通道（D-01）

在 **CRLF 渲染**的干净检出（`cimirror-head-crlf` @ `979217e93`）里，第一个炸的不是 v1.3，
而是更早的 `_validate_prior_v1_1()`（extension.py:672–680）。其中第 678 行仍用
`_sha256_file(...) != review_sha` **严格字节比较**，而它的同族 v1.2 / v1.3 位点
已经被 `29980ee3b` / `fbc212397` / `565604a9e` 相继改成了 `rendering_tolerant_match`：

```
[Stage A] v1.1 review report（严格通道）
  file     = plans/reference/delivery-runs/20261021-n-mainline/artifacts/04-continuation-source-revision-v1-1.md
  recorded = 5440b14ff76de69de1b7de12b56420b544d675f029e45cc79da4cb70ef222604
  LF 检出   raw = 5440b14f...  (3110 B)   == recorded      -> strict True   (主树恰为此形态 ⇒ 本机一直绿)
  CRLF 检出 raw = d5b807bf4f1b02ded189e04d2d4f758c237319a35580e6e095e0a2c8bb5f4a5d (3127 B)
                                                           -> strict False
  容忍比较  recorded ∈ {raw, LF→CRLF, CRLF→LF} 三变体       -> True   ← 语义没变，纯渲染差异
  => ExtensionLedgerRefuse: v1.1 source revision review report binding differs
```

旁证：`git show HEAD:<file> | sha256sum` = `5440b14f…` ＝ recorded ⇒ 记录值取自标准 LF blob，
CRLF 检出只是把它换了相貌。

### 4.4 为什么不是"缺未入库产物"族

```
reports/taiji_r2_daily_dialogue_recovery_v2_20261021_boot_accounting_audit_v1.json   -> TRACKED
reports/…_extension_v1_source_revision_v1_3.json / v1_2 / v1_1 / _contract.json      -> TRACKED
scripts/training/taiji_r2_dialogue_boot_accounting.py                                -> TRACKED
tests/taiji_native/test_r2_dialogue_boot_accounting_contract.py                      -> TRACKED
plans/reference/delivery-runs/20261021-n-mainline/artifacts/04-continuation-*.md     -> TRACKED
```

（`git ls-files --error-unmatch` 逐条验证。）文件都在，缺的是**跨机外观归一**，不是缺文件。

### 4.5 归族与重启条件

| 编号 | 现象 | 归族（intake §2.5 口径） | 重启条件 |
|---|---|---|---|
| CI 红点 B @`29980ee3b` | `v1.3 boot audit or source binding differs` | **维护机绝对路径**（次轴：平台编码 / CRLF 渲染） | ① 环境与治理面：禁止把机器绝对路径作为判据键写进修订单（已由 `565604a9e` 完成，本机证据＝LF 干净检出 import OK、落空 0/86）；<br>② 结构面（Phase 4 裁决，未做）：**校验不得放在导入期决定模块可导入性**——治理校验必须可延迟 / 可跳过 / 有明确失败语义 |
| D-01（HEAD 仍活着） | CRLF 检出下 `v1.1 source revision review report binding differs` | **平台编码契约 / 渲染未归一** | ① 把 extension.py:678 的同族位点也改成 `rendering_tolerant_match`（与前几个 commit 改 v1.2 / v1.3 位点同法，不动判据语义）；<br>② 或给 `plans/reference/**/*.md` 加 `.gitattributes` eol 规则钉死渲染；<br>③ 两条都不做，则必须**永久保留** Windows 腿"checkout 之前先 `core.autocrlf false`"这一步，并把理由写进注释判据 |
| CI 红点 A | deselect 12 条落空 | B 的下游症状（本机已闭环复现） | 随 B 消除；另外必须拆掉"守卫一红即跳过全部 pytest"的自伤结构（S2） |

---

## 5 deselect 清单核账（`ci.yml` 两处 `DSH_CI_DESELECT`）

### 5.1 两腿一致性（`ci.yml` @HEAD `979217e93`，814 行）

- Ubuntu 腿（`test` job）env：第 107 行起，**86 条**；
  Windows 腿（`test-windows` job）env：第 667 行起，**86 条**；
- **顺序完全一致 = True**；两腿互差集合均为空；腿内无重复条目；两条都不缺 `::方法名`。
- ⇒ **当前 HEAD 两腿清单逐条一致**（纪律满足），不存在 ㊵-466 那种"只改一条腿"的遗漏。

### 5.2 在干净检出里的落空数

| 被测树 | 去重条目 | 命中 | **落空** | 落空文件数 |
|---|---|---|---|---|
| `cimirror-head-lf`（HEAD，LF＝CI 同构） | 86 | 86 | **0** | 0 |
| `cimirror-head-crlf`（HEAD，CRLF） | 86 | 80 | **6** | 4 |
| `cimirror-299-lf`（CI 失败点，LF） | 93 | 81 | **12** | 5 |

**HEAD / LF 落空为 0**：当前清单在当前 HEAD 上是干净的，A 已随 B 的修复消失（本机证据）。

**`29980ee3b` 的 12 条落空**（与 CI 日志 `gh api …/jobs/121199207675/logs` 第 2757–2780 行逐条一致，
CI 打的是 24 行 error ＝ 12 条 × 2 腿）：

```
test_r2_continuation_evaluation_adapter_contract.py::test_preflight_binds_frozen_and_extension_sources_by_default
test_r2_continuation_evaluation_adapter_contract.py::test_preflight_binds_launch_provenance_ledger_and_budget
test_r2_continuation_evaluation_adapter_contract.py::test_preflight_records_historical_sources_without_byte_binding
test_r2_dialogue_continuation_extension_contract.py::test_valid_v1_2_revision_applies_approved_limits_to_ledger_guard
test_r2_dialogue_continuation_extension_contract.py::test_v1_2_revision_rejects_lineage_budget_and_source_drift[contract]
test_r2_dialogue_continuation_extension_contract.py::test_v1_2_revision_rejects_lineage_budget_and_source_drift[previous]
test_r2_dialogue_continuation_extension_contract.py::test_v1_2_revision_rejects_lineage_budget_and_source_drift[budget]
test_r2_dialogue_continuation_extension_contract.py::test_v1_2_revision_rejects_lineage_budget_and_source_drift[tracked_source]
test_r2_dialogue_continuation_extension_contract.py::test_v1_2_revision_rejects_lineage_budget_and_source_drift[review_report]
test_r2_dialogue_continuation_extension_runner_contract.py::test_start_charge_uses_latest_extension_snapshot_monotonically
test_r2_fast_ledger_resume_contract.py::test_manifest_binds_fast_ledger_anchor_sources_and_resume_checkpoint
test_r2_fast_ledger_resume_v2_contract.py::test_v2_records_scoped_memory_floor_and_deferred_arms
```

**更正 01-intake §2.2**：原文写"去重 13 条"。实际 CI 日志是 **12 条**（24 行 ＝ 12 × 2）。以日志为准。

### 5.3 清单核对的两个口径瑕疵（不是红点，但会误导数字）

1. `scripts/training/verify_ci_deselect_list.py` 的 `listed_entries()` 扫的是**整份 ci.yml** 里所有
   行首为 `--deselect` 的行 ⇒ **两条腿被重复计数**。所以那次 CI 打印"清单 186 条"＝93 条 × 2 腿；
   在 HEAD 上会打印"清单 172 条"＝86 条 × 2 腿。**这不能读成"清单从 186 条净减到 XX 条"**，口径不同。
2. 命中判定用的是子串包含 `if method in collected.stdout` —— 若某条的用例名恰好是另一条 node id 的子串，
   落空会被静默判成命中（与它想防的"静默失效"反向同源）。本阶段改用**精确 node id 集合**匹配。

---

## 6 缺陷列表

| 编号 | 等级 | 标题 | 复现步骤 | 期望 | 实际 |
|---|---|---|---|---|---|
| **D-01** | P1（活雷，当前 HEAD 仍在） | v1.1 review report 用**严格字节**比较，**CRLF 渲染的检出必炸**，同样 19 模块 collection error | ① `git worktree add --detach <wt> 979217e93`（不覆盖 autocrlf，本机 PortableGit 系统配置为 true）② `cd <wt>` ③ `CODEBUDDY_SAFE_DELETE_ENABLED=0 python -m pytest tests/ --co -q -m "not slow"` | 导入期治理校验对"语义未变、仅换行渲染不同"应当放行（同源的 v1.2 / v1.3 位点已如此） | `ExtensionLedgerRefuse: v1.1 source revision review report binding differs`；`Interrupted: 19 errors during collection`（28.29s） |
| **D-02** | P1（已修、缺 CI 侧取证） | CI 那次 19 模块 collection error 的本体（@`29980ee3b`） | ① `git -c core.autocrlf=false worktree add --detach <wt> 29980ee3b` ② 同上运行 `pytest --co` | 干净检出 ＝ CI 检出，不应因"谁在哪台机器上签的字"而不可导入 | 修复已由 `565604a9e` 落地（本机 LF 干净检出 @HEAD：import OK、落空 6/86 → 0/86）；**但至今没有任何一次 CI run 出过完整结论**（近期 run 全被 cancel-in-progress 干掉） |
| **D-03** | P3（数据卫生） | 守卫脚本把两腿清单重复计数，且用子串判定命中 | `python scripts/training/verify_ci_deselect_list.py` | 报"去重后的条目数"，且用精确 node id 匹配 | 报 186 / 172（＝93 / 86 × 2）；子串命中可能漏判落空 |
| **D-04** | P3（文稿更正） | 01-intake §2.2 记 run `40203419431` 的 commit 为 `fbc212397` | `gh run view 40203419431 --json headSha` | 与实际一致 | 实际 `29980ee3b8d65890edba89e8d44be32538fac19a`；另"去重 13 条"应为 12 条 |

---

## 7 性能 / 覆盖数据（本阶段口径）

本阶段是**取证**，不是全量测试；按指令未跑全量 `tests/`（本机约 15 分钟且可能 SIGTERM）。
所有"套件健康度"数据均为 **collection-only**：

| 指标 | HEAD / LF | HEAD / CRLF | CI 失败点 / LF | CI 实测（Windows 腿） |
|---|---|---|---|---|
| collection 结果 | 无 error | **19 errors** | **19 errors** | 19 errors |
| 收集用例数 | —（抽样单模块 11 collected） | 4482 / 4485（3 deselected） | 4474 / 4477（3 deselected） | 84 deselected, 2 skipped |
| collection 墙钟 | 6.22s（单模块） | **28.29s** | **28.38s** | 48.92s（runner 慢于本机约 1.7×） |
| deselect 落空 | 0 / 86 | 6 / 86 | 12 / 93 | 12 条（24 行） |

（分位数不适用：本阶段没有跑请求负载 / 吞吐，只有 collection 墙钟；刻意不给单一平均值糊弄。）

---

## 8 遗留风险

- **[R1 已关闭]** 本机同构复现方法建立，CI 失败路径 `29980ee3b` 与 HEAD 的 CRLF 路径均已复现成功。
- **[L1] D-01 未修**：HEAD 在 CRLF 渲染的检出下仍是 19 模块 collection error。
  今日 CI 唯一的避风港是"Windows 腿把 `core.autocrlf false` 排在 checkout 之前"这个**步骤顺序**。
  任何重排 checkout 步骤、换自托管 Windows runner、或某台机器带 `autocrlf=true` 的场景都会立即翻红。
- **[L2] Linux 侧未实测**：Ubuntu 腿同为 LF 检出，判据形式与本机 LF worktree 同构，
  但根目录是 `/home/runner/work/Seed/Seed`，本机（Windows）**无法证伪**。
  `seal_ast` 的映射是"按已知仓库路径结尾"的归一，理论上能吃下任意前缀，但这是推断不是读数。
- **[L3] 本机 psutil 7.2.2 ≠ CI 约束 `<7.0.0`**：本机 collection 结论可能与 CI 有细微差异。
- **[L4] C3 的绿依赖 git 历史取证**：HEAD 的 `boot_sources_match` 在位字节不命中，
  靠 `file_commits` 找到那 1 个 commit 的 blob 渲染变体才通过。这条路依赖"git 可用且能读到 blob"；
  CI 当前 `fetch-depth: 0` 满足，改成浅克隆之前必须重跑本阶段配方。
- **[L5] HEAD 在动**：取证期间 owner 并行会话把 main 从 `60f32a95f` 推到 `979217e93`
  （我的第一棵 HEAD worktree 因此钉在了旧 commit）。**所有结论绑定到具体 sha**，不要跨 sha 复用。
- **[L6] 未做到行为级验证**：本阶段只走到 collection 层，
  "那 19 个模块里的用例本身是红是绿"尚未测（留给后续阶段的本机通跑）。

---

## 9 给下游的注意事项

1. **Phase 4（架构裁决）/ Phase 6（实现）**：S1 的根不是"某个哈希算错了"，而是
   **"修订单把机器绝对路径 + 当时工作树字节写成冻结证据，然后在导入期用 `!=` 搬回来比"**。
   环境治理面（§4.5 重启条件①）已由 `565604a9e` 补上，**结构面没做**：
   治理校验仍在导入期决定模块可导入性。这是本次唯一必须裁决的结构性问题。
2. **任何碰治理面的改动，先在这个环境里自证**：`git -c core.autocrlf=false worktree add` + 三个探针 +
   `pytest --co`。**两套渲染（LF / CRLF）都要跑**——只跑 LF 会漏掉 D-01。
3. **不要动 Windows 腿那个"checkout 之前先 `core.autocrlf false`"的顺序**，除非先把 D-01 按 §4.5 修掉；
   真要动，先在注释里留下判据。
4. **deselect 守卫结构（S2）比 D-01 更危险**：它排在三个 pytest step 之前，一红就把后面全跳过
   ⇒ CI 事实上不测。收紧清单之前，先把守卫从"前置阻断"改成"后置报告"或非阻断。
5. **不要相信 push 后的下一次 CI**（S4：cancel-in-progress + 单腿 1.5h ⇒ 长期无完整结论）。
   本机同构证据优先。
6. **01-intake 的两处数字以本文件为准**（D-04）：run `40203419431` 的 head sha 是 `29980ee3b`；
   落空是 12 条不是 13 条。

---

## 10 附：S3 依赖面实测（应 implementation-planner 请求，pytest 9.1.1）

夹具在 `E:/Seed/.workbuddy/tmp/junit_shape/`（pass/fail/skip 三态）与
`E:/Seed/.workbuddy/tmp/junit_shape_err/`（模拟导入期抛异常，同 `ExtensionLedgerRefuse` 形态）。
**只读 conclusion，结论如下。**

**结 1 · 假设 A1（`<testcase>` 带 `file` 属性）不成立。**

```xml
<testcase classname="tests.taiji_native.test_r2_fast_ledger_resume_v4_contract"
          name="test_move_retries_a_transient_permission_error" time="0.037" />
```

pass / fail / skip **三态都没有 `file` 属性**——只有 `classname` + `name` + `time`。
⇒ S3 必须走 `classname` 派生面；前缀**必须带尾点**（`tests.taiji_native.`），
否则 `tests.seed` 会误吞 `tests.seed_platform.*`（team-lead 的 W-V1 已同旨）。

**结 2 · `classname` 是「相对 rootdir 的点分路径」——rootdir 变了胆汁也变。**

上面那条是在 worktree 里跑真实用例得到的；而把夹具放在 `E:/Seed/.workbuddy/tmp/junit_shape/` 下跑时，
rootdir 仍是 `E:\Seed`（pyproject 在那里），于是得到
`classname=".workbuddy.tmp.junit_shape.test_shape"`。
⇒ CI 上 rootdir 是 checkout 根，前缀稳定为 `tests.taiji_native.` / `tests.seed.` / `tests.`；
但**任何改变了 rootdir 的改动会连带改掉 classname**，ADR-S3 必须把 rootdir 写死。

**结 3 · collection error 会被写进 junit，但 `classname` 是空串（对 S3 是双刃剑）。**

```xml
<testcase classname="" name=".workbuddy.tmp.junit_shape_err.test_broken" time="0.000">
  <error message="collection failure">RuntimeError: simulated import-time failure ...</error>
</testcase>
```

- 它算进 `testsuite` 的 `errors="1"` 与 `tests="1"`；
- 而且 **`classname` 为空** ⇒ 按前缀归桶的三张面**一条也抓不到它**。
- 推论：**「三张面用例数之和 == 全量面」这条自校验，在发生 collection error 时会自然不等**
  ——这不是 bug，是它想要的性质，但必须显式表达，
  否则 `ADR-S3` 会把它当成"去重丢面"来排查（那是 R5，是另一件事）。
- 给 S3 的具体建议：自校验做成**两级**——
  ①**前置门禁**：全量面 junit 里出现任何 `classname=""` 的 `<testcase>` ⇒ 直接判红并打印模块名，
  （等价于把"collection error ⇒ 默默少跑"改成显式红，正好也是 S1/S6 的一层安全网）；
  ②通过 ① 之后，再断言三张面 node id 集合之和 == 全量面 node id 集合。

---

## 11 附二：交付给 W0.4 / M3a / W0.5 的三件成品（应 implementation-planner 请求）

### 11.1 W0.4 夹具（一条命令自动判红/判绿）

脚本 `E:/Seed/.workbuddy/tmp/w04_collection_jig.py`（**已双测通过**）：内部固定跑
`pytest tests/ --co -q -m "not slow" -p no:cacheprovider`，按 pytest 自己的汇总行判 collection error 数，
并额外打印 **HEAD sha / core.autocrlf / 哨兵文件换行渲染 / pytest 版本**，防止在错误的树上得出正确的结论。

```bash
# 期望绿（LF 干净检出 = CI 同构）
SEED_ROOT=E:/Seed/.workbuddy/tmp/cimirror-head-lf \
CODEBUDDY_SAFE_DELETE_ENABLED=0 PYTHONUTF8=1 \
C:/Users/23747/AppData/Local/Programs/Python/Python312/python.exe \
    E:/Seed/.workbuddy/tmp/w04_collection_jig.py --expect green

# 期望红 19（CRLF 干净检出 = 今日 HEAD 的活雷形态）
SEED_ROOT=E:/Seed/.workbuddy/tmp/cimirror-head-crlf \
CODEBUDDY_SAFE_DELETE_ENABLED=0 PYTHONUTF8=1 \
C:/Users/23747/AppData/Local/Programs/Python/Python312/python.exe \
    E:/Seed/.workbuddy/tmp/w04_collection_jig.py --expect red --expect-errors 19
```

退出码：`0` 实际＝期望（MATCH） / `2` MISMATCH（停工看日志） / `3` 环境不满足（含"忘了设
`CODEBUDDY_SAFE_DELETE_ENABLED=0`"）。实测读数：

| 树 | 结果 | collection error | 墙钟 | 首条治理异常 |
|---|---|---|---|---|
| `cimirror-head-lf`（LF） | `GREEN` → MATCH | 0 | 18.35s，4734/4737 collected | — |
| `cimirror-head-crlf`（CRLF） | `RED(19 errors)` → MATCH | 19 | 28.29s | `v1.1 source revision review report binding differs` |

### 11.2 L6 的答案：19 个模块的行为级通跑很便宜（1m25s），建议并入 M3a

在 `cimirror-head-lf`（HEAD，LF = CI 同构）上**不带任何 deselect** 通跑那 19 个模块：

```
3 failed, 249 passed in 85.20s (0:01:25)
FAILED test_r2_continuation_evaluation_adapter_contract.py::test_preflight_binds_frozen_and_extension_sources_by_default
FAILED test_r2_continuation_evaluation_adapter_contract.py::test_preflight_binds_launch_provenance_ledger_and_budget
FAILED test_r2_continuation_evaluation_adapter_contract.py::test_preflight_records_historical_sources_without_byte_binding
```

失败原因不是治理漂移，而是**未入库的语料缺失**：

```
AdapterRefuse: extension evaluation source is missing:
  <checkout-root>/data/conversation/OASST2/taiji_r2_daily_dialogue_episodes.jsonl
```

⇒ 这 **3 条正好等于清单里指向同一文件的那 3 条 deselect 条目**——清单在这里是"对的"，
重启条件就是 ci.yml 注释里写的①（`data/…` 语料入库）。

**给 M3a 的判据建议（避免把已知排除项读成新红）**：跑这 19 个模块时**不带** deselect，
断言"失败集合 == 清单里指向这 19 个文件的条目集合（当前为本上述 3 条）"；
多一条就是新红，少一条就是某条清单该收紧了。这条其实等于 **D2 的反向自校验**：
用"摘掉清单后是否如期红"来证明每条 deselect 确实还在被需要。

### 11.3 L2（Linux 侧）不需要为此新加 step —— 已被现有 run 证伪

`run 40207103896`（head `a8024efc7`）Ubuntu 腿日志第 2758 行实测：

```
deselect 清单 172 条，命中 172 条        # 0 条落空
6 failed, 3821 passed, 115 skipped, 85 deselected, 1 xfailed, 2 warnings in 3058.10s (0:50:58)
```

含义：守卫 step 在 **Ubuntu runner（根 `/home/runner/work/Seed/Seed`）** 上对每个条目各跑一次
真实 collection，172 条全命中 ⇒ **先前被红点 B 打掉的那 5 个 r2_\* 文件在 Linux + LF 检出下收集成功**，
即它们的导入期治理链（含 extension / seal_ast 映射）在 Linux 前缀下是通过的。
⇒ **L2 对本条病灶已证伪**；剩下的只是"全树 collection 在 Linux 上未被系统测过"，
这会被 S6 落地后 Ubuntu 腿自己的 collection 自动覆盖，不必为它单独加 step。

对 Q7(iii) 的意见：**作为 Windows 侧的 CRLF collection 探针，我支持**（便宜、正好挡 S1 这类导入期病，
且它在本机已被 CRLF 夹具证明能抓到 19 errors）；**但若把它装到 Ubuntu 来"顺带证伪 Linux"，
会把两件事混在一起**（`autocrlf=true` 的渲染轴 + Linux 平台轴），读数无法归属。
两件事分开：Linux 侧由上面的现有证据覆盖；CRLF 探针只留在 Windows 腿。

### 11.4 给 W0.5 的度量口径：别直接读 pytest 的 deselected 数

`E:/Seed/.workbuddy/tmp/count_deselected.py` 在 `cimirror-head-lf` 实测：

```
A: 只有 -m not slow     collected=4734/4737  deselected=3    ← 这 3 是 marker 挤出，不是清单
B: -m + 全清单           collected=4648/4737  deselected=89
=> 清单实际生效条数 = B - A = 86 / 86
```

⇒ **清单 86 条全部生效**（无静默失效），但 pytest 的 `deselected` 数是「清单生效数 + marker 挤出数」的叠加，
两股会互相掩护。**D2「条目数净减少」请度量去重后的清单条目数（或 B−A），不要直接读 deselected。**
（CI 那次打 "85 deselected" 是同一口径在 `a8024efc7` 上的读数，本机 979217e93 上是 89。）

### 11.5 顺带读到的事实：当前 HEAD 在 Ubuntu 上有 6 条结构性真红

同 run Ubuntu 腿全量回归 step 的 short summary（日志 2901–2906 行），6 条失败全部落在「结构性成长门族」：

```
test_continuous_structural_growth.py::test_continuous_structural_growth_gate
test_cross_domain_structural_gain.py::test_cross_domain_structural_gain_gate
test_interaction_group_multifamily.py::test_interaction_group_multifamily_leave_one_out_gate
test_online_interaction_structural_bridge.py::test_online_interaction_structural_bridge_gate
test_structural_workspace_net_gain.py::test_structural_workspace_net_gain_gate
test_terminal_three_domain_governance.py::test_terminal_three_domain_governance_gate
```

⇒ **DoD D1（同 commit 双绿）今天在这 6 条判据上不成立**，且这 6 条没有被 deselect 覆盖。
它们与本次 S1 无关（不是导入期治理，是浮点 / threshold 判据），我只是把读数落在这里给 Phase 4 与 WC 组用。

---

## 12 · WD.2 基线准备（基线 run 40171918472）

### 12.0 先修正我自己 §10 的规则（重要，直接影响 backend-architect 的实现）

§10 里我写的是"junit 里出现 `classname=""` ⇒ 判红"。**基线数据证明这条太粗，必须拆成两种**——
判据不是 classname，而是**子元素**：

| junit 行形态 | 含义 | 处置 |
|---|---|---|
| `<testcase classname="" name="<点分模块路径>"><error message="collection failure">…` | **真·collection error**（导入期抛异常） | 判红，打印模块名 |
| `<testcase classname="" name="<点分模块路径>"><skipped message="collection skipped">…` | **模块级 skip**（`pytest.skip(allow_module_level=True)`），合法 | 排除出面，但单独计数 |

我在 §10 的合成夹具只造了第一种，没造第二种，所以当时归纳不出这个区分。基线 run 里 4 条
`classname=""` 行**全部是第二种**（模块级 skip），0 条是第一种。若按 §10 原文实现"见空 classname 即判红"，
**Linux 腿会永远红**——因为 `tests/taiji_native/test_r2_fast_ledger_resume_v4_contract.py` 等
Windows NTFS 绑定模块在 Linux 上是合法的模块级 skip。backend-architect 的
`split_ci_junit_faces.py` 请按本表实现。

### 12.1 结论

1. **基线三面 node-id 集合已提取并双向互证**：
   - `report.xml`（全量）4580 行 → **4576 个唯一 node-id**（另有 4 条模块级 skip 行）；
   - 由全量按 classname 前缀派生：native **3901**、seed **185**、rest **490**；
     `3901 + 185 + 490 = 4576` **精确成立**，且 native∩seed = 0；
   - **派生面 vs 独立跑的 native-report.xml / seed-report.xml：双向 diff 均为 0**
     （only_in_a=0 且 only_in_b=0）——S3-A 的 classname 派生在真实数据上**逐 id 相等**，
     不是"数量相近"。这是 WD.2 对照实验的基线锚点。
2. **collection-error 清单＝空**（三张 junit 都是 0 条真 collection error）；
   模块级 skip 清单（4 条，都在全量面）：`tests.taiji_native.test_r2_fast_ledger_resume_v4_contract`、
   `tests.taiji_native.test_r2_worker_namespace_integration`（两条都是 "binds the Windows NTFS durability
   API at import time"，Linux 上合法跳过）、`tests.test_taiji_browser_environment`、
   `tests.test_taiji_browser_tool_loop`（"browser environment needs Playwright"）。
   这 4 条在任何"按前缀归桶"的面里都不出现，**必须**作为单独的第四类记账，否则三面之和会差 4。
3. **coverage 读数（B-4 门消费的数字）**：`core_reasoning_coverage_ci.json` →
   `core_line_pct = 65.25`（threshold 60.0，status=pass，checks 全 true；
   covered 2435 / statements 3732）；`coverage.xml` 全局 `line-rate = 0.6478`（64.78%），
   branch-rate 0.5104，lines 47643/73521（covered/valid）。趋势比对的基线锚点即 **65.25**。
4. 基线 run 本身：head `499c3892f9fd08e9ec335b7ad410d616ecdf3780`，conclusion `success`，
   2026-10-25T07:12:38Z 触发。

### 12.2 artifact 清单与提取读数

| artifact（`gh api …/runs/40171918472/artifacts` 实列，非猜测） | 内容 | 用途 |
|---|---|---|
| `verify-reports-3.12`（10,212 B） | 9 个 taiji_\*.json 验证报告 | M5/M6 门读数（本阶段未消费） |
| `test-results-3.12`（468,191 B） | `native-report.xml` / `seed-report.xml` / `report.xml` / `coverage.xml` | **本阶段主输入** |
| `test-results-windows`（300,828 B） | Windows 腿 junit（未下载，WD.2 若要跨腿对照再说） | — |
| `core-reasoning-coverage`（801 B） | `core_reasoning_coverage_ci.json` | **B-4 读数** |

三张 junit 的 pytest 自报口径 vs 本提取器口径：

| junit | pytest 自报（testsuite 属性） | 本提取器 | 差异解释 |
|---|---|---|---|
| native-report.xml | tests=3903, skipped=115, errors=0 | 3901 unique + 2 module-skip | pytest 把模块级 skip 行也算进 tests=；我的 unique 只收真用例 |
| seed-report.xml | tests=185 | 185 | 完全一致 |
| report.xml | tests=4580, skipped=131, errors=0 | 4576 unique + 4 module-skip | 同上 |

（注意：`tests=` 与 `unique` 的差 = 模块级 skip 行数。**backend-architect 的两级自校验若用 `tests=` 属性
当全量分母，会与 node-id 集合差出 4**——请统一用"唯一 node-id 数 + 模块级 skip 数"两个独立数字。）

### 12.3 产物路径

| 文件 | 说明 |
|---|---|
| `E:/Seed/.workbuddy/tmp/baseline-40171918472/test-results-3.12/` | native / seed / report 三张 junit + coverage.xml（原始产物，未改动） |
| `E:/Seed/.workbuddy/tmp/baseline-40171918472/verify-reports-3.12/` | 9 个验证报告 |
| `E:/Seed/.workbuddy/tmp/baseline-40171918472/core-reasoning-coverage/core_reasoning_coverage_ci.json` | B-4 门读数 |
| `E:/Seed/.workbuddy/tmp/baseline-40171918472/baseline_nodeid_sets.json` | **本阶段交付**：三面 node-id 排序列表 + 计数 + 模块级 skip / collection-error 清单 + 派生面对照 + coverage 读数 |
| `E:/Seed/.workbuddy/tmp/extract_baseline_faces.py` | 提取器（可重跑，SEED_TMP/SEED_ROOT 环境变量驱动） |

### 12.4 遗留风险

- **[B1] §10 原规则已被 §12.0 修正**：若 backend-architect 已经按旧文实现"空 classname 即判红"，
  需要按 §12.0 的两行表改（判据 = 子元素是 `<error message="collection failure">` 还是
  `<skipped message="collection skipped">`）。
- **[B2] 基线只有一个 run**：单样本无法区分"面集合稳定"与"面集合抖动"。
  建议再取 1–2 个成功 run 做同样的提取，把"跨 run 的 node-id 集合是否恒定"也钉下来
  （尤其 `-m "not slow"`、模块级 skip 与 Windows 腿的行为差异）。
- **[B3] `test-results-windows` 未下载**：本阶段只做了 Ubuntu 侧三面。若 WD.2 要跨腿对照
  （Windows 全量 vs Ubuntu 全量的 node-id 集合差），需要另跑一次提取——脚本可直接复用。
- **[B4] 模块级 skip 的 4 条是"面外幽灵"**：它们不在任何派生面里，但**在 pytest 的 `tests=` 里**。
  任何用 `testsuites` 属性做分母的实现都会与 node-id 集合差 4（见 §12.2 表）。

### 12.5 给下游

1. **backend-architect（split_ci_junit_faces.py）**：①判据表见 §12.0，别用 §10 原文；
   ②全量分母用"unique node-id + 模块级 skip 数"，别用 `tests=` 属性；③基线 JSON 的
   `comparisons.derived_native_vs_native_junit` 双向 diff 为 0，这就是你交付后我要复跑比对的目标值；
   ④复跑命令：`SEED_TMP=E:/Seed/.workbuddy/tmp python E:/Seed/.workbuddy/tmp/extract_baseline_faces.py`
   （我会用**自己重写的提取器**复核你的输出，不复用你的中间产物）。
2. **WD.2 对照实验**：基线三面计数 = native 3901 / seed 185 / rest 490 / 全量 4576（+4 模块级 skip）；
   对照时按**集合相等**判，不按数量；`baseline_nodeid_sets.json` 里的 `node_ids` 是排序列表，可直接 set() 比对。
3. **coverage 趋势**：基线 `core_line_pct = 65.25`（±0.05pp 容差）；`coverage.xml` 全局 64.78%。
   注意两者分母不同（前者是 B-4 声明面，后者全仓），不要混用。
   **补充（§12.6 三 run 实测）**：±0.05pp 容差只能用在 `core_line_pct` 上——它三 run 全等（65.25×3）；
   全局 `line-rate` 三 run 为 64.72 / 64.77 / 64.78，跨度 0.06pp，**已经越过 ±0.05pp**，
   拿它做趋势门会假警报。

### 12.6 基线恒定性：三 run 比对（应 team-lead 追加任务）

三个成功 run（时间序，均为 Ubuntu 腿、同口径提取器）：

| run | head | 时间 (UTC) | full 唯一 | native（派生） | seed | rest | 模块级 skip | collection-error | core_line_pct | 全局 line-rate |
|---|---|---|---|---|---|---|---|---|---|---|
| `40024125106` | `5e532c55f` | 10-23 21:05 | 4503 | 3828 | 185 | 490 | 2 | 0 | 65.25 | 64.72% |
| `40157476661` | `de98f6bcf` | 10-25 03:14 | 4517 | 3842 | 185 | 490 | 3 | 0 | 65.25 | 64.77% |
| `40171918472` | `499c3892f` | 10-25 07:12 | 4576 | 3901 | 185 | 490 | 4 | 0 | 65.25 | 64.78% |

（注：`40024125106` 在 GitHub 上留有 10 个 artifact、其中 `test-results-3.12` 同名 4 份——是多次 re-run 的重传件。
本提取按 artifact id 取了**最新完整组**（`12547385306` / `12546388240` / `12544807974`），选择依据与 id 已记入
`E:/Seed/.workbuddy/tmp/fetch_baseline_artifacts.py`，不靠 `gh run download -n` 的隐式择一。）

**结论：结构恒定（全部三 run 成立），内容随 commit 演进（全部差异可归因，无一项是提取抖动）。**

1. **三 run 恒定的量（WD.2 可以放心当锚）**：
   - **派生面 == 独立面：三个 run 双向 diff 全部为 0**（native、seed 各自）；
   - **native + seed + rest == full**：三 run 全部精确成立，且 native∩seed=0；
   - **collection-error：三 run 全部为 0**；
   - **seed 面（185）与 rest 面（490）三 run 完全相等**；
   - **B-4 `core_line_pct` 三 run 全等（65.25）**。
2. **不恒定的量与归因（native 面 3828 → 3842 → 3901；全量 4503 → 4517 → 4576）**，
   两两 diff 为 +30/−16 与 +59/−0，全部落在 native 面，归因三类、逐条可对证：
   - **新增用例入册**：`test_m5_ci_contract` +21（参数化用例）、`test_r2_evaluation_recovery_v2_contract`
     +29、`test_r2_engineering_recovery_entry_contract` +13、`test_r2_evaluation_recovery_preparation` +7、
     `test_seal_ast_hash_invariants` +4、`test_b0_n2_stop_reason_semantics_contract` +2、
     `test_n02_c4_net_gain_reader_contract` +2、`test_r2_resume_ledger_anchor_contract` +1、
     `test_interaction_group_online` +1（`test_online_admission_thresholds_absorb_cross_platform_float_noise`）等；
   - **模块级 skip 新增**：`test_r2_fast_ledger_resume_v4_contract` 从 10-25a 起在 Linux 上整模块跳过
     （"binds the Windows NTFS durability API at import time"）⇒ 该文件 9 条用例从面上消失（−9），
     同时全量面的模块级 skip 2→3；
   - **deselect 清单摘除/回填**：`test_cross_domain_structural_gain_gate`、
     `test_interaction_group_multifamily_leave_one_out_gate`、`test_online_interaction_structural_bridge_gate`
     及两条 relation 用例在 10-23→10-25a 间**被摘除**（ci.yml deselect 行 198→204，即族 A 第五例入册），
     又在 10-25a→10-25b 间**被摘出清单回跑**（204→188）——这正是 ci.yml 注释里
     "族 A 第五例 7 条已于 2026-10-25 随方案乙摘除"的两次操作在 junit 上的投影。
     （worker_namespace 的模块级 skip 也是 10-25b 才出现，skip 3→4。）
3. **对 WD.2 判据的直接推论**：
   - **跨 commit 的"三面集合与基线全等"不成立**——套件本身在长（两天 +73 条 native 用例）。
     WD.2 的对照必须在**同一 commit** 上做"改前/改后"，或退一步用**同 run 内**的
     "派生面 == 独立面"（本提取已证明该判据三 run 稳定成立）做恒定性锚；
   - 若必须跨 commit 比对，请把 12 个变更模块（清单见
     `E:/Seed/.workbuddy/tmp/baseline_stability_report.json` 的
     `native_module_attribution_oldest_to_newest`）按"已知入册变更"白名单归一，
     其余差异才判异常——不要抹平，逐条列出。
4. 产物：`E:/Seed/.workbuddy/tmp/baseline-40024125106/`、`E:/Seed/.workbuddy/tmp/baseline-40157476661/`
   （各自 `baseline_nodeid_sets.json`）＋ 三 run 汇总
   `E:/Seed/.workbuddy/tmp/baseline_stability_report.json`
   ＋ 提取器 `extract_baseline_faces.py`（已参数化 RUN_ID）／下载器 `fetch_baseline_artifacts.py`／
   比对器 `compare_runs.py`。

---

## 13. R-S3 独立复核（fresh eyes，2026-10-26）

对象：backend-architect 交付的 `scripts/training/split_ci_junit_faces.py`（commit `917774e70`，main）。
复核约束：不复用其中间产物——自建 worktree、自写提取/比对/夹具脚本。

### 13.1 复核环境（全部自建）

- worktree：`E:/Seed/.workbuddy/tmp/cimirror-rs3-review`（`git -c core.autocrlf=false worktree add --detach` @ `917774e70`，LF 渲染，与 CI 两腿同构；未使用对方提供的 `cimirror-baseline-499`）。
- 执行的脚本副本：`cimirror-rs3-review/scripts/training/split_ci_junit_faces.py`（从 worktree 调用，非 E:/Seed 工作树里的文件）。
- 输入：我于 §12 下载的基线 run `40171918472` artifact（`report.xml` / `native-report.xml` / `seed-report.xml`）。

### 13.2 真实基线数据复核（脚本对 report.xml 派生）

脚本期读数：`rows_total=4580 uid_total=4576 uid_native=3901 uid_seed=185 uid_other=490 module_skipped=4`，
四模块级 skip 名逐条与 §12 参考集一致（fast_ledger_v4_contract / worker_namespace_integration / browser_environment / browser_tool_loop）。**锚点 3901/185/490+4 全中。**

### 13.3 集合级比对（判"集合相等"，非数量相近）— 12/12 PASS

工具：`E:/Seed/.workbuddy/tmp/rs3_set_compare.py`（独立从派生 XML 重提集合，不复用对方中间产物）。

| # | 比对项 | 结果 |
|---|---|---|
| R1a | 派生 native 面 == 我的参考集（§12 提取器） | PASS（3901 双向 diff=0） |
| R1b | 派生 seed 面 == 我的参考集 | PASS（185 双向 diff=0） |
| R2a/b | 派生面 == 全量面点分前缀过滤（`tests.taiji_native.` / `tests.seed.`，带点防吞兄弟目录） | PASS |
| R3a/b | 派生面 == 基线**独立** native/seed junit（改前真实两趟 pytest 的产物） | PASS（3901/185 双向 diff=0） |
| R4a | other 面（全量−两面）== 490 | PASS |
| R4b | 三面并集==全量 且 行数守恒（划分无遗漏无重叠） | PASS |
| R5 | module_skipped=4 名单与参考一致 | PASS |
| R6 | classname='' 行结构性不在任何面 uid 中 | PASS |
| S1 | 派生文件头 `derived from report.xml … not a separate pytest run` 注释在场 | PASS |
| S2 | 幂等：二次执行输出字节级相同 | PASS |

R3 是关键一条：**派生面与改前独立执行面集合严格相等**，S3 去重语义成立的实证闭环。

### 13.4 退出码语义与归属陷阱（自建合成夹具）— 14/14 PASS

工具：`E:/Seed/.workbuddy/tmp/rs3_exitcode_matrix.py`（夹具仓 `rs3_review_fix/repo`，未复用对方 `rs3_synthetic/`）。

| 夹具 | 形态 | 期望 | 实测 |
|---|---|---|---|
| F0 | 合法混合面（native/seed/seed_platform/tests 根 + 1 条模块级 skip + 1 组重复行） | 0 | 0 ✅ |
| F1 | classname='' + `<error message="collection failure">` | 5 | 5 ✅（打印模块清单） |
| F2 | classname='' + `<failure>`（未知形态） | 5 | 5 ✅ fail-closed |
| F3 | classname='' 无子节点（未知形态） | 5 | 5 ✅ fail-closed |
| F4 | classname 指向不存在的磁盘文件（UNRESOLVED） | 3 | 3 ✅ |
| F5 | 全量面零 testcase | 4 | 4 ✅ |
| F6 | 源文件缺失 | 2 | 2 ✅ |
| F7 | XML 不可解析 | 2 | 2 ✅ |

F0 深检：seed_platform 用例**不入 seed 面**（`Path.is_relative_to()` 尾点陷阱结构性关闭）✅；模块级 skip 行排除出面且 `module_skipped=1` 单独记账 ✅；计数器两级重算（缺字段补 0、skipped=0 覆写）✅。

### 13.5 复核中的插曲（自我修正，非脚本缺陷）

F0f 首跑 FAIL：我误以 uid（唯一数 3）断言 junit `tests=` 字段。实测派生文件 `tests=4`（4 行 testcase，含我夹具里的重复行）——**junit `tests=` 语义是行数（含重复），uid 是去重数，两者口径不同但都对**。基线真实数据 `duplicate_node_ids=[]`，两口径无分叉，不影响任何结论。修正断言后 14/14。

### 13.6 复核结论

**独立复核通过，R-S3 可合入/已合入有效。**脚本行为与 ADR-S3（C-4/C-5）声明完全一致：
集合级三方相等（派生==参考==独立执行面）、五级退出码语义全部按约定触发、
fail-closed 面无漏（未知空 classname 形态、UNRESOLVED、零 testcase 均显式红）、
幂等、stdlib only。复核工具链留存：`rs3_set_compare.py`、`rs3_exitcode_matrix.py`、
worktree `cimirror-rs3-review`、夹具 `rs3_review_fix/`。

---

## 14. R-S3 双源复核（源 A/源 B，团队主理人派工，2026-10-26）

在 §13（源 A 集合级三方比对）基础上，按主理人派工清单对源 B（run `40207103896` Windows 腿
`report-win.xml`，head `a8024efc7`，全量 tests/、2 failure + 116 skipped）补做 a–e 逐点复核。
工具：`E:/Seed/.workbuddy/tmp/rs3_two_source_verify.py`（17 项断言）；派生输出
`rs3_review_out/winB-{native,seed}.xml`；repo-root 仍用自建 worktree `cimirror-rs3-review`（`917774e70`）。

### 14.1 逐点实测读数

| 点 | 源 A（40171918472, `499c3892f`） | 源 B（40207103896 Win, `a8024efc7`） |
|---|---|---|
| a. exit 0 + 守恒 | exit 0；3901+185+490==4576 ✅ | exit 0；3953+185+487==4625 ✅ |
| b. SET-EQUAL 划分 | 并集==全量、两两不交 ✅ | 同左 ✅ |
| c. 尾点陷阱抽检 | seed_face ⊆ `tests.seed.*`（185）、native ⊆ `tests.taiji_native.*`（3901）、`tests.seed_platform.*` 泄漏=0（全量 39 条全在 other）✅ | seed 185 / native 3953 同判、泄漏=0（seed_platform 39 条全在 other）✅ |
| d. failure 保全 | （源 A 无 failure） | 2 条 n1_plan167 TimeoutError **全部落 native 面**、classname=`tests.taiji_native.test_n1_plan167_train_decoder`、`<failure>` 子元素与 message 逐字保全 ✅ |
| e. 空 classname 普查 | 4 条全为 `<skipped message="collection skipped">`（module_skipped=4），collection-failure 形态 **0**、未知形态 **0**（exit-5 触发条件不存在，预期吻合）✅ | 2 条全为 `<skipped>`（browser_environment/browser_tool_loop），collection-failure 0、未知 0 ✅ |

### 14.2 源 B 额外读数

- 派生 stdout：`rows_total=4627 uid_total=4625 uid_native=3953 uid_seed=185 uid_other=487 module_skipped=2`。
- 全量 junit `skipped=116`（test 级 skip + 模块级 2 条），与主理人引述一致。
- **A 侧 4 条模块级 skip 在 B 只剩 2 条**：`test_r2_fast_ledger_resume_v4_contract` 与
  `test_r2_worker_namespace_integration` 在 B 正常收集（两文件在 `a8024efc7` 均在库，
  `git cat-file -e` 已验）——它们在 A 的 skip 属基线 commit 局部状态，非稳定行为。

### 14.3 跨源 native 面差异归因（AR 口径①：跨 commit 全等不作要求，逐条归因）

seed 面 **A==B 集合严格相等（185）**；native 面 diff=56（A-only 2 / B-only 54），全部归因闭合：

| 归因 | 条数 | 证据 |
|---|---|---|
| runner_contract 2 条参数化 ID 漂移（A `[1450000.0]/[1450100.0]` ↔ B `[1500000.0]/[1500100.0]`） | A−2 / B+2 | 两 commit 源码与 `BASE_PER_ARM_CPU_LIMIT=1_450_000.0` **逐字相同**；差异来自 `effective_budget_limits()`：`a8024efc7` 仓内含 v1.2+v1.3 修正 artifact（`reports/…_source_revision_v1_{2,3}.json`，`git ls-tree` 已验）⇒ 返回 approved caps `{per_arm: 1_500_000.0}`；`499c3892f` 两 artifact 均不在库 ⇒ 回落 BASE 1.45M |
| A 模块级 skip 的模块在 B 正常收集 | B+12 | fast_ledger_v4_contract 11 + worker_namespace_integration 1 |
| commit 演进新增用例 | B+42 | language_candidate v1/v2 17、n1_plan167 6、contract 6、rollout_trainer 6、n1_plan168 3、n5_38 2 等 |

### 14.4 新发现（登记，非阻塞）

**N-01（node-id 环境敏感）**：`test_start_ceiling_refusal_is_durably_recorded_before_training[<per_arm_limit>]`
的参数化 ID 内嵌**运行时解析的预算上限**（`effective_budget_limits()`，v1.3 病灶链同一模块级语句）。
跨 run/跨 commit 的 node-id 集合比对（WD.2、§12 三 run 恒定性）在预算 artifact 增删时会自然漂移
2 条——同 run 内派生面 vs 独立面对照（AR 口径①）不受影响；但**跨 commit 全等判据今后不可用**，
与 AR 口径①互为印证。建议：WD.2 第 1 个 CI 数据点（run 40226939396）比对时把这条预期进去。

### 14.5 复核插曲（自我修正，非脚本缺陷）

d 点首跑 FAIL：我用 `name`（方法名）过滤 n1_plan167，模块名在 **classname**；修正键为
`classname::name` 后 17/17。两条 failure 自始至终都在全量面且正确落 native 面。

### 14.6 结论

源 A/源 B 复核 **17/17 PASS**：a–e 全部通过，无任何不一致。源 B 比 §13 多覆盖了
failure 态保全（点 d）与跨源差异归因（§14.3）；R-S3 派生语义在第二种失败形态下依然成立。
等待第三源（run 40226939396 CI 实跑 report.xml）通知。

---

## 15. R-S3 第三源复核（run 40258536618，head `9da09708a`，CI 实跑，2026-10-26）

第三源 = WD.2 第 1 个 CI 数据点。源实物（主理人下载）：`run40258536618/junit/` 下
`u312/report.xml + native-report.xml + seed-report.xml`（Ubuntu 腿全量 + 派生面）与
`twin/report-win.xml`（Windows 腿全量）。验证工具 `rs3_third_source_verify.py`，**19/19 PASS**。

### 15.1 复核环境（新 worktree，且顺带实证 exit-3 语义）

- worktree `cimirror-src3`（`git -c core.autocrlf=false` @ `9da09708a`，LF 同构）。
- **首跑即实证了 exit-3 fail-closed**：先用 §13 的 `917774e70` worktree 跑，UNRESOLVED——
  R-S3 补充 commit 在其后新增的 `tests/test_split_ci_junit_faces_contract.py` 不在旧树里，
  `key='tests.test_split_ci_junit_faces_contract'` 解析不到磁盘文件 ⇒ exit 3。
  换 `9da09708a` 树后 exit 0。脚本在 `917774e70`→`9da09708a` 零 diff（`git diff --quiet` 已验）。

### 15.2 逐点实测读数

| 点 | u312（Ubuntu 腿） | twin（Windows 腿） |
|---|---|---|
| a. exit 0 + 守恒 | `3957+185+516==4658` ✅（与主理人 derive 读数逐字一致） | `3969+185+513==4667` ✅ |
| b. SET-EQUAL 划分 | 并集==全量、两两不交 ✅ | 同左 ✅ |
| b. N1-167 failure | **2 条**全落 native 面（`tests.taiji_native.test_n1_plan167_train_decoder`），`<failure>` message=`TimeoutError: N1-167 fixed diagnostic CPU/wall ceiling reached` 逐字保全 ✅ | 同左 2 条 ✅ |
| e. 空 classname 普查 | 4 条全 `<skipped>`（module_skipped=4），collection-failure 0、未知形态 0 ✅ | 2 条全 `<skipped>`（browser×2），0/0 ✅ |
| c. 派生 vs CI 实跑 face | **双向 diff=0**（native 3957、seed 185）；换行规范化后**逐字节相等** ✅ | （CI 未发布 twin 派生面，无从比对——S3 只在 Ubuntu 腿派生，符合设计） |
| e. N-01 参数化 ID | `[1500000.0]/[1500100.0]`，与预案一致（`9da09708a` 仓内 3 个预算修正 artifact，`git ls-tree` 已验）✅ | — |

### 15.3 点 d：module_skipped=4 名单归因（修正派工描述）

名单与源 A 逐字相同：fast_ledger_v4_contract / worker_namespace_integration / browser×2。
**变化说明与源 B 观察的归因修正**：fast_ledger_v4:10 与 worker_namespace:16 的 skip 条件是
`if sys.platform != "win32": pytest.skip(..., allow_module_level=True)`（源码已读）——
**平台门，非 commit 状态**：Ubuntu 腿恒 skip（源 A、第三源 u312 一致），Windows 腿恒收集
（源 B、第三源 twin 一致）。源 B"该两模块正常收集"是 Windows 腿的平台现象，两源观察并不矛盾。

### 15.4 跨腿差异归因（不抹平，逐条闭合）

| 差异 | 条数 | 归因 | 证据 |
|---|---|---|---|
| twin native − u312 native | +12 | 两个 Win-only 模块在 Windows 腿正常收集（fast_ledger_v4 11 + worker_namespace 1） | 15.3 的 `sys.platform` 门 |
| u312 other − twin other | −3 | `tests.resonance.test_real_checkpoint_slow` 3 条（:33/:43/:57 均 `@pytest.mark.slow`）被 Windows 腿 `-m "not slow"`（ci.yml:884）marker 挤出；Ubuntu 腿（ci.yml:499）无 marker 过滤 | junit 集合 diff + ci.yml 两腿命令对照 |
| seed 面 | 0 | 两腿严格相等（185） | — |

此即 §11 已记录的"deselected 数 = 清单生效数 + marker 挤出数"中 marker 挤出项的跨腿显形：
**跨腿 node-id 集合比对必须预期 marker 挤出差**，同 run 派生 vs 独立对照不受影响（AR 口径①再次被印证）。

### 15.5 复核插曲（自我修正 ×2，非被测脚本缺陷）

1. b 点首跑漏检全部 failure：我的验证器用 `tc.find("failure") or tc.find("error")`——ET 无子元素
   节点为 falsy，`<failure>`（带文本无子元素）全被吞；改显式 `is None` 判定后 19/19。
2. c 点首跑 byte FAIL：我的派生输出经 Windows `write_text` 换行翻译（45413 个 CRLF），CI 产物 LF；
   换行规范化后逐字节相等。产物内容本身一致，差异纯在我本机写文件方式。

### 15.6 结论与旁证

**第三源复核通过（19/19）**：R-S3 派生语义在真实 CI 数据面上成立——CI 发布的 native/seed face
与我独立重跑的派生输出集合/字节双等，N1-167 failure 保全，模块级 skip 与 N-01 漂移全部按预案命中。
旁证：主理人通报的 collection-crlf 探针红（autocrlf=true 检出下 D-01 族 19 模块 collection error）
与 §3/§4.3/D-01 的证据链预测完全吻合——Q7(iii) 探针的"真发现"判据成立。
遗留观察：twin 腿 CI 未发布派生面（S3 仅 Ubuntu 派生），Windows 腿面数据如需对照只能本机派生（本节已做）。

---

## 回流判定（最终版）

- 结论：**通过**
- 回流目标：（无）
- 返工原因：（无）
- 说明：本阶段是只读取证，上游不需要返工。取证命中了目标——
  建立了同构复现方法（`SEED_ROOT` + `git -c core.autocrlf=false` 干净 worktree，已钉 5 棵树），
  并在真正的 CI 失败 commit（`29980ee3b`，而非 intake 记的 `fbc212397`）上 100% 复现了
  `Interrupted: 19 errors during collection` 与 `v1.3 boot audit or source binding differs`。
  修复（`565604a9e`）在本机 LF 干净检出上已验绿，但同时发现同一条病灶链上的**另一个活雷 D-01**——
  它是给 Phase 4 的输入（结构裁决），不构成"上游必须先改我才能过关"的阻塞，
  因此按遗留风险 L1 登记，不触发回流。§4 里的所有具体值都可在 §3.2 的三步里复现。
  后续追加工作（§10~§13：junit 依赖面实测、W0.4/M3a、WD.2 三 run 基线恒定性、
  R-S3 独立复核）均无回流触发点。

## §16 M5/skipped 注入实验读数（主理人，2026-10-26）

### §16.0 基线登记（硬性 gate 输入，run 40288176554 @ 571431884）

**结论形态**：run conclusion=failure（预期内），总时长 59m59s。job 级：
- `test (3.12)` **success**：五门禁（deselect-guard / pytest-full / junit-faces / coverage-gate）
  outcome 全 success，gate-summary success；pytest-full 实跑 51m17s 无 timeout。
- `test-windows` **failure**：归因闭环 = pytest-windows outcome=failure（coe 吞 conclusion 为
  success——API quirk 第 3 次实证）→ gate-summary-windows 正确判红
  （::error::「至少一个门禁 step 未完成成功（outcome≠success）；本 job 判红」+ exit 1）。
- junit 实物（report-win.xml）：**失败恰 2 条 = `test_n1_plan167_train_decoder` 两用例，
  `TimeoutError: N1-167 fixed diagnostic CPU/wall ceiling reached`** ——唯一允许红位精确命中，零额外红位。
- **collection-crlf 探针回绿（count=0，exit 0，无 error 注解）——D-01 改后锚成立**
  （改前锚：40258536618 探针 19 模块红；两锚齐，D-01 正式关闭，AR §B.7 验收口径满足）。
- CVE advisory（coe 成员，五门禁之外）exit 1 被 coe 吞 = F10-R3 既定 advisory 模式常态
  （pip-audit 发现 9 漏洞/3 包即退出 1；R6 评估中高危为传递依赖，维持 advisory 是落账决策，
  非 S6 门禁盲区）。

**硬性 gate 判定：通过**（探针 count=0 + 唯一红位 N1-167）→ 注入实验按 brief 终版执行。
