# Phase 2 工程计划 · 根治 Seed CI 长红

- 计划时间: 2026-10-26（GMT+8）
- 岗位: implementation-planner（程可拆）
- 运行目录: `E:/Seed/delivery-runs/20261025-235503/`
- 上游: `artifacts/01-intake.md`（§2 取证 / §3 范围 / §4 DoD / §7 给下游）、`board.md`（决策 1 / 决策 2）
- 本阶段性质: **只读规划**。未修改仓库任何文件；唯一写入为本文件与 board.md 的 Phase 2 小节。
- 仓库基线: 初版读取时 `main` @ `60f32a95f`；**修订版 v2 时 `main` @ `979217e93`**（所有行号以 `979217e93` 为准）

> **修订记录 v3（2026-10-26，收到 api-tester `08-test.md` §11 补充取证后）**
> 1. **新增第 7 个结构缺陷 S7**：run 40207103896 的 step25 实测 **`6 failed, 3821 passed`**，
>    6 条全落在**结构性成长门族**（`*_gate`），**不在 deselect 清单内**，与 S1 无关（浮点/阈值判据）。
>    ⇒ **D1「同 commit 双绿」当前在这 6 条上不成立**。已新增 WG 组、M-S7 门、R13、Q9。
> 2. **更正一处归属**：api-tester 把这 6 条记为「全量回归 step」；实测日志 2901–2906 行属于
>    **step25 `Run native architecture contracts`**（step27 在该 run 中被跳过）。
>    ⇒ step25 墙钟 **50m58s**（不是我初版用的 38m15s，后者来自另一 run 的快机）。
> 3. **L2（Linux 侧未实测）已证伪** ⇒ R12 关闭。证据：同 run 日志**第 2758 行**
>    `deselect 清单 172 条，命中 172 条`（本人已 grep 复核）⇒ 那 5 个 r2_* 文件在
>    Linux + LF 检出下收集成功，`seal_ast` 映射能吃下 `/home/runner/work/Seed/Seed` 前缀。
> 4. **W0.4 夹具已落地为脚本**；W0.5 的 `deselected` 计数口径改为 **B−A 差分**；
>    **M3a 加入 D2 的反向自校验**（摘掉清单后是否如期红）。
> 5. 行号基准再更新：HEAD 已从 `979217e93` 推进到 **`720fb22a6`**。

> **修订记录 v2（2026-10-26，收到 api-tester `08-test.md` 取证后）**
> 本次修订不是文字润色，是**两处前提被实测推翻**：
> 1. **假设 A1 证伪** —— pytest 9.1.1 的 junit `<testcase>` **不带 `file` 属性**（§8 假设表已标「已证伪」）。
>    ⇒ S3 主路径从「`file` 属性派生」改为「`classname` 前缀派生」，并新增**两级自校验**（§3.3）。
> 2. **W0.2 的完成标准不可达** —— 我原写「指出唯一失败子条件」，实测是 **2 个**（`boot_audit_path` 与
>    `boot_accounting_sources` 同时 False），且**只存在于 `29980ee3b`，不在当前 HEAD**。
>    ⇒ M0 验收门已改写（§5），WB 组的范围与验收已扩（§2 WB / §3.1）。
> 另新增 D-01（extension.py:678 活雷）为一个独立工作包 WB.0，并新开待裁决项 Q7/Q8（§8）。

---

## 0 结论（一页）

1. **执行顺序不能按 S1→S2→S3→S4 的编号走**。必须先修「时间面」（S3 去重 + S4 并发 + 守卫提速），
   因为当前**一次完整 CI 反馈环 ≈ 1.5h 且大概率被取消**，任何针对 S1/S2 的改动都只能在这样的环里验证。
   先修路、再修车。
2. **S2 的自伤不止一处**。除 intake 已记的「守卫红 ⇒ 三个 pytest step 跳过」外，本阶段新取证发现：
   Ubuntu 腿三个 pytest step 与 B-4 覆盖率门**同样没有 `continue-on-error` / `if: always()`**，
   实测 run 40185913504 中 step25 一红，step26/27/28 全部 skipped。
   ⇒ 「一步红 ⇒ 全链失明」是**流水线级结构缺陷**，编号建议为 **S6**，需 team-lead 确认是否纳入范围。
3. **S1 的候选路径不能只按「改动小」排序**。DoD D3 明确要求「治理校验不再在导入期决定模块可导入性」，
   因此只修跨机映射（候选 P4）虽能消红，**不满足 D3**；必须与 P1/P3 之一组合，分两刀落。
4. **守卫现在是绿的，但红点并没有消失，只是换了形态**。HEAD/LF（＝CI 同构）落空 0/86、守卫 success；
   而 **HEAD/CRLF 仍是 19 模块 collection error**，炸在 `extension.py:678`。
   ⇒ 计划不得假设「现在仍然红」，也**不得因此认为 S1 已经解决**。
5. **S1 的病灶链已经复发过一次**（v1.3 位点 → v1.1 位点，共同结构是「机器局部外观写成冻结证据 +
   导入期 `!=` 比较」）。这条实证是 WA.1 裁决 P1/P3 时最硬的理由：只拆当前的位点，还会有第三个。
6. **一刀一回滚点是硬要求**：全部工作包按「单 commit + pathspec」落地，共 **14** 个回滚点（见 §6）。
   未跟踪项 `delivery-runs/`（本运行目录，位于仓内）、`.zcode/`、`output/playwright/client-ui-20261024/`
   一律不得进入任何 commit。

---

## 1 本阶段新取证（2026-10-26 只读实测，补充 01-intake）

> 全部为本次会话 `gh` / `Read` / `Grep` 真实读数；行号已逐条打开核对，不沿用 intake 数字。

### 1.1 规模与结构

| 事实 | 读数 |
|---|---|
| `.github/workflows/ci.yml` | 815 行（`wc -l` 报 814，末行无行尾换行） |
| `.github/workflows/m7-ci.yml` | 55 行 |
| 测试文件总数（`find tests -name 'test_*.py'`） | **808**：`tests/taiji_native` 706、`tests/` 根 56、`tests/seed` 33、`tests/seed_platform` 8、`tests/resonance` 5 |
| `tests/` 下 `__init__.py` 数量 | **0**（⇒ junit 的 `classname` 不带目录信息，见 §3.3 假设 A1） |
| conftest 数量 | 1（`tests/conftest.py`） |
| deselect 条目 | **Ubuntu 86 条 / Windows 86 条，逐条 diff 完全一致**（`sort` 后 `diff` 输出空） |
| Ubuntu env 块 | `ci.yml:106`（`env:`）→ `ci.yml:107`（`DSH_CI_DESELECT: >-`）→ 条目 `ci.yml:108–192` |
| Windows env 块 | `ci.yml:666`（`env:`）→ `ci.yml:667`（`DSH_CI_DESELECT: >-`）→ 条目 `ci.yml:668–752` |
| 守卫脚本 | `scripts/training/verify_ci_deselect_list.py`（89 行）；`listed_entries()` 29–44 行**扫全文件所有以 `--deselect` 开头的行**（⇒ 读到 172 条＝两腿各一份）；`main()` 47–85 行，**每条起一个 `pytest --co -q` 子进程**（63–77） |

### 1.2 墙钟实测（GitHub Actions 真实 job/step 时间戳）

**run 40207103896**（HEAD `a8024efc7`，在跑）—— Ubuntu `test (3.12)` job：

| 区间 | 起→止 | 耗时 |
|---|---|---|
| step1–6 setup/install | 15:28:54 → 15:30:16 | 1m22s |
| step7–13 ruff/black/mypy/pip-audit | 15:30:16 → 15:32:38 | 2m22s |
| step14–22 八个 verify 脚本 | 15:32:38 → 15:35:59 | 3m21s |
| **step24 守卫（172 次 `pytest --co`）** | 15:36:00 → 15:42:24 | **6m24s（success）** |
| step25 `Run native architecture contracts` | 15:42:24 → 仍在执行 | — |

**run 40185913504**（2026-10-25，cancelled）：

| step | 起→止 | 结论 |
|---|---|---|
| step24 守卫 | 10:49:16 → 10:54:35 | 5m19s，**success** |
| **step25 `Run native architecture contracts`** | 10:54:35 → 11:32:50 | **38m15s，failure** |
| step26 `Run Seed model-boundary contracts` | 11:32:50 → 11:32:50 | **skipped** |
| step27 `Run complete regression suite` | 同上 | **skipped** |
| step28 B-4 覆盖率门 | 同上 | **skipped** |

**run 40207103896**（2026-10-25，**已完成**，两腿均 failure）—— Ubuntu 腿 `test (3.12)`：

| step | 起→止 | 结论 |
|---|---|---|
| step24 守卫 | 15:36:00 → 15:42:24 | 6m24s，**success**（日志 2758 行：`deselect 清单 172 条，命中 172 条`） |
| **step25 `Run native architecture contracts`** | 15:42:24 → 16:33:25 | **50m58s，failure** —— `6 failed, 3821 passed, 115 skipped, 85 deselected, 1 xfailed` |
| step26 / 27 / 28 | 16:33:25 → 16:33:25 | **全部 skipped** |
| 整个 job | 15:28:53 → 16:33:29 | 64m36s，failure |
| Windows 腿整个 job | 15:28:53 → 17:05:16 | 96m23s，failure |

⇒ **两处必须由 v3 修正的读数**：
① **step25 的真实墙钟是 50m58s，不是 38m15s**。同一个 step 在不同 run 上差 1.33×
（公开 runner 池硬件差，ci.yml:32 注释已记 ~2×）⇒ **S3 的收益估算与 WD.4 的 timeout 回缩都必须用
50m58s 这个较慢读数做基线**，否则会在慢机上再次撞顶。
② **step27（唯一的 `--cov` 全量腿）在已查的两个 run 中都没有执行过**（均被前置 step 跳过）
⇒ **近期没有 step27 的完整墙钟读数**。WD.4 回缩 timeout 前必须先有一次 step27 实测，
而这只有在 S6 落地后才会发生 ⇒ **WD.4 的依赖应加上 WE.1**（v3 已改）。

⇒ **新发现 S6a**：`ci.yml:467–491` 三个 pytest step 与 `ci.yml:507–508` B-4 门**均无 `continue-on-error`、
无 `if: always()`**（step29/30/31 有 `if: always()`）。任一红 ⇒ 其后全部 step 被跳过，
本次 CI 的覆盖率结论与 junit 产物一并丢失。与 DoD D1「三个 pytest step 实际执行」直接冲突。

⇒ **新发现 S6b**：Ubuntu 腿把 22 个前置门禁（实测 7m05s）与 pytest 串行在同一 job 内，
既拉长单腿，又让「lint 红」与「测试红」互相掩盖。

### 1.3 快照与时态（api-tester W0 已定论，来源 `artifacts/08-test.md`）

**时态必须按 commit 分开说，混在一起会得出相反结论**：

| 检出形态 | extension import | collection | deselect 落空 |
|---|---|---|---|
| HEAD `979217e93` **LF**（＝CI 两腿同构） | **OK** | 无 error | **0 / 86** |
| HEAD `979217e93` **CRLF** | **FAIL** `v1.1 source revision review report binding differs` | **`Interrupted: 19 errors`（28.29s）** | 6 / 86 |
| `29980ee3b` **LF**（＝CI 失败 run 真身） | **FAIL** `v1.3 boot audit or source binding differs` | **`Interrupted: 19 errors`（28.38s）** | 12 / 93 |

推论（写入计划的前提）：

- **红点 B 在 HEAD/LF 上已不复现**，修复由 `565604a9e` 落地；intake §2.2 把失败 commit 记成 `fbc212397`
  是错的（实测 `gh run view 40203419431 --json headSha` = `29980ee3b8d65890edba89e8d44be32538fac19a`）。
  ⇒ **在 HEAD 上无法复现当年的红**，任何针对 S1 的验证必须钉在具体 sha 上（本人已复核该 sha）。
- **S1 没有死干净，换了个灶台在烧（D-01）**：HEAD 在 **CRLF** 检出下仍是 19 模块 collection error，
  炸在更前一级的 `extension.py:678`（`_sha256_file(_SOURCE_REVISION_V1_1_REVIEW) != review_sha`，
  严格字节比较）—— 同族的 v1.2 / v1.3 位点已被 `29980ee3b`/`fbc212397`/`565604a9e` 改成
  `rendering_tolerant_match`，**就剩它没改**（本人已打开 `extension.py:672–680` 核对，678 行确认）。
- **今天 CI 躲过 D-01 的唯一原因**是 Windows 腿把 `git config --global core.autocrlf false`
  排在 `actions/checkout` **之前**（`ci.yml:702–705`）。这是**步骤顺序依赖**，不是修复。
- **「快且中断」与「慢且失败」不是两种病**（修正我初版的推测）：Ubuntu 腿有守卫 step 挡在前面，
  守卫红 ⇒ pytest 被跳过；Windows 腿没有守卫 step，pytest 直接开跑 ⇒ 28–48s 就在导入期被打断。
  ⇒ 复现方法**一套就够**（干净 worktree + `--co`），干预点在 `import` 不在用例。

### 1.3b junit 形态实测（假设 A1 的证据，来源 `08-test.md` §10）

- pytest 9.1.1：`<testcase>` **只有 `classname` + `name` + `time`**，pass/fail/skip **三态均无 `file` 属性**。
- `classname` 是**相对 rootdir 的点分路径**（rootdir 变 ⇒ 前缀变；`tests/` 下无任何 `__init__.py`，
  实测 count=0，故 `classname` 不带目录信息这一担忧**不成立**——它带，例如
  `tests.taiji_native.test_r2_fast_ledger_resume_v4_contract`）。
- **collection error 会被写进 junit，但 `classname=""`**（`name` 是点分模块路径，算进 `tests`/`errors`）
  ⇒ 按前缀归桶的三张面**一条也抓不到它**。
  ⇒ 「三张面用例数之和 == 全量面」在发生 collection error 时**必然不等**，这是性质不是 bug。

### 1.4 S1 的改动面（grep 实测，供 §3.1 取舍用）

`extension.py:1548` 产出的两个常量的消费点，**绑定风格不同，直接决定候选路径的可行性**：

- **按属性访问（`X.PER_ARM_CPU_LIMIT`，惰性化友好）**
  - `scripts/training/prepare_taiji_r2_continuation_endpoints.py:126,136,137`
  - `scripts/training/run_taiji_r2_checkpoint_resume_adapter.py:102,133`
  - `scripts/training/run_taiji_r2_fast_ledger_resume.py:112,182`
  - `scripts/training/run_taiji_r2_fast_ledger_resume_v2.py:143`
- **按值 import / 全局重绑定（惰性化有断裂风险）**
  - `scripts/training/run_taiji_r2_daily_dialogue_continuation_extension.py:45,47`（`from ... import`）
    且 `1203–1204` 用 `global PER_ARM_CPU_LIMIT, AGGREGATE_CPU_LIMIT` 重新绑定
  - `tests/taiji_native/test_r2_continuation_budget_amendment_v1_4_contract.py:6,7`
  - `tests/taiji_native/test_r2_continuation_budget_amendment_v1_5_contract.py:6,7`
  - `tests/taiji_native/test_r2_dialogue_continuation_extension_contract.py:72,73`
    `monkeypatch.setattr(extension, "PER_ARM_CPU_LIMIT", ...)`、`144` 行直接赋值
- 引用这两个常量的测试文件共 **20 个**。
- `run_taiji_r2_daily_dialogue_continuation_extension.py:260` 断言
  `continuation_budget.effective_budget_limits() != (PER_ARM_CPU_LIMIT, AGGREGATE_CPU_LIMIT)`
  ⇒ 任何惰性化方案都必须保持「函数返回值 == 常量值」这一不变量。

---

## 2 工作分解（WBS）

图例 — 负责人：`TL` delivery-team-lead ／ `AR` architecture-reviewer ／ `IMP` backend-architect(CI/治理脚本) ／
`REV` code-reviewer ／ `AT` api-tester ／ `TW` tech-writer ／ `PM` product-manager（仅 Phase 14）。

### W0 组 · 诊断前置（决策 2：与 Phase 2 并行，已在办）

| # | 工作包 | 负责人 | 输入 | 输出 | 完成标准 | 依赖 |
|---|---|---|---|---|---|---|
| W0.1 ✅ | **本机同构复现方法**：构造与 GitHub runner 同构的检出（LF、干净 worktree、无未跟踪产物、`fetch-depth:0`）并复现 `ExtensionLedgerRefuse` | AT | intake §2.3；`extension.py:684–771`；`seal_ast.py:91/96/277/326` | 可重复执行的复现脚本 + 复现/不复现结论 | **已达成**（`08-test.md` §3.2 三步配方 + 5 棵已钉 worktree） | 无 |
| W0.2 ✅（标准已改写） | **定因**：给出失败判据的**左右值原文 + sha 绑定** | AT | W0.1 | 定因结论 | **已达成，但结论形态与初版不同**：失败子条件是 **2 个不是 1 个**（`boot_audit_path` 与 `boot_accounting_sources` 均 False，同源于「修订单把机器绝对路径＋当时工作树字节写成冻结证据」）；且**只存在于 `29980ee3b`**，HEAD 上三条件**全 True**。⇒ **「指出唯一失败子条件」这个初版标准作废**，M0 门已按此改写 | W0.1 |
| W0.3 ✅ | **当前时态快照** | AT | gh run/job API | 快照小节 | **已达成**：HEAD/LF 全绿（落空 0/86）；HEAD/CRLF 仍红（19 errors）；`29980ee3b`/LF 红（19 errors） | 无 |
| W0.4 🆕 | **CRLF 面回归夹具**：把 `cimirror-head-crlf` 的构造配方固化成一条命令，供 WB 组每刀复用 | AT | W0.1 | 一条可重复命令 + 判红断言 | 任何人在任意机器上一条命令即可产出「HEAD/CRLF ⇒ 19 errors」或「⇒ 0 errors」 | W0.1 |
| W0.5 🆕 | **守卫计数口径修正**（配合 WC.1）：确认「186 / 172 条」是**两腿 ×2 计数**，落空 12 条（不是 13 条） | AT | `08-test.md` §5.3 | 口径说明进 `08-test.md` | 后续任何「条目数净减少」的度量都标明是「去重后 per-leg 条数」 | 无 |

### WA 组 · Phase 4 架构裁决（AR）

| # | 工作包 | 负责人 | 输入 | 输出 | 完成标准 | 依赖 |
|---|---|---|---|---|---|---|
| WA.1 | **S1 结构裁决**：从 §3.1 的 P1–P4 中选定路径（可组合），并写明取舍与被否选项的理由 | AR | W0.2 定因；§1.4 消费面清单；intake §3 红线 | ADR-S1 进 `04-architecture.md` | ADR 明确「是否保留导入期校验」「判据语义是否变化（触及红线须回 owner）」「改动文件清单」 | W0.2 |
| WA.2 | **S2 机制裁决**：从 §3.2 的 S2-A/B/C 中选定替代机制；S2-D（守卫提速/判定口径）建议必做 | AR | §1.1 清单结构；§1.2 守卫耗时；intake §4 D2 | ADR-S2 | ADR 明确「清单是否保留 node-id 形态」「守卫是 blocking 还是 warn」「双腿一致如何结构性保证」 | W0.3 |
| WA.3 | **S3 去重裁决**：从 §3.3 的 S3-A/B/C/D 中选定，并回答「三张 junit 面语义如何保全」与「覆盖率面是否改变」 | AR | §1.1 规模；§1.2 step25=38m15s；`ci.yml:481–508` | ADR-S3 | ADR 明确覆盖率分母是否变化（变化即触碰 intake §3「覆盖率阈值不动」）、三张 artifact 是否改名 | 无 |
| WA.4 | **S4 并发裁决**：从 §3.4 的 S4-a/b/c/d 中选定，明确「是否保留 `cancel-in-progress: true`」及其代价 | AR | intake §2.1；`ci.yml:11–13`；`m7-ci.yml:16–18` | ADR-S4 | ADR 给出选定方案的排队积压上界估算与「owner 一天十余次 push」下的最坏反馈时延 | 无 |
| WA.5 | **S6 门禁解耦裁决**（决策 3 已确认纳入范围）：三个 pytest step + B-4 门的失败传播方式、Ubuntu 腿是否拆 job | AR | §1.2 S6a/S6b 实测；**决策 3 三项强制约束** | ADR-S6 | ① ADR 明确「continue-on-error + 汇总判红」还是「拆 job」；② **四个 `continue-on-error` 与汇总 step 同 commit**；③ **S6-ii（拆 job）仅在 S6-i 不足以达成 M2 时间目标时才做** | 无 |
| WA.6 | **回滚与提交序列裁决**：确认 §6 的 13 个回滚点粒度与 pathspec 清单 | AR + TL | §6 | `04-architecture.md` 附提交序列表 | 每个回滚点对应唯一 commit、唯一 pathspec、无跨包混合 | WA.1–WA.5 |

### WB 组 · Phase 6 实现 · S1（IMP）

| # | 工作包 | 负责人 | 输入 | 输出 | 完成标准 | 依赖 |
|---|---|---|---|---|---|---|
| WB.0 🆕 | **D-01 活雷拆除**：`extension.py:678` 的 `_sha256_file(_SOURCE_REVISION_V1_1_REVIEW) != review_sha` 改渲染容忍（与前几个 commit 改 v1.2 / v1.3 位点同法），或按 WA.1 选定的等效方案 | IMP | WA.1；W0.2/W0.4 | 改动 commit **R-S1D** | **LF 与 CRLF 两套检出都跑 `pytest tests/ --co -q`，均 0 error**（W0.4 夹具）；现有 LF 面不回退 | WA.1, W0.4 |
| WB.1 | **根因修复刀**：目标已由 W0.2 收敛 —— v1.3 位点（`extension.py:750–763`）**已由 `565604a9e` 修好并在 HEAD/LF 验绿**，本刀**不再需要改它**；保留为「若 W0.2 后续发现残留位点」的兜底工作包 | IMP | WA.1；W0.2 | 改动 commit R-S1a（**可能为空**） | 本机同构环境（W0.1）下 import 成功；两腿不再出现 19 模块 collection error | WA.1, W0.2 |
| WB.2 | **结构刀（满足 D3）**：按 WA.1 选定的 P1/P3 落地，使治理校验不再决定模块可导入性 | IMP | WA.1；WB.1 | 改动 commit R-S1b | ① `python -c "import scripts.training.taiji_r2_dialogue_continuation_extension"` 在**人为使校验失败的注入场景**下仍能 import；② 20 个引用常量的测试文件全部通过；③ `run_taiji_r2_daily_dialogue_continuation_extension.py:260` 的不变量仍成立 | WB.1 |
| WB.3 | **失败语义补钉**：确保治理拒绝在真正使用预算的路径上仍 fail-closed，且有明确报错 | IMP | WB.2 | 改动 commit R-S1c | 新增/复用一条测试：注入失败时，调用预算的入口抛 `ExtensionLedgerRefuse` 且错误信息可定位到具体子条件 | WB.2 |

### WC 组 · Phase 6 实现 · S2（IMP）

| # | 工作包 | 负责人 | 输入 | 输出 | 完成标准 | 依赖 |
|---|---|---|---|---|---|---|
| WC.1 | **守卫提速与判定口径**：`verify_ci_deselect_list.py:63–77` 改为「一次全量 `--co`（带 `--continue-on-collection-errors`）+ **精确 node-id 集合**比对」，并区分「收集失败」与「条目落空」；输出改为**去重后 per-leg 条数** | IMP | WA.2；W0.5 | 改动 commit R-S2a | ① 守卫 step 实测从 5–6m 降到 ≤2m；② 落空判定改精确匹配（不再子串）；③ 打印「去重后 N 条/腿」而非 ×2 数；④ 判定结果与旧实现在至少一次对照运行中一致 | WA.2, W0.5 |
| WC.2 | **清单单源化**：替代机制的数据源落地（S2-A 的外置清单 或 S2-B 的 marker 注册 + `pyproject.toml:68–73` markers） | IMP | WA.2 | 新增/改动文件 + commit R-S2b | 双腿消费同一数据源；**结构上不可能再出现两腿不一致**（有守卫或 CI 校验自证） | WA.2 |
| WC.3 | **守卫不再一键跳过测试**：按 WA.5 落地阻断降级/汇总判红 | IMP | WA.5 | 改动 commit R-S2c | 注入一次「守卫失败」，验证其后三个 pytest step **仍执行**且 job 仍判红 | WA.5, WC.1 |
| WC.4 | **清单分批收紧（D2 前半）**：按族分批摘除，每批附本机 + CI 双绿证据 | IMP + AT | WC.2；intake §6 R3 | 每批 1 个 commit（R-S2d-1…n） | 清单条目数**净减少**；剩余每条在数据源中带「重启条件」字段且本机已验证 | WC.2, WB.2 |

### WD 组 · Phase 6 实现 · 时间面 S3/S4（IMP）

| # | 工作包 | 负责人 | 输入 | 输出 | 完成标准 | 依赖 |
|---|---|---|---|---|---|---|
| WD.1 | **S3 去重**：按 WA.3 落地；若走 S3-A，新增 junit 分面脚本 | IMP | WA.3 | 改动 commit R-S3 | ① Ubuntu 腿只跑一次 `tests/`；② 三张面内容去重前后逐 testcase 可比对；③ `ci.yml:535–544` artifact 不改名；④ 分面按 `classname` 前缀（**带尾点**、rootdir 钉死）；⑤ **两级自校验落地**（`classname=""` 前置判红 + 集合相等断言） | WA.3 |
| WD.2 | **S3 语义保全对照实验**：去重前后三张 junit 的 testcase 集合对比（**两级**） | AT | WD.1 | 对照报告进 `08-test.md` | ① **先**断言全量面无 `classname=""`（有则判红并列出模块名，本项即失败）；② 通过后断言三张面 node-id 之并 == 全量面；③ 差异逐条解释（只允许「重复执行」的消除，不允许用例消失） | WD.1 |
| WD.3 | **S4 并发治理**：按 WA.4 落地 `ci.yml:11–13` | IMP | WA.4 | 改动 commit R-S4 | 变更后在连续 ≥3 次 push 的场景下，至少有一次 run 跑完整两腿并给出结论 | WA.4 |
| WD.4 | **timeout 回缩**：单腿墙钟下降后，把 `ci.yml:33` / `ci.yml:661` 的 180m 按实测下调，并把上调轨迹补进注释 | IMP | WD.1, WD.3 | 改动 commit R-S4b | 新 timeout ≥ 实测最长腿 ×1.5，且注释写明实测依据与取值日期 | WD.1, WD.3 |

### WE 组 · Phase 6 实现 · S6（IMP）

| # | 工作包 | 负责人 | 输入 | 输出 | 完成标准 | 依赖 |
|---|---|---|---|---|---|---|
| WE.1 | **Ubuntu 腿门禁解耦**：按 WA.5 落地（continue-on-error + 汇总 step，或拆 job） | IMP | WA.5；决策 3 | 改动 commit R-S6a（**四个 `continue-on-error` + 汇总判红 step 必须同 commit**） | 注入一次 step25 红，验证 step26/27/28 仍执行，且 **job conclusion 仍为 failure**（决策 3 约束②） | WA.5 |
| WE.2 | **（可选，受决策 3 约束③限制）前置门禁拆 job**：把 `ci.yml:239–446` 的 lint/format/mypy/verify 与 pytest 拆成并行 job | IMP | WA.5 | 改动 commit R-S6b | 仅在 WA.5 判定 S6-i 不足以达成 M2 时间目标时才做；拆后两 job 结论独立，单腿墙钟再降约 7m（对照 §1.2） | WA.5, WE.1 |

### WF 组 · Phase 7 / 8 / 13 / 14

| # | 工作包 | 负责人 | 输入 | 输出 | 完成标准 | 依赖 |
|---|---|---|---|---|---|---|
| WF.1 | 逐刀代码审查 | REV | WB/WC/WD/WE 各 commit | `07-review.md` | 每刀过审或给出返工判定（回流 board.md 触发本计划修订） | 各 commit |
| WF.2 | 本机同构回归 + CI 双绿取证 | AT | 全部改动 | `08-test.md` | 见 §5 验收门 M6 | WF.1 |
| WF.3 | CI 规范条目与台账（deselect 台账、守卫语义、并发与 timeout 约定、S5 技术债登记） | TW | WF.2；本文件 §3 | `13-documentation.md` | 台账含每条剩余 deselect 的重启条件与责任人；S5 单列为技术债条目 | WF.2 |
| WF.4 | 终验 DoD D1–D5 | TL + PM | 全部产物 | `14-acceptance.md` | 五条 DoD 逐条有证据位置，无一条靠「CI 变绿了」结案 | WF.2, WF.3 |

---

## 3 候选方案与取舍（**不裁决**，供 Phase 4 使用）

> 本节只列「改动面 / 级联风险 / 验证成本」，选型权归 architecture-reviewer，判据语义的放行权归 owner。

### 3.1 S1：导入期治理校验

现状：`scripts/training/taiji_r2_dialogue_continuation_extension.py:1546–1548` 为模块级语句
`PER_ARM_CPU_LIMIT, AGGREGATE_CPU_LIMIT = effective_budget_limits()`；
`effective_budget_limits()` 自 684 行起，v1.3 分支三条件（750–763）任一不成立即在 **771 行** raise。

**W0 定因后的关键变化（必须先读，否则会选错刀）**：

- v1.3 位点（750–763）**已经修好了**（`565604a9e`），HEAD/LF 实测三条件全 True、collection 0 error。
  ⇒ **不要再把 WB.1 的目标定成「修 750–763」**，那是已交付的活。
- 但**同一条病灶链已经炸了两次**：先是 v1.3 位点（绝对路径轴 + 渲染轴），现在是 v1.1 位点
  （`extension.py:678`，渲染轴，CRLF 下必炸）。两次的共同结构是
  **「把机器局部外观（绝对路径 / 工作树字节渲染）写成冻结证据，然后在导入期用 `!=` 搬回来比」**。
- 这条共同结构正是 **D3 要求消除的东西**。⇒ 只拆 D-01（P5-a）能消当前的红，**不能阻止第三次**；
  这是 WA.1 裁决 P1/P3 时最有力的一条理由：**该病灶链已经复发一次，有实证**。
- 顺带：`seal_ast` 的 C3 绿**不在位字节通道上**，靠 git 历史取证兜住（`08-test.md` §4.2 / L4）
  ⇒ 依赖 `fetch-depth: 0`，改浅克隆前必须重跑 W0.4 夹具。

| 候选 | 改法 | 改动面 | 级联风险 | 验证成本 |
|---|---|---|---|---|
| **P1 惰性求值** | 1548 行改为 PEP 562 模块级 `__getattr__`（或缓存包装），首次取用时才校验 | `extension.py:1546–1548`、1551–1570（`__all__`）；需处理 §1.4 全部「按值 import」点 | ① `test_r2_dialogue_continuation_extension_contract.py:72–73` 的 `monkeypatch.setattr` 会先触发 getattr ⇒ 校验失败时在 setattr 前就炸；② `run_taiji_r2_daily_dialogue_continuation_extension.py:1203–1204` 的 `global` 重绑定与 `__getattr__` 语义需重新定义；③ **判据语义从「导入即校验」变为「首用才校验」** —— 治理保证被推迟，是否可接受属 owner 放行项 | 高。需 W0.1 同构环境 + 20 个常量引用测试文件全回归 + 两腿 |
| **P2 conftest 层降级为 skip** | 不改治理代码；在 `tests/conftest.py` 加 collection 钩子，捕获导入期 `ExtensionLedgerRefuse` 并把模块转为 skip（带 reason） | `tests/conftest.py`（新增钩子）；治理代码零改动 | ① 把「治理拒绝」翻译成 skip 会掩盖真红，与 D3「带明确失败语义」张力大 —— 需把 skipped 模块数作为门禁输入；② pytest 在 collection 期拦截导入异常的能力需实测；③ 19 模块被 skip ⇒ 其 deselect 条目仍会落空 ⇒ **必须与 WC 组联动** | 中。**可不依赖真机复现**：本机注入强制 raise 即可验证 |
| **P3 校验挪出常量初始化** | 1548 行改绑 BASE/占位；v1.3 治理校验移入显式入口（如 `verify_v13_boot_audit()`），由真正需要预算的调用点显式调用 | `extension.py:1546–1548`、684–783（函数拆分）、`__all__`；调用点：`run_taiji_r2_daily_dialogue_continuation_extension.py:260/1204`、`run_taiji_r2_resume.py:215`、`prepare_taiji_r2_continuation_endpoints.py:126–137`、`run_taiji_r2_checkpoint_resume_adapter.py:102/133`、`run_taiji_r2_fast_ledger_resume.py:112/182`、`run_taiji_r2_fast_ledger_resume_v2.py:143` | ① 预算来源从「导入期钉死」变「调用期取」，任一消费点漏改即静默用 BASE 值 —— 需在每处加断言；② 与 intake §3 红线相邻（是否算「改变封印/判据语义」需 owner 裁定）；③ 260 行的等式断言需重新表述 | **最高**（改动面最广） |
| ~~**P4 只修 v1.3 映射层**~~ | — | — | — | **已由 `565604a9e` 交付并在 HEAD/LF 验绿，本候选作废** |

**W0 后新增候选（针对 D-01 活雷）**：

| 候选 | 改法 | 改动面 | 级联风险 | 验证成本 |
|---|---|---|---|---|
| **P5-a 678 行改渲染容忍**（推荐） | `_sha256_file(_SOURCE_REVISION_V1_1_REVIEW) != review_sha` ⇒ `not seal_ast.rendering_tolerant_match(review_sha, _SOURCE_REVISION_V1_1_REVIEW.read_bytes())` | `extension.py:678`（单行，与前几 commit 改 v1.2/v1.3 位点同法） | ① 与 v1.2/v1.3 位点**同法同语义**，判据语义不变的论证最直接；② 仍需 owner 确认「容忍比较 ≠ 放宽判据」（红线表述） | **低**：W0.4 夹具 LF/CRLF 双跑即可 |
| **P5-b `.gitattributes` 钉死渲染** | 给 `plans/reference/**/*.md` 加 `eol=lf` 规则 | `.gitattributes`（新增规则） | ① 影响面是**所有引用这些 md 的判据**（不止 678），可能连带改变其它位点的实测字节；② 已入库的工作树不会自动重写，需 `git add --renormalize` ⇒ 会产出一次大范围 diff，**与并行会话冲突风险高（R2）** | 中 |
| **P5-c 不改代码，只钉住 CI 顺序** | 保持 Windows 腿 `core.autocrlf false` 在 checkout 之前（`ci.yml:702–705`），把「为什么必须这个顺序」写进注释判据 | `ci.yml:702–705` 注释（零行为改动） | ① **活雷没拆**，只是把避雷针立住；② 任何重排 checkout 步骤 / 换自托管 Windows runner / 带 `autocrlf=true` 的机器都会立即翻红 | 最低。**建议与 P5-a 同时做**（P5-a 拆雷，P5-c 留判据） |
| **P5-d CRLF 面进 CI**（新增守护，非修复） | 加一个只做 collection 的廉价 step/job：`autocrlf=true` 检出 + `pytest tests/ --co -q`，**不跑用例** | 新增 1 个 step 或 1 个 job | ① 成本 ~2–3 min（本机 collection 28s，runner 慢约 1.7× ⇒ ~1min，加检出 ~2min）；② 会与 D4（降墙钟）轻微冲突，需 TL 权衡 | 低。**正好覆盖 S1 这类「导入期炸」的病**（`--co` 不跑用例却会触发导入） |

**计划侧建议（供裁决参考，不替 AR 拍板）**：
**WB.0（P5-a + P5-c，两行改动）＋ WB.2（P1 或 P3 结构刀）分两刀**。
即使结构刀被 owner 驳回，WB.0 已经把「CRLF 必炸」这颗雷拆掉且可独立回滚；
而 WB.2 才是阻止第三次复发的那一刀 —— 「已经复发过一次」是它最硬的论据。

### 3.2 S2：CI-only deselect 机制

现状：`ci.yml:108–192`（Ubuntu）与 `ci.yml:668–752`（Windows）**两份人工同步的清单**（实测逐条一致）；
`verify_ci_deselect_list.py` 每条起一个 `pytest --co` 子进程（实测 5–6 分钟），任一落空即 `exit 1`，
其后三个 pytest step 全跳过。

**两个口径瑕疵（api-tester §5.3 实测，会影响「条目数净减少」的度量）**：
① 守卫 `listed_entries()` 扫**整份 ci.yml** ⇒ 两腿**重复计数**：CI 打印的「186 条」＝93×2，「172 条」＝86×2。
⇒ **任何「清单条目数」的度量必须标明是「去重后 per-leg 条数」**，否则 D2 的「净减少」会被口径污染。
② 命中判定是子串 `if method in collected.stdout` ⇒ 用例名若恰为另一条 node id 的子串，**落空会被判成命中**
（与它想防的「静默失效」反向同源）。⇒ WC.1 必须改用**精确 node-id 集合**匹配。

| 候选 | 改法 | 改动面 | 级联风险 | 验证成本 |
|---|---|---|---|---|
| **S2-A 清单单源外置** | 清单移到仓内单一数据源（如 `tests/ci_deselect.txt` 或 YAML，带「重启条件/登记日/责任人」字段）；两腿各加一个 step 读出并注入 `$GITHUB_ENV`；守卫改读同一数据源 | 新增数据源文件；`ci.yml:106–193`、`666–753` 两处 env 改为运行时注入；`verify_ci_deselect_list.py:24–44` 改读数据源 | ① `$GITHUB_ENV` 多行注入需处理（建议生成单行空格串，与当前 `>-` 折叠等价）；② 两腿各需一个新 step（+2 step，耗时可忽略）；③ 数据源若被误删，守卫须报 fail-closed | 低。结构上消除「两腿手工同步」，直接服务 D2 |
| **S2-B 改用 pytest marker** | 86 条 node-id 改为在用例上打 `@pytest.mark.ci_unproven`；三步改 `-m "not ci_unproven"`；`pyproject.toml:68–73` 注册 marker | 约 60+ 个测试文件加 marker；`pyproject.toml`；`ci.yml:467–491`、`801–805`；守卫脚本改判「被 mark 且确被排除」 | ① 改动面最大；② marker 粒度（模块/函数）与现有 node-id 级不完全一致（部分文件只排 1–2 条）；③ 需开 `--strict-markers` 否则拼错的 marker 静默失效（与当前「落空静默」同形缺陷） | 高。**唯一能真正消灭清单腐化的机制**（marker 跟随用例，改名/移动不失效） |
| **S2-C 守卫降级为 warn** | 守卫加 `continue-on-error: true` + `::warning::` + job summary 打印落空清单；清单仍保留 node-id 形态 | `ci.yml:455–465` 一处 + 守卫脚本输出改 warning | ① 守卫失去阻断力 ⇒ 腐化会静默回归；② 缓解手段：Phase 13 台账周期性审计 + summary 可见 | 最低。**立刻消除「守卫红⇒测试全跳过」**（D2 第二半句） |
| **S2-D 守卫提速与判定口径**（与 A/B/C 正交，建议必做） | `verify_ci_deselect_list.py:63–77` 每条一个子进程 ⇒ 改为**一次全量 `--co` + 集合比对**，并用 `--continue-on-collection-errors` 区分「收集失败」与「条目落空」 | 仅 `verify_ci_deselect_list.py` | ① 全量收集 vs 单文件收集：某文件收集失败时，其余条目判定更准（优于现状）；② 判定口径变化需与旧实现做一次对照运行 | 低。**实测可省 5–6 分钟/腿**（§1.2） |

### 3.3 S3：Ubuntu 腿 tests/ 跑三遍

现状：`ci.yml:467–473`（`tests/taiji_native`）、`475–479`（`tests/seed`）、`481–491`（`tests/ --cov`）。
`tests/taiji_native` 706 个文件占全量 808 的 **87%**；实测 step25 单跑 38m15s（run 40185913504）。
只有 step27 带 `--cov`，其产物 `coverage.json` / `coverage.xml` 分别喂 `ci.yml:507–508` B-4 门
与 `ci.yml:520–533` trend summary。

| 候选 | 改法 | 「三张面语义」如何保全 | 风险与代价 |
|---|---|---|---|
| **S3-A 单跑全量 + junit 派生子面**（推荐） | 只保留 step27 的 `pytest tests/ --cov --junitxml=report.xml`；新增脚本按用例所属目录从 `report.xml` 派生 `native-report.xml` / `seed-report.xml` | 三张 artifact 名与路径不变（`ci.yml:535–544`）；覆盖率面**完全不变**（仍是全量带插桩）⇒ B-4 门与 trend 数字不受影响；「native 面」「seed 面」退化为同一份 junit 的目录视图 | **派生键从 `file` 改为 `classname`**（A1 已被 pytest 9.1.1 实测证伪，`08-test.md` §10）。三条硬约束见下方「S3-A 落地约束」 |
| ~~**S3-B 三分区互不重叠**~~ ❌ **已排除（决策 5）** | — | — | 「覆盖率分母变小」触碰 intake §3「覆盖率阈值不动」。TL 已裁定：**S3-A 的分母本就不变，不需要请示 owner**；S3-B 才需要请示，故直接排除。**退回 S3-B 属红线事项，届时须重新请示 owner** |
| **S3-C 拆并行 job** | `taiji_native` 与「其余」拆两个 job，覆盖率合并（`coverage combine`） | 面各自真实，但覆盖率需合并后喂 B-4 | 中等复杂度；需处理跨 job 产物流转与合并失败语义；Windows 腿结构不同，双腿不再同构 |
| **S3-D xdist 并行** | 加 `pytest-xdist`，`-n auto` | 面不变 | Ubuntu runner 2 核 ⇒ 收益有限（约 1.5–2×）；需新增依赖（install step 变更）。**登记为可选项，不在本次默认执行面** |

**S3-A 落地约束（A1 证伪后的替代设计，必须写进 ADR-S3）**：

1. **派生键 = `classname`，前缀必须带尾点**：用 `tests.taiji_native.`（不是 `tests.taiji_native`），
   否则 `tests.seed.` 会误吞 `tests.seed_platform.*`（`tests/seed_platform` 实测 8 个文件）。
2. **rootdir 必须钉死**：`classname` 是「相对 rootdir 的点分路径」，rootdir 变 ⇒ 前缀变 ⇒ 分面全错。
   ADR-S3 须写明 CI 上 rootdir = checkout 根，并把「改变 rootdir 的改动需重跑 WD.2」写成判据。
3. **自校验做成两级，不能只做一级**：
   - **① 前置门禁**：全量面 junit 里出现**任何 `classname=""` 的 `<testcase>`** ⇒ **直接判红并打印模块名**。
     （collection error 会被写进 junit 且 `classname` 为空串 ⇒ 按前缀归桶的三张面一条也抓不到它。
     把这一条做成显式红，等于给 S1/S6 加了一层安全网：**「collection error ⇒ 默默少跑」从此不可能静默**。）
   - **② 通过 ① 之后**，再断言三张面 node-id 集合之并 == 全量面 node-id 集合。
   - 若不分级，collection error 发生时 ② 会自然不等，会被误诊成「去重丢面」（R5），排查方向就错了。

### 3.4 S4：并发治理

现状：`ci.yml:11–13` `group: ${{ github.ref }}` + `cancel-in-progress: true`；
近 12 次 run ＝ 3 failure / 8 cancelled / 1 in_progress（本轮 `gh run list` 复核，与 intake 一致）。
`m7-ci.yml:16–18` 同款配置（不在范围内，但裁定口径应一致以便 Phase 13 写规范）。

| 候选 | 内容 | 保留 cancel-in-progress 的取舍与代价 |
|---|---|---|
| **S4-a 直接关闭** | `cancel-in-progress: false` | 代价最大：owner 一天十余次 push × 单腿 1.5h ⇒ 队列积压可达十余小时，反馈更慢。单独使用不推荐 |
| **S4-b 按事件分化** | `cancel-in-progress: ${{ github.event_name == 'pull_request' }}` | PR 上保留取消（省额度、反馈快）；push 到 main 全部跑完（串行排队）⇒ **main 上一定能拿到完整结论**。代价：main 队列仍会积压，需与 S3 缩短单腿配合 |
| **S4-c 快慢分流** | 新增快 workflow（lint/format/mypy/verify/守卫，实测 ~7m，`cancel: true`）；ci.yml 保留全量但 `cancel: false` | 快门始终有反馈；慢门不取消。代价：多一个 workflow 文件要维护；两个 workflow 的判据一致性需台账约束 |
| **S4-d 先缩短再评估** | 只做 S3；单腿降到 ~45m 后再决定是否保留 `true` | 保留 `true` 的代价随腿长下降而线性下降；但十余次 push/日下仍大概率互删 |

**计划侧建议**：S4-d + S4-b 组合（先 WD.1 缩短，再 WD.3 按事件分化），并把结论写进 Phase 13 台账。

### 3.5 S6（新发现，范围待 TL 确认）

| 候选 | 改法 | 风险 |
|---|---|---|
| **S6-i continue-on-error + 汇总判红** | 给 `ci.yml:467/475/481/507` 四步加 `continue-on-error: true`，并新增一个 `if: always()` 的汇总 step 统一判 exit code | 若漏掉汇总 step，job 会**变绿** ⇒ 比现状更糟。汇总 step 是本方案的必要组成，不可省略 |
| **S6-ii 拆 job** | lint/format/mypy/verify 拆为 `quality-gates` job，pytest 拆为 `pytest` job，互不 needs | 单腿墙钟再降 ~7m（§1.2）；代价：job 数增加，需确认并发额度（intake §2.1 记峰值并发 7 < 20 上限） |

---

## 4 执行顺序与可并行窗口

```
【并行窗口 0】✅ 已完成
  W0.1 同构复现 ──► W0.2 定因（结论：2 个子条件，仅在 29980ee3b）─┐
  W0.3 时态快照（HEAD/LF 绿、HEAD/CRLF 红、299/LF 红）──────────┤
  W0.4 CRLF 夹具（供 WB.0 复用）───────────────────────────────┤
  W0.5 守卫计数口径（×2 / 子串）───────────────────────────────┤
                                        ▼
【串行门 M1】 WA.1–WA.6 架构裁决（W0.2 已就绪，可立即裁决 S1）
        │
        ├─► WD.1 S3 去重 ──► WD.2 对照实验 ──┐   ← 与 WB/WC 无依赖，可先做
        ├─► WC.1 守卫提速 ───────────────────┤
        ├─► WD.3/WD.4 并发 + timeout ────────┤
        │                                     ▼
        │                          【门 M2】时间面达标
        │                                     │
        ├─► WE.1/WE.2 门禁解耦 ───────────────┤
        │                                     │
        ├─► WB.0 D-01 拆雷 ─►【门 M3a】LF+CRLF 双绿
        │        │
        └─► WB.1 根因刀（可能为空）─► WB.2 结构刀 ─► WB.3 失败语义 ─►【门 M3】D3
        │                                     │
        └─► WC.2 单源化 ─► WC.3 守卫降级 ─► WC.4 分批收紧 ─►【门 M4】D2
                                              ▼
                                   【门 M5】S6 解耦验证
                                              ▼
                                   【门 M6】双绿验收 D1
                                              ▼
                              WF.3 文档台账 ► WF.4 终验 D1–D5
```

**可并行**：
- **决策 4 已确认的并行窗口**：WD 组（动 `.github/workflows/ci.yml` + 新增脚本）与 WB 组（动
  `scripts/training/`）**文件面不重叠 ⇒ 可并行开发**，但**提交序列仍按 WD → WC.1 → WE → WB → WC.2-4 落地**。
- W0 组与 WA 组的 WA.3/WA.4/WA.5（不依赖 W0.2）可并行。
- WD 组（S3/S4）与 WB、WC、WE 组**互相独立**，可并行开工 —— 但 WD 必须先落地，
  因为它把后续每一轮验证的成本从 1.5h 压下来。
- WC.1（守卫提速）与 WC.2（单源化）可并行，但落地应分开 commit（§6）。
- WE.1 与 WC.3 都动「失败传播」，**必须串行**（先 WE.1 后 WC.3），否则两个 `continue-on-error`
  的叠加语义无法归因。

**必须串行**：
- W0.2 → WA.1 → WB.1 → WB.2 → WB.3（定因未出，结构刀无处下）。
- WD.1 → WD.2 → WD.4（timeout 只能按去重后的实测回缩）。
- WC.2 → WC.4（先有单源数据结构，才能逐条登记重启条件并分批摘除）。

**关键路径**：`WA.1 → WB.0 → WB.2 → M6`。
（**R1 已解除** —— W0.2 已定因，不再是关键路径上的不确定段。新的最长不确定段是
**WA.1 对 P1/P3 的裁决**：它是唯一需要 owner 放行红线的决策，驳回则 D3 无法达成。）

---

## 5 里程碑与验收门（对应 DoD D1–D5）

| 门 | 里程碑 | 验收内容 | 谁验收 | 失败退回 |
|---|---|---|---|---|
| **M0** ✅ | 诊断就绪（**判据已按决策 7 修订**） | **在 CI 失败 run 自身的 head commit（`29980ee3b`）上完成定因**，并给出失败判据的**左右值原文 + sha 绑定**（实测为 **2 个**子条件：`boot_audit_path`、`boot_accounting_sources`，不是初版写的「唯一」）；W0.3 三档时态快照；**W0.4 CRLF 夹具一条命令可复跑** | TL | （已达成）后续若 HEAD 再推进，须在新 sha 上重跑 W0.4 夹具再确认。注：「HEAD 上三条件全绿」是**修复已落地的证据**，不是「未能定因」 |
| **M1** | 裁决完成 | `04-architecture.md` 含 ADR-S1/S2/S3/S4/S6 + 提交序列；每条 ADR 写明被否选项的理由 | TL | 退回 AR 补取舍；判据语义类争议上报 owner |
| **M2** | 时间面达标（D4 前半） | ① Ubuntu 腿只跑一次 `tests/`；② WD.2 **两级**对照通过（先无 `classname=""`，再三面之并 == 全量面）；③ 守卫 step ≤2m；④ 单腿墙钟较基线（1h17m–1h30m）**下降 ≥30%** | AT + REV | 退回 WD.1；若覆盖率面被改变 ⇒ 上报 owner 裁定基线 |
| **M3a** 🆕 | D-01 活雷拆除（WB.0） | **LF 与 CRLF 两套干净检出**各跑一次 `pytest tests/ --co -q`，**均 0 error**；且 LF 面相对基线无回退（仍 0 error、落空 0/86）；若选 P5-c，则 `ci.yml:702–705` 的顺序判据已写进注释 | AT | 退回 WB.0；若 `rendering_tolerant_match` 被 owner 判为「放宽判据」⇒ 停刀，走 P5-b/P5-c + Q8 上报 |
| **M3** | S1 消除（D3） | ① 注入治理失败场景时模块**仍可 import**；② 真正使用预算的路径仍 fail-closed 且报错可定位；③ 20 个常量引用测试文件全绿；④ **LF 与 CRLF 两套检出均 `pytest tests/ --co -q` 0 error**（W0.4 夹具，WB.0 的验收）；⑤ 两腿无 collection error | AT | 退回 WB.2 / WB.0；若触及红线 ⇒ 停刀，回 owner |
| **M4** | S2 机制迁移（D2） | ① 清单条目数**净减少**（基线 86 条/腿）；② 剩余每条带「本机已验证的重启条件」；③ 双腿一致由结构保证（有自证校验）；④ 注入守卫红 ⇒ 三个 pytest step **仍执行** | AT + REV | 退回 WC.2 / WC.3；收紧暴露真红 ⇒ 按 R3 分批，不得一次收完 |
| **M5** | 门禁解耦（S6） | 注入 step25 红 ⇒ step26/27/28 仍执行，且 job conclusion 仍 failure（**不得变绿**） | AT | 退回 WE.1（多半是漏了汇总 step） |
| **M6** | 双绿（D1） | 同一 commit 上 Ubuntu 腿与 Windows 腿均绿，且三个 pytest step 状态为 **success（非 skipped）**；证据为 gh job/step 读数 | TL | 退回 M5；若 Windows 腿单独红 ⇒ 回 M3 复查平台侧差异 |
| **M7** | 落账（D5） | 13 个回滚点一一对应真实 commit；每个 commit 的 pathspec 不含 `delivery-runs/`、`.zcode/`、`output/playwright/`；并行会话改动未混入 | TL + REV | 退回 IMP 重做提交（禁止 force-push 已推送的历史，改用 revert commit） |
| **M8** | 终验（D1–D5） | `14-acceptance.md` 五条 DoD 逐条附证据位置 | TL + PM | 缺证据即不结案 |

**DoD 映射**：D1→M6；D2→M4；D3→M3；D4→M2 + WD.4；D5→M7（每刀见 §6）。

---

## 6 一刀一回滚点

> 纪律：每刀 = 1 个 commit + 显式 pathspec + 独立可 revert。禁止把两刀并进一个 commit。
> 当前未跟踪项必须排除：`delivery-runs/`（本运行目录在仓内）、`.zcode/`、`output/playwright/client-ui-20261024/`。

| 回滚点 | 对应工作包 | commit 主题（建议） | pathspec（建议） | 回滚方式 |
|---|---|---|---|---|
| **R0** | — | `chore(ci): 基线快照（无代码改动，仅留档）` | 无（打 tag 或记 commit sha） | — |
| **R-S4** | WD.3 | `fix(ci): 并发治理按事件分化（push 不取消，PR 取消）` | `.github/workflows/ci.yml` | `git revert <sha>` |
| **R-S2a** | WC.1 | `perf(ci): deselect 守卫改为单次收集+集合比对（5-6m→≤2m）` | `scripts/training/verify_ci_deselect_list.py` | revert |
| **R-S3** | WD.1 | `perf(ci): Ubuntu 腿 tests/ 只跑一次，native/seed 面由 junit 派生` | `.github/workflows/ci.yml` + 新增的分面脚本 | revert（脚本随之删除） |
| **R-S4b** | WD.4 | `fix(ci): 单腿 timeout 按实测回缩（180m→N）` | `.github/workflows/ci.yml` | revert |
| **R-S6a** | WE.1 | `fix(ci): 三个 pytest step 与 B-4 门解耦+汇总判红` | `.github/workflows/ci.yml` | revert |
| **R-S6b** | WE.2（可选） | `refactor(ci): 前置门禁拆为并行 quality-gates job` | `.github/workflows/ci.yml` | revert |
| **R-S1D** 🆕 | WB.0 | `fix(governance): v1.1 review report 校验改渲染容忍（拆 D-01 活雷）` | `scripts/training/taiji_r2_dialogue_continuation_extension.py`（仅 678 行）；若同做 P5-c 则另起一 commit 只动 `.github/workflows/ci.yml` 注释 | revert（P5-a 与 P5-c 分两个 commit，可分别回滚） |
| **R-S1a** | WB.1 | `fix(governance): v1.3 boot audit 跨机判定残留位点修复（不动模块级绑定）` | `scripts/training/seal_ast.py` 或 `scripts/training/taiji_r2_dialogue_continuation_extension.py`（二选一，不得同时） | revert（**本刀可能为空** —— v1.3 位点已由 `565604a9e` 修好，若 W0.2 无残留发现则不落 commit） |
| **R-S1b** | WB.2 | `refactor(governance): 预算治理校验移出模块级常量初始化` | `scripts/training/taiji_r2_dialogue_continuation_extension.py` + 受影响调用点 | revert（R-S1a 可独立保留） |
| **R-S1c** | WB.3 | `test(governance): 补预算入口 fail-closed 契约` | 新增/改动的测试文件 | revert |
| **R-S2b** | WC.2 | `refactor(ci): deselect 清单单源化（双腿消费同一数据源）` | 新增数据源文件 + `.github/workflows/ci.yml` + `pyproject.toml`（若走 marker） | revert（数据源与 yml 一并回退，避免半同步态） |
| **R-S2c** | WC.3 | `fix(ci): 守卫不再阻断后续 pytest step` | `.github/workflows/ci.yml` | revert |
| **R-S2d-1…n** | WC.4 | `fix(ci): deselect 批次 N 摘除（附本机+CI 双绿证据）` | 数据源文件（若 S2-A）／测试文件（若 S2-B） | **逐批 revert**（这就是分批的理由） |

- 顺序约定：先 R-S4 / R-S2a / R-S3（时间面），再 R-S6a，再 **R-S1D（拆 D-01 活雷）**，
  再 R-S1a→R-S1b→R-S1c（结构刀），最后 R-S2b→R-S2c→R-S2d-*。
- 回滚点总数由 13 增至 **14**（新增 R-S1D；若 P5-c 单独成 commit 则为 15）。
- 每个 commit 落库前用 `git status --porcelain` 确认无未跟踪项被 `git add -A` 卷入。
- **禁止**用 `git push --force` 修正历史；回滚一律用 revert commit（便于 Phase 13 台账记录）。

---

## 7 风险缓冲与前置验证

| 编号 | 风险 | 触发条件 | 缓冲 / 前置验证 |
|---|---|---|---|
| ~~**R1**~~ ✅ **已解除** | 本机无法同构复现 | — | api-tester 已建立同构复现方法（`SEED_ROOT` + `git -c core.autocrlf=false` 干净 worktree，已钉 5 棵），并在 `29980ee3b` 与 HEAD/CRLF 两条路径上均 100% 复现。**不再需要靠 CI 试错** |
| **R1b** 🆕 | **HEAD 在动导致结论跨 sha 失效**（api-tester L5） | W0 取证期间 main 已从 `60f32a95f` 推到 `979217e93` | 每刀开工前重跑 **W0.4 夹具**（一条命令）确认 LF/CRLF 两档结论仍成立；所有结论绑定 sha 写入 commit message |
| **R2**（沿用） | v1.3 boot audit 属 owner 在飞的治理面，改动与并行会话打架 | WB.1/WB.2 开工时 `main` 又推进了治理 commit | **前置验证**：每刀开工前 `git log --oneline -5 -- scripts/training/seal_ast.py scripts/training/taiji_r2_dialogue_continuation_extension.py`；若有新 commit 先重跑 W0.3 快照再动手 |
| **R3**（沿用） | deselect 收紧暴露真红 | WC.4 某批次后 CI 出现新增 failure | 缓冲：严格按族分批（建议 ≤10 条/批），每批独立 commit + 双绿证据；一批红则只 revert 该批 |
| **R4**（新增） | **S6 的 `continue-on-error` 让真红变绿** | WE.1 落地后 job 出现「有失败步骤但 conclusion=success」 | **前置验证**：M5 验收门强制注入一次红并断言 job conclusion=failure；汇总 step 与四个 `continue-on-error` 必须同 commit 落地 |
| **R5**（新增） | **S3 去重丢面**：派生脚本过滤条件写错，某张 junit 面用例缺失 | WD.2 对照实验出现用例数差异 | **前置验证**：WD.2 必须在**同一 commit 的前后两次运行**上做逐 testcase 集合比对，差异必须逐条解释；不允许「看起来差不多」结案 |
| **R6**（新增） | **S4 关闭取消后队列积压**，反馈反而更慢 | WD.3 落地后 main 上 queued run ≥5 | 缓冲：WD.3 必须在 WD.1（缩短单腿）之后落地；积压超阈值则回退到 S4-b 的 PR-only 取消 |
| ~~**R7**~~ ✅ **已关闭（假设被证伪，不是风险）** | junit 无 `file` 属性 | 已由 W-V1 / api-tester 双次实测 | pytest 9.1.1 确认 `<testcase>` 只有 `classname`+`name`+`time`。⇒ S3-A 派生键改 `classname`，落地约束三条已写入 §3.3；**不再是风险，是设计要求**。决策 5 的兜底链（node-id 映射 → S3-C）**未触发** |
| **R8**（新增） | 本仓并行会话多，`main` 在计划执行期持续推进，行号/sha 漂移 | 任一刀开工时 | 缓冲：本文件所有行号基于 `979217e93`（已从 `60f32a95f` 更新一次）；每刀开工前须重新核对涉及行号并在此文件修订记录里留痕 |
| **R9** 🆕 | **D-01 换个形态再复发**：只拆 678 行，不拆「导入期 + 渲染敏感字节比较」这个结构 ⇒ 会有第三个位点 | WB.2（P1/P3）被 owner 驳回或未落地 | 缓冲：WB.0（拆雷）与 WB.2（拆结构）**分两刀**，WB.0 先交付；另建议 P5-d（CRLF collection 探针进 CI，~2–3min）作为长期哨兵 |
| **R10** 🆕 | **rootdir 漂移导致分面全错**（`classname` 是相对 rootdir 的点分路径） | 任何改变 pytest rootdir 的改动 | 缓冲：ADR-S3 把 rootdir 写死为 checkout 根，并把「改 rootdir ⇒ 必重跑 WD.2」写成判据注释 |
| **R11** 🆕 | **C3 的绿依赖 git 历史取证**（api-tester L4）：在位字节不命中，靠 `file_commits` 的渲染变体兜住 | CI 改成浅克隆（`fetch-depth` ≠ 0），或相关文件历史被改写 | 缓冲：把「`fetch-depth: 0` 是判据依赖」写进 `ci.yml` 注释与 Phase 13 台账；改浅克隆前必跑 W0.4 夹具 |
| **R12** 🆕 | **Linux 侧未实测**（api-tester L2）：Ubuntu 腿根目录 `/home/runner/work/Seed/Seed`，本机 Windows 无法证伪 `seal_ast` 映射能吃下任意前缀 | WB.0/WB.2 只在本机（Windows）验证即合并 | 缓冲：WB.0 的验收必须包含**一次真实 Ubuntu 腿 run**；若本机无法证伪，就在 CI 上以 W0.4 夹具形式做一次只读验证 step（不阻塞主流程） |

---

## 8 假设与待确认问题

### 假设（未经验证即写入计划，执行前必须验证）

- ~~**A1**~~ ❌ **已证伪**（api-tester §10 实测，pytest 9.1.1）：`<testcase>` **不带 `file` 属性**。
  ⇒ 原「`file` 派生 + `--co` node-id 兜底」方案作废，改为 **`classname` 前缀派生**（§3.3 三条约束）。
  附带纠正：我初版写「`tests/` 下无 `__init__.py` ⇒ `classname` 不可用」是**错的** ——
  实测 `classname` 仍为 `tests.taiji_native.test_x`（相对 rootdir 的点分路径），可用。
- **A2**：`$GITHUB_ENV` 注入单行空格串与当前 YAML `>-` 折叠等价（S2-A 依赖）。需在 WC.2 首次落地时验证。
- **A3**：`tests/taiji_native`（706 文件）在 `tests/` 全量中耗时占绝对多数
  ⇒ 去掉 step25/26 后单腿墙钟下降 ≥30%。依据：step25 实测 38m15s vs 全腿基线 1h17m–1h30m。
- ~~**A4**~~ ✅ **已定论**：19 模块 collection error = 导入期 `ExtensionLedgerRefuse`，
  **不区分 OS，只区分检出渲染**（LF 绿 / CRLF 红）。⇒ **Windows 腿不需要单独的复现环境**，
  一套 `git -c core.autocrlf=false/true` worktree 即可覆盖两腿（api-tester §3.3 矩阵）。
- **A5** 🆕：把 `extension.py:678` 改成 `rendering_tolerant_match` 属于「同法同语义」，
  **不构成放宽判据**（同族 v1.2 / v1.3 位点已如此改过且 owner 已接受）。此假设需 owner 口头确认（Q3 的一部分）。

### 待确认问题（需 team-lead / owner 裁决，不在本岗职权）

- ~~**Q1**~~ ✅ **已裁：纳入**（决策 3）。S6 属本次执行面，附三项强制约束：
  ① 四个 `continue-on-error` 与「汇总判红 step」**同 commit** 落地；
  ② M5 门必须注入一次红并断言 `job conclusion=failure`；
  ③ S6-ii（拆 job）仅在 WA.5 判定 S6-i 不足以达成 M2 时间目标时才做。
- ~~**Q2**~~ ✅ **已裁：批准**（决策 4）。执行顺序定为 **WD（S3/S4/timeout）→ WC.1（守卫提速）→
  WE（S6）→ WB（S1）→ WC.2–4（S2 收紧）**。已接受「合并窗口内 CI 仍跑三遍」的 runner 浪费。
  边界：WD 组动 `.github/workflows/ci.yml` + 新增脚本，WB 组动 `scripts/training/`，
  **文件面不重叠 ⇒ 可并行开发，但提交序列仍按上述顺序落地**。
- **Q3（给 owner，经 AR 转达）**：WA.1 若选定 P1/P3，**「导入即校验」改为「首用才校验」是否触及
  intake §3 红线「不改变任何冻结判据/封印语义」**？需要 owner 显式放行。
- ~~**Q4**~~ ✅ **已撤销请示**（决策 5）：直接裁定走 S3-A，S3-B 排除。
  （TL 判其为伪二选一：S3-A 分母本就不变，不触红线；把「不需要请示的方案」与「需要请示的方案」
  并列去问 owner 是把选型义务外包。）⇒ **退回 S3-B 属红线事项，届时须重新请示 owner。**
- ~~**Q5**~~ ✅ **已裁**（决策 6）：`m7-ci.yml:16–18` 同款并发配置**只登记不改**；
  S4 裁定口径写入 Phase 13 台账供后续一致。
- ~~**Q6**~~ ✅ **已裁**（决策 6）：S5 仅由 WF.3 单列技术债条目，本次不改行为。
- **Q7** 🆕（给 TL）：**CRLF 面要不要进 CI？** 三档可选：
  (i) 不进，只在本机/W0.4 夹具验（成本 0，但 CI 不守护，D-01 类病会重演）；
  (ii) 加一条 CRLF 全腿（+45～90min，与 D4 直接冲突，**不推荐**）；
  (iii) **加一个只做 collection 的廉价 step/job**（`autocrlf=true` 检出 + `pytest tests/ --co -q`，
  不跑用例，~2–3min）—— 正好覆盖 S1 这类「导入期炸」的病。**计划侧推荐 (iii)**。
- **Q8** 🆕（给 owner，经 AR 转达）：若 P5-a（`rendering_tolerant_match`）被判为「放宽判据」而不得放行，
  则在 **P5-b（`.gitattributes` 钉渲染，但会产出大范围 renormalize diff，与并行会话冲突）** 与
  **P5-c（永久保留 Windows 腿 autocrlf 顺序 + 注释判据，活雷不拆）** 之间选哪一条？

---

## 9 给下游的注意事项

0. **W0 取证后的三条「别照旧做」**（v2 新增，最容易踩）：
   - **别再试图在 HEAD 上复现当年的红**。失败 run 的真身是 `29980ee3b`（不是 intake 记的 `fbc212397`）；
     HEAD/LF 已经绿了。要复现历史红点就 `git worktree add --detach <wt> 29980ee3b`。
   - **别只跑 LF 就宣告胜利**。HEAD/CRLF 仍然 19 模块 collection error（`extension.py:678`）。
     任何碰治理面的改动，**LF 与 CRLF 两套检出都要跑**（W0.4 夹具一条命令）。
   - **别把「清单 172 条」当成条目数**。守卫扫整份 ci.yml ⇒ 两腿 ×2；真实是 **86 条/腿**。
     D2 的「净减少」一律用**去重后 per-leg 条数**度量。
1. **不要按 S1→S2→S3→S4 的编号顺序动手**。先时间面（WD/WC.1），再 S6，再 **WB.0 拆 D-01**，
   再 S1 结构刀，最后 S2 收紧。理由：每一次验证环现在值 1.5h 且大概率被取消。
2. **ci.yml 的注释即判据**。本文件引用的 `ci.yml:22–32`（timeout 上调轨迹）、`34–105`（deselect 分批理由）、
   `194–218`（matrix 取舍）等注释，**删除或改写字面内容前必须确认其描述的判据仍然成立**；
   WD.4 回缩 timeout 时应在原注释后追加新读数，不要覆盖历史轨迹。
3. **守卫脚本 `verify_ci_deselect_list.py:29–44` 会扫到两腿共 172 条**（不是 86 条），
   且命中判定是**子串**（`if method in stdout`）⇒ 落空可能被误判成命中。
   任何改这清单的动作都要意识到：守卫成本 = 「条目数 × 2 × 一次 `pytest --co`」，
   而它的正确性 = 子串运气。WC.1 必须换成精确 node-id 集合匹配。
4. **S1 的常量有两种绑定风格**（§1.4）。惰性化方案必须先处理
   `test_r2_dialogue_continuation_extension_contract.py:72–73` 的 `monkeypatch.setattr` 与
   `run_taiji_r2_daily_dialogue_continuation_extension.py:1203–1204` 的 `global` 重绑定，
   否则会在「看起来改完了」之后才炸。
5. **行号会漂移**（R8）。本文件所有行号基于 HEAD `60f32a95f`；每刀开工前重新核对。
6. **未跟踪项**：`delivery-runs/`（本运行目录，在仓内）、`.zcode/`、`output/playwright/client-ui-20261024/`
   一律不碰、不 commit。提交前用 `git status --porcelain` 自检。
7. **验证方式红线**：不得用「推上去看 CI 变不变绿」验证任何一刀。每刀必须有本机可复现的证据
   （R1 + S4 会让 CI 试错退化成赌博）。
8. **计划会回流修订**：若 Phase 7 门禁给出返工判定，或 W0 定因改变了 S1 的路径选择，
   本文件按原因修订并同步 board.md 的 `## 回流记录`。
