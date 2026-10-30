# Phase 1 立项分派 · 根治 Seed CI 长红

- 立项时间: 2026-10-25 23:55 (GMT+8)
- 主理人: delivery-team-lead（岳定盘）
- 运行目录: `E:/Seed/delivery-runs/20261025-235503/`
- 仓库: `liulang5945-netizen/Seed`（origin，gh 已认证，可直接读 run 与 job 日志）

## 1 用户诉求

> 「根治我的 CI 问题」

关键词是**根治**，不是「把这次跑绿」。据此本立项的完成标准定为「CI 结构性病灶被消除或降级为受控状态」，
而非「新增 N 条 deselect 把当前红灯按下去」。

## 2 取证（全部为只读，2026-10-25 23:50–23:56 实测）

### 2.1 CI 运行面（`gh run list --workflow=ci.yml --limit 12`）

- 近 12 次 run：**3 次 failure、8 次 cancelled、1 次 in_progress**。
- cancelled 全部由 `concurrency.cancel-in-progress: true` 造成：owner 一天推十几个 commit，
  而单腿实测 1h17m–1h30m（timeout 已放到 180m）⇒ 新 push 永远取消旧 run，
  **连续多日没有一次 run 跑完并给出完整结论**。
- 在跑的 run `40207103896`（HEAD a8024efc7）已 25 分钟，两腿仍未结束。

### 2.2 最近一次完整失败 run `40203419431`（commit `29980ee3b`）

> **勘误 2026-10-26（api-tester 用 `gh run view --json headSha` 复核，见 `artifacts/08-test.md` D-04）**
> 本小节标题原记为 `fbc212397`，**原值错误**，实际 head sha 是 `29980ee3b8d65890edba89e8d44be32538fac19a`。
> 这处错误不是笔误级别：`fbc212397` 已经包含 v1.3 boot audit 的修复，在它上面**永远复现不出本红点**，
> 会导致后续复现工作全部落空。另：下文"去重 13 条"原值亦误，实为 **12 条**（CI 日志 24 行 ＝ 12 条 × 2 腿）。
> 除这两处外，本节其余读数（job 结论、step 名、错误文案、跳过情况）经复核无误。

job 结论：startup-smoke ×2 ✅、docker-build ✅、**test (3.12) ❌**、**test-windows ❌**。

**红点 A（Ubuntu 腿，step「Verify CI-only deselect list is not silently empty」）**

```
deselect 清单 186 条，命中 162 条
##[error]清单条目落空，deselect 会静默失效：
  tests/taiji_native/test_r2_continuation_evaluation_adapter_contract.py::test_preflight_binds_frozen_and_extension_sources_by_default
  …（去重 12 条，覆盖 5 个 r2_* 文件；原文记 13 条，见本节勘误）
  (该文件里没有同名用例)
```

**连带后果（比红本身更严重）**：该 step 位于三个 pytest step 之前，它一红，
`Run native architecture contracts` / `Run Seed model-boundary contracts` /
`Run complete regression suite` **全部被跳过**（gh 显示为 `-`，未执行）。
⇒ **这次 CI 根本没有跑到任何测试**，红灯只反映清单腐化，套件真实健康度未知。

**红点 B（Windows 腿，step「Run full regression tests on Windows」，3m07s 挂）**

```
ERROR tests/taiji_native/test_r2_fast_ledger_resume_v4_contract.py
  - scripts.training.taiji_r2_dialogue_continuation_extension.ExtensionLedgerRefuse:
    v1.3 boot audit or source binding differs
…（同类共 19 个模块）
!!!!!!!! Interrupted: 19 errors during collection !!!!!!!!!
2 skipped, 84 deselected, 1 warning, 19 errors in 48.92s
```

### 2.3 红点 B 的根因链（代码级，已定位到行）

1. `scripts/training/taiji_r2_dialogue_continuation_extension.py:1548`（**模块级语句**）：

   ```python
   PER_ARM_CPU_LIMIT, AGGREGATE_CPU_LIMIT = effective_budget_limits()
   ```

2. `effective_budget_limits()`（同文件 684 行起）在 v1.3 分支上做三重治理校验，
   任一不通过即 `raise ExtensionLedgerRefuse("v1.3 boot audit or source binding differs")`
   （771 行）。三个子条件：
   - `boot_audit_identified`：`revision["boot_audit_path"]` 与实际 audit 路径，各经 `seal_ast.repo_relative` 映射后相等；
   - `rendering_tolerant_match(revision["boot_audit_sha256"], audit.read_bytes())`；
   - `boot_sources_match`：两个 boot-accounting 源文件各经 `seal_ast.verify_path_pin` 验钉。

3. 因为这是**导入期**执行，`pytest` 在 collection 阶段无法用 skip 表达「环境不满足」，
   只能硬失败 ⇒ 19 个测试模块连带 collection error ⇒ 整个 Windows 腿 Interrupted。

4. **本机不复现**：`import scripts.training.taiji_r2_dialogue_continuation_extension`
   在本机成功返回 `(1500000.0, 4300000.0)`。audit json 与两个 boot-accounting 源文件
   经 `git ls-files --error-unmatch` 确认**均已入库**，故不是「缺文件」这一类。
   ⇒ 失败落在**跨机路径映射 / 字节读数**上，必须本机同构复现才能定因，禁止靠读代码推断了结。

### 2.4 红点 A 与红点 B 同源

落空的 **12** 条 deselect 指向的 5 个文件，正是被红点 B 的 collection error 打掉的模块。
**A 是 B 的下游症状**：模块收集失败 ⇒ 用例不存在 ⇒ deselect 落空 ⇒ 守卫红 ⇒ 后续 pytest 全跳过。

### 2.5 结构性病灶（非单点 bug，属「根治」范围）

| 编号 | 病灶 | 证据 |
|---|---|---|
| S1 | **导入期做环境相关强校验**：模块级常量绑定治理 audit | extension.py:1548；19 模块 collection error |
| S2 | **CI-only deselect 机制会腐化且会自伤**：单腿口径的清单条数从 6 条长到 93 条（现 HEAD 86 条），条目落空即红；守卫一红，后面三个 pytest step 全跳过，CI 事实上不测。⚠️ 口径更正（api-tester §5.3）：守卫脚本扫整份 ci.yml 所有 `--deselect` 行 ⇒ 两腿重复计数，故 CI 日志打「186 条」＝93×2、HEAD 会打「172 条」＝86×2，**该数字不可读作清单条数**；本文初稿「从 6 条长到 186 条」混淆了两种口径，已更正 | run 40203419431；核验见 `08-test.md` §5 |
| S3 | **Ubuntu 腿把 tests/ 跑三遍**：`tests/taiji_native` + `tests/seed` + `tests/`（后两趟含前一趟子集），这是超时从 30m→60m→90m→120m→180m 的主要推力 | ci.yml 481/475/467 三步 |
| S4 | **cancel-in-progress 与单腿 1.5h 叠加** ⇒ 长期拿不到完整结论 | §2.1 |
| S5 | 运行环境陈旧告警：`actions/checkout@v4` 等仍走 Node 20（被迫跑在 Node 24 上）；ubuntu-latest 将于 2026-10-19 起迁移 Ubuntu 26 | 每次 run 的 annotations |

## 3 范围界定

**在范围内**：`.github/workflows/ci.yml`、`scripts/training/verify_ci_deselect_list.py`、
触发 collection error 的治理校验代码（extension.py 的模块级绑定与其依赖的 seal_ast 映射/容忍面）、
被 S1/S2 牵连的测试文件、CI 结构（步骤去重、matrix、concurrency）。

**不在范围内**（本立项不做，另案）：
- M7 发布面（发布 workflow／安装启动／回滚／安全／安装包验收）——沿用 m7-ci.yml 的既有裁定；
- `taiji-harness/` 下的 19 个 workflow（外部 fork，未被本仓 CI 触发）；
- 任何需要重新训练、重新采集实验读数的改动；
- 覆盖率阈值、mypy 基线的调整（它们当前是绿的，不动）。

**红线**：不改变任何冻结判据/封印语义。`taiji/organs.py`、`fabric.py`、`model.py`
需 owner 显式放行才可修改；本次预计不触碰，若诊断后认为必须触碰，须回到 owner 处请示。

## 4 完成标准（DoD）

- D1：Ubuntu 腿与 Windows 腿**在同一个 commit 上**双绿，且三个 pytest step **实际执行**（不是被跳过）。
- D2：`deselect` 清单条目数**净减少**，且剩余每一条都有「本机已验证的重启条件」；
  S2 的自伤结构被消除（守卫不再能一键跳过全部测试）。
- D3：S1 消除——治理校验不再在导入期决定模块可导入性（改为可延迟/可跳过/带明确失败语义）。
- D4：CI 单腿墙钟时间**下降**（S3 去重 + S4 治理），使一次 push 能在合理时间内拿到完整结论。
- D5：所有改动落入一个可回滚的提交序列，每刀一回滚点；并行会话的改动不混入。

## 5 分派表

| 阶段 | 岗位 | 产物 |
|---|---|---|
| Phase 1 | delivery-team-lead（本人） | `01-intake.md`（本文件） |
| Phase 2 | implementation-planner | `02-plan.md` |
| Phase 3 | （裁剪：需求已由 DoD 与 §3 冻结，不单独派 PM，避免为流程而流程） | — |
| Phase 4 | architecture-reviewer | `04-architecture.md`（S1–S4 的结构裁决与 ADR） |
| Phase 5 | （裁剪：无 UI 交付物） | — |
| Phase 6 | backend-architect（CI/治理脚本侧，单人即可，改动面同构） | `06-implementation.md` |
| Phase 7 | code-reviewer | `07-review.md` |
| Phase 8 | api-tester（含本机同构复现与定因） | `08-test.md` |
| Phase 9–11 | （裁剪：无性能/安全/合规交付面；S5 的 Node 版本告警按技术债登记，不改行为） | — |
| Phase 12 | （裁剪：无部署物，CI 本身即交付物） | — |
| Phase 13 | tech-writer | `13-documentation.md`（CI 规范条目与台账） |
| Phase 14 | delivery-team-lead + product-manager | `14-acceptance.md` |

**裁剪理由（记入决策记录）**：本任务是「既有 CI 的缺陷根治」，无可交付的 UI、无部署目标、
无数据处理面，跑满 14 阶段会对 S3/S4 想要解决的「时间成本」本身形成反例。
保留的 8 个阶段覆盖了诊断→裁决→实现→审查→验证→落账的完整闭环。

## 6 关键风险

- R1：**本机不复现**。S1 的失败只在 runner 上出现（本机 import 通过）。
  若无法建立同构复现环境，任何修复都只能靠 CI 试错，而 S4 让每轮试错要 1.5h 且常被取消。
  ⇒ 缓解：Phase 8 首要交付物是「本机同构复现方法」。
- R2：**治理校验的语义归属**。v1.3 boot audit 是 owner 近期在飞的治理面
  （近 3 个 commit 都在改它），改动可能与并行的治理会话打架。
  ⇒ 缓解：诊断先出结论，改动前先与 owner 对齐口径，不动判据本身。
- R3：**deselect 清单的既得利益**。86 条清单是过去多轮「压红」的成果，
  一次性收紧可能暴露一批真红。⇒ 缓解：按 DoD D2 分批改，每批附双绿证据。

## 7 给下游

1. 全部事实取自 `gh` 真实读数与本仓代码行号，**不要凭记忆或推断覆盖**；任何结论要能指出证据位置。
2. 本机复现是硬要求：不允许「改完推上去看 CI 变不变绿」这种验证方式（R1 + S4 会让它退化成赌博）。
3. 所有路径用绝对路径；改 `.github/workflows/` 前先读全文件（当前 ci.yml 815 行，注释即判据，删注释前先确认语义）。
4. 提交纪律：每次提交强制 pathspec，一刀一回滚点，不混入并行会话改动
   （当前工作区存在未跟踪项 `.zcode/`、`output/playwright/client-ui-20261024/`、
   `plans/reference/PLAN-N1-175_*.md`，**一律不碰**）。
