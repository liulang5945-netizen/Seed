# Phase 4 架构裁决 · 根治 Seed CI 长红

- 裁决人: architecture-reviewer（谭界明）
- 裁决时间: 2026-10-26（GMT+8）
- 运行目录: `E:/Seed/delivery-runs/20261025-235503/`
- 上游: `board.md`（决策 1–6）、`artifacts/01-intake.md`（§3 红线 / §4 DoD）、`artifacts/02-plan.md`（全文）
- 基线: `main` @ `979217e93`（写作时 HEAD）。已核验 `60f32a95f → 979217e93` 之间
  `.github/workflows/ci.yml`、`scripts/training/verify_ci_deselect_list.py`、
  `scripts/training/taiji_r2_dialogue_continuation_extension.py`、`tests/conftest.py`、`pyproject.toml`
  **零改动**（`git diff --stat` 空输出）⇒ 本文引用的行号沿用 `60f32a95f` 仍然成立（R8 已闭合）。
- 本阶段性质: **只读裁决**。未修改仓库任何被追踪文件；唯一写入为本文件与 `board.md` 的 Phase 4 小节。

---

## 0 结论（一页）

| 编号 | 病灶 | 终裁 | 回滚点 | 判据语义是否变化 |
|---|---|---|---|---|
| **ADR-S2** | CI-only deselect 自伤 | **S2-A（清单单源外置，JSON）+ S2-D（守卫单次收集）**，并**同时**做守卫元数据校验（重启条件非空＝硬门） | R-S2a / R-S2b | 否（清单内容与 172 条一字不变；守卫判定口径按 §3.3 收敛，红仍然红） |
| **ADR-S3** | Ubuntu 腿 tests/ 跑三遍 | **S3-A**，且派生主键为 **junit `classname`**（不是 `file` 属性）＋**文件路径归属**（不是字符串前缀）＋ 三面自校验 | R-S3 | 否（覆盖率分母、artifact 名、B-4 阈值全不变） |
| **ADR-S4** | 并发自取消 | **S4-b（按事件分化）＋ S4-c（快通道）组合**；`cancel-in-progress` **在 PR 上保留 true、在 push main 上改为 false** | R-S4 / R-S4c | 否（只改运行节奏，不改任何判据） |
| **ADR-S6** | 一步红 ⇒ 全链失明 | **S6-i（continue-on-error + 汇总判红）**；**裁定不做 S6-ii（拆 job）** | R-S6a | 否（同一事实从「跳过」改为「汇总判红」，红不会变绿） |
| **ADR-S1** | 导入期治理校验 | 一式两份：先把 P1–P4 评估完 + 给出**决策表**（§6.5.5）；W0.2 已回传并按表收口（§6.5.7）＝P4（环境面，**已由 `565604a9e` 落地**）+ P2（结构兜底网）+ P3（**已锁死：Q-S1 已决（board 决策 16，2026-10-26）：owner 答「否」，P1/P3 判触红线不放行，WB.2 取消，D3 记「部分满足」，防复发改走「记录方式规范」路线（决策 16/17）**） | R-S1a/b/c/d | P1/P3 **已判触红线，不放行**（是非题 §6.5.6 已答，board 决策 16） |

**三条跨 ADR 的硬约束（任何一刀违反即返工）：**

1. **派生自己的东西必须自证**：所有新脚本（deselect 渲染 / junit 分面）一律 **fail-closed**，凡「读不到/解析不了/归属不了」一律 exit≠0，禁止退化成 warn 或空输出。
2. **红不能变绿**：任何新增 `continue-on-error` 都必须**同一个 commit** 把「汇总判红 step」落地（决策 3①②）。汇总 step 只认 `steps.<id>.outcome`（**不认 `conclusion`**），见 §3.6。
3. **不删 `ci.yml` 注释**：本阶段删除的只有「两腿 env 里的 172 行条目」，不涉及 `ci.yml:34–105` / `662–665` 的历史注释；需要改写的句子以「追加 + 指向」方式处理（§4.1.3）。

**必须先记下的两个纠正（与 `02-plan.md` 的数字不同，本文以真实读数为准）：**

- **A1 假设不成立**：`<testcase>` **没有 `file` 属性**（本机 pytest 9.1.1 实测 + team-lead 实测一致）。但兜底判断也错——`classname` **带目录**（`tests.<子目录>.<模块名>`），且与 `__init__.py` 无关 ⇒ 派生主路径不需要 `--co -q` 兜底，**R7 降级为「已缓解」**。
- **`02-plan.md` §5 M2 的基线包络写窄了**：Ubuntu 单腿实测 **1h17m46s–2h42m57s**（不是 1h17m–1h30m），且**两腿并行取整腿 max**。详见 §5.2。

---

## 1 架构说明与模块边界

### 1.1 切分

把 CI 拆成四个面，依赖单向，**所有跨面调用只走契约**：

```
【数据面】  .github/ci/ci-only-deselect.json（单一清单源，双腿同源）
              │  被读（只读）
              ▼
【渲染面】  scripts/training/render_ci_deselect_args.py        → stdout: 单行 deselect 参数串
            scripts/training/verify_ci_deselect_list.py       → 守卫（新版，见 C-3）
            scripts/training/split_ci_junit_faces.py          → junit 三面派生（新增）
              │  只被调用，不调用 workflow
              ▼
【执行面】  .github/workflows/ci.yml（Ubuntu test / test-windows 两腿）
            .github/workflows/ci-fast-lint.yml（快通道，新增）
              │
              ▼
【门禁面】  每腿一个 gate-summary step（汇总判红，新增）
```

**边界规则**

- 数据面 ∈ 脚本：不得被任何生产模块 import（它是 CI 配置，不是领域模型）。
- 渲染面 ∈ 脚本：stdlib only，**禁止引入第三方依赖**（PyYAML/requests 一律不许）；不加 `-p` 依赖 pytest 插件；输入输出只有 argv/stdin-env/stdout/exit code。
- 执行面 ∈ workflow：不允许在 `run:` 里写超过 5 行的 shell 逻辑（超过即下沉成脚本并可本机跑）；`continue-on-error` 只允许出现在白名单 step 上（§3.5）。
- 门禁面：**每腿唯一**汇总 step，且它没有 `continue-on-error`。

### 1.2 依赖方向（禁止反向）

```
ci.yml / ci-fast-lint.yml  ──►  渲染面脚本  ──►  数据面 JSON / report.xml
       （不得反向：脚本不许读 workflow 文本，
         唯一例外：守卫脚本不再扫 ci.yml —— S2-A 之后它只读 JSON）
```

> ⚠️ `verify_ci_deselect_list.py:29–44` 现状是**扫 ci.yml 全文**（`listed_entries()`），这是「
> 清单必须活在 workflow 里」这一历史约束的产物。S2-A 之后这条反向依赖必须删除。

### 1.3 契约面清单（下游只能照这里开工）

| 契约 | 编号 | 内容 | 消费方 |
|---|---|---|---|
| deselect 数据源 schema | **C-1** | §3.1 | render / guard / 双腿 |
| `render_ci_deselect_args.py` | **C-2** | §3.2 | 两腿各一个 step |
| `verify_ci_deselect_list.py`（新版） | **C-3** | §3.3 | Ubuntu 腿守卫 step |
| `split_ci_junit_faces.py` | **C-4** | §3.4 | Ubuntu 腿（Windows 不用） |
| Ubuntu 腿 step 契约 | **C-5** | §3.5 | 六个 step 的 id / continue-on-error / 产物 |
| 汇总判红 step 契约 | **C-6** | §3.6 | 两腿（Ubuntu 必做，Windows 同 commit 做） |
| 并发约定 | **C-7** | §3.7 | ci.yml（改）/ m7-ci.yml（只登记）/ 新快通道 |

### 1.4 落地后 Ubuntu 腿的 step 序列（to-be）

```
…既有的 install / ruff / black / frozen-marker / mypy ×2 / pip-audit / 8 个 verify / upload verify reports（不动）
① Verify CI-only deselect list           id=deselect-guard   continue-on-error: true   （读 C-1）
② Collect all tests under CRLF checkout  id=collection-crlf  continue-on-error: true   （决策 8；只 --co -q 不跑用例，见 ★）
③ Run complete regression suite          id=pytest-full      continue-on-error: true   （唯一一次 pytest tests/，含 --cov；产出 report.xml/coverage.*）
④ Derive native/seed junit faces         id=junit-faces      continue-on-error: true   （report.xml → native-report.xml / seed-report.xml，artifact 名不变）
⑤ Core reasoning path coverage gate (B-4) id=coverage-gate   continue-on-error: true   （仍读 coverage.json，阈值 60.0 不变）
⑥ Coverage trend summary                                     if: always()（不动）
⑦ Summarize CI gate outcomes             id=gate-summary     if: always()，无 continue-on-error
⑧ Upload core coverage report / Upload test results           if: always()（不动，artifact 名不变）
```

Windows 腿：只有一条 pytest step（`ci.yml:798–805`），形态已是「单跑全量」；本轮对它的改动只有
「给它 id + continue-on-error + 参与汇总」（WE.1 的一部分），**不引入分面**。

---

## 2 本阶段新取证（只读实测，用于支撑裁决）

### 2.1 五次成功运行的真实读数（`gh api …/jobs`，补 `02-plan.md` §1.2 的包络）

| run | Ubuntu 腿总时长 | step25 native | step26 seed | step27 full(`--cov`) | Windows 腿总时长 | Windows pytest |
|---|---|---|---|---|---|---|
| 40024125106 | 1h17m46s | 24m46s | 0m29s | 45m25s | 1h24m33s | 1h21m57s |
| 40096384119 | 2h35m18s | 49m34s | 1m08s | 1h30m03s | 1h52m18s | 1h50m02s |
| 40123047508 | 2h30m35s | 47m37s | 1m04s | 1h28m25s | 1h50m19s | 1h47m38s |
| 40157476661 | 1h55m34s | 36m41s | 0m46s | 1h07m17s | 1h33m15s | 1h31m02s |
| 40171918472 | **2h42m57s** | **52m33s** | 1m05s | **1h36m19s** | 1h02m07s | 1h00m22s |

⇒ **`02-plan.md` §5 M2 写的基线「1h17m–1h30m」只覆盖了最快那一次**；真实包络是 **1h17m46s–2h42m57s**（快机/慢机 ~2×，与 `ci.yml:32` 注释自述一致）。

⇒ **去掉 step25/26 的相对收益在五次样本上高度一致：32.3% / 32.4% / 32.5% / 32.7% / 32.9%**（因为 step27 是 step25 的超集）。Ubuntu 单腿 → 52m31s … 1h49m19s。**M2 的「≥30%」以 2.4pp 的余量通过，属于「险过」，必须盯**（§8 R9）。

⇒ **更重要的发现：S3 之后 Windows 腿变成新的关键路径**（1h02m–1h52m）。取整腿 max 后，
「一次 run 的墙钟」下降只有 **0%–33%**（平均 ~21%）：Windows 已经是单跑全量，S3 对它没有收益。
**DoD D4 要继续往下压，必须动 Windows 腿**——本次不改，登记为技术债 D-1（§8）。

### 2.2 junit 面（本机 pytest **9.1.1** 实测 + CI 实际安装版本核对）

- CI 上 Ubuntu 腿 `test (3.12)` 实际安装 **`pytest-9.1.1` / `pytest-cov-7.1.0` / `coverage-7.17.0`**
  （run 40203419431，`Install test dependencies` 步骤日志，2026-10-25T14:42:35Z）⇒ 与本机实测版本一致。
- `<testcase>` 属性实测：`{'classname','name','time'}`，**没有 `file`** ⇒ **A1 不成立**。
- `classname` = **rootdir 相对的点分模块路径**，无 `__init__.py` 时照样带目录：
  - 函数用例：`tests.seed.test_b` / `name=test_two`
  - 类级用例：`tests.seed.test_c.TestFoo` / `name=test_bar`
  - 参数化：`tests.seed.test_c` / `name=test_param[1]`
  - 收集失败的记录：`classname=''` / `name=tests.seed.test_c`（ module 点分路径写在 name 里）
- 与 `--co -q` 的 node-id 对照可**无损互转**（已实测：5 条用例一一对应）。
- ⚠️ **尾点陷阱已核实**：`tests/seed` 33 文件与 `tests/seed_platform` 8 文件是**兄弟目录**，
  `tests.seed` 做字符串前缀匹配会吞掉 `tests.seed_platform.*`。本文用「**解析成磁盘文件路径后再判定归属**」消除该陷阱（§3.4）。

### 2.3 deselect 清单与消费面（本次复核）

- 条目：**Ubuntu 86 / Windows 86（`ci.yml:108–193`、`ci.yml:668–753`），sort 后 diff 空**；
  守卫 `listed_entries()` 读到 **172** 条（两腿各一份）。
- 涉及文件 **56 个**：35 个文件 1 条、16 个 2 条、4 个 3 条、1 个 7 条（`test_n3_09_tau_rule_contract.py`）。
- 常量消费面（AST 精确核对，不是 grep 粗读）：
  - `from …extension import PER_ARM_CPU_LIMIT, AGGREGATE_CPU_LIMIT`：**1 处**
    （`scripts/training/run_taiji_r2_daily_dialogue_continuation_extension.py:44`）
  - `from …amendment_v1_4 / v1_5 import …`：**2 脚本 + 2 测试**
    （`run_taiji_r2_fast_ledger_resume_v3.py:26`、`…v4.py:30`、
     `test_r2_continuation_budget_amendment_v1_4_contract.py:5`、`…v1_5_contract.py:5`）
  - `v1_4/v1_5` 模块本身用 `from scripts.training import …extension as budget`（**别名 import，不触发 `__getattr__`**）
    且自带字面量常量 ⇒ **对 P1（惰性化）免疫**。
  - runner 内裸常量引用：**PER_ARM 13 处 / AGGREGATE 9 处**（含 1203–1204 的 `global` 重绑定）。

### 2.4 近 40 次 run 结论（`gh run list --limit 40` 复核，与 plan 一致）

`18 failure / 16 cancelled / 5 success / 1 in_progress`。近 12 次 `3 failure / 8 cancelled / 1 in_progress`
⇒ **「完整结论」本身是稀缺品**，这正是 ADR-S4/S6 要治的。

---

## 3 接口契约表（下游直接照此开工）

> 通用约定：所有脚本 run 时 cwd = 仓库根；stdout 为 UTF-8；退出码语义统一见 §3.8；
> 脚本自身**不得**写 `$GITHUB_STEP_SUMMARY` / `$GITHUB_ENV`（由 workflow step 决定重定向位置），
> 例外在契约里显式写明。

### 3.1 C-1 · deselect 数据源（`deselect.json`）

**路径（终裁）**：`.github/ci/ci-only-deselect.json`
理由：① 它是 CI 治理配置，不是测试内容，放 `tests/` 会被 pytest 与其它测试资产混读；
② `.github/` 下目前只有 `workflows/` 与 `dependabot.yml`，加一个 `ci/` 不引入新顶层目录约定，且与 workflow 同生命周期；
③ 两腿 + 守卫都指向同一路径 ⇒ **结构上不可能再出现两腿不一致**。

**格式（终裁：JSON，不是 YAML/纯文本）**
理由：① stdlib `json` 可用，**不新增任何依赖**（对比 YAML 要 PyYAML，需要在两腿 + 本机 + 本机同构环境都装，违背最小依赖原则）；
② 可被 `jq`/审阅 diff 消费；③ 评审时一行一条，diff 干净。
（列表 prose history 的去处见 §4.1.3，不塞进 JSON。）

```jsonc
{
  "schema": "seed-ci-only-deselect-v1",
  "updated_utc": "2026-10-26",
  "source_note": "清单 86 条，由 ci.yml 双腿 env 原样迁入；历史分族与批次沿革见 13-documentation.md deselect 台账",
  "entries": [
    {
      "node": "tests/taiji_native/test_a30_probe_help_runs.py::test_probe_help_exits_zero_and_lists_every_flag",
      "family": "a30",
      "batch": "batch-1",
      "registered_utc": "2026-10-07",
      "restart_condition": "runner --help 的编码路径修复（本机逐条皆绿、runner 上永久红）",
      "evidence": "run 37505960489"
    }
  ]
}
```

| 字段 | 类型 | 必填 | 约束 |
|---|---|---|---|
| `schema` | string | ✓ | 必须等于 `seed-ci-only-deselect-v1`，否则脚本一律 exit 2（版本闸门） |
| `updated_utc` | string | ✓ | YYYY-MM-DD |
| `entries` | array | ✓ | 非空；条目数 ≤ 200 |
| `entries[].node` | string | ✓ | 必须含且仅含一个 `::`；必须以 `tests/` 开头；跨条目唯一 |
| `entries[].family` | string | ✓ | 分族键（迁移期沿用注释里的家族名） |
| `entries[].batch` | string | ✓ | 批次键 |
| `entries[].registered_utc` | string | ✓ | 入册日期 |
| `entries[].restart_condition` | string | ✓ | **非空，且长度 ≥ 12 个字符 ⇒ DoD D2「每条带重启条件」由结构保证** |
| `entries[].evidence` | string | ✓ | run id 或证据文档相对路径 |

**校验不变式（任何读取方都必须执行，违反即 exit≠0）**：
`nodes_unique` ∧ `all_have_restart_condition` ∧ `all_nodes_wellformed` ∧ `entries_sorted_by_node`（顺序稳定，diff 干净）。

### 3.2 C-2 · `scripts/training/render_ci_deselect_args.py`（新增）

| 项 | 契约 |
|---|---|
| argv | `--source <path>`（默认 `.github/ci/ci-only-deselect.json`）、`--mode env`（写 `$GITHUB_ENV` 格式行）／`--mode args`（只打参数串，默认） |
| stdin | 无 |
| stdout | `--mode args`：`--deselect <node> --deselect <node> …`（**单行**，空格分隔，末尾无空格无换行符以外的字符）<br>`--mode env`：单行 `DSH_CI_DESELECT=<同上>` |
| stderr | 校验失败信息 + `::error::` 前缀行 |
| 退出码 | 0 成功；2 数据源缺失/schema 不符/校验不变式失败；3 条目格式非法 |
| 幂等 | 同一输入 ⇒ 同一输出字节序列（顺序＝文件中顺序，不排序） |
| 用法（workflow） | `python scripts/training/render_ci_deselect_args.py --mode env >> "$GITHUB_ENV"` |

**上线前必须验证的假设 A2**（`02-plan.md` §8）：`$GITHUB_ENV` 单行注入与现状 YAML `>-` 折叠等价。
验证方式：迁移首个 commit 落库后，在同一 run 里打印 `python -c "import os;print(os.environ['DSH_CI_DESELECT'].count('--deselect'))"`，
必须等于 86；**不等即视为该刀未完成**（此断言作为 step 的输出的一部分保留 1 个 run 后删掉断言本身，
不要长期留在里面）。若 `$GITHUB_ENV` 有长度/语法限制导致失败 ⇒ 退回到 CI：把同一串写入
`${{ runner.temp }}/ci_deselect_args.txt`，各 pytest step 用 `"$(cat …)"` 消费。**不得回到「双腿各写一份」**。

### 3.3 C-3 · `scripts/training/verify_ci_deselect_list.py`（改写）

| 项 | 契约 |
|---|---|
| 输入源 | **只读 C-1 的 JSON**（不再扫 ci.yml；删除 `listed_entries()` 对 workflow 的反向依赖） |
| argv | `--source <path>`（默认同上）；`--strict-metadata`（默认开） |
| 收集方式（终裁，取代「每条一个子进程」） | **一次**调用：`python -m pytest --co -q -p no:cacheprovider --continue-on-collection-errors <listed files…>`，`<listed files…>` = 去重后的 56 个文件路径（排序、固定顺序）。<br>理由：现状 172 次子进程实测 5m19s–6m24s；单进程收集 56 个文件的成本≈ 现状单文件成本之和 / 若干，预计 ≤2m；且不因「清单外的模块红」而中断。**不要用全量 `--co` 收集 809 个文件**（更慢、且把无关收集错误掺进判定面）。 |<br>⚠️ **必须修掉的反向同源缺陷**（team-lead 2026-10-26 提供）：现状判定是**子串包含**
（`verify_ci_deselect_list.py:78` 的 `if method in collected.stdout`）。只要某条用例名是另一条 node id 的
子串，落空就会被静默判成命中——**与这把守卫自己要防的「静默失效」反向同源**。
⇒ 新版必须把 `--co -q` 的输出解析成 **node-id 集合**，用 `entry_node in collected_set` **精确成员判定**
（含 `::Class::` 与参数化 `[...]` 后缀的整体匹配），禁止任何形式的 `str in str`。
**回归用例（必须写）**：构造两条互为子串的用例名，删掉其中一条 ⇒ 新版守卫**必须判红**（旧版会判绿）。
另：本脚本读到的条目数是 172（＝ 86×两腿），**不得把这个数字读成「清单条数」**（CI 历史上打过 186 ＝ 93×2）。 |
| 判定 | 记 `collected` = 收集到的 node-id 集合（含参数化后缀、`::Class::` 前缀）。<br>**MISS**（红）：条目 node ∉ `collected` **且**其文件被成功收集。<br>**UNCOLLECTABLE**（黄）：条目所在文件在本次收集里产出 collection error ⇒ 打 `::warning::` + 进 summary 列表，**不判红**（理由：该事实由 pytest step 自己的非零退出/ `<error>` 记录承担，红不被放宽，只是不再重复计一次）。<br>**NOFILE**（红）：条目里的文件路径在磁盘上不存在（typo / 文件被删 / 路径写错 —— 正是 ㊵-467 的原病）。 |
| stdout | `deselect 清单 <N> 条（来自 <source>）；MISS <a>；NOFILE <b>；UNCOLLECTABLE <c>`；每条一行明细 |
| `$GITHUB_STEP_SUMMARY` | 允许写 markdown 表（本脚本是唯一被允许写 summary 的新建脚本） |
| 退出码 | 0 全部命中；1 存在 MISS/NOFILE；2 数据源不可读/schema 不符/元数据校验失败；3 `pytest --co` 子进程本身无法启动（不含「子进程返回非零」——`--continue-on-collection-errors` 下返回非零是正常的） |
| 性能验收 | 该 step **目标 ≤2m，硬上限 ≤3m**（超过 ≤3m 即视为未达标，退回 `IMP` 重做，不得用「再拆子进程」糊过去） |
| fail-closed | 数据源被删/为空 ⇒ exit 2（**不允许**退化为「没清单＝全通过」） |

**语义校正记录（必须写进 commit message）**：旧实现对「收集失败」与「条目落空」报同一种红（`(该文件里没有同名用例)`）。
新实现把两者分开：**同一事实不会从红变黄**——收集失败一定使 pytest step 非零退出并被汇总 step 判红（§3.6）；
守卫只是不再把它重复算作「清单腐化」。

### 3.4 C-4 · `scripts/training/split_ci_junit_faces.py`（新增）

| 项 | 契约 |
|---|---|
| argv | `--source report.xml --native native-report.xml --seed seed-report.xml --repo-root .`（`--repo-root` **必填且显式**，不得依赖调用时的 cwd：`classname` 是相对 rootdir 的点分路径，rootdir 一变前缀就变） |
| 输入 | 唯一的全量 junit（xunit2，`<testsuites><testsuite name="pytest">…`） |
| 输出 | `native-report.xml`、`seed-report.xml`（**文件名与本目录不得改**，`ci.yml:535–544` 只读这两个名字） |
| 归属算法（**终裁逐条可落地**） | 1. `key = tc.classname or tc.name`（classname 为 `''` 时是收集失败记录，其 name 是点分模块路径）。<br>2. `parts = key.split('.')`；循环：`cand = "/".join(parts) + ".py"`，若 `(repo_root/cand).is_file()` 则定位成功；否则 `parts.pop()` 继续；耗尽即 **`UNRESOLVED` ⇒ 整个脚本 exit 3（fail-closed）**。<br>3. 用 **`Path.is_relative_to()` / parents 判定**（不是字符串前缀）：<br>`cand` 在 `tests/taiji_native/` 下 ⇒ native 面；在 `tests/seed/` 下 ⇒ native? 不 ⇒ **seed 面**；在 `tests/seed_platform/` `tests/resonance/` `tests/` 根 ⇒ **两面都不进**（与现状语义一致：现状 step25 只跑 `tests/taiji_native`、step26 只跑 `tests/seed`）。<br>4. 派生文件时：深拷贝 `<testsuites>`/`<testsuite>` 结构与所有子节点（`<system-out>`/`<system-err>`/`<failure>`/`<error>`/`<skipped>`/`<properties>`），只筛 `<testcase>`；**重算** testsuite 的 `tests/failures/errors/skipped/time`（`time` 取子节点之和 + 原 relative time，`atributes` 缺字段时补 0，不得留旧值）。 |
| 输出 head comment | 在两个派生文件里写 `<!-- derived from report.xml by split_ci_junit_faces.py; not a separate pytest run -->`，任何人打开都能看出它不是独立执行面 |
| exit code | 0；2 源文件缺失/不可解析；3 有 testcase 无法归属到磁盘文件；4 自校验失败（见下） |
| **自校验不变式（必做，两级、顺序不可颠倒；第 1 级按 2026-10-26 主理人 C-4 裁决后的形态学判定）** | **第 1 级（前置门禁，先跑）——判据是子元素，不是 classname 本身**（`08-test.md` §12.0 两行表，主理人裁决定版）：<br>· `classname=""` 且子节点为 `<error message="collection failure">` / `<failure>` / **无子节点或未知形态** ⇒ **exit 5**，并打印模块清单（点分路径来自 `name` 属性；未知形态一律 fail-closed）。<br>· `classname=""` 且子节点为 `<skipped>` ⇒ **合法绿跳，不判红，且排除出面、单独计数（`module_skipped`）**：这是模块级 `pytest.importorskip` / `pytest.skip(..., allow_module_level=True)` 的 junit 形态（实测位点：`tests/test_taiji_browser_environment.py:14`、`tests/test_taiji_browser_tool_loop.py:18`、`tests/taiji_native/test_r2_fast_ledger_resume_v4_contract.py:10`、`tests/taiji_native/test_r2_worker_namespace_integration.py:16`，基线 run 40171918472 共 4 条）。<br>**沿革记录（判据为何从「归属进面」改定「排除出面」）**：backend-architect 在 R-S3 初版（commit `899198273`）发现基线成功 run 的 report.xml 本就有 4 条 `classname=""`+`<skipped>` 合法记录，字面执行我原稿的「凡 `classname=""` 即红」会让每次 CI **永久假红**——违反「探针必须红在该红的地方」。AR 曾追认其过渡修正「按模块路径归属进面」；**随后主理人基于 api-tester 基线数据（`08-test.md` §12.0）终裁为「排除出面 + 单独计数 module_skipped」**（归属进面会让 R5 的集合比对被迫携带 skip 记录、且守恒分母被 pytest 的 `tests=` 属性污染），backend-architect 已按裁决 amend 落地为 `917774e70`。三态的精神不变：collection error（「默默少跑一批模块」）必须显式红，合法 skip 不许制造假红。<br>**第 2 级（第 1 级通过后再做）——分母用「唯一 node-id 数（classname+name 去重）」，不用 pytest 的 `tests=` 属性**（后者把模块级 skip 行也计入：基线 native 面原生 junit `tests=3903` 但唯一 node-id 只有 3901，全量 4580 vs 4576，恒差＝module_skipped）。守恒判据：**uid(native) + uid(seed) + uid(other) == uid(全量)** 严格相等（基线锚 **3901 + 185 + 490 = 4576**，`module_skipped=4` 单独记账打印、不进守恒和）；且 uid(native) > 0 ∧ uid(seed) > 0（悄悄吞面 ⇒ 立刻红）⇒ 违反 **exit 4**。<br>R5 / WD.2 的对照基准随裁决同步：**派生面 vs 同 run 独立面双向 diff = 0**（不再拿旧基线 run 的原生面直接比对 skip 记录）。<br>外加：native 面的每个 testcase 解析出的文件路径必须 `is_relative_to("tests/taiji_native")`、seed 面必须 `is_relative_to("tests/seed")` —— 这一条就是「尾点陷阱」的疫苗（`tests/seed_platform` 的归属会被判 False，`tests.seed` 字符串前缀会误吞它）。<br>**AR 复核（2026-10-26）**：已独立核验 amend commit `917774e70` 在 main、`split_ci_junit_faces.py` docstring 与 `08-test.md` §12.0 三处口径一致（两行表 + uid 分母 + module_skipped 单独记账）⇒ **按裁决文本定版，无需再裁定**。 |
| 依赖 | stdlib only（`xml.etree.ElementTree`、`pathlib`） |
| 幂等 | 同一 `report.xml` ⇒ 字节级相同的两个输出 |

> **给 api-tester 的对照实验（WD.2）约定**：在有 betrouwbare 基线（未去重的最后一次成功 run）的前提下，
> 比对三张面的 **node-id 集合**（由 classname+name 反解，规则与 §2.2 一致），
> 判据是「集合相等」而不是「数量相近」；差异逐条解释，禁止「差不多」。

### 3.5 C-5 · Ubuntu 腿 step 契约（改写）

| # | step name（**不得改名**：它们出现在命名规则与历史证据里） | 必要 `id` | `continue-on-error` | `if:` | 产物 |
|---|---|---|---|---|---|
| 1 | `Verify CI-only deselect list is not silently empty`（`ci.yml:455`） | `deselect-guard` | **true** | 无 | summary |
| 2 | `Run complete regression suite`（`ci.yml:481` 改写成唯一 pytest 步） | `pytest-full` | **true** | 无 | `report.xml` `coverage.xml` `coverage.json` |
| 3 | `Derive native/seed junit faces`（新增） | `junit-faces` | **true** | 无 | `native-report.xml` `seed-report.xml` |
| 4 | `Core reasoning path coverage gate (B-4)`（`ci.yml:507`） | `coverage-gate` | **true** | 无 | `reports/core_reasoning_coverage_ci.json` |
| 5 | `Collect all tests under CRLF checkout`（**新增，决策 8**） | `collection-crlf` | **true**（见下方 ★） | 无 | collection 日志 |
| 6 | `Coverage trend summary`（`ci.yml:520`） | — | 保持现状（无） | `always()` | summary |
| 7 | `Summarize CI gate outcomes`（**新增**） | `gate-summary` | **禁止** | `always()` | summary + exit code |
| — | `Run native architecture contracts`（`ci.yml:467`）/ `Run Seed model-boundary contracts`（`ci.yml:475`） | — | — | — | **删除**（由 §3.4 派生两面取代） |
| — | `Upload core coverage report` / `Upload test results`（`ci.yml:510`/`535`） | — | 无 | `always()` | artifact（**名称不得改**） |

**★ 与决策 8 的口径调和（本 ADR 的终裁；【已追认 2026-10-26，board 决策 11】）**：决策 8 要求该 step「真判红」并写
`continue-on-error: false`。**「真判红」这一半我照单全收，但载体不能是 `continue-on-error: false`**——
那正是 S6 要根治的病：它一红，其后所有 step（含 pytest 与 artifact 上传）又被跳过，M5 的注入实验会再次失败。
裁法：把它**并入 S6-i 的 `continue-on-error` 集合**，由 `gate-summary` 按 §3.6 的同一张表判（`outcome≠success ⇒ RED`）。
这样「真判红」完全保留（job conclusion 仍 failure），但失去的是「让整条腿失明」的能力。
**若 TL 坚持 `continue-on-error: false`，则该 step 必须移到最后（所有 pytest/派生/上传之后）**，
否则与决策 3 的强制约束①②正面冲突。

**该 step 对 M2 的时间影响（必须计入）**：它在 Ubuntu 腿上净增 ~2–3m（决策 8 估值）。
按 §5.2 的快样本算（Ubuntu 单腿 1h17m46s，去重省 25m15s ≈ 32.5%），**加上 3m 后省幅降到约 27%**，
M2 的「≥30%」会被它吃穿 ⇒ 处置二选一（本 ADR 指定默认项）：
①（默认）**M2 的降幅按「S3 的去重收益」单列计量，新增诊断 step 的成本不计入其分子**；
② 若 TL 要求按整腿计量，则**把该 step 移入 `ci-fast-lint.yml` 的并行快通道**（多条腿并行，零关键路径成本）。

**唯一保留的 pytest 命令（终裁文本）**：

> 【定位勘误 2026-10-26，随 R-S6a 审读登记】本表与 §3.6 的 yaml 原文把 `collection-crlf` 放在 **Ubuntu 腿**
> （决策 8 时点的默认项）；board 决策 14 已裁定**移至 Windows 腿**（渲染轴与平台轴分离，L2 已被
> run 40207103896 守卫 172/172 实测评证）。落地实现（`bc019e517`）按决策 14 执行，**视为合规**；
> 本表第 5 行的「腿归属」以决策 14 为准，其余列（id / continue-on-error: true / 只 `--co -q` 不跑用例）不变。

```yaml
      - name: Run complete regression suite
        id: pytest-full
        continue-on-error: true
        run: >-
          python -m pytest tests/ -q --cov
          --junitxml=report.xml -o junit_logging=all
          --cov-report=xml:coverage.xml
          --cov-report=json:coverage.json
          ${{ env.DSH_CI_DESELECT }}
```

- `-o junit_logging=all`：**加上**，因为去重前 step25/26 有它而 step27 没有；保留它 ⇒ 派生出的两个面的
  `<system-out>` 语义与去重前一致（R5 对照实验才能在同一口径上做）。
  代价：`report.xml` 变大。**尺寸闸**：若 `report.xml > 64 MiB`，把 `all` 降为 `log`（只记录失败）并在
  ADR 补记一节说明理由，不得再低于 `log`。
- `--cov` 的度量面与调用参数无变化 ⇒ **覆盖率分母不变** ⇒ `pyproject.toml:203 fail_under=60`、
  `ci.yml:508 --threshold 60.0`、trend summary 三个读数全部保持现状（不得因本刀而重定基线）。

### 3.6 C-6 · 汇总判红 step（**R4 的唯一解，必须逐字落地**）

> 【勘误 2026-10-26，同 §3.5 注】下方契约 yaml 中 `OUT_COLLECT: ${{ steps.collection-crlf.outcome }}`
> 属决策 8 时点的 Ubuntu 腿形态；决策 14 后该项移入 **Windows 腿的 gate-summary-windows**
> （`steps.pytest-windows.outcome` + `steps.collection-crlf.outcome` 两行），Ubuntu 腿汇总只判四条。
> 判定表（success/failure/cancelled/skipped/未知 ⇒ RED）与四条机制约定不受影响。

```yaml
      - name: Summarize CI gate outcomes
        id: gate-summary
        if: always()
        env:
          OUT_GUARD:   ${{ steps.deselect-guard.outcome }}
          OUT_PYTEST:  ${{ steps.pytest-full.outcome }}
          OUT_FACES:   ${{ steps.junit-faces.outcome }}
          OUT_COVER:   ${{ steps.coverage-gate.outcome }}
          OUT_COLLECT: ${{ steps.collection-crlf.outcome }}
        run: |
          set -u
          red=0
          check() {  # $1=label  $2=outcome
            case "$2" in
              success) printf '| %-16s | %-10s | ok   |\n' "$1" "$2" ;;
              failure) printf '| %-16s | %-10s | RED  |\n' "$1" "$2"; red=1 ;;
              cancelled) printf '| %-16s | %-10s | RED  |\n' "$1" "$2"; red=1 ;;
              skipped) printf '| %-16s | %-10s | RED  |\n' "$1" "$2"; red=1 ;;
              *) printf '| %-16s | %-10s | RED(unknown) |\n' "$1" "${2:-<empty>}"; red=1 ;;   # 含空串：step id 拼错/被删 ⇒ fail-closed
            esac
          }
          {
            echo "### CI gate outcomes (ADR-S6)"
            echo "| step | outcome | verdict |"
            echo "|---|---|---|"
            check "deselect-guard" "$OUT_GUARD"
            check "pytest-full"    "$OUT_PYTEST"
            check "junit-faces"    "$OUT_FACES"
            check "coverage-gate"  "$OUT_COVER"
            check "collection-crlf" "$OUT_COLLECT"
          } >> "$GITHUB_STEP_SUMMARY"
          if [ "$red" -ne 0 ]; then
            echo "::error::至少一个门禁 step 未完成成功（outcome≠success）；本 job 判红"
            exit 1
          fi
```

**必须遵守的四条机制性约定（写在 commit message 里）**

1. **只认 `steps.<id>.outcome`，绝不认 `steps.<id>.conclusion`**：设了 `continue-on-error: true` 之后
   `conclusion` 会**被改写成 success**，用 conclusion 判红等于把真红变绿（这正是 R4 的头号风险）。
2. **`skipped` 判红**：DoD D1 要求「三个 pytest step 实际执行（非 skipped）」。把 skipped 判红，
   等于在「汇总」这一层永久性地复押了这条判据——将来谁再引入短路，CI 立刻红，而不是悄悄丢信息。
3. **未知/空值判红**：`${{ steps.<id>.outcome }}` 在 id 不存在时解析为空串 ⇒ 必须红，否则改名即静默失效
   （与「merge deselect 静默失效」同一类病）。
4. **汇总 step 自身禁止 `continue-on-error`**，且必须是这组 step 里唯一不带它的。

** Windows 腿同 commit 处理**：给 `Run full regression tests on Windows`（`ci.yml:798`）加
`id=pytest-windows` + `continue-on-error: true`，并在其后加同名契约的汇总 step（只校验它自己一个 outcome）。
理由：两腿结构同构，后续 team-lead/PM 读到的是同一种失败传播语义。

### 3.7 C-7 · 并发约定（终裁）

```yaml
# ci.yml（改）
concurrency:
  group: ${{ github.ref }}
  cancel-in-progress: ${{ github.event_name == 'pull_request' }}
```

```yaml
# ci-fast-lint.yml（新增）
concurrency:
  group: lint-${{ github.ref }}
  cancel-in-progress: true
```

- `cancel-in-progress` 表达式只允许用 `github.*` 上下文（GitHub 限制），本写法满足。
  实现时**首个 PR 落地上 verify**：连续 push 两次，断言第二次的 run 取代第一次。
- `m7-ci.yml:16–18`：**只登记不改**（决策 6），在 Phase 13 台账里写「后续若动，采用同一口径」。

### 3.8 退出码总表（两个新脚本共用）

| code | 含义 | workflow 行为 |
|---|---|---|
| 0 | 成功 | 继续 |
| 1 | **业务性失败**（MISS / NOFILE / 校验不变式不符） | 汇总判红 ⇒ job failure |
| 2 | **输入不可用**（数据源/源文件缺失、schema 不符、不可解析） | 同上（fail-closed） |
| 3 | **内部不可判定**（testcase 归属不了 / 子进程无法启动） | 同上（fail-closed） |
| 4 | **第 2 级自校验失败**（三集合不等 / 某面为空） | 同上 |
| 5 | **第 1 级前置门禁失败**（全量面存在 `classname=""` 且为 collection-error/`<failure>`/未知形态的记录；`<skipped>` 形态＝合法模块级 skip，排除出面并单独计数 `module_skipped`，不判红） | 同上，且必须打印模块清单 |

---

## 4 数据模型与迁移

### 4.1 S2 · deselect 数据源迁移

#### 4.1.1 迁移步骤（**严格顺序，一步一 commit**）

| 步 | 动作 | 回滚点 |
|---|---|---|
| M-0 | 取 `ci.yml:108–193` 的 86 条，生成 `.github/ci/ci-only-deselect.json`（逐条 node；family/batch/restart_condition/evidence 由 `ci.yml:34–105` 的分族文字按族填充） | （与 M-1 同 commit 或紧邻 commit） |
| M-1 | `render_ci_deselect_args.py` + 两腿各加注入 step。**不做「双腿原位 env 清单保留一个 run 做影子对照」**（那会让 ci.yml 同时存在两个真源，正是本次要消灭的形态）；改为注入后在同一 step 里断言 `--deselect` 计数 == 86 | R-S2b |
| M-2 | 删除双腿 env 块里的 172 行条目，原文留只是在 git 历史里；原址加 3–6 行指向注释 | R-S2b（同一 commit，见下） |
| M-3 | 守卫改写为读 JSON + 单次收集（可单独先落，见决策 4 的 WC.1 顺序） | R-S2a |
| M-4 | 分批收紧，每批一个 commit，逐条删 MODE + 必要时回填 `evidence` 为双绿 run id | R-S2d-1…n |

> M-1 与 M-2 **必须在同一个 commit**（否则存在「JSON 有、yml 也有」的双真源窗口，正是本次要消灭的形态），
> 合并为一个回滚点 **R-S2b**。

#### 4.1.2 兼容与防腐

- **不影响本机**：本机从来不消费该清单（`ci.yml:69` 已写「本机不消费本清单」）⇒ 数据源落地不改变本机全量守护语义。
- **守卫必须 fail-closed**：JSON 被误删 ⇒ `render` 与 `guard` 同时 exit 2 ⇒ 汇总红（不是「没清单就全跑导致 86 红」那种噪声，也不是静默放行）。
- **结构保证两腿一致**：不再存在两份可漂移的副本；「凡跑 `pytest tests/...` 的步必须消费这份清单」
  （`ci.yml:54` 的既有判据）由「两腿各自在运行时从同一文件注入同一串参数」实现。
  为防止未来新增的 pytest step 忘记带上，新增两条兜底断言：
  在每个 pytest step 的 command 里保留字面 `${{ env.DSH_CI_DESELECT }}`（现状就是这么写的），
  评审时这是一个可视锚点；此外由 §3.2 的「计数 == 条目数」断言兜底。

#### 4.1.3 注释迁移规则（**红线相邻，必须照做**）

`ci.yml:34–105`（Ubuntu 批次沿革）与 `ci.yml:662–665`（双腿必须逐条一致的纪律原文）
**不删、不改写现有句子的语义**；允许且只允许三件事：

1. 新增行数 ≤ 8 行/处的「指向」注释：写明「清单已迁至 `.github/ci/ci-only-deselect.json`（日期 / 原因），本注释保留为历史沿革」；
2. 把「两处必须逐条一致」这一句**追加**一句「（已由单数据源结构性保证，2026-10-26）」——不删原句；
3. 全文历史批次文字原样保留（它们是 `-466/-467/-469` 等事故的第一手判据）。

违反即视为违反 intake §7.3「注释即判据，删注释前先确认语义」。

### 4.2 S3 · junit 三面派生的数据模型

| 面 | 去重前 | 去重后（ADR-S3） | 差异说明 |
|---|---|---|---|
| `native-report.xml` | `pytest tests/taiji_native`（真实独立执行） | 从 `report.xml` 派生（同一趟执行的目录视图） | 用例集合相等；`<system-out>` 由 `-o junit_logging=all` 保全；`time` 为该用例本次执行耗时（原本也是） |
| `seed-report.xml` | `pytest tests/seed` | 同上 | 同上 |
| `report.xml` | `pytest tests/ --cov` | 不变 | — |
| `coverage.xml` / `coverage.json` | 来自 step27 | 来自同一步 | **完全不变** ⇒ B-4 门与 trend summary 数字不受影响 |

**必须保全的三条不变式**（ violated ⇒ R-S3 回滚）：

1. `N(native) + N(seed) + N(other) == N(report)`（严格相等）；
2. 与去重前基线 run 的三张面 **node-id 集合相等**（WD.2 对照实验）；
3. `coverage.json` 喂出的 B-4 百分数与趋势数字，与基线 run 的差值 ≤ ±0.05pp（浮点/执行顺序抖动）。

### 4.3 迁移失败矩阵

| 失败 | 检测点 | 动作 |
|---|---|---|
| `$GITHUB_ENV` 多行/特殊字符失败（A2 不成立） | M-1 的计数断言 | 退回「写临时文件 + `$(cat)`」方案（§3.2），**不得回到双腿各一份** |
| 单次 `--co` 收集仍 >3m | 守卫 step 计时 | 退回「按 families 分 2–3 批收集」，仍保持子进程数 ≪ 172 |
| 派生自校验失败（不等/N=0） | `split_ci_junit_faces.py` exit 3/4 | revert R-S3，回到三跑，重查归属算法 |
| 覆盖数字变化 >0.05pp | WD.2 对照 | revert R-S3；若去重后覆盖率反而更高/更低，说明 exec. 集合有变，须逐条解释后再重做 |
| classname 语义随 pytest 升级变化 | **已用版本钉固定**（见下） | 升 pytest 时必须在**本机**先重跑 C-4 的自校验再改 CI |

**强制附带项（新增，作为 R-S3 的一部分）**：CI 的 `pip install` 步骤把
`pytest` / `pytest-cov` **钉版本**为 CI 实测在跑的版本（`pytest==9.1.1`、`pytest-cov==7.1.0`，run 40203419431 读数）。
理由：本 ADR 的派生正确性依赖 junit 的 `classname` 语义，而这是 **pytest 的版本行为**；
本仓 R7 纪律（`ci.yml:262–265`）已写明「凡拿输出当判据的工具必须钉版本」——pytest 现在就是这种工具。
**判据语义不变**（只是把浮动版本变成固定版本）。

### 4.4 兼容性风险

- `report.xml` 变大（`-o junit_logging=all`）⇒ artifact size、Actions UI 渲染速度。用 §3.5 的 64MiB 闸控制。
- 去掉 step25/26 后，**历史 PR check 名字会消失**（`Run native architecture contracts` 等两条 check）⇒
  若有 branch protection 规则按 check 名要求，需 TL/owner 确认改为要求 job 结论。【已关闭 2026-10-26，board 决策 10（主理人亲验）：删两条 check 名无外部影响。】（见 §9.3）
- 派生两面不再是独立执行 ⇒ 任何把这 XML 当「独立测试执行记录」的外部脚本，需要在文件头 comment 里看到说明（已加，§3.4）。

---

## 5 部署拓扑与非功能指标

### 5.1 拓扑（不变 / 变化）

| job | runs-on | 现状 | 落地后 |
|---|---|---|---|
| `test (3.12)` | ubuntu-latest | 24+ steps，含 3 个 pytest | 同上但只剩 1 个 pytest + 派生 + 汇总 |
| `test-windows` | windows-latest | 单 pytest | 单 pytest + 汇总 step |
| `docker-build` | ubuntu-latest | 不变 | 不变 |
| `startup-smoke` ×2 | ubuntu-latest | 不变 | 不变 |
| **`lint-fast`（新增）** | ubuntu-latest | — | setup-python + **torch CPU（必装，实测收口，见下）** + pip deps + ruff/black/mypy-core + froze-marker audit |

> 【收口 2026-10-26，commit `321e66a54`】本节原文「torch? 不需要 torch」的问号**已实测否定：必须装**。
> 理由：mypy core 棘轮（baseline 0）跑 `mypy seed taiji`，缺 torch 时每个 `import torch` 都报
> "Cannot find implementation or library stub" ⇒ 错误数 >0 ⇒ 棘轮假红，且读数与 ci.yml 不可比
> （违反 §6.3 ⑤ 验证项 3「同一 commit 两通道读数一致」）。落地口径：安装面与 ci.yml 逐字对齐
> （torch==2.13.0+cpu + ruff==0.16.4 / black==26.5.1 / mypy==2.3.1，去掉本通道不消费的
> pytest/pytest-cov/pip-audit 三项）；快通道并发组用 `lint-` 前缀独立于 ci.yml，两通道不互相取消；
> timeout 10m 兜 pip 冷缓存（torch CPU wheel ~200MB），≤6m 仍为目标值——**R-S4c 首跑实测后回填真实时延**，
> 若 mypy 无缓存冷跑 + torch 安装长期顶穿 6m，登记技术债（mypy 缓存/拆 job）而非放宽读数可比性。
> 快通道其余范围维持原判：不跑 8 个 verify、不跑 deselect 守卫、不跑任何 pytest。
>
> 【时延回填 2026-10-26】R-S4c 首跑实测（run 40258536518）：lint-fast **success 3m08s**，≤6m 达标，
> **不挂技术债条目**。保留一条观察项：本轮 Install torch 与 mypy 均为**热缓存读数**，「pip 冷缓存 + mypy 冷跑」
> 双冷场景暂无样本；出现双冷样本且 >6m 时再评估技术债（mypy 缓存/拆 job），在此之前不动契约。

> 快通道**不跑 8 个 verify、不跑 deselect 守卫、不跑任何 pytest**（这三项都属于「要么贵、要么必须钉在原处」）。
> ~~快通道不装 torch~~（【已收口】torch 必装，见上方状态注——它是 mypy 读数可比性的前提，不是可选成本）。
> 快通道组成：`Lint with ruff` / `Lint with ruff bugbear/simplify (baseline 0)` / `Format check with black` /
> `Audit frozen markers against the formatter` / `Type check core modules with mypy (baseline 0)`。
> 目标 ≤6m；它是**信息通道不是门禁替代**，fail 使该 check 红，但 UI 文案与台账必须写明「CI（ci.yml）仍是唯一发布门禁」。

### 5.2 时延预算（**全部取自 §2.1 的真实读数**）

| 指标 | 现状（实测） | S3 后（按 §2.1 逐样本推算） |
|---|---|---|
| Ubuntu 单腿 | 1h17m46s – **2h42m57s** | 52m31s – 1h49m19s（**−32.3% … −32.9%**） |
| Windows 单腿 | 1h02m07s – 1h52m18s | **不变** |
| **一次 run 墙钟**（两腿并行取 max） | 1h24m33s – 2h42m57s | 1h24m33s – 1h52m18s（**0% … −33%，平均 ≈ −21%**） |

⇒ **M2「单腿 ≥30%」达标，但余量只有 2.4pp**；且「一次 run 墙钟」的下降被 Windows 腿夹住。
D4 的进一步压缩**不属于本轮范围**（登记为 D-1）。

### 5.3 timeout（WD.4 的取值规则）

- 现状 `ci.yml:33` / `ci.yml:661` = 180m，且有 5 条上调轨迹注释。**不得直接回缩到 60/90m。**
- 取值规则（终裁）：`timeout = max(ceil(P95(实测单腿) × 1.3), 120m)`，且**必须有 ≥3 次落地后的两腿全绿读数**才允许改。
- 用当前 §2.1 的 P95 代入（Windows 1h52m18s 是最慢样本）⇒ **推荐 150m**（112m × 1.3 ≈ 146m，向上取整到 150m）。
  若届时仍有样本超过 115m ⇒ 保持 180m 不动。**比 150m 更小的值一律不接受**（现场证据不足）。
- 注释处理：**追加**新读数与原注释同一条时间线之后，**禁止覆盖**原 5 条轨迹（intake §7.3）。

### 5.4 其它非功能指标

| 项 | 目标 |
|---|---|
| 守卫 step 耗时 | ≤2m（硬上限 3m） |
| 派生 step 耗时 | ≤30s（纯 XML 解析，808 文件级规模下应远小于此） |
| artifact 名稳定性 | `verify-reports-*` / `core-reasoning-coverage` / `test-results-*` / `test-results-windows` **全部不变** |
| 失败可诊断性 | 任一刀失败 ⇒ job summary 里有「哪一步 outcome 是什么」的表（§3.6 产出） |
| 幂等 | 两脚本同输入 ⇒ 同输出；workflow 无副作用地重跑同一 commit ⇒ 同结论 |
| 依赖 | 两个新脚本 stdlib only（无新增 pip 包；唯一的依赖层改动是把 pytest / pytest-cov 钉到实测版本） |

---

## 6 ADR

> 每条 ADR 固定六段：**选定方案 / 被否选项及否定理由 / 改动文件清单 / 判据语义是否变化 / 验证方式 / 绑定的回滚点**。

### 6.1 ADR-S2 · CI-only deselect 机制的替代方案（WA.2）

**① 选定方案：S2-A（清单单源外置，JSON）+ S2-D（守卫单次收集），并把「重启条件非空」升级为硬校验。**

三个子决策：
- 数据源 = `.github/ci/ci-only-deselect.json`（为什么 JSON 见 §3.1）；
- 消费方式 = 两腿各一个 runtime 注入 step（`render_ci_deselect_args.py --mode env`），保留 `${{ env.DSH_CI_DESELECT }}` 作为消费锚点；
- 守卫 = 一次 `pytest --co` 收集 56 个去重文件 + 集合比对 + **MISS/NOFILE 判红、UNCOLLECTABLE 告警**（§3.3），
  元数据校验（每条必带非空 `restart_condition`）由守卫执行 ⇒ **D2 的「每条带重启条件」从承诺变成结构**。

**② 被否选项及否定理由**

| 候选 | 否定理由 |
|---|---|
| **S2-B（pytest marker）** | ① 改动面最大：约 60 文件加 marker + `pyproject.toml:68–73` 注册 + 两腿 pytest 命令改 `-m`（Windows 腿还要与现有 `-m "not slow"` 组合成 `not slow and not ci_unproven`，是一次语义变更）；② 必须开 `--strict-markers`，否则拼错的 marker 静默失效——与当前「落空静默」同形缺陷，等于换个地方再犯；③ DoD D2 要的「每条源文件层面的重启条件与沿革」被拆散进 56 个文件的装饰器里，反而削弱审计可读性。**它确实是唯一对「改名/移动用例」免疫的机制，故不作否决而作「二期候选」**，触发条件写进 Phase 13 台账：**收紧后 30 天内条目数回涨，或再次出现 ≥1 次落空事故 ⇒ 启动 marker 迁移**。本次不执行。 |
| **S2-C（守卫降级 warn）** | **明确否决**。它换来的是「守卫不再阻断后续 step」，而这个性质 S6-i（ADR-S6）已经用「continue-on-error + 汇总判红」给出了，**且后者保留了守卫的红**。S2-C 在拿到同款收益的同时把唯一的信号源降成 warning ⇒ 净收益为负。守卫保持 blocking。 |
| **S2-D 单独使用** | 不够：它提速但消除不了「两腿人工同步」这一根因 ⇒ 必须作为 S2-A 的强制伴随项，不单独成刀。 |

**③ 改动文件清单**

| 文件 | 动作 |
|---|---|
| `.github/ci/ci-only-deselect.json` | **新增**（86 条，node 与现状逐字一致） |
| `scripts/training/verify_ci_deselect_list.py` | **改写**（删 workflow 扫描 → 读 JSON；多条子进程 → 单次收集；加元数据校验） |
| `scripts/training/render_ci_deselect_args.py` | **新增** |
| `.github/workflows/ci.yml` | 删 `106–193`、`666–753` 的 env 条目块（保留注释，追加指向）；两腿各加 1 个注入 step；守卫 step 加 `id` + `continue-on-error`（后者与 ADR-S6 同 commit） |

**④ 判据语义是否变化**：**否**。清单内容逐条不变；㊵-466/467/469 三条既有判据（「凡跑 `pytest tests/...` 的步必须消费这份清单」「本机不消费」「条目必须自证命中」）由 §3.1/§3.2/§3.3 的结构继承，未改一字。唯一变化是守卫对「收集失败」不再重复计红（§3.3 已论证：该事实的红色责任迁移到 pytest step + 汇总 step，**未放宽任何东西**）。

**⑤ 验证方式（本机可复现，禁止「推上去看 CI 变不变绿」）**
1. `python scripts/training/render_ci_deselect_args.py --mode args | tr ' ' '\n' | grep -c -- '--deselect'` == **86**；
2. 与 `git show 60f32a95f:.github/workflows/ci.yml` 里任一条目做集合 diff ⇒ **空**；
3. 本机跑一次新版守卫（不依赖真机复现：注入一条假 node ⇒ 必须 exit 1；把 JSON 改名 ⇒ 必须 exit 2；
   删某条 `restart_condition` ⇒ 必须 exit 2）；
4. 计时在同一台机器上对比新旧守卫（目标 ≤2m）；
5. **必须**做一次旧/新判定对照运行（同一 HEAD 上分别跑两种守卫实现，MISS 集合必须一致）——plan §7 R5 的同款要求。

**⑥ 回滚点**：`R-S2a`（守卫改写）、`R-S2b`（数据源 + 注入 + 删 env，单一 commit）、`R-S2d-1…n`（分批收紧）。

---

### 6.2 ADR-S3 · Ubuntu 腿 tests/ 跑三遍的去重（WA.3）

**① 选定方案：S3-A（单跑全量 + junit 派生两张子面），派生主键 = `classname` → 磁盘文件路径归属，配三面自校验。**

终裁要点（与 team-lead 2026-10-26 的实测一致，并补两条加固）：
- **主键不是 `file` 属性**（A1 已证伪），也不是字符串前缀（有 `tests.seed` 吞 `tests.seed_platform` 的陷阱，
  `tests/seed_platform` 实测 8 个文件），而是：**classname → 逐段回退解析出磁盘上真实存在的 `.py` 路径 → 用 `Path.is_relative_to()` 判归属**（§3.4）。这条路径对「类级用例」和「收集失败记录」都成立，且**不需要 `--co -q` 兜底**（⇒ R7 降级为「已缓解」）。
- `--co -q` 兜底**保留 as **如果一个 include** contract violation 清零。 (Not needed as primary; keep only as WD.2 的对照实验工具。)
- **附加强制项：CI 的 pytest/pytest-cov 钉版本**（§4.3），因为本方案的正确性依赖 pytest 的 junit 行为。
- **保留 `-o junit_logging=all`**（尺寸闸 64MiB）。
- **不做** S6-ii（见 §6.4）——它与 S3 无关，但两者都想「让结论更快」，这里一并裁掉避免混淆。

**② 被否选项及否定理由**

| 候选 | 否定理由 |
|---|---|
| **S3-B（三分区互不重叠）** | 覆盖率分母变小 ⇒ 触碰 intake §3「覆盖率阈值不动」红线。**已被决策 5 排除**，这里不再重开。 |
| **S3-C（拆并行 job + coverage combine）** | ① 跨 job 产物流转 + 合并失败语义，复杂度高于派生 XML 一个数量级；② Windows 腿结构不同 ⇒ 双腿不再同构，违背本仓「两腿可比」的一贯纪律；③ 收益与 S3-A 相同（都是「只跑一次」）。**仅在 C-4 自校验不可实现时才考虑**。 |
| **S3-D（xdist `-n auto`）** | Ubuntu runner 2 核，收益 ~1.5–2×，但需新增依赖 + 改变并发执行语义（部分测试可能有共享 `checkpoints/` 锁竞争，本仓历史上出现过：`ci.yml:505–506` 注释）。**登记为 Windows 腿后续提速的候选**（D-1），本次不做。 |
| **「保留三条 pytest 命令、只给它们加 `-p xdist`」** | 同 S3-D，且保留三遍重复，收益最小。否。 |

**③ 改动文件清单**

| 文件 | 动作 |
|---|---|
| `scripts/training/split_ci_junit_faces.py` | **新增** |
| `.github/workflows/ci.yml` | 删 step `ci.yml:467–473` 与 `ci.yml:475–479` 两个 step；改写 `ci.yml:481–491`（加 `-o junit_logging=all`、加 `id`、`continue-on-error`）；新增 `Derive native/seed junit faces` step；两条 target deps 的 `pip install` 行加 `pytest==9.1.1 pytest-cov==7.1.0` 钉版本 |
| `pyproject.toml` | **不动**（`fail_under=60` 与 `[tool.coverage.*]` 一概不动） |

**④ 判据语义是否变化**：**否**。① 覆盖率：`--cov` 仍是全量 `tests/` + 全量 source ⇒ 分母不变，B-4（60.0）与 trend 读数不变；
② artifact：三张 XML 的名字与 Upload 路径（`ci.yml:535–544`）一字不改；③ **check 名字会减少两条**（`Run native architecture contracts` / `Run Seed model-boundary contracts`）——这是唯一的外部可见差异（影响面见 §4.4）。

**⑤ 验证方式**
1. 本机单元：`split_ci_junit_faces.py` 对一份**合成 report.xml**（含函数/类/参数化/`<error>` 四种形态 + `tests/seed_platform` 反例）⇒ 覆盖自校验成功；把 `tests/seed_platform` 的一个 classname 混进 seed 面 ⇒ **必须 exit 4**（这是尾点陷阱的回归测试，**必须写成用例**）；
2. **WD.2 对照实验**：与基线 run（推荐 `40171918472`——五次成功样本里 step 读数最完整的一条，三张 junit artifact 齐全）做 node-id 集合比对 ⇒ 严格相等；
3. 覆盖率数字 diff ≤ 0.05pp；
4. Ubuntu 单腿实测下降 ≥30%（按 §2.1 五样本推算为 32.3%–32.9%；WD.1 落地后必须回到这一节回填「实测值 + 取值日期 + run id」，推算值不得作为结案依据）。

**⑥ 回滚点**：`R-S3`（ci.yml + 新脚本同一 commit，revert 即删除脚本）。

---

### 6.3 ADR-S4 · 并发治理（WA.4）

**① 选定方案：S4-b（按事件分化）+ S4-c（快通道）组合；并附一条执行纪律。**

```yaml
concurrency:
  group: ${{ github.ref }}
  cancel-in-progress: ${{ github.event_name == 'pull_request' }}   # push→main: 不取消；PR: 取消
```

+ 新增 `.github/workflows/ci-fast-lint.yml`（lint/black/frozen-marker/mypy-core，`cancel-in-progress: true`，≤6m，见 §5.1）。

**明确回答「是否保留 `cancel-in-progress`」**：
- **PR 事件：保留 `true`**。PR 上连续 push 是同一次 review 的迭代，旧 run 的结论对同一 head 无意义，取消省额度且反馈快；
- **push 到 main：改为 `false`**。main 上每个 commit 都是一个已发布状态，**每个都必须留下完整结论**——这正是本仓近 40 次 run 只有 5 次 success 的机制性原因。
- **代价（量化，取自 §2.1）**：落在 main 上的 run 不再互删 ⇒ 队列改为串行排队。owner 一天十余次 push（≈ 12–13 次，集中在 ~12h 内，平均间隔 ≈ 1.0–1.1h），而 S3 之后一次 run 的墙钟（两腿取 max）为 **1h24m–1h52m**（慢机样本外推上界 ~2h45m）⇒ 忙碌时段利用率 ρ ≈ 1.3–1.6 > 1 ⇒ **白天队列会积压，夜间自然排空**。

> 【实证追记 2026-10-26】R-S4 落地（`6853d9589`）后首日观测：新 push 到 main 时，在飞 run **未被取消**、与前驱并行且结论各自保留（首个观测对：基线 run `9da09708a` 出结论期间，字形修复 run 40258536518 在飞）——与 push→main `cancel-in-progress: false` 的设计预测一致，R6 风险的「队列积压」面开始按预测形态运行。后续按 §2.1 的 ρ 估算持续观测 queued 积压（R6 阈值：queued ≥3 持续 6h ⇒ revert R-S4）。
  - **最坏反馈时延估算**：典型（间隔 ≥2h 的零星 push）≈ **一次 run ≈ 1h24m–1h52m**；繁忙日隊尾 ≈ **4–6h**（当日最后一批在次日凌晨前跑完）；理论上界受「当天推送总数」约束，**不会无限增长**。
  - **对比现状**：现状是「除了当天最后一次 push，其余所有 run 被自己取消」⇒ 中间 commit 的结论**永久缺失**，等价于 ∞。**S4-b 的代价（几小时延迟）严格优于现状（无结论）。**
- **为什么不是 S4-a（全部关闭）**：PR 上也关闭 ⇒ PR 上每次迭代都排队一小时多，review 反馈被拖到不可用地步，且白白烧额度。不采纳。
- **为什么不是 S4-d（只做 S3，保留 true）**：保留 true 就等于保留「除最后一次 push 外全部失忆」的机制，
  与本立项「根治」的目标直接冲突。而且 §2.1 揭示的是硬币另一面：即便腿变短了，
  中间所有 commit 仍然一次结论都留不下来。不采纳。
- **为什么加 S4-c**：S4-b 让时延从「无结论」变成「几小时」，需要一个 ≤6m 的签名通道，让每一刀落地后立刻知道 lint/format/mypy 是否挂了；`ci.yml:262–265` 的 R7 纪律已确立「工具必须钉版本」，所以快通道与慢通道的 ruff/black/mypy 读数天然一致。
- **配套执行纪律（不是代码，是本 ADR 的组成部分）**：实现期每天对 main 的「验证性质 push」≤ 2 次，间隔 ≥ 一次 run 时长；日常增量走 PR（PR 上 cancel:true，反馈快）。

**④ 判据语义是否变化**：**否**。只改运行节奏与反馈时延，不碰任何 `===`、阈值、assert。

**⑤ 验证方式**
1. PR 上连续 push 两次 ⇒ 第一个 run conclusion = `cancelled`，第二个跑完（这是「保留 true」生效的证据）；
2. main 上连续 push 两次 ⇒ **两个 run 都跑完**（第二个排队而不是取消）——这是 R6 的核心验证；
3. `ci-fast-lint.yml` 单次 ≤6m，且与 `ci.yml` 的同名检查在同一 commit 上结果一致（比对 4 个 check）；
4. 落地后连续 3 次 push，至少有一次给出**两腿完整结论**（WD.3 的完成标准）。

**⑥ 回滚点**：`R-S4`（并发治理）、`R-S4c`（快通道 workflow 文件，独立 pathspec）。
**回退触发条件**（写进台账，不必等 CR）：main 上 `queued` run ≥ 3 持续超过 6h ⇒ `git revert R-S4`
（`ci-fast-lint.yml` 保留不受影响）。

---

### 6.4 ADR-S6 · 门禁解耦（WA.5，决策 3 已纳入范围）

**① 选定方案：S6-i（continue-on-error + 汇总判红），不做 S6-ii。**

- 参与 `continue-on-error` 的四个 step（**落地时的形态**，见 §1.4 / §3.5）：
  `deselect-guard`、`pytest-full`、`junit-faces`、`coverage-gate`。
  注意：**按决策 4 的顺序（WD → WC.1 → WE），WE.1 落地时 step25/26 已不存在**，因此实际是四步而不再
  是 plan 原文的「467/475/481/507」。下游**必须按落地后的实际形态**执行，不得照抄旧行号。
- 汇总 step 契约：§3.6（逐字给出，含 `outcome` 而非 `conclusion`、`skipped` 判红、空值判红）。
- Windows 腿：`ci.yml:798` 加 `id=pytest-windows` + `continue-on-error: true` + 同名汇总（同一 commit）。

**R4 的头号风险如何被关闭**（三条必须同时成立）：
1. 五个 `continue-on-error`（`deselect-guard` / `collection-crlf` / `pytest-full` / `junit-faces` / `coverage-gate`）与汇总 step **同 commit**（决策 3①）；
2. 汇总 step 只认 `outcome`（含 skipped/cancelled/未知全判红）——**这一条让「漏掉汇总」在机械上不成立**：
   少任何一个被测 step，它的 outcome 解析为空 ⇒ 汇总判红，而不是绿；
3. M5 门必须**注入一次红**并断言 job conclusion = failure（决策 3②）。注入方式（本机/CI 均可复现）：
   临时把 `DSH_CI_DESELECT` 的注入 step 换成 `exit 1`，或临时给 `pytest-full` 加一条指定的 failing test，
   **注入必须是可逆的单次实验 commit，跑完即 revert**。

**junit/coverage artifact 的上传语义是否受影响**：**不变且更好**。
① 上传 step 原本就有 `if: always()`（`ci.yml:449/511/536`），继续照跑；
② 过去 step25 一红 ⇒ 后续全跳过 ⇒ `coverage.xml/json` 根本不产生 ⇒ 覆盖率结论丢失；
落地后 pytest step 即使失败也会写出 `report.xml` 与 `coverage.*`（pytest/coverage 在正常退出路径写盘），
⇒ **失败时反而有产物可供取证**；
③ 唯一新增失败模式：极端情况下（进程被 kill / OOM）文件缺失 ⇒ B-4（`--from-report coverage.json`）会以
显式报错判红，而不再「被跳过而无声」。这是严格改进，但要写进 Phase 13 台账让人们预期到这个新的红的外观。

**② 被否选项及否定理由**

| 候选 | 否定理由 |
|---|---|
| **S6-ii（把前置门禁拆成并行 quality-gates job）** | 用 §2.1 的真实读数算：S3 之后 Ubuntu 单腿 = 52m–1h49m，而一次 run 的墙钟 = max(Ubuntu, Windows) ∈ [1h24m, 1h52m]，**Windows 是关键路径**；拆 job 只能省掉 Ubuntu 腿前面约 7–12m 的串行前缀（含守卫 6m，而守卫自己马上会被 WC.1 压到 ≤2m），对 **run 墙钟零收益**。代价是 job 数 +1、跨 job 结论阅读成本 +1、check 名字再变一批。**裁定：本次不做**，登记为「待 Windows 关键路径被解决后再评估」（与 D-1 绑定）。 |
| **只给部分 step 加 continue-on-error（不加汇总）** | 等于把真红变绿，比现状更糟。**禁止**出现在任何 commit 里。 |
| **用 `if: always()` 代替 `continue-on-error`** | `if: always()` 只在「前一步失败」时继续，但 job 结论仍会红；看似等价，实则它**不能保护「手动取消/跳过」语义**，且会把「成功路径」也变成「总是运行」，容易掩盖上游短路。**统一用 continue-on-error + outcome**，不用 always()。 |

**③ 改动文件清单**：`.github/workflows/ci.yml`（Ubuntu 腿 4 个 step + 汇总 step；Windows 腿 1 个 step + 汇总 step）。

**④ 判据语义是否变化**：**否**。所有原本会红的事实仍然红；唯一变化是「红色如何在 job summary 里呈现」。对照如下：过去「后续 step skipped」这种静默状态不会被任何人看到；现在它会被显式抬到 `RED`（`skipped` 判红）——这是把隐性失效转成显性失效，属本立项目标，不是判据放宽。

**⑤ 验证方式**：M5（注入一次红 ⇒ step26/27/28 仍执行且 job conclusion = failure）；
外加「注入一次 **skipped**」实验（临时把某 pytest step 的 `if:` 设成永假）⇒ 必须红（证明 skipped 判红生效）。

**⑥ 回滚点**：`R-S6a`（唯一 commit，含五个 continue-on-error、新 id、两条汇总 step）。

---

### 6.5 ADR-S1 · 导入期治理校验（WA.1，**本阶段只做候选评估，不作终裁**）

> 终裁依赖 api-tester 的 `W0.2 三子条件定因`。本节交付的是：**评估矩阵 + 决策表**。
> owner 结论一到即可按下表收口，无需重启裁决。

#### 6.5.1 现状（已核对行号）

`scripts/training/taiji_r2_dialogue_continuation_extension.py:1546–1548` 模块级语句
`PER_ARM_CPU_LIMIT, AGGREGATE_CPU_LIMIT = effective_budget_limits()`；
`effective_budget_limits()` 自 684 行起，v1.3 分支三子条件（`750–763`）：
`boot_audit_identified` / `rendering_tolerant_match(boot_audit_sha256)` / `boot_sources_match`；
任一不成立即 771 行 raise `ExtensionLedgerRefuse("v1.3 boot audit or source binding differs")`。

#### 6.5.2 四候选评估矩阵

评分说明：**轴 A** = 「模块可导入性是否与治理校验解耦」（D3 字面）；
**轴 B** = 「那 19 个模块的用例能否在 runner 上真正跑起来」（D3 的意图，也是 D1 双绿的质量前提）；
**轴 C** = 与 §2.3 实测消费面的冲突；**轴 D** = 改动面与验证成本。

| 候选 | A 解耦 | B 用例真跑 | C 消费面冲突 | D 改动面 / 验证成本 | D3 结论 |
|---|---|---|---|---|---|
| **P1 惰性求值（PEP 562 `__getattr__`）** | **部分**：仅 `X.PER_ARM_CPU_LIMIT` 属性访问被延迟；`from X import PER_ARM_CPU_LIMIT` 仍在导入期解算 | **部分**：取决于消费风格 | **有 3 处硬冲突**：① `test_…extension_contract.py:72–73` 的 `monkeypatch.setattr` 会先读旧值 ⇒ 触发 `__getattr__` ⇒ 在注入之前就 raise（`raising=False` 也救不了，它只吞 AttributeError，不吞 ExtensionLedgerRefuse）；② 同文件 `:144–145 / :164–165 / :456 / :701–702` 的**直接赋值**（非 monkeypatch，语义与之不同，需逐个确认在新语义下不会触发 getattr）；③ `run_…daily_dialogue_continuation_extension.py:1203–1204` 的 `global` 重绑定与 `__getattr__` 语义冲突（setattr 到事务性 module 属性 vs `__getattr__` 只在缺失时触发），且该 runner 内裸常量引用 13+9 处必须逐个改写或保留 import | 中高：1 处 from-import（`runner:44`）+ ~22 处裸引用 + `__all__:1551–1571` + 若干测试的改写；需 W0.1 同构环境 | **单独使用只部分满足 D3** |
| **P2 conftest 层降级为 skip** | **否**（模块仍不可导入，异常照抛） | **否**（19 模块用例仍跑不了，但**其余全部照跑**） | 无冲突（不动治理代码，只动 `tests/conftest.py`） | 最低，**可不依赖真机复现**（本机注入强制 raise 即可验证 hook） | **只满足 D3 的「可跳过 + 明确失败语义」半句**，但**治的是另一个病**：把「 Interrupted: 19 errors ⇒ 整腿失明」变成「19 模块带 reason skip + 整腿其余全跑 + 显式红」 |
| **P3 校验挪出常量初始化** | **是**（1548 改为 BASE/占位，校验进显式入口） | **是**（19 模块照常跑） | **有 2 处**：① `runner:260` 的不变量 `continuation_budget.effective_budget_limits() != (PER_ARM_CPU_LIMIT, AGGREGATE_CPU_LIMIT)` 需重述——在 P3 下两边可能都是 BASE，断言会退化成永假；② `global` 重绑定的位点（`runner:1203–1204`）与新入口的调用时机必须重新钉住；另有**沉默降级风险**：任何消费点漏改即静默使用 BASE 值 ⇒ 必须在 6 处调用点（`runner:260/1204`、`run_taiji_r2_resume.py:215`、`prepare_…:126–137`、`checkpoint_resume_adapter:102/133`、`fast_ledger_resume:112/182`、`fast_ledger_resume_v2:143`）逐点加断言 | **最高** | **最完整满足 D3**，但成本与风险也最高 |
| **P4 只修根因（映射/字节层）** | 否（1548 不动） | 是（修好后 19 模块自然能跑） | 无 | 最低，**但前提是 W0.2 定因成功** | **单独使用不满足 D3**（决策依据：plan §3.1 与 DoD D3 原文）。plan 侧原本建议配 P1/P3；本文的推荐组合见 §6.5.4：**P4 + P2 为保底**。（状态更新：Q-S1 已决（决策 16）owner 答「否」，P3 不放行 ⇒ **终态组合就是 P4 + P2**，D3 记「部分满足」） |

#### 6.5.3 红线标识（**只标识，不放行**）

- **P1**：**可能触及** intake §3「不改变任何冻结判据/封印语义」。理由：它把 v1.3 boot audit 校验的**触发时机**
  从「进程导入该模块的瞬间」推迟到「首次取用常量」。三个子条件的输入、阈值、拒绝文案**一字不改**，
  但「拿到预算常量之前必然已被校验」这一**保证**变弱了（可延迟 ⇒ 存在「进程先拿到值、后失败」的路径被关闭，
  但也存在「进程一直不取用 ⇒ 从不校验」的新路径，正是 S6 才治完的那种「静默失效」家族）。
  ⇒ **我的判断：属于「判据的执行面」变化，不是「判据的内容」变化，但仍需 owner 是非决策**（§6.5.6 Q-S1）。
  **【已决 2026-10-26】owner 答「否」（board 决策 16）：P1 判触红线不放行，本判断与 owner 结论一致的终态是 P1 永久出局。**
- **P3**：**更明显地触及同一条红线**。理由：它把 v1.3 校验从「隐式、无条件、一次性」变成「显式、由调用点负责」，
  ** any 漏改 ⇒ 静默使用 BASE（预算级别的沉默降级）**。`taiji/organs.py`、`fabric.py`、`model.py` 不在本面，
  但「v1.3 boot audit」是 owner 在飞的治理面 ⇒ **必须 owner 放行**。
  **【已决 2026-10-26】owner 答「否」（board 决策 16）：P3 不放行，WB.2 取消，防复发改走「记录方式规范」路线（决策 16/17）。**
- **P2**：**不触及**红线（治理代码零改动，判据一次都没动）。它是四条里唯一「不需要请示也能落」的结构刀，
  但它单独不满足 D3 的 A 轴 ⇒ 只能作为**补充刀**，不能作为**结构刀**。
- **P4**：**是否触及取决于 W0.2**：若失败落在 `seal_ast` 的**映射**（`repo_relative` / `resolve_recorded_path` /
  `verify_path_pin`）⇒ 修的是「同一事实的识别方式」，**我的判断：不触及**（与 2026-10-25 那次
  「extension_contract 6 条摘除（治理面校验跨机化）」同型，owner 已有先例裁定）；
  若失败是因为**字节真的不同**（CRLF/LF 或 audit 文件被重写）⇒ 任何「放宽容忍」的动作都**直接触及红线**（这是把判据本身改松），必须停工请示。

#### 6.5.4 （补充）我认为应当被考虑的第五条路：P2′ **＝ P4 + P2 的组合**

理由：即使 P4 修好了根因，「导入期强校验」这个结构仍然会在**下一次**任何跨机差异出现时，
把整条 Windows 腿打成 Interrupted（今天的失明就是它造成的）。P2 是 S6 在 **collection 层**的同构解：
它不改变判据，只是把「失明」降级为「局部 skip + 显式红 + 其余全跑」。
⇒ **无论最终选 P1/P3 与否，我都建议 P2 作为独立一刀落地（成本最低、零红线、可独立 revert）**。

#### 6.5.5 决策表（W0.2 结果 → 选定候选）

> 【状态注 2026-10-26】本表是 W0.2 回传**前**预注册的决策辅助；W0.2 实际落在 X1/X4（主）+ X3（次），
> 且 Q-S1 已决（board 决策 16，owner 答「否」）⇒ 表中所有涉及 P3 的分支**全部作废**，终态走 §6.5.7 收口（P4+P2）。

| W0.2 判定 | 失败子条件 | 选定结构刀（默认路线） | 是否需 owner 放行 | 备注 |
|---|---|---|---|---|
| **X1** | `boot_audit_identified` 假，且左右值差异是**路径形态**（`E:\Seed\…` vs `/home/runner/…`） | **P4（修 `seal_ast.repo_relative` / `resolve_recorded_path` 的跨机归位）+ P2**（分两刀落地，先 P4 后 P2） | **否**（P4 属修映射，不改判据内容；P2 零改动治理代码） | 与 owner 2026-10-25 的「治理面校验跨机化」先例同型；仍建议在动工前把 diff 给 owner 过目（R2） |
| **X2** | `rendering_tolerant_match(boot_audit_sha256)` 假，且 audit **文件内容在两台机器上确实不同** | **停工，回 owner**：不是技术问题，是「audit 该以哪一份为准 / 是否重新封印」的治理问题 | **是**（必经） | 期间可用 P2 先把失明解掉（P2 不需要放行） |
| **X3** | `rendering_tolerant_match` 假，但差异**仅是换行/渲染**（属已登记的同一妥协） | **P4（把该差异纳入已有 `rendering_tolerant_match` 的既有口径，**不得**新增更宽的容忍）+ P2** | **否，但须 owner 事后追认**（本仓历史上同款妥协都留了 owner 裁定痕迹） | 「扩大容忍」= 判据变松 = 红线；只有「纳入既有口径」可接受 |
| **X4** | `boot_sources_match` 假，左右值差异是**路径 key 形态** | **P4（修 `seal_ast.verify_path_pin` 的 key 归一化 / `repo_relative`）+ P2** | **否** | 同 X1 |
| **X5** | **三条件在 runner 上全通过**，失败另有来源（如另一处模块级调用 / 依赖导入顺序 / OS 特有行为） | **P4 不适用** ⇒ 结构刀必须选：**P3（若 owner 放行）否则 P2 单独**（并向 owner 说明 D3 只能部分满足） | **是（P3）/ 否（P2 单独）** | 此时 X5 的「另有来源」必须先由 W0.2 给出具体位点，否则 P3 也无从下手 |
| **X6** | 三条件通过但 import **仍** 红，位点在本面团以外（例如 `scripts/` 里另一模块也有模块级校验） | 按 §6.5.5 的 X1–X5 规则**对真正失败的那个判据重试一遍**；同一标准适用于任何新位点 | 视具体位点 | 这正是 P2 应当独立落地的理由：它在任何 X* 分支下都降低失明半径 |

**推荐组合（我的建议，供 owner 决策时的默认项）**：`P4 + P2` 为**保底两刀**（零红线 / 或低风险 / 可独立 revert / 立刻把
Windows 腿从「失明」变成「局部 skip 且其余全跑」）；若 owner 对 Q-S1 放行，则改为 `P4 + P3 + P2`，其中 P2 保留作为兜底网。
**【已决 2026-10-26】Q-S1 已答「否」（board 决策 16）⇒ 终态组合定格为 `P4 + P2`，P3 分支作废。**
**P1 不推荐**：它的收益被 `from ... import` 消费风格与 monkeypatch/global 三处冲突吃掉大半，而改为 P3 反而更直白。

#### 6.5.6 需要 owner 回答的是非题（**不得由我放行**）【状态更新 2026-10-26：Q-S1 已答，Q-S2 未触发】

- **Q-S1**：「v1.3 boot audit 校验从『导入即校验』改为『首用才校验』（P1）或『调用点显式调用』（P3），
  是否触及 intake §3 红线『不改变任何冻结判据/封印语义』？」——**已答（board 决策 16，2026-10-26）：「否」**，
  即 P1/P3 均判触红线不放行；WB.2 取消；D3 记「部分满足」；防复发改走「记录方式规范」路线（决策 16/17）。
- **Q-S2**（仅当 W0.2=X2 触发）：「runner 上的 `v1.3 boot audit` 字节与仓库内的记录值不同，
  是否允许按 runner 值重新取样（重新封印）？」——答 **是/否**。**答「否」之前不要动 `extension.py:750–771`。**

#### 6.5.7 后验收口（**2026-10-26 追加**：W0.2 已回传 ⇒ 按 §6.5.5 决策表执行，不是新裁决）

> 派工时 W0.2 尚在办，故 §6.5.1–§6.5.6 以「评估 + 决策表」形态交付。本 ADR 写作期间，
> api-tester 已交付 `08-test.md`（Phase 8-前置）：**M0 达成**（决策 7），左右值已给全。
> 这里只做一件事：**把真实读数代进 §6.5.5 的决策表**，得到执行结论。

**W0.2 实际读数（`08-test.md` §4.1，均在失败 run 自己的 head `29980ee3b` 上取得）**

| 子条件 | 结论 | 左右值差异的**性质** |
|---|---|---|
| ① `boot_audit_identified` | False | recorded 是 `E://Seed//...`，runner 上是 `D://a//Seed//Seed//...` ⇒ **纯路径形态（映射）** |
| ② `rendering_tolerant_match(boot_audit_sha256)` | True | 该文件已被 `.gitattributes` 钉 `eol=lf` |
| ③ `boot_sources_match` | False | 键同样是**绝对路径形态**；值是 **CRLF 时代的工作树 sha256** ⇒ 映射 + 渲染两个次因叠加 |

**代入决策表 ⇒ 落在 X1 与 X4（主）叠加 X3（次），结论：**

| 项 | 结论 |
|---|---|
| **P4（环境面）** | **选中，且已执行**：失败归因为「维护机绝对路径」归族（次轴＝CRLF 渲染未归一）；修复已由并行会话的 `565604a9e` 落地，HEAD `979217e93` 在 LF 干净检出里 import OK、deselect 落空 **0/86**。**不需要 owner 放行**（属表 X1/X4：修的是「同一事实的识别方式」，不是判据内容）。 |
| **P2（结构兜底网）** | **选中**：D-01（`extension.py:678` 的 v1.1 review report 严格字节比较，**在 CRLF 检出的 HEAD 上仍会让 19 模块 collection error**，靠 Windows 腿「autocrlf 步骤顺序」才躲过，属活雷）证明：即使环境面修好，「导入期强校验」仍会在下一次渲染差异出现时让整腿失明。P2 零红线、成本最低、可独立 revert ⇒ **保留为一刀（R-S1d）**。 |
| **P3（结构正刀）** | **已锁死（2026-10-26）**：Q-S1 已决（board 决策 16）owner 答「否」⇒ **不放行，P3 出局**；WB.2 取消，防复发改走「记录方式规范」路线（决策 16/17）。终态交付为 **P4 + P2**，并在 `14-acceptance.md` 里如实写「D3 部分满足」（那 19 个模块的用例已能跑，但「导入期决定可导入性」的耦合仍在）。 |
| **P1（惰性化）** | **维持不推荐**。W0.2 没有改变它在 §6.5.2 的三处硬冲突（`monkeypatch.setattr` 先读旧值 / `global` 重绑定 / 13+9 处裸引用），而收益仍被 `from ... import` 消费风格吃掉。 |
| **P5-a / P5-b（D-01 的处置）** | **不在本 ADR 职权**：已由 TL 决策 9 预判（P5-a 属同族一致化、不是放宽；若 AR 标识需放行则由 TL 以是非题回 owner）。**我的标识意见**：P5-a 与 §6.5.5 X3 同源——「纳入**既有** `rendering_tolerant_match` 口径」属同族一致化可接受；**「新增更宽的容忍」不可接受**（那是把判据改松＝红线）。是否放行仍属 owner/TL 裁定，我不放行。 若最后走 P5-b（`.gitattributes`），作用域**必须收敛到受影响的 md 子集**，不得全仓 renormalize。 |
| **执行顺序** | 环境面（已落地）→ **P2（R-S1d）** → D-01 处置（决策 8/9 的面）→ ~~P3 待 owner 放行再动~~ **P3 已随决策 16 出局，此步取消**。 |

**附带要求（给 IMP/AT）**：凡碰治理面的改动，**必须在 LF 与 CRLF 两套干净检出里各跑一次**
（只跑 LF 会漏掉 D-01）；api-tester 已在 `E:/Seed/.workbuddy/tmp/` 留下 5 棵可直接复用的 worktree。

#### 6.5.8 WB 组范围裁决 + D-01（活雷）三候选终裁（2026-10-26 追加，回应 W0.2 后的范围调整）

**（1）WB.1（根因修复刀）：裁定「取消」，不落 commit。**
依据（真实读数，不是推断）：v1.3 位点（`extension.py:750–763`）已由并行会话的 `565604a9e` 修好，
api-tester 在 HEAD（`979217e93`）的 LF 干净检出里实测 import OK、collection 无 error、deselect 落空 **0/86**。
**让 IMP 再做一次已经做完的事，是本次为止最贵的一种浪费（一轮 CI 反馈环 ≈1.5–2.7h）。**
⇒ WB.1 从「会产出 commit 的工作包」降级为**验收动作**：由 AT 在 M3a / M3 里用 W0.4 夹具，
在 **LF 与 CRLF 两套干净检出**各跑一次 `pytest tests/ --co -q` 作为收口证据；
仅当该动作发现残留失败位点时，才启用 `R-S1a`（与 plan v2 §6 同口径：**本刀可能为空**）。

**（2）D-01（v1.1 review report 严格字节比较，`extension.py:678`）：三候选终裁 —— 选 ① P5-a，并同时做 P5-c。**

| 候选 | 裁定 | 裁决依据（实测） |
|---|---|---|
| ① **P5-a**（`678` 改 `rendering_tolerant_match`） | **选定** | 与**同一文件**里已迁移的三个兄弟位点同法：`extension.py:727`、`766`、`793` 都已经用 `seal_ast.rendering_tolerant_match`；`678` 是唯一还用 `_sha256_file(...) != review_sha` 的。该函数**自带契约测试**（`tests/taiji_native/test_seal_ast_hash_invariants.py:169–176`：接受换行渲染变体、**拒绝篡改字节**）⇒ 「容忍渲染 ≠ 放宽判据」。**我不标识红线**（与 TL 决策 9 同判）。 |
| ② **P5-b**（`.gitattributes` 钉渲染） | **否决** | **实测波及面**：现有 `.gitattributes` 是**逐例外最小作用域**（`reports/*.json`、`plans/manifests/n1_fresh_confirmation_v5_20261016.json`、两条 gz/json 例外，每条都附了论据），而 `plans/reference/**/*.md` 是 **580 个文件**（本次 `find` 计数）——scope 从「几条」跃迁到「几百条」。更关键的是该文件里已经存在**反向规则** `reports/taiji_n1_fresh_paired_train_20261016.json -text !eol`（刻意保留 CRLF 首版字节）⇒ 本仓确有「记录值取自 CRLF 工作树」的既有口径，renormalize 会连带改变其它位点的实测字节；且 `git add --renormalize` 产出大 diff，与并行会话（R2 / L5）正面冲突。**成本/风险在中高位，而收益只是把同一个现象从代码侧挪到配置侧，没有多消任何东西。** |
| ③ **P5-c-only**（只靠 Windows 腿步骤顺序维持绿） | **否决** | 「活雷不拆 + 靠 `git config --global core.autocrlf false` 排在 `actions/checkout` 之前」——今天 CI 躲过 D-01 的**唯一**原因就是这个顺序；任何重排步骤 / 换自托管 Windows runner / 带 `autocrlf=true` 的机器都会立刻翻红。与立项「根治」直接矛盾。**但 P5-c 的注释部分应当保留为附加项**（见下）。 |

**附带项（与 P5-a 同一工作包、单独 commit）**：把「为什么 `core.autocrlf false` 必须排在 checkout 之前」写成
`ci.yml` 的注释判据——注意不要用 plan v2 写的 `ci.yml:702–705`，**正确位置是 `ci.yml:757–758`（步骤
`Preserve committed bytes on checkout`）与其后的 `- uses: actions/checkout@v4`（`ci.yml:760`）**，
全文件只有这一处 `autocrlf`（已 grep 复核）。这是零行为改动的判据留痕，属 Phase 13 台账要收的东西。

**（3）验收硬约束（写进 WB.0 / R-S1D）**：任何碰治理面的改动，必须在 **LF 与 CRLF 两套干净检出里各跑一次**
`pytest tests/ --co -q`（复现配方见 `08-test.md` §3.2；红绿夹具已在 `E:/Seed/.workbuddy/tmp/` 下的
`cimirror-head-crlf`（红态）与 `-head-lf`（绿态））。
**只跑 LF 会漏掉 D-01** —— 这不是提示，是验收的前置条件。

#### 6.5.9 「P4-only 是否可接受」——本题正面回答：**不可接受**

- **给的实证**：同一条病灶链已经**复发过一次**——先在 v1.3 位点（绝对路径轴，`29980ee3b`）炸，owner 修完；
  然后在 v1.1 位点（`extension.py:678`，渲染轴）再炸，且 HEAD 上仍活（D-01）。
- **共同结构（一句话）**：把「某台机器的局部外观（绝对路径 / 那一刻的工作树字节）」写成冻结证据，
  **并在导入期用 `!=` 搬回来比**。
- **为什么 P4-only 必然有第三次**：两个因子里，P4 只能改变这一次的「外观形态」（路径 → 渲染 → 下一次的某个新形态）；
  第二个因子「**导入期强校验**」完全没有动。⇒ **拆位点是在赌下一个形态不出现，而实证是它已经出现过一次。**
- **阻止第三次的机制**【已决 2026-10-26】：Q-S1 已决（board 决策 16）owner 答「否」⇒ 走 ②：**只有 P2**（conftest 层把导入期失败转成
  带 reason 的 skip 并显式计数判红），它不能解耦导入性，但能把「整腿失明」降为「局部 skip + 其余全跑 + 显式红」，
  ⇒ **D3 记「部分满足」，在 `14-acceptance.md` 里如实写（决策 16 已确认该口径）；防复发另走「记录方式规范」路线（决策 16/17）。**
- **红线上报口径（是非题）**：见 §6.5.6 **Q-S1**。我只负责标识；**owner 答「否」不等于本题消失**——
  那时由 TL 决定是「接受 D3 部分满足」还是「另开一条 owner 认可的结构路径」。

**⑥ 回滚点**：`R-S1a`（P4 根因刀，pathspec 二选一，**不得同时动 `seal_ast.py` 与 `extension.py`**）、
`R-S1b`（结构刀，P3 落地时才需要；**【已取消 2026-10-26，board 决策 16：P3 不放行】**）、`R-S1c`（失败语义测试）、`R-S1e`（**新增**：P2 的 conftest hook，
pathspec `tests/conftest.py`）。

---

## 7 WA.6 · 提交序列与回滚点确认（13 个回滚点 → 本文确认 14 个）

| 序 | 回滚点 | pathspec（本文确认） | 粒度判定 | 结论 |
|---|---|---|---|---|
| 1 | **R-S2a** | `scripts/training/verify_ci_deselect_list.py` | 单文件，独立可 revert | ✅ 接受 |
| 2 | **R-S3** | `.github/workflows/ci.yml` + `scripts/training/split_ci_junit_faces.py`（ci.yml 里含 pytest 钉版本行） | 两文件同 commit，revert 即删脚本 | ✅ 接受（**新增脚本必须与 yml 同 commit**，否则会出现「脚本在、调用已删」的孤儿态） |
| 3 | **R-S4** | `.github/workflows/ci.yml`（仅 `concurrency` 段） | 单段 | ✅ 接受 |
| 4 | **R-S6a** | `.github/workflows/ci.yml`（4+1 个 Ubuntu step + 2 个 Windows step） | 单文件多段，但**必须单一 commit**（决策 3①） | ✅ 接受，标注 **不可再拆** |
| 5 | **R-S4b** | `.github/workflows/ci.yml`（两处 `timeout-minutes`） | 单文件 | ✅ 接受，但**取值按 §5.3 规则**，不满足则本刀推迟 |
| 6 | **R-S1a** | `scripts/training/seal_ast.py` **或** `scripts/training/taiji_r2_dialogue_continuation_extension.py`（**二选一，禁止同时**） | 单文件 | ✅ 接受（禁止同时 = 归因保 reside） |
| 7 | **R-S1b** | `scripts/training/taiji_r2_dialogue_continuation_extension.py` + 受影响调用点 | 多文件 | ~~接受（依赖 owner 对 Q-S1 的答复）~~ **已取消（2026-10-26）：Q-S1 决策 16 答「否」，P3 不放行，本回滚点随 WB.2 一并取消** |
| 8 | **R-S1c** | 新增/改动的测试文件 | 测试面 | ✅ 接受 |
| 9 | **R-S1e（新增）** | `tests/conftest.py`（P2 的 collection hook） | 单文件，零红线 | ✅ 建议采纳（§6.5.4）；**改名以避开 plan v2 的 `R-S1D`（WB.0 拆雷刀）** |
| 10 | **R-S2b** | `.github/ci/ci-only-deselect.json` + `scripts/training/render_ci_deselect_args.py` + `.github/workflows/ci.yml`（注入 step + 删 env 条目 + 指向注释） | 数据源与 yml **必须同 commit** | ✅ 接受，标注 **不可再拆**（理由见 §4.1.1） |
| 11 | **R-S2c** | 与 **R-S6a 合并** | — | ⚠️ **修订**：plan 把「守卫不再阻断」单列为 R-S2c，但它与 R-S6a 是同一处机制（同一个 continue-on-error 集合）。**取消 R-S2c 独立回滚点，并入 R-S6a**——否则会出现「两个 continue-on-error 叠加语义无法归因」的情形（plan §4 已警告）。 |
| 12 | **R-S2d-1…n** | `.github/ci/ci-only-deselect.json` | 每批一 commit，逐批 revert | ✅ 接受（每批 ≤10 条，R3） |
| 13 | **R-S4c（新增）** | `.github/workflows/ci-fast-lint.yml` | 单文件新增 | ✅ 采纳（ADR-S4 的快通道），独立于 R-S4 可单独 revert |

**序列（决策 4 顺序，本文确认无修改）**：
`R-S2a → R-S3 → R-S4 → R-S4c → R-S4b → R-S6a → R-S1a → R-S1d → R-S1c → R-S2b → R-S2d-1…n`
（【2026-10-26 更新】原序列中的 `[待放行] R-S1b` 已随 board 决策 16 取消，从序列中删除。）

说明：
- R-S4b（timeout 回缩）按 plan §4「WD.1 → WD.2 → WD.4」排在 R-S3 的对照实验之后、**且**必须拿到 ≥3 次实测；
- R-S2b（数据源）排在 R-S2d 之前是硬依赖（先有结构，才能逐条登记重启条件并分批摘）；
- R-S1e（P2）排在 R-S1a 之后：先修根因，再验证 hook 在正常情况下不被触发。

**提交纪律（沿用 plan §6，一条不减）**：每刀一 commit + 显式 pathspec；提交前 `git status --porcelain`
确认 `.zcode/`、`output/playwright/client-ui-20261024/`、`delivery-runs/`、`plans/reference/PLAN-N1-175_*.md`
未被捲入；回滚一律 `git revert`，**禁止 force-push**。

---

## 8 遗留风险

| 编号 | 风险 | 处置 / 去向 |
|---|---|---|
| **R9（新增）** | M2「≥30%」余量只有 2.4pp（32.3%–32.9%）。一旦机型更慢或套件增长，可能跌破 30% | WD.1 落地后**立即**回填实测比值；若 <30%，不得宣称 M2 达成，改由 TL 决定是否追加 Windows 侧提速（D-1） |
| **R10（新增）** | **Windows 腿成为新的关键路径**（1h02m–1h52m），S3/S6对她没有时间收益 ⇒ DoD D4 只能吃到 ~21% 的 run 级降幅 | 登记为技术债 **D-1**：候选 = Windows 腿 xdist / 按目录拆 job / `-m "not slow"` 面收窄。**本次不做**，但对 D4 的终验口径要按「两腿取 max」写进 `14-acceptance.md` |
| **R11（新增）** | junit `classname` 语义依赖 pytest 版本 ⇒ pytest 浮动升级会静默破坏派生 | 已用 `pytest==9.1.1` / `pytest-cov==7.1.0` 钉版本（§4.3）关闭；升级 pytest 时必须先在本机跑 C-4 的合成用例自校验 |
| **R12（新增）** | 去掉 `ci.yml:467/475` 会消失两条 check 名；若分支保护按 check 名配置会失效 | **已关闭（board 决策 10，主理人亲验）：删两条 check 名无外部影响** |
| **R13（新增）** | `$GITHUB_ENV` 假设 A2 未验证 | R-S2b 首个 commit 必须带计数断言（§3.2），失败即退到临时文件方案 |
| **R14（新增）** | `report.xml` 因 `junit_logging=all` 变大 | 64MiB 闸，超限降级为 `log` 并补记 ADR |
| **R4（沿用）** | continue-on-error 让真红变绿 | 已由 §3.6 的三条机制约定 + M5 注入实验关闭；**注入实验未完成 ⇒ R-S6a 视为未交付** |
| **R5（沿用）** | 去重丢面 | 已由三面计数不变式 + WD.2 node-id 集合比对关闭 |
| **R6（沿用）** | 关闭取消后队列积压 | 由 ADR-S4 的执行纪律 + 「queued ≥3 持续 6h ⇒ revert R-S4」阈值关闭 |
| **R7（沿用）** | A1（junit `file` 属性） | **已降级为「已缓解」**：已实证不成立，但 `classname` 路径足够（§2.2/§3.4）；`--co -q` 兜底降为对照实验工具 |
| **R15（新增）** | 决策 8 的 CRLF 收集 step 净增 ~2–3m，会把 M2 的 30% 降幅（余量 2.4pp）吃穿 |
  处置：默认按「去重收益单列计量」；若要求按整腿计量，则把该 step 移入并行快通道（§3.5 ★） |
| **R16（新增）** | 决策 8 写的是 `continue-on-error: false`，与决策 3①（所有门禁 step 不得再让后面被跳过）冲突 |
  处置：本 ADR 裁为并入 S6-i 的 continue-on-error 集合（仍判红）；**已获 TL 追认（board 决策 11：CRLF 收集 step 并入 S6-i 集合，不做 `continue-on-error: false`，手段服从意图）** |
| **R1 / R2 / R3（沿用）** | 本机复现 / 治理面并行会话 / 收紧暴露真红 | 分别流向 api-tester W0.x、owner 放行 Q-S1/Q-S2（**已决：Q-S1=否（决策 16），Q-S2 未触发**）、R-S2d 分批机制。【决策 25 修订，2026-10-26】原「6 条结构性真红」经两次主 CI 的 complete regression 面 junit 实证**全绿**，定性修正为 **native-arch 单独面漂浮红**（与 11_CI_FLOAT_EVIDENCE 吻合）；R3 的暴露形态据此改读，D1（CI 全绿）判据重启，R-S2d 分批机制本身不变 |
| **R8（沿用）** | main 持续推进致行号漂移 | **本阶段已核验到 `979217e93`，关键文件零改动**；每刀开工前按 plan §7 R8 重新核对 |

---

## 9 给下游的注意事项

1. **执行顺序不得按编号走**：`S2a → S3 → S4/S4c → S4b → S6a → S1a/S1d → S2b → S2d-*`（决策 4 + §7）。
   其中 **R-S6a 必须在 R-S3 之后**，因为那时 Ubuntu 腿的闸门形态已变（step25/26 不存在了）——
   **别照抄 `467/475/481/507` 这四个旧行号，照 §1.4 的形态走**。
2. **两个新脚本是本种群可以选择的唯一实现路径**：`render_ci_deselect_args.py` / `split_ci_junit_faces.py`，
   stdlib only、fail-closed、退出码按 §3.8。workflow 里不允许写它们的替代实现。
3. **尾点陷阱**（team-lead 2026-10-26 提出，本文用更硬的方式关闭）：junit 派生**禁止**用字符串前缀划分 `tests.seed`
   与 `tests.seed_platform`（两者是兄弟目录，`tests/seed_platform` 实测 8 文件）；必须用「解析到磁盘真实文件 +
   `Path.is_relative_to()`」，并**必须写一条把 seed_platform 的 classname 喂进去、断言脚本 exit=4 的回归测试**。
4. **汇总 step 只认 `outcome`**：用 `conclusion` 判红 = 把真红变绿（`continue-on-error` 会把 conclusion 改成 success）。
   这是 R4 唯一致命点。
5. **删的同时必须同 commit 补**：五个 `continue-on-error` 与汇总 step 成对出现；数据源与「注入 step + 删 env 条目」成对出现；
   新脚本与调用它的 yml 成对出现。**三者任一被拆成两个 commit 都视为违规**。
6. **不许删 `ci.yml` 注释**：本轮只删 172 行重复条目；`ci.yml:34–105 / 662–665` 的历史文字原地保留，
   追加指向注释即可（§4.1.3）。删注释前先确认它描述的事妈情境仍在。
7. **验证方式红线**（沿用 intake §7.2）：不得「推上去看 CI 变不变绿」。每刀必须有本机可复现的证据；
   S6 的折价实验与 S2 的守卫对照运行除外——它们必须在 CI 上注入一次，**但必须是可逆的单次实验 commit**。
8. **S1 一刀不许动红线**：`extension.py:750–771` 的三个子条件**不得修改**——Q-S1 已决（board 决策 16）owner 答「否」，
   P1/P3 均不放行，该约束从「等待答复」转为**永久约束**；Q-S2 本次未触发（W0.2 未落在 X2）。
   `taiji/organs.py`、`fabric.py`、`model.py` 不触碰。
9. **Phase 6（实现）开工前请 TL 先确认三件事**（只有这三件是我不能自己判的）【状态更新 2026-10-26：三件均已决】：
   ① Q-S1/Q-S2 → owner：**已答（决策 16：Q-S1 否；Q-S2 未触发）**；② 分支保护是否依赖被删的两条 check 名（R12）：
   **已确认无影响（决策 10，主理人亲验）**；③ 接受 13→14 个回滚点中「取消 R-S2c、并入 R-S6a」「新增 R-S1d / R-S4c」两处修订：**已接受（决策 3/序列确认）**。
10. **WD.2 的对照实验必须挑基準 run**：建议 `run 40171918472`（五次成功样本里 step 读数最完整的一条），
   并与该 run 的三张 artifact 做 node-id 集合比对（**不是**数量比对）。

---

## 10 回流判定

- **结论：通过**
- **回流目标**：（无）
- **返工原因**：（无）

说明：本阶段要求裁决的 WA.2–WA.5 与 WA.6 均已给出终裁（含被否选项理由、改动文件清单、判据语义是否变化、
验证方式、回滚点），需求缺口不存在 ⇒ 不回流。

WA.1（ADR-S1）的处理：派工时 W0.2 未回传，故按「评估 + 决策表」交付；写作期间 api-tester 已交付
`08-test.md`（M0 达成、决策 7），**已按 §6.5.5 的决策表就地收口（§6.5.7）**——这是 Table 的设计目的，
不需要重开裁决，也不构成对上游的返工请求。

写作期间还吸收了 board.md 的三条并行输入并已写进契约：① team-lead 自纠的「并集自校验会产生假红」
⇒ ADR-S3 改为**两级**（§3.4 第 1 级前置门禁 exit 5）；② 陷阱 3 的 rootdir ⇒ `--repo-root` 改为必填；
③ 决策 8 的 CRLF 收集 step ⇒ 纳入 C-5/C-6（§3.5 ★），并登记 R15/R16。

仍向上而非回流的三件事（写在 §9.9 与风险表）【状态更新 2026-10-26：三件均已决，无遗留】：
owner 的是非题 **Q-S1**（**已答「否」，board 决策 16**；Q-S2 仅在 X2 触发，本次 W0.2 未落在 X2 ⇒ **Q-S2 本次不触发**）、
TL 追认 **R16**（**已追认，board 决策 11**：并入 S6-i 的 continue-on-error 集合）、
TL/owner 确认 **R12**（**已关闭，board 决策 10**：被删两条 check 名无外部影响）。

---

## 附录 A · R-S2a / R-S3 合成回归用例规格（2026-10-26 追加，供「批次 0」工作包执行）

> 依据 team-lead 2026-10-26 排程裁定：四条用例（本附录 A1–A4）合并为一个工作包，
> 测试文件**单独 commit**（建议 `test(governance): deselect 守卫与 junit 分面的合成回归用例`），
> 不与 ci.yml 改动混刀；回滚点挂 R-S2a / R-S3 名下作补充 commit。
> 文件建议：`tests/test_ci_workflow_governance_regression.py`（单文件，stdlib + pytest only，全量可跑、秒级）。
> 两条「顺带覆盖」用例（A5/A6）与 A1 共用同一 fixture，成本近零，建议同刀带走；不带也不阻塞本工作包验收。

### A.0 共同 fixture（tmp_path 内合成仓库，绝不指向真实仓库树）

```
tmp/repo/
  .github/workflows/ci.yml        # 仅含以 --deselect 开头的条目行（可含一或两个 env 块）
  tests/taiji_native/test_sub_a.py
  tests/seed/test_b.py
  tests/seed_platform/test_sp.py
  tests/taiji_native/test_a.py
```

要点：
1. **守卫（A1/A5/A6）**：`import scripts.training.verify_ci_deselect_list as guard`，
   `monkeypatch.setattr(guard, "REPO", tmp_repo)`、`monkeypatch.setattr(guard, "WORKFLOW", tmp_repo/".github/workflows/ci.yml")`
   （两者是模块级 Path 常量，可安全打补丁；守卫的 `pytest --co` 子进程以 `cwd=REPO` 运行，
   会对 tmp 树真实收集 ⇒ 断言的是真实收集输出，不是 mock）。断言 `guard.main()` 的返回值。
   `GITHUB_STEP_SUMMARY` **可不设**：`write_step_summary()`（verify_ci_deselect_list.py:204–207）
   在该环境变量缺省时静默 return，不影响退出码；仅当需要断言 summary 内容时才
   `monkeypatch.setenv("GITHUB_STEP_SUMMARY", str(tmp_path/"summary.md"))`。
2. **分面（A2/A3/A4）**：argv 驱动，直接调
   `split.main()`（`--source/--native/--seed/--repo-root` 全部指向 tmp 树内文件），或子进程调用；
   合成 report.xml 必须是合法 xunit2（`<testsuites><testsuite name="pytest" …>` 包 `<testcase>`）。
3. 所有合成 `.py` 文件必须**真实存在且可被 pytest 收集**（A2 的归属算法按磁盘文件解析）；
   内容最小化（`def test_x(): assert True`）。

### A1 · C-3 子串匹配 vs 精确 node-id（守卫必须判红）【批准用例①】

| 项 | 规格 |
|---|---|
| fixture | `tests/taiji_native/test_sub_a.py` 只含 **`test_alpha_suffix`**（不存在 `test_alpha`）；ci.yml 条目含 `--deselect tests/taiji_native/test_sub_a.py::test_alpha` |
| 注入的历史缺陷 | 旧判定 `if method in collected.stdout`：`"test_alpha"` 是 `"test_alpha_suffix"` 的子串 ⇒ 旧版**判绿**（MISS 被静默吞掉） |
| 断言 | `guard.main() == 1`；stdout 含 `::error::` 且该行包含 `test_alpha`（MISS 明细可见） |
| 回归价值 | 守住 C-3「node-id 集合精确成员判定、禁 str-in-str」——谁退回子串判定，本用例立刻红 |

### A2 · C-4 尾点陷阱（`tests.seed` 不得吞 `tests.seed_platform`）【批准用例②】

| 项 | 规格 |
|---|---|
| fixture | report.xml 三条：`classname="tests.taiji_native.test_a"/name="test_x"`、`classname="tests.seed.test_b"/name="test_z"`、**`classname="tests.seed_platform.test_sp"/name="test_y"`**（三个 `.py` 在 tmp 树真实存在） |
| 断言（exit 与内容双断言） | ① `split.main() == 0`；② `native-report.xml` 的 `(classname,name)` 集合 == `{(test_a,test_x)}`；③ `seed-report.xml` == `{(test_b,test_z)}`；④ **`test_sp` 在两张派生面里都不出现**（它属 other，与去重前 step25/26 语义一致） |
| 注入的历史缺陷 | 字符串前缀实现（`startswith("tests.seed")`）会把 `test_sp` 吞进 seed 面 ⇒ 断言③④立刻失败——**守恒判不出它**（1+2+0==3 仍成立），所以本用例的价值在内容断言，不在 exit code |
| 回归价值 | 尾点陷阱的疫苗；ADR §3.4「外加条款」的行为级验证 |

### A3 · C-4 第 1 级门禁的两种形态（collection-failure ⇒ exit 5；`<skipped>` ⇒ 排除出面计数）【批准用例③，backend-architect】

| 项 | 规格 |
|---|---|
| fixture（红态） | report.xml 含 `<testcase classname="" name="tests.taiji_native.test_bad"><error message="collection failure"/></testcase>`（`08-test.md` §10 实证形态） |
| 断言（红态） | `split.main() == 5`；stdout/`--native` 前的输出包含模块名 `tests.taiji_native.test_bad` |
| fixture（绿态） | report.xml 含 `<testcase classname="" name="tests.taiji_native.test_skipmod"><skipped/></testcase>`（模块级 `importorskip`/`skip(allow_module_level=True)` 的 junit 形态） |
| 断言（绿态） | `split.main() == 0`；该记录**不进任何派生面**；输出中 `module_skipped` 计数 == 1（单独记账，不进守恒和）——C-4 裁决定版（`917774e70`） |
| 回归价值 | 「凡 `classname=""` 即红」会造成永久假红（基线 run 40171918472 本就有 4 条合法绿跳）；两态各锁一边 |

### A4 · C-4 第 2 级守恒违例 ⇒ exit 4【批准用例④，backend-architect；白盒】

| 项 | 规格 |
|---|---|
| 注入方式 | 白盒：import `split_ci_junit_faces`，monkeypatch 其**归册/分类函数**（实现里的单点 seam；若当前无独立函数，允许最小重构出 `classify(testcase) -> "native"|"seed"|"other"` 一类的纯函数 seam——这本身提升可测性），使其把一条本应归 native 的记录返回 `"drop"`（不进任何面） |
| fixture | report.xml 含 2 条 native + 1 条 seed（全部可解析到磁盘文件） |
| 断言 | 注入后 `split.main() == 4`；输出包含守恒违例信息（uid(native)+uid(seed)+uid(other) ≠ uid(全量)，1+1+0 ≠ 3）；**未注入时同 fixture `main() == 0`**（先跑基线断言再跑注入断言，两段同一 fixture） |
| 回归价值 | 守恒校验自己是活的（exit 4 通路有人守）；同时锁住「module_skipped 不进守恒和」的口径（若有人把 skip 行塞回守恒，A3 绿态 + A4 的分母口径会 jointly 失败） |

### A5（顺带）· NOFILE ⇒ exit 1

ci.yml 条目指向 tmp 树不存在的 `tests/taiji_native/test_ghost.py::test_x` ⇒ `guard.main() == 1`，stdout 含 `::error::` 与 `test_ghost`（㊵-467 原病路径）。与 A1 共 fixture，一条断言的事。

### A6（顺带）· 清单被清空 ⇒ exit 2（fail-closed）

ci.yml 只剩注释（无行首 `--deselect` 行）⇒ `guard.main() == 2`；同理 WORKFLOW 文件不存在 ⇒ `2`。**禁止**「没清单＝全通过」的退化。

### 验收口径（写进该工作包的完成标准）

1. 四条批准用例（A1–A4）全绿 + 两条顺带用例（A5/A6）建议同刀；
2. 测试文件不 import torch/领域模块，单文件总耗时 ≤ 15s（守卫子进程收集 4 个小文件，秒级）；
3. 测试**不得**修改被测脚本（只 monkeypatch 模块常量/纯函数 seam；A4 的 seam 重构除外，且属被测脚本的最小改动，须在 commit message 注明）；
4. commit message 注明「补 R-S2a/R-S3 完成标准之回归用例（ADR 附录 A）」。

### A.7 入库核验（2026-10-26，architecture-reviewer 对已落库回归的符合性复核）

批次 0 的两条批准用例在规格下发前已被 backend-architect 以更高强度落库，规格**已闭合，无需另行实现**。逐条核验（只读，HEAD=7aa52d9cf）：

| 附录规格 | 入库载体 | 核验结论 |
|---|---|---|
| A1 子串→精确 | `53c602667` → `tests/test_verify_ci_deselect_list_contract.py::test_substring_regression_deleted_case_must_be_red` + `test_end_to_end_substring_injection_with_real_pytest_collection` | ✅ 达标。前者锁 judge 三态归属（互为子串用例名删一条 ⇒ MISS+red），后者用真实 `pytest --co` 在 tmp 树端到端双向断言红绿，观测强度不低于规格的 `main()==1` 形态 |
| A2 尾点陷阱 | `7aa52d9cf` → `test_face_of_rejects_seed_platform_sibling_trap` + `test_dot_trap_seed_platform_never_enters_faces_and_conservation_holds` | ✅ 超出规格：除 exit 0 + `seed_platform` 不进任何派生面（内容断言）外，另加 `face_of` 对兄弟目录/tests 根返回 None 的直接断言与唯一 node-id 守恒 2+2+2=6 |
| A3 两形态 | `7aa52d9cf` → `test_collection_error_form_is_exit5_with_module_list`、`test_unknown_empty_classname_form_fails_closed`、`test_module_level_skip_is_excluded_and_counted_not_red` | ✅ 达标。红态 exit 5+模块清单、未知形态 fail-closed exit 5、绿态 exit 0+`module_skipped=2` 单独记账且 skip 记录不混进派生面——与 C-4 定版口径逐字一致 |
| A4 守恒违例 ⇒ exit 4 | `7aa52d9cf` → `test_seed_face_empty_is_exit4`（黑盒）；**后由 `428e86afd` 补全**：退化实现注入红腿（子进程 shim 替换 `face_of`，仓库脚本零改动）⇒ 疫苗通道 exit 4 + 判据原文断言 | ✅ **全量闭合**。三通道口径已入测试 docstring：疫苗通道「面混入非对应目录文件」/ 空面通道「某个面被悄悄吞掉」/ 三面之和不等通道**从 CLI 输入面不可构造**（正常归属是 testcase 划分，守恒由构造保证；能造成它的实现退化会先被疫苗拦截）——规格原判（白盒 seam 注入）被以更强的「真实脚本加载 + 零仓库改动」shim 落地，两份契约测试合计 25/25 |
| A5 NOFILE ⇒ 红 | `53c602667` → `test_nofile_is_red` | ✅ judge 级断言，红线语义相同 |
| A6 清空 ⇒ exit 2 | 未见专项用例 | ⚠️ 残留（可选项）。规格本身标注「不带不阻塞验收」；fail-closed 语义已由 `7aa52d9cf::test_missing_source_is_exit2`（分面脚本侧）部分佐证，守卫侧留作后续顺带补充 |

补充说明：`7aa52d9cf` 为独立 pathspec 补充 commit（挂 R-S3 名下，仅新增测试文件，revert 不影响 `917774e70`），符合 §A「不与主刀混 commit」的落库约束；其 docstring 的判据口径与 08-test.md §12.0 / ADR §3.4 三处一致（已抽查）。批次 0 工作包据此**关闭**，无需再排。

---

## 附录 B · R-S6a 契约符合性审读记录（`bc019e517`，2026-10-26，architecture-reviewer）

审读对象：`bc019e517`（ci.yml 单文件 +139/−4，本地 HEAD 顶端）。口径：§3.5 C-5 / §3.6 C-6 / §6.4 + team-lead 派单四项。方式：先通读 diff，再对 HEAD 成品文件独立核验（continue-on-error 全量分布 grep + 逐行读 run 块），不依赖提交者自述。

### B.1 逐项核验结果

| # | 契约项 | 结论 | 证据 |
|---|---|---|---|
| 1 | 五 step 集合完整性 | ✅ | Ubuntu 腿四门禁（deselect-guard/pytest-full/junit-faces/coverage-gate 均 coe:true）+ gate-summary；Windows 腿 pytest-windows + collection-crlf + gate-summary-windows。与 C-5 表逐行对应（腿归属差异见 B.3-③，合规） |
| 2 | 只认 `outcome` 不认 `conclusion` | ✅ | 两腿汇总 env 全部取 `steps.<id>.outcome`；run 块与注释均无 conclusion 引用 |
| 3 | fail-closed（skipped/cancelled/空值/未知） | ✅ | case 表：failure/cancelled/skipped ⇒ RED；`*` ⇒ RED(unknown) 且空串渲染 `<empty>`（`set -u` 下 env 已定义、空值进 `*` 分支） |
| 4 | 汇总步自身禁 coe + `if: always()` | ✅ | 两个汇总步均无 continue-on-error（全文件 coe 共 8 处：advisory×2 + Ubuntu×4 + Windows×2，汇总不在其中）；均 `if: always()` |
| 5 | §6.4 序列 / id↔env 绑定 | ✅ | OUT_GUARD←deselect-guard(470)、OUT_PYTEST←pytest-full(486)、OUT_FACES←junit-faces(507)、OUT_COVER←coverage-gate(532)；Windows OUT_PYTEST←pytest-windows(878)、OUT_COLLECT←collection-crlf(900)。id 全部真实存在 ⇒ 无「永远红」死局；advisory 两步（mypy legacy debt / pip-audit）历史在册、排除在门禁集合外合规 |
| 6 | Windows 探针按决策 14 定位 | ✅（附 ADR 文本同步，B.3-③） | collection-crlf 在 Windows 腿（autocrlf=true 本地克隆 + `pytest tests/ --co -q` 不跑用例），渲染轴与平台轴分离；失败模块清单落 step summary（决策 11） |
| 7 | 防「真红变绿」终检 | ✅（机械层面） | 任意门禁 outcome≠success ⇒ gate-summary `exit 1` ⇒ job failure；全 success ⇒ exit 0。上游非门禁步失败时门禁步 outcome=skipped ⇒ RED，方向正确（基础设施坏 ⇒ 红） |
| 8 | 判据语义不变 / 冻结注释 / 单 commit | ✅ | 覆盖率阈值、artifact 名、冻结注释（34–105/662–665）不在 diff hunk 内；单文件单 commit 符合决策 3① |

### B.2 返工点清单

**R-P1（必修，探针 fail-open）**：`collection-crlf` 的 run 块只统计 `ERROR collecting` 行——管道
`python -m pytest … | tee crlf_collect.log` 之后的 `PIPESTATUS[0]` 被丢弃（`set -o pipefail` 只影响管道返回值，
无 `set -e`，后续无人检查）。若 pytest 以**无任何 `ERROR collecting` 行的方式**非零退出（usage error=4、
internal error=3、no tests collected=5、解释器崩溃），`count=0` ⇒ **探针自身故障被静默判绿**。
这正是 S6 要根治的「静默失效」家族在探针体内的复发，违反 §0 硬约束 1（fail-closed）的精神与
C-5 第 5 行「探针红 ⇒ 汇总红」的可观测前提。**修法（一行）**：tee 之后立即 `rc=${PIPESTATUS[0]}`，
在 `count` 判定前加 `[ "$rc" -ne 0 ] && [ "$count" -eq 0 ] ⇒ ::error::探针异常 + exit 1`
（rc=1 且 count>0 走既有清单路径；rc=1 且 count=0 同样是自相矛盾 ⇒ 红）。

**R-P2（顺带，不阻塞）**：探针用 `git config --global core.autocrlf true → clone → 恢复 false` 改写全局配置，
clone 若中途失败全局态滞留（本腿后续无 git 消费步，风险低）。建议改 `git -c core.autocrlf=true clone --quiet . "$probe"`，
零全局态。可与 R-P1 同 commit（同为 collection-crlf run 块内的一行改动，属 R-S6a 补充 commit，独立 pathspec 不可再拆无涉）。

### B.3 文本同步（我方职责，已随本记录完成）

- ① §A.7 A4 行升级（`428e86afd` 全量闭合）——本日早前已完成；
- ② §5.1 torch 收口注（`321e66a54`）——已完成；
- ③ **决策 14 腿归属勘误**：C-5 表第 5 行与 C-6 yaml 原文把 collection-crlf 写在 Ubuntu 腿（决策 8 时点），
  决策 14 已移 Windows 腿。已在 §3.5/§3.6 各追加勘误注（不改历史文本），落地实现按决策 14 执行**视为合规**。

### B.4 结论

**返工点 1 条（R-P1，必修；R-P2 顺带）**，其余全部符合契约。R-P1/R-P2 修复后（一个 R-S6a 名下的补充 commit），
**即可放行 M5 注入实验与 skipped 注入实验**（可逆单次实验 commit，push 放行 + 基线 run 出结论后按既定方案报批执行）；
实验完成前 R-S6a 按 §8 R4 条款维持「未交付」开口不变。

### B.5 追记：diff 级复核通过，放行注入实验（2026-10-26）

复核对象：`8b8161300`（R-P1/R-P2 修复，pathspec 仅 ci.yml）+ `cfe892570`（A6 顺带项闭合，pathspec 仅测试文件）。复核人 team-lead 亲自落地（backend-architect 名下任务收回，与 mypy patch 工作树串行）。

| 项 | 核验结论 |
|---|---|
| R-P1 判定分支 | ✅ `rc=${PIPESTATUS[0]}` 紧贴 tee 管道（中间仅注释，PIPESTATUS 未被冲掉；该 step `shell: bash`，变量有效）。分支矩阵逐格核验：rc=0/count=0 ⇒ 两个 if 均不触发，step 以 0 收口（绿）；rc=1/count>0 ⇒ 走既有模块清单路径（真发现红，不被误伤）；rc=1/count=0 ⇒ 自相矛盾态，fail-closed 判红；rc=2/3/4/5/count=0 ⇒ 探针故障红，日志尾段 40 行落 summary 供归因。与 §B.2 R-P1 修法逐字一致 |
| R-P2 语义 | ✅ `git -c core.autocrlf=true clone --quiet . "$probe"`：-c 传入的 config 在 clone 初始化时写入**新克隆仓库**的 config，渲染转换发生在 clone 内的 checkout——runner 全局态零污染。与返工意图一致 |
| A6 闭合（cfe892570） | ✅ 两个 return-2 通道各触发一次：① WORKFLOW 指向不存在路径（数据源不可读）；② 文件合法存在但零 `--deselect` 条目（含 `::error::` 与判据原文输出断言）。与附录 A.6 规格一致，两份契约测试 26/26 |
| 留痕核验 | `b04e7607d` 在仓不可达（unknown revision）——先前的 A6 申报指向不存在的 commit，属虚构申报（board 决策 22 已记）；实际闭合以 `cfe892570` 为准。演练脚本 `rp1_drill.sh` / `gate_summary_drill.sh` 均在 `.workbuddy/tmp/` 实证存在 |
| commit 形态 | ✅ 两刀各自独立 pathspec（ci.yml / 测试文件），R-S6a / R-S2a 名下补充 commit，回滚互不纠缠 |

**终版结论：`8b8161300` + `cfe892570` 复核通过，M5 注入实验与 skipped 注入实验正式放行**——
按既定口径执行（可逆单次实验 commit + push 放行 + 基线 run 出结论后报批读数）；实验读数回填并复核通过后，
R-S6a 方可从「未交付」开口（§8 R4）转「已交付」。本审读阶段（附录 B 全部事项）就此关闭。

### B.6 追记：R-S6a 实战验收读数登记 + 决策 25 定性修订的 ADR 影响声明（2026-10-26）

**R-S6a 实战验收（主 CI run 40258536618，基线 9da09708a）——全套达成**（board 决策 25，team-lead 亲验读数，我按读数复核定性）：

- **Ubuntu 腿（test 3.12，1h46m）**：pytest-full 原始 outcome=failure（2×N1-167 timeout）被 coe 吞成显示绿，
  gate-summary 正确抓到 RED（3 ok + 1 RED ⇒ exit 1 ⇒ job 红）——**outcome 抓真红在实战首次实证**；
  derive faces 成功（uid 3957+185+516=4658 守恒、module_skipped=4、coverage 65.25 与三次基线一致）；
  后续 step 无一被跳过（S6 立项目标「失败时反而有产物可供取证」兑现）。
- **Windows 腿（2h2m）**：pytest-windows 同样吞绿抓红；collection-crlf 探针红 = **真发现分支**（D-01 族 19 模块
  CRLF collection error，清单落 summary）；**R-P1 的 fail-closed 分支未误触发**——真发现/探针故障两档行为分界精确。
- **快通道**：lint-fast 首跑 success 3m08s（≤6m 达标，时延已回填 §5.1；双冷场景无样本，观察项保留）。

**R4 开口维持**：实战验收 ≠ M5/skipped 注入实验——实战只证明了「红被抓」，注入实验要证明的是
「被吞的红不可能漏抓」（failure/skipped/空值三通道的机械保证）。brief 起草中（基线锚定 40258536618），
按双人约定我复核后主理人执行；读数复核通过前，R-S6a 在 §8 R4 口径下**仍记「未交付」开口**。

**决策 25 定性修订的影响声明**：「6 条结构性真红」修正为「native-arch 单独面漂浮红」（成长门族在两次主 CI
complete regression 面 junit 全绿），D1（CI 全绿）判据重启。对本 ADR 的牵连核验结果：
- **我的契约零依赖**：C-1…C-7 无任何条目以「主 CI 必红」为承重前提；R-S2d 分批机制不因此失效；
- **受影响的是预期管理文本**：`bc019e517` commit message 中「探针不引入额外伤害（job 本就因结构性 6 条失败而红）」
  的前提已随决策 25 过期——修订后的预期：主 CI 红位 = pytest-full timeout（N1-167）+ 探针 D-01 真红 +
  漂浮红扇面（native-arch 单独面），三者相互独立、均可归因；
- **D-01 处置权重上调**：探针实证 19 模块 CRLF collection error（R-S1a/P4 族修复的验收面），原 §6.5.8 的
  「探针每次红 ⇒ R-S1a 落地后回绿」预期不变，且现在多了两次主 CI 的 junit 面可作 R-S1a 前后对照。

### B.7 追记：R-S1a 拆雷核验（`571431884`，2026-10-26）

核验对象：D-01 拆雷 commit（单文件 +11/−1）。**结论：三点全部通过，R-S1a 可进入「在飞 run 回绿」确认阶段。**

**① 同构性 ✅**：`_validate_prior_v1_1` 内 678 行（改后 686）由
`_sha256_file(_SOURCE_REVISION_V1_1_REVIEW) != review_sha` 改为
`not seal_ast.rendering_tolerant_match(review_sha, ...read_bytes())`——与 v1.2/v1.3 三个兄弟位点
（改后行号 737/776/803）**逐字同构**（同一函数、同一参数序、同一 `or not` 链位）；函数内局部
`from scripts.training import seal_ast` 与 613/657/700 行既有式一致。`rendering_tolerant_match`
（seal_ast.py:277）是**既有历史通道**（docstring 自证「复用历史通道同一妥协」；三种换行渲染任一匹配即可，
真实内容变化任一渲染下必失配）——非新增放宽，与我 2026-10-25 对 P5-a 的裁定（同族一致化、不触红线）一致，
与决策 9 的 owner 预判吻合。**判据语义变化 = 把该位点纳入同族既有容忍通道，冻结对象（v1.1 review report
的「内容」）不变，冻结的「机器局部外观」（当时工作树字节渲染）按病灶共同结构修正**。
位点勘误属实：`extension.py:678` 为台账缩略转写，真实路径
`scripts/training/taiji_r2_dialogue_continuation_extension.py`，本 ADR 沿用缩写处一并以此为准。

**② 封印零级联独立复核 ✅**（不只信提交方扫描，我方独立 grep 全量）：
- `plans/*.json`：对该文件名**零引用**（含 sha/hash/guard 关键词交叉过滤后为零）；
- `reports/`：20 个 attempt/endpoints/readiness json 含源哈希快照，但均为**写入产物**——
  无任何脚本读回并与冻结值比对（scripts/ 内对该文件名的全部引用 = import 若干 + 两处路径清单）；
- 代码侧两处路径清单逐一开膛：
  `run_taiji_r2_resume.py:26 SOURCES`（:239 消费）= 同 run 内快照**写入**（uuid 命名新 manifest，无冻结比对）；
  `taiji_r2_continuation_evaluation_adapter.py:40 EXTENSION_SOURCE_PATHS`（:81 消费）= readiness/attempt
  的 `source_hashes` **同 run 内**一致性检测（readiness 时算一次、worker mid-run 复算比对，同检出同渲染，
  与 CRLF 轴无关）；`validate_taiji_r2_continuation_endpoint.py` 的比对全部是**运行时记录互证**
  （attempt↔lineage↔result），不冻结 .py 字节做跨 run 比对。
  ⇒ **唯一的跨 run 字节冻结守卫就是被拆的 678 本尊**，零级联判定成立。

**③ brief 基线改锚 ✅ 同意**：40288176554（571431884，在飞已实证 in_progress）出结论后，
D-01 修复 + 探针预期回绿 + 剩余红位仅 N1-167 ×2 ⇒ 注入实验的 job 红位可完全归因于注入本身，
比 40258536618（探针真红在册）更干净。六条复核清单不变。

**验收锚登记**：改前锚 = 决策 26（4658 守恒 + 探针 19 模块红）；改后锚待 40288176554 的 junit 面
（预期：CRLF 轴 taiji_native 4054 collected/0 error ⇒ 探针 count=0 回绿；守恒读数应 ≈4658±漂浮红扇面增量，
api-tester 第三源 19/19 PASS 已预验 SET-EQUAL）。两锚齐即 D-01 关闭，WB 面收口。
