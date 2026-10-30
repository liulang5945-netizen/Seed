# Phase 6 · 第一批实现（backend-architect · R-S2a / R-S3）

## 结论

两刀均完成，各自独立可 revert（一刀一 commit + 显式 pathspec）：

| 回滚点 | commit | pathspec |
|---|---|---|
| R-S2a | `53c602667` | `scripts/training/verify_ci_deselect_list.py` + `tests/test_verify_ci_deselect_list_contract.py`（主理人对 ADR §7 单文件粒度的扩展裁量） |
| R-S3 | `917774e70`（含主理人 C-4 契约修正的 amend） | `.github/workflows/ci.yml` + `scripts/training/split_ci_junit_faces.py`（必须同 commit，否则孤儿态） |
| R-S3 补充 | `7aa52d9cf` | `tests/test_split_ci_junit_faces_contract.py`（ADR §6.2 ⑤：尾点陷阱/守恒/两形态空 classname 合成回归 11 例，独立 pathspec 挂 R-S3 名下） |
| 批次 0 | `428e86afd` | `tests/test_split_ci_junit_faces_contract.py` + `tests/test_verify_ci_deselect_list_contract.py`（AR 批次 0 规格：尾点陷阱红绿双向 shim / 真前缀边界 / collection-failure / exit-4 通道判据原文区分；25/25 通过） |
| R-S4 | `8fb6e35f7` | `.github/workflows/ci.yml`（仅 concurrency 块：push→false / PR→true；回退触发条件入注释） |
| R-S4c | `321e66a54` | `.github/workflows/ci-fast-lint.yml`（新增 ≤6m 快通道；torch 必装实测收口 ADR §5.1 问号） |
| R-S6a | `bc019e517` | `.github/workflows/ci.yml`（Ubuntu 4 门禁 step continue-on-error + gate-summary；Windows pytest-windows + collection-crlf 探针 + Windows 汇总；单一 commit） |

**push 状态（2026-10-26）**：以上四刀均**本地落地未 push**（在飞 run 40226939396 保护禁令）；
放行后 push 顺序 = 批次 0 → R-S4 → R-S4c → R-S6a。

黑板上已追加「Phase 6 · 第一批实现」小节；未动「当前阶段」指针。

## 接口实现清单

### R-S2a · 守卫改写（C-3）

- **数据源（过渡态）**：仍扫 `ci.yml`（`listed_entries()` 保留），两腿 env 各 86 条 ⇒ 去重后
  **86 条 / 56 个文件**（86 才是清单条数；172 不是）。JSON 单数据源归 R-S2b。
- **收集方式**：废掉「每条一个 pytest --co 子进程」（旧 172 次，5m19s–6m24s），改为**一次**
  `python -m pytest --co -q -p no:cacheprovider --continue-on-collection-errors <56 files>`，
  文件路径排序固定顺序；**不**全量收集 tests/。
- **判定（三态，取代子串包含）**：`--co -q` 输出解析成 node-id 集合（判据：含 `::` 且首段以
  `.py` 结尾的行，排除 traceback 回显/汇总行），`entry in collected_nodes` **精确成员判定**
  （含 `::Class::` 与参数化 `[...]` 整体匹配），无任何 `str in str`。
  - MISS（红）：node ∉ 集合 且文件收集成功；
  - UNCOLLECTABLE（黄，`::warning::` 不判红）：所在文件本次收集产出 collection error
    （解析 `ERROR collecting <file>` 头与 short summary `ERROR <file>` 两形态）；
  - NOFILE（红）：文件在磁盘上不存在；条目缺 `::` 按腐化判 MISS（红）。
- **stdout**：`deselect 清单 86 条（来自 .github/workflows/ci.yml）；MISS a；NOFILE b；UNCOLLECTABLE c`
  + 每条非命中一行明细；`$GITHUB_STEP_SUMMARY` 写 markdown 表（契约允许的例外）。
- **退出码**：0 全命中；1 存在 MISS/NOFILE；2 数据源不可读/清单为空（fail-closed，不许退化为
  「没清单＝全通过」）；3 pytest 子进程无法启动（OSError / No module named pytest）。
- 契约测试 `tests/test_verify_ci_deselect_list_contract.py`（11 例）：C-3 硬要求的回归核心
  ——互为子串的两用例名删掉其一 ⇒ 新守卫判红（旧版误判绿），含真实 pytest --co 的端到端注入
  实验；另有参数化后缀/Class 链精确匹配、UNCOLLECTABLE 不判红、NOFILE 红、172≠86 去重口径、
  错误行解析等。

### R-S3 · Ubuntu 腿去重 + junit 派生（C-4/C-5 + C-4 修正）

- ci.yml：删 step25（`Run native architecture contracts`）与 step26（`Run Seed model-boundary
  contracts`）；pytest-full 加 `id` + `-o junit_logging=all`（**不加 continue-on-error**，
  衔接裁定见下）；新增 `Derive native/seed junit faces`（id: junit-faces）跑在 pytest-full 之后、
  artifact 上传之前；两腿 `Install test dependencies` 钉 `pytest==9.1.1` / `pytest-cov==7.1.0`
  （run 40203419431 实测读数，R11）。`ci.yml:34–105` 与 `662–665` 注释一个字未动；artifact 名
  与覆盖率面（--cov / fail_under=60 / --threshold 60.0 / trend summary）完全不动。
- `split_ci_junit_faces.py`（stdlib only / fail-closed / 幂等）：
  - argv：`--source --native --seed --repo-root` 全必填显式（rootdir 钉死，陷阱 3）；
  - 归属：`key = classname or name` → `parts=key.split('.')` 循环拼 `"/".join(parts)+".py"`
    直到 `(repo_root/cand).is_file()`，耗尽 ⇒ exit 3；归属用 `Path.is_relative_to()`
    （禁字符串前缀，尾点陷阱结构性关闭）；
  - `tests/taiji_native/` ⇒ native 面；`tests/seed/` ⇒ seed 面；`tests/seed_platform/`、
    `tests/resonance/`、`tests/` 根 ⇒ other（与去重前 step25/26 语义一致）；
  - 派生：深拷贝结构、只筛 `<testcase>`、重算 testsuite/testsuites 的
    tests/failures/errors/skipped/time（缺字段补 0）；头注释
    `<!-- derived from report.xml by split_ci_junit_faces.py; not a separate pytest run -->`；
  - **两级自校验顺序不可颠倒**（判据按主理人 C-4 契约修正，08-test.md §12.0 两行表）：
    - 第 1 级：空 classname 判**子元素**——`<error message="collection failure">`/`<failure>`/
      未知形态 ⇒ exit 5 + 模块清单；`<skipped>` ⇒ 合法模块级 skip（importorskip /
      allow_module_level）⇒ **排除出面 + 单独计数 module_skipped，不判红**；
    - 第 2 级：分母**不用** pytest 的 `tests=` 属性（它把模块级 skip 行也计入），用
      「唯一 node-id 数（classname+name 去重）」守恒：uid(native)+uid(seed)+uid(other)==uid(全量)
      且两面非空 ⇒ 违反 exit 4；module_skipped 单独记账打印；
    - 外加「面上文件必须真落对应目录」疫苗；
  - 退出码 0/2/3/4/5 按 ADR §3.8。

## 数据模型与迁移

- 迁移完全按 ADR §4.2 的 M-3（守卫改写可单独先落）与 §4.2 S3 数据模型（三面派生）执行；
  本批**不涉及** M-0/M-1/M-2（JSON 数据源 = R-S2b）与 M-4（分批收紧 = R-S2d）。
- 数据面新增 0 张表/0 个数据文件；派生面是纯函数（同输入 ⇒ 字节级同输出，已验证）。
- 回滚：`git revert 917774e70` 即回到三趟 pytest 形态（脚本与 yml 同生共死）；
  `git revert 53c602667` 回到旧守卫（子串判定 + 172 子进程）。

## 鉴权与错误码

- 两脚本均 fail-closed、退出码按 ADR §3.8 总表；密钥面无涉及（脚本只读仓库文件与 junit）。
- ci.yml 变更未引入任何 secrets/权限面变化。

## 自测结果

| 项 | 读数 |
|---|---|
| 新守卫（HEAD `4b783dcec` 干净 LF worktree `cimirror-head-lf`） | 86/86 命中，MISS 0，NOFILE 0，UNCOLLECTABLE 0，exit 0，**20.3s**（≤2m 验收线通过） |
| 守卫红态注入 | 注入不存在文件 ⇒ NOFILE 1，exit 1；pytest 对不存在文件参数 usage-error 无收集 ⇒ 全列判红 exit 1（fail-closed） |
| 守卫契约测试 | 11/11 passed（pytest 9.1.1 / Python 3.12.10） |
| 守卫与旧版一致性 | 一致：86/86、0 落空（旧版 CI 读数 run 40207103896 绿；api-tester 实测 0/86） |
| 派生脚本合成样本（空 classname 两形态各一条） | ① error 形态 ⇒ exit 5 + 模块清单；② skipped 形态 ⇒ 排除出面、module_skipped=2 单独计数、exit 0；③ seed_platform/根模块 ⇒ 不进两面、唯一 node-id 守恒 6=2+2+2、exit 0；④ 幂等字节级一致；⑤ seed 面空 ⇒ exit 4；⑥ UNRESOLVED ⇒ exit 3 |
| **真实基线（WD.2，同 run 锚：run 40171918472，head `499c3892f`）** | 基线 report.xml（4580 行）⇒ exit 0，uid(native)=**3901** / uid(seed)=**185** / uid(other)=**490**，守恒 3901+185+490=**4576**，module_skipped=**4**（清单与 api-tester §12 逐条一致） |
| **派生 vs 同 run 独立面（验收锚）** | 唯一 (classname,name) 集合**双向 diff=0**：native 3901/3901、seed 185/185，only_in_base=0 且 only_in_derived=0（逐 id 相等，非数量相近） |
| 基线幂等 | 二次运行 cmp 字节一致 |
| 覆盖率不变式 | core_line_pct 锚 65.25（threshold 60.0）；本机无法复现 CI --cov 环境，**如实记「待 CI 对照」**；容差 ±0.05pp 只适用 core_line_pct（三 run 全等），全局 line-rate 跨 run 波动 0.06pp 不作趋势门；--cov 参数未动 ⇒ 分母不变 |
| YAML 校验 | `yaml.safe_load` 通过；`git diff --stat` 确认只动预期 3 个 hunk |

## 遗留风险

1. ~~派生脚本的合成回归用例本批未入库~~ **已闭环**：`7aa52d9cf` 补入
   `tests/test_split_ci_junit_faces_contract.py`（11 例，覆盖尾点陷阱/两形态空
   classname/守恒/exit 2-5/幂等/计数重算/参数化 id 不吞并），两份契约测试合计 22/22 通过。
2. 覆盖率读数待 CI 对照（上表；容差口径按主理人澄清只适用 core_line_pct）。
3. report.xml 因 `-o junit_logging=all` 变大，64MiB 闸待首个 CI run 读数。
4. pytest 对不存在的文件参数会 usage-error（无收集输出）⇒ 守卫全列 MISS 判红——方向正确
   （fail-closed），但报错面偏大；NOFILE 检查会同时给出准确定位。
5. main 在本批执行期间又推进（`4eda281b3` N175 research commit 卷入历史）；两刀均为 pathspec
   commit 且 R-S3 amend 前已确认未推送远端，未受影响；后续刀开工前照例重跑 R8 核对。

## 给下游的注意事项

1. **R-S4**：concurrency 段本批未动，按 ADR §3.7 直接落。
2. **R-S6a**：pytest-full / junit-faces 加 `continue-on-error` 必须与 gate-summary 同 commit
   （本批已在 yml 两处注释写明裁定与理由）；届时 step id 用 `deselect-guard` /
   `pytest-full` / `junit-faces` / `coverage-gate` / `collection-crlf`。
3. 守卫的 node-id 行判据（首段以 `.py` 结尾）依赖 pytest `--co -q` 输出形态；pytest 升级前
   先跑 `tests/test_verify_ci_deselect_list_contract.py` 与派生脚本自校验（两处均已钉版本兜底）。
4. 基线取证环境可直接复用：worktree `E:/Seed/.workbuddy/tmp/cimirror-baseline-499`
   （= 基线 head `499c3892f`）、artifact 目录 `E:/Seed/.workbuddy/tmp/baseline-40171918472/`、
   合成夹具 `E:/Seed/.workbuddy/tmp/rs3_synthetic/`。
5. 提交纪律遵守情况：两刀均 `git commit -F <msg> -- <paths>`（显式 pathspec，never bare）；
   R-S3 的 C-4 修正以 amend 落地（amend 前核实 `899198273` 未推送远端，保持一刀一 commit）；
   `.zcode/`、`output/playwright/client-ui-20261024/`、`delivery-runs/`、
   `plans/reference/PLAN-N1-175_*.md` 未被卷入；未 force-push。

## 第二批（批次 0 / R-S4 / R-S4c / R-S6a）实现记录

### 批次 0 · 合成回归用例补全（`428e86afd`，纯测试文件）

AR 规格四条，全部落在两份既有契约测试文件（与 `tests/test_verify_ci_deselect_list_contract.py`
同族），25/25 通过：

| # | 规格 | 落点与读数 |
|---|---|---|
| ① | C-4 尾点陷阱红绿双向 | split 契约测试 `test_dot_trap_red_green_degenerate_prefix_impl_must_exit4`：绿腿（现行实现）exit 0 且 seed_platform 不进两面；红腿 = 子进程 shim 注入**退化实现**（不带尾点字符串前缀）⇒ 疫苗 exit 4，断言判据原文"面混入非对应目录文件"。shim 在 tmp_path 加载真实脚本模块替换 face_of，仓库脚本零改动 |
| ② | C-3 精确匹配边界 | 守卫契约测试 `test_true_prefix_entry_is_miss_while_old_substring_logic_would_pass`：条目 `::test_a` vs 收集 `::test_ab`（真前缀）；用例内同时断言旧子串判定式为真（旧版误判绿的本体）与新守卫 MISS 红 |
| ③ | collection-failure 形态 | split 契约测试 `test_collection_error_fixture_is_exit5_with_module_list`（跨 taiji_native/seed 两目录）+ 既有同款用例 ⇒ exit 5 + 模块清单 |
| ④ | 守恒违例形态 | exit-4 三子通道按判据原文区分：疫苗通道（①红腿触发"面混入非对应目录文件"）、空面通道（`test_seed_face_empty_is_exit4` 断言"某个面被悄悄吞掉"+uid(seed)=0）；**三面之和不等通道从 CLI 输入面不可构造**（分桶是划分、守恒由构造保证；能造成它的实现退化先被疫苗拦截）——口径已写进测试 docstring |

### 批次 1 · R-S4 并发分化（`8fb6e35f7`）

`cancel-in-progress: ${{ github.event_name == 'pull_request' }}`，group 键不变；
表达式只用 github.* 上下文。回退触发条件（main queued≥3 持续 6h ⇒ revert）写进注释。
YAML 校验过。**表达式行为（PR 连推取消 / main 连推排队）待 push 后取证**（本机无法
模拟 GitHub 事件面，如实登记）。

### 批次 2 · R-S4c 快通道（`321e66a54`）

`.github/workflows/ci-fast-lint.yml`：lint-fast job（ruff / B,SIM 棘轮 baseline 0 /
black / frozen-marker audit / mypy core 棘轮 baseline 0），运行块与 ci.yml 逐字一致；
group `lint-${{ github.ref }}`（与主腿互不取消）；cancel-in-progress: true；
timeout 10m（目标 ≤6m）。**实测收口 ADR §5.1 的"torch? 不需要 torch"问号**：
mypy core 棘轮缺 torch 会因 import 解析失败假红 ⇒ 安装面必须与 ci.yml 对齐
（torch==2.13.0+cpu + 同钉版本工具链，仅去 pytest/pytest-cov/pip-audit 三个本通道
不消费项）。≤6m 与 4 check 读数一致比对待 CI 取证。

### 批次 3 · R-S6a（`bc019e517`，单一 commit 不可再拆）

- Ubuntu：deselect-guard / pytest-full / junit-faces / coverage-gate 挂
  continue-on-error 并补 id；末置 gate-summary（if: always()）按四条 outcome 收口，
  表格打 stdout + $GITHUB_STEP_SUMMARY。
- Windows：pytest-windows（id + continue-on-error）+ collection-crlf 探针
  （autocrlf=true 本地克隆 + `pytest tests/ --co -q` 只收集；失败模块清单落 summary，
  决策 11；对 collection error 显式 exit 1 ⇒ outcome=failure）+ gate-summary-windows
  （判 pytest-windows + collection-crlf）。
- 四条机制约定：只认 outcome 不认 conclusion；skipped 判红；空值/未知判红；
  汇总自身禁止 continue-on-error（接线核验：两腿汇总 step 均无 continue-on-error，
  Ubuntu coe 集合=4 门禁+2 既有 advisory，Windows coe=2）。
- 本地红绿演练 **6/6**（`E:/Seed/.workbuddy/tmp/gate_summary_drill.sh`，run 块原样抽出）：
  全 success⇒0；failure/skipped/cancelled/空值⇒1。
- 预期行为声明：D-01 未修前 collection-crlf 探针每次 run 都红（结构性 6 条失败本就
  使 job 红，无额外伤害；R-S1a 落地后回绿）。
- **未完成开口（按 ADR §8 R4 条款如实登记）**：M5 注入实验（注入红 ⇒ 后续 step 仍
  执行且 job conclusion=failure）与 skipped 注入实验需 CI 取证，push 放行后以
  可逆单次实验 commit 执行；完成前 R-S6a 按 ADR 记"未交付"。

### R-S4b 预研登记（不做实现）

timeout 回缩需 ≥3 次落地后两腿全绿读数（ADR §5.3：`max(ceil(P95×1.3), 120m)`，
推荐 150m、更小不接受）；run 40226939396 是 R-S3 后新形态第 1 个数据点。
待读数，不动 timeout 值。

### B-mypy · `taiji/conditional_language_v3.py` 12 条错误修复草案（**未 commit**，patch 交付）

> 落地说明：本节由 backend-architect 补记（前次申报不实已向主理人如实答复并重做，
> 本节与 patch 均以实物自验后落笔）。

- **背景**：N175 线新文件（commit `27d9b28a7`）使 run 40226939426 死在 step 11 mypy core
  棘轮（baseline 0）。修复是 push 放行硬前置。
- **基准**：`7ba470563`（起草时该文件与 HEAD `bc019e517` 逐字节一致，`git diff 7ba470563 HEAD`
  对该文件为空，已复验）；**不 commit**，以 patch 交付
  `delivery-runs/20261025-235503/artifacts/mypy_v3_fix_draft.patch`
  （**SHA256 `e128624a4cbda858b89e0d77b71b5bba55ac5f3e56c674af748b1cf960621523`**，5680 字节，
  头部注明基准 commit 与 `git apply` 用法；revert 工作树后 `git apply --check` 干净适用）。
- **12 条错误**（mypy 2.3.1 与 CI 钉版一致，完整清单）：
  `:197 arg-type`（super().__init__(legacy_shell)）/ `:198 assignment`（self.config=config）/
  `:202 arg-type`（TaskCoreV3(config)）/ `:212 return-value`（core.empty 返回 V2 状态）/
  `:220/:230/:247 override`（validate/evidence_cache/attention_weights 收窄参数）/
  `:441 override`（checkpoint 收窄参数）/ `:227/:252 attr-defined`（V2 config 无
  max_input_evidence_bytes/max_output_position）/ `:428/:482 attr-defined`（trainer.lineage 未声明）。
  （此前 tail 截断漏记的 ：197/:198 两条已补入。）
- **修复设计（三决策）**：
  - **D-a 类级注解 `core: TaskCoreV3` + `config: TaskCoreConfigV3`**（ConditionalLanguageCandidateV3）：
    消 ：198/:212/:227/:252 四条。理由：V3 在 __init__ 中以独立版本号的 V3 core/配置实际替换了
    V2 父类构造所建同名属性，注解只是向检查器声明真实类型。`core` 收窄是 TaskCoreV2 真子类型
    ⇒ mypy 静默放行；`config` 是平行版本类（非继承）⇒ 不变性拒绝重定义，注解行加
    `# type: ignore[assignment]`。类级注解不赋值，运行时零改动（冒烟验证 nn.Module 机制不受影响）。
  - **D-b 类级注解 `lineage: dict[str, Any]`**（ConditionalLanguageTrainerV3）：消 ：428/:482。
    lineage 仅在迁移/加载路径赋值、构造不初始化；只声明不赋值，保持 :457 `getattr(self,
    "lineage", {})` 默认值语义（冒烟断言 `not hasattr(t, "lineage")` 通过）。
  - **D-c 六处定向 ignore 带意图理由**：`:197/:202` 各 `ignore[arg-type]`（V3 配置鸭子兼容
    V2 构造所需同名字段，TaskCoreV3 未重写 __init__ 故继承 V2 签名）；`:220/:230/:247/:441`
    各 `ignore[override]`（有意 LSP 收窄至独立版本状态 TaskStateV3，版本隔离由 validate()
    运行时 isinstance 强制，放宽签名反而放过跨版本状态）。
- **验证读数**：scoped `mypy --follow-imports=silent taiji/conditional_language_v3.py` ⇒
  `Success: no issues found in 1 source file`（12→0）；完整门禁
  `mypy --follow-imports=silent seed taiji` ⇒ `Success: no issues found in 149 source files`
  （**baseline 0 恢复**，exit 0）。运行时冒烟：构造 candidate/new_state/validate/trainer 全通过。
  `git diff` 逐行核验：每个删除行与新增行代码 token 相同，仅追加注解/注释与 checkpoint 签名
  换行重排——**运行时零改动、预注册数值零触碰**。
- **遗留风险**：① N175 线可能对同文件在飞修改，patch 落地前必须 rebase 校验（已写进 patch 头部）；
  ② patch 由 N175 或主理人落地时建议落成独立 commit（仅该文件，显式 pathspec），便于 revert；
  ③ 本地 mypy 环境与 CI 冷缓存存在差异，最终以 CI step 11 绿为准。

## 回流判定

- 结论：通过
- 回流目标：<仅返工时填>
- 返工原因：<仅返工时填>
- 附注：C-4 契约修正（空 classname 两形态 + 唯一 node-id 分母）由主理人裁决生效、已按裁决
  amend 进 R-S3，不构成回流；api-tester 将做独立复核（同 run 锚，口径见上表）。
