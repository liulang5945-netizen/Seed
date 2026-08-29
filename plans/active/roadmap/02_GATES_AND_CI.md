# 持续门禁、CI 纪律与停止项

> 本文由原总路线图按职责拆分而来。原始行号：993–1208；本文件属于当前计划资料或历史证据，具体身份以本文件所在目录为准。
> 这是当前仍有效的验证纪律；历史事故的详细过程保留在历史记录中。

## 14. 持续门禁

- Taiji/Seed/Legacy 所有权 AST 测试；
- v1 state/checkpoint schema 和确定性恢复；
- TSK-v8 K 系列回归；
- 当前阶段 A Gate 的 holdout、lesion 和跨 seed 结果；
- 数据 manifest、实验注册、代码 commit 和训练 lineage；
- planned/actual learned state 与资源预算；
- 后端、前端、桌面、Legacy-off 启动和安全门禁；
- 构建期静态产物与上游依赖同步：`npm run check:aliases`（`hljsAliases.js` vs 当前 highlight.js，过期 exit 1）。该断言有两个执行点，且**两条都在必经路径上**——(1) CI `build-frontend` job 的 `npx vitest run`，经由 `src/__tests__/hljsAliases.test.js` 的逐字节一致断言（该 job 已于 §13.7 解除 `needs: test`，不再会被上游红隐藏为 skipped）；(2) `scripts/release.py` 的 `[1/5]` 前置步骤（不受 `--skip-frontend` 影响，失败即中止且不清理旧产物）。凡引入「由依赖推导、写死进源码」的产物，都必须同时引入这类门禁，**并且确认它在每条必经路径上都真的会执行**——只挂在测试里会被「升级依赖后直接打包」绕过；挂在测试里但那个 job 被 `needs:` 挟持，则连测试都不会跑。

辅助训练结果必须标记 `native-assisted`；只有不依赖辅助 teacher 决策且能继续终身学习的路径才能标记 `native-local`。A0–A9 的目的追溯和 Gate 定义以 Taiji v1 架构文档为准。

最近一次全量回归的实测数字（pytest / vitest / 覆盖率）**只记在** [IMPLEMENTATION_STATUS_2026_08.md 的「验证基线」节](../../reference/IMPLEMENTATION_STATUS_2026_08.md)，本文件不复制这些数字——按 14.1 与 14.4，同一事实只允许一个权威源，抄一份就等于制造一个会过期的第二权威。

### 14.1 门禁自身的可信度纪律（2026-08-26 事故后新增）

一次 CI 事故暴露出「门禁写下来」不等于「门禁跑过」：提交 `470f2af` 同时引入了 `black==24.12.0` 这个 **PyPI 上不存在的版本**（`24.10.0` 之后直接是 `25.1.0`）和多道新 blocking 门禁及「存量已清零」注释。依赖安装步骤因此在 30 秒内失败，其后 **全部门禁被跳过**，CI 连续 8 天红灯，期间累积的 84 个提交没有被任何门禁检验过。

因此以下规则生效：

- 任何 pin 的版本号必须先确认上游真实存在。PyPI pin 查 PyPI，pre-commit `rev:` 查上游 **git tag**（两者是不同的命名空间，`24.12.0` 在两边都不存在）。
- `.github/workflows/ci.yml` 的 pip pin 与 `.pre-commit-config.yaml` 的 `rev:` 必须同步改动，保持本地钩子与 CI 同版本。当前统一为 `ruff==0.16.4` / `black==26.5.1`。
- 门禁注释里的数字必须是**实测值**，不是期望值。声明「存量已清零」之前必须有一次真正跑绿的 run 作为证据。
- 依赖安装步骤失败会让后续门禁静默跳过（显示 `-` 而非 ✗）。判断 CI 是否真的验证过代码，要看步骤是否执行，而不只看 job 的红绿。

### 14.2 mypy 类型债（核心层已转棘轮 blocking，2026-08-26 收口）

2026-08-26 修好 pin 后，上述 blocking 门禁首次真正执行，实测与注释不符：

| 门禁 | 原注释声称 | 本机首测（py3.12） | CI 实测（run 32986602722，3.10/3.12） | 现状 |
|---|---|---|---|---|
| `ruff check .` | 存量已清零 | 0 | 0 | blocking |
| `ruff check . --select B,SIM` | 存量 32 | 4 | 0 | blocking |
| `black --check .` | — | 68 个文件待重排 | 0 | blocking |
| `mypy --follow-imports=silent seed taiji` | 0 错误 | 47 → **63** | **63**（两腿一致） | **棘轮 blocking，基线已降至 0；待新提交 CI 双矩阵复核** |
| 全仓 mypy | 基线 212 | 259 → **275** | **281**（两腿一致） | advisory 观测 |

**47→63 / 259→281 的漂移根因不是代码退化，而是 `mypy` 与 `pip-audit` 在 CI 里从未钉版本。** `ruff`/`black` 早已按 `.pre-commit-config.yaml` 钉死（0.16.4 / 26.5.1），唯独这两个漏了。检查器静默升版会带来新检查项，于是**没人改代码，门禁数字自己会变**。由此确立通用规则：**凡把工具输出数字当阈值的门禁，工具本身必须钉版本**，否则棘轮基线随时失效。现已钉 `mypy==2.3.1`、`pip-audit==2.10.1`。

原先「不能设阈值」的顾虑（mypy 报错数随 Python 版本变化，本机单版本数字不足为凭）已被双矩阵实测**否证**：3.10 与 3.12 的核心数（63）与全仓数（281）完全相同，双矩阵取较大值即等于单值，可直接钉。

核心 63 错经与最后一次绿色提交 `42d268e` 对比确认**不是新增退化**（那时 mypy 仍是 `continue-on-error: true`），属存量类型债。分布：`taiji/adapter.py` 12、`world_learning.py` 9、`workspace.py` 8、`local_learning.py` 7、`contracts.py` 5、`procedural_memory.py` 4、`seed/language_provider.py` 4，其余 14 文件各 1–3。主因是 checkpoint / `state_dict` 反序列化后为 `object | Any` 缺少类型收窄——这类缺陷与 14.3 记录的 checkpoint 静默失败同源，须实修而非长期忽略。

**为什么选棘轮而不是「等实修到 0 再转 blocking」**：advisory 对退化零约束，63 涨到 100 也照样绿灯，门禁形同不存在；而等清完 63 项再上门禁，这期间新增退化无人拦。棘轮（`errors > MYPY_CORE_BASELINE` 即失败）把「不许变差」立刻变成硬约束，又不阻塞开发。步骤同时对解析失败显式 `exit 1`——门禁绝不允许在读不到数字时静默放行，这是 14.1 的直接应用。

收紧路径：每次实修使核心数下降后，把 `ci.yml` 中的 `MYPY_CORE_BASELINE` 同步下调（步骤会打 `::notice::` 提示当前实际值），单向收紧至 0；全仓层待核心归零后按同一棘轮形式转正。2026-08-27 已完成核心层归零：本机 `mypy==2.3.1` 对 `seed taiji` 的 44 个源文件报告 `Success: no issues found`，因此门禁基线已从 63 下调为 0；这只证明当前 checkout，不能替代 CI 的 3.10/3.12 双矩阵实跑。

### 14.3 checkpoint 往返对称不变量（2026-08-26 回归后新增）

停写后复跑全量测试暴露一处真实回归：`TSKV8Adapter.checkpoint()` 从不写出 `cognitive_state`、`restore()` 也从不恢复它，而 `reset_dynamics()` **会**覆写 `_cognitive_state`。于是 `TaijiModel.score_bytes()` 的 `checkpoint → reset_dynamics(episode_id="evaluation") → finally restore` 三段式只回滚了内核状态，认知状态被永久留在 `evaluation` episode 上并带着漂移的 tick；`native_checkpoint()` 随后把这对不一致的状态写盘，`restore_native()` 的一致性校验抛出 `native cognitive state is out of sync with kernel state`。触发路径是 `scripts/training/train_seed_corpus.py` 的 `_flush()` 调用 `score_bytes()`，而 `_flush(final=True)` 紧接 `_persist()`——**每一次最终 checkpoint 都写在被污染之后**。

因此以下规则生效：

- 任何被 `reset_dynamics()`（或其他 in-place 状态重置）改写的字段，必须同时出现在 `checkpoint()` 的 payload 和 `restore()` 的恢复路径里。三者缺一即为缺陷，不是风格问题。
- `checkpoint()`/`restore()` 是成对契约，新增可变运行时状态时必须同步改这两处，并补一条往返断言，而不是等 `restore_native()` 的不变量在训练末期才爆。
- 新增 payload 键必须带向后兼容分支：旧信封缺键时按内核状态重建，而不是抛错或静默留下不一致值。
- 该缺陷类会让长训在最后一步失败并丢弃 checkpoint，直接违反「训练之前检查是否能够正确保存 checkpoints」这条前置要求；因此 checkpoint 往返测试属于阻塞级，不接受 advisory。

### 14.4 `plans/active` 编制与单一下一步纪律（2026-08-26 收敛后新增）

同一轮排查发现 `plans/active/` 出现第 6 份文档 `TAIJI_CONCEPT_FORMATION_GATE_2026_08.md`，全仓无任何引用（README、测试、脚本均未提及），但自带一节「下一步唯一入口」，其后半段与本文件第 16 节的 Gate 链几乎逐字重复——即**存在第二个「唯一下一步」权威源**，这才是 `tests/seed/test_project_identity.py` 失败的实质，而不是文件数量超限。

处置方式是归并而非改名或放宽白名单：其独有的运行时事实已并入上文 P7 事实清单，重复的下一步整节删除，原文件移入 `plans/archive/implementation/` 并在归档索引登记，`plans/README.md` 与身份测试均不改动。

因此以下规则生效：

- 「当前唯一下一步」只允许出现在本文件第 16 节。任何其他文档若要记录进展，只能写已完成事实，不得设立自己的下一步入口。
- 新增 `plans/active/` 文档前必须先确认它不是既有文档某节的复制品；能并入现有章节的一律并入。白名单是编制约束，冲突时收敛内容，不是放宽白名单。
- 归档文档必须显式声明其历史「下一步」不得恢复执行，避免残留方向在后续调用中与总路线竞争。

### 14.5 双学习栈收口与「假绿检查」纪律（2026-08-26 收口后新增）

`verify_taiji_native_v7.py` 的原生性 AST 契约把 `backward` 列为禁用属性，但 `taiji/` 内 8 个模块共 13 处仍在用 `SGD/Adam + loss.backward()`，契约长期失效。同一次排查还发现 `no_autograd_parameters` 是**假绿**：`PerceptionModule.parameter_tensors()` 返回 `parameter.detach()` 视图，而 detach 视图的 `requires_grad` 恒为 `False`，于是这条检查无论参数真实状态如何都通过。

处置方式取上限最高的一条：不是放宽契约，而是把 autograd 学习平面整体替换为与内核一致的原生局部信用分配。新增 `taiji/local_learning.py` 作为唯一来源，每条规则都对 autograd 做逐位等价验证（最差偏差 5.96e-08）；`LocalAdam` 在 detached 张量上复现含偏差校正的 Adam 更新式，因此迁移没有连带改变优化器、上层已调好的学习率与收敛阈值继续有效。`parameter_tensors()` 改为返回活体参数，该检查转为真检查并通过。收口后 15/15 检查为 true，8 道阻塞 verify 全 pass，`tests/` 437 passed / 5 skipped。

因此以下规则生效：

- 契约与实现冲突时，先问「哪一侧代表想要的架构」。契约代表目标架构时，改实现，不改契约。
- 布尔检查必须能失败。`detach()`、`copy()`、`float()` 之类的转换会顺手清掉 `requires_grad`、梯度和设备信息，把断言变成恒真式；断言 `requires_grad`/`grad`/`device` 之类元数据时必须作用于活体对象。
- 新增一条检查后必须构造一次**故意的失败**来证明它会红，否则它只是装饰。
- 手写梯度替换 autograd 时，必须逐处对 autograd 做数值等价验证再删除对照代码；GRU 的 reset gate 会给 `_hh` 侧 n 段多乘一个 `reset` 因子，这类不对称写错时**不会报错、只会静默学不动**。
- **假绿的第二种形态：被测函数自带早退门（platform / feature flag / 环境变量），测试没钉住该门，断言就在部分平台上根本没进被测分支。** 实例：`api/routes_terminal.py::_normalize_terminal_input` 首行是 `if sys.platform != "win32": return text`，而 `tests/test_terminal_input_normalization.py` 里断言「已转成 CRLF」的 3 条用例未钉 `sys.platform`，于是它们在 Windows 本机绿、在 Linux CI 红；同文件的 `test_existing_crlf_not_double_converted` 更隐蔽——它在 Linux 上也绿，但绿的原因是压根没进转换分支，验证不了「不重复转换」。修法是用 fixture 显式 `monkeypatch.setattr(mod.sys, "platform", "win32")` 钉住转换分支，另设一条独立用例验非 win32 的原样返回。**凡断言「发生了某个转换」的用例，都要能说清它在哪个分支里被执行。**

### 14.6 本机工具链环境事实（沙箱）

- `black` 的默认缓存目录 `%LOCALAPPDATA%\black\black\Cache\<ver>\` 在本沙箱下不可写。black 不会报错退出，而是反复重试建临时文件，表现为**永不返回并满载 CPU**（实测单文件 30 秒墙钟烧掉 651 秒 CPU），且其多进程池被中断后会**泄漏 worker 进程**（一次排查中发现 ~90 个残留 `python.exe` + 3 个 `black.exe`）。修复方式是把缓存指进工作区：`$env:BLACK_CACHE_DIR="<repo>\.black_cache"`，之后 `black --check .` 4 秒完成。`.black_cache/` 已入 `.gitignore`。
- 诊断这类「无输出挂死」不能靠 stdout——输出重定向同样为空。有效手段是在新终端观察副作用（`git diff --stat` 看文件是否已被改写、`Get-Process` 看进程与 CPU、`Get-CimInstance Win32_Process` 取准确命令行），必要时用 `faulthandler.dump_traceback_later(N, exit=True)` 强制打栈。
- `reports/ci_verify/` 是 `ci.yml` 中 8 道 verify 门禁的 `--output` 产物目录，属运行产物，不入库。

### 14.7 「本地绿 / CI 红」的排查纪律（2026-08-26 新增）

本机是 Windows，CI 的 `test (3.10)` / `test (3.12)` 是 Linux，`test-windows` 才是 Windows。因此本地全绿从不等于 CI 会绿，**本地跑过不构成推送前的充分证据**。出现分歧时按此顺序排查，不许先归因为「环境差异」：

1. **先确认两边跑的是不是同一份代码。** 本次分歧的真实原因就在这一步：CI 红的那次跑的是 `ffe1da2`（迁移前，`taiji/` 内 14 处 `.backward()`，AST 契约理应 false），而本地 pass 时工作区已是迁移后的未提交代码。用 `git grep -c "<pattern>" <commit> -- <path>` 直接查历史提交的内容，而不是看当前工作区。**CI 报红时默认它是对的。**
2. **再看失败项是否平台相关。** 断言里出现 `\r\n`、路径分隔符、大小写敏感、文件锁、`sys.platform` 分支时，优先怀疑测试自身缺平台钉定（见 14.5 最后一条）。
3. **在本机复现 CI 的平台条件，而不是等下一次 CI。** 对纯逻辑的平台门，直接改 `mod.sys.platform` 后跑该测试文件即可复现；复现后必须验证「旧断言在此条件下确实红、新写法确实绿」，两侧都验才算修好。
4. **读日志要认准步骤归属。** `gh run view --job=<id> --log-failed` 的输出里混有 advisory 步骤（mypy 的 47/259 条错误）的大量噪声，它们不是失败原因；用 `Select-String "FAILED|VERIFY_RESULT|Process completed"` 定位，并核对行首的步骤名。

### 14.8 `needs:` 会把整条下游门禁隐藏为 skipped（2026-08-26 收口后新增）

这是「假绿」的第三种形态，且比前两种更隐蔽：前两种是**断言本身**不成立，这一种是**门禁根本没跑**。

- **事实**：`ci.yml` 中 `build-frontend`（:162）与 `docker-build`（:215）都声明 `needs: test`。`test` 连续红 7 次期间，这两个 job 一直是 `skipped` 而非 `failure`，从未真正执行。`test` 转绿的那一刻它们首次运行、立刻双红——这不是新引入的回归，而是**被上游红长期遮蔽的既存缺陷**。
- **纪律**：判断「CI 是否绿」不能只看有没有红色条目，必须核对**期望执行的 job 集合是否都真的执行了**。`gh run view <id>` 里 job 数量少于预期时，要顺着 `needs:` 链回溯是谁被 skip 了。修完一个长期红的上游门禁后，**默认下游还有未曾运行过的门禁在等着红**，不要在上游转绿时就宣布收口完成。
- **两处被暴露的既存缺陷及其修法**：
  - `docker-build` 在 `Dockerfile:57 COPY data/ ./data/` 失败（`"/data": not found`）。根因是 `data/` 命中 `.gitignore:132:data/`、`git ls-files data` 为 0，只存在于本机（约 1.9GB），CI 全新 checkout 下必然不存在。修法是收敛到项目**既有**的「大体积/本地资源走挂载不进镜像」约定（`.dockerignore` 里该约定已列 `checkpoints`/`logs`/`reports` 等）：Dockerfile 改为 `RUN mkdir -p ./data`，`docker-compose.yml` 增 `./data:/app/data`，`.dockerignore` 增 `data`。空目录是安全的——`routes_model_switch.load_runtime_pref()` 异常即返回 `{}`，`training/resume._resolve_datasets()` 返回 `missing` 列表而不抛错，且 CI 后续的 metadata/healthcheck 步骤均不触及 `data/`。
  - `build-frontend` 在 `npm audit --production --audit-level=high` 失败：`nanoid@3.3.12`（GHSA-28wg-ghj8-5hjv、GHSA-2v37-7h3g-55p8）与 `postcss@8.5.14`（GHSA-fxqj-rqcc-2cmp、GHSA-r28c-9q8g-f849）两个 high。两者都是 vite 的**传递**依赖，项目源码零引用，因此不写 `devDependencies`（那是语义造假、且只保证顶层 hoist），改用 `package.json` 的 `overrides` 对全树强制 `nanoid ^3.3.18` / `postcss ^8.5.26`，未来 vite 自升后可整块删除、不留孤儿声明。余下 2 个 moderate 是 dompurify/monaco-editor，其 `npm audit fix --force` 会把 monaco 升到 0.56.0 的 breaking change，在 `--audit-level=high` 下不阻塞，故不动。
- **对不可本机验证项的诚实处理**：本机无 Docker（`docker` 命令不存在），无法本地构建验证。此时不假称已验证，改做等价的**静态审计**：逐个核对 Dockerfile 全部 10 个 `COPY` 源在版本控制中的跟踪文件数（`git ls-files -- <path>`），确认 `data`/`checkpoints`/`logs` 均为 0 且已无任何 `COPY` 引用它们，同类根因一次排净。
- **静态审计只覆盖它所提的那个问题，不等于该门禁会绿（2026-08-26 二次收口补记）**：上述审计问的是「每个 `COPY` **源**是否存在」，因此它确实预测对了 build 层转绿；但它问不到「每个被 import 的**包**是否都在 `COPY` 清单里」，于是漏过了下一层缺陷——`docker-build` 的 build 步骤通过后，`Startup smoke and healthcheck` 以 `api/app.py:26 ModuleNotFoundError: No module named 'seed_platform'` 失败。根因是 Dockerfile 手工枚举的 `COPY` 清单与 `pyproject.toml:64` 的 `[tool.setuptools.packages.find].include = ["seed*", "taiji*", "seed_platform*", "neuroplex*", "api*"]` 是**两份互不校验的重复清单**，而 `packages.find` 对缺失目录是**静默跳过**的：漏拷 `seed_platform/` 后 `pip install -e ".[legacy]"` 依然退出 0，缺失只在容器启动时才炸。`seed_platform` 是 `api`/`neuroplex` 的运行时核心（全仓 60 处 import、10 个跟踪文件）。
- **修法要消除重复清单本身，而非补一个包**：除补 `COPY seed_platform/ ./seed_platform/` 外，在 `pip install` 之后加一道**构建期导入断言** `RUN python -c "import api.app"`，把「镜像内缺包」从运行时 smoke 前移到 build 层，此后任何漏拷贝立即在构建时失败。断言位置须在前端产物与 `data/` 之前才安全，这一点经核实：`api/app.py` 的 `StaticFiles`/`dist` 使用全在第 264 行之后的 app 工厂函数体内，模块级只做路径常量计算，`seed_platform/paths.py` 的 `makedirs` 均带 `exist_ok=True`，故 `import api.app` 不依赖 dist 或 `data/`（本机同句实测退出 0，证明断言不会误红）。清单一致性亦已复核：pyproject include 的 5 个包与 Dockerfile `COPY` 完全对齐，`MISSING: none`；`desktop/` 不在 include 内，仅由 `[project.scripts]` 的 PyQt 桌面入口使用，容器不需要。
- **由此得到的通用纪律**：任何"手工枚举 + 上游有权威清单"的结构都是复发源，收口时要么让枚举可校验、要么加一道断言让偏差立刻失败；而**多步 job 只有全部步骤都绿才叫绿**——`docker-build` 的 `Build image` 打勾极易被误读成该 job 已通过。
- **闭环已实证（2026-08-26）**：`gh run view 32984530278 --json status,conclusion` 返回 `status=completed` / `conclusion=success`，7 个 job（`test 3.10`/`test 3.12`/`test-windows`/`Startup smoke (legacy)`/`Startup smoke (no-legacy)`/`build-frontend`/`docker-build`）全绿，其中 `docker-build` 的 `Startup smoke and healthcheck` 通过，确认 `seed_platform` 漏拷已修且构建期导入断言不误红。查询时注意：run 未结束时 `status=queued` 且 `conclusion=""`，此刻 `--log-failed` 会拒绝执行，须等 `completed` 再判定，不要把中途快照当结论。
- **结构性收口：`needs` 已删除，本条从「纪律」降级为「历史成因」（2026-08-28，详见 §13.7）**。上面那条纪律（"核对 job 集合是否都真的执行了"）是**依赖人记得去查**的补偿手段，属于下位对策。经核实 `build-frontend` / `docker-build` 均不消费 `test` 的任何产物（前者 10 步自 `npm ci` 起自给、`e2e/smoke.cjs` 只依赖 `vite preview` 不碰后端；后者镜像构建自带依赖安装），两处 `needs: test` 已删除，5 个 job 全部 `needs = None`。此后该失效模式**不可能再发生**，而非"要记得检查"。`skipped` 现象本身的描述仍然成立，保留作为成因记录。

### 14.9 npm 侧的沙箱事实与安全升级手法

- `npm audit fix` 会重解析整棵依赖树，在本沙箱下**无输出挂死**（实测 15 分钟、node 进程仅耗 2.9 秒 CPU、`package-lock.json` mtime 未变，属网络/解析阻塞而非计算）。判定手法同 14.6：看进程 CPU 与文件 mtime 的副作用，不看 stdout。
- registry 本身可用：`npm view <pkg> version --json` 秒回。可用它确认目标版本后走窄范围路径——`npm install <pkg>@<exact> --no-audit --no-fund` 或改 `overrides` 后 `npm install --package-lock-only`（实测 14 秒）。
- 改 `overrides` 后必须验证 lock 与 package.json 是否同步，因为 CI 跑的是 `npm ci`（不同步会直接失败）。npm 11 在解析结果已满足 override 时不会往 lock 根条目写 `overrides` 字段，所以**不能以"lock 里搜不到 overrides"判定失败**，权威判据是 `npm ci --dry-run` 退出 0。
- Windows PowerShell 5.1 的 `ConvertFrom-Json` 无法处理空字符串键，而 lockfileVersion 3 的根条目正是 `packages[""]`；检查 lock 结构要用 `node -e`。
- 前端门禁必须四道齐验，只跑 audit 不够：`npm audit --production --audit-level=high`、`npx eslint src --ext .js,.vue`（0 errors，warnings 容忍）、`npx vitest run`、`npm run build` + `dist/index.html` 存在性。

### 14.10 仓库可发现性：元数据是必要条件，且受令牌能力边界限制（2026-08-26 新增）

- 实测 `gh repo view --json` 确认：仓库自 2026-07-15 公开，但 `description=""`、`repositoryTopics=null`、`homepageUrl=""`、`usesCustomOpenGraphImage=false`，1 star / 0 fork / 0 watcher。GitHub 用于分发流量的字段全空，这不是"设计不佳"而是该层未填写。
- `repositoryTopics=null` 的后果是**缺席全部 topic 浏览页**（不是排名靠后，是不在列表里）；`description=""` 则把搜索匹配面全部推给 README 全文，而仓库名 `Seed` 是高冲突通用词，几乎不可能靠名字被检索到。
- topic 选择必须按**真实仓库规模**取中间区间，不能取最热。实测 `gh api search/repositories?q=topic:<t>` 计数：`local-learning` 25、`sparse-neural-networks` 43、`predictive-coding` 116、`episodic-memory` 140、`hebbian-learning` 148、`neuromorphic-computing` 328、`world-models` 551、`computational-neuroscience` 728、`cognitive-architecture` 784、`online-learning` 888、`pytorch` 59222、`artificial-intelligence` 47251、`deep-learning` 104237、`machine-learning` 235783。在 116 个仓库的页面里会被看到，在 235783 个里等于不存在，故 `machine-learning` 不贴；`spiking-neural-networks`（597）虽在好区间也不贴，因为内核不是脉冲网络，贴上是误导。禁贴 `agi`——README 自身声明 `not an AGI claim`，贴上即自相矛盾。
- **令牌能力边界（重要）**：本机 `GH_TOKEN` 是 App/细粒度令牌。`gh api repos/... --jq .permissions` 返回 `admin:true`，但 `gh repo edit --description/--add-topic` 与 `PUT /repos/{owner}/{repo}/topics` 均返回 `HTTP 403 Resource not accessible by integration`。措辞中的 "by integration" 是判据：App 令牌能力由 App 声明的 permission set 决定，与账号是否 admin 无关；该令牌有 `contents:write`（故 `git push` 一路成功）但无 `administration:write`，而 description/topics/homepage 属 Administration 档。**结论：这三个字段无法由 agent 用当前令牌写入（换会话亦无效，见下），不要反复换写法撞同一面墙。** social preview 图 GitHub 从未提供 REST 接口，本来也只能网页上传。
- **绕行路线穷尽结果（三条全否）**：（1）GitHub MCP 暴露的 40 个工具只覆盖 issue / PR / 文件 / 分支 / release / 搜索，无任何 repo settings 写入能力，排除；（2）`agent-browser` 本机未安装（`CommandNotFoundException`），浏览器自动化需先 `npm i -g agent-browser`，为改两个字段引入全局依赖不值当；（3）`RequestAuthorization(administration:write)` 回执为 success，但**授权成功 ≠ 能力到账**，见下条。

- **「换新会话即可写入」这一推断已被实测否定（2026-08-26 新会话验证）**：在全新会话中按原 §16 逐条重跑，三条 API 路线全部失败：`gh repo edit --description` → `HTTP 403 Resource not accessible by integration`；`PUT /repos/{o}/{r}/topics` → 403，响应头 `X-Accepted-Github-Permissions: administration=write`；GraphQL `updateTopics` → `{"type":"FORBIDDEN","path":["updateTopics"]}`。同会话内二次 `RequestAuthorization` 仍返回 success，但进程内 `GH_TOKEN` 前缀与长度不变（`ghu_`/40），写入依旧 403。
- **判据与根因**：`ghu_` 前缀说明这是 GitHub App 的 user-to-server 令牌，其能力上限由 **App installation 声明的 permission set** 决定，而非由本地 `RequestAuthorization` 的回执决定。`gh api repos/... --jq .permissions` 返回 `admin:true`（那是**账号对仓库的角色**）而 `X-Oauth-Scopes` 为空、`X-Accepted-Github-Permissions: administration=write`（那是**接口要求的 App 权限档**）——两者是不同维度，前者为 true 完全不蕴含后者放行。REST 与 GraphQL 走同一权限档，故 GraphQL 不是绕过 403 的后门。
- **通用纪律**：授权类回执（"authorization granted"、"start a new conversation"）属于**未验证的能力承诺**，必须以一次真实写入调用作为唯一验收判据；不能把它写成计划里的"已解决"。同理，凡出现 `by integration` 措辞，不要再在同一令牌上换 REST/GraphQL/参数写法反复尝试——那是同一面墙的不同侧面，正确动作是换执行主体（本人网页操作或换用具备 `administration:write` 的 PAT）。
- **闭环结果（2026-08-27，换执行主体后一次成功）**：用户在 GitHub 网页端完成三项写入，agent 侧用只需 `metadata=read` 的 `gh repo view --json description,repositoryTopics,homepageUrl,usesCustomOpenGraphImage,openGraphImageUrl` 复核并通过。这印证了上条纪律：受阻的是**执行主体的权限档**，不是方案本身，换主体后零重试即成功。
- **复核纪律（易踩）**：不要肉眼比对 `gh repo view` 输出。description 须做**逐字符相等**判定（250+ 字符里一个折行或全角标点差异肉眼不可见），topics 须做**集合相等**判定（同时报 missing 与 unexpected，因为 GitHub 返回时按字母重排，顺序不同不等于内容不同，而漏一个/多一个才是真错）。social preview 的唯一可信判据是 `usesCustomOpenGraphImage: true` 加 `openGraphImageUrl` 非空——打开仓库页面看图会被浏览器缓存与 CDN 边缘缓存欺骗，看到旧图或看到新图都不足以定论。实测踩坑：PowerShell `>` 重定向会给 JSON 写入 UTF-8 BOM，Python `json.load` 直接抛 `Unexpected UTF-8 BOM`，须用 `encoding='utf-8-sig'` 读取。
- **social preview 图的设计判据（改数字时复用）**：卡片在时间线里通常只被扫视约 1 秒，能留下的只有一个数字加一句机制主张，故只印 `0% → 94.12%`（全仓最强实测数字，来自双区 `[64, 48]` benchmark、seed 7 的 byte-cycle accuracy，见 README L203-L210）与 `no backprop / no attention`（区分于任何 Transformer 仓库的最短差异化陈述），并附 `two-region [64, 48] benchmark · seed 7` 使数字可追溯；不印 logo 或抽象插画。用 PIL 程序化渲染而非文生图，因为文生图会把数字糊掉，而这张图的全部价值就在数字的可读性上。配色取自现有品牌资产 `frontend/public/logo-taiji-ink.jpg` 的宣纸白 `#FAFBF6` 与焦墨黑 `#060604`，2× 超采样 + LANCZOS 缩放保证字缘锐利。
- **首版两个缺陷及修正（记录以免重犯）**：（1）surprise 衰减曲线横穿底部文字，视觉上把 `surprise 5.4041 → 0.1069` 划成删除线——把自家指标划掉，语义完全反了；结论是**造成语义反转的装饰应删除而非挪位**，已移除该曲线。（2）太极水印用了 `INK_FAINT` 且坐标写死，压住 `94.12%`；改为浅色背景层，位置由实测文字宽度算出，空间不足时**自动不画**——宁可留白也不撞字。另外脚本自检本身也抓到过一次真实问题（`bbox=(100,66,1180,614)`，页脚距下边缘仅 26px，有被各平台按不同比例裁切的风险），压缩纵向节奏后收敛到 `(100,66) → (1180,596)`。
- README 首屏顺序是唯一不依赖令牌权限的杠杆，且转化价值高于 topics（topics 带人进来，首屏决定是否留下）。原首屏被"命名分工 + 免责声明"占据，而最有传播力的两个资产（Transformer 责任对照表、`0%→94.12%` / `98.02%` 数字）分别埋在 L52 与 L184。已重排为：一句话机制主张 → badge → 对照表 → 实测数字 → 明示 status。**诚实声明一条未删**，只是移出首屏主位，并新增 `## Project scope` 承接原命名段。
- 改 README 首屏必须回原文核对每个被前移的数字有出处（实测首屏 `94.12/5.4041/0.1069/98.02/83,841` 全部对应 L203-L210 原表），并确认锚点标题真实存在（`#reproducible-tsk-v8-kernel-results` → L197）以及旧免责声明残留计数为 0——Markdown 锚点失效与声明重复都不会报错，只会静默劣化。

### 14.11 平台停机产生的「假红」：run 级结论不可信，须看 job 级是否真的跑过（2026-08-26 新增）

这是「假绿」的镜像形态，同样会误导收口判断：CI 显示红，但代码毫无问题。

- **事实**：`2026-08-26T15:11:58Z` 起 GitHub Actions 发生 `impact: critical` 停机（incident `y1t7p9fzrlj2`，15:48 的官方更新为「throttled inbound traffic … upstream Vitess issues」）。期间三次推送的 run 呈现三种异常：`560525c` → `startup_failure`；`c8acff5`、`4e6a827` → run 级 `failure`；以及推送后一段时间内根本不产生 run。
- **判别手法（关键）**：不要看 `gh run view <id>` 的 run 级 `conclusion`，要看 `gh api repos/{o}/{r}/actions/runs/<id>/jobs`。实测 `32986449122` 的 5 个测试 job 全部 `status=queued` / `conclusion=null`，`32985649140` 的 5 个 job 全部 `cancelled`，两者的 `build-frontend`/`docker-build` 均 `skipped`。**job 从未被分配 runner 却出现 run 级 failure，这种形状不可能由测试失败产生**——测试失败必然伴随 job 已 `completed` 且有真实耗时。据此可判定为平台产物而非本仓缺陷。
- **同时排除本仓嫌疑的四项旁证**：`ci.yml` 在该时段无改动；`git diff dae6464..HEAD` 的非文档代码差异为空；`gh workflow list` 显示工作流仍 `active`（未被禁用）；`HEAD == origin/main`。另有一条**反向证据陷阱**：此时 `gh workflow run` 返回 403，容易被误读为「工作流被停用」，实为 `ghu_` 令牌缺 `actions:write`（见 14.10），与停机无关。
- **停机期间的正确动作**：把 CI 的门禁在本机跑一遍，作为唯一还能推进「代码是否真绿」的手段；平台恢复后立即以新 run 取代本机结论，并停掉本机长任务（本机只能覆盖 lint/版本/类型，覆盖不了 docker/前端/多矩阵）。
- **恢复后的权威结论**：`3e6e5b0` 的 run `32986602722` 为 `status=completed` / `conclusion=success`，5 个 job（`test 3.10`、`test 3.12`、`test-windows`、`Startup smoke (legacy)`、`Startup smoke (no-legacy)`）全绿。停机期那三次红全部作废，不需要任何代码修复。
- **通用纪律**：判定 CI 结论前先确认**job 真的执行过**（有 `started_at`、有耗时、`conclusion` 非 null）。这与 14.1「依赖安装失败会让后续门禁静默跳过」是同一条原则的两面——红与绿都不可只看颜色，要看执行事实。
- **停机尚未结束，`98e36db` 的棘轮门禁线上验证仍未完成（待办）**：`98e36db`（钉版本 + mypy 棘轮）推送成功（`3e6e5b0..98e36db`）后 5 分钟内 `gh run list` 未出现对应 run，`githubstatus` 的 Actions 组件仍为 `major_outage`。故**新棘轮门禁在 CI 中的首次实跑尚无证据**，不得记为已验证。停机期间改做本机等价校验：四个钉版本经 PyPI 逐个确认真实存在（`mypy==2.3.1` 2026-08-15、`pip-audit==2.10.1` 2026-06-10、`ruff==0.16.4` 2026-08-20、`black==26.5.1` 2026-05-18），排除了「幻觉版本号导致安装步骤失败、其后门禁全部静默跳过」这一已发生过的复发路径（见 14.1 与 ci.yml 的 `black==24.12.0` 注释）；棘轮的解析与比较逻辑亦已在本机用真实 mypy 输出复现（`parsed=63`、`baseline=63`、`PASS(equal)`）。平台恢复后须补验：核心步骤在 3.10/3.12 两腿都打印 `mypy core errors: 63 (baseline 63)` 且不失败。

### 14.12 项目改名的环境变量残留：`E:\taiji-neuron` 反复自动重建（2026-08-27 收口）

现象：删掉 `E:\taiji-neuron` 后它总会再出现。这不是本仓代码所为，而是改名（`taiji-neuron` → `Seed`）时只搬了目录、没清用户级环境变量。

- **先排除本仓嫌疑**：全仓 grep `taiji-neuron` 只命中历史痕迹——`neuroplex/loader.py:23` 的注释（说明历史 ckpt 用 `taiji.*` 命名空间序列化）与 `plans/archive/**` 里的旧绝对路径链接。无任何活代码创建该目录。
- **真实成因**：三个 **用户级（注册表）** 变量仍指向旧路径：`XDG_CONFIG_HOME=E:\taiji-neuron\.local\config`、`XDG_STATE_HOME=E:\taiji-neuron\.local\state`，以及 `Path` 中的 `E:\taiji-neuron\.npm-global`（机器级作用域干净）。遵循 XDG 规范的工具（`opencode`、`gh`）启动时若发现目标路径不存在会**整条重建目录链**，所以删除永远不生效。证据：目录 `CreationTime` 为 2026-08-25，而 `.local\config\opencode` 的 `LastWriteTime` 是 08-27 11:04（当天），且内容清一色是工具配置/状态（3672 文件、278 目录，绝大多数是 `opencode\node_modules`），零项目代码。
- **动手前先证明无损**：`opencode.jsonc` 只有一行 `$schema`（无个人配置）；`gh auth status` 显示登录来自 `GH_TOKEN` 环境变量而非该目录（`.local\state\gh\device-id` 只是 36 字节匿名遥测 ID）；`npm config get prefix` 本就是 `C:\Users\23747\AppData\Roaming\npm`，而 `E:\taiji-neuron\.npm-global` **根本不存在**（死 Path 项）。因此「改回系统默认位置」无需迁移任何数据。
- **已执行**：备份用户 `Path` 至 `C:\Users\23747\user_path_backup_20260827.txt` → 用户级删除 `XDG_CONFIG_HOME`/`XDG_STATE_HOME` → 从用户 `Path` 过滤掉含 `taiji-neuron` 的段。复核：用户级与机器级全变量扫描已无任何 `taiji` 命中，`npm prefix` 为默认值。
- **验证纪律（易踩）**：注册表改动**不回灌已运行的进程**，而子进程继承父进程的环境副本，所以必须用 `[Environment]::GetEnvironmentVariable(..., "User")` 直读注册表来判定，不能看 `$env:`。实测新开子进程里 `$env:XDG_CONFIG_HOME` 仍是旧值——说明当前 IDE 进程仍持有旧变量，**由它拉起的工具还会重建该目录**，须重启 IDE/终端后再删。
- **未完成的一步**：`Remove-Item E:\taiji-neuron -Recurse -Force` 被沙箱拒绝（仅允许写 `E:\Seed`），目录仍存在，需用户手动删除。环境变量已清，故删除一次即永久生效。

### 14.13 `taiji/` 对 torch 的依赖面实测（2026-08-27，为公网 demo 可行性做的前置核查）

起因：讨论"能否把成果部署到公网"。结论先行——现有 `frontend/` **不能**直接静态托管，三条代码级证据：`src/composables/apiClient.js` 的 `resolveApiBase()` 在生产构建下推导后端为 `${hostname}:8000`（部到 Pages 会去请求 `https://xxx.github.io:8000`）；`vite.config.js` 的 `server.proxy` 只在 `vite dev` 生效、`build` 后消失；同文件的 `strip-crossorigin` 插件注释明写"QWebEngineView 兼容"，说明它本就是桌面壳内嵌页。且 `api/` 有 36 个路由模块（terminal、训练控制、模型切换、workspace 文件读写），整站公开托管是**安全问题**而非难度问题，已排除。故候选收敛为「静态成果页」与「WASM 内核 demo」，后者需要一个不依赖 PyTorch 的最小推理内核，遂做本核查。

- **命名事实（先纠错）**：`taiji_native` 不是包名，是 `e:\Seed\taiji\` 在测试目录、脚本名与报告名里的对外称呼。`Glob **/taiji_native/**/*.py` 只会命中 `tests/taiji_native/`，据此判断"实现不在这里"是错的。
- **截断陷阱（易踩）**：全仓 `import torch` 的 grep 输出被截断到 100 个文件，可见结果里没有 `taiji/*.py`，一度误判为"零 torch 依赖"。**输出被截断时"没出现"不等于"不存在"**，必须把 path 收紧到目标目录重跑才算证据。实测 `taiji/` 有 28 个模块共 33 处 `import torch`。
- **决定性事实：torch 只承担张量库角色，不承担自动微分**。全包 `.backward()` **0 处**（仅 `model.py:31`、`world_learning.py:335` 的注释提到"不用它"）、`torch.optim` **0 处**、`torch.autograd` **0 处**；`@torch.no_grad()` / `with torch.no_grad()` 共 **77 处**覆盖全部学习与推理函数；`nn.Parameter` 仅 1 处且 `local_learning.freeze_parameters` 会 `requires_grad_(False)`。这是 14.5 那次学习平面迁移的直接后果：`taiji/local_learning.py` 手写了 `backproject_linear`、`gru_forward_trace`、`cosine_similarity_delta` 与复刻 Adam 更新式的 `LocalAdam`。**移植 torch 到 WASM 的真正难点是 autograd tape / 动态图 / dispatcher，这三样已被整体绕开。**
- **算子面（43 个 distinct torch 函数，无硬骨头）**：`zeros`54 `tensor`45 `stack`32 `mean`28 `zeros_like`28 `cat`26 `abs`18 `empty`17 `linalg.vector_norm`16 `softmax`11 `Generator`10 `tanh`10 为主体，其余为 `arange`/`relu`/`sigmoid`/`clamp`/`eye`/`dot`/`outer`/`bincount`/`argmax`/`multinomial` 等初等算子；`nn.functional` 侧只有 `cosine_similarity`/`cross_entropy`/`one_hot`/`normalize`/`mse_loss`/`softplus`/`softmax`/`binary_cross_entropy_with_logits`。nn 层仅 8 种：`Linear`25 `Module`8 `GRU`4 `Embedding`1 `Parameter`1 `Sequential`1 `Tanh`1。张量方法以 `.detach()`178 `.clone()`174 `.to()`140 `.unsqueeze()`33 为主，难替代的索引/稀疏算子极少（`scatter_`1、`scatter_add_`2、`index_select`2、`masked_fill`2），**无 einsum、无 conv、无 sparse_coo_tensor、无 as_strided**。仅 `torch.linalg.solve`（2 处，稠密小矩阵）与 `nn.GRU` 需要专门实现，而 GRU 门控数学已在 `local_learning.py:246-303` 被逐步展开。
- **最小推理内核边界：8 / 35 模块、6623 行（占 `taiji/` 25404 行的 26%）**。从 `taiji.model.Taiji` 出发的传递闭包为 `contracts`2583 + `memory`907 + `model`810 + `config`603 + `fabric`591 + `sparse`548 + `state`337 + `organs`244。**闭包外 27 个模块与推理无关**，含最大的 `adapter.py`(6803) 以及 `concept_formation`/`perception`/`workspace`/`world_learning`/`planning`/`executive`。`contracts.py` 虽占闭包 39%，但是纯 dataclass + `_check_version`/`_check_text`/`_check_unit` 校验 + hashlib，且 `fabric`/`sparse` 只从它取一个 `StructuralTopologyProposal`，实际可再砍大半。
- **checkpoint 序列化耦合：低**。全包 `torch.save` / `torch.load` **0 处调用**；`checkpoint()` 统一返回 `{name: tensor.detach().cpu().clone()}` 的纯 dict（见 `perception.py:910`、`workspace.py:342`、`affordance.py:529` 等），`CHECKPOINT_FORMAT = "taiji-native-v8"` 为自定义格式。导出为 JSON / safetensors / 裸 Float32Array 不需改动内核逻辑，只需在边界写 dump 脚本。
- **尚未核查、不得先行决定的一项**：上述 6623 行是 Python，WASM 化有两条路且各有代价——(a) Pyodide 装载 Python + 一层纯 JS/WASM mini-tensor 替换 torch：快，但需下载约 6MB 运行时并写 torch shim；(b) 用 Rust/C++ 重写内核再编译：产物干净，但等于重写数值代码，且必须与 Python 版保持逐位一致，风险最高——`plans/manifests/taiji_native_runtime_profile_v1.json` 的 controls 里本就有 `cross_device_numerical_consistency_when_available`，说明项目自身对数值一致性有硬要求。此二选一需要独立尽调后再定，**不能因为依赖面结论乐观就顺势拍板**。

### 14.14 CI 未设 `concurrency` 与 `timeout-minutes`（2026-08-28 记账，随后独立一轮已收敛）

**.github/workflows/ci.yml** 全文既无 `concurrency` 也无 `timeout-minutes`（grep 三个关键词零命中）。§13.7 删掉两处 `needs: test` 后 5 个 job 立即全并发，峰值 7 个（`test`×2 + `build-frontend` + `docker-build` + `startup-smoke`×2 + `test-windows`），低于公开仓库 20 的并发上限，故不构成资源争用。两项欠账引入后各自独立：

- 无 `concurrency` + `cancel-in-progress`：同一分支连续推送时旧 run 不取消，白烧额度；PR 迭代频繁时尤甚。
- 无 `timeout-minutes`：任一步骤挂死（`vite preview` 端口未就绪、compose 健康检查轮询、pip 解析）会跑满 runner 默认 6 小时上限才被杀。

**处理（commit `b11cb7f`，2026-08-28）**：
- 顶层加 `concurrency: group: ${{ github.ref }} + cancel-in-progress: true`——同一 ref 只保留最新 run，旧 run 被取消；不同分支按 ref 隔离不串扰。
- 按 run 33158773941 实测时长留 2–2.5 倍余量设逐 job timeout：`test`（两腿 495–697s）30m、`test-windows`（694s）30m、`docker-build`（139s）20m、`build-frontend`（108s）15m、`startup-smoke`（41–92s）15m。
- 行为验证：PyYAML 解析通过；合并远端 `2b81c0d` 后 run 33164390727 全部 7 job success。concurrency 是消除旧 run 浪费而非解决资源争用（峰值 7 < 上限 20），`cancel-in-progress` 对 main 连续推送语义正确。

未在记账轮一并处理是刻意的：记账轮的可审计意图是"解除门禁挟持"，把并发治理混进同一次提交会让改动动机不再单一，日后回溯无法判断某行是为哪个目标而改。此项已按此原则以独立提交收敛。

### 14.15 测试环境与生产环境的能力差异是门禁盲区（2026-08-28 收口后新增）

`jsdom` 的 `document.createElement` 不校验标签名，Blink 严格校验。同一行代码在 vitest 里通行、在客户端里抛 `InvalidCharacterError` 并摧毁整棵 `router-view` 子树——**181 个用例全绿而线上白屏**，根因不是用例写少了，是运行环境比生产宽松（详见 13.8）。

纪律：**发现一处环境宽松，就把校验补进 vitest `setupFiles` 层，而不是只补一个用例。** 当前 `frontend/src/__tests__/setup/blinkDom.js` 已把 `createElement` 收紧到 Blink 同级（`/^[A-Za-z][^\0\t\n\f\r >/]*$/`），在 `vite.config.js` 的 `test.setupFiles` 注册，对全部用例生效。后续若再遇同类差异（如 `URL` 解析、`ResizeObserver`、CSS 解析宽严不一），一律加到同一个 setup 模块内收敛，不要另起并列机制。

配套的可信度要求沿用 14.1：**新门禁必须被证明能变红**。本例的做法是临时回退业务修复，确认新用例以客户端里那条一模一样的错误失败，再恢复。

### 14.16 进程身份不能靠命令行文本匹配确定（2026-08-28 假警报后新增）

排查子进程回收时，我按命令行文本匹配挑"主进程"，选中的却是 shell 包装层，于是把一个**完全正确**的内核级机制误判为失效，白烧一轮排查（详见 13.8）。

纪律：**需要确认某进程是否为另一进程的父/子时，唯一可信来源是 `ParentProcessId` 反查，不是命令行文本、不是窗口标题、不是启动顺序。** 涉及端口占用时同理——`/api/health` 有响应只证明"有人在监听"，不证明"监听者是我起的"，必须用 `GetExtendedTcpTable` 拿到 owner PID 再比对进程树。

这与 14.7「本地绿 / CI 红」、13.5「PowerShell 注入 BOM」属同一类：**机制看起来没生效时，先怀疑验证手段本身。** 已连续三次发作，故单列成条。

### 14.17 OpenAPI 快照门禁只盖请求面，响应 schema 漂移不可见（2026-08-29 三修收口后新增）

`tests/test_openapi_snapshot.py` 只把 paths、methods、`parameters`、`requestBody` 的差异写进 `messages`，而 `_save_snapshot` 只在 `if messages:` 分支内可达。由此产生两个后果，而且是同一个缺陷的两面：

- **响应 schema 变了不会让门禁变红**——契约的返回面完全在检测范围外；
- **`--snapshot-update` 对响应面是空操作**——因为进不了 `if messages:`，基线永远不刷新，删掉的 schema 会无限期烂在快照里。

实测证据：修 #3 时已从产品代码删除 `LifeNeedsPayload`（四个 `default: 50.0` 的编造字段，见 13.x 生命数据链路），但 `tests/snapshots/openapi_baseline.json` 里它仍在。跑 `pytest tests/test_openapi_snapshot.py --snapshot-update` 报 `2 passed` 且 `git diff --stat` 无变化，只能直接调 `_save_snapshot(_generate_schema())` 手工重生成，得到 `10 insertions(+), 42 deletions(-)`。**门禁报绿的同时，公开契约记录里挂着四个产品早已不再返回的编造默认值。**

纪律：**改动任何 `response_model` 后，不要依赖快照门禁给结论，必须手工重生成基线并逐行读 diff。** 重生成命令：

```powershell
python -c "import sys; sys.path.insert(0,'tests'); from test_openapi_snapshot import _generate_schema, _save_snapshot; _save_snapshot(_generate_schema())"
```

这与 14.15 同源：**门禁绿不等于被覆盖，得先问清它到底比较了什么。** 待办（未做，不许当已完成引用）：把响应 schema 差异也纳入 `messages`，并按 14.1 的要求先证明新门禁能变红——即临时改一个 `response_model` 字段，确认门禁失败，再恢复。

### 14.18 训练进度分母必须是「本次实际工作量」，ETA 不许用 fraction 外推（2026-08-29 实测定案）

用户报「训练剩余时间不够准确」。按 systematic-debugging 纪律先量后改，过程中**两个直觉假设都被实测推翻**，真因是第三个：

- **推翻假设一（分母单位错）**：`resume` 端点用 `p.stat().st_size`（含 JSONL 结构开销），而 `consumed` 只累加纯文本字节。实测偏差只有 1.02~1.03 倍，**量级不足以解释用户可见的错误**。仍作为次要正确性问题一并修掉。
- **推翻假设二（吞吐非线性）**：`scripts/archive/diagnostics/diag_eta_rate.py` 实测 CPU 速率稳定在 **147±2 字节/s**（窗口/累计比值 1.00~1.05），线性外推本身没问题。顺带发现 `PROGRESS_EVERY` 旁边「≈311 ticks/s」的注释是过期数字，已改为实测值。
- **真因**：`_train_worker` 在 `ticks >= max_ticks` 处 break（前端 `max_symbols` 默认 200000），但 `fraction`/`eta` 仍以整个数据集的 `total_bytes`（实测 410 MB）作分母。实测收尾 `fraction = 0.000488`，上报 ETA `2,789,445s ≈ 32.3 天`，而真实剩余为 **0**——**误差约 279 万倍**，进度条卡在 0.05% 后直接跳完成。

四条不可回退的纪律：

1. **有效分母** `effective_total = min(total_bytes, max_ticks)`。分母是「本次实际要处理的字节数」，不是数据集大小；任何截断参数都必须进分母。
2. **ETA 用速率换算剩余量**（`remaining_bytes / rate`），**禁止 `elapsed * (1 - fraction) / fraction`**。后者在 `fraction` 极小时无界放大，前者在分母被高估或训练被截断时仍然有界。
3. **暂停挂钟时间必须从 `elapsed` 里扣除**（`paused_total`），否则恢复后 ETA 被暂停时长污染。
4. **收尾语义要诚实**：语料读完或达 `max_ticks` 才 `final=True → fraction=1.0, eta=0.0`；**用户主动停止不许伪造 100%**（`_emit_progress(final=not stopped)`）。

门禁：`tests/seed/test_training_progress_contract.py`（4 例，先红后绿验证过）。此前 `_train_worker`/`_emit_progress` **零测试覆盖**，且 SSE 的 `eta`/`samples_per_sec`/`total_steps` 字段不在 OpenAPI 基线内——正是 14.17 那个响应面盲区的又一次代价。

前端同批修掉三处显示缺陷：step 计数器原按 `epoch * total_steps / total_epochs` 算，后端 epoch 恒为 1/1 导致**永远显示 100%**、与旁边的 ETA 自相矛盾；`fmtTime` 缺「天」档，超过 24 小时的估算会退化；吞吐单位标注 `symbols/s` 与后端「字节/s」口径不符。

### 14.19 checkpoint 往返等价性门禁落地，实测暴露并修复第三方器官「存入不能读出」缺陷（2026-08-29）

补上了 §3 准入始终缺的那道门禁：`tests/seed/test_checkpoint_roundtrip_contract.py`（3 例）。不再空转——裸 `Seed` + `learn_bytes` 会让约 25 个组件保持 `None`，等价性断言会退化为「都是空的所以相等」，因此 `_wire_runtime` 真实挂载原生 provider、episodic/semantic 记忆、自适应神经元区域，`_commit_real_history` 写入一条真实情节记忆与一次真实结构生长，再走 `resume._train_worker` 真实落盘 → `torch.load` → `Seed.from_checkpoint` 的完整路径，对 lineage（tick/继续一步的预测与概率分布）、预算（structural_budget）、结构（unit_ids/topology 状态与条数）、provider artifact（artifact_id/mode）、可见指标（homeostasis/episodic_count）逐项断言。

**实测发现一处产品级往返不对称，并非人为打洞**：`TSKV8Adapter.checkpoint()`（:8797）与 `native_checkpoint()`（:9369）无条件序列化任意已挂载的语言器官，而 `_restore_language_organ`（:9173）只接受 `native-readable`/`structured-stub`，对外接成熟解码器直接 `ValueError`。又因 `SeedRuntime.load()` 先 `Seed.from_checkpoint()` 再 `activate_language_provider()`，导致**任何接入 guarded provider 的运行时都无法从自己的存档启动**——正是本文件 §14.3 所述「长训在最后一步失败并丢弃 checkpoint」的缺陷类。修复为显式可观测的「脱挂留痕」而非拒绝整份存档：`_restore_language_organ` 对外接 backend 记录 `_detached_language_organ_backend` 并置器官为 `None`（外接解码器权重是运行时绑定资源，本就该由启动时的 `activate_language_provider` 重新挂载），新增 `detached_language_organ_backend` 只读属性供运行时区分「脱挂待重绑」与「本来就是原生」，`attach_language_organ` 重绑成功时清除该标记；与 `rotate_language_provider`（`seed/language_provider.py:547-553`）「staging 前先置空三组件」的既有惯例一致。

证据链：RED——guarded 用例自然抛出该 `ValueError`；变异探针（monkeypatch 掉 `_restore_topology_proposals` 使其丢弃结构）证明等价性断言会以 `topology_count 期望 1 实测 0` 干净地抓住结构丢失。GREEN——3 例通过后再跑全量：后端 `563 passed, 6 skipped`、Ruff check/format 通过、核心 mypy `Success: no issues found in 44 source files`、前端 `42 files / 237 passed`、`check:api-contract`/`check:native-boundary`/`check:aliases` 全过。据此 §3 的往返等价性准入从「未满足」翻转为「已满足」，长训准入的前提补齐。

### 14.20 recovery portfolio 客户端审计回放 Gate：只读绑定键 + 结构化错误码 + 审计面板（2026-08-29）

§2 当前 Gate 的 S0/S1 落地。三条不可回退的纪律：

1. **客户端不得自行猜测绑定键**。审计视图的 parent loop / snapshot / revision 只能来自服务端只读投影：新增 `GET /api/workbench/taiji/recovery-branch/context`（`SeedRuntime.taiji_workbench_recovery_portfolio_context`），返回 `{has_portfolio, parent_loop_id, snapshot_id, revision, selected_branch_id}`；无 portfolio 时返回 `has_portfolio:false` 结构化空态而非错误。禁止输入框、固定 loop id 或「最近一次」猜测。
2. **错误必须结构化、可分支**。portfolio 快照路由把稳定错误消息映射为 `detail.error`：`portfolio_not_persisted`、`portfolio_snapshot_not_current`、`portfolio_parent_mismatch`、`portfolio_revision_stale`（附 `observed_revision`）、`portfolio_invalid`。客户端 `readJson` 已支持 `detail.error`，前端按码分支渲染结构化状态，而不是解析人类可读文本。
3. **stale 时保留最后一个已验证快照**。面板 (`RecoveryPortfolioAuditPanel`，组合进 WorkspaceView 右栏「属性与检查器」) 在 `portfolio_revision_stale` 时保留 `lastValid` 并标记过期，不覆盖；切换 parent loop 时先清空再重绑，卸载时清空；事件投影驱动重取（不新增独立轮询）。只读合同：仅调用 context/portfolio 两个 GET，vitest 静态断言组件源码不含 `preflightLoop`/`executeLoop`/`previewIntent`/`executeIntent`。

证据链：RED——context 端点不存在（404）与 `detail` 为字符串均实测变红；变异探针之一（monkeypatch `_portfolio_error_code` 恒返回 `portfolio_unavailable`）触发 `assert 'portfolio_unavailable' == 'portfolio_not_persisted'`，之二（monkeypatch 快照投影注入 `parameters`）触发 `assert 'parameters' not in {...}`。GREEN——前端 `43 files / 242 passed`（+面板 5 例覆盖五状态矩阵、stale-keep-last、parent 切换清空、只读静态断言）、后端 `568 passed, 6 skipped`（+`tests/test_recovery_portfolio_audit_gate.py` 5 例覆盖 S0 状态矩阵/脱敏/容量与空态、struct 错误码、S1 checkpoint 回放）、核心 mypy 0、Ruff 全过、API contract `46 literals PASS`（context 路径入 OpenAPI 基线，已 `--snapshot-update`）、ESLint 0、`npm run build` 通过。

**S2 packaged-client 现场取证已完成（2026-08-29）**：最终 `dist/Seed/Seed.exe` 在 `SEED_ENABLE_LEGACY=0`、`SEED_RUNTIME=1`、`SEED_PORT=8138` 且不额外设置 Qt 环境的条件下启动；真实 `LOCALAPPDATA` 受限时自动选择包内 `user_data`，后端健康 200，`runtime/status` 报 `model_name=seed:seed_corpus.pt`、`is_taiji=true`、`is_seed=true`、native provider。真实 Workspace 路由 `#/workspace?taiji_client=desktop` 的 UI 证据为 Workspace/右侧检查器/恢复组合审计面板均可见，空态为结构化 `portfolio_empty`；Playwright 记录无页面错误、无 Legacy/Transformer/HF/GGUF 标记，全部 API 请求均落在 8138 且为 200。客户端实际观察到的 native Workbench capability snapshot 为 `5572f3ff01de596e380bda518eff357c4191610bab836d54e9c505c9b58f256f`、revision `4`，与运行时同次启动的 `seed:seed_corpus.pt` 对齐。完整证据见 [packaged_client_s2_20260829.json](../../../reports/packaged_client_s2_20260829.json) 和截图 `output/playwright/seed-s2-packaged-workspace-final.png`。本次启动没有持久化 recovery portfolio，因此分支/墓碑的非空排序仍以 S0/S1 replay 证据为准；不把空态 canary 夸大为非空恢复演示。

据此，本节 recovery portfolio 审计 Gate 的 S0/S1/S2 三层全部闭合，允许进入 W7-G0；后续仍必须保持“能力由服务端真实投影、客户端只读观测、非空分支由 replay 证据覆盖”的边界。

### 14.21 W7-G0 五份后续 Gate 合同冻结（2026-08-29）

W7-G0 已完成，且没有把合同存在误写成能力完成。五份版本化 manifest 统一冻结了 claim、owner、真实输入、结构化输出、trace、资源预算、checkpoint 往返、S0/S1/S2 证据、red proof、holdout、lesion、失败隔离、rollback 和不可越界边界：

- [W7-R1 provider watchdog](../../manifests/taiji_w7_r1_provider_watchdog_v1.json)：以 artifact digest 隔离健康状态、冷却、探测和回退；provider 不能拥有认知决策。
- [W7-R2 interaction-group](../../manifests/taiji_w7_r2_interaction_group_v1.json)：只从真实 trace 推导贡献/互补/冲突/恢复归因，不预设神经元角色名单。
- [W7-R3 visual/desktop](../../manifests/taiji_w7_r3_visual_desktop_v1.json)：只投影已验证 native capability，覆盖窗口、任务栏、托盘、通知、DPI、可访问性和 reduced motion；视觉不构成智能证据。
- [W7-R4 CUDA](../../manifests/taiji_w7_r4_cuda_v1.json)：本机显式为 `hardware-blocked`；必须在真实 CUDA 主机上复跑固定 workload、跨设备 checkpoint 和容差合同。
- [W7-R5 open-domain growth](../../manifests/taiji_w7_r5_open_domain_growth_v1.json)：成长只能由真实失败/容量/恢复压力触发，并经过 shadow → holdout → lesion → rollback → 原子合并。

`tests/test_w7_gate_manifests.py` 的 3 个结构测试通过；它还强制 R4 保持 `hardware-blocked`、所有未来 manifest 不得写 `implementation.status=passed`。G0 完成后执行入口曾切换为 W7-R1；G0 不产生新自治能力，也不替代后续各自的 S0/S1/S2 实证。

### 14.22 W7-R1 S0：provider health 按内容寻址隔离（2026-08-29）

R1 的 S0 已通过。原健康记录只以 `artifact_id` 为锚点，存在同 ID 换内容后继承旧失败计数的漏洞；现已把 `artifact_digest` 纳入 `LanguageProviderHealthState`、adapter 观察/回退和可见 provider status。空 digest 只作为旧测试 stub 的“未知”值兼容，真实 product-chat artifact 仍由 artifact Gate 强制 SHA-256 内容寻址。健康状态 checkpoint 向后读取无 digest 的 v1 记录，但不会把未知 digest 当作已验证内容。

证据链：`tests/seed/test_provider_watchdog_gate.py` 的 3 例验证 S0 报告、同 ID 内容替换清零计数、旧 checkpoint 可读；`scripts/training/eval_taiji_provider_watchdog.py` 可从仓库根目录直接执行，报告 [taiji_w7_r1_provider_watchdog_20260829.json](../../../reports/taiji_w7_r1_provider_watchdog_20260829.json) 为 `gate.passed=true`，指标包含 healthy rate `1.0`、连续失败阈值 `2`、单次回退、冷却抑制 `2` 次和 checkpoint roundtrip `true`；旧 provider 定向回归 `20 passed`，Ruff check/format 对新增脚本和测试通过。S1 尚未完成，不把 S0 报告当成真实重启 replay 或 packaged-client 证据。

### 14.23 W7-R1 S1：provider watchdog checkpoint replay（2026-08-29）

R1 的 S1 已通过。评测脚本新增 `--stage s1`，使用真实 `TSKV8Adapter.native_checkpoint()` 而不是只还原健康 dataclass：原生 adapter 挂载内容寻址 artifact 和 allowlisted registry，先写入两次失败，再保存；恢复后校验 artifact manifest、digest、registry revision/active id 和健康计数均保持一致，继续一次失败后 `probe_count=3`、`consecutive_failures=3`、`degraded=true`、`rollback_pending=true`，再次 checkpoint 恢复仍保持一致。

证据为 [taiji_w7_r1_provider_watchdog_s1_20260829.json](../../../reports/taiji_w7_r1_provider_watchdog_s1_20260829.json)，`gate.passed=true`；R1 S0/S1 定向测试 `21 passed`。本证据证明 provider 健康 lineage 可恢复，不证明真实外部 decoder 在客户端已经完成切换或恢复；S2 仍需独立验证 packaged client 的只读投影。

### 14.24 W7-R1 S2：Legacy-off packaged client provider 观测（2026-08-29）

R1 的 S2 已通过。使用重新生成的 `dist/Seed/Seed.exe`，明确设置 `SEED_ENABLE_LEGACY=0`、`SEED_RUNTIME=1`，不额外注入 Qt 环境，按用户默认端口 8000 启动。启动日志显示 backend ready、native runtime activated、WebSocket ready、frontend `loadFinished(ok=True)`；health 与 runtime/status 均为 HTTP 200，`seed_active=true`、`taiji_available=true`、provider 为 `native-readable/active`，`artifact_digest` 字段可读（原生内置器官为空字符串，不冒充外部内容寻址 artifact），health probes 为 0、degraded 为 false。

Playwright 真实页面 `http://127.0.0.1:8000/#/?taiji_client=desktop` 标题为 `Seed · 聊天`；Workspace、`STATUS EVIDENCE`、`Taiji Native`、语言器官状态均可见；8 个 API 请求全部为 8000，无页面错误、请求失败或 Legacy/Transformer/HuggingFace 标记。此次只验证服务端状态的只读观测和 backend/network 绑定，明确没有把客户端 provider 轮换或外部 artifact decoder 当作已验证能力。完整记录见 [taiji_w7_r1_provider_watchdog_s2_20260829.json](../../../reports/taiji_w7_r1_provider_watchdog_s2_20260829.json) 与截图 `output/playwright/seed-s2-packaged-legacy-off.png`。

据此，R1 的 S0/S1/S2 三层闭合，随后进入 W7-R2-S0；R2 仍必须从真实 trace 做可证伪 interaction-group 归因，不得将 provider 健康或 UI 显示当作认知能力。

### 14.25 W7-R2 S0：trace-grounded interaction-group 确定性模拟（2026-08-29）

R2 的 S0 已通过。新增 `taiji/interaction_groups.py`，将 interaction group 定义为 trace 上的可检验归因记录，而不是“规划神经元/记忆神经元”等硬编码角色。输入为版本化 `InteractionTraceEvent`、`InteractionTraceEpisode` 和 `InteractionTraceCorpus`：每个事件绑定 `event_id`、不透明 `owner_id`、`episode_id`、`outcome_id`、`checkpoint_revision`；每个 episode 绑定不透明 `context_id`、Outcome 数值、恢复效果和资源消耗。训练 digest 只哈希训练 trace 结构，不吸收 holdout Outcome。

估计器先在同一 context 内要求 `(none, first, second, pair)` 四格真实观测，再跨 context 聚合，计算 group contribution、signed interaction（正为互补，负为冲突）、recovery interaction、uncertainty、资源代价和 member/group lesion effect。只有训练/holdout 方向一致且满足资源与置信度预算时才产生 admitted record；缺格、低置信、资源超限、holdout 方向改变和 checkpoint revision 混合均 fail-closed，并保留 reason-coded tombstone。`owner_policy` 只是后续可投影的 lineage ID，S0 不修改 executive、memory、tool 或 provider。

证据为 [taiji_w7_r2_interaction_groups_20260829.json](../../../reports/taiji_w7_r2_interaction_groups_20260829.json)，其中 2 个候选分别保持互补/冲突方向，4 个不足证据候选被拒绝，holdout 方向、lesion、checkpoint 和 owner policy lineage 均可恢复，role label 输入数为 0。`tests/taiji_native/test_interaction_group_gate.py` 的 5 例与命名边界/跨区域回归共 `12 passed`，Ruff、compileall 和 JSON/diff 检查通过。红测覆盖 holdout 污染、跨 revision、资源压力和 checkpoint source digest 篡改；S0 不被当作真实 Workbench 学习或因果智能，已转入下节 S1。

### 14.26 W7-R2 S1：原生适配器 trace replay/sandbox（2026-08-29）

R2-S1 已通过。新增 `project_native_adapter_episode()` 作为原生状态到 interaction trace 合同的唯一投影边界：`TSKV8Adapter.cognitive_snapshot()` 提供真实 `Event` 与已结算 `Outcome`，`native_checkpoint()` 提供 `taiji-native-v1` 的 checkpoint revision；投影保留实际 native `event_id`/Outcome 绑定，只允许调用方用 `None` 排除非候选事件，owner 仍是不透明证据句柄，不生成语义角色名。

评测脚本 `scripts/training/eval_taiji_interaction_groups.py --stage s1` 在 8 个 train、8 个 holdout 原生 episode 上运行真实适配器，逐 episode 保存并恢复 checkpoint 后重新投影；16 个 replay record 全部精确一致，再执行同一 factorial evaluator，仍得到 1 个正交互和 1 个负交互，4 个候选被拒绝，holdout 方向、lesion、trace digest、owner lineage 和 `role_label_input_count=0` 均通过。证据见 [taiji_w7_r2_interaction_groups_s1_20260829.json](../../../reports/taiji_w7_r2_interaction_groups_s1_20260829.json)，测试为 `tests/taiji_native/test_interaction_group_replay.py` 与 S0 Gate 合计 `6 passed`，Ruff、compileall、JSON/diff 检查通过。

该层仍是 replay/sandbox：奖励来自确定性 factorial fixture，尚不能替代真实 Workbench 的 workspace/memory/planner/recovery workflow，也不写回 executive、memory、tool 或 provider。下一步必须进入 R2-S2 真实 Workbench workflow，并继续保留这条回放 Gate。

### 14.27 W7-R2 S2：真实 Workbench workflow（2026-08-29）

R2-S2 已通过。评测脚本 `scripts/training/eval_taiji_interaction_groups.py --stage s2` 使用真实 `SeedRuntime + WorkbenchEnvironment`，从当前 capability snapshot 投影 workspace list/search/read/stat affordance，经 native executive 选择后执行真实只读 Workbench route；每个 workflow 同时记录 Taiji native `Event`、Workbench world evidence、Outcome、失败后的 recovery retry 和 `taiji-native-v1` checkpoint replay。`workspace.stat` 仅用于建立真实 baseline，并以 `owner_id=None` 排除候选归因，避免把观测基线误当作学习到的交互组。

S2 使用 complementary/conflicting 两组四格任务组合，共 8 个 train、8 个 holdout。所有 record 的 checkpoint replay、world evidence、executive selection 和 recovery trace 均通过；holdout 只读且不可向 learner 添加证据，角色名输入数为 0，最终仍得到 2 个 admitted group 与 4 个拒绝候选。测试为 `tests/taiji_native/test_interaction_group_workbench.py` 与 S1 replay 合计 `3 passed`，Ruff、compileall、JSON/diff 检查通过。该层证明交互组评估已接入真实 Workbench trace，不证明开放域工具自治、写入自治或高阶无限搜索；仍不得直接写回 executive、memory、tool 或 provider。证据见 [taiji_w7_r2_interaction_groups_s2_20260829.json](../../../reports/taiji_w7_r2_interaction_groups_s2_20260829.json)。R2 三层闭合，下一入口为 W7-R3-S0 visual/desktop evidence。

### 14.28 W7-R3 S0/S1：生命状态表达与打包客户端收口（2026-08-29）

R3-S0/S1 已通过。前端把原有五维雷达恢复为生命状态页的主视觉；Taiji 原生状态面板仅保留紧凑的 runtime/provider/workbench/homeostasis 摘要，避免与需求雷达重复。侧边栏只保留一个“生命状态”导航入口，移除底部实时脉冲块；“状态依据”从聊天、能力、知识库、训练和设置页移除，在生命页底部作为默认折叠的只读审计详情保留。托盘生命状态动作先调用 `_show_window()`，再切换 `location.hash='/life'`，修复隐藏窗口中只改 hash 导致的无效果。

S0 的 `frontend` 回归为 `43 files / 245 passed`，Vite build 通过，ESLint 0 errors；桌面契约与项目身份测试现为 `13 passed`。S1 首轮曾只比较 `index.html`，无法发现 Vite hash 变化后 PyInstaller 仍带入旧入口资源，且当前受限 CPU-only 主机的 QWebEngine 多进程 renderer 会停在根 HTML；这两种“后端健康但界面空白”的假绿已在本轮收紧。`desktop/seed.spec` 现逐文件枚举前端目录，`scripts/release.py` 比较完整的相对文件集合与字节（源码与包内均为 211 个文件），并以 `--clean` 重建；冻结客户端同时显式设置 `QTWEBENGINEPROCESS_PATH`。在当前主机基线下，桌面默认使用 `--disable-gpu --single-process` 到达 `loadFinished`，不使用降低安全性的 `--no-sandbox`；具备稳定 GPU 的环境仍可通过显式 `QTWEBENGINE_CHROMIUM_FLAGS` 覆盖。最终 Legacy-off/native 包 `dist/Seed/Seed.exe`（SHA-256 `76b432b43922d5d70c64fca36b8e7045f2f5d03d4492f09b68b47eb31756368b`）在 8151 端口启动，日志出现 `Page loaded ... (ok=True)`、当前 hash 版 JS/CSS、`/api/runtime/bootstrap`、`/api/runtime/status`、聊天和训练请求，health 为 `ok` 且 `taiji_available=true`、`seed_active=true`、`model_loaded=true`、provider 为 `native-readable/active`。Chrome 页面级生命雷达与 900px/760px IDE 窄布局通过，顶栏高度会随换行增长且无横向溢出；证据见 [R3-S2](../../../reports/taiji_w7_r3_visual_desktop_s2_20260829.json)。Windows 任务栏、托盘、通知和高 DPI 因 Computer Use 无法激活 Seed 窗口仍未取证，不作通过声明。

### 14.29 W7-R5-S0：生产执行学习通道（2026-08-29）

R5-S0 已通过。此前 `SeedRuntime` 只选择、准入并执行 Workbench candidate，虽然产生了真实 `Outcome` 和 `WorkbenchTaijiEvidence`，却没有调用 `record_executive_outcome`，因此 `LearnedAffordanceFeatures.online_updates` 在生产路径永远不增长。现已在提交 `WorldEvent` 后接通该入口：执行前保存被选 source affordance 与 `_affordance_context()`，严格校验当前 decision、intent_id、`source_affordance_id` 和返回 Outcome 的 intent 一致；不一致直接 fail-closed，无兜底。`learn=False` 保持评测/回放冻结，`learn=True` 才提交在线 affordance 学习；`fit_updates` 不被伪增，checkpoint 往返保留两个计数并恢复同一最后决策。实现位于 `api/seed_runtime.py`，回归位于 `tests/test_workbench_contract.py`；定向 native/executive/desktop/project identity 测试 `24 passed`，固定工作区的 R5 学习与 checkpoint 往返测试通过，Ruff/compileall/diff 检查通过。证据见 [taiji_w7_r5_s0_learning_channel_20260829.json](../../../reports/taiji_w7_r5_s0_learning_channel_20260829.json)。本 Gate 不实现 R5-A 内化转换器、不实现 R5-B 效应器注册表，也不改变 R3-S2 现场证据状态。

## 15. 停止项

在 P2 通过前：

- 不续跑旧 16M→100M raw-byte 长训；
- 不为 TSK-v8 继续增加认知补丁；
- 不写绑定固定 fan-in 的自定义 CUDA kernel；
- 不用增加神经元数量替代学习型抽象；
- 不删除 Legacy 对照；
- 不把旧 N/M 通过记录宣传为完整智能进展。
