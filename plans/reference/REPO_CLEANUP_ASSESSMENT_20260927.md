# 仓库可清理内容评估（2026-09-27，2026-09-29 修订 v2）

> **性质**：只读扫描结论，评估**未删除/移动/重命名任何文件**。判据沿用既有规则，不另起标准：
> 分类学见 [`docs/FOLDER_STRUCTURE_RULES.md`](../../docs/FOLDER_STRUCTURE_RULES.md)（S1–S8，六类目录 + 顶层台账），
> 内容卫生见 [`docs/REPO_HYGIENE_RULES.md`](../../docs/REPO_HYGIENE_RULES.md)（R1–R10，凭据/历史/守卫），
> 执行器为 [`scripts/clean_worktree.py`](../../scripts/clean_worktree.py)（默认 dry-run）。
>
> **取证时刻**：2026-09-27 17:37–18:45（GMT+8）。本仓常有并行会话，"是否仍在被写"以该时刻为准。
> 体积用 `du -sh`（磁盘占用）；第 0 档明细用清理工具（字节数），两者口径不同，差几个百分点。
>
> **修订记录**：
> - v1（2026-09-27）按**目录位置**切了四档：`output/`+`outputs/` 归"实验产物"档、`checkpoints/` 单独一档。
>   owner 指出两档重叠 —— 检查点 `.pt` 同时出现在两档（第 2 档的 `s40-*.pt` 快照与 16M 基座、
>   第 3 档的 `seed_beta.pt`/`p3b/`/`taiji_k_*`），"基座"出现两个、裁决判据（结论是否落账）也写重了。
> - v2（2026-09-29）改为按**重建代价**单轴切三档：零重建 / CPU+网络 / GPU 机时。
>   检查点不分家，统一归第 2 档。体积总数不变，仅归并口径。

---

## 0 一句话结论

仓库磁盘占用约 **121 GB**，其中：

| 分档（按重建代价切） | 体积 | 删掉的代价 | 是否需要裁决 |
|---|---|---|---|
| **第 0 档** 零重建代价（构建产物 + 工具缓存 + 运行残留） | **≈ 1.86 GB** | 无（程序自建 / 发布链已退役） | **否，可立即做** |
| **第 1 档** CPU + 网络重建（harness 构建与依赖缓存） | **≈ 7.3 GB** | 重编译 / `pnpm install` | 需要（活跃目录） |
| **第 2 档** GPU 机时重建（模型检查点 + 实验产物，不分目录） | **≈ 52 GB** | **重跑训练**（16M tick 级） | **需要** |
| 明确不删 | ≈ 61 GB | — | — |

**git 历史不需要瘦身**（§5）：已入库内容仅 213.78 MiB，无 `.pt` 泄漏，重写历史收益≈0、代价极大。

真正的取舍只有一条线：**越往下的档位，省的磁盘越多，但换回来要付出的不是磁盘、是机时。**

---

## 1 体积全景（du，磁盘实际占用）

| 目录 | 体积 | 处置 |
|---|---|---|
| `data/` | 60 G | **不删**（dev 运行数据根，含训练数据；09-27 17:57 仍在写） |
| `output/` | 50 G | 第 2 档（§4） |
| `taiji-harness/` | 7.5 G | 源码不删；其中构建/依赖缓存 ≈ 7.3 G 归第 1 档（§3） |
| `outputs/` | 1.1 G | 第 2 档（§4） |
| `checkpoints/` | 808 M | 第 2 档（§4）—— 与 output 里的 `.pt` 同类，不单独设档 |
| `dist/` + `build/` | 1.29 G | 第 0 档（§2） |
| 其余缓存/临时 | ≈ 0.57 G | 第 0 档（§2） |
| `.git/` | 113 M | **不动**（历史干净，§5） |

补充事实：**`dist/` 与 `build/` 最后写入 2026-09-22 13:25**，而本仓发布入口（`scripts/release.py`
与 Electron 打包链）按 S7 已于 2026-09-23 随旧线退役 ⇒ 这两个目录是**退役前最后一夜的遗留**，
删掉之后**不存在"下次还要重新打包"的重建成本**。

---

## 2 第 0 档：零重建代价，可立即删（≈ 1.86 GB / 9901 文件 + 39 空目录）

来源：`python scripts/clean_worktree.py`（dry-run，每一条都先断言 `git check-ignore` 通过才列入）。

| 目录 | 体积 | 文件数 | 重建成本 |
|---|---|---|---|
| `dist/` | 865.2 MB | 7610 | 无（发布链已退役） |
| `build/` | 471.9 MB | 23 | 无 |
| `.npm-cache/` | 141.0 MB | 1272 | 自动 |
| `.mypy_cache/` | 115.9 MB | 34 | 自动 |
| `_pytest_m2aa_runtime/` | 86.4 MB | 2 | 自动（测试自建） |
| `_pytest_m2aa_runtime_2/` | 86.4 MB | 2 | 自动 |
| `.seed_test_tmp/` | 66.8 MB | 33 | 自动 —— **09-27 17:02 仍在写，见 §6** |
| `.tmp-r2-language-manual/` | 64.5 MB | 4 | 无（一次性实验草稿，S3 用完即删） |
| `logs/` | 5.5 MB | 424 | 自动 |
| `_libs/` | 2.7 MB | 22 | 自动 |
| `.ruff_cache/` | 0.8 MB | 462 | 自动 |
| `.black_cache/` | 0.1 MB | 3 | 自动 |
| `__pycache__` / `neuroplex.egg-info` / `rag_data` / `user_data` / `agent_workspace` / `update_frontend` | ≈ 0 | 少量 | 自动 |
| **空目录** | — | 39 个 | 无（git 不跟踪空目录，S5） |

这一档的共同点：**没有任何一个文件的损失需要人工补偿。**

---

## 3 第 1 档：CPU + 网络重建 —— harness 构建与依赖缓存（≈ 7.3 GB）

`taiji-harness/` 是 dsh fork（活跃主线，09-27 17:52 仍有写入）。按 S8，根仓规则不越界管它内部，
故本节**只报告不作为**，是否清理由 owner 或 harness 侧约定裁定。

| 子项 | 体积 | git-ignored | 已跟踪 | 最后写入 | 说明 |
|---|---|---|---|---|---|
| `apps/desktop/.desktop-build/targets/` | 4.1 G | ✅ | 0 | 15:30 | 编译目标（Rust/Electron native） |
| `apps/desktop/.desktop-build/backend/` | 464 M | ✅ | 0 | — | 打包用的后端副本（含 wheelhouse） |
| `apps/desktop/.desktop-build/downloads/` | 241 M | ✅ | 0 | — | 下载下来的运行时/依赖 |
| `taiji-harness/node_modules/` | 2.5 G | ✅ | 0 | 12:38 | 依赖树 |
| `apps/web/dist/` | 19 M | ✅ | 0 | 17:19 | 前端构建输出 |

**四者全部 git-ignored、tracked 文件数 = 0**，删除不会动到任何入库内容。
代价是**重建时间**：`targets/` 要重编译（Rust native，量级数十分钟到数小时），`node_modules/` 要 `pnpm install`（网络）。
`website/`（61 M，32 个 tracked）、`.agents/`（26 M，3544 个 tracked）是**源码与笔记，不删**。

---

## 4 第 2 档：GPU 机时重建 —— 模型检查点与实验产物（≈ 52 GB，不分目录统一裁决）

> v2 修订要点：v1 把 `output/`+`outputs/` 与 `checkpoints/` 拆成两档，同类检查点跨档、判据重复。
> 现合并为一档，统一判据与统一"不可删锚点"清单。这一档的一切 `.pt` 与实验运行目录，删除代价同源：**重跑训练**。

### 4.1 output/ + outputs/（≈ 51 GB）

| 子项 | 体积 | 内容 |
|---|---|---|
| `output/_archived_manual-r5-canary_20260917/` | **32.1 GB** | 1190 个 `s40-*.pt` 快照（单文件 41.7 MB） |
| `output/taiji_r2_t1t2/` | 6.07 GB | 含 `_base16m/checkpoint_16000000.pt`（16M tick 训练基座） |
| `output/taiji_r2_t4/` + `t5/` + `t3/` | 3.7 GB | R2 逐区/逐任务读出 |
| `output/taiji_r2_readout_retrain/` | 0.34 GB | 受控重训三臂 |
| `output/taiji_m4r10_rule_audit/` 等 | ≈ 1.0 GB | M4 期实验 |
| `output/taiji-m2r1-phase-c-*/` | ≈ 1.4 GB | M2 期实验 |
| `output/_archived_manual-r5-canary_20260925/` `_20260926/` | ≈ 1.1 GB | 归档快照 |
| 其余 164 个老目录 | ≈ 2.1 GB | 历史实验 |
| `outputs/r2_h3_*` | ≈ 0.71 GB | R2-H3 系列 |
| `outputs/` 其余 | ≈ 0.34 GB | M1/M2 期小产物 |

### 4.1.1 裁决记录：三个 `_archived_manual-r5-canary_*` 归档（2026-09-29 核查，结论：可删）

对象：`_20260917`（32.1 GB / 1190 文件）+ `_20260925`（744.7 MB / 18 文件）+ `_20260926`（331.0 MB / 8 文件），合计 ≈ **33.2 GB**。

**身份判定：测试中间残留，不是训练产物。** 证据五条：

1. **登记在案**：`05_TECH_DEBT_REGISTER.md` DEBT-I7 第二实例——7 个 artifact-store 测试把
   `store_root` 指向共享目录 `output/manual-r5-canary/`，中间 `.pt` 按 PID 分组从不清理
   （实测 1054 个残留，`s29-*` 到 `s51-*`）；2026-09-17 处置＝"归档而非删除"（移入 `_20260917`）。
   `_20260925`/`_20260926` 是根因修好前又累积的同类残留（`s45-*`，按 PID 命名）的后续归档。
2. **命名指纹**：全部为 `sNN-*-<pid>.pt/.json`（PID 后缀 21364/21092/32740…），
   与登记的 `f"sNN-store-{os.getpid()}"` 写法吻合；无任何 `run_report.json`/`checkpoint_NNN` 等
   训练产物特征。成分：873+18+8 个 `.pt`（≈33.2 GB）+ 317 个小 `.json`（0.5 MB）。
3. **零代码引用**：`tests/`、`taiji/`、`scripts/`、`seed_platform/` 全部 grep 无
   `_archived_manual` 引用；`tests/` 仅两处提到 `output/manual-r5-canary`
   （守卫钉住 `README.md` + `_scratch.py` 的历史教训注释）。
4. **结论已落账**：DEBT-I7 处置记录完整——根因（7 个测试）、修复（改走
   `tests/_scratch.py::artifact_scratch_root()` 仓库外 scratch）、验证（8 项相关测试复跑全过、
   此前"单独过/全量失败"的 5 项假失败消除）。
5. **入库报告不依赖归档**：`reports/taiji_w7_r5c_s40_runtime_artifact_repeated_retention_20260831.json`
   只是格式名含 "s40"，未引用归档内任何文件路径。

**附带收益**：这些残留恢复出来只会让全量测试再次假失败（DEBT-I7 的原始病灶），删除反而消除风险。
**保留物**：`output/manual-r5-canary/README.md`（git 跟踪，守卫钉住）与 `native-canary.pt`
（2.9 MB，08-29 原有件，登记明言必须保留）。

### 4.2 checkpoints/（≈ 808 MB）

| 子项 | 体积 | 说明 |
|---|---|---|
| `checkpoints/p3b/` | 589.5 MB | P3b 战役的唯一共同检查点，历史结论依赖它 |
| `.p2-12-natural-language-write.pt` / `.p2-12-conflict.pt` / `seed_corpus.pt` / `resumed_seed_native.pt` / `seed_native.pt` | 各 37–41 MB（≈ 198 MB） | 老检查点 |
| **`checkpoints/seed_beta.pt`** | 4.0 MB | **产品默认基座（16M tick，来源登记 v2）** |
| `taiji_k_*` 系列小目录 | ≈ 5 MB | 实验残留 |

`checkpoints/` 09-27 14:09 有写入，属在役目录。

### 4.3 本档的不可删锚点（先钉死，再谈删）

1. **`checkpoints/seed_beta.pt`** —— 产品默认基座（16M tick，来源登记 v2）。不动。
2. **`output/` `outputs/` 下 27 个被 git 跟踪的证据文件**（R4 反向钉住的有意入库）：
   `output/manual-r5-*/README.md`、`output/playwright/*.png`（21 个截图）、
   `outputs/r2_h3_7b_repair_20260917/*/report.json` 等。删这些是删证据，不是清垃圾。
3. **结论尚未落账的实验产物**：判据只有一条 —— **该实验的结论是否已完整落进 `plans/` 与 `reports/`？**
   已落账 ⇒ 原始检查点可删；没落账 ⇒ 先补结论，再删。

### 4.4 裁决的实质

这 52 GB 的删除代价不是磁盘，是 **GPU 机时**（16M tick 级别训练，单臂数小时）。
`seed_beta.pt`、16M 训练基座（`checkpoint_16000000.pt`）、`p3b/` 共同检查点属于**重跑代价最高**的一类，
默认保留；其余按 4.3 第 3 条逐目录裁决。

---

## 5 明确不删 + git 不需要瘦身

**不删**：`data/`（60 G，活跃）· `security/`（运行凭据）· `.codex/`（内含 git worktree 副本，删它等于销毁 worktree）·
`.local/`（工具会话状态）· `taiji_data/`。

**`.git/` 不需要瘦身**（本次评估最干脆的一条否定结论）：

- 已入库内容合计 **213.78 MiB**（16061 个 blob），最大单文件 6.30 MiB（`reports/...json`）；
- **跟踪清单里没有任何 `.pt` 检查点** —— R3/R4 的守卫在生效；
- 对象库 pack 72.90 MiB + loose 28.88 MiB，合计 ≈ 102 MiB（du 113 M）。
- ⇒ `filter-repo` 能省的空间接近 0，代价却是**全部提交哈希改写**、文档/记忆里记录的 sha 全部失效、
  还要 force-push 并清理 `refs/codex/*` 与远端 `refs/pull/*`（R6）。**收益/代价不成比例，不做。**

---

## 6 风险（动手前必须知道）

1. **本机此刻有并行会话在写**（09-27 取证时刻）：`.seed_test_tmp`（17:02）、`data/`（17:57）、
   `taiji-harness/`（17:52，其中 `.desktop-build` 15:30、`node_modules` 12:38）、
   `output/taiji_r2_langfloor_D0/run_report.json`（16:11）。
   ⇒ 清理前先确认没有测试/训练在跑，否则第 0 档里的 `.seed_test_tmp`、`_pytest_*` 会打断别人的运行。
2. **`git status` 有 200+ 行改动**，其中 `taiji-harness` 的大量未跟踪文件是并行会话的 tsc 产物 ——
   不是评估产生的，不要顺手提交或清理。
3. **`output/` 与 `outputs/` 单复数并存**（S9 已记录的第三类同类疏漏）。清理不顺手合并；合并另立一刀。
4. 批删守卫：本机跑删除类脚本必须带 `CODEBUDDY_SAFE_DELETE_ENABLED=0`，否则被拦截或长时间无进展
   （易被误诊为磁盘卡死）。
5. 统计本身很贵：`taiji-harness` 的 `node_modules` 属海量小文件，Python `os.walk` 跑 36 分钟未出结果，
   改用 `du` 后仍需 10 分钟。**下次评估直接用 `du`，不要用 Python 遍历。**

---

## 7 建议的执行顺序（一刀一回滚点）

1. **第 0 档**：确认无并行任务在跑 → `CODEBUDDY_SAFE_DELETE_ENABLED=0 python scripts/clean_worktree.py --apply`
   → 跑 `tests/test_folder_structure_guard.py` + `tests/test_repo_secret_guard.py` 确认门仍绿 → 一个提交。
2. **第 1 档**：`taiji-harness/apps/desktop/.desktop-build`（4.8 G），按 harness 侧约定确认后可删，重建=重编译。
   `node_modules`（2.5 G）若近期还要开发则不动。
3. **第 2 档**：先核对 §4.3 锚点（`seed_beta.pt` / 27 个证据文件 / 未落账结论），其余**按目录逐个删**
   （每目录一个提交），不要一次清 52 GB。
4. **git 历史**：不做（§5）。

---

## 8 本轮未做的事

- 未删除/移动/重命名任何文件；
- 未改 `.gitignore`（本次发现的所有项都已被现有规则覆盖，无需新增）；
- 未动 `taiji-harness/` 内部（S8：根仓规则不越界）。
