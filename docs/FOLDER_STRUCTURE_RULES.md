# 目录结构规范（Folder Structure Rules）

> 本文件管「**什么东西放在哪**」；仓库内容卫生（密钥/历史/守卫）见
> [REPO_HYGIENE_RULES.md](REPO_HYGIENE_RULES.md)（R1–R10）。两份互补，规则编号不重叠
> （本文件用 S1–S8）。
>
> 起因同样不是假想敌：2026-09-21 一次清理删掉 **15,754 个文件 / 135 个空目录（约 3.5 GB）**——
> `build/`、`dist/`、`win-unpacked/`、三处 pytest 临时根、一批 `.tmp-*` 实验草稿。根因是
> **没有约定**：临时目录随手起名、实验工作区散在仓库根、构建产物与源码混在同一层。
> 本文件把分类学固化下来，并配套可重复执行的清理工具
> [`scripts/clean_worktree.py`](../scripts/clean_worktree.py)。

---

## S1 六类目录（先分类，再谈位置）

| 类 | 定义 | git | 删除代价 |
|---|---|---|---|
| **A 源码** | 手写、入库、丢了靠 git 恢复 | tracked | 不可删 |
| **B 账本与证据** | 计划、报告、验收证据、设计稿 | tracked | **不可删**（多处是 `!` 白名单的有意入库，见 R4） |
| **C 运行时产物** | 程序跑出来的：日志、缓存、检查点、用户数据 | ignored | 随时，程序会自建 |
| **D 构建产物** | 编译/打包输出 | ignored | 随时，构建会重建 |
| **E 工具缓存与状态** | mypy/ruff/black/npm/IDE | ignored | 随时 |
| **F 实验草稿** | 一次性的实验工作区、探针输出 | ignored | 用完即删 |

**判据**：一个目录只能属于一类。说不清它属于哪类，说明它不该存在。

---

## S2 顶层目录台账（唯一权威清单；新增/删除目录必须改这张表）

### A 源码（tracked，不可删）
`api/` · `frontend/` · `instruments/` · `neuroplex/` · `seed/` · `seed_platform/` ·
`taiji/` · `tests/` · `scripts/` · `desktop/` · `desktop-electron/`（src 与 package.json）

### 仓库配置（tracked，不可删）
`.github/`（CI workflow）· `.devcontainer/`（容器配置）

### B 账本与证据（tracked，不可删）
`plans/`（**只追加**）· `docs/` · `reports/`（注意 `!reports/_gate_*.log` 白名单例外，见 R4）·
`artifacts/` · `design/` · `eval-r5b-s1-20260830/` · `.playwright-mcp/` · `.workbuddy/`（memory 随仓库保留）

### C 运行时产物（ignored；**可删 ≠ 全都可删**）
- 随时可删：`logs/` · `rag_data/` · `user_data/` · `agent_workspace/` ·
  `.seed_test_tmp/` · `_pytest_*/` · `update_frontend/`（空的前端热替换层）
- **不删**：`data/` —— dev 模式的运行数据根（S6），实测 **63.5 GB**，内含训练数据
- **不删**：`security/` —— 运行凭据在内部（文件本体已被 `**/security/*` 覆盖，见 R3）
- **谨慎**：`checkpoints/`（实测 767 MB）—— `*.pt` 按 R3 不入库，但**产品基座在此**
- **谨慎**：`taiji_data/` —— `seed*.spec` 打包 datas 的来源
- **按命名族管理**：`output/` 与 `outputs/` —— 混合体：既有**有意入库**的验收证据
  （`output/manual-r5-*/README.md`、`output/playwright/*.png`，见 R4 反向钉住），
  又有大量按命名约定 ignore 的运行目录（`output/taiji_r2_*/`、`output/*-packaged-data/`）。
  **不能整体 ignore，也不能整体删除**；新增运行目录必须按命名约定进 `.gitignore`（R1）。

### D 构建产物（ignored，随时可删）
`build/` · `dist/`（PyInstaller）· `frontend/dist/`（Vite）·
`desktop-electron/dist/`（tsc）· `desktop-electron/release/`

**两个例外必须记住**：
- `frontend/dist/` 虽属 D，但**开发态与打包态都靠它出界面**——平时不删；
- `desktop-electron/release/` 里 **`SeedSetup-*.exe` + `.blockmap` 是最终交付物**，
  清理时保留，只删 `win-unpacked/` 与 `release.stale.*/`。
- 另：`dist/` 是 electron-builder `extraFiles` 的输入，删了它之后下次
  `npm run dist` 前需先跑 `release.py --electron` 重建。

### E 工具缓存与状态（ignored，随时可删）
`.black_cache/` · `.mypy_cache/` · `.ruff_cache/` · `.npm-cache/` · `.vscode/` ·
`__pycache__/` · `neuroplex.egg-info/` · `_libs/` · `node_modules/`（各处依赖树）

**两个例外不在此列**：`.codex/`（含 git worktree 副本，删它等于销毁 worktree）、
`.local/`（opencode/copilot 会话状态）——两者在清理工具里归入「需所有者裁定」。

### F 实验草稿（ignored，用完即删）
`.tmp-<主题>/`（命名约定见 S3）· `.m0-checkpoint-*/`

### 特殊（不在六类里，动前先问）
`direct-*/`：历史实验工作区，**部分文件已被 tracked**（含 `w3-loop.pt` 检查点）——
删除前必须逐个 `git ls-files` 确认，不能当普通草稿清。`.git/`、`.workbuddy/` 永不清理。

---

## S3 临时/实验目录的命名约定

1. **仓库根的实验草稿目录一律 `.tmp-<主题>/`**，例如 `.tmp-r2-language-manual/`。
   根级锚定（前导 `/` 的 ignore 规则）+ 主题后缀，满足 R1（按命名约定写，不按实例名）。
2. **用完即删**。跨会话还要用的实验，把结论写进 `plans/` 或 `reports/`，把可复用的代码
   挪进 `scripts/`，然后删目录——**目录本身不是结论的存放地**。
3. **不得**在根目录新造第四种前缀（`.scratch-`、`work-`、`新建文件夹`）。
   看到不在台账里的根级目录，按 S5 清理或按本条改名。

---

## S4 构建产物只能出现在构建系统的输出目录

| 构建 | 输出目录 | 由谁产出 |
|---|---|---|
| Vite（前端） | `frontend/dist/` | `npm run build` |
| tsc（Electron 主进程） | `desktop-electron/dist/` | `npm run build` |
| PyInstaller | `dist/` + `build/` | `python scripts/release.py …` |
| electron-builder | `desktop-electron/release/` | `npm run dist` |

**规则**：不要手工往这些目录里放/改任何东西；不要把构建产物拷到仓库其他位置
（要"留一份"就打 tag 或走 release.py，它自己会产出安装包）。

---

## S5 空目录 = 运行残留，不是需要保留的结构

git 不跟踪空目录，所以**空目录不可能有版本价值**。运行时需要的目录由程序在启动时
`makedirs` 自建（`seed_platform/paths.py::get_external_path` 就是这么做的），
因此**见到空目录直接删，不需要请示**——除非它在 S2 的"不可删"清单里。

---

## S6 运行时可写数据一律走 `get_external_path()`，不进仓库

检查点、用户数据、多模态上传、`agent_workspace` 的落盘位置由
`seed_platform/paths.py::get_writable_base_dir()` 决定（frozen 下首选
`%LOCALAPPDATA%\Taiji`，可用 `SEED_DATA_ROOT` 覆盖）。**新代码不得把可写数据
写到仓库根或源码树里**——那正是 S1 分类被打破的第一步。

---

## S7 打包产物的唯一入口是 `scripts/release.py`

`dist/`、`build/`、`desktop-electron/release/` 由 release.py 产出与清理
（`clean_outputs()`）；electron-builder 侧的清理由
`desktop-electron/scripts/clean-release.mjs` 承担（**只重命名、不删除**——本机
批量删除大目录不可靠的教训见该文件头注释）。手工往 `dist/`、`release/` 里塞东西、
或手工"清理"它们，都会破坏两条构建链的增量假设。

---

## S8 模块内部的存放约定（一句话版）

- `api/`、`neuroplex/`、`taiji/`、`seed/`、`seed_platform/`、`instruments/`：产品源码；
  运行期生成的文件一律走 S6 的数据根，不落模块目录
- `tests/`：只放测试与其 fixture；运行期临时目录进系统 temp 或 `.seed_test_tmp/`
- `scripts/`：可复用脚本；一次性数据修复脚本按既有约定 `scripts/download_*` 忽略或归档
- `docs/`：治理与规范文档（`REPO_HYGIENE_RULES.md`、本文件）
- `plans/`：计划与结项账本，**只追加**（见各计划文件的冻结纪律）
- `reports/`：验收证据与报告；`_gate_*.log` 白名单随仓库走
- `design/`：设计稿与候选方案（如 `logo-candidates/`），选定后原候选保留作决策记录
- `desktop/` 与 `desktop-electron/`：两个壳各自自足；**跨壳共享的约定写在两边的
  配置注释里**，不建"共享目录"

---

## 清理工具：`scripts/clean_worktree.py`

把 2026-09-21 那次手工清理固化成可重复执行的工具：

```bash
python scripts/clean_worktree.py --dry-run   # 默认：只列出将删项
python scripts/clean_worktree.py --apply     # 实际删除
```

安全带（与本次手工清理一致）：
1. 只删 **S2 表中标注 ignored** 的目录与其下的空目录；
2. 豁免清单硬编码：`frontend/dist`、`desktop-electron/dist`、
   `desktop-electron/release/SeedSetup-*`、`.m0-checkpoint-*`（名字像实验标记，保守保留）、
   `direct-*`（tracked，须人工逐个确认）；
3. 每个目标先断言 `git check-ignore`，不满足即跳过；
4. 需 `CODEBUDDY_SAFE_DELETE_ENABLED=0`（本机批删守卫会拦截，见该环境教训）。

---

## 提交前检查清单（追加到 REPO_HYGIENE_RULES 的清单之后）

```bash
# 5) 工作区结构：有没有不在台账里的根级目录 / 空目录
python scripts/clean_worktree.py            # 默认 dry-run，只看不动

# 6) 结构守卫（会响的门）：台账外目录即红
CODEBUDDY_SAFE_DELETE_ENABLED=0 python -m pytest tests/test_folder_structure_guard.py -q
```
