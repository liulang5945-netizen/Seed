# 仓库卫生规则（Repo Hygiene Rules）

> **起因不是假想敌**：2026-09-19，一对运行期凭据
> （`security/.storage_salt`、`output/p6-1d-packaged-data/security/.jwt_secret`）
> 被提交进历史并推送到远端。根因**不是"谁不小心"**，而是三条 `.gitignore` 写法缺陷
> 与一个把可推得的机器指纹当密钥材料的设计 —— 也就是说，**规则缺失**才会让同一类事故
> 反复发生。本文件把当时的教训固化成可执行条款。
>
> 配套机器守卫：`tests/test_repo_secret_guard.py`（5 条，提交前必跑）。
> 事件全记录：`plans/reference/REPO_SECRET_REMEDIATION_20260919.md`。

---

## R1 规则按「命名约定」写，不按「实例名」写

**反例**（本项目真实出现过两次）：

```gitignore
security/.jwt_secret              # ✗ 带斜杠 ⇒ 锚定在仓库根，护不住 output/<pkg>/security/
output/taiji_m5_k_p3_0_*          # ✗ 逐个实例名列举 ⇒ 新目录必然漏网
output/taiji_m5_k_p3_1_*
…（一长串）
```

**正例**：

```gitignore
**/security/.jwt_secret           # ✓ 任意深度
output/*-packaged-data/           # ✓ 覆盖整个命名族
```

**判据**：若你写规则时**需要知道某个具体目录名**，这条规则就是错的 —— 下一个同类目录会漏。

---

## R2 目录规则**不**自动覆盖子文件

`.gitignore` 里写 `outputs/` 只忽略**目录本身**，`outputs/x.txt` 仍会被跟踪。
这是本次第三道漏洞（`outputs/report.xml` 就是这么进来的）。

**对策**：用 `**/` 或显式通配，并**实测**：

```bash
git check-ignore --no-index -v outputs/deep/nested/file.txt   # 必须命中
```

**`--no-index` 是必需的**：对**已跟踪**的文件 `check-ignore` 会沉默，从而掩盖规则本身的错。

---

## R3 凭据与运行产物默认不入库

**凭据类**（一律不入库，无论是否"临时"）：`.jwt_secret`、`.storage_salt`、`.fernet_key`、
`audit_logs/`、`.env*`、`*.pem`、`*.key`、`id_rsa*`、`.netrc`。

**运行产物类**：junit/`report.xml`（**含 hostname**）、`*.pt` 检查点、
`<name>-packaged-data/`、逐次运行目录。

**两件事必须同时做**：① `.gitignore` 覆盖；② **守卫断言**（见 R4）。
只做 ① 会在"新命名"出现时失效；只做 ② 会让文件先被提交再被发现。

---

## R4 守卫必须**双向钉住**

只查一侧的断言会**接受坏中间态**。本项目现有 5 条守卫的分工：

| 方向 | 断言 | 防的是 |
|---|---|---|
| 正向 | 跟踪清单里不得出现凭据/`*.pt` | 已入库 |
| 正向 | `check-ignore` 必须命中嵌套路径 | 规则写错 |
| 正向 | 命名约定族（含**假想的新目录**）必须被覆盖 | 规则只护当前实例 |
| **反向** | 通配**不得**吞掉有意入库的产物 | 为了"不泄漏"把验收证据一起 ignore |
| **正向** | **在位凭据指纹 ≠ 任何历史 blob** | 规则对、清单空，但**在用值与历史里的一致** |

最后一条是本次事故**唯一**能提前发现的断言 —— 前四条当时全绿。

**且：动手删文件前，必须先读 `.gitignore` 全文与其中的 `!` 例外。**

本项目就有活例子：`.gitignore` 里写着

```gitignore
*.log
!reports/_gate_*.log      # ← 显式白名单
```

也就是说 `reports/_gate_*.log` 是**有意入库的证据**（回归门的完整输出）。
2026-09-19 的第二轮体检里，"发现有 `.log` 被跟踪"一度被当成问题；
若顺手 `git rm`，就正好犯了本条要防的错 —— **为了"干净"删掉了有意保留的产物**。
**"看起来一定该忽略的扩展名"不等于"确实该忽略"。**

---

## R5 判「是否已泄漏」要比对「**在位值 vs 历史 blob 的指纹**」

**两个错误方法**（都踩过）：

- ❌ **数提交个数** / 看某路径"有几个提交" —— 与"当前在用的值是否泄漏"无关；
- ❌ **多路径合并查询后逐路径归因** —— `git log -- a b c` 会把多路径混在一起，
  据此曾误判"根目录那对也进过历史"。

**正确做法（逐路径）**：

```bash
git rev-parse <commit>:<path>          # 取历史 blob 的 sha
sha256sum <path>                       # 取在位值的指纹
# 两者相同 ⇒ 当前在用的值可从历史取出 ⇒ 必须轮换（仅"停跟踪"不够）
```

---

## R6 历史重写必须**镜像隔离**

**永不在主仓库原地重写**。正确流程：

```bash
git clone --mirror <repo> <tmp-mirror>
# 在镜像里跑 filter-repo / filter-branch，验证，再 force-push
```

**两个必须知道的坑**：

1. **`git filter-repo` 只重写 `refs/heads/*` 与 `refs/tags/*`** ——
   **自定义命名空间的 ref**（本项目是 `refs/codex/*`）**不会**被重写，
   它指向的旧树会**继续保留**敏感对象。**验证时因此不能只看
   `git log --all -- <path>`（它走 diff 逻辑、显示 0 次）**，
   必须查 **blob 是否仍可达**（`git rev-list --all --objects`）。
   处理：删掉这类 ref 后 `repack -ad`。
2. **远端还有 force-push 改不到的引用**：GitHub 的 `refs/pull/*/head`。
   若某 PR 提交含敏感路径，旧对象在平台侧仍可能短期可达 ⇒ 彻底清除需删 PR 或联系平台支持。

**且**：改写后**所有提交哈希都变** ⇒ 文档/记忆里记录的旧 sha 全部失效，需一并说明。

---

## R7 危险操作前备份 `.git`，并**验证备份可读**

2026-09-19 曾因一个准备脚本超时被 SIGTERM，导致 `.git` 的
`HEAD`/`config`/`refs`/`index` 被清掉（对象库还在）。**唯一救命的是事故前 90 秒做的整目录备份。**

**备份后必须验证**：`git --git-dir=<backup> rev-parse HEAD` 与当前一致、提交数一致。
**并保留**：`git gc` / `git prune` 在确认无需回溯前不要跑。

---

## R8 敏感文件的处置顺序：**停跟踪 → 轮换 → （可选）清历史**

顺序不能颠倒：

1. **停跟踪**：`git rm --cached <file>` + `.gitignore` 覆盖 + 守卫断言（**本地文件保留**）；
2. **轮换**：**先确认"无存量密文"**（否则换密钥会丢数据），再让文件缺失并由程序重建，
   或写入新随机值；**备份旧值到仓库外**；
3. **清历史**：默认**不做**。理由 —— 若密钥材料本身可推得（见 R10），
   清历史的收益接近零；此时应把力气花在**改设计**上。

---

## R9 单复数、相似目录不得并存

本项目同时存在 `output/`（单数）与 `outputs/`（复数），规则只护了其中之一 ——
这是 R1/R2 之外**第三类**同类疏漏。**新增顶层目录前先查是否已有近义目录。**

---

## R10 密钥材料不得取自已公开/可推得的信息

**本项目的真实缺陷**：`SecureStorage` 曾用
`PBKDF2(platform.node()|platform.machine()|USERNAME|processor, salt)` 派生密钥。
这四个分量里，`hostname` 与 `USERNAME`**本来就在仓库内容里**（junit 产物直接写着 hostname），
`machine` / `processor` 取值空间极小 —— **等于把口令的一半公开**：
盐一旦泄漏（本仓库发生过），密钥即可离线重放，**安全性接近混淆而非加密**。

**规则**：密钥材料必须是**高熵随机值**并持久化（如 `.jwt_secret` 的做法）；
盐/机器指纹**不能**单独构成安全边界，只能作为辅助（防彩虹表）。
密钥可与所在机器绑定 —— 但绑定必须通过**显式的、随机的**密钥文件完成，而非"环境推断"。

---

## 提交前检查清单

```bash
# 1) 卫生守卫（5 条）
CODEBUDDY_SAFE_DELETE_ENABLED=0 python -m pytest tests/test_repo_secret_guard.py -q

# 2) 新增 ignore 规则是否真的命中（含嵌套）
git check-ignore --no-index -v <path>

# 3) 跟踪清单里是否夹带了不该进的东西
git ls-files | grep -Ei '(secret|salt|\.pem$|\.key$|\.pt$|audit_logs)' | grep -v tokenizer

# 4) 准备提交前，确认没有"在位值 == 历史 blob"
#    守卫 R4 的最后一条已覆盖；如需手工核对见 R5 的两条命令
```
