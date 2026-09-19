# 仓库卫生：已泄漏凭据的处置方案（2026-09-19）

日期：2026-09-19。状态：**方案待确认执行**（删除属写操作，需用户点头）。
发现者：`727c53f7`（9/19 12:24 提交，非本轮）；本文件补充**影响面分析 + 可执行处置**。

## §1 事实（本轮实测）

### 1.1 泄漏点与范围

| 项 | 事实 |
|---|---|
| 泄漏的值 | `output/p6-1d-packaged-data/security/.jwt_secret`（64 B）、`.storage_salt`（32 B）|
| 引入提交 | **`8f7fc6fa`**（2026-09-01，"Add semantic provider fallback and packaged runtime support"）|
| **已在远端** | **是** —— `8f7fc6fa` 在 `origin/main` 中 |
| 已停跟踪 | **是** —— `727c53f7` 已 `git rm --cached`，并已随本轮 push 上远端（`origin/main` 现为 `727c53f7`）|
| 当前被跟踪的凭据 | **0**（`git ls-files` 精确 basename 扫描为空）|
| 守卫 | `tests/test_repo_secret_guard.py` **2 passed**（1 条查跟踪清单，1 条用 `check-ignore --no-index` 实测嵌套路径）|

### 1.2 `.gitignore` 修复已验证**彻底**

实测规则命中（`git check-ignore --no-index -v`）：

| 探测路径 | 命中规则 |
|---|---|
| `output/p6-1d-packaged-data/security/.jwt_secret` | `.gitignore:77:**/security/.jwt_secret` ✅ |
| `output/pkg/security/.storage_salt` | `.gitignore:78:**/security/.storage_salt` ✅ |
| `output/anything/security/audit_logs/x.log` | `.gitignore:79:**/security/audit_logs/` ✅ |
| `security/.jwt_secret`（仓库根那套）| 同 L77 ✅ |

### 1.3 磁盘上的凭据：**须逐文件判定**（2026-09-19 更正）

> ⚠️ **本节原判有误**：原文按**目录**判"一直被 ignore ⇒ 未泄漏"。
> 更正依据：下表为**独立复现**（逐路径查询，不合并 `git log -- a b c`）+ 用户核对。

| 位置 | `.jwt_secret` | `.storage_salt` |
|---|---|---|
| `security/`（开发在用）| `4ad8fa88…`，**历史 0 个 blob** ⇒ 未泄漏 | **`f1d2ed7b…`；历史 blob `5bfda9cd` 指纹相同 ⇒ 已泄漏** ⚠️ |
| `dist/Seed/security/` | **与 root 同值**，历史 0 blob ⇒ 未泄漏 | **与 root 同值** ⇒ **同一个盐** ⚠️ |
| `output/p6-1d-packaged-data/security/` | 已轮换（曾 `e6401f86…`）| 已轮换（曾 `abe3c9f1…`）|

**两条更正**：

1. **`security/.storage_salt` 并非"未泄漏"** —— 其**在位值等于已推送历史里的 blob `5bfda9cd`**
   （指纹逐位相同，且该 blob 已在 `origin/main`）。
   原判错在**按目录判定**：**判"是否需轮换"必须比对"在位值与历史 blob 的指纹"**，
   而不是数提交个数、也不是看目录是否被 ignore。
   `.jwt_secret` 的"未泄漏"结论**成立**（该路径历史 0 个 blob）。
2. **三套凭据并非互不相同** —— `dist/Seed/security/` 与 `security/` 的**两个文件指纹都相同**
   ⇒ **dist 构建把开发机这套凭据原样打了进去**。（pkg 那套 `e6401f86`/`abe3c9f1` 才另有其值，
   此前引用的"两处取值互不相同"说的是 root vs pkg，未覆盖 root vs dist。）

⇒ **dist 那份一旦被分发，接收方拿到的是同一个盐** —— 这已超出仓库卫生，属**发布面**，
只有用户知道 dist 是否曾给过他人。


**该目录下总共只有 3 个文件**：`data/app_settings.json`(98 B) + 那两个凭据。
⇒ **无存量密文**，`git ls-files output/p6-1d-packaged-data/` 现为空。

### 1.4 「删除即轮换」成立（源码已核实）

`seed_platform/auth.py`：

- `JWTManager._load_or_generate_secret()`（L58）：文件不存在或长度 <32 ⇒
  **`secrets.token_hex(32)` 重新生成并落盘**；
- `SecureStorage._load_or_generate_salt()`（L202）：文件不存在或长度 <16 ⇒ **重新随机生成**；
- 文档串自述「盐为随机生成并持久化在安全目录的 `.storage_salt` 文件中（不再硬编码）」。

⇒ 两个文件都是**运行期自生成、可丢弃**的，删除后应用下次运行会自动重建。

## §2 处置方案

### A. **必做且无损**：轮换泄漏的那一套

删除磁盘上的：

```
output/p6-1d-packaged-data/security/.jwt_secret
output/p6-1d-packaged-data/security/.storage_salt
```

**为什么无损**：
1. 该目录是**打包测试运行产物**，`git grep p6-1d-packaged-data` 只命中守卫测试里的**路径字面量**，
   **无任何代码依赖它**；
2. 文件缺失时**应用自动重新生成**（§1.4）；
3. 该目录**无其他密文**，删 salt 不会造成"旧密文无法解密"。

**效果**：历史里那对值**立即失效** ⇒ 泄漏面被关闭（历史里的字节串从此没有利用价值）。

### B. **不做**（并说明理由）

- ~~不动 `security/` 与 `dist/Seed/security/`：它们**从未被跟踪**；且 `security/` 是开发运行目录，
  `AuthManager.storage` 可能存有**用其 salt 派生的存量密文**，删 salt 会导致**那些数据无法解密**。
  **未泄漏却主动轮换 = 白白承担数据损失风险。**~~
  **⚠️ 该理由已失效**（见 §1.3 更正）：`security/.storage_salt` **确实已泄漏**
  （在位值指纹 = 历史 blob `5bfda9cd`，且该 blob 已在 `origin/main`），"未泄漏"这一前提不成立；
  又因 `dist/Seed/security/` 与 root **同值**，同一个盐同样暴露。
  ⇒ **结论修正：这个盐应当轮换。** 阻塞点不是"会不会白担风险"，而是
  **"先备份旧密文、再换盐"的顺序**；而实测（§6.3）显示当前**没有存量 Fernet 密文**，
  故轮换代价接近无损。（前提：dist 若曾分发，换盐救不回已发出的副本。）
- **不重写历史**（filter-repo/BFG）：值已轮换后改写历史的收益很低，
  而代价是**改写全部后续提交哈希 + 必须 force push + 所有克隆需重新拉取**。
  若日后有合规要求再单独立项（届时先整体备份 `.git`）。

### C. 可选加固（低风险，建议随后做）

`output/` 目前是**逐个目录**列举 ignore（`.gitignore` L188–221 一长串），
`output/p6-1d-packaged-data/` 当年不在列表里 —— **这与锚定 bug 属同一类疏忽**（没写通配）。
但 `output/` 下**确实有 23 个有意入库的文件**（`manual-r5-*/README.md`、`output/playwright/*.png` 截图），
**因此不能整体 ignore**。可行的小加固：**把"每个打包运行目录"用通配覆盖**，例如
`output/p6-1d-packaged-data/`（连同 `output/p6-*`），并让守卫测试把"新出现的 security 目录"纳入检查。

## §3 待确认

**唯一需要动作的是 A**（删除 2 个文件，合计 96 字节，纯运行产物）。
是否执行 A？执行后我会：删除 → 验证 `git status` 干净 → 确认应用可重新生成（只读检查，不启动训练）。

## §4 执行记录（2026-09-19，已执行）

用户确认后执行 A，**先备份再删除**：

**① 备份（仓库外）** → `E:\Seed-backup-secrets-20260919\`

| 文件 | 大小 | sha256 前 16 位 |
|---|---|---|
| `.jwt_secret` | 64 B | **`e6401f86c1313d6d`** |
| `.storage_salt` | 32 B | `abe3c9f10563a1ab` |

**交叉印证**：备份得到的 `.jwt_secret` sha256 前 16 位 **与 `727c53f7` 提交信息里记录的
「pkg `e6401f86c1313d6d`」逐位吻合** ⇒ **确认删除的正是被提交进历史的那一套**。

**② 删除**：两个文件已从 `output/p6-1d-packaged-data/security/` 移除，该目录现在为空。
`git status` 除用户自己的 `taiji/organs.py` 外无变化（二者此前已 untracked）。

**③ 重建路径实证（隔离环境）**：把 `_security_dir` 重定向到临时目录后实例化：

| 步骤 | 结果 |
|---|---|
| 空目录 + `JWTManager()` | 自动生成 `.jwt_secret`（64 字符，`token_hex(32)`）✅ |
| `SecureStorage()` | 自动生成 `.storage_salt`（32 字节）✅ |
| 再次 `JWTManager()` | **复用**已有值，不重复生成 ✅ |

⇒ **「删除即轮换」成立**：历史里那对值从此失效，且该目录下次运行会自动重建
（重建出的新文件已被 `**/security/.jwt_secret` 规则护住，不会再被跟踪）。

**④ 附带发现**：`_security_dir()` 在**当前开发形态**下解析到 `E:\Seed\security`
（即未泄漏的那一套）；`output/p6-1d-packaged-data/security/` 是**打包运行形态**下的独立目录 ——
这解释了为何三套凭据互不相同。

**未做**：`security/`、`dist/Seed/security/` 未动（未泄漏，且前者 salt 可能派生着存量密文）；
历史未重写（见 §2 B）。

## §5 加固 C 已执行（2026-09-19）

**问题**：`.gitignore` 里 `output/` 是**逐个实例名**列举（`output/taiji_m5_k_p3_0_*` 等一长串），
当年 `p6-1d-packaged-data` 不在列表里 ⇒ 才被跟踪。**这与"带斜杠锚定"是同一类疏忽：
写死了实例名，而不是命名约定** ⇒ 下一个打包目录会再次漏网。

**改法**（两条互补的规则，见 `.gitignore` L215–220）：

```
output/*-packaged-data/     # 按命名约定覆盖整族
output/p6-*/                # 兜住当前实例
```

**为什么不能整体 ignore `output/`**：那里**确有意入库的产物** ——
`manual-r5-*/README.md` 与 `output/playwright/*.png`（共 23 个文件），是版本化的验收证据。

**守卫同步加强**（`tests/test_repo_secret_guard.py`，2 → **4 条**）：

| 测试 | 作用 |
|---|---|
| `..._runtime_credentials_and_checkpoints_are_not_tracked` | 查跟踪清单（凭据/审计日志/`*.pt`）|
| `..._ignore_rules_reach_nested_packaged_data` | `check-ignore --no-index` 实测嵌套 security 路径 |
| **`..._cover_packaged_data_dirs_by_convention`**（新） | 按约定覆盖：同时探 `p6-1d-packaged-data` 与**假想的** `anything-packaged-data` |
| **`..._do_not_swallow_intentionally_tracked_output`**（新） | **反向钉住**：通配不得吞掉有意入库的产物 |

**实测验证**：

| 探测路径 | 结果 |
|---|---|
| `output/p6-1d-packaged-data/` | IGNORED（`output/p6-*/`）|
| `output/p6-1d-packaged-data/data/app_settings.json` | **IGNORED** ⇒ 非凭据类运行产物也被护住 |
| `output/v9-candidate-packaged-data/security/.storage_salt` | **IGNORED**（`output/*-packaged-data/`）⇒ 未来目录覆盖 |
| `output/manual-r5-s0/README.md` / `output/playwright/*.png` | **not ignored，仍被跟踪** |
| `output/` 下跟踪文件数 | **23（不变，零误伤）** |
| 守卫测试 | **4 passed** |

⇒ 两道独立防线：**凭据**由 `**/security/*` 护住（与目录名无关）；
**运行产物**由 `<name>-packaged-data` 命名约定护住。二者都不再依赖"逐个实例名列举"。

## §6 用户核对的更正与新增（2026-09-19 13:0x，**本文件的上游更正**）

### 6.1 两条更正（均已独立复现，见 §1.3 与 §2B）

| # | 原文 | 更正后 |
|---|---|---|
| 1 | `security/` 那套"否（一直被 ignore）"⇒ 未泄漏 | **对 `.jwt_secret` 成立**（历史 0 blob）；**对 `.storage_salt` 不成立** —— 在位值 = 历史 blob `5bfda9cd`（指纹 `f1d2ed7b…` 逐位相同，已在 `origin/main`）|
| 2 | 三套凭据互不相同 | **`dist/Seed/security/` 是 `security/` 的拷贝**（两文件指纹都相同）|

**方法教训**（比结论本身更值钱）：
- **多路径合并查询（`git log -- a b c`）不能逐路径归因** —— 必须**逐路径**查；
- **判"是否需轮换"要比较"在位值与历史 blob 的指纹"**，不是数提交个数、也不是看目录是否被 ignore。

### 6.2 可利用性：应按「**已可被拿到仓库的人解开**」处置

派生密钥 = `PBKDF2(机器指纹, 盐)`，而机器指纹 = `hostname|machine|USERNAME|processor`，
四个分量在**仓库内**的暴露情况（实测）：

| 分量 | 仓库内暴露 |
|---|---|
| `USERNAME` | **30 个被跟踪文件命中**（含 `.workbuddy/memory/*`、`reports/*.json`）|
| `hostname` | **`report.xml` 里直接写着 `hostname="DESKTOP-45AH838"`** |
| `machine` | `AMD64` 一类，**取值空间极小** |
| `processor` | 未直接命中，但**取值空间同样小** |

⇒ **只差最后一个分量，且它可枚举**。加之**盐已在远端历史中** ⇒
**"接近混淆而非加密"这一说法偏保守**：应按**已泄漏**处置。

### 6.3 轮换代价：**实测接近无损**

| 检查 | 结果 |
|---|---|
| 消费方 | 全仓 `git grep` **没有任何代码**消费 `SecureStorage` / `AuthManager.storage`（唯一命中是 `winrt.windows.storage` 的无关导入）|
| 存量密文 | `security/ data/ output/ seed_platform/ neuroplex/ api/ .workbuddy/` 全扫，Fernet 前缀 `gAAAAA` **0 命中**；`data/app_settings.json` 是明文 |
| 未扫全 | `dist/`、`frontend/`（每根 4000 文件上限）；唯一命中 `_tk_data/icons.tcl` 是 Tk 自带数据的巧合串（假阳性）|

⇒ **换根目录盐不丢数据**（前提：dist 若曾分发，换盐救不回已发出的副本）。

### 6.4 **第三道同类漏洞**：`outputs/`（复数）

`output/`（单数）与 `outputs/`（复数）**同时存在**；`.gitignore` 对复数**只有**
`outputs/project-audit-*/` 一条 ⇒ 实测：

| 探测 | 结果 |
|---|---|
| `outputs/` | ignored（目录规则）|
| **`outputs/x.txt` / `outputs/report.xml`** | **NOT ignored** ⚠️ |
| `outputs/project-audit-1/a.txt` | ignored |

`git ls-files outputs report.xml` = **6 个**，其中根目录 **`report.xml` 是 pytest junit 产物，
含 `hostname="DESKTOP-45AH838"`**（⇒ 同时是 §6.2 的 `hostname` 泄漏源）。

**这与前两道同属一类：按实例名而不是按约定写规则。**

### 6.5 守卫的**结构性缺口**（用户指出，成立）

现有 4 条守卫测的是「**规则是否命中**」与「**跟踪清单是否为空**」，
**测不到「在位值等于历史 blob」** —— 所以 6.1 那个漏检**它拦不住**，
而且 `outputs/` 这类"目录被 ignore 但子文件没被 ignore"也拦不住。
⇒ 需补一条**比对在位凭据与历史 blob 指纹**的守卫（**红/绿各跑一次**证明它能响）。

### 6.6 根盐轮换：**已执行**（2026-09-19）

**执行顺序**（先把"无损"钉死，再换）：

1. **前提钉死**：全仓 Fernet token（前缀 `gAAAAA`）扫描 ——
   `dist/ frontend/ output/ outputs/ security/ data/ seed_platform/ neuroplex/ api/ .workbuddy/ artifacts/`
   共 **6739 个文件，0 命中** ⇒ **无存量密文** ⇒ 换盐不丢数据；
2. **备份泄漏的盐**（仓库外 `E:/Seed-backup-secrets-20260919/`）：
   `root_storage_salt.leaked`、`dist_storage_salt.leaked`（两者指纹均为 `f1d2ed7bdd5ea4f9`）；
3. **删除** `security/.storage_salt` 与 `dist/Seed/security/.storage_salt`；
4. **触发重建**：实例化 `SecureStorage()` ⇒ root 新盐 **`f75efc8df6516828`**（32 B，≠ 泄漏值）；
   `dist/` 那份在 dist 下次运行时自动重建。

**`security/.jwt_secret` 未动**（两处历史 blob 均为 0 ⇒ 未泄漏）。
**`seed_platform/auth.py` 未改** —— 用**机器指纹**当密钥材料是**设计层面**的弱点（§6.2），
属**另议**事项，不在本次"仓库卫生"范围内。

**守卫红/绿验证**（用户要求"红/绿各跑一次证明它能响"）：

| 跑次 | 结果 |
|---|---|
| 轮换前 | **`1 failed, 4 passed`** —— 报出 `security/.storage_salt: 在位值(f1d2ed7b…) == 历史 blob 5bfda9cd(f1d2ed7b…)` |
| 轮换后 | **`5 passed`** |

⇒ 新增的第 5 条守卫确实拦得住前 4 条放过的这一类问题。

### 6.7 本次未处理、已登记的两项

1. **`outputs/`（复数）的 5 个被跟踪文件**（`outputs/project-improvement-20260915/*`、
   `outputs/r2_h3_7b_repair_20260917/*/report.json`）：**看起来是有意入库的证据**，
   与 `output/` 下那 23 个同类，故**未动**；但它与 `output/` 一样存在
   "**目录规则不覆盖子文件**"的结构隐患（`outputs/x.txt` 实测 NOT ignored），需要时按约定补规则。
2. **`report.xml` 已脱离跟踪 + 已 ignore**（`report.xml` / `junit*.xml`）——
   它是 pytest junit 产物，**内含 `hostname`**，是 §6.2 机器指纹泄漏的 `hostname` 来源；
   本地文件保留（26363 B）。
3. **历史里的旧盐与旧密钥**：仍可被取出（值已失效）。是否 `filter-repo + force-push` 属**另一量级决定**。




