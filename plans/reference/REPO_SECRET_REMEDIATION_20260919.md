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

### 1.3 磁盘上有**三套**凭据，只有一套泄漏

| 位置 | 用途 | 是否泄漏 |
|---|---|---|
| `security/` | 开发运行的数据目录 | **否**（一直被 ignore）|
| `dist/Seed/security/` | 打包版（dist） | **否** |
| **`output/p6-1d-packaged-data/security/`** | **打包测试运行产物** | **是** ⚠️ |

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

- **不动 `security/` 与 `dist/Seed/security/`**：它们**从未被跟踪**；
  且 `security/` 是开发运行目录，`AuthManager.storage` 可能存有**用其 salt 派生的存量密文**，
  删 salt 会导致**那些数据无法解密**。**未泄漏却主动轮换 = 白白承担数据损失风险。**
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


