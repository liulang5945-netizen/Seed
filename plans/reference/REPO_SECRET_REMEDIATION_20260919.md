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
