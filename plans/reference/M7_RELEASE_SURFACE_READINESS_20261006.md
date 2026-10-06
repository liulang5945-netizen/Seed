# M7 发布面重启前对账（2026-10-06，只读取证，未改动任何产品代码）

> **性质**：本件是**盘查报告**，不是收官批准书，也不是发布批准。
> M7-CI 限定范围判据已于㊵-457达成（两道门在GitHub Actions ubuntu 绿）；本件回答的是
> 「发布面重启的前置是否已满足、若重启第一刀落在哪里、要花多少」。
> **M7 五项判据（01§2 M7 行：可安装运行包／manifest／正式CI／安全·回滚／发布说明）目前一条未满足**，
> 且其中四项的缺口在本件里逐项落到了具体文件与具体命令，不是印象判断。

---

## 1. 结论摘要

| 项 | 结论 |
|---|---|
| M7-CI 限定范围判据 | **已达成**（㊵-457，run 37453182065 success） |
| M7 五项发布判据 | **0/5**。CI 那一条达成的是「正式CI」，不等于发布判据达成 |
| 发布面重启的前置 | **已满足**（㊵-456 写明「重启＝CI 绿后另裁」，CI 已绿）⇒ **下一步是 owner 的裁，不是工程动作** |
| 唯一零凭据、零产物、分钟级的真实门 | **无**。发布面每一条都要凭据或要产物 |

---

## 2. 五项判据逐条取证

### 2.1 可安装运行包——**未达成，但本机已反复实证**

- 本地出包链路**实跑过 6 次**（`.20261002.1/.2`、`.20261003.1-.4`），产物`seed-0.1.7-alpha.1.<YYYYMMDD>.<idx>-win-x64.exe`，
  单包 **848 MB**，独立复核过 sha512 逐字符相等、blockmap 在位、通道只余三件（03 §5.7 ��六）。
- **但全部发生在本机，且全部走本地更新通道（loopback）**，不构成正式发布 workflow 的证据。
- 正式发布的分发面在 harness 上游是 `scripts/installed-update-cos.ts`（COS 桶）＋
  `publish-installed-update.ts`；本机配置 `apps/desktop/.env.windows` 走的是
  `DSH_DESKTOP_AUTO_UPDATE_ENV=test` + `DOWNLOAD_TEST_ORIGIN`（test档，非正式桶）。
  **正式桶凭据（`DOWNLOAD_TEST_COS_BUCKET` 那类）在 `.env.windows` 里不存在**——
  这正是 10-03 打包时`--build-version auto` 抛 `must be set to a non-empty value` 的同一根因。

### 2.2 manifest——**未达成**

- 版本基线校验本身**本机已绿**（本轮实测）：
  `release:verify --family dsh` ⇒ **`RC_VERIFY=0`**，报出 `315 member(s), 0.1.7-alpha.1`、
  publish order resolved、`0 peer declaration(s) unordered`。
  ⇒ **包版本对齐这一格的输入是干净的**，不需要修。
- 但正式发布的 manifest 分发（yml/通道文件生成、上传、幂等、校验和落盘）**未在 CI 跑过一次**。
- **结构性事实**：harness 自带 20 枚 workflow 全在 `taiji-harness/.github/workflows/`，
  **GitHub 只读仓库根 `.github/workflows/`，故这 20 枚全部惰性、从不出现在跑面上**
  （这正是M6 收官时"本 fork 零 CI"的结构原因）。真正在跑的只有根 `ci.yml`（Python 腿）
  与㊵-456 新建的根 `m7-ci.yml`（harness 两道门）。
  ⇒ **想让发布面进CI，必须把工作流放仓库根，不能放进子目录。**

### 2.3 正式CI——**M7-CI 这一条已达成，但仅限限定范围**

- 达成的是 ㊵-456 冻结的判据：`doc-sync`（43 叶）＋`hygiene`（18 叶）在 ubuntu 绿。
- 明确**不在**该范围（㊵-456 原文）：发布 workflow、安装启动/回滚/安全/发行包验收、
  e2e/coverage 重面、「零新增」放行。
- ⇒ **不得把 m7-ci 绿读作"发布面已验证"。**

### 2.4 安全·回滚——**未达成，且这一条有两个独立硬阻塞**

1. **代码签名凭据**：签名需要 `DSH_DESKTOP_WINDOWS_CER_FILE`／`SIGNTOOL`／`KEY_CONTAINER`／`TOKEN_PIN`
   （`scripts/windows-signing-stage.mjs`）。本机有 `.env.windows`配了这些，**但值是本机私有的**。
   CI 上要么把它们配成 repository secrets，要么走 `--unsigned`（**不签名 ≠ M7 安全判据达成**，
   只是绕过；owner 已装的包就走的是 unsigned 路径）。
2. **回滚**：回滚与 M6 判据③ 同源，M6 收官时被列入**排除项**（批准书 20261006）。
   `scripts/` 下有 `installed-update-identity.mjs`、`installed-update-package-content.ts`、
   `verify-installed-update-package.ts` 等回滚相关件，但**没有一条在 CI 跑过**。

### 2.5 发布说明——**未达成，且是零成本可做项**

- 本机出包 6 次，**没有任何一条 release notes 落库**。
- 这是五项判据里**唯一不需要凭据、不需要产物、不需要 GPU** 的一项。

---

## 3. 成本对照（把"哪条能自己走、哪条必须等 owner"划清）

| 判据 | 需要凭据 | 需要产物 | 需要 owner 动作 | 能否助手环境内自办 |
|---|---|---|---|---|
| 可安装运行包 | **是**（正式桶）或退unsigned | **是**（848 MB） | 桶凭据 / 接受 unsigned | **否**（sync spawn EBUSY，须普通终端） |
| manifest | 部分（yml 上传） | 是 | 桶凭据 | 否 |
| 正式CI | 否 | 否 | 否 | **是**（已达成） |
| 安全·回滚 | **是**（签名证书） | 少量 | 配 secrets 或接受 unsigned | 否 |
| 发布说明 | 否 | 否 | 否 | **是** |

---

## 4. 本轮新登记的事实（跨会话有用）

1. **`release:verify --family dsh` 本机绿**（`RC_VERIFY=0`，315 包同版本 `0.1.7-alpha.1`）——
   发布链的**版本对齐输入是干净的**，这条与 10-03 的build version 坑是两回事
   （那个坑在 electron-builder 的 productVersion，不在 dsh 包的发布族版本）。
2. **发布面进 CI 的前置是"把workflow 放仓库根"**，不是"写一个新 workflow"——
   harness 的 20 枚现成 workflow 因为在子目录而全惰性。这一条此前只在台账里以
   "本 fork 零 CI" 的形式出现过，**没被登记为发布面的具体阻塞**。
3. **`.env.windows` 里没有任何正式桶凭据**（只有 test 档 origin）⇒
   正式发布的分发面**在本机从未具备过配置**，与 10-03 的 `DOWNLOAD_TEST_COS_BUCKET` 抛错同源。

---

## 5. 引用边界（不得省略）

- 本件**不**声称 M7 收官，**不**声称发布面已验证，**不**替代任何发布批准。
- 引用 M7-CI 已达成时，必须同时写出 ㊵-456 的"明确不在范围"四项。
- 本件全部数字为本机取证（`RC_VERIFY=0`）与文件取证，**无一条来自 CI**；
  正式发布验证仍须按 01§2 M7 行"不能用局部历史测试数代替当前发布验证"。