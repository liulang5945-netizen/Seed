# DSH 0.2.x bug 修复清单与移植判据

> 数据源：上游 `dsh-v0.1.7-alpha.1...dsh-v0.2.1-alpha.1` 完整 compare diff（31,941,677 字节 / 6782 个文件条目），落盘 `E:/Seed/.fork-sync/upstream-0.1.7-to-0.2.1.diff`。
> 取证日期 2026-10-09。**本报告只做调研，未移植任何改动。**

## 1 规模

| 项 | 值 |
|---|---|
| 上游改动文件 | 6769 |
| 其中也在我们仓库里（交集） | **4675（69%）** |
| 新增 / 删除 | 1544 / 482（含 1010 个重命名行，多为 note 归档移动） |
| 上游新增 note 文档 | 132 篇：bug-fix **24**（12 个修复，中英双语）、feature 33、architecture 46 |

**结论**：fork 落后 6 个 tag，整体同步= 4675 个文件的重放；因此走"逐 bug 选择性吸收"。

## 2 上游 12 个 bug 与我们的相关性

| # | 上游修复（中文） | 落点 | 我们的相关性 |
|---|---|---|---|
| 1 | **为停止推进的 pnpm 运行设界** | `packages/boot/plugin-manager`（run-tree / operations） | **高**——2026-10-09 两次被 pnpm 卡死（`--lockfile-only` 卡在 `cdn.sheetjs.com`；`Already up to date` 假象） |
| 2 | **接管持有者已退出的写锁** | `packages/util/atomic-write`（`withFileLock`）+ plugin-manager 的 `run.json` 记录 | **高**——本仓曾出现"writer lock 残留窗口期读数"导致的 rc=1 未归因 |
| 3 | 按时间片执行 Session 列表工作 | session 列表 | 中高——客户端会话数增长后的列表性能 |
| 4 | 在关闭失败步骤前补齐工具结果 | agent 循环 | 中——失败步骤的记录完整性（影响 replay 与判读） |
| 5 | 工具结果的图文保留（多模态） | 工具结果渲染 | 中——图像类工具结果的保留 |
| 6 | Windows 打开经 explorer.exe 完成 | 桌面打开路径 | 中——客户端"打开文件"路径 |
| 7 | 刷新后任务标签页的恢复 | Web UI | 中——UI 状态恢复 |
| 8 | 列举工作区文件树中的目录链接 | 文件树 UI | 中低 |
| 9 | 分离 Chat 裁剪与独立滚动跟随意图 | Web UI 交互 | 低 |
| 10 | 输入回显在入档期间保持展示归属 | Web UI 展示 | 低 |
| 11 | 持久 pwsh 保留后端受控提示符 | pwsh 会话 | 低（本机打包场景走 bash） |
| 12 | 已看到的提示符标记保留自己的尾部宽限 | 提示符标记 | 低 |

## 3 建议优先移植的两个（已确认我们缺）

两者都在我们仓库里存在，但**没有**对应实现：

### 3.1 写锁接管（`packages/util/atomic-write`）

- 缺口：`src/` 中无"持有者进程已退出"的判据（无 `exited` / `takeover` 一类逻辑）。
- 上游机制：锁文件 `<file>.lock` 用独占创建 + `finally` 清理；进程未走到 `finally` 就留下锁，之后所有写者超时，除非人工删。修复以"**记录进程确已退出**"为接管判据（而非仅看 PID 存活），并对PID 复用、不可读锁、被他人已接管等情况保持等待。
- 附带更强的后续方案（上游 note 记为未采纳）：内核释放锁——POSIX `flock`（`@taiji/node-addon-system`）、Windows 命名信号量（`koffi`）。**本仓两个依赖都有**，可在移植时评估。
- 测试面（可作回归移植）：`atomic-write.spec.ts` 覆盖已退出持有者接管、8 个竞争者、活持有者/自进程/他人/空/不完整/非 PID/进程组/越界记录等待、PID 复用后不误接管、锁不可移除则等待、`EPERM` 拒绝重试等。

### 3.2 pnpm 运行设界（`packages/boot/plugin-manager`）—— **已评估：不可独立移植（2026-10-09）**

- 缺口：`src/` 中无 `run.json` 运行记录、无静默期超时逻辑。
- 上游机制：每次 pnpm run 期间在 `.plugin-manager/run.json` 记录；操作发现活跃记录时最多等 5 秒，超时则拒绝并指明进程与记录；执行侧对"静默不再推进"的运行做有界终止（进程树停止+ 有界排空持有管道）。
- **实测结论（移植尝试后回退）**：这个修复与上游的**"插件兼容性预检"特性长在同一个文件**——`operations.ts` 同时包含设界修复与 `readProfileVersionExemptions` / `evaluatePluginCompatibility` / `pluginCompatibilityWarning` 的调用，而后三个函数是上游 0.2.x 在 **`dsh-app-boot` 新增的 4 个文件**（`compatibility-preflight.ts` +187、`profile-compatibility.ts` +141、`plugin-compatibility.ts` +103，加`profile.ts` +132/-34）的一部分，约 630 行新特性代码 + 4 个新测试文件。
- 因此单独带修复不可行：要么引入整个兼容性预检特性（牵动核心包 app-boot，回归面不可控），要么手工切分代码（风险高于收益）。**本次回退，不移植**；`run-tree.ts` 与两个新测试文件已随之删除。
- 若将来需要，须作为独立立项："插件兼容性预检"按特性评估，不按 bug 修复夹带。

## 4 移植判据（建议）

1. ~~先移植 #2 与#1~~ —— **#2 已移植**（提交 `cac0ffc8f`，atomic-write 写锁接管，零新增依赖，33 测试全绿）；**#1 不可独立移植**（见 §3.2，与 app-boot 兼容性预检特性同文件），已回退。
2. #3–#7 按需：先看客户端实际是否已出现对应症状（没有症状就不急）。
3. #8–#12 与我们相关性低，暂不动。
4. 每次移植后跑：包自身测试 + 依赖该包的包测试 + `pnpm list`（验证 lockfile 未被带坏）+ `tsc -b tsconfig.host.json`。