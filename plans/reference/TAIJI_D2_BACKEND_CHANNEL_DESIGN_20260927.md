# D2 独立分发通道｜设计定稿（2026-09-27，owner 09-27 弹窗批准方向 (b)）

> 背景：G5-D2 要求训练后端（Python＋torch 量级）到达用户机器。打包链既有 Python 载荷通道
> （`prepare-primary-runtime.ts` ＋ `pythonPackages` 完整映射）随发布物走且受 public-index 体积
> 上限约束——torch 量级塞不进（Z1c/G5 §1-D2 已定价）。owner 裁定＝(b) 独立分发通道：
> wheelhouse／离线安装器，复用 `DSH_PRIMARY_RUNTIME` 载体覆盖口子；桌面拉起生命周期一并实现。

## 1 · 目标与非目标

**目标**：装起即用——首次启动后训练面板可用（runtime 常驻、健康可探），无需用户手工装 Python 依赖。
**非目标**：GPU/CUDA 分发（CPU 推理与轻训练即可满足 G5 验收）；模型权重分发（另有通道）；
macOS/Linux 安装器（win-x64 先行，通道设计保持平台中立）。

## 2 · 通道架构（三件套）

```
发布物根
├─ runtime/                     # 既有 primary-runtime（Python 本体＋Node＋pnpm，不动）
├─ backend/
│  ├─ wheelhouse/               # ① 离线 wheel 集：pip download 的后端依赖（torch CPU 等）
│  │  └─ index-metadata.json    #    生成指纹：lock 哈希、目标平台、总字节、包数
│  ├─ install-backend.py        # ② 离线安装器：--no-index --find-links wheelhouse
│  │                            #    幂等（目标 venv 就绪即跳过），校验 wheel sha256
│  └─ backend-manifest.json     #    需求清单（与仓库后端 lock 同源生成）
└─ app-update.yml 缺席 ⇒ 更新器关闭（R4 既有）
```

**口子**：`DSH_PRIMARY_RUNTIME` 可覆盖路径的既有语义（shared-office-runtime 设计原文明载
「其他载体自选输出目录、可只建 Python 载荷」）扩展为 `DSH_DESKTOP_BACKEND_ROOT`——桌面
`prepare-dsh.ts` 在 staging 时把 `backend/` 整树拷进产物 `resources.dsh/backend/`。

## 3 · 桌面拉起生命周期（DesktopBackendHost）

`apps/desktop/src/backend-host.ts`（新）：

1. **启动时机**：main 进程 ready 后、首窗口加载前；`enabled()` 门＝`resources.dsh/backend/backend-manifest.json`
   存在（与 update-coordinator 的存在性门同构——开发态/无后端产物时结构性关闭）。
2. **安装幂等**：首次（或 manifest 指纹变化）时 spawn `runtime/python install-backend.py`，
   目标 venv＝`%LOCALAPPDATA%/taiji/backend-venv`；安装流转发到渲染层的启动面板（进度可见）。
3. **拉起 runtime**：venv 就绪后 spawn `python -m api.main`（cwd＝venv 挂载的 runtime 树），
   端口策略＝**127.0.0.1 固定基址＋占用时顺延**（8000→8010），实际端口写 `%LOCALAPPDATA%/taiji/backend-endpoint.json`。
4. **健康等待**：`/api/health` 轮询（200ms 起步指数退避，上限 5s，总窗 120s）——就绪前 Life/模型面板
   显示「后端启动中」而非「不可用」。
5. **退出/崩溃**：app 退出时 kill 子进程树；runtime 崩溃退避重启（3 次内），第 4 次失败记入
   诊断面并提示日志路径。`hmr-live` 式共享 runtime（8000 已被占用）时：**顺延端口 + endpoint 文件**
   已天然兼容开发态（harss web 经 llm-taiji baseURL 配置指向任一端口）。

## 4 · 体积与许可账（实现期先报数再定稿）

- wheelhouse 只收 **CPU wheel**（torch+cpu 轮子约 200-300MB 压缩）＋后端直接依赖；
  预计安装后增量 **0.8-1.2GB**（装前报实测数，owner 保留否决点——裁定回执已记「体积上限未给数，实现期先报实测」）。
- 许可：wheelhouse 元数据里逐 wheel 记 license 字段；`verify-third-party-notices` 的对账口径在
  实现期对齐（torch 的 BSD＋依赖族已在既有 notices 脚本覆盖面内，实施时核验）。

## 5 · 实现分刀（每刀独立可验）

| 刀 | 内容 | 验收 |
| --- | --- | --- |
| P1-① | `prepare-backend-wheelhouse.ts`：从后端 lock 生成 wheelhouse（CPU 轮子）＋清单 | 本地产出 wheelhouse；manifest 记录指纹 |
| P1-② | `install-backend.py`＋staging 拷贝（`prepare-dsh.ts` 加 backend 树） | 干净目录上幂等装通 venv |
| P1-③ | `backend-host.ts` 生命周期（启/等/退避/退出）＋endpoint 文件 | 无后端产物时零行为变化；有产物时拉起并健康就绪 |
| P1-④ | 面板「后端启动中」态接线（ui-life 读取 endpoint 文件/健康） | 判据③面在装机产物上可复现 |

## 6 · 风险与既知约束

- 外部下载稳定性（08 ⑱ R4 条）：wheelhouse 的**生成**在本机做一次（可用镜像环境变量），
  产物随发布物走 ⇒ 用户侧零外部下载；生成机网络问题只影响出包，不影响装机。
- 体积上限：wheelhouse 不进 public-index（独立目录）⇒ 不触发 release workflow 的 size 门，
  但发布物总体积增长需在发布评审报备。
- 与共享 runtime 的端口协调：固定基址顺延＋endpoint 文件（见 §3.3），`hmr-live` 场景天然兼容。
