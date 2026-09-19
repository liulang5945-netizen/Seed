# Seed / Taiji 长期工作记忆

> 只保留**跨轮次仍有效**的契约、纪律与判据；过程细节与已结项阶段见各结项文档（§6 有索引）。
> 最后整理：2026-09-18。

## 1. 架构契约（机器强制，不得放宽凑绿）

- `taiji/` = 自足认知基底：**禁止导入** seed / seed_platform / neuroplex / transformers（含传递性），
  由 `tests/taiji_native/test_architecture_contract.py` 与 `test_naming_boundary_contract.py`
  **AST 级**强制（`ast.walk` 连函数级导入都抓）。
- 共享测量仪器放顶层 `instruments/`：依赖方向 **instruments → taiji 单向**（仅 `content_digest`），
  `taiji/` 对 instruments 零引用（duck-typed 参数）。`DocumentEmbedder` 在
  `instruments/document_embedding.py`；语义 encoder 的 embedder **必须显式注入**（fail closed）。
- `neuroplex/` = 冻结 Transformer 基线，**不得** import seed/taiji（单向替代关系）。
- 语义 checkpoint 红线：`taiji-document-embedder-v1` 的 payload 格式与 digest 锚**不得变更**。

## 2. 环境硬约束（动手前必读）

- Python 用 `C:/Users/23747/AppData/Local/Programs/Python/Python312/python.exe`
  （managed 3.13.12 **无** torch/ruff/black）；路径写正斜杠。
- **bash 外部命令全部缺失**（`ls`/`cat`/`grep`/`head`/`tail`/`mkdir`/`rm`）：
  文件操作用 Read/Write/Edit/Glob/Grep 工具；目录用 `python -c`；**提交信息用
  Write 写 `.git/COMMIT_MSG_*` 再 `git commit -F`**。
  - ⚠️ **反引号在任何 shell 引号里都会被当命令替换**：`git commit -m "…\`x\`…"` 会失败/被吃掉
    （已踩 2 次），**`python -c "…\`x\`…"` 同样**（2026-09-18 踩第 3 次，补丁静默没生效）。
    **凡内容含反引号/代码片段，一律用 Write 写文件或 Edit 工具，不要塞进 `-c`/`-m`。**
  - ⚠️ **管道里用这些命令会连带杀掉上游进程**：`... | tail -c 900` ⇒ `tail: command not found`
    ⇒ 管道破裂 ⇒ **Python 进程被 SIGPIPE 带走、报告没生成**（2026-09-18 白跑 9 分钟训练）。
  - **纪律**：输出过滤**必须用 `| python.exe -c "..."`**；`for`/`|`/重定向可安全使用。
- **严禁 heredoc**；`python -c` 内避免长中文（shim 逐行执行）。
- **跑 `tests/` 必须设 `CODEBUDDY_SAFE_DELETE_ENABLED=0`**：沙箱 safe-delete 批删守卫
  （按「单次 tool call 内删除 ≥50 路径」计数）劫持 `Path.unlink`/`os.remove` ⇒ 大批 gate 用例**假失败**。
  `dangerouslyDisableSandbox` **不能**绕开它。
- 全量 pytest ~15 min 会 SIGTERM ⇒ `run_in_background` 或分批。
- **`SystemExit` 级联先拿栈再归因**（`--tb=long -o junit_logging=all`），别用"共享状态/顺序污染"猜。
- `default_workspace_root()` 取 `get_setting("workspace_path")`；**凡读仓库文件的 gate/测试，
  `SeedRuntime.load(...)` 必须显式传 `workspace_root=PROJECT_ROOT`**（load 不继承构造时的 override）。
- **黑名单**：`reg.exe` 被安全策略拦截，不可绕过。

## 3. 工作流与文档纪律

**流程**：预注册 → 实现 → 门禁 → 报告 → 提交。负结果**如实落账**，绝不改绿；
报告对机制的描述必须与代码同步。

**改文档按类型分（不是"只追加 vs 改上文"二选一）**：

| 类型 | 处理 |
|---|---|
| 冻结判据 / 预注册（阈值、读数映射、停止线）| **只追加，永不改**；新结论写**新预注册** |
| 冻结证据 / 历史报告（结果快照）| **只追加**；被覆盖则恢复归档版 + 新版本另存 |
| **导航 / 状态文档**（如 `03_CURRENT_EXECUTION.md` 的"唯一下一步"）| **必须改**（它唯一的价值就是"当前"）|
| 原文写错（笔误、错误事实）| **改 + 显式注明原值** |

追加以**新 §** 形式，并在长文档顶部加「最新状态指针」指向最新 §。

**其他硬纪律**：
- 归属证明用**引用图检查**（谁 import 被改模块），不靠反复重跑。
- 回归测试必须**双向钉住**（能过 + 该拒绝的拒绝）。
- 测试改完**必须读回全文/跑测试**：Edit 可能匹配到错误缩进位置并"成功"，造成断言被嵌套而静默失效。
- 临时探针**用毕即删**（否则会污染清单/扫描型结论）。
- 同一文件多处替换**必须串行**（并行 Edit 会丢更新，两条都报 success）。
- **链接前缀**：`plans/reference/*.md` → `../../`；`plans/active/roadmap/*.md` → `../../../`；
  写完跑一次链接校验（解析所有 `](path)` 是否存在）。
- **正结果必须先问"能否在相邻设置复现"**；交互效应声明前**必须做规模扫描**。
- **写进代码的预期值必须有可否决通路**（否则是装饰性字段）；检验"合同是否接受某任务"要用
  **与规则无关但含合同层的脚本化执行**。
- ⚠️ **`counterfactual._run_scripted` 不能判合同合法性**：它直调 `execute_tool`，绕过
  `environment.policy_for` 发出的拦截 ⇒ 会把"没人能合法执行"的格认证为合法、
  使 0 增益被误读为结构结论（2026-09-13 实测踩中）。
- **写进代码的预期值必须有可否决通路**（否则是装饰性字段）；检验合同是否接受某任务要用
  **与规则无关但含合同层的脚本化执行**。
- ⚠️ ** 不能判合同合法性**：它直调 ，
  绕过  发出的拦截 ⇒ 会把没人能合法执行的格认证为合法、
  使 0 增益被误读为结构结论（2026-09-13 实测踩中）。

## 4. R2 诊断判据（可复用，本系列最有价值的部分）

**(4a) 边际退化判据** —— 「teacher-forced 指标改善、自主生成 `exact` 恒 0」已出现多次。
**决定性判据 = 生成内容是否随输入变化**：若**不同输入生成同一字节串**且它等于**训练集众数**，
则学到的是**边际分布而非条件分布** ⇒ **前向/机制/生成态层的解释全部不成立**，问题在**训练信号语义**。
诊断法极廉价：把 `generate()` 输出与 train split 的众数对照。

**(4b) 序列级 vs 位置级（缺口定位）** —— H-OBJ 的 `margin_gap = +4.69`（seed 917）证明
"正确材料整体更好"这一**序列级**依赖**确实被建立**，但该臂 dev `exact` 仍为 0
⇒ **"依赖已建立、不转化为产出"**（且只在 1/3 seeds 明显转正 ⇒ "依赖被建立"本身也不稳健）。
只读诊断进一步定位：缺口在**位置级** —— TF 首错位置 **`pos_0` 16/16**
（= **「prefix 状态 → 第一个回答字节」的条件化起点**）。
⇒ **序列级判断 ✓、位置级条件化 ✗，缺口在两者之间。**

**(4c) 预算分层判据** —— 一次否决只有在「**对照臂在该预算下已能产出非零 `exact`**」时才算**有效**。

| 预算 | control 臂 dev exact | 有效？ |
|---|---|---|
| 30 × 25 × 3 seeds | 0.25 / 0.0625 / 0.0 | ✅ |
| 2 × 12 | 全 0 | ❌（改动无法表达）|

**当"改动已生效"（loss/读数有非零变化）但"两臂指标逐位相同"时，正确解释是"预算不足以表达"，
不是"假设被否决"** —— 两者都必须写进报告的界限段，否则会连续误杀正确的假设。

**(4d) 抖动 vs 系统性判据** —— 单个失败门要判性质，**必须先定判据再补证**：
1/N seeds ⇒ 偶发抖动（可放行）；N/N ⇒ 系统性（不得放行）。
v3 的 loss 门是 **1/3**（→ 判抖动，matched 正常）；v4 是 **3/3 且伴随机制读数退化**（→ 系统性）。

**(4e) 仓库卫生判据（2026-09-19 实测得来，两条都踩过）** ——

- **判「某文件是否已泄漏」必须比较「在位值与历史 blob 的指纹」**，**不是**数提交个数、
  **也不是**看目录是否被 ignore。实测：`security/.storage_salt` 的路径历史有 blob，
  且**在位值指纹与已推送的 blob `5bfda9cd` 逐位相同** ⇒ 那个盐"可从远端历史取出"，
  而当时 4 条守卫全绿。反例：`security/.jwt_secret` 历史 0 个 blob ⇒ 未泄漏（此判成立）。
- **多路径合并查询不能逐路径归因**：`git log -- a b c` 把多路径混在一起 →
  曾据此误判"根目录那对也进过历史"。**必须逐路径查**。
- **守卫的结构性缺口**：只测「规则是否命中」+「跟踪清单是否为空」**测不到**
  「在位值 == 历史 blob」这一条 ⇒ 需专门补一条比对指纹的守卫（**红/绿各跑一次证明它能响**）。
- **"目录规则"不覆盖子文件**：`output/` 与 `outputs/` 各自出现"目录被 ignore 但
  `output/x.txt` NOT ignored"。**按实例名写规则会持续漏**，要按**命名约定**通配
  （`<name>-packaged-data/`、`junit*.xml`）；且**必须反向钉住**，别为不泄漏把验收证据一起 ignore 掉。
- **设计层弱点（登记，未改）**：`seed_platform/auth.py` 用**机器指纹**（`hostname|machine|USERNAME|processor`）
  作 PBKDF2 口令，其中 hostname/username **就在仓库里**（`report.xml` 里直接写着 hostname）、
  machine/processor 取值空间极小 ⇒ 一旦盐公开，密钥可离线重放。**安全性接近混淆而非加密。**

**(4f) 历史重写的隐藏陷阱（2026-09-19 实测）** ——

- **`git filter-repo` 只重写 `refs/heads/*` 与 `refs/tags/*`，不碰自定义命名空间的 ref**
  （本次是 `refs/codex/*`）。⇒ "路径已删但 blob 仍可达"的根因就是它：
  `git log --all -- <path>` 显示 0（走 diff 逻辑），而 `rev-list --objects` 仍列出该路径。
  **验证必须查「blob 是否仍可达」，不能只查「log 是否 0 次」**；处理 = 删这类 ref 后 `repack -ad`。
- **远端还有 force-push 改不到的引用**：GitHub 的 `refs/pull/*/head`。若某 PR 提交含敏感路径，
  旧对象在远端仍可能短期可达 ⇒ 彻底清除需删 PR 或联系平台支持。
- **正确做法：镜像隔离** —— `git clone --mirror` → 在镜像里 filter → 验证 → 强推；
  **主仓库全程只读**。历史重写**永远不要**在主仓库原地做。
- **改写后所有提交哈希都变**（本次 1696 个里约 67% 的前缀变化）⇒ 文档/记忆里记录的旧 sha 全部失效。
- **推前必须再验证一次**；`fetch --force` + `reset --hard` 前后要用**逐字节备份**保护未提交文件。

## 5. R2 当前状态（2026-09-18）

**当前配置**：char-v1 字级图 + induction + λ_copy=1.0，lr=0.01，microbatch 8，30 epochs。

**v3 尺度包 matched dev（train 1278 行 = 32 对象 × 12 色）—— 已结项**

| 门 | 阈值 | 实测 |
|---|---|---|
| **K1 多字值 `full_rate`（主门）** | ≥0.30 | **0.3621 ✅ 系列首次通过** |
| K2 `M4_flip` | ≥0.50 | **0.0000 ❌** |
| G1 `M3` | ≥0.26 | 0.3101 ✅（**= 表面天花板 0.1735 的 1.79×**）|
| G4 context margin | — | ✅（**`no_context_m1_max = 0.0`** ⇒ 无材料时不猜）|
| G5（D8 定义：≥2 seed >0.10）| — | ✅（2/3）|
| `first_byte` | ≥0.529 | 0.7529 ✅ |

per-seed `full_rate` = **[0.4310, 0.0000, 0.6552]**；M3 = 0.2907 / 0.1977 / 0.4419。
**⇒ 复制链已建成；`M4_flip_pair` / `fact_flip` / `combo_flip` 三 seed 全 0 ⇒ 绑定/取反未建立。**

**v4 再放大一档（train 4862 行 = 64 对象 × 24 色，28× v1）—— probe 系统性失败**

| seed | `copy_M1` | multibyte | `bias_end` | 最大 loss 跳幅 |
|---|---|---|---|---|
| 917 | 1.0000 | 1.0000 | 2.828 | +762.9% |
| **918** | **0.8470** | **0.6944** | 7.089 | **+1047.9%**（**未恢复**）|
| 919 | 1.0000 | 1.0000 | 4.828 | +124.3% |

**⇒ 规模-稳定性边界（固定 lr=0.01、固定配方）**：v1(174)→v3(1278, 7.34×) 同门 **1/3**、matched 正常；
v3→v4(4862, 再 3.8×) 同门 **3/3** 且机制层退化 ⇒ **v3 之内稳定、v4 之外不稳**。
（`within_wall_cap` 3/3 超时属"门未适配规模"，与机制无关。）

**待裁决**：(i) 承认"固定配方的规模上限"、v4 判超参域外负结果 ⇒ **转阶段收束/入口评审**（助手建议）；
(ii) 把"规模适配 lr"立为新变量（**需新预注册、分别检验**）；
(iii) 只修 wall_cap 仍放行 matched（**不推荐**：训练不稳时读数无法归因到机制）。

**可复用工具**：`probe_taiji_r2_d7_char_token.py --arm/--seed/--fixture/--tag`、
`eval_taiji_r2_d7_matched_dev.py --fixture {v1,v2,v3}/--out-report`、
`build_taiji_r2_d1_measurement_fixture.py --variant {v1,v2,v3,v4}`、
`diagnose_taiji_r2_readout.py`、`enable_epoch_shuffle`（默认关）、`enable_first_byte_weight`。

## 6. 历史归档索引（细节已移出，见结项文档）

| 阶段 | 关键结论 | 文档 |
|---|---|---|
| R2 语言路线（H3.5–H3.8、P3b-v2、H-GEN/H-OBJ/H-FBW）| 三次"加损失"在有效预算下全败；缺口不是损失设计能修的 | `plans/reference/M5_R2_*_CLOSURE_*.md` |
| 读出诊断 | **缺口 = 「prefix 状态 → 第一个回答字节」的条件化起点**（TF 首错 `pos_0` 16/16）| `M5_R2_READOUT_DIAGNOSIS_20260917.md` |
| 预算判别 | 12.5× 预算下 dev `exact` 从恒 0 变为非零 | `M5_R2_BUDGET_DISCRIMINATION_20260917.md` |
| R2-D3…D7 机制轴 | 几何→目标→因子→先验→粒度；D7 char-v1 系列最强 dev 读数 | `plans/reference/M5_R2_D*_*.md` |
| R2-D8 尺度包 | 实现门 → probe → matched（v3）→ (α) v4 | `M5_R2_D8_*.md`（含 `_ADJUDICATION_B_`、`_V4_MULTISEED_`）|
| B0/M4/N1 协作机制（9/13）| 组合机制上界、M4 反事实测量、结构空间探针 | `plans/reference/` 下 9/13 相关文档 |
| Git 恢复 | 备份在 `E:/Seed-backup-gitstate-20260913-183929/`；**确认无需回溯前不要跑 `git gc`/`git prune`** | 同上 |
