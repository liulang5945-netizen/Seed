# Seed 公测（Public Beta）路线图

> 依据：`reports/structure_cleanup_20260823.md`（结构整理）、成熟封装五维分析（2026-08-23）、
> 训练配套对比评估（2026-08-23）。本文是公测前唯一权威执行计划；各阶段完成判据全部可量化。

## 0. 现状基线（2026-08-23）

- 模型：`checkpoints/seed_corpus.pt`，800K raw-byte 训练，byte_ppl 23.1，A1–B1 判据全闭合。
- 产品接入：api / 前端 / 桌面端 / 移动端远程接入已打通（阶段 4/5 完成），回归 108 绿。
- 语料储备：`data/simple_zh/simple_zh_texts.jsonl`（1.3GB，约 10 亿字节，用户指定为公测大预算训练语料）。
- 诚实边界（plans/README.md）：**byte-level 生成尚未到人工可读**——这是公测的头号阻塞。

## 1. 公测标准定义（五维，全部可量化）

### 1.1 模型能力
| 指标 | 现状 | 公测门槛 | 测量方式 |
|---|---|---|---|
| holdout byte_ppl | 23.1（800K 训练） | ≤ 8.0 | `eval_seed_corpus.py --holdout-rows 256` |
| 字节轮次准确率（对话 holdout） | — | ≥ 55% | 同上的 panel 统计 |
| 对话人工可读率 | ~0%（不可读） | ≥ 60% 回复基本可读 | 50 条固定题集盲评脚本 `verify_seed_beta_dialogue.py`（M2 新建） |
| 多轮上下文 | 基底持久状态已验证（N7/N8） | ≥ 3 轮历史在回复中被正确引用 ≥ 60% | 同上题集的多轮子集 |
| 响应连贯性 | — | 无乱码字节率 ≥ 99%（UTF-8 有效解码） | 自动校验 |

### 1.2 训练规模（M1）
| 指标 | 公测门槛 | 判定 |
|---|---|---|
| 大预算训练 | 在 `simple_zh_texts.jsonl` 上累计 ≥ 1 亿符号（或预算耗尽时的最大可达值，须 ≥ 2000 万） | 进度曲线落盘 `reports/seed_beta_progress.jsonl` |
| 收敛性 | 后 20% 符号窗口的 holdout_surprise 单调不升，且低于前 20% 窗口 ≥ 30% | 进度曲线自动分析 |
| 规模画像 | `--scale` 不低于当前 800K 检查点所用画像 | 检查点 metadata 记录 |

### 1.3 工程稳定性
| 指标 | 公测门槛 |
|---|---|
| API 可用性 | 连续 1 小时压测（≥1000 次对话请求）成功率 ≥ 99%，无进程崩溃 |
| 检查点恢复 | 模拟崩溃恢复 10/10 成功；原子落盘（临时文件 + rename），无半写文件 |
| 异常处理 | 无效输入 / 超长输入 / 并发请求不产生 500 未捕获异常 |
| 回归门禁 | `pytest tests/ -q` 全绿（设计内跳过除外）+ 8 个 `verify_taiji_*` 全过 |

### 1.4 性能基线
| 指标 | 公测门槛 | 测量 |
|---|---|---|
| 首字节延迟 | ≤ 2 秒（256 字节回复） | `verify_seed_beta_perf.py`（M3 新建） |
| 生成吞吐 | ≥ 200 bytes/s（单机当前硬件） | 同上 |
| 运行时加载 | ≤ 30 秒 | 同上 |
| 性能守卫 | 万 tick 吞吐 ≥ 基线 80%（基线在 M1 后钉住） | 回归测试 |

### 1.5 前端 / 桌面端
| 指标 | 公测门槛 |
|---|---|
| 安装与启动 | 安装包一键安装；桌面端双击启动 ≤ 15 秒出界面；首次对话可用 |
| 核心交互 | 对话 / 历史 / 模型切换 / 设置全链路可用，无控制台错误 |
| 界面品质 | 通过对照清单评审（布局 / 间距 / 暗色模式 / 流式输出体验），达到项目既有设计标准（`docs/design/taiji-front-redesign/`） |

## 2. 差距分析与优先级（依赖排序）

```
M1 大预算训练 ──┬─→ M2 对话能力评测 ─→ M5 公测发布
M0 工程硬化 ────┼─→ M3 服务稳定性 ───┤
                └─→ M4 前端联调 ──────┘
```

| # | 阻塞项 | 来源 | 依赖 | 优先级 |
|---|---|---|---|---|
| 1 | 大预算训练未跑（头号阻塞：生成不可读直接源于训练量） | 训练配套评估 | 无 | P0 |
| 2 | 检查点非原子写、无元数据 | 成熟封装缺失 | 无（训练期间落盘前必须完成） | P0 |
| 3 | 无对话能力量化评测面板（现有 verify_seed_* 是机制判据，不是对话质量） | 公测标准 1.1 | M1 | P1 |
| 4 | 无性能基线与守卫 | 成熟封装缺失 | M1（基线钉住） | P1 |
| 5 | API 无压测 / 异常处理未审计 | 公测标准 1.3 | M0 | P1 |
| 6 | 前端界面品质未对照评审 | 公测标准 1.5 | 无 | P2 |
| 7 | 公测发布报告与用户文档 | 交付物 | M1–M4 | P2 |

**架构级风险声明**：若 M1 中期检查点显示 byte_ppl 下降趋势不足以逼近门槛（如 1 亿符号后仍 > 15），
属架构级决策点（加大画像 / 换语料混合比 / 承认公测降级），**暂停并向用户汇报**，不自行改架构。

## 3. 分阶段执行计划

### M0 工程硬化（先行，训练期间落盘依赖它）✅ 2026-08-23 完成
- [x] 检查点原子落盘：临时文件 + `os.replace`（`seed/persistence.py`，train_seed_corpus / api save 两处接线）
- [x] 信封 metadata：tick / 语料指纹 / 画像 / 时间戳（向后兼容，`restore()` 不要求 metadata）
- [x] 判据：新增 6 项测试（`tests/seed/test_seed_persistence.py`）；全仓 113 passed, 4 skipped

### M1 大预算训练（进行中）
- [x] 吞吐基准测量：~330–365 符号/秒（scale 2，CPU），1 亿符号 ≈ 3.2 天
- [x] 在 `simple_zh_texts.jsonl` 上启动长训（后台，`--checkpoint-every 2000000`，进度落 `reports/seed_beta_progress.jsonl`）
- [ ] 判据：§1.2 三条全过；训练期间每 1000 万符号记录一次收敛快照

### M2 对话能力集成（评测面板已就绪）
- [x] 新建 `verify_seed_beta_dialogue.py`：50 条固定题集（单轮 30 + 多轮 10 组×2 轮），输出可读率/UTF-8 有效率/多轮引用率/轮次有效率；四项判据与 §1.1 对齐，失败非零码退出；800K 基线：可读率 0%（头号差距，印证诚实边界）
- [ ] 以 M1 检查点替换 `checkpoints/seed_corpus.pt`（原子替换 + 旧版归档）
- [ ] 判据：§1.1 五项门槛全过

### M3 服务稳定性（判据全部达成 ✅）
- [x] `verify_seed_beta_perf.py`：加载 0.17s / 首字节 0.13s / 吞吐 352 bytes/s（800K 基线，训练并行争 CPU 下仍全过；守卫基线已钉住 `reports/seed_beta_perf_baseline_800k.json`）
- [x] API 压测 `verify_seed_beta_api_stress.py`：1000/1000 成功（成功率 100% ≥ 99%），平均延迟 0.70s，无 500；异常输入 6 类（空/100K 超长/64 轮历史/Unicode 边界/标记注入/空字节）全部受控（`reports/seed_beta_api_stress_20260823.json`）
- [x] 异常处理硬化：超长输入截断防护（`MAX_PROMPT_CHARS=2048`，提示+历史总预算，保留最近轮次）——实测 100K 提示从 309s 降到封顶成本；全仓 113 绿验证无副作用
- [x] 检查点崩溃恢复演练 `verify_seed_beta_recovery.py`：10/10 轮全过（三场景×10：序列化中途崩溃目标完好且无 .tmp 残留 / 半写遗留后干净替换+状态逐位一致 / 目标截断被明确拒绝且修复后可用）；演练暴露真实缺陷并修复：`atomic_save` 新增陈旧临时文件自愈清扫（`reports/seed_beta_recovery_20260823.json`）
- [x] 判据：§1.3 + §1.4 全过（压测 1000/1000、恢复 10/10、性能三项达标、回归全绿）

### M4 前端联调（页面评审通过，余桌面端冒烟）
- [x] 三轮浏览器评审：7 页面渲染/间距/观感达标；首轮暴露 4 项后端阻断（`runtime/status` 500、切换被 429、工具注册崩溃、轮询风暴）全部修复并复验；对话主路径、运行时切换+持久化+重启自动恢复、五页零 console error 实测通过（`reports/m4_frontend_review_20260823.md`）
- [x] 顺带补齐缺失实现层：`runtime_service`/`tool_service`/`tool_registry` 真实实现、`neuroplex.infra` EventBus 包、`start_auto_reload`、`/api/train/checkpoints`、`/api/rag/stats|clear`、限流 100→600/分钟、前端轮询收敛（成熟封装分析 12 项缺失中的实现层欠账部分已清）
- [x] 暗色模式对比度测量：「深邃暗色」主题采样 34/33 元素，最低对比度 4.91:1，全部 ≥ WCAG AA 4.5:1，无低于阈值项（截图 `_scratch/dark_01..03_*.png`）
- [x] `/api/train/resume_checkpoint` 流式续训端点（`api/training/resume.py`）：SSE 契约对齐前端 `resumeFromCheckpoint`（hardware_diag/progress/completed/`[DONE]`），小预算冒烟通过（续训 2000 ticks、检查点落盘、锁与停止联动），回归 113 绿
- [x] 桌面端开发入口冒烟（`desktop/main.py`）：窗口启动→后端拉起→Seed 激活→8765 ws→前端加载→对话全通；
  修复四项真实缺陷：子进程 PIPE 阻塞挂起（改日志文件重定向）、看门狗重启被陈旧 `_running` 标志短路、
  ws 服务依赖已删的 `neuroplex.core.api`（重写为 Seed 原生：EventBus 推送+状态+HTTP 聊天转发）、
  `start_taiji.py` 幽灵引用（改 `python -m neuroplex.core.websocket_server`）；就绪探测增加子进程 PID 校验；回归 113 绿
- [x] 桌面端双入口打包冒烟（1121.7MB，`Seed.exe`+`SeedBackend.exe`）：启动 2.9s 出界面、`/api/health` ok、`seed_active=true`、浏览器对话实测通过；修复数据资产缺失（见日志五续）
- [ ] 判据：§1.5 全过

### M5 公测发布
- [x] 公测发布报告主体 `reports/seed_public_beta_release_20260823.md`（草稿）：五维证据+已知限制+用户文档要点已填，依赖长训的项已标记，待 M1 恢复后复测定稿
- [x] 用户文档：README 新增 Public Beta 段落（入口/环境变量/证据索引）+ `docs/seed_public_beta_user_guide.md`（安装/对话/训练/环境变量/已知限制/FAQ）
- [ ] 判据：五维门槛证据齐全，工作区干净，推送云端

## 4. 持续推进机制

1. **阶段门禁**：每阶段收尾跑 `pytest tests/ -q` + 8 个 `verify_taiji_*`，全绿方可进入下一阶段。
2. **无回退原则**：任何阶段出现既有判据（M5–M7、A1–B1）回退，先修回退再继续。
3. **架构决策点**：改 `taiji/` 学习律 / 检查点格式破坏性变更 / 公测门槛降级——一律暂停汇报。
4. **进度留痕**：训练进度曲线、各阶段验收报告落 `reports/`；本文件的 checkbox 随进展更新。

## 5. 进度日志

### 2026-08-23 回归门禁排查（verify_taiji_m5 / n10 失败）
- **n10 根因（已修复）**：`f30729c`（8/22 19:23）有意把稀疏核衰减改为资格门控（防 800K 蒸发崩塌，当时 97 项全绿），
  但 `verify_taiji_n10` 的稠密参考式仍用旧全局衰减 → 恒差 5.4e-4。已把参考式对齐内核现行语义，
  算子等价误差降为 0.0（判据阈值 1e-6 未动，非降级）。
- **m5 根因（已修复，用户已批准）**：`e3dc6af`（8/23 00:59）引入 `bounded_reward = tanh(reward)`，
  m5 协议的 reward=1.0 被压到 0.762，单次曝光动作读出削弱 24%，7/8 → 6/8，恰好跌破 0.875 阈值。
  分离实验（seed 23）：撤销 tanh → 1.0 PASS；仅撤销冗余门控 → 0.875 仍贴线。
  修复：`bounded_reward = clip(reward, -1, 1)`（`taiji/memory.py`）——保留“有界”意图且恢复 [-1,1] 单位斜率。
  配套：`tests/seed/test_seed_sleep.py` 的巩固场景改用自评最差模式（界函数饱和区，对界形状不敏感）。
  验证：CI 8 项 + A1-B1 5 项全部 PASS（证据 `reports/_gate_clip_repair.log`），pytest 113 绿，
  真实模型内生回放 accepted 48/48（mean_priority 0.22）——机制无回退。
- 排除项：本轮回退（M0 只动持久化层）、CPU 争用、线程数抖动、影子包导入（`taiji.__file__` = E:\Seed）。
- M1 长训已重启（后台，截至日志时 70 万 ticks，holdout_surprise ≈ 2.9 区间）。
- 回归门禁恢复全绿，公测推进不再受阻。

### 2026-08-23 M3 收官：检查点崩溃恢复 10/10
- 新建 `verify_seed_beta_recovery.py`（10 轮×3 故障注入场景）。首轮 10/10 FAIL 排查：
  ① 脚本误用原始字节比较判“干净替换”——torch.save zip 存档字节每次保存必变（archive 名含对象 id），改为加载后状态等价（tick + score_bytes 逐位）；
  ② 暴露真实缺陷：进程被杀留下的半写 `.tmp` 残留，`atomic_save` 下次保存不清理。
- 修复：`seed/persistence.py` atomic_save 落盘前清扫同目录陈旧 `<目标名>.*.tmp`（目录自愈）。
- 复跑 10/10 PASS；pytest 113 passed / 4 skipped 全绿。**M3 判据全部达成**，仅余 M1 训练完成后的性能守卫基线复核。
- 下一步：M4 前端评审（按 `plans/active/M4_FRONTEND_REVIEW_CHECKLIST.md` 起服务逐页对照）。

### 2026-08-23 M4 前端评审：三轮浏览器评审 + 实现层欠账清理
- 首轮暴露阻断链：`runtime_service` 零参 stub 被 1 参调用 → 500 → 健康态永不 connected → 发送按钮恒禁用；
  全局安全限流 100/分钟 < 前端轮询量 → 连带 429，切换 POST 也被拦；工具注册表空 stub + `ToolDef` 不认 `func=` → 注册崩溃。
- 修复（全部真实实现，无假象）：`runtime_service`（六段聚合+逐段防御降级）、`tool_registry`/`tool_service`、
  `neuroplex.infra`（EventBus + 事件审计订阅）、`start_auto_reload`、运行环境偏好持久化（重启自动恢复 Seed）、
  `/api/train/checkpoints`、`/api/rag/stats|clear`、限流 600/分钟、前端轮询去重降频。
- 复验：三轮浏览器评审——对话主路径走通（首问 ≈11s 冷启动，乱码为 800K 检查点预期）、
  五页横幅清零、console error 0、切换+重启恢复全通；评审报告 `reports/m4_frontend_review_20260823.md`。
- M1 长训同期健康：2.8M ticks（holdout_surprise 2.84 持续下降），`checkpoints/seed_beta.pt` 定期落盘中。
- 每轮修复后回归全绿（113 passed）。
- 2026-08-23（后续）：`/api/train/resume_checkpoint` 续训端点落地并冒烟通过；暗色对比度测量达标（最低 4.91:1）；
  M1 长训 3.0M ticks 健康（holdout_surprise 2.89）；frontend/dist 已重建；余桌面端冒烟。
- 2026-08-23（再续）：桌面端开发入口冒烟全通（窗口/后端/Seed/8765 ws/前端/对话），修复四项桌面层缺陷（见 M4 节），
  依赖补齐（PyQt6/WebEngine/PyInstaller/websockets 13.1）；`desktop/build.py` 打包执行中。
- 2026-08-23（四续）：
  - **ws 接线**：前端 `useWebSocket` 原为死代码（全仓无调用方），已在 `App.vue` 接入（自动连接 8765 + 生命事件路由到 `runtimeStore.handleLifeEvent`），
    重建前端并浏览器复验：页面零 JS 异常、welcome 消息收到、8765 出现真实连接。已知限制：ws 与后端分属不同进程，
    生命事件跨进程推送需事件桥，公测由前端 10s 轮询兑底。
  - **打包首轮完成（1076MB）但冒烟暴露致命缺陷**：frozen 模式下 `sys.executable` 即 `Seed.exe` 自身，
    `[Seed.exe, -m uvicorn]` 递归拉起整个 GUI（日志多进程交错为证），8000 永不监听且资源风暴致系统卡死。
    修复：`desktop/main.py` frozen 分支改为进程内守护线程跑 uvicorn/8765 ws（与 `api/run_app.py` 打包设计一致），
    ROOT_DIR 改以 exe 目录为准，就绪探测增加线程存活判据；重新打包中。
  - **长训中断与续训机制验证**：系统重启后从 `seed_beta.pt`（4M）恢复；暴露并修复训练脚本续训计数器清零问题
    （`ticks` 改以 `model.tick` 为基线，`max_symbols` 语义改为续训增量），复跑后进度从 4.82M 正确继续。
    用户指示长训暂停、之后进行；检查点与续训命令已验证可靠。
- 2026-08-23（五续）：
  - **双入口打包方案落地**：进程内线程方案倒在 `Unable to configure formatter 'default'`（主入口已 basicConfig，
    uvicorn 再 dictConfig 同名 formatter 冲突）；multiprocessing spawn 方案倒在 frozen onedir bootstrap 卡死；
    最终改双入口单包：新增 `desktop/backend_worker.py`（`SeedBackend.exe <host> <port>` → uvicorn），
    `seed.spec` 双 Analysis + MERGE（新版需三元组）+ 双 EXE 共享 `_internal`，主进程 `Popen SeedBackend.exe`。
  - **数据资产缺失修复**：首轮冒烟 `/api/health` 500——PyInstaller 只收集 .py，纯数据文件未随包。
    `seed.spec` `datas` 补齐 `tokenizer_contract.json`、`domains/*.model`、`checkpoints/seed_corpus.pt`；
    补齐后 cortex 装配成功（MaturityTracker/SleepConsolidator/Neuromodulator 全启用）。
  - **打包冒烟全通**：启动约 2.9 秒出界面（后端 1.5s 就绪）、`/api/health` ok 且 `seed_active=true`、
    life 状态/6 工具注册正常；浏览器发“你好”收到回复并正常渲染（乱码为 4M 检查点预期），
    控制台 0 error，截图 `_scratch/verify_8000_chat_nihao_reply.png`。M4 §4 全部勾选，M4 判据达成。
- 2026-08-23（六续）：
  - **M5 交付物落地**：公测发布报告主体 `reports/seed_public_beta_release_20260823.md`（五维证据/已知限制/文档要点，
    依赖长训项已标记）；用户文档 `docs/seed_public_beta_user_guide.md` + README Public Beta 段落（身份守护回归 3 passed）。
  - **回归门禁新鲜复跑**：CI 8 项 `verify_taiji_*` 在打包/文档变更后全量重跑，8/8 `status: pass`
    （证据 `reports/_gate_full_20260823.log`）；pytest 113 passed / 4 skipped。除 M1 长训外无回退风险。
  - **对话面板 4.8M 复测**：对 `seed_beta.pt`（4.82M ticks）复跑 50 题面板，自动指标仍 0%（未达阈），
    但回复从字符堆叠质变为带段落/多轮格式的结构化伪对话（“老师：”“诗”“杜甫”），收敛方向健康；
    产物 `reports/seed_beta_dialogue_4m8_seedbeta_20260823.json`，发布报告 §2.1 已更新为双基线对照。
  - **性能基线 4.8M 复测**：加载 0.138s、首字节 0.071s、吞吐 572.2 B/s，3/3 达标且优于 800K；
    产物 `reports/seed_beta_perf_4m8_seedbeta_20260823.json`，发布报告 §2.4 双基线对照、§5 待办已勾。
  - **工作区清理与提交预案**：删陈旧根目录 `Seed.spec`（被 `desktop/seed.spec` 双入口取代）；
    `.gitignore` 补 `taiji/play_data/`（运行时数据）与 `!reports/_gate_*.log` 例外（门禁证据入库）；
    52 项变更的 5 段式提交计划落 `docs/history/commit_messages/commit_msg_public_beta_m3_m5.txt`（待用户批准后执行）。
- 2026-08-23（七续）：
  - **检查点提升**：`seed_beta.pt`（4.82M ticks）复制为运行时默认 `checkpoints/seed_corpus.pt`
    （旧文件备份为 `seed_corpus_prev_20260823.pt`），并同步到打包目录 `dist\Seed\_internal\checkpoints\`，
    用户打开即用最新检查点；发布报告 §1 已同步。
  - **`is_taiji:false` 疑点定论（设计预期，非回退）**：`data/runtime_preference.json` 持久化 `runtime:"seed"`，
    `api/app.py` 启动时按偏好直接恢复 Seed 原生运行时、跳过 Cortex 装配（故无装配日志），
    `is_taiji()` 仅对 Cortex 实例为真；早前 `is_taiji:true` 为当日首次启动尚未切换时的 Cortex 初态。
  - **提升后对话验收**：打包版 `/api/health` ok（`seed_active=true`、`model_name=seed:seed_corpus.pt`）、
    life/6 工具正常；`/api/chat/stream` SSE 200，final 回复含“诗”等 4.8M 质变信号（乱码为早期检查点预期），
    对话链路在提升后检查点上正常。
  - **交付物证据审计**：13 项交付物/证据文件全部在位；检查点三处（`checkpoints/seed_corpus.pt`、
    `dist\Seed\_internal\checkpoints\seed_corpus.pt`、`seed_beta.pt`）SHA256 一致（提升无歧义）；
    门禁日志实测 8 处 `status: pass`、0 处 fail/error；双入口 EXE 在位且进程运行中。
    除 M1 长训（用户指示暂停）与提交批准外，公测准备工作全部就绪。
- 2026-08-23（八续）：
  - **5 段式提交已执行**：73358ff（fix substrate）→ c81cd86（persistence）→ a62aedb（api）→
    5e8932b（desktop）→ d08db4a（docs），57 项变更全部归类入库，工作树干净；推送待长训定稿。
  - **首问延迟实测校准**：发布报告 §3.3 原载 10-21s 为偏好持久化落地前的旧数据；复测冷启动后
    首问首字节 27ms、完整回复 1.3s，已按实测更新报告。
  - **WinError 5 排查定论（环境限制，非产品缺陷）**：沙箱受限终端拉起的 Seed.exe 在
    Popen SeedBackend.exe 时被拒（日志 20:31 起重试循环）；直接启动 SeedBackend.exe 均成功且
    服务正常（/api/health ok、对话 200）；同日早前同链路冒烟 3 次全通。用户环境双击启动不受影响。
  - **续训语料漂移防护（本轮新增）**：排查发现续训入口缺省语料为对话小语料（108MB），
    与 M1 大预算基线（1.3GB `simple_zh_texts.jsonl`，`seed_beta.pt` 元数据指纹为证）不一致，
    照文档照抄会跑偏。修复：①用户指南续训命令显式补 `--corpus` + 指纹核对提示（顺带校准首问延迟旧数据）；
    ②`/api/train/resume_checkpoint` 新增语料漂移预警：检查点元数据指纹与本次续训语料不一致时，
    SSE 首个事件发 `warning`（不阻断，混合/换语料续训可能有意为之），已冒烟验证（预警+基线 4.8M 正确）；
    回归 pytest 113/4 全绿。待用户决策：是否将缺省语料直接改为大预算语料（当前仅预警不改默认）。
    - 2026-08-24：**M1 长训已启动**（用户指示“开始长训”）。命令：`python scripts/training/train_seed_corpus.py --resume checkpoints/seed_beta.pt --corpus data/simple_zh/simple_zh_texts.jsonl --max-symbols 95200000 --checkpoint checkpoints/seed_beta.pt --progress reports/seed_beta_progress.jsonl --checkpoint-every 2000000 --progress-every 100000 --scale 2`（PID 9944，日志 `logs/long_train_20260824.log/.err`）。
      启动验收：续训基线正确（首条新进度线 ticks=4900000、无重启）；指标稳定（mean_surprise 2.56 / holdout 3.05 / accuracy 0.348）；吞吐 ≈310 符号/秒；
      ETA ≈3.4 天（08-27 晚），完成时累计 ≈100M 符号，满足 M1 判据。
      待办：完成后检查点提升 `seed_corpus.pt` + 对话/性能复测 → 报告定稿 → 漂移防护批次（含用户手工 `resume.py` weights_only 加固）终稿提交+推送。
    - 2026-08-24（同日）：**长训已按用户指示停止**。实际运行 ≈9.25 小时，进度推进至 16.8M ticks（累计符号 ≈16.8M，均吞吐 ≈360 符号/秒，holdout_surprise 3.02→2.96 持续下降）；
      检查点已安全落盘：`checkpoints/seed_beta.pt` 元数据 tick=16000000、语料指纹 `simple_zh_texts.jsonl` 正确（10:23 写入，原子落盘无损）。
      后续约束：用户尚未摸清 Taiji 构建细节，**长训不得恢复**，待用户完成构建细节梳理后再规划训练（届时从 16M ticks 检查点续训）。
