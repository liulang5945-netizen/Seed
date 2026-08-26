# Seed / Taiji substrate 旧开发路线（已归档）

状态：**已被 Taiji Native v1 路线替代**

归档时间：2026-08-25

> 本文保留架构纠正前的 R0–R7/S1–S3 路线。旧路线把 Taiji 定位为 Seed 的底层 substrate，并把 raw-byte kernel 扩展当作主研究路径；其中工程和产品证据仍可追溯，但研究执行顺序全部失效。当前唯一路线见 [SEED_DEVELOPMENT_ROADMAP_2026_08.md](../../active/SEED_DEVELOPMENT_ROADMAP_2026_08.md)。

适用范围：Taiji 原生基底、Seed 模型组合、训练与评测、CUDA、产品服务、桌面客户端和公开测试

## 1. 最终目标

Seed 要成为一个以 Taiji 为唯一原生学习基底的、可训练、可验证、可部署的智能体系统：

1. 原生认知链不依赖 tokenizer、attention、Transformer、KV cache、BPTT、optimizer 或 autograd。
2. 输入、时间状态、记忆、预测、行动、反馈和持续学习由同一个可检查的局部学习系统闭环完成。
3. 模型容量由显式参数预算和资源策略规划，不由散落在代码中的魔法数字决定。
4. 每个“智能”机制都必须有对照组、损伤实验、跨随机种子重复和可追溯证据，不能只看演示效果。
5. 默认产品运行时可在不安装、不导入 Legacy NeuroPlex/Transformer 的条件下完成原生能力；Legacy 仅保留为离线科学基线和显式兼容扩展。
6. 公开测试版必须同时达到语言质量、稳定性、性能、安全、更新和桌面体验门槛。

当前不宣称 AGI。现阶段最核心的未完成事实是：Taiji 已形成原生学习与行动闭环，但 raw-byte 自由生成尚未达到稳定的人类可读语言质量。

## 1.1 产品交付轨道（与研究路线并行但不混淆）

用户体验、工程质量和桌面发布不再作为零散收尾工作，而是与 R0–R7 研究主线合并为同一条执行路线。产品轨道按 S1 → S2 → S3 顺序推进；每个阶段完成后才进入下一阶段。

| 产品阶段 | 范围 | 目标 | 当前状态 |
|---|---|---|---|
| S1 | 工作台/知识库/生命状态/设置/聊天/工作区 | 消灭死按钮、假数据和未接线入口，把已有后端能力交给用户 | 功能验收通过，待 R0 固定 Black 复验 |
| S2 | 依赖安全、测试盲区、覆盖率、门禁 | 处置 Legacy extra CVE，封堵高风险路由和前端 composable 测试缺口 | S1 完成后执行 |
| S3 | 版本基线、打包、更新提示、回滚 | 形成可发布的 Windows portable/installer 分发闭环 | S2 完成后执行 |

产品轨道的硬约束：门禁只收紧不放松；M1 长训保持暂停；界面样式使用主题变量；设置持久化沿用本地缓存初值、GET 校正、POST 保存、失败回滚的既有范式；任何可见按钮都必须有真实行为或明确下线。

S1 的执行面固定为五组：

1. 知识库：上传、文件元数据、索引状态、预览、清空和后端兼容回退。
2. 生命状态：去掉推导假数据，恢复 feed/sleep/play/evolve，导出真实状态快照。
3. 设置：语言、时区、密度、Taiji 阈值、运行时、保留策略、导出和重置全部持久化或明确受限。
4. 聊天：快捷 chip 必须有行为；未就绪的知识库/图像入口不显示为可点击假能力。
5. 工作区：重命名使用安全路径校验、冲突返回和已打开标签同步。

S2 的首批任务为 aiohttp/datasets 补丁升级、连续两次无 HIGH 后将 pip-audit 转 blocking，以及补齐 terminal、workspace/agent/memory/MCP、training resume 和前端无测试大户。S3 的最小方案是修正版本同步正则、构建时生成 `version.json`、增加受 SSRF 防护的版本查询，并只提供“提示 + 外链下载”，不重建不存在的全量自替换 `build_scripts` 栈。

## 2. 当前基线与诚实边界

### 2.1 已经完成

- Taiji 原生链已经固定为：raw-byte sensor → hierarchical predictive fabric → distributed episodic field → sparse motor receptors → byte motor → action feedback。
- `taiji/` 不导入 `seed`、`neuroplex` 或 `transformers`；`seed/` 通过 Taiji 公共 API 组合基底。
- N0–N11、M5、M6、M7 的现有机制判据已通过；M7 已证明 replay 会把内生重建内容写回慢通路并改变后续行为。
- 当前微型参考配置有 83,841 个 active learned parameters；byte-cycle accuracy 已实测从 0 提升到 94.12%。
- 参数预算、`CapacityPolicy`、策略 JSON、训练画像和 checkpoint 恢复已覆盖 CPU/CUDA device 语义。
- 300,000 参数预算的当前规划结果为 287,322 个 active learned scalars，而不是用“神经元数量”代替真实容量。
- 当前语言训练检查点约为 16,000,000 ticks；此前直跑 100M 的计划已经停止。
- 原生训练 API、SSE、前端 Seed 训练入口以及 Legacy 开关启动矩阵已有通过记录。
- 仓库分支策略为只保留 `main`。

### 2.2 尚未完成

- 当前开发机是 CPU-only PyTorch，尚无真实 CUDA 吞吐、显存、kernel trace 和数值一致性证据。
- 16M 检查点尚未完成一次使用冻结、无泄漏评测集的完整质量审计，不能据此判断只增加训练量是否有效。
- raw-byte 自由生成还没有达到公开测试版要求的可读率、上下文使用率和 UTF-8 有效率。
- 产品壳中的部分 Agent、RAG、life、multimodal、workflow 能力仍通过显式懒加载连接 `neuroplex/`；这不污染 Taiji 核心，但说明产品层还不能删除 Legacy 目录。
- active 计划中仍有“下一步是 M7”“训练基线是 800K”等过期描述，需要统一归档或更正。
- 当前工作区存在一批尚未形成最终提交和最终验证报告的工程加固改动；任何新实验都不能把这个移动基线当成正式对照。

## 3. 边界定义

### 3.1 “完全摆脱 Transformer”的四级定义

| 级别 | 定义 | 当前状态 |
|---|---|---|
| T0 算法独立 | Taiji 学习、推理和 checkpoint 不导入 Transformer | 已完成 |
| T1 原生服务独立 | `SEED_ENABLE_LEGACY=0` 时原生 API、训练和客户端主链可启动 | 已完成，需在最终基线复验 |
| T2 默认发行独立 | 默认安装包不携带 Transformer/RAG/Agent 重依赖，所有可见原生功能正常 | 大部分完成，待发行物审计 |
| T3 产品能力独立 | Agent、知识、工具、工作流等平台能力不以 `neuroplex` 为实现宿主 | 未完成 |
| T4 可移除 | 删除 Legacy 后，正式产品、测试、文档和打包均不损坏 | 未达到，也不是当前动作 |

目标是依次达到 T2、T3，再评估 T4。不能用直接删除目录的方式伪造“独立”；Legacy 在同预算科学比较结束前仍有保留价值。

### 3.2 硬编码分类

以后所有新增常量必须属于下列类别之一：

- **协议常量**：例如 256 个 byte 值和边界 receptor。允许固定，但必须集中定义、版本化并写入 checkpoint 元数据。
- **资源策略**：区域宽度、fan-in、memory units、时间/episode 维度、学习率、阈值和预算。必须来自 `TaijiConfig`、`CapacityPolicy` 或实验清单，禁止散落魔法数字。
- **学习状态**：突触权重、eligibility trace、预测状态、情景投影和 readout。必须由数据与局部规则更新并可保存、恢复、检查。
- **产品策略**：超时、重试、速率限制、训练 rung 和发布门槛。必须在产品/实验配置中显式声明，不能混入认知算法。
- **禁止项**：数据集专用回答表、固定短语映射、按 cue 分配永久槽位、隐藏 prompt、为单个测试样例写行为分支。

任何性能用静态常量都不得携带语义行为，并必须有等价参考实现或一致性测试。

## 4. 总体阶段顺序

各阶段严格按 R0 → R7 推进。阶段内部可以并行，但未达到退出门槛不得把后续结果称为正式证据。

| 阶段 | 目标 | 预计投入（不含等待算力） | 退出结果 |
|---|---|---:|---|
| R0 | 收敛工程基线与文档事实 | 1–3 天 | clean `main`、全门禁绿、唯一事实入口 |
| R1 | 真实 CUDA 表征 | 2–4 个 GPU 工作日 | CPU/CUDA 正确性、吞吐、显存和 profiler 报告 |
| R2 | 去硬编码与容量扩展闭环 | 1–2 周 | 预算驱动、资源感知、可迁移 checkpoint |
| R3 | 建立无泄漏评测与实验制度 | 3–5 天 | 冻结数据、同预算基线、实验台账 |
| R4 | 分阶段语言训练 | 1–4 周，依赖算力 | 20M→32M→64M→100M 条件式训练证据 |
| R5 | 扩展记忆、目标与自主性 | 2–4 周 | 长时干扰、持续学习、目标保持新门槛 |
| R6 | 完成产品层原生化 | 1–2 周 | 默认发行达到 T2/T3 |
| R7 | 公开测试版闭环 | 1–2 周 | 可签名、可更新、可恢复、可观测的 beta |

这些是工程量级估计，不是发布日期承诺。R1 和 R4 的实际历时主要由 GPU 与数据吞吐决定。

## 5. R0：收敛工程基线与计划事实

### 目标

先把当前大量工程加固改动变成可审查、可复现的正式基线。没有稳定基线，CUDA 或训练结果都无法可靠归因。

### 工作项

1. 按所有权和风险审查当前未提交改动，区分：CI/质量门禁、API 安全、前端测试、Legacy 行为保持、生成报告和临时修复脚本。
2. 只在 `main` 上形成小而清晰的提交；不得混入无法解释的生成物，也不得擅自删除用户产物。
3. 依次运行并记录：
   - `ruff` 基础规则与 B/SIM；
   - `black --check`；
   - `mypy seed taiji` 与 Legacy debt ratchet；
   - Taiji/Seed 专项测试和后端全量 pytest + coverage；
   - 前端 eslint、vitest、production build 和 Playwright smoke；
   - `SEED_ENABLE_LEGACY=0/1` API 启动矩阵；
   - Windows 桌面打包和图标/托盘/窗口圆角 smoke。
4. 生成时间晚于最终 diff 的验证报告，避免引用旧报告证明新代码。
5. 统一 active 文档：
   - 把 M7 改为已完成；
   - 把 800K 旧基线更新为 16M 已暂停检查点；
   - 将本文件设为唯一“下一步”入口；
   - 过时专项计划移入 archive，不删除历史证据。

### 退出门槛

- `git status` 干净，只有 `main`，本地 `main` 与远程关系明确。
- 上述门禁全部通过；跳过项必须是硬件条件导致并有原因记录。
- 任何 lint/mypy 忽略都有责任范围、基线数字和禁止增长门禁。
- `plans/README.md` 不再同时给出多个互相冲突的下一步。

## 6. R1：真实 CUDA 正确性和性能表征

### 目标

回答“Taiji 是否真正适配 CUDA”，并用数据决定是否需要自定义 sparse kernel。当前只能说代码支持 CUDA device，不能说性能已经适配。

### 基准矩阵

- 设备：固定一台 CPU 主机和至少一张真实 NVIDIA GPU；记录 GPU、驱动、CUDA、PyTorch、精度模式和功耗策略。
- 参数预算：0.3M、1M、3M，以及显存允许的最大一档。
- 模式：observe、learn、act/free-generate、checkpoint save/load。
- 序列：短序列延迟、稳定态吞吐、长序列内存增长和睡眠/replay。
- 对照：CPU reference、现有 PyTorch sparse/index 路径；只有 profiler 证明需要时才增加 fused/custom CUDA 路径。

### 必测指标

- build/load 时间、首 tick 延迟、稳定 ticks/s、生成 bytes/s；
- 峰值 VRAM、常驻内存、host↔device 同步次数、kernel 数量与占比；
- edge utilization、无效 padding/索引开销、replay 开销；
- 同 seed CPU/CUDA 的状态误差、行为一致率和 checkpoint 跨设备恢复；
- N0–N11、M5–M7 在 CUDA 上无机制回退。

### 决策门

- 若 1M 以上预算的 CUDA 稳态吞吐相对 CPU 达到至少 1.5×，且 profiler 无单一明显瓶颈，保留纯 PyTorch 实现，优先继续扩容。
- 若加速不足或 scatter/index/synchronization 主导耗时，进入一个受限的 sparse kernel 优化支线；每个优化必须与 reference 逐步等价并有 CPU fallback。
- 不因单一吞吐数字修改学习方程。

### 退出门槛

- 一份可复现 benchmark manifest、原始结果和结论报告。
- 明确选择“继续纯 PyTorch”或“实现指定算子”，不能留下泛化的“以后优化 CUDA”。
- 给出每个参数预算的建议硬件、显存余量和训练速度预估。

## 7. R2：去硬编码与容量扩展闭环

### 目标

让模型结构由目标参数量、可用内存和明确的学习策略共同规划；扩大模型时不再手改多处尺寸。

### 工作项

1. 给 `CapacityPolicy` 增加 schema/version，并把区域、fan-in、memory、meta、time、episode 比例全部纳入单一验证合同。
2. 增加资源感知规划器：输入 active parameter budget、目标设备、VRAM/RAM 上限和安全余量，输出可实例化配置。
3. 将“规划参数数”与“实际可学习标量数”保持严格相等；所有 profile 都用测试锁定。
4. 扫描 Taiji 原生链中的数值常量，按 §3.2 分类；可配置项迁入 policy，协议常量集中，禁止项直接删除。
5. 将 threshold、learning-rate 和 replay/sleep 策略分层：结构容量与学习动力学不能共用模糊的全局开关。
6. 建立 checkpoint schema migration：旧 checkpoint 可只读识别、可显式升级、升级后结果可验证；禁止静默猜测维度。
7. 为 0.1M–10M 至少四档预算建立构造、单步、保存、恢复、跨设备测试。

### 退出门槛

- 新容量档只需要改一份 policy/manifest，无需修改算法源码。
- 未分类魔法数字扫描门禁进入 CI。
- 任意受支持预算均满足 planned count = actual learned scalar count。
- checkpoint 不因扩容或 device 切换而失去身份、数据指纹或训练 tick。

## 8. R3：无泄漏评测与实验制度

### 目标

先建立可信测量，再决定是继续扩模型、改学习机制还是增加训练数据。

### 工作项

1. 固定 train/validation/test manifest，记录文件 hash、样本来源、许可、去重方式和 split 算法。
2. 建立永不训练的 holdout；人工 50 题面板固定题库版本，但评审时隐藏模型身份和答案来源。
3. 把当前 16M checkpoint 登记为不可覆盖的 lineage 节点，核对 corpus fingerprint、配置、代码 commit 和完整性 hash。
4. 用同参数预算、同数据字节数、同评测集运行 Legacy Transformer 离线基线；Legacy 不进入在线产品路径。
5. 建立实验注册表，至少记录：假设、唯一变量、随机种子、预算、数据 hash、代码 commit、指标、损伤组、结论和失败原因。
6. 同时报告 raw 输出和任何 UTF-8 约束输出。传输层合法化不能被计作语言能力提升。

### 固定指标

- byte NLL/perplexity、next-byte accuracy、校准误差；
- 原始 UTF-8 有效率、约束后 UTF-8 有效率；
- 50 题人工可读率、相关性、重复率和三轮上下文使用率；
- 长序列记忆保持、episode 干扰、replay 收益和 lesion 降幅；
- 训练 ticks/s、能耗/时间、峰值内存和 checkpoint 大小。

### 退出门槛

- 16M 当前质量有完整报告，能区分“训练不足”“容量不足”“输出器问题”和“机制不足”。
- 相同结果可由 manifest + commit + seed 重建。
- 只有一个经证据支持的 R4 训练配置进入下一阶段。

## 9. R4：条件式语言训练阶梯

### 原则

不从 16M 直接盲跑到 100M。每一级训练都必须先达到继续门槛，未达到时回到 R2/R3 定位，不用更多计算掩盖架构问题。

### 训练阶梯

| Rung | 累计 ticks | 目的 | 继续条件 |
|---|---:|---|---|
| L0 | 16M | 审计已有 checkpoint，不训练 | lineage、holdout 和指标完整 |
| L1 | 20M | +4M canary，验证学习曲线仍有效 | holdout 改善且 N/M 门禁无回退 |
| L2 | 32M | 验证中程可读性与上下文增益 | 可读率、ppl、上下文至少两项实质改善 |
| L3 | 64M | 验证容量与数据扩展斜率 | 无明显平台期、干扰或灾难性遗忘 |
| L4 | 100M | 仅作为 beta 候选训练 | 前三级证据支持继续投入 |

### 每个 rung 的固定流程

1. 训练前锁定 commit、policy、数据 manifest、seed、device 和目标 tick。
2. 原子 checkpoint + 周期备份；恢复后验证 hash、tick 和短轨迹一致性。
3. 每个评估点只跑冻结 holdout，不把失败样本回灌同一实验。
4. 同时运行 N0–N11、M5–M7 机制回归，防止语言指标提升但行动/记忆机制退化。
5. 记录学习曲线斜率；连续两个 rung 的相对改善低于预设最小收益，或人工可读率不升，则停止扩算力并回到机制诊断。

### beta 语言门槛

- holdout byte perplexity ≤ 8.0；
- byte-turn accuracy ≥ 55%；
- 固定 50 题盲测人工可读率 ≥ 60%；
- 三轮上下文有效使用率 ≥ 60%；
- 原始或明确标注的传输约束模式下 valid UTF-8 ≥ 99%；两者必须分开报告。

这些是发布门槛，不是训练目标函数，不能通过硬编码输出规则直接优化。

## 10. R5：从语言预测扩展到可验证智能

### 目标

语言可用之后，重点不再只是降低 byte loss，而是扩大记忆容量、目标保持、环境干预和持续学习能力。

### 新机制门槛

1. **M8 记忆容量与干扰**：在 10²、10³、10⁴ episode 规模测量召回、覆盖和相似记忆干扰；必须优于时间匹配与内容损伤对照。
2. **M9 长程目标保持**：在奖励延迟和干扰事件下保持目标状态；action、value 或 replay lesion 必须显著降低成功率。
3. **M10 持续学习**：连续引入新任务后保留旧任务能力，报告 plasticity/stability 曲线，不允许为每个任务添加永久专用槽。
4. **M11 自我评估闭环**：内部判断必须能预测外部成功率，并能改变探索/学习预算；比较无判断和随机判断对照。
5. **M12 多器官组合**：文本之外的新 sensor/motor 必须复用 Taiji substrate 合同，而不是接入另一个隐藏 Transformer。

### 共同验收规则

- 至少 12 个预注册随机种子；
- 主要结论必须有无机制、内容损伤、时间损伤或随机策略对照；
- 报告效应量和失败种子，不只报告均值；
- 所有通过标准写入测试/实验脚本，不能靠手工挑选轨迹。

## 11. R6：产品层原生化与 Legacy 清理边界

### 目标

把“通用产品能力”和“Legacy 模型实现”解耦，使默认产品达到 T2/T3，同时保留可选离线基线。

### 迁移顺序

1. 先画出所有 `seed/`、`api/`、`frontend/`、`seed_platform/` 到 `neuroplex/` 的 import/runtime 图。
2. 将工具协议、知识索引接口、工作流状态、文件/终端适配器等模型无关能力迁入 `seed_platform/` 或独立 plugin adapter。
3. 每迁移一项，先加 golden/contract test，再切换原生路由，最后关闭对应 Legacy fallback。
4. `SEED_ENABLE_LEGACY=0` 下对所有正式路由做 import trace，禁止运行时触达 `neuroplex`/`transformers`。
5. 把 Legacy 依赖放入显式 extra 和独立构建矩阵；默认 wheel/desktop 包执行依赖树审计。
6. 只有当默认产品、API、训练、测试、文档和打包全部不依赖 Legacy，且同预算比较报告已归档，才讨论从仓库删除 `neuroplex/`。

### 不做的事

- 不重构冻结的 TransformerBlock 来追求代码美观。
- 不为了降低 mypy 数字大规模改写无正式消费者的 Legacy 模块。
- 不保留静默 fallback；用户选择原生模式后，缺失能力应返回明确错误或原生实现状态。

### 退出门槛

- 默认安装、API、桌面客户端和原生训练在物理移开 Legacy 包的测试环境中通过。
- 正式路由的运行时 import trace 为零 Legacy 命中。
- Legacy 只存在于 `legacy` extra、离线 benchmark 和历史兼容测试。

## 12. R7：公开测试版工程闭环

### 功能与体验

- 客户端页面只展示真实 Taiji 能力、配置、训练状态和指标，不保留与当前架构不一致的占位概念。
- 侧边栏在支持的最小窗口高度内无需滚动即可到达全部主入口。
- 桌面任务栏、窗口、托盘和托盘通知统一使用 Seed/Taiji logo；窗口圆角在 PyQt6 和打包产物中验证。
- 动效只用于状态表达，必须提供 reduced-motion/fallback，不能影响训练和推理线程。

### 稳定与安全门槛

- API 1 小时、1000 请求成功率 ≥ 99%；
- crash recovery 10/10；
- 首 byte ≤ 2 秒，生成 ≥ 200 B/s，模型加载 ≤ 30 秒；
- 桌面冷启动 ≤ 15 秒；
- 更新 URL、防路径穿越、依赖 CVE、签名和回滚门禁全部阻断发布；
- 日志默认不泄露训练语料、提示内容、密钥和本地路径。

### 发布物

- 可复现 Windows installer/portable build；
- 版本化 checkpoint、模型卡、数据卡、限制说明和迁移说明；
- 自动更新、回滚、崩溃恢复和离线启动 smoke；
- beta 反馈渠道和隐私说明。

## 13. 持续质量门禁

每个阶段都必须维持：

- 架构边界 AST/import test；
- planned/actual parameter count 一致性；
- checkpoint schema 和跨设备恢复；
- N0–N11、M5–当前最高 M 门槛；
- 数据 manifest 与 checkpoint lineage；
- core mypy 零新增、Legacy debt 不增长；
- 后端/前端测试、coverage、lint、构建、启动矩阵；
- 安全依赖扫描和桌面打包 smoke。

若门禁失败，修复当前阶段，不通过提高阈值、扩大 ignore 或删除测试继续推进。

## 14. 关键决策点

路线只在下列证据节点暂停讨论：

1. **R1 CUDA 决策**：纯 PyTorch 是否足够，还是只实现 profiler 指定的 sparse kernel。
2. **R3 机制决策**：16M 结果究竟受训练量、容量、输出器还是学习机制限制。
3. **R4 扩算力决策**：是否有证据进入下一训练 rung。
4. **R6 删除决策**：Legacy 是否只保留归档，或继续作为可安装离线基线。
5. **R7 发布决策**：全部发布门槛是否真实达到。

每次决策只提交一个推荐方案，附原始数据、失败风险和回退路径。

## 15. 当前唯一下一步

~~**立即完成 R0 质量闭环：在 CI 固定的 Black 24.12.0 环境复验 `black --check .`，随后进入 S2 工程质量与稳定性阶段。**~~

> **2026-08-26 更正（本文件已归档，此"唯一下一步"作废）**：Black 24.12.0 在 PyPI 与 `psf/black` 的 tag 列表中均不存在（24.10.0 之后直接跳到 25.1.0），所谓"CI 固定的 Black 24.12.0 环境"从未存在过——该 pin 使 CI 的依赖安装步骤直接失败，其后全部门禁被跳过。R0 的 black 闭环已于 2026-08-26 在 `black==26.5.1` 下完成（68 个文件一次性格式化，`black --check .` 仍为阻塞门禁）。当前唯一下一步以 [plans/README.md](../../README.md) 为准，门禁纪律与类型债见 [SEED_DEVELOPMENT_ROADMAP_2026_08.md](../../active/SEED_DEVELOPMENT_ROADMAP_2026_08.md) 第 14 节。

~~S1 功能验收结果记录在 `reports/seed_s1_acceptance_20260825.md`。当前 R0 代码收敛已落到 `main`，只剩固定 Black 版本的退出状态需要确认。~~在 R0/S1 关闭前不续跑 100M、不改 Taiji 学习方程、不写自定义 CUDA kernel，也不删除 `neuroplex/`。
