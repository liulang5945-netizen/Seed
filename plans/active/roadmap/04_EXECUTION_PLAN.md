# Seed / Taiji 详细执行蓝图

> 本文件细化当前路线，但**不创建第二个“当前唯一下一步”**。唯一执行入口与即时优先级始终以 [03_CURRENT_EXECUTION.md](03_CURRENT_EXECUTION.md) 为准。

## 1. 目标、顺序与依赖

Taiji 的目标不是重造 Transformer，也不是把生物名词硬编码进产品；它应在成熟的语言、编译、存储、执行和硬件技术之上，建立可观察、可恢复、可验证、可增长的原生认知与学习闭环。每个阶段都必须首先证明：状态归谁所有、是否能 checkpoint 往返、失败时如何隔离与回滚、以及独立于界面的 Gate 证据。

| 顺位 | 工作包 | 产出性质 | 解锁条件 |
|---|---|---|---|
| 当前 | Recovery portfolio 客户端审计回放 Gate | 只读、可审计产品证据 | 本表第 2 节通过 |
| W7-G0 | 全部 R 工作包合同冻结 | 可证伪 Gate 合同 | 当前 Gate 通过 |
| W7-R1 | 语言 provider watchdog | 受限语言外设的安全降级 | G0 对 provider 合同通过 |
| W7-R2 | interaction-group 与恢复归因 | 基于真实 trace 的可检验学习 | R1 的 provider 失败语义可观察 |
| W7-R3 | 视觉与桌面体验 | 对真实能力的表达层 | R1/R2 的真实状态可投影 |
| W7-R4 | CUDA 运行时 | 硬件专用的可恢复加速路径 | 有真实 CUDA 主机；此前保持 `hardware-blocked` |
| W7-R5 | 开放域结构成长与自进化 | 有资源治理、可回滚的增长 | R1–R4 证据与长期评估基线齐备 |

禁止跳过这一顺序：视觉不能包装未被证明的能力；provider 不能进入认知决策；CUDA 不能用 CPU 推测代替；增长不能由“神经元数量”或演示需求触发。

## 2. 当前 Gate：recovery portfolio 客户端审计回放

### 2.1 职责与落点

- **Owner：** Workbench 的右侧“属性与检查器”区域；建议新增独立只读组件（例如 `RecoveryPortfolioAuditPanel`），由 `WorkspaceView` 组合，不放入 Monaco 编辑器或聊天消息单元。
- **唯一数据路径：** `useWorkbenchProjection.refreshRecoveryPortfolio(...)` → `GET /api/workbench/taiji/recovery-branch/portfolio`。客户端不得复制、推导或修复 recovery ledger。
- **绑定方式：** 以已存在的 workbench audit event / loop context 所携带的 parent loop、snapshot 与 revision 为键。若事件投影缺少这些标识，先补充只读 lineage projection；不得用用户输入框、固定 loop id 或本地“最近一次”猜测替代。
- **明确非能力：** 不调用 `maintain`、`register`、`select`、`execute` 或任何修改型接口；不展示 parameters、完整 evidence payload、候选策略或可直接复用的执行输入。

### 2.2 可见信息与状态模型

视图只展示服务端已经脱敏、可审计的投影：

1. 当前 snapshot / revision / tick、容量上限、TTL、状态计数和数据新鲜度。
2. branch 的生命周期（active、selected、completed、failed、expired）、最小身份、预算、frontier 和 source evidence / after-state digest lineage。
3. eviction tombstone：被逐出的原因、次序、关联 revision，不回显被删除的可执行细节。
4. 空数据、权限/能力不可用、父循环不匹配、预期 revision 已过期、网络失败与被服务端拒绝的结构化状态。

发生 stale / revision mismatch 时，界面保留最后一个**已验证**快照并标记其过期，不能用新响应覆盖；选择另一个循环或卸载面板时必须清除其关联状态，避免跨循环串读。

### 2.3 Gate 测试与退出条件

| 层级 | 必做证据 | 故意破坏（必须变红） |
|---|---|---|
| S0 组件/投影 | 固定 fixtures 覆盖五种生命周期、容量压力、tombstone、空态、错误态与刷新时序 | 让 revision、lineage 排序、eviction 原因或参数脱敏错误 |
| S1 checkpoint/replay | 用已恢复的 portfolio checkpoint 回放；校验 branch 与 tombstone 顺序、关联 revision 和 freshness | 删除 tombstone、篡改 checkpoint revision、交叉 parent loop |
| S2 packaged client | Legacy-off 的实际 Workspace 路径中查看审计面板并记录 capability / network / UI 证据 | 阻断只读端点或错误暴露 mutation 操作 |

退出条件是：三层均通过；审计组件仅有 capability/event/portfolio GET 访问；前后端测试证明不输出敏感 parameters；正确处理 stale state；从客户端实际可追溯到同一 checkpoint revision。完成后，才可进入 W7-G0。

## 3. 并行 training / dataset / life-status 改动：已独立收口（2026-08-29）

原先滞留在工作树中的那组 training / dataset / life-status 改动及诊断脚本，已按本节要求以独立提交 `cd39632` 收口，不与只读审计 Gate 混提。实际修掉三个用户报告的产品缺陷：

- **不能连续训练多个资料**：`/api/train/files` 只平铺扫描 `data/` 顶层，子目录里的数据集不可见；改为递归扫描 + 原生格式过滤并返回相对路径，resume / native 侧同步按相对路径解析。
- **loss 曲线不显示**：面板隐藏时画布尺寸为 0×0，恢复可见后未重绘；改为 active 监听 + 尺寸兜底。
- **生命系统「已接入原生」但无数据**：链路四层同时断裂——adapter 无读访问器、`SeedRuntime` 未暴露 homeostasis、`LifeNeedsPayload` 的 `default: 50.0` 把空 `needs` 编造成四个假值并丢弃原生 `stress`、前端无对应渲染契约。详见 [03_CURRENT_EXECUTION.md](03_CURRENT_EXECUTION.md) W5 条目与 [02_GATES_AND_CI.md §14.17](02_GATES_AND_CI.md)。

回归证据：后端 `556 passed, 6 skipped`、前端 `42 files / 237 passed`、核心 mypy 0 错误、Ruff / ESLint / build / API contract 全通过（数字权威源见 [IMPLEMENTATION_STATUS_2026_08.md](../../reference/IMPLEMENTATION_STATUS_2026_08.md)）。

**仍未满足的准入（不得当作已完成引用）**：本节原定的训练类改动最小准入——创建 checkpoint → 关闭运行时 → 恢复 → 继续一步 → 对 lineage、预算、结构、provider artifact 和可见指标做等价性断言——这轮**没有执行**。因此上述 dataset / resume 改动目前只算「数据集可发现性与 API 契约修复」，**不构成训练能力宣称**；在补上该往返等价性 Gate 之前不得据此启动长训。诊断脚本 `diag_train_multi_dataset.py` / `diag_train_select_all.py` 已按既有惯例归入 `scripts/archive/diagnostics/` 作为实测证据。

### 3.1 训练 ETA / 进度分母修复（2026-08-29 追加收口）

用户续报「训练剩余时间也不够准确」。实测定案：`max_ticks` 截断了训练循环，但 `fraction`/`eta` 仍以整份数据集作分母，导致 ETA 误差约 **279 万倍**（上报 32.3 天 vs 真实剩余 0），进度条卡在 0.05%。已修：有效分母 `min(total_bytes, max_ticks)`、ETA 改为 `remaining/rate` 有界换算、暂停时长从 `elapsed` 扣除、收尾语义区分「完成」与「用户停止」；前端同批修掉恒显 100% 的 step 计数器、缺「天」档的 `fmtTime` 和错误的吞吐单位标注。完整根因链与四条纪律见 [02_GATES_AND_CI.md §14.18](02_GATES_AND_CI.md)，门禁为 `tests/seed/test_training_progress_contract.py`（4 例，先红后绿）。

回归证据：后端 `560 passed, 6 skipped`、前端 `42 files / 237 passed`、Ruff / 核心 mypy / ESLint / build / API contract / native boundary 全通过。**本节 §3 的 checkpoint 往返等价性准入仍未满足**——ETA 修复只让进度显示可信，不构成训练能力宣称，长训准入不变。

## 4. W7-G0：先冻结每个后续 Gate 的合同

为 R1–R5 各建立版本化 Gate manifest（存于 `plans/manifests/`，并由代码/测试引用），每份至少包含：

- 要证明或要推翻的 claim、唯一 owner、输入及结构化输出；
- capability / trace / 资源预算 / checkpoint revision 的前置条件；
- S0 小型确定性模拟、S1 replay/sandbox、S2 packaged-client 或真实工作台的分层证据；
- red proof、holdout/lesion/cross-seed 设计、失败模式、隔离及回滚路径；
- manifest、实验、代码提交三者的关联，以及“替代了哪项旧假设”。

G0 不实现新自治。它的退出条件是所有 R 工作包都有可执行的合同、测试入口和不可越界声明；未填入真实输入或 checkpoint 的项目保持 blocked，而不是以空示例宣布就绪。

## 5. W7-R1：provider watchdog（语言外设健康治理）

语言 provider 只充当“嘴巴/耳朵”式 realization，不拥有目标、工具选择或认知状态。该工作包以 artifact digest 为隔离单位，维护版本化健康记录：接受率、校验失败、超时、加载失败、artifact 漂移、fallback 与 canary 失败。

- **控制状态：** `healthy → degraded → quarantined → probing`，具备迟滞、冷却窗口和复归阈值，避免一次错误切换或反复抖动。
- **回退：** 仅能选用 allowlist 中、内容寻址、已通过 canary 的前一 artifact；否则输出 Taiji 原生的可读结构化降级结果，不伪造语言回答。
- **持久化：** controller 状态、当前/上一 artifact、失败摘要、计数和冷却期限进入 checkpoint；不得持久化无必要的 prompt / 对话历史。
- **产品边界：** UI 只观测，不在客户端自行切换 provider；任何切换都由后端的可恢复 controller 决定。

验收包含 artifact 漂移、连续失败、超时、冷却、探测恢复、错误 fallback、checkpoint 中断恢复和多 provider 隔离的红绿测试；真实 canary 必须对同一 artifact digest 记录结果。

## 6. W7-R2：interaction-group 与恢复归因

R2 研究的是实际工作流中哪些交互区域共同提升/损害结果，不预设“规划神经元”“记忆神经元”等角色标签。观测来源只能是 W2/W3 的真实 trace：workspace route、memory、planner、tool、recovery、资源消耗和 outcome。

- 以贡献、互补、冲突、恢复效果与资源代价形成候选 group，并保留来源与 revision。
- 先做 leave-one-group-out、pairwise 和局部 counterfactual；高阶交互只有在多 seed / holdout 证据充分时才允许建立。
- 学到的策略写回对应 owner 的 policy 或 memory，不创建吞没全局的单体 controller。
- 基线必须包括单策略、无 group、随机 group 与无归因；指标必须同时报告收益、遗忘、恢复时间、参数/连接、内存、延迟与能耗近似。

退出条件是见到真实 holdout 上的优势、lesion 后有可解释变化、checkpoint 往返不改变 group 溯源，且无硬编码角色名单。

## 7. W7-R3：视觉与桌面体验（依赖真实状态）

R3 只把已经存在的能力变得可辨识、可访问、可信，不用 mock 补齐产品。范围包括：

- 所有侧边导航无隐藏核心项；状态、操作、错误和 provider 降级均来自 native facade 的真实 capability / lineage。
- 统一设计 token、DPI、键盘导航、焦点、动效降级和长文本/错误态；每个重要页面具有窄窗口与高 DPI 截图回归。
- Windows 桌面外壳使用 Taiji 标识，覆盖应用窗口、任务栏、托盘、通知、圆润窗口形状与最小化后行为；图标变换或流转效果不得影响静态可识别性和无障碍替代文本。
- packaged-client smoke 必须验证启动、托盘、通知、Legacy-off、真实 workbench capability 与网络错误显示，而非只测开发服务器。

视觉任务只有在 R1/R2 的状态模型稳定后接入；如先前端发现后端没有真实字段，应退回相应 owner 补充投影，不能由前端猜测。

## 8. W7-R4：CUDA（当前 `hardware-blocked`）

当前没有可用 CUDA 主机，因此本工作包保留完整设计与 CPU 基线，不实现或伪称 GPU 性能。硬件到位后的顺序固定为：

1. 固定 workload、manifest、seed、checkpoint revision、device/dtype 与容差合同；建立 CPU 基线。
2. 证明 CPU → CUDA → CPU 的 checkpoint 可恢复，且结构、lineage、预算与 artifact 状态一致；数值只允许合同声明的容差差异。
3. 先 profile 再优化；只有 profile 证据显示瓶颈时，才引入向量化、fused kernel 或专用加速。
4. 执行 deterministic / tolerance、OOM、降级、跨设备恢复、吞吐/延迟/显存及能耗指标 Gate。

无硬件时允许维护测试 fixture、设备抽象与禁止路径测试，禁止提交依赖未验证 GPU 的结论。

## 9. W7-R5：开放域结构成长与自进化

“自进化”不是无限追加权重，也不等同于重新从零训练。它是针对持续失败簇、容量压力、长期遗忘或恢复不足，由 owner 提出、有资源治理和可回滚实验支持的结构改变。

- **触发：** 真实任务失败簇 + 独立 holdout + 资源压力，不能凭规模目标、人工角色清单或单个演示触发。
- **提案：** 由对应 owner 产生可解释、带父 lineage、预算和预期收益的连接/记忆/局部模块增长候选；全局治理者只做预算、冲突与淘汰，不决定认知内容。
- **验证：** shadow learn → holdout → lesion → rollback → 原子合并。未通过的候选必须留下 tombstone 与失败原因，不能污染已验证结构。
- **长期评估：** 同时报收益、遗忘、恢复时间、结构规模、内存、延迟、能耗和跨 seed 稳定性；通过后才增加稳定容量。

R5 的 checkpoint 必须包含结构 revision、parent/child lineage、资源账本、提案状态、淘汰 tombstone 和复现实验 manifest。任何不能恢复这些状态的“成长”都不进入正式模型。

## 10. 共同工程纪律、提交与 CI

1. 一个提交只收口一个 owner 明确的 slice；计划、manifest、实现与测试必须同批更新，不能以“后补测试”跨越 Gate。
2. 修改训练路径前先运行 checkpoint 保存/恢复/继续训练的最小测试；训练实验的输出、seed、manifest 和 checkpoint digest 必须可关联。
3. 每个 slice 至少运行其受影响的 Python / frontend 测试、静态检查和必要的构建；CI 失败先修 CI，再进行下一工作包。历史通过记录不能代替当前差异验证。
4. S0、S1、S2 的证据分别记录；只有 S2 成功后，才将一次性调试记录从 active/reference 移入 archive。
5. 长期核心合同、当前未关闭缺口和当前实现事实留在 active/reference；已替代路线、试验日志和失效方案归档。删除任何目录或脚本前先证明无运行时、构建、测试或发布引用。

## 11. 停止条件与路线更新

出现以下情况时暂停自动推进，先修正事实源并单独提交：API owner 不清、checkpoint 不可恢复、Gate 的红测不红、CI 失败、前端投影与真实 runtime 不一致、或需要改变“语言 provider 不参与 cognition”的边界。每完成一项 Gate，更新 [03_CURRENT_EXECUTION.md](03_CURRENT_EXECUTION.md)、实现状态和对应 manifest，再选择其顺序中唯一后继项。
