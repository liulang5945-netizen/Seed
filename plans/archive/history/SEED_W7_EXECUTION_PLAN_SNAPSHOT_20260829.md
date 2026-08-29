# 归档：Seed / Taiji W7 执行蓝图快照（2026-08-29）

> 本文件保留 W7-R1–R5 当时的设计和已完成过程，不提供当前执行顺序。后续计划已重写到 [04_EXECUTION_PLAN.md](../../active/roadmap/04_EXECUTION_PLAN.md)，即时动作只看 [03_CURRENT_EXECUTION.md](../../active/roadmap/03_CURRENT_EXECUTION.md)。下文“下一步/当前”均是历史语境。

## 1. 目标、顺序与依赖

Taiji 的目标不是重造 Transformer，也不是把生物名词硬编码进产品；它应在成熟的语言、编译、存储、执行和硬件技术之上，建立可观察、可恢复、可验证、可增长的原生认知与学习闭环。每个阶段都必须首先证明：状态归谁所有、是否能 checkpoint 往返、失败时如何隔离与回滚、以及独立于界面的 Gate 证据。

| 顺位 | 工作包 | 产出性质 | 解锁条件 |
|---|---|---|---|
| 已完成 | Recovery portfolio 客户端审计回放 Gate | 只读、可审计产品证据 | S0/S1/S2 通过 |
| 已完成 | W7-G0：全部 R 工作包合同冻结 | 可证伪 Gate 合同 | 5 份 v1 manifest + manifest contract test |
| 已完成 | W7-R1 | 受限语言外设的安全降级 | G0 对 provider 合同通过 |
| 已完成 | W7-R2 | interaction-group 与恢复归因 | R1 的 provider 失败语义可观察 |
| 当前 | W7-R3 | 视觉与桌面体验 | R1/R2 的真实状态可投影；**S2 页面级已取证，Windows shell 为 `tool-blocked`** |
| W7-R4 | CUDA 运行时 | 硬件专用的可恢复加速路径 | 有真实 CUDA 主机；此前保持 `hardware-blocked` |
| W7-R5 | 开放域结构成长与自进化 | 有资源治理、可回滚的增长 | R1–R4 证据与长期评估基线齐备；R3-S2 阻塞期允许先做 S0 前置切片 |

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

### 2.4 落地进展（2026-08-29）

S0（组件/投影）与 S1（checkpoint 回放）已在代码层闭合，证据与红/绿链见 [02_GATES_AND_CI.md §14.20](../../active/roadmap/02_GATES_AND_CI.md)：

- **只读绑定键**：新增 `GET /api/workbench/taiji/recovery-branch/context`，把持久化的 parent loop / snapshot / revision 以只读投影形式作为客户端唯一绑定来源（§2.1 禁止输入框/固定 id/「最近一次」猜测）；无 portfolio 时返回 `has_portfolio:false` 结构化空态。
- **结构化错误码**：portfolio 快照路由把稳定错误映射为可分支的 `detail.error`（`portfolio_not_persisted` / `portfolio_snapshot_not_current` / `portfolio_parent_mismatch` / `portfolio_revision_stale` + `observed_revision` / `portfolio_invalid`）。
- **前端审计面板**：`RecoveryPortfolioAuditPanel` 组合进 WorkspaceView 右栏「属性与检查器」，事件投影驱动重取（不新增独立轮询）；渲染 §2.2 全部四类信息（快照元数据/生命周期与 lineage/墓碑/结构化空态与错误态）；stale 时保留最后一个已验证快照并标记过期、切换 parent loop 或卸载时清空关联状态；只读（仅 context/portfolio 两个 GET，vitest 静态断言源码不含任何 mutation 投影方法）。不展示 parameters / evidence / 可复用执行输入。

**S2 packaged-client 已完成（2026-08-29）**：最终 `dist/Seed/Seed.exe` 在 Legacy-off、native runtime、8138 自定义端口、真实 `LOCALAPPDATA` 环境下启动；当首选数据根不可用时选择包内 `user_data`，不需要手工 Qt 环境覆盖。真实 `#/workspace?taiji_client=desktop` 路径显示 Workspace、右侧检查器和 `RecoveryPortfolioAuditPanel`；所有观测 API 均为 8138/200，无页面错误或 Legacy/Transformer/HF/GGUF 标记。客户端可追溯到 `seed:seed_corpus.pt` 与 native capability snapshot `5572f3ff01de596e380bda518eff357c4191610bab836d54e9c505c9b58f256f` revision `4`。本次启动的 portfolio 是结构化空态，非空分支排序/墓碑继续由 S0/S1 replay 覆盖；证据文件为 [packaged_client_s2_20260829.json](../../../reports/packaged_client_s2_20260829.json)。据此，S2 退出项完成，W7-G0 入口解锁。

## 3. 并行 training / dataset / life-status 改动：已独立收口（2026-08-29）

原先滞留在工作树中的那组 training / dataset / life-status 改动及诊断脚本，已按本节要求以独立提交 `cd39632` 收口，不与只读审计 Gate 混提。实际修掉三个用户报告的产品缺陷：

- **不能连续训练多个资料**：`/api/train/files` 只平铺扫描 `data/` 顶层，子目录里的数据集不可见；改为递归扫描 + 原生格式过滤并返回相对路径，resume / native 侧同步按相对路径解析。
- **loss 曲线不显示**：面板隐藏时画布尺寸为 0×0，恢复可见后未重绘；改为 active 监听 + 尺寸兜底。
- **生命系统「已接入原生」但无数据**：链路四层同时断裂——adapter 无读访问器、`SeedRuntime` 未暴露 homeostasis、`LifeNeedsPayload` 的 `default: 50.0` 把空 `needs` 编造成四个假值并丢弃原生 `stress`、前端无对应渲染契约。详见 [03_CURRENT_EXECUTION.md](../../active/roadmap/03_CURRENT_EXECUTION.md) W5 条目与 [02_GATES_AND_CI.md §14.17](../../active/roadmap/02_GATES_AND_CI.md)。

回归证据：后端 `556 passed, 6 skipped`、前端 `42 files / 237 passed`、核心 mypy 0 错误、Ruff / ESLint / build / API contract 全通过（数字权威源见 [IMPLEMENTATION_STATUS_2026_08.md](../../reference/IMPLEMENTATION_STATUS_2026_08.md)）。

**准入已满足（2026-08-29）**：本节原定的训练类最小准入——创建 checkpoint → 关闭运行时 → 恢复 → 继续一步 → 对 lineage、预算、结构、provider artifact 和可见指标做等价性断言——已由 `tests/seed/test_checkpoint_roundtrip_contract.py`（3 例）补齐并转入绿门禁。过程中实测发现并修复一处产品级往返不对称：`checkpoint()` 无条件序列化任意语言器官而 `_restore_language_organ` 只认 native/structured 两种 backend，导致**任何接入 guarded provider 的运行时无法从自己的存档启动**；修复为显式可观测的「脱挂留痕」（`detached_language_organ_backend`）而非拒绝整份存档。红/绿证据与规则更新见 [02_GATES_AND_CI.md §14.19](../../active/roadmap/02_GATES_AND_CI.md)。据此 dataset / resume 改动正式构成训练能力宣称，长训准入前提补齐。诊断脚本 `diag_train_multi_dataset.py` / `diag_train_select_all.py` 已按既有惯例归入 `scripts/archive/diagnostics/` 作为实测证据。

### 3.1 训练 ETA / 进度分母修复（2026-08-29 追加收口）

用户续报「训练剩余时间也不够准确」。实测定案：`max_ticks` 截断了训练循环，但 `fraction`/`eta` 仍以整份数据集作分母，导致 ETA 误差约 **279 万倍**（上报 32.3 天 vs 真实剩余 0），进度条卡在 0.05%。已修：有效分母 `min(total_bytes, max_ticks)`、ETA 改为 `remaining/rate` 有界换算、暂停时长从 `elapsed` 扣除、收尾语义区分「完成」与「用户停止」；前端同批修掉恒显 100% 的 step 计数器、缺「天」档的 `fmtTime` 和错误的吞吐单位标注。完整根因链与四条纪律见 [02_GATES_AND_CI.md §14.18](../../active/roadmap/02_GATES_AND_CI.md)，门禁为 `tests/seed/test_training_progress_contract.py`（4 例，先红后绿）。

回归证据：后端 `560 passed, 6 skipped`、前端 `42 files / 237 passed`、Ruff / 核心 mypy / ESLint / build / API contract / native boundary 全通过。**本节 §3 的 checkpoint 往返等价性准入已由 `test_checkpoint_roundtrip_contract.py` 补齐并转绿**（见上节正文与 [02_GATES_AND_CI.md §14.19](../../active/roadmap/02_GATES_AND_CI.md)）——ETA 修复让进度显示可信，往返门禁让长训落盘可恢复，二者共同闭合训练能力宣称。

## 4. W7-G0：先冻结每个后续 Gate 的合同

为 R1–R5 各建立版本化 Gate manifest（存于 `plans/manifests/`，并由代码/测试引用），每份至少包含：

- 要证明或要推翻的 claim、唯一 owner、输入及结构化输出；
- capability / trace / 资源预算 / checkpoint revision 的前置条件；
- S0 小型确定性模拟、S1 replay/sandbox、S2 packaged-client 或真实工作台的分层证据；
- red proof、holdout/lesion/cross-seed 设计、失败模式、隔离及回滚路径；
- manifest、实验、代码提交三者的关联，以及“替代了哪项旧假设”。

G0 不实现新自治。它的退出条件是所有 R 工作包都有可执行的合同、测试入口和不可越界声明；未填入真实输入或 checkpoint 的项目保持 blocked，而不是以空示例宣布就绪。

**G0 已完成（2026-08-29）**：五份合同分别为 [R1 provider watchdog](../../manifests/taiji_w7_r1_provider_watchdog_v1.json)、[R2 interaction-group](../../manifests/taiji_w7_r2_interaction_group_v1.json)、[R3 visual/desktop](../../manifests/taiji_w7_r3_visual_desktop_v1.json)、[R4 CUDA](../../manifests/taiji_w7_r4_cuda_v1.json)、[R5 open-domain growth](../../manifests/taiji_w7_r5_open_domain_growth_v1.json)。`tests/test_w7_gate_manifests.py` 校验五份 manifest 都具备 claim/owner/input/output/trace/resource/checkpoint、S0/S1/S2、red proof、holdout、lesion、失败隔离、rollback 和越界声明；R4 明确保持 `hardware-blocked`，其他工作包为 `contract_frozen`，没有任何未来 Gate 被伪标记为 `passed`。因此当时的下一执行入口切换为 W7-R1 实现，不提前实现 R2–R5 自治；当前执行入口以 [03_CURRENT_EXECUTION.md](../../active/roadmap/03_CURRENT_EXECUTION.md) 为准。

## 5. W7-R1：provider watchdog（语言外设健康治理）

语言 provider 只充当“嘴巴/耳朵”式 realization，不拥有目标、工具选择或认知状态。该工作包以 artifact digest 为隔离单位，维护版本化健康记录：接受率、校验失败、超时、加载失败、artifact 漂移、fallback 与 canary 失败。

- **控制状态：** `healthy → degraded → quarantined → probing`，具备迟滞、冷却窗口和复归阈值，避免一次错误切换或反复抖动。
- **回退：** 仅能选用 allowlist 中、内容寻址、已通过 canary 的前一 artifact；否则输出 Taiji 原生的可读结构化降级结果，不伪造语言回答。
- **持久化：** controller 状态、当前/上一 artifact、失败摘要、计数和冷却期限进入 checkpoint；不得持久化无必要的 prompt / 对话历史。
- **产品边界：** UI 只观测，不在客户端自行切换 provider；任何切换都由后端的可恢复 controller 决定。

验收包含 artifact 漂移、连续失败、超时、冷却、探测恢复、错误 fallback、checkpoint 中断恢复和多 provider 隔离的红绿测试；真实 canary 必须对同一 artifact digest 记录结果。

**S0/S1/S2 已通过（2026-08-29）**：健康状态已从仅绑定 `artifact_id` 收紧为绑定 `artifact_id + artifact_digest`；同 ID 内容替换会建立新记录，不继承旧失败计数。`tests/seed/test_provider_watchdog_gate.py` 与 [评测脚本](../../../scripts/training/eval_taiji_provider_watchdog.py) 的 S0 覆盖 healthy/连续失败/单次回退/cooldown/legacy checkpoint 兼容；S1 使用真实 `TSKV8Adapter.native_checkpoint()` 保存→恢复→继续失败探针，校验 artifact、digest、registry、计数和阈值后的 lineage。S2 使用明确 Legacy-off 的 `dist/Seed/Seed.exe` 在默认 8000 端口启动，健康和 runtime/status 均为 200，native provider 状态及 digest 字段可读，Workspace/状态证据/UI provider 投影可见，8 个 API 请求全部绑定 8000，无页面错误、请求失败或 Legacy/Transformer/HF 标记。S2 运行的是 `native-readable` 内置器官，外部 provider artifact 轮换没有被虚报为客户端证据。报告为 [S0](../../../reports/taiji_w7_r1_provider_watchdog_20260829.json)、[S1](../../../reports/taiji_w7_r1_provider_watchdog_s1_20260829.json) 和 [S2](../../../reports/taiji_w7_r1_provider_watchdog_s2_20260829.json)，定向 provider 回归 `21 passed`；R1 完成，随后进入 W7-R2-S0。

## 6. W7-R2：interaction-group 与恢复归因

R2 研究的是实际工作流中哪些交互区域共同提升/损害结果，不预设“规划神经元”“记忆神经元”等角色标签。观测来源只能是 W2/W3 的真实 trace：workspace route、memory、planner、tool、recovery、资源消耗和 outcome。

- 以贡献、互补、冲突、恢复效果与资源代价形成候选 group，并保留来源与 revision。
- 先做 leave-one-group-out、pairwise 和局部 counterfactual；高阶交互只有在多 seed / holdout 证据充分时才允许建立。
- 学到的策略写回对应 owner 的 policy 或 memory，不创建吞没全局的单体 controller。
- 基线必须包括单策略、无 group、随机 group 与无归因；指标必须同时报告收益、遗忘、恢复时间、参数/连接、内存、延迟与能耗近似。

退出条件是见到真实 holdout 上的优势、lesion 后有可解释变化、checkpoint 往返不改变 group 溯源，且无硬编码角色名单。

**S0/S1/S2 已通过（2026-08-29）**：新增 `taiji/interaction_groups.py` 与真实 Workbench 评测入口，S0 用版本化 trace 做 factorial counterfactual，S1 用真实 `TSKV8Adapter` 做 `Event/Outcome` 投影和 `taiji-native-v1` 精确回放，S2 通过 `SeedRuntime + WorkbenchEnvironment` 实际执行 workspace list/search/read，验证 capability snapshot、native executive selection、world evidence、recovery retry、holdout/lesion 和 checkpoint 归属。报告 [S2](../../../reports/taiji_w7_r2_interaction_groups_s2_20260829.json) 中 8 个 train、8 个 holdout record 全部 replay 一致，2 个 group admitted、4 个候选拒绝，`role_label_input_count=0`，未写回 executive/provider。反例覆盖跨 revision、holdout 污染、资源压力、source digest 篡改以及缺 Workbench 证据；R2 完成，下一步进入 W7-R3-S0 visual/desktop evidence。

## 7. W7-R3：视觉与桌面体验（依赖真实状态）

R3 只把已经存在的能力变得可辨识、可访问、可信，不用 mock 补齐产品。范围包括：

- 所有侧边导航无隐藏核心项；状态、操作、错误和 provider 降级均来自 native facade 的真实 capability / lineage。
- 统一设计 token、DPI、键盘导航、焦点、动效降级和长文本/错误态；每个重要页面具有窄窗口与高 DPI 截图回归。
- Windows 桌面外壳使用 Taiji 标识，覆盖应用窗口、任务栏、托盘、通知、圆润窗口形状与最小化后行为；图标变换或流转效果不得影响静态可识别性和无障碍替代文本。
- packaged-client smoke 必须验证启动、托盘、通知、Legacy-off、真实 workbench capability 与网络错误显示，而非只测开发服务器。

视觉任务只有在 R1/R2 的状态模型稳定后接入；如先前端发现后端没有真实字段，应退回相应 owner 补充投影，不能由前端猜测。

**R3-S0/S1 已通过（2026-08-29）**：生命状态页恢复原有五维雷达作为 Taiji 状态主视觉；原生运行时摘要压缩为辅助卡片，需求条不再与雷达重复占据首屏；侧边栏删除重复的底部生命状态脉冲块，只保留“系统 → 生命状态”导航。`RuntimeEvidenceStrip` 从聊天、能力、知识库、训练和设置页移除，在生命页底部保留为默认折叠审计详情，展开后仍读取同一 runtime projection。桌面托盘“生命状态”动作改为先恢复窗口再执行 `#/life`，不再只改变隐藏 WebView 的 hash。S0 前端 `43 files / 245 passed`、Vite build、ESLint（0 errors）和桌面契约 `12 passed`；S1 通过 `scripts/release.py --skip-nsis` 重建 `dist/Seed/Seed.exe`，内置前端 index 与源码构建产物字节一致，Legacy-off/native runtime 在 8138 端口健康 canary 为 ok/taiji_available/seed_active/model_loaded 全真。证据见 [R3-S1](../../../reports/taiji_w7_r3_visual_desktop_s1_20260829.json)。

**R3-S2 状态：页面级取证通过，Windows shell 取证为 `tool-blocked`**。Chrome 已安装，生命页与窄布局页面证据已完成；但 Windows Computer Use 当前无法激活 Seed 窗口，不能把桌面背景截图当作窗口、任务栏、托盘、通知或高 DPI 证据。详细记录见 [R3-S2](../../../reports/taiji_w7_r3_visual_desktop_s2_20260829.json)。恢复后的固定动作是：

1. 在能够激活窗口的 Windows 会话中，先复跑 `scripts/release.py --check-only --skip-nsis` 确认取证包仍为同一新包，避免用旧包取证。
2. 沿已冻结的 [R3 manifest](../../manifests/taiji_w7_r3_visual_desktop_v1.json) 补齐真实窗口、任务栏、托盘、通知和高 DPI 证据；已通过的 Chrome 页面与窄布局证据保留，不新增前端认知状态、不伪造运行时能力。
3. 在 shell 证据通过前，R3-S2 与 manifest 的 `implementation.status` 均保持未通过；R5 进度不能替代窗口、任务栏、托盘、通知和 DPI 证据。

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

**manifest 归属（避免误读）：** 已冻结的 [taiji_w7_r5_open_domain_growth_v1.json](../../manifests/taiji_w7_r5_open_domain_growth_v1.json)（`status: contract_frozen`，`implementation.status: not_started`）只覆盖上述开放域结构成长，其 owner 是 `taiji/structural_growth.py`。下面 §9.1 的知识内化转换器与 §9.2 的可注册效应器**不在该 manifest 范围内**，需要另立一份新的 R5 manifest（尚未创建），并同样满足 [tests/test_w7_gate_manifests.py](../../../tests/test_w7_gate_manifests.py) 的结构约束（`status ∈ {contract_frozen, hardware-blocked}`、S0/S1/S2 三层、`checkpoint.required` 为真、`implementation.status != "passed"`）。§9.0 是不需要新 manifest 的缺口修复，不得被当作 §9.1/§9.2 的实现。

### 9.0 R5-S0 前置切片：接通生产里断开的执行学习通道

在做任何内化或效应器成长之前，必须先修掉一个已核实的生产缺口：`api/seed_runtime.py` 调用了 `synthesize_executive_candidates()`（L810）和 `select_executive(...)`（L813），但 `api/` 下**没有任何 `record_executive_outcome` 调用点**；而 `taiji/adapter.py` L4715 的 `record_executive_outcome` 是驱动 `LearnedAffordanceFeatures.online_update` 的唯一入口。因此当前打包客户端**选择 affordance 却从不学习**，`outcome_head` 一直停留在初始化状态。

本切片已完成：红测锁定了真实执行路径在修复前不会增加学习计数；现已在 evidence 提交点（构造 `WorkbenchTaijiEvidence` 并 `record_world_event` 之后）接上 `record_executive_outcome`。执行前保存 source affordance 与 affordance context，严格校验当前 executive decision、intent_id 和 `source_affordance_id`；任一不一致即 fail-closed，`learn=False` 不产生学习副作用，`learn=True` 才执行 online update。checkpoint 往返保留 `fit_updates` / `online_updates` 并恢复同一最后决策。此切片不写转换器、不动 read-only 限制、不引入注册表。证据见 [R5-S0](../../../reports/taiji_w7_r5_s0_learning_channel_20260829.json)。

### 9.1 R5-A：知识内化转换器与可删性判据（skill/mcp 学进权重后可删）

**目标不是把工具接进训练系统，而是让外挂的描述性知识被学进权重后可以物理删除。** 这条线的边界由已核实的学习目标决定：`LearnedAffordanceFeatures.fit` 的回归目标是 `item.reward`、损失为 MSE，因此这个通道内化的是「在当前世界状态与感知上下文下，这个 affordance 值多少」，**不是**工具的执行语义。由此得到明确的可删/不可删划分：

- **可删：** skill 的散文规范、工具选择提示词、路由规则——它们原本在替模型做选择，模型学会打分后即为冗余。
- **不可删：** MCP 的真实执行通道。那是身体不是知识，删掉它不是内化而是截肢。
- **例外（属 R5-B 的 L4 层，不在本线）：** 纯计算类能力，其执行体本身也可被替代。

**转换器落点与依赖方向。** 既有依赖方向是 `seed_platform → taiji`（`seed_platform/workbench.py` 内部惰性 `from taiji import WorldEvent`），因此转换器不能住在 `taiji/` 里反向 import workbench 类型。新增 `taiji/internalization.py`，只接受纯结构化 DTO：

```python
@dataclass(frozen=True)
class AffordanceOutcomeRecord:
    evidence_id: str
    capability_id: str
    after_state_digest: str
    snapshot_id: str
    snapshot_revision: int
    tick: int
    success: bool
    status: str
    error_code: str = ""
    reward_terms: Mapping[str, float] = field(default_factory=dict)
```

`reward_terms` 遵循「缺测就不出现」，不给默认值——这是 W5 homeostasis 编造事故（`default: 50.0` 把空 `needs` 补成四个假值且门禁全绿，见 [03_CURRENT_EXECUTION.md](../../active/roadmap/03_CURRENT_EXECUTION.md)）的直接教训。

转换器主体 `AffordanceInternalizationConverter.convert(record, *, state, affordances, percept_features, world_latent, world_uncertainty)` 返回 `tuple[AffordanceFeatureTrainingExample, ...]`，七步全部 fail-closed：

1. **快照新鲜度**：复用既有 stale 语义，`record.snapshot_id` 与当前快照不一致直接抛错。
2. **失败调用不造 affordance**：evidence 级 `to_taiji_affordances` 在 `success=False` 时返回 `()`，因此转换器对失败记录返回空元组并在账本记 `no_affordance`，绝不从 `capability_id` 凭空构造 affordance；负奖励只在真实 affordance 存在且 status 为 error 时产生。
3. **grounding 只能来自 `ground()`**：调用后断言 `feature_provenance == "world-state-grounding"` 且 `features.numel() == source.input_dim`（当前 `BASE_FEATURE_DIM = 17`），禁止用 workbench result 字段手工拼向量。
4. **上下文向量必填**：生产 `context_dim = perception.feature_dim ≠ 0`，缺失会触发 `fit` 的 `RuntimeError`；取值规则与 `adapter._affordance_context` 完全一致（`world.latent` 为空则回落到 percept features），不复制第二份规则。
5. **`world_uncertainty` 越界直接抛错**，不做 clamp——clamp 会把 bug 变成看起来合法的数据。
6. **奖励合成显式加权**：`reward = sum(reward_terms[k] * WEIGHTS[k])`。首版只允许 `{"success": ±1.0}`，因为 workbench 当前只发出 `reward=1.0 if success else -1.0`；risk / reversible / latency / result-size 项必须等真实证据出现后再加。
7. **确定性 id**：`example_id = f"internalize:{record.evidence_id}:{grounded.affordance_id}"`，两段均为内容寻址，保证重放与去重成立。

**两条学习通道。** 在线通道即 `record_executive_outcome → online_update`（代码已具备，缺生产调用点，见 §9.0）；离线巩固通道用有界 replay buffer 在空闲/睡眠期批量 `fit`。**必须批量**：`fit` 每次调用都在方法内新建 `LocalAdam`，逐条 `fit` 会反复清零动量。buffer 以 `(evidence_id, affordance_id)` 去重，否则 `fit_updates` 虚高、可删性判据的输入被污染。

**可删性判据：五道 lesion 全过才允许删除。** 复用 `scripts/training/eval_taiji_p7_grounded_multistep.py` 已实现的三种 lesion 技术：

1. **外挂充分性**：删掉 skill 散文后 holdout 选择质量仍在声明容差内。
2. **内化必要性**：`attach_affordance_features(None)` 必须抛出含 `learned affordance feature source` 的错误，且性能塌陷。
3. **grounding 必要性**：零 grounding lesion 必须造成退化，否则学到的只是常数偏置。
4. **可恢复性**：checkpoint 往返保留 `fit_updates` / `online_updates` 并复现决策（§10 硬约束）。
5. **遗忘上界**：旧任务保持率在声明界内。

三态账本 `external → shadow → internalized`，失败分支写 `tombstone` 与原因，映射到 §9 的 `shadow learn → holdout → lesion → rollback → 原子合并`。**只有到达 `internalized` 才物理删除外挂条目并写墓碑。**

### 9.2 R5-B：可注册效应器与身体成长（L0→L4）

**现状约束（已核实）：** `seed_platform/workbench.py` 的 `execute_tool`（L1790）是 15 个硬编码 `elif tool_name == ...` 加 `else: unknown_capability`；能力清单来自硬编码的 `CapabilitySnapshot.default()`；**全文件没有任何 `def register`**。新增一个能力必须改源码，这就是身体无法成长的根因。

**已有的三个可复用抓手：** `CapabilityDescriptor.source` ≈ bundle 来源可追溯；`CapabilityDescriptor.enabled` ≈ 已声明未启用，天然表达 `proposed` / `shadow`；`CapabilitySnapshot.snapshot_id + revision` ≈ 内容寻址的装配身份。并且 Seed 有一个社区插件平台没有的安全网：evidence 携带 `snapshot_id`，任何注册表变更都会顶掉 `snapshot_id` 并让旧 evidence 立即判定为 stale——**身体一变，旧经验自动失效**。

**可迁移的插件平台原理（借机制、不借语言选择）：** 插件是接收 `ctx` 的函数而非接口实现，依赖注入决定加载顺序；一切注册都是可撤销副作用（卸载即调用 disposer，不手工维护全局表）；产品装配是运行时一等公民（profile → bundle → patch → loader，patch 按 `id` **整体替换** config 而非深合并，装配结果可 dump 审计）；自修补分两步——校验元数据、预编译、生成不可变包 ID 并**只保存候选而不激活**，激活是独立动作。

**唯一必须偏离的一点：** 社区平台要求工具描述写清「何时调用/前置条件/失败语义/副作用」，因为那是给 LLM 读来选工具的。本项目**不得照搬**——[03_CURRENT_EXECUTION.md](../../active/roadmap/03_CURRENT_EXECUTION.md) 的边界是语言 provider 不参与 cognition。选择必须继续由结构化特征驱动，描述文本只服务于人和审计。

**红线：** `tests/seed/test_legacy_plugin_gate.py` 断言 `SEED_ENABLE_LEGACY=0` 时 router 数为 0 且 `load_legacy_cortex` 抛错。新注册表不得以任何形式复活该路径，否则机制收敛被破坏。

分级方案，每级带可证伪 Gate：

- **L0 注册表重构，零新增能力。** `execute_tool` 改为经注册表分派，15 个能力全部变成内置插件注册，`unknown_capability` 语义不变。**Gate：`CapabilitySnapshot.default()` 的 `snapshot_id` 与 `revision`（当前为 4）完全不变**——若变了就不是纯重构。
- **L1 可撤销挂载/卸载 + 声明式启用。** 注册返回 disposer，卸载即撤销并顶 revision；`enabled` 由外部 profile/patch 驱动。Taiji 可在预算内启用/停用已有能力行——这已是真实的身体改变。**Gate：** 卸载后 `snapshot_id` 变化、旧 evidence 判 stale、重新注册恢复可用；patch 按 `id` 整体替换而非深合并。
- **L2 解锁写入。** 用显式风险策略替换 `workbench.py` L447 的硬拒绝，只允许 `reversible=True` 的写入，并要求 before-state 捕获与事务回滚。这才是越过只读的真实解锁。**Gate：** 写失败必回滚、before/after 双 digest 可追溯、不可逆能力仍被拒。
- **L3 Taiji 提出的组合效应器。** 新能力 = 已有效应器的组合，参数 schema 由真实证据导出，不做自由源码生成；走 `proposal → shadow → holdout → lesion → 原子合并`，失败留墓碑。**Gate：** 提案带父 lineage 与预算；失败候选不污染已验证清单。
- **L4 沙箱内的源码生成效应器。** 对齐 `cordis_define` 语义：校验元数据、预编译、不可变包 ID、**只保存候选不激活**，激活为独立步骤。仅在 L0–L3 全绿且资源账本就位后开启。

L0–L2 属工程收敛；**L3–L4 才是 §9 意义上的结构自进化**，必须携带结构 revision、parent/child lineage、资源账本、提案状态、tombstone 和复现 manifest（见本节开头的 checkpoint 要求）。

## 10. 共同工程纪律、提交与 CI

1. 一个提交只收口一个 owner 明确的 slice；计划、manifest、实现与测试必须同批更新，不能以“后补测试”跨越 Gate。
2. 修改训练路径前先运行 checkpoint 保存/恢复/继续训练的最小测试；训练实验的输出、seed、manifest 和 checkpoint digest 必须可关联。
3. 每个 slice 至少运行其受影响的 Python / frontend 测试、静态检查和必要的构建；CI 失败先修 CI，再进行下一工作包。历史通过记录不能代替当前差异验证。
4. S0、S1、S2 的证据分别记录；只有 S2 成功后，才将一次性调试记录从 active/reference 移入 archive。
5. 长期核心合同、当前未关闭缺口和当前实现事实留在 active/reference；已替代路线、试验日志和失效方案归档。删除任何目录或脚本前先证明无运行时、构建、测试或发布引用。

## 11. 停止条件与路线更新

出现以下情况时暂停自动推进，先修正事实源并单独提交：API owner 不清、checkpoint 不可恢复、Gate 的红测不红、CI 失败、前端投影与真实 runtime 不一致、或需要改变“语言 provider 不参与 cognition”的边界。每完成一项 Gate，更新 [03_CURRENT_EXECUTION.md](../../active/roadmap/03_CURRENT_EXECUTION.md)、实现状态和对应 manifest，再选择其顺序中唯一后继项。
