# 旧资产总索引

> 文档状态：[历史] — 本页只登记旧方向、旧实验和冻结合同的处置，不发布当前指令。当前方向见[MAINLINE](../MAINLINE.md)，队首见[03](../active/roadmap/03_CURRENT_EXECUTION.md)。

## 1. 判定规则

资产角色以[MAINLINE现行步骤](../MAINLINE.md#3-唯一执行队列)和本页具体用途为准；[reference/README合同映射](README.md#1-总计划步骤对应的原合同)只是原合同身份表，不能据“还在表中”判旧候选仍是当前主模型。已结案算法/模型可以继续作证据、基线、组件或长期储备，但不能：

- 被当前工作包引用为“下一步”；
- 作为第二套主线或第二套能力验收；
- 因文件名含 `PLAN`/`SPEC`/`NEXT` 就自动获得调度权。

旧文的“当前/下一步/唯一动作”只解释其写作时点。

## 2. 资产族与处置

| 族 | 范围/代表入口 | 处置 | 可复用部分 |
|---|---|---|---|
| N120–N174 | `PLAN-N1-17x_*`、`SPEC-N1-17x_*`、`*_execution_closeout*` | 历史实验 | 数据构造、恢复/计时仪器、失败原因 |
| N175 | [评价交接](PLAN-N1-175_evaluation_handoff_20261026.md)、[盲评](PLAN-N1-175_blind_review_20261026.md)、[原因复盘](PLAN-N1-175_prior_results_cause_review_20261026.md) | 冻结/未判面保留 | R 0/3、S 6/312 的边界；不代判、不重生成 |
| N176 | [课程计划](PLAN-N1-176_foundation_curriculum_20261026.md)、[F0/F1 合同](SPEC-N1-176_F0_F1_pilot_20261026.md)、[架构主张审计](PLAN-N1-176_architecture_claims_audit_20261026.md) | 冻结资产/证据 | 已生成课程与消费登记；不重生成 |
| N177 A–F | [实施计划](PLAN-N1-177_joint_cognition_slice_20261026.md)、[177.5 T1](SPEC-N1-177_5_T1_F0_F1_pilot_20261027.md) | 旧基线来源 | 接口、来源、恢复、失败模式 |
| N177 E1 | [E1 消费合同](SPEC-N1-177_E1_state_consumption_contract_20261029.md)、[E 决策](PLAN-N1-177_E_state_consumption_and_credit_decision_20261029.md) | 证据/历史设计 | 状态依赖任务、固定分母、cut/rescue 语义 |
| N177 F | [F 编码器](SPEC-N1-177_F_value_encoder_20261029.md)、[F1 冻结提案](SPEC-N1-177_F1_freeze_proposal_20261029.md) | 负结果/旧基线 | `corrected_old_v1` 的字节编码与失败对照 |
| N178 | [解析覆盖提案](SPEC-N1-178_parse_coverage_extension_proposal_20261030.md) | 诊断 | 解析缺口解释；不自动扩量 |
| R2 日对话/CC-1/extension | `PLAN-R2-*`、[工程计划](delivery-runs/20261021-n-mainline/artifacts/02-plan.md)、[过渡矩阵](delivery-runs/20261021-n-mainline/artifacts/17-plan-transition-and-reconciliation.md) | 冻结工程 | 恢复、评测适配、来源纪律 |
| VISION/VX | [VISION](VISION_FUTURE_TECHNOLOGY.md)、[持续发展 V2](../active/architecture/TAIJI_CONTINUAL_DEVELOPMENT_V2.md) | 长期储备 | 42 条设计条款作输入；不产生当前工单 |
| M0–M8/批准 | [M0–M7 收束](M0_M7_PROJECT_CONSOLIDATION_20261006.md)、[M5 批准](M5_EXIT_APPROVAL_20260920.md)、[M6 排除](M6_SCOPE_EXCLUSION_APPROVAL_20261006.md) | 冻结批准 | 原范围、排除、重启条件 |
| UI/产品面 | [UX 规范](UI_LIFE_UX_SPEC_20261003.md)、[视觉方向](UI_LIFE_VISUAL_DIRECTION_20261003.md) | 发布面 | 生命页字段与 VX6 子包 |
| 09 机制词典 | [09 N1–N6 映射](../active/roadmap/09_NEXT_MAINLINE_PLAN.md) | 组件参考 | 机制责任/历史映射；不是当前主计算图 |
| 01 阶段地图 | [01](../active/roadmap/01_SCOPE_AND_PHASES.md) | 历史 | M0–M8 与 N 层次追溯 |

## 3. 唯一复入口

旧资产若要重新成为当前主线，必须：

1. 在[MAINLINE](../MAINLINE.md)第 3 节登记为“候选阶段”；
2. 写一份新的当前合同，明确任务、数据、基线、预算、失败出口；
3. 通过现行步骤要求的算法/工程、定价和有界行为比较；原R4/R5编号或历史通过不自动转成新准入。

不允许通过新增一份 `PLAN-NEXT-*` 文档自动复辟。

## 4. 保留纪律

- 不删除、不改名、不重写冻结件；路径与 Git 身份是证据。
- 旧报告的数字不并入新候选成绩；同一 bundle 才能拼能力。
- 旧 sealed 面不因新主线重读；新面另立合同。
- 旧“下一步”不再调度；当前实际动作在MAINLINE的执行位置，03只提供跳转。

本轮已实际迁移/删除的旧交接、根目录CI材料及恢复身份见[2026-10-30整理索引](../archive/history/n_mainline_20261030/README.md)。被runtime/CI/测试直接读取的旧文档与源文件保留原位，不凭旧资产分类删除。

## 5. 新核心对旧算法资产的具体处理（2026-10-31）

原则：**旧模型兼容不再是新核心设计负担；经过验证的工具、证据和研究经验保留使用资格，旧能力不自动转移。** 本节是09.2资产分流与迁移边界，不是另一条执行队列；本轮仅核对源码/引用并登记，未搬删代码、权重或数据。

### 5.1 直接复用、适配与重新实现

| 资产/真实路径 | 去向 | 新核心使用条件 |
|---|---|---|
| `taiji/persistence.py`的atomic_save、`internalization.py`的content_digest、`utf8_state.py`等语法/来源工具 | 优先审计后复用实现 | 检查新payload/张量/原子提交和结束协议；只复用通用函数，不顺带继承旧模型类型、全局runtime或隐式状态 |
| `joint_task_state.py`的来源、pending/receipt/取消/重入等协议资产 | 复用语义和反例，按新主体适配 | 它绑定旧TaskStateV3，不能直接把整个JointTaskState当新脑；新H/F/参数/拓扑及journal须共同恢复 |
| `structural_lineage.py`、`structural_*`及`continual_checkpoint.py` | 复用适用的身份、血缘、预算/迁移/回退工具，适配schema | 逐项核对旧region/owner/参数映射假设；旧结构事件passed不替新增长事件签字 |
| `developmental_synapse.py`的快慢/eligibility资产，`adaptive_residual_*`的shadow、压力和出生尺度经验 | 复用经验证思想/局部工具，核心更新与连接按v1.1重实现 | 新F为key×value近端更新，H/图和信用不同；不把旧local_update或F1桥整体套成新算法 |
| `sparse.py`、`fabric.py`、`memory.py`、`neuron_region.py`、TSK adapter | 原位保留为旧底座、窄机制基线和可提取部件 | 仍有model/adapter/记忆/测试消费者及冻结源码身份；不迁走破坏消费者，不强制新核心继承固定fan-in或共享低维读出 |
| `task_core*`、`conditional_language*`、`native_content_core*`及其训练/评价适配 | 固定容量与内容学习基线保留，退出新脑默认骨干 | 保留历史配方/原代码身份；新实现只按已注册对照引用，不在旧runner上继续叠新脑 |
| 原评分器、来源审计、恢复/计时、保持评估与相关测试 | 复用量尺语义和适用通用函数，旧版本保留 | 固定分母、正确性干预、资源和真实恢复继续使用；新任务范围/门值另冻结，不把旧scorer全绿当新能力 |
| 生命周期/homeostasis/sleep/play资产与UI读数 | 保留研究部件与产品观察，按实际接口选择适配 | 固定策略是对照，真实内源策略另验；不能只挂旧sleep函数或UI面板就称新主体已连续活动 |

本轮已查到的真实依赖包括adapter/model→fabric→sparse、memory/identity→sparse、旧JointTaskState→TaskStateV3、各内容版本与旧runner/测试互相引用。因此“退出主线”先登记角色，不等于现在就能移动源文件。新实现依赖须保持显式、窄化；不能把巨型旧adapter作为隐藏的新脑运行内核。

### 5.2 模型、数据与报告

| 资产 | 处理方式 |
|---|---|
| 旧模型权重 | 新核心无需继承；正式基线端点/直接引用/恢复所需文件先保留原位并绑定hash。它们可作历史下界或明确基线，不伪称新模型父代 |
| 新旧参数迁移 | 只有建立状态/参数语义映射、函数与信用验证后才称继承。不同算法之间同形状或padding不构成知识迁移；若蒸馏/再学习要单独声明来源、曝光与费用 |
| 旧train材料 | 权利/许可和原实验用途允许时，可按新合同登记为训练或开发来源；原始来源与历史曝光不能抹除 |
| query/holdout/已消费面 | 保留历史成绩和曝光账；后继同面读数只作回归/开发，不重新称未见确认。改变用途需在新合同明确，原sealed权限不因资产复用开放 |
| 正负报告、冻结SPEC、manifest、正式输出与引用端点 | 保留原始身份、数字、失败和费用；新算法的成绩只能绑定它自己的候选/入口/数据/状态 |
| 重复快照、结束的缓存、无独有内容操作流水和可再生中间件 | 按S9先登记独有价值、引用/写者和恢复证据，再删除；当前本轮不执行模型产物清理 |

新核心的首次bootstrap允许开启新参数谱系，不能假称已无损继承旧模型能力；从新核心开始的后续生长则必须在自身已学主体上增量进行。旧模型可弃用和后续成长必须继承，并不矛盾。

### 5.3 代码和大文件实际退役的顺序

1. 09.2先列一个窄的复用依赖表：实现复用、语义适配、仅作对照/证据、明确重实现；每项有来源revision和对应反例。
2. 旧代码保持原路径供现有消费者/源码绑定读取；若要物理归档，先迁移imports、CLI、测试、默认入口和文档，确认无未迁消费者。冻结hash绑定文件不靠随手改注释实现退役。
3. 权重/中间文件清理前核对当前加载者/写者、latest/恢复指针、直接引用、最终端点及hash；活跃产物和唯一恢复点保护。冷归档需有可验证路径解析，不能只把文件移走却留下虚假的可恢复声明。
4. 删除重复过程文本时用Git提交/路径/blob保留精确恢复身份，不为相同全文再复制归档。报告、独有设计、未完成补丁不按“失败”删除。
5. 新核心有可加载替代入口并通过其范围后，再决定产品默认切换和旧实现退役；CI中旧回归是否保留/分组按消费者与证据责任裁决，不能删测试掩盖真实失败。

这一处置不另开旧算法训练或要求为新脑重跑旧全部课程。优先复用已证明可靠的工程能力，算法与新能力只按本次实际路径认定。
