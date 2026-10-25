# Seed 客户端「生命」页实施设计

更新：2026-10-25。现行版本：面向下一主线的原生发展观测与干预方案；原路径保留以兼容已有引用。模型架构、状态所有权和VX0–VX7顺序由[VISION](VISION_FUTURE_TECHNOLOGY.md)统一维护，本页只规定VX6的页面、接口映射、开发子包与验收。

## 0. 版本裁决与当前事实

2026-10-03的“固定六区块、全部默认展开、不加页内导航、不新增locale、粘贴整份CSS”的约束退出现行设计。它们服务于旧运行时控制面，不能继续约束目标、条件选择、反馈和能力证据的呈现。旧全文可从`c97d74d00:plans/reference/UI_LIFE_UX_SPEC_20261003.md`及同提交的视觉文件恢复，不在工作树新建整篇副本。

保留已交付的正确语义：未知不记为零/空闲/空名册；来源独立可用；显式刷新真实重读；原危险操作需要确认；断线及锁状态限制动作。对应本地修复已在`24d7232f3`实现，当前生命页和接口的既有定向验证分别为65项和123项；这些证据不覆盖下述尚未实现的页面或模型自主性。安装包发布另行验证。

现有页面从`LifeSnapshot`读取当前连接的运行时，不观察全部研究进程。`/api/life/status`及调度/喂食/睡眠/玩耍控制属于Legacy路径；`homeostasis.py`是原生调节原型，不能因两者都叫“生命”而合并。`DevelopmentState`、`GoalState`、`HomeostaticState`等模型状态已经存在，但还没有本页所需的完整`DevelopmentSnapshot/DevelopmentEvent`观测协议；以下拟新增字段一律不算已实现。

当前N主线仍只按[03](../active/roadmap/03_CURRENT_EXECUTION.md)完成R2既定训练和配对结案。VX6.0的只读研究任务观察可作为独立维护先行；其余子包随对应原生数据和权限逐段接入，不把整页完成追加为CC-1收官前置。

## 1. 页面目的与所有权

用户来这里应能回答四个问题：现在观察的是哪个模型/任务；它在追求什么和根据什么选择；结果与学习是否经过核验；自己能安全改变什么。生命页是这些事实的观测窗口和授权操作入口，不能成为替模型决定探索/睡眠/训练的第二套大脑。

| 所有者 | 负责的事实/选择 | 页面可以做什么 |
|---|---|---|
| Taiji | 持续目标/承诺、缺口估计、原生候选与条件价值、所选行动、模型状态与学习记录 | 展示原生快照和证据；提出带作用域的人工请求 |
| Seed runtime/Host | 身份、实际执行回执、权限/资源账、进程观测、恢复与发布 | 读取、核验和执行获准操作；清楚呈现拒绝与结果未知 |
| 研究训练器 | 自己的launch、耐久游标、result/readiness、成本账和停止回执 | 在独立“研究任务”范围只读观察，不伪装成对话中采用的模型 |
| Legacy兼容层 | 旧生命调度与对应动作 | 仅在“兼容工具”内独立标识；不回填Taiji缺失状态 |

“兴趣/疲劳/压力”若继续展示，必须带“原型估计/已校准/未验证”及实际依据。默认主视图不显示“饥饿80%”“意识已觉醒”或无证据的成长总分。

## 2. 信息架构与布局

保留侧栏现有`life`入口和主面板id，不新增第二个生命应用。页内改为五个任务视图，默认“总览”；宽屏用单行页签，窄屏用同一选择控件，保持可访问名称和地址语义一致。页签只改变呈现，不切换模型或运行任务。

```text
生命                         [观察对象：当前对话运行时 ▾] [刷新]
身份：模型/bundle/状态版本    来源：原生/研究任务/兼容层   最近成功读取/连接情况
[总览] [目标与选择] [训练与证据] [资源与权限] [兼容工具]

总览
  当前重点：已完成/进行中/等待/尚未提供原生调节
  [目标/承诺摘要]   [最近一次选择：做什么、为何、结果是否已核验]
  [真实资源与限制] [能力/保持证据摘要]
  需要用户处理：信息不足/待授权/失败或结果未知；无事项时不制造警告
```

连接状态与来源范围始终可见；技术错误详情折叠。总览只列可核验摘要，详细目标/候选/输出/费用在各自视图，不在五处复制同一事实。未知分区保留简短“此模型尚未提供”，不铺满空卡。操作对象与数据对象一起显示，运行时离线不抹掉独立研究任务的已读取状态。

| 视图 | 主要内容 | 主要动作 |
|---|---|---|
| 总览 | 身份、当前重点、最新选择/真实结果、资源、需要处理的事项 | 刷新、进入对应详情；有合格请求才显示处理入口 |
| 目标与选择 | 目标/承诺、完成依据、候选与价值分量、实际选择、事件链 | 查看依据；原生接口就绪后可请求取消某目标，不直接编辑模型分数 |
| 训练与证据 | 当前运行时训练、已注册研究任务、学习回执、新旧能力与恢复记录 | 查看只读研究详情；运行时训练控制保留在自己的范围内 |
| 资源与权限 | 已授权范围、实际费用与未知尾账、存储/计算、可执行操作及拒绝原因 | 查看范围；通过明确请求更改授权，不由页面替模型加预算 |
| 兼容工具 | Legacy调度/喂食/睡眠/玩耍、旧知识库与必要运行维护 | 保留当前动作及确认，醒目标“兼容工具，非Taiji自主策略” |

训练数据、检查点、知识文件管理保留真实文件操作，不再作为默认生命总览的主要内容。训练文件仍先于知识文件声明；改造时以独立label定位测试，去除仅依赖picker下标的布局约束。既有`#life-source/#life-readings/#life-training/#life-training-data/#life-checkpoints/#life-knowledge/#life-consolidation/#life-host`入口映射到相应视图并定位目标，不能留下失效深链。

| 兼容锚 | 进入的视图/子区 |
|---|---|
| `#life-source` | 总览/连接与来源 |
| `#life-readings` | 总览/原生状态；Legacy读数给出到兼容工具的明确入口 |
| `#life-training`、`#life-training-data`、`#life-checkpoints` | 训练与证据/当前选定对象对应子区，不自动切模型或启动任务 |
| `#life-knowledge`、`#life-consolidation` | 兼容工具/旧知识文件与材料整理；新的原生学习证据使用训练与证据的独立区块 |
| `#life-host` | 资源与权限/当前运行时 |

深链先选择视图再定位子区；未接入能力仍有诚实说明，不把旧锚改成执行命令。观察对象改变时取消待确认意图，已发送操作仍按原subject/operation_id结算，不能把回执或第二次确认应用到新对象。

### 2.1 旧设计逐项去向

| 旧主页面元素 | 新位置与替代关系 |
|---|---|
| 来源与新鲜度 | 常驻来源/身份条与恢复带，不再独占一张大卡 |
| 原生需求/驱力百分比 | 总览只放有依据的摘要；原型值进入目标与选择的调节详情，常驻原型/校准标签 |
| 启动/停止调度、喂食/睡眠/玩耍 | 退出原生主视图，保留在已挂载Legacy的兼容工具；新自主窗口命令另按VX5/VX6.3接入，不把旧路由改名复用 |
| 训练控制、数据集、checkpoint | 训练与证据按观察对象展示；研究任务只读，runtime人工控制继续有自己的作用域 |
| 旧知识库与记忆/巩固 | Legacy RAG文件维护进入兼容工具；原生记忆/巩固只展示对应VX的来源、操作与结果，不把检索文档数当Taiji知识能力 |
| 宿主/鉴权/内存 | 资源与权限；关键连接/授权拒绝同时在总览恢复带可见 |

Legacy未挂载时不为填满兼容视图额外装回模块。旧路由的兼容行为留在其原范围，任何退役由消费者核验后独立删除；新设计不依赖Legacy才能成立。

## 3. 观察对象、状态与空态

### 3.1 观察对象不是默认模型切换器

选择器区分“当前对话运行时”“研究任务/<run-id>/<arm>”“Legacy兼容层”。每个范围独立展示`modelBundleId/checkpointId/manifestId`等适用身份；未提供的身份显示未知。研究候选即使有final，也标“研究候选，未由本页采用”；选择观察它不加载checkpoint、不占用训练writer，也不修改对话路由。

### 3.2 共同观测封套（拟新增）

所有新投影携带`schemaVersion、sourceKind、sourceId、observationId、observedAt、lastSuccessAt、availability、subjectIdentity`。`availability`区分`unsupported/loading/fresh/stale/unreachable/refused`；操作还读取作用域内的许可和锁。未知字段缺省，不填0、false或空列表；显式空列表才代表成功读取无条目。老运行时继续消费现有`LifeSnapshot`，不能把它强制解释为新协议。

| 情形 | 用户看到什么 | 操作规则 |
|---|---|---|
| 不支持原生目标/调节 | “此模型尚未提供原生发展观测”及已知版本 | 不显示可用的“自主运行/睡眠学习”按钮，不用Legacy补齐 |
| 连接失败或鉴权拒绝 | 当前来源未读取、最近成功时间、恢复入口；有缓存则明确“旧读数” | 写操作停止；仍可查看其他健康来源 |
| 等待/空闲已被原生策略报告 | 等待理由、恢复条件和实际成本 | 不把“没有输出”判坏，也不因等待自动开训 |
| 目标完成/取消 | 完成证据或取消回执 | 不重新制造同一目标，不用取消请求发出代替已取消 |
| 执行已发出但结果未知 | 相应decision/action id与“结果尚未核验” | 禁止重复点击重放同一外部动作，先依原生事务恢复 |
| 研究日志较旧、没有进程身份 | “最近一次耐久进度/未核验进程是否存活” | 不仅凭writer锁、PID或mtime写“正在训练” |
| 单个研究报告缺失/尾部写入中 | 该run的缺项/读取时点；可保留最后成功读数 | 不清空其他run，不将partial转为final |

## 4. 字段、数据来源与依赖

本节是待实施的数据映射，字段名为拟名。模型数据先由对应VX包产生，runtime只读投影，Host只做传输/核验，UI不推断模型内部目标或构造奖励。运行时实际参数学习仍由模型/训练合同管理。

| 显示字段/对象 | 来源与现有触点 | 缺口及拟实现位置 | 页面呈现/禁止推断 |
|---|---|---|---|
| 活跃模型与配置checkpoint | 现有`LifeSnapshot.artifacts`、`/api/artifacts`；health来自`/api/runtime/status` | VX6.1补bundle/state schema身份，不能只拿文件名匹配模型 | 活跃、下次配置、研究候选分列 |
| `GoalCommitment`目标/承诺 | 现有`CognitiveState.goals/GoalState`可参考；模型目标连续性由VX1负责 | VX1提供goal id、来源、状态、完成准则、证据及取消记录；投影进拟`DevelopmentSnapshot.goals` | 进度只有可测分母时显示；未知不生造百分比 |
| 知识/技能缺口 | 现有`DevelopmentState.capability_gaps`只是标识；VX1/3/4负责可学习信息和信用 | 拟gap id、任务范围、校准不确定性、误差证据、可学性估计、评估版本 | 标识存在不等于已定位根因，不把高误差直接标“应该探索” |
| `ResourceSnapshot`真实资源 | Host实际账、权限；原生`RecoveryBudgetState/DevelopmentState`仅承接适用范围 | VX1/4确定预算所有权；VX6.0读取研究账的实耗/未知/预留分列 | CPU秒/字节/步等带单位；不拿fatigue替代剩余预算 |
| 调节/情绪式状态 | 现有`HomeostaticState`及`homeostasis_status()` | VX2/5输出测量/估计方法、来源tick、策略版本与作用；无此记录仍标调节原型 | 原型状态不是主观感受；不据数值高低臆测危险程度 |
| `DecisionTrace`条件选择 | 现有`ExecutiveCandidate/Decision`与原生action intent/outcome只是局部触点 | VX2/3产出候选集身份、条件价值分量、所选项、模型/人工/固定策略来源与关联goal ids | 分数是估计而非成功；解释文本若无因果读数只标“模型解释” |
| `VerifiedOutcome`真实结果 | 现有`Outcome`、workbench执行与回执链 | VX3/4提供计划/已执行/结果未知/已结算及证据id，不由UI自己结算 | 已发送、工具返回、任务完成分开 |
| `LearningReceipt`学习/保持 | checkpoint/训练记录是资产，VX4负责独立收益与旧能力保持 | 拟parent/child bundle、允许更新组、实际更新、评价/保持范围、结果/成本、采用状态 | loss下降、更新次数、文件更大不显示为能力已成长 |
| `DevelopmentEvent`连续事件 | 由VX1/2/4/5原生生命周期生产 | 单调事件序号、subject identity、goal/decision/outcome/receipt关联和来源；VX6.1提供有界分页 | 恢复可重放去重，UI不补造缺失事件 |
| 研究任务及臂状态 | 现有R2 launch/result/readiness/progress/CPU ledger等小metadata | VX6.0新增Host侧只读reader与R2报告适配；不读取corpus/selector/generations或大checkpoint | durable/未耐久/停止/完成证据分别列，不改旧训练writer |

### 4.1 接口与代码路径

继续复用`packages/api/life-controller`的Host/Client/Remote和`ui-life`页面；不另造一套全局订阅。现有`LifeSnapshot.training/checkpoints`缺失语义、`ctx.life`命令和`baseline→snapshot`恢复流保留。

- VX6.0拟在`life-controller/src/research-reader.ts`增加只读reader，在Host配置登记允许的`projectRoot/runPrefix/adapterSchema`及run身份；`src/types.ts`增加`ResearchRunView`与逐来源封套，作为独立`researchRuns`投影。来源不经localhost runtime，所以runtime离线也能读到已注册run。拟新增`researchRunDetails(runId)`只返回有界metadata；浏览器不能提交任意文件路径，也没有run start/stop/reset动词。
- VX6.1由`api/seed_runtime.py`增加只读`development_status()`投影；在现有`api/routes_runtime.py`承接拟`GET /api/runtime/development`及拟`GET /api/runtime/development/events`有界事件读取。`TaskStateV1/GoalCommitment/DecisionTrace/VerifiedOutcome/LearningReceipt`由对应VX模型/宿主责任方生产，不能写在HTTP路由里临时制造。返回格式命名为拟`DevelopmentSnapshot/DevelopmentEvent`，按已授权`subject_id`读取，事件使用`cursor/limit`有界分页；读取只使用已存在的状态，不因GET自行加载模型、挂载控制器、推进tick或生成决策。
- `life-controller/src/runtime-client.ts`按能力声明探测该新接口，`404`标unsupported；`src/types.ts`新增可选`development`与事件游标。Host `feed.ts`按来源轮询并隔离失败，Client model只安装观测，`ui-life`从现有声明的响应式来源读取。新字段先只读，无客户端自写“模型已学习”分支。
- VX6.2将`LifePanel.tsx`拆成页面内拥有的总览/目标选择/训练证据/资源权限/兼容工具组件；共享slot仍由`ui-life`自己的registration声明。筛选和页签是UI状态，模型数据仍在Client对象层；不为临时方便把服务对象传给新增业务组件或新增跨feature运行时导入。
- VX6.3只在对应原生命令就绪后增加目标取消/暂停自主窗口等typed Remote请求；必须携带subject identity、操作id和期望状态revision，服务端做许可、锁与幂等/冲突检查。`trainPause`仍只暂停当前runtime训练；不得复用它伪装“暂停自主运行”。

新Remote/DTO需同批更新Host/Client Typert生成物、API目录、所有消费者和文档；按现有Host→Client→Web有序构建验证，不能只重新打一个UI bundle就宣布接线完成。

### 4.2 研究任务只读reader规则

注册对象是Host配置中的run manifest身份，不是UI硬编码`E:/Seed`或某个R2文件名前缀。首个适配器仅支持已声明R2 schema；其他格式标unsupported，不按猜测解析。读取活动JSON/JSONL时允许末尾未提交行，公布最近完整记录；采样字节、分页条数、轮询周期由实施配置给定并在测试中钉住上限，不全量扫描大日志。

首版待实施的工程默认值明确为：5秒轮询，最多8个观察对象，单对象progress尾读64KiB，单份metadata上限256KiB，单轮总读取上限4MiB，事件/详情每页20项；作为Host配置校验，不写死成不可扩展协议。超限标该来源读取受限并保留游标，按对象轮转续读，不能由单个大来源耗尽每轮预算；不转为全量扫描或无限重试。性能验收按这些配置执行，扩大上限需重新记录读取成本。

run/arm身份、终点计数、耐久receipt/result、cost口径从各自来源核对；进程存活单独依Host可核验的PID创建身份/退出观测，没有这项就不显示running。单一final文件、writer锁不存在或CPU summary存在都不能独自代表实验完成。任何reader失败只影响该来源，不写checkpoint、账本、progress或预算修订件。

## 5. 操作与用户流程

| 操作 | 对象与前置 | 用户看到的反馈 |
|---|---|---|
| 切换观察对象/刷新 | 只读；run注册且来源可读 | 来源及观测时间更新，不切模型、不启动任务 |
| 查看研究任务 | manifest身份、臂、报告revision已知 | 耐久进度、终态证据、费用口径与缺项；无训练控制按钮 |
| 查看原生目标/选择 | 对应VX快照提供该字段 | goal→candidate→decision→execution→outcome→learning链；缺环显式断开 |
| 提交/取消目标 | 目标服务支持、用户作用域与revision匹配 | 请求中→接受/拒绝→原生状态回执；接受请求不等于目标已完成 |
| 暂停/恢复自主窗口 | VX5调度器已存在、窗口身份与剩余授权明确 | 执行机会停止/恢复的回执；不是关掉底层研究训练 |
| 手动触发巩固/探索 | 对应操作已证明有效、允许更新范围/资源明示、对象明确 | 清楚标“人工干预”；不能据此记原生自主选择通过 |
| 改变预算/采用checkpoint | 宿主授权与相应版本准入，不由UI直接改模型状态 | 原有确认、来源/影响清单及服务端拒绝保留；研究候选不一键默认采用 |
| Legacy动作 | Legacy确实挂载且健康 | 固定在兼容工具，解释作用域；不显示为Taiji学得策略 |

三个必须跑通的流程：

1. **现在的研究人员查看续训**：选已注册run→看各臂耐久计数/last observed/是否核验进程→查看final与恢复记录→只读退出。断线、尾行写入中、stale账本均不改变训练。
2. **未来使用者理解主动行动**：总览看到原生目标及“为什么现在核对”→打开候选与条件价值→看真实执行/结果→查看学习或保持证据。没有可信证据就只说“候选建议/尚未核验”。
3. **取消与资源不足**：用户取消目标/授权窗口到期→原生返回取消/等待→页面显示停止原因与未结动作；恢复后同一目标、事件和操作id不重复执行。当前已在跑的外部动作按事务结果处理，不把点下取消当作已回滚。

## 6. VX6开发子包与有限出口

此表是前端/集成排工，模型工作仍只用主VISION的VX编号，不另建一条生命主线。每包均提交对应代码、契约/行为测试、双语文案和受影响快照；机器证据沿`reports/`与既有测试artifact路径，不新增近义顶层目录。

| 子包/责任 | 前置及具体改动 | 完成条件 | 失败出口 |
|---|---|---|---|
| **VX6.0 只读研究观察 / Host集成＋前端** | 当前可独立先行。types/config→只读reader/R2适配→独立来源feed→训练证据视图；只消费已有metadata，不改训练器 | 注册一个固定R2 fixture与一个本地已存在run；离线runtime仍能看到研究记录，下述固定18类fixture覆盖尾行/缺件/漂移/停止/完成/未知进程；全部请求与文件写入断言确认无训练副作用 | schema不支持/来源不一致按run显式拒绝，发布读取限制与失败，包到此结束；不为凑running标签启动模型 |
| VX6.1 原生发展观测 / runtime＋API集成 | VX1提供持续状态；拟development接口、封套、事件游标及可选LifeSnapshot投影；完成所有消费者/生成物迁移 | 同一bundle/version/tick来源可追踪；缺字段、404、过期、部分来源健康、重连baseline/事件去重全有合同证明；读取前后模型状态不变 | 不完整则界面保留unsupported，不由UI造字段；记录具体缺口并结案 |
| VX6.2 生命页重组 / 前端＋UX | VX6.0与6.1的可用字段；现有单页拆为五视图，保持入口与深链；复用当前视觉token | 用户能在总览定位对象/重点/结果/下一可用操作；320px内容列与常规桌面、明暗色、键盘/读屏/减少动效无溢出或焦点丢失；有真实装配快照 | 交互不成立回到明确组件/信息缺口，不以装饰指标假装完整 |
| VX6.3 授权干预 / runtime＋Host＋前端 | VX2/4实际操作与可信反馈，涉及自主窗口须等VX5；增加typed命令与操作revision/回执 | 取消、窗口暂停、人工操作、重复请求、错误对象和锁冲突均通过执行器及UI测试；人工/固定/原生选择来源不混淆 | 失败保留状态未知或原对象，不自动重试外部副作用；缺API时不展示可点击占位动作 |
| VX6.4 联合采用验收 / 产品＋测评＋集成 | VX5调度记录、VX4收益/保持、VX6.2/6.3；与VX7同一bundle交付 | 模型记录→runtime DTO→Host→Client→页面逐段身份一致；同一任务可看选择、真实结果、成本及保持；恢复/取消后的行为与显示一致；用户验收范围记录并提交 | UI失败回对应投影/交互，模型失败回VX2/3/4/5；不得用页面验收通过补发模型自主性或意识结论 |

VX6.0首个fixture列表在开工时一次冻结为：正常推进、已知停止、耐久完成、仅final缺result、仅result缺恢复、partial尾行、缺文件、未知schema、摘要stale、路径越界、run/arm身份不符、未知进程、PID身份不符、发布前下界与未知尾账、超限轮转、离线runtime+健康研究源、单run失败+其他run健康、重连去重，共18类。固定列表完成或触及包内读/计算上限后结案，不自动追加题型。只读真实run验证不重新加载模型，也不消费正式生成题面。

## 7. 受影响测试与迁移清单

| 层 | 现有触点 / 拟新增用例 | 必须钉住的行为 |
|---|---|---|
| Python投影 | 现有runtime/workbench/contracts相关测试；拟`tests/seed/test_development_status.py` | 只读、主体身份、缺失值、schema；不能因读状态改变驱力、学习或规划 |
| Host reader/API | `life-controller/tests/life-controller.host.spec.ts`、`resource-cleanup.host.spec.ts`；拟`research-runs.host.spec.ts` | 18类fixture；独立来源失败、边界路径、限额、退出清理、无checkpoint/账本写入 |
| Client/恢复 | `life-controller/tests/client-apply.client.spec.ts`与当前model/stream测试 | 可选字段、来源替换、baseline、游标/事件去重和旧状态标识；不写业务影子状态 |
| 页面行为 | `ui-life/tests/panel.client.spec.tsx`、`apply.client.spec.ts` | 五视图、对象切换、未知/等待/取消、作用域控件、确认与服务端拒绝；用label/id取picker，解除旧下标顺序依赖 |
| 真实Web装配 | 已有`apps/web/tests/life-status.e2e.ts`及owner-local expected | 离线、部分健康、研究观察、原生字段逐段就绪、锁/取消/恢复、窄宽/明暗；mock仅外部HTTP，不mock整个UI模型 |
| 发布 | 现有有序Host→Client→Web构建、TypeScript、GUI与包完整性检查 | Typert/目录/双语文案同版本，web和Electron共享同一实现；本地预览、用户采用、安装包发布分别记录 |

旧main panel id和现有runtime控制API先保留；新DTO版本须明确向后兼容或拒绝策略，不能静默把旧字段改叫学得驱力。Legacy默认不新增装配，只移动入口与作用域说明。旧主动模式按钮可以保留在兼容工具，但不得在原生总览仍以无来源“喂食/睡眠/玩耍”出现。模型根价值/参数、研究预算和训练checkpoint没有前端直接编辑入口。

视觉采用[Seed认知观测台风格](UI_LIFE_VISUAL_DIRECTION_20261003.md)：白/墨蓝灰与Seed绿、单一当前重点、真实事件脉络和舒展留白；字阶、主题token、五视图构图及审美验收以该规格为准。当前已实现API的唯一说明仍为[Life subsystem](../../taiji-harness/docs/subsystems/life.md)。本页提出的development/researchRuns DTO和命令尚待对应开发包实施，设计完成不等于接口存在或模型已自主。
