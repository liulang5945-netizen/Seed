# N主线当前执行

更新：2026-10-25，Asia/Shanghai。本页维护唯一活动队首与本轮收官合同；[10](10_CONTINUOUS_CAPABILITY_EXECUTION_PLAN.md)维护详细WBS，[09](09_NEXT_MAINLINE_PLAN.md)维护机制继承，[07](07_MINI_MODEL_DELIVERY.md)维护能力/用户验收。

## 目标、当前动作与本轮边界

持续Goal：沿N主线，把态极发展成由自身认知机制驱动、能通过自然中文可靠完成连续任务、跨会话积累和修正经验、从反馈中学会新能力并保持旧能力的原生认知模型；每轮以同一可恢复版本的未见表现、保持和实际成本验收。

**CC-1与恢复v1两次正式attempt按工程失败封存；恢复v2按CLOSED_COMPLETE收口（四端含基线不可读、无配方排序）；CC-2的N1-167/168/169/170/171/172已完成：N1-170逐字节证据＋位置寻址在已见4题上把TF从0.68升到0.99、自由匹配从0–3.9%升到0–78.1%；N1-171在16题4折轮换下因64次整批更新训练侧未拟合（训练TF中位数.535–.578）按INCONCLUSIVE_UNDERFIT结案；N1-172在折0/1把训练TF抬到.941–.953后，训练题自由匹配仍仅0–4.7%、留出绑定命中0/8，按FAIL_NO_GENERALIZATION结案，直接瓶颈前移到自由滚动一致性。当前唯一下一步是N1-173自由滚动训练探针（scheduled sampling，折0/1、≤128次更新、≤520 CPU，CC-2剩余额度内）；20,000正式对照需新预算。** 原完整路径仍为训练终点→四端点恢复→配对输出→固定判读→结案；本轮已取得真实配对读数与首个可复现中文生成。后继不重复训练，也不把记忆复现写成未见泛化。

CC-1结案与持续Goal完成分别记录。CC-2归因、CC-3语言/L2、CC-4同bundle联合能力/L3、CC-5连续经验、CC-6学习与保持/L4、CC-7扩面仍依10按证据准入；本轮不以完成全部长期阶段为收官条件。

## 当前运行与证据快照

2026-10-25只读核对三臂result/readiness及progress终态：

| 端点 | 已观察的completion（Asia/Shanghai） | 小metadata状态 |
|---|---|---|
| warm_start | 2026-10-23 23:33:32，sequence 74028 | result.complete=true，readiness=ready |
| fresh_same_capacity | 2026-10-24 13:51 | result.complete=true，readiness=ready |
| full_turn_1m | 2026-10-25 07:42:51 | result.complete=true，readiness=ready；completion记账1,521,529.860816 CPU秒 |

三臂训练终态已发布。2026-10-25 11:16 真实extension端点预检通过并持久化 `…_paired_eval_endpoints.json`（一个公开的 `endpoints_validated` metadata manifest + adapter持有的 `…_paired_eval_endpoints.json`）：parent_baseline＋三臂同snapshot、`can_query_dev=false`、`can_promote=false`。11:17 `--validate-only` 返回 ready，`expected_generation_outputs=156`，端点顺序 parent_baseline→full→warm→fresh。此前**未**读取selector/prompt、未生成任何输出；本次准入只做metadata与文件校验，四端点恢复由受监督worker在`--execute`内执行并持久化receipt。

预检打通时做了三处必要修正（合成合同18+19支通过）：①预检先激活v1.5、仅在其缺位时回退v1.4，避免用1.55M旧上限校验warm的1.58M实耗；②launch provenance新增已审查的`FAST_LAUNCH_V4_SOURCE_SHA256_CI_GUARD`集合，对应v4合同测试新增的非Windows整模块跳过（训练运行字节未变）；③adapter的baseline默认值改为零参`_baseline_receipt()`，补上冻结校验器要求的固定baseline路径。冻结评测源码未改。

11:17 执行一次W5b `--execute`（39×4＝156）。attempt `abffec0c…e0080e` 于11:18:01形成，11:18:03因 **`worker exited before authenticated bootstrap-ready handshake`** 停止；`query_attempts=0`、`persisted_generation_outputs=0`、`fresh_process_restores={}`。失败发布时测得CPU 60.78125秒／wall 63.953秒；11:20:19追加的`worker_startup_tail_audited`为`unknown_tail=true`、`charged_child_cpu_upper=null`，启动尾账尚未闭合，不能把发布前读数当最终全成本或直接退款剩余14,400秒额度。按合同资格已消费，不重试、不换名、不补题。

根因：adapter 的 `WORKER_MODULE` 指向 worker **库模块**（无 `__main__`），`python -m` 静默退出；已改为版本化入口 `evaluate_…_extension_v1`（其 `--worker` 分支才是文档化IPC循环），并新增回归测试（命令里的 `-m` 模块必须能响应 `--help`）。adapter 合同20支通过，冻结评测器源码未改。该修复只防复现，不构成重跑授权。

三臂实际费用合计 **4,512,256.891 / 4,560,000**，训练余量 **47,743.109** CPU秒。R5无输出可判，未执行。

有效训练累计额度仍为D1批准的v1.5：**1,590,000 process-CPU秒/臂、4,560,000总计**，旧实耗、reserve与未知尾账全部保留；amendment `21e0497e`，锚sequence 66699，来源见[18预算裁决](../../reference/delivery-runs/20261021-n-mainline/artifacts/18-d1-budget-decision.md)。warm的result/readiness为发布前下界1,580,449.388716，completion/账本为1,580,449.419966；不把发布前receipt误称完整退出全成本。

续训历史中两次并行曾因长锁/跨臂尾账及Windows共享summary替换竞态停止，保留原失败、保守结算和修复（`2b985255b`、`8e48a2d05`）；最终逐臂串行到终点。旧逐时进度/ETA从Git及原progress/receipt读取，不再作为活动重启指令。endpoint/provider、独立extension evaluator/worker和来源接线的既有合成证据继续继承；真实metadata预检通过，但评价新进程四端点恢复未发生；一次`--execute`在worker启动握手上失败，原始attempt/failure/result及追加尾账保留。

## 本轮CC-1的固定收官合同

完整收官必须满足R1–R6；负结果同样可以完整收官。以下是固定流程与证据条件，未新增语言能力通过率门槛。

| 条件 | 可核验的结束标准 |
|---|---|
| R1 三臂到原终点 | 每臂预训练**30,096 episodes / 14,517,912 target-byte updates**，再SFT **525 episodes / 113,209 updates**；具备**3份result、3份readiness、6个阶段final checkpoint及独立恢复证据**。partial不能替代final。 |
| R2 身份、终态与费用闭合 | parent、实际launch、source/corpus/bundle/schema、预算修订链与ledger可核验；每臂唯一completion，终态无活writer/未结reservation。已测实耗、保守未知尾账、未实测发布/退出成本分别声明，不把发布前receipt当作退出全成本实测。 |
| R3 四端点准入 | parent_baseline＋三臂使用同一endpoint snapshot；受监督worker完成四端点恢复并持久化receipt后才读selector。文件存在本身不代表恢复通过；缺任一端点不消费题面。 |
| R4 一次固定W5b | **39个唯一family×4端点＝156次generation**；attempts＝outputs＝persisted＝156，每端点39，reuse＝0；沿原冻结题面、顺序、decoder与消费协议执行一次。保留原始输出及逐次费用/来源/失败。 |
| R5 固定人工判读 | 同面前**12 family×4＝48个端点回答**逐项判读中文可读性、贴题、历史连贯与重复；每端点每项分母12，填写判断、依据、不可判项及判读责任人。自动生成的review展示或机器result.complete不能替代填好的判读表。读数为描述性配对证据，不直接判L2。 |
| R6 结案落盘并提交 | 汇总R1–R5的实际完成/缺项、证据路径、收益/无收益/退化/混合或不可判、保持覆盖、实际成本与失败；在`plans/reference/`既有`PLAN-R2-…execution_closeout…`命名族形成唯一结案，日期取实际结案日；同步本页、10及黑板并提交。只保留一个后继建议，新实验另成有界包。 |

证据沿现有`reports/taiji_r2_daily_dialogue_recovery_v2_20261021_extension_v1_*`路径族；评价子族为`…_paired_eval_*`，包含endpoint manifest/readiness、attempt/ledger/progress、face与prompt SHA、generations、result/cost receipt、review或failure。不得通过改名重开一次性消费。

## 明确的终态与停止线

| 本轮状态 | 判定及允许动作 |
|---|---|
| `CLOSED_COMPLETE` 完整结案 | R1–R6全部满足。能力读数可改善、无收益、退化、混合或不可判；**无需出现正收益才能结束本轮**。本轮配方不自动追加训练或补题。 |
| `CLOSED_PARTIAL` 证据不足封存 | 硬预算/资源出口导致三臂或必需分母不完整，保存耐久partial、停止回执、已消费范围、费用/未知尾账和缺项后结案。不能判新语言能力胜败；W5b若尚未准入则不启动。 |
| `CLOSED_FAILURE` 工程失败封存 | 保存、来源、恢复、耐久发布、worker/IPC或one-shot执行失败，保存原始故障/partial、已消费范围与费用后结案；工程故障不能写为模型能力负结果。 |
| `REVIEW_PENDING` 仅待判读 | 机器输出齐全但R5/R6未齐；只完成既定判读与结案，不再训练或增加查询。此状态不是完整收官。 |

- **训练止于v1.5现有累计上限。** D1已解决，本轮不自动再次提额、加epoch、换数据/模型、降终点或改分母。任一必需臂触顶、资源拒绝或保存/恢复失败，停止后续启动，按partial/failure出口封存；正常在跑但尚无final属于等待，不能误判失败。
- 三臂已发布完整训练metadata，不再按旧并行/串行快照重复启动。历史共享账本故障和保守费用保留；评价前仍需核验真实终态、来源、恢复与当前费用门，不能仅看final文件。
- **W5b独立CPU与wall上限各14,400秒**，不借训练额度；涵盖监督/恢复/生成/IPC/终止与发布预留。receipt记录发布前实测并保留finalization reserve，不能宣称已实测receipt自身、最终result及进程退出全成本。execute形成attempt即消费一次资格，失败也不重试、不换入口/名字、不补题、不续跑。
- 无法确认进程终止时，停止派发新任务并记录终止/成本未知，不能宣称进程树已回收；普通观察超时先复核同一实际进程。该异常须处置留证，不能以重启掩盖。
- 记录本包相关检查与对应提交CI。全库既有维护债、M7发布、全部长期CR以及新的信息/信用实验不追加为CC-1收官条件；`can_promote=false`、N1-166及其未消费确认面维持原冻结。

## 独立交付与结案准备

CI由用户另行安排的任务负责，本任务不再修改其文件或叠加修复。此前已准备本轮CC-1.6的固定人工判读与R6结案交接，见[02现有交接表](../../reference/delivery-runs/20261021-n-mainline/artifacts/02-plan.md)。四维口径、48行记录字段、固定分母、成本口径和R1–R6证据栏已明确；不新建平行计划，不预建评测保留路径、不读取题面或提前生成输出。原W5b因工程故障没有产生判读输入；该固定判读规格随有界评价恢复合同承接，不填模型失败分数。 用户后继授权的客户端对话/生命页修复及DSH选择性借鉴已完成本地构建、回归和隔离页面验收，范围及暂缓项见[08](08_UPSTREAM_SYNC_PLAYBOOK.md)与[现行体验修订](../../reference/UI_LIFE_UX_SPEC_20261003.md)；未整体升级Harness或发布安装包，不追加为本轮研究收官条件。

2026-10-25后继方案修订：完整架构/驱动力/VX0–VX7工程包统一归[VISION](../../reference/VISION_FUTURE_TECHNOLOGY.md)，N只通过09的J1信息消费/J2信用/J3连续状态与条件选择取必要切片。生命页新发展运行台按[VX6设计](../../reference/UI_LIFE_UX_SPEC_20261003.md)分段实施，当前已交付的是连接/未知语义修复，不代表新五视图或自主策略已实现；VX6.0只读研究观察器可独立先行。此设计修订不改本页CC-1终点、预算、冻结面或研究队首。

## 当前唯一下一步：N1-173 自由滚动训练探针（N1-172已结案）

恢复v2已按 **CLOSED_COMPLETE** 收口：用户2026-10-25确认48行判读——四端（含parent基线）在冻结decoder下**不可读、不贴题、机械重复**，有历史行不连贯；因此本轮**不能给出配方排序**，属能力负结果而非工程失败。机器事实：`admission_001`工程失败70.703 CPU；`admission_002`四端真实恢复345.578 CPU；`formal_001`完成156/156、逐端39、reuse=0、0更新，**唯一marker在selector前发布**，发布前651.703 CPU/966.015 wall；窗口剩余9,847.531 CPU/12,740.687 wall，无未知尾账。本轮不追加训练、不补题、不再读dev面。

CC-2/N1-167已完成48/48只读矩阵，134.46875 CPU/137.125 wall，四端原件不变；configured TF字节约22%–25%、完整汉字TF约0.7%–3.4%，0/16 oracle救援、0/16其他读出救援，prompt替换实际改变分布。见[N1-167结案](../../reference/PLAN-N1-167_execution_closeout_20261025.md)，不判输入零消费，也不据此直接推翻架构。

本轮自动接力N1-168已完成固定8轮、32额外episode/9320目标字节，zero/final独立恢复通过，原parent不变。TF字节24.85%→32.10%、NLL4.396→3.775，但完整汉字TF仅2→4/148，自由前缀匹配≤0.39%，0/4小集门通过。实际fabric、predictive context与response-plan参数改变，不能判无学习；任务收益转化仍未解决。两包模型进程共253.359375 CPU/259.266 wall，低于本轮1800/2400。见[N1-168结案](../../reference/PLAN-N1-168_execution_closeout_20261025.md)。

用户已明确选择“推进共享条件语言主链（推荐）”。[N1-169](../../reference/PLAN-N1-169_shared_language_chain_design_20261025.md)按CLOSED_BOUNDED结案：工程E通过，但64 batch标定TF .658–.703、自由0–3.9%，预注册门槛未达；只读注意力诊断显示decoder在TF下也只用2–3个证据槽，把证据当固定话题向量。

**N1-170（逐字节证据＋位置寻址）已按CLOSED_COMPLETE结案**：同一64 batch/seed169/同4题，TF **.982/.992/.984/.992**、自由128 byte前缀匹配 **.577/0/.031/.781（2/4题≥.10）**、CJK 29/31、38/39、36/38、39/40、损失5.553→0.301 nats；TF使用104–121个不同源byte位置（v1为2–3），三条预注册主判据全部达成。参数970,945（v1 938,177，均≤100万），9/9测试、三份checkpoint新进程恢复、默认模型未变、0 dev；费用60.828 CPU/64.047 wall，CC-2累计381.953/393.672，余1,418.047 CPU。边界：4个已见train题、target首128 byte、单seed，属**记忆/复现**而非未见泛化；`e888c23e` TF .992却自由匹配0（生成了另一题的回答），题-答绑定尚不均匀。见[N1-170结案](../../reference/PLAN-N1-170_execution_closeout_20261025.md)。

**N1-171（16题4折轮换留出探针）已按CLOSED_BOUNDED结案，出口为INCONCLUSIVE_UNDERFIT**：选择为16 episode/16 family（prompt 183–452 byte、target 144–359 byte，selection digest `5db33945…`）；每折12训4留出、64次整批更新、每折98,304 byte曝光，参数970,945。训练TF中位数仅 **.535–.578（前置门.90未达）**，训练损失5.55→1.59–1.69且仍在下降；16个留出题自由匹配≥0.20命中 **0/16**，留出TF中位数.168–.199与错配target的.158–.195基本同档，留出完整汉字0/33–41；对照A/B均值0–2%。同一折不同留出prompt的自由输出高度同质。因此**不能**判「v2不能泛化」或「绑定失败」，只能判「64次整批更新下训练侧未拟合，留出读数处于机会水平」。5份checkpoint新进程恢复通过、7份来源SHA不变、0 dev、can_promote=false；费用485.484 CPU/493.891 wall，CC-2累计867.437/887.563，余932.563 CPU/1,512.437 wall。见[N1-171结案](../../reference/PLAN-N1-171_execution_closeout_20261025.md)。

**N1-172（fit-first曝光探针）已按CLOSED_BOUNDED结案，预注册出口FAIL_NO_GENERALIZATION**：同一selection digest `5db33945…`，折0/1各12训4留出，更新上限176、训练TF中位数≥0.90早停；两折均在96步停止（TF中位数 **.9531/.9414**，损失0.44/0.48 nats），说明「更新预算不足」已被排除。但同一批训练题的自由128 byte前缀匹配仅 **0–4.7%（中位数.0078/0）**，留出自由匹配0–0.8%、8题命中0/8；留出自身target的TF（.172–.188）与错配target（.172–.174）无差异，对照A/B均值0–0.8%。自由输出已是学到的通用句式而非乱码，但不落在目标回答上。因此出口名称按预注册如实记录，机制读数应判为「TF拟合达成、自由滚动一致性未达成、留出绑定未证明」，不能判「已记住但不会泛化」。3份checkpoint新进程恢复、8份来源SHA不变、0 dev、can_promote=false；费用379.406 CPU/387.313 wall，CC-2累计1,246.843/1,274.876，余 **553.157 CPU/1,125.124 wall**。见[N1-172结案](../../reference/PLAN-N1-172_execution_closeout_20261025.md)。

N1-172给出的唯一后继是**N1-173自由滚动训练探针（scheduled sampling）**：同一16题与折0/1、同一v2与12题整批，只改训练时喂入decoder的上一字节——按课程概率`p=min(0.5, 已更新步/256)`用模型自己的贪心字节、否则用目标字节，损失仍只对目标计算；两折×最多128次更新、训练TF≥0.90早停，上限520 CPU/1,000 wall。判据为训练题自由匹配中位数≥0.50（滚动可判前置）且8个留出题中≥4题自由匹配≥0.20并比A/B各高≥0.15。若滚动前置仍未达，则把「自由滚动一致性」登记为独立瓶颈，只在明确批准的新预算下讨论更大规模或结构改造。20,000 CPU正式同资源对照仍属未批准新预算。

旧恢复v2的结案与证据见[恢复v2结案](../../reference/PLAN-R2-EVALUATION-RECOVERY-V2_execution_closeout_20261025.md)、[48行判读](../../reference/PLAN-R2-EVALUATION-RECOVERY-V2_human_review_20261025.md)、[执行审计](../../../reports/taiji_r2_evaluation_recovery_v2_execution_audit_20261025.json)。

用户于2026-10-25**采纳[消费边界v2](../../reference/delivery-runs/20261021-n-mainline/artifacts/04-evaluation-consumption-boundary-v2.md)**，授权按原额度与固定范围自动完成工程准入、一次真实题面消费、48条判读与结案。实施计划与判据预注册见[05计划](../../reference/delivery-runs/20261021-n-mainline/artifacts/05-evaluation-recovery-v2-plan.md)。

| 项 | v2现行口径 |
|---|---|
| 不变项 | 四端checkpoint SHA、39×4=156生成、48回答判读、0参数/优化器更新、`can_promote=false`、原39-family开发面 |
| 工程准入 | 独立`admission_NNN`命名空间：真实Job/IPC/token/NTFS＋四端只读恢复；不进入selector、不读dev正文 |
| 正式消费 | 独立`formal_NNN`命名空间：冻结四端恢复→selector前耐久发布唯一marker→156生成；**marker后任何失败不重试** |
| 窗口账本 | 历史尾账预留3,400 CPU/180 wall＋v1已测84.484375/94.236＋准入500/500＋正式10,330/12,500＝14,314.484/13,274.236，低于14,400；每次操作前核对余额 |
| 失败出口 | marker前工程失败不消耗正式资格、只追加费用；marker不可读/半写按已消费停止；源码/端点漂移、余额不足、异命名空间一律拒绝 |

旧CC-1与恢复v1仍为已消费的`CLOSED_FAILURE`，原始attempt/failure/result/ledger、合同与source SHA不改；v2不追认历史资格、不换题面、不降分母。

CC-1 已按 **CLOSED_FAILURE** 结案，唯一结案文件为 [PLAN-R2-DAILY-DIALOGUE-CONTINUATION-CC1_execution_closeout_20261025.md](../../reference/PLAN-R2-DAILY-DIALOGUE-CONTINUATION-CC1_execution_closeout_20261025.md)：训练R1与训练侧R2/metadata预检完成；R3新进程恢复未完成，R4一次性attempt工程失败且资格已消费、R5无可判输出，R6记录缺项与尚未闭合的评价尾账。工程故障不写作模型能力负结果；三臂的完整可恢复终态保留。

**用户批准的恢复v1已实际执行，首个恢复任务在授权路径检查失败，按CLOSED_FAILURE结案；不重试。** metadata预检与真实bootstrap通过，但冻结授权器仍默认旧attempt/ledger，未加载模型，0/4恢复、0/156输出、未读本次题面。源/权限接线遗漏由本任务承担，不能写成模型负结果。完整事实与费用见[恢复v1结案](../../reference/PLAN-R2-EVALUATION-RECOVERY-V1_execution_closeout_20261025.md)。

2026-10-25本次推进：只读审计/真实bootstrap工具及版本化恢复provider/entry已实现；原extension worker新增可选显式provider/path注入，旧默认不变，新parent/worker同snapshot。修复原入口两次、新恢复入口一次的真实握手/Job树退出均过，0模型任务/恢复/题查询/更新，原失败四件SHA不变。最终[新入口探针回执](../../../reports/taiji_r2_evaluation_recovery_v1_bootstrap_20261025.json)及机器恢复合同已落盘；三个probe共10.4375 CPU秒、12.313 wall秒。此前机器合同关闭并实际拒绝正式入口；本次用户明确批准后启用同一合同授权字段，先metadata预检，再执行一次。

2026-10-25“下一步”协议讨论已形成[消费边界v2草案](../../reference/delivery-runs/20261021-n-mainline/artifacts/04-evaluation-consumption-boundary-v2.md)：推荐在首次真实selector/题面或缓存读取前发布耐久消费marker；此前的工程修复只增加同窗口费用，不逐次消耗新科学资格。marker存在或状态不明后，即使0输出也不得重试；不按首条成功输出计次。旧CC-1/恢复v1失败、资格和source保持原状，预算/候选/156与48分母不改；本次只交付待裁决规则，不启动v2或读取题面。

当前三臂无需新的训练启动；现有v4入口只保留来源/历史恢复用途。J/VX架构和驱动实施另按其合同，不延长本轮。

## 历史与计划去向

[17过渡矩阵](../../reference/delivery-runs/20261021-n-mainline/artifacts/17-plan-transition-and-reconciliation.md)登记旧计划的继承、合并、退役与后继准入；[历史全文恢复](../../reference/PROJECT_HISTORY_CONSOLIDATION_20261023.md)、[更早N历史](../../archive/history/N_MAINLINE_EXECUTION_BEFORE_CONSOLIDATION_20261022.md)保留旧判断/失败。历史0/12不套到未测新候选，训练步数不代替语言能力；整项目尚未完成，M阶段限定收束不补发L2/L3/L4。

2026-10-25恢复v1后继：可信provider的attempt/ledger/face三处路径已补齐；全合成4恢复/156生成使用真实token/face校验通过，真实Job/IPC/NTFS token-claim探针到达模型加载前的指定停止点并确认树退出。当前修复不刷新已消费合同SHA或重开资格。唯一后继是一次讨论工程准入与题面消费是否应分离，避免未触达题面的工程错误反复消耗科学试验资格；本轮保持失败封存。
