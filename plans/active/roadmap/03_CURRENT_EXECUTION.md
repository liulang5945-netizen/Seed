# N主线当前执行

更新：2026-10-25，Asia/Shanghai。本页维护唯一活动队首与本轮收官合同；[10](10_CONTINUOUS_CAPABILITY_EXECUTION_PLAN.md)维护详细WBS，[09](09_NEXT_MAINLINE_PLAN.md)维护机制继承，[07](07_MINI_MODEL_DELIVERY.md)维护能力/用户验收。

## 目标、当前动作与本轮边界

持续Goal：沿N主线，把态极发展成由自身认知机制驱动、能通过自然中文可靠完成连续任务、跨会话积累和修正经验、从反馈中学会新能力并保持旧能力的原生认知模型；每轮以同一可恢复版本的未见表现、保持和实际成本验收。

**三臂已完成训练终点，四端点准入已通过；唯一一次W5b在worker启动握手上工程失败，本轮按CLOSED_FAILURE封存。** 顺序固定为三臂训练终点→四端点恢复/准入→一次W5b→固定人工判读→结案；本轮止于R4失败，未进入R5。后续语言能力读数必须作为新的有界包重新登记，不复用本次已消费的one-shot资格。

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

11:17 执行一次W5b `--execute`（39×4＝156）。attempt `abffec0c…e0080e` 于11:18:01形成，11:18:03因 **`worker exited before authenticated bootstrap-ready handshake`** 停止；`query_attempts=0`、`persisted_generation_outputs=0`、`unknown_tail=false`，CPU 60.78秒／wall 63.95秒（均在其独立14,400秒额度内）。按合同资格已消费，不重试、不换名、不补题。

根因：adapter 的 `WORKER_MODULE` 指向 worker **库模块**（无 `__main__`），`python -m` 静默退出；已改为版本化入口 `evaluate_…_extension_v1`（其 `--worker` 分支才是文档化IPC循环），并新增回归测试（命令里的 `-m` 模块必须能响应 `--help`）。adapter 合同20支通过，冻结评测器源码未改。该修复只防复现，不构成重跑授权。

三臂实际费用合计 **4,512,256.891 / 4,560,000**，训练余量 **47,743.109** CPU秒。R5无输出可判，未执行。

有效训练累计额度仍为D1批准的v1.5：**1,590,000 process-CPU秒/臂、4,560,000总计**，旧实耗、reserve与未知尾账全部保留；amendment `21e0497e`，锚sequence 66699，来源见[18预算裁决](../../reference/delivery-runs/20261021-n-mainline/artifacts/18-d1-budget-decision.md)。warm的result/readiness为发布前下界1,580,449.388716，completion/账本为1,580,449.419966；不把发布前receipt误称完整退出全成本。

续训历史中两次并行曾因长锁/跨臂尾账及Windows共享summary替换竞态停止，保留原失败、保守结算和修复（`2b985255b`、`8e48a2d05`）；最终逐臂串行到终点。旧逐时进度/ETA从Git及原progress/receipt读取，不再作为活动重启指令。endpoint/provider、独立extension evaluator/worker和来源接线的既有合成证据继续继承；真实四端点准入通过，W5b以一次`--execute`实测并在worker启动握手上失败，原始attempt/failure/result保留。

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

CI由用户另行安排的任务负责，本任务不再修改其文件或叠加修复。当前只准备本轮CC-1.6的固定人工判读与R6结案交接，见[02现有交接表](../../reference/delivery-runs/20261021-n-mainline/artifacts/02-plan.md)。四维口径、48行记录字段、固定分母、成本口径和R1–R6证据栏已明确；不新建平行计划，不预建评测保留路径、不读取题面或提前生成输出。当前准备完成，正式判读仍等待原W5b真实持久化输出。 用户后继授权的客户端对话/生命页修复及DSH选择性借鉴已完成本地构建、回归和隔离页面验收，范围及暂缓项见[08](08_UPSTREAM_SYNC_PLAYBOOK.md)与[现行体验修订](../../reference/UI_LIFE_UX_SPEC_20261003.md)；未整体升级Harness或发布安装包，不追加为本轮研究收官条件。

2026-10-25后继方案修订：完整架构/驱动力/VX0–VX7工程包统一归[VISION](../../reference/VISION_FUTURE_TECHNOLOGY.md)，N只通过09的J1信息消费/J2信用/J3连续状态与条件选择取必要切片。生命页新发展运行台按[VX6设计](../../reference/UI_LIFE_UX_SPEC_20261003.md)分段实施，当前已交付的是连接/未知语义修复，不代表新五视图或自主策略已实现；VX6.0只读研究观察器可独立先行。此设计修订不改本页CC-1终点、预算、冻结面或研究队首。

## 结案与本轮之后

CC-1 已按 **CLOSED_FAILURE** 结案，唯一结案文件为 [PLAN-R2-DAILY-DIALOGUE-CONTINUATION-CC1_execution_closeout_20261025.md](../../reference/PLAN-R2-DAILY-DIALOGUE-CONTINUATION-CC1_execution_closeout_20261025.md)：R1–R3 完成、R4 一次性W5b工程失败且资格已消费、R5无可判输出、R6落盘。工程故障不写作模型能力负结果；三臂的完整可恢复终态保留。

**唯一下一步：进入 CC-2/VX0**——先整理已消费证据与错误表，再选择唯一必要诊断（信息是否进入决策路径），并按有界包重新登记问题、对照、固定分母、完整预算与停止线。不在本轮追加训练、复用已消费资格或补题。

当前三臂无需新的训练启动；现有v4入口只保留来源/历史恢复用途。J/VX架构和驱动实施另按其合同，不延长本轮。

## 历史与计划去向

[17过渡矩阵](../../reference/delivery-runs/20261021-n-mainline/artifacts/17-plan-transition-and-reconciliation.md)登记旧计划的继承、合并、退役与后继准入；[历史全文恢复](../../reference/PROJECT_HISTORY_CONSOLIDATION_20261023.md)、[更早N历史](../../archive/history/N_MAINLINE_EXECUTION_BEFORE_CONSOLIDATION_20261022.md)保留旧判断/失败。历史0/12不套到未测新候选，训练步数不代替语言能力；整项目尚未完成，M阶段限定收束不补发L2/L3/L4。
