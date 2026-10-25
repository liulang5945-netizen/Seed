# R2 评价工程恢复 v2 实施计划与判据预注册

日期：2026-10-25，Asia/Shanghai。状态：**已执行；机器侧完成，R5 待责任判读人确认**。用户于本日选择「采纳 v2（推荐）」，授权在[消费边界 v2](04-evaluation-consumption-boundary-v2.md)的原额度与固定范围内，自动完成工程准入、一次真实题面消费、固定判读与有限结案。执行结果见[结案](../../PLAN-R2-EVALUATION-RECOVERY-V2_execution_closeout_20261025.md)。

本文件同时承担 N 主线开工前的三件套：架构条款对账（§0）、实施计划（§1–§5、§7）、判据预注册（§6）。判据先于任何真实题面访问写定，事后不得改判据或分母。

## 0. 架构条款对账

- **本包不新增认知机制**：不引入新的状态/快参数/慢参数路径，不触发 VISION 的 VX0–VX7 工程包；CC-2 归因、CC-3 语言/L2 等仍按 10 的准入另立包。
- **只执行已有条款**：VISION 的评测诚实性（同一可恢复版本、未见任务上的真实表现、真实资源成本）、N 主线的「工程故障不得写作模型能力负结果」、CC-1 的固定 156/48 分母与冻结四端点。
- **继承而不改写**：CC-1 与恢复 v1 均为已消费的 `CLOSED_FAILURE`，原始 attempt/failure/result/ledger 字节保持不变；v2 只承接其 R3–R6 缺项。

## 1. 不变项（任何一项漂移即停止）

| 项 | 固定值 |
|---|---|
| 端点 | parent_baseline＋full_turn_1m＋warm_start＋fresh_same_capacity，顺序不变 |
| checkpoint SHA | `41655eb8…`、`5654b9ea…`、`c8a1068f…`、`86b91675…`（与 v1 合同一致，执行前逐文件复核） |
| 生成分母 | 39 family × 4 端点 = **156**，逐端点 39，reuse=0 |
| 判读分母 | 前 12 family × 4 端点 = **48** 条回答，四维：中文可读性、贴题、历史连贯、重复 |
| 模型状态 | 0 参数更新、0 优化器更新；`can_promote=false` |
| 题面 | 原 39-family 开发面，不升级为独立确认集；N1-166 与其它封存面不可读 |
| 冻结源 | evaluator/selector/worker 协议不改；v2 只通过注入点（preflight/prompt_selector/worker command/budget）接入 |

## 2. 两阶段与命名空间

| 阶段 | 命名空间 | 可做什么 | 失败处理 |
|---|---|---|---|
| 工程准入 | `…extension_v1_paired_eval_recovery_v2_admission_NNN` | 真实 bootstrap、真实 Windows Job/IPC/token/NTFS、**真实四端点只读恢复**；不进入 selector，不读 dev 正文 | 每次独立 ID 与耐久费用；marker 未发布时可在剩余额度内修复并升序号重来，不消耗正式资格 |
| 正式消费 | `…extension_v1_paired_eval_recovery_v2_formal_NNN` | 冻结四端恢复 → 发布唯一 marker → 真实 selector → 156 生成 | marker 前失败=工程失败，可升序号重试（受余额约束）；**marker 后任何失败都不重试、不改名、不补题** |

- 命名空间轮换只分配工程 ID，绝不重置窗口账本；同一 `evaluation_window_id` 累计所有阶段。
- 工程阶段只允许合成模型/语料做完整链验证；真实四端恢复属准入（不读题面），可用真实 checkpoint。
- 进入正式阶段前按冻结恢复语义复核四端模型、初始状态与源码快照，不把准入残留带入比较。

## 3. 统一面访问入口与耐久 marker

1. `provider.preflight()`：授权/旧证据/窗口余额/端点 SHA/源码快照的单一入口。marker 不存在时放行；marker 存在时仅当当前正式命名空间的 attempt 与 marker 的 `attempt_id` 一致才放行（供冻结 execute 在结束前复检），否则拒绝。
2. `provider.make_prompt_selector(paths)`：冻结 execute 唯一的题面入口。内部顺序严格为：**核对 marker 不存在 → 绑定当前 attempt → 耐久发布 marker → 追加窗口账本 → 调用原 `_select_dev_prompts()`**。发布失败即失败，不降级为「没读到就算没消费」。
3. v2 worker：`restore` 走 provider 的 preflight/paths；`generate` 前额外要求 marker 已存在且 face 路径属于本命名空间。
4. 只扫描 `split` 字段的 train 语料加载不触发消费；进入 dev selector（含 family/episode 选择、缓存命中、把已选正文交给审阅者）一律视为消费。
5. marker 固定路径跨命名空间唯一，内容绑定 `attempt_id`、attempt/readiness 摘要、四端 checkpoint SHA、source 快照摘要、selector PID 与时间；不可读/半写/状态不明一律按「可能已消费」停止。

## 4. 窗口账本与额度

外层窗口继承 **CPU 14,400 / wall 14,400**，不新增额度、不改训练三臂总账。

| 项 | CPU 秒 | wall 秒 | 说明 |
|---|---:|---:|---|
| 历史未知尾账预留 | 3,400 | 180 | v1 已批准的条件上界，保持不动、不退 |
| v1 已测费用（validate+探针+恢复） | 84.484375 | 94.236 | 40.71875＋10.4375＋33.328125；43.891＋12.313＋38.032 |
| v2 工程准入上限 | 500 | 500 | 真实 bootstrap＋四端真实恢复＋退出；单端恢复预留 60 秒（任务硬上界仍为冻结的 120 秒） |
| v2 正式一次上限 | 10,330 | 12,500 | 冻结整跑 reserve 10,200＋约 65 秒实测预检＋300 秒发布/退出预留 |
| 合计 | 14,314.484 | 13,274.236 | 低于 14,400；CPU 余 85.516、wall 余 1,125.764 |

首次准入（`admission_001`）在保护区计算处停止：实测 70.703 CPU / 71.328 wall，0 恢复、0 题面、树已退出；该费用已入同一账本，未消耗正式资格。准入恢复的保护区由冻结的 120 秒/端调为 60 秒/端（任务硬上界不变），这是工程阶段的风险分配，不是正式阶段改动。

- 每次操作前用同一账本核对剩余量；余额不足以覆盖「尚未完成的 step＋固定 reserve」就不开始下一动作。
- 已知实耗、条件上界与未知分别列明；未知尾账没有可审上界时停止，不先跑再补账。
- 正式阶段的 CPU 上限可按准入实测下调（不得高于 10,330，且不得低于冻结 whole-run reserve 10,200＋实测预检）。

## 5. 预注册的停止与失败出口

| 情形 | 判定 |
|---|---|
| marker 发布前任何工程失败 | 不消耗正式资格，费用入账；修复后升序号准入，不写模型负结果 |
| marker 已发布、生成 0/部分 | 正式资格已消费，禁止重试/换名/补题；按 ENGINEERING_ABORT 或 CLOSED_FAILURE 记录 |
| marker 存在但不可读、半写或状态不明 | 按可能已消费停止；不得删除 marker 重开 |
| 源码/端点/readiness 在 selector 前后漂移 | 立即停止，保存原始证据，不发布 result |
| 余额不足、磁盘<512 MiB、可用内存<8 GiB | 不启动下一动作，记录具体缺项 |
| 异命名空间/异 face/旧 prefix 复用 | 拒绝；不得以新 prefix 清账 |

## 6. 判据预注册（SPEC，先于执行）

| 出口 | 判据 |
|---|---|
| `CLOSED_COMPLETE` | 四端 restore receipt 齐全且 `equal=true`；唯一 marker 存在；attempts=outputs=persisted=156 且逐端点 39、reuse=0；48 条判读表逐项填写（含依据与不可判项）；R2 费用闭合；结案文件落盘并提交 |
| `REVIEW_PENDING` | 机器输出 156/156 齐全、marker 与费用闭合，但 48 条人工判读未填；只允许完成既定判读与结案，不再训练或增加查询 |
| `CLOSED_FAILURE` | 保存/来源/恢复/发布/worker/IPC/生成失败；保存原始故障、已消费范围与费用；marker 状态必须显式声明 |
| `CLOSED_PARTIAL` | 硬预算/资源出口导致必需分母不完整；保存耐久 partial、停止回执、已消费范围与未知尾账 |

- 读数只作描述性配对证据，**不设新的语言通过率门槛**，不据此宣称 L2/L3/L4 已通过。
- 工程故障不得写成模型能力负结果；0 输出不得反证未消费。
- 事后不得改判据、删难题、降分母或换模型；负结果/无收益/退化/混合均可完整结案。

## 7. 实现与验证清单

| 项 | 内容 |
|---|---|
| 新 provider | `scripts/training/taiji_r2_evaluation_recovery_v2.py`：合同、窗口账本、命名空间、marker、preflight、selector、budget |
| 新入口 | `scripts/training/evaluate_taiji_r2_evaluation_recovery_v2.py`：`--admit` / `--validate-only` / `--execute` / `--worker` |
| 新 worker 门 | 生成前强制 marker 存在且 face 属于本命名空间 |
| 契约测试 | marker 前失败不消费但计费；marker 后 0 输出拒绝；marker 半写/不可读按已消费；旁路/异命名空间/源码漂移/余额不足/端点漂移全部拒绝 |
| 门禁 | `ruff`、`black`（100 列）、`mypy`、相关契约测试；一次只验证一个变量变更 |

## 8. 唯一后继

按 §7 实现并通过门禁 → `--admit` 真实四端恢复 → `--validate-only` → 一次 `--execute` → 48 条判读 → 结案并提交。除此之外不并行开启新实验、不改训练，也不提前读取题面。
