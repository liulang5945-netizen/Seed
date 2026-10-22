# Phase 4 补充：R2 continuation extension v1 架构审查

日期：2026-10-22
范围：为现有 R2 自然中文多轮对话三臂训练 partial 建立经用户批准的新累计预算与续训契约。此审查只批准进入实现，不代表代码门、来源门、训练门或模型能力门通过。

## 结论

架构方向通过，可按本文件进入版本化 extension runner、账本和合成测试实现。训练仍未获本架构报告放行：实现必须通过独立代码审查、精确来源/兼容性核验、parent checkpoint fresh-process 恢复、预算与资源预检后，才可启动。

续训含义已固定为：三个 arm 分别从自己的最后一个已验证 durable checkpoint 继续，保留既有权重、游标、RNG 和训练计划，不从 zero 重训、不合并三臂权重。用户于2026-10-22 15:19将原批准的1,000,000 CPU秒/arm、3,000,000 aggregate修订为当前窗口每臂累计1,200,000秒、三臂累计3,600,000秒。额度包含各臂旧实际账与未知尾账，不是旧账之上的新增额度；此修订只改变预算ceiling，不重置charge、父来源或checkpoint谱系。

本补充取代此前计划在当前冻结 480,000 秒旧上限内继续的执行假设，也取代“从 zero 重新开始”对本次续训窗口的适用性；旧合同、旧账本与失败记录原样保留，不回写或伪装为从未发生。

## 架构与边界

采用只增不改的双层谱系：旧 attempt、旧 progress、已有 result（若存在）及原 checkpoint 目录只读；每个 arm 创建独立的 continuation attempt/progress/result sidecar。新 checkpoint 写入 `data/checkpoints/r2_daily_dialogue_recovery_v2_20261021/continuation_v1/<arm>/<stage>/`，不覆盖原目录中的 `latest.json` 或任何 checkpoint。新目录以校验过的 parent bundle 与 receipt 为初始恢复点，并以 parent receipt SHA 为 lineage anchor；原始路径和 parent SHA 留在 extension manifest 中。assistant_sft 阶段的新提交也写入该版本化根目录；原 assistant canary 与已有证据只读。

旧 attempt 没有独立 `attempt_id` 字段，因此以 `run + arm + 原 attempt 文件 SHA-256` 作为不可歧义的 parent identity。旧 attempt 仍显示 `training` 并不是成功证明；extension 自己的状态、停止原因与终点结果必须单独记录。旧 result 缺失时，新终点只能写入 extension result；之后评测器必须通过新显式 parent/extension manifest 识别端点，不能把结果伪写回旧 attempt。

不采用覆盖旧 `latest.json`/progress/result 的方案，因为它会混淆之前训练与本次续训的证据边界；不采用修改旧 `source_hashes`、将 CPU 账重置为零、或把3.6M解释为新增预算的方案，因为这些会丢失已知成本与来源身份。也不直接 monkeypatch 旧 runner 的 budget/path 常量；旧 runner 的 cap 和产物路径是冻结的，应由新版本 runner 明确实现扩展合同。

## 版本化接口与数据合同

扩展 manifest 使用独立 schema，例如 `taiji-r2-dialogue-continuation-extension-v1`，每 arm write-once 创建；事件按序号 append-only。最低字段如下：

| 字段 | 类型 / 约束 | 含义 |
|---|---|---|
| `extension_id`, `schema`, `arm` | 字符串；schema 固定 v1 | 新窗口身份，不与旧 attempt ID 混用 |
| `parent_attempt` | 路径、schema、原状态、文件 SHA-256 | 指向旧 attempt；原文件不改写 |
| `parent_checkpoint` | stage、cursor、episode/global step、bundle SHA、checkpoint digest、receipt 名称与 SHA、parent source-binding 摘要 | 指向该 arm 的最后已验证 durable checkpoint |
| `approved_budget` | `per_arm_cpu_seconds=1200000`、`aggregate_cpu_seconds=3600000`、2026-10-22修订批准来源 | 明确为累计 ceiling，包含旧账、保守尾账、预检、训练、子进程、保存/恢复及失败开销 |
| `accounting_start` | UTC 时间、boot identity、host CPU 秒、逻辑 CPU 上界、parent charged 值、未知尾账上界、初始累计上界及剩余额度 | 启动时的耐久 CPU 基线；审查快照不得代替该字段 |
| `continuation_sources` | 当前 runner/module/test/source hash 与兼容性报告 SHA | 新版本独立绑定；旧 parent source hash 保留且不重写 |
| `status` | `prepared/running/stopped/complete/failed_unknown` | 只描述该 extension，不改写旧 attempt 状态 |

事件至少包括耐久的 `continuation_started`、`parent_checkpoint_restored`、`durable_training_commit`、`continuation_stopped` 和 `continuation_completed`。在启动任何 restore worker 或训练更新前，必须用本地已验证文件系统的 write-through/原子新建语义持久化 extension intent 与预算起点；初始文件或目录耐久失败则不启动训练。每个 extension result 记录 parent identity、extension manifest SHA、最后 checkpoint receipt、训练分母、累计 CPU、停止/完成状态、0 dev/reference/generation 查询和 `can_promote=false`。

## 预算、尾账与谱系账本

旧冻结上限为 full_turn_1m 240,000、warm_start 与 fresh_same_capacity 各 120,000 process-CPU 秒，合计480,000秒。先前批准的1M/3M窗口已于15:19修订为当前每arm 1,200,000、三臂聚合3,600,000秒；旧余额不足的原因是boot identity变化后恢复规则必须对未知时间使用 `elapsed_wall × max(previous_logical_cpu_upper, current_logical_cpu_count)`。这一规则保持不变。

只读审查基线（2026-10-22 13:30:50 Asia/Shanghai；UTC 05:30:50.602314）：

| arm | 最后 durable episodes / cursor | Parent receipt SHA-256 | Parent bundle SHA-256 | 13:30历史保守累计上界 CPU 秒 | 到此前1M上限的历史余额 CPU 秒 |
|---|---:|---|---|---:|---:|
| `full_turn_1m` | 9,472 / epoch 2, index 1,948 | `8ec115196c39f9936ea3704ac5aaa8934f8efeaf8794f58e709f285ceb3a23bb` | `d4d5887fd44b68d5e3d5e42a2d8832a83ca6495ec1f43f8f5847d28ad004c89d` | 780,628.774 | 219,371.226 |
| `warm_start` | 14,528 / epoch 3, index 3,242 | `f8a2a72eec4d67ca362351158f4d11d19aee5bd044cec5b2ed7a399396b07583` | `ed74609d541f0340f0d243e9e34b496cc3061427bc572923215399dfd309c416` | 780,081.702 | 219,918.298 |
| `fresh_same_capacity` | 14,528 / epoch 3, index 3,242 | `cdf91736fd6ea1c837ed075886c478b6df430f2494ecaf1163f81eefa6ad067e` | `ca00b48dd28d0466a8c9b20cb224341af75b0e4de6b0625d55534c5413b0a9dc` | 779,196.622 | 220,803.378 |

该历史时点aggregate tail上界为2,339,907.098 CPU秒，距离此前3M上限还剩660,092.902秒。full-turn已在旧cap下写入 `recovery_stopped`，值为748,972.986107 CPU秒；warm/fresh没有该停止事件。三个parent attempts均为`training`、均无final。上表是此前预算决策的历史审查基线，不是执行许可或实耗CPU；开跑前必须从当时parent checkpoint/extension ledger、当前boot identity和wall time重新计算，若任何累计上界达到当前ceiling则拒绝启动。

### 15:20附近当前只读快照

新批准窗口的当前累计ceiling为1.2M/arm、3.6M aggregate。只读parent audit确认parent receipt/bundle/cursor仍是上表所列同一组；按冻结尾账规则重算：

| arm | 当前保守累计上界 CPU 秒 | 到1.2M的当前余额 CPU 秒 |
|---|---:|---:|
| `full_turn_1m` | 937,951.469 | 262,048.531 |
| `warm_start` | 937,416.609 | 262,583.391 |
| `fresh_same_capacity` | 936,544.407 | 263,455.593 |
| **aggregate** | **2,811,912.485** | **788,087.515** |

这是时间敏感的未知尾账上界，不是已发生的新增训练CPU；所有执行前门必须重新采样。

首次 continuation 的 `charged_start` 必须至少取 max(已知 parent CPU 事件、parent checkpoint ledger 的 charged CPU 加冻结尾账规则所得的当前未知尾账上界)，并在 intent 中耐久记录新的 boot/host CPU/wall snapshot，以免首个 extension checkpoint 前强杀后继续从更早 parent 时间重复推算或清零。后续每次恢复都以 `max(previous_cumulative_charge, latest_extension_ledger_tail_charge)` 计费；episode 内未提交的重算、准备、restore、验证子进程、checkpoint 保存和失败收敛开销均计入。每臂在剩余 headroom 不足以覆盖最坏单 episode 与失败收敛 reserve 时，不得启动下一 episode。

执行器必须有共享的、耐久且串行化的aggregate guard；并发启动要原子读取/记账三臂当前上界，并证明 `sum(charged_upper_by_arm) ≤ 3,600,000`，单臂还需满足 `charged_upper < 1,200,000`。单臂ceiling不能替代aggregate检查。该预算是一个续训窗口，不提供任意次数重启后重新获得的额度；若再次中断并触及ceiling，应停止并提交证据，任何新预算都需另行决策。

尾账增长具有时间敏感性：本机24个逻辑CPU下，若boot identity仍不匹配，wall每增加一小时，每臂上界最多增加86,400 CPU秒。因此13:30与15:20附近的余额都不能静态沿用；执行前必须重算。既有吞吐推算只是预计值，不能替代ceiling证明。

## 来源绑定与执行门

旧 attempt 的 parent source hash map 必须作为 lineage 原样保存；不得编辑旧 SHA 或把当前 SHA 冒充旧 SHA。当前源码与 parent hash 之间仍有待处理的绑定差异：

来源证明按文件性质区分。Git-tracked训练代码、构建代码、配置与合同测试必须记录精确parent/current SHA以及验证过的Git commit和blob OID/SHA；parent SHA仍来自当前绑定时，不得只凭工作区内容声称Git来源。父attempt绑定的`data/`输入或checkpoint工件若未进入Git，只有当前字节SHA逐字节等于parent SHA且parent attempt、receipt、bundle三者SHA完整一致时，才可分类为`parent_bound_external_artifact`，并在兼容性报告中绑定这三项父谱系；任何外部工件SHA变化、非`data/`代码文件无Git对象、或谱系缺项均fail closed。此分类只解决已精确绑定的数据工件没有Git对象的情况，不豁免源码哈希或语义差异审查。

| 文件 | parent SHA-256 | 当前 SHA-256 | parent 精确 Git 源 | arm |
|---|---|---|---|---|
| `scripts/data_prep/build_taiji_r2_daily_dialogue_full_turn_pretraining_corpus.py` | `d88b126ffd3caeccddfc6094d182403e052af8dbde80358871a6378034e0dfd8` | `0173946126a4fdd304b5c55f6aa73342f7455205236cdc2bd08a9544cfc3116b` | `02e82d921d14782fca511276020957220910a5c6` | all |
| `scripts/training/run_taiji_r2_daily_dialogue_full_turn_pretraining.py` | `4065f25ed47d3c823abc0cafeeb6f6d4a055986078f93fdd107cfa16b78aebef` | `69691de36b79841c56cc4e1783cfda7c8cd2b8137524e4ef5b1d48529b7013b9` | `02e82d921d14782fca511276020957220910a5c6` | all |
| `scripts/training/run_taiji_r2_daily_dialogue_recovery.py` | `e84e48215b72d6ad212c25d3dab63aca119654d7b28c94fde6fa2bc3d98c9e04` | `92205705d740e7949ff933366f5de964d97d40b4d0cfe9ba0447b098c15ff81c` | `85bfc59c8b60708525ff1e8f8c28f1d5ead1ba48` | all |
| `tests/taiji_native/test_r2_daily_dialogue_recovery_contract.py` | `fe889be52580947c1241aadcfcde117e0bc0cd604636d41026c5df6b07f5696f` | `5e93253af290fa65eece542f4de18a677b052a9ce4df53de3d88506543da5f3a` | `85bfc59c8b60708525ff1e8f8c28f1d5ead1ba48` | all |
| `tests/taiji_native/test_r2_dialogue_recovery_contract.py` | `4fd94e699a397b6f5b569aefae53b8f8bcb01eb953dfe029186b06718cbae40e` | `abc55fecbd3cf6b62bce86d17ef68e9a7eadbcd330c63daa43ea9626f78001de` | `0669123da231d19993ec5fe795187791cf0dd628` | all |
| `scripts/training/run_taiji_r2_daily_dialogue_warmstart_matched.py` | `266ea28b5b4af8381a7633eb7aaa2be8cc75d7b758e1ab620dc4a281b3328e44` | `2aac78a86a5509dee6baf66b80a993d297c67758ba7668a760be64c0a9e74a02` | `b6bb45b52702479ad20f91502bed1cb34766d324` | warm/fresh |
| `tests/taiji_native/test_r2_daily_dialogue_warmstart_matched_contract.py` | `b12861083031c31646df19ee3d52c5626b98d7c7ded9399ade28719e94d6d041` | `827db17d6b014c6167230f329771d46d9d156b399b7ad3a63b5f57e0a4cce1cf` | `b6bb45b52702479ad20f91502bed1cb34766d324` | warm/fresh |

审查了对应Git source diff：所列训练/数据构建与测试差异为格式换行/缩进、未使用`line_number`改为`_`、显式`strict=False`和等价语法/比较表达式改写；未见训练循环、超参、数据顺序或目标更新分母的实质变化。**这不表示parent SHA与当前SHA相同，也不构成自动豁免。** 新runner必须逐项把Git-tracked源码的parent/current SHA、Git blob/commit、source compatibility report SHA与新runner/helper/test SHA写入extension manifest；Git未跟踪的父绑定训练数据则仅在精确SHA匹配时用parent attempt/receipt/bundle SHA建立`parent_bound_external_artifact`谱系。任一来源无法按对应类别证明时都拒绝执行。不得直接绕过旧gate、只跳过报错文件或通过临时修改parent hash继续。

正式训练前需全部满足：

1. 实现 versioned runner、manifest/append-only ledger、parent checkpoint fork 与独立 result 路径；训练目标、epoch、分母、stage 顺序及 RNG/cursor 语义保持冻结。
2. 合成测试覆盖：parent bundle/receipt 复制逐字节校验与 fresh-process restore、旧目录/attempt/progress/result hash 不变、从父 cursor 延续的最终等价性、32 episode 保存边界、初始账本落盘后首个 commit 前强杀、checkpoint/receipt/latest 各写入故障、boot-change wall×24、source drift、单臂及 aggregate ceiling、并发争用和未知尾账 fail closed。
3. Ruff/静态门/相关测试全部通过，代码审查 P0/P1/P2 清零；当前评测器通过的 Phase 7 门不能替代 continuation runner 的独立代码审查。
4. 每臂核验 parent attempt SHA、latest receipt/bundle SHA、corpus/data/source binding、cursor与恢复回执；CPU/aggregate headroom 和资源门在开跑前重新采样，写入耐久启动 intent 后再执行。
5. 每臂单 writer，资源允许时可并发；受单一 aggregate guard 约束。任何来源、恢复、资源、预算或 durability 证据缺失均停止且不得换臂重试。

## 数据兼容与迁移

不做历史数据迁移，不改旧 attempt schema，不清理旧 checkpoint，也不回写旧 progress。新 sidecar 通过 parent attempt SHA 和 parent receipt SHA 连接旧证据，并用独立 extension schema 记录本窗口。未来 R2V2 evaluator 只能在新结果齐全、readiness 与 fresh-process restoration 证明完整后接受该 extension；不能只看旧 attempt 的 `training` 状态。

## 部署拓扑与非功能要求

使用本地固定 NTFS 和既有 `reports/`、`data/checkpoints/` 路径；新 manifest、aggregate ledger 与 checkpoint latest 采用已验证的同目录临时文件、flush/fsync、write-through 原子发布与 single-writer 锁。网络、ReFS、可移动盘及未经验证文件系统拒绝。整臂 CPU 用预算监督器计入父/子进程、检查点保存/恢复、失败尾账与中断后的 unknown tail；新鲜恢复子进程有有限 deadline 和强制终止/确认。资源门沿用可用内存至少 8 GiB、CPU 双采样均不高于 85% 的既有约束；并发时还须验证三臂组合资源与实测速率。

## ADR

| 决策 | 选择 | 原因 | 拒绝的做法 |
|---|---|---|---|
| 训练入口 | 新版本 extension runner + sidecar manifest | 预算与来源合同发生版本变化，必须有独立身份与审查 | 修改/monkeypatch 旧固定 cap runner 的常量和路径 |
| checkpoint 谱系 | 版本化 extension checkpoint root，以 parent receipt SHA 锚定 | 保留旧 latest 指针与全部旧 bundle/receipt，不混淆两次执行 | 在旧 root 覆盖 `latest.json` 或删除旧提交 |
| CPU 累计 | 含旧实耗与未知尾账的 monotonic cumulative ledger | 恢复后不重置；预算缺口可被保守尾账解释 | 将当前1.2M预算视为额外额度、重置CPU、用progress episode推算CPU |
| 来源漂移 | 父 hash 不变；extension 单独记录 source compatibility 和自身 hash | 可证明训练图连续，也保留历史来源身份 | 重写旧 hash、忽略 mismatch 或隐式 overlay |
| 能力与评测边界 | 训练结束后仍需 readiness、独立恢复与冻结 W5b 评测 | 工程继续执行不等于能力提升 | 把训练完成或保存恢复通过视为晋级证据 |

## 遗留风险

- 审查基线到实际执行之间尾账会随wall×24增长；当前预算上限固定为1.2M/arm，超过即没有余量，不能靠隐式追加解决。
- source hash mismatch 虽然当前 diff 未见训练语义变化，仍须由实现与独立审查记录精确兼容证据。
- 三臂终点尚无结果；本合同不能预测或承诺自然对话能力、保持性、泛化、模型上限或晋级结论。
- 旧 attempt 状态字段和历史 `recovery_stopped` 只描述旧窗口；不能清理、覆盖或被新结果误标成功。

## 给下游的唯一事项

实现并测试版本化 extension runner、parent checkpoint fork、耐久且单调的 CPU/aggregate ledger、source compatibility receipt 与 sidecar 结果；提交后由独立 code-reviewer 按本合同复核。只有所有实现、源码、恢复、资源和预算门均通过，才从每臂对应现存 checkpoint 继续训练。真实 W5b 仍等待三个 extension final、readiness 和 fresh-process restore 全部完成；0 dev/reference、0 selector、0 `--execute`，`can_promote=false`、N1-166继续冻结。

## 回流判定

- 结论：架构方向通过，允许进入实现阶段；不放行真实训练。
- 回流目标：无。
- 返工原因：无上游需求缺口；源码、实现、恢复与预算门列为下游必须完成的验收项。
