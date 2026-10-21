# Phase 4 补充架构合同：R2 recovery v2 四端点配对评测

日期：2026-10-21。状态：架构审查通过；本补充只为W5评测实现定界，不阻塞Phase 8三臂训练，不授权现在读取dev、运行模型或生成。

## 结论

原 `04-architecture.md` 的配对逻辑保留为旧R2-07评测历史，但不适用于当前v2结果接口：原合同依赖旧06/07 result schema、既存06两端 generation，并只为两条07臂新增78次生成。客户端更新后采用的新恢复runner输出 `taiji-r2-dialogue-recovery-result-v1`，且训练过程明确 `generation_queries=0`；恢复计划亦记录旧06未保存39面generation。因此当前必须使用独立版本化v2评测合同，默认在相同39条prompt上为4个端点各生成一次，共156次新调用，不能把78条旧输出当作已存在。

## 模块边界与端点

- 基线：旧R2-03R1只读parent checkpoint `data/checkpoints/r2_daily_dialogue_oasst2_20261021_continuation/dialogue_context_final.pt`。绑定冻结文件SHA `41655eb8b5fcd7e446acb924e94e458ce9b1e966a96a95517198a5ac38df2086` 与payload `checkpoint_digest=1a77364a42633686dc9e1e9ee6a13a7e09b4f07770e3841d873bacd9eadaeaeb`；按原baseline配置构造推理trainer，不进行训练。
- 新端点一：`full_turn_1m` 的 `assistant_sft` 最终checkpoint。
- 新端点二：`warm_start` 的 `assistant_sft` 最终checkpoint。
- 新端点三：`fresh_same_capacity` 的 `assistant_sft` 最终checkpoint。
- 三个v2 checkpoint必须来自各自结果 `reports/taiji_r2_daily_dialogue_recovery_v2_20261021_{arm}_result.json` 的 `stages.assistant_sft.path`；中间 `pretraining_final.pt` 仅用于血缘及更新量核验，不作为对话端点。恢复根在 `data/checkpoints/r2_daily_dialogue_recovery_v2_20261021/{arm}/`。
- 当前固定面每个family只选一个prompt/history；四端点共享完全相同的39行输入，所以完整面为39 families、156 endpoint generations。旧06未完成合法39面generation；只有发现带完整source与checkpoint证明、且family/prompt SHA及端点checkpoint digest完全相同的已存输出时才可逐端点复用。任何身份无法证明的历史输出均令preflight拒绝并要求新的架构裁决，不得靠endpoint名称或相同prompt猜测等价。

## 元数据合同与拒绝条件

每臂result必须满足：schema=`taiji-r2-dialogue-recovery-result-v1`、`status=complete`、`complete=true`、arm与路径相符、`restart_from_zero=true`；`bindings.run`固定为 `r2_daily_dialogue_recovery_v2_20261021`，其arm、zero SHA、source hash map与同臂readiness逐字一致。每条绑定源路径均须存在且当前SHA匹配，结果文件SHA及stage checkpoint路径/字节SHA/restore digest也须逐项核验。不能只信attempt里的 `status=training` 或progress文本；v2 runner的最终result和两阶段receipt才是终点事实。

三个结果都须同时满足：

| 阶段 | 固定分母 | 固定字节更新 | final检查点 |
|---|---:|---:|---|
| `pretraining` | 30,096 episodes | 14,517,912 | 存在且独立恢复回执与文件SHA一致 |
| `assistant_sft` | 525 episodes | 113,209 | 存在且独立恢复回执与文件SHA一致 |

`result.dev_queries=0`、`generation_queries=0`、`can_promote=false`必须成立。新评测只接受三个result、三个readiness、基线checkpoint和所有source binding均齐全的状态；任一缺失、partial、计数/全局步不同、路径越出固定root、SHA/digest漂移、预算超限、已有同名评测attempt/result/progress或资源门失败，都须在第一次读取保留prompt前停止。不能修改训练runner或恢复合同来弥补评测输入不合格。

评测自身要有freeze manifest，绑定新架构补充、评测runner、契约测试、prompt选择/生成依赖及固定训练计划。其结果还应记录三个v2结果/readiness SHA、基线checkpoint SHA、四个终点checkpoint SHA、生成前后checkpoint文件SHA和模型摘要；任一生成期间变化即拒绝。

## Prompt、生成与持久化顺序

执行顺序固定为：只读metadata/source preflight → 核验三臂final receipts与基线 → 四端点fresh-process restore → 创建write-once attempt → 用冻结的 `_select_dev_prompts()` 规则选39个family并写入规范化prompt face/SHA → 对每个未被精确证明已消费的endpoint逐条生成并持久化 → 汇总及人工审阅。该selector只将 `episode_id/family_id/split/system/context/history/user_input/unknown_policy` 映射为 `PromptOnlyEpisode`；评测结果不得包含reference answer、assistant target或exact-response评分。

每条generation先持久化 `query_started(index, arm, prompt_sha256)` 并flush/fsync，再开始该次调用；生成结果返回后立即持久化输出并fsync。每端点最多39次；当前无历史合法输出分支下 `query_attempts=generation_outputs=persisted_generation_outputs=156`。发生异常、强杀、部分写盘或恢复失败即保留attempt和已落盘行，消费本次attempt并停止；任何情况下不自动重试、不跳过family、不从最后一条继续。使用与既有全轮脚本一致的 `_generate_readonly` 语义：`constrained_decode=true`、sampling关闭、`max_generation_bytes=512`，生成前后恢复/比对模型摘要且正式/优化器更新均为0。

建议新增入口 `scripts/training/evaluate_taiji_r2_daily_dialogue_recovery_v2.py` 和独立synthetic契约 `tests/taiji_native/test_r2_daily_dialogue_recovery_v2_evaluation_contract.py`；只写 `reports/taiji_r2_daily_dialogue_recovery_v2_20261021_paired_eval_*`。`--validate-only`只能审元数据与固定文件SHA，不选择prompt、不加载模型、不生成。真实dev读取只能由独立 `--execute` 在所有preflight和fresh-process restores完成后发生。

## 预算与结果

- v2恢复计划的评测增量上限是 **14,400 process-CPU秒**，覆盖source/face校验、全部四端点恢复、生成、落盘、失败调用与汇总；不得沿用旧04文档的7,200 CPU秒或旧“78新/78复用”假设，也不得把评测CPU二次计入训练三臂硬预算。该增量计入恢复计划总新增上限503,400秒。
- 保留既有单线程、启动前两次CPU采样均≤85%、可用内存≥8 GiB的资源门；任何一项不满足就在读取prompt前拒绝。旧架构的14,400秒wall上限继续作为更严格的墙钟停止门；先到任一限额都停止并留证。
- 输出schema必须逐项记录四endpoint identity与checkpoint SHA、输入face及逐prompt SHA、result/readiness hashes、复用与新调用计数、156组合输出、child/parent/total CPU、资源采样、错误和持久化计数；显式写 `parameter_updates=0`、`optimizer_updates=0`、`reference_answer_scoring_queries=0`、`exact_match_scoring_queries=0`、`can_promote=false`。
- 人工审阅固定同一39面前12个family，四端并列判断中文可读性、贴题、历史连贯和重复。表面格式/停止指标只作描述统计；本小样本对照不等于独立确认、自然语言能力晋级或N1-166准入。

## 风险与下游

当前v2训练结果尚未齐备，评测实现与synthetic测试可并行开发；不得读真实dev正文、运行模型或产生generation，直到所有三臂终点与恢复门完整。训练中不得修改任何source-bound plan、runner、数据、checkpoint或结果。测试使用合成39行prompt和虚拟result/receipt，必须证明缺失终点会在prompt selector前拒绝、四端点映射和156分母正确、失败查询不重试、source drift/重复family/错prompt SHA/磁盘不足/CPU与内存超限均fail closed，且checkpoint SHA不变。

## 回流判定

- 结论：通过；不需要上游返工，也不阻塞Phase 8训练。
- 回流目标：无。
- 给下游：按此v2接口实现独立评测runner和synthetic契约测试；先代码审查和定向验证，所有v2训练终点齐备后才可开启一次真实39面评测。
