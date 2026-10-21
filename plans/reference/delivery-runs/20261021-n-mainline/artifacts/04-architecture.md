# Phase 4 架构合同：R2-07 独立配对评测 v1

## 结论
通过架构门禁。实现独立、只读 checkpoint 的配对评测入口；当前真实运行严格阻断。主理人新核实沙箱外 Get-Process/CIM 无 python，progress 停在06 episode14880、07 warm9984/fresh9856，且无 partial/final；旧 attempt=training 不构成存活证明。本合同不授权补跑、中断恢复或读取真实 dev 正文。

## 架构说明与模块边界
- 入口为 scripts/training/evaluate_taiji_r2_daily_dialogue_warmstart_matched.py；不得修改两个训练 runner、冻结计划或旧报告。
- 元数据预检只读三个 result、source hashes、checkpoint 字节 SHA、恢复回执；完成后才允许读06 generations。不得调用 `_select_dev_prompts`，不得扫描 dev JSONL 或加载 reference。
- prompt 直接复用06已保存 records 的 history/current_user_turn；旧两端输出原样合并。模型恢复只使用 train corpus 与07 assistant final full bundle。生成调用06 `_generate_readonly` 同一逻辑，前后核验内存模型摘要及磁盘文件 SHA。
- 阶段依次为 metadata-preflight → fresh-process restore → generation-face validation → one-shot generation → aggregation。任一失败停止，不允许自动重试或跳过 family。

## 接口契约表
|接口/字段|类型与来源|输出/拒绝条件|
|---|---|---|
|CLI --validate-only|无生成的元数据验证；默认不访问06 generation正文|readiness；缺失result立即拒绝，0 dev/模型查询|
|CLI 默认执行|固定三结果、06 generation、07两臂final路径；禁止任意checkpoint覆盖参数|先创建排他attempt；已有attempt/result/progress拒绝|
|06 result|schema=taiji-r2-daily-dialogue-full-turn-pretraining-result-v1，status=complete，complete=true|full_turn_pretraining.episode_count=30096、byte_updates=14517912；assistant_calibration.episode_count=525、byte_updates=113209；total_target_byte_updates=14631121|
|07 result×2|schema=taiji-r2-warmstart-matched-result-v1，arm=warm_start/fresh_same_capacity，status=complete，complete=true|pretraining.episodes=30096、target_byte_updates=14517912；assistant_calibration.episodes=525、target_byte_updates=113209；各global_step须等于其阶段字节更新量；model_parameters=278010|
|来源核验|三个result的plan_sha256、source_hashes，以及result/runner/test/本合同/评测器SHA|逐个验证当前文件 SHA，不忽略缺项；readiness与result绑定一致；checkpoint路径须位于各冻结root|
|06恢复证据|fresh_process_restores.full_turn_final/assistant_sft_final；阶段final_checkpoint|path、sha256、checkpoint_digest与文件及回执一致；其他已声明恢复记录也逐项验证|
|07恢复证据|checkpoint_restores.full_turn_final/assistant_zero/assistant_final；pretraining.checkpoint、assistant_calibration.checkpoint|path/sha256及restore.checkpoint_digest逐项一致；再对子进程恢复的assistant final独立核验|
|新进程restore|沿用06 restore-worker assistant_sft语义，torch.load(weights_only=True)、LanguageAlignmentTrainer.from_checkpoint|receipt须含checkpoint_digest、cpu_seconds及完整恢复成功字段；不执行train/信用探针|
|06 generation report|schema=taiji-r2-daily-dialogue-full-turn-generation-v1，status=complete，SHA等于06 result.generation_report_sha256|input_families=input_episode_count=39，generation_outputs=78，两个冻结endpoint各39；constrained_decode=true、sampling=false、max_generation_bytes=512；reference_answer_included=false、exact_response_scored=false|
|prompt identity|records原序index、episode_id、family_id、history_turn_count、history:list[list[str]]、current_user_turn:str|恰39个唯一family及episode，index=1..39，history计数正确；只允许所列输入字段；对规范JSON输入逐行SHA及总face SHA，输出前后不变|
|错误|PREFLIGHT/PROVENANCE/RESTORE/FACE/RESOURCE/GENERATION/PERSISTENCE|固定失败status、错误信息、最后query索引、已调用/已完成/已持久化计数；保留partial证据|

## 数据模型与迁移
所有新增报告使用 reports/taiji_r2_daily_dialogue_warmstart_matched_20261021_paired_eval_* 前缀，readiness.json、attempt.json、progress.jsonl、generations.json、result.json、review.md；write-once创建，逐行flush+fsync。attempt是执行消费凭据，失败后仍保留。

结果schema为 taiji-r2-warmstart-matched-paired-evaluation-v1。字段至少包括 status、complete、source_hashes、input_result_sha256、generation_face_sha256、prompt_hashes、checkpoint_sha256_before/after、fresh_process_restores、model_digest_before/after、query_attempts、generation_outputs、persisted_generation_outputs、reused_generation_outputs=78、combined_generation_outputs=156、input_families=39、parameter_updates=0、optimizer_updates=0、reference_answer_scoring_queries=0、exact_match_scoring_queries=0、can_promote=false、cpu_seconds_parent/children/total、per_arm_cpu_seconds、resource_samples、errors。

每行保留06原始输入字段及 previous_8epoch_candidate、full_turn_pretrained_candidate 两输出，新增 warm_start、fresh_same_capacity；输出结构复用06 `_generate_readonly` 的 generated_text/raw_generated_sha256/generated_bytes_hex/generated_bytes_length/utf8_valid/no_replacement/response_boundary_present/stop_reason/pii_pattern_redacted/repeated_4gram_rate。合并记录不得再生成旧端点。

progress先写 query_started(index,arm,prompt_sha256)，fsync后才调用生成；每次调用前递增query_attempts，调用返回立即持久化输出，才递增persisted。上限78次；错误或写盘失败即停止并消费attempt，禁止补齐重跑。不得从末尾继续或把异常调用计为零。成功须query_attempts=outputs=persisted=78。review固定06 records[:12]同一family，每题四端并列；人工可读性/贴题/连贯/重复的判断另列，不能用格式汇总替代。

不迁移或覆盖既有数据。未来恢复训练产生不同来源结果时，须先追加版本化执行合同与新路径，明确源血缘，再修订本评测输入绑定；不能悄悄把旧路径替换成新训练。

## 部署拓扑与非功能指标
CPU、单线程、两臂顺序恢复与生成，最多一个评测工作进程及一个短期恢复子进程；OMP/MKL/OPENBLAS与torch均1线程。启动及每条生成前内存≥8 GiB；CPU连续两次1秒采样>85%则拒绝新查询。输出预留≥512 MiB；每次调用前检查预算且为最终失败报告留CPU余量。

新增评测CPU硬上限7200秒（包括预检、所有恢复子进程、失败调用和持久化），wall上限14400秒；分别记每臂开销，共享开销等分计入两臂。更严格的有效预算=min(7200, 120000-warm训练CPU + 120000-fresh训练CPU)，且每臂训练加评测归属CPU≤120000；06原CPU≤240000、关联总CPU≤480000。任一剩余为负/未知即拒绝；不自动扩大原预算。7200为本合同新增执行上限，落地前由主理人同步增量计划并绑定版本，不修改冻结训练计划。计量采用process_time加所有子进程receipt CPU，禁止只算wall。生成期间始终只读checkpoint，不写训练产物。

## ADR
选择独立评测器与元数据先行：避免改写运行中源码及冻结源绑定，已有06结果保留完整血缘。拒绝重新从dev选择prompt，因为可能引入新输入或加载reference；拒绝复跑06两端点，因为破坏一次性查询与同面保存证据；拒绝以attempt=training判定存活。当前缺失终点是执行阻断，不是架构需求缺口，故无需上游返工。

## 遗留风险
- 三run疑似中断且无终点，当前不能运行真实评测；恢复训练方案属于另一个版本化合同。
- 06 generation尚未存在时无法预先确认真实face结构，测试必须采用合成39行与虚拟回执，不读取真实dev正文。
- 私有生成/恢复函数可能发生漂移；必须绑定源码SHA并契约测试真实接口字段，不能靠宽松get默认值放行。
- 远端CI日志请求403 rate limit，现阶段不能归因CI失败；实现后本地定向验证并报告远端不确定性。
- 资源预算可能无余量；拒绝执行并留证，不默认扩充。小模型同面描述性比较不能证明语言能力或晋升。

## 给下游的注意事项
实现新增评测器及独立契约测试，覆盖缺终点在prompt访问前拒绝、不同schema字段、重复family/错prompt SHA/源漂移拒绝、78新调用与156组合输出、异常调用计数与持久化失败不重试、预算/子进程CPU归属、checkpoint前后SHA不变。不得启动模型、训练、真实生成；不改当前冻结文件。主理人另行处理训练中断取证与恢复合同。

## 回流判定
- 结论：通过
- 回流目标：无
- 返工原因：无
