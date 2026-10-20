# PLAN-R2-DAILY-DIALOGUE-01：多轮中文对话续训与自由生成检查

**状态：已按用户授权冻结；本批完成后继续以自然对话生成作为优先方向。**  
**日期：2026-10-21**  
**主线调整：** N1-165A 已结案，但原训练充分性门仅2/3通过，`can_promote=false`；未执行、未消费面[N1-166](PLAN-N1-166_compositional_transfer_confirmation_20261020.md)延后。N1-165A结案见[中文结案记录](PLAN-N1-165A_execution_closeout_20261021.md)。

## 要回答的问题

把真实中文多轮对话材料用于态极原生语言模型的续训后，模型能否在未训练的整棵自然对话树上，依据可见历史生成连贯、可结束、不过度重复的下一句？本批评价自由生成；不以训练集逐字正确率或合成问答题得分代表对话能力。

本批只主张“在OASST2中文对话树留出面上的局部续训效果”。不主张开放域知识、一般智能、产品可用或能力晋级，也不改默认模型和用户入口。

## 语料来源与准入

- 使用[OpenAssistant OASST2](https://huggingface.co/datasets/OpenAssistant/oasst2)，固定Hub revision `179dd21fc55192153d94adb0e0ce8f69e222bf75`。数据卡标注Apache-2.0；原始论文记载数据由志愿者生成并经人工标注，提供对话树结构。数据源与许可依据：[数据卡](https://huggingface.co/datasets/OpenAssistant/oasst2)、[原始论文](https://arxiv.org/abs/2304.07327)。
- 固定源文件：`data/conversation/OASST2/2023-11-05_oasst2_ready.messages.jsonl.gz`，SHA-256 `a9f240c4c77aa1378364f70d37e753c07ba284e247b019d700e1947a0e5da751`。原始文件和派生语料保留在本地忽略目录，不提交原始对话内容。
- 初步只读统计：全文件135,174条消息，中文8,615条。逐行筛选要求中文、`ready_for_export`、非删除、审查通过、非合成；移除带PII、垃圾、语言误标、不当、仇恨或性内容标记的消息，并用邮件/电话正则再筛一次。保留标记为暴力但通过其余审核的内容，因为该标签也会覆盖叙事或新闻讨论；该候选不部署，仅用于本地研究。
- 对每个中文用户消息，最多选一条回复：只考虑rank 0、rank 1或无rank回复；存在多个时选rank最低者，再以message id稳定打破平局。当前用户消息不超过1,024字节，目标回复不超过480字节，最多保留3轮完整历史，最终prefix不超过4,096字节。全流程不保存user_id；含邮箱/电话模式的消息排除。
- 整棵`message_tree_id`用固定盐`oasst2-daily-tree-split-v1`作稳定hash划分：70% train、10% dev、20% final；同树绝不跨split。预检所得710个episode分为525/62/123，分别来自340/39/82棵树。builder与manifest运行时再次核对这些数量和SHA；任何变化均停训并修订数据合同。

## 固定模型与对照

- 从当前可恢复的Taiji原生H3.7B byte-aligned parent开始：[parent checkpoint](../../reports/r2_h3_7b_validation/dev_byte_aligned/checkpoint.pt)，SHA-256 `88de777fe05cdafb0a164e7fe7c21dc4417edc7fd3ce325f91885fd547708287`；关联报告SHA-256 `366f06d74ded0560bccf571ebf4177d07901e75b3ee8bc64cb50c478202ada4f`。
- 该parent此前2轮/24个人工问答训练，已通过保存恢复工程检查，但其旧dev自由生成响应边界率为0；逐字匹配率也为0。这些读数只说明需要在真实对话材料上重训生成，不能替代本批新语料评价。
- 保持同一parent、H3.7B byte-aligned响应读出、推理解码、目标回复及总更新量。只比较上下文处理方式：
  - **多轮对话候选**：输入当前用户句和最多3轮干净历史。
  - **匹配单轮对照**：使用完全相同的OASST2用户/助手目标，只去掉先前历史，保留当前用户句。
- 两臂各从同一parent独立恢复，按固定episode id顺序遍历完整train split，训练2个完整epoch；不做随机子采样或中途选点。总计2个续训终点、4个fresh full-bundle保存恢复（zero/final各2）。这是一组配对续训，不冒充多seed稳定性结论。

## 保存、预算与停止

- 两个臂的正式zero bundle就是checkpoint前置门：以完整512字节实际配置各原子保存一次，由fresh process恢复并核对checkpoint digest、父模型/RNG/响应计划状态和corpus digest；该门为0参数更新、0生成。只有两个zero bundle均通过才启动训练，不另做重复预检。
- 每臂训练2个完整epoch，不中途按train/dev读数选点、不改变epoch数、不追加训练。两个final bundle均fresh-process恢复，核对父checkpoint未变、目标编码器只绑定train split、模型/训练器checkpoint digest完整。
- 全批本机CPU上限120,000秒，包含来源/语料核验、零步checkpoint往返预检、两个完整训练、4次独立bundle恢复及最终自由生成；达到上限立即保存已完成证据并停，不缩短语料或重试。
- 任一source/parent/checkpoint SHA、tree split、保存恢复或累计CPU合同不匹配时停止并保留失败回执。保留所有旧计划与失败结果，不生成新题、不查询旧dev/holdout、不晋级。

## 自由生成检查

- 从final split按固定SHA排序抽取12个不同tree：6个当前句无历史、6个带历史；选择规则在builder中冻结。parent分别生成有/无历史版本，多轮候选只喂历史，单轮对照只喂当前句，总计`12 × 4 = 48`次生成。其余111个final episode本批保持未查询。
- 生成器只接收当轮用户输入和允许的历史；不读目标回复。最终报告不记录reference answer、不计算exact response或teacher-forced准确率。dev split不查询。
- 逐条保存12个留出prompt/history、输出UTF-8字节、输出长度、end marker停止情况；汇总UTF-8有效率、边界率、重复输出率和重复4-gram率。另附父档与两臂的并列生成样本供人直接检查；这12个树只构成定性首轮观察，自动指标不替代人工对连贯、贴题、自然度和重复的判断，也不作为统计性泛化门。
- 即使所有工程门通过，`can_promote=false`。只有heldout自由生成相对parent有稳定可见改善且多轮候选超过匹配单轮对照，才建议后续扩展训练；否则转数据/解码/信用问题归因，不自动重复训练。

## 产物路径

- 数据builder：`scripts/data_prep/build_taiji_r2_daily_dialogue_corpus.py`
- 训练/生成runner：`scripts/training/run_taiji_r2_daily_dialogue_material_comparison.py`
- 本地派生episodes：`data/conversation/OASST2/taiji_r2_daily_dialogue_episodes.jsonl`
- 本地派生manifest：`data/conversation/OASST2/taiji_r2_daily_dialogue_manifest.json`
- 中文执行结案：`plans/reference/PLAN-R2-DAILY-DIALOGUE-01_execution_closeout_20261021.md`
- readiness、attempt、progress、result与生成样本：`reports/taiji_r2_daily_dialogue_20261021_*`
- 候选full bundles：`data/checkpoints/r2_daily_dialogue_oasst2_20261021/`
