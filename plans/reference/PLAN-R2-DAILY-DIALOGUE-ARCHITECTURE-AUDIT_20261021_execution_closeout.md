# PLAN-R2-DIALOGUE-04执行结案：自然对话生成通路审计

**状态：完成。** 只读审计CPU低于120秒；没有训练、参数更新、生成、dev/holdout访问或checkpoint写入。

## 关键发现

1. **当前“base”不是预训练语言模型。** `validate_taiji_r2_h3_7b_dev.py` 在正常训练分支中用`TaijiConfig.capacity_profile(300000, seed=20260917, ...)`新建模型，没有加载外部语言模型权重；随后只在12个小型任务episode上训练2个epoch。父checkpoint留下24条history、1,074个响应字节更新。其原dev报告exact-response率与end-marker率均为0。对应语料`tests/fixtures/r2_h3_5a_response_plan_v3.jsonl`仍在本地，SHA-256为`dafd4990317a702bee2d970f12afe8588abb3ba4023b3c7424ae983bd37755c2`。
2. **H3.7B不是37亿参数的模型名。** 这里的`h3_7b_byte_aligned_chunks`是H3.7 byte-aligned响应目标几何的标识。按当前checkpoint loader重建，模型有278,010个active scalar参数，dense-equivalent为436,941；旧报告写277,970，差40，因历史代码revision `07113c3d6fb344a00c1cd7f7c36588ee8254f22c`不在本地Git对象库，暂时无法解释。审计保留两种读数，不把两者混为一谈。
3. **生成和训练都在原始字节面上运行。** 输入标记、中文和回复都编码成UTF-8字节，输出面为256个byte值加1个boundary symbol。训练时prompt按`learn=False`送入，assistant回复与字面`<|end|>`逐字节按`learn=True`送入；代码没有标准cross-entropy/`loss.backward()`/optimizer接口。生成使用确定性greedy argmax、UTF-8合法字节约束、512字节上限，并在字节流出现字面`<|end|>`时停止。
4. **本轮自然材料仍是小规模回复微调。** OASST2原始包有135,174条消息、8,615条中文；严格过滤后接受5,250条中文消息，但builder只选710个回复episode，分为525 train、62 dev、123 final，覆盖461个tree family。它为每个prompter选一个最佳assistant回复。当前R2训练只用了525个train episode、共340个family；39个dev family未查询。原82个final family已在前两批生成检查中全部消费。
5. **终点读数仍不支持自然交流能力。** PLAN-R2-DIALOGUE-03R1在70个family×6端点保存420个自由生成输出；8-epoch多轮候选end-marker率90%、UTF-8有效率100%、重复4-gram率8.51%。但预定人工检查的12个样例中，连贯且贴题的自然回复为0/12。格式和重复读数不能替代语义能力。

## 判读

最有力的已证事实是：一个不足30万active scalar、从Taiji配置新建、只在12个任务episode上短训的模型，被要求在525个自然中文回复样本上获得日常对话能力。它不是在大规模通用语言预训练模型上做对话微调。这个训练起点与材料规模很可能是输出乱码的主要限制；本审计没有证明单一因果机制，也不把问题归结为“RNN固有问题”。

本轮紧接着完成的[PLAN-R2-DIALOGUE-05R1材料清点](PLAN-R2-DAILY-DIALOGUE-MATERIAL-INVENTORY-RECOVERY1_execution_closeout_20261021.md)确认：训练family内有3,762条安全过滤后的user/assistant消息、1,780,881个UTF-8字节；39个dev family未做模型查询，旧final82 family已全部消费。唯一下一步转为[PLAN-R2-DIALOGUE-06](PLAN-R2-DAILY-DIALOGUE-FULL-TURN-PRETRAINING_20261021.md)：扩展到角色感知全轮对话预训练，再做assistant回复校准；使用未触碰dev family作一次性自由生成审阅，不再追加旧的525回复小样本配方。

## 绑定产物

- 冻结审计方案：[PLAN-R2-DIALOGUE-04](PLAN-R2-DAILY-DIALOGUE-ARCHITECTURE-AUDIT_20261021.md)
- 机器可读读数：[architecture audit JSON](../../reports/taiji_r2_daily_dialogue_architecture_audit_20261021.json)
- 实际参数重建源：[Taiji model](../../taiji/model.py) · [language alignment trainer](../../taiji/language_alignment.py)
- 父模型生成源：[H3.7B validation runner](../../scripts/training/validate_taiji_r2_h3_7b_dev.py)
- 自然材料构造源：[OASST2 dialogue builder](../../scripts/data_prep/build_taiji_r2_daily_dialogue_corpus.py)
- 生成质量结案：[PLAN-R2-DIALOGUE-03R1](PLAN-R2-DAILY-DIALOGUE-CONTINUATION-RECOVERY1_execution_closeout_20261021.md)
