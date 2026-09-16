# M5 R2 aligned language pilot 预注册

> 更新：2026-09-16。本文把[R2语言目标与信用分配设计](M5_R2_LANGUAGE_TARGET_CREDIT_DESIGN_20260916.md)落实为首个可运行的原生训练入口。
> 状态：本文的v1预注册合同保持不变；结构化episode、保存恢复preflight、plumbing smoke、zero/child paired诊断、static/slow/fast/fast_slow四臂对照和P2序列级只读评价已完成。后续G1/H3.1/H3.2/H3.3/H3.4均作为独立版本化诊断，不回填为本文v1的正式pilot结果；当前尚未形成能力晋级或Mini验收结论。实际执行附录见文末。

## 1. 目的和范围

本pilot只回答一个主线问题：

> 在Taiji原生predictive readout和同一模型状态不变的前提下，把训练单位从无边界字节流改成显式prompt/context/response episode，是否能让response段获得可审计的学习信用？

本pilot不是：

- Mini模型用户验收；
- CAP-0重测；
- 外部模型接入；
- P5.2d在线回写；
- P3b结果的延长或重跑；
- 对开放域语言能力的声明。

## 2. 版本和模型所有权

| 项目 | 冻结内容 |
|---|---|
| 训练格式 | taiji-native-language-alignment-v1 |
| episode序列化 | r2-episode-markers-v1 |
| 原生所有者 | Taiji predictive readout、predictive context和native fabric |
| 外部provider | 禁止；N模式只允许Taiji原生生成 |
| 默认入口 | 不修改seed默认chat入口 |
| P3b父checkpoint | 不可写；pilot必须使用新输出路径并记录父摘要 |
| 数据集 | 结构化JSONL；每行一个显式episode |
| 评价对象 | response teacher-forced accuracy、response surprise、原生生成文本、UTF-8合法性和checkpoint只读性 |
| P2序列评价 | response boundary、generation stop reason、required/forbidden terms、unknown markers/policy、sequence criterion和exact response；规则评分只作显式代理 |

## 3. 数据合同

每行必须包含：

- episode_id：稳定唯一ID；
- family_id：用于防止同一家族跨train/dev/final泄漏；
- task_family：任务族；
- split：train、dev、final或retention；
- system：行为边界；
- context：可引用材料，可以为空；
- history：用户和助手历史对；
- user_input：当前问题；
- response：目标回答；
- unknown_policy：answer、say_unknown、clarify或refuse。
- required_terms / forbidden_terms：可审计的必需内容和禁止内容；为空表示该项不适用。
- unknown_markers：`say_unknown`的允许表达，与required_terms分开记录。

语料构建器拒绝：

1. 缺少response/target；
2. 非法或重复episode_id；
3. 同一个family_id跨split；
4. 控制标记出现在用户数据中；
5. 没有train、dev或final任一主split。

固定序列化为：

系统标记 → context标记 → history标记 → user标记 → assistant标记 → response → end标记。

prefix只建立条件状态，target只包含response和end标记。所有原始JSONL、文件hash、episode摘要和corpus digest必须随运行报告保存。

## 4. 训练算法

每个episode执行以下步骤：

1. 从当前模型开始，重置动态状态并写入原生boundary；
2. 逐字节消费prefix，learn=false，禁止prefix本身产生持久学习更新；
3. 逐字节消费response target，readout=predictive，learn=true；
4. 只在target字节上记录prior prediction、accuracy和surprise；
5. 一个episode完成后再进入下一个episode，不跨episode泄漏动态状态。

第一版默认：

- response_repeats=1；
- use_memory=false；
- learn_fabric=true；
- learn_predictive_context=true；
- learn_predictive_readout=true；
- max_generation_bytes=512；
- epoch和episode数由命令行明确给出，不默认无限训练。

这个算法仍然使用Taiji的局部原生学习，而不是引入autograd、Transformer或外部decoder；变化点是学习信用的边界从整段字节流收紧为回答目标段。后续若要加入基础语言动力学、序列级评价或多时间尺度适应，必须新建版本，不在本预注册里悄悄改变目标。

## 5. 训练前硬门

正式pilot前，scripts/training/train_taiji_r2_aligned.py 必须自动执行：

1. 零步checkpoint原子保存；
2. 全新进程加载零步checkpoint；
3. 固定episode的只读生成和checkpoint摘要相等；
4. 在零步子checkpoint上执行一次response更新；
5. 原子保存child checkpoint并在全新进程恢复；
6. child digest变化且parent digest保持；
7. 恢复后只读评分结果可重复；
8. 调用方的parent模型未被preflight改变。

任一硬门失败，不得开始正式pilot。所有preflight输出必须进入与正式checkpoint隔离的preflight目录。

## 6. 运行路径和产物

唯一入口：

scripts/training/train_taiji_r2_aligned.py

运行时至少生成：

- parent checkpoint摘要；
- preflight/zero.pt；
- preflight/child.pt；
- R2训练checkpoint；
- R2训练报告；
- corpus manifest和digest；
- 每个episode的response训练记录；
- dev和final的原生生成记录。
- 每个episode的raw bytes hex、生成长度、边界是否出现、停止原因、sequence validity和语义代理判分。

输出目录不得是以下保护文件：

- checkpoints/seed_beta.pt；
- checkpoints/seed_corpus.pt；
- checkpoints/resumed_seed_corpus.pt；
- P3b的任何arm checkpoint；
- 产品默认入口checkpoint。

## 7. 正式pilot预算和停止规则

先执行plumbing smoke，不把smoke当作能力证据：

| 层 | 规模 | 用途 | 能宣布什么 |
|---|---:|---|---|
| preflight | 1个train episode、1次更新 | 检查保存恢复与信用边界 | 只能宣布链路可运行 |
| smoke | 不超过4个train episode、1 epoch | 检查训练入口、报告和dev/final读取 | 只能宣布实现没有断裂 |
| pilot | 由独立数据manifest冻结；默认不超过256个train episode、1 epoch | 首次目标对齐因果试验 | 可判定目标信号是否存在，不自动晋级 |

停止条件：

1. checkpoint保存或恢复任一失败，立即停止；
2. prefix被写入学习或episode之间发生状态泄漏，立即停止；
3. 输出来自外部provider、回退模板或隐式随机初始化，立即停止；
4. response proxy改善但dev/final原生生成不改善，停止同质扩大，进入目标/读出复审；
5. 只有UTF-8合法性改善而条件回答不改善，归为S1改善，不进入L2；
6. 数据hash、split或family血缘漂移，停止并重新冻结。

## 8. 结果解释

报告必须同时给出：

- response teacher-forced accuracy；
- response mean surprise；
- dev/final生成原文；
- UTF-8合法率和替换字符率；
- exact response rate，仅作受控pilot指标；
- native_mode=true；
- external_provider=false；
- checkpoint_read_only=true；
- parent/child/corpus digest。

以下不构成通过：

- 训练脚本运行完成；
- loss或surprise下降；
- 生成一段非空文本；
- 可读率上升；
- UTF-8合法、边界出现或规则代理分上升，但回答内容没有在未见问题上满足条件；
- 单个训练episode精确复现。

Mini/L2/L3评价继续由07负责，等R2主线bundle形成后统一执行；本pilot报告不得写入Mini验收通过。

## 9. 原预注册状态和执行顺序（历史合同）

已完成：

- 结构化episode对象和split/family隔离；
- 原生response-only学习入口；
- 原生生成与只读评分；
- checkpoint digest、atomic save和恢复；
- 直接preflight：passed；
- UTF-8受限原生读出及overlong/代理范围边界修正；
- 四episode plumbing smoke：训练、报告、dev/final记录完成；
- zero/child paired诊断：局部surprise下降、原生输出对输入改写敏感，但exact response未改善；
- static/slow/fast/fast_slow developmental F1对照：四臂preflight与恢复均通过，但没有exact response收益；
- P2序列级只读评价：required/forbidden/unknown规则分离，raw bytes和stop reason进入报告；dev/final边界、内容和unknown policy仍未通过；
- 代码级测试文件已加入，受限环境pytest临时目录权限问题需单独记录；py_compile和不依赖临时目录的约束测试通过。

原预注册的下一步曾是升级G1条件回答接口/读出路由：把task family和unknown policy显式放入prefix，版本化序列化合同，记录不同episode是否被同一个已见答案吞并，并在P2只读评价下核对条件变化和停止原因；不直接扩到256 episode正式pilot，不把规则评分写入fast/slow。该顺序已经由独立版本化的G1/H3.1/H3.2诊断接替；Mini验收和P5.2d修正继续后置。

## 10. 实际执行附录（2026-09-16）

本附录是对预注册之后实际发生的独立诊断的事实记录，不修改上文v1的前瞻合同，也不把诊断结果包装成正式pilot或能力通过。

### 10.1 H3.1 原生序列路径对照

在同一v2 native predictive readout owner上保留constrained greedy，并增加bounded beam候选路径。beam确实改变了候选输出，但只是交换两个train episode的正确/错误归属，collision、exact和sequence的聚合结果没有改善；恢复重复性和checkpoint只读守卫成立。因此搜索解码被判定为诊断工具，不作为能力修复或用户交付路径。

### 10.2 H3.2 response-start 候选读出

新增独立的`predictive_readout.response_start` candidate，仅负责显式assistant boundary后的首字节，后续字节仍由protected `predictive_readout` 产生。候选具有独立学习、checkpoint、恢复和margin诊断边界，不使用`task_family` oracle、不接外部provider、不修改默认入口。

执行血缘：

- dataset：`tests/fixtures/r2_language_alignment_smoke.jsonl`，SHA-256为`DBB85B49AFC724644834365E6B1700AFBD3E7230E4985FCD68A38C45FC5DD3A7`；corpus digest为`ed2a19a20475c0faebc0bbb862486a0e42640990f0368233005baba6f1553d52`；
- runner：`scripts/training/train_taiji_r2_aligned.py`，20 epoch，2个train episode，CPU，constrained decode，response-start candidate开启；
- preflight：passed；full diagnostic report SHA-256为`F56C6E8B491EDC60A8659024227F7394D50054F20640791DB27E3E6C9F9C6AD3`；checkpoint SHA-256为`A6526B8A065D3F0AA871DE310F06494947F70F3A98CF626E93BF9605685C0507`；
- 结果：train exact/sequence/response-start top1均为1.0；dev/final exact、sequence和response-start top1均为0；paired exact delta为0，恢复重复性为true。

判读：response-start candidate能够拟合已见训练条件，但未把条件表示迁移到未见答案或未知策略；不进入S2/L2/Mini。下一包已转入R2-H3.3条件表示泛化与课程/容量诊断。

### 10.3 H3.3-A 泛化剖面

对H3.2隔离checkpoint执行了独立只读泛化剖面，未进行训练。结果：train目标首字节top1=1.0，dev/final目标首字节top1=0；dev/final目标首字节均未在train出现，dev/final首字节margin分别为-0.29182和-0.27589；training_performed=false、checkpoint_read_only=true、recovery_repeatable=true。该结果表明原smoke同时含有“输出起点支持集不足”和“条件迁移失败”两个因素，不能直接据此判定核心架构容量不足。

下一步不是扩大原smoke的同质训练，而是构造共享首字节支持、完整response仍未见的family-disjoint控制集，再做课程与容量对照；该控制集属于H3.3诊断，不是Mini/L2题集。

### 10.4 H3.3-B 共享支持控制集

新增诊断集`tests/fixtures/r2_language_alignment_generalization.jsonl`，其dev/final目标首字节和`unknown_policy`均在train有支持，但完整response与family保持未见。早期10 epoch、300k参数预算结果的train response proxy=0.61164、paired sensitivity=0.28571，后来因容量预算会计修正而标记为**探索性、已被预算一致复测替代**，不再作为当前数值证据。

因此下一包改为H3.3-C phase-consistent完整response candidate：从assistant boundary开始让同一隔离native owner承担整段response的概率与学习，再与本控制集做同预算对照；不把该诊断集当作Mini/L2验收题集。

### 10.5 H3.3预算一致复测与H3.4逐位置审计

H3.3-B/C使用同一数据、同一300k参数预算和10 epoch，core=262,839，单个隔离candidate=11,051，effective=273,890，均在训练前通过checkpoint保存/恢复preflight。B启用`predictive_readout.response_start`，C启用`predictive_readout.response_phase`；两者train/dev/final exact与sequence均为0，dev/final首字节top1均为0.25/0，paired改写敏感性均为0.42857。B的dev/final首字节margin为0.06973/-0.02639，C为-0.10842/-0.18765。对应压缩结果分别见`reports/taiji_r2_h3_3b_course_300k_10e_20260916.json`和`reports/taiji_r2_h3_3c_course_300k_10e_20260916.json`。

H3.4使用上述两个child checkpoint和同一控制集，逐位置记录目标rank、概率、熵、累计似然，再绑定自由生成的原始bytes、UTF-8合法性、无替换字符、end-marker边界、停止原因和碰撞。B/C的continuation legal top1均为train/dev/final=0.78431/0.37143/0.26667，end-marker位置均为1.0；自由生成的UTF-8、无替换和边界率均为1.0，但exact/sequence仍为0，dev/final碰撞率为0.5/0.333。结果见`reports/taiji_r2_h3_4_conditional_credit_20260916.json`。判读为：停止与编码不是主要瓶颈，失败集中在条件response首字节和未见continuation迁移；不进入正式pilot、L2或Mini，不继续同质byte训练。H3.5已完成表示复审，H3.5-A已冻结持久response plan候选合同，下一步实现隔离candidate与plumbing smoke。
