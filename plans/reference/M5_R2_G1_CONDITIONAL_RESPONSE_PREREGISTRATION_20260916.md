# M5 R2-G1 条件回答接口与原生读出路由预注册

> 更新：2026-09-16。本文是 R2-P2 之后的独立接口版本，用于验证训练输入与原生回答读出是否同构。它不追改 [R2 aligned language pilot 预注册](M5_R2_ALIGNED_LANGUAGE_PILOT_PREREGISTRATION_20260916.md) 的 v1 结果，也不授权正式 256-episode pilot、默认入口切换、外部模型接入或 Mini 验收。
> 当前状态：G1-S0/S1 已完成协议与受控 smoke；preflight通过，但20 epoch混合训练仍出现50% train output collision、train exact=0.5，dev/final sequence criterion=0，paired改写敏感性=0。随后完成H2/H3只读审计：train prefix context distinct=2/2、context pair L2=0.9694、next-byte probability L1=0.0771、argmax difference=0、recovery repeatable=true。H3.1 beam只改变候选归属，没有聚合收益；H3.2 response-start candidate在20 epoch后train exact/sequence/top1=1.0，但dev/final exact/sequence/top1=0，paired delta=0，preflight和恢复均通过；H3.3-B/C同预算泛化与H3.4逐位置条件信用审计已经完成，仍未形成S2/L2能力结论，下一包转H3.5目标/数据/表示合同复审。

## 1. 目的

P2 已经证明评价层可以区分 UTF-8 合法性、回答边界、必需内容、禁止内容、未知策略和 exact response，但四臂输出仍未形成未见条件回答。20 epoch 诊断还出现了训练集局部代理上升、不同问题被同一个已见答案吞并的现象。

G1 只检验一个更窄的因果问题：

> 在不把 `task_family` 等评测标签直接喂给模型的前提下，显式的回答策略/边界与统一的 prompt serialization，能否让不同输入保持可区分的原生读出路径？

G1 不把“输出不同”当作正确答案，也不把规则评分当作完整语义理解。它只为判断下一层应该进入回答表示、读出 owner 还是数据目标复审提供证据。

## 2. 与 P2 的关系

| 项目 | P2 已冻结 | G1 新增 |
|---|---|---|
| 训练单位 | 结构化 episode、family/split 隔离 | 不改 episode 边界；更换为显式策略字段参与 prompt |
| 任务族 | `task_family` 仅作数据分组与报告分层 | 继续禁止直接作为 oracle 输入 |
| 策略 | `unknown_policy` 主要用于只读判分 | 写入 `<|policy|>`，作为运行时可提供的行为约束 |
| 读出 | native predictive readout、受限 UTF-8 | 保持同一 owner，加入停止原因和输出碰撞诊断 |
| 版本 | `taiji-native-language-alignment-v1` / `r2-episode-markers-v1` 的 P2 结果 | 新版本 `taiji-native-language-alignment-v2` / `r2-conditional-response-v2` |
| 信用 | required/forbidden/unknown 只读 | 仍不把规则结果写入 fast/slow；先验证条件路由 |

协议升级后，v1 checkpoint、v1 corpus digest 和 v2 prompt 不可混用。若恢复时版本、serialization 或 corpus digest 不一致，必须显式拒绝，而不是静默迁移。

## 3. 条件输入合同

v2 prefix 固定为：

```text
<|system|>
system text
<|policy|>
unknown_policy
<|context|>
context or <|none|>
<|history_user|>
...
<|history_assistant|>
...
<|user|>
user input
<|assistant|>
```

约束如下：

1. `system`、`policy`、`context`、`history` 和 `user_input` 都进入同一个 Taiji native predictive state。
2. `task_family` 只保留在 corpus、报告和分层统计中，不作为模型输入；这样 G1 不依赖评测集提供的标签路由。
3. `unknown_policy` 是系统行为合同，不是答案标签；运行时若没有显式策略，使用固定的 `answer` 默认值并在报告中披露。
4. 回答 target 仍是 `response + "\\n<|end|>\\n"`；生成器遇到完整 end marker 时停止，未遇到则记录 `max_generation_bytes`，不偷偷截成成功。
5. 保留 UTF-8 受限读出；合法字节、边界和语义规则分层记账。

## 4. G1 诊断指标

### 4.1 单样本指标

每个 episode 必须记录：

- `reference_response`、`unknown_policy`、`unknown_markers`；
- `generated_bytes_hex`、`generated_bytes_length`、`generated_text`；
- `utf8_valid`、`no_replacement`、`response_boundary_present`、`generation_stop_reason`；
- `required_terms_hit`、`required_term_coverage`、`forbidden_terms_hit`、`unknown_policy_satisfied`；
- `sequence_valid`、`semantic_criteria_pass`、`sequence_criterion_pass`、`exact_response`；
- `native_mode=true`、`external_provider=false`、只读评价标记。

### 4.2 条件路由指标

在 `train`、`dev` 和 `final` 分别报告：

- `unique_generated_texts`：输出文本的去重数量；
- `generated_text_collision_rate`：不同 episode 生成同一文本的比例；
- `response_boundary_rate` 和 `sequence_criterion_pass_rate`；
- 按 `task_family` 分层时只作分析，不用于修改模型输入；
- paired 原题/改写题的 prompt sensitivity 和 child output changed rate。

这些指标的用途是识别“输入被读到但所有答案走同一路径”的路由缺陷。它们不能单独证明答案正确、上下文保持或持续适应。

## 5. 受控验证顺序

### G1-S0：协议与恢复

1. 用四个受控 episode 生成 v2 corpus manifest 和 digest。
2. 运行零步 checkpoint 保存、全新进程恢复、parent 保护和 atomic save。
3. 验证 v1 checkpoint 或错误 corpus digest 被拒绝。
4. 固定解码设置，确认同一 v2 checkpoint 的只读评分可重复且不改变 checkpoint。

### G1-S1：条件区分 smoke

1. 仅使用隔离小样例，不启动正式 256 episode pilot。
2. 允许有限的诊断性过拟合，但必须同时记录每个 train episode 的生成结果，而不是只看 teacher-forced proxy。
3. 比较 zero、child、static、slow、fast、fast_slow 的输出碰撞、边界、语义和 exact response。
4. 至少保留一个不同 `unknown_policy` 的 final episode，验证策略字段进入输入后不会被当成 reference answer。
5. 所有评分在 read-only 模式完成；不得用 semantic 规则直接更新 fast/slow。

### G1-S2：出口判断

| 观察 | 允许的结论 | 下一层 |
|---|---|---|
| 条件变化后输出仍全部相同 | 输入表示或读出 owner 没有形成可用路由 | 内部表示/容量/读出评审 |
| 输出可区分但 required/unknown 不改善 | 有条件路由，目标内容或序列信用不足 | 目标与结果信用复审 |
| train 可区分，dev/final 碰撞高 | 可能记忆样本或 prompt 泄漏 | 数据族、表示泛化和保持复审 |
| dev/final 条件内容与边界可重复改善 | S2 候选证据，可申请独立 G1 pilot | 冻结新题集和正式预算，不自动进入 L2 |

本轮实际落点：train prefix context 已保持2/2 distinct，context pair L2=0.9694、next-byte probability L1=0.0771，但 argmax difference=0；原题改写的 context L2=1.0004、probability L1=0.0797，recovery repeatable=true。H3.1的bounded beam能改变候选序列，却只交换两个train样本的正确/错误归属，collision、exact和sequence聚合均无改善。H3.2的response-start candidate使20 epoch train exact/sequence/top1=1.0，但dev/final exact/sequence/top1=0，paired delta=0；判定为“边界首字节可对已见条件拟合，但条件表示未迁移”，不进入正式 pilot，进入H3.3。

无论哪种结果，都不能把 UTF-8 合法率或 surprise 改善写成模型已经会回答。

## 6. 禁止事项

- 不把 `task_family` 作为 oracle token 或隐藏 routing label；
- 不修改默认 `SeedRuntime.chat` 入口，不把 G1 隔离 checkpoint 直接挂到产品；
- 不复用 P3b 报告名、保护 checkpoint 或 P5.2d 未提交文件；
- 不把 `semantic_criteria_pass` 当作自由语义理解，也不把它直接当作反向梯度；
- 不因 G1 smoke 的局部改善扩大同质训练轮数；
- 不在 G1 完成前启动 Mini/L2/L3 用户验收。

## 7. 产物和血缘

G1 每次运行必须独立保存：

- v2 dataset manifest、源文件 hash、corpus digest；
- parent、zero、child checkpoint 和 checkpoint digest；
- 代码 revision、配置、seed、参数预算和设备；
- train/dev/final raw output 与上述单样本/聚合指标；
- preflight 日志、失败原因、环境限制和停止判断。

结果只能写入新的 G1 报告路径；P2 v1 报告按历史证据保留，不回填成 v2 结果。

## 8. H3.1/H3.2 实际执行结果

H2/H3内部表示与读出 owner 审计已经完成：使用同一 v2 checkpoint，在 prefix 结束处保存 native predictive context/state、下一字节概率分布及其 digest；跨 episode、原题/改写题和恢复重复性均已核对。状态已分离但输出仍碰撞。

H3.1 bounded beam在同一protected predictive owner上运行，保留greedy、原始bytes、UTF-8约束、stop reason、checkpoint digest和恢复重复性。结果显示候选输出发生变化，但两个train episode的结果只是互换，不能证明读出修复。

H3.2新增独立的`predictive_readout.response_start` candidate，只在显式assistant boundary后的第一个byte生效，后续仍使用protected continuation readout。它可以独立学习、checkpoint、恢复和只读测量，且不改变共享fabric、task_family输入或默认入口。20 epoch结果为：train exact=1.0、sequence=1.0、response-start top1=1.0；dev/final exact=0、sequence=0、response-start top1=0；paired exact delta=0；preflight、恢复和只读守卫为true。这个结果只说明已见样本拟合，不说明模型具备迁移回答能力。

## 9. H3.4结论与当前唯一下一步

H3.3-A只读泛化剖面已经完成：旧smoke的dev/final目标首字节均未在train出现，因而需要先排除输出支持集不足。预算一致的H3.3-B/C随后使用同一family-disjoint控制集、同一300k总预算和10 epoch完成对照：core=262,839、candidate=11,051、effective=273,890，preflight/恢复通过；两者train/dev/final exact与sequence均为0，dev/final首字节top1均为0.25/0，paired sensitivity均为0.42857。结论是支持集混杂已排除，但整段response仍未形成可迁移读出；不再追加同质byte训练。

H3.3-C的response-phase owner从assistant boundary开始承担整段response的概率和局部学习，不加入`task_family` oracle、外部provider、结果信用或默认入口。H3.4随后在固定B/C child上逐位置记录target rank/probability/entropy/cumulative likelihood，并绑定free-generation的UTF-8、无替换、end-marker、边界、停止原因和输出碰撞；结果见`reports/taiji_r2_h3_4_conditional_credit_20260916.json`。B/C的continuation legal top1均为train/dev/final=0.78431/0.37143/0.26667，end-marker位置均为1.0；自由生成合法率、无替换率和边界率均为1.0，但exact/sequence仍为0，dev/final碰撞率为0.5/0.333。结论是停止/编码不是主瓶颈，失败集中在条件response首字节与未见continuation的可迁移性。

当前唯一下一步是H3.5目标/数据/表示合同复审：检查byte级credit是否过细、prefix state是否能被response readout使用、family-disjoint课程是否覆盖共享起点但不同内容，并保持response边界、未知策略、历史上下文、旧能力保持、恢复和300k预算约束。H3.5必须形成版本化的高上限条件目标/表示方案及最小可证伪对照；若不能提出可证伪机制，才进入一次明确总预算的结构容量比较。未完成前不启动正式pilot、不进入L2/Mini、不修改默认入口或P3b/P5.2d历史与未提交实验文件。
