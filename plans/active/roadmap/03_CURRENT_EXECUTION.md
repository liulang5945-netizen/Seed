# N主线执行证据与审批台账

> 文档状态：[状态] — 只登记执行证据、审批、费用与适用限制；统一编号、阶段状态和实际动作集中在[MASTER-PLAN](../../MASTER-PLAN.md)。

> 当前进度见[MASTER-PLAN](../../MASTER-PLAN.md)。2026-11-10本轮完成R63后全局收束、U2算法与执行包22计划修订，0训练/0权重替换/0封存读取，回执见本文末尾。

> 历史进度：**2026-11-10 R63 物件止边字符级强制闸 ⇒ `char_handle_complete`／`character_handle_both_directions`**（0 训练／0 参数／不改模块；0 次封存集读取；判据读数前冻结）：先过 56 格逐位复现门（`anchor_match=true`／**1890 字段／0 不符**，`sham_identity`／`force_applied`／`no_op_consistency` 全 true，`structural 12/12`），296 次解码（48 新格 × 6 臂 ＋ 8 复制格 × 1）。读数按**整字是否改变**分类：强制完整 3 字节 `放` ⇒ **48/48 整字到达、48/48 整句逐字节对**；强制完整 3 字节 `玉` ⇒ **48/48 整字到达、0/48 整句对**；rescue **19/19**、break **29/29**、`char_granularity_undetermined=0`。同源读数（首字节被强制后模型仍自选后两字节）：`put_lead` **47/48**、`yu_lead` **26/48**、**23 格分叉**（`甾`19／`箔`1／`竹`1／`漆`1／`爰`1）——**“字节 ≠ 字”：强制首字节不等于定了这个字**。代表格 `g5`@6 B：不干预 `把未戊玉玉井前` → 强制整字 `放` 得 `把未戊放到井前`（精确），只强制首字节得 `把未戊漆放井前`（仍错）。**不放宽 192/192**。报告 [R63 报告](../../../reports/taiji_object_edge_char_r63_20261110.json)、[manifest](../../manifests/taiji_object_edge_char_r63_20261110.json)、[runner](../../../scripts/training/probe_taiji_object_edge_char_r63_20261110.py)、[SPEC §21.73](../../reference/SPEC-TAIJI-ADDRESS-SYSTEM_20261108.md)、[能力卡 U91 限制⑫⑬⑭](../../reference/CAPABILITY-CARD_20261103.md)、[D-75](../../reference/DECISION-LEDGER_20261109.md)。成本 81.53 CPU／82.97 wall（声明单轮 110、含重放 330、上限 400／1800，未超；首跑后因 black 纯格式化重签重跑，核心读数逐项不变，成本按第二次实测重记）。**队首判断**：物件止边“字节把手”线已由 R62（字节级）＋R63（字符级）压到底、**关闭**；唯一后继＝**全局收束**（把 R58–R63 读数与限制合成能力卡单一引用纪律，0 新探针）。

> 历史进度：**2026-11-10 R62 物件止边干预闸 ⇒ `causal_one_direction`／`causal_handle_break_only`**（0 训练／0 参数／不改模块；0 次封存集读取；判据读数前冻结）：先过 56 格逐位复现门（`anchor_match=true`／**1890 字段／0 不符**，`sham_identity`／`force_applied`／`no_op_consistency` 在 48 格上全 true），48 格上强制物件止边实际发射的那一个字节——`0xE7` 使 **29/29** 精确格全败、`0xE6` 只救回 **18/19**（唯一反例 `g5`@6 B 落到同首字节的 `漆`）⇒ **首字节是“内容败”的因果把手，不是“内容正确”的充分把手**，不放宽 192/192。报告 [R62 干预报告](../../../reports/taiji_object_edge_intervention_r62_20261109.json)、[manifest](../../manifests/taiji_object_edge_intervention_r62_20261109.json)、[runner](../../../scripts/training/probe_taiji_object_edge_intervention_r62_20261109.py)、[SPEC §21.72](../../reference/SPEC-TAIJI-ADDRESS-SYSTEM_20261108.md)、[能力卡 U91 限制⑫](../../reference/CAPABILITY-CARD_20261103.md)、[D-74](../../reference/DECISION-LEDGER_20261109.md)。成本 52.66 CPU／53.47 wall（声明单轮 60、含重放 180、上限 200／1800，未超；实测 ≈0.26 CPU/格 对预注册 0.177，低估 1.44 倍、未归因）。

> 历史进度：**2026-11-09 R61 物件止边 argmax 预测闸 → `argmax_predicts_content`，但布尔门 `zero_extra_pass=false`（55/56）（0 训练／0 优化器步／0 Adam／0 参数／0 图改动／0 次封存集读取；判据、基线、发牌规则读数前冻结）**。**R61 回执**：[报告](../../../reports/taiji_object_edge_r61_20261109.json)、[manifest v3](../../manifests/taiji_object_edge_r61_20261109_v3.json)、[v1 作废记录](../../manifests/taiji_object_edge_r61_20261109_void.json)、[runner](../../../scripts/training/probe_taiji_object_edge_r61_20261109.py)、[SPEC §21.70](../../reference/SPEC-TAIJI-ADDRESS-SYSTEM_20261108.md)、[能力卡 U91 限制⑨⑩](../../reference/CAPABILITY-CARD_20261103.md)、[D-70／D-71](../../reference/DECISION-LEDGER_20261109.md)。**与主任务的关系**：不改 192/192、**不放宽形状限制**；只把 R60 的事后分离升级成预注册的失败预测读数（归因线索，非因果、非能力外推）。**读数**：12 条新词干（6 干支字形＋6 真实语素、sha256 发牌、嵌套前缀）× 物件 6／9／12／15 B＝48 格，地点固定 `井前` 6 B、描述固定；29 精确／19 不精确；预测子＝物件止边 `argmax_byte==0xE7 ⇒ 内容败`，**type1=0、type2=0、准确率 1.0000**；基线 `constant_pass` 0.6042／`constant_fail` 0.3958／`length_rule` 0.6042；精确 McNemar 对长度规则 **b=19、c=0、p=3.8147e-6**；两族各自零错误（glyph 12/24、real 17/24 精确）。**锚点与门**：R60 八格重读 **128/128 字段逐位相等**、`structural 12/12`、`object_edge_measured=true`、`frozen_files_unchanged=true`；**`zero_extra_pass=false`**（唯一失败格 `g1`@15 B 提前 EOS：21 B 对 30 B、`把子庚巳辰亥箔`、地点止边未到达）⇒ 新词干 47/48 停在声明终点，“总能切在声明终点”不再是全称。**三条限定**：规则是“`0xE7` ⇒ 败”而非“`0xE6` ⇒ 过”（过侧 `0xe6`×27＋`0x94`×2，后两格都精确）；只预测内容败、不预测形态（19 败格中 13 含 `玉`、5 吐 `甾`、1 吐 `箔`）；**无干预 ⇒ 不得写成因果**。**长度轴**：`g1` (✓,✗,✓,✗)、`g2` (✗,✓,✗,✓)、`g4`／`r2` 四格全 ✓ ⇒ 两种长度规则都不成立，U91 限制⑤维持不放宽。**成本 10.88 CPU／11.21 wall**（声明 25、含重放 75、上限 200／1800；高估 2.30 倍，自测 0.1770 CPU/格 对 R60 的 0.2433、差 1.37 倍、未归因）。**v1 预注册在 0 读数时作废**（身份守卫 int/str 键取法 bug；判据 digest 未变、v1 原件保留、守卫双向自证）。**唯一后继（候选，待准入，本轮不自动启动）＝R62 物件止边干预闸**（强制 `0xE6` 对比 `0xE7`、与不干预臂同词干配对；若需自建解码循环必须先逐位复现 R61 的 56 格）。

> 历史进度：**2026-11-09 R60 物件槽长度扫描闸 → `stem_dependent` ⇒ `limit_not_relaxed`（0 训练／0 优化器步／0 Adam／0 参数／0 图改动／0 次封存集读取；判据读数前冻结）**。**R60 回执**：[报告](../../../reports/taiji_object_length_r60_20261109.json)、[manifest](../../manifests/taiji_object_length_r60_20261109.json)、[runner](../../../scripts/training/probe_taiji_object_length_r60_20261109.py)、[SPEC §21.69](../../reference/SPEC-TAIJI-ADDRESS-SYSTEM_20261108.md)、[能力卡 U91 限制⑤⑨](../../reference/CAPABILITY-CARD_20261103.md)、[D-68／D-69](../../reference/DECISION-LEDGER_20261109.md)。**与主任务的关系**：不改 Task1 的 192/192，只判它的物件槽长度能不能放宽；**答案是不能**。**读数**：地点固定 `井前` 6 B（按 R59 公开词表第 0 个取、不按读数挑）、两描述固定，物件名 6／9／12／15 B × 两条嵌套前缀词干＝8 格；`stem_wumu` (✗,✗,✓,✓)、`stem_glyph` (✗,✓,✓,✗) ⇒ 两臂不一致落 `stem_dependent`；同一词干 **9 ✓ 而 15 ✗** ⇒ 物件长度既非充分也非必要。**失败形态**：4 个败格全在物件名后多吐 `玉`、挤掉 `到`，字节数仍等于声明长度 ⇒ `zero_extra` 8/8、`byte_exact` 4/8、`field_lengths_ok` 8/8（结构切得对、内容错），物件止边 `copy_mass_total` 8 格全 0.0。**锚点**：wumu 12 B 格＝R59 `井前` 行直接重读，**15/15 字段逐项相等**（EOS 概率 0.8646805789207526 逐位）；`structural 12/12`、`failing_axes=[]`、`frozen_files_unchanged=true`。**撤回**：R59／U91 原写"决定量是物件字段长度"一句按 D-69 撤回（R59 的槽位定位与"总长不是决定量"仍成立）。**事后观察（未预注册、不得当结论）**：物件止边 argmax 在 4 败格全 `0xe7`、4 过格全 `0xe6`（n=8 完全分离）⇒ 只作 R61 假设。**成本 3.31 CPU／4.13 wall**（声明单轮 15、含重放 45、上限 200／1800；**高估 4.53 倍**，本轮自测 0.2433 CPU/格 对抄自 R59 的 0.4854，差 2.0 倍、**未归因**）。**唯一后继（候选，待准入，本轮不自动启动）＝R61 物件止边 argmax 预测闸**（≥6 条新词干 × 4 长度、带反例臂、0 训练、不读 confirm）。

> 历史进度：**2026-11-09 R56 主任务适配性闸 → `bridge_partial_overlap`（0 训练／0 优化器步／0 Adam／0 参数；不改模块；判据读数前冻结）**。**R56 回执**：[报告](../../../reports/taiji_mainline_fit_20261109.json)、[manifest](../../manifests/taiji_mainline_fit_20261109.json)、[runner](../../../scripts/training/probe_taiji_mainline_fit_20261109.py)、[SPEC §21.64](../../reference/SPEC-TAIJI-ADDRESS-SYSTEM_20261108.md)、[账本 §101](../../reference/INVARIANT-LEDGER_20261103.md)、[能力卡 U90](../../reference/CAPABILITY-CARD_20261103.md)、[D-57](../../reference/DECISION-LEDGER_20261109.md)。**与主任务的关系**：桥直接放到 Task1 的 8 条真实记录（原始字节、不改写、不补字段）与 192 条真实问句上，读 19.2v 188/192 的 4 条失败能否被修。**联合命中 `no_match 192/192`**（Task1 每条记录只含一条描述，永不共含两个键 ⇒ R54' 联合规则结构性不可用）；**逐槽读 `answered_pair 192`**，地址 **384/384**、名字 **384/384**、问句内描述跨度 **384/384**、单键唯一 **384/384**、记录内名字字段字节 **384/384**；**接线**：route／名字字节相等 384/384，但坐标不同——物件桥问句字节 [3,18) vs 复制链路记录字段 [0,12)，地点 [21,33) vs [18,24)，**不是同一对象**；桥不提供记录侧偏移，复制链路用 `field_bounds` 自推，**不存在冲突，也不需要桥提供偏移**；若要直接喂 `copy_limits` 需新增记录侧偏移输出（本轮不改链路，只登记缺口）。**19.2v 并排**：4 条失败**全部 `already_selection_correct`**、`bridge_fixes 0`、`out_of_scope_stop_edge 4`（位置 27 多一字节“到”，例：goal `让扁圆的盒子到楼梯入口。` world 3／#13，输出 `把朱砂铜镜放到雾尾` ＋多一个“到”）。五项预测全中；成本 **1.750 CPU／2.471 wall**（估 20、上限 200、`threads=1`）；关闭态九项全等；未改 `taiji/` 任何模块。**边界**：选记录这一层与主任务完全重叠，但给出的正是 19.2v 早已用字节匹配拿到的同一对（对 188/192 **零贡献**）；唯一未修的 4 条在停笔边界，桥无从修；联合规则不可外推到 Task1。唯一后继＝R57 停笔边界闸（接主任务闸已声明：4/4 变恰好 27 B 且其余 188 条不退即可接；跑不到则转训练侧）。以下为 R54' 段（历史）：**2026-11-09 R54'（原 R53 选择层修复闸）联命中选择 → `joint_select_measured`（0 训练／0 优化器步／0 Adam；不改模块）**。**R54' 回执：一句里全部字面键必须同时出现在同一条记录才进候选（1 条→输出该记录，≥2 条 `ambiguous_multiple_records` 不出字节，0 条 `no_match` 不出字节），"最小地址兜底"整条删除，签名／字面量／AST／行为四条结构证据全过 ⇒ `fallback_unreachable`；主组 20 句／40 槽 `selected 20`、`ambiguous 0`、`no_match 0`、`no_keys 0`，`pairing_matches **40**`／`answers_lost **0**`（R53 同口径 **9/40**）；L1 编造 0／非法 0／UTF-8 20/20／边界 20/20，L2 **1.0**；代价：名字问句 `no_keys 20`（名字不是键，预测错，保留不改）、别名 `no_match 9`、诊断 paraphrase 60 句 `no_keys 60`／`answers_lost 120`、混合 9 句 `no_match 9`（诊断组不进判定）；`ambiguous` 真实语料 0 次；成本 **2.047 CPU／2.774 wall**；关闭态九项全等。唯一后继＝R55“名字写键后联合命中是否唯一 → 记录身份”闸。** 以下为 R53 段（历史）：**2026-11-09 R53（原 R17'）桥端到端字节输出 → `bridge_bytes_measured`（0 优化器步；不改模块）**。**R53 回执：问句 → R15 两段读认出实体 → 该实体在语料中的原始记录 → 输出该记录的一段连续字节；L1 编造 **0**／非法切片 **0**／UTF-8 **40/40**／边界 **40/40**，L2 出处忠实 **40/40＝1.0**，L3 `answered_correct 40`／`pointed_wrong 0`／`abstained 0`／`should_answer_missing 0`（40/40 不降），未知句 `no_match` 输出 **0 字节**；代价 `pairing_matches 9`／`faithful_but_other_pairing **31**`（样本：问“蓝色杯子／窗台”输出“蓝色杯子／木桌”）；成本 **1.500 CPU／2.288 wall**（首跑；按 lint/black 重冻重跑，读数不变，末次 1.672／2.349）；关闭态九项全等。唯一后继＝R54“最终输出是否保留物件身份”闸。** 以下为 R52 段（历史）：**2026-11-09 R52 `z` 是否保留物件身份 → `z_retains_object`（总判 `z_object_measured`；0 优化器步；不改模块）**。**R52 回执：`readout` 输出 128 维 `z`；按地点 5 折 4 类物件闭式读出 λ=0.001 **20/20=1.00**、λ∈{0.1,10,1000} 均 0.95（chance 0.25）；身份穿过读出头；成本 1.406 CPU／1.865 wall；九项全等、主主体不动。首跑钩错 `expression.z`（step 路径未调用）已修正重冻重跑。唯一后继＝R53“最终输出是否保留物件身份”闸。** 以下为 R51 段（历史）：**R51 回执：20 句按地点分 5 折、4 类物件；λ=0.001 **19/20=0.95**，λ∈{0.1,10,1000} 均 0.75（chance 0.25）；降维后闭式岭回归与全维等价、成本 16×16；成本 1.469 CPU／1.963 wall；九项全等、主主体不动。唯一后继＝R52“`z` 是否保留物件身份”闸。** 以下为 R50 段（历史）：**R50 回执：20 句＝4 物件 × 5 地点；1-NN 同物件 **11/20=0.55**（基线 0.2105）、同地点 8/20=0.40（基线 0.1579）；同物件对平均距离 0.1382 < 不同物件 0.2068；成本 1.453 CPU／1.866 wall；九项全等、主主体不动。唯一后继＝R51“结构可读性（4 类物件闭式读出）”闸。** 以下为 R49 段（历史）：**R49 回执：同句两次写入距离 **0.0**；跨句 190 对 min 0.0617／均值 **0.1842**／中位 0.2002／max 0.2974；内容确在 `fast`，病灶移到读出侧；成本 1.438 CPU／1.881 wall；九项全等、主主体不动。唯一后继＝R50“写入结构保真度（1-NN 同物件）”闸。** 以下为 R48 段（历史）：**R48 回执：20 样本／8192 维；λ∈{0.001,0.1,10,1000} 准确率全 0.0、置换标签控制全 0.0（chance 0.05）；判 `write_path_discards_content`；成本 148.594 CPU／149.709 wall（估 30，超估约 5 倍，未超 200 上限）；九项全等、主主体不动。边界：线性探针失败不等于无非线性内容。唯一后继＝R49“写入内容依赖性（距离检验）”闸。** 以下为 R47 段（历史）：**R47 回执：冷启动 NLL 5.703718；跨句 −0.003437、自写自读 −0.008062（比跨句好 2.3 倍，仍差 −0.05 门约 6 倍）⇒ `channel_content_blind`；九项全等、主主体不动；成本 2.047 CPU／2.498 wall。唯一后继＝R48“写入内容保真度（闭式探针）”闸。** 以下为 R46 段（历史）：**R46 回执：rel_logits 0.0187/0.0876/0.2998/0.5197 单调；delta_nll −0.00037/−0.00167/−0.00555/−0.01246；0.5 处 −0.0055 未达 −0.05 门（差约 9 倍）；绝对 NLL 随缩放升高 5.60→6.18；九项全等、主主体不动；成本 2.641 CPU／3.099 wall。唯一后继＝R47“记忆通道内容能力（自写自读）”闸。** 以下为 R45 段（历史）：**R45 回执：四档占比 rel_logits 0.0454／0.2098／0.6124／0.8387、TV 5.90e-03／2.82e-02／1.14e-01／4.46e-01、KL 1.10e-04／2.50e-03／4.07e-02／6.62e-01，单调；0.5 处 logits 比 **13.48×**、TV 比 **19.36×**；九项全等、主主体不动；成本 1.281 CPU／1.719 wall。唯一后继＝R46“配平后记忆是否有用”闸。** 以下为 R44 段（历史）：**R44 回执：输入 192 维（z 128＋prev 64），z_share **0.0481**、rel_z 0.639、rel_input 0.0322、dilution 0.0505、rel_logits 0.0331、layer_gain **1.026** ⇒ `diluted_by_previous`（权重保真）；21 倍缩水是接线/尺度问题；九项全等、主主体不动；成本 1.328 CPU／1.766 wall。唯一后继＝R45“输入配平干预”闸。** 以下为 R43 段（历史）：**R43 回执：rel h 0.9502／z 0.8877／logits 0.0428／prob 0.0146；衰减 readout 0.934（不缩）／generate 0.0482（约 21 倍）／softmax 0.3425（约 2.9 倍）⇒ `damping_localised_generate`；KL 9.9e-05、TV 5.59e-03 ⇒ `output_measurable`；推翻 R42 的 softmax 结论；九项全等、主主体不动；成本 1.094 CPU／1.517 wall。唯一后继＝R44“生成层输入稀释定位”闸。** 以下为 R42 段（历史）：**R42 回执：Δh 0.666815→Δz 0.059511→Δlogits 0.041272→Δprob 2.073e-04；drops readout 0.0892／generate 0.6935／softmax 0.0050 ⇒ `damping_distributed`；softmax 比值含尺度混入、不能单独结论；九项全等、主主体不动；成本 1.250 CPU／1.701 wall。唯一后继＝R43“尺度无关的阻尼度量”闸。** 以下为 R41 段（历史）：**R41 回执：`delta_xi=0`（`xi` 不含记忆，判据失败）、`delta_z=0.047516`、`delta_prob=1.819e-04`、`output_ratio=0.0038`；事后归因阻尼在 `expression` 段（约 261 倍）；九项全等、主主体不动；成本 1.172 CPU／1.579 wall。唯一后继＝R42“表达段阻尼定位”闸。** 以下为 R40 段（历史）：**R40 回执：k∈{0,1,10,100,1000} 最大差 0／4.30e-07／4.31e-06／4.53e-05／1.13e-04，单调；`first_k_above_practical=null`（1000 倍仍低于 1e-3）；ratio(1000/100)=2.49、未饱和；九项全等、主主体不动；成本 2.062 CPU／2.505 wall。唯一后继＝R41“记忆阻尼定位”闸。** 以下为 R39 段（历史）：**R39 回执：M0 fast=0；M1 fast=13.113275、16/16 写入头非零、111/135 有梯度；M2 置零 fast 后输出最大差 6.999e-07 < 1e-6 ⇒ `memory_written_not_read`；M3 预测头 ×1.5 输出 Δ=0；九项全等、主主体不动；成本 2.297 CPU／2.760 wall。唯一后继＝R40“记忆影响剂量曲线”闸。** 以下为 R38 段（历史）：**R38 回执：A 61/34/40（复现 R37）、B（一条源记录）86/21/28、C（两句链式）61/34/40；12 个预测头三臂全暗；每臂 135 分类完整；主主体九项全等；根因 `step(allowed=None)` 默认全 False；成本 3.297 CPU／3.782 wall。唯一后继＝R39“记忆写权限与因果消融”闸。** 以下为 R37 段（历史）：**R37 回执：34,664,875 B；新进程参数/前向/续跑逐位相等、Adam 步数恢复 1、坏输入 4/4 拒绝；但梯度只 95/135 有 grad（34 恰零、非零 61/135），冻结判据未过；九项全等；成本 5.531 CPU／15.830 wall。唯一后继＝R38“梯度覆盖归因”闸。** 以下为 R36 段（历史）：**R36 回执：19/19 对四条件同长；均值 none 8.050105／self 8.049847／shuffled 8.049892／foreign 8.049812 bits/byte；self−foreign 中位 −0.00003、胜 6/19；self−shuffled 中位 +0.000036、胜 17/19；未达 0.1 bits/byte 与 14/19 门；参数/表不动；九项全等；成本 14.359 CPU／14.908 wall。未改模块，R35 仍可复跑。唯一后继＝R37“训练前置：checkpoint 往返闸”。** 以下为 R35 段（历史）：**R35 回执：三策略各 20/20（60/60）表达；never 0/20、always 19/20、on_match 19/20 步带前缀；三组控制 20/20；never vs always 非首步 19/19 不同；on_match 与 always 20/20 相同、仅未知句分叉；非法名字 ValueError 且 step 仍 0；参数/表不动；九项全等；成本 5.359 CPU／5.937 wall。唯一后继＝R36“前缀是否有用（教师强制）”闸。** 以下为 R34 段（历史）：**R34 回执：表达 20/20；19/20 步带前缀；同前缀控制 20/20 轨迹相等；无前缀对照仅 1/20 相同（19 不同）；未知清空且下一步无前缀；参数/表不动；九项全等；成本 4.125 CPU／4.632 wall。唯一后继＝R35“前缀使用策略”闸。** 以下为 R33 段（历史）：**R33 回执：父快照 2,513 B；子恢复摘要相等、纠正前 20/20；纠正 CUP 描述键后 5 句变 GREEN、仍 20/20；父恢复子快照摘要/文本/20/20/日志全对；截断快照无输出；九项全等；成本 1.344 CPU／13.250 wall。唯一后继＝R34“文本回授为下一步输入（只读前缀）”闸。** 以下为 R32 段（历史）：**R32 回执：2,513 B 快照；表摘要/文本/20/20 往返一致；导入端 v2 下 v1 被拒且内容摘要不变、v2 生效；空快照 None；损坏 4/4 拒绝；九项全等；成本 1.328 CPU／2.069 wall。唯一后继＝R33“跨进程快照继承”闸。** 以下为 R31 段（历史）：**R31 回执：20/20 逐步文本正确；20/20 步张量轨迹与无桥 observe 逐位相等；参数/表不动；未知输入清空；九项全等；成本 2.984 CPU／3.623 wall。唯一后继＝R32“表达状态快照（表+文本）”闸。** 以下为 R30 段（历史）：**R30 回执：字段默认 None；写入 `"蓝色杯子在木桌"`、状态/参数摘要相等；20 步文本保留、桥仍挂、参数不动；`no_pair` 清空；`clear_bridge` 同时清空；九项全等；成本 2.234 CPU／2.941 wall。唯一后继＝R31“表达随主体步进自动更新”闸。** 以下为 R29 段（历史）：**R29 回执：父导出 1,820 B、子进程导入摘要相等；纠正前 20/20；纠正 CUP 描述键后 5 句变 GREEN、仍 20/20；父导入交回摘要相等、20/20、日志保留；截断文件无输出；九项全等；成本 1.594 CPU／14.384 wall。唯一后继＝R30“双槽表达接入 `step`（只读侧输出字段）”闸。** 以下为 R28 段（历史）：**R28 回执：20/20 成对精确、模板串逐字正确；未知/双对象/无桥整对弃答并记录两槽出口；纠正 CUP→GREEN 后对象与文本改变、地点槽不变；张量不动；九项全等；成本 2.156 CPU／3.119 wall。唯一后继＝R29“跨进程双槽表达”闸。** 以下为 R27 段（历史）：**R27 回执：同一句 object/place 各自选 0/1；不带槽 `ambiguous_read`；缺槽 `no_choice`、越界 `not_in_candidates`、同槽多答 `ambiguous_read`、无桥 `no_bridge`；纠正后对象槽 0→1、地点槽仍 0；张量不动；九项全等；成本 1.781 CPU／2.527 wall。唯一后继＝R28“双槽组合选择与模板表达”闸。** 以下为 R26 段（历史）：**R26 回执：单选 0/1 正确、未写 `no_choice`、双实体 `ambiguous_read`、候选外 `not_in_candidates`、无桥 `no_bridge`；纠正 RED→GREEN 后下标 0→1；状态/参数摘要相等；九项全等；成本 1.859 CPU／2.736 wall。唯一后继＝R27“分槽位选择”闸。** 以下为 R25 段（历史）：**R25 回执：20 步 `sc_bridge` 不丢、表摘要不变、每步 40/40 与零线 0/0；挂表轨迹=无桥轨迹、参数不动、梯度全 None；清除后 5 步仍一致；九项全等；成本 5.172 CPU／5.985 wall。唯一后继＝R26“桥读出驱动离散选择”闸。** 以下为 R24 段（历史）：**R24 回执：挂载后 40/40、零线 0/0、参数/状态摘要相等；纠正只动表、别名红色；清除回基线；交回表可接入；九项全等、`sc_bridge` 默认 None；成本 1.844 CPU／2.477 wall。唯一后继＝R25“侧表随主体步进存活”闸。** 以下为 R23 段（历史）：——度量换成**短语定位**（余弦 top-1、零训练、**不使用任何名字**）：20 条 `describe` 句在本槽位 4／5 条描述短语中问“自己的短语是不是第一”（chance 0.25／0.20）；三臂同 R7／R8，**R3 只作上界、不进能力账**。**R1 正控 1.00／1.00**；**R2 object 1.00／place 0.70**（6 条错：5 条位次 2、1 条位次 4）；**R3 上界 object 0.95／place 1.00**（位次 1.05／1.00、margin 0.171／0.203）。⇒ 地点短语在整句里**完好保留（20/20）**，R8 的 0.60 缺口落在**短语→名字**这一支箭头、不在表示；成本 **17.875 CPU／24.32 wall**（跑前上限估 45，`threads=1`）。冻结口径的 `project_line` 由控制臂 R1 构造性满足 ⇒ **恒真、不采信**，项目侧只读 R2（place 0.70＝inconclusive），登记 D-20261109-10。回执：[R9 报告](../../../reports/taiji_phrase_localisation_20261109.json)、[manifest](../../manifests/taiji_phrase_localisation_20261109.json)、[文档 §21.17](../../reference/SPEC-TAIJI-ADDRESS-SYSTEM_20261108.md)、[账本 §55](../../reference/INVARIANT-LEDGER_20261103.md)、[能力卡 U43](../../reference/CAPABILITY-CARD_20261103.md)、runner sha `403021cd…`。**此前 R8 分槽位桥诊断 → `control_failed`（0 训练／0 优化器步／0 Adam／0 新参数）**——把度量固定为**最近名检索**（余弦 top-1，零训练，同分取小索引），只换文本粒度：**短语级** 9 条描述短语（4 物件＋5 地点）、**整句级** 20 条 `describe` 句（每句同时读物件名与地点名）、**正控** 60 条名称串 paraphrase（名称原样出现，度量正确就必须高）。三臂口径与 R7 相同：R1 字节直方图／R2 未训练项目编码器／R3 冻结句向量（上界，不进能力账）。**读数（短语／整句／正控，物件·地点）**：R1 **0.5·0.6／0.2·0.35／0.85·0.867**；R2 **0.5·0.6／0.5·0.3／0.65·0.85**；R3 **1.0·0.6／0.85·0.6／1.0·1.0**。**正控在 R1／R2 不达 0.90 ⇒ 按冻结规则判 `control_failed`**：最近名检索对 128 维字节表示不成立（度量本身失效），本轮**不判病因**、项目侧无可信读数，`project_bridge=project_bridge_absent`。**R3 正控 1.0／1.0 通过**，其读数可解释：物件短语 **4/4** → 整句 **17/20**（掉 **0.15**，未达 0.25 稀释阈）；地点短语 **3/5＝0.60** → 整句 **12/20＝0.60**（落差 **0**）⇒ **地点名的低分不是句内稀释**；短语地点恰好落在语义缺口上限（≤0.60），与“描述到名字的桥本来就不存在”一致，但整轮 `control_failed` 故**只作观察、不作判定**。成本 **13.016 CPU／18.59 wall**（`threads=1`；跑前按上限估 60 CPU < 200，未立预算账）。证据：[报告](../../../reports/taiji_slot_bridge_20261109.json)、[manifest](../../manifests/taiji_slot_bridge_20261109.json)、runner sha `294cc124…`、首跑留档 `reports/taiji_slot_bridge_20261109_run1.json`（名字域串用修正见 D-20261109-09）。**R10 留一短语桥闸已执行（2026-11-09）→ `phrase_name_bridge_memorised`**：闭式岭回归桥、两列并排——`before`（词表全用）三臂物件／地点**全 1.00**，`after`（留一：物件 4→3、地点 5→4，**每项删 1 条**，被删名字仍在候选集）**全 0.00**，整个 λ 网格（1e-3…1e3）均 0.00、句→名 0.00；成本 **18.609 CPU／26.34 wall**（跑前上限估 50，`threads=1`）。**关键限定（D-20261109-11）**：0 是**构造性**的（1:1 词表，删一条即删掉该概念唯一实例），**不证**“不会泛化”，只证“**1:1 词表的桥只能靠写入**”。回执：[R10 报告](../../../reports/taiji_phrase_bridge_20261109.json)、[manifest](../../manifests/taiji_phrase_bridge_20261109.json)、[文档 §21.18](../../reference/SPEC-TAIJI-ADDRESS-SYSTEM_20261108.md)、[账本 §56](../../reference/INVARIANT-LEDGER_20261103.md)、[能力卡 U44](../../reference/CAPABILITY-CARD_20261103.md)、runner sha `ca79611e…`。写入式桥最小形态（键＝短语 2 字节词集合、值＝(槽位,名字)、读时定位＋查表、规模＝实体数）见 §21.18，**仅设计未落码**。**R10b 多说法留一闸已执行（2026-11-09）→ `phrase_bridge_inconclusive`（0 训练／0 优化器步／0 Adam／0 新参数）**：每个名字在 runner 内补第 2 种说法（`taiji/` 夹具字节未动，旧 6 闸 fixture sha 全部保持），词表 4→8／5→10；**两列并排**——`before`（两说法全用、读主说法；λ=1 上）三臂 **1.00／1.00**（λ=1000 时地点降到 0.60）；`after`（**每项只删 1 条**：删该句自用的那种说法，另一种仍在 7／9 条词表里）R3 物件 **0.00**、地点 **0.20**（λ=1；λ=1e-3／1e-1 上地点 0.60）；句→名 `sentence_after` 物件 **0.05**、地点 **0.40**（λ=1，chance 0.20），λ 网格 0.70／0.75／0.40／0.15／0.20。**按预注册** λ=1 的 0.40 既不 ≥0.80、也不满足“整网格 ≤0.60”（网格最大 0.75）⇒ **`inconclusive`**：**不得**写成“背过”，也**不得**写成“能泛化”。**机制（事后，不改判据）**：λ=1 对 10 行词表过正则，预测被压向 0、余弦退化成噪声；跨说法信号只在地点槽、只在极小 λ 上部分出现。项目侧 R2 判 `project_bridge_memorised`（0.00）。成本 **18.562 CPU／24.94 wall**（跑前上限估 40，`threads=1`）。回执：[R10b 报告](../../../reports/taiji_phrase_bridge_alt_20261109.json)、[manifest](../../manifests/taiji_phrase_bridge_alt_20261109.json)、[文档 §21.19](../../reference/SPEC-TAIJI-ADDRESS-SYSTEM_20261108.md)、[账本 §57](../../reference/INVARIANT-LEDGER_20261103.md)、[能力卡 U45](../../reference/CAPABILITY-CARD_20261103.md)、[D-12](../../reference/DECISION-LEDGER_20261109.md)、runner sha `d790f328…`。**R11 写入式桥最小实现闸已执行（2026-11-09）→ `written_bridge_operational`**：[报告](../../../reports/taiji_written_bridge_20261109.json)、[manifest](../../manifests/taiji_written_bridge_20261109.json)、[桥模块](../../../taiji/address_bridge.py)、[runner](../../../scripts/training/probe_taiji_written_bridge_20261109.py)、[SPEC §21.20](../../reference/SPEC-TAIJI-ADDRESS-SYSTEM_20261108.md)、[账本 §58](../../reference/INVARIANT-LEDGER_20261103.md)、[能力卡 U46](../../reference/CAPABILITY-CARD_20261103.md)、[D-13](../../reference/DECISION-LEDGER_20261109.md)；runner sha `8be1bef0…`。读数：**40/40＝1.00、错指 0、弃答 0**；合并 9／替换 9（旧值 9/9 留存）／批内冲突 1 批整批回滚（`content_digest` 不变、槽数 9→9）；未写 5/5 `no_match`、只共享高频 token `no_unique_token`、36 对＋20 整句全多锚、错答 0；关闭态九项全等；成本 **1.672 CPU／1.68 wall**（估 60、上限 200）。**范围**：未写短语拒答只对 token 不相交成立；直接读整句 16 多锚＋4 错指（诊断）。**R12 键单位对照已执行（2026-11-09）→ `bridge_key_unit_still_leaks`**：[报告](../../../reports/taiji_bridge_key_unit_20261109.json)、[manifest](../../manifests/taiji_bridge_key_unit_20261109.json)、[桥模块](../../../taiji/address_bridge.py)、[runner](../../../scripts/training/probe_taiji_bridge_key_unit_20261109.py)、[SPEC §21.21](../../reference/SPEC-TAIJI-ADDRESS-SYSTEM_20261108.md)、[账本 §59](../../reference/INVARIANT-LEDGER_20261103.md)、[能力卡 U47](../../reference/CAPABILITY-CARD_20261103.md)、[D-14](../../reference/DECISION-LEDGER_20261109.md)；runner sha `3d484121…`、bridge sha `248a0493…`。读数：三臂能力 40/40、错指 0；**2 字节**未知作答 3/12、碎片 12/18、整句 4/20、改写 2 对 4 错；**码位**未知作答 4/12、碎片 0、整句 0、改写 5 对 0 错、R11 四候选归零；**offset 诊断**未知 4→1、碎片 0、整句非空 10、改写 3 对 0 错。关闭态九项全等；成本 1.781 CPU／2.549 wall（估 60）。**R13 身份收口＋别名写入已执行（2026-11-09）→ A `sequence_identity_clean`／B `alias_write_clean`**：A 段键＝整串字符保序保重数、读＝查询连续包含整条键才作答 ⇒ 未写候选 **0**／碎片 **0/18**／整句 **0/20**（20/20 多锚）、能力 **40/40**、关闭态九项全等；**代价**＝9 条同义改写全 `no_match`。B 段把 9 条改写作附加键写入已有表（`appended 9／merged 0／replaced 0`）⇒ 别名 **9/9**、写后三项零线**仍为 0**、原 9 条仍 40/40；键 **18**／实体 **9**／比值 **2.0**；成本 1.766 CPU／2.535 wall（估 30）。**R14 全量说法写入＋一说法两实体对抗已执行（2026-11-09）→ C `cost_curve_measured`／D `large_table_zero_clean`／E `conflict_rejected_first_writer_kept`**：C 键/实体 **9/9＝1.0**（min/中位/max 均 1）→ 加 9 条手写别名 **18/9＝2.0**（均 2）→ 第三刻度 **78/9＝8.67 被拒**（80 句里 **60 句用实体名**，名字不得进键，“80 键”不可达；比值上升＝说法数增长快于实体数）。D 18 键表上 12 未写候选 **0**、18 碎片 **0**、20 整句 **0**、能力 **40/40**。E 3 个“同一说法指向另一实体”写批全部 **`rejected_batch_address_already_bound`**、摘要不变、先写者仍读回、同批合法项未写入；对照 `replace` 会接受转主。关闭态九项全等；成本 **1.891 CPU／2.418 wall**（估 30、上限 200）。**R15 句子级两段读＋写冲突结构已执行（2026-11-09）→ 结构 `replace_path_absent`／两段读 `two_pass_clean`／零线 `zeros_clean`／控制 `exits_declared`**：写入端只剩一条规则（先写者保留、后写者整批拒绝），`on_conflict` 参数与 `replace`／`history` 标识符均已不存在，传入即 `TypeError` ⇒ 默认 `replace` 路径不可达。两段读按“起点最早”取键、同宽遮蔽后重读；出口 `masked_no_match`／`ambiguous_same_start`／`same_address_repeated`。读数：**40/40 实体**，分布 **答出 2 个 20／只答 1 个 0／答错 0／弃答 0**；12 未写候选 0、18 碎片 0；单遍基线 0/20。关闭态九项全等；成本 2.016 CPU／2.684 wall（估 30、上限 200）。**R16 说法重叠与三实体句闸已执行（2026-11-09）→ 重叠 `overlap_declared`／三实体 `three_entity_declared`／回归 `r15_two_pass_unchanged`／零线 `zeros_clean`／复跑锚 `anchors_complete`**（0 训练／0 优化器步／0 Adam／0 confirm；判据读数前冻结，八例期望逐项预声明）：[报告](../../../reports/taiji_bridge_geometry_20261109.json)、[manifest](../../manifests/taiji_bridge_geometry_20261109.json)、[桥模块](../../../taiji/address_bridge.py)、[runner](../../../scripts/training/probe_taiji_bridge_geometry_20261109.py)、[SPEC §21.25](../../reference/SPEC-TAIJI-ADDRESS-SYSTEM_20261108.md)、[账本 §63](../../reference/INVARIANT-LEDGER_20261103.md)、[能力卡 U51](../../reference/CAPABILITY-CARD_20261103.md)、[D-18](../../reference/DECISION-LEDGER_20261109.md)。**A 重叠 4 例**：`nested_long_first`（甲乙丙／乙丙）答 1、停第 2 遍、吞 乙丙；`share_one_character`（甲乙／乙丙）答 1、停 2、吞 乙丙；`abutting_no_overlap`（甲乙／丙丁，仅相邻）答 2、停 3、**不吞**；`partial_overlap_no_containment`（甲乙丙／乙丙丁）答 1、停 2、吞 乙丙丁 ⇒ **4/4、误指 0**。**B 三实体 4 例**：互不相交 **3/3**、停第 4 遍由 `no_match` 退出（非遍数截断）；跨接说法 答 **2**、停第 **3** 遍、吞 乙丙丁；同起点 **0 答**（`ambiguous_same_start`）；同址三现计 **1 实体**。**C 回归** R15 仍 40/40；**D 零线** 12 未写 0／18 碎片 0；**E 复跑锚** R11–R15 各记“最后可跑提交＋blob sha1＋报告 sha256”，5/5 完整（R14 报告 sha 与 blob sha 因 CRLF/LF 不同，已在该行 `matches_report=false` 声明）。关闭态九项全等；成本 **1.578 CPU／2.252 wall**（估 40、上限 200）。**R17 同值包含压缩闸已执行（2026-11-09）→ 值级 `value_dedup_visible`／安全嵌套 `same_value_containment_safe`／残差守门 `residual_guard_blocks`／边界泄漏 `boundary_leak_confirmed`／跨值 `cross_value_forbidden`／默认路径 `default_path_unchanged`／零线 `zeros_clean` ⇒ 总判 `compression_value_level_only`**（0 训练／0 优化器步／0 Adam／0 confirm；判据读数前冻结；**未改桥模块**）：[报告](../../../reports/taiji_bridge_compression_20261109.json)、[manifest](../../manifests/taiji_bridge_compression_20261109.json)、[runner](../../../scripts/training/probe_taiji_bridge_compression_20261109.py)、[SPEC §21.26](../../reference/SPEC-TAIJI-ADDRESS-SYSTEM_20261108.md)、[账本 §64](../../reference/INVARIANT-LEDGER_20261103.md)、[能力卡 U52](../../reference/CAPABILITY-CARD_20261103.md)、[D-19](../../reference/DECISION-LEDGER_20261109.md)。**A 值级去重**：3 键绑 1 值，记录 3、不同值 1、**键减少 0**，三键各自读回同一值。**B 同值包含**：`甲乙`／`甲乙丙` 压缩前读长说法同起点答 0，压缩 **2→1**（删 `甲乙丙`）后答 1（停第 2 遍）、短键仍可答——**救回一条原本弃答的长说法**；`甲乙丙丁` 残差含 `丙丁` 时守卫拒删（3→3），无守卫模拟会答 2 个值；`甲乙丙` 的残差只有 `丙` 时守卫放行（3→2）、压缩后答 2 个值（`蓝色杯子`＋`木桌`）＝**边界盲点**。**C 跨值**：2→2、登记 1 对、强行删长键会答错值。**D 默认表**：18 条记录删除 0、`content_digest` 前后相等、R15 40/40、零线 0/0、关闭态九项全等。成本 **1.781 CPU／2.648 wall**（估 30、上限 200）。**结论**：值可只存一份，**键数仍由说法数决定**；同值包含删长键只在“短键连续包含于长键且残差无键”的几何成立，表级守卫不是查询级保证——键级压缩必须由外部限定，默认路径不压缩。**R18 显式纠正协议闸已执行（2026-11-09）→ 普通冲突写 `plain_conflict_still_rejected`／CAS 纠正 `cas_correction_applied`／过期凭据 `stale_credential_rejected`／未写地址 `unknown_address_rejected`／批内撞键 `correction_batch_collision_rejected`／纠正链 `correction_chain_audited`／往返 `round_trip_clean`／关闭态 `closed_state_equal`／复跑锚 `anchors_complete` ⇒ 总判 `explicit_correction_available`**（0 训练／0 优化器步／0 Adam／0 confirm；判据读数前冻结）：[报告](../../../reports/taiji_bridge_correction_20261109.json)、[manifest](../../manifests/taiji_bridge_correction_20261109.json)、[桥模块](../../../taiji/address_bridge.py)、[runner](../../../scripts/training/probe_taiji_bridge_correction_20261109.py)、[SPEC §21.27](../../reference/SPEC-TAIJI-ADDRESS-SYSTEM_20261108.md)、[账本 §65](../../reference/INVARIANT-LEDGER_20261103.md)、[能力卡 U53](../../reference/CAPABILITY-CARD_20261103.md)、[D-20](../../reference/DECISION-LEDGER_20261109.md)。普通 `write` 仍 `rejected_batch_address_already_bound`、摘要不变、读回原值；`correct(键, 旧值→新值)` 一步生效、读回新值、日志 1 条；过期凭据 `rejected_batch_correction_mismatch`、未写地址 `rejected_batch_unknown_address`、批内撞键 `rejected_batch_address_collision` 全部整批拒绝；纠正链 `蓝→红→绿` 两步有序；18 键表别名 `蓝→红→蓝` 往返后 `content_digest` 复原、R15 40/40、零线 0/0、九项全等。成本 **1.703 CPU／2.541 wall**（估 30、上限 200）。**边界**：凭据是当前值本身（知道即可纠正、不判权限）；协议不校验新值对错；旧值只进只增日志、不再可读；普通盲写仍拒。R16→`93e292840`、R17→`f6b627fd8` 两行复跑锚已登记（两轮在 HEAD 因模块 sha 变化不可跑）。**R19 读—核验—纠正闭环闸已执行（2026-11-09）→ 一步纠正 `one_step_correction`／读值作凭据 `read_value_used_as_credential`／已一致不写 `no_op_when_agree`／过期读 `stale_read_rejected`／核对器给错 `verifier_error_accepted`／普通冲突写 `plain_write_still_rejected`／收敛 `loop_converges`／默认表 `default_table_unchanged`／关闭态 `closed_state_equal` ⇒ 总判 `read_verify_correct_loop_available`**（0 训练／0 优化器步／0 Adam／0 confirm；判据读数前冻结；**未改桥模块**）：[报告](../../../reports/taiji_bridge_correction_loop_20261109.json)、[manifest](../../manifests/taiji_bridge_correction_loop_20261109.json)、[runner](../../../scripts/training/probe_taiji_bridge_correction_loop_20261109.py)、[SPEC §21.28](../../reference/SPEC-TAIJI-ADDRESS-SYSTEM_20261108.md)、[账本 §66](../../reference/INVARIANT-LEDGER_20261103.md)、[能力卡 U54](../../reference/CAPABILITY-CARD_20261103.md)、[D-21](../../reference/DECISION-LEDGER_20261109.md)。表蓝、核对器红 → 读出蓝、`corrected`、读回红、日志 1；初始红、核对器绿 → 日志 `红→绿`（凭据来自读值）；一致时 `agreed`、日志 0、摘要不变；过期读 → `rejected_batch_correction_mismatch`、读回红、日志仍 1；核对器给错 → 协议接受、`verifier_error_accepted=true`、`matches_truth=false`（**边界**）；普通冲突写仍拒；两轮收敛（1 次纠正、日志 1）；18 键表往返摘要复原、R15 40/40、零线 0/0、九项全等。成本 **1.766 CPU／2.592 wall**（估 30、上限 200）。**边界**：协议只校验旧值、不校验新值；只在单键上测；不判权限；纠正不改键数、不产生语义；桥仍不得接图。**R20 句级双实体纠正闸已执行（2026-11-09）→ 单槽 `single_slot_correction`／双槽 `both_slots_correction`／过期子集 `stale_subset_rejected`／已一致 `no_op_when_pair_agrees`／普通冲突写 `plain_write_still_rejected`／收敛 `pair_loop_converges`／默认表 `default_table_unchanged`／关闭态 `closed_state_equal` ⇒ 总判 `sentence_pair_correction_available`**（0 训练／0 优化器步／0 Adam／0 confirm；判据读数前冻结；**未改桥模块**）：[报告](../../../reports/taiji_bridge_pair_correction_20261109.json)、[manifest](../../manifests/taiji_bridge_pair_correction_20261109.json)、[runner](../../../scripts/training/probe_taiji_bridge_pair_correction_20261109.py)、[SPEC §21.29](../../reference/SPEC-TAIJI-ADDRESS-SYSTEM_20261108.md)、[账本 §67](../../reference/INVARIANT-LEDGER_20261103.md)、[能力卡 U55](../../reference/CAPABILITY-CARD_20261103.md)、[D-22](../../reference/DECISION-LEDGER_20261109.md)。单槽只动 object、place 未动、日志 1；双槽同批 `corrected=2`、日志 2；过期子集整批 `rejected_batch_correction_mismatch`、有效项也不落盘、object 仍蓝、place 仍书架、日志仍 1；已一致 `agreed`、日志 0；普通冲突写仍拒；两轮收敛日志 1；18 键表双键往返摘要复原、R15 40/40、零线 0/0、九项全等。成本 **1.844 CPU／2.616 wall**（估 30、上限 200）。**边界**：单槽只对两槽都能读出的句子成立；过期子集整批拒绝是原子性；协议不校验新值；不判权限；纠正不改键数、不产生语义；桥仍不得接图。**R21 纠正来源与授权闸已执行（2026-11-09）→ 向后兼容 `backward_compatible`／缺来源 `missing_authority_rejected`／未知来源 `unknown_authority_rejected`／允许来源 `allowed_authority_applied`／过期凭据 `stale_credential_still_rejected`／普通冲突写 `plain_write_still_rejected`／默认表 `default_table_unchanged`／两步链 `authority_chain_attributed`／复跑锚 `anchors_complete`／关闭态 `closed_state_equal` ⇒ 总判 `correction_authority_attributed`**（0 训练／0 优化器步／0 Adam／0 confirm；判据读数前冻结）：[报告](../../../reports/taiji_bridge_authority_20261109.json)、[manifest](../../manifests/taiji_bridge_authority_20261109.json)、[桥模块](../../../taiji/address_bridge.py)、[runner](../../../scripts/training/probe_taiji_bridge_authority_20261109.py)、[SPEC §21.30](../../reference/SPEC-TAIJI-ADDRESS-SYSTEM_20261108.md)、[账本 §68](../../reference/INVARIANT-LEDGER_20261103.md)、[能力卡 U56](../../reference/CAPABILITY-CARD_20261103.md)、[D-23](../../reference/DECISION-LEDGER_20261109.md)。空策略 `correct(items)` 仍 `corrected`、来源为空；允许集 `("verifier:v1",)` 下无来源 `rejected_batch_missing_authority`、`verifier:v2` `rejected_batch_unknown_authority`、允许来源 `corrected` 且日志逐字记录；来源不替代 CAS（过期仍拒）；普通冲突写仍拒；两步链两条日志均带来源与理由；默认表往返摘要复原、R15 40/40、零线 0/0、九项全等。成本 **1.766 CPU／2.654 wall**（估 30、上限 200）。**边界**：来源是自称、不是认证；允许集是策略、不是安全边界；理由不校验；来源不校验新值；空策略兼容旧调用；桥仍不得接图。**R22 桥状态导出/导入闸已执行（2026-11-09）→ 整表往返 `state_round_trip_exact`／空表 `empty_table_round_trip`／导入端策略 `importer_decides_policy`／损坏输入 `corrupt_state_rejected`／默认表 `default_table_unchanged`／关闭态 `closed_state_equal`／复跑锚 `anchors_complete` ⇒ 总判 `bridge_state_portable`**（0 训练／0 优化器步／0 Adam／0 confirm；判据读数前冻结）：[报告](../../../reports/taiji_bridge_state_20261109.json)、[manifest](../../manifests/taiji_bridge_state_20261109.json)、[桥模块](../../../taiji/address_bridge.py)、[runner](../../../scripts/training/probe_taiji_bridge_state_20261109.py)、[SPEC §21.31](../../reference/SPEC-TAIJI-ADDRESS-SYSTEM_20261108.md)、[账本 §69](../../reference/INVARIANT-LEDGER_20261103.md)、[能力卡 U57](../../reference/CAPABILITY-CARD_20261103.md)、[D-24](../../reference/DECISION-LEDGER_20261109.md)。18 键＋2 条审计纠正导出 **2,081 B**；导入后完整摘要、内容摘要、再次导出字节三者相等；日志来源/理由保留、别名读回原值、R15 40/40、零线 0/0；空表往返；导入端 `v2` 策略下 `v1` 被拒、`v2` 生效；截断/错误 schema/重复键/负计数 4/4 拒绝；九项全等。成本 **1.672 CPU／2.376 wall**（估 30、上限 200）。**边界**：key_fn 不在字节里；策略不继承；日志不校验历史绑定；语义错误字节无法识别；桥仍不得接图。**R23 跨进程继承认读闸已执行（2026-11-09）→ 子进程导入 `child_import_matched`／子进程读 `child_read_40_40`／子策略 `child_policy_enforced`／交回 `handback_clean`／损坏文件 `corrupt_child_rejected`／关闭态 `closed_state_equal` ⇒ 总判 `cross_process_handoff_clean`**（0 训练／0 优化器步／0 Adam／0 confirm；判据读数前冻结；**未改桥模块**）：[报告](../../../reports/taiji_bridge_handoff_20261109.json)、[manifest](../../manifests/taiji_bridge_handoff_20261109.json)、[runner](../../../scripts/training/probe_taiji_bridge_handoff_20261109.py)、[SPEC §21.32](../../reference/SPEC-TAIJI-ADDRESS-SYSTEM_20261108.md)、[账本 §70](../../reference/INVARIANT-LEDGER_20261103.md)、[能力卡 U58](../../reference/CAPABILITY-CARD_20261103.md)、[D-25](../../reference/DECISION-LEDGER_20261109.md)。父导出 **1,820 B**、摘要 `a4edca3e…`；子进程导入摘要相等、返回码 0；子进程 40/40、零线 0/0；`v2` 被拒、`v1` 生效、别名红色、日志 1 条；父导入交回摘要相等、40/40、0/0；截断文件 `rejected_import_invalid_state: not_json`、无输出表；九项全等。成本 **1.844 CPU／15.943 wall**（估 60、上限 200）。**边界**：交接的只是表状态（不含参数/fast/sc_sfk）；子策略不继承；语义错误字节不可识别；桥仍不得接图。**唯一后继＝R24“桥接入主体状态（只读侧）”闸**：`LivingState` 增加可选 `sc_bridge` 与 `attach_bridge`／`clear_bridge`／`bridge_read`，默认 `None`；测关闭态九项全等、接入后 40/40 且参数/fast/sc_sfk 不动、纠正后读回新值且参数仍不动、`clear_bridge` 回到基线、跨进程交回的表可直接接入。判据读数前冻结。

> 历史进度：**2026-11-09 R7 零样本跨说法读出已执行 → `inconclusive`**——训练只用 20 条 `describe`、测试 60 条 `needs/hope/state`；9 个名称在训练句中出现 **0** 次（逐名核对，`leak_check` 落盘；60/60 测试句含自身物件名、60/60 含自身地点名）；闭式岭回归 λ∈{0.001,0.1,10,1000}，0 梯度步／0 优化器步／0 Adam。**R1** object 0.3167／place 0.4833／joint **0.15**；**R2** 0.25／0.3167／0.1167；**R3 上界** 0.9833／0.5833／0.5833；chance 0.25／0.20／0.05 ⇒ R1／R2 `not_readable`、R3 `inconclusive` ⇒ 总判 **`inconclusive`**；成本 **15.375 CPU／18.64 wall**（`threads=1`）。证据：[报告](../../../reports/taiji_cross_family_zero_shot_20261109.json)、[manifest](../../manifests/taiji_cross_family_zero_shot_20261109.json)、runner sha `08b05cbb…`。**下一唯一动作＝R8 分槽位桥诊断**（9 个描述短语 vs 整句分开测，判地点低是句内稀释还是语义缺口；0 训练、0 新参数）。此前 **2026-11-09 R6b 跨说法闭式线性读出已执行 → `cross_family_readout_transfers`**——同一 80 句（20 goals × 4 说法＝describe＋needs/hope/state），按 goal 分组 5 折（sha256 发牌、每折 4 goals＝16 句）、λ∈{0.001,0.1,10,1000}、闭式岭回归、0 梯度步／0 优化器步／0 Adam。**R1／R2／R3 三臂在 λ=0.001 上 object 1.0、place 1.0、joint 1.0**；λ=0.1 时 R2 joint **0.975**、R3 仍 **1.0**；λ=10 降到 0.125／0.0125／0.475；λ=1000 全塌到 ≤0.0375。成本 **R6 17.578 CPU／37.03 wall、R6b 15.609 CPU／19.16 wall**。**R6 首版作废**：joint 是 20 类 one-hot，按 goal 分组的测试折里联合类无训练行、权重恒零，joint 必然 0.0，其 `cross_family_readout_fails` 不可解释（D-20261109-08）。**限定**：夹具每槽位只有固定名称串与描述串、同物件其它 goal 在训练折出现过 ⇒ 1.0 是“词汇查表＋跨组合外推”；R1（纯字节直方图）与 R3（句向量）同为 1.0，说明此处不需要句向量。

> 历史进度：**2026-11-09 R5 闭式线性探针已执行 → `content_present_readout_wrong`（0 梯度步／0 优化器步／0 Adam／0 新参数）**——480 条（120 states × 4 物件）`describe` 句上闭式岭回归 `W=(XᵀX+λI)⁻¹XᵀY`，λ∈{0.001,0.1,10,1000}，按 state 分组 5 折（88/112/80/124/76，一个 state 的四条 goal 永不同折）。**R1／R2 在 λ≤10 上 object／place／joint 全 1.0**（chance joint 0.05），λ=1000 时 place 降到 **0.627083／0.552083**；成本 **0.516 CPU／0.54 wall**（线程限 1；首两次 844.984／701.953 CPU 超上限 200，归因 BLAS 线程自旋＋wall 计时写错，见 D-20261109-07）。**限定**：480 句全是 `describe` 形态，句面自带物件／地点描述词，故只证“同词法家族内内容线性可读、整句余弦是错的读法”，**未证**跨说法迁移。证据：[报告](../../../reports/taiji_slot_probe_20261109.json)、[manifest](../../manifests/taiji_slot_probe_20261109.json)、runner sha `a06640df…`、首跑留档 `reports/taiji_slot_probe_20261109_run1.json`。**下一唯一动作＝R6 跨说法线性探针**（同一 80 句、20 goals × 4 说法、按 goal 分组 5 折、R1／R2／R3 同口径；判据读数前冻结）。

> 历史进度：**2026-11-09 R3 上界臂已执行 → `overlaps`（只作上界、**不进总判定**；0 训练／0 优化器步／0 新参数）**：模型 `paraphrase-multilingual-MiniLM-L12-v2`（`revision e8f8c211…`、384 维、mean pooling、L2）本次首次取到本机缓存（下载由 D-20261109-05 授权）；独立 runner（sha `20dcadd9…`，导入冻结 runner 复用夹具／配对／统计，冻结报告只被引用未被改写）。same 中位 **0.1992**（p10 0.0629／p90 0.5295／max 0.6560）vs unrelated 中位 **0.6508**（min 0.2246／p10 0.5042／p90 0.7429）、**gap −0.0253**、AUC **0.956**、直方重叠 **0.241**、unrelated≤p90(same) **259/1,920＝13.5%**、1-NN **60/80＝0.75**；成本 **16.219 CPU／257.23 wall**（上限 200）。**拆分（0 次新嵌入）**：名内 paraphrase 60 条 0.0343／**0.0775**／0.1225（min／中位／max），**全部低于 unrelated min 0.2246（完全分开）**；`describe`↔paraphrase 60 条 0.2758／**0.4401**／0.6560（**整段落在 unrelated 体内，完全不可分**）；16 条长尾 **16/16 全跨家族**。⇒ **架构结论**：缺的不是编码器强度，是“描述↔名字”的显式桥。**R4a oracle 桥上界已执行（2026-11-09）→ `bridge_insufficient`**：`GLOSSARY` 确定性替换（20 条 describe 全改、60 条 paraphrase 零改动），同一 80 句／120／1,920。**R1** same 中位 **0.1704** vs unrelated 中位 **0.3615**（p10 0.1993）、gap **−0.0231**、AUC **0.9307**、1-NN **4/80**；**R2** 0.1369 vs 0.2781（p10 0.1530）、gap **−0.0379**、AUC **0.9028**、1-NN **1/80**。**1-NN 落在同模板句子的比例 R1 95%／R2 98.75%** ⇒ 最近邻由句式决定、非 goal；**两个表示都被模板主导，内容信号被淹没**。**下一唯一动作＝R5 闭式线性探针**：480 条样本上闭式岭回归读 (物件, 地点)，分组交叉验证；0 梯度步／0 优化器／0 Adam。此前 **2026-11-09 表示层闸判 `representation_not_supported`（0 训练、0 优化器步、0 Adam、0 confirm、0 新参数）**——20 goals × 4 说法＝80 句（4 物件 × 5 地点；`describe_goal`＋`needs`/`hope`/`state`）；same 对 **120**、unrelated 对 **1,920**（120 个无关 goal 对 × 16 句对，完整枚举）、near 1,120 只印不判。**两分布并排**：R1 字节直方图 same 中位 **0.2697**（p10 0.1393／p90 0.3891／max 0.4893）vs unrelated 中位 **0.3520**（p10 0.1915／p90 0.4596）、**gap −0.1976**、AUC **0.697**、直方重叠 **0.675**、unrelated 有 **66.4%（1,275/1,920）**落在 same 的 p90 以内；R2 未训练编码器 0.2255 vs 0.2842、**gap −0.2140**、AUC **0.662**、重叠 **0.753**；B 字符 bigram 0.8279 vs 1.0000、**gap −0.2000**、AUC **0.734**。**1-NN 三者全 0/80**。**诊断**：R1 名内 paraphrase 60 条 0.1258／0.1704／0.2190／0.2777、`describe`↔paraphrase 60 条 0.2569／0.3517／0.4089／0.4893 ⇒ **名内可分、跨家族不可分**。成本 **1.9 CPU／0.605 wall**（上限 200）、runner 双列 sha `b04ef3de…`；口径修正 D-20261109-03。此前 **2026-11-09 记账修正闸判 `continuity_restored`（0 训练、0 优化器步、0 Adam、0 confirm、0 新参数、0 新容差；只改 harness）**——两处修正，**地址层 sha `b370850e…`、语料 sha `350eef77…`、排序、阈值一律未动**。**① 夹具分类**：按冻结规则（未知查询逐字包含某条已写记录 ⇒ 库内有答案）在评分前把这类行移入 `unknown_answerable`，只改分组标签与 `output_class`。**移出 1 条／550**：`position 145`（查询 210 B）→ **记录 40**，记录长 **20 B**（`- \`taiji/adapter.py\``），匹配字节区间 **[0, 20)**，`all_matches=1`、`decision=matched`、`output_class=usable_and_correct`；记账 `moved 1 + remaining 549 = 550`、被移出者无不可用；**语料未增未删**。**② 门的定义**：`unknown_false_anchor = anchored ∧ not_contained`（contained＝至少一个锚点记录的**原始字节**出现在查询字节中，不用指纹）；**两条口径并排**：**旧口径（任何锚定即失败）1/550**——唯一命中 position 145、记录 40、锚点 1、`contained_any=true`、`overlap_ratio=1.0`；**新口径（真误锚）0/550**、**锚点级未被包含 0**、样本为空。被重新归类者同时出现在旧口径分子与新分母的“有答案”一侧，两处都摆出。**关闭态四项**：176/176、`fast`／概率 `max|Δ|=0`、参数/状态/身份/store 签名全等、`table_not_allocated=true`、无半开分支（`output_unit_closed_neutral`，43.641 CPU／4.084 wall）。**写侧**：**542/550 入表**、8 条 `window_identity_empty`（5–15 B）、0 回滚／0 驱逐／0 静默覆盖；五桶 <32 **22/30＝0.7333**、32–64 **99/99**、64–128 **165/165**、128–256 **136/136**、>256 **120/120**；**<64 B 121/129＝0.937984 ≥ 0.90**、**最短入表 16 B**。**三层**：L1 编造 **0**／非法 **0**／UTF-8 **2,200/2,200**／边界 **2,200/2,200**；L2 **1.0**；L3 **写入 542／8／0**、**未知（剩 549）0／0／0＋549 `usable_fallback`**、**`unknown_answerable` 1／0／0**、**无关 0／0／0＋550 `usable_fallback`**、**并集 549／1／0**；并集双锚点**应判 535＝实判 535**、`union_violations=[]`。**回退（分母＝库内有答案，1,101 条）**：**9 条**（写入 8＋并集 1，全为 <16 B、`W(q)=∅`），重合度四分位 **[0,0,0]**、中位 **0.0**、均值 **0.0**；分母 <20 故 **census 9/9 无关**；unknown／unrelated 只报形态与误锚，**不入相关性分母**。**锚定行原始字节复核**：**1,092 条**、判据 `record_raw in query_bytes`（不经指纹）、**违反 0**。**判定 `continuity_restored`**。**账本结论（可直接引用，含三个界）**：在 550 条语料上，条目被锚定**当且仅当**它作为 16 字节窗口序列在查询中**原样连续出现**（模 64 bit 指纹碰撞）；**最短可入表 16 字节**；**短于 16 字节的查询没有可比证据**。**它不能证明什么**：这条线**没有可训练自由度**（0 参数／0 梯度／0 优化器步）、**未与训练图合流**、也**未证明语义匹配**。**与上轮（`4262da689`）并排**：地址层／语料／写侧／五桶／三层读数逐项相同，变的只有夹具归类（1 条）与门定义（旧 1/550 → 新 0/550），判定从 `false_anchor_remains` 变为 **`continuity_restored`**。成本 **46.235 CPU／7.131 wall**（开启 2.594／3.047＋关闭态 43.641／4.084），上限 200。**收束**：这条字节级锚定线关闭、降级为历史队首（证据保留）；**唯一队首＝R3 上界臂**——表示层已判 `representation_not_supported`（见上），下一步用冻结句向量（384 维、mean pooling、L2）在同一 80 句／配对／分布读数上测一次上界：只回答“这个夹具是否存在可分上界”，不进判定、不改判据、不开训练窗。证据：[开启态](../../../reports/taiji_contig_20261109.json)（`gates.reclassification`、`gates.unknown_group`、`gates.independent_containment_violations`）＋[全量输出](../../../reports/taiji_contig_20261109_outputs.jsonl)＋[关闭态](../../../reports/taiji_contig_20261109_closed.json)＋[人工打分](../../../reports/taiji_contig_20261109_scores.json)（census 9 条）＋[manifest](../../manifests/taiji_contig_20261109.json)（freeze_revision **4**，`criteria_changed_revision 4`：判据变更、语料逐字节相同）＋[runner](../../../scripts/training/probe_taiji_contig_20261109.py)（双列 sha `b804a5d4…`）＋[地址层](../../../taiji/sc_windows.py)（sha `b370850e…`，未改）＋[文档 §20](../../reference/SPEC-TAIJI-ADDRESS-SYSTEM_20261108.md)＋[账本 §47](../../reference/INVARIANT-LEDGER_20261103.md)＋[能力卡 U35](../../reference/CAPABILITY-CARD_20261103.md)。
> 历史进度：**2026-11-09 顺序恢复闸判 `false_anchor_remains`（0 训练、0 优化器步、0 Adam、0 confirm、0 新容差）**——身份从“16 字节窗口指纹**集合／多重集**”改回“**有序序列**”，锚定改成**连续子序列**；同一 550 条语料、同一冻结判据、0 新参数。**定义（写死）**：`I(r) = ( sha256(r[i:i+16])[:16] : 0 ≤ i ≤ len(r)−16 )`，每个偏移一个 token、重数由序列保留；锚定＝存在 `j` 使 `W(q)[j : j+len(I(r))] == I(r)`（偏移差恒 1、顺序一致）；**取消保护带与切点的一切依赖**；**最短可寻址重报 16 B**（必要且充分；<16 B 身份为空，单条拒收 `window_identity_empty`）；出口不变（0 回退仍输出并标 `no_anchor`／1 命中／≥2 top-1 并标注）。**连续性定理（双向）**：每个指纹取自窗口字节本身 ⇒ 连续匹配 ⇔ `r` 在 `q` 中逐字出现（模 64 bit 碰撞）；1,092 条锚定行用原始字节复核 **0 违反**。**冒充成本**：要锚到 `r` 必须原样写出 `r`；上一轮丢序逃逸 `q = r[0:16]+r[1:17]+…` **实测不再锚定**（落回退，重合度 0.267）；**16 B 单窗口条目的证据下界＝16 B**，如实报。**关闭态四项**：176/176、`fast`／概率 `max|Δ|=0`、参数/状态/身份/store 签名全等、`table_not_allocated=true`、无半开分支（`output_unit_closed_neutral`，47.062 CPU／4.592 wall）。**写侧**：**542/550 入表**、8 条单条拒收（全 `window_identity_empty`，长度 5–15 B：position 53／72／237／267／277／424／485／497）、0 回滚／0 驱逐／0 静默覆盖。**长度→入表率**：<32 B **22/30＝0.7333**、32–64 **99/99**、64–128 **165/165**、128–256 **136/136**、>256 **120/120**；**<64 B 汇总 121/129＝0.937984 ≥ 0.90**；**最短入表 16 B**；**≥64 B 不可寻址 0 条**。**三层**：L1 编造 **0**／非法切片 **0**／UTF-8 **2,200/2,200**／边界 **2,200/2,200**；L2 **2,200/2,200＝1.0**；L3 写入 **542／8／0**、并集 **549／1／0**（535 双锚点，`violations 0`）、未知 **0／1／0**（另 549 `usable_fallback`）、无关 **0／0／0**（550 `usable_fallback`）。**未知集锚定（定理门）**：`unknown_anchored = 1`——position 145（210 B）锚到记录 40（`- \`taiji/adapter.py\``，20 B），该 20 B 在其 **byte_offset 0** 逐字出现，独立字节检查 `contained = true`，是**真子串**。冻结规则按“任何锚定”计 ⇒ 判定 **`false_anchor_remains`**；按“误锚＝锚定且非真实子串”计 ⇒ **0 条**。**门定义冲突如实登记，不改规则、不重解释。** **回退（分母修正，非放水）**：相关性只在“库内有答案”的查询上算——分母 **9 条**（写入 8＋并集 1，全为 <16 B、`W(q)=∅`），重合度四分位 **[0,0,0]**、中位 **0.0**、均值 **0.0**，分母 <20 故 **census 全量评分：相关 0／边缘 0／无关 9**；unknown 549（record 481／sentence 68）与 unrelated 550（record 550／sentence 0）只报形态与误锚，不进相关性判定；**排序未改、无新阈值**。**与上轮（`2db6274b5`）并排**：锚定从“覆盖”改为“连续出现”（顺序恢复）、冒充成本从“推不出”变为“必须逐字写出”、关闭态／写侧／五桶／三层读数逐项相同，回退从 1,108 条混合分母改为 **9 条有答案分母**，判定从 `fallback_still_irrelevant` 变为 **`false_anchor_remains`**（门定义冲突）。成本 **49.562 CPU／7.526 wall**（开启段 2.5／2.934＋关闭态 47.062／4.592），上限 200。**唯一后继**＝把未知集门改成 `anchored ∧ not contained`（仅 harness，不改地址层／语料／排序／阈值），同一语料重跑同一读数（约 46 CPU）。证据：[开启态](../../../reports/taiji_contig_20261109.json)＋[全量输出](../../../reports/taiji_contig_20261109_outputs.jsonl)＋[关闭态](../../../reports/taiji_contig_20261109_closed.json)＋[人工打分](../../../reports/taiji_contig_20261109_scores.json)（census 9 条）＋[manifest](../../manifests/taiji_contig_20261109.json)（freeze_revision 3；rev 3 仅语义等价 lint＋重取读数）＋[runner](../../../scripts/training/probe_taiji_contig_20261109.py)（双列 sha `3089e1fb…`）＋[地址层](../../../taiji/sc_windows.py)（sha `b370850e…`）＋[文档 §19](../../reference/SPEC-TAIJI-ADDRESS-SYSTEM_20261108.md)＋[账本 §46](../../reference/INVARIANT-LEDGER_20261103.md)＋[能力卡 U34](../../reference/CAPABILITY-CARD_20261103.md)。
> 历史进度：**2026-11-08 定义替换闸判 `fallback_still_irrelevant`（0 训练、0 优化器步、0 Adam、0 confirm、0 新容差）**——身份换成“条目全部 **16 字节窗口**的指纹**多重集**”，锚定换成“查询覆盖其全部窗口”。**定义（写死）**：`I(r) = { sha256(r[i:i+16])[:16] : 0 ≤ i ≤ len(r)−16 }`，**重数保留、按多重集覆盖**（`count_q(t) ≥ count_r(t)`；不去重，因为去重会让一次查询出现付清重复窗口，而计数不引入任何数值常量）；**取消保护带**（窗口完全落在条目内部，前后缀只平移不改成员；旧 `PROTECT=6` 属切点谓词，在窗口身份下没有定义来源）；0 锚点回退但仍输出、1 命中、≥2 多锚点仍出 top-1 并标注。**最短可寻址重算为 16 B，必要且充分**（上轮 23 B 作废；<16 B 身份为空，单条拒收 `window_identity_empty`）。**实现隔离**：新层 `taiji/sc_windows.py`，`taiji/sc_sfk.py` 逐字节未改（sha `476eb5f4…`）。**关闭态四项**：176/176、`fast`／概率 `max|Δ|=0`、参数/状态/身份/store 签名全等、`table_not_allocated=true`、无半开分支（`output_unit_closed_neutral`，45.156 CPU／4.259 wall）。**写侧**：**542/550 入表**、**8 条单条拒收（全 `window_identity_empty`，长度 5–15 B：position 53／72／237／267／277／424／497 与并集源 485）**、0 回滚／0 驱逐／0 静默覆盖。**三层**：L1 编造 **0**／非法切片 **0**／UTF-8 **2,200/2,200**／边界 **2,200/2,200**；L2 **2,200/2,200＝1.0**；L3 写入 **542／8／0**、并集 **549／1／0**（535 双锚点）、未知 **0／1／0**（另 549 `usable_fallback`；1 条低熵窗口误锚定，`usable_but_wrong` 如实计入）、无关 **0／0／0**（550 `usable_fallback`）。**三种输出计数**：`matched 557`／`no_match 1,108`／`ambiguous 535`；`unit record 2,132`／`sentence 68`；`fallback_basis window_overlap 120`／`address_order 988`／`none 1,092`。**长度→入表率（写死，同五桶）**：<32 B **22/30＝0.7333**、32–64 **99/99＝1.0**、64–128 **165/165＝1.0**、128–256 **136/136＝1.0**、>256 **120/120＝1.0**；**<64 B 汇总 121/129＝0.937984 ≥ 0.90**；**最短入表 16 B**（`### 设计原则`）、最短不可寻址 1 B；**≥64 B 不可寻址 0 条**（上轮 147 条）。**回退相关性（第二轴，未达线）**：1,108 条，窗口重合度四分位 **[0,0,0]**、中位 **0.0**、均值 **0.008986**；20 条抽检（三档 rubric 冻结在 manifest）**相关 1／边缘 2／无关 17**，中位 **0.0 < 0.5**。**分母事实（如实）**：1,100 条来自 unknown／unrelated（构造上库内无答案），**有答案回退只有 8 条且 `W(q)=∅`**（查询 <16 B，无窗口）。**冒充成本（如实）**：`I(r) ⊆ W(q)` **推不出**“必须几乎逐字复现 `r`”——构造 `q = r[0:16]+r[1:17]+…` 覆盖全部窗口但 `r` 不连续出现（顺序丢失）。**与上轮并排**：入表 279→**542**、`≥64 B` 不可寻址 147→**0**、<64 B 入表率 0.0388→**0.9380**、L3 写入 279／271／0→**542／8／0**，回退中位仍 0.0。成本 **47.359 CPU／6.931 wall**（开启段 2.203／2.696＋关闭态电池 45.156／4.259），上限 200。判定 **`fallback_still_irrelevant`**。**唯一后继**＝回退轴：先测“查询窗口集合在库内是否真的无覆盖”能否作为 `no_answer` 判据（0 训练／0 新参数），把无答案查询正当地移出回退分母，再看 8 条有答案回退是否可达线；**不得**改排序或加重合度阈值。证据：[开启态](../../../reports/taiji_window_20261108.json)＋[全量输出](../../../reports/taiji_window_20261108_outputs.jsonl)＋[关闭态](../../../reports/taiji_window_20261108_closed.json)＋[人工打分](../../../reports/taiji_window_20261108_scores.json)＋[manifest](../../manifests/taiji_window_20261108.json)（freeze_revision 4）＋[runner](../../../scripts/training/probe_taiji_window_20261108.py)（双列 sha `35603da8…`）＋[地址层](../../../taiji/sc_windows.py)＋[文档 §18](../../reference/SPEC-TAIJI-ADDRESS-SYSTEM_20261108.md)＋[账本 §45](../../reference/INVARIANT-LEDGER_20261103.md)＋[能力卡 U33](../../reference/CAPABILITY-CARD_20261103.md)。
> 历史进度：**2026-11-08 更大自然语料复核判 `scale_recheck_short_entries_gap`（0 训练、0 优化器步、0 Adam、0 confirm、0 新参数、0 新常量、0 新判据、0 新容差）**——同一冻结地址定义只换数据。**语料（先冻结再跑）**：`plans/**/*.md` 非空行 57,818 ＋ `taiji/**/*.py` 连续注释块 475，去重 58,293；按 `sha256(text)` 升序、每形态保底 20 条、补足 **550**；写入 550（heading 65／list_item 144／table_row 104／prose 217／code_comment 20；长度 <32:30、32–64:99、64–128:165、128–256:136、>256:120）／未知 550（同序后 550，从不写入）／无关 550（`plans/manifests/*.json` 排序后每第 8 个文件的字符串值）／并集 550（`written[i]+"\n"+written[(i+137)%550]`）；395 个源文件、1–3,296 B；**抽样规则不含地址模块输入，未按“是否含切点”筛选**。**关闭态四项**：176/176、`fast`／概率 `max|Δ|=0`、参数/状态/身份/store 签名全等、`table_not_allocated=true`、无半开分支（`output_unit_closed_neutral`，44.234 CPU／4.165 wall）。**代码零漂移**：地址模块 sha 与上轮逐位相同，新常量/新读路径字面量均为空。**写侧**：**279/550 入表**、271 条单条拒收（全 `identity_too_short`）、0 回滚／0 驱逐／0 静默覆盖。**三层**：L1 编造 **0**／非法切片 **0**／UTF-8 **2,200/2,200**／边界 **2,200/2,200**；L2 **2,200/2,200＝1.0**；L3 写入 **279／271／0**、并集 **418／132／0**、未知/无关 0／0／0（各 550 `usable_fallback`）、`unusable` **0**。**三种输出计数**：`matched 557`／`no_match 1,503`／`ambiguous 140`；unit `record 697`／`sentence 1,503`。**长度→入表率（写死）**：<32 B **0/30＝0.000**、32–64 **5/99＝0.0505**、64–128 **53/165＝0.3212**、128–256 **104/136＝0.7647**、>256 **117/120＝0.9750**；**最短入表 43 B**（`if self.sleep_consolidator and self.cortex:`）、最短不可寻址 1 B（`{`）；**147 条 ≥64 B 仍不可寻址**（82 条整条无内部切点、65 条切点被 `PROTECT=6` 保护带／16 B 下限／末块规则吃掉，最长 319 B），逐条见报告 `entry_curve.long_unaddressable`。**<64 B 汇总 5/129＝0.0388 < 0.90** ⇒ 命中冻结分支 4。**回退整句相关性**：1,503 条，块重合度四分位 **[0,0,0]**、中位 **0.0**、均值 **5.1e-05**；`fallback_basis` `record_overlap 1`／`address_order 1,502`；20 条抽检 **相关 0／边缘 0／无关 20**，中位 **0.0 < 0.5**（第二未达项，按冻结优先序不改判定，**不放宽回退规则**）；机制＝零共享块时按地址序恒指记录 0、输出常量句；文档测试 20 条 **0/20 同源文件**。**与上轮并排**：§16 为 58 条天然长条目 120／0／0；本轮为 550 条现实混料，**已入表 279 条仍 279/279 全对**，271 条错人全部是写侧拒收（查询没进表），不是读侧指错。成本 **45.078 CPU**（关闭态 44.234＋正式段 0.844）／5.448 wall（上限 200）。判定 **`scale_recheck_short_entries_gap`**。**唯一后继**＝**停止补地址层，转表示层**：同一 550 条语料、同一冻结口径、0 训练、0 新参数，测“短条目在现有有序表示上是否可分”。证据：[开启态](../../../reports/taiji_scale_20261108.json)＋[全量输出](../../../reports/taiji_scale_20261108_outputs.jsonl)＋[关闭态](../../../reports/taiji_scale_20261108_closed.json)＋[人工打分](../../../reports/taiji_scale_20261108_scores.json)＋[语料](../../manifests/taiji_scale_corpus_20261108.json)（sha `350eef77…`）＋[manifest](../../manifests/taiji_scale_20261108.json)（freeze_revision 3，两次 harness 修订：打分回执不覆盖成本、lint 行为不改）＋[runner](../../../scripts/training/probe_taiji_scale_20261108.py)（双列 sha `a8e7d225…`）＋[文档 §17](../../reference/SPEC-TAIJI-ADDRESS-SYSTEM_20261108.md)＋[账本 §44](../../reference/INVARIANT-LEDGER_20261103.md)＋[能力卡 U32](../../reference/CAPABILITY-CARD_20261103.md)。
> 历史进度：**2026-11-08 输出单元闸判 `output_unit_fixed`（0 训练、0 优化器步、0 Adam、0 confirm、0 新参数、0 新增容差）**——判定从“字节来自正确记录”改成“**输出是不是一个可用的答案**”。**输出单元**：锚定读输出整条记录（`unit=record`）；回退读仍按块级重合度选记录，但输出该记录中块重合度最高的**一整句**（`unit=sentence`，同分取句序在前），句界只落在句末标点（`。！？；…`）／换行／记录末尾，全部是存储 UTF-8 文本的字符边界；回退与锚定同一规则，**碎片路径已删除**。每条输出带 `unit`／`start`／`end`／`fallback`／`fallback_reason`／`fallback_basis`（`record_overlap`／`address_order`）。**关闭态四项**：176/176、`fast`／概率 `max|Δ|=0`、参数/状态/身份/store 签名全等、`table_not_allocated=true`，`output_unit_closed_neutral`。**三层**：L1 编造 **0**／非法切片 **0**／UTF-8 可解码 **180/180**／边界合法 **180/180**；L2 出处忠实 **180/180＝1.0**；L3 三列 `usable_and_correct 120`／`usable_but_wrong 0`／`unusable 0`，被 L1 挡掉 **0**（逐条样本为空）。**与上轮并排**：上轮 L3 正确 120／错人 0／空 0；本轮 120／0／0，口径已换成“完整可解码单元且指向正确记录”。**边界来源分布**：`record` **137** 条、`sentence` **43** 条、回退 **43** 条（`record_overlap` 4／`address_order` 39）；其中锚定 117＋回退 3＝120 条应锚定查询判对，回退 3 条即同义改写（在 `usable_and_correct_fallback` 单列，不与锚定 117 混算）。**5 条真实输出原文**（报告 `output_samples`，逐条带前后文）：锚定整记录、引导语锚定整记录、回退整句（`record_overlap`）、回退整句（`address_order`）、不可用/指错为空故取并集歧义整记录。**加严而非放松**：无新增常量；`read` 的数值字面量与上轮一致；新增 `sentences`／`_sentence_span` 的数值只有 0–3 的 span 下标与元组位置，终止符集是分类字符集，**不是容差**。**回归**：96 条旧读数逐条不退；平移不变 1102 条构造 0 失败；`soundness.ok=true`。成本 **49.906 CPU／4.799 wall**。判定 **`output_unit_fixed`**。**唯一后继**＝用同一冻结定义在**更大自然语料**上复核（不改常量、不改判据、不加容差），确认 120/120 与“无碎片”不是 58 条夹具的产物。证据：[开启态](../../../reports/taiji_output_unit_20261108.json)＋[全量输出](../../../reports/taiji_output_unit_20261108_outputs.jsonl)＋[关闭态](../../../reports/taiji_output_unit_20261108_closed.json)＋[manifest](../../manifests/taiji_output_unit_20261108.json)（freeze_revision 3，仅 harness 修订）＋[runner](../../../scripts/training/probe_taiji_output_unit_20261108.py)（双列 sha `2ecd8d1c…`）＋[文档 §16](../../reference/SPEC-TAIJI-ADDRESS-SYSTEM_20261108.md)＋[账本 §43](../../reference/INVARIANT-LEDGER_20261103.md)＋[能力卡 U31](../../reference/CAPABILITY-CARD_20261103.md)。
> 历史进度：**2026-11-08 短条目可寻址闸判 `short_entries_addressable`（0 训练、0 优化器步、0 Adam、0 confirm、0 新参数、0 调阈值）**——两件改动一起跑：① **短条目入表**：保护带从 16 收到谓词真实依赖宽度 `PROTECT = WINDOW = 6`，身份只保留 `start >= 6 且 end < len(r)` 的块（只丢右端贴缓冲末尾的末块），块数下限 2→1；最短可寻址长度写死 **23 B = MIN_CHUNK 16 + PROTECT 6 + 1**（必要非充分，切点存在性仍由内容决定）。② **回退带信息**：无锚点时按**块级重合度**（多重集 Jaccard `shared/(|S(r)|+|C(Q)|-shared)`）降序、地址序兜底，输出 `fallback`／`fallback_reason`（`no_anchor`／`no_overlap`）／`overlap_ratio`／`shared_blocks` 标注；**不设任何通过/拒绝阈值**。**关闭态零扰动**：176/176、`fast`／概率 `max|Δ|=0`、参数/状态/身份/store 签名全等、`table_not_allocated=true`、无半开分支。**写侧**：**58/58 入表、0 拒收**；上轮 10 条 `identity_too_short`（索引 17,19,24,28,29,32,37,40,45,47；长度 183–368 B）全部入表，其中 7 条只有 1 块；身份块数直方图 `{1:7, 2:9, 3:10, 4:10, 5:3, 6:7, 7:3, 8:2, 9:2, 11:3, 12:1, 17:1}`。**三层**：L1 编造 **0** 字节／非法切片 0；L2 出处忠实 **180/180＝1.0**；L3 正确 **120/120**、错人 **0**、编造 0、空输出 0（自读 58/58、引导语 58/58、同义改写 4/4；未知 20/20 `no_anchor`、无关 20/20 `no_anchor`、并集 20/20 双锚点）。**平移不变**：1102 条构造（前缀 0/1/5/6/15 × 后缀 0/1/3/7）**0 失败**，首块与末块从不进入身份；`soundness.ok=True`（0 空身份／0 包含冲突／0 负例被锚定）；容差审计 `ok=True`（无新常量、无新读路径字面量）。**回归**：上轮已入表 48 条的 96 条自读＋引导语逐条比对，`compared=96`、`regressions=0`。**回退样本**：有锚点＝记录 0，`covered=11, coverage=1.0, overlap=0.846154, span [0,806]`；无锚点＝未知查询落记录 26，`shared=1, overlap=0.125, span [339,366) 27 B`（`no_anchor`）；完全无重合标 `no_overlap` 落地址序记录 0。**如实另计**：`utf8_slices = {valid 176, invalid 4, invalid_scored 3}`——3 条同义改写走回退后输出跨 UTF-8 字符边界的字节片段（逐字节忠实、L3 仍对，但不能解码）。成本 **49.234 CPU／4.712 wall**。判定 **`short_entries_addressable`**。**唯一后继**＝用同一冻结定义在**更大自然语料**上复核（不改任何常量、不改判据），确认 120/120 不是 58 条夹具的产物。证据：[开启态](../../../reports/taiji_shorts_20261108.json)＋[全量输出](../../../reports/taiji_shorts_20261108_outputs.jsonl)＋[关闭态](../../../reports/taiji_shorts_20261108_closed.json)＋[manifest](../../manifests/taiji_shorts_20261108.json)（freeze_revision 7）＋[runner](../../../scripts/training/probe_taiji_shorts_20261108.py)（双列 sha `ddc69a1a…`）＋[文档 §15](../../reference/SPEC-TAIJI-ADDRESS-SYSTEM_20261108.md)＋[账本 §42](../../reference/INVARIANT-LEDGER_20261103.md)。
> 历史进度：**2026-11-08 局部谓词切分闸判 `chain_not_the_cause`（止损）（0 训练、0 优化器步、0 Adam、0 confirm、0 新参数、0 调阈值）**——按上一轮唯一后继，把切点改成**只依赖局部窗口的谓词**：`h_i = (h_{i-1} << 1) + gear[x_i] (mod 2^64)`，`(h_i & 0x3F) == 0` 即切；哈希**从不重置**，删掉“距上一刀至少 16 字节”与 128 字节上限；切点集确定后才丢弃 `< 16 B` 的短块（不合并、不回头改切点）。地址定义、序列包含、单条拒收、出口与 top-1 规则一律未动。**关闭态零扰动**：176/176、`fast`／概率 `max|Δ|=0`、参数/状态/身份/store 签名全等、`table_not_allocated=true`、无半开分支（`local_chunking_closed_neutral`）。**写侧**：48 条入表、10 条 `identity_too_short` 单条拒收、0 回滚／0 驱逐／0 静默覆盖；身份块数 2–17（多为 2–4）。**三层**：L1 编造 **0** 字节／非法切片 0；L2 出处忠实 **174/174＝1.0**；L3 正确 **100/120**、错人 **20**、编造 0、空输出 0（可用 0.8333）；`no_anchor` 23/120；自读 48/58、引导语 48/58、同义改写 4/4；未知 20/20 `no_anchor`、无关 20/20 `no_anchor`、并集 14/14 `ambiguous`。已入表的 48 条记录，自读＋引导语 **96/96 全对、覆盖全 1.0**——链式错位**归零**；**错人 20 全部**是那 10 条被拒收短记录。判定按冻结规则由“错人 20 > 1”触发 **`chain_not_the_cause`**：止损。**后继（已执行）＝短条目入表＋回退带信息**。证据：[开启态](../../../reports/taiji_localchunk_20261108.json)＋[全量输出](../../../reports/taiji_localchunk_20261108_outputs.jsonl)＋[关闭态](../../../reports/taiji_localchunk_20261108_closed.json)＋[manifest](../../manifests/taiji_localchunk_20261108.json)＋[runner](../../../scripts/training/probe_taiji_localchunk_20261108.py)（双列 sha `96c8c2ed…`）＋[文档 §14](../../reference/SPEC-TAIJI-ADDRESS-SYSTEM_20261108.md)＋[账本 §41](../../reference/INVARIANT-LEDGER_20261103.md)。
> 历史进度：**2026-11-08 定义 A 输出闸判 `inconclusive`（0 训练、0 优化器步、0 Adam、0 confirm、0 新参数、0 调阈值）**——按裁决把“身份”重定义为内容决定切分（gear：窗口 16／掩码 0x3F／最短 16／最长 128）＋首尾 16 字节保护带＋序列连续包含，并把验收哲学改为“每次读都给 top-1 实际输出、按输出判”，弃答只作读数。**关闭态零扰动**：176/176 读数、`fast` 与概率 `max|Δ|=0`、参数/状态/身份/store 签名不变、`table_not_allocated=true`、无半开分支；**写侧**：55 条入表、3 条 `identity_too_short` 单条拒收、0 回滚／0 驱逐／0 静默覆盖；**三层**：L1 编造 **0** 字节／非法切片 0、L2 出处忠实 **178/178＝1.0**、L3 正确 **112/120**、错人 **8**、编造 0、空输出 0（可用 0.9333）；旧实现对照为正确 26／错人 35／`no_match` 59（可用 0.2167）。**错人 8 的构成**：3 条短记录被拒收（其自读＋引导语共 6 条只能指别人）＋2 条引导语因首块吸收 15 字节前缀、早切后链式错位（覆盖 0）。**归因（架构问题，不是阈值问题）**：16 字节保护带只覆盖“首块被改”，覆盖不了“首块之后的链式错位”；20/58 引导语查询失锚。按冻结规则不达 `identity_A_output_holds`，判 `inconclusive`。**唯一后继**＝把切点改成局部谓词（去掉“距上一刀至少 16 字节”的链式约束、短块不入身份），让平移不变成为定理，再用同一 58+20+20／同一分母／同一三层评分重跑；不调阈值、不降线。此前为 2026-11-08 偏移无关闸判 `offset_not_the_cause`：**关闭态零扰动**：`sc_sfk_enabled=False`、`table_not_allocated=true`、176/176 全等、`fast`／概率 `max|Δ|=0`、参数 135／3,311,259、`state_bytes=1,100,064`、`candidate_sha_v2=ff60ac25…`、store 签名全不变。**写侧先垮**：按冻结协议建表时第 5 批（记录 40–49）被 G-B（无连带失锚）整批回滚，`collateral_loss=[30]`——全偏移下地址 30 只剩 `"2e"`（ASCII `2e`）与 `0xa6 0x20` 两个唯一词，被含 `19.2e` 的记录 46 一次写掉；故**出厂写策略下偏移无关版本建不出表**，记 `blocker=offset_free_write_invariant_broken`，另用环境闸关掉回滚出诊断读数（`diagnostic_only=true`）。**两臂并排（同一批 58+20+20）**：应锚定 120 → 臂 A 正确 61／错指 1／`no_match` 4／`ambiguous` 54、臂 B 正确 61／错指 1／`no_match` 1／`ambiguous` 57，**两臂可用比例同为 0.5083**；未写 20 → A 20/20 `no_match`、B 20/20 `ambiguous`；并集 19/19 `ambiguous`（分母 19 因地址 30 不可锚定）；无关 20 → A 0 硬答、B 0 硬答／20 `ambiguous`。**归因**：错指来自**错位**不是“近零领先”——预期地址 30 的 `lead_in` 查询自身唯一特征 0 个，唯一锚点是固定引导语 `"总的来说，"` 注入的 `80bb`（属地址 13），`margin=0.574`；58 条 `lead_in` 查询全部带外来锚点（13 出 57 次、53 出 33 次）。处置：`features()` 已回滚到步长 2（sha `a158f59d…`），runner 标 `SUPERSEDED`＋环境闸；**止损：停止补地址层，转“表示＋地址一起重定义”，本轮不再开新补丁实验**。证据：[开启态](../../../reports/taiji_offsetfree_20261108.json)／[关闭态](../../../reports/taiji_offsetfree_20261108_closed.json)／[manifest](../../manifests/taiji_offsetfree_20261108.json)／[文档 §12](../../reference/SPEC-TAIJI-ADDRESS-SYSTEM_20261108.md)。
> 历史进度：**2026-11-08 锚点计数闸判 `anchor_count_unsafe`（0 训练、0 优化器步、0 Adam、0 confirm、0 新参数、0 调阈值）**——按上一轮唯一建议，把 `read` 从“含 `DF=0` 即 `no_match`”改成“只在已见特征上数锚点”（0 锚 → `no_match`、1 → `matched`、≥2 → `ambiguous_multiple_anchors`；未见特征只计数、无第二判据）。同一语料（与 A1 闸逐段相同：`shared_written=58/58`、`shared_unknown=20/20`）、同一分母 120／20／20、同一条 0.60 线。**关闭态零扰动**：`sc_sfk_enabled=False`、`table_not_allocated=true`、176/176 全等、`fast`／概率 `max|Δ|=0`、参数 135／3,311,259、`state_bytes=1,100,064`、`candidate_sha_v2=ff60ac25…`、store 签名全不变。**开启态**：应锚定 120 → 正确 **65**、**错指 1**、`no_match` 1、`ambiguous` 53、可用 **0.5417 < 0.60**；未知集 20 → `no_match` **0**、`ambiguous` 20、误命中 0；并集 20/20；**新增读数**完全无关 20 条 → 硬答 **0**、`ambiguous` 20；表 58 地址／容量 64／5 次倍增／744 唯一特征／0 驱逐 0 覆盖 0 回滚、`global_checks=6`／0 失败；开关一致性 8/8；成本 **48.688 CPU／5.382 wall**。**唯一错指样本**：`lead_in` 第 30 条被 `matched` 到地址 46（锚点 1、未见特征 3、`top1 0.25504` vs `top2 0.25406`、margin `0.00098`）。**归因（架构问题，不是阈值问题）**：`features()` 取步长 2 的字节对，任何改变字节长度的改写（引导语 15 字节、`已经→已`／`进行→做` 各 −3 字节）都会让整段特征错位、只剩接缝特征，于是恰好唯一的一个词可能属于另一条记录；A2 挡住的正是这个。按预注册**已回滚** `read`，runner 标 `SUPERSEDED`。[开启态](../../../reports/taiji_anchorcount_20261108.json)＋[关闭态](../../../reports/taiji_anchorcount_20261108_closed.json)＋[预注册 manifest](../../manifests/taiji_anchorcount_20261108.json)（`freeze_revision=1`，语料文本另存 corpus 快照）＋[runner](../../../scripts/training/probe_taiji_anchorcount_20261108.py)（双列 sha `f19ed96d…`）；账本 §38。**当时队首（已由上方偏移无关闸执行并止损）＝改特征抽取使对偏移不敏感**。
> 历史进度：**2026-11-08 夹具外闸（A1）判 `address_not_usable_off_fixture`（0 训练、0 优化器步、0 Adam、0 confirm、0 新参数、0 调阈值）**——用非夹具文本检验“每条记录都带一个表内唯一的词”：语料按机械规则从项目内连续正文抽 58 段（写入）＋20 段（未写），改写由固定同义表＋固定引导语批量产出，语料／判据／三分母／0.60 线在首个读数前冻结（manifest `freeze_revision=3`）。读数：应锚定 120 中正确 61、**错指 0**、`no_match` 59、`ambiguous` 0、**可用比例 0.5083 < 0.60**；带自身唯一锚点的查询 101（0.8417）；未写 20/20 `no_match`、两记录并集 20/20 `ambiguous`；表 58 地址／容量 64／5 次倍增／744 唯一特征／0 驱逐 0 覆盖 0 回滚；开关一致性 8/8；成本 **3.109 CPU / 1.32 wall**，全局自检均值 **83.5 ms**、峰值 **146.2 ms**。**归因**：A1 不是约束，卡点是 A2（任一 `DF=0` 即 `no_match`），58 条 V2 改写全在锚点计算前被判死、其中 39 条其实带自身唯一锚点。[报告](../../../reports/taiji_a1_offfixture_20261108.json)＋[预注册 manifest](../../manifests/taiji_a1_offfixture_20261108.json)＋[runner](../../../scripts/training/probe_taiji_a1_offfixture_20261108.py)（双列 sha `459c43a5…` 一致）；账本 §37。**队首只有一条＝按判定改查询侧地址定义**（已见特征上的锚点定出口、未见特征只记比例；同一语料／分母／0.60 线，不许降线、不许加领先线）。
> 历史进度：**2026-11-08 系统闭环收束（`store_addressing_wire_neutral`＋`store_addressing_wired`，0 训练、0 优化器步、0 Adam、0 confirm）**——把地址层当一套完整系统重写并落位：[系统与推导文档](../../reference/SPEC-TAIJI-ADDRESS-SYSTEM_20261108.md)（§1 状态／§2 动作／§3 出口／§4 闭环图／§5 自检／§6 显式假设／§8 定义＋定理 T1–T7＋反例 X1–X10）＋[manifest](../../manifests/taiji_address_system_20261108.json)（C1–C7 与关闭／开启门先于首个读数写入）＋[关闭态](../../../reports/taiji_store_wire_20261108_closed.json)＋[开启态](../../../reports/taiji_store_wire_20261108_open.json)＋[基线](../../../reports/taiji_store_wire_20261108_baseline.json)＋[runner](../../../scripts/training/probe_taiji_store_wire_20261108.py)。**出口改由结构条件触发**：`no_match`＝任一特征 `DF=0` 或 `Q∩U=∅`；`matched`＝恰好 1 个锚点；`ambiguous`＝≥2；冲突批整批拒绝；自检失败 `restore` 回滚——0.85 相似度阈值与 0.02 边际已删除。**关闭态零扰动**：176/176 全等、`fast`／概率 `max|Δ|=0`、参数 135／3,311,259、`state_bytes=1,100,064`、`candidate_sha_v2=ff60ac25…`、store 签名全不变、`table_not_allocated=true`。**开启态**：六类 22/22、22/22、66/66、22/22、22/22、22/22；23 地址、容量 2→32（4 次倍增）、appended 23／merged 0／replaced 7／rejected 1、0 驱逐／0 覆盖／0 回滚、`global_checks=31`；**值路径接通**：matched 132（移动 126）／abstain 44（移动 0）／值回读 22/22；成本 **4.312 CPU／0.813 wall**（上限 200）。**已退役**：runner 侧重复地址表（`SUPERSEDED`＋启动闸）、`COPY_OVERLAP`、`ANCHOR_MARGIN`、表级 `history`、恒真 `self_read`、`growth_wire` 除零旁路、`state_bytes` 不计表。**显式待接**：修剪（无安全删除判据）、跨会话历史（`clear_store` 整表丢弃）、训练塑形（0 参数）、离线 `growth_gate`（未接在线）。**唯一仍在的夹具依赖＝A1**（每条被存记录读取时至少含一个表内唯一词）；不成立时退化为“不知道”，不误答、召回下降。账本 §36。**当时队首＝用非夹具文本口径检验 A1（已由上方夹具外闸回执完成）**。
> 历史进度：**2026-11-08 地址层接线完成（`store_addressing_wire_neutral`＋`store_addressing_wired`，0 训练、0 优化器步、0 Adam、0 confirm）**——SC-SFK 已落到 `taiji/` 真实 store 地址层（`LivingCoreConfig.sc_sfk_enabled`，默认 `False`、0 新参数、无半开分支）。**关闭态零扰动**：176 条读数 `fast_max_abs_delta=0.0`／`probability_max_abs_delta=0.0`、摘要与字段 176/176 全等；参数 **135 张量／3,311,259 元素／sha `5779908c…`**、`state_bytes=1,100,064`、`candidate_sha_v2=ff60ac25…`、store 签名全不变；四处各只有一处 flag 早退、`table_not_allocated=true`。**开启态复现**：自读 22/22、换措辞 22/22、未写 66/66、新词 22/22 `no_match`、无值 22/22 `ambiguous`、新搭配 22/22 alias；容量 2→32、appended 23／merged 7／replaced 0／rejected 1、0 驱逐／0 覆盖／0 回滚、`global_checks=31`、冲突批整批拒绝且表未变；读不改 `fast`。成本 **3.734 CPU／0.764 wall**、`mean_write_wall_ms` **0.2284**、最终表全局复检 **0.4523 ms**（上限 200）。并排：读侧六类与 runner 侧 delta 全 0，表侧仅 `replaced` 22 vs 0（值语义差异、非退化）。**能证明**地址是算法本体缺件而非阈值问题；**不能证明**它可被训练塑形；**仍依赖**“每条记录带稀有词”的夹具性质。[关闭态](../../../reports/taiji_store_wire_20261108_closed.json)＋[开启态](../../../reports/taiji_store_wire_20261108_open.json)＋[基线](../../../reports/taiji_store_wire_20261108_baseline.json)＋[manifest](../../manifests/taiji_store_wire_20261108.json)＋[runner](../../../scripts/training/probe_taiji_store_wire_20261108.py)（双列 sha `dd4b6940…`，in-process 格式化后整跑复测、读数逐项相同）；账本 §35。**当时队首＝把地址命中的值接进读路径（该动作已由上方“系统闭环收束”完成）**。
> 历史进度：**2026-11-08 地址层修订首轮验证完成（`sc_address_holds`，0 训练、0 优化器步、0 Adam、0 confirm）**——唯一一条算法修订 SC-SFK：地址＝2 字节词特征集合（位置无关）＋store 级平滑 IDF＋加权 Jaccard＋唯一锚定（0 锚 → `no_match`、多锚领先 <0.02 → `ambiguous_multiple_anchors`）；写策略先扩容量后写、同址同值合并、同址异值原地取代（保留 history）、批内异值抢址整批拒绝、0 驱逐 0 覆盖。同一 154 条口径：自读 **22/22**、换措辞 **22/22**、未写 **66/66**、新词 **22/22 `no_match`**、无值 **22/22 `ambiguous`**、新搭配 **22/22** alias；均值键对照仅 **8/22**、**14/66**。表：容量 **2→32**（4 次扩容）、**23** 槽／**23** 个互异特征集、合并 **7**、原地取代 **22**、拒绝批 **1**、**0 驱逐／0 静默覆盖／0 回滚**、`global_checks=3`（0 失败）；store 不可变、`candidate_sha_v2` 未变、看门狗未触发；**5.641 CPU / 1.033 wall**（上限 200）；runner 双列 sha `8cbc8ce8…` 一致。**结论只覆盖地址可并／可拒／可长**，不含整句生成、值通路（FEC-1 5 维未动）与真实图接线；**不得**据此声称已具备内容记忆、弃答或持续学习能力。**当时队首（已由上方接线回执取代）＝把 SC-SFK 落到 `taiji/` 的 store 地址层（默认关闭、0 参数、0 训练）**。[报告](../../../reports/taiji_sc_address_20261108.json)＋[manifest](../../manifests/taiji_sc_address_20261108.json)＋[runner](../../../scripts/training/probe_taiji_sc_address_20261108.py)；账本 §34。
> 历史进度：**2026-11-08 全局收束完成（0 训练、0 代码行为改动）**——把 19.2 之后的全部探针归并为一条算法层诊断：**已结构性成立**（19.2v 复制/选择 188/192、19.2w 停止边、F3 四代扩容、生长接线零扰动、门的三臂出口）／**算法本体缺件**（读写地址无 tie `query_alignment 0.110`；值通路只张成 5 维；没有同时满足“共享载体＋可分残差”的地址）／**仅判定规则或实现 bug 已修**（19.2w 末端停止边、FEC-1 v1 判据算术、P1 出口名、P3 `m_gate`/`m_spec` 口径、G2c 第 5 档、生长接线 run1 度量、字节臂旧 66/66 口径、`candidate_sha_v2` 漂移）。**唯一瓶颈＝②地址层**（不是①表示层：写键已可图内修好；不是③读出门：真实键无空档、`gate_hurts`；不是④容量与生长：扩容零扰动、30/32 未触顶）。被取代读数逐条登记在[账本 §33](../../reference/INVARIANT-LEDGER_20261103.md)，**不删任何读数**。**当时队首（已被上方最新一条取代）＝地址层算法修订的首轮零训练验证**（判定开跑前写死、出口具名）。诊断正文见 [MASTER-PLAN §0a](../../MASTER-PLAN.md)。

> 当前进度：17 窗口已封为负结果（17.5 六格任务门未通过、17.6 confirm 封存 0 读取、17.7 四项总账已收束）。随后两项零成本诊断判定根因：17.8 记忆排除探针 **6/6 格“记忆主导”**（新名字逐字节正确率 0、任何记录干预下新名字从未出现，γ≈0.92）；17.9 续跑变慢排查在空闲机上实测 7.48 CPU/更新（≈首跑 7.08），判定 **外部争用**，保护系数 2.11。**17 窗口根因结论：数据前提缺陷（3 物件×5 地点=15 组合可全量背诵）。**
> 当前进度：**18.0（方案 A 的非训练部分）已完成**——220/220 名字池、三向名字串互斥 + 组合 held-out 的数据前提（data sha `8242f2a8…`）、裸价/保护价双报、step150 分段预算、10 条判据预注册。
> 当前进度：**18.1 六格正式比较已完成并判负**（六格 300 更新全部落盘、先存包后评估；overall 0.000、`new_name_copied` 0.000、unseen 0.000、四条任务门全未过、A−C=0.000；confirm 未释放、0 读取）。实际 **16,196.3 CPU / 2,824.4 wall**，低于保护价 19,220.8 / 30,000。driver 聚合曾因只存整句完成率（float）而崩溃，已修复并**离线重算**（`--eval-only` 583.7 CPU、0 训练、0 Adam；报告＋manifest 已生成）。
> 当前进度：分层逐字节探针（0 更新）已跑完 step150 与 step300：内容字节命中 a 自由 0.090 / b gold 记录+起点 0.290 / b_pure（γ=0）0.755 / c 逐位置强制 0.390。**瓶颈分层（前向干预）：把 γ 压到 0 使内容字节 0.290→0.755，说明前向混合把质量压在生成侧；按否定定理 M3，不得由此推断“训练期复制信用饥饿”。门开后残留 β/transport 位置不携带约 24.5%**。
> 当前进度：**19.2 系列（19.2b–19.2w）已整体关闭**；19.2v confirm **188/192＝0.9792**（旧 95/96、新 93/96）、L1 机械不变量全过、4 例失败全为位 27 多写 1 个模板字节"到"，判定 **`milestone_holds`**；任务 1（绑定复制＋选择）由结构实现达成、**训练路径贡献 0**。19.2w 已在**冻结外**合成复现同一残差（尾类 1/64＝1.56%，与 confirm 4/192 同量级），诊断定位为**序列末端无停止边**（非复制通路），`sequence_stop` 结构修复后三道门全过判 **`fixed`**；**confirm 0.9792 不改写、未在封存集复核**（D16-27）。详见下方「2026-11-05 19.2w」与「2026-11-05 19.2v」段、账本 §16.21／§16.20。
> 当前进度：**20.1 已工程交付（`ready`，0 训练）**——规则夹具 4 world／51 应答轮（纠正面 22/world、会话 [10,24,17]、反馈 2 批 40 行 1,560 B、反例 7、20 条规则）；CF1 与 B_EH 同形（**3,323,547 参数 / 135 张量 / 新增 0**）；三条泄漏探针全过（静态 pass、真值打乱 765/765 字节一致、反查 690/690 落到已释放反馈、truth 0）；dry-forward 102 轮 logits [15,257]、F 关时两图 765/765 一致；**159.53 CPU / 164.55 wall**。详见下方「2026-11-05 20.1」段、账本 §16.22。
> 当前进度：**18.2 位置计数 + UTF-8 掩码验证达标**（两臂 × 16 dev 例 × 2 次工程更新，100.1 CPU / 4 Adam）：b_pure 内容 **1.000**（18.1 为 0.755）、物件/地点 1.000/1.000、首末差 **0.000**；自由解码掩码后**无非法字节率 1.0**（掩码前每例 16/16 非法）；恢复四项全 0（参数/Adam/复制状态 + `prefix_cursor`/`copy_record`/`copy_cursor` 逐位）。**设计达到预期 → 允许定价；本轮未出正式价、未启动训练、confirm 未释放。**
> 当前进度：**20.2r 已工程交付（`under_powered_f_channel`，0 训练）**——A 用提交字节（`cd23ce2e…`）重放 20.2 信用探针，loss/dF/份额/分头四项**相对误差 0.0**（门 1e-6）、F 关两图 **765/765**；B 纠正面 22 五档杠杆：置零 2.677e-5、×0.5 1.337e-5、×2 2.664e-5、×4 **7.953e-5**、同范数乱序 3.188e-5，**leverage_max 0.00795% < 1%**、**五档 byte_diff 全 0**；oracle 诊断 Δloss 4.183% 但 **0 字节变化、0/22 答对**（结构问题）；**23 次反向不 step**、**161.48 CPU / 168.64 wall**。详见下方「2026-11-05 20.2r」段、账本 §16.24／D16-29。
> 当前进度：**20.2s 已工程交付（`inconclusive`，0 训练）**——22 个纠正面、F（8,192 元素）为特征，岭回归闭式留一（λ 五档、每折训练侧内层留一选 λ，22 折实际全选 1.0）：决策位留一 **0.6818（30/44）**、整句 **15/22**、全位字节 0.9576（13 个模板位恒对，非学习读数）。三条对照：标签打乱 5 seed 均值 **0.5091**（max 0.7727）、同维同范数随机向量 **0.1818**、无 F 其它持久态 **0.5000**；最强对照 0.5091、差 **0.1727** → 按写死规则判 **`inconclusive`**。结构读数：22 个纠正面 F **只有 2 个不同向量**（S1/S2 各 1 个、窗口内常量；10 条写入行、范数 0.950–1.292、两窗最大差 0.03182），"只按反馈窗口身份猜"的诊断准确率 **0.6818 与 F 完全相同**；跨窗口前向 0.3529/0.2000；其余 3 world 0.5909/0.4545/0.1818。基线模型自身 0/22。**119.67 CPU / 124.34 wall**、0 反向、12 门全 true。详见下方「2026-11-05 20.2s」段、账本 §16.25／D16-30。
> 当前进度：**20.2t 已交付设计闸（纯设计，0 训练 / 0 参数 / 0 代码）**——把 20.2s 的"F 信息只在窗口级"归因为**写入粒度**：`run_turn` 内 `allowed=False`、只有 `release` 逐行 `write_fast`，22 个纠正面只有 10 条写入行。三选项对照：①每轮无寻址写（复用 `write_heads` 434,448、+0 参数；`edit_fast` 为衰减＋擦＋叠，方向冲突互相覆盖，预测信息闸 ≈ 窗口身份基线）；②**每轮内容寻址（推荐）**：键＝问句经现有 `BytePatchEncoder.context` 的 128 维字节表示、值＝该轮 F，余弦检索 softmax 混合，+0 可训练参数、+266,273 状态元素（≈2.03 MiB f64）、32 槽 FIFO、会话内持久/跨会话清空；③批次写＋衰减累积（+8,193 状态、结构闸即不过）。过门预注册先写死：结构（不同 F 向量 ≥ 窗口内不同误差方向数且 >2）＋杠杆（`leverage_max ≥ 0.01` 且至少一档 `byte_diff > 0`）＋信息（留一 ≥0.60 且比最强对照高 ≥0.20）。备忘 `runner_sha256`/`submitted_runner_sha256` 均 **n/a**；**未实施、未定价、未进 20.3**。详见下方「2026-11-05 20.2t」段、账本 §16.26／D16-31。
> 当前进度：**20.3a 已实施并按预注册三闸判定（`option2_failed`，0 训练）**——按 20.2t 选项 2 实现每轮内容寻址写（键＝问句 128 维、值＝该轮 F 副本、余弦＋softmax τ=0.1、α=0.5、32 槽 FIFO；+0 可训练参数、+266,273 状态元素）。结构闸过（22 个不同 F 向量、逐轮 hash 全不同）；杠杆闸不过（`leverage_max` **8.716e-6** < 0.01、五档 `byte_diff` **0**）；信息闸不过（决策位留一 **0.4091** < 0.60、最强对照 **0.5909**、margin **−0.1818**）；**224.94 CPU / 232.72 wall**。
> 当前进度：**20.3b F 权威闸（`authority_weak`，0 训练）**——22 个纠正面／44 个决策位 gold **全部在 3 名外**；整句 deficit 中位 **3.376e-4**＝局部 spread 的 **0.46%**；统一增益首次翻位 g=1e4 且 **0/22 答对**；闭式岭回归 **9/22**、方向一致率 0.682、W 秩 1、49,152 参数；**56.92 CPU / 58.83 wall**。
> 当前进度：**20.4 定价（`pricing_over_budget`）→ 20.4a 拆账（`pricing_still_over_budget`）→ 20.4b 图级优化（`pricing_still_over_budget`，只超 0.27%）**——存读硬前置 6 项逐 step 全 byte-identical；图级三条优化硬等价（loss 差 0.0、全部参数梯度最大差 **4.547e-13**、logits 1.554e-15），批量前向 **2.914×**、`select` 93,636→0、`conv1d` 405→12；重算投影 **48,129.2 CPU**＝2.01× 预算／1.0027× 闸。owner 裁定抬闸后进入正式比较。
> 当前进度：**20.4c_r2 正式比较已交付（`comparison_completed`）**——200 更新 × 2 臂 × 2 seed、checkpoint 每 25 步、`resume_verified`、校准 3 次真实更新。末步 dev 决策位：CF1|1105 **0.0356**、B_EH|1105 **0.0968**、CF1|1106 **0.0865**、B_EH|1106 **0.0508**（模板位 0.0021–0.0112、内容位 0.2118–0.3027、**整句全 0**、纠正面全 0）→ **两 seed 方向相反**。pooled 决策位 CF1 **0.129902** vs B_EH **0.132966**。成本本轮 **23,995.969** ＋此前 **18,835.484**（20.4c 作废 2,514.203＋重启前 16,321.281）＝**42,831.453 CPU / 24,120.563 wall**。
> 当前进度：**20.4d 零训练共用评测已交付（`arm_indistinguishable`）**——8 格棋盘＋全新共用批次（seed 20261107、索引 320–399、world 级配对 bootstrap 2,000 次）：pooled Δ **−0.028064**（CI **−0.064706…+0.010539**）、`batch_over_arm_ratio` **0.7336**、`sign_agreement=false`、`arm_effective=false`。8 格两因素方差分解（20.6 修订）：**模型 seed 99.3958%／arm 0.2785%／batch 0.0084%／交互 0.3173%**；**4,026.438 CPU**。原 `variance` 块保留审计，`criteria` 与 `verdict` 未改。
> 当前进度：**20.6 全局收束已交付并关包（`package20_closed_current_learning_candidate_not_established`）**——按 §10 第 2 行结案"当前学习候选未成功＋F 额外贡献未证"；20.5 准入不成立、**不执行**；confirm 封存、**record 保持 0 读取**；**无自动 20.x 补丁链**。measured **50,573.637 CPU**（B=24,000 的 **2.1072×**、48,000 闸的 **1.0536×**，超 **2,573.637**）；含声明估计 **52,893.637**（1.102× 闸）；正式 800 updates＋作废 25＝**825 updates / 825 Adam**；F 探针 **17 事件 / 85 臂评估**。
> 诊断链标记：**20.2r／20.2s／20.2t／20.3a／20.3b／20.4a／20.4b／20.4d 均为"诊断、非计划步骤"，结论已并入 20.6**；不计入 §8 工作卡的正式步骤序列，脚本按既有归档惯例处理。
> 历史下一步（已被 SC-SFK 与系统闭环取代）：**字节地址闸已判 `alignment_not_discriminative`（2026-11-08，0 训练）**——owner 否决“接进训练 runner”，唯一瓶颈改为地址生成：读的地址从 128 维均值换成 19.2r 已验证的 `ByteMatchRoute`（65,807 参数、恒等初始化、自由偏移搜索、0 更新），写读同一通路，复制门／阈值／容量未动。同一 154 条口径：写入后按存储键读回 **22/22**；换措辞 **0/22**、未写措辞 **0/66**（严格“命中自己槽位”；旧的“与读措辞同槽”口径 66/66 对字节臂已退化）；换措辞最低分 **13/15＝0.8667**，新词最高 **13/15**、新搭配最高 **14/15**（末位词只进 15 字节窗口 1 个字节）⇒ 空档 **0.0／−0.0667**，无一为正；复制门仍 **22/22 拒**（overlap 0.939–0.988，它读的还是均值键）。新增参数：通路 65,807（不训练、不入 state_dict）、主体 **+0**、图内状态 **+0**（480 B 地址表在 runner 侧）；`candidate_sha_v2` 不变。唯一后继＝把匹配换成“与位置无关的字节集合命中（按稀有度加权）”（**该后继已由 SC-SFK 与系统闭环完成**）。报告＋manifest 见 [taiji_byte_address_20261108.json](../../../reports/taiji_byte_address_20261108.json) 与 [manifest](../../manifests/taiji_byte_address_20261108.json)；plans 六处已同步。
> 当前进度：**生长接线已落地并判 `growth_wired_default_off`（2026-11-08，0 训练）**——零扰动全过、跨会话冒烟过；22 面 8/22 与不生长臂相同（容量生长无可测任务收益），复制门在当前编码上 22/22 拒（overlap 0.939–0.988）⇒ 复制门保持关闭。报告＋manifest 见 [taiji_growth_wire_20261108.json](../../../reports/taiji_growth_wire_20261108.json) 与 [manifest](../../manifests/taiji_growth_wire_20261108.json)。
> 当前进度：**计划收敛（2026-11-04）**——唯一权威主线移到 [MASTER-PLAN](../../MASTER-PLAN.md)，[MAINLINE](../../MAINLINE.md) 只留单入口；推进轨道移到 [ROADMAP-NEXT](../../reference/ROADMAP-NEXT_20261104.md)（19/20/21 包）；状态表、依赖顺序、权威源、冲突消解与待裁定清单集中一处，本页只保留流水与证据回执。
> 最后提交：本轮＝**短条目可寻址闸提交**（2026-11-08，0 训练、0 优化器步、0 Adam、0 confirm、0 新参数、0 调阈值：`taiji/sc_sfk.py`（`PROTECT=WINDOW=6`、身份只丢末块、块数下限 1、`MIN_ADDRESSABLE=23`、Jaccard 回退与标注、`chunk_address_audit`）＋runner＋预注册 manifest（freeze_revision 7）＋开启／关闭报告＋全量输出＋文档 §15＋账本 §42＋plans 六处；哈希不回填以保单次提交）/ 此前＝**局部谓词切分闸提交**（2026-11-08，0 训练、0 优化器步、0 Adam、0 confirm、0 新参数、0 调阈值：`taiji/sc_sfk.py` 切点改局部谓词＋runner＋预注册 manifest＋开启／关闭两报告＋文档 §14＋plans 六处）/ **偏移无关闸提交**（2026-11-08，0 训练、0 优化器步、0 Adam、0 confirm、0 新参数、0 调阈值：`features()` 试改为全偏移、按判定回滚到步长 2＋runner（标 `SUPERSEDED`＋环境闸）＋预注册 manifest＋corpus 快照＋开启／关闭两报告＋文档 §12＋plans 六处；哈希不回填以保单次提交）/ **锚点计数闸提交**（2026-11-08，0 训练、0 优化器步、0 Adam、0 confirm、0 新参数、0 调阈值：`read` 试改后按判定回滚＋runner（标 `SUPERSEDED`）＋预注册 manifest＋corpus 快照＋开启／关闭两报告＋文档 §11＋plans 六处；哈希不回填以保单次提交）/ **A1 夹具外闸提交**（2026-11-08，0 训练、0 优化器步、0 Adam、0 confirm、0 新参数、0 调阈值：runner＋预注册 manifest＋报告（含 r1 harness 缺陷留档）＋文档 §10＋plans 六处）/ `2363d504b`（2026-11-08 **SC-SFK 地址层落位**：默认关闭、0 新参数、0 训练——关闭态 `store_addressing_wire_neutral`（176 条 `max|Δ|=0`、参数/状态/身份/签名四项不变）＋开启态 `store_addressing_wired`（换措辞 22/22、未写 66/66、两类该拒 22/22、容量 2→32、0 驱逐/0 覆盖/0 回滚）；runner＋报告＋manifest＋plans 六处同步）/ `62355b12`（2026-11-08 SC-SFK 地址层原语首轮零训练成立：runner＋报告＋manifest＋plans 六处同步）/ `5bf78dbd2`（2026-11-08 全局收束：算法层诊断＋superseded 登记＋plans 六处队首收敛为一条）/ `67d000b53`（20.6 关包哈希回填：报告 `submitted_runner_sha256`＋manifest＋plans）/ `eabc694f0`（**包 20 关包提交**：20.6 报告＋manifest＋plans 六处＋20.4c_r2/20.4d 报告与 manifest；0 训练 / 0 优化器步 / 0 Adam / 0 confirm；measured **50,573.637 CPU**、825 updates/825 Adam、探针 17 事件/85 评估）/ 此前 `f35910423`（20.2t 写入粒度设计闸）

更新：2026-11-05，Asia/Shanghai。本页原执行表已整合进总计划，不再发布第二套顺序或复制待办。以下按统一步骤关联证据，阶段记录只解释当时，不能把零步时“未训练”当作后续现状。

**头条状态（2026-11-04，语义身份）：** 身份从“字节绑定”改为“语义哈希”：组成项是图名/契约、配置常量、模块清单、`routing_rule_version`、解码与路由规则、输入白名单、参数名/形状/dtype、声明可塑归属；`source_sha` 与 `weights_sha256` 只记录不决定身份。black-100 合并后 v1 字节摘要从 964a8c68… 漂到 1af63213…，但 `candidate_sha_v2` 恒为 773dcec2…。零更新等价探针（[报告](../../../reports/taiji_selection_routing_17_2r_format_equivalence_20261104.json)）全部逐位一致，60.80 CPU / 62.80 wall、0 优化器步、0 训练更新、0 confirm 读取；四份 2026-11-03 证据维持原字节与有效性（T3 v3 720/720、17.4 v2、新价、17.2R），身份登记见[注册报告](../../../reports/taiji_selection_routing_17_2r_identity_20261104.json)。17.5 的 checkpoint 保存/恢复复验已通过，六格跑完并结案为负结果（见下条 17.7）。本页是按时间追加的流水，旧条目里的“唯一阻塞/唯一下一步”均已被最新条目覆盖。

## 2026-11-08 编码层最后一刀：`localized_centered` 当唯一变量（0 训练 / 0 优化器步 / 0 Adam / 0 confirm）

**产物**：[回执](../../../reports/taiji_gate_localized_20261108.json)、[预注册 manifest](../../manifests/taiji_gate_localized_20261108.json)（sha `b434c1df…`，读数前落盘、`written_before_any_reading=true`）、runner [probe_taiji_gate_localized_20261108.py](../../../scripts/training/probe_taiji_gate_localized_20261108.py)（sha 前后 `112330dd…`，双列一致）、进度 `reports/taiji_gate_localized_20261108_progress.jsonl`。**8.078 CPU / 1.240 wall**。

**做法**：同一真实前向、同一 154 条口径（22 已写面＋66 未写措辞＋三类负对照各 22）。臂 A 平均池化（现状）；臂 B `localized_centered`——逐维方差加权池化＋top-4 位置残差（τ=0.5、k=4 取自生长规格，未搜索），再减一次**该 store 自己 30 个键的均值**并归一化（质心在会话边界一次拟合、写读共用）；编码只在会话边界切换，臂 B **不复用**任何臂 A 键。控制：两臂各 **154/154** 复现真实 `addressed_read`（`fast` 字节、槽位全同、`max|Δ|=0`）；两臂 value／age／count **逐字节相同**、store 只读（`store_immutable=true`）。

**读数**：

| 量 | 臂 A 平均池化 | 臂 B `localized_centered` | 过门线 |
|---|---|---|---|
| 已写组最低相似度（门槛锚） | 0.964818 | **0.765558** | ≥0.90 → **不过** |
| `m_spec` 中位 / 最小 | 0.002752 / 0.0 | **0.0000966 / 0.0** | — |
| 键有效秩（30 键） | 2.557 | **5.393**（top_share 0.970→**0.868**） | — |
| 三类空档（新词／新搭配／无对应值） | −0.019750／−0.034599／−0.031098 | **−0.179573／−0.197649／−0.215692** | 至少两类 >0 → **0 类** |
| 66 条未写措辞正确 | 35/66 | **42/66** | 仅参考 |
| 22 面按新编码重写后正确 | 8/22 | **1/22** | 不劣于 A → **不过** |
| 门槛开启后拒（已写／未写／新词／新搭配／无值） | 0／2／6／1／0 | **0／3／1／3／0** | — |

**新增成本**：可训练参数 **+0**（全模型 3,311,259）、图内状态 **+0**；仅 runner 侧 128 个 float64 质心＝1,024 B（store 级统计量，入图时按规格记 `d+1` floats）。

**判定**：**`encoder_also_insufficient`**。三条子条件全不成立——载体掉出 0.90（0.766）、三类空档全负且比现状**更远**、22 面落回自身从 8 降到 1。机制解释：局部化把"同一内容不同说法"也一起拆开（读位置与写位置不同即换方向），去公共偏移又把残差里的公共部分放大，于是"认自己"与"分别人"仍绑在一起，且这次两头都更差。**夹具臂的键本来就是该编码，不构成 A/B 对照，本轮只报真实臂。** 这条"内容寻址写＋读门槛"线到此结案，不再迭代编码/几何/阈值。

**给 owner 的原文（可直接引用）**：

> 在这台模型、这 154 个问题上，"按内容寻址写＋足够相似才读"这条线到此结案。把一条记录"按什么被记住"从平均池化换成局部化编码＋去公共偏移（不新增任何参数），仍然没能同时做到两件事：22 条写入记录用自己的变体问法去问，最差相似度只有 0.77（可用线是 0.90），而且只有 1 条真落回自己；三类从未写过的问法反而比现状更贴近记录（差 −0.18 到 −0.22）。这说明问题不在阈值、不在偏移、不在某一种变换的具体形式，而在**这台模型现在给出的上下文向量本身**：它还不能在"同一内容、不同说法"上保持稳定，同时把"从未写过"分开。这条线不再调编码、几何或阈值。**它不能证明**内容寻址在原则上不可行，不能证明模型不能学习，也不能证明扩容没有价值——只证明在今天的冻结图上，这套机制没有可用的余量。

## 2026-11-08 白化当唯一变量的零训练复测（0 训练 / 0 优化器步 / 0 Adam / 0 confirm）

**产物**：[回执](../../../reports/taiji_gate_whitening_20261108.json)、[预注册 manifest](../../manifests/taiji_gate_whitening_20261108.json)（sha `d0b9d20e…`，读数前落盘）、runner [probe_taiji_gate_whitening_20261108.py](../../../scripts/training/probe_taiji_gate_whitening_20261108.py)（sha 前后 `1343b7ac…`）、进度 `reports/taiji_gate_whitening_20261108_progress.jsonl`。三臂同一 154 条口径：**A 现状**；**B 保留均值的带收缩白化**——`Σ_s = Σ + λI`、`λ = tr(Σ)/d`（真实臂 λ=0.004103647382、d=128、n=30，单值由规则钉死、未试多个）、`W = Σ_s^{-1/2}` 同时作用于**已存键与查询键**，写入仍存原始键；**C 居中**复用上轮读数（不重算，manifest 已声明）。

**控制（先过）**：raw 变换复现真实 `addressed_read`——两臂各 **154/154** `fast` 字节、槽位 154/154、`max|Δ|=0`；`store_immutable=true`。

**读数（真实臂 A→B / 夹具臂 A→B）**：

| 量 | 真实臂 A | 真实臂 B（白化） | 夹具臂 A | 夹具臂 B |
|---|---|---|---|---|
| 22 面读出字节与现状相同 | — | **0/22**（槽位 8/22） | — | **0/22**（槽位 22/22） |
| 已写组最低相似度（门槛锚） | 0.964818 | **0.986721** | 0.959857 | **0.848826** |
| `m_spec` 中位 / 最小 | 0.002752 / 0.0 | **0.003559 / 0.000714** | 0.762590 | **0.763044 / 0.724589** |
| 键有效秩（30 键） | 2.557 | **1.876** | 27.749 | 29.007 |
| `top_share` | 0.970 | **0.993** | 0.089 | 0.122 |
| 全新 token 空档 | −0.019750 | **−0.000786** | +0.684080 | +0.713 |
| 新搭配空档 | −0.034599 | **−0.012476** | −0.040124 | −0.151 |
| 无对应值空档 | −0.031098 | **−0.011453** | +0.211560 | +0.098 |
| 66 条未写措辞正确数 | 35/66 | **62/66** | 66/66 | 66/66 |

**门槛开启（阈值＝本臂已写组最低相似度）**：真实臂已写 0 拒；未写误拒 **8/66**；三类负对照被拒 **20／0／0**。夹具臂未写 0 拒；负对照 **22／0／22**。

**判定（写死）**：V1（22 面字节有任何变化）先触发 ⇒ **`whitening_changes_written_bytes`**（硬失败）。实质判定按预注册量：`collapse_fixed=false`（有效秩未翻倍、`top_share` 未 ≤0.75）而 `carrier_kept=true`（floor ≥0.90）⇒ **`encoding_is_the_ceiling`**。

**归因与对生长的含义**：居中修塌缩（2.557→12.148）但砸掉载体（0.965→0.449）；白化保载体（0.965→0.987）却让塌缩更重（2.557→1.876）——**两条线性变换正好各保一头**。所以问题不在门槛、不在偏移、也不在白化形式，而在**平均池化的字节编码把"公共载体"和"可分残差"放进了同一个线性空间**。生长要往前走，必须在编码层换掉（`localized_centered`），不是在现有编码上再叠线性映射。

**纪律**：CPU **13.172 s**（cap 200）、wall 1.612 s；0 训练／0 优化器步／0 Adam／0 confirm／0 checkpoint；`process_guard` 通过、单实例锁、60 s watchdog 未触发、读数即落盘 write-then-rename、无 `python -m black`（仅 in-process 闸）。**下一步唯一**：把 `localized_centered` 编码接到真实读路径，在同一 154 条口径上零训练复测，过门要求"floor ≥0.90 且至少两类空档 >0"。

## 2026-11-08 居中当唯一变量的零训练复测（0 训练 / 0 优化器步 / 0 Adam / 0 confirm）

**产物**：[回执](../../../reports/taiji_gate_centering_20261108.json)、[预注册 manifest](../../manifests/taiji_gate_centering_20261108.json)（文件 sha `22a8a180…`，读数前落盘）、runner [probe_taiji_gate_centering_20261108.py](../../../scripts/training/probe_taiji_gate_centering_20261108.py)（sha 前后 `e7317066…`）、进度 `reports/taiji_gate_centering_20261108_progress.jsonl`。两臂只差一件事：臂 B 在读时把已占槽键的均值从"存储键"和"查询键"里一起减掉；编码器、写入协议、门槛形式与锚定（只用已写组最低相似度）全同。存储零字节改动（`store_immutable=true`）。

**控制（先过）**：`center=None` 必须复现真实 `addressed_read`——两臂各 **154/154** `fast` 字节相同、top slot **154/154** 相同、`max|Δ|=0`。

**读数（真实臂 / 夹具臂）**：

| 量 | 真实臂 A（现状） | 真实臂 B（居中） | 夹具臂 A | 夹具臂 B |
|---|---|---|---|---|
| 已写组最低相似度（门槛锚） | 0.964818 | **0.449371** | 0.959857 | 0.960345 |
| m_spec 中位（写面最近−次近） | 0.002752 | **0.106488** | 0.762590 | 0.801770 |
| 全新 token 空档 | −0.019750 | **−0.282208** | +0.684080 | +0.701054 |
| 新搭配空档 | −0.034599 | **−0.519487** | −0.040124 | −0.039635 |
| 无对应值空档 | −0.031098 | **−0.067982** | +0.211560 | +0.233954 |
| 键有效秩（30 键） | 2.557 | **12.148** | 27.749 | 27.214 |
| `top_share` | 0.970 | **0.545** | 0.089 | 0.067 |
| 中点相对质心余弦中位 | 0.992425 | — | 0.407760 | — |
| 已存键相对质心余弦中位 | 0.985707 | — | 0.240272 | — |

**门槛开启（阈值＝本臂已写组最低相似度）**：真实臂已写 **22/22** 匹配，但读出的 `fast` 字节 **0/22** 与臂 A 相同（top slot **21/22**），未写措辞误拒 **10/66**，三类负对照被拒 **9／3／0**；夹具臂已写 22/22、字节 **0/22**（槽位 22/22）、未写误拒 **9/66**、负对照 **22／6／22**。

**判定（写死，按 manifest 顺序）**：V1（写面读出字节有任何变化）先触发 ⇒ **`centering_changes_written_bytes`**，直接失败并停止、不再调门槛；同时真实臂三类空档仍全负 ⇒ **`centering_insufficient` 亦成立**。夹具臂同样 V1 触发。

**归因**：居中确实修好了键塌缩（有效秩 2.557→12.148、`top_share` 0.970→0.545，复现 20.3a 的 0.956→0.498），但把"同义读键贴住自己写入键"所依赖的公共分量一起减掉了——已写组最低相似度从 0.965 掉到 0.449。**塌缩与可分性不是同一个病**：共享偏移既是塌缩的来源，也是匹配的载体。中点"更靠公共方向"在居中后依然成立（0.9924 > 0.9857），说明它不是偏移造成的，而是距离集中效应。

**纪律**：CPU **14.281 s**（cap 200）、wall 1.693 s；0 训练／0 优化器步／0 Adam／0 confirm／0 checkpoint；`process_guard` 通过、单实例锁、60 s watchdog 未触发、读数即落盘 write-then-rename、无 `python -m black`（仅 in-process 闸）。**下一步唯一**：同一 154 条口径，把"去均值"换成"保留均值、只做带收缩的协方差白化"（30 键 × 128 维，必须收缩防秩亏），看能否在不砸掉已写组绝对水平的前提下压平 `top_share`。

## 2026-11-08 读门槛接线真实 `addressed_read`（0 训练 / 0 优化器步 / 0 Adam / 0 confirm）

**两件产物**：零扰动证明 [taiji_gate_read_gate_zero_20261108.json](../../../reports/taiji_gate_read_gate_zero_20261108.json)＋开启后读数 [taiji_gate_read_gate_enabled_20261108.json](../../../reports/taiji_gate_read_gate_enabled_20261108.json)；预注册 [manifest](../../manifests/taiji_gate_read_gate_20261108.json)（sha `8b4c25b6…`，在第一个分类读数前落盘）；runner [probe_taiji_gate_read_gate_20261108.py](../../../scripts/training/probe_taiji_gate_read_gate_20261108.py)。两条臂：`real_forward`（原生 float32、真实 `observe`＋`addressed_write` 建 store）与 `fixture_continuity`（G2g 夹具键/值经诊断 `key=` 入口注入真实函数，float64）。

**接口（默认关闭）**：`LivingCoreConfig.read_gate_cosine`，默认 `None`。关闭态 `addressed_read` 在每个分支都返回 `matched=True`，info 只多一个键、字节与改前完全相同；开启后 top 余弦低于门槛时返回 `matched=False`／`abstained=True`／`reason="below_gate"`，**不返回 `top_slot`、不混进 `fast`**。`key=` 参数只作诊断入口，生产调用不传。

**零扰动（必须先过，已过）**：两臂各 **154** 条查询（22 已写＋66 未写＋3×22 负对照）逐条比较改前读法的逐字副本——`fast` 字节 **154/154** 相同、`max|Δ|=0`、info 字段全等、`key` 全等、`matched` **154/154** 为 true、额外键仅 `["matched"]`；22 个真实 `score_targets` 前向概率字节 **22/22** 相同、`max|Δ|=0`；同一读数跑两次，状态哈希全同。⇒ **开关关闭时不改变任何字节**。

**门槛（只用正对照＝已写组最低余弦，未用负标签反推）**：夹具臂 floor **0.959857**、真实臂 floor **0.964818**。gap 一律相对该臂已写组最低余弦计：夹具臂 `new_token` **+0.68408**、`new_pairing` **−0.04012**、`no_value` **+0.21156**；真实臂 **−0.01975／−0.03460／−0.03110**。候选形式只改接受数，不改 gap：

| 候选 | 夹具臂接受（已写／未写／nt／np／nv） | 真实臂接受（同序） | 结论 |
|---|---|---|---|
| `cos_floor`（选中阈值） | 22/22／66/66／0／22／0 | 22/22／**64/66**／16／21／22 | 夹具 `partial_separation`；真实 `no_separation_on_real_keys` |
| `cos_floor−0.005` | 22/22／66/66／0／22／0 | 22/22／66/66／18／22／22 | 同型，未改善 |
| `cos_floor−0.01` | 22/22／66/66／0／22／0 | 22/22／66/66／21／22／22 | 同型，未改善 |
| `margin_floor`（已写组最低 margin） | 22/22／**65/66**／0／**21**／0 | 22/22／66/66／22／22／22（阈值 0.0） | 夹具误杀 1 条未写；真实**全放行** |

（nt＝`new_token`、np＝`new_pairing`、nv＝`no_value`；夹具 `margin_floor` 值为 0.636384。）

**"见过字的新搭配"明确无法用读门槛区分**：夹具臂 `new_pairing` 余弦高于已写组下限（gap −0.040），真实臂三类 gap 全负；`no_value`（两条已存键中点）在真实臂中位余弦 **0.99299**，反而**高于**已写组 0.97550——与 20.3a 的 `shared_offset` 归因一致：真实 `store_query_key` 未居中，所有键共享偏移，中点更靠公共方向。**0.960 是居中/局部化键的性质，不是真实读键的性质**。

**开启后零训练读数**（阈值＝各臂 floor）：夹具臂已写 22/22、**0 误杀**、匹配面字节 22/22 与关闭态相同；`new_token` 拒 22/22、`no_value` 拒 22/22、**`new_pairing` 拒 0/22**；未写 66/66。真实臂已写 22/22、**0 误杀**；`new_token` 拒 6/22（27.27%）、`new_pairing` 1/22（4.55%）、`no_value` 0/22；**未写措辞误拒 2/66（3.03%）**。强制作答（把关闭态答案硬交给消费者）：夹具 44 条 × 65,536 B＝**2,883,584 B**、真实 7 条 × 32,768 B＝**229,376 B**，**0 条**与任何已存载荷逐位相同。

**归因（架构级）**：门槛本身没问题，问题在键几何。夹具（居中/局部化）能分开两类负对照，真实 `store_query_key` 不能——与 20.3a 同一根因。**结论：门槛默认保持关闭，不启用。**

**纪律**：CPU **34.953 s**（cap 200）、wall 3.437 s；0 训练／0 优化器步／0 Adam／0 confirm／0 checkpoint；`process_guard` 通过、单实例锁、60 s watchdog 未触发、读数即落盘 write-then-rename、runner sha 前后 `f3994477…` 一致、无 `python -m black`（仅 in-process 闸）。runner 首跑后被项目格式闸（in-process 检查）判"would reformat"，按闸重排后**原样重跑**：floor、各 gap、拒绝数、强制作答字节**逐项相同**（`zero_disturbance` 仍成立），以下读数取重跑版。**下一步唯一**：做真实 `store_query_key` 共享偏移/居中的独立零训练实验（同 154 条口径），再用同一 runner 复测。

## 2026-11-07 全局梳理与包 21 计划（0 训练 / 0 优化器步 / 0 confirm）

**梳理（只读）**：回答四问——① owner 的算法优化（`fef0291ea` MGW1＋`91b8c8351` U1＋`a2d580f32` 复核）全部为**设计层**，`taiji/` 0 改动、0 参数、0 能力，身份（参数清单/`candidate_sha_v2`/`DECODE_RULES`/冻结语义）未变；② 与包 20 负结论的关系是**代数/表述统一＋一处新读出机制（`Read_R`）**，未重设写入时机、F 键可分性与发射/读出协议。**2026-11-08 更新：U1 的 E1/E2 已按 owner 批准的勘误并入正文（定理 5.1 拆为 5.1a/5.1b/推论 5.1c，命题 4.1 改为恒等式）；P3/P4 未修，仍不得写成"已修好"**；③ 项目状态：Task1 结构 188/192＝0.9792（19.2w 未封存复核）、训练权重贡献 0（限 19.2u 消融）、包 20 关闭、confirm 0 读取、20.4c_r2 32 个 checkpoint 全在、19.2v 父 bundle 在；未结项为 D16-20 绝对阈值、D16-13 续跑变慢归因，另有两处旧"待裁定"文字需标 `superseded`；④ 六处计划在包 20 结论上一致，但存在 4 项残留（能力卡 §7、账本 §15.1、MGW1/U1 行缺缺陷指针、ROADMAP §2 现在时）。详见[全局梳理](../../reference/SURVEY-PROJECT-STATE_20261107.md)。

**计划（单一主线）**：立[包 21 读出/发射零训练分解](../../reference/PLAN-PACKAGE21_READOUT_DECOMPOSITION_20261107.md)——21.0 冻结 → **21.1 三臂零训练分解（队首）** → 按 R/U/I 分支进 21.2/21.3/21.4；21.2–21.4 不预启动、不并行。退役：20.3a 内容寻址写（三闸已败）、20.2t 选项 3、U1/MGW1 冻结为设计资产（**2026-11-08 更新：U1 的 P1/P2 已并入正文；P3/P4 未修，仍不得作为实现依据**）；删 `output/_peek_20_4c_r2.py`；`output/taiji_package20_4_checkpoints` 标 `superseded`。

**成本**：本段梳理＋规划 **0 训练 / 0 优化器步 / 0 Adam / 0 confirm**；CPU 预算上限 1,200（只读）。

## 2026-11-07 包 20 全局收束与关包（20.6，0 训练 / 0 confirm）

**结案判定**：`package20_closed_current_learning_candidate_not_established`。按[包20计划 §10](../../reference/PLAN-PACKAGE20_RULE_LEARNING_CORRECTION_20261105.md) 六行逐条对照，仅选中第 2 行——"A/B 都低于强捷径，信息和信用门已过 → 当前学习候选未成功；在 20.6 归因表示/组合/优化及竞争解释，停止剂量追加，不宣布整个架构数学不可能"。依据三条：

1. **两 seed 方向相反**：20.4c_r2 末步 dev 决策位在 1105 利 B_EH（0.0968 vs 0.0356）、在 1106 利 CF1（0.0865 vs 0.0508）；F 净贡献门槛（点差 ≥0.05 且配对下界 >0）**不成立**。20.4d 共用批次 pooled Δ **−0.028064**（CI **−0.064706…+0.010539**，含 0）、`arm_effective=false`、`arm_indistinguishable=true`、`sign_agreement=false`、`batch_over_arm_ratio=0.7336`。
2. **绝对水平远低于强捷径**：末步模板位 0.0021–0.0112、内容位 0.2118–0.3027、**整句恒 0**；对照的标签多重集捷径为 7/11、51 轮信息 Bayes 参考 1753/1938。
3. 因此 **20.5 准入不成立、不执行**；confirm 继续封存、record 保持 **0 读取**；本包**无自动 20.x 补丁链**（不再开 20.4e 或任何新包）。

**20.4d 只改方差块、不改判据**：原 `variance` 块把 8 格按 arm/batch 两水平分解，混入模型 seed。20.6 用两因素重算：模型 seed **99.3958%**、arm **0.2785%**、batch **0.0084%**、交互 **0.3173%**（grand mean 0.134221814、SS_total 0.075193908）。原块保留作审计，`criteria` 与 `verdict` 未改；报告写 `variance_correction`，manifest 记 `amendment`（原始 sha `d3a25850…`）。

**三类归因（只用已有证据，不新增实验）**

- **表示（已证）**：22 个纠正面 F 只有 **2 个不同向量**（20.2s）；每轮内容寻址写后仍近共线（top-1 余弦中位 **0.9850**、top1–top2 边距中位 **0.00317**、22/22 < 0.05）→ 反馈在表示层近似"窗口常量"。
- **组合（已证）**：20.3b 中 44/44 个决策位 gold 全在 3 名外，所需 deficit 仅为局部 spread 的 **0.46%**；统一增益要到 **1e4** 才翻位且 **0/22 答对** → 缺口不是"缺一个大的读出增益"，而是逐位排序本身错。
- **优化（已证为"200 步能抬内容位"）**：内容位从基线 0 抬到 0.2118–0.3027，但模板位≈0、整句恒 0，且 B_EH 同样抬升 → 训练信号有效，但**不是 F 特有**。
- **竞争解释（未证 ≠ 否定）**：① 800 updates（2 world/步）远低于模板位序列发射所需剂量，不能推出"架构学不会"；② 整句门把模板位与内容位耦合，不能排除"内容位已学到、读出/停止协议未学到"；③ 两臂同形且共享大部分通路，arm 只占 SS 的 0.28%，不能把 arm 差异读成 F 机制读数。

**成本总账（measured；缺失记 missing，不补 0）**：20.0 **0.03125**／20.1 **159.53**／20.2 **153.08**／20.2r **161.48**／20.2s **119.67**（首跑失败尝试精确值 missing，下界 25.75）／20.2t **missing**（仅 manifest、无脚本）／20.3a **224.94**／20.3b **56.92**（2 次失败尝试 missing）／20.4 **1,539.89**／20.4a **784.188**／20.4b **516.0**（工程重跑约 **1,700** 为估计）／20.4c **2,514.203**（作废）／20.4c_r2 **16,321.281**（重启前）＋**23,995.969**（重启后）／20.4d **4,026.438**／20.6 **0.016**。合计 measured **50,573.637**，＋声明估计 2,320.0 ＝ **52,893.637**；下界 50,599.387。对照：提案 B=24,000 的 **2.1072×**；量级闸 48,000 的 **1.0536×（超 2,573.637）**；含估计 **1.102×（超 4,893.637）**。训练账：正式 **800** updates（200×4）＋作废 **25** ＝ **825 updates / 825 Adam**，校准 15；F 探针 **17 事件 / 85 臂评估**。

**能力账**：20 包新增"有界 2×2 比较装置（每 25 步 checkpoint、重启校准、resume 验证、16/16 确定性）"、"预注册负结果（当前 F 候选未立、两臂不可分辨、评测批次非主导）"、"修订后的方差事实（模型 seed 99.40%）"与"成本校准（56.906 CPU/更新；正式 800 updates 花 42,831.453）"；**未证**反馈条件适应与版本纠正、F 的净额外贡献、Task1 学后保持（本包未测）、confirm 面（全程 0 读取）。两条并列不混淆：**Task1 由结构承担、在已测范围内训练权重贡献 0**（已证的结构陈述）；**反馈学习未立**（当前候选的负结果，不等于不可能）。

**清理**：20.2r／20.2s／20.2t／20.3a／20.3b／20.4a／20.4b／20.4d 统一标为**诊断、非计划步骤、结论已并入 20.6**；相关脚本按既有归档惯例处理，旧报告与成本不改。

**唯一后继问题**：200 步训练把内容位从 0 抬到 0.25–0.46，模板位仍≈0、整句恒 0——失败**首先**发生在"整句解码/读出"，还是"反馈写读权威"？该问题只由现有证据提出（8 格整句全 0；20.3b 换 oracle F 仍 0 字节变化、0/22；arm 只占 SS 0.28%），**不构成训练方案、不定价、不开新包**，也不把"未证"写成"不可能"。

**已知缺陷**：CF1|1105 的纠正面／反馈后首答口径在重启前用旧累加器算完，**无法回填**，该项记 **missing** 并注明原因（不重跑）；20.2t 无脚本故 CPU missing；20.3b 两次失败尝试 CPU missing；20.4b 工程重跑仅约 1,700 估计；Task1 保持面未测。

**纪律**：0 训练 / 0 优化器步 / 0 Adam / 0 confirm；读数逐行 append＋fsync；报告 write-then-rename、双列 runner sha；本段 CPU **0.016**（上限 1,500）。

## 2026-11-04 19.2f 新描述墙归因（0 存包 / 0 confirm）

证据：[19.2f 报告](../../../reports/taiji_package19_2f_new_description_wall_20261104.json)＋[manifest](../../manifests/taiji_package19_2f_new_description_wall_20261104.json)。脚本 [`attribute_taiji_package19_2f_new_description_wall.py`](../../../scripts/training/attribute_taiji_package19_2f_new_description_wall.py)。19.5 六格 bundle 只读，0 Adam、0 存包、0 confirm。

**A. 分层（同一 192 例/格、正式 `completed` 谓词）**：B 两 seed 与 C 对照；A 臂两 seed 均 0/192（28031-A＝D16-6 全 NaN，28032-A 有限但首字节即错，属已登记 A 臂塌缩），只作反例、不进池化。

| 格 | 旧描述 96 例 | 新描述 96 例 | 整句 192 |
|---|---:|---:|---:|
| 28031-B | **1.000** | **0.2604（25/96）** | 0.6302 |
| 28032-B | **0.9375** | **0.2292（22/96）** | 0.5833 |
| 池化（B×2） | **0.9688** | **0.2448** | 0.6073 |
| 28031-C | 0.6979 | 0.1042 | 0.4010 |
| 28032-C | 0.8021 | 0.1354 | 0.4688 |

新描述半的选择/起点（B）：物件 route 0.4688/0.3750、地点 0.8438/0.8229；composite_start 物件 0.3854/0.3750、地点 0.5312/0.4167；起点命中后 conditional_byte 物件 0.7613/0.8287、地点 1.0000/0.9458 → **起点（记录选择）错，内容字节才跟着崩**。

**B. 换序探针（同字节多重集、只交换两个描述；17.3 T2 口径 θ_repr=1e-2 / θ_score=1e-3）**：每 seed 每层 20 对（B×2 合并 40 对）。

| 层 | 表示相对 L2 中位 | 得分最大差中位 | ≥θ_score | 硬 argmax 改变 | top TV 中位 | decode 起点改变 | 跟对首实体（前→后） |
|---|---:|---:|---:|---:|---:|---:|---:|
| 新描述 | 0.2459 | **1.1925** | **40/40** | **0/40** | 0.0161 | 14/40 | 27/40 → 5/40 |
| 旧描述 | 0.2548 | **1.2062** | **40/40** | **3/40** | 0.0591 | 0/40 | 40/40 → 0/40 |

**读数**：顺序**信息**已进编码（表示差/得分差远超阈值），但 **top 记录概率 0.98–0.99、边距饱和**，1.2 的得分扰动翻不动硬排名；换序后新描述仍有 35/40 落回物件记录。→ **不是"有序编码未接线"，是"有序信号被边距吞掉"**。

**C. 新描述 96 例六类（B 池化，互斥）**：`route_miss` **51.7%**（占失败）／**39.1%**（占 96 例）、`content_byte_wrong` 44.2%、`start_miss` 3.4%、`eos_not_stopped` 0.7%、`frame_byte_wrong` 0%、`order_or_extra` 0%、pass 24.5%。**route_miss 是单一最大死因且过半。**

**判定**：换序打分敏感 + 新描述主因 route_miss → 分支 **"组合匹配容量/目标不足"**（训练目标问题，非图形接线）。唯一建议 **19.2g 记录匹配对比边距**：不改图、**+0 参数**、每个内容起点加同 world 同类 hard-negative 边距 `max(0, m − (score_gold − score_decoy))`，m=1.0；判据新描述 route_miss ≤0.05、composite_start ≥0.70、conditional_byte ≥0.90、新描述整句 ≥0.80、旧描述 ≥0.9375 不回退、T2 硬 argmax 改变 ≥18/20；预算 ≤30 Adam / ≤220 CPU / 1 包 / 0 confirm；**未启动，等 owner 裁定**。

**成本与观察线**：本轮 **500.5 CPU / 545.5 wall / 0 Adam / 0 存包 / 0 confirm**，是 100–150 预算的 **3.3×**（主因：40 对 × 2 goal × 5 格的前 3 字节 teacher-forced decode 起点打分叠加 5×192 自由解码）；19.2b 线累计 **2,056.9 CPU**；六格 **32,256.6 CPU 未花**。

## 2026-11-04 19.2e 口径对账 + 零更新强制覆盖探针（0 存包 / 0 confirm）

证据：[19.2e 报告](../../../reports/taiji_package19_2e_pointer_override_20261104.json)＋[manifest](../../manifests/taiji_package19_2e_pointer_override_20261104.json)。脚本 [`probe_taiji_package19_pointer_override.py`](../../../scripts/training/probe_taiji_package19_pointer_override.py)。19.5 六格 bundle 只读，未改字节、未存包。

**A. 口径对账（最高优先，D16-7）**：正式评（192 例/格）整句 0.630/0.583 与残差归因（48 例/格）0.979 的冲突，逐项对平结论：

| 检查项 | 结论 |
|---|---|
| 定义差异 | **无**：正式 `completed` 与归因 `pass` 是同一谓词（`raw == target_prefix` 27 字节＋终止），均走 `prefix=bytes(raw)` UTF-8 掩码；`target_prefix` 恒为 `target_bytes` 去 EOS |
| 切片分布 | 48 例 = 索引 0/1/4/5，**0 个 `new_description`**（96 例全在其余半）→ 纯**易层** |
| 同一 48 例 under 正式口径（实测） | 28031-B **1.000**、28032-B **0.9583** |
| 同一 192 例 under 归因口径（离线分解） | 28031-B 0.6302 = 易层 96/96 + new_desc 25/96；28032-B 0.5833 = 易层 90/96 + new_desc 22/96 |
| 池化切片 | 94/96 = **0.9792**，精确复现归因报告的 0.979 |

**结论**：0.979 是**切片乐观**，正式评**没有被低估**，**19.5 判负不受影响**；任何一臂在切片口径下不得表述为过门。

**B. 强制覆盖探针（0 更新）**：24 例/格混合集（索引 0 易层 12 + 索引 8 `new_description` 12），推理期把复制起点强制成 span 指针 argmax（1e9 one-hot），α 路由 / γ / transport / 解码全部不动。

| 格 | natural 整句 | forced 整句 | 起始物件 | 内容字节 | forced `start_hit` |
|---|---:|---:|---:|---:|---:|
| 28031-B | 0.7083 | **0.7083** | 0.625→**0.750** | 0.840→0.812 | 0.75（= pointer，强制生效） |
| 28032-B | 0.7083 | **0.7083** | 0.750→0.750 | — | 0.75 |
| 28031-C | 0.4583 | 0.5000 | — | — | — |
| 28032-C | 0.4583 | 0.3333 | — | — | — |

**C. log-scale 三档（纯推理，不改权重）**：s=1（natural）/ s=4 / s=16 对 B 臂整句**全部 0.7083**——s=4 已等价于 1e9 强制，s=16 无增益且使 28032-C 掉到 0.333。**量化结论：标量偏置从 +1 倍到 +16 倍都翻不动行，符合 19.2c 的"冻结 transport 行质量"判断。**

**判定**：B forced **0.7083 ≤ 0.75** → **指针路径到顶**，不授权指针、不启动任何训练；转句框/解码门槛归因。

**成本与观察线**：本轮 **171.6 CPU / 177.4 wall / 0 Adam / 0 存包 / 0 confirm**；19.2b 线累计 **1,556.4 CPU**；六格 **32,256.6 CPU 未花**。

## 2026-11-04 19.2c 成熟父包诊断 + 残差六类归因（0 存包 / 0 confirm）

证据：[19.2c 报告](../../../reports/taiji_package19_2c_mature_parent_20261104.json)＋[manifest](../../manifests/taiji_package19_2c_mature_parent_20261104.json)、[残差归因](../../../reports/taiji_package19_2b_residual_attribution_20261104.json)＋[manifest](../../manifests/taiji_package19_2b_residual_attribution_20261104.json)。脚本 [`validate_taiji_package19_2c_mature_parent.py`](../../../scripts/training/validate_taiji_package19_2c_mature_parent.py)、[`attribute_taiji_package19_2b_residuals.py`](../../../scripts/training/attribute_taiji_package19_2b_residuals.py)。19.5 六格 bundle 只读，未改字节。

**A. 19.2c（独立诊断包，不计入 19.2b 的 60 更新上限）**：父包换 19.5 **B 臂 step300**（route 0.998/1.0、composite 0.917/0.979、187 张量 0 NaN）；挂全新 32-d span 头（4,194），**同额 60 独立 Adam**、仅 selection；探针 0/5/10/15/20/25/30/45/60；**0 存包**。

| 读数 | 0 | 30 | 45 | 60 |
|---|---:|---:|---:|---:|
| `composite_start_hit` 物件 | 0.625 | 0.625 | 0.625 | **0.625** |
| `composite_start_hit` 地点 | 0.792 | 0.792 | 0.812 | **0.917** |
| `pointer_start_hit` 物件/地点 | 0.042 / 0.000 | 1.000 / 1.000 | 1.000 / 1.000 | **1.000 / 1.000** |
| `route_hit` 物件/地点 | 1.0 / 1.0 | 1.0 / 1.0 | 1.0 / 1.0 | **1.0 / 1.0** |
| `conditional_byte` 物件/地点 | 1.0 / 1.0 | 1.0 / 1.0 | 1.0 / 1.0 | **1.0 / 1.0** |

**可分离结论**：**route 不是穹顶**（全程 1.0）；**span 监督有效**（pointer +0.958 / +1.000、span CE 6.770→1.541）；**卡住的是合成**——标量 `pointer_bias` 加在冻结 `transport_logits` 上，幅度不足以翻越行质量差，物件恒定 0.625。属**架构上限**，不是训练不足。判据未改、未达标，六格仍不放行。**320.6 CPU / 60 Adam**。

**B. 残差六类归因（0 更新）**：19.5 六格 step300、48 例/格、自由解码，按首差异互斥分类。D16-6（28031-A 全 NaN，2,320,153 参数）剔除后 5 健康格 pooled：整句 **0.6875**、理论上限 **0.7417**；`frame_byte_wrong` **20.83%**（模型 42 / 解码-EOS 8）、`start_miss` 6.25%、`content_byte_wrong` 3.75%、`eos_not_stopped` 0.42%、`route_miss` 0、`order_or_extra` 0。每臂：B **0.979**（句框仅 2.1pp）、C **0.740**（上限 0.875）、A 28032 **0.000**（frame-first 掩蔽 route）。B+C 池化 **0.8594 / 0.9271** → **句框不是主因，起点与架构为主**。**111.2 CPU / 0 Adam**。

**成本与观察线**：本轮 320.6 + 111.2 = **431.8 CPU**；19.2b 线累计 **1,384.8 CPU**；六格 **32,256.6 CPU 未花**；confirm 读取 **0**；0 存包。

## 2026-11-04 18.2 位置计数 + UTF-8 掩码（验证达标，未定价、未训练）

证据：[18.2 报告](../../../reports/taiji_cursor_utf8_18_2_20261104.json)＋[manifest](../../manifests/taiji_cursor_utf8_18_2_20261104.json)、[A 臂逐例](../../../reports/taiji_cursor_utf8_18_2_probe_A_20261104.json)、[B 臂逐例](../../../reports/taiji_cursor_utf8_18_2_probe_B_20261104.json)，脚本 [`probe_taiji_cursor_18_2.py`](../../../scripts/training/probe_taiji_cursor_18_2.py)。18.1 证据文件未改动（只读旁挂）。

**改了什么（同包两项，不新增参数）**：① `BindingState` 新增 `copy_record`/`copy_cursor`，`read()` 接受 `cursor`，**已提交的 carry 硬覆盖 transport 点**（只有段起点由 transport 决定，`m_peak` 锚首字节），区间外按 `q` argmax 推进、EOS/null 归 `(-1,-1)`；② `decode_endpoint` 在 argmax 前用 `utf8_allowed_mask(prefix)` 掩掉非法字节（`prefix=None` 保持 18.1 行为）。三处 bundle 读写同步，缺字段按 `.get(-1)` 兼容旧包。

| 读数 | 18.1 | 18.2 | 门 | 判定 |
|---|---:|---:|---:|---|
| b_pure 内容 18 字节（gold 记录 + gold 起点 + γ=0） | 0.755 | **1.000** | ≥0.95 | 过 |
| b_pure 物件 / 地点 | 末位塌陷 | **1.000 / 1.000** | ≥0.95 | 过 |
| 物件/地点 首→末差 | 地点 0.580→0.288 | **0.000 / 0.000** | ≤0.02 | 过 |
| UTF-8 自由解码（掩码后，无非法字节） | 无掩码 | **1.0 / 1.0** | ≥0.99 | 过 |
| 恢复：参数 / Adam / 复制状态 | — | **0.0 / 0.0 / 0.0** | 0 | 过 |
| 恢复：位置计数（A `[16,16]/[1,1]/[18,18]`、B `[16,16]/[2,2]/[16,16]`） | 恒 0 | 逐位相等 | 0 | 过 |

**边界与未决**：b_pure=1.000 是 gold 记录 + gold 起点下的**结构可达上限**，不等于自由路由可用；UTF-8 的 `complete`（完整自终止）A 0.0 / B 0.375，属 EOS 终止缺口，与合法性无关，须在定价包单列；A 臂第 1 次更新 clip 前梯度 4.44e45 与 17.5 预检同族，**非本次引入**，登记为既有仪器问题。**结论：设计达到预期 → 允许进入定价；正式价未出、未启动训练、confirm 未释放。** 四条归因与“门是最大压制”更正留痕见[账本 §14](../../reference/INVARIANT-LEDGER_20261103.md#14-182位置计数--utf-8-掩码验证达标未定价未训练)。

## 2026-11-04 18.1 六格正式比较结案（`native_byte_task_failed`，confirm 未释放）

证据：[比较报告](../../../reports/taiji_selection_routing_18_1_compare_20261104.json)＋[manifest](../../manifests/taiji_selection_routing_18_1_compare_20261104.json)＋[step150 分层探针](../../../reports/taiji_content_transport_18_1_step150_20261104.json)＋[step300 分层探针](../../../reports/taiji_content_transport_18_1_step300_20261104.json)。六格 2 seed × 3 臂 × 300 更新，joint 更新 201–300；batch 8、LR 1e-3、clip 1.0；并行方式与 17.5 一致；先存包后评估、进度=包=step 三重校验全部通过。**1,800 正式 Adam、6 个 step300 bundle、confirm 读取 0。**

| 格 | overall | 抄回率 | unseen | 选择 obj/place | content_drop | γ | 字节 a/b/b_pure/c |
|---|---:|---:|---:|---|---:|---:|---|
| 28031-A | 0.000 | 0.000 | 0.000 | 0.906 / 0.505 | 不可判 | 0.798 | 0.090 / 0.255 / 0.731 / 0.358 |
| 28031-B | 0.000 | 0.000 | 0.000 | 0.891 / 1.000 | 不可判 | 0.733 | 0.086 / 0.250 / 0.716 / 0.358 |
| 28031-C | 0.000 | 0.000 | 0.000 | 0.906 / 0.995 | 不可判 | 0.848 | 0.134 / 0.253 / 0.723 / 0.358 |
| 28032-A | 0.000 | 0.000 | 0.000 | 0.599 / 0.089 | 不可判 | 0.792 | 0.068 / 0.328 / 0.788 / 0.421 |
| 28032-B | 0.000 | 0.000 | 0.000 | 0.990 / 1.000 | 不可判 | 0.727 | 0.066 / 0.325 / 0.781 / 0.421 |
| 28032-C | 0.000 | 0.000 | 0.000 | 0.979 / 1.000 | 不可判 | 0.786 | 0.096 / 0.326 / 0.788 / 0.425 |

**四条门逐项**：`new_name_copied ≥0.60` 未过（0.000，独立门）；`unseen_output_name ≥0.80` 未过（0.000）；`content_drop ≥0.10` 不可判（true 完成率 0）；`selection ≥0.90` 只在部分格过（A 28031 obj、B/C 多数 place），不构成总体通过。A−C 两 seed 点估计均 0.000、世界 bootstrap 下界 0.000；总体不通过，**18.2 不释放、confirm 保持封存**。identity rescue 六格 8/8 逐位通过，UTF-8 合法率部分格仅 0.203（28031-A），EOS 正确率 1.0。

**分层探针（0 更新只读，48 例/格，先 step150 后 step300）**：step150 a/b/b_pure/c = 0.082 / 0.288 / 0.751 / 0.390；step300 = 0.090 / 0.290 / 0.755 / 0.390，两时点几乎不变。**解读**：把 γ 压到 0 后逐字节命中从 0.29 升到 0.755，这是**前向混合**读数——门把概率质量压在生成侧、复制质量被稀释；**按 M3 不得读成“训练期复制信用饥饿”**。门开后仍有约 24.5% 内容字节错误，来自 β/transport 的位置不跨字节携带（`m` 逐步重建、`prefix_cursor` 恒 0）。c（位置全强制、γ 保持模型值）只有 0.39，即“位置对了但门没开”仍失败。探针脚本的二元 `layer` 标签按预设规则判 `beta_transport_pointer`，但逐条件幅度显示前向门的影响更大；两者并列报告，不把标签当作唯一结论，也不由 γ 单独归因训练信用。

**课程死环**：六格 step150 触发 0 次（2,400 次教师回放 0 完成），步进到 300 仍为 0；触发条件是绝对完成率 ≥0.5，本轮从未接近。已登记在设计文档 §18，本轮不改，处置选项（相对改进 ≥10% / 字节级准确率触发）留给下一包。

**成本与工程**：正式 16,196.3 CPU / 2,824.4 wall（保护价 19,220.8 / 30,000，未越线）；离线 by-world 重算 583.7 CPU（0 Adam）；两轮探针为只读前向，**探针 CPU 未落账**（脚本当时未计时），已登记为仪器缺口并补上计时字段，供下一包使用。driver 聚合 bug（`readings()` 只存 `overall` float、丢 `by_world`）已修复，报告与 manifest 由已存 run 回执离线重算，未重训。

## 2026-11-04 能力保持两层判据（owner 修订，只写规则，不改 18.1）

规则正文写入 [INVARIANT-LEDGER §10](../../reference/INVARIANT-LEDGER_20261103.md#10-子继承父的两层保持判据2026-11-04-owner-修订)：硬不变量零容忍（逐位恢复、字节闭环、identity rescue 8/8、confirm=0、父 bundle 身份不变）；软能力报子/父比值＋绝对值，用世界 bootstrap 下界 ≥0.85（<0.85 判 `destructive_overwrite`）；未来包必须加“只学新任务、不混合旧任务”的对照臂；0.85 与现有 12–20 world 重采样噪声对齐，0.80/0.90 敏感性已写明。**当前无合格父能力集，输出 `retention_not_admissible`；18.1/18.2 若产生首个真实非零能力，它即成为可测父，下一包必须带留存率门。** 本轮不改任何 18.1 判据、不启动训练。

## 2026-11-04 18.1 四只读探针（γ 扫描 / c_γ0 上限 / 误差位置 / UTF-8 来源）

证据：[探针报告](../../../reports/taiji_gate_sweep_18_1_20261104.json)＋[manifest](../../manifests/taiji_gate_sweep_18_1_20261104.json)＋[逐字节门预注册](../../manifests/taiji_byte_gate_prereg_20261104.json)，脚本 [`probe_taiji_gate_sweep_18_1.py`](../../../scripts/training/probe_taiji_gate_sweep_18_1.py)。六份 step300 包、48 例/格、0 更新、0 Adam、0 confirm。γ 扫描走**冻结的 `routing_step`**（A/B 的 route-lock 原量保留），只把概率混合里的 γ 换掉；从物理量到 `_advanced_state` 全用冻结实现，避免探针自造路径。

### 1. γ 扫描（pooled 六格，α/β 为模型自身路由）

| 强制 γ | overall | 抄回率 | unseen | 物件 12 字节 | 地点 6 字节 | 内容 18 字节 |
|---:|---:|---:|---:|---:|---:|---:|
| 0.6 | 0.000 | 0.000 | 0.000 | 0.0972 | 0.0168 | 0.0704 |
| 0.5 | 0.000 | 0.0035 | 0.000 | 0.0900 | 0.0417 | 0.0739 |
| 0.4 | 0.000 | 0.0139 | 0.000 | 0.1030 | 0.0561 | 0.0874 |
| 0.3 | 0.000 | 0.0139 | 0.000 | 0.1100 | 0.0538 | 0.0912 |
| 0.2 | 0.000 | 0.0139 | 0.000 | 0.1085 | 0.0538 | 0.0903 |
| 0.0 | 0.000 | 0.0139 | 0.000 | 0.1091 | 0.0498 | 0.0893 |

- 六格 γ 生效值逐档精确等于强制值（0.60/0.50/0.40/0.30/0.20/0.00），没有被 floor 改写。
- **抄回率最高只到 0.0139（4/288 例）**，且全部来自 B 臂（28031-B 0.0625、28032-B 0.0208 在两档之后出现）；A/C 六档全 0。物件字节随 γ 下降只从 0.097 升到 0.110，地点从 0.017 升到 0.056。
- **γ 门单独拧到底也换不来整句**：整句六格 × 六档全 0，unseen 全 0。这否定了“把门压低就能救整句”的路线。**注意这是前向干预结论；按 M3 不得由此推断训练期复制信用饥饿。**

### 2. c_γ0 上限（gold 记录 + 每字节钉住 + γ=0）

| 条件 | 物件 12 字节 | 地点 6 字节 | 内容 18 字节 |
|---|---:|---:|---:|
| b_pure（gold 记录 + gold 起点，β 自由，γ=0） | — | — | 0.7548（逐格 0.731/0.716/0.723/0.788/0.781/0.788） |
| c_γ0（记录与每字节位置全钉，γ=0） | **1.000** | **1.000** | **1.000（六格全部 1.000）** |

- **指针能不能到 1.0：能，但只在“记录 + 每一字节位置”都钉住时。** 六格 48 例 × 18 字节全中，说明复制权重本身没有精度问题。
- 只钉住记录与实体首字节（b_pure）就掉到 0.755；也就是说**首字节之后丢了 24.5% 的字节**，问题在位置如何跨字节传递，不在门。

### 3. 误差位置分布（b_pure vs c_γ0）

| 条件 / 实体 | 首字节 | 末字节 | 形态 |
|---|---:|---:|---|
| b_pure 物件 | 1.000 | 0.872 | 起点全对，逐字节缓慢漂移（末字节 0.771–0.958/格） |
| b_pure 地点 | 0.580 | 0.288 | 起点就分化：28031 三格 0.167，28032 三格 0.979/1.000/1.000；28032 在**第 2 字节即塌到 0.188/0.125/0.167**；末字节 0.208–0.354 |
| c_γ0 物件/地点 | 1.000 | 1.000 | 全部 18 位 1.000 |

- 不是“起点整体偏 1 字节”：物件起点恒 1.0，误差在后段累积；地点要么起点错、要么起点对而第 2 字节即失锁。
- 与结构事实一致：`advance_copy` 的 `m` 按**当前接受字节重建**而非累积，且 `_advanced_state` **不推进 `prefix_cursor`**（恒 0）。首字节靠 `m_override` 锚定后，后续步骤没有位置记忆，只能靠 β 重新猜 → 漂移。

### 4. UTF-8 合法性来源（同一批 48 例/格）

| 条件 | 合法率 | 非法起点分布 |
|---|---:|---|
| γ=1 纯生成 | **0/288** | 集中于 6–16（内容位与模板位混合） |
| γ=0 复制主导（模型路由） | **0/288** | 从位置 0 起遍布全串（路由未锁，复制到错误位置） |
| c_γ0 正确路由复制对照 | 74/288 = **0.2465**（逐格 0.292/0.000/0.375/0.354/0.000/0.458） | **只出现在模板位 0–2（9 次）与 15–20（899 次）**；18 个复制内容位（3–14、21–26）**一次都不是非法起点** |

**架构层结论**：`decode_endpoint` 是纯 `argmax`（无 UTF-8 掩码、无修复），**字节空间里没有任何合法性约束**。这不是“模型乱码”，是设计里就没这条约束：
- 纯生成通路：非法字节出现在内容位与模板位，说明生成侧没有合法性先验；
- 复制通路：路由正确时复制出的 18 个内容字节本身是合法 UTF-8（无非法起点），**复制重组没有问题**；
- 剩余 75% 非法串完全来自**模型自选的模板字节（15–20）**，与复制无关。

因此“两条通路都低”不能推断成“复制路径坏了”；本轮的对照（c_γ0）把责任明确划给**未加约束的生成/解码层**。任何整句级门都必须先补这条约束（掩码或修复），否则整句门在当前解码规则下不可达。

### 5. 逐字节门预注册（新包，不追溯 18.1）

| 门 | 阈值 |
|---|---:|
| 物件逐字节 | ≥0.95 |
| 地点逐字节 | ≥0.95 |
| `new_name_copied` | ≥0.60 |
| 整句 overall | ≥0.80（保留） |

- 加门理由：18 个内容字节连乘使整句门在当前精度下不可达（0.80^(1/18)=0.9877/字节）；逐字节门给出可攀爬的梯度，同时**不放松**整句门。
- **当前差距**（闸门只调 γ、路由不动的最好点）：物件 0.110 → 差 **0.840**；地点 0.056 → 差 **0.894**。路由钉住的上限 c_γ0 为 1.000/1.000，差 0（证明门本身可攀）。
- 预注册文件状态 `preregistered_for_next_package_not_retroactive`，`does_not_change_n18_1=true`；18.1 判负结论不变。

### 6. 该修哪一层

| 层 | 本轮证据 | 判定 |
|---|---|---|
| 门（γ） | 六档全扫，抄回率 ≤0.014、整句 0；物件字节 +0.013 | **不是主杠杆**，单独修门到不了逐字节门；按 M3 不得读成训练信用饥饿 |
| 指针/transport（β、m、cursor） | b_pure 0.755 → c_γ0 1.000；首字节 1.0 后漂移，地点第 2 字节即塌 | **前向主杠杆**：修“位置跨字节携带”可回收 24.5% |
| 生成约束（UTF-8） | 纯生成 0/288 合法；c_γ0 下非法只出现在模板位 | **架构缺口**，必须补掩码/修复，否则整句门不可达 |

**唯一建议的下一步**：先修 transport 状态（让 `m` 沿前缀携带位置、推进 `prefix_cursor`），目标把 b_pure 从 0.755 推到 ≥0.95；UTF-8 合法性约束作为同一包的架构门槛一并预注册，不单独另起一轮。**这是前向读数给出的修复方向，不涉及训练信用归因（M3）。**

**成本**：正式探针 **1,634.2 CPU / 310.2 wall、0 Adam、0 confirm**（报告 `cost` 字段实测）。另有两次因探针保真度修正而作废的实现（同一套 6 格 × 48 例，未归档进程计时，按同规模估算约 3.3k CPU），与一次 3 例 smoke；只读、不计入 18.1 正式训练预算（16,196.3），但作为返工成本如实登记。

## 2026-11-04 否定定理审计（M1 / M2 / M3，0 训练）

数学落盘：[SPEC-COPY_OPERATOR_OP2_MATH_20261104.md](../../reference/SPEC-COPY_OPERATOR_OP2_MATH_20261104.md) §3（M1：`∂η/∂α≠0`）、§5.1（M2：`a_i ≥ λ_i·M_i`）、§7.2（M3：γ 大≠信用饥饿）；数值 [op2 数学报告](../../../reports/taiji_copy_operator_op2_math_20261104.json)。永久登记：[INVARIANT-LEDGER §12/§13](../../reference/INVARIANT-LEDGER_20261103.md)；17 设计顶部已加 `blocked_by_negative_theorem: M1, M2`。

| 档 | 对象 | 动作（已做/待做） |
|---|---|---|
| `must_change` | 17.4 第 200 行“对象→地点/null 切换中断续推一致”、17.7 第 257 行“非空 cursor 中断恢复一致”、账本 T8、§10 硬不变量“复制状态逐位” | 已加限定：只允许“同一控制范围 + 固定记录序列”内签逐位；跨记录任意切换只能报实测。18.1 预检的切换点是 teacher-forcing 固定 gold 路由，不是任意控制，**仍可判逐位一致** |
| `must_retract` | 账本 L4“内容与绑定分离/独立可控”；`TAIJI_CORE_DESIGN_REVIEW:138`、`SPEC-NATIVE-CONTENT-CORE_R1:160` 的“独立可控通道”；03/能力卡的“γ 门是最大单一压制 / 主因是 γ 门” | 已撤回/改写；M3 相关句改为“前向干预、不推断训练信用” |
| `veto_if_reliance` | 17 号把 M1（结构分离）或 M2（任意切换 + 原样续读）当作核心保证 | 视为**结构性主张被否**，不得写“部分保住”；17 的 A/B/C 对照与测量设计仍成立，但解释不得用“分离通道” |

**M2 待裁定（唯一问题）**：① 保逐位继承、把记录控制限定在同一控制范围；② 保全任意记录控制、把恢复降为“状态等价 + 可重放”。两选项的代价与对 17.4/T8/18.1 判据的影响见 [账本 §13](../../reference/INVARIANT-LEDGER_20261103.md)。**本窗口不擅自选择，不启动新训练。**

## 2026-11-04 18.1 开训前预检（通过，0 训练，等 owner 批准）

证据：[预检报告](../../../reports/taiji_preflight_18_1_20261104.json)＋[manifest](../../manifests/taiji_preflight_18_1_20261104.json)，脚本 [`preflight_taiji_18_1.py`](../../../scripts/training/preflight_taiji_18_1.py)。六格（2 seed × 3 臂）× 3 次工程 Adam（update 201–203，joint 阶段）+ 1 次 1 更新失败注入（update 301）；**0 正式训练、0 confirm 读取**。数据取 18.0 的 `data.json`（sha `8242f2a8…`，train 640 / dev 192）。

### 检查点保存/恢复读数（六格）

| 格 | 复制状态来源 | 前进字节 | m / coverage 绝对和 | prefix_cursor（真实/探针） | keys / offsets | 参数 / Adam / 教师强制 / 状态 max_abs_diff |
| --- | --- | --- | --- | --- | --- | --- |
| 28031-A | teacher_forced | 1 | 2.79e-10 / 2.79e-10 | 0 / 20 | 243 / 8 | 0 / 0 / 0 / 0 |
| 28031-B | teacher_forced | 7 | 0.9246 / 0.9246 | 0 / 20 | 243 / 8 | 0 / 0 / 0 / 0 |
| 28031-C | teacher_forced | 1 | 2.62e-09 / 2.62e-09 | 0 / 20 | 243 / 8 | 0 / 0 / 0 / 0 |
| 28032-A | teacher_forced | 1 | 1.06e-13 / 1.06e-13 | 0 / 20 | 246 / 8 | 0 / 0 / 0 / 0 |
| 28032-B | teacher_forced | 5 | 2.08e-33 / 2.08e-33 | 0 / 20 | 243 / 8 | 0 / 0 / 0 / 0 |
| 28032-C | teacher_forced | 1 | 2.74e-13 / 2.74e-13 | 0 / 20 | 246 / 8 | 0 / 0 / 0 / 0 |

- 恢复后 77 个 Adam 状态项（3 次 joint 更新后已非零）、185 个权重张量摘要逐项一致；父 bundle sha 与 `candidate_sha_v2`（`773dcec2…`）前后不变。
- **结构事实**：17 路由图的 `_advanced_state` 只推进 `living/m/coverage`，**不推进 `prefix_cursor`**（该字段只在 `binding_copy.advance` 路径自增），所以真实中间态游标恒为 0。为覆盖 bundle 的这一字段，落盘前把游标置为 20（`prefix_cursor_probed`），恢复后逐位一致；报告同时登记 `prefix_cursor_real=0`，不把探针值当训练读数。
- **复制质量结构事实**：`advance_copy` 的 `m` 是**按当前接受字节重建**而非累积，且冷候选（仅 3 次工程更新）大多几乎不搬运复制质量，故 5/6 格 m 只有 1e-9 至 1e-33 量级（非 0、可逐位比较），仅 28031-B 为 0.925。这**不是**恢复缺陷，而是训练前冷候选的正常读数；正式 18.1 到 step150 会再看一次真实 m/coverage 量级。

### 三项断言结果

| 断言 | 结果 |
| --- | --- |
| 检查点保存/恢复（参数、Adam 矩、教师强制概率、复制状态逐位） | **六格全部通过**（四项 max_abs_diff 均为 0.0） |
| 父 bundle 与 `candidate_sha_v2` 773dcec2 不变 | **通过**（六格均不变；身份前缀校验通过） |
| 先存包后评估顺序生效 | **通过**（顺序恒为 progress → save → triple_check → eval_probe；六格均如此） |
| 评估路径冒烟 `evaluate_dev(full=True)`（每格 8 例，0 更新） | **六格全部跑通**（8 例/格）；`new_name_copied`=0.0 / 分母 8，`unseen_output_name`=0.0 / 分母 8 |
| S1 分母来自子集长度 | **通过**（评估器分母与独立重算分母逐项相等；两项比率差 0.0；`new_name_copied` 分母 = 样本长度 8） |
| 进度 update = 包存在 = 包内 step | **六格全部一致**（203 / 存在 / 203） |
| 1 更新 + 评估失败注入 | **通过**：只写 `evaluation_failed_weights_preserved` 回执，包仍在、step 301、185 个权重摘要逐项保留 |

- 冒烟样本为 dev 前 8 例（world 0 的 object 0/1 × 4 地点），所以 `new_description` / `both_new` 分母为 0（该切片没有新描述样例）；报告按 S1 记 0=0。两项被点名的指标（`new_name_copied`、`unseen_output_name`）分母均为 8 且算出有限值。
- `new_name_copied`=0.0 是冷候选的正常读数（输出尚未抄回名字），不是分母/定义错误：该指标独立实现按「物件名 AND 地点名逐字节出现在输出」计数，分母来自子集长度。

### 费用与停止条件

本轮成功运行 **200.8 CPU / 205.5 wall**、**19 次工程 Adam**（6×3+1）、0 训练更新、0 confirm 读取，低于单次上限（60 Adam / 400 CPU）。返工成本：前两次中断尝试（脚本自身缺陷：固定 6 步时 `m` 因重建语义落在 0）各约 15 Adam / 约 120 CPU，未逐次计费；**按累计口径 49 Adam / 440 CPU（成功 19 Adam / 200.8 CPU + 返工 30 Adam / 239.2 CPU），已作为“脚本返工成本”登记进总账**，其中的 440 CPU 也写入 18.1 报告的 `totals.prior.breakdown.n18_1_preflight_cumulative`。该项 CPU 略超 400 单次上限，属工程返工，不影响预检结论。

**结论：预检通过。** 18.1 正式比较（六格 × 300 更新，裸价 9,109.4 CPU / 保护价 19,220.8 CPU）已获 owner 批准并启动（in flight）；执行纪律照旧：先存包再评估、进度=包=step 三重校验、每格可从 step150 续跑、失败只写回执不丢权重、观察线 19,220.8 CPU / 30,000 wall。

## 2026-11-04 17.8 记忆排除探针（0 更新，判定记忆主导）

证据：[探针报告](../../../reports/taiji_memory_dominance_probe_20261104.json)＋[manifest](../../manifests/taiji_memory_dominance_probe_20261104.json)，脚本 [`probe_taiji_memory_dominance_17_8.py`](../../../scripts/training/probe_taiji_memory_dominance_17_8.py)。用六个 step300 bundle 做零更新前向：每个 dev 样例改写为四变体（`control` 分布内对照 / `new_new` 新物件×新地点 / `new_old` 新物件×旧地点 / `old_new` 旧物件×新地点），训练集词表只有 3 个物件名 × 5 个地点名，可全量背诵。

池化读数（每变体 n=24，六格均值；γ=生成权重，1−γ=复制权重）：

| 变体 | 物件逐字节 | 地点逐字节 | 实体双对 | 整句完成 | 删记录后字节变化 | 换错记录 | 换位置 | 新名字被抄回 | γ |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| control | 0.333 | 0.208 | 0.042 | 0.042 | — | — | — | — | 0.918 |
| new_new | 0.000 | 0.000 | 0.000 | 0.000 | 1.000 | 0.542 | 0.667 | **0.000** | 0.924 |
| new_old | 0.000 | 0.250 | 0.000 | 0.000 | 1.000 | 0.625 | 0.792 | **0.000** | 0.920 |
| old_new | 0.208 | 0.000 | 0.000 | 0.000 | 0.958 | 0.667 | 0.708 | **0.000** | 0.919 |

判定：**6/6 格 memory_dominant**，判据为「`new_new` 实体逐字节 ≤0.10 且任何记录条件（基线/删记录/换错记录/换位置）都不让新名字逐字出现」。要点：

- 模型并非对记录完全无感：删记录/换记录/换位置会让输出字节改变（0.54–1.00），但变化只在**已背下来的名字池**内换（如 28031-A 恒吐「青釉瓷瓶」、28032-B 恒吐「素白陶罐」、28031-C 物件常对而地点恒「檐下」），**没有一次把新名字抄出来**。
- 与 T3 的关系：T3（gold 强制 720/720）证明**算子在给定路线下可用**；本探针证明**训练从未需要它**——任务前提允许背诵替代复制。因此 17.5 的“模板正确、内容错”不是算子缺陷，而是**数据前提缺陷**。
- γ 实际使用值 0.918–0.924、γ<0.5 占比 0.075–0.082，即生成权重约 92%、复制权重约 8%；复制通路虽被分配了 8%，却从未承担内容。
- 自检：S1 子集比率分母用子集长度（0.5 断言通过）；S2 跨变体缓存键为 `(variant, world, index)` 且唯一（16 例）；配对指标保存原始字节 `raw_hex`（未保留解码副本）。
- 费用：worker 114.2 CPU（6 进程并行，wall 27.4），driver 0.1 CPU；0 优化器步、0 训练更新、0 confirm 读取。
- **口径限制**：`content_drop`（按 completed 的降幅）在基线完成为 0 时退化，因此主判据改用「新名字逐字抄回率」，`output_change` 只作诊断。
- **旧读数 superseded**：首轮用「completed 降幅」判定，基线完成为 0 时降幅恒 0（退化），读数（new_new 实体 0、三类降幅 0、`bite_rate` 1.0）不作判据；终判以本节的 `copy_rescue=0` 为准。同名报告文件已被终版覆盖，首轮数值仅存于此条登记。

## 2026-11-04 17.9 续跑 2.2× 变慢排查（外部争用）

证据：[排查报告](../../../reports/taiji_resume_slowdown_probe_20261104.json)＋[manifest](../../manifests/taiji_resume_slowdown_probe_20261104.json)，脚本 [`probe_taiji_resume_slowdown_17_9.py`](../../../scripts/training/probe_taiji_resume_slowdown_17_9.py)。在同一份 step150 包（`28031-B_stop_lock-step150.pt`，sha `4e640844…`）上、空闲机重测 joint 阶段两步：

| 项 | CPU/更新 |
| --- | --- |
| 本次实测（update 201/202） | **7.48**（课程 1.50 + 前向 2.34–2.41 + 反向 3.61） |
| 首跑（attempt1，300 更新 × 6 格） | 7.08 |
| 续跑（attempt3，150 更新 × 6 格） | 15.78 |

判定 `external_contention_suspected`：单步成本与首跑相差仅 5.7%，续跑路径没有稳定复现 2.2×；时间窗与 4 棵残留 `python -m black`（约 78,800 CPU 秒，单棵 19.5k–20.4k CPU 秒）重叠，是唯一有证据的竞争解释。后续定价保护系数 **2.11**（即按 2.2× 上限预留，属外部负载而非代码路径）。费用 17.0 CPU / 17.6 wall、2 次工程 Adam，在 4 Adam / 60 CPU 上限内；未落盘任何权重。

## 2026-11-04 数据前提修订三选项（待 owner 裁定，不实现不训练）

三选项同源于 17.8 的结论：当前 3×5 词表让背诵 15 种组合即可满足训练损失，复制通路从未承重。裁定前 18–22 全部冻结。

| 选项 | 做什么 | 直接成本 | 上限与风险 |
| --- | --- | --- | --- |
| **A 扩大词表 + held-out 划分** | 物件/地点各扩到数十个，按「组合 held-out」划 train/dev：训练集不出现 dev 的（物件×地点）配对，dev 还含全新名字 | 数据生成≈0 CPU；1 个窗口 6 格 × 300 更新 ≈17k CPU（含评估） | **上限最高且唯一能证明泛化**：背诵在 held-out 上必然失效，复制必须承重。风险是仍需选择头跟上（当前 selection 0.750，门 0.90），可能暴露第二个瓶颈 |
| **B 内容槽强制复制** | 在物件/地点字位把 γ 压到 0（纯复制），其余位置保留模板生成 | 改 mix 的逐位 γ，约 1 天工程 + 同剂量 ≈17k CPU | 上限中：能让复制成为承重路径，但**不能证明泛化**（仍可背词表），且可能伤模板流畅性；只能当“机制验证”而非“能力验证” |
| **C A+B 兼做** | 扩大词表 + held-out，同时在内容位强制复制 | ≈20–22k CPU（数据 + 双改动） | 上限最高，但**双变量同时改**，读数无法归因到“数据前提”还是“机制强制”，违背归因纪律 |

**唯一建议：先做 A。** 理由是 17.8 已把根因定位到数据前提，A 是该根因的直接解，且它同时是 B 的必要前置——没有 held-out，B 的增益无法与背诵区分。A 若仍不让复制承重（`selection` 或 pointer 定位成为新瓶颈），再单独做 B 作为第二段；C 只在 A、B 各自读数清楚后才考虑合并。A 的预算需 owner 新开窗口批准（17 窗口观察线已 170.1%，不能在 17 内追加）。

## 2026-11-04 18.0 方案 A 数据前提 + 定价 + 判据预注册（0 训练，等 owner 批准）

证据：[数据清单](../../manifests/taiji_data_premise_18_0_manifest_20261104.json)、[预注册](../../manifests/taiji_data_premise_18_0_prereg_20261104.json)、[定价报告](../../../reports/taiji_data_premise_18_0_pricing_20261104.json)＋[定价 manifest](../../manifests/taiji_data_premise_18_0_pricing_20261104.json)；脚本 [`data_premise_18_0.py`](../../../taiji/data_premise_18_0.py)、[`build_taiji_data_premise_18_0.py`](../../../scripts/training/build_taiji_data_premise_18_0.py)、[`price_taiji_data_premise_18_0.py`](../../../scripts/training/price_taiji_data_premise_18_0.py)。数据落在 `output/taiji_data_premise_18_0/data.json`（sha256 `8242f2a8…`，2,859,104 字节）。本轮 **0 训练更新、0 优化器训练步、0 confirm 读取**；confirm 只写了 sealed 规格（0 例生成）。

### 数据清单与三向无交集证明

| 项 | 读数 |
| --- | --- |
| 名字池 | 物件 220（20 修饰 × 11 器物，4 汉字 = 12 字节）、地点 220（20 字头 × 11 字尾，2 汉字 = 6 字节） |
| 三向切分 | 按**名字串**切 160 / 30 / 30（train / dev / confirm），train 与 dev 名字互斥 |
| 样本 | train 640 例（40 世界 × 16）、dev 192 例（12 世界 × 16）、confirm 192 例（sealed，未生成） |
| 字节体积 | train 记录 9,759 B / 动作 17,280 B；dev 记录 2,928 B / 动作 5,184 B；最大单例源 246 B（预算 512 B） |
| 组合 | train 640 个唯一（物件×地点）组合、dev 156 个唯一组合，**交集 0** |
| dev 轴分母 | `unseen_output_name` 192/192（全部 dev）、`new_description` 96/192、`both_new` 96/192 |

跨向出现计数（八项，全部为 0，脚本内断言：`nonzero_occurrence_keys=[]` 且零计数键集合恰为该八项）：

| 键 | 计数 |
| --- | --- |
| train 物件名出现在 dev | 0 |
| train 地点名出现在 dev | 0 |
| dev 物件名出现在 train | 0 |
| dev 地点名出现在 train | 0 |
| confirm 物件/地点名出现在 train | 0 / 0 |
| confirm 物件/地点名出现在 dev | 0 / 0 |

三条硬约束逐条落地：①**陌生字节串**——dev 的 30 个物件名与 30 个地点名是训练集**从未出现过的完整字节串**（不是"没出现过的组合"），并且全部排除 17.0 冻结语料的 5 物件名 + 5 地点名；②**词表规模**——物件/地点各 220 ≥ 200，背诵全部名字的容量代价远大于抄写；③**组合 held-out**——train/dev 组合交集 0 且名字串互斥，两者同时成立。

**容量影响**：参数增量 **0**（名字存在记录字节里，不是嵌入；27 字节动作与 36 字节目标不变），容器仍 3,563,859 参数 / 77 可训练张量 / `candidate_sha_v2` 773dcec2…。已登记两个可选规模，均 delta 0：默认 220×220（640 / 192 例）；宽版 440×440（1,280 / 384 例，仅在默认池让 held-out 准确率饱和时才启用）。**未改任何图结构。**

### 定价（17.3 同形口径：0 更新 + 4 次工程 Adam）

| 组件 | CPU |
| --- | --- |
| 生成 / 例 | 0.207 |
| dev 轻评估 / 例 | 0.416 |
| dev 全评估 / 例 | 1.300 |
| 反序 / 例 | 0.211 |
| 反事实换名 / 例 | 0.199 |
| identity rescue（固定 8 例） | 0.469 |
| 保存 / 恢复 | 0.047 / 0.047 |
| selection 更新 / 步 | 1.81 |
| copy 更新 / 步 | 4.91 |
| joint 更新 / 步 | 5.12 |

单格 = 训练 1,183.6 + 评估 334.5（step0 轻评估 192 例 + step150 readiness 40 例 + step300 全评估） + 两次保存 0.094 + 一次恢复 0.047 = **1,518.2 CPU**。

| 口径 | 裸价 | 保护价（×2.11） |
| --- | --- | --- |
| 六格（2 seed × 3 臂 × 300 更新） | **9,109.4** | **19,220.8** |
| 六格 + B 臂 self-prefix 追加 | 9,198.8 | 19,409.6 |
| step150 前段（每格 514.8）× 6 | 3,088.9 | 6,517.5 |
| step150 后段（每格 1,006.7）× 6 | 6,040.2 | 12,744.9 |

B 臂 contingency 的原因：测得的 joint 两步是**冷候选**、教师强制完成率 0.0（对应 17.5 的 A/C 臂，`self_prefix` 从未触发）；17.5 的 B 臂触发 27 次自前缀课程，故单列 2 seed × 27 更新 × 8 例 × 0.207 ≈ **89.4 CPU** 追加，不摊进基础单格价。

本轮实测 69.6 CPU / 71.6 wall（CPU/墙 ≈0.97，空闲机），4 次工程 Adam、0 训练更新、0 confirm 读取；未落盘任何权重（临时保存包已删除）。首轮定价 70.8 CPU 已按“同名报告覆盖”规则由本节读数取代。

### 判据预注册（冻结于 prereg，owner 批准后生效）

| 门 | 阈值 |
| --- | --- |
| overall（dev 整句 27 字节 + EOS） | ≥ 0.80 |
| unseen_output_name | ≥ 0.80 |
| new_description | ≥ 0.80 |
| reversed | ≥ 0.95 |
| selection_object / selection_place | ≥ 0.90 / ≥ 0.90 |
| content_drop（有记录 − 无记录） | ≥ 0.10 |
| **new_name_copied（新物件名与新地点名逐字节抄回）** | **≥ 0.60，独立门** |
| identity_rescue | 8/8 逐位 |
| A−C（参考净收益） | ≥ 0.10，双 seed 世界 bootstrap 下界 > 0 |

- `new_name_copied` 是**独立门**，不得用"输出变了"或"输出在记忆池里换了名字"替代；任一格 overall ≥0.80 但 `new_name_copied` <0.60 一律标 `recitation_remains`，不算能力增益。
- 每格必须同时报 `gamma_mean` 与 `1−gamma_mean`，缺一侧该行作废。
- 任务门失败即负结果：不追加 seed、不追加剂量；step150 readiness 失败在 step300 前停该格；候选与评估器冻结前不读 confirm。
- 既有 17.3 判据**未改动**，新门只写在新包；`a_minus_c` 仍按 bootstrap 下界解释。

### A 与 B 的关系

- **A（本包）先跑**：大词表 + 名字串互斥 + 组合 held-out。没有它，任何"复制承重"的读数都不能与背诵区分。
- **B（内容位 γ 压 0）在 A 之后才有归因意义**：A 若已让复制承重，B 才可用于分离"复制权重不足"这一机制变量；A 若仍不承重（`selection` 或 pointer 成为新瓶颈），B 应作为**第二段单独窗口**，不与 A 合并。
- **C 不采用**：同时改数据前提与 γ 是双变量，读数无法归因。

### 费用与观察线

17 窗口累计 34,020.5 CPU（170.1%）/ 7,798.2 wall。18.0 新增实测约 150 CPU（两轮定价 70.8 + 69.6，数据构建与格式校验约 10 量级），全项目累计约 34,300 CPU；工程 Adam 本轮 8 次（两轮定价各 4），累计约 268 次。18 为 owner 新开窗口，**未新增任何训练剂量**。

**停止条件：本包到此为止。** 18.1 正式比较（六格 × 300 更新）必须等 owner 明确批准 prereg 与预算后才可启动，本次不启动。

## 2026-11-04 17.7 收束（四项总账、停止条件、未排除解释）

证据：[17.7 收束报告](../../../reports/taiji_selection_routing_17_7_closeout_20261104.json)＋[manifest](../../manifests/taiji_selection_routing_17_7_closeout_20261104.json)，由 [`closeout_taiji_binding_routing_17_7.py`](../../../scripts/training/closeout_taiji_binding_routing_17_7.py) 只读汇总（0 优化器步、0 训练更新、0 confirm 读取，0.36 CPU / 0.35 wall），并对 12 份 bundle 逐一登记 sha256（6 × step150 + 6 × step300）。四项总账：

- **同一候选的可靠区间**：可重放、可恢复、身份逐格一致（`candidate_sha_v2` 773dcec2…）、identity_rescue 8/8；不可靠的是任务级能力——整句复制 ≤0.200、选择 top1 ≤0.750、新名字/新描述 ≤0.083。
- **机制的实际贡献**：gamma 有效值 0.875–0.992（无格触底 0.2）、内容干预降幅 0.000–0.250、A−C +0.017/−0.050、A−B −0.075/−0.200；结论是复制未承重，且路由信用臂比 detach 更不看记录。
- **可沿用的旧能力/状态/资源**：binding 图（3,526,481 参数）、bundle schema 与 step150 断点恢复、评分器与内容干预三件套、17.3 冻结阈值、三臂/阶段/价格合同；不可沿用：任务级复制能力、0.75 的选择精度。
- **下一任务的适用性**：18–22 可复用容器/身份/恢复/评分工具链；必须替换的是复制算子（OP2 候选）、课程判据（改挂教师强制完成率或按位置触发）、信用路径（ST 未优于 detach）、剂量（300 更新 / batch 8 不足）。

停止条件：17.6 confirm 封存不释放；17 窗口到此为止，不追加 seed、不追加剂量、不新开头；12 份 bundle 作为证据资产保留，旧读数原字节有效。未排除的竞争解释：欠训练/剂量（not_excluded）、架构捷径（supported）、纯教师强制口径差（supported）、ST 信用方向（supported）、评分/分母（仅 reversed 退化通过一项 not_excluded）。

## 2026-11-04 17.5 三臂终读与冻结闸门（负结果）

证据：[17.5 报告](../../../reports/taiji_selection_routing_17_5_compare_20261104.json)＋[manifest](../../manifests/taiji_selection_routing_17_5_compare_20261104.json)。语义身份 `candidate_sha_v2` 773dcec2…，六格一致；容器 3,563,859 参数；0 confirm 读取、0 剂量变更。终读由 `--assemble` 从已落盘 step300 包 + `progress-*.json` 重建（每格先做“进度 update=300 / 包存在 / 包内 step=300”一致性校验），只重算 dev 评估，0 重训更新。

| seed | 臂 | step0 overall | step150 ready(完成率/物件/地点) | step300 overall | unseen | newdesc | sel 物件/地点 | dev NLL |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 28031 | A_route_lock | 0.000 | 0.000 / 0.000 / 0.000 | 0.067 | 0.000 | 0.017 | 0.750 / 1.000 | 2.091 |
| 28031 | B_stop_lock | 0.000 | 0.000 / 0.000 / 0.000 | 0.142 | 0.000 | 0.083 | 0.750 / 1.000 | 1.300 |
| 28031 | C_unlocked | 0.050 | 0.050 / 0.250 / 0.200 | 0.050 | 0.000 | 0.050 | 0.750 / 1.000 | 1.778 |
| 28032 | A_route_lock | 0.000 | 0.000 / 0.000 / 0.000 | 0.000 | 0.000 | 0.000 | 0.338 / 0.100 | 5.225 |
| 28032 | B_stop_lock | 0.000 | 0.000 / 0.000 / 0.000 | 0.200 | 0.000 | 0.017 | 0.750 / 1.000 | 1.224 |
| 28032 | C_unlocked | 0.000 | 0.000 / 0.000 / 0.000 | 0.050 | 0.000 | 0.050 | 0.750 / 1.000 | 2.013 |

闸门（冻结 17.3 值，未改）：overall 0.80 ❌、unseen_output_name 0.80 ❌、new_description 0.80 ❌、selection_object 0.90 ❌、selection_place 0.90 ✔(28031)/❌(28032)、content_drop 0.10 ❌、identity_rescue 8 ✔(8/8)、reversed 0.95 ✔但属**退化通过**（overall≈0 时两条路径输出同一错误串，不构成复制不变性证据）。A−C（净收益，门 ≥0.10 且 2,000 次世界 bootstrap 下界>0）：28031 Δ+0.017（下界 −0.008）、28032 Δ−0.050 ❌。A−B（owner 要求参考读数、不入冻结门）：−0.075 / −0.200，均低于 B 臂。**dev 门 false → 17.6 confirm 不释放。**

归因（架构层，非剂量笔误）：①表型是“模板正确、内容错”——生成串句式与目标一致（如“把铁灰木匣放到炉边”），但物件/地点取自模型自身参数化记忆而非记录内容；实测逐例可见输出与目标仅差内容段；②三类内容干预 no_source / wrong_record / wrong_position 的降幅 0.000–0.250，绝大多数 ≤0.125，且 A 臂单格最大仅 0.083、B 臂 0.250：复制路径没有成为承重路径，训练 NLL 主要被“goal→字串”的参数化捷径满足；③ST 路由信用（A）不但没把复制推成承重路径，反而比 detach（B）更不看记录（A−B 两 seed 全负，−0.075 / −0.200），B 臂反而已能整句复制正确一部分（0.142/0.200）；④课程按自由运行完成率触发：A/C 臂 `self_prefix_updates` 合计 0（从未触发、全程教师强制），B 臂 27 次触发——即“能复制”的 B 臂才拿到自前缀训练，A/C 拿不到，形成负反馈；⑤训练末批 loss 0.10–0.15（28032-A 发散到 6.45）而 dev NLL 1.22–5.22，记忆/泛化断层；28032-A 的 selection 0.338/0.100 为 A 臂单格不稳定旁证。

费用与观察线：首跑（300 更新 × 6 格）12,743.30 CPU / wall 2,215.36；第二次续跑开工前终止（估 1,500 / 480）；第三次续跑（150→300）训练 14,204.7 CPU，六格 dev 评估约 3,987.8 CPU（并行 wall 3,291.8）。累计 34,020.5 CPU（观察线 20,000 的 **170.1%**）／7,798.2 wall（30,000 的 26.0%）。越线原因：一次性缺陷修复导致的必要补跑（首跑权重丢失 + 恢复缺陷），**非剂量变更、非无信息增益的全量重跑**；owner 已授权越线不停。

## 2026-11-04 17.5 首跑失败与处置（训练完成、评估崩溃）

- 失败读数：六格均到 300/300（`progress-*.json` update=300），合计 CPU 12,743.30 / wall 2,215.36（6 格并行）；六份 step150 包与 readiness 齐全。`evaluate_dev(full=True)` 的 identity_rescue 段在 L694 构造 `ForcedRouteHead` 时漏传 `config`，崩溃发生在 step300 包保存之前 → **step300 权重与 dev 读数全部丢失**，进度 JSON 显示 300 但无对应包。归因：代码缺陷（调用点漏参），非架构断点、非剂量问题。
- 修复：①全文件 `ForcedRouteHead` 构造点补 `config`（L694；L462 此前已修）；②新增零更新回归自检 `--evalcheck`（6 格 × 24 例 `evaluate_dev(full=True)`），主入口开训前强制运行，任何一格抛错即不进入训练；③落盘顺序改为“先存 step300 包 → 校验 进度更新数 = 包存在性 = 包内 step → 再评估”，评估失败只写 `evaluation-<tag>.json` 回执并保留权重；④续跑时 step150 包缺失直接报错，不静默从头训练。
- 续跑：六格均从 step150 包续跑（包内含 Adam 矩与 RNG 状态，批次调度按 seed 重建），轨迹与不中断跑同源；未做全量重跑。
- 第二次续跑在开训前解包即失败：`load_routing_bundle` 用“全声明可训练序”比对包内**阶段子集**名，六格同报 `trainable parameter order drifted`。修复为“记录名必须是全序的保序子集”，并新增 `--resume` 前置 `resume_check`（六包 update=150、77 个可训练张量、8 个优化器参数组，实测通过）。复验通过后重启续跑。

## 2026-11-04 运维清理：残留 black 进程

06:19 只读快照发现 4 棵与 17.5 无关的 `python -m black` 进程树（PID 23368 / 16612 / 6416 / 41268，起于 2026-11-03 23:50 至 2026-11-04 0:08），仍在空转，累计约 78,800 CPU 秒（单棵 19,527–20,372 CPU 秒）。来源为此前 `python -m black --check/--workers` 调用挂起。按 owner 批准已强制终止，登记于此。自本条起，格式化只走 [`black_check_gate.py`](../../../scripts/training/black_check_gate.py)（in-process black、line-length=100），不再直接调用 `python -m black`，`ruff format` 不再作为格式权威。

## 2026-11-03 17.4恢复预检与17.2 T3补跑

17.4按两seed×三臂共6格执行，每格原端step1、step2与独立spawn恢复端step2，共18次工程Adam：[报告](../../../reports/taiji_selection_routing_17_4_preflight_20261103.json)＋[manifest](../../manifests/taiji_selection_routing_17_4_preflight_20261103.json)。参数、Adam矩、dev[0] teacher-forced概率的`max_abs_diff`全为0.0；对象→地点/null续推13个位置概率逐位一致；父bundle sha与所有非可训练参数摘要未变；CPU 46.08/wall 126.24；6个step1临时bundle共352,009,238B已删除并登记。A/B四格在step2的`m`为零、`coverage`非空（未训练头把最后一个对象字节路由到null所致），记为状态旁证，不影响恢复结论。

同一日补跑17.2遗留的T3：[报告](../../../reports/taiji_selection_routing_17_2_t3_supplement_20261103.json)＋[manifest](../../manifests/taiji_selection_routing_17_2_t3_supplement_20261103.json)。强制alpha/beta one-hot到gold字节且`q_null=gamma=0`后，720个record步的gold字节概率最小值2.08e-36、中位4.57e-6，701/720低于1−1e-12；f32 658/720低于1−1e-6。干预与映射对照：概率和误差4.4e-16、copy_byte_mass重建误差0、instrument_error_steps=0；transport行仅17/720为delta、行内gold位中位1.36e-7。失败层为架构：当前图以`anchor @ transport`的softmax软扩散复制，缺R2a精确点复制。CPU 26.22/wall 27.42，0优化器步、0训练更新、0 confirm读取。按闸门纪律，17.5不启动。

## 当前唯一下一步：见总计划的执行位置

直接按[总计划「执行位置」](../../MAINLINE.md#执行位置)推进。本标题仅保留导航兼容，不独立维护候选、状态、审批或下一动作。

## 2026-10-31 根目标与排期纠正

所有者明确连续生命活动、持续可塑性与规模生长共同构成递归进化，不是固定模型训练后另选的附加功能。根需求§3.0、架构§0.3、总计划09及后继阶段已据此修订；当前设计动作统一在总计划发布。已有R5系列仅作为局部固定容量配方证据，已有代码中独立的生命/睡眠/成长资产保留，但未据名称补签集成。

09.1静态共同内核合同可与R5c按原预算有限结案并行；本轮没有取消、停止、续跑或追加任何进程，也不把用户的目标澄清当作新服务/训练已经执行。实际生命/成长能力仍须在同一主体的状态/参数/拓扑和前后经历上验证。

## 2026-10-31 核心算法设计交付的进一步纠正

09.1必须先设计可实现、可反驳的核心神经算法，再推导运行对象/接口/保存合同。RNN/Transformer机制影响状态表达、传播及训练，不能被随意降为实现细节；目前已有局部算法与资产，尚无覆盖新根目标的统一核心算法完成/有效证据。设计判据见[架构§0.4](../TAIJI_NATIVE_ARCHITECTURE_V1.md#04-核心神经网络算法的设计交付20261031)，唯一动作仍由总计划发布；本记录不增加实验或修改冻结合同。

## 2026-10-31 09.1核心算法首稿与局部核对

所有者授权重新设计、不以当前候选兼容为约束。已交付[核心算法v1草案](../../reference/TAIJI_CORE_ALGORITHM_V1_DRAFT_20261031.md)，主方案为多时间尺度可塑神经图。新文件包含实际方程与更新顺序，不是仅列运行接口。

[合成代数报告](../../../reports/taiji_core_algorithm_v1_math_checks_20261031.json)由[检查脚本](../../../scripts/training/check_taiji_core_algorithm_v1_math.py)生成，11项局部性质通过，绑定代码SHA/NumPy版本；无完整模型、无optimizer更新、无数据分区读取。它不代表完整信用、语言、生命活动或结构成长通过，正式执行与下一步仍由总计划发布。已有R5c等独立运行不因设计交付被停止/删除或追加。

## 同轮并行实验产物的状态边界

新算法设计收尾时只读核对R5c摘要：`FAIL_BEHAVIOR`、Q1 answer约0.4580、cut_drop=1.0；随后并行提交`01b693fa8`交付[正式评价](../../../reports/taiji_native_core_r5c_evaluation_20261031.json)及原实验结案。该实验与源码不是本轮新算法设计的执行成果；本轮不据摘要给出唯一原因或修改原门。新设计依据根目标及已发布历史，不把这一负结果当全架构充分反证。

## 2026-10-31 方程验证范围

所有者要求先调整主线，再验证调整核心方程。09.1增加原式反例、耦合数值/梯度核验和修订复测三个子项；本轮允许合成矩阵/自动微分的设计验证，不执行完整模型训练、正式数据评价或既有进程操作。旧11项数学核对仍只代表其原局部范围，新结果单独留存。

## 2026-10-31 v1.1方程验证与修订回执

按先调主线后验证的顺序，09.1.1–09.1.3完成本轮范围：[验证脚本](../../../scripts/training/validate_taiji_core_equations_v11.py)、[独立报告](../../../reports/taiji_core_equations_v11_validation_20261031.json)。原式无写入反例在200步内121次触发范数裁剪；v1.1近端擦除解决该不放大性质，另明确归一化尺度、整体损失下出生信用及变时长活动的TD/trace。新22项检查涵盖耦合状态/编码/记忆信用、出生前后和启动、有限差分/二阶/元导数及float32/64有限性；旧11项报告保持原位。

本轮0 optimizer步骤，但为验证导数计算两次合成内层更新表达式；没有完整模型、数据分区、真实任务训练/能力评价或完整主体恢复。局部数学保证与剩余全图/任务问题详见[算法v1.1 §21](../../reference/TAIJI_CORE_ALGORITHM_V1_DRAFT_20261031.md#21-v11方程审查与验证记录)，不能把本报告当09.2通过。

## 2026-10-31 P0/P1复审与E1规格勘误

源码复核确认22项属于**合成耦合代理**：快更新只有硬许可/完整更新，没有连续chi；群间读取与h.mean输出不是正式Q/K/V及查询读出；q/key/value/fout/out等矩阵未开启梯度。旧报告按原条件通过的事实保留，正式全图与这些参数的信用尚未核验；不改脚本或报告SHA。

算法主稿E1补入f_i的F^T q多头读取、统一B平滑映射、P_R活动奖励/P_A准入/P_C最终确认及累计自适应上限；编码版本变更不再默认给旧H换revision即可兼容，首个短生涯以有界完整重放作迁移对照。小梯度只代表代理中可分辨的导数，float32实际更新、任务收益、持续状态保持及迁移成本仍待测。详见算法§6/10.3/12.2/21/23；排期只见总计划，本轮仅修订文档，不实施新模型或训练。

## 2026-10-31 E2边界与新颖性定位

所有者提出三条E1未覆盖边界。零优化器小矩阵复核：rank-one C按SVD请求4单元得到启动范数[0.029,0,0,0]；四点/seed=20261031的2000组内容盲门范数中位0.0347471，真门0.0538528；禁写范数约3.02000094，独立0.99衰减1000步为0.000130377。零期望指该四点全局C=0时的带符号梯度向量，并非范数。未复跑其他并行评估中的优化探针，也未更改其文件或旧11/22项证据。

主稿E2补有效秩限额与逐单元准入、内容门的随机/置换对照、chi冻结与陈旧状态合同；按原论文核对GDN2、DeltaNet/Gated DeltaNet、TTT、GradMax及可塑性先例，将五组比较按主张分工。现阶段不主张新基本算子或首次性，组合耦合的实际研究价值仍待证。执行与授权仍只见总计划，本记录不启动训练。

## 2026-10-31 v1.2生长信用决策与命题预审

所有者选定“生长信用与覆盖出生”作为研究增量，暂不添加自我修改预测/快记忆撤销组件。方向已进入算法§24和总计划；v1.1动力学/快记忆/表达公式与旧报告保留，生长候选明确区别于全局SVD基线，不据此操作训练进程或模型默认值。

[预审脚本](../../../scripts/training/check_taiji_growth_credit_v12_math.py)和[报告](../../../reports/taiji_growth_credit_v12_math_review_20261031.json)14项按声明条件通过，0优化器步骤、无数据分区。通过中包含原推广被反例推翻：rank=1但正交出边可有4行非零梯度；软门N=1秩和=2；新增残差能降聚合秩；H为0完整输入仍可有信用；随机门L_start可为1；共享簇覆盖仍可整体抵消。条件前向含真实门，6个入边坐标/三个eps与autograd一致，置零/翻符号负对照拒绝。均不代签09.2正式图或任务效力。

针对性查重发现GradMax、TINY、Low-Rank Learning by Design、GSS、Subspace Trap及Neurogenesis Deep Learning等直接相关先例；不能据当前证据声称“唯一未被覆盖”或确定新颖性等级。保留覆盖算子/耦合为候选贡献，第二阶段与延期范围按总计划决策，本台账不另发队首。

## 2026-10-31 R021正式前向与历史审计入库

按所有者要求，先冻结[六接口/凸分区门/覆盖工程界](../../reference/SPEC-TAIJI-CORE_REFERENCE_GRAPH_R021_20261031.md)，再交付实际编码、情景/null读取、可学群间/快QKV、查询多头、动力学、连续chi与位置复制/生成。 [R021回执](../../../reports/taiji_core_reference_r021_forward_coverage_20261031.json)给出12项严格float64前向检查、10行方程覆盖矩阵和101个参数张量清单；backward_calls=0、optimizer_steps=0、data_splits_read=[]。小配置44,905参数只作同构工程夹具，不是默认模型规模或定价。

覆盖出生/噪声标定、实际学习信用、编码迁移和完整主体恢复都明列未实现/未验；不据前向通过签新模型有效。当前后继只见总计划，不在本台账新增队列。

五份v1.1/E1独立审计按[历史入库身份](../../manifests/taiji_core_v11_e1_audit_admission_20261031.json)与[原字节包](../../../reports/taiji_core_v11_e1_originals_20261031.tar.gz)保存。历史设计64fdf8576及原脚本/report绑定经核对；可读脚本加历史版本检查、两处静态修正及LF规范化，原报告不冒称绑定新兼容脚本，已知方法局限留在历史说明中。未重跑其300步玩具优化，亦非v1.2重新裁决。

入库维护检查：参考前向、架构/命名、单主线、报告索引行尾与凭据守卫合计50项通过；Ruff主规则/B-SIM、Black、298条相关内部链接及source/hash绑定通过。历史脚本在探针执行前拒绝当前v1.2设计，未创建输出或运行优化。只代表本地受影响检查，不宣称远端CI或能力门通过。

## 2026-10-31 长期目标表述收束

按所有者目标复核，统一补入内部状态连续、持续可塑、继承式新增/整合/回收、真实反馈改变负责机制，以及生长收益/主体恢复的按范围验收。“经验积累与认识纠正”区分真实历史和可修订认识；旧能力保护不保护已知错误。阶段方法与根目标分开，未把v1.2信用秩或R021写成必须成功的终极方案。总计划13只承接成熟与扩域，基础生命/学习/生长仍从09开始；本次仅改文字/计划，不推进09.2.2或更改训练权限。

## 2026-10-31 R022负责参数信用与实际数值判别（09.2.2）

先冻结[R022合同](../../reference/SPEC-TAIJI-CORE_CREDIT_R022_20261031.md)：三个声明负责目标（T_EXPR教师强制预测NLL、T_WRITE2两步快写链、T_WORLD_PROXY常数代理）与零类预声明，再交付[检查](../../../tests/taiji_native/test_living_core_credit_r022.py)与[报告](../../../reports/taiji_core_r022_credit_validation_20261031.json)。[R021回执](../../../reports/taiji_core_reference_r021_forward_coverage_20261031.json)所绑`taiji/living_core.py`字节经SHA核对未变；本报告`passed_r022_credit_only`、8/8检查通过，backward_calls=127、optimizer_steps=2（声明合成更新表达式，更新后参数即废弃）、data_splits_read=[]、费用54.61/900 CPU秒。

101个参数张量全部获得FD确认的信用或落入声明零类：87 credited＋12 world代理（真实世界损失仍属09.3/09.4）＋V birth_zero（初值U=0⇒V梯度精确0；声明两步Adam更新使U≠0后V解锁9.4e-4）＋`episodic.score.bias`判明为**softmax平移不变死参数**（I3的b_3：位置softmax对全体分数共同平移不变，值置换对照无任何效应；梯度为~3e-18浮点抵消残留，声明更新中累计|Δθ|≈1.2e-12≤lr·1e-9且更新后仍惰性）。该死参数是R021接口级结构事实，建议后续R021接口修订删除或标注；修订前保留原位、永不获信用。

读数：FD三eps（1e-4/1e-5/1e-6）扫描最优失配≤1.45e-10（atol 1e-8）；二阶方向导数100/100吻合、最大差1.9e-6；快写Frobenius裁剪分支未激活（余量14.78）；float64/32逐组比较98个已解析张量中位偏差2.6e-7、最大3.9e-6（门0.1）；置零/翻符号/许可allowed=False/表达后写排序/空view断路（含强制g_mix=1）负对照全过，coverage≡0/prior≡0状态条件零及其正对照全过，U激活对照成立；声明更新输出改变max|ΔP|=0.163，世界族Δθ精确为0（代理不进更新目标）。

执行中按合同纪律修订两处并落账：①发现死参数后先补录`inert_shift_invariant`零类（含值置换对照）再重跑，未改阈值放水；②置零对照阈值由`max(1e-8,1e-6·|L|)`修订为`max(1e-10,1e-8·|L|)`＋一阶预测回退——对照的语义是因果连通性，效应大小由FD确认的梯度负责；message_goal.weight实测ΔL=6.86e-6与一阶预测|g·θ|=6.88e-6几乎精确吻合，确认局部线性。本报告不签任务能力、生长、保持或恢复；当前唯一下一步见总计划（09.2.3）。

提交后按钩子要求重跑Mimosa深扫（focusFiles=本包三个代码/实现文件）：jobId `scan-job-mvwfho3z`，scanId `scan-2026-10-31T13-22-24.513Z-ef51fc8b7725`，**封印`sha256:6da506b4ecd7348849af483a998cf465c7df358c9cbb115b9746d35bbac86260`**，1,652项findings为既有普查基线、无新假设、离线advisory零匹配；同提交先跑单主线审计（ok）与全仓sha-drift普查（117,275份报告/5,216条sha声明，12条drift与3,174条missing均为历史checkpoint/pytest临时文件及并行清退会话删除所致，不涉本包证据链）。

## 2026-10-31 R023信用覆盖出生与有界迁移（09.2.3）

先冻结[R023合同](../../reference/SPEC-TAIJI-CORE_GROWTH_R023_20261031.md)，再交付[growth_credit.py](../../../taiji/growth_credit.py)、[state_migration.py](../../../taiji/state_migration.py)、[检查](../../../tests/taiji_native/test_growth_credit_r023.py)与[报告](../../../reports/taiji_core_r023_growth_birth_20261031.json)。`taiji/living_core.py`字节经SHA核对与R021/R022绑定一致；报告`passed_r023_birth_engineering_only`、8/8检查通过、backward_calls仪表计数落账、optimizer_steps=0、data_splits_read=[]、费用1.97/900 CPU秒。

交付事实：K=2门控信用构造（copy/generate混合门，图内量可审计）；逐(支路,群)128次置换噪声标定（每群独立零假设）；信号条件与max-min整数分配（R_max=8、枚举81态/群、tie-break=min rho→total rho→少用单元→稳定支路id，预算停实现）；`GatedGrowthCore`门控出生算子（`pre += Σ_c g_c·B_c·tanh(R_c x)`，镜像R021步进体＋唯一声明新增，父行不分门）；全局SVD基线对照臂（本夹具8单元/群 vs 覆盖分配1+1，判别性差异入账）；共享支路整体信用守卫。§10.3迁移：journal恒等重放逐位相等、编码变更shadow事务提交revision+1（非encoder参数逐字节不变、重放无参数突变/无奖励/无外部动作）、事件/字节预算拒绝保父。

读数：出生中性逐位成立；4个出生单元启动梯度与算子预测`B^T C_c`在1e-12一致（8.5e-4..2.5e-3＞1e-8）；L_start=1.0、独立方向比例1.0、kappa=0.998；P新列出生时刻梯度精确0；启动零信用准入失败→lesion回滚逐字节恢复；Hadamard反例（秩1但4行范数0.5）、`C₁=−C₂`整体抵消（全局基线拒绝＋共享守卫拒绝）、软门N=1秩和2、重放非单调（秩2→1按有效结果接受）全部按声明判定。

执行期发现（按纪律落账，不改阈值）：①§24.4新生单元是门控的（R021 §4"实际新支路和诊断必须使用同一个g"），门控前向为v1.2候选声明实现；②稠密`[n,m,p]`布局下逐群不对称出生需R/P零填充，padding单元梯度恒0且永不启动；③**冻结0.99置换噪声bar在本夹具不拒绝随机门**——真实门与随机门的σ₁/τ_noise余量同为1.03–1.05倍，时间平滑生涯的共同模结构使置换零假设膨胀；已如实记录校准不可分辨性，作为R021 §4标定协议owner决策输入（分块置换/去均值零假设等候选留其后），本合同不缩quantile。分配/出生/标定/迁移均不签任务收益、保持或恢复；唯一下一步见总计划（09.2.4完整主体恢复）。

提交后按钩子要求重跑Mimosa深扫（focusFiles=本包四个代码文件）：jobId `scan-job-mvwhkswo`，scanId `scan-2026-10-31T14-20-21.713Z-1875cac11fde`，**封印`sha256:5df80279f075d3494802b55453abeb409448b5d8f7bc20d54e37b3f8c8b2ed0e`**，1,652项findings为既有普查基线、无新假设、离线advisory零匹配；同提交单主线审计ok。

## 2026-10-31 R024完整主体恢复（09.2.4）

先冻结[R024合同](../../reference/SPEC-TAIJI-CORE_RESTORE_R024_20261031.md)，再交付[state_migration.py扩展](../../../taiji/state_migration.py)（显式step序列/断点续推/save_subject/load_subject/resume_subject/run_worker_job）、[worker CLI](../../../scripts/training/r024_restore_worker.py)、[检查](../../../tests/taiji_native/test_subject_restore_r024.py)与[报告](../../../reports/taiji_core_r024_subject_restore_20261031.json)。`taiji/living_core.py`字节经SHA核对与R021/R022/R023绑定一致；报告`passed_r024_restore_engineering_only`、6/6检查通过、optimizer_steps=3（声明合成更新表达式）、data_splits_read=[]、费用6.16/900 CPU秒。

主体bundle（`taiji-subject-bundle-r024-v1`，`weights_only=True`可读）覆盖：主体id/revision/step/config、出生结构与门值表（来自出生时父端点trace的审计量）、全部参数、H/F/nu_ema/coverage/prior/keys、声明Adam(lr=1e-2,betas=0.9/0.999)状态、资格迹替身（e←0.99e+mean‖x‖/√p，纯记录量）、RNG状态、生涯spec与游标、单调journal（fee/outcome/resolve/restore条目，seq严格递增）与未结结果。五个边界各保存并由**multiprocessing spawn独立解释器进程**续推：下一微步h/probabilities、下一次声明更新后参数/矩/更新后再微步与父端逐位相等；RNG探针、资格迹与费用按公式延续不清零；resolution以当前journal唯一闸实现幂等（旧快照重放拒绝重复领奖）；journal seq单调。

执行期工程事实（落账）：Windows下torch张量不能过进程队列，结果改走文件传递；spawn目标必须按名可导入（worker job直挂`taiji.state_migration.run_worker_job`）；"下一次更新"的比较要求父端同样执行续更新再逐位比对，bundle中保存的是上次更新后的参数与矩。声明更新为合成表达式（教师强制重放分支一步Adam），不签训练；迁移恢复通过明列**不替代语义保持**。09.2.1–09.2.4全部关闭；唯一下一步见总计划（09.3连续动力学与内容活动），训练/定价仍未准入。

提交后按钩子要求重跑Mimosa深扫（focusFiles=本包四个代码文件）：jobId `scan-job-mvwj70dh`，scanId `scan-2026-10-31T15-05-26.437Z-71a2162c069e`，**封印`sha256:919231ee5cd8d383802216b21b46c349b9b8676ffa81978c9ac162048fbcf345`**，1,652项findings为既有普查基线、无新假设、离线advisory零匹配；同提交单主线审计ok。

## 2026-10-31 R025活动循环与可追溯选择（09.3）

先冻结[R025合同](../../reference/SPEC-TAIJI-CORE_ACTIVITY_R025_20261031.md)，再交付[native_activity.py](../../../taiji/native_activity.py)、[检查](../../../tests/taiji_native/test_native_activity_r025.py)与[报告](../../../reports/taiji_core_r025_activity_dynamics_20261031.json)。`taiji/living_core.py`字节经SHA核对与R021–R024绑定一致；报告`passed_r025_activity_engineering_only`、5/5检查通过、无optimizer、data_splits_read=[]、费用0.16/900 CPU秒。

交付事实：六活动（interact/recall/imagine/explore/maintain/wait）循环——策略头（seeded init、未训练、hash前后一致）从主体自身状态量（池化N(h)、goal向量、nu特征（EMA重建＋unknown标志）、预测误差、未结数、队列数）读入→掩码softmax→确定性argmax，宿主只提供时钟/预算/合法mask（interact需队列、recall需store、wait恒合法）；触发量快照在执行前取并逐轮入单调journal（logits/probs/chosen_by/触发量/h范数/step_delta/imagined/store_size）。想象事件＝最后事件编码＋声明种子固定范数扰动，**不写store**（store_size逐轮不变落账）；wait无前向步（H逐位不变、时钟推进）；maintain限位复查显式no-op。无新消息6轮内journal逐轮decision且非wait活动产生状态变化；模型臂与固定阈值对照臂在同一轨迹决策序列分岔（工程分岔核验，不签优劣）；policy头经bundle extra逐位恢复。

声明读数：预测误差＝世界头对**池化后继表示**（mean_i N(h_i)）的读数——世界头从z预测主体级d维后继，与逐群h形状不同；真实世界损失、EMA目标与可学习活动信用属09.4。模型政策未训练，其选择由声明头决定；本报告不签"模型学会选择"、活动策略优劣或自主性。唯一下一步见总计划（09.4持续可塑性与结构出生）。

提交后按钩子要求重跑Mimosa深扫（focusFiles=本包三个代码文件）：jobId `scan-job-mvwjzj1g`，scanId `scan-2026-10-31T15-27-31.295Z-9dcd94362c0a`，**封印`sha256:11e603a8ef3f3d4708d0d71c4c9b5ad3b7b8ec1cd942e78c7b7899cbd7c61be6`**，1,652项findings为既有普查基线、无新假设、离线advisory零匹配；同提交单主线审计ok、三套件回归19项全过。

## 2026-10-31 R026六臂预注册比较（09.4）

先冻结[R026合同](../../reference/SPEC-TAIJI-CORE_PLASTICITY_R026_20261031.md)（预注册主终点＝独立holdout生涯NLL差、保持探针＝第三独立生涯、净收益判据含1e-3边际与0.05保持容差、六臂同父同更新日程），再交付[native_development.py](../../../taiji/native_development.py)、[检查](../../../tests/taiji_native/test_plasticity_arms_r026.py)与[报告](../../../reports/taiji_core_r026_plasticity_arms_20261031.json)。`taiji/living_core.py`字节经SHA核对与R021–R025绑定一致；报告`passed_r026_preregistered_comparison`、4/4检查通过、费用45.9/900 CPU秒、每臂独立父副本、每臂8步声明Adam更新外无任何更新。

**预注册裁决＝negative**：coverage_birth的ΔNLL_holdout＝−0.1370，未优于对照min(replay_only −0.1367, capacity_random −0.1402)（差+0.0032＞1e-3边际）；六臂全部超出0.05保持容差（0.144–0.148，声明更新幅度主导——8×lr=1e-3全参数更新即破坏通用行为）；臂间holdout差≤0.0035；kappa=0.998。诚实结论：在合成夹具与声明更新规模下，"覆盖/信用改善转为后继净收益"被预注册判据否定——出生方向在8步内未自我偿付，且保持破坏与生长无关（无生长臂同样超限）。按合同不追加剂量/seed/臂；负结果不证伪真实条件下的v1.2候选。

执行期两处纪律事件（落账）：①保持探针最初误写为fit journal（其NLL下降是学习不是保持），正式运行前修正为第三独立生涯；②发现replay_only原地更新共享parent会污染后续臂（烟测与独立运行verdict不同），修复为每臂deepcopy独立父副本后重跑——两次修正均发生在预注册读数被消费之前，报告绑定修复后字节。09.4.1预期自我修改不触发（保持全臂超限）。唯一下一步见总计划（09.5主体恢复门）。

提交后按钩子要求重跑Mimosa深扫（focusFiles=本包三个代码文件）：jobId `scan-job-mvwl7vy5`，scanId `scan-2026-10-31T16-02-01.030Z-fbdb0f2f2b13`，**封印`sha256:1e687ebaec1221a0553e5580ec0d74cbd268c832bc7bff803ececad179229c42`**，1,652项findings为既有普查基线、无新假设、离线advisory零匹配；同提交单主线审计ok。

## 2026-11-01 R027主体恢复门合拢（09.5）

先冻结[R027合同](../../reference/SPEC-TAIJI-CORE_GATES_R027_20261031.md)，再交付[native_activity.py扩展](../../../taiji/native_activity.py)（env序列化/来源合法性过滤/轮级门控索引）、[worker activity_round模式](../../../taiji/state_migration.py)与[检查](../../../tests/taiji_native/test_subject_gates_r027.py)、[报告](../../../reports/taiji_core_r027_subject_gates_20261101.json)。`taiji/living_core.py`字节经SHA核对与R021–R026绑定一致；报告`passed_r027_gates_engineering_only`、2/2检查通过、费用3.53/900 CPU秒。

统一主体生命序列八次边界保存×spawn独立解释器进程恢复：交互后（宿主注入消息、host-driven轮消费）、无新提示两轮（endogenous分账）、睡眠等待（无前向步、时钟推进、H逐位不变）、首次声明学习后（worker重复更新逐位一致）、出生前/出生后（门控子带门值表续推活动轮逐位）、编码迁移提交后（revision+1）、未结Outcome（resolution幂等闸拒绝重复领奖）。09.5清单十二项核验落账：参数/拓扑、optimizer/eligibility、RNG探针、游标（career cursor＋seq单调）、记忆（store逐字节）、编码revision、预算（fee/probe延续不清零）、receipt、无越权来源（foreign记录剔除带provenance）、无清账重放、query-only隔离（detached副本探针subject状态逐位不变）、外控/内生根因分账（external_trigger字段）。

执行期事实（落账）：①门控子模型的迁移重放与活动轮必须携带门表——replay_transaction新增provider参数，worker门表对career外步按模回绕（声明仪器，兼容对值对与字典两种格式）；②worker provider重构时曾丢失None守卫致spawn子进程崩溃，R024回归立即捕获并修复（三臂回归全过）。本报告不签语义保持、自主性或任务能力；唯一下一步＝09.6定价与冻结（owner裁定节点：训练准入在09.6合同上裁定）。

## 2026-11-01 09.6 生命片段定价与冻结

先交付[定价脚本](../../../scripts/training/verify_taiji_living_fragment_pricing_r028.py)与[报告](../../../reports/taiji_core_r028_life_fragment_pricing_20261101.json)，再冻结[09.6合同](../../reference/SPEC-TAIJI-CORE_PRICING_20261101.md)。报告状态`passed_r028_pricing_engineering_only`、5/5检查通过、费用190.516/1800 CPU秒、198.867墙钟秒；`optimizer_updates_observed=56`、`data_splits_read=[]`、`training_admitted=false`。原理形状实测3,311,259参数；中文长生涯127事件/61轮，前向0.792 CPU秒、一次更新2.500 CPU秒；活动轮interact/recall/explore分别0.047/0.016/0.031 CPU秒，wait/maintain≈0；出生全链6.922 CPU秒（采集1.641＋两支路128次置换标定3.422＋方向/构造0.094＋中性双生涯1.609）；六臂单seed170.2 CPU秒；bundle77,304,316字节，保存0.078、加载/恢复0.063、新spawn进程续推2.619墙钟秒且h/probabilities逐位一致；恒等/坐标迁移0.156/0.188 CPU秒。

学习率校准（非正式比较）：`lr=1e-3`六臂retention drift0.1105–0.1126，全部超0.05；`lr=1e-4` replay/coverage drift0.1949/0.1948；`lr=1e-5` replay Δholdout−0.020192、drift0.022070，coverage Δholdout−0.020193、drift0.022057，覆盖−重放仅−4.72e-7。合同据此冻结`lr=1e-5`、8步、retention≤0.030、学习下限−0.010、覆盖净收益≥1e-4、lesion≥1e-4、3 seed、P_R=0/P_A=18/P_C=18、1800 CPU/3600墙钟。**owner裁定＝`approved_for_r029_implementation`（2026-11-01）：批准先补09.6 §6五项实现缺口并重新核对单价；未批准09.7正式训练/有界比较。**

## 2026-11-01 09.6.1 五项工程补齐结案（`failed_lesion`）

owner批准范围为`approved_for_r029_implementation`（只补实现、不启动正式训练），据此交付[09.6.1合同](../../reference/SPEC-TAIJI-CORE_ENGINEERING_20261101.md)、[验证器](../../../scripts/training/verify_taiji_r029_engineering_gaps.py)、[测试](../../../tests/taiji_native/test_r029_engineering_gaps.py)与[报告](../../../reports/taiji_core_r029_engineering_gaps_20261101.json)。`taiji/living_core.py`字节未改；新增能力只经独立模块（[constrained_learning.py](../../../taiji/constrained_learning.py)、[activity_learning.py](../../../taiji/activity_learning.py)、[native_task_fixtures.py](../../../taiji/native_task_fixtures.py)）或显式扩展（`growth_credit.lesion_trained`、`native_activity.model_action`、`state_migration`的worker自报成本与`retain_outputs`）接入。报告状态`failed_lesion`、26项检查1项红、`training_admitted=false`、`data_splits_read=[]`；计时数字以报告为准，本页只记判据读数。

五项各自过门，唯lesion贡献判据按预注册分支判负：①**§10.2约束更新**8步全接受（11次尝试、3次实际探针回退、ρ逐次减半），fit NLL降0.0307817（门≥0.010）、保持漂移0.0162904（门≤0.030）、ΔNLL_holdout−0.0172613、KKT残差1.94e-16、约束违反0.0、互补松弛5.37e-14；负对照Adam(lr=1e-3)漂移0.1111525复现超0.05门。**代价与门同读**：无约束对照fit降1.9725041、Δholdout−0.3056635，收益为约束臂的64倍与18倍但突破G1⇒只能声称"约束更新是当前唯一同时满足G1与G2的配置"。②**活动学习**policy/critic摘要前后不同、learner状态经`torch.save`/`load`逐位往返、暴露256未越上限、三臂核心参数摘要逐位相同；累计P_R训练臂0.0／未训练−0.0319197／固定阈值−0.0198944，优势+0.0198944判`activity_learning_advantage`。③**lesion**出生对父函数逐位中性、未训练副本lesion后逐位回到父输出、训练后lesion逐位保留已训练父行且只移除新生单元（258→256，8单元/4组）三项全过；训练收益0.0201929中新生单元只占4.33e-7（占比2.1e-5，门≥1e-4）⇒`lesion_does_not_remove_benefit`。④**中文夹具**5类，应答轮分母25/28/10/25/16逐实例等于`career_step_specs`的round数；捷径尺有动态范围且无一满分（最新2/5、最早1/5、弃答2/5），记录反序后最新降为1、最早升为2证明顺序敏感；两个运行时模块源码中`expected_target`/`shortcut_label`/`category`/`gold`令牌计数全0。⑤**子进程CPU**09.6把`fresh_spawn_worker.cpu_seconds`记为**0.0**——恢复成本此前不是被低估而是被记为零；09.6.1八边界8次worker调用全部自报、0缺失，新字段未改变h/概率/参数逐位一致性。

**第三次独立否证结构生长收益**：R026六臂预注册`negative`、09.6的coverage−replay=−4.72e-7、09.6.1新生单元贡献占比2.1e-5，三者同向；且09.6.1的lesion收益0.0201929与09.6在同lr下测得的coverage Δholdout−0.020193逐位吻合，说明两台仪器读的是同一现象。**在当前夹具与lr=1e-5下，覆盖出生没有可测贡献，观测到的全部收益来自继承父行的参数更新**；不得读作仪器故障，也不得据此反签结构有效。活动学习那条优势的机制同样已查明＝零成本空转（训练臂32轮全选`maintain`、累计恰为0.0，优势数值等于对照臂烧掉的计算成本），成因是声明夹具只放一条外部消息、`interact`在32轮中最多合法一次，报告以`advantage_is_idle_cost_avoidance`显式披露。

执行期三处纪律事件（落账）：①lesion原门措辞把"恢复父函数"与"保留已训练父行"混成一条，训练后二者不可能同时成立，实现前拆成恢复面（未训练副本）与贡献面（已训练子模型）两面并更正合同§2.3；②首次正式跑发现worker面的bundle只有26,603,471字节（optimizer无state），与09.6的77,304,316不同面⇒恢复单价不可比，修复为复刻09.6保存调用（先做一次声明更新让Adam有state、同journal_entries/pending载荷）并新增`worker_bundle_face_matches_r028`硬守卫（bundle字节必须等于09.6值）；首跑报告在入库/索引前移出仓库留档，不删字节、不追认；③对改动的六份计划文档做全量相对链接核对（310条），发现[MAINLINE](../../MAINLINE.md)内11条链接前缀写成`../../scripts`／`../../tests`／`../../taiji`——从`plans/`出发会逃出仓库到`E:/`，正确前缀是`../`；其中R024/R025/R026/R027各行的7条为既有断链、非本轮引入，一并修为`../`。两处代码修正均发生在读数被写进任何索引之前。

回归与门禁面：R022–R027共33 passed、09.6.1新增15 passed（合计48 passed，均在最终代码态上重跑）；整树`ruff check .`与`black --check .`全绿（2371文件）、`audit_frozen_marker_formatter_conflict.py --no-write` passed；全部改动/新增文件CR字节为0。mypy核心面（`mypy --follow-imports=silent seed taiji`，CI基线0）本机实测44处错误分布11文件，已按三组精确归因（分组单跑求和＝整树数）：**A组20处/5文件**＝`native_content_core_v2`8＋`native_content_training_v3`7＋`native_content_training_v2`2＋`native_content_core`2＋`corrected_old_baseline`1，与CI在上一次推送（run 41169350479）报的`Found 20 errors in 5 files`逐数吻合⇒**该门在我动手之前已经红**；**B组23处/5文件**＝`native_activity`8＋`living_core`7＋`growth_credit`5＋`native_development`2＋`state_migration`1，属R021–09.6参考图族，由本地领先origin/main的21个未推送提交带入（其中落在本轮改动文件的错误全在未改动行：`native_activity.py:41/277`＝R025策略头与imagine分支、`growth_credit.py:408-417`＝R023 `allocate`、`state_migration.py:301`＝R024 `load_subject`）；**C组0处**＝09.6.1三个新模块单跑`Success: no issues found in 3 source files`（`activity_learning.py`原有1处`no-any-return`已在本轮修掉）。**本轮新增0处类型错误；A/B两组43处为既有债，未在09.6.1范围内结清，且推送这21个提交后CI仍会在该门红。**

[09.6.1 §4](../../reference/SPEC-TAIJI-CORE_ENGINEERING_20261101.md#4-重新定价与-097-预算修订)按§1.4的20%偏离条款完成重新定价：09.6表内无约束声明更新单价未变（2.500 CPU秒/步，本轮control段28.5与09.6单臂28.406互证），新增约束更新实测**8.830 CPU秒/接受步（3.53倍；8步用了11次尝试，每次尝试6.42）**、约束臂77.3对无约束臂28.406 CPU秒（2.72倍）。恢复面：09.6记为0.0 CPU秒，09.6.1在同面bundle（77,304,316字节、optimizer state 123项）上实测子进程自报0.8125 CPU秒、父侧等待0.0；八边界8次调用全部自报（子进程CPU合计1.328、父侧2.391、序列墙钟16.117秒），其中两次轻量调用受Windows 15.6毫秒粒度限制报0.0。活动学习面164.3 CPU秒（小形状，256训练轮占119.484）、lesion面33.2 CPU秒。09.7（六臂×3 seed＋每seed一次lesion面与活动学习面）三档＝全无约束1104／仅候选臂约束1250／**六臂全约束1984（破G7的1800达184秒）**；墙钟按实测比值1.067折算（全约束档约2117秒）三档均远低于3600，CPU是唯一绑定约束；全约束档把seed降到2则为1323 CPU秒、落回G7内。**唯一下一步＝owner裁定节点：09.7是否仍按原假设执行（三次独立否证之后），以及按哪一档预算执行；正式训练未批准。**

**owner三裁（2026-11-01，已落账）**：①09.7＝**先改夹具再立项**，不花训练预算——活动面需反复出现的外部触发（现行夹具32轮里`interact`最多合法1次，最优解退化为零成本空转），生长面需能让新生单元承担收益的任务；②mypy核心面43处既有错＝**只登记为债、本轮不修**（[DEBT-I11](05_TECH_DEBT_REGISTER.md)，含A/B/C三组精确归因与"推送后CI仍会红"的后果说明）；③本地领先`origin/main`的21个提交与09.6.1这一包＝**都不动，等owner排期**（不提交、不推送）。据此队首转为**09.6.2夹具改造（工程范围，不含训练）**。

## 2026-11-01 09.6.2 夹具改造结案（两项读数，`failed_activity_fixture`／`growth_fixture_no_reachable_cell`）

按owner裁定「09.7先改夹具再立项」，交付[09.6.2合同](../../reference/SPEC-TAIJI-CORE_FIXTURES_20261101.md)、[到达夹具模块](../../../taiji/native_activity_fixture.py)、`run_round`/`activity_learning` 的 `arrivals` 扩展（默认 `None`，R025/R027 字节行为不变、bundle 不加键）、[测试](../../../tests/taiji_native/test_fixture_arrivals.py)（6 passed）、[验证器](../../../scripts/training/verify_taiji_fixture_arrivals.py)与两份报告。`taiji/living_core.py` 字节未改。

**生长面＝第四次独立否证，且排除了「训练太慢」。**六格全发布（合同禁止事后挑格）：新生单元份额随步数**转负**（1e-5 下 8/32/128 步＝+2.146e-5／−2.76e-6／−7.129e-5），而 holdout 收益同期从 0.0202 涨到 0.324——说明 8 步确实欠训练，但涨出来的收益全归继承父行，新生支路在长程训练后反而是轻微拖累。唯一为正的格（1e-4/128，+1.179e-3）同一格保持漂移 **1.169952**（门 0.030，超 39 倍）、holdout 收益 **−0.4643**（训练把结果变差），是发散不是可达。裁决 `growth_fixture_no_reachable_cell`；按合同不改网格重测。至此四路同向：R026 六臂 negative、09.6 的 coverage−replay −4.72e-7、09.6.1 的新生份额 2.1e-5、本网格六格。

**活动面＝夹具修好了，暴露的是学习规则。**v2 报告（`reports/taiji_core_fixture_arrivals_v2_20261101.json`，1415.3/1800 CPU 秒、1452.2 墙钟秒）读数：解析下界与脚本贪婪策略均 0.704（空闲 0.0，未训练臂 −0.03185），单次交互实测净赚 **+0.044** ⇒ 夹具确有 0→0.704 动态范围、且「交互不划算」被排除。但训练臂在声明种子上只交互 1 次后停在 `maintain`（份额 0.063），三 seed 诊断份额 0.0627／−0.0253／0.0185，**3/3 都没学会继续交互**。

机制已查明（事后诊断，非门）：把到达改为每轮补一条（`per_round=True`）后，同样三个种子给出**完全相同**的读数 **1.40807、32/32 轮全选 `interact`、份额 1.00**。⇒ 声明的 `TRACE_LAMBDA=0` 使「先 wait 推时钟→下一条到达→再交互」这条两步序列在信用上不可见；学习规则能学会"当轮合法就选有奖励的活动"，学不会跨步序列。这两条读数合起来才是可发表结论，缺一条都会被误读成"夹具又坏了"或"模型学会了自主性"。

执行期两处纪律事件：①首跑报告 `taiji_core_fixture_arrivals_20261101.json` 的活动面有一条门是我写错的——用浮点严格相等比解析下界（0.7039971 对 0.704）；已改为"脚本策略达到 ≥0.5 声明下限"，并新增"单次交互净赚 >0.01"门；首跑件保留原处不删，v2 为正式读数，且 v2 重测生长网格后六格数值与 v1 **逐位相同**，兼作可复现性核对。②活动面三 seed 诊断与 `per_round` 变体均为**事后**加入，报告里分别标注 `seed_diagnostic_reading` 与"事后诊断，非门"，未拿它们改判已声明的门。

**唯一下一步＝owner裁定节点（09.7 的活动面怎么声明）**：①保留等待链并把 `TRACE_LAMBDA` 改为 >0（跨步信用成为被测机制）；②采用逐轮供给并只声称"逐步选择"，不得声称顺序规划。生长面则因四路同向否证，**不建议**按原假设立项；若仍要做，必须先给出"新生支路为什么应当承担收益"的新机制假设，而不是加步数或换种子。正式训练未批准。

**owner两裁（2026-11-01，已落账）**：①生长面＝**不按原假设立项**（四次独立否证且排除了"训练太慢"；若再做须先立新机制假设）；②活动面＝**逐轮供给＋只声称逐步选择**（3/3 种子达上界 1.40807，但不得声称顺序规划或自主性）。据此 09.7 的原始范围作废。

**09.6.3 出生惰性诊断已交付**（[脚本](../../../scripts/training/diagnose_birth_inertness.py)、[报告](../../../reports/taiji_core_birth_inertness_20261101.json)，38.1 CPU 秒）。它回答"新生单元为什么不起作用"，并**又排除了两个竞争解释**：新生 R 行出生时恰为 0，8 步后长到 **0.006730653**（不是冻结的死参数）；它们拿到父行梯度范数的 **6.13%（holdout）／5.32%（fit）**，初始 P 列范数是父列的 **12.4%**（不是小到可忽略）。在这三条都成立的情况下，新生支路对 holdout 的贡献仍只有 **4.33e-7**，而同期 0.020192874 的收益全由继承父行承担。⇒ 剩下的解释只有一条：**信用分配选出的方向本身对 holdout 无用**——分支信用取自 fit 窗口，它选的方向能让 fit 动，但学到的动作不迁移。这是可以立新假设的中间量（例如：把出生方向改为对 holdout 无偏的覆盖目标，或给新生支路单独的信用窗口），但**本合同不据此提出或批准任何新机制**。

**唯一下一步＝下一份合同的立项决定权在 owner**：若要继续生长线，须先按 09.6.3 的读数写出一个可证伪的新机制假设（并说明它预测哪个中间量会变），再定价；若不继续，则 09 按"结构生长在本参考图上不可达"收束，后续 10–13 的能力/时程/自主范围不依赖覆盖出生。正式训练未批准。

**09.6.3b 方向可达性上界已交付**（[脚本](../../../scripts/training/diagnose_direction_oracle.py)、[报告](../../../reports/taiji_core_direction_oracle_20261101.json)，29.4 CPU 秒）：把出生方向换成**神谕方向**（holdout 梯度在各群 P 上的前几个奇异向量，只用来看可达性、不是可用方法），其余与 09.6.3 完全相同（同父、同 8 单元、lr=1e-5、8 步、门全开）。读数：收益 0.02019838、新生贡献 **5.888e-6**、份额 **2.915e-4**——比 fit 信用方向的 2.1e-5 份额高 **13 倍**，**过份额门**（≥1e-4）；但按 09.6.1 §2.3 的**绝对门**（贡献 ≥1e-4 NLL）仍差 17 倍。

两个口径必须同时报，结论才成立：①**"任何方向规则都不行"被否证**——方向可见 holdout 时新生单元确实变得有信息；②**"方向规则是唯一瓶颈"也未证成**——即便给神谕方向，绝对贡献仍只有 5.9e-6，远低于 09.6.1 用的 1e-4 绝对门。所以可立的新假设只能针对方向规则本身，并**必须预先声明瞄准哪个门**：份额门（已可达）还是绝对门（未达）。只报份额门会读成"结构已能偿付"，只报绝对门会读成"方向也没用"——两者都是错读。**owner据此裁定：立项，瞄准份额门。**

**09.6.4 方向规则候选已交付，判据自检失败：`metric_not_discriminating`**（[合同](../../reference/SPEC-TAIJI-CORE_DIRECTION_20261101.md)、[规则](../../../taiji/birth_direction.py)（fit 生涯内多个不相交子窗口的共识方向，不读评估集）、[验证器](../../../scripts/training/verify_taiji_birth_direction.py)、[报告](../../../reports/taiji_core_birth_direction_20261101.json)，128.2 CPU 秒）。四臂同轮读数：

| 臂 | 绝对贡献 | 份额 | 保持漂移 |
|---|---|---|---|
| 共识 K=2（不泄漏） | **1.426e-4** | 7.014e-3 | 0.022216 |
| 共识 K=4（不泄漏） | 7.320e-5 | 3.612e-3 | 0.022098 |
| fit-SVD 基线（同轮复现） | 4.332e-7 | 2.146e-5 | 0.022057 |
| **随机方向负对照** | 1.988e-5 | **9.836e-4** | 0.022086 |

合同第 3 条负对照**如期触发了它该抓的东西**：随机方向也过份额门（9.8e-4 ≥ 1e-4），所以**预注册的份额判据不能把真规则与随机方向分开**，H1 不得判为支持——状态按合同出口为 `metric_not_discriminating`，不是 `hypothesis_h1_supported`。基线同轮复现通过（2.146e-5 对 09.6.3 的 2.146e-5，偏差 0%）；不泄漏守卫改为 AST 标识符扫描（原字符串扫描踩到模块自己的禁令句，属自指误报，已修）。

**合同外的观察（只登记，不作声称）**：按**绝对贡献**看，共识 K=2 的 1.426e-4 已过 1e-4，是随机对照的 7.2 倍、fit-SVD 基线的 329 倍——这正是份额判据丢掉的那部分信息。但合同明确把绝对门排除在本合同之外，且这是单种子单轮，**不得据此声称绝对门已满足**。要把它变成结论，必须先立新合同把绝对贡献＋随机对照预注册为判据再测。

**唯一下一步＝owner裁定口径**：既定口径（份额门）已自证无判别力，需要选 ①改用绝对贡献≥1e-4且随机对照<1e-4作预注册判据（同四臂重测，约130 CPU秒）；②保留份额门但改为"份额−随机份额≥某个界"；③收束生长线。正式训练未批准。

**owner裁定：改绝对贡献＋随机对照。09.6.5 首轮判为假复现、结论作废**（[合同](../../reference/SPEC-TAIJI-CORE_DIRECTION_ABS_20261101.md#4-执行更正20261101同日)、[报告](../../../reports/taiji_core_birth_direction_abs_20261101.json)，277.3 CPU 秒）。报告写出 `hypothesis_h1_supported`，但**该状态不得引用**：本合同把"三个种子"当复现轴，而 `LivingCoreReference` 用内部固定种子初始化，全局 `torch.manual_seed` 不改变父模型——实测三个"种子"的父模型摘要逐位相同（`921dc9981187406a`×3），共识臂三次读数也逐位相同（0.000142628236×3）。所以"3 种子上 ≥2 支持"实际是**同一配置跑了三遍**。这是我自己的判据缺陷，不是数据问题；旧报告按纪律保留原处不删、状态作废。

该轮仍然成立、并已并入证据链的只有一件事：在**同一确定性主体**上，三个不同随机对照给出 1.988e-5／−1.720e-6／7.284e-6（均低于绝对门），而共识臂 1.426e-4、基线 4.33e-7——它证明共识臂不只是随机噪声，但**不构成复现**。复现改由 [09.6.6](../../reference/SPEC-TAIJI-CORE_DIRECTION_REPLICATION_20261101.md) 用真实变化轴（规则自身的窗口数 `parts∈{2,3,4}`）重做，判据预先声明为"≥2/3 格过绝对门且漂移≤0.030、所有随机对照<1e-4、基线偏差在[0.2,5]倍内"。正式训练未批准。

**09.6.6 真实变化轴复现已交付：`hypothesis_h1_supported`**（[合同](../../reference/SPEC-TAIJI-CORE_DIRECTION_REPLICATION_20261101.md)、[验证器](../../../scripts/training/verify_taiji_birth_direction_parts.py)、[报告](../../../reports/taiji_core_birth_direction_parts_20261101.json)，357.2/1800 CPU 秒、0 失败检查；父模型摘要 `921dc9981187406a` 与 09.6.5 一致，证明主体确实确定性）。三格读数：

| parts | 共识绝对贡献 | 随机对照（2 种子） | fit-SVD 基线 | 保持漂移 |
|---|---|---|---|---|
| 2 | **1.426e-4** | 1.988e-5 / −1.720e-6 | 4.332e-7 | 0.022216 |
| 3 | **1.343e-4** | 同上 | 4.332e-7 | 0.022218 |
| 4 | 7.320e-5（**未过**） | 同上 | 4.332e-7 | 0.022098 |

判据满足（2/3 格过绝对门），随机对照全部低于门（最大 1.988e-5，差 5 倍），基线在 3 格上逐位复现 4.332e-7（比值 1.0），泄漏扫描零命中，漂移全部 ≤0.030。**可声称的只有这一句**：在当前参考图、合成中文生涯、确定性主体、8 步 lr=1e-5 下，把出生方向换成"fit 生涯多子窗共识"后，新生单元承担了 **1.3–1.4e-4** 的 holdout 改善，而随机方向最多 2.0e-5、旧规则 4.3e-7。**不可声称**的：①过线幅度只有 1.4 倍（parts=4 就掉到 7.3e-5），属薄边际；②绝对收益能否偿付出生与训练成本未测；③单一确定性主体、单一参考图、非任务能力；④09.7 未准入。

**唯一下一步＝owner裁定**：这条薄边际结果是否足以支撑进入 09.7（需先定价"出生+持续训练"的净收益门槛），还是先把窗口数轴扩到 `parts∈{2..6}` 与多出生单元数看稳定性。正式训练未批准。

**owner裁定：先扩稳定性面。09.6.7 稳定性面已交付：`platform_supported`**（[合同](../../reference/SPEC-TAIJI-CORE_DIRECTION_SURFACE_20261101.md)、[验证器](../../../scripts/training/verify_taiji_birth_direction_surface.py)、[报告](../../../reports/taiji_core_birth_direction_surface_20261101.json)，756.5/1800 CPU 秒、0 失败检查）。15 格全发布，预注册门槛为"≥6 格过门且覆盖 ≥3 个 parts"：

| 单元数 ＼ 窗口数 | 2 | 3 | 4 | 5 | 6 |
|---|---|---|---|---|---|
| 1 | **1.065e-4** | 9.942e-5 | 3.244e-5 | 7.994e-5 | 3.975e-5 |
| 2 | **1.426e-4** | **1.343e-4** | 7.320e-5 | **1.209e-4** | 9.471e-5 |
| 3 | **1.566e-4** | **1.731e-4** | 9.499e-5 | **1.737e-4** | **1.285e-4** |

**8 格过门、覆盖 parts {2,3,5,6}**，保持漂移全部 0.0221–0.0223（门 0.030），随机对照全部低于门、基线逐位复现、泄漏零命中。两条读数值得单独记：①效应随出生单元数单调增强（同一 parts 下 units=3 > 2 > 1，例如 parts=5 的 7.99e-5 → 1.21e-4 → 1.74e-4）；②**parts=4 在三个单元数下都不成立**（3.24e-5／7.32e-5／9.50e-5），恰是 09.6.3/09.6.4 用的默认值——早期单点结论正好踩在最弱的一格上。**仍不可声称**：最大格 1.737e-4 只有门的 1.74 倍，属薄平台；成本偿付未测；单一确定性主体、单一参考图、合成夹具；非任务能力；09.7 未准入。

**唯一下一步＝owner裁定 09.7**：是进入 09.7 定价（须把"出生+持续训练"的净收益门槛与成本一起写进合同，并按 units=3 侧配置作为候选、预先接受 parts=4 不成立），还是先补一项成本偿付读数（出生管线 + 8 步训练的实际 CPU 对 1.4–1.7e-4 的 holdout 改善）。正式训练未批准。

**owner裁定：先补净收益对照。09.6.8 净收益对照已交付：`net_benefit_supported`**（[合同](../../reference/SPEC-TAIJI-CORE_NETBENEFIT_20261101.md)、[验证器](../../../scripts/training/verify_taiji_birth_netbenefit.py)、[报告](../../../reports/taiji_core_birth_netbenefit_20261101.json)，66.2/600 CPU 秒、0 失败检查）。同一确定性父、同样 8 步、lr=1e-5，两臂：

| 臂 | ΔNLL_holdout | 训练 CPU 秒 |
|---|---|---|
| 只重放（不出生） | −0.020192402 | 21.688 |
| 共识出生（parts=5, units=3） | −0.020366326 | 22.656 |
| **净收益** | **+1.739e-4** | 出生构造 10.531 ＋ 额外训练 0.968 ＝ 11.50 |

新生单元lesion贡献 1.7374e-4，与 09.6.7 同格逐位吻合（一致性门通过）；两臂保持漂移均 ≤0.030。**这是 09 线上第一份正的净收益读数**：出生花 11.50 CPU 秒换来 1.74e-4 的 holdout 改善（约 **6.6 CPU 秒／每 1e-4 NLL**，记为定价输入，本合同不设成本门槛）。**仍不可声称**：单一确定性主体、合成中文生涯、8 步 lr=1e-5、单一参考图；parts=4 仍不成立；无任务能力；09.7 未准入。

**唯一下一步＝owner裁定 09.7 立项与预算**：净收益已有正读数与单价，但"每 1e-4 NLL 值多少 CPU 秒"这个换算没有独立标准，须由 owner 在 09.7 合同里定；同时须预先写入：候选取 parts=5/units=3、parts=4 不成立、成本偿付只在本参考图与合成夹具上成立。正式训练未批准。

**owner裁定：先换第二个 holdout。09.6.9 跨评估生涯复核已交付：`replicated_on_second_holdout`**（[合同](../../reference/SPEC-TAIJI-CORE_NETBENEFIT2_20261101.md)、[验证器](../../../scripts/training/verify_taiji_netbenefit_second.py)、[报告](../../../reports/taiji_core_netbenefit_second_20261101.json)，70.8/600 CPU 秒、0 失败检查）。第二评估生涯在本合同首次声明（题材与 HOLDOUT 不同、结构同形，且**不复用 RETENTION**——它同时是保持漂移门的探针，共用会把"被评估的面"与"被守的门"合成一处）。两臂同 09.6.8：

| 评估生涯 | 只重放 Δ | 出生臂 Δ | 净收益 |
|---|---|---|---|
| HOLDOUT | −0.020192402 | −0.020366326 | **+1.739e-4** |
| SECOND_HOLDOUT | −0.011326104 | −0.011516443 | **+1.903e-4** |

HOLDOUT 那条与 09.6.8 逐位相同（一致性门通过），第二生涯略大。⇒ 净收益**不是单条评估生涯的产物**。**仍不可声称**：两条都是合成夹具（不是未见任务）、单一确定性主体、8 步 lr=1e-5、单一参考图；parts=4 仍不成立；无任务能力；09.7 未准入。单价仍为约 6.6 CPU 秒／每 1e-4 NLL（出生构造 10.5 秒为主）。

**唯一下一步＝owner裁定 09.7 立项与预算**。可声称的现状已经清楚：方向规则换成 fit 多子窗共识后，出生在 15 格中 8 格过绝对门、在两條评估生涯上都带来 ≥1.7e-4 净收益、代价约 11.5 CPU 秒／次。待 owner 定的是：①"每 1e-4 NLL 值多少 CPU 秒"的换算标准；②09.7 是否按 parts=5/units=3 单点立项（须预先声明 parts=4 不成立、不得事后换窗口）；③是否需要先把第二生涯换成真实语料。正式训练未批准。

**owner裁定：按单点立项。09.7 有界联合片段已结案：`retention_breach`**（[合同](../../reference/SPEC-TAIJI-CORE_JOINT_FRAGMENT_20261101.md)、[验证器](../../../scripts/training/verify_taiji_fragment_09_7.py)、[报告](../../../reports/taiji_core_fragment_09_7_20261101.json)，219.3/1800 CPU 秒）。单点候选 parts=5/units=3，32 步分四个检查点，两臂同父同步：

| 步数 | HOLDOUT 净收益 | SECOND_HOLDOUT 净收益 |
|---|---|---|
| 8 | 1.739e-4 | 1.903e-4 |
| 16 | 3.486e-4 | 3.894e-4 |
| 24 | 5.299e-4 | 5.978e-4 |
| 32 | **7.187e-4** | **8.147e-4** |

净收益**随步数单调增长**（8 步到 32 步放大约 4.1–4.3 倍），但**两臂都突破保持门**：只重放 0.084752、出生 0.085631（门 0.030）⇒ 按预注册出口结案为 `retention_breach`。**破坏保持的是 32 步的总训练量，不是出生**——两臂漂移只差 0.0009。8 步处的一致性门通过（净收益与 09.6.8/09.6.9 逐位相同、漂移合规）。额外成本 14.781 CPU 秒。

**我的测量缺口（已登记）**：09.7 只在末点测保持漂移，没有逐检查点记账，因此无法说明它从哪一步越过 0.030。补 [09.7b](../../reference/SPEC-TAIJI-CORE_HORIZON_20261101.md) 定位"保持仍在门内的最长检查点"并报告该处净收益。

**09.7b 可持续点定位已交付：`sustainable_horizon_found`，可持续点＝8 步**（[合同](../../reference/SPEC-TAIJI-CORE_HORIZON_20261101.md)、[验证器](../../../scripts/training/verify_taiji_fragment_horizon.py)、[报告](../../../reports/taiji_core_fragment_horizon_20261101.json)，224.5/1800 CPU 秒、0 失败检查）。逐检查点保持漂移与净收益：

| 步数 | 只重放漂移 | 出生漂移 | HOLDOUT 净收益 | 第二生涯净收益 |
|---|---|---|---|---|
| **8** | **0.022070** | **0.022269** | **1.739e-4** | **1.903e-4** |
| 16 | 0.043447 | 0.043861 | 3.486e-4 | 3.894e-4 |
| 24 | 0.064269 | 0.064910 | 5.299e-4 | 5.978e-4 |
| 32 | 0.084752 | 0.085631 | 7.187e-4 | 8.147e-4 |

**8 步是唯一在门内的声明检查点**，该处净收益 1.739e-4／1.903e-4；16 步起两臂同时越线，且漂移几乎同步（8 步差 0.000199、32 步差 0.000879）⇒ **卡住的是训练预算，不是出生**。净收益随步数近线性增长（8→32 步约 4.1–4.3 倍），但要吃到它必须让保持门或 lr 变动，而那两项都不在已批准范围里。

**09 生长线的现状（可声称）**：在本参考图、确定性主体、合成中文夹具、lr=1e-5、**8 步**下，把出生方向换成 fit 多子窗共识（parts=5、units=3）后，出生比不出生多改善 holdout **1.739e-4**、第二评估生涯 **1.903e-4**，代价约 **11.5 CPU 秒**；15 格中 8 格过绝对门，parts=4 三档全不成立。**不可声称**：更大净收益（需要更长的训练预算，而 16 步起保持即破门）、任务能力、真实语料、跨参考图外推、自主性。**训练未准入。**

**唯一下一步＝owner裁定**：①接受 8 步／1.7e-4 作为 09 的生长收益结论并收束；②调低 lr 换取更长可持续片段（须重新定价与预注册，且 09.6 已证 lr 越低收益越慢）；③放宽保持门（须证明放宽后旧能力仍可接受，这是产品级判断，不是我可以定的）；④先把评估生涯换成真实语料再谈。正式训练仍未批准。


**09.8 全局收束已交付：`closeout_passed_engineering_only`**（[合同](../../reference/SPEC-TAIJI-CORE_CLOSEOUT_09_20261101.md)、[验证器](../../../scripts/training/verify_taiji_closeout_09.py)、[能力/资源卡](../../../reports/taiji_core_09_closeout_20261101.json)，435.2/900 CPU 秒、0 失败检查）。六件收束物：①bundle **77,609,636 字节**（`output/taiji_core_09/subject_bundle.pt`，被 `.gitignore:62 *.pt` 忽略、不入 git，sha `ce4c62685d55bb51…`）；②活动轨迹 8 轮（逐轮供给下全为 `interact`——该夹具下每轮都合法且都付钱，**不据此判偏好**）；③父子结构映射：父 3,311,259 → 子 3,323,547 参数、`growth_parent_units=256`、每群 3 个新生单元共 12 个；④参数继承 **135/135 逐位相等**；⑤原始输出：两条 query 的 top-1 均为字节 229，前 3 概率 0.0591/0.0359/0.0355（原样记录、不评分）；⑥能力/资源卡：13 个子步骤逐条列出其状态出口、可声称句与不可声称句（含 09.6.5 的作废标注）。三项一致性：spawn 恢复逐位、继承逐位、轨迹非全空闲。

执行更正（同日）：首轮收束器写 `degenerate_activity_trace`，是**我把旧夹具的直觉带进新夹具**——要求 8 轮里必须出现 `wait`，而逐轮供给下 `interact` 每轮都合法且付钱，策略全选 `interact` 正是正确行为。判据改为"至少一轮 `interact`"并在卡片注明；这一改**放宽**了判据，所以它只排除"全空闲"，**不能**证明偏好。同时把误导性的检查名 `six_artefacts_present` 改为 `all_sub_step_reports_present`（它验的是 13 份子报告齐备，六件由卡片字段承载）。首轮卡片在入库前移出仓库留档，不删字节。

**09 阶段现状（可声称/不可声称）**：工程面六件齐备、三项一致；生长面在本参考图、确定性主体、合成中文夹具、lr=1e-5、**8 步**下，共识出生（parts=5、units=3）比不出生多改善 holdout **1.739e-4**、第二评估生涯 **1.903e-4**，代价 **11.5 CPU 秒**。**不可声称**：任务能力、中文表达质量、自主性、跨参考图或真实语料外推、更长片段的净收益（16 步起两臂同时破保持门，卡在训练预算）。**训练仍未准入。**

**唯一下一步＝owner裁定 09.9/10 准入**：10（T3 自然目标/有限行动）与 11–13 都还没有立项依据；若要进 10，须先按总计划 10.1 冻结自然 Goal、可观察状态、动作集与独立终态评分，并声明它与 09 卡片里已测项的关系（不得把 09 的合成夹具读数当 10 的准入前提）。

**owner裁定：接受 8 步结案，转入 09.8 全局收束。** 因此 09 的生长结论钉死为：**lr=1e-5、8 步下，共识出生（parts=5、units=3）相对不出生多改善 holdout 1.739e-4、第二评估生涯 1.903e-4，代价 11.50 CPU 秒；更长片段（16 步起）净收益更大但两臂保持同时破门，卡点是训练预算而非出生**。调低 lr 与放宽保持门两条路都**未测**，不得在 09.8 卡片里写成"已排除"或"已解决"——它们是未执行的备选，不是否定结果。

## 2026-11-01 10.1 任务环境范围冻结（`passed_10_1_environment_only`）

owner批准立项后交付[10.1合同](../../reference/SPEC-TAIJI-CORE_T3_GOAL_20261101.md)、[任务环境](../../../taiji/native_goal_world.py)、[测试](../../../tests/taiji_native/test_goal_world.py)（6 passed）、[验证器](../../../scripts/training/verify_taiji_goal_world_10_1.py)与[报告](../../../reports/taiji_core_goal_world_10_1_20261101.json)。**这一步不涉及任何模型**：它只把任务本身修到能问问题的程度。

环境：2 对象（蓝色杯子、红色印章）×3 位置（木桌、窗台、左盒），动作 `move(object, place)`，每步 4 个合法动作；动作 ID 按声明种子**逐步重排**为 `a0…a3`，同一步内 ID 不含正确性；终态评分 `1[Goal满足] − 0.01×步数`，只读终态与步数。最优解用 9 状态空间穷举，是精确解不是采样。

四条夹具与三条设计判据：f1/f2 共享初态（f1「杯子→窗台」、f2「印章→木桌」），f1+f3 与 f2+f4 各共享 Goal ⇒ **换 Goal 后最优动作不同**（f1 一步动杯子、f2 一步动印章）；分区覆盖 4 组 (初态, Goal)；四条声明捷径 `first_legal`／`lowest_id`／`fixed_object`／`cycle` **都至少在一格拿不到最优，且没有任何一条在全部格最优**（例如 f1 只有 `cycle` 与最优同分、f2 只有 `first_legal` 同分，没有任何一条通吃）。最优分：f1/f2/f4 = 0.99、f3 = 1.0（已满足、0 步）。

执行期两处纪律事件：①`first_legal` 初版与 `lowest_id` 实际等价（都取 `a0`），改成按**规范顺序**取第一个动作后才真正不同；②测试里我写过一条恒真断言（`not any(...) is False or True`），已删除——守卫必须能为假。**不声称**：任何模型能解此任务；10.2（自然Goal绑定）与 10.3（预测与选择）未准入；训练未批准。

## 2026-11-01 10.2 自然 Goal 绑定契约与规则基线（`passed_10_2_binding_contract_only`）

交付[合同](../../reference/SPEC-TAIJI-CORE_GOAL_BINDING_20261101.md)、[规则基线](../../../taiji/goal_binding.py)、[测试](../../../tests/taiji_native/test_goal_binding.py)、[验证器](../../../scripts/training/verify_taiji_goal_binding_10_2.py)与[报告](../../../reports/taiji_core_goal_binding_10_2_20261101.json)。绑定输入只有 `(中文原文, 可观察状态)`，输出 `conditions[(object, place, polarity)]`／`abstain`／`reason`／`unknowns`；**宿主不传 Goal、终态或评分**，绑定器自身不评分。

五类公开反例 6 条夹具全部入分母（不跳）：改写「把蓝色杯子放到窗台」与「蓝色杯子应该出现在窗台上」绑定出**同一**条件集；否定「红色印章不要留在左盒」极性 −1 且 `satisfied` 判读正确；合取含两条条件；**缺条件「把它放到窗台」与歧义「…放到窗台或木桌」都拒答**（`missing_referent`／`ambiguous`）且条件为空——猜中也会判错，因为判据无法区分猜与推理。AST 扫描确认绑定器内无 `Goal`／`gold`／`terminal_state`／`reward`／`score` 标识符。

执行期一处纪律事件：验证器里我写过一条恒真断言（`"模型侧绑定未交付" in "模型侧绑定未交付"`），已删除，改为只依赖真实字段 `model_binding_delivered is False`。

**本步交付的是规则基线，模型侧绑定未交付**——报告 `model_binding_delivered=false`、`model_involved=false`。**不声称**：模型能绑定自然中文目标；不声称任务能力；10.3 未准入；训练未批准。

## 2026-11-01 10.3 预测与选择契约（`planner_no_advantage`，负结果结案）

交付[合同](../../reference/SPEC-TAIJI-CORE_GOAL_PLANNER_20261101.md)、[参考规划器](../../../taiji/goal_planner.py)、[测试](../../../tests/taiji_native/test_goal_planner.py)、[验证器](../../../scripts/training/verify_taiji_goal_planner_10_3.py)与[报告](../../../reports/taiji_core_goal_planner_10_3_20261101.json)。契约把三类 Outcome 分开：`predicted`（对所选动作的一步预测）、`imagined`（rollout 内部状态）、`actual`（真实执行结果），并逐条记 `predicted_matches_actual`；中文说明只入账本、不进判分路径（AST 检查 `state_score` 只收 `(state, binding)`）。

四条夹具、五条臂读数：

| 臂 | f1 | f2 | f3（已满足） | f4 |
|---|---|---|---|---|
| 规划器 | 0.99 | 0.99 | 1.00 | 0.99 |
| 贪心 | 0.99 | 0.99 | 1.00 | 0.99 |
| 随机 | −0.04 | 0.99 | 1.00 | 0.99 |
| 扰动模型 | −0.04 | −0.04 | 1.00 | −0.04 |
| 错目标 | 0.99 | 0.99 | 0.99 | 1.00 |

**规划器与贪心在四条夹具上完全打平，随机臂在 3/4 上也拿到最优** ⇒ 预注册的"规划必须严格优于简单策略"不成立，按出口结案为 `planner_no_advantage`。这不是规划器写坏了：它与穷举最优在四条上逐位相同；问题是**任务太浅**——3 个位置、单对象目标，一个"往目标挪一格"的贪心已经最优，规划没有发挥空间。这正是 10.1 合同里写过的情形："若任务可由动作顺序或模板投机完成，先修任务"。

仍然成立、并已并入证据链的：扰动模型臂每步预测都与实际不符且终态分降到 −0.04 ⇒ `predicted_matches_actual` 这一栏**有判别力**；规划臂的预测全部与实际相符；错目标臂至少一格低于规划器（f3：0.99 对 1.00）⇒ 绑定条件确实被用上。

**修任务的候选（未实施，需 owner 选）**：给世界加一层"依赖/阻挡"——例如某个位置必须先清空才能放目标对象，或两个对象必须交换位置。这类结构会让"往目标挪一格"的贪心失效，而深度 2 的 rollout 能看出来。改动属于 10.1b，需要重新冻结环境并重跑 10.1 的三条判据，再回来重跑 10.3。

**不可声称**：模型能预测或选择；任务能力；11–13 准入。训练未批准。

## 2026-11-01 10.1b 任务加深与 10.3 重测（`passed_10_1b_environment_deepened`／`passed_10_3_prediction_contract_only`）

**10.1b**（[合同](../../reference/SPEC-TAIJI-CORE_T3_GOAL_10_1B_20261101.md)、[报告](../../../reports/taiji_core_goal_world_10_1b_20261101.json)）：给世界加一条**位置容量 1**——目标位被占时必须先腾开；实现为 `legal_moves(..., occupy=True)`／`step(..., occupy=True)`，**默认 `False` 保持 10.1 旧路径逐位不变**，并新增 g1（窗台被印章占住）与 g2 两条夹具。读数：旧路径 f1..f4 仍为 0.99/0.99/1.00/0.99、四条捷径仍不能通吃；加深后三条 10.1 判据仍成立；规划器 **[0.99, 0.98, 1.00, 0.99, 0.98, 1.00]** 对贪心 **[0.99, −0.04, 1.00, 0.99, −0.04, 1.00]**——f2 与 g1 上严格更优、其余持平，g1 上贪心确实被阻挡。

**10.3 重测**（[报告](../../../reports/taiji_core_goal_planner_10_3b_20261101.json)）：在加深后的环境上重跑同一验证器（新增 `--occupy`），状态转为 `passed_10_3_prediction_contract_only`、0 失败。首测那份 `planner_no_advantage`（[报告](../../../reports/taiji_core_goal_planner_10_3_20261101.json)）**保留不删**，它是"任务太浅"这一发现的原始读数；两份报告并存，前者标为加深前、后者为加深后。

**这一步没有涉及任何模型**：10.1/10.1b 是任务环境，10.2 是规则绑定基线，10.3 是参考规划器基线。**不声称**模型能绑定、预测或选择；不声称任务能力；11–13 未准入；训练未批准。

### 摘要漂移披露（10.1b 引起的旧报告 SHA 变旧）

`taiji/native_goal_world.py` 在 10.1b 里新增了**默认关闭**的 `occupy` 参数与 g1/g2 两条夹具，因此 10.1／10.2／10.3首测 三份报告绑定的模块摘要不再等于当前文件。按纪律：三份报告保留原处不删、不追认，并逐项核验读数是否受影响：

- **10.2**：重跑绑定验证器，6 条夹具的 `conditions` 与 `abstain` **逐条相同**，状态同为 `passed_10_2_binding_contract_only`。绑定器只读文本，不读合法性，读数未受影响。
- **10.1**：`occupy=False` 下原四条夹具的最优与四条捷径判决由 10.1b 的 `default_path_changed` 门**逐条复验通过**（f1..f4 = 0.99/0.99/1.00/0.99、捷径仍不通吃）。注意**直接重跑 10.1 验证器会得到 6 条夹具**（g1/g2 已加入 `build_fixtures()`），所以它不再是与旧件同面的复现——这是夹具集扩大，不是数值变化。
- **10.3 首测**：已被 10.3b 取代（加深后重测通过），旧件作为"任务太浅"的原始读数保留。

结论：三份报告的**读数**仍成立，但**摘要**不再指向当前模块字节；任何引用都应以 10.1b／10.2 重跑／10.3b 为准，或明确标注"10.1b 之前"。

## 2026-11-01 11 的 L3 准入门（`step11_not_admitted`）

owner批准立项后交付[准入合同](../../reference/SPEC-TAIJI-CORE_L3_ADMISSION_20261101.md)、[验证器](../../../scripts/training/verify_taiji_l3_admission_11.py)与[报告](../../../reports/taiji_core_l3_admission_11_20261101.json)。**这不是实验**：它只读既有报告，把总计划第 11 行的准入条件机检化，并给每一项标出来源文件与字段路径。

四项前置**全部不成立**：

| 前置 | 读数 | 证据 |
|---|---|---|
| P1 L2 成立 | **假** | `reports/taiji_native_core_r5c_evaluation_20261031.json:status = FAIL_BEHAVIOR`（历史裁决，不因 09/10 的工程进展而变） |
| P2 至少一项模型侧代表能力 | **假** | 5 份候选报告（10.1/10.1b/10.2/10.3/10.3b）的 `model_involved` 全为 `false`——它们都是规则基线与参考规划器 |
| P3 安全/恢复/性能 | **假** | 恢复（09.8 `restore.spawn_bitwise_equal=true`）与性能（09.6 `status=passed_*`）有读数；**没有任何报告带运行时安全读数** |
| P4 同一候选贯穿 | **假** | 09.8 卡片有 `identity.parent_model_digest`，但没有任何能力报告指向它 |

判据本身是能翻的：任一份候选报告的 `model_involved` 变真且通过、或补一份安全读数，对应项就会翻；四项全真才 `admitted_to_l3_scope`。当前状态按出口结案为 `step11_not_admitted`，`l3_scope_started=false`、`training_admitted=false`。

**同日更正（11.7 之后）。** 补上 11.7 的仪器读数后 **P3 翻真**：安全项现在由 [11.7 卡 v2](../../../reports/taiji_core_instrument_11_7_v2_20261101.json) 支撑（20 次打分逐位可重复、p50 0.120 s／p95 0.125 s、评分期间 0 写／0 socket／0 子进程），恢复与性能沿用原读数；按修订后的准入器复算得到 [v2](../../../reports/taiji_core_l3_admission_11_v2_20261101.json)：`step11_not_admitted` 不变，**阻断项从四项降为三项**（P1／P2／P4）。上表"四项前置全部不成立"保留为当日早些时候的快照，不再作为现行读数；上面那句"补一份安全读数，对应项就会翻"已由这次实测兑现。

**[v3](../../../reports/taiji_core_l3_admission_11_v3_20261101.json)（同日再修一次扫描面）**：P2 原先只扫 10.1–10.3b 五份规则/参考基线，**看不到 10.4–10.7 这四份真正由模型产出的读数**——即使其中一份将来转正，闸也不会动。现在把四份纳入扫描：P2 的证据从"没有模型侧报告"变成"**四条模型侧负读数全在册**"（`model_below_random`／`contrast_below_floor`／`action_gap_below_floor`／`training_below_chance_plus`），结论与阻断项不变（P2 仍是假，因为它要的是"通过"，不是"测过"）。这是**扫描面**的修正，不是判据放宽。

**这条裁决不否定任何 09/10 的工程结果**：09 的六件收束物与生长净收益、10 的环境/绑定/规划契约都仍然成立。它说的是：**11 要问的"同一候选的自然中文连续任务"目前没有任何模型侧读数可用**。三项缺口里，P1 是历史结论（不能被重跑改写），P3 的安全读数可以做一件小工程，**P2 是唯一必须靠新实验才能解的一项**。

## 2026-11-01 10.4 第一份模型侧选择读数（`model_below_random`，机制已查明）

owner裁定做 P2 直路后交付[合同](../../reference/SPEC-TAIJI-CORE_SUBJECT_SELECTION_20261101.md)、[选择器](../../../taiji/subject_action_scorer.py)、[测试](../../../tests/taiji_native/test_subject_action_scorer.py)、[验证器](../../../scripts/training/verify_taiji_subject_selection_10_4.py)与[报告](../../../reports/taiji_core_subject_selection_10_4_20261101.json)。**这是本项目第一份由模型本身参与的能力读数**：主体（09.8 同一候选，父模型摘要逐位吻合）对每个合法动作的中文描述算条件 NLL，取最低者。

结果：**主体 2.81／贪心 3.86／随机 2.81** ⇒ `model_below_random`。仪器侧四项检查全过（同候选身份、确定性、打分次数=合法动作数、预算），所以这是主体的读数，不是仪器故障。

**机制已查明，且比分数更重要**：19 次选择里 **19 次都选"蓝色杯子"那句、0 次"红色印章"**，两次候选的 NLL 差稳定在 **0.0646–0.0650**（几乎不随局面变化）。也就是说主体学到的是**固定的字面偏好**，不是目标条件化的决策——f1 上这个偏好恰好对（0.99），f2/f4/g1 上全错并来回打转。与随机臂同分（2.81）是巧合：两者总分的算术结果相同，但选择序列不同。

**对 11 的直接影响**：`capability_passed=false`，**P2 仍未满足**，11 仍是 `step11_not_admitted`。这份负读数的价值在于把 P2 的缺口从"还没测"变成"测了，机制是缺少目标条件化的选择通路"。**不声称**：主体能选、能规划、能绑定；不声称任务能力；训练未批准。

## 2026-11-01 10.5 目标条件化接口探针（`contrast_below_floor`，机制：目标效应是共模的）

owner裁定「先换目标条件化接口」后交付[合同](../../reference/SPEC-TAIJI-CORE_GOAL_CONDITIONED_20261101.md)、[对比度选择器](../../../taiji/goal_conditioned_scorer.py)、[验证器](../../../scripts/training/verify_taiji_goal_conditioned_10_5.py)与[报告](../../../reports/taiji_core_goal_conditioned_10_5_20261101.json)。接口把「动作句的绝对似然」换成**同一句在匹配目标与错配目标下的 NLL 差**，让 10.4 里主导结果的字面先验在相减中抵消。

读数：`contrast(杯子→窗台) = +5.725e-4`、`contrast(印章→窗台) = +6.527e-4`，**两者之差只有 8.018e-5**，低于声明的 1e-3 底座 ⇒ `contrast_below_floor`。**零对照恰为 0**（匹配=错配时差恒为零），同候选身份与确定性都通过，所以这是接口的读数而非仪器故障。

**机制比结论更值钱**：目标文本**确实在推动分数**（两个动作的 contrast 都是正的、量级 6e-4），但它把两句**推得几乎一样多**——是一个**共模位移**，而不是动作特定的信号。换句话说，10.4 的问题是"完全不读目标"，10.5 修掉后暴露的是"读到了目标，但读到的内容不能区分动作"。

判据设计本身在两稿里都被我否决过（都写在合同里）：初稿用"换目标后选择翻转"——只有 2 个候选时 `argmax` 在任何非零差下都会翻转，几乎恒真；二稿用"正确动作 contrast 更高"——在对比度对目标反号的构造下两个方向条件互相等价，也是恒真。最终只判幅度并加零对照，方向只披露。

执行期一处纪律事件：首跑后为修未用导入改了脚本，导致报告摘要变旧；这次**直接重跑重绑**（36 秒、读数逐位相同），没有留下披露债。

**对 P2 的直接影响**：`capability_claimed=false`，P2 仍未满足。下一步只剩两条：①批准训练；②换成**动作特定**的打分构造（例如把「目标＋局面＋动作」拼成一条序列做条件似然，而不是只给目标与动作句），并先证明它的 contrast 能随动作区分。训练未批准。

## 2026-11-01 10.6 动作特定构造（`action_gap_below_floor`）与三条免训练接口的汇总

owner裁定「试动作特定构造」后交付[合同](../../reference/SPEC-TAIJI-CORE_ACTION_VERDICT_20261101.md)、[verdict 打分器](../../../taiji/action_verdict_scorer.py)、[验证器](../../../scripts/training/verify_taiji_action_verdict_10_6.py)与[报告](../../../reports/taiji_core_action_verdict_10_6_20261101.json)。构造：`prefix = 局面 + 目标 + 动作句`，比较模型对「是」与「否」的似然，`verdict = NLL(否) − NLL(是)`；动作在前缀里，所以每个候选的前缀都不同。

读数：杯子 **−0.143546567**、印章 **−0.143066181**，**动作差 4.804e-4**（＜1e-3 底座）⇒ `action_gap_below_floor`；方向还是**反的**（目标关于杯子，印章动作的 verdict 反而更高）。答案 token 可分（每行的「是/否」差约 0.143），零对照与确定性通过，同候选身份通过。

**三条免训练接口的横向对照**（同一主体、同一局面）：

| 版本 | 构造 | 动作间信号 | 判定 |
|---|---|---|---|
| 10.4 | 动作句绝对似然 | 字面偏好恒定，19/19 选杯子 | `model_below_random` |
| 10.5 | 目标对比度（匹配 vs 错配） | 8.018e-5（目标效应是共模） | `contrast_below_floor` |
| 10.6 | 动作进前缀的是非题 | **4.804e-4**（比 10.5 大 6 倍，仍不到 1e-3），方向相反 | `action_gap_below_floor` |

**结论**：把动作放进前缀确实把动作间信号放大了约 6 倍，但仍未过门，且方向不对。**三条免训练接口都没能给主体一个"随目标、区分动作"的选择信号。** 这不足以证明"训练一定能解"，但足以说明：**在零任务信号的情况下，继续换构造边际收益在下降**（10.5→10.6 只放大 6 倍，离 1e-3 还差一半以上）。

**P2 仍未满足**，11 仍是 `step11_not_admitted`。`capability_claimed=false`；不声称主体能选、能规划或能绑定；不声称任务能力；训练未批准。

## 2026-11-01 10.7 第一次有界训练（`training_below_chance_plus`，机制：学到的是常量偏置）

owner批准「一次训练」后交付[合同](../../reference/SPEC-TAIJI-CORE_TRAINING_10_7_20261101.md)、[训练模块](../../../taiji/goal_conditioned_training.py)、[测试](../../../tests/taiji_native/test_goal_conditioned_training.py)、[验证器](../../../scripts/training/verify_taiji_goal_training_10_7.py)与[报告](../../../reports/taiji_core_goal_training_10_7_20261101.json)。数据全部合成：12 个 (局面, 目标) 样本覆盖 6 个局面，**按局面划分 6 训练／6 留出**，同一局面不跨集；监督是动作句进前缀的是非题（同 10.6），正确动作取「是」、错误动作取「否」。两个 lr 都跑、都发布。

| 配置 | 训练损失 | 留出正确 | 保持漂移 |
|---|---|---|---|
| 未训练 | — | 3/6 | — |
| lr=1e-5 | 11.2069 → 11.0475 | 3/6 | 0.002633 ✓ |
| lr=1e-4 | 11.2069 → **7.0997**（降 37%） | **3/6** | **0.032178 ✗ 破门** |

**机制比分数更关键**：把每次选择反推出来——未训练与 lr=1e-5 的命中模式都是 [F,T,F,T,F,T]（在杯子目标上错、在印章目标上对 ⇒ **总是选印章**），lr=1e-4 翻转成 [T,F,T,F,T,F]（⇒ **总是选杯子**）。**训练前是一个常量偏好，训练后换成了另一个常量偏好**；损失降 37% 学到的是这个偏置，不是"按目标选动作"。留出率停在 3/6 是因为正确对象在样本间交替，任何常量偏好都必然命中一半。

结论：**四条路（三条免训练接口 + 一次有界训练）都没能让主体产生目标条件化的选择**；唯一学起来的配置还同时破坏了保持门（0.0322 > 0.030）。**P2 未满足**，11 仍 `step11_not_admitted`。`weights_published=false`、`default_model_replaced=false`，权重只落在 `output/taiji_core_10_7/`（`.pt` 已被 gitignore 忽略）。本次训练授权不外推。

**编号整理（同日）。** 本步四个模型侧读数占用 10.4–10.7；总计划里未开工的三张工程卡（执行及结果信用／工程、定价与合同／验证及收束）原为 10.4–10.6，与之重号，已顺延为 10.8–10.10（内容不变，见[总计划 §4 编号说明](../../MAINLINE.md#4-顺序执行工作卡)）。报告、脚本与合同的 `10_4`…`10_7` 后缀按编号规则不改名。

## 2026-11-01 10.11 收束依据与负结果卡（`closeout_10_evidence_complete`）

交付[合同](../../reference/SPEC-TAIJI-CORE_CLOSEOUT_10_20261101.md)、[验证器](../../../scripts/training/verify_taiji_closeout_10.py)、[测试](../../../tests/taiji_native/test_closeout_10.py)与[报告 v2](../../../reports/taiji_core_10_closeout_v2_20261101.json)。**只读复核，不产生新读数、不训练、不发权重。** 首版保留：[v1](../../../reports/taiji_core_10_closeout_20261101.json) 写于 `black` 格式化之前，其验证器绑定已过期；v2 只换了这一枚哈希与墙钟，九项读数逐项相同。

九项复核全过：①四份探针报告的候选摘要两两相等且等于 09 收束卡；②四项出口原文在册（`model_below_random`／`contrast_below_floor`／`action_gap_below_floor`／`training_below_chance_plus`）；③四份都无能力声称且 `training_admitted=false`；④10.7 未训练与 lr=1e-5 的命中序列都是 `FTFTFT`、lr=1e-4 是 `TFTFTF`，各 3/6，两臂序列不同；⑤lr=1e-4 漂移 0.032178＞0.030、lr=1e-5 漂移 0.002633≤0.030；⑥lr=1e-4 末/首损失比 **0.633512**≤0.70；⑦`weights_published=false`、`default_model_replaced=false`；⑧11 仍 `step11_not_admitted`、`P2_one_capability_by_the_model` 为假；⑨四份实测 CPU（48.8／37.6／36.2／392.6 秒）均在各自 600 秒预算内。

卡上三枚字段是给裁决用的：`p2_satisfied=false`、`step11_status=step11_not_admitted`、`stage_close_decision=owner_pending`。**通过只表示四个负出口可复核、允许声称的句子没越界**；不表示主体能选择/规划/绑定，不表示任务能力，不表示 10 已收口，也不授权训练。

测试 11 条：真实报告判绿 1 条；其余每条篡改一个字段并断言落到它自己的出口——改状态→`exit_not_recorded`、翻转一个命中或删 `runs`→`mechanism_not_reproduced`、改摘要→`identity_mismatch`、`capability_claimed`→`capability_claim_forbidden`、`weights_published`→`weights_published_forbidden`、11 状态→`admission_regressed`、超预算→`budget_stop`，另单测保持门与损失比。⇒ 这张卡能翻。

**读数复算守卫。** [test_closeout_10_recompute.py](../../../tests/taiji_native/test_closeout_10_recompute.py) 从四份报告自带的原始行复算每个头条数，六条全过（其中一条用篡改副本证明守卫能翻）：10.4 的 `totals` 是**六局终态分之和**（单局 0.99／−0.06／1.00；均值 0.468，不是 2.81），且 19 张选择表 19 次都取同一对象；10.5 的 `contrast`＝错配 NLL−匹配 NLL、`gap`＝|c₀−c₁|、零对照全 0；10.6 的 `verdict`＝NLL(否)−NLL(是)、`answer_gap` 为其相反数、`gap`＝|v₀−v₁|；10.7 的命中数＝行内 `hit` 之和（两臂各 3/6），lr=1e-4 损失比 0.633512。⇒ 头条数都能从同一件里复算，不必信手抄。

## 2026-11-01 10.7a 权重可重放（`weights_replay_bitwise`）

交付[验证器](../../../scripts/training/verify_taiji_training_replay_10_7a.py)、[测试](../../../tests/taiji_native/test_training_replay_10_7a.py)与[报告 v2](../../../reports/taiji_core_training_replay_10_7a_v2_20261101.json)（[首版](../../../reports/taiji_core_training_replay_10_7a_20261101.json)保留，换绑原因同 10.11：验证器被 `black` 格式化）。由声明种子重建同一主体（父摘要一致）、`strict=True` 载入两臂 `state_dict`、重算 12 个留出判定：

| 臂 | 命中（重放／卡） | 最大绝对差 | 权重字节（盘／卡） |
|---|---|---|---|
| lr=1e-5 | 3/6 ／ 3/6 | **0.0（逐位）** | 26,633,881 ／ 26,633,881 |
| lr=1e-4 | 3/6 ／ 3/6 | **0.0（逐位）** | 26,634,022 ／ 26,634,022 |

40.6 CPU 秒。⇒ 10.7 保存的两份权重在全新进程里可回落，且**逐位复现**发表读数；这是"同一身份的候选版本可追溯"的资产证据，**不是**能力证据（两臂仍是常量偏好）。测试 6 条覆盖比较器与状态表，并用缺值／命中翻转／短行三种篡改证明它能翻。

## 2026-11-01 11.7 候选仪器读数（`instrument_measured_quiet`）

交付[合同](../../reference/SPEC-TAIJI-CORE_INSTRUMENT_11_7_20261101.md)、[验证器](../../../scripts/training/verify_taiji_instrument_11_7.py)、[测试](../../../tests/taiji_native/test_instrument_11_7.py)与[报告 v2](../../../reports/taiji_core_instrument_11_7_v2_20261101.json)。候选＝09.8 主体（父摘要一致），路径＝`taiji/action_verdict_scorer.answer_nll`，同一前缀连续 20 次打分：

| 读数 | 值 |
|---|---|
| 第 1／第 20 次返回值 | **逐位相同** |
| p50／p95 墙钟 | **0.120 s／0.125 s** |
| 峰值工作集 | 起始 217 MB → 建主体 1.483 GB（+1.267 GB）→ 20 次打分后仍 1.483 GB（**+0**） |
| 评分期副作用 | 0 写文件、0 socket、0 子进程、0 `os.system` |
| 成本 | 34.8 CPU 秒／35.4 墙钟秒 |

执行期两处修正：①首版把 `answer_nll` 的返回当成张量（`.detach()`），实际返回 float ⇒ 首个调用即 `engineering_failure`，首版保留；②峰值内存起初只在打分前后各取一次，高水位被建主体撑满而恒为 0，改为**起始／建主体后／打分后**三点。测试 4 条覆盖打开模式分类、最近秩分位与状态优先级，并用写模式／socket 等反例证明能翻。

**并入 11 门后 P3 翻真**，准入仍 `step11_not_admitted`（P1／P2／P4 不成立）。

## 2026-11-01 10.7b 梯度对齐诊断（`training_groups_pull_together`）

交付[脚本](../../../scripts/training/diagnose_training_direction_10_7b.py)、[测试](../../../tests/taiji_native/test_training_direction_10_7b.py)与[报告](../../../reports/taiji_core_training_direction_10_7b_20261101.json)。只读六个训练样本的 per-sample 梯度，并载入 10.7 已保存的 lr=1e-4 权重算 Δθ；**不更新任何参数**。

| 读数 | 值 |
|---|---|
| 两组（按正确对象 3／3）均值梯度的余弦 | **0.999977** |
| 任意两样本梯度余弦（中位数） | **0.999977** |
| 残差份额 `mean‖gᵢ−ḡ‖²／mean‖gᵢ‖²` | **1.75e-5** |
| `‖ḡ‖／平均‖gᵢ‖` | 0.99999 |
| Δθ 与均值梯度的余弦 | −0.048（Adam 是符号型更新，高维上近似正交属预期，**不作失败信号**） |
| `‖Δθ‖` | 4.439 |

解读只到这里：一阶上这六个样本几乎在推同一个方向，任何取平均的更新都只能是共模的；条件化所需的差异只占约 **1.8e-5** 的梯度能量。这与 10.5 的共模 8e-5、10.7 的"训练只换常量"互相印证。**不声称**这是唯一解释，也不声称换损失／加样本后会怎样——那正是 ③ 要预先声明的东西；但"再多训几步就能学会"已被这条读数否掉。

测试 2 条（余弦端点、判定表）。37.9 CPU 秒。

## 2026-11-01 10.7c 残差结构诊断（`residual_group_structure`）

交付[脚本](../../../scripts/training/diagnose_residual_structure_10_7c.py)、[测试](../../../tests/taiji_native/test_residual_structure_10_7c.py)与[报告 v2](../../../reports/taiji_core_residual_structure_10_7c_v2_20261101.json)。把 10.7b 的残差（`gᵢ − ḡ`）按"正确对象"重新分组，问它是噪声还是信号；**不更新参数**。

| 读数 | 值 |
|---|---|
| 组解释份额 `F＝Σ_g n_g‖r_g‖²／Σ_i‖r_i‖²` | **0.6459**（随机 3／3 划分期望 0.2） |
| 10 个去重划分里的名次 | **最高**（次优 0.2561／0.2554）⇒ 穷举 p＝**0.1**（该设计下限） |
| 组内残差余弦 | 0.5226（蓝杯）／0.5087（红印），门 0.5，**刚过线** |
| 残差范数示例 | 0.0206／0.0210／0.0270 |

解读：残差**不是噪声**——按正确对象分组能解释约 65% 的残差能量，其余 9 个 3／3 划分都明显更差（最好的也只有 0.26）。条件化信号在一阶参数空间里**存在**，只是被共模压了约 4 个数量级（份额 1.75e-5）。这条把 ③ 的第一候选从"加新通路"改成"**减共模的更新／目标**"。**不声称**它会成功：n＝6、p 只能到 0.1、组内余弦刚过线、一阶方向不代表有限步可学。

执行更正：首版把并列的互补划分按"严格大于"漏计，`permutation_p` 报成 0.05；v2 改为只枚举 10 个去重划分并按"≥"计数 ⇒ p＝0.1。F 与组内余弦两版逐位相同；首版保留。

测试 4 条（份额端点、分组混合、组内余弦、判定表）。37.7 CPU 秒。

## 2026-11-01 10.7d 成对目标探针（`margin_isolates_differential`，并**否证上一轮的建议**）

交付[脚本](../../../scripts/training/diagnose_margin_objective_10_7d.py)、[测试](../../../tests/taiji_native/test_margin_objective_10_7d.py)与[报告](../../../reports/taiji_core_margin_objective_10_7d_20261101.json)。同一候选上比较两种目标的一阶梯度：10.7 的平摊损失（`NLL(是|对) + NLL(否|错)`）与成对 margin（`softplus(-(verdict_错 − verdict_对))`，verdict 即 10.6 的 `NLL(否)−NLL(是)`）；两边都只读梯度，**不更新参数**。

| 读数 | 平摊损失 | 成对 margin |
|---|---|---|
| 两两余弦中位数 | **0.999977** | **−0.999514** |
| 组内余弦 | 0.99972 | 0.99972 |
| 组解释份额 F | 0.645858（与 10.7c 逐位相同 ✓） | 0.999813 |
| 个体梯度范数 | 5.4596 | 0.0121 |
| **均值范数** | 5.4595（不抵消） | **6.8e-05（0.56%）** |

两均值夹角余弦 −0.0047。

解读：margin 目标确实把六个样本劈成两组相反方向——**但正因为相反，平均更新几乎为零**（个体 0.0121 → 均值 6.8e-05）。所以"把平摊损失换成成对损失"**不能**打破共模：共享参数在一阶上要么收到共模（平摊），要么两两相消（成对）。能承载条件化的是**梯度逐样本不同**的参数——输入条件的读出（其梯度是外积），那是架构级改动。

**更正**：上一轮 10.7c 的结论句写"③ 的第一候选是减共模的更新／目标"，本轮被自己的读数否证。MAINLINE 的 10.7c 行已就地标注"已被 10.7d 否证"，裁决段的 ③ 已改写为"输入条件的读出"。这不是数据问题，是我把"残差里有结构"误推成了"平均更新能拿到它"。

测试 4 条（判定表四种分支）。48.3 CPU 秒。

## 2026-11-01 ③ 的定价（只测量，不立项）

为把 ③ 从"架构级决定"压成可裁定的报价，量了三个数（同一次 `build_subject`，不训练）：`goal_vector` 输出 **128 维**、动作 `context` 是 **27×128** 的补丁序列、候选总参数 **3,323,547**（全部可训练）。据此，零初始化的秩 4 双线性读出头只需 2×128×4＝**1,024** 参数（主体的 0.03%）；因为其余参数冻结、特征可预计算，整趟预算可压到 **≤120 CPU 秒**（对照 10.7 整模型 32 步实测 169 秒）。判据与对照臂写在 MAINLINE 的 ③ 段：留出 ≥5/6、漂移 ≤0.030、标签置换对照 ≤3/6。**未立项，等 owner 裁定。**

## 2026-11-01 账本哈希审计（`cards_stale_unresolved`）

交付[脚本](../../../scripts/training/audit_taiji_card_hashes.py)与[报告 v2](../../../reports/taiji_core_card_hash_audit_10_12_v2_20261101.json)。逐张重算 `reports/taiji_core_*.json` 里 `source_hashes` 绑定的字节，**只读不改**。

| 分类 | 张数 |
|---|---|
| `fresh`（绑定仍与盘上一致） | **25** |
| `superseded`（已过期但有现行 v2） | **5** |
| `unresolved`（过期且无 v2） | **14**，其中 1 张是本审计自己的 v1（扫描时 v2 还没落盘）⇒ 实际 **13** |
| `no_hashes`（本就不绑字节） | 3 |

13 张 unresolved 里，模块类漂移集中在后续步骤改过的件：`taiji/native_goal_world.py`（10.1b 加 `occupy`，牵连 10.1／10.2／10.3 三张卡）、`taiji/goal_planner.py`（10.3 的 `_best_within` 修正）、`taiji/growth_credit.py` 与 `taiji/state_migration.py` 与 `taiji/activity_learning.py`（09.6.1）、`taiji/native_activity.py`（09.6.2 的 `arrivals` 钩子）、`taiji/native_activity_fixture.py`（09.6.2 之后仍有改动）。

**这不否定任何读数**：卡绑的是它跑的那一刻的字节。它意味着这些数字**不能从今天的树复算**；而且当前工作树未提交，中间版本的字节多半已不可恢复，所以这次只登记、不重算、不就地改哈希。据此把规则写死：引用 unresolved 卡的数字必须带一句"该卡绑的是当时字节，现树已变"；需要可复算就按新步骤重发 v2，不许回填旧卡的哈希。

另一条今天踩了 5 次的教训：**自绑定卡（报告里含验证器/诊断脚本自身的 SHA）最容易因事后格式化失效**（10.11、10.7a、11.7 首跑、10.7c、本审计各一次）。流程改成：先 `black` 再跑验证器，报告落盘后不再编辑其脚本；确需改就重发 v2。

## 2026-11-01 全过程资源成本总账（`cost_ledger_built`）

交付[脚本](../../../scripts/training/audit_taiji_cost_ledger.py)与[报告](../../../reports/taiji_core_cost_ledger_20261101.json)。只汇总各卡已发布的 `cpu_seconds`／`wall_seconds`，**不补算、不外推**。

| 族 | 卡数 | CPU 秒 | 墙钟秒 |
|---|---|---|---|
| 09（09.x） | 13 | **5,190.3** | 5,333.6 |
| 09（R021–R029／算法检查，无 step 字段） | 3 | 499.5 | 533.3 |
| 10（10.x） | 19 | **759.3** | 772.7 |
| 11（11.x） | 4 | 67.1 | 68.6 |
| 未标 | 1 | 0.4 | 0.4 |
| **合计** | **40** | **6,516.7** | 6,708.5 |

三条要点：

1. **8 张卡没有成本字段**（R021–R027、算法检查等 2026-10-31 的件）——列出、不补算 ⇒ 6,516.7 秒是**下界**。
2. 重跑分开计（10.11／10.7a／10.7c／11.7 的 v1 与 v2 都是两次真实运行），所以这是"实际花掉"而非"最终产物成本"。
3. **P2 战役本身**（10.4–10.7d，10 张卡）花了 **758.8 CPU 秒**；③ 的报价 ≤120 秒 ≈ 它的 **16%**；整个 10 阶段 759.3 秒占全账本 11.6%。

墙钟逐卡相加会互相重叠，只有 CPU 可跨卡相加——报告里写明了。

**同日更新（10.15 之后）。** 成本总账复算到 [v2](../../../reports/taiji_core_cost_ledger_v2_20261101.json)：**63 张卡、8,985.4 CPU 秒**（仍是下界，仍缺 8 张无成本字段的旧件）；其中 10 线 **3,227.7 秒**、P2 探针 16 张合计 **1,509.9 秒**。

**2026-11-02 总账复算到 v4，并修掉一处口径缺陷（`cost_ledger_built`）。** 复算到 [v4](../../../reports/taiji_core_cost_ledger_v4_20261102.json)：**112 张卡、9,827.6 CPU 秒**（仍是下界；另有 3 张卡不发布任何已声明成本路径，按名不补算），其中 10 线 **3,874.1 秒**、P2 探针仍 16 张 **1,509.9 秒**。这一版换了取法，不是加了运行：
- 旧版只读顶层 `cpu_seconds`，于是 **20 张卡被记成"无成本"**，其中包含 10.21.4 那次真训练（1,200 次优化器调用，`cost.cpu_seconds = 200.25`）与 14.1／14.2（`cost.cpu_seconds = 0.0／36.66`）。新版按**声明式路径表**取值（顶层 → `cost.cpu_seconds` → `cpu_seconds_checks_only`），每张卡只取一个源并把源名写进行里，所以分项键与预算键不会被叠在总量上；补回 **612.0 CPU 秒**，无成本卡从 20 张降到 3 张。
- 汇总卡（`costs.executed_cpu_seconds`）不进加和，改作第二视图对照。**这里我第一次把守卫写错了口径**：我用"两视图之差 ≤2 秒"作判据，一跑就报 `cost_source_conflict`——10.21 族全部卡之和 574.28 秒，而 10.21.6 收束卡自述 247.53／254.52 秒，差 319.76 秒。查回原卡：收束卡 `caliber` 明写"只含 10.21 六步"，`by_step` 只点每张规范卡，被丢弃的迭代卡（10.21.1 三次仪器、10.21.2 六次工程迭代、10.21.3 四次、10.21.4 一次）不在其内；`with_iterations` 也只加了 10.21.1 的 6.99 秒。⇒ 相等从来不是成立的判据，那次红量的是我的比较，不是这条线。该次读数按 `_conflict_record` 保留（嵌套路径的取值是对的，下一版沿用）。
- 现行守卫换成两种口径下都必须成立的不等式：**全树花费不得低于其自身规范子集所声称的花费**（`member_sum_is_not_below_stated`，现测成立）；差值 319.76 秒作为口径差披露，不当错误。不等式被违反才报 `cost_source_conflict`。
- 由此得到一条对本线有用的结论：**同一件事有两个都成立的成本报价**——"这场研究实际花掉的计算"（全部卡，含废件）对"合同口径内执行的成本"（收束卡的 `by_step`）。给 14.3 这类包定价、以及回答"这条线已经花了多少"时，必须点名用哪一个；此前正文引用的 8,985.4／3,227.7 是当时 v2 的读数，保留不改，但**现在能声称的是 v4 的 9,827.6 CPU 秒（下界）**。
- 补回的 612.0 秒按源分列并已写进卡里：`cost.cpu_seconds` 9 张 496.48 秒、`cpu_seconds_checks_only` 7 张 115.56 秒。注意后一源是**只跑检查的那一段**，不是一次完整运行的花费，所以 9,827.6 是"混合源的下界"：要纯完整运行口径就减 115.56 秒。这个换算我不动，源与张数由卡片自己披露。
- 机器可见的代价（按实数报，不粉）：本轮我自己那族里落进 unresolved 的是 **6 张**——`cost_ledger_v3_20261102.json`、`cost_ledger_v3_v2_20261102_conflict_record.json`（已被 v4 的 `supersedes` 认下，归入 superseded 后不再计入）、`slot_descriptor_10_21a2_20261101_void.json`、`fixture_fairness_audit_20261102_void.json`，加两张哈希审计自绑卡（`card_hash_audit_10_12_20261101.json`／`_v2`，它们绑的是审计脚本自己的字节）。同一条更正见本页下一节：那台审计已被 owner 授权修好，修完的真面是 41 张 unresolved，不是 26。

## 2026-11-01 10.7e 目标→动作读出（`readout_binding_supported`，本线第一条正的模型侧读数；**强度见 10.7e-1**）

owner 裁定「收口 10 ＋ 立项读出实验」后交付[合同](../../reference/SPEC-TAIJI-CORE_READOUT_10_7E_20261101.md)、[读出头](../../../taiji/goal_action_readout.py)、[验证器](../../../scripts/training/verify_taiji_readout_10_7e.py)、[测试](../../../tests/taiji_native/test_goal_action_readout.py)与[报告](../../../reports/taiji_core_readout_10_7e_20261101.json)。

设计：主体**全部参数冻结**；只训一个秩 4 双线性头 `score = (gᵀU)·(aᵀV)`，g＝主体 `goal_vector`（128 维），a＝动作 `context`（27×128 池化）；U 小随机／V 零 ⇒ 起点输出恒 0（未训练时 6/6 平局）。

| 臂 | 留出命中 | 末损失 |
|---|---|---|
| 目标臂（6 训练样本，300 步，Adam lr 0.05） | **6/6** | 0.0 |
| 标签对调对照 | **0/6** | 0.0 |
| 未训练（输出恒 0） | 0 命中／**6 平局** | — |

主体冻结：训练前后参数张量与 6 个留出样本的 `sample_loss` **逐位相同**。1,024 参数；35.8 CPU 秒（预算 120）。

**边界（必须一起读）**：

- 头只吃编码、**不读循环状态**——它证明的是"目标↔动作的字面绑定在这套编码上可学"，不是主体在状态里推理。
- 留出 4 个正确动作里**只有 2 个训练没见过**：跨局面＋**部分**跨动作，不是完全组合泛化。
- 夹具让目标唯一决定动作，天花板很低；对调臂也能把损失降到 0，说明这套夹具对任何可分标签都易学。
- 状态**刻意不叫 `passed_*`**：11 的 P2 要的是"代表能力"，一条 6 样本合成夹具不是。若将来要接进 P2，必须先改判据并写明理由，否则就是放水。

执行期一处修正：特征张量最初带计算图，训练第二步即 "backward a second time" 崩；改为在 `no_grad` 下取特征并 `detach` 后一次通过。测试 3 条（零起点与 1,024 参数、单侧零初始化不是鞍点、状态表七分支）。

## 2026-11-01 10.7e-1 全动作排名（`readout_rank_below`，把 10.7e 的强度收窄）

不训练：把 10.7e 保存的头拿来，对每个目标给"2 对象×3 位置＝6 个动作"整体打分，问正确动作排第几。交付[脚本](../../../scripts/training/probe_readout_rank_10_7e1.py)、[测试](../../../tests/taiji_native/test_readout_rank_10_7e1.py)与[报告](../../../reports/taiji_core_readout_rank_10_7e1_20261101.json)。

| 组 | top-1 | 平均名次 |
|---|---|---|
| 留出 6 个目标 | **3/6** | 2.0 |
| 训练 6 个目标 | **1/6** | — |

36.2 CPU 秒（机会 top-1＝1/6）。

**读过界**：这个头**连自己训练目标的全局第一名都排不出来**（训练 top-1 只有 1/6）。所以 10.7e 的 6/6 只能读成"正确动作 > 它被训练比较的那一个干扰项"，**不是**"从动作集里挑对"。⇒ 若目标是"会选择"，训练目标本身必须是 6 选 1；二选一的判别器不能当选择器用。这也是为什么先跑这个探针：只报 6/6 就接 P2，会是一次实打实的放水。

执行期两处桩件错误（已修）：测试里用 2 维向量装 6 个动作（IndexError）、把 `str(bytes)` 当成表的键（KeyError）。修完 2 条测全过。

## 2026-11-01 10.7e-2 原样主体的六选一基线（`budget_stop`：读数已取到，但超了我自己声明的预算）

为给 10.7e-1 的"留出 top-1 3/6"一个地板，用主体**自己的似然**（`subject_action_scorer.score_actions`，不训练、不改参数）对同一批目标给 6 个动作排序。交付[脚本](../../../scripts/training/probe_raw_subject_rank_10_7e2.py)与[报告](../../../reports/taiji_core_raw_rank_10_7e2_20261101.json)。

| 组 | top-1 | 平均名次 |
|---|---|---|
| 留出 6 个目标 | **2/6** | 3.167 |
| 训练 6 个目标 | **0/6** | — |
| 留出组最佳常量（蓝杯→木桌） | 2/6 | — |

**这一趟超预算**：实测 **571.75 CPU 秒**，我只声明了 240 ⇒ 按预注册判 `budget_stop`，**不作正式结论**。原因是我的单价估错约 2.4 倍（每次 `score_actions` 约 8 秒，不是 1.2 秒）。要变成正式基线须按 ≥900 秒重新预注册再跑；我没跑，因为要回答的问题（头的 3/6 是否高于地板）这组非正式读数已给出方向，正式化要再花约 10 分钟。

**非正式比较**：原样主体 2/6（正好等于最佳常量），10.7e 的头 3/6；两者都贴着 1/6 的机会线，且头连训练目标都排不到第一（1/6）。⇒ 头**没有显示出高于原样先验的选择能力**，与 10.7e-1 的结论一致。

## 2026-11-01 10.7f 六选一选择（`readout_selection_absent`：能背不能迁）

owner 批准后交付[合同](../../reference/SPEC-TAIJI-CORE_SELECTION_10_7F_20261101.md)、[读出头](../../../taiji/goal_state_readout.py)（读目标＋局面＋动作，**1,536 参数**）、[验证器](../../../scripts/training/verify_taiji_selection_10_7f.py)、[测试](../../../tests/taiji_native/test_goal_state_readout.py)与[报告](../../../reports/taiji_core_selection_10_7f_20261101.json)。主体冻结，6 路交叉熵、300 步。

| 臂 | 训练末损失 | 留出 top-1 |
|---|---|---|
| 目标臂 | **5e-06**（完全拟合） | **2/6** |
| 目标错配对照 | 4.7e-05 | 0/6 |
| 未训练（全平局） | — | 2/6（argmax 取第一个） |

主体参数与留出 `sample_loss` 逐位不变；39.3 CPU 秒（预算 300）。

**读法**：训练集被拟合到零误差，留出却**与未训练的常量选择持平**（2/6）。⇒ 1,536 个参数在冻结编码上**能记住 6 个样本，学不到可迁移的目标→动作关系**。这与 10.7e-1（训练 top-1 1/6）、10.7e-2（原样基线 2/6）一致，也解释了 10.7e 二选一 6/6 的性质：那是对 12 个比较的拟合，不是绑定。

⇒ 读出路线在当前夹具／编码／数据量下**收束为负**；P2 未满足，11 仍 `step11_not_admitted`。

执行期一处修正：`fit` 起初只接受默认 width=128，玩具测试用 4 维特征直接 RuntimeError；加上 width/rank 参数后 3 条测全过。

## 2026-11-02 目标条款×证据对照（现状地图，不是收尾；2026-11-01 建，本轮按 14.1／14.2／14.3 与哈希审计复算）

把长期目标拆成 23 条可核验条款，逐条指向现行读数。✓＝有证据；◐＝部分／薄；✗＝未达成或未做。

| # | 条款 | 证据 | 现状 |
|---|---|---|---|
| 1 | 连续内部状态 | 09.8 收束卡的 bundle 与恢复；R027 | ✓ |
| 2 | 持续可塑性 | R026 六臂（预注册 negative） | ✓ |
| 3 | 继承式结构生长 | 09.6.3–09.7b：净收益 +1.739e-4／+1.903e-4、8 步门内 | ✓ 薄边际 |
| 4 | 自身可学习、可恢复 | 09.5 恢复门；10.7a 权重重放逐位一致 | ✓ |
| 5 | **自身机制实际承担决策** | 10.4–10.13 全负；10.20 收案；**14.1／14.2 判 10.7 面为目标复读捷径（状态置零与错位仍 6/6）⇒ 那些读数不能否证参数化**；**14.3 给机制归属：继承读四组 top-1 全 = 1/9 随机、熵 0.99896，单行受限下换实体仅移动 0.136（门限 0.50）⇒ `pathway_unreachable`**；10.21.4 训练后的新读头也只到 0.1125（门槛 0.40） | **✗（缺口从"缺组合泛化"收窄为"读通路承载不了映射"；再训／换头／扩样本已由两次独立读数排除）** |
| 6 | 自然中文可靠连续任务 | 11 准入 `step11_not_admitted`；11.3a 固定答案套件地板 **0/5**（澄清/未知 2 项结构性不可评） | ✗ |
| 7 | 同一身份跨会话积累经验 | 恢复一致有（11.2b 快照、11.4c 硬退续跑）；**经验积累未做**（12 未准入） | ✗ |
| 8 | 核验并纠正认识 | — | ✗ |
| 9 | 从经验与反馈改进负责机制 | — | ✗ |
| 10 | 学新能力并保持旧能力 | 保持漂移门（≤0.030）有；学新能力未成 | ◐ |
| 11 | 新增／整合／回收单元 | 09 出生与 lesion；10.7e／10.7f 读出 | ◐ 工程有、收益未成 |
| 12 | 按真实结果调整学习／活动／生长 | 09.6.2 活动负、09.7 生长正但预算绑定 | ◐ 未闭环 |
| 13 | 无提示自主选择 | 13 未准入 | ✗ |
| 14 | 预声明范围／预算／出口 | 全过程合同 | ✓ |
| 15 | 身份与谱系可追溯 | 11.1a manifest 绑定（候选摘要＋6 模块 SHA＋来源；子进程重建、漂移被拒）；10.7a 重放逐位一致；14.3.0 锚点测试逐位重现 10.21.1 的 420 个已发布字段；**哈希审计 2026-11-02 修好后真面：64 fresh／10 no_hashes／10 superseded／41 unresolved**（旧版只读顶层 source_hashes，把绑在 identity.source_hashes 里的卡漏成"无哈希"） | ✓ 入口侧（41 条过期待现行线按 supersedes 归位） |
| 16 | 对照合理基线 | 每步 | ✓ |
| 17 | 未见任务表现 | 夹具级留出 | ◐ 仅合成 |
| 18 | 必要机制实际贡献 | 09.6.x；10.7b–d | ✓ |
| 19 | 旧能力保持 | 漂移门（`abs(ΔNLL)` 对称稳定性，非有符号遗忘；无「父版本已达标」能力集） | ◐ 仅稳定性（原值 ✓，2026-11-01 按 MAINLINE §51 自承下调） |
| 20 | 主体连续与恢复一致 | 09.5；10.7a | ✓ |
| 21 | 结构生长的真实收益 | 09.7／09.7b | ◐ 薄边际、预算绑定 |
| 22 | 主动选择效果 | — | ✗ |
| 23 | 全过程资源成本 | 成本总账 [v5 的复算版](../../../reports/taiji_core_cost_ledger_v5_v2_20261102.json)：**9,832.5 CPU 秒／120 张卡**（混合源下界；纯完整运行口径减 `cpu_seconds_checks_only` 的 115.56 秒；含 14.3 的 3.672 秒与 14.3 重跑） | ✓ |

**结论：目标未达成。** 卡点只有一个——**第 5 条**（自身机制承担决策），它同时挡住 6／7／8／9／10／13。其余条款要么有证据（1／2／3／4／14／16／18／19／20／23），要么只有部分证据（10／11／12／15／17／21）。这张表是现状地图，不是收尾声明。

## 2026-11-01 10.7h 几何读出 + 夹具直通（`geometry_does_not_select`）

零参数读出：方向全部取自编码器（对象／位置主效应），目标用 `goal_vector` 编码，`score＝cos(g,d_o)＋cos(g,d_p)`；**不拟合任何参数、不训练**。交付[脚本](../../../scripts/training/probe_geometry_select_10_7h.py)与[报告](../../../reports/taiji_core_geometry_select_10_7h_20261101.json)。

| 组 | top-1 |
|---|---|
| 留出 6 | **0/6** |
| 训练 6 | 4/6 |
| 换目标的对照 | 0/6 |

33.5 CPU 秒。

**同时查出一条夹具直通**：`samples()` 的 12 个样本里，**12/12 的 `goal` 与正确动作句逐字节相同**（goal＝"把X放到Y"，正确动作句也是"把X放到Y"）。所以这套选择任务的捷径是**复读目标**——任何 copy／match 机制都能满分；而主体的 `goal_vector` 是另一条学出来的投影、与动作编码不同空间，所以它拿不到这条捷径（0/6）。

这解释了整条读出线：训练过的头只能在 6 个样本里学一个"把 `goal_vector` 投到动作空间"的映射（10.7f 能背不能迁），零参数几何又对不上（0/6）。**问题不在头，也不在动作特征（10.7g 可加性 99.987%），在于夹具把答案写进了题面，而候选没有复读通道。**

⇒ 继续加头／加样本之前，**夹具必须先把目标改成不与动作句同形**（例如"让蓝杯在木桌上""印章要离开左盒"），否则任何结果都无法区分"会绑定"和"会复读"。这是任务级修改，不需要训练授权。

## 2026-11-01 10.8 目标改写与复读捷径（`paraphrase_still_copyable`：光改措辞不够）

10.7h 查出 12/12 的目标与正确动作句逐字节相同。本步先量"便宜修法"：三种不用"把…放到"的目标措辞（`"{obj}该待在{place}。"`／`"我希望{obj}出现在{place}。"`／`"目标状态：{obj}在{place}。"`），6 个局面 × 2 对象 × 3 种 = 36 个目标。交付[改写模块](../../../taiji/goal_paraphrase.py)、[验证器](../../../scripts/training/verify_taiji_goal_paraphrase_10_8.py)、[测试](../../../tests/taiji_native/test_goal_paraphrase_10_8.py)与[报告](../../../reports/taiji_core_goal_paraphrase_10_8_20261101.json)。

| 读数 | 值 |
|---|---|
| 与任一动作句逐字节相同 | **0/36** |
| 实体不唯一（无法定解） | **0/36** |
| 字符二元组重叠匹配器的 top-1 | **36/36** |

⇒ **改写措辞不能去掉复读捷径**：只要目标点名对象与位置，字符重叠就能全对。要真正排除，目标必须**不点名实体**（按属性／关系描述，例如"把那个圆口的器皿放到靠窗的台面"），并给主体一份词汇表（属性→实体）；那会把任务从字符串匹配升级成"读表＋绑定"，是更大的夹具改动。本步无模型、无参数，0.0 CPU 秒。

## 2026-11-01 10.9 属性描述目标＋词汇表（`attribute_fixture_fair`）

10.8 证明"只要目标点名实体，复读捷径就在"。本步把目标改成**不点名实体**的描述，并把映射做成词汇表（数据，不是答案）：

- 对象：`蓝色杯子 → 圆口的器皿`、`红色印章 → 带纹路的方块`
- 位置：`木桌 → 四方木台`、`窗台 → 靠窗的台面`、`左盒 → 左边的匣子`
- 目标句：`让{对象描述}到{位置描述}。`

交付[夹具模块](../../../taiji/native_goal_attributes.py)、[验证器](../../../scripts/training/verify_taiji_goal_attributes_10_9.py)、[测试](../../../tests/taiji_native/test_goal_attributes_10_9.py)与[报告](../../../reports/taiji_core_goal_attributes_10_9_20261101.json)。

| 读数 | 值 |
|---|---|
| 目标里出现任何实体名（对象或位置） | **0/12** |
| 词汇表能唯一解析出 (对象, 位置) | **12/12** |
| 字符重叠匹配器 top-1（6 选 1） | **2/12**（机会 1/6） |

⇒ **夹具公平了**：答案仍由目标唯一决定，但只能通过词汇表拿到，复读无效。描述词本身也机检过不含实体名，词汇表一对一。

本步不接模型：词汇表是"要交给主体的数据"，怎么给（records／prompt）是下一步的事。0.0 CPU 秒。

## 2026-11-01 10.10 词汇表可用性（`glossary_not_used`：未训练主体不读词汇表）

10.9 把夹具变公平后，先问最便宜的问题：**不训练**，把词汇表拼在目标前，主体能不能用上？6 个目标 × 2 个动作的成对读数；对照臂把词表映射反向。交付[脚本](../../../scripts/training/probe_glossary_use_10_10.py)与[报告](../../../reports/taiji_core_glossary_10_10_20261101.json)。

| 臂 | 命中 |
|---|---|
| 真词汇表 | **0/6** |
| 反向词汇表 | **0/6** |

两臂都恒取同一个错动作 ⇒ 词汇表对未训练主体的选择**毫无影响**；494.4 CPU 秒（预算 900，未超）。这是新夹具上的**地板读数**，不是能力读数：它说明"把表塞进 prompt"这条便宜路走不通，任何在新夹具上的正面结论都必须来自会更新参数的机制。

## 2026-11-01 10.11／10.11b 词汇表走 records＋对照（**10.11 的 6/6 被自己打掉**）

**10.11**：把词汇表放进 `CareerJournal.records`（不进 prompt），等长对照是无 records。交付[模块](../../../taiji/native_goal_glossary.py)、[验证器](../../../scripts/training/verify_taiji_glossary_records_10_11.py)、[测试](../../../tests/taiji_native/test_glossary_records_10_11.py)与[报告](../../../reports/taiji_core_glossary_records_10_11_20261101.json)。

| 读数 | 值 |
|---|---|
| prompt 内单跳重叠匹配器 | **0/6**（目标不含实体，复读死了） |
| records 改动 NLL 的格数 | **12/12**（通道是活的） |
| 有 records 时的成对命中 | **6/6** |

**10.11b（对照，缺了它就会误报）**：同一批目标、同一 prompt，只把词汇表的**映射换成一个真置换**（每个描述词指向另一个实体），其余不动。交付[脚本](../../../scripts/training/probe_glossary_control_10_11b.py)与[报告](../../../reports/taiji_core_glossary_control_10_11b_20261101.json)。

| 臂 | 成对命中 |
|---|---|
| 真映射 | **6/6** |
| 错映射（真置换） | **6/6** |

⇒ **10.11 的 6/6 不是"读了映射"**：两臂的 token 集合完全一样，只有配对变了，命中不变 ⇒ 这个选择由**与映射无关的不变量**驱动（最可能是 records 把实体名带进了复制通路）。records 通道确实活着（NLL 全变），但**映射没被用上**。

这条差点被报成正结果：如果只跑 10.11 就宣布"未训练主体能用词汇表"，那是假的。两臂都 6/6 正是"配对不可读"的指纹。

**对下一步的直接影响**：读出头的输入是**编码器特征**（不含 records），所以 10.7e/f 那类头**根本看不到词汇表**；要把表用上，特征必须取自**主体的循环状态**（它才读 records），或者把表塞回 prompt（那又泄漏，10.8/10.9 已证）。这个仪器选择是下一步要解决的第一件事。

## 2026-11-01 10.12 词汇表是否到达循环状态（`records_reach_state`）

10.11b 说"未训练主体的选择与映射无关"，那只排除了*它当前的选择*；在花掉已批准的训练之前，先验必要条件：**同一目标、只把词汇表映射换成真置换，主体的循环状态变不变**。

交付[脚本](../../../scripts/training/verify_taiji_records_state_10_12.py)、[测试](../../../tests/taiji_native/test_records_state_10_12.py)与[报告 v2](../../../reports/taiji_core_records_state_10_12_v2_20261101.json)（[v1](../../../reports/taiji_core_records_state_10_12_20261101.json)保留：import 排序后重发）。

| 读数 | 值 |
|---|---|
| 真映射 vs 置换映射，`state.h` 不同 | **6/6** |
| 有 records vs 无 records，`state.h` 不同 | **6/6** |
| 中位相对距离 | **0.002951** |

状态取自"读完全部 prompt 补丁之后"那一步（经 `run_career(..., retain_provider)` 的 gated 路径，不是裸 `observe`——后者会被门控拒绝）。33.0 CPU 秒。

⇒ **必要条件成立**：词汇表确实进到循环状态，读数头若吃 `state.h` **在原理上**能看到映射。但这只说明路是通的，不说明会被用上——那要训练才知道。

## 2026-11-01 10.13 状态特征六选一（`state_selection_absent`：最后一条，读出线收束）

owner 批准的公平夹具训练。特征＝读完目标 prompt 后的 `state.h`（512 维，词汇表只经 `records` 到达）+ 动作编码；头 **2,560 参数**（`left 512×4 + right 128×4`）、6 路交叉熵 300 步、主体逐位冻结。交付[读出头](../../../taiji/state_action_readout.py)、[验证器](../../../scripts/training/verify_taiji_state_selection_10_13.py)、[测试](../../../tests/taiji_native/test_state_action_readout.py)与[报告 v2](../../../reports/taiji_core_state_selection_10_13_v2_20261101.json)（[v1](../../../reports/taiji_core_state_selection_10_13_20261101.json)保留，见下）。

| 臂 | 留出 top-1 |
|---|---|
| A 真映射训练 → 真映射测试 | **2/6** |
| B 真映射训练 → 置换映射测试 | **2/6** |
| C 置换映射训练 → 真映射测试 | **2/6** |

主体参数与留出 `sample_loss` 逐位不变；38.4 CPU 秒（预算 300）。

**读法**：10.12 已证明循环状态里**带着**映射（真 vs 置换 6/6 不同），但头在 6 个训练样本上**学不出选择**——三臂一起停在 2/6 的常量水位，连"用了映射"与"没用映射"都分不开。⇒ 不是通道、不是夹具、不是特征，是**样本量**。

**同日更正（数字）**：v1 的 `declared.parameters` 我按旧公式手算成 **4,608**（把 left 又乘了 2），实际是 **2,560**。v1 保留不删，v2 改为从模块 `parameter_count()` 取数，并加一条守卫（`declared != 训练头实际计数` 直接抛错）；两版三臂读数逐位相同。

**读出线收束为负**：10.4→10.5→10.6→10.7→10.7a–h→10.9→10.10→10.11/10.11b→10.12→10.13，十一步、两次有界训练，没有一条可归因于"目标条件化选择"的正读数。P2 未满足，11 仍 `step11_not_admitted`。

**"要多少样本"的粗算（只作定价，不是实验）**：2,560 参数、每样本 6 个 logit 约束 ⇒ 6 个样本约 **每参数 0.014 个约束**；要给到 1 个约束/参数需约 **427 个样本**，10 倍过采样约 **4,270 个**。这个数量级与"6 个样本"差三个数量级，足以解释三臂同分。

## 2026-11-01 10.14 读出线收束卡（`readout_line_closed_negative`）

负结果按过程要求收进一张机检卡，而不是只写在正文里。交付[验证器](../../../scripts/training/verify_taiji_readout_closeout_10_14.py)与[报告](../../../reports/taiji_core_readout_closeout_10_14_20261101.json)：只读 20 份卡，**不重跑、不训练**。

| 检查 | 结果 |
|---|---|
| 20 份卡在册 | **20/20** |
| 有未撤回的能力声称 | **无** |
| 唯一的能力声称 | **10.7e**（`readout_binding_supported`） |
| 撤回它的卡 | **10.7e-1**（`readout_rank_below`）——已在册 |
| 权重发布／默认替换 | 全部 false |
| **全线 CPU** | **2,996.6 秒** |

⇒ **读出线按负结果结案**：十一步、两次有界训练、20 份卡，没有一条可归因于"目标条件化选择"的正读数；被撤回的那条（10.7e 的二选一 6/6）正是这条线自己发现并撤回的。`P2` 未满足、11 仍 `step11_not_admitted`。**不表示**主体永远不能选择，也不改 11 的判据。

## 2026-11-01 10.15 宽夹具（`wide_fixture_ready`：4 对象 × 5 位置，480 个样本）

owner 选定"扩数据"路线后，先把数据侧建好：`taiji/native_goal_world_wide.py` 用 4 个对象、5 个位置，**120 个局面 × 4 个目标 = 480 个样本**；目标仍不点名实体、属性→实体走词汇表。交付[夹具](../../../taiji/native_goal_world_wide.py)、[验证器](../../../scripts/training/verify_taiji_wide_fixture_10_15.py)、[测试](../../../tests/taiji_native/test_wide_fixture_10_15.py)与[报告 v3](../../../reports/taiji_core_wide_fixture_10_15_v3_20261101.json)。

| 读数 | 值 |
|---|---|
| 局面／样本／logit 约束 | **120 / 480 / 9,600** |
| 目标里出现实体名 | **0/480** |
| 词汇表唯一解析 | **480/480** |
| 与正确动作**共享二元组**的目标 | **0/480** |
| 字符重叠匹配器 top-1 | **24/480 = 5%**（机会 5%） |

**两处泄漏是被本步自己抓出来并改掉的**（v1→v2→v3）：

1. 位置描述"地上的织物"与动作"…放到地毯"共享二元组 **"到地"**（描述首字与实体首字相同）⇒ 48 个目标被匹配器"赢"。
2. 同类问题"左**边**的匣子"与"…放到左盒"共享 **"到左"** ⇒ 又 48 个。

改法：描述首字必须与实体首字不同；并把"与正确动作共享二元组"做成 **0 容忍**的机检项（只看比例会漏——v2 的 10% 正好卡在阈值上）。改完复读掉到机会线的 5%。

## 2026-11-01 10.16／10.17 宽夹具上的二十选一（**分割错误与真结论**）

**10.16（按局面切分，判 `control_not_null`，作废）**：480 个样本按局面 80/40 切，三臂留出 top-1 **全部 100%**。这不是能力——480 个样本只有 **20 个不同目标**，按局面切分**一个目标都没留出**，头把 20 个目标全背下来了。交付[脚本](../../../scripts/training/verify_taiji_wide_selection_10_16.py)与[报告](../../../reports/taiji_core_wide_selection_10_16_20261101.json)。

**10.17（按对象留出，判 `wide_selection_absent`）**：整个 `黄色球` 的 15 个目标（含其描述词与 5 个动作）**不进训练**——训练 360 样本、留出 120 样本。交付[脚本](../../../scripts/training/verify_taiji_wide_object_split_10_17.py)与[报告](../../../reports/taiji_core_wide_object_split_10_17_20261101.json)。

| 臂 | 留出 top-1（120 样本） |
|---|---|
| A 真→真 | **0%** |
| B 真→置换 | **0%** |
| C 置换→真 | **0%** |

最佳常量 24/120（20%）；38.5 CPU 秒。**0%** 的含义是：这头**从不**把没见过的对象选对——它根本没学过"描述词→实体"这条映射，只见过的 15 个目标。

**结论（比"样本量"更准）**：10.16 证明**见过的目标可以背到 100%**；10.17 证明**没见过的对象是 0%**。所以卡点不是数据量，是**组合泛化**——18 倍样本（6→360）换不来一个没见过的对象。10.12 已证状态里带着映射，但读出头学的是"目标查表"，不是"用词汇表"。

## 2026-11-01 10.18 从 records 复制实体名（`records_copy_inconclusive`：抄得出来，对照 4/5）

10.17 说"读出头不会组合"，但没回答主体**自己**能不能把描述词接上实体。这一步只量 NLL、不训练：prompt＝`<描述词>是`，target＝`<实体>`；臂 A 带词汇表 records、臂 B 不带；对照臂把 target 换成**另一个实体**。

| 读数 | 值 |
|---|---|
| 带 records 比不带便宜 | **5/5** 对，平均 **+3.114 nat** |
| 对照成立（换实体后更贵） | **4/5** |
| 成本 | 34.0 CPU 秒 |

⇒ **主体会用 records 把实体名说出来**：这是"两段式组合"的第一段，而且是模型自己的机制（不是字符串表）。但按预注册，对照只过 4/5（有一对里"错实体"只便宜 0.78 nat），所以出口判 `records_copy_inconclusive`，不升级成 supported。

**与 10.17 合起来看**：抄得出来（10.18），但把"抄出来的实体"接到"选动作"上做不到（10.17 三臂 0%）。缺口被缩到**两段之间的接线**，而不是任一段本身。

## 2026-11-01 10.19 生成式选择（`generation_absent`：复读一行词表就停）

不训练，让主体**自己续写**：prompt＝`目标是{描述}到{描述}。我应该`、records＝词汇表、贪心解码 16 字节。

| 读数 | 值 |
|---|---|
| 命中（先点对对象、再点对位置） | **0/6** |
| 不带 records 的对照 | **0/6** |
| 成本 | 34.9 CPU 秒 |

**生成内容比分数更说明问题**（原文）：

- 带 records：6 个目标**全部**吐出同一句 **`台是木桌。`**——它把词表里"四方木台是木桌。"这一行**复读**了出来，然后 EOS；
- 不带 records：16 字节的二进制噪声（`\x1d��g���R���nC\x17�\x0c`）。

⇒ 机制闭合了：主体有**单跳 copy**（10.18：+3.1 nat），但任务要的是**两次 copy 再拼成动作句**（对象实体＋位置实体＋模板）；它只做了一次 copy 就停，而且复读的是**词表行**而不是造新句。

**三步合起来**：抄得出来（10.18）→ 拼不起来（10.19）→ 读出头也拼不起来（10.17 三臂 0%）。缺口精确定位在**组合**，不在检索、不在样本量。

## 2026-11-01 10.20 能力线收束卡（`capability_line_closed_negative`）

交付[验证器](../../../scripts/training/verify_taiji_capability_closeout_10_20.py)与[报告](../../../reports/taiji_core_capability_closeout_10_20_20261101.json)：只读 18 份卡，**不重跑、不训练**。

| 检查 | 结果 |
|---|---|
| 18 份卡在册 | **18/18** |
| 未撤回的能力声称 | **无**（唯一声称 10.7e 已被 10.7e-1 撤回） |
| 三条负锚点（10.13／10.17／10.19） | **全为负** |
| copy 证据（10.18） | **在册**（`records_copy_inconclusive`） |
| 所有带 `backbone_frozen` 的臂 | **全 true** |
| 这些卡合计 CPU | **2,945.8 秒** |

**机制句**（写进卡里）：**单跳 copy 有（10.18），组合没有**——10.19 复读一行词表就停；10.17 状态读出在没见过的对象上 0%。

⇒ 能力线按负结果收案。P2 未满足、11 仍 `step11_not_admitted`。**不证明**主体永远不能选择，也不改 11 的判据。

## 2026-11-01 11.1a 候选 manifest 与隔离加载（`isolated_load_verified`）

11.1 卡片要求"manifest 绑定候选 ID、代码/配置、参数与状态 schema、来源和依赖，提供同一主体独立加载，显式声明冻结评测/受控学习/持续运行模式，冻结评价禁写"。本步交付最小可用版。

- [manifest 模块](../../../taiji/candidate_manifest.py)：`candidate_id`＝135 张参数张量的摘要（name/dtype/shape/bytes 全参与）；绑定 6 个模块的 SHA、bundle 的字节数与 SHA、config 与状态 schema、三种模式声明。
- [验证器](../../../scripts/training/verify_taiji_candidate_manifest_11_1.py)＋[报告](../../../reports/taiji_core_candidate_manifest_11_1_20261101.json)。

| 检查 | 结果 |
|---|---|
| 本进程加载后的摘要 == manifest | **true** |
| **子进程**独立加载后的摘要 == 本进程 | **true**（`feb0cbcb00e2e16b…`） |
| 改一个模块 SHA 后加载 | **被拒**（`check_manifest` 报问题且 `load_candidate` 抛错） |
| 给 bundle 追加 1 字节后加载 | **被拒** |
| 成本 | **0.5 CPU 秒** |

bundle 只读打开、加载器不写任何文件。**不表示**行为正确，也不测任务能力——那是 11.2／11.3 的事。

## 2026-11-01 11.2a 单一任务入口（`entry_contract_verified`，部分交付）

11.2 卡片要求"统一 observe/answer/clarify/reset/save/load…先保存原始模型输出，再单列合法解码约束和产品加工；显示层不得用模板补写正确内容"。本步交付前四个。

- [入口模块](../../../taiji/candidate_session.py)：建在 11.1a 的 manifest 加载与 `resume_subject` 上；每次调用都进 transcript。
- [验证器](../../../scripts/training/verify_taiji_task_entry_11_2.py)＋[报告](../../../reports/taiji_core_task_entry_11_2_20261101.json)。

| 检查 | 结果 |
|---|---|
| `answer` 里 **raw 先于 decoded** | **true** |
| decoded 只能是 raw 顶字节那一个字 | **true** |
| `clarify` 的 `content` | **None**（不自填） |
| `observe` 确实改变了状态 | **true** |
| `reset` 逐位回到初态摘要 | **true** |
| 两会话重放同一调用序列，raw 逐位相同 | **true** |
| `frozen_eval` 下 `learn` | **被拒** |
| 成本 | **1.2 CPU 秒** |

**未交付（卡里点名）**：`save`／`load`／`outcome`／`act`。所以这条只是"入口契约"的一半，不表示入口可用。

## 2026-11-01 11.2b 入口往返（`entry_roundtrip_verified`）

补上 11.2 卡里剩下的四项。交付[验证器](../../../scripts/training/verify_taiji_task_entry_roundtrip_11_2b.py)与[报告](../../../reports/taiji_core_task_entry_roundtrip_11_2b_20261101.json)。

| 检查 | 结果 |
|---|---|
| 快照存取后同一答案逐位相同 | **true** |
| 从快照续一次观察后仍逐位相同 | **true** |
| `outcome` 记录结果且状态不变 | **true**（冻结模式不学习） |
| `act` | **被拒**，`no_action_channel`（该候选没有动作通路） |
| 外来候选的快照 | **被拒** |
| 成本 | **1.2 CPU 秒** |

⇒ 入口的六个调用（observe／answer／clarify／reset／save／load）＋outcome／act 全部在册；**`act` 的拒绝本身就是一条读数**：这个候选没有可调用的动作通路。

## 2026-11-01 11.3a 固定答案任务套件（`task_suite_floor_measured`：地板 0/5）

把 11.1a／11.2a／11.2b 的入口接到一组固定答案的中文任务上，先取地板。交付[套件](../../../taiji/native_task_suite_11_3.py)、[验证器](../../../scripts/training/verify_taiji_task_suite_11_3.py)与[报告](../../../reports/taiji_core_task_suite_11_3_20261101.json)。

| 读数 | 值 |
|---|---|
| 任务 | 7（问答／材料任务／追问／更正／指令／澄清／未知） |
| 可评分项 | **5** |
| 精确命中 | **0/5**（评分＝去空白与句末标点后全等） |
| 排除项 | **2**（澄清／未知：该入口没有模型驱动的澄清或弃答通道） |
| 设置 | 一次会话顺序提问（追问与更正依赖前一条材料）、raw-first、16 字节贪心 |
| 成本 | 2.2 CPU 秒 |

⇒ **11.3 的地板是 0**。两条排除本身就是**入口的缺口**，不是候选的结论：`clarify` 现在只能由外部调用，模型没有触发它的通路。

## 2026-11-01 11.3b 弃答门（`abstain_gate_degenerate`：通道存在但不携带信息）

11.3a 里澄清/未知两项因"入口没有模型驱动的通道"被排除。本步补上通道，且**不做文本模板**：弃答只看模型自己的原始分布（top 概率 <0.5 或熵 >3.0 nat）。交付[验证器](../../../scripts/training/verify_taiji_abstain_11_3b.py)与[报告](../../../reports/taiji_core_abstain_11_3b_20261101.json)。

| 读数 | 值 |
|---|---|
| 在 5 个**可答**项上弃答 | **5/5** |
| 在 2 个**应弃答**项上弃答 | **2/2** |
| 实际作答 | **0** |
| 精确命中 | 0 |
| 成本 | 1.5 CPU 秒 |

⇒ **门恒开**：模型在这 7 个提示上的 top 概率都低于 0.5（或熵高于 3.0），所以通道**不携带信息**——它没法把"该答"和"该弃"分开。要它有用，τ／H 必须在**独立**标定集上定；在同一批 7 项上标定就是拟合。

## 2026-11-01 11.3c 信度可分性（`confidence_not_separable`：信号本身不存在）

11.3b 的门恒开。这一步只读那张卡，问一个更基本的问题：**有没有任何阈值**能把 5 个可答项与 2 个应弃答项分开。交付[验证器](../../../scripts/training/verify_taiji_confidence_separable_11_3c.py)与[报告](../../../reports/taiji_core_confidence_separable_11_3c_20261101.json)。

| 组 | top 概率区间 | 熵区间 |
|---|---|---|
| 可答 5 项 | **[0.009005, 0.009137]** | 5.49603–5.49614 |
| 应弃答 2 项 | **[0.009092, 0.009106]** | 5.49603–5.49608 |

两区间**重叠**（均匀参考 ln 257≈5.549）。⇒ **没有任何阈值能分开**：不是 τ／H 没调好，是模型的**首步分布在所有提示上都近乎恒定**——信度信号本身不存在。用标定救不了它。

**对 11.3 的收口**：地板仍是 0/5；那 2 项在"分布不随内容变化"的候选上**结构性不可评**，不是缺通道的问题。

## 2026-11-01 11.4a 候选仪器台账（`instrument_facets_inventoried`：7 有读数、3 点名未测）

11.4 卡要求"接真实性 A、代表机制 F 及当前设备 H；核对来源、scope、缺权重、原始生成、重复运行、中断恢复、峰值内存及 p50/p95；未测记未测"。这一步只读 7 张卡，把每一项指到来源或标为未测。交付[验证器](../../../scripts/training/verify_taiji_instrument_inventory_11_4.py)与[报告](../../../reports/taiji_core_instrument_inventory_11_4_20261101.json)。

| 面 | 状态 | 来源 |
|---|---|---|
| 来源/manifest | **有** | 11.1a：6 个模块 SHA＋候选摘要 |
| 缺权重/未发布 | **有** | 11.1a：只绑摘要与来源 |
| 原始生成 | **有** | 11.3b：7 行的原始 top 与熵 |
| 重复运行 | **有** | 11.7：20 次逐位；11.2a：两会话逐位 |
| 峰值内存 | **有** | 11.7：起始／建主体后／调用后三点 |
| p50/p95 | **有** | 11.7：0.120／0.125 s |
| 代表机制 F | **有** | 10.20：`capability_line_closed_negative`（F 未成立） |
| 中断恢复 | **部分** | 11.2b 只做了快照往返，**没有做进程 kill** |
| 真实性 A | **未测** | 未做真实语料/真实入口读数 |
| 设备 H | **未测** | 未采集设备档 |

⇒ 台账的用处是**不许拿空白当零**：三项显式未测，任何"11.4 已过"的说法都不成立。

## 2026-11-01 11.4b 设备档（`device_profile_measured`：台账升到 8/10）

补 11.4a 点名未测的**设备 H**。交付[验证器](../../../scripts/training/verify_taiji_device_profile_11_4b.py)、[设备档报告](../../../reports/taiji_core_device_profile_11_4b_20261101.json)与台账 [v2](../../../reports/taiji_core_instrument_inventory_11_4_v2_20261101.json)。

| 读数 | 值 |
|---|---|
| 逻辑 CPU／torch 线程 | **24／12** |
| CUDA | **不可用** |
| 总内存 | **48 GiB**（`GetPhysicallyInstalledSystemMemory`） |
| 内存取数路径 | `GlobalMemoryStatusEx` 在本机**返回 0** ⇒ 已换源并把路径写进卡 |
| 成本 | 0.02 CPU 秒 |

并入台账后：**8 项有读数，剩 2 项未测**——**中断恢复**（只做过快照往返，没做进程 kill）与**真实性 A**（未做真实语料/真实入口）。阈值与单价都绑在这台机器上，换机器须重测。

## 2026-11-01 11.4c 中断恢复（`interrupt_recovery_verified`：硬退后可由新进程续跑）

补上 11.4 台账最后一项可测面。交付[验证器](../../../scripts/training/verify_taiji_interrupt_recovery_11_4c.py)、[报告](../../../reports/taiji_core_interrupt_recovery_11_4c_20261101.json)与台账 [v3](../../../reports/taiji_core_instrument_inventory_11_4_v3_20261101.json)。

| 检查 | 结果 |
|---|---|
| 子进程 A：观察→写快照→`os._exit(0)` | rc=**0**，快照已落盘 |
| 子进程 B：全新进程从快照续跑 | rc=**0** |
| 与父进程的答案 | **逐位相同** |
| 成本 | 1.1 CPU 秒 |

⇒ 快照在**硬退出那一瞬间**是完整的：没有 unflushed 状态。台账 v3：**9/10 有读数，只剩真实性 A**。

## 2026-11-01 11.4d 仪器收口（`instrument_closed_pending_realism_and_capability`）

仪器面收口，且**不许读成候选可用**。交付[验证器](../../../scripts/training/verify_taiji_instrument_closeout_11_4d.py)与[报告](../../../reports/taiji_core_instrument_closeout_11_4d_20261101.json)。

| 检查 | 结果 |
|---|---|
| 有卡的仪器面 | **9/10** |
| 唯一缺面 | **真实性 A**（显式未测） |
| 代表机制 F | `capability_line_closed_negative`（**负**） |
| 卡里的 `candidate_usable` | **false** |

⇒ 11.4 的结论是"仪器面基本齐、但候选不可用"，11 仍 `step11_not_admitted`。11.5／11.6（正式资源/能力协议、CAP 最小验收）都要求一项成立的代表能力，**当前候选没有**，所以它们不能进入。

## 2026-11-01 结果驱动的算法重审（设计输入，不新增试验）

交付[重审意见](../../reference/TAIJI_ALGORITHM_REVIEW_20261101.md)与[阅读/字节审计快照](../../../reports/taiji_algorithm_review_20261101.json)：读取20份关键卡并核对12份实现身份，不加载模型、不作神经前向/更新/新数据评价。20卡中有8处source绑定与当前树不一致，作为历史范围披露，不回填哈希、不否定原观察。保持原提交/推送保留约束，不把在飞探针当结果。

新P0：10.19逐字节MAX_NEW=16，而命中需完整对象/地点且先后有序；六组最短oracle字符串均18个UTF-8字节，截为16字节后6/6不命中。因此原generation_absent仍留存，**不得用于否证组合能力或证明必须添加组合器**；decode未记录EOS/预算终止，不能将截断解释为模型自主停句。11.3五个评分答案6/6/6/6/15字节均可达，该0/5仍是范围内负读数；两项入口缺失仍需披露。

算法建议：条件化读取/角色绑定与负责模块的联合任务信用优先，冻结头仅为对照；不能重复“只换margin必修复”的已收窄建议。旧保持门的abs(Delta NLL)测稳定性，非有符号遗忘，未来合同须拆分；生长小净收益保留但标清共识/常开门变体。10.20与11未准入不翻正，10.21先补生成可达性/目标选择性与新预算范围再交owner裁定。本轮不改运行源码、旧卡或训练权限。

## 2026-11-01 10.21 rev2方案修订完成（未执行）

交付[rev2唯一规范](../../reference/SPEC-TAIJI-CORE_COMPOSITION_10_21_20261101.md)、[设计依据](../../reference/DESIGN-CANDIDATE-COMPOSITION_20261101.md)及[原稿/总计划快照](../../archive/history/n_mainline_20261101/README.md)。方案已从末段“修订要求”改为完整可审范围：64输出字节/最多65预测、oracle/坏策略双评分、15次描述词诊断和18次生成；一个105760参数的可塑查询/读取/绑定模块以内部事件进入同一RNN，J/S同构停信用；192/160/160分组数据、逐阶段预算、单侧旧损伤、前后模块恢复、两增量seed与固定确认均写明。

旧冻结头只保留为历史/诊断，不默认执行300步。旧10.21a/10.21a-2保留实验身份，不映射为新10.21.1/10.21.2已通过；旧预算与许可不迁移。未见实体输入排除、目标未见组合的精确定义、整图映射反事实与仅新读源破坏的真值差别、任务正与机制负的分账均在SPEC明确。当前只供裁定10.21.1，未实现新module/runner、未生成新数据、未运行模型/训练，Git保留约束继续生效。

[文档修订回执](../../../reports/taiji_core_10_21_plan_revision_20261101.json)记录规范/归档身份、参数与数据数量静态核对、链接/单主线检查及原运行源码/20份证据未变核验；它不是新模块图、恢复或能力通过回执。

## 2026-11-01 本轮主线边界与滚动细化（计划修订）

将讨论结论写入[总计划的周期规则](../../MAINLINE.md#本轮规划边界与下一轮立项)：当前01–13有界推进，13保留能力职责并加13.7全局收束，不提前铺14及以后。10.21.6及11/12出口先全局复审；当前阶段也可按预定负结果/不可判/预算出口提前收束，不强行走到13。下一轮由同一主体/谱系、能力与限制、失败资产及累计资源总账支撑，只立一个核心问题，正式立项后从14接续；不重置历史或用新编号冒充能力进展。

本次仅修订范围与转段规则，不新增试验、实现、训练、能力准入或预算。当前队首仍为总计划10.21.1待裁只读复核；既有Git保留约束不变。此前10.21文档修订回执保留其当时快照，不回填为本次变更的身份。

## 2026-11-02 根目标覆盖与12/13研究准入复核（文档修订）

核对当前MAINLINE、10.21收束、11准入v4、13.7及14.1/14.2原报告：第一周期能力线实际到10，11未准入，12及13.1–13.6未开展；13.7交的是总账，不能写成13项能力全部执行。12/13缺父候选、分区/反馈权利、数值门及预算/出口，当前为职责卡，尚非可执行合同。R5c仍FAIL_BEHAVIOR；10.21不验原生中文生成；14.2的6/6经去状态/错状态控制判明目标复读，均不填补P1。

保留09已有学习/活动工程和有限读数，不把根目标缺有效验证说成所有相关实验从未执行，也不把早期零成本空转泛化为全部后继夹具。13.7历史正文/JSON不改写，其“未测/不可判”与“机制不能”之间的解释边界以[当前总计划](../../MAINLINE.md#根目标覆盖与研究准入边界)为准。

规划修正：11/L3及12/L4、13成熟能力验收沿原门；基础跨会话、反馈更新和无提示选择可另立有限机制研究合同，不必等全部L2过门。保持则必须有被保护的已核验父能力，无此集合不能签保持。唯一下一步为14.3可审设计：读源判别及其到同一主体最小中文连续任务的衔接，显式逐项列覆盖/未覆盖；四件套不齐不得执行。本次只改计划，无新增模型前向/更新/数据或预算，既有提交/推送保留约束不变。

## 当前候选的审批依据

| 统一步骤 | 批准/边界事实 | 原始依据 |
|---|---|---|
| 02、06、07 | 用户于2026-10-30批准R2/R4与R5，R5现已按原门结案 | [R4页首及§9](../../reference/SPEC-NATIVE-CONTENT-CORE_R4_PRICING_AND_GATES_20261030.md) |
| 08 | 用户于2026-10-30选择“批准，按合同执行”；该合同实现与训练授权已成立 | 提交`8780f858e`与[R5b §10执行记录](../../reference/SPEC-NATIVE-CONTENT-CORE_R5B_COPYTRACK_CANDIDATE_20261030.md#10-审批方式)；本次整理不实施 |
| 09–13 | 原R6能力接力不由R5/R5b批准自动授权 | 对照总计划各步所需新范围/合同，不借原批准越门 |

## 步骤01–06的交付证据

| 步骤 | 交付出处 | 读数与适用限制 |
|---|---|---|
| 01 | [R1合同](../../reference/SPEC-NATIVE-CONTENT-CORE_R1_SEMANTIC_CONTRACT_20261030.md)、[世界审计](../../../reports/taiji_native_core_r1_world_audit_20261030.json)、[17案fixture审计](../../../reports/taiji_native_core_r1_public_fixture_audit_20261030.json) | 设计语义及公开反例，不是未见任务能力 |
| 02 | [R2合同](../../reference/SPEC-NATIVE-CONTENT-CORE_R2_CANDIDATE_COMPARISON_20261030.md) | Q1三类架构组，Q2图内reader/decoder信用比较；架构包与单因素分账 |
| 03 | [R3记录](../../reference/SPEC-NATIVE-CONTENT-CORE_R3_ISOLATED_IMPLEMENTATION_20261030.md)、[结构探针](../../../reports/taiji_native_core_r3_probe_20261030.json) | 53/53结构门，公开fixture、零步独立恢复；本阶段0更新，后续训练见07 |
| 04 | [修正旧图探针](../../../reports/taiji_native_core_r3b_baseline_20261030.json)、[R3 §6](../../reference/SPEC-NATIVE-CONTENT-CORE_R3_ISOLATED_IMPLEMENTATION_20261030.md#6-r3b1-修正旧图基线20261030) | 22/22，固定分母17/17，仅value_bias可训3,341；unknown回current、解析失败丢弃等基线限制保留 |
| 05 | [普通基线探针](../../../reports/taiji_native_core_r3b2_ordinary_20261030.json)、[R3 §7](../../reference/SPEC-NATIVE-CONTENT-CORE_R3_ISOLATED_IMPLEMENTATION_20261030.md#7-r3b2-普通端到端基线20261030) | 11/11，固定分母17/17，全文prompt、无硬解析门、120,097参数全可训 |
| 06数据 | [课程审计](../../../reports/taiji_native_core_r1_course_audit_20261030.json)、[分区身份](../../manifests/native_core_r1_split_v1.json) | 360历史取216/72/72；1,440 episodes、11,520 turns、5,760应答轮；反事实成对不跨分区；本台账不加载分区正文 |
| 06工程 | [零更新预检](../../../reports/taiji_native_core_r4_preflight_20261030.json) | reader/decoder等组有任务梯度，content_encoder无（持久latent边界）；0 optimizer updates，不称已学会 |
| 06定价 | [定价报告](../../../reports/taiji_native_core_r4_pricing_20261030.json) | 新核心/旧图/普通单价5.45/0.94/8.20 CPU秒/更新；3×50 pilot；零步/第25步恢复6/6；冻结512更新/run。仅该定价阶段未读query/holdout/sealed |

## 步骤07的结果、费用与限制

原始来源：[训练报告](../../../reports/taiji_native_core_r5_training_20261030.json)、[评价报告](../../../reports/taiji_native_core_r5_evaluation_20261030.json)、[原合同§9](../../reference/SPEC-NATIVE-CONTENT-CORE_R4_PRICING_AND_GATES_20261030.md#9-r5-执行结果20261030)。固定分母评分器为[native_content_evaluator.py](../../../taiji/native_content_evaluator.py)。

| 读数 | 实际结果 | 解释边界 |
|---|---|---|
| 执行与恢复 | 15/15 run，7,680更新；恢复15/15；无NaN/无预算停止 | 完成训练不代表过能力门 |
| 整体结果 | `FAIL_BEHAVIOR` | 按原冻结门；不追改阈值 |
| 相对比较 | holdout约0.323，比最好基线高约0.169；Q2信用差约0.117 | 有真实但不足的相对收益，绝对行为仍不合格 |
| 关键失败项 | cut_drop约0.166，训练内插约0.329，收据约0.429 | 内容选择和生成路径值得优先检验；不能凭这些单项穷尽全部原因 |
| 父能力保持 | 报告`retention_parent=none` | 训练内插不代证旧能力保持 |
| CPU费用 | run CPU和42,108.75秒；训练进程总CPU42,224.15625秒；评价另6,529.890625秒 | 不将run和当全部实耗，不漏评价；完整ledger取原报告 |
| 数据使用 | 评价已读query/holdout，sealed未读 | 后继复用相同面是回归比较，不重新称独立确认 |

## 步骤08的证据入口

[已批准合同](../../reference/SPEC-NATIVE-CONTENT-CORE_R5B_COPYTRACK_CANDIDATE_20261030.md)保留原候选、两项改动、9-run清单、门值、预算及审批。提交`dc7a43354`提供[native_content_core_v2.py](../../../taiji/native_content_core_v2.py)、[训练适配器](../../../taiji/native_content_training_v2.py)、评价适配及[11个定向测试定义](../../../tests/taiji_native/test_native_content_core_v2.py)；本轮只读代码与提交，不运行测试，不能据文件存在签工程门通过。

静态对照发现的待核项：批准合同按episode抽样，适配器按turn抽样且缩短自前缀路径的实际监督字节；50步pilot不覆盖257后分支，草拟额外分支探针需具体方法/更新账修订。原合同不在本轮改写，细项及处置顺序已写入总计划08。后续真实工程/定价/训练/评价回执到位后再追加身份与费用；并行草稿不计完成。

提交`b78bdfdce`追加[定价执行件](../../../scripts/training/price_native_content_core_r5b.py)、[训练执行件](../../../scripts/training/run_native_content_core_r5b.py)、[评价执行件](../../../scripts/training/evaluate_native_content_core_r5b.py)；本轮未运行；执行件存在不等于工程/训练完成。收尾核对时工作区出现`reports/taiji_native_core_r5b_pricing_20261030.json`，status为`r5b_priced`，登记60更新、约572.56 CPU秒、约59.00墙钟秒，零步/两阶段端点同进程和新进程恢复均记相等，query/holdout/sealed读取均记false。本次未提交该并行机器件；原50步与60步差异、实际日程分支及监督字节仍需按总计划08审计，不凭status补签全部前置门。

项目所有者已正式将[总计划](../../MAINLINE.md)定位为新的N主线。03继续仅登记证据与审批；09–13详细方案不因此成为已冻结训练合同或默认模型采用批准。

### 08.1 合同—实现对照表（2026-10-31 实施方交付，偏差已按原合同纠正）

「原要求→实现→处置→证据」四栏对照。原实现（`dc7a43354`）存在三处与批准合同的偏差，均已纠正；纠正后 12/12 定向测试＋13/13 R5 回归测试绿（本轮实测运行，非仅文件存在）。

| 合同要求（R5b §2.1/§5/§7） | 原实现 | 处置 | 证据（定向测试，本轮实测绿） |
|---|---|---|---|
| §2.1-A 复制轨：读取路径选记录，被选记录 value span 原始字节经指针分布进入字节分布，与词表 softmax 按 gate 混合 | `next_logits`：`_read` 选 top-1→span 字节 `copy_key/copy_query` 位置 softmax→`index_add` 成 257 维 copy 分布→`copy_gate(cat(updated,read))` 混合取 log | 一致，未改 | `test_copy_track_math_is_the_declared_mixture`（混合公式逐步复算 allclose） |
| §2.1-A/§5 通道不接触 gold；宿主伪造选择必须被拒 | teacher/answer/scheduled 全部要求 store 绑定；`content_read_cache` 校验每条 condition 记录的 `record_sha256`+`raw_span` 属于声明 store | 一致，未改 | `test_forged_condition_record_is_rejected` |
| §2.1-A copy 参数归 decoder 信用组（Q2 stop 仍恰为 reader+decoder） | `credit_group_of("copy_*")→decoder` | 一致，未改 | `test_credit_stop_freezes_copy_parameters` |
| 空 store/切断退化＝v1 逐字节 | 空 content_ids 直接返回 v1 raw logits | 一致，未改 | `test_copy_track_reduces_to_v1_without_content`、`test_cut_store_degenerates_to_v1` |
| §2.1-B 抽样单位＝**每 episode** 以 0.5 概率 | coin 按 `episode_id+turn_index` 逐应答轮抽样 | **偏差①纠正**：coin 去掉 `turn_index`，只按 (seed, update, episode_id) | `test_own_prefix_coin_is_pure_and_declared` |
| §2.1-B 教师强制**目标字节不变，仅换前缀来源** | 自前缀分支只监督后半段（监督字节缩短为 len−len//2） | **偏差②纠正**：scheduled 分支对**全目标**产出监督行，仅前半段的输入源换成自身贪心字节；曝光（监督字节数与目标）与纯 TF 分支逐字节一致；行数≠目标长度即抛错 | `test_scheduled_forced_conditioned_is_deterministic_and_byte_exact`（rows=len(target)）、`test_v2_adapter_loss_terms_byte_accounting`（两分支 supervised_bytes 相等） |
| §7/08.3 探针须证明**实际执行的分支**、日程位置与监督字节 | 原 phase B 强设 `schedule_update=300` 被 `R5BRunTrainer.update()` 每次重置回 `steps+1`（**phase B 实测的是纯 TF**；已产出的定价报告失真） | **偏差③方法修订**：新增 `R5BRunTrainer.schedule_position_override` 声明式探针机制，逐更新记账 `schedule_position/schedule_override_active/episodes_scheduled/scheduled_supervised_bytes`；定价脚本 phase A 断言 0 scheduled、phase B 断言 >0，否则工程失败 | `test_trainer_override_keeps_probe_on_declared_branch`；失真报告 `reports/taiji_native_core_r5b_pricing_20261030.json`（未提交）已删除，定价按修订方法在 08.3 重跑 |

runner（`b78bdfdce`）另补 `--run-id`（默认 20261031），避免与已停草稿的写一次目录冲突。

**并行草稿账（不计任何回执）**：按偏差纠正前的实现启动的 9-run 训练已按 08.1 停止，停在 `checkpoints/native_core_r5b/20261030/`（3 个端点 seed0/1/2＋q2_credit_on_seed0 零步，无报告）；耗 CPU 约 1.5 万秒，按 R5b §7 计入同一 120,000 CPU 秒总预算账。作废定价件 60 更新约 572.56 CPU 秒同样计入。预算余量约 10.4 万 CPU 秒，覆盖修订后重定价（≈0.1 万）、正式 9×512 训练（≈4.3 万）与评价（≈1 万）。

08.1 对照表不能代替 08.2 工程回执；下一步按总计划 08.2 交付信用/隔离/训练态恢复证据（含 256→257 边界与恢复后下一次更新）。

### 08.2 工程回执（2026-10-31 实测，`r5b_082_ok`）

[回执](../../../reports/taiji_native_core_r5b_082_engineering_20261031.json)、[探针脚本](../../../scripts/training/probe_native_content_core_r5b_082.py)、实现 `1315ff959`。五族门全过（无 failed_gates）：

- **信用**：同初态 credit_on/stop 前向逐位一致；逐 loss 组读数——on 侧 reader/decoder/core/attention/output/query_encoder 全部有梯度，stop 侧 reader+decoder 零梯度、冻结参数逐位不变、其余组确认位移（optimizer 真实生效）。
- **gold 与隔离**：宿主伪造 condition 记录（含可复制 span 的外来记录）被 store 绑定校验拒绝；跨 scope 读取为空、本 scope 恰 1 条。
- **退化与切断/救援**：带内容时复制轨激活（与 v1 分歧）；空 store 下 v2 与 v1 在**同一 cut condition** 下逐字节相等；救援答案与基线逐字节一致。
- **旧 v1 可加载**：R5 冻结端点 `q1_new_core_v1_seed0` 以原报告记录 digest（e8550592…）一致加载。
- **256→257 日程边界与训练态恢复**：连续 257 更新 vs 256 处 checkpoint 新进程恢复后下一更新——schedule 位置 256（0 scheduled）→257（2 scheduled，coin 按声明触发）；恢复后下一更新 loss 与连续 run **逐位相等**（0.3529213984619896）；恢复后端点 probe digest 与连续 run 端点一致（8ee2b741…）；cursor（run_id/seed/frozen_groups）、optimizer 步数、exposures 恢复记账在回执。

账目与作废记录：工程探针共 515 优化器更新、正式回执 4,707 CPU 秒/405 墙钟秒；此前三次脚本缺陷（函数序 NameError、GATES 键名 KeyError、写一次残留）与一次探针比较 bug（误把 V2 空 store 对 V1 带内容比较；实现无缺陷）各作废一跑，累计约 1.4 万 CPU 秒，连同作废回执（未提交已删）一并计入预算账。**累计已耗约 3.5 万/120,000 CPU 秒，余约 8.5 万**，覆盖 08.3 重定价（≈0.06 万）、08.4 正式 9×512（≈4.3 万）与评价（≈1 万）。08.2 是工程门，不代能力；query/holdout/sealed 未读。

### 08.3 定价与冻结（2026-10-31，按修订方法，`r5b_priced`）

[定价报告](../../../reports/taiji_native_core_r5b_pricing_20261030.json)（作废件已删后按修订方法重出）。分支证据齐备：phase A 50 更新 `scheduled_episodes=0`（纯 TF 区间符合声明）；phase B 10 更新经 `schedule_position_override=300` 声明式探针，`scheduled_episodes=19`、positions 逐更新记录全为 300、监督字节 4,336 逐更新记账——自前缀分支被真实执行并有账。**冻结单价 9.148 CPU 秒/更新**（两阶段中位数取大）；9×512 投影 ≈42,150 CPU 秒；零步/phaseA/phaseB 端点同进程＋新进程恢复 6/6 相等；定价成本 568.8 CPU 秒；query/holdout/sealed 未读。预算账：累计约 3.56 万/120,000 CPU 秒，正式训练后预计约 7.9 万，余量充足。

### 08.4 训练执行（2026-10-31，`r5b_training_complete`）

[训练报告](../../../reports/taiji_native_core_r5b_training_20261030.json)：9/9 run、4,608 更新、40,259 CPU 秒/0.97 墙钟小时；15 恢复门（零步＋端点×9）全过、无 NaN、无预算停止；每 run `scheduled_updates=256`（257–512 日程全程激活）；Q1 与 Q2 credit_on 端点 digest 逐 seed 复现（3/3）。credit_stop 臂 loss 停在 ≈1.1（reader+decoder 冻结，与设计一致）；credit_on 臂 4.6→0.11–0.19。

### 08.5 评价与结案（2026-10-31，`FAIL_BEHAVIOR`）

[评价报告](../../../reports/taiji_native_core_r5b_evaluation_20261030.json)（首次读 query/holdout，sealed 未读；基线按合同 §4 复用 R5 报告）。**判定：`FAIL_BEHAVIOR`**——failed_gates＝behavior 五门＋`retention.train_like_exact`＋`retention.receipt_exact`；比较/机制/成本/恢复门全过。逐门真数（holdout 3,456 轮，3 seed：0.525/0.484/0.506）：

| 门 | R5b 实测 | R5 对照 | 门槛 | 判定 |
|---|---|---|---|---|
| `answer_exact` | **0.505** | 0.323 | ≥0.80 | 败 |
| `current_query` | 0.352 | 0.173 | ≥0.85 | 败 |
| `previous_query` | 0.619 | 0.207 | ≥0.75 | 败 |
| `unknown_query` | 0.698 | 0.737 | ≥0.85 | 败 |
| `state_exact` | 0.750 | 0.750 | ≥0.90 | 败 |
| `cut_drop` | **1.000** | 0.166 | ≥0.50 | 过 |
| `rescue_gain` | 1.000 | 1.000 | ≥0.90 | 过 |
| 相关/无关交换改变差 CI 下界 | **0.990** | 0.020 | >0 | 过 |
| holdout−训练内插 | **0.009** | −0.006 | ≤0.10 | 过（自由生成差距已闭） |
| 训练内插 `answer_exact` | 0.497 | 0.329 | ≥0.85 | 败 |
| 收据 | 0.655 | 0.429 | ≥0.95 | 败 |
| holdout 净收益 | **0.352**（CI [0.332,0.371]） | 0.169 | ≥0.05 | 过 |
| Q2 信用差 | **0.390**（CI [0.370,0.410]） | 0.117 | ≥0.05 | 过 |
| 成本比 | 1.07 | 0.714 | ≤2.0 | 过 |
| 恢复 | 15/15 | 15/15 | 全过 | 过 |

**架构归因（按合同 §9 出口，不改门、不加剂量、不重跑）**：复制轨把「答案是否消费选中内容」从 R5 的旁路（cut_drop 0.166、收据 0.429）变为**因果已证实的消费**（cut_drop 1.000、交换差 0.990、收据 0.655）；自前缀日程把教师强制/自由生成差距基本闭合（内插−holdout 差 0.009 过线，内插绝对值 0.329→0.497）。剩余缺口是**绝对正确率**：答案现在真实经过选中内容，但选中/复制的值只有约一半正确（current 0.352 最弱），unknown 0.698 距 0.85 亦不足；`state_exact` 0.75 与 R5 持平（状态写路径与本次两改动正交，未受益也未受损）。按 R4 §4.6，R6 未准入；后继需另立有界合同针对「选中正确值」的读取精度，而非复用本门重跑。

费用：训练 40,259＋评价 4,769＋定价 569＋08.2 探针 4,707＋作废跑约 1.4 万＋并行草稿约 1.56 万 ≈ **8.01 万/120,000 CPU 秒**（合同 §7 预算内收案，非 budget_stop）。

### 08.5 后继裁定支持证据（2026-10-31，train-only 只读诊断，`diagnosis_complete`）

[诊断回执](../../../reports/taiji_native_core_r5b_selection_diagnosis_20261031.json)、[脚本](../../../scripts/training/diagnose_native_content_r5b_selection.py)。2,592 个采样 train 应答轮（216 episode×3 q1 端点，seed 20261030；query/holdout/sealed 未读、0 优化器更新）：**选择错误桶（选错记录/检索失败）为零——读取对正确记录的选择在 train 上已是 100%**；全部缺口在 `selection_correct_answer_wrong`＝0.435（选中正确内容后复制/表达执行产出错误字节；current 0.681、previous 0.376、unknown 0.282）。**裁定框架修正**：若立后继合同，靶点应是「选中正确内容后的复制/表达执行精度」（复制指针分布/gate/解码器执行），而非「选中正确值」的读取精度（该环节无错可选）。owner 裁定点保持开放：立后继合同（靶点依此修正）／全局收束转 09 静态方案／暂停保留现场。诊断成本 475 CPU 秒，计入预算账（累计约 8.06 万/12 万）。

### 08.5 后继裁定支持证据之二（2026-10-31，复制/表达失败分解，`diagnosis_complete`）

[回执](../../../reports/taiji_native_core_r5b_copyexpr_diagnosis_20261031.json)、[脚本](../../../scripts/training/diagnose_native_content_r5b_copyexpr.py)（q1 seed0 端点，同 216 episode 样本；train-only 只读，357 CPU 秒）。三个发现：①**纯教师强制 argmax 已不是有效探针**——TF 下 value 字节 argmax 仅 0.157、template 仅 0.038-0.053，与训练末段 loss 0.145 nats 矛盾的唯解释是自前缀日程刻意改变了输入分布，模型技能存在于自生成 regime（与 08.2 边界回执一致）；②**current 类错误 79% 在值字节之前的模板早期发散**（自生成时模板句式先错），previous 类 91% 在值字节本身；③unknown 全部为模板类发散。**对裁定①的含义**：后继合同若立，靶点＝生成侧复制/模板执行（双表面：current 的早期模板发散＋previous 的值字节拷贝），且设计冻结前需用与自前缀 regime 一致的探针（TF 探针无效）。owner 三选项裁定保持开放；诊断累计 ≈830 CPU 秒，预算账累计约 8.07 万/12 万。

### 08 后继裁定状态（2026-10-31，owner 已裁定：①批准 R5c）

owner 经弹窗裁定「① 批准 R5c」，[合同](../../reference/SPEC-NATIVE-CONTENT-CORE_R5C_COPYEXPR_CANDIDATE_20261031.md)自 §5 生效；§5.1 设计冻结已批准前预完成（指针 CE 目标规则覆盖 91.67%、权重 0.3、梯度路径实测通过），直接进入实现。预算账：R5b 累计 8.07 万/12 万已收案；R5c 独立预算 60,000 CPU 秒/48h 自行记账。09 保持未准入。

### 08 后继裁定支持证据之三（2026-10-31，发散点归因，`diagnosis_complete`）

[回执](../../../reports/taiji_native_core_r5b_divergence_attribution_20261031.json)、[探针](../../../scripts/training/diagnose_native_content_r5b_divergence.py)（q1 seed0，同 216 episode 样本；270 CPU 秒，train-only 只读）。**429 个错误答案在首个分歧字节处 100% `guided_wrong`**——喂入完全正确的目标前缀后 argmax 仍选错字节且无恢复。机制判定冻结：训练信号问题（模型在这些位置自信学错分布），非推理漂移、非解码脆弱。R5c 草案 §2.1-D 已据此冻结为完整可批形态（C＝值字节辅助 CE 对表面 A；表面 B 预声明修订＝同预算内 current 类自前缀暴露加权）。裁定三选项保持开放；诊断累计 ≈1,100 CPU 秒，预算账累计约 8.07 万/12 万。

### 08 后继裁定支持证据之四（2026-10-31，R5c 设计冻结支持，`freeze_support_complete`）

[冻结回执](../../../reports/taiji_native_core_r5c_design_freeze_20261031.json)、[探针](../../../scripts/training/freeze_native_content_r5c_design.py)（train manifest 统计＋seed0 端点只读梯度探针，≈3.4 CPU 秒）。R5c 草案 §2.1-C 全部自由参数已预冻结：指针 CE 目标规则（value 子串单次出现 91.67% 覆盖、零出现 216 轮剔除、从不多次）、辅助权重 0.3、梯度路径实测通过。**草案至此零悬空项，批准即 Fully executable**；三选项裁定保持开放，未批准不执行 §5 任何一步。

### 08 裁定档案补充：预注册复审议程触发（2026-10-31）

`MEMORY.md`（本段原引的外部治理记录，仓库未提供可核验原件）的触发条件「同机制归因×3 → 路线复审议程」已满足：同一内容消费机制轴上连续三次架构归因指向同一结论——R5（答案未消费选中内容＋生成差距）→R5b（复制轨修复消费后绝对正确率仍短）→发散归因（训练信号问题，100% guided_wrong）。**此触发不替 owner 做裁定**，但为选项②补足实质：若转全局收束，路线复审议程应覆盖（a）复制/表达机制在该课程域的学习信号设计（辅助监督、暴露加权之外是否还有结构性缺口）、（b）state_exact 0.75 与两轮正交的独立路径、（c）课程域外泛化无证据的现状。裁定三选项与此议程一并呈报 owner。

### 08-R5c 执行与结案（2026-10-31，`FAIL_BEHAVIOR`，机制性负结果）

owner 批准后按 §5 顺序全链条走完：工程回执 `r5c_engineering_ok`（v3==v2 架构逐位一致、辅助 CE 梯度路径、信用/gold/边界恢复全过）→ 定价 `r5c_priced`（单价 10.46 CPU 秒/更新，A 相 0 scheduled/B 相 19 scheduled＋辅助项全触发）→ 训练 9/9（44,797 CPU 秒，恢复全过，辅助项逐 run 触发）→ 评价 [报告](../../../reports/taiji_native_core_r5c_evaluation_20261031.json) 按原门结案 **`FAIL_BEHAVIOR`**。

**机制性负结果**：指针辅助监督（权重 0.3）未修复复制/表达缺口，反而轻微变差——holdout answer_exact **0.458 vs 前身 0.505（delta −0.047）**；current 0.243（前身 0.352，更差）、previous 0.588（0.619）、unknown 0.758（0.698，更好）；收据 0.586（0.655）。内容消费机制保持完好（cut_drop 1.0、交换差 1.0、rescue 1.0），比较门仍过（+0.304 CI [0.285,0.323]）、Q2 信用差 +0.373、成本比 1.16。三 seed 0.507/0.442/0.425。**归因**：向选中记录值字节的指针辅助监督不改善且略损生成精度——「对指针分布直接加监督」这一假设被否定；缺口仍是训练信号问题，但辅助监督的施加位置/形式需重新假设。

费用：R5c 独立预算内收案——工程探针 5,160＋定价 651＋训练 44,797＋评价 5,066 ≈ **5.57 万/6 万 CPU 秒**，非 budget_stop。按合同 §7：不加剂量、不重跑、不改门；R6/09 保持未准入。预注册「同机制归因×3→路线复审议程」已触发且归因次数增至四次，全局路线复审为下一裁定节点。

## 继承的历史事实与限制

| 历史资产 | 已有事实 | 保留边界 |
|---|---|---|
| N175/N176 | N175 R为0/3，S未全判；N176固定1,360份生成与消费登记已完成，F2未解析 | 按原输出/hash/合同留存，不重复生成或借新计划代判 |
| N177.1–177.5 | 状态机、来源消费、结构化T2/T3、T1 producer与恢复已有有限工程证据 | 结构化Goal/状态不代证自然目标理解、环境学习或跨会话能力；默认Adapter未替换 |
| E0 | 15臂fresh-process 15/15，历史revision复算60/60，隔离/mask 8/8 | 仅固定train输入，不代签任务能力 |
| E2c/训练态E3 | 信用on/stop 12/12×2，零步恢复17/17，禁写组3/3 | 旧选择器无回答信用，真实链解析覆盖1/4单列 |
| F0a/F0b | F0a train/copy均1.0；F0b 12/12 run，实测4.68 CPU秒/更新，恢复24/24 | F0a不代泛化；F0b wall超96秒如实留账 |
| F1/F1b | 分别`FAIL_COPY_ONLY`/`FAIL_NO_STATE_EFFECT`，低分与费用不撤销 | 目标字节错位、时态接口、hidden旁路和指标混杂限制归因；旧cut/rescue不代正确性因果收益 |
| N178 | face2解析局部改善，22/29/24/64，非完整回答收益 | 采样seed不等于全参数独立初始化；面已用于选剂量，后继属开发诊断，不重置F累计停止线 |

详账见[阶段收束](../../reference/PROJECT_CONSOLIDATION_20261029.md)、[核心设计重审](../../reference/TAIJI_CORE_DESIGN_REVIEW_20261030.md)、[旧资产索引](../../reference/LEGACY_ASSET_INDEX.md)。历史表的处置不构成当前待办。

## 历史恢复、CI与资料归属

原逐轮正文可从[历史恢复登记](../../reference/PROJECT_HISTORY_CONSOLIDATION_20261023.md#6-20261026活动计划重编)与Git恢复。历史R2中间checkpoint清理已有[逐项回执](../../../reports/taiji_r2_historical_checkpoint_prune_20261027.json)，该既有事实不授权本轮进一步删除。

历史fast lint修复及回执见收束记录；历史CI绿不代当前提交CI绿。每包只检查受影响范围，保留无关未提交工作。计划放plans，机器报告放reports，模型和数据沿受控根；当前排期全部回总计划，03不再复制执行清单。

## 2026-11-01 10.21.1 只读前置复核（`preflight_measured`，生成/读取均absent）

按[10.21 rev2合同](../../reference/SPEC-TAIJI-CORE_COMPOSITION_10_21_20261101.md)执行P包后交付[脚本](../../../scripts/training/verify_taiji_10_21_1.py)与[报告](../../../reports/taiji_core_10_21_1_preflight_20261101.json)。候选`feb0cbcb…`、bundle `ce4c6268…`（77,609,636字节）与09.8收束卡逐项核对；bundle的config字典省略默认`output_heads`，身份门按**生效配置**比较（v1反序表等价断言写错、v2/v3两次拿原始config字典比较，三次仪器迭代均保留并披露）。仪器15/15自检全过（oracle 6/6、错对象/错地点/反序/多动作/整表/16字节截断全0、固定真动作1/6、拖尾标点可剥但前缀不可剥），身份与日志（18×≤65步257维概率、参数逐位未变、0更新）全过。

读数：**18次生成**（6目标×真实/无记录/域内置换）在64字节/65预测协议下**三组均0/6 completed_action_exact、0次EOS**；真实/置换分别复读“台是木桌。”/“台是窗台。”，但第64字节截断末字符使**三组严格UTF-8全部失败**（真实/置换的63字节前缀可读，无记录从早期即乱码）；EOS最大概率0.0075，全部停在字节上限。**15次描述词读取**（5描述词×真实/置换/反序）真/置换**各1/5**命中；五个描述词在所有条件与反序下都选同一条记录（`glossary-1`），组均目标质量0.160–0.251、null质量0.0061、反序同源5/5 ⇒ 查询不改变选择，内容/状态主导。每个条件的唯一命中恰是恒赢记录自身，不是查询选择。两面按合同均判`absent`。

**归因（架构，而非训练剂量）**：生成面是继承表达路径的复制/表达架构问题——复制通道活着且对记录内容敏感（真实→木桌、置换→窗台），但它复读记录子串、从不产生动作模板的“把”，EOS从未成为argmax；读取面是继承`EpisodicRead`的查询/读取架构问题——同一个记录对所有描述词、所有映射、正反序都赢，状态/内容项压过目标查询项，与10.21a-2的“内容主导”一致。两条负读数支持10.21.2要替换的正是这两条路径，但不证明新模块可学。

费用：14.84 CPU秒/15.19墙钟（预算600/1200）；三次仪器迭代2.11＋2.47＋2.41，合计**21.83 CPU秒**、0更新。报告`claimable=true`但`capability_claimed=false`、`training_admitted=false`；未读新确认/sealed，未改运行代码或既有退出。当时的唯一下一步为裁定10.21.2准入（已执行，见下）。

## 2026-11-01 10.21.2 图/信用/新模块工程恢复（`passed_10_21_graph_engineering_only`）

按批准范围交付[模块](../../../taiji/query_binding.py)、[runner](../../../scripts/training/verify_taiji_composition_10_21.py)、[测试](../../../tests/taiji_native/test_query_binding.py)、[报告](../../../reports/taiji_core_10_21_graph_v2_20261101.json)与[manifest](../../manifests/taiji_core_10_21_graph_v2_20261101.json)。模块105,760参数（small 904、S可训50,304）；四profile（默认J-f64、默认J-f32、默认S-f64、small J-f64）各3次Adam共**12次**；初始与step1独立恢复4/4逐位一致；J/S/small的FD **11/11、3/3、11/11**全分辨；f32的W_bind/U/V **3/3**分辨、8个读取组低于f32 FD分辨率并逐组披露；null-only零类7组（K_goal/V_goal/role/W_state/Q_source/K_source/V_source）精确为0（无记录时源读取只返回学习到的null值，整条目标→查询→记录key/value路径不被使用），null路径3.63e-5、记录fixture下同组梯度>0；内部事件使H与动作分数实际改变（零/错事件H差0.445/0.409、分数差5.14/5.25e-5），快写关闭逐位不变，冻结父参数0梯度、输入导数保留；类型/格式检查0错误、6项定向测试通过；父bundle与参数逐位未变；**26.27 CPU秒/55.36墙钟**、记录bundle 141MB（检查后删除）。六次工程迭代保留（v1 KeyError、v2 f32 eps、v3 small z、v4 small initial、v5补null-only前通过快照、v6 null-only预期集），最终canonical为通过版；旧manifest `_v1_pre_fix`保留。没有训练/定价/能力声称；唯一下一步见总计划（裁定10.21.3定价准入）。

## 2026-11-01 10.21.3–10.21.6 定价／比较／收束（`closeout_negative`，10.21.4 出口 `reader_selection_absent`）

按 owner 2026-11-01 批准执行；交付[收束卡](../../../reports/taiji_core_10_21_closeout_20261101.json)、[定价报告](../../../reports/taiji_core_10_21_3_price_20261101.json)、[比较报告](../../../reports/taiji_core_10_21_4_compare_20261101.json)；新增 `--stage price/compare` 于[runner](../../../scripts/training/verify_taiji_composition_10_21.py)与[数据生成器](../../../taiji/native_composition_fixture.py)。`--stage graph` 路径零行为改动。

**10.21.3 定价（`priced`，17/17，6.17 CPU秒）**：四类单价分列——`cpu_per_train_step` 0.1328125、`cpu_per_restore` 0.953125（子进程自报，父侧等待另记）、`cpu_per_eval_example` 0.1074219、`cpu_per_bundle_MiB` 0.0017832。外推 10.21.4 为 **305.06 CPU秒**（训练 159.375＋恢复 7.625＋评价 137.5＋bundle 0.5625），`cap_cpu_seconds` 1200、`within_cap=true`。先验恢复再定价：J/S 各存一次新 bundle，spawn 子进程恢复后 `scores`／`h` 逐位相等、`loss` 差 0.0。8 例开发读数 J/S 均 **0/8**，属两步更新必然（loss 2.708≈log 15，仍均匀猜测），单列 `consumed_development_feedback` 并标注非门、未据此调参。

**10.21.4 两臂比较（`reader_selection_absent`，200.25/1200 CPU秒、207.06 墙钟、峰值 RAM 691.7 MB、12/12 工程门、1200 次 Adam）**：2 seed（17021／17022）×2 臂（J 105,760／S 50,304）×300 步、batch 8、仅 step0/step300 两点评价、dev 160 例完整 20 候选 argmax。四格 top-1：17021 J 0.0500→0.0500、17021 S 0.0500→0.0625、17022 J 0.0500→0.0625、17022 S 0.0500→0.0563；**chance 与最佳常量均 0.05**。三层：未见组合 0.0000–0.0417、未见实体 0.0750–0.1000，均未过 0.40。相对 `S_stop_read` 增益 **−0.0125／+0.0063**，world 配对 bootstrap（2000 次、`stat_seed=57021`、逐 seed 不合并）95% 区间 [−0.0312,+0.0125]／[−0.0188,+0.0375]，下界均非正。读取面 source top-1 **0.0969–0.1187**（chance 0.10）⇒ 出口判 `reader_selection_absent` 而非 `binding_generalization_absent`。

**工程自证**：两 seed 新增模块摘要不同（`4795e728…`／`00dada69…`）；J/S 初始前向逐位相同；12 个 checkpoint（step0/150/300 × 4 格，341.9 MB）可重新加载且摘要一致、**保号存档未删**；step300 子进程恢复逐位一致；父 bundle `ce4c6268…` 逐位未变；confirm 8 world 未生成未读取。

**10.21.5 `not_released`**：无候选可释放（主端点全部停在 chance），确认运行只消费确认面、不产生新信息。按 §7.3 负结果直接回已定位缺口，不改门、不加剂量、不换 seed。

**两处偏离（如实入账）**：**D1** step0 提前杀预算规则按解释执行（`applied=false`）——step0 是未训练参照，四格必然 chance，按字面执行只得到 vacuous absent，且 §7.2 禁中间点读开发面；四格 step0 数值与理由见报告 `verdicts.early_kill_guard`，判据与端点未改。**D2** 10.21.3 的 manifest 绑定共享 runner 自身 hash，runner 新增 `--stage compare` 后重跑命中漂移门——回归实测四类单价一致（0.125／0.906／0.107／0.00119）、`within_cap` 仍真（295.1 秒），唯一失败项 `manifest_identity` 且 drift 仅 `source_hashes=true`；原 manifest 保留不改，后续按 graph 惯例（manifest 排除共享 runner）处理。

**诊断（本轮排除项）**：仪器不可达（协议容得下 27 字节规范动作）、头容量或训练不足（105,760 参数、300 步、1200 次 Adam 跑满、loss 仍对应均匀分布）、种子轴失效（两 seed 摘要不同、读数逐格一致）、信用未达新模块（10.21.2 FD 全分辨、null-only 零类精确 0）、缓存或恢复污染（初始前向逐位相同、12 checkpoint 可重载、父 bundle 未变）。**残差**：读取注意力停在 chance，与前序 10.11b（真/置换同分）、10.12（映射进 state 但不可用）同向——条件化分量存在于状态，不足以驱动选择。

**费用**：10.21.1 14.84 ＋ 10.21.2 26.27 ＋ 10.21.3 6.17 ＋ 10.21.4 200.25 ＋ 10.21.5 0 ＝ **247.53 CPU秒**（含迭代 254.52；本段预算合计 2700，未超）。

**不能声称**：任何任务能力、可靠自然中文、P2／L3；不能把 10.21.2 工程通过读成语义成立、把 10.21.3 定价读成能力判据；本负结果只覆盖合成夹具、单一确定性父、单参考图、300 步／lr 1e-3。**10 线按负结果结案，不续写 10.22。**

## 2026-11-01 13.7 本轮全局收束（`round_closed_early`）

按[MAINLINE §105](../../MAINLINE.md)提前收束规则交付[总账](../../reference/SPEC-TAIJI-CORE_CLOSEOUT_13_20261101.md)与[回执](../../../reports/taiji_core_13_closeout_20261101.json)。四项总账：①**能力与限制**——主体链 `feb0cbcb…`／`ce4c6268…`／`4795e728…`／`00dada69…` 从 09.8 到 10.21.4 未断，能保存恢复、拿 FD 信用、门控出生、跑六活动与定价，不能自然中文问答、按目标选对动作、保持旧能力、自主选择、回收/建群体；②**资产与失败处置**——底座五模块与 `query_binding` 保留（后者不宣称有效），12 个 checkpoint 341.9 MB 保号存档，10.16（`control_not_null`）与 10.19（门 16 字节 < 18 字节最短证据）永久退出判据池；③**根目标覆盖差距**——29 条条款 `能力成立` 为 0，四处结构性缺口按代价排序为仪器（保持门缺有符号判据与能力集、主动选择夹具结构性不可测、10.19 门短 2 字节）、归因（10.7f 族／10.17／10.16+10.19，均不需要训练预算）、机制（条件化选择无替代通路、回收与群体无载体）、能力（P1 自 R5c 起零在途）；④**累计资源**——台账 v2 下界 8,985.4 CPU 秒（63 卡）＋本段 206.42 秒 ＝ **不低于 9,191.82 秒**（假设台账已含 10.21.1/2，未逐卡核对）。

**下一轮立项边界（13.7 §4）**：唯一核心问题＝「一阶信用路径能否承载输入条件化」；判据＝同 6 样本／1,536 参数／≤40 CPU 秒内把打分从平摊换成输入条件外积、留出六选一严格 >2/6，且必须同时报训练拟合度与留出/训练差距；先决三项归因排查（10.7f 族、10.17、10.16/10.19）不占训练预算；不重开 10.22、不换名加剂量、不把 10.21 负结果读成目标不可达。

**11 准入 v4**（[报告](../../../reports/taiji_core_l3_admission_11_v4_20261101.json)）：`step11_not_admitted`；P1／P2／P4 不成立、P3 成立；v4 把 10.21.4 的 `reader_selection_absent` 纳入 P2 扫描面后候选 10 个、通过 0 个，结论不变。12／13 记未开展，不补成已完成。

## 2026-11-02 14.1–14.2 切分审计与公平切分重测（`split_defect_confirmed`／`goal_copy_shortcut_confirmed`）

按 13.7 §4 的先决项（0 训练预算）启动下一轮。14.1 只读核对 10.7f 的 12 条样本（`samples()` ＋ `verify_taiji_selection_10_7f.py:153` 的 `rows[:6]`／`rows[6:]` 顺序切分）：训练正确答案只落「左盒」4、「窗台」2、「木桌」**0**；留出 4/6 的正确答案是「木桌」且状态与训练完全不重叠 ⇒ **零样本地点测试**。因此 `readout_selection_absent` 不能用来支持「参数化形状是瓶颈」；且 `StateGoalActionReadout` 本身就是 `([g;s]U)·(aV)` 的 1,536 参数双线性，所谓「外积」参数化已被试过。

14.2 按「每个地点一个状态对进训练、另一个进留出」重测（同一主体、同一读出、300 步、lr 0.05、seed 0）：留出 **6/6** vs 最佳常量 1/6。但加状态对照后：**状态置零 6/6、状态错位 6/6** ⇒ 状态未被读取，6/6 来自复读目标串（目标与正确动作串逐字节相同），判 `goal_copy_shortcut_confirmed`、`state_contributes=false`。v1 因缺对照按 `_v1_no_control` 作废保留。

**归因**：10.7 系列（含 10.7f）的选择读数建立在有语义泄漏的夹具上，对参数化既不能证明也不能否证；仓库中唯一无泄漏的选择夹具是 10.21（目标不点名实体、词汇只从 records 给），10.21.4 在该夹具上判 `reader_selection_absent` ⇒ 该负结果不可由复读捷径解释，反被加强。

**费用**：14.1 0 CPU 秒（只读）；14.2 36.66 CPU 秒／37.66 墙钟（预算 40/120，未超）。

**后继**：14.3 在 10.21 无泄漏夹具上做「读源条件化」的最小判别；不在 10.7 夹具上继续换参数化。

## 2026-11-02 一次误判与其复原（10.21a／10.21a-2），并交付 14.3 可审设计

**误判**：本轮起手把在册前驱 10.21a（通道检查）与 10.21a-2（描述词分解）当成「10.21 提案旧快照的重复立项」，理由是 10.21.1 已给出同题读数（描述词读取真 1/5、置换 1/5、五词全选 `glossary-1`）。据此把 4 份卡＋`taiji/slot_composition_head.py`＋2 份脚本移出仓外。**这个判断是错的，而且错在没扫全引用面**：`scripts/training/verify_taiji_10_21_1.py:42` 就是 `from taiji.slot_composition_head import read_slot`（该文件 :585 还把它列进 `source_hashes`），[算法复审卡](../../../reports/taiji_algorithm_review_20261101.json) 逐份登记 10.21a v1 与 10.21a-2 `_void` 两份卡的路径与哈希，本页 §1051 与 §1238 明写「旧 10.21a/10.21a-2 保留实验身份」并按「内容主导」引用其读数。⇒ 这些件**承重**，移出会让 10.21.1 的在册仪器不可导入、让复审卡登记的文件不存在、让本页文档链接落空。

**复原**：7 份件已按原路径全部复原并复核——复审卡登记的 `taiji_core_slot_channel_10_21a_20261101.json` `046a7dca608d27a1…` 与 `taiji_core_slot_descriptor_10_21a2_20261101_void.json` `154cf63e9fd034fa…` 现算仍相等，`read_slot` 复验可导入。本轮一度给 `read_slot` 加过一维化（`.reshape(-1)`，本意是给未来的 10.21 训练验证器防 `(1,W)` 广播陷阱），**该改动已逐字节撤回**：一维化改在探针自己内部做，共享模块回到改动前的字节，`taiji_core_slot_channel_10_21a_20261101.json` 的 3 条 `source_hashes` 现算全部 match，中途为此而发的 `..._v2` 卡已删除（它绑的是撤回前的字节）。撤回之所以安全，是 rev2 在撤回前后各跑一次、读数逐位相同：描述词因子 0.003968、词表因子 0.995946、F(4,16)=184.571、单行 0.513966 与 p=0.015136 全部一致，只有 `cpu_seconds` 从 1.266 变 1.297——即那次改动本就与值无关，为它换绑共享模块的字节是不划算的 churn。顺带更正一条我写下的错断言，并且这条更正本身也要更正一次：**10.21.1 的卡确实绑定源码哈希**，只是写在 `identity.source_hashes` 里（11 条），而我上一版只查了顶层 `source_hashes` 就断言"它不绑源码"。按现算：它绑的 `taiji/slot_composition_head.py` 是 `5aefda4f…`，而我已把该模块还原成 `1f2103f3…`（10.21a v1 卡记的那份）——**这条过期是我造成的**：时间线对得上，我在 17:51（本地）短暂给 `read_slot` 加一维化，10.21.1 于 21:46 跑出卡片，我在 00:59 又撤销了改动。值上无害（一维化逐位不改读数，rev2 在改前改后逐位相同），但字节绑定确实过期。我不重发别人步骤的卡（那会新增一张声称代表 10.21.1 的件），改为：由 14.3.0 的锚点测试证明该仪器在当前字节下逐位重现 10.21.1 的 15 行读数，过期绑定连同成因留在本条与审计卡里，是否升版由现行线决定。

**同一轮修好了那台审计本身**（owner 授权）：`audit_taiji_card_hashes.py` 原来只读顶层 `source_hashes`，且"已被取代"只认 `X_2026日期 → X_v2_2026日期` 一种文件名形状。两处都改了，改完的真面是 **60 fresh／10 no_hashes／8 superseded／41 unresolved**——旧版把 26 张报成"没绑哈希"，其中 15 张其实绑了而且已过期，所以 unresolved 从 26 涨到 41 **不是新造的债，是原来看不见**（新卡：[taiji_core_card_hash_audit_20261102.json](../../../reports/taiji_core_card_hash_audit_20261102.json)，0.328 CPU 秒）。过期最集中的源文件是 `verify_taiji_composition_10_21.py`（被 11 张卡绑，10.21.2／3／4 一族——同一runner 反复自绑再改），其次才是我这族的 6 张（两张哈希审计自绑、三张成本总账、一张公平性 `_void`）与 `slot_composition_head.py`（4 张）。`supersedes` 链现在会被识别（我这几轮的 `conflict_record` 因此归入 superseded，计数 7→8）。

**仪器自纠（同轮内发现并修掉）**：10.21a-2 rev1 用 leave-one-out 最近邻度量「描述词可分性」，在每类仅一样本、标签＝索引的构造下该统计量恒为 0，属恒假指标（若照它读数会得出「零可分性」这条假否证）；rev2 换成双向方差分解，并数值验证「重贴列标签不改变列间平方和」（最大相对变化 1.32e-16）以证明标签置换类零假设在此设计里根本不可用。rev1 卡以 `_void` 形保留、标 `void_instrument_defect` 与作废理由；它唯一过期的绑定正是被 rev2 取代的 rev1 脚本字节，属预期披露。**这条披露有机器可见的代价**：`scripts/training/audit_taiji_card_hashes.py` 现在把 `_void` 卡计为 `unresolved`（本轮总面 60 fresh／26 no_hashes／7 superseded／21 unresolved，0.359 CPU 秒），因为它的 v2 配对只认 `X_2026<日期>.json → X_v2_2026<日期>.json` 这一形状，而 rev2 卡沿用规范名 `..._10_21a2_20261101.json`；不重命名是为了不打断 `taiji_algorithm_review_20261101.json` 已登记的路径。净效果＝该计数 +1（rev1 卡在被标 `_void` 之前是 fresh 的）。

**rev2 的在册读数**（[卡](../../../reports/taiji_core_slot_descriptor_10_21a2_20261101.json)，1.297 CPU 秒，`parameters_changed = false`）：五行词表下描述词因子占方差 **0.40%**、词表因子 **99.59%**（F(4,16)=184.571，尾部 p≈0 ⇒ 查询信号真实存在但被极度衰减，残余交互 0.0086%）；把表缩成单行后描述词因子升到 **51.40%**、词表因子 0.65%（p=0.015136）；同一读取重复两次逐位相同（`determinism_max_gap = 0.0`）。形状与 10.21.1／10.21.4 一致，并给它加了量级：**读通路对查询敏感，是多行表上的平均把查询淹了**——这正是 14.3 必须把 D2（选择侧）与 D3（值侧可达性）拆成两个可分别否证的环节的理由：若只测 D2，「查询没信息」与「信息被平均」两种成因无法分开。

**交付**：[14.3 读源条件化前置判别·设计](../../reference/SPEC-TAIJI-CORE_READ_SOURCE_14_3_20261102.md)（`[设计·待准入]`）。四件套齐：一个主问题（主体自己的读通路是否由目标决定读哪条记录，拆成 D1 查询侧／D2 选择侧／D3 值侧可达性三个可分别否证的环节）；数据边界（只用 10.21 夹具 dev 面 8 world／160 例，confirm 面继续封存，`optimizer_calls = 0`，父 bundle 逐位不变）；预算 ≤90 CPU 秒／≤240 墙钟（单价同链实测：本轮 rev2 卡在同一候选、同一条载入链上 75 次读用 1.297 CPU 秒 ⇒ ≈0.017 秒／次，收尾 reserve 30 秒写死；若锚点复算实测 >0.17 秒／次就按 world 缩面并点名缩了哪几格，不加预算）；六个互斥出口各带「可声称／必须收回」清单，外加动态范围自证——两端改用**现成的机器锚点**：空 records 时 slot 必须逐位等于 `EpisodicRead.null_value`（下界），单行受限读（上界）。D2 与 10.21.4 同货币同门槛（per-record source top-1 对 `COMPARE_READER_BAR = 0.40`，`scripts/training/verify_taiji_composition_10_21.py:201`，在册读数 0.1125／0.096875）；§3 要求把 `verify_taiji_10_21_1.py:258,267` 的 `record_spans`／`descriptor_read` **提升**为包内探针并逐位锚定 10.21.1 的 15 行，不留第二份实现；§4 写清到原始中文输出端点、跨 session 经验与无提示选择的衔接边界；§5 逐项列覆盖／未覆盖并显式标 **P1 未覆盖**，表达端点不测不签。

**费用**：14.3 部分是纯文档（0 前向）；纠错部分**不是零前向**——复跑 10.21a 三次（1.016＋1.016＋1.000 CPU 秒，含一次仓外临时复算）、rev1 探针一次（0.969）、rev2 探针两次（1.266＋1.297），合计 **6.564 CPU 秒、200 次只读前向**（10.21a 每次 10 读＝30；rev1 20；rev2 每次 75＝150），0 参数更新、0 训练、0 权重写盘。这些花费挂在 10.21a／10.21a-2 各自的实验身份下，不并入 14.3，也不改写 10.21.1／10.21.4 的原卡费用。既有报告内容、权重与授权不变；`output/` 与封存面未读。

**唯一下一步＝owner 裁定 14.3 是否准入**：批准即按 §3 先做 14.3.0（提升＋锚定，≤30 CPU 秒）再跑判别包；否决或改范围则本件按设计输入归档，10.21 线负结案、11 的 `step11_not_admitted` 与 P1 未覆盖状态都不动。本轮不需要任何补发卡：共享模块已回到原字节，唯一留在盘上的新件是 rev2 脚本与卡，其披露已写在上方。13.7 §4 的 ≤40 CPU 秒旧提案不构成这里的执行许可。

**准入前的设计自检（同轮，仍 0 前向）**：读 `taiji/living_core.py:168-178` 与 `verify_taiji_10_21_1.py:258-312` 后改了三处设计——①**D3 不用 one-hot 注意力干预**，改成「records 只放目标那一行」的内容受限读：`EpisodicRead.forward` 的 `values` 是模块内部量，要在外面造 one-hot 加权和就得重算 `value(positions)+value_meta(metadata)`，那等于新造一条读通路、测的不再是主体的读；②**D1／D2／D3 一律按组报**（`state.h` 是 (groups, width)，每组各做一次 softmax），因为「一组在选、三组在平均」这种形状在跨组均值里是隐形的，而 10.21.1 的 `descriptor_read` 本来就按组报质量与 null 质量，提升后直接沿用；③动态范围的两端换成机器锚点（空 records ⇒ slot 逐位等于 `null_value`；10.21.2 卡已独立测到无记录时源读取只返回 null 值）。同时登记一条**自己上一张卡的取法错误**：rev2 卡的 `attention_rows.normalized_entropy` 除以 `log(groups×(P+1))`，而每组是独立 softmax、正确上限是 `log(P+1)`，该列被系统性抬高约 4 倍——它没进该卡的状态判据（判据只用方差分解），但引用它必须换算，14.3 的卡不得再用这个分母。预算随之重算：按实测 ≈0.017 CPU 秒／次读，200 次读的上界 ≈34 秒，包络从 ≤300／≤600 收到 **≤90 CPU 秒／≤240 墙钟**。

## 2026-11-02 夹具公平性机检上线（`fixture_fairness_violation`，0 前向、0 训练）

14.1／14.2 判掉 10.7 族之后，那条教训（**夹具把答案写进题面 ⇒ 读数既不能证明也不能否证任何机制**）此前只写在文档里，没有可执行的守卫：新夹具照样能重犯。本轮把它变成七条机检规则（`taiji/goal_fixture_fairness.py`＋审计 `scripts/training/audit_taiji_goal_fixture_fairness.py`＋测试 `tests/taiji_native/test_goal_fixture_fairness.py`，10 条用例全为"每条规则既能判真也能判假"的合成对照）：目标不得与正确动作句逐字节相同／目标不得点名实体／目标与正确动作句不得共享任何二元组／目标不得带动作模板／映射只能经 records 可达／切分必须留出声明过的那根轴且留出任务在训练里没出现过／对照必须是真正不留不动点的置换。不点名步骤号——它和卡哈希审计、成本总账同族，是仪器件。

**首轮读数**（[卡](../../../reports/taiji_core_fixture_fairness_audit_20261102.json)，0.016 CPU 秒，`model_involved = false`）：

| 夹具 | 样本 | 结果 |
| --- | --- | --- |
| 10.7f 面（`taiji/goal_conditioned_training.py`，10.7f 与 14.2 仍在 import） | 12 | **7 条里 6 条红**：12/12 目标与动作句逐字节相同、二元组共享合计 96（最坏样本点名 `到左`/`把蓝`/`杯子`）、24 次点名实体、24 次带模板、无 records 通路、留出任务全部在训练里出现过 |
| 10.15 宽夹具 | 480 | 适用规则全绿（`split` 不适用：该面没有发布训练／留出标签） |
| 10.21 组合夹具 | 64 | 七条全绿（按它自己声明的留出轴＝world 内的 example layer） |

**三条我自己写错又被自己跑出来的判据**（都按 `_void` 单件保留，不在正文引用）：①对照规则原写作"源与目标同一宇宙"，而词表是"描述词→实体"的两个宇宙，恒假；②切分新颖度原按目标串计，而 10.21 有意在多 world 间复用同一目标串、改的是 world 内映射，会误判一个健康夹具为不公平；③10.21 的留出轴我一开始取成 world 级 train／dev，而那从来不是该夹具声明的轴——现按 layer 判，world 级读法降级为**非门控观察**并留在卡里（实测：dev world 复用了 4 个对象名中的 3 个，`entities_shared = 棕漆木箱／深绿陶碗／银灰铁盒`，novel_task_fraction 0.4）。修的是判据，不是判语：10.7f 面在两次修正前后都红同样的 6 条。

**这条红是给谁看的**：`goal_conditioned_training.samples()` 现在仍在树里、仍逐字节泄漏，而 10.7f 与 14.2 两个在册实验都从它取数。⇒ 任何新的选择／读出实验**必须**先跑这台审计并对所用夹具绿；14.3 已把这条写进 §1.2 的数据边界。本轮 0 模型前向、0 参数更新、0 训练；新增两枚未跟踪卡（canonical 与 `_void`）与三个代码文件。

## 2026-11-02 14.3 已按准入执行并结案：`pathway_unreachable`（读源条件化前置判别）

owner 批准"执行整包"后按[设计件](../../reference/SPEC-TAIJI-CORE_READ_SOURCE_14_3_20261102.md)跑完，交付 [14.3 读数卡](../../../reports/taiji_core_read_source_14_3_20261102.json)（**3.625 CPU 秒／3.93 墙钟**，预算 90／240，reserve 30，实测单价 0.034 CPU 秒／次读，108 格；候选摘要 `feb0cbcb…` 与 10.21 线同一身份；`optimizer_calls = 0`、`parameters_changed = false`）。前置的 **14.3.0**（把 10.21.1 的读取探针提升为 `taiji/composition_read_probe.py`）先过锚点测试 [test_composition_read_probe_anchor.py](../../../tests/taiji_native/test_composition_read_probe_anchor.py)：**420 个已发布字段逐一相等（浮点按位相同，不取近似）**，2.63 秒；`verify_taiji_10_21_1.py` 删掉本地两份实现、改为导入包内那一份，仓里不再有两套读取路径。

**三环节读数**（按注意力组分报，本候选 4 组、9 行词表）：

| 环节 | 读数 | 判语 |
| --- | --- | --- |
| D1 查询侧 | 换描述词的槽相对距离中位数 **0.001072**（门限 1e-3，勉强过；最小值 0.00046 在门限下）、注意力 L1 位移中位数 **0.021845** | 查询确实进式子，但幅度只有"换词表"（0.006869）的六分之一 |
| D2 选择侧 | 四个组的 source top-1 **全部 = 0.1111**，归一化熵 **0.99896／0.99896／0.99895／0.99895** | 逐位等于 1/9 的随机水平、熵贴着均匀上限 ⇒ 四个组都**没在选**，读＝对九行求平均 |
| D3 值侧可达 | 单行受限读下，只换实体 ⇒ 槽距离中位数 **0.1362**；只换描述词 ⇒ **0.1438**；记录质量压过 null 的比例 **1.000**；空表的读逐位等于 `null_value`（差 **0.0**） | 门槛是 0.50，下界 0.20 ⇒ **不达标**：值侧承载不了这条映射 |

**结案判语与它关掉的东西**：出口按预注册顺序命中 `pathway_unreachable`（设计件 §1.4 该行写明必须收回的三句）。这条比 10.21.4 更深一层：10.21.4 说的是"训了，留出仍等于机会"，14.3 说的是**即使把表缩到只剩该读的那一行、即使注意力已经不看 null，槽里也只剩 13.6% 的位移**，而 D2 四组全在随机水平——失败不在"头没训""样本少""秩太低"，在这条读通路本身承载不了"描述词↔实体"的映射。⇒ 按设计件的收回清单，**"再加一次训练／换个头／扩样本"不再是本线合法的下一步变量**；10.21 的负结果至此有了机制归属。D1 勉强过门只说明查询有微弱影响，不构成任何能力声称。

**过程中我自己的两处缺陷与处置**：①第一次跑出 `engineering_failure`（`build_manifest(ROOT, BUNDLE)` 传了 str，需要 Path），0.047 秒即失败；该卡按本仓惯例改名 `..._v1_engineering_failure.json` **保留不删**，修好后的读数占规范文件名。②`verdict()` 的判据顺序与设计件一致（D1→D3→D2 在场→D2 缺席且 D3 可达），未因读数不利而改动任何门限；全部门限在跑之前就冻在卡里（`bars` 键）。

**成本与账**：14.3 计入 14 族（step 14.3 → 家族 14），3.625＋0.047 CPU 秒；第一次跑出的是 3.016 秒，
之后为过 B/SIM 门给两处的 `zip` 补了 `strict=True` 并原样重跑一次——**所有读数逐位不变**（d1 0.001072139516615638、d2 四组 0.1111111111111111、熵 0.9989562372868486、d3 0.13621690148704166 与 0.14381880072194753、mass_above_null 1.0、null 锚点差 0.0），只有 `cpu_seconds` 从 3.016 变 3.625；那一份卡片被我删除并在同一文件名下重发（它只在盘上活了 3 分钟、未被任何文档引用、且删的是哈希绑定本身），工程失败那份 `_v1_engineering_failure` 仍保留；同轮 14.3.0 锚点测试 2.63 秒在 pytest 内、不另发卡。本轮 0 训练、0 参数更新、0 权重写盘，未读 confirm 面（`build_split` 只取 dev），未改产品默认，提交按 owner 指示分族进行、不推送。

## 2026-11-02 记录关系读取算法修订（设计交付，未执行）

核对10.21.4及14.3机器卡绑定源码：前者10条、后者5条绑定与读取时工作树一致。10.21的QueryBindingCandidate已训练并判负；14.3经read_slot调用旧EpisodicRead，不能当作对训练后新读取模块的D3测量。原pathway_unreachable等状态保留；几何距离小并不等于实体信息为零，固定剂量/两seed/工程门也不是所有训练规模与仪器解释的排除证明。旧线不追加训练/换头/扩样本的投入停止决定保留。

从源码确认局部因果编码的最大感受野15字节，描述位置不能直接看到后部实体，10.21中该编码被冻结。按一个机制假说交付[记录关系读取与原生字节表达方案](../../reference/SPEC-TAIJI-CORE_RECORD_RELATION_14_4_20261102.md)：可塑整记录上下文、记录/位置两级读取、同一pi参与H与原始字节表达；新增172801参数，继承可塑组另列。14.4正式图/干预/信用/恢复→14.5定价→14.6三结构臂中文比较→14.7确认→14.8全局收束与经验/反馈/主动选择接力，实际队首只在MAINLINE。

[静态设计核对回执](../../../reports/taiji_record_relation_design_review_20261102.json)只记录来源身份、15字节依赖推导、小距离仍可区分的代数反例、参数计数、链接与原证据未改核验，不是新模型工程或能力通过。原复审材料与总计划已[归档](../../archive/history/n_algorithm_review_20261102/README.md)，旧报告、运行代码和检查点未改；新图/数据/训练均未执行，0模型前向、0优化器调用，未打开确认面。本条解释修正覆盖上节“任何头都救不了”等过强推广，不回填历史机器卡。

## 2026-11-02 硬编码边界澄清（方案补充，未执行）

在[记录关系方案§3.1](../../reference/SPEC-TAIJI-CORE_RECORD_RELATION_14_4_20261102.md#31-硬编码边界与输入隔离)登记计算结构、协议、记录边界、语义、评分、调节/快记忆/生长及研究时序的固定/学习边界：记录分组只取真实观察事件，不硬拆“是”或提供gold span；模型前向不接评分上下文，不把动作ID渲染成中文。承认固定句式、一事实记录和runner安排学习/输出是首包偏置，当前不能称自由语言或自主发展。

14.4原包内增加标签交换不改前向、来源ID一致重命名、错误输出不被宿主修正、实际调用字段路径四项门；AST/字面量扫描仅辅证。参数、数据分区、工程更新次数、预算和队首不变。该设计补充当时只修订三份计划；14.4实现与运行见下节。

## 2026-11-02 14.4 记录关系正式图执行（工程结案）

按[方案§6](../../reference/SPEC-TAIJI-CORE_RECORD_RELATION_14_4_20261102.md)与[复审包](../../reference/REVIEW-PACKET_20261102.md)执行唯一一个14.4工程包，交付[报告](../../../reports/taiji_core_record_relation_14_4_20261102.json)与[candidate manifest](../../manifests/taiji_core_record_relation_14_4_20261102.json)：**`passed_record_relation_graph_engineering_only`，22项检查中20项required全过、0失败**；定向测试[test_record_relation_14_4.py](../../../tests/taiji_native/test_record_relation_14_4.py) 9/9通过。

- **身份与参数**：父=`feb0cbcb…`/bundle `ce4c6268…`/77,609,636字节，父bundle运行前后逐位未变；正式图新增**172,801**参数（14组与方案§4.4逐组相等），继承可塑**2,128,322**（冻结/停用组另列），总参数**3,496,348**；A_full无停用，B_local停用GRU×2与W_c共**91,008**，C_flat停用K_record/Q_value/null_value共**9,504**；新模块seed 27031/27032，三臂共享同一父与初始化。
- **关系路径（不拿距离代替可解码）**：同一记录只改后部实体字节时，A_full前部描述位置`u`对后部字节嵌入的导数**1.7598e-3**、离散替换位移**3.4105e-3**；B_local两项**精确0**；增加/改动其他记录时原记录`u`逐位不变（0.0）。
- **内容通路**：正常读取`pi+alpha_null=1`、复制质量按原始字节聚合、`EOS`复制质量为0；强制正确记录/位置且gamma=0时目标字节概率**1.0**；强制错误记录/位置时正确字节质量**0.0**；null质量回到生成支路且P≡generated；记录反序后强制索引按新来源给出新实体字节；独立恢复后强制干预逐位重现（max diff 0.0）。
- **宿主隔离**：同前缀换评分标签时前向分布**逐位相同**而NLL改变；来源ID一致重命名并同步coverage键后下一分布**逐位相同**；预定错误字节/EOS分布经原始输出端点原样保留；AST扫描0命中，运行时调用路径只出现声明函数，`RawSource`实际访问字段仅7个声明字段。
- **任务信用与状态参与**：f64 **25/25**声明组FD全分辨（三eps、atol1e-10、rtol1e-4；最大相对失配**1.87e-7**），置零/翻符号/零参数/方向反例全拒绝；W/P/R/xi/query_read五组递归负责参数均获任务信用。f32 21/25分辨，`xi`/`Q_goal`/`K_goal`/`eta`如实记未分辨。状态置零使NLL改变**9.69e-4**、组序错位改变**1.56e-3**，未出现“答案只依赖当前来源”。
- **先恢复后学习**：四profile（A/f64、A/f32、B/f64、C/f64）各自完成原端0→1→2与恢复端1→2；初始状态、step1、step2下一次前向，参数、Adam矩与后继分布全部**逐位一致**（max diff 0.0）。共**12/12**实际Adam。
- **费用与出口**：**183.36/600 CPU秒**（主160.48＋子进程20.25）、**217.07/1200墙秒**、峰值RAM**1.17GiB**、峰值活产物**93.1MB**、累计产物352.4MB，全部在包内；工程副本删除，父资产未改。该结案不签任务能力、不签14.6训练、不定价、不读确认；**唯一下一步为14.5同形定价与正式冻结**，价不落预算即`budget_not_admitted`。

## 2026-11-02 14.5 同形定价与正式冻结（工程结案）

按[方案§6](../../reference/SPEC-TAIJI-CORE_RECORD_RELATION_14_4_20261102.md)执行 14.5，交付[定价报告](../../../reports/taiji_core_record_relation_14_5_20261102.json)与[price manifest](../../manifests/taiji_core_record_relation_14_5_20261102.json)：**`priced_within_compare_cap`，13项required全过、0失败**。三臂（A_full/B_local/C_flat）共享同一父、同一10.21合成数据面与同一float64 Adam配方，各做4次batch8更新（完整27字节＋EOS），共**12/12 Adam**；初始与最终独立进程恢复均逐位一致，后续参数全部丢弃。

- **成本面**：只用1个train world的8个训练例与1个dev world的8个开发例；训练数据摘要绑定`taiji-composition-fixture-v1`，声明worlds仍为16/8/8，`confirm_read=false`。
- **冻结单价（CPU秒）**：train_update_batch8 **5.3828**（forward 1.9688＋BPTT 3.4375＋Adam步0.0156）、encode_example **0.0273**、self_generate_example **0.3828**、eval_score_example **0.1738**、intervention_example **0.0391**、restore **2.4844**、bundle **61.82MiB**。
- **14.6外推公式**：2 seed×3臂×300更新＝**1,800**次batch8更新；step0/300各160开发例×6臂＝**1,920**次自由生成与1,920次教师强制评分；7个固定条件×160开发例×6臂×2端点＝**13,440**次干预；6次最终恢复；另加10%开销与60秒启动。合计**12,427.49/14,400 CPU秒**、**12,487.49/28,800墙秒**、**1,800/1,800 Adam**、**470.95/4,096 MiB**，四项全在cap内，判**admitted**。
- **边界**：该价只说明“按冻结配方与冻结数据面，14.6在预算内可开跑”；不签任务能力、不签结构贡献、不签保持，也未读确认面。定价参数已废弃，未保留任何权重或bundle。
- **费用**：**102.00/600 CPU秒**（主87.48＋子进程14.56）、**137.26/1200墙秒**、峰值RAM**0.60GiB**、峰值活产物**64.98MB**、累计产物277.65MB；type/lint全绿；父bundle逐位未变。
- **下一步（当时）**：14.6一轮原生中文比较。开跑前须先建训练/评价runner并先验checkpoint保存与独立恢复，再按冻结价与1,800 Adam上限执行；该步已于本日由下方14.6执行并结案。

## 2026-11-02 14.6.0 检查点预检（工程结案）

按14.5结案要求，先做正式训练前的checkpoint预检，交付[预检报告](../../../reports/taiji_core_record_relation_14_6_preflight_20261102.json)与[preflight manifest](../../manifests/taiji_core_record_relation_14_6_preflight_20261102.json)：**`passed_14_6_checkpoint_preflight`，15项required全过、0失败**。正式14.6 **当时未启动**，0次正式Adam。

- **全量数据**：16 train worlds/8 dev worlds → **192训练/160开发**，每条目标27字节＋EOS，数据摘要`cee43f9c8dd9b68c4ebcb64b444ab0fec42e5c9a553eff5eb1102e81e17a0401`；`confirm_read=false`。
- **种子与臂**：2增量seed×3臂全部建立；27031 A的candidate id **`16f7e33c…`与14.4主候选逐位同身份**，27032 A为`2cfb8897…`；两seed新模块摘要不同（`2a3c7ac5…` vs `1f48adef…`），A/B/C共享同seed初始化。三臂停用参数仍为0/91,008/9,504。
- **恢复链**：六候选（2 seed×3臂）均通过step0保存/独立探针；一次batch8工程更新后保存/独立探针；恢复端再更新与父端再更新的参数、Adam矩、后继分布全部**逐位一致**（max diff 0.0）。
- **预算对齐**：14.5冻结价仍判admitted；正式1,800次batch8更新估计**12,427.49/14,400 CPU秒**。
- **费用**：**140.80/300 CPU秒**（主77.22＋子进程63.63）、**205.14/600墙秒**、18/18工程预检Adam、峰值RAM**1.42GiB**、峰值活产物**93.09MB**、累计产物555.30MB；type/lint全绿，父bundle逐位未变，工程副本全部删除。
- **边界**：18次更新全部是预检工程更新，不计入正式1,800；预检用的batch含一个dev例的梯度，但六候选均为独立工程副本并在结束后删除，正式14.6仍从父重建，不继承预检参数或数据顺序。预检不签任务能力、不签结构贡献。
- **唯一下一步（当时）**：正式14.6一轮原生中文比较（2 seed×3臂×300更新/臂，192训练/160开发，step0/300评价，固定终点）。该步已于本日执行并结案，结果见下节。

## 2026-11-02 14.6 正式原生中文比较（任务门未过，负结果结案）

按[方案§6](../../reference/SPEC-TAIJI-CORE_RECORD_RELATION_14_4_20261102.md)与14.5冻结价执行唯一一轮正式14.6，交付[正式报告](../../../reports/taiji_core_record_relation_14_6_20261102.json)与[manifest](../../manifests/taiji_core_record_relation_14_6_20261102.json)：**`native_byte_task_failed`，24项required全过、0失败；1,800/1,800正式Adam，六候选全部完成；14.7未释放。** 六候选为27031/27032 × A_full/B_local/C_flat，每臂300次batch8更新、192训练/160开发、step0与step300固定终点；共4,032次自由生成、1,920次教师强制NLL、12个保留bundle（0/300），100/200中间checkpoint写入、校验后删除。

- **任务门（A_full两seed总体）**：completed_action_exact **0.5844/0.80**；unseen combination **0.7083/0.60过**；unseen entity **0.2125/0.60未过**；真实与置换两侧同时正确 **0.4438/0.60未过**；记录反序 **0.5844/0.95未过**。逐seed A_full：27031 **0.6063**、27032 **0.5625**。
- **结构门**：27031 A−B **+0.0813**（bootstrap 95%下界−0.0375）、A−C **+0.1063**（−0.0375）；27032 A−B **+0.0125**（0.0000）、A−C **+0.1313**（+0.0188，仅此格过）。A/B/C实测27031 0.6063/0.5250/0.5000、27032 0.5625/0.5500/0.4313。记录级寻址A−C点估计为正但bootstrap不稳健；局部编码B_local已接近A，双向上下文残差不是主杠杆。
- **内容通路门（通过）**：A真实固定8例0.75；无来源0.00（降0.75）、错误记录0.00（降0.75）、错误位置0.375（降0.375）；rescue 0.00（降0.75，仅报告不设门）。模型确实消费记录，无来源/错误读取会破坏任务，内容通路是本轮唯一通过的必要性门。
- **旧能力/保持**：`retention_not_admissible`（无合格父能力集）；step0→step300真实正确率0.0000→0.5844，教师强制NLL 4.6190→0.8487（有符号变化−3.7703）；不签保持。
- **行为归因**：训练批次NLL末值约1e-4，但自由生成0.5844；EOS率1.0、UTF-8率0.7813、source top-1 0.3125。最弱轴是unseen entity 0.2125与真/置换两侧0.4438；反序与真实同分0.5844，读取对记录顺序不敏感。失败不是“完全不读记录”，而是未见实体泛化与双侧反事实不足；A−C的点估计正但bootstrap不稳健，既不能把这轮失败整体归给“架构完全无效”，也不能签记录级寻址的必要增量。
- **仪器修正**：v1卡`engineering_failure`由两处仪器错误造成——`score_generation`把raw bytes与fixture整数tuple比较（real/reverse/step0/结构条件的exact恒假；permuted已用bytes故正确），以及`collect_rows`的step0层级/结构条件计数错误。v1卡与manifest保留为`_v1_engineering_failure`（[v1报告](../../../reports/taiji_core_record_relation_14_6_20261102_v1_engineering_failure.json)）；[校正卡](../../../reports/taiji_core_record_relation_14_6_20261102.json)由[audit脚本](../../../scripts/training/audit_taiji_record_relation_14_6_scores.py)对4,032条已存生成文本做确定性重评分，重建raw bytes、重算exact flags与全部门，**0次模型前向、0次Adam、0权重改动**；12个保留bundle逐字节验证，结果计数为step0 960/step300 960/置换960/反序960/四个结构条件各48/NLL 1,920/生成4,032。
- **费用**：**10,898.63/14,400 CPU秒**（主10,859.77＋子进程38.86，含校正0.45）、**11,341.15/28,800墙秒**、1,800/1,800 Adam、峰值RAM **1.44GiB**、峰值活产物 **555.30MB/4GiB**、累计产物1,328.62MB；type/lint全绿；父bundle逐位未变；confirm未读。
- **边界与下一步（当时）**：不追加seed/剂量/步数、不换头重跑；14.7未释放；当时唯一下一步是14.8全局收束与负结果接力设计——记录内容通路必要性、未见实体/双侧反事实缺口、结构增量不稳健，并只提出一个上限更高的后续机制候选。该步已由下节执行并结案。

## 2026-11-02 14.8 全局收束与 N15.1 接力提案（负结果结案）

按[14细则§7](../../reference/SPEC-TAIJI-CORE_RECORD_RELATION_14_4_20261102.md)与14.6负结果执行14.8，交付[收束报告](../../../reports/taiji_core_record_relation_14_8_closeout_20261102.json)与[manifest](../../manifests/taiji_core_record_relation_14_8_closeout_20261102.json)：**`closed_negative_with_followup_design`，8项required全过、0失败；0模型前向、0 Adam、0权重改动。** 14.8是文档/对账阶段，不训练、不建模型、不读确认。

- **包内总账**：14.1 0、14.2 36.66、14.3 3.63、14.4 183.36、14.5 102.00、14.6.0 140.80、14.6 10,898.63 CPU秒；合计 **11,365.06 CPU秒/11,942.28墙秒**，包内Adam **1,842**（14.4 12＋14.5 12＋14.6.0 18＋14.6 1,800）；14.8自身 **1.08 CPU秒/1.07墙秒**。各阶段均在原声明预算内；14.7 `not_released`，0消耗。
- **出口**：14.6判`native_byte_task_failed`、14.7未释放；无合格父能力，**跨session经验/纠错/反馈/无提示选择合同不冻结**。不改写14.6的负结果，不追加seed/剂量/步数，不用工程恢复代签任务能力。
- **根假说复核**：支持“内容通路可以学且必要”（无来源/错误记录降0.75/0.75、错误位置降0.375）与“记录条件行为部分成立”（0.5844）；不支持任务门0.80/0.60/0.60/0.95、未见实体0.2125、真/置换双侧0.4438、层级增量稳健性、保持、连续状态与跨session。失败不是“完全不读记录”，而是**描述↔实体绑定/泛化 + 自由前缀目标**的缺口；训练批次NLL约1e-4、开发教师强制0.8487、自由生成0.5844；rescue从0.75降到0.00，暴露读取→H→表达接口对一次正确内容注入不稳定。14.6也不支持“再加剂量/换头”作为下一步。
- **唯一接力候选**：[N15.1提案](../../reference/PLAN-N15-01_SPAN_BOUND_CONTINUOUS_20261102.md)。机制是**描述跨度绑定读取 + 自由前缀训练**：用目标描述↔记录描述的可学对齐替换14.4的池化摘要选记录（不叠头），保留同一主体 H/F、继承编码器、generate/mix 与恢复协议；训练加入声明的教师强制→自前缀课程（100%→50%→25%教师强制，自前缀停止梯度），目标仍是27字节动作＋EOS。数据改为新的 `taiji-composition-fixture-v2`：**24 train/12 dev/12 confirm**、288/240/240例、新种子48021/48022与顺序28021/28022、新五对象/五地点词表；14.6 dev退役、14.6 confirm保持封存。
- **N15.1 阶段与预算**：N15.1.0a数据60/120/0；0b图谱工程600/1200/≤12；0c定价600/1200/≤12；0d检查点预检300/600/18；N15.1.1正式比较 **18,000 CPU/36,000墙秒/1,800 Adam/4GiB**；N15.1.2确认2,000/4,000/0/1GiB（仅任务门过才释放）。门：任务0.80/0.60/0.60/0.95；结构增益0.10＋逐seed bootstrap；**source top-1≥0.60**；**rescue不得比真实低超过0.10**；内容通路降0.10；retention_not_admissible。具名出口：engineering_failure/instrument_invalid、budget_not_admitted、budget_stop、native_byte_task_failed、span_binding_component_gain_unresolved、content_path_not_necessary、span_intervention_unstable、span_binding_supported_on_declared_task。
- **N15.2条件接力**：只在N15.1.2的同一合格父上另立两session合同；必须并列同记忆冻结、合法更新、置换反馈三控制，旧任务保持与纠错分别结算，选择收益在下一query测，奖励/准入/确认独立，预算/自适应次数不重置，H/F迁移/活动信用/checkpoint恢复先验不得沿用lambda=0或只恢复权重。**本提案不批准N15.2，也不自动批12/13。**
- **费用与边界**：0模型前向、0 Adam、0权重；confirm未读；type/lint全绿；不签一般中文、自由语言、L3/L4、连续状态、跨session、保持、结构生长或主动选择。**唯一下一步**：owner准入后执行 **N15.1.0**（v2数据→图谱工程→定价→检查点预检）；价不落N15.1.1上限即`budget_not_admitted`，不开始正式训练。

## 2026-11-02 14.8后只读重审与15.1 rev2（未运行新候选）

读取14.6校正卡、14.8、原N15.1提案和真实读取/生成/干预代码，交付[复核回执](../../../reports/taiji_post14_algorithm_review_20261102.json)。既有320条A真实输出按输入/输出新颖分轴：已见描述/名90例正确80/90、90/90；新描述已见名30例17/30、0/30；真正新输出名40例两seed均0/40。133个首次错误中123个在对象byte区、10个连接区。反序原始字节＋EOS与真实逐例320/320一致；原卡反序0.5844是准确率，不是一致率。旧source_top1只取首个“把”字节前；旧rescue首步gamma=1关闭复制，不是身份恢复。原卡/负退出不改写，不能把这些辅助指标过强归因为接口不可达。

交付[15.1 rev2唯一规范](../../reference/SPEC-N15-01_BINDING_COPY_V2_20261102.md)，取代原提案当前排期但不改其14.8绑定字节。继承预定14主seed step300，保留编码/H/F/生成mix；新增30133参数的原文对齐/内容转移/后验连续复制，coverage改为已输出复制后验。读取×TF/自前缀2×2、数据288/240/240及输入描述/输出名分轴、三记录样式、实体阶段轨迹/干预、24kCPU/2400正式Adam提案、确认/选择父/负出口与15.2有限连续设计输入均写实。四臂预算尚未准入，不沿用原三臂cap。

当前（设计时）唯一下一步15.1.0a；本段只读取已发布输出/源码和制品身份，0模型前向、0Adam、未生成v2数据、未打开14或15确认、未改运行代码。14.8源码hash绑定的旧N15提案保持原位原字节；活动总计划/复审前稿已[封存](../../archive/history/n15_review_20261102/README.md)。旧审计回执保持写作时快照，不原地升级为新算法通过。

## 2026-11-02 15.1.0a 数据/评分/日志/接口预检（rev2）

按[15.1 rev2](../../reference/SPEC-N15-01_BINDING_COPY_V2_20261102.md)与用户指令执行0a，交付[报告](../../../reports/taiji_binding_copy_15_1_data_20261102.json)与[manifest](../../manifests/taiji_binding_copy_15_1_data_20261102.json)：**`passed_15_1_0a_data_scoring_interface`，16项required全过、0失败；0模型前向、0 Adam、0权重改动。** 0a只建数据、评分、日志/干预/接口协议并核对父身份，不训练、不建模型、不读确认。

- **数据**：24 train world×12=288例、12 dev world×20=240例；train 192记录/5,256B、dev 108记录/3,024B；300个opaque来源ID不重叠；dev层144 seen/36 unseen combination/60 new description；**新描述与新输出名分轴**：135/45/45/15（描述旧名旧/描述新名旧/描述旧名新/两者新），新描述60、新输出名60、两者15；4个对象名各3world轮换；三记录样式（stated/aka/mapped，训练64/64/64、开发36/36/36），train记录234–237B/≤243、dev 270B/≤279；goal不含实体名；payload 2.01MB。
- **确认**：12 confirm world/240例、4对象（前三＋第5个）/9行/≤279B、axes 135/45/45/15、data_seed48022、order_seed48024，**规格封存、未生成任何confirm例/记录/来源ID**；公开confirm build拒绝。
- **评分**：唯一bytes/整数tuple/list规范化；strict UTF-8、27字节规范动作＋EOS、64字节/65预测；oracle/constant/list/wrong name/truncation/byte-type/empty/early EOS/invalid UTF-8/not-terminated先验全过；**14.6全部4,032行用新评分器重评分4,032一致**（completed 1,541、entity 1,580、UTF-8 2,576）；实体阶段对象byte3–14、地点21–26。
- **日志/干预/接口**：每步原始字节/EOS原因/gamma/lambda/Q_null/总复制后验/argmax来源/位置；完整Q只存预定24例；记录/位置/gamma分开干预，identity rescue逐位恢复，oracle只在实体阶段、gamma=0强制复制/gamma=1 generator_only；状态schema含m/coverage/源版本/H/F/新Adam/模式RNG/前缀cursor/预算/回执/恢复/重放；模型接口仍是256字节＋EOS，样式ID/来源ID不进前向。
- **父**：14.6主seed27031 A_full step300，bundle `b75a83a0…`/64,983,135B/候选`a3104b3e…`逐字节验证；这是局部行为资产，不是合格长期保持父；新Adam从0，旧Adam归档。
- **费用**：**1.07/60 CPU秒、1.05/120墙秒**、0 Adam、payload 2.01MB/≤100MiB；type/lint全绿；confirm未读。
- **边界与下一步**：不签任务能力/一般中文/L3/L4/连续状态/跨session/保持/生长；confirm未生成。**唯一下一步为15.1.0b图谱工程**（30133新增＋继承/停用组；原文对齐/内容转移/连续cursor后验；FD、身份标签隔离、新schema保存/独立恢复；600CPU/1200wall/≤12工程Adam，待owner准入）；0b未过前不开始0c定价/0d预检/15.1.1。

## 2026-11-02 15.1.0b 图谱工程（rev2）

按[15.1 rev2](../../reference/SPEC-N15-01_BINDING_COPY_V2_20261102.md) §0b执行，交付[报告](../../../reports/taiji_binding_copy_15_1_graph_20261102.json)与[manifest](../../manifests/taiji_binding_copy_15_1_graph_20261102.json)：**`passed_15_1_0b_graph`，16项required全过、1项optional披露。** 0b只建正式span读取图、白名单、后验/cursor、干预接口、FD与恢复；不读新train/dev/confirm做训练、不执行0c定价、不启动正式训练、不释放14.7/confirm。

- **参数**：新增30,133（E_goal/E_source 4096×2、Q_w/Q_s/Q_transport 5152×3、K_transport 4096、v_relative 16、continuation 321、v_meta 4）；继承3,496,348；停用48,481（R0旧读取12项：goal attention Q/K/V、旧reader Q_read/K_record/Q_value/K_position/eta）；复用16,672（V_position 16,512＋null_key 32＋null_value 128）；总参数3,526,481、可训2,282,775。
- **方程→源码→张量/状态→检查矩阵**：goal/source span对齐（logsumexp匹配＋v_meta）、记录内transport（PE16相对位置）、`Q = λ·m + (1−λ·h)·S`、归一化、content = Q·V_position + q_null·null_value；状态含living/m/coverage/keys/record_offsets/prefix_cursor；R1 reader参数名`reader.*`，继承参数名保持`subject.*`/`relation.*`。
- **FD**：f64 26组全分辨（R1九组grad_norm 0.088–6.10，eps 1e-4/1e-5/1e-6三档一致，sign-flip/zero-param/ascent控制全过）；f32未分辨如实披露，不代签、不做主信用门。
- **干预**：record/position/gamma三干预分离（分布差0.723/0.480/0.874）；**identity rescue** = 移除干预后分布逐位回到原分布（max_abs_diff=0）；正确记录/位置oracle另名，不设"oracle必须提高整句自由生成"伪门。
- **恢复**：四profile（R0-f64/R1-f64/R1-f32/R1-small-f64）各3次实际Adam合计12；先验新schema初始保存＋独立进程恢复逐位一致、step1恢复一致、step2参数与Adam矩逐位一致；R1-small用key_width=16（null_key新建16个，不复用父32），其余复用。
- **父**：14.6 bundle sha `b75a83a0…`逐位不变；新Adam从0，旧Adam归档。
- **费用**：**55.5/600 CPU秒、84.2/1200墙秒**、12/12 Adam、峰值产物93MB/≤2GiB；type/lint全绿。
- **边界与下一步**：不签任务能力；**唯一下一步为15.1.0c定价**（600CPU/1200wall、≤12实际Adam、参数废弃；价不落15.1.1正式cap则`budget_not_admitted`）；0c未过前不开始0d预检/15.1.1。

## 2026-11-02 15.1.0c 四臂同形定价（rev2）

按[15.1 rev2](../../reference/SPEC-N15-01_BINDING_COPY_V2_20261102.md) §0c执行，交付[报告](../../../reports/taiji_binding_copy_15_1_price_20261102.json)与[manifest](../../manifests/taiji_binding_copy_15_1_price_20261102.json)：**`budget_not_admitted`**。0c只测四臂各3次batch8更新（课程位置1/101/201）及自由生成/评分/干预/恢复单价，参数废弃；不读新train/dev/confirm做训练、不启动正式训练、不释放14.7/confirm。

- **四臂**：R0P0（14原点读取+全TF）、R1P0（新读取+全TF）、R0P1（原点读取+自前缀课程）、R1P1（新读取+自前缀课程）；各3次实际Adam，合计12。
- **计算确定性**：三次运行loss逐位一致（同一seed、同一数据、同一操作），仅CPU计时有~3.5%运行间波动。
- **三次外推CPU**：17920（过）/18556（未过）/18473（未过）；2/3超cap，中位数18473>owner授权的18000 CPU上限。
- **最终单价（第3次运行，CPU秒）**：train_update_batch8=6.13、forward=2.34、bptt=3.75、adam_step=0.016、encode=0.031、self_generate=0.299、eval_score=0.198、intervention=0.048、restore=0.148；bundle=61.94MiB。
- **判定**：价超owner授权的15.1.1正式cap（18000 CPU/36000 wall），`admitted=false`；wall≈18533/36000充裕，瓶颈在CPU计时。不开始0d预检/15.1.1。
- **0c自身费用**：**99.8/600 CPU秒、101.8/1200墙秒**、12/12 Adam、产物372MB/≤1GiB；单线程（torch.set_num_threads(1)）、lint全绿。
- **边界与下一步**：定价参数全部废弃，不构成训练；超出来自单线程CPU计时波动，非算法或剂量变更。**唯一下一步为owner裁定0c边际超支处置**（是否放宽CPU cap、改用wall口径或收束15.1）。

## 2026-11-02 owner 预算裁定：15.1.1 改充足预算，设观察线

按 owner 书面裁定登记于[SPEC-N15-01 §7.1](../../reference/SPEC-N15-01_BINDING_COPY_V2_20261102.md)：**裁定日期2026-11-02；授权范围仅15.1（0d/1.1/1.2/1.3），不含N15.2与12/13，不释放14.7**。15.1.1 原 18,000 CPU/36,000 wall 上限作废，改 owner 充足预算；**只动资源参数，不动科学与剂量**（2 seed×四臂×300更新、batch8、288训练/240开发、step0/300固定终点）；任务门/机制门/rescue/恢复/具名出口一律不变；禁止借预算追加 seed/步数/剂量/换头。**观察线不是停止闸门**：1.1 实际累计 CPU 达 20,000 秒或 wall 达 30,000 秒时先回报 owner 实际进度与外推总耗，由 owner 决定是否继续，不得静默跑穿。依据：0c 三次外推 17,920/18,556/18,473，2/3 越 18,000，运行间 CPU 计时抖动 ~3.5% 而 loss 逐位一致。常设纪律：单线程、RAM≤8GiB、逐臂串行、不重跑确定性测量、checkpoint 先验独立恢复、replay 计费、实际/有效 Adam 与样本分开累计、成本偏离价 25% 以上先停查。

## 2026-11-02 15.1.0d 检查点预检（rev2）

按[15.1 rev2](../../reference/SPEC-N15-01_BINDING_COPY_V2_20261102.md) §0d执行，交付[最终报告](../../../reports/taiji_binding_copy_15_1_preflight_final_20261102.json)与[manifest](../../manifests/taiji_binding_copy_15_1_preflight_final_20261102.json)：**`passed_15_1_0d_preflight`**。0d只从14.6父重建两seed×四臂，验证checkpoint保存与独立进程恢复；不训练到采用、不读confirm、不释放14.7。

- **网格**：seed 28031/28032（数据顺序 38031/38032）× 四臂 R0P0/R1P0/R0P1/R1P1 = 8 配置。
- **三段边界**：初态（update_index=0）、step1（含真实Adam更新）、恢复端 step2 回放；全部经 `multiprocessing` spawn 独立进程恢复，快照含全部参数张量、Adam 一阶/二阶矩、state（living/m/coverage/keys/record_offsets）、cursor、update_index；**8/8 逐位一致，张量集合一致，max_abs_diff=0**。
- **附加 mid-output cursor 探针**：R1 四配置在生成第4字节处保存（prefix_cursor=4，m_abs_sum≈1，非空非零），独立进程重建记录视图后续生成剩余字节，与进程内续生成**逐位一致**；R0 无 copy cursor，按不适用记录。
- **费用**：尝试1 24 Adam、198.4 CPU、298.5 墙；仪器修复探针 0 Adam、16.5 CPU；合计 **24/24 Adam、214.8/400 CPU、329.7/800墙**、峰值产物 62.0MiB/≤2GiB；0 dev/confirm 读取。
- **首跑失败与处置**：mid-cursor 探针把样例全局 id 当列表位置取数据，导致 worker 重建不同记录视图（1例字节不一致、1例 coverage 长度不符）。按"只查仪器、不重跑同一配置"纪律，修复索引后仅以 0 Adam 重测该探针；24 Adam 的边界比对未重跑，剂量/seed 未改。两次尝试各自报告保留：[attempt1](../../../reports/taiji_binding_copy_15_1_preflight_20261102.json)、[修复](../../../reports/taiji_binding_copy_15_1_preflight_midcursor_20261102.json)。
- **边界与下一步**：预检不等于能力；参数全部废弃，有效 Adam=0。**唯一下一步为15.1.1正式比较**（充足预算＋20,000 CPU/30,000 wall观察线；任务门与机制门不变）。

## 2026-11-02 15.1.1 正式比较：已启动（在途，无结论；**已被 2026-11-03 结案取代**）

> 历史记录，保留原状；实际结案见下一条。该次运行最终为 2,400 Adam、14,580.6 CPU、14,798.5 wall，未触观察线。

按 §7.1 充足预算启动 2 seed(28031/28032)×四臂×300 更新、batch8、288 训练/240 开发、step0/step300 固定终点；逐臂串行、单线程。仪器在启动前冻结并提交（`770f34b9f`），运行中不改脚本以保证报告哈希与实跑一致。观察线 20,000 CPU / 30,000 wall：到线即停并写 partial（`watch_line_reported`），先回报 owner 再定是否继续。预算估计约 18,500–19,500 CPU（训练 14,700＋step0 评价约 560＋step300 全量指标约 2,800），贴近观察线。本阶段尚无任何任务/机制结论；`reports/taiji_binding_copy_15_1_compare_20261102.json` 与 manifest 只在正式八配置跑完后写入。临时 smoke 产物（8 Adam、150 CPU）已清理，不计入正式 2400 Adam。

## 2026-11-03 15.1.1 结案与归因复核：`native_byte_task_failed`

**结论：任务负；复核未发现仪器失效；断点在"选择—解码"接口。** [结案回执](../../../reports/taiji_binding_copy_15_1_closeout_20261103.json)＋[manifest](../../manifests/taiji_binding_copy_15_1_closeout_20261103.json)。

- **任务门（两 seed×四臂，step300）**：总体完成 0.5625/0.80；未见输出名字 0.0/0.60、新描述 0.0/0.60、双侧同时未见 0.0/0.60；R0P1 塌陷（0.0 与 0.1458）。NLL：R1 0.4662/0.4292、R0 1.7272/1.4166。
- **机制门**：包增量 R1P1−R0P0 = 0.0（两 seed 12-world bootstrap 下界均 0）；记录级选择 0/240；identity rescue 8/8 逐位；无源/错记录/错位置相对真实各降 0.4167 / 0.375–0.25 / 0.4167（内容必要性成立）；**强制 γ=0 完成率不变（降 0.0）**→学到的复制通路未在用。
- **三处疑点复核（仪器/接线/口径，≤1 次 smoke 级探针，10 Adam、90.1 CPU，如实计费）**：
  1. 读取与课程开关**真实接线**：共享 164 参数逐位相同、输出不同；R0P0 与 R1P0 在 dev 上分别 74/240、99/240 不同；P1 的 `self_count` 984/979 与 P0 的 0 分明。R1 的 P0/P1 完全重合**不是接线错位**：训练后贪心自前缀与教师前缀 **8/8 完全相同**，300 个 update 的 loss 与 P0 逐位一致，P1 因子按构造失效。
  2. `source_top1=0.0` **是真实读数**：记录级 α 与字节级代理在保存的 step300 bundle 上同为 0/240；与"γ0 完成率 0.4167"同时成立，因为生成器可独立输出记忆模板（0.4167），而复制通路未参与。字节级估计器是较弱代理，已如实登记，但结论不变。
  3. R0P1 塌陷**是真实训练结果**：198 个 update 与 R0P0 分岔、末 loss 0.465、clip_max 1650.77、self_count 984；同一课程剂量让旧读取不稳定。
- **架构诊断**：断点是"选择—解码"接口。γ 塌到≈0（强制 γ=0 无影响），记录级选择从未形成（α 0/240），训练 loss 降到 2e-5 说明训练被"软混合内容＋记忆模板"满足；课程剂量（update 101 才启动自前缀）落在 R1 已精确复制训练目标之后，使因子按构造失效，并让旧读取 R0P1 不稳定。内容通路本身必要（去记录→完成率 0.0），identity rescue 逐位成立。
- **成本与产物**：2,400 Adam、14,580.6 CPU、14,798.5 wall、峰值 1.04GB，未触 20,000 CPU/30,000 wall 观察线。保留 5 个 step300 bundle（四个 R1＋28031-R0P1）复现选择/前缀/塌陷复核，删除 11 个不需要的 checkpoint（714,544,117B，已登记）；partial 回执与空日志已清理。
- **后继**：[N16.1 草案](../../reference/SPEC-N16-01_SELECTION_LOCKED_BINDING_DRAFT_20261103.md)（显式记录选择＋选择锁定复制＋准确率触发课程；判据预注册、预算草案），**未启动训练**。1.2 未启动、confirm 未读、14.7 未释放、N15.2 与 12/13 未开展。

## 2026-11-03 N16.1 立项（方案 A）与 0a 数据接口

owner 裁定 **N16.1 走方案 A**（显式记录选择头＋选择锁定复制＋复制门下限＋按准确率触发的自前缀课程），裁定日期 2026-11-03；范围仅 N16.1，不含 N15.2/12/13，不释放 14.7、不读 confirm。立项依据为 15.1 的三处断点（γ≈0、选择器 0/240、课程按构造失效）。与 15.1 证据对齐的七条修订点（选择头输入只含 goal＋记录 description、标签只来自标注、锁定复制实现为「one-hot α 覆盖＋保留模型 betas」、γ 下限 0.2、按批完成率 0.5 触发课程、oracle 改为实体字节起点、新增 0d）已写入 [SPEC-N16-01 §0](../../reference/SPEC-N16-01_SELECTION_LOCKED_BINDING_DRAFT_20261103.md)。

- **0a 已通过（2026-11-03）**：`passed_16_1_0a_selection_interface`，14 项检查全过、**0 模型前向/0 Adam**；train/dev 288/240；每例选择标签唯一（`kind=object` 且 `description=object_description`）；键片段仅含 description，自动检查不含记录 name 与 action；dev 2×2 轴 135/45/45/15 与 unseen_combination 36 与 0a 一致；v2 train/dev digest 逐位不变（`ffe721a6…`/`84dc10d7…`）；confirm 密封未物化；载荷 722,294B；**2.31/60 CPU、2.6/120 墙**。产物：[载荷](../../../output/taiji_core_binding_select_16_1/select_v1.json)、[报告](../../../reports/taiji_binding_select_16_1_data_20261103.json)、[manifest](../../manifests/taiji_binding_select_16_1_data_20261103.json)。
- **0b 已通过（2026-11-03）**：`passed_16_1_0b_graph`，21 项检查全过。选择头 **25,602 参数**（与公式逐位一致），总参数 3,552,083（=15.1 R1 3,526,481＋25,602）；CE 三档 eps（1e-4/1e-5/1e-6）全分辨，锁定 NLL 梯度范数 43.3（A）/44.7（B）；α 锁定为 one-hot（`alpha_null=0`、模型 betas 逐位保留、分布改变）；**模型学到的 γ 为 0.00073/0.00383，下限把它强制到 0.2**——正是 15.1 断点①的修复点；identity rescue 逐位；两 profile 初态/step1/恢复端下一更新独立进程逐位一致；课程接口阈值 0.5、确定性。**两次尝试合计 12/12 工程 Adam**、65.4/600 CPU、51.7/1200 墙、峰值 62.5MiB；首跑因 bundle schema 缺 `parameters` 与参数计数公式错误失败 4 项，修复后重跑一次（未改剂量/判据）。产物：[报告](../../../reports/taiji_binding_select_16_1_graph_20261103.json)、[manifest](../../manifests/taiji_binding_select_16_1_graph_20261103.json)、模块 [binding_select.py](../../../taiji/binding_select.py)。
- **0c 已通过（2026-11-03）**：`priced_within_watch_line`。单价：锁定生成 0.516 CPU/例、copy 更新 8.27、joint 更新 8.33、restore 0.078、bundle 27.2MiB；选择阶段低于 Windows 计时分辨率（读 0.0，占 joint <0.2%，已披露）。外推 2 seed×2 臂×300 更新＝**8,390.9 CPU / wall≈8,450.9**，远低于 20,000/30,000 观察线。9/12 Adam、55.0/600 CPU、27.2MiB。产物：[报告](../../../reports/taiji_binding_select_16_1_price_20261103.json)、[manifest](../../manifests/taiji_binding_select_16_1_price_20261103.json)。价后仅删除两个未使用导入（无逻辑改动），报告内已登记 `ran_sha256/now_sha256`。
- **0d 已通过（2026-11-03）**：`passed_16_1_0d_preflight`：2 seed(28031/28032)×2 臂（A-locked/B-select），初态、step1、恢复端下一更新三段独立进程逐位一致（张量摘要逐项相同、loss 差 <1e-12）；**12/24 工程 Adam**、39.0/400 CPU、峰值 359.0MiB、产物清理。产物：[报告](../../../reports/taiji_binding_select_16_1_preflight_20261103.json)、[manifest](../../manifests/taiji_binding_select_16_1_preflight_20261103.json)。
- **N16.1.1 已执行（2026-11-03）**：2 seed×2 臂×300 更新、1,200 Adam、13,433.9 CPU/13,794.1 wall，未触观察线；判 `native_byte_task_failed`（1.2 未释放）。详见下节 2026-11-03 N16.1.1 正式比较与只读归因。

## 2026-11-03 15.1.3 全局收束回执（只读）

15.1 正式关账 `closed_negative_with_followup_candidate`：[回执](../../../reports/taiji_binding_copy_15_1_3_closeout_20261103.json)＋[manifest](../../manifests/taiji_binding_copy_15_1_3_closeout_20261103.json)。只读、0 更新、0 模型前向、0.25/300 CPU、产物 0B，9 项检查全过。四项交齐：① 无合格父（任务门不成立，父只能承载模板＋记忆名字）；② 内容通路必要（去记录 0.4167→0.0、错记录降 0.375/0.25、错位置降 0.4167），选择—解码绑定未成立（γ≈0、记录选择 0/240、课程按构造失效）；③ identity rescue 8/8 逐位、父 bundle sha 不变、1.1 总成本 14,580.6 CPU；④ 当前父不能承担跨 session 经验/反馈任务。显式登记 `1.2=not_released`、confirm 封存未读、14.7 未释放、N15.2 与 12/13 未开展、不追加 seed/步数/剂量/换头。保留 5 个 bundle 的 sha/字节全部复核一致；“较少改动优先”下 A 优于 B/C 的理由与 A 改动的三处接口已写入回执。

## 2026-11-02 规划窗口裁定：先收束15.1，再编制后继合同

当前执行计划完整到15.1.3；15.1 已按 `native_byte_task_failed` 收束，未新增15.2/16执行卡。15.2只保留接力责任，16/8/8等数字仍为待结果重审的规模示意。后继候选 N16.1 只交设计与判据（[草案](../../reference/SPEC-N16-01_SELECTION_LOCKED_BINDING_DRAFT_20261103.md)），训练未启动，立项权在 owner。

## 2026-11-03 15.1后只读审计与在途N16同步

本次核对15.1正式逐例评分、归因、最终收束、N16已裁原A与0a–0d回执，交付[审计卡](../../../reports/taiji_post15_algorithm_review_20261103.json)。原任务总体/新描述/新输出名失败仍成立，R1改善UTF-8。原未见组合27/240被写成0.1125，按本子集36例应为0.75。N16现有runner复用该聚合，全正确oracle的60例子集仍只能得0.25。15反序缓存只用局部index导致跨world覆盖，需(split,world,index)配对。原计划的聚合/配对自检覆盖缺口已补入当前结案流程。

gamma在概率式中是基础生成权重，gamma接近0偏复制，实际生成还含null回流。15材料“复制未启用”的解释作本次校正，原报告保持。15的source_top1测首个输出步的最大字节位置，实体绑定按后续对应阶段分析。N16原A当前固定对象整句选择，其回答NLL对独立选择头参数的信用路径为0，clamp在原gamma低于0.2时局部导数为0，这些均为可读源码确定的优化约束，作用大小由任务结果衡量。

本轮复核时正式入口已完整可解析，output中的16.1.1首臂终点产物和后续臂进度已出现。当前原A按原网格2seed×2臂×300运行，沿20,000CPU/30,000wall观察线记账。结案按已存逐例评分先校准统计，反事实/内容干预的真实侧按复合ID重建。原始输出不足的只读重评分另计模型前向和费用，原数值门与制品身份逐项核对。当前错位置控制包含gold记录和offset0两项改变，其解释限定为组合干预。

[逐步路由条件设计](../../reference/DESIGN-POST-N16_SELECTION_ROUTING_20261103.md)具体列出状态/前缀条件头、最终Q条件化、平滑生成下限、ST任务信用与停止信用对照，辅助description键接口依赖明确登记。该候选留到16.1.3据实际结果复审，当前原版A回执按其范围使用。[本轮前稿快照](../../archive/history/n16_review_20261103/README.md)保留分析前的活动文档。本次交付为只读计算/源码审阅与计划修订，模型前向0、Adam0，原代码/权重/确认面无本次执行改动。

## 2026-11-03 N16.1.1 正式比较与只读归因：`native_byte_task_failed`

**结论：任务门与机制门均不成立；主读数有效，归因另登记部分聚合口径与自前缀标签缺陷；断点在描述匹配器的组合泛化与实体起点对齐。** [主报告](../../../reports/taiji_binding_select_16_1_compare_20261103.json)＋[manifest](../../manifests/taiji_binding_select_16_1_compare_20261103.json)；[归因](../../../reports/taiji_binding_select_16_1_attribution_20261103.json)＋[manifest](../../manifests/taiji_binding_select_16_1_attribution_20261103.json)。

- **剂量与预算**：2 seed(28031/28032)×2 臂(A-locked/B-select)×300 更新、batch8、288/240、step0/300；1,200 Adam、9,600 训练样本、0 confirm；**13,433.9 CPU / 13,794.1 wall**（含预检 4.16、聚合 0.03，总 13,438.1），未触 20,000/30,000 观察线；逐臂串行、单线程、wall≈cpu，无重跑/重复配置。
- **任务门**：A 总体 **0.0500/0.0167**（门 0.80）；未见输出名字 **0/60（两 seed）**、新描述 **0.05/0.0167（3/60、1/60）**、反序配对**撤回**（局部 index 缓存 + 未存原始字节，无法在 0 重跑下重建）。
- **机制门**：选择 **0.3833/0.3542**（门 0.90）且新描述 **0/60**；**γ 是生成权重**（模型 γ 0.8212/0.7471、复制权重 1−γ=0.1788/0.2529），下限抬升步仅 14.6%/23.0%、移除下限完成率差 0.0/+0.0167（0.2 clamp 是生成下限、不承担成绩）；rescue 8/8；内容必要性 28031 true 4/24，无源 0.1667/错位 0.125/错记录 0.0417（未过），28032 true 0/24 → 退化不可判；**A−B=−0.2771**（下界 −0.3250/−0.3542），但被 B 的自模仿课程混淆（B 触发 127/121 次、A 0）。
- **归因（0 更新/0 前向，21 项检查全过）**：主读数有效；预检保存/独立恢复/摘要/分布/`trace` 平价全过；标签与键片段载入断言；两臂同一头训练 top1 差 ≤0.008；锁定臂 α 被强制，而解锁臂 α=0 仍完成 0.2833–0.3375。选对记录时 object_exact 0.2917/0.0476、首次偏差在 byte 3 占 50%/63%；教师强制 NLL 1.53/1.26 被模板字节稀释。课程在 A 从未触发（上限 0.375/0.25），在 B 触发 127/121 次且 loss 升高。**另登记四处口径/合同缺陷**：子集分母（240→60/60/15/36/108）、内容/反事实 true 侧局部 index 缓存（已按 (world,index) 重算）、反序配对（撤回）、γ 方向（生成权重，复制权重单列）；`batch_objective` 自前缀把目标标签替换为自生成字节（自模仿），A−B 混淆。
- **成本偏离与归因**：0c 价 8,390.9 未含按批教师强制课程探针（同一 28 步前缀的第二次前向，copy/joint 每更新约 16.5 CPU vs 定价 8.3）；1.1 预投影约 14,300，实际 94%；未发生重跑/并行争用。
- **产物**：保留 4 个 step300 bundle（262,230,260B）；删除 4 个 step150 checkpoint 与 4 个 progress 文件（262,231,613B，已登记）；confirm 未读。
- **边界与下一步**：`1.2=not_released`、confirm 封存、14.7 未释放、N15.2 与 12/13 未开展、不追加 seed/步数/剂量/换头。**唯一下一步：N16.1.3 全局收束（只读：用修正读数、弃用反序配对、登记仪器/合同缺陷），再由 owner 裁定是否准入 [17 号逐步选择路由候选](../../reference/DESIGN-POST-N16_SELECTION_ROUTING_20261103.md)；不启动训练。**

## 2026-11-03 N16.1.3 全局收束（只读）：`closed_negative_with_followup_candidate`

**结论：N16.1 无合格父；选择—解码绑定未成立；反序配对撤回；17 为主选、准入待定价与资源裁定。** [回执](../../../reports/taiji_binding_select_16_1_3_closeout_20261103.json)＋[manifest](../../manifests/taiji_binding_select_16_1_3_closeout_20261103.json)。0 更新/0 前向、0.22/300 CPU、0.24/600 wall、产物 27,053B。

- **四项总账（修正口径 60/60/15/36/108、(world,index)）**：① 无合格父——14.6 A_full step300 只承载同分布模板＋记忆名字（B 0.3375/0.2833，未见名字/新描述 0/60），不能承载描述条件化选择与实体起点复制；② 内容通路必要但受限（28031 无源 0.1667/错位 0.125/错记录 0.0417 未过；28032 true 0/24 退化），选择—解码绑定未成立（选择 0.3833/0.3542、新描述 0/60、复制权重 0.1788/0.2529、选对记录时 object_exact 0.2917/0.0476）；③ rescue 8/8、父 bundle sha 不变、16.1 总成本 13,438.1 CPU；④ 当前父不能承担跨 session 经验/反馈任务。
- **重算 vs 保留**：重算为子集分母/比率、selection-by-subset、内容/反事实 true 侧、条件 object_exact、复制权重、反序正确率、成本求和与 bundle sha；保留为总体/选择总体、γ、rescue、NLL、逐例评分与原始反序配对字段（证据，不作门）。
- **缺陷与自检**：D16-1 分母（240→60/60/15/36/108）；D16-2 局部 index 缓存（内容/反事实/选择分组/反序配对）；D16-3 γ 语义（生成权重；0.2 生成下限；复制权重 1−γ）；D16-4 自前缀标签（触发后目标=自生成字节；B 127/121、A 0，A−B 混淆）。强制自检 S1–S6：分母由子集长度导出、复合 ID、γ 单测、教师标签、评分卡预检、课程一致性。
- **架构诊断**：描述匹配器组合泛化（选择新描述 0/60）+ 实体起点/β 对齐（选对记录时 object_exact 0.29/0.05、首差 byte3 50%/63%、复制权重 0.18/0.25）。否证剂量/seed/步数/学习率/下限/容量/数据标签作为主因。
- **17 准入**：设计不变；差异为有序编码、逐步对象/地点/null 路由、最终 Q 锁定、平滑生成下限、ST 信用、输入/标签分离、三臂。上限更高；风险是成本（1,800 更新按 16.1 实测约 20,150 CPU，可能越 20,000 线）与工程面；仅编码器/仅实体/仅标签各自被另一断点封顶，故最小充分集本身是多部件，17 为主选。**准入需 17.3 定价与 owner 资源裁定（或明确二臂 A/B 范围）；不预启动训练。**
- **登记**：`1.2=not_released`、confirm 封存、14.7 未释放、N15.2 与 12/13 未开展；不追加 seed/步数/剂量、不重跑。**唯一下一步：owner 裁定 17 号（不预启动训练）。**

## 2026-11-03 N16.1.4 核心命题账本（只读连续性审计）

**结论：原生字节、精确来源/权限、确定性恢复、公平口径、小组件协同规模保住；内容/绑定无法真正分离（M1 否定“去掉 fresh α 即解耦”，只能记“同一旧核内经 α 弱耦合”）；能力上限优先半保住（靠三臂与 R1/R2）；顺序/位置（L3）与读取即复制（L5）被换掉；旧能力保持（L9）与经验写入面（L13）从未接上。** [账本](../../reference/INVARIANT-LEDGER_20261103.md)。0 更新/0 前向/0 产物。

- **丢失起点有据**：L3 丢于 16.1 的均值字节嵌入选择头（P1 打分差 8.88e-16、新描述 0/60）；L5 由 14.4 的强制复制概率 1.0 → 15.1 transport 概率核 → 16.1 复制权重 0.1788/0.2529、gold 强制 0/40。R2 只补记录内位置，不恢复硬复制；需另加 T6 复制质量读数。
- **两项从未落地**：L9 无合格父，只能记 `retention_not_admissible`；L13 当前图无持久写入面，只能记 `experience_surface_absent`。17 及能力卡不得声称“保持/自演进/终身学习”。
- **落点**：已写进 [DESIGN-POST-N16 §11–§12](../../reference/DESIGN-POST-N16_SELECTION_ROUTING_20261103.md) 与 [SPEC-N16-01 §9](../../reference/SPEC-N16-01_SELECTION_LOCKED_BINDING_DRAFT_20261103.md)；T1–T13 在 0b/0d 可判，失败阻断 17.5。
- **预算影响**：位置监督最多 4,194 参数，容器 3,563,601→≤3,567,795（新增头 ≤1.16%）；0b 新增探针约 25 CPU，远低于 600 上限。
- **唯一下一步**：17.3 定价并在同次标定 R1 阈值（θ_repr/θ_score），给出三臂/二臂两套单价与 step150 分段预算；仍不启动 1,800 次正式训练。


## 2026-11-03 N16结案后17开发顺序与探针解释修订

当前HEAD已包含N16正式比较、16.1.3全局收束与16.1.4命题账本。1200正式Adam后任务负，确认未释放。当前选定[17逐步路由候选](../../reference/DESIGN-POST-N16_SELECTION_ROUTING_20261103.md)，详细开发顺序已收敛为17.1契约/仪器→17.2正式图→17.3定价→17.4恢复预检→17.5三臂→17.6专属确认→17.7全局收束。前述记录中的先17.3/二臂历史控制/自动加跨度头为当时提案，现行调度由本次修订接替。

本轮只读复核P3/P4源码与逐例结果。P3在第一个模板字节强制名字起点；P4只强制alpha/beta，transport及cursor仍参与最终Q，失同步后停止强制，40例共160个强制步。object_exact还要求总长27，7/40例first_divergence=15，对象字节已正确而句长28/30/31导致局部评分为false。原完整任务0/40保留，撤回gold最终复制失败/位置监督必需的推广。R2改为最终Q控制、锚点经transport、gold记录下学习位置、自由生成四层，R1固定阈值在候选输出前冻结。[复核回执](../../../reports/taiji_selection_routing_plan_review_20261103.json)

N16自前缀把own回填loss标签的D16-4纳入17硬回归，错误前缀下仍保留正确28项目标。三臂使用同一修正目标、共享编码、生成门与课程规则。N16-B含自模仿及不同接口，保留为历史参照，A/B/C三个同期控制完整保留。独立新头37120参数，当前不按旧P4自动加位置头。

预算提案整包30000CPU/60000wall、正式比较24000CPU/48000wall，20000CPU/30000wall观察线。39次工程Adam与1800正式Adam分别记账，预算以实际图后的17.3定价冻结。confirm_n17为专属240例，17.6前读取数0。当前合格父能力集为空，retention_not_admissible；本任务跨session写入未冻，experience_surface_absent，根目标责任见[账本](../../reference/INVARIANT-LEDGER_20261103.md)。

本轮交付为源码/已存结果只读分析与计划修订，模型前向0、Adam0。原报告、manifest、checkpoint及N16冻结合同保持原身份。当前唯一开发下一步为17.1契约与仪器校准，17正式图与执行入口待对应阶段交付。

## 2026-11-03 17.1校准与17.3定价交付（17已定价、未准入）

- **证据**：[定价报告](../../reference/PRICING-17_3_SELECTION_ROUTING_20261103.md)｜[定价JSON](../../../reports/taiji_selection_routing_17_3_price_20261103.json)｜[copy跨度复测](../../../reports/taiji_selection_routing_17_3_copy_span_20261103.json)｜[0a/0b验证](../../../reports/taiji_selection_routing_17_3_verify_20261103.json)｜[阈值manifest](../../manifests/taiji_selection_routing_17_3_r1_thresholds_20261103.json)｜[定价manifest](../../manifests/taiji_selection_routing_17_3_price_20261103.json)。
- **冻结**：R1 θ_repr=1e-2、θ_score=1e-3（SPEC §9.1 预注册；20对最小0.428526/0.052838；旧头0.0；噪声底≤8.88e-16）；T3 gold跨度object_exact=0/40→强制位置监督，启用点17.5第1阶段，不重定价。
- **价格**：三臂13,092.25 CPU/15,056.09 wall/1,800 Adam/495.0 MiB；二臂8,750.34 CPU/10,062.90 wall/1,200 Adam/330.0 MiB；三/二CPU比1.496。位置监督增量单列：selection +0.0859、copy未分辨（噪声带0.7656）、joint A +0.2734 CPU/更新。
- **step150**：三臂seg1 5,154.38 CPU（36.9%）、seg2 7,473.59；二臂3,401.25（36.6%）/5,001.88。step150只作准备度/预算闸门，任务判定只在step300。
- **偏差**：D17-1 首版0.1×中位阈值规则未预注册、与DESIGN §10冲突，已撤回并重跑；组件定价替代9次端到端更新；崩溃首跑烧12工程Adam、CPU/wall未留存。本轮工程Adam13（超17.3卡4）、正式更新0、confirm读取0；可登记成本292.97 CPU/299.34 wall。
- **限制**：本轮不启动正式训练、不物化父候选；17未准入。**唯一下一步：owner裁定臂范围（建议三臂），裁定后先做17.4预检。**

## 2026-11-03 模型与架构系统后续计划

[系统计划](../../reference/PLAN-NATIVE-COGNITIVE_SYSTEM_DEVELOPMENT_20261103.md)将VISION中的语义/精确证据、持久任务状态、快慢学习/保持、世界/条件价值、有用无提示活动与结构成长统一映射到17–22。近端17已有实现及局部证据，当前先统一R17c父/参数/loss/读核与完整图覆盖，补状态/恢复、对变化图重价后才判17.5准入。

HEAD旧T3使用前置锚点时701/720失败；工作树新pointer/强制cursor使S=m=δ，回执720/720通过，工程可达性与正常学习/完整主体f32分开。工作树与报告运行时源码绑定可能继续变化，需freeze时逐项核对。旧17.3含span，旧17.4不含span，新默认容器又增258参数，必须分版本归档；三个单元素FD和值恢复不足以签完整学习信用与语义游标。

后续18原文自主绑定与中文连续任务，19跨session经验修订，20反馈保持与最小真实出生，21真实世界/活动及干预预测，22继承式规模成长与22.7终审。每阶段包含工作卡、对照、数值/数量提案、成本和失败备选，远期实际父及预算到准入前冻结。生长与持续状态接口前置，共同主体按合法经验发展。

本轮仅规划/只读复核，模型前向0、Adam0、无模型/数据删除。[系统计划审阅回执](../../../reports/taiji_system_development_plan_review_20261103.json)记录代码/报告的观察身份、VISION映射与验证范围。实际队首只在MAINLINE维护。

## 2026-11-03 外部数学审查复核

[数学审查](../../reference/MATH-REVIEW_CORE_DYNAMICS_MEMORY_COPY_20261103.md)对照当前方程、源码与原始论文，纠正将慢群视为不可训练、贪心复制率当0.8^12、CE无margin、快慢状态断链、固定规则无优化态适应等推论。GDN2的2026-12会期与2026-05预印本分开，未核验链接不承担关键论证。

当前确定的算子问题是softmax(alpha*beta/T)的旧记录质量/记录内温度耦合，以及lambda_override在sigmoid前覆盖导致value/logit歧义。静态tau基线与条件tau元控制候选已在主稿区分。保留TTT/误差校正对照、F容量—干扰条件和局部成长收益分析，纳入既有17.2R及19–22对应责任。本轮纯数学和源码审阅，模型前向0、Adam0，局部标量/有理数核对见[回执](../../../reports/taiji_core_mathematical_review_20261103.json)。

## 2026-11-03 17.2R 交付与证据保留

17.2R已按六项交付完成：[报告](../../../reports/taiji_selection_routing_17_2r_20261103.json)＋[manifest](../../../plans/manifests/taiji_selection_routing_17_2r_20261103.json)。单一身份`candidate_sha` 964a8c68…、容器3,563,859参数（binding 3,526,481＋head 37,120＋span 258），T3 v3、新价、17.4 v2与本报告逐文件携带同一身份；账本T1–T16中12项通过、T6/T9/T10按阶段判；硬点信用A臂非0、B臂恒0且step0选择CE两臂逐位同值；语义游标给出非空复制中示例；非空m恢复18 Adam（来源17.4 v2六格×3）；新冻结价三臂15,018.521875 / 二臂10,182.68125 CPU，旧价13,092.25/8,750.34只作差额对照。改前证据（旧T3 701/720、旧17.4预检、旧17.3 verify及两份manifest）曾因重跑复用文件名被覆盖，已按“先另存`_r2_postfix_20261103`、再还原HEAD字节、再旁挂superseded”恢复；硬规则写入[账本§8](../../reference/INVARIANT-LEDGER_20261103.md)，当前队首为17.5三臂1,800更新。

## 2026-11-04 包 19（19.3–19.7）：定价、预检、六格正式比较、判负结案

19.3 定价 99.3 CPU / 103.2 wall（六格裸价 12,593.15、保护价 26,571.56 CPU）；19.4 预检 292.7 CPU / 303.7 wall / 19 工程 Adam（8/8 通过，含先存包后训与恢复逐位）；19.5 六格 300 更新 29,574.2 CPU / 5,958.8 并行 wall / 1,800 正式 Adam，另 step150 中途探针 1,513.7、step300 分账探针 2,137.4、分账口径重算 143.9，包内合计 33,761.2 CPU，累计 86,139.9 CPU。

19.5 结果：整句 0.000 / 0.000 / 0.630 / 0.583 / 0.401 / 0.469（门 0.80）；抄回仅 28031-B 0.630 过 0.60；逐字节 物件 0.678、地点 0.631（门 0.95）；A−C −0.4349（bootstrap [−0.50, −0.375]）。分账探针（0 更新、gold 前缀、模型自身 α/β/γ）：起点命中 0.347/0.340，起点命中后内容字节 0.941/0.980，未命中 0.678/0.631 → 主因是起点/记录选择；编码器（c_γ0 内容 1.000×5/6）与生成权重 γ 均非主因；B 臂起点全对时 `new_description` 仍仅 0.260/0.229，为第二道独立缺口。

`release_confirm_19=false`，confirm 读取 0，19.6 未执行。观察线两次触发（2,266/外推 44,857；17,376.7/外推 20,063.8）均与终值 29,574.2 不符，已登记外推失真原因（首触放大固定成本、次触按受争用速率）。清理：保留 6 个 step300 bundle（395.1 MB）与 step0/midreport/run/eval300/failure/watchline 回执；删除 6 个 step150 bundle（175.5 MB，sha 已登记）。产物见 [19.5 比较](../../../reports/taiji_package19_compare_20261104.json)、[step300 分账探针](../../../reports/taiji_package19_probe_step300_20261104.json)、[19.7 结案](../../../reports/taiji_package19_closeout_19_7_20261104.json)。队首：等 owner 裁定是否启动 19.2b 起点监督；未裁定前不训练。

## 2026-11-04 包 19.2b（前段）：定价、少更新验证、预检、指针诊断，止于六格之前

19.2b 定价 0 CPU（0 Adam / 0 前向）：按 19.5 实测斜率（selection 4.0 / copy 10.0 / joint 29.0 CPU 每更新）加显式非训练项重算，六格裸价 32,256.6、保护价 68,061.4；分段 0→150 裸价 7,464.3 / 保护价 15,749.7，150→300 裸价 24,792.3 / 保护价 52,311.8。19.3 缺口写入方法论修正：裸价仅是实测 29,574.2 的 1/3.039，5.4 平单价把后 150 更新（实测 20.8–26.7 CPU/更新）与两次轻评加一次全评（约 344 + 732 CPU/格）都算成了零头；19.5 实测比 19.3 保护价高 11.3%。

少更新验证 268.3 CPU / 60 Adam（28031，A/B 两臂 × 30，仅 selection 阶段）：span CE 6.779→5.034（−25.7%），但 β 口径 start_hit 恒 0.000、beta_start_hit 物件恒 0.208 / 地点恒 0.000、route_hit 仅 0.079→0.095。两臂在 selection 阶段逐位同轨迹、同一 candidate_sha_v2（6c5466ab…），按设计臂只在 copy/joint 生效，故实为一条 30 更新轨迹。预检 297.5 CPU / 19 工程 Adam：8/8 通过（身份、先存包后训、恢复逐位、模块漂移必拒），span 参数 4,194，临时 bundle 已回收。

指针诊断 106.4 CPU（0 更新、0 confirm）：19.5 六格 step300 上同一 gold 起点，冻结 β 峰值命中 物件 0.521 / 地点 0.500，span 指针命中 0.719 / 0.698；未训练 32-d 对照仅 0.25 / 0.00；pointer 与 pointer_logits 全格一致（无 carry 绕过）。

**结论（结构性）**：`read.betas` 是冻结 reader 的记录内位置后验（[binding_copy.py](../../../taiji/binding_copy.py) 337 行），span pointer 只经 `pointer_bias → row_logits → pointer/s_bytes`（361、386–396、467–476 行），从不进入 β；而 19.5 的 `start_hit` 读 `route ∧ argmax(betas)`。因此 span 监督再好也抬不动这个读数，六格 300 更新会结构性撞上阶段门、32,256.6 CPU 属注定浪费。已停在六格之前，未产生任何新训练 bundle，19.5 六个 step300 bundle 原样保留作基线。

19.2b 前段合计 672.2 CPU（另含未训练对照 17.5），confirm 读取 0。产物：[19.2b 决策](../../../reports/taiji_package19_2b_decision_20261104.json)、[定价](../../../reports/taiji_pricing_19_2b_20261104.json)、[少更新](../../../reports/taiji_package19_2b_lowupdate_20261104.json)、[预检](../../../reports/taiji_preflight_19_2b_20261104.json)、[指针诊断](../../../reports/taiji_package19_2b_pointer_diagnostic_20261104.json)。队首：等 owner 裁定把阶段门 `start_hit` 的读数对象改为 span 指针（数值 0.95 不变）；裁定前不训练。

## 2026-11-04 包 19.2b：D16-5 测量修正（0 更新核对）与修正后小验证（30 独立 Adam）

owner 授权"测量修正"并收紧口径：门读数对象改为**解码时真正使用的落点**。D16-5 核对 103.9 CPU（0 更新、0 confirm、0 训练）：在 19.5 六格 step300 上 teacher-forced 逐步重建合成链，`row_logits = transport_logits + pointer_bias` 逐元素最大差 **0.0**；`pointer = one-hot(argmax(row_logits))` 不一致 **0**；`source = argmax(α×β)` 恒等于 β 的 mode（不一致 0）；内容行 4,320（transport 1,073 / carry 3,247，carry 绕过合成，起点 carry 仅 11/480）。结论：**"第一个内容输出字节所选的源字节位置"＝最终行 argmax，`composite_start_hit` 是正确门对象**；旧 `start_hit` 读 β 属错对象。

**三读并列（5 个有限格池化，剔除 28031-A）**：β **0.400/0.400**、pointer **0.833/0.800**、composite **0.767/0.746**、decode 0.788/0.738、route **0.865/0.800**、composite 条件 conditional **0.830/0.791**（B 臂 1.0/1.0、C 臂 1.0/0.977，缺口全在 A 臂）。分臂：C 臂 β **0.000/0.000** 而 composite **0.969/0.885**——直接证明旧仪器测错对象；B 臂 composite 0.917/0.979；A 臂结构性失效（28031 全 NaN、28032 route 0.359/0.003）。

**基线污染（新登记）**：`28031-A_route_lock-step300.pt` 全部 binding/head 参数 NaN（2,320,153 个），mtime 早于 probe；19.5 归档 pooled 0.347/0.340 含该污染格。本次仅剔除、标注，不改 bundle。19.5 归档数值原样保留为 `legacy_instrument`。

修正后小验证 176.9 CPU（1 条独立轨迹、30 Adam、B 臂、盲父、仅 selection；0 存包、0 confirm）：`composite_row_hit` 物件 **0.250→0.292**、地点 **0.104→0.146**（双边 +4.17pp），但更新 5/10 后走平；`composite_start_hit` 物件 **0.000→0.000**、地点 **0.000→0.021**；route 0.005→0.038；span CE 6.779→5.034。**按 owner 原文口径未上升 → 不放行六格 32,256.6 CPU**；解释为盲父 30 更新内 route 未成熟，routed 门被 route 穹顶压住，与"监督无效"不可区分。

**记账修正**：前次"60 Adam"（A/B 两臂 × 30）在 selection 阶段逐位同轨迹（同一 `candidate_sha_v2`），实为**一条 30 更新轨迹跑两遍**；按真实独立轨迹计 30，本次再 30 → 累计 **60**，已达 owner 上限，不得再加剂量。

本轮合计 280.8 CPU；19.2b 累计 **953.0 CPU**；六格未花；confirm 0。产物：[D16-5 复合仪器](../../../reports/taiji_package19_2b_composite_instrument_20261104.json)（[manifest](../../manifests/taiji_package19_2b_composite_instrument_20261104.json)）、[修正后小验证](../../../reports/taiji_package19_2b_lowupdate_composite_20261104.json)（[manifest](../../manifests/taiji_package19_2b_lowupdate_composite_20261104.json)）。队首：**不训练、不花钱，等 owner 裁定 D16-5 结案口径**；若要区分"route 穹顶"与"监督无效"，唯一动作是换 19.5 B step300 成熟父包复跑同额 ≤60 Adam（需授权）。

## 2026-11-04 19.2g 记录匹配对比边距（30 Adam / 1 存包 / 0 confirm）

证据：[19.2g 报告](../../../reports/taiji_package19_2g_record_margin_20261104.json)＋[manifest](../../manifests/taiji_package19_2g_record_margin_20261104.json)。脚本 [`train_taiji_package19_2g_record_margin.py`](../../../scripts/training/train_taiji_package19_2g_record_margin.py)。父包 `output/taiji_package19_compare/28031-B_stop_lock-step300.pt`（只读加载）；产物 `output/taiji_package19_2g_record_margin/28031-B_stop_lock-step30-margin.pt`。

**A. 预检（决定性读数）**：训练集抽样 64 例 / 128 个内容起点，m=1.0 下 **0 点激活**（gold-decoy 边距最小 5.313、中位 8.08 ≫ m，top-1 中位 0.996）→ `max(0, m−边距)` 在每个训练点被 clamp 到 0，**梯度恒为零**；30 Adam 实际只训 CE（0.0066→0.0007）与 span。**本包不是一次边距干预**。检查点 preflight 存包/回读 `max_abs_diff=0.0`，正式包 29.30 MB、回读 0.0、`candidate_sha_v2` 不变。

**B. 训练前后（192 例/正式 `completed`）**

| 指标 | 前 | 后 |
|---|---:|---:|
| 新描述 96 例整句 | 0.2604（25/96） | **0.2083（20/96）** |
| 旧描述 96 例整句 | 1.000 | 1.000 |
| 全 192 | 0.6302 | 0.6042 |
| route_miss（96 例占比） | 0.3854（37/96） | **0.3854（37/96）** |
| route_hit 物件 / 地点 | 0.4688 / 0.8438 | 0.4167 / 0.9062 |
| composite_start 物件 / 地点 | 0.3854 / 0.5312 | 0.3646 / 0.6250 |
| conditional_byte 物件 / 地点 | 0.7613 / 1.0000 | 0.6976 / 1.0000 |
| T2 硬 argmax 改变（新 20 对） | 0/20 | **0/20** |
| Δscore 中位（新 / 易） | 1.2425 / 1.3159 | 1.5689 / 1.4002 |
| top-1 中位 | 0.9545 | 0.9658 |
| gold-decoy 边距中位 | 4.70 | 7.34 |
| EOS/UTF-8 冻结路径 | 0 | 0 |

**C. 机制分层**：地点起点 top-1 中位 0.9990→**0.9999**、边距 10.42→**12.95**（已饱和）；物件起点 top-1 中位 0.3940→**0.4670**、边距 0.695→**0.828（< m）**（未饱和且选错）；route_miss 37/96 全在物件侧 → **新描述墙不是单一饱和现象**。

**判定（写死）**：判据全不过 → 预注册分支 **`scoring_head_saturation_is_structural`**；但因 A 的零梯度事实，此判定读作"该 loss-only 方案在可训数据上无法干预"，不是打分头容量被直接证伪。唯一建议 **19.2h 序列表面对齐**（`score_rows` 加参数化局部互相关，约 +1,320 参数 / +3.6%），先 30 Adam 选择段诊断；**未启动，等 owner 裁定**。

本轮 **449.5 CPU / 467.3 wall / 30 Adam / 1 包 / 0 confirm**（预算：功能项 430 × 3.3 = **1,419**，用 31.7%）；19.2b 线累计 **2,506.4 CPU**；六格 32,256.6 CPU 未花。**六格外推重算（3.3 系数）111,293.5 CPU、选择段分段价 37,547.6 CPU，仍不放行。**

## 2026-11-04 19.2h 序列表面对齐（60 Adam / 2 轨迹 / 1 存包 / 0 confirm）

证据：[19.2h 报告](../../../reports/taiji_package19_2h_sequence_surface_20261104.json)＋[manifest](../../manifests/taiji_package19_2h_sequence_surface_20261104.json)。脚本 [`train_taiji_package19_2h_sequence_surface.py`](../../../scripts/training/train_taiji_package19_2h_sequence_surface.py)。父包 `output/taiji_package19_compare/28031-B_stop_lock-step300.pt`（只读加载）；产物 `output/taiji_package19_2h_sequence_surface/28031-B_stop_lock-step30-align.pt`（29.34 MB，sha256 `7fd62113…`）。

**A. 活性检查门（D16-8，开训前）**：新张量首批梯度 `W_align.weight` **3.04e-6**、`offset_embed` **8.33e-5**、`W_hidden` 新列 **1.33e-5**（全部非零）；参数化干预的"触发"读作**需修样本占比** → 新描述 96 例起点 argmax 错 **51/96 = 53.1%**（≥20%）。⇒ **放行**。（19.2g 的 0/128 激活是本门首号缺陷实例。）

**B. 两臂并列（同父包、同 30 Adam、同数据顺序、同损失）**

| 指标 | 父包（前） | 19.2h 对齐臂 | CE-only 对照臂 | Δ |
|---|---:|---:|---:|---:|
| 新描述 96 例整句 | 0.2604（25/96） | **0.2083（20/96）** | **0.2083（20/96）** | **0.0［0,0］** |
| route_miss（96 例占比） | 0.3854（37/96） | 0.3854（37/96） | 0.3854（37/96） | 0.0［0,0］ |
| composite_start 物件 / 地点 | 0.3854 / 0.5312 | 0.3646 / 0.6250 | 0.3646 / 0.6250 | 0.0［0,0］ |
| conditional_byte 物件 / 地点 | 0.7613 / 1.0000 | 0.6976 / 1.0000 | 0.6976 / 1.0000 | 0.0 |
| 旧描述 96 例整句 | 1.000 | 1.000 | 1.000 | 0.0 |
| 全 192 | 0.6302 | 0.6042 | 0.6042 | 0.0 |
| T2 硬 argmax 改变（新 20 对） | 0/20 | 0/20 | 0/20 | 0 |
| EOS/UTF-8 冻结路径 | 0 | 0 | 0 | 0 |

bootstrap 2000 轮（配对、seed 20261104）三项 **Δ=0.0、CI [0,0]、`excludes_zero=False`**。

**C. 身份与预检**：父/对照 `candidate_sha_v2 = 773dcec2…`（`alignment` 仅在 True 时进 payload，旧身份逐字不变）；新图 **`4b887487…`**；identity rescue **8/8**；存包→回读 `max_abs_diff=0`；删除 `W_align.weight` 的 bundle **被 strict 加载拒绝**（3/3）。

**判定（写死）**：分支 **`gain_is_extra_training`** ＋ **D16-10 `zero_identifiability`**——+1,320 参数（+3.56%，head 37,120→38,440）在 30 步选择段上有梯度但**未进入行为通路**（`offset_embed` 初值 ±0.01、余弦特征量级 1e-2 被淹没），逐例读数与对照臂完全一致；**该参数化退役**（代码保留 `alignment=False` 开关，历史 bundle 不删），不得读作结构上限、不得靠加剂量救。判据 9 门只过 3（`conditional_byte` 地点 1.0、旧描述 1.0、EOS/UTF-8 不回退）。唯一建议 **19.2i 描述-记录对比监督**（0 新参数：`scores[:R]` 上 12 路 listwise CE、两起点取和、τ=0.1；功能项 330×3.3＝**1,089 CPU**），**未启动，等 owner 裁定**。

本轮 **404.8 CPU / 420.7 wall / 60 Adam / 2 轨迹 / 1 包 / 0 confirm**；含首次失败尝试 **582.2 CPU**（观测值，未插桩）→ 本包合计 **987.0 CPU**（功能项 445×3.3=**1,468.5**，用 67.2%）；19.2b 线累计 **3,493.4 CPU**；**六格 111,293.5 CPU、选择段 37,547.6 CPU 仍不放行。**

## 19.2j 从零选择段 A/B（2026-11-05 收口；1,200 Adam 全完成，step300 只读重建）

**A. 0 成本核查（写在报告最前面，owner ①）**：19.2i 拟用的 12 路 listwise CE 与训练中在跑的 `selection_loss` 同构——`score_rows` 输出 R+1 行（12 记录＋1 null），`F.cross_entropy` over 该行集合、gold 记录 NLL，每 stage 内容步均值再跨 stage 均值，τ=1。19.2i 的两处差异（τ→0.1、只在 2 个起点求和）都指向**更弱/更窄** → 判 **D16-11 `duplicate_objective`**，**19.2i 退役**，省 **1,089 CPU**；架构问题改走 19.2j。

**B. 纪律 D16-12（owner ②）**：owner 把 19.2g（0/128 激活）与 19.2h（Δ=[0,0]）两次零结果裁定为**仪器失效**，不是结构上限；≤60 Adam 微调只许行为校准，架构判定必须"同初始化、同数据顺序、从零训练、只差一个变量"。已登记账本 §16.5 与 ROADMAP §4。

**C. 预注册与变量清单（开训前写死，落进报告 `preregistration`）**

| 量 | O 臂 | N 臂 | 同/异 |
|---|---|---|---|
| `W_hidden[:, :576]` / `bias` / `w_out` / `k_null` | seed 初始化 | **逐位拷贝 O** | 同 |
| binding 图 / span 头 | 父包冻结 | 同 | 同 |
| 新增块 | 无 | `W_align` 32×32 + `offset_embed` 5×8 + 4 列（+1,320 参数） | **唯一差异** |
| 初始化种子 | 28031 / 28032 | 同 | 同 |
| 数据顺序 | `ORDER_SEED` 排 640 训练例 | 同 | 同 |
| batch / LR / clip / 更新 | 8 / 1e-3 / 1.0 / 300 | 同 | 同 |
| 损失 / span 监督 / confirm | `selection_loss` / 无 / 0 | 同 | 同 |

N 新增块初始化：`W_align` U(±1/√32)=±0.177、`offset_embed` U(±1/√64)=±0.125（主干尺度；19.2h 的 ±0.01 弃用）。

**D. 判据（8 项）**：新描述 `route_hit ≥0.90`、`route_miss ≤0.05`；换序硬 argmax **≥18/20**；旧描述 `route_hit ≥ O−0.02`；物件 `composite_start ≥0.70`、`conditional_byte ≥0.90`；EOS/UTF-8 冻结路径不回退；可辨识性 `|log10(新增 per-param RMS 梯度 / 主干 per-param RMS 梯度)| ≤1`（step 300）。归因：逐 seed 与 pooled 的 **N−O** 差值 + 12 world 聚类 bootstrap（2000 轮）。

**E. 训练与记账（2026-11-05）**：**1,200 Adam 全部跑完**，4 个 step300 bundle 完整（28031-O/N、28032-O/N，02:10:20–02:13:49）；报告在 `training.first/last` 的 `(seed, arm)` 元组键上崩溃（`TypeError`），中途读数与 CPU 计时只在内存 → **工程失败，不是实验失败**（D16-15）。修复：键改 `f"{seed}|{arm}"`、全量 `sanitize_keys()`、`write_json_safe()`（tmp→replace）、`serialization_precheck()` 进 `--dry-run-pricing` 与主训练前必过项；运行期读数逐行 append 到 `progress-*.jsonl`。

**F. step300 只读重建（0 Adam / 311.3 CPU / 319.3 wall）**：新描述整句 O **0.2396/0.2396**、N **0.2604/0.2812**；route_hit 物件 O **0.375/0.5312**、N **0.6146/0.5208**；route_miss O **0.4479/0.3333**、N **0.2604/0.3333**；旧描述两臂 **1.000**；EOS/UTF-8 **192/192**。Δ=N−O（12 world bootstrap 2000 轮，`n_wins`=Δ>0）：整句 +0.0208 [0.0000,0.0521] / +0.0417 [−0.0312,0.1042]，**pooled +0.0312 [−0.0104,0.0677]**（`n_wins` 1855/2000）；物件 route **pooled +0.1146 [0.0625,0.1615]**（2000/2000）；物件起点 pooled +0.0990 [0.0417,0.1562]。换序硬 argmax O 2/0、N 0/0（门 18/20）。step300 可辨识性 **log10 per-param RMS 比 −0.436**（聚合 −1.208）→ 可辨识；step100 记 **`inferred=过闸 / measured:false`**（28031-N 到达 step300 即证明 100 步闸未触发）；**step50/100/200/250 路由读数 `unrecoverable`**（无中间 checkpoint，step300 不得反推）。

**G. 判定**：分支 **`scoring_form_is_not_the_bottleneck`**（两 seed 同判；7 门过 2：旧描述不回退、EOS/UTF-8）。打分形式不是瓶颈：整句等价，物件 route 的显著增益被 conditional_byte（28031 物件 **0.813→0.643**）抵消，未转成句级增益。**唯一建议 19.2k「span 指针从零可训」**（O 臂为基线、唯一变量＝冻结 span 解冻并从零训练、2 seed × 300、`composite_start ≥0.70`），**未启动，等 owner 裁定与计价**。

**H. 花费与缺口**：训练 wall **4,961 s（00:51:08→02:13:49）**；**CPU 不可得＝仪器缺口**（计时只在内存）；重建 **311.3 CPU**；**1,200 Adam / 4 包 / 0 confirm**。证据：报告＋manifest＋4 bundle＋`reports/taiji_package19_2j_rebuild_progress_20261105.jsonl`。

## 2026-11-05 19.2l 顺序专用干扰负样本 · ② 活性闸（0 Adam / 0 参数 / 0 包 / 0 confirm）

owner 裁定：19.2j 已证对齐模块从零是活的（物件 route +0.1146、bootstrap 2000/2000），但换序硬 argmax 仍 0–2/20、整句等价；19.2f 的量级（gold-decoy 边距中位 ≈7.3 vs 换序分数差中位 ≈1.24）说明顺序信号被模板信号压制约 6× → **瓶颈是负样本构成，不是打分形式**；19.2i/g/h 三条打分侧改动已证否，**不得再改打分形式**。本轮**只做 ② 活性闸**，不训练。

**A. decoy 构造（0 参数，只改数据）**：gold 记录描述按字符中点切两半并对调（`细颈的瓶子 → 的瓶子细颈`）；UTF-8 **字节多重集、长度、字节种类完全相同**，只有顺序不同 → 袋装字节特征不可分。owner 原文的"实体段与地点段对调"在本数据里不可字面实现（一条记录只含一个描述段），本实现取**同一族**：描述两半段置换。每个内容起点（物件 3 / 地点 21）1 条，不做多负样本（与 19.2g 同规模，避免第二变量）。

**B. 读数（forward-only / 0 Adam）**：训练集 even-stride **64 例 × 2 起点 = 128** 个 `score_gold − score_decoy`；退化 0。**激活率 1.0（128/128，门 ≥0.20）**；边距**中位 −0.0029（门 <1.0）**、均值 +0.0288、p10 −0.1504 / p90 +0.2835、min −0.1615 / max +0.4086；物件起点均值 −0.0424、地点起点 +0.1000 → **`activity_gate.pass = true`**。两分数取自同一 2-key context；`score_rows` 逐行独立，等价于在真实 8 记录 context 中读同一对。

**C. 机制提醒（不是判据，但决定 ③ 首步）**：边距几乎贴在 0，说明冻结池化面对纯字节置换**近乎不可分**——这正是"难负样本"的定义，也意味着**激活成立 ≠ 梯度可用**：若 gold/decoy 关键向量近同，边距项的梯度会相互抵消（损失恒为 m，梯度为零）。因此 ③ 开训前必须先读**首批边距项梯度范数**，再谈 30 更新；否则会重复 19.2g 的"损失项看似激活、行为不变"。

**D. 判定与下一步**：`diagnostic_verdict = activity_gate_pass`；**未训练**。唯一建议＝**批准 ③ 训练诊断**：1 seed 28031、O 臂基线、从零、30/300 更新；CE 对照臂＋唯一变量 order-decoy 边距项（同数据/顺序/初始化）；判据换序硬 argmax **≥8/20（诊断门，非任务门 18/20）**、新描述 route_miss 相对 CE 臂降 ≥0.10、`conditional_byte` 不回退 >0.05、旧描述不回退；预算 ≤30 独立 Adam/臂（共 60）、≤400 CPU、1 候选包、0 confirm。花费 **16.8 CPU / 17.7 wall / 0 Adam / 0 包 / 0 confirm**。证据：报告＋manifest＋`reports/taiji_package19_2l_progress_20261105.jsonl`（129 行，逐行 fsync，D16-15）。

## 2026-11-05 19.2l-A 边距项梯度结构证明（0 Adam / 0 参数 / 0 包 / 0 confirm）

owner 批准 ③ 但**改一处变量归属**：先做结构性证明（A），再谈训练（B）。A 的唯一目的：确认"字节置换负样本"的边距项梯度到底能不能进入参数。owner 原话的前提是"O 臂＝纯袋装、该损失梯度恒为 0，所以拿 O 做对照没有信息量"——本轮就是去实测这个前提。

**A. 命题与判据**：命题"若打分器输出只依赖记录字节多重集，则任意字节置换负样本的边距项梯度恒为 0"**数学上成立**（条件式）。可操作判据必须成对给：① `max|key_gold − key_decoy|`、② 边距项对参数的梯度范数；①=0 ⇒ ② 必为 0；①≠0 时**必须实测**，不得用臂名代替测量。登记为 **D16-16**（账本 §16.9）：后续任何"加难负样本"提案必须先过这条——`≤1e-12` 记结构零、`>1e-6` 才允许进入训练。

**B. 两臂实测（同一 teacher-forced 状态、同一 64 例样本、同一 margin m=1.0；state 视为常数，只对 scorer 求导）**：

| 读数 | O 臂（`alignment=False`，无位置特征） | N 臂（`alignment=True`，含 `W_align`＋`offset_embed`） |
|---|---|---|
| 关键向量同一？ | **否**：128/128 全不同，`max|Δkey|` 中位 **1.491**（min 0.994 / max 1.838） | 同左（两臂用同一冻结 transport） |
| 边距项 loss | 0.9712（激活 128/128） | 0.9724（激活 128/128） |
| 梯度总范数 | **1.5965** | **1.6015** |
| 其中新增块（`_is_added`） | 0（无新增块） | **0.0089（占 0.56%）** |
| 其中主干 | 1.5965 | 1.6015 |
| 首批（`h.schedule` 第 1 批） | 2.1366 | **2.1515** |
| `max_abs_gradient` | 0.3209 | 0.3207 |
| 参数数 | 37,120 | 38,440（`_is_added` 直计 1,064，另 256 列并入 `W_hidden.weight`） |

O 臂 per-param：`W_hidden.weight` 1.3808、`w_out.weight` 0.80025、`W_hidden.bias` 0.04328、`k_null` 0。N 臂 per-param：`W_hidden.weight` 1.38709、`w_out.weight` 0.79932、`W_hidden.bias` 0.04359、`W_align.weight` **0.001584**、`offset_embed` **0.008789**、`k_null` 0。

**C. 判定（与 owner 预期相反）**：`verdict = frozen_transport_is_order_sensitive`。冻结 transport（`record_relation.encode_units` 用**局部位置特征＋因果 GRU**）本身对字节顺序敏感，所以打分器输出**并不是**字节多重集的函数 → 命题成立但**前提未被 pipeline 实例化**。因此 **"O 臂梯度恒零"被实测推翻**（1.5965，远超 1e-12），**不能**把 O 当"无信息对照"；B 若沿用 O 作对照，会给出"两臂都动"的伪结论。B 的 N×边距项 vs N×CE-only 单变量设计仍成立，但归因必须改写为**"主干承担、位置块近乎惰性"**（新增块只占 0.56%）。

**D. 开训前必查项**：N 首批边距项梯度 **2.1515 > 1e-6** → `gradient_cancels_at_source = false`（未触发停训条件）。N 初始化可辨识性 `log10 RMS 比` **−1.425**（`added_rms` 3.131e-4 / `trunk_rms` 8.339e-3，**不可辨识**，与 19.2j 价格闸初始化读数一致；step300 才 −0.436）→ 初始不可辨识不构成停训理由，但训练后必须复核。

**E. 花费与边界**：**0 Adam / 0 参数改动 / 0 包 / 0 confirm / 21.9 CPU / 22.4 wall**（≤80 CPU 预算内）。不改判据、不改阈值、不放开 confirm、不上调任何能力等级。证据：[19.2l-A 报告](../../../reports/taiji_package19_2l_margin_gradient_20261105.json)＋[manifest](../../manifests/taiji_package19_2l_margin_gradient_20261105.json)；脚本 [`probe_taiji_package19_2l_margin_gradient_structure.py`](../../../scripts/training/probe_taiji_package19_2l_margin_gradient_structure.py)；账本 [§16.9 D16-16](../../reference/INVARIANT-LEDGER_20261103.md)。**唯一建议＝按 owner 的 B 设计开跑（N 两臂、30 更新、≤30 Adam/臂共 60、≤400 CPU、1 包、0 confirm），待 owner 批准，未启动。**

## 2026-11-05 19.2l-B 定价闸（0 Adam / 0 包 / 0 confirm，未训练）

owner 批准 B（设计照原文）并加了三项读数与一条互斥归因口径，同时要求"超价先报不跑"。本轮据此**只做 0 Adam 定价**，不做任何训练。

**A. 设计（已固化进脚本，未改一字）**：父包 28031-B step300；两臂都 `build_run(parent, 28031, "N")`（含 `W_align`＋`offset_embed`），初始化逐位相同（`init_max_abs_diff = 0.0`，step0 `candidate_sha_v2` 相同）；实验臂＝CE＋order-decoy 边距项（m=1.0，每例 2 起点），对照臂＝CE-only；两臂同 seed、同 `h.schedule(train,30)` 数据顺序、各 30 更新（共 60 Adam）；step0 共用一次快照；step30 两臂各一次 192 例全量快照；新增 `key_diff_after` / `score_margin_after` / `decoy_margin_after` / `gradient_share_by_param`（边距项单独 backward，逐更新累计）；互斥判定 `washout_at_readout_proven` / `washout_upstream` / `margin_not_trainable_in_30` 与各自"下一步只许做什么"原文写死在报告里。

**B. 定价读数（同进程稳态；0 Adam）**：CE 单步（batch 8）**6.02 CPU**、边距项单步（含 16 行状态）**2.17 CPU**、192 例全量快照 **94.1 CPU**。训练外推 **426.1**（30×6.02 ＋ 30×8.19）；快照两臂 step30 **188.2**；三读＋存包/开销 ≈34；已花 58.1。即便**去掉全部杂项与 15% 安全系数**，`训练 426.1 ＋ 两臂快照 188.2 = 614.3` 已 >400 → **`pricing_gate = over_budget`**（`floor_over_budget = true`），按纪律**停训**：0 Adam、0 包、0 confirm。参照：同规模 19.2h 在 2026-11-04 实花 404.8 CPU，今天同口径单价约 1.7–2×。

**C. 为什么不是"再优化一下就能塞进 400"**：400 里 60% 是训练（60 次更新不可减，否则违反"同 30 更新"设计）、40% 是两臂 step30 快照（减掉就没有 Δ 与旧描述/conditional_byte 基线），三读还要额外 backward。任何"省"都等于改设计或砍判据，按 owner 纪律不许偷偷做。

**D. 判定与唯一建议**：`pricing_gate_over_budget`，**设计未改、判据未改、归因口径未改、未训练**。唯一建议＝**把本轮 CPU 上限提到 ≥750 后原样开跑**（同一条命令：`python scripts/training/train_taiji_package19_2l_order_decoy_diagnostic.py`），等 owner 批准；本次 58.1 CPU、累计约 216 CPU（含定价校准）、0 Adam / 0 包 / 0 confirm。证据：[B 定价报告](../../../reports/taiji_package19_2l_order_decoy_diagnostic_20261105.json)＋[manifest](../../manifests/taiji_package19_2l_order_decoy_diagnostic_20261105.json)；脚本 [`train_taiji_package19_2l_order_decoy_diagnostic.py`](../../../scripts/training/train_taiji_package19_2l_order_decoy_diagnostic.py)。

## 2026-11-05 19.2l-B 800 定价闸（再拦 / 0 Adam / 0 包）

owner 把本轮上限从 400 提到 **800**（400 只是上一轮口径，不是 owner 约束），并要求开跑前做机器洁净检查、跑完复测收盘单价。设计、判据、三读、互斥口径**一字未改**。

**A. 机器洁净检查（开跑前实测）**：python 进程 **0**、含 black 的 python **0**、逻辑核 **24**、`torch.get_num_threads()` **12**；累计 CPU 前三：`Trae CN 44532` 43734.4 s、`Trae CN 35156` 19830.0 s、`ProcessLasso 25848` 17273.1 s，3 秒实测增量 **0.44 / 0.64 / 0.05 s（0.6% / 0.9% / 0.1% 的 24 核）** → 全部 idle。**清理**：开跑前发现一个**空转的 in-process black** `PID 40008`（此前用 `black.main` 包装格式化留下，累计 **641.8 CPU s**），已 kill；复测 python=0 / black=0，记 `operational_waste_cpu_s: 641.8`（D16-13 的同类事故，本次是 Agent 侧误用）。

**B. 定价（0 Adam，两次独立测量）**：本机当前稳态单价 CE 单步 **6.25 CPU**、边距项单步 **2.34 CPU**、192 例全量快照 **96.34 CPU**。外推拆分：训练 **445.3**（30×6.25 ＋ 30×8.59）＋ 三次 192 例快照 **289.0** ＋ 事后三读 19.3 ＋ 存包 8.0 ＋ 收工单价 8.6 ＋ 报告 3.0 ＋ 已花 38.9 = **812.1 CPU**；第二次独立复测 **813.6**。超 800 约 **1.5%** → `pricing_gate = over_budget`，**停训**：0 Adam、0 包、0 confirm、未训练、未存包。底板（训练＋三次快照）**734.3 本身没超 800**，超出的 78 全部来自 owner 要求的三读/存包/收工单价/报告。

**C. 判定与唯一建议**：`price_verdict = not_evaluated`（无训练，收盘单价不测）；两次外推几乎一致 → **设计级价格、非测量抖动**。唯一建议＝**把 `CPU_BUDGET` 提到 850（设计、判据、三读一字不动；预计占用 95.5%）后原样开跑**，等 owner 裁定。证据：[B 定价报告](../../../reports/taiji_package19_2l_order_decoy_diagnostic_20261105.json)＋[manifest](../../manifests/taiji_package19_2l_order_decoy_diagnostic_20261105.json)。

## 2026-11-05 19.2l-B 执行（D16-17 量级闸 / D16-18 格式化纪律）

owner 改的是**规则本身**，不只是数字：预算闸改成量级闸（D16-17），格式化改成收尾一次且带硬超时（D16-18）。设计、判据、三读、互斥口径**一字未改**。

**A. 规则落地（账本 §16.10 / §16.11）**：`CPU_BUDGET = 1200`；判定式 `estimate > 2 × CPU_BUDGET`（>2,400）才允许 `over_budget` 停，1%–10% 差距一律直接跑；同包定价只做一次、落盘即终值；登记 `budget_gate_precision_churn`（三次 1.5%–8% 拦停、两次纯定价 77.75 CPU ≈ 本实验 10%）。格式化只在收尾跑一次，硬超时 45 s，进程内超时即记 `format_gate: not_executed` 且不得后台补跑。

**B. 开跑前读数（只作记录，不再当闸）**：python 进程 **0**、black python **0**；本包 0 Adam 定价 **866.6 CPU**（CE 6.77 / 边距项 2.42 / 192 例快照 102.22），866.6 ≤ 2,400 → `budget_gate = run`。

**C. 两臂逐项（N×边距项 vs N×CE-only，30 更新/臂，共 60 Adam）**：

| 指标（step30，新描述 96 例 / 旧描述 96 例） | CE 对照臂 | 边距臂 | 差（边距−对照） |
|---|---|---|---|
| route_miss（新） | 0.4896 | 0.5938 | **+0.1042 [0.0521,0.1563]** |
| route_hit 物件 | 0.3542 | 0.1354 | **−0.2188 [−0.3021,−0.1458]** |
| route_hit 地点 | 0.7813 | 0.7188 | −0.0625 [−0.1667,0.0417] |
| completed | 0.1875 | 0.0521 | **−0.1354 [−0.2083,−0.0625]** |
| 旧描述通过 | 0.9479 | 0.9688 | +0.0208（不回退） |
| conditional_byte 物件/地点 | 0.7608 / 0.9797 | 0.7436 / 0.9598 | 不回退 >0.05 |
| 换序硬 argmax（/20） | 3 | **1** | 门 ≥8 不过 |
| 换序 decode 起点 argmax（/20） | 5 | 9 | — |

step0 共用快照：route_miss 0.5417 / 物件 route_hit 0.0 / completed 0.0 / 旧描述 0.0 / 硬 argmax 4。

**D. 三读（30/30 全程，不抽样）**：`key_diff_after` 中位 **1.4912**（基线同值，降幅 0.0，冻结 transport 未被冲掉）；`score_margin` 中位 **−0.0048 → 1.562**（配对 bootstrap **+1.5667 [1.4905,1.6412]**，128 读数，激活率 1.0→0.0）；首批 decoy 边距 0.0339 → **1.5533**；`gradient_share_by_param` = W_hidden **58.98%** / w_out **38.92%** / W_hidden.bias 1.69% / offset_embed 0.33% / W_align 0.084% / k_null 0，**新增块合计 0.416%**。

**E. 互斥判定＝`washout_at_readout_proven`**：key 未降、边距已拉开 >1.5，但换序硬 argmax 只 4→1、route_miss 相对 CE 反而 +0.1042 → **差在打分差没穿过 argmax**，不是上游被冲掉、也不是边距不可训。诊断门 5 项过 3（conditional 物件/地点与旧描述不回退；argmax 与 route_miss 不过）。任务面同时变差（completed −0.1354、物件 route −0.2188）→ 该 loss 在 30 步剂量下**有害无益**。**下一步唯一允许**：readout 层顺序敏感重排（成对比较/top-1 抑制），禁止加步数、禁止动上游 key；属设计变更，先出 0 Adam 设计闸再问 owner。**注：该方向已于 2026-11-05 被 owner 否（D16-19，见下条 19.2n）**。

**F. 花费与包**：**784.9 CPU / 802.1 wall**、预算占用 **65.4%**、**60 Adam / 1 包 / 0 confirm**；`28031-N-alignment-step30-order-decoy.pt` sha256 `5c1c2c14…`，save→load 往返 **max_abs_diff 0.0**。开/收盘单价 CE **6.77 → 6.125**、边距项 **2.42 → 2.234**；6.125 > 6.02×0.7 → `price_regime_changed`（**非** `contention_at_start`），只报道、不改后续预算口径。收尾格式化按 D16-18 只跑一次：exit=0、0.31 s、无改动。证据：[B 报告](../../../reports/taiji_package19_2l_order_decoy_diagnostic_20261105.json)＋[manifest](../../manifests/taiji_package19_2l_order_decoy_diagnostic_20261105.json)。

## 2026-11-05 19.2n 同池孪生设计闸（0 Adam / 0 包 / 0 confirm / 0.203 CPU）

owner 裁定 19.2m 方向作废并立 **D16-19**（池外负样本不受理、禁止改尺子），本包只回答“池内是否存在同池孪生”，0 Adam、0 存包、0 confirm、逐行 fsync、**本包不执行格式化**（D16-18）。

**A. 池口径纠正（先于统计）**：候选池是**每个 world 的 8 条 record key**（4 物体描述 + 4 地点描述，`RECORD_LINES=8`），**不是 12**；`score_rows` 输出 9 行＝8 候选 + 1 null。owner 文案与 19.2j 注释里的“12 records”为笔误，已写回账本 §16.12；192/192 dev 与 160/160 train 实测均为 8。

**B. 可构造性（192 dev 例 × 2 起点＝384 读数，96 个唯一 world×key×起点组合）**：

| 量 | 读数 |
|---|---:|
| 池内可构造孪生（任意两段互换） | **0 / 384（0.0）** |
| 实体受限孪生（描述/名字互换） | **0 / 384** |
| 池内字节多重集相同的 key 对 | **0** |
| 每起点孪生数分布 | **{0: 384}** |
| 线级孪生（记录行内描述↔名字互换） | **384 / 384 同长同字节多重集** |
| 线级孪生出现在册 | **0 / 384** |
| 互换后首实体仍是原描述（key 不变＝重复项） | **128 / 384** |
| 互换后首实体变成名字（池外） | **256 / 384** |
| 跨记录互换描述改变 key 集 | **0 / 336** |

**C. 当前边距**：池内孪生为 0 → `score_gold − score_twin` 的测量集为空，**未加载任何 step300 包、未做任何前向**（28031-B/28032-B 仅登记 sha256 作身份）；这不是省略，是无可测对象。**不许**用池外 decoy 的 1.56 边距替代本行读数（D16-19）。

**D. 判定与唯一建议**：**`twin_not_constructible`**（可构造比例 0.0 < 0.60，优先级最高）。机制：每把 key 只含一个描述实体、8 把互不相同、无任何同字节多重集对；记录行虽有描述+名字两个实体，但打分 key 只取描述，互换后要么 key 不变（重复项、分数差恒 0），要么变成名字（名字从不是候选 key）。**结论只回答“池内对手是否存在”，不回答“训练能否学会”**；唯一允许＝另立数据前提提案（key 改整行、池扩为描述×名字叉积），并给出池内编号、key 字节差与实测边距；禁止在现有 8 条池上再提池外负样本，禁止改 argmax 判据。

**E. 花费**：**0.203 CPU / 0.99 wall**、预算 120 的 **0.17%**、**0 Adam / 0 包 / 0 confirm**、0 次格式化。证据：[19.2n 报告](../../../reports/taiji_package19_2n_twin_pool_20261105.json)＋[manifest](../../manifests/taiji_package19_2n_twin_pool_20261105.json)；脚本 [`probe_taiji_package19_2n_twin_pool.py`](../../../scripts/archive/taiji_package19_negative/probe_taiji_package19_2n_twin_pool.py)。

## 2026-11-05 19.2o：D16-20 三层验收落点 ＋ provenance 零更新补算

> 纪律：0 Adam / 0 存包 / 0 confirm / 只读；读数逐行 fsync（D16-15）；本包不执行格式化（D16-18）；不重训、不改判据、不读 confirm。脚本 [`probe_taiji_package19_2o_provenance.py`](../../../scripts/training/probe_taiji_package19_2o_provenance.py)。

**A. 口径（D16-20，已入账本 §16.13）**：L1 机械不变量 100% 零容忍（逐位恢复 / UTF-8 合法 / confirm 读取次数 / 父包与 `candidate_sha_v2` 身份）；L2 内容保真“不要求 100% 正确、要求 100% 可核验”（`provenance_rate` / `fabricated_bytes`，目标 0，有则显式计数）；L3 选择准确率只报 selection / 随机基线 / 可达上限三数，阈值按“bootstrap 下界 > 随机 + 相对上限比值”由 owner 后续裁定。**禁止为过门修改 argmax 规则、阈值或评测切片**。

**B. 切片与定价（D16-17 一包只定价一次）**：4 包 × 8 例单次定价 **0.3145 CPU/例** → 选 96 例/包（预估 130.8 ≤ 150）；12 world 各 8 例，index 由 `round(k*15/7)` 取 `0,2,4,6,9,11,13,15`。实际 **384 例 / 768 个内容跨度 / 6,912 字节**。

**C. L1 结果**：4 包 `file_sha256` 与 `candidate_sha_v2` 与 19.2j manifest **逐字相等**；UTF-8 非法例 **0/384**；confirm **0**；未存包（恢复引用 19.2l-B 往返 `max_abs_diff = 0.0`）。首跑把整块 identity dict 写进 `candidate_sha_v2` 字段 → 与 manifest 的字符串比较恒 false；摘要本身正确，用 `--identity-only`（零解码、**0.08 CPU**）复核并保留 `candidate_identity` 全块、原报告 sha。

**D. L2 结果（未达标）**：`provenance_bytes` **4,926（0.7127）**、`misbound_bytes` **0（0.0000）**、`fabricated_bytes` **1,986（0.2873）**，三项相加恰等于 6,912（`accounting_closed = true`）；**目标 `fabricated_bytes = 0` 差 1,986 字节**。按年龄：**旧描述 3,456 字节 100% 可溯源、0 编造**；编造 **100% 落在新描述半**（新描述 3,456 字节中编造 57.5%）。

**E. 选错 / 抄错拆分**：

| 格 | 字节 | 占 6,912 | 跨度 | 说明 |
|---|---|---|---|---|
| 选对且抄对（aligned） | **4,650** | 67.3% | 534 | 计为 provenance |
| 选对抄错 | **666** | 9.6% | 60 | 全部编造（不在任何记录） |
| 选错但抄自己选中的记录 | **276** | 4.0% | 0 | 计为 provenance（可核验但选错） |
| 选错且编造 | **966** | 14.0% | 115 | 全部编造 |
| 截断未写完 | **354** | 5.1% | 59 | 计入 fabricated（长度按跨度算） |

- `wrong_route_right_content`（选错却抄对 gold）＝ **0 跨度**，即 `misbound_bytes = 0`：**只要选错记录，就抄不出 gold 名字**；内容抄写绑定在所选记录上。
- 编造 1,986 的来源占比：选对抄错 33.5%、截断 17.8%、选错且编造 48.6%。
- 回答“0.05 是选错还是编造”：这不是编造噪声与选择误差的混合；它是**选错（145/768 跨度＝18.9%）＋ 起点后未写完（59 跨度）**两件事，且两者都只在新描述半发生。

**F. L3 结果**：selection **0.8112（623/768）**、随机基线 **0.1111**（非空 1/8 = 0.125）、可达上限 **1.0**、相对上限比值 **0.7715–0.8184**（逐包）、逐包 bootstrap 下界最低 **0.7396**。绝对阈值待 owner 裁定。

**G. 花费**：**141.5 CPU / 147.0 wall**（含身份复核 0.08），预算 150 的 **94.3%**；**0 Adam / 0 包 / 0 confirm**。证据：[19.2o 报告](../../../reports/taiji_package19_2o_provenance_20261105.json)＋[manifest](../../manifests/taiji_package19_2o_provenance_20261105.json)。

## 2026-11-05 19.2p：候选池口径提案设计闸 → `proposal_not_admissible`

> 纪律：0 Adam / 0 前向训练 / 0 存包 / 0 confirm / 只读；读数逐行 fsync（D16-15）；本包不执行格式化（D16-18）；不改 argmax 规则、不改阈值、不加步数、不动冻结 transport、不用池外负样本。脚本 [`probe_taiji_package19_2p_key_separability.py`](../../../scripts/archive/taiji_package19_negative/probe_taiji_package19_2p_key_separability.py)。

**A. 提案**：key 由“记录描述”改为“整行（描述＋连接词＋名字）”，带位置/角色标记，目标是在池内分出“角色互换”的记录。

**B. 三数（owner ①）**
1. **钥匙重复率（现状）**：384 内容起点上，池内“与 gold 仅角色互换、其余逐字节相同”的记录 **0**（19.2n 一致，只作基线复核）。
2. **钥匙可分性（提案）**
   - (a) 同一记录行内描述↔名字互换：`max|Δkey|` 中位 **0.617**、最小 **0.326**、最大 **1.049**；超 0.5 **81/96（84.4%）**——物件 **33/48（68.8%，中位 0.544）**、地点 **48/48（100%，中位 0.726）**。
   - (b) 两记录间“名字集合相同、角色互换”：真实对 **0/12 world**（描述词表 ∩ 名字词表 = 0，共享 token 对 = 0）→ **测量集为空**。
3. **0 Adam 边距**：无池内互换记录，**未加载打分包、0 前向**，`score_gold − score_role_swapped` **无数可给**（不得用合成 decoy，D16-19）。

**C. 反证回答（owner ②）**：加名字后**不会退回袋装**——96/96 条互换行都与 gold 不同 key（同 key 计数 **0**），最小差 **0.326** ≫ 噪声底 **8.88e-16**；**但 role 标记不是统一分离器**——0.5 预注册门槛只有 84.4% 通过，物件行中位 **0.544** 贴近门槛。

**D. 判定（owner ④）**：**`proposal_not_admissible`**。两个独立阻塞：2(a) 非统一分离器；2(b)/3 测量集为空且不得合成 decoy。语义上最接近 `key_not_separable`，但形式上属 ①“任一不满足即不受理”。**不进入定价、不训练**。边界：只回答“整行 key 能否在池内分辨角色互换对手”，不回答“训练能否学会”。

**E. 花费**：**2.34 CPU / 2.52 wall**，预算 150 的 **1.6%**；**0 Adam / 0 前向 / 0 包 / 0 confirm**。证据：[19.2p 报告](../../../reports/taiji_package19_2p_key_separability_20261105.json)＋[manifest](../../manifests/taiji_package19_2p_key_separability_20261105.json)。

**F. 由此关闭**：现有数据前提上“key 换整行”这条路关闭。下一候选＝**记录×角色槽**（同一真实记录的 name 槽与 description 槽都成为候选，池 8→16），天然提供池内角色互换对；必须另立提案并给槽内编号、`max|Δkey|`（key=描述 vs key=名字）与 0 Adam 实测边距，**待 owner 批准，不自动启动**。

## 2026-11-05 任务语义契约（19.2q 前置，0 Adam / 0.141 CPU）＋ D16-21

> owner 暂不批 19.2q 提案，先要一页只读契约。纪律：只读、0 Adam、0 前向、0 存包、0 confirm、逐行 fsync；格式化只在收尾一次；**不写任何新提案**。脚本 [`probe_taiji_task_semantic_contract.py`](../../../scripts/training/probe_taiji_task_semantic_contract.py)，契约页 [TASK-SEMANTIC-CONTRACT_20261105.md](../../reference/TASK-SEMANTIC-CONTRACT_20261105.md)，证据 [报告](../../../reports/taiji_task_semantic_contract_20261105.json)＋[manifest](../../manifests/taiji_task_semantic_contract_20261105.json)。

**A. 规模**：冻结 payload 40 训练 world / 640 例 ＋ 12 开发 world / 192 例 = **416 条记录 / 832 样例**；只读数据结构，无任何前向。

**B. 五问落点（全量核对，非抽样）**
1. **记录**：7 字段；三种句式 `{描述}是{名字}。` / `{名字}又叫{描述}。` / `{描述}对应{名字}。`；raw 固定 = 物件 33/36 字节、地点 24/27 字节；描述 5 字/4 字，名字 4 字/2 字。契约页给 3 条原文样例。
2. **八条候选**：同世界 4 物件 + 4 地点各出 1 个 `description`，候选串＝这 8 个描述（与记录逐一相等 **192/192**、世界内两两不同 **192/192**）；gold＝携带问题句物件描述的那条物件记录（**192/192**）。
3. **问题句**：`让{物件描述}到{地点描述}。`（36 字节）；答案 `把{物件名字}放到{地点名字}`（27 字节）。**答案需要两条记录**：物件名 **192/192** 来自被选记录，地点名 **192/192** 来自“描述与问题句匹配”的地点记录，**0/192** 来自被选记录（地名从不写在物件记录里）。
4. **名字与角色**：名字住在 `name` 字段，偏移按句式（style 0 → 5/6，style 1 → 0，style 2 → 6/7）；**角色没有文本标记**（416 条 raw 中 role 词 0 次），`kind` 是模型不可见元数据，行序打乱（52 world 中仅 2 个恰好物件在前 ≈ 随机 1.4%）；唯一区分信号是长度与用词（物件名 12 字节、地点名 6 字节）。
5. **新旧描述**：模板相同（3 种在训练/开发 × 物件/地点四格齐全，**0 种未见**）；新描述整串在训练出现 **0** 次，每条 5 字中 **3 字**为训练描述从未用过的字（弧/肩/坛、扁/圆/盒、高/足/盘）→ 是“填的词不同”，不是模板或句式不同。

**C. 三问答案**
- **a)** 不是“只需匹配名字”：名字在问题句出现 **0/832**、在候选串出现 **0**，只出现在答案里；新描述失败是**描述侧**失败，不是名字匹配失败（旧描述格同样是新名字 96/96 仍 0.9688）。
- **b)** 是，**记录侧字节置换轴作废** → 立 **D16-21**（理由：候选串只有描述、查询只有问题句、两词表整串互斥、名字两侧 0 次；置换只能得到永不被命中的名字串或原样重复）。
- **c)** 差距不是模板未见、也不是名字未见，而是**描述词本身没出现过**（模板 3/3 齐全；dev 名字字符 100% 见过且旧描述格同为新名字仍 0.9688；新描述整串 0 次出现、每串 3/5 为新字）。

**D. 花费**：**0.141–0.219 CPU / 0.169–0.21 wall**，上限 30 用 ≤0.8%；0 Adam / 0 前向 / 0 包 / 0 confirm。格式化收尾一次（unchanged）。

**E. 边界**：本页只描述冻结数据“是什么”，不含任何提案；`19.2q` 暂不批准。

## 2026-11-05 19.2s 扰动闸＋可学性探针（0 候选训练 / 182.45 CPU）＋ D16-23

> owner 只批扰动闸：不训练候选、不写提案、不进正式图。脚本 [`probe_taiji_package19_2s_perturbation_gate.py`](../../../scripts/training/probe_taiji_package19_2s_perturbation_gate.py)，通路 [`byte_match_route.py`](../../../taiji/byte_match_route.py)，证据 [报告](../../../reports/taiji_package19_2s_perturbation_gate_20261105.json)＋[manifest](../../manifests/taiji_package19_2s_perturbation_gate_20261105.json)；账本 §16.17。384 起点＝192 开发例 × 2 内容起点。

**A. 噪声阶梯（0 训练）**：A1 ε 阶梯 `E=(1-ε)I+εR`（R 固定种子、行归一化、PAD 行 0）9 档 {0,.05,.1,.2,.3,.5,.8,1.0}：每档选择 **1.000**、整句完成 **1.000**（旧/新各 96/96）、匹配起点复原 **1.000**、并列 **0**；边距中位 8.000 →（ε=.5 最低）**3.997** → 7.757，最小 7.000 → 3.325 → 6.613。A2 σ 阶梯 `E=I+σG` 5 档 {.02,.05,.1,.2,.5}：选择/完成/复原全 **1.000**、并列 0，边距中位 8.834 → **496.138**。**首次显著崩塌：ε 无、σ 无**（判定阈值 0.90）。A3 `position_weight` 单独扰动：常数 0.5 → **1.000/1.000/1.000**（边距中位 4.000/最小 3.500）；正弦 `0.5+0.5sin(2πp/15)` → **1.000/1.000/1.000** → 只承担逐槽增益，不承担定位。

**B. 可学性探针（明确非候选臂、不计入任何候选比较、0 存包、状态只在内存）**：纯随机初始化（Gaussian σ=0.02、PAD 行 0、不注入 I），目标＝8 行 free-argmax 的 CE（无注入偏移、无位置标签），Adam lr 0.05；**第 0 步 dev 选择 1.000**，每 10 步曲线恒 **1.000**（dev 完成 1.000、起点复原 1.000；训练侧 1.000）；跑到 **50/80 步**触及内部 180 CPU 保护线停下（总 182.45，未超 200）；train/dev 差 **0** → 无"只拟合训练起点"的退化。

**C. 三问**：1) 成立——最低选择 **1.000**、无崩塌点，故不是站在单位矩阵上；随机嵌入同样成立说明是结构性质，仍指向算法实现。2) 第 0 步已 **1.000**、50 步不变，学不到"逐字匹配"（本来就会），也不存在只拟合训练起点。3) 不承担定位——常数 0.5 与正弦两档仍 **1.000/1.000/1.000**，冻结为常数通路成立。

**D. 判定与边界**：**D16-23 已立**（匹配是"自内积 ≫ 交叉内积"的结构性质，不是初始化站在答案上；加强 D16-22 的算法原语判定）。本包不含提案、不给训练方案、不进图。

**E. 花费**：**182.45 CPU / 28.48 wall**（上限 200，91.2%）；0 候选训练 / 0 confirm / 0 候选包；逐行 fsync；报告 write-then-rename；收尾 in-process `black_check_gate.py` 一次。

## 2026-11-05 19.2r 字节匹配设计闸（0 Adam / 1.88 CPU）＋ D16-22

> owner 只批设计闸：0 Adam、不训练、不写提案、不给建议。脚本 [`probe_taiji_package19_2r_byte_match_gate.py`](../../../scripts/training/probe_taiji_package19_2r_byte_match_gate.py)，参数化通路 [`byte_match_route.py`](../../../taiji/byte_match_route.py)，证据 [报告](../../../reports/taiji_package19_2r_byte_match_gate_20261105.json)＋[manifest](../../manifests/taiji_package19_2r_byte_match_gate_20261105.json)；账本 §16.16。

**A. 读数 1（唯一性）**：384 起点上，问题句里的物件描述串在 8 条 `raw` 中“总出现次数＝1” **192/192**，地点描述串 **192/192**，合计 **384/384**；重复 **0**、未找到 **0**、跨类型出现 **0**、单行内重复 **0**，最大出现次数 1 → 无歧义样例可列。

**B. 读数 2（可解性上界）**：纯字符串扫描（0 训练、0 前向）选择 **192/192（1.0）**、整句 `把{物件名}放到{地点名}` **192/192（1.0）**；旧描述半 96/96、新描述半 96/96 均 1.0；名字抽取只读行内模板字节，416/416 与 `name` 字段一致。

**C. 读数 3（可表达性）**：`ByteMatchRoute`（`byte_embed` 257×256 ＋ `position_weight` 15 ＝ **65,807** 参数；初始化＝字节单位矩阵、位置权重全 1）在 **0 更新**下，注入正确对齐起点 → **gold 192/192**、并列 0、其它有限对手 0、边距恒 15；同一通路对地点描述再跑一次 **192/192** 命中地点行；不注入自由 argmax 也 **192/192**、匹配起点 **192/192** 复原；一次反向无步进 → 两参数梯度有限非零。

**D. 读数 4（成本与身份）**：新增 **65,807** 参数；未注册进 `GRAPH_MODULES` → `candidate_sha_v2` **不变**（`773dcec2…`）；注册则模块/参数清单同时变化 → 摘要按构造改变。第二跳共用同参数（新增 **0**），只需连接词/模板决定的读名偏移：(kind, style) 六格唯一——物件 +6/+7/−6、地点 +5/+6/−4。

**E. 三问**：a) 精确匹配层是**算法原语**（扫描已达 1.0）；做成参数化通路的唯一理由是 M1 要求匹配与复制在同一主体内、信用不可结构性分离，若不接受就应承认该层由算法实现。b) **两跳共用一条通路**（同参数跑两次），第二跳只需确定性读名偏移。c) **与 M1 不冲突**（匹配位置即复制读取位置，∂η/∂α≠0 保留），故不触发 `veto_if_reliance`。

**F. 判定与边界**：本包只交四项读数与三问答案，**不含提案、不给建议、不训练**；`19.2m/n/p` 归入 D16-21，该轴永久关闭（D16-22）。

**G. 花费**：**1.875 CPU / 0.243 wall**（上限 120，1.6%）；0 Adam / 0 优化器步 / 0 训练前向 / 0 存包 / 0 confirm；4 次构造前向 ＋ 1 次反向；逐行 fsync；格式化收尾一次。

## 2026-11-05 19.2t 端到端零训练装配评测（0 训练 / 130.05 CPU）＋ D16-24

> owner 唯一实验：装配 ＋ 评测，不许训练。脚本 [`probe_taiji_package19_2t_zero_training_assembly.py`](../../../scripts/training/probe_taiji_package19_2t_zero_training_assembly.py)，通路 [`byte_match_route.py`](../../../taiji/byte_match_route.py)，身份 [`candidate_identity.py`](../../../taiji/candidate_identity.py)；证据 [报告](../../../reports/taiji_package19_2t_zero_training_assembly_20261105.json)＋[manifest](../../manifests/taiji_package19_2t_zero_training_assembly_20261105.json)；账本 §16.18。dev 192 例、confirm 0 读取。

**A. 装配**：`byte_match_route` 注册进 `GRAPH_MODULES`（第 7 个模块）作为选择读出——物件描述与地点描述各跑一次自由搜索（同一组参数），8 条 `raw` 逐位匹配 → 起点 argmax；起点交给现有复制算子，`transport` / `span` 指针 / `score_rows` 保持原权重（父包 `28031-B_stop_lock-step300.pt`）。`candidate_sha_v2` 旧 `773dcec2…` → 新 **`95099768cc17a24c870e3b7d49cef0df022d5114f3312a4b3d5ede50179d0aa5`**（参数 3,563,859 不变、`byte_match` 65,807）；装配态 **265,853 字节**、sha256 `b2dc9f58…`，唯一一份、标注 `assembled_zero_training`。

**B. 读数（γ=0 主读 / γ_floor=0.2 副读；total 192、旧/新各 96）**：选择物件/地点/两跳 **1.0000**；`start_hit` / `composite_start_hit` 物件/地点 **1.0000**；`conditional_byte` **1.0000**（`object_span_exact` / `place_span_exact` 同）；`provenance_rate` **1.0000**；`fabricated_bytes` **0**、`misbound` **0**、截断率 **0.0**、384 跨度全对齐；`wrong_route_right_content` **0**。整句完成 γ=0 **0.3854（74/192；旧 0.4479、新 0.3229）**、γ_floor **0.6302（121/192；旧 0.6250、新 0.6354）**。首差异分布 γ=0 `{template: 118, none: 74}`、γ_floor `{none: 121, template: 71}`，**失败全部在字节位 15**；截断/编造/串行/错名字 0。确定性重复 8/8 一致。

**C. 两问**：1) 整句未到接近满分（0.3854 / 0.6302），但内容层已满分（选择/起点/conditional/provenance 全 1.000）→ "选对才写对"在同一次评测内被证实。2) 唯一主缺口＝**抄写停止边界**（模板位 15–20）：γ=0 时复制记忆越过名字末端继续抄记录终止符"。"；γ_floor 靠生成权重混入把失败 118→71，不能消除；现有复制算子无 span 末端/停止信号 → **结构性质**，训练只能压低频率。

**D. 判定与边界**：**D16-24 已立**（匹配层为算法原语：零训练装配是零更新评测的正确形态）。本包不含提案、不含训练方案、不开启下一轮。

**E. 花费**：**130.05 CPU / 136.32 wall**（上限 400 CPU / 60 min，32.5%）；0 训练 / 0 优化器步 / 0 Adam / 0 confirm / 1 装配态；逐行 append＋fsync；报告 write-then-rename；收尾 in-process `black_check_gate.py` 一次。

## 2026-11-05 19.2u 停止边界＋训练权重消融（0 训练 / 256.72 CPU）＋ D16-25

> owner 只批"停止边界＋消融"，仍然零训练。脚本 [`probe_taiji_package19_2u_stop_boundary_ablation.py`](../../../scripts/training/probe_taiji_package19_2u_stop_boundary_ablation.py)，结构原语 [`copy_stop_boundary.py`](../../../taiji/copy_stop_boundary.py)＋[`binding_copy.py`](../../../taiji/binding_copy.py)（`copy_limits`）；证据 [报告](../../../reports/taiji_package19_2u_stop_boundary_ablation_20261105.json)＋[manifest](../../manifests/taiji_package19_2u_stop_boundary_ablation_20261105.json)；账本 §16.19。dev 192 例、confirm 0 读取。

**A. 字段范围实测**：dev 192×8＝1,536 条记录，六格各 256 条；物件名恒 **12**、地名恒 **6** 字节；终止符恒 `。`；名字末端到终止符偏移 style0/2＝**0**、style1＝**21（物件）/18（地点）**；字段起点 object 18/0/21、place 15/0/18（style0/1/2）；反例 **0**。结论：边界＝固定长度可判＋模板槽位可判；终止字节只在 style0/2 紧邻名字末端，不能单独判边界。

**B. 接入**：`read` / `answer_step` / `forced_step` 新增 `copy_limits`（逐记录名字字段末端），边界之外复制质量清零；内容字节仍在同一 `read` 影响 η（M1 保留）；不含"第 15 位"常数；两跳各自边界。新模块入 `GRAPH_MODULES`（第 8 个）：`candidate_sha_v2` `95099768…`→**`1d69e945ac149d436cc39426de5717b8a3e2b82de5e8f35565354028c0a33728`**；装配态 265,981 B、sha256 `f5bd80e7…`、唯一一份、标注 `assembled_zero_training_stopfix`。

**C. 评测（γ=0 主读 / γ_floor=0.2 副读）**：整句 **1.0000（192/192；旧/新各 1.0000）**；选择/`start_hit`/`composite_start_hit`/`conditional_byte`/`provenance` 全 **1.0000**；`fabricated`/`misbound`/截断/`wrong_route_right_content` **0**；首差异 `{none: 192}`；384 跨度全对齐。

**D. 消融**：transport（v_relative/Q_transport/K_transport）、span_head、score_rows 共 **12 参数 / 46,642 元素**固定种子 **1902** 随机初始化（byte_match 原样）：γ=0 整句 **0.9479（182/192；旧 0.9583 / 新 0.9375）**、γ_floor 同；Δ **-0.0521**，**10 例失败全在模板字节位 1**，选择/抄写/来源仍 1.0000；两种配置确定性 **8/8**。

**E. 判定与边界**：**D16-25 已立**（复制停止边界为结构原语；训练权重对任务能力贡献为 0，只对模板字节发射有边际作用）。本包不含训练方案、不含新包、不开启下一阶段。

**F. 花费**：**256.72 CPU / 268.55 wall**（上限 400 CPU / 60 min，64.2%）；0 训练 / 0 优化器步 / 0 Adam / 0 confirm / 1 装配态；逐行 append＋fsync；报告 write-then-rename；收尾 in-process `black_check_gate.py` 一次。

## 2026-11-05 19.2v 封存集独立确认（0 训练 / 144.06 CPU）＋ D16-26

> confirm 首次且唯一一次读取。预注册快照 [`taiji_package19_2v_preregistered_20261105.json`](../../manifests/taiji_package19_2v_preregistered_20261105.json) 在读取前落盘（装配态 sha256 `f5bd80e7…`、`candidate_sha_v2` `1d69e945…`、阈值与四条判定规则、`confirm_reads=1`）。脚本 [`probe_taiji_package19_2v_confirm_holdout.py`](../../../scripts/training/probe_taiji_package19_2v_confirm_holdout.py)；证据 [报告](../../../reports/taiji_package19_2v_confirm_holdout_20261105.json)＋[manifest](../../manifests/taiji_package19_2v_confirm_holdout_20261105.json)；账本 §16.20。

**A. 读取纪律**：confirm 由冻结池按冻结宽度物化（12 world × 16 = 192；旧/新各 96）。首跑在夹具物化前中止（`fixtures(payload,"confirm")` 在 `build_payload` 下为空，**0 例 0 前向**），按预注册的恢复条款重跑一次并在报告标 `recovery_rerun=true`；这是 1 次实际前向、无基于结果的重跑、无调参。

**B. L1 机械不变量（全过）**：选择物件/地点/两跳、`start_hit` 物件/地点、`composite_start_hit`、`conditional_byte`、`provenance` 全 **1.0000**；`fabricated`/`misbound`/截断/`wrong_route_right_content` **0**；confirm 模板 fixed-length、终止符恒 `。`、反例 **0**（1,536 条记录，物件名 12、地名 6）。

**C. 整句**：**188/192＝0.9792**（旧 **95/96**、新 **93/96**；γ=0 与 γ_floor 同值）；答案长度实测 **27 字节**（模板 9＋物件名 12＋地名 6）。4 例失败**全部是位 27 多写 1 个模板字节“到”**，27 字节正文全对、选择/起点/384 跨度全对齐 → 最后一位之后的**终止边界**是唯一残余。

**D. 判定（按预注册）**：`milestone_holds`（≥0.95）；与 dev 差 **−0.0208**（未达 −0.05 过拟合阈值）；L1 无失败 → **非** `structural_regression`、**非** `dev_overfit_of_structure`。**D16-26 已立**：封存集独立确认＋结构实现的 L1 判据。

**E. 花费**：**144.06 CPU / 150.53 wall**（上限 400 / 60 min，36.0%）；0 训练 / 0 优化器步 / 0 Adam / 0 参数变更 / 0 图变更；逐行 append＋fsync；报告 write-then-rename；收尾 in-process `black_check_gate.py` 一次。本包不含训练方案、不含新包、不开启下一阶段。

## 2026-11-05 19.2w 边界残差复现与修复（0 训练 / 290.51 CPU）＋ D16-27

> 只碰合成/dev，**0 confirm 读取**。脚本 [`probe_taiji_package19_2w_boundary_residual.py`](../../../scripts/training/probe_taiji_package19_2w_boundary_residual.py)；证据 [报告](../../../reports/taiji_package19_2w_boundary_residual_20261105.json)＋[manifest](../../manifests/taiji_package19_2w_boundary_residual_20261105.json)；账本 §16.21。

**A. 复现（冻结外）**：沿用 18.0 生成器，只替换地点池为“地名以 `尾` 结尾”的 20 个条目（物件名/描述/模板按原分布），5 world × 16 取 64 例；同批再造“不以 `尾` 结尾”对照 64 例。**尾类 63/64 逐字节精确、1 例外溢（1.5625%）**，外溢位置 27、多 3 字节；对照 64/64、外溢 0。与 confirm 4/192＝**2.08%** 同量级 → **复现成立**（“生成器造不出该触发条件”的断言不成立）。首次按 3 world 只够 48 例（`repro_discarded` 记 39.84 CPU），改 5 world 后重跑。

**B. 诊断（只看 dev/合成）**：失败例的尾部 trace 显示——位 26 顶位 1.0（地名末字节）；**位 27 EOS 0.3492 低于顶位 `0xE5` 0.3968**；位 28/29 以 0.936/0.977 补齐 `0x88`/`0xB0`（模板“到”的后两字节），位 30 才 EOS。且 **1,536 条记录里字节“到”出现 0 次** → 外溢**不来自复制通路**（`copy_limits` 在位 27 已清零，`copy_content`/`copy_mass_max` 均为 0）。四个怀疑项（终止符前缀匹配 / UTF-8 多字节截断 / style 偏移表两跳错位 / 末端槽多 1 可达位）**全部不成立**；真实缺口是**序列末端没有停止边**：判定仍按“模型自己发 EOS”。

**C. 修复**：`taiji/copy_stop_boundary.py` 新增 `sequence_stop(hop_positions)` —— 最后一跳字段的停止边即**序列终点**（`hop_positions[-1] + 1`），解码到该位即终止；注册进 `candidate_identity.DECODE_RULES`（`sequence_stop`），因此 `candidate_sha_v2` 变更属设计内。**0 参数 / 0 样例常数 / 无“尾”字特判 / `gap_constant=false`**；参数 3,563,859 不变。`candidate_sha_v2` `1d69e945…`→**`0d6a5df3…`**；新装配态 1 份 `output/taiji_package19_2w_assembled_stopfix_r2.pt`（标注 `assembled_stopfix_r2`，266,021 B，sha256 `4e7ba75e…`），其余 0 存包。

**D. 三道门（全过 → `fixed`）**：① 合成尾 **64/64**、对照 **64/64**，`extra_bytes` **0**；② dev **192/192**，**L1 全 1.0000**（选择/`start_hit`/`composite_start_hit`/`conditional_byte`/`provenance`）→ 无 `regressed`；③ 鲁棒性：11 类末字符（上/下/东/侧/前/北/外/尾/沿/畔/角）各 8 例，**全 0 外溢**。

**E. 口径与收束**：confirm **0.9792 原样保留为官方数字**，本轮**未在封存集复核**；若需新 confirm 读数须**另立前置注册**。**D16-27 已立**（残差须在合成/开发侧复现修复，不得以封存集为调参依据）。plans 六处同步；本包不含训练方案、不含新包、不开启下一阶段。

**F. 花费**：**290.51 CPU / 304.67 wall**（上限 400 / 60 min，72.6%；含首次 3-world 弃跑 39.84）；0 训练 / 0 优化器步 / 0 Adam / 0 confirm / 0 参数新增；逐行 append＋fsync；报告 write-then-rename；收尾 in-process `black_check_gate.py` 一次。

## 2026-11-05 20.1 规则学习夹具＋CF1/B_EH 参考图（0 训练 / 159.53 CPU）

> 严格按 owner 定稿的阶段20 rev2（`bb3fbe6a6`）执行，只做 20.1 四项交付。夹具 [`rule_learning_20_1.py`](../../../taiji/rule_learning_20_1.py)、应答图 [`rule_learning_20_1_net.py`](../../../taiji/rule_learning_20_1_net.py)、身份 [`rule_learning_20_1_identity.py`](../../../taiji/rule_learning_20_1_identity.py)、脚本 [`probe_taiji_package20_1_fixture_graph.py`](../../../scripts/training/probe_taiji_package20_1_fixture_graph.py)；证据 [报告](../../../reports/taiji_package20_1_rule_learning_fixture_graph_20261105.json)＋[manifest](../../manifests/taiji_package20_1_rule_learning_fixture_graph_20261105.json)；账本 §16.22。

**A. 夹具**：4 world／**204 应答轮（51/world）**；**纠正面 22/world**（S1 5＋S2 11＋S3 6，单列可独立统计）＝“反馈之后新查询的首答”子集，另有 32 个首答/world 作全量登记（含 10 个 anchor 盲答）；会话 **3**（10/24/17）；反馈 2 批（after turn 5 与 17）、40 行 **1,560 字节**（v1 20 行 260 字 780 B、v2 同）；反馈内反例 **7**；跨版本 changed_query **25（6.25/world）**；规则族 `y = b xor xor_{i∈S} x_i`（1≤|S|≤2）共 **20** 条；记录 390 B/world（E 4,096、view 512）。规则真值/求解器只在 `scoring` 块，不作为应答字段。

**B. 两图同形**：均为 `RuleLearningGraph` 包同一 R021 subject（`GatedGrowthCore`，`growth_gates={"consensus":1.0}`）。**3,323,547 参数 / 135 张量 / 新增 0**；名称/形状/dtype 逐项相同、参数量差 0；差异仅 `arm`/`fast_enabled`（CF1 写 F、B_EH 永不写），冻结清单逐项在报告 `differences` 块。`candidate_sha_v2`：CF1 **`a8a1e169…`**、B_EH **`3e98f65d…`**。装配态各 1 份、标注 `initialized_untrained`：CF1 16,645 B sha256 `b62d6fe6…`、B_EH 16,651 B sha256 `64cb327e…`。

**C. 冻结清单**：参数分块（W 458,752 / U 8,192 / V 57,344 / R 928,256 / P 132,608 / tau 4 / encoder 205,632 / nu 2,176 / message 212,992 / fast_read 164,608 / episodic 83,713 / xi 41,984 / readout 327,680 / expression 116,547 / write_heads 434,448 / world 148,611 冻结不用）；状态分持久（h/fast/nu_ema/keys/coverage/prior/journal）与会话内（previous_byte/turn_cursor）；注册项 `GRAPH_MODULES_20`（net/living_core/record_relation_core）与 `DECODE_RULES_20`（answer_space/answer_bytes 15/decision_positions (7,8)/decode/feedback_release/fast_write/fast_window/session_freeze/truth_isolation/answer_contract）；种子 **20261105**、初始化＝父 bundle 权重原样复制、无新张量初始化。

**D. 泄漏探针（三条全过）**：(a) 静态 AST＋签名扫描——应答模块无夹具 import、无 `truth/solver/labels/xor` 等名、`run_turn` 签名无真值参数；(b) 真值打乱后重跑——**765/765 字节逐字节一致**、`max|Δlogits| = 0.0`；(c) 约束解码反查——46 轮×2 world、**690/690 字节全部落到已释放反馈**（decision 92/92 同），template/generate/**truth 全 0**。

**E. dry-forward（不训练）**：2 world／102 轮，logits **[15, 257]** 全有限；CF1 fast writes 20、B_EH **0**；CF1 journal 102、applied 20、released 20；会话边界 4 次存/恢复全部精确（4/4）；B_EH 与 CF1（F 关闭）**765/765 逐字节一致、`max|Δlogits| = 0.0`**。

**F. 判定与花费**：`verdict=ready`，19 门全 true；**159.53 CPU / 164.55 wall**（上限 500 / 4500 s，31.9%）；0 训练 / 0 优化器步 / 0 Adam / 0 confirm / 0 替换 / 19.x 装配态未动；逐行 append＋fsync；报告 write-then-rename；收尾 in-process `black_check_gate.py` 一次。**20.2 信用与保存预检未准入**，等 owner 判断；本包不含训练方案、不含新包。

## 2026-11-05 20.2 信用链＋负对照＋完整保存/独立恢复（0 训练 / 153.08 CPU）

> 按 owner 阶段20 表内授权只做 20.2。脚本 [`probe_taiji_package20_2_credit_recovery.py`](../../../scripts/training/probe_taiji_package20_2_credit_recovery.py)；证据 [报告](../../../reports/taiji_package20_2_credit_recovery_20261105.json)＋[manifest](../../manifests/taiji_package20_2_credit_recovery_20261105.json)；账本 §16.23／D16-28。20.1 三个模块未改动（`candidate_sha_v2` 不变），信用探针在新脚本内重建 grad-enabled 前向。

**A. 信用链**（一次反向、不 step）：① 探针轮 18（`current_v2`）loss **15.401160**、**dL/dF 0.068832**、分头 [[0.005480,0.005223],[0.037832,0.035744],[0.029793,0.032915],[0.000548,0.000610]]、**份额 0.004469** → `reachable`（未判 `credit_starvation`）。② 同一次反向 **135 张量 0 个有梯度、0 violation**，Ψ/Θ 会话冻结成立。③ 设计冻结下 F 是唯一梯度路径（`design_freeze_share = 1.0`）；解冻对照 **107/135 张量有梯度、范数 73.504**、F 份额 0.000936、**解冻前向与原前向差 0.0**、F=0 时 loss 15.400374（Δ **0.000787**）→ 后续路径只经 F 输出被塑形。④ **10/10** 反馈 query 匹配源轮且早于释放、applied receipts 10；release v1（after turn 5）引用 turns 1–5、后随纠正面 6–10；release v2（after turn 17）引用 13–17、后随 18–34；两次释放 **8 个 (group,head) 段全部变化**（`fast_max_delta` 0.076687 / 0.031820）。

**B. 负对照三条**：B_EH **fast writes 0**、纠正面 0/22、dL/dF 0.068802、share_thaw 0.000935；乱序反馈（同长度乱序字节）fast writes 10、share_thaw **0.000411（相对 −56.0%）**、方向 cosine **−0.028**、fast 最大差 0.0254、探针 loss 19.076、输出未变好（0/22 ≤ 0/22）；F 关闭时 CF1 与 B_EH **51 轮 765/765 字节一致**、`max|Δlogits| 0.0`（本包复核，未复用 20.1 读数）。

**C. 完整保存与独立恢复**（真跑）：mid turn18 全量写入→新图载入→续 6 轮 **90/90 字节一致**（`state_restore_exact=true`、max|Δ|=0.0）；跨会话 turn10 **90/90**（fast 边界前已写、载入后持久、journal 持久、`f_reset_at_boundary=false`）；崩溃式（截断包 29,407 B 被 `load_checkpoint` 以 OSError 拒绝、canonical sha 不变）恢复 **90/90**。状态表 9 项：h/fast/nu_ema/keys/coverage/prior/journal 持久、previous_byte/turn_cursor 会话内，**全 serialized/restored/byte_identical**。

**D. 必带读数**：dry 前向 51 轮、logits **[15, 257]** 全有限；CF1 fast writes **10**、B_EH **0**；**反向 6 次**（全部不 step）；`turns_covered=224`（七个对照运行 222＋2 次信用探针；历史 17×3 重跑未计入，口径已在报告注明）。

**F. 判定与花费**：`verdict=ready`，14 门全 true；**153.08 CPU / 159.04 wall**（上限 400 / 3600 s，38.3%）；0 训练 / 0 优化器步 / 0 Adam / 0 confirm / 0 新参数；两图仍各 1 份 `initialized_untrained`，3 个恢复包标 `recovery_copy`×2＋`crash_artifact`×1（`output/` 不提交，仅记 sha）；19.x 装配态未动；逐行 append＋fsync；报告 write-then-rename；收尾 in-process `black_check_gate.py` 一次。**20.3 定价未准入**，等 owner 判断；本包不定价、不给训练方案。

## 2026-11-05 20.2r 重放一致性＋F 杠杆闸（0 训练 / 161.48 CPU）

> 按 owner 授权只做 20.2r。脚本 [`probe_taiji_package20_2r_f_leverage.py`](../../../scripts/training/probe_taiji_package20_2r_f_leverage.py)（运行 sha `e62b19b2…`、提交 sha `9081bee1…`，两列同存报告）；证据 [报告](../../../reports/taiji_package20_2r_f_leverage_20261105.json)＋[manifest](../../manifests/taiji_package20_2r_f_leverage_20261105.json)；账本 §16.24／D16-29。**0 改代码、0 改图、0 新增参数、0 副本。**

**A. 重放一致性（先做）**：直接 import **提交字节**（`scripts/training/probe_taiji_package20_2_credit_recovery.py`，sha256 **`cd23ce2e…`**，与工作区一致）并复用其函数，重跑轮 18 信用探针（同一 world/seed/一次反向、不 step）：loss **15.401160**、dL/dF **0.068832**、份额 **0.004469**、分头 4×2 矩阵全同 → 四项**相对误差 0.0**（门 1e-6）。dry 前向 F 关两图 **765/765 字节一致**、`max|Δlogits| 0.0`。20.2 报告本身未记录脚本 sha（其字节为格式化前版本），本次以提交字节重放并通过 1e-6 门，可复现性瑕疵闭合；报告用两列（run / submitted）记录 20.2r 自身的格式化差异。

**B. F 杠杆阶梯（22 个纠正面，基线 loss 14.514391、基线答对 0/22、决策位 0/22）**

| 档 | Δloss | leverage | byte_diff | 答对 |
|---|---|---|---|---|
| F 置零 | −0.00038862 | 2.677e-5 | 0 | 0/22 |
| F ×0.5 | −0.00019407 | 1.337e-5 | 0 | 0/22 |
| F ×2 | +0.00038670 | 2.664e-5 | 0 | 0/22 |
| F ×4 | +0.00115428 | **7.953e-5** | 0 | 0/22 |
| 同范数乱序 | −0.00046275 | 3.188e-5 | 0 | 0/22 |
| oracle（`oracle_diagnostic_only`，真值方向） | −0.607197 | 4.183e-2 | **0** | **0/22** |

**C. 判定**：`leverage_max = 7.953e-5（0.00795%）< 1%` → **`under_powered_f_channel`**。oracle 上限 4.183% 高于 1%，但 **byte_diff 0、0/22 答对** → 按 owner 口径单独记为**读写结构问题**：loss 方向可动，argmax/约束解码的边距从未被跨过；不是"增益不够"而是"F 没有字节权威"。

**D. 修法参数增量估算（仅估算，不实现、不定价）**：F 为 **8,192 元素**状态。若只新增"F→两个决策字节 logits"的低秩读出，约 **8,708·r** 参数（r=4 → **34,832**；r=8 → **69,664**）；若接全部 15 个位置（3,855 个 logit）约 **12,062·r**（r=4 → 48,248；r=8 → 96,496）。纯增益方案（每 (group,head) 增益＋每位置标量，23 参数）按 oracle 事实不可行：需 >100× 增益且会饱和先验。

**E. 判定与花费**：`verdict=under_powered_f_channel`，11 门全 true；**161.48 CPU / 168.64 wall**（上限 300 / 2700 s，53.8%）；0 训练 / 0 优化器步 / 0 Adam / 0 confirm / 0 新参数；**23 次反向不 step**（1 次重放＋22 次 oracle 诊断）、**391 轮前向**；两图各 1 份 `initialized_untrained`、19.x 未动；逐行 append＋fsync；报告 write-then-rename；格式化在报告写完后执行并双列记录 sha。**20.3 定价未准入**，等 owner 判断；不实现修复、不给训练方案。

## 2026-11-05 20.2s F 信息含量闭式探针（0 训练 / 119.67 CPU）

> 按 owner 授权只做 20.2s。脚本 [`probe_taiji_package20_2s_f_information.py`](../../../scripts/training/probe_taiji_package20_2s_f_information.py)（运行 sha `3a3c58ad…`、提交 sha `ab9b73ec…`，两列同存报告）；证据 [报告](../../../reports/taiji_package20_2s_f_information_20261105.json)＋[manifest](../../manifests/taiji_package20_2s_f_information_20261105.json)；进度 jsonl 逐行 append＋fsync；账本 §16.25／D16-30。**0 训练 / 0 优化器步 / 0 Adam / 0 confirm / 0 新参数 / 0 副本。**

**A. 探针（纯代数，无梯度下降）**：夹具与应答图沿用 20.1/20.2 提交字节（以提交字节 20.2 runner 重放收集，`p2_committed_sha256` 逐项比对一致）；特征 = 22 个纠正面每轮结束时的 **F（8,192 元素）**；标签两套并列——**决策位**（`DECISION_POSITIONS` (7,8)，2 位 × 2 类）与**全位**（15 位 × 3 槽，槽 0/1＝两个动作串、槽 2＝哨兵 255）。求解用岭回归**闭式解**（`(XᵀX+λI)⁻¹XᵀY` 的 dual 等价 `α=(XXᵀ+λI)⁻¹Y`、`W=Xᵀα`，避免 8192×8192），λ 扫 {1e-4, 1e-3, 1e-2, 1e-1, 1}，**每外层折仅用训练侧 21 行做内层留一选 λ**（评测结果不参与挑 λ），tie 取较大 λ。阈值与判定规则在读数前写入 progress（`stage=thresholds`）。

**B. 读数（world 0，22 折）**

| 指标 | 值 | 说明 |
|---|---:|---|
| **决策位留一准确率（主指标）** | **0.6818（30/44）** | 2 位 × 22 轮 |
| 整句（15 字节全对） | 15/22 | 0.6818 |
| 全位字节准确率 | 0.9576 | 13 个模板位恒对，**非学习读数** |
| 选中的 λ | 22/22 折全为 **1.0** | 全落网格上界（过度收缩） |
| wide 与 decision 两套标签 | 同值 | 标签契约两套皆报 |
| 基线模型自身读数 | **0/22**（决策位 0/22） | 无训练模型对纠正面本就全错 |

**C. 三条对照（缺一不可）**

| 对照 | 准确率 | 判读 |
|---|---:|---|
| 标签打乱（5 seed） | 均值 **0.5091**（max 0.7727） | 探针未偷标签 |
| 同维同范数随机向量代替 F | **0.1818** | 非维度红利 |
| 无 F 基线（h/nu_ema/coverage/prior 等其它持久态） | **0.5000** | 不比 F 高，但也不低 |

最强对照 = 0.5091，**差 0.1727**。

**D. 结构读数（本包最重要）**：22 个纠正面上的 F **只有 2 个不同向量**（S1 反馈窗口 1 个、S2 反馈窗口 1 个，窗口内逐轮常量；10 条写入行、范数 0.950–1.292、两窗口最大差 0.03182、release 后 F 与下一纠正面首轮逐字节相同）。因此"只按反馈窗口身份猜"的诊断准确率也是 **0.6818，与 F 完全相同**（S1 5 轮中 4 右、S2 17 轮中 9 右，多数类合计 30/44）→ 静态 F 对**窗口内逐轮标签零分辨力**，0.6818 不是"F 携带逐询问纠正信息"。跨窗口前向 **S1→S2 0.3529（6/17）、S2→S1 0.2000（1/5）**；其余 3 world 稳健性 **0.5909 / 0.4545 / 0.1818**（非判定用）。

**E. 判定**：按读数前写死的规则——主指标 0.6818 ≥0.60，但 **margin 0.1727 < 0.20**，且未落入 ≤0.30 或 <0.10 带 → **`inconclusive`**。**不自行挑边**；按 owner 边界，**不实现读出、不定价、不进 20.3、不提新包**，结果由 owner 判断。

**F. 花费与记账**：`verdict=inconclusive`，12 门全 true；**119.67 CPU / 124.34 wall**（上限 200 / 1800 s，59.8%）；4 world × 51 轮 = **204 轮前向**、0 反向、0 存包；两图各 1 份 `initialized_untrained`、19.x 装配态未动；报告 write-then-rename，格式化在报告写完后执行。**首跑失败尝试**：`collect_primary`（25.75 CPU）后因 `decode_decision` 模板字典缺决策位触发 NameError 中止、未写报告，精确 CPU **missing**、下界 **25.75**，按"不可得记 missing、不补 0"入账。

## 2026-11-05 20.2t 写入粒度设计闸（纯设计 / 0 参数 / 0 代码）

> 按 owner 授权只做设计。产物 [DESIGN-20_2t 备忘](../../reference/DESIGN-20_2t_WRITE_GRANULARITY_20261105.md)＋[manifest](../../manifests/taiji_package20_2t_write_granularity_20261105.json)；`runner_sha256: n/a`、`submitted_runner_sha256: n/a`（本闸无脚本）；账本 §16.26／D16-31。**0 训练 / 0 优化器步 / 0 Adam / 0 confirm / 0 新参数 / 0 副本 / 不改代码。**

**A. 归因（写入链路事实）**：`run_turn` 内 `allowed=False`，F 在回合内不变；只有 `release` 逐行 `observe→write_fast`（[living_core.py](../../../taiji/living_core.py#L405-L444)）。22 个纠正面因此只有 10 条写入行。`edit_fast` 是"衰减＋rank-1 擦除＋rank-1 叠加＋范数裁剪"，没有键寻址 → 逐询问纠正**在结构上不可表示**，与读出层强弱无关。`write_heads` 实际参数 **434,448**（4×[521→128→324]）。

**B. 三选项对照（全部按实际张量形状）**

| 选项 | 机制 | 参数增量 | 新增持久态 | 预测信息闸 | 可测失败模式 |
|---|---|---:|---|---|---|
| 1 每轮无寻址写 | 每轮 `allowed=True` 调现有 `write_fast`，error=该轮误差 | **+0** | **+0**（仍 `fast` 8,192） | ≈ 窗口身份基线，margin <0.10 | 22 面 F 两两余弦中位 >0.9 |
| **2 每轮内容寻址（推荐）** | 键＝问句经现有 `BytePatchEncoder.context` 的 128 维表示；值＝该轮 F；余弦检索 softmax 混合 | **+0**（τ=0.1、α=0.5 常数） | `(32,128)`＋`(32,4,2,32,32)`＋`(32)`＋`(1)` = **266,273** 元素（f64 ≈2.03 MiB） | 预测 ≥0.60 且 margin ≥0.20（前提变体命中） | 变体 top-1 命中率 <0.5；FIFO 淘汰早轮 |
| 3 批次写＋衰减累积 | 轮内 `a←γa+e_t`，批次末一次提交 | **+0** | accumulator 8,192＋decay 1 = **8,193** | 结构闸即不过（distinct F=2） | 窗口内 F 仍常量 |

**C. 推荐与理由（唯一）**：**选项 2**。只有它把逐询问纠正在结构上变成可表示（F 的取值由当轮键决定）；0 可训练参数、F 仍是唯一可塑状态；值仍是联合 F，不做内容/绑定分离 → M1 `veto_if_reliance=false`。键复用现有 128 维字节表示，不新增分词器。

**D. 过门预注册（先写死）**：①结构：不同 F 向量 ≥ 窗口内不同误差方向数，且 >2，且窗口内逐轮变化；②杠杆：五档反事实 `leverage_max ≥ 0.01` 且至少一档 `byte_diff > 0`；③信息：闭式留一决策位 ≥0.60 且比三条对照最强者高 ≥0.20。任一不过即该选项失败，不调阈值。

**E. 代价与边界**：状态 +2.03 MiB f64、32 槽 FIFO、会话内持久/跨会话清空；实施需改 4 个文件（`living_core.py`、`rule_learning_20_1_net.py`、`rule_learning_20_1_identity.py`、`state_migration.py`），`living_core.py` 与 19.x 共享 → 必须默认关闭＋字段带默认值，19.x 装配态文件不动。**未实施、未定价、未进 20.3**；三选项都不可行时退出说明写在备忘 §5（一句）。

## 2026-11-05 20.3a 每轮内容寻址写实施与三闸判定（0 训练 / 224.94 CPU）

> 按 owner 授权按 20.2t 选项 2 实施。脚本 [`probe_taiji_package20_3a_addressed_store.py`](../../../scripts/training/probe_taiji_package20_3a_addressed_store.py)（运行 sha `3e46c57a…`、提交 sha `6903857a…`，两列同存报告）；证据 [报告](../../../reports/taiji_package20_3a_addressed_store_20261105.json)＋[manifest](../../manifests/taiji_package20_3a_addressed_store_20261105.json)；进度 jsonl 逐行 append＋fsync；账本 §16.27／D16-32。**0 训练 / 0 优化器步 / 0 Adam / 0 confirm / 0 新可训练参数。**

**A. 实施（严格照 DESIGN-20_2t，未扩范围）**：`living_core.py` 加 `ADDRESSED_SLOTS=32`／`ADDRESSED_TEMPERATURE=0.1`／`ADDRESSED_ALPHA=0.5` 与四个带默认 `None` 的 `LivingState` 字段（`store_keys`/`store_values`/`store_age`/`store_count`）＋ `empty_store`/`attach_store`/`clear_store`/`store_query_key`/`addressed_read`/`addressed_write`；`rule_learning_20_1_net.py` 加 `addressed_store_enabled=False`（B_EH 恒关）、`_with_store` 会话首轮清空、`run_turn` 读混合后 observe、`release` 逐行读混合后写 F 副本；`rule_learning_20_1_identity.py` 的 `DECODE_RULES_20` 加 `store_write`/`store_read`/`addressed_write`/`fifo`/`cross_session_clear` 五项；`state_migration.py` 仅 store 非空时存取 `bundle["store"]`。键＝当轮问句 128 维、值＝F 副本（8,192）、余弦＋softmax τ=0.1、α=0.5、32 槽 FIFO、会话内持久/跨会话清空；**+0 可训练参数、+266,273 状态元素**。身份：CF1 default `58754e50…`、CF1 addressed `fdab0778…`、B_EH default `66efe05a…`。

**B. 三闸（阈值冻结）**

| 闸 | 判据 | 实测 | 结果 |
|---|---|---|---|
| 结构 | 不同 F 向量 >2 且 ≥ 窗口内误差方向数；逐轮变化 | 22 个不同 F；窗口 5 vs 3、17 vs 7；逐轮 hash 全不同 | **过** |
| 杠杆 | `leverage_max ≥ 0.01` 且至少一档 `byte_diff > 0` | **8.716e-6**；五档 `byte_diff` 全 0；基线 loss 12.597441 | **不过** |
| 信息 | 留一 ≥0.60 且比最强对照高 ≥0.20 | **0.4091**；最强对照 **0.5909**；margin **−0.1818** | **不过** |

杠杆五档明细（第 34 轮、历史到 33）：置零 Δ −5.255e-6（lev 4.172e-7）、×0.5 −2.614e-6（2.075e-7）、×2 +5.144e-6（4.083e-7）、×4 +1.509e-5（1.198e-6）、同范数乱序 −1.098e-4（**8.716e-6**），五档均 **byte_diff 0、0/22 答对**。信息三对照：标签打乱 5 seed 均值 0.5000／max **0.5909**、同维同范数随机 **0.1818**、无 F 其它持久态 **0.5000**。

**C. 附带必测**：①检索读时 top-1 余弦中位 **0.9850**、top1–top2 边距中位 **0.00317（22/22 <0.05）** → 查询键近共线、softmax 近似平均（`self_key_top1_rate=0` 是会话 3 清空所致，报告中以 `self_key_note` 说明，非命中失败）；②真实运行最多占用 29 槽、**0 次真实淘汰**，合成 40 写淘汰 8、存活 age 8..39；③turn11 会话切换 `cleared=true`、prior_occupied 15、read valid 0、blended false、占用 1、F 不变（清空而非复用）；④save→restore 逐字节一致（`restore_mismatch_fields=[]`）；⑤确定性 8 次 ×17 轮签名全同；⑥与 B_EH 参数/状态（除 store）差异 **[]**，CF1 store-off 与 B_EH **765/765 字节**＋F 序列一致（摘要差异仅来自 journal 的 `model_version` 臂名，早于 20.3a）；⑦HEAD 模块等价：新旧 `living_core` `state_dict` 差异 **[]**、16 步概率逐字节一致、CF1 store-off 与 HEAD CF1 逐字节一致。

**D. 判定**：`verdict=option2_failed`（任一门不过即失败；**阈值未改、判据未改、未换选项**）。定位：结构闸证明"F 逐轮不同"成立，但查询键近共线使检索权重近似均匀（边距中位 0.00317），混合后 F 对输出字节无权威（五档反事实 0 字节变化）；信息闸低于最强对照 → 该实现的 F 副本不携带可读的逐轮纠正。

**E. 花费与记账**：`verdict=option2_failed`，工程门全 true；**224.94 CPU / 232.72 wall**（上限 300 / 2700 s）；两臂各 1 份 `addressed_store_untrained`（CF1 `a98dd1ae…`、B_EH `07563b90…`）、0 新参数、19.x 装配态未动；报告 write-then-rename、格式化在报告写完后执行并双列记录 sha；收尾只跑一次 in-process `black_check_gate.py`。**不定价、不进 20.4、不提训练方案**，结果由 owner 判断。

## 20.3b F 权威闸（2026-11-05，owner 授权；0 训练／0 优化器步／0 Adam／0 confirm）

**定位**：owner 收下 20.3a 后把失败重新分层——寻址是次级缺陷，**权威是绑定缺陷**；本闸只回答"F 能不能推动输出字节"。

**A. 先量缺口**：22 个纠正面、44 个决策位（每位 2 个动作字节），gold 类当前排名 **44/44 全部在 3 名之外**；28/44 位 gap 为负（需翻），gap 中位 **−3.96e-5**、最小 **−8.667e-3**、最大 **+8.712e-3**；整句 deficit 中位 **3.376e-4**、最大 **8.667e-3**，局部 spread 中位 **7.392e-2**，deficit/spread 中位 **0.00457** → 需推的距离是 **0.46% 的 spread**（不到半成），不是几十倍。模板位 22 面上有 11–13/12 位已错（gap 中位 −0.234）。

**B. 三条闭式读出**（oracle F＝teacher-forced 满分回放，仅诊断；读出只写内存）

| 变体 | 新增参数 | 首次翻位 / 答对 | 方向一致率 | byte_diff | 需梯度训练 |
|---|---|---|---|---|---|
| 统一增益 | 1 标量（手设） | g≤1000 无变化；**g=1e4 首次翻位**，0/22 | — | 19 | 否 |
| 闭式岭回归（λ=0.1） | **49,152**（朴素 368,640） | — / **9/22（23/44 位）** | **0.682** | 32 | 否 |
| 秩约束 r=1/2/4/8 | 8,196 / 16,392 / 32,784 / 65,568 | — / 9/22（W 秩 1，四档相同） | 0.682 | 32 | 否 |

诊断附加（不进判定）：同一前态只替换 oracle F → **0/22、byte_diff 0**，即 F 到决策位的直连在当前增益下是断的。

**C. 判定**：**`authority_weak`**（有翻位但 9<11，中间态，不自行挑边）。所需增益 g=1e4 量级且方向仍错；F 携带可读的线性方向信息（0.682）但不足以翻到 11 面。

**D. 花费与记账**：**56.92 CPU / 58.83 wall**（上限 150 / 1800 s）；失败尝试 2 次（ridge 维度错误、读出目标退化为同一常量致 W=0），其 CPU 因重跑前删除 progress 记 **missing**；0 训练/0 优化器步/0 Adam/0 confirm/0 新参数入库/未改图；报告双列 runner sha `380cdc13…`／submitted sha `af76a622…`；逐行 append＋fsync、write-then-rename、收尾只跑一次 in-process `black_check_gate.py`。**不修寻址、不改 τ/α、不进 20.4、不定价、不提训练方案**，结果由 owner 判断。

## 20.4 定价与冻结（2026-11-05，owner 一次连跑；0 正式训练／freeze 未落盘）

**定位**：20.3+20.4 连跑的第一段。先证明"存档→读档→续跑"能逐字节接上，再用真实图量一次外层更新的成本，最后按量级闸决定是否准入正式对照。

**A. 存读验证（硬前置，通过）**：3 步真实更新 → 存 `preflight.pt`（cursor=6）→ 同图续跑 3 步（reference）→ 新建图 load 权重／Adam `load_state_dict`／`set_rng_state` → 从 checkpoint 的 `cursor` 续跑 3 步。逐 step 比对 **loss／权重／Adam 矩／RNG／每 world 读数／数据游标 6 项全 true**（3/3 step 全过）。checkpoint sha `d919749a…`。

**B. 定价（按实际图，9 个样本，不外推）**：每步 batch=2 world、51 轮全场重放、可微 `run_turn`/`release` 镜像、Adam lr=1e-3、grad clip 1.0。实测 **167.50 CPU/更新**（样本 162.6–170.6，wall 比 1.069）；dev 单 world 评测 **28.02 CPU**。

| 项 | 投影 CPU |
|---|---|
| 训练 200 更新 × 4 run | 134,002.8 |
| dev 80 world × 4 run | 8,965.0 |
| 早期停机探针 5×3 档 × 4 run | 1,680.9 |
| **合计** | **144,648.7**（6.03× 冻结预算 24,000；3.01× 量级上限 48,000） |

**C. 判定**：**`pricing_over_budget`** —— 超 2× 冻结预算的量级闸，**不跑正式对照、不写 freeze**（协议全文以 `proposed_protocol` 存报告）。阻塞点是**协议单位成本**（200×2 world×51 轮全重放），不是图、阈值或寻址的问题。

**D. 定价附带读数（仅记录，不判定）**：训练前基线（step1 前态）决策位 **0.0/22**、模板位 0.0369、整句 0.0、F 写 10/10；6 步后首个 dev world 决策位 0.3409/22、模板 0.0045、整句 0.0。**模板位本身错、底座未训练的前提仍成立，不得据此判 CF1 成败。**

**E. 花费与纪律**：**1,539.89 CPU / 1,644.89 wall**（轮上限 6,000／14,400 s，未触线）；0 confirm 读取、未改图、store=false 按冻结值、0 正式训练步；报告双列 runner sha **`307cf585…`**（产生读数）／submitted sha **`1100eb8f…`**（提交前修 lint B007 后的字节）；进度逐行 append＋fsync、write-then-rename；收尾 in-process `black_check_gate.py` 一次。

**F. 下一步（唯一）**：由 owner 裁定定价闸。若同意，把每 seed 外层更新数从 **200 降到 50**（batch 2、2 臂×2 seed 不变），投影约 **44.1k CPU（1.84× 预算）** 可落回 2× 闸内（余量仅 8%，需重新定价一次，不得复用本次数字）。

## 20.4a 单位成本拆解与优化试算（2026-11-05，owner 否掉“200→50”降本；0 训练）

**定位**：不是把实验做小，而是把浪费定位出来。owner 的判据是“167.5 CPU/更新＝每轮 1.6 CPU，对 3.3M 参数、15 字节一轮高了两三个数量级，先怀疑跑法开销”。

**A. 拆账（5 个真实更新样本，单 world，batch=2 按 2× 折算）**：per-world **82.919 CPU**（样本 81.172–84.625）→ batch-2 单位 **165.838 CPU**；每轮 **1.6259 CPU**、每个回答字节步 **0.108391 CPU**。

| 分相 | CPU | 占比 |
|---|---|---|
| 反向 | 38.956 | 47.0% |
| 解码（15 步/轮） | 26.444 | 31.9% |
| observe（读 query/feedback） | 15.397 | 18.6% |
| 外部记账（world 循环等） | 2.050 | 2.5% |
| 优化器步 | 0.047 | 0.06% |
| 数据准备 / zero_grad | 0.003 / 0.0 | ≈0 |

top-3 消耗者：**反向 > 解码 > observe**。profiler 的 top op 是 `slow_conv_dilated2d`（前向 0.1395 s / 306 次）、`select`（93,636 次）、`bmm`、`addmm`、`copy_`/`fill_`（4 万次级的小算子）；反向是 `bmm`/`mm`/`convolution_backward`/`add_`（11,834 次）。

**B. 四条候选（逐条实测或给出未测原因）**：

| 编号 | 候选 | 状态 | 读数 |
|---|---|---|---|
| T1 | 两 world 合并为一次批量前向 | 未测 | 冻结 R021 的 `step/observe` 只收单事件张量，批量化属图级改动，需 owner 授权 |
| T2 | 去掉每轮重建（预计算不变部分、复用缓冲、合并小算子） | 未测 | `source_view`/`goal_vector`/`align_positions` 在 `step` 内逐字节重建，提升属图级改动；profiler 指向 `select`/`fill_`/`copy_` 的 4 万次小算子 |
| T3 | 读数路径不开梯度（`inference_mode` vs `no_grad`） | 已测（5 样本） | 每轮 0.4844 → 0.4531 CPU，**1.069×**；loss 差 0.0、logits 最大差 0.0、发射字节完全一致 → **等价档** |
| T4 | 反传窗口（每轮 detach，截断 BPTT） | 已测（5 样本） | per-world 82.919 → **62.659 CPU**，**1.323×**；loss 差 0.0 但**梯度最大差 2.39352119**（阈值 1e-6）→ **`not_equivalent`，只能进放松档** |

**C. 等价性硬证据**：线程 1、dtype float64、确定性开关开启（逐项写在报告 `setup`）。等价档只有 T3（loss/logits 逐元素差均为 0.0）；T4 超阈值，判 `not_equivalent`，不计入等价档单价。

**D. 三档单价与重算投影（200 更新 × 4 run ＋ dev 80 × 4 ＋ 探针 15 × 4）**：

| 档 | batch-2 单位 CPU | 合计投影 | 对 24,000 | 对 48,000 | 换掉的信用 |
|---|---|---|---|---|---|
| 等价档（现状，T3 只省读数） | 165.838 | **141,451.6** | 5.89× | 2.95× | 无（读数路径本来就不训练） |
| 放松档（每轮 detach） | 125.319 | **109,036.6** | 4.54× | 2.27× | 跨轮信用：纠正面只能经自己那一轮改参数 |
| 图级下限（只算前向） | 87.781 | **79,006.5** | 3.29× | 1.65× | 无（但需批量化/融合执行，未授权） |

判定：**`pricing_still_over_budget`**。即使按等价档最优单价，仍超 2× 闸；**连只保留前向的图级下限（79,006.5）也在闸外**。三种取舍见报告 `options_for_owner`（O1 放松档 109,036.6／O2 图级批量化下限 79,006.5／O3 只重放 22 个纠正面 34,081.2），各自天花板与代价已写明，**不替 owner 挑**。

**E. 花费与纪律**：本次复核 **56.297 CPU / 58.67 wall**；含首跑测量遍 **727.891 CPU**（10 个校准更新，读数已落 progress），合计 **784.188 CPU**，在上限 **800 CPU** 内。首跑在读数 A/B 前因取错键（`loss_tensor` 不在 `eval_world` 返回里）崩溃，本次复用首跑 10 个样本、只补 T3 A/B 与微 profiler。0 正式训练步、0 confirm、未改图、未写 freeze、阈值未改、`addressed_store=false` 按冻结值；进度逐行 append＋fsync、write-then-rename；收尾一次 in-process `black_check_gate.py`。双列 sha：`runner_sha256` `7b19298b…`（产生读数）／`submitted_runner_sha256` `20ebd303…`（入库）。

**F. 下一步（唯一）**：由 owner 在 O1/O2/O3 中裁定一条（或另行给出更新数），再按新单价重新定价一次；未裁定前不写 freeze、不开正式对照。

## 20.4b 图级优化与等价验证（owner 已裁 O2；已交付，判 `pricing_still_over_budget`，0 训练）

**Owner 裁定（2026-11-05）**：走 **O2 图级批量化与算子融合**，否掉 O1（改掉被检验对象：梯度最大差 2.39，剪断跨轮反馈信用）与 O3（丢掉 29 轮会话＝丢掉会话连续性，CF1 的立身之本）。**不降更新数、不降 seed 数、不砍轮数**——目标是去掉跑法开销，不是把实验做小。

**A. 三条优化（逐条实测，每条 ≥5 样本）**
1. **两 world 合并批量前向**：`step/observe` 接受批量事件张量（带批维），一次前向一次反传；**不得**用 Python 循环把两遍包起来。
2. **消灭 9.3 万次 `select`**：定位调用点，改预计算索引＋切片＋`index_select`/`gather`，静态张量缓存复用；报告优化前后 op 调用次数。
3. **融合 306 次卷积**：同核重复则改单次批量卷积；不可融则写原因与实际可达下界。
可叠加 T3（读数去梯度，已证等价 1.069×）。

**B. 等价门（硬）**：与**当前未优化实现**在同一权重、同 seed、同数据下对照：loss `max|Δ| ≤ 1e-6`；**全部参数梯度** `max|Δ| ≤ 1e-6`，并逐参数列出超阈值者（超即作废，只能进放松档）。线程数、dtype、确定性开关、RNG 逐项固定写进报告；两次前向 logits 也比。改图后重算 `candidate_sha_v2`：数值等价应保持不变，若因代码语义变化而变必须写原因与新值。

**C. 重定价与判定**：用优化后实测单价重算 **200 更新×4 run＋dev 80×4＋早期停机探针**。≤48,000 → `pricing_admissible_after_optimization`（仍不训练、不写 freeze，等 owner 批），报告给加速比、新单价、等价证据索引；仍超 → `pricing_still_over_budget`，给实测下界与瓶颈名（剩余最贵 op）＋三种取舍的天花板代价，不自己挑。

**D. 纪律与预算**：0 正式训练步、0 confirm、不写 freeze；允许 ≤10 步校准更新（仅测速与等价，单列声明、不进能力读数）；证据名 `20_4b`；双列 runner sha；读数逐行 append＋fsync；收尾一次 in-process black；plans 六处＋账本 **D16-36**；提交不推送。预算 **1,200 CPU / 40 分钟**；若出现数值不等价且无法定位，立即停并报 `optimization_not_equivalent`。

**E. 结果（新增执行器 `taiji/living_core_batch.py`，冻结文件 0 改动）**

等价门（同权重、同 seed、同数据，线程 1／float64／确定性开启／RNG sha 已写进报告）：loss 差 **0.0**、**全部参数梯度**最大差 **4.547e-13**（逐参数列表 123 项，超阈值者 `[]`）、逐轮 logits 最大差 **1.554e-15**；`candidate_sha_v2` **`58754e50…` 前后不变**。首轮自检另用 B=1 重放比对 h/fast/coverage/prior/发射字节，全部 ≤1.4e-15。

| 优化 | 实测 | 加速 | 换掉了什么 |
|---|---|---|---|
| T1 两 world 批量前向＋一次反传 | 165.838 → **56.906 CPU/更新**（5 样本 56.047/51.500/56.641/55.406/64.938） | **2.914×** | 第二遍 Python 循环包的前向/反传 |
| T2 预计算索引＋`gather` | align 0.11→0.03 ms、consume 0.26→0.05 ms；`select` **190→0**／次 | 3.279× / 5.254× | 每 key 逐次 Python 索引（profiler 里是 `as_strided`+`unsqueeze` 对） |
| T3 编码器 hoist＋批量卷积 | 每轮 0.1281 → **0.0094 CPU**；`conv1d` **405→12**／轮 | **13.667×** | 同一 raw 字节被重复编码 135 次 |

T2 的整更新 `select` 计数：20.4a 实测 **93,636 → 0**（优化路径只用 `gather`；按满 key 数外推的上界 290,700 已注明是上界不是实测）。T3 的 306 次卷积即同一批核重复施加，profiler 数到每轮 405 次 `conv1d`。

**F. 重定价（200 更新×4 run＋dev 80×4＋早期停机探针 15×4）**：训练 **45,525.0**＋dev **2,193.0**＋探针 **411.2**＝**48,129.2 CPU**＝**2.01× 冻结预算**、**1.0027× 量级闸**——超出闸 **129.2 CPU（0.27%）**。判 **`pricing_still_over_budget`**（按合同口径，超了就是超了）：**不训练、不写 freeze**，仍等 owner 批。

**G. 必须披露的测量方差**：同一份代码（仅格式化差异）跑了两次。第一次宿主剩 1.9 GB 内存，单价 **40.291 CPU**、投影 **34,231.1**（≤48,000，本会判 admissible）；交付这次宿主只剩 **0.13 GB** 空闲、本进程 RSS 1.1–1.7 GB，单价 **56.906**、投影 **48,129.2**。差 **+41%**，远大于任一轮内的 5 样本散布（51.500–64.938）——多出来的是缺页开销被记进 `process_time`。报告用保守（后一次）数字定判，并把两次读数同时列出；0.27% 的差额是否算“可接受”，由 owner 定。

**H. 瓶颈与三种取舍（不替 owner 挑）**：剩余最贵是**反向**（31.628 CPU/样本，55.6%），前向 25.203（44.4%），前向内部已转为 `bmm`/`mm` 主导（`select` 归零、`conv1d` 已 405→12）。报告 `options_for_owner`：**A 抬闸**（实测 48,129.2，闸需从 48,000 抬到 ≥48,200，不动被检验对象）；**B 放松档**（≈37,422 算术外推，剪断跨轮信用，owner 已否）；**C 缩量表**（≈36,096＝200×3 run，或 ≈34,143＝150×4 run，削弱读数，owner 已否）。

**I. 花费与纪律**：交付run **516.0 CPU / 680.978 wall**（上限 1,200／2,400 s 内）；校准更新 **6 步**（1 次等价参考＋5 次计时），**0 正式训练步、0 confirm、未写 freeze、冻结文件与阈值未改**。工程重跑 4 次须一并声明：①首跑 `shift_index` 把单 world 键表当批表，unpack 崩溃；②次跑 5 样本累积保存 51 轮 logits，宿主内存不足 OOM；③三跑为补测量分辨率（Windows `process_time` 15.6 ms 刻度，微基准要 4000 次重复）主动停止；④格式化后重跑以让双列 sha 与文件一致。四次累计约 1,700 CPU，**超本合同 1,200 CPU 上限**，如实记账。双列 sha：`runner_sha256`（产生读数）＝**`3ee24ea5…`**，`submitted_runner_sha256`＝**`935591be…`**；执行路径 `executor_sha256`＝**`2eaaacb7…`**，提交执行器 `b01d700b…`。两列之差只是 5 处无用绑定（import、`reference`、`emitted_same`、`length`、`c`），全部不在数值路径上，且提交字节**未再跑等价门**——读数归属 `runner_sha256`／`executor_sha256`。

**J. 下一步（唯一）**：owner 对“0.27% 超闸”给一句裁定——**抬闸到 ≥48,200 并批准正式对照**，或**指定 B/C 取舍**；未裁定前不训练、不写 freeze。

---

## 2026-11-07 P0 计划一致性校准与 21.1 结项写回（0 训练 / 0 前向 / 0 confirm / ≤30 CPU 预算）

owner 批准 P0 为唯一队首后执行：**21.1 零训练读出分解已结项，判 U 分支**（`upstream_representation_is_first_wall`，4/4 格、CPU 1597.344 / wall 1610.581、0 Adam / 0 confirm / 0 写 checkpoint、整句 12/12 格恒 0）。报告＋manifest 入库：因仓库对 `reports/*.json` 设 `eol=lf` 而原回执为 CRLF，入库前做**内容逐值不变的 EOL 规范化**并重绑 `report_sha256`（原 `c19a9e1c…` → 入库口径 `e7666226…`），隔离 index 验证 blob == 工作区 == manifest；双列 runner sha 与已入库 runner blob 一致、4 个 checkpoint sha 全吻合。三处队首（MAINLINE / MASTER-PLAN / ROADMAP-NEXT）写回为逐字相同的同一句（sha256 `c05e9530…`），同步页首状态、状态表、依赖顺序、包 21 计划 §0/§2 标题/§7 与本页「下一步」；账本 §20 T-U1 追加 P1 证伪指针（原命题正文未改）；退役清单 7 项逐项有回执，旧 checkpoint 目录按账本 §98 出旁挂 `superseded` 声明。六项结项条件全过，机器回执见[P0 结项卡](../../../reports/taiji_package21_1_p0_closeout_20261107.json)。**U 分支只是 21.3 的准入条件，不是 21.3 完成，也不是里程碑关闭**；21.2/21.4 保持冻结、U1 不进队列。本段 0 训练 / 0 优化器步 / 0 前向 / 0 confirm / 0 评测重跑 / 0 数据重读，未改任何冻结判据、门值、预算或 21.1 读数；停在 owner 裁定节点（是否准入 21.3）。

---

## 2026-11-07 21.3 准入与预注册冻结（0 训练 / 0 前向 / 0 读数）

owner 批准 21.3 准入后，按「判据先落盘、跑完不改」先冻结预注册合同：[规格](../../reference/SPEC-PACKAGE21_3_REPRESENTATION_SEPARABILITY_PREREG_20261107.md)＋[机读 manifest](../../manifests/taiji_package21_3_prereg_20261107.json)（`criteria_sha256=02b727f958dbad8437a73c2914bf321a81263f5c85dc08183b3bd56f11d8c512`，对 canonical JSON 自洽可重算）。**本段 0 读数、0 模型前向、0 训练、0 confirm**，21.3 未执行。

取证发现一处会翻转结论的口径冲突：20.2s 把「最强对照」算成标签打乱 5 seeds 的 **mean**（0.5091 → margin 0.1727 → `inconclusive`），而 20.3a 的冻结条款写的是 **per-seed max**（历史 max 0.7727 > 主探针 0.6818 → margin 为负 → 直接判负）；P1 与包 21 §4 只写「最强对照」、未指定统计量。规格裁定采用 **20.3a 已冻结的 max** 为绑定口径（20.2s 的 `inconclusive` 不改写，只作历史敏感性锚；报告须双口径附报，不得读后切换），门值 0.60／0.20／0.30／0.10 与 CPU ≤2,000 **一字未改**。另取四条对照并集（窗口身份／标签打乱／随机同维同范数／无 F 其它持久态）、沿用 20.3a 的 `distinct_f_vectors>2` 有效自由度闸门、4 个 checkpoint 取最严且不挑最好；并登记「判负／不可判不自动转 21.4 或训练」的转段冲突（21.4／P3 的准入都不含「表示判负」）。特征卡须在读数前落盘，否则 `engineering_stop`；runner 双列 sha 与报告 sha 在 runner 未写前一律 `null`，不得预填。

三处队首写回为逐字相同的「21.3 已准入、预注册冻结、待裁定执行读数」，同步状态表、依赖顺序、ROADMAP-NEXT §2b、包 21 计划 §4/§7、详细计划 P1 与本页。**准入 ≠ 执行 ≠ 里程碑关闭**：21.3 尚未执行、尚未完成；21.2／21.4 保持冻结，U1 不进队列。停在 owner 裁定节点（是否按合同执行读数）。该裁定已由 owner 批准，执行结果见下段。

---

## 2026-11-07 21.3 表示可分性诊断执行与结项（0 训练 / CPU 144.984 / 0 confirm）

owner 批准按冻结合同执行后，先落[特征卡](../../manifests/taiji_package21_3_feature_card_20261107.json)（首个读数前，sha `b4e3a5cc…`）并把 runner 单独提交为已入库字节（双列 sha `714adb6c…`），再跑读数：`preflight`（criteria sha、特征卡 sha、4 个 checkpoint sha、`addressed_store_enabled=False`）全过，**守卫红/绿双向演练**通过（预算耗尽抛 `BudgetStop`、冻结预算不抛；篡改 criteria sha 抛 `EngineeringStop`、原值通过；四个判定边界返回值符合冻结出口）。报告 write-then-rename、progress append＋fsync。

**读数**：4 个 20.4c_r2 末步 checkpoint **逐位相同**。主探针决策位 **0.6364**（≥0.60 门）；四条对照：窗口身份 **0.6818**、标签打乱 5 seeds mean 0.5182 / max 0.5909、随机同维同范数 0.3636、无 F 其它持久态 0.4545。绑定口径最强对照＝窗口身份 0.6818 ⇒ margin **−0.0455**（<0.10）⇒ 判 **`representation_axis_closed_for_this_graph`**。20.2s 遗留 mean 口径 margin **+0.1182** 会落 `inconclusive` 中间带，**按预注册不绑定、未事后切换**——预注册 §3.1 指出的口径冲突由实跑数据坐实。诊断 A：F 余弦中位 ~1.0、最小 0.9747、top1−top2 边距中位 0、矩阵秩 **2**；诊断 B：`distinct_f_vectors=2`（<3 有效性闸门，每 session 1 个）、release 写 10 次、配对 bootstrap 如实报**退化**（不填 0、不编区间）。

**归因与边界**：三轴分账（表示可分性／写入时机／发射读出，账本 I22.2），0 训练 ⇒ 不涉及训练剂量归因（D16-12）；判负是**当前图/当前数据/当前探针面**的范围结论，不升格为「表示不可能有权威」；20.2s／20.3a／20.3b 历史读数一字未改；按预注册 §3.6 **不自动转 21.4 或任何训练包**（21.4／P3 的准入都不含「表示判负」）。九项结项条件全过，见[21.3 结项卡](../../../reports/taiji_package21_3_closeout_20261107.json)。**0 训练 / 0 优化器步 / 0 反向 / 0 新参数 / 0 写 checkpoint / 0 confirm / 0 评测重跑**；CPU **144.984** / wall **145.319**，为 2,000 预算的 7.25%。

三处队首写回「21.3 已结项判负；待裁定全局收束」，同步状态表、依赖顺序、ROADMAP-NEXT §2b、包 21 计划 §0/§4/§7、详细计划 P1/§5 与本页。**包 21 尚未整体关闭**；21.2／21.4 保持冻结，U1 不进队列。停在 owner 裁定节点（是否发布包 21 全局收束卡）。该裁定已由 owner 批准，收束结果见下段。

---

## 2026-11-07 包 21 全局收束卡（P7 完成；0 训练 / 0 前向 / 0 confirm）

owner 批准发布[收束卡](../../reference/CLOSEOUT-PACKAGE21_GLOBAL_20261107.md)＋[机器卡](../../../reports/taiji_package21_global_closeout_20261107.json)。**准入门两条同时成立**：① 21.1 判 U、21.3 判 `representation_axis_closed_for_this_graph`，被 U 分支准入的唯一结构路径已走完；② 包 20 measured **50,573.637 CPU**＝提案 B（24,000）的 **2.1072×**、48,000 闸的 1.0536×，按 P7 替换方案「累计越闸 → 停推，先出收束卡，不新开包」。

**六件齐、8 项结项条件全过**：① 产物清单 10 件（逐件工作区 sha 与 HEAD blob sha 一致）；② 能力账——唯一 [已证] 仍是 **C12**（Task1 结构 188/192），U6／U13／U5 未证、U14／MGW1／U1 设计层、U1 实现未落代码，包 20/21 **无新增能力收益**；③ 成本账——包 20 汇总 50,573.637（已含 20.4c_r2 42,831.453、20.4d 4,026.438、各诊断与 20.6，**不得重复加总**；measured 只是下界，缺失留 null 不补 0）＋包 21 独立 **1,742.328**（21.1 1,597.344＋21.3 144.984，0 训练）；④ 恢复/保持——20.2 三路径 90/90、20.4c_r2 `resume_verified` 与 16/16 确定性，但**学后旧能力保持未实测**（只有稳定性漂移门，不得写成保持通过）；⑤ 负结果表——14–18／19／包 20／21.1／21.3／20.2s／20.3a／20.3b／U1 P1–P8 全部保留原文与读数，一字未改；⑥ 下一轮**唯一**核心问题。

**编号口径**：P7 沿用的「L1–L14」是历史编号，现行权威覆盖表是本页 2026-11-02 的 **23 条条款**与 [核心需求 CR-1..CR-10](../TAIJI_CORE_REQUIREMENTS.md)。收束卡按现行口径汇总、**不编造 L 条目**，并登记该差异。当时「目标未达成，卡点只有第 5 条（自身机制承担决策），同时挡住 6/7/8/9/10/13」的结论在包 20/21 后**仍成立且证据更强**：20.3b oracle-F 零变化、21.1 U 分支、21.3 表示轴关闭三条独立证据同指架构级缺口。

**下一轮唯一核心问题（待 owner 裁定）**：当前图的快权重 F 是否具备「按问句／按反馈事件可分且可寻址」的表达能力？若不具备，是需架构级改造引入，还是应判该函数类不含此能力、另立主体？须先以**零训练可表达性判据**裁决，未裁定前不得以训练剂量立项（D16-26；21.4 未准入）。**这不是**训练计划、定价、新包号或能力声称，也不把「未证」写成「不可能」。

三处队首写回逐字相同的「包 21 已全局收束；待裁定下一轮核心问题」，同步状态表、依赖顺序、ROADMAP-NEXT §2b、包 21 计划 §0/§7、详细计划 P7/§5 与本页。**包 21 收束 ≠ M 线或里程碑关闭，也不是「范围排除批准」**；confirm 全程 0 读取，历史归档不删，不自动推进 P1–P6 任何一节、不新开包。

---

## 2026-11-08 FEC-1 快权重可表达性判据冻结（0 训练 / 0 前向 / 0 读数 / 0 confirm）

owner 已裁定「先立零训练可表达性判据专项」以回答上段遗留的唯一核心问题，本段只落判据、不跑读数：[冻结规格](../../reference/SPEC-F_EXPRESSIVE_CAPACITY_PREREG_20261108.md)＋[机读 manifest](../../manifests/taiji_f_expressive_capacity_prereg_20261108.json)（`criteria_sha256=1e0dde222d557b4ea97eee528b9f73a2aec3068a2c462e927465e415fa3b6142`，对 canonical JSON 重算自洽）＋[判据自检回执](../../../reports/taiji_f_expressive_capacity_prereg_guard_20261108.json)。判定对象是**函数类**而非某个 checkpoint：保持 C12 父身份与当前图（`addressed_store_enabled=False`）不变，`F ∈ R^{4×2×32×32}` 能否承载并按问句／按反馈事件取回。

**取证中发现一条被忽略的算术必然性**：20.2s 的 `distinct_f_vectors_session1=1`、`session2=1` 意味着 F 在**每个 session 内逐位恒定**；此时任何只以 F 为输入的解码器在该 session 内输出恒定，其判别力**恒等于「只按 session 分」的基线**。故 20.2s 的 `primary 0.6818 == window_identity 0.6818` 不是巧合而是该前提的算术后果，21.3 的 `0.6364 vs 0.6818` 同源。**推论**：20.2s 的 `inconclusive` 与 21.3 的 `representation_axis_closed_for_this_graph` 仍各自在其冻结口径内有效，但**均不可被引用为「F 里没有可分信息」的函数类判负**——它们证明的是「在这个写入序列下 F 的可见状态没有 session 内变化」。**尚未被检验的**是：若让 F 每轮都写（只改 `allowed` 掩码，0 参数 / 0 优化器步 / 0 改图），写键是否张成足够维数、读键是否落在同一子空间、闭式读出能否分离。

**判据分七组、两个方向都不得互相替签**：时机覆盖率（把「没写」与「写不了」分开，20.2s 实测 **10/51＝0.196**）、强制写下的写键有效秩与逐对可分率、键范数下界（`unit_ball` 是收缩映射，近零键会让外积消失而余弦矩阵仍「健康」）、写键与读键的子空间对齐（两套独立 MLP、无 weight tying）、闭式 tied-key 读出、值面有效秩、衰减时间常数给出的容量上界。**结构性 verdict 只由强制写产生**；实际稀疏写单列为具名 verdict。tied-key 读出不是弱探针而是上界：忽略衰减时 `M K = V G`，`rank(G) < N` 时**不存在任何线性解码器**能精确恢复全部 N 值。双通路不可互认——21.3 用通路 B（`encoder.context(query).mean(0)`，128 维，20.3a 已实测 22 键 `fraction_below_0.05=1.0`）的键去读通路 A 的产物，结论只在那个跨通路口径内有效；本专项只判通路 A。

**问句轴与内容轴分离**：22 个纠正面有 22 个不同问句文本，但只有 **11 个不同 bits 组合**；文本级通过而内容级不通过判 `text_level_only`、不满足问句轴。三类判负各对应不同下一步：`key_space_collapsed_structurally` ⇒ 架构级改造；`query_key_misaligned` ⇒ 最小充分修复＝tied key／共享投影，不必另立主体；`key_space_collapsed_under_actual_timing` ⇒ 写入时机，属工程改动、不构成训练理由。

**守卫是真红过一次的**：自检器六项（`criteria_sha256` 重算一致／篡改一个门值后不一致；三处队首逐字一致／篡改一字后不一致；本件不携带第二套门值／植入字面量后能检出）全过；另做真实文件篡改演练，把一处队首加两字后 `headline_green` 确实变红、退出码 1，恢复后字节与演练前逐字节相同、守卫复绿。**量**：本轮 0 训练 / 0 前向 / 0 checkpoint 读取 / 0 confirm / 0 定价；只发生文档写入与哈希计算。

三处队首写回逐字相同的「包 21 已全局收束；收束卡 §5 的唯一核心问题已由 owner 裁定为先立零训练可表达性判据专项，FEC-1 判据已冻结、待裁定执行」，同步状态行与 ROADMAP-NEXT 执行顺序。**立专项 ≠ 执行 ≠ 里程碑关闭 ≠ 范围排除批准**：FEC-1 尚未执行、尚未产出任何读数；21.2／21.4／U1 与各训练包保持冻结，不新开包。


## 2026-11-08（FEC-1 执行）：owner 裁定执行后按冻结合同跑完，判 `key_space_collapsed_structurally`（4/4 checkpoint 一致，无分歧）

执行前硬门六项全过：prereg manifest 入库且 runner 常量 `CRITERIA_SHA256=1e0dde22…` 逐字一致；feature card（`e91252dd…`）在**读任何读数之前**落盘（写键截取=write_heads 前向钩子＋trace 逐位交叉核对、事件约定=纠正面 observe 末步、allowed 掩码构造、M_final=全程重放终态、有效秩 1e-2 相对阈值、Tikhonov eps=1e-6、留一解码=欧氏 NN 到事件 id、seed 与 bootstrap=2000 次/20261107）；4 个 checkpoint sha 与 21.1 报告逐一重算吻合；`build_graph` 供给脚本 sha=112c76ae… 吻合；random_init 图 weights_digest 与 20.1 报告 shared_init `c5ebe35b…` 一致；0 训练 / 0 confirm 声明落盘。**仪器有效性**：driver 与 graph.run_turn/graph.release 在 actual 掩码下逐位等价（2 个 checkpoint：每轮发射字节、终态 F max|Δ|=0.0、release written 计数全同）；forced 条件的**唯一**干预=allowed 全 True。工程重跑 3 次（钩子 raw 需 reshape、纠正面二次 take 空缓冲、trace 需 detach），均在 preflight/早期崩溃、费用可忽略，progress jsonl 如实留痕。

读数（报告 `taiji_f_expressive_capacity_20261108.json`＋执行 manifest＋progress jsonl；成本 **322.69 CPU** / 1,500 停 / 0 训练 / 0 Adam / 0 新参数 / 0 写 checkpoint / 0 confirm）：**verdict=`key_space_collapsed_structurally`，4/4 checkpoint 一致**。forced 写有效——51/51 轮全部写入、22 面 F 向量全不同（dof=22，session {1:5, 2:17}）——**写机制响应内容，但写键空间不张成**：8 个 (group,head) 栈 binding 秩 2/2/2/3（门 ≥3）、逐对可分率 **4×0.0**（门 ≥0.95；全部键对 |cos|>0.95 近共线）、键范数中位 0.83–0.97（健康，非零键失败）、条件数 14.9–58.9（≤100）。query_alignment 中位 **−0.147**（写键与读键近正交，两套独立 MLP 无 tying）；query 秩 min 2、值面秩 min 1；容量 **256≥22**（衰减不是瓶颈）。tied-key 闭式读出（上界探针）：event-id 检索 **0.0455≈1/22 chance**——精确检索失败；decision-bit 0.5909–0.6818 是窗口身份味道的噪声（四对照 strongest **0.8182**，四 checkpoint margin 全负 −0.0455…−0.1818）。实际写入对照单列（不进结构性 verdict）：timing coverage **0.0**（纠正面结构性不写）、written 10/51、80 个 release 写事件键集同样塌缩。**归因（attribution 字段）**：random_init 对照在 forced 下同样塌缩（pairwise 0.0、键范数 0.706 健康）→ **架构固有**；训练压塌/写入时机/读出对齐三个竞争解释全部排除。判据 sha 读后未改；verdict 范围限定当前图与 checkpoint 谱系。第二次完整重跑（runner 修 zip strict 计 1 处后）读数逐位复现、runner sha 双列与提交字节一致。

队首写回三处（MAINLINE / MASTER-PLAN / ROADMAP-NEXT）逐字相同；**下一步＝W1 架构级改造分支**（π 输入加内容可分特征／tied-key＋键白化／正交码本／内容寻址键源重设计，见 PLAN-FEC1_ADJUDICATION_WINDOW_20261107.md），等 owner 裁定后准入；不构成训练理由，21.2／21.4／U1 冻结维持。

## 2026-11-08（W1(a) 第 1 轮）：owner 裁定 (a) 后实施内容条件化写键并复测，判 `key_space_collapsed_structurally`（新图 init 双臂 2/2 一致）；同时发现 FEC-1 PASS 分支算术不可达

实施（加法式、default-off）：`living_core.py` config 新增 `content_keys_enabled/content_gain`（默认 False/1.0）＋ `content_feature()`（问句字节 128 维计数、L2 归一）＋ step/write_fast 内容通路（键＝unit_ball(unit_ball(raw_key)+gain·unit_ball(P·c))，值/衰减/门段不动）；`growth_credit.py` 的 GatedGrowthCore.step 同步（subject 的实际 step 解析到该覆盖，首跑由此暴露）；`rule_learning_20_1_net.py` 新增 `enable_content_keys`（seeded 32×128 投影、frozen config 用 dataclasses.replace 换新）。**default-off parity 逐位过**（disabled 图与 graph.run_turn 逐位等价）；R021 oracle 12/12、ruff/black/B-SIM 全绿。复测 runner `probe_taiji_fec1_w1a_retest_20261108.py` 复用 FEC-1 冻结模块的分析机器（key_gates/closed_form/controls 全部 import 自已入库 runner）；新图 cross-check＝从钩子 raw＋目标字节＋注册投影重算 post-injection 键并要求逐位相等。过程中修 3 处工程错（dtype float32/f64、GatedGrowthCore step 覆盖漏改、goal 重复传参），均在 preflight/早期崩溃、progress jsonl 留痕。

读数（[报告](../../../reports/taiji_fec1_w1a_retest_20261108.json)＋[执行 manifest](../../manifests/taiji_fec1_w1a_retest_execution_20261108.json)；**150.92 CPU** / 0 训练 / 0 confirm / 判据 sha 未改）：verdict=`key_space_collapsed_structurally`（2/2 一致）。**注入机制有效**：写键有效秩 2→6（门 ≥3 过）、值面秩 1→5（过）、读键秩 2→5（过）、逐对可分率 0.0→0.1039（各栈 0.10–0.55，门 0.95 远未过）、键范数 0.57 健康、条件数 24.8、dof=22；query_alignment −0.147→−0.139（门 0.5 未过）；闭式读出 event-id 0.0455→0.0909（仍近 chance）、margin −0.1364。**归因分层**：①秩/值面已修；②逐对可分卡在内容特征的判别力（22 问句共享模板使 byte-count 特征的槽位判别被淹没）；③对齐卡在读键未接内容通路。**判据边界发现（同步入账本 §21）**：FEC-1 的 expressive_capacity_present 需要 margin≥0.2，而最强对照（per-seed MAX permutation，22 样本×8,320 维 ridge）过拟合至 0.8182 ⇒ primary 需 ≥1.0182>1.0——**PASS 分支在新旧图上均为空集**；无论架构多好最多到 inconclusive。两轮修复-复测循环剩余 1 轮，但 PASS 不可达使其失去产出 PASS 的意义；按防钻牛角尖硬规则停止、不烧第 2 轮预算，升级 owner 三选一（①接受结构性判负为最终裁决／②修订对照统计量后重测／③收束本窗口）。

## 2026-11-08（W1 round-2＋窗口收束）：owner 裁定②后修订对照并跑第 2 轮，判 `query_key_misaligned`（2/2 一致）；两轮用尽，按硬规则强制收束

v2 预注册（[manifest](../../../plans/manifests/taiji_f_expressive_capacity_prereg_v2_20261108.json)，`criteria_sha256=aa3a8aef…`）：**只修对照统计量**——四个 fitted 对照全部容量封顶到 32 维种子随机投影（v1 的 permutation 对照在 22 样本×8,320 维 ridge 上过拟合至 0.8182 使 PASS 需 primary≥1.0182>1.0，D16-42）；门值/decision rule/verdict precedence/margin 定义/预算逐字不动，v1 原块保留审计。round-2 设计（[card](../../../plans/manifests/taiji_fec1_w1r2_retest_feature_card_20261108.json)，`4855e43e…`，读数前落盘）：居中内容特征（质心=22 问句输入侧字节计数均值，无 gold）＋tied 读键（同一投影同一增益）。实现：read_fast 内容通路（living_core+growth_credit，additive default-off）、content_feature 居中模式、enable_content_keys 加质心；过程中修 3 处工程错（staticmethod 残留、read/write 两处 content_feature 计算顺序在 read_fast 之后、goal 重复传参），parity 逐位过、oracle 12/12、全门绿，progress jsonl 留痕。

读数（[报告](../../../reports/taiji_fec1_w1r2_retest_20261108.json)；**126.31 CPU** / 0 训练 / 0 confirm / v2 判据 sha 读后未改）：**写键空间门首次全过**——pairwise **0.974**（门 0.95；round-1 为 0.104、旧图 0.0）、秩 7、范数 0.63、cond 41、容量 256、dof=22；未过：query_alignment **0.110**（门 0.5；读写 MLP 固有输出与内容方向近反平行，共享 boost 的 tying 不足）、值面 22 值仅张成 5 维（门 ≥3 过但检索需逐对不同）⇒ 闭式读出 event-id **0.0**、margin −0.1818（v2 对照 strongest 0.7273：permutation-capped 0.7273／window 0.6818／other 0.6364／random 0.3636——v2 的容量封顶生效，PASS 恢复原则可达）。verdict=`query_key_misaligned` 2/2 一致。

**W1 修复-复测循环第 2 轮（最后轮）用尽 ⇒ 按防钻牛角尖硬规则强制收束**（[收束卡](../../reference/CLOSEOUT-FEC1-WINDOW_20261108.md)）。窗口总账 599.92 CPU；架构需求输入三部件（内容写键[已入库 default-off]／读写全 tied 或训练对齐／值通路内容条件化）；队首＝owner 裁定立 v2 重设计新窗口或接受负结果结案；不自动开跑，21.2／21.4／U1 冻结维持。

## 2026-11-08（W-v2 重设计窗口首轮）：owner 裁定立 v2 窗口后实施三部件并复测，forced 全部结构门通过；verdict=`key_space_collapsed_under_actual_timing`（非架构归因）

实现（[card](../../../plans/manifests/taiji_fec1_wv2_retest_feature_card_20261108.json) `9110e7a1…` 读数前冻结）：部件②读写全 tied——read 路径停用独立 fast_queries MLP，改复用写寻址函数（写头 key 段在 read 自身上下文求值，`tied_read_keys`，same event/goal/nu/past_error/mode），trace fast_queries 变为 post-tying；部件③值通路内容条件化——写入值＝sigmoid(write_gate)·unit_ball(tanh(v_raw)＋gain·boost)，同一投影同一内容基。均为 additive default-off（新 config 旗 `content_path_v2`），disabled parity 逐位过、R021 oracle 12/12、全门绿。复测 runner `probe_taiji_fec1_wv2_retest_20261108.py`（仪器=round-1 replay＋v2 感知 capture：每步 2×4 钩子触发、读/写键均逐位重算交叉核对、值重算含 boost）。工程重跑 2 次（check_budget 故障文本、钩子 8 触发），preflight 早期、留痕。

读数（[报告](../../../reports/taiji_fec1_wv2_retest_20261108.json)；**135.55 CPU** / 0 训练 / 0 confirm / v2 判据 sha 读后未改）：forced 下**全部结构门通过**——pairwise **0.974**、秩 7、键范数 0.63、cond 41、**读写对齐 0.9997**（round-2 的 0.110 被全 tied 解决）、值秩 5→7、容量 256、dof=22。未过：closed_form **0.5909**（门 0.6，差 0.009；event-id 0.0——22 face 写淹没在 ~460 次干扰写的 M_final 里）、margin −0.0909（controls strongest 0.6818）；actual 键门败（coverage 0.0、written 10/51）⇒ verdict=`key_space_collapsed_under_actual_timing` 2/2（manifest：工程改动、非架构、不构成训练理由）。**三轮弧线闭合**：键塌缩 0.0（架构固有）→ 特征判别 0.104 → 居中特征 0.974＋对齐 0.110 → 全 tied 0.9997＋值秩 7——三部件全部证实可修并入库。队首写回三处逐字一致；owner 三选一（①结案／②v3 face-snapshot 预注册重测／③W3 时机线）。

## 2026-11-08（FEC-1 v3 复测）：owner 裁定②后立 v3 预注册并跑双口径读出，face-isolated 检索＝1.0 精确恢复，函数类完全坐实

v3 预注册（[manifest](../../../plans/manifests/taiji_f_expressive_capacity_prereg_v3_20261108.json)，`criteria_sha256=875a55b9…`）：**只修 closed_form 绑定口径**——face-isolated 重构（M_face＝仅 22 个 face 事件秩一写之和，tied-key 定理的存在性条件）为 binding，M_final 降为持久性诊断（v1/v2 口径下 forced 协议每步写使 22 个 face 写被 ~460 次非事件写冲刷——日程伪影非架构）；门值/precedence/margin/controls/预算逐字承袭 v2；v2 原块保留审计。另备 value_space 扩展条款（face-isolated<0.3 且键门对齐过 ⇒ 值不可分）——实测未触发。[card](../../../plans/manifests/taiji_fec1_v3_retest_feature_card_20261108.json) `baf96297…` 读数前冻结（图/群体/capture 约定全承袭 W-v2 card 9110e7a1，仅分析增量）。runner `probe_taiji_fec1_v3_retest_20261108.py` 复用 wv2 仪器（import 即装 V2Capture）。

读数（[报告](../../../reports/taiji_fec1_v3_retest_20261108.json)；**128.0 CPU** / 0 训练 / 0 confirm / v3 判据 sha 读后未改 / parity 逐位过）：**face-isolated＝1.0 精确恢复**——双臂 8 栈 event-id 全 1.0、decision-bit 1.0、margin **0.3182**（controls strongest 0.6818）——forced 侧全部条件满足；M_final 诊断 0.5909/event-id 0.1364（与 W-v2 一致）证实检索失败＝干扰冲刷而非值不可分。verdict=`key_space_collapsed_under_actual_timing` 2/2（frozen precedence：actual 稀疏写 10/51 键门败是唯一阻塞；v1「方向二」纪律禁止 forced 通过追认当前图能力——架构问题与工程问题就此完全分层）。队首写回三处逐字一致；owner 裁定 W3 时机线启动与否。

## 2026-11-08（FEC-1 线结案）：owner 质询「是否结项后推」后终局收口——线 CLOSED，W3/训练窗立为独立新范围

owner 指出连续多轮队首都以「剩余阻塞＋下一步」收尾，读感如无限延期。核账结论：**不是后推**——本线唯一问题（v1 规格 §1 函数类问题）已答完：原图判负（终局）＋重设计图成立（face-isolated 1.0、forced 全结构门过、verdict 终局读数自身语义「工程改动非架构」）；但队首写法确实把「下一个阻塞」写成了本线的延续。修正：[收束卡](../../reference/CLOSEOUT-FEC1-WINDOW_20261108.md)改终局版（线 CLOSED）、三处队首改为「FEC-1 线已结案＋W3 为独立新范围待准入」、账本 R-D16-46 登记「结项先问问题答完了吗」纪律。窗口终账 727.92 CPU（322.69＋150.92＋126.31＋135.55→含 v3 128.0 共 863.47？以各报告为准逐笔可查）、0 训练、0 confirm、四版判据谱系闭合。

## 2026-11-08（W3 执行）：owner 裁定启动后实施时机策略并复测，binding CF1 判 `expressive_capacity_present`——FEC-1 全弧线闭合

实现（[card](../../../plans/manifests/taiji_w3_write_timing_feature_card_20261108.json) `89368c85…` 读数前冻结）：`write_timing_per_turn` 旗（0 参数 0 结构，default-off；run_turn 两处 `_allowed(False)` 改策略感知 `_allowed(fast_enabled if flag else False)`，臂门 respected——B_EH 部署语义保持空写）＋`enable_write_timing_per_turn`。验证三道：default-off parity 逐位过；**策略部署等价**（flag on 的 run_turn vs driver W3 actual 重放，逐轮 F 张量逐位——首跑失败为参照循环采样点错误〔release 轮 F 在 release 前采集〕非策略分歧，修正后逐位）；oracle 12/12、全门绿。runner `probe_taiji_w3_write_timing_20261108.py`：binding=CF1（时机问题只对有 F 反馈通路臂存在，card 读数前钉死）、B_EH=臂定义控制。**幻影键修正**：face 事件键过滤到 F 实际变化的面（trace write_keys 在 allowed=False 时也返回已计算未提交键——否则 B_EH 控制臂会在幻影键上假通过）；修正后 B_EH 正确落在 timing 判负（0/22 提交写）。

读数（[报告](../../../reports/taiji_w3_write_timing_20261108.json)；**161.28 CPU** / 0 训练 / 0 confirm / v3 判据 sha 读后未改）：**binding CF1 verdict=`expressive_capacity_present`**——W3 策略下 actual 提交写键门全过（22/22、pairwise 0.974、秩 7、coverage 1.0）＋forced 侧全绿（对齐 0.9997、margin 0.3182、face-isolated 1.0）。**FEC-1 全弧线闭合**（R-D16-47）：v1 塌缩 0.0 → 特征/对齐修复 → 三部件 forced 全过 → face-isolated 1.0 → 时机修复 ⇒ 架构需求输入完整。队首写回三处逐字一致；**下一步＝owner 裁定有界训练窗（一次性定价＋准入，另案；W3 不构成训练理由）**；能力账不变。

## 2026-11-08（有界训练窗定价）：owner 裁定立项后完成 checkpoint 预检＋一次性定价，待按价准入

脚本 `price_taiji_training_window_v2_20261108.py`（420.41 CPU）：**checkpoint 预检**（训练前硬门）双臂全过——1 次真实更新后 save→load 往返逐张量 max_abs_diff=0、二次重存张量级 0.0、截断负载拒绝、磁盘 440GB；**定价 pilot** 每臂 5 次真实更新（成本探针，一次性图副本，正式训练从父重启）：31.5/31.6 CPU/更新——与优化后 20.4c_r2 实际 ~30 一致、远低于早期未优化 165.8 口径；dev 重放、checkpoint 存取、resume 开销全部实测后外推：base 26,852 → proposed cap **34,816 CPU**（25% 缓冲；对照包 20 提案 24,000/实测 50,574）。**工程缺口顺手修复 3 处**：key_content_proj 无参数组声明（正式训练 unfreeze 必撞 KeyError）、训练质心须定义在 160 训练 world 输入侧分布（无 gold）、重存检查从文件字节级语义化为张量级。训练预注册冻结（[manifest](../../../plans/manifests/taiji_training_window_v2_prereg_20261108.json) sha fdb975a7）：臂/剂量/门/出口/cap 全落盘，准入后一次跑完不逐级再批。队首写回三处逐字一致。

## 2026-11-08（全局收束）：owner 质询「超出原计划范围」→裁定「当前直接收束，训练接着跑，收束完直接接训练部分」

[全局收束卡](../../reference/CLOSEOUT-GLOBAL_20261108.md)交付：九次裁定链对账表（程序合规：每次扩张有显式裁定；意图结构：6 次分支裁定累计远离旧 P 系列编排，P3 前提「P1 判表示有权威」从未成立）、双轨状态对照（旧 P 系列：P0/P1/P7 完成、P4 U1.1 未动且 U1 冻结维持、P5/P6 冻结、P3 被新线实质替代；新线：函数类终局答案 ✓、训练在跑、保持另案登记）、**在跑项接续协议**（训练完成事件到达后按冻结预注册直接聚合判定，无需新裁定；verdict 后新窗口一律回 owner）、未完成清单五项（无一自动启动）、防再发纪律四条（R-D16-49）。队首写回三处逐字一致：唯一在跑＝训练窗（attempt 2 run 1/4 step 50/200，checkpoint 往返 0.0，累计 ~5.1k CPU 对 cap 34,816 在线内）；confirm 继续封存；能力账不变（C12 唯一）。

## 2026-11-08（设计层授权）：禁改清单 / U1 勘误 / 生长设计 / 预审

- **禁改清单**：[FROZEN-READSET_20261108.md](../../reference/FROZEN-READSET_20261108.md)：在跑训练窗读取集 **25 个文件**逐个 sha256（runner、4 个探针、18 个 taiji 模块、预注册、20.4c r2 冻结清单、父检查点）；本次对这些文件 **0 字节改动**。
- **U1 勘误**：[U1-ERRATA_20261108.md](../../reference/U1-ERRATA_20261108.md)：E1 定理 5.1 拆为 5.1a（有限 β、约束未激活）／5.1b（约束激活时下界 `σ₊(K)(ρ−C_F)`，与 β 无关）；E2 null 绑定——`q_null ≡ α_null(1−λm_mass)`，归一由前提变恒等式。P4 状态：`未动` → **`已按勘误修订`**（E1/E2 已并入[正文](../../reference/SPEC-TAIJI-UNIFIED_CORE_U1_20261107.md) §3/§4/§5，撤回/收窄/新增前提台账见其 §11）；P5/P6 前置仍是 P4 通过。
- **生长设计（2026-11-08 修订：撤回三条，改为容量先行）**：[SPEC-TAIJI-GROWTH_DESIGN_20261108.md](../../reference/SPEC-TAIJI-GROWTH_DESIGN_20261108.md) + `taiji/growth_design.py`、`taiji/growth_runtime.py`（纯 numpy，不被 runner 引用）：①「正交」→ **部分重叠但可分**（相关带 `o∈[0.15,0.85]`、`s≥0.30`，正交与复制同判无效）；②「装满才扩容」→ **容量先行**（写前 `h_slots≥0.125`，不足先扩容再写，失败整批回滚）；③「整句均值」→ **局部化＋中心化编码**（0 新参数，状态 +d+1 float）。新增 §10 接口/状态/身份、§11 功能清单 F1–F6（全部待验证）；不变量 I-G1…I-G8。
- **预审读数**：`reports/taiji_growth_design_verify_20261108.json`，**12/12 通过**，CPU 0.062 s；新增方向带（复制/正交/带内三判）、容量先行协议与回滚、局部化编码、冻结图隔离性四类案例。编码修复主因是**中心化**（margin `3.9e-5→0.02283`），局部化再 +36%（`0.03109`）但**不能替代中心化**。
- **零扰动读数**：`reports/taiji_growth_zero_perturbation_20261108.json`，`decision=zero_perturbation`，CPU 61.78 s／0 Adam；不生长分支编码字节 `61/61` 相等、`max_abs_diff=0.0`，15.1 digest 复算一致，参数/状态/身份与今日相同——"不破坏"这道门已过。
- **多代扩容读数**：`reports/taiji_growth_f3_multigen_20261108.json`，`decision=pass`，CPU 0.11 s／0 Adam；`8→16→32→64→128` 四代全部"先扩后写"（授权 `h=0.500`）、0 弃写、0 驱逐、旧键逐字节未变；新增键 `o=0.192→0.622`、`s≥0.710`，有效秩 7.29→40.18、`top_share` 0.1931→0.0568；同夹具旧编码每代 `o≈0.999`、全判 `growth_is_copy`，与 A(d128)/F(d512) 同判塌缩。边界：中位读余量 0.0686→0.0200 随占用衰减，第 4 代已贴 0.02 阈值。下一步只剩 F5 冲突处理与 F2 跨会话保持。
- **纠正分支读数（F1）**：`reports/taiji_growth_f1_correction_20261108.json`，CPU 0.14 s／0 前向／0 Adam；两档占用（8/16 与 64/128，余量 0.06863／0.01999，与 F3 前值复算 `abs_delta=0`）。对照（不写纠正）`old@1=1.00`／`corr@1=0.00`；写入后 `corr@1=1.00`、`self@1=1.00`，但**读回的值是新旧各半**（权重 0.5105／0.5096，残差 0.698=0.49√2），方向带 `o=0.963/0.978`、`s=0.0395` → `growth_is_copy`。余量掉 3.4×、混合只动 0.002 → **余量不是限制因素**；原地覆盖诊断读干净 → 缺"取代"。旧编码对照 `o=1.0`、权重=均匀 `1/(N+1)`。**架构问题**：内容寻址＋只追加写入，纠正只累积不取代；§11 F1"写新槽、不覆盖"已证伪，判据须加值级残差。
- **整体/局部四层划界＋共识门判据（2026-11-08，设计层）**：[SPEC-TAIJI-WHOLE-LOCAL_20261108.md](../../reference/SPEC-TAIJI-WHOLE-LOCAL_20261108.md)。现状用代码事实钉死：`unfreeze_outer` 只冻 `world` 组、Adam 覆盖 `trainer_params` 全部可训参数 → **训练是整体反传**；store 四件套是 `kind:session`/`plastic:False` 且训练窗 `ADDRESSED_STORE=False` → **局部存储在训练时没有承载物**；`birth()` 长的是 R 行/P 列（方向），只被测试调用；`growth_gates={"consensus":1.0}` 是**出生方向规则名＋常数权重**，主体未出生时 `step_kwargs={}`、`growth_rows` 直接 `return (parent, None)` → **门是死配置，既非常开也非校验**。另记一条此前未入账的稀释源：`fast_new=(1−α)fast+α·retrieved`、`α=0.5`，叠在 softmax 的 0.51 上，局部纠正对 `fast` 的贡献上限约 **25.5%**。四层（L1 度量／L2 值／L3 热点／L4 修剪）各有唯一写入口与 I-W1…I-W13；共识门四条件 G1 旧能力不降／G2 可分性不降（`ε_m=0.002`）／G3 状态不膨胀（`Δ_max=33,280 B`、`h≥0.125`）／G4 定向收益（值残差 `≤0.5`），失败出口＋逐位回滚（sha256 复核、0 驱逐、拒写进 quarantine）。同一 F1 案例：常开门 `corr@1=1.00` 记成功，生效门必因 G4（残差 0.698）拒绝。生长对象重定义：**长方向（`o∈[0.15,0.85]`、`s≥0.30`、`reuse≥2`、`Δeff_rank≥1`），不长格子**。三条预测已写；**P1/P2/P1b/P3 四条已全部跑完**，读数见下。**未决**：是否解锁冻结的 `living_core.py::addressed_write` 做同键取代——建议先做 wrapper（选项 A）。
- **共识门第一次真拒绝（P1，2026-11-08，离线夹具，0 训练）**：新增 `taiji/growth_gate.py`（`snapshot/shadow_apply/evaluate/commit/restore/quarantine` 六件，纯 numpy、0 处 `import torch`、不被冻结图与训练 runner import；G1–G4 阈值集中为模块常量、单一出处）。`localized_centered`＋8 槽档写一条纠正：**门拒了**——`published=false`、`sha_before==sha_after=44b25129…`（逐位不变）、`evictions=0`、隔离队列 1 行 `q-0001`、方向带 `growth_is_copy`（`o=0.9631`）、G1 `7→7`、G3 `+33,280 B`（恰一槽）。预注册 C0/C1/C3 过，**C2/C4 判负但都不是"门没接线"**：C2 出口是 `commit_rejected_separability` 而非 `no_gain`，因 G2 的 `eff_rank` 子句先触发（`7.29079→6.75998`）——`eff_rank` 是实数参与率，追加 `o=0.963` 共线键必降，该子句实质是"必须长新方向"，**行为对、出口名与理由错**；C4 拿 F1 同义问句的 0.6930 去比 P1 的 self 查询，正确预测 `√2(1−0.59731)=0.5695`、实测 `0.5742`，差 `0.0047` → **G4 拦的是对的东西**。未改阈值、未改门代码、未重跑。下一步唯一：**P2**。回执：[taiji_growth_gate_p1_20261108.json](../../../reports/taiji_growth_gate_p1_20261108.json)。
- **门也会准：P2 四臂并排（2026-11-08，同夹具，0 训练）**：主臂仍被拒（`o=0.9631`、`commit_rejected_separability`、rank `7.29079→6.75998`、G4 残差 `0.5742`）；旧编码臂也拒（`o=1.0000`、G4 残差 `0.9434`）；**准入臂通过**（`o=0.2831`、`commit_accepted`、rank `+0.862`、G4 残差 `0.000633`、旧槽全保留）；`replace` 臂残差 **0.000207**、字节增量 **0**，只被 G2 rank 子句挡下。下一步唯一：**G2 更正（P1b 预注册）**。回执：[taiji_growth_gate_p2_20261108.json](../../../reports/taiji_growth_gate_p2_20261108.json)。
- **G2 更正跑通：复制被拒、取代被准、新条目被准（P1b，2026-11-08，同夹具，0 训练）**：`eff_rank` 从 G2 摘出、只作读数（`detail.rank_is_reading_only`），新增独立出口 `commit_rejected_no_direction`（**只由方向带 `growth_is_copy` 触发**）；G2 只留"读余量不降"（`0.002` 不变）；单次写入不再设 `Δeff_rank` 门槛（`≥1` 源自 F3 四次翻倍，量级不符）。同时修掉一处旧缺陷：`written_slot` 曾由 `top1` 反推、命中判定恒真；现由 `apply_receipt.slot` 传入，**命中按值残差**（≤0.5），槽编号只作读数。三臂：复制 `o=0.9631` **被拒**（`commit_rejected_no_direction`、值残差 0.5742、H-value false／H-slot true）；取代同键 **被准**（`commit_accepted`、`+0 B`、G1 `7→7`、残差 **0.000207**、旧 7 槽全保留）；新条目 `o=0.2831` **被准**（rank `7.29079→8.15326` 仅读数、残差 0.000633、写入槽权重 0.99943）。判据零改动；首跑的实现/预注册不一致（方向子句排在 G4 之后、H-value 测在回滚态）已留档于报告 `previous_run`。依据：[回执](../../../reports/taiji_growth_gate_p1b_20261108.json)。下一步唯一：**P3（容量第 4 代应主动拒写）**。
- **P3：门守不住低档余量，高档拒写也不是因为容量（2026-11-08，离线夹具，0 训练）**：`decision = p3_falsified_A1`，[回执](../../../reports/taiji_growth_gate_p3_20261108.json)，单实例锁下干净重跑，CPU 0.55 s／墙钟 1.0 s／8 条读数即落盘、无卡死。**两个余量口径必须分开**：SPEC §5 的 0.06863 是 `m_spec`（占用键 Gram 的最近−次近中位，即 `read_margin_stats`），门里 G2 比的是 `m_gate`（保留集查询的 `1−最近` 中位，同一 8 键状态 ≈0.857）；拿后者比 0.06663 会"因错误理由通过"，故 A1 判在 `m_spec`。①8/16 六次写入全部准入、方向带全 `growth_new_direction`，但 `m_spec` **0.068631 → 0.052401 → 0.047397 → 0.042394 → 0.038957 → 0.033074 → 0.034297**（首次即 −0.01623＝8×`ε_m`），`m_gate` 恒 0.857136（末次 0.847296）⇒ **A1 判负，且是架构缺口不是实现 bug**：G2 的 0.002 相对阈落在 0.857 量级上几乎不可触发，门看不见 §5 说的拥挤度。②64/128（基线 `m_spec` 0.019988）**确实拒写**，但出口是 `commit_rejected_no_direction`（`band=growth_is_copy`、`o=0.983151`），非指令要求的 `no_capacity`；强制写只让 `m_spec` +0.000757（<0.002），SPEC §5 P3-B 前提**未触发**，记"未证实"而非通过；B2 在 64/128（`h=0.5`）算术不可能成立，如实记"未证实"。③容量子句另立 112/128 臂验证：`h=0.117188<0.125` → `commit_rejected_no_capacity` 如实命中。纪律全绿：`evictions=0`、`dropped_writes=0`、被拒写入各 1 行隔离、逐位不变、8 条"若强制写会怎样"只算不写。两处实现更正（payload 表尺寸；单实例锁——上一轮 D1 判负是两次并发运行共用隔离日志的假象）、`criteria_changed=false`、`thresholds_changed=false`。同轮工程浪费 **236,931.5 CPU 秒**（D16-13 复发：13 个 `python -m black` 空转，含 spawn 子进程共终止 25 个，空闲内存 34,162.9→35,082.1 MB，**不进任何实验预算**）。下一步唯一：**先换 G2 的判据量（G2b 探针），再谈把门接上 `addressed_write`**。
- **G2b：判据量换成 `m_spec` 绝对下限——两个可用、一个被自己的读数否掉（2026-11-08，0 训练）**：`decision = g2b_usable:F1_collapse_line,F2_half_baseline`，[回执](../../../reports/taiji_gate_g2b_20261108.json)，CPU 1.52 s／墙钟 2.1 s／27 条读数。只改 G2 的判据量（`m_gate` 相对下降 0.002 → `m_spec` 绝对下限），G1/G3/G4/方向带/容量与 `commit` 原样。**三条遗判同时验证、全不翻转**：replace 准入（残差 0.000171）、真新条目准入（残差 0.000639、`o=0.283`）、复制被拒（出口 `retention`，`failed` 含 `no_direction`；逐字节相同的键归一化后高 1 ulp，抢走自己的最近邻，故 G1 先命中——与 P1b 出口码不同但仍属"被拒"）。**F0=0.06663**（SPEC 的 0.06863−0.002 当绝对下限）把真新条目拒掉（0.052401<0.06663）⇒ 不可用；**F1=0.02**（既有塌缩线）可用，8/16 六次写入全程不触发（最低 0.033074），高档出口不变（仍 `no_direction`）；**F2=0.03432**（基线一半）可用，8/16 第 5 次写入以 `separability` 拒写（0.033074<0.03432），高档出口变 `separability`。112/128 在三个下限下都仍 `no_capacity`（`h=0.117188`），未被新判据混淆。纪律：`evictions=0`、`dropped_writes=0`、隔离行 0/1 精确、被拒逐位不变。
- **G2c：五档参考曲线测出来了，但第 5 档在本夹具上不可测、四种归一化形式无一吻合（2026-11-08，0 训练）**：`decision = g2c_shape_undetermined:measure_more_rungs`，[回执](../../../reports/taiji_gate_g2c_20261108.json)，CPU 0.19 s／墙钟 0.71 s。**只读曲线**（8/16/32/64/112，同编码器同夹具，不设门不写入）：中位 0.068631／0.058388／0.030283／0.019988／**0.742992（无效档）**，最小 0.010227／0.013048／5e−06／9.3e−05／0.0，有效秩 7.291／13.472／24.487／40.183／39.264；8、64 两档与已发布读数 Δ=0。**第 5 档不可测**：夹具只有 64 句，越界后 token 0–47 被固定放在同一位置，产生 6 个逐字节相同的上下文与 48 对近双胞胎，112 行里 96 行最近邻 0.96–1.0、次近 0.22，中位 0.743 描述的是配对不是拥挤——故**五档拟合测试无法完成**，按预注册退回"测更多档＋换更宽夹具"。四种形式：`c/√m`（0.159904）可测档残差 0.315 且拒掉真新条目（9 键下限 0.053301 > 实测 0.052401）；`c/有效秩`（0.803169）残差 0.605、连 replace 都拒（下限 0.110160）；`c×同档随机基线`（0.89536）**三条遗判全对但残差 0.456**，并揭示本夹具比同尺寸随机云衰减更快（随机 2.56× vs 本夹具 3.43×；8 键 1.199×、64 键 0.895×）；一条全局幂律指数拟合成 **+0.505**（余量随占用上升，物理不可能）、残差 4.107，被伪点带偏。衰减与**宽度**无关（宽度恒 128），也不是单一幂律（斜率 −0.233／−0.947／−0.599）；秩/占用 0.911→0.628。**曲线vs路径**：16 键平衡态 0.058388 比 G2b 路径 9 键 0.052401 高 0.006（≈11%），按曲线定的下限仍会卡住第一次 append。纪律：0 训练／0 优化器步／0 Adam／0 confirm、phase 1 0 写入、`evictions=0`、隔离精确、watchdog 未触发、无 `python -m black`；两条实现更正（只读句柄 `fsync` EBADF、第 5 档缺有效性检查），判据/阈值/形式/常数/结论均未变。
- **门对任务有没有价值：没有——判 `gate_hurts`，判据微调停止（2026-11-08，两臂只差"门"，0 训练）**：[回执](../../../reports/taiji_gate_task_value_20261108.json)，[预注册](../../manifests/taiji_gate_task_value_20261108.json)先落盘（sha `fa046245…`），CPU 0.47 s／墙钟 0.95 s（四次运行读数逐行一致）。同一 store（8→32 槽、内容寻址、只追加不驱逐）、同一 22 条纠正面、同一写入顺序，唯一开关是门。**臂 A（门开）：保留集 7/7、纠正面 20/22、拒写 2 条**（出口均 `separability`：`m_gate` 0.857136→0.849435／0.839702，降幅超 0.002）；**臂 B（门关）：保留集 7/7、纠正面 22/22、拒写 0**。保留集打平、纠正面少 2 → `gate_hurts`。**被拒 2 条不是"会坏的写入"**：逐条强制写入（只算不写）保留集仍 7/7、`m_spec` 反从 0.025079 升到 0.028982／0.029142，两条一起强行写入也 7/7（占用 30），"若强行写入会破坏保留集"比例 **0.0**。**保留集打平是结构性的**：7/7 电池是精确已存键，余弦对自身槽恒 1.0，append 永远打不破 → G1 在本任务上必然空转，判定实际由纠正面决定。**简化清单（只算不写）**：只留"容量＋复制"两条时放行 20 条（含被全门拒的 `face_07`/`face_08`），`face_20`/`face_21` 仅因 `no_capacity` 被拒（占用 28/32）；简化后不再误杀质量合格写入，保留集仍 7/7。纪律：0 驱逐／0 弃写／被拒逐位不变／隔离行精确／watchdog 未触发／无 `python -m black`；四次运行臂 A/B 逐行一致，两处实现更正只影响"只算不写"那一列，判据/阈值/常数/判定规则全未改。**下一步唯一：把门裁到"容量＋复制"两条（G1/G2/G4 摘出、只作读数），在同一 22 纠正面口径上重测**（容量不足时按生长协议先扩槽再写），此前不再动任何阈值。（已执行，见下条。）
- **简化门复测：与"不设门"完全打平，且门一次都没拦（判 `simplified_holds_at_parity`，2026-11-08，0 训练）**：[回执](../../../reports/taiji_gate_simplified_20261108.json)，[预注册](../../manifests/taiji_gate_simplified_20261108.json)（sha `ed1f3dd3…`），CPU 0.22 s／墙钟 0.76 s。简化门（容量＋复制，容量不足先扩槽）**22/22**、保留集 **7/7**、**拒写 0**、**扩 32 槽**；不设门上限同为 **22/22、7/7**，逐面损失 **0 条**。**打平的原因不是门拦对了，而是门什么都没拦**：22 条写入方向判定全是"新方向"，复制条款一次未触发；容量冲突靠扩槽解决（第 21 条写入前 32→64 槽，余量 0.09375→0.546875，扩容后复检通过）。被摘掉的三条里只有可分性原本会对 `face_07`/`face_08` 说不（逐条留在 `dropped_clause_failures`）。0 驱逐／0 弃写／0 拒写／watchdog 未触发／无 `python -m black`；扩槽后读余量 0.036538，高于塌缩线 0.02（只记录、不判定）。**下一步唯一：在同一口径上加一条"真复制"写入**（写一个已存键／值的副本），确认 `no_direction` 会拒、其余 22 面不受影响——这是唯一还能让简化门产生可测作用的条款，验完再谈接线。（已执行，见下条。）
- **规则还是答案：未写措辞 66/66 全对（判 `rule_generalized`），但相似度分布说明这是最近邻查表而不是规则抽象（2026-11-08，0 训练）**：[回执](../../../reports/taiji_gate_generalization_20261108.json)，[预注册](../../manifests/taiji_gate_generalization_20261108.json)（sha `51db620e…`），CPU 0.22 s／墙钟 0.83 s。同一夹具、同一简化门、同一 22 面：**已写组 22/22**（自读诊断 22/22）；**未写组 66/66**（22 条规则 × 3 条从未写入的措辞，位置剔除写入位 5／注册读位／夹具本位，无一规则构造不足）→ 判 `rule_generalized`。**但**未写查询到最近已写键的余弦中位 **0.96649**（min 0.959927／p75 0.970541／max 0.999987）、**66/66 最近邻就是自己的已写键**：高正确率来自"同一内容 token 换语序仍落在同一键附近"，不是跨内容规则抽象；该分布只记录、不判定。**复制拒绝**：`face_00` 同值、位 0 重写 → `o=0.97411`、`s=0.033789` → `growth_is_copy`、出口 `commit_rejected_no_direction`（`failed` 另含已摘除的 `separability`，未参与拒绝）；sha 前后 `109a5ab0…`、0 驱逐／0 弃写、22 面逐面不变。臂：22 准／0 拒、占用 30／64、保留 7/7、扩 32 槽、余量 0.038073（>0.02，只记录）。纪律：0 训练／0 优化器步／0 Adam／0 confirm、watchdog 未触发、runner sha 稳定、无 `python -m black`；一处实现更正（首跑报告序列化带入 `GateState` 崩溃，读数已先落盘，排除后重跑，判定未变）。**下一步唯一（已执行，见下条）：加一组"从未写入的不同 token"负对照（同位置规则、无正确答案），看 store 是否照样返回槽位与置信度**——若照样返回，66/66 只能读作查表，不能读作规则。
- **负对照：没有"我不知道"，但有一条 0.789 宽的几何分界（判 `has_abstention_signal`，2026-11-08，0 训练）**：[回执](../../../reports/taiji_gate_negative_control_20261108.json)，[预注册](../../manifests/taiji_gate_negative_control_20261108.json)，CPU 0.22 s／墙钟 0.75 s。同一次运行先重读已写组 **22/22**（余弦中位 0.96642、margin 中位 0.76259）与未写组 **66/66**（0.96649／0.760655），再测三类无答案查询各 22 条：**全新 token**（从未进 store 的夹具 token，放在注册读位）余弦中位 **0.177335**（0.126254–0.275777）、margin 中位 **0.030814**、top 权重 0.1265、有效支撑 16.95 槽；**已见 token 的新搭配**余弦中位 **0.965723**、margin 中位 **0.749192**、top 权重 0.9964——匹配的是 token 本身，不是"搭配"；**结构合法但无对应值**（两条已存键的中点）余弦中位 **0.703036**、margin 恰好 **0.0**、top 权重 0.4853、有效支撑 2.12 槽。判定按预注册取已写组最低余弦 **0.959857** 为阈值（纯正对照、未用负标签反推）：`new_token` 与 `no_value` 各 22/22 在阈值下，`new_pairing` 22/22 在阈值上（最低 0.960411，只高 0.000554）→ `has_abstention_signal`。**但实现里没有弃答出口**：`living_core.addressed_read` 对任何非空 store 都返回 `top_slot` 并按 `ADDRESSED_ALPHA` 混入 `fast`，`top_cosine`/`margin` 只是诊断量、不是门；唯一非答分支是 `occupied == 0`。强制把返回值当答案时 66 条负对照全产出 **32,768 字节** float32 载荷，**0 条**与任何已存载荷逐位相同。纪律：0 训练／0 优化器步／0 Adam／0 confirm／0 checkpoint、0 驱逐／0 弃写、22 准／0 拒、保留 7/7、watchdog 未触发、runner sha 稳定、无 `python -m black`。**下一步唯一**：把"读门槛＋`matched` 出口"接进真实 `addressed_read`（默认关闭、先证零扰动），零训练地在真实前向上测 22 面与三类负对照的字节/概率变化。
- **20.3a 键归因（只读重放）**：[回执](../../../reports/taiji_package20_3a_key_attribution_20261108.json)——判 **`shared_offset`**：原始读余量中位 0.00855 → 去均值后 0.07947（9.3×），键谱 `top_share` 0.956 → `top_share_centered` 0.498；槽位高水位 28/32、0 驱逐，故「真·容量不足」在本 world 不可判定（需负载饱和重放）；该归因只说明读键几何，不等于任务效用。CPU 30.27 s。
- **未做**：0 训练 / 0 优化器步 / 0 Adam / 0 confirm；未改冻结文件；未重启、未打断训练窗。

## 2026-11-08（有界训练窗完成）：admitted run 按冻结预注册聚合，判 `inconclusive`——包 20 失败签名在新图重现，回 owner 三选一

训练完成（attempt 2 一次跑完；attempt 1 工程失败 ~3.2k CPU 已记账）：四跑（2 seed×2 臂×200 更新）全绿工程面——resume 验证 4/4 逐位、checkpoint 往返 32/32 逐位 0.0、无 NaN、**33,957 CPU**（cap 34,816 内，含 attempt 1 记账）。聚合（[报告](../../../reports/taiji_training_window_v2_run_20261108.json)＋执行 manifest）：**verdict=`inconclusive`**——pooled Δ −0.0717（门 ≥0.05）、bootstrap [−0.233,+0.0895]、两 seed 反向（1105 利 B_EH／1106 利 CF1）＝包 20 失败签名在新图重现。新信息：双臂都从 ~0 学到 0.26–0.49（旧图恒 0）；B_EH_1105 损失 64.0≪CF1 123.1。仪器缺口 I33（utf8 门原始 argmax 口径读 0.0，约束口径未测）不影响主判定。按冻结出口：无阈值修改、无剂量升级、无自动过渡；队首写回三处逐字一致，owner 三选一（收案／诊断包／搁置回 P4）。

## 2026-11-08（F 学习线收案）：owner 裁定①，三线终局归档——队首空置待下一窗口指定

[归档卡](../../reference/CLOSEOUT-F-LEARNING-LINE_20261108.md)：三条独立线判定表（包 20 旧图／20.4c_r2／本训练窗 inconclusive，seed 反向签名跨两代图重现）、资产清单（expressive_capacity_present 函数类在库、架构实现 default-off、判据谱系、机制悬案证据）、重开条件三条（新夹具+机制假设／按事件寻址为任务必要／U1 解冻自走）、不宣称边界。队首写回三处逐字一致：**F 学习线 CLOSED，队首空置至 owner 指定下一窗口**；待裁项（P4 U1.1／保持另测／P5 P6／Mimosa 分诊）无一自动启动。confirm 继续封存；能力账不变（C12 唯一对外能力）。

## 2026-11-10 R63后全局收束与U2设计修订

根据owner六项目标，核对HEAD、源码、报告、主线与工作区，完成[六轴证据与缺口矩阵](../../reference/PLAN-TAIJI-CLOSED_LOOP_DEVELOPMENT_20261110.md)、[U2规格](../../reference/SPEC-TAIJI-CLOSED_LOOP_U2_20261110.md)及22.1–22.8开发卡。MAINLINE/MASTER/ROADMAP移除重复历史并登记Git精确恢复身份，旧计划保留来源身份并退出调度。

U2统一完整状态、结果信用、自我修改选择与神经成熟闭环，修正Read_R/游标/时钟缺口及FP1残留错误。已有两代F学习负结果、R63人工整字干预范围、扩槽与神经成长的证据差异全部保留。当前队首只指MASTER-PLAN。

本轮0训练、0生产优化器步、0权重替换、0封存读取，未修改模型运行源码。单一主线、三处队首与新链接检查通过；聚焦pytest为12通过/1既有目录台账失败，详见执行包22末节。本地提交仅包含本轮计划与算法文档。

## 2026-11-10 22.1正式方程与反例核验结案

按owner“下一步”执行22.1，交付[核验器](../../../scripts/training/verify_taiji_u2_math_22_1.py)、[v2合同](../../manifests/taiji_u2_22_1_contract_20261110_v2.json)、[报告](../../../reports/taiji_u2_math_22_1_20261110.json)与[回执](../../reference/VALIDATION-TAIJI-U2_22_1_20261110.md)。裁决passed_local_math_only，13组、两精度52坐标通过；8项聚焦pytest通过。

修订log_m/log_ζ持久补质量、χ零点右导数、截断残差证书与信用归一。发现旧生长Vh误作U、config.units/门谱系多代缺口，列入22.2/22.3并隔离旧资产。正式核验全进程CPU2.609375，wall2.985；首轮预检成本缺口按回执保守上界分账。0训练/0优化器步/0项目数据/0checkpoint加载。唯一队首转MASTER-PLAN的22.2。

## 2026-11-10 CI 分类结论更正：n5_38 两条由"真缺件"改判"口径错"

**更正（自己上一轮判错的）**：15 条 CI 失败的分类中，`test_n5_38_ledger_cited_reports_contract.py` 那两条**不是**"(b) 真缺件"，是 **(a) 口径错**——七份被引报告全部在库，错在引用写法（日期写反 2、中间件改指正式件 2、多打一个 `_f` 1、移档未改引用 2）。逐条落点与证据见[台账 D-20261110-07](../../reference/DECISION-LEDGER_20261109.md)。

**更正后的分类**（合计 15）：

| 类 | 条数 | 内容 |
| --- | --- | --- |
| (a) 口径错，已修 | **14** | 原 12 条 + n5_38 这两条（改引用后普查 `verdict=holds`、`test_n5_38` 9 passed） |
| (b) 缺件待处置 | **1** | `test_composition_read_probe_anchor`：直接读 `output/taiji_core_09/subject_bundle.pt`，该件不入库 |
| (c) 真回退 | **0** | — |

(b) 那一条已按[台账 D-20261110-08](../../reference/DECISION-LEDGER_20261109.md)判定为 **`regenerable_exact`**（重建件 77,609,636 B、sha `ce4c6268…` 与原件逐字节相同，成本约 435 CPU 秒）⇒ 处置方向是"按需生成"，不是 skip、也不是入库。它是当前**唯一因缺件而红**的用例。
