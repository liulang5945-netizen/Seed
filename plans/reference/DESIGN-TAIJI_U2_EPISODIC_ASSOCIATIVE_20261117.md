# 22.5.X 情景与联想记忆轴设计卡（有界、跨会话、线索提取；人类记忆优点）

> 卡号：`22.5.X`｜日期：2026-11-17｜状态：**设计＋重划范围已预注册（§9）；零训练可行性已获 owner 授权，帽 120 CPU／240 wall**｜性质：22.5 的长期记忆新轴，**不是** W.1.1 工作记忆的延续，也不是无限上下文。
> 依据：owner 2026-11-17 定位——「我对这个记忆的定位是情景记忆和联想记忆，即人脑的优点」；[W.1 工作记忆结果](RESULT-TAIJI_U2_W1_MEMORY_20261117.md)（4 槽有界，仅作前置）、[22.5 全局复核](REVIEW-TAIJI_U2_22_5_GLOBAL_20261117.md)。
> 纪律：本卡只做**零训练设计**；数据构建、可行性探针、定价另立阶段；**0 训练、0 封存读取**。

## 1 单问题

W.1.1 的 4 槽工作记忆只能支撑单 episode 的延迟匹配，**不构成跨会话经验**。本卡问：

> **在有限容量、有限上下文的约束下，怎样让同一主体跨会话形成可提取、可纠正、可遗忘的**情景记忆**与**联想记忆**，并让这些记忆实际参与推理/决策/表达？**

不是"把上下文无限拉长"；不是"存一个向量库"；不是 W.1.1 的 4 槽缓冲。

## 2 两种记忆的能力定义（可验收口径）

### 2.1 情景记忆（episodic）

- **内容**：具体事件 `(what, where, when, who/self, source, confidence, owner/scope)`；
- **形成**：工作记忆中的当前 episode 经**选择性写入**进入有界长期 store；可巩固、可重放；
- **提取**：由**线索**（实体/地点/时间/上下文/部分内容）触发，回忆**具体事件**，不是只给统计摘要；
- **特性**：时间顺序、来源/证据、置信度、可重建（不要求逐字节）、**遗忘/衰减**、干扰；
- **跨会话**：同一主体身份下，下一次会话仍可提取并纠正旧事件。

### 2.2 联想记忆（associative）

- **内容**：实体—属性、实体—地点、事件—结果、共现/因果关系；
- **形成**：跨 episode 的共现/反馈强化，形成有界关联权重；
- **提取**：**内容寻址/模式补全**（部分线索→完整模式）、相似性泛化；
- **特性**：竞争/干扰、强度衰减、巩固（重放/睡眠式离线）、可纠正；
- **泛化**：未见组合上的合理联想，不得只记训练对。

### 2.3 共同边界（必须写进判据）

- **有界**：store 容量、关联数量、每次提取预算都有硬上限；**不承诺无限上下文**；
- **选择性**：写入/巩固/遗忘由策略决定，不是把所有输入都存下来；
- **来源与证据**：每条记忆带来源/时间/置信度，可核验、可纠正、可标记冲突；
- **可恢复**：store 与模型状态可独立保存/恢复；
- **可审计**：写入/提取/遗忘/纠正都有日志。

## 3 架构选项（先设计，不实现）

- **A（建议先做）＝有界情景 store ＋ 联想图 ＋ 策略层**：外置/旁挂的有界 store（记录级）＋ 联想图（实体/属性/事件边，带强度）＋ 写入/巩固/提取/遗忘/纠正策略；单位核心 B 图负责当前编码与读出，store 负责跨会话记忆。可先零训练验证"能否存/取/纠正"，再训练策略。
- **B＝体内长期可塑单位**：新增长期可塑单位/连接、离线巩固/重放、休眠/唤醒；上限更高，但属于新结构，需单独结构卡与恢复设计。
- **C＝混合**：A 的 store ＋ B 的联想层；先 A 后 B。

**共同约束**：不把"无限上下文"当方案；不把 W.1.1 的 4 槽缓冲冒充长期记忆；不把向量检索命中率当情景记忆。

## 4 任务与数据（先合成，后自然中文）

- **情景任务**：多会话事件序列（实体、地点、时间、动作、结果）；会话后用线索（实体/地点/时间/部分内容）回忆具体事件；含未见组合、干扰事件、冲突来源、纠正回合。
- **联想任务**：跨会话学习实体—属性/地点/结果关联；部分线索→补全；未见组合泛化；干扰对/反事实。
- **反馈**：真实对错/来源可信度；允许"不知道/求证"；不允许把失败洗成成功。
- **分区**：train/fit/check 按事件组合与线索划分；fit 含未见组合；check 另立、本卡不读。
- **自然中文**：本轴先用合成/受限中文；自然连续任务归 23，不在本卡冒充。

## 5 判据（预注册草案，读数前冻结）

**情景门**：线索回忆 ≥0.80；what/where/when 三元组 ≥0.75；跨会话保持（下一次会话仍可提取）；来源核验 ≥0.80；冲突/纠正后旧值不再冒充当前；置信度校准（高置信错误率低于低置信）；遗忘曲线可测；容量上限内。
**联想门**：模式补全 ≥0.80；未见组合泛化 ≥0.70；干扰/反事实控制；巩固后增益；容量/竞争可测。
**工程/边界**：有界 store、写入/提取/遗忘日志、独立恢复逐位、成本 ≤实测帽、无封存泄漏。
**出口**：`episodic_associative_feasible`／`task_not_learnable`／`instrument_invalid`／`budget_stop`；训练另立。

## 6 零训练可行性（下一步）

- 构建小规模合成情景/联想数据（含未见组合、干扰、来源、冲突）；
- 用当前单位图/提议 store 做**零训练**编码/检索探针：当前表示能否承载 what/where/when 与关联？不能则先设计 store/关联层，不训练；
- 仪器：编码 oracle、分区 disjoint、正对照、置换/干扰控制、参数不变；0 优化器步；
- 预算：自设帽 **120 CPU／240 wall**；0 封存。

## 7 能签与不能签

**能签**（后续阶段）：情景/联想任务定义、数据边界、判据、可行性读数、成本。**不能签**：模型已经会人类情景/联想记忆；无限上下文；W.1.1 等于长期记忆；任何能力/父/check②/22.6。

## 8 当前状态与唯一后继

**当前**：设计完成；**0 训练**；W.1.1 保持冻结、**不训练**（仅工作记忆前置）。**唯一后继**：零训练可行性（情景/联想数据＋编码/检索探针＋定价）→ 按 §5 出口分类 → 若可行，写 `22.5.X.1` 冻结契约＋定价，再请 owner 授权训练；若不可行，先设计 store/关联层或换任务。裁决前 0 训练新增、0 能力、0 父；check② 不读、22.6 不解锁。

## 9 重划范围与预注册判据（owner 2026-11-17 授权甲；**读数前冻结**）

### 9.1 授权与范围

owner 2026-11-17 弹窗裁决＝**「授权＋按缺口重划（推荐）」**：X 零训练可行性阶段获授权，帽 **120 CPU／240 wall**、**0 优化器步、0 封存读取、check 分区不读**；范围＝只测**五项真空白**（地点通道、记录级置信度＋校准、冲突/纠正、遗忘曲线、跨会话门）＋**单位图↔记忆面桥接**，**已有读数的项不重测**。同一弹窗另裁「推送这 19 笔」，已按常设规则先打印 `git log --oneline origin/main..HEAD` 逐条列清单再推，rc **0**、范围 **`204c2f7c8..ece6bebee`**，推后 HEAD 与 `refs/remotes/origin/main` 同 **`ece6bebee`**、未推 **0**；**该授权用完，新提交未获推送授权**。

### 9.2 先于判据的静态事实（逐条自证，不是转述）

- **联想场默认开**：`taiji/model.py:195` 无条件 `self.memory = EpisodicField(...)`；`observe` 走 `recall`（`model.py:2458`）／`write`（`model.py:2213`）。
- **记录级 store 默认关**：`taiji/config.py:348` `episodic_memory_default_mount: bool = False`，容量 `config.py:350` `episodic_memory_capacity: int = 1024`。⇒ 旧项目记忆「有库的是没挂载的记忆」**部分过期**：场是默认挂载的，记录级 store 才是默认关。
- **无衰减**：`config.py:358` `memory_confidence_decay: float = 0.0`；`taiji/episodic_memory.py` 全文 `decay`／`forget` 各 **0** 命中。
- **记录无置信度字段**：`taiji/contracts.py:1945-1961` 字段全集＝`memory_id, episode_id, tick, cue, action_intent, outcome, world_transition, prediction_error, provenance, event_ids, assembly_ids, object_ids, relation_ids, attribute_ids, version`。置信度只住在 `MemoryState.episodic_confidence`（`contracts.py:1124`，**单一标量**、`_check_unit` 约束在 [0,1]）与 `WorldAffordance.confidence`（`contracts.py:920`）。
- **无冲突/纠正 API**：`episodic_memory.py` 全文 `conflict`／`correct`／`supersede` 各 **0** 命中；但 `write` 对**同 `memory_id` 会替换**（`:106`），`expire` 可显式撤销（`:115-130`）⇒ 纠正有两条现成机械路径，冲突标记没有。
- **打分只用 cue**：`retrieve` 的余弦打分在 `:237-252`；`episode_id`／`object_id`／`attribute` 是**合取过滤不是重加权**（`:194-209` 原文明写「scores are computed from `cue` only」）。
- **无会话门**：`episode_id=None` 默认搜全库（`:196-198` 原文「the storage layer still has no session gate, and DEBT-G106 stays open」）。
- **容量 FIFO 驱逐**：`:109-113`；`evicted_count` 与 `_expired` 是**两列不许合并**（`:38-40`）。
- **checkpoint 往返**：`:258-289`，format `taiji-episodic-memory-v1`；对象/属性派生索引**刻意不持久化**（`:42-49`），载入时由 `write` 重建。
- **where 无专用字段但有开放通道**：`WorldObject.attributes`（`contracts.py:817`）、`WorldEvent.attributes`（`:863`）、`EpisodicMemoryRecord.attribute_ids` 写作 `"名=值"` 串（`:1958-1960`）⇒ `place=X` **今天可写、可寻址**，但**不进打分**。
- **单位图侧编码无 place/time/entity 通道**：`taiji/u2_unit_input_dms.py:98` `rgb_layout="[4 attribute one-hot, present, test marker, sample marker, bias]"`、`:99` `question_layout`；16 列×8 维、6 通道（`:19-23`）。
- **两条链不相交**：`taiji/u2_*.py` 共 **36** 枚，对 `episodic_memory|semantic_memory|EpisodicMemoryStore|EpisodicField` **零命中**（全目录现数）。

### 9.3 已有读数（**不重测**，引用并注明代际）

- `reports/taiji_m5_episodic_field_20260821.json`：`status="pass"`、`metrics.full_field.action_accuracy=0.875`、`metrics.full_field.provenance_accuracy=1.0`、`metrics.full_field.mean_time_code_cosine=0.5125222061760724`、`metrics.recurrent_association_lesion.action_accuracy=0.25`、`checks.recurrent_completion_is_causally_necessary=true`。
- `reports/taiji_p4_episodic_recall_baseline_20260825.json`：`gate.passed=true`、`conditions.full.action_recall=1.0`、`conditions.checkpoint_continuation.action_recall=1.0`、`conditions.retrieval_lesion.action_recall=0.0`。
- `reports/taiji_g5_episodic_write_four_grid_v3_20260926.json`：四臂 `strict_hits` 全 **0**、`arms.W1R1.write_strength_mean=0.889001`、`gaps_vs_W0R0.W1R1=0` ⇒ 写入发生了、读取零增益。
- `reports/taiji_episodic_read_sharing_face_20261011.json`：`read_site_count=11`、`reads_in_product_default_chain=1`、`write_site_count=44`；**其 `verdicts.retrieve_signature_has_no_episode_param=true` 已过期**（现 `retrieve` 带 `episode_id`/`object_id`/`attribute`，`episodic_memory.py:190-192`）⇒ **只登记不代改**。
- **代际限制**：以上取自 2026-08～10 月的面，与本卡 2026-11-17 的单位图面**不同源**；不得当作本卡判据的读数，也不得与本卡读数相减。

### 9.4 被测两面与桥接

**Face S**＝`EpisodicMemoryStore`（记录级长期 store）；**Face G**＝`RouteBGraph`＋`unit-input-dms-v1`（当前单位表示）。桥接问题＝**G 的特征能否承载 what/where/when**。
**where 的口径限定**：DMS 的 `position` 是**测试行列位**（0..7），**不是场景地点**；场景 place 通道在编码里不存在（§9.2）。两者分开报，**不许互换**。

### 9.5 冻结判据（读数前写死）

**Face G**（3 seed＝`2026111701/02/03`，沿用 W 面同 seed 以便对照）：

- **主判据 `triple_fit_min = 0.75`**：fit 上 what ∧ where ∧ when 三者 argmax **同时正确**的比例。
- what＝sample 属性 4 类 one-vs-rest；where＝position 8 类 one-vs-rest；when＝delay ∈ {0,1,2,4,8} 5 类 one-vs-rest。训练/评分沿用 W 的 **train 64／fit 32** 组合划分，fit 含未见组合。
- **逐字段准确率是披露读数不是门**（判负要能定位到字段）。**头塌缩（`fit_distinct<2`）同样是披露不是仪器红**——真实特征上塌缩是关于表示的发现，尺子的量程由 oracle 臂与错排臂建立，不由真实特征臂建立。
- 正对照 **`positive_triple_min = 0.80`**（oracle 特征＝sample／position／delay 三个 one-hot 拼接）；错排对照 **`deranged_triple_max = 0.50`**。
- **尺子动态范围**＝正对照过 ∧ 错排在带下（**两端都要走到**）。

**Face S**（五项，每项必须两端都走到）：

- **S1 地点**：`place_narrowing_exact=1.0`；`place_in_cue_top1_min=0.80`；`place_attribute_only_top1_max=0.50`；两臂差 `place_arm_gap_min=0.30`。
- **S2 置信度**：记录无置信度字段（静态断言 dataclass fields）；用 `retrieve` 分数做代理。**`calibration_ordering_holds`（高置信半区错误率严格小于低置信半区）是门读数不是仪器检查**——"分数自信地错"是一条真发现，不许把它判成仪器坏。**尺子量程由统计量自身建立**：① oracle 正对照＝把分数换成"对＝1.0／错＝0.0"后同一统计量必须给出严格排序；② `errors_exist`（错误数 > 0）；③ 置换对照＝**20 次确定性置换**（`random.Random(STORE_SEED+trial)`）的**平均**绝对差 `permuted_ordering_abs_diff_max=0.10`（单次置换抽样噪声太大，判据落在均值上，最大值只披露）。三者合取即 `ruler_has_dynamic_range`。**刺激＝竞争性部分线索** `(1-α)·onehot(place) + α·onehot(other)`，α∈**0.05–0.60**，记录噪声尺度 **0.05–1.55**。**更正留痕（读数前）**：初版只用"干净线索＋记录噪声"，首次 smoke 实测 **error_count=0／128**⇒高低半区错误率都是 0、"严格小于"恒假、尺子没量程；故改为竞争性部分线索并把量程判据从"真实排序成立"移到"oracle 统计量成立＋确有错误"。
- **S3 冲突/纠正**：默认臂 `contradiction_persists=true`（两条都在、无冲突标记）；同 id 覆盖臂与 `expire` 臂各 `corrected_top1=1.0` ∧ `stale_reachable=false`；默认臂与纠正臂**必须不同**。
- **S4 遗忘**：大容量臂 `retention_flat=true`（最大干扰下保持率与零干扰**逐位相等**）；小容量臂 `evicted_count_exact=true` ∧ `retention_drops=true`；`store_has_no_decay=true`（静态）。
- **S5 跨会话**：`checkpoint_roundtrip_exact=true`（记录 `value_digest` 相等）；`cross_session_retention_min=0.80`；`gate_off_leakage=true`（默认 `retrieve` 取到别会话记录）；`gate_on_exclusion_exact=1.0`。

**仪器**（任一 uncovered ⇒ `instrument_invalid`）：编码 oracle、分区 disjoint、图参数不变（`value_digest` 前后相等）、图优化器步 0、store 记录契约往返、每项尺子两端、正/错排对照。

**判据门与仪器检查分两本账**：`checks` 只装仪器项（进 `uncovered`），`gates` 装判据读数（驱动出口 3／4，**不进 uncovered**）。把判据门混进 uncovered 会让**真负结果被错判成 `instrument_invalid`**——首次 smoke 实测撞到（`triple_reaches_gate` 进了 uncovered），已按本条分开；同理"头塌缩"与"分数不校准"都是发现不是仪器红。

### 9.6 出口（互斥、按序、穷尽）

1. **`budget_stop`**：`cap_cpu`／`cap_wall` 先于全面完成被超。
2. **`instrument_invalid`**：任一仪器检查 uncovered。
3. **`task_not_learnable`**：**任一 seed** 的 triple fit **< 0.75**。
4. **`episodic_associative_feasible`**：全 seed triple **≥ 0.75** ∧ 五项 store 读数各自两端都走到。

**"feasible" 的口径**：只签「当前单位表示能承载事件三元组 ＋ 五项缺口已被精确定位（含按构造缺席）」，**不签**模型已会情景/联想记忆、不签无限上下文、不签 W.1.1 等于长期记忆。五项里按构造缺席的（记录级置信度、冲突/纠正 API、衰减）是**设计工作不是训练工作**；判 feasible **也不等于它们已实现**。

### 9.7 预算与纪律

设计帽 **120 CPU／240 wall**（owner 授权）。实价用**批量计时**测（单次调用会落在 `process_time` 分辨率约 **0.0156 s** 之下＝计时抽签，r2 已踩过并撤回过一个 1.5 CPU 的假帽）；帽＝实测×2.0 且**不得超设计帽**；带配额循环留收尾 reserve（停止判据 `elapsed >= allowance - reserve`）。**0 优化器步（图）、0 权重写入、0 封存读取、check 分区不读**；线性头是**诊断器**不是产品训练。复用 W 探针的 `make_episode`／`make_dataset`／`dataset_digest`／`neutral_stage`／`fit_head`／`positive_control` 与 W.1 的 `run_memory_episode`／`features` 形状，**不重抄生成链、不改被复用 helper 一字节**。单价锚点（同机同 helper 实测）：W 可行性 45 次 LBFGS 拟合＋39 次 features 调用（各 96 episode）＝**21.4531 CPU／21.688 wall**；W.1 同结构＝**16.1875 CPU／16.485 wall**。

### 9.8 未测与限制

- 不测自然中文连续任务（归 23）；不测选项 B（体内长期可塑单位，新结构另卡）。
- 联想层只登记既有 `EpisodicField` 的静态在场与既有读数引用，**不新测模式补全**（m5 已有 0.875＋因果必要）。
- 本卡是**测量不是干预**：不动产品默认、不改编码合同、不改被复用的 helper。
- where 的场景地点通道缺席是静态事实；本卡只量「开放属性通道可寻址但不打分」这一后果，**不设计新通道**。
- 判负不可归因到单一变量：本卡同时看两面五项，任一出口只签**范围级**结论，逐项定位靠披露读数而非出口本身。
