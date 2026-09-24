# R2 架构级立项：**复制回路（copy circuit）——「先告知→后提问」的结构性缺口与最小架构增量**（2026-09-25）

状态：**停在所有者决策点**（§7）。前件：`M5_R2_ARCH_LEVEL_PROPOSAL_20260925.md`（三方案甲/乙/丙）＋
`M5_R2_T8_MECHANISM_DESIGN_20260925.md`（分裂诊断设计）。所有者裁定（2026-09-25）：**「R2 显示架构有问题，
需要先解决架构的问题」**——本件把该裁定落成可实施的架构设计。

## §0 一句话

T8 的「(c) 召回后解码有罪」建立在**零向量注入**上（无效判定）；A0 重判在**真实写入 28 事件、
非零内容反馈、寻址确实在召回**的条件下复证了 (c)——**F1 读出没有任何「由召回内容发射答案字节」的通路，
这不是训练能解决的：可学习的通道必须先存在**。最小架构增量＝给语言回路装一条**复制回路**（写入门＋内容直读 copy 通道），
数学形态在本仓 `sequence_content_workspace.py` 已有可证先例。

## §1 证据链（全部实测；新增三条为 2026-09-25 A0）

| # | 发现 | 证据 |
|---|---|---|
| 1–9 | （承前件 §1：槽结构只在区 0／稀释非抹掉／读法修正不解决答对／CAP 四模型全 0／t1·t5 F1 从未写入／A+掩码仍 0／三协议探针无召回回声 0／归纳 0/30） | 前件 §1 表 |
| **10** | **T8 判定无效**：其 Exp-1 反馈范数全 0.0、forced 与 zero 文本逐字相同——写入段用 `readout="predictive"` 从不触发 act/settle ⇒ `write_count` 恒 0 ⇒ 注入的是零向量 | `reports/taiji_r2_recall_split_20260925.json`（norms 全 0）＋`taiji/model.py:1815-1829/2173-2174/2245-2255`（写路径唯一经 action 读出） |
| **11** | **写回路与语言回路在 episode 级互斥**：同一 dynamics episode 内禁止切换 readout（action 写入→predictive 生成必须跨 reset） | `taiji/model.py:1752`（`readout changed inside an active dynamics episode`——A0 首跑实测撞出） |
| **12** | **A0 重判（base_16M 与 A_16M 一致）**：Phase W 真写入 28 事件（反馈范数轨迹 0.12→0.37）；**native 开闸臂生成期真实召回反馈每 tick 非零在进**（0.16–0.39，两模型逐位相同＝A 臂只动 F1，记忆/fabric 权重同 base，合理）⇒ **(b) 寻址是活的**；但强制非零臂与 native 臂均解不出「阿岩」，且 forced 与 zero 文本**已不同**（反馈确在改变动力学）⇒ **(c) 真有罪：内容召回得到、进不了字节发射** | `reports/taiji_r2_recall_split_v2_20260925.json`＋`scripts/training/diag_taiji_r2_recall_split_v2.py`（判读冻结于 docstring） |

## §2 架构缺口清单（每条挂事实；这就是「架构的问题」的完整定义）

- **G1 语言回路无写入门**——情景记忆唯一写路径＝`observe(readout="action")→act→settle_action→pending→下一 observe 写入`；
  语言训练/生成全程 `predictive`，**「被告知的内容」没有任何合法入口进入情景记忆**。
  （`model.py:1815-1829/2173-2174/2245-2255`；A0 Phase W 是绕道证明：只有走 action 才写得进。）
- **G2 episode 级读写互斥**——写入段与生成段不能同 episode（`model.py:1752`）⇒「先告知→后提问」在**协议层**就不连贯。
- **G3 读出无内容通道（本立项的正主）**——F1 的输入＝本 tick 区 0 状态经固定受体压到 48 维＋私有残差
  （`organs.py:301-310/657-668`、`model.py:2031-2033`）；576 维 `cortical_feedback` 只以 **×0.25 增益进下一 tick 各区 drive**
  （`fabric.py:309-327`、`model.py:1997`）；F1 当前 tick 可见的召回物只有 257 维**加性证据**（`memory_read_gain×confidence×action_evidence`，
  `model.py:2064-2075`＋`organs.py:705-708`）——是「记忆押哪个字节」的标量偏置，**不是可复制的内容**。
  且反馈本身是**回归出的皮质状态向量**（`memory.py:402-406`），不是干净字节序列。A0 判据 12 实测：此形态的通道承载不了复制。
- **G4 协议闸门关死（必要非充分）**——`chat→generate_input→generate` 链不暴露 `use_memory`，内部硬编码
  `learn=False, use_identity=False`（`api/seed_runtime.py:317-322`、`model.py:2900-2916`）；`learn_bytes` 默认
  「长期情景场保持损伤态」（`model.py:2611-2615`）。A0 Exp-N 已证：**只开闸答不对**（闸门是 G3 的下游）。
- **G5 身份器官不存内容**——cue→action/outcome 符号三元组＋余弦寻址（`identity_organ.py:384-396`、`cue_binding.py:63-72`），
  无字节/文本字段。D/E 的题不经过它。

**可复用资产（避免重造）**：① `SequenceContentWorkspace` 的 **copy 混合分布**——
`vocab softmax ⊕ C-conditioned copy distribution`，参数 `copy_query_state/copy_query_content/copy_gate_weight/copy_gate_bias/copy_induce_bias`
（`sequence_content_workspace.py:21-22/295-300/610/638/680-686`）＝真复制的数学形态，**本仓内已有且经 R2 训练验证**，只是没接进 native 回路；
② `CueBindingBank` 竞争寻址槽；③ `PROVENANCE_KINDS=("experienced","imagined","replayed","external")`——
「被告知」可先复用 `external` 源码，不扩枚举不动 one-hot 维度。

## §3 修复设计：复制回路（A2）四步，每步独立可验收

**总原则**：新参数**门控初始化 gate≡0 ⇒ 与旧模型逐位不变**（本仓「参数位级不变分账」纪律）；旧 checkpoint 直载新代码。

- **A2.1 语言写入门（解 G1/G2）**：predictive 回路上新增 `record_told_content(bytes, cue_context)` 类入口——
  把**告知段的字节序列＋段末皮质 cue** 存进情景内容存储（新 store，不塞 `EpisodicField` 的三元组权重——G3 已证回归向量承载不了内容）；
  provenance=`external`。诊断/训练脚本先调，产品协议（chat 的 system/告知轮）接线放 A2.4。
  **验收**：写入后 `content_store.count>0`；旧行为零变化（新 store 默认无读者）。
- **A2.2 内容直读 copy 通道（解 G3，本立项核心）**：F1 概率改为
  `softmax(vocab_logits + copy_gate·copy_dist)` 形态：query＝F1 context（48 维）＋区 0 状态；
  keys/values＝召回事件的**字节序列**（A2.1 的存储）；gate＝`sigmoid(w·[state;content_match]+b)` **init=0**；
  copy_dist＝内容位置的匹配打分（照抄 `sequence_content_workspace` 的 `_copy_weights/_copy_distribution` 数学）。
  新参数面＝5 个小张量（与 workspace 同型）。**结构存在性判据（零训练）**：A2.1 写入「我叫阿岩。」后，
  在 ask 位强喂该事件 ⇒ copy_dist 对「阿岩」首字节 argmax 命中 ⇒ 通道存在性成立。
- **A2.3 召回条件发射训练（原 M-3 升格为「训新通道」）**：问答结构语料（`simple_zh/dialogue_extended_clean.jsonl` 已有 问：/答： 形）
  ＋ teacher-forcing 下 copy 通道参与损失（gate 与 query/key 可学；fabric/既有 F1 冻结或分组学习率——细节在实施预注册里冻结）。
  A0 判据 12 同时**修正了 M-3 的前提**：没有 A2.2 的通道，M-3「教模型从召回发射」无物可学——**甲单独走已被证伪，并入本步**。
- **A2.4 协议开闸（解 G4）**：`chat` 链透传 `use_memory=True`＋告知轮走 A2.1；`generate` 内部 `use_identity=False` 维持
  （身份不存内容，开了也无用，G5）。

**依赖顺序**：A2.1→A2.2（存在性判据）→A2.3（5M 判读 ≈5.5 h，16M 定论 ≈22 h）→A2.4。A2.2 的存在性判据**零训练**，
是全线最便宜的证伪点——它红了，整个 (c) 判定再翻一次案。

## §4 判据（先冻结，实施预注册引用本表）

1. **结构存在性**（A2.2，零训练）：强喂内容事件 ⇒ copy_dist 命中告知字节 argmax；gate init=0 时输出与旧模型**逐位一致**。
2. **端到端**（A2.3 后）：CAP D+E 机器计分 **> 0**（对照：同 checkpoint 未接 A2 时 0/36，T6 已钉）；副读数成句率。
3. **回归门**：CAP A/B/C 维不劣化；native-readable 表层字节分布不劣于基线（平台边界）；`learn_bytes` 既有合同测试全绿。
4. **消融**：gate≡0（冻结）≡旧模型逐位；copy 通道置零 ⇒ 回落到 G3 现状（0 命中）——证明增益确由新通道携带。

## §5 与既有方案的关系（收敛，不并存旧路）

- **T8 结论修正**：「走 M-3 不需架构改动」作废——其 (c) 判定虽被 A0 复证，但 T8 的**推理路径**（零向量注入＋predictive 写入）无效；
  M-3 单独走无物可学（§3.A2.3）。T8 文档与 `52ce2078` 提交结论以本件为准（历史不改，判读件互指）。
- **甲/乙/丙三方案**：甲并入 A2.3；乙＝本件（获依据，正式立项）；丙（重表述 CAP）**不采**——D/E 是产品要的「被告知后回答」能力，
  换题等于放弃该能力维。
- **与 M4 成长纪律的关系**：A2.2 是**加性新参数**（非结构生长事件），走「同最终容量对照＋保持/恢复门」的既有分账，不动 G 环。

## §6 风险（如实）

- **copy 通道的 keys 是字节序列，不是激活**——这是与 `EpisodicField` 全部现有语义的一次决裂（内容存储第一次进记忆系统）；
  容量/淘汰策略先取最简（每 episode 最近 K 条，K 冻结在预注册）。
- 训练窗口与 R2 排队冲突（5M≈5.5 h/16M≈22 h）；A2.1/A2.2 实施与存在性判据**零训练**，不受此限。
- gate 学不开（A2.3 后仍 0）＝「通道在、归纳学不会」——那将是比现在**精确得多的否定**（有通道的否定），届时再裁。

## §7 决策点（本件请求的批准）

**批准 A2.1＋A2.2 实施**（零训练、gate init=0 逐位不变、含 §4.1 结构存在性判据）；A2.3/A2.4 在 §4.1 绿后按预注册另行开闸。
不批＝停在 A0 证据上（架构缺口已钉死，但 CAP D+E 维持 0）。
