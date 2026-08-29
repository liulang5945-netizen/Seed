# 架构妥协审计：改进方向与候选路线

> 本文由原总路线图按职责拆分而来。原始行号：966–1672；本文件属于当前计划资料或历史证据，具体身份以本文件所在目录为准。
> 这是原审计的上限提升排序、方向 B 和后续候选路线部分。

## 三、上限提升潜力排序（Top 10）

| 排名 | 妥协点 | 类型 | 提升幅度 | 实施难度 |
|------|--------|------|---------|---------|
| 1 | S1 共振从未被端到端训练 | 系统性 | 协作涌现 +30-50% | 高 |
| 2 | S2 256K emb 配 16K tokenizer | 系统性 | 词覆盖 +30-50% | 中 |
| 3 | S6 域 token re-encode 往返 | 系统性 | 推理速度 3-5x + 长文本 | 中 |
| 4 | S3 Loss 单一化 | 系统性 | 协作 +15-30% + 回答 +15-25% | 中 |
| 5 | S11 512 token 硬截断 | 系统性 | 长上下文能力 | 中 |
| 6 | S5 数据规模偏小 | 系统性 | PPL +30-50% | 中 |
| 7 | S9 生物学机制是占位 | 系统性 | 结构性容量 | 高 |
| 8 | S4 训练步数偏短 | 系统性 | 收敛深度 +20-35% | 低 |
| 9 | S12 多轮对话靠拼接 | 系统性 | 多轮连贯性 | 中 |
| 10 | S7 side_channels 全连接 | 系统性 | 效率 +40% 质量 +5-10% | 中 |

---

## 四、关键洞察

### 4.0 ★★★ **架构本源定性：涌现已存在，核心缺陷是自适应激活**（2026-08-04 认知重构）

**核心定性修正（2026-08-04 认知重构）**：之前把"单神经元较强"定性为"压制涌现的因素"是**根本性错误**。正确的认知：

**涌现的定义 = 单个无法完成 + 协作能完成。当前架构已满足这个定义——涌现已存在。**

**涌现的证据**（数据验证）：

| 指标 | 最强个体 | 5 协作 | 涌现证据 |
|------|---------|--------|---------|
| 历史协作 PPL | ~42（无法正常对话）| 32.9 | ✅ 协作显著优于最强个体 |
| 当前 E4 PPL | ~19.7（仍无法正常对话）| 15.4 | ✅ 协作显著优于最强个体 |
| EMERGE | — | 21.7% | ✅ 协作收益稳定 |

**关键事实**：单个神经元 PPL ~20-42（参考 GPT-2 small PPL ~30 勉强可读），**无法正常对话**；5 个协作 PPL 15-33，**能正常对话**。这正是涌现的定义——协作产生了单神经元不具备的能力（正常对话）。

**"单神经元较强"是优势，不是劣势**：
- 优势：协作层规模不需要那么大就能产生涌现 → **效率高**
- 对比方向 B：1-5M 小神经元需要巨大协作层才能涌现，效率低且未验证
- 类比：5 个有基础能力的人协作，比 100 个几乎无能的人协作更容易出成果

**人脑类比不是唯一标准**：
- 之前用"人脑协作:神经元比 1000:1"作为标准，把当前 0.5:1 定性为"偏低"——这是错误套用
- 人脑的 1000:1 是生物约束（神经元体积大、突触可密集生长），神经网络无此约束
- **不按人脑也未必上限不高**——涌现的关键是"单个无法完成 + 协作能完成"，不是参数比例

**唯一核心缺陷：自适应激活**（用户明确指出）：
- 当前协作层基本**稠密计算**（所有 side_channels + cross_spec 每次全部参与）
- R1 软路由提供轻度稀疏性，但不够强 → 不同样本应激活不同神经元组合
- **已有设计**：R1 共振分软路由 + C14 动态 shared_expert_weight + C9 自适应停止
- **需要强化**：让协作层针对不同样本（如数学问题 vs 日常对话）激活不同神经元子集
- 这是提升上限的关键方向，比"扩大协作层规模"更重要

**方向 B 的定位修正**：
- 之前认为方向 B 是"上限更高的方向" → **错误**
- 正确认知：方向 B 是"另一种涌现路径"（更小神经元 + 更大协作层），**不是上限更高的路径**
- 当前方案已验证涌现存在，方向 B 是未验证假设
- **方向 B 触发条件修正**：不再是"当前方案上限不足时的备选"，而是"探索另一种涌现机制的实验"，优先级降低

**参数分布真相**（5 神经元阵容，2026-08-04 精确核算）：

| 部件 | 参数量 | 占比 | 训练状态 |
|------|--------|------|---------|
| 神经元主体（backbone）| 338M | 49% | 冻结 |
| body 最后 2 层（微调）| 185M | 27% | 微调（lr×0.1）|
| shared_embedding | 131M | — | 冻结 |
| **协作层**（side_channels + cross_spec）| **167.8M**（side 25.2M + 投影 142.6M）| **24%** | 从头训练 |

**协作层规模扩展能力**（unified 放大效应，§6.1 详述）：
- side_channels：O(N²) 随神经元数增长
- cross_spec 投影层：规格升级触发 U 跳跃全局放大（历史 4C→4C+1S 协作层翻倍 +111%）
- 当前 0.5:1 的比例**不是缺陷**——单神经元较强意味着不需要巨大协作层，但需要时可通过规格升级 + 数量扩展提升

**决策时点（修正）**：
- **Step 1（进行中）**：zh 综合体 + EOS+短答案重训 → 验证能正常对话（涌现的输出验证）
- **Step 2**：加入 code/math/en 等特定能力神经元，测试跨域涌现（§4.0b 候选 1）
- **Step 3**：强化自适应激活（R1 软路由 → 真正的 top-K 稀疏路由），提升协作效率
- 方向 B 不再是"上限不足时的备选"，而是"探索另一种涌现机制的实验"，优先级降低

---

### 4.0b 涌现的深化探讨 + 新能力方向（2026-08-04 认知重构）

**定性修正**：§4.0 已确认**涌现已存在**（单神经元无法对话 + 5 协作能对话 = 涌现）。本节探讨已实现的涌现 + 可进一步探索的新能力方向。

**与单体大模型的本质区别**（修正）：
- 之前认为"协作层稠密 → 数学等价于大模型" → **部分正确但不完整**
- 正确认知：即使协作层稠密，**信息流路径不同**（多分支并行 + 场共振融合 + 跨规格投影 vs 单链 transformer）
- 更关键的是：单体大模型所有参数端到端训练，态极神经元主体冻结只训练协作层 → **协作层学到的是"如何协调已有能力"**，这是单体模型不具备的

**已实现的涌现**（修正：从"非涌现"重新定性为"涌现"）：

| 能力 | 机制 | 实测 | 涌现定性 |
|------|------|------|---------|
| **协作能对话，单神经元不能** | 多轮共振 + side_signals + 场状态注入 | EMERGE 21.7%，单神经元 PPL ~42 无法对话 | ✅ **核心涌现**（协作产生新能力）|
| **置信度加权融合** | C8 per-sample confidence + 场写入强度 | 高置信神经元贡献更大 | 涌现的支撑机制 |
| **跨规格信息融合** | cross_spec 投影层（compact 2048 + standard 3072 → unified）| 不同规格神经元优势互补 | 涌现的支撑机制 |
| **动态协作权重** | C14 shared_expert_weight + R1 共振分软路由 | 不同样本激活不同神经元组合 | 涌现的支撑机制（待强化）|
| **多轮深化推理** | 多轮共振（round 1 独立 → round 2+ 注入场状态）| 信息在轮次间累积 | 涌现的支撑机制（待挖掘）|

**核心涌现**是第一行：协作产生了单神经元不具备的能力（正常对话）。其他行是支撑这个涌现的机制，不是独立的涌现。

#### 真正的"新能力涌现"是什么（探讨）

涌现指的是**单个神经元不具备、协作后产生的新能力**。当前架构理论上可能涌现的新能力：

**候选 1：跨域类比推理**（可能性：中 → **历史已实验过，有实现可能性**）
- 机制：不同神经元学到不同领域的隐式表征，场共振让它们在统一空间碰撞
- 涌现表现：模型能做出"类似 X 领域的 Y 领域推理"（如用物理直觉解数学题）
- **历史实验结论**：早期用 5 个不同类别小神经元做过跨域实验，**PPL 显示有涌现迹象**
- **当时搁置原因**（关键）：
  1. 5 个不同类别小神经元训练数据**没有互通性**——单一领域数据不包含正常对话等内容
  2. 所有单神经元**无法对话** → 无法通过输出判断涌现（只能靠 PPL 间接判断）
  3. 缺乏"能正常对话的综合体"作为基底 → 涌现效果被"输出无法理解"掩盖
- **新验证路径**（用户提出的清晰思路）：
  - **Step 1**：当前 zh 综合体（5 神经元 + EOS+短答案重训）能正常对话（进行中）
  - **Step 2**：在已能对话的基底上，**加入特定能力的神经元**（如 code/math/en 神经元）
  - **Step 3**：直观测试涌现——观察加入新神经元后，综合体的输出是否出现新能力（如对话中能解答代码问题、数学推理）
  - 这比"5 个孤立域神经元硬凑"更直观，因为综合体本身能输出可读对话
- 当前是否实现：**历史 PPL 显示迹象，但未通过输出验证**。新路径待当前重训完成后启动

**候选 2：动态能力组合**（可能性：高）
- 机制：R1 软路由 + C14 动态权重，不同样本激活不同神经元组合
- 涌现表现：对未见问题，模型能动态选择最合适的神经元组合（而非固定路由）
- 当前是否实现：**部分实现**。R1 已接入训练，但单神经元能独立生成 → 协作非必需
- 关键阻碍：单神经元能独立完成任务时，动态组合的"涌现"退化为"可选优化"

**候选 3：场状态累积推理**（可能性：中高）
- 机制：多轮共振让场状态累积信息，类似"思考过程"
- 梯度流经过 side_signals + field_state，模型学到"如何利用场状态"
- 涌现表现：复杂问题需要多步推理时，场状态累积产生单步无法得出的答案
- 当前是否实现：**部分实现**。forward_train 全可微多轮共振已接入（S1 修复）
- 关键阻碍：n_rounds=2 太少，且训练数据都是单步问答，无多步推理样本

**候选 4：置信度校准**（可能性：高）
- 机制：C8 confidence + C12 共振分对比投影，模型学到"知道自己不知道"
- 涌现表现：对不确定的问题输出低置信度，而非胡乱回答
- 当前是否实现：**机制已实现，但未验证效果**。需要专门校准测试（ECE/Brier score）

#### 涌现的现状与提升方向（2026-08-04 认知重构）

**涌现已存在**（核心修正）：之前把"单神经元能独立完成基础任务"定性为"压制涌现"是错误的。

| 条件 | 当前状态 | 说明 |
|------|---------|------|
| 单神经元无法独立完成任务 | ✅ **满足** | 单神经元 PPL ~20-42，**无法正常对话** |
| 协作能完成任务 | ✅ **满足** | 5 协作 PPL 15-33，**能正常对话** |
| 协作层稀疏激活 | ⚠️ 部分满足 | R1 软路由提供轻度稀疏，**核心缺陷，待强化** |
| 训练任务需要协作 | ✅ 满足 | 对话任务单神经元无法独立完成 |

**核心修正**：
1. ~~"单神经元太强压制涌现"~~ → **错误**。单神经元有基础能力但无法正常对话，这正是涌现的前提
2. ~~"训练任务太简单不需要协作"~~ → **错误**。对话任务单神经元无法独立完成，必须协作
3. **唯一核心缺陷**：自适应激活不足（协作层稠密，R1 软路由需要强化为真正的 top-K 稀疏路由）

**提升涌现上限的方向**（修正）：
- ~~更小神经元~~ → 不需要，当前单神经元"有基础能力但无法独立完成"是最佳区间
- **强化自适应激活**（R1 → top-K 稀疏路由）→ **核心方向**
- **跨域神经元扩展**（加入 code/math/en 神经元）→ 测试新能力涌现
- **多步推理任务**（增加 n_rounds + 多步推理数据）→ 挖掘场状态累积推理潜力

#### 对当前方向的指导（2026-08-04 修正：跨域实验路径优先）

1. **当前方向价值**：工程层面的能力增强（协作 PPL 降低 21.7%）是真实的，值得继续优化
2. **涌现验证优先路径**（用户提出，比方向 B 更直观且可验证）：
   - **Step 1（进行中）**：当前 zh 综合体 + EOS+短答案重训 → 验证能正常对话
   - **Step 2**：加入 code/math/en 等特定能力神经元到已能对话的基底
   - **Step 3**：直观测试跨域涌现（对话中是否出现代码/数学能力）
   - **优势**：综合体本身能输出可读对话 → 涌现效果可直接观察，不像早期"5 孤立域硬凑"无法判断
3. **关键认知**：早期跨域实验失败不是因为机制无效，而是因为**缺乏能对话的基底**。现在有了能对话的基底，跨域涌现值得重新验证
4. **方向 B 的触发理由不变**：若跨域加入新神经元后涌现明显 → 当前架构可承载涌现，方向 B 暂不启动；若跨域加入后无涌现 → 才需要考虑方向 B（更小神经元 + 被迫协作）
5. **中间路径**：当前架构 + 更难任务（多步推理数据）+ 稀疏路由（R1 强化），可能部分触发涌现

#### 跨域 Step 2 数据准备（2026-08-05 梳理，工作3）

**目标**：为 code/math 特殊神经元训练准备混合数据 + 梳理接入流程（§4.0b 候选1 Step 2）。

**关键决策（用户指正）**：每个 neuron 保留自己的域 tokenizer（code 12K / math 10K），通过**词库转译**实现语义转换：
- 输入统一 general 256K 空间（[batch_align_and_embed](file:///e:/taiji-neuron/taiji/resonance/translator.py#L452) 用 general_sp 编码输入，目标用 domain_sp 编码）
- 推理转译用 S6 alignment_table（domain→general 预计算映射，[cortex.py:1198](file:///e:/taiji-neuron/taiji/brain/cortex.py#L1198)）
- 推理 forward 已支持不同 vocab（[ensemble.py:1263-1284](file:///e:/taiji-neuron/taiji/resonance/ensemble.py#L1263-L1284) `same_vocab` 检查，不同时走 neuron_logits 提取）

**混合数据策略（已验证可行）**：

| 数据源 | 规模 | 用域 tokenizer 编码 | 说明 |
|--------|------|--------------------|------|
| `data/sft/code_sft.pt`（CodeAlpaca） | 3000 条 | byte_ratio 7.3% ✅ | 英文代码指令-响应 |
| `data/sft/math_sft.pt`（GSM8K） | 3000 条 | byte_ratio 2.2% ✅ | 英文数学推理 |
| `data/sft/en_sft.pt`（英文 alpaca 对话） | 3000 条 | byte_ratio 2-7% ✅ | 混合对话能力 |
| `data/distill/code_texts.jsonl` | 36,810 行 | ✅ 英文 | 预训练风格扩充 |
| `data/distill/math_texts.jsonl` | 22,904 行 | ✅ 英文 | 预训练风格扩充 |

**结论**：code/math neuron 用各自域 tokenizer 训练，混合数据 = 域 SFT 数据 + 英文对话数据（en_sft），目标编码全部高效（byte_ratio 2-7%）。**不混中文对话**（code tokenizer 编中文 byte_fallback 57% 低效）；中文语义通过 general 256K 统一输入空间 + S6 转译在协作层处理。

**接入流程**：
1. 训练 code/math neuron 本体（P8-1 `train_neurons_from_scratch.py --domain code`，混合域 SFT + en_sft）
2. 加入综合体推理：forward 已支持（same_vocab 检查）
3. 协作层训练：**缺口 M 已修复**（见下）——`forward_train` 跨 vocab 融合
4. 跨域涌现评估：对话中测试代码/数学能力

**状态**：混合数据策略已验证 ✅（2026-08-05）；**缺口 M 已修复 ✅（2026-08-05）**；训练 code/math neuron 待当前 zh 训练完成后执行。

### 缺口 M 修复：forward_train 跨 vocab 联合训练（2026-08-05 实施）

**原问题**：`forward_train` 融合阶段要求所有 neuron vocab 一致（[ensemble.py:1673-1680](file:///e:/taiji-neuron/taiji/resonance/ensemble.py#L1673-L1680) 原实现），否则 `torch.stack` 崩溃——跨域协作层训练的前置阻塞。

**修复方案（词库转译矩阵投影）**：
- [translator.py](file:///e:/taiji-neuron/taiji/resonance/translator.py) 新增通用词库转译工具：
  - `tokenizer_fingerprint(sp)`：tokenizer 指纹（vocab_size + 首/中/尾 piece 抽样），用于缓存失效判断
  - `build_domain_to_domain_alignment(source_sp, target_sp)`：source token → target token 对齐（byte fallback 正确处理）
  - `build_logits_alignment_matrix(...)`：构建 [V_src, V_tgt] 稀疏投影矩阵（行归一化 1/N，logits 尺度守恒），带缓存 + 指纹失效
- [ensemble.py](file:///e:/taiji-neuron/taiji/resonance/ensemble.py)：
  - `set_tokenizer_hub(hub)`：注入 TokenizerHub（与 cortex 同源）
  - `forward_train` 新增 `target_domain` 参数；vocab 不一致时用转译矩阵把各 neuron logits 投影到 target 域空间再融合
  - 向后兼容：vocab 一致路径零开销（不传 target_domain 也可运行）
- [finetune_cross_spec.py](file:///e:/taiji-neuron/scripts/training/finetune_cross_spec.py)：forward_train 传 `target_domain=DOMAIN`

**词库热插拔（一并解决）**：
- S6 对齐表缓存（`_domain_to_general_cache`）原为一次性构建永不失效；现缓存项携带 tokenizer 指纹，tokenizer 被替换（重训/热插拔注册）后自动失效重建
- [cortex.py](file:///e:/taiji-neuron/taiji/brain/cortex.py) 新增 `invalidate_alignment_cache(domain=None)` 手动失效接口
- TokenizerHub.register_domain 本身已支持热插拔（新域注册不影响现有 neuron）

**词库可编辑可拓展层（AlignmentRules，2026-08-05 新增）**——匹配新增特殊神经元词表：
- [translator.py](file:///e:/taiji-neuron/taiji/resonance/translator.py) `AlignmentRules`：人工规则层覆盖自动转译，匹配键用 **piece 文本**（tokenizer 无关、可编辑，不用脆弱 token id）
- 支持域特定规则 + 全局规则（`"*"`）；每次增删递增 version → 下游转译矩阵/对齐表缓存自动失效
- 持久化 JSON（`save()`/`load()` 热加载），默认 `taiji/domains/alignment_rules.json`
- 接入：`ensemble.set_alignment_rules()` + `cortex.set_alignment_rules()`（S6 也支持人工覆盖）
- 新增特殊神经元时：注册 tokenizer + （可选）add_override 补专业术语映射

**跨域协作层训练脚本（train_cross_domain_collab.py，2026-08-05 新增）**：
- 多域 neuron（code/math/zh）联合训练协作层（side_channels + 投影层 + Sparse Router）
- 域轮转 + batch 级 `target_domain`，缺口 M 词库转译融合路径；`--rules-path` 挂载 AlignmentRules
- 自动匹配 neuron vocab 的 tokenizer（zh neuron 20K → `sp_zh_v20k.model`，防御 vocab 错位）
- 冒烟验证通过（verify_v3 多域 neuron 完整跑通训练循环 + checkpoint）

**验证**：`_smoke_cross_vocab_gap_m.py` 8/8 通过（转译构建/矩阵归一化/缓存复用/热插拔失效/跨 vocab 融合梯度流/向后兼容/override 覆盖/持久化/规则变更缓存失效）；真实 code→zh 转译验证：`def`→`['▁','▁def']`、换行语义保持 ✓，矩阵 [12000, 50000] 构建仅 0.1s。

### 4.0c ★★★ **自适应激活设计：R1 软路由 → top-K 稀疏路由**（2026-08-05 设计）

> 本节是 §4.0 确定的"唯一核心缺陷"的**具体设计方案**。用户要求"梳理，同时可以着手设计"，此处完成梳理 + 设计落地，待训练完成后实施。

#### 1. 自适应激活针对什么

**明确：针对输入样本（样本驱动）**，不针对装载硬件。

- **样本驱动**：不同输入（数学问题 vs 日常对话）应激活不同神经元子集 → 这是模型架构层面的自适应激活，本设计的核心。
- **硬件调度**：根据可用显存/算力动态调整激活数量 → 工程层问题，与模型架构正交，不在本设计范围内。
- **两者关系**：样本驱动的 top-K 选择是基础，硬件调度可在 top-K 基础上进一步调整 K 值（未来扩展）。

#### 2. 现有自适应激活机制盘点（梳理）

| 机制 | 路径 | 类型 | 局限 |
|------|------|------|------|
| C9 自适应停止 | 推理 `forward()` | 轮次级 | 只控制何时停止，不控制激活谁 |
| R1 共振分软路由 | 训练 `forward_train()` | 软加权 | **稠密计算**，所有神经元都参与，权重≈0 也算 |
| active_filter | 推理 `forward()` | 硬过滤 | 基于场方向拥挤度，非能力路由；H5 显示跨 embedding 空间不可比 |
| per_position entropy融合 | 推理 `forward()` | per-token软加权 | 0.01 floor，没有真正关闭神经元 |
| C14 动态shared_expert_weight | 推理 `forward()` | shared权重动态 | 只调整 shared vs domain，非神经元子集选择 |
| active_nids参数 | 推理 `forward()` | 外部路由接口 | 没有路由器实现，需外部指定 |

#### 3. 核心缺陷（三点）

1. **训练路径完全稠密**：`forward_train` 中 `active_ids = list(self.neurons.keys())`（[ensemble.py:1137](file:///e:/taiji-neuron/taiji/resonance/ensemble.py#L1137)），所有神经元参与每轮计算和融合。softmax 只是软加权（[ensemble.py:1390](file:///e:/taiji-neuron/taiji/resonance/ensemble.py#L1390)），即使某神经元权重≈0，它的 forward 计算仍然进行，算力浪费。

2. **训练-推理不一致**：训练用 soft softmax 全神经元融合；推理用 per_position entropy + active_filter 硬过滤。模型训练时从未见过"部分神经元被关闭"的情况，导致推理时分布偏移。

3. **没有样本驱动的路由器**：当前 scores 基于 field_state cosine（场聚合状态），不是"输入样本特征 → 路由决策"。H5 注释明确（[ensemble.py:1590-1595](file:///e:/taiji-neuron/taiji/resonance/ensemble.py#L1590-L1595)）：field.score() 跨 embedding 空间不可比，已被禁用。

#### 4. 设计：Probe-based Sparse Router（基于探针的稀疏路由器）

##### 4.1 核心思路

引入可学习的 Router，基于 **round 1 probe**（每神经元独立前向，已在 `forward_train` 中执行）的响应，为每个神经元产生路由分，选择 top-K 神经元参与 round 2+ 的深度协作。

##### 4.2 为什么选 Probe-based 而非 Input-based（上限优先）

| 方案 | Router 输入 | 额外开销 | 上限 | 选择 |
|------|------------|---------|------|------|
| Input Router | shared_embedding mean-pool | 零 | 中（只看输入，不看响应）| 备选 |
| **Probe Router** | round 1 field_vectors + confidence | 零（round 1 已存在）| **高**（看每神经元实际响应）| **推荐** |

**Probe Router 上限更高的原因**：它能看到"每个神经元对当前输入的初步响应"，类似人脑"先瞥一眼再决定谁深入处理"。round 1 独立前向已在 `forward_train` line 1203+ 执行，**零额外开销**。

##### 4.3 Router 结构

```
Router 输入（per-neuron）：
  - field_vector: [B, D_field]    # round 1 每神经元的场写入向量（已投影到 unified 维度）
  - confidence:   [B]              # round 1 per-sample 置信度（C8）
  - score_vec:    [B, D_score]    # round 1 评分投影向量（C12，若存在）

Router 输出：
  - routing_scores: [B, N]        # 每样本对每神经元的路由分
  - top_k_mask:     [B, N]        # hard top-K 选择（forward）
  - soft_weights:   [B, N]        # soft softmax 权重（backward 梯度流）
```

Router 实现：per-neuron MLP(`D_field + 1 + D_score → hidden → 1`)，对每个神经元独立评分，然后 batch 级 softmax + top-K。

##### 4.4 可微 top-K 选择（Straight-Through Estimator）

借鉴 Switch Transformer / GShard：

```python
# Forward: hard top-K 选择
top_k_indices = routing_scores.topk(K, dim=-1).indices
hard_mask = zeros(B, N).scatter_(-1, top_k_indices, 1.0)
# 被选中的神经元用 routing_scores 归一化后的权重
selected_weights = (routing_scores * hard_mask).softmax(dim=-1)  # 只在选中神经元上归一化

# Backward: 梯度通过 soft softmax 流回所有神经元
soft_weights = routing_scores.softmax(dim=-1)
# STE: forward 用 hard, backward 用 soft
final_weights = hard_mask * selected_weights + (soft_weights - soft_weights.detach())
```

- **Forward**：只有 K 个神经元参与 round 2+ 计算（算力节省）
- **Backward**：梯度通过 soft softmax 流回所有神经元（Router 可学习）

##### 4.5 负载均衡 loss（防模式坍塌）

升级当前 `balance_loss = -(weights * log(weights)).sum()`（负熵）为 Switch Transformer 风格：

```python
# f_i: 神经元 i 被选中的批次比例（hard, detach）
f = hard_mask.mean(dim=0).detach()  # [N]
# P_i: Router 对神经元 i 的平均概率（soft, detach）
P = soft_weights.mean(dim=0).detach()  # [N]
# 负载均衡 loss: N × Σ(f_i × P_i)，越小越均衡
balance_loss = N * (f * P).sum()
```

##### 4.6 K 值确定

- **起步**：固定 K=3（5 神经元中选3个 + shared_expert 始终激活 = 4 个参与 round 2+）
- **升级**：动态 K（基于路由分分布的熵，高熵→多选，低熵→少选）
- **shared_expert 处理**：general 神经元始终激活，不参与 top-K 选择（保证基础语言能力）

##### 4.7 Warm-up 策略（防冷启动）

Router 初始随机，可能选错神经元。Warm-up 分阶段：
- **Phase 0（前 10% 步）**：K=N（全选），Router 只学习评分，不影响激活
- **Phase 1（10%-30% 步）**：K 线性从 N 降到目标 K（如 5→3）
- **Phase 2（30%+ 步）**：固定目标 K，Router 完全生效

#### 5. 与现有机制的整合

| 现有机制 | 整合方式 | 理由 |
|---------|---------|------|
| R1 共振分软路由 | **替换**为 Router soft weights | Router 学习路由，比场状态 cosine 更直接；H5 已证明 field.score() 跨 embedding 不可比 |
| C9 自适应停止 | **保留**，与 Router 正交 | C9 控制轮次（何时停），Router 控制激活（谁参与）|
| C14 动态shared_expert_weight | **保留** | shared_expert 始终激活，C14 调整其权重，与 Router 正交 |
| active_filter | **替换**为 Router top-K | Router 是主动选择，active_filter 是被动过滤 |
| per_position融合 | **保留**作为 fallback | 在选中的 K 个神经元内进行 per_position 融合 |
| balance_loss | **升级**为 Switch 风格 | 负熵 → Switch 负载均衡，更稳定 |
| C12 contrastive_loss | **保留** | 约束 Router 评分与 NLL 排序对齐，让 Router 学到"能力路由"|

#### 6. 训练-推理一致性

| 维度 | 训练（forward_train）| 推理（forward）| 一致性 |
|------|---------------------|---------------|--------|
| Router 选择 | STE（hard forward + soft backward）| hard top-K | ✅ 一致 |
| 参与神经元 | round 2+ 只 K 个 | round 2+ 只 K 个 | ✅ 一致 |
| 融合权重 | Router soft weights | Router soft weights | ✅ 一致 |
| 负载均衡 | 训练时计算 balance_loss | 推理时不需 | ✅ 正常 |

**消除当前的训练-推理不一致**（训练稠密 vs 推理过滤）。

#### 7. 实施路径

##### 阶段1：Router 实现（不破坏现有训练）✅ 已完成（commit 3526274）
- [ensemble.py](file:///e:/taiji-neuron/taiji/resonance/ensemble.py) 新增 `SparseRouter` 类
- Router 输入：round 1 field_vectors + confidence + score_vec
- Router 输出：top-K mask + soft weights（STE）
- 负载均衡 loss（Switch 风格）
- 向后兼容：`use_sparse_router=False` 时退化为当前稠密模式

##### 阶段2：接入 forward_train ✅ 已完成（commit 3526274）
- `forward_train` round 1 后计算 Router 输出
- round 2+ 只对 top-K 神经元注入 side_signals + field_state
- 融合用 Router soft weights（per-sample STE）
- 新增 load_balance_loss 到总 loss（替换原负熵 balance_loss）
- smoke test 通过（forward/backward/归一化/梯度流验证）

##### 阶段3：接入推理 forward ✅ 已完成（commit 54e95e5）
- 推理路径同样用 Router 选择 top-K（round 1 后）
- 保证训练-推理一致（"激活谁"一致）
- 融合在 top-K 内 per-position（保留 entropy 融合）
- active_nids 参数与 Router 协同（外部指定优先，否则用 Router）
- [eval_dialogue.py](file:///e:/taiji-neuron/scripts/training/eval_dialogue.py) 自动检测 checkpoint 是否含 Router 状态

##### 阶段4：训练验证 ⏳ 待训练完成后
- [finetune_cross_spec.py](file:///e:/taiji-neuron/scripts/training/finetune_cross_spec.py) 新增 `--use_sparse_router` flag（已完成）
- 对比稠密 vs 稀疏的 EMERGE、PPL、推理速度
- 验证 warm-up 策略有效性

#### 8. 上限分析

| 维度 | 当前稠密 | 稀疏路由 | 提升 |
|------|---------|---------|------|
| 算力效率 | O(N) 每轮 | O(K) round 2+ | N=20,K=5 时 75% 节省 |
| 协作质量 | 所有神经元参与 | 最合适神经元深入 | 聚焦→质量↑ |
| 可扩展性 | 算力线性增长 | 算力对数增长 | 支持更多神经元协作 |
| 训练-推理一致 | ❌ 不一致 | ✅ 一致 | 消除分布偏移 |
| 路由可学习 | ❌ 场状态 cosine | ✅ MLP 学习 | 适应任务分布 |

#### 9. 风险与缓解

| 风险 | 缓解 |
|------|------|
| Router 冷启动（选错神经元）| Warm-up 策略（Phase 0 全选，逐步降 K）|
| 模式坍塌（总选同一组）| Switch 风格负载均衡 loss |
| 与场共振冲突 | Router 选择后场更聚焦（正面效果，非冲突）|
| shared_expert 惰性 | C14 动态权重已处理（共振弱→sw 高→shared 兜底）|
| K 值选错 | 起步固定 K=3，后续升级动态 K |

#### 10. 实施时机

**当前训练完成后**（Epoch 8 预计 PPL < 20）：
1. 先验证 zh 综合体能正常对话（test_api_dialogue.py）
2. 若对话质量达标 → 实施 Sparse Router（Step 3）
3. 若对话质量不达标 → 先排查训练问题，再考虑 Router

**不提前实施的原因**：Router 需要在已能对话的基底上训练，否则无法验证 Router 是否提升协作质量。

---

### 4.0d ★★ **自适应激活设计深化：3 个工程妥协 + 上限更高选项**（2026-08-05 设计讨论）

> §4.0c 已给出基础设计方案（Probe-based Sparse Router）并实现了阶段1-3（commit 3526274/54e95e5）。
> 本节审视实现中的 **3 个工程妥协**，给出上限更高的替代选项，供决策。
> 用户明确要求聚焦"设计自适应激活机制"，本节是设计层面的完整讨论（不实现）。

#### 妥协 1：稀疏粒度是 batch 级并集（当前实现）vs per-sample（上限更高）

**当前实现**（[ensemble.py:1016-1019](file:///e:/taiji-neuron/taiji/resonance/ensemble.py#L1016-L1019)）：
```python
selected_mask = hard_mask.sum(dim=0) > 0  # batch 级并集
```
一个 batch 内**任一样本**选中的神经元，全部参与 round 2+。batch=4 时，4 个样本各选 3 个不同神经元，并集可能接近全部 N。**稀疏收益随 batch 增大而递减**。

**上限更高方案：per-sample top-K**
- 每个样本独立选 K 个神经元，round 2+ 用 [B, K] 索引
- 真正的算力节省需在 forward 层用 sparse mask（Switch Transformer capacity factor）
- **复杂度**：side_signals 是 per-pair 投影（[N, B, D]→[N, B, hidden]），per-sample 稀疏需要对每样本 mask 掉非选中神经元的所有 side_channels 计算

**决策依据**：

| N | 并集实际激活（batch=4）| per-sample 激活 | 差距 |
|---|----------------------|-----------------|------|
| 5（当前）| 4-5（节省 0-20%）| 3+shared=4（节省 20%）| 小 |
| 20（未来）| 12-16（节省 20-40%）| 5+shared=6（节省 70%）| 显著 |

**推荐**：当前阶段保持 batch 并集（N=5 差距小，per-sample 复杂度高收益低）；设计上预留 per-sample 接口，N 增大后升级。这是"随规模增长"的正确时点判断，不是永久妥协。

#### 妥协 2：Router 无学习信号约束 vs 对比约束（上限更高）

**当前实现**：Router 只通过 CE loss 的 STE 梯度隐式学习（soft_weights 梯度流回 Router）。
**问题**：没有显式信号告诉 Router "**哪个神经元擅长当前样本**"。Router 可能学到"按响应强度路由"（大神经元主导），而非"按能力路由"（谁擅长当前样本谁上）。

**上限更高方案：C12 对比约束扩展到 Router**
共振分已有 C12 contrastive loss（[ensemble.py:1603+](file:///e:/taiji-neuron/taiji/resonance/ensemble.py#L1603)）：约束共振分与 per-neuron NLL 排序对齐。Router 应复用同一约束：

```python
# ideal: NLL 低的神经元应获高路由权重（谁能更好预测当前样本）
ideal_weights = F.softmax(-nll / 0.5, dim=0)  # [N]
# actual: Router soft_weights
actual_weights = router_soft_weights.mean(dim=0)  # [N]
# KL(actual || ideal) 让 Router 学"能力路由"
router_contrastive_loss = (actual_weights * (actual_weights.clamp(min=1e-8).log()
                            - ideal_weights.clamp(min=1e-8).log())).sum()
```

**为什么这是上限提升的关键**：
- 无约束 Router：路由分只反映"响应强弱"，大神经元天然强响应 → 路由退化回"大神经元主导"
- 有对比约束：Router 被迫学习"谁在**当前样本**上预测最好"，小神经元在它擅长的主题上获得高路由分 → **真正的样本驱动自适应激活**

**推荐**：加入对比约束。改动小（复用现有 contrastive_loss 逻辑），上限提升大。

#### 妥协 3：固定 K vs 熵驱动动态 K（上限更高）

**当前实现**：固定 K=3。
**局限**：简单样本 1-2 个神经元足够，复杂样本需要 4-5 个。固定 K 要么浪费算力（简单样本），要么能力不足（复杂样本）。

**上限更高方案：熵驱动动态 K**
```python
# 路由分分布的熵：高熵（Router 不确定）→ 多选；低熵（明确）→ 少选
entropy = -(soft_weights * torch.log(soft_weights + 1e-8)).sum(-1)  # [B]
K_b = int(torch.clamp(entropy / math.log(N) * N, K_min, K_max).item())
```
- 低熵（Router 99% 确定某神经元）→ K=1-2，省算力
- 高熵（Router 犹豫）→ K=4-5，保证能力
- 上限：**每样本算力分配与任务难度匹配**（类似"简单问题快答，复杂问题慢想"）

**推荐**：预留动态 K 接口（Router.forward 已支持 effective_k 计算），起步固定 K=3 验证，稳定后升级熵驱动。

#### 设计决策汇总

| 决策点 | 当前实现 | 上限更高 | 推荐 | 实施成本 |
|--------|---------|---------|------|---------|
| 稀疏粒度 | batch 级并集 | per-sample top-K | 当前并集，预留接口 | 高（side_channels mask）|
| 学习信号 | 隐式（CE 梯度）| **C12 对比约束** | **加入对比约束** | 低（复用现有逻辑）|
| K 值 | 固定 3 | 熵驱动动态 | 起步固定，预留动态接口 | 中 |
| 路由时机 | round 1 后单次 | 每轮动态 | 单次（round1 响应已充分）| 无需改 |
| 路由输入 | field_vector+conf+score_vec | +任务特征（域）| 当前够用，跨域后加域特征 | 中 |

**结论**：3 个妥协中，**对比约束（妥协2）是当前阶段唯一值得立即实施的**——改动小、上限提升大、且直接服务于"样本驱动自适应激活"的核心目标。per-sample（妥协1）和动态 K（妥协3）留待 N 增大后升级。

#### 实施状态（2026-08-05 用户决策：三个都选上限更高方案，已全部实施）

| 决策点 | 用户决策 | 实施状态 |
|--------|---------|---------|
| 对比约束（妥协2）| **立即实施** | ✅ 已实施（router_contrastive_loss 加入总 loss，权重 0.1）|
| per-sample top-K（妥协1）| **现在升级** | ✅ 已实施（per-sample hard_mask 控制 side_signals + field_state 注入）|
| 熵驱动动态 K（妥协3）| **现在设计并实施** | ✅ 已实施（Phase 2 熵驱动，低熵少选/高熵多选）|

**实现要点**（commit 待填）：
1. **SparseRouter.forward 升级**：动态 K（每样本独立，k_min=1, k_max=N-1）+ per-sample top-K（每样本选 K_b 个，shared_expert 始终激活）+ 返回 `k_per_sample [B]` + `top_k_ids [B][K]`
2. **forward_train 接入**：round 1 后 Router 选 per-sample top-K；round 2+ 用 per-sample mask 控制：
   - side_signals：post 只接收该样本 top-K 的 pre 信号（`pre_vec * pre_mask.unsqueeze(-1)`）
   - field_state：只累加每样本 top-K 神经元的写入（`all_vecs_weighted * mask_t`）
   - 融合：per-sample final_weights（STE）
3. **forward（推理）接入**：同样 per-sample mask 控制 side_signals + field 写入，保证训练-推理一致
4. **对比约束**：`router_contrastive_loss = KL(router_soft_weights.mean(0) || softmax(-nll/0.5))`，与 C12 共享 per_neuron_nll 计算
5. **修复的 bug**：负载均衡 loss 的 P 原本 detach（Router 无梯度），改为可微；round 2 循环误清 Router 缓存，改为不重置

**测试验证**（全部通过）：
- SparseRouter 单元：Phase0 K=N、Phase2 熵驱动 K=[4,3,4,4]（每样本不同）、shared 始终激活、mask 行和=K、final_weights 归一化
- 梯度流：load_balance grad=2.44、contrastive grad=1.96、CE-path grad=21.40（三条路径全部非零）
- 集成：forward_train（use_sparse_router=True）正常，router_contrastive 激活，Router 梯度 54.18
- 向后兼容：use_sparse_router=False 时 Router 不创建，dense 模式完全正常（router_contrastive=0）
- 推理 forward()：use_sparse_router=True 正常，shared_expert 保留

**待训练验证**（阶段4）：训练完成后 `--use_sparse_router --sparse_router_top_k 3 --sparse_router_warmup_steps 2000` 对比稠密 vs 稀疏的 EMERGE、PPL、推理速度。

---

### 4.1 ~~"共振"是推理技巧，从未被训练~~（已过时，S1 修复后全可微）

**历史定性已过时**：S1 修复后 `forward_train` 是全可微多轮共振路径——训练时确实注入 field_state、side_signals、调质、gamma 振荡，所有机制进入梯度流。共振不再是"推理时拼凑"。

**当前状态**：共振已训练，但 n_rounds=2 较少，且训练任务是单步问答，共振的"多轮深化"潜力未充分发挥。真正的提升方向是增加 n_rounds + 引入多步推理训练数据（见 §4.0b 候选 3）。

### 4.2 tokenizer 错配是隐性天花板

256K embedding 配 16K tokenizer，14.6 万 embedding 行是死参数。所有 PPL 数字都被这层"tokenizer 噪声"掩盖，不解决它，后续所有优化都被掩盖。

### 4.3 协作层纯 CE 导致协作不涌现

协作层训练用纯 CE，不约束"协作是否真的比单神经好"。side_channels 学成噪声调制，很多场景协作 PPL ≥ 最强个体。需要 margin ranking + diversity + load balancing 三联 loss。

### 4.4 三阶段割裂导致表示空间无法协同

base → dialogue → cross_spec 三阶段从未联合训练，每阶段冻结前者。表示空间无法协同适配，side_channels 只能在固定表示上做线性调制。

### 4.5 生物学机制是"装饰"而非"骨架"

STDP/调质/Gamma/睡眠/新生全部以 Optional 注入，可独立开关。这意味着它们是"装饰性"的，不是架构的"骨架"。真正的生物学架构应该让这些机制成为不可移除的核心组件。

---

## 五、建议的改进路径（上限优先）

### 阶段 1：修复隐性天花板（S2 + S4）
- 训 256K general tokenizer
- 训练步数提到 12000-16000
- **不解决这两个，后续所有优化都被掩盖**

### 阶段 2：让共振可训练（S1）
- 可微多轮共振（Gumbel-softmax / straight-through）
- 让 forward_train 接入场+侧通道+调质
- **这是把共振从推理技巧变成可学习能力的唯一路径**

### 阶段 3：多任务 loss（S3）
- SFT answer masking
- 协作层 margin ranking + diversity + load balancing
- 跨域对比 loss（hub neuron 设计）

### 阶段 4：推理路径优化（S6 + S11 + S12）
- 域 token 对齐表
- 长上下文（attention sink / 分块共振）
- 多轮对话状态管理

### 阶段 5：生物学机制深化（S9）
- STDP 影响注意力/FFN 权重
- 多频段振荡 + 跨频耦合
- 真正睡眠重放
- 自组织新生

---

## 六、方向 B 备案：小神经元 + 强协作架构（2026-08-04 设计）

### 6.1 当前方向优先级与备案触发条件

**当前方向（优先）**：5 神经元阵容 + EOS + 短答案 + 数据扩充，继续优化。

**关键认知（2026-08-04 二次修正，代码公式实测）**：协作层参数随神经元数 + 规格升级双维度增长，**且规格升级存在"unified 放大效应"**：

**精确公式**（从代码验证）：
- side_channels：每条 `nn.Linear(pre.field_dim, post.hidden_size, bias=False)` = pre.field_dim × post.hidden_size，**pairwise = O(N²)**
- CrossSpecProjector（T6 升级为 2 层 MLP）：
  - 正向（神经元 fd → unified U）：`linear1(in→U) + linear2(U→U)` = **U × (fd + U)**
  - 反向（U → 神经元 fd）：**fd × (U + fd)**
- **U = max(所有神经元 field_dim)** ← 这是关键放大机制

**unified 放大效应（用户指出的核心机制）**：加入更大规格神经元会把 U 提升到新规格的 field_dim，**导致所有已有神经元的投影层 out_dim 变大 → 整个协作层参数被放大**。这不是新增一个投影层的线性增长，而是全局跳跃。

**历史验证（参考之前增加中等规格 standard 的经验）**：

| 阵容 | U | side_channels | 投影层 | 协作层总 | 增量 |
|------|-----|--------------|--------|---------|------|
| 4C（历史）| 2048 | 12.6M | 67.1M | **79.7M** | — |
| 4C+1S（当前）| 3072 | 25.2M | 142.6M | **167.8M** | **+88.1M (+111%)** |
| +1 EXPERT | 4096 | 48.2M | 269.5M | **317.7M** | **+149.9M (+89%)** |
| +2 EXPERT | 4096 | 79.7M | 336.6M | **416.3M** | +98.6M (+31%) |

**结论（完全验证用户判断）**：
1. **增加更大规格神经元 → 协作层扩展规模大幅跃升**：历史增加 standard 使协作层翻倍（+111%），增加 EXPERT 再 +89%
2. **unified 放大是主力**：+1 EXPERT 的 +149.9M 增量中，投影层放大贡献 126.9M（因 U 3072→4096），side_channels 新增只贡献 23M
3. **U 提升是单次性跳跃**：+2 EXPERT 增量降到 +31%（U 已到 4096 不再变，只剩 pairwise side_channel 增长）——**规格升级的放大效应是一次性的，重复同规格收益递减**
4. **最优扩展策略**：规格阶梯升级（引入新规格触发 U 跳跃）+ 数量扩展（同规格 O(N²)）双管齐下
5. 注意：上一版修正的 130M/105M/190M 数字有误（U 值用错 + 公式错误），以上表为准

**方向 B 备案触发条件**（任一满足）：
1. 当前架构在 EOS+短答案重训后，API 质量仍不达标且 PPL 已收敛
2. 增加到 20+ 神经元后协作层仍未承载主要能力（EMERGE < 30%）
3. 单神经元能力过强导致协作被边缘化（移除协作后 ensemble PPL 下降 < 10%）
4. **新增**：规格升级到 EXPERT 后协作:主体比未提升（说明规格升级无法替代 N 增长）

---

### 6.2 方向 B 神经元规格设计

| 参数 | 当前（中等神经元）| 方向 B（小神经元）|
|------|----------------|----------------|
| hidden_size | 512 / 768 | **128-256** |
| 层数 | 6-12 | **2-4** |
| 单神经元参数 | 51-134M | **1-5M** |
| 神经元数量 | 5 | **20-50** |
| 神经元总参数 | 338M | 50-200M |
| **协作层参数** | 130M | **500M-2B** |
| 协作:神经元 比 | 0.4:1 | **10:1 以上** |

### 6.3 训练流程五阶段

#### 阶段 0：规格设计
- 20-30 个神经元，每个 2-3M 参数，hidden=256，层数=3
- 协作层目标参数量 360M+（含 side_channels + cross_spec + 场演化层 + 协作注意力）

#### 阶段 1：数据分工策略（推荐 C）
- **策略 C：随机数据子集 + 训练自动分化**
- 每个神经元随机看 1/N 数据，训练过程中自然分化
- 不预定义主题/能力边界（避免人为偏见），类似人脑经验驱动分化
- 已有 [CoactivationTracker](file:///e:/taiji-neuron/taiji/resonance/tribal.py) 基础设施追踪分化模式

#### 阶段 2：单神经元预训练（弱能力初始化）
- 每个神经元独立训练（只看自己的 1/N 数据子集）
- **关键：PPL 故意停在 80-120**（不完全收敛，保留学习空间）
- 不要训练到 PPL < 50，否则单神经元能力过强，协作又成附属
- 这是"被迫协作"的前提——单神经元无法独立生成有效回答

#### 阶段 3：协作层训练（核心阶段）
- 冻结神经元主体，训练协作层（从头初始化）
- 协作层组件设计：
  - side_channels（全连接，每对神经元双向 excite/inhibit）
  - cross_spec 投影层（统一到 512 维场空间）
  - **场状态演化层**（新增，多轮可微场状态更新，~200M 参数）
  - **协作注意力**（新增，神经元间注意力机制，~100M 参数）
- Loss 设计：
  ```
  Loss = CE_loss(ensemble_output, target)        # 协作输出逼近目标
       + λ × diversity_loss(neuron_outputs)       # 防止神经元输出同质化
       + μ × cooperation_pressure(neuron_outputs) # 单神经元 PPL < 50 时加惩罚
  ```
- 目标：ensemble PPL < 30（协作显著优于单神经元）

#### 阶段 4：端到端微调
- 解冻 body 最后 1-2 层
- 联合微调（body lr << 协作层 lr，保护神经元专业化）
- 目标：ensemble PPL < 20

#### 阶段 5：评估
- EMERGE > 50%（协作远超最强个体）
- API 对话质量达标

### 6.4 关键设计决策（待研究，非立即执行）

| 决策点 | 选项 | 倾向 |
|-------|------|------|
| 协作层架构 | A. 复用 side_channels + cross_spec / B. 新增场演化层 + 协作注意力 | **B**（协作层需足够容量）|
| 单神经元预训练强度 | A. 不预训练 / B. PPL 80-120 / C. PPL 30-50 | **B**（弱能力但保留基础）|
| 神经元数量 | A. 10 / B. 20-30 / C. 50-100 | **B**（已有 tribal 拓扑支持）|

### 6.5 与当前流程的本质差异

| 维度 | 当前流程 | 方向 B 流程 |
|------|---------|-----------|
| 单神经元预训练 | PPL 30-50（强能力）| 充分收敛（规模 1-5M 天然弱能力，无需欠训练）|
| 协作层角色 | 协作是锦上添花 | **协作是能力载体** |
| 协作层参数占比 | 24% | **>70%** |
| 单神经元能否独立 | 能 | **不能** |
| 协作涌现强度 | 弱 | 强（被迫协作）|

### 6.6 当前状态

**状态**：备案，不立即执行。
**优先路径**：当前 5 神经元阵容 + EOS + 短答案重训 → 评估 → 若不达标再考虑增加神经元数 → 若仍不达标才触发方向 B。
