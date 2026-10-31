# v1.1 先验工作核查与新颖性定位（prior-art-2，2026-10-31）

> 历史入库标记：评估对象为64fdf8576的v1.1/E1；不是对当前v1.2的重新裁决。原字节及入库变换见plans/manifests/taiji_core_v11_e1_audit_admission_20261031.json。

**结论先行**
1. §3 十条来源全部真实存在，未发现伪造 arXiv 号；2 条标注需修正，2 条原文受登录墙未读（仅核实 DOI/题名）。
2. v1.1 的"近端擦除 + 独立写入"**不是新算子**：Gated DeltaNet-2 已解耦通道级擦除/写入门；Delta 规则可读作在线梯度/最小二乘一步；TTT 把快状态更新当作自监督步；fast-weight programmer 早已把快权重视为梯度步。v1.1 只把擦除近端非扩张化并加有界状态地板，属已有算子重组。
3. 可能新意仅在**耦合**：快记忆 × 函数中性出生 × 可塑性监测 × 内源活动/世界模型进入同一主体。"未见同构工作"≠新颖，draft §3 亦自认未查重。
4. 不可主张：新快权重算子；首次解耦擦除/写入；首个可增长/弹性网络保可塑性；首个多时间尺度记忆架构；"自进化/生命"。
   可主张：近端非扩张擦除 + 独立写入的稳定性条件与有限算例界（200 步衰减≈4.80e-8）；三者同主体的可执行耦合与单写者语义；受限写入反例负结果（121 次裁剪/200 步→修订）。

## A. §3 引用核查

| 来源（draft 标注） | 核实 | 问题/建议 |
|---|---|---|
| Nested Learning/HOPE，NeurIPS 2025（[2512.24695](https://arxiv.org/abs/2512.24695)） | 真实；题名 *Nested Learning: The Illusion of Deep Learning Architectures*；NeurIPS 2025 [海报页](https://neurips.cc/virtual/2025/poster/116123) | HOPE 是该文实例非并列论文；arXiv v1（2025-12-31）晚于会期，建议标会议版 |
| Titans，2025（[2501.00663](https://arxiv.org/abs/2501.00663)） | 真实；*Titans: Learning to Memorize at Test Time*，NeurIPS 2025 | 准确；建议补会议名 |
| ATLAS，2025（[2505.23735](https://arxiv.org/abs/2505.23735)） | 真实；*ATLAS: Learning to Optimally Memorize the Context at Test Time* | "重放当前与过去"未逐句核实，宜收窄 |
| Gated DeltaNet-2，2026-05（[2605.22791](https://arxiv.org/abs/2605.22791)） | 真实；2026-05-21 v1，NVlabs 技术报告；摘要确证通道级擦除门 b_t+写入门 w_t，GDN/KDA 推广；1.3B/100B-token 见摘要 | 准确；勿引其吞吐作本模型吞吐 |
| Differentiable Plasticity，2018（[1804.02464](https://arxiv.org/abs/1804.02464)） | 真实；ICML 2018 oral | 准确 |
| e-prop，Nat Commun 2020（[s41467-020-17236-y](https://europepmc.org/article/med/32681001)） | DOI/题名核实：*A solution to the learning dilemma for recurrent networks of spiking neurons* | 原文登录墙未读；描述与题名一致 |
| DreamerV3，Nature 2025（[s41586-025-08744-2](https://www.nature.com/articles/s41586-025-08744-2)） | 真实；*Mastering diverse control tasks through world models* | 准确 |
| GradMax，ICLR 2022（[2201.05125](https://arxiv.org/abs/2201.05125)） | 真实；摘要确证"加神经元不影响已学"、SVD 最优初始化 | 准确 |
| Loss of Plasticity，Nature 2024（[s41586-024-07711-7](https://www.nature.com/articles/s41586-024-07711-7)） | **部分核实**：题名命中，URL 未打开（登录墙） | 建议引 DOI 复核卷期 |
| Growing and Elastic Networks，2026-08（[2608.01475](https://arxiv.org/abs/2608.01475)） | 真实；*Plasticity of Growing and Elastic Neural Networks in Online Continual Learning*，Kong & Sutton，2026-08-02 | 准确（增长+剪死单元、近常数尺寸） |

## B. 最接近的 5 个工作及差异

| 工作 | 与 v1.1 关系 | 差异 |
|---|---|---|
| [Gated DeltaNet-2（2605.22791）](https://arxiv.org/abs/2605.22791) | 擦除/写入解耦同源先例 | 通道级门控 vs 近端非扩张擦除+有界状态；1.3B 结论不可外推 |
| [Gated DeltaNet（2412.06464）](https://arxiv.org/abs/2412.06464) | "门控快速擦除+delta 定向更新"正是 v1.1 结构 | v1.1 加非扩张约束与状态地板；域不同 |
| [DeltaNet 并行化（2406.06484）](https://arxiv.org/abs/2406.06484) | Delta 规则即带 L2 的在线梯度/RLS 一步 | 近端形式是其变体，非新规则 |
| [TTT（2407.04620）](https://arxiv.org/abs/2407.04620) | 快状态=模型、更新=自监督学习步 | v1.1 用监督/近端更新并绑结构出生 |
| [Nested Learning（2512.24695）](https://arxiv.org/abs/2512.24695) | 多频更新、上下文流、优化器-记忆联合设计 | v1.1 规模极小、无 LM 成绩，仅耦合层面对话 |

## C. draft 未引但必须补的先验（10 条）

| 先验 | 差异 |
|---|---|
| [Fast Weight Programmers（2102.11174）](https://arxiv.org/abs/2102.11174) | 快权重=可微编程/梯度步；v1.1 是其小规模实现 |
| [Fast Weights（1610.06258）](https://arxiv.org/abs/1610.06258) | 快权重当注意力记忆，无门控/近端约束 |
| [semi-MDP（Sutton et al. 1999）](https://dblp.org/rec/journals/ai/SuttonPS99.html) | 时间抽象理论；v1.1 群/子目标分工缺形式化 |
| [Option-Critic（1609.05140）](https://arxiv.org/abs/1609.05140) | option 端到端学习；v1.1 无终止/内部策略规则 |
| [Progressive Nets（1606.04671）](https://arxiv.org/abs/1606.04671) | 固定列出生、不回收；v1.1 主张回收 |
| [Switch Transformers（2101.03961）](https://arxiv.org/abs/2101.03961) | 条件化专家路由；v1.1 出生是结构增长非路由 |
| [Backpropamine（2002.10585）](https://arxiv.org/abs/2002.10585) | 可微神经调制可塑性；v1.1 局部规则多为手调 |
| [World Models（1803.10122）](https://arxiv.org/abs/1803.10122) | 内源想象先例；v1.1 不把模拟当经历 |
| [Lyle et al. 2023 可塑性诊断](https://ui.adsabs.harvard.edu/abs/2023arXiv230301486L/abstract) | 死单元/有效秩诊断；v1.1 指标应对齐 |
| [EWC（1612.00796）](https://arxiv.org/abs/1612.00796) | 防遗忘正则基线；v1.1 关注可塑性而非仅遗忘 |

## D. 必须补的对比基线（5 条）

1. Gated DeltaNet-2（解耦擦除/写入）在相同受限门序列上的裁剪次数与稳定性；
2. Gated DeltaNet/DeltaNet（含 L2 隐式形式）——检验近端擦除是否只是等价改写；
3. TTT-Linear——检验"快记忆=学习步"是否已覆盖 v1.1 收益；
4. 可塑性指标：Dohare et al. 2024 与 Lyle et al. 2023；
5. 结构出生：GradMax、Progressive Nets、Kong & Sutton 2026。

## E. 未能核实与行号异常

- **行号异常**：任务称 646–665 行为"§22.1 归类"，实际读到 §20 末成本注记与 §21 表格前两行（快记忆擦除、近零状态/查询归一化），**未含 §22.1**；建议 lead 核对行号。
- 未能核实：ATLAS 的具体重放论断；Loss of Plasticity 与 e-prop 原文（登录墙）；"近端非扩张擦除"用于快权重的直接同构先例未检索到——"未发现"不等于新颖。
