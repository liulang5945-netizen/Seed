# 共同内容核心：重构-2 候选与三类基线比较合同

> 状态阅读说明：本件对应统一步骤02，已由用户批准（依据[R4页首批准记录](SPEC-NATIVE-CONTENT-CORE_R4_PRICING_AND_GATES_20261030.md)）；下文“草案/待评审”为起草时快照，不再表示当前审批状态。进度与动作统一见[总计划](../MAINLINE.md)，本注不改比较合同。

日期：2026-10-30，Asia/Shanghai。状态：**设计合同草案，待用户评审；R3 隔离实现已在其下推进（见[R3 记录](SPEC-NATIVE-CONTENT-CORE_R3_ISOLATED_IMPLEMENTATION_20261030.md)），但不授权训练、评价或读取新 sealed 面。** 上级方向见[共同内容核心重构方案](PLAN-NATIVE-CONTENT-CORE_REDESIGN_20261030.md)、[R1 统一语义合同](SPEC-NATIVE-CONTENT-CORE_R1_SEMANTIC_CONTRACT_20261030.md)；当前队首与进度只看[03](../active/roadmap/03_CURRENT_EXECUTION.md)。

## 1. 目的与边界

R1 已把任务、时间、byte 目标、固定分母、cut/rescue 正确性、来源/版本和公开 fixture 统一；本件只解决 R2 的问题：**定义一个完整候选和两组公平基线，明确比较问题、接口责任、数据/权限/资源口径和退出**。它不定义参数规模、训练预算、数值门或具体编码器算法；这些在 R3/R4 另行冻结。

本件不把旧 F/N178 的结果洗成新证据，不重读旧 sealed 面，不把 parser 覆盖或局部 loss 改善当作候选优势。

## 2. 三类比较对象

| 对象 | 角色 | 必须保留 | 不能得出的结论 |
|---|---|---|---|
| `corrected_old_v1` | 修正语义的旧图基线 | 硬 span 解析、离散 store、current/previous 时间选择、固定 value bias/冻结语言头；但 byte 对齐、固定分母、时间查询按 R1 修正 | 不能因旧图被修正就称新核心无优势；也不能让旧图继续背负已知错目标 |
| `ordinary_v1` | 普通端到端学习基线 | 同输入/目标/数据/权限/资源的成熟序列学习器；无硬解析门；同评分协议 | 不借外部模型、额外预训练、不同语料或更大参数偷换研究对象；它是对照，不是采用对象 |
| `new_core_v1` | 主候选 | 可学习事件/内容编码与写入、版本化 store、时间条件读取、逐步内容条件生成；任务信用可作用于读取与表达 | 多部分一起变化时只称架构包收益；未过 R3/R4 不得称已学会或已采用 |

### 2.1 完整候选的最低要求

`new_core_v1` 必须是一个**单一可恢复原生入口**，不是若干独立局部头的集合：

- 同一 bundle 包含内容编码/写入、版本化 store、时间条件 reader、逐步 decoder、来源/权限身份和完整 checkpoint；
- query-only 回答不得从事实原始字节、hidden 旁路、source cache 或宿主模板直接取得答案；
- 内容身份同时有 latent、exact_ref、source、valid time、revision、encoder revision；
- 任务 loss 至少作用于声明的 reader 与 decoder 两组；辅助解析 CE 单列；
- 旧图/普通基线不得被削弱或给予更少数据/权限/预算；三组共用同一评分器和固定分母。

## 3. 比较问题

### Q1 架构包比较

在独立未见历史上比较 `new_core_v1`、`corrected_old_v1`、`ordinary_v1`：

- 同一数据分区、父能力保持集、输入/状态权限、评分器、参数/计算总预算；
- 主指标是全轮 `answer_exact`、`state_exact`、current/previous/unknown 分类，以及旧保持、恢复、真实成本；
- 允许 `new_core_v1` 使用 R1 §2.2 的接口，但不得使用 gold 候选、旧 sealed 或宿主模板；
- 三组失败模式分别报告，不能只报总分。

### Q2 信用单因素

在 `new_core_v1` 内部只切指定信用：

- `credit_on` 与 `credit_stop` 同初态前向、同结构、同数据/曝光/更新，只切 reader/decoder 的任务信用；
- 比较任务净收益、梯度实际接收者、禁写组和旧保持；
- 不把 Q2 的差异并入 Q1 的架构包收益；Q1 先选架构包，Q2 再解释机制。

## 4. 接口责任

R2 必须把 R1 §2.2 的接口映射到三类对象：

| 接口 | `corrected_old_v1` | `ordinary_v1` | `new_core_v1` |
|---|---|---|---|
| ContentState | 离散 store + 硬解析；保留来源/版本 | 无显式 store；历史在序列输入中 | latent+exact_ref+来源/版本/encoder revision |
| TimeIntent | selector 接收 current/previous | 序列输入直接看到时间条件 | reader 接收 current/previous/unknown |
| ContentReader | 硬解析+时间 selector；固定 bias | 序列模型内部读取 | 每步读 query/time+选中内容+前缀 |
| DynamicDecoder | 冻结语言头+固定 bias | 同预算序列解码器 | 每步内容条件 logits |
| CreditMatrix | 回答信用只到下游；解析辅助 CE | 端到端任务 loss | 任务信用到 reader+decoder；辅助 CE 分账 |
| Checkpoint | 旧图+store | 序列模型完整状态 | 同一 bundle 完整状态 |

表只定义责任，不规定实现算法；`corrected_old_v1` 的硬解析/固定 bias 是其基线特征，不是新核心的实现。

## 5. 数据、分区与公平性

- 使用 R1 世界审计的 360-history 样本计划（216 train / 72 query / 72 holdout，4 episodes/history）；精确覆盖和 split 在 R2 用公开 fixture/覆盖求解器冻结。
- family/反事实按完整历史整体划分；同一历史改写不得跨 train/确认面；旧已消费 sealed 面只作开发诊断，不作最终门。
- 三组共用同一 train/query/holdout、同一输入可见性、同一目标 byte、同一固定分母；训练进程不得读 query/holdout 正文或标签。
- 普通基线可以使用成熟序列模型，但必须同参数/计算总预算、同数据/同目标/同评分；不得给新核心额外预训练或削弱基线。
- 父能力保持集使用已冻结的旧探针；三组都测，不能只测新核心。
- 资源以 R4 冻结的训练/前处理/恢复/评价总成本边界和参数上限口径比较；R2 不写具体数值。

## 6. 评分与判据

- 沿用 R1 §5–§6：固定分母、首字节/第 t 步目标、current/previous/unknown、完整回答与 state 分账、相关/无关交换、cut/rescue 正确率、旧保持、恢复和真实成本。
- Q1 主判定是独立未见历史上的合取：行为净收益、机制/状态正确性、旧保持、恢复、成本；一个 pass 字段不能覆盖保持或工程失败。
- Q2 判定只看同初态/同结构的信用差异；梯度非零、loss 下降、参数变化均不替代任务净收益。
- 解析/状态 oracle 只作隔离工程上界；不能进入生产路径或能力主张。
- 三组均报告失败模式：解析失败、时间错选、旧值回退 current、hidden 旁路、固定 bias 局限、普通基线容量/数据不足。

## 7. 退出与停止

- R3 隔离实现与零步恢复已在本草案下推进并完成结构门（见[R3 记录](SPEC-NATIVE-CONTENT-CORE_R3_ISOLATED_IMPLEMENTATION_20261030.md)）；R2 用户评审未完成前不进入 R4 定价/训练与 R5 评价。
- 若无法定义完整候选、公平基线或可判失败的门，先结束 R2 设计并修订；不得用“先跑一版”绕过。
- 工程失败/预算停止不是科学反证；单独留证，不自动重跑或提额。
- 旧 F/N178 跨包重审计数和费用不清零；新包在 R4 前明确本次范围的累计停止线/复审决定。

## 8. 明确不做

- 不在 R2 实现代码、加载模型、训练、评价或读新 sealed。
- 不把 R2 合同当作能力通过、预算承诺或采用批准。
- 不一次推倒 TSK、语言、记忆、世界和全部训练机制；按 Q1 架构包比较选择。
- 不把 `corrected_old_v1` 的硬解析/固定 bias 局限称为新核心优势；也不把 `ordinary_v1` 的通用能力当作原生认知。

## 9. R2 出口清单

- [x] 三类对象角色、完整候选最低要求和 Q1/Q2 分离已写。
- [x] R1 §2.2 接口已映射到三类对象；无 gold/sealed/模板旁路。
- [x] 数据/分区/公平性/参数预算口径已定；三组同数据/权限/资源。
- [x] 固定分母、current/previous/unknown、cut/rescue 正确性、旧保持、恢复、成本门已写。
- [x] 失败模式与停止线已写；工程/预算失败单独。
- [ ] 用户已评审 R2 合同；未确认前不进入 R4 训练/定价与 R5 评价（R3 隔离实现仅在评审并行推进）。
