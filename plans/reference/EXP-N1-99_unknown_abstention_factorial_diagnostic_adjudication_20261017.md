# EXP-N1-99 · 2026-10-17 · 已知 v6 unknown 题面的观测数 × 名称距离探针

按已批准的 [PLAN-N1-99](PLAN-N1-99_unknown_abstention_factorial_diagnostic_20261017.md) 完成一次性已知题面机制探针。原 v6 4-observation／d=1 行复用既有报告；B/C/D 发射135条新诊断查询。主读数见 [机器报告](../../reports/taiji_n1_unknown_abstention_factorial_diagnostic_20261017.json)，SHA-256 `ff9c6aa78e543bf61dafe6de7a04152dc29ea416ab389d4da3e9be8b3256a6e8`。

## 1. 执行完整性

- readiness 通过：v6 manifest SHA-256 `7dddf00f3f2e24381728855b218e67bc8160fd6badeb1a7a933937d999766d49`、既有确认报告 SHA-256 `3d924e19d97b59ea12505111603b7d562ada8e8d9bdac71351a51c24c5918849` 均与批准计划绑定；9枚端点齐备，A格45行来自已出版v6报告。
- B/C/D共135条新查询，A/B/C/D各45行，总CPU 51.796875秒，低于300秒上限。9枚checkpoint全bundle前后中性，160项来源守卫未变，`parameter_updates=optimizer_steps=training_runs=0`，`can_promote=false`、`unseen_claim_allowed=false`。
- one-shot attempt SHA-256 `9b82e234e31d95c5ef16f6579e35809a8774ffe64a65efa8bc9a02ec99bfa17e`，执行期间源码未变。v6确认面未重查，没有新dev或holdout查询。
- **输入摘要补充**：主执行报告没有按 PLAN-N1-99 §3 序列化逐行输入 SHA。没有重跑模型；独立只读核验器从manifest、冻结源码及 `name_rng_seed=9902` 重建20个条件输入，并逐项连接180行、9枚端点与原v6基线。补充件 [逐行输入绑定核验](../../reports/taiji_n1_unknown_abstention_factorial_binding_verification_20261017.json) SHA-256 `efcc48b7c8b1a7d9d646f149e72687ee11f6ea6e2e50dd049565d69395c9fada`，`status=verified`、180行绑定、20项输入绑定、模型加载/checkpoint加载/新查询均为0。输入 SHA 是事后重建，因此记录为执行器报告字段缺口，不伪称由主报告直接记录。

## 2. 逐格结果

每个条件共15行／臂，分母为5个问题 × 3 seeds；A格取自v6已出版报告。

| 条件 | 观测数 | Hamming距离 | position | terminal | cut | 三臂合计 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| A | 4 | d=1 | 0/15 | 4/15 | 2/15 | 6/45 |
| B | 2 | d=1 | 0/15 | 3/15 | 0/15 | 3/45 |
| C | 4 | d=2 | 1/15 | 4/15 | 2/15 | 7/45 |
| D | 2 | d=2 | 1/15 | 3/15 | 1/15 | 5/45 |

position的两次正确弃权都落在 `v6-confirm-015`：C/d=2/4-observation是seed11，D/d=2/2-observation是seed13；其余13行仍未逐字输出 `unknown`。在这5个题面上，把观测数由2改4没有改变position的命中数；d=2相对d=1各多1/15。这个读数只说明一个窄小的已知面效应，不能支持一般化或因果强度主张。

## 3. 实际 train 来源中的分布缺口

run88的训练attempt锁定了v3 native-evidence curriculum（attempt SHA `6a3124a1293c33dcbdc45bc55fb34971200236cb4673e52c9c981335e0bc6234`；manifest SHA `ac56fb0aba2ef602779f58749e6f8c35324c40c433fd16ce55b89689235b0604`）。只筛train split的摘要见 [train exposure audit](../../reports/taiji_n1_hard_neighbor_train_exposure_audit_20261017.json)，SHA-256 `a40d2866e31a844a61619a44084d05570682e645501bb54a34875a0c4a748dff`：192项、四类各48；unknown 48项全是2条观测／3个实体，目标实体均缺席，最近干扰名Hamming距离为8/9/10，d≤2训练实体对为0。该事实只限定这个被run88守卫的训练组件，不外推到运行器连接的其它目标。

这支持下一步把近邻 hard view 加入原训练课程，同时给lookup/revision正例也提供近邻干扰，避免模型学成“名字相似就拒答”。具体执行冻结在 [PLAN-N1-100](PLAN-N1-100_balanced_hard_neighbor_training_and_v7_20261017.md)。

## 4. 流程偏差与判定边界

Plan99 明确限定不访问dev/holdout。一次执行后的探索性脚本错误地汇总了v3 manifest的dev/holdout结构元数据（类别、观测数、实体数、目标实体出现情况和名称距离）；没有模型加载、推理、训练或查询，这些汇总没有用于Plan99门值或判读。该越界已登记到08台账；后续训练数据设计只使用train split，dev/holdout仍不在读取或评分范围内。

N1 C3仍未达成；v5/v6均已消费且不可重测；本件不是晋级证据，保持集、C4净增益与三器官出口仍各自未完成。
