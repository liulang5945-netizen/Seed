# PLAN-N1-160A 执行结案

**日期：2026-10-20**
**状态：完成；seed130 face 的六个 endpoint 已全部评分**
**执行补遗：** [PLAN-N1-160A](PLAN-N1-160A_remaining_endpoint_continuation_20261020.md)
**上位方案：** [PLAN-N1-160](PLAN-N1-160_seed130_candidate_parent_fresh_replication_20261020.md)
**seed130题面：** [face manifest](../manifests/n1_plan160_v32_candidate_parent_seed130_20261020.json)

## 完整性与资源

N1-160首轮在seed11 candidate endpoint留下的480条有效记录已逐项核对并原样并入；N1-160A只对其余五个endpoint新增2,400条查询，合并后覆盖全face的2,880条及72个endpoint×类别×长度分层。续接progress有2,400行且与合并结果的后2,400行逐行相同；前480行与N1-160停止报告完全一致。seed130题面没有重建，没有重复查询。

来源与checkpoint哈希保持不变；六个bundle独立恢复6/6，续测加载剩余五个bundle。N1-160A readiness、恢复预检、续测分别耗用18.953、99.141、160.859 CPU秒，本阶段累计278.953/3,600秒；连同先前已消费的10,421.219秒，关联累计10,700.172/18,000秒。0训练、0参数更新、0 dev/holdout/confirmation查询，执行无错误，`can_promote=false`。

逐题结果SHA-256为`e499fb690e11334a7e798629ea06f8275febb21b67bb2cc1bccb4543add579dc`；新增progress SHA-256为`bc928dc7cdda33a90e146d722f7c888b9e65bbdc9310ce35f1cbdfd8e7d0d1dc`。结果报告包含合并后的2,880条逐题行、72个分层、来源映射及停止边界；progress按仓库规则留在本机忽略文件中，哈希已写入结果。

## 三张fresh face的复制读数

下表均为同一face内的candidate/parent配对读数。每张面总计1,440条/分支；已知类为1,080条；unknown的每个长度为120条。

| Face seed | 总命中 candidate / parent | 已知类 candidate / parent | Unknown 8行 candidate / parent | 16行 candidate / parent | 32行 candidate / parent |
|---|---:|---:|---:|---:|---:|
| 128 | 1,401 / 1,341（+60） | 1,076 / 1,078（−2） | 96 / 65（+31） | 110 / 93（+17） | 119 / 105（+14） |
| 129 | 1,384 / 1,324（+60） | 1,070 / 1,069（+1） | 93 / 66（+27） | 103 / 88（+15） | 118 / 101（+17） |
| 130 | 1,387 / 1,328（+59） | 1,070 / 1,077（−7） | 98 / 67（+31） | 103 / 85（+18） | 116 / 99（+17） |

逐题配对中，candidate相对parent修正/新增的unknown正确弃答分别为seed128 `69/7`、seed129 `62/3`、seed130 `72/6`；unknown净增分别为`+62`、`+59`、`+66`。三张面都复现了unknown正确弃答的合计提升。提升并非每个seed×上下文长度单元都出现：seed129的seed12/16行略降，seed130的seed12/32行持平。已知类总分在前两张面近乎持平，第三张面少7分，差异集中于lookup，故不能概括成所有已知类都无损。

## 判读与边界

这组结果支持一个有限结论：在三张互斥的程序生成synthetic face上，N1-155A `unknown8_rebalanced` candidate相对各自N1-150 `length_mixed` parent的unknown精确弃答提升重复出现，已知类总命中保持在接近水平。数据仍来自同一类题面生成机制，seed12的部分格子没有提升；它不是自然语言泛化、真实用户能力或模型晋级证据。N1-160的原训练与promotion边界不变，`can_promote=false`。

N1-160首轮失败回执仍保留在[attempt](../../reports/taiji_n1_plan160_candidate_parent_attempt_20261020.json)与[partial result](../../reports/taiji_n1_plan160_candidate_parent_result_20261020.json)；N1-160A续接明细见[完整结果](../../reports/taiji_n1_plan160a_candidate_parent_result_20261020.json)。

## 后续唯一队首

先收束并修订N1证据路线：把三面synthetic复制明确登记为“可复现的合成诊断结果”，与自然语言泛化、能力晋级和N系列出口分开；下一份测试计划必须指出它要跨越的证据级别及独立面边界。此结案不授权把现有结果晋升为能力结论。
