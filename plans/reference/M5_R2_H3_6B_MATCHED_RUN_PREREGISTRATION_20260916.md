# M5 R2-H3.6-B matched dev 训练预注册

> 状态：预注册已冻结；三 seed control/treatment dev 与三组只读 bridge ablation 已完成，因 dev 停止门失败在 final 前结项。
>
> 可机读合同：`plans/reference/contracts/r2_h3_6b_matched_run_v1.json`
>
> 数据：`tests/fixtures/r2_h3_5a_response_plan_v3.jsonl`，digest `0bc5b5540d4b622f907942f42d7b809e825932e50c6e6c505a3cd091edabd691`

## 1. 目的与唯一因果问题

H3.5-A 已证明 signed-hash span 目标不能稳定迁移，并且 plan bridge 会把随机目标的误差带入 renderer。H3.6-B 只回答一个问题：在完全相同的 response-plan candidate、数据、预算、训练顺序和评价链上，把 signed-hash span 替换为 H3.6-A 的冻结目标，是否改善 dev 上未见 response 的条件化表示与序列输出。

它不回答开放域语言能力、Mini/L2/L3、运行期持续适应、VISION 架构采用或默认入口替换问题。任何这些主张都必须另有合同和独立证据。

## 2. 冻结范围

- 数据固定为12 train、8 dev、4 final；family 与完整 response 跨 split 隔离；四类 `unknown_policy`、history、context permutation 和共享起点不同内容保持不变。
- seed 固定为`20260916`、`20260917`、`20260918`；CPU；每臂10 epoch；每个epoch最多12个train episode；总参数上限300,000；训练顺序按fixture顺序，不打乱为隐藏变量。
- 两臂都使用同一个`response_plan_readout` candidate、32维plan、同一protected predictive renderer和同一plan bridge。control只使用legacy signed-hash span；treatment使用H3.6-A target encoder。
- H3.6 target encoder只在训练开始前从当前candidate parent checkpoint读取train response，拟合参数只来自train；拟合后的encoder payload、parent checkpoint digest、corpus digest和train target map digest必须写入trainer checkpoint。训练开始后不得再用当前child模型重算target。
- dev/final只做冻结child的read-only评分和输出记录，不向trainer注入target，不读取参考答案来改变生成路径，不使用`task_family`、`split`、评分字段或人工标签作为runtime输入。
- 开发运行必须带`--defer-final`；所有seed/arm的dev与paired报告冻结并检查后，final至多统一读取一次。final不用于调参、改target、改epoch或决定是否补跑某个seed。

## 3. 对照与消融

| 单元 | 结构 | 目标 | 作用 |
|---|---|---|---|
| control | response-plan readout，32维 | legacy signed-hash span | 隔离target geometry变化 |
| treatment | 同一response-plan readout，32维 | H3.6-A train-whitened native response state + compositional char n-gram | 检验新的内容计划目标 |
| treatment ablation | treatment child checkpoint | `plan_bridge`置零，只读、不学习 | 检验收益是否来自plan对renderer的因果作用，而非额外参数或碰撞变化 |

control与treatment必须使用相同model seed、candidate初始化、训练预算和episode顺序；target encoder不增加Taiji参数。bridge ablation不重新训练、不读取新标签、不改变checkpoint中的其他参数。

## 4. 必须记录的血缘与机器门

每个run的报告必须至少包含：

1. dataset manifest/digest、代码revision、完整config、model parent digest、trainer zero-step digest；
2. target geometry名称、target encoder payload digest、encoder parent digest、fit split、fit episode ids、effective rank、native context dim、target width和train target map digest；control也必须显式记录`target_geometry=legacy_signed_hash_span`；
3. 参数计划数、candidate参数数、effective active parameters、预算上限、episode/epoch/update计数；
4. zero-step磁盘保存→新对象恢复→同输入评分、一次child response更新→child磁盘恢复、parent保护、RNG/pending/lineage摘要；
5. train/dev raw bytes、UTF-8/无替换/end-marker/sequence、exact response、required-term coverage、surprise、collision、context/probability route与paired改写敏感性；不得只写聚合布尔值。

正式训练前逐run必须通过：

- 目标encoder payload schema、corpus/parent digest和geometry重建一致；
- trainer zero-step checkpoint保存恢复一致；
- 一次最小response update能改变child而不改变parent；
- 参数不超过300,000、磁盘空间与checkpoint尺寸检查通过；
- runtime oracle、final读取和protected artifact写入检查通过。

任一run前置失败，停止全部后续seed，不读取final。

## 5. dev 停止规则与判读

先冻结六个dev child和三组bridge ablation，再按预注册指标判断。若任一项出现以下情况，立即停止，不用追加epoch修复：

- checkpoint/恢复/parent保护/预算/target血缘失败；
- treatment只改善train，不改善dev sequence/exact或paired input sensitivity；
- 只有UTF-8、end-marker、boundary或surprise改善，没有内容覆盖/序列收益；
- bridge ablation不撤销treatment的核心收益，或收益只表现为碰撞上升；
- 三个seed方向不一致且没有预先冻结的稳定性解释；
- 任何结果需要读取final、task label、reference response或人工语义标签才能成立。

dev阶段只有在三seed treatment相对control的主要方向一致、至少一个非代理序列指标改善、paired改写敏感性不下降、bridge ablation撤销主要收益、且所有保持/恢复/预算门通过时，才允许进入一次性final读取。否则H3.6-B以负结果结项，回到数据/target/representation/readout判读，不追加同质训练。

即使dev与final方向成立，也只允许进入H3.6机制复评；不自动形成S2、L2、Mini、R3在线适应或默认入口授权。

## 6. 当前执行顺序

1. 将本合同和JSON合同加入提交；
2. 为训练入口增加显式H3.6 target geometry选择、train-only encoder拟合和target payload digest报告；
3. 对control/treatment各跑零步保存恢复、容量/磁盘前置，不开始epoch；
4. 训练授权已执行完毕；六个dev child与三组只读bridge ablation均已完成。seed方向、非代理序列收益和bridge因果门未同时成立，按合同停止，不读取final、不追加epoch。
