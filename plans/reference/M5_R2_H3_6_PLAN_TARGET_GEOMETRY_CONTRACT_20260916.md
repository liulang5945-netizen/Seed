# M5 R2-H3.6 计划目标几何与信用接口合同

> 状态：三种子无训练审计完成；H3.6-A target encoder与trainer保存恢复 smoke、H3.6-B control/treatment零步前置及三seed matched dev/bridge ablation均已完成；H3.6-B按预注册在final前负结果结项。

## 结论

H3.5-A 的 signed-hash span 为每段回答分配近似随机方向，相近回答没有稳定共享结构，因此 planner 在未见 dev 上的 target cosine 接近零或为负。H3.6 用同一12/8/4数据和三个模型种子比较六种32维训练目标，不更新任何 Taiji 参数。

纯 native response state 的平均最近邻迁移较高，但最小样本距离只有0.012–0.028，存在塌缩风险；纯字符 n-gram 的迁移只有0.417；raw hybrid 对种子敏感。唯一同时满足三种子迁移下限高于 signed-hash、且最小样本距离始终超过0.21的是 `train-whitened native response state + compositional char n-gram`，因此选为 H3.6-A。

## H3.6-A 目标合同

1. teacher 只在训练期读取 reference response；运行时 planner 仍只能读取 prefix native state。
2. native teacher 是 protected Taiji 对 reference response 做只读 teacher forcing 后的平均 response state。
3. whitening 的均值、基向量、特征值和有效秩只允许由 train split 拟合，必须进入 candidate checkpoint；dev/final 只能应用冻结变换。
4. compositional 分量由字符1/2/3-gram的确定性 signed projection 构成，用于保持共享局部结构并防止 native teacher 塌缩。
5. 两个分量分别归一化后等权合成并再次归一化；首版不搜索混合权重、宽度或 n-gram 范围。
6. policy、task family、required terms、split 和人工语义标签仅用于只读评价，不进入 teacher、planner 或 renderer。
7. checkpoint 必须拒绝缺失/错误 corpus digest、target format、whitening transform 或 teacher parent digest。

## H3.6-A 实现与 smoke 结果

实现位于`taiji/response_plan_target.py`，训练器接线位于`taiji/language_alignment.py`。真实fixture为12/8/4、corpus digest为`0bc5b5540d4b622f907942f42d7b809e825932e50c6e6c505a3cd091edabd691`；target encoder绑定的parent checkpoint digest为`73a43242887f4bb5d64a55a99d12642204aed709402e7bafb05e331cc13c3108`，target encoder digest为`6aa82199c8f123532fd739e35687bedf1aa0cc6a2e760fc23412d2211b6fa4d9`。本次运行的有效白化秩为11，model context为45维，输出宽度为32，24条train/dev/final target均为有限单位向量。

正式可复核报告为`reports/taiji_r2_h3_6_target_encoder_smoke_20260916.json`。报告证明：

1. fit只读取12条train response；model checkpoint在fit和全量read-only target reconstruction之后保持原digest。
2. target payload可写盘、加载并保持同一target digest；错误corpus/parent绑定在单元测试中拒绝。
3. H3.6 target map已经接入`LanguageAlignmentTrainer`，zero-step trainer checkpoint写入encoder payload和train target map，并能保存恢复为相同checkpoint digest。
4. `training_performed=false`；这不是能力、对话、S2、L2或Mini证据。

> 注：parent digest以本次 smoke 的完整 candidate parent 为准；若后续改变seed、预算、plan readout或代码血缘，必须重新生成独立 encoder，不能复用这个 digest。

## 最小可证伪顺序

H3.6-A 的实现、trainer接线与 target reconstruction/save-restore smoke 已完成；H3.6-B matched dev预注册也已冻结，control/treatment各自的`--preflight-only`已通过：effective=276,610≤300,000，zero-step/child restore、parent保护、atomic save和treatment target血缘均成立，且`training_performed=false`。随后按授权完成10 epoch matched dev与三组只读bridge ablation：control dev sequence为`0/0.125/0.125`，treatment为`0/0.25/0`，均值没有改善且seed方向不一致；exact与required-term coverage均未形成收益，bridge ablation也未撤销可确认的treatment核心收益。因此H3.6-B不读取final、不追加epoch、不晋级S2/L2/Mini；后续必须回到target/representation/readout设计复审，另立可证伪合同。

即使 encoder smoke 通过，下一轮能力对照仍需新预注册；不得读取H3.5-A final，也不得把本次离线最近邻分数写成语言能力。
