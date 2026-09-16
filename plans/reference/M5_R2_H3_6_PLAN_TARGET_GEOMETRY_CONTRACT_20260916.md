# M5 R2-H3.6 计划目标几何与信用接口合同

> 状态：三种子无训练审计完成；选定 H3.6-A 唯一候选，尚未授权新一轮能力训练。

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

## 最小可证伪顺序

先实现版本化 target encoder、train-only fit/apply、payload/digest 和恢复测试；随后只做 target reconstruction smoke，要求 train target 可学、dev target cosine 不低于零且不破坏parent。完成前不运行10 epoch matched训练。若 encoder 本身在恢复、split隔离或geometry复算上不一致，立即停止。

即使 encoder smoke 通过，下一轮能力对照仍需新预注册；不得读取H3.5-A final，也不得把本次离线最近邻分数写成语言能力。
