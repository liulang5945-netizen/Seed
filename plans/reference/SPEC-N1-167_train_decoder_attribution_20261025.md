# SPEC-N1-167：训练侧decoder诊断预注册

日期：2026-10-25。冻结于模型/训练样本诊断前，承接[PLAN-N1-167](PLAN-N1-167_train_decoder_attribution_20261025.md)。这是描述性归因包，不修改旧评价分数线。

## 固定分母与记录

4个共同train样本×4端点×3读出=48计划单元；每单元TF最多128目标字节，自由生成最多128字节。oracle在没有配置plan的端点记N/A，不当0分，不删除单元。另有16项原训练路径只读复放、16项循环错配prompt TF及16项configured生成装配核对。

逐项记录episode/prompt/target digest、实际readout owner、参数/目标geometry、原始与mask后TF top1/NLL、首错、0–15/16–31/32–47/48–63/64–127相位组、输出字节/文本/停止与目标前缀匹配。对照只按预定顺序，不挑最好样本/端点/中间checkpoint。

## 解释规则（按顺序，不要求一定选出机制赢家）

1. 原configured与仪器configured在同限制下不逐字节一致：仪器/调用装配失败，先修本包，不能解释模型能力。
2. TF top1≥0.80且自由生成目标前缀匹配<0.20：提示前缀暴露/滚动误差问题；仅适用于满足该条件的题/端点。
3. oracle相对configured TF top1提升≥0.10且NLL降低≥0.50 bit：提示预测plan相对renderer的瓶颈；oracle泄漏答案，不是自主成功。不满足不证明plan无用。
4. protected相对configured达到同样TF增益线：提示配置读出路径/目标竞争值得优先检查；不以一次分支切换直接改默认模型。
5. configured/protected/oracle的TF和自由生成均差：不能只把问题归为decoder约束，也不能靠oracle阴性断言架构不可学；优先检查已有readout/目标/更新责任的可学性。
6. 错配prompt的同真前缀平均JS<1e-6且top1一致≥0.99：此题/端点的受测路径对该干预几乎不敏感；不同于“整个模型没有信息消费”。其余值连续报告，不造第二次通过门。

不把最大字节数停止或UTF-8合法直接当可读/贴题；本包train目标只用于明确标记的TF、oracle和诊断评分，不访问dev/holdout/reference/N1-166。

## 不变量与失败

0参数/优化器更新、0checkpoint写入、原文件SHA不变。每单元还原完整模型snapshot并核对digest；oracle只更改动态plan状态。计数/身份/restore/数值非有限或原件变化立即停止留证。超预算按partial结束，不缩分母宣告complete。

CPU≤1800、wall≤2400；完成固定矩阵或资源/工程出口即收束，报告唯一后继建议。任何新增训练、容量/结构变更、额外评价范围均不由本SPEC自动授权。
