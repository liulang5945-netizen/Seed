# PLAN-N1-150B：N1-150长上下文训练CPU上限修订

**状态：依据用户授权已批准并冻结，仅修订资源上限。**本补充取代PLAN-N1-150 §保存恢复、评分与门中的CPU≤7,200秒；训练配方、endpoint、更新数、评分分母、门与数据边界均不改变。

## 修订依据

PLAN-N1-150让candidate的全部9,216步都使用8/16/32行上下文，control的全部步骤都使用8行；此前N1-141相同55,296次总更新和7,488条计分的实耗为2,581.750秒，但其中长上下文只覆盖transition slots的一小部分。把每个任务都放入最长32行会显著增加前向/反向成本，沿用7,200秒有过早停机风险。

## 冻结新上限

- CPU总上限修订为18,000秒，覆盖来源/schedule readiness、六个zero bundle保存/独立恢复、55,296次更新、六个final bundle独立恢复、7,488条train-only计分及末端哈希复核。
- 仍每endpoint9,216更新、六endpoint55,296总更新；每endpoint960条原N1-132门+288条length描述，六端点7,488条总评分。
- 预算用尽即停止并保存partial checkpoint/attempt；不因接近上限而削减上下文、改变schedule、减少评分或追加训练；one-shot attempt不重试。
- 0 dev/holdout/confirmation查询，原门不变，`can_promote=false`。
