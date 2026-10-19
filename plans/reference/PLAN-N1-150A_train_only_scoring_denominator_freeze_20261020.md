# PLAN-N1-150A：N1-150 train-only计分分母补充冻结

**状态：依据用户授予的执行计划批准权已批准并冻结。**此补充只把N1-150已有的“原门+8/16/32行描述性读数”锁定为精确计分分母，不改变训练配方、目标、endpoint或N1-132门。

## 冻结计分量

- 每endpoint运行冻结N1-132原始train gate共960条，原有分母、类别、阈值及合取规则逐项不变。
- 另对同一批v3 train父任务做8/16/32行长度分层描述性评分，每种长度96条（lookup/revision/transition/unknown各24条），共288条/endpoint。候选与控制使用相同的parent item ID及length-view ID。
- 六个endpoint合计7,488条train-only计分（6×(960+288)）。新增length分层不构成训练门，不据此晋升或修改N1-132原门。
- scoring期间冻结模型、Adam、RNG、history；不访问dev、holdout或confirmation。

## 绑定

- N1-150主计划SHA-256：`21416667c74656a50f21261d2764794910a6366e25fbdeac86eb87981cc768ed`
- N1-150训练仍为六endpoint×9,216更新=55,296次，CPU≤7,200秒，保留zero/final完整bundle save/restore前置。
