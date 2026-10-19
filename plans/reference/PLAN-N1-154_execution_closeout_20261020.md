# PLAN-N1-154 执行结案：fresh内容长度诊断逐题配对归因

**状态：完整结束。**只读取N1-153A的2,880条逐题结果，按seed/题目/类别/长度计算配对转移；CPU0.03125秒，0模型/检查点加载、0查询、0更新，`can_promote=false`。

## 配对读数

| 分支 | 已知类命中 8行 | 16行 | 32行 | unknown正确弃答 8行 | 16行 | 32行 |
|---|---:|---:|---:|---:|---:|---:|
| 长度混合candidate | 360/360 | 359/360 | 356/360 | 76/120 | 91/120 | 103/120 |
| 8行control | 353/360 | 340/360 | 302/360 | 109/120 | 115/120 | 120/120 |

32行已知类360组配对中，candidate-only正确56项、control-only正确2项、双方正确300项、双方错误2项；candidate优势集中在lookup（55项candidate-only）及2项transition净优势。unknown 120组中candidate正确弃答103项、control 120项，差距17项。

从8到32行，candidate共有34项错转对、11项对转错；control有11项错转对、51项对转错。unknown弃答的变化方向与已知类保持相反：candidate随长度改善但仍低于control。该结果限于seed127 fresh synthetic diagnostic，不代表真实世界语言能力或泛化晋级，也不更改N1-132/133的历史门。

完整[逐题配对归因JSON](../../reports/taiji_n1_plan154_fresh_context_itemwise_attribution_20261020.json)，SHA-256：`da296992376b27fc785067d03b505968dbbee6987a3edc13d6c90773772f7e0b`。

## 下一步

唯一下一步为已批准的[N1-155](PLAN-N1-155_short_context_unknown_rebalance_and_fresh_diagnostic_20261020.md)：在保留N1-132原门及六endpoint匹配续训的前提下，只把candidate未知类训练暴露向8行上下文再平衡，随后用新fresh face测量；不访问dev/holdout、不改晋级门，`can_promote=false`。
