# PLAN-N1-153A 执行结案：N1-153 fresh-face逐题结果绑定恢复

**状态：修复测量完整结束。**N1-153原一次性attempt保留为失败记录；N1-153A复用同一seed127 face，以独立路径重新完成六endpoint评分。

## 执行记录

- N1-153原attempt完成seed11/length_mixed的480次查询，但逐题行未在summary helper异常前写入最终JSON；原result保留`query_count=480`、`diagnostic_rows=0`、`status=stopped`。六个final bundle之前已独立恢复6/6，source SHA不变，0训练更新。
- N1-153A修复summary绑定并重跑同一face的全部六endpoint：新增2,880条查询，其中首endpoint480条为修复性重复、其余2,400条为原attempt未执行部分；全程不重建题面、不改题目或答案。
- 最终result包含2,880条逐题输出、六endpoint×长度×类别共72个分层（每格40条），六个checkpoint final恢复6/6；SHA与source守卫通过。
- 计入N1-153原attempt后累计3,360条诊断调用，CPU累计350.609/3,600秒；0训练/更新/dev/holdout/confirmation，`can_promote=false`。

## Fresh face描述性结果

| 分支 | 已知类命中 8行 | 16行 | 32行 | unknown正确弃答 8行 | 16行 | 32行 |
|---|---:|---:|---:|---:|---:|---:|
| 长度混合candidate | 360/360 | 359/360 | 356/360 | 76/120 | 91/120 | 103/120 |
| 8行control | 353/360 | 340/360 | 302/360 | 109/120 | 115/120 | 120/120 |

32行已知类的逐题配对中candidate有56条单独答对、control有2条单独答对；双方都对300条、都错2条。unknown短上下文弃答是candidate的主要差距。这是新生成synthetic face上的描述性结果，不改N1-132/N1-133历史门，也不作现实世界能力晋级结论。

完整[逐题结果JSON](../../reports/taiji_n1_plan153a_fresh_context_result_binding_recovery_result_20261020.json)，SHA-256：`af8853b4f4677b469164e3691e429476d02887adc5a47f9981ed5b976926d06a`。原N1-153失败[结果回执](../../reports/taiji_n1_plan153_v28_fresh_context_length_confirmation_result_20261020.json)和N1-153A修复计划均保留。

## 下一步

唯一下一步为已批准的[N1-154](PLAN-N1-154_fresh_context_itemwise_attribution_20261020.md)：对2,880条逐题fresh输出做离线配对归因，按seed、类型、长度统计错转对/对转错；CPU≤30秒，0模型/检查点加载、0查询、0更新。
