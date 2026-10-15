# N1-63 · 三种子三臂固定训练裁决

九模型完整，能力判定未通过，64确认不准入，v4保持零查询。不补训、不采用中点、不改原门。

总55296更新、82944 QA序列、15552条训练/已知probe评分；CPU1798.5秒，低于2400秒。18次独立zero/final恢复通过，终止后18档字节再次核对一致；来源、源码与CPU计账守卫通过。执行器提交`f3951166d`。

| seed | arm | 原题/192 | pair/96 | 已知改名/192 | 改名pair/96 | 原四门 |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| 11 | position | 191 | 96 | 190 | 96 | 通过 |
| 11 | terminal | 78 | 0 | 43 | 0 | 未通过 |
| 11 | cut | 192 | 96 | 192 | 96 | 通过 |
| 12 | position | 191 | 96 | 185 | 95 | 通过 |
| 12 | terminal | 87 | 1 | 59 | 0 | 未通过 |
| 12 | cut | 192 | 96 | 190 | 96 | 通过 |
| 13 | position | 174 | 95 | 156 | 95 | 未通过 |
| 13 | terminal | 89 | 0 | 59 | 0 | 未通过 |
| 13 | cut | 192 | 96 | 188 | 96 | 通过 |

seed13 position原题四类48/48/37/41，transition低于40；改名48/47/27/34，总分和transition/unknown类门均不足。该seed中点原题192、改名190，说明有后程退化，但中点未预注册为可选档，不能替代终点。cut已知课程读数更强，不能据此声称完整观察信用有净优势；也不能把共享52母档的新关系初始化三seed说成端到端独立训练。

报告`reports/taiji_n1_relation_paired_train_20261015.json` SHA256 `d982909f6722d37040fb21f7d869aed6817f3a42a3c4b08300b6746ef14e2922`。checkpoint目录`data/checkpoints/n1_relation_paired_train_20261015/`，保留全部zero/final及完整Adam/RNG/history。v3 dev/holdout与v4查询均为0。

唯一下一步：[65离线退化诊断](PLAN-N1-65_relation_seed_drift_diagnostic_20261015.md)，先核对逐题与loss轨迹，不盲改结构。完整项目仍欠旧能力保持、C4净收益、同bundle三器官和N系列其他出口；本批不支持默认推广、开放域或AGI主张。历史CI债保留，未远端推送。
