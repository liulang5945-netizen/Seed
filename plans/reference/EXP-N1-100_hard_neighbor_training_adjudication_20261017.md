# EXP-N1-100 · 平衡近邻训练批次判读 · 2026-10-17

## 结论

九模型批次完整结束，工程检查全通过，三个 position seed 均满足原训练充分门；PLAN-N1-100 的训练阶段通过。此结果只说明冻结训练集、配对问题、改名和 transition quartet 上的训练门通过，不是 v7 未见能力证据，也不代表 N1 C3 已成立。`can_promote=false`，`TRAIN_SHA=None`。

主报告：`reports/taiji_n1_hard_neighbor_training_20261017.json`，SHA-256 `dad10ec72c605367f99da035bab07d522ce1793154915ebcc3c7f1022f411909`。训练 attempt、逐模型进度和九个 loss/measurement/score shards 与主报告同批封存。

## 固定配方与执行完整性

- 每模型24,576次更新，seeds 11/12/13 × position/terminal/cut，共221,184次更新；每模型12,288原始视图、12,288 hard views，共110,592次 hard-view更新。3072项schedule循环四次，SHA-256 `dc3a96ad17d43bd898e403ecd119c12c781317a71ab06ba39cdab6db24c40d50`。
- hard schedule、观测/距离格、source/item身份、五组relative-update、单次Adam proposal、末周期学习率和所有九模型轨迹均由运行期守卫复核。九模型各自的hard schedule审计、LR schedule、zero/final独立恢复均有效；预检九次恢复加正式18次恢复全部相等。
- 43,200条固定视图评分行、497,664个训练QA序列、387,072个训练session均与执行器重算数一致。总CPU 9,967.125秒，其中训练9,916.734375秒、zero checkpoint预检50.390625秒，低于18,000秒上限。
- 98项来源守卫未变，代码身份未变，CPU accounting完整。无v3 dev、v3 holdout或确认面查询；v7 face在训练门通过后单独生成。

## Position训练门

每类原题与改名题均为48/48。原门按原始分母执行：

| Seed | 原题 | Pair | 改名 | 改名 pair | 原/改名 quartet joint | 结论 |
|---|---:|---:|---:|---:|---:|---|
| 11 | 192/192 | 96/96 | 192/192 | 96/96 | 48/48、48/48 | 通过 |
| 12 | 192/192 | 96/96 | 192/192 | 96/96 | 48/48、48/48 | 通过 |
| 13 | 192/192 | 94/96 | 192/192 | 92/96 | 48/48、48/48 | 通过 |

三个position seed均超过原题/改名各173/192、各类别40/48、pair/改名pair各87/96和quartet joint各44/48的冻结门。未改门或追加更新。

## 对照臂与边界

Terminal臂在三个seed均低于position；position与cut的训练读数接近，cut只有seed11的pair 93/96、改名pair 94/96，其余cut汇总均为满分或接近满分。对照臂仅用于同批描述，不改变position训练门或之后的确认判据。

所有最终checkpoint已冻结并可独立恢复。〔㊵-1054〕v7面已按固定seed107生成，construction report `reports/taiji_n1_hard_neighbor_v7_face_checks_20261017.json` SHA-256 `d0f1c8866557e2433d6d4d9d289ca6d71163907e4e23871ccb80938c3ac0a7bd`，manifest `plans/manifests/n1_hard_neighbor_v7_20261017.json` SHA-256 `c5a7ef0bc18bf9376e5038348204e838bbc11d88ad6784605c858b12f6d992dc`。40项、四类各10，sacrifice/confirm各20且每类各5；150个实体、120个材料值分别与v3 train/v5/v6集合零重叠，110个干扰实体距离均为d=1；13项面构造机检全过。造面阶段0模型加载、0更新、0查询。当前唯一下一步是执行20项sacrifice范围门；只有`usable_endpoints≥6/9`通过后，才查询confirm并一次性比较九枚新端点与旧九端点。四项v7确认门沿用原值；N1 C3、保持集、C4与三器官证据出口仍独立开放。
