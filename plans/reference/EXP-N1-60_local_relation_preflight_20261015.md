# EXP-N1-60 · 关系tape工程裁决

60关系tape零训练v2通过，累计CPU26.00秒（含v1失败与定位），62056参数，初始逐位同52/旧Adam/信用/cut/独立恢复通过；唯一下一步61固定6144更新/≤300秒CPU，原QA+0.25保真及四门不变，0dev/holdout。

报告：[v2工程证据](../../reports/taiji_n1_local_relation_preflight_v2_20261015.json)，SHA256 `8cc7500a303f7a3c1eeef01affb033a8ca5955e3dd30b5c38e9bacb14ee70736`。实现6ac261dbb，仪器修复6d7034f19。首次v1报告保留为rejected：随机信用探针污染全局RNG，参数/Adam/session逐项相同而rng不同；v1 CPU9.40625与定位7.203125计入v2累计26.0秒，没有放宽上限。

14工程检查全过，新增6576至62056参数；初始logits/二hop地址逐位同52，旧Adam35项保持，新6项无moment。非零头后关系GRU/local/问题/writer信用与cut观察分支屏蔽、source独立/首步一次后续复用/严格恢复/UTF8变长合同完成，最终4合同通过。0真实课程更新/评分、0自由生成/dev/holdout。旧失败与全项目出口债保留。
