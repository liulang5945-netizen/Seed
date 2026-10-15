# EXP-N1-40 同观察双问题固定校准裁决

用户弹窗批准40，冻结1d52acd64、pair manifest SHA1c3857b1ae66c227503d443ea13051a4dbaa0b65707f4fb259a3fe64c222fddb后，37残差模型fresh seed11固定9216更新。同观察两问题QA各0.5、观察保真0.25仅一次；训练QA序列精确13824，192原项顺序/轮数不变。35相关合同通过，Ruff/B/SIM过。

| 更新 | 原题/192 | lookup/revision/transition/unknown，各/48 | pair/96 | 保真 |
| --- | --- | --- | --- | --- |
| 3072 | 67 | 0/0/25/42 | 0 | 100% |
| 6144 | 86 | 1/4/48/33 | 2 | 99.9443% |
| 9216 | 127 | 8/32/47/40 | 26 | 99.8886% |

[正式报告](../../reports/taiji_n1_paired_question_train_20261015.json)完整，864组teacher＋自由，CPU219.375<300，zero/final独立恢复、全部参数非空Adam/RNG/来源/状态、源码/双manifest/源工程报告守卫过。0 dev/holdout。original_training_sufficient=false、pair_training_sufficient=false、training_sufficient=false；新增配对分数不能代替原门。没有新未见能力结论，不能放宽门或追加训练。

此结果没有建立新目标优于原目标：38原题140，40原题127且配对仍26；同seed但不同QA序列量/CPU，不作公平能力因果结论。pair由0→2→26表明不是完全不学习，但不能据斜率推断额外训练必通过。

唯一建议41从33固定seed11已经191/192的relative终点迁入零残差候选，先逐项证明旧行为保留，再固定学习已批准的配对目标。这是新初始化/配方，非继续40或默认升级。33未见失败、36/38/40训练失败保持，全部checkpoint留存。全项目C1工程过，原C2严格收益零，未见门/旧能力保持/C4/三器官/其他N及根CI债未闭合。
