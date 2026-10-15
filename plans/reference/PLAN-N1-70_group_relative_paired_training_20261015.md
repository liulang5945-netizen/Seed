# PLAN-N1-70 · 固定分组实际位移三seed三臂正式训练

## 前置

69工程完整，报告SHA `938bee47ae766c24efdfc20c1e43c49332e518e4d7e59e65c78ba1c57dd743c7`，九共同初态/九恢复/控制信用合同通过。正式执行前先解除69-CI中新机械合同的本机checkpoint依赖，模型与冻结工程身份不修改。

## 固定执行

从52完整母档按62迁移重新创建seed11/12/13 × position/terminal/cut九fresh模型，不用68终点或带任何合成更新的工程副本。每seed三臂完整bundle指纹必须与69/62初态相同；62056参数、35旧Adam状态、原会话/RNG保持。

九模型按seed、arm串行，各6144正式更新，总55296/82944正式QA序列；同43固定3072改名流/40双问题、QA均值+0.25保真、Adam lr0.001，67同五组relative0.001/floor1.0实际位移约束。三个arm唯一差异为原读取/观察信用干预，不能弱化cut或给position不同预算。

独立本机CPU≤2400秒，线程1，包含前置、所有评分/更新、九zero及九final保存与独立恢复。每档主进程和worker各一合成下一步副本，共额外36机械更新/72条合成QA，单独列账并计入同预算。每模型zero独立恢复通过后才能训练，final独立恢复后才认完整。

0/3072/6144各完整四视图（每次576组），九模型共15552组已知train/probe teacher+free；position零档全576项逐项复现63对应seed原零档。每步loss及五组实际/提议位移全保留，九模型全部固定次数完成，不据中点或某seed选胜者。原会话逐项不变，每模型新增6144会话；所有41项moment及逐参数step检查，cut writer保持母档18432，其余按原信用+6144。

来源52/62/63/65/66/67/68/69报告、原档、完整课程、严格v4双seal与所有代码身份执行前后守卫。0v3dev/holdout、0v4模型查询，只核已冻文件字节。预算/工程错误保存当前partial并停止整批，不重启或续训；所有旧证据保留。

## 原门与首次确认前置

三个position终点各须通过原四门：原题≥173/192且每类≥40/48，pair≥87/96，已知改名≥173/192且每类≥40/48，改名pair≥87/96。控制臂不要求达到position门但固定6144更新全跑。任一position不足则失败，v4保持零查询，不追加或改门。

仅九模型完整、18独立恢复和全部position四门/守卫通过且代码/档案冻结后，另冻71一次性九模型×32＝288条v4确认；沿14原门，每position seed≥16/32且每类≥2/8，三seed position合计各领先terminal/cut≥12/96。评估后v4消费，不重训再当未见。

## 产物与全项目边界

执行器`scripts/training/run_taiji_n1_group_relative_paired.py`，合同`tests/taiji_native/test_native_group_relative_paired_training_contract.py`；checkpoint目录`data/checkpoints/n1_group_relative_paired_train_20261015/`，exclusive完整报告`reports/taiji_n1_group_relative_paired_train_20261015.json`。逐步完整读数采用无损紧凑JSON保存，不删字段或样本，另写可读裁决。

共享52母档的新关系初始化三seed不是端到端独立训练。C4、旧能力保持、同bundle三器官、N系列其他出口和CI历史债保留；不推送、删除数据或推广默认，不作开放域/AGI或项目收官主张。
