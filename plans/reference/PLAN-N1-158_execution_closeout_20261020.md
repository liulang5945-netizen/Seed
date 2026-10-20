# PLAN-N1-158 执行结案

**日期：2026-10-20**  
**状态：完整结束；仅为seed128 synthetic face描述性归因**  
**计划：** [PLAN-N1-158](PLAN-N1-158_plan157a2_fresh_face_itemwise_attribution_20261020.md)  
**报告：** [逐题配对归因](../../reports/taiji_n1_plan158_plan157a2_fresh_face_itemwise_attribution_20261020.json)

## 结果

对N1-157A2的2,880条已消费结果完成逐题配对，形成36个seed×长度×kind分层、72个branch×seed×kind×长度相邻长度转移分层。原结果、progress、face与source哈希均通过；CPU1.65625/30秒，0模型/检查点加载、0推理、0新查询、0训练/更新、0 dev/holdout/confirmation，`can_promote=false`。

三seed合并后candidate exact为1,401/1,440，N1-150父模型为1,341/1,440。已知lookup/revision/transition合计candidate 1,076/1,080、父模型1,078/1,080；差距来自unknown：8/16/32行candidate分别96/120、110/120、119/120，父模型分别65/120、93/120、105/120。逐题unknown配对的candidate-only/parent-only依次为35/4、20/3、14/0。

这是seed128 synthetic face上的描述性读数，不是confirmation、因果隔离或能力晋级。完整配对与长度转移表见[归因报告](../../reports/taiji_n1_plan158_plan157a2_fresh_face_itemwise_attribution_20261020.json)。

## 后续唯一队首

依据用户授予的执行计划批准权，已批准[PLAN-N1-159](PLAN-N1-159_seed129_candidate_parent_fresh_replication_20261020.md)：用新seed129、来源完全排除seed128 face的fresh面，按相同六端点比较N1-155A candidate与N1-150父模型，最多2,880条查询、CPU≤3,600秒。此为独立复制，不训练、不改变门、不晋级。
