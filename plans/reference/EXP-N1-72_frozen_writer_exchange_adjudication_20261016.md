# N1-72 · 冻结writer交换诊断裁决

完整6912/6912已知train/probe组，CPU96.5625秒≤180；六个native条件各576行逐项复现70；全部条件parent参数/Adam/RNG/历史恢复相同，代码与来源守卫通过。零训练/副本更新、零dev/holdout/v4查询。

结果`reports/taiji_n1_frozen_writer_exchange_20261016.json`，SHA `a26b9e467f735c813a9b1ec97638bf811d82edfd23f843c95812154c124fa827`，3896147 bytes。

|seed|条件|原题/192|pair/96|已知改名/192|改名pair/96|
|---|---|---:|---:|---:|---:|
|11|position native|192|96|191|96|
|11|position换母writer|192|96|191|96|
|11|cut native / 换position writer|192|96|191|96|
|12|position native|190|96|188|96|
|12|position换母writer|192|96|187|95|
|12|cut native / 换position writer|192|96|191|96|
|13|position native|192|96|190|96|
|13|position换母writer|192|96|190|96|
|13|cut native / 换position writer|192|96|191|96|

逐项交换对照：seed11/13 position换母writer，及全部cut换position writer，各576项free增/损0/0、raw bytes变化0、teacher symbols变化0。seed12 position换母writer：free增2/损2、raw变化6、teacher变化5；不是完全无影响。writer参数相对母档L2漂移约11.48%/11.81%/10.75%，所以不能把“参数确实更新”冒充行为收益。

结论仅限最后6144步：其writer更新不是这组任务高分的主要来源，少数协同偏差存在；不能泛化为所有历史信用无效。cut始终继承52母档训练过的writer，并保留共享embedding的其他学习通道。71原门仍失败，v4已消费，不追认通过、不重查、也不为追求优势削弱cut。

唯一下一步73零训练完整信用来源与下一学习问题审计：从既有protocol/moments/源码明确阶段cut和端到端cut的边界，联系根需求的取回/信用/联合行为缺口，冻结一个不继承该混杂的新初始化工程合同。保留当前候选、全部负证据及旧checkpoint，不再盲目增加最后阶段融合/读取模块。C4/旧能力保持/同bundle三器官和N系列出口仍欠。
