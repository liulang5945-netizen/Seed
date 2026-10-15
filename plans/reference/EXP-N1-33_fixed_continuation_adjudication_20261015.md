# EXP-N1-33 · 固定续接与首次v3留出裁决（2026-10-15）

用户弹窗批准32，冻结`8f53d4b6a`后按[SPEC33](SPEC-N1-33_fixed_continuation_execution_contract_20261015.md)完成九模型等量追加，各3072、共27,648更新（历史累计82,944，各模型9216）。参数/Adam/RNG/来源沿31各终点完整续接，不挑seed/臂、不重置优化器、不修改31失败结果。

[预检报告](../../reports/taiji_n1_relative_continuation_preflight_20261015.json) SHA `9ce670f590bbe917d03715c3f95bc1c86855218299734b0683610b6e477001c4`，全部九输入、零更新信用和新zero独立恢复通过，CPU13.15625秒计入本批总额。[正式报告](../../reports/taiji_n1_relative_continuation_20261015.json) SHA `871f054fcff6a72f713849de2a8b3b1ae6680e4581bccad317fef65a3c133045`。

累计CPU493.234375秒<600，含工程、正式父和18恢复worker；1728组最终train齐备。三个position191/192、192/192、191/192，四类最低47/48，充分性前置均通过。九final档/参数/代码/协议冻结并写pre_holdout_freeze后，才首次评估全部288组v3 holdout（每组自由输出＋teacher副读数），dev0。所有旧输入/新final档、参数/数据/源码/源报告及预算守卫通过。

| seed | train position/terminal/cut，各/192 | holdout position/terminal/cut，各/32 | position holdout四类lookup/revision/transition/unknown，各/8 |
| --- | --- | --- | --- |
| 11 | 191 /134 /151 | 18 /6 /4 | 3 /5 /5 /5 |
| 12 | 192 /147 /142 | 15 /5 /10 | 1 /8 /5 /1 |
| 13 | 191 /151 /109 | 16 /6 /5 | 1 /7 /5 /3 |

position合49/96，terminal17/96、cut19/96；平均优势10.6667/32与10/32，三seed均高于两控制，观察到此任务面的一致相对差距。但seed12仅15<16，且seed12 lookup/unknown、seed13 lookup均1<2，原候选小试门未过。complete=true表示执行完整，can_advance=false、can_promote=false；不能仅因均值差满足而放行，更不代表C3/C4/N1或整个项目通过。

只分析已存输出/标签、额外模型查询0：position的lookup原本24题合5正确，其余包括7个另一观察值、11个unknown与1个其他串；unknown合9/24，15个错误全输出某观察值。revision20/24，而transition15/24且三seed共同错在同三个holdout ID（006/018/030），多数为方向词混淆；不把所有失误归因单一因素。现有证据更支持先诊断实体匹配/存在性，而不是继续扩大训练量。teacher同样错的样本不能仅解释为自由滚动失误。

相关80测试通过（8.04秒）。全项目仍C1工程过/C2原严格收益零；隔离候选出现有界相对收益，但未见门/旧能力保持/C4/同bundle三器官及其他N、人工与根CI债均未齐，不能替换默认或宣称开放域能力。

本次已消费v3 holdout，之后不得把同一面当未见确认来按分数选参或循环补训；32固定追加到此停止。唯一建议[34实体绑定反事实诊断](PLAN-N1-34_entity_binding_diagnostic_review_20261015.md)，仅train源样本变体、零更新，先检验实体重命名与query选择的脆弱性，再决定一个机制修订；任何新配方的确认面须另冻，不能覆盖本次失败。
