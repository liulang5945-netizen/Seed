# EXP-N1-31 · 九模型训练前置裁决（2026-10-15）

30已弹窗批准，最终代码冻结`9d200a42c`，按[SPEC31](SPEC-N1-31_relative_paired_execution_contract_20261015.md)完成九模型55,296更新、1728组最终train teacher/自由评估、18次zero/final独立恢复。CPU939.796875秒<1200，参数/源码/manifest/29源报告/全部final档及计时守卫全过。

[正式报告](../../reports/taiji_n1_relative_paired_20261015.json) SHA `b10fe3092c9fa6abd08f64fed48f02e59c4261a20c58a49a85f9702d0c5e06dd`。v2预检累计14.0625秒（含v1的6.78125秒），[v2报告](../../reports/taiji_n1_relative_paired_preflight_v2_20261015.json) SHA `e1c6ae0571eb14e9e346b798bae249f240bde15f38736ecf1e345e83f85f6c42`；原v1留存，修正协议后才训练。

| seed | position /192 | terminal /192 | cut /192 | position四类lookup/revision/transition/unknown，各/48 |
| --- | --- | --- | --- | --- |
| 11 | 190 | 91 | 104 | 48 /47 /48 /47 |
| 12 | 180 | 101 | 90 | 46 /48 /38 /48 |
| 13 | 180 | 96 | 71 | 48 /48 /39 /45 |

三个position总命中均≥173，但seed12/13的transition38/39<40，所以训练充分性前置失败。协议工程执行完整（complete=true）与能力门通过不同；status=training-insufficient、can_advance=false、can_promote=false。**holdout/dev模型查询均0，未建立pre_holdout_freeze；报告中holdout计数零值只是空集合，不是0/32成绩。** 本次没有泛化结论，也不能把train对照差当holdout配对收益。

仅分析已保存输出和train标签，新增模型查询0：position错误分别2/12/12项，全部自然boundary；seed12有10个transition错误（south→north6、west→east3、east→north1），seed13有9个（west→east8、south→north1）。teacher错误与自由错误项一致，不能仅归因自由滚动或停止；不是全部材料提取学不会。seed11相同结构transition48/48，seed12/13查词/修订94/96、96/96，说明当前证据尚未证明结构容量不足，也未证明更多训练必然有效。

相关78测试通过（7.13秒）；最终协议修正及计划守卫30通过（7.32秒）。根远端CI仍为旧eaf53e7失败、m7-ci成功，不代表本HEAD全库绿。全项目C1工程过、C2原严格零收益；C3跨seed充分性/未见/保持、C4净增益、同bundle三器官及其余N/CI债未齐。

31按原规则停止，不在本协议补训、重跑或放松40门。唯一建议另立[32固定九模型等量续接](PLAN-N1-32_fixed_paired_continuation_review_20261015.md)，检验是否仅固定训练剂量不足；保留本次失败，未见集继续封存。新更新上限必须用户弹窗决定，批准前不执行。
