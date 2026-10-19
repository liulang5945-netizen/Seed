# PLAN-N1-145 执行结案：v23确认结果逐题错误归因

**状态：只读分析完整通过。**N1-144已消费的960条confirm全部与题面逐项匹配；确认门复算与原结果逐值一致。分析CPU 0.015625秒，未加载模型/checkpoint、未查询或更新，`can_promote=false`。本报告是post-hoc描述，不作因果或能力结论。

## 输入与完整性

- 只读取N1-144的960条confirm行、v23题面及绑定它们的计划/attempt/progress报告；12端点各80题且题目ID集合一致，每类20题。
- N1-144确认汇总复算一致，源文件SHA均匹配；分析结果绑定N1-144计划、v23面、执行结果、attempt、progress及分析代码/测试哈希。
- 5项契约测试通过，Ruff和py_compile通过；本分析没有打开checkpoint字节或训练状态。
- 逐题分层表见[归因结果](../../reports/taiji_n1_v23_itemwise_error_attribution_20261019.json)。

## N1-141 candidate 的描述性读数

| 类别 | 正确数 | 观察位置/答案长度分层 |
| --- | ---: | --- |
| lookup | 2/60 | 起始位置0/8/16/24/31分别1/0/0/1/0（各12题）；6字符0/30，7字符2/30 |
| revision | 60/60 | 五个起始位置各12/12；6/7字符各30/30 |
| transition | 45/60 | 五个起始位置依次8/12、9/12、9/12、9/12、10/12 |
| unknown | 60/60 | 不适用目标事实位置与材料答案长度 |

lookup命中稀少，位置上没有稳定单调趋势；transition在后部位置没有一致下降。每个位置只有12题，答案长度分层每格30题，因此这些只是当前面上的描述性读数，不能据此认定位置或长度没有因果作用。

逐题比较显示差异集中在lookup与transition之间的抵消：candidate相对N1-132 parent的lookup少6题、transition多28题、unknown多1题，合计优势23；相对N1-140 block candidate的lookup少11题、transition多10题，revision/unknown相同，合计少1题；相对N1-141 control的lookup少13题、transition多23题，revision/unknown相同，合计优势10。该分解仍只描述已消费v23面，不把单类增益解释为新能力。

## 后续

依据用户授予的执行计划批准权，已批准并冻结[PLAN-N1-146](PLAN-N1-146_v23_output_source_item_attribution_draft_20261019.md)：只对同一批已消费输出做离线字节解码，检查错误材料答案是否对应题内某个实体的事实值，并区分revision陈旧值、邻近实体值和非词汇输出；不新增模型查询或训练。v23不再作为任何新确认面。

## 绑定证据

- N1-145冻结计划SHA-256：`647c70092b6efae729f29849bf77e36032f899b63c05399f8c497e679fa651d9`
- N1-144确认结果SHA-256：`67038a9848b3c7ac088fea1983d9ba40ea153e91c8a4c436db830e8894e80388`
- N1-145归因结果SHA-256：`07e4a18ffae7b2deb94f1bf6e49985839658f77188ecf940eae80e98da3883f1`
