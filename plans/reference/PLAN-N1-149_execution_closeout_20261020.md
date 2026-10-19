# PLAN-N1-149 执行结案：8/16/32行嵌套弃答长度曲线

**状态：diagnostic完整执行，无通过门、无能力晋升。**seed126的40个logical pair在8/16/32行均一次生成并查询；2,880条逐题输出、12个endpoint bundle、source/checkpoint SHA均完整，0训练/更新/dev/旧v3 holdout，`can_promote=false`。

## 执行完整性

- v26面240题：40个目标/答案模板×3个嵌套长度×known/unknown；target事实固定索引0，已知/未知同长度配对只在目标事实槽位不同。
- source exclusion并集39,442实体名/16,608材料值，加入已消费v25面1,320名/1,320值，零交集；source hash均保持不变。
- 12/12 endpoint完整bundle独立恢复。face准备CPU3.766秒，readiness CPU135.750秒，query CPU1,779.781秒；累计1,919.297/7,200秒。
- 12端点各240题，共2,880条；progress逐行保存，attempt/progress SHA与result绑定。配对汇总由2,880条结果独立重算一致，执行无错误。

## Known回答与Unknown弃答

| endpoint family | known正确（8行） | known正确（16行） | known正确（32行） | unknown弃答（8行） | unknown弃答（16行） | unknown弃答（32行） |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| N1-132 parent | 84/120 | 56/120 | 17/120 | 105/120 | 117/120 | 118/120 |
| N1-140 block candidate | 94/120 | 66/120 | 28/120 | 115/120 | 119/120 | 120/120 |
| N1-141 candidate | 82/120 | 41/120 | 14/120 | 118/120 | 120/120 | 120/120 |
| N1-141 control | 101/120 | 79/120 | 32/120 | 112/120 | 118/120 | 119/120 |

在每个family的120组成对比较中，known正确数均随8→16→32行严格下降。8→16与16→32所有480个相邻endpoint×pair比较中，均无“短长度错误、长长度正确”的恢复；N1-141 candidate分别有41/120和27/120组由正确转错，8→32累计68/120组失去正确答案。unknown弃答随长度维持或上升。

这支持本诊断面上的描述性结论：追加非目标干扰行时，模型更常将本来可答的known问题转为弃答；短上下文known查询还有较多正确输出，unknown条件则保持高弃答。此读数不能确定内部机制，也不是未见确认或能力晋升证据。

## 后续

依据用户授予的执行计划批准权，已批准并冻结[PLAN-N1-150](PLAN-N1-150_matched_long_context_training_continuation_draft_20261020.md)：从同seed N1-132 position父bundle配对续接length-mixed candidate与8行control，各9,216次更新；candidate按8/16/32行等量训练，control全为8行，保留原192 train任务与N1-132冻结门。训练前须完成完整bundle保存/恢复前置；不访问dev/holdout/confirmation，不自动晋升。

## 绑定证据

- N1-149计划SHA-256：`d6dfb5a9e6e6f5d563a7d137ef838a197c7bbff98cbca52d85fe4a11de2e6497`
- v26 face SHA-256：`8b5486cb2ee929aeacf3a50b19e117813fcecafdbdc06efc9ac0eea208651db8`
- attempt SHA-256：`865efbb0f18ea8dc312f142f3ab175be24b08cb7f943d2b99989d763a47f6ede`
- progress SHA-256：`142719da9fe5d3da83cce24a4a33926599f09339f9a8ba5bfcba1e655f81d8a5`
- 结果SHA-256：`878e2457072f81f508312993a10b35e3bb70151b709fa752ede79550d57a4ea8`
