# PLAN-N1-152 执行结案：长度响应逐题配对归因

**状态：完整结束。**只读取N1-151A封存结果中的1,728条长度分层行；0模型/检查点加载、0查询、0更新，CPU0.046875/30秒，`can_promote=false`。

## 核验结果

- 6个endpoint×8/16/32行×4类，每层24项；candidate/control在每个seed、长度、题目ID上完整配对。重复ID、缺层、类别不平衡均为0。
- 长度混合candidate已知类总体命中为8/16/32行的216/216、216/216、215/216；8行control为215/216、210/216、190/216。32行时的差异集中在lookup：candidate 71/72、control 48/72；revision为72/72对72/72，transition为72/72对70/72。
- 逐题candidate/control配对中，32行lookup有23项candidate单独答对、0项control单独答对；transition另有2项candidate单独答对。该项差异来自同一训练任务集。
- unknown字面弃答在candidate的8/16/32行是42/72、60/72、65/72，control是72/72、69/72、70/72。8行时control比candidate多30项正确弃答；32行时差距缩到5项。
- candidate分支8→16有18项错转对、0项对转错；16→32有6项错转对、2项对转错；8→32有23项错转对、1项对转错。control分支8→16有8项对转错、0项错转对；16→32有21项对转错、2项错转对；8→32有27项对转错、0项错转对。

这些读数显示，在本批train-only分层面上，candidate维持了较长上下文的已知类命中，而短上下文unknown弃答偏弱；control短上下文弃答较稳，已知lookup在长上下文下降。它们是同一训练集上的描述性配对结果，不证明独立泛化或单一机制因果，不更改N1-132原门，也不晋级。

完整[逐题配对归因JSON](../../reports/taiji_n1_plan152_length_itemwise_attribution_20261020.json)，SHA-256：`97277a783b6b3e6d62a56d87feeaae9caa2337d411d80bb53f537956981aa0bd`。

## 下一步

唯一下一步是已批准的[N1-153](PLAN-N1-153_fresh_context_length_paired_diagnostic_20261020.md)：使用与N1-150训练及所有已消费面都不重叠的新实体名/材料值，比较这六枚final checkpoint在8/16/32行fresh合成任务上的配对表现；最多2,880条查询，零训练，CPU≤3,600秒，不改变任何既有门，不作能力晋级。目的在于确认本次train-only长度差异是否在新内容上仍可观察。
