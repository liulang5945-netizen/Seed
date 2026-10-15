# PLAN-N1-37 保留原表示的问题残差对齐

状态：2026-10-15用户选择弹窗已批准37，实施独立残差候选及零更新工程验证；不含正式训练。依据[36裁决](EXP-N1-36_question_alignment_calibration_20261015.md)，固定9216终点154/192，lookup28/revision30未过40；不循环加训。

唯一推荐：保留35逐问题字节软对齐，但把fused=tanh(fusion([state,context]))改为fused=state+tanh(fusion([state,context]))，融合层权重/bias初始化为零。参数仍44,085（新增3616不变），先保留原模型所有共享层seed11初始化。第一轮乃至多轮输出在新支路为零时必须逐位等同旧相对地址模型；新融合梯度可学，后续question key/value应在非零融合权重下具有信用。观察保真不变，双hop/相对地址及自由边界不变，任意字节可对齐，无硬编码实体/来源/gold模板或教师。

新format与question_mechanism明确残差语义，不静默载入35替换型checkpoint，所有历史checkpoint和报告保留。terminal只用各来源末态；cut对QA与辅助两支无writer旁路；临时问题状态不跨session/tape。不是恢复35继续训练，新训练必须另冻配方。

本次只申请结构实现、合同与独立CPU≤60秒零更新工程预检：旧模型共有参数逐位一致，新零支路多输出步logits/address逐位一致，teacher/free共享、question/source隔离、finite、cut、残差融合初始梯度与非零融合时question层梯度、zero保存/独立恢复参数/Adam/RNG/来源/对齐/输出签名。人工设置合成微型测试融合权重以查计算路径仅限临时测试，不作模型能力或训练；预检不optimizer.step、0课程/dev/holdout/34变体查询。正式训练/全新确认测面之后单独批准。

替代为继续训练35无冻结剂量证据且违背36停止承诺；删除问题对齐无法回应34指向敏感性，均不推荐。残差修订提供明确可验证的内容基线，但尚不证明泛化或训练一定改善。保持/C4/三器官等原出口不变；不默认采用、不推送/删除。
