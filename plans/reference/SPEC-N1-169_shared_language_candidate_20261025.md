# SPEC-N1-169：共享条件语言候选E与一次train-only标定

日期：2026-10-25。真实更新前冻结，承接[已采纳PLAN](PLAN-N1-169_shared_language_chain_design_20261025.md)。本包只验证工程和已见可学性/费用，不是泛化、同预算算法胜负或默认采用。

## 图与状态不变量

1. TaskCore256/embedding64/key64、16槽/32字节chunk、257字节字表；真实原始字节及来源跨度保留，workspace不足时只按固定滚动槽规则处理编码，不静默删原文。观察总量上界4096，超过拒绝。
2. TaskState锚定bundle/scope/session/native-state-ref和参数revision；turn reset保留任务hidden/证据，session/development明确清空。参数更新后旧state/cache不得继续生成。旧CognitiveState仍是旧器官真值，新增ref不是宣称它们已经共享新语义。
3. logits[t]仅见prompt及target[:t]；同一个next_logits用于TF/生成；候选输出为原始byte，不由renderer补答案。真实条件路径不断梯度，保存/外部载入时显式detach；训练batch从头重建图。
4. 状态/证据交换改变实际logits，恢复原状态救援；元数据ID改名本身不改logits；序列损失到encoder/输入byte embedding、读取和decoder均有可观测梯度；detach编码路径后对应梯度消失，不以loss下降替代此检查。
5. 新checkpoint严格格式/version/digest，含AdamW、局部RNG、参数revision、TaskState/UTF-8生成游标及课程cursor；零步和一次更新后、下一生成和下一合法更新一致，unknown/缺件拒绝。旧native默认路径和checkpoint前后摘要不变。

## 一次标定固定范围

- seed169，4个既有train episode按N1-167顺序。prompt不截断（均≤512），target取首128 byte；实际每batch495 byte、64batch=31680，不能冒称训练了完整长回答。
- batch4；AdamW lr0.001、weight_decay0、betas默认0.9/0.999、clip1；不搜索结构/种子/学习率，不选择query成绩最好的中间档。
- 0/1/8/32/64步各对4题记录未掩码TF、完整汉字TF及128字节贪心生成与目标前缀匹配；0/1/64三个bundle均新进程恢复。同一batch下一更新的等价性另在合成数据测试。
- 控制为原full-turn parent及N168 child的只读同样4题/128字节诊断与参数/时间记录，不能宣称它们与fresh候选拥有相同历史。正式对照未执行。
- E任一失败先修可逆工程；R若资源/数值/保存/身份失败则保留failure/partial和费用，不自动续跑选另一配方。原dev、holdout、reference、N1-166均不读取。

## 判读与有限结束

如4题均TF≥0.80且自由前缀匹配≥0.80，仅称128字节已见切片可学；TF高而自由匹配低则记录暴露/首步/滚动缺口，不能称语言已通过。所有连续读数、逐题差异、字节/字符差异完整报告，不通过降低门或增加步数取得阳性。

CPU≤1200、wall≤1800（含真实恢复/保存/控制诊断），本轮总账不提高。工程验证及固定64步/资源出口完成后冻结正式比较合同，以实测吞吐和完整reserve一次估算费用；候选正/负/不可判均可结束该标定。
