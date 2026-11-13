# 训练期内容监督与字节生成的共同信用校准

> P3.8.1b，2026-11-13。独立固定对照，90 CPU／180 wall，训练CPU≤60、最多2000次全批Adam。预留后未分配900 CPU。

## 1. 命题和控制

[P3.8.1结案](CLOSEOUT-TAIJI_U2_FIRST_ROUND_CALIBRATION_20261113.md)确认只有字节CE的本次基线收敛到常量回答，context变异显著收缩。当前检验任务对齐的内容目标是否改善共享表示及实际生成。该项是新的训练配方，原固定模型／2000步的负结果保持原身份。

模型body使用原step0 checkpoint，65169参数的初值、空Adam、数据顺序与原种子完全相同。控制组复用已冻结的原2000步结果。回归证明新实现中的byte CE、全部body梯度及单步Adam与原配方等值，未使用辅助头时，额外参数组不改变body更新。

本试验结束于同样的2000步或60训练CPU，逐项报告控制的实际更新／样本数和费用。达到2000步时两组各呈现360000例。费用单列，不由更新数相同推定CPU配平。该对照同时改变内容监督和类别平衡，单次比较只裁决整个配方。

## 2. 唯一目标与责任

从train原六字节答案按hex排序提取监督类别，仅在训练和评分中映射到索引。当前四类计数为30／30／30／90，权重w_c=N/(K·n_c)，即1.5／1.5／1.5／0.5，整体样本平均权重1。索引不来自实体真ID或私有坐标。

共享context仍为原CNN／问题／乘性融合之后的tanh向量，完全沿原推理路径。另设Linear(64,K)，使用head seed2026111310、标准初始化，新增260个训练参数。body／head各一组Adam，超参数相同，head moment从缺省开始。

\[
J=J_{byte}+J_{content},\quad
J_{content}=N^{-1}\sum_n w_{c_n}\,CE(W_c h_n+b_c,c_n).
\]

byte CE沿原7符号均值，content CE按180样本均值，系数均1。内容目标更新图像／问题编码和共享context，GRU与257字节头由byte目标负责。辅助头由content目标负责，梯度覆盖和None模式在回归中检查。

训练checkpoint保存body、辅助头、两个Adam参数组、类别／权重来源、更新号／样本数、RNG、运行条件和源码。step0／step1／每50步／末点均按原规则保存及恢复，字节概率／全部23参数梯度与Adam由独立worker比对。

## 3. 推理隔离与判据

生成与评分只调用原FirstRoundBaseline，辅助头为独立对象。末点导出只含body和其Adam的原schema checkpoint，由原factory恢复并重新进行全部train／dev greedy生成。移除辅助头后的输出须精确一致。运行字母表仍为257字节，UTF-8／64字节／EOS保持原定义。

主判据仍为首问train≥0.95、dev≥0.80、每种原始问题dev≥0.80。分类头准确率及context饱和／变异只作诊断。训练标签类别和答案列表不进入运行时生成，不以辅助头分类分数代签字节任务。

RGB置零与空问题控制各≤0.50；独立恢复、源或数据失配为engineering_failure，超90 CPU／180 wall为budget_stop。完整未达标为content_credit_not_calibrated；达标为first_round_calibrated_with_content_credit。仅签当前单对象／六问题／三几何族、单初始化首问范围。Native、跨轮、自驱、保持与成长资格继续按原合同。

## 4. 资源及后继

本卡90 CPU使用P3.8原1020草案之外的90余量，不增原1110资源。P3.8.1预留120后原未分配990，本卡预留后900，后续Native工程240／runner600／审计60仍为逐卡草案。该对照执行一次，结案后按实际结果决定是否进入Native更新责任检查。
