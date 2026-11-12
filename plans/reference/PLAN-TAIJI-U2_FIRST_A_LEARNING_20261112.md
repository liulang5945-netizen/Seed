# 首轮A有界学习与B后继准入

> owner已裁定学习前置。当前范围是首个开发性学习pilot，正式A父的≥0.80、3seed、独立check门继续由22.5.0验收。

## 1. 原生接口

持续goal与answer_request分别保留原文、版本和归属。请求按文本控制事件进入已有Native状态，使用已有可学习文本编码及H更新。物理预测继续读取持续goal，语言读取当前请求。请求不会自动写成外部事实，也不会强制策略选择express。

新增可选separate_requests profile。启用时保存answer_request与utterance_request共42字段，subject schema为v4。194个参数张量逐项继承；旧profile保留40字段。请求存在性使用ν第4项及对应missing位。新的测量定义会影响状态解释，当前只签参数继承，采用新genesis。

GRU/Transformer回答基线使用完整UTF-8、真实时空坐标/缺失、动作/结果及合法记录元数据。原8步定价模型仍按原身份保留。新模块位于scripts/training/task_baselines_22_4_a.py。

## 2. Pilot范围

首轮为RGB场景中的当前实体位置回答，保留audio/proprio及公开执行结果输入。训练场景1–2对象，开发fit为未见seed的3对象组合。六个颜色名称在训练可见，当前请求采用固定中文模板，回答为完整位置短语或未知并由模型自主产生EOS。

train64、fit32、独立定价8个scene，四类目标各自平衡。生成器预先排除位置分类边界0.1内的目标，并核对公开RGB oracle；该筛选只使用环境可辨性，模型结果不参与。fit属于开发数据，本轮check/final/confirm读取均0。

Native、GRU、Transformer各用固定初始化seed20261110。三者同公开输入、同目标字节、相同Adam和相同数据顺序。训练损失为逐字教师强制NLL，推理只用原生257字节输出与UTF-8/EOS合法掩码，64字节范围。

## 3. 资源与剂量

Pilot总帽1200 CPU/2400 wall，从22.5的19200 CPU中预留，包含准备、定价、保存恢复、训练和评估。正式A/B剩余18000 CPU。22.4既有账及超支继续保留。

采用同训练CPU配平。读fit结果前，各架构在独立定价scene上运行8次前向/反向，权重不更新；记录更新代数及保存开销。原先新增的最低32步缺少有效剂量实证，首次实价冲突后已保留compute_parity_unavailable记录；本pilot以正的可支付更新数至256次更新，batch2 scene，实际次数由冻结价格唯一确定，使计划训练CPU差≤10%。当前v3执行合同冻结Native24、GRU243、Transformer188步，计划CPU32.95/34.17/34.15；行为门、64字节和正式A父门均保留。不可配平则compute_parity_unavailable停止。

Adam lr=1e-3、betas=(0.9,0.999)、eps=1e-8、weight_decay=0；不clip，不按fit调整学习率或增加剂量。所有梯度须有限。三架构训练CPU与评估/全过程成本分别报告，pilot不据此宣称全生命周期架构优越。

## 4. 验收与出口

每个架构先做参数/optimizer/RNG/数据游标checkpoint写入、新实例读回及下一批损失/梯度一致性检查。首个真实更新后再次核对非空Adam状态。失败及重启均记账。

初始与训练末在同一fit32自由生成，全分母包含缺失、无EOS、非法UTF-8和内容错误。Native学习信号门为fit≥0.50且较冻结初始化增益≥0.15，记录RGB和文本负责参数梯度、实际位移及NLL。该门只签本pilot，正式A父0.80门保留。

三架构均发布实际读数和成本。若Native学习而未优于强基线，分开记录学习成立与架构优势未证。若只有NLL降低，结案task_gain_absent。负结果不自动加步、换种子或读取新check。

## 5. 后继

有界学习结案后，按结果确定正式A的3seed/check准入。B使用实际A父验证经验、纠正与保持，预先检查关键证据跨窗后的责任路径。C/D/J承接实际学习父及其错误分布，当前开发样例不代签这些能力。

本轮源码位置为scripts/training/first_a_support_22_5.py及first_a_runner_22_5.py，recipe/价格/执行合同在plans/manifests，机器回执在reports，运行数据由get_external_path解析至data/experiments/taiji_u2/first_a/<run_id>。


## 实际准入

[执行v3](../manifests/taiji_u2_first_A_execution_20261112_v3.json)与[checkpoint预检](../../reports/taiji_u2_first_A_preflight_20261112_v3.json)已冻结。先前实价预检[原始记录](../../reports/taiji_u2_first_A_preflight_20261112.json)因人为最低32步停止，v2在读取fit模型结果前撤销该约束；v3补评价预算guard并重建初始checkpoint，再次核对参数身份和下一批逐参数梯度。原价格复用，原始支持源码按SHA256归档。首轮训练由owner已选的有界学习指令授权，0 check/final读取。
