# 配对学习实际CPU准入

> P3.5.4，2026-11-13。已冻结执行合同，后继P3.5.5。新任务模型精度在准入时读取0次。

[准备报告](../../reports/taiji_u2_paired_learning_admission_20261113.json)采用迁移后的Native-query及Transformer研究父，参数、真实moment和旧计数完整继承。新任务update/cursor从0开始，Adam历史计数保留。非空optimizer逐值读回、同一配对episode下一损失及全部参数梯度精确一致。

每组8个训练episode在真实learner上前反向，Adam仅作用于独立克隆参数/梯度/moment。Native完整操作中位0.296875 CPU秒，TF0.125；两组共16次影子Adam，真实learner更新0，前后参数与optimizer摘要一致。新票据以真正的实际CPU而非固定更新数确定终点，价格只用于资源与空间估算。

[执行合同](../manifests/taiji_u2_paired_learning_execution_20261113.json)为每个可学习组600训练CPU秒、最多8192更新、单步最多2秒。最后一个未越600的accepted checkpoint用于终点评测；多计算的一步保存为overshoot并隔离，其CPU与真实更新计数完整记录，随后恢复上一accepted参数、Adam、RNG和游标。两个选定终点CPU差容差5%。

Native同父冻结组在相同360轮训练/开发范围给出基准。可学习组终点分别报告训练、dev、无RGB、无问题完整360轮及旧P3.4 dev32。主门为Native训练≥0.90、dev≥0.80、相对同父冻结及两种部分输入控制各增益≥0.15。正式多实体A及3seed/check资格继续按原合同。

每个更新前后查源、单写者锁、opened/accepted/quarantined账本、参数/moment/RNG/游标全量保存读回接入运行器。新增3个运行器/影子所有权回归加配对9项，共12项通过。实际准备CPU9.984375符合120帽，运行器总帽2400 CPU/4800 wall；原生图180票据之外再预留120+2400后，项目未分配10290 CPU。准备阶段所有训练操作均为价格探针，尚无学习更新。

下一步按本合同执行P3.5.5，并完整收束任务、恢复、保留与实际资源读数。原始预算停止或工程失败均具名结案。
