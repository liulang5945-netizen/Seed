# 配平控制与研究父检查点回执

> 22.5.P3.4.1，2026-11-12。passed_parent_preflight_only。0优化器更新、fit/check/final读取0。下一步P3.4.2统一实价及任务分区冻结。

## 1. 正常候选与同一旧父

| 分支 | 参数/张量 | 旧父 |
|---|---|---|
| Native-old | 22,742/194 | P2 Native-conditional |
| Native-query | 23,326/201 | 相同194张量，逐位继承 |
| Native-capacity | 23,326/199 | 相同194张量，逐位继承 |
| Transformer | 22,373/45 | P2 Transformer-token |

Native三分支处于相同参数revision，研究父资格为空。query与capacity新增参数分别用2026111201/2026111202初始化，Adam全部重置为相同lr=0.001、betas=0.9/0.999、eps=1e-8、weight_decay=0。具体研究谱系、旧checkpoint字节摘要及全部源pin写入[父清单](../manifests/taiji_u2_evidence_A_parents_20261112.json)。

CapacityResidual采用16→23→8加一个标量，共391+192+1=584个参数，输入request与H读取。输出残差乘数为0.2·sigmoid(a)，a初值0，初始乘数0.1。它通过同一正常主体处理证据来源与恢复，额外残差仅读取request/H。其5个参数张量全部获得完整回答损失梯度。该控制用于检验新增参数/条件残差映射能否解释收益。

两臂具有相同残差乘数初值，随机初始化的实际激活幅值仍有差异，不能据此宣称完全匹配优化条件。容量控制仅声明width=8的当前small候选范围。

## 2. 保存及独立接续

新episodic learner schema显式绑定候选类型、原生bundle、参数、训练模式、Adam、RNG、数据摘要/顺序/游标及父谱系。逐例仍从公开输入重新构建主体，episodic检查点内主体state为明示空边界。连续主体恢复采用各自typed runtime。类型错配、源错配和覆盖已有父文件均拒绝。

[机器报告](../../reports/taiji_u2_evidence_A_parent_preflight_20261112.json)交付四份具名initial.bin。每个保存后先在同进程读回参数/Adam/元数据，再由独立进程在相同两个公开price例上复算完整字节NLL及逐参数梯度。639条梯度与None模式完全一致，最大差0。四份Adam均为空，未把初始化预检视为已有训练moment的实测。

实际CPU19.625，符合180 CPU/360 wall工程帽；该帽从16950预留，当前未分配16770。价格样本共8例，未读新fit。后续实际训练的首个更新仍须逐步保存并读回真实非空Adam，沿同一候选正常入口验证。

新增7项控制/learner回归及既有证据、请求、基线、物理、恢复/重编码共115项通过。Black、Ruff、mypy及提交门按本轮源码核验。

## 3. 来源与后继

新增控制型工厂源pin并提取可复用旧源校验函数后，P3.3旧回执涉及的两个文件已按[本卡合同](../manifests/taiji_u2_p3_4_1_parent_contract_20261112.json)保留确切源码快照。旧P2 learner沿既有迁移清单核验，其原认知源码保持匹配。新父的各项源身份按本卡当前代码冻结。

下一步唯一为[P3.4.2统一实价与分区](PLAN-TAIJI_U2_EVIDENCE_LEARNING_20261112.md)。四臂按相同信息权利和自由字节评分确定训练CPU及剂量，新fit在模型结果读取前固定。当前签署研究父可恢复，任务能力按实际学习结果裁决。
