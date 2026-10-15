# SPEC-N1-22 · 已批准21方案执行合同（2026-10-15）

用户明确“批准21”，同时要求以后决策使用选择弹窗；同类低成本排程继续自主，不反复请求。需要决策时展示选项影响和唯一推荐，并继续不依赖决定的工作；真正需授权的步骤等实际答复，不把超时/自动续跑当批准。

沿[21方案](PLAN-N1-21_observation_fidelity_learning_review_20261015.md)实施，Taiji内独立loss模块，无新增参数、无推理改变、无copy分布或运行时teacher。原始writer位置状态h_i与V(h_i)经过原R，重构当前真实外部观察字节，L=L_QA+0.25*meanCE_observation。terminal辅助用同一全观察位置；cut在QA和辅助两支均detach writer状态，前向相等、早期信用不可达。

先做独立工程预检（CPU≤60秒）：全部四类首train项×三臂共12次前向/反向，无optimizer更新；逐项验证position/cut总loss相等、早期梯度有/无、terminal辅助相等、参数不变。独立zero checkpoint恢复验证live-parent参数/Adam/RNG/来源/首步logits/辅助logits一致；初始权重与19冻结zero相同。该预检不作为训练或能力证据。

预检通过后单独执行一次train-only校准：seed11新初始化，同v3 SHA ac56fb0aba2ef602779f58749e6f8c35324c40c433fd16ce55b89689235b0604，192train固定循环、Adam0.001、16/32/16、6144更新。节点0/768/1536/3072/6144各全部192自由/TF及观察保真，共960组，0 dev/holdout。独立CPU≤300秒，单线程并计两个恢复worker；训练前zero恢复，终点final恢复，协议明确区分旧QA-only，不覆盖旧运行器/档/报告。触边保留partial、不缩分母/自动重跑。

终点≥173/192且每类≥40/48才满足训练充分性；和19同初始化/信息/更新作描述对比，单seed不是机制优越性证明。保真字节准确高或loss低不能替代QA行为门。若充分性未过，按21预定分支复审内容寻址；若通过，再单独冻结新配对/未见/保持出口。C1/C2/C3/C4及同bundle三器官原债务保留，默认采用/远端推送/数据删除不在本批。
