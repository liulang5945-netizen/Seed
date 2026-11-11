# U2 22.3.0与22.3.1父合同及挂起信用回执

> 文档状态：[证据] — 先在13ede2810提交22.3–22.8详细计划，再执行本子卡。裁决为passed_pending_conditional_credit_only，22.3总体未通过，下一子卡22.3.2有限窗因果回放。

## 1. 父身份与交付

父源码与22.2原始证据按13466c975保留。本轮逐一核实其合同source pins可在该提交恢复，当前改变另按新合同记录，旧报告不回写。

| 产物 | 作用 |
|---|---|
| [详细推进计划](PLAN-TAIJI-CLOSED_LOOP_DEVELOPMENT_20261110.md) | 分层恢复、十二断点、六次共享前缀更新及两次故障余量，后继任务/策略/七臂成长/收束 |
| [恢复合同](../manifests/taiji_u2_22_3_recovery_contract_20261110.json) | 父身份、当前source pins、16项检查、40状态字段及分层信用范围 |
| [主体v2](../../taiji/u2_subject.py) | 保存逐分支行为时输入，重建旧V，校验版本/shape/权重/缓存 |
| [回归测试](../../tests/taiji_native/test_u2_recovery_credit_22_3.py) | 真实序列化、critic/actor信用、非线性混合、损坏字段、旧goal/未知/终态及Θ负例 |
| [核验器](../../scripts/training/verify_taiji_u2_recovery_credit_22_3.py) | 新鲜主体与内存bundle，0优化器更新，运行条件随报告写盘 |
| [正式报告](../../reports/taiji_u2_recovery_credit_22_3_20261110.json) | 16检查通过，保留未恢复窗内Θ信用的实际负例 |

## 2. 本轮修复的确定断点

旧policy的value先在各分支做非线性critic，再按权重求和。旧bundle detach后只剩该值和混合features，receive继续从缓存值构造损失，critic无法从该结果得到参数信用。

v2保存critic_features[K,2d]、critic_weights[K]及Ψ版本，receive在对应冻结Ψ下重算旧V。输入固定，critic参数保留学习图。缓存value只作一致性检查，不能把requires_grad赋给缓存标量代替负责参数的路径。混合特征先平均再经critic通常不等于原定义，独立的非线性负例也被检出。

v1无pending包可显式迁移值并开始新学习窗。缺必要输入的v1 pending包拒绝条件/完整信用导入。v2补验参数/状态/history版本，pending的逐分支输入、布局、有限性、权重归一和Ψ版本。完整文件/外层执行器/学习窗验证仍归22.3.2及后续子卡。

## 3. 实际读数与边界

| 读数 | 结果 |
|---|---|
| 序列化前critic梯度范数合计 | 0.028651310753027667 |
| 恢复后重建critic梯度范数合计 | 0.028651310753027667 |
| critic梯度max_abs_delta | 0.0 |
| actor资格迹条件信用 | 原/恢复路径逐张量符合冻结门 |
| 缓存value加载后requires_grad | false，修复未依赖保留原计算图 |
| 检查数 | 16，通过含六类输入损坏和明确负对照 |
| 参数/顶层状态 | 191张量、40字段 |

同时实际检出另一条边界。旧预测窗中motion信用范数为0.020637059690908323，仅恢复值后的对应旧预测信用为None。pending.prior和prediction_key_view的原图同样没有保留，完整Θ学习窗尚未恢复。这个负例是后继必修项，不能用本轮critic通过或缓存数值相等覆盖。

当前bundle明确声明body_graph=values_only_requires_window_replay。完整22.3通过需要32逻辑操作以内的因果shadow重放及原/恢复两侧同一窗的Θ/Ψ梯度与实际更新验证。跨窗允许声明的值截断，不能把额外截断与完整窗恢复混为一项。

## 4. 条件、资源与后继

本轮为CPU/float64，intra-op线程1、interop线程12，Python3.12.10、Torch2.13.0+cpu，OMP/MKL/OPENBLAS未设置。正式报告CPU4.96875秒、wall6.609秒，0优化器attempt、0正式训练、0项目数据或旧checkpoint加载。

子卡工程预算120 CPU/240 wall，属于22.3总帽1200/2400及最多8次attempt之内。主恢复方案保持有限窗完整信用，当前部分结果不准入22.4。下一步实现runtime envelope、原始操作/自输出日志、执行器幂等和影子重放，再开展十二检查点及两组真实更新。

42项聚焦pytest通过，覆盖本子卡16项、原参考主体13项、数学3项、计划5项及运行条件5项。默认Ruff、B/SIM、Black和主体模块mypy通过。等值结论只覆盖本报告所记录的单进程序列化、CPU/float64条件，独立进程与完整Θ窗仍是未通过项。
