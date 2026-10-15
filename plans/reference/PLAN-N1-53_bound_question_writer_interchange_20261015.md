# PLAN-N1-53 · 冻结 writer / 绑定后问题状态交叉诊断

## 固定来源与问题

仅使用52冻结终点（SHA256 `81bff9166fe684db1ade5eaec2e136a17a1fac51e574572e61e4fb72c073ef81`）和52完整报告（SHA256 `cb26c88e8dab98819a3dbcd48348f7737230872f5444039d460af77e7d00809f`）。区分改名失败来自观察writer还是完成局部绑定与上下文GRU的问题状态，不用oracle地址。

## 冻结执行合同

沿44 checked_views的192原题及96配对题、对应已知改名，共288组。各组五条件：原始参考、改名原生、改名换原始writer、改名换原始绑定后问题状态、两者均换；共1440条gold-prefix teacher序列。独立CPU≤60秒，58秒预留收尾；0更新、0自由生成、0dev/holdout。目标与位置严格相同。

writer供体仅替换相同位置的raw states/keys/values，保留接收者session/source和改名原始字节。问题供体在正常bind_question出口捕获，包含最终hidden及全部question_states，保留零初始address且不携带conditioned tape；接收者在同一出口替换，不能把原始问题字节送入改名局部匹配。

576条原生控制逐项复现52终点teacher symbols/correct count，CE容差1e-6；288条双供体logits必须逐位等于原始参考。按条件、题型、原题/配对及旧free正确组合记录teacher整题、CE、可见字节/边界、第二hop地址JS/TV。旧free字段仅离线分组，不新生成。

模型、Adam、RNG、历史session、源报告/checkpoint、manifest、代码均须前后指纹不变；仅清理自身内存临时session。实际运行前验证供体中性/双供体精确、来源位置和目标对齐、train-only及1440完整分母并提交工具。失败保留，不重采样、不加训。完成后据证据冻结唯一机制修订。

全项目旧能力保持/C4/三器官与CI债不因本诊断通过而完成；不推送或删除证据。
