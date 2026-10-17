# 85 条件准入的四条件候选 v5 首次确认（2026-10-16）

此件只冻结未见评估方法，不授权在候选训练工程完整和三position六训练门成立前查询。83已过九零态工程；84已启动但在Codex客户端更新重启后中断，仅存零步检查点，实际中断前更新数未知，工程不完整。84裁决见[EXP-N1-84](EXP-N1-84_transition_factorial_paired_training_adjudication_20261016.md)。候选新尝试86采用同一冻结训练协议；本计划仍保持 `TRAIN_SHA=None`，86完整且通过全部原前置前禁止绑定或查询。77失败/78拒入保留；既有v5仍未消费，不重新抽样或查看答题结果。

## 唯一准入链

执行器TRAIN_SHA=None，先完成artifact-free测试并提交。84九模型固定终点后，独立receipt必须核验全221184更新、497664QA、387072训练会话、43200行、18独立恢复、72副本步/288合成QA、CPU≤14400、完整源链与九无损分片精确双SHA/计数/五阶段/同view逐项对齐。三position都须通过84六门；控制全部保留并工程完整。检查完整checkpoint参数/Adam/RNG/history与各已封存parent signature逐位相同，不做签名更新。前置不足即拒入，TRAIN_SHA保持None，attempt不存在，v5不消费；不得改门/选seed/补训以救原协议。

只有准入成立，才把84终端原字节SHA明确绑定85并提交后执行一次。沿76原manifest `plans/manifests/n1_fresh_confirmation_v5_20261016.json`，SHA `22ba37976bc6eaaaf2425b84714a6efa6e95e7e5aa05f32eb3a7c0c10d9ffc2d`，32题/四类各8；九模型同32题，共288 teacher+free序列，自然boundary/最多64输出字节。显式holdout路径，不改split绕训练守卫，不给oracle地址/解析器/条件标签。0训练/优化器/副本更新，0dev/v3holdout/v4；独立CPU≤180秒，串行单线程。逐题清临时会话，全bundle前后不变。

## 原14判据原样保留

每position seed≥16/32且每类≥2/8；三seed总分分别领先terminal与cut各≥12/96（平均≥4/32）。四门合取，未达如实失败，不以总分高/接近/联合训练门通过替代。保持/C4/三器官与全项目出口仍独立欠缺，`can_promote=false`，不能替换默认模型。

## 消费与产物

全部准入守卫通过后、首题前独占写attempt，写入即v5已消费，即使中断也不能重新查询/选择重跑。完整保留partial行/查询计数与恢复/源链/代码/CPU状态。报告只在288行/九完整加载/状态中性和守卫齐备时判工程完整。

预声明：`scripts/training/eval_taiji_n1_transition_factorial_confirmation.py`；`tests/taiji_native/test_native_transition_factorial_confirmation_contract.py`；`taiji_n1_transition_factorial_frozen_confirmation_20261016.json（该件从未落盘；2026-10-18 ㊵-1111 就地更正，逐枚处置见那一条）`与`_attempt.json`；结果裁决`plans/reference/EXP-N1-85_transition_factorial_confirmation_adjudication_20261016.md`。84receipt helper为共享准入，不改旧77/78/82/83源码。准备实现不加载真实模型、不查询任何确认题；运行另受以上SHA绑定前置。

## 实施前置记录

84/85共30项artifact-free合同通过，两根Ruff通过；84真实来源只读预检77来源守卫与九完整零态路径通过，0更新/查询。逐步QA调度/有限损失与五组实际位移、五阶段六view精确目标/原始复现/汇总逐项一致、无损双SHA、cut状态与全历史恢复均有合同；85 TRAIN_SHA保持None，未准入/未查询。共享代码提交后冻结并开始84，不能在训练运行中改执行器/目标/原门。完整HEAD CI债仍保留。
