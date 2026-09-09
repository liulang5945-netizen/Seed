# M4.V2 B3-K C-entry capacity-parity audit

状态：已完成只读审计；当前 formal 对照被标记为 capacity/update-budget confounded；未启动新训练，未改变默认 runtime。

## 审计输入

- formal 报告：[taiji_m4v2_b3_k_c_formal_20260910.json](../../reports/taiji_m4v2_b3_k_c_formal_20260910.json)
- fixed-large build 报告：[taiji_m4v2_b3_k_c_fixed_large_build_20260910.json](../../reports/taiji_m4v2_b3_k_c_fixed_large_build_20260910.json)
- 机器可复核审计：[taiji_m4v2_b3_k_c_capacity_parity_audit_20260910.json](../../reports/taiji_m4v2_b3_k_c_capacity_parity_audit_20260910.json)
- 审计脚本：scripts/training/audit_taiji_m4v2_b3_k_c_capacity_parity.py

审计覆盖相同的 3 个 model seed × 3 个 course seed，共 9 个 cell；检查每格的参数字节、实际参数更新步数、checkpoint 写入、训练 episode、推理 trace，以及 formal 与 fixed-large build 报告的一致性。

## 当前事实

| 指标 | candidate | fixed-large | 解释 |
|---|---:|---:|---|
| 参数字节 | 19,332 | 38,664 | fixed-large 为 2 倍容量 |
| 实际训练更新步数 | 6 | 14,252 | 不是同一训练预算 |
| 训练 checkpoint 写入数 | 2 | 9 | 写入策略也不同 |
| 训练 episode indexes | 相同 | 相同 | 数据课程边界一致 |
| validation/sealed inference trace | 6 | 6 | 评分调用数量一致 |

因此，formal 的 candidate-quality Gate、technical Gate 和 resource Gate 仍然有效；但 candidate 对 fixed-large 的 0/9 胜出不能被解释为“Taiji 学习规则在同容量同预算下失败”。它只说明当前较小容量、较少更新的 candidate 没有胜过更大且训练更久的 fixed-large。fixed-large 的结果仍然是当前候选竞争力不足的信号，但不是隔离学习规则因果差异的证据。

审计报告的当前状态为 blocked-current-comparison-confounded，can_promote=false。原 formal 报告和 C-entry closure 不被覆盖。

## 已预注册的 parity contract

1. 保留原 9 格 cell 矩阵、同一 frozen parent、同一 C-entry course digest、同一 target-tensor multiset 和同一 validation/sealed 输入。
2. 采用“上调 candidate 容量”的高上限路线：目标是 candidate 与 fixed-large 的 parameter_bytes 比例不超过 1%，不为了制造公平而缩小 fixed-large。
3. 两条路线每格使用相同的训练 episode indexes、相同的 train/holdout 隔离和相同的实际参数-update budget；当前 fixed-large 的 14,252 只能作为现有观测值，不能直接冒充已经完成的 parity budget。
4. checkpoint parity 以“逻辑训练 checkpoint 发射数”比较；prefit、fresh-restore、rollback 和 audit 复制物必须单独列账，不能把格式或审计副本字节数当作学习规则收益。训练/推理耗时、峰值 RSS 下界、checkpoint 总字节仍全部报告。
5. 两条路线都必须通过 fresh restore、rollback、parent unchanged、K3 unchanged、candidate update distinct 和资源完整性 Gate；sealed 只在正式评分时读取。
6. 学习规则比较只有在 parity contract 全部通过后才有效；promotion 仍要求 candidate 在 9/9 sealed cell 胜过 strong control，并通过既有 technical/resource/quality Gate。

## 下一阶段边界

当前唯一下一步是实现“上调 candidate 容量 + 同预算”的 parity preflight，先生成并审计 paired artifact contract，不立即训练。preflight 未通过前禁止追加训练、调学习率、复活 R5/旧结构增长路线，或接入 default runtime/provider/MCP/client/CUDA。

