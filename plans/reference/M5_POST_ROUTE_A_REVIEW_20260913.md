# 路线 A 后的证据复审与路线 B 入场约束

> 2026-09-13；当前基线 f9825943。只读报告和源码，不训练、不改冻结产物。执行顺序见[当前方案](../active/roadmap/03_CURRENT_EXECUTION.md)。

## 1. 新成果确认

- [A 结果](../../reports/M5_P5_2C_TRIPLE_PRIME_REPRESENTATION_REPAIR_RESULT_20260913.md)与[JSON](../../reports/taiji_p5_2c_triple_prime_representation_repair_20260913.json)：rank=2、width=6、prediction_distinctness=2，新 surface 列非零；九门前八通过，第九失败，outcome=transfer_signal_constant。两个未见对的 pair-relative 增益分别为 −0.5、0。
- C、A 均已完成；下一阶段不再重复 A。A 未解决位置身份与条件化选择，也未证明协作收益。
- [P5.2b 当前 JSON](../../reports/taiji_p5_2b_group_causal_corpora_20260913.json)为 groups=0、rejected=6，全部拒绝原因 low_confidence。九门通过代表当前语料合同，不等于旧伪成功 admitted pair 恢复有效。
- f9825943 已将 DocumentEmbedder 移至 instruments，语义 encoder 注入 fail closed；目标集 64 passed 是该提交记录，非本轮重跑成绩。

## 2. 新发现：评分使用不同参照

来源：[A runner](../../scripts/training/eval_taiji_p5_2c_triple_prime_representation_repair_gate.py)的 `_score_pair` 与 evaluate 中 C2 构造；前身 [c′ runner](../../scripts/training/eval_taiji_p5_2c_prime_unseen_combination_transfer_gate.py)也有相同混用。

| 项 | 实际公式 | 当前值 |
|---|---|---|
| object | mean(P_ij − max(S_i,S_j)) | −0.5 |
| C2 | mean(max_all S − B) | 1.5 |
| margin | object > C2 + 0.15 | required=1.65 |

两个差值参照不同：一个扣 pair 内最强单体，一个扣空白。它们都叫 mean_gain_vs_strongest_single，但不能解释成公平同参照的策略差值。

另核对[冻结预注册](M5_P5_2C_TRIPLE_PRIME_REPRESENTATION_REPAIR_PREREGISTRATION_20260913.md) §5：collaboration_holds 要求至少一个留出 pair 的 mean_gain_vs_strongest_single>0；实际代码要求所选对象 mean_realized_interaction>0。两者不是等价命题。本次两者都不满足，故不据此追认通过，但“完全按同一判据执行、只剩表征原因”的叙述需要收窄。

处理：保留原判定，追加本复审；新版本以手算测试统一目标、参照和字段，修订论证与阈值重新预注册。此处未修改 runner。

## 3. 可达性：当前成功矩阵不足以证明同任务协作

直接读取 A JSON success_matrix：固定单体最多 6/24，b+c、b+d 为 12/24。按每个 block 对所有单体取最大成功数，再逐 pair 相减，最大超额成功数为 0。

因此跨任务覆盖比一个固定专家广，与同任务超过所有单体是两件事。前者有证据，后者在当前矩阵中不存在。对全体单体 oracle，前三个 holdout context 单体已经得到 +1 奖励上限；最后一个全失败。保持当前矩阵时，重新排序不能创造正协作增量。

即便假设组合能解决第四类，mean(P − max_all S) 的理论上界也只有 0.5（前三类增量最多 0、第四类最多 2，四类平均）；不能用当前 1.65 当这个同参照收益的可达门槛。该计算仅是上界诊断，未授权修改旧门槛。

建议路线 B 分开验证路由/排序收益与复合依赖任务的协作收益。任务需先验证非平凡且可满足；block-3 缺证据时正确拒绝不算违规或模型缺陷。新的任务定义在正式实施前审阅。

## 4. 预测两档不代表避开冗余

JSON：a+d=−0.024590，其余五对=−0.116803。[select](../../taiji/interaction_group_transfer.py)按 predicted_interaction 越大越优排序。因此“a+d 单独分出来”不能证明“已学会避开它”，当前数值方向恰相反。正式选择只在两个未见对中进行，它们同分，也未检验可选冗余对的排除行为。

新计划只继承“表征机械生效”。下一轮要求独立排序验证、任务效用一致与真实执行收益；非零系数、秩、校准符号不能替代这些门。

## 5. 数据与归因纪律

既有 144/88 载体经过多轮结果分析，保留为开发回归；新的最终泛化测试应独立冻结。task_index % 4 的模板先验必须显式披露，不能将 holdout 的能力面送入模型。

路线 B 推荐条件化、位置敏感表示，但本轮证据不支持“只改表示必定成功”。测量口径、任务结构、成员协调与动作生成均需分层归因。保留 C/A 基线，避免同时改变所有环节而无法解释收益。

## 6. 工程与历史维护

本轮查询 [CI 34753643532](https://github.com/liulang5945-netizen/Seed/actions/runs/34753643532)对应 f9825943；运行尚未结束，两条 Linux 已在 Ruff 失败，Windows 运行中，frontend/Docker/两种 startup smoke 成功。这取代“最新只有 c7bbd389”的旧入口描述；不声明全仓绿。

Git 修复按[既有报告](../../reports/M5_P5_2C_TRIPLE_PRIME_REPRESENTATION_REPAIR_RESULT_20260913.md)保留历史，不清理备份、不 gc/prune。当前 main 与本地 origin/main 均为 f9825943；不凭旧提交链推断成果丢失。

技术债引用图只能排除部分直接依赖，不能证明不存在进程状态/文件/环境的间接影响。28 个 SystemExit 的历史计数不是当前结论；后续用可复现顺序和父提交对照定位，不直接调整测试白名单。

## 7. 本轮产物

重组当前详细方案，更新 plans 入口与 C/A/B 决策状态，追加技术债状态澄清，归档整理前执行方案。原始 JSON、预注册和结果报告保持不变，用户未提交的 .workbuddy 文件保持不动。

本轮验证：六份变更/归档文档的 84 个本地链接可解析；`python -m pytest tests/seed/test_project_identity.py -q` 为 5 passed；`git diff --check` 通过。未执行全仓测试、训练或完整 gate evaluate。
