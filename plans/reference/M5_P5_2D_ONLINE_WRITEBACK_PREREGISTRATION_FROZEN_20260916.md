# M5 P5.2d — 在线 Outcome 回写验收 Gate 预注册（冻结版）

日期：2026-09-16。状态：**已冻结**（本文提交即冻结）。上游：B 链闭合（B2-v4 `collaboration_supported`，[报告](../../reports/taiji_b0_b2v4_collaboration_20260916.json)）满足 [Route B 预注册](M5_B0_ROUTE_B_PREREGISTRATION_FROZEN_20260915.md) §「收益保持不过 ⇒ 不得进入 P5.2d」的前置。授权依据：用户启动 P5.2d 预注册设计（2026-09-16）+「遇到决策点默认最高上限方案」。

## §0 recon 结论

1. **库级机制已存在**：[interaction_group_online.py](../../taiji/interaction_group_online.py) 提供 `InteractionGroupOnlineLearner`（`apply_feedback` 唯一变更入口：stale-parent 校验、duplicate 拒绝、lineage/terminal-Outcome/resource/uncertainty/candidate-id 拒绝规则；`rollback_to` 带墓碑审计——已回滚 trial 不可静默重放）+ 内容寻址 `InteractionGroupOutcomeFeedback`（`parent_checkpoint_digest` 绑定执行前选定、`source_split` 恒为 `online`——测试结果无法混入训练谱系）+ checkpoint 格式 `taiji-interaction-group-online-v1`。
2. **本工作包 = 验收门，不是新建**：roadmap §8 协议——只消费合格 train outcome、测试仅评分；多轮比较冻结父代、等预算更新和 child；验新旧任务、stale parent、重复 feedback/token、中断恢复；learner rollback 与环境事务撤销分别验证。**拒绝规则零改动**（如需改动须单独预注册，本包不做）。

## §1 假设

H-d：在真实执行反馈的在线循环下，`InteractionGroupOnlineLearner` 能在**不回归旧任务预测**的前提下改进新任务选择（更新臂 ≥ 冻结父代），与等预算离线 child 选择一致，且 stale/duplicate/中断恢复/回滚/环境事务撤销五类边界全部按库契约成立。

## §2 冻结设计

1. **被试对象**：库 `InteractionGroupTransferLearner`（`minimum_utility=-10.0`、`maximum_uncertainty=2.0`，与 B2-v1 对照臂一致）拟合 B1 遗留拟合语料（8 结构格，与 B2 链同一语料），外包 `InteractionGroupOnlineLearner` **默认拒绝阈值**（`minimum_interaction=0.0`、`maximum_feedback_uncertainty=1.0`、`maximum_resource_cost=10.0`，零改动）。
2. **在线轮次**：R=6 轮，每轮在 create__override 的新 context（steps 7-12，索引与候选面 steps 0-5、族行 step 6 不相交）上真实执行**在线选定 pair**（`select` 的 unseen_only + blocked 语义）及该 context 的两单体与空白（供因果估计），以**成功率口径** `realized_interaction = P − S_i − S_j + B`（与遗留语料 records 同口径）构造 `InteractionGroupOutcomeFeedback`（native terminal Outcome、真实 resource_cost、parent digest 执行前快照）并 `apply_feedback`。
3. **三臂对比**（R 轮后）：冻结父代（零更新）/ 在线更新臂 / **等预算 child**（同遗留语料 + 同 6 条 feedback records 一次性离线训练）。评估于新任务（create__observation、create__mismatch 的 steps 7-12 共 12 新 context，真实执行更新臂与父代各自选择）与旧任务（遗留预测 digest 逐项比对 + 遗留 held-out 执行抽查）。
4. **验收类（全部须过）**：
   - **新任务收益**：更新臂在新 context 的真实执行增益 ≥ 父代选择增益，且更新臂对 (a,c) 的排序在反馈后升至首位（库口径预测）；
   - **旧任务保持**：更新臂对全部遗留 (pair, cell) 查询的预测与父代逐项一致（append-only 语义）；遗留 held-out 执行不回归；
   - **幂等**：同一 `feedback_id` 重复 `apply_feedback` 被拒（duplicate）且 learner checkpoint digest 不变；
   - **stale parent**：绑定过期 parent digest 的 feedback 被拒（stale），审计 admission 落盘；
   - **中断恢复**：第 3 轮后 checkpoint → fresh 进程恢复 → 续跑 4-6 轮，最终 learner digest 与不间断运行**逐字节一致**；
   - **回滚与撤销分离**：`rollback_to` 后 learner 预测回到父代状态（digest 一致）、该 feedback_id 进墓碑不可重放；**环境事务撤销**独立验证——trial 的 workbench 副作用经事务 undo 消除（undo token 与 learner 状态无关）。
5. **测试仅评分**：以 holdout 派生 feedback 尝试回写须被 lineage 校验拒绝（`source_split` 契约演示），测试结果零进入训练谱系。
6. **资源**：逐轮 resource_cost 记账；episode 预算 ≈ 800-900；wall cap **90s**（新仪器含三臂 + fresh 进程恢复，较 B2 门 60s 放宽并在此声明理由）。
7. **门禁**：G1 干预真实性（全部反馈来自真实执行）、G2 数据隔离（trial/eval contexts 与候选面、族行、遗留语料索引不相交）、G6 预算、checkpoint 恢复 + 篡改拒绝（库格式）机械继承；三态 `online_writeback_supported` / `partial` / `failed`（任一验收类不过 ⇒ 至少 partial，机制类边界失败 ⇒ failed）。

## §3 预期与判读

- 成功路径：全部验收类过 ⇒ `online_writeback_supported`——P5.2d 出口达成（新旧任务收益与资源通过、更新可撤回），在线闭环进入 P5.1h 前置就绪。
- 失败路径 A（学习类不过：新任务收益或保持失败）：库 learner 的在线可学习性被否证 ⇒ 归因到记录口径/轮次预算/选择语义，回库级修复预注册；**不得顺带开放结构成长**。
- 失败路径 B（机制类不过：stale/duplicate/中断/回滚/撤销任一失败）：库契约缺陷 ⇒ 技术债登记 + 库修复单，不改验收判据。

## §4 纪律

负结果不改绿；报告不覆写；拒绝规则/阈值/契约零改动；测试评分与训练谱系分离贯穿；临时脚本用毕即删；冻结后不改 §2。

## §5 产物顺序

1. `scripts/training/eval_taiji_p5_2d_online_writeback_gate.py`（scoped 静态检查先行）。
2. 执行落盘 `reports/taiji_p5_2d_online_writeback_20260916.json` + 验收类对账。
3. roadmap 状态表更新 + 独立提交。
