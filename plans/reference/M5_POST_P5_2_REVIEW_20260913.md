# M5 / P5.2 之后的结果复审与推进依据

> 日期：2026-09-13。源码/报告基线：`102b81e1`。本次只读核对既有证据并重组计划，没有训练、重算实验或改写冻结报告。
> 本文提供分析依据；唯一执行顺序见[当前推进方案](../active/roadmap/03_CURRENT_EXECUTION.md)。

## 1. 核验方法与证据层次

本次对照最近提交、预注册、落盘 JSON、实际执行函数和 CI workflow。四类状态分别记账：已实现接口、已测机制、准入后的状态、经批准的产品采用。某一层通过不自动证明下一层。

近期成果足以把研究从“有没有接口和可学习信号”推进到“预测是否控制真实执行、执行经验是否改善协作”。新方案保留既有成绩，同时补齐证据尚未触达的环节。

## 2. 晋级已经完成，权限范围需要精确继承

[scorecard v8](../../reports/taiji_m5_k_axis_scorecard_v8_20260912.json)的 `verdict.promotion_gate` 与 `can_promote` 均为 true；[2026-09-12 晋级宣布](M5_K_PROMOTION_DECLARATION_20260912.md)记录项目所有者独立批准。因此旧“等待 K worker 联合课程/等待附着”的当前状态已经失效。

已验链包括 P4.14 联合课程、rollout review、opt-in runtime 附着与 rollback。宣布的四项限制仍生效：五类合成载体、结构成长未触发、附着是进程内显式选择且默认行为不变、S 为 control-only evidence。

**规划影响**：可以继承已验证 K/G 状态和消费合同；默认路径采用、持久化以及其他研究轴的晋级仍须独立验收。P5 新报告的 can_promote=false 与 K 轴限定晋级并不矛盾。

## 3. P5.1 知识来源证据：收益成立的测量范围

### 3.1 演进关系

P5.1 证明受控内容迁移；P5.1b 暴露语义改写辨别不足；P5.1c 用成对对比建立辨别力；P5.1d 接入锚定 encoder；P5.1e 在构造语料上验证同预算收益。这些实验的问题、样本和指标不同，不能把数值连成单一通用能力增长曲线。

P5.1f 转向真实发布语料后按冻结判据失败。其报告和归因应保留：轨迹数相等未约束实际 calls/examples，准入失败令核心读数不可测，恒 reward 使部分回归指标饱和，wall 超限。损失很低仅能说明对应目标可拟合，不能单凭它证明所有任务相关信息“投影无损”。

### 3.2 P5.1g 的直接事实

来源：[预注册](M5_P5_1G_REAL_CORPUS_QUOTA_BUDGET_PREREGISTRATION_20260912.md)、[报告](../../reports/taiji_p5_1g_real_corpus_quota_budget_20260912.json)。

| 项 | 已记录事实 | 边界 |
|---|---|---|
| 预算 | train/holdout/retention calls 配额为 989/385/253 | 轨迹数不等，末条前缀截断；calls 相等不是总计算量相等 |
| 迁移读数 | sourced trial a-gate 0.651376，placebo 0，delta 0.651376 | placebo 的词表与目标不相交，零分有结构性原因 |
| 频率对照 | 超 per-tick 基线 0.284376 | 提供额外辨别，但不替代共享词表/任务结构的内容对照 |
| 准入 | 两臂 admitted=false、rolled_back=true | trial 成绩未进入产品有效 child 状态 |
| 校准 | §5 已观察 a-gate 并比较 hidden/epochs | 该读数含模型选择影响；新泛化结论应使用独立未读测试 |
| 成本 | wall 435.931s，在新 cap 内；checkpoint/replica 门通过 | 不能用新版本通过覆盖 P5.1f 超时 |

**规划影响**：保留 `real_corpus_content_benefit_supported` 原判定；把“真实语料产品准入”单列成工作包。已测 retention 平台支持进一步归因，不足以证明所有结构与训练方案都无法达到准入。

## 4. P5.2：合同执行与模型动作预测是两份证据

来源：[预注册](M5_P5_2_WORKBENCH_SIMULATION_CONTRACT_PREREGISTRATION_20260912.md)、[报告](../../reports/taiji_p5_2_workbench_simulation_contract_20260912.json)、[runner](../../scripts/training/eval_taiji_p5_2_workbench_simulation_contract_gate.py)。

### 4.1 已验证成果

九门均为 true，outcome 为 `workbench_simulation_contract_supported`。报告记录 180 个执行动作、语言识别/切换、30 场景撤销恢复、readout train/holdout/a-gate 为 1.0、lesion holdout 为 0.153846、a-gate 相对频率基线 margin 为 0.384615。

这提供了有价值的真实合同测试载体、可学习动作词表、脚本轨迹和审计结构。模拟审批者是确定性策略，不能外推为任意真实用户授权场景。场景限定于当前四类 scripted IDE 模板。

### 4.2 源码与叙述之间的接线缺口

`_execute_scene(environment, scene, root)` 在 `for tick, step in enumerate(scene.steps, start=1)` 中构造 `ActionIntent(step.kind, parameters=_resolve_params(step.params, state))`。调用方 `execute_all` 也只传入 environment、scene、root，没有 learner。

`_train_readout` 返回程序 learner，`_accuracy` 调用序列评分计算预测准确率。其预测尚未写入上述执行循环。因此既有证据不能支持“模型预测已经控制了那 180 个动作”的更强表述。

此外参数由脚本生成；即使动作类型准确率为 1.0，也没有直接证明路径、patch、语言目标及撤销 token 都由模型正确生成。

**规划影响**：下一步先补预测到 ActionIntent 的接线，逐字段披露参数绑定责任，主指标切换为冻结任务的实际完成情况。预测不正确时不得用脚本答案纠正后计为成功。

## 5. P5.2 trace 对 transfer learner 的适用性

[报告](../../reports/taiji_p5_2_workbench_simulation_contract_20260912.json)的 `interaction_trace` 为 train 40、holdout 12、固定两个 owner，`evaluator_consumed=true`，但 `evaluation_summary.groups=0`、`rejected=0`。

runner 的 `_trace_episodes` 每步写 generator intent event 与 environment outcome event；所有 episode 使用同一 context 字符串，成员组合固定。当前检查确保该组合字段存在且 evaluator 可调用。

[build_member_evidence](../../taiji/interaction_group_transfer.py)只为存在同 context inactive baseline 和 singleton 对照的成员构建 profile；`observe_records` 拒绝未知成员与 holdout-derived evidence；`candidate` 对未知成员或已观察组合默认拒绝。

因此两 owner 投影并不满足完整的迁移训练前提。直接接 learner 可能得到空 profile/无候选，而不是新的学习实验。

**规划影响**：先建立真实干预矩阵和非空 train-only 群体记录，保持环境主体作为宿主，独立核验认知成员贡献。没有 baseline/singleton 时把群体收益记为不可估计；不从 schema 成功推断协作收益。

## 6. CI：历史 Gate 与实际 workflow 的差异

本次通过 GitHub CLI 查询：[run 34579613959](https://github.com/liulang5945-netizen/Seed/actions/runs/34579613959)，head 为 `c7bbd389`，两条 Linux test 在 `Lint with ruff` 失败，Windows cancelled；frontend、Docker、startup smoke 成功。没有将该旧提交结果归因为当前新增代码的失败。

[当前 workflow](../../.github/workflows/ci.yml)的 core mypy threshold 为 0。P5.1g/P5.2 预注册中“61 错持平”“4 failed/1199 passed/6 skipped 零新增”并不是该 CI 门槛。报告的静态项主要记录 scope/命令和 executed_before_run，不能作为完整日志替代。

**规划影响**：建立命令级失败清单并按新回归、研究阻塞、发布债务分类。正式训练先过相关检查，发布/默认采用必须满足实际 workflow；不根据历史计数提高正式门槛。本次不替历史运行补发全仓通过结论。

现有文档测试还硬编码 P4.7 的“唯一执行项”标题，本次随当前执行入口更新断言，保留单入口和所有 active 链接解析约束；这不修改研究判据。

## 7. 方向取舍

主线选 P5.2 深化，因为它直接回答 CR-1 认知所有权、CR-2/3 协作和 CR-5/9 真实结果学习。先做插件/provider 可以改善工程体验，但不会补齐上述因果缺口；立即结构扩容缺少当前公平容量归因。

建议依赖为：预测驱动执行 → 群体因果语料 → 未见组合迁移 → 在线结果回写。真实语料 admission 为采用对应 child 的硬依赖。接口稳定后再接 runtime 行为评审和插件/provider；结构成长在出现实际瓶颈后重开。

此取舍是本轮方案建议，不是已获独立批准的新实验。具体臂、指标阈值、资源 cap 与最终数据由下一份预注册在运行前冻结。

## 8. 本轮产物与保留原则

- 活动执行方案改为状态/依赖/工作包/验收/停止点结构。
- plans 首页改为最新证据导航，避免停在已完成的 P5.1c。
- 原执行流水与原索引保存在 [20260913 归档](../archive/history/20260913_plan_reorganization/EXECUTION_BEFORE_REVIEW.md)，仅相对链接按目录机械重定位。
- 未覆盖任何历史 JSON、checkpoint 或预注册；未删除未知临时目录、模型或语料。
- 本轮核验：五份修订/归档 Markdown 的 205 个本地链接全部可解析；`python -m pytest tests/seed/test_project_identity.py -q` 为 5 passed；`git diff --check` 通过。未运行全仓测试，也未将历史远端 CI 标记为通过。
