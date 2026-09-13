# M5 K 轴晋级宣布（限定范围；项目所有者独立批准）

> 宣布日期：2026-09-12。批准人：项目所有者（独立批准，[scorecard v8](M5_K_AXIS_SCORECARD_V8_CONTRACT_20260912.md) 为机器边界基准）。本宣布是 [A8 晋级评审合同](M5_K_A8_PROMOTION_REVIEW_20260912.md) §4 结果映射的终局落盘；它宣布**机器可核验的晋级条件全部满足**，并以下列限定范围与诚实边界为生效前提。

## 1. 宣布范围（限定）

**M5 K 轴持续学习机制晋级**：以下证据链所构成的「继承式持续学习机制」在其实验载体上闭合、可验证、已附着真实 runtime——

1. 能力 formal：K1/K2/K3 standalone（scorecard v1/v2 收束）；
2. 学习机制：C 阶段 formal v2（FS 候选）→ 表示因子化（P4.9）→ 投影求解器更新机制（P4.11）；
3. 课程级验证：单任务 9-cell 零方差（P4.12）→ 两阶段累积课程 9/9 + 向后保持零失败（P4.13）→ K worker 联合课程 4/4（P4.14）；
4. runtime 边界：checkpoint/中断恢复/owner 迁移合同（P3.0–P3.2）→ 默认 runtime rollout review（消费合同 `taiji-default-runtime-rollout-attachment-v1`，4/4 cell 消费等价）→ 真实 runtime 附着（`api/taiji_runtime_attachment.py` + `SeedRuntime.attach_k_g_state`，4/4 cell runtime 读出面逐字段复现、rollback 精确、非干扰全过）。

## 2. 诚实边界（冻结，随宣布生效）

1. **五类合成课程仍是唯一实验载体**：本宣布不构成通用语言、完整认知能力、一般任务能力或开放泛化的主张；"载体之上的外推"属未来独立验证；
2. **结构成长未被触发**：`growth_admitted=false` 贯穿全部实验报告；A8 准则的「分化/增长/剪枝」子句在本轴为**未被触发**（P4.7 关闭容量假设、P4.9–P4.11 在固定容量内消解张力），这是如实记录而非"已满足/已跳过"；原生线的结构成长机械（budget/rollback 门）保持可用，待真实缺口出现时按预注册启用；
3. **附着为显式 opt-in、进程内内存态**：默认 chat/workbench 行为路径保持不变；默认路径是否采用该机制、附着持久化，均属后续独立工程与审批决策，不在本宣布内；
4. **S 为架构性 control-only evidence**（P3.1 合同）；learned-S 为非阻塞的未来探索项。

## 3. 不变项与后续纪律

- `growth_admitted=false` 贯穿；不读取 sealed 的纪律不变；本宣布不修改任何已冻结判据、不覆写任何历史报告；
- 后续任何行为切换（默认路径采用、附着持久化、能力面扩张）仍走「预注册 → 冻结 → 独立批准 → 执行 → 机器报告」的既有节奏；
- 唯一下一步：**P5.1 知识来源预注册**——把已验证的持续学习机制接入既有 S 轴内化机械（governed corpus / Skill-MCP artifact 边界 / 语义 embedding 内化），与普通数据同预算比较内化与未见任务收益；工具知识与宿主执行权限分离的边界不变；先摸底现有机械与语料现状，再冻结预注册。

## 4. 批准记录

- 批准人：项目所有者；批准时间：2026-09-12；批准形式：对话中明确「批准」（对 [A8 评审合同](M5_K_A8_PROMOTION_REVIEW_20260912.md) §3 四项裁决全批 + 本宣布范围确认）；
- 机器边界基准：[scorecard v8](M5_K_AXIS_SCORECARD_V8_CONTRACT_20260912.md)（13 项 promotion gate 全 true；`promotion_gate=true`、`can_promote=true` 机械置位）；
- 本宣布落盘后，晋级状态从「等待独立批准」变为「已晋级（限定范围）」；v8 报告与全部历史产物不覆盖。
