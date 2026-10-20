# PLAN-N1-160A：完成seed130 fresh face剩余endpoint评分

**状态：已依据用户授予的执行计划批准权批准，执行中**
**日期：2026-10-20**
**上位方案：** [PLAN-N1-160](PLAN-N1-160_seed130_candidate_parent_fresh_replication_20261020.md)

## 触发与失败面封存

N1-160首次执行完成seed11 `unknown8_rebalanced` candidate的480条评分后，在endpoint汇总阶段错误要求其余五个endpoint同时存在，因而报`summary stratum differs`并停止。故障来自新runner再次复制了全量汇总假设；source/face/checkpoint未变，progress中的480条完整有效，0训练/更新/dev/holdout/confirmation。

失败result SHA-256为`b6446a5542e9c4c149f00d36f4a9b0e4ed8b9ea6fe82cec68f76537d6b7d04d1`，attempt SHA-256为`be6001a8322ad110715ebfe1527a18dcc59cf2edc58bf27e5a37a7b03c8baf8c`，progress SHA-256为`984e50b6ee32b060ef84ba60b0292844238e542a671221530b1ee02d8b89dd78`，seed130 face SHA-256为`5b15a47cd7ce0df046bc269fb1bf3a01f4f5242ce3c318c7f953e68547777ee3`。

## 批准的续接

修复runner为按当前endpoint汇总，并增加单endpoint summary回归测试。N1-160A保留candidate/parent端点、seed130 face、题目、指标和判读边界，采用独立readiness、preflight、attempt、progress与result路径。复用原停止报告中的480条逐题结果，不再加载或查询该endpoint；只对剩余五个endpoint各查询480条，共**2,400条新查询**。最终结果合并已存480条和新2,400条，覆盖face全2,880条；seed130不重生成。

不训练、不更新、不重新抽样、不重复任何已消费题目；不访问dev、holdout或confirmation，`can_promote=false`。

## CPU与停止规则

- N1-160A新增阶段CPU≤3,600秒，含新readiness、六bundle独立恢复、旧480行逐项绑定核对和剩余2,400条查询。
- 失败前已用关联CPU`10421.21875`秒计入上位18,000秒限制；N1-160A关联累计不得超过18,000秒。
- N1-160A启动后若再失败，封存回执并停止，不覆盖、不自动重试。
