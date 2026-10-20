# PLAN-N1-159A：完成seed129 fresh face剩余endpoint评分

**状态：已依据用户授予的执行计划批准权批准，执行中**  
**日期：2026-10-20**  
**上位方案：** [PLAN-N1-159](PLAN-N1-159_seed129_candidate_parent_fresh_replication_20261020.md)

## 触发与失败面封存

N1-159首次执行完成seed11 `unknown8_rebalanced` candidate的480条评分后，在endpoint汇总阶段错误要求其余五个endpoint也必须同时存在，因而报`score stratum mismatch`并停止。故障来自仍沿用全量汇总假设；source/face/checkpoint未变，progress中的480条完整有效，0训练/参数更新/dev/holdout/confirmation。失败result SHA-256为`47ba3fae3f5921ea950f980edcfbb7b015cc477e7dc94265f1871f288a2a6e38`，attempt SHA-256为`2aee1645c68db01f2b2455cfbd758183cfc40abf40b555d6812905f5efb3e9ce`，progress SHA-256为`f9910360aa7f7a87f5328b6cdba5c896f53498321d72b54a0d0903c14f3cae00`，seed129 face SHA-256为`b445e96f7231ba874efa5300f99fd26a92a90548e56d14ba9caf881d30b729d3`。

## 批准的续接

更正汇总器，使其只汇总当前endpoint；新增单endpoint汇总契约测试。N1-159A保留N1-159 candidate/parent比较、seed129 face、指标和查询顺序，使用独立readiness、preflight、attempt、progress及result路径。复用原480条逐题结果，不加载或查询其endpoint；只对剩余五个endpoint各查询480条，共**2,400条新查询**。最终报告合并已有480条与新增2,400条，覆盖该面完整2,880条，seed129 face不重生成。

本续接不训练、不更新参数、不重抽数据、不重复任何已消费题目；不访问dev、holdout或confirmation，`can_promote=false`。

## CPU与停止规则

- N1-159A新增阶段CPU≤3,600秒，含source readiness、六bundle独立恢复、旧480行重验与剩余2,400条评分。
- 已用关联CPU`9956.421875`秒计入上位18,000秒限制；N1-159A结束后的关联累计CPU不得超过18,000秒。
- 本更正只允许一次write-once续接；若再次在启动后失败，封存结果，不覆盖、不自动重试。
