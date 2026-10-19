# PLAN-N1-157A：完成seed128 fresh face剩余endpoint评分

**状态：已依据用户授予的执行计划批准权批准，执行中**  
**日期：2026-10-20**  
**上位方案：** [PLAN-N1-157](PLAN-N1-157_seed128_candidate_vs_parent_fresh_diagnostic_20261020.md)

## 触发与失败面封存

N1-157首次执行完成seed11 `unknown8_rebalanced` candidate的480条评分后，在汇总该endpoint时复用了N1-155专用的六endpoint汇总器。它把N1-150 `uniform_length_replay`分支当成必需项，因此错误报缺层并停止；故障发生在seed11 candidate逐题输出已保存之后，未查询第二个endpoint。

失败报告SHA-256为`137d822768b6d75405dfa8a521bd6e0754e6632200e8cb674416a20464aa912d`，attempt SHA-256为`59ac9c47bde0835c7dd2a02d22bd42e130e1d8c24a1e886cd094fb898350b492`，progress SHA-256为`7619efac685175e87968e58bdcc362ea9c5bb38ebdb0d8cff0dad6af0980809a`。已消费部分严格为seed11 candidate的8/16/32行×四类各40题，共480条；progress逐行完整，source/checkpoint哈希不变，0训练/更新/dev/holdout/confirmation查询。seed128 face及其审计保留，绝不重生成。

## 批准的单一更正

修正汇总器，使其只汇总当前endpoint。N1-157A复用同一张已封存seed128 face、同一组六endpoint与同一指标；读取并验证已保存的480条作为首endpoint记录，跳过该endpoint的模型加载与查询，只查询另外五个endpoint各480条，共**2,400条新查询**。结案报告合并既有480条与新结果，覆盖同一face全部2,880条；任何已消费题目/endpoint都不重跑。

所有更正轮readiness、preflight、attempt、progress和result使用独立`n1_plan157a`路径；原N1-157 attempt、480条进度、停止报告及seed128 face全部保持原样。readiness绑定旧失败回执、旧progress、face、六个checkpoint及N1-157A代码；训练不启动，原始计划门、比较对象和晋级边界不变。

## CPU与停止规则

- N1-157A新阶段CPU≤3,600秒，包含新readiness、六bundle独立恢复、旧480行核对和剩余2,400条评分。
- 已有`9502.03125`秒关联CPU计入上位18,000秒上限；更正后的关联总CPU不得超过18,000秒。
- candidate/parent两侧仍是描述性fresh synthetic诊断，不是confirmation；不访问dev、holdout或confirmation，`can_promote=false`。
- 更正阶段启动后若再失败，封存回执，不覆盖、不重试；任何后续动作需新故障归因与计划裁定。
