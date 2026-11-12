# 22.4.A.1 原生目标/请求分离与完整输入基线工程回执

> 文档状态为工程回执。2026-11-12。机读裁决归[工程报告](../../reports/taiji_u2_22_4_A_1_native_request_baseline_20261112.json)。本回执只签工程范围，不签任何学习收益或能力资格。

## 1. 裁决与范围

本卡工程范围按`passed_engineering_with_recorded_limits`交付：原生持续goal/当前request分离、复用Native文本控制事件与可学习读取、普通基线全字节/真实坐标/缺失/动作结果/连续状态适配。硬门「角色混淆或信用/恢复失败则不准入」的对应检查全部通过；**身份重冻与实价尚未完成**，准入22.4.A.2前不得引用本回执作为基线身份或价格依据。

0训练、0 `optimizer.step`、0项目数据读取。能力资格为空。

## 2. 原生请求接口

- `U2Config.separate_requests`开启后，`request_candidate()`自元数据候选分叉：**194个参数张量逐字继承**（22,742标量），状态登记40→**42**（新增`answer_request`/`utterance_request`）。
- 请求为历史原子操作`request`/`cancel_request`：未决决策或活跃 utterance 时拒绝；同id版本不得回退；goal只允许在utterance边界变更且`goal_version`严格递增。
- 读取条件分离：`language_goal`在请求活跃时用请求原始字节，持续`goal_raw`不变；`nu[4]`为请求活跃测量位、`nu[12]`恒0，导出时校验测量一致。
- 信用：回答NLL梯度在float32/64下均到达请求编码、`text_input`与native嵌入（非零且有限）。
- 恢复：半截utterance保存/加载后逐值一致，请求身份绑定utterance，完整历史重建一致；v4 bundle拒绝v3 schema冒充。
- 同内容不同id的请求产生**逐位相同**的H与输出（id仅协议身份）；不同内容产生不同读取——角色混淆检查通过。

## 3. 全字节公共流基线

`TaskSequenceBaseline`（GRU h=40／单层因果Transformer d_model=28）**逐字复用native可训练前端**（encoder/metadata/modality/embedding/text_input/convolutions/goal/outcome/action/记录元数据与来源），交付[静态审计](../../reports/taiji_u2_baseline_task_entry_audit_20261112.json)要求的五项适配：

| 审计要求 | 交付 |
|---|---|
| 全UTF-8 goal/request摄入 | 完整问题文本＋角色tag，无8字节截取 |
| 真实时空坐标与缺失 | 实际coordinates逐模态传入，缺失模态不产token |
| 实际动作与结果 | action/outcome事件进入同一公共流 |
| 连续状态与恢复 | GRU隐状态/Transformer键值连续；`parameter_revision`变更强制重编码；save/load带源身份钉 |
| 同公共records与元数据 | `record_units`同入口、同valid掩码 |

当前/历史请求前缀碰撞已消除：两请求读取不同、token数>8。参数量GRU **23,509/34**（+3.37%）、Transformer **22,373/45**（−1.62%），均在22.4.4 ±10%配平带内；**这是新源身份，22.4.4冻结的21,977/35旧8步身份不得复用**，身份与实价重冻归22.4.A.2。序列预算显式`sequence_budget_stop`，无静默截断。

## 4. 检查与成本

- 新增测试**19项全过**（请求9＋基线10，含float32/64、恢复、信用、角色、预算负例），CPU 4.71875/wall 6.71038。
- u2全套回归**218项全过**（wall 60.93），u1冻结产物与22.4.1执行图读数不受影响。
- CI棘轮恢复：HEAD存在5条mypy红（22.4.2提交引入）、WIP新增8条，共13条**全部以纯标注/等价改写修复**（`restore`返回类型、`bodies/answers/sessions`标注、`max`键lambda、前端类型声明），`mypy --follow-imports=silent seed taiji` **0错/248文件**；ruff（含B/SIM）与black全绿。

## 5. 源覆盖与已知限制

本轮改动使6份22.4.2冻结合同的源pin失效。按既有overlay纪律新增**18份精确字节存档＋34条覆盖登记**（清单共36条），并顺带闭合了HEAD即已存在的可恢复旧pin（events/executor/runtime测试等）。逐条sha核验0失配。

**不可恢复3条**：被v2修正案取代的world_implementation **v1失败回执**所钉`u2_task_world.py abac4595…`、verify脚本`570c1e2f…`、测试`7993f0ba…`从未入库，无法重建，按原样登记为已知限制，不伪造字节；v1失败证据身份保留，现行实现以v2合同为准。

其余限制：请求接口只在合成夹具上验证；基线身份与实价未冻；22.4.1的u2强基线臂仍绑定其自身冻结执行图，本卡不改。

## 6. 后继

队首进入**22.4.A.2**：冻结单帧/多实体A与时间B的范围、train/fit与未见轴、新源强基线身份、同资源配平、数值门与学习步数/停止；随后22.4.A.3 checkpoint预检与预算ticket，22.5.P1执行首轮有界学习。正式学习仍由具体执行合同准入。