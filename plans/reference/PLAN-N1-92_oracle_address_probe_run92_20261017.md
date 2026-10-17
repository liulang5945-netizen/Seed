# PLAN-N1-92 · oracle 地址探针：在已知面那 32 题上问"候选有没有被送进答案头"· 2026-10-17

owner 于 2026-10-17 在 EXP-N1-91 §6 的三条候选里裁定 **乙′**：把 oracle 注入探针做在**失败实际发生**的那 32 题上，并以"已知面上的机制探针"身份出版。本件是那条裁定的预注册，先冻后跑，跑在任何一次模型装载之前。

## 0. 本件不许声称什么

①`oracle_is_capability=false`（沿用 2026-10-15 那份 oracle 件已经钉过的语义）：把注意力钉到正确磁带位置之后答题变好，**不**证明模型会答题，只证明"答案符号的可得性依赖于地址供给"；②本件**不是未见读数**——v5 面已消费、`retry_allowed=false`，报告里 `unseen_claim_allowed=false` 恒置，任何"未见能力"主张不得引用本件；③本件不授权改产品码：读通路的产品侧改动是另一格，需 owner 另行批准；④本件不进 C3／C4／三器官任一格的证据链，发表资格前置不变。

## 1. 输入与绑定（缺件即响亮失败，attempt 不写）

九枚端点＝run88c position ×3 ＋ 89a terminal ×3 ＋ 89b cut ×3，全部走 `eval_taiji_n1_transition_factorial_confirmation_run89.py` 已有的族回执装载与 `load_with_family_protocol`，不另起装载路径；题面＝`plans/manifests/n1_fresh_confirmation_v5_20261016.json`（字节 SHA `22ba37976bc6eaaa…`，与 90 报告 `confirmation_seals.v5` 同值，两口径各自披露）；oracle 机制＝**import** 已封存的 `audit_taiji_n1_address_intervention.py` 的 `weights()`（`:61-76`）、`teacher()`（`:79-97`）、`profile()`（`:100-112`），不重抄解码环（该件字节被 75 枚在库报告钉住，也不许改）。

地址提取只能重写（被钉件的 `addresses()` 带 `split=="train"` 硬拒），因此**逐位锚点是准入条件**：在 2026-10-15 那份 train 面（`n1_native_evidence_curriculum_v3_20261015.json`，SHA `ac56fb0aba2ef602…`）上，我的提取器对每一枚可寻址题给出的 `(source, positions)` 必须与该件 `addresses()` 返回值**逐个整数相等**，命中数与不等数都要出版，不等数不为 0 ⇒ 本件拒跑。

〔2026-10-17 ㊵-1026 **§1 的复用假设被首模实测否证，本段取代原文的"import `weights()`／`teacher()`"那一句**〕锚点那一半已为真（`reports/..._attempt.failed-1.json`：checked_equal 96／unequal 0／rejected 0，覆盖 16/32 也按实测落定），但首模之后第一次进 oracle 支就崩：`TypeError: linear(): argument 'input' (position 1) must be Tensor, not LocalQuestionReadState`，出处＝`audit_taiji_n1_address_intervention.py:62` 的 `tape.keys @ model.query(hidden)` 对 `model.question_state(...)` 的返回值形状做了假设，而九枚端点用的类别是 `taiji/native_local_relation_evidence_candidate.py:15` 的 `LocalRelationEvidenceCandidate`（经 `ContextualQuestionEvidenceCandidate` 继承 `taiji/native_question_aligned_evidence_candidate.py` 的 `question_state`（`:66`）与 `step`（`:105`)，读状态是复合对象不是张量）。装载出处＝`run_taiji_n1_transition_factorial_paired.py:208`（`engineering.LocalRelationEvidenceCandidate.restore_bundle(...)`）。⇒ **"import 那两枚函数"在本族端点上不成立**，本件的机制那半条链必须另写一个类别专用适配器；为不违反"别重抄生成链"，新增两条准入条件替换原句：①适配器的 **native 支必须在 16 题 × 9 端点上逐位复现 `model.teacher_logits(...)`**（不等数入件、非 0 即整件拒），这是"我只替换了注意力"的唯一证人；②oracle 支的注意力替换只许走 `tape.keys`／`positions` 这两枚公开量，不许读写入器或参数。首模崩在写 attempt 之后、写报告之前，那份 attempt 已按不可覆盖的方式改名留存（`..._attempt.failed-1.json`，字节未动），报告槽位仍空。〕

〔2026-10-17 ㊵-1027 **条件①第一次执行就把第一版适配器拒了，成因已定位到行**〕适配器五枚函数（`forced_weights`／`prepare_state`／`adapter_step`／`oracle_teacher`／`native_anchor`）已落进仪器，oracle 两支确实可计算，但 `native_anchor` 在首模三题上**全 False** ⇒ 本版适配器不得用于出版，试跑里出现的 `position_exact=True` 之类一律不得引用。成因＝`taiji/native_relative_evidence_candidate.py:21` 的 `RelativeReadState(hidden, address, tape)` 第二字段名是 `address`，而 `:89` 构造时传进去的是上一步的 `weights` ⇒ `:80 if state.address is not None` 那条地址混合支（`self.location`／`self.transport`／`gate`）从第二跳起是**活路不是死路**；第一版只复现了 `:78-79` 与 `:85-91`，少了一级机制，所以分歧不是数值噪声。补法在此写死：照抄 `:78-91` 全段（含地址支），唯一改动仍只是替换 `:79` 的 softmax 结果，并再以 native 锚点为证人；**两次锚点（首模验证与正式运行）都为真之前 `--execute-approved` 不得发射**。〕


## 2. 覆盖与排除（按实测题面形状，不按说明）

现读 v5 的 32 枚：`lookup` 8 枚答案全为 6 字节且**都**有唯一一条 ` uses <answer>.` 观测；`revision` 8 枚同样；`transition` 8 枚答案 4–5 字节且 ` unique source` 命中 **0/8**；`unknown` 8 枚答案为 7 字节的字面 `unknown` 且无源。⇒ 可寻址集＝**16 枚／端点**（lookup＋revision），transition 与 unknown 各 8 枚**结构性排除**并逐枚给出排除原因码；总数＝16 × 9 × 3 模式＝432 条 teacher 序列。价＝按 2026-10-15 件实测 4.5 秒／576 条线性外推约 3.4 秒，另加九次装载，本件上限设 ≤120 秒 CPU 并实测用后即报。

## 3. 判据（四条，先写数再跑；动态范围问在结论问之前）

- **J1 `ruler_usable`（尺的动态范围）**：逐端点统计 native 模式的 six_exact，**必须 ≤12/16**（即至少 4 枚失败）该端点才计入后续判据；九枚里合格端点数 <6 ⇒ 整件判 `ruler_unusable`，不出机制结论，也不许"换线值再跑"。推导线写在件里：旧件 native 29/96≈30%，同比例落到 16 枚约 11.2 ⇒ 取整为 12，只要求"至少 4 枚可失败"。
- **J2 `oracle_lift`**：在 J1 合格的端点上，position 模式相对 native 的 `paired_six_gains ≥ 4/16` 且 `paired_six_losses ≤ 2/16`，二者合取；合格端点的多数（≥半数）满足 ⇒ 记 `oracle_lift_supported`，否则 `oracle_lift_not_resolved`。
- **J3 `arm_indistinguishable`**：按 seed 分组，seed 内三臂 position 模式的 six_exact 极差 `max-min ≤ 1/16` 对三个 seed 全部成立 ⇒ 记 `arm_indistinguishable=true`。这条**不是**门槛，是结论格：它为真时说明"读出臂"这个自变量在机制层也不可辨（比 EXP-N1-90 的能力层否证更深一格）。
- **J4 停下那一步的红利（只出版不判级）**：逐端点逐臂出版 position 模式下 `seven_exact` 与 `six_exact` 之差，以及末位（第 7 步）的 `byte_correct` 相对 native 的变化。旧件是"答案各步抬、末位不抬"，本件只报同轴读数是否复现。

## 4. 出口（四条互斥，读数出来后才点名哪条为真）

⑴ `oracle_lift_supported` 为真且 `arm_indistinguishable` 为假 ⇒ 下一格是**产品侧读通路改动**（让读取通路真把候选送进答案头），需 owner 批准动产品码，本件不自选；⑵ 两条都真 ⇒ "读出臂"框架本身要改写为"证据供给量"框架，本件点名要收回的句子＝"position 臂是同一实体迁移的杠杆"（该句在 EXP-N1-90 §6 已收回一次，此处是机制层的第二次、独立证据）；⑶ `oracle_lift_supported` 为假 ⇒ 瓶颈不在磁带地址而在读出／答案头一层，N1"改证据读取"这条线判否收束，转 C4 或另立读出头那一格的题面，二者都需 owner 裁；⑷ `ruler_unusable` ⇒ 本件只出版"尺没有动态范围"这一条，任何机制归属句（包括⑶）都不得写，改为另立更难的题面并先经 owner。四条互斥性由取值保证：⑷优先于其余三条，⑶与⑴⑵互斥，⑵是⑴的合取否定支。

## 5. 一次性、失败路径与前置

仪器 `scripts/training/probe_taiji_n1_oracle_address_run92.py`：`--readiness`（含 §1 的逐位锚点与 §2 的覆盖实测，不加载模型、不查询）与 `--execute-approved` 互斥；首模之前独占写 `reports/taiji_n1_oracle_address_probe_run92_attempt.json`，出版单枚 `reports/taiji_n1_oracle_address_probe_run92.json`，写靶已存在即拒。零训练（`parameter_updates=0`）、零远端动作、零外部配额。报告必须自带 `unseen_claim_allowed=false`、`oracle_is_capability=false`、逐位锚点的等／不等计数、以及 §2 的排除原因码。前置＝89a 与 89b 两批核验件为真、90 那份确认报告与其 attempt 在库（三批训练结论沿用，本件不重跑训练）。

## 6. 守卫测的形状（先写死，防我把探针做成一把只能为真的尺）

`tests/taiji_native/test_n1_run92_oracle_address_probe_contract.py` 必须走：①逐位锚点在真实 train 面上为等（命中数 >0，否则该锚点是空过）；②对合成题面（把 positions 顺序挪一位）锚点必须**报不等**；③§2 的覆盖分类在真实 v5 上为 16/32 且 transition／unknown 的排除原因码各 8 枚；④`--readiness` 与 `--execute-approved` 互斥两支与写靶已存在那支都要走到（拒入必须 rc=2 而非 traceback，DEBT-G146 那一族）；⑤`J1`／`J3` 的阈值在合成计数表上双向都能翻转（给 12 与 13 两档、给极差 1 与 2 两档）。
