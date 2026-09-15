# M5 WP-5a — B2 离线学习选择与真实收益 Gate 预注册（冻结版）

日期：2026-09-15。状态：**已冻结**（本文提交即冻结；估计目标、门禁与三态全部继承[路线 B 冻结版](M5_B0_ROUTE_B_PREREGISTRATION_FROZEN_20260915.md)（binder v1），本文只冻结 B2 特有协议，原文冲突处以路线 B 为准）。
上游：B1 表示门 `representation_discriminative`（[报告](../../reports/taiji_b0_b1_representation_20260915.json)，提交 `0d43ed97`，feature_rank 4）；B1 训练授权由用户 2026-09-15 给出，B2 属同一协作轴推进（[推进计划修订](M5_B0_CLOSEOUT_AND_PLAN_REVISION_20260914.md) WP-5a）。

## §1 研究问题（继承路线 B §1，不重述）

B2 回答：**B1 表示驱动的选择器所选组合，在候选面上经真实合同执行后，能否产出超过冻结对照的真实收益？** H1（路由）/H2（协作）判别式与门禁（G1–G6）全部按路线 B §2/§5 执行；B1 的判别度只是排序先验，不构成任何收益主张。

## §2 B2 特有协议（本文冻结内容）

1. **成员与 fit**：与 B1 完全一致——四成员 family-specialist readout（`_train_members`，rule_revision=1），fit 语料 = 8 个非候选结构格 step 0（官方 train-only 面：`build_member_evidence` → `observe_members` → `evaluator.train_only_candidates`（资源上限 64，披露）→ `observe_records`）。
2. **选择协议**：`learner.select(6 pairs, unseen_only=False)` 取 `predicted_interaction` 最高者为 **selected pair**（表示不条件化于 context，故三候选格同一选择——这是当前表示的如实边界，如实报告不得粉饰为逐格路由）。
3. **执行**：候选面 3 格 × 6 context（frozen steps 0-5）× 11 cells × 2 repeats = **396 episodes** 全 factorial 经真实合同执行——selected pair、lesion（其两单成员格）、基线、全部对照一次取得。
4. **估计目标**（路线 B §2 冻结公式）：主判据 `gain_vs_all_singleton_oracle = mean(P − max_i S_i)` 逐格；`required = 1.65`、`available = 6`、k 判据 `5/6`（`reference_requirements()` 计算并核对冻结表）；`causal_interaction = P − S_i − S_j + B` 分项报告。
5. **门禁判定**（路线 B §5 原文）：
   - G1 干预真实性：非 baseline cell 零步 ⇒ false（P5.2b 三层修复承接）；
   - G2 数据隔离：§4 条款（候选 pair 结果不入 fit——结构断言 fit task_ids ∩ candidate task_ids = ∅）；
   - G3 排序验证：重命名 + 顺序置换稳定（B1 同款）；
   - G4 任务门（H1）：selected pair 增益 > 最强可部署对照（**best fixed singleton**；oracle 不可部署不作 H1 对照）；
   - G5 协作门（H2）：`gain_vs_all_singleton_oracle` 逐格达标（k ≥ 5/6）**且** lesion 归因成立（selected pair 的两成员各自 lesion——即其单成员格——在 pair 成功的 context 上失败）**且** `interleaved > 0`；
   - G6 预算门：wall ≤ **60s**（B1 实测 37.5s + 选择开销；超限如实记 failed 不调 cap）。
6. **对照臂**（路线 B §7/工作包清单）：no-learning（未拟合表示 → 无候选 → 回退固定 (a,b)）、best fixed singleton（可部署最强单体路由）、best observed fixed pair（诊断列）、random pair（seed 固定）、A-count、加性模型、**selected pair**。全部对照的增益来自同一 factorial 执行，零额外执行成本。
7. **checkpoint**：fitted transfer learner JSON 落盘 + fresh-process 恢复复现 selection；篡改拒绝。

## §3 三态（路线 B §10 原文）

- `collaboration_supported`：G4、G5 均过且 lesion 成立、`interleaved > 0`（仍受路线 B §7 界限：承重结构因素 1 个，三条语言路由同一指纹）。
- `routing_only`：G4 过、G5 不过。
- `representation_ineffective`：表示不可区分候选（rank ≤ 1 或预测恒定）。
- `failed`：任一机械/合同门失败（含 G1/G2/G3/G6）。

## §4 停止点

公平 oracle 无法超过 `required` ⇒ 停止回参照讨论；候选格有效性门任一失败 ⇒ 该格如实剔除；冻结判据/任务/binder 不得在实现中顺手改；负结果不改绿。

## §5 产物顺序

1. `scripts/training/eval_taiji_b0_b2_selection_gate.py`（scoped 静态检查先行）。
2. 执行落盘 `reports/taiji_b0_b2_selection_20260915.json` + G1–G6 对账。
3. roadmap 状态表与唯一下一步更新 + 独立提交。

## §6 纪律

负结果如实落账；报告不覆写不改绿；`growth_admitted=false`、`can_promote=false` 贯穿；判别度/选择分数不得替代整模型能力或协作主张；临时脚本用毕即删；远端未查询不表述「CI 已绿」。
