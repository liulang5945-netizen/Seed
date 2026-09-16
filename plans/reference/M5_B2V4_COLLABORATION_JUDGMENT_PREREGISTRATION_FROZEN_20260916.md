# M5 B2-v4 — 协作判定 Gate 预注册（冻结版）

日期：2026-09-16。状态：**已冻结**（本文提交即冻结）。上游：B2-v3 `outcome=routing_only`（[报告](../../reports/taiji_b0_b2v3_family_20260916.json)，提交 `1f5c49ab`）+ 本轮 D5/M1 recon 勘误（2026-09-16）。授权依据：用户指示「遇到决策点默认最高上限方案，直到主线完成」+ 用户启动协作工作包（2026-09-16）。

## §0 recon 结论（工作包形态修正）

1. **D5 已解决、M4 已落地**：冻结 gate 即 `RULE_REVISION=1, COMPOSITION_RULE=m4_failure_handoff`——D5 反事实六变体中的唯一成功者已在此前 Route B 工作包中实施为冻结规则。M1-a/b/c 均被反事实实测否决（[测量](M5_B0_M1_COUNTERFACTUAL_RESULT_20260913.md)：m1a 无效、m1b 破坏基线、m2/m2a/m3 回归）。**本工作包不含任何机制改动。**
2. **G5 的真实阻塞是记账量纲，非机制、非阈值**：v3 报告实测 (a,c) 在候选面 `interleaved=18/18`、`lesion_attribution_holds=true`、逐格成功率 1.0；但三个 B2 runner 的 `cell_gain` 按成功率口径（失败=0，增益上限 1.0）对照 `REQUIRED_GAIN=1.65`。而 1.65 所在的判定体系（B0 逃生通道闭式「成功−空白=2」、D5 实测 `+2.000 > 1.65`）定义在**测量字典 ±1 结果口径**上（`SUCCESS_OUTCOME=1.0 / FAILURE_OUTCOME=−1.0`，探针与字典冻结约定）。
3. **⇒ 协作工作包 = 判定口径对齐 + 完整 G5 判定**，机制与阈值均不动。

## §1 假设

H-v4：在冻结机制（rule_revision=1）与 B2-v3 冻结选择程序下，(a,c) 部署在候选面（3 create 格 × steps 0-5）满足 G5 全部子判据——按测量字典 ±1 口径 `mean(P − max_i S_i) ≥ 1.65`、正增益 context 数 ≥ `REQUIRED_K`、lesion 归因成立、`interleaved > 0` ⇒ **`collaboration_supported`**（B 轴协作主张首次落地）。

## §2 冻结设计

1. **不变项**：四成员与 checkpoint、候选面（3 create 格 × steps 0-5 = 18 contexts）、机制 `m4_failure_handoff`（rule_revision=1）、`REQUIRED_GAIN=1.65`、`REQUIRED_K=5`、wall cap 60s——全部原样继承。
2. **选择程序**：B2-v3 冻结程序确定性重现实例化（同拟合语料 48 旧行 + 12 族行、leave-one-shape-out、同 4 维特征、同 ridge 机械）；选择结果必须与 v3 报告记录的 `v3_selected_pairs` 一致（不一致即中止并如实落账——选择程序漂移门）。
3. **执行**：与 v3 同一执行协议的全新一次 factorial（报告不覆写、不引用旧报告数值）；另加**反序诊断列**：仅 create__override 格（6 contexts）以 pair 内优先级反转执行，仅作稳健性记录、不参与门禁（M4 残余风险条款的落地：冻结顺序为主张口径）。
4. **G4（不变）**：逐格成功率口径 `pair_mean > best_singleton_mean`（部署对照判定，与 v3 一致）。
5. **G5（口径对齐）**：逐格在**测量字典 ±1 结果口径**上判定，四个子判据全部满足——
   - `policy_mean_gain_vs_all_singleton_oracle(table, pair) ≥ 1.65`（冻结 A-control-C2 估计量，`policy_mean_gain_vs_all_singleton_oracle`，与 D5 实测同一定义）；
   - 正增益 context 数 ≥ `REQUIRED_K=5`（该格 6 contexts 内）；
   - lesion：正增益 context 上全部单体结果 < pair 结果；
   - `classify_pair_trajectory` 的 `contexts_interleaved > 0`。
   同时按 D1 推荐披露可部署对照参照列（`best_fixed_singleton` 口径增益）作诊断。
6. **停止原因审计**：记录 episode 停止原因分布；`all_members_blocked`（M4 引入的新停止原因）出现次数与所在 cell 逐格落盘，纳入门禁语义审查记录。
7. **门禁继承**：G1 干预真实性、G2 数据隔离、G3 排名稳定性（对选择程序重现实例化重跑 rename/顺序置换两门）、G6 预算、checkpoint 恢复+篡改拒绝——机械与 v3 一致。三态判定 `collaboration_supported` / `routing_only` / `failed` 原文继承。
8. **对照臂**（同一 factorial）：v3 选择、v2 对照、v1 库对照、no-learning (a,b)、best fixed singleton、best observed fixed pair（诊断列）、random pair（seed 17）——按 ±1 口径与成功率口径**双列**披露，避免再发生量纲混读。

## §3 预期与判读

- 成功路径：G4 + G5 全过 ⇒ **`collaboration_supported`**——B2 出口（协作级）达成；主张范围 = 冻结机制 + 冻结选择程序 + create 候选面（族内），`growth_admitted=false`、`can_promote=false` 贯穿（能力主张 ≠ 产品采用）。
- 失败路径 A（±1 口径下增益 < 1.65 或 k/lesion/interleaved 任一不过）：如实落 `failed`，协作主张回到「任务面扩展」决策（D2 剩余项）。
- 失败路径 B（选择程序漂移门触发）：中止并归因 v3→v4 再现性，修复属工程缺陷登记，不改判据。

## §4 纪律

负结果不改绿；报告不覆写；判据/阈值/机制零改动；±1 口径与成功率口径双列披露义务；`all_members_blocked` 不得被解释为合同违例；临时脚本用毕即删；冻结后不改 §2。

## §5 产物顺序

1. `scripts/training/eval_taiji_b0_b2v4_collaboration_gate.py`（scoped 静态检查先行；复用 v3 拟合/选择机械 + 探针 outcome/table/dictionary 记账）。
2. 执行落盘 `reports/taiji_b0_b2v4_collaboration_20260916.json` + G1–G6 对账 + 双口径对照。
3. roadmap 状态表更新 + 独立提交。
