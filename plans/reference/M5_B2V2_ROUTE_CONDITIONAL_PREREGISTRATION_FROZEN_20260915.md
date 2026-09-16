# M5 B2-v2 — 路由条件化互补表示 选择 Gate 预注册（冻结版）

日期：2026-09-15。状态：**已冻结**（本文提交即冻结）。上游：B2-v1 `outcome=failed`（[报告](../../reports/taiji_b0_b2_selection_20260915.json)，提交 `5abb123a`）+ 只读归因（roadmap B2 行，2026-09-15）：① 6 维特征无法表达跨能力互补（(a,c) 与 (a,b) 特征行相同而真实收益 1.0 vs 0.0）；② 交互符号在训练面（patch/none 行冗余惩罚）与候选面（create 行互补收益）间反转。授权依据：用户指示「遇到决策点默认最高上限方案，直到主线完成」。

## §1 假设

H-v2：**路由条件化的独占覆盖特征**能把「互补独占」与「冗余重叠」分成两个通道，使选择器在训练面学到"独占覆盖 → 正交互"，并在候选面（create × 3 语言路由）把 member-a+member-c 排到首位，产生真实执行收益（G4 过）；协作主张仍另过 G5。

机制依据（v1 归因）：member-b 与 member-c 在 index%4 块先验下是**孪生 profile**（贡献均 0.25、surface 均 {2}）——任何 pair 级 profile 特征都无法区分 (a,b) 与 (a,c)。**更细的先验 = 内容路由（none/create/patch）**：b 的能力在 patch 路由（patch__none），c 的能力在 create 路由（create__none）；内容路由是任务/goal 的可见结构属性（部署时选择器可见），非结果泄漏，预注册披露。

## §2 冻结设计

1. **成员与语料不变**：四成员 readout（rule_revision=1）；fit 语料 = 8 非候选结构格 × step 0（与 B1/B2-v1 完全一致，隔离条款同 §4）；候选面 = 3 create 格 × steps 0-5。**只改表示，不改语料**——v1→v2 差异可归因于特征修正。
2. **v2 选择器（runner-local，库不动）**：库 `InteractionGroupTransferLearner` 的 `_pair_features` 固定且块数=4，无法承载路由粒度 ⇒ v2 在 runner 内实现 ridge 选择器，披露自有 checkpoint 格式 + fresh-process 恢复 + 篡改拒绝 + 重命名/顺序置换稳定性。库 learner 保留作 B1 对照不删除。
3. **路由条件化 profile**（train-only）：`delta[m][route]` = 成员 m 在该内容路由训练格上的 mean(单成员成功率 − baseline 成功率)（none/create/patch 三路由；8 格映射：none×3、create×1、patch×4）。全局 `contribution[m]` 照旧。
4. **v2 pair 特征**（逐候选 cell，route = 该 cell 的内容路由）：bias 1.0；`exclusive_i`（i 覆盖 route 且 j 不覆盖 ⇒ max(0, delta_i[route])，否则 0）；`exclusive_j` 对称；`redundant`（双覆盖 ⇒ min(max(0,delta_i),max(0,delta_j))）；`neither_covers` ∈ {0,1}；`mean_contrib`、`product_contrib`。共 8 维。
5. **拟合目标**：fit 语料上全部 6 pair × 8 格的 `interaction = P − S_i − S_j + B`（48 行，全 train-only；较 v1 的 6 条评估器记录更充分）。ridge λ=0.1，特征标准化，闭式解。**诊断前置**：48 行交互表全量落盘报告；若训练面正互补模式缺失（全部 ≤0），v2 如实拟合（预测趋零）并把「训练面需扩格」记为轴级发现——两分支都不改判据。
6. **选择协议**：对每个候选 cell 用 v2 预测 `interaction` 最高的 pair（三格同预测，与 v1 相同的 context-free 边界，如实披露）。
7. **执行与门禁**：与 B2-v1 逐字相同——396 episodes 真实合同 factorial 一次取得 selected/lesion/基线/全部对照；G1–G6 判定原文继承；三态 `collaboration_supported` / `routing_only` / `failed`（表示失效态由 `representation` 门承接）；wall cap 60s。
8. **对照臂**（同一 factorial）：no-learning 回退 (a,b)、best fixed singleton、best observed fixed pair（诊断列）、random pair（seed 17）、A-count、加性、**B1 库 learner 的选择**（v1 基线对照——验证 v2 相对 v1 的选择改进）。

## §3 预期与判读

- 成功路径：v2 把 (a,c) 排首位 ⇒ G4 过 ⇒ 若逐格 gain ≥1.65 且 k≥5/6 且 lesion+interleaved ⇒ `collaboration_supported`（仍受路线 B §7 界限）。
- 失败路径 A（v2 仍选错 pair）：表示假设被否证 ⇒ 轴级结论"profile 级 pair 模型不足以表达该互补结构"，回 VISION 候选（结构升级）或接受边界。
- 失败路径 B（v2 选对 pair 但收益不足）：执行/任务层面归因，回 G5 分账细节。
- 诊断分支（训练面无正互补）：拟合必然趋零 ⇒ 如实记录 + 「训练面扩格」成为唯一后继决策。

## §4 纪律

负结果不改绿；报告不覆写；`growth_admitted=false`、`can_promote=false` 贯穿；判别度/选择分数不得替代整模型能力或协作主张；临时脚本用毕即删；冻结后不改 §2。

## §5 产物顺序

1. `scripts/training/eval_taiji_b0_b2v2_route_gate.py`（scoped 静态检查先行）。
2. 执行落盘 `reports/taiji_b0_b2v2_route_20260915.json` + G1–G6 对账 + v1→v2 对照。
3. roadmap 状态表更新 + 独立提交。
