# M5 B2-v3 — 族内跨形状 pair 记录选择 Gate 预注册（冻结版）

日期：2026-09-16。状态：**已冻结**（本文提交即冻结）。上游：B2-v2 `outcome=failed`（[报告](../../reports/taiji_b0_b2v2_route_20260915.json)，提交 `8bfbe031`）+ 诊断分支命中（48 行训练交互表 `positive_rows=0`）+ 本轮 recon（2026-09-16）。授权依据：用户指示「遇到决策点默认最高上限方案，直到主线完成」。

## §0 recon 结论（设计修订记录）

1. **多文件拆分不可行**：`p52a._bind` 对*所有*动作类只绑定 `task.main_path`（交接探针实测并明载）；伴文件语言子句无人能绑定 ⇒ 拆分子句格必为全 pair 不可解。`goal_files/goal_language` 虽为 dict，但绑定层使其不可达。
2. **单文件子句空间已穷尽**：语言子句 = {可推断?}×{需显式设置?}，第 4 象限（不可推断且免显式）被合同拒绝（无证据选语言必须带 override 旗标），其余三象限恰为候选面三格；patch 行被 d 四步链全覆盖（[cell_is_combination_only](../../scripts/training/probe_taiji_b0_structure_space.py) 预测与 48 行实测一致）。
3. **⇒ 冻结合同内唯一诚实的扩格 = 候选族 train-context 扩面**：成员训练语料仍不含 create×language 形状（成员从未见过候选形状——保持不变）；只有*选择器*获得候选形状在 train-context 上的 pair 执行记录。这正是部署场景（同形状历史经验 → 留出 context 选 pair）。
4. **v2 特征在正样本就位后仍会误导**：mean_contrib 是混淆项（d 的高贡献来自 patch/none 格，与 create 形状无关；若保留，ridge 会把 (c,d) 排到 (a,c) 之前）。⇒ 特征改为 pair 级族 profile。

## §1 假设

H-v3：**create 族内 pair 互补排名跨语言路由不变**——(a,c) 在 create__{observation,override,mismatch} 三格均应 +1（事后最优 pooled +1.0 支持）。因此用族内*其他形状* train-context 的 pair 交互记录即可为留出形状选出正确 pair，且该排名可由 G3 等变性门禁约束。主张边界：**族内（create 内容路由）跨语言路由路由**；跨族路由与协作（G5）明确不在主张内。

## §2 冻结设计

1. **不变项**：四成员与 checkpoint（rule_revision=1）、候选面（3 create 格 × steps 0-5 = 18 contexts）、门禁 G1–G6 判定原文、wall cap 60s、对照臂集合——与 B2-v1/v2 逐字相同；机制 `m4_failure_handoff` 不动（M1/D5 推迟到协作工作包）。
2. **扩面拟合语料**：8 旧结构格 @ `FIT_CONTEXT_STEP`（48 行，如实保留）+ **create__observation 与 create__mismatch @ step 6**（各 6 pair 行；task index = `INDEX_BASE + ordinal*10 + 6`，与候选 context 索引 steps 0-5 不相交）。新行前缀 `b2v3fit`，与候选前缀隔离。
3. **leave-one-shape-out**：预测候选格 X 时，拟合行 = 48 旧行 + 其余两个 create 形状 @ step 6 的 12 行，**排除 X 自身行**（三套拟合，杜绝同格自证）。
4. **v3 特征（pair 级族 profile，4 维）**：特征向量 = `[1, fam[p][C], mean_contrib[p], product_contrib[p]]`，其中 `fam[p][C]` 为该 pair 在拟合行（按候选格内容路由 C 过滤）上的 mean(interaction)，`mean_contrib`/`product_contrib` 沿用 v2 定义（singleton 剖面，旧 8 格语料）。ridge λ=0.1、标准化、闭式解、截距不罚——机械与 v2 相同。
5. **选择协议**：每候选格 argmax 预测 interaction；平局取扫描序首个（first-min，保序 rename 下天然等变）。
6. **G3 稳定性**：保序成员重命名 + pair 扫描序置换两门，机械继承 v2 修复后的实现（不排序、保列位）。
7. **诊断前置**：step-6 行全量落盘；若 (a,c) 在 step-6 行 ≤0（与候选面 +1.0 不一致）→ 族内不变性被否证，如实落账，门禁照常判定。
8. **对照臂**（同一 factorial）：B1 库 learner（旧语料）、B2-v2 路由条件化 ridge（旧语料）、no-learning (a,b)、best fixed singleton、best observed fixed pair（诊断列）、random pair（seed 17）、A-count、加性。三方（v1/v2/v3）同执行数据对比 ⇒ 差异可精确归因于拟合语料+特征修订。
9. **G2 数据隔离口径**：候选 contexts（steps 0-5）与拟合 contexts（step 6）索引不相交；**形状重叠如实披露**（这正是主张边界：in-shape/族内）；成员训练语料不含候选形状——不变。

## §3 预期与判读（三态继承）

- 成功路径：v3 选 (a,c) ⇒ G4 过（1.0 > 部署对照 0.0）；G5 因 `interleaved=0` 仍不过 ⇒ **`routing_only`**——B2 轴首个非 failed 结局，主张按 §1 边界披露。
- 失败路径 A（step-6 行与候选面不一致 / fam 排名不迁移）：族内不变性被否证 ⇒ 轴级结论升级为「合同扩展（新原语或多文件绑定）+ 成员重训是唯一前路」——作为独立工作包另行预注册（Design Q），不在 v3 内实施。
- 失败路径 B（选对 pair 但 G4 不过）：执行/预算层归因，回分账细节。

## §4 纪律

负结果不改绿；报告不覆写；`growth_admitted=false`、`can_promote=false` 贯穿；`routing_only` 不得表述为协作或整模型能力；主张边界（族内/留出 context）必须在报告 interpretation 中显式出现；临时脚本用毕即删；冻结后不改 §2。

## §5 产物顺序

1. `scripts/training/eval_taiji_b0_b2v3_family_gate.py`（scoped 静态检查先行；复用 v2 runner 的执行/门禁骨架，仅拟合语料与特征段替换）。
2. 执行落盘 `reports/taiji_b0_b2v3_family_20260916.json` + G1–G6 对账 + v1/v2/v3 三方对照。
3. roadmap 状态表更新（含 §0 设计修订记录回写）+ 独立提交。
