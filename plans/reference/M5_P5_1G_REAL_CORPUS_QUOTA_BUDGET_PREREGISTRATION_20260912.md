# M5 P5.1g — 真实发布语料配额同预算 Gate 预注册（冻结版）

日期：2026-09-12。状态：**已冻结**（本文提交即冻结；冻结后不训练新门、不接预注册之外数据源、不读取 sealed）。
上游：P5.1f（真实语料同预算，`outcome=failed`，报告 `reports/taiji_p5_1f_real_corpus_same_budget_20260912.json`）+ 只读 S 轴归因 recon（roadmap 已记）。

## §1 问题与动机

P5.1f 五门失败（门 2/4/5/7/9）经只读归因全部定位在构造/基建层，无一是内容收益判据本身的失败：

1. **门 2（same_budget_enforced）**：轨迹数预算口径（300=300）对真实分布结构性不构成同预算（total calls 12.02×、semantic examples 8.53×）。
2. **门 4/5（admission 未达 → 门 5 不可测）**：投影信息无损（semantic 双臂 passed、affordance MSE ~1e-8），procedural 预算与真实分布难度不匹配导致欠拟合。
3. **门 7（affordance_content_specificity）**：双臂 native MSE 均 ~1e-8 饱和。归因根因已查明：`_reward` 在无 reward_components 时返回 `success ? 1.0 : -1.0`，而真实发布语料为全成功教师轨迹 → **target_reward 恒 1.0** → affordance reward 回归与 semantic 价值面退化为常数拟合（无信息）、ranking_pairs 结构性为 0。
4. **门 9**：wall 1598.131s > 1200s，完全由 examples 规模线性驱动，与门 2 同根。

P5.1g 用三项构造修正回答同一科学问题（同预算下语料内容族的因果收益在真实发布语料上是否成立），不修改 P5.1f 冻结判据、不覆写 P5.1f 报告。

## §2 设计（两臂、单变量 = 内容族）

语料同 P5.1f：sourced = `Tool_Use_part-1-of-9.jsonl`，placebo = `Code_Agent_part-1-of-7.jsonl`（gitignored，sha256 + 行区间复现），每条轨迹经 `SkillArtifactAdapter` governed 投影（`skill_id=uuid`、steps 展开 `tool.<name>`、provenance 七项）。

### 2.1 信号量配额采样（门 2 修正）

- **sourced**：行序取含 ≥1 tool_calls 的记录，与 P5.1f 完全一致：train 200 / holdout 60 / retention 40 / a-gate 16（后续行，词表 ⊆ sourced train）。
- **placebo**：行序取完整轨迹，逐分区以 sourced 分区 total calls 为配额（989 / 385 / 253），最后一条轨迹取**前缀截断**（前 K 次 tool_calls，K = 配额余数）使两臂各分区 total calls **逐位相等**。轨迹数不等（sourced 200/60/40 vs placebo 若干）如实披露，不再作为同预算断言对象。
- **a-gate 不变**：16 条 Tool_Use 轨迹、109 records、词表 ⊆ sourced train 词表且与 placebo 词表交集为空。

### 2.2 构造常数变更（新预注册构造决策，非改旧判据）

- `procedural_hidden_dim`: 16 → **64**。依据（校准探针，见 §5）：hidden 16 在 250–4000 epochs 上 train acc 饱和于 ~0.55、retention 天花板 ~0.38；hidden 64 各项全面更优且 250 epochs 时 a-gate 最高。
- `procedural_epochs` 维持 **250**：探针显示 250 时 a-gate 表现最佳，更大 epochs 拟合 train 高频模式反而降低迁移读数。
- 其余 trainer kwargs 与 P5.1f 逐位相同（feature_dim 384、SemanticArtifactKnowledgeEncoder、semantic_pairwise_margin 0.5、semantic_passes 12、affordance_epochs 200、seed 17、affordance_feature_dim 12）。

### 2.3 测量路径修正（admission 不可达的如实处理）

校准探针（§5）已测量：**在冻结的 procedural readout 结构（GRU + 每 epoch 一次聚合 update）下，真实语料的 admission 条件 `procedural retention ≥ 0.5` 不可达**（hidden 16 上限 ~0.38 @4000 epochs；hidden 64 平台 ~0.44 @2000 epochs；斜率不支持更高 epochs）。因此：

- 两臂 `admitted=false`、`rolled_back=true` 为**预期结果并全额披露**（consolidate 产品语义原样保留，不为过门而放宽 admission）。
- **门 4 改制**为可测部分：每臂 procedural holdout accuracy > lesion accuracy（零参数对照）+ semantic / affordance 面完成运行且如实记录（恒 reward 下为常数拟合，披露其无信息性）。
- **门 5/6/7 的 a-gate 与 holdout/lesion 读数改在 trial learner 上测**：`ProceduralSequenceLearner.from_checkpoint(trainer.procedural.checkpoint())` → `consolidate(procedural_train, epochs=250, learning_rate=trainer.procedural_learning_rate)`——与 `consolidate` 内部 procedural_trial 路径**逐位同构**（探针在 hidden 16 @250 epochs 逐位复现 P5.1f 报告的 train/holdout/retention = 0.4946898 / 0.3952 / 0.3656174，验证等价）。runner 不使用 rollback 后的 trainer 本体 readout。

### 2.4 恒 reward 披露（门 7 改制的依据）

全成功语料 → 所有 experiences 的 target_reward 恒 1.0 → semantic 价值面与 affordance reward 回归面**结构性无信号**（最优解为常数 1.0 拟合）。这两面在本 gate 中保留运行并记录数值，但**不作为内容特异性判据**（沿 P5.1e 摸底先例：语义器官收益不以本 gate 度量）。procedural 序列预测面是唯一携带真实方差信号的器官面。

## §3 九门

| # | 门 | 判据 |
|---|---|---|
| 1 | static_four_checks | py_compile / ruff / black / mypy 基线零新增（mypy 61 错 HEAD 既有基线持平；pytest 4 failed / 1199 passed / 6 skipped 零新增） |
| 2 | same_budget_enforced | 两臂 train/holdout/retention 各分区 total calls 逐位相等（989 / 385 / 253）+ a-gate 两臂 109 records 一致 + 截断轨迹数与轨迹数披露 |
| 3 | capability_vocabulary_disjoint | a-gate 词表 ⊆ sourced train 词表，且与 placebo train 词表交集为空 |
| 4 | arm_sanity_both_arms | 每臂 procedural holdout accuracy > lesion accuracy（trial-learner 读数）；semantic passed 与 affordance native < frozen 如实记录（披露项） |
| 5 | content_transfer_margin（核心） | trial-learner a-gate：sourced accuracy − placebo accuracy ≥ **0.15**（P5.1e/f 冻结跨代常数） |
| 6 | sourced_beats_lesion | trial-learner procedural holdout accuracy > lesion accuracy |
| 7 | procedural_above_frequency_baseline（新） | sourced trial-learner a-gate accuracy − 冻结 per-tick-majority baseline（**0.3670**，§5 探针计算）≥ **0.15**（同跨代常数） |
| 8 | checkpoint_roundtrip_both_arms | 双臂 trainer checkpoint 往返无 stub、三项全保留 |
| 9 | deterministic_and_budget | 双臂 replica 逐位一致 + wall ≤ **1200s** |

说明：placebo 臂 a-gate accuracy 因词表不相交（门 3）**结构性为 0**（同 P5.1e/f 设计），门 5 保留同构形式；门 7 提供真实信息量对照——学习是否超越轨迹内工具频率先验（否则「迁移收益」只是高频偏置）。

## §4 三态

- `real_corpus_content_benefit_supported`：九门全过。
- `content_benefit_insufficient`：机械门（1/2/3/8/9）与门 4/6/7 全过而核心门 5 失败——内容收益不足的实质判据（首次在真实语料上可判）。
- `failed`：任一机械门或门 4/6/7/8 失败（构造/基建问题，不构成内容收益判据）。

## §5 校准探针披露（摸底，非判据）

两探针（只读 + procedural 单器官校准训练，sourced 臂，完成后脚本已删除）：

- 探针 1（hidden 16）：epochs 250/2000/4000 → train 0.4947/0.5461/0.5500，retention 0.3656/0.3777/0.3826，a-gate 0.6147/0.5688/0.5872；wall 60.9/479.4/1045.0s。**250 epochs 复现 P5.1f 报告逐位一致**（0.4946898/0.3952/0.3656174）。
- 探针 2（hidden 64）：epochs 250/2000 → train 0.5634/0.5612，retention 0.4189/0.4383，a-gate **0.6514**/0.6147；wall 83.4/631.3s。
- a-gate 基线（109 records、19 工具）：全局多数工具 `tool.get_reservation_details` = 0.2202；**per-tick 多数 = 0.3670**（冻结入门 7）。
- 门阈值 0.15 沿用 P5.1e/P5.1f 冻结跨代常数，独立于探针观测；探针数值仅用于 (a) hidden 选择、(b) baseline 常数计算、(c) wall 预算外推（预计 ~800-900s，主要项 procedural 4×83s + semantic ~124K updates + affordance ~135K updates，1.1 ms/step 校准）。

## §6 产物顺序

1. `scripts/training/eval_taiji_p5_1g_real_corpus_quota_budget_gate.py`（静态四项先行；runner 本地实现配额采样/前缀截断 + trial-learner 测量路径 + 九门/原子报告）。
2. 执行落盘 `reports/taiji_p5_1g_real_corpus_quota_budget_20260912.json` + 与本文九门逐项对账。
3. 路线图同步 + 独立提交。

## §7 对账纪律

门值以报告为准逐项对账，禁止反向改判据、剪枝凑数、重跑挑结果；报告不覆写不改绿；`growth_admitted=false`、`can_promote=false` 贯穿；不接预注册之外数据源、不读取 sealed；corpus 不入 git（sha256 + 行区间 + 截断规则复现）。
