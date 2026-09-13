# M5.K 默认 runtime rollout review 预注册：晋级评审最后一项入场条件

> 冻结日期：2026-09-11。前置：[scorecard v5 合同](M5_K_AXIS_SCORECARD_V5_CONTRACT_20260911.md)（`k_worker_joint_course_completed=true`，晋级评审入场条件仅剩 `default_runtime_rollout_review_completed=false`）。本文冻结 review 的**范围**（默认 runtime 在何种合同下消费 post-K/求解器 checkpoint）、**rollout 指标**与**失败回滚边界**；冻结后按 §8 顺序执行。本 review 不是 P4.x 实验系列的新实验：不引入新训练、新臂或新机制，`fit_called=false` 贯穿。

## 1. 可证伪假设

P4.14 的全门行为（4/4 cell）是在研究 harness 中、由内存中的 learner 对象直接驱动取得的。默认 runtime 若要消费同一批 checkpoint，必须经过一条与研究 harness 不同的路径：**磁盘上的 content-addressed artifact → manifest 驱动的 fail-closed load → 独立进程恢复 → 行为 rollout → rollback**。

**假设**：通过该消费路径重新加载 P4.14 post-K/求解器 checkpoint 后，全部已冻结门指标**逐数值复现** P4.14 报告记录值；即「研究侧机制闭合」的产物在合同化消费路径下是同一行为，不因加载方式、进程边界或加载器实现而漂移。任何偏差即消费缺口，如实入账。

## 2. 范围与消费合同（review 冻结的对象）

**默认 runtime 现状盘点（只读，入 review 报告）**：默认 runtime 为 `api/seed_runtime.py` 的 `SeedRuntime`，消费 `checkpoints/seed_corpus.pt`（`seed-native-v1`：taiji adapter/kernel/cognitive_state/components）。`api/`、`seed_platform/`、`taiji/adapter.py` 中无任何 `g_selection`/`k_worker` 引用——K/G 机制当前完全不在默认 runtime 上，此 fail-closed 现状如实记录。运行时基线 = review 运行时 `seed_corpus.pt` 的 content digest（前后必须一致）。

**消费对象**：P4.14 联合课程的全部 4 个 cell（batch-0/1 × seed-0/1）——post-K K1 worker（`taiji-structured-semantic-training-v1`）/ K2 worker（`taiji-structured-semantic-transition-v3`）+ 投影后扩展 G 头（`taiji-extended-g-selection-learner-v1`，17 参数）。post-K K1/K2 跨 cell 逐位相同（P4.14 已记录），phase-G checkpoint 逐 cell 独立（4 个唯一 digest）——合同逐 cell 钉定，不允许混装。

**消费合同 `taiji-default-runtime-rollout-attachment-v1`**（落盘为 `plans/manifests/taiji_m5_k_default_runtime_rollout_attachment_v1.json`，git 入库）：

1. **内容寻址 attachment manifest**：逐 cell 的 K1/K2/G artifact digest、artifact 来源路径（P4.14 run dir，逐位校验后才可消费）、lineage（P4.14 report digest + manifest digest）、嵌入安全不变量（K1 `confidence_floor=0.55`/`fact_threshold=0.65`/`ambiguity_ceiling=0.12`、G `confidence_floor=0.55`/`selection_margin=0.05`/`parameter_count=17`、K 参数 5,648）、冻结门阈值引用（§3）；
2. **fail-closed load 顺序**：attachment manifest 自身 digest 校验 → P4.14 lineage 校验 → 本地 artifact 逐位 digest 校验 → 嵌入不变量校验 → 独立进程恢复 → 篡改/错 cell 混装/错 lineage 拒绝。任一步失败即拒绝消费：不回退、不降级、不重建 checkpoint（cohort 数据可按 manifest digest 确定性重物化，checkpoint 一律不可）；
3. **运行时禁止项**：消费路径不得 fit、不得改写任何研究 artifact 或默认 checkpoint、不得以运行时侧 override 覆盖 checkpoint 内嵌安全不变量（不变量从 checkpoint 读出并核验，不由调用方提供）；
4. **落点（诚实边界）**：本 review 在沙箱 runner 中以合同相容的 loader 演示消费路径并定义 runtime 侧 adapter 的合同；不修改 `api/`/`seed_platform/` 产品代码、不接线上服务、不解冻任何 owner——真正产品 attach 属 A8 晋级评审之后的独立工程与审批决策。

**rollout 定义**：一次 rollout = 消费路径对一组冻结 cohort 的完整评估回合——load → preflight → 逐条 observation → K1 readout → K2 readout → G selection → 安全投影 → 隔离 Workbench 记账——产出与 P4.14 相同口径的指标。cohort（Phase K novel validation + P1 五类 validation；Phase G validation/holdout/retention-sibling/retention-newtask）按 P4.14 manifest 确定性重物化并逐位校验 digest，不得重生成新身份。

## 3. rollout 指标与门（全部冻结，零变更；每 cell 4/4 全部适用）

| 门 | 冻结判据 | 来源 |
|---|---|---|
| 消费等价（本 review 核心新证据） | 通过消费路径加载后，Phase K 评估（novel K2 content 2/2；旧类 K1/K2 goal/content 4/4；安全 abstention 6/6；workbench 非降）与 Phase G 评估（validation/holdout/retention-newtask：target `0.8`、utility `0.8675`、0 safe violation、reobserve projection 通过；retention-sibling `1.0/1.0`）**逐字段等于 P4.14 报告记录值** | P4.14 报告 |
| post-K digest 锚 | K1/K2 digest = P4.14 报告 post-K 值（`9fe6f48187ad…` / `81a1891c470b…`），且 G 相评估前后不变（`k_unchanged` 消费版） | P4.14 报告 |
| G checkpoint 锚 | 逐 cell phase-G digest 与 P4.14 本地产物逐位一致；`parent_manifest_digest`/`feature_source_state_digest`/`model_state_digest` 通过 | P4.14 产物 |
| 嵌入不变量保持 | 上述 §2.1 全部嵌入字段与 checkpoint 内值逐位一致；loader 无 override 路径 | 合同 |
| 非干扰 | `fit_called=false`；`seed_corpus.pt` digest 前后一致；全部研究 artifact digest 前后一致 | 本 review |
| 恢复 / rollback / 篡改 | 每 cell 独立进程恢复后行为逐数值等价；rollback（P3.0/P4.13 机械）通过；tampered/错误 lineage/错 cell 混装全部拒绝 | P3.0/P4.13 合同 |
| 资源绝对预算 | 每 cell rollout ≤ `120s`、review 总计 ≤ `600s`；checkpoint 字节与内存/延迟记录 | P4.12 教训 |

## 4. 矩阵

**4/4 cell 全部执行、全部通过**（无子采样、无例外；任一 cell 失败即整体失败）。理由：消费对象就是 P4.14 的全部 cell，晋级评审入场条件不允许抽样；4 cell 覆盖两个 G checkpoint 变体维度（batch × seed）与共享 post-K worker 的全部组合。

## 5. 失败回滚边界与结果映射（执行后禁止调门）

| 分支 | 条件 | 判定 |
|---|---|---|
| `default_runtime_rollout_review_supported` | 4/4 cell 全门（消费等价 + 机械 + 非干扰 + rollback） | scorecard v6 将 `default_runtime_rollout_review_completed=true`（唯一翻转）；**A8 晋级评审获得召开资格，仍须独立批准**；`can_promote=false`、其余 veto 原样保留 |
| `rollout_consumption_gap` | 任一 cell 消费等价门失败（行为指标 ≠ 记录值） | 消费合同或 runtime 侧重建与研究 harness 存在真实偏差——停止，进入桥接设计；不得调门、不得重训、不得改写 P4.14 产物 |
| `rollback_unverified` | 任一 cell 独立恢复/rollback 失败 | 消费合同缺少可恢复路径——诚实停止；attach 无从谈起 |
| 机械失败 | digest/lineage/篡改门失败、artifact 缺失或不匹配、资源预算超限 | `status=failed` + 归因，停止；不静默重物化 checkpoint |

**回滚边界（冻结）**：本 review 全程不触碰默认 checkpoint 与研究 artifact（digest 证明即回滚保证）；每 cell 的 rollback 门（独立进程恢复 → 行为逐数值等价）**在行为门失败时仍必须执行并通过**——失败也要留下可恢复的证据，不得因失败跳过恢复验证。

## 6. 边界（诚实声明）

- 本 review 是**沙箱合同验证**，不是产品接入：不修改产品代码、不启动线上服务、无用户流量、无 owner 解冻；`default_runtime_owner_attached` 与 `learning_mechanism_attached_default_runtime` 保持 `false`，属 A8 晋级评审的决策对象；
- runtime 侧 adapter 的生产实现不在本 review 范围；本 review 只交付并验证其合同；
- validation-only；不读取 sealed；`fit_called=false`、零训练步、零投影重跑；不重算任何历史 formal；
- 五类合成课程仍是唯一实验载体；本 review 不把机制证据写成完整认知能力或结构成长；`growth_admitted=false`、`can_promote=false` 贯穿。

## 7. 与既有教训的对齐

- **P3.0/P4.13 教训**：rollback = 独立进程恢复 + 行为逐数值等价，机械门直接复用原函数；
- **P4.12 教训**：资源 cap 以绝对预算定义，不用比率式软门；
- **P2.2/P3.1 教训**：typed abstention 与 confidence floor 是 checkpoint 内嵌合同，运行时不得覆盖、不得降 floor；
- **P4.14 教训**：post-K digest（`9fe6f48187ad…`/`81a1891c470b…`）是 K 侧消费锚；phase-G checkpoint 逐 cell 独立，禁止混装；
- **fail-closed 纪律**：任何 digest 漂移即停止，先修机械再谈行为。

## 8. 产物顺序

1. `scripts/training/eval_taiji_m5_k_default_runtime_rollout_review.py`（消费合同 + rollout runner：attachment manifest 构建/校验 + fail-closed loader + 独立进程 rollout + rollback/tamper 门 + 资源审计；py_compile/ruff/mypy 先行）；
2. 执行产出 `plans/manifests/taiji_m5_k_default_runtime_rollout_attachment_v1.json` + `reports/taiji_m5_k_default_runtime_rollout_review_20260911.json`；任一停止线触发即停；
3. 路线图同步 + scorecard v6（若 `default_runtime_rollout_review_supported` 则 `default_runtime_rollout_review_completed=true`，A8 晋级评审入场条件齐备）+ 独立提交。

## 9. 执行记录（2026-09-11，review 已运行）

1. **Runner**：`scripts/training/eval_taiji_m5_k_default_runtime_rollout_review.py`（py_compile / ruff / black / mypy --follow-imports=silent 通过后执行；执行中修复三处机械问题，均属合同构建错误而非工件问题：(a) attachment 自摘要校验须用 `_digest_without` 排除 `manifest_digest` 键；(b) G artifact 的钉定摘要是**排除其内嵌 `checkpoint_digest` 键**的约定（与 P4.13 `_save_extended_checkpoint` 一致），K 工件无自摘要键用全量摘要；(c) K2 transition 的内嵌 `fact_threshold=0.55`（区别于 K1 的 `0.65`），合同按角色分别钉定；独立恢复校验器对 extended G 须用 P4.13 `_independent_extended_restore`，不能用 P3.5 的基础版 G 校验器）。
2. **消费合同落盘**：`plans/manifests/taiji_m5_k_default_runtime_rollout_attachment_v1.json`（`taiji-default-runtime-rollout-attachment-v1`；逐 cell K1/K2/G artifact digest 钉自 P4.14 报告、P4.14 lineage、嵌入不变量、`post_k_k_digests_shared` 锚、绝对资源预算）。
3. **报告** `reports/taiji_m5_k_default_runtime_rollout_review_20260911.json`：**`outcome=default_runtime_rollout_review_supported`，4/4 cell 全门通过**（`experiment_passed=true`、`fit_called=false`、`training_performed=false`）。
   - **消费等价（核心新证据）**：通过磁盘加载的消费路径，Phase K（novel K2 content 2/2、旧类 K1/K2 goal/content 4/4、安全 abstention 6/6、workbench 非降）与 Phase G（validation/holdout/retention-newtask target `0.8`/utility `0.8675`/0sv；retention-sibling `1.0/1.0`）在全部 4 个 cell **逐字段等于 P4.14 报告记录值**；cohort 重物化（novel 记录 digest + 每 batch 88 个 candidate-set digest）与 P4.14 manifest 逐位一致。
   - **机械门**：artifact digest/嵌入不变量全过（K 参数 5,648、G 参数 17、confidence floor 0.55、K1 fact_threshold 0.65 / K2 0.55、G parent_manifest_digest = P4.1 `source_p3_2_manifest_digest`、feature source digest 跨 cell 唯一共享）；K1/K2/G 篡改全部拒绝；wrong-cell 混装被合同拒绝；post-K 锚（`9fe6f48187ad…`/`81a1891c470b…`）与报告一致。
   - **恢复/rollback**：每 cell K 目录（pilot verifier）与 extended G（P4.13 verifier）独立进程恢复全过；全新重载后 G selection 零 mismatch、K 侧 p1/novel 摘要逐字段相等（行为逐数值等价）；G 评估后 K consumer state 重序列化 digest 仍等于 post-K 锚（`k_unchanged` 消费版）。
   - **非干扰**：`api/seed_runtime.py` 与 `checkpoints/seed_corpus.pt` sha256 前后一致；全部研究 artifact 与 parent 锚（P3.2 K workers、P3.5 G checkpoint）digest 前后一致；runtime inventory 对 `seed_runtime.py` 的 K/G 引用扫描为零。
   - **资源绝对预算**：cell rollout 2.0–2.1s ≪ 120s，review 总计 267.0s ≪ 600s，0 违规。
4. **scorecard v6**：[M5_K_AXIS_SCORECARD_V6_CONTRACT_20260911.md](M5_K_AXIS_SCORECARD_V6_CONTRACT_20260911.md) 冻结并运行——`default_runtime_rollout_review_completed=true`（唯一门翻转），`promotion_gate=false`、`can_promote=false`；A8 晋级评审两个入场条件齐备，评审召开须独立批准。
5. `growth_admitted=false`、`can_promote=false` 贯穿；无 owner 解冻、无默认 runtime 接入、无 sealed 读取。唯一下一步 = **A8 晋级评审**（逐项裁决其余 veto）。
