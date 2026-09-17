# R2-H3.8 matched dev 预注册与数值预算冻结（冻结版）

2026-09-17。状态：**已冻结**（本文提交即冻结）。上游：H3.8 [隔离原型实现合同](M5_R2_H3_8_ISOLATED_PROTOTYPE_CONTRACT_20260917.md)（§4/§5 要求数值预算与 dev 指标先于 dev 数据冻结）+ [数据合同](../../reports/r2_h3_8_data_contract_20260917.json)（corpus digest `5518ff500bbcb3551bc60cc13783442cd9fce870e8d17ee9dd4a7e165a17f6a8`）+ [train-only 可学习性](../../reports/r2_h3_8_train_learnability_20260917.json)（两臂可学、共同 LR 冻结 0.01）。**dev split 从未被评分**——本文全部阈值在看不到任何 dev 数值的情况下冻结。

## §1 问题与主张边界

问题：修复六维归因（[总账](M5_R2_H3_9_ATTRIBUTION_LEDGER_20260917.md)：监督不含前缀依赖、renderer 无跨 byte 可学习状态）后，**带内容寻址工作空间的可训练联合序列原型，在未见实体与未见措辞的 dev 上是否优于同预算无工作空间基线**。

边界：dev 结果仅支持「该隔离原型在冻结数据合同与预算下的 dev 级比较」，**不等于 L2/Mini 晋级、不等于产品采用、不读 final**；`growth_admitted=false`、`can_promote=false` 贯穿。

## §2 数值预算冻结（三 seed × 两臂 + 病灶列）

| 项 | 冻结值 |
|---|---|
| seeds | 20260917 / 20260918 / 20260919（各臂各 seed 独立初始化） |
| 数据 | corpus digest `5518ff50…`；train split 25 episodes；**dev 16 episodes 仅评分**；final 不读 |
| 优化 | Adam，共享学习率 **0.01**（可学习性探针已证两臂在该率下稳定；**不按臂调参**）；lr 不调度 |
| 步数 | 30 epochs × 25 episodes = **750 步/臂**；batch = 1 episode |
| 墙钟上限 | **20 分钟/臂**；超时即停，如实落账，不自动加预算 |
| 存储上限 | 512 MiB 空闲要求 + 每次 checkpoint ≤ 8 MiB |
| checkpoint 周期 | epoch 0（零步）/ 15 / 30 各存一次；每次存后新进程复核 digest 一致 |
| 恢复/异常策略 | 任何非有限值、预算不匹配、digest 不一致 ⇒ 该臂该 seed 停止并保留失败态；不自动重试、不扩 epoch |
| 快照 | 训练后可训练参数以 CPU float32 克隆入报告目录（不入主模型） |

**参数预算披露**（合同 §3「匹配不了就分账」）：workspace 臂 **105,057** 参数；baseline 臂 **65,073** 参数（差 39,984 = 内容寻址路径）；两臂共享张量初值相同（按名确定性初始化）。本比较因此不是等参数比较，**差异必须与参数差一同读**；更宽的 baseline 对等实验属后续（若主门为负）。

## §3 评估与指标（全部只读，模型副本执行）

1. **M1 train-fit**：训练后 train split 的 teacher-forced 准确率（仅诊断两臂是否欠训练）。
2. **M2 dev teacher-forced 准确率**（主指标）：dev 16 episodes 的逐 byte 命中率。
3. **M3 dev exact-response 率**（次指标）：贪心生成（≤64 byte），`生成 bytes == response bytes` 且 **在 boundary 停止**才计命中。
4. **M4 boundary-stop 率**：生成在 boundary 停止的比例（边界行为不退化）。
5. **M5 病灶落差**（workspace 臂专属）：`M2(未病灶) − M2(W 清零副本)`；W 清零 = 推理前把 `workspace_key`/`workspace_value` 置零的**副本**（训练 checkpoint 永不被改）。
6. 每 shape（五类）分项披露，不单独设门（16 episodes 分辨率不足）。

## §4 门（全部门先冻结，逐 seed 判定）

| 门 | 判据 | 失败含义 |
|---|---|---|
| G-fit | 每 seed：两臂 M1 ≥ **0.70** | 欠训练 seed 剔除并披露；≥2 seed 欠训练 ⇒ 结果映射 `computation_graph_return`（不比较） |
| G-win | 每 seed：M2(workspace) **>** M2(baseline) 且三 seed 均值差 ≥ **+0.03** | 负结果 ⇒ `no_workspace_benefit`，按总账回计算图 |
| G-boundary | 每 seed：M4(workspace) ≥ M4(baseline) − **0.05** | 边界退化 ⇒ 该 seed 判不通过 |
| G-lesion | 每 seed：M5 **> 0** | 工作空间非承重 ⇒ `workspace_unused` 发现（接线级否证，与能力无关） |
| G-regression | 冻结 revision 上五门实现门套件仍全绿（scoped） | 工程回归 ⇒ 停止并修复 |

**结果映射**：五门全过 ⇒ `workspace_benefit_supported`（dev 级、逐 seed、含参数差分账）；G-win 不过 ⇒ `no_workspace_benefit`；G-lesion 不过 ⇒ `workspace_unused`；G-fit 系统性不过 ⇒ `computation_graph_return`。

## §5 对照与后续（合同 §5.2）

- **workspace lesion**（M5）与**关闭 workspace 训练信用的重训对照**分别命名、分别记账；重训对照**仅在 G-win 通过后**启动（先存在真实收益，再判断撤销收益），其预算届时另行冻结。
- 未见实体/措辞的 dev 是本比较的核心（数据合同已保证实体池、模板措辞、(模板,实体) 组合三向不相交）。
- final 保持未读；本包任何结论不得表述为 L2 或整模型能力。

## §6 产物

`scripts/training/eval_taiji_r2_h3_8_matched_dev.py` → `reports/r2_h3_8_matched_dev_20260917.json`（逐 seed × 臂 × 指标 + 门判定 + 参数差分账 + 停止原因）→ roadmap 更新 + 独立提交。

## §7 修订（2026-09-17，v2 单一前缀通道重跑——承载实现合同 §7 的 dev 侧同步）

v1 执行落 `workspace_unused`（`55f7bf62`）。实现合同 [§7 修订](M5_R2_H3_8_ISOLATED_PROTOTYPE_CONTRACT_20260917.md)冻结了图变更（workspace 臂 prefix 唯一路径 = W 读；`start_vector` 替代 `renderer_start`；checkpoint 版本 1→2；基线臂 `renderer_width` 96 参数对齐 103,601 vs 101,025）。本节同步 dev 侧条款：

1. **§2/§3/§4 全部不变**：三 seed、LR 0.01、30 epochs、batch 1、门与阈值（fit 0.70 / win +0.03 / boundary −0.05 / lesion > 0 / 实现门仍绿）、指标 M1–M5 原样沿用——**修订的是图，不是判据**。
2. **执行前新增硬前置（合同 §7.2 结构门）**：W 清零后不同 prefix 的 teacher-forced logits 逐位相等——不过此门不得跑任何 dev。
3. **产物路径**：实现门重跑覆写门报告（同文件重执行属仪器复跑，报告内记录 v2 revision 绑定）；train-only 可学习性检查落**新文件** `reports/r2_h3_8_train_learnability_v2_20260917.json`（v2 图下两臂须仍可学，否则按 §4.3 停止线回图、不跑 dev）；matched dev 落**新文件** `reports/r2_h3_8_matched_dev_v2_20260917.json`。v1 各报告原样保留。
4. **判读收紧（预承诺）**：同一 dev 已被 v1 评分一次，v2 的 G-win 即使通过也**只作工程证据**（机制修复有效），prospective 泛化主张仍留给后续独立 final/新语料评审；G-lesion 不过 ⇒ 唯一通道图中无旁路可解释，按合同 §7.6 停止该架构族追加、回候选重议。
5. **closure-retrain 对照条件不变**：仅 v2 G-win 通过后另行冻结预算。