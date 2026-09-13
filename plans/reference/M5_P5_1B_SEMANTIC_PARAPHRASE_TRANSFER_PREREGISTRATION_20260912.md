# P5.1b 预注册：语义改写迁移 Gate（语义 encoder 接入 + 骨架地板对照）

> 冻结日期：2026-09-12。前置：[P5.1 内容迁移 Gate](M5_P5_1_SOURCED_KNOWLEDGE_TRANSFER_PREREGISTRATION_20260912.md)（`sourced_knowledge_transfer_supported`——词表重叠泛化成立）。本 Gate 把泛化要求从「词表重叠」升级为「**语义改写**」（查询与语料零内容词重叠），接入 S3 已锚定的语义 embedding（`DocumentEmbedder`，MiniLM 384 维）。**先导探针**（`scripts/training/probe_taiji_p5_1b_semantic_paraphrase_transfer.py`，validation-only）已实测：sourced trial A 族改写值 `0.603`（W1 同词参照 `1.003`）、placebo 训练 trial A 族改写值 `0.0`、结构 encoder 对无关内容改写查询打 `0.790`（骨架地板）。本文冻结正式门；执行后禁止调门。

## 1. 可证伪假设

语义 embedding 内化器官在「同预算、同查询集、都训练到自身 train MSE→0」的两臂下：**sourced 臂（与查询语义同族）对改写查询的内部值显著高于 placebo 臂**——即语义器官的值迁移由**内容语义**驱动而非表层 token 或骨架；同时结构哈希 encoder（E4 现用）对同骨架的无关内容查询也打高分（骨架地板），证明其不具备内容寻址能力——语义 encoder 的接入必要性由此成立。

## 2. 三臂设计（同预算；查询集相同）

- **语料**：E4 治理边界（`SkillArtifactAdapter` → admitted）；W1 = family A 语料（workspace 工作流，W1 表层词）；placebo = family B 语料（network 工作流）；各 3 个 skill（18 artifacts / 9 events）；holdout/retention 分区独立构造（family A）；
- **semantic-sourced 臂**：trainer（`feature_dim=384`）+ 语义 encoder swap（`SemanticArtifactKnowledgeEncoder`：渲染 name+description+steps 文本 → `DocumentEmbedder.embed`，渲染排除 scope_id 身份字段）→ consolidate（admission 照常）；
- **semantic-placebo-trained 臂**：同配置 family-B 语料；其 admission 大概率回滚（探针实测 rolled_back）——内容因果对照使用其**训练 trial**（与 consolidate 内部同构：`InternalizedFeatureLearner.from_checkpoint` + 同序 consolidate），即「都训练到自身 train MSE→0」的纯内容对照；
- **structural-sourced 臂**：E4 原生结构哈希 encoder（`feature_dim=64`）+ 同 W1 语料（encoder 必要性对照）；
- **查询集**：W2A = family A 工作流的改写描述（3 条）；W2B = family B 工作流的改写描述（3 条）；**表层断言（冻结）**：W1∩W2A∩W2B 内容词重叠 = ∅（探针实测 shared = []）；
- **查询特征**：语义臂 = `embed([text])`（文本查询，无 artifact 包装——内部查询不接受外部 artifact 的 E4 纪律不变）；结构臂 = 手工构造的 text-only knowledge unit 哈希（无 steps——不泄漏 capability 标识符）。

## 3. 门（全部冻结）

| 门 | 判据 |
|---|---|
| `paraphrase_surface_disjoint` | W1/W2A/W2B 内容词两两重叠 = ∅（构造性断言） |
| `embedder_anchored_deterministic` | 记录 model_id/revision/config digest；同文本两次 embed 的 digest 一致 |
| `same_budget_enforced` | 三臂 artifact/event 数与 trainer 配置逐项相同 |
| **`semantic_paraphrase_transfer`（核心门）** | sourced trial 的 A 族改写均值 ≥ **`0.3`**（依据：探针实测 `0.603`，margin 取效应的 ~50%，同时高于结构臂数值噪声） |
| **`semantic_content_causality`（核心门）** | sourced A-para − placebo-trained A-para ≥ **`0.3`**（探针实测 `0.603 − 0.0`；同查询/同预算/都训到 train MSE→0 的纯内容对照） |
| `structural_skeleton_floor_recorded` | 结构臂 A-para ≥ 0.3 且 B-para ≥ 0.3（对无关内容也打高分——骨架地板复现，作为 E4 `internal_value` 检查的解释细化记录；该门失败则修订对 E4 的解释而非调门） |
| `checkpoint_roundtrip_both_semantic_arms` | stub-encoder 基底恢复 + 语义 encoder swap 后 checkpoint digest 逐位一致（恢复训练态） |
| `quarantined_artifact_rejected` | quarantine artifact 被 consolidate 拒绝 |
| `semantic_internalized_both_arms` | 两臂 admission 内的 semantic causal gate passed（回滚臂记录 `rolled_back=true`） |
| 资源绝对预算 | 验收总 wall ≤ `600s`（含 embedder 加载；探针实测含加载 ~30s） |

## 4. 结果映射（执行后禁止调门）

| 分支 | 条件 | 判定 |
|---|---|---|
| `semantic_paraphrase_transfer_supported` | 全部门通过 | **语义改写迁移成立**——内容因果收益在零表层重叠下保持；骨架地板发现入账；唯一下一步 = 语义 encoder 正式接入产品内化边界的工程预注册（trainer 注入合同）+ 更大更多样语料的规模化 |
| `semantic_transfer_insufficient` | 机械门全过但两个核心门任一失败 | 语义改写迁移不成立（探针效应不复现或属几何偶然）——如实入账，回 S 轴归因；骨架地板发现仍有效 |
| 机械失败 | 其余门失败 | `status=failed` 诚实停止 |

## 5. 诚实边界

- 探针发现**改写迁移是部分的**（0.603 < W1 同词参照 1.003）且**几何依赖**（placebo-trained 对 B 族改写迁移为 0——3 文本语料过小/过共线）；本 Gate 只断言 sourced-vs-placebo 的同查询因果差，不断言迁移饱和；
- 结构臂的地板发现（0.873/0.790）是对 **E4 `internal_value` 检查的解释细化**（其高分部分来自骨架 token而非内容寻址），不改变 E4 其余 9 项检查的效力；
- 任务面单一：semantic value 内部查询；语料为构造 governed fixtures；工具知识与宿主执行权限分离不变；
- `growth_admitted=false`、`can_promote=false` 贯穿（K 轴晋级宣布范围不因此扩大）。

## 6. 产物顺序

1. `scripts/training/eval_taiji_p5_1b_semantic_paraphrase_transfer.py`（三臂 runner + 全部门 + manifest；py_compile/ruff/black/mypy 先行）；
2. 执行产出 `reports/taiji_p5_1b_semantic_paraphrase_transfer_20260912.json`；任一停止线触发即停；
3. 路线图同步 + 独立提交。

## 7. 执行记录（2026-09-12，已运行——核心门诚实失败）

1. Runner 静态纪律全过（py_compile/ruff/black/mypy）；表层断言按冻结口径实现（三元交集 W1∩W2A∩W2B 内容词 = ∅，实测成立；W2A/W2B 两两共享词仅作记录：inspect/bounded/stored/end 与 limited/operations）。执行过程中修正两处 runner 实现错误（surface 门误用全 alpha token 与误把记录列表当布尔），均属实现侧、不触碰冻结判据。

2. 报告 `reports/taiji_p5_1b_semantic_paraphrase_transfer_20260912.json`：**`outcome=semantic_transfer_insufficient`——核心门诚实失败**（`semantic_content_causality` delta `0.0167` < 冻结 `0.3`）。实测细节：
   - semantic-sourced：A 族改写 `0.6026`（W1 同词参照 `1.0029`——部分迁移，非饱和）、B 族改写 `0.5190`；
   - **placebo 训练 trial 的 A 族改写值 `0.5859`**——与 sourced 臂几乎相同；即语义值函数在 3 文本语料规模上学到的是「工作流风格文本 → 高值」的风格/骨架泛化，而非内容寻址（家族辨别力 A−B 仅 `0.084`，与结构臂的 `0.0835` 相当）；
   - **治理管线 admission 内容敏感**：placebo 臂因 procedural 迁移失败被原子回滚（`rolled_back=true`）——probe 早期的「placebo 全 0」实为回滚伪象；本 Gate 用其训练 trial 完成了纯内容对照并给出诚实负结果；
   - 结构骨架地板复现：结构臂 A-para `0.8733` / B-para `0.7898`（对无关内容同打高分）——对 E4 `internal_value` 检查的解释细化（其高分部分来自骨架 token，非内容寻址）；
   - 机械门全过（锚定/确定性、同预算、checkpoint 往返两臂一致、quarantine 拒绝、两 trial causal gate 通过）；总 wall `12.2s` ≪ 600s。

3. 判定：**语义改写迁移在本尺度不成立**——值函数的改写泛化不区分内容族。负结果如实入账，直接塑造下一步设计：内容寻址需要**内容辨别性训练信号**——内化学习器已有 `pairwise_margin`/ranking 更新机制（家族间成对偏好），是 P5.1c 的第一设计杠杆；同时语料需更大更多样（3 文本族过小过共线）。`growth_admitted=false`、`can_promote=false` 贯穿。唯一下一步 = **P5.1c 预注册（成对对比训练：家族辨别力 Gate）**，先探针后冻结。
