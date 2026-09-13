# P5.1 预注册：有来源语料的内容因果迁移 Gate（同预算 placebo 对照）

> 冻结日期：2026-09-12。前置：[晋级宣布](M5_K_PROMOTION_DECLARATION_20260912.md)（M5 K 轴已晋级，P5.1 解冻）、E4 artifact internalization Gate（`taiji_w7_e4_artifact_internalization_20260901.json` 全 10 检查通过）、S3 语义 embedding 内化（S 轴）。**先导探针**（`scripts/training/probe_taiji_p5_1_sourced_content_transfer.py`，validation-only、无门）已实测数值依据：sourced holdout `0.5` > 其 lesion `0.333`，placebo holdout `0.0`，delta `0.5`。本文冻结正式门；执行后禁止调门。

## 1. 可证伪假设

E4 已证明内化机械成立（holdout beat lesion、retention、执行分离），但未证明**语料内容本身的因果收益**。本 Gate 的假设：在 E4 治理边界与同预算条件下，内化**内容与评估任务同一 workflow 族**的有来源语料，相比**同预算、同结构、同流程但内容无关**的 placebo 语料，在**全新 unseen 任务**（holdout partition，过程下一动作预测）上产生可测收益——即收益可归因于**语料内容**而非内化行为或预算本身。

## 2. 两臂设计（单变量 = 语料内容族；预算逐项相同）

- **载体**：E4 真实治理边界——`SkillArtifactAdapter` 投影 → `admitted` 状态 → `ArtifactInternalizationTrainer`（两臂配置逐位相同：`feature_dim=64`、`procedural_hidden_dim=16`、`seed=17`、`semantic_passes=12`、`procedural_epochs=250`、`affordance_epochs=200`）；
- **sourced 臂**：3 个 train skill artifacts，workflow 族 A（capability 词表 `editor.open/editor.read/editor.inspect` 的旋转链），与 holdout 任务同族；
- **placebo 臂**：3 个 train skill artifacts，workflow 族 B（`network.search/index.scan/cache.fetch`），与 holdout 任务零 capability 词表重叠，其余构造逐位同构（同 target 结构、同 event 数、同 partition 流程）；
- **共享评估集**：holdout（2 个族 A 新 scope artifacts + events）、retention（2 个族 A 新 scope artifacts + events）——两臂评估同一 holdout/retention；
- **词表断言（冻结）**：placebo capability 词表 ∩ holdout capability 词表 = ∅；sourced capability 词表 ⊇ holdout capability 词表；结构字段 token（source_kind 等）的必然共享不计入。

## 3. 门（全部冻结）

| 门 | 判据 |
|---|---|
| `same_budget_enforced` | 两臂 artifact 数、event 数、unit_kind 组成、trainer 配置逐项相同（构造性断言） |
| `capability_vocabulary_disjoint` | §2 词表断言成立 |
| `sourced_transfer_beats_lesion` | sourced holdout accuracy > sourced lesion holdout accuracy（臂内内化必要性） |
| **`content_transfer_margin`（核心门）** | sourced holdout − placebo holdout ≥ **`0.15`**（依据：探针实测 delta `0.5`、placebo `0.0`；0.15 为实测效应的 30%，远高于构造噪声） |
| `checkpoint_roundtrip_both_arms` | 两臂 trainer checkpoint 往返 digest 一致 |
| `quarantined_artifact_rejected` | quarantine 状态 artifact 被 consolidate 拒绝 |
| `semantic_internalized_both_arms` | 两臂 semantic 内化 passed |
| 资源绝对预算 | 验收总 wall ≤ `300s`（探针实测秒级） |

## 4. 结果映射（执行后禁止调门）

| 分支 | 条件 | 判定 |
|---|---|---|
| `sourced_knowledge_transfer_supported` | 全部门通过 | **语料内容因果收益成立**——P5.1 假设成立；唯一下一步 = P5.1 规模化预注册（真实 governed 语料、更大任务面、S3 语义 embedding 管线接入） |
| `sourced_content_insufficient` | 机械门全过但 `content_transfer_margin` 失败 | 知识来源假设被否（内化行为收益 ≠ 内容收益）——回 S 轴归因，不进入规模化 |
| 机械失败 | checkpoint/quarantine/预算失败 | `status=failed` 诚实停止 |

## 5. 诚实边界

- 本 Gate 的泛化 = **特征哈希编码下的结构/词表重叠泛化**，不是语义改写泛化（S3 语义 embedding 管线是独立后续，不属本 Gate）；
- 任务面单一：procedural next-action 预测；不含语义/affordance 任务面的因果对照；
- 语料为构造的 governed fixtures（走真实 E4 治理边界与投影适配器），不是生产语料；
- 工具知识与宿主执行权限分离沿用 E4 合同（`client_execution_not_internalized`）；本 Gate 不授予任何执行权限；
- `fit_called` 语义沿用内化 trainer（consolidate 即训练）；`growth_admitted=false`、`can_promote=false` 贯穿（K 轴晋级宣布不因本 Gate 扩大范围）。

## 6. 产物顺序

1. `scripts/training/eval_taiji_p5_1_sourced_knowledge_transfer.py`（两臂 runner + 全部门 + manifest；py_compile/ruff/black/mypy 先行）；
2. 执行产出 `reports/taiji_p5_1_sourced_knowledge_transfer_20260912.json`；任一停止线触发即停；
3. 路线图同步 + 独立提交。

## 7. 执行记录（2026-09-12，已运行）

1. Runner 静态纪律全过（py_compile/ruff/black/mypy）；预算逐项断言、capability 词表断言（holdout 词表 = {editor.open, editor.read, editor.inspect}，sourced ⊇ holdout、placebo ∩ holdout = ∅）成立。
2. 报告 `reports/taiji_p5_1_sourced_knowledge_transfer_20260912.json`：**`outcome=sourced_knowledge_transfer_supported`，全门通过**。实测：sourced holdout `0.5` > lesion `0.333`；placebo holdout `0.0`（= 其 lesion）；**content delta `0.5` ≥ 0.15**；两臂 checkpoint 往返一致；quarantine 拒绝；semantic 内化两臂通过；总 wall `0.83s` ≪ 300s。
3. 判定：**语料内容因果收益成立**——同预算下收益归因于来源内容而非内化行为本身。唯一下一步 = P5.1 规模化预注册。`growth_admitted=false`、`can_promote=false` 贯穿。
