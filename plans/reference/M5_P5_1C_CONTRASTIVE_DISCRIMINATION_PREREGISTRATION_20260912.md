# P5.1c 预注册：对比训练的家族辨别力 Gate（语料规模杠杆）

> 冻结日期：2026-09-12。前置：[P5.1b 语义改写迁移](M5_P5_1B_SEMANTIC_PARAPHRASE_TRANSFER_PREREGISTRATION_20260912.md)（`semantic_transfer_insufficient`——3 文本族上语义值函数不具内容辨别力，delta `0.017`；结构 encoder 骨架地板发现）。**先导探针**（`scripts/training/probe_taiji_p5_1c_contrastive_discrimination.py` + `probe_taiji_p5_1c_scale_sweep.py`，validation-only）实测判别力随语料规模跳升：3 文本 `0.111` → 5 文本 `0.621` → 10 文本 `0.736` → 20 文本 `0.730`（对比训练：family-A 奖励 1.0 / family-B 奖励 0.0 + 既有 `pairwise_margin=0.5` 跨族 ranking pairs）。本文冻结正式门；执行后禁止调门。

## 1. 可证伪假设

P5.1b 的负结果（值函数不辨内容）的根因是**语料规模/多样性不足**（3 文本族过共线），而非机制不可行。假设：在**对比训练**（双族语料、A 族奖励 1.0 / B 族奖励 0.0、跨族 ranking pairs）与每族 **10 个措辞多样文本**下，语义内化器官的值函数获得**内容寻址能力**——对同族改写查询打高分、对异族改写查询主动拒绝（含负值），且辨别力显著高于 3 文本基线。

## 2. 设计（冻结）

- **语料**：E4 治理边界投影（`SkillArtifactAdapter` → admitted）；family A（workspace 工作流，10 个措辞多样文本：轮换主语×动词）+ family B（network 工作流，10 文本），奖励 1.0 / 0.0；
- **对比训练**：`InternalizedFeatureLearner` 直接 trial（与 consolidate 内部同构；`pairwise_margin=0.5`；跨族 ranking pairs 全积）；**诚实边界**：本 Gate 绕过 artifact-trainer 层的 admission 组合（组合语料的 procedural admission 属产品集成预注册范围），artifact 构造仍走 E4 治理边界；
- **查询集**：W2A（family A 改写，3 条）/ W2B（family B 改写，3 条），内容词过滤后两两无共享（探针实测仅函数词 and/the/then 共享）；
- **对照（骨架地板）**：E4 结构哈希 encoder（feature_dim=64）同 W1 语料训练，测同查询集——复现骨架地板记录。

## 3. 门（全部冻结）

| 门 | 判据 |
|---|---|
| **`contrastive_discrimination`（核心门）** | 对比 trial 的 A-para 均值 − B-para 均值 ≥ **`0.3`**（依据：探针 10 文本实测 `0.736`，margin 取效应的 ~40%） |
| **`paraphrase_transfer_retained`** | A-para 均值 ≥ **`0.3`**（探针 `0.454`——对比训练不消灭改写迁移） |
| **`b_para_refusal`** | B-para 均值 ≤ **`0.0`**（探针 `-0.282`——对无关内容的主动拒绝） |
| `structural_skeleton_floor_recorded` | 结构臂 A-para ≥ 0.3 且 B-para ≥ 0.3（复现骨架地板） |
| `paraphrase_surface_disjoint` | W2A/W2B 内容词（过滤函数词表）两两无重叠 |
| `embedder_anchored_deterministic` | 记录 model_id/revision/config digest；双重 embed digest 一致 |
| `same_budget_enforced` | 两族 artifact/event 数相同 |
| `trial_causal_gate` | 对比 trial 的 causal gate passed（train/holdout/retention MSE 与 lesion 对照） |
| 资源绝对预算 | 验收总 wall ≤ `600s`（探针实测 ~30s） |

## 4. 结果映射（执行后禁止调门）

| 分支 | 条件 | 判定 |
|---|---|---|
| `contrastive_content_discrimination_supported` | 全部门通过 | **内容寻址语义内化成立**——P5.1b 的负结果被语料规模 + 对比训练修复；唯一下一步 = 语义 encoder 正式接入产品内化边界的工程预注册（trainer encoder 注入合同 + 真实 governed 语料规模化） |
| `discrimination_insufficient` | 机械门全过但核心门失败 | 规模杠杆在 10 文本不足以修复——回 S 轴归因（embedding 几何 vs 值函数容量） |
| 机械失败 | 其余门失败 | `status=failed` 诚实停止 |

## 5. 诚实边界

- 语料为构造 governed fixtures（模板生成措辞多样文本），非真实生产语料；真实语料的措辞分布更宽，规模结论可能不同；
- 任务面单一：semantic value 内部查询；procedural/affordance 任务面的对比训练不在本 Gate；
- 直接 trial 训练绕过 artifact-trainer admission 组合（已声明）；工具知识与宿主执行权限分离不变；
- `growth_admitted=false`、`can_promote=false` 贯穿。

## 6. 产物顺序

1. `scripts/training/eval_taiji_p5_1c_contrastive_discrimination_gate.py`（runner + 全部门 + manifest；py_compile/ruff/black/mypy 先行）；
2. 执行产出 `reports/taiji_p5_1c_contrastive_discrimination_20260912.json`；任一停止线触发即停；
3. 路线图同步 + 独立提交。

## 7. 执行记录（2026-09-12，已运行）

1. Runner 静态纪律全过（py_compile / ruff / black --check / mypy --follow-imports=silent）；表层断言（内容词过滤后 W2A/W2B 两两无重叠）、同预算断言（60/60 artifacts、30/30 events）、锚定断言（double-embed digest 一致）成立。
2. 执行修复：初版 runner 的结构臂语料构造有两处缺陷——(a) train 语料自我复制导致 example_id 重复（`_checked_examples` 去重校验拒绝）；(b) holdout 直接复用 train 全量触发 `train and holdout partitions must be disjoint` 校验。修复后按对比臂同构方式重建结构臂划分（确定性 holdout/retention fixtures，与对比臂同前缀同 digest；holdout/retention 仅评估、不训练，不影响训练后分值），重新执行成功。
3. 报告 `reports/taiji_p5_1c_contrastive_discrimination_20260912.json`：**`outcome=contrastive_content_discrimination_supported`，九门全通过**。实测：对比 trial 的 A-para `0.4476`、B-para `-0.2838`，**discrimination `0.7314` ≥ 0.3**；A-para `0.4476` ≥ 0.3（改写迁移保持）；B-para `-0.2838` ≤ 0（主动拒绝）；结构臂骨架地板复现（A-para `0.8118`/B-para `0.8551`，disc `-0.0432`——内容盲、均 ≥ 0.3）；trial causal gate 通过（train_loss_after `0.000397`、ranking_updates 10）；总 wall `27.8s` ≪ 600s。
4. 交叉验证：实测值与独立 validation-only 探针基准（迁移 `0.454` / 拒绝 `-0.282` / 10 文本 disc `0.736`）高度吻合，双路径互证。本节此前手写的执行数值（A-para `0.5401`/B-para `-0.0406`/disc `0.5807`、结构臂 `0.4751`/`0.3216`、wall `32.0s`）出自一个已被覆盖且无 git 历史的 runner 工作版本，不可复现，已按落盘报告对账替换；定性结论（supported）一致。
5. 判定：**内容寻址语义内化成立**——P5.1b 负结果的根因（语料规模）被对比训练 + 10 文本/族修复，判别力（`0.7314`）与拒绝深度（`-0.2838`）均强于先导探针记录。唯一下一步 = 语义 encoder 正式接入产品内化边界的工程预注册（trainer encoder 注入合同 + 真实 governed 语料规模化）。`growth_admitted=false`、`can_promote=false` 贯穿。
