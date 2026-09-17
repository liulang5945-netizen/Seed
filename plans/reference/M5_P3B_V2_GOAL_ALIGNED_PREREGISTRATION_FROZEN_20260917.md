# M5 P3b-v2 预注册（**冻结**）：目标对齐的语言训练

日期：2026-09-17。状态：**冻结**（用户 2026-09-17 授权执行训练）。
取代 [P3b 草案](M5_P3B_ALIGNED_LANGUAGE_TRAINING_PREREGISTRATION_20260915.md) 的「数据侧重」假设。

依据：[P1 诊断](M5_CAP0_P1_LANGUAGE_SUPERVISION_DIAGNOSIS_20260915.md)（训练目标是纯字节自监督）、
[P3a](M5_CAP0_P1_LANGUAGE_SUPERVISION_DIAGNOSIS_20260915.md) §9/§10（编码层可修，语义层未动）、
`reports/r2_h3_7b_validation/`（prior/credit 一致性修复通过工程验证）、
H3.8 结项（机制族按预承诺停止投入）。

## §1 为什么假设要从「数据侧重」升级为「目标对齐」

P3b 草案假设缺口来自**数据分布**（对话密度低）。H3.3→H3.8 的结项证据否证了它：

- **机制族全部落空**：条件接口 / 计划槽 / target encoder / 4-slot workspace / 单一前缀通道
  ⇒ 两次 matched dev 均 `workspace_unused`，lesion 非全正 ⇒ 归因升级到 **W-generation 路径本身**；
- **P1 证明**：训练目标是**纯字节自监督**（唯一信号 `observe(symbol, learn=True)`），
  目标里**根本没有"回答"**；
- **P3a 证明**：推理侧约束能把可读性从 `0/4` 修到 `4/4`，但 **C/D/E 仍是 `0 / 0.0625 / 0.15`**
  ⇒ **编码层不是瓶颈**。

**假设 H（P3b-v2）**：缺口在**目标** —— 训练不区分"回答"与"续写"。
把 response-only 信用接成**条件化目标**（给定 `context + user_input` ⇒ 产出 `response`），
在 H3.7B 修复的一致性基础上能改善 C/D/E。

**反假设**：若在对齐目标上仍无改善 ⇒ 缺口在**架构层**（序列建模能力）。
本工作包**结项并归因架构**，不再追加机制或同质训练。

## §2 协议（冻结）

| 项 | 取值 |
|---|---|
| **入口** | `scripts/training/train_taiji_r2_aligned.py`（自带 `checkpoint_roundtrip_preflight` 零步前置） |
| **起点** | `reports/r2_h3_7b_validation/byte_aligned/checkpoint.pt`（H3.7B 已通过工程验证的态，含 byte-aligned target + 0.75/0.25 混合信用） |
| **数据** | **结构化 `LanguageEpisode` jsonl**（8 键：`episode_id`/`family_id`/`task_family`/`split`/`context`/`user_input`/`response`/`required_terms`）。基准为 `tests/fixtures/r2_h3_5a_response_plan_v3.jsonl`（24 行）；**P3b-v2 的差异化是放大规模**：用模板 × 实体池合成 train/dev/final 三分离集，规则与产物 sha256 随报告归档 |
| **目标** | response-only 条件化信用（沿用 H3.7B 的 0.75/0.25 混合与 byte-aligned 16-byte 窗口），**不改架构** |
| **推理链路** | **必须与 P3a 一致**：进程内放宽 legacy 守卫 + UTF-8 约束解码（否则与基线不可比） |
| **评测** | 每个检查点在**冻结评价集 v1** 上评测；`trained_during_eval=false`；原始回答留档 |
| **预算** | **先 pilot**（沿用 H3.7B 规模：2 epochs × 12 train episodes，CPU 单线程，300k 参数上限，wall cap 30 min）；**pilot 有改善才放大**，放大档由 pilot 实测吞吐外推后冻结 |

## §3 判据（可机检，冻结）

沿用 P3b 的 J1–J5，**新增 J6**：

| # | 判据 |
|---|---|
| **J1** | 前后对照在**同一链路**上完成（报告 `chain` 一致） |
| **J2** | C / D / E **三项均严格高于** P3a 基线（`0.00` / `0.0625` / `0.15`） |
| **J3** | 达标线：**C/E ≥ 70%、D ≥ 80%**（07 §4.2），且 B 经人工复核达标 |
| **J4** | 零回归：G 硬安全失败仍为 **0**；A / H 不退化 |
| **J5** | 报告显式区分机检判分与人工复核；未测维度记 `untested` |
| **J6（新）** | **dev `exact_response` 率 > 0**。H3.x 全系列的 exact 恒为 0 ⇒ 这是"目标对齐是否真的发生"的**直接证据**，比 surprise 更能证伪 |

## §4 停止线（任一触发即停止，不扩预算）

1. **非有限值** / 参数预算超限 / 更新预算不匹配；
2. **J6 未达成**（dev exact 仍为 0）⇒ 目标对齐未发生，按**反假设**结项；
3. 连续 **3** 个检查点 C/D/E 无改善 ⇒ 停，归因架构层；
4. 出现**退化**（B/G 或 A/H 退步）⇒ 停并回滚到起点 checkpoint；
5. wall cap 超限。

## §5 明确不做

- **不**下线最低线、**不**换模型/换入口凑分、**不**把 E/T 模式算作 N 能力；
- **不**在看完成绩后调整判据（07 §4.2）；
- **不**修改冻结评价集、**不**改加载器与 binder（WP-6）；
- **不**继续加机制（H3.8 架构族已按预承诺停止投入）。

## §6 待确认项（执行前必须补齐，否则不启动）

1. **数据合成脚本与 sha256**（模板 × 实体池，train/dev/final 三分离 + 对抗形状）；
2. ✅ **起点 checkpoint 已确认**：`reports/r2_h3_7b_validation/byte_aligned/checkpoint.pt`，
   **16,502,913 B，sha256 `285d4b23a88bc458e486201a382a9e66b3551a1ff29f077f6c1f6368defa696f`**；
   其报告 `status=passed_engineering_only`，并记录 `corpus_digest` / `code_revision` /
   `zero_fresh_process` / `child_fresh_process` / `factorized_preflight` 全绿
   ⇒ 作为 P3b-v2 的起点满足"已通过工程验证"的前置；
3. **放大档预算**（pilot 实测吞吐 × 期望机时）。

> **冻结含义**：§2/§3/§4 的内容在 pilot 启动后**不得修改**。
> 任何调整都必须写**新预注册**（本文件只追加后续章节）。

## §7 执行进展（**只追加**）

### §7.1 数据侧完成（2026-09-17）

生成器：`scripts/training/build_taiji_r2_p3b_v2_corpus.py`（沿用 H3.8 数据合同的分离规则）；
产物：`tests/fixtures/r2_p3b_v2_goal_aligned_episodes_v1.jsonl`；
报告：`reports/r2_p3b_v2_corpus_20260917.json`。

| 项 | 值 |
|---|---|
| 规模 | **80 行** = train 40 / dev 20 / final 20（**3.3 ×** H3.5a 的 24 行，满足"放大规模"） |
| 五对抗形状 | 每个 split 都齐备：fact replacement / negation / unknown / combination / same-opening |
| `corpus_digest` | `4d19974783cbd093a968a8e9fb1a4e945e060843a8c34f3bc1c571fce0311f2b` |
| 生成器自检 | **7 项全过**：五形状齐备 · 实体池互斥 · 提问 split 内唯一 · 提问跨 split 互斥 · 组合跨 split 互斥 · `required_terms` 均在 response 内 · train 无 holdout 泄漏 |
| 加载实测 | `LanguageEpisodeCorpus.from_jsonl` 成功（manifest `format=taiji-native-language-alignment-v2`、`serialization=r2-conditional-response-v2`、`digest=5fcebf34…`） |

**生成期修掉的三处自检问题**（记录以备复核）：
1. `unknown` 形状原先带 `required_terms=[value]`，但它的 response 是"没有信息" ⇒
   `required_terms` 改为空（诚实弃答不应被要求命中实体）；
2. 实体池原用 **笛卡尔积**，导致同一实体多值 ⇒ 同 split 内提问重复 ⇒ 改为**一一配对**；
3. dev/final 的 `combination` 模板 `user_input` 漏了 `{item}` ⇒ 4 个实体产生相同提问 ⇒ 补上。

**注**：`episode_id` / `family_id` 必须是 ASCII 稳定标识符（`LanguageEpisode` 强制），
故用 `<split>-<shape>-<index>`，中文实体只出现在文本槽里。

### §7.2 剩余阻塞项

- **放大档预算**（§6 第 3 项）：待 pilot 实测吞吐外推。
- pilot 可立即启动（§2 的入口与起点已确认）。

### §7.3 **起点修正**：语料绑定使「继承 H3.7B 权重」与「放大语料规模」不可兼得（2026-09-17）

**首次 pilot 失败**：`ValueError: language alignment checkpoint corpus digest mismatch`。

**根因（设计如此，非缺陷）**：`LanguageAlignmentTrainer.from_checkpoint` 校验
`corpus_digest` 绑定 —— H3.7B 的 `byte_aligned/checkpoint.pt` 记录
`corpus_digest = 0bc5b5540d4b622f907942f42d7b809e825932e50c6e6c505a3cd091edabd691`，
**正是 `tests/fixtures/r2_h3_5a_response_plan_v3.jsonl`（24 行）**；
而 P3b-v2 的语料（80 行）digest 不同 ⇒ 无法 `--resume`。

**⇒ §2 的"起点"条款按下列方式修正（只追加，不改 §2 原文）**：

P3b-v2 采用 **从零构建模型 + P3b-v2 语料**（`--response-plan-*` 显式复刻 H3.7B 的
factorized 配置），**不使用 `--resume`**。理由：

1. **H3.7B 的 child 只训了 1 epoch × 12 episodes（`global_step 537`）** ⇒ 权重增量很小，
   继承价值有限；
2. **H3.7B 的核心修复是代码级的**（prior + 0.75/0.25 混合信用，位于
   `LanguageAlignmentTrainer`）⇒ **从零构建同样生效**，不依赖权重继承；
3. **P3b-v2 的假设就是"目标对齐 + 放大规模"** ⇒ 必须换语料，而换语料按设计就不能 resume；
4. 从零构建**更干净**（无 H3.5a 24 行语料的历史污染）。

**对 J1 的影响**：J1 要求"前后对照同链路"。P3a 基线（`constrained_decode`）与
P3b-v2 的评测仍需同链路；但**训练起点**从"H3.7B child"改为"从零构建"这件事，
**必须在报告里显式披露**，不得表述为"在 H3.7B 之上继续"。

### §7.4 pilot 结果（**按 §4 停止线第 2 条结项**，2026-09-17）

产物 `reports/p3b_v2_pilot_20260917.json`（2 epochs × 12 episodes，`--defer-final`）：

| dev 指标（20 episodes） | baseline（零步） | child（训练后） | 变化 |
|---|---|---|---|
| `teacher_forced_accuracy` | 0.0000 | **0.1969** | **+0.1969** |
| `teacher_forced_mean_surprise` | 5.5984 | **4.5277** | **−1.0707** |
| **`exact_response`** | **0/20** | **0/20** | **0** |

⇒ **J6 未达成** ⇒ 按 §4 第 2 条 **结项并归因架构层**；
结论与界限详见 [pilot 结项报告](M5_P3B_V2_PILOT_CLOSURE_20260917.md)。

**§2 / §3 / §4 的冻结条款未被修改**；本 pilot **不产生** L2 晋级或能力收益主张；
final 未读；起点为从零构建（§7.3）。

### §7.4 用户要求收束：中断登记（2026-09-17，只追加）

本次停止已核实的P3b-v2进程25624，未删除磁盘产物，未改J1–J6或停止线。现有checkpoint早于该进程启动，payload记录24episodes/1080byte更新，但目标为旧`h3_7_factorized_response_chunks`，与byte-aligned文字约定存在偏差；无完整pilot结果报告，不能据此判成功或判反假设成立。状态为“中断、未判定”，不追加训练、不重启。身份摘要、当前语料digest、checkpoint SHA256及唯一后续见[项目收束记录](PROJECT_CONSOLIDATION_20260917.md)及[机器记录](../../reports/project_consolidation_20260917.json)。本节取代§7.2的“pilot可立即启动”导航作用，不改写冻结协议。
