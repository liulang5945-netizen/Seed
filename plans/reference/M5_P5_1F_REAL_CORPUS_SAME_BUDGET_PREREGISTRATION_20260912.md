# M5 · P5.1f 真实语料同预算因果收益 Gate 预注册（sourced vs placebo · 冻结 2026-09-12）

> 冻结于 P5.1e `same_budget_content_benefit_supported`（delta `0.25` ≥ `0.15`，九门全过，报告 `reports/taiji_p5_1e_same_budget_content_benefit_20260912.json`）之后。本文件把 P5.1e 的家族排他两臂设计从 governed 构造 fixtures 升级到**真实发布语料**；冻结后执行前不训练新门、不接本文件之外的新数据源、不读取 sealed。

## 1. 假设

同预算下，真实发布 agent 轨迹语料（OpenBMB UltraData-SFT-Agent-2609，2026-09-07，483,661 条）的内容族训练，能把 next-tool-call 程序能力迁移到同族 held-out 真实轨迹——即 P5.1e 在构造 fixtures 上建立的「同预算内容因果收益」（delta `0.25`）在真实分布语料上保持。frozen reference：P5.1e `delta=0.25`（fixtures）。

## 2. 两臂设计（家族排他 · 真实语料版）

- **语料源（只读；已在 repo、gitignored、不入 git）**：sourced 族 = `data/ultradata/SFT-Agent-2609/data/Tool_Use/Tool_Use_part-1-of-9.jsonl`；placebo 族 = `data/ultradata/SFT-Agent-2609/data/Code_Agent/Code_Agent_part-1-of-7.jsonl`。报告记录两文件 sha256、采样行区间、provenance（dataset 名、publisher=openbmb、HF 出处、LICENSE 文件、record `source`/`domain` 字段如 `areal_tau2`）。
- **确定性采样（冻结规则）**：part 文件内按行序取前 N 条**含 ≥1 次 tool_calls 的记录**（无调用记录跳过并计数入报告）；sourced 臂 train `200` / holdout `60` / retention `40`；placebo 臂同额 `200/60/40`；a-gate slice = Tool_Use 后续符合条件的记录取 `16`（工具 ⊆ sourced train 词表，剪枝计数如实入报告，同构 P5.1e 附录 #2）。
- **每条真实轨迹 → 声明式 manifest（governed 投影）**：`SkillArtifactAdapter`：`skill_id=uuid`、`version=数据集 tag`、`steps=[tool.<function_name>...]`（assistant tool_calls 顺序展开）、knowledge={source, domain, provenance}；`DeclarativeSourceRegistry.register` → E1 ledger projection → trainer。**「真实」指内容来源（真实发布语料、真实分布、真实共享工具词表），投影 schema 仍为 governed 声明式。**
- **单变量 = 内容族**：Tool_Use（τ2 风格客服/零售/电信/航空工具族）vs Code_Agent（apply_patch/read_file/rg/shell 等软件工程工具族）；两臂均为真实程序性语料、同构任务面（next-tool-call）、trainer 配置逐位相同；比 P5.1e 的「同为程序性 fixtures」更严格的控制。
- **a-gate 输出空间**：next tool call（`tool.<name>` token）；词表由 sourced train slice 派生并冻结；臂内语义面 = 真实轨迹 user 指令文本（口径沿 P5.1e trainer API；embedder anchor 沿 P5.1e：MiniLM-L12-v2 revision `e8f8c211226b894fcb81acc59f3b34ba3efd5f42`，double_embed true）。

## 3. 判据门表（九门，判据面与 P5.1e 逐项同构）

| # | 门 | 判据 |
|---|---|---|
| 1 | static_four_checks | ruff + `black --check` + mypy（`--follow-imports=silent seed taiji`，61 错 HEAD 基线）+ pytest（HEAD 基线 4 项既有失败如实入账、零新增），全绿 |
| 2 | same_budget_enforced | 运行时断言：两臂 artifact 数、experience 数、unit_kind 组成、ranking pairs 数逐项相同；绝对计数入报告 |
| 3 | capability_vocabulary_disjoint | sourced 词表 ⊇ a-gate 词表 且 placebo 词表 ∩ a-gate 词表 = ∅（真实词表交集非空则如实 failed，禁止剪枝凑过） |
| 4 | arm_sanity_both_arms | 每臂 `consolidate` admission passed=true（semantic + procedural holdout > lesion + retention ≥ 0.5 + affordance native < frozen，四条件全过） |
| 5 | **content_transfer_margin（核心）** | a-gate next-tool-call procedural accuracy：sourced − placebo ≥ **0.15**（P5.1 冻结常数原样继承） |
| 6 | sourced_beats_lesion | sourced 臂自身族 holdout accuracy > lesion holdout accuracy（admission 内含、显式断言） |
| 7 | affordance_content_specificity | sourced native affordance holdout MSE < frozen 且 < placebo native（关系门） |
| 8 | checkpoint_roundtrip_both_arms | 两臂 checkpoint 往返 digest 一致 |
| 9 | deterministic_and_budget | 同 seed replica 报告逐位一致；wall ≤ 1200s |

## 4. 三态映射

- `real_corpus_content_benefit_supported`：九门全过。
- `content_benefit_insufficient`：门 1–4、6–9 全过而核心门 5 delta < 0.15 → 如实记录实测 delta，回 S 轴归因；不回改判据。
- `failed`：任一机械门（1/2/3/8/9）或 arm-sanity 门（4/6/7）失败 → 构造或基建问题（含真实词表意外交集），不构成内容收益判据。

## 5. 诚实边界

- 语料为 OpenBMB UltraData（L3 合成+增强、教师模型轨迹）**真实发布语料**——非人类作者原始数据、非本脚本构造；与本 gate 判据相关的「真实」= 真实分布、真实共享工具词表、真实多域任务结构。
- corpus 本体 gitignored 不入 git；报告记录 sha256 + 行区间 + provenance 以复现；不触网、不下载、不重分发；许可以数据集 LICENSE 与官方 README 为准，本 gate 只读采样。
- unseen = 同族新内容（同词表内未见过的新轨迹/新任务），非全新词表；语义器官收益不以本 gate 度量（沿 P5.1e ill-posed 论证，语义面降为臂内 admission sanity，门 4）。
- quarantine 拒绝行为继承 E4/P5.1 认证，不重复设门；`growth_admitted=false`、`can_promote=false` 贯穿，不授执行权限。
- 绝对计数（examples/experience 换算率）仅设结构等式门，绝对值记录入报告。

## 6. 产物顺序

1. `scripts/training/eval_taiji_p5_1f_real_corpus_same_budget_gate.py`（静态四项先行）；
2. 执行落盘 `reports/taiji_p5_1f_real_corpus_same_budget_20260912.json` + 与本文件九门对账；
3. 路线图同步 + 独立提交。

## 7. 对账条款

报告以实测为准；若实测与本文件判据冲突，如实报告，禁止反向修改判据或报告凑数。判定依据以本文件冻结原文 + 带日期修订附录为唯一口径。
