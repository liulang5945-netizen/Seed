# Plans 目录分类整理方案

## 现状诊断（26 份 .md 文件，已全量核对）

当前 `plans/` 已有 `README.md` + `active/` + `archive/`（`audits/`、`architecture/`、`history/`、`reference/`、`implementation/`），但存在四类问题：

1. **分类错位**：`archive/architecture/TAIJI_TRANSFORM  ER_SHELL_AUDIT_20260822.md` 实为审计文档（审计 Transformer 壳），却放在 `architecture/` 下，应与 `AUDIT_2026_08.md`、`ARCHITECTURE_COMPROMISE_AUDIT.md` 同归 `audits/`。
2. **早期/废止设计混杂在 archive/architecture/**：`TAIJI0_PATCH_PROTOTYPE_RETIRED_20260821.md`、`HUB_NEURON_DESIGN.md`、`COMPREHENSIVE_NEURON_ARCHITECTURE_PLAN.md`、`BODY_LIFE_BRAIN_INTEGRATION_PLAN.md` 均为早期或已废止的架构设计稿，建议统一收到新建的 `archive/architecture_design/`，与「现行架构」明确区分。
3. **Legacy 底稿散落在 active/**：`NEUROPLEX_MECHANISM_RUNTIME_MAP_20260820.md`、`BOOTSTRAP_CRITERIA.md`、`DESIGN_PRINCIPLES.md`、`TAIJI_VS_HUMAN_BRAIN_COMPARISON.md` 描述的是 Legacy NeuroPlex（冻结基线）的事实基线/对比/原则，并非 Taiji 新底座权威设计，却与权威架构混在 `active/`。建议收到新建的 `archive/authored/`。
4. **命名不一致**：`archive/implementation/` 下的 `2026-07-27-side-channels-implementation.md`（日期前缀）、`H1-H8-mechanism-fixes.md`（连字符）与 `REMEDIATION_PLAN.md`（大写下划线）风格不统一。

---

## 目标结构

```
plans/
├── README.md                              # 入口，更新链接与归档说明
├── active/                               # 权威现行 + 当前执行计划
│   ├── SEED_ARCHITECTURE.md
│   ├── TAIJI_SUBSTRATE_ARCHITECTURE.md
│   ├── BIO_INSPIRED_ARCHITECTURE_PLAN.md
│   ├── ARCHITECTURE_DIRECTION_2026_08.md
│   ├── AGI_FIELD_MEMORY_PLAN.md
│   // 以上 5 份 = README「当前权威文档」
│   ├── SEED_PUBLIC_BETA_ROADMAP.md       # 公测路线图（现行执行）
│   └── M4_FRONTEND_REVIEW_CHECKLIST.md   # 前端评审清单（现行执行）
│
├── archive/
│   ├── audits/                           # 审计（含外部 + 内部审计报告）
│   │   ├── AUDIT_2026_08.md
│   │   ├── ARCHITECTURE_COMPROMISE_AUDIT.md
│   │   └── TAIJI_TRANSFORMER_SHELL_AUDIT_20260822.md   ← 从 architecture/ 移入
│   ├── architecture_design/              # 新建：早期/废止架构设计
│   │   ├── TAIJI0_PATCH_PROTOTYPE_RETIRED_20260821.md
│   │   ├── HUB_NEURON_DESIGN.md
│   │   ├── COMPREHENSIVE_NEURON_ARCHITECTURE_PLAN.md
│   │   └── BODY_LIFE_BRAIN_INTEGRATION_PLAN.md
│   ├── authored/                         # 新建：Legacy 底稿 / 事实基线 / 对比参考
│   │   ├── NEUROPLEX_MECHANISM_RUNTIME_MAP_20260820.md   ← 从 active/ 移入
│   │   ├── BOOTSTRAP_CRITERIA.md                        ←  从 active/ 移入
│   │   ├── DESIGN_PRINCIPLES.md                         ← 从 active/ 移入
│   │   └── TAIJI_VS_HUMAN_BRAIN_COMPARISON.md          ← 从 active/ 移入
│   ├── history/                          # 不变
│   │   ├── H  ISTORY_DIALOGUE_TRAINING.md
│   │   ├── HISTORY_PROJECT_EVENTS.md
│   │   └── HISTORY_MECHANISM_EXPERIMENTS.md
│   ├── reference/                        # 不变
│   │   └── TRAINING_REFERENCE.md
│   └── implementation/                   # 重命名统一风格
│       ├── REMEDIATION_PLAN.md
│       ├── H1_H8_MECHANISM_FIXES.md          ← 改自 H1-H8-mechanism-fixes.md
│       └── SIDE_CHANNELS_IMPLEMENTATION.md   ← 改自 2026-07-27-side-channels-implementation.md
```

---

## 命名规范（统一约定）

- **目录**：一律小写 + 下划线（`architecture_design`、`authored`、`audits`）。
- **文件**：去掉冗余日期前缀；统一 `UPPER_SNAKE`（如 `SIDE_CHANNELS_IMPLEMENTATION.md`），与现有 `SEED_ARCHITECTURE.md` 等主流风格保持一致；保留内容中已存在的语义日期后缀（`20260821`、`20260822`）以维持可追溯性。

## 具体移动/重命名清单

| 原路径 | 目标路径 | 动作 |
|---|---|---|
| `plans/archive/architecture/TAIJI_TRANSFORMER_SHELL_AUDIT_20260822.md` | `plans/archive/audits/TAIJI_TRANSFORMER_SHELL_AUDIT_20260822.md` | 移动 |
| `pl  ans/active/NEUROPLEX_MECHANISM_RUNTIME_MAP_20260820.md` | `plans/archive/authored/NEUROPLEX_MECHANISM_RUNTIME_MAP_20260820.md` | 移动 |
| `plans/active/BOOTSTRAP_CRITERIA.md` | `plans/archive/authored/BOOTSTRAP_CRITERIA.md` | 移动 |
| `plans/active/DESIGN_PRINCIPLES.md` | `plans/archive/authored/DESIGN_PRINCIPLES.md` | 移动 |
| `plans/active/TAIJI_VS_HUMAN_BRAIN_COMPARISON.md` | `plans/archive/authored/TAIJI_VS_HUMAN_BRAIN_COMPARISON.md` | 移动 |
| `plans/archive/implementation/2026-07-27-side-channels-implementation.md` | `plans/archive/implementation/SIDE_CHANNELS_IMPLEMENTATION.md` | 重命名 |
| `plans/archive/implementation/H1-H8-mechanism-fixes.md` | `plans/archive/implementation/H1_H8_MECHANISM_FIXES.md` | 重命名 |

（其余文件保持原位）

## 链接同步

移动/重命名后，需更新所有指向上述文件的**相对链接**，至少涉及：
- `plans/README.md` 中的「归档」说明与可能的引用。
- `archive/audits/` 内审计报告之间、以及 `archive/authored/` 文档之间可能存在互链（如 `TAIJI_TRANSFORMER_SHELL_AUDIT` 曾指向 `active/TAIJI_OPERATOR_DESIGN.md`，该文件不存在，需确认是否死链）。

执行时会先用 grep 全量扫描 `.md` 中的相对路径引用，列出所有受影响链接并一并修正，确保零断链。

## 风险与执行方式

- 仅「移动 / 重命名」，绝不删除任何内容；操作在 `E:\Seed\plans`（项目目录，非个人目录），可逆。
- 按每批 ≤10 文件执行，逐批校验。
- 完成后用 grep 复核链接，产出概览文档。

## 待确认

请在确认后我开始执行。如果你希望：保留 `DESIGN_PRINCIPLES.md` / `TAIJI_VS_HUMAN_BRAIN_COMPARISON.md` 在 `active/` 而非 `archive/authored/`，或调整任一目标目录，请指出。
